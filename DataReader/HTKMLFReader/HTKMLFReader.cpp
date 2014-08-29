//
// <copyright file="HTKMLFReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <objbase.h>
#include "basetypes.h"

#include "htkfeatio.h"                  // for reading HTK features
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "simplesenonehmm.h"            // for MMI scoring
#include "msra_mgram.h"                 // for unigram scores of ground-truth path in sequence training

#include "rollingwindowsource.h"        // minibatch sources
#include "utterancesourcemulti.h"
#include "utterancesource.h"
#include "utterancesourcemulti.h"
#include "readaheadsource.h"
#include "chunkevalsource.h"
#include "minibatchiterator.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "HTKMLFReader.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

int msra::numa::node_override = -1;     // for numahelpers.h

namespace Microsoft { namespace MSR { namespace CNTK {

	// Create a Data Reader
	//DATAREADER_API IDataReader* DataReaderFactory(void)

	template<class ElemType>
	void HTKMLFReader<ElemType>::Init(const ConfigParameters& readerConfig)
	{
		m_mbiter = NULL;
		m_frameSource = NULL;
		m_frameSourceMultiIO = NULL;
		m_readAheadSource = NULL;
		m_lattices = NULL;

		m_truncated = readerConfig("Truncated", "false");
		convertLabelsToTargets = false;

		m_numberOfuttsPerMinibatch = readerConfig("nbruttsineachrecurrentiter", "1");
		m_actualnumberOfuttsPerMinibatch = m_numberOfuttsPerMinibatch;
		m_sentenceEnd.assign(m_numberOfuttsPerMinibatch, true);
		m_processedFrame.assign(m_numberOfuttsPerMinibatch, 0);
		m_toProcess.assign(m_numberOfuttsPerMinibatch,0);
		m_switchFrame.assign(m_numberOfuttsPerMinibatch,0);
		m_noData = false;

		string command(readerConfig("action",L"")); //look up in the config for the master command to determine whether we're writing output (inputs only) or training/evaluating (inputs and outputs)

		m_legacyMode = false;
		string legacyMode(readerConfig("legacyMode", "false"));
		if (legacyMode=="true")
		{
			m_legacyMode = true;
		}

		if (m_legacyMode)
		{
			InitLegacy(readerConfig);
		}
		else
		{
			if (command == "write")
			{
				trainOrTest=false;
				PrepareForWriting(readerConfig);
			}
			else
			{
				trainOrTest=true;
				PrepareForTrainingOrTesting(readerConfig);
			}
		}
	}
	template<class ElemType>
	void HTKMLFReader<ElemType>::InitLegacy(const ConfigParameters& readerConfig)
	{

		if (readerConfig.Exists(DefaultFeaturesName()) && readerConfig.Exists(DefaultLabelsName()) ||    readerConfig.Exists(DefaultInputsName()) && readerConfig.Exists(DefaultOutputsName()))
		{
			trainOrTest=true;
			InitToTrainOrTest(readerConfig);
		}
		else if ((readerConfig.Exists(DefaultFeaturesName()) || readerConfig.Exists(DefaultInputsName())) && !(readerConfig.Exists(DefaultLabelsName()) || readerConfig.Exists(DefaultOutputsName())))
		{
			trainOrTest=false;
			InitToWriteOutput(readerConfig);
		}
		//else if (readerConfig.Exists(DefaultFeaturesName()) || readerConfig.Exists(DefaultInputsName())) // alone without outputs or writetofile
		//{
		//	throw std::runtime_error(msra::strfun::strprintf ("HTKMLFReader(Init): seems like are you attempting unsupervised training -- currently unsupported...\n"));
		//}
		else
		{
			throw std::runtime_error(msra::strfun::strprintf ("HTKMLFReader(Init): are you training or evaluating?\n"));
		}

	}

	// Load all input and output data. 
	// Note that the terms features imply be real-valued quanities and 
	// labels imply categorical quantities, irrespective of whether they 
	// are inputs or targets for the network
	template<class ElemType>
	void HTKMLFReader<ElemType>::PrepareForTrainingOrTesting(const ConfigParameters& readerConfig)
	{
		vector<wstring> scriptpaths;
		vector<wstring> mlfpaths;
		vector<vector<wstring>>mlfpathsmulti;
		size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
		vector<vector<wstring>> infilesmulti;
		vector<wstring> filelist;
		size_t numFiles;
		wstring unigrampath(L"");
		//wstring statelistpath(L"");
		size_t randomize = randomizeAuto;
		size_t iFeat, iLabel;
		iFeat = iLabel = 0;
		vector<wstring> statelistpaths;
		bool framemode = true;

		// for the multi-utterance process
		m_featuresBufferMultiUtt.assign(m_numberOfuttsPerMinibatch,NULL);
		m_featuresBufferAllocatedMultiUtt.assign(m_numberOfuttsPerMinibatch,0);
		m_labelsBufferMultiUtt.assign(m_numberOfuttsPerMinibatch,NULL);
		m_labelsBufferAllocatedMultiUtt.assign(m_numberOfuttsPerMinibatch,0);

		std::vector<std::wstring> featureNames;
		std::vector<std::wstring> labelNames;
		GetDataNamesFromConfig(readerConfig, featureNames, labelNames);
		if (featureNames.size() + labelNames.size() <= 1)
		{
			throw new std::runtime_error("network needs at least 1 input and 1 output specified!");
		}
			
		//load data for all real-valued inputs (features)
		foreach_index(i, featureNames)
		{
			ConfigParameters thisFeature = readerConfig(featureNames[i]);
			featDims.push_back(thisFeature("dim"));
			string type = thisFeature("type","Real");
			if (type=="Real"){
				nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
			}
			else{
				throw new std::runtime_error("feature type must be Real");
			}

			featureNameToIdMap[featureNames[i]]= iFeat;
			scriptpaths.push_back(thisFeature("scpFile"));
			featureNameToDimMap[featureNames[i]] = featDims[i];

			m_featuresBufferMultiIO.push_back(NULL);
			m_featuresBufferAllocatedMultiIO.push_back(0);

			iFeat++;			
		}

		foreach_index(i, labelNames)
		{
			ConfigParameters thisLabel = readerConfig(labelNames[i]);
			if (thisLabel.Exists("labelDim"))
				labelDims.push_back(thisLabel("labelDim"));
			else if (thisLabel.Exists("dim"))
				labelDims.push_back(thisLabel("dim"));
			else
				throw new std::runtime_error("labels must specify dim or labelDim");

			string type;
			if (thisLabel.Exists("labelType"))
				type = thisLabel("labelType"); // let's deprecate this eventually and just use "type"...
			else
				type = thisLabel("type","Category"); // outputs should default to category

			if (type=="Category")
				nameToTypeMap[labelNames[i]] = InputOutputTypes::category;
			else
				throw new std::runtime_error("label type must be Category");

			statelistpaths.push_back(thisLabel("labelMappingFile",L""));

			labelNameToIdMap[labelNames[i]]=iLabel;
			labelNameToDimMap[labelNames[i]]=labelDims[i];
			mlfpaths.clear();
			mlfpaths.push_back(thisLabel("mlfFile"));
			mlfpathsmulti.push_back(mlfpaths);

			m_labelsBufferMultiIO.push_back(NULL);
			m_labelsBufferAllocatedMultiIO.push_back(0);

			iLabel++;

			wstring labelToTargetMappingFile(thisLabel("labelToTargetMappingFile",L""));
			if (labelToTargetMappingFile != L"")
			{
				std::vector<std::vector<ElemType>> labelToTargetMap;
				convertLabelsToTargetsMultiIO.push_back(true);
				if (thisLabel.Exists("targetDim"))
				{
					labelNameToDimMap[labelNames[i]]=labelDims[i]=thisLabel("targetDim");
				}
				else
					throw new std::runtime_error("output must specify targetDim if labelToTargetMappingFile specified!");
				size_t targetDim = ReadLabelToTargetMappingFile (labelToTargetMappingFile,statelistpaths[i], labelToTargetMap);	
				if (targetDim!=labelDims[i])
					throw new std::runtime_error("mismatch between targetDim and dim found in labelToTargetMappingFile");
				labelToTargetMapMultiIO.push_back(labelToTargetMap);
			}
			else
			{
				convertLabelsToTargetsMultiIO.push_back(false);
				labelToTargetMapMultiIO.push_back(std::vector<std::vector<ElemType>>());
			}
		}

		if (iFeat!=scriptpaths.size() || iLabel!=mlfpathsmulti.size())
			throw std::runtime_error(msra::strfun::strprintf ("# of inputs files vs. # of inputs or # of output files vs # of outputs inconsistent\n"));

		if (readerConfig.Exists("randomize"))
		{
			const std::string& randomizeString = readerConfig("randomize");
			if (randomizeString == "None")
			{
				randomize = randomizeNone;
			}
			else if (randomizeString == "Auto")
			{
				randomize = randomizeAuto;
			}
			else
			{
				randomize = readerConfig("randomize");
			}
		}

		if (readerConfig.Exists("frameMode"))
		{
			const std::string& framemodeString = readerConfig("frameMode");
			if (framemodeString == "false")
			{
				framemode = false;
			}
		}

		int verbosity = readerConfig("verbosity","2");

		// determine if we partial minibatches are desired
		std::string minibatchMode(readerConfig("minibatchMode","Partial"));
		m_partialMinibatch = !_stricmp(minibatchMode.c_str(),"Partial");

		// get the read method, defaults to "blockRandomize" other option is "rollingWindow"
		std::string readMethod(readerConfig("readMethod","blockRandomize"));

		// see if they want to use readAhead
		m_readAhead = readerConfig("readAhead", "false");

		// read all input files (from multiple inputs)
		// TO DO: check for consistency (same number of files in each script file)
		numFiles=0;
		foreach_index(i,scriptpaths)
		{
			filelist.clear();
			std::wstring scriptpath = scriptpaths[i];
			fprintf(stderr, "reading script file %S ...", scriptpath.c_str());
			size_t n = 0;
			for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/; )
			{
				filelist.push_back (reader.wgetline());
				n++;
			}

			fprintf (stderr, " %lu entries\n", n);

			if (i==0)
				numFiles=n;
			else
				if (n!=numFiles)
					throw std::runtime_error (msra::strfun::strprintf ("number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

			infilesmulti.push_back(filelist);
		}

		if (readerConfig.Exists("unigram"))
			unigrampath = readerConfig("unigram");

		// load a unigram if needed (this is used for MMI training)
		msra::lm::CSymbolSet unigramsymbols;
		std::unique_ptr<msra::lm::CMGramLM> unigram;
		size_t silencewordid = SIZE_MAX;
		size_t startwordid = SIZE_MAX;
		size_t endwordid = SIZE_MAX;
		if (unigrampath != L"")
		{
			unigram.reset (new msra::lm::CMGramLM());
			unigram->read (unigrampath, unigramsymbols, false/*filterVocabulary--false will build the symbol map*/, 1/*maxM--unigram only*/);
			silencewordid = unigramsymbols["!silence"];     // give this an id (even if not in the LM vocabulary)
			startwordid = unigramsymbols["<s>"];
			endwordid = unigramsymbols["</s>"];
		}

		if (!unigram)
			fprintf (stderr, "trainlayer: OOV-exclusion code enabled, but no unigram specified to derive the word set from, so you won't get OOV exclusion\n");

		// currently assumes all mlfs will have same root name (key)
		set<wstring> restrictmlftokeys;     // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
		if (infilesmulti[0].size() <= 100)
		{
			foreach_index (i, infilesmulti[0])
			{
				msra::asr::htkfeatreader::parsedpath ppath (infilesmulti[0][i]);
				const wstring key = regex_replace ((wstring)ppath, wregex (L"\\.[^\\.\\\\/:]*$"), wstring());  // delete extension (or not if none)
				restrictmlftokeys.insert (key);
			}
		}
		// get labels

		//if (readerConfig.Exists("statelist"))
		//	statelistpath = readerConfig("statelist");

		double htktimetoframe = 100000.0;           // default is 10ms 
		//std::vector<msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>> labelsmulti;
		std::vector<std::map<std::wstring,std::vector<msra::asr::htkmlfentry>>> labelsmulti;
		std::vector<std::wstring> pagepath;
		foreach_index(i, mlfpathsmulti)
		{
			msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>  
				labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], unigram ? &unigramsymbols : NULL, (map<string,size_t>*) NULL, htktimetoframe);      // label MLF
			// get the temp file name for the page file
			labelsmulti.push_back(labels);
		}


		if (!_stricmp(readMethod.c_str(),"blockRandomize"))
		{
			// construct all the parameters we don't need, but need to be passed to the constructor...
			std::pair<std::vector<wstring>,std::vector<wstring>> latticetocs;
			std::unordered_map<std::string,size_t> modelsymmap;
			m_lattices = new msra::dbn::latticesource(latticetocs, modelsymmap);

			// now get the frame source. This has better randomization and doesn't create temp files
			m_frameSource = new msra::dbn::minibatchutterancesourcemulti(infilesmulti, labelsmulti, featDims, labelDims, randomize, *m_lattices, m_latticeMap, framemode);

		}
		else
		{
			foreach_index(i, infilesmulti)
			{
				wchar_t tempPath[MAX_PATH];
				GetTempPath(MAX_PATH, tempPath);
				wchar_t tempFile[MAX_PATH];
				GetTempFileName(tempPath, L"CNTK", 0, tempFile);
				//wstring pagefile = tempFile;
				pagepath.push_back(tempFile);
			}

			const bool mayhavenoframe=false;
			int addEnergy = 0;

			//m_frameSourceMultiIO = new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, featDims, labelDims, randomize, pagepath, mayhavenoframe, addEnergy);
			//m_frameSourceMultiIO->setverbosity(verbosity);
			m_frameSource = new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, featDims, labelDims, randomize, pagepath, mayhavenoframe, addEnergy);
			m_frameSource->setverbosity(verbosity);
		}

	}

	// Load all input and output data. 
	// Note that the terms features imply be real-valued quanities and 
	// labels imply categorical quantities, irrespective of whether they 
	// are inputs or targets for the network
	template<class ElemType>
	void HTKMLFReader<ElemType>::PrepareForWriting(const ConfigParameters& readerConfig)
	{
		vector<wstring> scriptpaths;
		vector<wstring> filelist;
		size_t numFiles;
		size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
		size_t evalchunksize = 2048;
		vector<size_t> realDims;
		size_t iFeat = 0;

		std::vector<std::wstring> featureNames;
		std::vector<std::wstring> labelNames;
		GetDataNamesFromConfig(readerConfig, featureNames, labelNames);

		foreach_index(i, featureNames)
		{
			ConfigParameters thisFeature = readerConfig(featureNames[i]);
			vdims.push_back(thisFeature("dim"));
			string type = thisFeature("type","Real");
			if (type=="Real"){
				nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
			}
			else{
				throw new std::runtime_error("feature type must be Real");
			}

			featureNameToIdMap[featureNames[i]]= iFeat;
			realDims.push_back(vdims[i]);
			scriptpaths.push_back(thisFeature("scpFile"));
			featureNameToDimMap[featureNames[i]] = vdims[i];

			m_featuresBufferMultiIO.push_back(NULL);
			m_featuresBufferAllocatedMultiIO.push_back(0);
			iFeat++;
		}

		if (labelNames.size()>0)
			throw new std::runtime_error("writer mode does not support labels as inputs, only features");

		numFiles=0;
		foreach_index(i,scriptpaths)
		{
			filelist.clear();
			std::wstring scriptpath = scriptpaths[i];
			fprintf(stderr, "reading script file %S ...", scriptpath.c_str());
			size_t n = 0;
			for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/; )
			{
				filelist.push_back (reader.wgetline());
				n++;
			}

			fprintf (stderr, " %d entries\n", n);

			if (i==0)
				numFiles=n;
			else
				if (n!=numFiles)
					throw std::runtime_error (msra::strfun::strprintf ("HTKMLFReader::InitEvalReader: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

			inputFilesMultiIO.push_back(filelist);
		}

		m_fileEvalSource = new msra::dbn::FileEvalSource(vdims,evalchunksize);

		double htktimetoframe = 100000.0;           // default is 10ms 
	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::InitToTrainOrTest(const ConfigParameters& readerConfig)
	{
		if (readerConfig.Exists(DefaultInputsName()) && readerConfig.Exists(DefaultOutputsName())) // check this first in case one of the inputs is named DefaultFeaturesName() and one of outputs is named DefaultLabelsName()
		{
			multiIO=true; // also support single i/o if # of inputs and outputs is 1 - eventually do away with singleIO path
			InitMultiIO(readerConfig);
		}
		else if (readerConfig.Exists(DefaultFeaturesName()) && readerConfig.Exists(DefaultLabelsName()))
		{
			multiIO=false;
			InitSingleIO(readerConfig);
		}
		else
		{
			throw std::runtime_error(msra::strfun::strprintf ("HTKMLFReader(InitTrain): need to specify features & labels or inputs & outputs\n"));
		}
	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::InitToWriteOutput(const ConfigParameters& readerConfig)
	{
		multiIO=readerConfig("multiIO","false");
		InitEvalReader(readerConfig);

		/*
		if (readerConfig("multiIO","false"))
		{
		multiIO = true;
		InitEvalMultiIO(readerConfig);
		}
		else
		{
		multiIO = false;
		InitEvalSingleIO(readerConfig);
		}
		*/
	}
	// Init - Reader Initialize for multiple data sets
	// config - [in] configuration parameters for the datareader
	// Sample format below:
	//reader=[
	//    # reader to use
	//    readerType=UCIFastReader
	//    miniBatchMode=Partial
	//    randomize=None
	//    features=[
	//    dim=429
	//    file=c:\speech\swb300h\data\archive.swb_mini.52_39.notestspk.dev.small.scplocal
	//    ]
	//    labels=[
	//    file=c:\speech\swb300h\data\swb_mini.1504.align.small.statemlf
	//    #labelMappingFile=<statelist path goes here>
	//    labelDim=1504
	//    ]
	//]
	template<class ElemType>
	void HTKMLFReader<ElemType>::InitSingleIO(const ConfigParameters& readerConfig)
	{
		ConfigParameters configFeatures = readerConfig(DefaultFeaturesName());
		ConfigParameters configLabels = readerConfig(DefaultLabelsName());
		size_t vdim = configFeatures("dim","429");
		size_t udim = configLabels("labelDim","1504");

		vector<wstring> infiles;
		vector<wstring> mlfpaths;               // path to read MLF file from (-I)  --TODO: should we allow wildcards?
		size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
		wstring scriptpath(configFeatures("file"));
		wstring mlfpath(configLabels("file"));
		wstring statelistpath(configLabels("labelMappingFile",L""));
		wstring unigrampath(L"");
		mlfpaths.push_back (mlfpath);
		m_udim = udim;
		m_partialMinibatch = false;
		size_t randomize = randomizeAuto;
		bool framemode = true;
		m_featuresBuffer=NULL;
		m_featuresBufferAllocated=0;

		m_featuresBufferMultiUtt.assign(m_numberOfuttsPerMinibatch,NULL);
		m_featuresBufferAllocatedMultiUtt.assign(m_numberOfuttsPerMinibatch,0);

		m_labelsBuffer=NULL;
		m_labelsBufferAllocated=0;

		m_labelsBufferMultiUtt.assign(m_numberOfuttsPerMinibatch,NULL);
		m_labelsBufferAllocatedMultiUtt.assign(m_numberOfuttsPerMinibatch,0);
		int verbosity = readerConfig("verbosity","2");

		if (readerConfig.Exists("randomize"))
		{
			const std::string& randomizeString = readerConfig("randomize");
			if (randomizeString == "None")
			{
				randomize = randomizeNone;
			}
			else if (randomizeString == "Auto")
			{
				randomize = randomizeAuto;
			}
			else
			{
				randomize = readerConfig("randomize");
			}
		}

		if (readerConfig.Exists("frameMode"))
		{
			const std::string& framemodeString = readerConfig("frameMode");
			if (framemodeString == "false")
			{
				framemode = false;
			}
		}
		// determine if we partial minibatches are desired
		std::string minibatchMode(readerConfig("minibatchMode","Partial"));
		m_partialMinibatch = !_stricmp(minibatchMode.c_str(),"Partial");

		fprintf(stderr, "dbn: reading script file %S ...", scriptpath.c_str());
		size_t n = 0;
		for (msra::files::textreader reader(scriptpath); reader && infiles.size() <= firstfilesonly/*optimization*/; )
		{
			infiles.push_back (reader.wgetline());
			n++;
		}
		fprintf (stderr, " %lu entries\n", n);

		// load a unigram if needed (this is used for MMI training)
		msra::lm::CSymbolSet unigramsymbols;
		std::unique_ptr<msra::lm::CMGramLM> unigram;
		size_t silencewordid = SIZE_MAX;
		size_t startwordid = SIZE_MAX;
		size_t endwordid = SIZE_MAX;
		if (unigrampath != L"")
		{
			unigram.reset (new msra::lm::CMGramLM());
			unigram->read (unigrampath, unigramsymbols, false/*filterVocabulary--false will build the symbol map*/, 1/*maxM--unigram only*/);
			silencewordid = unigramsymbols["!silence"];     // give this an id (even if not in the LM vocabulary)
			startwordid = unigramsymbols["<s>"];
			endwordid = unigramsymbols["</s>"];
		}
		if (!unigram)
			fprintf (stderr, "trainlayer: OOV-exclusion code enabled, but no unigram specified to derive the word set from, so you won't get OOV exclusion\n");

		set<wstring> restrictmlftokeys;     // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
		if (infiles.size() <= 100)
		{
			foreach_index (i, infiles)
			{
				msra::asr::htkfeatreader::parsedpath ppath (infiles[i]);
				const wstring key = regex_replace ((wstring)ppath, wregex (L"\\.[^\\.\\\\/:]*$"), wstring());  // delete extension (or not if none)
				restrictmlftokeys.insert (key);
			}
		}

		// get labels
		double htktimetoframe = 100000.0;           // default is 10ms 
		msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>  
			labels(mlfpaths, restrictmlftokeys, statelistpath, unigram ? &unigramsymbols : NULL, (map<string,size_t>*) NULL, htktimetoframe);      // label MLF

		// see if they want to use readAhead
		m_readAhead = readerConfig("readAhead", "false");

		// get the read method, defaults to "blockRandomize" other option is "rollingWindow"
		std::string readMethod(readerConfig("readMethod","blockRandomize"));
		m_partialMinibatch = !_stricmp(minibatchMode.c_str(),"Partial");

		if (!_stricmp(readMethod.c_str(),"blockRandomize"))
		{
			// construct all the parameters we don't need, but need to be passed to the constructor...
			std::pair<std::vector<wstring>,std::vector<wstring>> latticetocs;
			std::unordered_map<std::string,size_t> modelsymmap;
			m_lattices = new msra::dbn::latticesource(latticetocs, modelsymmap);

			// now get the frame source. This has better randomization and doesn't create temp files
			m_frameSource = new msra::dbn::minibatchutterancesource(infiles, labels, vdim, udim, randomize, *m_lattices, m_latticeMap, framemode);
		}
		else
		{
			// get the temp file name for the page file
			wchar_t tempPath[MAX_PATH];
			GetTempPath(MAX_PATH, tempPath);
			wchar_t tempFile[MAX_PATH];
			GetTempFileName(tempPath, L"CNTK", 0, tempFile);
			wstring pagepath = tempFile;
			const bool mayhavenoframe=false;
			int addEnergy = readerConfig("addEnergy","0");
			m_frameSource = new msra::dbn::minibatchframesource(infiles, labels, vdim, udim, randomize, pagepath, mayhavenoframe, addEnergy);
			m_frameSource->setverbosity(verbosity);
		}


		wstring labelToTargetMappingFile(configLabels("labelToTargetMappingFile",L""));
		if (labelToTargetMappingFile != L"")
		{
			udim = m_udim = configLabels("targetDim");
			size_t targetDim = ReadLabelToTargetMappingFile (labelToTargetMappingFile,statelistpath,labelToTargetMap);	
			if (targetDim!=udim)
				throw new std::runtime_error("mismatch between targetDim and dim found in labelToTargetMappingFile");
			convertLabelsToTargets=true;		
		}
	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::InitMultiIO(const ConfigParameters& readerConfig)
	{
		vector<wstring> scriptpaths;
		vector<wstring> mlfpaths;
		vector<vector<wstring>>mlfpathsmulti;
		size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
		vector<vector<wstring>> infilesmulti;
		vector<wstring> filelist;
		size_t numFiles;
		wstring unigrampath(L"");
		//wstring statelistpath(L"");
		size_t randomize = randomizeAuto;
		size_t iFeat, iLabel;
		size_t thisDim;
		//vector<size_t> realDims;
		//vector<size_t> categoryDims;
		iFeat = iLabel = 0;
		vector<wstring> statelistpaths;

		ConfigArray inputNames = readerConfig(DefaultInputsName());
		foreach_index(i, inputNames)
		{
			ConfigParameters thisInput = readerConfig(inputNames[i]);
			if (thisInput.Exists("labelDim"))
				thisDim = thisInput("labelDim");
			else if (thisInput.Exists("dim"))
				thisDim = thisInput("dim");
			else
				throw new std::runtime_error("HTKMLFReader::InitMultiIO input must specify dim or labelDim");

			string type;
			if (thisInput.Exists("labelType"))
				type = thisInput("labelType"); // let's deprecate this eventually and just use "type"...
			else
				type = thisInput("type","Real"); // inputs should default to real
			if (type=="Real")
				nameToTypeMap[inputNames[i]] = InputOutputTypes::inputReal;
			else if (type=="Category")
				nameToTypeMap[inputNames[i]] = InputOutputTypes::inputCategory;
			else
				throw new std::runtime_error("HTKMLFReader::InitMultiIO input type must be Real or Category");

			wstring labelToTargetMappingFile(thisInput("labelToTargetMappingFile",L""));

			switch (nameToTypeMap[inputNames[i]])
			{
			case InputOutputTypes::inputReal: 			
				featureNameToIdMap[inputNames[i]]= iFeat;
				featDims.push_back(thisDim);
				scriptpaths.push_back(thisInput("file"));
				featureNameToDimMap[inputNames[i]] = thisDim;

				m_featuresBufferMultiIO.push_back(NULL);
				m_featuresBufferAllocatedMultiIO.push_back(0);

				iFeat++;
				break;			
			case InputOutputTypes::inputCategory:
				labelNameToIdMap[inputNames[i]]=iLabel;
				mlfpaths.clear();
				mlfpaths.push_back(thisInput("file"));
				mlfpathsmulti.push_back(mlfpaths);
				statelistpaths.push_back(thisInput("labelMappingFile",L""));
				m_labelsBufferMultiIO.push_back(NULL);
				m_labelsBufferAllocatedMultiIO.push_back(0);

				if (labelToTargetMappingFile != L"")
				{
					std::vector<std::vector<ElemType>> labelToTargetMap;
					convertLabelsToTargetsMultiIO.push_back(true);
					if (thisInput.Exists("targetDim"))
					{
						thisDim = thisInput("targetDim");
						labelNameToDimMap[inputNames[i]]=thisDim;
						labelDims.push_back(thisDim);
					}
					else
						throw new std::runtime_error("HTKMLFReader::InitMultiIO input must specify targetDim if labelToTargetMappingFile specified!");
					size_t targetDim = ReadLabelToTargetMappingFile (labelToTargetMappingFile,statelistpaths[i], labelToTargetMap);	
					if (targetDim!=thisDim)
						throw new std::runtime_error("mismatch between targetDim and dim found in labelToTargetMappingFile");
					labelToTargetMapMultiIO.push_back(labelToTargetMap);
				}
				else
				{
					labelDims.push_back(thisDim);
					labelNameToDimMap[inputNames[i]] = thisDim;
					convertLabelsToTargetsMultiIO.push_back(false);
					labelToTargetMapMultiIO.push_back(std::vector<std::vector<ElemType>>());
				}

				iLabel++;

				break;
			default:
				throw new std::runtime_error("HTKMLFReader::InitMultiIO unknown input type");
				break;
			}

		}

		ConfigArray outputNames = readerConfig(DefaultOutputsName());
		foreach_index(i, outputNames)
		{
			ConfigParameters thisOutput = readerConfig(outputNames[i]);
			if (thisOutput.Exists("labelDim"))
				thisDim = thisOutput("labelDim");
			else if (thisOutput.Exists("dim"))
				thisDim = thisOutput("dim");
			else
				throw new std::runtime_error("HTKMLFReader::InitMultiIO output must specify dim or labelDim");

			string type;
			if (thisOutput.Exists("labelType"))
				type = thisOutput("labelType"); // let's deprecate this eventually and just use "type"...
			else
				type = thisOutput("type","Category"); // outputs should default to category

			if (type=="Real")
				nameToTypeMap[outputNames[i]] = InputOutputTypes::outputReal;
			else if (type=="Category")
				nameToTypeMap[outputNames[i]] = InputOutputTypes::outputCategory;
			else
				throw new std::runtime_error("HTKMLFReader::InitMultiIO output type must be Real or Category");

			wstring labelToTargetMappingFile(thisOutput("labelToTargetMappingFile",L""));
				
			switch (nameToTypeMap[outputNames[i]])
			{
			case InputOutputTypes::outputReal: 			
				featureNameToIdMap[outputNames[i]]= iFeat;
				featDims.push_back(thisDim);
				featureNameToDimMap[outputNames[i]] = thisDim;

				scriptpaths.push_back(thisOutput("file"));

				m_featuresBufferMultiIO.push_back(NULL);
				m_featuresBufferAllocatedMultiIO.push_back(0);

				iFeat++;
				break;			
			case InputOutputTypes::outputCategory:
				labelNameToIdMap[outputNames[i]]=iLabel;
				labelDims.push_back(thisDim);
				labelNameToDimMap[outputNames[i]]=thisDim;
				mlfpaths.clear();
				mlfpaths.push_back(thisOutput("file"));
				mlfpathsmulti.push_back(mlfpaths);
				statelistpaths.push_back(thisOutput("labelMappingFile",L""));
				m_labelsBufferMultiIO.push_back(NULL);
				m_labelsBufferAllocatedMultiIO.push_back(0);

				if (labelToTargetMappingFile != L"")
				{
					std::vector<std::vector<ElemType>> labelToTargetMap;
					convertLabelsToTargetsMultiIO.push_back(true);
					if (thisOutput.Exists("targetDim"))
					{
						thisDim = thisOutput("targetDim");
						labelNameToDimMap[outputNames[i]]=thisDim;
						labelDims.push_back(thisDim);
					}
					else
						throw new std::runtime_error("HTKMLFReader::InitMultiIO output must specify targetDim if labelToTargetMappingFile specified!");
					size_t targetDim = ReadLabelToTargetMappingFile (labelToTargetMappingFile,statelistpaths[i], labelToTargetMap);	
					if (targetDim!=thisDim)
						throw new std::runtime_error("mismatch between targetDim and dim found in labelToTargetMappingFile");
					labelToTargetMapMultiIO.push_back(labelToTargetMap);
				}
				else
				{
					labelDims.push_back(thisDim);
					labelNameToDimMap[outputNames[i]] = thisDim;
					convertLabelsToTargetsMultiIO.push_back(false);
					labelToTargetMapMultiIO.push_back(std::vector<std::vector<ElemType>>());
				}

				iLabel++;
				break;
			default:
				throw new std::runtime_error("HTKMLFReader::InitMultiIO output type must be Real or Category");
				break;
			}

		}

		if (iFeat!=scriptpaths.size() || iLabel!=mlfpathsmulti.size())
			throw std::runtime_error(msra::strfun::strprintf ("HTKReaderMultiIO: # of inputs files vs. # of inputs or # of output files vs # of outputs inconsistent\n"));

		//if (vdims.size()!=scriptpaths.size() || udims.size()!=mlfpathsmulti.size())
		//    throw std::runtime_error(msra::strfun::strprintf ("HTKReaderMultiIO: # of inputs files vs. # of inputs or # of output files vs # of outputs inconsistent\n"));

		if (readerConfig.Exists("randomize"))
		{
			const std::string& randomizeString = readerConfig("randomize");
			if (randomizeString == "None")
			{
				randomize = randomizeNone;
			}
			else if (randomizeString == "Auto")
			{
				randomize = randomizeAuto;
			}
			else
			{
				randomize = readerConfig("randomize");
			}
		}

		int verbosity = readerConfig("verbosity","2");

		// determine if we partial minibatches are desired
		std::string minibatchMode(readerConfig("minibatchMode","Partial"));
		m_partialMinibatch = !_stricmp(minibatchMode.c_str(),"Partial");

		// get the read method, defaults to "blockRandomize" other option is "rollingWindow"
		std::string readMethod(readerConfig("readMethod","blockRandomize"));

		// see if they want to use readAhead
		m_readAhead = readerConfig("readAhead", "false");

		// read all input files (from multiple inputs)
		// TO DO: check for consistency (same number of files in each script file)
		numFiles=0;
		foreach_index(i,scriptpaths)
		{
			filelist.clear();
			std::wstring scriptpath = scriptpaths[i];
			fprintf(stderr, "reading script file %S ...", scriptpath.c_str());
			size_t n = 0;
			for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/; )
			{
				filelist.push_back (reader.wgetline());
				n++;
			}

			fprintf (stderr, " %lu entries\n", n);

			if (i==0)
				numFiles=n;
			else
				if (n!=numFiles)
					throw std::runtime_error (msra::strfun::strprintf ("HTKReaderMultiIO: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

			infilesmulti.push_back(filelist);
		}

		if (readerConfig.Exists("unigram"))
			unigrampath = readerConfig("unigram");


		// load a unigram if needed (this is used for MMI training)
		msra::lm::CSymbolSet unigramsymbols;
		std::unique_ptr<msra::lm::CMGramLM> unigram;
		size_t silencewordid = SIZE_MAX;
		size_t startwordid = SIZE_MAX;
		size_t endwordid = SIZE_MAX;
		if (unigrampath != L"")
		{
			unigram.reset (new msra::lm::CMGramLM());
			unigram->read (unigrampath, unigramsymbols, false/*filterVocabulary--false will build the symbol map*/, 1/*maxM--unigram only*/);
			silencewordid = unigramsymbols["!silence"];     // give this an id (even if not in the LM vocabulary)
			startwordid = unigramsymbols["<s>"];
			endwordid = unigramsymbols["</s>"];
		}
		if (!unigram)
			fprintf (stderr, "trainlayer: OOV-exclusion code enabled, but no unigram specified to derive the word set from, so you won't get OOV exclusion\n");

		// currently assumes all mlfs will have same root name (key)
		set<wstring> restrictmlftokeys;     // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
		if (infilesmulti[0].size() <= 100)
		{
			foreach_index (i, infilesmulti[0])
			{
				msra::asr::htkfeatreader::parsedpath ppath (infilesmulti[0][i]);
				const wstring key = regex_replace ((wstring)ppath, wregex (L"\\.[^\\.\\\\/:]*$"), wstring());  // delete extension (or not if none)
				restrictmlftokeys.insert (key);
			}
		}
		// get labels

		//if (readerConfig.Exists("statelist"))
		//	statelistpath = readerConfig("statelist");

		double htktimetoframe = 100000.0;           // default is 10ms 
		//std::vector<msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>> labelsmulti;
		std::vector<std::map<std::wstring,std::vector<msra::asr::htkmlfentry>>> labelsmulti;
		std::vector<std::wstring> pagepath;
		foreach_index(i, mlfpathsmulti)
		{
			msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>  
				labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], unigram ? &unigramsymbols : NULL, (map<string,size_t>*) NULL, htktimetoframe);      // label MLF
			// get the temp file name for the page file
			labelsmulti.push_back(labels);
		}


		if (!_stricmp(readMethod.c_str(),"blockRandomize"))
		{
			throw new std::runtime_error("readMethod=blockRandomize is not yet supported for multi IO\n");
		}
		else
		{
			foreach_index(i, infilesmulti)
			{
				wchar_t tempPath[MAX_PATH];
				GetTempPath(MAX_PATH, tempPath);
				wchar_t tempFile[MAX_PATH];
				GetTempFileName(tempPath, L"CNTK", 0, tempFile);
				//wstring pagefile = tempFile;
				pagepath.push_back(tempFile);
			}

			const bool mayhavenoframe=false;
			int addEnergy = 0;

			m_frameSourceMultiIO = new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, featDims, labelDims, randomize, pagepath, mayhavenoframe, addEnergy);
			m_frameSourceMultiIO->setverbosity(verbosity);
		}
	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::InitEvalSingleIO(const ConfigParameters& readerConfig)
	{
		size_t evalchunksize = 2048;
		vector<wstring> filelist;

		// parse features config
		ConfigParameters configFeatures = readerConfig(DefaultFeaturesName());
		vdims.push_back(configFeatures("dim","429"));
		wstring scriptpath(configFeatures("file"));	
		ConfigArray subsetInfo = configFeatures("subset","0:1");
		size_t subset = subsetInfo[0];
		size_t subsets = subsetInfo[1];

		m_featuresBuffer=NULL;
		m_featuresBufferAllocated=0;

		// parse output file config
		ConfigParameters configOutput = readerConfig("write");
		udims.push_back(configOutput("dim","10"));
		outputPath = configOutput("path",".");
		outputExtension = configOutput("ext","mfc");
		outputNodeName = configOutput("nodeName");
		outputScp = configOutput("scpFile","");
		const std::string& outputType = configOutput("type","");
		if (outputType=="scaledLogLikelihood")
			scaleByPrior= true;
		else
			scaleByPrior = false;

		vector<wstring> infiles;
		size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing

		m_outputBufferMultiIO.push_back(NULL);
		m_outputBufferAllocatedMultiIO.push_back(0);

		int verbosity = readerConfig("verbosity","2");


		fprintf(stderr, "dbn: reading script file %S ...", scriptpath.c_str());
		size_t n = 0;
		for (msra::files::textreader reader(scriptpath); reader && inputFiles.size() <= firstfilesonly/*optimization*/; )
		{
			inputFiles.push_back (reader.wgetline());
			n++;
		}
		fprintf (stderr, " %d entries\n", n);	

		if (subsets>1)
		{
			vector<wstring> subsetFiles;
			subsetFiles.reserve(inputFiles.size()/subsets + 1);
			for (size_t i=subset;i<inputFiles.size();i+=subsets)
				subsetFiles.push_back(std::move(inputFiles[i]));
			::swap(inputFiles,subsetFiles);
		}


		m_chunkEvalSource = new msra::dbn::chunkevalsource(vdims[0],udims[0],evalchunksize);
	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::InitEvalMultiIO(const ConfigParameters& readerConfig)
	{
		vector<wstring> scriptpaths;
		//vector<wstring> outputPaths;
		//vector<wstring> outputextensions;
		//vector<vector<wstring>> infilesmulti;
		vector<wstring> filelist;
		size_t numFiles;
		size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
		size_t evalchunksize = 2048;
		ConfigArray inputNames = readerConfig(DefaultInputsName());
		foreach_index(i, inputNames)
		{
			ConfigParameters thisInput = readerConfig(inputNames[i]);
			vdims.push_back(thisInput("dim"));
			scriptpaths.push_back(thisInput("file"));

			featureNameToIdMap[inputNames[i]]= i;
			nameToTypeMap[inputNames[i]] = InputOutputTypes::inputReal;
			m_featuresBufferMultiIO.push_back(NULL);
			m_featuresBufferAllocatedMultiIO.push_back(0);
		}

		ConfigArray outputNames = readerConfig("write");
		foreach_index(i, outputNames)
		{
			ConfigParameters thisOutput = readerConfig(outputNames[i]);
			udims.push_back(thisOutput("dim","10"));
			outputPaths.push_back(thisOutput("path"));
			if (thisOutput.Exists("ext"))
			{
				outputExtensions.push_back(thisOutput("ext"));
			}
			else
			{
				outputExtensions.push_back(L"mfc");
			}
			outputNodeNames.push_back(thisOutput("nodeName"));
			outputNameToIdMap[outputNodeNames[i]]=i;
			nameToTypeMap[outputNames[i]] = InputOutputTypes::networkOutputs;
			m_outputBufferMultiIO.push_back(NULL);
			m_outputBufferAllocatedMultiIO.push_back(0);
			outputScps.push_back(thisOutput("scpFile",""));
			const std::string& outputType = thisOutput("type","");
			if (outputType=="scaledLogLikelihood")
				scaleByPriorMultiIO.push_back(true);
			else
				scaleByPriorMultiIO.push_back(false);
		}


		numFiles=0;
		foreach_index(i,scriptpaths)
		{
			filelist.clear();
			std::wstring scriptpath = scriptpaths[i];
			fprintf(stderr, "dbn: reading script file %S ...", scriptpath.c_str());
			size_t n = 0;
			for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/; )
			{
				filelist.push_back (reader.wgetline());
				n++;
			}

			fprintf (stderr, " %d entries\n", n);

			if (i==0)
				numFiles=n;
			else
				if (n!=numFiles)
					throw std::runtime_error (msra::strfun::strprintf ("HTKReaderMultiIO: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

			inputFilesMultiIO.push_back(filelist);
		}

		m_chunkEvalSourceMultiIO = new msra::dbn::chunkevalsourcemulti(vdims,udims,evalchunksize);

		double htktimetoframe = 100000.0;           // default is 10ms 

	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::InitEvalReader(const ConfigParameters& readerConfig)
	{
		vector<wstring> scriptpaths;
		//vector<wstring> outputPaths;
		//vector<wstring> outputextensions;
		//vector<vector<wstring>> infilesmulti;
		vector<wstring> filelist;
		size_t numFiles;
		size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
		size_t evalchunksize = 2048;
		ConfigArray input = readerConfig(DefaultInputsName());
		if (input.size()<1)
			throw new std::runtime_error("reader needs at least one input specified!");

		for (size_t i=0;i<input.size(); i++) // inputNames should map to node names
		{
			ConfigParameters thisInput = input[i];//readerConfig(inputNames[i]);
			if (thisInput.size()>1)
				throw new std::runtime_error("unexpected config parse in InitEvalReader!");

			auto iter = thisInput.begin();
			std::wstring thisName = msra::strfun::utf16(iter->first);

			ConfigParameters thisParams = thisInput(thisName);
			vdims.push_back(thisParams("dim"));
			scriptpaths.push_back(thisParams("file"));
			wstring type = thisParams("type","Real");

			featureNameToIdMap[thisName]= i;
			featureNameToDimMap[thisName]=vdims[i];
			if (type == L"Real")
			{
				nameToTypeMap[thisName] = InputOutputTypes::inputReal;
			}
			else
			{
				throw std::runtime_error ("HTKMLFReader::InitEvalReader: input type for HTKMLFReader expected to be Real");
			}
			m_featuresBufferMultiIO.push_back(NULL);
			m_featuresBufferAllocatedMultiIO.push_back(0);
		}
		numFiles=0;
		foreach_index(i,scriptpaths)
		{
			filelist.clear();
			std::wstring scriptpath = scriptpaths[i];
			fprintf(stderr, "dbn: reading script file %S ...", scriptpath.c_str());
			size_t n = 0;
			for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/; )
			{
				filelist.push_back (reader.wgetline());
				n++;
			}

			fprintf (stderr, " %d entries\n", n);

			if (i==0)
				numFiles=n;
			else
				if (n!=numFiles)
					throw std::runtime_error (msra::strfun::strprintf ("HTKMLFReader::InitEvalReader: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

			inputFilesMultiIO.push_back(filelist);
		}

		m_fileEvalSource = new msra::dbn::FileEvalSource(vdims,evalchunksize);

		double htktimetoframe = 100000.0;           // default is 10ms 
	}

	// destructor - virtual so it gets called properly 
	template<class ElemType>
	HTKMLFReader<ElemType>::~HTKMLFReader()
	{
		delete m_mbiter;
		delete m_readAheadSource;
		if (multiIO)
		{
			delete m_frameSourceMultiIO;
			if (!m_featuresBufferMultiIO.empty())
			{
				if ( m_featuresBufferMultiIO[0] != NULL)
				{
					foreach_index(i, m_featuresBufferMultiIO)
					{
						delete[] m_featuresBufferMultiIO[i];
						m_featuresBufferMultiIO[i] = NULL;
					}
				}
			}
			if (!m_labelsBufferMultiIO.empty())
			{
				if (m_labelsBufferMultiIO[0] != NULL)
				{
					foreach_index(i, m_labelsBufferMultiIO)
					{
						delete[] m_labelsBufferMultiIO[i];
						m_labelsBufferMultiIO[i] = NULL;
					}
				}
			}
		}
		else
		{

			delete m_frameSource;
			delete m_lattices;
			if (m_featuresBuffer!=NULL)
			{
				delete[] m_featuresBuffer;
				m_featuresBuffer=NULL;
			}
			if (m_labelsBuffer!=NULL)
			{
				delete[] m_labelsBuffer;
				m_labelsBuffer=NULL;

			}
			if (/*m_numberOfuttsPerMinibatch > 1 && */m_truncated)
			{
				for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i ++)
				{
					if (m_featuresBufferMultiUtt[i] != NULL)
					{
						delete[] m_featuresBufferMultiUtt[i];
						m_featuresBufferMultiUtt[i] = NULL;
					}
					if (m_labelsBufferMultiUtt[i] != NULL)
					{
						delete[] m_labelsBufferMultiUtt[i];
						m_labelsBufferMultiUtt[i] = NULL;
					}

				}
			}
		}
	}

	//StartMinibatchLoop - Startup a minibatch loop 
	// mbSize - [in] size of the minibatch (number of frames, etc.)
	// epoch - [in] epoch number for this loop
	// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
	template<class ElemType>
	void HTKMLFReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
	{
		m_mbSize = mbSize;

		if (m_legacyMode)
		{
			if (trainOrTest)
			{
				StartMinibatchLoopTrain(mbSize,epoch,requestedEpochSamples);
			}
			else
			{
				StartMinibatchLoopEval(mbSize,epoch,requestedEpochSamples);	
			}
		}
		else
		{
			if (trainOrTest)
			{
				StartMinibatchLoopToTrainOrTest(mbSize,epoch,requestedEpochSamples);
			}
			else
			{
				StartMinibatchLoopToWrite(mbSize,epoch,requestedEpochSamples);	
			}
		}
		checkDictionaryKeys=true;
	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
	{
		size_t datapasses=1;
		//size_t totalFrames = m_frameSource->totalframes();
		size_t totalFrames;
		totalFrames = m_frameSource->totalframes();

		size_t extraFrames = totalFrames%mbSize;
		size_t minibatches = totalFrames/mbSize;

		// if we are allowing partial minibatches, do nothing, and let it go through
		if (!m_partialMinibatch)
		{
			// we don't want any partial frames, so round total frames to be an even multiple of our mbSize
			if (totalFrames > mbSize)
				totalFrames -= extraFrames;

			if (requestedEpochSamples == requestDataSize)
			{
				requestedEpochSamples = totalFrames;
			}
			else if (minibatches > 0)   // if we have any full minibatches
			{
				// since we skip the extraFrames, we need to add them to the total to get the actual number of frames requested
				size_t sweeps = (requestedEpochSamples-1)/totalFrames; // want the number of sweeps we will skip the extra, so subtract 1 and divide
				requestedEpochSamples += extraFrames*sweeps;
			}
		}
		else if (requestedEpochSamples == requestDataSize)
		{
			requestedEpochSamples = totalFrames;
		}

		// delete the old one first (in case called more than once)
		delete m_mbiter;
		msra::dbn::minibatchsource* source = m_frameSource;
		if (m_readAhead)
		{
			if (m_readAheadSource == NULL)
			{
				m_readAheadSource = new msra::dbn::minibatchreadaheadsource (*source, requestedEpochSamples);
			}
			else if (m_readAheadSource->epochsize() != requestedEpochSamples)
			{
				delete m_readAheadSource;
				m_readAheadSource = new msra::dbn::minibatchreadaheadsource (*source, requestedEpochSamples);
			}
			source = m_readAheadSource;
		}
		m_mbiter = new msra::dbn::minibatchiterator(*source, epoch, requestedEpochSamples, mbSize, datapasses);
		if (!m_featuresBufferMultiIO.empty())
		{
			if (m_featuresBufferMultiIO[0]!=NULL) // check first feature, if it isn't NULL, safe to assume all are not NULL? 
			{
				foreach_index(i, m_featuresBufferMultiIO)
				{
					delete[] m_featuresBufferMultiIO[i];
					m_featuresBufferMultiIO[i]=NULL;
					m_featuresBufferAllocatedMultiIO[i]=0;
				}
			}
		}
		if (!m_labelsBufferMultiIO.empty())
		{
			if (m_labelsBufferMultiIO[0]!=NULL)
			{
				foreach_index(i, m_labelsBufferMultiIO)
				{
					delete[] m_labelsBufferMultiIO[i];
					m_labelsBufferMultiIO[i]=NULL;
					m_labelsBufferAllocatedMultiIO[i]=0;
				}
			}
		}
		if (m_numberOfuttsPerMinibatch && m_truncated == true)
		{
			m_noData = false;
			m_featuresStartIndexMultiUtt.assign(m_featuresBufferMultiIO.size()*m_numberOfuttsPerMinibatch,0);
			m_labelsStartIndexMultiUtt.assign(m_labelsBufferMultiIO.size()*m_numberOfuttsPerMinibatch,0);
			for (size_t u = 0; u < m_numberOfuttsPerMinibatch; u ++)
			{
				if (m_featuresBufferMultiUtt[u] != NULL)
				{
					delete[] m_featuresBufferMultiUtt[u];
					m_featuresBufferMultiUtt[u] = NULL;
					m_featuresBufferAllocatedMultiUtt[u] = 0;
	}
				if (m_labelsBufferMultiUtt[u] != NULL)
				{
					delete[] m_labelsBufferMultiUtt[u];
					m_labelsBufferMultiUtt[u] = NULL;
					m_labelsBufferAllocatedMultiUtt[u] = 0;
				}
				ReNewBufferForMultiIO(u);
			}	
		}
	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::StartMinibatchLoopToWrite(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
	{
		m_fileEvalSource->Reset();
		m_fileEvalSource->SetMinibatchSize(mbSize);
		//m_chunkEvalSourceMultiIO->reset();
		inputFileIndex=0;

		if (m_featuresBufferMultiIO[0]!=NULL) // check first feature, if it isn't NULL, safe to assume all are not NULL? 
		{
			foreach_index(i, m_featuresBufferMultiIO)
			{
				delete[] m_featuresBufferMultiIO[i];
				m_featuresBufferMultiIO[i]=NULL;
				m_featuresBufferAllocatedMultiIO[i]=0;
			}
		}

	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::StartMinibatchLoopTrain(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
	{
		size_t datapasses=1;
		//size_t totalFrames = m_frameSource->totalframes();
		size_t totalFrames;
		if (multiIO)
			totalFrames=m_frameSourceMultiIO->totalframes();
		else
			totalFrames=m_frameSource->totalframes();

		size_t extraFrames = totalFrames%mbSize;
		size_t minibatches = totalFrames/mbSize;

		// if we are allowing partial minibatches, do nothing, and let it go through
		if (!m_partialMinibatch)
		{
			// we don't want any partial frames, so round total frames to be an even multiple of our mbSize
			if (totalFrames > mbSize)
				totalFrames -= extraFrames;

			if (requestedEpochSamples == requestDataSize)
			{
				requestedEpochSamples = totalFrames;
			}
			else if (minibatches > 0)   // if we have any full minibatches
			{
				// since we skip the extraFrames, we need to add them to the total to get the actual number of frames requested
				size_t sweeps = (requestedEpochSamples-1)/totalFrames; // want the number of sweeps we will skip the extra, so subtract 1 and divide
				requestedEpochSamples += extraFrames*sweeps;
			}
		}
		else if (requestedEpochSamples == requestDataSize)
		{
			requestedEpochSamples = totalFrames;
		}

		// delete the old one first (in case called more than once)
		delete m_mbiter;
		msra::dbn::minibatchsource* source = multiIO?m_frameSourceMultiIO:m_frameSource;
		if (m_readAhead)
		{
			if (m_readAheadSource == NULL)
			{
				m_readAheadSource = new msra::dbn::minibatchreadaheadsource (*source, requestedEpochSamples);
			}
			else if (m_readAheadSource->epochsize() != requestedEpochSamples)
			{
				delete m_readAheadSource;
				m_readAheadSource = new msra::dbn::minibatchreadaheadsource (*source, requestedEpochSamples);
			}
			source = m_readAheadSource;
		}
		m_mbiter = new msra::dbn::minibatchiterator(*source, epoch, requestedEpochSamples, mbSize, datapasses);
		if (multiIO){
			//m_mbiter = new msra::dbn::minibatchiterator(*m_frameSourceMultiIO, epoch, requestedEpochSamples, mbSize, datapasses);
			if (!m_featuresBufferMultiIO.empty())
			{
				if (m_featuresBufferMultiIO[0]!=NULL) // check first feature, if it isn't NULL, safe to assume all are not NULL? 
				{
					foreach_index(i, m_featuresBufferMultiIO)
					{
						delete[] m_featuresBufferMultiIO[i];
						m_featuresBufferMultiIO[i]=NULL;
						m_featuresBufferAllocatedMultiIO[i]=0;
					}
				}
			}
			if (!m_labelsBufferMultiIO.empty())
			{
				if (m_labelsBufferMultiIO[0]!=NULL)
				{
					foreach_index(i, m_labelsBufferMultiIO)
					{
						delete[] m_labelsBufferMultiIO[i];
						m_labelsBufferMultiIO[i]=NULL;
						m_labelsBufferAllocatedMultiIO[i]=0;
					}
				}
			}
		}
		else{
			//m_mbiter = new msra::dbn::minibatchiterator(*m_frameSource, epoch, requestedEpochSamples, mbSize, datapasses);

			if (m_featuresBuffer!=NULL)
			{
				delete[] m_featuresBuffer;
				m_featuresBuffer=NULL;
				m_featuresBufferAllocated=0;
			}
			if (m_labelsBuffer!=NULL)
			{
				delete[] m_labelsBuffer;
				m_labelsBuffer=NULL;
				m_labelsBufferAllocated=0;
			}
			if (m_numberOfuttsPerMinibatch && m_truncated == true)
			{
				m_noData = false;
				for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i ++)
				{
					if (m_featuresBufferMultiUtt[i] != NULL)
					{
						delete[] m_featuresBufferMultiUtt[i];
						m_featuresBufferMultiUtt[i] = NULL;
						m_featuresBufferAllocatedMultiUtt[i] = 0;
					}
					if (m_labelsBufferMultiUtt[i] != NULL)
					{
						delete[] m_labelsBufferMultiUtt[i];
						m_labelsBufferMultiUtt[i] = NULL;
						m_labelsBufferAllocatedMultiUtt[i] = 0;
					}
					ReNewBuffer(i);
				}	
			}
		}
	}

	template<class ElemType>
	void HTKMLFReader<ElemType>::StartMinibatchLoopEval(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
	{
		if (multiIO){
			m_fileEvalSource->Reset();
			m_fileEvalSource->SetMinibatchSize(mbSize);
			//m_chunkEvalSourceMultiIO->reset();
			inputFileIndex=0;

			if (m_featuresBufferMultiIO[0]!=NULL) // check first feature, if it isn't NULL, safe to assume all are not NULL? 
			{
				foreach_index(i, m_featuresBufferMultiIO)
				{
					delete[] m_featuresBufferMultiIO[i];
					m_featuresBufferMultiIO[i]=NULL;
					m_featuresBufferAllocatedMultiIO[i]=0;
				}
			}
		}
		else
		{
			m_chunkEvalSource->reset();
			inputFileIndex=0;
			if (m_featuresBuffer!=NULL)
			{
				delete[] m_featuresBuffer;
				m_featuresBuffer=NULL;
				m_featuresBufferAllocated=0;
			}

		}

		checkDictionaryKeys = true;

	}
#if 0
	// GetMinibatch - Get the next minibatch (features and labels)
	// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
	//             [out] each matrix resized if necessary containing data. 
	// returns - true if there are more minibatches, false if no more minibatchs remain
	template<class ElemType>
	bool HTKMLFReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
	{
		if (trainOrTest)
		{
			if (multiIO)
				return GetMinibatchMultiIO(matrices);
			else
				return GetMinibatchSingleIO(matrices);
		}
		else
		{
			if (multiIO)
				return GetMinibatchEval(matrices);
			else 
				return GetMinibatchEvalSingleIO(matrices);
		}
	}
#endif // 0
	// GetMinibatch - Get the next minibatch (features and labels)
	// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
	//             [out] each matrix resized if necessary containing data. 
	// returns - true if there are more minibatches, false if no more minibatchs remain
	template<class ElemType>
	bool HTKMLFReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
	{
		if (m_legacyMode)
		{
			if (trainOrTest)
			{
				if (multiIO)
					return GetMinibatchMultiIO(matrices);
				else
					return GetMinibatchSingleIO(matrices);
			}
			else
			{
				if (multiIO)
					return GetMinibatchEval(matrices);
				else 
					return GetMinibatchEvalSingleIO(matrices);
			}
		}
		else
		{
			if (trainOrTest)
			{
				return GetMinibatchToTrainOrTest(matrices);
			}
			else
			{
				return GetMinibatchToWrite(matrices);
			}
		}
	}

	template<class ElemType>
	bool HTKMLFReader<ElemType>::GetMinibatchToTrainOrTest(std::map<std::wstring, Matrix<ElemType>*>& matrices)
	{
		size_t id;
		size_t dim;
		bool skip = false;

		// on first minibatch, make sure we can supply data for requested nodes
		std::map<std::wstring,size_t>::iterator iter;
		if 	(checkDictionaryKeys)
		{
			for (auto iter=matrices.begin();iter!=matrices.end();iter++)
			{
				if (nameToTypeMap.find(iter->first)==nameToTypeMap.end())
					throw std::runtime_error(msra::strfun::strprintf("minibatch requested for input node %ws not found in reader - cannot generate input\n",iter->first.c_str()));

			}
			checkDictionaryKeys=false;
		}

		do 
		{
			if (m_truncated == false)
			{
			if (!(*m_mbiter))
				return false;

			const size_t mbstartframe = m_mbiter->currentmbstartframe();

			// now, access all features and and labels by iterating over map of "matrices"
			std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
			for (iter = matrices.begin();iter!=matrices.end(); iter++)
			{
				// dereference matrix that corresponds to key (input/output name) and 
				// populate based on whether its a feature or a label
				Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

				if (nameToTypeMap[iter->first] == InputOutputTypes::real)
				{
					//                switch (nameToTypeMap[iter->first])
					//                {
					//                case InputOutputTypes::inputReal:
					//                case InputOutputTypes::outputReal:
					//                    {
					// fprintf(stderr, "name:  %S  type: input features\n", (iter->first).c_str());

					id = featureNameToIdMap[iter->first];
					dim = featureNameToDimMap[iter->first];
					const msra::dbn::matrixstripe feat = m_mbiter->frames(id);
					const size_t actualmbsize = feat.cols();   // it may still return less if at end of sweep TODO: this check probably only needs to happen once
					assert (actualmbsize == m_mbiter->currentmbframes());
					skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSourceMultiIO->totalframes() > actualmbsize);

					// check to see if we got the number of frames we requested
					if (!skip)
					{
						// copy the features over to our array type
						assert(feat.rows()==dim); // check feature dimension matches what's expected

						if (m_featuresBufferMultiIO[id]==NULL)
						{
							m_featuresBufferMultiIO[id] = new ElemType[feat.rows()*feat.cols()];
							m_featuresBufferAllocatedMultiIO[id] = feat.rows()*feat.cols();
						}
						else if (m_featuresBufferAllocatedMultiIO[id]<feat.rows()*feat.cols()) //buffer size changed. can be partial minibatch
						{
							delete[] m_featuresBufferMultiIO[id];
							m_featuresBufferMultiIO[id] = new ElemType[feat.rows()*feat.cols()];
							m_featuresBufferAllocatedMultiIO[id] = feat.rows()*feat.cols();
						}
						// shouldn't need this since we fill up the entire buffer below
						//memset(m_featuresBufferMultiIO[id],0,sizeof(ElemType)*feat.rows()*feat.cols());

						if (sizeof(ElemType) == sizeof(float))
						{
							for (int j=0; j < feat.cols(); j++) // column major, so iterate columns
							{
								// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
								memcpy_s(&m_featuresBufferMultiIO[id][j*feat.rows()],sizeof(ElemType)*feat.rows(),&feat(0,j),sizeof(ElemType)*feat.rows());
							}
						}
						else
						{
							for (int j=0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
							{
								for (int i = 0; i < feat.rows(); i++)
								{
									m_featuresBufferMultiIO[id][j*feat.rows()+i] = feat(i,j);
								}
							}
						}
						data.SetValue(feat.rows(), feat.cols(), m_featuresBufferMultiIO[id],matrixFlagNormal);
					}
				}
				else if (nameToTypeMap[iter->first] == InputOutputTypes::category)
				{
					//break;
					//case InputOutputTypes::inputCategory:
					//case InputOutputTypes::outputCategory:
					//    {
					// fprintf(stderr, "name:  %S  = output features\n", (iter->first).c_str());
					id = labelNameToIdMap[iter->first];
					dim = labelNameToDimMap[iter->first];
					const vector<size_t> & uids = m_mbiter->labels(id);

					//size_t m_udim = udims[id];

					// need skip logic here too in case labels are first in map not features
					const size_t actualmbsize = uids.size();   // it may still return less if at end of sweep TODO: this check probably only needs to happen once
					assert (actualmbsize == m_mbiter->currentmbframes());
					skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSourceMultiIO->totalframes() > actualmbsize);

					if (!skip)
					{
						// copy the labels over to array type
						//data.Resize(udims[id], uids.size());
						//data.SetValue((ElemType)0);

						// loop through the columns and set one value to 1
						// in the future we want to use a sparse matrix here
						//for (int i = 0; i < uids.size(); i++)
						//{
						//    assert(uids[i] <udims[id]);
						//    data(uids[i], i) = (ElemType)1;
						//}

						if (m_labelsBufferMultiIO[id]==NULL)
						{
							m_labelsBufferMultiIO[id] = new ElemType[dim*uids.size()];
							m_labelsBufferAllocatedMultiIO[id] = dim*uids.size();
						}
						else if (m_labelsBufferAllocatedMultiIO[id]<dim*uids.size())
						{
							delete[] m_labelsBufferMultiIO[id];
							m_labelsBufferMultiIO[id] = new ElemType[dim*uids.size()];
							m_labelsBufferAllocatedMultiIO[id] = dim*uids.size();
						}
						memset(m_labelsBufferMultiIO[id],0,sizeof(ElemType)*dim*uids.size());                


						if (convertLabelsToTargetsMultiIO[id])
						{
							size_t labelDim = labelToTargetMapMultiIO[id].size();
							for (int i = 0; i < uids.size(); i++)
							{
								assert(uids[i] < labelDim);
								size_t labelId = uids[i];
								for (int j = 0; j < dim; j++)
								{
									m_labelsBufferMultiIO[id][i*dim + j] = labelToTargetMapMultiIO[id][labelId][j];
								}
							}
						}
						else
						{
							// loop through the columns and set one value to 1
							// in the future we want to use a sparse matrix here
							for (int i = 0; i < uids.size(); i++)
							{
								assert(uids[i] < dim);
								//labels(uids[i], i) = (ElemType)1;
								m_labelsBufferMultiIO[id][i*dim+uids[i]]=(ElemType)1;
							}
						}


						data.SetValue(dim,uids.size(),m_labelsBufferMultiIO[id],matrixFlagNormal);
					}
				}
				else{
					//default:
					throw runtime_error(msra::strfun::strprintf("GetMinibatchMultiIO:: unknown InputOutputType for %S\n",(iter->first).c_str()));
				}

			}
			// advance to the next minibatch
			(*m_mbiter)++;
			}
			else
			{
				if (m_noData)
				{
					bool endEpoch = true;
					for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
					{
						if (m_processedFrame[i] != m_toProcess[i])
						{
							endEpoch = false;
						}
					}
					if(endEpoch)
					{
						return false;
					}
				}
				size_t numOfFea = m_featuresBufferMultiIO.size();
				size_t numOfLabel = m_labelsBufferMultiIO.size();
				vector<size_t> actualmbsize;
				actualmbsize.assign(m_numberOfuttsPerMinibatch,0);
				for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
				{
					size_t startFr = m_processedFrame[i];
					size_t endFr = 0;
					if ((m_processedFrame[i] + m_mbSize) < m_toProcess[i])
					{
						if(m_processedFrame[i] > 0)
						{
							m_sentenceEnd[i] = false;
							m_switchFrame[i] = m_mbSize+1;
						}
						else
						{
							m_switchFrame[i] = 0;
							m_sentenceEnd[i] = true;
						}
						actualmbsize[i] = m_mbSize;
						endFr = startFr + actualmbsize[i];
						std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
						for (iter = matrices.begin();iter!=matrices.end(); iter++)
						{
							// dereference matrix that corresponds to key (input/output name) and 
							// populate based on whether its a feature or a label
							Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

							if (nameToTypeMap[iter->first] == InputOutputTypes::real)
							{
								id = featureNameToIdMap[iter->first];
								dim = featureNameToDimMap[iter->first];

								if (m_featuresBufferMultiIO[id]==NULL)
								{
									m_featuresBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
									m_featuresBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
								}
								else if (m_featuresBufferAllocatedMultiIO[id]<dim*m_mbSize*m_numberOfuttsPerMinibatch) //buffer size changed. can be partial minibatch
								{
									delete[] m_featuresBufferMultiIO[id];
									m_featuresBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
									m_featuresBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
								}
								// shouldn't need this since we fill up the entire buffer below
								//memset(m_featuresBufferMultiIO[id],0,sizeof(ElemType)*feat.rows()*feat.cols());

								if (sizeof(ElemType) == sizeof(float))
								{
									for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
									{
										// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
										memcpy_s(&m_featuresBufferMultiIO[id][(k*m_numberOfuttsPerMinibatch+i)*dim],sizeof(ElemType)*dim,&m_featuresBufferMultiUtt[i][j*dim+m_featuresStartIndexMultiUtt[id+i*numOfFea]],sizeof(ElemType)*dim);
									}
								}
								else
								{
									for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
									{
										for (int d = 0; d < dim; d++)
										{
											m_featuresBufferMultiIO[id][(k*m_numberOfuttsPerMinibatch+i)*dim+d] = m_featuresBufferMultiUtt[i][j*dim+d+m_featuresStartIndexMultiUtt[id+i*numOfFea]];
										}
									}
								}
							}
							else if (nameToTypeMap[iter->first] == InputOutputTypes::category)
							{
								id = labelNameToIdMap[iter->first];
								dim = labelNameToDimMap[iter->first];
								if (m_labelsBufferMultiIO[id]==NULL)
								{
									m_labelsBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
									m_labelsBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
								}
								else if (m_labelsBufferAllocatedMultiIO[id]<dim*m_mbSize*m_numberOfuttsPerMinibatch)
								{
									delete[] m_labelsBufferMultiIO[id];
									m_labelsBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
									m_labelsBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
								}

								for (size_t j = startFr,k=0; j < endFr; j++,k++)
								{
									for (int d = 0; d < dim; d++)
									{
										m_labelsBufferMultiIO[id][(k*m_numberOfuttsPerMinibatch+i)*dim + d] = m_labelsBufferMultiUtt[i][j*dim+d+m_labelsStartIndexMultiUtt[id+i*numOfLabel]];
									}
								}
							}
						}
						m_processedFrame[i] += m_mbSize;
					}
					else
					{
						actualmbsize[i] = m_toProcess[i] - m_processedFrame[i];
						endFr = startFr + actualmbsize[i];

						std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
						for (iter = matrices.begin();iter!=matrices.end(); iter++)
						{
							// dereference matrix that corresponds to key (input/output name) and 
							// populate based on whether its a feature or a label
							Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

							if (nameToTypeMap[iter->first] == InputOutputTypes::real)
							{
								id = featureNameToIdMap[iter->first];
								dim = featureNameToDimMap[iter->first];

								if (m_featuresBufferMultiIO[id]==NULL)
								{
									m_featuresBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
									m_featuresBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
								}
								else if (m_featuresBufferAllocatedMultiIO[id]<dim*m_mbSize*m_numberOfuttsPerMinibatch) //buffer size changed. can be partial minibatch
								{
									delete[] m_featuresBufferMultiIO[id];
									m_featuresBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
									m_featuresBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
								}
								if (sizeof(ElemType) == sizeof(float))
								{
									for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
									{
										// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
										memcpy_s(&m_featuresBufferMultiIO[id][(k*m_numberOfuttsPerMinibatch+i)*dim],sizeof(ElemType)*dim,&m_featuresBufferMultiUtt[i][j*dim+m_featuresStartIndexMultiUtt[id+i*numOfFea]],sizeof(ElemType)*dim);
									}
								}
								else
								{
									for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
									{
										for (int d = 0; d < dim; d++)
										{
											m_featuresBufferMultiIO[id][(k*m_numberOfuttsPerMinibatch+i)*dim+d] = m_featuresBufferMultiUtt[i][j*dim+d+m_featuresStartIndexMultiUtt[id+i*numOfFea]];
										}
									}
								}
							}
							else if (nameToTypeMap[iter->first] == InputOutputTypes::category)
							{
								id = labelNameToIdMap[iter->first];
								dim = labelNameToDimMap[iter->first];
								if (m_labelsBufferMultiIO[id]==NULL)
								{
									m_labelsBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
									m_labelsBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
								}
								else if (m_labelsBufferAllocatedMultiIO[id]<dim*m_mbSize*m_numberOfuttsPerMinibatch)
								{
									delete[] m_labelsBufferMultiIO[id];
									m_labelsBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
									m_labelsBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
								}
								for (size_t j = startFr,k=0; j < endFr; j++,k++)
								{
									for (int d = 0; d < dim; d++)
									{
										m_labelsBufferMultiIO[id][(k*m_numberOfuttsPerMinibatch+i)*dim + d] = m_labelsBufferMultiUtt[i][j*dim+d+m_labelsStartIndexMultiUtt[id+i*numOfLabel]];
									}
								}
							}
						}
						m_processedFrame[i] += (endFr-startFr);
						m_switchFrame[i] = actualmbsize[i];
						startFr = m_switchFrame[i];
						endFr = m_mbSize;
						bool reNewSucc = ReNewBufferForMultiIO(i);
						for (iter = matrices.begin();iter!=matrices.end(); iter++)
						{
							// dereference matrix that corresponds to key (input/output name) and 
							// populate based on whether its a feature or a label
							Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

							if (nameToTypeMap[iter->first] == InputOutputTypes::real)
							{
								id = featureNameToIdMap[iter->first];
								dim = featureNameToDimMap[iter->first];
								if (sizeof(ElemType) == sizeof(float))
								{
									for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
									{
										// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
										memcpy_s(&m_featuresBufferMultiIO[id][(j*m_numberOfuttsPerMinibatch+i)*dim],sizeof(ElemType)*dim,&m_featuresBufferMultiUtt[i][k*dim+m_featuresStartIndexMultiUtt[id+i*numOfFea]],sizeof(ElemType)*dim);
									}
								}
								else
								{
									for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
									{
										for (int d = 0; d < dim; d++)
										{
											m_featuresBufferMultiIO[id][(j*m_numberOfuttsPerMinibatch+i)*dim+d] = m_featuresBufferMultiUtt[i][k*dim+d+m_featuresStartIndexMultiUtt[id+i*numOfFea]];
										}
									}
								}
							}
							else if (nameToTypeMap[iter->first] == InputOutputTypes::category)
							{
								id = labelNameToIdMap[iter->first];
								dim = labelNameToDimMap[iter->first];
								for (size_t j = startFr,k=0; j < endFr; j++,k++)
								{
									for (int d = 0; d < dim; d++)
									{
										m_labelsBufferMultiIO[id][(j*m_numberOfuttsPerMinibatch+i)*dim + d] = m_labelsBufferMultiUtt[i][k*dim+d+m_labelsStartIndexMultiUtt[id+i*numOfLabel]];
									}
								}
							}
						}

						if (reNewSucc) m_processedFrame[i] += (endFr-startFr);

					}
				}
				std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
				for (iter = matrices.begin();iter!=matrices.end(); iter++)
				{
					// dereference matrix that corresponds to key (input/output name) and 
					// populate based on whether its a feature or a label
					Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels
					if (nameToTypeMap[iter->first] == InputOutputTypes::real)
					{
						id = featureNameToIdMap[iter->first];
						dim = featureNameToDimMap[iter->first];
						data.SetValue(dim, m_mbSize*m_numberOfuttsPerMinibatch, m_featuresBufferMultiIO[id],matrixFlagNormal);
					}
					else if (nameToTypeMap[iter->first] == InputOutputTypes::category)
					{
						id = labelNameToIdMap[iter->first];
						dim = labelNameToDimMap[iter->first];
						data.SetValue(dim, m_mbSize*m_numberOfuttsPerMinibatch, m_labelsBufferMultiIO[id],matrixFlagNormal);
					}
				}
				skip=false;
			}
		}   // keep going if we didn't get the right size minibatch
		while(skip);

		return true;
	}

	template<class ElemType>
	bool HTKMLFReader<ElemType>::GetMinibatchToWrite(std::map<std::wstring, Matrix<ElemType>*>& matrices)
	{
		std::map<std::wstring,size_t>::iterator iter;
		if 	(checkDictionaryKeys)
		{
			for (auto iter=featureNameToIdMap.begin();iter!=featureNameToIdMap.end();iter++)
			{
				if (matrices.find(iter->first)==matrices.end())
				{
					fprintf(stderr,"GetMinibatchToWrite: feature node %ws specified in reader not found in the network\n",iter->first.c_str());
					throw std::runtime_error("GetMinibatchToWrite: feature node specified in reader not found in the network.");
				}
			}
			/*
			for (auto iter=matrices.begin();iter!=matrices.end();iter++)
			{
				if (featureNameToIdMap.find(iter->first)==featureNameToIdMap.end())
					throw std::runtime_error(msra::strfun::strprintf("minibatch requested for input node %ws not found in reader - cannot generate input\n",iter->first.c_str()));
			}
			*/
			checkDictionaryKeys=false;
		}

		if (inputFileIndex<inputFilesMultiIO[0].size())
		{
			m_fileEvalSource->Reset();

			// load next file (or set of files)
			foreach_index(i, inputFilesMultiIO)
			{
				msra::asr::htkfeatreader reader;

				const auto path = reader.parse(inputFilesMultiIO[i][inputFileIndex]);
				// read file
				msra::dbn::matrix feat;
				string featkind;
				unsigned int sampperiod;
				msra::util::attempt (5, [&]()
				{
					reader.read (path, featkind, sampperiod, feat);   // whole file read as columns of feature vectors
				});
				fprintf (stderr, "evaluate: reading %d frames of %S\n", feat.cols(), ((wstring)path).c_str());
				m_fileEvalSource->AddFile(feat, featkind, sampperiod, i);
			}
			inputFileIndex++;

			// turn frames into minibatch (augment neighbors, etc)
			m_fileEvalSource->CreateEvalMinibatch();

			// populate input matrices

			std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
			for (iter = matrices.begin();iter!=matrices.end(); iter++)
			{
				// dereference matrix that corresponds to key (input/output name) and 
				// populate based on whether its a feature or a label

				if (nameToTypeMap.find(iter->first)!=nameToTypeMap.end() && nameToTypeMap[iter->first] == InputOutputTypes::real)
				{
					Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels
					//if (nameToTypeMap[iter->first] == InputOutputTypes::inputReal)
					//{
					size_t id = featureNameToIdMap[iter->first];
					size_t dim = featureNameToDimMap[iter->first];

					const msra::dbn::matrix feat = m_fileEvalSource->ChunkOfFrames(id);
					const size_t actualmbsize = feat.cols();   // it may still return less if at end of sweep TODO: this check probably only needs to happen once

					// copy the features over to our array type
					assert(feat.rows()==dim); // check feature dimension matches what's expected

					if (m_featuresBufferMultiIO[id]==NULL)
					{
						m_featuresBufferMultiIO[id] = new ElemType[feat.rows()*feat.cols()];
						m_featuresBufferAllocatedMultiIO[id] = feat.rows()*feat.cols();
					}
					else if (m_featuresBufferAllocatedMultiIO[id]<feat.rows()*feat.cols()) //buffer size changed. can be partial minibatch
					{
						delete[] m_featuresBufferMultiIO[id];
						m_featuresBufferMultiIO[id] = new ElemType[feat.rows()*feat.cols()];
						m_featuresBufferAllocatedMultiIO[id] = feat.rows()*feat.cols();
					}
					// shouldn't need this since we fill up the entire buffer below
					//memset(m_featuresBufferMultiIO[id],0,sizeof(ElemType)*feat.rows()*feat.cols());

					if (sizeof(ElemType) == sizeof(float))
					{
						for (int j=0; j < feat.cols(); j++) // column major, so iterate columns
						{
							// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
							memcpy_s(&m_featuresBufferMultiIO[id][j*feat.rows()],sizeof(ElemType)*feat.rows(),&feat(0,j),sizeof(ElemType)*feat.rows());
						}
					}
					else
					{
						for (int j=0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
						{
							for (int i = 0; i < feat.rows(); i++)
							{
								m_featuresBufferMultiIO[id][j*feat.rows()+i] = feat(i,j);
							}
						}
					}
					data.SetValue(feat.rows(), feat.cols(), m_featuresBufferMultiIO[id],matrixFlagNormal);
				}
			}
			return true;
		}
		else
		{
			return false;
		}
	}

	// GetMinibatch - Get the next minibatch (features and labels)
	// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
	//             [out] each matrix resized if necessary containing data. 
	// returns - true if there are more minibatches, false if no more minibatchs remain
	template<class ElemType>
	bool HTKMLFReader<ElemType>::GetMinibatchSingleIO(std::map<std::wstring, Matrix<ElemType>*>& matrices)
	{
		// current implmentation expects 'features' and 'labels' as names. In the future multiple feature sets may be added
		Matrix<ElemType>& features = *matrices[DefaultFeaturesName()];
		Matrix<ElemType>& labels = *matrices[DefaultLabelsName()];
		// we want to repeat while 
		bool skip = false;
		do 
		{
			if (m_truncated == false)
			{
				if (!(*m_mbiter))
					return false;
				const msra::dbn::matrixstripe feat= m_mbiter->frames();
				const vector<size_t> & uids = m_mbiter->labels();
				const size_t mbstartframe = m_mbiter->currentmbstartframe();
				const size_t actualmbsize = feat.cols();   // it may still return less if at end of sweep
				assert (actualmbsize == m_mbiter->currentmbframes());

				skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSource->totalframes() > actualmbsize);
				// check to see if we got the number of frames we requested
				if (!skip)
				{
					if (m_featuresBuffer==NULL)
					{
						m_featuresBuffer = new ElemType[feat.rows()*feat.cols()];
						m_featuresBufferAllocated = feat.rows()*feat.cols();
					}
					else if (m_featuresBufferAllocated<feat.rows()*feat.cols()) //buffer size changed. can be partial minibatch
					{
						delete[] m_featuresBuffer;
						m_featuresBuffer = new ElemType[feat.rows()*feat.cols()];
						m_featuresBufferAllocated = feat.rows()*feat.cols();
					}
					// shouldn't need this since we fill up the entire buffer below
					//memset(m_featuresBuffer,0,sizeof(ElemType)*feat.rows()*feat.cols());

					if (sizeof(ElemType) == sizeof(float))
					{
						for (int j=0; j < feat.cols(); j++) // column major, so iterate columns
						{
							// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
							memcpy_s(&m_featuresBuffer[j*feat.rows()],sizeof(ElemType)*feat.rows(),&feat(0,j),sizeof(ElemType)*feat.rows());
						}
					}
					else
					{
						for (int j=0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
						{
							for (int i = 0; i < feat.rows(); i++)
							{
								m_featuresBuffer[j*feat.rows()+i] = feat(i,j);
							}
						}
					}
					features.SetValue(feat.rows(), feat.cols(), m_featuresBuffer,matrixFlagNormal);

					// copy the labels over to array type
					//labels.Resize(m_udim, uids.size());
					//labels.SetValue((ElemType)0);
					if (m_labelsBuffer==NULL)
					{
						m_labelsBuffer = new ElemType[m_udim*uids.size()];
						m_labelsBufferAllocated = m_udim*uids.size();
					}
					else if (m_labelsBufferAllocated<m_udim*uids.size())
					{
						delete[] m_labelsBuffer;
						m_labelsBuffer = new ElemType[m_udim*uids.size()];
						m_labelsBufferAllocated = m_udim*uids.size();
					}
					memset(m_labelsBuffer,0,sizeof(ElemType)*m_udim*uids.size());                

					if (convertLabelsToTargets)
					{
						size_t labelDim = labelToTargetMap.size();
						for (int i = 0; i < uids.size(); i++)
						{
							assert(uids[i] < labelDim);
							size_t labelId = uids[i];
							for (int j = 0; j < m_udim; j++)
							{
								m_labelsBuffer[i*m_udim + j] = labelToTargetMap[labelId][j];
							}
						}
					}
					else
					{
						// loop through the columns and set one value to 1
						// in the future we want to use a sparse matrix here
						for (int i = 0; i < uids.size(); i++)
						{
							assert(uids[i] < m_udim);
							//labels(uids[i], i) = (ElemType)1;
							m_labelsBuffer[i*m_udim+uids[i]]=(ElemType)1;
						}
					}
					labels.SetValue(m_udim,uids.size(),m_labelsBuffer,matrixFlagNormal);
				}

				// advance to the next minibatch
				(*m_mbiter)++;
			}
			else if (m_numberOfuttsPerMinibatch == 1)
			{
				if (!(*m_mbiter))
					return false;

				msra::dbn::matrixstripe featOri= m_mbiter->frames();
				vector<size_t> & uidsOri = m_mbiter->labels();
				const size_t mbstartframeOri = m_mbiter->currentmbstartframe();
				const size_t actualmbsizeOri = featOri.cols();   // it may still return less if at end of sweep
				assert (actualmbsizeOri == m_mbiter->currentmbframes());

				const msra::dbn::matrixstripe feat= m_mbiter->frames();
				const vector<size_t> & uids = m_mbiter->labels();
				const size_t mbstartframe = m_mbiter->currentmbstartframe();
				size_t actualmbsize = 0;
				size_t startFr = m_processedFrame[0];
				m_sentenceEnd[0] = false;
				if ((m_processedFrame[0] + m_mbSize) < actualmbsizeOri)
				{
					actualmbsize = m_mbSize;
				}else
				{
					actualmbsize = actualmbsizeOri - m_processedFrame[0];

				}
				size_t endFr = startFr + actualmbsize;

				skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSource->totalframes() > actualmbsize);
				// check to see if we got the number of frames we requested
				if (!skip)
				{
					if (m_featuresBuffer==NULL)
					{
						m_featuresBuffer = new ElemType[feat.rows()*actualmbsize];
						m_featuresBufferAllocated = feat.rows()*actualmbsize;
					}
					else if (m_featuresBufferAllocated<feat.rows()*actualmbsize) //buffer size changed. can be partial minibatch
					{
						delete[] m_featuresBuffer;
						m_featuresBuffer = new ElemType[feat.rows()*actualmbsize];
						m_featuresBufferAllocated = feat.rows()*actualmbsize;
					}
					// shouldn't need this since we fill up the entire buffer below
					//memset(m_featuresBuffer,0,sizeof(ElemType)*feat.rows()*feat.cols());

					if (sizeof(ElemType) == sizeof(float))
					{
						for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
						{
							// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
							memcpy_s(&m_featuresBuffer[k*feat.rows()],sizeof(ElemType)*feat.rows(),&feat(0,j),sizeof(ElemType)*feat.rows());
						}
					}
					else
					{
						for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
						{
							for (int i = 0; i < feat.rows(); i++)
							{
								m_featuresBuffer[k*feat.rows()+i] = feat(i,j);
							}
						}
					}
					features.SetValue(feat.rows(), actualmbsize, m_featuresBuffer,matrixFlagNormal);

					// copy the labels over to array type
					//labels.Resize(m_udim, uids.size());
					//labels.SetValue((ElemType)0);
					if (m_labelsBuffer==NULL)
					{
						m_labelsBuffer = new ElemType[m_udim*actualmbsize];
						m_labelsBufferAllocated = m_udim*actualmbsize;
					}
					else if (m_labelsBufferAllocated<m_udim*actualmbsize)
					{
						delete[] m_labelsBuffer;
						m_labelsBuffer = new ElemType[m_udim*actualmbsize];
						m_labelsBufferAllocated = m_udim*actualmbsize;
					}
					memset(m_labelsBuffer,0,sizeof(ElemType)*m_udim*actualmbsize);                

					if (convertLabelsToTargets)
					{
						size_t labelDim = labelToTargetMap.size();
						for (size_t i = startFr,k=0; i < endFr; i++,k++)
						{
							assert(uids[i] < labelDim);
							size_t labelId = uids[i];
							for (int j = 0; j < m_udim; j++)
							{
								m_labelsBuffer[k*m_udim + j] = labelToTargetMap[labelId][j];
							}
						}
					}
					else
					{
						// loop through the columns and set one value to 1
						// in the future we want to use a sparse matrix here
						for (size_t i = startFr, k=0; i < endFr; i++,k++)
						{
							assert(uids[i] < m_udim);
							//labels(uids[i], i) = (ElemType)1;
							m_labelsBuffer[k*m_udim+uids[i]]=(ElemType)1;
						}
					}
					labels.SetValue(m_udim,actualmbsize,m_labelsBuffer,matrixFlagNormal);
				}
				m_processedFrame[0] += actualmbsize;
				// advance to the next minibatch
				if (m_processedFrame[0] == actualmbsizeOri)
				{
					(*m_mbiter)++;
					m_processedFrame[0] = 0;
					m_sentenceEnd[0] = true;
				}
			} 
			else
			{

				if (m_noData)
				{
					bool endEpoch = true;
					for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
					{
						if (m_processedFrame[i] != m_toProcess[i])
						{
							endEpoch = false;
						}
					}
					if(endEpoch)
					{
						return false;
					}
				}
				if (m_noData==false)
				{
					const msra::dbn::matrixstripe featRef= m_mbiter->frames();
					fdim = featRef.rows();
				}
				if (m_featuresBuffer==NULL)
				{
					m_featuresBuffer = new ElemType[fdim*m_mbSize*m_numberOfuttsPerMinibatch];
					m_featuresBufferAllocated = fdim*m_mbSize*m_numberOfuttsPerMinibatch;
				} else if (m_featuresBufferAllocated<fdim*m_mbSize*m_numberOfuttsPerMinibatch) //buffer size changed. can be partial minibatch
				{
					delete[] m_featuresBuffer;
					m_featuresBuffer = new ElemType[fdim*m_mbSize*m_numberOfuttsPerMinibatch];
					m_featuresBufferAllocated = fdim*m_mbSize*m_numberOfuttsPerMinibatch;
				}
				if (m_labelsBuffer==NULL)
				{
					m_labelsBuffer = new ElemType[m_udim*m_mbSize*m_numberOfuttsPerMinibatch];
					m_labelsBufferAllocated = m_udim*m_mbSize*m_numberOfuttsPerMinibatch;
				}
				else if (m_labelsBufferAllocated<m_udim*m_mbSize*m_numberOfuttsPerMinibatch)
				{
					delete[] m_labelsBuffer;
					m_labelsBuffer = new ElemType[m_udim*m_mbSize*m_numberOfuttsPerMinibatch];
					m_labelsBufferAllocated = m_udim*m_mbSize*m_numberOfuttsPerMinibatch;
				}
				memset(m_labelsBuffer,0,sizeof(ElemType)*m_udim*m_mbSize*m_numberOfuttsPerMinibatch);   
				vector<size_t> actualmbsize;
				actualmbsize.assign(m_numberOfuttsPerMinibatch,0);
				for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
				{
					size_t startFr = m_processedFrame[i];
					size_t endFr = 0;
					if ((m_processedFrame[i] + m_mbSize) < m_toProcess[i])
					{
						if(m_processedFrame[i] > 0)
						{
							m_sentenceEnd[i] = false;
							m_switchFrame[i] = m_mbSize+1;
						}else
						{
							m_switchFrame[i] = 0;
							m_sentenceEnd[i] = true;
						}
						actualmbsize[i] = m_mbSize;
						endFr = startFr + actualmbsize[i];	
						if (sizeof(ElemType) == sizeof(float))
						{
							for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
							{
								// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
								memcpy_s(&m_featuresBuffer[(k*m_numberOfuttsPerMinibatch+i)*fdim],sizeof(ElemType)*fdim,&m_featuresBufferMultiUtt[i][j*fdim],sizeof(ElemType)*fdim);
							}
						}
						else
						{
							for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
							{
								for (int d = 0; d < fdim; d++)
								{
									m_featuresBuffer[(k*m_numberOfuttsPerMinibatch+i)*fdim+d] = m_featuresBufferMultiUtt[i][j*fdim+d];
								}
							}
						}


						for (size_t j = startFr,k=0; j < endFr; j++,k++)
						{
							for (int d = 0; d < m_udim; d++)
							{
								m_labelsBuffer[(k*m_numberOfuttsPerMinibatch+i)*m_udim + d] = m_labelsBufferMultiUtt[i][j*m_udim+d];
							}
						}
						m_processedFrame[i] += m_mbSize;
					}else
					{
						actualmbsize[i] = m_toProcess[i] - m_processedFrame[i];
						endFr = startFr + actualmbsize[i];
						if (sizeof(ElemType) == sizeof(float))
						{
							for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
							{
								// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
								memcpy_s(&m_featuresBuffer[(k*m_numberOfuttsPerMinibatch+i)*fdim],sizeof(ElemType)*fdim,&m_featuresBufferMultiUtt[i][j*fdim],sizeof(ElemType)*fdim);
							}
						}
						else
						{
							for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
							{
								for (int d = 0; d < fdim; d++)
								{
									m_featuresBuffer[(k*m_numberOfuttsPerMinibatch+i)*fdim+d] = m_featuresBufferMultiUtt[i][j*fdim+d];
								}
							}
						}
						for (size_t j = startFr,k=0; j < endFr; j++,k++)
						{
							for (int d = 0; d < m_udim; d++)
							{
								m_labelsBuffer[(k*m_numberOfuttsPerMinibatch+i)*m_udim + d] = m_labelsBufferMultiUtt[i][j*m_udim+d];
							}
						}
						m_processedFrame[i] += (endFr-startFr);
						m_switchFrame[i] = actualmbsize[i];
						startFr = m_switchFrame[i];
						endFr = m_mbSize;
						bool reNewSucc = ReNewBuffer(i);

						if (sizeof(ElemType) == sizeof(float))
						{
							for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
							{
								// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
								memcpy_s(&m_featuresBuffer[(j*m_numberOfuttsPerMinibatch+i)*fdim],sizeof(ElemType)*fdim,&m_featuresBufferMultiUtt[i][k*fdim],sizeof(ElemType)*fdim);
							}
						}
						else
						{
							for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
							{
								for (int d = 0; d < fdim; d++)
								{
									m_featuresBuffer[(j*m_numberOfuttsPerMinibatch+i)*fdim+d] = m_featuresBufferMultiUtt[i][k*fdim+d];
								}
							}
						}
						for (size_t j = startFr,k=0; j < endFr; j++,k++)
						{
							for (int d = 0; d < m_udim; d++)
							{
								m_labelsBuffer[(j*m_numberOfuttsPerMinibatch+i)*m_udim + d] = m_labelsBufferMultiUtt[i][k*m_udim+d];
							}
						}

						if (reNewSucc) m_processedFrame[i] += (endFr-startFr);

					}
				}
				features.SetValue(fdim, m_mbSize*m_numberOfuttsPerMinibatch, m_featuresBuffer,matrixFlagNormal);
				labels.SetValue(m_udim,m_mbSize*m_numberOfuttsPerMinibatch,m_labelsBuffer,matrixFlagNormal);
				skip=false;
				/*				actualmbsize = 0;
				skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSource->totalframes() > actualmbsize);
				// check to see if we got the number of frames we requested
				if (!skip)
				{
				if (m_featuresBuffer==NULL)
				{
				m_featuresBuffer = new ElemType[feat.rows()*actualmbsize];
				m_featuresBufferAllocated = feat.rows()*actualmbsize;
				}
				else if (m_featuresBufferAllocated<feat.rows()*actualmbsize) //buffer size changed. can be partial minibatch
				{
				delete[] m_featuresBuffer;
				m_featuresBuffer = new ElemType[feat.rows()*actualmbsize];
				m_featuresBufferAllocated = feat.rows()*actualmbsize;
				}
				// shouldn't need this since we fill up the entire buffer below
				//memset(m_featuresBuffer,0,sizeof(ElemType)*feat.rows()*feat.cols());

				if (sizeof(ElemType) == sizeof(float))
				{
				for (int j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
				{
				// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
				memcpy_s(&m_featuresBuffer[k*feat.rows()],sizeof(ElemType)*feat.rows(),&feat(0,j),sizeof(ElemType)*feat.rows());
				}
				}
				else
				{
				for (int j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
				{
				for (int i = 0; i < feat.rows(); i++)
				{
				m_featuresBuffer[k*feat.rows()+i] = feat(i,j);
				}
				}
				}
				features.SetValue(feat.rows(), actualmbsize, m_featuresBuffer,matrixFlagNormal);

				// copy the labels over to array type
				//labels.Resize(m_udim, uids.size());
				//labels.SetValue((ElemType)0);
				if (m_labelsBuffer==NULL)
				{
				m_labelsBuffer = new ElemType[m_udim*actualmbsize];
				m_labelsBufferAllocated = m_udim*actualmbsize;
				}
				else if (m_labelsBufferAllocated<m_udim*actualmbsize)
				{
				delete[] m_labelsBuffer;
				m_labelsBuffer = new ElemType[m_udim*actualmbsize];
				m_labelsBufferAllocated = m_udim*actualmbsize;
				}
				memset(m_labelsBuffer,0,sizeof(ElemType)*m_udim*actualmbsize);                

				if (convertLabelsToTargets)
				{
				size_t labelDim = labelToTargetMap.size();
				for (int i = startFr,k=0; i < endFr; i++,k++)
				{
				assert(uids[i] < labelDim);
				size_t labelId = uids[i];
				for (int j = 0; j < m_udim; j++)
				{
				m_labelsBuffer[k*m_udim + j] = labelToTargetMap[labelId][j];
				}
				}
				}
				else
				{
				// loop through the columns and set one value to 1
				// in the future we want to use a sparse matrix here
				for (int i = startFr, k=0; i < endFr; i++,k++)
				{
				assert(uids[i] < m_udim);
				//labels(uids[i], i) = (ElemType)1;
				m_labelsBuffer[k*m_udim+uids[i]]=(ElemType)1;
				}
				}
				labels.SetValue(m_udim,actualmbsize,m_labelsBuffer,matrixFlagNormal);
				}
				m_processedFrame[0] += actualmbsize;
				// advance to the next minibatch
				if (m_processedFrame[0] == actualmbsizeOri)
				{
				(*m_mbiter)++;
				m_processedFrame[0] = 0;
				m_sentenceEnd[0] = true;
				}*/
			}
		}   // keep going if we didn't get the right size minibatch
		while (skip);

		return true;
	}
	template<class ElemType>
	bool HTKMLFReader<ElemType>::ReNewBufferForMultiIO(size_t i)
	{
		if (m_noData)
		{
			return false;
		}
		size_t numOfFea = m_featuresBufferMultiIO.size();
		size_t numOfLabel = m_labelsBufferMultiIO.size();

		size_t totalFeatNum = 0;
		foreach_index(id, m_featuresBufferAllocatedMultiIO)
		{
			const msra::dbn::matrixstripe featOri = m_mbiter->frames(id);
			size_t fdim = featOri.rows();
			const size_t actualmbsizeOri = featOri.cols(); 
			m_featuresStartIndexMultiUtt[id+i*numOfFea] = totalFeatNum;
			totalFeatNum = fdim * actualmbsizeOri + m_featuresStartIndexMultiUtt[id+i*numOfFea];
		}
		if (m_featuresBufferMultiUtt[i]==NULL)
		{
			m_featuresBufferMultiUtt[i] = new ElemType[totalFeatNum];
			m_featuresBufferAllocatedMultiUtt[i] = totalFeatNum;
		}					
		else if (m_featuresBufferAllocatedMultiUtt[i] < totalFeatNum) //buffer size changed. can be partial minibatch
		{
			delete[] m_featuresBufferMultiUtt[i];
			m_featuresBufferMultiUtt[i] = new ElemType[totalFeatNum];
			m_featuresBufferAllocatedMultiUtt[i] = totalFeatNum;
		}

		size_t totalLabelsNum = 0;
		for (map<std::wstring,size_t>::iterator it = labelNameToIdMap.begin(); it != labelNameToIdMap.end(); ++it) 
		{
			size_t id = labelNameToIdMap[it->first];
			size_t dim  = labelNameToDimMap[it->first];

			const vector<size_t> & uids = m_mbiter->labels(id);
			size_t actualmbsizeOri = uids.size();
			m_labelsStartIndexMultiUtt[id+i*numOfLabel] = totalLabelsNum;
			totalLabelsNum = m_labelsStartIndexMultiUtt[id+i*numOfLabel] + dim * actualmbsizeOri;
		}
		
		if (m_labelsBufferMultiUtt[i]==NULL)
		{
			m_labelsBufferMultiUtt[i] = new ElemType[totalLabelsNum];
			m_labelsBufferAllocatedMultiUtt[i] = totalLabelsNum;
		}
		else if (m_labelsBufferAllocatedMultiUtt[i] < totalLabelsNum)
		{
			delete[] m_labelsBufferMultiUtt[i];
			m_labelsBufferMultiUtt[i] = new ElemType[totalLabelsNum];
			m_labelsBufferAllocatedMultiUtt[i] = totalLabelsNum;
		}

		memset(m_labelsBufferMultiUtt[i],0,sizeof(ElemType)*totalLabelsNum);

		bool first = true;
		foreach_index(id, m_featuresBufferMultiIO)
		{
			const msra::dbn::matrixstripe featOri = m_mbiter->frames(id);
			const size_t actualmbsizeOri = featOri.cols(); 
			size_t fdim = featOri.rows();
			if (first)
			{
				m_toProcess[i] = actualmbsizeOri;
				first = false;
			} else
			{
				if (m_toProcess[i] != actualmbsizeOri)
				{
					throw std::runtime_error("The multi-IO features has inconsistent number of frames!");
				}
			}
			assert (actualmbsizeOri == m_mbiter->currentmbframes());
			const size_t mbstartframe = m_mbiter->currentmbstartframe();


			if (sizeof(ElemType) == sizeof(float))
			{
				for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns
				{
					// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
					memcpy_s(&m_featuresBufferMultiUtt[i][k*fdim+m_featuresStartIndexMultiUtt[id+i*numOfFea]],sizeof(ElemType)*fdim,&featOri(0,k),sizeof(ElemType)*fdim);
				}
			}
			else
			{
				for (int k=0; k < actualmbsizeOri; k++) // column major, so iterate columns in outside loop
				{
					for (int d = 0; d < featOri.rows(); d++)
					{
						m_featuresBufferMultiUtt[i][k*featOri.rows()+d+m_featuresStartIndexMultiUtt[id+i*numOfFea]] = featOri(d,k);
					}
				}
			}
		}
		
		for (map<std::wstring,size_t>::iterator it = labelNameToIdMap.begin(); it != labelNameToIdMap.end(); ++it) 
		{
			size_t id = labelNameToIdMap[it->first];
			size_t dim  = labelNameToDimMap[it->first];

			const vector<size_t> & uids = m_mbiter->labels(id);
			size_t actualmbsizeOri = uids.size();

			if (convertLabelsToTargets)
			{
				size_t labelDim = labelToTargetMap.size();
				for (int k=0; k < actualmbsizeOri; k++)
				{
					assert(uids[k] < labelDim);
					size_t labelId = uids[k];
					for (int j = 0; j < dim; j++)
					{
						m_labelsBufferMultiUtt[i][k*dim + j + m_labelsStartIndexMultiUtt[id+i*numOfLabel]] = labelToTargetMap[labelId][j];
					}
				}
			}
			else
			{
				// loop through the columns and set one value to 1
				// in the future we want to use a sparse matrix here
				for (int k=0; k < actualmbsizeOri; k++)
				{
					assert(uids[k] < dim);
					//labels(uids[i], i) = (ElemType)1;
					m_labelsBufferMultiUtt[i][k*dim+uids[k]+m_labelsStartIndexMultiUtt[id+i*numOfLabel]]=(ElemType)1;
				}
			}
		}
		m_processedFrame[i] = 0;

		(*m_mbiter)++;
		if (!(*m_mbiter))
			m_noData = true;

		return true;	
	}
	template<class ElemType>
	bool HTKMLFReader<ElemType>::ReNewBuffer(size_t i)
	{
		if (m_noData)
		{
			return false;
		}
		const msra::dbn::matrixstripe featOri= m_mbiter->frames();
		const vector<size_t> & uids = m_mbiter->labels();
		const size_t actualmbsizeOri = featOri.cols(); 
		size_t fdim = featOri.rows();
		m_toProcess[i] = actualmbsizeOri;
		assert (actualmbsizeOri == m_mbiter->currentmbframes());
		const size_t mbstartframe = m_mbiter->currentmbstartframe();
		if (m_featuresBufferMultiUtt[i]==NULL)
		{
			m_featuresBufferMultiUtt[i] = new ElemType[fdim* actualmbsizeOri];
			m_featuresBufferAllocatedMultiUtt[i] = fdim*actualmbsizeOri;
		}					
		else if (m_featuresBufferAllocatedMultiUtt[i] <fdim*actualmbsizeOri) //buffer size changed. can be partial minibatch
		{
			delete[] m_featuresBufferMultiUtt[i];
			m_featuresBufferMultiUtt[i] = new ElemType[fdim*actualmbsizeOri];
			m_featuresBufferAllocatedMultiUtt[i] = fdim*actualmbsizeOri;
		}

		if (sizeof(ElemType) == sizeof(float))
		{
			for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns
			{
				// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
				memcpy_s(&m_featuresBufferMultiUtt[i][k*fdim],sizeof(ElemType)*fdim,&featOri(0,k),sizeof(ElemType)*fdim);
			}
		}
		else
		{
			for (int k=0; k < actualmbsizeOri; k++) // column major, so iterate columns in outside loop
			{
				for (int d = 0; d < featOri.rows(); d++)
				{
					m_featuresBufferMultiUtt[i][k*featOri.rows()+d] = featOri(d,k);
				}
			}
		}
		if (m_labelsBufferMultiUtt[i]==NULL)
		{
			m_labelsBufferMultiUtt[i] = new ElemType[m_udim*actualmbsizeOri];
			m_labelsBufferAllocatedMultiUtt[i] = m_udim*actualmbsizeOri;
		}
		else if (m_labelsBufferAllocatedMultiUtt[i]<m_udim*actualmbsizeOri)
		{
			delete[] m_labelsBufferMultiUtt[i];
			m_labelsBufferMultiUtt[i] = new ElemType[m_udim*actualmbsizeOri];
			m_labelsBufferAllocatedMultiUtt[i] = m_udim*actualmbsizeOri;
		}
		memset(m_labelsBufferMultiUtt[i],0,sizeof(ElemType)*m_udim*actualmbsizeOri);                

		if (convertLabelsToTargets)
		{
			size_t labelDim = labelToTargetMap.size();
			for (int k=0; k < actualmbsizeOri; k++)
			{
				assert(uids[k] < labelDim);
				size_t labelId = uids[k];
				for (int j = 0; j < m_udim; j++)
				{
					m_labelsBufferMultiUtt[i][k*m_udim + j] = labelToTargetMap[labelId][j];
				}
			}
		}
		else
		{
			// loop through the columns and set one value to 1
			// in the future we want to use a sparse matrix here
			for (int k=0; k < actualmbsizeOri; k++)
			{
				assert(uids[k] < m_udim);
				//labels(uids[i], i) = (ElemType)1;
				m_labelsBufferMultiUtt[i][k*m_udim+uids[k]]=(ElemType)1;
			}
		}
		m_toProcess[i] = actualmbsizeOri;
		m_processedFrame[i] = 0;

		(*m_mbiter)++;
		if (!(*m_mbiter))
			m_noData = true;

		return true;
	}

	template<class ElemType>
	bool HTKMLFReader<ElemType>::GetMinibatchMultiIO(std::map<std::wstring, Matrix<ElemType>*>& matrices)
	{
		size_t id;
		size_t dim;
		bool skip = false;
		do 
		{
			if (!(*m_mbiter))
				return false;

			const size_t mbstartframe = m_mbiter->currentmbstartframe();

			// now, access all features and and labels by iterating over map of "matrices"
			std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
			for (iter = matrices.begin();iter!=matrices.end(); iter++)
			{
				// dereference matrix that corresponds to key (input/output name) and 
				// populate based on whether its a feature or a label
				Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

				if (nameToTypeMap[iter->first] == InputOutputTypes::inputReal || 
					nameToTypeMap[iter->first] == InputOutputTypes::outputReal)
				{
					//                switch (nameToTypeMap[iter->first])
					//                {
					//                case InputOutputTypes::inputReal:
					//                case InputOutputTypes::outputReal:
					//                    {
					// fprintf(stderr, "name:  %S  type: input features\n", (iter->first).c_str());

					id = featureNameToIdMap[iter->first];
					dim = featureNameToDimMap[iter->first];
					const msra::dbn::matrixstripe feat = m_mbiter->frames(id);
					const size_t actualmbsize = feat.cols();   // it may still return less if at end of sweep TODO: this check probably only needs to happen once
					assert (actualmbsize == m_mbiter->currentmbframes());
					skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSourceMultiIO->totalframes() > actualmbsize);

					// check to see if we got the number of frames we requested
					if (!skip)
					{
						// copy the features over to our array type
						assert(feat.rows()==dim); // check feature dimension matches what's expected

						if (m_featuresBufferMultiIO[id]==NULL)
						{
							m_featuresBufferMultiIO[id] = new ElemType[feat.rows()*feat.cols()];
							m_featuresBufferAllocatedMultiIO[id] = feat.rows()*feat.cols();
						}
						else if (m_featuresBufferAllocatedMultiIO[id]<feat.rows()*feat.cols()) //buffer size changed. can be partial minibatch
						{
							delete[] m_featuresBufferMultiIO[id];
							m_featuresBufferMultiIO[id] = new ElemType[feat.rows()*feat.cols()];
							m_featuresBufferAllocatedMultiIO[id] = feat.rows()*feat.cols();
						}
						// shouldn't need this since we fill up the entire buffer below
						//memset(m_featuresBufferMultiIO[id],0,sizeof(ElemType)*feat.rows()*feat.cols());

						if (sizeof(ElemType) == sizeof(float))
						{
							for (int j=0; j < feat.cols(); j++) // column major, so iterate columns
							{
								// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
								memcpy_s(&m_featuresBufferMultiIO[id][j*feat.rows()],sizeof(ElemType)*feat.rows(),&feat(0,j),sizeof(ElemType)*feat.rows());
							}
						}
						else
						{
							for (int j=0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
							{
								for (int i = 0; i < feat.rows(); i++)
								{
									m_featuresBufferMultiIO[id][j*feat.rows()+i] = feat(i,j);
								}
							}
						}
						data.SetValue(feat.rows(), feat.cols(), m_featuresBufferMultiIO[id],matrixFlagNormal);
					}
				}
				else if (nameToTypeMap[iter->first] == InputOutputTypes::inputCategory || 
					nameToTypeMap[iter->first] == InputOutputTypes::outputCategory)
				{
					//break;
					//case InputOutputTypes::inputCategory:
					//case InputOutputTypes::outputCategory:
					//    {
					// fprintf(stderr, "name:  %S  = output features\n", (iter->first).c_str());
					id = labelNameToIdMap[iter->first];
					dim = labelNameToDimMap[iter->first];
					const vector<size_t> & uids = m_mbiter->labels(id);

					//size_t m_udim = udims[id];
					//size_t m_udim = labelDims[id];

					// need skip logic here too in case labels are first in map not features
					const size_t actualmbsize = uids.size();   // it may still return less if at end of sweep TODO: this check probably only needs to happen once
					assert (actualmbsize == m_mbiter->currentmbframes());
					skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSourceMultiIO->totalframes() > actualmbsize);

					if (!skip)
					{
						// copy the labels over to array type
						//data.Resize(udims[id], uids.size());
						//data.SetValue((ElemType)0);

						// loop through the columns and set one value to 1
						// in the future we want to use a sparse matrix here
						//for (int i = 0; i < uids.size(); i++)
						//{
						//    assert(uids[i] <udims[id]);
						//    data(uids[i], i) = (ElemType)1;
						//}

						if (m_labelsBufferMultiIO[id]==NULL)
						{
							m_labelsBufferMultiIO[id] = new ElemType[dim*uids.size()];
							m_labelsBufferAllocatedMultiIO[id] = dim*uids.size();
						}
						else if (m_labelsBufferAllocatedMultiIO[id]<dim*uids.size())
						{
							delete[] m_labelsBufferMultiIO[id];
							m_labelsBufferMultiIO[id] = new ElemType[dim*uids.size()];
							m_labelsBufferAllocatedMultiIO[id] = dim*uids.size();
						}
						memset(m_labelsBufferMultiIO[id],0,sizeof(ElemType)*dim*uids.size());                


						if (convertLabelsToTargetsMultiIO[id])
						{
							size_t labelDim = labelToTargetMapMultiIO[id].size();
							for (int i = 0; i < uids.size(); i++)
							{
								assert(uids[i] < labelDim);
								size_t labelId = uids[i];
								for (int j = 0; j < dim; j++)
								{
									m_labelsBufferMultiIO[id][i*dim + j] = labelToTargetMapMultiIO[id][labelId][j];
								}
							}
						}
						else
						{
							// loop through the columns and set one value to 1
							// in the future we want to use a sparse matrix here
							for (int i = 0; i < uids.size(); i++)
							{
								assert(uids[i] < dim);
								//labels(uids[i], i) = (ElemType)1;
								m_labelsBufferMultiIO[id][i*dim+uids[i]]=(ElemType)1;
							}
						}


						data.SetValue(dim,uids.size(),m_labelsBufferMultiIO[id],matrixFlagNormal);
					}
				}
				else{
					//default:
					throw runtime_error(msra::strfun::strprintf("GetMinibatchMultiIO:: unknown InputOutputType for %S\n",(iter->first).c_str()));
				}

			}
			// advance to the next minibatch
			(*m_mbiter)++;
		}   // keep going if we didn't get the right size minibatch
		while(skip);

		return true;
	}

	template<class ElemType>
	bool HTKMLFReader<ElemType>:: GetMinibatchEval(std::map<std::wstring, Matrix<ElemType>*>&matrices)
	{
		std::map<std::wstring,size_t>::iterator iter;
		if 	(checkDictionaryKeys)
		{
			for (auto iter=featureNameToIdMap.begin();iter!=featureNameToIdMap.end();iter++)
			{
				if (matrices.find(iter->first)==matrices.end())
					throw std::runtime_error(msra::strfun::strprintf("input node %ws not found in reader - cannot generate input\n",iter->first.c_str()));

			}
			/*
			for (auto iter=matrices.begin();iter!=matrices.end();iter++)
			{
			if (featureNameToIdMap.find(iter->first)==featureNameToIdMap.end())
			throw std::runtime_error(msra::strfun::strprintf("input node %ws not found in reader - cannot generate input\n",iter->first.c_str()));
			}
			*/
			checkDictionaryKeys=false;
		}

		// if (frames left to process in current file(s) )
		// {
		//		populate minibatch from current file
		//		return true
		// }
		// else
		// {
		//		populate matrices as empty to indicate end of file
		//		if (files left to process)
		//		{
		//			load new file
		//			return true
		//		}
		//		else
		//		{
		//			return false
		//		}
		// }

		if (inputFileIndex<inputFilesMultiIO[0].size())
		{
			m_fileEvalSource->Reset();

			// load next file (or set of files)
			foreach_index(i, inputFilesMultiIO)
			{
				msra::asr::htkfeatreader reader;

				const auto path = reader.parse(inputFilesMultiIO[i][inputFileIndex]);
				// read file
				msra::dbn::matrix feat;
				string featkind;
				unsigned int sampperiod;
				msra::util::attempt (5, [&]()
				{
					reader.read (path, featkind, sampperiod, feat);   // whole file read as columns of feature vectors
				});
				fprintf (stderr, "evaluate: reading %d frames of %S\n", feat.cols(), ((wstring)path).c_str());
				m_fileEvalSource->AddFile(feat, featkind, sampperiod, i);
			}
			inputFileIndex++;

			// turn frames into minibatch (augment neighbors, etc)
			m_fileEvalSource->CreateEvalMinibatch();

			// populate input matrices

			std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
			for (iter = matrices.begin();iter!=matrices.end(); iter++)
			{
				// dereference matrix that corresponds to key (input/output name) and 
				// populate based on whether its a feature or a label

				if (nameToTypeMap.find(iter->first)!=nameToTypeMap.end() && nameToTypeMap[iter->first] == InputOutputTypes::inputReal)
				{
					Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels
					//if (nameToTypeMap[iter->first] == InputOutputTypes::inputReal)
					//{
					size_t id = featureNameToIdMap[iter->first];
					size_t dim = featureNameToDimMap[iter->first];

					const msra::dbn::matrix feat = m_fileEvalSource->ChunkOfFrames(id);
					const size_t actualmbsize = feat.cols();   // it may still return less if at end of sweep TODO: this check probably only needs to happen once

					// copy the features over to our array type
					assert(feat.rows()==dim); // check feature dimension matches what's expected

					if (m_featuresBufferMultiIO[id]==NULL)
					{
						m_featuresBufferMultiIO[id] = new ElemType[feat.rows()*feat.cols()];
						m_featuresBufferAllocatedMultiIO[id] = feat.rows()*feat.cols();
					}
					else if (m_featuresBufferAllocatedMultiIO[id]<feat.rows()*feat.cols()) //buffer size changed. can be partial minibatch
					{
						delete[] m_featuresBufferMultiIO[id];
						m_featuresBufferMultiIO[id] = new ElemType[feat.rows()*feat.cols()];
						m_featuresBufferAllocatedMultiIO[id] = feat.rows()*feat.cols();
					}
					// shouldn't need this since we fill up the entire buffer below
					//memset(m_featuresBufferMultiIO[id],0,sizeof(ElemType)*feat.rows()*feat.cols());

					if (sizeof(ElemType) == sizeof(float))
					{
						for (int j=0; j < feat.cols(); j++) // column major, so iterate columns
						{
							// copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
							memcpy_s(&m_featuresBufferMultiIO[id][j*feat.rows()],sizeof(ElemType)*feat.rows(),&feat(0,j),sizeof(ElemType)*feat.rows());
						}
					}
					else
					{
						for (int j=0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
						{
							for (int i = 0; i < feat.rows(); i++)
							{
								m_featuresBufferMultiIO[id][j*feat.rows()+i] = feat(i,j);
							}
						}
					}
					data.SetValue(feat.rows(), feat.cols(), m_featuresBufferMultiIO[id],matrixFlagNormal);
				}
			}
			return true;
		}
		else
		{
			return false;
		}
	}
	// GetMinibatch - Get the next minibatch (features and labels)
	// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
	//             [out] each matrix resized if necessary containing data. 
	// returns - true if there are more minibatches, false if no more minibatchs remain
	template<class ElemType>
	bool HTKMLFReader<ElemType>::GetMinibatchEvalSingleIO(std::map<std::wstring, Matrix<ElemType>*>& matrices)
	{
		// confirm expected dictionary names are there...
		// only needed first time...

		if (checkDictionaryKeys)
		{
			if (matrices.find(DefaultFeaturesName())==matrices.end())
				throw std::runtime_error("input node 'features' not found - expected for single I/O networks\n");
			if (matrices.find(outputNodeName)==matrices.end())
				throw std::runtime_error(msra::strfun::strprintf("output node %ws not found - cannot generate output\n",outputNodeName.c_str()));
			if (scaleByPrior && matrices.find(DefaultPriorName())==matrices.end())
				throw std::runtime_error("network Prior node not found - needed to output likelihoods\n");

			checkDictionaryKeys=false;
		}
		Matrix<ElemType>& features = *matrices[DefaultFeaturesName()];
		Matrix<ElemType>& outputs = *matrices[outputNodeName];
		Matrix<ElemType>& logPriors = *matrices[DefaultPriorName()];

		msra::asr::htkfeatreader reader;
		msra::dbn::matrix preds;

		size_t numuptodate=0;
		int addEnergy = 0;
		reader.AddEnergy(addEnergy);

		size_t evalMBSize=2048;
		size_t actualMBSize=0;
		// first check if output matrices are populated. if so, write their contents to one or more files
		// then reset outputs to size = 0
		// then fill features with new context

		if (outputs.GetNumCols()>0)
		{
			size_t returnedMBSize = m_chunkEvalSource->currentchunksize();

			assert(outputs.GetNumCols()==returnedMBSize);
			assert(outputs.GetNumRows()==udims[0]);
			ElemType maxLogPrior = (ElemType)-1e30f;
			if (scaleByPrior)
			{
				assert(logPriors.GetNumRows()==outputs.GetNumRows());
				assert(logPriors.GetNumCols()==1);

				// not necessary, but for compatibility with DBN tool...
				for (int i=0; i<logPriors.GetNumRows();i++)
				{
					if (logPriors(i,0) > maxLogPrior)
						maxLogPrior = logPriors(i, 0);
				}
			}

			// need to convert from cntk format back to msra format
			// is there a faster way to do this? 
			preds.resize(udims[0],returnedMBSize);
			for (int j=0; j< outputs.GetNumCols(); j++)
			{
				for (int i=0; i<outputs.GetNumRows(); i++)
				{
					preds(i,j) = (float)outputs(i,j);
					if (scaleByPrior)
						preds(i,j) -= (float)(logPriors(i,0) - maxLogPrior);
				}
			}
			m_chunkEvalSource->writetofiles(preds);
		}

		while (m_chunkEvalSource->currentchunksize() < evalMBSize && inputFileIndex<inputFiles.size())
		{
			const auto path = reader.parse(inputFiles[inputFileIndex]);
			wstring outfile = MakeOutPath (outputPath, path, outputExtension);
			outfiles.push_back(outfile);
			if (msra::files::fuptodate (outfile, path.physicallocation()))
			{
				fprintf(stderr, "evaluate: %S is up to date...skipping\n", outfile.c_str());
				numuptodate++;
				continue;
			}
			// read file
			msra::dbn::matrix feat;
			string featkind;
			unsigned int sampperiod;
			msra::util::attempt (5, [&]()
			{
				reader.read (path, featkind, sampperiod, feat);   // whole file read as columns of feature vectors
			});
			fprintf (stderr, "evaluate: reading %d frames of %S\n", feat.cols(), ((wstring)path).c_str());

			m_chunkEvalSource->addfile(feat, featkind, sampperiod, outfile);

			inputFileIndex++;

		}

		if (m_chunkEvalSource->currentchunksize()>0)
		{

			m_chunkEvalSource->createevalminibatch();
			const msra::dbn::matrix feat = m_chunkEvalSource->chunkofframes();

			if (m_featuresBuffer==NULL)
			{
				m_featuresBuffer = new ElemType[feat.rows()*feat.cols()];
				m_featuresBufferAllocated = feat.rows()*feat.cols();
			}
			else if (m_featuresBufferAllocated<feat.rows()*feat.cols()) //buffer size changed. can be partial minibatch
			{
				delete[] m_featuresBuffer;
				m_featuresBuffer = new ElemType[feat.rows()*feat.cols()];
				m_featuresBufferAllocated = feat.rows()*feat.cols();
			}
			memset(m_featuresBuffer,0,sizeof(ElemType)*feat.rows()*feat.cols());

			//unfortunatelly direct memcpy can't be done safely here because feat.p is always float*, not ElemType*
			for (int j=0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
			{
				for (int i = 0; i < feat.rows(); i++)
				{
					m_featuresBuffer[j*feat.rows()+i] = feat(i,j);
				}
			}
			features.SetValue(feat.rows(), feat.cols(), m_featuresBuffer,matrixFlagNormal);

			return true;
		}
		else
		{
			// done - have all output files written, so write out SCP file 
			if (!outputScp.empty())
			{
				WriteOutputScp();
			}
			return false;
		}

	}

	template<class ElemType>
	bool HTKMLFReader<ElemType>::GetMinibatchEvalMultiIO(std::map<std::wstring, Matrix<ElemType>*>& matrices)
	{
		if (checkDictionaryKeys)
		{
			std::map<std::wstring,size_t>::iterator iter;
			for (iter=featureNameToIdMap.begin();iter!=featureNameToIdMap.end();iter++)
			{
				if (matrices.find(iter->first)==matrices.end())
					throw std::runtime_error(msra::strfun::strprintf("output node %ws not found - cannot generate output\n",iter->first.c_str()));
			}
			for (iter=outputNameToIdMap.begin();iter!=outputNameToIdMap.end();iter++)
			{
				if (matrices.find(iter->first)==matrices.end())
					throw std::runtime_error(msra::strfun::strprintf("output node %ws not found - cannot generate output\n",iter->first.c_str()));
			}			
			checkDictionaryKeys=false;
		}
#if 0
		// write data from previous batch to disk
		std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
		for (iter = matrices.begin();iter!=matrices.end(); iter++)
		{
			wstring nodeName = iter->first;
			if (nameToTypeMap.find(nodeName)!=nameToTypeMap.end()) // node is one we care about (could be input or output)
			{

				if (nameToTypeMap[nodeName]==InputOutputTypes::networkOutputs) // it's an output - let's write the data to disk 
				{
					Matrix<ElemType>& output = *matrices[iter->first]; // can be features or labels
					size_t index = outputNodeNames
						switch (nameToTypeMap[iter->first])
					{
						case InputOutputTypes::inputReal:

							msra::asr::htkfeatreader reader;
							msra::dbn::matrix preds;

							size_t numuptodate=0;
							int addEnergy = 0;
							reader.AddEnergy(addEnergy);

							size_t evalMBSize=2048;
							size_t actualMBSize=0;
							// first check if output matrices are populated. if so, write their contents to one or more files
							// then reset outputs to size = 0
							// then fill features with new context

							if (outputs.GetNumCols()>0)
							{
								size_t returnedMBSize = m_chunkEvalSource->currentchunksize();

								assert(outputs.GetNumCols()==returnedMBSize);
								if (scaleByPrior)
								{
									assert(logPriors.GetNumRows()==outputs.GetNumRows());
									assert(logPriors.GetNumCols()==1);
								}

								// need to convert from cntk format back to msra format
								// is there a faster way to do this? 
								preds.resize(udims[0],returnedMBSize);
								for (int j=0; j< outputs.GetNumCols(); j++)
								{
									for (int i=0; i<outputs.GetNumRows(); i++)
									{
										preds(i,j) = (float)outputs(i,j);
										if (scaleByPrior)
											preds(i,j) -= (float)logPriors(i,0);
									}
								}
								m_chunkEvalSource->writetofiles(preds);
							}

							throw std::runtime_error("need to reset saveandflush explcitly");
#endif 
		
							throw std::runtime_error("HTKMLFReader: GetMinibatchEvalMultiIO not implemented yet!");
							return true;
					}

					// GetLabelMapping - Gets the label mapping from integer to type in file 
					// mappingTable - a map from numeric datatype to native label type stored as a string 
					template<class ElemType>
					const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& HTKMLFReader<ElemType>::GetLabelMapping(const std::wstring& sectionName)
					{
						return m_idToLabelMap;
					}

					// SetLabelMapping - Sets the label mapping from integer index to label 
					// labelMapping - mapping table from label values to IDs (must be 0-n)
					// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
					template<class ElemType>
					void HTKMLFReader<ElemType>::SetLabelMapping(const std::wstring& sectionName, const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& labelMapping)
					{
						m_idToLabelMap = labelMapping;
					}

					template<class ElemType>
					size_t HTKMLFReader<ElemType>::ReadLabelToTargetMappingFile (const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap)
					{
						if (labelListFile==L"")
							throw std::runtime_error("HTKMLFReader::ReadLabelToTargetMappingFile(): cannot read labelToTargetMappingFile without a labelMappingFile!");

						vector<std::wstring> labelList;
						size_t count, numLabels;
						count=0;
						// read statelist first
						msra::files::textreader labelReader(labelListFile);
						while(labelReader)
						{
							labelList.push_back(labelReader.wgetline());
							count++;
						}
						numLabels=count;
						count=0;
						msra::files::textreader mapReader(labelToTargetMappingFile);
						size_t targetDim = 0;
						while(mapReader)
						{
							std::wstring line(mapReader.wgetline());
							// find white space as a demarcation
							std::wstring::size_type pos = line.find(L" ");
							std::wstring token = line.substr(0,pos);
							std::wstring targetstring = line.substr(pos+1);

							if (labelList[count]!=token)
								throw new std::runtime_error("HTKMLFReader::ReadLabelToTargetMappingFile(): mismatch between labelMappingFile and labelToTargetMappingFile");

							if (count==0)
								targetDim = targetstring.length();
							else if (targetDim!=targetstring.length())
								throw new std::runtime_error("HTKMLFReader::ReadLabelToTargetMappingFile(): inconsistent target length among records");

							std::vector<ElemType> targetVector(targetstring.length(),(ElemType)0.0);
							foreach_index(i, targetstring)
							{
								if (targetstring.compare(i,1,L"1")==0)
									targetVector[i] = (ElemType)1.0;
								else if (targetstring.compare(i,1,L"0")!=0)
									throw new std::runtime_error("HTKMLFReader::ReadLabelToTargetMappingFile(): expecting label2target mapping to contain only 1's or 0's");
							}
							labelToTargetMap.push_back(targetVector);
							count++;
						}

						// verify that statelist and label2target mapping file are in same order (to match up with reader) while reading mapping
						if (count!=labelList.size())
							throw new std::runtime_error("HTKMLFReader::ReadLabelToTargetMappingFile(): mismatch between lengths of labelMappingFile vs labelToTargetMappingFile");

						return targetDim;
					}

					// GetData - Gets metadata from the specified section (into CPU memory) 
					// sectionName - section name to retrieve data from
					// numRecords - number of records to read
					// data - pointer to data buffer, if NULL, dataBufferSize will be set to size of required buffer to accomidate request
					// dataBufferSize - [in] size of the databuffer in bytes
					//                  [out] size of buffer filled with data
					// recordStart - record to start reading from, defaults to zero (start of data)
					// returns: true if data remains to be read, false if the end of data was reached
					template<class ElemType>
					bool HTKMLFReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
					{
						throw std::runtime_error("GetData not supported in HTKMLFReader");
					}

					// form an output path
					//  - 'outext' is not supposed to have a . in it
					//    No extension change if empty.
					//  - 'outdir' is not supposed to have a trailing slash
					//    No path change if empty.
					template<class ElemType>
					std::wstring HTKMLFReader<ElemType>::MakeOutPath (const std::wstring & outdir, std::wstring file, const std::wstring & outext)
					{
						// replace directory
						if (!outdir.empty())
						{
							file = regex_replace (file, wregex (L".*[\\\\/:]"), wstring()); // delete path
							size_t nsl = 0, nbsl = 0;   // count whether the path uses / or \ convention, and stick with it
							foreach_index (i, outdir)
							{
								if (outdir[i] == '/') nsl++;
								else if (outdir[i] == '\\') nbsl++;
							}
							file = outdir + (nbsl > nsl ? L"\\" : L"/") + file;   // prepend new path
						}
						// replace output extension
						if (!outext.empty())
						{
							file = regex_replace (file, wregex (L"\\.[^\\.\\\\/:]*$"), wstring());  // delete extension (or not if none)
							file += L"." + outext;      // and add the new one
						}
						return file;
					}

					template<class ElemType>
					void HTKMLFReader<ElemType>::WriteOutputScp()
					{
						wstring operation = L"HTKMLFReader::WriteOutputScp()";
						fprintf (stderr, "%S: creating script file '%S'\n", operation.c_str(), outputScp.c_str());
						msra::util::attempt (5, [&]()
						{
							auto_file_ptr f = fopenOrDie (outputScp, L"wbS");
							foreach_index (i, outfiles)
							{
								// save in default code page, assuming that's how hapiVite would interpret it
								int rc = fprintf (f, "%S\n", outfiles[i].c_str());
								if (rc <= 0) throw std::runtime_error (msra::strfun::strprintf ("%S: error '%s' when saving script to '%S'", operation.c_str(), strerror (errno), outputScp.c_str()));
							}
							fflushOrDie (f);
						});
					}

					template<class ElemType>
					bool HTKMLFReader<ElemType>::DataEnd(EndDataType endDataType)
					{
						// each minibatch is considered a "sentence"
						// other datatypes not really supported...
						// assert(endDataType == endDataSentence);
						// for the truncated BPTT, we need to support check wether it's the end of data
						bool ret = false;
						switch (endDataType)
						{
						case endDataNull:
							assert(false);
							break;
						case endDataEpoch:
						case endDataSet:
							assert(false); // not support
							break;
						case endDataSentence:
							if (m_truncated)
								ret = m_sentenceEnd[0];
							else
								ret = true; // useless in current condition
							break;
						}
						return ret;
					}

					template<class ElemType>
					void HTKMLFReader<ElemType>::SetSentenceEndInBatch(vector<size_t> &sentenceEnd)
					{
						sentenceEnd.resize(m_switchFrame.size());
						for (size_t i = 0; i < m_switchFrame.size() ; i++)
						{
							sentenceEnd[i] = m_switchFrame[i];
						}
					}

					// GetFileConfigNames - determine the names of the features and labels sections in the config file
					// features - [in,out] a vector of feature name strings
					// labels - [in,out] a vector of label name strings
					template<class ElemType>
					void HTKMLFReader<ElemType>::GetDataNamesFromConfig(const ConfigParameters& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels)
					{
						for (auto iter = readerConfig.begin(); iter != readerConfig.end(); ++iter)
						{
							auto pair = *iter;
							ConfigParameters temp = iter->second;
							// see if we have a config parameters that contains a "file" element, it's a sub key, use it
							if (temp.ExistsCurrent("scpFile"))
							{
								features.push_back(msra::strfun::utf16(iter->first));
							}
							else if (temp.ExistsCurrent("mlfFile"))
							{
								labels.push_back(msra::strfun::utf16(iter->first));
							}
							
						}
					}
#if 0
					// GetFileConfigNames - determine the names of the features and labels sections in the config file
					// features - [in,out] a vector of feature name strings
					// labels - [in,out] a vector of label name strings
					template<class ElemType>
					void HTKMLFReader<ElemType>::GetDataNamesFromConfigLegacy(const ConfigParameters& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels)
					{
						for (auto iter = readerConfig.begin(); iter != readerConfig.end(); ++iter)
						{
							auto pair = *iter;
							ConfigParameters temp = iter->second;
							// see if we have a config parameters that contains a "file" element, it's a sub key, use it
							if (temp.ExistsCurrent("file"))
							{
								if (temp.ExistsCurrent("labelMappingFile") 
									|| temp.ExistsCurrent("labelDim")
									|| temp.ExistsCurrent("labelType"))
								{
									labels.push_back(msra::strfun::utf16(iter->first));
								}
								else
								{
									features.push_back(msra::strfun::utf16(iter->first));
								}
							}
						}
					}
#endif // 0
					template class HTKMLFReader<float>;
					template class HTKMLFReader<double>;

				}}}