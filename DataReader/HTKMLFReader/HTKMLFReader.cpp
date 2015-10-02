//
// <copyright file="HTKMLFReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "basetypes.h"

#include "htkfeatio.h"                  // for reading HTK features
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "simplesenonehmm.h"            // for MMI scoring
#include "msra_mgram.h"                 // for unigram scores of ground-truth path in sequence training

#include "rollingwindowsource.h"        // minibatch sources
#include "utterancesourcemulti.h"
#include "chunkevalsource.h"
#include "minibatchiterator.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "commandArgUtil.h"
#include "HTKMLFReader.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

#ifdef __unix__
#include <limits.h>
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UNINT32;
#endif
#pragma warning (disable: 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this

#ifdef _WIN32
int msra::numa::node_override = -1;     // for numahelpers.h
#endif

namespace msra { namespace lm {
/*static*/ const mgram_map::index_t mgram_map::nindex = (mgram_map::index_t) -1; // invalid index
}}

namespace Microsoft { namespace MSR { namespace CNTK {

    // Create a Data Reader
    //DATAREADER_API IDataReader* DataReaderFactory(void)

    template<class ElemType>
        void HTKMLFReader<ElemType>::Init(const ConfigParameters& readerConfig)
        {
            m_truncated = readerConfig("Truncated", "false");
            m_fullutt = readerConfig("FullUtt", "false"); //read full utt in truncated mode
            m_convertLabelsToTargets = false;

            ConfigArray numberOfuttsPerMinibatchForAllEpochs = readerConfig("nbruttsineachrecurrentiter", "1");
            m_numberOfuttsPerMinibatchForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;

            for (int i = 0; i < m_numberOfuttsPerMinibatchForAllEpochs.size(); i++)
            {
                m_numberOfuttsPerMinibatch = m_numberOfuttsPerMinibatchForAllEpochs[i];
                if (m_numberOfuttsPerMinibatch < 1)
                {
                    LogicError("nbrUttsInEachRecurrentIter cannot be less than 1.");
                }

                if (!m_truncated && m_numberOfuttsPerMinibatch != 1)
                {
                    LogicError("nbrUttsInEachRecurrentIter has to be 1 if Truncated is set to false.");
                }
            }

            m_numberOfuttsPerMinibatch = m_numberOfuttsPerMinibatchForAllEpochs[0];

            m_actualnumberOfuttsPerMinibatch = m_numberOfuttsPerMinibatch;
            m_sentenceEnd.assign(m_numberOfuttsPerMinibatch, true);
            m_processedFrame.assign(m_numberOfuttsPerMinibatch, 0);
            m_toProcess.assign(m_numberOfuttsPerMinibatch,0);
            m_switchFrame.assign(m_numberOfuttsPerMinibatch,0);
            m_validFrame.assign(m_numberOfuttsPerMinibatch, 0);
            m_noData = false;

            string command(readerConfig("action",L"")); //look up in the config for the master command to determine whether we're writing output (inputs only) or training/evaluating (inputs and outputs)

            if (readerConfig.Exists("legacyMode"))
                RuntimeError("legacy mode has been deprecated\n");

            if (command == "write"){
                m_trainOrTest = false;
                PrepareForWriting(readerConfig);
            }
            else{
                m_trainOrTest = true;
                PrepareForTrainingOrTesting(readerConfig);
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
            vector<size_t> numContextLeft;
            vector<size_t> numContextRight;

            // for the multi-utterance process
            m_featuresBufferMultiUtt.assign(m_numberOfuttsPerMinibatch, nullptr);
            m_featuresBufferAllocatedMultiUtt.assign(m_numberOfuttsPerMinibatch,0);
            m_labelsBufferMultiUtt.assign(m_numberOfuttsPerMinibatch, nullptr);
            m_labelsBufferAllocatedMultiUtt.assign(m_numberOfuttsPerMinibatch,0);

            // for the multi-utterance process for lattice and phone boundary
            m_latticeBufferMultiUtt.assign(m_numberOfuttsPerMinibatch, nullptr);
            m_labelsIDBufferMultiUtt.resize(m_numberOfuttsPerMinibatch);
            m_phoneboundaryIDBufferMultiUtt.resize(m_numberOfuttsPerMinibatch);
            std::vector<std::wstring> featureNames;
            std::vector<std::wstring> labelNames;
            // for hmm and lattice 
            std::vector<std::wstring> hmmNames;
            std::vector<std::wstring> latticeNames;
            GetDataNamesFromConfig(readerConfig, featureNames, labelNames, hmmNames, latticeNames);
            if (featureNames.size() + labelNames.size() <= 1)
            {
                RuntimeError("network needs at least 1 input and 1 output specified!");
            }

            //load data for all real-valued inputs (features)
            foreach_index(i, featureNames)
            {
                ConfigParameters thisFeature = readerConfig(featureNames[i]);
                m_featDims.push_back(thisFeature("dim"));
                ConfigArray contextWindow = thisFeature("contextWindow", "1");
                if (contextWindow.size() == 1) // symmetric
                {
                    size_t windowFrames = contextWindow[0];
                    if (windowFrames % 2 == 0 )
                        RuntimeError("augmentationextent: neighbor expansion of input features to %d not symmetrical", windowFrames);
                    size_t context = windowFrames / 2;           // extend each side by this
                    numContextLeft.push_back(context);
                    numContextRight.push_back(context);

                }
                else if (contextWindow.size() == 2) // left context, right context
                {
                    numContextLeft.push_back(contextWindow[0]);
                    numContextRight.push_back(contextWindow[1]);
                }
                else
                {
                    RuntimeError("contextFrames must have 1 or 2 values specified, found %d", contextWindow.size());
                }
                // update m_featDims to reflect the total input dimension (featDim x contextWindow), not the native feature dimension
                // that is what the lower level feature readers expect
                m_featDims[i] = m_featDims[i] * (1 + numContextLeft[i] + numContextRight[i]); 

                string type = thisFeature("type","Real");
                if (type=="Real"){
                    m_nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
                }
                else{
                    RuntimeError("feature type must be Real");
                }

                m_featureNameToIdMap[featureNames[i]]= iFeat;
                scriptpaths.push_back(thisFeature("scpFile"));
                m_featureNameToDimMap[featureNames[i]] = m_featDims[i];

                m_featuresBufferMultiIO.push_back(nullptr);
                m_featuresBufferAllocatedMultiIO.push_back(0);

                iFeat++;            
            }

            foreach_index(i, labelNames)
            {
                ConfigParameters thisLabel = readerConfig(labelNames[i]);
                if (thisLabel.Exists("labelDim"))
                    m_labelDims.push_back(thisLabel("labelDim"));
                else if (thisLabel.Exists("dim"))
                    m_labelDims.push_back(thisLabel("dim"));
                else
                    RuntimeError("labels must specify dim or labelDim");

                string type;
                if (thisLabel.Exists("labelType"))
                    type = thisLabel("labelType"); // let's deprecate this eventually and just use "type"...
                else
                    type = thisLabel("type","Category"); // outputs should default to category

                if (type=="Category")
                    m_nameToTypeMap[labelNames[i]] = InputOutputTypes::category;
                else
                    RuntimeError("label type must be Category");

                statelistpaths.push_back(thisLabel("labelMappingFile",L""));

                m_labelNameToIdMap[labelNames[i]]=iLabel;
                m_labelNameToDimMap[labelNames[i]]=m_labelDims[i];
                mlfpaths.clear();
                if (thisLabel.ExistsCurrent("mlfFile"))
                {
                    mlfpaths.push_back(thisLabel("mlfFile"));
                }
                else
                {
                    if (!thisLabel.ExistsCurrent("mlfFileList"))
                    {
                        RuntimeError("Either mlfFile or mlfFileList must exist in HTKMLFReder");
                    }
                    wstring list = thisLabel("mlfFileList");
                    for (msra::files::textreader r(list); r;)
                    {
                        mlfpaths.push_back(r.wgetline());
                    }
                }
                mlfpathsmulti.push_back(mlfpaths);

                m_labelsBufferMultiIO.push_back(nullptr);
                m_labelsBufferAllocatedMultiIO.push_back(0);

                iLabel++;

                wstring labelToTargetMappingFile(thisLabel("labelToTargetMappingFile",L""));
                if (labelToTargetMappingFile != L"")
                {
                    std::vector<std::vector<ElemType>> labelToTargetMap;
                    m_convertLabelsToTargetsMultiIO.push_back(true);
                    if (thisLabel.Exists("targetDim"))
                    {
                        m_labelNameToDimMap[labelNames[i]]=m_labelDims[i]=thisLabel("targetDim");
                    }
                    else
                        RuntimeError("output must specify targetDim if labelToTargetMappingFile specified!");
                    size_t targetDim = ReadLabelToTargetMappingFile (labelToTargetMappingFile,statelistpaths[i], labelToTargetMap);    
                    if (targetDim!=m_labelDims[i])
                        RuntimeError("mismatch between targetDim and dim found in labelToTargetMappingFile");
                    m_labelToTargetMapMultiIO.push_back(labelToTargetMap);
                }
                else
                {
                    m_convertLabelsToTargetsMultiIO.push_back(false);
                    m_labelToTargetMapMultiIO.push_back(std::vector<std::vector<ElemType>>());
                }
            }

            //get lattice toc file names 
            std::pair<std::vector<wstring>, std::vector<wstring>> latticetocs;
            foreach_index(i, latticeNames)
                //only support one set of lattice now
            {
                ConfigParameters thislattice = readerConfig(latticeNames[i]);


                vector<wstring> paths;
                expand_wildcards(thislattice("denlatTocFile"), paths);
                latticetocs.second.insert(latticetocs.second.end(), paths.begin(), paths.end());

                if (thislattice.Exists("numlatTocFile"))
                {
                    paths.clear();
                    expand_wildcards(thislattice("numlatTocFile"), paths);
                    latticetocs.first.insert(latticetocs.first.end(), paths.begin(), paths.end());
                }

            }

            //get HMM related file names
            vector<wstring> cdphonetyingpaths, transPspaths;
            foreach_index(i, hmmNames)
            {
                ConfigParameters thishmm = readerConfig(hmmNames[i]);

                vector<wstring> paths;
                cdphonetyingpaths.push_back(thishmm("phoneFile"));
                transPspaths.push_back(thishmm("transpFile", L""));
            }

            // mmf files 
            //only support one set now
            if (cdphonetyingpaths.size() > 0 && statelistpaths.size() > 0 && transPspaths.size() > 0)
                m_hset.loadfromfile(cdphonetyingpaths[0], statelistpaths[0], transPspaths[0]);
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

            m_framemode = readerConfig("frameMode", "true");

            int verbosity = readerConfig("verbosity","2");

            // determine if we partial minibatches are desired
            std::string minibatchMode(readerConfig("minibatchMode","Partial"));
            m_partialMinibatch = !_stricmp(minibatchMode.c_str(),"Partial");

            // get the read method, defaults to "blockRandomize" other option is "rollingWindow"
            std::string readMethod(readerConfig("readMethod","blockRandomize"));

            if (readMethod == "blockRandomize" && randomize == randomizeNone)
            {
                fprintf(stderr, "WARNING: Randomize cannot be set to None when readMethod is set to blockRandomize. Change it Auto");
                randomize = randomizeAuto;
            }

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

                /* 
                   do "..." expansion if SCP uses relative path names
                   "..." in the SCP means full path is the same as the SCP file
                   for example, if scp file is "//aaa/bbb/ccc/ddd.scp"
                   and contains entry like 
                   .../file1.feat
                   .../file2.feat
                   etc.
                   the features will be read from
                //aaa/bbb/ccc/file1.feat
                //aaa/bbb/ccc/file2.feat
                etc. 
                This works well if you store the scp file with the features but 
                do not want different scp files everytime you move or create new features
                */
                wstring scpdircached;
                for (auto & entry : filelist)
                    ExpandDotDotDot(entry, scriptpath, scpdircached);

                infilesmulti.push_back(filelist);
            }

            if (readerConfig.Exists("unigram"))
                unigrampath = (wstring)readerConfig("unigram");

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
            //    statelistpath = readerConfig("statelist");

            double htktimetoframe = 100000.0;           // default is 10ms 
            //std::vector<msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>> labelsmulti;
            std::vector<std::map<std::wstring,std::vector<msra::asr::htkmlfentry>>> labelsmulti;
            //std::vector<std::wstring> pagepath;
            foreach_index(i, mlfpathsmulti)
            {
                const msra::lm::CSymbolSet* wordmap = unigram ? &unigramsymbols : NULL;
                msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>  
                labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], wordmap, (map<string,size_t>*) NULL, htktimetoframe);      // label MLF
                // get the temp file name for the page file
                labelsmulti.push_back(labels);
            }

            if (!_stricmp(readMethod.c_str(),"blockRandomize"))
            {
                // construct all the parameters we don't need, but need to be passed to the constructor...
                m_lattices.reset(new msra::dbn::latticesource(latticetocs, m_hset.getsymmap()));

                // now get the frame source. This has better randomization and doesn't create temp files
                m_frameSource.reset(new msra::dbn::minibatchutterancesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, numContextLeft, numContextRight, randomize, *m_lattices, m_latticeMap, m_framemode));
                m_frameSource->setverbosity(verbosity);
            }
            else if (!_stricmp(readMethod.c_str(),"rollingWindow"))
            {
#ifdef _WIN32
                std::wstring pageFilePath;
#else
                std::string pageFilePath;
#endif
                std::vector<std::wstring> pagePaths;
                if (readerConfig.Exists("pageFilePath"))
                {
                    pageFilePath = readerConfig("pageFilePath");

                    // replace any '/' with '\' for compat with default path
                    std::replace(pageFilePath.begin(), pageFilePath.end(), '/','\\'); 
#ifdef _WIN32               
                    // verify path exists
                    DWORD attrib = GetFileAttributes(pageFilePath.c_str());
                    if (attrib==INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY))
                        throw std::runtime_error ("pageFilePath does not exist");                
#endif
#ifdef __unix__
                struct stat statbuf;
                if (stat(pageFilePath.c_str(), &statbuf)==-1)
                {
                    throw std::runtime_error ("pageFilePath does not exist");
                }

#endif
            }
                else  // using default temporary path
                {
#ifdef _WIN32
                    pageFilePath.reserve(MAX_PATH);
                    GetTempPath(MAX_PATH, &pageFilePath[0]);
#endif
#ifdef __unix__
                pageFilePath.reserve(PATH_MAX);
                pageFilePath = "/tmp/temp.CNTK.XXXXXX";
#endif
                }

#ifdef _WIN32
                if (pageFilePath.size()>MAX_PATH-14) // max length of input to GetTempFileName is MAX_PATH-14
                    throw std::runtime_error (msra::strfun::strprintf ("pageFilePath must be less than %d characters", MAX_PATH-14));
#endif
#ifdef __unix__
            if (pageFilePath.size()>PATH_MAX-14) // max length of input to GetTempFileName is PATH_MAX-14
                throw std::runtime_error (msra::strfun::strprintf ("pageFilePath must be less than %d characters", PATH_MAX-14));       
#endif
                foreach_index(i, infilesmulti)
                {
#ifdef _WIN32
                    wchar_t tempFile[MAX_PATH];
                    GetTempFileName(pageFilePath.c_str(), L"CNTK", 0, tempFile);
                    pagePaths.push_back(tempFile);
#endif
#ifdef __unix__
                char* tempFile;
                //GetTempFileName(pageFilePath.c_str(), L"CNTK", 0, tempFile);
                tempFile = (char*) pageFilePath.c_str();
                int fid = mkstemp(tempFile);
                unlink (tempFile);
                close (fid);
                pagePaths.push_back(GetWC(tempFile));
#endif
                }

                const bool mayhavenoframe=false;
                int addEnergy = 0;

                m_frameSource.reset(new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, numContextLeft, numContextRight, randomize, pagePaths, mayhavenoframe, addEnergy));
                m_frameSource->setverbosity(verbosity);
            }
            else
            {
                RuntimeError("readMethod must be rollingWindow or blockRandomize");
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
            vector<size_t> numContextLeft;
            vector<size_t> numContextRight;

            std::vector<std::wstring> featureNames;
            std::vector<std::wstring> labelNames;
            //lattice and hmm
            std::vector<std::wstring> hmmNames;
            std::vector<std::wstring> latticeNames;

            GetDataNamesFromConfig(readerConfig, featureNames, labelNames, hmmNames, latticeNames);

            foreach_index(i, featureNames)
            {
                ConfigParameters thisFeature = readerConfig(featureNames[i]);
                realDims.push_back(thisFeature("dim"));

                ConfigArray contextWindow = thisFeature("contextWindow", "1");
                if (contextWindow.size() == 1) // symmetric
                {
                    size_t windowFrames = contextWindow[0];
                    if (windowFrames % 2 == 0)
                        RuntimeError("augmentationextent: neighbor expansion of input features to %d not symmetrical", windowFrames);
                    size_t context = windowFrames / 2;           // extend each side by this
                    numContextLeft.push_back(context);
                    numContextRight.push_back(context);

                }
                else if (contextWindow.size() == 2) // left context, right context
                {
                    numContextLeft.push_back(contextWindow[0]);
                    numContextRight.push_back(contextWindow[1]);
                }
                else
                {
                    RuntimeError("contextFrames must have 1 or 2 values specified, found %d", contextWindow.size());
                }
                // update m_featDims to reflect the total input dimension (featDim x contextWindow), not the native feature dimension
                // that is what the lower level feature readers expect
                realDims[i] = realDims[i] * (1 + numContextLeft[i] + numContextRight[i]);

                string type = thisFeature("type","Real");
                if (type=="Real"){
                    m_nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
                }
                else{
                    RuntimeError("feature type must be Real");
                }

                m_featureNameToIdMap[featureNames[i]]= iFeat;
                scriptpaths.push_back(thisFeature("scpFile"));
                m_featureNameToDimMap[featureNames[i]] = realDims[i];

                m_featuresBufferMultiIO.push_back(nullptr);
                m_featuresBufferAllocatedMultiIO.push_back(0);
                iFeat++;
            }

            if (labelNames.size()>0)
                RuntimeError("writer mode does not support labels as inputs, only features");

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

                fprintf (stderr, " %d entries\n", (int)n);

                if (i==0)
                    numFiles=n;
                else
                    if (n!=numFiles)
                        throw std::runtime_error (msra::strfun::strprintf ("HTKMLFReader::InitEvalReader: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

                m_inputFilesMultiIO.push_back(filelist);
            }

            m_fileEvalSource.reset(new msra::dbn::FileEvalSource(realDims, numContextLeft, numContextRight, evalchunksize));
        }



    // destructor - virtual so it gets called properly 
    template<class ElemType>
        HTKMLFReader<ElemType>::~HTKMLFReader()
        {
            if (!m_featuresBufferMultiIO.empty())
            {
                if (m_featuresBufferMultiIO[0] != nullptr)
                {
                    foreach_index(i, m_featuresBufferMultiIO)
                    {
                        m_featuresBufferMultiIO[i] = nullptr;
                    }
                }
            }
            if (!m_labelsBufferMultiIO.empty())
            {
                if (m_labelsBufferMultiIO[0] != nullptr)
                {
                    foreach_index(i, m_labelsBufferMultiIO)
                    {
                        m_labelsBufferMultiIO[i] = nullptr;
                    }
                }
            }
            if (/*m_numberOfuttsPerMinibatch > 1 && */m_truncated)
            {
                for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i ++)
                {
                    m_featuresBufferMultiUtt[i] = nullptr;
                    m_labelsBufferMultiUtt[i] = nullptr;
                    m_latticeBufferMultiUtt[i].reset();
                }
            }
        }

    //StartMinibatchLoop - Startup a minibatch loop 
    // mbSize - [in] size of the minibatch (number of frames, etc.)
    // epoch - [in] epoch number for this loop
    // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
    template<class ElemType>
        void HTKMLFReader<ElemType>::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/)
        {
            assert(subsetNum < numSubsets);
            assert(((subsetNum == 0) && (numSubsets == 1)) || this->SupportsDistributedMBRead());

            m_mbSize = mbSize;

            m_numberOfuttsPerMinibatch = m_numberOfuttsPerMinibatchForAllEpochs[epoch];

            m_actualnumberOfuttsPerMinibatch = m_numberOfuttsPerMinibatch;
            m_sentenceEnd.assign(m_numberOfuttsPerMinibatch, true);
            m_processedFrame.assign(m_numberOfuttsPerMinibatch, 0);
            m_toProcess.assign(m_numberOfuttsPerMinibatch, 0);
            m_switchFrame.assign(m_numberOfuttsPerMinibatch, 0);

            m_validFrame.assign(m_numberOfuttsPerMinibatch, 0);
            if (m_trainOrTest)
            {
                // For distributed reading under truncated BPTT of LSTMs, we distribute the utterances per minibatch among all the subsets
                if (m_truncated)
                {
                    if ((numSubsets > 1) && (m_numberOfuttsPerMinibatch < numSubsets))
                    {
                        LogicError("Insufficient value of 'nbruttsineachrecurrentiter'=%d for distributed reading with %d subsets", m_numberOfuttsPerMinibatch, numSubsets);
                    }

                    m_numberOfuttsPerMinibatch = (m_numberOfuttsPerMinibatch / numSubsets) + ((subsetNum < (m_numberOfuttsPerMinibatch % numSubsets)) ? 1 : 0);
                }

                StartMinibatchLoopToTrainOrTest(mbSize, epoch, subsetNum, numSubsets, requestedEpochSamples);
            }
            else
            {
                // No distributed reading of mini-batches for write
                if ((subsetNum != 0) || (numSubsets != 1))
                {
                    LogicError("Distributed reading of mini-batches is only supported for training or testing");
                }

                StartMinibatchLoopToWrite(mbSize,epoch,requestedEpochSamples);    
            }
            m_checkDictionaryKeys=true;
        }

    template<class ElemType>
        void HTKMLFReader<ElemType>::StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples)
        {
            size_t datapasses=1;
            size_t totalFrames = m_frameSource->totalframes();

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

            m_mbiter.reset(new msra::dbn::minibatchiterator(*m_frameSource, epoch, requestedEpochSamples, mbSize, subsetNum, numSubsets, datapasses));
            if (!m_featuresBufferMultiIO.empty())
            {
                if (m_featuresBufferMultiIO[0] != nullptr) // check first feature, if it isn't NULL, safe to assume all are not NULL? 
                {
                    foreach_index(i, m_featuresBufferMultiIO)
                    {
                        m_featuresBufferMultiIO[i] = nullptr;
                        m_featuresBufferAllocatedMultiIO[i]=0;
                    }
                }
            }
            if (!m_labelsBufferMultiIO.empty())
            {
                if (m_labelsBufferMultiIO[0] != nullptr)
                {
                    foreach_index(i, m_labelsBufferMultiIO)
                    {
                        m_labelsBufferMultiIO[i] = nullptr;
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
                        m_featuresBufferMultiUtt[u] = NULL;
                        m_featuresBufferAllocatedMultiUtt[u] = 0;
                    }
                    if (m_labelsBufferMultiUtt[u] != NULL)
                    {
                        m_labelsBufferMultiUtt[u] = NULL;
                        m_labelsBufferAllocatedMultiUtt[u] = 0;
                    }
                    if (m_latticeBufferMultiUtt[u] != NULL)
                    {
                        m_latticeBufferMultiUtt[u].reset();
                    }
                    ReNewBufferForMultiIO(u);
                }    
            }
        }

    template<class ElemType>
        void HTKMLFReader<ElemType>::StartMinibatchLoopToWrite(size_t mbSize, size_t /*epoch*/, size_t /*requestedEpochSamples*/)
        {
            m_fileEvalSource->Reset();
            m_fileEvalSource->SetMinibatchSize(mbSize);
            m_inputFileIndex=0;

            if (m_featuresBufferMultiIO[0] != nullptr) // check first feature, if it isn't NULL, safe to assume all are not NULL? 
            {
                foreach_index(i, m_featuresBufferMultiIO)
                {
                    m_featuresBufferMultiIO[i] = nullptr;
                    m_featuresBufferAllocatedMultiIO[i]=0;
                }
            }
        }

        template<class ElemType>
        bool HTKMLFReader<ElemType>::GetMinibatch4SE(std::vector<shared_ptr< const msra::dbn::latticesource::latticepair>> & latticeinput, 
            vector<size_t> &uids, vector<size_t> &boundaries, vector<size_t> &extrauttmap)
        {
            if (m_trainOrTest)
            {
                return GetMinibatch4SEToTrainOrTest(latticeinput, uids, boundaries, extrauttmap);
            }
            else
            {
                return true;
            }
        }
        template<class ElemType>
        bool HTKMLFReader<ElemType>::GetMinibatch4SEToTrainOrTest(std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> & latticeinput, 
            std::vector<size_t> &uids, std::vector<size_t> &boundaries, std::vector<size_t> &extrauttmap)
        {
            latticeinput.clear();
            uids.clear();
            boundaries.clear();
            extrauttmap.clear();
            if (m_truncated == false)
            {
                if (*m_mbiter)
                {
                    //uids.clear();
                    uids = m_mbiter->labels();
                    boundaries = m_mbiter->bounds();
                    //if (m_mbiter->haslattice())
                    size_t nummblattice = m_mbiter->currentmblattices();
                    

                    for (int j = 0; j < nummblattice; j++) // column major, so iterate columns in outside loop
                    {
                        latticeinput.push_back(m_mbiter->lattice(j));
                    }
                }
            }
            else
            {
                
                for (size_t i = 0; i < m_extraUttsPerMinibatch.size(); i++)
                {
                    latticeinput.push_back(m_extraLatticeBufferMultiUtt[i]);
                    uids.insert(uids.end(), m_extraLabelsIDBufferMultiUtt[i].begin(), m_extraLabelsIDBufferMultiUtt[i].end());
                    boundaries.insert(boundaries.end(), m_extraPhoneboundaryIDBufferMultiUtt[i].begin(), m_extraPhoneboundaryIDBufferMultiUtt[i].end());
                }
                
                extrauttmap.insert(extrauttmap.end(), m_extraUttsPerMinibatch.begin(), m_extraUttsPerMinibatch.end());
            }
            return true;
        }

        template<class ElemType>
        bool HTKMLFReader<ElemType>::GetHmmData(msra::asr::simplesenonehmm * hmm)
        {
            *hmm = m_hset;
            return true;
        }
    // GetMinibatch - Get the next minibatch (features and labels)
    // matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
    //             [out] each matrix resized if necessary containing data. 
    // returns - true if there are more minibatches, false if no more minibatchs remain
    template<class ElemType>
        bool HTKMLFReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
        {
            if (m_trainOrTest)
            {
                return GetMinibatchToTrainOrTest(matrices);
            }
            else
            {
                return GetMinibatchToWrite(matrices);
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
            if     (m_checkDictionaryKeys)
            {
                for (auto iter=matrices.begin();iter!=matrices.end();iter++)
                {
                    if (m_nameToTypeMap.find(iter->first)==m_nameToTypeMap.end())
                        RuntimeError("minibatch requested for input node %ls not found in reader - cannot generate input\n", iter->first.c_str());

                }
                m_checkDictionaryKeys=false;
            }

            do 
            {
                if (!m_truncated)       // frame mode or whole utterances
                {
                    if (!(*m_mbiter))   // hit the end
                        return false;

                    // now, access all features and and labels by iterating over map of "matrices"
                    bool first = true;
                    typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
                    for (iter = matrices.begin();iter!=matrices.end(); iter++)
                    {
                        // dereference matrix that corresponds to key (input/output name) and 
                        // populate based on whether its a feature or a label
                        Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

                        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)     // --- read features
                        {

                            id = m_featureNameToIdMap[iter->first];
                            dim = m_featureNameToDimMap[iter->first];
                            const msra::dbn::matrixstripe feat = m_mbiter->frames(id);
                            const size_t actualmbsize = feat.cols();   // it may still return less if at end of sweep TODO: this check probably only needs to happen once
                            if (first)  // initialize MBLayout
                            {
                                // entire minibatch is one utterance
                                m_pMBLayout->Init(1, actualmbsize, !m_framemode);
                                if (m_pMBLayout->RequireSentenceSeg())       // in framemode we leave the flags empty
                                {
                                    m_pMBLayout->Set(0, 0, MinibatchPackingFlags::SequenceStart);
                                    m_pMBLayout->SetWithoutOr(0, actualmbsize - 1, MinibatchPackingFlags::SequenceEnd);  // BUGBUG: using SetWithoutOr() because original code did; but that seems inconsistent
                                }
                                first = false;
                            }
                            else
                                if (m_pMBLayout->GetNumTimeSteps() != actualmbsize)
                                    RuntimeError("GetMinibatch: Multiple feature streams with inconsistent minibatch size detected, e.g. %d vs. %d.", (int)m_pMBLayout->GetNumTimeSteps(), (int)actualmbsize);

                            assert (actualmbsize == m_mbiter->currentmbframes());
                            skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSource->totalframes() > actualmbsize);

                            // check to see if we got the number of frames we requested
                            if (!skip)
                            {
                                assert(feat.rows()==dim); // check feature dimension matches what's expected

                                if ((m_featuresBufferMultiIO[id] == nullptr) ||
                                    (m_featuresBufferAllocatedMultiIO[id] < (feat.rows() * feat.cols())) /*buffer size changed. can be partial minibatch*/)
                                {
                                    m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), feat.rows() * feat.cols());
                                    m_featuresBufferAllocatedMultiIO[id] = feat.rows() * feat.cols();
                                }

                                // copy the features over to our array type
                                if (sizeof(ElemType) == sizeof(float))
                                {
                                    for (int j=0; j < feat.cols(); j++) // column major, so iterate columns
                                    {
                                        // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                                        memcpy_s(&m_featuresBufferMultiIO[id].get()[j * feat.rows()], sizeof(ElemType) * feat.rows(), &feat(0, j), sizeof(ElemType) * feat.rows());
                                    }
                                }
                                else
                                {
                                    for (int j=0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
                                    {
                                        for (int i = 0; i < feat.rows(); i++)
                                        {
                                            m_featuresBufferMultiIO[id].get()[j * feat.rows() + i] = feat(i, j);
                                        }
                                    }
                                }
                                data.SetValue(feat.rows(), feat.cols(), m_featuresBufferMultiIO[id].get(), matrixFlagNormal);
                            }
                        }
                        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)    // --- read labels
                        {
                            id = m_labelNameToIdMap[iter->first];
                            dim = m_labelNameToDimMap[iter->first];
                            const vector<size_t> & uids = m_mbiter->labels(id);

                            // need skip logic here too in case labels are first in map not features
                            const size_t actualmbsize = uids.size();   // it may still return less if at end of sweep TODO: this check probably only needs to happen once
                            assert (actualmbsize == m_mbiter->currentmbframes());
                            skip = (!m_partialMinibatch && m_mbiter->requestedframes() != actualmbsize && m_frameSource->totalframes() > actualmbsize);

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

                                if ((m_labelsBufferMultiIO[id] == nullptr) ||
                                    (m_labelsBufferAllocatedMultiIO[id] < (dim * uids.size())))
                                {
                                    m_labelsBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * uids.size());
                                    m_labelsBufferAllocatedMultiIO[id] = dim * uids.size();
                                }
                                memset(m_labelsBufferMultiIO[id].get(), 0, sizeof(ElemType) * dim * uids.size());

                                if (m_convertLabelsToTargetsMultiIO[id])
                                {
                                    size_t labelDim = m_labelToTargetMapMultiIO[id].size();
                                    for (int i = 0; i < uids.size(); i++)
                                    {
                                        assert(uids[i] < labelDim); labelDim;
                                        size_t labelId = uids[i];
                                        for (int j = 0; j < dim; j++)
                                        {
                                            m_labelsBufferMultiIO[id].get()[i * dim + j] = m_labelToTargetMapMultiIO[id][labelId][j];
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
                                        m_labelsBufferMultiIO[id].get()[i * dim + uids[i]] = (ElemType)1;
                                    }
                                }


                                data.SetValue(dim, uids.size(), m_labelsBufferMultiIO[id].get(), matrixFlagNormal);
                            }
                        }
                        else    //default:
                            RuntimeError("GetMinibatch: unknown InputOutputType for %ls", (iter->first).c_str());

                    }
                    // advance to the next minibatch
                    (*m_mbiter)++;
                }
                else if (m_truncated && !m_fullutt)
                {
                    // In truncated BPTT mode, a minibatch only consists of the truncation length, e.g. 20 frames.
                    // The reader maintains a set of current utterances, and each next minibatch contains the next 20 frames.
                    // When the end of an utterance is reached, the next available utterance is begin in the same slot.
                    if (m_noData)
                    {
                        bool endEpoch = true;
                        for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
                        {
                            if (m_processedFrame[i] != m_toProcess[i])
                                endEpoch = false;
                        }
                        if(endEpoch)
                            return false;
                    }
                    size_t numOfFea = m_featuresBufferMultiIO.size();
                    size_t numOfLabel = m_labelsBufferMultiIO.size();

                    m_pMBLayout->Init(m_numberOfuttsPerMinibatch, m_mbSize, true/*sequential*/);

                    vector<size_t> actualmbsize(m_numberOfuttsPerMinibatch,0);
                    for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
                    {
                        size_t startFr = m_processedFrame[i];
                        size_t endFr = 0;
                        if ((m_processedFrame[i] + m_mbSize) < m_toProcess[i])
                        {
                            if (m_processedFrame[i] > 0)
                            {
                                m_sentenceEnd[i] = false;
                                m_switchFrame[i] = m_mbSize+1;
                                if (m_processedFrame[i] == 1)
                                    m_pMBLayout->SetWithoutOr(i, 0, MinibatchPackingFlags::SequenceEnd);   // TODO: shouldn't both Start and End be set? TODO: can we just use Set()?
                            }
                            else
                            {
                                m_sentenceEnd[i] = true;
                                m_switchFrame[i] = 0;
                                m_pMBLayout->SetWithoutOr(i, 0, MinibatchPackingFlags::SequenceStart);
                            }
                            actualmbsize[i] = m_mbSize;
                            endFr = startFr + actualmbsize[i];
                            typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
                            for (iter = matrices.begin();iter!=matrices.end(); iter++)
                            {
                                // dereference matrix that corresponds to key (input/output name) and 
                                // populate based on whether its a feature or a label
                                Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

                                if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                                {
                                    id = m_featureNameToIdMap[iter->first];
                                    dim = m_featureNameToDimMap[iter->first];

                                    if ((m_featuresBufferMultiIO[id] == nullptr) ||
                                        (m_featuresBufferAllocatedMultiIO[id] < (dim * m_mbSize * m_numberOfuttsPerMinibatch)) /*buffer size changed. can be partial minibatch*/)
                                    {
                                        m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbSize * m_numberOfuttsPerMinibatch);
                                        m_featuresBufferAllocatedMultiIO[id] = dim * m_mbSize * m_numberOfuttsPerMinibatch;
                                    }

                                    if (sizeof(ElemType) == sizeof(float))
                                    {
                                        for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
                                        {
                                            // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                                            memcpy_s(&m_featuresBufferMultiIO[id].get()[(k * m_numberOfuttsPerMinibatch + i) * dim], sizeof(ElemType) * dim, &m_featuresBufferMultiUtt[i].get()[j * dim + m_featuresStartIndexMultiUtt[id + i * numOfFea]], sizeof(ElemType) * dim);
                                        }
                                    }
                                    else
                                    {
                                        for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
                                        {
                                            for (int d = 0; d < dim; d++)
                                                m_featuresBufferMultiIO[id].get()[(k * m_numberOfuttsPerMinibatch + i) * dim + d] = m_featuresBufferMultiUtt[i].get()[j * dim + d + m_featuresStartIndexMultiUtt[id + i * numOfFea]];
                                        }
                                    }
                                }
                                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                                {
                                    id = m_labelNameToIdMap[iter->first];
                                    dim = m_labelNameToDimMap[iter->first];
                                    if ((m_labelsBufferMultiIO[id] == nullptr) ||
                                        (m_labelsBufferAllocatedMultiIO[id] < (dim * m_mbSize * m_numberOfuttsPerMinibatch)))
                                    {
                                        m_labelsBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbSize * m_numberOfuttsPerMinibatch);
                                        m_labelsBufferAllocatedMultiIO[id] = dim * m_mbSize * m_numberOfuttsPerMinibatch;
                                    }

                                    for (size_t j = startFr,k=0; j < endFr; j++,k++)
                                    {
                                        for (int d = 0; d < dim; d++)
                                            m_labelsBufferMultiIO[id].get()[(k * m_numberOfuttsPerMinibatch + i) * dim + d] = m_labelsBufferMultiUtt[i].get()[j * dim + d + m_labelsStartIndexMultiUtt[id + i * numOfLabel]];
                                    }
                                }
                            }
                            m_processedFrame[i] += m_mbSize;
                        }
                        else
                        {
                            actualmbsize[i] = m_toProcess[i] - m_processedFrame[i];
                            endFr = startFr + actualmbsize[i];

                            typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
                            for (iter = matrices.begin();iter!=matrices.end(); iter++)
                            {
                                // dereference matrix that corresponds to key (input/output name) and 
                                // populate based on whether its a feature or a label
                                Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

                                if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                                {
                                    id = m_featureNameToIdMap[iter->first];
                                    dim = m_featureNameToDimMap[iter->first];

                                    if ((m_featuresBufferMultiIO[id] == nullptr) ||
                                        (m_featuresBufferAllocatedMultiIO[id] < (dim * m_mbSize * m_numberOfuttsPerMinibatch)) /*buffer size changed. can be partial minibatch*/)
                                    {
                                        m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbSize * m_numberOfuttsPerMinibatch);
                                        m_featuresBufferAllocatedMultiIO[id] = dim * m_mbSize * m_numberOfuttsPerMinibatch;
                                    }

                                    if (sizeof(ElemType) == sizeof(float))
                                    {
                                        for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
                                        {
                                            // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                                            memcpy_s(&m_featuresBufferMultiIO[id].get()[(k * m_numberOfuttsPerMinibatch + i) * dim], sizeof(ElemType) * dim, &m_featuresBufferMultiUtt[i].get()[j * dim + m_featuresStartIndexMultiUtt[id + i * numOfFea]], sizeof(ElemType) * dim);
                                        }
                                    }
                                    else
                                    {
                                        for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
                                        {
                                            for (int d = 0; d < dim; d++)
                                            {
                                                m_featuresBufferMultiIO[id].get()[(k * m_numberOfuttsPerMinibatch + i) * dim + d] = m_featuresBufferMultiUtt[i].get()[j * dim + d + m_featuresStartIndexMultiUtt[id + i * numOfFea]];
                                            }
                                        }
                                    }
                                }
                                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                                {
                                    id = m_labelNameToIdMap[iter->first];
                                    dim = m_labelNameToDimMap[iter->first];
                                    if ((m_labelsBufferMultiIO[id] == nullptr) ||
                                        (m_labelsBufferAllocatedMultiIO[id] < (dim * m_mbSize * m_numberOfuttsPerMinibatch)))
                                    {
                                        m_labelsBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbSize * m_numberOfuttsPerMinibatch);
                                        m_labelsBufferAllocatedMultiIO[id] = dim * m_mbSize * m_numberOfuttsPerMinibatch;
                                    }
                                    for (size_t j = startFr,k=0; j < endFr; j++,k++)
                                    {
                                        for (int d = 0; d < dim; d++)
                                            m_labelsBufferMultiIO[id].get()[(k * m_numberOfuttsPerMinibatch + i) * dim + d] = m_labelsBufferMultiUtt[i].get()[j * dim + d + m_labelsStartIndexMultiUtt[id + i * numOfLabel]];
                                    }
                                }
                            }
                            m_processedFrame[i] += (endFr-startFr);
                            m_switchFrame[i] = actualmbsize[i];
                            if (actualmbsize[i] < m_mbSize)
                                m_pMBLayout->Set(i, actualmbsize[i], MinibatchPackingFlags::SequenceStart); // NOTE: this ORs, while original code overwrote in matrix but ORed into vector
                            if (actualmbsize[i] == m_mbSize)
                                m_pMBLayout->Set(i, actualmbsize[i] - 1, MinibatchPackingFlags::SequenceEnd); // NOTE: this ORs, while original code overwrote in matrix but ORed into vector
                            startFr = m_switchFrame[i];
                            endFr = m_mbSize;
                            bool reNewSucc = ReNewBufferForMultiIO(i);
                            for (iter = matrices.begin();iter!=matrices.end(); iter++)
                            {
                                // dereference matrix that corresponds to key (input/output name) and 
                                // populate based on whether its a feature or a label
                                //Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

                                if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                                {
                                    id = m_featureNameToIdMap[iter->first];
                                    dim = m_featureNameToDimMap[iter->first];
                                    if (sizeof(ElemType) == sizeof(float))
                                    {
                                        for (size_t j = startFr,k = 0; j < endFr; j++,k++) // column major, so iterate columns
                                        {
                                            // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                                            memcpy_s(&m_featuresBufferMultiIO[id].get()[(j * m_numberOfuttsPerMinibatch + i) * dim], sizeof(ElemType) * dim, &m_featuresBufferMultiUtt[i].get()[k * dim + m_featuresStartIndexMultiUtt[id + i * numOfFea]], sizeof(ElemType) * dim);
                                        }
                                    }
                                    else
                                    {
                                        for (size_t j=startFr,k=0; j < endFr; j++,k++) // column major, so iterate columns in outside loop
                                        {
                                            for (int d = 0; d < dim; d++)
                                                m_featuresBufferMultiIO[id].get()[(j * m_numberOfuttsPerMinibatch + i) * dim + d] = m_featuresBufferMultiUtt[i].get()[k * dim + d + m_featuresStartIndexMultiUtt[id + i * numOfFea]];
                                        }
                                    }
                                }
                                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                                {
                                    id = m_labelNameToIdMap[iter->first];
                                    dim = m_labelNameToDimMap[iter->first];
                                    for (size_t j = startFr,k=0; j < endFr; j++,k++)
                                    {
                                        for (int d = 0; d < dim; d++)
                                            m_labelsBufferMultiIO[id].get()[(j * m_numberOfuttsPerMinibatch + i) * dim + d] = m_labelsBufferMultiUtt[i].get()[k * dim + d + m_labelsStartIndexMultiUtt[id + i * numOfLabel]];
                                    }
                                }
                            }

                            if (reNewSucc) m_processedFrame[i] += (endFr-startFr);

                        }
                    }
                    for (auto iter = matrices.begin();iter!=matrices.end(); iter++)
                    {
                        // dereference matrix that corresponds to key (input/output name) and 
                        // populate based on whether its a feature or a label
                        Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels
                        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                        {
                            id = m_featureNameToIdMap[iter->first];
                            dim = m_featureNameToDimMap[iter->first];
                            data.SetValue(dim, m_mbSize*m_numberOfuttsPerMinibatch, m_featuresBufferMultiIO[id].get(), matrixFlagNormal);
                        }
                        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                        {
                            id = m_labelNameToIdMap[iter->first];
                            dim = m_labelNameToDimMap[iter->first];
                            data.SetValue(dim, m_mbSize*m_numberOfuttsPerMinibatch, m_labelsBufferMultiIO[id].get(), matrixFlagNormal);
                        }
                    }
                    skip = false;
                }
                else if (m_truncated && m_fullutt)  //truncated = true, fullutt = true
                {
                    m_extraLatticeBufferMultiUtt.clear();
                    m_extraLabelsIDBufferMultiUtt.clear();
                    m_extraPhoneboundaryIDBufferMultiUtt.clear();
                    m_extraUttsPerMinibatch.clear();
                    if (m_noData && m_toProcess[0] == 0)    //no data left for the first channel of this minibatch, 
                    {
                        return false;
                    }
                    
                    vector<size_t> actualmbsize;
                    actualmbsize.assign(m_numberOfuttsPerMinibatch, 0);
                    //decide the m_mbSize 
                    m_mbSize = m_toProcess[0];
                    for (size_t i = 1; i < m_numberOfuttsPerMinibatch; i++)
                    {
                        if (m_mbSize < m_toProcess[i])
                            m_mbSize = m_toProcess[i];
                    }

                    //bool finishExtraFill = false;
                    m_pMBLayout->Init(m_numberOfuttsPerMinibatch, m_mbSize, true/*sequential*/);
                    for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
                    {                        
                        m_validFrame[i] = m_toProcess[i];
                        m_pMBLayout->Set(i, 0, MinibatchPackingFlags::SequenceStart);                        
                        m_pMBLayout->Set(i, m_validFrame[i] - 1, MinibatchPackingFlags::SequenceEnd);
                                
                        m_extraUttsPerMinibatch.push_back(i);
                        fillOneUttDataforParallelmode(matrices, 0, m_validFrame[i], i, i);
                        if (m_mbiter->haslattice())
                        {
                            m_extraLatticeBufferMultiUtt.push_back(m_latticeBufferMultiUtt[i]);
                            m_extraLabelsIDBufferMultiUtt.push_back(m_labelsIDBufferMultiUtt[i]);
                            m_extraPhoneboundaryIDBufferMultiUtt.push_back(m_phoneboundaryIDBufferMultiUtt[i]);                            
                        }
                        ReNewBufferForMultiIO(i);                                                                    
                    }

                    //insert extra utt to the channel with space
                    
                    

                    size_t nextMinibatchUttnum = 0;
                    bool inserted;
                    m_extraUttNum = 0;
                    for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
                    {
                        while (nextMinibatchUttnum <= i)
                        {
                            size_t framenum = m_toProcess[i];
                            inserted = false;
                            if (framenum > 0)
                            {
                            for (size_t j = 0; j < m_numberOfuttsPerMinibatch; j++)
                            {
                                if (framenum + m_validFrame[j] < m_mbSize)
                                {
                                    m_extraUttsPerMinibatch.push_back(j);
                                    if (m_mbiter->haslattice())
                                    {
                                        
                                            m_extraLatticeBufferMultiUtt.push_back(m_latticeBufferMultiUtt[i]);
                                            m_extraLabelsIDBufferMultiUtt.push_back(m_labelsIDBufferMultiUtt[i]);
                                            m_extraPhoneboundaryIDBufferMultiUtt.push_back(m_phoneboundaryIDBufferMultiUtt[i]);
                                    }
                                    fillOneUttDataforParallelmode(matrices, m_validFrame[j], framenum, j, i);
                                    m_pMBLayout->Set(j, m_validFrame[j], MinibatchPackingFlags::SequenceStart);
                                    m_pMBLayout->Set(j, m_validFrame[j] + framenum - 1, MinibatchPackingFlags::SequenceEnd);
                                    

                                    
                                    ReNewBufferForMultiIO(i);                                    
                                    m_validFrame[j] += framenum;
                                    m_extraUttNum++;
                                    inserted = true;
                                    break;
                                    }
                                }
                            }
                            if (!inserted)
                            {                                
                                nextMinibatchUttnum++;
                            }
                        }
                    }

                    for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
                    {
                        for (size_t t = m_validFrame[i] ; t < m_mbSize; t++)
                        {
                            m_pMBLayout->Set(i, t, MinibatchPackingFlags::NoInput);                            
                        }
                    }
                    typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
                    for (iter = matrices.begin(); iter != matrices.end(); iter++)
                    {
                        // dereference matrix that corresponds to key (input/output name) and 
                        // populate based on whether its a feature or a label
                        Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels
                        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                        {
                            id = m_featureNameToIdMap[iter->first];
                            dim = m_featureNameToDimMap[iter->first];
                            data.SetValue(dim, m_mbSize*m_numberOfuttsPerMinibatch, m_featuresBufferMultiIO[id].get(), matrixFlagNormal);
                        }
                        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                        {
                            id = m_labelNameToIdMap[iter->first];
                            dim = m_labelNameToDimMap[iter->first];
                            data.SetValue(dim, m_mbSize*m_numberOfuttsPerMinibatch, m_labelsBufferMultiIO[id].get(), matrixFlagNormal);
                        }


                    }
                    skip = false;
                }
            }   // keep going if we didn't get the right size minibatch
            while(skip);

            return true;
        }

    template<class ElemType>
        void HTKMLFReader<ElemType>::fillOneUttDataforParallelmode(std::map<std::wstring, Matrix<ElemType>*>& matrices, size_t startFr, size_t framenum, size_t channelIndex, 
            size_t sourceChannelIndex)
        {
            size_t id;
            size_t dim;
            size_t numOfFea = m_featuresBufferMultiIO.size();
            size_t numOfLabel = m_labelsBufferMultiIO.size();

            typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
            for (iter = matrices.begin(); iter != matrices.end(); iter++)
            {
                // dereference matrix that corresponds to key (input/output name) and 
                // populate based on whether its a feature or a label
                Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

                if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                {
                    id = m_featureNameToIdMap[iter->first];
                    dim = m_featureNameToDimMap[iter->first];

                    if (m_featuresBufferMultiIO[id] == nullptr || m_featuresBufferAllocatedMultiIO[id] < dim*m_mbSize*m_numberOfuttsPerMinibatch)
                    {
                        m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim*m_mbSize*m_numberOfuttsPerMinibatch);
                        memset(m_featuresBufferMultiIO[id].get(), 0, sizeof(ElemType)*dim*m_mbSize*m_numberOfuttsPerMinibatch);
                        m_featuresBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
                    }
                    /*else if () //buffer size changed. can be partial minibatch
                    {
                        delete[] m_featuresBufferMultiIO[id];
                        m_featuresBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
                        memset(m_featuresBufferMultiIO[id].get(), 0, sizeof(ElemType)*dim*m_mbSize*m_numberOfuttsPerMinibatch);
                        m_featuresBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
                    }*/



                    if (sizeof(ElemType) == sizeof(float))
                    {
                        for (size_t j = 0, k = startFr; j < framenum; j++, k++) // column major, so iterate columns
                        {
                            // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                            memcpy_s(&m_featuresBufferMultiIO[id].get()[(k*m_numberOfuttsPerMinibatch + channelIndex)*dim], sizeof(ElemType)*dim, &m_featuresBufferMultiUtt[sourceChannelIndex].get()[j*dim + m_featuresStartIndexMultiUtt[id + sourceChannelIndex*numOfFea]], sizeof(ElemType)*dim);
                        }
                    }
                    else
                    {
                        for (size_t j = 0, k = startFr; j < framenum; j++, k++) // column major, so iterate columns in outside loop
                        {
                            for (int d = 0; d < dim; d++)
                            {
                                m_featuresBufferMultiIO[id].get()[(k*m_numberOfuttsPerMinibatch + channelIndex)*dim + d] = m_featuresBufferMultiUtt[sourceChannelIndex].get()[j*dim + d + m_featuresStartIndexMultiUtt[id + sourceChannelIndex*numOfFea]];
                            }
                        }
                    }
                }
                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                {
                    id = m_labelNameToIdMap[iter->first];
                    dim = m_labelNameToDimMap[iter->first];
                    if (m_labelsBufferMultiIO[id] == nullptr || m_labelsBufferAllocatedMultiIO[id] < dim*m_mbSize*m_numberOfuttsPerMinibatch)
                    {
                        m_labelsBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim*m_mbSize*m_numberOfuttsPerMinibatch);
                        memset(m_labelsBufferMultiIO[id].get(), 0, sizeof(ElemType)*dim*m_mbSize*m_numberOfuttsPerMinibatch);
                        m_labelsBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
                    }
                    /*else if (m_labelsBufferAllocatedMultiIO[id] < dim*m_mbSize*m_numberOfuttsPerMinibatch)
                    {
                        delete[] m_labelsBufferMultiIO[id];
                        m_labelsBufferMultiIO[id] = new ElemType[dim*m_mbSize*m_numberOfuttsPerMinibatch];
                        memset(m_labelsBufferMultiIO[id].get(), 0, sizeof(ElemType)*dim*m_mbSize*m_numberOfuttsPerMinibatch);
                        m_labelsBufferAllocatedMultiIO[id] = dim*m_mbSize*m_numberOfuttsPerMinibatch;
                    }*/

                    /*if (channelIndex == 0)
                        memset(m_labelsBufferMultiIO[id], 0, sizeof(ElemType)*dim*m_mbSize*m_numberOfuttsPerMinibatch);*/
                    for (size_t j = 0, k = startFr; j < framenum; j++, k++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            m_labelsBufferMultiIO[id].get()[(k*m_numberOfuttsPerMinibatch + channelIndex)*dim + d] = m_labelsBufferMultiUtt[sourceChannelIndex].get()[j*dim + d + m_labelsStartIndexMultiUtt[id + sourceChannelIndex*numOfLabel]];
                        }
                    }
                }
            }
        }
    template<class ElemType>
        bool HTKMLFReader<ElemType>::GetMinibatchToWrite(std::map<std::wstring, Matrix<ElemType>*>& matrices)
        {
            std::map<std::wstring,size_t>::iterator iter;
            if     (m_checkDictionaryKeys)
            {
                for (auto iter=m_featureNameToIdMap.begin();iter!=m_featureNameToIdMap.end();iter++)
                {
                    if (matrices.find(iter->first)==matrices.end())
                    {
                        fprintf(stderr,"GetMinibatchToWrite: feature node %ls specified in reader not found in the network\n", iter->first.c_str());
                        throw std::runtime_error("GetMinibatchToWrite: feature node specified in reader not found in the network.");
                    }
                }
                /*
                   for (auto iter=matrices.begin();iter!=matrices.end();iter++)
                   {
                   if (m_featureNameToIdMap.find(iter->first)==m_featureNameToIdMap.end())
                   throw std::runtime_error(msra::strfun::strprintf("minibatch requested for input node %ws not found in reader - cannot generate input\n",iter->first.c_str()));
                   }
                   */
                m_checkDictionaryKeys=false;
            }

            if (m_inputFileIndex<m_inputFilesMultiIO[0].size())
            {
                m_fileEvalSource->Reset();

                // load next file (or set of files)
                foreach_index(i, m_inputFilesMultiIO)
                {
                    msra::asr::htkfeatreader reader;

                    const auto path = reader.parse(m_inputFilesMultiIO[i][m_inputFileIndex]);
                    // read file
                    msra::dbn::matrix feat;
                    string featkind;
                    unsigned int sampperiod;
                    msra::util::attempt (5, [&]()
                            {
                            reader.read (path, featkind, sampperiod, feat);   // whole file read as columns of feature vectors
                            });
                    fprintf (stderr, "evaluate: reading %d frames of %S\n", (int)feat.cols(), ((wstring)path).c_str());
                    m_fileEvalSource->AddFile(feat, featkind, sampperiod, i);
                }
                m_inputFileIndex++;

                // turn frames into minibatch (augment neighbors, etc)
                m_fileEvalSource->CreateEvalMinibatch();

                // populate input matrices
                bool first = true;
                typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
                for (iter = matrices.begin();iter!=matrices.end(); iter++)
                {
                    // dereference matrix that corresponds to key (input/output name) and 
                    // populate based on whether its a feature or a label

                    if (m_nameToTypeMap.find(iter->first)!=m_nameToTypeMap.end() && m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                    {
                        Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels
                        size_t id = m_featureNameToIdMap[iter->first];
                        size_t dim = m_featureNameToDimMap[iter->first];

                        const msra::dbn::matrix feat = m_fileEvalSource->ChunkOfFrames(id);
                        if (first)
                        {
                            m_pMBLayout->Init(1, feat.cols(), true/*sequential*/);
                            m_pMBLayout->Set(0, 0, MinibatchPackingFlags::SequenceStart);
                            m_pMBLayout->SetWithoutOr(0, feat.cols() - 1, MinibatchPackingFlags::SequenceEnd);  // BUGBUG: using SetWithoutOr() because original code did; but that seems inconsistent
                            first = false;
                        }

                        // copy the features over to our array type
                        assert(feat.rows()==dim); dim; // check feature dimension matches what's expected

                        if ((m_featuresBufferMultiIO[id] == nullptr) ||
                            (m_featuresBufferAllocatedMultiIO[id] < (feat.rows() * feat.cols())) /*buffer size changed. can be partial minibatch*/)
                        {
                            m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), feat.rows() * feat.cols());
                            m_featuresBufferAllocatedMultiIO[id] = feat.rows() * feat.cols();
                        }

                        if (sizeof(ElemType) == sizeof(float))
                        {
                            for (int j=0; j < feat.cols(); j++) // column major, so iterate columns
                            {
                                // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                                memcpy_s(&m_featuresBufferMultiIO[id].get()[j * feat.rows()], sizeof(ElemType) * feat.rows(), &feat(0, j), sizeof(ElemType) * feat.rows());
                            }
                        }
                        else
                        {
                            for (int j=0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
                            {
                                for (int i = 0; i < feat.rows(); i++)
                                {
                                    m_featuresBufferMultiIO[id].get()[j * feat.rows() + i] = feat(i, j);
                                }
                            }
                        }
                        data.SetValue(feat.rows(), feat.cols(), m_featuresBufferMultiIO[id].get(), matrixFlagNormal);
                    }
                }
                return true;
            }
            else
            {
                return false;
            }
        }


    template<class ElemType>
        bool HTKMLFReader<ElemType>::ReNewBufferForMultiIO(size_t i)
        {
            if (m_noData)
            {
                if ((i == 0) && m_fullutt)
                    m_toProcess[i] = 0;
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
            if ((m_featuresBufferMultiUtt[i] == NULL) || (m_featuresBufferAllocatedMultiUtt[i] < totalFeatNum))
            {
                m_featuresBufferMultiUtt[i] = AllocateIntermediateBuffer(-1 /*CPU*/, totalFeatNum);
                m_featuresBufferAllocatedMultiUtt[i] = totalFeatNum;
            }                    

            size_t totalLabelsNum = 0;
            for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it) 
            {
                size_t id = m_labelNameToIdMap[it->first];
                size_t dim  = m_labelNameToDimMap[it->first];

                const vector<size_t> & uids = m_mbiter->labels(id);
                size_t actualmbsizeOri = uids.size();
                m_labelsStartIndexMultiUtt[id+i*numOfLabel] = totalLabelsNum;
                totalLabelsNum = m_labelsStartIndexMultiUtt[id+i*numOfLabel] + dim * actualmbsizeOri;
            }

            if ((m_labelsBufferMultiUtt[i] == NULL) || (m_labelsBufferAllocatedMultiUtt[i] < totalLabelsNum))
            {
                m_labelsBufferMultiUtt[i] = AllocateIntermediateBuffer(-1 /*CPU */, totalLabelsNum);
                m_labelsBufferAllocatedMultiUtt[i] = totalLabelsNum;
            }

            memset(m_labelsBufferMultiUtt[i].get(), 0, sizeof(ElemType)*totalLabelsNum);

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
                } 
                else
                {
                    if (m_toProcess[i] != actualmbsizeOri)
                    {
                        throw std::runtime_error("The multi-IO features has inconsistent number of frames!");
                    }
                }
                assert (actualmbsizeOri == m_mbiter->currentmbframes());

                if (sizeof(ElemType) == sizeof(float))
                {
                    for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns
                    {
                        // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                        memcpy_s(&m_featuresBufferMultiUtt[i].get()[k*fdim + m_featuresStartIndexMultiUtt[id + i*numOfFea]], sizeof(ElemType)*fdim, &featOri(0, k), sizeof(ElemType)*fdim);
                    }
                }
                else
                {
                    for (int k=0; k < actualmbsizeOri; k++) // column major, so iterate columns in outside loop
                    {
                        for (int d = 0; d < featOri.rows(); d++)
                        {
                            m_featuresBufferMultiUtt[i].get()[k*featOri.rows() + d + m_featuresStartIndexMultiUtt[id + i*numOfFea]] = featOri(d, k);
                        }
                    }
                }
            }

            for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it) 
            {
                size_t id = m_labelNameToIdMap[it->first];
                size_t dim  = m_labelNameToDimMap[it->first];

                const vector<size_t> & uids = m_mbiter->labels(id);
                size_t actualmbsizeOri = uids.size();

                if (m_convertLabelsToTargetsMultiIO[id])
                {
                    size_t labelDim = m_labelToTargetMapMultiIO[id].size();
                    for (int k=0; k < actualmbsizeOri; k++)
                    {
                        assert(uids[k] < labelDim); labelDim;
                        size_t labelId = uids[k];
                        for (int j = 0; j < dim; j++)
                        {
                            m_labelsBufferMultiUtt[i].get()[k*dim + j + m_labelsStartIndexMultiUtt[id + i*numOfLabel]] = m_labelToTargetMapMultiIO[id][labelId][j];
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
                        m_labelsBufferMultiUtt[i].get()[k*dim + uids[k] + m_labelsStartIndexMultiUtt[id + i*numOfLabel]] = (ElemType)1;
                    }
                }
            }
            //lattice
            if (m_latticeBufferMultiUtt[i] != NULL)
            {
                m_latticeBufferMultiUtt[i].reset();
            }

            if (m_mbiter->haslattice())
                m_latticeBufferMultiUtt[i] = std::move(m_mbiter->lattice(0));
            m_labelsIDBufferMultiUtt[i].clear();
            m_labelsIDBufferMultiUtt[i] = m_mbiter->labels();
            m_phoneboundaryIDBufferMultiUtt[i].clear();
            m_phoneboundaryIDBufferMultiUtt[i] = m_mbiter->bounds();
            m_processedFrame[i] = 0;

            (*m_mbiter)++;
            if (!(*m_mbiter))
                m_noData = true;

            return true;    
        }

    // GetLabelMapping - Gets the label mapping from integer to type in file 
    // mappingTable - a map from numeric datatype to native label type stored as a string 
    template<class ElemType>
        const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& HTKMLFReader<ElemType>::GetLabelMapping(const std::wstring& /*sectionName*/)
        {
            return m_idToLabelMap;
        }

    // SetLabelMapping - Sets the label mapping from integer index to label 
    // labelMapping - mapping table from label values to IDs (must be 0-n)
    // note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
    template<class ElemType>
        void HTKMLFReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& labelMapping)
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
                    RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): mismatch between labelMappingFile and labelToTargetMappingFile");

                if (count==0)
                    targetDim = targetstring.length();
                else if (targetDim!=targetstring.length())
                    RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): inconsistent target length among records");

                std::vector<ElemType> targetVector(targetstring.length(),(ElemType)0.0);
                foreach_index(i, targetstring)
                {
                    if (targetstring.compare(i,1,L"1")==0)
                        targetVector[i] = (ElemType)1.0;
                    else if (targetstring.compare(i,1,L"0")!=0)
                        RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): expecting label2target mapping to contain only 1's or 0's");
                }
                labelToTargetMap.push_back(targetVector);
                count++;
            }

            // verify that statelist and label2target mapping file are in same order (to match up with reader) while reading mapping
            if (count!=labelList.size())
                RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): mismatch between lengths of labelMappingFile vs labelToTargetMappingFile");

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
        bool HTKMLFReader<ElemType>::GetData(const std::wstring& /*sectionName*/, size_t /*numRecords*/, void* /*data*/, size_t& /*dataBufferSize*/, size_t /*recordStart*/)
        {
            throw std::runtime_error("GetData not supported in HTKMLFReader");
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
                case endDataEpoch:
                case endDataSet:
                    throw std::logic_error("DataEnd: does not support endDataTypes: endDataNull, endDataEpoch and endDataSet");
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
                sentenceEnd[i] = m_switchFrame[i];
        }

    template<class ElemType>
        void HTKMLFReader<ElemType>::CopyMBLayoutTo(MBLayoutPtr pMBLayout)
        {
                pMBLayout->CopyFrom(m_pMBLayout);
        }


    // GetFileConfigNames - determine the names of the features and labels sections in the config file
    // features - [in,out] a vector of feature name strings
    // labels - [in,out] a vector of label name strings
    template<class ElemType>
        void HTKMLFReader<ElemType>::GetDataNamesFromConfig(const ConfigParameters& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels,
            std::vector<std::wstring>& hmms, std::vector<std::wstring>& lattices)            
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
                else if (temp.ExistsCurrent("mlfFile")|| temp.ExistsCurrent("mlfFileList"))
                {
                    labels.push_back(msra::strfun::utf16(iter->first));
                }
                else if (temp.ExistsCurrent("phoneFile"))
                {
                    hmms.push_back(msra::strfun::utf16(iter->first));
                }
                else if (temp.ExistsCurrent("denlatTocFile"))
                {
                    lattices.push_back(msra::strfun::utf16(iter->first));
                }

            }
        }

    template<class ElemType>
        void HTKMLFReader<ElemType>::ExpandDotDotDot(wstring & featPath, const wstring & scpPath, wstring & scpDirCached) 
        {
            wstring delim = L"/\\";

            if (scpDirCached.empty()) 
            {
                scpDirCached = scpPath;
                wstring tail; 
                auto pos = scpDirCached.find_last_of(delim);
                if (pos != wstring::npos)
                {
                    tail = scpDirCached.substr(pos + 1);
                    scpDirCached.resize(pos);
                }
                if (tail.empty()) // nothing was split off: no dir given, 'dir' contains the filename
                    scpDirCached.swap(tail);            
            }
            size_t pos = featPath.find(L"...");
            if (pos != featPath.npos)
                featPath = featPath.substr(0, pos) + scpDirCached + featPath.substr(pos + 3);
        }

    template class HTKMLFReader<float>;
    template class HTKMLFReader<double>;
}}}
