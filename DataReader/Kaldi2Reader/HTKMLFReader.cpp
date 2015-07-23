//
// <copyright file="HTKMLFReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "basetypes.h"

#include "htkfeatio.h"                  // for reading HTK features
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "simplesenonehmm.h"            // for MMI scoring

#include "rollingwindowsource.h"        // minibatch sources
#include "utterancesourcemulti.h"
#ifdef _WIN32
#include "readaheadsource.h"
#endif
#include "chunkevalsource.h"
#include "minibatchiterator.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "HTKMLFReader.h"
#include "commandArgUtil.h"
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

//#include <iostream>

//int msra::numa::node_override = -1;     // for numahelpers.h

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    void HTKMLFReader<ElemType>::Init(const ConfigParameters& readerConfig)
    {
        m_mbiter = NULL;
        m_frameSource = NULL;
        m_lattices = NULL;
        m_sequenceTrainingIO = NULL;
        m_noData = false;
        m_convertLabelsToTargets = false;
        m_doSeqTrain = false;

        if (readerConfig.Exists("legacyMode"))
        {
            RuntimeError("legacy mode has been deprecated\n");
        }

        // If <m_framemode> is false, throw away any utterance that is longer
        // than the specified <m_maxUtteranceLength>.
        m_maxUtteranceLength = readerConfig("maxUtteranceLength", "1500");

        // m_truncated:
        //     If true, truncate utterances to fit the minibatch size. Otherwise
        //     the actual minibatch size will be the length of the utterance.
        // m_numberOfuttsPerMinibatch:
        //     If larger than one, then each minibatch contains multiple
        //     utterances.
        m_truncated = readerConfig("Truncated", "false");
        m_numberOfuttsPerMinibatch = readerConfig("nbruttsineachrecurrentiter", "1");
        if (m_numberOfuttsPerMinibatch < 1)
        {
            LogicError("nbrUttsInEachRecurrentIter cannot be less than 1.\n");
        }

        // Initializes variables related to multi-utterance.
        m_actualnumberOfuttsPerMinibatch = m_numberOfuttsPerMinibatch;
        m_sentenceEnd.assign(m_numberOfuttsPerMinibatch, true);
        m_processedFrame.assign(m_numberOfuttsPerMinibatch, 0);
        m_toProcess.assign(m_numberOfuttsPerMinibatch, 0);
        m_switchFrame.assign(m_numberOfuttsPerMinibatch, 0);
        m_currentBufferFrames.assign(m_numberOfuttsPerMinibatch, 0);
        m_uttInfo.resize(m_numberOfuttsPerMinibatch);

        // Checks if we need to do sequence training.
        if (readerConfig.Exists("seqTrainCriterion"))
        {
            m_doSeqTrain = true;
            m_seqTrainCriterion = wstring(readerConfig("seqTrainCriterion"));
            if ((m_seqTrainCriterion != L"mpfe")
                && (m_seqTrainCriterion != L"smbr"))
            {
                LogicError("Current Supported sequence training criterion are: mpfe, smbr.\n");
            }
        }

        // Checks if framemode is false in sequence training.
        m_framemode = readerConfig("frameMode", "true");
        if (m_framemode && m_doSeqTrain)
        {
            LogicError("frameMode has to be false in sequence training.\n");
        }

        // Checks if partial minibatches are allowed.
        std::string minibatchMode(readerConfig("minibatchMode", "Partial"));
        m_partialMinibatch = !_stricmp(minibatchMode.c_str(), "Partial");

        // Checks if we are in "write" mode or "train/test" mode.
        string command(readerConfig("action",L""));
        if (command == "write")
        {
            m_trainOrTest = false;
            PrepareForWriting(readerConfig);
        }
        else
        {
            m_trainOrTest = true;
            PrepareForTrainingOrTesting(readerConfig);
        }
        
    }

    template<class ElemType>
    void HTKMLFReader<ElemType>::PrepareForSequenceTraining(const ConfigParameters& readerConfig)
    {
        // Parameters that we are looking for.
        wstring denlatRspecifier, aliRspecifier, transModelFilename, silencePhoneStr;
        ElemType oldAcousticScale;
        ElemType acousticScale;
        ElemType lmScale;
        bool oneSilenceClass;

        // Makes sure that "denlats" and "alignments" sections exist.
        if (!readerConfig.Exists("denlats"))
        {
            LogicError("Sequence training requested, but \"denlats\" section is not provided.\n");
        }
        if (!readerConfig.Exists("alignments"))
        {
            LogicError("Sequence training requested, but \"alignments\" section is not provided.\n");
        }

        // Processes "denlats" section. 
        ConfigParameters denlatConfig = readerConfig("denlats");
        if (!denlatConfig.Exists("rx"))
        {
            LogicError("Rspecifier is not provided for denominator lattices.\n");
        }
        if (!denlatConfig.Exists("kaldiModel"))
        {
            LogicError("Rspecifier is not provided for Kaldi model.\n");
        }
        denlatRspecifier = wstring(denlatConfig("rx"));
        transModelFilename = wstring(denlatConfig("kaldiModel"));
        silencePhoneStr = wstring(denlatConfig("silPhoneList", ""));
        oldAcousticScale = denlatConfig("oldAcousticScale", "0.0");
        acousticScale = denlatConfig("acousticScale", "0.2");
        lmScale = denlatConfig("lmScale", "1.0");
        oneSilenceClass = denlatConfig("oneSilenceClass", "true");

        // Processes "alignments" section.
        ConfigParameters aliConfig = readerConfig("alignments");
        if (!aliConfig.Exists("rx"))
        {
            LogicError("Rspecifier is not provided for alignments.\n");
        }
        aliRspecifier = wstring(aliConfig("rx"));

        // Initializes sequence training interface.
        m_sequenceTrainingIO = new KaldiSequenceTrainingIO<ElemType>(
            denlatRspecifier, aliRspecifier, transModelFilename,
            silencePhoneStr, m_seqTrainCriterion, oldAcousticScale,
            acousticScale, lmScale, oneSilenceClass);

        // Scans the configurations to get "seqTrainDeriv" type input and
        // "seqTrainObj" type input. Both are feature nodes, we feed derivatives
        // to training criterion node through "seqTrainDeriv" and feed objective
        // through "seqTrainObj".
        bool hasDrive = false, hasObj = false;
        for (auto iter = readerConfig.begin(); iter != readerConfig.end(); ++iter)
        {
            ConfigParameters temp = iter->second;
            if (temp.ExistsCurrent("type"))
            {
                if (temp("type") == "seqTrainDeriv")
                {
                    m_nameToTypeMap[msra::strfun::utf16(iter->first)] = InputOutputTypes::seqTrainDeriv;
                    hasDrive = true;
                }
                else if (temp("type") == "seqTrainObj")
                {
                    m_nameToTypeMap[msra::strfun::utf16(iter->first)] = InputOutputTypes::seqTrainObj;
                    hasObj = true;
                }
            }
        }
        if (!hasDrive || !hasObj)
        {
            LogicError("Missing seqTrainDeriv or seqTrainObj type feature\n");
        }
    }

    // Loads input and output data for training and testing. Below we list the
    // categories for different input/output:
    // features:      InputOutputTypes::real
    // labels:        InputOutputTypes::category
    // derivatives:   InputOutputTypes::seqTrainDeriv
    // objectives:    InputOutputTypes::seqTrainObj
    //
    // Note that we treat <derivatives> and <objectives> as features, but they
    // will be computed in the reader, rather then reading from disks. Those
    // will then be fed to training criterion node for training purposes.
    template<class ElemType>
    void HTKMLFReader<ElemType>::PrepareForTrainingOrTesting(const ConfigParameters& readerConfig)
    {
        // Loads files for sequence training.
        if (m_doSeqTrain)
        {
            PrepareForSequenceTraining(readerConfig);
        }
        else
        {
            m_sequenceTrainingIO = NULL;
        }

        // Variables related to multi-utterance.
        // m_featuresBufferMultiUtt:
        //     Holds pointers to the data trunk for each utterance.
        // m_featuresBufferAllocatedMultiUtt:
        //     Actual data stores here.
        m_featuresBufferMultiUtt.assign(m_numberOfuttsPerMinibatch, NULL);
        m_featuresBufferAllocatedMultiUtt.assign(m_numberOfuttsPerMinibatch, 0);
        m_labelsBufferMultiUtt.assign(m_numberOfuttsPerMinibatch, NULL);
        m_labelsBufferAllocatedMultiUtt.assign(m_numberOfuttsPerMinibatch, 0);

        // Gets a list of features and labels. Note that we assume feature
        // section has sub-field "scpFile" and label section has sub-field
        // "mlfFile".
        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;
        GetDataNamesFromConfig(readerConfig, featureNames, labelNames);
        if (featureNames.size() + labelNames.size() <= 1)
        {
            RuntimeError("network needs at least 1 input and 1 output specified!");
        }
            
        // Loads feature files.
        size_t iFeat = 0;
        vector<size_t> numContextLeft;
        vector<size_t> numContextRight;
        size_t numExpandToUtt = 0;
        vector<msra::asr::FeatureSection *> & scriptpaths = m_trainingOrTestingFeatureSections;
        foreach_index(i, featureNames)
        {
            ConfigParameters thisFeature = readerConfig(featureNames[i]);
            bool expandToUtt = thisFeature("expandToUtterance", "false"); // should feature be processed as an ivector?
            m_expandToUtt.push_back(expandToUtt);
            if (expandToUtt)
                numExpandToUtt++;

            // Figures out the context.
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

            if (expandToUtt && (numContextLeft[i] != 0 || numContextRight[1] != 0))
                RuntimeError("contextWindow expansion not permitted when expandToUtterance=true");

            // Figures the actual feature dimension, with context.
            m_featDims.push_back(thisFeature("dim"));
            m_featDims[i] = m_featDims[i] * (1 + numContextLeft[i] + numContextRight[i]); 

            // Figures out the category.
            string type = thisFeature("type", "Real");
            if (type == "Real")
            {
                m_nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
            }
            else
            {
                RuntimeError("feature type must be Real");
            }

            m_featureNameToIdMap[featureNames[i]] = iFeat;
            scriptpaths.push_back(new msra::asr::FeatureSection(thisFeature("scpFile"), thisFeature("rx"), thisFeature("featureTransform", "")));
            m_featureNameToDimMap[featureNames[i]] = m_featDims[i];

            m_featuresBufferMultiIO.push_back(NULL);
            m_featuresBufferAllocatedMultiIO.push_back(0);

            iFeat++;            
        }

        // Loads label files.
        size_t iLabel = 0;
        vector<wstring> statelistpaths;
        vector<wstring> mlfpaths;
        vector<vector<wstring>> mlfpathsmulti;
        foreach_index(i, labelNames)
        {
            ConfigParameters thisLabel = readerConfig(labelNames[i]);

            // Figures out label dimension.
            if (thisLabel.Exists("labelDim"))
                m_labelDims.push_back(thisLabel("labelDim"));
            else if (thisLabel.Exists("dim"))
                m_labelDims.push_back(thisLabel("dim"));
            else
                RuntimeError("labels must specify dim or labelDim");

            // Figures out the category.
            string type;
            if (thisLabel.Exists("labelType"))
                type = thisLabel("labelType"); // let's deprecate this eventually and just use "type"...
            else
                type = thisLabel("type","Category"); // outputs should default to category
            if (type == "Category")
                m_nameToTypeMap[labelNames[i]] = InputOutputTypes::category;
            else
                RuntimeError("label type must be Category");

            // Loads label mapping.
            statelistpaths.push_back(thisLabel("labelMappingFile",L""));

            m_labelNameToIdMap[labelNames[i]] = iLabel;
            m_labelNameToDimMap[labelNames[i]] = m_labelDims[i];
            mlfpaths.clear();
            mlfpaths.push_back(thisLabel("mlfFile"));
            mlfpathsmulti.push_back(mlfpaths);

            m_labelsBufferMultiIO.push_back(NULL);
            m_labelsBufferAllocatedMultiIO.push_back(0);

            iLabel++;

            // Figures out label to target mapping.
            wstring labelToTargetMappingFile(thisLabel("labelToTargetMappingFile",L""));
            if (labelToTargetMappingFile != L"")
            {
                std::vector<std::vector<ElemType>> labelToTargetMap;
                m_convertLabelsToTargetsMultiIO.push_back(true);
                if (thisLabel.Exists("targetDim"))
                {
                    m_labelNameToDimMap[labelNames[i]] = m_labelDims[i] = thisLabel("targetDim");
                }
                else
                    RuntimeError("output must specify targetDim if labelToTargetMappingFile specified!");
                size_t targetDim = ReadLabelToTargetMappingFile (labelToTargetMappingFile, statelistpaths[i], labelToTargetMap);    
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

        // Sanity check.
        if (iFeat != scriptpaths.size() || iLabel != mlfpathsmulti.size())
            throw std::runtime_error(msra::strfun::strprintf ("# of inputs files vs. # of inputs or # of output files vs # of outputs inconsistent\n"));

        if (iFeat == numExpandToUtt)
            throw std::runtime_error("At least one feature stream must be frame-based, not utterance-based\n");

        if (m_expandToUtt[0]) // first feature stream is ivector type - that will mess up lower level feature reader
            throw std::runtime_error("The first feature stream in the file must be frame-based not utterance based. Please reorder the feature blocks of your config appropriately\n");

        // Loads randomization method.
        size_t randomize = randomizeAuto;
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
        
        // Open script files for features.
        size_t numFiles = 0;
        size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
        vector<wstring> filelist;
        vector<vector<wstring>> infilesmulti;
        foreach_index(i, scriptpaths)
        {
            filelist.clear();
            std::wstring scriptpath = scriptpaths[i]->scpFile;
            fprintf(stderr, "reading script file %S ...", scriptpath.c_str());
            size_t n = 0;
            for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/; )
            {
                filelist.push_back (reader.wgetline());
                n++;
            }

            fprintf (stderr, " %lu entries\n", n);

            if (i == 0)
                numFiles=n;
            else
                if (n != numFiles)
                    throw std::runtime_error (msra::strfun::strprintf ("number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

            infilesmulti.push_back(filelist);
        }

        // Opens MLF files for labels.
        set<wstring> restrictmlftokeys;
        double htktimetoframe = 100000.0;           // default is 10ms 
        std::vector<std::map<std::wstring,std::vector<msra::asr::htkmlfentry>>> labelsmulti;
        int targets_delay = 0;
        if (readerConfig.Exists("targets_delay"))
        {
            targets_delay = readerConfig("targets_delay");
        }
        foreach_index(i, mlfpathsmulti)
        {
            msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>  
                labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], htktimetoframe, targets_delay);      // label MLF
            // get the temp file name for the page file
            labelsmulti.push_back(labels);
        }

        // Get the readMethod, default value is "blockRandomize", the other
        // option is "rollingWindow". We only support "blockRandomize" in
        // sequence training.
        std::string readMethod(readerConfig("readMethod", "blockRandomize"));
        if (!_stricmp(readMethod.c_str(), "blockRandomize"))
        {
            // construct all the parameters we don't need, but need to be passed to the constructor...
            std::pair<std::vector<wstring>,std::vector<wstring>> latticetocs;
            std::unordered_map<std::string,size_t> modelsymmap;
            m_lattices = new msra::dbn::latticesource(latticetocs, modelsymmap);

            // now get the frame source. This has better randomization and doesn't create temp files
            m_frameSource = new msra::dbn::minibatchutterancesourcemulti(
                scriptpaths, infilesmulti, labelsmulti, m_featDims, m_labelDims,
                numContextLeft, numContextRight, randomize, *m_lattices, m_latticeMap, m_framemode, m_expandToUtt);

        }
        else if (!_stricmp(readMethod.c_str(), "rollingWindow"))
        {
            // "rollingWindow" is not supported in sequence training.
            if (m_doSeqTrain)
            {
                LogicError("rollingWindow is not supported in sequence training.\n");
            }
            if (numExpandToUtt>0)
            {
                LogicError("rollingWindow reader does not support expandToUtt. Change to blockRandomize.\n");
            }
            std::string pageFilePath;
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
            if (pageFilePath.size()>MAX_PATH-14) // max length of input to GetTempFileName is PATH_MAX-14
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
                tempFile = (char *)pageFilePath.c_str();
                int fid = mkstemp(tempFile);
                unlink (tempFile);
                close (fid);
                pagePaths.push_back(GetWC(tempFile));
#endif
            }

            const bool mayhavenoframe=false;
            int addEnergy = 0;

            //m_frameSourceMultiIO = new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, randomize, pagepath, mayhavenoframe, addEnergy);
            //m_frameSourceMultiIO->setverbosity(verbosity);
            int verbosity = readerConfig("verbosity","2");
            m_frameSource = new msra::dbn::minibatchframesourcemulti(scriptpaths, infilesmulti, labelsmulti, m_featDims, m_labelDims, numContextLeft, numContextRight, randomize, pagePaths, mayhavenoframe, addEnergy);
            m_frameSource->setverbosity(verbosity);
        }
        else
        {
            RuntimeError("readMethod must be rollingWindow or blockRandomize");
        }

    }

    // Loads input and output data for training and testing. Below we list the
    // categories for different input/output:
    // features:      InputOutputTypes::real
    // labels:        InputOutputTypes::category
    template<class ElemType>
    void HTKMLFReader<ElemType>::PrepareForWriting(const ConfigParameters& readerConfig)
    {
        // Gets a list of features and labels. Note that we assume feature
        // section names have prefix "features" and label section names have
        // prefix "labels".
        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;
        GetDataNamesFromConfig(readerConfig, featureNames, labelNames);

        // Loads feature files.
        size_t iFeat = 0;
        vector<size_t> numContextLeft;
        vector<size_t> numContextRight;
        vector<size_t> realDims;
        vector<msra::asr::FeatureSection *> & scriptpaths = m_writingFeatureSections;
        foreach_index(i, featureNames)
        {
            ConfigParameters thisFeature = readerConfig(featureNames[i]);

            // Figures out the context.
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

            // Figures out the feature dimension, with context.
            realDims.push_back(thisFeature("dim"));
            realDims[i] = realDims[i] * (1 + numContextLeft[i] + numContextRight[i]);

            // Figures out the category.
            string type = thisFeature("type", "Real");
            if (type=="Real")
            {
                m_nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
            }
            else
            {
                RuntimeError("feature type must be Real");
            }

            m_featureNameToIdMap[featureNames[i]]= iFeat;
            scriptpaths.push_back(new msra::asr::FeatureSection(thisFeature("scpFile"), thisFeature("rx"), thisFeature("featureTransform", "")));
            m_featureNameToDimMap[featureNames[i]] = realDims[i];

            m_featuresBufferMultiIO.push_back(NULL);
            m_featuresBufferAllocatedMultiIO.push_back(0);
            iFeat++;
        }

        // Writing labels is not supported.
        if (labelNames.size() > 0)
            RuntimeError("writer mode does not support labels as inputs, only features");

        // Opens script files correspond to features.
        size_t numFiles = 0;
        vector<wstring> filelist;
        size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing
        size_t evalchunksize = 2048;
        foreach_index(i,scriptpaths)
        {
            filelist.clear();
            std::wstring scriptpath = scriptpaths[i]->scpFile;
            fprintf(stderr, "reading script file %S ...", scriptpath.c_str());
            size_t n = 0;
            for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly/*optimization*/; )
            {
                filelist.push_back (reader.wgetline());
                n++;
            }

            fprintf (stderr, " %zu entries\n", n);

            if (i==0)
                numFiles=n;
            else
                if (n!=numFiles)
                    throw std::runtime_error (msra::strfun::strprintf ("HTKMLFReader::InitEvalReader: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

            m_inputFilesMultiIO.push_back(filelist);
        }

        m_fileEvalSource = new msra::dbn::FileEvalSource(realDims, numContextLeft, numContextRight, evalchunksize);
    }

    // destructor - virtual so it gets called properly 
    template<class ElemType>
    HTKMLFReader<ElemType>::~HTKMLFReader()
    {
        if (m_mbiter != NULL)
        {
            delete m_mbiter;
            m_mbiter = NULL;
        }
        if (m_frameSource != NULL)
        {
            delete m_frameSource;
            m_frameSource = NULL;
        }
        if (m_lattices != NULL)
        {
            delete m_lattices;
            m_lattices = NULL;
        }
        if (m_sequenceTrainingIO != NULL)
        {
            delete m_sequenceTrainingIO;
            m_sequenceTrainingIO = NULL;
        }

        if (!m_featuresBufferMultiIO.empty())
        {
            foreach_index(i, m_featuresBufferMultiIO)
            {
                if (m_featuresBufferMultiIO[i] != NULL)
                {
                    delete[] m_featuresBufferMultiIO[i];
                    m_featuresBufferMultiIO[i] = NULL;
                }
            }

        }

        if (!m_labelsBufferMultiIO.empty())
        {
            foreach_index(i, m_labelsBufferMultiIO)
            {
                if (m_labelsBufferMultiIO[i] != NULL)
                {
                    delete[] m_labelsBufferMultiIO[i];
                    m_labelsBufferMultiIO[i] = NULL;
                }
            }

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

        foreach_index(i, m_trainingOrTestingFeatureSections) {
            if (m_trainingOrTestingFeatureSections[i] != NULL)
            {
                delete m_trainingOrTestingFeatureSections[i];
                m_trainingOrTestingFeatureSections[i] = NULL;
            }
        }

        foreach_index(i, m_writingFeatureSections) {
            if (m_writingFeatureSections[i] != NULL)
            {
                delete m_writingFeatureSections[i];
                m_writingFeatureSections[i] = NULL;
            }
        }
    }

    // StartMinibatchLoop - Startup a minibatch loop 
    // mbSize - [in] size of the minibatch (number of frames, etc.)
    // epoch - [in] epoch number for this loop
    // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
    template<class ElemType>
    void HTKMLFReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
    {
        m_mbSize = mbSize;

        if (m_trainOrTest)
        {
            StartMinibatchLoopToTrainOrTest(mbSize, epoch, requestedEpochSamples);
        }
        else
        {
            StartMinibatchLoopToWrite(mbSize, epoch, requestedEpochSamples);    
        }
        m_checkDictionaryKeys=true;
    }

    template<class ElemType>
    void HTKMLFReader<ElemType>::StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
    {
        size_t datapasses=1;
        size_t totalFrames = m_frameSource->totalframes();
        size_t extraFrames = totalFrames % mbSize;
        size_t minibatches = totalFrames / mbSize;

        // If partial minibatch is not allowed, we have to modify <totalFrames>
        // and <requestedEpochSamples>.
        if (!m_partialMinibatch)
        {
            if (totalFrames > mbSize)
            {
                totalFrames -= extraFrames;
            }

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

        // Gets a new minibatch iterator.
        if (m_mbiter != NULL)
        {
            delete m_mbiter;
            m_mbiter = NULL;
        }
        msra::dbn::minibatchsource* source = m_frameSource;
        m_mbiter = new msra::dbn::minibatchiterator(*source, epoch, requestedEpochSamples, mbSize, datapasses);

        // Clears feature and label buffer.
        if (!m_featuresBufferMultiIO.empty())
        {
            foreach_index(i, m_featuresBufferMultiIO)
            {
                if (m_featuresBufferMultiIO[i] != NULL)
                {
                    delete[] m_featuresBufferMultiIO[i];
                    m_featuresBufferMultiIO[i] = NULL;
                    m_featuresBufferAllocatedMultiIO[i] = 0;
                }
            }
        }
        if (!m_labelsBufferMultiIO.empty())
        {
            foreach_index(i, m_labelsBufferMultiIO)
            {
                if (m_labelsBufferMultiIO[i] != NULL)
                {
                    delete[] m_labelsBufferMultiIO[i];
                    m_labelsBufferMultiIO[i] = NULL;
                    m_labelsBufferAllocatedMultiIO[i] = 0;
                }
            }
        }
        m_noData = false;
        m_featuresStartIndexMultiUtt.assign(m_featuresBufferMultiIO.size() * m_numberOfuttsPerMinibatch, 0);
        m_labelsStartIndexMultiUtt.assign(m_labelsBufferMultiIO.size() * m_numberOfuttsPerMinibatch, 0);
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

    template<class ElemType>
    void HTKMLFReader<ElemType>::StartMinibatchLoopToWrite(size_t mbSize, size_t /*epoch*/, size_t /*requestedEpochSamples*/)
    {
        m_fileEvalSource->Reset();
        m_fileEvalSource->SetMinibatchSize(mbSize);
        //m_chunkEvalSourceMultiIO->reset();
        m_inputFileIndex=0;

        foreach_index(i, m_featuresBufferMultiIO)
        {
            if (m_featuresBufferMultiIO[i] != NULL)
            {
                delete[] m_featuresBufferMultiIO[i];
                m_featuresBufferMultiIO[i] = NULL;
                m_featuresBufferAllocatedMultiIO[i] = 0;
            }
        }

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

    // The notation of "utterance" here is a little bit tricky:
    //     If <frameMode> is true, then it is just a trunk of data we read from
    //     minibatch iterator.
    //     If <frameMode> is false, then it is a real utterance we read from
    //     minibatch iterator.
    // Note, startFrame and endFrame are left close and right open. For example,
    // if startFrame = 5, endFrame = 10, then we copy frames 5, 6, 7, 8, 9.
    template<class ElemType>
    bool HTKMLFReader<ElemType>::PopulateUtteranceInMinibatch(
        std::map<std::wstring, Matrix<ElemType>*>& matrices,
        size_t uttIndex, size_t startFrame,
        size_t endFrame, size_t mbSize, size_t mbOffset)
    {
        bool success = true;

        // Sanity check.
        if (startFrame >= endFrame)
        {
            return false;
        }
        if (endFrame - startFrame > mbSize)
        {
            return false;
        }
        if (m_doSeqTrain && m_numberOfuttsPerMinibatch > 1)
        {
            LogicError("nbrUttsInEachRecurrentIter has to be 1 in sequence training.\n");
        }

        size_t numOfFea = m_featuresBufferMultiIO.size();
        size_t numOfLabel = m_labelsBufferMultiIO.size();
        typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
        for (iter = matrices.begin(); iter != matrices.end(); iter++)
        {
            if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
            {   // Features.
                size_t id = m_featureNameToIdMap[iter->first];
                size_t dim = m_featureNameToDimMap[iter->first];
  
                if (m_featuresBufferMultiIO[id] == NULL)
                {
                    m_featuresBufferMultiIO[id] = new ElemType[dim * mbSize * m_numberOfuttsPerMinibatch];
                    m_featuresBufferAllocatedMultiIO[id] = dim * mbSize * m_numberOfuttsPerMinibatch;
                }
                else if (m_featuresBufferAllocatedMultiIO[id] < dim * mbSize * m_numberOfuttsPerMinibatch)
                {   // Buffer too small, we have to increase it.
                    delete[] m_featuresBufferMultiIO[id];
                    m_featuresBufferMultiIO[id] = new ElemType[dim * mbSize * m_numberOfuttsPerMinibatch];
                    m_featuresBufferAllocatedMultiIO[id] = dim * mbSize * m_numberOfuttsPerMinibatch;
                }
  
                if (sizeof(ElemType) == sizeof(float))
                {   // For float, we copy entire column.
                    for (size_t j = startFrame, k = 0; j < endFrame; j++, k++)
                    {
                        memcpy_s(&m_featuresBufferMultiIO[id][((k + mbOffset) * m_numberOfuttsPerMinibatch + uttIndex) * dim],
                                 sizeof(ElemType) * dim,
                                 &m_featuresBufferMultiUtt[uttIndex][j * dim + m_featuresStartIndexMultiUtt[id + uttIndex * numOfFea]],
                                 sizeof(ElemType) * dim);
                    }
                }
                else
                {   // For double, we have to copy element by element.
                    for (size_t j = startFrame, k=0; j < endFrame; j++, k++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            m_featuresBufferMultiIO[id][((k + mbOffset) * m_numberOfuttsPerMinibatch + uttIndex) * dim + d]
                                = m_featuresBufferMultiUtt[uttIndex][j * dim + d + m_featuresStartIndexMultiUtt[id + uttIndex * numOfFea]];
                        }
                    }
                }
            }
            else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
            {   // Labels.
                size_t id = m_labelNameToIdMap[iter->first];
                size_t dim = m_labelNameToDimMap[iter->first];
  
                if (m_labelsBufferMultiIO[id] == NULL)
                {
                    m_labelsBufferMultiIO[id] = new ElemType[dim * mbSize * m_numberOfuttsPerMinibatch];
                    m_labelsBufferAllocatedMultiIO[id] = dim * mbSize * m_numberOfuttsPerMinibatch;
                }
                else if (m_labelsBufferAllocatedMultiIO[id] < dim * mbSize * m_numberOfuttsPerMinibatch)
                {
                    delete[] m_labelsBufferMultiIO[id];
                    m_labelsBufferMultiIO[id] = new ElemType[dim * mbSize * m_numberOfuttsPerMinibatch];
                    m_labelsBufferAllocatedMultiIO[id] = dim * mbSize * m_numberOfuttsPerMinibatch;
                }
  
                for (size_t j = startFrame, k=0; j < endFrame; j++, k++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        m_labelsBufferMultiIO[id][((k + mbOffset) * m_numberOfuttsPerMinibatch + uttIndex) * dim + d]
                            = m_labelsBufferMultiUtt[uttIndex][j * dim + d + m_labelsStartIndexMultiUtt[id + uttIndex * numOfLabel]];
                    }
                }
            }
            else if (m_doSeqTrain)
            {
                // TODO(GUOGUO): if we are going to allow "m_truncate" for
                //               sequence training, we will have to modify the
                //               following -- the following always assume we
                //               start filling the minibatch from index 0.
                // If we do sequence training we have to populate the derivative
                // features as well as the objective features. But unlike the
                // features and labels, we put them in to <matrices> directly.
                // We assume we only process one utterance at a time in the
                // current implementation.
                assert(uttIndex == 0);
                if (m_nameToTypeMap[iter->first] == InputOutputTypes::seqTrainDeriv)
                {
                    wstring uttID = m_uttInfo[uttIndex][0].first; 
                    Matrix<ElemType>& data = *matrices[iter->first];
                    if (m_sequenceTrainingIO->HasDerivatives(uttID))
                        m_sequenceTrainingIO->GetDerivatives(startFrame, endFrame, mbSize, uttID, data);
                    else
                    {
                        data.Resize(data.GetNumRows(), mbSize);
                        data.SetValue(0);
                    }
                }
                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::seqTrainObj)
                {
                    wstring uttID = m_uttInfo[uttIndex][0].first; 
                    Matrix<ElemType>& data = *matrices[iter->first];
                    if (m_sequenceTrainingIO->HasDerivatives(uttID))
                        m_sequenceTrainingIO->GetObjectives(startFrame, endFrame, uttID, data);
                    else
                        data.SetValue(0);
                }
            }
        }
        return success;
    }

    template<class ElemType>
    bool HTKMLFReader<ElemType>::GetMinibatchToTrainOrTest(std::map<std::wstring, Matrix<ElemType>*>& matrices)
    {
        bool skip = false;

        // On first minibatch, check if we have input for given names.
        if (m_checkDictionaryKeys)
        {
            std::map<std::wstring,size_t>::iterator iter;
            for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
            {
                if (m_nameToTypeMap.find(iter->first) == m_nameToTypeMap.end())
                {
                    throw std::runtime_error(msra::strfun::strprintf("minibatch requested for input node %S not found in reader - cannot generate input\n", iter->first.c_str()));
                }

            }
            m_checkDictionaryKeys=false;
        }

        size_t currentMBSize = m_mbSize;
        do 
        {
            // Checks if we have finished all the utterances.
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
                if (endEpoch)
                {
                    return false;
                }
            }

            // If <m_truncated> is true, <currentMBSize> is <m_mbSize>
            // If <m_truncated> is false, <currentMBSize> equals to the longest
            // utterance in the minibatch.
            if (!m_truncated)
            {
                currentMBSize = 0;
                for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
                {
                    if (m_currentBufferFrames[i] > currentMBSize)
                    {
                        currentMBSize = m_currentBufferFrames[i];
                    }
                }
            }

            // We initialize the sentence boundary information before we process
            // the utterances.
            m_sentenceBegin.Resize(m_numberOfuttsPerMinibatch, currentMBSize);
            m_minibatchPackingFlag.resize(currentMBSize);
            for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
            {
                for (size_t j = 0; j < currentMBSize; j++)
                {
                    m_sentenceBegin.SetValue(i, j, (ElemType) SENTENCE_MIDDLE);
                }
            }
            std::fill(m_minibatchPackingFlag.begin(), m_minibatchPackingFlag.end(), MinibatchPackingFlag::None);

            // Iterates over utterances. m_numberOfuttsPerMinibatch = 1 is a
            // special case.
            for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
            {
                size_t startFrame = m_processedFrame[i];
                size_t endFrame = 0;

                if ((startFrame + currentMBSize) < m_toProcess[i])
                {
                    // There is only 1 case:
                    //     1. <m_framemode> is false, and <m_truncated> is true.
                    assert(m_framemode == false);
                    assert(m_truncated == true);

                    // Sets the utterance boundary.
                    if (startFrame == 0)
                    {
                        m_sentenceBegin.SetValue(i, 0, (ElemType)SENTENCE_BEGIN);
                        m_minibatchPackingFlag[0] |= MinibatchPackingFlag::UtteranceStart;
                    }

                    endFrame = startFrame + currentMBSize;
                    bool populateSucc = PopulateUtteranceInMinibatch(matrices, i, startFrame, endFrame, currentMBSize);
                    m_processedFrame[i] += currentMBSize;
                }
                else if ((startFrame + currentMBSize) == m_toProcess[i])
                {
                    // There are 3 cases:
                    //     1. <m_framemode> is false, and <m_truncated> is true,
                    //        and it reaches the end of the utterance.
                    //     2. <m_framemode> is false, and <m_truncated> is false
                    //        and it reaches the end of the utterance.
                    //     3. <m_framemode> is true, then we do not have to set
                    //        utterance boundary.
                    
                    // Sets the utterance boundary.
                    if (m_framemode == false)
                    {
                        // If <m_truncated> is false, then the whole utterance
                        // will be loaded into the minibatch.
                        if (m_truncated == false)
                        {
                            assert(startFrame == 0);
                            m_sentenceBegin.SetValue(i, 0, (ElemType)SENTENCE_BEGIN);
                            m_minibatchPackingFlag[0] |= MinibatchPackingFlag::UtteranceStart;
                        }

                        // We have to set the utterance end.
                        m_sentenceBegin.SetValue(i, m_sentenceBegin.GetNumCols() - 1, (ElemType)SENTENCE_END);
                        m_minibatchPackingFlag[m_sentenceBegin.GetNumCols() - 1] |= MinibatchPackingFlag::UtteranceEnd;
                    }

                    // Now puts the utterance into the minibatch, and loads the
                    // next one.
                    endFrame = startFrame + currentMBSize;
                    bool populateSucc = PopulateUtteranceInMinibatch(matrices, i, startFrame, endFrame, currentMBSize);
                    m_processedFrame[i] += currentMBSize;
                    bool reNewSucc = ReNewBufferForMultiIO(i);
                }
                else
                {
                    // There are 3 cases:
                    //     1. <m_framemode> is true, then it must be a partial
                    //        minibatch.
                    //     2. <m_framemode> is false, <m_truncated> is true,
                    //        then we have to pull frames from next utterance.
                    //     3. <m_framemode> is false, <m_truncated> is false,
                    //        then the utterance is too short, we should try to
                    //        pull next utterance.
                    
                    // Checks if we have reached the end of the minibatch.
                    if (startFrame == m_toProcess[i])
                    {
                        for (size_t k = 0; k < currentMBSize; k++)
                        {
                            m_sentenceBegin.SetValue(i, k, (ElemType) NO_LABELS);
                            m_minibatchPackingFlag[k] |= MinibatchPackingFlag::NoLabel;

                            // Populates <NO_LABELS> with real features, the
                            // following implementation is not efficient...
                            assert(m_toProcess[i] > 0);
                            PopulateUtteranceInMinibatch(matrices, i, 0, 1, currentMBSize, k);
                        }
                        continue;
                    }

                    // First, if <m_framemode> is true, then it must be a
                    // partial minibatch, and if that is not allowed, we have to
                    // skip this minibatch.
                    if (m_framemode && !m_partialMinibatch)
                    {
                        skip = true;
                        bool reNewSucc = ReNewBufferForMultiIO(i);  // Should return false?
                        continue;
                    }

                    // Second, we set utterance boundary for the partial
                    // minibatch, and then load it.
                    if (m_framemode == false)
                    {
                        // If <m_truncated> is false, then the whole utterance
                        // will be loaded into the minibatch.
                        if (m_truncated == false)
                        {
                            assert(startFrame == 0);
                            m_sentenceBegin.SetValue(i, 0, (ElemType)SENTENCE_BEGIN);
                            m_minibatchPackingFlag[0] |= MinibatchPackingFlag::UtteranceStart;
                        }

                        // We have to set the utterance end.
                        assert(m_toProcess[i] - startFrame - 1 < m_sentenceBegin.GetNumCols());
                        m_sentenceBegin.SetValue(i, m_toProcess[i] - startFrame - 1, (ElemType)SENTENCE_END);
                        m_minibatchPackingFlag[m_toProcess[i] - startFrame - 1] |= MinibatchPackingFlag::UtteranceEnd;
                    }
                    endFrame = m_toProcess[i];
                    size_t currentMBFilled = endFrame - startFrame;
                    bool populateSucc = PopulateUtteranceInMinibatch(matrices, i, startFrame, endFrame, currentMBSize);
                    m_processedFrame[i] += currentMBFilled;
                    bool reNewSucc = ReNewBufferForMultiIO(i);

                    // Third, if the next utterance can fit into the current
                    // minibatch, we also pack the next utterance.
                    while (reNewSucc && (currentMBFilled + m_toProcess[i] <= currentMBSize))
                    {
                        // Sets the utterance boundary.
                        assert(currentMBFilled + m_toProcess[i] <= m_sentenceBegin.GetNumCols());
                        m_sentenceBegin.SetValue(i, currentMBFilled, (ElemType)SENTENCE_BEGIN);
                        m_minibatchPackingFlag[currentMBFilled] |= MinibatchPackingFlag::UtteranceStart;
                        m_sentenceBegin.SetValue(i, currentMBFilled + m_toProcess[i] - 1, (ElemType)SENTENCE_END);
                        m_minibatchPackingFlag[currentMBFilled + m_toProcess[i] - 1] |= MinibatchPackingFlag::UtteranceEnd;
                        populateSucc = PopulateUtteranceInMinibatch(matrices, i, 0, m_toProcess[i], currentMBSize, currentMBFilled);
                        assert(m_processedFrame[i] == 0);
                        m_processedFrame[i] = m_toProcess[i];
                        currentMBFilled += m_toProcess[i];
                        reNewSucc = ReNewBufferForMultiIO(i);
                    }

                    // Finally, pulls frames from next utterance if the current
                    // minibatch is not full.
                    if (reNewSucc && !m_framemode && m_truncated)
                    {
                        populateSucc = PopulateUtteranceInMinibatch(matrices, i, 0, currentMBSize - currentMBFilled, currentMBSize, currentMBFilled);
                        m_processedFrame[i] += currentMBSize - currentMBFilled;
                        if (currentMBFilled < currentMBSize)
                        {
                            m_sentenceBegin.SetValue(i, currentMBFilled, (ElemType)SENTENCE_BEGIN);
                            m_minibatchPackingFlag[currentMBFilled] |= MinibatchPackingFlag::UtteranceStart;
                        }
                    }
                    else
                    {
                        for (size_t k = currentMBFilled; k < currentMBSize; k++)
                        {
                            m_sentenceBegin.SetValue(i, k, (ElemType) NO_LABELS);
                            m_minibatchPackingFlag[k] |= MinibatchPackingFlag::NoLabel;

                            // Populates <NO_LABELS> with real features, the
                            // following implementation is not efficient...
                            assert(m_toProcess[i] > 0);
                            PopulateUtteranceInMinibatch(matrices, i, 0, 1, currentMBSize, k);
                        }
                    }
                }
            }

            typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
            for (iter = matrices.begin(); iter != matrices.end(); iter++)
            {
                Matrix<ElemType>& data = *matrices[iter->first];
                if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                {
                    size_t id = m_featureNameToIdMap[iter->first];
                    size_t dim = m_featureNameToDimMap[iter->first];
                    data.SetValue(dim, currentMBSize * m_numberOfuttsPerMinibatch, m_featuresBufferMultiIO[id] , matrixFlagNormal);
                }
                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                {
                    size_t id = m_labelNameToIdMap[iter->first];
                    size_t dim = m_labelNameToDimMap[iter->first];
                    data.SetValue(dim, currentMBSize * m_numberOfuttsPerMinibatch, m_labelsBufferMultiIO[id], matrixFlagNormal);
                }
            }
            skip=false;
        }
        while(skip);

        return true;
    }

    template<class ElemType>
    bool HTKMLFReader<ElemType>::GetMinibatchToWrite(std::map<std::wstring, Matrix<ElemType>*>& matrices)
    {
        std::map<std::wstring,size_t>::iterator iter;
        if (m_checkDictionaryKeys)
        {
            for (auto iter = m_featureNameToIdMap.begin(); iter != m_featureNameToIdMap.end(); iter++)
            {
                if (matrices.find(iter->first) == matrices.end())
                {
                    fprintf(stderr,"GetMinibatchToWrite: feature node %S specified in reader not found in the network\n", iter->first.c_str());
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
            m_checkDictionaryKeys = false;
        }

        if (m_inputFileIndex<m_inputFilesMultiIO[0].size())
        {
            m_fileEvalSource->Reset();

            // load next file (or set of files)
            foreach_index(i, m_inputFilesMultiIO)
            {
                msra::asr::htkfeatreader reader;

                const auto path = reader.parse(m_inputFilesMultiIO[i][m_inputFileIndex], m_writingFeatureSections[i]);
                // read file
                msra::dbn::matrix feat;
                string featkind;
                unsigned int sampperiod;
                msra::util::attempt (5, [&]()
                {
                    reader.readAlloc (path, featkind, sampperiod, feat);   // whole file read as columns of feature vectors
                });
                fprintf (stderr, "evaluate: reading %zu frames of %S\n", feat.cols(), ((wstring)path).c_str());
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
                        m_sentenceBegin.Resize((size_t)1, (size_t)feat.cols());
                        m_minibatchPackingFlag.resize(feat.cols());
                        m_sentenceBegin.SetValue((ElemType) SENTENCE_MIDDLE);
                        m_sentenceBegin.SetValue(0, 0, (ElemType) SENTENCE_BEGIN);
                        m_sentenceBegin.SetValue(0, (size_t)feat.cols()-1, (ElemType) SENTENCE_END);
                                
                        std::fill(m_minibatchPackingFlag.begin(), m_minibatchPackingFlag.end(), MinibatchPackingFlag::None);
                        m_minibatchPackingFlag[0] = MinibatchPackingFlag::UtteranceStart;
                        m_minibatchPackingFlag[(size_t)feat.cols()-1] = MinibatchPackingFlag::UtteranceEnd;
                        first = false;
                    }

                    // copy the features over to our array type
                    assert(feat.rows()==dim); dim; // check feature dimension matches what's expected

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
                else
                {   // Resizes other inputs so they won't affect actual minibatch size.
                    Matrix<ElemType>& data = *matrices[iter->first];
                    data.Resize(data.GetNumRows(), 1);
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
            return false;
        }

        size_t numOfFea = m_featuresBufferMultiIO.size();
        size_t numOfLabel = m_labelsBufferMultiIO.size();

        // Number of frames we get from minibatch iterator.
        m_currentBufferFrames[i] = m_mbiter->currentmbframes();

        // If we operate at utterance level, we get the utterance information.
        if (!m_framemode)
        {
            m_uttInfo[i] = m_mbiter->getutteranceinfo();
            assert(m_uttInfo[i].size() > 0);
            // For now, let's just assume that the utterance length will be
            // larger than the minibatch size. We can handle this more
            // gracefully later, e.g., remove the utterance in the reader.
            if (m_uttInfo[i].size() > 1)
            {
                fprintf(stderr, "WARNING: Utterance length is smaller than the minibatch size, you may want to remove the utterance or reduce the minibatch size.\n");
            }

            if (!m_truncated && (m_currentBufferFrames[i] > m_maxUtteranceLength))
            {
                (*m_mbiter)++;
                if (!(*m_mbiter))
                    m_noData = true;
                fprintf(stderr, "WARNING: Utterance \"%S\" has length longer than the %d, skipping it.\n", m_uttInfo[i][0].first.c_str(), m_maxUtteranceLength);
                return ReNewBufferForMultiIO(i);
            }

        }

        // Sets up feature buffers.
        // foreach_index(id, m_featuresBufferAllocatedMultiIO)
        size_t totalFeatNum = 0;
        for (auto it = m_featureNameToIdMap.begin(); it != m_featureNameToIdMap.end(); ++it)
        {
            size_t id = m_featureNameToIdMap[it->first];
            const msra::dbn::matrixstripe featOri = m_mbiter->frames(id);
            size_t fdim = featOri.rows();
            const size_t actualmbsizeOri = featOri.cols();
            m_featuresStartIndexMultiUtt[id + i * numOfFea] = totalFeatNum;
            totalFeatNum = fdim * actualmbsizeOri + m_featuresStartIndexMultiUtt[id + i * numOfFea];
        }
        if (m_featuresBufferMultiUtt[i] == NULL)
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

        // Sets up label buffers.
        size_t totalLabelsNum = 0;
        for (auto it = m_labelNameToIdMap.begin(); it != m_labelNameToIdMap.end(); ++it) 
        {
            size_t id = m_labelNameToIdMap[it->first];
            size_t dim  = m_labelNameToDimMap[it->first];

            const vector<size_t> & uids = m_mbiter->labels(id);
            size_t actualmbsizeOri = uids.size();
            m_labelsStartIndexMultiUtt[id + i * numOfLabel] = totalLabelsNum;
            totalLabelsNum = m_labelsStartIndexMultiUtt[id + i * numOfLabel] + dim * actualmbsizeOri;
        }
        if (m_labelsBufferMultiUtt[i] == NULL)
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
        memset(m_labelsBufferMultiUtt[i], 0, sizeof(ElemType) * totalLabelsNum);

        // Copies features to buffer.
        // foreach_index(id, m_featuresBufferMultiIO)
        bool first = true;
        for (auto it = m_featureNameToIdMap.begin(); it != m_featureNameToIdMap.end(); ++it)
        {
            size_t id = m_featureNameToIdMap[it->first];
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
                    memcpy_s(&m_featuresBufferMultiUtt[i][k * fdim + m_featuresStartIndexMultiUtt[id + i * numOfFea]], sizeof(ElemType) * fdim, &featOri(0, k), sizeof(ElemType) * fdim);
                }
            }
            else
            {
                for (int k=0; k < actualmbsizeOri; k++) // column major, so iterate columns in outside loop
                {
                    for (int d = 0; d < featOri.rows(); d++)
                    {
                        m_featuresBufferMultiUtt[i][k * fdim + d + m_featuresStartIndexMultiUtt[id + i * numOfFea]] = featOri(d, k);
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
                        m_labelsBufferMultiUtt[i][k * dim + j + m_labelsStartIndexMultiUtt[id + i * numOfLabel]] = m_labelToTargetMapMultiIO[id][labelId][j];
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
                    m_labelsBufferMultiUtt[i][k * dim + uids[k] + m_labelsStartIndexMultiUtt[id + i * numOfLabel]]=(ElemType)1;
                }
            }
        }
        m_processedFrame[i] = 0;

        (*m_mbiter)++;
        if (!(*m_mbiter))
            m_noData = true;

        return true;    
    }

    // Gets a copy of the utterance that corresponds to the current minibatches,
    // which will be used to do a neural network forward computation.
    template<class ElemType>
    bool HTKMLFReader<ElemType>::GetForkedUtterance(std::wstring& uttID,
                                                    std::map<std::wstring, Matrix<ElemType>*>& matrices)
    {
        if (!m_doSeqTrain)
        {
            return false;
        }
        assert(m_framemode == false);

        // For the moment we only support single utterance.
        if (m_numberOfuttsPerMinibatch != 1)
        {
            RuntimeError("The current sequence training implementation does not support multiple utterances.\n");
        }

        // Under our current assumption, we only have one utterance at a time.
        uttID = m_uttInfo[0][0].first;
        if (!m_sequenceTrainingIO->HasDerivatives(uttID))
        {
            size_t startFrame = 0;
            size_t endFrame = m_uttInfo[0][0].second;
            size_t currentMBSize = endFrame - startFrame;
            bool populateSucc = PopulateUtteranceInMinibatch(
                matrices, 0, startFrame, endFrame, currentMBSize);
            if (!populateSucc)
            {
                return false;
            }

            // Sets sentence boundary.
            m_sentenceBegin.Resize(1, currentMBSize);
            m_minibatchPackingFlag.resize(currentMBSize);
            for (size_t i = 0; i < currentMBSize; i++)
            {
                m_sentenceBegin.SetValue(0, i, (ElemType) SENTENCE_MIDDLE);
            }
            std::fill(m_minibatchPackingFlag.begin(), m_minibatchPackingFlag.end(), MinibatchPackingFlag::None);
            m_sentenceBegin.SetValue(0, 0, (ElemType)SENTENCE_BEGIN);
            m_sentenceBegin.SetValue(0, m_sentenceBegin.GetNumCols() - 1, (ElemType) SENTENCE_END);
            m_minibatchPackingFlag[0] = MinibatchPackingFlag::UtteranceStart;
            m_minibatchPackingFlag[m_sentenceBegin.GetNumCols() - 1] = MinibatchPackingFlag::UtteranceEnd;

            typename std::map<std::wstring, Matrix<ElemType>*>::iterator iter;
            for (iter = matrices.begin(); iter != matrices.end(); iter++)
            {
                Matrix<ElemType>& data = *matrices[iter->first];
                if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                {
                    size_t id = m_featureNameToIdMap[iter->first];
                    size_t dim = m_featureNameToDimMap[iter->first];
                    data.SetValue(dim, currentMBSize * m_numberOfuttsPerMinibatch, m_featuresBufferMultiIO[id] , matrixFlagNormal);
                }
                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                {
                    size_t id = m_labelNameToIdMap[iter->first];
                    size_t dim = m_labelNameToDimMap[iter->first];
                    data.SetValue(dim, currentMBSize * m_numberOfuttsPerMinibatch, m_labelsBufferMultiIO[id], matrixFlagNormal);
                }
            }
            return true;
        }

        return false;
    }

    template<class ElemType>
    bool HTKMLFReader<ElemType>::ComputeDerivativeFeatures(const std::wstring& uttID,
                                                           const Matrix<ElemType>& outputs)
    {
        return m_sequenceTrainingIO->ComputeDerivatives(uttID, outputs);
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
    void HTKMLFReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<typename IDataReader<ElemType>::LabelIdType, LabelType>& labelMapping)
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
        {
            sentenceEnd[i] = m_switchFrame[i];
        }
    }

    template<class ElemType>
    void HTKMLFReader<ElemType>::SetSentenceSegBatch(Matrix<ElemType> &sentenceBegin, vector<MinibatchPackingFlag>& minibatchPackingFlag)
    {
        sentenceBegin.SetValue(m_sentenceBegin);
        minibatchPackingFlag = m_minibatchPackingFlag;
    }

    // For Kaldi2Reader, we now make the following assumptions
    // 1. feature sections will always have a sub-field "scpFile"
    // 2. label sections will always have a sub-field "mlfFile"
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

    template class HTKMLFReader<float>;
    template class HTKMLFReader<double>;
}}}
