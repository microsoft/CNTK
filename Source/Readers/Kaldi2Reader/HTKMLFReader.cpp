//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"
#include "basetypes.h"

#include "htkfeatio.h" // for reading HTK features

#include "rollingwindowsource.h" // minibatch sources
#include "utterancesourcemulti.h"
#ifdef _WIN32
#include "readaheadsource.h"
#endif
#include "chunkevalsource.h"
#include "minibatchiterator.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "HTKMLFReader.h"
#include "Config.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

#ifdef __unix__
#include <limits.h>
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UNINT32;
#endif
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this

//#include <iostream>

//int msra::numa::node_override = -1;     // for numahelpers.h

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    m_mbiter = NULL;
    m_frameSource = NULL;
    m_lattices = NULL;
    m_seqTrainDeriv = NULL;
    m_uttDerivBuffer = NULL;
    m_minibatchBuffer.resize(0);
    m_minibatchBufferIndex = 0;
    m_noData = false;
    m_convertLabelsToTargets = false;
    m_doSeqTrain = false;
    m_getMinibatchCopy = false;
    m_doMinibatchBuffering = false;
    m_doMinibatchBufferTruncation = false;

    if (readerConfig.Exists(L"legacyMode"))
    {
        RuntimeError("legacy mode has been deprecated\n");
    }

    // If <m_framemode> is false, throw away any utterance that is longer
    // than the specified <m_maxUtteranceLength>.
    m_maxUtteranceLength = readerConfig(L"maxUtteranceLength", 10000);

    // m_truncated:
    //     If true, truncate utterances to fit the minibatch size. Otherwise
    //     the actual minibatch size will be the length of the utterance.
    // m_numberOfuttsPerMinibatch:
    //     If larger than one, then each minibatch contains multiple
    //     utterances.
    m_truncated = readerConfig(L"Truncated", false);
    m_numberOfuttsPerMinibatch = readerConfig(L"nbruttsineachrecurrentiter", 1);
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
    if (readerConfig.Exists(L"seqTrainCriterion"))
    {
        m_doSeqTrain = true;
        m_seqTrainCriterion = (const wstring&) readerConfig(L"seqTrainCriterion", L"");
        if ((m_seqTrainCriterion != L"mpfe") && (m_seqTrainCriterion != L"smbr"))
        {
            LogicError("Current Supported sequence training criterion are: mpfe, smbr.\n");
        }
    }

    // Checks if framemode is false in sequence training.
    m_framemode = readerConfig(L"frameMode", true);
    if (m_framemode && m_doSeqTrain)
    {
        LogicError("frameMode has to be false in sequence training.\n");
    }

    // Checks if partial minibatches are allowed.
    std::string minibatchMode(readerConfig(L"minibatchMode", "Partial"));
    m_partialMinibatch = EqualCI(minibatchMode, "Partial");

    // Figures out if we have to do minibatch buffering and how.
    if (m_doSeqTrain)
    {
        m_doMinibatchBuffering = true;
        if (m_truncated)
        {
            m_truncated = false;
            m_doMinibatchBufferTruncation = true;
        }
    }

    // Checks if we are in "write" mode or "train/test" mode.
    wstring command(readerConfig(L"action", L""));
    if (command == L"write")
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

template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::PrepareForSequenceTraining(const ConfigRecordType& readerConfig)
{
    // Parameters that we are looking for.
    wstring denlatRspecifier, aliRspecifier, transModelFilename, silencePhoneStr;
    ElemType oldAcousticScale;
    ElemType acousticScale;
    ElemType lmScale;
    bool oneSilenceClass;

    // Makes sure that "denlats" and "alignments" sections exist.
    if (!readerConfig.Exists(L"denlats"))
    {
        LogicError("Sequence training requested, but \"denlats\" section is not provided.\n");
    }
    if (!readerConfig.Exists(L"alignments"))
    {
        LogicError("Sequence training requested, but \"alignments\" section is not provided.\n");
    }

    // Processes "denlats" section.
    const ConfigRecordType& denlatConfig = readerConfig(L"denlats");
    if (!denlatConfig.Exists(L"rx"))
    {
        LogicError("Rspecifier is not provided for denominator lattices.\n");
    }
    if (!denlatConfig.Exists(L"kaldiModel"))
    {
        LogicError("Rspecifier is not provided for Kaldi model.\n");
    }
    denlatRspecifier = (const wstring&) denlatConfig(L"rx");
    transModelFilename = (const wstring&) (denlatConfig(L"kaldiModel"));
    silencePhoneStr = (const wstring&) (denlatConfig(L"silPhoneList", L""));
    oldAcousticScale = denlatConfig(L"oldAcousticScale", 0.0);
    acousticScale = denlatConfig(L"acousticScale", 0.2);
    lmScale = denlatConfig(L"lmScale", 1.0);
    oneSilenceClass = denlatConfig(L"oneSilenceClass", true);

    // Processes "alignments" section.
    const ConfigRecordType& aliConfig = readerConfig(L"alignments");
    if (!aliConfig.Exists(L"rx"))
    {
        LogicError("Rspecifier is not provided for alignments.\n");
    }
    aliRspecifier = (const wstring&) (aliConfig(L"rx"));

    // Scans the configurations to get "readerDeriv" type input and
    // "readerObj" type input. Both are feature nodes, we feed derivatives
    // to training criterion node through "readerDeriv" and feed objective
    // through "readerObj".
    bool hasDrive = false, hasObj = false;
    // for (auto iter = readerConfig.begin(); iter != readerConfig.end(); ++iter)
    for (const auto& id : readerConfig.GetMemberIds())
    {
        const ConfigRecordType& temp = readerConfig(id);
        if (temp.ExistsCurrent(L"type"))
        {
            wstring type = temp(L"type");
            if (EqualCI(type, L"readerDeriv") || EqualCI(type, L"seqTrainDeriv") /*for back compatibility */)
            {
                m_nameToTypeMap[id] = InputOutputTypes::readerDeriv;
                hasDrive = true;
            }
            else if (EqualCI(type, L"readerObj") || EqualCI(type, L"seqTrainObj") /*for back compatibility */)
            {
                m_nameToTypeMap[id] = InputOutputTypes::readerObj;
                hasObj = true;
            }
        }
    }
    if (!hasDrive || !hasObj)
    {
        LogicError("Missing readerDeriv or readerObj type feature\n");
    }

    // Initializes sequence training interface.
    m_seqTrainDeriv = new KaldiSequenceTrainingDerivative<ElemType>(
        denlatRspecifier, aliRspecifier, transModelFilename,
        silencePhoneStr, m_seqTrainCriterion, oldAcousticScale,
        acousticScale, lmScale, oneSilenceClass);

    // Initializes derivative buffering.
    m_doMinibatchBuffering = true;
    if (m_uttDerivBuffer != NULL)
    {
        LogicError("Derivative buffer has already been set, are you doing "
                   "sequence with some other metric that using derivative "
                   "buffering?\n");
    }
    m_uttDerivBuffer = new UtteranceDerivativeBuffer<ElemType>(
        m_numberOfuttsPerMinibatch, m_seqTrainDeriv);
}

// Loads input and output data for training and testing. Below we list the
// categories for different input/output:
// features:      InputOutputTypes::real
// labels:        InputOutputTypes::category
// derivatives:   InputOutputTypes::readerDeriv
// objectives:    InputOutputTypes::readerObj
//
// Note that we treat <derivatives> and <objectives> as features, but they
// will be computed in the reader, rather then reading from disks. Those
// will then be fed to training criterion node for training purposes.
template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::PrepareForTrainingOrTesting(const ConfigRecordType& readerConfig)
{
    // Loads files for sequence training.
    if (m_doSeqTrain)
    {
        PrepareForSequenceTraining(readerConfig);
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
    vector<msra::asr::FeatureSection*>& scriptpaths = m_trainingOrTestingFeatureSections;
    foreach_index (i, featureNames)
    {
        const ConfigRecordType& thisFeature = readerConfig(featureNames[i]);
        m_featDims.push_back(thisFeature(L"dim"));
        intargvector contextWindow = thisFeature(L"contextWindow", ConfigRecordType::Array(intargvector(vector<int>{1})));
        if (contextWindow.size() == 1) // symmetric
        {
            size_t windowFrames = contextWindow[0];
            if (windowFrames % 2 == 0)
                RuntimeError("augmentationextent: neighbor expansion of input features to %d not symmetrical", (int) windowFrames);
            size_t context = windowFrames / 2; // extend each side by this
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
            RuntimeError("contextFrames must have 1 or 2 values specified, found %d", (int) contextWindow.size());
        }

        // Figures the actual feature dimension, with context.
        m_featDims[i] = m_featDims[i] * (1 + numContextLeft[i] + numContextRight[i]);

        // Figures out the category.
        wstring type = thisFeature(L"type", L"real");
        if (EqualCI(type, L"real"))
        {
            m_nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
        }
        else
        {
            InvalidArgument("feature type must be 'real'");
        }
        m_featureNameToIdMap[featureNames[i]] = iFeat;
        assert(iFeat == m_featureIdToNameMap.size());
        m_featureIdToNameMap.push_back(featureNames[i]);
        scriptpaths.push_back(new msra::asr::FeatureSection(thisFeature(L"scpFile"), thisFeature(L"rx"), thisFeature(L"featureTransform", L"")));
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
    foreach_index (i, labelNames)
    {
        const ConfigRecordType& thisLabel = readerConfig(labelNames[i]);

        // Figures out label dimension.
        if (thisLabel.Exists(L"labelDim"))
            m_labelDims.push_back(thisLabel(L"labelDim"));
        else if (thisLabel.Exists(L"dim"))
            m_labelDims.push_back(thisLabel(L"dim"));
        else
            InvalidArgument("labels must specify dim or labelDim");

        // Figures out the category.
        wstring type;
        if (thisLabel.Exists(L"labelType"))
            type = (const wstring&) thisLabel(L"labelType"); // let's deprecate this eventually and just use "type"...
        else
            type = (const wstring&) thisLabel(L"type", L"category"); // outputs should default to category
        if (EqualCI(type, L"category"))
            m_nameToTypeMap[labelNames[i]] = InputOutputTypes::category;
        else
            InvalidArgument("label type must be Category");

        // Loads label mapping.
        statelistpaths.push_back(thisLabel(L"labelMappingFile", L""));

        m_labelNameToIdMap[labelNames[i]] = iLabel;
        assert(iLabel == m_labelIdToNameMap.size());
        m_labelIdToNameMap.push_back(labelNames[i]);
        m_labelNameToDimMap[labelNames[i]] = m_labelDims[i];
        mlfpaths.clear();
        mlfpaths.push_back(thisLabel(L"mlfFile"));
        mlfpathsmulti.push_back(mlfpaths);

        m_labelsBufferMultiIO.push_back(NULL);
        m_labelsBufferAllocatedMultiIO.push_back(0);

        iLabel++;

        // Figures out label to target mapping.
        wstring labelToTargetMappingFile(thisLabel(L"labelToTargetMappingFile", L""));
        if (labelToTargetMappingFile != L"")
        {
            std::vector<std::vector<ElemType>> labelToTargetMap;
            m_convertLabelsToTargetsMultiIO.push_back(true);
            if (thisLabel.Exists(L"targetDim"))
            {
                m_labelNameToDimMap[labelNames[i]] = m_labelDims[i] = thisLabel(L"targetDim");
            }
            else
                RuntimeError("output must specify targetDim if labelToTargetMappingFile specified!");
            size_t targetDim = ReadLabelToTargetMappingFile(labelToTargetMappingFile, statelistpaths[i], labelToTargetMap);
            if (targetDim != m_labelDims[i])
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
        throw std::runtime_error(msra::strfun::strprintf("# of inputs files vs. # of inputs or # of output files vs # of outputs inconsistent\n"));

    // Loads randomization method.
    size_t randomize = randomizeAuto;
    if (readerConfig.Exists(L"randomize"))
    {
        const std::string& randomizeString = readerConfig(L"randomize");
        if (EqualCI(randomizeString, "none"))
        {
            randomize = randomizeNone;
        }
        else if (EqualCI(randomizeString, "auto"))
        {
            randomize = randomizeAuto;
        }
        else
        {
            randomize = readerConfig(L"randomize");
        }
    }

    // Open script files for features.
    size_t numFiles = 0;
    size_t firstfilesonly = SIZE_MAX; // set to a lower value for testing
    vector<wstring> filelist;
    vector<vector<wstring>> infilesmulti;
    foreach_index (i, scriptpaths)
    {
        filelist.clear();
        std::wstring scriptpath = scriptpaths[i]->scpFile;
        fprintf(stderr, "reading script file %S ...", scriptpath.c_str());
        size_t n = 0;
        for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly /*optimization*/;)
        {
            filelist.push_back(reader.wgetline());
            n++;
        }

        fprintf(stderr, " %lu entries\n", n);

        if (i == 0)
            numFiles = n;
        else if (n != numFiles)
            throw std::runtime_error(msra::strfun::strprintf("number of files in each scriptfile inconsistent (%d vs. %d)", numFiles, n));

        infilesmulti.push_back(filelist);
    }

    // Opens MLF files for labels.
    set<wstring> restrictmlftokeys;
    double htktimetoframe = 100000.0; // default is 10ms
    std::vector<std::map<std::wstring, std::vector<msra::asr::htkmlfentry>>> labelsmulti;
    int targets_delay = 0;
    if (readerConfig.Exists(L"targets_delay"))
    {
        targets_delay = readerConfig(L"targets_delay");
    }
    foreach_index (i, mlfpathsmulti)
    {
        msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>
            labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], htktimetoframe, targets_delay); // label MLF
        // get the temp file name for the page file
        labelsmulti.push_back(labels);
    }

    // Get the readMethod, default value is "blockRandomize", the other
    // option is "rollingWindow". We only support "blockRandomize" in
    // sequence training.
    std::string readMethod(readerConfig(L"readMethod", "blockRandomize"));
    if (EqualCI(readMethod, "blockRandomize"))
    {
        // construct all the parameters we don't need, but need to be passed to the constructor...
        std::pair<std::vector<wstring>, std::vector<wstring>> latticetocs;
        std::unordered_map<std::string, size_t> modelsymmap;
        
        // Note, we are actually not using <m_lattices>, the only reason we
        // kept it was because it was required by
        // <minibatchutterancesourcemulti>.
        m_lattices = new msra::dbn::latticesource(latticetocs, modelsymmap, L"");
        
        // now get the frame source. This has better randomization and doesn't create temp files
        m_frameSource = new msra::dbn::minibatchutterancesourcemulti(
            scriptpaths, infilesmulti, labelsmulti, m_featDims, m_labelDims,
            numContextLeft, numContextRight, randomize, *m_lattices, m_latticeMap, m_framemode);
    }
    else if (EqualCI(readMethod, "rollingWindow"))
    {
        // "rollingWindow" is not supported in sequence training.
        if (m_doSeqTrain)
        {
            LogicError("rollingWindow is not supported in sequence training.\n");
        }
        std::wstring pageFilePath;
        std::vector<std::wstring> pagePaths;
        if (readerConfig.Exists(L"pageFilePath"))
        {
            pageFilePath = (const wstring&) readerConfig(L"pageFilePath");

            // replace any '/' with '\' for compat with default path
            std::replace(pageFilePath.begin(), pageFilePath.end(), '/', '\\');
#ifdef _WIN32
            // verify path exists
            DWORD attrib = GetFileAttributes(pageFilePath.c_str());
            if (attrib == INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY))
                throw std::runtime_error("pageFilePath does not exist");
#endif
#ifdef __unix__
            struct stat statbuf;
            if (stat(wtocharpath(pageFilePath).c_str(), &statbuf) == -1)
            {
                RuntimeError("pageFilePath does not exist");
            }

#endif
        }
        else // using default temporary path
        {
#ifdef _WIN32
            pageFilePath.reserve(MAX_PATH);
            GetTempPath(MAX_PATH, &pageFilePath[0]);
#endif
#ifdef __unix__
            pageFilePath.reserve(PATH_MAX);
            pageFilePath = L"/tmp/temp.CNTK.XXXXXX";
#endif
        }

#ifdef _WIN32
        if (pageFilePath.size() > MAX_PATH - 14) // max length of input to GetTempFileName is PATH_MAX-14
            throw std::runtime_error(msra::strfun::strprintf("pageFilePath must be less than %d characters", MAX_PATH - 14));
#endif
#ifdef __unix__
        if (pageFilePath.size() > PATH_MAX - 14) // max length of input to GetTempFileName is PATH_MAX-14
            throw std::runtime_error(msra::strfun::strprintf("pageFilePath must be less than %d characters", PATH_MAX - 14));
#endif
        foreach_index (i, infilesmulti)
        {
#ifdef _WIN32
            wchar_t tempFile[MAX_PATH];
            GetTempFileName(pageFilePath.c_str(), L"CNTK", 0, tempFile);
            pagePaths.push_back(tempFile);
#endif
#ifdef __unix__
            char* tempFile;
            // GetTempFileName(pageFilePath.c_str(), L"CNTK", 0, tempFile);
            tempFile = (char*) pageFilePath.c_str();
            int fid = mkstemp(tempFile);
            unlink(tempFile);
            close(fid);
            pagePaths.push_back(GetWC(tempFile));
#endif
        }

        const bool mayhavenoframe = false;
        int addEnergy = 0;

        // m_frameSourceMultiIO = new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, randomize, pagepath, mayhavenoframe, addEnergy);
        // m_frameSourceMultiIO->setverbosity(verbosity);
        int verbosity = readerConfig(L"verbosity", 2);
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
template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::PrepareForWriting(const ConfigRecordType& readerConfig)
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
    vector<msra::asr::FeatureSection*>& scriptpaths = m_writingFeatureSections;
    foreach_index (i, featureNames)
    {
        ConfigParameters thisFeature = readerConfig(featureNames[i]);

        // Figures out the context.
        ConfigArray contextWindow = thisFeature("contextWindow", "1");
        if (contextWindow.size() == 1) // symmetric
        {
            size_t windowFrames = contextWindow[0];
            if (windowFrames % 2 == 0)
                RuntimeError("augmentationextent: neighbor expansion of input features to %d not symmetrical", (int) windowFrames);
            size_t context = windowFrames / 2; // extend each side by this
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
            RuntimeError("contextFrames must have 1 or 2 values specified, found %d", (int) contextWindow.size());
        }

        // Figures out the feature dimension, with context.
        realDims.push_back(thisFeature("dim"));
        realDims[i] = realDims[i] * (1 + numContextLeft[i] + numContextRight[i]);

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
        assert(iFeat == m_featureIdToNameMap.size());
        m_featureIdToNameMap.push_back(featureNames[i]);
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
    size_t firstfilesonly = SIZE_MAX; // set to a lower value for testing
    size_t evalchunksize = 2048;
    foreach_index (i, scriptpaths)
    {
        filelist.clear();
        std::wstring scriptpath = scriptpaths[i]->scpFile;
        fprintf(stderr, "reading script file %S ...", scriptpath.c_str());
        size_t n = 0;
        for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly /*optimization*/;)
        {
            filelist.push_back(reader.wgetline());
            n++;
        }

        fprintf(stderr, " %zu entries\n", n);

        if (i == 0)
            numFiles = n;
        else if (n != numFiles)
            throw std::runtime_error(msra::strfun::strprintf("HTKMLFReader::InitEvalReader: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles, n));

        m_inputFilesMultiIO.push_back(filelist);
    }

    m_fileEvalSource = new msra::dbn::FileEvalSource(realDims, numContextLeft, numContextRight, evalchunksize);
}

// destructor - virtual so it gets called properly
template <class ElemType>
HTKMLFReader<ElemType>::~HTKMLFReader()
{
    delete m_mbiter;
    delete m_frameSource;
    delete m_lattices;
    delete m_seqTrainDeriv;
    delete m_uttDerivBuffer;

    foreach_index(i, m_featuresBufferMultiIO)
        delete[] m_featuresBufferMultiIO[i];

    foreach_index(i, m_labelsBufferMultiIO)
        delete[] m_labelsBufferMultiIO[i];

    for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
    {
        delete[] m_featuresBufferMultiUtt[i];
        delete[] m_labelsBufferMultiUtt[i];
    }

    foreach_index (i, m_trainingOrTestingFeatureSections)
        delete m_trainingOrTestingFeatureSections[i];

    foreach_index (i, m_writingFeatureSections)
        delete m_writingFeatureSections[i];
}

// StartMinibatchLoop - Startup a minibatch loop
// mbSize - [in] size of the minibatch (number of frames, etc.)
// epoch - [in] epoch number for this loop
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template <class ElemType>
void HTKMLFReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    m_mbSize = mbSize;
    m_currentMBSize = mbSize;

    if (m_trainOrTest)
    {
        StartMinibatchLoopToTrainOrTest(mbSize, epoch, requestedEpochSamples);
    }
    else
    {
        StartMinibatchLoopToWrite(mbSize, epoch, requestedEpochSamples);
    }
    m_checkDictionaryKeys = true;
}

template <class ElemType>
void HTKMLFReader<ElemType>::StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    size_t datapasses = 1;
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
        else if (minibatches > 0) // if we have any full minibatches
        {
            // since we skip the extraFrames, we need to add them to the total to get the actual number of frames requested
            size_t sweeps = (requestedEpochSamples - 1) / totalFrames; // want the number of sweeps we will skip the extra, so subtract 1 and divide
            requestedEpochSamples += extraFrames * sweeps;
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
    size_t currentMBSize = (m_framemode == true) ? mbSize : 1;
    m_mbiter = new msra::dbn::minibatchiterator(*source, epoch, requestedEpochSamples, currentMBSize, datapasses);

    // Resets utterance derivative buffering class.
    if (m_doMinibatchBuffering)
    {
        assert(m_uttDerivBuffer != NULL);
        m_uttDerivBuffer->ResetEpoch();
    }

    // Clears minibatch buffer.
    m_minibatchBuffer.clear();
    m_getMinibatchCopy = false;
    m_minibatchBufferIndex = 0;
    m_uttInfo.clear();
    m_minibatchUttInfo.clear();

    // Clears feature and label buffer.
    if (!m_featuresBufferMultiIO.empty())
    {
        foreach_index (i, m_featuresBufferMultiIO)
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
        foreach_index (i, m_labelsBufferMultiIO)
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
    for (size_t u = 0; u < m_numberOfuttsPerMinibatch; u++)
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

template <class ElemType>
void HTKMLFReader<ElemType>::StartMinibatchLoopToWrite(size_t mbSize, size_t /*epoch*/, size_t /*requestedEpochSamples*/)
{
    m_fileEvalSource->Reset();
    m_fileEvalSource->SetMinibatchSize(mbSize);
    // m_chunkEvalSourceMultiIO->reset();
    m_inputFileIndex = 0;

    foreach_index (i, m_featuresBufferMultiIO)
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
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool HTKMLFReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
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
template <class ElemType>
bool HTKMLFReader<ElemType>::PopulateUtteranceInMinibatch(
    const StreamMinibatchInputs& matrices,
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

    size_t numOfFea = m_featuresBufferMultiIO.size();
    size_t numOfLabel = m_labelsBufferMultiIO.size();
    for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
    {
        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
        { 
            // Features.
            size_t id = m_featureNameToIdMap[iter->first];
            size_t dim = m_featureNameToDimMap[iter->first];

            if (m_featuresBufferMultiIO[id] == NULL)
            {
                m_featuresBufferMultiIO[id] = new ElemType[dim * mbSize * m_numberOfuttsPerMinibatch];
                m_featuresBufferAllocatedMultiIO[id] = dim * mbSize * m_numberOfuttsPerMinibatch;
            }
            else if (m_featuresBufferAllocatedMultiIO[id] < dim * mbSize * m_numberOfuttsPerMinibatch)
            { 
                // Buffer too small, we have to increase it.
                delete[] m_featuresBufferMultiIO[id];
                m_featuresBufferMultiIO[id] = new ElemType[dim * mbSize * m_numberOfuttsPerMinibatch];
                m_featuresBufferAllocatedMultiIO[id] = dim * mbSize * m_numberOfuttsPerMinibatch;
            }

            if (sizeof(ElemType) == sizeof(float))
            { 
                // For float, we copy entire column.
                for (size_t j = startFrame, k = 0; j < endFrame; j++, k++)
                {
                    memcpy_s(&m_featuresBufferMultiIO[id][((k + mbOffset) * m_numberOfuttsPerMinibatch + uttIndex) * dim],
                             sizeof(ElemType) * dim,
                             &m_featuresBufferMultiUtt[uttIndex][j * dim + m_featuresStartIndexMultiUtt[id + uttIndex * numOfFea]],
                             sizeof(ElemType) * dim);
                }
            }
            else
            { 
                // For double, we have to copy element by element.
                for (size_t j = startFrame, k = 0; j < endFrame; j++, k++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        m_featuresBufferMultiIO[id][((k + mbOffset) * m_numberOfuttsPerMinibatch + uttIndex) * dim + d] = 
                            m_featuresBufferMultiUtt[uttIndex][j * dim + d + m_featuresStartIndexMultiUtt[id + uttIndex * numOfFea]];
                    }
                }
            }
        }
        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
        { 
            // Labels.
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

            for (size_t j = startFrame, k = 0; j < endFrame; j++, k++)
            {
                for (int d = 0; d < dim; d++)
                {
                    m_labelsBufferMultiIO[id][((k + mbOffset) * m_numberOfuttsPerMinibatch + uttIndex) * dim + d] = 
                        m_labelsBufferMultiUtt[uttIndex][j * dim + d + m_labelsStartIndexMultiUtt[id + uttIndex * numOfLabel]];
                }
            }
        }
    }
    return success;
}

template <class ElemType>
bool HTKMLFReader<ElemType>::GetOneMinibatchToTrainOrTestDataBuffer(
    const StreamMinibatchInputs& matrices)
{
    bool skip = false;

    // On first minibatch, check if we have input for given names.
    if (m_checkDictionaryKeys)
    {
        for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
        {
            if (m_nameToTypeMap.find(iter->first) == m_nameToTypeMap.end())
            {
                throw std::runtime_error(msra::strfun::strprintf(
                    "minibatch requested for input node %S not found in"
                    "reader - cannot generate input\n",
                    iter->first.c_str()));
            }
        }
        m_checkDictionaryKeys = false;
    }

    // If we are doing utterance derivative buffering, we need to keep the
    // utterance information.
    if (m_doMinibatchBuffering)
    {
        m_minibatchUttInfo.assign(m_numberOfuttsPerMinibatch,
                                  std::vector<std::pair<wstring, size_t>>(0));

        // For the moment we don't support same utterance in the same
        // minibatch.
        m_hasUttInCurrentMinibatch.clear();
        for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
        {
            while (m_hasUttInCurrentMinibatch.find(m_uttInfo[i][0].first) != m_hasUttInCurrentMinibatch.end())
            {
                fprintf(stderr, "WARNING: Utterance \"%S\" already exists "
                                "in the minibatch, skipping it.\n",
                        m_uttInfo[i][0].first.c_str());
                ReNewBufferForMultiIO(i);
            }
            if (m_uttInfo[i].size() > 0)
            {
                m_hasUttInCurrentMinibatch[m_uttInfo[i][0].first] = true;
            }
        }
    }

    m_currentMBSize = m_mbSize;
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

        // If <m_truncated> is true, <m_currentMBSize> is <m_mbSize>
        // If <m_truncated> is false, <m_currentMBSize> equals to the longest
        // utterance in the minibatch.
        if (!m_truncated)
        {
            m_currentMBSize = 0;
            for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
            {
                if (m_currentBufferFrames[i] > m_currentMBSize)
                {
                    m_currentMBSize = m_currentBufferFrames[i];
                }
            }
        }

        // We initialize the sentence boundary information before we process
        // the utterances.
        if (m_framemode)
        {
            assert(m_numberOfuttsPerMinibatch == 1);
            m_pMBLayout->InitAsFrameMode(m_currentMBSize);
        }
        else
        {
            m_pMBLayout->Init(m_numberOfuttsPerMinibatch, m_currentMBSize);
        }
        /*for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
            {
                for (size_t j = 0; j < m_currentMBSize; j++)
                {
                    m_pMBLayout->SetWithoutOr(i, j, MinibatchPackingFlags::None);
                }
            }*/

        // Iterates over utterances. m_numberOfuttsPerMinibatch = 1 is a
        // special case.
        for (size_t i = 0; i < m_numberOfuttsPerMinibatch; i++)
        {
            size_t startFrame = m_processedFrame[i];
            size_t endFrame = 0;
            // Sets the utterance boundary.
            if (!m_framemode)
            {
                if (m_toProcess[i] > startFrame)
                {
                    m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, i, -(ptrdiff_t) startFrame, m_toProcess[i] - startFrame);
                }
            }
            // m_pMBLayout->Set(i, 0, MinibatchPackingFlags::SequenceStart);

            if ((startFrame + m_currentMBSize) < m_toProcess[i])
            {
                // There is only 1 case:
                //     1. <m_framemode> is false, and <m_truncated> is true.
                assert(m_framemode == false);
                assert(m_truncated == true);

                endFrame = startFrame + m_currentMBSize;
                bool populateSucc = PopulateUtteranceInMinibatch(matrices, i, startFrame, endFrame, m_currentMBSize);
                if (m_doMinibatchBuffering && populateSucc)
                {
                    m_minibatchUttInfo[i].push_back(m_uttInfo[i][0]);
                    m_hasUttInCurrentMinibatch[m_uttInfo[i][0].first] = true;
                }
                m_processedFrame[i] += m_currentMBSize;
            }
            else if ((startFrame + m_currentMBSize) == m_toProcess[i])
            {
                // There are 3 cases:
                //     1. <m_framemode> is false, and <m_truncated> is true,
                //        and it reaches the end of the utterance.
                //     2. <m_framemode> is false, and <m_truncated> is false
                //        and it reaches the end of the utterance.
                //     3. <m_framemode> is true, then we do not have to set
                //        utterance boundary.

                // Sets the utterance boundary.
                /*if (m_framemode == false)
                    {
                        if (startFrame == 0)
                        {
                            m_pMBLayout->Set(i, 0, MinibatchPackingFlags::SequenceStart);
                        }

                        // We have to set the utterance end.
                        m_pMBLayout->Set(i, m_pMBLayout->GetNumTimeSteps() - 1, MinibatchPackingFlags::SequenceEnd);
                    }*/

                // Now puts the utterance into the minibatch, and loads the
                // next one.
                endFrame = startFrame + m_currentMBSize;
                bool populateSucc = PopulateUtteranceInMinibatch(matrices, i, startFrame, endFrame, m_currentMBSize);
                if (m_doMinibatchBuffering && populateSucc)
                {
                    m_minibatchUttInfo[i].push_back(m_uttInfo[i][0]);
                    m_hasUttInCurrentMinibatch[m_uttInfo[i][0].first] = true;
                }
                m_processedFrame[i] += m_currentMBSize;
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
                    m_pMBLayout->AddGap(i, 0, m_currentMBSize);
                    for (size_t k = 0; k < m_currentMBSize; k++)
                    {
                        // m_pMBLayout->Set(i, k, MinibatchPackingFlags::NoInput);

                        // Populates <NO_INPUT> with real features, the
                        // following implementation is not efficient...
                        PopulateUtteranceInMinibatch(matrices, i, 0, 1, m_currentMBSize, k);
                    }
                    continue;
                }

                // First, if <m_framemode> is true, then it must be a
                // partial minibatch, and if that is not allowed, we have to
                // skip this minibatch.
                if (m_framemode && !m_partialMinibatch)
                {
                    skip = true;
                    bool reNewSucc = ReNewBufferForMultiIO(i); // Should return false?
                    continue;
                }

                // Second, we set utterance boundary for the partial
                // minibatch, and then load it.
                /*  if (m_framemode == false)
                    {
                        if (startFrame == 0)
                        {
                            m_pMBLayout->Set(i, 0, MinibatchPackingFlags::SequenceStart);
                        }

                        // We have to set the utterance end.
                        assert(m_toProcess[i] - startFrame - 1 < m_pMBLayout->GetNumTimeSteps());
                        m_pMBLayout->Set(i, m_toProcess[i] - startFrame - 1, MinibatchPackingFlags::SequenceEnd);
                    }*/
                endFrame = m_toProcess[i];
                size_t currentMBFilled = endFrame - startFrame;
                bool populateSucc = PopulateUtteranceInMinibatch(matrices, i, startFrame, endFrame, m_currentMBSize);
                if (m_doMinibatchBuffering && populateSucc)
                {
                    m_minibatchUttInfo[i].push_back(m_uttInfo[i][0]);
                    m_hasUttInCurrentMinibatch[m_uttInfo[i][0].first] = true;
                }
                m_processedFrame[i] += currentMBFilled;
                bool reNewSucc = ReNewBufferForMultiIO(i);

                // Third, if the next utterance can fit into the current
                // minibatch, we also pack the next utterance.
                while (reNewSucc && (currentMBFilled + m_toProcess[i] <= m_currentMBSize))
                {
                    // Sets the utterance boundary.
                    assert(currentMBFilled + m_toProcess[i] <= m_pMBLayout->GetNumTimeSteps());
                    m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, i, currentMBFilled, currentMBFilled + m_toProcess[i]);
                    // m_pMBLayout->Set(i, currentMBFilled, MinibatchPackingFlags::SequenceStart);
                    // m_pMBLayout->Set(i, currentMBFilled + m_toProcess[i] - 1, MinibatchPackingFlags::SequenceEnd);
                    populateSucc = PopulateUtteranceInMinibatch(matrices, i, 0, m_toProcess[i], m_currentMBSize, currentMBFilled);
                    if (m_doMinibatchBuffering && populateSucc)
                    {
                        m_minibatchUttInfo[i].push_back(m_uttInfo[i][0]);
                        m_hasUttInCurrentMinibatch[m_uttInfo[i][0].first] = true;
                    }
                    assert(m_processedFrame[i] == 0);
                    m_processedFrame[i] = m_toProcess[i];
                    currentMBFilled += m_toProcess[i];
                    reNewSucc = ReNewBufferForMultiIO(i);
                }

                // Finally, pulls frames from next utterance if the current
                // minibatch is not full.
                if (reNewSucc && !m_framemode && m_truncated)
                {
                    populateSucc = PopulateUtteranceInMinibatch(matrices, i, 0, m_currentMBSize - currentMBFilled, m_currentMBSize, currentMBFilled);
                    if (m_doMinibatchBuffering && populateSucc)
                    {
                        m_minibatchUttInfo[i].push_back(m_uttInfo[i][0]);
                        m_hasUttInCurrentMinibatch[m_uttInfo[i][0].first] = true;
                    }
                    m_processedFrame[i] += m_currentMBSize - currentMBFilled;
                    if (currentMBFilled < m_currentMBSize)
                    {
                        m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, i, currentMBFilled, currentMBFilled + m_toProcess[i]);
                        // m_pMBLayout->Set(i, currentMBFilled, MinibatchPackingFlags::SequenceStart);
                    }
                }
                else
                {
                    m_pMBLayout->AddGap(i, currentMBFilled, m_currentMBSize);
                    for (size_t k = currentMBFilled; k < m_currentMBSize; k++)
                    {
                        // m_pMBLayout->Set(i, k, MinibatchPackingFlags::NoInput);

                        // Populates <NO_INPUT> with real features, the
                        // following implementation is not efficient...
                        PopulateUtteranceInMinibatch(matrices, i, 0, 1, m_currentMBSize, k);
                    }
                }
            }
        }

        skip = false;
    } while (skip);

    return true;
}

template <class ElemType>
bool HTKMLFReader<ElemType>::ShouldCopyMinibatchFromBuffer()
{
    if (m_doMinibatchBuffering)
    {
        // If <m_getMinibatchCopy> is false, then we should copy data from
        // buffer for back-propagation.
        if (m_getMinibatchCopy == false && m_minibatchBuffer.size() > 0)
        {
            m_minibatchBufferIndex = 0;
            return true;
        }

        // If <m_getMinibatchCopy> is true, we first have to re-compute
        // the likelihood for the frames that are already in the buffer.
        if (m_getMinibatchCopy == true && m_minibatchBufferIndex + 1 < m_minibatchBuffer.size())
        {
            m_minibatchBufferIndex += 1;
            return true;
        }
    }

    return false;
}

template <class ElemType>
void HTKMLFReader<ElemType>::CopyMinibatchToBuffer()
{
    size_t originalMBSize = m_currentMBSize;
    size_t currentMBSize = m_currentMBSize;
    size_t numMinibatches = 1;
    if (m_doMinibatchBufferTruncation)
    {
        currentMBSize = m_mbSize;
        numMinibatches = (ElemType) originalMBSize / (ElemType) m_mbSize;
        numMinibatches += (originalMBSize % m_mbSize == 0) ? 0 : 1;
    }

    for (size_t i = 0; i < numMinibatches; ++i)
    {
        MinibatchBufferUnit currentMinibatch;

        size_t startIndex = i * currentMBSize;
        size_t numFrames =
            (startIndex + currentMBSize <= originalMBSize) ? currentMBSize : (originalMBSize - startIndex);

        // Sets MBLayout.
        // currentMinibatch.pMBLayout->CopyFromRange(m_pMBLayout, startIndex, numFrames);
        currentMinibatch.pMBLayout->Init(m_pMBLayout->GetNumParallelSequences(), numFrames);
        const auto& sequences = m_pMBLayout->GetAllSequences();
        for (const auto& seq : sequences)
        {
            if (seq.tEnd > startIndex && seq.tBegin < (ptrdiff_t)(startIndex + numFrames))
            {
                auto shiftedSeq = seq;
                shiftedSeq.tBegin -= startIndex;
                shiftedSeq.tEnd -= startIndex;
                currentMinibatch.pMBLayout->AddSequence(shiftedSeq);
            }
        }
        // Sets the minibatch size for the current minibatch.
        currentMinibatch.currentMBSize = numFrames;

        // Sets the utterance information for the current minibatch.
        currentMinibatch.minibatchUttInfo.assign(
            m_numberOfuttsPerMinibatch,
            std::vector<std::pair<wstring, size_t>>(0));
        for (size_t j = 0; j < m_minibatchUttInfo.size(); ++j)
        {
            size_t uttStartIndex = 0;
            for (size_t k = 0; k < m_minibatchUttInfo[j].size(); ++k)
            {
                if (startIndex >= uttStartIndex + m_minibatchUttInfo[j][k].second)
                {
                    uttStartIndex += m_minibatchUttInfo[j][k].second;
                    continue;
                }
                if (startIndex + numFrames <= uttStartIndex)
                {
                    break;
                }
                currentMinibatch.minibatchUttInfo[j].push_back(m_minibatchUttInfo[j][k]);
                uttStartIndex += m_minibatchUttInfo[j][k].second;
            }
        }

        size_t startDataCopy = startIndex * m_numberOfuttsPerMinibatch;
        size_t endDataCopy = (startIndex + numFrames) * m_numberOfuttsPerMinibatch;

        // Copies features.
        currentMinibatch.features.resize(0);
        for (size_t i = 0; i < m_featuresBufferMultiIO.size(); ++i)
        {
            std::vector<ElemType> tmpFeatures(
                m_featuresBufferMultiIO[i] + startDataCopy * m_featureNameToDimMap[m_featureIdToNameMap[i]],
                m_featuresBufferMultiIO[i] + endDataCopy * m_featureNameToDimMap[m_featureIdToNameMap[i]]);
            currentMinibatch.features.push_back(tmpFeatures);
        }

        // Copies labels.
        currentMinibatch.labels.resize(0);
        for (size_t i = 0; i < m_labelsBufferMultiIO.size(); ++i)
        {
            std::vector<ElemType> tmpLabels(
                m_labelsBufferMultiIO[i] + startDataCopy * m_labelNameToDimMap[m_labelIdToNameMap[i]],
                m_labelsBufferMultiIO[i] + endDataCopy * m_labelNameToDimMap[m_labelIdToNameMap[i]]);
            currentMinibatch.labels.push_back(tmpLabels);
        }

        m_minibatchBuffer.push_back(currentMinibatch);
    }
}

template <class ElemType>
void HTKMLFReader<ElemType>::CopyMinibatchFromBufferToMatrix(
    size_t index,
    StreamMinibatchInputs& matrices)
{
    assert(m_minibatchBuffer.size() > index);

    // Restores the variables related to the minibatch.
    m_pMBLayout->CopyFrom(m_minibatchBuffer[index].pMBLayout);
    m_currentMBSize = m_minibatchBuffer[index].currentMBSize;
    m_minibatchUttInfo = m_minibatchBuffer[index].minibatchUttInfo;

    // Copies data to the matrix.
    for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
    {
        Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first);
        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
        {
            size_t id = m_featureNameToIdMap[iter->first];
            size_t dim = m_featureNameToDimMap[iter->first];
            assert(id < m_minibatchBuffer[index].features.size());
            data.SetValue(dim,
                          m_minibatchBuffer[index].features[id].size() / dim,
                          data.GetDeviceId(),
                          m_minibatchBuffer[index].features[id].data(),
                          matrixFlagNormal);
        }
        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
        {
            size_t id = m_labelNameToIdMap[iter->first];
            size_t dim = m_labelNameToDimMap[iter->first];
            assert(id < m_minibatchBuffer[index].labels.size());
            data.SetValue(dim,
                          m_minibatchBuffer[index].labels[id].size() / dim,
                          data.GetDeviceId(),
                          m_minibatchBuffer[index].labels[id].data(),
                          matrixFlagNormal);
        }
        else if (m_doMinibatchBuffering)
        {
            if (m_nameToTypeMap[iter->first] == InputOutputTypes::readerDeriv)
            {
                if (m_getMinibatchCopy)
                {
                    assert(m_currentMBSize * m_numberOfuttsPerMinibatch == m_pMBLayout->GetNumCols());
                    if (data.GetNumCols() != m_pMBLayout->GetNumCols())
                    {
                        data.Resize(data.GetNumRows(), m_pMBLayout->GetNumCols());
                    }
                    matrices.GetInputMatrix<ElemType>(iter->first).SetValue(0);
                }
                else
                {
                    m_uttDerivBuffer->GetDerivative(m_minibatchUttInfo,
                                                    m_pMBLayout,
                                                    &matrices.GetInputMatrix<ElemType>(iter->first)); // TODO: use a reference instead of a ptr
                }
            }
            else if (m_nameToTypeMap[iter->first] == InputOutputTypes::readerObj)
            {
                if (m_getMinibatchCopy)
                {
                    assert(m_currentMBSize * m_numberOfuttsPerMinibatch == m_pMBLayout->GetNumCols());
                    if (data.GetNumCols() != m_pMBLayout->GetNumCols())
                    {
                        data.Resize(1, m_pMBLayout->GetNumCols());
                    }
                    data.SetValue(0);
                }
                else
                {
                    m_uttDerivBuffer->GetObjective(m_minibatchUttInfo,
                                                   m_pMBLayout,
                                                   &matrices.GetInputMatrix<ElemType>(iter->first)); // TODO: use a reference instead of a ptr
                }
            }
        }
    }

    if (m_getMinibatchCopy == false)
    {
        assert(index == 0);
        m_minibatchBuffer.pop_front();
    }
}

template <class ElemType>
void HTKMLFReader<ElemType>::CopyMinibatchToMatrix(
    size_t size,
    const vector<ElemType*>& featureBuffer,
    const vector<ElemType*>& labelBuffer,
    StreamMinibatchInputs& matrices) const
{
    for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
    {
        Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first);
        if (m_nameToTypeMap.at(iter->first) == InputOutputTypes::real)
        {
            size_t id = m_featureNameToIdMap.at(iter->first);
            size_t dim = m_featureNameToDimMap.at(iter->first);
            assert(id < featureBuffer.size());
            data.SetValue(dim, size, data.GetDeviceId(), featureBuffer[id], matrixFlagNormal);
        }
        else if (m_nameToTypeMap.at(iter->first) == InputOutputTypes::category)
        {
            size_t id = m_labelNameToIdMap.at(iter->first);
            size_t dim = m_labelNameToDimMap.at(iter->first);
            assert(id < labelBuffer.size());
            data.SetValue(dim, size, data.GetDeviceId(), labelBuffer[id], matrixFlagNormal);
        }
        else if (m_doMinibatchBuffering)
        {
            if (m_nameToTypeMap.at(iter->first) == InputOutputTypes::readerDeriv)
            {
                assert(m_currentMBSize * m_numberOfuttsPerMinibatch == m_pMBLayout->GetNumCols());
                if (data.GetNumCols() != m_pMBLayout->GetNumCols())
                {
                    data.Resize(data.GetNumRows(), m_pMBLayout->GetNumCols());
                }
                data.SetValue(0);
            }
            else if (m_nameToTypeMap.at(iter->first) == InputOutputTypes::readerObj)
            {
                assert(m_currentMBSize * m_numberOfuttsPerMinibatch == m_pMBLayout->GetNumCols());
                if (data.GetNumCols() != m_pMBLayout->GetNumCols())
                {
                    data.Resize(1, m_pMBLayout->GetNumCols());
                }
                data.SetValue(0);
            }
        }
    }
}

template <class ElemType>
bool HTKMLFReader<ElemType>::GetMinibatchToTrainOrTest(
    StreamMinibatchInputs& matrices)
{
    // We either copy a new minibatch from buffer or read one from minibatch
    // iterator.
    bool success = false;
    if (ShouldCopyMinibatchFromBuffer())
    {
        CopyMinibatchFromBufferToMatrix(m_minibatchBufferIndex, matrices);
        return true;
    }
    else
    {
        success = GetOneMinibatchToTrainOrTestDataBuffer(matrices);
        if (success)
        {
            if (m_getMinibatchCopy)
            {
                assert(m_minibatchBuffer.size() == 0);
                CopyMinibatchToBuffer();
                CopyMinibatchFromBufferToMatrix(0, matrices);
                m_minibatchBufferIndex = 0;
            }
            else
            {
                CopyMinibatchToMatrix(
                    m_currentMBSize * m_numberOfuttsPerMinibatch,
                    m_featuresBufferMultiIO, m_labelsBufferMultiIO, matrices);
            }
        }

        // If we are in the "copy" mode, and we cannot get a full minibatch,
        // then we have computed the posteriors for all the minibatches.
        if (m_doMinibatchBuffering && !success && m_getMinibatchCopy)
        {
            m_uttDerivBuffer->SetEpochEnd();
        }

        return success;
    }

    return false;
}

template <class ElemType>
bool HTKMLFReader<ElemType>::GetMinibatchToWrite(StreamMinibatchInputs& matrices)
{
    std::map<std::wstring, size_t>::iterator iter;
    if (m_checkDictionaryKeys)
    {
        for (auto iter = m_featureNameToIdMap.begin(); iter != m_featureNameToIdMap.end(); iter++)
        {
            if (matrices.find(iter->first) == matrices.end())
            {
                fprintf(stderr, "GetMinibatchToWrite: feature node %S specified in reader not found in the network\n", iter->first.c_str());
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

    if (m_inputFileIndex < m_inputFilesMultiIO[0].size())
    {
        m_fileEvalSource->Reset();

        // load next file (or set of files)
        foreach_index (i, m_inputFilesMultiIO)
        {
            msra::asr::htkfeatreader reader;

            const auto path = reader.parse(m_inputFilesMultiIO[i][m_inputFileIndex], m_writingFeatureSections[i]);
            // read file
            msra::dbn::matrix feat;
            string featkind;
            unsigned int sampperiod;
            msra::util::attempt(5, [&]()
                                {
                                    reader.readAlloc(path, featkind, sampperiod, feat); // whole file read as columns of feature vectors
                                });
            fprintf(stderr, "evaluate: reading %zu frames of %S\n", feat.cols(), ((wstring) path).c_str());
            m_fileEvalSource->AddFile(feat, featkind, sampperiod, i);
        }
        m_inputFileIndex++;

        // turn frames into minibatch (augment neighbors, etc)
        m_fileEvalSource->CreateEvalMinibatch();

        // populate input matrices
        bool first = true;
        for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
        {
            // dereference matrix that corresponds to key (input/output name) and
            // populate based on whether its a feature or a label

            if (m_nameToTypeMap.find(iter->first) != m_nameToTypeMap.end() && m_nameToTypeMap[iter->first] == InputOutputTypes::real)
            {
                Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first); // can be features or labels
                size_t id = m_featureNameToIdMap[iter->first];
                size_t dim = m_featureNameToDimMap[iter->first];

                const msra::dbn::matrix feat = m_fileEvalSource->ChunkOfFrames(id);
                if (first)
                {
                    m_pMBLayout->Init(1, feat.cols());
                    m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, feat.cols());
                    // m_pMBLayout->SetWithoutOr(0, feat.cols() - 1, MinibatchPackingFlags::SequenceEnd);
                    first = false;
                }

                // copy the features over to our array type
                assert(feat.rows() == dim);
                dim; // check feature dimension matches what's expected

                if (m_featuresBufferMultiIO[id] == NULL)
                {
                    m_featuresBufferMultiIO[id] = new ElemType[feat.rows() * feat.cols()];
                    m_featuresBufferAllocatedMultiIO[id] = feat.rows() * feat.cols();
                }
                else if (m_featuresBufferAllocatedMultiIO[id] < feat.rows() * feat.cols()) // buffer size changed. can be partial minibatch
                {
                    delete[] m_featuresBufferMultiIO[id];
                    m_featuresBufferMultiIO[id] = new ElemType[feat.rows() * feat.cols()];
                    m_featuresBufferAllocatedMultiIO[id] = feat.rows() * feat.cols();
                }
                // shouldn't need this since we fill up the entire buffer below
                // memset(m_featuresBufferMultiIO[id],0,sizeof(ElemType)*feat.rows()*feat.cols());

                if (sizeof(ElemType) == sizeof(float))
                {
                    for (int j = 0; j < feat.cols(); j++) // column major, so iterate columns
                    {
                        // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                        memcpy_s(&m_featuresBufferMultiIO[id][j * feat.rows()], sizeof(ElemType) * feat.rows(), &feat(0, j), sizeof(ElemType) * feat.rows());
                    }
                }
                else
                {
                    for (int j = 0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
                    {
                        for (int i = 0; i < feat.rows(); i++)
                        {
                            m_featuresBufferMultiIO[id][j * feat.rows() + i] = feat(i, j);
                        }
                    }
                }
                data.SetValue(feat.rows(), feat.cols(), data.GetDeviceId(), m_featuresBufferMultiIO[id], matrixFlagNormal);
            }
            else
            { // Resizes other inputs so they won't affect actual minibatch size.
                Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first);
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

template <class ElemType>
bool HTKMLFReader<ElemType>::ReNewBufferForMultiIO(size_t i)
{
    if (m_noData)
    {
        m_currentBufferFrames[i] = 0;
        m_processedFrame[i] = 0;
        m_toProcess[i] = 0;
        m_uttInfo[i].clear();
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
            fprintf(stderr, "WARNING: Utterance length is smaller than the "
                            "minibatch size, you may want to remove the utterance or "
                            "reduce the minibatch size.\n");
        }

        if (!m_truncated && (m_currentBufferFrames[i] > m_maxUtteranceLength))
        {
            (*m_mbiter)++;
            if (!(*m_mbiter))
            {
                m_noData = true;
            }
            fprintf(stderr, "WARNING: Utterance \"%S\" has length longer "
                            "than the %zd, skipping it.\n",
                    m_uttInfo[i][0].first.c_str(), m_maxUtteranceLength);
            return ReNewBufferForMultiIO(i);
        }

        if (m_doMinibatchBuffering && !m_uttDerivBuffer->HasResourceForDerivative(m_uttInfo[i][0].first))
        {
            (*m_mbiter)++;
            if (!(*m_mbiter))
            {
                m_noData = true;
            }
            fprintf(stderr, "WARNING: Utterance \"%S\" does not have "
                            "resource to compute derivative, skipping it.\n",
                    m_uttInfo[i][0].first.c_str());
            return ReNewBufferForMultiIO(i);
        }

        // We don't support having two utterances in the same buffer.
        if (m_doMinibatchBuffering && m_hasUttInCurrentMinibatch.find(m_uttInfo[i][0].first) != m_hasUttInCurrentMinibatch.end())
        {
            (*m_mbiter)++;
            if (!(*m_mbiter))
            {
                m_noData = true;
            }
            fprintf(stderr, "WARNING: Utterance \"%S\" already exists in "
                            "the minibatch, skipping it.\n",
                    m_uttInfo[i][0].first.c_str());
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
    else if (m_featuresBufferAllocatedMultiUtt[i] < totalFeatNum) // buffer size changed. can be partial minibatch
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
        size_t dim = m_labelNameToDimMap[it->first];

        const vector<size_t>& uids = m_mbiter->labels(id);
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
        assert(actualmbsizeOri == m_mbiter->currentmbframes());

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
            for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns in outside loop
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
        size_t dim = m_labelNameToDimMap[it->first];

        const vector<size_t>& uids = m_mbiter->labels(id);
        size_t actualmbsizeOri = uids.size();

        if (m_convertLabelsToTargetsMultiIO[id])
        {
            size_t labelDim = m_labelToTargetMapMultiIO[id].size();
            for (int k = 0; k < actualmbsizeOri; k++)
            {
                assert(uids[k] < labelDim);
                labelDim;
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
            for (int k = 0; k < actualmbsizeOri; k++)
            {
                assert(uids[k] < dim);
                // labels(uids[i], i) = (ElemType)1;
                m_labelsBufferMultiUtt[i][k * dim + uids[k] + m_labelsStartIndexMultiUtt[id + i * numOfLabel]] = (ElemType) 1;
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
template <class ElemType>
bool HTKMLFReader<ElemType>::GetMinibatchCopy(
    std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    StreamMinibatchInputs& matrices,
    MBLayoutPtr pMBLayout)
{
    // We need to get a "copy" of the minibatch to do the forward
    // computation for sequence training.
    if (m_doMinibatchBuffering)
    {
        assert(m_framemode == false);
        if (m_uttDerivBuffer->NeedLikelihoodToComputeDerivative())
        {
            m_getMinibatchCopy = true;
            if (GetMinibatchToTrainOrTest(matrices))
            {
                pMBLayout->CopyFrom(m_pMBLayout);
                uttInfo = m_minibatchUttInfo;
                m_getMinibatchCopy = false;
                return true;
            }
            m_getMinibatchCopy = false;
        }
        return false;
    }
    return false;
}

template <class ElemType>
bool HTKMLFReader<ElemType>::SetNetOutput(
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    const MatrixBase& outputsb,
    const MBLayoutPtr pMBLayout)
{
    const auto& outputs = dynamic_cast<const Matrix<ElemType>&>(outputsb); // TODO: a NULL check, to be sure
    // Set the likelihoods for the utterance with which we can comput the
    // derivatives. Note that the minibatch may only contain partial output
    // for the utterance, <m_uttDerivBuffer> takes care of "gluing" them
    // together.
    if (m_doMinibatchBuffering)
    {
        assert(m_framemode == false);
        return m_uttDerivBuffer->SetLikelihood(uttInfo, outputs, pMBLayout);
    }
    return false;
}

// GetLabelMapping - Gets the label mapping from integer to type in file
// mappingTable - a map from numeric datatype to native label type stored as a string
template <class ElemType>
const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& HTKMLFReader<ElemType>::GetLabelMapping(const std::wstring& /*sectionName*/)
{
    return m_idToLabelMap;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void HTKMLFReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<LabelIdType, LabelType>& labelMapping)
{
    m_idToLabelMap = labelMapping;
}

template <class ElemType>
size_t HTKMLFReader<ElemType>::ReadLabelToTargetMappingFile(const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap)
{
    if (labelListFile == L"")
        throw std::runtime_error("HTKMLFReader::ReadLabelToTargetMappingFile(): cannot read labelToTargetMappingFile without a labelMappingFile!");

    vector<std::wstring> labelList;
    size_t count, numLabels;
    count = 0;
    // read statelist first
    msra::files::textreader labelReader(labelListFile);
    while (labelReader)
    {
        labelList.push_back(labelReader.wgetline());
        count++;
    }
    numLabels = count;
    count = 0;
    msra::files::textreader mapReader(labelToTargetMappingFile);
    size_t targetDim = 0;
    while (mapReader)
    {
        std::wstring line(mapReader.wgetline());
        // find white space as a demarcation
        std::wstring::size_type pos = line.find(L" ");
        std::wstring token = line.substr(0, pos);
        std::wstring targetstring = line.substr(pos + 1);

        if (labelList[count] != token)
            RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): mismatch between labelMappingFile and labelToTargetMappingFile");

        if (count == 0)
            targetDim = targetstring.length();
        else if (targetDim != targetstring.length())
            RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): inconsistent target length among records");

        std::vector<ElemType> targetVector(targetstring.length(), (ElemType) 0.0);
        foreach_index (i, targetstring)
        {
            if (targetstring.compare(i, 1, L"1") == 0)
                targetVector[i] = (ElemType) 1.0;
            else if (targetstring.compare(i, 1, L"0") != 0)
                RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): expecting label2target mapping to contain only 1's or 0's");
        }
        labelToTargetMap.push_back(targetVector);
        count++;
    }

    // verify that statelist and label2target mapping file are in same order (to match up with reader) while reading mapping
    if (count != labelList.size())
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
template <class ElemType>
bool HTKMLFReader<ElemType>::GetData(const std::wstring& /*sectionName*/, size_t /*numRecords*/, void* /*data*/, size_t& /*dataBufferSize*/, size_t /*recordStart*/)
{
    throw std::runtime_error("GetData not supported in HTKMLFReader");
}

template <class ElemType>
bool HTKMLFReader<ElemType>::DataEnd()
{
    // each minibatch is considered a "sentence"
    // for the truncated BPTT, we need to support check whether it's the end of data
    if (m_truncated)
        return m_sentenceEnd[0];
    else
        return true; // useless in current condition
}

template <class ElemType>
void HTKMLFReader<ElemType>::SetSentenceEndInBatch(vector<size_t>& sentenceEnd)
{
    sentenceEnd.resize(m_switchFrame.size());
    for (size_t i = 0; i < m_switchFrame.size(); i++)
    {
        sentenceEnd[i] = m_switchFrame[i];
    }
}

// For Kaldi2Reader, we now make the following assumptions
// 1. feature sections will always have a sub-field "scpFile"
// 2. label sections will always have a sub-field "mlfFile"
template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::GetDataNamesFromConfig(const ConfigRecordType& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels)
{
    for (auto& id : readerConfig.GetMemberIds())
    {
        if (!readerConfig.CanBeConfigRecord(id))
            continue;
        const ConfigRecordType& temp = readerConfig(id);
        // see if we have a config parameters that contains a "file" element, it's a sub key, use it
        if (temp.ExistsCurrent(L"scpFile"))
        {
            features.push_back(id);
        }
        else if (temp.ExistsCurrent(L"mlfFile"))
        {
            labels.push_back(id);
        }
    }
}

template class HTKMLFReader<float>;
template class HTKMLFReader<double>;
} } }
