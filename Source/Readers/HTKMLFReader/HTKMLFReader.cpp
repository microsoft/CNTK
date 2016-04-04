//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#include "htkfeatio.h"       // for reading HTK features
#include "latticearchive.h"  // for reading HTK phoneme lattices (MMI training)
#include "simplesenonehmm.h" // for MMI scoring
#include "msra_mgram.h"      // for unigram scores of ground-truth path in sequence training

#include "rollingwindowsource.h" // minibatch sources
#include "utterancesourcemulti.h"
#include "chunkevalsource.h"
#include "minibatchiterator.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "Config.h"
#include "ScriptableObjects.h"
#include "HTKMLFReader.h"
#include "TimerUtility.h"
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

#ifdef _WIN32
int msra::numa::node_override = -1; // for numahelpers.h
#endif

namespace msra { namespace lm {

/*static*/ const mgram_map::index_t mgram_map::nindex = (mgram_map::index_t) -1; // invalid index
}
}

namespace msra { namespace asr {
    /*static*/ std::unordered_map<std::wstring, unsigned int> htkfeatreader::parsedpath::archivePathStringMap;
    /*static*/ std::vector<std::wstring> htkfeatreader::parsedpath::archivePathStringVector;
}}

namespace Microsoft { namespace MSR { namespace CNTK {

// Create a Data Reader
//DATAREADER_API IDataReader* DataReaderFactory(void)

template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    m_truncated = readerConfig(L"truncated", false);
    m_convertLabelsToTargets = false;

    intargvector numberOfuttsPerMinibatchForAllEpochs = readerConfig(L"nbruttsineachrecurrentiter", ConfigRecordType::Array(intargvector(vector<int>{1})));
    m_numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;

    for (int i = 0; i < m_numSeqsPerMBForAllEpochs.size(); i++)
    {
        if (m_numSeqsPerMBForAllEpochs[i] < 1)
        {
            LogicError("nbrUttsInEachRecurrentIter cannot be less than 1.");
        }
    }

    m_numSeqsPerMB = m_numSeqsPerMBForAllEpochs[0];
    m_pMBLayout->Init(m_numSeqsPerMB, 0); // (SGD will ask before entering actual reading --TODO: This is hacky.)

    m_noData = false;

    wstring command(readerConfig(L"action", L"")); // look up in the config for the master command to determine whether we're writing output (inputs only) or training/evaluating (inputs and outputs)

    if (readerConfig.Exists(L"legacyMode"))
        RuntimeError("legacy mode has been deprecated\n");

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

// Load all input and output data.
// Note that the terms features imply be real-valued quantities and
// labels imply categorical quantities, irrespective of whether they
// are inputs or targets for the network
template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::PrepareForTrainingOrTesting(const ConfigRecordType& readerConfig)
{
    vector<wstring> scriptpaths;
    vector<wstring> RootPathInScripts;
    wstring RootPathInLatticeTocs;
    vector<wstring> mlfpaths;
    vector<vector<wstring>> mlfpathsmulti;
    size_t firstfilesonly = SIZE_MAX; // set to a lower value for testing
    vector<vector<wstring>> infilesmulti;
    size_t numFiles;
    wstring unigrampath(L"");

    size_t randomize = randomizeAuto;
    size_t iFeat, iLabel;
    iFeat = iLabel = 0;
    vector<wstring> statelistpaths;
    vector<size_t> numContextLeft;
    vector<size_t> numContextRight;
    size_t numExpandToUtt = 0;

    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;

    // for hmm and lattice
    std::vector<std::wstring> hmmNames;
    std::vector<std::wstring> latticeNames;
    GetDataNamesFromConfig(readerConfig, featureNames, labelNames, hmmNames, latticeNames);
    if (featureNames.size() + labelNames.size() <= 1)
    {
        InvalidArgument("network needs at least 1 input and 1 output specified!");
    }

    // load data for all real-valued inputs (features)
    foreach_index (i, featureNames)
    {
        const ConfigRecordType& thisFeature = readerConfig(featureNames[i]);
        m_featDims.push_back(thisFeature(L"dim"));

        bool expandToUtt = thisFeature(L"expandToUtterance", false); // should feature be processed as an ivector?
        m_expandToUtt.push_back(expandToUtt);
        if (expandToUtt)
            numExpandToUtt++;

        intargvector contextWindow = thisFeature(L"contextWindow", ConfigRecordType::Array(intargvector(vector<int>{1})));
        if (contextWindow.size() == 1) // symmetric
        {
            size_t windowFrames = contextWindow[0];
            if (windowFrames % 2 == 0)
                InvalidArgument("augmentationextent: neighbor expansion of input features to %d not symmetrical", (int) windowFrames);

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
            InvalidArgument("contextFrames must have 1 or 2 values specified, found %d", (int) contextWindow.size());
        }

        if (expandToUtt && (numContextLeft[i] != 0 || numContextRight[i] != 0))
            RuntimeError("contextWindow expansion not permitted when expandToUtterance=true");

        // update m_featDims to reflect the total input dimension (featDim x contextWindow), not the native feature dimension
        // that is what the lower level feature readers expect
        m_featDims[i] = m_featDims[i] * (1 + numContextLeft[i] + numContextRight[i]);

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
        scriptpaths.push_back(thisFeature(L"scpFile"));
        RootPathInScripts.push_back(thisFeature(L"prefixPathInSCP", L""));
        m_featureNameToDimMap[featureNames[i]] = m_featDims[i];

        m_featuresBufferMultiIO.push_back(nullptr);
        m_featuresBufferAllocatedMultiIO.push_back(0);

        iFeat++;
    }

    foreach_index (i, labelNames)
    {
        const ConfigRecordType& thisLabel = readerConfig(labelNames[i]);
        if (thisLabel.Exists(L"labelDim"))
            m_labelDims.push_back(thisLabel(L"labelDim"));
        else if (thisLabel.Exists(L"dim"))
            m_labelDims.push_back(thisLabel(L"dim"));
        else
            InvalidArgument("labels must specify dim or labelDim");

        wstring type;
        if (thisLabel.Exists(L"labelType"))
            type = (const wstring&) thisLabel(L"labelType"); // let's deprecate this eventually and just use "type"...
        else
            type = (const wstring&) thisLabel(L"type", L"category"); // outputs should default to category

        if (EqualCI(type, L"category"))
            m_nameToTypeMap[labelNames[i]] = InputOutputTypes::category;
        else
            InvalidArgument("label type must be 'category'");

        statelistpaths.push_back(thisLabel(L"labelMappingFile", L""));

        m_labelNameToIdMap[labelNames[i]] = iLabel;
        m_labelNameToDimMap[labelNames[i]] = m_labelDims[i];
        mlfpaths.clear();
        if (thisLabel.ExistsCurrent(L"mlfFile"))
        {
            mlfpaths.push_back(thisLabel(L"mlfFile"));
        }
        else
        {
            if (!thisLabel.ExistsCurrent(L"mlfFileList"))
            {
                InvalidArgument("Either mlfFile or mlfFileList must exist in HTKMLFReder");
            }

            wstring list = thisLabel(L"mlfFileList");
            for (msra::files::textreader r(list); r;)
            {
                mlfpaths.push_back(r.wgetline());
            }
        }
        mlfpathsmulti.push_back(mlfpaths);

        m_labelsBufferMultiIO.push_back(nullptr);
        m_labelsBufferAllocatedMultiIO.push_back(0);

        iLabel++;

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
            {
                RuntimeError("output must specify targetDim if labelToTargetMappingFile specified!");
            }

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

    // get lattice toc file names
    std::pair<std::vector<wstring>, std::vector<wstring>> latticetocs;
    foreach_index (i, latticeNames) // only support one set of lattice now
    {
        const ConfigRecordType& thisLattice = readerConfig(latticeNames[i]);

        vector<wstring> paths;
        expand_wildcards(thisLattice(L"denLatTocFile"), paths);
        latticetocs.second.insert(latticetocs.second.end(), paths.begin(), paths.end());

        if (thisLattice.Exists(L"numLatTocFile"))
        {
            paths.clear();
            expand_wildcards(thisLattice(L"numLatTocFile"), paths);
            latticetocs.first.insert(latticetocs.first.end(), paths.begin(), paths.end());
        }
        RootPathInLatticeTocs = (wstring) thisLattice(L"prefixPathInToc", L"");
    }

    // get HMM related file names
    vector<wstring> cdphonetyingpaths, transPspaths;
    foreach_index (i, hmmNames)
    {
        const ConfigRecordType& thisHMM = readerConfig(hmmNames[i]);

        cdphonetyingpaths.push_back(thisHMM(L"phoneFile"));
        transPspaths.push_back(thisHMM(L"transPFile", L""));
    }

    // mmf files
    // only support one set now
    if (cdphonetyingpaths.size() > 0 && statelistpaths.size() > 0 && transPspaths.size() > 0)
        m_hset.loadfromfile(cdphonetyingpaths[0], statelistpaths[0], transPspaths[0]);

    if (iFeat != scriptpaths.size() || iLabel != mlfpathsmulti.size())
        RuntimeError("# of inputs files vs. # of inputs or # of output files vs # of outputs inconsistent\n");

    if (iFeat == numExpandToUtt)
        RuntimeError("At least one feature stream must be frame-based, not utterance-based");

    if (m_expandToUtt[0]) // first feature stream is ivector type - that will mess up lower level feature reader
        RuntimeError("The first feature stream in the file must be frame-based not utterance based. Please reorder the feature blocks of your config appropriately");

    if (readerConfig.Exists(L"randomize"))
    {
        wstring randomizeString = readerConfig.CanBeString(L"randomize") ? readerConfig(L"randomize") : wstring();
        if      (EqualCI(randomizeString, L"none")) randomize = randomizeNone;
        else if (EqualCI(randomizeString, L"auto")) randomize = randomizeAuto;
        else                                        randomize = readerConfig(L"randomize"); // TODO: could this not just be randomizeString?
    }

    m_frameMode = readerConfig(L"frameMode", true);
    m_verbosity = readerConfig(L"verbosity", 2);

    if (m_frameMode && m_truncated)
    {
        InvalidArgument("'Truncated' cannot be 'true' in frameMode (i.e. when 'frameMode' is 'true')");
    }

    // determine if we partial minibatches are desired
    wstring minibatchMode(readerConfig(L"minibatchMode", L"partial"));
    m_partialMinibatch = EqualCI(minibatchMode, L"partial");

    // get the read method, defaults to "blockRandomize" other option is "rollingWindow"
    wstring readMethod(readerConfig(L"readMethod", L"blockRandomize"));

    if (readMethod == L"blockRandomize" && randomize == randomizeNone)
        InvalidArgument("'randomize' cannot be 'none' when 'readMethod' is 'blockRandomize'.");

    if (readMethod == L"rollingWindow" && numExpandToUtt>0)
        RuntimeError("rollingWindow reader does not support expandToUtt. Change to blockRandomize.\n");

    // read all input files (from multiple inputs)
    // TO DO: check for consistency (same number of files in each script file)
    numFiles = 0;
    foreach_index (i, scriptpaths)
    {
        vector<wstring> filelist;
        std::wstring scriptpath = scriptpaths[i];
        fprintf(stderr, "reading script file %ls ...", scriptpath.c_str());
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
            RuntimeError("number of files in each scriptfile inconsistent (%d vs. %d)", (int) numFiles, (int) n);

        // post processing file list :
        //  - if users specified PrefixPath, add the prefix to each of path in filelist
        //  - else do the dotdotdot expansion if necessary
        wstring rootpath = RootPathInScripts[i];
        if (!rootpath.empty()) // use has specified a path prefix for this  feature
        {
            // first make slash consistent (sorry for linux users:this is not necessary for you)
            std::replace(rootpath.begin(), rootpath.end(), L'\\', L'/');

            // second, remove trailing slash if there is any
            // TODO: when gcc -v is 4.9 or greater, this should be: std::regex_replace(rootpath, L"\\/+$", wstring());
            size_t stringPos = 0;
            for (stringPos = rootpath.length() - 1; stringPos >= 0; stringPos--) 
            {
                if (rootpath[stringPos] != L'/')
                {
                    break;
                }
            }
            rootpath = rootpath.substr(0, stringPos + 1);

            // third, join the rootpath with each entry in filelist
            if (!rootpath.empty())
            {
                for (wstring& path : filelist)
                {
                    if (path.find_first_of(L'=') != wstring::npos)
                    {
                        vector<wstring> strarr = msra::strfun::split(path, L"=");
#ifdef WIN32
                        replace(strarr[1].begin(), strarr[1].end(), L'\\', L'/');
#endif

                        path = strarr[0] + L"=" + rootpath + L"/" + strarr[1];
                    }
                    else
                    {
#ifdef WIN32
                        replace(path.begin(), path.end(), L'\\', L'/');
#endif
                        path = rootpath + L"/" + path;
                    }
                }
            }
        }
        else
        {
            /*
            do "..." expansion if SCP uses relative path names
            "..." in the SCP means full path is the same as the SCP file
            for example, if scp file is "//aaa/bbb/ccc/ddd.scp"
            and contains entry like
            .../file1.feat
            .../file2.feat
            etc.
            the features will be read from
            // aaa/bbb/ccc/file1.feat
            // aaa/bbb/ccc/file2.feat
            etc.
            This works well if you store the scp file with the features but
            do not want different scp files everytime you move or create new features
            */
            wstring scpdircached;
            for (auto& entry : filelist)
                ExpandDotDotDot(entry, scriptpath, scpdircached);
        }

        infilesmulti.push_back(std::move(filelist));
    }

    if (readerConfig.Exists(L"unigram"))
        unigrampath = (const wstring&) readerConfig(L"unigram");

    // load a unigram if needed (this is used for MMI training)
    msra::lm::CSymbolSet unigramsymbols;
    std::unique_ptr<msra::lm::CMGramLM> unigram;
    size_t silencewordid = SIZE_MAX;
    size_t startwordid = SIZE_MAX;
    size_t endwordid = SIZE_MAX;
    if (unigrampath != L"")
    {
        unigram.reset(new msra::lm::CMGramLM());
        unigram->read(unigrampath, unigramsymbols, false /*filterVocabulary--false will build the symbol map*/, 1 /*maxM--unigram only*/);
        silencewordid = unigramsymbols["!silence"]; // give this an id (even if not in the LM vocabulary)
        startwordid = unigramsymbols["<s>"];
        endwordid = unigramsymbols["</s>"];
    }

    if (!unigram && latticetocs.second.size() > 0)
        fprintf(stderr, "trainlayer: OOV-exclusion code enabled, but no unigram specified to derive the word set from, so you won't get OOV exclusion\n");

    // currently assumes all mlfs will have same root name (key)
    set<wstring> restrictmlftokeys; // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
    if (infilesmulti[0].size() <= 100)
    {
        foreach_index (i, infilesmulti[0])
        {
            msra::asr::htkfeatreader::parsedpath ppath(infilesmulti[0][i]);
            const wstring ppathStr = (wstring) ppath;

            // delete extension (or not if none) 
            // TODO: when gcc -v is 4.9 or greater, this should be: regex_replace((wstring)ppath, wregex(L"\\.[^\\.\\\\/:]*$"), wstring()); 
            int stringPos = 0;
            for (stringPos = (int) ppathStr.length() - 1; stringPos >= 0; stringPos--) 
            {
                if (ppathStr[stringPos] == L'.' || ppathStr[stringPos] == L'\\' || ppathStr[stringPos] == L'/' || ppathStr[stringPos] == L':')
                {
                    break;
                }
            }

            if (ppathStr[stringPos] == L'.') {
                restrictmlftokeys.insert(ppathStr.substr(0, stringPos));
            }
            else 
            {
                restrictmlftokeys.insert(ppathStr);
            }
        }
    }
    // get labels

    // if (readerConfig.Exists(L"statelist"))
    //    statelistpath = readerConfig(L"statelist");

    double htktimetoframe = 100000.0; // default is 10ms
    // std::vector<msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>> labelsmulti;
    std::vector<std::map<std::wstring, std::vector<msra::asr::htkmlfentry>>> labelsmulti;
    // std::vector<std::wstring> pagepath;
    foreach_index (i, mlfpathsmulti)
    {
        const msra::lm::CSymbolSet* wordmap = unigram ? &unigramsymbols : NULL;
        msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>
        labels(mlfpathsmulti[i], restrictmlftokeys, statelistpaths[i], wordmap, (map<string, size_t>*) NULL, htktimetoframe); // label MLF
        // get the temp file name for the page file

        // Make sure 'msra::asr::htkmlfreader' type has a move constructor
        static_assert(std::is_move_constructible<msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence>>::value,
                      "Type 'msra::asr::htkmlfreader' should be move constructible!");

        labelsmulti.push_back(std::move(labels));
    }

    if (EqualCI(readMethod, L"blockRandomize"))
    {
        // construct all the parameters we don't need, but need to be passed to the constructor...

        m_lattices.reset(new msra::dbn::latticesource(latticetocs, m_hset.getsymmap(), RootPathInLatticeTocs));
        m_lattices->setverbosity(m_verbosity);

        // now get the frame source. This has better randomization and doesn't create temp files
        bool minimizeReaderMemoryFootprint = readerConfig(L"minimizeReaderMemoryFootprint", true);
        m_frameSource.reset(new msra::dbn::minibatchutterancesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, 
                                                                         numContextLeft, numContextRight, randomize, 
                                                                         *m_lattices, m_latticeMap, m_frameMode, 
                                                                         minimizeReaderMemoryFootprint, m_expandToUtt));
        m_frameSource->setverbosity(m_verbosity);
    }
    else if (EqualCI(readMethod, L"rollingWindow"))
    {
        std::wstring pageFilePath;
        std::vector<std::wstring> pagePaths;
        if (readerConfig.Exists(L"pageFilePath"))
        {
            pageFilePath = (const wstring&) readerConfig(L"pageFilePath");

#ifdef _WIN32
            // replace any '/' with '\' for compat with default path
            std::replace(pageFilePath.begin(), pageFilePath.end(), '/', '\\');

            // verify path exists
            DWORD attrib = GetFileAttributes(pageFilePath.c_str());
            if (attrib == INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY))
                RuntimeError("pageFilePath does not exist");
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
            pageFilePath = L"/tmp/temp.CNTK.XXXXXX";
#endif
        }

#ifdef _WIN32
        if (pageFilePath.size() > MAX_PATH - 14) // max length of input to GetTempFileName is MAX_PATH-14
            RuntimeError("pageFilePath must be less than %d characters", MAX_PATH - 14);
#else
        if (pageFilePath.size() > PATH_MAX - 14) // max length of input to GetTempFileName is PATH_MAX-14
            RuntimeError("pageFilePath must be less than %d characters", PATH_MAX - 14);
#endif
        foreach_index (i, infilesmulti)
        {
#ifdef _WIN32
            wchar_t tempFile[MAX_PATH];
            GetTempFileName(pageFilePath.c_str(), L"CNTK", 0, tempFile);
            pagePaths.push_back(tempFile);
#endif
#ifdef __unix__
            char tempFile[PATH_MAX];
            strcpy(tempFile, msra::strfun::utf8(pageFilePath).c_str());
            int fid = mkstemp(tempFile);
            unlink(tempFile);
            close(fid);
            pagePaths.push_back(GetWC(tempFile));
#endif
        }

        const bool mayhavenoframe = false;
        int addEnergy = 0;

        m_frameSource.reset(new msra::dbn::minibatchframesourcemulti(infilesmulti, labelsmulti, m_featDims, m_labelDims, 
                                                                     numContextLeft, numContextRight, randomize, 
                                                                     pagePaths, mayhavenoframe, addEnergy));
        m_frameSource->setverbosity(m_verbosity);
    }
    else
    {
        RuntimeError("readMethod must be 'rollingWindow' or 'blockRandomize'");
    }
}

// Load all input and output data.
// Note that the terms features imply be real-valued quanities and
// labels imply categorical quantities, irrespective of whether they
// are inputs or targets for the network
// TODO: lots of code dup with the other Prepare function
template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::PrepareForWriting(const ConfigRecordType& readerConfig)
{
    vector<wstring> scriptpaths;
    size_t numFiles;
    size_t firstfilesonly = SIZE_MAX; // set to a lower value for testing
    size_t evalchunksize = 2048;
    vector<size_t> realDims;
    size_t iFeat = 0;
    vector<size_t> numContextLeft;
    vector<size_t> numContextRight;

    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;
    // lattice and hmm
    std::vector<std::wstring> hmmNames;
    std::vector<std::wstring> latticeNames;

    GetDataNamesFromConfig(readerConfig, featureNames, labelNames, hmmNames, latticeNames);

    foreach_index (i, featureNames)
    {
        const ConfigRecordType& thisFeature = readerConfig(featureNames[i]);
        realDims.push_back(thisFeature(L"dim"));

        intargvector contextWindow = thisFeature(L"contextWindow", ConfigRecordType::Array(intargvector(vector<int>{1})));
        if (contextWindow.size() == 1) // symmetric
        {
            size_t windowFrames = contextWindow[0];
            if (windowFrames % 2 == 0)
            {
                RuntimeError("augmentationextent: neighbor expansion of input features to %d not symmetrical", (int)windowFrames);
            }

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

        // update m_featDims to reflect the total input dimension (featDim x contextWindow), not the native feature dimension
        // that is what the lower level feature readers expect
        realDims[i] = realDims[i] * (1 + numContextLeft[i] + numContextRight[i]);

        wstring type = thisFeature(L"type", L"real");
        if (EqualCI(type, L"real"))
        {
            m_nameToTypeMap[featureNames[i]] = InputOutputTypes::real;
        }
        else
        {
            RuntimeError("feature type must be 'real'");
        }

        m_featureNameToIdMap[featureNames[i]] = iFeat;
        scriptpaths.push_back(thisFeature(L"scpFile"));
        m_featureNameToDimMap[featureNames[i]] = realDims[i];

        m_featuresBufferMultiIO.push_back(nullptr);
        m_featuresBufferAllocatedMultiIO.push_back(0);
        iFeat++;
    }

    if (labelNames.size() > 0)
        RuntimeError("writer mode does not support labels as inputs, only features");

    numFiles = 0;
    foreach_index (i, scriptpaths)
    {
        vector<wstring> filelist;
        std::wstring scriptpath = scriptpaths[i];
        fprintf(stderr, "reading script file %ls ...", scriptpath.c_str());
        size_t n = 0;
        for (msra::files::textreader reader(scriptpath); reader && filelist.size() <= firstfilesonly /*optimization*/;)
        {
            filelist.push_back(reader.wgetline());
            n++;
        }

        fprintf(stderr, " %d entries\n", (int) n);

        if (i == 0)
            numFiles = n;
        else if (n != numFiles)
            RuntimeError("HTKMLFReader::InitEvalReader: number of files in each scriptfile inconsistent (%d vs. %d)", (int) numFiles, (int) n);

        m_inputFilesMultiIO.push_back(std::move(filelist));
    }

    m_fileEvalSource.reset(new msra::dbn::FileEvalSource(realDims, numContextLeft, numContextRight, evalchunksize));
}

//StartMinibatchLoop - Startup a minibatch loop
// requestedMBSize - [in] size of the minibatch (number of frames, etc.)
// epoch - [in] epoch number for this loop
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template <class ElemType>
void HTKMLFReader<ElemType>::StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/)
{
    assert(subsetNum < numSubsets);
    assert(((subsetNum == 0) && (numSubsets == 1)) || this->SupportsDistributedMBRead());

    m_mbNumTimeSteps = requestedMBSize; // note: ignored in frame mode and full-sequence mode

    m_numSeqsPerMB = m_numSeqsPerMBForAllEpochs[epoch];

    // For distributed reading under utterance mode, we distribute the utterances per minibatch among all the subsets
    if (m_trainOrTest && !m_frameMode)
    {
        if ((numSubsets > 1) && (m_numSeqsPerMB < numSubsets))
        {
            LogicError("Insufficient value of 'nbruttsineachrecurrentiter'=%d for distributed reading with %d subsets", (int) m_numSeqsPerMB, (int) numSubsets);
        }

        m_numSeqsPerMB = (m_numSeqsPerMB / numSubsets) + ((subsetNum < (m_numSeqsPerMB % numSubsets)) ? 1 : 0);
    }

    m_pMBLayout->Init(m_numSeqsPerMB, 0); // (SGD will ask before entering actual reading --TODO: This is hacky.)

    // resize the arrays
    // These are sized to the requested number. If not all can be filled, it will still return this many, just with gaps.
    // In frame mode, m_numSeqsPerMB must be 1. However, the returned layout has one 1-frame sequence per frame.
    m_sentenceEnd.assign(m_numSeqsPerMB, true);
    m_processedFrame.assign(m_numSeqsPerMB, 0);
    m_numFramesToProcess.assign(m_numSeqsPerMB, 0);
    m_switchFrame.assign(m_numSeqsPerMB, 0);
    m_numValidFrames.assign(m_numSeqsPerMB, 0);

    if (m_trainOrTest)
    {
        // for the multi-utterance process
        m_featuresBufferMultiUtt.assign(m_numSeqsPerMB, nullptr);
        m_featuresBufferAllocatedMultiUtt.assign(m_numSeqsPerMB, 0);
        m_labelsBufferMultiUtt.assign(m_numSeqsPerMB, nullptr);
        m_labelsBufferAllocatedMultiUtt.assign(m_numSeqsPerMB, 0);

        // for the multi-utterance process for lattice and phone boundary
        m_latticeBufferMultiUtt.assign(m_numSeqsPerMB, nullptr);
        m_labelsIDBufferMultiUtt.resize(m_numSeqsPerMB);
        m_phoneboundaryIDBufferMultiUtt.resize(m_numSeqsPerMB);

        if (m_frameMode && (m_numSeqsPerMB > 1))
        {
            LogicError("nbrUttsInEachRecurrentIter cannot be more than 1 in frame mode reading.");
        }

        // BUGBUG: in BPTT and sequence mode, we should pass 1 or 2 instead of requestedMBSize to ensure we only get one utterance back at a time
        StartMinibatchLoopToTrainOrTest(requestedMBSize, epoch, subsetNum, numSubsets, requestedEpochSamples);
    }
    else
    {
        // No distributed reading of mini-batches for write
        if ((subsetNum != 0) || (numSubsets != 1))
        {
            LogicError("Distributed reading of mini-batches is only supported for training or testing");
        }

        m_pMBLayout->Init(requestedMBSize, 0); // (SGD will ask before entering actual reading --TODO: This is hacky.)

        StartMinibatchLoopToWrite(requestedMBSize, epoch, requestedEpochSamples);
    }
    m_checkDictionaryKeys = true;
}

template <class ElemType>
void HTKMLFReader<ElemType>::StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples)
{
    size_t datapasses = 1;
    size_t totalFrames = m_frameSource->totalframes();

    size_t extraFrames = totalFrames % mbSize;
    size_t minibatches = totalFrames / mbSize;

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

    m_mbiter.reset(new msra::dbn::minibatchiterator(*m_frameSource, epoch, requestedEpochSamples, mbSize, subsetNum, numSubsets, datapasses));
    // Advance the MB iterator until we find some data or reach the end of epoch
    while ((m_mbiter->currentmbframes() == 0) && *m_mbiter)
    {
        (*m_mbiter)++;
    }

    m_noData = false;
    if (!(*m_mbiter))
        m_noData = true;

    if (!m_featuresBufferMultiIO.empty())
    {
        if (m_featuresBufferMultiIO[0] != nullptr) // check first feature, if it isn't NULL, safe to assume all are not NULL?
        {
            foreach_index (i, m_featuresBufferMultiIO)
            {
                m_featuresBufferMultiIO[i] = nullptr;
                m_featuresBufferAllocatedMultiIO[i] = 0;
            }
        }

        m_featuresStartIndexMultiUtt.assign(m_featuresBufferMultiIO.size() * m_numSeqsPerMB, 0);
    }

    if (!m_labelsBufferMultiIO.empty())
    {
        if (m_labelsBufferMultiIO[0] != nullptr)
        {
            foreach_index (i, m_labelsBufferMultiIO)
            {
                m_labelsBufferMultiIO[i] = nullptr;
                m_labelsBufferAllocatedMultiIO[i] = 0;
            }
        }

        m_labelsStartIndexMultiUtt.assign(m_labelsBufferMultiIO.size() * m_numSeqsPerMB, 0);
    }

    for (size_t u = 0; u < m_numSeqsPerMB; u++)
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

template <class ElemType>
void HTKMLFReader<ElemType>::StartMinibatchLoopToWrite(size_t mbSize, size_t /*epoch*/, size_t /*requestedEpochSamples*/)
{
    m_fileEvalSource->Reset();
    m_fileEvalSource->SetMinibatchSize(mbSize);
    m_inputFileIndex = 0;

    if (m_featuresBufferMultiIO[0] != nullptr) // check first feature, if it isn't NULL, safe to assume all are not NULL?
    {
        foreach_index (i, m_featuresBufferMultiIO)
        {
            m_featuresBufferMultiIO[i] = nullptr;
            m_featuresBufferAllocatedMultiIO[i] = 0;
        }
    }
}

template <class ElemType>
bool HTKMLFReader<ElemType>::GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>>& latticeinput,
                                             vector<size_t>& uids, vector<size_t>& boundaries, vector<size_t>& extrauttmap)
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
template <class ElemType>
bool HTKMLFReader<ElemType>::GetMinibatch4SEToTrainOrTest(std::vector<shared_ptr<const msra::dbn::latticepair>>& latticeinput,
                                                          std::vector<size_t>& uids, std::vector<size_t>& boundaries, std::vector<size_t>& extrauttmap)
{
    latticeinput.clear();
    uids.clear();
    boundaries.clear();
    extrauttmap.clear();
    for (size_t i = 0; i < m_extraSeqsPerMB.size(); i++)
    {
        latticeinput.push_back(m_extraLatticeBufferMultiUtt[i]);
        uids.insert(uids.end(), m_extraLabelsIDBufferMultiUtt[i].begin(), m_extraLabelsIDBufferMultiUtt[i].end());
        boundaries.insert(boundaries.end(), m_extraPhoneboundaryIDBufferMultiUtt[i].begin(), m_extraPhoneboundaryIDBufferMultiUtt[i].end());
    }

    extrauttmap.insert(extrauttmap.end(), m_extraSeqsPerMB.begin(), m_extraSeqsPerMB.end());
    return true;
}

template <class ElemType>
bool HTKMLFReader<ElemType>::GetHmmData(msra::asr::simplesenonehmm* hmm)
{
    *hmm = m_hset;
    return true;
}
// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
// TODO: Why do we have two read functions? Is one not a superset of the other?
template <class ElemType>
bool HTKMLFReader<ElemType>::GetMinibatch(StreamMinibatchInputs& matrices)
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

template <class ElemType>
bool HTKMLFReader<ElemType>::GetMinibatchToTrainOrTest(StreamMinibatchInputs& matrices)
{
    size_t id;
    size_t dim;
    bool skip = false;

    // on first minibatch, make sure we can supply data for requested nodes
    std::map<std::wstring, size_t>::iterator iter;
    if (m_checkDictionaryKeys)
    {
        for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
        {
            if (m_nameToTypeMap.find(iter->first) == m_nameToTypeMap.end())
            {
                RuntimeError("minibatch requested for input node %ls not found in reader - cannot generate input\n", iter->first.c_str());
            }
        }
        m_checkDictionaryKeys = false;
    }

    Timer aggregateTimer;
    if (m_verbosity > 2)
        aggregateTimer.Start();

    do
    {
        if (!m_truncated)
        {
            // -------------------------------------------------------
            // frame mode or whole utterances
            // -------------------------------------------------------

            m_extraLatticeBufferMultiUtt.clear();
            m_extraLabelsIDBufferMultiUtt.clear();
            m_extraPhoneboundaryIDBufferMultiUtt.clear();
            m_extraSeqsPerMB.clear();
            if (m_noData && m_numFramesToProcess[0] == 0) // no data left for the first channel of this minibatch,
            {
                return false;
            }

            // BUGBUG: We should decide how many utterances we are going to take, until the desired number of frames has been filled.
            //         Currently it seems to fill a fixed number of utterances, regardless of their length.

            // decide the m_mbNumTimeSteps
            // The number of columns is determined by the longest utterance amongst the desired set.
            // I.e. whatever is user-specified as the MB size, will be ignored here (that value is, however, passed down to the underlying reader).  BUGBUG: That is even more wrong.
            // BUGBUG: We should honor the mbSize parameter and fill up to the requested number of samples, using the requested #parallel sequences.
            // m_mbNumTimeSteps  = max (m_numFramesToProcess[.])
            m_mbNumTimeSteps = m_numFramesToProcess[0];
            for (size_t i = 1; i < m_numSeqsPerMB; i++)
            {
                if (m_mbNumTimeSteps < m_numFramesToProcess[i])
                    m_mbNumTimeSteps = m_numFramesToProcess[i];
            }

            if (m_frameMode)
            {
                assert(m_numSeqsPerMB == 1); // user must not request parallel sequences
                m_pMBLayout->InitAsFrameMode(m_mbNumTimeSteps);
            }
            else
            {
                m_pMBLayout->Init(m_numSeqsPerMB, m_mbNumTimeSteps);
            }

            // create a MB with the desired utterances
            // First fill each parallel sequence with one utterance. No packing yet.
            // Note that the code below is a little misleading for frame mode.
            // In frame mode, this reader thinks it has only one parallel sequence (m_numSeqsPerMB == 1),
            // but it reports it to the outside as N parallel sequences of one frame each.
            skip = (m_frameMode && !m_partialMinibatch && (m_mbiter->requestedframes() != m_mbNumTimeSteps) && (m_frameSource->totalframes() > m_mbNumTimeSteps));
            for (size_t i = 0; i < m_numSeqsPerMB; i++)
            {
                if (!skip)
                {
                    // a stopgap
                    if (m_numFramesToProcess[i] > 0 && m_latticeBufferMultiUtt[i] && m_latticeBufferMultiUtt[i]->getnumframes() != m_numFramesToProcess[i])
                    {
                        // BUGBUG: we just found that (due to some bugs yet to be tracked down),
                        // the filled number of frames is inconsistent with the number frames in lattices (though it rarely occurs)
                        // This is just a stopgap, to be removed after the bugs are found and fixed
                        bool needRenew = true;
                        while (needRenew)
                        {
                            size_t framenum = m_numFramesToProcess[i];
                            fprintf(stderr, "WARNING: mismatched number of frames filled in the reader: %d in data vs %d in lattices. Ignoring this utterance %ls\n",
                                    (int) framenum, (int) m_latticeBufferMultiUtt[i]->getnumframes(), m_latticeBufferMultiUtt[i]->getkey().c_str());
                            ReNewBufferForMultiIO(i);
                            needRenew = m_numFramesToProcess[i] > 0 && m_latticeBufferMultiUtt[i] && m_latticeBufferMultiUtt[i]->getnumframes() != m_numFramesToProcess[i];
                        }
                    }
                    m_numValidFrames[i] = m_numFramesToProcess[i];
                    if (m_numValidFrames[i] > 0)
                    {
                        if (m_frameMode)
                        {
                            // the layout has already been initialized as entirely frame mode above
                            assert(i == 0); // this reader thinks there is only one parallel sequence
                            for (size_t s = 0; s < m_pMBLayout->GetNumParallelSequences(); s++)
                            {
                                assert(s < m_numValidFrames[i]); // MB is already set to only include the valid frames (no need for gaps)
                            }
                        }
                        else
                        {
                            m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, i, 0, m_numValidFrames[i]);
                        }

                        m_extraSeqsPerMB.push_back(i);
                        fillOneUttDataforParallelmode(matrices, 0, m_numValidFrames[i], i, i);

                        if (m_latticeBufferMultiUtt[i] != nullptr)
                        {
                            m_extraLatticeBufferMultiUtt.push_back(m_latticeBufferMultiUtt[i]);
                            m_extraLabelsIDBufferMultiUtt.push_back(m_labelsIDBufferMultiUtt[i]);
                            m_extraPhoneboundaryIDBufferMultiUtt.push_back(m_phoneboundaryIDBufferMultiUtt[i]);
                        }
                    }
                }
                ReNewBufferForMultiIO(i);
            }

            if (!skip)
            {
                m_extraNumSeqs = 0;
                if (!m_frameMode)
                {
                    for (size_t src = 0; src < m_numSeqsPerMB;)
                    {
                        size_t framenum = m_numFramesToProcess[src];
                        if (framenum == 0)
                        {
                            src++;
                            continue;
                        }
                        if (m_latticeBufferMultiUtt[src] != nullptr && m_latticeBufferMultiUtt[src]->getnumframes() != framenum)
                        {
                            // BUGBUG: we just found that (due to some bugs yet to be tracked down),
                            // the filled number of frames is inconsistent with the number frames in lattices (though it rarely occurs)
                            // This is just a stopgap, to be removed after the bugs are found and fixed
                            fprintf(stderr, "WARNING: mismatched number of frames filled in the reader: %d in data vs %d in lattices. Ignoring this utterance %ls\n",
                                    (int) framenum, (int) m_latticeBufferMultiUtt[src]->getnumframes(), m_latticeBufferMultiUtt[src]->getkey().c_str());
                            src++;
                            continue;
                        }

                        bool slotFound = false;
                        for (size_t des = 0; des < m_numSeqsPerMB; des++) // try to found a slot
                        {
                            if (framenum + m_numValidFrames[des] < m_mbNumTimeSteps)
                            { 
                                // found !
                                m_extraSeqsPerMB.push_back(des);
                                if (m_latticeBufferMultiUtt[src] != nullptr)
                                {
                                    m_extraLatticeBufferMultiUtt.push_back(m_latticeBufferMultiUtt[src]);
                                    m_extraLabelsIDBufferMultiUtt.push_back(m_labelsIDBufferMultiUtt[src]);
                                    m_extraPhoneboundaryIDBufferMultiUtt.push_back(m_phoneboundaryIDBufferMultiUtt[src]);
                                }

                                fillOneUttDataforParallelmode(matrices, m_numValidFrames[des], framenum, des, src);
                                m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, des, m_numValidFrames[des], m_numValidFrames[des] + framenum);

                                ReNewBufferForMultiIO(src);
                                m_numValidFrames[des] += framenum;
                                m_extraNumSeqs++;
                                slotFound = true;
                                break;
                            }
                        }

                        if (!slotFound)
                        {
                            src++; // done with this source;  try next source;
                        }
                    }

                    // and declare the remaining gaps as such
                    for (size_t i = 0; i < m_numSeqsPerMB; i++)
                        m_pMBLayout->AddGap(i, m_numValidFrames[i], m_mbNumTimeSteps);
                } // if (!frameMode)

                for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
                {
                    // dereference matrix that corresponds to key (input/output name) and
                    // populate based on whether its a feature or a label
                    Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first); // can be features or labels
                    if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                    {
                        id = m_featureNameToIdMap[iter->first];
                        dim = m_featureNameToDimMap[iter->first];
                        data.SetValue(dim, m_mbNumTimeSteps * m_numSeqsPerMB, data.GetDeviceId(), m_featuresBufferMultiIO[id].get(), matrixFlagNormal);
                    }
                    else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                    {
                        id = m_labelNameToIdMap[iter->first];
                        dim = m_labelNameToDimMap[iter->first];
                        data.SetValue(dim, m_mbNumTimeSteps * m_numSeqsPerMB, data.GetDeviceId(), m_labelsBufferMultiIO[id].get(), matrixFlagNormal);
                    }
                }
            }
        }
        else // if m_truncated
        {
            // -------------------------------------------------------
            // truncated BPTT
            // -------------------------------------------------------

            // In truncated BPTT mode, a minibatch only consists of the truncation length, e.g. 20 frames.
            // The reader maintains a set of current utterances, and each next minibatch contains the next 20 frames.
            // When the end of an utterance is reached, the next available utterance is begin in the same slot.
            if (m_noData) // we are returning the last utterances for this epoch
            {
                // return false if all cursors for all parallel sequences have reached the end
                bool endEpoch = true;
                for (size_t i = 0; i < m_numSeqsPerMB; i++)
                {
                    if (m_processedFrame[i] != m_numFramesToProcess[i])
                        endEpoch = false;
                }

                if (endEpoch)
                    return false;
            }

            size_t numOfFea = m_featuresBufferMultiIO.size();
            size_t numOfLabel = m_labelsBufferMultiIO.size();

            // create the feature matrix
            m_pMBLayout->Init(m_numSeqsPerMB, m_mbNumTimeSteps);

            vector<size_t> actualmbsize(m_numSeqsPerMB, 0);
            for (size_t i = 0; i < m_numSeqsPerMB; i++)
            {
                // fill one parallel-sequence slot
                const size_t startFr = m_processedFrame[i]; // start frame (cursor) inside the utterance that corresponds to time step [0]

                // add utterance to MBLayout
                assert(m_numFramesToProcess[i] > startFr || (m_noData && m_numFramesToProcess[i] == startFr));
                if (m_numFramesToProcess[i] > startFr)
                {   // in an edge case (m_noData), startFr is at end
                    m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, i, -(ptrdiff_t)startFr, m_numFramesToProcess[i] - startFr);
                }

                if (startFr + m_mbNumTimeSteps < m_numFramesToProcess[i]) // end of this minibatch does not reach until end of utterance
                {
                    // we return the next 'm_mbNumTimeSteps' frames, filling all time steps
                    if (startFr > 0) // not the beginning of the utterance
                    {
                        m_sentenceEnd[i] = false;
                        m_switchFrame[i] = m_mbNumTimeSteps + 1;
                    }
                    else // beginning of the utterance
                    {
                        m_sentenceEnd[i] = true;
                        m_switchFrame[i] = 0;
                    }
                    actualmbsize[i] = m_mbNumTimeSteps;
                    const size_t endFr = startFr + actualmbsize[i]; // actual end frame index of this segment
                    for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
                    {
                        // dereference matrix that corresponds to key (input/output name) and
                        // populate based on whether its a feature or a label
                        Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first); // can be features or labels

                        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                        {
                            id = m_featureNameToIdMap[iter->first];
                            dim = m_featureNameToDimMap[iter->first];

                            if ((m_featuresBufferMultiIO[id] == nullptr) ||
                                (m_featuresBufferAllocatedMultiIO[id] < (dim * m_mbNumTimeSteps * m_numSeqsPerMB)) /*buffer size changed. can be partial minibatch*/)
                            {
                                m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbNumTimeSteps * m_numSeqsPerMB);
                                m_featuresBufferAllocatedMultiIO[id] = dim * m_mbNumTimeSteps * m_numSeqsPerMB;
                            }

                            if (sizeof(ElemType) == sizeof(float))
                            {
                                for (size_t j = startFr, k = 0; j < endFr; j++, k++) // column major, so iterate columns
                                {
                                    // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                                    memcpy_s(&m_featuresBufferMultiIO[id].get()[(k * m_numSeqsPerMB + i) * dim], 
                                             sizeof(ElemType) * dim, 
                                             &m_featuresBufferMultiUtt[i].get()[j * dim + m_featuresStartIndexMultiUtt[id + i * numOfFea]], 
                                             sizeof(ElemType) * dim);
                                }
                            }
                            else // double: must type-cast, cannot memcpy()
                            {
                                for (size_t j = startFr, k = 0; j < endFr; j++, k++) // column major, so iterate columns in outside loop
                                {
                                    for (int d = 0; d < dim; d++)
                                    {
                                        m_featuresBufferMultiIO[id].get()[(k * m_numSeqsPerMB + i) * dim + d] = 
                                            m_featuresBufferMultiUtt[i].get()[j * dim + d + m_featuresStartIndexMultiUtt[id + i * numOfFea]];
                                    }
                                }
                            }
                        }
                        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                        {
                            id = m_labelNameToIdMap[iter->first];
                            dim = m_labelNameToDimMap[iter->first];
                            if ((m_labelsBufferMultiIO[id] == nullptr) ||
                                (m_labelsBufferAllocatedMultiIO[id] < (dim * m_mbNumTimeSteps * m_numSeqsPerMB)))
                            {
                                m_labelsBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbNumTimeSteps * m_numSeqsPerMB);
                                m_labelsBufferAllocatedMultiIO[id] = dim * m_mbNumTimeSteps * m_numSeqsPerMB;
                            }

                            for (size_t j = startFr, k = 0; j < endFr; j++, k++)
                            {
                                for (int d = 0; d < dim; d++)
                                {
                                    m_labelsBufferMultiIO[id].get()[(k * m_numSeqsPerMB + i) * dim + d] = 
                                        m_labelsBufferMultiUtt[i].get()[j * dim + d + m_labelsStartIndexMultiUtt[id + i * numOfLabel]];
                                }
                            }
                        }
                    }
                    m_processedFrame[i] += m_mbNumTimeSteps;
                }
                else // if (startFr + m_mbNumTimeSteps < m_numFramesToProcess[i])   (in this else branch, utterance ends inside this minibatch)
                {
                    // utterance ends: first copy this segment (later, we will pack more utterances in)
                    assert(startFr == m_processedFrame[i]);
                    actualmbsize[i] = m_numFramesToProcess[i] - startFr; // parallel sequence is used up to this point
                    const size_t endFr = startFr + actualmbsize[i];      // end frame in sequence
                    assert(endFr == m_numFramesToProcess[i]);            // we are at the end

                    // fill frames for the tail of this utterance
                    for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
                    {
                        // dereference matrix that corresponds to key (input/output name) and
                        // populate based on whether its a feature or a label
                        Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first); // can be features or labels

                        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                        {
                            id = m_featureNameToIdMap[iter->first];
                            dim = m_featureNameToDimMap[iter->first];

                            if ((m_featuresBufferMultiIO[id] == nullptr) ||
                                (m_featuresBufferAllocatedMultiIO[id] < (dim * m_mbNumTimeSteps * m_numSeqsPerMB)) /*buffer size changed. can be partial minibatch*/)
                            {
                                m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbNumTimeSteps * m_numSeqsPerMB);
                                m_featuresBufferAllocatedMultiIO[id] = dim * m_mbNumTimeSteps * m_numSeqsPerMB;
                            }

                            if (sizeof(ElemType) == sizeof(float))
                            {
                                for (size_t j = startFr, k = 0; j < endFr; j++, k++) // column major, so iterate columns
                                {
                                    // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                                    memcpy_s(&m_featuresBufferMultiIO[id].get()[(k * m_numSeqsPerMB + i) * dim], 
                                             sizeof(ElemType) * dim, 
                                             &m_featuresBufferMultiUtt[i].get()[j * dim + m_featuresStartIndexMultiUtt[id + i * numOfFea]], 
                                             sizeof(ElemType) * dim);
                                }
                            }
                            else
                            {
                                for (size_t j = startFr, k = 0; j < endFr; j++, k++) // column major, so iterate columns in outside loop
                                {
                                    for (int d = 0; d < dim; d++)
                                    {
                                        m_featuresBufferMultiIO[id].get()[(k * m_numSeqsPerMB + i) * dim + d] = 
                                            m_featuresBufferMultiUtt[i].get()[j * dim + d + m_featuresStartIndexMultiUtt[id + i * numOfFea]];
                                    }
                                }
                            }
                        }
                        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                        {
                            id = m_labelNameToIdMap[iter->first];
                            dim = m_labelNameToDimMap[iter->first];
                            if ((m_labelsBufferMultiIO[id] == nullptr) ||
                                (m_labelsBufferAllocatedMultiIO[id] < (dim * m_mbNumTimeSteps * m_numSeqsPerMB)))
                            {
                                m_labelsBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbNumTimeSteps * m_numSeqsPerMB);
                                m_labelsBufferAllocatedMultiIO[id] = dim * m_mbNumTimeSteps * m_numSeqsPerMB;
                            }

                            for (size_t j = startFr, k = 0; j < endFr; j++, k++)
                            {
                                for (int d = 0; d < dim; d++)
                                {
                                    m_labelsBufferMultiIO[id].get()[(k * m_numSeqsPerMB + i) * dim + d] = 
                                        m_labelsBufferMultiUtt[i].get()[j * dim + d + m_labelsStartIndexMultiUtt[id + i * numOfLabel]];
                                }
                            }
                        }
                    }
                    m_processedFrame[i] += (endFr - startFr);               // advance the cursor
                    assert(m_processedFrame[i] == m_numFramesToProcess[i]); // we must be at the end
                    m_switchFrame[i] = actualmbsize[i];
                    // if (actualmbsize[i] != 0)
                    //    m_pMBLayout->Set(i, actualmbsize[i] - 1, MinibatchPackingFlags::SequenceEnd); // NOTE: this ORs, while original code overwrote in matrix but ORed into vector
                    // at this point, we completed an utterance--fill the rest with the next utterance

                    // BUGBUG: We should fill in a loop until we fill the minibatch for the case where just one ReNew is not sufficient
                    // to fill up the remaining slots in the minibatch
                    bool reNewSucc = ReNewBufferForMultiIO(i);
                    if (actualmbsize[i] < m_mbNumTimeSteps) // we actually have space
                    {
                        if (reNewSucc) // we actually have another utterance to start here
                        {
                            const size_t startT = m_switchFrame[i];
                            // Have to take the min, if the next sequence is shorted then truncation length.
                            const size_t endT = std::min(m_mbNumTimeSteps, startT + m_numFramesToProcess[i]);
                            // Note: Don't confuse startT/endT with startFr/endFr above.

                            // add sequence to MBLayout
                            m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, i, startT, startT + m_numFramesToProcess[i]);

                            // copy the data
                            for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
                            {
                                // dereference matrix that corresponds to key (input/output name) and
                                // populate based on whether its a feature or a label
                                // Matrix<ElemType>& data = *matrices[iter->first]; // can be features or labels

                                if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                                {
                                    id = m_featureNameToIdMap[iter->first];
                                    dim = m_featureNameToDimMap[iter->first];
                                    if (sizeof(ElemType) == sizeof(float))
                                    {
                                        for (size_t t = startT, fr = 0; t < endT; t++, fr++) // column major, so iterate columns
                                        {
                                            // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns (for SSE alignment)
                                            memcpy_s(&m_featuresBufferMultiIO[id].get()[(t * m_numSeqsPerMB + i) * dim], 
                                                     sizeof(ElemType) * dim, 
                                                     &m_featuresBufferMultiUtt[i].get()[fr * dim + m_featuresStartIndexMultiUtt[id + i * numOfFea]], 
                                                     sizeof(ElemType) * dim);
                                        }
                                    }
                                    else
                                    {
                                        for (size_t t = startT, fr = 0; t < endT; t++, fr++) // column major, so iterate columns in outside loop
                                        {
                                            for (int d = 0; d < dim; d++)
                                            {
                                                m_featuresBufferMultiIO[id].get()[(t * m_numSeqsPerMB + i) * dim + d] = 
                                                    m_featuresBufferMultiUtt[i].get()[fr * dim + d + m_featuresStartIndexMultiUtt[id + i * numOfFea]];
                                            }
                                        }
                                    }
                                }
                                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                                {
                                    id = m_labelNameToIdMap[iter->first];
                                    dim = m_labelNameToDimMap[iter->first];
                                    for (size_t t = startT, fr = 0; t < endT; t++, fr++)
                                    {
                                        for (int d = 0; d < dim; d++)
                                        {
                                            m_labelsBufferMultiIO[id].get()[(t * m_numSeqsPerMB + i) * dim + d] = 
                                                m_labelsBufferMultiUtt[i].get()[fr * dim + d + m_labelsStartIndexMultiUtt[id + i * numOfLabel]];
                                        }
                                    }
                                }
                            }

                            m_processedFrame[i] += (endT - startT);

                            // BUGBUG: since we currently cannot fill >1 utterances, at least let's check
                            size_t a = actualmbsize[i] + (endT - startT);

                            // actualmbsize[i] += (endT - startT);          // BUGBUG: don't we need something like this?
                            if (a < m_mbNumTimeSteps)
                            {
                                fprintf(stderr, "GetMinibatchToTrainOrTest(): WARNING: Packing a second utterance did still not fill all time slots; filling slots from %d on as gaps.\n", (int) a);
                                // declare the rest as a gap
                                m_pMBLayout->AddGap(i, a, m_mbNumTimeSteps);

                                // Have to renew, so that there is data for the next read.
                                ReNewBufferForMultiIO(i);
                            }
                        }
                        else // we did have space for more, but no more data is available. BUGBUG: we should update actualmbsize[i] above and re-test here
                        {
                            // declare the rest as a gap
                            m_pMBLayout->AddGap(i, actualmbsize[i], m_mbNumTimeSteps);
                        }
                    } // if (actualmbsize[i] < m_mbNumTimeSteps)         // we actually have space
                }
            } // for (size_t i = 0; i < m_numSeqsPerMB; i++)
            // we are done filling all parallel sequences

            for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
            {
                // dereference matrix that corresponds to key (input/output name) and
                // populate based on whether its a feature or a label
                Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first); // can be features or labels
                if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
                {
                    id = m_featureNameToIdMap[iter->first];
                    dim = m_featureNameToDimMap[iter->first];
                    data.SetValue(dim, m_mbNumTimeSteps * m_numSeqsPerMB, data.GetDeviceId(), m_featuresBufferMultiIO[id].get(), matrixFlagNormal);
                }
                else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
                {
                    id = m_labelNameToIdMap[iter->first];
                    dim = m_labelNameToDimMap[iter->first];
                    data.SetValue(dim, m_mbNumTimeSteps * m_numSeqsPerMB, data.GetDeviceId(), m_labelsBufferMultiIO[id].get(), matrixFlagNormal);
                }
            }
            skip = false;
        }           // if truncated then else
    } while (skip); // keep going if we didn't get the right size minibatch

    if (m_verbosity > 2)
    {
        aggregateTimer.Stop();
        double totalMBReadTime = aggregateTimer.ElapsedSeconds();
        fprintf(stderr, "Total Minibatch read time = %.8g\n", totalMBReadTime);
    }

    return true;
}

// copy an utterance into the minibatch given a location (parallel-sequence index, start frame)
// TODO: This should use DataFor(). But for that, DataFor() will have to move out from ComputationNode. Ah, it has!
template <class ElemType>
void HTKMLFReader<ElemType>::fillOneUttDataforParallelmode(StreamMinibatchInputs& matrices, size_t startFr,
                                                           size_t framenum, size_t channelIndex, size_t sourceChannelIndex)
{
    size_t id;
    size_t dim;
    size_t numOfFea = m_featuresBufferMultiIO.size();
    size_t numOfLabel = m_labelsBufferMultiIO.size();

    for (auto iter = matrices.begin(); iter != matrices.end(); iter++)
    {
        // dereference matrix that corresponds to key (input/output name) and
        // populate based on whether its a feature or a label
        Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first); // can be features or labels

        if (m_nameToTypeMap[iter->first] == InputOutputTypes::real)
        {
            id = m_featureNameToIdMap[iter->first];
            dim = m_featureNameToDimMap[iter->first];

            if (m_featuresBufferMultiIO[id] == nullptr || m_featuresBufferAllocatedMultiIO[id] < dim * m_mbNumTimeSteps * m_numSeqsPerMB)
            {
                m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbNumTimeSteps * m_numSeqsPerMB);
                memset(m_featuresBufferMultiIO[id].get(), 0, sizeof(ElemType) * dim * m_mbNumTimeSteps * m_numSeqsPerMB);
                m_featuresBufferAllocatedMultiIO[id] = dim * m_mbNumTimeSteps * m_numSeqsPerMB;
            }

            if (sizeof(ElemType) == sizeof(float))
            {
                for (size_t j = 0, k = startFr; j < framenum; j++, k++) // column major, so iterate columns
                {
                    // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                    memcpy_s(&m_featuresBufferMultiIO[id].get()[(k * m_numSeqsPerMB + channelIndex) * dim], sizeof(ElemType) * dim, &m_featuresBufferMultiUtt[sourceChannelIndex].get()[j * dim + m_featuresStartIndexMultiUtt[id + sourceChannelIndex * numOfFea]], sizeof(ElemType) * dim);
                }
            }
            else
            {
                for (size_t j = 0, k = startFr; j < framenum; j++, k++) // column major, so iterate columns in outside loop
                {
                    for (int d = 0; d < dim; d++)
                    {
                        m_featuresBufferMultiIO[id].get()[(k * m_numSeqsPerMB + channelIndex) * dim + d] = m_featuresBufferMultiUtt[sourceChannelIndex].get()[j * dim + d + m_featuresStartIndexMultiUtt[id + sourceChannelIndex * numOfFea]];
                    }
                }
            }
        }
        else if (m_nameToTypeMap[iter->first] == InputOutputTypes::category)
        {
            id = m_labelNameToIdMap[iter->first];
            dim = m_labelNameToDimMap[iter->first];
            if (m_labelsBufferMultiIO[id] == nullptr || m_labelsBufferAllocatedMultiIO[id] < dim * m_mbNumTimeSteps * m_numSeqsPerMB)
            {
                m_labelsBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), dim * m_mbNumTimeSteps * m_numSeqsPerMB);
                memset(m_labelsBufferMultiIO[id].get(), 0, sizeof(ElemType) * dim * m_mbNumTimeSteps * m_numSeqsPerMB);
                m_labelsBufferAllocatedMultiIO[id] = dim * m_mbNumTimeSteps * m_numSeqsPerMB;
            }

            for (size_t j = 0, k = startFr; j < framenum; j++, k++)
            {
                for (int d = 0; d < dim; d++)
                {
                    m_labelsBufferMultiIO[id].get()[(k * m_numSeqsPerMB + channelIndex) * dim + d] = 
                        m_labelsBufferMultiUtt[sourceChannelIndex].get()[j * dim + d + m_labelsStartIndexMultiUtt[id + sourceChannelIndex * numOfLabel]];
                }
            }
        }
    }
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
                fprintf(stderr, "GetMinibatchToWrite: feature node %ls specified in reader not found in the network\n", iter->first.c_str());
                RuntimeError("GetMinibatchToWrite: feature node specified in reader not found in the network.");
            }
        }

        /*
        for (auto iter=matrices.begin();iter!=matrices.end();iter++)
        {
        if (m_featureNameToIdMap.find(iter->first)==m_featureNameToIdMap.end())
        RuntimeError(msra::strfun::strprintf("minibatch requested for input node %ws not found in reader - cannot generate input\n",iter->first.c_str()));
        }
        */
        m_checkDictionaryKeys = false;
    }

    if (m_inputFileIndex < m_inputFilesMultiIO[0].size())
    {
        m_fileEvalSource->Reset();

        // load next file (or set of files)
        size_t nfr = 0;
        foreach_index (i, m_inputFilesMultiIO)
        {
            msra::asr::htkfeatreader reader;

            const auto path = reader.parse(m_inputFilesMultiIO[i][m_inputFileIndex]);
            // read file
            msra::dbn::matrix feat;
            string featkind;
            unsigned int sampperiod;
            msra::util::attempt(5, [&]()
            {
                reader.read(path, featkind, sampperiod, feat); // whole file read as columns of feature vectors
            });

            if (i == 0)
            {
                nfr = feat.cols();
            }
            else if (feat.cols() == 1 && nfr > 1)
            { 
                // This broadcasts a vector to be multiple columns, as needed for i-vector support
                msra::dbn::matrix feat_col(feat);
                feat.resize(feat.rows(), nfr);
                for (size_t i = 0; i < feat.rows(); i++)
                    for (size_t j = 0; j < feat.cols(); j++)
                        feat(i, j) = feat_col(i, 0);
            }

            fprintf(stderr, "evaluate: reading %d frames of %ls\n", (int) feat.cols(), ((wstring) path).c_str());
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
                Matrix<ElemType>& data = matrices.GetInputMatrix<ElemType>(iter->first); // can be features or labels   (TODO: Really? Didn't we just ^^^ check that it is 'real'?)
                size_t id = m_featureNameToIdMap[iter->first];
                size_t dim = m_featureNameToDimMap[iter->first];

                const msra::dbn::matrix feat = m_fileEvalSource->ChunkOfFrames(id);

                // update the MBLayout
                if (first)
                {
                    m_pMBLayout->Init(1, feat.cols());
                    m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, feat.cols()); // feat.cols() == number of time steps here since we only have one parallel sequence
                    // m_pMBLayout->Set(0, 0, MinibatchPackingFlags::SequenceStart);
                    // m_pMBLayout->SetWithoutOr(0, feat.cols() - 1, MinibatchPackingFlags::SequenceEnd);  // BUGBUG: using SetWithoutOr() because original code did; but that seems inconsistent
                    first = false;
                }

                // copy the features over to our array type
                assert(feat.rows() == dim);
                dim; // check feature dimension matches what's expected

                if ((m_featuresBufferMultiIO[id] == nullptr) ||
                    (m_featuresBufferAllocatedMultiIO[id] < (feat.rows() * feat.cols())) /*buffer size changed. can be partial minibatch*/)
                {
                    m_featuresBufferMultiIO[id] = AllocateIntermediateBuffer(data.GetDeviceId(), feat.rows() * feat.cols());
                    m_featuresBufferAllocatedMultiIO[id] = feat.rows() * feat.cols();
                }

                if (sizeof(ElemType) == sizeof(float))
                {
                    for (int j = 0; j < feat.cols(); j++) // column major, so iterate columns
                    {
                        // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                        memcpy_s(&m_featuresBufferMultiIO[id].get()[j * feat.rows()], sizeof(ElemType) * feat.rows(), &feat(0, j), sizeof(ElemType) * feat.rows());
                    }
                }
                else
                {
                    for (int j = 0; j < feat.cols(); j++) // column major, so iterate columns in outside loop
                    {
                        for (int i = 0; i < feat.rows(); i++)
                        {
                            m_featuresBufferMultiIO[id].get()[j * feat.rows() + i] = feat(i, j);
                        }
                    }
                }
                data.SetValue(feat.rows(), feat.cols(), data.GetDeviceId(), m_featuresBufferMultiIO[id].get(), matrixFlagNormal);
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
        if ((i == 0) && !m_truncated)
            m_numFramesToProcess[i] = 0;

        return false;
    }

    size_t numOfFea = m_featuresBufferMultiIO.size();
    size_t numOfLabel = m_labelsBufferMultiIO.size();

    size_t totalFeatNum = 0;
    foreach_index (id, m_featuresBufferAllocatedMultiIO)
    {
        const msra::dbn::matrixstripe featOri = m_mbiter->frames(id);
        size_t fdim = featOri.rows();
        const size_t actualmbsizeOri = featOri.cols();
        m_featuresStartIndexMultiUtt[id + i * numOfFea] = totalFeatNum;
        totalFeatNum = fdim * actualmbsizeOri + m_featuresStartIndexMultiUtt[id + i * numOfFea];
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
        size_t dim = m_labelNameToDimMap[it->first];

        const vector<size_t>& uids = m_mbiter->labels(id);
        size_t actualmbsizeOri = uids.size();
        m_labelsStartIndexMultiUtt[id + i * numOfLabel] = totalLabelsNum;
        totalLabelsNum = m_labelsStartIndexMultiUtt[id + i * numOfLabel] + dim * actualmbsizeOri;
    }

    if ((m_labelsBufferMultiUtt[i] == NULL) || (m_labelsBufferAllocatedMultiUtt[i] < totalLabelsNum))
    {
        m_labelsBufferMultiUtt[i] = AllocateIntermediateBuffer(-1 /*CPU */, totalLabelsNum);
        m_labelsBufferAllocatedMultiUtt[i] = totalLabelsNum;
    }

    memset(m_labelsBufferMultiUtt[i].get(), 0, sizeof(ElemType) * totalLabelsNum);

    bool first = true;
    foreach_index (id, m_featuresBufferMultiIO)
    {
        const msra::dbn::matrixstripe featOri = m_mbiter->frames(id);
        const size_t actualmbsizeOri = featOri.cols();
        size_t fdim = featOri.rows();
        if (first)
        {
            m_numFramesToProcess[i] = actualmbsizeOri;
            first = false;
        }
        else
        {
            if (m_numFramesToProcess[i] != actualmbsizeOri)
            {
                RuntimeError("The multi-IO features has inconsistent number of frames!");
            }
        }
        assert(actualmbsizeOri == m_mbiter->currentmbframes());

        if (sizeof(ElemType) == sizeof(float))
        {
            for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns
            {
                // copy over the entire column at once, need to do this because SSEMatrix may have gaps at the end of the columns
                memcpy_s(&m_featuresBufferMultiUtt[i].get()[k * fdim + m_featuresStartIndexMultiUtt[id + i * numOfFea]], sizeof(ElemType) * fdim, &featOri(0, k), sizeof(ElemType) * fdim);
            }
        }
        else
        {
            for (int k = 0; k < actualmbsizeOri; k++) // column major, so iterate columns in outside loop
            {
                for (int d = 0; d < featOri.rows(); d++)
                {
                    m_featuresBufferMultiUtt[i].get()[k * featOri.rows() + d + m_featuresStartIndexMultiUtt[id + i * numOfFea]] = featOri(d, k);
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
                    m_labelsBufferMultiUtt[i].get()[k * dim + j + m_labelsStartIndexMultiUtt[id + i * numOfLabel]] = m_labelToTargetMapMultiIO[id][labelId][j];
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
                m_labelsBufferMultiUtt[i].get()[k * dim + uids[k] + m_labelsStartIndexMultiUtt[id + i * numOfLabel]] = (ElemType) 1;
            }
        }
    }

    // lattice
    if (m_latticeBufferMultiUtt[i] != NULL)
    {
        m_latticeBufferMultiUtt[i].reset();
    }

    if (m_mbiter->haslattice())
    {
        m_latticeBufferMultiUtt[i] = std::move(m_mbiter->lattice(0));
        m_phoneboundaryIDBufferMultiUtt[i].clear();
        m_phoneboundaryIDBufferMultiUtt[i] = m_mbiter->bounds();
        m_labelsIDBufferMultiUtt[i].clear();
        m_labelsIDBufferMultiUtt[i] = m_mbiter->labels();
    }

    m_processedFrame[i] = 0;

    Timer mbIterAdvancementTimer;
    if (m_verbosity > 2)
        mbIterAdvancementTimer.Start();

    // Advance the MB iterator until we find some data or reach the end of epoch
    do
    {
        (*m_mbiter)++;
    } while ((m_mbiter->currentmbframes() == 0) && *m_mbiter);

    if (m_verbosity > 2)
    {
        mbIterAdvancementTimer.Stop();
        double advancementTime = mbIterAdvancementTimer.ElapsedSeconds();
        fprintf(stderr, "Time to advance mbiter = %.8g\n", advancementTime);
    }

    if (!(*m_mbiter))
        m_noData = true;

    return true;
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
void HTKMLFReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& labelMapping)
{
    m_idToLabelMap = labelMapping;
}

template <class ElemType>
size_t HTKMLFReader<ElemType>::ReadLabelToTargetMappingFile(const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap)
{
    if (labelListFile == L"")
        RuntimeError("HTKMLFReader::ReadLabelToTargetMappingFile(): cannot read labelToTargetMappingFile without a labelMappingFile!");

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
    RuntimeError("GetData not supported in HTKMLFReader");
}

template <class ElemType>
bool HTKMLFReader<ElemType>::DataEnd()
{
    // each minibatch is considered a "sentence"
    // for the truncated BPTT, we need to support check wether it's the end of data
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
        sentenceEnd[i] = m_switchFrame[i];
}

template <class ElemType>
void HTKMLFReader<ElemType>::CopyMBLayoutTo(MBLayoutPtr pMBLayout)
{
    pMBLayout->CopyFrom(m_pMBLayout);
}

template <class ElemType>
size_t HTKMLFReader<ElemType>::GetNumParallelSequences()
{
    if (!m_frameMode)
        if (m_numSeqsPerMB != m_pMBLayout->GetNumParallelSequences())
            LogicError("HTKMLFReader: Number of parallel sequences in m_pMBLayout did not get set to m_numSeqsPerMB.");
    return m_pMBLayout->GetNumParallelSequences(); // (this function is only used for validation anyway)
}

// GetFileConfigNames - determine the names of the features and labels sections in the config file
// features - [in,out] a vector of feature name strings
// labels - [in,out] a vector of label name strings
template <class ElemType>
template <class ConfigRecordType>
void HTKMLFReader<ElemType>::GetDataNamesFromConfig(const ConfigRecordType& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels,
                                                    std::vector<std::wstring>& hmms, std::vector<std::wstring>& lattices)
{
    for (const auto& id : readerConfig.GetMemberIds())
    {
        if (!readerConfig.CanBeConfigRecord(id))
            continue;

        const ConfigRecordType& temp = readerConfig(id);
        // see if we have a config parameters that contains a "file" element, it's a sub key, use it
        if (temp.ExistsCurrent(L"scpFile"))
        {
            features.push_back(id);
        }
        else if (temp.ExistsCurrent(L"mlfFile") || temp.ExistsCurrent(L"mlfFileList"))
        {
            labels.push_back(id);
        }
        else if (temp.ExistsCurrent(L"phoneFile"))
        {
            hmms.push_back(id);
        }
        else if (temp.ExistsCurrent(L"denlatTocFile"))
        {
            lattices.push_back(id);
        }
    }
}

template <class ElemType>
void HTKMLFReader<ElemType>::ExpandDotDotDot(wstring& featPath, const wstring& scpPath, wstring& scpDirCached)
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

template <class ElemType>
unique_ptr<CUDAPageLockedMemAllocator>& HTKMLFReader<ElemType>::GetCUDAAllocator(int deviceID)
{
    if (m_cudaAllocator != nullptr)
    {
        if (m_cudaAllocator->GetDeviceId() != deviceID)
        {
            m_cudaAllocator.reset(nullptr);
        }
    }

    if (m_cudaAllocator == nullptr)
    {
        m_cudaAllocator.reset(new CUDAPageLockedMemAllocator(deviceID));
    }

    return m_cudaAllocator;
}

template <class ElemType>
std::shared_ptr<ElemType> HTKMLFReader<ElemType>::AllocateIntermediateBuffer(int deviceID, size_t numElements)
{
    if (deviceID >= 0)
    {
        // Use pinned memory for GPU devices for better copy performance
        size_t totalSize = sizeof(ElemType) * numElements;
        return std::shared_ptr<ElemType>((ElemType*) GetCUDAAllocator(deviceID)->Malloc(totalSize), 
                                         [this, deviceID](ElemType* p)
                                         {
                                             this->GetCUDAAllocator(deviceID)->Free((char*) p);
                                         });
    }
    else
    {
        return std::shared_ptr<ElemType>(new ElemType[numElements], 
                                         [](ElemType* p)
                                         {
                                             delete[] p;
                                         });
    }
}

template class HTKMLFReader<float>;
template class HTKMLFReader<double>;
} } }
