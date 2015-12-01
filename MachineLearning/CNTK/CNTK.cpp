//
// <copyright file="CNTK.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// cn.cpp : Defines the entry point for the console application.
//

// TODO: should we split all these DoXXX() up into separate commands? Mainly to separate common vs. non-standard/special ones?

#define _CRT_NONSTDC_NO_DEPRECATE   // make VS accept POSIX functions without _

#include "stdafx.h"
#include "Actions.h"
#include <string>
#include <chrono>
#include <algorithm>
#if defined(_WIN32)
#include "io.h"
#endif
#include "buildinfo.h"
#include "hostname.h"
#ifdef LEAKDETECT
#include "vld.h" // for memory leak detection
#endif
#include <vector>
#include <iostream>
#include <queue>
#include <set>
#include <memory>

#include "Basics.h"
#include "ComputationNetwork.h"
#include "ComputationNode.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "SimpleNetworkBuilder.h"
#include "NDLNetworkBuilder.h"
#include "SynchronousExecutionEngine.h"
#include "ModelEditLanguage.h"
#include "CPUMatrix.h"  // used for SetNumThreads()
#include "SGD.h"
#include "MPIWrapper.h"
#include "commandArgUtil.h"
#include "MultiNetworksSGD.h"
#include "SimpleEvaluator.h"
#include "SimpleOutputWriter.h"
#include "MultiNetworksEvaluator.h"
#include "BestGpu.h"
#include "ProgressTracing.h"
#include "fileutil.h"
#include "ScriptableObjects.h"
#include "BrainScriptEvaluator.h"
#include "BrainScriptParser.h"

#ifndef let
#define let const auto
#endif

// TODO: Get rid of this global
Microsoft::MSR::CNTK::MPIWrapper *g_mpi = nullptr;

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

// internal test routine forward declaration
template <typename ElemType>
void TestCn(const ConfigParameters& config);

template <typename ElemType>
void DoEvalBeamSearch(const ConfigParameters& config, IDataReader<ElemType>& reader);

template <typename T>
struct compare_second
{
    bool operator()(const T &lhs, const T &rhs) const { return lhs.second < rhs.second; }
};

void RedirectStdErr(wstring logpath)
{
    fprintf(stderr, "Redirecting stderr to file %S\n", logpath.c_str());
    auto f = make_shared<File>(logpath.c_str(), fileOptionsWrite | fileOptionsText);
    if (dup2(fileno(*f), 2) == -1)
    {
        RuntimeError("unexpected failure to redirect stderr to log file");
    }
    setvbuf(stderr, NULL, _IONBF, 16384);   // unbuffer it
    static auto fKept = f;                  // keep it around (until it gets changed)
}

std::string WCharToString(const wchar_t* wst)
{
    std::wstring ws(wst);
    std::string s(ws.begin(), ws.end());
    s.assign(ws.begin(), ws.end());
    return s;
}

template <typename ElemType>
void DumpNodeInfo(const ConfigParameters& config)
{
    wstring modelPath = config(L"modelPath");
    wstring nodeName = config(L"nodeName", L"__AllNodes__");
	wstring nodeNameRegexStr = config(L"nodeNameRegex", L"");
    wstring defOutFilePath = modelPath + L"." + nodeName + L".txt";
    wstring outputFile = config(L"outputFile", defOutFilePath);
    bool printValues = config(L"printValues", true);

    ComputationNetwork net(-1);  //always use CPU
    net.LoadFromFile<ElemType>(modelPath);
    net.DumpNodeInfoToFile(nodeName, printValues, outputFile, nodeNameRegexStr);
}

template <typename ElemType>
void DoEvalBase(const ConfigParameters& config, IDataReader<ElemType>& reader)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;

    int traceLevel = config(L"traceLevel", "0");
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");

    ConfigArray evalNodeNames = config(L"evalNodeNames", "");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelPath);

    SimpleEvaluator<ElemType> eval(net, numMBsToShowResult, traceLevel);
    eval.Evaluate(&reader, evalNodeNamesVector, mbSize[0], epochSize);
}

template <typename ElemType>
void DoEval(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DoEvalBase(config, testDataReader);
}

// Special early implementation of RNNs by emulating them as a DNN.
// The code is very restricted to simple RNNs. 
// The idea can be used for more complicated network but need to know which nodes are stateful or time-dependent so that unroll is done in a correct way to represent recurrent networks. 
// TODO: can probably be removed.
template <typename ElemType>
void DoEvalUnroll(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;
    wstring path2EvalResults = config(L"path2EvalResults", L"");

    auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelPath);

    MultiNetworksEvaluator<ElemType> eval(net);
    double evalEntropy;
    eval.EvaluateUnroll(&testDataReader, mbSize[0], evalEntropy, path2EvalResults == L"" ? nullptr : path2EvalResults.c_str(), epochSize);
}

template <typename ElemType>
void DoCrossValidate(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;

    ConfigArray cvIntervalConfig = config(L"crossValidationInterval");
    intargvector cvInterval = cvIntervalConfig;

    size_t sleepSecondsBetweenRuns = config(L"sleepTimeBetweenRuns", "0");

    int traceLevel = config(L"traceLevel", "0");
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");

    ConfigArray evalNodeNames = config(L"evalNodeNames", "");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    std::vector<std::vector<double>> cvErrorResults;
    std::vector<std::wstring> cvModels;

    DataReader<ElemType> cvDataReader(readerConfig);

    bool finalModelEvaluated = false;
    for (size_t i = cvInterval[0]; i <= cvInterval[2]; i += cvInterval[1])
    {
        wstring cvModelPath = msra::strfun::wstrprintf(L"%ls.%lld", modelPath.c_str(), i);

        if (!fexists(cvModelPath))
        {
            fprintf(stderr, "model %ls does not exist.\n", cvModelPath.c_str());
            if (finalModelEvaluated || !fexists(modelPath))
                continue; // file missing
            else
            {
                cvModelPath = modelPath;
                finalModelEvaluated = true;
            }
        }

        cvModels.push_back(cvModelPath);
        auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, cvModelPath);

        SimpleEvaluator<ElemType> eval(net, numMBsToShowResult, traceLevel);

        fprintf(stderr, "model %ls --> \n", cvModelPath.c_str());
        auto evalErrors = eval.Evaluate(&cvDataReader, evalNodeNamesVector, mbSize[0], epochSize);
        cvErrorResults.push_back(evalErrors);

        ::Sleep(1000 * sleepSecondsBetweenRuns);
    }

    //find best model
    if (cvErrorResults.size() == 0)
    {
        LogicError("No model is evaluated.");
    }

    std::vector<double> minErrors;
    std::vector<int> minErrIds;
    std::vector<double> evalErrors = cvErrorResults[0];
    for (int i = 0; i < evalErrors.size(); ++i)
    {
        minErrors.push_back(evalErrors[i]);
        minErrIds.push_back(0);
    }

    for (int i = 0; i<cvErrorResults.size(); i++)
    {
        evalErrors = cvErrorResults[i];
        for (int j = 0; j<evalErrors.size(); j++)
        {
            if (evalErrors[j] < minErrors[j])
            {
                minErrors[j] = evalErrors[j];
                minErrIds[j] = i;
            }
        }
    }

    fprintf(stderr, "Best models:\n");
    fprintf(stderr, "------------\n");
    for (int i = 0; i < minErrors.size(); ++i)
    {
        fprintf(stderr, "Based on Err[%d]: Best model = %ls with min err %.8g\n", i, cvModels[minErrIds[i]].c_str(), minErrors[i]);
    }
}

template <typename ElemType>
void DoWriteOutput(const ConfigParameters& config)
{
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));
    readerConfig.Insert("randomize", "None");  //we don't want randomization when output results

    DataReader<ElemType> testDataReader(readerConfig);

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "2048");
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;

    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    ConfigArray outputNodeNames = config(L"outputNodeNames", "");
    vector<wstring> outputNodeNamesVector;
    for (int i = 0; i < outputNodeNames.size(); ++i)
    {
        outputNodeNamesVector.push_back(outputNodeNames[i]);
    }

    auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelPath);

    SimpleOutputWriter<ElemType> writer(net, 1);

    if (config.Exists("writer"))
    {
        ConfigParameters writerConfig(config(L"writer"));
        bool bWriterUnittest = writerConfig(L"unittest", "false");
        DataWriter<ElemType> testDataWriter(writerConfig);
        writer.WriteOutput(testDataReader, mbSize[0], testDataWriter, outputNodeNamesVector, epochSize, bWriterUnittest);
    }
    else if (config.Exists("outputPath"))
    {
        wstring outputPath = config(L"outputPath"); // crashes if no default given? 
        writer.WriteOutput(testDataReader, mbSize[0], outputPath, outputNodeNamesVector, epochSize);
    }
    //writer.WriteOutput(testDataReader, mbSize[0], testDataWriter, outputNodeNamesVector, epochSize);
}

namespace Microsoft { namespace MSR { namespace CNTK {

    TrainingCriterion ParseTrainingCriterionString(wstring s)
    {
        if (!_wcsicmp(s.c_str(), L"crossEntropyWithSoftmax"))
            return TrainingCriterion::CrossEntropyWithSoftmax;
        if (!_wcsicmp(s.c_str(), L"sequenceWithSoftmax"))
            return TrainingCriterion::SequenceWithSoftmax;
        else if (!_wcsicmp(s.c_str(), L"squareError"))
            return TrainingCriterion::SquareError;
        else if (!_wcsicmp(s.c_str(), L"logistic"))
            return TrainingCriterion::Logistic;
        else if (!_wcsicmp(s.c_str(), L"noiseContrastiveEstimation") || !_wcsicmp(s.c_str(), L"noiseContrastiveEstimationNode"/*spelling error, deprecated*/))
            return TrainingCriterion::NCECrossEntropyWithSoftmax;
        else if (!!_wcsicmp(s.c_str(), L"classCrossEntropyWithSoftmax"))    // (twisted logic to keep compiler happy w.r.t. not returning from LogicError)
            LogicError("trainingCriterion: Invalid trainingCriterion value. Valid values are (crossEntropyWithSoftmax | squareError | logistic | classCrossEntropyWithSoftmax| sequenceWithSoftmax)");
        return TrainingCriterion::ClassCrossEntropyWithSoftmax;
    }

    EvalCriterion ParseEvalCriterionString(wstring s)
    {
        if (!_wcsicmp(s.c_str(), L"errorPrediction"))
            return EvalCriterion::ErrorPrediction;
        else if (!_wcsicmp(s.c_str(), L"crossEntropyWithSoftmax"))
            return EvalCriterion::CrossEntropyWithSoftmax;
        else if (!_wcsicmp(s.c_str(), L"sequenceWithSoftmax"))
            return EvalCriterion::SequenceWithSoftmax;
        else if (!_wcsicmp(s.c_str(), L"classCrossEntropyWithSoftmax"))
            return EvalCriterion::ClassCrossEntropyWithSoftmax;
        else if (!_wcsicmp(s.c_str(), L"noiseContrastiveEstimation") || !_wcsicmp(s.c_str(), L"noiseContrastiveEstimationNode"/*spelling error, deprecated*/))
            return EvalCriterion::NCECrossEntropyWithSoftmax;
        else if (!_wcsicmp(s.c_str(), L"logistic"))
            return EvalCriterion::Logistic;
        else if (!!_wcsicmp(s.c_str(), L"squareError"))
            LogicError("evalCriterion: Invalid trainingCriterion value. Valid values are (errorPrediction | crossEntropyWithSoftmax | squareError | logistic | sequenceWithSoftmax)");
        return EvalCriterion::SquareError;
    }

}}};

template <typename ElemType>
void DoCreateLabelMap(const ConfigParameters& config)
{
    // this gets the section name we are interested in
    std::string section = config(L"section");
    // get that section (probably a peer config section, which works thanks to heirarchal symbol resolution)
    ConfigParameters configSection(config(section));
    ConfigParameters readerConfig(configSection("reader"));
    readerConfig.Insert("allowMapCreation", "true");
    DEVICEID_TYPE deviceId = CPUDEVICE;
    size_t minibatchSize = config(L"minibatchSize", "2048");
    int traceLevel = config(L"traceLevel", "0");
    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;
    GetFileConfigNames(readerConfig, featureNames, labelNames);

    // setup minibatch matrices
    Matrix<ElemType> featuresMatrix(deviceId);
    Matrix<ElemType> labelsMatrix(deviceId);
    std::map<std::wstring, Matrix<ElemType>*> matrices;
    matrices[featureNames[0]] = &featuresMatrix;
    if (labelNames.size() == 0)
        RuntimeError("CreateLabelMap: no labels found to process");

    // now create the reader and loop through the entire dataset to get all the labels
    auto start = std::chrono::system_clock::now();
    for (const std::wstring& labelsName : labelNames)
    {
        // take the last label file defined (the other one might be input)
        matrices[labelsName] = &labelsMatrix;

        // get the label mapping file name
        ConfigParameters labelConfig(readerConfig(labelsName));
        std::string labelMappingFile;
        if (labelConfig.ExistsCurrent(L"labelMappingFile"))
        {
            labelMappingFile = labelConfig(L"labelMappingFile");
        }
        else if (readerConfig.ExistsCurrent(L"labelMappingFile"))
        {
            labelMappingFile = labelConfig(L"labelMappingFile");
        }
        else
        {
            RuntimeError("CreateLabelMap: No labelMappingFile defined");
        }

        if (fexists(labelMappingFile))
        {
            fprintf(stderr, "CreateLabelMap: the label mapping file '%s' already exists, no work to do.\n", labelMappingFile.c_str());
            return;
        }
        fprintf(stderr, "CreateLabelMap: Creating the mapping file '%s' \n", labelMappingFile.c_str());

        DataReader<ElemType> dataReader(readerConfig);

        dataReader.StartMinibatchLoop(minibatchSize, 0, requestDataSize);
        int count = 0;
        while (dataReader.GetMinibatch(matrices))
        {
            Matrix<ElemType>& features = *matrices[featureNames[0]];
            count += features.GetNumCols();
            if (traceLevel > 1)
                fprintf(stderr, "."); // progress meter
        }
        dataReader.StartMinibatchLoop(minibatchSize, 1, requestDataSize);

        // print the results
        if (traceLevel > 0)
            fprintf(stderr, "\nread %d labels and produced %s\n", count, labelMappingFile.c_str());
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;
    if (traceLevel > 1)
        fprintf(stderr, "%f seconds elapsed\n", (float)(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()) / 1000);
}

//////////////////////////////////////////////////////////////////////////
//  for action SVD
//      An action "SVD" performs the following process to transform an existing model: 
//          1.  For a Learnable Parameter A whose name matches with the user specified regex, 
//              A is approximated by two matrice multiplication B*C ; 
//          2.  In order to keep the low-rank structure in training, 
//              the original A node will be replaced by A' whose opertions is Times
//              with its left children being B and right chilren being 
//
//      To use this command,
//          user need to specify: 
//                  1)  modelPath           -- path to the existing model 
//                  2)  outputmodelPath     -- where to write the transformed model 
//                  3)  KeepRatio           -- how many percentage of energy we want to keep
//					4)	AlignedSize			-- the resultant number of signular values is aligned to e.g., 32 or 64  
//                  5)  ParameterName       -- name (regex) of the parameter node we want to perform a SVD decomposition 
//              
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//  helper function for DoParameterSVD 
//////////////////////////////////////////////////////////////////////////
bool ParseSVDConfigFile(wstring fn, map<wstring, float>& config)
{
    msra::files::textreader reader(fn);
    for (; reader;)
    {
        wstring line = reader.wgetline();
        vector<wstring> tokens = msra::strfun::split(line, L"\t ");
        if (tokens.size() != 2)
            return false;
        config[tokens[0]] = (float)msra::strfun::todouble(tokens[1]);
    }
    return true;
}
// a brief on the SVD config file usage 
void SVDConfigFileUsage()
{
    fprintf(stderr, "usage of SVDConfigFile\n");
    fprintf(stderr, "A SVDConfigFile is referred in main config by \"SVDConfig\"\n");
    fprintf(stderr, "Each line in this file specifies a group of Learnable Parameter nodes using regex and the KeepRatio associated with that group\n");
    fprintf(stderr, "An example: \n");
    fprintf(stderr, "W0         1.0\n");
    fprintf(stderr, "W[1-5]     0.4\n");


}
template<class ElemType>
void  DoParameterSVD(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceID = -1;        // use CPU for SVD 
    wstring modelPath = config(L"modelPath");
    wstring outputmodelPath = config(L"outputmodelPath");
    map<wstring, float>     svdconfig;

    float keepratio = config(L"KeepRatio", "0.4");
	size_t AlignedSize = config(L"AlignedSize", "8");
    wstring svdnodeRegex = config(L"NodeNameRegex", L"");
    if (!svdnodeRegex.empty())
    {
        svdconfig[svdnodeRegex] = keepratio;
    }
    else
    {
        // alternatively, user can also use a config to specify KeepRatios for different groups of nodes 
        wstring svdnodeConfigFile = config(L"SVDConfig", L"");
        if (!ParseSVDConfigFile(svdnodeConfigFile, svdconfig))
        {
            SVDConfigFileUsage();
            return;
        }
    }


    if (modelPath.empty())
    {
        fprintf(stderr, "ERROR: in DoParameterSVD, modelPath is empty!\n");
        return;
    }


    ComputationNetwork net(deviceID);
    net.LoadFromFile<ElemType>(modelPath);

    net.PerformSVDecomposition<ElemType>(svdconfig, AlignedSize);
    if (!outputmodelPath.empty())
        net.SaveToFile(outputmodelPath);

}


///
/// for action writeWordAndClassInfo
///
/// read training text file
///
/// the outputs are the vocabulary, word2class and class2idx file with the information below
///     vocabulary format is as follows
///       0      42068  </s>    0
///       1      50770  the 0
///       2      45020  <unk>   1
///       the first column is word index
///       the last column is class index of the word
///       the second column and the third column are for information purpose and 
///       are not really used in generating outputs for later process in the neural networks training
///
///    wrd2cls in dense matrix in[vocab_size X 1].it maps a word to its class id.
///    cls2idx in dense matrix in[nbr_cls X 1].it maps a class to its first word index.
///
/// to be used for class-based entropy, the outputs have the following assumptions
/// A1 : words are sorted so that words that are in the same class are together
///    i.e., wrds2cls[0] <= wrd2cls[1] <= ... <= wrd2cls[vocab_size - 1]
/// A2 : class ids are sorted so that cls2idx[0] < cls2idx[1] < cls2idx[2] < ... < cls2idx[nbr_cls - 1]
template <typename ElemType>
void DoWriteWordAndClassInfo(const ConfigParameters& config)
{
    string inputFile = config(L"inputFile"); // training text file without <unk>
    string outputWord2Cls = config(L"outputWord2Cls");
    string outputVocabFile = config(L"outputVocabFile");
    string outputCls2Index = config(L"outputCls2Index");
    size_t  vocabSize = config(L"vocabSize");
    int  nbrCls = config(L"nbrClass", "0");
    int  cutoff = config(L"cutoff", "1");

    DEVICEID_TYPE deviceId = CPUDEVICE;
    Matrix<ElemType> wrd2cls(deviceId);
    Matrix<ElemType> cls2idx(deviceId);

    //FILE *fp = fopen(inputFile.c_str(), "rt");
    ifstream fp(inputFile.c_str());
    if (!fp)
    {
        RuntimeError("inputFile cannot be read");
    }

    if (nbrCls > 0)
    {
        cls2idx.Resize(nbrCls, 1);
    }
    std::unordered_map<string, double> v_count;

    /// get line
    string str;
    vector<string> vstr;
    long long prevClsIdx = -1;
    string token;
    while (getline(fp, str))
    {
        str.erase(0, str.find_first_not_of(' '));       //prefixing spaces
        str.erase(str.find_last_not_of(' ') + 1);         //surfixing spaces
        int sposition = str.find("</s> ");
        int eposition = str.find(" </s>");
        if (sposition == str.npos)
        {
            str = "</s> " + str;
        }

        if (eposition == str.npos)
        {
            str = str + " </s>";
        }

        vstr = msra::strfun::split(str, "\t ");
        for (int i = 1; i < vstr.size(); i++)
        {
            v_count[vstr[i]]++;
        }
    }
    fp.close();

    std::cerr << "no truncated vocabulary: " << v_count.size() << std::endl;

    std::vector<std::string> m_words;
    std::set<std::string> m_remained_words;
    std::unordered_map<std::string, size_t> m_index;

    std::vector<double> m_count;
    std::vector<int> m_class;// class index of each word

    typedef std::pair<std::string, double> stringdouble;
    std::priority_queue<stringdouble, std::vector<stringdouble>, compare_second<stringdouble> >
        q(compare_second<stringdouble>(), std::vector<stringdouble>(v_count.begin(), v_count.end()));

    size_t wordCountLessCutoff = v_count.size();
    if (cutoff > 0)
        for (std::unordered_map<std::string, double>::iterator iter = v_count.begin(); iter != v_count.end(); iter++)
        {
            if (iter->second <= cutoff)
            {
                wordCountLessCutoff--;
            }
        }
    if (wordCountLessCutoff <= 0)
        RuntimeError("no word remained after cutoff");

    if (vocabSize > wordCountLessCutoff)
    {
        std::cerr << "warning: actual vocabulary size is less than required." << endl;
        std::cerr << "\t\tRequired vocabulary size:" << vocabSize << endl;
        std::cerr << "\t\tActural vocabulary size:" << v_count.size() << endl;
        std::cerr << "\t\tActural vocabulary size after cutoff:" << wordCountLessCutoff << endl;
        std::cerr << "\t\tWe will change to actual vocabulary size: " << wordCountLessCutoff << endl;
        vocabSize = wordCountLessCutoff;
    }
    wrd2cls.Resize(vocabSize, 1);

    std::unordered_map<std::string, double> removed;
    double unkCount = 0;
    size_t size = 0;
    size_t actual_vocab_size = vocabSize - 1;
    while (size < actual_vocab_size  && !q.empty())
    {
        size++;
        std::string word = q.top().first;
        double freq = q.top().second;
        if (word == "<unk>")
        {
            unkCount += freq;
            actual_vocab_size++;
        }
        removed[q.top().first] = q.top().second;
        q.pop();
    }
    while (!q.empty())
    {
        unkCount += q.top().second;
        q.pop();
    }
    removed["<unk>"] = unkCount;
    std::priority_queue<stringdouble, std::vector<stringdouble>, compare_second<stringdouble> >
        p(compare_second<stringdouble>(), std::vector<stringdouble>(removed.begin(), removed.end()));
    cerr << "p.size():" << p.size() << endl;
    m_count.resize(removed.size());
    double total = 0;
    double dd = 0;
    if (nbrCls > 0)
    {
        for (std::unordered_map<std::string, double>::iterator iter = removed.begin(); iter != removed.end(); iter++)
        {
            total += iter->second;
        }

        for (std::unordered_map<std::string, double>::iterator iter = removed.begin(); iter != removed.end(); iter++)
        {
            dd += sqrt(iter->second / total);
        }
    }

    double df = 0;
    size_t class_id = 0;
    m_class.resize(p.size());

    while (!p.empty())
    {
        std::string word = p.top().first;
        double freq = p.top().second;
        if (nbrCls > 0)
        {
            df += sqrt(freq / total) / dd;
            if (df > 1)
            {
                df = 1;
            }

            if (df > 1.0 * (class_id + 1) / nbrCls && class_id < nbrCls)
            {
                class_id++;
            }
        }

        size_t wid = m_words.size();
        bool inserted = m_index.insert(make_pair(word, wid)).second;
        if (inserted)
            m_words.push_back(word);

        m_count[wid] = freq;
        if (nbrCls > 0)
        {
            m_class[wid] = class_id;
        }
        p.pop();
    }

    std::ofstream ofvocab;
    ofvocab.open(outputVocabFile.c_str());
    for (size_t i = 0; i < m_index.size(); i++)
    {
        if (nbrCls > 0)
            wrd2cls(i, 0) = (ElemType)m_class[i];
        long long clsIdx = nbrCls > 0 ? m_class[i] : 0;
        if (nbrCls > 0 && clsIdx != prevClsIdx)
        {
            cls2idx(clsIdx, 0) = (ElemType)i; /// the left boundary of clsIdx
            prevClsIdx = m_class[i];
        }
        ofvocab << "     " << i << "\t     " << m_count[i] << "\t" << m_words[i] << "\t" << clsIdx << std::endl;
    }

    ofvocab.close();
    if (nbrCls > 0)
    {
        /// write the outputs
        msra::files::make_intermediate_dirs(s2ws(outputWord2Cls));
        ofstream ofp(outputWord2Cls.c_str());
        if (!ofp)
            RuntimeError("cannot write to %s", outputWord2Cls.c_str());
        for (size_t r = 0; r < wrd2cls.GetNumRows(); r++)
            ofp << (int)wrd2cls(r, 0) << endl;
        ofp.close();

        msra::files::make_intermediate_dirs(s2ws(outputCls2Index));
        ofp.open(outputCls2Index.c_str());
        if (!ofp)
        {
            RuntimeError("cannot write to %s", outputCls2Index.c_str());
        }

        for (size_t r = 0; r < cls2idx.GetNumRows(); r++)
        {
            ofp << (int)cls2idx(r, 0) << endl;
        }
        ofp.close();
    }
}

template<class ElemType>
class BrainScriptNetworkBuilder : public IComputationNetBuilder<ElemType>
{
    typedef shared_ptr<ComputationNetwork> ComputationNetworkPtr;
    ComputationNetworkPtr m_net;
    ScriptableObjects::ConfigLambdaPtr m_createNetworkFn;
    DEVICEID_TYPE m_deviceId;
public:
    // the constructor remembers the config lambda
    // TODO: Really this should just take the lambda itself, or rather, this class should just be replaced by a lambda. But we need the IConfigRecord for templates to be compile-compatible with old CNTK config.
    BrainScriptNetworkBuilder(const ScriptableObjects::IConfigRecord & config)
    {
        m_deviceId = config[L"deviceId"];   // TODO: only needed for LoadNetworkFromFile() which should go away anyway
        m_createNetworkFn = config[L"createNetwork"].AsPtr<ScriptableObjects::ConfigLambda>();
    }
    // not supported for old CNTK
    BrainScriptNetworkBuilder(const ConfigParameters & config) { NOT_IMPLEMENTED; }

    // build a ComputationNetwork from description language
    virtual /*IComputationNetBuilder::*/ComputationNetworkPtr BuildNetworkFromDescription(ComputationNetwork* = nullptr) override
    {
        vector<ScriptableObjects::ConfigValuePtr> args;    // this lambda has no arguments
        ScriptableObjects::ConfigLambda::NamedParams namedArgs;
        let netValue = m_createNetworkFn->Apply(move(args), move(namedArgs), L"BuildNetworkFromDescription");
        m_net = netValue.AsPtr<ComputationNetwork>();
        if (m_net->GetDeviceId() < 0)
            fprintf(stderr, "BrainScriptNetworkBuilder using CPU\n");
        else
            fprintf(stderr, "BrainScriptNetworkBuilder using GPU %d\n", (int)m_net->GetDeviceId());
        return m_net;
    }

    // load an existing file--this is the same code as for NDLNetworkBuilder.h (OK to copy it here because this is temporary code anyway)
    // TODO: This does not belong into NetworkBuilder, since the code is the same for all. Just create the network and load the darn thing.
    virtual /*IComputationNetBuilder::*/ComputationNetwork* LoadNetworkFromFile(const wstring& modelFileName, bool forceLoad = true,
        bool bAllowNoCriterionNode = false, ComputationNetwork* anotherNetwork = nullptr) override
    {
        if (!m_net || m_net->GetTotalNumberOfNodes() == 0 || forceLoad) //not built or force load   --TODO: why all these options?
        {
            auto net = make_shared<ComputationNetwork>(m_deviceId);
            net->LoadFromFile<ElemType>(modelFileName, FileOptions::fileOptionsBinary, bAllowNoCriterionNode, anotherNetwork);
            m_net = net;
        }
        m_net->ResetEvalTimeStamp();
        return m_net.get();
    }
};

// TODO: decide where these should go. Also, do we need three variables?
extern wstring standardFunctions;
extern wstring commonMacros;
extern wstring computationNodes;

// helper that returns 'float' or 'double' depending on ElemType
template<class ElemType> static const wchar_t * ElemTypeName();
template<> /*static*/ const wchar_t * ElemTypeName<float>()  { return L"float"; }
template<> /*static*/ const wchar_t * ElemTypeName<double>() { return L"double"; }

function<ComputationNetworkPtr(DEVICEID_TYPE)> GetCreateNetworkFn(const ScriptableObjects::IConfigRecord & config)
{
    // createNetwork() is a BrainScript lambda that creates the model
    // We create a C++ wrapper around it, which we then pass to Train().
    auto createNetworkConfigLambda = config[L"createNetwork"].AsPtr<ScriptableObjects::ConfigLambda>();
    return [createNetworkConfigLambda](DEVICEID_TYPE /*deviceId*/)
    {
        // execute the lambda
        vector<ScriptableObjects::ConfigValuePtr> args;    // this lambda has no arguments
        ScriptableObjects::ConfigLambda::NamedParams namedArgs;
        let netValue = createNetworkConfigLambda->Apply(move(args), move(namedArgs), L"BuildNetworkFromDescription");
        // typecast the result to the desired type
        return netValue.AsPtr<ComputationNetwork>();
    };
}
function<ComputationNetworkPtr(DEVICEID_TYPE)> GetCreateNetworkFn(const ConfigParameters &) { NOT_IMPLEMENTED; }  // old CNTK config does not support lambdas

// function to create an object of a certain type, using both old CNTK config and BrainScript
template<class C>
shared_ptr<C> CreateObject(const ScriptableObjects::IConfigRecord & config, const wchar_t * id)
{
    // TODO: CNTK config added "traceLevel = 0" to 'config'. In BS, we cannot do that (IConfigRecord is immutable). Solution: Just say "traceLevel = 0" in the BS macros for readers.
    return config[id].AsPtr<C>();       // BS instantiates this object through this call
}
template<class C>
shared_ptr<C> CreateObject(const ConfigParameters & config, const wchar_t * id)
{
    ConfigParameters readerConfig(config(id));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));        // TODO: fix this by adding it to all config blocks. Easy to fix in BS as 'config with [ traceLevel = 0 ]'.
    return make_shared<C>(readerConfig); // old CNTK config specifies a dictionary which then must be explicitly instantiated
}

template <class ConfigRecordType, typename ElemType>
void DoTrain(const ConfigRecordType & config)
{
    bool makeMode = config(L"makeMode", true);
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    // determine the network-creation function
    // We have several ways to create that network.
    function<ComputationNetworkPtr(DEVICEID_TYPE)> createNetworkFn;

    if (config.Exists(L"createNetwork"))
    {
        createNetworkFn = GetCreateNetworkFn(config); // (we need a separate function needed due to template code)
    }
    else if (config.Exists(L"SimpleNetworkBuilder"))
    {
        const ConfigRecordType & simpleNetworkBuilderConfig(config(L"SimpleNetworkBuilder"));
        auto netBuilder = make_shared<SimpleNetworkBuilder<ElemType>>(simpleNetworkBuilderConfig);  // parses the configuration and stores it in the SimpleNetworkBuilder object
        createNetworkFn = [netBuilder](DEVICEID_TYPE deviceId)
        {
            return shared_ptr<ComputationNetwork>(netBuilder->BuildNetworkFromDescription());       // this operates based on the configuration saved above
        };
    }
    // legacy NDL
    else if (config.Exists(L"NDLNetworkBuilder"))
    {
        const ConfigRecordType & ndlNetworkBuilderConfig(config(L"NDLNetworkBuilder"));
        shared_ptr<NDLBuilder<ElemType>> netBuilder = make_shared<NDLBuilder<ElemType>>(ndlNetworkBuilderConfig);
        createNetworkFn = [netBuilder](DEVICEID_TYPE deviceId)
        {
            return shared_ptr<ComputationNetwork>(netBuilder->BuildNetworkFromDescription());
        };
    }
    // legacy test mode for BrainScript. Will go away once we fully integrate with BS.
    else if (config.Exists(L"ExperimentalNetworkBuilder"))
    {
        // We interface with outer old CNTK config by taking the inner part, which we get as a string, as BrainScript.
        // We prepend a few standard definitions, and also definition of deviceId and precision, which all objects will pull out again when they are being constructed.
        // BUGBUG: We are not getting TextLocations right in this way! Do we need to inject location markers into the source? Moot once we fully switch to BS
        wstring sourceCode = config(L"ExperimentalNetworkBuilder");
        let expr = BS::ParseConfigDictFromString(standardFunctions + computationNodes + commonMacros
            + msra::strfun::wstrprintf(L"deviceId = %d ; precision = '%ls' ; network = new ComputationNetwork ", (int)deviceId, ElemTypeName<ElemType>())  // TODO: check if typeid needs postprocessing
            + sourceCode, vector<wstring>());    // source code has the form [ ... ]
        createNetworkFn = [expr](DEVICEID_TYPE /*deviceId*/)
        {
            // evaluate the parse tree--specifically the top-level field 'network'--which will create the network
            let object = EvaluateField(expr, L"network");                                // this comes back as a BS::Object
            let network = dynamic_pointer_cast<ComputationNetwork>(object);   // cast it
            // This should not really fail since we constructed the source code above such that this is the right type.
            // However, it is possible (though currently not meaningful) to locally declare a different 'precision' value.
            // In that case, the network might come back with a different element type. We need a runtime check for that.
            if (!network)
                RuntimeError("BuildNetworkFromDescription: network has the wrong element type (float vs. double)");
            // success
            network->ResetEvalTimeStamp();
            return network;
        };
    }
    else
    {
        RuntimeError("No network builder found in the config file. NDLNetworkBuilder or SimpleNetworkBuilde must be specified");
    }

    auto dataReader = CreateObject<DataReader<ElemType>>(config, L"reader");

    shared_ptr<DataReader<ElemType>> cvDataReader;
    if (config.Exists(L"cvReader"))
        cvDataReader = CreateObject<DataReader<ElemType>>(config, L"cvReader");

    shared_ptr<SGD<ElemType>> optimizer;
    if (config.Exists(L"optimizer"))
    {
        optimizer = CreateObject<SGD<ElemType>>(config, L"optimizer");
    }
    else // legacy CNTK config syntax: needs a record called 'SGD'
    {
        const ConfigRecordType & configSGD(config(L"SGD"));
        optimizer = make_shared<SGD<ElemType>>(configSGD);
    }

    optimizer->Train(createNetworkFn, deviceId, dataReader.get(), cvDataReader.get(), makeMode);
}

namespace Microsoft { namespace MSR { namespace ScriptableObjects {

    using namespace Microsoft::MSR::CNTK;

    // -----------------------------------------------------------------------
    // register ComputationNode with the ScriptableObject system
    // -----------------------------------------------------------------------

    class TrainAction { };
    template<> shared_ptr<Object> MakeRuntimeObject<TrainAction>(const IConfigRecordPtr configp)
    {
        const IConfigRecord & config = *configp;
        wstring precision = config[L"precision"];            // dispatch on ElemType
        if (precision == L"float")
            DoTrain<IConfigRecord, float>(config);
        else if (precision == L"double")
            DoTrain<IConfigRecord, double>(config);
        else
            RuntimeError("invalid value '%ls' for 'precision', must be 'float' or 'double'", precision.c_str());

        return make_shared<Object>();   // return a dummy object
    }

    // register ComputationNode with the ScriptableObject system
    ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<TrainAction> registerTrainAction(L"TrainAction");
}}}

template <typename ElemType>
void DoAdapt(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    ConfigParameters configSGD(config(L"SGD"));
    bool makeMode = config(L"makeMode", "true");

    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType>* dataReader = new DataReader<ElemType>(readerConfig);

    DataReader<ElemType>* cvDataReader = nullptr;
    ConfigParameters cvReaderConfig(config(L"cvReader", L""));

    if (cvReaderConfig.size() != 0)
    {
        cvReaderConfig.Insert("traceLevel", config(L"traceLevel", "0"));
        cvDataReader = new DataReader<ElemType>(cvReaderConfig);
    }

    wstring origModelFileName = config(L"origModelFileName", L"");
    wstring refNodeName = config(L"refNodeName", L"");

    SGD<ElemType> sgd(configSGD);

    sgd.Adapt(origModelFileName, refNodeName, dataReader, cvDataReader, deviceId, makeMode);

    delete dataReader;
    delete cvDataReader;
}

/**
This implements sequence to sequence translation paper in
http://arxiv.org/pdf/1409.3215.pdf

*/
template <typename ElemType>
void DoEncoderDecoder(const ConfigParameters& config)
{
    vector<IComputationNetBuilder<ElemType>*> netBuilders;
    vector<IDataReader<ElemType>*> trainDataReader;
    vector<IDataReader<ElemType>*> validationDataReader;

    ConfigParameters configSGD = config(L"SGD");
    bool makeMode = config(L"makeMode", "true");
    IComputationNetBuilder<ElemType>* encoderNetBuilder = NULL;
    IComputationNetBuilder<ElemType>* decoderNetBuilder = NULL;

    ConfigParameters readerConfig = config(L"encoderReader");
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType>* encoderDataReader = new DataReader<ElemType>(readerConfig);

    ConfigParameters decoderReaderConfig = config(L"decoderReader");
    DataReader<ElemType>* decoderDataReader = new DataReader<ElemType>(decoderReaderConfig);

    ConfigParameters cvEncoderReaderConfig = config(L"encoderCVReader");
    DataReader<ElemType>* cvEncoderDataReader = new DataReader<ElemType>(cvEncoderReaderConfig);

    ConfigParameters cvDecoderReaderConfig = config(L"decoderCVReader");
    DataReader<ElemType>* cvDecoderDataReader = new DataReader<ElemType>(cvDecoderReaderConfig);

    if (config.Exists("EncoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"EncoderNetworkBuilder");
        encoderNetBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        LogicError("Need encoder network");
    }

    if (config.Exists("DecoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"DecoderNetworkBuilder");
        decoderNetBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        LogicError("Need decoder networks");
    }

    MultiNetworksSGD<ElemType> sgd(configSGD);

    sgd.InitTrainEncoderDecoderWithHiddenStates(configSGD);

    netBuilders.push_back(encoderNetBuilder);
    netBuilders.push_back(decoderNetBuilder);
    trainDataReader.push_back(encoderDataReader);
    trainDataReader.push_back(decoderDataReader);
    validationDataReader.push_back(cvEncoderDataReader);
    validationDataReader.push_back(cvDecoderDataReader);

    sgd.EncoderDecoder(netBuilders, (int)config(L"deviceId"), trainDataReader, validationDataReader, makeMode);

    delete encoderDataReader;
    delete decoderDataReader;
    delete cvEncoderDataReader;
    delete cvDecoderDataReader;
}

/**
DoBidirecionEncoderDecoder
*/
template <typename ElemType>
void DoBidirectionEncoderDecoder(const ConfigParameters& config)
{

    ConfigParameters configSGD = config(L"SGD");
    bool makeMode = config(L"makeMode", "true");
    IComputationNetBuilder<ElemType>* encoderNetBuilder = NULL;
    IComputationNetBuilder<ElemType>* forwardDecoderNetBuilder = NULL;
    IComputationNetBuilder<ElemType>* backwardDecoderNetBuilder = NULL;
    vector<IComputationNetBuilder<ElemType>*> netBuilders;
    vector<IDataReader<ElemType>*> trainDataReader;
    vector<IDataReader<ElemType>*> validationDataReader;

    ConfigParameters readerConfig = config(L"encoderReader");
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType>* encoderDataReader = new DataReader<ElemType>(readerConfig);

    ConfigParameters decoderReaderConfig = config(L"decoderReader");
    DataReader<ElemType>* decoderDataReader = new DataReader<ElemType>(decoderReaderConfig);

    ConfigParameters backwardDecoderReaderConfig = config(L"backwardDecoderReader");
    DataReader<ElemType>* backwardDecoderDataReader = new DataReader<ElemType>(backwardDecoderReaderConfig);

    ConfigParameters cvEncoderReaderConfig = config(L"encoderCVReader");
    DataReader<ElemType>* cvEncoderDataReader = new DataReader<ElemType>(cvEncoderReaderConfig);

    ConfigParameters cvDecoderReaderConfig = config(L"decoderCVReader");
    DataReader<ElemType>* cvDecoderDataReader = new DataReader<ElemType>(cvDecoderReaderConfig);

    ConfigParameters cvBackwardDecoderReaderConfig = config(L"BackwardDecoderCVReader");
    DataReader<ElemType>* cvBackwardDecoderDataReader = new DataReader<ElemType>(cvBackwardDecoderReaderConfig);

    if (config.Exists("EncoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"EncoderNetworkBuilder");
        encoderNetBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
        LogicError("Need encoder network");

    if (config.Exists("DecoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"DecoderNetworkBuilder");
        forwardDecoderNetBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        LogicError("Need decoder networks");
    }

    if (config.Exists("BackwardDecoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"BackwardDecoderNetworkBuilder");
        backwardDecoderNetBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        LogicError("Need decoder networks");
    }

    MultiNetworksSGD<ElemType> sgd(configSGD);

    sgd.InitTrainEncoderDecoderWithHiddenStates(configSGD);

    netBuilders.push_back(encoderNetBuilder);
    netBuilders.push_back(forwardDecoderNetBuilder);
    netBuilders.push_back(backwardDecoderNetBuilder);
    trainDataReader.push_back(encoderDataReader);
    trainDataReader.push_back(decoderDataReader);
    trainDataReader.push_back(backwardDecoderDataReader);
    validationDataReader.push_back(cvEncoderDataReader);
    validationDataReader.push_back(cvDecoderDataReader);
    validationDataReader.push_back(cvBackwardDecoderDataReader);

    sgd.EncoderDecoder(netBuilders, (int)config(L"deviceId"), trainDataReader, validationDataReader, makeMode);

    delete encoderDataReader;
    delete decoderDataReader;
    delete cvEncoderDataReader;
    delete cvDecoderDataReader;
    delete backwardDecoderDataReader;
    delete cvBackwardDecoderDataReader;
}

/**
Originally, this is for testing models trained using the sequence to sequence translation method below
http://arxiv.org/pdf/1409.3215.pdf
Later on, it is extended to be more general to include a sequence of network operations. 
*/
template <typename ElemType>
void DoEvalEncodingBeamSearchDecoding(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    vector<IDataReader<ElemType>*> readers;
    ConfigParameters readerConfig = config(L"encoderReader");
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> encoderReader(readerConfig);

    ConfigParameters decoderReaderConfig = config(L"decoderReader");
    decoderReaderConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> decoderReader(decoderReaderConfig);

    readers.push_back(&encoderReader);
    readers.push_back(&decoderReader);

    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    wstring encoderModelPath = config(L"encoderModelPath");
    wstring decoderModelPath = config(L"decoderModelPath");

    intargvector mbSize = minibatchSize;

    int traceLevel = config(L"traceLevel", "0");
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");

    vector<ComputationNetworkPtr> nets;
    auto encoderNet = ComputationNetwork::CreateFromFile<ElemType>(deviceId, encoderModelPath, FileOptions::fileOptionsBinary, true);

    auto decoderNet = ComputationNetwork::CreateFromFile<ElemType>(deviceId, decoderModelPath, FileOptions::fileOptionsBinary, false, encoderNet.get());

    nets.push_back(encoderNet);
    nets.push_back(decoderNet);
    ConfigArray evalNodeNames = config(L"evalNodeNames");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    ConfigArray outputNodeNames = config(L"outputNodeNames");
    vector<wstring> outputNodeNamesVector;
    for (int i = 0; i < outputNodeNames.size(); ++i)
    {
        outputNodeNamesVector.push_back(outputNodeNames[i]);
    }

    ElemType beamWidth = config(L"beamWidth", "1");

    ConfigParameters writerConfig = config(L"writer");
    DataWriter<ElemType> testDataWriter(writerConfig);

    MultiNetworksEvaluator<ElemType> eval(decoderNet, numMBsToShowResult, traceLevel);
    eval.InitTrainEncoderDecoderWithHiddenStates(config);

    eval.EncodingEvaluateDecodingBeamSearch(nets, readers, 
                                            testDataWriter, evalNodeNamesVector,
                                            outputNodeNamesVector,
                                            mbSize[0], beamWidth, epochSize);
}

/**
This is beam search decoder.

Developed by Kaisheng Yao.

It is used in the following work:
K. Yao, G. Zweig, "Sequence-to-sequence neural net models for grapheme-to-phoneme conversion" in Interspeech 2015
*/
template <typename ElemType>
void DoBeamSearchDecoding(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig = config(L"reader");
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DoEvalBeamSearch(config, testDataReader);
}

template <typename ElemType>
void DoEvalBeamSearch(const ConfigParameters& config, IDataReader<ElemType>& reader)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;

    int traceLevel = config(L"traceLevel", "0");
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");

    auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelPath);

    ConfigArray evalNodeNames = config(L"evalNodeNames");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    ConfigArray outputNodeNames = config(L"outputNodeNames");
    vector<wstring> outputNodeNamesVector;
    for (int i = 0; i < outputNodeNames.size(); ++i)
    {
        outputNodeNamesVector.push_back(outputNodeNames[i]);
    }

    ElemType beamWidth = config(L"beamWidth", "1");

    ConfigParameters writerConfig = config(L"writer");
    DataWriter<ElemType> testDataWriter(writerConfig);

    MultiNetworksEvaluator<ElemType> eval(net, numMBsToShowResult, traceLevel);
    eval.BeamSearch(&reader, testDataWriter, evalNodeNamesVector, outputNodeNamesVector, mbSize[0], beamWidth, epochSize);
}

#if 0
// TODO: per discussion with Dong Yu, Guoguo Chen, and Yu Zhang, this function can be removed.
template <typename ElemType>
void DoSequenceTrain(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    ConfigParameters configSGD(config(L"SGD"));
    bool makeMode = config(L"makeMode", "true");

    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    IComputationNetBuilder<ElemType>* netBuilder = NULL;
    if (config.Exists("NDLNetworkBuilder"))
    {
        ConfigParameters configNDL(config(L"NDLNetworkBuilder"));
        netBuilder = (IComputationNetBuilder<ElemType>*)new NDLBuilder<ElemType>(configNDL);
    }
    else if (config.Exists("SimpleNetworkBuilder"))
    {
        ConfigParameters configSNB(config(L"SimpleNetworkBuilder"));
        netBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        RuntimeError("No network builder found in the config file. NDLNetworkBuilder or SimpleNetworkBuilde must be specified");
    }

    DataReader<ElemType>* dataReader = new DataReader<ElemType>(readerConfig);

    DataReader<ElemType>* cvDataReader = nullptr;
    ConfigParameters cvReaderConfig(config(L"cvReader", L""));

    if (cvReaderConfig.size() != 0)
    {
        cvReaderConfig.Insert("traceLevel", config(L"traceLevel", "0"));
        cvDataReader = new DataReader<ElemType>(cvReaderConfig);
    }

    wstring origModelFileName = config(L"origModelFileName", L"");

    SGD<ElemType> sgd(configSGD);

    sgd.SequenceTrain(netBuilder, origModelFileName, dataReader, cvDataReader, deviceId, makeMode);

    delete dataReader;
    delete cvDataReader;
}
#endif

template <typename ElemType>
void DoEdit(const ConfigParameters& config)
{
    wstring editPath = config(L"editPath");
    wstring ndlMacros = config(L"ndlMacros", "");
    NDLScript<ElemType> ndlScript;
    if (!ndlMacros.empty())
    {
        ndlScript.LoadConfigFile(ndlMacros);
    }
    MELScript<ElemType> melScript;
    melScript.LoadConfigFileAndResolveVariables(editPath, config);
}

template <typename ElemType>
void DoConvertFromDbn(const ConfigParameters& config)
{
    wstring modelPath = config(L"modelPath");
    wstring dbnModelPath = config(L"dbnModelPath");

    auto netBuilder = make_shared<SimpleNetworkBuilder<ElemType>>(config);
    ComputationNetworkPtr net = netBuilder->BuildNetworkFromDbnFile(dbnModelPath);
    net->SaveToFile(modelPath);
}

// do topological plot of computation network 
template <typename ElemType>
void DoTopologyPlot(const ConfigParameters& config)
{
    wstring modelPath = config(L"modelPath");
    wstring outdot = config(L"outputDotFile");           // filename for the dot language output, if not specified, %modelpath%.dot will be used
    wstring outRending = config(L"outputFile");      // filename for the rendered topology plot
    // this can be empty, in that case no rendering will be done
    // or if this is set, renderCmd must be set, so CNTK will call re       
    wstring RenderCmd = config(L"RenderCmd");               // if this option is set, then CNTK will call the render to convert the outdotFile to a graph
    // e.g. "d:\Tools\graphviz\bin\dot.exe -Tpng -x <IN> -o<OUT>"
    //              where <IN> and <OUT> are two special placeholders

    //========================================
    // Sec. 1 option check
    //========================================
    if (outdot.empty())
    {
        outdot = modelPath + L".dot";
    }

    wstring rescmd;
    if (!outRending.empty())        // we need to render the plot
    {
        std::wregex inputPlaceHolder(L"(.+)(<IN>)(.*)");
        std::wregex outputPlaceHolder(L"(.+)(<OUT>)(.*)");

        rescmd = regex_replace(RenderCmd, inputPlaceHolder, L"$1" + outdot + L"$3");
        rescmd = regex_replace(rescmd, outputPlaceHolder, L"$1" + outRending + L"$3");
    }

    ComputationNetwork net(-1);
    net.LoadFromFile<ElemType>(modelPath);
    net.PlotNetworkTopology(outdot);
    fprintf(stderr, "Output network description in dot language to %S\n", outdot.c_str());

    if (!outRending.empty())
    {
        fprintf(stderr, "Executing a third-part tool for rendering dot:\n%S\n", rescmd.c_str());
#ifdef __unix__
        const auto rc = system(msra::strfun::utf8(rescmd).c_str()); rc/*ignoring the result--this gets flagged by gcc if we don't save the return value*/;
#else
        _wsystem(rescmd.c_str());
#endif
        fprintf(stderr, "Done\n");
    }
}

size_t GetMaxEpochs(const ConfigParameters& configParams)
{
    ConfigParameters configSGD(configParams("SGD"));
    size_t maxEpochs = configSGD("maxEpochs");

    return maxEpochs;
}

// special temporary function to guard against a now invalid usage of "truncated" which exists in some IPG production setups
static void DisableLegacyTruncationSettings(const ConfigParameters& TopLevelConfig, const ConfigParameters& commandConfig)
{
    if (TopLevelConfig.ExistsCurrent(L"Truncated"))
    {
        return;
    }

    // if any of the action has set a reader/SGD section and has different Truncated value for reader and SGD section 
    ConfigArray actions = commandConfig(L"action");
    for (size_t i = 0; i < actions.size(); i++)
    {
        if (actions[i] == "train" || actions[i] == "trainRNN")
        {
            ConfigParameters sgd = ConfigParameters(commandConfig(L"SGD"));
            ConfigParameters reader = ConfigParameters(commandConfig(L"reader"));
            // reader and SGD sections are two must-have sections in train/trainRNN 
            if (reader.ExistsCurrent(L"Truncated") && !sgd.ExistsCurrent(L"Truncated"))
            {
                InvalidArgument("DisableLegacyUsage: setting Truncated only in reader section are not allowed. Please move Truncated=true/false to the top level section.");
            }
        }
    }
}
static void DisableLegacyUsage(const ConfigParameters& TopLevelConfig, const ConfigArray& commands)
{
    for (size_t i = 0; i < commands.size(); i++)
    {
        ConfigParameters cfgParameters(TopLevelConfig(commands[i]));
        DisableLegacyTruncationSettings(TopLevelConfig, cfgParameters);
    }
}

// process the command
template <typename ElemType>
void DoCommands(const ConfigParameters& config)
{
    ConfigArray command = config(L"command", "train");

    int numCPUThreads = config(L"numCPUThreads", "0");
    numCPUThreads = CPUMatrix<ElemType>::SetNumThreads(numCPUThreads);

    if (numCPUThreads>0)
    {
        std::cerr << "Using " << numCPUThreads << " CPU threads" << endl;
    }

    bool progressTracing = config(L"progressTracing", false);

    // temporary hack to prevent users from failling for a small breaking change related to the "truncated" flag (will be redone bigger and better some day)
    DisableLegacyUsage(config, command);

    // summarize command info upfront in the log and stdout
    size_t fullTotalMaxEpochs = 0;
    for (int i = 0; i < command.size(); i++)
    {
        //get the configuration parameters that match the command
        ConfigParameters commandParams(config(command[i]));
        ConfigArray action = commandParams("action", "train");

        // determine the action to perform, and do it
        for (int j = 0; j < action.size(); j++)
        {
            if (action[j] == "train"            || action[j] == "trainRNN"
#if 0
                || action[j] == "trainSequence" || action[j] == "trainSequenceRNN"
#endif
                )
            {
                wstring modelPath = commandParams("modelPath");
                std::wcerr << "CNTKModelPath: " << modelPath << endl;
                size_t maxEpochs = GetMaxEpochs(commandParams);
                std::cerr << "CNTKCommandTrainInfo: " + command[i] << " : " << maxEpochs << endl;
                fullTotalMaxEpochs += maxEpochs;
            }
        }
    }
    std::cerr << "CNTKCommandTrainInfo: CNTKNoMoreCommands_Total : " << fullTotalMaxEpochs << endl;

    // set up progress tracing for compute cluster management
    if (progressTracing && ((g_mpi == nullptr) || g_mpi->IsMainNode()))
        ProgressTracing::TraceTotalNumberOfSteps(fullTotalMaxEpochs);   // enable tracing, using this as the total number of epochs

    size_t fullEpochsOffset = 0;

    // execute the commands
    for (int i = 0; i < command.size(); i++)
    {
        //get the configuration parameters that match the command
        ConfigParameters commandParams(config(command[i]));
        ConfigArray action = commandParams("action", "train");

        if (progressTracing && ((g_mpi == nullptr) || g_mpi->IsMainNode()))
            ProgressTracing::SetStepOffset(fullEpochsOffset);   // this is the epoch number that SGD will log relative to

        // determine the action to perform, and do it
        for (int j = 0; j < action.size(); j++)
        {
            if (action[j] == "train" || action[j] == "trainRNN")
            {
                std::cerr << "CNTKCommandTrainBegin: " + command[i] << endl;
                DoTrain<ConfigParameters, ElemType>(commandParams);
                std::cerr << "CNTKCommandTrainEnd: " + command[i] << endl;
                fullEpochsOffset += GetMaxEpochs(commandParams);
            }
#if 0
            else if (action[j] == "trainSequence" || action[j] == "trainSequenceRNN")
            {
                std::cerr << "CNTKCommandTrainBegin: " + command[i] << endl;
                DoSequenceTrain<ElemType>(commandParams);
                std::cerr << "CNTKCommandTrainEnd: " + command[i] << endl;
                fullEpochsOffset += GetMaxEpochs(commandParams);
            }
#endif
            else if (action[j] == "adapt")
            {
                DoAdapt<ElemType>(commandParams);
            }
            else if (action[j] == "test" || action[j] == "eval")
            {
                DoEval<ElemType>(commandParams);
            }
            else if (action[j] == "testunroll")
            {
                DoEvalUnroll<ElemType>(commandParams);
            }
            else if (action[j] == "edit")
            {
                DoEdit<ElemType>(commandParams);
            }
            else if (action[j] == "cv")
            {
                DoCrossValidate<ElemType>(commandParams);
            }
            else if (action[j] == "write")
            {
                DoWriteOutput<ElemType>(commandParams);
            }
            else if (action[j] == "devtest")
            {
                TestCn<ElemType>(config); // for "devtest" action pass the root config instead
            }
            else if (action[j] == "dumpnode")
            {
                DumpNodeInfo<ElemType>(commandParams);
            }
            else if (action[j] == "convertdbn")
            {
                DoConvertFromDbn<ElemType>(commandParams);
            }
            else if (action[j] == "createLabelMap")
            {
                DoCreateLabelMap<ElemType>(commandParams);
            }
            else if (action[j] == "writeWordAndClass")
            {
                DoWriteWordAndClassInfo<ElemType>(commandParams);
            }
            else if (action[j] == "plot")
            {
                DoTopologyPlot<ElemType>(commandParams);
            }
            else if (action[j] == "SVD")
            {
                DoParameterSVD<ElemType>(commandParams);
            }
            else if (action[j] == "trainEncoderDecoder")
            {
                DoEncoderDecoder<ElemType>(commandParams);
            }
            else if (action[j] == "testEncoderDecoder")
            {
                DoEvalEncodingBeamSearchDecoding<ElemType>(commandParams);
            }
            else if (action[j] == "trainBidirectionEncoderDecoder")
            {
                DoBidirectionEncoderDecoder<ElemType>(commandParams);
            }
            else if (action[j] == "beamSearch")
            {
                DoBeamSearchDecoding<ElemType>(commandParams);
            }
            else
            {
                RuntimeError("unknown action: %s  in command set: %s", action[j].c_str(), command[i].c_str());
            }

            NDLScript<ElemType> ndlScript;
            ndlScript.ClearGlobal(); // clear global macros between commands
        }
    }
}

std::string TimeDateStamp()
{
#if 0   // "safe" version for Windows, not needed it seems
    __time64_t localtime;

    _time64(&localtime);// get current time and date
    struct tm now;
    _localtime64_s(&now, &localtime);  // convert
#else
    time_t t = time(NULL);
    struct tm now = *localtime(&t);
#endif
    char buf[30];
    sprintf(buf, "%04d/%02d/%02d %02d:%02d:%02d", now.tm_year + 1900, now.tm_mon + 1, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec);
    return buf;
}

void PrintBuiltInfo()
{
    fprintf(stderr, "-------------------------------------------------------------------\n");
    fprintf(stderr, "Build info: \n\n");
    fprintf(stderr, "\t\tBuilt time: %s %s\n", __DATE__, __TIME__);
    fprintf(stderr, "\t\tLast modified date: %s\n", __TIMESTAMP__);
#ifdef _BUILDTYPE_ 
    fprintf(stderr, "\t\tBuild type: %s\n", _BUILDTYPE_);
#endif 
#ifdef _MATHLIB_
    fprintf(stderr, "\t\tMath lib: %s\n", _MATHLIB_);
#endif
#ifdef _CUDA_PATH_
    fprintf(stderr, "\t\tCUDA_PATH: %s\n", _CUDA_PATH_);
#endif 
#ifdef _CUB_PATH_
    fprintf(stderr, "\t\tCUB_PATH: %s\n", _CUB_PATH_);
#endif 
#ifdef _GIT_EXIST
    fprintf(stderr, "\t\tBuild Branch: %s\n", _BUILDBRANCH_);
    fprintf(stderr, "\t\tBuild SHA1: %s\n", _BUILDSHA1_);
#endif
#ifdef _BUILDER_ 
    fprintf(stderr, "\t\tBuilt by %s on %s\n", _BUILDER_, _BUILDMACHINE_);
#endif 
#ifdef _BUILDPATH_
    fprintf(stderr, "\t\tBuild Path: %s\n", _BUILDPATH_);
#endif
    fprintf(stderr, "-------------------------------------------------------------------\n");
}

void PrintUsageInfo()
{
    fprintf(stderr, "-------------------------------------------------------------------\n");
    fprintf(stderr, "Usage: cntk configFile=yourConfigFile\n");
    fprintf(stderr, "For detailed information please consult the CNTK book\n");
    fprintf(stderr, "\"An Introduction to Computational Networks and the Computational Network Toolkit\"\n");
    fprintf(stderr, "-------------------------------------------------------------------\n");
}

// ---------------------------------------------------------------------------
// main() for use with BrainScript
// ---------------------------------------------------------------------------

wstring ConsumeArg(vector<wstring> & args)
{
    if (args.empty())
        InvalidArgument("Unexpected end of command line.");
    wstring arg = args.front();
    args.erase(args.begin());
    return arg;
}
template<class WHAT>
static void Append(vector<wstring> & toWhat, const WHAT & what) { toWhat.insert(toWhat.end(), what.begin(), what.end()); }
static wstring PathToBSStringLiteral(const wstring & path)  // quote a pathname for BS
{
    let hasSingleQuote = path.find(path, L'\'') != wstring::npos;
    let hasDoubleQuote = path.find(path, L'"')  != wstring::npos;
    if (hasSingleQuote && hasDoubleQuote)
        InvalidArgument("Pathname cannot contain both single (') and double (\") quote at the same time: %ls", path.c_str());
    else if (hasSingleQuote)
        return L"\"" + path + L"\"";
    else
        return L'"' + path + L'"';
}

int wmainWithBS(int argc, wchar_t* argv[])   // called from wmain which is a wrapper that catches & reports Win32 exceptions
{
    vector<wstring> args(argv, argv+argc);
    let exePath = ConsumeArg(args);

    // startup message
    // In case of a redirect of stderr, this will be printed twice, once upfront, and once again after the redirect so that it goes into the log file
    wstring startupMessage = msra::strfun::wstrprintf(L"running on %ls at %ls\n", msra::strfun::utf16(GetHostName()).c_str(), msra::strfun::utf16(TimeDateStamp()).c_str());
    startupMessage += msra::strfun::wstrprintf(L"command line: %ls", exePath.c_str());
    for (const auto & arg : args)
        startupMessage += L" " + arg;

    fprintf(stderr, "%ls\n", startupMessage.c_str());

    // parse command-line options
    vector<wstring> sourceFiles;
    vector<wstring> includePaths;
    vector<wstring> overrides;
    wstring workingDir;
    while (!args.empty())
    {
        let option = ConsumeArg(args);
        if (option == L"-f" || option == L"--file")                                 // -f defines source files
            Append(sourceFiles, msra::strfun::split(ConsumeArg(args), L";"));
        else if (option == L"-I")                                                   // -I declares an include search path
            Append(includePaths, msra::strfun::split(ConsumeArg(args), L";"));
        else if (option == L"-D")                                                   // -D defines variables inline on the command line (which may override BS)
            overrides.push_back(ConsumeArg(args));
        else if (option == L"--cd")                                                 // --cd sets the working directory
            workingDir = ConsumeArg(args);
        else
            InvalidArgument("Invalid command-line option '%ls'.", option.c_str());
    }

    // change working directory
    if (workingDir != L"")
        _wchdir(workingDir.c_str());

    // compile the BrainScript
    wstring bs = L"[\n";
    bs += standardFunctions + computationNodes + commonMacros + L"\n";   // start with standard macros
    for (const auto & sourceFile : sourceFiles)
        bs += L"include " + PathToBSStringLiteral(sourceFile) + L"\n";
    bs += L"\n]\n";
    for (const auto & over : overrides)
        bs += L"with [ " + over + L" ]\n";

    fprintf(stderr, "\n\nBrainScript -->\n\n%ls\n\n", bs.c_str());

    let expr = BS::ParseConfigExpression(bs, move(includePaths));   // parse
    let valp = BS::Evaluate(expr);                                  // evaluate parse into a dictionary
    let & config = valp.AsRef<ScriptableObjects::IConfigRecord>();  // this is the dictionary

    // legacy parameters that have changed spelling
    if (config.Find(L"DoneFile"))       // variables follow camel case (start with lower-case letters)
        InvalidArgument("Legacy spelling of 'DoneFile' no longer allowed. Use 'doneFile'.");
    if (config.Find(L"command"))        // spelling error, should be plural. Using 'actions' instead to match the data type.
        InvalidArgument("Legacy spelling of 'command' no longer allowed. Use 'actions'.");
    if (config.Find(L"type"))
        InvalidArgument("Legacy name 'type' no longer allowed. Use 'precision'.");

    // parallel training
    g_mpi = nullptr;
    bool paralleltrain = config(L"parallelTrain", false);
    if (paralleltrain)
        g_mpi = new MPIWrapper();

    // logging
    wstring logpath = config(L"stderr", L"");
    if (logpath != L"")
    {
        logpath += L"_actions"; // TODO: for old CNTK, this was a concatenation of all action names, which we no longer know
        logpath += L".log";     // TODO: why do we need to append this here?

        if (paralleltrain)
            logpath += msra::strfun::wstrprintf(L"rank%d", (int)g_mpi->CurrentNodeRank());

        RedirectStdErr(logpath);
        fprintf(stderr, "%ls\n", startupMessage.c_str());
    }

    // echo config info to log
    PrintBuiltInfo();

    // execute the actions
    //std::string type = config(L"precision", "float");
    int numCPUThreads = config(L"numCPUThreads", 0);
    numCPUThreads = CPUMatrix<float/*any will do*/>::SetNumThreads(numCPUThreads);
    if (numCPUThreads > 0)
        fprintf(stderr, "Using %d CPU threads.\n", numCPUThreads);

    bool progressTracing = config(L"progressTracing", false);
    size_t fullTotalMaxEpochs = 1;              // BUGBUG: BS does not allow me to read out the max epochs parameters, as that would instantiate and thus execute the objects
    // set up progress tracing for compute cluster management
    if (progressTracing && ((g_mpi == nullptr) || g_mpi->IsMainNode()))
        ProgressTracing::TraceTotalNumberOfSteps(fullTotalMaxEpochs);   // enable tracing, using this as the total number of epochs

    // MAIN LOOP that executes the actions
    auto actionsVal = config[L"actions"];
    // Note: weird behavior. If 'actions' is a scalar value (rather than an array) then it will have been resolved already after the above call. That means, it has already completed its action!
    //       Not pretty, but a direct consequence of the lazy evaluation. The only good solution would be to have a syntax for arrays including length 0 and 1.
    //       Since this in the end behaves indistinguishable from the array loop below, we will keep it for now.
    if (actionsVal.Is<ScriptableObjects::ConfigArray>())
    {
        const ScriptableObjects::ConfigArray & actions = actionsVal;
        for (int i = actions.GetIndexRange().first; i <= actions.GetIndexRange().second; i++)
        {
            actions.At(i, [](const wstring &){});  // this will evaluate and thus execute the action
        }
    }
    // else action has already been executed, see comment above

    // write a doneFile if requested
    wstring doneFile = config(L"doneFile", L"");
    if (doneFile != L"")
    {
        FILE* fp = fopenOrDie(doneFile.c_str(), L"w");
        fprintf(fp, "successfully finished at %s on %s\n", TimeDateStamp().c_str(), GetHostName().c_str());
        fcloseOrDie(fp);
    }
    fprintf(stderr, "COMPLETED\n"), fflush(stderr);

    delete g_mpi;
    return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// main() for old CNTK config language
// ---------------------------------------------------------------------------

int wmainOldCNTKConfig(int argc, wchar_t* argv[])   // called from wmain which is a wrapper that catches & repots Win32 exceptions
{
    ConfigParameters config;
    std::string rawConfigString = ConfigParameters::ParseCommandLine(argc, argv, config);

    // get the command param set they want
    wstring logpath = config(L"stderr", L"");

    //  [1/26/2015 erw, add done file so that it can be used on HPC]
    wstring DoneFile = config(L"DoneFile", L"");
    ConfigArray command = config(L"command", "train");

    // paralleltrain training
    g_mpi = nullptr;
    bool paralleltrain = config(L"parallelTrain", "false");
    if (paralleltrain)
    {
        g_mpi = new MPIWrapper();
    }

    if (logpath != L"")
    {
        for (int i = 0; i < command.size(); i++)
        {
            logpath += L"_";
            logpath += (wstring)command[i];
        }
        logpath += L".log";

        if (paralleltrain)
        {
            std::wostringstream oss;
            oss << g_mpi->CurrentNodeRank();
            logpath += L"rank" + oss.str();
        }
        RedirectStdErr(logpath);
    }

    PrintBuiltInfo();
    std::string timestamp = TimeDateStamp();

    //dump config info
    fprintf(stderr, "running on %s at %s\n", GetHostName().c_str(), timestamp.c_str());
    fprintf(stderr, "command line: \n");
    for (int i = 0; i < argc; i++)
    {
        fprintf(stderr, "%s ", WCharToString(argv[i]).c_str());
    }

    // This simply merges all the different config parameters specified (eg, via config files or via command line directly),
    // and prints it.
    fprintf(stderr, "\n\n>>>>>>>>>>>>>>>>>>>> RAW CONFIG (VARIABLES NOT RESOLVED) >>>>>>>>>>>>>>>>>>>>\n");
    fprintf(stderr, "%s\n", rawConfigString.c_str());
    fprintf(stderr, "<<<<<<<<<<<<<<<<<<<< RAW CONFIG (VARIABLES NOT RESOLVED)  <<<<<<<<<<<<<<<<<<<<\n");

    // Same as above, but all variables are resolved.  If a parameter is set multiple times (eg, set in config, overriden at command line),
    // All of these assignments will appear, even though only the last assignment matters.
    fprintf(stderr, "\n>>>>>>>>>>>>>>>>>>>> RAW CONFIG WITH ALL VARIABLES RESOLVED >>>>>>>>>>>>>>>>>>>>\n");
    fprintf(stderr, "%s\n", config.ResolveVariables(rawConfigString).c_str());
    fprintf(stderr, "<<<<<<<<<<<<<<<<<<<< RAW CONFIG WITH ALL VARIABLES RESOLVED <<<<<<<<<<<<<<<<<<<<\n");

    // This outputs the final value each variable/parameter is assigned to in config (so if a parameter is set multiple times, only the last
    // value it is set to will appear).
    fprintf(stderr, "\n>>>>>>>>>>>>>>>>>>>> PROCESSED CONFIG WITH ALL VARIABLES RESOLVED >>>>>>>>>>>>>>>>>>>>\n");
    config.dumpWithResolvedVariables();
    fprintf(stderr, "<<<<<<<<<<<<<<<<<<<< PROCESSED CONFIG WITH ALL VARIABLES RESOLVED <<<<<<<<<<<<<<<<<<<<\n");

    fprintf(stderr, "command: ");
    for (int i = 0; i < command.size(); i++)
    {
        fprintf(stderr, "%s ", command[i].c_str());
    }

    //run commands
    std::string type = config(L"precision", "float");
    // accept old precision key for backward compatibility
    if (config.Exists("type"))
    {
        type = config(L"type", "float");
    }

    fprintf(stderr, "\nprecision = %s\n", type.c_str());
    if (type == "float")
    {
        DoCommands<float>(config);
    }
    else if (type == "double")
    {
        DoCommands<double>(config);
    }
    else
    {
        RuntimeError("invalid precision specified: %s", type.c_str());
    }

    // still here , write a DoneFile if necessary 
    if (!DoneFile.empty())
    {
        FILE* fp = fopenOrDie(DoneFile.c_str(), L"w");
        fprintf(fp, "successfully finished at %s on %s\n", TimeDateStamp().c_str(), GetHostName().c_str());
        fcloseOrDie(fp);
    }
    fprintf(stderr, "COMPLETED\n"), fflush(stderr);

    delete g_mpi;
    return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// main wrapper that catches C++ exceptions and prints them
// ---------------------------------------------------------------------------

int wmain1(int argc, wchar_t* argv[])   // called from wmain which is a wrapper that catches & repots Win32 exceptions
{
    try
    {
        if (argc <= 1)
            InvalidArgument("No command-line argument given.");
        // detect legacy CNTK configuration
        bool isOldCNTKConfig = false;
        for (int i = 0; i < argc && !isOldCNTKConfig; i++)
            isOldCNTKConfig |= !_wcsnicmp(L"configFile=", argv[i], 11);
        if (isOldCNTKConfig)
            return wmainOldCNTKConfig(argc, argv);
        // run from BrainScript
        return wmainWithBS(argc, argv);
    }
    catch (const ScriptableObjects::ScriptingException &err)
    {
        fprintf(stderr, "EXCEPTION occurred: %s\n", err.what());
        err.PrintError();
        return EXIT_FAILURE;
    }
    catch (const std::exception &err)
    {
        fprintf(stderr, "EXCEPTION occurred: %s\n", err.what());
        PrintUsageInfo();
        return EXIT_FAILURE;
    }
    catch (...)
    {
        fprintf(stderr, "Unknown ERROR occurred");
        PrintUsageInfo();
        return EXIT_FAILURE;
    }
}

#ifdef __WINDOWS__
void terminate_this() { fprintf(stderr, "terminate_this: aborting\n"), fflush(stderr); exit(EXIT_FAILURE); }

int wmain(int argc, wchar_t* argv[])    // wmain wrapper that reports Win32 exceptions
{
    set_terminate (terminate_this); // insert a termination handler to ensure stderr gets flushed before actually terminating
    // Note: this does not seem to work--processes with this seem to just hang instead of terminating
    __try
    {
        return wmain1 (argc, argv);
    }
    __except (1/*EXCEPTION_EXECUTE_HANDLER, see excpt.h--not using constant to avoid Windows header in here*/)
    {
        fprintf (stderr, "CNTK: Win32 exception caught (such an access violation or a stack overflow)\n");  // TODO: separate out these two into a separate message
        fflush (stderr);
        exit (EXIT_FAILURE);
    }
}
#endif

#ifdef __UNIX__
/// UNIX main function converts arguments in UTF-8 encoding and passes to Visual-Studio style wmain() which takes wchar_t strings.
int main(int argc, char* argv[])
{
    // TODO: change to STL containers
    wchar_t **wargs = new wchar_t*[argc];
    for (int i = 0; i < argc; ++i)
    {
        wargs[i] = new wchar_t[strlen(argv[i]) + 1];
        size_t ans = ::mbstowcs(wargs[i], argv[i], strlen(argv[i]) + 1);
        assert(ans == strlen(argv[i]));
    }
    int ret = wmain1(argc, wargs);
    for (int i = 0; i < argc; ++i)
        delete[] wargs[i];
    delete[] wargs;
    return ret;
}
#endif
