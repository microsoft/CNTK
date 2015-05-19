//
// <copyright file="cn.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// cn.cpp : Defines the entry point for the console application.
//

#define _CRT_NONSTDC_NO_DEPRECATE   // make VS accept POSIX functions without _

#include "stdafx.h"
#include "ComputationNetwork.h"
#include "ComputationNode.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "SimpleNetworkBuilder.h"
#include "NDLNetworkBuilder.h"
#include "SynchronousExecutionEngine.h"
#include "ModelEditLanguage.h"
#include "SGD.h"
#include <string>
#include "Basics.h"
#include "commandArgUtil.h"
#include "SimpleEvaluator.h"
#include "SimpleOutputWriter.h"
#include <chrono>
#include <algorithm>
#if defined(_WIN32)
#include "io.h"
#include "buildinfo.h"
#endif
#include "hostname.h"
#ifdef LEAKDETECT
#include "vld.h" // for memory leak detection
#endif
#include <vector>
#include <iostream>
#include <queue>
#include <set>
#include "BestGpu.h"


// MPI builds on windows require the following installed to "c:\program files\Microsoft MPI\"
// HPC Pack 2012 R2 MS-MPI Redistributable Package
// http://www.microsoft.com/en-us/download/details.aspx?id=41634

#ifdef MPI_SUPPORT
#include "mpi.h"
#pragma comment(lib, "msmpi.lib")
#endif
int numProcs;
int myRank;

using namespace std;
using namespace Microsoft::MSR::CNTK;

// internal test routine forward declaration
template <typename ElemType>
void TestCn(const ConfigParameters& config);

template <typename T>
struct compare_second
{
    bool operator()(const T &lhs, const T &rhs) const { return lhs.second < rhs.second; }
};

void RedirectStdErr(wstring logpath)
{
    fprintf(stderr, "Redirecting stderr to file %S\n", logpath.c_str());
    msra::files::make_intermediate_dirs(logpath);
    auto_file_ptr f(logpath.c_str(), "wb");
    if (dup2(fileno(f), 2) == -1)
        RuntimeError("unexpected failure to redirect stderr to log file");
    setvbuf(stderr, NULL, _IONBF, 16384);   // unbuffer it
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
    wstring modelPath = config("modelPath");
    wstring nodeName = config("nodeName", L"__AllNodes__");
    wstring defOutFilePath = modelPath + L"." + nodeName + L".txt";
    wstring outputFile = config("outputFile", WCharToString(defOutFilePath.c_str()).c_str());
    bool printValues = config("printValues", "true");

    ComputationNetwork<ElemType> net(-1);  //always use CPU
    net.LoadFromFile(modelPath);
    net.DumpNodeInfoToFile(nodeName, printValues, outputFile);
}

template <typename ElemType>
void DoEvalBase(const ConfigParameters& config, IDataReader<ElemType>& reader)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config("minibatchSize", "40960");
    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config("modelPath");
    intargvector mbSize = minibatchSize;

    int traceLevel = config("traceLevel", "0");
    size_t numMBsToShowResult = config("numMBsToShowResult", "100");

    ConfigArray evalNodeNames = config("evalNodeNames", "");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    ComputationNetwork<ElemType> net(deviceId);
    net.LoadFromFile(modelPath);
    net.ResetEvalTimeStamp();

    SimpleEvaluator<ElemType> eval(net, numMBsToShowResult, traceLevel);
    eval.Evaluate(reader, evalNodeNamesVector, mbSize[0], epochSize);
}

template <typename ElemType>
void DoEval(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig(config("reader"));
    readerConfig.Insert("traceLevel", config("traceLevel", "0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DoEvalBase(config, testDataReader);
}

template <typename ElemType>
void DoEvalUnroll(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig(config("reader"));
    readerConfig.Insert("traceLevel", config("traceLevel", "0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config("minibatchSize", "40960");
    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config("modelPath");
    intargvector mbSize = minibatchSize;
    wstring path2EvalResults = config("path2EvalResults", L"");

    ComputationNetwork<ElemType> net(deviceId);
    net.LoadFromFile(modelPath);
    net.ResetEvalTimeStamp();

    SimpleEvaluator<ElemType> eval(net);
    ElemType evalEntropy;
    eval.EvaluateUnroll(testDataReader, mbSize[0], evalEntropy, path2EvalResults == L"" ? nullptr : path2EvalResults.c_str(), epochSize);
}

template <typename ElemType>
void DoCrossValidate(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig(config("reader"));
    readerConfig.Insert("traceLevel", config("traceLevel", "0"));

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config("minibatchSize", "40960");
    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config("modelPath");
    intargvector mbSize = minibatchSize;

    ConfigArray cvIntervalConfig = config("crossValidationInterval");
    intargvector cvInterval = cvIntervalConfig;

    size_t sleepSecondsBetweenRuns = config("sleepTimeBetweenRuns", "0");

    int traceLevel = config("traceLevel", "0");
    size_t numMBsToShowResult = config("numMBsToShowResult", "100");

    ConfigArray evalNodeNames = config("evalNodeNames", "");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    std::vector<std::vector<ElemType>> cvErrorResults;
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
        ComputationNetwork<ElemType> net(deviceId);
        net.LoadFromFile(cvModelPath);
        net.ResetEvalTimeStamp();

        SimpleEvaluator<ElemType> eval(net, numMBsToShowResult, traceLevel);

        fprintf(stderr, "model %ls --> \n", cvModelPath.c_str());
        std::vector<ElemType> evalErrors;
        evalErrors = eval.Evaluate(cvDataReader, evalNodeNamesVector, mbSize[0], epochSize);
        cvErrorResults.push_back(evalErrors);

        ::Sleep(1000 * sleepSecondsBetweenRuns);
    }

    //find best model
    if (cvErrorResults.size() == 0)
        throw std::logic_error("No model is evaluated.");

    std::vector<ElemType> minErrors;
    std::vector<int> minErrIds;
    std::vector<ElemType> evalErrors = cvErrorResults[0];
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
    ConfigParameters readerConfig(config("reader"));
    readerConfig.Insert("traceLevel", config("traceLevel", "0"));
    readerConfig.Insert("randomize", "None");  //we don't want randomization when output results

    DataReader<ElemType> testDataReader(readerConfig);

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config("minibatchSize", "2048");
    wstring modelPath = config("modelPath");
    intargvector mbSize = minibatchSize;

    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    ConfigArray outputNodeNames = config("outputNodeNames", "");
    vector<wstring> outputNodeNamesVector;
    for (int i = 0; i < outputNodeNames.size(); ++i)
    {
        outputNodeNamesVector.push_back(outputNodeNames[i]);
    }

    ComputationNetwork<ElemType> net(deviceId);
    net.LoadFromFile(modelPath);
    net.ResetEvalTimeStamp();

    SimpleOutputWriter<ElemType> writer(net, 1);

    if (config.Exists("writer"))
    {
        ConfigParameters writerConfig(config("writer"));
        bool bWriterUnittest = writerConfig("unittest", "false");
        DataWriter<ElemType> testDataWriter(writerConfig);
        writer.WriteOutput(testDataReader, mbSize[0], testDataWriter, outputNodeNamesVector, epochSize, bWriterUnittest);
    }
    else if (config.Exists("outputPath"))
    {
        wstring outputPath = config("outputPath"); // crashes if no default given? 
        writer.WriteOutput(testDataReader, mbSize[0], outputPath, outputNodeNamesVector, epochSize);
    }
    //writer.WriteOutput(testDataReader, mbSize[0], testDataWriter, outputNodeNamesVector, epochSize);
}

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            TrainingCriterion ParseTrainingCriterionString(wstring s)
            {
                msra::strfun::tolower_ascii(s);
                if (s == L"crossentropywithsoftmax")
                    return TrainingCriterion::CrossEntropyWithSoftmax;
                else if (s == L"squareerror")
                    return TrainingCriterion::SquareError;
                else if (s == L"noisecontrastiveestimationnode")
                    return TrainingCriterion::NCECrossEntropyWithSoftmax;
                else if (s != L"classcrossentropywithsoftmax")    // (twisted logic to keep compiler happy w.r.t. not returning from LogicError)
                    LogicError("trainingCriterion: Invalid trainingCriterion value. Valid values are (CrossEntropyWithSoftmax | SquareError | ClassCrossEntropyWithSoftmax)");
                return TrainingCriterion::ClassCrossEntropyWithSoftmax;
            }

            EvalCriterion ParseEvalCriterionString(wstring s)
            {
                msra::strfun::tolower_ascii(s);
                if (s == L"errorprediction")
                    return EvalCriterion::ErrorPrediction;
                else if (s == L"crossentropywithsoftmax")
                    return EvalCriterion::CrossEntropyWithSoftmax;
                else if (s == L"classcrossentropywithsoftmax")
                    return EvalCriterion::ClassCrossEntropyWithSoftmax;
                else if (s == L"noisecontrastiveestimationnode")
                    return EvalCriterion::NCECrossEntropyWithSoftmax;
                else if (s != L"squareerror")
                    LogicError("evalCriterion: Invalid trainingCriterion value. Valid values are (ErrorPrediction | CrossEntropyWithSoftmax | SquareError)");
                return EvalCriterion::SquareError;
            }

        }
    }
};

template <typename ElemType>
void DoCreateLabelMap(const ConfigParameters& config)
{
    // this gets the section name we are interested in
    std::string section = config("section");
    // get that section (probably a peer config section, which works thanks to heirarchal symbol resolution)
    ConfigParameters configSection(config(section));
    ConfigParameters readerConfig(configSection("reader"));
    readerConfig.Insert("allowMapCreation", "true");
    DEVICEID_TYPE deviceId = CPUDEVICE;
    size_t minibatchSize = config("minibatchSize", "2048");
    int traceLevel = config("traceLevel", "0");
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
        if (labelConfig.ExistsCurrent("labelMappingFile"))
            labelMappingFile = labelConfig("labelMappingFile");
        else if (readerConfig.ExistsCurrent("labelMappingFile"))
            labelMappingFile = labelConfig("labelMappingFile");
        else
            RuntimeError("CreateLabelMap: No labelMappingFile defined");

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
//		An action "SVD" performs the following process to transform an existing model: 
//			1.	For a Learnable Parameter A whose name matches with the user specified regex, 
//			    A is approximated by two matrice multiplication B*C ; 
//			2.	In order to keep the low-rank structure in training, 
//				the original A node will be replaced by A' whose opertions is Times
//				with its left children being B and right chilren being 
//
//		To use this command,
//			user need to specify: 
//					1)	modelPath			-- path to the existing model 
//					2)  outputmodelPath		-- where to write the transformed model 
//					3)  KeepRatio			-- how many percentage of energy we want to keep
//					4)  ParameterName		-- name (regex) of the parameter node we want to perform a SVD decomposition 
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
        vector<wstring> tokens=msra::strfun::split(line, L"\t ");
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
template<typename ElemType> 
void  DoParameterSVD(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceID = -1;        // use CPU for SVD 
    wstring modelPath = config("modelPath");
    wstring outputmodelPath = config("outputmodelPath");
    map<wstring, float>     svdconfig; 

    float keepratio = config("KeepRatio", "0.4");
    wstring svdnodeRegex = config("NodeNameRegex", L"");
    if (!svdnodeRegex.empty())
    {
        svdconfig[svdnodeRegex] = keepratio;
    }
    else
    {
        // alternatively, user can also use a config to specify KeepRatios for different groups of nodes 
        wstring svdnodeConfigFile = config("SVDConfig", L"");
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

    
    ComputationNetwork<ElemType> net(deviceID); 
    net.LoadFromFile(modelPath);

    net.PerformSVDecomposition(svdconfig);
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
///       0	     42068	</s>	0
///       1	     50770	the	0
///       2	     45020	<unk>	1
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
    string inputFile = config("inputFile"); // training text file without <unk>
    string outputWord2Cls = config("outputWord2Cls");
    string outputVocabFile = config("outputVocabFile");
    string outputCls2Index = config("outputCls2Index");
    size_t  vocabSize = config("vocabSize");
    size_t  nbrCls = config("nbrClass");
    int  cutoff = config("cutoff", "1");

    DEVICEID_TYPE deviceId = CPUDEVICE;
    Matrix<ElemType> wrd2cls(deviceId);
    Matrix<ElemType> cls2idx(deviceId);

    FILE *fp = fopen(inputFile.c_str(), "rt");
    if (fp == nullptr)
        RuntimeError("inputFile cannot be read");

    cls2idx.Resize(nbrCls, 1);
    std::unordered_map<string, double> v_count;

    /// get line
    char ch2[2048];
    string str;
    vector<string> vstr;
    long long prevClsIdx = -1;
    string token;
    while (fgets(ch2, 2048, fp) != nullptr)
    {
        str = ch2;
        str = trim(str);
        int sposition = str.find("</s> ");
        int eposition = str.find(" </s>");
        if (sposition == str.npos || eposition == str.npos)
            str = "</s> " + str + " </s>";
        vstr = msra::strfun::split(str, "\t ");
        for (int i = 1; i < vstr.size(); i++)
            v_count[vstr[i]]++;
    }
    fclose(fp);

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
            if (iter->second <= cutoff)
                wordCountLessCutoff--;
    if (wordCountLessCutoff <= 0)
        RuntimeError("no word remained after cutoff\n");
    
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

    for (std::unordered_map<std::string, double>::iterator iter = removed.begin(); iter != removed.end(); iter++)
        total += iter->second;
    for (std::unordered_map<std::string, double>::iterator iter = removed.begin(); iter != removed.end(); iter++)
        dd += sqrt(iter->second / total);
    double df = 0;
    size_t class_id = 0;
    m_class.resize(p.size());

    while (!p.empty())
    {
        std::string word = p.top().first;
        double freq = p.top().second;
        df += sqrt(freq / total) / dd;
        if (df > 1)
            df = 1;
        if (df > 1.0 * (class_id + 1) / nbrCls && class_id < nbrCls)
            class_id++;

        size_t wid = m_words.size();
        bool inserted = m_index.insert(make_pair(word, wid)).second;
        if (inserted)
            m_words.push_back(word);

        m_count[wid] = freq;
        m_class[wid] = class_id;
        p.pop();
    }
    std::ofstream ofvocab;
    ofvocab.open(outputVocabFile.c_str());
    for (size_t i = 0; i < m_index.size(); i++)
    {
        wrd2cls(i, 0) = (ElemType)m_class[i];
        long long clsIdx = m_class[i];
        if (clsIdx != prevClsIdx)
        {
            cls2idx(clsIdx, 0) = (ElemType)i; /// the left boundary of clsIdx
            prevClsIdx = m_class[i];
        }
        ofvocab << "     " << i << "\t     " << m_count[i] << "\t" << m_words[i] << "\t" << m_class[i] << std::endl;
    }
    ofvocab.close();

    /// write the outputs
    fp = fopen(outputWord2Cls.c_str(), "wt");
    if (fp == nullptr)
        RuntimeError("cannot write to %s", outputWord2Cls.c_str());

    for (size_t r = 0; r < wrd2cls.GetNumRows(); r++)
        fprintf(fp, "%d\n", (int)wrd2cls(r, 0));
    fclose(fp);

    fp = fopen(outputCls2Index.c_str(), "wt");
    if (fp == nullptr)
        RuntimeError("cannot write to %s", outputCls2Index.c_str());
    for (size_t r = 0; r < cls2idx.GetNumRows(); r++)
        fprintf(fp, "%d\n", (int)cls2idx(r, 0));
    fclose(fp);
}

template <typename ElemType>
void DoTrain(const ConfigParameters& config)
{
    ConfigParameters configSGD(config("SGD"));
    bool makeMode = config("makeMode", "true");

    ConfigParameters readerConfig(config("reader"));
    readerConfig.Insert("traceLevel", config("traceLevel", "0"));

    IComputationNetBuilder<ElemType>* netBuilder = NULL;

    if (config.Exists("NDLNetworkBuilder"))
    {
        ConfigParameters configNDL(config("NDLNetworkBuilder"));
        netBuilder = (IComputationNetBuilder<ElemType>*)new NDLBuilder<ElemType>(configNDL);
    }
    else if (config.Exists("SimpleNetworkBuilder"))
    {
        ConfigParameters configSNB(config("SimpleNetworkBuilder"));
        netBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        RuntimeError("No network builder found in the config file. NDLNetworkBuilder or SimpleNetworkBuilde must be specified");
    }

    DataReader<ElemType>* dataReader = new DataReader<ElemType>(readerConfig);

    DataReader<ElemType>* cvDataReader = nullptr;
    ConfigParameters cvReaderConfig(config("cvReader", L""));

    if (cvReaderConfig.size() != 0)
    {
        cvReaderConfig.Insert("traceLevel", config("traceLevel", "0"));
        cvDataReader = new DataReader<ElemType>(cvReaderConfig);
    }

    SGD<ElemType> sgd(configSGD);

    sgd.Train(netBuilder, dataReader, cvDataReader, makeMode);

    delete netBuilder;
    delete dataReader;
    delete cvDataReader;
}

template <typename ElemType>
void DoAdapt(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    ConfigParameters configSGD(config("SGD"));
    bool makeMode = config("makeMode", "true");

    ConfigParameters readerConfig(config("reader"));
    readerConfig.Insert("traceLevel", config("traceLevel", "0"));

    DataReader<ElemType>* dataReader = new DataReader<ElemType>(readerConfig);

    DataReader<ElemType>* cvDataReader = nullptr;
    ConfigParameters cvReaderConfig(config("cvReader", L""));

    if (cvReaderConfig.size() != 0)
    {
        cvReaderConfig.Insert("traceLevel", config("traceLevel", "0"));
        cvDataReader = new DataReader<ElemType>(cvReaderConfig);
    }

    wstring origModelFileName = config("origModelFileName", L"");
    wstring refNodeName = config("refNodeName", L"");

    SGD<ElemType> sgd(configSGD);

    sgd.Adapt(origModelFileName, refNodeName, dataReader, cvDataReader, deviceId, makeMode);

    delete dataReader;
    delete cvDataReader;
}

template <typename ElemType>
void DoEdit(const ConfigParameters& config)
{
    wstring editPath = config("editPath");
    wstring ndlMacros = config("ndlMacros", "");
    NDLScript<ElemType> ndlScript;
    if (!ndlMacros.empty())
        ndlScript.LoadConfigFile(ndlMacros);
    MELScript<ElemType> melScript;
    melScript.LoadConfigFileAndResolveVariables(editPath, config);
}

template <typename ElemType>
void DoConvertFromDbn(const ConfigParameters& config)
{
    //config.Insert("deviceId","-1"); //force using CPU

    wstring modelPath = config("modelPath");
    wstring dbnModelPath = config("dbnModelPath");

    IComputationNetBuilder<ElemType>* netBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(config);
    ComputationNetwork<ElemType>& net = netBuilder->LoadNetworkFromFile(dbnModelPath);
    net.SaveToFile(modelPath);
    delete (netBuilder);
}

// do topological plot of computation network 
template <typename ElemType>
void DoTopologyPlot(const ConfigParameters& config)
{
	wstring modelPath = config("modelPath");
	wstring outdot = config("outputDotFile");           // filename for the dot language output, if not specified, %modelpath%.dot will be used
	wstring outRending = config("outputFile");      // filename for the rendered topology plot
	// this can be empty, in that case no rendering will be done
	// or if this is set, renderCmd must be set, so CNTK will call re       
	wstring RenderCmd = config("RenderCmd");               // if this option is set, then CNTK will call the render to convert the outdotFile to a graph
	// e.g. "d:\Tools\graphviz\bin\dot.exe -Tpng -x <IN> -o<OUT>"
	//              where <IN> and <OUT> are two special placeholders

	//========================================
	// Sec. 1 option check
	//========================================
	if (outdot.empty())
	{
		outdot = modelPath +L".dot";
	}

	wstring rescmd;
	if (!outRending.empty())        // we need to render the plot
	{
		std::wregex inputPlaceHolder(L"(.+)(<IN>)(.*)");
		std::wregex outputPlaceHolder(L"(.+)(<OUT>)(.*)");

		rescmd = regex_replace(RenderCmd, inputPlaceHolder, L"$1"+outdot+L"$3");
		rescmd = regex_replace(rescmd, outputPlaceHolder, L"$1"+outRending+L"$3");
	}


	ComputationNetwork<ElemType> net(-1);
	net.LoadFromFile(modelPath);
	net.PlotNetworkTopology(outdot);
    fprintf(stderr, "Output network description in dot language to %S\n", outdot.c_str());

	if (!outRending.empty())
	{
        fprintf(stderr, "Executing a third-part tool for rendering dot:\n%S\n", rescmd.c_str());
#ifdef __unix__
        system(msra::strfun::utf8(rescmd).c_str());
#else
		_wsystem(rescmd.c_str());
#endif
        fprintf(stderr, "Done\n");
	}
}



// process the command
template <typename ElemType>
void DoCommand(const ConfigParameters& config)
{
    ConfigArray command = config("command", "train");

    int numCPUThreads = config("numCPUThreads", "0");
    numCPUThreads = CPUMatrix<ElemType>::SetNumThreads(numCPUThreads);

    if (numCPUThreads>0)
        std::cerr << "Using " << numCPUThreads << " CPU threads" << endl;

    for (int i = 0; i < command.size(); i++)
    {
        //get the configuration parameters that match the command
        ConfigParameters commandParams(config(command[i]));
        ConfigArray action = commandParams("action", "train");

        // determine the action to perform, and do it
        for (int j = 0; j < action.size(); j++)
        {
            if (action[j] == "train" || action[j] == "trainRNN")
                DoTrain<ElemType>(commandParams);
            else if (action[j] == "adapt")
                DoAdapt<ElemType>(commandParams);
            else if (action[j] == "test" || action[j] == "eval")
                DoEval<ElemType>(commandParams);
            else if (action[j] == "testunroll")
                DoEvalUnroll<ElemType>(commandParams);
            else if (action[j] == "edit")
                DoEdit<ElemType>(commandParams);
            else if (action[j] == "cv")
                DoCrossValidate<ElemType>(commandParams);
            else if (action[j] == "write")
                DoWriteOutput<ElemType>(commandParams);
            else if (action[j] == "devtest")
                TestCn<ElemType>(config); // for "devtest" action pass the root config instead
            else if (action[j] == "dumpnode")
                DumpNodeInfo<ElemType>(commandParams);
            else if (action[j] == "convertdbn")
                DoConvertFromDbn<ElemType>(commandParams);
            else if (action[j] == "createLabelMap")
                DoCreateLabelMap<ElemType>(commandParams);
            else if (action[j] == "writeWordAndClass")
	            DoWriteWordAndClassInfo<ElemType>(commandParams);
            else if (action[j] == "plot")
                DoTopologyPlot<ElemType>(commandParams);
            else if (action[j] == "SVD")
                DoParameterSVD<ElemType>(commandParams);
            else
                RuntimeError("unknown action: %s  in command set: %s", action[j].c_str(), command[i].c_str());

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

#ifdef MPI_SUPPORT
// Oh, my gosh, this is going to be ugly. MPI_INIT needs a char* argv[], so let's interface.
int MPIAPI MPI_Init(_In_opt_ int *argc, _Inout_count_(*argc) wchar_t*** argv)
{
    // this maps from the strings 
    std::map<std::string, wchar_t*> recover_wstring;

    // do the mapping to 8-bit encoding for MPI_Init()
    vector<vector<char>> argv_string_vector;
    transform(*argv, *argv + *argc, std::back_inserter(argv_string_vector),
        [&recover_wstring](wchar_t*pws)->vector<char>
    {
        std::string tmp = msra::strfun::utf8(std::wstring(pws));
        recover_wstring[tmp] = pws;
        vector<char> rv(tmp.begin(), tmp.end());
        rv.push_back('\0');
        return rv;
    }
    );
    vector<char*> argv_charptr_vector;
    transform(argv_string_vector.begin(), argv_string_vector.end(), std::back_inserter(argv_charptr_vector),
        [](std::vector<char>&cs)->char*{ return &(cs[0]); }
    );
    char** argv_char = &(argv_charptr_vector[0]);

    // Do the initialization
    int rv = MPI_Init(argc, &argv_char);

    // try and reconstruct how MPI_Init changed the argv
    transform(argv_char, argv_char + *argc, stdext::checked_array_iterator<wchar_t**>(*argv, *argc),
        [&recover_wstring](char*pc)->wchar_t*
    {
        auto it = recover_wstring.find(std::string(pc));
        if (it == recover_wstring.end())
            RuntimeError("Unexpected interaction between MPI_Init and command line parameters");
        return it->second;
    }
    );

    // pass through return value from internal call to MPI_Init()
    return rv;
}
#endif

#ifdef _WIN32
void PrintBuiltInfo()
{
    fprintf(stderr, "-------------------------------------------------------------------\n");
    fprintf(stderr, "Build info: \n\n");
    fprintf(stderr, "\t\tBuilt time: %s %s\n", __DATE__, __TIME__);
    fprintf(stderr, "\t\tLast modified date: %s\n", __TIMESTAMP__);
    fprintf(stderr, "\t\tBuilt by %s on %s\n", _BUILDER_, _BUILDMACHINE_);
    fprintf(stderr, "\t\tBuild Path: %s\n", _BUILDPATH_);
    fprintf(stderr, "\t\tCUDA_PATH: %s\n", _CUDA_PATH_);
#ifdef _GIT_EXIST
    fprintf(stderr, "\t\tBuild Branch: %s\n", _BUILDBRANCH_);
    fprintf(stderr, "\t\tBuild SHA1: %s\n", _BUILDSHA1_);
#endif
    fprintf(stderr, "-------------------------------------------------------------------\n");

}
#endif

int wmain(int argc, wchar_t* argv[])
{

    try
    {
#ifdef MPI_SUPPORT
        {
            int rc;
            rc = MPI_Init(&argc, &argv);
            if (rc != MPI_SUCCESS)
            {
                MPI_Abort(MPI_COMM_WORLD, rc);
                RuntimeError("Failure in MPI_Init: %d", rc);
            }
            MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
            MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
            fprintf(stderr, "MPI: RUNNING ON (%s), process %d/%d\n", getenv("COMPUTERNAME"), myRank, numProcs);
            fflush(stderr);
        }
#else
        numProcs = 1;
        myRank = 0;
#endif

        ConfigParameters config;
        std::string rawConfigString = ConfigParameters::ParseCommandLine(argc, argv, config);

        // get the command param set they want
        wstring logpath = config("stderr", L"");
        //  [1/26/2015 erw, add done file so that it can be used on HPC]
        wstring DoneFile = config("DoneFile", L"");
        ConfigArray command = config("command", "train");

        if (logpath != L"")
        {
            for (int i = 0; i < command.size(); i++)
            {
                logpath += L"_";
                logpath += (wstring)command[i];
            }
            logpath += L".log";
            if (numProcs > 1)
            {
                std::wostringstream oss;
                oss << myRank;
                logpath += L"rank" + oss.str();
            }
            RedirectStdErr(logpath);
        }

#ifdef _WIN32
        PrintBuiltInfo();
#endif
        std::string timestamp = TimeDateStamp();

        if (myRank == 0) // main process
        {
            //dump config info
            fprintf(stderr, "running on %s at %s\n", GetHostName().c_str(), timestamp.c_str());
            fprintf(stderr, "command line options: \n");
            for (int i = 1; i < argc; i++)
                fprintf(stderr, "%s ", WCharToString(argv[i]).c_str());

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
        }

        //run commands
        std::string type = config("precision", "float");
        // accept old precision key for backward compatibility
        if (config.Exists("type"))
            type = config("type", "float");
        if (myRank == 0)
            fprintf(stderr, "\nprecision = %s\n", type.c_str());
        if (type == "float")
            DoCommand<float>(config);
        else if (type == "double")
            DoCommand<double>(config);
        else
            RuntimeError("invalid precision specified: %s", type.c_str());

        // still here , write a DoneFile if necessary 
        if (!DoneFile.empty()){
            FILE* fp = fopenOrDie(DoneFile.c_str(), L"w");
            fprintf(fp, "successfully finished at %s on %s\n", TimeDateStamp().c_str(), GetHostName().c_str());
            fcloseOrDie(fp);
        }
    }
    catch (const std::exception &err)
    {
        fprintf(stderr, "EXCEPTION occurred: %s", err.what());
#ifdef _DEBUG
        DebugBreak();
#endif
        return EXIT_FAILURE;
    }
    catch (...)
    {
        fprintf(stderr, "Unknown ERROR occurred");
#ifdef _DEBUG
        DebugBreak();
#endif
        return EXIT_FAILURE;
    }
#ifdef MPI_SUPPORT
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}

#ifdef __UNIX__
int main(int argc, char* argv[])
{
    wchar_t **wargs = new wchar_t*[argc];
    for (int i = 0; i < argc; ++i)
    {
        wargs[i] = new wchar_t[strlen(argv[i]) + 1];
        size_t ans = ::mbstowcs(wargs[i], argv[i], strlen(argv[i]) + 1);
        assert(ans == strlen(argv[i]));
    }
    int ret = wmain(argc, wargs);
    for (int i = 0; i < argc; ++i)
        delete[] wargs[i];
    delete[] wargs;
    return ret;
}
#endif
