//
// <copyright file="cn.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// cn.cpp : Defines the entry point for the console application.
//

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
#include "commandArgUtil.h"
#include "SimpleEvaluator.h"
#include "SimpleOutputWriter.h"
#include "datareader.h"
#include <chrono>
#include <algorithm>
#include "io.h"
#include "hostname.h"
#ifdef LEAKDETECT
#include "vld.h" // for memory leak detection
#endif
#include <vector>
#include "BestGpu.h"

using namespace std;
using namespace Microsoft::MSR::CNTK;

// internal test routine forward declaration
template <typename ElemType>
void TestCn(const ConfigParameters& config);

void RedirectStdErr(wstring logpath)
{
    fprintf (stderr, "Redirecting stderr to file %S\n", logpath.c_str());
    msra::files::make_intermediate_dirs (logpath);
    auto_file_ptr f (logpath.c_str(), "wb");
    if (_dup2 (_fileno (f), 2) == -1)
        throw runtime_error ("unexpected failure to redirect stderr to log file");
    setvbuf (stderr, NULL, _IONBF, 16384);   // unbuffer it
}

template <typename ElemType>
void DumpNodeInfo(const ConfigParameters& config)
{
    wstring modelPath = config("modelPath");
    wstring nodeName = config("nodeName",L"__AllNodes__");
    wstring defOutFilePath = modelPath + L"." + nodeName + L".txt";
    wstring outputFile = config("outputFile",  WCharToString(defOutFilePath.c_str()).c_str());
    bool printValues = config("printValues", "true");

    ComputationNetwork<ElemType> net(-1);  //always use CPU
    net.LoadFromFile(modelPath);
    net.DumpNodeInfoToFile(nodeName, printValues, outputFile);
}

template <typename ElemType>
void DoEvalBase(const ConfigParameters& config, IDataReader<ElemType>& reader)
{
    short deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config("minibatchSize", "40960");
    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config("modelPath");
    intargvector mbSize = minibatchSize;

    UINT16 traceLevel = config("traceLevel", "0");    
    size_t numMBsToShowResult = config("numMBsToShowResult", "100");

    ConfigArray evalNodeNames = config("evalNodeNames","");
    vector<wstring> evalNodeNamesVector;
    for (int i=0; i < evalNodeNames.size(); ++i)
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
    ConfigParameters readerConfig = config("reader");
    readerConfig.Insert("traceLevel",config("traceLevel","0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DoEvalBase(config, testDataReader);
}

template <typename ElemType>
void DoEvalUnroll(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig = config("reader");
    readerConfig.Insert("traceLevel",config("traceLevel","0"));

    DataReader<ElemType> testDataReader(readerConfig);

    short deviceId = DeviceFromConfig(config);
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
    eval.EvaluateUnroll(testDataReader, mbSize[0], evalEntropy,  path2EvalResults == L""? nullptr : path2EvalResults.c_str(), epochSize);
}

template <typename ElemType>
void DoCrossValidate(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig = config("reader");
    readerConfig.Insert("traceLevel",config("traceLevel","0"));

    short deviceId = DeviceFromConfig(config);
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

    UINT16 traceLevel = config("traceLevel", "0");    
    size_t numMBsToShowResult = config("numMBsToShowResult", "100");

    ConfigArray evalNodeNames = config("evalNodeNames","");
    vector<wstring> evalNodeNamesVector;
    for (int i=0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    std::vector<std::vector<ElemType>> cvErrorResults;
    std::vector<std::wstring> cvModels;

    DataReader<ElemType> cvDataReader(readerConfig);

    bool finalModelEvaluated = false;
    for (size_t i=cvInterval[0]; i<=cvInterval[2]; i+=cvInterval[1])
    {
        wstring cvModelPath = msra::strfun::wstrprintf (L"%ws.%lld", modelPath.c_str(), i);

        if (!fexists (cvModelPath)) 
        {
            fprintf(stderr, "model %ws does not exist.\n", cvModelPath.c_str());
            if (finalModelEvaluated || !fexists (modelPath))
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

        fprintf(stderr, "model %ws --> \n",cvModelPath.c_str());
        std::vector<ElemType> evalErrors;
        evalErrors = eval.Evaluate(cvDataReader, evalNodeNamesVector, mbSize[0], epochSize);
        cvErrorResults.push_back(evalErrors);

        Sleep(DWORD(1000*sleepSecondsBetweenRuns));
    }

    //find best model
    if (cvErrorResults.size() == 0)
        throw std::logic_error("No model is evaluated.");

    std::vector<ElemType> minErrors;
    std::vector<int> minErrIds;
    std::vector<ElemType> evalErrors = cvErrorResults[0];
    for (int i=0; i < evalErrors.size(); ++i)
    {
        minErrors.push_back(evalErrors[i]);
        minErrIds.push_back(0);
    }

    for (int i=0; i<cvErrorResults.size(); i++)
    {
        evalErrors = cvErrorResults[i];
        for (int j=0; j<evalErrors.size(); j++)
        {
            if (evalErrors[j] < minErrors[j])
            {
                minErrors[j] = evalErrors[j];
                minErrIds[j] = i;
            }        
        }
    }

    fprintf(stderr, "Best models:\n");
    fprintf(stderr,"------------\n");
    for (int i=0; i < minErrors.size(); ++i)
    {
        fprintf(stderr,"Based on Err[%d]: Best model = %ws with min err %.8g\n", i, cvModels[minErrIds[i]].c_str(), minErrors[i]);
    }
}

template <typename ElemType>
void DoWriteOutput(const ConfigParameters& config)
{
    ConfigParameters readerConfig = config("reader");
    readerConfig.Insert("traceLevel",config("traceLevel","0"));
    readerConfig.Insert("randomize","None");  //we don't want randomization when output results

    DataReader<ElemType> testDataReader(readerConfig);

    short deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config("minibatchSize", "2048");
    wstring modelPath = config("modelPath");
    intargvector mbSize = minibatchSize;
    
    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    ConfigArray outputNodeNames = config("outputNodeNames","");
    vector<wstring> outputNodeNamesVector;
    for (int i=0; i < outputNodeNames.size(); ++i)
    {
        outputNodeNamesVector.push_back(outputNodeNames[i]);
    }

    ComputationNetwork<ElemType> net(deviceId);
    net.LoadFromFile(modelPath);
    net.ResetEvalTimeStamp();

    SimpleOutputWriter<ElemType> writer(net, 1);

	if (config.Exists("writer"))
	{
		ConfigParameters writerConfig = config("writer");
        bool bWriterUnittest = writerConfig("unittest","false");
		DataWriter<ElemType> testDataWriter(writerConfig);
		writer.WriteOutput(testDataReader,mbSize[0], testDataWriter, outputNodeNamesVector, epochSize, bWriterUnittest);
	}
	else if (config.Exists("outputPath"))
	{
		wstring outputPath = config("outputPath"); // crashes if no default given? 
		writer.WriteOutput(testDataReader, mbSize[0], outputPath, outputNodeNamesVector, epochSize);
	}
	//writer.WriteOutput(testDataReader, mbSize[0], testDataWriter, outputNodeNamesVector, epochSize);
}

namespace Microsoft { namespace MSR { namespace CNTK {

TrainingCriterion ParseTrainingCriterionString(wstring s)
{
    transform(s.begin(), s.end(), s.begin(),tolower); 
    if (s==L"crossentropywithsoftmax")
        return TrainingCriterion::CrossEntropyWithSoftmax;
    else if (s==L"squareerror")
        return TrainingCriterion::SquareError;
    else if (s==L"classcrossentropywithsoftmax")
        return TrainingCriterion::ClassCrossEntropyWithSoftmax;
    else
        throw std::exception("trainingCriterion: Invalid trainingCriterion value. Valid values are (CrossEntropyWithSoftmax | SquareError)");
}

EvalCriterion ParseEvalCriterionString(wstring s)
{
    transform(s.begin(), s.end(), s.begin(),tolower); 
    if (s==L"errorprediction")
        return EvalCriterion::ErrorPrediction;
    else if (s==L"crossentropywithsoftmax")
        return EvalCriterion::CrossEntropyWithSoftmax;
    else if (s==L"classcrossentropywithsoftmax")
        return EvalCriterion::ClassCrossEntropyWithSoftmax;
    else if (s==L"squareerror")
        return EvalCriterion::SquareError;
    else 
        throw std::exception("evalCriterion: Invalid trainingCriterion value. Valid values are (ErrorPrediction | CrossEntropyWithSoftmax | SquareError)");
}


}}};

template <typename ElemType>
void DoCreateLabelMap(const ConfigParameters& config)
{
    // this gets the section name we are interested in
    std::string section = config("section");
    // get that section (probably a peer config section, which works thanks to heirarchal symbol resolution)
    ConfigParameters configSection=config(section);
    ConfigParameters readerConfig = configSection("reader");
    readerConfig.Insert("allowMapCreation","true");
    short deviceId = CPUDEVICE;
    size_t minibatchSize = config("minibatchSize", "2048");
    int traceLevel = config("traceLevel","0");
    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;
    GetFileConfigNames(readerConfig, featureNames, labelNames);

    // setup minibatch matrices
    Matrix<ElemType> featuresMatrix(deviceId);
    Matrix<ElemType> labelsMatrix(deviceId);
    std::map<std::wstring, Matrix<ElemType>*> matrices;
    matrices[featureNames[0]] = &featuresMatrix;
    if (labelNames.size() == 0)
        throw runtime_error("CreateLabelMap: no labels found to process");

    // now create the reader and loop through the entire dataset to get all the labels
    auto start = std::chrono::system_clock::now();
    for (const std::wstring& labelsName: labelNames)
    {
        // take the last label file defined (the other one might be input)
        matrices[labelsName] = &labelsMatrix;

        // get the label mapping file name
        ConfigParameters labelConfig = readerConfig(labelsName);
        std::string labelMappingFile;
        if (labelConfig.ExistsCurrent("labelMappingFile"))
            labelMappingFile = labelConfig("labelMappingFile");
        else if (readerConfig.ExistsCurrent("labelMappingFile")) 
            labelMappingFile = labelConfig("labelMappingFile");
        else
            throw runtime_error("CreateLabelMap: No labelMappingFile defined");

        if (fexists(labelMappingFile))
        {
            fprintf(stderr,"CreateLabelMap: the label mapping file '%s' already exists, no work to do.\n", labelMappingFile.c_str());
            return;
        }
        fprintf(stderr,"CreateLabelMap: Creating the mapping file '%s' \n", labelMappingFile.c_str());

        DataReader<ElemType> dataReader(readerConfig);

        dataReader.StartMinibatchLoop(minibatchSize, 0, requestDataSize);
        int count = 0;
        while (dataReader.GetMinibatch(matrices))
        {
            Matrix<ElemType>& features = *matrices[featureNames[0]];
            count += features.GetNumCols();
            if (traceLevel > 1)
                fprintf(stderr,"."); // progress meter
        }
        dataReader.StartMinibatchLoop(minibatchSize, 1, requestDataSize);

        // print the results
        if (traceLevel > 0)
            fprintf(stderr,"\nread %d labels and produced %s\n", count, labelMappingFile.c_str());
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = end-start;
    if (traceLevel > 1)
        fprintf(stderr, "%f seconds elapsed\n", (float)(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count())/1000);
}


template <typename ElemType>
void DoTrain(const ConfigParameters& config)
{
    ConfigParameters configSGD=config("SGD");
    bool makeMode = config("makeMode", "true");

    ConfigParameters readerConfig = config("reader");
    readerConfig.Insert("traceLevel",config("traceLevel","0"));

    IComputationNetBuilder<ElemType>* netBuilder = NULL;

    if (config.Exists("NDLNetworkBuilder"))
    {
        ConfigParameters configNDL = config("NDLNetworkBuilder");
        netBuilder = (IComputationNetBuilder<ElemType>*)new NDLBuilder<ElemType>(configNDL);
    }
    else if (config.Exists("SimpleNetworkBuilder"))
    {
        ConfigParameters configSNB = config("SimpleNetworkBuilder");
        netBuilder = (IComputationNetBuilder<ElemType>*)new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        throw runtime_error("No network builder found in the config file. NDLNetworkBuilder or SimpleNetworkBuilde must be specified" );
    }

    DataReader<ElemType>* dataReader = new DataReader<ElemType>(readerConfig);

    DataReader<ElemType>* cvDataReader = nullptr;
    ConfigParameters cvReaderConfig = config("cvReader", L"");
    
    if (cvReaderConfig.size() != 0)
    {
        cvReaderConfig.Insert("traceLevel",config("traceLevel","0"));
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
    short deviceId = DeviceFromConfig(config);

    ConfigParameters configSGD=config("SGD");
    bool makeMode = config("makeMode", "true");

    ConfigParameters readerConfig = config("reader");
    readerConfig.Insert("traceLevel",config("traceLevel","0"));

    DataReader<ElemType>* dataReader = new DataReader<ElemType>(readerConfig);

    DataReader<ElemType>* cvDataReader = nullptr;
    ConfigParameters cvReaderConfig = config("cvReader", L"");
    
    if (cvReaderConfig.size() != 0)
    {
        cvReaderConfig.Insert("traceLevel",config("traceLevel","0"));
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
    wstring ndlMacros = config("ndlMacros","");
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
// process the command
template <typename ElemType>
void DoCommand(const ConfigParameters& config)
{
    ConfigArray command = config("command", "train");
    for (int i=0; i < command.size(); i++)
    {
        //get the configuration parameters that match the command
        ConfigParameters commandParams=config(command[i]);
        ConfigArray action = commandParams("action","train");

        // determine the action to perform, and do it
        for (int j=0; j < action.size(); j++)
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
            else
                throw runtime_error("unknown action: " + action[j] + " in command set: " + command[i]);
			    
			NDLScript<ElemType> ndlScript;
			ndlScript.ClearGlobal(); // clear global macros between commands
        }
    }
}

std::string TimeDateStamp()
{
    __time64_t localtime;

    _time64 (&localtime);// get current time and date
    struct tm now;
    _localtime64_s (&now, &localtime);  // convert
    std::string s; s.resize(30);
    sprintf_s ( (char*)s.c_str(), s.size(), "%04d/%02d/%02d %02d:%02d:%02d",
        now.tm_year + 1900, now.tm_mon + 1, now.tm_mday,
        now.tm_hour, now.tm_min, now.tm_sec);
    return s;
}

std::string WCharToString(const wchar_t* wst)
{
    std::wstring ws(wst);
    std::string s(ws.begin(), ws.end());
    s.assign(ws.begin(), ws.end());
    return s;
}

int wmain(int argc, wchar_t* argv[])
{
    try
    {
        ConfigParameters config;
        std::string rawConfigString = ConfigParameters::ParseCommandLine(argc, argv, config);

        // get the command param set they want
        wstring logpath = config("stderr", L"");
        ConfigArray command = config("command", "train");

        if (logpath != L"")
        {
            for (int i=0; i < command.size(); i++)
            {
                logpath += L"_";
                logpath += command[i];
            }
            logpath += L".log";
            RedirectStdErr(logpath);
        }

        std::string timestamp = TimeDateStamp();

        //dump config info
        fprintf (stderr, "running on %s at %s\n", GetHostName().c_str(), timestamp.c_str());
        fprintf (stderr, "command line options: \n");
        for (int i = 1; i < argc; i++)
            fprintf (stderr, "%s ", WCharToString(argv[i]).c_str());

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
        for (int i=0; i < command.size(); i++)
        {
            fprintf(stderr, "%s ", command[i].c_str());
        }

        //run commands
        std::string type = config("precision", "float");
        // accept old precision key for backward compatibility
        if (config.Exists("type"))
            type = config("type", "float");
        fprintf(stderr, "\nprecision = %s\n", type.c_str());
        if (type == "float")
            DoCommand<float>(config);
        else if (type == "double")
            DoCommand<double>(config);
        else
            throw runtime_error("invalid precision specified: " + type);
    }
    catch(std::exception &err)
    {
        fprintf(stderr, "EXCEPTION occurred: %s", err.what());
#ifdef _DEBUG
        DebugBreak();
#endif
        return -1;
    }
    catch(...)
    {
        fprintf(stderr, "Unknown ERROR occurred");
#ifdef _DEBUG
        DebugBreak();
#endif
        return -1;
    }    
    return 0;
}
