//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// cn.cpp : Defines the entry point for the console application.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include <string>
#include <chrono>

#include "Basics.h"
#include "ComputationNetwork.h"
#include "DataReader.h"
#include "SimpleNetworkBuilder.h"
#include "SGD.h"
#include "SimpleEvaluator.h"
#include "NetworkDescriptionLanguage.h"
#include "NDLNetworkBuilder.h"

using namespace std;
using namespace Microsoft::MSR::CNTK;

#if 0 // unused, and does not build without SimpleSGD which is not present in this Solution
template <typename ElemType>
void TestBing(const ConfigParameters& config)
{
    if (!config.Exists("train.set"))
    {
        std::cout<<"USAGE: cn.exe train.set featureDim networkDescription learnRatesPerMB mbSize epochSize maxEpochs outdir test.set test.set.size"<<endl;
        exit(0);
    }

    size_t vdim = config("featureDim");
    size_t udim = 1;
    vector<wstring> filepaths;
    filepaths.push_back(config("train.set"));

    DataReader<ElemType> dataReader(vdim, udim, filepaths, config);
    ConfigArray layerSizes(config("networkDescription"));
    SimpleNetworkBuilder<ElemType> netBuilder(layerSizes, TrainingCriterion::SquareError, EvalCriterion::SquareError, L"Sigmoid", true, false, false, &dataReader);


    ConfigArray learnRatesPerMB(config("learnRatesPerMB"));
    ConfigArray mbSize(config("mbSize"));
    size_t epochSize = config("epochSize");
    size_t maxEpochs = config("maxEpochs");
    float momentumPerMB = 0.9;//0.9f;
    std::string outDir = config("outdir");
    wstring modelPath = wstring(msra::strfun::utf16(outDir)).append(L"\\bingranknet.dnn");

    SimpleSGD<ElemType> sgd(learnRatesPerMB, mbSize, epochSize, maxEpochs, modelPath, momentumPerMB);
    sgd.Train(netBuilder, dataReader, true);

    std::cout<<std::endl<<std::endl<<std::endl<<std::endl<<"Testing ..... "<<std::endl;

    // test
    vector<wstring> testfilepaths;
    testfilepaths.push_back( config("test.set"));
    size_t testSize = config("test.set.size");
    DataReader<ElemType> testDataReader(vdim, udim, testfilepaths, config);

    wstring finalNetPath = modelPath.append(L".").append(to_wstring(maxEpochs-1));

    SimpleEvaluator<ElemType> eval(netBuilder.LoadNetworkFromFile(finalNetPath, false));
    eval.Evaluate(testDataReader, 1024, (finalNetPath.append(L".results.txt")).c_str(),testSize);
}
#endif

template <typename ElemType>
void DoEval(const ConfigParameters& config);
template <class ConfigRecordType, typename ElemType>
void DoTrain(const ConfigRecordType& config);

template <typename ElemType>
void TestMNist(const ConfigParameters& configBase)
{
    ConfigParameters config(configBase("mnistTrain"));
    DoTrain<ConfigParameters, ElemType>(config);
    ConfigParameters configTest(configBase("mnistTest"));
    DoEval<ElemType>(configTest);
}

template <typename ElemType>
void TestSpeech(const ConfigParameters& configBase)
{
    ConfigParameters config(configBase("speechTrain"));
    DoTrain<ConfigParameters, ElemType>(config);
    ConfigParameters configTest(configBase("speechTest"));
    DoEval<ElemType>(configTest);
}

template <typename ElemType>
void TestReader(const ConfigParameters& configBase)
{
    // int nonexistant = configBase("nonexistant");  // use to test global exception handler
    ConfigParameters config(configBase("mnistTest"));
    ConfigParameters readerConfig(config("reader"));
    readerConfig.Insert("traceLevel", config("traceLevel", "0"));

    size_t mbSize = config("minibatchSize");
    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    DataReader dataReader(readerConfig);

    // get names of features and labels
    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;
    GetFileConfigNames(readerConfig, featureNames, labelNames);

    // setup minibatch matrices
    int deviceId = 0;
    auto featuresMatrix = make_shared<Matrix<ElemType>>(deviceId);
    auto labelsMatrix   = make_shared<Matrix<ElemType>>(deviceId);
    MBLayoutPtr pMBLayout = make_shared<MBLayout>();
    StreamMinibatchInputs matrices;
    matrices.AddInput(featureNames[0], featuresMatrix, pMBLayout, TensorShape());
    matrices.AddInput(labelNames[0],   labelsMatrix  , pMBLayout, TensorShape());

    auto start = std::chrono::system_clock::now();
    int epochs = config("maxEpochs");
    epochs *= 2;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);
        int i = 0;
        while (dataReader.GetMinibatch(matrices))
        {
            Matrix<ElemType>& features = matrices.GetInputMatrix<ElemType>(featureNames[0]);
            Matrix<ElemType>& labels   = matrices.GetInputMatrix<ElemType>(labelNames[0]);

            if (labels.GetNumRows() == 0)
            {
                fprintf(stderr, "%4d: features dim: %lu x %lu - [%.8g, %.8g, ...]\n", i++, features.GetNumRows(), features.GetNumCols(), features(0, 0), features(0, 1));
            }
            else
            {
                fprintf(stderr, "%4d: features dim: %lu x %lu - [%.8g, %.8g, ...] label dim: %lu x %lu - [%d, %d, ...]\n", i++, features.GetNumRows(), features.GetNumCols(), features(0, 0), features(0, 1), labels.GetNumRows(), labels.GetNumCols(), (int) labels(0, 0), (int) labels(0, 1));
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;
    fprintf(stderr, "%f seconds elapsed", (float) (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()) / 1000);
}

template <typename ElemType>
void TestSequenceReader(const ConfigParameters& configBase)
{
    // int nonexistant = configBase("nonexistant");  // use to test global exception handler
    ConfigParameters config = configBase("sequenceTest");

    size_t mbSize = config("minibatchSize");
    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    for (int fileType = 0; fileType < 2; ++fileType)
    {
        ConfigParameters readerConfig = config(fileType ? "readerSequence" : "readerSentence");
        readerConfig.Insert("traceLevel", config("traceLevel", "0"));

        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;
        GetFileConfigNames(readerConfig, featureNames, labelNames);

        DataReader dataReader(readerConfig);

        // get names of features and labels
        std::vector<std::wstring> files;
        files.push_back(readerConfig(L"file"));

        // setup minibatch matrices
        auto featuresMatrix = make_shared<Matrix<ElemType>>();
        auto labelsMatrix   = make_shared<Matrix<ElemType>>();
        MBLayoutPtr pMBLayout = make_shared<MBLayout>();
        StreamMinibatchInputs matrices;
        matrices.AddInput(featureNames[0], featuresMatrix, pMBLayout, TensorShape());
        matrices.AddInput(labelNames[0],   labelsMatrix  , pMBLayout, TensorShape());

        auto start = std::chrono::system_clock::now();
        int epochs = config("maxEpochs");
        epochs *= 2;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);
            for (int i = 0; dataReader.GetMinibatch(matrices); i++)
            {
                auto& features = matrices.GetInputMatrix<ElemType>(featureNames[0]);
                auto& labels   = matrices.GetInputMatrix<ElemType>(labelNames[1]);
                fprintf(stderr, "%4d: features dim: %lu x %lu - [%.8g, %.8g, ...] label dim: %d x %d - [%d, %d, ...]\n", i, features.GetNumRows(), features.GetNumCols(), features(0, 0), features(0, 1), labels.GetNumRows(), labels.GetNumCols(), (int) labels(0, 0), (int) labels(0, 1));
            }
        }
        auto end = std::chrono::system_clock::now();
        auto elapsed = end - start;
        fprintf(stderr, "%f seconds elapsed", (float) (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()) / 1000);
    }
}

bool IsParameter(std::map<std::string, ConfigValue>& paramsMap, std::string value)
{
    return paramsMap.find(value) != paramsMap.end();
}

template <typename ElemType>
void TestMacros(const ConfigParameters& configBase)
{
    NDLScript<ElemType> script = configBase("ndlFull");
    ComputationNetworkPtr net = make_shared<ComputationNetwork>();
    NDLNodeEvaluatorImpl<ElemType> nodeEvaluator(net);
    script.Evaluate(nodeEvaluator, L"", ndlPassInitial);
}

template <typename ElemType>
void TestConfiguration(const ConfigParameters& configBase)
{
    ConfigParameters configMacros = configBase("macroExample");
    for (auto iterMacro = configMacros.begin(); iterMacro != configMacros.end(); iterMacro++)
    {
        std::map<std::string, ConfigValue> paramsMap;
        ConfigParameters configCN = iterMacro->second;
        if (configCN.Exists("parameters"))
        {
            ConfigArray params = configCN("parameters");
            for (int i = 0; i < params.size(); ++i)
                paramsMap[params[i]] = ConfigValue("uninitialized");
        }
        ConfigParameters configNodes = configCN("NodeList");
        for (auto iter = configNodes.begin();
             iter != configNodes.end(); iter++)
        {
            std::wstring nodeName;
            nodeName = msra::strfun::utf16(iter->first);
            ConfigArray configNode = iter->second;
            std::string opName = configNode[0];
            if (IsParameter(paramsMap, opName))
            {
                ;
            }
            if (opName == "InputValue" && configNode.size() >= 2)
            {
                size_t rows = 0;
                if (!IsParameter(paramsMap, configNode[1]))
                    rows = configNode[1];
            }
            else if (opName == "LearnableParameter" && configNode.size() >= 3)
            {
                size_t rows = 0;
                if (!IsParameter(paramsMap, configNode[1]))
                    rows = configNode[1];
                size_t cols = 0;
                if (!IsParameter(paramsMap, configNode[2]))
                    cols = configNode[2];
                bool learningRateMultiplier = 0;
                bool init = false;
                ConfigArray initData;

                // look for optional parameters
                for (int i = 3; i < configNode.size(); ++i)
                {
                    bool needsGradient = false;
                    ConfigParameters configParam = configNode[i];
                    if (configParam.Exists("learningRateMultiplier")) // TODO: should this be a test for 'true' rather than Exists()?
                        needsGradient = (float)configParam("learningRateMultiplier") > 0? true : false;
                    else if (configParam.Exists("init"))
                    {
                        init = true;
                        initData = configParam["init"];
                    }
                }
                // if initializing, do so now
                if (init)
                {
                    bool uniform = true;
                    ElemType initValueScale = 1;
                    size_t inputSize = cols;

                    if (initData.size() > 0)
                        initValueScale = initData[0];
                    if (initData.size() > 1)
                        uniform = EqualCI(initData[1], "uniform");
                }
            }
        }

        // now link up all the nodes
        configNodes = configCN("Relation");
        for (auto iter = configNodes.begin(); iter != configNodes.end(); iter++)
        {
            std::wstring nodeName = msra::strfun::utf16(iter->first);
            ConfigArray configNode = iter->second;
            int numChildren = (int) configNode.size();
            for (int i = 0; i < numChildren; ++i)
            {
                std::wstring nodeName = configNode[i];
            }
        }

        ConfigParameters configRoots = configCN("RootNodes");
        ConfigArray configNode = configRoots("FeatureNodes");
        for (size_t i = 0; i < configNode.size(); i++)
        {
            std::wstring nodeName = configNode[i];
        }

        if (configRoots.Exists("LabelNodes"))
        {
            configNode = configRoots("LabelNodes");
            for (size_t i = 0; i < configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
            }
        }

        if (configRoots.Exists("CriterionNodes"))
        {
            configNode = configRoots("CriterionNodes");
            for (size_t i = 0; i < configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
            }
        }

        if (configRoots.Exists("CriteriaNodes")) // legacy
        {
            configNode = configRoots("CriteriaNodes");
            for (size_t i = 0; i < configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
            }
        }

        if (configRoots.Exists("NodesReqMultiSeqHandling"))
        {
            configNode = configRoots("NodesReqMultiSeqHandling");
            for (size_t i = 0; i < configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
            }
            fprintf(stderr, "WARNING: 'NodesReqMultiSeqHandling' flag is defunct\n");
        }

        if (configRoots.Exists("EvalNodes"))
        {
            configNode = configRoots("EvalNodes");
            for (size_t i = 0; i < configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
            }
        }

        if (configRoots.Exists("OutputNodes"))
        {
            configNode = configRoots("OutputNodes");
            for (size_t i = 0; i < configNode.size(); i++)
            {
                std::wstring nodeName = configNode[i];
            }
        }
    }
}

template <typename ElemType>
void TestCommandLine(const ConfigParameters& configBase)
{
    //    commandLine=[
    ConfigParameters config(configBase("commandline"));
    ConfigParameters stringTests(config("stringTests"));
    ConfigParameters unicodeTests(stringTests("unicodeTests"));
    ConfigParameters arrayTests(config("arrayTests"));
    ConfigParameters dictTests(config("dictTests"));
    //    # config file parsing, basic types
    //    int=20
    int i = config("int", "5555");
    // cout << i << endl;
    i = config("nothere", "1234");
    // cout << i << endl;
    //    long=8100000
    long l = config(L"long", "5555");
    // cout << l << endl;
    l = config("nothere", "1234");
    //    size_t=12345678901234
    // should get the same thing asking from a double nested config
    l = unicodeTests(L"long", "5555");
    // cout << l << endl;
    l = unicodeTests("nothere", "1234");
    //    size_t=12345678901234
    size_t s = config("size_t", "5555");
    // cout << s << endl;
    s = config(L"nothere", "1234");

    // get stuff from base level config (3 levels down)
    string type = unicodeTests("type");
    string command = unicodeTests("command");

    //    boolTrue=t
    bool bt = config("boolTrue");
    //    boolFalse=f
    bool bf = config("boolFalse");
    //    boolImpliedTrue
    bool bit = config("boolImpliedTrue");
    bit;
    bool bif = config.Exists(L"boolImpliedFalse");
    bif;
    bf = config("nothere", "false");
    // cout << bf << endl;
    //    float=1234.5678
    float f = config("float", "555.555");
    // cout << f << endl;
    f = config("nothere", "1.234");
    //    double=1.23456e-99
    double d = config("double", "555.555");
    // cout << d << endl;
    d = stringTests("nothere", "1.2345");
    //    string=string1
    std::string str = stringTests("string");
    str = stringTests(L"nothere", "default");
    // cout << str << endl;
    str = stringTests("nothere", "defString");
    // cout << str << endl;
    //    stringQuotes="This is a string with quotes"
    str = stringTests("stringQuotes");
    //    wstring=東京
    std::wstring wstr = unicodeTests("wstring");
    wstr = (std::wstring) unicodeTests(L"nothere", L"newValue");
    // wcout << wstr << endl;
    wstr = (std::wstring) unicodeTests("nothere", L"defWstring");
    //    wstringQuotes="東京に行きましょう． 明日"
    std::wstring wstrQuotes = unicodeTests("wstringQuotes");
    //
    //    #array tests
    //    arrayEmpty={}
    ConfigArray arrayEmpty = arrayTests("arrayEmpty");
    //    arraySingle=hello
    ConfigArray arraySingle = arrayTests("arraySingle");
    //    arrayMultiple=hello;there
    ConfigArray arrayMultiple = arrayTests(L"arrayMultiple");
    //    arrayMultipleBraces={hello:there:with:braces}
    ConfigArray arrayMultipleBraces = arrayTests("arrayMultipleBraces");
    // arrayMultiple = arrayTests("nothere", arrayMultipleBraces); - no longer supported, can add if we need it
    //    arrayMultipleSeparatorBraces={|hello|there|with|custom|separator|and|braces}
    ConfigArray arrayMultipleSeparatorBraces = arrayTests("arrayMultipleSeparatorBraces");
    //    arrayQuotedStrings={
    //      "c:\path with spaces\file.txt"
    //      "d:\data\Jan 21 1999.my file.txt"
    //    }
    ConfigArray arrayQuotedStrings = arrayTests("arrayQuotedStrings");
    str = arrayQuotedStrings[0];
    str = arrayQuotedStrings[1];
    //    arrayRepeat=1*1:2*2:3*3
    ConfigArray arrayRepeat = arrayTests("arrayRepeat");
    //    arrayHetro={string;明日;123;1.234e56;True;dict=first=1;second=2}
    ConfigArray arrayHetro = arrayTests("arrayHetro");
    str = arrayHetro[0];
    wstr = (std::wstring) arrayHetro[1];
    i = arrayHetro[2];
    d = arrayHetro[3];
    bt = arrayHetro[4];
    ConfigParameters dict(arrayHetro[5]);
    std::string name = dict.Name();
    //    arrayNested={|1:2:3|first:second:third|t:f:t|{|identical*2|separator*1|nested*3}}
    ConfigArray arrayNested = arrayTests(L"arrayNested");
    name = arrayNested.Name();
    ConfigArray array1 = arrayNested[0];
    name = array1.Name();
    ConfigArray array2 = arrayNested[1];
    name = array2.Name();
    ConfigArray array3 = arrayNested[2];
    name = array3.Name();
    ConfigArray array4 = arrayNested[3];
    //
    //    #dictionary tests
    //    dictEmpty=[]
    ConfigParameters dictEmpty(dictTests("dictEmpty"));
    //    dictSingle=first=1
    ConfigParameters dictSingle(dictTests("dictSingle"));
    //    dictMultiple=first=hello;second=there
    ConfigParameters dictMultiple(dictTests("dictMultiple"));
    //    dictMultipleBraces=[first=hello;second=there;third=with;forth=braces]
    ConfigParameters dictMultipleBraces(dictTests(L"dictMultipleBraces"));
    // dictMultiple = dictTests("nothere", dictMultipleBraces); no longer supported, can add if we need
    //    dictMultipleSeparatorBraces=[|first=hello|second=1|thirdBool|forth=12.345|fifth="quoted string"|sixth=古部|seventh=braces]
    ConfigParameters dictMultipleSeparatorBraces(dictTests("dictMultipleSeparatorBraces"));
    //    dictQuotedStrings=[
    //        files={
    //            "c:\path with spaces\file.txt"
    //            "d:\data\Jan 21 1999.my file.txt"
    //        }
    //        mapping="e:\the path\that has\spaces"
    //    ]
    ConfigParameters dictQuotedStrings(dictTests("dictQuotedStrings"));
    arrayQuotedStrings = dictQuotedStrings("files");
    const char* mapping = dictQuotedStrings("mapping");
    mapping;
    //
    //    #super nesting
    //    dictNested=[
    //        array={
    //            dict1=1;dict2=2;dict3=3
    //            #embedded comment - non-named dictionary
    //            [
    //                first=1
    //                second=2
    //                third=3
    //            ]
    //            {
    //                [%first=n1%second=n2%third=n3%forth=embedded;old;separator]
    //                d1=n1;d2=n2;d3=n3
    //            }
    //        }
    //        dict2=[first=1;second=2;third=3]
    //        dict3=[@
    //            first=1@second=2
    //            third=3
    //            forth=4@fifth=5
    //        ]
    //    ]
    ConfigParameters dictNested(config("dictNested"));
    name = dictNested.Name();
    ConfigArray array = dictNested("array");
    name = array.Name();
    ConfigParameters dictElement1(array[0]);
    name = dictElement1.Name();
    ConfigParameters dictElement2(array[1]);
    name = dictElement2.Name();
    ConfigArray arrayNest(array[2]);
    name = arrayNest.Name();
    ConfigParameters dictNElement1(arrayNest[0]);
    name = dictNElement1.Name();
    ConfigParameters dictNElement2(arrayNest[1]);
    name = dictNElement2.Name();
    ConfigParameters dict2(dictNested(L"dict2"));
    name = dict2.Name();
    ConfigParameters dict3(dictNested("dict3"));
    name = dict3.Name();
    //]

    // File file(L"c:\\temp\\testing.txt", fileOptionsRead | fileOptionsText);
    // char marker[5];
    // if (file.TryGetMarker(fileMarkerBeginSection, L"BVAL"))
    // {
    //    if (file.TryGetMarker(fileMarkerBeginSection, L"BNEW"))
    //    {
    //        if (!file.TryGetMarker(fileMarkerBeginSection, L"OTHER"))
    //        {
    //            float val;
    //            file >> val;
    //            printf("pass");
    //        }
    //    }
    // }
    // TestConfiguration<ElemType>(configBase);
    TestMacros<ElemType>(configBase);
}

template <typename ElemType>
void TestCn(const ConfigParameters& config)
{
    TestCommandLine<ElemType>(config);
    // TestSequenceReader<ElemType>(config);
    TestReader<ElemType>(config);
    TestMNist<ElemType>(config);
    TestSpeech<ElemType>(config);
    // TestBing<ElemType>(config);
}

template void TestCn<float>(const ConfigParameters& config);
template void TestCn<double>(const ConfigParameters& config);

// generate TestRoutines
void GenerateTemplates()
{
    ConfigParameters config;
    TestCn<float>(config);
    TestCn<double>(config);
}
