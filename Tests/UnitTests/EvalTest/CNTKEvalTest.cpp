// CNTKEvalTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Eval.h"
#include "DataReader.h"
#include "Config.h"
using namespace Microsoft::MSR::CNTK;

// process the command
template <typename ElemType>
void DoCommand(const ConfigParameters& configRoot)
{
    ConfigArray command = configRoot("command", "train");
    ConfigParameters config = configRoot(command[0]);
    ConfigParameters readerConfig(config("reader"));
    readerConfig.Insert("traceLevel", config("traceLevel", "0"));

    ConfigArray minibatchSize = config("minibatchSize", "256");
    intargvector mbSizeArr = minibatchSize;
    size_t mbSize = 20000; // mbSizeArr[0];
    size_t epochSize = config("epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    ConfigParameters configFeatures = readerConfig(L"features");
    size_t dimFeatures = configFeatures("dim");
    ConfigParameters configLabels = readerConfig(L"labels");
    size_t dimLabels = configLabels("labelDim");
    ConfigParameters configSgd = config("SGD");
    std::wstring modelPath = configSgd("modelPath");

    StreamMinibatchInputs inputMatrices;
    StreamMinibatchInputs outputMatrices;
    std::wstring inputName = L"features";
    std::wstring outputName = L"CE.BFF.FF.P";
    int deviceId = 0;
    auto matrix = make_shared<Matrix<ElemType>>(dimFeatures, mbSize, deviceId);
    MBLayoutPtr pMBLayout = make_shared<MBLayout>();
    inputMatrices.AddInput(inputName, matrix, pMBLayout, TensorShape(dimFeatures));
    outputMatrices.AddInput(outputName, make_shared<Matrix<ElemType>>(dimLabels, mbSize, deviceId), pMBLayout, TensorShape(dimLabels));

    std::map<std::wstring, std::vector<ElemType>*> input;
    std::map<std::wstring, std::vector<ElemType>*> output;
    std::vector<ElemType>* arr = input[inputName] = new std::vector<ElemType>(dimFeatures * mbSize);
    output[outputName] = new std::vector<ElemType>(dimLabels * mbSize);

    Eval<ElemType> eval(config);

    auto dataReader = make_shared<DataReader>(readerConfig);
    string strPath(modelPath.begin(), modelPath.end());
    eval.CreateNetwork(strPath);
    dataReader->StartMinibatchLoop(mbSize, 0, epochSize);
    eval.StartEvaluateMinibatchLoop(outputName);
    while (dataReader->GetMinibatch(inputMatrices))
    {
        void* data = (void*) arr->data();
        size_t dataSize = arr->size() * sizeof(ElemType);
        void* mat = &(*matrix)(0, 0);
        size_t matSize = matrix->GetNumElements() * sizeof(ElemType);
        memcpy_s(data, dataSize, mat, matSize);
        eval.Evaluate(input, output);
    }
}

int wmain(int argc, wchar_t* argv[])
{
    try
    {
        ConfigParameters config;
        ConfigParameters::ParseCommandLine(argc, argv, config);

        // get the command param set they want
        wstring logpath = config("stderr", L"");
        ConfigArray command = config("command", "train");

        // dump config info
        fprintf(stderr, "command: ");
        for (int i = 0; i < command.size(); i++)
        {
            fprintf(stderr, "%s ", command[i].c_str());
        }

        // run commands
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
            RuntimeError("invalid precision specified: %s", type.c_str());
    }
    catch (std::exception& err)
    {
        fprintf(stderr, "EXCEPTION occurred: %s", err.what());
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
#ifdef _DEBUG
        DebugBreak();
#endif
        return -1;
    }
    catch (...)
    {
        fprintf(stderr, "Unknown ERROR occurred");
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
#ifdef _DEBUG
        DebugBreak();
#endif
        return -1;
    }
    return 0;
}
