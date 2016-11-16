//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#include <functional>
#include <thread>
#include <iostream>
#include "CNTKLibrary.h"

using namespace CNTK;

#include <graphid.pb.h>
#include <google/protobuf/util/json_util.h>

extern "C"
{
#include <b64/cencode.h>
}

class FpgaFunction : public UserDefinedFunctionHandler
{
public:

    FpgaFunction(
        std::vector<Variable>& inputs,
        Dictionary&& functionConfig,
        const std::wstring& name,
        const std::wstring& uid)
        : _inputs(inputs), _functionConfig(functionConfig), _name(name), _uid(uid)
    {
    }

    virtual /*BackPropStatePtr*/void ForwardFloat(
        std::vector<float>& out,
        const std::vector<float>& left,
        const std::vector<float>& right
    ) override
    {
        fprintf(stderr, "FpgaFunction::Forward(...) called\n");

        for (auto n = 0; n < out.size(); n++)
        {
            out[n] = n;
        }
    }

    virtual void Backward(
        ////const BackPropStatePtr& /*state*/,
        ////const std::unordered_map<Variable, ValuePtr>& /*rootGradientValues*/,
        ////std::unordered_map<Variable, ValuePtr>& /*backPropagatedGradientValuesForInputs*/
    ) override
    {
        NOT_IMPLEMENTED;
    }

private:
    std::vector<Variable> _inputs;
    Dictionary _functionConfig;
    const std::wstring _name;
    const std::wstring _uid;
};


template <typename FunctionType>
void Traverse(const FunctionPtr& rootFunction, std::unordered_set<FunctionPtr>& visitedFunctions, const FunctionType& functor)
{
    visitedFunctions.insert(rootFunction);
    functor(rootFunction);

    std::vector<Variable> rootFunctionInputs = rootFunction->Inputs();
    for (const auto& rootInput : rootFunctionInputs)
    {
        if (rootInput.IsOutput() && visitedFunctions.find(rootInput.Owner()) == visitedFunctions.end())
        {
            const auto& function = rootInput.Owner();
            Traverse(function, visitedFunctions, functor);
        }
    }
}

std::string ConstructUniqueName(const std::wstring& uid, const std::wstring& name)
{
    return std::string(uid.begin(), uid.end()) + "/" + std::string(name.begin(), name.end());
}

std::string ConstructUniqueName(const std::wstring& name)
{
    return std::string(name.begin(), name.end());
}

graphIR::Graph CntkGraphToGraphIr(FunctionPtr evalFunc, const DeviceDescriptor& device)
{
    graphIR::Graph &graph = *(new graphIR::Graph());

    graphIR::GraphInfo graphInfo;
    graphInfo.set_description("my description");
    graphInfo.set_framework_name("cntk-2.0beta1.0");
    graphInfo.set_framework_version("2.0beta1.0");
    graphInfo.set_graph_version("1.0");
    graphInfo.set_model_name("my-sluhandson.cntk");

    graph.set_allocated_graph_info(&graphInfo);

    auto serilized = evalFunc->Serialize();

    {
        evalFunc->SaveModel(L"00_fvm.log", false);
    }

    std::unordered_set<FunctionPtr> functions;
    Traverse(evalFunc->RootFunction(), functions, [&graph](const FunctionPtr& f){
        fprintf(stderr, "now at %S opcode %S\n", f->Uid().c_str(), f->OpName().c_str());
        
        graphIR::Node *node = graph.add_nodes();

        node->set_name(ConstructUniqueName(f->Uid(), f->Name()));

        auto name = f->OpName();
        node->set_op(std::string(name.begin(), name.end()));

        auto d = f->Attributes();
        std::stringstream strstr(std::ios_base::in | std::ios_base::out | std::ios_base::binary);
        strstr << d;
        auto where = strstr.tellp();
        auto str = strstr.str();

        base64_encodestate state;
        base64_init_encodestate(&state);

        graphIR::InitArg initArg;
        initArg.set_dbytes(4); // fp32 is 4 bytes per entry
        char *sout = new char[str.length() * 2];
        memset(sout, 0, str.length() * 2);
        base64_encode_block((const char *)str.c_str(), str.length(), sout, &state);
        base64_encode_blockend(sout, &state);

        if (strlen(sout) > 100)
        {
            strcpy_s(sout + 90, str.length()*2 - 100, "...");
        }

        (*node->mutable_ext_attrs())["##CNTK##NODE##"] = sout;

        delete[] sout;



        for (auto out : f->Placeholders())
        {
            fprintf(stderr, "oops\n");
        }

        for (auto out : f->Inputs())
        {
            auto input = node->add_inputs();

            input->set_name(ConstructUniqueName(out.Uid(), out.Name()));

            name = L"fp32";
            input->set_dtype(std::string(name.begin(), name.end()));

            input->set_dbytes(4); // fp32 is 4 bytes per entry

            fprintf(stderr, "    <= %S type %d [", out.Name().c_str(), out.GetDataType());

            int rank = 0;
            for (auto dims : out.Shape().Dimensions())
            {
                input->add_shape(dims);

                if (rank++ != 0) fprintf(stderr, ", ");
                fprintf(stderr, "%d", dims);
            }

            fprintf(stderr, "]\n");
        }

        for (auto out : f->Parameters())
        {
            const auto& buf = out.Value()->DataBuffer<float>();

            size_t rank = 1;
            for (auto dims : out.Shape().Dimensions())
            {
                rank *= dims;
            }

            graphIR::InitArg initArg;
            initArg.set_dbytes(4); // fp32 is 4 bytes per entry

            base64_encodestate state;
            base64_init_encodestate(&state);

            char *sout = new char[rank * 4 * 2];
            memset(sout, 0, rank * 4 * 2);
            base64_encode_block((char *)buf, rank * 4, sout, &state);
            base64_encode_blockend(sout, &state);

            // TODO: remove this to export the entire data, not just
            //       the first 120bytes...
            if (strlen(sout) > 100)
            {
                strcpy_s(sout + 90, rank * 4 * 2 -100, "...");
            }

            initArg.set_data_base64(sout);
            delete [] sout;
            
            (*node->mutable_init_attrs())[ConstructUniqueName(out.Uid(), out.Name())] = initArg;

            fprintf(stderr, "    == %S type %d value %f\n", out.Name().c_str(), out.GetDataType(), buf[0]);
        }

        for (auto out : f->Constants())
        {
            const auto& buf = out.Value()->DataBuffer<float>();

            size_t rank = 1;
            for (auto dims : out.Shape().Dimensions())
            {
                rank *= dims;
            }

            graphIR::InitArg initArg;
            initArg.set_dbytes(4); // fp32 is 4 bytes per entry


            base64_encodestate state;
            base64_init_encodestate(&state);

            char *sout = new char[rank * 4 * 2];
            memset(sout, 0, rank * 4 * 2);
            base64_encode_block((const char *)buf, rank * 4, sout, &state);
            base64_encode_blockend(sout, &state);

            if (strlen(sout) > 100)
            {
                strcpy_s(sout + 90, rank * 4 * 2 - 100, "...");
            }

            initArg.set_data_base64(sout);
            delete[] sout;

            (*node->mutable_init_attrs())[ConstructUniqueName(out.Uid(), out.Name())] = initArg;

            fprintf(stderr, "    == %S type %d value %f\n", out.Name().c_str(), out.GetDataType(), buf[0]);
        }

        for (auto &iter = f->Attributes().begin(); iter != f->Attributes().end(); iter++)
        {
            DictionaryValue value = iter->second;


            std::wstring resultValue = L"";

            switch (value.ValueType())
            {
            case DictionaryValue::Type::Bool:
                resultValue = std::to_wstring(iter->second.Value<bool>());
                break;

            case DictionaryValue::Type::Int:
                resultValue = std::to_wstring(iter->second.Value<int>());
                break;

            case DictionaryValue::Type::SizeT:
                resultValue = std::to_wstring(iter->second.Value<size_t>());
                break;

            case DictionaryValue::Type::Double:
                resultValue = std::to_wstring(iter->second.Value<double>());
                break;

            case DictionaryValue::Type::String:
                resultValue = iter->second.Value<std::wstring>();
                break;

            case DictionaryValue::Type::Float:
                resultValue = std::to_wstring(iter->second.Value<float>());
                break;

            default:
                resultValue = std::wstring(L"<<unsupported>>");
                break;
            }

            (*node->mutable_ext_attrs())[ConstructUniqueName(iter->first)] = std::string(resultValue.begin(), resultValue.end());
        }

        // Combine nodes are special, they just reflect their input to their output
        if (f->OpName() != L"Combine")
        {
            for (auto out : f->Outputs())
            {
                auto output = node->add_outputs();
                output->set_name(ConstructUniqueName(out.Uid(), out.Name()));

                name = L"fp32";
                output->set_dtype(std::string(name.begin(), name.end()));

                output->set_dbytes(4); // fp32 is 4 bytes per entry

                fprintf(stderr, "    => %S type %d [", out.Name().c_str(), out.GetDataType());

                int rank = 0;
                for (auto dims : out.Shape().Dimensions())
                {
                    output->add_shape(dims);

                    if (rank++ != 0) fprintf(stderr, ", ");
                    fprintf(stderr, "%d", dims);
                }

                fprintf(stderr, "]\n");
            }
        }
    });

    fprintf(stderr, "\n\n");
    for (auto func : functions)
    {
        fprintf(stderr, "X uid %S, op %S\n", func->Uid().c_str(), func->OpName().c_str());
    }

    return graph;
}

CNTK::FunctionPtr GraphIrToCntkGraph(graphIR::Graph &graphIrPtr, CNTK::FunctionPtr modelFuncPtr)
{
    return nullptr;
}

FunctionPtr FpgaFunctionFactory(
    const std::wstring& op,
    std::vector<Variable>& inputs,
    Dictionary&& functionConfig,
    const std::wstring& functionName,
    const std::wstring& uid)
{
    fprintf(stderr, "Inspecting %-32S%S\n", uid.c_str(), functionName.c_str());

    if (op == L"Times")
    {
        fprintf(stderr, "    OVERRIDING as fpga node.\n");

        auto functionConfigCopy = functionConfig;
        auto interceptTarget = std::make_shared<FpgaFunction>(inputs, std::move(functionConfigCopy), functionName, uid);

        return UserDefinedFuntion(
                inputs,
                std::move(functionConfig),
                functionName,
                uid,
                interceptTarget);
    }

    return nullptr;
}


bool GetVariableByName(std::vector<Variable> variableLists, std::wstring varName, Variable& var)
{
    for (auto it = variableLists.begin(); it != variableLists.end(); ++it)
    {
        if (it->Name().compare(varName) == 0)
        {
            var = *it;
            return true;
        }
    }
    return false;
}

inline bool GetInputVariableByName(FunctionPtr evalFunc, std::wstring varName, Variable& var)
{
    return GetVariableByName(evalFunc->Arguments(), varName, var);
}

inline bool GetOutputVaraiableByName(FunctionPtr evalFunc, std::wstring varName, Variable& var)
{
    return GetVariableByName(evalFunc->Outputs(), varName, var);
}

void RunEvaluationClassifier(FunctionPtr evalFunc, const DeviceDescriptor& device)
{
    std::vector<std::wstring> inputNodeNames = { L"rawAnswer", L"rawContext", L"rawQuery"/*, L"contextSeqAxis", L"sourceSeqAxis"*/ };

    std::vector<Variable> inputVars;
    for (auto inputNodeName : inputNodeNames)
    {
        Variable inputVar;

        if (!GetInputVariableByName(evalFunc, inputNodeName, inputVar))
        {
            fprintf(stderr, "Input variable %S is not available.\n", inputNodeName.c_str());
            throw("Input variable not found error.");
        }

        inputVars.push_back(inputVar);
    }

    // Evaluate the network in several runs 
    size_t numSamples = 3;
    size_t iterationCount = 4;
    unsigned int randSeed = 2;
    srand(randSeed);
    for (size_t t = 0; t < iterationCount; ++t)
    {
        printf("\n\n\n");

        std::unordered_map<Variable, ValuePtr> arguments;

        for (auto inputVar : inputVars)
        {
            std::vector<float> inputData(inputVar.Shape().TotalSize() * numSamples);

            for (size_t i = 0; i < inputData.size(); ++i)
            {
                inputData[i] = ((float)rand()) / RAND_MAX;
            }

            // Create input data shape. Adding sequence length and numSamples as axes.
            // Todo: remove sequence length when only numSamples is supported.
            // Todo: add convenience APIs to simplify data preparation here.
            NDShape inputShape = inputVar.Shape().AppendShape({ 1, numSamples });
            ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData, true));

            arguments[inputVar] = inputValue;
        }

        // Define output.
        std::unordered_map<Variable, ValuePtr> outputs;

        for (auto ov : evalFunc->Outputs())
        {
            ValuePtr outputValue;
            Variable outputVar;

            outputVar = ov;
            outputs[ov] = outputValue;
        }

        // Evaluate the model
        evalFunc->Forward(arguments, outputs, device);

        ////for (auto outputTuple : outputs)
        ////{
        ////    // Get output value
        ////    auto outputVar = outputTuple.first;
        ////    auto outputValue = outputTuple.second;

        ////    // Todo: remove sequence length when only numSamples is supported.
        ////    // Todo: add convenience APIs to simplify retrieval of output results.
        ////    NDShape outputShape = outputVar.Shape().AppendShape({ 1, numSamples });
        ////    std::vector<float> outputData(outputShape.TotalSize());
        ////    NDArrayViewPtr cpuArrayOutput = MakeSharedObject<NDArrayView>(outputShape, outputData, false);
        ////    cpuArrayOutput->CopyFrom(*outputValue->Data());

        ////    assert(outputData.size() == outputVar.Shape()[0] * numSamples);
        ////    fprintf(stderr, "Evaluation result:\n");
        ////    size_t dataIndex = 0;
        ////    auto outputDim = outputVar.Shape()[0];
        ////    for (size_t i = 0; i < numSamples; i++)
        ////    {
        ////        fprintf(stderr, "Iteration:%lu, Sample %lu:\n", t, i);
        ////        fprintf(stderr, "    ");
        ////        dataIndex = i * outputDim;
        ////        for (size_t j = 0; j < std::min((size_t)10, outputDim); j++)
        ////        {
        ////            fprintf(stderr, "%f ", outputData[dataIndex++]);
        ////        }
        ////        if (outputDim > 10)
        ////        {
        ////            fprintf(stderr, "...");
        ////        }
        ////        fprintf(stderr, "\n");
        ////    }
        ////}
    }
}



void MultiThreadsEvaluation(bool isGPUAvailable)
{
    auto device = DeviceDescriptor::CPUDevice();

    // The model file will be trained and copied to the current runtime directory first.
    auto modelFuncPtr = CNTK::Function::LoadModel(DataType::Float, L"\\CNTK\\Tests\\UnitTests\\CntpGraphIrC\\BingModelRoot\\Out\\proto2.dnn", device, FpgaFunctionFactory);

    // convert cntk to graphir
    auto graphIrPtr = CntkGraphToGraphIr(modelFuncPtr, device);

    RunEvaluationClassifier(modelFuncPtr, device);

    // convert graphir back to cntk (with the original cntk model as template)
    auto modelImportFuncPtr = GraphIrToCntkGraph(graphIrPtr, modelFuncPtr);

    // TODO: verify that roundtrip is completed.

    fflush(stderr);
}
