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
#include "LSTM/LstmGraphNode.h"

using namespace CNTK;

#pragma warning(push, 0)
#include <graphid.pb.h>
#include <google/protobuf/util/json_util.h>
#pragma warning(pop)

extern "C"
{
#include <b64/cencode.h>
}


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

graphIR::Graph CntkGraphToGraphIr(FunctionPtr evalFunc)
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

		{
			base64_encodestate state;
			base64_init_encodestate(&state);

			char *sout = new char[str.length() * 2];
			memset(sout, 0, str.length() * 2);
			base64_encode_block((const char *)str.c_str(), (int)str.length(), sout, &state);
			base64_encode_blockend(sout, &state);

			if (strlen(sout) > 100)
			{
				strcpy_s(sout + 90, str.length() * 2 - 100, "...");
			}

			(*node->mutable_ext_attrs())["##CNTK##NODE##"] = sout;

			delete[] sout;
		}

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
                input->add_shape((int)dims);

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
            base64_encode_block((char *)buf, (int)(rank * 4), sout, &state);
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
            base64_encode_block((const char *)buf, (int)(rank * 4), sout, &state);
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
                    output->add_shape((int)dims);

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

CNTK::FunctionPtr GraphIrToCntkGraph(graphIR::Graph &/*graphIrPtr*/, CNTK::FunctionPtr /*modelFuncPtr*/)
{
    return nullptr;
}
