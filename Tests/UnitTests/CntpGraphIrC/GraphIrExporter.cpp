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

#pragma warning(push, 0)
#include <graphid.pb.h>
#pragma warning(pop)

extern "C"
{
#include <b64/cencode.h>
}

using namespace CNTK;

#define MAX_BASE64_EXPORT_LENGTH 100

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

static std::string EncodeBase64(const char *buf, size_t len)
{
	base64_encodestate state;
	char *sout = new char[len * 2];
	memset(sout, 0, len * 2);

	base64_init_encodestate(&state);
	base64_encode_block(buf, (int)len, sout, &state);
	base64_encode_blockend(sout, &state);

	// TODO: remove once we export everything.
    static_assert(MAX_BASE64_EXPORT_LENGTH > 10, "base64 export should at least be 10");
    if (strlen(sout) > MAX_BASE64_EXPORT_LENGTH)
	{
		strcpy_s(sout + MAX_BASE64_EXPORT_LENGTH - 8, len * 2 - MAX_BASE64_EXPORT_LENGTH, "...");
	}

	auto result = std::string(sout);

	delete[] sout;

	return result;
}

graphIR::Graph* CntkGraphToGraphIr(std::wstring filename, FunctionPtr evalFunc)
{
	auto graphInfo = new graphIR::GraphInfo();
    graphInfo->set_framework_name("CNTK");
    graphInfo->set_framework_version("2.0beta3.0"); // TODO: call cntk function to retrieve version string
    graphInfo->set_graph_version("0.1");
    graphInfo->set_description("Exported by the Graph Ir Exporter from CNTK");
    graphInfo->set_model_name(std::string(filename.begin(), filename.end()));

	auto graph = new graphIR::Graph();
	graph->set_allocated_graph_info(graphInfo);

    // get the data describing the graph
    auto graphDict = evalFunc->Serialize();

    // assume we get a root-directory, so retrieve the primitive functions.
    for (auto funct : graphDict[L"primitive_functions"].Value<std::vector<DictionaryValue>>())
    {
        printf("function: %S\n", funct.Value<Dictionary>()[L"uid"].Value<std::wstring>().c_str());
    }

	std::unordered_set<FunctionPtr> functions;
	Traverse(evalFunc->RootFunction(), functions,
		[&graph](const FunctionPtr& f)
	{
		fprintf(stderr, "now at %S opcode %S\n", f->Uid().c_str(), f->OpName().c_str());

		graphIR::Node *node = graph->add_nodes();

		node->set_name(ConstructUniqueName(f->Uid(), f->Name()));

		auto name = f->OpName();
		node->set_op(std::string(name.begin(), name.end()));

		auto d = f->Attributes();
		std::stringstream strstr(std::ios_base::in | std::ios_base::out | std::ios_base::binary);
		strstr << d;
		auto where = strstr.tellp();
		auto str = strstr.str();

		graphIR::InitArg initArg;
		initArg.set_dbytes(4); // fp32 is 4 bytes per entry

		(*node->mutable_ext_attrs())["##CNTK##NODE##"] = EncodeBase64(str.c_str(), str.length());

		for (auto out : f->Placeholders())
		{
			fprintf(stderr, "Placeholder not expected: %S\n", out.Name().c_str());
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
				fprintf(stderr, "%lu", (unsigned long)dims);
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

			graphIR::InitArg initArg2;
			initArg2.set_dbytes(4); // fp32 is 4 bytes per entry
			initArg2.set_data_base64(EncodeBase64((char*)buf, rank * 4));

			(*node->mutable_init_attrs())[ConstructUniqueName(out.Uid(), out.Name())] = initArg2;

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

			graphIR::InitArg initArg3;
			initArg3.set_dbytes(4); // fp32 is 4 bytes per entry
			initArg3.set_data_base64(EncodeBase64((char *)buf, rank * 4));

			(*node->mutable_init_attrs())[ConstructUniqueName(out.Uid(), out.Name())] = initArg3;

			fprintf(stderr, "    == %S type %d value %f\n", out.Name().c_str(), out.GetDataType(), buf[0]);
		}

		for (auto iter = f->Attributes().begin(); iter != f->Attributes().end(); iter++)
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
					fprintf(stderr, "%lu", (unsigned long)dims);
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

CNTK::FunctionPtr GraphIrToCntkGraph(graphIR::Graph* /*graphIrPtr*/, CNTK::FunctionPtr /*modelFuncPtr*/)
{
    return nullptr;
}


std::wstring PrintNDArrayView(const NDArrayView& value)
{
    auto result = std::wstring(L"type=") + std::to_wstring(static_cast<unsigned int>(value.GetDataType()));

    result += L", shape=[";
    auto size = value.Shape().Rank();
    for (auto i = 0; i < size; i++)
    {
        if (i) result += L",";
        result += std::to_wstring(value.Shape()[i]);
    }
    result += L"]";

    result += std::wstring(L", storageFromat=") + std::to_wstring(static_cast<unsigned int>(value.GetStorageFormat()));

    size = value.Shape().TotalSize();
    if (value.GetDataType() == DataType::Float)
    {
        const float* buffer32 = value.DataBuffer<float>();
        //memcpy(TO_BASE64, buffer, (int)size * sizeof(float));

        result += L", datasize=" + std::to_wstring(size) + L"floats";
    }
    else if (value.GetDataType() == DataType::Double)
    {
        const double* buffer64 = value.DataBuffer<double>();
        //memcpy(TO_BASE64, buffer64, (int)size * sizeof(double));

        result += L", datasize=" + std::to_wstring(size) + L"doubles";
    }

    return result;
}

namespace GRAPHIR
{
    std::ostream& operator<<(std::ostream& stream, const Dictionary& dictionary);
}


void PrintDictionaryValue(const std::wstring& name, const DictionaryValue& value, int indent)
{
    std::wstring result;

    switch (value.ValueType())
    {
    case DictionaryValue::Type::Bool:
        result = std::to_wstring(value.Value<bool>());
        break;
    case DictionaryValue::Type::Int:
        result = std::to_wstring(value.Value<int>());
        break;
    case DictionaryValue::Type::SizeT:
        result = std::to_wstring(value.Value<size_t>());
        break;
    case DictionaryValue::Type::Float:
        result = std::to_wstring(value.Value<float>());
        break;
    case DictionaryValue::Type::Double:
        result = std::to_wstring(value.Value<double>());
        break;
    case DictionaryValue::Type::String:
        result = value.Value<std::wstring>();
        break;
    case DictionaryValue::Type::NDShape:
        result = L"value.Value<NDShape>()";
        break;
    case DictionaryValue::Type::Axis:
        result = L"staticAxisIndex=" + std::to_wstring(value.Value<Axis>().StaticAxisIndex(false)) +
            L", name=\"" + value.Value<Axis>().Name() + L"\"" +
            L", isOrdered=" + std::to_wstring(value.Value<Axis>().IsOrdered());
        break;
    case  DictionaryValue::Type::NDArrayView:
        result = PrintNDArrayView(value.Value<NDArrayView>());
        break;
    case DictionaryValue::Type::Vector:
        result = L"value.Value<std::vector<DictionaryValue>>()";
        break;
    case DictionaryValue::Type::Dictionary:
        result = L"value.Value<Dictionary>()";
        break;
    case DictionaryValue::Type::None:
    default:
        result = L"<<not defind>>";
        break;
    }

    for (size_t n = 0; n < indent; n++)
        printf(" ");
    (name.size() > 0) ? printf("%S = %S\n", name.c_str(), result.c_str()) : printf("%S\n", result.c_str());

    switch (value.ValueType())
    {
    case DictionaryValue::Type::Vector:
        for (auto subNode : value.Value<std::vector<DictionaryValue>>())
        {
            PrintDictionaryValue(L"", subNode, indent + 2);
        }
        break;

    case DictionaryValue::Type::Dictionary:
        std::stringstream s;
        GRAPHIR::operator<<(s, value.Value<Dictionary>());


        for (auto subNode : value.Value<Dictionary>())
        {
            if ((subNode.second.ValueType() != DictionaryValue::Type::Vector) &&
                (subNode.second.ValueType() != DictionaryValue::Type::Dictionary))
            {
                PrintDictionaryValue(subNode.first, subNode.second, indent + 2);
            }
        }
        for (auto subNode : value.Value<Dictionary>())
        {
            if (subNode.second.ValueType() == DictionaryValue::Type::Vector)
            {
                PrintDictionaryValue(subNode.first, subNode.second, indent + 2);
            }
        }
        for (auto subNode : value.Value<Dictionary>())
        {
            if (subNode.second.ValueType() == DictionaryValue::Type::Dictionary)
            {
                PrintDictionaryValue(subNode.first, subNode.second, indent + 2);
            }
        }
        break;
    }
}

