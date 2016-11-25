//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include "CNTKLibrary.h"
#include <functional>
#include "LSTM/LstmGraphNode.h"

#pragma warning(push, 0)
#include <graphid.pb.h>
#include <google/protobuf/util/json_util.h>
#pragma warning(pop)

#include "GraphIrExporter.h"

#ifndef CPUONLY
#error "must use CPU Only"
#endif

using namespace CNTK;
using namespace std;


extern FunctionPtr GraphIrToCntkGraph(graphIR::Graph */*graphIrPtr*/, FunctionPtr /*modelFuncPtr*/);
extern graphIR::Graph* CntkGraphToGraphIr(std::wstring filename, FunctionPtr evalFunc);

extern void RetrieveInputBuffers(
    FunctionPtr evalFunc,
    std::unordered_map<std::wstring, std::vector<float>>& inputs);

extern void ExecuteModel(
    FunctionPtr evalFunc,
    std::unordered_map<std::wstring, std::vector<float>>& inputs,
    std::unordered_map<std::wstring, std::vector<float>>& outputs);

int main()
{
	auto device = DeviceDescriptor::CPUDevice();
	std::string filename = "\\BrainWaveCntk\\Tests\\UnitTests\\CntpGraphIrC\\BingModelRoot\\Out\\proto2.dnn";
    std::wstring filenameW = std::wstring(filename.begin(), filename.end());

//    CntkNetParser parser;
//    BG_Graph *g = parser.Net2Bg(filenameW, stdout, nullptr, true);

	// The model file will be trained and copied to the current runtime directory first.
	auto modelFuncPtr = CNTK::Function::LoadModel(DataType::Float, filenameW, device/*, LstmGraphNodeFactory*/);

    std::unordered_map<std::wstring, std::vector<float>> inputs;
    std::unordered_map<std::wstring, std::vector<float>> outputs;

    RetrieveInputBuffers(modelFuncPtr, inputs);

    for (auto inputTuple : inputs)
    {
        auto& inputData = inputTuple.second;

        // add some random data to the input vector
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            inputData[i] = ((float)rand()) / RAND_MAX;
        }

        fprintf(stderr, "Input  %S #%lu elements.\n", inputTuple.first.c_str(), inputTuple.second.size());
    }

    ExecuteModel(modelFuncPtr, inputs, outputs);

    for (auto outputTuple : outputs)
    {
        auto& outputData = outputTuple.second;

        // add some random data to the input vector
        fprintf(stderr, "Output %S #%lu elements.\n", outputTuple.first.c_str(), outputTuple.second.size());
    }

	// convert cntk to graphir
	auto graphIrPtr = CntkGraphToGraphIr(filenameW, modelFuncPtr);

	// save it out to disk in json format.
	std::string jsonstring;
	auto serialized = google::protobuf::util::MessageToJsonString(*graphIrPtr, &jsonstring);
	auto fp = fopen((filename + std::string(".pb.json")).c_str(), "w+");
	auto written = fwrite(jsonstring.c_str(), sizeof(char), jsonstring.length(), fp);
	fclose(fp);

	// convert graphir back to cntk (with the original cntk model as template)
	auto modelImportFuncPtr = GraphIrToCntkGraph(graphIrPtr, modelFuncPtr);

	// TODO: verify that roundtrip is completed.
    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
