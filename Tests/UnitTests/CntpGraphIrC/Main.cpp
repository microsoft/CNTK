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

#ifndef CPUONLY
#error "must use CPU Only"
#endif

using namespace CNTK;
using namespace std;


extern FunctionPtr GraphIrToCntkGraph(graphIR::Graph &/*graphIrPtr*/, FunctionPtr /*modelFuncPtr*/);
extern graphIR::Graph CntkGraphToGraphIr(std::wstring filename, FunctionPtr evalFunc);
extern void EvaluateGraph(FunctionPtr evalFunc, const DeviceDescriptor& device);

int main()
{
	auto device = DeviceDescriptor::CPUDevice();
	std::string filename = "\\CNTK\\Tests\\UnitTests\\CntpGraphIrC\\BingModelRoot\\Out\\proto2.dnn";

	// The model file will be trained and copied to the current runtime directory first.
	auto modelFuncPtr = CNTK::Function::LoadModel(DataType::Float, std::wstring(filename.begin(), filename.end()), device, LstmGraphNodeFactory);

	EvaluateGraph(modelFuncPtr, device);

	// convert cntk to graphir
	auto graphIrPtr = CntkGraphToGraphIr(std::wstring(filename.begin(), filename.end()), modelFuncPtr);

	// save it out to disk in json format.
	std::string jsonstring;
	auto serialized = google::protobuf::util::MessageToJsonString(graphIrPtr, &jsonstring);
	auto fp = fopen((filename + std::string(".pb.json")).c_str(), "w+");
	auto written = fwrite(jsonstring.c_str(), sizeof(char), jsonstring.length(), fp);
	fclose(fp);

	// convert graphir back to cntk (with the original cntk model as template)
	auto modelImportFuncPtr = GraphIrToCntkGraph(graphIrPtr, modelFuncPtr);

	// TODO: verify that roundtrip is completed.
    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
