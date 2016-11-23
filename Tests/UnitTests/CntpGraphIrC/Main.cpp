//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"
#include <functional>
#include "LSTM/LstmGraphNode.h"

using namespace CNTK;

#pragma warning(push, 0)
#include <graphid.pb.h>
#include <google/protobuf/util/json_util.h>
#pragma warning(pop)

using namespace std;

extern FunctionPtr GraphIrToCntkGraph(graphIR::Graph &/*graphIrPtr*/, FunctionPtr /*modelFuncPtr*/);
extern graphIR::Graph CntkGraphToGraphIr(FunctionPtr evalFunc);
extern void EvaluateGraph(FunctionPtr evalFunc, const DeviceDescriptor& device);

int main()
{
	auto device = DeviceDescriptor::CPUDevice();

#ifndef CPUONLY
#error "must use CPU Only"
#else
    fprintf(stderr, "Run tests using CPU-only build.\n");
#endif

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

	// The model file will be trained and copied to the current runtime directory first.
	auto modelFuncPtr = CNTK::Function::LoadModel(DataType::Float, L"\\CNTK\\Tests\\UnitTests\\CntpGraphIrC\\BingModelRoot\\Out\\proto2.dnn", device, LstmGraphNodeFactory);

	EvaluateGraph(modelFuncPtr, device);

	// convert cntk to graphir
	auto graphIrPtr = CntkGraphToGraphIr(modelFuncPtr);
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
