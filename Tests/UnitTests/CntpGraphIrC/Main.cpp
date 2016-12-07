//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GraphIrExporter.h"
#include <functional>
#include "LSTM/LstmGraphNode.h"


#include <fstream>

#ifndef CPUONLY
#error "must use CPU Only"
#endif

using namespace CNTK;
using namespace std;

void DumpAsJson(const google::protobuf::Message& message, const std::string& filename)
{
    // save it out to disk in json format.
    string jsonstring;
    auto serialized = google::protobuf::util::MessageToJsonString(message, &jsonstring);
    auto fp = fopen((filename + string(".pb.json")).c_str(), "w+");
    auto written = fwrite(jsonstring.c_str(), sizeof(char), jsonstring.length(), fp);

    assert(written == jsonstring.length());
    fclose(fp);
}

void DumpAsBinary(const google::protobuf::Message& message, const std::string& filename)
{
    auto filePath = (filename + string(".pb"));

    std::fstream fs;
    fs.open(filePath, std::fstream::in | std::fstream::out | std::fstream::trunc);

    message.SerializeToOstream(&fs);

    fs.close();
}

int main()
{
	auto device = DeviceDescriptor::CPUDevice();
	string filename = "\\Cntk\\Tests\\UnitTests\\CntpGraphIrC\\BingModelRoot\\Out\\proto2.dnn";
    wstring filenameW = wstring(filename.begin(), filename.end());

	// The model file will be trained and copied to the current runtime directory first.
	auto modelFuncPtr = CNTK::Function::LoadModel(filenameW, device/*, LstmGraphNodeFactory*/);

    auto message = GRAPHIR::Serialize(modelFuncPtr);
    auto result = GRAPHIR::Deserialize(message, modelFuncPtr);

    DumpAsBinary(*message, filename + string(".serialized"));
    DumpAsJson(*message, filename + string(".serialized_json"));

    //PrintDictionaryValue(L"ROOT", serializedFunc, 0);

    unordered_map<wstring, vector<float>> inputs;
    unordered_map<wstring, vector<float>> outputs;
    RetrieveInputBuffers(modelFuncPtr, inputs);

    for (auto inputTuple : inputs)
    {
        auto& inputData = inputTuple.second;

        // add some random data to the input vector
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            inputData[i] = ((float)rand()) / RAND_MAX;
        }

        fprintf(stderr, "Input  %S #%lu elements.\n", inputTuple.first.c_str(), (unsigned long)inputTuple.second.size());
    }

    ExecuteModel(modelFuncPtr, inputs, outputs);

    for (auto outputTuple : outputs)
    {
        // tell the user what we received.
        fprintf(stderr, "Output %S #%lu elements.\n", outputTuple.first.c_str(), (unsigned long)outputTuple.second.size());
    }

	// convert cntk to graphir
	auto graphIrPtr = CntkGraphToGraphIr(filenameW, modelFuncPtr);

	// save it out to disk in json format.
    DumpAsJson(*graphIrPtr, filename + string(".pb.json"));

	// convert graphir back to cntk (with the original cntk model as template)
	auto modelImportFuncPtr = GraphIrToCntkGraph(graphIrPtr, modelFuncPtr);

	// TODO: verify that roundtrip is completed.
    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
