//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GraphIrExporter.h"
#include <functional>
#include "LSTM/LstmGraphNode.h"

#ifndef CPUONLY
#error "must use CPU Only"
#endif

using namespace CNTK;
using namespace std;

int main()
{
	auto device = DeviceDescriptor::CPUDevice();
	auto filename = std::string("\\BrainWaveCntk\\Tests\\UnitTests\\CntpGraphIrC\\BingModelRoot\\Out\\proto2.dnn");
    auto filenameW = wstring(filename.begin(), filename.end());

	// The model file will be trained and copied to the current runtime directory first.
	auto modelFuncPtr = CNTK::Function::LoadModel(filenameW, device);

    // json dump does not contain entire raw array data
    // because the output would be too big.
    MAX_BASE64_EXPORT_LENGTH = 100;
    auto message = GRAPHIR::Serialize(modelFuncPtr);
    DumpAsJson(*message, filename + string(".serialized_json"));

    // re-serialize with entire array buffers to dump
    // in binary format. this includes full array data.
    MAX_BASE64_EXPORT_LENGTH = INT_MAX;
    auto message2 = GRAPHIR::Serialize(modelFuncPtr);
    DumpAsBinary(*message2, filename + string(".serialized"));

    //
    // Execute the original function
    //

    unordered_map<wstring, vector<float>> inputs;
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

    unordered_map<wstring, vector<float>> outputs1;
    ExecuteModel(modelFuncPtr, inputs, outputs1);

    for (auto outputTuple : outputs1)
    {
        // tell the user what we received.
        fprintf(stderr, "ORIG Output %S #%lu elements.\n", outputTuple.first.c_str(), (unsigned long)outputTuple.second.size());
    }

    //
    // Execute the original function
    //

    // note: must use the binary serialization since json
    // does not contain full array data.
    auto importedFunction = GRAPHIR::Deserialize(message2);

    unordered_map<wstring, vector<float>> outputs2;
    ExecuteModel(importedFunction, inputs, outputs2);

    for (auto outputTuple : outputs2)
    {
        // tell the user what we received.
        fprintf(stderr, "IMPT Output %S #%lu elements.\n", outputTuple.first.c_str(), (unsigned long)outputTuple.second.size());
    }

	// TODO: verify that roundtrip is completed.
    assert(outputs2.size() == outputs1.size());
    for (auto n1 : outputs1)
    {
        auto n2 = *outputs2.find(n1.first);
        assert(outputs2.find(n1.first) != outputs2.end());

        assert(n1.second.size() == n2.second.size());
        assert(memcmp(&n1.second[0], &n2.second[0], n1.second.size() * sizeof(float)) == 0);
    }

    fprintf(stderr, "Export/Import completed, both functions executed, results match.\n");
    fflush(stderr);
}
