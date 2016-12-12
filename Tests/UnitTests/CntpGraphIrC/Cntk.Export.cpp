//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GraphIrExporter.h"
#include <functional>
//#include "LSTM/LstmGraphNode.h"

#include <algorithm>

using namespace CNTK;
using namespace std;

#include <fstream>


static void Usage()
{
    fprintf(stderr, "usage: Cntk.Export command [files]* where commands can be one of:\n");
    fprintf(stderr, "  export [cntk-model]* - Exports cntk model(s) into GraphIr\n");
    fprintf(stderr, "  import [cntk-model]* - Imports GraphIr model(s) into cntk\n");
    fprintf(stderr, "  evaluate cntk-model graphir-model - Compares the output of original with the exported model.\n");
}

int main(int argc, char *argv[])
{
    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

    auto device = DeviceDescriptor::CPUDevice();

    if (argc < 2)
    {
        Usage();
        return 1;
    }

    std::string command = argv[1];
    std::transform(command.begin(), command.end(), command.begin(), ::tolower);

    if (command == "export")
    {
        for (int n = 2; n < argc; n++)
        {
            printf("Exporting CNTK %s\n", argv[n]);

            std::string filename = argv[n];
            std::wstring graphIrFilename = TransformCntkToGraphIr(filename, device);

            printf("  => %S\n", graphIrFilename.c_str());
        }
    }
    else if (command == "import")
    {
        for (int n = 2; n < argc; n++)
        {
            printf("Importing GraphIr %s\n", argv[n]);

            std::string filename = argv[n];
            std::wstring cntkFilename = TransformGraphIrToCntk(filename, device);

            printf("  => %S\n", cntkFilename.c_str());
        }
    }
    else if (command == "evaluate")
    {
        if (argc != 4)
        {
            Usage();
            return 2;
        }

        // first, evaluate the  original model
        // note: since the inputs is empty, they will be retieved from the
        //       model and filled with random data.
        std::string cntkFilename = argv[2];
        std::unordered_map<std::wstring, std::vector<float>> inputs;
        std::unordered_map<std::wstring, std::vector<float>> outputs1;
        ExecuteModelOnRandomData(cntkFilename, inputs, outputs1, device);

        for (auto& outputTuple : outputs1)
        {
            // tell the user what we received.
            fprintf(stderr, "FILE1 Output %S #%lu elements.\n", outputTuple.first.c_str(), (unsigned long)outputTuple.second.size());
        }

        // now, evaluate the re-imported model on the same input data.
        auto tempFile = TransformGraphIrToCntk(argv[3], device);
        fprintf(stderr, "Importing graphir back to cntk: %S\n", tempFile.c_str());

        std::string cntkReImportedFilename = std::string(tempFile.begin(), tempFile.end());
        std::unordered_map<std::wstring, std::vector<float>> outputs2;
        ExecuteModelOnRandomData(cntkReImportedFilename, inputs, outputs2, device);

        for (auto& outputTuple : outputs2)
        {
            // tell the user what we received.
            fprintf(stderr, "FILE2 Output %S #%lu elements.\n", outputTuple.first.c_str(), (unsigned long)outputTuple.second.size());
        }

        // Verify that results match
        // TODO: verify that inputs match (only indirect test done above)
        assert(outputs2.size() == outputs1.size());
        for (auto& n1 : outputs1)
        {
            auto n2 = *outputs2.find(n1.first);
            assert(outputs2.find(n1.first) != outputs2.end());

            assert(n1.second.size() == n2.second.size());
            assert(memcmp(&n1.second[0], &n2.second[0], n1.second.size() * sizeof(float)) == 0);
        }

        fprintf(stderr, "Export/Import completed, both functions executed, results match.\n");
    }
    else
    {
        Usage();
        return 2;
    }

    fflush(stderr);
}
