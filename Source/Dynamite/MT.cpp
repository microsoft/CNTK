//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "PlainTextDeseralizer.h"
//#include "Layers.h"
//#include "Common.h"
//#include "TimerUtility.h"

#include <cstdio>
#include <map>
#include <set>
#include <vector>

#define let const auto

using namespace CNTK;
using namespace std;

//using namespace Dynamite;

void Train(const DeviceDescriptor& device, bool useSparseLabels)
{
    let srcVocabSize = 2330;
    let tgtVocabSize = 2330;
    //const size_t inputDim = 2000;
    //const size_t embeddingDim = 500;
    //const size_t hiddenDim = 250;
    //const size_t attentionDim = 20;
    //const size_t numOutputClasses = 5;
    //
    //const wstring trainingCTFPath = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf";
    //
    //// static model and criterion function
    //auto model_fn = CreateModelFunction(numOutputClasses, embeddingDim, hiddenDim, device);
    //auto criterion_fn = CreateCriterionFunction(model_fn);
    //
    //// dynamic model and criterion function
    //auto d_model_fn = CreateModelFunctionUnrolled(numOutputClasses, embeddingDim, hiddenDim, device);
    //auto d_criterion_fn = CreateCriterionFunctionUnrolled(d_model_fn);

    // data
    let minibatchSource = CreateCompositeMinibatchSource(MinibatchSourceConfig({ PlainTextDeserializer(
        {
            PlainTextStreamConfiguration(L"src", srcVocabSize, { L"d:/work/Karnak/sample-model/data/train.src" }, { L"d:/work/Karnak/sample-model/data/vocab.src", L"<s>", L"</s>", L"<unk>" }),
            PlainTextStreamConfiguration(L"tgt", tgtVocabSize, { L"d:/work/Karnak/sample-model/data/train.tgt" }, { L"d:/work/Karnak/sample-model/data/vocab.tgt", L"<s>", L"</s>", L"<unk>" })
        })},
        /*randomize=*/false/*for now*/));
}

int mt_main(int argc, char *argv[])
{
    argc; argv;
    try
    {
        Train(DeviceDescriptor::GPUDevice(0), true);
        //Train(DeviceDescriptor::CPUDevice(), true);
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
