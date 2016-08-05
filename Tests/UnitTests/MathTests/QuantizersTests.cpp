//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../../../Source/Math/Quantizers.h"
#include "../../../Source/Math/Helpers.h"

using namespace Microsoft::MSR::CNTK;
namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(QuantizersUnitTests)

BOOST_FIXTURE_TEST_CASE(FloatToShort, RandomSeedFixture)
{
    float input[3] = { -10.0f, 0, 5.0f };
    short output[3] = { 0, 0, 0 };
    
    float* inputCorrect = new float[3];
    for (size_t i = 0; i < 3; i++)
        inputCorrect[i] = input[i];

    short outputCorrect[3] = { -32767, 0, 16384 };

    ArrayRef<float> inputAr(input, 3);
    ArrayRef<short> outputAr(output, 3);

    std::unique_ptr<QuantizerBase<float, short>> symQuantPtr(new SymmetricQuantizer<float, short>(10.0f, 0));
    symQuantPtr->Quantize(inputAr, outputAr);
    for (size_t i = 0; i < 3; i++) 
        BOOST_CHECK_EQUAL(output[i], outputCorrect[i]);

    symQuantPtr->Dequantize(outputAr, inputAr);
    for (size_t i = 0; i < 3; i++)
        BOOST_CHECK_EQUAL(round(input[i] * (10^4)), round(inputCorrect[i] * (10^4)));

    std::unique_ptr<QuantizerBase<float, short>> symQuantPtr2(new SymmetricQuantizer<float, short>(inputAr, 0));
    symQuantPtr2->Quantize(inputAr, outputAr);
    for (size_t i = 0; i < 3; i++)
        BOOST_CHECK_EQUAL(output[i], outputCorrect[i]);

    symQuantPtr2->Dequantize(outputAr, inputAr);
    for (size_t i = 0; i < 3; i++)
        BOOST_CHECK_EQUAL(round(input[i] * (10 ^ 4)), round(inputCorrect[i] * (10 ^ 4)));

    delete[] inputCorrect;

}


BOOST_AUTO_TEST_SUITE_END()

} } } }