//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "EvaluationNodes.h"

using namespace Microsoft::MSR::CNTK;
namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(EditDistanceTests)

BOOST_AUTO_TEST_CASE(ComputeEditDistanceErrorTest)
{
    Matrix<float> firstSeq(CPUDEVICE);
    Matrix<float> secondSeq(CPUDEVICE);
    vector<size_t> samplesToIgnore;
    size_t seqSize = 10;
    firstSeq.Resize(1, seqSize);
    secondSeq.Resize(1, seqSize);
    for (size_t i = 0; i < seqSize; i++)
    {
        firstSeq(0, i) = (float)i;
        secondSeq(0, i) = (float)i - 1;
    }
    MBLayoutPtr pMBLayout = make_shared<MBLayout>(1, seqSize, L"X");
    pMBLayout->AddSequence(0, 0, 0, seqSize);

    float ed = EditDistanceErrorNode<float>::ComputeEditDistanceError(firstSeq, secondSeq, pMBLayout, 1, 1, 1, true, samplesToIgnore);
    assert((int)ed == 2);

    for (size_t i = 0; i < seqSize; i++)
    {
        secondSeq(0, i) = (float)i;
    }

    ed = EditDistanceErrorNode<float>::ComputeEditDistanceError(firstSeq, secondSeq, pMBLayout, 1, 1, 1, true, samplesToIgnore);
    assert((int)ed == 0);

    secondSeq(0, seqSize-1) = (float)123;

    ed = EditDistanceErrorNode<float>::ComputeEditDistanceError(firstSeq, secondSeq, pMBLayout, 1, 1, 1, true, samplesToIgnore);
    assert((int)ed == 1);
}

BOOST_AUTO_TEST_SUITE_END()

} } } }