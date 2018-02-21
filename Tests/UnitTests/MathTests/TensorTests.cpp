//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
//
#include "stdafx.h"
#include "TensorView.h"
#include "Sequences.h"
#include "TensorTestsHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(MathTensorTests)

BOOST_AUTO_TEST_CASE(ElementwiseAddition)
{
    Test::TensorTest<float> tensorTester;

    // --- elementwise

    // elementwise sum
    tensorTester.OneTensorTest("elementwise addition", 1e-8, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BroadcastingTest(TensorShape{ 512, 256 }, TensorShape{ 512, 256 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(AdditionWithSimpleBroadcasting)
{
    Test::TensorTest<float> tensorTester;

    // --- broadcasting

    // simple broadcasting
    tensorTester.OneTensorTest("addition wth simple broadcasting", 1e-8, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BroadcastingTest(TensorShape{ 3, 2 }, TensorShape{ 3, 1 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(BiasAddition)
{
    Test::TensorTest<float> tensorTester;

    // typical bias for convolutional layer
    tensorTester.OneTensorTest("bias addition (broadcasting)", 1e-8, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BroadcastingTest(TensorShape{ 28, 28, 128, 32 }, TensorShape{ 1, 1, 128 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(BiasAddition2)
{
    Test::TensorTest<float> tensorTester;
    // BUGBUG: This test is strange--Print() shows different values with depth 128 instead of 64, but IsEqual() does not fail with 1e-3 tolerance.
    //         Something fishy going on. Dimension overflow?
    tensorTester.OneTensorTest("bias addition (broadcasting)", 1e-8, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BroadcastingTest(TensorShape{ 256, 256, 64, 32 }, TensorShape{ 1, 1, 64 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(BiasGradient)
{
    Test::TensorTest<float> tensorTester;
    // --- reduction

    // typical bias gradient (reduction) for FF-DNN
    tensorTester.OneTensorTest("bias gradient (reduction)", 1e-4, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BiasGradientTest(TensorShape{ 2048, 1024 }, TensorShape(2048), deviceId);
    });
}

BOOST_AUTO_TEST_CASE(BiasGradient2)
{
    Test::TensorTest<float> tensorTester;

    // typical bias gradient (reduction) for convolutional layer
    tensorTester.OneTensorTest("bias gradient (reduction)", 1e-1, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BiasGradientTest(TensorShape{ 256, 256, 64, 32 }, TensorShape{ 1, 1, 64 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(ColumnSliceMultAndAdd)
{
    ColumnSliceMultAndAddTest<float>(2048, 2048, 256, 0);
}

BOOST_AUTO_TEST_CASE(RnnForwardProp)
{
    TestRnnForwardPropSRP<float>();
}

BOOST_AUTO_TEST_CASE(OldRnnForwardProp)
{
    TestOldRnnForwardPropSRP<float>();
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Half_MathTensorTests)

BOOST_AUTO_TEST_CASE(ElementwiseAddition)
{
    Test::TensorTest<half> tensorTester;

    // --- elementwise

    // elementwise sum
    tensorTester.OneTensorTest("elementwise addition", 1e-8, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BroadcastingTest(TensorShape{ 512, 256 }, TensorShape{ 512, 256 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(AdditionWithSimpleBroadcasting)
{
    Test::TensorTest<half> tensorTester;

    // --- broadcasting

    // simple broadcasting
    tensorTester.OneTensorTest("addition wth simple broadcasting", 1e-8, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BroadcastingTest(TensorShape{ 3, 2 }, TensorShape{ 3, 1 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(BiasAddition)
{
    Test::TensorTest<half> tensorTester;

    // typical bias for convolutional layer
    tensorTester.OneTensorTest("bias addition (broadcasting)", 1e-8, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BroadcastingTest(TensorShape{ 28, 28, 128, 32 }, TensorShape{ 1, 1, 128 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(BiasAddition2)
{
    Test::TensorTest<half> tensorTester;
    // BUGBUG: This test is strange--Print() shows different values with depth 128 instead of 64, but IsEqual() does not fail with 1e-3 tolerance.
    //         Something fishy going on. Dimension overflow?
    tensorTester.OneTensorTest("bias addition (broadcasting)", 1e-8, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BroadcastingTest(TensorShape{ 256, 256, 64, 32 }, TensorShape{ 1, 1, 64 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(BiasGradient)
{
    Test::TensorTest<half> tensorTester;
    // --- reduction

    // typical bias gradient (reduction) for FF-DNN
    tensorTester.OneTensorTest("bias gradient (reduction)", 0.2f, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BiasGradientTest(TensorShape{ 2048, 1024 }, TensorShape(2048), deviceId);
    });
}

BOOST_AUTO_TEST_CASE(BiasGradient2)
{
    Test::TensorTest<half> tensorTester;

    // typical bias gradient (reduction) for convolutional layer
    tensorTester.OneTensorTest("bias gradient (reduction)", 1.0f, [&tensorTester](DEVICEID_TYPE deviceId)
    {
        return tensorTester.BiasGradientTest(TensorShape{ 256, 256, 64, 32 }, TensorShape{ 1, 1, 64 }, deviceId);
    });
}

BOOST_AUTO_TEST_CASE(ColumnSliceMultAndAdd)
{
    ColumnSliceMultAndAddTest<half>(2048, 2048, 256, 0);
}

BOOST_AUTO_TEST_CASE(RnnForwardProp)
{
    TestRnnForwardPropSRP<half>();
}

BOOST_AUTO_TEST_CASE(OldRnnForwardProp)
{
    TestOldRnnForwardPropSRP<half>();
}

BOOST_AUTO_TEST_SUITE_END()

}}}}
