//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Performance unit tests should go here
//

#include "stdafx.h"
#include "Matrix.h"
#include "CPUMatrix.h"
#include "TensorView.h"
#include "Sequences.h"
#include <boost/test/parameterized_test.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace Microsoft::MSR::CNTK;
using namespace boost::unit_test;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(MathPerformanceTests)

template<class FN>
static void OneTensorTest(const char* what, double tolerance, const FN& fn)
{
    cout << "===== Tensor test '" << what << "'\n   ";

    // run on GPU and CPU
    let resultGPU = fn(0);
    let resultCPU = fn(-1);

    // dump top corner of the result to get a feel for the error
    resultGPU.GetSOB().Print("GPU result", 0, 7, 0, 9);
    resultGPU.GetSOB().TransferToDeviceIfNotThere(-1, true, false, true);
    resultCPU.GetSOB().Print("CPU result", 0, 7, 0, 9);

    // compare
    let isSame = resultGPU.GetSOB().IsEqualTo(resultCPU.GetSOB(), (ElemType)tolerance);
    BOOST_CHECK(isSame);
}

template<class ElemType>
static TensorView<ElemType> CreateTensor(TensorShape shape, int randomSeed, DEVICEID_TYPE deviceId, bool isResult = false)
{
    let numElements = shape.GetNumElements();

    if (isResult)
        cout << " ->";
    cout << " [" << string(shape) << "]";
    if (isResult)
        cout << " \t// " << (deviceId < 0 ? "C" : "G") << "PU\n   " << flush;

    // random init
    mt19937 rng(randomSeed);
    uniform_real_distribution<float> nd(-1, 1);
    vector<ElemType> init(numElements);
    generate(begin(init), end(init), [&] { return nd(rng); });

    // create storage object (one-column matrix)
    let sob = make_shared<Matrix<ElemType>>(numElements/*rows*/, 1/*cols*/, init.data(), deviceId);

    // create TensorView
    return TensorView<ElemType>(sob, shape);
}

template<class ElemType>
static TensorView<ElemType> BroadcastingTest(TensorShape layerShape, TensorShape biasShape, DEVICEID_TYPE deviceId)
{
    int randomSeed = 1;
    let  input = CreateTensor<ElemType>(layerShape, randomSeed++, deviceId);
    auto bias = CreateTensor<ElemType>(biasShape, randomSeed++, deviceId);

    auto result = CreateTensor<ElemType>(layerShape, randomSeed++, deviceId, true);
    result.AssignSumOf(input, bias);
    return result;
}

BOOST_AUTO_TEST_CASE(test01) {
    OneTensorTest("elementwise addition", 1e-8, [](DEVICEID_TYPE deviceId)->TensorView<float>
            {
                return BroadcastingTest<float>(TensorShape{ 512, 256 }, TensorShape({ 512, 256 }), deviceId);
            });
}

BOOST_AUTO_TEST_SUITE_END()

}
} } }