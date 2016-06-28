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
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(MathPerformance)

template <typename FN>
struct TensorTestParameters{
    const char * testString;
    double tolerance;
    const FN& fn;
};

template<typename FN>
class TensorTest {
public:
    void OneTensorTest(const char* what, double tolerance, const FN& fn)
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
};

TensorTestParameters<TensorView<float>> parameters[];

TensorTest<float> tester;

bool InitUnitTest()
{
    return false;
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }