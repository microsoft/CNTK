//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <math.h>
#ifdef _WIN32
#include <crtdefs.h>
#endif 
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/CPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

class MatrixLearnerFixture : public RandomSeedFixture
{
public:
    static const size_t dim1 = 256;
    static const size_t dim2 = 128;
    static const size_t dim3 = 2048;

    SingleMatrix matSG;
    SingleMatrix matSGsparse;
    SingleMatrix matM;
    SingleMatrix matMsparse;
    SingleMatrix matG;
    SingleMatrix matGsparseBSC;
    SingleMatrix timestamps;

    MatrixLearnerFixture() :
        matSG(c_deviceIdZero),
        matSGsparse(c_deviceIdZero),
        matM(c_deviceIdZero),
        matMsparse(c_deviceIdZero),
        matG(c_deviceIdZero),
        matGsparseBSC(c_deviceIdZero),
        timestamps(c_deviceIdZero)
    {
        // smoothed gradient
        matSG = SingleMatrix::RandomGaussian(dim1, dim2, c_deviceIdZero, -1.0f, 1.0f, IncrementCounter());
        matSGsparse = SingleMatrix(matSG.DeepClone());

        // model
        matM = SingleMatrix::RandomGaussian(dim1, dim2, c_deviceIdZero, -1.0f, 1.0f, IncrementCounter());
        matMsparse = SingleMatrix(matM.DeepClone());

        // generates gradient
        SingleMatrix matG1(c_deviceIdZero);
        matG1.AssignTruncateBottomOf(Matrix<float>::RandomUniform(dim2, dim3, c_deviceIdZero, -300.0f, 0.1f, IncrementCounter()), 0);

        SingleMatrix matG1sparseCSC(matG1.DeepClone());
        matG1sparseCSC.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, true);

        SingleMatrix matG2 = SingleMatrix::RandomGaussian(dim1, dim3, c_deviceIdZero, -1.0f, 1.0f, IncrementCounter());

        SingleMatrix::MultiplyAndWeightedAdd(1, matG2, false, matG1, true, 0, matG);

        matGsparseBSC.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
        SingleMatrix::MultiplyAndAdd(matG2, false, matG1sparseCSC, true, matGsparseBSC);
        timestamps = SingleMatrix::RandomGaussian(1, dim2, c_deviceIdZero, -1.0f, 1.0f, IncrementCounter());
    }

    void RunOnDevices(std::function<void()> func)
    {
        for (int deviceId : {-1, 0})
        {
            matSG.TransferToDeviceIfNotThere(deviceId, true);
            matSGsparse.TransferToDeviceIfNotThere(deviceId, true);
            matM.TransferToDeviceIfNotThere(deviceId, true);
            matMsparse.TransferToDeviceIfNotThere(deviceId, true);
            matG.TransferToDeviceIfNotThere(deviceId, true);
            matGsparseBSC.TransferToDeviceIfNotThere(deviceId, true);
            timestamps.TransferToDeviceIfNotThere(deviceId, true);
            func();
        }
    }
};

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(MatrixLearnerSuite)

// tests FSAdagrad sparse vs. dense
BOOST_FIXTURE_TEST_CASE(FSAdagradSparse, MatrixLearnerFixture)
{
    // run learner
    double targetAdagradAvDenom_x_sqrtAdagradSqrFrames = 0.5;
    matSG.FSAdagradUpdate(matG, matM, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, 0.0001, 1.0, 0.9, 1.0);

    matSGsparse.FSAdagradUpdate(matGsparseBSC, matMsparse, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, 0.0001, 1.0, 0.9, 1.0);

    BOOST_CHECK(matSG.IsEqualTo(matSGsparse, c_epsilonFloatE5));
    BOOST_CHECK(matM.IsEqualTo(matMsparse, c_epsilonFloatE5));
}

// tests RmsProp sparse vs. dense
BOOST_FIXTURE_TEST_CASE(RmsPropSparse, MatrixLearnerFixture)
{
    // run learner
    float avg = matSG.RmsProp(matG, 0.99f, 1.2f, 10.0f, 0.75f, 0.1f, true, false);
    float avgSparse = matSGsparse.RmsProp(matGsparseBSC, 0.99f, 1.2f, 10.0f, 0.75f, 0.1f, true, false);

    BOOST_CHECK(matSG.IsEqualTo(matSGsparse, c_epsilonFloatE3));
    BOOST_CHECK(fabsf(avg - avgSparse) < c_epsilonFloatE5);
}

// tests AdaDelta sparse vs. dense
BOOST_FIXTURE_TEST_CASE(AdaDeltaSparse, MatrixLearnerFixture)
{
    // run learner
    RunOnDevices([this]()
    {
        timestamps.SetValue(0.0f);
        auto ts = reinterpret_cast<int*>(timestamps.Data());
        matSG.AdaDeltaUpdate(matG, matM, 0.5f, 0.95f, 1e-8f, nullptr, 0);
        matSGsparse.AdaDeltaUpdate(matGsparseBSC, matMsparse, 0.5f, 0.95f, 1e-8f, ts, 1);
        matSGsparse.AdaDeltaFlushState(matGsparseBSC.GetNumCols(), 0.95f, ts, 1);

        BOOST_CHECK(matSG.IsEqualTo(matSGsparse, c_epsilonFloatE4));
        BOOST_CHECK(matM.IsEqualTo(matMsparse, c_epsilonFloatE4));
    });
}

BOOST_AUTO_TEST_SUITE_END()
}}}}
