//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/Helpers.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_DefaultBehaviorTestForConstructors, RandomSeedFixture)
{
    const SingleMatrix matrixA1(c_deviceIdZero);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA1.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(0, matrixA1.GetNumCols());
    BOOST_CHECK_EQUAL(0, matrixA1.GetNumRows());

    const SingleMatrix matrixA2(CPUDEVICE);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixA2.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(0, matrixA2.GetNumCols());
    BOOST_CHECK_EQUAL(0, matrixA2.GetNumRows());

    const SingleMatrix matrixA3(13, 12, c_deviceIdZero);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA3.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(12, matrixA3.GetNumCols());
    BOOST_CHECK_EQUAL(13, matrixA3.GetNumRows());

    float arr[5 * 45];
    const SingleMatrix matrixA4(5, 45, arr, c_deviceIdZero, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA4.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(45, matrixA4.GetNumCols());
    BOOST_CHECK_EQUAL(5, matrixA4.GetNumRows());

    const SingleMatrix matrixA5(45, 5, arr, CPUDEVICE, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixA5.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(5, matrixA5.GetNumCols());
    BOOST_CHECK_EQUAL(45, matrixA5.GetNumRows());
}

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_AccessPatternAndTransferTest, RandomSeedFixture)
{
    float arr[5 * 45];
    const SingleMatrix matrixA(5, 45, arr, c_deviceIdZero, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());

    // GetValue calls operator() const, leaving the matrix in the BOTH state
    float x = matrixA.GetValue(0, 0);
    BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixA.GetCurrentMatrixLocation());
    foreach_coord(i, j, matrixA)
    {
        x = matrixA.GetValue(i, j);
        BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixA.GetCurrentMatrixLocation());
    }

    SingleMatrix matrixB(15, 15, arr, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());

    // non-const operator leaves it in CPU state so that writing to it is valid
    float& y = matrixB(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    matrixB(4, 2) = y;
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    foreach_coord (i, j, matrixB)
    {
        y = matrixB(i, j);
        matrixB(j, i) = y;
        BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    }

    matrixB.TransferFromDeviceToDevice(CPUDEVICE, c_deviceIdZero, false);
    BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(c_deviceIdZero, CPUDEVICE, false);
    BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(CPUDEVICE, c_deviceIdZero, true);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(c_deviceIdZero, CPUDEVICE, true);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
}

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_GravitatingTowardsPreferredDevice, RandomSeedFixture)
{
    SingleMatrix matrixA = SingleMatrix::RandomGaussian(64, 23, c_deviceIdZero, 0, 2, IncrementCounter());
    SingleMatrix matrixB = SingleMatrix::Eye(23, c_deviceIdZero);

    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());

    // Set the current matrix location by reading a value of the matrix (via non-const operator())
    float& x = matrixA(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixA.GetCurrentMatrixLocation());
    x = matrixB(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());

    const SingleMatrix matrixC = matrixA * matrixB;
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixC.GetCurrentMatrixLocation());
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }
