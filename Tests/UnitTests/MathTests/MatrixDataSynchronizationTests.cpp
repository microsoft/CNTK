//
// <copyright file="MatrixDataSynchronizationTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "..\..\..\Math\Math\Matrix.h"
#include "..\..\..\Math\Math\Helpers.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            namespace Test
            {
                BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

                    // This tests defaulting to GPU. Your machine has to have CUDA5 GPU to pass it
                    BOOST_AUTO_TEST_CASE(MatrixDataSynchronization_DefaultBehaviorTestForConstructors)
                {
                    SingleMatrix matrixA;
                    BOOST_CHECK(CurrentDataLocation::GPU == matrixA.GetCurrentMatrixLocation());
                    BOOST_CHECK_EQUAL(0, matrixA.GetNumCols());
                    BOOST_CHECK_EQUAL(0, matrixA.GetNumRows());

                    SingleMatrix matrixA1(AUTOPLACEMATRIX);
                    BOOST_CHECK(CurrentDataLocation::GPU == matrixA1.GetCurrentMatrixLocation());
                    BOOST_CHECK_EQUAL(0, matrixA1.GetNumCols());
                    BOOST_CHECK_EQUAL(0, matrixA1.GetNumRows());

                    SingleMatrix matrixA2(static_cast<size_t>(13), static_cast<size_t>(12));
                    BOOST_CHECK(CurrentDataLocation::GPU == matrixA2.GetCurrentMatrixLocation());
                    BOOST_CHECK_EQUAL(12, matrixA2.GetNumCols());
                    BOOST_CHECK_EQUAL(13, matrixA2.GetNumRows());

                    float arr[255];
                    SingleMatrix matrixA3(5, 45, arr, matrixFlagNormal);
                    BOOST_CHECK(CurrentDataLocation::GPU == matrixA3.GetCurrentMatrixLocation());
                    BOOST_CHECK_EQUAL(45, matrixA3.GetNumCols());
                    BOOST_CHECK_EQUAL(5, matrixA3.GetNumRows());
                }

                // This tests defaulting to GPU and transfering. Your machine has to have CUDA5 GPU to pass it
                BOOST_AUTO_TEST_CASE(MatrixDataSynchronization_AccessPatternAndTransferTest)
                {
                    float arr[255];
                    const SingleMatrix matrixA(5, 45, &arr[0], matrixFlagNormal);
                    BOOST_CHECK(CurrentDataLocation::GPU == matrixA.GetCurrentMatrixLocation());
                    BOOST_CHECK_EQUAL(45, matrixA.GetNumCols());
                    BOOST_CHECK_EQUAL(5, matrixA.GetNumRows());

                    float x = matrixA(1, 1);
                    BOOST_CHECK(matrixA.GetCurrentMatrixLocation() != CurrentDataLocation::GPU);
                    foreach_coord(i, j, matrixA)
                    {
                        x = matrixA(i, j);
                        BOOST_CHECK(matrixA.GetCurrentMatrixLocation() != CurrentDataLocation::GPU);
                    }

                    SingleMatrix matrixB(15, 15, arr, matrixFlagNormal);
                    BOOST_CHECK(CurrentDataLocation::GPU == matrixB.GetCurrentMatrixLocation());
                    BOOST_CHECK_EQUAL(15, matrixB.GetNumCols());
                    BOOST_CHECK_EQUAL(15, matrixB.GetNumRows());

                    x = matrixB(1, 1);
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::CPU);
                    matrixB(4, 2) = x;
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::CPU);
                    foreach_coord(i, j, matrixB)
                    {
                        x = matrixB(i, j);
                        matrixB(j, i) = x;
                        BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::CPU);
                    }

                    matrixB.TransferFromDeviceToDevice(-1, 0, false);
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::BOTH);
                    matrixB.TransferFromDeviceToDevice(0, -1, false);
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::BOTH);
                    matrixB.TransferFromDeviceToDevice(-1, 0, true);
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::GPU);
                    matrixB.TransferFromDeviceToDevice(0, -1, true);
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::CPU);

                    SingleMatrix matrixC(-1);
                    BOOST_CHECK(matrixC.GetCurrentMatrixLocation() == CurrentDataLocation::CPU);

                    SingleMatrix matrixD(15, 15, arr, matrixFlagNormal, -1);
                    BOOST_CHECK(matrixD.GetCurrentMatrixLocation() == CurrentDataLocation::CPU);
                }

                // This tests defaulting to GPU, transfering and gravitating towards preferred device. 
                // Your machine has to have CUDA5 GPU to pass it
                BOOST_AUTO_TEST_CASE(MatrixDataSynchronization_GravitatingTowardsPreferredDevice)
                {
                    SingleMatrix matrixA = SingleMatrix::RandomGaussian(64, 23, 0, 2);
                    SingleMatrix matrixB = SingleMatrix::Eye(23);

                    BOOST_CHECK(matrixA.GetCurrentMatrixLocation() == CurrentDataLocation::GPU);
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::GPU);

                    // Set the current matrix location by reading a value of the matrix
                    float x = matrixA(1, 1);
                    x;
                    BOOST_CHECK(matrixA.GetCurrentMatrixLocation() == CurrentDataLocation::CPU);
                    x = matrixB(1, 1);
                    x;
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::CPU);

                    SingleMatrix matrixC = matrixA * matrixB;
                    BOOST_CHECK(matrixA.GetCurrentMatrixLocation() == CurrentDataLocation::GPU);
                    BOOST_CHECK(matrixB.GetCurrentMatrixLocation() == CurrentDataLocation::GPU);
                    BOOST_CHECK(matrixC.GetCurrentMatrixLocation() == CurrentDataLocation::GPU);
                }

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}
