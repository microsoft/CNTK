//
// <copyright file="GPUMatrixTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// GPUMatrix unit tests should go here
//
#include "stdafx.h"
#include "../../../Math/Math/GPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            namespace Test
            {
                const int deviceId = 0;
                const int rows = 16;
                const int cols = 23;
                const float epsilon = 0.0001f;

                BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

#if 0
// TODO commented temporarily

				BOOST_AUTO_TEST_CASE(GPUMatrixConstructorNoFlag)
				{
                    // TODO: consider splitting into several tests
					GPUMatrix<float> m0(deviceId);
					BOOST_CHECK(m0.IsEmpty());

					GPUMatrix<float> m1(12, 53, deviceId);
					BOOST_CHECK_EQUAL(12, m1.GetNumRows());
					BOOST_CHECK_EQUAL(53, m1.GetNumCols());
					BOOST_CHECK_EQUAL(12 * 53, m1.GetNumElements());

                    std::array<float, 2> array = { 1, 14 };
					m1.SetValue(1, 2, deviceId, array.data());

                    unique_ptr<float[]> result(m1.CopyToArray());
                    BOOST_CHECK_EQUAL_COLLECTIONS(result.get(), result.get() + 2, array.begin(), array.end());

					GPUMatrix<float> m1Copy(m1);
					BOOST_CHECK(m1.IsEqualTo(m1Copy));
                }

#endif

				BOOST_AUTO_TEST_CASE(GPUMatrixConstructorFlagNormal)
				{
                    std::array<float, 6> array = { 1, 2, 3, 4, 5, 6 };
					GPUMatrix<float> m(2, 3, deviceId, array.data(), matrixFlagNormal);

                    unique_ptr<float[]> result(m.CopyToArray());
                    BOOST_CHECK_EQUAL_COLLECTIONS(result.get(), result.get() + 6, array.begin(), array.end());
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixIdentityAndZero)
                {
                    // TODO: consider splitting into two separate tests?
                    const int size = 60;
                    GPUMatrix<float> m0(GPUMatrix<float>::Eye(size, deviceId));
                    unique_ptr<float[]> result0(m0.CopyToArray());

                    for (int i = 0; i < size; i++) {
                        for (int j = 0; j < size; j++) {
                            BOOST_CHECK_CLOSE(result0[i * size + j], i == j, 0.01);
                        }
                    }

                    GPUMatrix<float> m1(GPUMatrix<float>::Zeros(size, size, deviceId));
                    unique_ptr<float[]> result1(m1.CopyToArray());
                    for (int i = 0; i < size; i++) {
                        for (int j = 0; j < size; j++) {
                            BOOST_CHECK_CLOSE(result1[i * size + j], 0.0f, 0.01);
                        }
                    }
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixElementWiseOperations)
                {
                    const float val = 3.0;

                    GPUMatrix<float> m0(rows, cols, deviceId);
                    m0.SetValue(val);
                    GPUMatrix<float> m1(rows, cols, deviceId);
                    GPUMatrix<float> mr(rows, cols, deviceId);

                    // test element wise power
                    float alpha = 2.0f;
                    GPUMatrix<float>::ElementWisePower(alpha, m0, m1);
                    mr.SetValue(std::pow(val, alpha));
                    BOOST_CHECK(mr.IsEqualTo(m1, epsilon));

                    alpha = 0.234f;
                    GPUMatrix<float>::ElementWisePower(alpha, m0, m1);
                    mr.SetValue(std::pow(val, alpha));
                    BOOST_CHECK(mr.IsEqualTo(m1, epsilon));

                    // test element wise absolute value
                    m0.SetValue(-val);
                    m1.AssignAbsOf(m0);
                    mr.SetValue(val);
                    BOOST_CHECK(mr.IsEqualTo(m1));

                    // TODO: add other element wise operations?
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixInplaceOperations)
                {
                    const float val = 0.42f;

                    GPUMatrix<float> m(rows, cols, deviceId);
                    GPUMatrix<float> mr(rows, cols, deviceId);

                    m.SetValue(val);
                    m.InplaceExp();
                    mr.SetValue(std::exp(val));
                    BOOST_CHECK(mr.IsEqualTo(m, epsilon));

                    m.SetValue(val);
                    m.InplaceLog();
                    mr.SetValue(std::log(val));
                    BOOST_CHECK(mr.IsEqualTo(m, epsilon));

                    m.SetValue(val);
                    m.InplaceTanh();
                    mr.SetValue(std::tanh(val));
                    BOOST_CHECK(mr.IsEqualTo(m, epsilon));

                    m.SetValue(-val);
                    m.InplaceAbs();
                    mr.SetValue(val);
                    BOOST_CHECK(mr.IsEqualTo(m, epsilon));

                    m.SetValue(val);
                    m.InplaceSqrt();
                    mr.SetValue(std::sqrt(val));
                    BOOST_CHECK(mr.IsEqualTo(m, epsilon));

                    m.SetValue(val);
                    m.InplaceSigmoid();
                    mr.SetValue(1 / (std::exp(-val) + 1));
                    BOOST_CHECK(mr.IsEqualTo(m, epsilon));

                    // TODO: there are two more inplace operations. Test these? compare to CPU results?
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixAddAndSub)
                {
                    std::array<float, 6> array0 = { 1, 2, 3, 4, 5, 6 };
                    GPUMatrix<float> m0(2, 3, deviceId, array0.data(), matrixFlagNormal);

                    std::array<float, 6> array1 = { 11, 12, 13, 14, 15, 16 };
                    GPUMatrix<float> m1(2, 3, deviceId, array1.data(), matrixFlagNormal);

                    std::array<float, 6> array2 = { 12, 14, 16, 18, 20, 22 };
                    GPUMatrix<float> m2(2, 3, deviceId, array2.data(), matrixFlagNormal);

                    std::array<float, 3> arrayCRS = { 10, 10, 10 };
                    GPUMatrix<float> mc(2, 1, deviceId, arrayCRS.data(), matrixFlagNormal);
                    GPUMatrix<float> mr(1, 3, deviceId, arrayCRS.data(), matrixFlagNormal);
                    GPUMatrix<float> ms(1, 1, deviceId, arrayCRS.data(), matrixFlagNormal);

                    GPUMatrix<float>  m3 = m2 - m0;
                    BOOST_CHECK(m3.IsEqualTo(m1));

                    m3 += m0;
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m3 = m0 + 10;
                    BOOST_CHECK(m3.IsEqualTo(m1));

                    m3 -= 10;
                    BOOST_CHECK(m3.IsEqualTo(m0));

                    m3 = m1 + m0;
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m3 -= m0;
                    BOOST_CHECK(m3.IsEqualTo(m1));

                    m3 = m1 - 10;
                    BOOST_CHECK(m3.IsEqualTo(m0));

                    m3 += 10;
                    BOOST_CHECK(m3.IsEqualTo(m1));

                    m3 -= mc;
                    BOOST_CHECK(m3.IsEqualTo(m0));

                    m3 += mc;
                    BOOST_CHECK(m3.IsEqualTo(m1));

                    m3 -= mr;
                    BOOST_CHECK(m3.IsEqualTo(m0));

                    m3 += mr;
                    BOOST_CHECK(m3.IsEqualTo(m1));

                    m3.AssignDifferenceOf(m3, ms);
                    BOOST_CHECK(m3.IsEqualTo(m0));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixNorms)
                {
                    std::array<float, 6> array = {
                        1, 4, 2, 
                        5, 3, 6 
                    };
                    GPUMatrix<float> m0(2, 3, deviceId, array.data(), matrixFlagNormal);

                    GPUMatrix<float> m3(deviceId);
                    m0.VectorNorm1(m3, true);
                    array[0] = 5; array[1] = 7; array[2] = 9;
                    GPUMatrix<float> m2(1, 3, deviceId, array.data(), matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m0.VectorNorm1(m3, false);
                    m2.Resize(2, 1);
                    array[0] = 6; array[1] = 15;
                    m2.SetValue(2, 1, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m0.VectorNorm2(m3, true);
                    m2.Resize(1, 3);
                    array[0] = 4.1231f; array[1] = 5.3852f; array[2] = 6.7082f;
                    m2.SetValue(1, 3, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2, 0.0005f));

                    m0.VectorNorm2(m3, false);
                    m2.Resize(2, 1);
                    array[0] = 3.7417f; array[1] = 8.7750f;
                    m2.SetValue(2, 1, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2, 0.0005f));

                    array[0] = 1; array[2] = 2; array[4] = 3;
                    array[1] = 4; array[3] = 5; array[5] = 6;
                    GPUMatrix<float> m00(2, 3, deviceId, array.data(), matrixFlagNormal);

                    GPUMatrix<float> m1(deviceId);
                    m00.VectorMax(m1, m3, true);
                    m2.Resize(1, 3);
                    array[0] = 4; array[1] = 5; array[2] = 6;
                    m2.SetValue(1, 3, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m00.VectorMax(m1, m3, false);
                    m2.Resize(2, 1);
                    array[0] = 3.; array[1] = 6;
                    m2.SetValue(2, 1, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m0.VectorNormInf(m3, true);
                    m2.Resize(1, 3);
                    array[0] = 4; array[1] = 5; array[2] = 6;
                    m2.SetValue(1, 3, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m0.VectorNormInf(m3, false);
                    m2.Resize(2, 1);
                    array[0] = 3.; array[1] = 6;
                    m2.SetValue(2, 1, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    array[0] = 1; array[2] = 2; array[4] = 3;
                    array[1] = 4; array[3] = 5; array[5] = 6;
                    m00.SetValue(2, 3, m2.GetComputeDeviceId(), array.data(), matrixFlagNormal);
                    BOOST_CHECK_EQUAL(6, m00.MatrixNormInf());

                    BOOST_CHECK(abs(m0.FrobeniusNorm() - 9.5394) < 0.0001);
                    BOOST_CHECK(abs(m0.MatrixNormInf() - 6) < 0.0001);
                    BOOST_CHECK_EQUAL(21, m00.MatrixNorm1());

                    GPUMatrix<float> a = GPUMatrix<float>::Eye(4096, deviceId);
                    BOOST_CHECK_EQUAL(4096, a.MatrixNorm0());

                    GPUMatrix<float> b = GPUMatrix<float>::Eye(5, deviceId);
                    BOOST_CHECK_EQUAL(5, b.MatrixNorm0());
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixRandomUniform)
                {
                    auto m = GPUMatrix<float>::RandomUniform(768, 50, deviceId, -0.035f, 0.035f, 1L);
                    unique_ptr<float[]> result(m.CopyToArray());

                    for (int i = 0; i < 768 * 50; ++i)
                    {
                        BOOST_CHECK(result[i] <= 0.035);
                        BOOST_CHECK(result[i] > -0.035);
                    }
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixColumnSlice)
                {
                    std::array<float, 6> array = {
                        1, 4, 2,
                        5, 3, 6
                    };
                    GPUMatrix<float> m0(2, 3, deviceId, array.data(), matrixFlagNormal);
                    GPUMatrix<float> m1(2, 2, deviceId, array.data(), matrixFlagNormal);

                    GPUMatrix<float> m2 = m0.ColumnSlice(0, 2);
                    BOOST_CHECK(m2.IsEqualTo(m1));

                    std::array<float, 4> array3 = { array[2], array[3], array[4], array[5] };
                    GPUMatrix<float> m3(2, 2, deviceId, array3.data(), matrixFlagNormal);

                    m2 = m0.ColumnSlice(1, 2);
                    BOOST_CHECK(m2.IsEqualTo(m3));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixRowSlice)
                {
                    std::array<float, 15> array0 = {
                        1, 2, 3, 
                        4, 5, 6, 
                        7, 8, 9, 
                        10, 11, 12, 
                        13, 14, 15
                    };
                    GPUMatrix<float> m0(5, 3, deviceId, array0.data(), matrixFlagNormal);

                    std::array<float, 6> array1 = {
                        3, 4, 8, 
                        9, 13, 14
                    };
                    GPUMatrix<float> m1(2, 3, deviceId, array1.data(), matrixFlagNormal);

                    GPUMatrix<float> m2(deviceId);
                    m2.AssignRowSliceValuesOf(m0, 2, 2);
                    BOOST_CHECK(m2.IsEqualTo(m1));

                    std::array<float, 15> array3 = {
                        0, 0, 3,
                        4, 0, 0, 
                        0, 8, 9, 
                        0, 0, 0,
                        13, 14, 0
                    };
                    GPUMatrix<float> m3(5, 3, deviceId, array3.data(), matrixFlagNormal);

                    m3 += m0;
                    m0.AddToRowSliceValuesOf(m1, 2, 2);
                    BOOST_CHECK(m3.IsEqualTo(m0));

                    m2.AddWithRowSliceValuesOf(m1, 0, 2);
                    std::array<float, 6> array4 = {
                        6, 8, 16,
                        18, 26, 28
                    };
                    GPUMatrix<float> m4(2, 3, deviceId, array4.data(), matrixFlagNormal);
                    BOOST_CHECK(m2.IsEqualTo(m4));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixKhatriRaoProduct)
                {
                    std::array<float, 12> arrayA = {
                        0.8147f, 0.9058f, 0.1270f, 0.9134f, 
                        0.6324f, 0.0975f, 0.2785f, 0.5469f, 
                        0.9575f, 0.9649f, 0.1576f, 0.9706f
                    };
                    GPUMatrix<float> a(3, 4, deviceId, arrayA.data());

                    std::array<float, 8> arrayB = {
                        0.9572f, 0.4854f, 0.8003f, 0.1419f,
                        0.4218f, 0.9157f, 0.7922f, 0.9595f
                    };
                    GPUMatrix<float> b(2, 4, deviceId, arrayB.data());

                    std::array<float, 24> arrayD = {
                        0.7798f, 0.8670f, 0.1215f, 0.3954f, 
                        0.4396f, 0.0616f, 0.7310f, 0.5061f, 
                        0.0781f, 0.1296f, 0.0897f, 0.0138f, 
                        0.1175f, 0.2307f, 0.4038f, 0.2550f, 
                        0.5008f, 0.8768f, 0.7644f, 0.1249f, 
                        0.7689f, 0.9258f, 0.1512f, 0.9313f
                    };
                    GPUMatrix<float> d(6, 4, deviceId, arrayD.data());

                    GPUMatrix<float> c(deviceId);
                    c.AssignKhatriRaoProductOf(a, b);
                    BOOST_CHECK(c.IsEqualTo(d, epsilon));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixAddColumnReshapeProductOf)
                {
                    // tests column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
                    std::array<float, 12> arrayA = {
                        0.6557f, 0.0357f,
                        0.8491f, 0.9340f,
                        0.6787f, 0.7577f, 
                        0.7431f, 0.3922f,
                        0.6555f, 0.1712f, 
                        0.7060f, 0.0318f,
                    };
                    GPUMatrix<float> a(6, 2, deviceId, arrayA.data());

                    std::array<float, 6> arrayB = {
                        0.2769f, 0.0462f, 
                        0.0971f, 0.8235f,
                        0.6948f, 0.3171f
                    };
                    GPUMatrix<float> b(3, 2, deviceId, arrayB.data());

                    std::array<float, 4> arrayD0 = {
                        0.2867f, 0.1266f,
                        1.2913f, 0.4520f
                    };
                    GPUMatrix<float> d0(2, 2, deviceId, arrayD0.data());

                    std::array<float, 4> arrayD1 = {
                        0.2657f, 0.3636f,
                        1.0923f, 0.6416f
                    };
                    GPUMatrix<float> d1(2, 2, deviceId, arrayD1.data());

                    GPUMatrix<float> c(2, 2, deviceId);
                    c.SetValue(0.0f);
                    c.AddColumnReshapeProductOf(a, b, false);
                    BOOST_CHECK(c.IsEqualTo(d0, epsilon));

                    c.SetValue(0.0f);
                    c.AddColumnReshapeProductOf(a, b, true);
                    BOOST_CHECK(c.IsEqualTo(d1, epsilon));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixInnerProduct)
                {
                    std::array<float, 6> array = {
                        1, 4, 2,
                        5, 3, 6
                    };
                    GPUMatrix<float> m0(2, 3, deviceId, array.data(), matrixFlagNormal);

                    GPUMatrix<float> m1(deviceId), m2(deviceId);
                    m1.AssignInnerProductOf(m0, m0, true);
                    m2.AssignVectorNorm2Of(m0, true);
                    m1.InplaceSqrt();
                    BOOST_CHECK(m1.IsEqualTo(m2));

                    m1.AssignInnerProductOf(m0, m0, false);
                    m2.AssignVectorNorm2Of(m0, false);
                    m1.InplaceSqrt();
                    BOOST_CHECK(m1.IsEqualTo(m2));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixAssignRepeatOf)
                {
                    std::array<float, 6> array0 = {
                        1, 2, 
                        6, 7, 
                        11, 12
                    };
                    GPUMatrix<float> m0(2, 3, deviceId, array0.data(), matrixFlagNormal);

                    GPUMatrix<float>  m1(deviceId);
                    m1.AssignRepeatOf(m0, 1, 1);
                    BOOST_CHECK(m1.IsEqualTo(m0));

                    std::array<float, 36> array2 = {
                        1, 2, 1, 2, 1, 2,
                        6, 7, 6, 7, 6, 7,
                        11, 12, 11, 12, 11, 12,
                        1, 2, 1, 2, 1, 2,
                        6, 7, 6, 7, 6, 7,
                        11, 12, 11, 12, 11, 12
                    };
                    GPUMatrix<float> m2(6, 6, deviceId, array2.data(), matrixFlagNormal);

                    m1.AssignRepeatOf(m0, 3, 2);
                    BOOST_CHECK(m1.IsEqualTo(m2));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixRowElementOperations)
                {
                    GPUMatrix<float> m0 = GPUMatrix<float>::RandomUniform(20, 28, deviceId, -1, 1);
                    GPUMatrix<float> m1 = GPUMatrix<float>::RandomUniform(1, 28, deviceId, 1, 2);

                    GPUMatrix<float> m2(deviceId);
                    m2.SetValue(m0);
                    m2.RowElementMultiplyWith(m1);
                    m2.RowElementDivideBy(m1);

                    BOOST_CHECK(m0.IsEqualTo(m2, epsilon));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixColumnElementOperations)
                {
                    GPUMatrix<float> m0 = GPUMatrix<float>::RandomUniform(20, 28, deviceId, -1, 1);
                    GPUMatrix<float> m1 = GPUMatrix<float>::RandomUniform(20, 1, deviceId, 1, 2);

                    GPUMatrix<float> m2(deviceId);
                    m2.SetValue(m0);
                    m2.ColumnElementMultiplyWith(m1);
                    m2.ColumnElementDivideBy(m1);

                    BOOST_CHECK(m0.IsEqualTo(m2, epsilon));
                }

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}