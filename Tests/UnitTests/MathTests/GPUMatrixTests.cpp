//
// <copyright file="GPUMatrixTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// GPUMatrix unit tests should go here
//
#include <boost/test/unit_test.hpp>
#include "stdafx.h"
#include "..\..\..\Math\Math\GPUMatrix.h"

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

				BOOST_AUTO_TEST_CASE(GPUMatrixConstructorNoFlag)
				{
					GPUMatrix<float> m0(0 /*deviceId*/);
					BOOST_CHECK(m0.IsEmpty());

					GPUMatrix<float> m1(12, 53, 0 /*deviceId*/);
					BOOST_CHECK_EQUAL(12, m1.GetNumRows());
					BOOST_CHECK_EQUAL(53, m1.GetNumCols());
					BOOST_CHECK_EQUAL(12*53, m1.GetNumElements());

					// GPUMatrix doesn't support operator() --> using SetValue and CopyToArray
					float *fArray = new float[2];
					fArray[0] = 1; fArray[1] = 14;
					m1.SetValue(1, 2, 0 /* deviceId */, fArray);
					float* result = m1.CopyToArray();
					BOOST_CHECK_EQUAL(*result, 1.0f);
					BOOST_CHECK_EQUAL(*(result+1), 14.0f);

					GPUMatrix<float> m1C(m1);
					BOOST_CHECK(m1.IsEqualTo(m1C));
				}

				// GPUMatrix doesn't support flag 'FormatRowMajor'
				BOOST_AUTO_TEST_CASE(GPUMatrixConstructorFlagNormal)
				{
					float *fArray = new float[6];
					fArray[0] = 1; fArray[1] = 2; fArray[2] = 3;
					fArray[3] = 4; fArray[4] = 5; fArray[5] = 6;

					GPUMatrix<float> m(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

					// GPUMatrix doesn't support operator() --> using CopyToArray
					float* result = m.CopyToArray();
					BOOST_CHECK_EQUAL(*result, 1.0f);
					BOOST_CHECK_EQUAL(*(result + 1), 2.0f);
					BOOST_CHECK_EQUAL(*(result + 2), 3.0f);
					BOOST_CHECK_EQUAL(*(result + 3), 4.0f);
					BOOST_CHECK_EQUAL(*(result + 4), 5.0f);
					BOOST_CHECK_EQUAL(*(result + 5), 6.0f);
				}

                BOOST_AUTO_TEST_CASE(GPUMatrixIdentityAndZero)
                {
                    const int size = 60;
                    GPUMatrix<float> m0(GPUMatrix<float>::Eye(size, 0 /*deviceId*/));
                    float* result = m0.CopyToArray();
                    for (int i = 0; i < size; i++) {
                        for (int j = 0; j < size; j++) {
                            BOOST_CHECK_CLOSE(result[i*size + j], i == j, 0.01);
                        }
                    }

                    GPUMatrix<float> m1(GPUMatrix<float>::Zeros(size, size, 0 /*deviceId*/));
                    result = m1.CopyToArray();      
                    for (int i = 0; i < size; i++) {
                        for (int j = 0; j < size; j++) {
                            BOOST_CHECK_CLOSE(result[i*size + j], 0.0f, 0.01);
                        }
                    }
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixElementWiseOperations)
                {
                    const int rows = 16;
                    const int cols = 23;
                    const float val = 3.0f;

                    GPUMatrix<float> m0(rows, cols, 0 /*deviceId*/);
                    m0.SetValue(val);
                    GPUMatrix<float> m1(rows, cols, 0 /*deviceId*/);
                    GPUMatrix<float> mr(rows, cols, 0 /*deviceId*/);

                    // test element wise power
                    float alpha = 2.0f;
                    GPUMatrix<float>::ElementWisePower(alpha, m0, m1);
                    mr.SetValue(std::pow(val, alpha));
                    BOOST_CHECK(mr.IsEqualTo(m1, 0.00001f));

                    alpha = 0.234f;
                    GPUMatrix<float>::ElementWisePower(alpha, m0, m1);
                    mr.SetValue(std::pow(val, alpha));
                    BOOST_CHECK(mr.IsEqualTo(m1, 0.00001f));

                    // test element wise absolute value
                    m0.SetValue(-val);
                    m1.AssignAbsOf(m0);
                    mr.SetValue(val);
                    BOOST_CHECK(mr.IsEqualTo(m1, 0.00001f));

                    // TODO: add other element wise operations?
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixInplaceOperations)
                {
                    const int rows = 16;
                    const int cols = 23;
                    const float val = 0.42f;

                    GPUMatrix<float> m(rows, cols, 0 /*deviceId*/);
                    GPUMatrix<float> mr(rows, cols, 0 /*deviceId*/);

                    m.SetValue(val);
                    m.InplaceExp();
                    mr.SetValue(std::exp(val));
                    BOOST_CHECK(mr.IsEqualTo(m, 0.00001f));

                    m.SetValue(val);
                    m.InplaceLog();
                    mr.SetValue(std::log(val));
                    BOOST_CHECK(mr.IsEqualTo(m, 0.00001f));

                    m.SetValue(val);
                    m.InplaceTanh();
                    mr.SetValue(std::tanh(val));
                    BOOST_CHECK(mr.IsEqualTo(m, 0.00001f));

                    m.SetValue(-val);
                    m.InplaceAbs();
                    mr.SetValue(val);
                    BOOST_CHECK(mr.IsEqualTo(m, 0.00001f));

                    m.SetValue(val);
                    m.InplaceSqrt();
                    mr.SetValue(std::sqrt(val));
                    BOOST_CHECK(mr.IsEqualTo(m, 0.00001f));

                    m.SetValue(val);
                    m.InplaceSigmoid();
                    mr.SetValue(1 / (std::exp(-val) + 1));
                    BOOST_CHECK(mr.IsEqualTo(m, 0.00001f));

                    // TODO: there are two more inplace operations. Test these? compare to CPU results?
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixAddAndSub)
                {
                    float *fArray = new float[6];
                    fArray[0] = 1; fArray[2] = 2; fArray[4] = 3;
                    fArray[1] = 4; fArray[3] = 5; fArray[5] = 6;
                    GPUMatrix<float> m0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    fArray[0] = 11; fArray[2] = 12; fArray[4] = 13;
                    fArray[1] = 14; fArray[3] = 15; fArray[5] = 16;
                    GPUMatrix<float> m1(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    fArray[0] = 12; fArray[2] = 14; fArray[4] = 16;
                    fArray[1] = 18; fArray[3] = 20; fArray[5] = 22;
                    GPUMatrix<float> m2(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    fArray[0] = 10;
                    fArray[1] = 10;
                    GPUMatrix<float> mc(2, 1, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    fArray[0] = 10; fArray[1] = 10; fArray[2] = 10;
                    GPUMatrix<float> mr(1, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    fArray[0] = 10;
                    GPUMatrix<float> ms(1, 1, 0 /*deviceId*/, fArray, matrixFlagNormal);

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
                    float *fArray = new float[6];
                    fArray[0] = 1; fArray[2] = 2; fArray[4] = 3;
                    fArray[1] = 4; fArray[3] = 5; fArray[5] = 6;
                    GPUMatrix<float> m0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    GPUMatrix<float> m3(0 /*deviceId*/);
                    m0.VectorNorm1(m3, true);
                    fArray[0] = 5; fArray[1] = 7; fArray[2] = 9;
                    GPUMatrix<float> m2(1, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m0.VectorNorm1(m3, false);
                    m2.Resize(2, 1);
                    fArray[0] = 6; fArray[1] = 15;
                    m2.SetValue(2, 1, m2.GetComputeDeviceId(), fArray, matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    m0.VectorNorm2(m3, true);
                    m2.Resize(1, 3);
                    fArray[0] = 4.1231f; fArray[1] = 5.3852f; fArray[2] = 6.7082f;
                    m2.SetValue(1, 3, m2.GetComputeDeviceId(), fArray, matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2, 0.0005f));

                    m0.VectorNorm2(m3, false);
                    m2.Resize(2, 1);
                    fArray[0] = 3.7417f; fArray[1] = 8.7750f;
                    m2.SetValue(2, 1, m2.GetComputeDeviceId(), fArray, matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2, 0.0005f));

                    fArray[0] = 1; fArray[2] = 2; fArray[4] = 3;
                    fArray[1] = 4; fArray[3] = 5; fArray[5] = 6;
                    GPUMatrix<float> m00(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    GPUMatrix<float> m1(0 /*deviceId*/);
                    m00.VectorMax(m1, m3, true);
                    m2.Resize(1, 3);
                    fArray[0] = 4; fArray[1] = 5; fArray[2] = 6;
                    m2.SetValue(1, 3, m2.GetComputeDeviceId(), fArray, matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2, 0.0001f));

                    m00.VectorMax(m1, m3, false);
                    m2.Resize(2, 1);
                    fArray[0] = 3.; fArray[1] = 6;
                    m2.SetValue(2, 1, m2.GetComputeDeviceId(), fArray, matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2, 0.0001f));

                    m0.VectorNormInf(m3, true);
                    m2.Resize(1, 3);
                    fArray[0] = 4; fArray[1] = 5; fArray[2] = 6;
                    m2.SetValue(1, 3, m2.GetComputeDeviceId(), fArray, matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2, 0.0001f));

                    m0.VectorNormInf(m3, false);
                    m2.Resize(2, 1);
                    fArray[0] = 3.; fArray[1] = 6;
                    m2.SetValue(2, 1, m2.GetComputeDeviceId(), fArray, matrixFlagNormal);
                    BOOST_CHECK(m3.IsEqualTo(m2));

                    fArray[0] = 1; fArray[2] = 2; fArray[4] = 3;
                    fArray[1] = 4; fArray[3] = 5; fArray[5] = 6;
                    m00.SetValue(2, 3, m2.GetComputeDeviceId(), fArray, matrixFlagNormal);
                    BOOST_CHECK_EQUAL(6, m00.MatrixNormInf());

                    BOOST_CHECK(abs(m0.FrobeniusNorm() - 9.5394) < 0.0001);
                    BOOST_CHECK(abs(m0.MatrixNormInf() - 6) < 0.0001);
                    BOOST_CHECK_EQUAL(21, m00.MatrixNorm1());

                    GPUMatrix<float> a = GPUMatrix<float>::Eye(4096, 0 /*deviceId*/);
                    BOOST_CHECK_EQUAL(4096, a.MatrixNorm0());

                    GPUMatrix<float> b = GPUMatrix<float>::Eye(5, 0 /*deviceId*/);
                    BOOST_CHECK_EQUAL(5, b.MatrixNorm0());
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixRandomUniform)
                {
                    GPUMatrix<float> a = GPUMatrix<float>::RandomUniform(768, 50, 0 /* device id */, -0.035f, 0.035f, 1L);
                    float* arr = a.CopyToArray();

                    for (int i = 0; i<768 * 50; ++i)
                    {
                        BOOST_CHECK(arr[i] <= 0.035);
                        BOOST_CHECK(arr[i]>-0.035);
                    }

                    delete[] arr;
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixColumnSlice)
                {
                    float *fArray = new float[6];
                    fArray[0] = 1; fArray[1] = 4; fArray[2] = 2;
                    fArray[3] = 5; fArray[4] = 3; fArray[5] = 6;
                    GPUMatrix<float> m0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    GPUMatrix<float> m1(2, 2, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    GPUMatrix<float> m2 = m0.ColumnSlice(0, 2);
                    BOOST_CHECK(m2.IsEqualTo(m1, 0.0001f));

                    GPUMatrix<float> m3(2, 2, 0 /*deviceId*/, fArray + 2, matrixFlagNormal);

                    m2 = m0.ColumnSlice(1, 2);
                    BOOST_CHECK(m2.IsEqualTo(m3, 0.0001f));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixRowSlice)
                {
                    float *fArray = new float[15];
                    fArray[0] = 1; fArray[5] = 6; fArray[10] = 11;
                    fArray[1] = 2; fArray[6] = 7; fArray[11] = 12;
                    fArray[2] = 3; fArray[7] = 8; fArray[12] = 13;
                    fArray[3] = 4; fArray[8] = 9; fArray[13] = 14;
                    fArray[4] = 5; fArray[9] = 10; fArray[14] = 15;
                    GPUMatrix<float> m0(5, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    float *fArray1 = new float[6];
                    fArray1[0] = 3; fArray1[2] = 8; fArray1[4] = 13;
                    fArray1[1] = 4; fArray1[3] = 9; fArray1[5] = 14;
                    GPUMatrix<float> m1(2, 3, 0 /*deviceId*/, fArray1, matrixFlagNormal);

                    GPUMatrix<float> m2(0 /*deviceId*/);
                    m2.AssignRowSliceValuesOf(m0, 2, 2);
                    BOOST_CHECK(m2.IsEqualTo(m1, 0.0001f));

                    float *fArray3 = new float[15];
                    fArray3[0] = 0; fArray3[5] = 0; fArray3[10] = 0;
                    fArray3[1] = 0; fArray3[6] = 0; fArray3[11] = 0;
                    fArray3[2] = 3; fArray3[7] = 8; fArray3[12] = 13;
                    fArray3[3] = 4; fArray3[8] = 9; fArray3[13] = 14;
                    fArray3[4] = 0; fArray3[9] = 0; fArray3[14] = 0;
                    GPUMatrix<float> m3(5, 3, 0 /*deviceId*/, fArray3, matrixFlagNormal);

                    m3 += m0;
                    m0.AddToRowSliceValuesOf(m1, 2, 2);
                    BOOST_CHECK(m3.IsEqualTo(m0, 0.0001f));

                    m2.AddWithRowSliceValuesOf(m1, 0, 2);
                    float *fArray4 = new float[6];
                    fArray4[0] = 6; fArray4[2] = 16; fArray4[4] = 26;
                    fArray4[1] = 8; fArray4[3] = 18; fArray4[5] = 28;
                    GPUMatrix<float> m4(2, 3, 0 /*deviceId*/, fArray4, matrixFlagNormal);
                    BOOST_CHECK(m2.IsEqualTo(m4, 0.0001f));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixKhatriRaoProduct)
                {
                    float *fArray = new float[24];
                    fArray[0] = 0.8147f; fArray[3] = 0.9134f; fArray[6] = 0.2785f; fArray[9] = 0.9649f;
                    fArray[1] = 0.9058f; fArray[4] = 0.6324f; fArray[7] = 0.5469f; fArray[10] = 0.1576f;
                    fArray[2] = 0.1270f; fArray[5] = 0.0975f; fArray[8] = 0.9575f; fArray[11] = 0.9706f;
                    GPUMatrix<float> a(3, 4, 0 /*deviceId*/, fArray);

                    fArray[0] = 0.9572f; fArray[2] = 0.8003f; fArray[4] = 0.4218f; fArray[6] = 0.7922f;
                    fArray[1] = 0.4854f; fArray[3] = 0.1419f; fArray[5] = 0.9157f; fArray[7] = 0.9595f;
                    GPUMatrix<float> b(2, 4, 0 /*deviceId*/, fArray);

                    // a00 * b00, a01 * b01, a02 * b02, a03 * b03
                    // a10 * b00, a11 * b01, a12 * b02, a13 * b03
                    // a20 * b00, ...
                    // a00 * b10
                    // a10 * b10
                    // a20 * b10
                    fArray[0] = 0.7798f; fArray[6] = 0.7310f; fArray[12] = 0.1175f; fArray[18] = 0.7644f;
                    fArray[1] = 0.8670f; fArray[7] = 0.5061f; fArray[13] = 0.2307f; fArray[19] = 0.1249f;
                    fArray[2] = 0.1215f; fArray[8] = 0.0781f; fArray[14] = 0.4038f; fArray[20] = 0.7689f;
                    fArray[3] = 0.3954f; fArray[9] = 0.1296f; fArray[15] = 0.2550f; fArray[21] = 0.9258f;
                    fArray[4] = 0.4396f; fArray[10] = 0.0897f; fArray[16] = 0.5008f; fArray[22] = 0.1512f;
                    fArray[5] = 0.0616f; fArray[11] = 0.0138f; fArray[17] = 0.8768f; fArray[23] = 0.9313f;
                    GPUMatrix<float> D(6, 4, 0 /*deviceId*/, fArray);

                    GPUMatrix<float> c(0 /*deviceId*/);
                    c.AssignKhatriRaoProductOf(a, b);
                    BOOST_CHECK(c.IsEqualTo(D, 0.0001f));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixAddColumnReshapeProductOf)
                {
                    // tests column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
                    float *fArray = new float[12];
                    fArray[0] = 0.6557f; fArray[6] = 0.7431f;
                    fArray[1] = 0.0357f; fArray[7] = 0.3922f;
                    fArray[2] = 0.8491f; fArray[8] = 0.6555f;
                    fArray[3] = 0.9340f; fArray[9] = 0.1712f;
                    fArray[4] = 0.6787f; fArray[10] = 0.7060f;
                    fArray[5] = 0.7577f; fArray[11] = 0.0318f;
                    GPUMatrix<float> a(6, 2, 0 /*deviceId*/, fArray);

                    fArray[0] = 0.2769f; fArray[3] = 0.8235f;
                    fArray[1] = 0.0462f; fArray[4] = 0.6948f;
                    fArray[2] = 0.0971f; fArray[5] = 0.3171f;
                    GPUMatrix<float> b(3, 2, 0 /*deviceId*/, fArray);

                    fArray[0] = 0.2867f; fArray[2] = 1.2913f;
                    fArray[1] = 0.1266f; fArray[3] = 0.4520f;
                    GPUMatrix<float> D0(2, 2, 0 /*deviceId*/, fArray);

                    fArray[0] = 0.2657f; fArray[2] = 1.0923f;
                    fArray[1] = 0.3636f; fArray[3] = 0.6416f;
                    GPUMatrix<float> D1(2, 2, 0 /*deviceId*/, fArray);

                    GPUMatrix<float> c(2, 2, 0 /*deviceId*/);
                    c.SetValue(0.0f);
                    c.AddColumnReshapeProductOf(a, b, false);
                    BOOST_CHECK(c.IsEqualTo(D0, 0.0001f));

                    c.SetValue(0.0f);
                    c.AddColumnReshapeProductOf(a, b, true);
                    BOOST_CHECK(c.IsEqualTo(D1, 0.0001f));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixInnerProduct)
                {
                    float *fArray = new float[6];
                    fArray[0] = 1; fArray[2] = 2; fArray[4] = 3;
                    fArray[1] = 4; fArray[3] = 5; fArray[5] = 6;
                    GPUMatrix<float> m0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    GPUMatrix<float> m1(0 /*deviceId*/), m2(0 /*deviceId*/);
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
                    float *fArray = new float[36];
                    fArray[0] = 1; fArray[2] = 6; fArray[4] = 11;
                    fArray[1] = 2; fArray[3] = 7; fArray[5] = 12;
                    GPUMatrix<float> m0(2, 3, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    GPUMatrix<float>  m1(0 /*deviceId*/);
                    m1.AssignRepeatOf(m0, 1, 1);
                    BOOST_CHECK(m1.IsEqualTo(m0, 0.0001f));

                    fArray[0] = 1; fArray[0 + 6] = 6; fArray[0 + 12] = 11; fArray[0 + 18] = 1; fArray[0 + 24] = 6; fArray[0 + 30] = 11;
                    fArray[1] = 2; fArray[1 + 6] = 7; fArray[1 + 12] = 12; fArray[1 + 18] = 2; fArray[1 + 24] = 7; fArray[1 + 30] = 12;
                    fArray[2] = 1; fArray[2 + 6] = 6; fArray[2 + 12] = 11; fArray[2 + 18] = 1; fArray[2 + 24] = 6; fArray[2 + 30] = 11;
                    fArray[3] = 2; fArray[3 + 6] = 7; fArray[3 + 12] = 12; fArray[3 + 18] = 2; fArray[3 + 24] = 7; fArray[3 + 30] = 12;
                    fArray[4] = 1; fArray[4 + 6] = 6; fArray[4 + 12] = 11; fArray[4 + 18] = 1; fArray[4 + 24] = 6; fArray[4 + 30] = 11;
                    fArray[5] = 2; fArray[5 + 6] = 7; fArray[5 + 12] = 12; fArray[5 + 18] = 2; fArray[5 + 24] = 7; fArray[5 + 30] = 12;
                    GPUMatrix<float> m3(6, 6, 0 /*deviceId*/, fArray, matrixFlagNormal);

                    m1.AssignRepeatOf(m0, 3, 2);
                    BOOST_CHECK(m1.IsEqualTo(m3, 0.0001f));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixRowElementOperations)
                {
                    GPUMatrix<float>   m0 = GPUMatrix<float>::RandomUniform(20, 28, 0 /*deviceId*/, -1, 1);
                    GPUMatrix<float>   m1 = GPUMatrix<float>::RandomUniform(1, 28, 0 /*deviceId*/, 1, 2);

                    GPUMatrix<float>   m3(0 /*deviceId*/);
                    m3.SetValue(m0);
                    m3.RowElementMultiplyWith(m1);
                    m3.RowElementDivideBy(m1);

                    BOOST_CHECK(m0.IsEqualTo(m3, 0.0001f));
                }

                BOOST_AUTO_TEST_CASE(GPUMatrixColumnElementOperations)
                {
                    GPUMatrix<float>   m0 = GPUMatrix<float>::RandomUniform(20, 28, 0 /*deviceId*/, -1, 1);
                    GPUMatrix<float>   m1 = GPUMatrix<float>::RandomUniform(20, 1, 0 /*deviceId*/, 1, 2);

                    GPUMatrix<float>   m3(0 /*deviceId*/);
                    m3.SetValue(m0);
                    m3.ColumnElementMultiplyWith(m1);
                    m3.ColumnElementDivideBy(m1);

                    BOOST_CHECK(m0.IsEqualTo(m3, 0.0001f));
                }

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}