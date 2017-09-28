//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// MathPerformanceTests.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
//#define NOMINMAX
//#include "Windows.h"
#include "Matrix.h"
#include "CPUMatrix.h"
#include "TensorView.h"
#include "Sequences.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace Microsoft::MSR::CNTK;
using namespace std;

template <class ElemType>
void randomInitializeCPUMatrix(CPUMatrix<ElemType>& M, float min = -10, float max = 10)
{
    foreach_coord (i, j, M)
    {
        M(i, j) = (1.0 * rand() / RAND_MAX) * max + min;
    }
}

template <class ElemType>
void randomInitializeMatrix(Matrix<ElemType>& M, float min = -10, float max = 10)
{
    foreach_coord (i, j, M)
    {
        M(i, j) = (1.0 * rand() / RAND_MAX) * max + min;
    }
}

template <class ElemType>
void MultiplyAndWeightedAddTest(int n, int k, int m)
{
    cout << "Testing CPUMatrix" << endl;
    cout << "A(" << n << "x" << k << ") and B(" << k << "," << m << ")" << endl;
    CPUMatrix<ElemType> A(n, k);
    randomInitializeCPUMatrix<ElemType>(A);
    CPUMatrix<ElemType> B(k, m);
    randomInitializeCPUMatrix<ElemType>(B);
    CPUMatrix<ElemType> C(n, m);
    auto t_start = clock();
    CPUMatrix<ElemType>::MultiplyAndWeightedAdd(0.324, A, false, B, false, 0.632, C);
    auto t_end = clock();
    std::cout << "CPU Matrix in: " << 1.0 * (t_end - t_start) / CLOCKS_PER_SEC << " seconds" << endl;
    std::cout << n << " " << k << " " << m << endl;

    cout << "Testing Matrix" << endl;
    Matrix<ElemType> AG((size_t) n, (size_t) k);
    randomInitializeMatrix<ElemType>(AG);
    Matrix<ElemType> BG((size_t) k, (size_t) m);
    randomInitializeMatrix<ElemType>(BG);
    Matrix<ElemType> CG((size_t) n, (size_t) m);
    auto t_startG = clock();
    Matrix<ElemType>::MultiplyAndWeightedAdd(0.324, AG, false, BG, false, 0.632, CG);
    auto t_endG = clock();
    std::cout << "Matrix in: " << 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC << " seconds" << endl;
}

template <class ElemType>
void AddMultiplyAndInplaceSigmoidTest(int n, int k, int m)
{
    cout << "Testing CPUMatrix" << endl;
    cout << "A(" << n << "x" << k << ") and B(" << k << "," << m << ")" << endl;
    CPUMatrix<ElemType> A(n, k);
    randomInitializeCPUMatrix<ElemType>(A);
    CPUMatrix<ElemType> B(k, m);
    randomInitializeCPUMatrix<ElemType>(B);
    CPUMatrix<ElemType> D(n, m);
    randomInitializeCPUMatrix<ElemType>(D);
    CPUMatrix<ElemType> C(n, m);
    auto t_start = clock();
    C = A * B + D;
    // C.InplaceSigmoid();
    auto t_end = clock();
    std::cout << "CPU Matrix in: " << 1.0 * (t_end - t_start) / CLOCKS_PER_SEC << " seconds" << endl;
    std::cout << n << " " << k << " " << m << endl;

    cout << "Testing Matrix" << endl;
    Matrix<ElemType> AG((size_t) n, (size_t) k);
    randomInitializeMatrix<ElemType>(AG);
    Matrix<ElemType> BG((size_t) k, (size_t) m);
    randomInitializeMatrix<ElemType>(BG);
    Matrix<ElemType> DG((size_t) n, (size_t) m);
    randomInitializeMatrix<ElemType>(DG);
    Matrix<ElemType> CG((size_t) n, (size_t) m);
    auto t_startG = clock();
    CG = AG * BG + DG;
    // CG.InplaceSigmoid();
    auto t_endG = clock();
    std::cout << "Matrix in: " << 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC << " seconds" << endl;
}

template <class ElemType>
void SquareMultiplyAndAdd10TimesAvgTest(int n, int count)
{
    cout << "Testing CPUMatrix" << endl;
    cout << "A(" << n << "x" << n << ") and B(" << n << "," << n << ")" << endl;

    double cpu_avg = 0;
    for (int i = 0; i < count; ++i)
    {
        CPUMatrix<ElemType> A(n, n);
        randomInitializeCPUMatrix<ElemType>(A);
        CPUMatrix<ElemType> B(n, n);
        randomInitializeCPUMatrix<ElemType>(B);
        CPUMatrix<ElemType> C(n, n);
        auto t_start = clock();
        CPUMatrix<ElemType>::MultiplyAndWeightedAdd(0.324, A, false, B, false, 0.632, C);
        auto t_end = clock();
        double val = 1.0 * (t_end - t_start) / CLOCKS_PER_SEC;
        if (i == 0)
        {
            cpu_avg = val;
        }
        else
        {
            cpu_avg = cpu_avg * (i - 1) / i + val / i;
        }
    }

    cout << "Testing Matrix" << endl;
    double m_avg = 0;
    for (int i = 0; i < count; ++i)
    {
        Matrix<ElemType> AG((size_t) n, (size_t) n);
        randomInitializeMatrix<ElemType>(AG);
        Matrix<ElemType> BG((size_t) n, (size_t) n);
        randomInitializeMatrix<ElemType>(BG);
        Matrix<ElemType> CG((size_t) n, (size_t) n);
        auto t_startG = clock();
        Matrix<ElemType>::MultiplyAndWeightedAdd(0.324, AG, false, BG, false, 0.632, CG);
        auto t_endG = clock();
        double val = 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC;
        if (i == 0)
        {
            m_avg = val;
        }
        else
        {
            m_avg = m_avg * (i - 1) / i + val / i;
        }
    }

    cout << "Based on " << count << " runs:" << endl;
    cout << "Average time for CPUMatrix is: " << cpu_avg << " seconds" << endl;
    cout << "Average time for Matrix is: " << m_avg << " seconds" << endl;
    cout << "CPUMatrix/Matrix ratio is: " << cpu_avg / m_avg << " seconds" << endl;
}

// simple test suite for TensorView
//  - this is meant for performance optimization
//  - correctness is defined as same result between GPU and CPU
template <class ElemType>
struct TensorTest
{
    // helper to create a randomly initialized tensor object
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

    // test bias gradient (reduction)
    static TensorView<ElemType> BiasGradientTest(TensorShape layerShape, TensorShape biasShape, DEVICEID_TYPE deviceId)
    {
        int randomSeed = 1;
        let  gradient = CreateTensor(layerShape, randomSeed++, deviceId);
        auto bias = CreateTensor(biasShape, randomSeed++, deviceId, true);
        //gradient.GetSOB().Print("incoming gradient", 0, 9, 0, 9);
        //bias.GetSOB().Print("bias gradient", 0, 9, 0, 9);
        bias.DoCopyOf(1, gradient, 1);
        //bias.GetSOB().Print("updated bias gradient", 0, 9, 0, 9);
        return bias;
    }

    // test broadcast summation gradient
    static TensorView<ElemType> BroadcastingTest(TensorShape layerShape, TensorShape biasShape, DEVICEID_TYPE deviceId)
    {
        int randomSeed = 1;
        let  input  = CreateTensor(layerShape, randomSeed++, deviceId);
        auto bias   = CreateTensor(biasShape,  randomSeed++, deviceId);
        //input.GetSOB().Print("input data", 0, 9, 0, 9);
        //bias.GetSOB().Print("bias", 0, 9, 0, 9);
        auto result = CreateTensor(layerShape, randomSeed++, deviceId, true);
        result.AssignSumOf(input, bias);
        return result;
    }

    // run one test for both GPU and CPU and verify they are the same
    template<typename FN>
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
        cout << (isSame ? " --> SUCCEEDED. =====\n" : " --> FAILED (GPU and CPU results differ). =====\n") << endl << flush;
        if (!isSame)
            sin(1.0);  // set breakpoint here
    }

    // main entry point (misusing the constructor)
    /*void*/ TensorTest()
    {
        // --- elementwise

        // elementwise sum
        OneTensorTest("elementwise addition", 1e-8, [](DEVICEID_TYPE deviceId) -> TensorView<ElemType>
        {
            return BroadcastingTest(TensorShape{ 512, 256 }, TensorShape({ 512, 256 }), deviceId);
        });

        // --- broadcasting

        // simple broadcasting
        OneTensorTest("addition wth simple broadcasting", 1e-8, [](DEVICEID_TYPE deviceId) -> TensorView<ElemType>
        {
            return BroadcastingTest(TensorShape{ 3, 2 }, TensorShape({ 3, 1 }), deviceId);
        });
        // typical bias for convolutional layer
        OneTensorTest("bias addition (broadcasting)", 1e-8, [](DEVICEID_TYPE deviceId) -> TensorView<ElemType>
        {
            return BroadcastingTest(TensorShape{ 28, 28, 128, 32 }, TensorShape({ 1, 1, 128 }), deviceId);
        });
        // BUGBUG: This test is strange--Print() shows different values with depth 128 instead of 64, but IsEqual() does not fail with 1e-3 tolerance.
        //         Something fishy going on. Dimension overflow?
        OneTensorTest("bias addition (broadcasting)", 1e-8, [](DEVICEID_TYPE deviceId) -> TensorView<ElemType>
        {
            return BroadcastingTest(TensorShape{ 256, 256, 64, 32 }, TensorShape({ 1, 1, 64 }), deviceId);
        });

        // --- reduction

        // typical bias gradient (reduction) for FF-DNN
        OneTensorTest("bias gradient (reduction)", 1e-4, [](DEVICEID_TYPE deviceId) -> TensorView<ElemType>
        {
            return BiasGradientTest(TensorShape{ 2048, 1024 }, TensorShape(2048), deviceId);
        });
        // typical bias gradient (reduction) for convolutional layer
        OneTensorTest("bias gradient (reduction)", 1e-1, [](DEVICEID_TYPE deviceId) -> TensorView<ElemType>
        {
            return BiasGradientTest(TensorShape{ 256, 256, 64, 32 }, TensorShape({ 1, 1, 64 }), deviceId);
        });
    }
};

template <class ElemType>
void MandSTest(int count, int devId)
{
    ElemType* arr = new ElemType[count];
    for (int i = 0; i < count; ++i)
        arr[i] = (1.0 * rand()) / RAND_MAX + 1;

    ElemType* data1 = new ElemType[1024 * 4096];
    for (int i = 0; i < 1024 * 4096; ++i)
        data1[i] = (1.0 * rand()) / RAND_MAX + 1;

    ElemType* data2 = new ElemType[1024 * 4096];
    for (int i = 0; i < 1024 * 4096; ++i)
        data2[i] = (1.0 * rand()) / RAND_MAX + 1;

    ElemType* data3 = new ElemType[1024];
    for (int i = 0; i < 1024; ++i)
        data3[i] = (1.0 * rand()) / RAND_MAX + 1;

    cout << "Testing Matrix" << endl;
    Matrix<ElemType> A(1024, 4096, data1, true, devId);
    Matrix<ElemType> B(4096, 1024, data2, true, devId);
    Matrix<ElemType> V(1024, 1, data3, true, devId);
    Matrix<ElemType> C(1024, 1024, devId);

    auto t_startG = clock();
    for (int i = 0; i < count; ++i)
    {
        // Matrix<ElemType>::MultiplyAndWeightedAdd(arr[i],A,false,B,false,3.2*arr[i],C);
        C += (A * arr[i]) * (B * (arr[i] * 2.3));
    }
    auto t_endG = clock();
    double valM = 1.0 * (t_endG - t_startG) / (CLOCKS_PER_SEC * count);
    cout << "Matrix C+=alpha*A*beta*B in: " << valM << " seconds" << endl;

    cout << "Testing CPUMatrix" << endl;
    CPUMatrix<ElemType> AC(1024, 4096, data1, true);
    CPUMatrix<ElemType> BC(4096, 1024, data2, true);
    CPUMatrix<ElemType> VC(1024, 1, data3, true);
    CPUMatrix<ElemType> CC(1024, 1024);

    auto t_startC = clock();
    for (int i = 0; i < count; ++i)
    {
        CPUMatrix<ElemType>::MultiplyAndWeightedAdd(arr[i], AC, false, BC, false, 3.2 * arr[i], CC);
        // CC+=(arr[i]*AC)*((arr[i]*2.3)*BC);
    }
    auto t_endC = clock();
    double valMC = 1.0 * (t_endC - t_startC) / (CLOCKS_PER_SEC * count);
    cout << "CPUMatrix C+=alpha*A*beta*B in: " << valMC << " seconds" << endl;

    delete[] arr;
    delete[] data1;
    delete[] data2;
    delete[] data3;
}

int wmain()
{
    // MandSTest<float>(100, 2);

    /*cout<<endl<<"********************Matrix SquareMultiplyAndWeightedAdd10TimesAvg TEST********************"<<endl;
    SquareMultiplyAndAdd10TimesAvgTest<float>(4096,10);

    cout<<endl<<"********************Matrix AddMultiplyAndInplaceSigmoid TEST********************"<<endl;    
    AddMultiplyAndInplaceSigmoidTest<float>(11,10,12);    
    AddMultiplyAndInplaceSigmoidTest<float>(110,100,120);    
    AddMultiplyAndInplaceSigmoidTest<float>(1100,1000,1200);    
    AddMultiplyAndInplaceSigmoidTest<float>(11000,10000,1200);

    cout<<endl<<"********************Matrix MultiplyAndWeightedAdd TEST********************"<<endl;    
    MultiplyAndWeightedAddTest<float>(11,10,12);    
    MultiplyAndWeightedAddTest<float>(110,100,120);    
    MultiplyAndWeightedAddTest<float>(1100,1000,1200);    
    MultiplyAndWeightedAddTest<float>(11000,10000,12000);*/

    return 0;
}
