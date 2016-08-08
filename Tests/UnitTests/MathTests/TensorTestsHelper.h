//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Matrix.h"
#include "CPUMatrix.h"
#include "TensorView.h"
#include "Sequences.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

template <class ElemType>
struct TensorTest
{
    // run one test for both GPU and CPU and verify they are the same
    template<typename FN>
    void OneTensorTest(const char* what, double tolerance, const FN& fn)
    {
        fprintf(stderr, "===== Tensor test '%s'\n", what);

        // run on GPU and CPU
        let resultGPU = fn(0);
        let resultCPU = fn(-1);

        // dump top corner of the result to get a feel for the error
        resultGPU.GetSOB().Print("GPU result", 0, 7, 0, 9);
        resultGPU.GetSOB().TransferToDeviceIfNotThere(-1, true, false, true);
        resultCPU.GetSOB().Print("CPU result", 0, 7, 0, 9);

        BOOST_CHECK(resultGPU.GetSOB().IsEqualTo(resultCPU.GetSOB(), (ElemType)tolerance));
    }

    // helper to create a randomly initialized tensor object
    TensorView<ElemType> CreateTensor(TensorShape shape, int randomSeed, DEVICEID_TYPE deviceId, bool isResult = false)
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
    TensorView<ElemType> BiasGradientTest(TensorShape layerShape, TensorShape biasShape, DEVICEID_TYPE deviceId)
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
    TensorView<ElemType> BroadcastingTest(TensorShape layerShape, TensorShape biasShape, DEVICEID_TYPE deviceId)
    {
        int randomSeed = 1;
        let  input = CreateTensor(layerShape, randomSeed++, deviceId);
        auto bias = CreateTensor(biasShape, randomSeed++, deviceId);
        //input.GetSOB().Print("input data", 0, 9, 0, 9);
        //bias.GetSOB().Print("bias", 0, 9, 0, 9);
        auto result = CreateTensor(layerShape, randomSeed++, deviceId, true);
        result.AssignSumOf(input, bias);
        return result;
    }
};

template <class ElemType>
void SetToInitStateValueForResetSeg(const Matrix<ElemType>& sentenceBegin, size_t nStream, ElemType initStateValue, Matrix<ElemType>& newprevstate)
{
    Matrix<ElemType> colSeg(sentenceBegin.GetDeviceId());
    colSeg.Resize(nStream, nStream);
    size_t nStateRow = newprevstate.GetNumRows();

    assert(nStream == sentenceBegin.GetNumRows());

    // only set state to init state value for segmentation = 0, and -1
    // e.g., -1 0 1 -> 0 0 1 -> 0 0 -1 -> 1 1 0

    Matrix<ElemType> colPos(sentenceBegin.GetDeviceId());
    colPos.SetValue(sentenceBegin);                                                     // -1 0 1
    colPos.InplaceTruncateBottom(1 << 0 /*(int)MinibatchPackingFlags::SequenceStart*/); // TODO: these flags no longer exist, this test probably no longer applies
    Matrix<ElemType>::Scale((ElemType)-1.0, colPos);
    colPos += 0; // (int)MinibatchPackingFlags::None; // TODO: these flags no longer exist, this test probably no longer applies
    colSeg.SetDiagonalValue(colPos);
    Matrix<ElemType> ones(sentenceBegin.GetDeviceId());
    ones.Resize(nStateRow, nStream);
    ones.SetValue((ElemType)1);
    // add default state value if it is for reset
    Matrix<ElemType>::MultiplyAndWeightedAdd(initStateValue, ones, false, colSeg, false, 1.0, newprevstate); // += [0 initStateValue 0 ]
}

template <class ElemType>
void rnnForwardPropSRP(Matrix<ElemType>& functionValues, size_t mNbr, Matrix<ElemType>& pastActivity, Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& colBegin, const Matrix<ElemType>& needToCompute)
{
    size_t ncol = functionValues.GetNumCols();
    size_t ntime = ncol / mNbr;
    Matrix<ElemType> out = functionValues.ColumnSlice(0, mNbr);
    Matrix<ElemType> inp((DEVICEID_TYPE)functionValues.GetDeviceId());

    for (size_t d = 0; d < ntime; d++)
    {
        if (d == 0)
            inp = pastActivity.ColumnSlice(d, mNbr);
        else
            inp = inputFunctionValues.ColumnSlice(d, mNbr);

        if (needToCompute.ColumnSlice(d, 1).Get00Element() == 1)
        {
            Matrix<ElemType> colSegPastActivity((DEVICEID_TYPE)functionValues.GetDeviceId());
            Matrix<ElemType> colSeg((DEVICEID_TYPE)functionValues.GetDeviceId());
            colSeg.Resize(mNbr, mNbr);
            colSeg.SetValue(0);
            colSegPastActivity.SetValue(colBegin);
            colSegPastActivity.InplaceTruncateBottom(1 << 0 /*(int)MinibatchPackingFlags::SequenceStart*/); // TODO: these flags no longer exist, this test probably no longer applies
            colSeg.SetDiagonalValue(colSegPastActivity);
            Matrix<ElemType>::Multiply(inp, false, colSeg, false, out);
            ElemType initStateValue = (ElemType) 0.1;
            SetToInitStateValueForResetSeg<ElemType>(colBegin, mNbr, initStateValue, out);
        }
    }
}

template <class ElemType>
void oldRNNForwardPropSRP(const size_t timeIdxInSeq, const int delay, const bool reset, const ElemType default_activity, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pastActivity, const Matrix<ElemType>& inputFunctionValues, const size_t indexInBatch, const size_t mNbr)
{
    assert(delay > 0);

    if (functionValues.GetNumRows() != inputFunctionValues.GetNumRows() ||
        functionValues.GetNumCols() != inputFunctionValues.GetNumCols())
        functionValues.Resize(inputFunctionValues.GetNumRows(),
        inputFunctionValues.GetNumCols());

    int iPastIndex = (int)((int)timeIdxInSeq - (int)delay) * (int)mNbr;
    int d = iPastIndex;
    if (d < 0)
        d = (int)functionValues.Mod((float)iPastIndex, (float)pastActivity.GetNumCols());
    // this can point to the past activity of the previous mninibatch

    Matrix<ElemType> out = functionValues.ColumnSlice(timeIdxInSeq * mNbr + indexInBatch, 1);
    Matrix<ElemType> inp((DEVICEID_TYPE)functionValues.GetDeviceId());

    if (reset)
        out.SetValue(default_activity);
    else
    {
        if (iPastIndex < 0)
            inp = pastActivity.ColumnSlice(d + indexInBatch, 1);
        else
            inp = inputFunctionValues.ColumnSlice(d + indexInBatch, 1);
        out.AssignValuesOf(inp);
    }
}

template <class ElemType>
void oldRnnForwardPropSRP(Matrix<ElemType>& functionValues, size_t mNbr, Matrix<ElemType>& pastActivity, Matrix<ElemType>& inputFunctionValues)
{
    size_t ncol = functionValues.GetNumCols();
    size_t ntime = ncol / mNbr;
    for (size_t timeIdxInSeq = 0; timeIdxInSeq < ntime; timeIdxInSeq++)
    {
        for (size_t i = 0; i < mNbr; i++)
        {
            bool reset = false;

            if (timeIdxInSeq == 0)
            {
                reset = true;
            }
            oldRNNForwardPropSRP<ElemType>(timeIdxInSeq, 1, reset, (ElemType) 0.1, functionValues, pastActivity, inputFunctionValues, i, mNbr);
        }
    }
}

template <class ElemType>
void ColumnSliceMultAndAddTest(int n, int k, int m, DEVICEID_TYPE deviceID)
{
    Matrix<ElemType> AG((size_t)n, (size_t)k, deviceID);
    AG.SetUniformRandomValue(-1, 1);

    Matrix<ElemType> BG((size_t)k, (size_t)m, deviceID);
    BG.SetUniformRandomValue(-1, 1);

    Matrix<ElemType> CG((size_t)n, (size_t)m, deviceID);
    Matrix<ElemType> DG((size_t)n, (size_t)m, deviceID);

    auto t_startG = clock();
    Matrix<ElemType>::MultiplyAndAdd(AG, false, BG, false, CG);
    auto t_endG = clock();

    fprintf(stderr, "MultiplyAndAdd Directly:  %f seconds\n", 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC);

    t_startG = clock();
    for (int i = 0; i < m; i++)
    {
        Matrix<ElemType> col_BG = BG.ColumnSlice(i, 1);
        Matrix<ElemType> col_CG = CG.ColumnSlice(i, 1);
        Matrix<ElemType>::MultiplyAndAdd(AG, false, col_BG, false, col_CG);
    }
    t_endG = clock();
    fprintf(stderr, "MultiplyAndAdd With ColumnSlice:  %f seconds\n", 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC);

    t_startG = clock();
    for (int i = 0; i < m; i++)
    {
        Matrix<ElemType> col_BG = BG.ColumnSlice(i, 1);
        Matrix<ElemType> col_CG = CG.ColumnSlice(i, 1);
        Matrix<ElemType>::MultiplyAndAdd(AG, false, col_BG, false, col_CG);
    }
    t_endG = clock();
    fprintf(stderr, "MultiplyAndAdd With ColumnSlice&:  %f seconds\n", 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC);

    Matrix<ElemType> col_BG1(0), col_CG1(0);
    t_startG = clock();
    for (int i = 0; i < m; i++)
    {
        col_BG1.AssignColumnSlice(BG, i, 1);
        col_CG1.AssignColumnSlice(CG, i, 1);
        Matrix<ElemType>::MultiplyAndAdd(AG, false, col_BG1, false, col_CG1);
    }
    t_endG = clock();
    fprintf(stderr, "MultiplyAndAdd With AssignColumnSlice:  %f seconds\n", 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC);

    t_startG = clock();
    for (int i = 0; i < m; i++)
    {
        Matrix<ElemType> col_CG = CG.ColumnSlice(i, 1);
        Matrix<ElemType> col_DG = DG.ColumnSlice(i, 1);
        col_DG.AssignSigmoidOf(col_CG);
    }
    t_endG = clock();
    fprintf(stderr, "AssignSigmoidOf With ColumnSlice:  %f seconds\n", 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC);

    t_startG = clock();
    for (int i = 0; i < m; i++)
    {
        col_BG1.AssignColumnSlice(BG, i, 1);
        col_CG1.AssignColumnSlice(CG, i, 1);
        col_BG1.AssignSigmoidOf(col_CG1);
    }
    t_endG = clock();
    fprintf(stderr, "AssignSigmoidOf With AssignColumnSlice:  %f seconds\n", 1.0 * (t_endG - t_startG) / CLOCKS_PER_SEC);
}

template <class ElemType>
void TestRnnForwardPropSRP(size_t nRow = 100, size_t nCol = 1000, size_t mNbr = 10, DEVICEID_TYPE deviceID = 0)
{
    Matrix<ElemType> functionValues(deviceID);
    Matrix<ElemType> colBegin(deviceID);
    Matrix<ElemType> pastActivity(deviceID);
    Matrix<ElemType> inputFunctionValues(deviceID);
    Matrix<ElemType> needToCompute(deviceID);

    functionValues.Resize(nRow, nCol);
    colBegin.Resize(mNbr, 1);
    pastActivity.Resize(nRow, nCol);
    inputFunctionValues.Resize(nRow, nCol);
    needToCompute.Resize(1, nCol / mNbr);
    needToCompute.SetValue(0);
    needToCompute.ColumnSlice(0, 1).SetValue(1);
    auto t_start = clock();
    rnnForwardPropSRP<ElemType>(functionValues, mNbr, pastActivity, inputFunctionValues, colBegin, needToCompute);
    auto t_end = clock();
    fprintf(stderr, "testRnnForwardPropSRP:  %f seconds\n", 1.0 * (t_end - t_start) / CLOCKS_PER_SEC);
}

/**
The old way of resetting RNN state, which used if statement. Also only supports up to two sentences within a minibatch
*/
template <class ElemType>
void TestOldRnnForwardPropSRP(size_t nRow = 100, size_t nCol = 1000, size_t mNbr = 10, DEVICEID_TYPE deviceID = 0)
{
    Matrix<ElemType> functionValues(deviceID);
    Matrix<ElemType> colBegin(deviceID);
    Matrix<ElemType> pastActivity(deviceID);
    Matrix<ElemType> inputFunctionValues(deviceID);

    functionValues.Resize(nRow, nCol);
    colBegin.Resize(mNbr, 1);
    pastActivity.Resize(nRow, nCol);
    inputFunctionValues.Resize(nRow, nCol);
    auto t_start = clock();
    oldRnnForwardPropSRP<ElemType>(functionValues, mNbr, pastActivity, inputFunctionValues);
    auto t_end = clock();
    fprintf(stderr, "TestOldRnnForwardPropSRP:  %f seconds\n", 1.0 * (t_end - t_start) / CLOCKS_PER_SEC);
}

}}}}
