//
// <copyright file="MathPerformanceTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// MathPerformanceTests.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <Windows.h>
#include <chrono> 
#include <iostream>
#include <vector>
#include "Matrix.h"
#include "CPUMatrix.h"
#include "Sequences.h"
using namespace Microsoft::MSR::CNTK;
using namespace std;

template<class ElemType>
void SetToInitStateValueForResetSeg(const Matrix<ElemType>& sentenceBegin,
    size_t nStream, ElemType initStateValue, Matrix<ElemType>& newprevstate)
{
    Matrix<ElemType> colSeg(sentenceBegin.GetDeviceId());
    colSeg.Resize(nStream, nStream);
    size_t nStateRow = newprevstate.GetNumRows();

    assert(nStream == sentenceBegin.GetNumRows());

    /// only set state to init state value for segmentation = 0, and -1
    /// e.g., -1 0 1 -> 0 0 1 -> 0 0 -1 -> 1 1 0 

    Matrix<ElemType> colPos(sentenceBegin.GetDeviceId());
    colPos.SetValue(sentenceBegin); /// -1 0 1
    colPos.InplaceTruncateBottom((int)MinibatchPackingFlags::SequenceStart);
    Matrix<ElemType>::Scale((ElemType)-1.0, colPos); 
    colPos += (int)MinibatchPackingFlags::None;
    colSeg.SetDiagonalValue(colPos);  
    Matrix<ElemType> ones(sentenceBegin.GetDeviceId());
    ones.Resize(nStateRow, nStream);
    ones.SetValue((ElemType)1);
    /// add default state value if it is for reset
    Matrix<ElemType>::MultiplyAndWeightedAdd(initStateValue, ones, false, colSeg, false, 1.0, newprevstate);  /// += [0 initStateValue 0 ]
}

template<class ElemType>
void rnnEvaluateThisNodeSRP(Matrix<ElemType>& functionValues, size_t mNbr, Matrix<ElemType>& pastActivity, Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& colBegin, const Matrix<ElemType>& needToCompute)
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
            colSegPastActivity.InplaceTruncateBottom((int)MinibatchPackingFlags::SequenceStart);
            colSeg.SetDiagonalValue(colSegPastActivity);
            Matrix<ElemType>::Multiply(inp, false, colSeg, false, out);
            ElemType initStateValue = (ElemType) 0.1;
            SetToInitStateValueForResetSeg<ElemType>(colBegin, mNbr, initStateValue, out);
        }
    }
}

template<class ElemType>
void oldRnnEvaluateThisNodeSRP(Matrix<ElemType>& functionValues, size_t mNbr, Matrix<ElemType>& pastActivity, Matrix<ElemType>& inputFunctionValues)
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
            oldRNNEvaluateThisNodeSRP<ElemType>(timeIdxInSeq, 1, reset, (ElemType) 0.1, functionValues, pastActivity, inputFunctionValues, i, mNbr);
        }
    }
}

template<class ElemType>
void oldRNNEvaluateThisNodeSRP(const size_t timeIdxInSeq, const int delay, const bool reset, const ElemType default_activity, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pastActivity, const Matrix<ElemType>& inputFunctionValues, const size_t indexInBatch, const size_t mNbr)
{
    assert(delay > 0);

    if (functionValues.GetNumRows() != inputFunctionValues.GetNumRows() ||
        functionValues.GetNumCols() != inputFunctionValues.GetNumCols())
        functionValues.Resize(inputFunctionValues.GetNumRows(),
        inputFunctionValues.GetNumCols());

    int iPastIndex = (int)((int) timeIdxInSeq - (int)delay) * (int)mNbr;
    int d = iPastIndex;
    if (d < 0)
        d = (int)functionValues.Mod((float)iPastIndex, (float)pastActivity.GetNumCols());
    /// this can point to the past activity of the previous mninibatch

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
        out.SetValue(inp);
    }
}

/**
The new way of resetting RNN state. 
*/
template<class ElemType>
void TestRnnEvaluateThisNodeSRP(size_t nRow = 100, size_t nCol = 1000, size_t mNbr = 10, DEVICEID_TYPE deviceID = 0)
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
    rnnEvaluateThisNodeSRP<ElemType>(functionValues, mNbr, pastActivity, inputFunctionValues, colBegin, needToCompute);
    auto t_end = clock();
    std::cout << "testRnnEvaluateThisNodeSRP: " << 1.0*(t_end - t_start) / CLOCKS_PER_SEC << " seconds" << endl;
}

/**
The old way of resetting RNN state, which used if statement. Also only supports up to two sentences within a minibatch
*/
template<class ElemType>
void TestOldRnnEvaluateThisNodeSRP(size_t nRow = 100, size_t nCol = 1000, size_t mNbr = 10, DEVICEID_TYPE deviceID = 0)
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
    oldRnnEvaluateThisNodeSRP<ElemType>(functionValues, mNbr, pastActivity, inputFunctionValues);
    auto t_end = clock();
    std::cout << "TestOldRnnEvaluateThisNodeSRP: " << 1.0*(t_end - t_start) / CLOCKS_PER_SEC << " seconds" << endl;
}

template<class ElemType>
void randomInitializeCPUMatrix(CPUMatrix<ElemType> &M, float min=-10, float max=10)
{
    foreach_coord(i,j,M)
    {
        M(i,j)=(1.0*rand()/RAND_MAX)*max+min;
    }
}

template<class ElemType>
void randomInitializeMatrix(Matrix<ElemType> &M, float min=-10, float max=10)
{
    foreach_coord(i,j,M)
    {
        M(i,j)=(1.0*rand()/RAND_MAX)*max+min;
    }
}

template<class ElemType>
void MultiplyAndWeightedAddTest(int n, int k, int m)
{
    cout<<"Testing CPUMatrix"<<endl;
    cout<<"A("<<n<<"x"<<k<<") and B("<<k<<","<<m<<")"<<endl;
    CPUMatrix<ElemType> A(n,k);
    randomInitializeCPUMatrix<ElemType>(A);
    CPUMatrix<ElemType> B(k,m);
    randomInitializeCPUMatrix<ElemType>(B);
    CPUMatrix<ElemType> C(n,m);
    auto t_start = clock();
    CPUMatrix<ElemType>::MultiplyAndWeightedAdd(0.324,A,false,B,false,0.632,C);
    auto t_end = clock();
    std::cout<<"CPU Matrix in: "<<1.0*(t_end-t_start)/CLOCKS_PER_SEC<<" seconds"<<endl;
    std::cout<<n<<" "<<k<<" "<<m<<endl;


    cout<<"Testing Matrix"<<endl;
    Matrix<ElemType> AG((size_t)n,(size_t)k);
    randomInitializeMatrix<ElemType>(AG);
    Matrix<ElemType> BG((size_t)k,(size_t)m);
    randomInitializeMatrix<ElemType>(BG);
    Matrix<ElemType> CG((size_t)n,(size_t)m);
    auto t_startG = clock();
    Matrix<ElemType>::MultiplyAndWeightedAdd(0.324,AG,false,BG,false,0.632,CG);
    auto t_endG = clock();
    std::cout<<"Matrix in: "<<1.0*(t_endG-t_startG)/CLOCKS_PER_SEC<<" seconds"<<endl;
}

template<class ElemType>
void AddMultiplyAndInplaceSigmoidTest(int n, int k, int m)
{
    cout<<"Testing CPUMatrix"<<endl;
    cout<<"A("<<n<<"x"<<k<<") and B("<<k<<","<<m<<")"<<endl;
    CPUMatrix<ElemType> A(n,k);
    randomInitializeCPUMatrix<ElemType>(A);
    CPUMatrix<ElemType> B(k,m);
    randomInitializeCPUMatrix<ElemType>(B);
    CPUMatrix<ElemType> D(n,m);
    randomInitializeCPUMatrix<ElemType>(D);
    CPUMatrix<ElemType> C(n,m);
    auto t_start = clock();
    C=A*B+D;
    //C.InplaceSigmoid();
    auto t_end = clock();
    std::cout<<"CPU Matrix in: "<<1.0*(t_end-t_start)/CLOCKS_PER_SEC<<" seconds"<<endl;
    std::cout<<n<<" "<<k<<" "<<m<<endl;


    cout<<"Testing Matrix"<<endl;
    Matrix<ElemType> AG((size_t)n,(size_t)k);
    randomInitializeMatrix<ElemType>(AG);
    Matrix<ElemType> BG((size_t)k,(size_t)m);
    randomInitializeMatrix<ElemType>(BG);
    Matrix<ElemType> DG((size_t)n,(size_t)m);
    randomInitializeMatrix<ElemType>(DG);
    Matrix<ElemType> CG((size_t)n,(size_t)m);
    auto t_startG = clock();
    CG=AG*BG+DG;
    //CG.InplaceSigmoid();
    auto t_endG = clock();
    std::cout<<"Matrix in: "<<1.0*(t_endG-t_startG)/CLOCKS_PER_SEC<<" seconds"<<endl;
}

template<class ElemType>
void ColumnSliceMultAndAddTest(int n, int k, int m, DEVICEID_TYPE deviceID)
{
    cout<<"Testing Matrix"<<endl;

    Matrix<ElemType> AG((size_t)n,(size_t)k, deviceID);
    AG.SetUniformRandomValue(-1,1);

    Matrix<ElemType> BG((size_t)k,(size_t)m, deviceID);
    BG.SetUniformRandomValue(-1,1);

    Matrix<ElemType> CG((size_t)n,(size_t)m, deviceID);
    Matrix<ElemType> DG((size_t)n,(size_t)m, deviceID);

    auto t_startG = clock();
    Matrix<ElemType>::MultiplyAndAdd(AG, false, BG, false, CG);
    auto t_endG = clock();
    std::cout<<"MultiplyAndAdd Directly: "<<1.0*(t_endG-t_startG)/CLOCKS_PER_SEC<<" seconds"<<endl;

    t_startG = clock();
    for (int i=0; i<m; i++)
    {
        Matrix<ElemType> col_BG = BG.ColumnSlice(i,1);
        Matrix<ElemType> col_CG = CG.ColumnSlice(i,1);
        Matrix<ElemType>::MultiplyAndAdd(AG, false, col_BG, false, col_CG);
    }
    t_endG = clock();
    std::cout<<"MultiplyAndAdd With ColumnSlice: "<<1.0*(t_endG-t_startG)/CLOCKS_PER_SEC<<" seconds"<<endl;

    t_startG = clock();
    for (int i=0; i<m; i++)
    {
        Matrix<ElemType> col_BG = BG.ColumnSlice(i,1);
        Matrix<ElemType> col_CG = CG.ColumnSlice(i,1);
        Matrix<ElemType>::MultiplyAndAdd(AG, false, col_BG, false, col_CG);
    }
    t_endG = clock();
    std::cout<<"MultiplyAndAdd With ColumnSlice&: "<<1.0*(t_endG-t_startG)/CLOCKS_PER_SEC<<" seconds"<<endl;

    Matrix<ElemType> col_BG1, col_CG1;
    t_startG = clock();
    for (int i=0; i<m; i++)
    {
        col_BG1.AssignColumnSlice(BG, i,1);
        col_CG1.AssignColumnSlice(CG, i,1);
        Matrix<ElemType>::MultiplyAndAdd(AG, false, col_BG1, false, col_CG1);
    }
    t_endG = clock();
    std::cout<<"MultiplyAndAdd With AssignColumnSlice: "<<1.0*(t_endG-t_startG)/CLOCKS_PER_SEC<<" seconds"<<endl;

    t_startG = clock();
    for (int i=0; i<m; i++)
    {
        Matrix<ElemType> col_CG = CG.ColumnSlice(i,1);
        Matrix<ElemType> col_DG = DG.ColumnSlice(i,1);
        col_DG.AssignSigmoidOf(col_CG);
    }
    t_endG = clock();
    std::cout<<"AssignSigmoidOf With ColumnSlice: "<<1.0*(t_endG-t_startG)/CLOCKS_PER_SEC<<" seconds"<<endl;

    t_startG = clock();
    for (int i=0; i<m; i++)
    {
        col_BG1.AssignColumnSlice(BG, i,1);
        col_CG1.AssignColumnSlice(CG, i,1);
        col_BG1.AssignSigmoidOf(col_CG1);
    }
    t_endG = clock();
    std::cout<<"AssignSigmoidOf With AssignColumnSlice: "<<1.0*(t_endG-t_startG)/CLOCKS_PER_SEC<<" seconds"<<endl;
}

template<class ElemType>
void SquareMultiplyAndAdd10TimesAvgTest(int n, int count)
{
    cout<<"Testing CPUMatrix"<<endl;
    cout<<"A("<<n<<"x"<<n<<") and B("<<n<<","<<n<<")"<<endl;

    double cpu_avg=0;
    for (int i=0;i<count;++i)
    {
        CPUMatrix<ElemType> A(n,n);
        randomInitializeCPUMatrix<ElemType>(A);
        CPUMatrix<ElemType> B(n,n);
        randomInitializeCPUMatrix<ElemType>(B);
        CPUMatrix<ElemType> C(n,n);
        auto t_start = clock();
        CPUMatrix<ElemType>::MultiplyAndWeightedAdd(0.324,A,false,B,false,0.632,C);
        auto t_end = clock();
        double val=1.0*(t_end-t_start)/CLOCKS_PER_SEC;
        if (i==0)
        {
            cpu_avg=val;
        }
        else
        {
            cpu_avg=cpu_avg*(i-1)/i+val/i;
        }
    }

    cout<<"Testing Matrix"<<endl;
    double m_avg=0;
    for (int i=0;i<count;++i)
    {    
        Matrix<ElemType> AG((size_t)n,(size_t)n);
        randomInitializeMatrix<ElemType>(AG);
        Matrix<ElemType> BG((size_t)n,(size_t)n);
        randomInitializeMatrix<ElemType>(BG);
        Matrix<ElemType> CG((size_t)n,(size_t)n);
        auto t_startG = clock();
        Matrix<ElemType>::MultiplyAndWeightedAdd(0.324,AG,false,BG,false,0.632,CG);
        auto t_endG = clock();
        double val=1.0*(t_endG-t_startG)/CLOCKS_PER_SEC;
        if (i==0)
        {
            m_avg=val;
        }
        else
        {
            m_avg=m_avg*(i-1)/i+val/i;
        }
    }

    cout<<"Based on "<<count<<" runs:"<<endl;
    cout<<"Average time for CPUMatrix is: "<<cpu_avg<<" seconds"<<endl;
    cout<<"Average time for Matrix is: "<<m_avg<<" seconds"<<endl;
    cout<<"CPUMatrix/Matrix ratio is: "<<cpu_avg/m_avg<<" seconds"<<endl;
}


template<class ElemType>
void MandSTest(int count, int devId)
{
    ElemType *arr = new ElemType[count];
    for (int i=0;i<count;++i) arr[i]=(1.0*rand())/RAND_MAX+1;

    ElemType *data1 = new ElemType[1024*4096];
    for (int i=0;i<1024*4096;++i) data1[i]=(1.0*rand())/RAND_MAX+1;

    ElemType *data2 = new ElemType[1024*4096];
    for (int i=0;i<1024*4096;++i) data2[i]=(1.0*rand())/RAND_MAX+1;

    ElemType *data3 = new ElemType[1024];
    for (int i=0;i<1024;++i) data3[i]=(1.0*rand())/RAND_MAX+1;

    cout<<"Testing Matrix"<<endl;
    Matrix<ElemType> A(1024,4096,data1,true,devId);
    Matrix<ElemType> B(4096,1024,data2,true,devId);
    Matrix<ElemType> V(1024,1,data3,true,devId);
    Matrix<ElemType> C(1024,1024,devId);
        
    auto t_startG = clock();
    for (int i=0;i<count;++i)
    {        
        //Matrix<ElemType>::MultiplyAndWeightedAdd(arr[i],A,false,B,false,3.2*arr[i],C);
        C+=(A*arr[i])*(B*(arr[i]*2.3));        
    }
    auto t_endG = clock();
    double valM=1.0*(t_endG-t_startG)/(CLOCKS_PER_SEC*count);
    cout<<"Matrix C+=alpha*A*beta*B in: "<<valM<<" seconds"<<endl;

    cout<<"Testing CPUMatrix"<<endl;
    CPUMatrix<ElemType> AC(1024,4096,data1,true);
    CPUMatrix<ElemType> BC(4096,1024,data2,true);
    CPUMatrix<ElemType> VC(1024,1,data3,true);
    CPUMatrix<ElemType> CC(1024,1024);
        
    auto t_startC = clock();
    for (int i=0;i<count;++i)
    {
        CPUMatrix<ElemType>::MultiplyAndWeightedAdd(arr[i],AC,false,BC,false,3.2*arr[i],CC);
        //CC+=(arr[i]*AC)*((arr[i]*2.3)*BC);        
    }
    auto t_endC = clock();
    double valMC=1.0*(t_endC-t_startC)/(CLOCKS_PER_SEC*count);
    cout<<"CPUMatrix C+=alpha*A*beta*B in: "<<valMC<<" seconds"<<endl;
    



    delete[] arr;
    delete[] data1;
    delete[] data2;
    delete[] data3;
}

int wmain()
{
    ColumnSliceMultAndAddTest<float>(2048, 2048, 256, 0);

    TestRnnEvaluateThisNodeSRP<float>();

    TestOldRnnEvaluateThisNodeSRP<float>();

    //MandSTest<float>(100, 2);

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

