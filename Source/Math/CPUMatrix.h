//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once
#include "Basics.h" // for RuntimeError()
#include "Matrix.h"
#include "File.h"
#include "Helpers.h"
#include "CommonMatrix.h"
#include <vector>
#include <stdio.h>
#include <ctime>
#include <limits.h>

// NOTE NOTE NOTE:
// use CPUSingleMatrix and CPUDoubleMatrix instead of using the template directly
///////////////////////////////////////////////

// This class is exported from the Math.dll
namespace Microsoft { namespace MSR { namespace CNTK {

double logadd(double x, double y);

//To compy with BLAS libraries matrices are stored in ColMajor. However, by default C/C++/C# use RowMajor
//convertion is need when passing data between CPUMatrix and C++ matrices
template <class ElemType>
class MATH_API CPUMatrix : public BaseMatrix<ElemType>
{
    typedef BaseMatrix<ElemType> B;
    using B::m_numRows;
    using B::m_numCols;
    using B::m_pArray;
    using B::m_computeDevice;
    using B::m_elemSizeAllocated;
    using B::m_externalBuffer;
    using B::m_format;
public:
    CPUMatrix();
    CPUMatrix(const size_t numRows, const size_t numCols);
    CPUMatrix(const size_t numRows, const size_t numCols, ElemType* pArray, const size_t matrixFlags = matrixFlagNormal);
    CPUMatrix(const CPUMatrix<ElemType>& deepCopyFrom);                      // copy constructor, deep copy
    CPUMatrix<ElemType>& operator=(const CPUMatrix<ElemType>& deepCopyFrom); // assignment operator, deep copy
    CPUMatrix(CPUMatrix<ElemType>&& moveFrom);                               // move constructor, shallow copy
    CPUMatrix<ElemType>& operator=(CPUMatrix<ElemType>&& moveFrom);          // move assignment operator, shallow copy

    ~CPUMatrix();

public:
    using B::OwnBuffer;
    using B::GetNumElements;
    using B::IsEmpty;
    using B::GetNumRows;
    using B::GetNumCols;
    using B::SetOwnBuffer;
    using B::VerifySize;

    size_t BufferSize() const
    {
        return m_numRows * m_numCols * sizeof(ElemType);
    }
    ElemType* BufferPointer() const
    {
        return m_pArray;
    }

    CPUMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;
    CPUMatrix<ElemType>& AssignColumnSlice(const CPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols);
    CPUMatrix<ElemType>& SetColumnSlice(const CPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols);

    void CopyColumnsStrided(const CPUMatrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride);

    CPUMatrix<ElemType> Diagonal() const;

    ElemType Adagrad(CPUMatrix<ElemType>& gradients, const bool needAveMultiplier);
    void FSAdagrad(CPUMatrix<ElemType>& gradients, CPUMatrix<ElemType>& functionValues, ElemType learnRatePerSample, ElemType momentum, ElemType adaWeight, ElemType adaMul);
    ElemType RmsProp(CPUMatrix<ElemType>& gradients,
                     ElemType RMS_GAMMA,
                     ElemType RMS_WGT_INC,
                     ElemType RMS_WGT_MAX,
                     ElemType RMS_WGT_DEC,
                     ElemType RMS_WGT_MIN,
                     const bool needAveMultiplier);

    void Reshape(const size_t numRows, const size_t numCols);
    void Resize(const size_t numRows, const size_t numCols, bool growOnly = true); // by default we only reallocate if need to grow
    ElemType* CopyToArray() const;                                                 // allocated by the callee but need to be deleted by the caller
    size_t CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const;    // allocated by the callee but need to be deleted by the caller
    void CopySection(size_t numRows, size_t numCols, ElemType* dst, size_t colStride) const;

    inline ElemType& operator()(const size_t row, const size_t col)
    {
        return m_pArray[LocateElement(row, col)];
    }
    inline const ElemType& operator()(const size_t row, const size_t col) const
    {
        return m_pArray[LocateElement(row, col)];
    }
    inline ElemType Get00Element() const
    {
        return m_pArray[0];
    }

    void SetValue(const ElemType v);
    void SetValue(const CPUMatrix<ElemType>& deepCopyFrom);
    void SetValue(const size_t numRows, const size_t numCols, ElemType* pArray, size_t matrixFlags = matrixFlagNormal);

    void MaskColumnsValue(const CPUMatrix<char>& columnsMask, ElemType val);

    void SetColumn(const ElemType* colPointer, size_t colInd);
    void SetColumn(const CPUMatrix<ElemType>& valMat, size_t colInd);
    void SetColumn(const ElemType val, size_t j);

    void SetDiagonalValue(const ElemType v);
    void SetDiagonalValue(const CPUMatrix<ElemType>& vector);
    void SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed = USE_TIME_BASED_SEED);
    void SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);
    void SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, unsigned long seed = USE_TIME_BASED_SEED);
    void AddGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);

    CPUMatrix<ElemType> Transpose();
    CPUMatrix<ElemType>& AssignTransposeOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& DoGatherColumnsOf (ElemType beta, const CPUMatrix<ElemType>& m, const CPUMatrix<ElemType>& a, ElemType alpha);
    CPUMatrix<ElemType>& DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& m, const CPUMatrix<ElemType>& a, ElemType alpha);

    CPUMatrix<ElemType>& operator+=(const ElemType alpha);
    CPUMatrix<ElemType>  operator+(const ElemType alpha) const;
    CPUMatrix<ElemType>& AssignSumOf(const ElemType alpha, const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& operator+=(const CPUMatrix<ElemType>& a);
    CPUMatrix<ElemType>  operator+(const CPUMatrix<ElemType>& a) const;
    CPUMatrix<ElemType>& AssignSumOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);

    CPUMatrix<ElemType>& operator-=(const ElemType alpha);
    CPUMatrix<ElemType>  operator-(const ElemType alpha) const;
    CPUMatrix<ElemType>& AssignDifferenceOf(const ElemType alpha, const CPUMatrix<ElemType>& a);
    CPUMatrix<ElemType>& AssignDifferenceOf(const CPUMatrix<ElemType>& a, const ElemType alpha);

    CPUMatrix<ElemType>& operator-=(const CPUMatrix<ElemType>& a);
    CPUMatrix<ElemType>  operator-(const CPUMatrix<ElemType>& a) const;
    CPUMatrix<ElemType>& AssignDifferenceOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);

    CPUMatrix<ElemType>& operator*=(const ElemType alpha);
    CPUMatrix<ElemType>  operator*(const ElemType alpha) const;
    CPUMatrix<ElemType>& AssignProductOf(const ElemType alpha, const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>  operator*(const CPUMatrix<ElemType>& a) const;
    CPUMatrix<ElemType>& AssignProductOf(const CPUMatrix<ElemType>& a, const bool transposeA, const CPUMatrix<ElemType>& b, const bool transposeB);

    CPUMatrix<ElemType>& operator/=(ElemType alpha);
    CPUMatrix<ElemType>  operator/(ElemType alpha) const;

    CPUMatrix<ElemType>& operator^=(ElemType alpha);     // element-wise power
    CPUMatrix<ElemType>  operator^(ElemType alpha) const; // element-wise power
    CPUMatrix<ElemType>& AssignElementPowerOf(const CPUMatrix<ElemType>& a, const ElemType power);

    CPUMatrix<ElemType>& ElementMultiplyWith(const CPUMatrix<ElemType>& a);
    CPUMatrix<ElemType>& AssignElementProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);
    CPUMatrix<ElemType>& AddElementProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);

    CPUMatrix<ElemType>& AssignElementDivisionOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);
    CPUMatrix<ElemType>& ElementDivideBy(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& ColumnElementMultiplyWith(const CPUMatrix<ElemType>& a);
    CPUMatrix<ElemType>& RowElementMultiplyWith(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& ColumnElementDivideBy(const CPUMatrix<ElemType>& a);
    CPUMatrix<ElemType>& RowElementDivideBy(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& ElementInverse();
    CPUMatrix<ElemType>& AssignElementInverseOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceSigmoid();
    CPUMatrix<ElemType>& AssignSigmoidOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceLinearRectifierDerivative();
    CPUMatrix<ElemType>& AssignLinearRectifierDerivativeOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceSigmoidDerivative();
    CPUMatrix<ElemType>& AssignSigmoidDerivativeOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceTanh();
    CPUMatrix<ElemType>& AssignTanhOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceLogSoftmax(const bool isColWise);
    CPUMatrix<ElemType>& AssignLogSoftmaxOf(const CPUMatrix<ElemType>& a, const bool isColWise);

    CPUMatrix<ElemType>& InplaceHardmax(const bool isColWise);
    CPUMatrix<ElemType>& AssignHardmaxOf(const CPUMatrix<ElemType>& a, const bool isColWise);

    // sequence training
    CPUMatrix<ElemType>& DropFrame(const CPUMatrix<ElemType>& label, const CPUMatrix<ElemType>& gamma, const ElemType& threshhold);
    CPUMatrix<ElemType>& AssignSequenceError(const ElemType hsmoothingWeight, const CPUMatrix<ElemType>& label, const CPUMatrix<ElemType>& dnnoutput, const CPUMatrix<ElemType>& gamma, ElemType alpha);
    CPUMatrix<ElemType>& InplaceSqrt();
    CPUMatrix<ElemType>& AssignSqrtOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceExp();
    CPUMatrix<ElemType>& AssignExpOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceLog();
    CPUMatrix<ElemType>& AssignLogOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceLog10();
    CPUMatrix<ElemType>& AssignLog10Of(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceCosine();
    CPUMatrix<ElemType>& AssignCosineOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceNegativeSine();
    CPUMatrix<ElemType>& AssignNegativeSineOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceAbs();
    CPUMatrix<ElemType>& AssignAbsOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& InplaceTruncateBottom(const ElemType threshold);
    CPUMatrix<ElemType>& AssignTruncateBottomOf(const CPUMatrix<ElemType>& a, const ElemType threshold);
    CPUMatrix<ElemType>& InplaceTruncateTop(const ElemType threshold);
    CPUMatrix<ElemType>& AssignTruncateTopOf(const CPUMatrix<ElemType>& a, const ElemType threshold);
    CPUMatrix<ElemType>& InplaceTruncate(const ElemType threshold);
    CPUMatrix<ElemType>& InplaceSoftThreshold(const ElemType threshold);

    CPUMatrix<ElemType>& SetToZeroIfAbsLessThan(const ElemType threshold);

    ElemType SumOfAbsElements() const; // sum of all abs(elements)
    ElemType SumOfElements() const;    // sum of all elements
    CPUMatrix<ElemType>& AssignSumOfElements(const CPUMatrix<ElemType>& a);

    bool IsEqualTo(const CPUMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;

    static void VectorSum(const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c, const bool isColWise);

    void VectorNorm1(CPUMatrix<ElemType>& c, const bool isColWise) const;
    CPUMatrix<ElemType>& AssignVectorNorm1Of(CPUMatrix<ElemType>& a, const bool isColWise);

    void VectorNorm2(CPUMatrix<ElemType>& c, const bool isColWise) const;
    CPUMatrix<ElemType>& AssignVectorNorm2Of(CPUMatrix<ElemType>& a, const bool isColWise);

    void AssignNoiseContrastiveEstimation(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& bias,
                                          CPUMatrix<ElemType>& tmp, CPUMatrix<ElemType>& c);

    void AssignSoftmaxSum(const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& softmax);

    void AssignNCEUnnormalizedEval(const CPUMatrix<ElemType>& a,
                                   const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& bias, CPUMatrix<ElemType>& c);

    CPUMatrix<ElemType>& AssignNCEDerivative(const CPUMatrix<ElemType>& tmp, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t inputIndex, CPUMatrix<ElemType>& c);

    void VectorNormInf(CPUMatrix<ElemType>& c, const bool isColWise) const;
    CPUMatrix<ElemType>& AssignVectorNormInfOf(CPUMatrix<ElemType>& a, const bool isColWise);

    CPUMatrix<ElemType>& AssignInnerProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const bool isColWise);
    CPUMatrix<ElemType>& AssignKhatriRaoProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);
    CPUMatrix<ElemType>& AddColumnReshapeProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const bool transposeAColumn);

    CPUMatrix<ElemType>& AddWithScaleOf(ElemType alpha, const CPUMatrix<ElemType>& a);

    ElemType FrobeniusNorm() const;
    CPUMatrix<ElemType>& AssignFrobeniusNormOf(const CPUMatrix<ElemType>& a);

    ElemType MatrixNormInf() const;
    ElemType MatrixNorm1() const;
    ElemType MatrixNorm0() const; // number of non-zero elemets
    CPUMatrix<ElemType>& AssignSignOf(const CPUMatrix<ElemType>& a);
    CPUMatrix<ElemType>& AddSignOf(const CPUMatrix<ElemType>& a);

    CPUMatrix<ElemType>& AssignRowSliceValuesOf(const CPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);
    CPUMatrix<ElemType>& AddToRowSliceValuesOf(const CPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);
    CPUMatrix<ElemType>& AddWithRowSliceValuesOf(const CPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);
    // CPUMatrix<ElemType>&  AssignRowStackValuesOf(const std::vector<const CPUMatrix<ElemType>*>& inputMatrices, const size_t sliceStartCol, const size_t sliceNumCols);

    CPUMatrix<ElemType>& AssignToRowSliceValuesOf(const CPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);

    CPUMatrix<ElemType>& AssignRepeatOf(const CPUMatrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats);
    CPUMatrix<ElemType>& AddToRowRepeatValuesOf(const CPUMatrix<ElemType>& a, const size_t numRowRepeats);

    CPUMatrix<ElemType>& AssignPositiveAndShiftedNegSample(const CPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber);
    CPUMatrix<ElemType>& AddFoldedPositiveAndShiftedNegSample(const CPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber);

    void VectorMax(CPUMatrix<ElemType>& maxIndexes, CPUMatrix<ElemType>& maxValues, const bool isColWise, int topK = 1) const;
    void VectorMin(CPUMatrix<ElemType>& minIndexes, CPUMatrix<ElemType>& minValues, const bool isColWise) const;

    CPUMatrix<ElemType>& AssignNumOfDiff(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, bool searchInCol = false);

    void Print(const char* matrixName, ptrdiff_t rowStart, ptrdiff_t rowEnd, ptrdiff_t colStart, ptrdiff_t colEnd) const;
    void Print(const char* matrixName = nullptr) const; // print whole matrix. can be expensive

    void ReadFromFile(FILE* f, const char* matrixName); // matrixName is used to verify that correct matrix is read.
    void WriteToFile(FILE* f, const char* matrixName);  // matrixName is used to verify that correct matrix is read.

    CPUMatrix<ElemType>& AssignPackedConvolutionInput(const CPUMatrix<ElemType>& inputSubBatch,
                                                      const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                      const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                      const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                      const bool zeroPadding = false);
    CPUMatrix<ElemType>& UnpackConvolutionInput(CPUMatrix<ElemType>& inputSubBatch,
                                                const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                const bool zeroPadding = false) const;
    CPUMatrix<ElemType>& AssignMaxPoolingResult(const CPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
    CPUMatrix<ElemType>& AddMaxPoolingGradient(const CPUMatrix<ElemType>& outputGradientBatch, const CPUMatrix<ElemType>& inputBatch, const CPUMatrix<ElemType>& outputBatch,
                                               const size_t channels,
                                               const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                               const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                               const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
    CPUMatrix<ElemType>& AssignAveragePoolingResult(const CPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                    const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                    const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                    const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
    CPUMatrix<ElemType>& AddAveragePoolingGradient(const CPUMatrix<ElemType>& outputGradientBatch,
                                                   const size_t channels,
                                                   const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                   const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                   const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);

    void ConvolutionForward(const CPUMatrix<ElemType>& kernel, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
                            const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const;
    void ConvolutionBackwardData(const CPUMatrix<ElemType>& kernel, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
                                 const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& grad) const;
    void ConvolutionBackwardKernel(const CPUMatrix<ElemType>& in, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
                                   const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& kernelGrad) const;

    void MaxPoolingForward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<ElemType>& output) const;
    void MaxPoolingBackward(const CPUMatrix<ElemType>& out, const CPUMatrix<ElemType>& in,
                            const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices,
                            CPUMatrix<ElemType>& grad) const;

    void AveragePoolingForward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<ElemType>& output) const;
    void AveragePoolingBackward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices,
                                CPUMatrix<ElemType>& grad) const;

    void BatchNormalizationForward(const CPUMatrix<ElemType>& scale, const CPUMatrix<ElemType>& bias, double expAvgFactor, double blendFactor, CPUMatrix<ElemType>& runMean, CPUMatrix<ElemType>& runInvStdDev,
                                   CPUMatrix<ElemType>& out, double epsilon, CPUMatrix<ElemType>& saveMean, CPUMatrix<ElemType>& saveInvStdDev) const;
    void BatchNormalizationBackward(const CPUMatrix<ElemType>& in, CPUMatrix<ElemType>& grad, const CPUMatrix<ElemType>& scale, const CPUMatrix<ElemType>& saveMean, const CPUMatrix<ElemType>& saveInvStdDev,
                                    CPUMatrix<ElemType>& scaleGrad, CPUMatrix<ElemType>& biasGrad) const;

public:
    static int SetNumThreads(int numThreads); // note: this does not depend on <ElemType>, i.e. you can call it on any <ElemType>

    // static BLAS functions
    static void SVD(const CPUMatrix<ElemType>& A, CPUMatrix<ElemType>& SIGMA, CPUMatrix<ElemType>& U, CPUMatrix<ElemType>& VT, CPUMatrix<ElemType>& W);

    static void MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const bool transposeA, const CPUMatrix<ElemType>& b, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c);
    static void MultiplyAndAdd(const CPUMatrix<ElemType>& a, const bool transposeA, const CPUMatrix<ElemType>& b, const bool transposeB, CPUMatrix<ElemType>& c);
    static void Multiply(const CPUMatrix<ElemType>& a, const bool transposeA, const CPUMatrix<ElemType>& b, const bool transposeB, CPUMatrix<ElemType>& c);
    static void Multiply(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);
    static void Multiply1x1AndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType beta, CPUMatrix<ElemType>& c);

    static void ScaleAndAdd(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c);
    static void AddScaledDifference(const ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);
    static void AssignScaledDifference(const ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);
    static void AddScaledDifference(const CPUMatrix<ElemType>& alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);    // alpha must be 1X1
    static void AssignScaledDifference(const CPUMatrix<ElemType>& alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c); // alpha must be 1X1

    static void AddElementToElement(const CPUMatrix<ElemType>& a, const size_t ai, const size_t aj, CPUMatrix<ElemType>& c, const size_t ci, const size_t cj);
    // static void AddLogElementToElement(const CPUMatrix<ElemType>& a, const size_t ai, const size_t aj, CPUMatrix<ElemType>& c, const size_t ci, const size_t cj);
    static void AssignElementToElement(const CPUMatrix<ElemType>& a, const size_t ai, const size_t aj, CPUMatrix<ElemType>& c, const size_t ci, const size_t cj);

    static void MinusOneAt(CPUMatrix<ElemType>& c, const size_t position);

    static void Scale(ElemType alpha, CPUMatrix<ElemType>& a);
    static void Scale(CPUMatrix<ElemType> alpha, CPUMatrix<ElemType>& a); // In this case Matrix alpha must be 1x1
    static void Scale(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c);
    static void InnerProduct(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, const bool isColWise);
    static ElemType InnerProductOfMatrices(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);
    static void ElementWisePower(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c);

    static bool AreEqual(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const ElemType threshold = 1e-8);

    static void TensorShuffleScaleAndAdd(ElemType keepWeight, const CPUMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);

    void TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, ElemType alpha, ElementWiseOperator op,
                  const std::array<size_t, 2>& offsets,
                  const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                  const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 2>& reducingStrides);
    void TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType alpha, ElementWiseOperator op,
                  const std::array<size_t, 3>& offsets,
                  const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                  const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 3>& reducingStrides);
    void TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& c, ElemType alpha, ElementWiseOperator op,
                  const std::array<size_t, 4>& offsets,
                  const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                  const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 4>& reducingStrides);

    static CPUMatrix<ElemType> Ones(const size_t rows, const size_t cols);
    static CPUMatrix<ElemType> Zeros(const size_t rows, const size_t cols);
    static CPUMatrix<ElemType> Eye(const size_t rows);
    static CPUMatrix<ElemType> RandomUniform(const size_t rows, const size_t cols, const ElemType low, const ElemType high, unsigned long seed = USE_TIME_BASED_SEED);
    static CPUMatrix<ElemType> RandomGaussian(const size_t rows, const size_t cols, const ElemType mean, const ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);

    // return true if v is an element in matrix c
    static bool HasElement(const CPUMatrix<ElemType>& a, const ElemType v = 0.0);

public:
    CPUMatrix<ElemType>& AssignElementProductOfWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift, size_t negnumber);
    static void InnerProductWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, const bool isColWise, size_t shift, size_t negnumber);
    // extract out a row from a, assign it to [this].
    CPUMatrix<ElemType>& GetARowByIndex(const CPUMatrix<ElemType>& a, const size_t index);
    static void ConductRowElementMultiplyWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, const size_t shift, bool bFirstmatrixfixed);
    CPUMatrix<ElemType>& AssignElementProductOfWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const size_t shift);

public:
    friend File& operator>>(File& stream, CPUMatrix<ElemType>& us)
    {
        stream.GetMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
        size_t elsize;
        stream >> elsize;
        if (sizeof(ElemType) != elsize)
            RuntimeError("Template argument size doesn't match those in file");
        std::wstring matrixName;
        size_t numRows, numCols;
        int format;
        stream >> matrixName >> format >> numRows >> numCols;
        ElemType* d_array = new ElemType[numRows * numCols];
        for (size_t i = 0; i < numRows * numCols; ++i)
            stream >> d_array[i];
        stream.GetMarker(fileMarkerEndSection, std::wstring(L"EMAT"));
        us.SetValue(numRows, numCols, d_array, matrixFlagNormal);

        delete[] d_array;
        return stream;
    }
    friend File& operator<<(File& stream, const CPUMatrix<ElemType>& us)
    {
        stream.PutMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
        stream << sizeof(ElemType);

        std::wstring s = std::wstring(L"unnamed");
        int format = us.m_format;
        stream << s << format;

        stream << us.m_numRows << us.m_numCols;
        for (size_t i = 0; i < us.GetNumElements(); ++i)
            stream << us.m_pArray[i];
        stream.PutMarker(fileMarkerEndSection, std::wstring(L"EMAT"));
        return stream;
    }

public:
    ElemType LogAddSumOfElements() const;

public:
    // for RCRF
    static void RCRFBackwardCompute(const CPUMatrix<ElemType>& alpha, CPUMatrix<ElemType>& beta,
                                    const CPUMatrix<ElemType>& lbls,
                                    const CPUMatrix<ElemType>& pair_scores);
    static void _rcrfBackwardCompute(size_t t, size_t k, const CPUMatrix<ElemType>& alpha,
                                     CPUMatrix<ElemType>& beta,
                                     const CPUMatrix<ElemType>& pair_scores);

    static void RCRFTransGrdCompute(const CPUMatrix<ElemType>& lbls,
                                    const CPUMatrix<ElemType>& alpha,
                                    const CPUMatrix<ElemType>& beta,
                                    const CPUMatrix<ElemType>& pair_scores,
                                    CPUMatrix<ElemType>& grd);

    static void _rcrfTransGrdCompute(size_t i,
                                     const CPUMatrix<ElemType>& lbls,
                                     const CPUMatrix<ElemType>& alpha,
                                     const CPUMatrix<ElemType>& beta,
                                     const CPUMatrix<ElemType>& pair_scores,
                                     CPUMatrix<ElemType>& grd,
                                     const size_t tPos // position
                                     );

protected:
    size_t LocateElement(const size_t i, const size_t j) const;
    size_t LocateColumn(const size_t j) const;

private:
    void ZeroInit(); // should only be used by constructors.
    void Clear();
};

typedef CPUMatrix<float> CPUSingleMatrix;
typedef CPUMatrix<double> CPUDoubleMatrix;
} } }
