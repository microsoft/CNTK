//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include "Platform.h"
#include "File.h"
#include "Helpers.h"
#include "CommonMatrix.h"
#include "TensorShape.h" // only for SmallVector; I was hoping to keep this out
#include "BestGpu.h" // for CPUONLY macro
#include "ConcStack.h"
#include <string>
#include <vector>
#include <array>
#include <ctime>
#include <iostream> // for cout/cerr
#include <memory>   // for unique_ptr
#include <limits.h> // for ULONG_MAX

#ifndef _WIN32
#include <unistd.h>
#endif

// predeclare cublasHandle_t
struct cublasContext;
typedef struct cublasContext* cublasHandle_t;
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

#ifdef _WIN32
#ifndef MATH_API
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#endif /* MATH_API */
#else  // no DLLs in Linux
#define MATH_API
#endif

#ifndef USE_TIME_BASED_SEED
#define USE_TIME_BASED_SEED ULONG_MAX
#endif

// Stream management functions
void MATH_API SetStream(cudaStream_t stream);
cudaStream_t MATH_API GetStream();

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// DeviceBoundNumber -- This class represents a number which resides on a particular device. Use it to avoid unnecessary transfers between CPU and GPU
// -----------------------------------------------------------------------

template <class ElemType>
class MATH_API DeviceBoundNumber
{
private:
    DEVICEID_TYPE m_computeDevice;
    ElemType* m_data;

public:
    DeviceBoundNumber()
    {
        m_data = NULL;
    };
    DeviceBoundNumber(const DeviceBoundNumber<ElemType>& deepCopy);
    DeviceBoundNumber(DeviceBoundNumber<ElemType>&& shallowCopy);
    ~DeviceBoundNumber();
    DEVICEID_TYPE GetDeviceId() const
    {
        return m_computeDevice;
    }
    ElemType* ExposePointer2Value() const
    {
        return m_data;
    }
    // performs shallow copy only
    void ShallowCopyFrom(ElemType* newVal, int newValsDevceId);
};

// -----------------------------------------------------------------------
// GPUMatrix
// -----------------------------------------------------------------------

void PrepareDevice(DEVICEID_TYPE deviceId);

template <class ElemType>
class MATH_API GPUMatrix : public BaseMatrix<ElemType>
{
    typedef BaseMatrix<ElemType> B;
    using B::m_numRows;
    using B::m_numCols;
    using B::m_pArray; // without this, base members would require to use thi-> in GCC

    template <typename T>
    friend class GPUMatrix;

public:
    static const int MaxGpus = 8; // support up to 8 GPUs
    using BaseMatrix<ElemType>::m_computeDevice;
    using BaseMatrix<ElemType>::m_elemSizeAllocated;
    using BaseMatrix<ElemType>::m_format;
    using BaseMatrix<ElemType>::m_externalBuffer;
    using BaseMatrix<ElemType>::m_nz;
    using BaseMatrix<ElemType>::OwnBuffer;
    using BaseMatrix<ElemType>::GetNumElements;
    using BaseMatrix<ElemType>::IsEmpty;
    using BaseMatrix<ElemType>::GetArray;
    using BaseMatrix<ElemType>::GetNumRows;
    using BaseMatrix<ElemType>::GetNumCols;

private:
    static cublasHandle_t s_cuHandle[MaxGpus];
    static void* s_curandGenerator;

// Have to use disable the warning to avoid issues with __declspec(dllexport) on Windows (C4251).
// Also, NVCC FE corresponding warning has to be disabled, see MathCUDA.vcxproj.
// The only workaround is to use naked pointer.
#pragma warning(push)
#pragma warning(disable : 4251)
    mutable std::unique_ptr<conc_stack<std::unique_ptr<GPUMatrix<ElemType>>>> m_workspace;
#pragma warning(pop)

private:
    void performElementWiseFunction(const ElementWiseOperator kind, const ElemType* src);
    size_t LocateElement(const size_t i, const size_t j) const;
    size_t LocateColumn(const size_t j) const;
    void Clear();
    void ZeroInit(int deviceId);

    std::unique_ptr<GPUMatrix<ElemType>> GetOrCreateWorkspace() const;
    void ReleaseWorkspace(std::unique_ptr<GPUMatrix<ElemType>> src) const;

public:
    explicit GPUMatrix(int deviceId);
    GPUMatrix(const size_t numRows, const size_t numCols, int deviceId);
    GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, ElemType* pArray, const size_t matrixFlags = matrixFlagNormal);
    GPUMatrix(const GPUMatrix<ElemType>& deepCopyFrom);
    GPUMatrix<ElemType>& operator=(const GPUMatrix<ElemType>& deepCopyFrom); // assignment operator, deep copy
    GPUMatrix(GPUMatrix<ElemType>&& moveFrom);
    GPUMatrix<ElemType>& operator=(GPUMatrix<ElemType>&& moveFrom); // move assignment operator, shallow copy
    ~GPUMatrix(void);

    static void SetDevice(DEVICEID_TYPE deviceId);
    int GetComputeDeviceId() const;
    DEVICEID_TYPE PrepareDevice(DEVICEID_TYPE deviceId = -1) const;

    static cublasHandle_t GetCublasHandle(int computeDevice = -1);
    ElemType* CopyToArray() const;                                              // allocated by the callee but need to be deleted by the caller
    size_t CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const; // allocated by the callee but need to be deleted by the caller
    void CopySection(size_t numRows, size_t numCols, ElemType* dst, size_t colStride) const;

    void ChangeDeviceTo(DEVICEID_TYPE to_id);

public:
    GPUMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;
    GPUMatrix<ElemType>& AssignColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols);
    GPUMatrix<ElemType>& SetColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols);

    void CopyColumnsStrided(const GPUMatrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride);

    GPUMatrix<ElemType> Diagonal() const;

    size_t BufferSize() const
    {
        return m_numRows * m_numCols * sizeof(ElemType);
    }
    ElemType* BufferPointer() const
    {
        return m_pArray;
    }

    ElemType Adagrad(GPUMatrix<ElemType>& gradients, const bool needAveMultiplier);
    void FSAdagrad(GPUMatrix<ElemType>& gradients, GPUMatrix<ElemType>& functionValues, ElemType learnRatePerSample, ElemType momentum, ElemType adaWeight, ElemType adaMul);
    ElemType RmsProp(GPUMatrix<ElemType>& gradients, ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN, const bool needAveMultiplier);

    void Reshape(const size_t numRows, const size_t numCols);
    void Resize(const size_t numRows, const size_t numCols, bool growOnly = true); // by default we only reallocate if need to grow

    ElemType& operator()(const size_t /*row*/, const size_t /*col*/)
    {
        LogicError("GPUMatrix doesn't support this");
    }
    const ElemType& operator()(const size_t /*row*/, const size_t /*col*/) const
    {
        LogicError("GPUMatrix doesn't support this");
    }
    ElemType Get00Element() const;

    void SetValue(const ElemType v);
    void SetValue(const ElemType* d_v); // d_v is pointer to the the value in GPU memory
    void SetColumn(const ElemType* colPointer, size_t colInd);
    void SetColumn(const GPUMatrix<ElemType>& valMat, size_t colInd);

    void MaskColumnsValue(const GPUMatrix<char>& columnsMask, ElemType val);

    void SetValue(const GPUMatrix<ElemType>& deepCopyFrom);
    void SetValue(const size_t numRows, const size_t numCols, int deviceId, ElemType* pArray, size_t matrixFlags = matrixFlagNormal);

    void SetDiagonalValue(const ElemType v);
    void SetDiagonalValue(const GPUMatrix<ElemType>& vector);
    void SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed = USE_TIME_BASED_SEED);
    void SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);
    void SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, unsigned long seed = USE_TIME_BASED_SEED);

    GPUMatrix<ElemType> Transpose() const;
    GPUMatrix<ElemType>& AssignTransposeOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& operator+=(const ElemType alpha);
    GPUMatrix<ElemType> operator+(const ElemType alpha) const;
    GPUMatrix<ElemType>& AssignSumOf(const ElemType alpha, const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& operator+=(const GPUMatrix<ElemType>& a);
    GPUMatrix<ElemType> operator+(const GPUMatrix<ElemType>& a) const;
    GPUMatrix<ElemType>& AssignSumOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);

    GPUMatrix<ElemType>& operator-=(const ElemType alpha);
    GPUMatrix<ElemType> operator-(const ElemType alpha) const;
    GPUMatrix<ElemType>& AssignDifferenceOf(const ElemType alpha, const GPUMatrix<ElemType>& a);
    GPUMatrix<ElemType>& AssignDifferenceOf(const GPUMatrix<ElemType>& a, const ElemType alpha);

    GPUMatrix<ElemType>& operator-=(const GPUMatrix<ElemType>& a);
    GPUMatrix<ElemType> operator-(const GPUMatrix<ElemType>& a) const;
    GPUMatrix<ElemType>& AssignDifferenceOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);

    GPUMatrix<ElemType>& operator*=(const ElemType alpha);
    GPUMatrix<ElemType> operator*(const ElemType alpha) const;
    GPUMatrix<ElemType>& AssignProductOf(const ElemType alpha, const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType> operator*(const GPUMatrix<ElemType>& a) const;
    GPUMatrix<ElemType>& AssignProductOf(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB);

    GPUMatrix<ElemType>& operator/=(ElemType alpha);
    GPUMatrix<ElemType> operator/(ElemType alpha) const;

    GPUMatrix<ElemType>& operator^=(ElemType alpha);     // element-wise power
    GPUMatrix<ElemType> operator^(ElemType alpha) const; // element-wise power
    GPUMatrix<ElemType>& AssignElementPowerOf(const GPUMatrix<ElemType>& a, const ElemType power);

    GPUMatrix<ElemType>& ElementMultiplyWith(const GPUMatrix<ElemType>& a);
    GPUMatrix<ElemType>& AssignElementProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
    GPUMatrix<ElemType>& AddElementProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);

    GPUMatrix<ElemType>& AssignElementDivisionOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
    GPUMatrix<ElemType>& ElementDivideBy(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& ColumnElementMultiplyWith(const GPUMatrix<ElemType>& a);
    GPUMatrix<ElemType>& RowElementMultiplyWith(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& ColumnElementDivideBy(const GPUMatrix<ElemType>& a);
    GPUMatrix<ElemType>& RowElementDivideBy(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& ElementInverse();
    GPUMatrix<ElemType>& AssignElementInverseOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceLinearRectifierDerivative();
    GPUMatrix<ElemType>& AssignLinearRectifierDerivativeOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceSigmoidDerivative();
    GPUMatrix<ElemType>& AssignSigmoidDerivativeOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceSigmoid();
    GPUMatrix<ElemType>& AssignSigmoidOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceTanh();
    GPUMatrix<ElemType>& AssignTanhOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceLogSoftmax(const bool isColWise);
    GPUMatrix<ElemType>& AssignLogSoftmaxOf(const GPUMatrix<ElemType>& a, const bool isColWise);

    GPUMatrix<ElemType>& InplaceHardmax(const bool isColWise);
    GPUMatrix<ElemType>& AssignHardmaxOf(const GPUMatrix<ElemType>& a, const bool isColWise);

    // sequence training
    GPUMatrix<ElemType>& DropFrame(const GPUMatrix<ElemType>& label, const GPUMatrix<ElemType>& gamma, const ElemType& threshhold);
    GPUMatrix<ElemType>& AssignSequenceError(const ElemType hsmoothingWeight, const GPUMatrix<ElemType>& label, const GPUMatrix<ElemType>& dnnoutput, const GPUMatrix<ElemType>& gamma, ElemType alpha);

    GPUMatrix<ElemType>& InplaceSqrt();
    GPUMatrix<ElemType>& AssignSqrtOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceExp();
    GPUMatrix<ElemType>& AssignExpOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceLog();
    GPUMatrix<ElemType>& AssignLogOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceCosine();
    GPUMatrix<ElemType>& AssignCosineOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceNegativeSine();
    GPUMatrix<ElemType>& AssignNegativeSineOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceAbs();
    GPUMatrix<ElemType>& AssignAbsOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& InplaceTruncateBottom(const ElemType threshold);
    GPUMatrix<ElemType>& AssignTruncateBottomOf(const GPUMatrix<ElemType>& a, const ElemType threshold);
    GPUMatrix<ElemType>& InplaceTruncateTop(const ElemType threshold);
    GPUMatrix<ElemType>& AssignTruncateTopOf(const GPUMatrix<ElemType>& a, const ElemType threshold);
    GPUMatrix<ElemType>& InplaceTruncate(const ElemType threshold);
    GPUMatrix<ElemType>& InplaceSoftThreshold(const ElemType threshold);

    GPUMatrix<ElemType>& SetToZeroIfAbsLessThan(const ElemType threshold);

    DeviceBoundNumber<ElemType> Sum_AsDeviceBoundNum() const;
    ElemType SumOfAbsElements() const; // sum of all abs(elements)
    ElemType SumOfElements() const;    // sum of all elements
    GPUMatrix<ElemType>& AssignSumOfElements(const GPUMatrix<ElemType>& a);

    ElemType Max() const;
    bool IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;

    static void VectorSum(const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c, const bool isColWise);

    void VectorNorm1(GPUMatrix<ElemType>& c, const bool isColWise) const;
    GPUMatrix<ElemType>& AssignVectorNorm1Of(GPUMatrix<ElemType>& a, const bool isColWise);

    void VectorNorm2(GPUMatrix<ElemType>& c, const bool isColWise) const;
    GPUMatrix<ElemType>& AssignVectorNorm2Of(GPUMatrix<ElemType>& a, const bool isColWise);

    void VectorNormInf(GPUMatrix<ElemType>& c, const bool isColWise) const;
    GPUMatrix<ElemType>& AssignVectorNormInfOf(GPUMatrix<ElemType>& a, const bool isColWise);

    GPUMatrix<ElemType>& AssignInnerProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const bool isColWise);
    GPUMatrix<ElemType>& AssignKhatriRaoProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
    GPUMatrix<ElemType>& AddColumnReshapeProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const bool transposeAColumn);

    GPUMatrix<ElemType>& AddWithScaleOf(ElemType alpha, const GPUMatrix<ElemType>& a);

    ElemType FrobeniusNorm() const;
    GPUMatrix<ElemType>& AssignFrobeniusNormOf(const GPUMatrix<ElemType>& a);

    ElemType MatrixNormInf() const;
    ElemType MatrixNorm1() const;
    ElemType MatrixNorm0() const; // number of non-zero elemets
    GPUMatrix<ElemType>& AssignSignOf(const GPUMatrix<ElemType>& a);
    GPUMatrix<ElemType>& AddSignOf(const GPUMatrix<ElemType>& a);

    GPUMatrix<ElemType>& AssignToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);
    GPUMatrix<ElemType>& AssignRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);
    GPUMatrix<ElemType>& AddToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);
    GPUMatrix<ElemType>& AddWithRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);
    // GPUMatrix<ElemType>&  AssignRowStackValuesOf(const std::vector<const GPUMatrix<ElemType>*>& inputMatrices, const size_t sliceStartCol, const size_t sliceNumCols);

    GPUMatrix<ElemType>& AssignRepeatOf(const GPUMatrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats);
    GPUMatrix<ElemType>& AddToRowRepeatValuesOf(const GPUMatrix<ElemType>& a, const size_t numRowRepeats);

    GPUMatrix<ElemType>& AssignPositiveAndShiftedNegSample(const GPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber);
    GPUMatrix<ElemType>& AddFoldedPositiveAndShiftedNegSample(const GPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber);

    void VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise) const;
    void VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise, int topK) const;
    void VectorMin(GPUMatrix<ElemType>& minIndexes, GPUMatrix<ElemType>& minValues, const bool isColWise) const;

    GPUMatrix<ElemType>& AssignNumOfDiff(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, bool searchInCol = false);

    GPUMatrix<ElemType>& AssignInnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);

    void AssignNoiseContrastiveEstimation(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const GPUMatrix<ElemType>& bias,
                                          size_t sampleCount, GPUMatrix<ElemType>& tmp, GPUMatrix<ElemType>& c);
    void AssignNCEDerivative(GPUMatrix<ElemType>& tmp, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, size_t inputIndex, GPUMatrix<ElemType>& c);
    void AssignNCEUnnormalizedEval(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
    void AssignSoftmaxSum(const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& softmax);

    void Print(const char* matrixName, size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd) const;
    void Print(const char* matrixName = NULL) const; // print whole matrix. can be expensive

    GPUMatrix<ElemType>& AssignPackedConvolutionInput(const GPUMatrix<ElemType>& inputSubBatch,
                                                      const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                      const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                      const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                      const bool zeroPadding = false);
    GPUMatrix<ElemType>& UnpackConvolutionInput(GPUMatrix<ElemType>& inputSubBatch,
                                                const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                bool zeroPadding = false) const;
    GPUMatrix<ElemType>& AssignMaxPoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
    GPUMatrix<ElemType>& AddMaxPoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch, const GPUMatrix<ElemType>& inputBatch, const GPUMatrix<ElemType>& outputBatch,
                                               const size_t channels,
                                               const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                               const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                               const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
    GPUMatrix<ElemType>& AssignAveragePoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                    const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                    const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                    const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
    GPUMatrix<ElemType>& AddAveragePoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch,
                                                   const size_t channels,
                                                   const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                   const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                   const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);

public:
    // static BLAS functions
    static void MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c);
    static void MultiplyAndAdd(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c);
    static void Multiply(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c);
    static void Multiply(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
    static void Multiply1x1AndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, ElemType beta, GPUMatrix<ElemType>& c);

    static void ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c);
    static void ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
    static void AddScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
    static void AssignScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
    static void AddScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
    static void AssignScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);

    static void AddElementToElement(const GPUMatrix<ElemType>& a, const size_t ai, const size_t aj, GPUMatrix<ElemType>& c, const size_t ci, const size_t cj);

    // minus one at a specific position
    static void MinusOneAt(GPUMatrix<ElemType>& c, const size_t position);

    static void Scale(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c);
    static void Scale(GPUMatrix<ElemType>& alpha, GPUMatrix<ElemType>& a); // In this case matrix alpha must be 1x1
    static void Scale(ElemType alpha, GPUMatrix<ElemType>& a);
    static void InnerProduct(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const bool isColWise);
    static ElemType InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
    static void ElementWisePower(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c);

    static bool AreEqual(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const ElemType threshold = 1e-8);

    static void TensorShuffleScaleAndAdd(ElemType keepWeight, const GPUMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);

    void TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, ElemType alpha, ElementWiseOperator op,
                  const std::array<size_t, 2>& offsets,
                  const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                  const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 2>& reducingStrides);
    void TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, ElemType alpha, ElementWiseOperator op,
                  const std::array<size_t, 3>& offsets,
                  const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                  const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 3>& reducingStrides);
    void TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const GPUMatrix<ElemType>& c, ElemType alpha, ElementWiseOperator op,
                  const std::array<size_t, 4>& offsets,
                  const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                  const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 4>& reducingStrides);

    static void CreateCurandObject(unsigned long seed, const char* caller);
    static void ResetCurandObject(unsigned long seed, const char* caller);
    static GPUMatrix<ElemType> Ones(const size_t rows, const size_t cols, int deviceId);
    static GPUMatrix<ElemType> Zeros(const size_t rows, const size_t cols, int deviceId);
    static GPUMatrix<ElemType> Eye(const size_t rows, int deviceId);
    static GPUMatrix<ElemType> RandomUniform(const size_t rows, const size_t cols, int deviceId, const ElemType low, const ElemType high, unsigned long seed = USE_TIME_BASED_SEED);
    static GPUMatrix<ElemType> RandomGaussian(const size_t rows, const size_t cols, int deviceId, const ElemType mean, const ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);

    static bool HasElement(const GPUMatrix<ElemType>& a, const ElemType v = 0.0);

    static ElemType GetLearnRateForBlock_Helper(const GPUMatrix<ElemType>& Gradients, const GPUMatrix<ElemType>& SmoothedGradients);

    ElemType LogAddSumOfElements() const;

public:
    GPUMatrix<ElemType>& AssignElementProductOfWithShiftNeg(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const size_t shift, const size_t nt);
    static void InnerProductWithShiftNeg(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const size_t shift, const size_t nt);
    GPUMatrix<ElemType>& GetARowByIndex(const GPUMatrix<ElemType>& a, const size_t m);
    static void ConductRowElementMultiplyWithShift(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const size_t shift, const bool isafixed);

    GPUMatrix<ElemType>& AssignElementProductOfWithShift(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const size_t shift);

public:
    static void RCRFBackwardCompute(
        const GPUMatrix<ElemType>& alpha, GPUMatrix<ElemType>& beta,
        const GPUMatrix<ElemType>& lbls,
        const GPUMatrix<ElemType>& pos_scores, const GPUMatrix<ElemType>& pair_scores, const int shift = 1);
    static void RCRFTransGrdCompute(const GPUMatrix<ElemType>& lbls,
                                    const GPUMatrix<ElemType>& alpha,
                                    const GPUMatrix<ElemType>& beta,
                                    const GPUMatrix<ElemType>& pair_scores,
                                    GPUMatrix<ElemType>& grd,
                                    const int startLbl, // the time 0 start symbol in the output layer
                                    const int shift);

public:
    friend File& operator>>(File& stream, GPUMatrix<ElemType>& us)
    {
        stream.GetMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
        size_t elsize;
        stream >> elsize;
        if (sizeof(ElemType) != elsize)
            LogicError("Template argument size doesn't match those in file");
        std::wstring matrixNameDummy; // Note this is not used anymore, just a dummy for compatability.
        size_t numRows, numCols;
        int format;
        stream >> matrixNameDummy >> format >> numRows >> numCols;
        ElemType* d_array = new ElemType[numRows * numCols];
        for (size_t i = 0; i < numRows * numCols; ++i)
            stream >> d_array[i];
        stream.GetMarker(fileMarkerEndSection, std::wstring(L"EMAT"));
        us.SetValue(numRows, numCols, us.GetComputeDeviceId(), d_array, matrixFlagNormal | format);
        delete[] d_array;
        return stream;
    }
    friend File& operator<<(File& stream, const GPUMatrix<ElemType>& us)
    {
        stream.PutMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
        stream << sizeof(ElemType);

        // TODO: This is now ignored on input, so we can should change to an empty string. This might break parsing, and must be tested first
        std::wstring s = std::wstring(L"unnamed");
        int format = us.m_format;
        stream << s << format;

        stream << us.m_numRows << us.m_numCols;
        ElemType* pArray = us.CopyToArray();
        for (size_t i = 0; i < us.GetNumElements(); ++i)
            stream << pArray[i];
        
        delete[] pArray;

        stream.PutMarker(fileMarkerEndSection, std::wstring(L"EMAT"));
        return stream;
    }
};

typedef GPUMatrix<float> GPUSingleMatrix;

}}}

#ifndef CPUONLY

#include <cuda_runtime.h>

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE>
const char* CudaErrString(ERRTYPE x); // actual error function is defined inside .cu files
template <typename ERRTYPE>
static void CudaCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode)
{
    if (retCode != successCode)
    {
        try
        {
#ifdef _WIN32
            const char* hostname = getenv("COMPUTERNAME");
#else
            char hostname[HOST_NAME_MAX];
            if (gethostname(hostname, HOST_NAME_MAX) != 0)
                strcpy(hostname, "?");
#endif
            int currentCudaDevice;
            cudaGetDevice(&currentCudaDevice);
            Microsoft::MSR::CNTK::RuntimeError("%s failure %d: %s ; GPU=%d ; hostname=%s ; expr=%s", libName, (int)retCode, CudaErrString(retCode), currentCudaDevice, hostname ? hostname : "?", exprString);
        }
        catch (const std::exception& e) // catch, log, and rethrow since CUDA code sometimes hangs in destruction, so we'd never get to see the error
        {
            std::cerr << e.what() << std::endl;
            throw;
        }
    }
}

#define CUDA_CALL(expr)     (CudaCall((expr), #expr, "CUDA",     cudaSuccess))
#define CUBLAS_CALL(expr)   (CudaCall((expr), #expr, "CUBLAS",   CUBLAS_STATUS_SUCCESS))
#define CUSPARSE_CALL(expr) (CudaCall((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS))
#define CURAND_CALL(expr)   (CudaCall((expr), #expr, "CURAND",   CURAND_STATUS_SUCCESS))
#define CUDNN_CALL(expr)    (CudaCall((expr), #expr, "cuDNN",    CUDNN_STATUS_SUCCESS))

// -----------------------------------------------------------------------
// SyncGuard -- synchronize around CUDA calls
// -----------------------------------------------------------------------

class SyncGuard
{
    static bool DoSync()
    {
#ifdef NO_SYNC // this strange way of writing it allows modifying this variable at runtime in the debugger
        static bool do_sync = false;
#else
        static bool do_sync = true;
#endif
        return do_sync;
    }
    cudaEvent_t m_done;
public:
    SyncGuard()
    {
        m_done = nullptr;
        if (DoSync())
            CUDA_CALL(cudaEventCreate(&m_done));
    }
    ~SyncGuard()
    {
        if (DoSync())
        {
            // The regular use of this destructor is to synchronize the GPU, but also
            // to check for errors. So this destructor is where CUDA errors would be thrown.
            // If this destructor runs during stack unwinding, then a different error has
            // already happened that should be reported; so we only clean up the resource.
            if (std::uncaught_exception())
                cudaEventDestroy(m_done);
            else
            {
                // failures in a prior launch might be reported here
                CUDA_CALL(cudaEventRecord(m_done));
                CUDA_CALL(cudaEventSynchronize(m_done));
                CUDA_CALL(cudaEventDestroy(m_done));
            }
        }
    }
};

#endif // CPUONLY
