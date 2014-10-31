//
// <copyright file="GPUMatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once
#include <string>
#include <vector>
#include <ctime>
#include "File.h"
#include "Helpers.h"
#include "CommonMatrix.h"

// predeclare cublasHandle_t
struct cublasContext;
typedef struct cublasContext *cublasHandle_t;
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;

#ifdef _WIN32
#ifndef	MATH_API
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#endif	/* MATH_API */
#else	// no DLLs in Linux
#define MATH_API 
#endif

#ifndef USE_TIME_BASED_SEED
#define USE_TIME_BASED_SEED ULONG_MAX
#endif

// Stream management functions
void MATH_API SetStream(cudaStream_t stream);
cudaStream_t MATH_API GetStream();

namespace Microsoft { namespace MSR { namespace CNTK {    

    void PrepareDevice(short deviceId);

    //This class represents a number which resides on a particular device. Use it to avoid unnecessary transfers between CPU and GPU
    template<class ElemType>
    class MATH_API DeviceBoundNumber
    {
    private:
        int m_computeDevice;
        ElemType* m_data;
    public:        
        DeviceBoundNumber() {m_data=NULL;};        
        DeviceBoundNumber(const DeviceBoundNumber<ElemType> &deepCopy);
        DeviceBoundNumber(DeviceBoundNumber<ElemType> &&shallowCopy);
        ~DeviceBoundNumber();
        int GetDeviceId() const {return m_computeDevice;}
        ElemType* ExposePointer2Value() const {return m_data;}
        //performs shallow copy only
        void ShallowCopyFrom(ElemType* newVal,int newValsDevceId);
    };

    template<class ElemType>
    class MATH_API GPUMatrix : public BaseMatrix<ElemType>
    {
        typedef BaseMatrix<ElemType> B; using B::m_numRows; using B::m_numCols; using B::m_pArray;   // without this, base members would require to use thi-> in GCC
    public:
        static const int MaxGpus = 8;  // support up to 8 GPUs
    private:
        static cublasHandle_t s_cuHandle[MaxGpus];
        static void *s_curandGenerator;

    private:
        void performInplaceFunction(int kind);
        size_t LocateElement (const size_t i, const size_t j) const;
        size_t LocateColumn (const size_t j) const;        
        void Clear();
        void ZeroInit(int deviceId);

    public:
        GPUMatrix(int deviceId=0);
        GPUMatrix(FILE* f, const char * matrixName, int deviceId=0);
        GPUMatrix(const size_t numRows, const size_t numCols, int deviceId=0);
        GPUMatrix(const size_t numRows, const size_t numCols, ElemType *pArray, const size_t matrixFlags=matrixFlagNormal,int deviceId=0);        
        GPUMatrix(const GPUMatrix<ElemType>& deepCopyFrom);    
        GPUMatrix<ElemType>& operator=(const GPUMatrix<ElemType>& deepCopyFrom);  //assignment operator, deep copy
        GPUMatrix(GPUMatrix<ElemType>&& moveFrom);
        GPUMatrix<ElemType>& operator=(GPUMatrix<ElemType>&& moveFrom);  //move assignment operator, shallow copy
        ~GPUMatrix(void);       

        static int GetBestGPUDeviceId();  
        int GetComputeDeviceId() const;
        void PrepareDevice(short deviceId=-1) const;

        static cublasHandle_t GetCublasHandle(int computeDevice=-1);
        ElemType* CopyToArray() const; //allocated by the callee but need to be deleted by the caller
        size_t CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const;  //allocated by the callee but need to be deleted by the caller

        void ChangeDeviceTo(int to_id);

    public:

        GPUMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;
        GPUMatrix<ElemType>& AssignColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols);

        size_t BufferSize() const {return m_numRows*m_numCols*sizeof(ElemType);}
        ElemType* BufferPointer() const {return m_pArray;}

        void Adagrad(GPUMatrix<ElemType>& gradients);
        void RmsProp(GPUMatrix<ElemType>& gradients, ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN);
        void Reshape(const size_t numRows, const size_t numCols);
        void Resize(const size_t numRows, const size_t numCols, bool growOnly = true);  //by default we only reallocate if need to grow

        ElemType& operator() (const size_t /*row*/, const size_t /*col*/) { throw std::logic_error("GPUMatrix doesn't support this"); }
        const ElemType& operator() (const size_t /*row*/, const size_t /*col*/) const { throw std::logic_error("GPUMatrix doesn't support this"); }
        ElemType Get00Element() const;

        void SetValue(const ElemType v);
        void SetValue(const ElemType* d_v); //d_v is pointer to the the value in GPU memory
        void SetColumn(const ElemType* colPointer, size_t colInd);
        void SetValue(const GPUMatrix<ElemType>& deepCopyFrom);
        void SetValue(const size_t numRows, const size_t numCols, ElemType *pArray, size_t matrixFlags=matrixFlagNormal, int deviceId=MANAGEDEXTERN);        

        void SetDiagonalValue(const ElemType v);
        void SetDiagonalValue(GPUMatrix<ElemType>& vector);
        void SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed=USE_TIME_BASED_SEED);
        void SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed=USE_TIME_BASED_SEED);
        void SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, unsigned long seed=USE_TIME_BASED_SEED); 

        GPUMatrix<ElemType> Transpose() const;
        GPUMatrix<ElemType>& AssignTransposeOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& operator+= (const ElemType alpha);
        GPUMatrix<ElemType> operator+ (const ElemType alpha) const;
        GPUMatrix<ElemType>& AssignSumOf(const ElemType alpha, const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& operator+= (const GPUMatrix<ElemType>& a);
        GPUMatrix<ElemType> operator+ (const GPUMatrix<ElemType>& a) const;
        GPUMatrix<ElemType>& AssignSumOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);

        GPUMatrix<ElemType>& operator-= (const ElemType alpha);
        GPUMatrix<ElemType> operator- (const ElemType alpha) const;
        GPUMatrix<ElemType>& AssignDifferenceOf(const ElemType alpha, const GPUMatrix<ElemType>& a);
        GPUMatrix<ElemType>& AssignDifferenceOf(const GPUMatrix<ElemType>& a, const ElemType alpha);

        GPUMatrix<ElemType>& operator-= (const GPUMatrix<ElemType>& a);
        GPUMatrix<ElemType> operator- (const GPUMatrix<ElemType>& a) const;
        GPUMatrix<ElemType>& AssignDifferenceOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);

        GPUMatrix<ElemType>& operator*= (const ElemType alpha);
        GPUMatrix<ElemType> operator* (const ElemType alpha) const;
        GPUMatrix<ElemType>& AssignProductOf(const ElemType alpha, const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType> operator* (const GPUMatrix<ElemType>& a) const;     
        GPUMatrix<ElemType>& AssignProductOf(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB);

        GPUMatrix<ElemType>& operator/= (ElemType alpha);
        GPUMatrix<ElemType> operator/ (ElemType alpha) const;

        GPUMatrix<ElemType>& operator^= (ElemType alpha); //element-wise power
        GPUMatrix<ElemType> operator^ (ElemType alpha) const; //element-wise power
        GPUMatrix<ElemType>& AssignElementPowerOf(const GPUMatrix<ElemType>& a, const ElemType power);

        GPUMatrix<ElemType>& ElementMultiplyWith (const GPUMatrix<ElemType>& a);
        GPUMatrix<ElemType>& AssignElementProductOf (const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
        GPUMatrix<ElemType>& AddElementProductOf (const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);

        GPUMatrix<ElemType>& AssignElementDivisionOf (const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
        GPUMatrix<ElemType>& ElementDivideBy(const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& ColumnElementMultiplyWith(const GPUMatrix<ElemType>& a);
        GPUMatrix<ElemType>& RowElementMultiplyWith(const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& ColumnElementDivideBy(const GPUMatrix<ElemType>& a);
        GPUMatrix<ElemType>& RowElementDivideBy(const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& ElementInverse ();
        GPUMatrix<ElemType>& AssignElementInverseOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceLinearRectifierDerivative();
        GPUMatrix<ElemType>& AssignLinearRectifierDerivativeOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceSigmoidDerivative();
        GPUMatrix<ElemType>& AssignSigmoidDerivativeOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceSigmoid ();
        GPUMatrix<ElemType>& AssignSigmoidOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceTanh ();
        GPUMatrix<ElemType>& AssignTanhOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceLogSoftmax (const bool isColWise);
        GPUMatrix<ElemType>& AssignLogSoftmaxOf (const GPUMatrix<ElemType>& a, const bool isColWise);

        GPUMatrix<ElemType>& InplaceSqrt ();
        GPUMatrix<ElemType>& AssignSqrtOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceExp ();
        GPUMatrix<ElemType>& AssignExpOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceLog ();
        GPUMatrix<ElemType>& AssignLogOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceCosine ();
        GPUMatrix<ElemType>& AssignCosineOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceNegativeSine ();
        GPUMatrix<ElemType>& AssignNegativeSineOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceAbs ();   
        GPUMatrix<ElemType>& AssignAbsOf (const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>& InplaceTruncateBottom (const ElemType threshold);
        GPUMatrix<ElemType>& AssignTruncateBottomOf (const GPUMatrix<ElemType>& a, const ElemType threshold);
        GPUMatrix<ElemType>& InplaceTruncateTop (const ElemType threshold);
        GPUMatrix<ElemType>& AssignTruncateTopOf (const GPUMatrix<ElemType>& a, const ElemType threshold);

        GPUMatrix<ElemType>& SetToZeroIfAbsLessThan (const ElemType threshold);

        DeviceBoundNumber<ElemType> Sum_AsDeviceBoundNum() const;
        ElemType SumOfAbsElements () const; //sum of all abs(elements)
        ElemType SumOfElements () const; //sum of all elements
        GPUMatrix<ElemType>& AssignSumOfElements(const GPUMatrix<ElemType>& a);

        ElemType Max () const;
        bool IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;

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
        ElemType MatrixNorm0() const; //number of non-zero elemets
        GPUMatrix<ElemType>& AssignSignOf(const GPUMatrix<ElemType>& a);
        GPUMatrix<ElemType>& AddSignOf(const GPUMatrix<ElemType>& a);

        GPUMatrix<ElemType>&  AssignRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows); 
        GPUMatrix<ElemType>&  AddToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows); 
        GPUMatrix<ElemType>&  AddWithRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows);

        GPUMatrix<ElemType>&  AssignRepeatOf(const GPUMatrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats);

        void VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise) const;
        void VectorMin(GPUMatrix<ElemType>& mainndexes, GPUMatrix<ElemType>& minValues, const bool isColWise) const;

        GPUMatrix<ElemType>&   AssignNumOfDiff(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b); 


        GPUMatrix<ElemType>& AssignInnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b); 

        void Print(const char* matrixName, size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd) const;
        void Print(const char* matrixName = NULL) const; //print whole matrix. can be expensive

        void ReadFromFile(FILE* f, const char * matrixName); //matrixName is used to verify that correct matrix is read.
        void WriteToFile(FILE* f, const char * matrixName); //matrixName is used to verify that correct matrix is read.

        GPUMatrix<ElemType>&  AssignPackedConvolutionInput(const GPUMatrix<ElemType>& inputSubBatch, 
                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample, 
                                                 const bool zeroPadding = false); 
        GPUMatrix<ElemType>&  UnpackConvolutionInput(GPUMatrix<ElemType>& inputSubBatch, 
                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample, 
                                                 bool zeroPadding = false) const; 
        GPUMatrix<ElemType>& AssignMaxPoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels, 
                                                 const size_t inputWidth, const size_t inputHeight,  const size_t inputSizePerSample, 
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
        GPUMatrix<ElemType>& AddMaxPoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch, const GPUMatrix<ElemType>& inputBatch, const GPUMatrix<ElemType>& outputBatch, 
                                                 const size_t channels, 
                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample, 
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
        GPUMatrix<ElemType>& AssignAveragePoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels, 
                                                 const size_t inputWidth, const size_t inputHeight,  const size_t inputSizePerSample, 
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
        GPUMatrix<ElemType>& AddAveragePoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch, 
                                                 const size_t channels, 
                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample, 
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
    public:
        //static BLAS functions
        static void MultiplyAndWeightedAdd(ElemType alpha,const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, 
            ElemType beta, GPUMatrix<ElemType>& c);
        static void MultiplyAndAdd(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c);
        static void Multiply(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c);
        static void Multiply(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);

        static void ScaleAndAdd(ElemType alpha,const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c);
        static void AddScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
        static void AssignScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
        static void AddScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
        static void AssignScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);

        static void AddElementToElement(const GPUMatrix<ElemType>& a, const size_t ai, const size_t aj, GPUMatrix<ElemType>& c, const size_t ci, const size_t cj); 

        static void Scale(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c);
        static void Scale(GPUMatrix<ElemType> &alpha, GPUMatrix<ElemType>& a); //In this case matrix alpha must be 1x1
        static void Scale(ElemType alpha, GPUMatrix<ElemType>& a);       
        static void InnerProduct (const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const bool isColWise);
        static ElemType InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);                
        static void ElementWisePower (ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c);        

        static bool AreEqual(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const ElemType threshold = 1e-8);

        static GPUMatrix<ElemType> Ones(const size_t rows, const size_t cols);
        static GPUMatrix<ElemType> Zeros(const size_t rows, const size_t cols);
        static GPUMatrix<ElemType> Eye(const size_t rows);
        static GPUMatrix<ElemType> RandomUniform(const size_t rows, const size_t cols, const ElemType low, const ElemType high, unsigned long seed=USE_TIME_BASED_SEED);
        static GPUMatrix<ElemType> RandomGaussian(const size_t rows, const size_t cols, const ElemType mean, const ElemType sigma, unsigned long seed=USE_TIME_BASED_SEED);

        static ElemType GetLearnRateForBlock_Helper(const GPUMatrix<ElemType> &Gradients, const GPUMatrix<ElemType> &SmoothedGradients);
    public:
        friend File& operator>>(File& stream, GPUMatrix<ElemType>& us)
        {
            stream.GetMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
            size_t elsize;
            stream>>elsize;
            if (sizeof(ElemType)!=elsize)
#ifndef	LINUX
                throw std::exception("Template argument size doesn't match those in file");
#else
                throw std::exception();
#endif
            std::wstring matrixName;
            size_t numRows, numCols;
            int format;
            stream>>matrixName>>format>>numRows>>numCols;
            ElemType* d_array = new ElemType[numRows*numCols];
            for (size_t i=0;i<numRows*numCols;++i)
                stream>>d_array[i];
            stream.GetMarker(fileMarkerEndSection, std::wstring(L"EMAT"));
            us.SetValue(numRows,numCols,d_array, matrixFlagNormal | format);
            delete[] d_array;
            us.m_matrixName = new wchar_t[matrixName.length()+1];
            wmemcpy(us.m_matrixName,matrixName.c_str(),matrixName.length()+1);
            //us.m_matrixName = matrixName;
            return stream;
        }
        friend File& operator<<(File& stream, const GPUMatrix<ElemType>& us)
        {
            stream.PutMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
            stream<<sizeof(ElemType);

            std::wstring s = (us.m_matrixName==NULL)? std::wstring(L"unnamed") : std::wstring(us.m_matrixName);
            int format = us.m_format;
            stream << s << format;

            stream<<us.m_numRows<<us.m_numCols;
            ElemType *m_pArray = us.CopyToArray();
            for (size_t i=0;i<us.GetNumElements();++i) 
                stream<<m_pArray[i];
            delete[] m_pArray;
            stream.PutMarker(fileMarkerEndSection, std::wstring(L"EMAT"));
            return stream;
        }
    };

    typedef GPUMatrix<float> GPUSingleMatrix;

}}}

