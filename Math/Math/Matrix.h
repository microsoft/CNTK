//
// <copyright file="Matrix.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// TODO:
//  - remove empty-matrix checks: if an op is well-defined with empty matrices, then do it
//  - Resize() must be cheap if it does nothing  (I already did that for CPU, still to be done for GPU)
//  - an overload for Resize() to match another matrix
//  - need a way to grow a minibatch matrix without destroying its content, something like PushColumns()

#pragma once

#ifdef    _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else    // no DLLs on Linux
#define    MATH_API 
#endif

#include "Basics.h"
#include "File.h"
#include "CommonMatrix.h"
#include <limits.h>
#include <memory>   // for shared_ptr

// This class is exported from the Math.dll
namespace Microsoft { namespace MSR { namespace CNTK {


    enum CurrentDataLocation
    {
        NONE, CPU, GPU, BOTH
    };

    enum MatrixType
    { 
       UNDETERMINED, DENSE, SPARSE
    };

    // TODO: create an <ElemType>-agnostic base class, then move generic functions such as getting dims, resizing, and getting/setting as scalars
    class MATH_API MatrixBase
    {
    protected:
        //virtual ~MatrixBase() { };
        // TODO: currently this causes link errors when building DLLs
    };

    // avoid pulling in these header files for consumers of this class
    template<class ElemType> class GPUMatrix;
    template<class ElemType> class CPUMatrix;
    template<class ElemType> class GPUSparseMatrix;
    template<class ElemType> class CPUSparseMatrix;
    template<class ElemType> class DeviceBoundNumber;

    //To compy with BLAS libraries matrices are stored in ColMajor. However, by default C/C++/C# use RowMajor
    //convertion is need when passing data between Matrix and C++ matrices
    //For the best performance compile CNTKMath project with NO_SYNC preprocessor directive
    //!!!WARNING!!! This class is NOT THREAD SAFE. Test and add necessary modifications if using in multi-threaded environment    
    template<class ElemType>
    class MATH_API Matrix : public MatrixBase
    {
    private:
        mutable BaseMatrix<ElemType> *m_baseMatrix;
        mutable GPUMatrix<ElemType> *m_GPUMatrix;
        mutable CPUMatrix<ElemType> *m_CPUMatrix;
        mutable GPUSparseMatrix<ElemType> *m_GPUSparseMatrix;
        mutable CPUSparseMatrix<ElemType> *m_CPUSparseMatrix;
        mutable MatrixType m_matrixType;
        mutable CurrentDataLocation m_currentDataLocation; //Indicates which matrix is current        
        mutable DEVICEID_TYPE m_preferredDeviceId;

        mutable size_t m_numTimesDeviceChanged;
        mutable size_t m_numTimesMatrixTypeChanged;
        mutable int m_devicesTransferedTo[2];
            
        //Moves matrix from device id_from to device with id_to. This method doesn't change preferred device Id
        void _transferFromDeviceToDevice(int id_from, int id_to, bool ismoved=true,bool emptyTransfer=false) const; 
        //Moves matrix from current device to device with id_to. This method doesn't change preferred device Id
        void _transferToDevice(int id_to, bool ismoved=true, bool emptyTransfer=false) const; 
        static void DecideAndMoveToRightDevice(const Matrix<ElemType>& a, const Matrix<ElemType>& b);
        static void DecideAndMoveToRightDevice(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c);
        static void CopyElementsFromDenseToSparse(CPUMatrix<ElemType>& from, CPUSparseMatrix<ElemType>& dest);

    public:
        //Constructors, destructors and other static matrix builders
        //Each constructor can take deviceId as parameter.
        //If deviceId<0 then the matrix will be based in RAM (CPUMatrix)
        //Elseif deviceId>=0 and <AUTOPLACEMATRIX, then the matrix will be based on GPU with specified deviceId
        //Else (default) if deviceId=AUTOPLACEMATRIX, the class will try to place itself on the best GPU, if fails it will go to CPU
        //The default behaiviour should be deviceId=AUTOPLACEMATRIX        
        Matrix(DEVICEID_TYPE deviceId=AUTOPLACEMATRIX); 
        Matrix(BaseMatrix<ElemType>* baseMatrix, ElemType *pArray, DEVICEID_TYPE deviceId); // constructor for setting Matrix from a base matrix (externally managed butter pArray)
        Matrix(FILE* f, const char * matrixName, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const MatrixType matrixType = DENSE); //matrixName is used to verify that correct matrix is read.
        Matrix(const size_t numRows, const size_t numCols, DEVICEID_TYPE deviceId = AUTOPLACEMATRIX, const MatrixType matrixType = DENSE, const MatrixFormat matrixFormat = matrixFormatDense);
        Matrix(const size_t numRows, const size_t numCols, ElemType *pArray, const size_t matrixFlags=matrixFlagNormal, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const size_t nnz=0);
        Matrix(const Matrix<ElemType>& deepCopyFrom, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);  //copy constructor, deep copy
        Matrix<ElemType>& operator=(const Matrix<ElemType>& deepCopyFrom);  //assignment operator, deep copy
        Matrix(Matrix<ElemType>&& moveFrom);  //move constructor, shallow copy
        Matrix<ElemType>& operator=(Matrix<ElemType>&& moveFrom);  //move coment operator, shallow copy

        static Matrix<ElemType> Ones(const size_t rows, const size_t cols, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);
        static Matrix<ElemType> Zeros(const size_t rows, const size_t cols, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);
        static Matrix<ElemType> Eye(const size_t rows, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);

#define USE_TIME_BASED_SEED ULONG_MAX
        static Matrix<ElemType> RandomUniform(const size_t rows, const size_t cols, const ElemType low, const ElemType high, unsigned long seed = USE_TIME_BASED_SEED, DEVICEID_TYPE deviceId = AUTOPLACEMATRIX);
        static Matrix<ElemType> RandomGaussian(const size_t rows, const size_t cols, const ElemType mean, const ElemType sigma, unsigned long seed=USE_TIME_BASED_SEED, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);

        void Clear();
        ~Matrix();

    private:
        Matrix(const MatrixFlags matrixFlags, const MatrixType matrixType, const MatrixFormat matrixFormat, DEVICEID_TYPE deviceID); //only used internally to initialize a blank matrix
        Matrix(const MatrixFlags matrixFlags, const MatrixType matrixType, DEVICEID_TYPE deviceID); //only used internally to initialize a blank matrix
        Matrix(const MatrixFlags matrixFlags, DEVICEID_TYPE deviceID); //only used internally to initialize a blank matrix
        void Init(DEVICEID_TYPE deviceID); //only used internally to initialize a blank matrix
        void SetDataLocation(CurrentDataLocation location, MatrixType type=UNDETERMINED) const;

    public:
        MatrixType GetMatrixType() const {return m_matrixType;};
        MatrixFormat GetFormat() const { return m_baseMatrix->GetFormat(); }
        bool OwnBuffer() const {return m_baseMatrix->OwnBuffer();}
        int GetDeviceId() const; //-1 if CPU, otherwise GPU CUDA device id
        DEVICEID_TYPE GetPreferredDeviceId() const { return m_preferredDeviceId; }; //-1 if CPU, otherwise GPU CUDA device id
        void SetPreferredDeviceId(DEVICEID_TYPE preferredDeviceId){ if (m_preferredDeviceId != preferredDeviceId) m_preferredDeviceId = preferredDeviceId; }
        //Moves matrix from device id_from to device with id_to. 
        //If emptyTransfer=true, then no data is ever moved, just corresponding GPU/CPU matrices are deleted and then created using empty constructor
        void TransferFromDeviceToDevice(int id_from, int id_to, bool ismoved = false,/*if false then keep source and set location to BOTH*/ bool emptyTransfer = false, bool updatePreferredDevice = true) const;
        //Same as TransferFromDeviceToDevice() but moves only if it is currently not on the target device
        void TransferToDeviceIfNotThere(int id_to, bool ismoved = false, bool emptyTransfer = false, bool updatePreferredDevice = true) const;
        void TransferToDeviceIfNotThereAndNotAutoPlace(int id_to, bool ismoved = false, bool emptyTransfer = false, bool updatePreferredDevice = true) const;
        CurrentDataLocation GetCurrentMatrixLocation() const { return m_currentDataLocation; };
        void SwitchToMatrixType(const MatrixType newMatrixType, const MatrixFormat newMatrixFormat, const bool keepValues); //sets matrix type between dense and sparse
        size_t GetNumRows() const;
        size_t GetNumCols() const;
        size_t GetNumElements() const;
        bool HasNoElements() const { return GetNumElements() == 0; }
        wchar_t* GetMatrixName() const;
        void SetMatrixName(const wchar_t* s);
        bool IsEmpty() const;  
        size_t BufferSize() const;
        ElemType* BufferPointer() const;
        size_t NzCount() const;

        ElemType* CopyToArray() const; //allocated by the callee but need to be deleted by the caller
        size_t CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const;  //allocated by the callee but need to be deleted by the caller

        Matrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;

        // difference between AssignColumnSlice and SetColumnSlice 
        // AssignColumnSlice :      this(:, startColumn:startColumn+numCols-1) = fromMatrix(:, startColumn: startColumn+numCols-1) 
        // SetColumnSlice    :      this(:, startColumn:startColumn+numCols-1) = fromMatrix(:, 0: startColumn+numCols-1) 
        // AssignColumnSlice do not transfer data, it uses external data
        // SetColumnSlice    copies data 

        Matrix<ElemType>& AssignColumnSlice(const Matrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols);
        Matrix<ElemType>& SetColumnSlice(const Matrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols);

        void ShiftBy(int numShift);

        // TODO: all these scalars should be passed as doubles and cast down inside
        void NormalGrad(Matrix<ElemType>& gradients, Matrix<ElemType>& functionValues, const ElemType learnRatePerSample, const ElemType momentum);
        ElemType Adagrad(Matrix<ElemType>& gradients, const bool needAveMultiplier);
        void FSAdagrad(size_t mbSize, Matrix<ElemType>& gradients, Matrix<ElemType>& functionValues, const ElemType learnRatePerSample, const ElemType momentum);
        ElemType RmsProp(Matrix<ElemType>& gradients, ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN, const bool needAveMultiplier);

        // TODO: should Reshape() return a new Matrix object that contains a reference to the original?
        void Reshape(const size_t numRows, const size_t numCols);
        void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, bool growOnly = true);  //by default we only reallocate if need to grow        

        // update number of columns
        // TODO: a future version may want to enforce retaining the content, to allow dynamically growing layouts column by column (when size is not known upfront)
        void ResizeColumns(const size_t numCols) { Resize(GetNumRows(), numCols); }

        // similarl to the repmat operation in matlab or octave
        static Matrix<ElemType> RepMat(const Matrix<ElemType>& frmMat, const size_t rows, const size_t cols);

        size_t GetAllocatedSize() const;
        void Reset(); // reset for sparse matrix

        const ElemType operator() (const size_t row, const size_t col) const;
        ElemType& operator() (const size_t row, const size_t col);
        ElemType Get00Element() const;

        void SetValue(const ElemType v);
        void SetValue(const DeviceBoundNumber<ElemType>& db_number);
        void SetValue(const Matrix<ElemType>& deepCopyFrom, const MatrixFormat format=matrixFormatSparseCSR);
        void SetValue(const size_t numRows, const size_t numCols, ElemType *pArray, const size_t matrixFlags=matrixFlagNormal, int deviceId=MANAGEDEXTERN);
        void SetValue(const size_t rIdx, const size_t cIdx, ElemType val);  // set matrix sparsely
        void SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE *h_CSCCol, const CPUSPARSE_INDEX_TYPE *h_Row, const ElemType *h_Val,
            const size_t nz, const size_t numRows, const size_t numCols);

        void SetColumn(const ElemType* colPointer, size_t colInd);
        void SetColumn(const ElemType val, size_t colInd);
        void SetColumn(const Matrix<ElemType>& valMat, size_t colInd);

        void SetDiagonalValue(const ElemType v);
        void SetDiagonalValue(Matrix<ElemType>& vector);
        void SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed=USE_TIME_BASED_SEED);
        void SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed=USE_TIME_BASED_SEED);
        void SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, unsigned long seed=USE_TIME_BASED_SEED); 
        void AddGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed=USE_TIME_BASED_SEED);
        Matrix<ElemType>& AssignNoiseContrastiveEstimation(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c, const Matrix<ElemType>& bias, Matrix<ElemType>& tmp);

        Matrix<ElemType>& AssignNCEDerivative(const Matrix<ElemType>& tmp, const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c, size_t inputIndex);
        Matrix<ElemType>& AssignSoftmaxSum(const Matrix<ElemType>& a, const Matrix<ElemType>& softmax);
        Matrix<ElemType>& AssignNceUnnormalizedEval(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c, const Matrix<ElemType>& bias);

        Matrix<ElemType> Transpose(); // This method doesn't change state of Matrix. It should be a const function
        Matrix<ElemType>& AssignTransposeOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& operator+= (const ElemType alpha);
        Matrix<ElemType> operator+ (const ElemType alpha) const;
        Matrix<ElemType>& AssignSumOf(const ElemType alpha, const Matrix<ElemType>& a);

        Matrix<ElemType>& operator+= (const Matrix<ElemType>& a);
        Matrix<ElemType> operator+ (const Matrix<ElemType>& a) const;
        Matrix<ElemType>& AssignSumOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b);

        Matrix<ElemType>& operator-= (const ElemType alpha);
        Matrix<ElemType> operator- (const ElemType alpha) const;
        Matrix<ElemType>& AssignDifferenceOf(const ElemType alpha, const Matrix<ElemType>& a);
        Matrix<ElemType>& AssignDifferenceOf(const Matrix<ElemType>& a, const ElemType alpha);

        Matrix<ElemType>& operator-= (const Matrix<ElemType>& a);
        Matrix<ElemType> operator- (const Matrix<ElemType>& a) const;
        Matrix<ElemType>& AssignDifferenceOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b);

        Matrix<ElemType>& operator*= (const ElemType alpha);
        Matrix<ElemType> operator* (const ElemType alpha) const;
        Matrix<ElemType>& AssignProductOf(const ElemType alpha, const Matrix<ElemType>& a);

        Matrix<ElemType> operator* (const Matrix<ElemType>& a) const;
        Matrix<ElemType>& AssignProductOf (const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB);

        Matrix<ElemType>& operator/= (ElemType alpha);
        Matrix<ElemType> operator/ (ElemType alpha) const;        

        Matrix<ElemType>& operator^= (ElemType alpha); //element-wise power
        Matrix<ElemType> operator^ (ElemType alpha) const; //element-wise power
        Matrix<ElemType>& AssignElementPowerOf(const Matrix<ElemType>& a, const ElemType power);

        Matrix<ElemType>& ElementMultiplyWith (const Matrix<ElemType>& a);
        Matrix<ElemType>& AssignElementProductOf (const Matrix<ElemType>& a, const Matrix<ElemType>& b);
        Matrix<ElemType>& AddElementProductOf (const Matrix<ElemType>& a, const Matrix<ElemType>& b);

        Matrix<ElemType>& AssignElementDivisionOf (const Matrix<ElemType>& a, const Matrix<ElemType>& b);
        Matrix<ElemType>& ElementDivideBy(const Matrix<ElemType>& a);

        Matrix<ElemType>& ColumnElementMultiplyWith(const Matrix<ElemType>& a);
        Matrix<ElemType>& RowElementMultiplyWith(const Matrix<ElemType>& a);

        Matrix<ElemType>& ColumnElementDivideBy(const Matrix<ElemType>& a);
        Matrix<ElemType>& RowElementDivideBy(const Matrix<ElemType>& a);

        Matrix<ElemType>& ElementInverse ();
        Matrix<ElemType>& AssignElementInverseOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceLinearRectifierDerivative();
        Matrix<ElemType>& AssignLinearRectifierDerivativeOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceSigmoidDerivative();
        Matrix<ElemType>& AssignSigmoidDerivativeOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceSigmoid ();
        Matrix<ElemType>& AssignSigmoidOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceTanh ();
        Matrix<ElemType>& AssignTanhOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceLogSoftmax (const bool isColWise);
        Matrix<ElemType>& AssignLogSoftmaxOf (const Matrix<ElemType>& a, const bool isColWise);

		//sequence training 
		Matrix<ElemType>& DropFrame(const Matrix<ElemType>& label, const Matrix<ElemType>& gamma, const ElemType & threshhold);
		Matrix<ElemType>& AssignSequenceError(const ElemType hsmoothingWeight, const Matrix<ElemType>& label, const Matrix<ElemType>& dnnoutput, const Matrix<ElemType>& gamma, ElemType alpha);
        Matrix<ElemType>& InplaceSqrt ();
        Matrix<ElemType>& AssignSqrtOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceExp ();
        Matrix<ElemType>& AssignExpOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceLog ();
        Matrix<ElemType>& AssignLogOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceCosine ();
        Matrix<ElemType>& AssignCosineOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceNegativeSine ();
        Matrix<ElemType>& AssignNegativeSineOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceLog10 ();
        Matrix<ElemType>& AssignLog10Of (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceAbs ();
        Matrix<ElemType>& AssignAbsOf (const Matrix<ElemType>& a);

        Matrix<ElemType>& InplaceTruncateBottom (const ElemType threshold);
        Matrix<ElemType>& AssignTruncateBottomOf (const Matrix<ElemType>& a, const ElemType threshold);
        Matrix<ElemType>& InplaceTruncateTop (const ElemType threshold);
        Matrix<ElemType>& AssignTruncateTopOf (const Matrix<ElemType>& a, const ElemType threshold);
        Matrix<ElemType>& InplaceTruncate (const ElemType threshold);
        Matrix<ElemType>& InplaceSoftThreshold(const ElemType threshold);

        Matrix<ElemType>& SetToZeroIfAbsLessThan (const ElemType threshold);

        DeviceBoundNumber<ElemType> Sum_AsDeviceBoundNum() const;
        ElemType SumOfAbsElements () const; //sum of all abs(elements)
        ElemType SumOfElements () const; //sum of all elements
        Matrix<ElemType>& AssignSumOfElements(const Matrix<ElemType>& a);

        ElemType LogAddSumOfElements() const;

        Matrix<ElemType>&  AssignToRowSliceValuesOf(const Matrix<ElemType>& a, const size_t startIndex, const size_t numRows);
        Matrix<ElemType>&  AssignRowSliceValuesOf(const Matrix<ElemType>& a, const size_t startIndex, const size_t numRows);
        Matrix<ElemType>&  AddToRowSliceValuesOf(const Matrix<ElemType>& a, const size_t startIndex, const size_t numRows); 
        Matrix<ElemType>&  AddWithRowSliceValuesOf(const Matrix<ElemType>& a, const size_t startIndex, const size_t numRows);
        Matrix<ElemType>&  AssignRowStackValuesOf(const std::vector<const Matrix<ElemType>*>& inputMatrices, const size_t sliceStartCol, const size_t sliceNumCols);

        Matrix<ElemType>&  AssignRepeatOf(const Matrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats);
        Matrix<ElemType>&  AddToRowRepeatValuesOf(const Matrix<ElemType>& a, const size_t numRepeats);

        Matrix<ElemType>&  AssignPositiveAndShiftedNegSample(const Matrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber);
        Matrix<ElemType>&  AddFoldedPositiveAndShiftedNegSample(const Matrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber);
        
        bool IsEqualTo(const Matrix<ElemType>& a, const ElemType threshold = 1e-8) const;

        static void VectorSum(const Matrix<ElemType>& a, Matrix<ElemType>& c, const bool isColWise);

        void VectorNorm1(Matrix<ElemType>& c, const bool isColWise) const;
        Matrix<ElemType>& AssignVectorNorm1Of(Matrix<ElemType>& a, const bool isColWise);

        void VectorNorm2(Matrix<ElemType>& c, const bool isColWise) const;
        Matrix<ElemType>& AssignVectorNorm2Of(Matrix<ElemType>& a, const bool isColWise);

        void VectorNormInf(Matrix<ElemType>& c, const bool isColWise) const;
        Matrix<ElemType>& AssignVectorNormInfOf(Matrix<ElemType>& a, const bool isColWise);

        Matrix<ElemType>& AssignInnerProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const bool isColWise);
        Matrix<ElemType>& AssignKhatriRaoProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b);
        Matrix<ElemType>& AddColumnReshapeProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const bool transposeAColumn);

        Matrix<ElemType>& AddWithScaleOf(ElemType alpha, const Matrix<ElemType>& a);

        ElemType FrobeniusNorm() const;
        Matrix<ElemType>& AssignFrobeniusNormOf(const Matrix<ElemType>& a);

        ElemType MatrixNormInf() const;
        ElemType MatrixNorm1() const;
        ElemType MatrixNorm0() const; //number of non-zero elemets
        Matrix<ElemType>& AssignSignOf(const Matrix<ElemType>& a);
        Matrix<ElemType>& AddSignOf(const Matrix<ElemType>& a);
        void VectorMax(Matrix<ElemType>& maxIndexes, Matrix<ElemType>& maxValues, const bool isColWise) const;
        void VectorMin(Matrix<ElemType>& mainndexes, Matrix<ElemType>& minValues, const bool isColWise) const;

        Matrix<ElemType>&  AssignNumOfDiff(const Matrix<ElemType>& a, const Matrix<ElemType>& b); 

        Matrix<ElemType>& AssignInnerProductOfMatrices(const Matrix<ElemType>& a, const Matrix<ElemType>& b); //this method will resize(1,1) first

        bool HasNan (const char * name) const;
        size_t CountNanInf() const;

        void Print(const char* matrixName, size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd) const;
        void Print(const char* matrixName = nullptr) const; //print whole matrix. can be expensive

        Matrix<ElemType>& AssignPackedConvolutionInput(const Matrix<ElemType>& inputSubBatch, 
                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample, 
                                                 const bool zeroPadding = false); 
        Matrix<ElemType>& UnpackConvolutionInput(Matrix<ElemType>& inputSubBatch, 
                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample, 
                                                 const bool zeroPadding = false) const; 
        Matrix<ElemType>& AssignMaxPoolingResult(const Matrix<ElemType>& inputBatch, const size_t channels, 
                                                 const size_t inputWidth, const size_t inputHeight,  const size_t inputSizePerSample, 
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
        Matrix<ElemType>& AddMaxPoolingGradient(const Matrix<ElemType>& outputGradientBatch, const Matrix<ElemType>& inputBatch, const Matrix<ElemType>& outputBatch, 
                                                 const size_t channels, 
                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample, 
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
        Matrix<ElemType>& AssignAveragePoolingResult(const Matrix<ElemType>& inputBatch, const size_t channels, 
                                                 const size_t inputWidth, const size_t inputHeight,  const size_t inputSizePerSample, 
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
        Matrix<ElemType>& AddAveragePoolingGradient(const Matrix<ElemType>& outputGradientBatch, 
                                                 const size_t channels, 
                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample, 
                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample, 
                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample);
    public:
        ElemType Exp10(ElemType num); 
        ElemType Mod(ElemType x , ElemType y);
        ElemType LogAdd(ElemType x, ElemType y);

    public:
        static DEVICEID_TYPE GetBestGPUDeviceId(); //{ return GPUMatrix<ElemType>::GetBestGPUDeviceId();}

        //static BLAS functions

        // singular value decomposition of A as A = U*SIGMA*VT
        static void SVD(const Matrix<ElemType>& A, Matrix<ElemType>& SIGMA, Matrix<ElemType>& U, Matrix<ElemType>& VT, Matrix<ElemType>& W);

        static void MultiplyAndWeightedAdd(ElemType alpha, const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB, 
            ElemType beta, Matrix<ElemType>& c);
        static void MultiplyAndAdd(const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB, Matrix<ElemType>& c);
        static void Multiply(const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB, Matrix<ElemType>& c);
        static void Multiply(const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c);

        static void ScaleAndAdd(ElemType alpha, const Matrix<ElemType>& a, Matrix<ElemType>& c);
        static void ScaleAndAdd(ElemType alpha, const Matrix<ElemType>& a, ElemType beta, Matrix<ElemType>& c);
        static void AddScaledDifference(const ElemType alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c);
        static void AssignScaledDifference(const ElemType alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c);
        static void AddScaledDifference(const Matrix<ElemType>& alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c);
        static void AssignScaledDifference(const Matrix<ElemType>& alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c);

        static void AddElementToElement(const Matrix<ElemType>& a, const size_t ai, const size_t aj, Matrix<ElemType>& c, const size_t ci, const size_t cj); 
        //static void AddLogElementToElement(const Matrix<ElemType>& a, const size_t ai, const size_t aj, Matrix<ElemType>& c, const size_t ci, const size_t cj); 
        static void AssignElementToElement(const Matrix<ElemType>& a, const size_t ai, const size_t aj, Matrix<ElemType>& c, const size_t ci, const size_t cj); 
        static void MinusOneAt(Matrix<ElemType>& c, const size_t position);

        static void Scale(ElemType alpha, Matrix<ElemType>& a);
        static void Scale(const Matrix<ElemType>& alpha, Matrix<ElemType>& a); //In this case Matrix alpha must be 1x1
        static void Scale(ElemType alpha, const Matrix<ElemType>& a, Matrix<ElemType>& c);
        static void InnerProduct (const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c, const bool isColWise);
        static ElemType InnerProductOfMatrices(const Matrix<ElemType>& a, const Matrix<ElemType>& b);
        static void ElementWisePower (ElemType alpha, const Matrix<ElemType>& a, Matrix<ElemType>& c);

        static bool AreEqual(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const ElemType threshold = 1e-8);
        static bool HasElement(const Matrix<ElemType>& a, const ElemType value = 0.0);

    public:
        void Read(File& stream);
        void Write(File& stream) const;

        Matrix<ElemType>& Shift(const Matrix<ElemType>& a, int shift);

        Matrix<ElemType>& AssignElementProductOfWithShiftNeg(const Matrix<ElemType>& a, const Matrix<ElemType>& b, size_t shift, size_t negnumber);
        Matrix<ElemType>& AssignInnerProductOfWithShiftNeg(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const bool isColWise, size_t shift, size_t negnumber);
        static void InnerProductWithShiftNeg(const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c, const bool isColWise, size_t shift, size_t negnumber);
        Matrix<ElemType>& GetARowByIndex(const Matrix<ElemType>& a, size_t index);
        static void ConductRowElementMultiplyWithShift(const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c, size_t shift, bool bFirstmatrixfixed);
        Matrix<ElemType>& AssignElementProductOfWithShift(const Matrix<ElemType>& a, const Matrix<ElemType>& b, size_t shift);

    public:
        static void RCRFBackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& beta,
            Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls,
            const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const int shift);

        static void RCRFTransGrdCompute(const Matrix<ElemType>& lbls,
            const Matrix<ElemType>&   alpha,
            const Matrix<ElemType>& beta,
            const Matrix<ElemType>& pair_scores,
            Matrix<ElemType>& grd,
            const int startLbl, /// the time 0 start symbol in the output layer
            const int shift);

        template<typename T>
        friend class MatrixQuantizer;

        template<typename T>
        friend class QuantizedMatrix;
    };

    // overload I/O operators
    template<class ElemType>
    File& operator>>(File& stream, Matrix<ElemType>& M) { M.Read(stream); return stream; }
    template<class ElemType>
    File& operator<<(File& stream, const Matrix<ElemType>& M) { M.Write(stream); return stream; }

    typedef Matrix<float> SingleMatrix;
    typedef Matrix<double> DoubleMatrix;

    // MBLayout -- layout information of minibatch
    // This stores:
    //  - number of time steps and parallel sequences (their product is equal to the #columns in the minibatch)
    //  - whether the data is sequential or not
    //  - MinibatchPackingFlags for every (sequence, time step)
    //  - a column-wise OR of those flags for fast testing entire time steps at once
    // This object allocates its storage lazily, i.e. if there are no flags ever set, no memory is allocated. This is transparent to the caller.
    // Note: With truncated BPTT, it is possible to have sequential data, yet not a single flag set in a minibatch (if all frames are sequence-internal ranges).
    // Contract between ComputationNode, ComputationNetwork, and MBLayout:
    //  - if a node has no MBLayout, m_{function,gradient}Values are not samples (they are not activations or input data), but e.g. model parameters
    //  - ComputationNode::GetNumCols() == MBLayout::GetNumTimeSteps() * MBLayout::GetNumParallelSequences()
    //  - ComputationNetwork ensures that m_{function,gradient}Values are allocated correctly before calling EvaluateThisNode() on a node
    // TODO: move this to an appropriate place and name it properly. This class has no relationship with Matrix
    // NOTE: This class represents an ongoing abstraction of an originally distributed/code-duped way of defining and accessing the MB layout.
    //       Some code below represents the actual use cases I encountered. Not all are, I believe, needed to be as they are; this class could be simplified/streamlined much further.
    //       Some wackiness below is explained by this.
    // TODO: frame-randomized MBs are now represented as one stream of many frames. This is wrong; they should be one-frame utterances with many streams. Once we fully abstract out Data access, this can be changed easily.
    struct MBLayout
    {
        typedef std::shared_ptr<MBLayout> MBLayoutPtr;

        MBLayout() : m_sentenceBoundaryFlags(CPUDEVICE) { Init(1, 0, false); }
        MBLayout(size_t numParallelSequences, size_t numTimeSteps, bool dataIsSequential) : m_sentenceBoundaryFlags(CPUDEVICE) { Init(numParallelSequences, numTimeSteps, dataIsSequential); }

        // copy the content of another MBLayoutPtr over
        // Use this instead of actual assignment to make it super-obvious that this is not copying the pointer but actual content. The pointer is kept fixed.
        void CopyFrom(const MBLayoutPtr & other) { *this = *other; }
        void MoveFrom(MBLayoutPtr other) { *this = move(*other); other->Init(0, 0, false); }    // destructive copy that steals ownership if the content, like std::move()
    private:
        MBLayout & operator=(const MBLayout &) = default;   // make this private --use CopyFrom() instead, which makes it very clear that it's copying content, not copying the reference
    public:

        // resize and reset all frames to None (note: this is an invalid state and must be fixed by caller afterwards)
        void Init(size_t numParallelSequences, size_t numTimeSteps, bool dataIsSequential)
        {
            // remember the dimensions..
            m_numParallelSequences = numParallelSequences;
            m_numTimeSteps = numTimeSteps;
            m_dataIsSequential = dataIsSequential;
            // ...but don't actually allocate anything
            m_sentenceBoundaryFlags.Resize(0, 0);
            m_minibatchPackingFlags.clear();
        }

        size_t GetNumTimeSteps()         const { return m_numTimeSteps; }
        size_t GetNumParallelSequences() const { return m_numParallelSequences; }   // note: if initialized as a dummy, m_numParallelSequences is set to 1

    private:
        // test whether we have not allocated anything (will also return true if the minibatch is empty)
        bool IsEmpty() const { return m_minibatchPackingFlags.empty(); }
        // call this before ever writing anything--this will create the matrix/vector upon first use
        void LazyAlloc() const
        {
            if (!IsEmpty() || m_numTimeSteps == 0)
                return;
            // this is where the actual allocation happens
            m_sentenceBoundaryFlags.Resize(m_numParallelSequences, m_numTimeSteps);
            m_sentenceBoundaryFlags.SetValue((float)((int)MinibatchPackingFlags::None));
            m_minibatchPackingFlags.assign(m_sentenceBoundaryFlags.GetNumCols(), MinibatchPackingFlags::None);
        }
    public:

        // compare whether two layouts are the same
        bool operator==(const MBLayout & other) const
        {
            // for now just check the object identity
            // TODO: in the future, we also need to compare the content; and we need to define "equal", e.g. w.r.t. missing features
            return this == &other;
        }

        // get boundary flags
        MinibatchPackingFlags Get(size_t t) const { return IsEmpty() ? MinibatchPackingFlags::None : m_minibatchPackingFlags[t]; }
        MinibatchPackingFlags Get(size_t id, size_t t) const { return IsEmpty() ? MinibatchPackingFlags::None : (MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(id, t); }

        // test boundary flags for a specific condition
        bool Is(size_t t, MinibatchPackingFlags f) const { return (Get(t) & f) != 0; }
        bool Is(size_t id, size_t t, MinibatchPackingFlags f) const { return (Get(id, t) & f) != 0; }
        // TODO: swap id and t for all of these functions; t is the more important parameter

        // tests if Is() is false for every frame and sequence
        // If this returns true, it means that boundary information need not be considered, just process the whole thing in one go.
        // TODO: Can it ever happen that no flag is set, yet we have m_numParallelSequences != 1? Or does that simply not matter?
        // This is currently the case for frame randomization.
        bool IsAllNone() const { return IsEmpty(); }

        // set a boundary flag (OR it on top of the existing layout)
        void Set(size_t id, size_t t, MinibatchPackingFlags f)
        {
            if (f == MinibatchPackingFlags::None)   // actually not setting anything: skip allocation
                return;
            if ((f & (MinibatchPackingFlags::SequenceStart | MinibatchPackingFlags::SequenceEnd)) && !m_dataIsSequential)
                LogicError("MBLayout::Set: attempted to set SequenceStart or -End in a layout with !m_dataIsSequential");
            LazyAlloc();
            m_sentenceBoundaryFlags.SetValue(id, t, (float)(((MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(id, t)) | f));
            m_minibatchPackingFlags[t] |= f;
        }

        bool RequireSentenceSeg() const { return m_dataIsSequential; }        // this is the name of a function on DataReader which really belongs here

    private:
        size_t m_numTimeSteps;
        size_t m_numParallelSequences;
        bool m_dataIsSequential;
        // TODO: ^^ is m_dataIsSequential necessary? Can it be derived from, say, m_numTimeSteps == 1 && IsAllNone()?

        /// a matrix of n_stream x n_length
        /// n_stream is the number of streams
        /// n_length is the maximum lenght of each stream
        /// for example, two sentences used in parallel in one minibatch would be
        /// [2 x 5] if the max length of one of the sentences is 5
        /// the elements of the matrix is 0, 1, or -1, defined as ((int) MinibatchPackingFlags::SequenceStart), ((int) MinibatchPackingFlags::None), ((int) MinibatchPackingFlags::NoInput) in cbasetype.h 
        /// 0 1 1 0 1
        /// 1 0 1 0 0 
        /// for two parallel data streams. The first has two sentences, with 0 indicating begining of a sentence
        /// the second data stream has two sentences, with 0 indicating begining of sentences
        /// you may use 1 even if a sentence begins at that position, in this case, the trainer will carry over hidden states to the following
        /// frame. 
        mutable Matrix<float> m_sentenceBoundaryFlags;  // (t,stream)
        // ^^ float -> MinibatchPackingFlags, right? Or unsigned char; or change that to 'char' because Matrix<char> already exists
        // This matrix ^^ is always in CPU memory  --TODO: should rather be a matrix of some int
        /// conditionally point to either a pointer to that provided by network, or point to 
        /// an individual sentence boundary info, which happens if timeStep > 1 is required for PastValue node
        /// a matrix of 1 x n_length
        /// != 0 denotes the case that there exists sentence begin or no_labels case in this frame
        /// == 0 denotes such case is not in this frame
        mutable vector<MinibatchPackingFlags> m_minibatchPackingFlags;  // column-wise OR over m_sentenceBoundaryFlags for fast testing

    public:
        // specialized functions to replicate old behavior that shouldn't be there but I cannot test
        // TODO: these should all go away one day

        // get info for one frame; used in DelayedValueNode
        // TODO: clean this up, we can do this more nicely. DelayedValueNode can just access individual elements, like everybody else.
        pair<Matrix<float>, MinibatchPackingFlags> GetFrame(size_t t) const
        {
            LazyAlloc();
            return make_pair(m_sentenceBoundaryFlags.ColumnSlice(t, 1), m_minibatchPackingFlags[t]);
        }

        // same as Set() but not ORing  --TODO: is this distinction needed?
        void SetWithoutOr(size_t id, size_t t, MinibatchPackingFlags f)
        {
            if (f == MinibatchPackingFlags::None)
                return;
            LazyAlloc();
            m_sentenceBoundaryFlags.SetValue(id, t, (float)(int)f); // no OR
            m_minibatchPackingFlags[t] |= f;
        }
        // needed in DelayedValueNodeBase
        // TODO: this is wicked in that the matrix keeps only the NoLabel flag, while the vector keeps all (just gets ORed into)
        void Mask(size_t id, size_t t, MinibatchPackingFlags f)
        {
            if (IsEmpty())
                return;
            m_sentenceBoundaryFlags.SetValue(id, t, (float)(((MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(id, t)) & f));
            //m_minibatchPackingFlags[t] &= f;
        }
        // for LSTMNode ony, which is deprecated, only to make it compile easily:  also used in FindBestPathWithVariableLength() and FindBestPath() in a strange way
        Matrix<float> & GetM() { LazyAlloc(); return m_sentenceBoundaryFlags; }
    };
    typedef MBLayout::MBLayoutPtr MBLayoutPtr;

    // there is a version of ColumnSlice() in ComputationNode that abstracts the number of streams
    // TODO: This may not belong here, but having it in ComputeNode would require syntax changes, while having it as a member here only requires a local find-replace. Let's make it work first, then decide how to refactor.
    // the looping versions of EvaluateThisNode(FrameRange()) and ComputeInputPartial() take a frame range, through this structure
    // It can cast from a size_t, i.e. those functions can be called passing a size_t in place of the FrameRange.
    // TODO: GetNumParallelSequences() should be subsumed here & removed from nodes
    // TODO: We should also have a FrameRange that selects a single sequence instead of all.
    // TODO: Where this design currently breaks:
    //  - BatchModeNodes must access GetNumParallelSequences(), yet operate on the whole sequence
    //  - likewise, LSTMNode does its own iteration, hence needs access to GetNumParallelSequences() or NumCols() in the whole-batch iterator
    //  - RecurrentNodes access frames with a time shift, where out-of-bounds ones access a different matrix' values
    //  - RecurrentNodes iterate over individual slices--need a sub-setting constructor from a FrameRange to another?
    //  - RecurrentNodes access boundary info with a similar pattern, but boundary info has a different #streams (namely, 1)
    // TODO: This will in the future be able to hold sub-ranges for nested loops as well.
    // BUGBUG: These are currently broken and will need to be fixed:
    //  - ClassBasedCrossEntropyWithSoftmaxNode:
    //      FrameRange frameRange(t, 1);
    //    using a different #sequences. Solve by treating all frames as one sequence (in FrameRange)
    //  - ReshapeNode:
    //      Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check(frameRange.t() * outputSamplesInRecurrentStep, outputSamplesInRecurrentStep, m_pMBLayout));
    //    using a differeren #sequences. Find out what this really means.
    struct FrameRange
    {
        const size_t timeIdxInSeq;              // start frame
        const size_t samplesInRecurrentStep;    // number of samples in this step       --BUGBUG: this should be part of MBLayout, not FrameRange
        // can construct from a single size_t -> a single-frame range
        //FrameRange(size_t timeIdxInSeq) : timeIdxInSeq(timeIdxInSeq), samplesInRecurrentStep(0)/*FIX THIS*/{}
        FrameRange(size_t timeIdxInSeq, size_t samplesInRecurrentStep) : timeIdxInSeq(timeIdxInSeq), samplesInRecurrentStep(samplesInRecurrentStep){}
        // or without arguments -> entire minibatch / no frame-range
        FrameRange() : timeIdxInSeq(0), samplesInRecurrentStep(SIZE_MAX/*all frames (map)*/) {}
        // code that can only handle single-frame ranges will call t() to get the time index, which will throw if numFrames != 1
        // Some functions need just the time index, e.g. for looking up stuff in m_boundaryInfo. That's where an unscaled index is needed (as opposed to startColumn()).
        size_t t() const { EnsureNotAllFrames(); return timeIdxInSeq; }
        // multi-frame slice case: these two get startFrame and numFrames
        size_t StartColumn() const { EnsureNotAllFrames(); return timeIdxInSeq * samplesInRecurrentStep; }
        size_t NumCols() const { EnsureNotAllFrames(); return samplesInRecurrentStep; }
        // TODO: remove these ^^ two in favor of these vv
        size_t StartColumn(const shared_ptr<MBLayout> & pMBLayout) const { EnsureNotAllFrames(); VerifyMBLayout(pMBLayout); return timeIdxInSeq * pMBLayout->GetNumParallelSequences(); }
        size_t NumCols(const shared_ptr<MBLayout> & pMBLayout) const { EnsureNotAllFrames(); VerifyMBLayout(pMBLayout); return pMBLayout->GetNumParallelSequences(); }
        bool IsAllFrames() const { return samplesInRecurrentStep == SIZE_MAX; } // if true then above functions may not be called; caller must use entire batch instead

        const FrameRange & Check(size_t expectedStartColumn, size_t expectedNumCols, const shared_ptr<MBLayout> & pMBLayout) const
        {
            if (!IsAllFrames() && (samplesInRecurrentStep != pMBLayout->GetNumParallelSequences() || expectedStartColumn != StartColumn(pMBLayout) || expectedNumCols != NumCols(pMBLayout)))
                LogicError("FrameRange::Check: FrameRange object gives different range than original explicit code. Logic is borked.");
            return *this;
        }
    private:
        FrameRange(const FrameRange & other);// : timeIdxInSeq(other.timeIdxInSeq), numFrames(other.numFrames) { }
        void operator=(const FrameRange &);
        void EnsureNotAllFrames() const
        {
            if (IsAllFrames())
                LogicError("FrameRange::t() called when frame range refers to whole minibatch");
        }
        // TODO: this will go away once we remove samplesInRecurrentStep from this class
        void VerifyMBLayout(const shared_ptr<MBLayout> & pMBLayout) const
        {
            if (pMBLayout->GetNumParallelSequences() != samplesInRecurrentStep)
                LogicError("VerifyMBLayout: MBLayout inconsistent with local copy of samplesInRecurrentStep");
        }
    };

}}}
