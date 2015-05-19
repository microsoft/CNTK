//
// <copyright file="Matrix.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "CPUMatrix.h"
#include "CPUSparseMatrix.h"
#include "GPUMatrix.h"
#include "GPUSparseMatrix.h"

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

    //To compy with BLAS libraries matrices are stored in ColMajor. However, by default C/C++/C# use RowMajor
    //convertion is need when passing data between Matrix and C++ matrices
    //For the best performance compile CNTKMath project with NO_SYNC preprocessor directive
    //!!!WARNING!!! This class is NOT THREAD SAFE. Test and add necessary modifications if using in multi-threaded environment    
    template<class ElemType>
    class MATH_API Matrix 
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
        Matrix(const size_t numRows, const size_t numCols, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const MatrixType matrixType = DENSE);
        Matrix(const size_t numRows, const size_t numCols, ElemType *pArray, const size_t matrixFlags=matrixFlagNormal, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, const size_t nnz=0);
        Matrix(const Matrix<ElemType>& deepCopyFrom, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);  //copy constructor, deep copy
        Matrix<ElemType>& operator=(const Matrix<ElemType>& deepCopyFrom);  //assignment operator, deep copy
        Matrix(Matrix<ElemType>&& moveFrom);  //move constructor, shallow copy
        Matrix<ElemType>& operator=(Matrix<ElemType>&& moveFrom);  //move coment operator, shallow copy

        static Matrix<ElemType> Ones(const size_t rows, const size_t cols, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);
        static Matrix<ElemType> Zeros(const size_t rows, const size_t cols, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);
        static Matrix<ElemType> Eye(const size_t rows, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);
        static Matrix<ElemType> RandomUniform(const size_t rows, const size_t cols, const ElemType low, const ElemType high, unsigned long seed=USE_TIME_BASED_SEED, DEVICEID_TYPE deviceId=AUTOPLACEMATRIX);
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
        void TransferFromDeviceToDevice(int id_from, int id_to, bool ismoved=false, bool emptyTransfer=false, bool updatePreferredDevice=true) const; 
        CurrentDataLocation GetCurrentMatrixLocation() const { return m_currentDataLocation; };
        void SwitchToMatrixType(const MatrixType newMatrixType, const MatrixFormat newMatrixFormat, const bool keepValues); //sets matrix type between dense and sparse
        size_t GetNumRows() const;
        size_t GetNumCols() const;
        size_t GetNumElements() const;
        wchar_t* GetMatrixName() const;
        void SetMatrixName(const wchar_t* s);
        bool IsEmpty() const;  
        size_t BufferSize() const;
        ElemType* BufferPointer() const;
        size_t NzCount() const;

        ElemType* CopyToArray() const; //allocated by the callee but need to be deleted by the caller
        size_t CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const;  //allocated by the callee but need to be deleted by the caller

        Matrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;
        Matrix<ElemType>& AssignColumnSlice(const Matrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols);

        void ShiftBy(int numShift) ;

        void NormalGrad(Matrix<ElemType>& gradients, Matrix<ElemType>& functionValues, const ElemType learnRatePerSample, const ElemType momentum);
        ElemType Adagrad(Matrix<ElemType>& gradients, const bool needAveMultiplier);
        ElemType RmsProp(Matrix<ElemType>& gradients, ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN, const bool needAveMultiplier);
       
        void Reshape(const size_t numRows, const size_t numCols);
        void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, bool growOnly = true);  //by default we only reallocate if need to grow        
        size_t GetAllocatedSize() const;
        void Reset(); //reset for sparse matrix

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

        Matrix<ElemType> Transpose();
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

        Matrix<ElemType>&  AssignRepeatOf(const Matrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats);
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
        friend File& operator>>(File& stream, Matrix<ElemType>& M)
        {
            char type;
            stream>>type;
            if (type=='d')
            {
                if (M.GetDeviceId()<0)
                {
                    if (M.m_CPUMatrix==NULL) M.m_CPUMatrix = new CPUMatrix<ElemType>();
                    stream>>(*M.m_CPUMatrix);
                    M.SetDataLocation(CPU, DENSE);
                }
                else
                {
                    if (M.m_GPUMatrix==NULL) M.m_GPUMatrix = new GPUMatrix<ElemType>();
                    stream>>(*M.m_GPUMatrix);  
                    M.SetDataLocation(GPU, DENSE);
                }                
            }
            else if (type=='s')
            {
                if (M.GetDeviceId()<0)
                {
                    NOT_IMPLEMENTED;//You might want to tranfer your matrix to GPU
                }
                else
                {
                    if (M.m_GPUSparseMatrix==NULL) M.m_GPUSparseMatrix = new GPUSparseMatrix<ElemType>();
                    stream>>(*M.m_GPUSparseMatrix); 
                    M.SetDataLocation(GPU, SPARSE);
                }                
            }
            else
                LogicError("wrong matrix type!");
            return stream;

        }
        friend File& operator<<(File& stream, const Matrix<ElemType>& M)
        {
            if (M.GetMatrixType()==MatrixType::DENSE)
            {
                stream<<'d';
                if (M.GetDeviceId()<0)
                {
                    stream<<(*M.m_CPUMatrix);
                }
                else
                {
                    stream<<(*M.m_GPUMatrix);
                }                
            }
            else
            {
                stream<<'s';
                if (M.GetDeviceId()<0)
                {
                    NOT_IMPLEMENTED;
                    //stream<<(*M.m_CPUMatrix);
                }
                else
                {
                    stream<<(*M.m_GPUSparseMatrix);
                }           
            }
            return stream;
        }

    public:

		public:
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

    };

    typedef Matrix<float> SingleMatrix;
    typedef Matrix<double> DoubleMatrix;
}}}
