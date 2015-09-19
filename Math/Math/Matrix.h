//
// <copyright file="Matrix.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

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

    // there is a version down there of ColumnSlice() that abstracts the number of streams
    // TODO: This may not belong here, but having it in ComputeNode would require syntax changes, while having it as a member here only requires a local find-replace. Let's make it work first, then decide how to refactor.
    // the looping versions of EvaluateThisNode() and ComputeInputPartial() take a frame range, through this structure
    // It can cast from a size_t, i.e. those functions can be called passing a size_t in place of the FrameRange.
    // TODO: GetNumParallelSequences() should be subsumed here & removed from nodes
    // TODO: Where this design currently breaks:
    //  - BatchModeNodes must access GetNumParallelSequences(), yet operate on the whole sequence
    //  - likewise, LSTMNode does its own iteration, hence needs access to GetNumParallelSequences() or NumCols() in the whole-batch iterator
    //  - RecurrentNodes access frames with a time shift, where out-of-bounds ones access a different matrix' values
    //  - RecurrentNodes iterate over individual slices--need a sub-setting constructor from a FrameRange to another?
    //  - RecurrentNodes access boundary info with a similar pattern, but boundary info has a different #streams (namely, 1)
    // TODO: Turns out, a FrameRange is either a whole batch or a single frame.
    struct FrameRange
    {
        const size_t timeIdxInSeq;              // start frame
        const size_t samplesInRecurrentStep;    // number of samples in this step       --BUGBUG: this should be part of MBLayout, not FrameRange
        // can construct from a single size_t -> a single-frame range
        //FrameRange(size_t timeIdxInSeq) : timeIdxInSeq(timeIdxInSeq), samplesInRecurrentStep(0)/*FIX THIS*/{}
        FrameRange(size_t timeIdxInSeq, size_t samplesInRecurrentStep) : timeIdxInSeq(timeIdxInSeq), samplesInRecurrentStep(samplesInRecurrentStep){}
        // or without arguments -> entire minibatch / no frame-range
        FrameRange() : timeIdxInSeq(0), samplesInRecurrentStep(SIZE_MAX) {}
        // code that can only handle single-frame ranges will call t() to get the time index, which will throw if numFrames != 1
        // Some functions need just the time index, e.g. for looking up stuff in m_boundaryInfo. That's where an unscaled index is needed (as opposed to startColumn()).
        size_t t() const { EnsureNotAllFrames(); return timeIdxInSeq; }
        // multi-frame slice case: these two get startFrame and numFrames
        size_t StartColumn() const { EnsureNotAllFrames(); return timeIdxInSeq * samplesInRecurrentStep; }
        size_t NumCols() const { EnsureNotAllFrames(); return samplesInRecurrentStep; }
        bool IsAllFrames() const { return samplesInRecurrentStep == SIZE_MAX; } // if true then above functions may not be called; caller must use entire batch instead
    private:
        FrameRange(const FrameRange & other);// : timeIdxInSeq(other.timeIdxInSeq), numFrames(other.numFrames) { }
        void operator=(const FrameRange &);
        void EnsureNotAllFrames() const
        {
            if (IsAllFrames())
                LogicError("FrameRange::t() called when frame range refers to whole minibatch");
        }
    };

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

        // special convenience function to apply ColumnSlice() to getting a frame range
        // It assumes that columns are frames, and returns a sub-range.
        // TODO: decide whether this belongs here or elsewhere
        Matrix<ElemType> FrameSlice(const FrameRange & frameRange
            // TODO: temporary only until this has been tested to work:
            , size_t expectedStartColumn, size_t expectedNumCols
            ) const
        {
            if (frameRange.IsAllFrames()) return ColumnSlice(0, GetNumCols());  // TODO: can we just return a reference to ourselves? --ownership problem
            // TODO: temporary only until this has been tested to work:
            if (expectedStartColumn != frameRange.StartColumn() || expectedNumCols != frameRange.NumCols())
                LogicError("FrameSlice: FrameRange object gives different range than original explicit code. Logic is borked.");
            return ColumnSlice(frameRange.StartColumn(), frameRange.NumCols());
        }

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
        ElemType RmsProp(Matrix<ElemType>& gradients, ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN, const bool needAveMultiplier);

        // TODO: should Reshape() return a new Matrix object that contains a reference to the original?
        void Reshape(const size_t numRows, const size_t numCols);
        void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, bool growOnly = true);  //by default we only reallocate if need to grow        
        /// similarly to the repmat operation in matlab or octave
        static Matrix<ElemType> RepMat(const Matrix<ElemType>& frmMat, const size_t rows, const size_t cols);
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
    // Currently this is to bind the two somewhat inconsistent boundary flags and packing flags.
    // Once that is unified, we can clean it up further. For now, it's just moving the data members and encapsulating access to them where possible.
    // This should probably also contain m_actualNumParallelSequencesInEachRecIter (which should be node-dependent).
    // TODO: move this to an appropriate place and name it properly
    // NOTE: This class represents an abstraction of an originally distributed/code-duped way of defining and accessing the MB layout.
    //       The code below represents the actual use cases I encountered. Not all are, I believe, needed to be as they are; this class could be simplified/streamlined much further.
    //       Some wackiness below is explained by this.
    // TODO: frame-randoized MBs are now represented as one stream of many frames. This is wrong; they should be one-frame utterances with many streams. Once we fully abstract out Data access, this can be changed easily.
    struct MBLayout
    {   
        MBLayout() : m_sentenceBoundaryFlags(CPUDEVICE) { }
    private:    // one day...
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
        Matrix<float> m_sentenceBoundaryFlags;  // (t,stream)
        // ^^ float -> MinibatchPackingFlags, right? Or unsigned char; or change that to 'char' because Matrix<char> already exists
        // This matrix ^^ is always in CPU memory  --TODO: should rather be a matrix of some int
        /// conditionally point to either a pointer to that provided by network, or point to 
        /// an individual sentence boundary info, which happens if timeStep > 1 is required for PastValue node
        /// a matrix of 1 x n_length
        /// != 0 denotes the case that there exists sentence begin or no_labels case in this frame
        /// == 0 denotes such case is not in this frame
        vector<MinibatchPackingFlags> m_minibatchPackingFlags;
        // ^^ This is some form of aggregate of m_sentenceBoundaryFlags taken over all streams. TODO: find out the exact condition
    public:

        bool Is(size_t t, MinibatchPackingFlags f) const { return m_minibatchPackingFlags[t] & f; }
        bool Is(size_t id, size_t t, MinibatchPackingFlags f) const { return ((MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(id, t)) & f; }

        // get info for one frame; used in DelayedValueNode
        // TODO: clean this up, we can do this more nicely
        pair<Matrix<float>, MinibatchPackingFlags> GetFrame(size_t t) const
        {
            return make_pair(m_sentenceBoundaryFlags.ColumnSlice(t, 1), m_minibatchPackingFlags[t]);
        }

        // set a boundary flag
        // This ORs the flags, i.e. it assumes that the matrix has been cleared before.
        // NOTE: original code that calls this did not OR the matrix, but did OR the vector value. I visually checked that it was cleared before, but might have gotten it wrong.
        void Set(size_t id, size_t t, MinibatchPackingFlags f)
        {
            m_sentenceBoundaryFlags.SetValue(id, t, (float)(((MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(id, t)) | f));
            m_minibatchPackingFlags[t] |= f;
        }
        // same but not ORing  --TODO: is this distinction needed?
        void Reset(size_t id, size_t t, MinibatchPackingFlags f)
        {
            m_sentenceBoundaryFlags.SetValue(id, t, (float)(int)f);
            m_minibatchPackingFlags[t] |= f;
        }
        // needed in DelayedValueNodeBase
        // TODO: this is wicked in that the matrix keeps only the NoLabel flag, while the vector keeps all (just gets ORed into)
        void Mask(size_t id, size_t t, MinibatchPackingFlags f)
        {
            m_sentenceBoundaryFlags.SetValue(id, t, (float)(((MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(id, t)) & f));
            //m_minibatchPackingFlags[t] &= f;
        }

        // for LSTMNode ony, which is deprecated, only to make it compile easily:  also used in FindBestPathWithVariableLength() and FindBestPath() in a strange way
        Matrix<float> & GetM() { return m_sentenceBoundaryFlags; }
        // and for DecimateMinibatchWithSentences() which should be revised
        vector<MinibatchPackingFlags> & GetV() { return m_minibatchPackingFlags; }

        // resize and reset all frames to None (note: this is an invalid state and must be fixed by caller afterwards)
        void Resize(size_t numStreams, size_t numFrames)
        {
            m_sentenceBoundaryFlags.Resize(numStreams, numFrames);
            m_sentenceBoundaryFlags.SetValue((float)((int)MinibatchPackingFlags::None));
            m_minibatchPackingFlags.assign(m_sentenceBoundaryFlags.GetNumCols(), MinibatchPackingFlags::None);
        }

        // test a pre-condition  --TODO: we only resize this thing here, so this should not be necessary in the future
        void validate() const { if (m_minibatchPackingFlags.size() != m_sentenceBoundaryFlags.GetNumCols()) LogicError("MBLayout: GetSize() != GetNumTimeSteps()"); }

        // these accessors were for now just collected from actual usage; need to be cleaned up once this compiles again
        size_t GetNumTimeSteps()  const { validate(); return m_sentenceBoundaryFlags.GetNumCols(); }
        size_t GetNumParallelSequences() const { return IsAllNone() ? 1 : m_sentenceBoundaryFlags.GetNumRows(); }   // 1 stream if no matrix
        size_t GetSize() const { validate(); return m_minibatchPackingFlags.size(); }
        // ^^ TODO: add a check whether Size() == GetNumTimeSteps(); it really should, unless I misunderstood
        bool IsAllNone() const { validate(); return m_minibatchPackingFlags.empty(); }
#if 0   // we have this pattern often:
        // TODO: mbSize and #slices must also move into MBLayout 
        evalnet->SetActualMiniBatchSize(mbSize);
        dataReader->CopyMBLayoutTo(evalnet->GetMBLayoutPtr());
        evalnet->VerifyActualNumParallelSequences(dataReader->GetNumParallelSequences());
#endif
#if 0   // a VERY TELLING piece of code
        // packing flags = frame-wise or over all streams of start and end
        for (size_t nt = 0; nt < nMBSize; nt++)
        {
            for (size_t ns = 0; ns < nSlices; ns++)
            {
                if (newBoundary(ns, nt) == ((int) MinibatchPackingFlags::SequenceStart))
                    pMBLayout->m_minibatchPackingFlags[nt] |= MinibatchPackingFlags::SequenceStart;
                if (newBoundary(ns, nt) == ((int) MinibatchPackingFlags::SequenceEnd))
                    pMBLayout->m_minibatchPackingFlags[nt] |= MinibatchPackingFlags::SequenceEnd;
            }
        }
#endif
    };
    typedef std::shared_ptr<MBLayout> MBLayoutPtr;

}}}
