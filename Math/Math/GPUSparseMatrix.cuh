//
// <copyright file="GPUSparseMatrix.cuh" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once
#include "GPUMatrix.cuh"
#include <functional>

namespace Microsoft { namespace MSR { namespace CNTK {    

    //GPU Sparse Matrix, using cuSPARSE library.
    //By default we are assuming CSR representation
    // NOTE m_elemSizeAllocated (in base matrix) means the number of non-zero elements we have allocated space
    // We are packing the CSR format (pointed to by m_pArray) as follows:
    // ElemType elements[m_elemSizeAllocated]
    // int colIdx[m_elemSizeAllocated]
    // int rowIdxStart[m_numRows+1]
    template<class ElemType>
    class MATH_API GPUSparseMatrix : public BaseMatrix<ElemType>
    {
        typedef BaseMatrix<ElemType> B; using B::m_numRows; using B::m_numCols; using B::m_pArray; using B::m_elemSizeAllocated; using B::m_nz; using B::m_format;   // easier access to base members
    private:
        void ZeroInit();
        void Init();
        void ClearNew();
    private:
        void performInplaceFunction(int kind);
        void DeepCopy(const GPUSparseMatrix<ElemType>& deepCopyFrom);
        void Clear();
        void PrepareBuffer(size_t m, size_t n, bool canReuseBuffer, std::function<size_t (int* csrRowPtrC)> func);
        size_t ElemCountFromBufferSize(size_t totalBufferSize);
        void PrepareDevice(short deviceId=-1) const;

    public:
        GPUSparseMatrix(const MatrixFormat format, const int deviceId);
        
        GPUSparseMatrix();
        GPUSparseMatrix(const size_t numRows, const size_t numCols, const size_t nz, ElemType* pArray, const size_t matrixFlags=matrixFormatSparseCSR, int deviceId=MANAGEDEXTERN, const size_t elemSizeAllocated=0);
    
        GPUSparseMatrix(const GPUSparseMatrix<ElemType>&);
        GPUSparseMatrix(const GPUMatrix<ElemType>&);
#ifndef	LINUX
        GPUSparseMatrix(GPUSparseMatrix<ElemType>&&);
#endif	/* LINUX */
        ~GPUSparseMatrix();
    public:
        void Resize(const size_t numRows, const size_t numCols, size_t size = 0);
        void Reset();

    public:
        // return col pointer, which is immediately following the non-zero element
        // in memory format is always in the following order:
        // Non-zero data elements, Full index locations, compressed index locations
        // In CSR row data is compressed, in CSC col data is compressed
        const ElemType* NzLocation() const {return m_pArray;}
        ElemType* NzLocation() {return m_pArray;}
        size_t NzCount() const {return m_nz;}
        size_t NzSize() const {return sizeof(ElemType)*m_nz;} // actual number of element bytes in use
        int* IndexLocation() const {return (int*)(m_pArray+m_elemSizeAllocated);}
        size_t IndexSize() const {return sizeof(int)*m_nz;} // actual number of index bytes in use
        int* CompressedIndexLocation() const {return IndexLocation() + m_elemSizeAllocated;}
        size_t CompressedIndexCount() const 
        {
            if (m_format&matrixFormatCompressed)
            {
                size_t cnt = (m_format&matrixFormatRowMajor)?m_numRows:m_numCols;
                if (cnt) cnt++; // add an extra element on the end for the "max" value
                return cnt;
            }
            return m_nz; // COO format
        }
        // get size for compressed index
        size_t CompressedIndexSize() const {return (CompressedIndexCount())*sizeof(int);}
        size_t BufferSize() const {return NzSize() + IndexSize() + CompressedIndexSize();}
        ElemType* BufferPointer() const;

        // the column and row locations will swap based on what format we are in. Full index always follows the data array
        int* RowLocation() const {return (m_format&matrixFormatRowMajor)?CompressedIndexLocation():IndexLocation();}
        size_t RowSize() const {return (m_format&matrixFormatRowMajor)?CompressedIndexSize():IndexSize();} 
        int* ColLocation() const {return (m_format&matrixFormatRowMajor)?IndexLocation():CompressedIndexLocation();}
        size_t ColSize() const {return (m_format&matrixFormatRowMajor)?IndexSize():CompressedIndexSize();} // actual number of row bytes in use

        void SetValue(const GPUSparseMatrix<ElemType>& deepCopyFrom);
        void SetValue(const GPUMatrix<ElemType>& denseMatrix);
        void ResizeAs(const GPUSparseMatrix<ElemType>& a);
        //void Resize(const size_t numRows, const size_t numCols, const size_t numNZ);

        GPUSparseMatrix<ElemType> Transpose() const;
        void InplaceTranspose();
        GPUSparseMatrix<ElemType>& AssignTransposeOf(const GPUSparseMatrix<ElemType>& a);

        GPUMatrix<ElemType> CopyToDenseMatrix();
        GPUSparseMatrix<ElemType>& operator=(const GPUSparseMatrix<ElemType>& deepCopy);
#ifndef	LINUX
        GPUSparseMatrix<ElemType>& operator=(GPUSparseMatrix<ElemType>&& moveFrom);
#endif	/* LINUX */
        GPUSparseMatrix<ElemType> operator+ (const GPUSparseMatrix<ElemType>& a) const;
        GPUSparseMatrix<ElemType> operator- (const GPUSparseMatrix<ElemType>& a) const;
        GPUSparseMatrix<ElemType>& operator^= (ElemType alpha); //element-wise power        
        GPUSparseMatrix<ElemType> operator^ (ElemType alpha) const; //element-wise power
        GPUSparseMatrix<ElemType>& operator*= (ElemType alpha);
        GPUSparseMatrix<ElemType> operator*(ElemType alpha) const;
        GPUSparseMatrix<ElemType>& AssignElementPowerOf(const GPUSparseMatrix<ElemType>& a, const ElemType power);        

        bool IsEqualTo(const GPUSparseMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;
        bool IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;
    public:
        int GetComputeDeviceId(void) const;
        size_t GetNZElements() const {return m_nz;}
        //Sets sparse matrix in CSR format. this acts as deep copy
        void SetMatrixFromCSRFormat(int *h_CSRRow, int *h_Col, ElemType *h_Val, size_t nz, size_t numRows, size_t numCols, bool IsOnDevice=false, int devId=0);
        void SetMatrixFromCSCFormat(size_t *h_row, size_t *h_rowIdx, size_t size, size_t blockSize);
        void SetMatrixFromLabelAndClass(size_t *h_row, size_t *h_block2Id, size_t *h_block2UniqId, size_t labelSize, size_t expandedSize, size_t blockSize);
        //Gets sparse matrix in CSR format. this acts as deep copy. All passed pointers must be NULL. the function will allocate memory itself.
        void GetMatrixFromCSRFormat(int*& h_CSRRow, int*& h_Col, ElemType*& h_Val, size_t &nz, size_t &numRows, size_t &numCols) const;

    public:       
        GPUSparseMatrix<ElemType>& ElementInverse ();
        GPUSparseMatrix<ElemType>& AssignElementInverseOf (const GPUSparseMatrix<ElemType>& a);

        GPUSparseMatrix<ElemType>& InplaceLinearRectifierDerivative();
        GPUSparseMatrix<ElemType>& AssignLinearRectifierDerivativeOf (const GPUSparseMatrix<ElemType>& a);

        GPUSparseMatrix<ElemType>& InplaceSigmoid ();
        GPUSparseMatrix<ElemType>& AssignSigmoidOf (const GPUSparseMatrix<ElemType>& a);

        GPUSparseMatrix<ElemType>& InplaceTanh ();
        GPUSparseMatrix<ElemType>& AssignTanhOf (const GPUSparseMatrix<ElemType>& a);

        GPUSparseMatrix<ElemType>& InplaceSqrt ();
        GPUSparseMatrix<ElemType>& AssignSqrtOf (const GPUSparseMatrix<ElemType>& a);

        GPUSparseMatrix<ElemType>& InplaceExp ();
        GPUSparseMatrix<ElemType>& AssignExpOf (const GPUSparseMatrix<ElemType>& a);

        GPUSparseMatrix<ElemType>& InplaceLog ();
        GPUSparseMatrix<ElemType>& AssignLogOf (const GPUSparseMatrix<ElemType>& a);

        GPUSparseMatrix<ElemType>& InplaceAbs ();   
        GPUSparseMatrix<ElemType>& AssignAbsOf (const GPUSparseMatrix<ElemType>& a);

        GPUSparseMatrix<ElemType>& InplaceTruncate (const ElemType threshold);

        GPUSparseMatrix<ElemType>& InplaceTruncateBottom (const ElemType threshold);
        GPUSparseMatrix<ElemType>& AssignTruncateBottomOf (const GPUSparseMatrix<ElemType>& a, const ElemType threshold);
        GPUSparseMatrix<ElemType>& InplaceTruncateTop (const ElemType threshold);
        GPUSparseMatrix<ElemType>& AssignTruncateTopOf (const GPUSparseMatrix<ElemType>& a, const ElemType threshold);

        GPUSparseMatrix<ElemType>& SetToZeroIfAbsLessThan (const ElemType threshold);

        ElemType SumOfElements () const; //sum of all elements
        ElemType SumOfAbsElements () const; //sum of all abs(elements)
        ElemType FrobeniusNorm() const;
        ElemType MatrixNormInf() const;
        ElemType MatrixNorm1() const;
        ElemType MatrixNorm0() const { return (ElemType)GetNZElements(); };
    public:        
        //Performs C = alpha ∗ op ( S ) ∗ D + beta ∗ C; Where S is sparse and D and C are dense
        static void MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b, 
            const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c);
        static void MultiplyAndWeightedAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& S, const bool transposeS, const GPUMatrix<ElemType>& D, 
            ElemType beta, GPUMatrix<ElemType>& C); 
        static void MultiplyAndAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA, const GPUSparseMatrix<ElemType>& rhs, 
            const bool transposeB, GPUSparseMatrix<ElemType>& c);
        static void ScaleAndAdd(const ElemType alpha, const GPUSparseMatrix<ElemType>& lhs, GPUMatrix<ElemType>& c);
        
        static void ClassEntropy(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& weight,
            const GPUSparseMatrix<ElemType> & label, const GPUMatrix<ElemType>& cls, 
            const GPUMatrix<ElemType>& idx2cls, GPUSparseMatrix<ElemType>& etp, GPUMatrix<ElemType>& entropyScore);
        static void ClassEntropyError(GPUSparseMatrix<ElemType>& a);
        static void ClassEntropyGradientOfInput(const GPUSparseMatrix<ElemType>& error, const GPUMatrix<ElemType>& weight,  GPUMatrix<ElemType>& grd);
        static void ClassEntropyGradientOfWeight(const GPUSparseMatrix<ElemType>& error,  const GPUMatrix<ElemType>& input, const GPUSparseMatrix<ElemType> & label, const GPUMatrix<ElemType>& cls, 
        const GPUMatrix<ElemType>& idx2cls, GPUSparseMatrix<ElemType>& grd);

        void NormalGrad(GPUMatrix<ElemType>& c, const ElemType momentum);
        
        static void Multiply(const GPUSparseMatrix<ElemType>& S, const GPUMatrix<ElemType>& D, GPUMatrix<ElemType>& C);
        static void Multiply(const GPUMatrix<ElemType>& D, const GPUSparseMatrix<ElemType>& S, GPUMatrix<ElemType>& C);
        static void Multiply(const GPUSparseMatrix<ElemType>& S1, bool transposeS1, const GPUSparseMatrix<ElemType>& S2, bool transposeS2, GPUSparseMatrix<ElemType> &C);
        GPUSparseMatrix<ElemType>& AssignProductOf(const GPUSparseMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b, const bool transposeB);

        static ElemType InnerProductOfMatrices(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
        static ElemType InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b);
        static void ScaleAndAdd(ElemType alpha,const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c);
        static void ScaleAndAdd(ElemType alpha,const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
        static void ScaleAndAdd(ElemType alpha,const GPUMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
        static void Scale(ElemType alpha, GPUSparseMatrix<ElemType>& a);
        static void ElementWisePower (ElemType alpha, const GPUSparseMatrix<ElemType>& a, GPUSparseMatrix<ElemType>& c);
        static bool AreEqual(const GPUSparseMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b, const ElemType threshold = 1e-8);
        static bool AreEqual(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const ElemType threshold = 1e-8);
        static bool AreEqual(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b, const ElemType threshold = 1e-8);

        static void Unrolling (//GPUMatrix<ElemType>& debugArray,
            GPUMatrix<ElemType>& output, const GPUMatrix<ElemType>& a, GPUSparseMatrix<ElemType>& unrollMapping, 
            const int inputWidth, const int inputHeight, const int inputChannelNum,
            const int FltWidth,const int FltHeight, const int FltChannel,
            const int FltStepW,  const int FltStepH);

        //For these two, I should also add a version which would return GPUSparseMatrix, since Dense.*Sparse =Sparse.*Dense=Sparse
        static GPUMatrix<ElemType> ElementProductOf (const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
        static GPUMatrix<ElemType> ElementProductOf (const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b);     
        //GPUSparseMatrix<ElemType>& AssignElementProductOf (const GPUSparseMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b);


    public:
        // See: http://stackoverflow.com/questions/4660123/overloading-friend-operator-for-template-class/4661372#4661372
        template <class ElemTypeDummy>
        friend MATH_API File& operator>>(File& stream, GPUSparseMatrix<ElemTypeDummy>& us);
        template <class ElemTypeDummy>
        friend MATH_API File& operator<<(File& stream, const GPUSparseMatrix<ElemTypeDummy>& us);

        bool m_legacy;
        int m_colIdx; //used to SetValue()
        ElemType *m_val; // values
        size_t *m_row; //row/col ids in CSC/CSR format
        size_t *m_pb; //begin ids of col/row in CSC/CSR format
        size_t *m_rowIdx; //indexer of m_row
        size_t *m_col; //used in COO format

        size_t m_blockSize; //block size        
        ElemType *m_blockVal; //block values
        size_t *m_blockIds; //block ids

        size_t m_expandedSize; // expanded label size
        size_t* m_block2Id; // label block id to first word location
        size_t* m_block2UniqId; // label block id to unique first word location        

    };
}}}    

