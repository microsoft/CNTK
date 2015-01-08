//
// <copyright file="GPUSparseMatrix.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "GPUMatrix.h"
#include "CPUSparseMatrix.h"
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
        typedef BaseMatrix<ElemType> B; using B::m_numRows; using B::m_numCols; using B::m_pArray; using B::m_elemSizeAllocated; using B::m_nz; using B::m_format;   // without this, base members would require to use thi-> in GCC

    public:
        GPUSparseMatrix(const size_t numRows, const size_t numCols, const size_t numNZ, const MatrixFormat matrixFormat = MatrixFormat::matrixFormatSparseCSR, const DEVICEID_TYPE computeDevice = AUTOPLACEMATRIX);

        GPUSparseMatrix(const MatrixFormat matrixFormat = MatrixFormat::matrixFormatSparseCSR,
            const DEVICEID_TYPE computeDevice = AUTOPLACEMATRIX);
    
        GPUSparseMatrix(const GPUSparseMatrix<ElemType>&);

        GPUSparseMatrix(const GPUMatrix<ElemType>&, const MatrixFormat matrixFormat = MatrixFormat::matrixFormatSparseCSR);

#ifndef    LINUX
        GPUSparseMatrix(GPUSparseMatrix<ElemType>&&);
#endif    /* LINUX */

        ~GPUSparseMatrix();

    public:
        void Reset();

    public:
        // return col pointer, which is immediately following the non-zero element
        // in memory format is always in the following order:
        // Non-zero data elements, Full index locations, compressed index locations
        // In CSR row data is compressed, in CSC col data is compressed
        const ElemType* NzValues() const {return m_pArray;}
        ElemType* NzValues() {return m_pArray;}
        size_t NzSize() const {return sizeof(ElemType)*m_nz;} // actual number of element bytes in use

        GPUSPARSE_INDEX_TYPE* MajorIndexLocation() const { return (GPUSPARSE_INDEX_TYPE*)(m_pArray + m_elemSizeAllocated); } //this is the major index, row/col ids in CSC/CSR format
        size_t MajorIndexCount() const { return m_nz; }
        size_t MajorIndexSize() const { return sizeof(GPUSPARSE_INDEX_TYPE)*MajorIndexCount(); } // actual number of major index bytes in use

        GPUSPARSE_INDEX_TYPE* SecondaryIndexLocation() const { return MajorIndexLocation() + m_elemSizeAllocated; } //this is the compressed index, col/row in CSC/CSR format
        size_t SecondaryIndexCount() const 
        {
            if (m_format&matrixFormatCompressed)
            {
                size_t cnt = (m_format&matrixFormatRowMajor)?m_numRows:m_numCols;
                if (cnt > 0) cnt++; // add an extra element on the end for the "max" value
                return cnt;
            }
            else
                return m_nz; // COO format
        }
        // get size for compressed index
        size_t SecondaryIndexSize() const { return (SecondaryIndexCount())*sizeof(GPUSPARSE_INDEX_TYPE); }

        size_t BufferSizeNeeded() const { return NzSize() + MajorIndexSize() + SecondaryIndexSize(); }
        size_t BufferSizeAllocated() const { return m_totalBufferSizeAllocated; }
        ElemType* BufferPointer() const;

        // the column and row locations will swap based on what format we are in. Full index always follows the data array
        GPUSPARSE_INDEX_TYPE* RowLocation() const { return (m_format&matrixFormatRowMajor) ? SecondaryIndexLocation() : MajorIndexLocation(); }
        size_t RowSize() const {return (m_format&matrixFormatRowMajor)?SecondaryIndexSize():MajorIndexSize();} 
        GPUSPARSE_INDEX_TYPE* ColLocation() const { return (m_format&matrixFormatRowMajor) ? MajorIndexLocation() : SecondaryIndexLocation(); }
        size_t ColSize() const {return (m_format&matrixFormatRowMajor)?MajorIndexSize():SecondaryIndexSize();} // actual number of bytes in use

        void SetValue(const GPUSparseMatrix<ElemType>& deepCopyFrom);
        void SetValue(const CPUSparseMatrix<ElemType>& deepCopyFrom);
        void SetValue(const GPUMatrix<ElemType>& denseMatrix, const MatrixFormat matrixFormat);
        void SetValue(const GPUMatrix<ElemType>& denseMatrix);

        void ResizeAsAndCopyIndexFrom(const GPUSparseMatrix<ElemType>& a, const bool growOnly = true);
        void Resize(const size_t numRows, const size_t numCols, const size_t numNZ, const MatrixFormat matrixFormat, const bool growOnly = true); //matrix format will affect the size to allocate
        void Resize(const size_t numRows, const size_t numCols, const size_t numNZ, const bool growOnly = true);

        GPUSparseMatrix<ElemType> Transpose() const;
        void InplaceTranspose();
        GPUSparseMatrix<ElemType>& AssignTransposeOf(const GPUSparseMatrix<ElemType>& a);

        GPUMatrix<ElemType> CopyToDenseMatrix() const;
        void CopyToDenseMatrix(GPUMatrix<ElemType> &denseMatrix) const;
        void CopyToCPUSparseMatrix(CPUSparseMatrix<ElemType> &cpuSparseMatrix) const;
        void ChangeDeviceTo(DEVICEID_TYPE toId);

        GPUSparseMatrix<ElemType>& operator=(const GPUSparseMatrix<ElemType>& deepCopy);
#ifndef    LINUX
        GPUSparseMatrix<ElemType>& operator=(GPUSparseMatrix<ElemType>&& moveFrom);
#endif    /* LINUX */
        GPUSparseMatrix<ElemType> operator+ (const GPUSparseMatrix<ElemType>& a) const;
        GPUSparseMatrix<ElemType> operator- (const GPUSparseMatrix<ElemType>& a) const;
        GPUSparseMatrix<ElemType>& operator^= (const ElemType alpha); //element-wise power        
        GPUSparseMatrix<ElemType> operator^ (const ElemType alpha) const; //element-wise power
        GPUSparseMatrix<ElemType>& operator*= (const ElemType alpha);
        GPUSparseMatrix<ElemType> operator*(const ElemType alpha) const;
        GPUSparseMatrix<ElemType>& AssignElementPowerOf(const GPUSparseMatrix<ElemType>& a, const ElemType power);        

        bool IsEqualTo(const GPUSparseMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;
        bool IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;
    public:
        virtual DEVICEID_TYPE GetComputeDeviceId(void) const;
        size_t GetNumNZElements() const {return m_nz;}

        //Sets sparse matrix in CSR format. this acts as deep copy
        void SetMatrixFromCSRFormat(const GPUSPARSE_INDEX_TYPE *h_CSRRow, const GPUSPARSE_INDEX_TYPE *h_Col, const ElemType *h_Val, 
            const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice = false, const DEVICEID_TYPE devId = -1);
        void SetMatrixFromCSCFormat(const GPUSPARSE_INDEX_TYPE *h_CSCCol, const GPUSPARSE_INDEX_TYPE *h_Row, const ElemType *h_Val,
            const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice = false, const DEVICEID_TYPE devId = -1);
        void SetMatrixFromLabelAndClass(size_t *h_row, size_t *h_block2Id, size_t *h_block2UniqId, size_t labelSize, size_t expandedSize, size_t blockSize);
        //Gets sparse matrix in CSR format. this acts as deep copy. All passed pointers must be NULL. the function will allocate memory itself.
        void GetMatrixFromCSRFormat(GPUSPARSE_INDEX_TYPE*& h_CSRRow, GPUSPARSE_INDEX_TYPE*& h_Col, ElemType*& h_Val, size_t &nz, size_t &numRows, size_t &numCols) const;

        void GetMatrixFromCSCFormat(GPUSPARSE_INDEX_TYPE*& h_CSCCol, GPUSPARSE_INDEX_TYPE*& h_Row, ElemType*& h_Val, size_t &nz, size_t &numRows, size_t &numCols) const;

        void ConvertToSparseFormat(MatrixFormat newFormat);
        void ConvertToSparseFormat(MatrixFormat newFormat, GPUSparseMatrix<ElemType>& outMatrix) const;

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
        ElemType MatrixNorm0() const { return (ElemType)GetNumNZElements(); };
    public:        
        //Performs C = alpha ∗ op ( S ) ∗ D + beta ∗ C; Where S is sparse and D and C are dense
        static void MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b, 
            const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c);
        static void MultiplyAndWeightedAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& S, const bool transposeS, const GPUMatrix<ElemType>& D, 
            const bool transposeD, ElemType beta, GPUMatrix<ElemType>& C);
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

        //For these two, I should also add a version which would return GPUSparseMatrix, since Dense.*Sparse =Sparse.*Dense=Sparse
        static GPUMatrix<ElemType> ElementProductOf (const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
        static GPUMatrix<ElemType> ElementProductOf (const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b);     

    public:
        // See: http://stackoverflow.com/questions/4660123/overloading-friend-operator-for-template-class/4661372#4661372
        template <class ElemTypeDummy>
        friend MATH_API File& operator>>(File& stream, GPUSparseMatrix<ElemTypeDummy>& us);
        template <class ElemTypeDummy>
        friend MATH_API File& operator<<(File& stream, const GPUSparseMatrix<ElemTypeDummy>& us);

     private:
         template <class OutType, class InType>
         static void CopyBuffer(OutType * outBuffer, const InType * inBuffer, const size_t size);
         //caller needs to release the returned pointer
         static GPUSPARSE_INDEX_TYPE * ConvertCPUBuffer(const size_t * inBuffer, const size_t size);
         //caller needs to release the returned pointer
         static size_t * ConvertCPUBuffer(const GPUSPARSE_INDEX_TYPE * inBuffer, const size_t size);

    private:
        void ZeroInit(const MatrixFormat matrixFormat, const DEVICEID_TYPE deviceId);

    private:
        void performInplaceFunction(const int kind);
        void DeepCopy(const GPUSparseMatrix<ElemType>& deepCopyFrom);
        void Clear();
        void PrepareBuffer(const size_t numRows, const size_t numCols, const bool canReuseBuffer, std::function<size_t(int* csrRowPtrC)> func);
        size_t ElemCountFromBufferSize(const size_t totalBufferSize) const;
        size_t ElemCountFromBufferSize() const;
        void PrepareDevice(const DEVICEID_TYPE deviceId = -1) const;

     private:

        size_t m_totalBufferSizeAllocated;

        size_t m_blockSize; //block size        
        ElemType *m_blockVal; //block values
        size_t *m_blockIds; //block ids

        size_t m_expandedSize; // expanded label size
        size_t* m_block2Id; // label block id to first word location
        size_t* m_block2UniqId; // label block id to unique first word location        

    };
}}}    

