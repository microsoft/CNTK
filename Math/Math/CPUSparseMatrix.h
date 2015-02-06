//
// <copyright file="GPUMatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <stdio.h>
#include "CPUMatrix.h"
#include <map>
#include <unordered_map>

#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#endif    /* Linux - already defined in CPUMatrix.h */

namespace Microsoft { namespace MSR { namespace CNTK {    

    template<class ElemType>
    class MATH_API CPUSparseMatrix : public BaseMatrix<ElemType>
    {
        typedef BaseMatrix<ElemType> B; using B::m_elemSizeAllocated; using B::m_computeDevice; using B::m_externalBuffer; using B::m_format; using B::m_matrixName;
        using B::m_numCols; using B::m_numRows; using B::m_nz; using B::m_pArray;    // without this, base members would require to use thi-> in GCC

    private:
        void ZeroInit();
        void CheckInit(const MatrixFormat format);

    public:
        CPUSparseMatrix(const MatrixFormat format);
        CPUSparseMatrix(const MatrixFormat format, const size_t numRows, const size_t numCols, const size_t size);
        
        
        ~CPUSparseMatrix();

    public:
        using B::GetNumCols; using B::GetNumRows;

        void SetValue(const size_t row, const size_t col, ElemType val); 
        void SetValue(const CPUSparseMatrix& /*val*/) { NOT_IMPLEMENTED; }

        void ShiftBy(int /*numShift*/) { NOT_IMPLEMENTED; }

        size_t BufferSize() const {return m_elemSizeAllocated*sizeof(ElemType);}
        ElemType* BufferPointer() const;

        void SetGaussianRandomValue(const ElemType /*mean*/, const ElemType /*sigma*/, unsigned long /*seed*/) { NOT_IMPLEMENTED; }
        
        static void ClassEntropy(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& weight,
            const CPUSparseMatrix<ElemType> & label, const CPUMatrix<ElemType>& cls, 
            const CPUMatrix<ElemType>& idx2cls, CPUSparseMatrix<ElemType>& etp, CPUMatrix<ElemType>& entropyScore);

        static void ClassEntropyError(CPUSparseMatrix<ElemType>& a);

        static void ClassEntropyGradientOfInput(
            const CPUSparseMatrix<ElemType>& error, 
            const CPUMatrix<ElemType>& weight,
            CPUMatrix<ElemType>& grd);

        static void ClassEntropyGradientOfWeight(
            const CPUSparseMatrix<ElemType>& error,             
            const CPUMatrix<ElemType>& input,
            const CPUSparseMatrix<ElemType> & label, 
            const CPUMatrix<ElemType>& cls, 
            const CPUMatrix<ElemType>& idx2cls, 
            CPUSparseMatrix<ElemType>& grd);

        static void MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA, 
            const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c);
       
        static void MultiplyAndAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA, 
            const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, CPUSparseMatrix<ElemType>& c);
        
        static void ScaleAndAdd(const ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, CPUMatrix<ElemType>& c);

        static bool AreEqual(const CPUSparseMatrix<ElemType>& a, const CPUSparseMatrix<ElemType>& b, const ElemType threshold = 1e-8);

        /// sum(vec(a).*vec(b))
        static ElemType InnerProductOfMatrices(const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/) { NOT_IMPLEMENTED; }
        
        static void AddScaledDifference(const ElemType /*alpha*/, const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/, CPUMatrix<ElemType>& /*c*/,
            bool /*bDefaultZero*/ ) { NOT_IMPLEMENTED; }
        static void AddScaledDifference(const ElemType /*alpha*/, const CPUMatrix<ElemType>& /*a*/, const CPUSparseMatrix<ElemType>& /*b*/, CPUMatrix<ElemType>& /*c*/,
            bool /*bDefaultZero*/ ) { NOT_IMPLEMENTED; }
        
        int GetComputeDeviceId() const {return -1;}
        
        void Resize(const size_t numRows, const size_t numCols, size_t numNZElemToReserve = 0, const bool growOnly = true, const bool keepExistingValues = true);
        void Reset();

        inline ElemType defaultElem()
        {
            ElemType defaultValue;
            memset(&defaultValue, 0, sizeof(ElemType));
            return defaultValue;
        }

        const ElemType& operator() (const size_t row, const size_t col) const
        {
            if (col >= m_numCols || row >= m_numRows)
            {
                throw std::runtime_error("Position outside matrix dimensions");
            }

            if (m_format == MatrixFormat::matrixFormatSparseCSC)
            {
                size_t start = m_compIndex[col];
                size_t end = m_compIndex[col + 1];
                for (size_t p = start; p < end; p++)
                {
                    size_t i = m_unCompIndex[p];
                    if (i == row)
                    {
                        return m_pArray[p];
                    }
                }

                return m_default;
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

    public:
        void NormalGrad(CPUMatrix<ElemType>& c, const ElemType momentum);
        void Adagrad(CPUMatrix<ElemType>& c);

        public:
        CPUSparseMatrix<ElemType>& InplaceTruncateTop (const ElemType /*threshold*/) { NOT_IMPLEMENTED; }
        CPUSparseMatrix<ElemType>& InplaceTruncateBottom (const ElemType /*threshold*/) { NOT_IMPLEMENTED; }
        CPUSparseMatrix<ElemType>& InplaceTruncate (const ElemType /*threshold*/);

    public:
        void Print(const char* /*matrixName*/) const { NOT_IMPLEMENTED; }

    public:
        const ElemType* NzValues() const { return m_pArray; }
        ElemType* NzValues() { return m_pArray; }
        size_t NzSize() const { return sizeof(ElemType)*m_nz; } // actual number of element bytes in use

        CPUSPARSE_INDEX_TYPE* MajorIndexLocation() const { return m_unCompIndex; } //this is the major index, row/col ids in CSC/CSR format
        size_t MajorIndexCount() const { return m_nz; }
        size_t MajorIndexSize() const { return sizeof(CPUSPARSE_INDEX_TYPE)*MajorIndexCount(); } // actual number of major index bytes in use

        CPUSPARSE_INDEX_TYPE* SecondaryIndexLocation() const { return m_compIndex; } //this is the compressed index, col/row in CSC/CSR format
        size_t SecondaryIndexCount() const
        {
            if (m_format&matrixFormatCompressed)
            {
                size_t cnt = (m_format&matrixFormatRowMajor) ? m_numRows : m_numCols;
                if (cnt > 0) cnt++; // add an extra element on the end for the "max" value
                return cnt;
            }
            else
                return m_nz; // COO format
        }
        // get size for compressed index
        size_t SecondaryIndexSize() const { return (SecondaryIndexCount())*sizeof(CPUSPARSE_INDEX_TYPE); }

        // the column and row locations will swap based on what format we are in. Full index always follows the data array
        CPUSPARSE_INDEX_TYPE* RowLocation() const { return (m_format&matrixFormatRowMajor) ? SecondaryIndexLocation() : MajorIndexLocation(); }
        size_t RowSize() const { return (m_format&matrixFormatRowMajor) ? SecondaryIndexSize() : MajorIndexSize(); }
        CPUSPARSE_INDEX_TYPE* ColLocation() const { return (m_format&matrixFormatRowMajor) ? MajorIndexLocation() : SecondaryIndexLocation(); }
        size_t ColSize() const { return (m_format&matrixFormatRowMajor) ? MajorIndexSize() : SecondaryIndexSize(); } // actual number of bytes in use

    private:
        int m_colIdx; //used to SetValue()
        size_t m_compIndexSize;

        //non-zero values are stored in m_pArray
        CPUSPARSE_INDEX_TYPE *m_unCompIndex; //row/col ids in CSC/CSR format
        CPUSPARSE_INDEX_TYPE *m_compIndex; //begin ids of col/row in CSC/CSR format

        size_t m_blockSize; //block size
        size_t *m_blockIds; //block ids

        ElemType m_default;
    };

    typedef CPUSparseMatrix<float> CPUSingleSparseMatrix;
    typedef CPUSparseMatrix<double> CPUDoubleSparseMatrix;

}}}    

