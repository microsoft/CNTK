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
#endif	/* Linux - already defined in CPUMatrix.h */

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

        void SetValue(const size_t rIdx, const size_t cIdx, ElemType val); 
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

        /// sum(vec(a).*vec(b))
        static ElemType InnerProductOfMatrices(const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/) { NOT_IMPLEMENTED; }
        
        static void AddScaledDifference(const ElemType /*alpha*/, const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/, CPUMatrix<ElemType>& /*c*/,
            bool /*bDefaultZero*/ ) { NOT_IMPLEMENTED; }
        static void AddScaledDifference(const ElemType /*alpha*/, const CPUMatrix<ElemType>& /*a*/, const CPUSparseMatrix<ElemType>& /*b*/, CPUMatrix<ElemType>& /*c*/,
            bool /*bDefaultZero*/ ) { NOT_IMPLEMENTED; }
        
        int GetComputeDeviceId() const {return -1;}
        
        void Resize(const size_t numRows, const size_t numCols, size_t size = 0);
        void Reset();

    public:
        void NormalGrad(CPUMatrix<ElemType>& c, const ElemType momentum);
        void Adagrad(CPUMatrix<ElemType>& c);

        public:
        CPUSparseMatrix<ElemType>& InplaceTruncateTop (const ElemType /*threshold*/) { NOT_IMPLEMENTED; }
        CPUSparseMatrix<ElemType>& InplaceTruncateBottom (const ElemType /*threshold*/) { NOT_IMPLEMENTED; }
        CPUSparseMatrix<ElemType>& InplaceTruncate (const ElemType /*threshold*/);

    public:
        void Print(const char* /*matrixName*/) const { NOT_IMPLEMENTED; }

        int m_colIdx; //used to SetValue()
        ElemType *m_val; // values
        size_t *m_row; //row/col ids in CSC/CSR format
        size_t *m_pb; //begin ids of col/row in CSC/CSR format

        size_t m_blockSize; //block size        
        ElemType *m_blockVal; //block values
        size_t *m_blockIds; //block ids
    };

    typedef CPUSparseMatrix<float> CPUSingleSparseMatrix;
    typedef CPUSparseMatrix<double> CPUDoubleSparseMatrix;

}}}    

