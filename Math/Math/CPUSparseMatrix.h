//
// <copyright file="GPUMatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <stdio.h>
#include "cpumatrix.h"
#include <map>
#include <unordered_map>

#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif

namespace Microsoft { namespace MSR { namespace CNTK {    

    template<class ElemType>
    class MATH_API CPUSparseMatrix : public BaseMatrix<ElemType>
    {

    private:
        void ZeroInit();

    public:
        CPUSparseMatrix(const MatrixFormat format);
        CPUSparseMatrix(const MatrixFormat format, const size_t numRows, const size_t numCols, const size_t size);

        ~CPUSparseMatrix();

    public:
        void SetValue(const size_t rIdx, const size_t cIdx, ElemType val); 
        void SetValue(const CPUSparseMatrix& /*val*/) { NOT_IMPLEMENTED; }

        void ShiftBy(int /*numShift*/) { NOT_IMPLEMENTED; }

        size_t BufferSize() const {return m_elemSizeAllocated*sizeof(ElemType);}
        ElemType* BufferPointer() const;

        void SetGaussianRandomValue(const ElemType /*mean*/, const ElemType /*sigma*/, unsigned long /*seed*/) { NOT_IMPLEMENTED; }
        
        static void ClassEntropy(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& weight,
            const CPUSparseMatrix<ElemType> & label, const CPUMatrix<ElemType>& cls, 
            const CPUMatrix<ElemType>& idx2cls, CPUSparseMatrix<ElemType>& etp, CPUMatrix<ElemType>& entropyScore);

        static void CPUSparseMatrix<ElemType>::ClassEntropyError(CPUSparseMatrix<ElemType>& a);

        static void CPUSparseMatrix<ElemType>::ClassEntropyGradientOfInput(
            const CPUSparseMatrix<ElemType>& error, 
            const CPUMatrix<ElemType>& weight,
            CPUMatrix<ElemType>& grd);

        static void CPUSparseMatrix<ElemType>::ClassEntropyGradientOfWeight(
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
        static ElemType InnerProductOfMatrices(const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& b) { NOT_IMPLEMENTED; }
        
        static void AddScaledDifference(const ElemType alpha, const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, 
            bool bDefaultZero ) { NOT_IMPLEMENTED; }
        static void AddScaledDifference(const ElemType alpha, const CPUMatrix<ElemType>& a, const CPUSparseMatrix<ElemType>& b, CPUMatrix<ElemType>& c, 
            bool bDefaultZero ) { NOT_IMPLEMENTED; }
        
        int GetComputeDeviceId() const {return -1;}
        
        void Resize(const size_t numRows, const size_t numCols, size_t size = 0);
        void Reset();

    public:
        void NormalGrad(CPUMatrix<ElemType>& c, const ElemType momentum);
        void Adagrad(CPUMatrix<ElemType>& c);
        void RmsProp(CPUMatrix<ElemType>& c);

        public:
        CPUSparseMatrix<ElemType>& InplaceTruncateTop (const ElemType threshold) { NOT_IMPLEMENTED; }
        CPUSparseMatrix<ElemType>& InplaceTruncateBottom (const ElemType threshold) { NOT_IMPLEMENTED; }
        CPUSparseMatrix<ElemType>& InplaceTruncate (const ElemType threshold);

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

