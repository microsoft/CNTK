#pragma once

#include "ColumnQuantizer.h"
#include "QuantizedMatrix.h"

#ifdef    _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else    // no DLLs on Linux
#define    MATH_API 
#endif

namespace Microsoft { namespace MSR { namespace CNTK {
    
    // This type does the quantization on a matrix
    // This is a technique to reduce the cost of communicating 
    // the gradient matrices during aggregation across all nodes in
    // data-parallel SGD training, at the end of each minibatch.
    // Refer this paper http://research.microsoft.com/apps/pubs/?id=230137
    // for details.
    template<class ElemType>
    class MATH_API MatrixQuantizer
    {
    public:
        static MatrixQuantizer<ElemType>* CreateMatrixQuantizer(size_t numRows, size_t numCols, int deviceId, bool useAsync);
        virtual ~MatrixQuantizer();

        // Disallow copy and move construction and assignment
        MatrixQuantizer(const MatrixQuantizer&) = delete;
        MatrixQuantizer& operator=(const MatrixQuantizer&) = delete;
        MatrixQuantizer(MatrixQuantizer&&) = delete;
        MatrixQuantizer& operator=(MatrixQuantizer&&) = delete;

        //change the state internal residual + qmatrix
        virtual void QuantizeAsync(const Matrix<ElemType>& inMatrix, QuantizedMatrix<ElemType>& outQMatrix, bool zeroThresholdFor1Bit) = 0;
        virtual void WaitQuantizeAsyncDone() = 0;

        //unquantize the matrix, not change the state of any internal data structure
        virtual void UnquantizeAsync(QuantizedMatrix<ElemType>& inQMatrix, Matrix<ElemType>& outMatrix, bool add = false) = 0;
        virtual void WaitUnquantizeAsyncDone() = 0;

        int GetDeviceId() const 
        {
            return m_residual->GetDeviceId();
        }
        
        void ResetResidue();

        const Matrix<ElemType>& GetResidualMatrix() const
        {
            return *m_residual;
        }
        
    protected:
        MatrixQuantizer(size_t numRows, size_t numCols, int deviceId);
        
    protected:

        // the residual matrix 
        Matrix<ElemType>* m_residual;
    };

    // This type records and synchronizes events on the main 
    // matrix computation work stream
    class MATH_API MatrixComputeStreamEvent
    {
    public:
        static MatrixComputeStreamEvent* Create(int deviceId);
        virtual ~MatrixComputeStreamEvent();

        virtual void SynchronizeEvent();

        template <typename ElemType>
        void SynchronizeQuantizationComputeStreamWithEvent();

    protected:
        MatrixComputeStreamEvent(int deviceId);

    protected:
        int m_deviceId;
    };

}}}
