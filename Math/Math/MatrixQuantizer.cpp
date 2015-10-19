#include "stdafx.h"
#include "Matrix.h"
#include "MatrixQuantizer.h"
#include "MatrixQuantizerCPU.h"
#include "BestGpu.h"    // for CPUONLY
#ifndef CPUONLY
#include "MatrixQuantizerGPU.h"
#endif

namespace Microsoft { namespace MSR { namespace CNTK {
    
    template<class ElemType>
    /*static*/ MatrixQuantizer<ElemType>*
    MatrixQuantizer<ElemType>::CreateMatrixQuantizer(size_t numRows, size_t numCols, int deviceId)
    {
        if (deviceId >= 0)
        {
#ifndef CPUONLY
            return new MatrixQuantizerGPU<ElemType>(numRows, numCols, deviceId);
#else
            RuntimeError("CreateMatrixQuantizer: attempted to use GPU while compiled without GPU support");
#endif
        }
        else
        {
            return new MatrixQuantizerCPU<ElemType>(numRows, numCols);
        }
    }

    template<class ElemType>
    MatrixQuantizer<ElemType>::MatrixQuantizer(size_t numRows, size_t numCols, int deviceId)
    {
        m_residual = new Matrix<ElemType>(numRows, numCols, deviceId, DENSE);
    }

    template<class ElemType>
    MatrixQuantizer<ElemType>::~MatrixQuantizer()
    {
        if (nullptr != m_residual)
        {
            delete m_residual;
            m_residual = nullptr;
        }    
    }

    template<class ElemType>
    void MatrixQuantizer<ElemType>::ResetResidue()
    {
        m_residual->SetValue(0.0);
    }


    template class MatrixQuantizer<float>;
    template class MatrixQuantizer<double>;
    
}}}
