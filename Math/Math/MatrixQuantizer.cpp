#include "stdafx.h"
#include "Matrix.h"
#include "MatrixQuantizer.h"
#include "MatrixQuantizerCPU.h"
#include "BestGpu.h"    // for CPUONLY
#include "MatrixQuantizerGPU.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    
    template<class ElemType>
    /*static*/ MatrixQuantizer<ElemType>*
    MatrixQuantizer<ElemType>::CreateMatrixQuantizer(size_t numRows, size_t numCols, int deviceId, bool useAsync)
    {
        if (deviceId >= 0)
        {
#ifndef CPUONLY
            bool useDedicatedComputeStream = useAsync;
            return new MatrixQuantizerGPU<ElemType>(numRows, numCols, deviceId, useDedicatedComputeStream);
#else
            UNREFERENCED_PARAMETER(useAsync);
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
    

    MatrixComputeStreamEvent* MatrixComputeStreamEvent::Create(int deviceId)
    {
        if (deviceId >= 0)
            return new GPUMatrixComputeStreamEvent(deviceId);
        else
            return new MatrixComputeStreamEvent(deviceId);
    }

    MatrixComputeStreamEvent::~MatrixComputeStreamEvent() 
    {
    }

    void MatrixComputeStreamEvent::SynchronizeEvent()
    {
    }

    template <typename ElemType>
    void MatrixComputeStreamEvent::SynchronizeQuantizationComputeStreamWithEvent()
    {
        if (m_deviceId >= 0)
        {
            GPUMatrixComputeStreamEvent* GPUEvent = dynamic_cast<GPUMatrixComputeStreamEvent*>(this);
            GPUEvent->SynchronizeQuantizationComputeStreamWithEvent<ElemType>();
        }
    }

    MatrixComputeStreamEvent::MatrixComputeStreamEvent(int deviceId) 
        : m_deviceId(deviceId)
    {
    }

    // Explicit template instantiations
    template MATH_API void MatrixComputeStreamEvent::SynchronizeQuantizationComputeStreamWithEvent<float>();
    template MATH_API void MatrixComputeStreamEvent::SynchronizeQuantizationComputeStreamWithEvent<double>();
}}}
