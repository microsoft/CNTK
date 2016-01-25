#include "stdafx.h"
#include "Matrix.h"
#include "MatrixQuantizerImpl.h"
#include "MatrixQuantizerCPU.h"
#include "MatrixQuantizerGPU.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
/*static*/ MatrixQuantizerImpl<ElemType>* MatrixQuantizerImpl<ElemType>::Create(int deviceId, bool useAsync)
{
    if (deviceId >= 0)
    {
#ifndef CPUONLY
        bool useDedicatedComputeStream = useAsync;
        return new MatrixQuantizerGPU<ElemType>(deviceId, useDedicatedComputeStream);
#else
        useAsync;
        RuntimeError("CreateMatrixQuantizer: attempted to use GPU while compiled without GPU support");
#endif
    }
    else
    {
        return new MatrixQuantizerCPU<ElemType>();
    }
}

template class MatrixQuantizerImpl<float>;
template class MatrixQuantizerImpl<double>;

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

template <typename ElemType>
void MatrixComputeStreamEvent::SynchronizeDataTransferFetchStreamWithEvent()
{
    if (m_deviceId >= 0)
    {
        GPUMatrixComputeStreamEvent* GPUEvent = dynamic_cast<GPUMatrixComputeStreamEvent*>(this);
        GPUEvent->SynchronizeDataTransferFetchStreamWithEvent<ElemType>();
    }
}

MatrixComputeStreamEvent::MatrixComputeStreamEvent(int deviceId)
    : m_deviceId(deviceId)
{
}

// Explicit template instantiations
template MATH_API void MatrixComputeStreamEvent::SynchronizeQuantizationComputeStreamWithEvent<float>();
template MATH_API void MatrixComputeStreamEvent::SynchronizeQuantizationComputeStreamWithEvent<double>();
template MATH_API void MatrixComputeStreamEvent::SynchronizeDataTransferFetchStreamWithEvent<float>();
template MATH_API void MatrixComputeStreamEvent::SynchronizeDataTransferFetchStreamWithEvent<double>();
} } }
