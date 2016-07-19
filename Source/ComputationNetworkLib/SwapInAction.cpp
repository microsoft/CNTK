//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapInAction.h"
#include "SwapOutAction.h"
#include "GPUMatrix.h"

#ifndef CPUONLY
    #include <cuda.h>
#endif



namespace Microsoft { namespace MSR { namespace CNTK {


void PrintPtrAttributes3(float *ptr)
{
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att,ptr);
    cout << "memtype: " << att.memoryType << endl;
    cout << "device: " << att.device << endl;
    cout << "host: " << att.hostPointer << endl;
    cout << "device ptr: " << att.devicePointer << endl;
}

void PrintFreeMemory2()
{
    size_t free, total;
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    cout << "free memory: " << free << endl;
}


SwapInAction::SwapInAction(SwapOutAction *swpout, Matrix<float> *GPUBuffer)
{
    m_bufferCPU = swpout->GetCPUMatrix();
    m_bufferGPU = GPUBuffer;
    m_swpout = swpout;
    m_isAsynchronous = true;

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    m_swapInStream = stream;
    m_isSwappingToGPU = false;

    m_rows = m_bufferGPU->GetNumRows();
    m_cols = m_bufferGPU->GetNumCols();
    m_bytes = m_rows*m_cols*sizeof(float);
    m_timer = CUDATimer();
    m_syncCounter = 0;
    //cout << m_rows << "x" << m_cols << endl;
}

 

void SwapInAction::BeginAction()
{
    //cout << m_rows << "x" << m_cols << " vs. ";
    //
    //if(m_rows != m_bufferGPU->GetNumRows() || m_cols != m_bufferGPU->GetNumCols())
    //{
    //    cout << m_rows << "x" << m_cols << " vs. ";
    //    cout << m_bufferGPU->GetNumRows() << "x" << m_bufferGPU->GetNumCols() << endl;
    //}
    //m_timer.tick("resize");


    if(m_rows != m_bufferGPU->GetNumRows() ||
       m_cols != m_bufferGPU->GetNumCols())
       {
    //cout << "VIOLATION: " << m_rows << "x" << m_cols <<  " vs. " << m_bufferGPU->GetNumRows() << "x" << m_bufferGPU->GetNumCols() << endl;
    
    //cout << "VIOLATION: " << m_swpout->GetRows() << "x" << m_swpout->GetCols() <<  " vs. " << m_bufferGPU->GetNumRows() << "x" << m_bufferGPU->GetNumCols() << endl;
    }
    //cout << "VIOLATION: " << m_rows << "x" << m_cols <<  " vs. " << m_bufferGPU->GetNumRows() << "x" << m_bufferGPU->GetNumCols() << endl;
    m_bufferGPU->Resize(m_swpout->GetRows(),m_swpout->GetCols());

    //float *gpupointer = m_bufferGPU->Data();
    //cout << "GPU pointer: " << gpupointer << endl;
    //cout << "alloc" << endl;
    //PrintPtrAttributes3(gpupointer);
    //cout << "-------------------" << endl;
    //cout << m_swpout->GetBytes() << endl;
    //PrintFreeMemory2();
    //CUDA_CALL(cudaSetDevice(0));
    //CUDA_CALL(cudaMalloc((void**)&gpupointer, m_swpout->GetBytes()));
    //PrintPtrAttributes3(gpupointer);
    //cout << "GPU pointer2: " << gpupointer << endl;

    //m_timer.tick("resize");

    if(m_isSwappingToGPU)
    {
        // swap in already active; this happens if we need this buffer in two timesteps in a row
        // all fine!
        //cout << "Warning: Overlapping swap-ins detected!" << endl;
        return;
        //EndAction();
    }
    
    //cout << m_swpout->GetBytes() << " vs. " << m_bytes << endl;
    //PrintPtrAttributes3(m_bufferCPU);
    //PrintPtrAttributes3(m_bufferGPU->Data());
    //gpupointer = m_bufferGPU->Data();
    //CUDA_CALL(cudaMalloc((void**)&gpupointer, m_swpout->GetBytes()));
    //cout << "-------------------" << endl;
    CUDA_CALL(cudaMemcpyAsync(m_bufferGPU->Data(), m_bufferCPU, m_bytes, cudaMemcpyDefault, m_swapInStream));
    m_isSwappingToGPU = true;
}


void SwapInAction::EndAction()
{
    //m_timer.tick();
    CUDA_CALL(cudaStreamSynchronize(m_swapInStream));
    //m_timer.tick();
    m_isSwappingToGPU = false;
    //m_syncCounter++;
    if(m_syncCounter > 1000)
    {
        cout << m_timer.tock()/1000.0f << endl;
        cout << "resize: " << m_timer.tock("resize")/1000.0f << endl;
        m_syncCounter = 0;
    }
}


}}}
