#pragma once

#include "MatrixQuantizer.h"
#include "ColumnQuantizer.h"
#include "QuantizedMatrix.h"
#include "GPUMatrix.h"
#ifndef CPUONLY
#include <cuda_runtime_api.h>
#include <cuda.h>  
#endif // !CPUONLY
#include <vector>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {
            
    template<class ElemType>
    class MatrixQuantizerGPU : public MatrixQuantizer<ElemType>
    {
    public:
        MatrixQuantizerGPU(const Matrix<ElemType>& inMatrix, bool forceSync = false);
        ~MatrixQuantizerGPU();

        // Disallow copy and move construction and assignment
        MatrixQuantizerGPU(const MatrixQuantizerGPU&) = delete;
        MatrixQuantizerGPU& operator=(const MatrixQuantizerGPU&) = delete;
        MatrixQuantizerGPU(MatrixQuantizerGPU&&) = delete;
        MatrixQuantizerGPU& operator=(MatrixQuantizerGPU&&) = delete;

        void QuantizeAsync(QuantizedMatrix<ElemType>& outQMatrix, bool zeroThresholdFor1Bit) override;
        void WaitQuantizeAsyncDone() override;
    
        void UnquantizeAsync(QuantizedMatrix<ElemType>& inQMatrix, Matrix<ElemType>& outMatrix, bool add = false) override;
        void WaitUnquantizeAsyncDone() override;            

    private:        
        // Helper function to get a temporary intermediate matrix on the GPU to store quantization results
        QuantizedMatrix<ElemType>& GetTempGPUQuantizedMatrix(size_t nBits, bool& newlyAllocated);
        
#ifndef CPUONLY
        // Record a event to flag the completion of quantization/unquantization kernel on the compute stream
        void RecordQuantizeCompleteEvent(cudaStream_t computestream) const;

        // Synchronize the fetch stream to the quantization completion event and record an event on the fetch
        // stream to flag the completion of fetching the quantization results from the GPU
        void SyncQuantizeCompleEventAndFetchAndRecordFetchCompleteEvent(char *cpuBuffer, char*gpuBuffer, size_t size) const;

        // Synchronize the compute stream to the assign completion event to ensure that subsequent compute stream operations
        // wait for the assign stream operations, scheduled so far, to finish
        void SyncAssignCompleteEvent(cudaStream_t computestream)const;

        //for concurrent computation and memcpy
        //  - assign to GPU : CPU-to-GPU,started by CPU when data read; flags assigncomplete
        //  - GPU-side operation        --waits for assigncomplete; flags quantizecomplete
        //  - fetch from GPU            --waits for quantizecomplete; flags fetchcomplete
        //  - CPU-side access of buffer --read: waits for fetchcomplete, write: waits for assigncomplete
        
        cudaStream_t GetComputeStream() const;         // get the priority compute stream
        cudaStream_t GetFetchStream()  const;          // and the copy streams
        cudaStream_t GetAssignStream() const;

    private:
        //helper functions for gpus
        static void Sync();
        static void SyncStream(cudaStream_t stream);
        static void SyncEvent(cudaEvent_t ev);
        
    private:
        static cudaStream_t m_fetchStream;
        static cudaStream_t m_assignStream;

        mutable cudaEvent_t m_quantizeCompleteEvent;
        mutable cudaEvent_t m_fetchCompleteEvent;
        mutable cudaEvent_t m_assignCompleteEvent;
#endif // !CPUONLY

    private:
        bool m_forceSync;
        bool m_quantizeOpIncludedFetch;

        // A temporary intermediate QuantizedMatrix buffer on the GPU
        QuantizedMatrix<ElemType>* m_tempGPUQuantizedMatrix; 
    };
    
}}}
