#pragma once

#include "MatrixQuantizer.h"
#include "ColumnQuantizer.h"
#include "QuantizedMatrix.h"
#include "GPUMatrix.h"
#include <cuda_runtime_api.h>
#include <cuda.h>  
#include <vector>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {
            
    template<class ElemType>
    class MatrixQuantizerGPU : public MatrixQuantizer<ElemType>
    {
    public:
        MatrixQuantizerGPU(const Matrix<ElemType>& inMatrix, bool forceSync = false);
        ~MatrixQuantizerGPU();

        // Disallow copy construction and assignment
        MatrixQuantizerGPU(const MatrixQuantizerGPU&) = delete;
        MatrixQuantizerGPU& operator=(const MatrixQuantizerGPU&) = delete;

        void QuantizeAsync(QuantizedMatrix<ElemType>& outQMatrix) override;
        void WaitQuantizeAsyncDone() override;
    
        void UnquantizeAsync(QuantizedMatrix<ElemType>& inQMatrix, Matrix<ElemType>& outMatrix, bool add = false) override;
        void WaitUnquantizeAsyncDone() override;            

    private:        
        //helper functions
        QuantizedMatrix<ElemType>& GetTempGPUQuantizedMatrix(size_t nBits);
        
        void FlagQuantizeCompleteEvent(cudaStream_t computestream) const;
        void SyncQuantizeCompleEventAndFetchAndFlagFetchCompleteEvent(char *cpuBuffer, char*gpuBuffer, size_t size) const;
        void SyncFetchCompleteEvent(cudaStream_t computestream)const;
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
        static size_t GetNumDevice();
        static void Sync();
        static void SyncStream(cudaStream_t stream);
        static void SyncEvent(cudaEvent_t ev);
        
    private:
        static  int numDevices;
        
        static std::vector<cudaStream_t> m_fetchStreams;
        static std::vector<cudaStream_t> m_assignStreams;

        mutable cudaEvent_t m_quantizeCompleteEvent;
        mutable cudaEvent_t m_fetchCompleteEvent;
        mutable cudaEvent_t m_assignCompleteEvent;

        bool m_forceSync;
        bool m_quantizeOpIncludedFetch;
        QuantizedMatrix<ElemType>* m_tempGPUQuantizedMatrix; // A temporary intermediate QuantizedMatrix buffer on the GPU               
    };
    
}}}
