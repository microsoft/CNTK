//
// Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Encapsulates NCCLs dependencies
#pragma once

#pragma warning( push )
#pragma warning ( disable : 4100 ) // Disable warning 4100 (unrefrenced paramter)

#include "Matrix.h"
#include "MPIWrapper.h"

#include <vector>
#include <type_traits>

// Forward declare CUDA stuff
typedef struct CUstream_st* cudaStream_t;
typedef struct ncclComm* ncclComm_t;

namespace Microsoft { namespace MSR { namespace CNTK {

class NcclComm
{
#ifdef USE_NCCL
private:
    enum class DataType : int {FLOAT, DOUBLE};
    void AllReduceImpl(void* inputbuffer, void* outputbuffer, size_t count, DataType dtype);
    void BroadcastImpl(void* buffer, size_t count, MPI_Datatype dtype, int root);
    cudaStream_t m_stream;
    ncclComm_t m_ncclComm;
#endif

public:
    NcclComm(int deviceId, const MPIWrapperPtr& mpiComm);
    ~NcclComm();
    bool IsSupported();
    void Sync(); // waits for outstanding reductions to complete
    
    template <typename ElemType>
    void AllReduce(ElemType* inputBuffer, ElemType* outputBuffer, size_t count)
    {
#ifdef USE_NCCL
        DataType dtype = DataType::FLOAT;
        if (std::is_same<ElemType, double>::value)
            dtype = DataType::DOUBLE;
        else if (!std::is_same<ElemType, float>::value)
            RuntimeError("NcclComm Unsupported reduction type");

        AllReduceImpl(inputBuffer, outputBuffer, count, dtype);
#else
        RuntimeError("NcclComm: CNTK was built without NCCL support.");
#endif
    }

    template <typename ElemType>
    void AllReduce(const std::vector<Matrix<ElemType>*>& grads)
    {
#ifdef USE_NCCL
        DataType dtype = DataType::FLOAT;
        if (std::is_same<ElemType, double>::value)
            dtype = DataType::DOUBLE;
        else if (!std::is_same<ElemType, float>::value)
            RuntimeError("NcclComm Unsupported reduction type");

        for (size_t i=0; i<grads.size(); ++i)
        {
            if (grads[i]->Data() == nullptr) // Hack in case of eval
                continue;
            AllReduceImpl(grads[i]->Data(), grads[i]->Data(), grads[i]->GetNumElements(), dtype);
        }
#else
        RuntimeError("NcclComm: CNTK was built without NCCL support.");
#endif
    }

    void Broadcast(void* buffer, size_t count, MPI_Datatype dtype, int root)
    {
#ifdef USE_NCCL
        BroadcastImpl(buffer, count, dtype, root);
#else
        RuntimeError("NcclComm: CNTK was built without NCCL support.");
#endif
    }
};

#pragma warning( pop )

}}}
