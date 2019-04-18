//
// Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Encapsulates NCCLs dependencies
#pragma once

#pragma warning( push )
#pragma warning ( disable : 4100 ) // Disable warning 4100 (unrefrenced parameter)

#include "Matrix.h"
#include "MPIWrapper.h"

#include <vector>
#include <type_traits>

#define __PROFILE__
#ifdef __PROFILE__
#include <chrono>
#include <ctime>
using namespace std;
static std::chrono::time_point<std::chrono::system_clock> profileStartTime;
static std::chrono::time_point<std::chrono::system_clock> profileEndTime;
static int profileCnt = 0;
static vector<double> profileTimeVec;
#endif

// Forward declare CUDA stuff
typedef struct CUstream_st* cudaStream_t;
typedef struct ncclComm* ncclComm_t;

namespace Microsoft { namespace MSR { namespace CNTK {

class NcclComm
{
#ifdef USE_NCCL
private:
    enum class DataType : int
    {
        FLOAT = 0,
        DOUBLE,
        HALF,
        INT,
        COUNT,
    };
    void AllReduceImpl(void* inputbuffer, void* outputbuffer, size_t count, DataType dtype, MPI_Op op);
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
    void AllReduce(ElemType* inputBuffer, ElemType* outputBuffer, size_t count, MPI_Op op = MPI_SUM)
    {
#ifdef USE_NCCL
        DataType dtype = DataType::FLOAT;
        if (std::is_same<ElemType, float>::value)
            dtype = DataType::FLOAT;
        else if (std::is_same<ElemType, double>::value)
            dtype = DataType::DOUBLE;
        else if (std::is_same<ElemType, half>::value)
            dtype = DataType::HALF;
        else if (std::is_same<ElemType, int>::value)
            dtype = DataType::INT;
        else
            RuntimeError("NcclComm Unsupported reduction type");

        AllReduceImpl(inputBuffer, outputBuffer, count, dtype, op);
#else
        RuntimeError("NcclComm: CNTK was built without NCCL support.");
#endif
    }

    template <typename ElemType>
    void AllReduce(const std::vector<Matrix<ElemType>*>& grads, MPI_Op op = MPI_SUM)
    {
#ifdef USE_NCCL
        DataType dtype = DataType::FLOAT;
        if (std::is_same<ElemType, float>::value)
            dtype = DataType::FLOAT;
        else if (std::is_same<ElemType, double>::value)
            dtype = DataType::DOUBLE;
        else if (std::is_same<ElemType, half>::value)
            dtype = DataType::HALF;
        else if (std::is_same<ElemType, int>::value)
            dtype = DataType::INT;
        else
            RuntimeError("NcclComm Unsupported reduction type");

#ifdef __PROFILE__
        if (profileTimeVec.size() == 0)
        {
            profileTimeVec.resize(grads.size());
            fill(profileTimeVec.begin(), profileTimeVec.end(), 0.0);
        }
#endif
        for (size_t i=0; i<grads.size(); ++i)
        {
            if (grads[i]->Data() == nullptr) // Hack in case of eval
                continue;
#ifdef __PROFILE__
            profileStartTime = std::chrono::system_clock::now();
#endif
            AllReduceImpl(grads[i]->Data(), grads[i]->Data(), grads[i]->GetNumElements(), dtype, op);
#ifdef __PROFILE__
            profileEndTime = std::chrono::system_clock::now();
            profileTimeVec[i] += (std::chrono::duration<double>(profileEndTime - profileStartTime)).count();
#endif
        }

#ifdef __PROFILE__
        ++profileCnt;
        if (profileCnt % 100 == 0)
        {
            for (size_t i = 0; i < grads.size(); ++i)
            {
                if (grads[i]->Data() == nullptr) // Hack in case of eval
                    continue;
                fprintf(stderr, "Aggregate size = %d, aggregate time = %.8gs\n", (int)(grads[i]->GetNumElements()), profileTimeVec[i]);
                profileTimeVec[i] = 0.0;
        }
    }
#endif

#else
        RuntimeError("NcclComm: CNTK was built without NCCL support.");
#endif
    }   

    template <typename ElemType>
    void AllGather(ElemType* inputBuffer, ElemType* outputBuffer, size_t count)
    {
#ifdef USE_NCCL
        DataType dtype = DataType::FLOAT;
        if (std::is_same<ElemType, float>::value)
            dtype = DataType::FLOAT;
        else if (std::is_same<ElemType, double>::value)
            dtype = DataType::DOUBLE;
        else if (std::is_same<ElemType, half>::value)
            dtype = DataType::HALF;
        else if (std::is_same<ElemType, int>::value)
            dtype = DataType::INT;
        else
            RuntimeError("NcclComm Unsupported reduction type");

        AllGatherImpl(inputBuffer, outputBuffer, count);
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

#pragma warning( pop )
};

}}}
