//
// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "NcclComm.h"

#ifdef USE_NCCL
#include "GPUMatrix.h"
#include <nccl.h>
#include <cuda_runtime.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// allows to write cudaFunction() || "error"   (CUDA runtime)
static void operator||(cudaError_t rc, const char *msg)
{
    if (rc != cudaSuccess)
        RuntimeError("%s: %s (cuda error %d)", msg, cudaGetErrorString(rc), (int) rc);
}

NcclComm::NcclComm(int deviceId, const MPIWrapperPtr& mpi)
    : m_ncclComm(nullptr), m_stream(nullptr)
{
    if (mpi->IsMultiHost())
        return;

    size_t numRanks = mpi->NumNodesInUse();
    std::vector<int> allDevs(numRanks);
    mpi->Allgather(&deviceId, 1, MPI_INT, allDevs.data(), 1, MPI_INT);

    for (size_t r = 0; r<numRanks; r++)
    {
        if (allDevs[r] == CPUDEVICE)
        {
            fprintf(stderr, "NcclComm: disabled, at least one rank using CPU device\n");
            return;
        }
        for (size_t s = 0; s<r; s++)
            if (allDevs[r] == allDevs[s])
            {
                fprintf(stderr, "NcclComm: disabled, same device used by more than one rank\n");
                return;
            }
    }

    ncclUniqueId ncclId;
    ncclResult_t res;

    res = ncclGetUniqueId(&ncclId);
    if (res != ncclSuccess)
        RuntimeError("NcclComm failed to obtain ncclUniqueId: %s", ncclGetErrorString(res));

    mpi->Bcast(&ncclId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0);

    PrepareDevice(deviceId);
    res = ncclCommInitRank(&m_ncclComm, numRanks, ncclId, mpi->CurrentNodeRank());
    if (res != ncclSuccess)
      RuntimeError("NcclComm failed to initialize ncclComm_t: %s", ncclGetErrorString(res));

    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking)
        || "cudaStreamCreateWithFlags failed";
    fprintf(stderr, "NcclComm: initialized\n");
}

NcclComm::~NcclComm()
{
    if (m_stream != nullptr)
        cudaStreamDestroy(m_stream);
    if (m_ncclComm != nullptr)
        ncclCommDestroy(m_ncclComm);
}

bool NcclComm::IsSupported()
{
    return m_ncclComm != nullptr;
}

void NcclComm::AllReduceImpl(void* buffer, size_t count, DataType dtype)
{
    ncclResult_t res;
    if (dtype == DataType::FLOAT)
    {
        res = ncclAllReduce(buffer, buffer, count, ncclFloat, ncclSum, m_ncclComm, m_stream);
    }
    else
    {
        assert(dtype == DataType::DOUBLE);
        res = ncclAllReduce(buffer, buffer, count, ncclDouble, ncclSum, m_ncclComm, m_stream);
    }

    if (res != ncclSuccess)
        RuntimeError("NcclComm ncclAllReduce failed: %s", ncclGetErrorString(res));
}

void NcclComm::Sync()
{
    cudaStreamSynchronize(m_stream) || "NcclComm: cudaStreamSynchronize failed";
}

}}} // end namespaces

#else // !USE_NCCL
namespace Microsoft { namespace MSR { namespace CNTK {

NcclComm::NcclComm(int /*deviceId*/, const MPIWrapperPtr& /*mpi*/) { }

NcclComm::~NcclComm() { }

bool NcclComm::IsSupported()
{
    return false;
}

void NcclComm::Sync() { }

}}} // end namespaces
#endif
