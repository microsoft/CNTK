//
// Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "NcclComm.h"

#ifdef USE_NCCL
#include "GPUMatrix.h"
#include <nccl.h>
#include <nvml.h>
#include <cuda_runtime.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// allows to write cudaFunction() || "error"   (CUDA runtime)
static void operator||(cudaError_t rc, const char *msg)
{
    if (rc != cudaSuccess)
        RuntimeError("%s: %s (cuda error %d)", msg, cudaGetErrorString(rc), (int) rc);
}

ncclRedOp_t ncclRedOpFromMpiOp(MPI_Op op)
{
    if (op == MPI_SUM) return ncclSum;
    else if (op == MPI_MAX) return ncclMax;
    else if (op == MPI_MIN) return ncclMin;
    else if (op == MPI_PROD) return ncclProd;
    else RuntimeError("Invalid MPI_Op");
}

NcclComm::NcclComm(int deviceId, const MPIWrapperPtr& mpi)
    : m_ncclComm(nullptr), m_stream(nullptr)
{
    cudaDeviceSynchronize();
    size_t numRanks = mpi->NumNodesInUse();

    auto nvmlRes = nvmlInit();
    std::array<char, NVML_DEVICE_UUID_BUFFER_SIZE> thisDeviceUUID{ 'C', 'P', 'U', 0 };

    if (nvmlRes != NVML_SUCCESS)
    {
        fprintf(stderr, "NcclComm: disabled, failed to initialize NVML library, error code %s\n", nvmlErrorString(nvmlRes));
    }
    else
    {
        if (deviceId != CPUDEVICE)
        {
            nvmlDevice_t thisDevice;
            nvmlRes = nvmlDeviceGetHandleByIndex(deviceId, &thisDevice);
            if (nvmlRes != NVML_SUCCESS)
            {
                fprintf(stderr, "NcclComm: disabled, failed to obtain nvmlDevice handle: %s\n", nvmlErrorString(nvmlRes));
            }
            else
            {
                nvmlRes = nvmlDeviceGetUUID(thisDevice, thisDeviceUUID.data(), thisDeviceUUID.size());
                if (nvmlRes != NVML_SUCCESS)
                {
                    fprintf(stderr, "NcclComm: disabled, failed to obtain nvmlDevice UUID: %s\n", nvmlErrorString(nvmlRes));
                }
            }
        }
        else
        {
            fprintf(stderr, "NcclComm: disabled, at least one rank using CPU device\n");
        }
    }
    std::vector<std::array<char, NVML_DEVICE_UUID_BUFFER_SIZE>> allDeviceUUIDs(numRanks);
    mpi->Allgather(thisDeviceUUID.data(), NVML_DEVICE_UUID_BUFFER_SIZE, MPI_CHAR, allDeviceUUIDs[0].data(), NVML_DEVICE_UUID_BUFFER_SIZE, MPI_CHAR);

    std::array<char, NVML_DEVICE_UUID_BUFFER_SIZE> defaultDeviceUUID{ 'C', 'P', 'U', 0 };
    for (auto deviceUUID : allDeviceUUIDs)
    {
        if (deviceUUID == defaultDeviceUUID) {
            return;
        }
    }

    for (size_t r = 0; r < numRanks; r++)
    {
        for (size_t s = 0; s < r; s++)
        {
            if (strcmp(allDeviceUUIDs[r].data(), allDeviceUUIDs[s].data()) == 0)
            {
                fprintf(stderr, "NcclComm: disabled, same device %s used by more than one rank\n", allDeviceUUIDs[0].data());
                nvmlShutdown();
                return;
            }
        }
    }
    nvmlShutdown();

    ncclUniqueId ncclId = {};
    ncclResult_t res;

    if (mpi->IsMainNode())
    {
        ncclGetUniqueId(&ncclId);
    }

    mpi->Bcast(&ncclId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0);

    static const ncclUniqueId emptyNcclId = {};
    if (memcmp(&ncclId, &emptyNcclId, sizeof(ncclId)) == 0)
    {
        fprintf(stderr, "NcclComm failed to obtain ncclUniqueId: %s\n", ncclGetErrorString(res));
        return;
    }

    PrepareDevice(deviceId);
    res = ncclCommInitRank(&m_ncclComm, numRanks, ncclId, mpi->CurrentNodeRank());
    if (res != ncclSuccess)
    {
        fprintf(stderr, "NcclComm failed to initialize: %s. Set the ENV \"NCCL_DEBUG=INFO\" for more information.\n", ncclGetErrorString(res));
        if (m_ncclComm != nullptr)
            ncclCommDestroy(m_ncclComm);
        return;
    }

    cudaStreamCreateWithFlags(&m_stream, cudaStreamDefault)
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

void NcclComm::AllReduceImpl(void* inputbuffer, void *outputbuffer, size_t count, DataType dtype, MPI_Op op)
{
    ncclResult_t res;
    class NcclTypeLookup
    {
        ncclDataType_t ncclTypes[(int)DataType::COUNT];
    public:
        NcclTypeLookup()
        {
            ncclTypes[(int)DataType::FLOAT]  = ncclFloat;
            ncclTypes[(int)DataType::DOUBLE] = ncclDouble;
            ncclTypes[(int)DataType::HALF] = ncclHalf;
            ncclTypes[(int)DataType::INT]    = ncclInt;
        }
        ncclDataType_t Lookup(DataType dtype)
        {
            return ncclTypes[(int)dtype];
        }
    };

    static NcclTypeLookup s_ncclTypeLookup;

    res = ncclAllReduce(inputbuffer, outputbuffer, count, s_ncclTypeLookup.Lookup(dtype), ncclRedOpFromMpiOp(op), m_ncclComm, m_stream);

    if (res != ncclSuccess)
        RuntimeError("NcclComm ncclAllReduce failed: %s", ncclGetErrorString(res));
}

void NcclComm::BroadcastImpl(void* buffer, size_t count, MPI_Datatype dtype, int root)
{
    ncclResult_t res;
    if (dtype == MPI_CHAR)
    {
        res = ncclBcast(buffer, count, ncclChar, root, m_ncclComm, m_stream);
    }
    else
    {
        RuntimeError("NcclComm Broadcast supports Char type only");
    }
    if (res != ncclSuccess)
    {
        RuntimeError("NcclComm ncclBcast failed: %s", ncclGetErrorString(res));
    }
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
