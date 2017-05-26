//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#pragma once

#if HAS_MPI
// Please see https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-Windows#ms-mpi or
// https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-Linux#open-mpi for setup instructions
// of an MPI implementation on your platform.

#ifdef _MSC_VER
// Suppress warning for non-ASCII characters in MS-MPI headers
#pragma warning(push)
#pragma warning(disable : 4819) // The file contains a character that cannot be represented in the current code page (...). Save the file in Unicode format to prevent data loss
#include "mpi.h"
#pragma warning(pop)
#else
#include "mpi.h"
#endif
#else
// Note: the following macros/typedefs define some of the MPI related functions and constants such that code
//       using these functionality will compile cleanly - but will not actually perform the MPI operation.
//       The clean way to go is to move any code related to mpi into the mpiwrapper class implementation and decide
//       in this class if to use mpi.h or not.
typedef void *MPI_Comm;
typedef enum _MPI_Datatype { MPI_CHAR, MPI_INT, MPI_FLOAT, MPI_DOUBLE, MPI_UNSIGNED, MPI_LONG_LONG_INT } MPI_Datatype;

#define MPI_IN_PLACE          ((void*)(int)-1)
#define MPI_SUM               ((MPI_Op)0x58000003)

#define MPI_STATUSES_IGNORE  (MPI_Status*)1
#define MPI_STATUS_IGNORE    (MPI_Status*)1
#define MPI_UNDEFINED        (-32766)

typedef int MPI_Op;
typedef int MPI_Request;
typedef void *MPI_Status;
#endif

#include <errno.h> 
#include <string>
#include <array>
#include <vector>
#include <memory>

#include "CommonMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct MpiFail : public std::string
{
    MpiFail(const std::string &what)
        : std::string(what)
    {
    }
};

extern int operator||(int rc, const MpiFail &what);

class MPIWrapper;
typedef std::shared_ptr<MPIWrapper> MPIWrapperPtr;

extern "C" void GetMpiWrapper(MPIWrapper **mpi);

// Note: This is now a pure interface, so please don't add
//       any functionality to this class.
//       Instead, make your own implementation class, add/change
//       functions there as needed and use a private interface to
//       these functions.
//       In case you need to add functions that affect all
//       implementations, add a pure virtual function here and
//       update any affected implementation.
class MPIWrapper : public std::enable_shared_from_this<MPIWrapper>
{
public:
    MPIWrapper() {}
    virtual ~MPIWrapper() {}

    static MPIWrapperPtr GetInstance(bool create = false);
    static void DeleteInstance();
    static MPIWrapperPtr s_mpi;

    // Note that specifically, this function is such that it does not require
    // MPI initialization. Moreover, it can be used without actually loading any
    // MPI libs.
    // TODO: Once we move to dynamic loading for MPI libs on Linux, move it to utilities.
    static int GetTotalNumberOfMPINodes();

    virtual size_t NumNodesInUse() const = 0;
    virtual size_t CurrentNodeRank() const = 0;
    virtual bool IsMainNode() const = 0;
    virtual std::wstring CurrentNodeName() const = 0;
    virtual bool IsIdle() const = 0;
    virtual bool UsingAllNodes() const = 0;
    virtual size_t MainNodeRank() const = 0;
    virtual bool IsMultiHost() const = 0;

    // Use GPUDirect RDMA support
    virtual bool UseGpuGdr() = 0;

    // -----------------------------------------------------------------------
    // data-exchange functions (wrappers around MPI functions)
    // -----------------------------------------------------------------------

    virtual int Finalize(void) = 0;
    virtual int Wait(MPI_Request* request, MPI_Status* status) = 0;
    virtual int Waitany(int count, MPI_Request array_of_requests[], int* index, MPI_Status* status) = 0;
    virtual int Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) = 0;
    virtual int Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, /*MPI_Comm comm,*/ MPI_Request* request) = 0;
    virtual int Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, /*MPI_Comm comm,*/ MPI_Status* status) = 0;
    virtual int Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, /*MPI_Comm comm,*/ MPI_Request* request) = 0;
    virtual int Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, /*MPI_Comm comm,*/ MPI_Request* request) = 0;
    virtual int Abort(int errorcode) = 0;
    virtual int Error_string(int errorcode, char* string, int* resultlen) = 0;

    // helpers to determine the MPI_Datatype of a pointer
    static MPI_Datatype GetDataType(char *);
    static MPI_Datatype GetDataType(int *);
    static MPI_Datatype GetDataType(float *);
    static MPI_Datatype GetDataType(double *);
    static MPI_Datatype GetDataType(size_t *);

    // allreduce of a vector
    virtual void AllReduce(std::vector<size_t>& accumulator) const = 0;
    virtual void AllReduce(std::vector<int>& accumulator) const = 0;
    virtual void AllReduce(std::vector<double>& accumulator) const = 0;
    virtual void AllReduce(std::vector<float>& accumulator) const = 0;

    // for raw pointer
    virtual void AllReduce(size_t* sendData, size_t numElements, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduce(int* sendData, size_t numElements, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduce(double* sendData, size_t numElements, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduce(float* sendData, size_t numElements, MPI_Op op = MPI_SUM) const = 0;

    virtual void AllReduce(size_t* sendData, size_t* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduce(int* sendData, int* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduce(double* sendData, double* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduce(float* sendData, float* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const = 0;

    virtual void AllReduceAsync(size_t* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduceAsync(int* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduceAsync(double* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduceAsync(float* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const = 0;

    virtual void AllReduceAsync(size_t* sendData, size_t* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduceAsync(int* sendData, int* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduceAsync(double* sendData, double* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const = 0;
    virtual void AllReduceAsync(float* sendData, float* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const = 0;

    virtual void Bcast(size_t* sendData, size_t numElements, size_t srcRank) = 0;
    virtual void Bcast(double* sendData, size_t numElements, size_t srcRank) = 0;
    virtual void Bcast(float* sendData, size_t numElements, size_t srcRank) = 0;
    virtual void Bcast(void* buffer, int count, MPI_Datatype datatype, int root) = 0;

    virtual void AllGatherAsync(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, MPI_Request* request) const = 0;
    virtual void AllGatherAsync(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, MPI_Request* request) const = 0;
    virtual void AllGatherAsync(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, MPI_Request* request) const = 0;
    virtual void AllGatherAsync(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, MPI_Request* request) const = 0;

    virtual void AllGather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements) const = 0;
    virtual void AllGather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements) const = 0;
    virtual void AllGather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements) const = 0;
    virtual void AllGather(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements) const = 0;
    virtual void Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype) const = 0;

    virtual void Gather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, size_t rootRank) const = 0;
    virtual void Gather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, size_t rootRank) const = 0;
    virtual void Gather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, size_t rootRank) const = 0;
    virtual void Gather(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, size_t rootRank) const = 0;

    virtual void Gatherv(const size_t *sendData, size_t numSendElements, size_t *receiveData, int recvCounts[], int offsets[], size_t rootRank) const = 0;
    virtual void Gatherv(const char *sendData, size_t numSendElements, char *receiveData, int recvCounts[], int offsets[], size_t rootRank) const = 0;
    virtual void Gatherv(const int *sendData, size_t numSendElements, int *receiveData, int recvCounts[], int offsets[], size_t rootRank) const = 0;
    virtual void Gatherv(const float *sendData, size_t numSendElements, float *receiveData, int recvCounts[], int offsets[], size_t rootRank) const = 0;
    virtual void Gatherv(const double *sendData, size_t numSendElements, double *receiveData, int recvCounts[], int offsets[], size_t rootRank) const = 0;

    // wait for all ranks to reach here
    virtual int WaitAll() = 0;
    virtual void WaitAny(MPI_Request* requests, int numRequests, int* index) = 0;
    virtual void Wait(MPI_Request* request) = 0;
    virtual int WaitAll(std::vector<MPI_Request>& requests) = 0;
};

}}}
