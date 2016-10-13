//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#pragma once

#if HAS_OPENMPI
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
// Note: the following macros define some of the MPI related functions and constants such that code
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
    virtual bool IsIdle() const = 0;
    virtual bool UsingAllNodes() const = 0;
    virtual size_t MainNodeRank() const = 0;

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

    // helpers to determine the MPI_Datatype of a pointer
    static MPI_Datatype GetDataType(char *);
    static MPI_Datatype GetDataType(int *);
    static MPI_Datatype GetDataType(float *);
    static MPI_Datatype GetDataType(double *);
    static MPI_Datatype GetDataType(size_t *);

    // allreduce of a vector
    virtual void AllReduce(std::vector<size_t>&accumulator) const = 0;
    virtual void AllReduce(std::vector<int>&accumulator) const = 0;
    virtual void AllReduce(std::vector<double>&accumulator) const = 0;
    virtual void AllReduce(std::vector<float>&accumulator) const = 0;

    // for raw pointer
    virtual void AllReduce(size_t*pData, size_t nData) = 0;
    virtual void AllReduce(int*pData, size_t nData) = 0;
    virtual void AllReduce(double*pData, size_t nData) = 0;
    virtual void AllReduce(float*pData, size_t nData) = 0;

    virtual void Bcast(size_t*pData, size_t nData, size_t srcRank) = 0;
    virtual void Bcast(double*pData, size_t nData, size_t srcRank) = 0;
    virtual void Bcast(float*pData, size_t nData, size_t srcRank) = 0;

    // wait for all ranks to reach here
    virtual int WaitAll() = 0;
};


#if HAS_OPENMPI

class MPIWrapperMpi : public MPIWrapper
{
    int m_myRank;
    int m_numMPINodes;
    size_t m_numNodesInUse;

    // MPI communicator that reflects the current subset selection
    MPI_Comm m_currentComm;

    // MPI_Init() with delay-loading the msmpi.dll (possibly causing a failure if missing; we want to catch that)
    int MPI_Init_DL();

    // Workaround for the issue with MPI hanging when we have non-0 exit codes from CNTK processes
    // OpenMPI has a confirmed race condition on killing child process vs. handling their non-zero exit statuses, resulting
    // in a deadlock, where all processes killed but MPI is still waiting.
    // This happens when several perfectly synchronized processes (for example on MPI barrier)
    // simulatenously exit with non-0 exit code.
    // As a workaround, we simply sleep 50*rank miliseconds, effectively "de-synchronizing processes" at exit,
    // allowing MPI to sequentially handle terminations
    static int s_myRank;
    static void MPIWorkaroundAtExit();

public:
    MPIWrapperMpi();

    // Note: we don't clear the sub-communication here although we should, because in case of a crash, this prevents the EXE from terminating.
    // It's OK since this class is a singleton anyway that gets instantiated exactly once at program startup.
    ~MPIWrapperMpi();

private:
    void Ping(const char *msg) const;
    MPI_Comm Communicator() const;

    void RequestNodes(const char *msg, size_t requestednodes = SIZE_MAX /*default: all*/);

public:

    size_t NumNodesInUse() const;
    size_t CurrentNodeRank() const;
    bool IsMainNode() const;
    bool IsIdle() const;
    bool UsingAllNodes() const;
    size_t MainNodeRank() const;

    // -----------------------------------------------------------------------
    // data-exchange functions (wrappers around MPI functions)
    // -----------------------------------------------------------------------

    virtual int Finalize(void);
    virtual int Wait(MPI_Request* request, MPI_Status* status);
    virtual int Waitany(int count, MPI_Request array_of_requests[], int* index, MPI_Status* status);
    virtual int Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);
    virtual int Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, /*MPI_Comm comm,*/ MPI_Request* request);
    virtual int Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, /*MPI_Comm comm,*/ MPI_Status* status);
    virtual int Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, /*MPI_Comm comm,*/ MPI_Request* request);
    virtual int Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, /*MPI_Comm comm,*/ MPI_Request* request);

    // allreduce of a vector
    virtual void AllReduce(std::vector<size_t>&accumulator) const;
    virtual void AllReduce(std::vector<int>&accumulator) const;
    virtual void AllReduce(std::vector<double>&accumulator) const;
    virtual void AllReduce(std::vector<float>&accumulator) const;

    // for raw pointer
    virtual void AllReduce(size_t*pData, size_t nData);
    virtual void AllReduce(int*pData, size_t nData);
    virtual void AllReduce(double*pData, size_t nData);
    virtual void AllReduce(float*pData, size_t nData);

    virtual void Bcast(size_t*pData, size_t nData, size_t srcRank);
    virtual void Bcast(double*pData, size_t nData, size_t srcRank);
    virtual void Bcast(float*pData, size_t nData, size_t srcRank);

    // wait for all ranks to reach here
    int WaitAll();
};

#endif

class MPIWrapperEmpty : public MPIWrapper
{
public:
    MPIWrapperEmpty();

    // Note: we don't clear the sub-communication here although we should, because in case of a crash, this prevents the EXE from terminating.
    // It's OK since this class is a singleton anyway that gets instantiated exactly once at program startup.
    ~MPIWrapperEmpty();

    size_t NumNodesInUse() const;
    size_t CurrentNodeRank() const;
    bool IsMainNode() const;
    bool IsIdle() const;
    bool UsingAllNodes() const;
    size_t MainNodeRank() const;

    // -----------------------------------------------------------------------
    // data-exchange functions (wrappers around MPI functions)
    // -----------------------------------------------------------------------

    virtual int Finalize(void);
    virtual int Wait(MPI_Request* request, MPI_Status* status);
    virtual int Waitany(int count, MPI_Request array_of_requests[], int* index, MPI_Status* status);
    virtual int Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);
    virtual int Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, /*MPI_Comm comm,*/ MPI_Request* request);
    virtual int Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, /*MPI_Comm comm,*/ MPI_Status* status);
    virtual int Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, /*MPI_Comm comm,*/ MPI_Request* request);
    virtual int Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, /*MPI_Comm comm,*/ MPI_Request* request);

    // allreduce of a vector
    virtual void AllReduce(std::vector<size_t>&accumulator) const;
    virtual void AllReduce(std::vector<int>&accumulator) const;
    virtual void AllReduce(std::vector<double>&accumulator) const;
    virtual void AllReduce(std::vector<float>&accumulator) const;

    // for raw pointer
    virtual void AllReduce(size_t*pData, size_t nData);
    virtual void AllReduce(int*pData, size_t nData);
    virtual void AllReduce(double*pData, size_t nData);
    virtual void AllReduce(float*pData, size_t nData);

    virtual void Bcast(size_t*pData, size_t nData, size_t srcRank);
    virtual void Bcast(double*pData, size_t nData, size_t srcRank);
    virtual void Bcast(float*pData, size_t nData, size_t srcRank);

    // wait for all ranks to reach here
    int WaitAll();
};

}}}
