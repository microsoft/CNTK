//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//
#include "Include/Basics.h"
#include "Include/MPIWrapper.h"
#include "Include/EnvironmentUtil.h"

#if HAS_MPI
#pragma comment(lib, "msmpi.lib")
#else
#define MPI_SUCCESS             0
#define MPI_ERR_INTERN          1
#define MPI_MAX_ERROR_STRING    512
#endif

#define FFLUSH_SUCCESS          0

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// Specific MPIWrapper class definitions.
// -----------------------------------------------------------------------

#if HAS_MPI
class MPIWrapperMpi : public MPIWrapper
{
    int m_myRank;
    std::wstring m_myName;
    int m_numMPINodes;
    size_t m_numNodesInUse;
    bool m_multiHost;

    // MPI communicator that reflects the current subset selection
    MPI_Comm m_currentComm;

    // MPI_Init() is loading the msmpi.dll. Failing to load the dll will terminate the
    // application.
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
    std::wstring CurrentNodeName() const;
    bool IsIdle() const;
    bool UsingAllNodes() const;
    size_t MainNodeRank() const;
    bool IsMultiHost() const;

    // Use GPUDirect RDMA support
    virtual bool UseGpuGdr() override;

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
    virtual int Abort(int errorcode);
    virtual int Error_string(int errorcode, char* string, int* resultlen);

    // allreduce of a vector
    virtual void AllReduce(std::vector<size_t>& accumulator) const;
    virtual void AllReduce(std::vector<int>& accumulator) const;
    virtual void AllReduce(std::vector<double>& accumulator) const;
    virtual void AllReduce(std::vector<float>& accumulator) const;

    // for raw pointer
    virtual void AllReduce(size_t* sendData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(int* sendData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(double* sendData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(float* sendData, size_t numElements, MPI_Op op = MPI_SUM) const;

    virtual void AllReduce(size_t* sendData, size_t* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(int* sendData, int* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(double* sendData, double* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(float* sendData, float* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const;

    virtual void AllReduceAsync(size_t* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(int* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(double* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(float* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;

    virtual void AllReduceAsync(size_t* sendData, size_t* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(int* sendData, int* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(double* sendData, double* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(float* sendData, float* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;

    virtual void Bcast(size_t* sendData, size_t numElements, size_t srcRank);
    virtual void Bcast(double* sendData, size_t numElements, size_t srcRank);
    virtual void Bcast(float* sendData, size_t numElements, size_t srcRank);
    virtual void Bcast(void* buffer, int count, MPI_Datatype datatype, int root);

    virtual void AllGatherAsync(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, MPI_Request* request) const;
    virtual void AllGatherAsync(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, MPI_Request* request) const;
    virtual void AllGatherAsync(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, MPI_Request* request) const;
    virtual void AllGatherAsync(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, MPI_Request* request) const;

    virtual void AllGather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements) const;
    virtual void AllGather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements) const;
    virtual void AllGather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements) const;
    virtual void AllGather(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements) const;
    virtual void Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype) const;

    virtual void Gather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, size_t rootRank) const;
    virtual void Gather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, size_t rootRank) const;
    virtual void Gather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, size_t rootRank) const;
    virtual void Gather(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, size_t rootRank) const;

    virtual void Gatherv(const size_t *sendData, size_t numSendElements, size_t *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;
    virtual void Gatherv(const char *sendData, size_t numSendElements, char *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;
    virtual void Gatherv(const int *sendData, size_t numSendElements, int *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;
    virtual void Gatherv(const float *sendData, size_t numSendElements, float *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;
    virtual void Gatherv(const double *sendData, size_t numSendElements, double *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;

    // wait for all ranks to reach here
    virtual int WaitAll();
    virtual void WaitAny(MPI_Request* requests, int numRequests, int* index);
    virtual void Wait(MPI_Request* request);
    virtual int WaitAll(std::vector<MPI_Request>& requests);
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
    std::wstring CurrentNodeName() const;
    bool IsIdle() const;
    bool UsingAllNodes() const;
    size_t MainNodeRank() const;
    bool IsMultiHost() const;
    // Use GPUDirect RDMA
    virtual bool UseGpuGdr() override;

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
    virtual int Abort(int errorcode);
    virtual int Error_string(int errorcode, char* string, int* resultlen);

    // allreduce of a vector
    virtual void AllReduce(std::vector<size_t>& accumulator) const;
    virtual void AllReduce(std::vector<int>& accumulator) const;
    virtual void AllReduce(std::vector<double>& accumulator) const;
    virtual void AllReduce(std::vector<float>& accumulator) const;

    // for raw pointer
    virtual void AllReduce(size_t* sendData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(int* sendData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(double* sendData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(float* sendData, size_t numElements, MPI_Op op = MPI_SUM) const;

    virtual void AllReduce(size_t* sendData, size_t* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(int* sendData, int* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(double* sendData, double* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const;
    virtual void AllReduce(float* sendData, float* receiveData, size_t numElements, MPI_Op op = MPI_SUM) const;

    virtual void AllReduceAsync(size_t* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(int* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(double* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(float* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;

    virtual void AllReduceAsync(size_t* sendData, size_t* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(int* sendData, int* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(double* sendData, double* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;
    virtual void AllReduceAsync(float* sendData, float* receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const;

    virtual void Bcast(size_t* sendData, size_t numElements, size_t srcRank);
    virtual void Bcast(double* sendData, size_t numElements, size_t srcRank);
    virtual void Bcast(float* sendData, size_t numElements, size_t srcRank);
    virtual void Bcast(void* buffer, int count, MPI_Datatype datatype, int root);

    virtual void AllGatherAsync(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, MPI_Request* request) const;
    virtual void AllGatherAsync(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, MPI_Request* request) const;
    virtual void AllGatherAsync(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, MPI_Request* request) const;
    virtual void AllGatherAsync(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, MPI_Request* request) const;
    virtual void Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype) const;

    virtual void AllGather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements) const;
    virtual void AllGather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements) const;
    virtual void AllGather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements) const;
    virtual void AllGather(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements) const;

    virtual void Gather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, size_t rootRank) const;
    virtual void Gather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, size_t rootRank) const;
    virtual void Gather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, size_t rootRank) const;
    virtual void Gather(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, size_t rootRank) const;

    virtual void Gatherv(const size_t *sendData, size_t numSendElements, size_t *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;
    virtual void Gatherv(const char *sendData, size_t numSendElements, char *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;
    virtual void Gatherv(const int *sendData, size_t numSendElements, int *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;
    virtual void Gatherv(const float *sendData, size_t numSendElements, float *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;
    virtual void Gatherv(const double *sendData, size_t numSendElements, double *receiveData, int recvCounts[], int offsets[], size_t rootRank) const;

    // wait for all ranks to reach here
    virtual int WaitAll();
    virtual void WaitAny(MPI_Request* requests, int numRequests, int* index);
    virtual void Wait(MPI_Request* request);
    virtual int WaitAll(std::vector<MPI_Request>& requests);
};


// -----------------------------------------------------------------------
// Factory pattern.
// Note: the following code would go into a specific mpi wrapper implementation
//       dll and would be called by the loader to retrieve the mpi wrapper
//       instance.
//       The code below would be split into two independent dlls. One
//       referencing the platform mpi libraries, the other providing only
//       empty stubs.
// -----------------------------------------------------------------------

extern "C" void GetMpiWrapper(MPIWrapper **mpi)
{
#if HAS_MPI
    *mpi = new MPIWrapperMpi();
#else
    *mpi = new MPIWrapperEmpty();
#endif
}

// -----------------------------------------------------------------------
// Generic MPIWrapper functions (not related to a specific implementation)
// -----------------------------------------------------------------------

std::shared_ptr<MPIWrapper> MPIWrapper::s_mpi = nullptr;

int operator||(int rc, const MpiFail &what)
{
    if (rc == MPI_SUCCESS)
    {
        return rc;
    }

    fprintf(stderr, "%s, MPI error %d\n", what.c_str(), rc);
    fflush(stderr);

    if (MPIWrapper::s_mpi != nullptr)
    {
        // (special case: we use that code to indicate a missing msmpi.dll...)
        if (rc != MPI_ERR_INTERN)
        {
            char errbuf[MPI_MAX_ERROR_STRING + 1] = { 0 };
            int len = MPI_MAX_ERROR_STRING;
            MPIWrapper::s_mpi->Error_string(rc, &errbuf[0], &len);

            fprintf(stderr, "%s, MPI error %d: %s\n", what.c_str(), rc, errbuf);
            fflush(stderr);

            // we abort through this, so that the MPI system gets the memo
            MPIWrapper::s_mpi->Abort(rc);

            // TODO: or does that only signal an issue, and we should still terminate ourselves?
            // BUGBUG: We'd also need to Abort through the other sub-set communicator
        }
    }

    RuntimeError("%s", what.c_str());
}

// TODO: this is not threadsafe.
//       to make this threadsafe, remove the "create" parameter,
//       replace the s_mpi init with a run-once statement (or guard it with a mutex),
//       and remove the DeleteInstance() function.
MPIWrapperPtr MPIWrapper::GetInstance(bool create)
{
    if (create)
    {
        if (s_mpi != nullptr)
            LogicError("Creating MPIWrapper instance after a GetInstance call has been already made!");
        else
        {
            MPIWrapper *mpi = nullptr;

            // retrieves the raw pointer
            GetMpiWrapper(&mpi);
            if (mpi == nullptr)
                LogicError("Creating MPIWrapper failed to retrieve instance!");

            // makes it a shared pointer
            s_mpi = shared_ptr<MPIWrapper>(mpi);
        }
    }

    return s_mpi;
}

void MPIWrapper::DeleteInstance()
{
    s_mpi = nullptr;
}

// helpers to determine the MPI_Datatype of a pointer
MPI_Datatype MPIWrapper::GetDataType(char *)
{
    return MPI_CHAR;
}

MPI_Datatype MPIWrapper::GetDataType(int *)
{
    return MPI_INT;
}

MPI_Datatype MPIWrapper::GetDataType(float *)
{
    return MPI_FLOAT;
}

MPI_Datatype MPIWrapper::GetDataType(double *)
{
    return MPI_DOUBLE;
}

MPI_Datatype MPIWrapper::GetDataType(size_t *)
{
    return sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_LONG_LONG_INT;
}

#if HAS_MPI
// -----------------------------------------------------------------------
// MPIWrapper that actually calls into msmpi.dll
// -----------------------------------------------------------------------

int MPIWrapperMpi::s_myRank = -1;

MPIWrapperMpi::MPIWrapperMpi()
    : m_currentComm(MPI_COMM_WORLD)
{
    static bool initialized = false;
    if (initialized)
        LogicError("MPIWrapperMpi: this is a singleton class that can only be instantiated once per process");

    initialized = true;

    if (GetMathLibTraceLevel() > 0)
    {
        fprintf(stderr, "MPIWrapperMpi: initializing MPI\n");
        fflush(stderr);
    }

    MPI_Init_DL() || MpiFail("mpiaggregator: MPI_Init");
    MPI_Comm_rank(MPI_COMM_WORLD, &m_myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_numMPINodes);
    m_numNodesInUse = m_numMPINodes;
    m_multiHost = true;

    // Verify that the environment variable used by GetTotalNumberOfMPINodes()  
    // matches what the MPI API says.
    assert(EnvironmentUtil::GetTotalNumberOfMPINodes() == m_numNodesInUse);

    char name[BUFSIZ];
    int length;
    MPI_Get_processor_name(name, &length);
    m_myName = std::wstring(name, name + length);

    // Applying MPI workaround
    s_myRank = m_myRank;
    atexit(&MPIWrapperMpi::MPIWorkaroundAtExit);

    // by default we use all of them
    RequestNodes("MPIWrapperMpi");

    if (GetMathLibTraceLevel() > 0)
    {
        if (m_numMPINodes > 1)
            fprintf(stderr, "mpihelper: we are cog %d in a gearbox of %d\n", (int)m_myRank, (int)m_numMPINodes);
        else
            fprintf(stderr, "mpihelper: only one MPI process: MPI operation will be boring\n");

        fflush(stderr);
    }

    // do an initial handshake
    Ping("mpihelper");

    // stagger the jobs just a little to get a sort-of deterministic order e.g. in GPU allocation when running on one machine
    // continue 0.5 seconds apart
    ::Sleep((DWORD)(500 * CurrentNodeRank()));
}

// Note: we don't clear the sub-communication here although we should, because in case of a crash, this prevents the EXE from terminating.
// It's OK since this class is a singleton anyway that gets instantiated exactly once at program startup.
MPIWrapperMpi::~MPIWrapperMpi()
{
    if (GetMathLibTraceLevel() > 0)
        fprintf(stderr, "~MPIWrapperMpi\n");

    int rc = fflush(stderr);
    if (!std::uncaught_exception())
    {
        if (rc != FFLUSH_SUCCESS)
        {
#ifdef _WIN32
            RuntimeError("MPIWrapperMpi: Failed to flush stderr, %d", ::GetLastError());
#else
            RuntimeError("MPIWrapperMpi: Failed to flush stderr, %d", errno);
#endif
        }
    }
}

// MPI_Init() is loading the msmpi.dll. Failing to load the dll will terminate the
// application.
int MPIWrapperMpi::MPI_Init_DL()
{
    // don't initialize if that has been done already
    int flag = 0;
    MPI_Initialized(&flag);
    if (flag)
        return MPI_SUCCESS;

    int argc = 0;
    char **argv = NULL;
    // TODO(qiwye) Multiverso(parameter server) will benefit from MPI_THREAD_MULTIPLE .
    int requiredThreadLevelSupport = MPI_THREAD_SERIALIZED;
    int provided;
    int ret = MPI_Init_thread(&argc, &argv, requiredThreadLevelSupport, &provided);
    if (provided != requiredThreadLevelSupport)
        LogicError("Failed to initialize MPI with the desired level of thread support");

    return ret;
}

// Workaround for the issue with MPI hanging when we have non-0 exit codes from CNTK processes
// OpenMPI has a confirmed race condition on killing child process vs. handling their non-zero exit statuses, resulting
// in a deadlock, where all processes killed but MPI is still waiting.
// This happens when several perfectly synchronized processes (for example on MPI barrier)
// simulatenously exit with non-0 exit code.
// As a workaround, we simply sleep 50*rank miliseconds, effectively "de-synchronizing processes" at exit,
// allowing MPI to sequentially handle terminations
void MPIWrapperMpi::MPIWorkaroundAtExit()
{
    Sleep(s_myRank * 50);
}

void MPIWrapperMpi::Ping(const char *msg) const
{
#undef USE2NDCOMM
#ifndef USE2NDCOMM
    if (NumNodesInUse() != m_numMPINodes)
    {
        if (GetMathLibTraceLevel() > 0)
        {
            fprintf(stderr, "ping [%s]: cannot be applied to subset (%d) of nodes, skipping\n", msg, (int)NumNodesInUse());
            fflush(stderr);
        }
        return;
    }
#endif
    std::vector<int> handshake;
    handshake.push_back(1);

    fprintf(stderr, "ping [%s]: %d nodes pinging each other\n", msg, (int)NumNodesInUse());
    fflush(stderr);

    AllReduce(handshake);

    if (GetMathLibTraceLevel() > 0)
    {
        fprintf(stderr, "ping [%s]: all %d nodes responded\n", msg, handshake[0]);
        fflush(stderr);
    }
}

void MPIWrapperMpi::RequestNodes(const char *msg, size_t requestednodes /*default: all*/)
{
    Ping("requestnodes (before change)");

    // undo current split
#ifdef USE2NDCOMM
    if (m_currentComm != MPI_COMM_WORLD /*no subset*/ && m_currentComm != MPI_COMM_NULL /*idle nodes*/)
    {
        fprintf(stderr, "requestnodes: MPI_Comm_free %x\n", (int)m_currentComm);
        fflush(stderr);
        MPI_Comm_free(&m_currentComm) || MpiFail("requestnodes: MPI_Comm_free"); // will leave MPI_COMM_NULL here
    }
#endif
    // reset to MPI_COMM_WORLD
    m_currentComm = MPI_COMM_WORLD;
    // create a new split (unless all nodes were requested)
    if (requestednodes < (size_t)m_numMPINodes)
    {
#ifdef USE2NDCOMM
        fprintf(stderr, "requestnodes: MPI_Comm_split %d\n", (node() < requestednodes) ? 1 : MPI_UNDEFINED);
        fflush(stderr);
        MPI_Comm_split(communicator(), (node() < requestednodes) ? 1 : MPI_UNDEFINED, 0, &m_currentComm) || MpiFail("requestnodes: MPI_Comm_split");
        fprintf(stderr, "requestnodes: MPI_Comm_split -> %x\n", (int)m_currentComm);
        fflush(stderr);
#endif
    }
    else
    {
        // leave m_currentComm as MPI_COMM_WORLD
        // and clip to #nodes
        requestednodes = m_numMPINodes;
    }

    m_numNodesInUse = requestednodes;

    if (GetMathLibTraceLevel() > 0)
    {
        fprintf(stderr, "requestnodes [%s]: using %d out of %d MPI nodes (%d requested); we (%d) are %s\n",
            msg, (int)m_numNodesInUse, (int)m_numMPINodes, (int)requestednodes,
            (int)CurrentNodeRank(), IsIdle() ? "out (idle)" : "in (participating)");
        fflush(stderr);
    }
    Ping("requestnodes (after change)");

    // If all ranks run on a single host, we can enable optimized communication
    // paths (e.g. NCCL). To determine if a single machine is being used, we
    // check that MPI_Get_processor_name matches for all ranks.
    const int nameMax = MPI_MAX_PROCESSOR_NAME + 1;
    char myName[nameMax] = { 0 };
    int  myNameLen = 0;
    MPI_Get_processor_name(myName, &myNameLen) || MpiFail("requestnodes: MPI_Get_processor_name");
    myName[myNameLen] = '\0';

    std::vector<char> nameBuffer(m_numNodesInUse * nameMax);
    char* allNames = nameBuffer.data();
    MPI_Allgather(myName, nameMax, MPI_CHAR, allNames, nameMax, MPI_CHAR, m_currentComm)
        || MpiFail("requestnodes: MPI_Allgather");

    m_multiHost = false;
    for (size_t i = 1; i<m_numNodesInUse; i++)
    {
        if (strcmp(allNames, allNames + i*nameMax) != 0)
        {
            m_multiHost = true;
            break;
        }
    }

    fprintf(stderr, "requestnodes [%s]: using %d out of %d MPI nodes on %s (%d requested); we (%d) are %s\n",
        msg, (int)m_numNodesInUse, (int)m_numMPINodes, m_multiHost ? "multiple hosts" : "a single host",
        (int)requestednodes, (int)CurrentNodeRank(), IsIdle() ? "out (idle)" : "in (participating)");
    fflush(stderr);
}

bool MPIWrapperMpi::IsMultiHost() const
{
    return m_multiHost;
}

MPI_Comm MPIWrapperMpi::Communicator() const
{
    return m_currentComm;
}

int MPIWrapperMpi::Finalize(void)
{
    return MPI_Finalize();
}

// wait for all ranks to reach here
int MPIWrapperMpi::WaitAll()
{
    return MPI_Barrier(m_currentComm) || MpiFail("waitall: MPI_Barrier");
}

int MPIWrapperMpi::Wait(MPI_Request* request, MPI_Status* status)
{
    return MPI_Wait(request, status);
}

int MPIWrapperMpi::WaitAll(std::vector<MPI_Request>& requests)
{
    return MPI_Waitall((int)requests.size(), &requests[0], MPI_STATUSES_IGNORE) || MpiFail("waitall: MPI_Waitall");
}

int MPIWrapperMpi::Waitany(int count, MPI_Request array_of_requests[], int* index, MPI_Status* status)
{
    return MPI_Waitany(count, array_of_requests, index, status);
}

int MPIWrapperMpi::Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])
{
    return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

int MPIWrapperMpi::Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Request* request)
{
    return MPI_Isend(buf, count, datatype, dest, tag, m_currentComm, request);
}

int MPIWrapperMpi::Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Status* status)
{
    return MPI_Recv(buf, count, datatype, source, tag, m_currentComm, status);
}

int MPIWrapperMpi::Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Request* request)
{
    return MPI_Irecv(buf, count, datatype, source, tag, m_currentComm, request);
}

int MPIWrapperMpi::Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Request* request)
{
    return MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, m_currentComm, request);
}

int MPIWrapperMpi::Abort(int errorcode)
{
    // we abort through this, so that the MPI system gets the memo
    return MPI_Abort(MPI_COMM_WORLD, errorcode);
}

int MPIWrapperMpi::Error_string(int errorcode, char* str, int* resultlen)
{
    return MPI_Error_string(errorcode, str, resultlen);
}

bool MPIWrapperMpi::UseGpuGdr()
{
    // Only support GPUDirect RDMA on Unix and built with GDR
#if defined(USE_CUDA_GDR) && defined(__unix__)
    return true;
#else
    return false;
#endif
}

size_t MPIWrapperMpi::NumNodesInUse() const
{
    return m_numNodesInUse;
}

size_t MPIWrapperMpi::CurrentNodeRank() const
{
    return m_myRank;
}

std::wstring MPIWrapperMpi::CurrentNodeName() const
{
    return m_myName;
}

bool MPIWrapperMpi::IsMainNode() const
{
    return m_myRank == 0;
} // we are the chosen one--do extra stuff like saving the model to disk

bool MPIWrapperMpi::IsIdle() const
{
    return CurrentNodeRank() >= NumNodesInUse();
} // user had requested to not use this many nodes

bool MPIWrapperMpi::UsingAllNodes() const
{
    return NumNodesInUse() == m_numMPINodes;
} // all nodes participate (used to check whether we can use MPI_Allreduce directly)

size_t MPIWrapperMpi::MainNodeRank() const
{
    return 0;
}

// allreduce of a vector
void MPIWrapperMpi::AllReduce(std::vector<size_t>& accumulator) const
{
    auto *dataptr = accumulator.data();
    size_t totalnumelements = accumulator.size();

    // use MPI to compute the sum over all elements in (dataptr, totalnumelements) and redistribute to all nodes
    AllReduce(dataptr, totalnumelements);
}

void MPIWrapperMpi::AllReduce(std::vector<int>& accumulator) const
{
    auto *dataptr = accumulator.data();
    size_t totalnumelements = accumulator.size();

    // use MPI to compute the sum over all elements in (dataptr, totalnumelements) and redistribute to all nodes
    AllReduce(dataptr, totalnumelements);
}

void MPIWrapperMpi::AllReduce(std::vector<double>& accumulator) const
{
    auto *dataptr = accumulator.data();
    size_t totalnumelements = accumulator.size();

    // use MPI to compute the sum over all elements in (dataptr, totalnumelements) and redistribute to all nodes
    AllReduce(dataptr, totalnumelements);
}

void MPIWrapperMpi::AllReduce(std::vector<float>& accumulator) const
{
    auto *dataptr = accumulator.data();
    size_t totalnumelements = accumulator.size();

    // use MPI to compute the sum over all elements in (dataptr, totalnumelements) and redistribute to all nodes
    AllReduce(dataptr, totalnumelements);
}

// for raw pointer
void MPIWrapperMpi::AllReduce(size_t* sendData, size_t numElements, MPI_Op op) const
{
    AllReduce(static_cast<size_t*>(MPI_IN_PLACE), sendData, numElements, op);
}

void MPIWrapperMpi::AllReduce(int* sendData, size_t numElements, MPI_Op op) const
{
    AllReduce(static_cast<int*>(MPI_IN_PLACE), sendData, numElements, op);
}

void MPIWrapperMpi::AllReduce(double* sendData, size_t numElements, MPI_Op op) const
{
    AllReduce(static_cast<double*>(MPI_IN_PLACE), sendData, numElements, op);
}

void MPIWrapperMpi::AllReduce(float* sendData, size_t numElements, MPI_Op op) const
{
    AllReduce(static_cast<float*>(MPI_IN_PLACE), sendData, numElements, op);
}

void MPIWrapperMpi::AllReduce(size_t* sendData, size_t* receiveData, size_t numElements, MPI_Op op) const
{
    MPI_Allreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator()) || MpiFail("Allreduce: MPI_Allreduce");
}

void MPIWrapperMpi::AllReduce(int* sendData, int* receiveData, size_t numElements, MPI_Op op) const
{
    MPI_Allreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator()) || MpiFail("Allreduce: MPI_Allreduce");
}

void MPIWrapperMpi::AllReduce(double* sendData, double* receiveData, size_t numElements, MPI_Op op) const
{
    MPI_Allreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator()) || MpiFail("Allreduce: MPI_Allreduce");
}

void MPIWrapperMpi::AllReduce(float* sendData, float* receiveData, size_t numElements, MPI_Op op) const
{
    MPI_Allreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator()) || MpiFail("Allreduce: MPI_Allreduce");
}

void MPIWrapperMpi::Bcast(size_t* sendData, size_t numElements, size_t srcRank)
{
    MPI_Bcast(sendData, (int)numElements, GetDataType(sendData), (int)srcRank, Communicator()) || MpiFail("Bcast: MPI_Bcast");
}

void MPIWrapperMpi::AllReduceAsync(size_t* sendData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
    AllReduceAsync(static_cast<size_t*>(MPI_IN_PLACE), sendData, numElements, request, op);
}

void MPIWrapperMpi::AllReduceAsync(int* sendData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
    AllReduceAsync(static_cast<int*>(MPI_IN_PLACE), sendData, numElements, request, op);
}

void MPIWrapperMpi::AllReduceAsync(double* sendData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
    AllReduceAsync(static_cast<double*>(MPI_IN_PLACE), sendData, numElements, request, op);
}

void MPIWrapperMpi::AllReduceAsync(float* sendData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
    AllReduceAsync(static_cast<float*>(MPI_IN_PLACE), sendData, numElements, request, op);
}

void MPIWrapperMpi::AllReduceAsync(size_t *sendData, size_t *receiveData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
    MPI_Iallreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallreduce");
}

void MPIWrapperMpi::AllReduceAsync(int *sendData, int *receiveData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
    MPI_Iallreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallreduce");
}
void MPIWrapperMpi::AllReduceAsync(double *sendData, double *receiveData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
    MPI_Iallreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallreduce");
}
void MPIWrapperMpi::AllReduceAsync(float *sendData, float *receiveData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
    MPI_Iallreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallreduce");
}


void MPIWrapperMpi::Bcast(double* sendData, size_t numElements, size_t srcRank)
{
    MPI_Bcast(sendData, (int)numElements, GetDataType(sendData), (int)srcRank, Communicator()) || MpiFail("Bcast: MPI_Bcast");
}

void MPIWrapperMpi::Bcast(float* sendData, size_t numElements, size_t srcRank)
{
    MPI_Bcast(sendData, (int)numElements, GetDataType(sendData), (int)srcRank, Communicator()) || MpiFail("Bcast: MPI_Bcast");
}

void MPIWrapperMpi::Bcast(void* buffer, int count, MPI_Datatype datatype, int root)
{
    MPI_Bcast(buffer, count, datatype, root, Communicator()) || MpiFail("Bcast: MPI_Bcast");
}

void MPIWrapperMpi::AllGatherAsync(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, MPI_Request* request) const
{
    MPI_Iallgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallgather");
}

void MPIWrapperMpi::AllGatherAsync(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, MPI_Request* request) const
{
    MPI_Iallgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallgather");
}

void MPIWrapperMpi::AllGatherAsync(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, MPI_Request* request) const
{
    MPI_Iallgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallgather");
}

void MPIWrapperMpi::AllGatherAsync(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, MPI_Request* request) const
{
    MPI_Iallgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallgather");
}

void MPIWrapperMpi::AllGather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements) const
{
    MPI_Allgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator()) || MpiFail("AllReduceAsync: MPI_Allgather");
}

void MPIWrapperMpi::AllGather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements) const
{
    MPI_Allgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator()) || MpiFail("AllReduceAsync: MPI_Allgather");
}

void MPIWrapperMpi::AllGather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements) const
{
    MPI_Allgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator()) || MpiFail("AllReduceAsync: MPI_Allgather");
}

void MPIWrapperMpi::AllGather(const double *sendData, size_t numSendElements, double*receiveData, size_t numRecvElements) const
{
    MPI_Allgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator()) || MpiFail("AllReduceAsync: MPI_Allgather");
}

void MPIWrapperMpi::Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype) const
{
    MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, Communicator()) || MpiFail("AllReduceAsync: MPI_Allgather");
}

void MPIWrapperMpi::Gather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, size_t rootRank) const
{
    MPI_Gather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gather");
}

void MPIWrapperMpi::Gather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, size_t rootRank) const
{
    MPI_Gather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gather");
}

void MPIWrapperMpi::Gather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, size_t rootRank) const
{
    MPI_Gather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gather");
}

void MPIWrapperMpi::Gather(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, size_t rootRank) const
{
    MPI_Gather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gather");
}

void MPIWrapperMpi::Gatherv(const size_t *sendData, size_t numSendElements, size_t *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
    MPI_Gatherv(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, recvCounts, offsets, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gatherv");
}

void MPIWrapperMpi::Gatherv(const char *sendData, size_t numSendElements, char *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
    MPI_Gatherv(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, recvCounts, offsets, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gatherv");
}

void MPIWrapperMpi::Gatherv(const int *sendData, size_t numSendElements, int *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
    MPI_Gatherv(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, recvCounts, offsets, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gatherv");
}

void MPIWrapperMpi::Gatherv(const float *sendData, size_t numSendElements, float *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
    MPI_Gatherv(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, recvCounts, offsets, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gatherv");
}

void MPIWrapperMpi::Gatherv(const double *sendData, size_t numSendElements, double *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
    MPI_Gatherv(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, recvCounts, offsets, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gatherv");
}

// wait for an async request to finish
void MPIWrapperMpi::Wait(MPI_Request* request)
{
    MPI_Wait(request, MPI_STATUSES_IGNORE) || MpiFail("Wait: MPI_Wait");
}

void MPIWrapperMpi::WaitAny(MPI_Request* requests, int numRequests, int* index)
{
    MPI_Waitany(numRequests, requests, index, MPI_STATUSES_IGNORE) || MpiFail("WaitAny: MPI_Waitany");
}

#endif


// -----------------------------------------------------------------------
// MPIWrapperEmpty that does nothing
// -----------------------------------------------------------------------

#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced formal parameter

MPIWrapperEmpty::MPIWrapperEmpty()
{
    static bool initialized = false;
    if (initialized)
    {
        LogicError("MPIWrapperEmpty: this is a singleton class that can only be instantiated once per process");
    }

    initialized = true;
    fprintf(stderr, "MPIWrapperEmpty: initializing\n");
    fflush(stderr);
}

MPIWrapperEmpty::~MPIWrapperEmpty()
{
    fprintf(stderr, "~MPIWrapperEmpty\n");
}

bool MPIWrapperEmpty::IsMultiHost() const
{
    return false;
}

bool MPIWrapperEmpty::UseGpuGdr()
{
    return false;
}

int MPIWrapperEmpty::Finalize(void)
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::WaitAll()
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::Wait(MPI_Request* request, MPI_Status* status)
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::WaitAll(std::vector<MPI_Request>& requests)
{
    return MPI_UNDEFINED;
}


int MPIWrapperEmpty::Waitany(int count, MPI_Request array_of_requests[], int* index, MPI_Status* status)
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Request* request)
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Status* status)
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Request* request)
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Request* request)
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::Abort(int errorcode)
{
    return MPI_UNDEFINED;
}

int MPIWrapperEmpty::Error_string(int errorcode, char* str, int* resultlen)
{
    if (!str || !resultlen)
    {
        return MPI_UNDEFINED;
    }

    *resultlen = sprintf(str, "Error-%d", errorcode);
    return MPI_SUCCESS;
}

size_t MPIWrapperEmpty::NumNodesInUse() const
{
    return 1;
}

size_t MPIWrapperEmpty::CurrentNodeRank() const
{
    return 0;
}

std::wstring MPIWrapperEmpty::CurrentNodeName() const
{
    return L"localhost";
}

bool MPIWrapperEmpty::IsMainNode() const
{
    return true;
}

bool MPIWrapperEmpty::IsIdle() const
{
    return CurrentNodeRank() >= NumNodesInUse();
}

bool MPIWrapperEmpty::UsingAllNodes() const
{
    return true;
}

size_t MPIWrapperEmpty::MainNodeRank() const
{
    return 0;
}

// allreduce of a vector
void MPIWrapperEmpty::AllReduce(std::vector<size_t>& accumulator) const
{
}

void MPIWrapperEmpty::AllReduce(std::vector<int>& accumulator) const
{
}

void MPIWrapperEmpty::AllReduce(std::vector<double>& accumulator) const
{
}

void MPIWrapperEmpty::AllReduce(std::vector<float>& accumulator) const
{
}

// for raw pointer
void MPIWrapperEmpty::AllReduce(size_t* sendData, size_t numElements, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduce(int* sendData, size_t numElements, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduce(double* sendData, size_t numElements, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduce(float* sendData, size_t numElements, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduce(size_t* sendData, size_t* receiveData, size_t numElements, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduce(int* sendData, int* receiveData, size_t numElements, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduce(double* sendData, double* receiveData, size_t numElements, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduce(float* sendData, float* receiveData, size_t numElements, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduceAsync(size_t* sendData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduceAsync(int* sendData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduceAsync(double* sendData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduceAsync(float* sendData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduceAsync(size_t *sendData, size_t *receiveData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
}

void MPIWrapperEmpty::AllReduceAsync(int *sendData, int *receiveData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
}
void MPIWrapperEmpty::AllReduceAsync(double *sendData, double *receiveData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
}
void MPIWrapperEmpty::AllReduceAsync(float *sendData, float *receiveData, size_t numElements, MPI_Request* request, MPI_Op op) const
{
}

void MPIWrapperEmpty::Bcast(size_t* sendData, size_t numElements, size_t srcRank)
{
}

void MPIWrapperEmpty::Bcast(double* sendData, size_t numElements, size_t srcRank)
{
}

void MPIWrapperEmpty::Bcast(float* sendData, size_t numElements, size_t srcRank)
{
}

void MPIWrapperEmpty::Bcast(void* buffer, int count, MPI_Datatype datatype, int root)
{
}

void MPIWrapperEmpty::AllGatherAsync(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, MPI_Request* request) const
{
}

void MPIWrapperEmpty::AllGatherAsync(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, MPI_Request* request) const
{
}

void MPIWrapperEmpty::AllGatherAsync(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, MPI_Request* request) const
{
}

void MPIWrapperEmpty::AllGatherAsync(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, MPI_Request* request) const
{
}

void MPIWrapperEmpty::AllGather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements) const
{
}

void MPIWrapperEmpty::AllGather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements) const
{
}

void MPIWrapperEmpty::AllGather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements) const
{
}

void MPIWrapperEmpty::AllGather(const double *sendData, size_t numSendElements, double*receiveData, size_t numRecvElements) const
{
}

void MPIWrapperEmpty::Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype) const
{
}

void MPIWrapperEmpty::Gather(const size_t *sendData, size_t numSendElements, size_t *receiveData, size_t numRecvElements, size_t rootRank) const
{
}

void MPIWrapperEmpty::Gather(const int *sendData, size_t numSendElements, int *receiveData, size_t numRecvElements, size_t rootRank) const
{
}

void MPIWrapperEmpty::Gather(const float *sendData, size_t numSendElements, float *receiveData, size_t numRecvElements, size_t rootRank) const
{
}

void MPIWrapperEmpty::Gather(const double *sendData, size_t numSendElements, double *receiveData, size_t numRecvElements, size_t rootRank) const
{
}

void MPIWrapperEmpty::Gatherv(const size_t *sendData, size_t numSendElements, size_t *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
}

void MPIWrapperEmpty::Gatherv(const char *sendData, size_t numSendElements, char *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
}

void MPIWrapperEmpty::Gatherv(const int *sendData, size_t numSendElements, int *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
}

void MPIWrapperEmpty::Gatherv(const float *sendData, size_t numSendElements, float *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
}

void MPIWrapperEmpty::Gatherv(const double *sendData, size_t numSendElements, double *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
{
}


void MPIWrapperEmpty::Wait(MPI_Request* request)
{
}

void MPIWrapperEmpty::WaitAny(MPI_Request* requests, int numRequests, int* index)
{
}

#pragma warning(pop)

}}}
