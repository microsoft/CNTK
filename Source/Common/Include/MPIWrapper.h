//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

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
#pragma comment(lib, "msmpi.lib")

#include <errno.h> 
#include <string>
#include <array>
#include <vector>
#include <memory>

#include "CommonMatrix.h"

#define FFLUSH_SUCCESS 0

namespace Microsoft { namespace MSR { namespace CNTK {

struct MpiFail : public std::string
{
    MpiFail(const std::string &what)
        : std::string(what)
    {
    }
};

static int operator||(int rc, const MpiFail &what)
{
    if (rc == MPI_SUCCESS)
    {
        return rc;
    }

    fprintf(stderr, "%s, MPI error %d\n", what.c_str(), rc);
    fflush(stderr);

    // (special case: we use that code to indicate a missing msmpi.dll...)
    if (rc != MPI_ERR_INTERN)
    {
        char errbuf[MPI_MAX_ERROR_STRING + 1] = {0};
        int len;
        MPI_Error_string(rc, &errbuf[0], &len);
        fprintf(stderr, "%s, MPI error %d: %s\n", what.c_str(), rc, errbuf);
        fflush(stderr);

        // we abort through this, so that the MPI system gets the memo
        MPI_Abort(MPI_COMM_WORLD, rc);

        // TODO: or does that only signal an issue, and we should still terminate ourselves?
        // BUGBUG: We'd also need to Abort through the other sub-set communicator
    }
    RuntimeError("%s", what.c_str());
}

class MPIWrapper;
typedef std::shared_ptr<MPIWrapper> MPIWrapperPtr;

class MPIWrapper : public std::enable_shared_from_this<MPIWrapper>
{
    int m_myRank;
    std::wstring m_myName;
    int m_numMPINodes;
    size_t m_numNodesInUse;
    bool m_multiHost;

    // MPI communicator that reflects the current subset selection
    MPI_Comm m_currentComm;

    static MPIWrapperPtr s_mpi;

    // MPI_Init() with delay-loading the msmpi.dll (possibly causing a failure if missing; we want to catch that)
    int MPI_Init_DL()
    {
#ifdef WIN32
        __try
#endif
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
#ifdef WIN32
        __except (EXCEPTION_EXECUTE_HANDLER)
        {
            fprintf(stderr, "mpihelper: msmpi.dll missing\n");
            return MPI_ERR_INTERN;
        }
#endif
    }

    // Workaround for the issue with MPI hanging when we have non-0 exit codes from CNTK processes
    // OpenMPI has a confirmed race condition on killing child process vs. handling their non-zero exit statuses, resulting
    // in a deadlock, where all processes killed but MPI is still waiting.
    // This happens when several perfectly synchronized processes (for example on MPI barrier)
    // simulatenously exit with non-0 exit code.
    // As a workaround, we simply sleep 50*rank miliseconds, effectively "de-synchronizing processes" at exit,
    // allowing MPI to sequentially handle terminations
    static int s_myRank;
    static void MPIWorkaroundAtExit()
    {
        Sleep(s_myRank * 50);
    }

public:
    MPIWrapper()
        : m_currentComm(MPI_COMM_WORLD)
    {
        static bool initialized = false;
        if (initialized)
        {
            LogicError("MPIWrapper: this is a singleton class that can only be instantiated once per process");
        }

        initialized = true;
        
        if (GetMathLibTraceLevel() > 0)
        {
            fprintf(stderr, "MPIWrapper: initializing MPI\n");
            fflush(stderr);
        }

        MPI_Init_DL() || MpiFail("mpiaggregator: MPI_Init");
        MPI_Comm_rank(MPI_COMM_WORLD, &m_myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_numMPINodes);
        m_numNodesInUse = m_numMPINodes;
        m_multiHost = true;

        // Verify that the environment variable used by GetTotalNumberOfMPINodes()  
        // matches what the MPI API says. There're actually two possible cases:
        // 1) when we're running with mpiexec both values have to match;
        // 2) when we're running without mpiexec, the former will return 0, and
        // the later will be set to 1.
        assert((GetTotalNumberOfMPINodes() == 0 && m_numNodesInUse == 1) ||
                (GetTotalNumberOfMPINodes() == m_numNodesInUse));

        char name[BUFSIZ];
        int length;
        MPI_Get_processor_name(name, &length);
        m_myName = std::wstring(name, name+length);

        // Applying MPI workaround
        s_myRank = m_myRank;
        atexit(&MPIWrapper::MPIWorkaroundAtExit);

        // by default we use all of them
        RequestNodes("MPIWrapper");

        if (GetMathLibTraceLevel() > 0)
        {
            if (m_numMPINodes > 1)
                fprintf(stderr, "mpihelper: we are cog %d in a gearbox of %d\n", (int) m_myRank, (int) m_numMPINodes);
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

    // Note that specifically, this function is such that it does not require
    // MPI initialization. Moreover, it can be used without actually loading any
    // MPI libs.
    // TODO: Once we move to dynamic loading for MPI libs on Linux, move it to utilities.
    static int GetTotalNumberOfMPINodes()
    {
#ifdef WIN32
        const char* p = std::getenv("PMI_SIZE");
#else
        const char* p = std::getenv("OMPI_COMM_WORLD_SIZE");
#endif
        if (!p)
        {
            return 0;
        }
        else
        {
            return std::stoi(string(p));
        }
    }

    // Note: we don't clear the sub-communication here although we should, because in case of a crash, this prevents the EXE from terminating.
    // It's OK since this class is a singleton anyway that gets instantiated exactly once at program startup.
    ~MPIWrapper()
    {
        if (GetMathLibTraceLevel() > 0)
        {
            fprintf(stderr, "~MPIWrapper\n");
        }

        // Do not finalize in event of an exception since calling MPI_Finalize without
        // all pending communications being finished results in a hang
        int rc = fflush(stderr);
        if (!std::uncaught_exception())
        {
            if (rc != FFLUSH_SUCCESS)
            {
            #ifdef _WIN32
                RuntimeError("MPIWrapper: Failed to flush stderr, %d", ::GetLastError());
            #else
                RuntimeError("MPIWrapper: Failed to flush stderr, %d", errno);
            #endif
            }

            MPI_Finalize();
        }
    }

private:
    void Ping(const char *msg) const
    {
#undef USE2NDCOMM
#ifndef USE2NDCOMM
        if (NumNodesInUse() != m_numMPINodes)
        {
            fprintf(stderr, "ping [%s]: cannot be applied to subset (%d) of nodes, skipping\n", msg, (int) NumNodesInUse());
            fflush(stderr);
            return;
        }
#endif
        std::array<int, 1> handshake;
        handshake[0] = 1;

        if (GetMathLibTraceLevel() > 0)
        {
            fprintf(stderr, "ping [%s]: %d nodes pinging each other\n", msg, (int) NumNodesInUse());
            fflush(stderr);
        }

        AllReduce(handshake);

        if (GetMathLibTraceLevel() > 0)
        {
            fprintf(stderr, "ping [%s]: all %d nodes responded\n", msg, handshake[0]);
            fflush(stderr);
        }
    }

    void RequestNodes(const char *msg, size_t requestednodes = SIZE_MAX /*default: all*/)
    {
        Ping("requestnodes (before change)");

// undo current split
#ifdef USE2NDCOMM
        if (m_currentComm != MPI_COMM_WORLD /*no subset*/ && m_currentComm != MPI_COMM_NULL /*idle nodes*/)
        {
            fprintf(stderr, "requestnodes: MPI_Comm_free %x\n", (int) m_currentComm);
            fflush(stderr);
            MPI_Comm_free(&m_currentComm) || MpiFail("requestnodes: MPI_Comm_free"); // will leave MPI_COMM_NULL here
        }
#endif
        // reset to MPI_COMM_WORLD
        m_currentComm = MPI_COMM_WORLD;
        // create a new split (unless all nodes were requested)
        if (requestednodes < (size_t) m_numMPINodes)
        {
#ifdef USE2NDCOMM
            fprintf(stderr, "requestnodes: MPI_Comm_split %d\n", (node() < requestednodes) ? 1 : MPI_UNDEFINED);
            fflush(stderr);
            MPI_Comm_split(communicator(), (node() < requestednodes) ? 1 : MPI_UNDEFINED, 0, &m_currentComm) || MpiFail("requestnodes: MPI_Comm_split");
            fprintf(stderr, "requestnodes: MPI_Comm_split -> %x\n", (int) m_currentComm);
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
                    msg, (int) m_numNodesInUse, (int) m_numMPINodes, (int) requestednodes,
                    (int) CurrentNodeRank(), IsIdle() ? "out (idle)" : "in (participating)");
            fflush(stderr);
        }
        Ping("requestnodes (after change)");

        // If all ranks run on a single host, we can enable optimized communication
        // paths (e.g. NCCL). To determine if a single machine is being used, we
        // check that MPI_Get_processor_name matches for all ranks.
        const int nameMax = MPI_MAX_PROCESSOR_NAME + 1;
        char myName[nameMax] = {0};
        int  myNameLen = 0;
        MPI_Get_processor_name(myName, &myNameLen) || MpiFail("requestnodes: MPI_Get_processor_name");
        myName[myNameLen] = '\0';

        std::vector<char> nameBuffer(m_numNodesInUse * nameMax);
        char* allNames = nameBuffer.data();
        MPI_Allgather(myName, nameMax, MPI_CHAR, allNames, nameMax, MPI_CHAR, m_currentComm)
            || MpiFail("requestnodes: MPI_Allgather");

        m_multiHost = false;
        for(size_t i=1; i<m_numNodesInUse; i++)
        {
            if (strcmp(allNames, allNames+i*nameMax) != 0)
            {
                m_multiHost = true;
                break;
            }
        }

        fprintf(stderr, "requestnodes [%s]: using %d out of %d MPI nodes on %s (%d requested); we (%d) are %s\n",
                msg, (int) m_numNodesInUse, (int) m_numMPINodes, m_multiHost ? "multiple hosts" : "a single host",
                (int) requestednodes, (int) CurrentNodeRank(), IsIdle() ? "out (idle)" : "in (participating)");
        fflush(stderr);
    }

public:

    static MPIWrapperPtr GetInstance(bool create = false)
    {
        if (create)
        {
            if (s_mpi != nullptr)
                LogicError("Creating MPIWrapper instance after a GetInstance call has been already made!");
            else
                s_mpi = std::make_shared<MPIWrapper>();
        }

        return s_mpi;
    }

    static void DeleteInstance()
    {
        s_mpi = nullptr;
    }

    MPI_Comm Communicator() const
    {
        return m_currentComm;
    }
    size_t NumNodesInUse() const
    {
        return m_numNodesInUse;
    }
    size_t CurrentNodeRank() const
    {
        return m_myRank;
    }
    std::wstring CurrentNodeName() const
    {
        return m_myName;
    }
    bool IsMainNode() const
    {
        return m_myRank == 0;
    } // we are the chosen one--do extra stuff like saving the model to disk
    bool IsIdle() const
    {
        return CurrentNodeRank() >= NumNodesInUse();
    } // user had requested to not use this many nodes
    bool UsingAllNodes() const
    {
        return NumNodesInUse() == m_numMPINodes;
    } // all nodes participate (used to check whether we can use MPI_Allreduce directly)
    size_t MainNodeRank() const
    {
        return 0;
    }

    bool IsMultiHost()
    {
        return m_multiHost;
    }

    // -----------------------------------------------------------------------
    // data-exchange functions (wrappers around MPI functions)
    // -----------------------------------------------------------------------

    // helpers to determine the MPI_Datatype of a pointer
    static MPI_Datatype GetDataType(char *)
    {
        return MPI_CHAR;
    }
    static MPI_Datatype GetDataType(int *)
    {
        return MPI_INT;
    }
    static MPI_Datatype GetDataType(float *)
    {
        return MPI_FLOAT;
    }
    static MPI_Datatype GetDataType(double *)
    {
        return MPI_DOUBLE;
    }
    static MPI_Datatype GetDataType(size_t *)
    {
        return sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_LONG_LONG_INT;
    }

    // allreduce of a vector
    template <typename VECTORLIKEOBJECT>
    void AllReduce(VECTORLIKEOBJECT &accumulator) const
    {
        auto *dataptr = accumulator.data();
        size_t totalnumelements = accumulator.size();

        // use MPI to compute the sum over all elements in (dataptr, totalnumelements) and redistribute to all nodes
        AllReduce<typename VECTORLIKEOBJECT::value_type>(dataptr, totalnumelements);
    }

    // for raw pointer
    template <class ElemType>
    void AllReduce(ElemType* sendData, size_t numElements, MPI_Op op = MPI_SUM) const
    {
        AllReduce<ElemType>(static_cast<ElemType*>(MPI_IN_PLACE), sendData, numElements, op);
    }

    template <class ElemType> 
    void AllReduceAsync(ElemType* sendData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const
    {
        AllReduceAsync<ElemType>(static_cast<ElemType*>(MPI_IN_PLACE), sendData, numElements, request, op);
    }

    template <class ElemType>
    void AllGatherAsync(const ElemType *sendData, size_t numSendElements, ElemType *receiveData, size_t numRecvElements, MPI_Request* request) const
    {
        MPI_Iallgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallgather");
    }

    template <class ElemType>
    void AllGather(const ElemType *sendData, size_t numSendElements, ElemType *receiveData, size_t numRecvElements) const
    {
        MPI_Allgather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), Communicator()) || MpiFail("AllReduceAsync: MPI_Allgather");
    }

    template <class ElemType>
    void AllReduceAsync(ElemType *sendData, ElemType *receiveData, size_t numElements, MPI_Request* request, MPI_Op op = MPI_SUM) const
    {
        MPI_Iallreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator(), request) || MpiFail("AllReduceAsync: MPI_Iallreduce");
    }

    template <class ElemType>
    void AllReduce(ElemType *sendData, ElemType *receiveData, size_t numElements, MPI_Op op = MPI_SUM) const
    {
        MPI_Allreduce(sendData, receiveData, (int)numElements, GetDataType(sendData), op, Communicator()) || MpiFail("AllReduce: MPI_Allreduce");
    }

    template <class ElemType>
    void Gather(const ElemType *sendData, size_t numSendElements, ElemType *receiveData, size_t numRecvElements, size_t rootRank) const
    {
        MPI_Gather(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, (int)numRecvElements, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gather");
    }

    template <class ElemType>
    void Gatherv(const ElemType *sendData, size_t numSendElements, ElemType *receiveData, int recvCounts[], int offsets[], size_t rootRank) const
    {
        MPI_Gatherv(sendData, (int)numSendElements, GetDataType(receiveData), receiveData, recvCounts, offsets, GetDataType(receiveData), (int)rootRank, Communicator()) || MpiFail("AllReduceAsync: MPI_Gatherv");
    }

    template <class ElemType>
    void Bcast(ElemType *pData, size_t nData, size_t srcRank)
    {
        MPI_Bcast(pData, (int) nData, GetDataType(pData), (int) srcRank, Communicator()) || MpiFail("Bcast: MPI_Bcast");
    }

    // wait for an async request to finish
    void Wait(MPI_Request* request)
    {
        MPI_Wait(request, MPI_STATUSES_IGNORE) || MpiFail("Wait: MPI_Wait");
    }

    void WaitAny(MPI_Request* requests, int numRequests, int* index)
    {
        MPI_Waitany(numRequests, requests, index, MPI_STATUSES_IGNORE) || MpiFail("WaitAny: MPI_Waitany");
    }

    // wait for all ranks to reach here
    void WaitAll()
    {
        MPI_Barrier(m_currentComm) || MpiFail("waitall: MPI_Barrier");
    }

    void WaitAll(std::vector<MPI_Request>& requests)
    {
        MPI_Waitall((int)requests.size(), &requests[0], MPI_STATUSES_IGNORE) || MpiFail("waitall: MPI_Waitall");
    }

    bool IsCudaAware() const
    {
#ifdef __unix__
        return true;
#else
        return false;
#endif
    }
};

}}}
