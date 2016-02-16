#pragma once

// This uses mpi.h which requires the Microsoft MPI SDK to be installed on Windows
// [cf. https://msdn.microsoft.com/en-us/library/bb524831(v=vs.85).aspx]
// download msmpisdk.msi at https://www.microsoft.com/en-us/download/details.aspx?id=49926 and run it
// and the MPI dev package on Linux (sudo apt-get install libopenmpi-dev openmpi-bin openmpi-doc)
#include "mpi.h"
#pragma comment(lib, "msmpi.lib")

#include <string>
#include <array>
#include <vector>
#include <algorithm> // for find

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

// the following structure is defined for MA-related purpose. 
// To use MA for the current HTKMLFReader, we need to keep MPI nodes synced with each other
// Each node needs to know his peer's status: DataProcessing or DataEnd 
// 1.   When a node has processed enough samples (SyncFrequencyInFrames), 
//      1) it will first send its current status "DataProcessing" to peers (non-blocking call)
//      2) it will call MPI_Recv to receive other's status (blocking call)
//      3) if all the peers are in the DataProcessing status, a collective Bcast call will be posted 
//         if some peer have been in the DataEnd status, no model sync will be performed
// 2.   When a node has arrived at the end of epoch (DataEnd)
//      1) it will first send its current status "DataEnd" to peers (non-blocking call)
//      2) it will call MPI_Recv to receive other's status to complete MPI_Isend call from others 
//      3) it will call MPI_Barrier to wait for all the other peers 

enum class NodeStatus
{
    DataProcessing = 0,
    DataEnd = 1 
};


class MPIWrapper
{
private:
    int m_myRank;
    int m_numMPINodes;
    size_t m_numNodesInUse;

    // MPI communicator that reflects the current subset selection
    MPI_Comm m_currentComm;

    int m_trace;        // use trace to log performed related numbers
    
    // MA-related status 
    std::vector<NodeStatus>     m_peerStatus; 
    int                         m_numSyncPerformed;         
    // this is the counter of number sync performed, it will NOT get reset after each epoch. 
    // This is used as MPI_TAG to distinguish messages sent at different stages 
    // This means the maximum number of sync to be safely performed is 2,147,483,647 
    // If the SyncFreqInFrames is set as 40K, this means it allows to sweep 238 million hours of speech data, assuming 10ms per sample 

private: 
    bool PeersHaveEndProcessing()
    {
        auto iter=std::find(m_peerStatus.begin(), m_peerStatus.end(), NodeStatus::DataEnd);
        return iter != m_peerStatus.end(); 
        /*
        bool b = false;
        for (auto x : m_peerStatus)
        {
            if (x == NodeStatus::DataEnd)
            {
                b = true; break;
            }
        }
        return b;
        */
    }

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
        // Note: we can't use g_mpi, since MPI stack is already down at this point
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
        fprintf(stderr, "MPIWrapper: initializing MPI\n");
        fflush(stderr);

        MPI_Init_DL() || MpiFail("mpiaggregator: MPI_Init");
        MPI_Comm_rank(MPI_COMM_WORLD, &m_myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_numMPINodes);
        m_numNodesInUse = m_numMPINodes;

        // Applying MPI workaround
        s_myRank = m_myRank;
        atexit(&MPIWrapper::MPIWorkaroundAtExit);

        // by default we use all of them
        RequestNodes("MPIWrapper");

        if (m_numMPINodes > 1)
            fprintf(stderr, "mpihelper: we are cog %d in a gearbox of %d\n", (int) m_myRank, (int) m_numMPINodes);
        else
            fprintf(stderr, "mpihelper: only one MPI process: MPI operation will be boring\n");

        fflush(stderr);

        // do an initial handshake
        Ping("mpihelper");

        m_trace = 0; 
        m_numSyncPerformed = 0; 
        m_peerStatus.resize(NumNodesInUse(), NodeStatus::DataProcessing);
        // stagger the jobs just a little to get a sort-of deterministic order e.g. in GPU allocation when running on one machine
        // continue 0.5 seconds apart
        ::Sleep((DWORD)(500 * CurrentNodeRank()));
    }

    // Note: we don't clear the sub-communication here although we should, because in case of a crash, this prevents the EXE from terminating.
    // It's OK since this class is a singleton anyway that gets instantiated exactly once at program startup.
    ~MPIWrapper()
    {
        fprintf(stderr, "~MPIWrapper\n");
        fflush(stderr);
        MPI_Finalize();
    }

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

        fprintf(stderr, "ping [%s]: %d nodes pinging each other\n", msg, (int) NumNodesInUse());
        fflush(stderr);

        AllReduce(handshake);
        fprintf(stderr, "ping [%s]: all %d nodes responded\n", msg, handshake[0]);
        fflush(stderr);
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
        fprintf(stderr, "requestnodes [%s]: using %d out of %d MPI nodes (%d requested); we (%d) are %s\n",
                msg, (int) m_numNodesInUse, (int) m_numMPINodes, (int) requestednodes,
                (int) CurrentNodeRank(), IsIdle() ? "out (idle)" : "in (participating)");
        fflush(stderr);
        Ping("requestnodes (after change)");
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
    void SetTrace(int traceLevel)
    {
        m_trace = traceLevel; 
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
        if ((NumNodesInUse() > 1) && (Communicator() != MPI_COMM_NULL))
        {
            MPI_Allreduce(MPI_IN_PLACE, dataptr, (int) totalnumelements, GetDataType(dataptr), MPI_SUM, Communicator()) || MpiFail("allreduce: MPI_Allreduce");
        }
    }

    // for raw pointer
    template <class ElemType>
    void AllReduce(ElemType *pData, size_t nData)
    {
        if ((NumNodesInUse() > 1 && (Communicator() != MPI_COMM_NULL)))
        {
            MPI_Allreduce(MPI_IN_PLACE, pData, (int) nData, GetDataType(pData), MPI_SUM, Communicator()) || MpiFail("Allreduce: MPI_Allreduce");
        }
    }

    template <class ElemType>
    void Bcast(ElemType *pData, size_t nData, size_t srcRank)
    {
        if ((NumNodesInUse() > 1) && (Communicator() != MPI_COMM_NULL))
        {
            MPI_Bcast(pData, (int) nData, GetDataType(pData), (int) srcRank, Communicator()) || MpiFail("Bcast: MPI_Bcast");
        }
    }

    // wait for all ranks to reach here
    void WaitAll()
    {
        MPI_Barrier(m_currentComm) || MpiFail("waitall: MPI_Barrier");
    }

    //////////////////////////////////////////////////////////////////////////
    //  MA-related helper function 
    //  OnStartDataProcessing:      This function is called at the begining of each epoch 
    //  PeersHaveEndProcessing:     return true if any of my peers have arrived at DataEnd
    //  OnArriveAtSyncPoint:        This function is called when a potential sync point is arrived 
    //                              It returns true if a sync is needed, false if no (which means someone has already arrive at DataEnd)
    //  OnPerformedOneSync:         This function is called after a sync is performed 
    //  OnArriveAtEndOfDataProcessing: This function is called when arriving at the end of each epoch
    //////////////////////////////////////////////////////////////////////////
    void OnStartDataProcessing()
    {
        WaitAll(); 
        m_peerStatus.resize(NumNodesInUse());
        std::fill(m_peerStatus.begin(), m_peerStatus.end(), NodeStatus::DataProcessing);
    }
    bool OnArriveAtSyncPoint()
    {   
        bool ret = false; 
        if (!PeersHaveEndProcessing())
        {
            int sentSignal = (int)NodeStatus::DataProcessing; 
            vector<MPI_Request> sendRequests(NumNodesInUse());
            // 1. send my status to peers (non-blocking)
            for (int dest = 0; dest < (int)NumNodesInUse(); dest++)
            {
                if (dest != m_myRank)
                {
                    MPI_Isend(&sentSignal, 1, MPI_INT, dest, m_numSyncPerformed, m_currentComm, &sendRequests[dest]);
                }
            }
            // 2. recv status from others (blocking call)
            for (int src = 0; src < (int)NumNodesInUse(); src++)
            {
                if (src != m_myRank)
                {
                    int recvSignal = 0; 
                    MPI_Status status; 
                    MPI_Recv(&recvSignal, 1, MPI_INT, src, m_numSyncPerformed, m_currentComm, &status); 
                    // for debugging purpose, to be removed when mature 
                    assert(status.MPI_SOURCE == src); 
                    assert(status.MPI_TAG == m_numSyncPerformed); 
                    m_peerStatus[src] = (NodeStatus)recvSignal;
                }
            }
            // 3. makes sure the sending operation has completed 
            for (int dest = 0; dest < (int)NumNodesInUse(); dest++)
            {
                if (dest != m_myRank)
                {
                    MPI_Wait(&sendRequests[dest], MPI_STATUS_IGNORE); 
                }
            }
            // 4. check peer status and return whether we can sync 
            ret=!PeersHaveEndProcessing(); 
        }
        return ret; 
    }
    void OnPerformedOneSync()
    {
        m_numSyncPerformed++;
    }
    bool OnArriveAtEndOfDataProcessing()
    {
        vector<MPI_Request> sendRequests(NumNodesInUse());
        int sentSignal = (int)NodeStatus::DataEnd; 
        // 1. send my status to notify peers 
        for (int dest = 0; dest < (int)NumNodesInUse(); dest++)
        {
            if (dest != m_myRank)
            {
                MPI_Isend(&sentSignal, 1, MPI_INT, dest, m_numSyncPerformed, m_currentComm, &sendRequests[dest]);
            }
        }
        // 2. recv others 
        for (int src = 0; src < NumNodesInUse(); src++)
        {
            if (src != m_myRank && m_peerStatus[src] == NodeStatus::DataProcessing)
            {
                int recvSignal = 0; 
                MPI_Status status; 
                MPI_Recv(&recvSignal, 1, MPI_INT, src, m_numSyncPerformed, m_currentComm, &status);
                m_peerStatus[src] = (NodeStatus)recvSignal; 
                assert(status.MPI_SOURCE == src); 
                assert(status.MPI_TAG == m_numSyncPerformed);
            }
        }
        // 3. make sure sending operation finished 
        for (int dest = 0; dest < (int)NumNodesInUse(); dest++)
        {
            if (dest != m_myRank)
            {
                MPI_Wait(&sendRequests[dest], MPI_STATUS_IGNORE);
            }
        }
        m_peerStatus[m_myRank] = NodeStatus::DataEnd;

        WaitAll();
        return true; 
    }
};
}
}
}

extern Microsoft::MSR::CNTK::MPIWrapper *g_mpi;
