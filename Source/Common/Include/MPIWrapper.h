#pragma once

// TODO: make this discoverable/settable during compilation, not hardcoded here
#if !defined(HAS_OPENMPI)
#define HAS_OPENMPI 1
#endif

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
typedef void *MPI_Comm;
typedef enum _MPI_Datatype { MPI_CHAR, MPI_INT, MPI_FLOAT, MPI_DOUBLE, MPI_UNSIGNED, MPI_LONG_LONG_INT } MPI_Datatype;

#define MPI_COMM_WORLD        0
#define MPI_IN_PLACE          0
#define MPI_SUM               2

#define MPI_SUCCESS           0
#define MPI_STATUSES_IGNORE  -3
#define MPI_STATUS_IGNORE    -2
#define MPI_UNDEFINED        -1

#define MPI_MAX_ERROR_STRING  64

#define MPI_Finalize()                      MPI_SUCCESS
#define MPI_Wait(a, b)                      a,b,MPI_UNDEFINED
#define MPI_Waitany(a, b, c, d)             a,b,c,d,MPI_UNDEFINED
#define MPI_Waitall(a, b, c)                a,b,c,MPI_UNDEFINED
#define MPI_Isend(a, b, c, d, e, f, g)      a,b,c,d,e,f,MPI_UNDEFINED
#define MPI_Recv(a, b, c, d, e, f, g)       a,b,c,d,e,f,g,MPI_UNDEFINED
#define MPI_Irecv(a, b, c, d, e, f, g)      a,b,c,d,e,f,g,MPI_UNDEFINED
#define MPI_Iallreduce(a, b, c, d, e, f, g) a,b,c,d,e,f,g,MPI_UNDEFINED

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
    int m_myRank;
    int m_numMPINodes;
    size_t m_numNodesInUse;

    // MPI communicator that reflects the current subset selection
    MPI_Comm m_currentComm;

    static MPIWrapperPtr s_mpi;

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
    MPIWrapper();

    // Note that specifically, this function is such that it does not require
    // MPI initialization. Moreover, it can be used without actually loading any
    // MPI libs.
    // TODO: Once we move to dynamic loading for MPI libs on Linux, move it to utilities.
    static int GetTotalNumberOfMPINodes();

    // Note: we don't clear the sub-communication here although we should, because in case of a crash, this prevents the EXE from terminating.
    // It's OK since this class is a singleton anyway that gets instantiated exactly once at program startup.
    ~MPIWrapper();

private:
    void Ping(const char *msg) const;

    void RequestNodes(const char *msg, size_t requestednodes = SIZE_MAX /*default: all*/);

public:

    static MPIWrapperPtr GetInstance(bool create = false);

    static void DeleteInstance();

    MPI_Comm Communicator() const;
    size_t NumNodesInUse() const;
    size_t CurrentNodeRank() const;
    bool IsMainNode() const;
    bool IsIdle() const;
    bool UsingAllNodes() const;
    size_t MainNodeRank() const;

    // -----------------------------------------------------------------------
    // data-exchange functions (wrappers around MPI functions)
    // -----------------------------------------------------------------------

    // helpers to determine the MPI_Datatype of a pointer
    static MPI_Datatype GetDataType(char *);
    static MPI_Datatype GetDataType(int *);
    static MPI_Datatype GetDataType(float *);
    static MPI_Datatype GetDataType(double *);
    static MPI_Datatype GetDataType(size_t *);

    // allreduce of a vector
    template <typename VECTORLIKEOBJECT>
    void AllReduce(VECTORLIKEOBJECT &accumulator) const;

    // for raw pointer
    template <class ElemType>
    void AllReduce(ElemType *pData, size_t nData);

    template <class ElemType>
    void Bcast(ElemType *pData, size_t nData, size_t srcRank);

    // wait for all ranks to reach here
    void WaitAll();
};

}}}
