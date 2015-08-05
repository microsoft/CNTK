#pragma once
#include "mpi.h"
#pragma comment(lib, "msmpi.lib")
#include <string>
#include <array>
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {
struct mpifail : public std::string { mpifail (const std::string & what) : std::string (what) {} };
static int operator|| (int rc, const mpifail & what)
{
    //fprintf (stderr, "%s returned MPI status %d\n", what.c_str(), rc); fflush (stderr);
    if (rc == MPI_SUCCESS)
        return rc;
    fprintf (stderr, "%s, MPI error %d\n", what.c_str(), rc); fflush (stderr);
    if (rc != MPI_ERR_INTERN)       // (special case: we use that code to indicate a missing msmpi.dll...)
    {
        char errbuf[MPI_MAX_ERROR_STRING + 1] = { 0 };
        int len;
        MPI_Error_string (rc, &errbuf[0], &len);
        fprintf (stderr, "%s, MPI error %d: %s\n", what.c_str(), rc, errbuf); fflush (stderr);
        // we abort through this, so that the MPI system gets the memo
        MPI_Abort (MPI_COMM_WORLD, rc);
        // TODO: or does that only signal an issue, and we should still terminate ourselves?
        // BUGBUG: We'd also need to Abort through the other sub-set communicator
    }
    throw std::runtime_error (what);
}

class MPIWrapper
{
    int ourrank;            // we are this process ...
    int mpinodes;           // ...out of this many
    size_t nodesinuse;      // actually using this many
    MPI_Comm currentcomm;   // MPI communicator that reflects the current subset selection
    int MPI_Init_DL()       // MPI_Init() with delay-loading the msmpi.dll (possibly causing a failure if missing; we want to catch that)
    {
        //Sleep (10000);                          // (not sure why this is needed, but Jasha added this and I moved it here)
#ifdef WIN32
        __try
#endif
        {
            int argc = 0; char**argv = NULL;    // TODO: do we need these?
            int provided;
            return MPI_Init_thread (&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
        }
#ifdef WIN32
        __except (1/*EXCEPTION_EXECUTE_HANDLER, see excpt.h--not using constant to avoid Windows header in here*/)
        {
            fprintf (stderr, "mpihelper: msmpi.dll missing\n");
            return MPI_ERR_INTERN;
        }
#endif
    }
public:
    MPIWrapper() : currentcomm (MPI_COMM_WORLD)
    {
        static bool inited = false;
        if (inited)
            throw std::logic_error ("MPIWrapper: this is a singleton class that can only be instantiated once per process");
        inited = true;                      // MPI must be initialized exactly once
        fprintf (stderr, "MPIWrapper: initializing MPI\n"); fflush (stderr);
        try
        {
            MPI_Init_DL() || mpifail ("mpiaggregator: MPI_Init");
            MPI_Comm_rank (MPI_COMM_WORLD, &ourrank);
            MPI_Comm_size (MPI_COMM_WORLD, &mpinodes);
        }
        catch (...)
        {
#define FAKEMPI
#ifndef FAKEMPI
            throw;
#else       // for debugging, we can simulate it without actually having MPI installed
            ourrank = 0;
            mpinodes = 1;
            fprintf (stderr, "mpihelper: MPI_Init failed; faking MPI mode on one node (for debugging purposes only)\n");
#endif
        }
        nodesinuse = mpinodes;
        requestnodes ("MPIWrapper");                     // by default we use all of them

        if (mpinodes > 1)
            fprintf (stderr, "mpihelper: we are cog %d in a gearbox of %d\n", (int) ourrank, (int) mpinodes);
        else
            fprintf (stderr, "mpihelper: only one MPI process: MPI operation will be boring\n");
         fflush (stderr);
         // do an initial handshake, for the fun of it
         ping ("mpihelper");
         // stagger the jobs just a little to get a sort-of deterministic order e.g. in GPU allocation when running on one machine
         ::Sleep ((DWORD) (500 * node()));  // continue 0.5 seconds apart
    }
    // Note: we don't clear the sub-communication here although we should, because in case of a crash, this prevents the EXE from terminating.
    // It's OK since this class is a singleton anyway that gets instantiated exactly once at program startup.
    ~MPIWrapper() { fprintf (stderr, "~MPIWrapper\n"); fflush (stderr); /*requestnodes ("~mpihelper");*//*clear sub-comm*/ MPI_Finalize(); }
    // ping each other
    void ping (const char * msg) const
    {
         //fprintf (stderr, "ping [%s]: entering\n", msg); fflush (stderr);
#undef USE2NDCOMM
#ifndef USE2NDCOMM
         if (nodes() != mpinodes)
         {
             fprintf (stderr, "ping [%s]: cannot be applied to subset (%d) of nodes, skipping\n", msg, (int) nodes()); fflush (stderr);
             return;
         }
#endif
         std::array<int,1> handshake;
         handshake[0] = 1;
         fprintf (stderr, "ping [%s]: %d nodes pinging each other\n", msg, (int) nodes()); fflush (stderr);
         allreduce (handshake);
         fprintf (stderr, "ping [%s]: all %d nodes responded\n", msg, handshake[0]); fflush (stderr);
         //fprintf (stderr, "ping [%s]: exiting\n", msg); fflush (stderr);
    }

    bool forceuseallnodes() const { return false; }  // enable to forbid using a subset of nodes, for testing purposes
    void requestnodes (const char * msg, size_t requestednodes = SIZE_MAX/*default: all*/)
    {
        //fprintf (stderr, "requestnodes [%s,%d]: entering\n", msg, (int) requestednodes); fflush (stderr);
        if (forceuseallnodes() && requestednodes < SIZE_MAX)
        {
            requestednodes = SIZE_MAX;
            fprintf (stderr, "requestnodes: being forced to always use all nodes despite not being optimal\n");
        }
        //fprintf (stderr, "requestnodes: currentcomm is initially %x\n", (int) currentcomm); fflush (stderr);
        //fprintf (stderr, "requestnodes: was asked to use %d out of %d MPI nodes\n", (int) requestednodes, mpinodes); fflush (stderr);
        ping ("requestnodes (before change)");
        // undo current split
#ifdef USE2NDCOMM
        if (currentcomm != MPI_COMM_WORLD/*no subset*/ && currentcomm != MPI_COMM_NULL/*idle nodes*/)
        {
            fprintf (stderr, "requestnodes: MPI_Comm_free %x\n", (int) currentcomm); fflush (stderr);
            MPI_Comm_free (&currentcomm) || mpifail ("requestnodes: MPI_Comm_free");    // will leave MPI_COMM_NULL here
        }
#endif
        currentcomm = MPI_COMM_WORLD;       // reset to MPI_COMM_WORLD
        //fprintf (stderr, "requestnodes: currentcomm is %x\n", (int) currentcomm); fflush (stderr);
        // create a new split (unless all nodes were requested)
        if (requestednodes < (size_t)mpinodes)
        {
#ifdef USE2NDCOMM
            fprintf (stderr, "requestnodes: MPI_Comm_split %d\n", (node() < requestednodes) ? 1 : MPI_UNDEFINED); fflush (stderr);
            MPI_Comm_split (communicator(), (node() < requestednodes) ? 1 : MPI_UNDEFINED, 0, &currentcomm) || mpifail ("requestnodes: MPI_Comm_split");
            fprintf (stderr, "requestnodes: MPI_Comm_split -> %x\n", (int) currentcomm); fflush (stderr);
#endif
        }
        else    // leave currentcomm as MPI_COMM_WORLD
            requestednodes = mpinodes;      // and clip to #nodes
        nodesinuse = requestednodes;
        fprintf (stderr, "requestnodes [%s]: using %d out of %d MPI nodes (%d requested); we (%d) are %s\n",
                 msg, nodesinuse, mpinodes, (int) requestednodes,
                 node(), isidle() ? "out (idle)" : "in (participating)");
        fflush (stderr);
        //fprintf (stderr, "requestnodes: currentcomm is %x, finally\n", (int) currentcomm); fflush (stderr);
        ping ("requestnodes (after change)");
        //fprintf (stderr, "requestnodes [%s,%d -> %d]: exiting\n", msg, (int) requestednodes, (int) nodes()); fflush (stderr);
    }
    // get the communicator that reflects the selected nodes
    MPI_Comm communicator() const { return currentcomm; }
    size_t nodes() const { return nodesinuse; }
    size_t node() const { return ourrank; }
    size_t ismainnode() const { return ourrank == 0; }          // we are the chosen one--do extra stuff like saving the model to disk
    bool isidle() const { return node() >= nodes(); }           // user had requested to not use this many nodes
    bool usingallnodes() const { return nodes() == mpinodes; }  // all nodes participate (used to check whether we can use MPI_Allreduce directly)
    size_t mainnode()const {return 0;}

    // -----------------------------------------------------------------------
    // data-exchange functions (wrappers around MPI functions)
    // -----------------------------------------------------------------------

    // helpers to determine the MPI_Datatype of a pointer
    static MPI_Datatype getdatatype (char *)   { return MPI_CHAR; }
    static MPI_Datatype getdatatype (int *)    { return MPI_INT; }
    static MPI_Datatype getdatatype (float *)  { return MPI_FLOAT; }
    static MPI_Datatype getdatatype (double *) { return MPI_DOUBLE; }
    static MPI_Datatype getdatatype (size_t *) { return sizeof (size_t) == 4 ? MPI_UNSIGNED : MPI_LONG_LONG_INT; }

    // allreduce of a vector
    template<typename VECTORLIKEOBJECT>
    void allreduce (VECTORLIKEOBJECT & accumulator) const
    {
        auto * dataptr = accumulator.data();
        size_t totalnumelements = accumulator.size();
        // use MPI to compute the sum over all elements in (dataptr, totalnumelements) and redistribute to all nodes
        //fprintf (stderr, "allreduce: all-reducing matrix with %d elements\n", (int) totalnumelements); fflush (stderr);
        //fprintf (stderr, "allreduce:MPI_Allreduce\n"); fflush (stderr);
        if (nodes() > 1 && communicator() != MPI_COMM_NULL)
            MPI_Allreduce (MPI_IN_PLACE, dataptr, (int) totalnumelements, getdatatype (dataptr), MPI_SUM, communicator()) || mpifail ("allreduce: MPI_Allreduce");
        //fprintf (stderr, "allreduce: all-reduce done\n"); fflush (stderr);
    }
    // allreduce of a scalar
    template<typename T>
    void allreducescalar (T & val)
    {
        struct scalarasvectorref_t { T * p; scalarasvectorref_t (T & r) : p(&r) {} T * data() const { return p; } size_t size() const { return 1; } } r (val); // wraps 'val' as a VECTORLIKEOBJECT
        allreduce (r);
    }

    // redistribute a vector from main node to all others
    template<typename VECTORLIKEOBJECT>
    void redistribute (VECTORLIKEOBJECT & data) const
    {
        ping ("redistribute");
        auto * dataptr = data.data();
        size_t totalnumelements = data.size();
        // use MPI to send over all elements from the main node
        fprintf (stderr, "redistribute: redistributing matrix with %d elements %s this node\n", (int) totalnumelements, ismainnode() ? "from" : "to"); fflush (stderr);
        MPI_Bcast (dataptr, (int) totalnumelements, getdatatype (dataptr), 0/*send from this node*/, communicator()) || mpifail ("redistribute: MPI_Bcast");
    }

    // redistribute a variable-length string
    void redistributestring (std::string & str) const
    {
        ping ("redistribute (string)");
        // first transmit the size of the string
        std::array<int,1> len;
        len[0] = (int) str.size();
        redistribute (len);
        // then the string --we transmit it as a char vector
        std::vector<char> buf (str.begin(), str.end()); // copy to a char vector
        buf.resize (len[0]);                            // this will keep the main node's string at correct length, while extending or shrinking others, which is OK because those get overwritten
        redistribute (buf);                             // exchange as a char vector
        str.assign (buf.begin(), buf.end());            // and convert back to string
    }

    // send a buffer to 'tonode' while receiving a buffer from 'fromnode'
    template<typename BUFFER1, typename BUFFER2>
    void sendrecv (const BUFFER1 & fetchbuffer, size_t tonode,
                   BUFFER2 & recvsubbuffer, size_t fromnode)
    {
        //fprintf (stderr, "@@sendrecv [%d]: sending %d bytes to %d while receiving %d bytes from %d\n", (int) node(), (int) fetchbuffer.size(), (int) tonode, (int) recvsubbuffer.size(), (int) fromnode); fflush (stderr);
        MPI_Sendrecv (const_cast<char*> (fetchbuffer.data())/*header file const bug*/, (int) fetchbuffer.size(),   MPI_CHAR, (int) tonode,   (int) (nodes()*nodes() + tonode),
                      recvsubbuffer.data(),                                            (int) recvsubbuffer.size(), MPI_CHAR, (int) fromnode, (int) (nodes()*nodes() + node()),
                      communicator(), MPI_STATUS_IGNORE) || mpifail ("sendrecv: MPI_Sendrecv");
    }


    //slave sending tag,  0:K-1
    //called by master, to send msg to slave
    size_t tagforsendmaster(int tonode)
    {
        if (!ismainnode()) throw std::runtime_error("can not call tagforsendmaster from slave");
        return mainnode() * nodes() + tonode;
    }

    //called by slave, for recieving msg from master
    size_t tagforrecievemaster()
    {
        if (ismainnode()) throw std::runtime_error("can not call tagforrecievemaster from master");
        return mainnode() * nodes() + node();
    }

    //called by slave, for slave sending to master, start from K*(k+1) 
    size_t tagforsendslave(int tonode)//tonode = master(0)
    {
        if (ismainnode()) throw std::runtime_error("can not call tagforsendslave from master");
        return (node()+1) * nodes() + tonode;
    }
    //called by master, from master to reciving msg from slave
    size_t tagforrecieveslave(int fromnode)
    {
        if (!ismainnode()) throw std::runtime_error("can not call tagforrecieveslave from slave");
        return (fromnode+1) * nodes() + node();
    }

    // asynchronous send and receive
    // Call this, then do other stuff, and then call sencrevbwait() to finish it off (you must call it).
    std::vector<MPI_Request> sreq;  // lazily grown
    std::vector<MPI_Request> rreq;
    MPI_Request * getrequest (std::vector<MPI_Request> & req, size_t handle)
    {
        //fprintf (stderr, "@@getrequest [%c]: %d\n", &req == &sreq ? 's' : 'r', handle); fflush (stderr);
        if (handle >= req.size())                       // grow the handle array
            req.resize (handle +1, MPI_REQUEST_NULL);
        //if (req[handle] != MPI_REQUEST_NULL)            // sanity check
        //    fprintf (stderr, "@@getrequest: orphaned async send or recv operation %d\n", handle); fflush (stderr);
        if (req[handle] != MPI_REQUEST_NULL)            // sanity check
            throw std::logic_error ("getrequest: orphaned async send or recv operation");
        return &req[handle];        // MPI functions want a pointer
    }

    template<typename BUFFER>
    void sendasync (const BUFFER & fetchbuffer, size_t tonode, size_t asynchandle)
    {
       // fprintf (stderr, "@@sendasync: %d bytes to %d with handle %d and tag %d\n", fetchbuffer.size(), tonode, asynchandle, (int) (asynchandle * nodes() + tonode)); fflush (stderr);
        MPI_Isend (const_cast<char*> (fetchbuffer.data())/*header file const bug*/, (int) fetchbuffer.size(),   MPI_CHAR, (int) tonode,
                   (int) (asynchandle * nodes() + tonode), communicator(), getrequest (sreq, asynchandle)) || mpifail ("sendrecv: MPI_Isend");
    }

    template<typename BUFFER>
    void sendasync (const BUFFER & fetchbuffer, size_t tonode, int tag, size_t asynchandle)
    {
        //fprintf (stderr, "@@sendasync: %d bytes to %d with handle %d and tag %d\n", fetchbuffer.size(), tonode, asynchandle, (int) (asynchandle * nodes() + tonode)); fflush (stderr);
        MPI_Isend (const_cast<char*> (fetchbuffer.data())/*header file const bug*/, (int) fetchbuffer.size(),   MPI_CHAR, (int) tonode,
                  tag, communicator(), getrequest (sreq, asynchandle)) || mpifail ("sendrecv: MPI_Isend");
    }

        template<typename BUFFER>
    void sendasync (const BUFFER & fetchbuffer, size_t tonode, int tag, std::vector<MPI_Request> & sreqs, size_t asynchandle)
    {
       // fprintf (stderr, "@@sendasync: %d bytes to %d with handle %d and tag %d\n", fetchbuffer.size(), tonode, asynchandle, (int) (asynchandle * nodes() + tonode)); fflush (stderr);
        MPI_Isend (const_cast<char*> (fetchbuffer.data())/*header file const bug*/, (int) fetchbuffer.size(),   MPI_CHAR, (int) tonode,
                  tag, communicator(), getrequest (sreqs, asynchandle)) || mpifail ("sendrecv: MPI_Isend");
    }

    template<typename BUFFER>
    void recvasync (BUFFER & recvsubbuffer, size_t fromnode, size_t asynchandle)
    {
        //fprintf (stderr, "@@recvasync: %d bytes from %d with handle %d and tag %d\n", recvsubbuffer.size(), fromnode, asynchandle, (int) (asynchandle * nodes() + node())); fflush (stderr);
        MPI_Irecv (recvsubbuffer.data(),                                            (int) recvsubbuffer.size(), MPI_CHAR, (int) fromnode,
                   (int) (asynchandle * nodes() + node()), communicator(), getrequest (rreq, asynchandle)) || mpifail ("sendrecv: MPI_Irecv");
    }

    template<typename BUFFER>
    void recvasync (BUFFER & recvsubbuffer, size_t fromnode, int tag, size_t asynchandle)
    {
        //fprintf (stderr, "@@recvasync: %d bytes from %d with handle %d and tag %d\n", recvsubbuffer.size(), fromnode, asynchandle, (int) (asynchandle * nodes() + node())); fflush (stderr);
        MPI_Irecv (recvsubbuffer.data(),                                            (int) recvsubbuffer.size(), MPI_CHAR, (int) fromnode,
                   tag, communicator(), getrequest (rreq, asynchandle)) || mpifail ("sendrecv: MPI_Irecv");
    }

    template<typename BUFFER>
    void recvasync (BUFFER & recvsubbuffer, size_t fromnode, int tag,  std::vector<MPI_Request>& rreqs, size_t asynchandle)
    {
        // fprintf (stderr, "@@recvasync: %d bytes from %d with handle %d and tag %d\n", recvsubbuffer.size(), fromnode, asynchandle, (int) (asynchandle * nodes() + node())); fflush (stderr);
        MPI_Irecv (recvsubbuffer.data(),                                            (int) recvsubbuffer.size(), MPI_CHAR, (int) fromnode,
                   tag, communicator(), getrequest (rreqs, asynchandle)) || mpifail ("sendrecv: MPI_Irecv");
    }


    void sendwaitall()          // wait for all pending send requests to complete
    {
        auto & req = sreq;
        //fprintf (stderr, "@@sendwaitall\n"); fflush (stderr);
        MPI_Waitall ((int) req.size(), req.data(), MPI_STATUSES_IGNORE) || mpifail ("sendwaitall: MPI_Waitall");
        //fprintf (stderr, "@@sendwaitall: done\n"); fflush (stderr);
    }

     void sendwaitall(std::vector<MPI_Request>& sreqs)  
     {
        //fprintf (stderr, "@@sendwaitall\n"); fflush (stderr);
        MPI_Waitall ((int) sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE) || mpifail ("sendwaitall: MPI_Waitall");
        //fprintf (stderr, "@@sendwaitall: done\n"); fflush (stderr);
     }


    void recievewaitall()          // wait for all pending send requests to complete
    {
        auto & req = rreq;
        //fprintf (stderr, "@@recievewaitall\n"); fflush (stderr);
        MPI_Waitall ((int) req.size(), req.data(), MPI_STATUSES_IGNORE) || mpifail ("recievewaitall: MPI_Waitall");
        //fprintf (stderr, "@@recievewaitall: done\n"); fflush (stderr);
    }

    void recievewaitall(std::vector<MPI_Request>& req)          // wait for all pending send requests to complete
    {
        //fprintf (stderr, "@@recievewaitall\n"); fflush (stderr);
        MPI_Waitall ((int) req.size(), req.data(), MPI_STATUSES_IGNORE) || mpifail ("recievewaitall: MPI_Waitall");
        //fprintf (stderr, "@@recievewaitall: done\n"); fflush (stderr);
    }

    void waitall() // wait for all ranks to reach here
    {
        MPI_Barrier(currentcomm) || mpifail ("waitall: MPI_Barrier");
    }

    bool recvwaitany (bool blocking, size_t & handle)   // get zero or one pending receive request
    {
        return recvwaitany(blocking, rreq, handle);
    }

    bool recvwaitany (bool blocking, std::vector<MPI_Request>& req, size_t & handle)   // get zero or one pending receive request
    {
        int i, f;
        if (blocking)   // blocking: last one will return MPI_UNDEFINED
            MPI_Waitany ((int) req.size(), req.data(), &i, MPI_STATUS_IGNORE) || mpifail ("recvwaitany: MPI_Waitany");
        else            // non-blocking: if none it will return MPI_UNDEFINED
            MPI_Testany ((int) req.size(), req.data(), &i, &f, MPI_STATUS_IGNORE) || mpifail ("recvwaitany: MPI_Testany");
        //fprintf (stderr, "@@recvwaitany [%sblocking]: got %d\n", blocking? "" : "non-", i); fflush (stderr);
        if (i == MPI_UNDEFINED)
            return false;
        handle = i;
        return true;
    }
};
}}}
