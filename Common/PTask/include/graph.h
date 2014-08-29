//--------------------------------------------------------------------------------------
// File: graph.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "primitive_types.h"
#include "Lockable.h"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <deque>

namespace PTask {

    static const int MAX_GRAPH_NAME = 256;
    static const int MAX_GRAPH_EVENT_NAME = 2*MAX_GRAPH_NAME;
    static const DWORD DEFAULT_RUNNING_TIMEOUT = INFINITE;
    static const DWORD DEFAULT_MONITOR_TIMEOUT = 2000;

    class Graph;
    class Accelerator;
    class Datablock;
    class DatablockTemplate;
    class Task;
    class Channel;
    class Port;
    class CompiledKernel;
    class BlockPool;
    class BlockPoolOwner;
    class GraphInputChannel;
    class GraphOutputChannel;
    class InternalChannel;
    class InitializerChannel;
    class MultiChannel;
    class GraphProfiler;
    class ScopedPoolManager;
    class Partitioner;

    typedef struct taskpathstats_t {
        Task * pTerminus;
        size_t uiOptimalLength;
        size_t uiOptimalPathCount;
        size_t uiMaxPathLength;
        size_t uiTotalPathCount;

        taskpathstats_t() :
            pTerminus(NULL),
            uiOptimalLength(MAXUINT64),
            uiOptimalPathCount(0),
            uiMaxPathLength(0),
            uiTotalPathCount(0) {}

        taskpathstats_t(Task * pTask) :
            pTerminus(pTask),
            uiOptimalLength(MAXUINT64),
            uiOptimalPathCount(0),
            uiMaxPathLength(0),
            uiTotalPathCount(0) {}

    } TASKPATHSTATS;

    typedef struct ptasker_thread_desc_base_t {
        Graph * pGraph;
        GRAPHSTATE * pGraphState;        
        HANDLE hRuntimeTerminateEvent;
        HANDLE hGraphRunningEvent;
        HANDLE hGraphStopEvent;
        HANDLE hGraphTeardownEvent;
        HANDLE hGraphRunnerThread;
        BOOL bIsPooledThread;
        UINT nThreadIdx;
        UINT nGraphRunnerThreadCount;
    } GRAPHRUNNERDESC;

    typedef struct ptasker_thread_desc_t : ptasker_thread_desc_base_t {
        Task * pTask;
    } TASKERTHREADDESC, *PTASKERTHREADDESC;

    typedef struct ptasker_thread_desc_st_t : ptasker_thread_desc_base_t {
        HANDLE hReadyQ;
        int nTasks;
        Task ** ppTasks;
    } TASKERTHREADDESC_ST, *PTASKERTHREADDESC_ST;

    class Graph : public Lockable {

        friend class XMLWriter;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates the graph. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   The new graph. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Graph * CreateGraph();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a graph. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="lpszName"> [in,out] If non-null, the name. </param>
        ///
        /// <returns>   The new graph. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Graph * CreateGraph(char * lpszName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static BOOL DestroyGraph(Graph * pGraph, BOOL bGCSweep=FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Require block pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
        /// <param name="nDataSize">        Size of the data. </param>
        /// <param name="nMetaSize">        Size of the meta. </param>
        /// <param name="nTemplateSize">    Size of the template. </param>
        /// <param name="nBlocks">          (Optional) The blocks. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        RequireBlockPool(
            __in DatablockTemplate * pTemplate,
            __in int                 nDataSize, 
            __in int                 nMetaSize, 
            __in int                 nTemplateSize,
            __in int                 nBlocks=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Require block pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="nDataSize">        Size of the data. </param>
        /// <param name="nMetaSize">        Size of the meta. </param>
        /// <param name="nTemplateSize">    Size of the template. </param>
        /// <param name="nBlocks">          (Optional) The blocks. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        RequireBlockPool(
            __in int                 nDataSize, 
            __in int                 nMetaSize, 
            __in int                 nTemplateSize,
            __in int                 nBlocks=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Require block pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="pTemplate">    [in,out] If non-null, the template. </param>
        /// <param name="nBlocks">      (Optional) The blocks. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        RequireBlockPool(
            __in DatablockTemplate * pTemplate,
            __in int                 nBlocks=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Request a pooled block. </summary>
        ///
        /// <remarks>   crossbac, 8/21/2013. </remarks>
        ///
        /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
        /// <param name="uiDataSize">       Size of the data. </param>
        /// <param name="uiMetaSize">       Size of the meta. </param>
        /// <param name="uiTemplateSize">   Size of the template. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock * 
        RequestPooledBlock(
            __in DatablockTemplate * pTemplate,
            __in UINT                uiDataSize,
            __in UINT                uiMetaSize,
            __in UINT                uiTemplateSize
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Puts the graph in the running state. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="bSingleThreaded">  (optional) single thread flag. If true, the runtime will use
        ///                                 a single thread to manage all tasks in the graph. Otherwise,
        ///                                 the runtime will use a thread per task. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void Run(BOOL bSingleThreaded = FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets a graph to its initial state. </summary>
        ///
        /// <remarks>   Crossbac, 5/2/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void Reset();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Stops the graph--this method is synchronous in that it will not
        ///             return until all outstanding dispatches have completed or have
        ///             been moved to the scheduler's deferred queue. </summary>
        ///             
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void Stop();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Teardown the graph and free all resources associated with it. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void Teardown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the current state of this graph. You must hold a lock on this graph
        ///             to call this function.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   The state. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual GRAPHSTATE GetState();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a task. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pKernel">          [in,out] If non-null, the kernel. </param>
        /// <param name="uiInputPortCount"> Number of input ports. </param>
        /// <param name="pvInputPorts">     [in,out] If non-null, the pv input ports. </param>
        /// <param name="uiOutputPorts">    The output ports. </param>
        /// <param name="pvOutputPorts">    [in,out] If non-null, the pv output ports. </param>
        /// <param name="lpszTaskName">     [in,out] If non-null, name of the task. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Task*
        AddTask(
            CompiledKernel *	pKernel,
            UINT				uiInputPortCount,
            Port**				pvInputPorts,
            UINT				uiOutputPorts,
            Port**				pvOutputPorts,
            char *				lpszTaskName
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a super task. A super task may have more than
        ///             one accelerator class, and consequently multiple compiled 
        ///             kernels. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="uiKernelCount">    the number of kernels. </param>
        /// <param name="ppKernels">        [in,out] If non-null, the kernels. </param>
        /// <param name="uiInputPortCount"> Number of input ports. </param>
        /// <param name="pvInputPorts">     [in,out] If non-null, the pv input ports. </param>
        /// <param name="uiOutputPorts">    The output ports. </param>
        /// <param name="pvOutputPorts">    [in,out] If non-null, the pv output ports. </param>
        /// <param name="lpszTaskName">     [in,out] If non-null, name of the task. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Task*
        AddTask(
            UINT                uiKernelCount,
            CompiledKernel **	ppKernels,
            UINT				uiInputPortCount,
            Port**				pvInputPorts,
            UINT				uiOutputPorts,
            Port**				pvOutputPorts,
            char *				lpszTaskName
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a task by name </summary>
        ///
        /// <remarks>   jcurrey, 5/8/2013. </remarks>
        ///
        /// <param name="lpszTaskName">  [in] Name of the task. </param>
        ///
        /// <returns>   null if it fails, else the task. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Task * GetTask(char * lpszTaskName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Make the graph mutable. Can fail if PTask::Runtime::*GraphMutabilityMode is
        ///             turned off, which it is by default. A graph is 'mutable' if previously configured
        ///             structures can be changed. (In an unmutable graph, structures are 'write-once').
        ///             For example, in a mutable graph, port bindings can be changed, which in an
        ///             unmutable graph, once a port is bound to an object such as a channel, it cannot
        ///             be unbound and bound to some other channel later. Currently, respect for the
        ///             mutability of substructures in a graph is somewhat spotty, as it has only been
        ///             tested to deal with mutable control propagation bindings.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/8/2014. </remarks>
        ///
        /// <param name="bMutable"> (Optional) true to make the graph mutable. </param>
        ///
        /// <returns>   true if the mode change succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        SetMutable(
            __in BOOL bMutable=TRUE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is mutable. A graph is 'mutable' if previously configured
        ///             structures can be changed. (In an unmutable graph, structures are 'write-once').
        ///             For example, in a mutable graph, port bindings can be changed, which in an
        ///             unmutable graph, once a port is bound to an object such as a channel, it cannot
        ///             be unbound and bound to some other channel later. Currently, respect for the
        ///             mutability of substructures in a graph is somewhat spotty, as it has only been
        ///             tested to deal with mutable control propagation bindings.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/8/2014. </remarks>
        ///
        /// <returns>   true if mutable, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsMutable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an input channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="port">                 [in] non-null, the port. </param>
        /// <param name="lpszChannelName">      (optional) [in] If non-null, name of the channel. </param>
        /// <param name="bSwitchChannel">       (optional) if true, make the channel switchable. </param>
        /// <param name="pDefaultTriggerPort">  (optional) [in,out] If non-null, the default trigger
        ///                                     port. </param>
        /// <param name="uiTriggerSignal">      (optional) the trigger signal. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual GraphInputChannel * 
        AddInputChannel(
            __in Port *        pPort, 
            __in char *        lpszChannelName=NULL, 
            __in BOOL          bSwitchChannel=FALSE,
            __in Port *        pDefaultTriggerPort=NULL,
            __in CONTROLSIGNAL luiTriggerSignal=DBCTLC_NONE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an initializer channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pPort">                [in] non-null, the port. </param>
        /// <param name="lpszChannelName">      (optional) [in] If non-null, name of the channel. </param>
        /// <param name="bSwitchChannel">       (optional) if true, make the channel switchable. </param>
        /// <param name="pDefaultTriggerPort">  (optional) [in,out] If non-null, the default trigger
        ///                                     port. </param>
        /// <param name="luiTriggerSignal">     (optional) the lui trigger signal. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual InitializerChannel * 
        AddInitializerChannel(
            __in Port *        pPort, 
            __in char *        lpszChannelName=NULL, 
            __in BOOL          bSwitchChannel=FALSE,
            __in Port *        pDefaultTriggerPort=NULL,
            __in CONTROLSIGNAL luiTriggerSignal=DBCTLC_NONE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an output channel to to the graph. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="port">             [in] non-null, the port. </param>
        /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
        /// <param name="pTriggerTask">     (optional) the trigger channel. </param>
        /// <param name="luiTriggerSignal"> (optional) the trigger signal. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual GraphOutputChannel * 
        AddOutputChannel(
            __in Port * port, 
            __in char * lpszChannelName=NULL,
            __in Task * pTriggerTask=NULL,
            __in CONTROLSIGNAL luiTriggerSignal=DBCTLC_NONE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an internal channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pSrc">             [in,out] If non-null, source for the. </param>
        /// <param name="pDst">             [in,out] If non-null, destination for the. </param>
        /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
        /// <param name="bSwitchChannel">   (optional) if true, make the channel switchable. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual InternalChannel * 
        AddInternalChannel(
            __in Port * pSrc, 
            __in Port * pDst, 
            __in char * lpszChannelName=NULL, 
            __in BOOL   bSwitchChannel=FALSE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a multi channel. </summary>
        ///
        /// <remarks>   crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ppDst">                [in] non-null, an array of destination ports to bind. </param>
        /// <param name="nDestPorts">           The number of destination ports in ppDst. </param>
        /// <param name="lpszChannelName">      (optional) [in] If non-null, name of the multi channel. </param>
        /// <param name="bSwitchChannel">       (optional) if true, make the channel switchable.
        ///                                     Currently applied uniformly to all bound ports, which is
        ///                                     probably not sufficiently flexible. The work around is to
        ///                                     add a multi channel to one of the ports and call
        ///                                     MultiChannel::CoalesceChannel on explicitly created
        ///                                     GraphInputChannel objects bound to the remaining ports.
        ///                                     Another workaround is to bug Chris to fix it, which might
        ///                                     be faster. </param>
        /// <param name="pDefaultTriggerPort">  (optional) [in,out] If non-null, the default trigger
        ///                                     port. </param>
        /// <param name="uiTriggerSignal">      (optional) the trigger signal. </param>
        ///
        /// <returns>   null if it fails, else a new multi-channel. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual MultiChannel * 
        AddMultiChannel(
            __in Port **       ppDst, 
            __in int           nDestPorts,
            __in char *        lpszChannelName=NULL, 
            __in BOOL          bSwitchChannel=FALSE,
            __in Port *        pDefaultTriggerPort=NULL,
            __in CONTROLSIGNAL luiTriggerSignal=DBCTLC_NONE
            );


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a channel by name </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
        ///
        /// <returns>   null if it fails, else the channel. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Channel * GetChannel(char * lpszChannelName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind descriptor port. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pDescribedPort">   [in,out] If non-null, the described port. </param>
        /// <param name="pDescriberPort">   [in,out] If non-null, the describer port. </param>
        /// <param name="func">             (optional) the func. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void BindDescriptorPort(Port * pDescribedPort, Port* pDescriberPort, DESCRIPTORFUNC func=DF_SIZE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind control port. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pController">      [in,out] If non-null, the controller. </param>
        /// <param name="pGatedPort">       [in,out] If non-null, the gated port. </param>
        /// <param name="bInitiallyOpen">   (optional) the initially open. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void BindControlPort(Port * pController, Port * pGatedPort, BOOL bInitiallyOpen=FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind control propagation port. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pInputPort">   [in,out] If non-null, the input port. </param>
        /// <param name="pOutputPort">  [in,out] If non-null, the output port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void BindControlPropagationPort(Port * pInputPort, Port * pOutputPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind iteration ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pInputPort">   [in,out] If non-null, the input port. </param>
        /// <param name="pOutputPort">  [in,out] If non-null, the output port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void BindIterationScope(Port * pMetaPort, Port * pScopedPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind control propagation port. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pInputPort">           [in,out] If non-null, the input port. </param>
        /// <param name="pControlledChannel">   [in,out] If non-null, the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void BindControlPropagationChannel(Port * pInputPort, Channel * pControlledChannel);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets affinity for the entire graph to a single accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="uiAcceleratorId">  Identifier for the accelerator. </param>
        /// <param name="affinityType">     Type of the affinity. </param>
        ///
        /// <returns>   PTRESULT--use PTSUCCESS()/PTFAILED() macros </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT SetAffinity(UINT uiAcceleratorId, AFFINITYTYPE affinityType);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets strict affinitized accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 3/27/2014. </remarks>
        ///
        /// <returns>   null if it fails, else the strict affinitized accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator * GetStrictAffinitizedAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Set the partitioning mode to use for this graph.
        ///     Should pass only one of the following values:
        ///       GRAPHPARTITIONINGMODE_NONE = 0:
        ///         The runtime will not partition graphs across multiple available accelerators.
        ///       GRAPHPARTITIONINGMODE_HINTED = 1:
        ///         The runtime will partition graphs across multiple available accelerators,
        ///         according to hints given explicitly by the application via PTask::SetSchedulerPartitionHint().
        ///       GRAPHPARTITIONINGMODE_HEURISTIC = 2:
        ///         The runtime will partition graphs across multiple available accelerators,
        ///         available accelerators, using a set of experimental heuristics.
        ///       AUTOPARTITIONMODE_OPTIMAL = 2:
        ///         The runtime will attempt to auto-partition graphs across multiple
        ///         available accelerators, using a graph cut algorithm that finds the min-cut.
        ///
        ///     The value cannot be changed after graph finalization (calling Run() on the graph).
        ///
        ///     The default is the value specified by PTask::Runtime::GetDefaultGraphPartitioningMode()
        ///     at the time the graph is created.
        /// </summary>
        ///
        /// <remarks>   jcurrey, 1/27/2014. </remarks>
        ///
        /// <param name="mode"> The graph partitioning mode. </param>
        ///
        ///-------------------------------------------------------------------------------------------------
    
        void SetPartitioningMode(int mode);
    
        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Get the partitioning mode to use for this graph.
        ///     Will return one of the following values:
        ///       GRAPHPARTITIONINGMODE_NONE = 0:
        ///         The runtime will not partition graphs across multiple available accelerators.
        ///       GRAPHPARTITIONINGMODE_HINTED = 1:
        ///         The runtime will partition graphs across multiple available accelerators,
        ///         according to hints given explicitly by the application via PTask::SetSchedulerPartitionHint().
        ///       GRAPHPARTITIONINGMODE_HEURISTIC = 2:
        ///         The runtime will partition graphs across multiple available accelerators,
        ///         available accelerators, using a set of experimental heuristics.
        ///       AUTOPARTITIONINGMODE_OPTIMAL = 2:
        ///         The runtime will attempt to auto-partition graphs across multiple
        ///         available accelerators, using a graph cut algorithm that finds the min-cut.
        ///
        ///     The default is the value specified by PTask::Runtime::GetDefaultGraphPartitioningMode()
        ///     at the time the graph is created.
        /// </summary>
        ///
        /// <remarks>   jcurrey, 1/27/2014. </remarks>
        ///
        /// <returns>   The graph partitioning mode. </returns>
        ///-------------------------------------------------------------------------------------------------
    
        int GetPartitioningMode();
    
        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Configure the optimal partitioner for this graph.
        ///
        ///     Currently the number of partitions must be set to 2.
        ///
        ///     The working directory defaults to "C:\temp" and the prefix to "ptask_optimal_partition", 
        ///     resulting in the following files being created:
        ///       C:\temp\ptask_optimal_partitioner.partition_input.txt
        ///       C:\temp\ptask_optimal_partitioner.partition_output.txt
        ///
        ///     The partitioning mode must be set to AUTOPARTITIONINGMODE_OPTIMAL before calling this method.
        /// </summary>
        ///
        /// <remarks>   jcurrey, 1/31/2014. </remarks>
        ///
        /// <param name="numPartitions"> The number of partitions to divide the graph into. </param>
        /// <param name="workingDir"> The directory in which the model and solution files should be written. </param>
        /// <param name="fileNamePrefix"> The prefix of the model and solution file names. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void
        ConfigureOptimalPartitioner(
            int numPartitions = 2,
            const char * workingDir = NULL,
            const char * fileNamePrefix = NULL
            );
    
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Set a partition of the graph explicitly. This is experimental code that
        ///             takes a partition provided as a vector of ints, created by an 
        ///             external tool. This is very brittle, (sensitive to the order in which
        ///             node identifiers are assigned) and needs a different API if we
        ///             find that this is a performance profitable approach. Currently,
        ///             the vector is essentially pasted out of a text file received from
        ///             Renato.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 11/21/2013. </remarks>
        ///
        /// <param name="vPartition">       [in,out] If non-null, the partition. </param>
        /// <param name="nPartitionHints">  The partition hints. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void
        SetExplicitPartition(
            __in int * vPartition,
            __in UINT nPartitionHints
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Records the fact that tasks in this graph have explicit scheduler partition hints. 
        ///             This simplifies the task of establishing a partition during graph finalization:
        ///             if there are partition hints in the graph, the finalizer elides heuristic partitioning
        ///             entirely and uses the per-task hints it encounters. </summary>
        ///
        /// <remarks>   crossbac, 6/28/2013. </remarks>
        ///
        /// <param name="bHintsConfigured"> The hints configured. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        SetHasSchedulerPartitionHints(
            __in BOOL bHintsConfigured=TRUE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if tasks in this graph have explicit scheduler partition hints. 
        ///             This simplifies the task of establishing a partition during graph finalization:
        ///             if there are partition hints in the graph, the finalizer elides heuristic partitioning
        ///             entirely and uses the per-task hints it encounters. </summary>
        ///
        /// <remarks>   crossbac, 6/28/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        GetHasSchedulerPartitionHints(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Traverse a (supposedly) quiescent graph looking for sticky blocks that might
        /// 			get reused when the graph becomes non-quiescent. </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ReleaseStickyBlocks();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Drain channels. </summary>
        ///
        /// <remarks>   Crossbac, 3/5/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DrainChannels();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Prime initializers. </summary>
        ///
        /// <remarks>   Crossbac, 3/5/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void PrimeInitializers();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes a trigger operation. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        /// <param name="luiCode">  The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        ExecuteTriggers(
            __in Port *         pPort, 
            __in CONTROLSIGNAL  luiCode
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes the global trigger operation. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        /// <param name="luiCode">  The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        ExecuteTriggers(
            __in Channel *       pChannel, 
            __in CONTROLSIGNAL   luiCode
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets port map adjacencies. </summary>
        ///
        /// <remarks>   Crossbac, 11/20/2013. </remarks>
        ///
        /// <param name="pTask">                    [in,out] If non-null, the task. </param>
        /// <param name="bTaskIsSrc">               The task is source. </param>
        /// <param name="pPorts">                   [in,out] If non-null, the ports. </param>
        /// <param name="pSet">                     [in,out] If non-null, the set. </param>
        /// <param name="bIgnoreExposedChannels">   The ignore exposed channels. </param>
        ///
        /// <returns>   The number of adjacencies induced by the given port map. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT
        GetPortMapAdjacencies(
            __in    Task * pTask,
            __in    BOOL bTaskIsSrc,
            __in    std::map<UINT, Port*>* pPorts,
            __inout std::set<Task*>* pSet,
            __in    BOOL bIgnoreExposedChannels
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a map from task to the channels that express adjacency according to the flag
        ///             parameters. Exposed channels never express true adjacency, but since exposed
        ///             channels always represent a potential transition across memory spaces, it may be
        ///             important to include them somehow in the weight model.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 11/20/2013. </remarks>
        ///
        /// <param name="pMap">                     [in,out] [in,out] If non-null, the map. </param>
        /// <param name="bIgnoreExposedChannels">   The ignore exposed channels. </param>
        ///
        /// <returns>   The forward channel count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT
        GetAdjacencyMap(
            __out std::map<Task*, std::set<Task*>*>* &pMap,
            __in BOOL bIgnoreExposedChannels
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Free adjacency map. </summary>
        ///
        /// <remarks>   Crossbac, 11/20/2013. </remarks>
        ///
        /// <param name="pMap"> [in,out] If non-null, the first parameter. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        FreeAdjacencyMap(
            __inout std::map<Task*, std::set<Task*>*>* pMap
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Writes a file that describes the vertex and edge weights in the graph to be used
        ///             as input to Renato et al's graph partitioning algorithm. Format described below.
        ///
        ///             Currently the weight for each vertex is taken from the partition hint associated with
        ///             the corresponding task (set via Task::SetSchedulerPartitionHint). Every task in the
        ///             graph must have a hint associated with it or this method will raise an error.
        ///
        ///             Currently the edge weight is the same for all edges, and defaults to 1. 
        ///
        ///
        ///             Experimental code. File format:
        ///             Assume you have a graph with n nodes and m edges. Nodes must have ids from 1 to n.
        ///             
        ///             The first line just specifies the graph size and the format (weights on vertices
        ///             and costs on edges): <n> <m> 11
        ///             (The 11 is magic sauce. This one really does go to 11.)
        ///             
        ///             This is followed by n lines, each describing the adjacency list of one vertex (in
        ///             order). Each line is of the form: <vertex_weight> <v1> <ew1> <v2> <ew2> ... <vk> <ewk>
        ///             
        ///             For example, if the v-th line is “7 17 3 15 4 3 8”, then you know that vertex v
        ///             has weight 7 and has three outgoing edges: (v,17), (v,15), and (v,3). These edges
        ///             cost 3, 4, and 8, respectively.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 11/20/2013. </remarks>
        ///
        /// <param name="filename">                 [in] Filename of the file to write the model to. </param>
        /// <param name="edgeWeight">               [in] Weight to apply to each edge. </param>
        ///
        /// <returns>   The number of vertices in the model. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT 
        WriteWeightedModel(
            __in char * filename,
            __in UINT edgeWeight = 1
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Writes a dot file. To compile the DOT file generated by this function, you will
        ///             need to download and install graphviz : 1. Download and install the msi from
        ///             http://www.graphviz.org/Download_windows.php 2. To compile the DOT file, in a
        ///             command prompt type the following
        ///                dot -Tpng &lt;DOT file&gt; -o &lt;Output PNG file&gt;
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="filename">             [in,out] If non-null, filename of the file. </param>
        /// <param name="drawPorts">            (optional) the draw ports. </param>
        /// <param name="bPresentationMode">    (optional) the presentation mode. </param>
        /// <param name="bShowSchedulerHints">  (Optional) the show scheduler hints. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        WriteDOTFile(
            __in char * filename, 
            __in BOOL   drawPorts=FALSE,
            __in BOOL   bPresentationMode=FALSE,
            __in BOOL   bShowSchedulerHints=FALSE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Writes a dot file. 
        ///             To compile the DOT file generated by this function, you will need to 
        ///             download and install graphviz :
        ///             1. Download and install the msi from http://www.graphviz.org/Download_windows.php
        ///             2. To compile the DOT file, in a command prompt type the following
        ///                dot -Tpng <DOT file> -o <Output PNG file>
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="filename">     [in,out] If non-null, filename of the file. </param>
        /// <param name="drawPorts">    (optional) the draw ports. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        WritePresentationDOTFile(
            __in char * filename, 
            __in BOOL drawPorts=FALSE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Writes a dot file color-coding any partitioning of the graph for 
        ///             mandatory or soft affinity (dependent or otherwise). 
        ///             To compile the DOT file generated by this function, you will need to 
        ///             download and install graphviz :
        ///             1. Download and install the msi from http://www.graphviz.org/Download_windows.php
        ///             2. To compile the DOT file, in a command prompt type the following
        ///                dot -Tpng <DOT file> -o <Output PNG file>
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="filename">     [in,out] If non-null, filename of the file. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        WritePartitionDOTFile(
            __in char * filename
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Writes a dot file, with colors and annotations chosen to facilitate debugging of
        ///             graphs which make no forward progess. To use this the caller needs to download
        ///             and install graphviz : 1. Download and install the msi from
        ///             http://www.graphviz.org/Download_windows.php 2. To compile the DOT file, in a
        ///             command prompt type the following
        ///                dot -Tpng &lt;DOT file&gt; -o &lt;Output PNG file&gt;
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="filename">             [in,out] If non-null, filename of the file. </param>
        /// <param name="vReadyTasks">          [in,out] If non-null, the ready tasks. </param>
        /// <param name="vReadyChannelMap">     [in,out] [in,out] If non-null, the ready channel map. </param>
        /// <param name="vBlockedChannelMap">   [in,out] [in,out] If non-null, the blocked channel map. </param>
        /// <param name="drawPorts">            (optional) the draw ports. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        WriteDiagnosticDOTFile(
            __in char * filename, 
            __in std::set<Task*>& vReadyTasks,
            __in std::map<Task*, std::set<Channel*>*>& vReadyChannelMap,
            __in std::map<Task*, std::set<Channel*>*>& vBlockedChannelMap,
            __in BOOL   drawPorts=FALSE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Writes a dot file, with colors and annotations chosen to facilitate debugging of
        ///             control signal propagation in graphs. To use this the caller needs to download
        ///             and install graphviz : 1. Download and install the msi from
        ///             http://www.graphviz.org/Download_windows.php 2. To compile the DOT file, in a
        ///             command prompt type the following
        ///                dot -Tpng &lt;DOT file&gt; -o &lt;Output PNG file&gt;
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="filename">             [in] non-null, filename of the file. </param>
        /// <param name="vReadyTasks">          [in] If non-null, a list of tasks that are known to be
        ///                                     ready to dispatch. If null, no assumptions are made about
        ///                                     which tasks are ready. </param>
        /// <param name="vReadyChannelMap">     [in] If non-null, the ready channel map. If null, no
        ///                                     assumptions are made about which channels are
        ///                                     ready/blocked. </param>
        /// <param name="vBlockedChannelMap">   [in] If non-null, the blocked channel map. If null, no
        ///                                     assumptions are made about which channels are
        ///                                     ready/blocked. </param>
        /// <param name="drawPorts">            (optional) the draw ports. </param>
        /// <param name="pHighlightTasks">      [in,out] (Optional) [in,out] (Optional) If non-null,
        ///                                     (Optional) the highlight tasks. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        WriteControlPropagationDiagnosticDOTFile(
            __in char * filename, 
            __in std::set<Task*>* vReadyTasks=NULL,
            __in std::map<Task*, std::set<Channel*>*>* vReadyChannelMap=NULL,
            __in std::map<Task*, std::set<Channel*>*>* vBlockedChannelMap=NULL,
            __in BOOL   drawPorts=FALSE,
            __in std::set<Task*>* pHighlightTasks=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   OBSOLETE!
        ///             Searches for paths from input tasks (those with exposed input channels)
        ///             to output tasks (those with exposed output channels). Returns a comprehensive
        ///             list. Since the primary use of this API is to find good candidate control
        ///             propagation paths, the bFilterUnpredicatedEdgeNodes tells the method whether or
        ///             not to include paths that terminate in tasks that have no non-trivial predication
        ///             on bound output channels. If bDiscardSuboptimalPaths is true, the output map will
        ///             not include paths with length greater than that of the shortest path for the
        ///             given input/output task. If bMapKeyedByTerminusTask is true, then the output map
        ///             is keyed by the endpoint of each path, otherwise by the input. If bFilterConstantEdgeNodes
        ///             is true, the input edge node detection will attempt to filter out exposed 
        ///             input channels that are setup (apparently) to be used as constants (bound to
        ///             ports that have no inout binding and the sticky property). 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="vInputTasks">                  [in,out] [in,out] If non-null, the input tasks. </param>
        /// <param name="vOutputTasks">                 [in,out] [in,out] If non-null, the output tasks. </param>
        /// <param name="vPrioritizedPaths">            [in,out] [in,out] If non-null, the paths. </param>
        /// <param name="bFilterUnpredicatedEdgeNodes"> (Optional) the filter unpredicated edge nodes. </param>
        /// <param name="bFilterConstantEdgeNodes">     (Optional) the filter constant edge nodes. </param>
        /// <param name="bDiscardSuboptimalPaths">      (Optional) the discard suboptimal paths. </param>
        /// <param name="bMapKeyedByTerminusTask">      (Optional) the map keyed by terminus task. </param>
        /// <param name="bVerbose">                     (Optional) the verbose. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        FindExposedPathTasksEx(
            __inout std::set<Task*>& vInputTasks,
            __inout std::set<Task*>& vOutputTasks,
            __inout std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>& vPrioritizedPaths,
            __in    BOOL bFilterUnpredicatedEdgeNodes=FALSE,
            __in    BOOL bFilterConstantEdgeNodes=FALSE,
            __in    BOOL bDiscardSuboptimalPaths=FALSE,
            __in    BOOL bMapKeyedByTerminusTask=TRUE,
            __in    BOOL bVerbose=FALSE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   OBSOLETE! Searches for paths from input tasks (those with exposed input channels)
        ///             to output tasks (those with exposed output channels). Returns a comprehensive
        ///             list. Since the primary use of this API is to find good candidate control
        ///             propagation paths, the bFilterUnpredicatedEdgeNodes tells the method whether or
        ///             not to include paths that terminate in tasks that have no non-trivial predication
        ///             on bound output channels. If bDiscardSuboptimalPaths is true, the output map will
        ///             not include paths with length greater than that of the shortest path for the
        ///             given input/output task. If bMapKeyedByTerminusTask is true, then the output map
        ///             is keyed by the endpoint of each path, otherwise by the input. If
        ///             bFilterConstantEdgeNodes is true, the input edge node detection will attempt to
        ///             filter out exposed input channels that are setup (apparently) to be used as
        ///             constants (bound to ports that have no inout binding and the sticky property).
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="vInputTasks">          [in,out] [in,out] If non-null, the input tasks. </param>
        /// <param name="vOutputTasks">         [in,out] [in,out] If non-null, the output tasks. </param>
        /// <param name="vPrioritizedPaths">    [in,out] [in,out] If non-null, the paths. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        FindControlSignalPaths(
            __inout std::set<Task*>& vInputTasks,
            __inout std::set<Task*>& vOutputTasks,
            __inout std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>& vPrioritizedPaths
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synthesize control propagation paths required to ensure that all graph objects
        ///             whose behavior may be predicated by the specified set of signals (bitwise-or'd
        ///             together)
        ///             are reachable along some control propagation path. This is heuristic,
        ///             experimental code--generally, specifying such paths is a task we leave to the
        ///             programmer. In many simple cases, we conjecture that we can synthesize the
        ///             required paths using well known graph traversal algorithms. However, this is
        ///             decidedly conjecture, the programmer is advised to use this API at his/her own
        ///             risk.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/3/2014. </remarks>
        ///
        /// <param name="luiControlSignals">    [in] Bit-wise or of all control signals of interest.
        ///                                     Technically, these can be collected automatically by
        ///                                     traversing the graph as well: if DBCTLC_NONE is specified, we
        ///                                     traverse the graph to examine predicated objects and collect
        ///                                     the set of "signals-of-interest". </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        SynthesizeControlPropagationPaths(
            __in CONTROLSIGNAL luiControlSignals=DBCTLC_NONE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for the super-set of all "control signals of interest" for a graph.  
        ///             A control signal is "of interest" if there exists an object within this graph
        ///             whose behavior is predicated in some way by the presence or absence of a given
        ///             signal. This function traverses the graph and returns the bit-wise OR of all such
        ///             signals.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <returns>   The bitwise OR of all found control signals of interest. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual CONTROLSIGNAL GetControlSignalsOfInterest();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   find the set of terminal tasks which must be reachable along some control
        ///             propagation path for the given set of control signals to correctly exercise
        ///             all predicates in graph for the given (bit-wise OR of) control signal(s).
        ///             If no control signal is specified, this API will first search the graph to
        ///             collect the set of all control signals of interest.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <param name="vTasks">               [in,out] [in,out] If non-null, the tasks. </param>
        /// <param name="luiSignalsOfInterest"> (Optional) the lui signals of interest. </param>
        ///
        /// <returns>   the number of terminals found. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT 
        GetControlSignalOutputTerminalsOfInterest(
            __inout std::set<Task*>& vOutputTasks, 
            __in    CONTROLSIGNAL luiSignalsOfInterest=DBCTLC_NONE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   find the set of terminal tasks which must be reachable along some control
        ///             propagation path for the given set of control signals to correctly exercise all
        ///             predicates in graph for the given (bit-wise OR of) control signal(s). If no
        ///             control signal is specified, this API will first search the graph to collect the
        ///             set of all control signals of interest.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <param name="vInputTasks">              [in,out] [in,out] If non-null, the tasks. </param>
        /// <param name="bFilterConstantEdgeNodes"> (Optional) the filter constant edge nodes. </param>
        ///
        /// <returns>   the number of terminals found. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT 
        GetControlSignalInputTerminalsOfInterest(
            __inout std::set<Task*>& vInputTerminals, 
            __in    BOOL bFilterConstantEdgeNodes=TRUE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   find the set of terminal tasks which must be reachable along some control
        ///             propagation path for the given set of control signals to correctly exercise all
        ///             predicates in graph for the given (bit-wise OR of) control signal(s). If no
        ///             control signal is specified, this API will first search the graph to collect the
        ///             set of all control signals of interest.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <param name="vInputTerminals">      [in,out] [in,out] If non-null, the tasks. </param>
        /// <param name="vOutputTerminals">     [in,out] [in,out] If non-null, the output terminals. </param>
        /// <param name="luiSignalsOfInterest"> (Optional) the lui signals of interest. </param>
        ///
        /// <returns>   the number of terminals found. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT 
        GetControlSignalTerminalsOfInterest(
            __inout std::set<Task*>& vInputTerminals,
            __inout std::set<Task*>& vOutputTerminals, 
            __in    CONTROLSIGNAL luiSignalsOfInterest=DBCTLC_NONE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synthesize a control propagation path that traverses the given list
        ///             of tasks. This was written as part of support for automatic inference/construction
        ///             of control propagaion paths, which is fundamentally heuristic. However,
        ///             if the programmer knows the path and is willing to trust the runtime to
        ///             make good choices about which ports and channels to use to synthesize a
        ///             path, it can be called from external code.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/3/2014. </remarks>
        ///
        /// <param name="vPath">    [in,out] [in,out] If non-null, full pathname of the file. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        SynthesizeControlPropagationPath(
            __in std::vector<Task*>& vPath
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synthesize one hop in a control propagation path. This was written as part of
        ///             support for automatic inference/construction of control propagaion paths, which
        ///             is fundamentally heuristic. Do not call from external code.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/3/2014. </remarks>
        ///
        /// <param name="pCurrentTask">         [in,out] [in,out] If non-null, full pathname of the file. </param>
        /// <param name="ppPredecessorChannel"> [in,out] On entry, the inbound channel.
        ///                                              On exit, outbound channel selected on pCurrentTask to
        ///                                              extend the control propagation path across this
        ///                                              taskNode. </param>
        /// <param name="pSuccessorTask">       [in,out] If non-null, the immediate successor task in the
        ///                                     path. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        SynthesizeControlPropagationHop(
            __in    Task *     pCurrentTask,
            __inout Channel ** ppPredecessorChannel,
            __in    Task *     pSuccessorTask
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets immediate successors. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="pTask">        [in,out] If non-null, the task. </param>
        /// <param name="vSuccessors">  [in,out] [in,out] If non-null, the successors. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        GetImmediateSuccessors(
            __in Task * pTask,
            __inout std::set<Task*>& vSuccessors
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets immediate successors. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="pTask">        [in,out] If non-null, the task. </param>
        /// <param name="vSuccessors">  [in,out] [in,out] If non-null, the successors. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        GetAllSuccessors(
            __in Task * pTask,
            __inout std::set<Task*>& vVisited,
            __inout std::set<Task*>& vSuccessors
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets immediate predecessors. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="pTask">        [in,out] If non-null, the task. </param>
        /// <param name="vSuccessors">  [in,out] [in,out] If non-null, the successors. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        GetImmediatePredecessors(
            __in Task * pTask,
            __inout std::set<Task*>& predecessors
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets immediate successors. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="pTask">        [in,out] If non-null, the task. </param>
        /// <param name="vSuccessors">  [in,out] [in,out] If non-null, the successors. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        GetAllPredecessors(
            __in Task * pTask,
            __inout std::set<Task*>& vVisited,
            __inout std::set<Task*>& vPredecessors
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finds all outbound paths from a given task to tasks on the "edge" of the graph. A
        ///             graph is on the edge if it has exposed output channels. An optional terminals of
        ///             interest parameter filters paths that terminate in edge task not in the specified
        ///             set of "interesting tasks".
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="pTask">                [in,out] If non-null, the task. </param>
        /// <param name="vVisited">             [in,out] [in,out] If non-null, the successors. </param>
        /// <param name="vPaths">               [in,out] [in,out] If non-null, the paths. </param>
        /// <param name="vTerminalsOfInterest"> [in,out] (Optional) If non-null, (Optional) the terminals
        ///                                     of interest. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        FindAllOutboundPaths(
            __in    Task * pTask,
            __inout std::set<Task*>& vVisited,
            __inout std::vector<std::vector<Task*>>& vPaths,
            __in    std::set<Task*>& vTerminalsOfInterest
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Print outbound paths. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2014. </remarks>
        ///
        /// <param name="ios">                  [in,out] The ios. </param>
        /// <param name="vPrioritizedPaths">    [in,out] [in,out] If non-null, the paths. </param>
        /// <param name="vPathStats">           [in,out] [in,out] If non-null, the path stats. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        PrintOutboundPaths(
            __inout std::ostream &ios,
            __in    std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>& vPrioritizedPaths,
            __in    std::map<Task*, TASKPATHSTATS>& vPathStats
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Prioritize outbound paths. Based on the vector of paths, create a map from Task
        ///             to paths prioritized to prefer shorter paths. The bMapKeyedByTerminusTask
        ///             controls whether the keys in the map are the terminus or the origin of the task.
        ///             If bDiscardSuboptimalPaths is true, then paths with length greater than the
        ///             shortest are discarded.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/2/2014. </remarks>
        ///
        /// <param name="vPaths">                   [in,out] [in,out] If non-null, the paths. </param>
        /// <param name="vPrioritizedPaths">        [in,out] [in,out] If non-null, the prioritized paths. </param>
        /// <param name="vPathStats">               [in,out] [in,out] If non-null, the path stats. </param>
        /// <param name="bDiscardSuboptimalPaths">  The discard suboptimal paths. </param>
        /// <param name="bMapKeyedByTerminusTask">  The map keyed by terminus task. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        PrioritizeOutboundPaths(
            __in    std::vector<std::vector<Task*>>& vPaths,
            __inout std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>& vPrioritizedPaths,
            __inout std::map<Task*, TASKPATHSTATS>& vPathStats,
            __in    BOOL bDiscardSuboptimalPaths,
            __in    BOOL bMapKeyedByTerminusTask
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check to see if the graph is well-formed. This is not an exhaustive check, but a
        ///             collection of obvious sanity checks. If the bFailOnWarning flag is set, then the
        ///             runtime will exit the process if it finds anything wrong with the graph.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="bVerbose">         (optional) the verbose. </param>
        /// <param name="bFailOnWarning">   (optional) fail on warning flag: if set, exit the process
        ///                                 when malformed graph elements are found. </param>
        ///
        /// <returns>   PTRESULT--use PTSUCCESS/PTFAILED macros: PTASK_OK:   the graph is well-formed
        ///             PTASK_ERR_GRAPH_MALFORMED: the graph is malformed in a way that cannot be
        ///                                        tolerated by the runtime. Or the the issue may be
        ///                                        tolerable but the user requested fail on warning.
        ///             PTASK_WARNING_GRAPH_MALFORMED: the graph is malformed in a way that can be
        ///                                        tolerated by the runtime.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT CheckGraphSemantics(BOOL bVerbose=TRUE,
                                             BOOL bFailOnWarning=FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notify the graph that an event has occurred that can affect the ready state of a
        ///             task. Whether or not the graph needs to respond depends on the thread-task
        ///             mapping policy for the graph. If the graph is using 1:1 mapping, then no action
        ///             is necessary: the graph runner procs will respond to the port status signals. In
        ///             all other modes, the graph runner threads share a queue of tasks that *may* be
        ///             ready. Those threads take from the front of that queue and attempt to dispatch.
        ///             Avoiding queuing for tasks that are not ready is consequently worth some effort,
        ///             but the ready check is expensive so we make conservate estimate instead. The
        ///             bReadyStateKnown flag provides a way for the caller to override that check. A
        ///             TRUE bReadyStateKnown flag means we know it can be dispatched, so enqueue it
        ///             without further ado.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/28/2013. </remarks>
        ///
        /// <param name="pTask">            [in,out] If non-null, the task. </param>
        /// <param name="bReadyStateKnown"> TRUE if the ready state is known, FALSE if
        ///                                 a check is required before enqueueing the task. 
        ///                                 </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SignalReadyStateChange(
            __in Task * pTask, 
            __in BOOL bReadyStateKnown
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets (a pointer to) the graph name. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the name. </returns>
        ///-------------------------------------------------------------------------------------------------

        const char * GetName();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is running. </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2013. </remarks>
        ///
        /// <returns>   true if running, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsRunning();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the graph running event object. </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2013. </remarks>
        ///
        /// <returns>   The graph running event. </returns>
        ///-------------------------------------------------------------------------------------------------

        HANDLE GetGraphRunningEvent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the graph running event object. </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2013. </remarks>
        ///
        /// <returns>   The graph running event. </returns>
        ///-------------------------------------------------------------------------------------------------

        HANDLE GetGraphStopEvent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets runtime pause event. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   The runtime pause event. </returns>
        ///-------------------------------------------------------------------------------------------------

        HANDLE GetRuntimePauseEvent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets terminate event. </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2013. </remarks>
        ///
        /// <returns>   The terminate event. </returns>
        ///-------------------------------------------------------------------------------------------------

        HANDLE GetRuntimeTerminateEvent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports the graph state: diagnostic tool for finding problems with lack of
        ///             forward progress.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <param name="iosW">                         [in,out] The ios. </param>
        /// <param name="lpszDiagnosticDOTFilePath">    [in,out] If non-null, full pathname of the
        ///                                             diagnostic dot file. </param>
        /// <param name="bForceGraphDisplay">           true to force graph display. </param>
        ///-------------------------------------------------------------------------------------------------

        void ReportGraphState(
            std::ostream &iosW,
            char * lpszDiagnosticDOTFilePath,
            BOOL bForceGraphDisplay
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pause and report graph state for all known graphs: another diagnostics tool.
        ///             Reset the graph running event, and set the probe graph event. This will cause the
        ///             monitor thread to dump the graph and diagnostics DOT file. The event
        ///             synchronization require to do this right (that is, to ensure the graph quiesces
        ///             before we dump)
        ///             is non-trivial; so we do it in the most unprincipled way possible; Sleep! TODO:
        ///             fix this so we can actually probe the graph from other processes.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void PauseAndReportGraphStates();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the dispatch for a particular task should be deferred based on the graph
        ///             state. This is a static method because we don't want the scheduler to have lock
        ///             task and graph data structures to decide if a task should be moved from the run
        ///             queue to the deferred queue. It is OK for this answer to be stale by the time the
        ///             scheduler acts on it, since it will only be true when a graph is trying to pause,
        ///             which is a state transition that typically involves some latency and some
        ///             blocking synchronization. The graph manager keeps a list of pausing graphs to
        ///             simplify this check.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the graph. </param>
        ///
        /// <returns>   true if queiscing, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL ShouldDeferDispatch(Task * pTask);        

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalize the structure of the graph before running it. 
        ///             This is an opportunity to complete any bindings that
        ///             could not be completed before all graph objects were 
        ///             present, sort any port traversal orders, preallocate block pools,
        ///             etc. 
        /// 		    </summary>
        ///
        /// <remarks>   Crossbac </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Finalize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Serialize the graph structure to a file such that it can be deserialized in another 
        ///             process and used as if the graph had been instantiated there directly with calls to 
        ///             the graph construction API.
        ///
        ///             Only the graph topology (Tasks, Ports, Channels etc) and static information about
        ///             those entities, including scheduling-related properties, are stored. Datablocks
        ///             and Task executions in flight are not recorded.
        ///             </summary>
        ///
        /// <remarks>   jcurrey, 5/5/2013. </remarks>
        ///
        /// <param name="filename">   The name of the file to serialize the graph to. </param>
        ///-------------------------------------------------------------------------------------------------

        void Serialize(const char * filename);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deserialize the graph structure from a file such that it can be used as if the 
        ///             graph had been instantiated there directly with calls to the graph construction API.
        ///
        ///             Only the graph topology (Tasks, Ports, Channels etc) and static information about
        ///             those entities, including scheduling-related properties, are stored. Datablocks
        ///             and Task executions in flight are not recorded.
        ///             </summary>
        ///
        /// <remarks>   jcurrey, 5/5/2013. </remarks>
        ///
        /// <param name="filename">   The name of the file to deserialize the graph from. </param>
        ///-------------------------------------------------------------------------------------------------

        void Deserialize(const char * filename);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is alive. A graph is 'alive' until it has 
        ///             reached any of the teardown states. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if alive, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsAlive();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is runnable but not running. A graph is only running if it is
        ///             in the RUNNABLE state.</summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if running, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsStopped();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is finalized. This graph is finalized if it
        ///             has called its "finalize" method and is in the runnble, running,
        ///             quiescing, or teardown related states.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if finalized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsFinalized();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is torn down. This graph is torn down if it
        ///             has called its "teardown" method. Lock required.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if finalized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsTorndown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is in the middle of a teardown operation. Lock required.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if finalized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsTearingDown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets quiescent event. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   The quiescent event. </returns>
        ///-------------------------------------------------------------------------------------------------

        HANDLE GetQuiescentEvent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        Graph();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        Graph(HANDLE hRuntimeTerminateEvent);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="lpszName"> [in] non-null, the name. </param>
        ///-------------------------------------------------------------------------------------------------

        Graph(HANDLE hRuntimeTerminateEvent, char * lpszName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets backend frameworks. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the backend frameworks. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::set<ACCELERATOR_CLASS> * GetBackendFrameworks();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pause and report graph state for this graph: another diagnostics tool. Reset the graph
        ///             running event, and set the probe graph event. This will cause the monitor
        ///             thread to dump the graph and diagnostics DOT file. The event synchronization
        ///             require to do this right (that is, to ensure the graph quiesces before we dump)
        ///             is non-trivial; so we do it in the most unprincipled way possible; Sleep!
        ///             TODO: fix this so we can actually probe the graph from other processes.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void PauseAndReportGraphState();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Stream insertion operator for graph state enum. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="os">       [in,out] The operating system. </param>
        /// <param name="eState">   The state. </param>
        ///
        /// <returns>   The shifted result. </returns>
        ///-------------------------------------------------------------------------------------------------

        friend std::ostream& operator<<(std::ostream& os, const GRAPHSTATE& eState);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets managed object. </summary>
        ///
        /// <remarks>   Crossbac, 7/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void SetManagedObject();


    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Graph();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Forcible teardown of the graph induced by runtime terminate before the
        ///             graph is actually torn-down/deleted. This is an important scenario because
        ///             in managed space, graphs are gc'd
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void ForceTeardown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   this is the work associated with a dtor, encapsulated so that we
        ///             can leave the object alive if the runtime exits before a GC deletes the
        ///             graph.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void Destroy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Internal API: stops this graph, expects lock held. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void __Stop();


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Add the given task thread to the pool of threads that are known to be awaiting
        ///             new potentially ready tasks to enter the ready queue.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///
        /// <param name="uiThreadPoolIdx">  Zero-based index of the thread pool. </param>
        ///
        /// <returns>   true if the thread should enter the wait state, false
        ///             if there is already work available to dequeue. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL EnqueueWaitingTaskThread(UINT uiThreadPoolIdx);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Remove the given task thread from the pool of threads that
        ///             are known to be awaiting new potentially ready tasks to enter
        ///             the ready queue, and which have been signalled to attempt to
        ///             dequeue something. When the oustanding signals list is empty
        ///             but the runq is non-empty, we need to signal another thread. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///
        /// <param name="uiThreadPoolIdx">  Zero-based index of the thread pool. </param>
        ///-------------------------------------------------------------------------------------------------

        void AcknowledgeReadyQSignal(UINT uiThreadPoolIdx);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies a dispatch attempt complete. </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        void NotifyDispatchAttemptComplete(Task* pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Raises the Tasks available signal if appropriate, resets
        ///             signals if appropriate: notify the graph runner thread(s) that
        ///             there is something on the ready q. In single-thread mode it is sufficient
        ///             to set a single event that the graph-runner proc waits on. In multi-threaded
        ///             mode, the implementation is more nuanced, since we don't want a large number
        ///             of threads to wake up and charge pell-mell to contend for the run queue. 
        ///             When we are using the run queue with multiple threads, we maintain a list
        ///             of waiting threads, and choose randomly amongst them which thread to wake up. 
        ///             That thread is resposible for dequeueing something from the run queue and removing
        ///             itself from the waiter list as efficiently as possible to minimize contention.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void SignalTasksAvailable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Raises the Tasks available signal if appropriate, resets
        ///             signals if appropriate: notify the graph runner thread(s) that
        ///             there is something on the ready q. In single-thread mode it is sufficient
        ///             to set a single event that the graph-runner proc waits on. In multi-threaded
        ///             mode, the implementation is more nuanced, since we don't want a large number
        ///             of threads to wake up and charge pell-mell to contend for the run queue. 
        ///             When we are using the run queue with multiple threads, we maintain a list
        ///             of waiting threads, and choose randomly amongst them which thread to wake up. 
        ///             That thread is resposible for dequeueing something from the run queue and removing
        ///             itself from the waiter list as efficiently as possible to minimize contention.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void SignalTaskQueueEmpty();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Starts the graph runner threads for the single-thread mode. </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void LaunchGraphRunnerThreadsST();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Starts the graph runner threads for the multi-thread mode. </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void LaunchGraphRunnerThreadsMT();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports the graph state: diagnostic tool for finding problems with lack of
        ///             forward progress.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <param name="ios">              [in,out] The ios. </param>
        /// <param name="szLabel">          [in,out] If non-null, the label. </param>
        /// <param name="pTask">            [in,out] If non-null, the task. </param>
        /// <param name="pPortMap">         [in,out] If non-null, the port map. </param>
        /// <param name="bIsInput">         true if this object is input. </param>
        /// <param name="vReadyChannels">   [in,out] [in,out] If non-null, the ready channels. </param>
        /// <param name="vBlockedChannels"> [in,out] [in,out] If non-null, the blocked channels. </param>
        ///-------------------------------------------------------------------------------------------------

        void ReportBoundChannelState(
            std::ostream &ios,
            char * szLabel,
            Task * pTask,
            std::map<UINT, Port*>* pPortMap,
            BOOL bIsInput,
            std::set<Channel*>& vReadyChannels,
            std::set<Channel*>& vBlockedChannels
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes a trigger operation. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        /// <param name="luiCode">  The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        ExecuteTrigger(
            __in Port *         pPort, 
            __in CONTROLSIGNAL  luiCode
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes a trigger operation. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        /// <param name="luiCode">  The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        ExecuteTrigger(
            __in Channel *      pChannel, 
            __in CONTROLSIGNAL  luiCode
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind trigger. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pPort">            [in,out] If non-null, the port. </param>
        /// <param name="pChannel">         [in,out] If non-null, the channel. </param>
        /// <param name="luiTriggerSignal"> (optional) the trigger signal. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        BindTrigger(
            __in Port *        pPort, 
            __in Channel *     pChannel, 
            __in CONTROLSIGNAL luiTriggerSignal
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind trigger. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pChannel">         [in,out] If non-null, the channel. </param>
        /// <param name="pTask">            [in,out] (optional) the trigger signal. </param>
        /// <param name="luiTriggerSignal"> The lui trigger signal. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        BindTrigger(
            __in Channel * pChannel, 
            __in Task * pTask, 
            __in CONTROLSIGNAL luiTriggerSignal
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is alive. A graph is 'alive' until it has 
        ///             reached any of the teardown states. No lock is required
        ///             for this protected version of the method. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if alive, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsAlive();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is running. A graph is only running if it is in the RUNNING
        ///             state.No lock is required for this protected version of the method.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if running, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsRunning();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is runnable but not running. A graph is only running if it is
        ///             in the RUNNABLE state.No lock is required for this protected version of the
        ///             method.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if running, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsStopped();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is finalized. This graph is finalized if it has called its
        ///             "finalize" method and is in the runnble, running, quiescing, or teardown related
        ///             states.No lock is required for this protected version of the method.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if finalized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsFinalized();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is torndown. This graph is finalized if it has called its
        ///             Teardown method. No lock is required for this protected version of the method.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if finalized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsTorndown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this graph is in the middle of a teardown operation. No Lock required.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   true if finalized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsTearingDown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates signal objects for the ready Q. How many we need and whether or not they
        ///             are manual reset events depends on some runtime and graph-level settings.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CreateReadyQueueSignals();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deletes the ready queue signal (event objects--what we actually create here
        ///             depends on a the threading mode and other runtime settings).
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///
        /// <param name="uiTaskThreads">    The task threads. </param>
        ///-------------------------------------------------------------------------------------------------

        void DeleteReadyQueueSignals(UINT uiTaskThreads);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Called by graph runner threads before entering their run loop. A graph 
		/// 			may be considered truly 'running' when all it's threads are up. </summary>
		///
		/// <remarks>	crossbac, 8/12/2013. </remarks>
		///-------------------------------------------------------------------------------------------------

		void SignalGraphRunnerThreadAlive();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Wait until all graph runner threads are alive. </summary>
		///
		/// <remarks>	crossbac, 8/12/2013. </remarks>
		///-------------------------------------------------------------------------------------------------

		void WaitGraphRunnerThreadsAlive();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Get any backend-specific per-thread initialization off the critical path. Called
        ///             by graph runner threads before they signal that they are alive.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 8/12/2013. </remarks>
        ///
        /// <param name="dwThreadId">           Identifier for the thread. </param>
        /// <param name="uiThreadRoleIndex">    The thread. </param>
        /// <param name="uiThreadRoleMax">      The thread role maximum. </param>
        /// <param name="bPooledThread">        True if the thread is a globally pooled thread. </param>
        ///-------------------------------------------------------------------------------------------------

		void 
        PrimeGraphRunnerThread(
            __in DWORD dwThreadId, 
            __in UINT uiThreadRoleIndex,
            __in UINT uiThreadRoleMax,
            __in BOOL bPooledThread
            );

        /// <summary> Handle of the graph running event 
        /// 		  This event is set when the graph enters the
        /// 		  running state--all dispatch threads wait on this
        /// 		  event at the top of the dispatch loop.
        /// 		  </summary>
        HANDLE m_hGraphRunningEvent;
        
        /// <summary> Handle of the graph stop event 
        /// 		  </summary>
        HANDLE m_hGraphStopEvent;      

        /// <summary> Handle of the graph teardown event 
        /// 		  </summary>
        HANDLE m_hGraphTeardownEvent;      

        /// <summary>   is the graph done tearing down? </summary>
        HANDLE m_hGraphTeardownComplete;

        /// <summary> Handle of the runtime terminate event. Dispatch threads
        /// 		  and push/pull callers wait on this event (in addition to
        /// 		  whatever event is semantically meaningful for their wait,
        /// 		  such as a port status event) when they block, allowing all
        /// 		  worker threads in the system to wake up and clean up when
        /// 		  Graph::Terminate is called.
        /// 		  </summary>
        HANDLE m_hRuntimeTerminateEvent;

        /// <summary>   Handle of the quiescent event for this graph. </summary>
        HANDLE m_hGraphQuiescentEvent;       

        /// <summary>   Handle for a thread that monitors the forward
        ///             progress of the graph. Watchdog timer proc.
        ///             </summary>
        HANDLE m_hMonitorProc;        

        /// <summary>   Handle of the probe graph event: diagnostics tool. </summary>
        HANDLE m_hProbeGraphEvent;

        /// <summary>   The block pool owners. </summary>
        std::vector<BlockPoolOwner*> m_vBlockPoolOwners;

        /// <summary> A list of handles for task 
        /// 		  dispatch threads. </summary>
        std::map<HANDLE, BOOL> m_lpvhTaskThreads;
        
        /// <summary> A map of all channels in the graph </summary>
        std::map<std::string, Channel*> m_vChannels;
        
        /// <summary> A map of all the tasks in the graph </summary>
        std::map<std::string, Task*> m_vTasks;
        
        /// <summary> A map of all ports in the graph </summary>
        std::map<Port*, Task*> m_pPortMap;
        
        /// <summary> The channel source map. Each entry maps
        /// 		  a channel to its source port map, if it  
        /// 		  does have a port connected to its source.
        /// 		  </summary>
        std::map<Channel*, Port*> m_pChannelSrcMap;
        
        /// <summary> The channel destination map. Each entry
        /// 		  maps a channel to its destination port, 
        /// 		  if the channel has a destination port.
        /// 		  </summary>
        std::map<Channel*, Port*> m_pChannelDstMap;

        /// <summary>   The triggered channel map. </summary>
        std::map<Port*, std::map<Channel*, CONTROLSIGNAL>*> m_pTriggeredChannelMap;

        /// <summary>   The triggered channel signal map. </summary>
        std::map<Port*, std::map<CONTROLSIGNAL, std::set<Channel*>*>*> m_pTriggerSignalMap;

        /// <summary>   The channel triggers. </summary>
        std::map<Channel*, std::map<CONTROLSIGNAL, std::set<Task*>*>*> m_pChannelTriggers;

        /// <summary>   The bitwise OR of all trigger control signals. </summary>
        CONTROLSIGNAL m_luiTriggerControlSignals;

        /// <summary> Name of the graph </summary>
        char * m_lpszGraphName;

        /// <summary>   The life-cycle state of this graph. See comments
        ///             at GRAPHSTATE declaration for state machine details.
        ///             </summary>
        GRAPHSTATE m_eState;

        /// <summary>   true if this graph is 'mutable'. See comments for
        ///             SetMutable and IsMutable member functions. </summary>
        BOOL m_bMutable;

        /// <summary>   true if runtime terminate forced us to teardown this graph
        ///             before any Stop/Teardown/dtor cycle was initiated as a normal
        ///             part of the graph lifecycle. Handling this correctly is
        ///             important since graph objects may be wrapped by managed objects
        ///             that are not guaranteed to be GC'd before we exit PTask. 
        ///             </summary>
        BOOL m_bForceTeardown;

        /// <summary>   true to forced teardown complete. </summary>
        BOOL m_bForcedTeardownComplete;

        /// <summary>   true to destroy complete. </summary>
        BOOL m_bDestroyComplete;

        /// <summary>   true to managed object. </summary>
        BOOL m_bManagedObject;

        /// <summary>   true if the graph has been "finalized". </summary>
        BOOL m_bFinalized;

        /// <summary>   The partitioning mode for this graph. </summary>
        GRAPHPARTITIONINGMODE m_ePartitioningMode;

        /// <summary>   true if an explicit partition of the graph has
        ///             been provided. </summary>
        BOOL m_bExplicitPartition;

        /// <summary>   The explicit partition. FIXME--this needs a real interface</summary>
        int * m_pExplicitPartition;

        /// <summary>   The number of explicit partitition elements. </summary>
        UINT  m_uiExplicitPartitionElems;

        /// <summary>   true if any tasks in this graph have scheduler partition hints. </summary>
        BOOL m_bHasSchedulerPartitionHints;
        
        /// <summary>   true if the graph has ever entered the running state. </summary>
        BOOL m_bEverRan;

        /// <summary>   true if the run time is in single-threaded mode. </summary>
        BOOL m_bSingleThreaded;

        /// <summary>   true if the graph needs to use the ready q to multiplex
        ///             more than one task onto a single graph-runner thread. </summary>
        BOOL m_bUseReadyQ;

		/// <summary>	true if graph runner threads must be 'primed'. Some backend runtimes
		/// 			(read: CUDA runtime API) do some per-thread initialization on the
		/// 			first API call that induces significant latency (order seconds). 
		/// 			To get this overhead off the critical path, if a graph contains tasks
		/// 			that use these APIs (or other APIs with similar properties) we 
		/// 			prime each thread on entry. Currently this is a gross-hack that is
		/// 			CUDA-specific: TODO, factor this out more cleanly!
		/// 			</summary>
		BOOL m_bMustPrimeGraphRunnerThreads;

        /// <summary>   true if cross runtime sharing checks required. 
        ///             We encapsulate devices as accelerators *per-runtime*
        ///             as accelerator objects, which means it's actually possible
        ///             for a DX accelerator to be unavailable because the same GPU
        ///             is being used through a CU Accelerator. This complicates
        ///             scheduling quite a bit and is not the common case, so
        ///             we assume we don't need those checks unless we encounter
        ///             a graph that requires them.
        ///             </summary>
        BOOL m_bCrossRuntimeSharingChecksRequired;

        /// <summary>   The backend frameworks used by the graph. Helps us
        ///             correctly set the cross runtime check member as we
        ///             add tasks to the graph. </summary>
        std::set<ACCELERATOR_CLASS> m_vBackendFrameworks;

        /// <summary>   When task port status events are signalled, they 
        ///             can also add themselves to this list, used in single
        ///             thread mode to avoid having to use kernel wait objects
        ///             when there are too many tasks. </summary>
        std::deque<Task*> m_vReadyQ;

        /// <summary>   When task port status events are signalled, they 
        ///             can add themselves to the ready queue (in single
        ///             thread). The m_vReadyQ member maintains the FIFO order,
        ///             while the ready set helps us avoid duplicate entries;
        ///             a task may be signalled multiple times before it becomes
        ///             ready for dispatch. 
        ///             </summary>
        std::set<Task*> m_vReadySet;

        /// <summary>   lock for the ready q. </summary>
        CRITICAL_SECTION m_csReadyQ;

        /// <summary>   event(s) to signal when something is added to the ready q. 
        ///             Size of the array is determined by the number of threads in
        ///             the graph runner proc pool.
        ///             </summary>
        HANDLE *        m_phReadyQueue;

        /// <summary>   The list of threads from the task thread pool that
        ///             are (ostensibly) idle. This member is used to select
        ///             a thread to signal when a task enters the ready queue
        ///             and there are multiple threads sharing the ready queue. 
        ///             </summary>
        std::set<UINT>  m_vWaitingThreads;

        /// <summary>   The list of threads from the task thread pool that
        ///             are (ostensibly) attempting to respond to a ready queue
        ///             event. 
        ///             </summary>
        std::set<UINT>  m_vOutstandingNotifications;

        /// <summary>   The list of tasks for which we currently have task
        ///             threads performing work. Since we enqueue tasks when their
        ///             port status signals are set, rather than when they are definitively
        ///             ready, enqueue attempts are possible while a dispatch is
        ///             in flight. When this happens, we need to remember it
        ///             and enqueue such tasks when the outstanding dispatch attempt
        ///             completes. </summary>
        std::set<Task*> m_vOutstandingTasks;

        /// <summary>   The list of tasks we deferred because they were already in flight for a dispatch
        ///             attempt when the enqueue occurred. Since we enqueue tasks when their port status
        ///             signals are set, rather than when they are definitively ready, enqueue attempts
        ///             are possible while a dispatch is in flight. When this happens, put the task in
        ///             the deferred list and raise another signal when the outstanding attempt completes.
        ///             </summary>
        std::set<Task*> m_vDeferredTasks;

		/// <summary>	The number of graph runner threads that have been created but have
		/// 			not yet executed to the ack point (allowing the run method to return). </summary>
		volatile UINT	m_uiNascentGraphRunnerThreads;

        /// <summary>   The outstanding execute threads. </summary>
        volatile UINT   m_uiOutstandingExecuteThreads;

        /// <summary>   The outstanding execute threads. </summary>
        volatile UINT   m_uiThreadsAwaitingRunnableGraph;

        /// <summary>   The ready q surplus is incremented every time
        ///             we attempt to signal the arrival of new work but
        ///             have no waiting threads. In such a case we must
        ///             remember to raise the signal later when threads
        ///             become available again. </summary>
        UINT            m_uiReadyQSurplus;

        /// <summary>   lock for waiting thread and outstanding dispatch lists.
        ///             </summary>
        CRITICAL_SECTION m_csWaitingThreads;

        /// <summary>   true if the graph has cycles. Used as a heuristic for 
        ///             scheduling. If a graph is large and has cycles, it is 
        ///             typically better for the scheduler to try to some partitioning
        ///             of the graph (for individual GPUs) up front than it is 
        ///             </summary>
        BOOL            m_bGraphHasCycles;

        /// <summary>   true if the graph is large. </summary>
        BOOL            m_bLargeGraph;

        /// <summary>   The graph profiler. </summary>
        GraphProfiler * m_pGraphProfiler;

        /// <summary>   true if this graph uses strict accelerator assignment (there is a 
        ///             single accelerator to which all its tasks should be bound). </summary>
        BOOL            m_bStrictAcceleratorAssignment;

        /// <summary>   true if this graph uses strict accelerator assignment (there is a
        ///             single accelerator to which all its tasks should be bound) *AND* 
        ///             that assignment was the result of user-API calls to set a uniform
        ///             affinity across all tasks. This flag is needed because the round-robin
        ///             policy and the graph-level affinity implementations share these flags. </summary>
        BOOL            m_bUserManagedStrictAssignment;

        /// <summary>   The affinitized accelerator if m_bStrictAcceleratorAssignment is true. </summary>
        UINT            m_uiAffinitizedAccelerator;

        /// <summary>   The strict affinity accelerator. </summary>
        Accelerator *   m_pStrictAffinityAccelerator;

        /// <summary>   The optimal graph partitioner associated with this graph. </summary>
        Partitioner *   m_pPartitioner;

        /// <summary>   The accumulate assignment counter. </summary>
        static UINT     m_uiAccAssignmentCounter;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializer shared by different constructors. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="lpszGraphName">    [in,out] If non-null, name of the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        void Initialize(HANDLE hRuntimeTerminateEvent, char * lpszGraphName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pops the ready q. </summary>
        ///
        /// <remarks>   Crossbac, 1/29/2013. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Task * PopReadyQ();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a super (multi-platform) task. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pKernel">      [in,out] If non-null, the kernel. </param>
        /// <param name="lpszTaskName"> [in,out] If non-null, name of the task. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Task * 
        CreateSuperTask(
            __in CompiledKernel **  ppKernels, 
            __in UINT               uiKernelCount, 
            __in char *             lpszTaskName
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a platform specific task. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pKernel">      [in,out] If non-null, the kernel. </param>
        /// <param name="lpszTaskName"> [in,out] If non-null, name of the task. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Task * CreatePlatformSpecificTask(CompiledKernel * pKernel, char * lpszTaskName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a platform specific task. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pKernel">      [in,out] If non-null, the kernel. </param>
        /// <param name="accClass">     The acc class. </param>
        /// <param name="lpszTaskName"> [in,out] If non-null, name of the task. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Task * CreatePlatformSpecificTask(CompiledKernel * pKernel, ACCELERATOR_CLASS accClass, char * lpszTaskName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Preallocate output blocks. For every block pool owner candidate in the graph,
        ///             create pool of datablocks, allowing the system to avoid memory allocation on
        ///             outputs for every ptask invocation. Selection of candidates is currently
        ///             heuristic.
        ///             
        ///             FIXME: TODO: refine these heuristics.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void AllocateBlockPools();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Release block pools. 
        /// 		    </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ReleaseBlockPools();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   PTask's data aware scheduling strategy is effective for regular graph structures,
        ///             but runs into problems particularly in the presence of large graphs with cycles.
        ///             The greedy  approach of scheduling whereever data is most recent fails to plan
        ///             ahead to ensure that data on back edges is in the same memory space as data on
        ///             forward edges (some level of global view is necessary to handle this), and a
        ///             brute force search for the most performant assignment is impractical due to the
        ///             size of the graph. In such cases we want to use a graph partitioning algorithm,
        ///             and schedule partitions on available accelerators according to min-cost cuts. Udi
        ///             and Renato are currently helping with this, but in the meantime, some simple
        ///             heuristics can be used to help guide the scheduler to approximate this kind of
        ///             partitioning approach. This is a temporary fix. The method checks whether the
        ///             graph is large and contains cycles, and if so, it use some heuristics to
        ///             find transition points that would likely be chosen by a graph partitioning
        ///             algorithm, based on the presence or absence of certain types of block pools.
        ///             </summary>
        ///                     
        /// <remarks>   Crossbac, 5/3/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Partition();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Partition the graph explicitly based on information provided in the
        ///             call to SetExplicitPartition
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 11/21/2013. </remarks>
        ///
        /// <param name="vPartition">       [in,out] If non-null, the partition. </param>
        /// <param name="nPartitionHints">  The partition hints. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void ExplicitPartition();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Optimal partition in process. </summary>
        ///
        /// <remarks>   Crossbac, 3/18/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void OptimalPartitionInProcess();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Heuristic partition. </summary>
        ///
        /// <remarks>   Crossbac, 11/21/2013. </remarks>
        ///
        /// <param name="eClass">   The class. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void HeuristicPartition(ACCELERATOR_CLASS eClass);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Round robin partition. </summary>
        ///
        /// <remarks>   Crossbac, 11/21/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void RoundRobinPartition();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets partition identifier. </summary>
        ///
        /// <remarks>   Crossbac, 11/21/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetPartitionId(
            __in Task * pTask,
            __in std::map<Task*, int>* pNodeNumberMap
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the dominant accelerator class of the graph. 
        ///             A class is dominant if all the non-host tasks in the graph have
        ///             that class, and all dependent ports in the graph also have that
        ///             class. This is used to help the partitioner figure out if it
        ///             can indeed attempt a static partition of the graph.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/18/2014. </remarks>
        ///
        /// <returns>   The dominant accelerator class. </returns>
        ///-------------------------------------------------------------------------------------------------

        ACCELERATOR_CLASS GetDominantAcceleratorClass();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Propagate channel consistency properties. If a channel has the
        ///             "wants most recent view" property set, and descriptor ports have been
        ///             bound to that channel or its destination port, then those associated
        ///             deferred channels also need to have the "want most recent view" property
        ///             to avoid having the descriptor blocks get out of sync with the blocks on the
        ///             channel being described. Since these channels are created automatically, 
        ///             the programmer should never really have to know they exist at all, and therefore
        ///             should not be held responsible for setting that property. Consequently, we do this
        ///             before the graph is put in the run state. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void PropagateChannelConsistencyProperties();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Graph runner proc. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pVoidCastGraph">   the graph object, typecast to void* </param>
        ///
        /// <returns>   DWORD: 0 on thread exit. </returns>
        ///-------------------------------------------------------------------------------------------------

        static DWORD WINAPI GraphMonitorProc(LPVOID pVoidCastGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Graph runner proc. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pVoidCastGraph">   the graph object, typecast to void* </param>
        ///
        /// <returns>   DWORD: 0 on thread exit. </returns>
        ///-------------------------------------------------------------------------------------------------

        static DWORD WINAPI GraphRunnerProc(LPVOID pVoidCastGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Graph runner proc single-threaded, with queue. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pVoidCastGraph">   the graph object, typecast to void*. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static DWORD WINAPI GraphRunnerProcSTQ(LPVOID pVoidCastGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Graph runner proc mt. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pVoidCastGraph">   the graph object, typecast to void*. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static DWORD WINAPI GraphRunnerProcMTQ(LPVOID pVoidCastGraph);

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets an expected dispatch count for every task in the graph for which there is a
        ///             corresponding entry in the provided invocation count map.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="pvInvocationCounts">   [in,out] If non-null, the pv invocation counts. </param>
        /// <param name="nScalar">              The scalar. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetExpectedDispatchCounts(std::map<std::string, UINT> * pvInvocationCounts, UINT nScalar);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of oustanding blocks queued in channels in the graph. If the
        ///             pvOutstanding parameter is non-null, fill in a map from channels to outstanding
        ///             block counts as well.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="pvOutstanding">    [in,out] If non-null, the pv outstanding. </param>
        ///
        /// <returns>   The oustanding block counts. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetOustandingBlockCounts(std::map<Channel*, UINT> * pvOutstanding);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find all ports with non-zero control codes. When a graph is stopped or in some
        ///             quiescent state, it should generally be the case that no active control codes are
        ///             left lingering: this kind of situation can lead to control codes associated with
        ///             a previous stream or iteration affecting control flow for subsequent ones, which
        ///             is both undesirable and extremely hard to debug.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CheckPortControlCodes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find all channels with non-zero control codes. When a graph is stopped or in some
        ///             quiescent state, it should generally be the case that no active control codes are
        ///             left lingering: this kind of situation can lead to control codes associated with
        ///             a previous stream or iteration affecting control flow for subsequent ones, which
        ///             is both undesirable and extremely hard to debug.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
        
        void CheckChannelControlCodes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check that block pools contain only datablocks with no control signals. </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CheckBlockPoolStates();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Traverse a (supposedly) quiescent graph looking for active control signals </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CheckControlSignals();


    protected:

        /// <summary>   The dispatch count map, used for expected dispatch countin. </summary>
        std::map<std::string, UINT> * m_vDispatchCounts;

        /// <summary>   Block pool manager for pools scoped to this graph. </summary>
        ScopedPoolManager * m_pScopedPoolManager;

        ScopedPoolManager * GetPoolManager();
        void                OnTaskThreadAlive();
        void                OnTaskThreadExit();
        void                OnTaskThreadBlockRunningGraph();
        void                OnTaskThreadWakeRunningGraph();
        void                OnTaskThreadBlockTasksAvailable();
        void                OnTaskThreadWakeTasksAvailable();
        void                OnTaskThreadDequeueAttempt();
        void                OnTaskThreadDequeueComplete(Task * pTask);
        void                OnTaskThreadDispatchAttempt();
        void                OnTaskThreadDispatchComplete(BOOL bSuccess);
        HANDLE              __stdcall LaunchThread(LPTHREAD_START_ROUTINE lpRoutine, GRAPHRUNNERDESC * lpTaskDesc);

        friend class Scheduler;
        friend class GraphProfiler;
    };

};
#endif