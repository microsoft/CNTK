//--------------------------------------------------------------------------------------
// File: task.h
//
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _TASK_H_
#define _TASK_H_

#include "primitive_types.h"
#include "ReferenceCounted.h"
#include "GeometryEstimator.h"
#include "accelerator.h"
#include <vector>
#include <map>
#include <set>

class CHighResolutionTimer;
class CSharedPerformanceTimer;

namespace PTask {

    static const UINT DEFAULT_DISPATCH_ITERATIONS = 1;
    static const DWORD PORT_STATUS_TIMEOUT = 500;
    static const DWORD DISPATCH_TIMEOUT = 500;
    static const int DEF_OS_PRIO = -1;
    static const int MIN_PTASK_PRIO = 0;
    static const int MAX_PTASK_PRIO = 10;
    static const int DEF_PTASK_PRIO = 5;
    static const double D0_WEIGHT = 0.5;
    static const double D1_WEIGHT = 0.35;
    static const double D2_WEIGHT = 0.15;

    class TaskProfile;
    class DispatchCounter;
    class Channel;
    class Datablock;
    class DatablockTemplate;
    class CompiledKernel;
    class GeometryEstimator;
    class AsyncContext;
    class Port;
    class PBuffer;

    class Task : public ReferenceCounted 
    {
    
    friend class GeometryEstimator;
    friend class TaskProfile;
    friend class DispatchCounter;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="hRuntimeTerminateEvt"> Handle of the terminate event. </param>
        /// <param name="hGraphTeardownEvent">  Handle of the stop event. </param>
        /// <param name="hGraphStopEvent">      Handle of the running event. </param>
        /// <param name="hGraphRunningEvent">   The graph running event. </param>
        /// <param name="pCompiledKernel">  The CompiledKernel associated with this task. </param>
        ///-------------------------------------------------------------------------------------------------

        Task(
            __in HANDLE hRuntimeTerminateEvt, 
            __in HANDLE hGraphTeardownEvent, 
            __in HANDLE hGraphStopEvent, 
            __in HANDLE hGraphRunningEvent,
            __in CompiledKernel * pCompiledKernel
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Task();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a task based on a compiled kernel. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pAccelerators">    [in] non-null, accelerators to compile for. </param>
        /// <param name="pKernel">          [in] non-null, the kernel. </param>
        ///
        /// <returns>   HRESULT, use windows SUCCEEDED/FAILED macros. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT Create(std::set<Accelerator*>& pAccelerators, CompiledKernel * pKernel )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an asynchronous context for the task. </summary>
        ///
        /// <remarks>   crossbac, 12/20/2011.
        ///             </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true for success, false if it fails</returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        CreateDispatchAsyncContext(
            __in Accelerator * pAccelerator
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind a port to a task variable at the given index. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        /// <param name="index">    Zero-based index of the variable to bind. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT BindPort(Port * pPort, PORTINDEX index);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind port. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="t">        The type of the port. </param>
        /// <param name="index">    Zero-based index of the variable to unbind. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port * UnbindPort(PORTTYPE t, PORTINDEX index);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind a port. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pPort">    [in] non-null, the port. </param>
        ///
        /// <returns>   HRESULT. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT UnbindPort(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes a task. Do not return until it has executed unless the
        ///             runtime or the graph is torn down. This Execute method should only 
        ///             be used when the runtime is using the 1:1 threads:tasks mapping
        ///             because the thread calling execute will block on the port status
        ///             event if the task is not ready. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="peGraphState">     [in] non-null, the system alive flag (for early return). </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void Execute(GRAPHSTATE * peGraphState);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Attempt to execute a task: if all it's inputs and outputs are ready, schedule it
        ///             and dispatch it. If the task is not ready, return without dispatching. This
        ///             method should be used for all cases when the 1:1 thread:task mapping is *not* in
        ///             use.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="peGraphState"> [in] non-null, the system alive flag (for early return). </param>
        ///
        /// <returns>   true if the task was ready and dispatched false otherwise. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL AttemptExecute(GRAPHSTATE * peGraphState);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dispatch the ptask. Default implementation calls platform-specific 
        /// 			members to set up bindings and perform the dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL Dispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sorts meta ports into an order that guarantees they can perform their operations
        ///             at dispatch time. Usually, the metaport visitation order does not matter. However,
        ///             when there are multiple allocator ports, bound to output ports that collaborate
        ///             in a descriptor port relationship, we cannot allocate the block until all the
        ///             metaports involved have determined the channel sizes for the block. Consequently,
        ///             in this situation, we must be sure to process all descriptor ports before non-
        ///             descriptor ports. This condition is sufficient to ensure that we can always
        ///             perform the block allocation at the described port once we arrive, and since the
        ///             order does not change once the graph is running, we perform this operation as
        ///             part of OnGraphComplete(), which is invoked just before the graph enters the
        ///             RUNNING state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 2/15/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL SortMetaPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   When the graph is complete, (indicated because Graph.Run was called), this method
        ///             is called on every task to allow tasks to perform and one-time initializations
        ///             that cannot be performed without knowing that the structure of the graph is now
        ///             static. The base class method performs some operations such as sorting metaports
        ///             and then calls the platform specific version, which is required of every 
        ///             subclass, since there is some diversity in what is required here. For example,
        ///             in CUDA we must compute parameter byte offsets; in OpenCL, we do nothing...etc.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/5/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void OnGraphComplete();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Executes the worker thread entry action. </summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///-------------------------------------------------------------------------------------------------

		virtual void OnWorkerThreadEntry(DWORD dwThreadId);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   When the graph is complete, (indicated because Graph.Run was called), this method
        ///             is called on every task to allow tasks to perform platform-specific one-time
        ///             initializations that cannot be performed without knowing that the structure of
        ///             the graph is now static. For example, computing parameter offset maps for
        ///             dispatch.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/5/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void PlatformSpecificOnGraphComplete()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initialize instrumentation. </summary>
        ///
        /// <remarks>   t-nailaf, 06/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void InitializeInstrumentation();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalize instrumentation (cleanup allocated resources, etc). </summary>
        ///
        /// <remarks>   t-nailaf, 06/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void FinalizeInstrumentation();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dispatch the ptask. Abstract. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificDispatch()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enqueue this ptask on the scheduler run-queue, indicating ready
        /// 			that all its inputs are available and it is ready to invoke. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL ScheduleOrEnqueue(BOOL& bQueued, BOOL bBypassQueue);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the constant port map. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the input port map. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual std::map<UINT, Port*> * GetConstantPortMap();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the meta port map. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the input port map. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual std::map<UINT, Port*> * GetMetaPortMap();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the input port map. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the input port map. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual std::map<UINT, Port*> * GetInputPortMap();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the output port map. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the output port map. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual std::map<UINT, Port*> * GetOutputPortMap();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets all inbound channels. </summary>
        ///
        /// <remarks>   crossbac, 7/3/2014. </remarks>
        ///
        /// <param name="vChannels">    [in,out] [in,out] If non-null, the channels. </param>
        ///
        /// <returns>   The inbound channels. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetInboundChannels(std::set<Channel*>& vChannels);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets outbound channels. </summary>
        ///
        /// <remarks>   crossbac, 7/3/2014. </remarks>
        ///
        /// <param name="vChannels">    [in,out] [in,out] If non-null, the channels. </param>
        ///
        /// <returns>   The outbound channels. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetOutboundChannels(std::set<Channel*>& vChannels);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the super-set of all "control signals of interest" for this graph object.  
        ///             A control signal is "of interest" if the behavior of this object is is predicated
        ///             in some way by the presence or absence of a given signal. This function returns
        ///             the bit-wise OR of all such signals.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <returns>   The bitwise OR of all found control signals of interest. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual CONTROLSIGNAL GetControlSignalsOfInterest();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the task name. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="s">    [in] non-null, the name of the task. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetTaskName(char * s);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the task name. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the task name. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual const char * GetTaskName();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Get the CompiledKernel associated with this task. </summary>
        ///
        /// <remarks>   jcurrey, 5/5/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual CompiledKernel * GetCompiledKernel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the compute geometry. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="tgx">  (optional) the thread group parameters, X dimension. </param>
        /// <param name="tgy">  (optional) the thread group parameters, Y dimension. </param>
        /// <param name="tgz">  (optional) the thread group parameters, Z dimension. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetComputeGeometry(int tgx=1, int tgy=1, int tgz=1 )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the block and grid sizes of task. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="grid">     The grid. </param>
        /// <param name="block">    The block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetBlockAndGridSize(PTASKDIM3 grid, PTASKDIM3 block)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a scheduler hint for this task. This is largely an experimental interface to
        ///             provide a way for the programmer to give the scheduler hints about how it can
        ///             deal with large graphs. Generally PTask hopes to find an optimal or near optimal
        ///             schedule (by which we mean mapping from tasks to compute resources) dynamically
        ///             by reacting to performance history approximations and dynamic data movement. With
        ///             large, or complex graphs, this does not always work for a few reasons:
        ///             
        ///             1. When PTask thinks a particular task should go on an accelerator because its  
        ///                inputs are already materialized there, and that accelerator is unavailable, it
        ///                blocks that task for a while (with a configurable threshold) but then
        ///                eventually schedules it someplace non-optimal to avoid starving the task. With
        ///                a large graph, time spent in the ready Q can be long, and the starvation
        ///                avoidance mechanism winds up causing compromises in locality that can
        ///                propagate deep into the graph.
        ///             
        ///             2. The experimental scheduler that does a brute force search of the assignment
        ///             space
        ///                has combinatorial complexity. Searching the entire space simply takes too long.
        ///             
        ///             3. The graph partitioning scheduler relies on the presence of block pools at
        ///                exposed channels as a heuristic to find partition points. This works, but only
        ///                if the programmer has actually bothered to configure pools there.
        ///             
        ///             While we are working on better graph partitioning algorithms, if the programmer
        ///             actually knows something that can help the scheduler, there ought to be a way to
        ///             let PTask know.
        ///             
        ///             PTask interprets the uiPartitionHint as a unique identifier for the partition the
        ///             graph should be mapped to (where all tasks in a partion should have mandatory
        ///             affinity for the same GPU), and interprets the user cookie pointer as an opaque
        ///             pointer to any additional data structure that a particular (modularized)
        ///             scheduler can use to maintain state.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/28/2013. </remarks>
        ///
        /// <param name="uiPartitionHint">      The scheduler hint. </param>
        /// <param name="lpvSchedulerCookie">   [in,out] If non-null, the lpv user cookie. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        SetSchedulerPartitionHint(
            __in UINT   uiPartitionHint,
            __in void * lpvSchedulerCookie=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets any scheduler hint for this task. This is largely an experimental interface
        ///             to provide a way for the programmer to give the scheduler hints about how it can
        ///             deal with large graphs. Generally PTask hopes to find an optimal or near optimal
        ///             schedule (by which we mean mapping from tasks to compute resources) dynamically
        ///             by reacting to performance history approximations and dynamic data movement. With
        ///             large, or complex graphs, this does not always work for a few reasons:
        ///             
        ///             1. When PTask thinks a particular task should go on an accelerator because its  
        ///                inputs are already materialized there, and that accelerator is unavailable, it
        ///                blocks that task for a while (with a configurable threshold) but then
        ///                eventually schedules it someplace non-optimal to avoid starving the task. With
        ///                a large graph, time spent in the ready Q can be long, and the starvation
        ///                avoidance mechanism winds up causing compromises in locality that can
        ///                propagate deep into the graph.
        ///             
        ///             2. The experimental scheduler that does a brute force search of the assignment
        ///             space
        ///                has combinatorial complexity. Searching the entire space simply takes too long.
        ///             
        ///             3. The graph partitioning scheduler relies on the presence of block pools at
        ///                exposed channels as a heuristic to find partition points. This works, but only
        ///                if the programmer has actually bothered to configure pools there.
        ///             
        ///             While we are working on better graph partitioning algorithms, if the programmer
        ///             actually knows something that can help the scheduler, there ought to be a way to
        ///             let PTask know.
        ///             
        ///             PTask interprets the uiPartitionHint as a unique identifier for the partition the
        ///             graph should be mapped to (where all tasks in a partion should have mandatory
        ///             affinity for the same GPU), and interprets the user cookie pointer as an opaque
        ///             pointer to any additional data structure that a particular (modularized)
        ///             scheduler can use to maintain state.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/28/2013. </remarks>
        ///
        /// <param name="lppvSchedulerCookie">  If non-null, on exit will contain a pointer to the user
        ///                                     cookie pointer provided in the corresponding setter. </param>
        ///
        /// <returns>   The scheduler partition hint. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT 
        GetSchedulerPartitionHint(
            __out void ** lppvSchedulerCookie=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the task has a scheduler partition hint. </summary>
        ///
        /// <remarks>   crossbac, 6/28/2013. </remarks>
        ///
        /// <returns>   true if scheduler partition hint, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        HasSchedulerPartitionHint(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the wait handle for this task's port status event. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   The port status event. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HANDLE GetPortStatusEvent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Raises a Port status change signal. The known ready flag is used to inform
        ///             queueing of the next dispatch attempt, based on the thread-to-task mapping in use
        ///             by the graph. If we are using 1:1 mapping, the flag is meaningless because each
        ///             task has it's own thread and does not impact other tasks by doing a ready-state
        ///             check that does not lead to a dispatch because the task is not really ready. In
        ///             all other modes, this signal is accompanies by placement on a queue for graph
        ///             runner procs to make dispatch attempts, so there is some motivation to avoid
        ///             queueing tasks before they are truly ready. Since that check is expensive and
        ///             requires complex synchronization, we make a conservative estimate at enqueue time
        ///             and skip queuing tasks for which we can be sure we will receive a subsequent
        ///             signal. When set, the known ready flag allows us to skip that check and is
        ///             computed at a time when we already hold all the locks we need.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="bKnownReady">  (optional) the known ready. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SignalPortStatusChange(BOOL bKnownReady=FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Raises a Dispatch signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void SignalDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Take care of book-keeping that must occur when the
        /// 			task enters the scheduler's ready queue. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void OnEnterReadyQueue();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Book-keeping that must occur when the task's 
        /// 			wait time on the ready queue must be updated. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void OnUpdateWait();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Book-keeping that occurs when the task
        /// 			leaves the ready queue and begins dispatch.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void OnBeginDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Book-keeping that occurs when the task
        /// 			completes dispatch.
        /// 			</summary>        
        /// 			
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void OnCompleteDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assign an accelerator to this task for dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetDispatchAccelerator(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the currently assigned dispatch accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the dispatch accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator * GetDispatchAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets task priorities. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="nPTaskPrio">   The task prio. </param>
        /// <param name="nProxyOSPrio"> (optional) the proxy operating system prio. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetPriority(int nPTaskPrio, int nProxyOSPrio=DEF_OS_PRIO);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the effective priority of the task. 
        ///             We can't preempt GPU work, making priority a very coarse tool. 
        ///             GOAL: we want effective priority to be high if:
        ///                 a) priority is high
        ///                 b) wait time is high
        ///                 c) gpu usage is low
        ///                 d) os proxy priority is high
        ///             So the approach is to perturb the base priority
        ///             positively in proportion to the difference between this task's
        ///             wait time and the average, and negatively in proportion to the
        ///             delta between this task's dispatch time and the average. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="nPrio">    The prio. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetEffectivePriority(int nPrio);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the task priority. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   The priority. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual int GetPriority();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the effective priority. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   The effective priority. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual int GetEffectivePriority();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Calculates the effective priority. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="dAvgDispatch">     The average dispatch. </param>
        /// <param name="dAvgCurrentWait">  The average current wait. </param>
        /// <param name="dAvgDecayedWaits"> The average decayed waits. </param>
        /// <param name="dAvgOSPrio">       The average operating system prio. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void ComputeEffectivePriority(double dAvgDispatch, double dAvgCurrentWait, double dAvgDecayedWaits, double dAvgOSPrio);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a usage stats. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pStats">   The stats. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void GetUsageStats(PPTASKUSAGESTATS pStats);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a dispatch thread. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="h">    Handle of the h. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetDispatchThread(HANDLE h);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the dispatch thread. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   The dispatch thread. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HANDLE GetDispatchThread();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets an accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="cls">  The cls. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetAcceleratorClass(ACCELERATOR_CLASS cls);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   The accelerator class. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual ACCELERATOR_CLASS GetAcceleratorClass();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets target port class. If there is a dependent binding for this
        ///             port, return the appropriate class. Otherwise, return the 
        ///             task accelerator class.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/3/2013. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        ///
        /// <returns>   The dependent port class. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual ACCELERATOR_CLASS GetPortTargetClass(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a dependent accelerator class. This interface provides a way for the
        ///             programmer to make a task's dependences on accelerators other than the dispatch
        ///             accelerator explicit, and is required for correctness if a task uses more than
        ///             one accelerator (for example, a HostTask using a GPU through CUDA+thrust). When
        ///             the scheduler is preparing a task for dispatch it must acquire, in addition to
        ///             the dispatch accelerator, an accelerator object of the appropriate class for
        ///             every entry in the m_vDependentAcceleratorClasses list before dispatch.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/9/2012. </remarks>
        ///
        /// <param name="cls">                          The accelerator class. </param>
        /// <param name="nRequiredInstances">           The number of instances of the class required. </param>
        /// <param name="bRequestDependentPSObjects">   True if platform-specific context objects should
        ///                                             be provided in the task entry point from platform-
        ///                                             specific dispatch. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        BindDependentAcceleratorClass(
            __in ACCELERATOR_CLASS cls, 
            __in int nRequiredInstances,
            __in BOOL bRequestDependentPSObjects
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number dependent accelerator classes. The scheduler must acquire an
        ///             accelerator object for each depenendent accelerator class entry in addition to
        ///             the dispatch accelerator before dispatching a task. See AddDependentAccelerator
        ///             class documentation for more details.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/9/2012. </remarks>
        ///
        /// <returns>   The number of dependent accelerator classes. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual int GetDependentBindingClassCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has dependent accelerator bindings. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2013. </remarks>
        ///
        /// <returns>   true if dependent accelerator bindings, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasDependentAcceleratorBindings();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if dependent bindings on this task expect to have
        ///             platform specific objects (such as device-id, streams, etc) passed
        ///             to it at dispatch time. </summary>
        ///
        /// <remarks>   Crossbac, 7/11/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL DependentBindingsRequirePSObjects();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the dependent accelerator class at the given index. The scheduler must
        ///             acquire an accelerator object for each depenendent accelerator class entry in
        ///             addition to the dispatch accelerator before dispatching a task. See
        ///             AddDependentAccelerator class documentation for more details.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/9/2012. </remarks>
        ///
        /// <param name="nIndex">       The index. </param>
        /// <param name="nRequired">    [in,out] The number of instances of that class required for
        ///                             dispatch. Currently required/assumed to be one. </param>
        ///
        /// <returns>   The dependent accelerator class. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual ACCELERATOR_CLASS 
        GetDependentAcceleratorClass(
            __in int nIndex, 
            __out int &nRequired 
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the assigned dependent accelerator for the given port. The scheduler must
        ///             acquire an accelerator object for each depenendent accelerator class entry in
        ///             addition to the dispatch accelerator before dispatching a task. See
        ///             AddDependentAccelerator class documentation for more details. The mapping to 
        ///             individual ports is resolved in ResolveDepedentPortBindings after the scheduler
        ///             has made the assignment, but before the Task::Bind* methods are called to bind
        ///             ins/outs for dispatch. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/9/2012. </remarks>
        ///
        /// <param name="idx">  The index. </param>
        ///
        /// <returns>   null if it fails, else the assigned dependent accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator * 
        GetAssignedDependentAccelerator(
            __in int nIndex
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the assigned dependent accelerator for the given port. The scheduler must
        ///             acquire an accelerator object for each depenendent accelerator class entry in
        ///             addition to the dispatch accelerator before dispatching a task. See
        ///             AddDependentAccelerator class documentation for more details. The mapping to
        ///             individual ports is resolved in ResolveDepedentPortBindings after the scheduler
        ///             has made the assignment, but before the Task::Bind* methods are called to bind
        ///             ins/outs for dispatch.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/9/2012. </remarks>
        ///
        /// <param name="pPort">    [in,out] The port. </param>
        ///
        /// <returns>   null if it fails, else the assigned dependent accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator * 
        GetAssignedDependentAccelerator(
            __in Port * pPort
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assign the dependent accelerator at the given index. The scheduler must acquire
        ///             an accelerator object for each depenendent accelerator class entry in addition to
        ///             the dispatch accelerator before dispatching a task. See AddDependentAccelerator
        ///             class documentation for more details.        
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/9/2012. </remarks>
        ///
        /// <param name="idx">          The index. </param>
        /// <param name="pAccelerator"> [in] the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        AssignDependentAccelerator(
            __in int idx, 
            __in Accelerator * pAccelerator
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resolve dependent port requirements. </summary>
        ///
        /// <remarks>   crossbac, 5/21/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        ResolveDependentPortRequirements();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases all dependent accelerators. </summary>
        ///
        /// <remarks>   crossbac, 5/9/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void ReleaseDependentAccelerators();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Provide an API for letting the user tell us if a task implementation
        ///             allocates memory through APIs that are not visible to PTask. The canonical
        ///             examples of this are HostTasks that use thrust or cublas: if there is 
        ///             high memory pressure, ptask can sometimes avert an OOM in user code by forcing
        ///             a gc sweep before attempting a dispatch. We can only do this if the programmer
        ///             lets us know that temporary space will be required.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <param name="bState">   true to state. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetUserCodeAllocatesMemory(BOOL bState);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if a task implementation
        ///             allocates memory through APIs that are not visible to PTask. The canonical
        ///             examples of this are HostTasks that use thrust or cublas: if there is 
        ///             high memory pressure, ptask can sometimes avert an OOM in user code by forcing
        ///             a gc sweep before attempting a dispatch. We can only do this if the programmer
        ///             lets us know that temporary space will be required.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <param name="bState">   true to state. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL UserCodeAllocatesMemory();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the usage timer. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the usage timer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual CHighResolutionTimer *GetUsageTimer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases any inflight datablocks after dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ReleaseInflightDatablocks();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resolve in out port bindings. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ResolveInOutPortBindings();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resolve meta port bindings. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ResolveMetaPortBindings();

#if 0
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets an async context for the proposed operation. Generally, we prefer to avoid
        ///             scheduling execution and data transfer in the same async context because that
        ///             forces serialization. Often it is necessary serialization (e.g. producer-consumer
        ///             relationship between task execution and data xfer), but often it is not (e.g.
        ///             subsequent task invocations using different data should not wait for transfers on
        ///             data produced by this execution). This method returns an async context in which
        ///             we should schedule a given operation.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in] non-null, the dispatch accelerator. </param>
        /// <param name="eAsyncOperationType">  Type of operation. </param>
        /// <param name="pTransferTarget">      (optional) [in,out] If non-null, the transfer target. </param>
        ///
        /// <returns>   null if it fails, else the dispatch context. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual AsyncContext * 
        GetOperationAsyncContext(
            __in Accelerator * pDispatchAccelerator,
            __in ASYNCHRONOUS_OPTYPE eAsyncOperationType,
            __in Accelerator * pTransferTarget=NULL
            );
#endif

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets an async context for the proposed operation. Generally, we prefer to avoid
        ///             scheduling execution and data transfer in the same async context because that
        ///             forces serialization. Often it is necessary serialization (e.g. producer-consumer
        ///             relationship between task execution and data xfer), but often it is not (e.g.
        ///             subsequent task invocations using different data should not wait for transfers on
        ///             data produced by this execution). This method returns an async context in which
        ///             we should schedule a given operation.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in] non-null, the dispatch accelerator. </param>
        /// <param name="eAsyncContextType">    Type of operation. </param>
        ///
        /// <returns>   null if it fails, else the dispatch context. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual AsyncContext * 
        GetOperationAsyncContext(
            __in Accelerator * pDispatchAccelerator,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a synchronization timestamp. 
        /// 			This value indicates the last time the task synchronized
        /// 			with the context of the given accelerator.</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] the accelerator. </param>
        ///
        /// <returns>   The synchronization timestamp. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetSynchronizationTimestamp(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a geometry estimator function. 
        /// 			This allows the user to specify a callback that maps 
        /// 			a set of Datablocks (the task inputs) to the block  
        /// 			and grid dimensions required for dispatch. This version
        /// 			is most flexible: common scenarios are supported by
        /// 			SetCanonicalGeometryEstimator.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="lpfn"> The lpfn. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetGeometryEstimator(LPFNGEOMETRYESTIMATOR lpfn);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Sets a canonical geometry estimator function. Choose from among a common set of
        /// 			functions that map input datablocks to block and grid dimensions required for
        /// 			dispatch. In the common case we can infer these values given the results of the
        /// 			SetGeometry call, but this function and its sibling (SetGeometryEstimator) allow
        /// 			the user to override the default behavior of the runtime. See documentation for
        /// 			GEOMETRYESTIMATORTYPE.
        /// 			</summary>
        ///
        /// <remarks>	Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="eEstimatorType"> 	The canonical estimator type. </param>
        /// <param name="nElemsPerThread">	The elems per thread. </param>
        /// <param name="nGroupSizeX">	  	The group size x coordinate. </param>
        /// <param name="nGroupSizeY">	  	The group size y coordinate. </param>
        /// <param name="nGroupSizeZ">	  	The group size z coordinate. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
		SetCanonicalGeometryEstimator(
			__in GEOMETRYESTIMATORTYPE eEstimatorType,
            __in int nElemsPerThread,
			__in int nGroupSizeX,
			__in int nGroupSizeY,
			__in int nGroupSizeZ
			);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the geometry estimator function. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   The user-defined geometry estimator function. </returns>
        ///-------------------------------------------------------------------------------------------------

        LPFNGEOMETRYESTIMATOR GetGeometryEstimator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Gets the current canonical geometry estimator. </summary>
        ///
        /// <remarks>	Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pnElemsPerThread">	If non-null, the elems per thread. </param>
        /// <param name="pnGroupSizeX">	   	[in,out] If non-null, the pn group size x coordinate. </param>
        /// <param name="pnGroupSizeY">	   	[in,out] If non-null, the pn group size y coordinate. </param>
        /// <param name="pnGroupSizeZ">	   	[in,out] If non-null, the pn group size z coordinate. </param>
        ///
        /// <returns>	The canonical geometry estimator. </returns>
        ///-------------------------------------------------------------------------------------------------

        GEOMETRYESTIMATORTYPE 
		GetCanonicalGeometryEstimator(
			__out int * pnElemsPerThread,
			__out int * pnGroupSizeX,
			__out int * pnGroupSizeY,
			__out int * pnGroupSizeZ
			);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this task has scheduling constraints induced either by
        ///             programmer-specified affities or because it produces unmarshallable
        ///             data. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <returns>   true if scheduling constraints, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasSchedulingConstraints();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this task produces unmigratable data on any output port. 
        /// 			An output port produces unmigratable data if the blocks have pointer
        /// 			values that are valid only in the context of the accelerator that
        /// 			initialized them. For example, if a datablock contains a hash-table
        /// 			whose buckets are allocated using device-side malloc.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL ProducesUnmigratableData();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if there are scheduling constraints for this task 
        ///             that are the result of calls to set affinity. Collecting all
        ///             constraints and choosing amongst available resources can be 
        ///             complex, and there are a few fast-path cases we want to handle
        ///             without requiring the scheduler to examine the task in detail. 
        ///             In particular, if there are mandatory assignments that can 
        ///             satisfied from cached pointers, we want to avoid repeated traversals
        ///             to figure out the same information. If there are constraints
        ///             to satisfy of any form, this method will return true. If there
        ///             are mandatory constraints that have already been discovered, 
        ///             on exit the ppTaskMandatory and ppDepMandatory pointers will be
        ///             set. On exit, a TRUE value for bAssignable indicates that the
        ///             scheduler can proceed directly to trying to acquire those mandatory
        ///             resources. Any other condition means the scheduler has to take the
        ///             full path to examine constraints and choose. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/28/2013. </remarks>
        ///
        /// <param name="ppTaskMandatory">  [in,out] If non-null, the task mandatory. </param>
        /// <param name="ppDepMandatory">   [in,out] If non-null, the dep mandatory. </param>
        /// <param name="bAssignable">      [in,out] The assignable. </param>
        ///
        /// <returns>   true if scheduling constraints, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        HasSchedulingConstraints(
            __out Accelerator ** ppTaskMandatory,
            __out Accelerator ** ppDepMandatory,
            __out BOOL& bAssignable
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Collect the scheduling constraints for a given ptask. Classify these constraints into
        ///     mandatory constraints and preferences, where the former must be honored, while the latter
        ///     can be handled with best-effort, but may not necessarily be honored.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="vMandatoryConstraints">    [out] non-null, mandatory constraints. </param>
        /// <param name="vPreferences">             [out] non-null, preferred accelerators. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        CollectSchedulingConstraints(
            __out std::set<Accelerator*> &vMandatoryConstraints, 
            __out std::set<Accelerator*> &vPreferences
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Collect the scheduling constraints for dependent ports on a given task. The task
        ///             itself may have constraints, but when tasks have depenendent ports, affinity may
        ///             be specified for those dependent bindings. This routine collects maps (port-
        ///             >accelerator) of preferred and mandatory scheduling constraints induced by those
        ///             dependent port bindings.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="vMandatoryConstraints">    [out] non-null, mandatory constraints. </param>
        /// <param name="vMandatorySet">            [out] the set the accelerators represented in
        ///                                         the output map vMandatoryConstraints
        ///                                         belongs to. </param>
        /// <param name="vPreferences">             [out] non-null, preferred accelerators. </param>
        /// <param name="vPreferenceSet">           [out] the set of accelerators represented in the
        ///                                         preference map.</param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        CollectDependentSchedulingConstraints(
            __out std::map<Port*, Accelerator*> &vMandatoryConstraintMap, 
            __out std::set<Accelerator*> &vMandatorySet,
            __out std::map<Port*, Accelerator*> &vPreferenceMap,
            __out std::set<Accelerator*> &vPreferenceSet
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   If affinity for a particular accelerator has been configured for this
        ///             task, either on a dependent port or otherwise, return it. This should
        ///             not be used by the scheduler, since affinity for more than one accelerator
        ///             is possible either because of soft affinity or because the task needs
        ///             multiple accelerator classes (e.g. a host task with affinity for a particular
        ///             CPU and a particular dependent GPU). This method is a helper for heuristics
        ///             and visualizers. Achtung. </summary>
        ///
        /// <remarks>   Crossbac, 11/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the first affinitized accelerator found. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * GetAffinitizedAcceleratorHint();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the mandatory dependent accelerator affinitized for this task,
        ///             or null if no such accelerator exists. </summary>
        ///
        /// <remarks>   Crossbac, 11/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the mandatory affinitized accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * GetMandatoryDependentAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'uiIndex' has mandatory dependent affinities. </summary>
        ///
        /// <remarks>   Crossbac, 11/15/2013. </remarks>
        ///
        /// <returns>   true if mandatory dependent affinities, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasMandatoryDependentAffinities();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets task-accelerator affinity. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] the accelerator. </param>
        /// <param name="affinityType"> Type of the affinity. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL SetAffinity(Accelerator* pAccelerator, AFFINITYTYPE affinityType);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets task-accelerator affinity. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="vAccelerators">    [in,out] non-null, the accelerators. </param>
        /// <param name="pvAffinityTypes">  [in,out] List of types of affinities. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL SetAffinity(std::vector<Accelerator*> &vAccelerators, std::vector<AFFINITYTYPE> &pvAffinityTypes);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets task-accelerator affinity for dependent ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] the accelerator. </param>
        /// <param name="affinityType"> Type of the affinity. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        SetDependentAffinity(
            Port * pPort, 
            Accelerator* pAccelerator, 
            AFFINITYTYPE affinityType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets task-accelerator affinity. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="vAccelerators">    [in,out] non-null, the accelerators. </param>
        /// <param name="pvAffinityTypes">  [in,out] List of types of affinities. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        SetDependentAffinity(
            Port * pPort, 
            std::vector<Accelerator*> &vAccelerators, 
            std::vector<AFFINITYTYPE> &pvAffinityTypes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a mandatory accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name=""> none</param>
        ///
        /// <returns>   null if it fails, else the mandatory accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * GetMandatoryAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the dispatch iteration count. This is the number of times the platform-
        ///             specific dispatch routine should invoke acclerator code for this dispatch. The
        ///             default is 1. If the Task has a MetaPort whose meta-function is of type
        ///             MF_SIMPLE_ITERATOR, then blocks received on that port will be interpreted as an
        ///             integer-valued iteration count. Dispatch will call GetIterationCount before
        ///             dispatching, and will clear the count (reset to 1) after.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///
        /// <returns>   The dispatch iteration count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetDispatchIterationCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the dispatch iteration count. This is the number of times the platform-
        ///             specific dispatch routine should invoke acclerator code for this dispatch. The
        ///             default is 1. If the Task has a MetaPort whose meta-function is of type
        ///             MF_SIMPLE_ITERATOR, then blocks received on that port will be interpreted as an
        ///             integer-valued iteration count. Dispatch will call GetIterationCount before
        ///             dispatching, and will clear the count (reset to 1) after.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///
        /// <param name="nIterations">  The iterations. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetDispatchIterationCount(UINT nIterations);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets the dispatch iteration count. This is the number of times the platform-
        ///             specific dispatch routine should invoke acclerator code for this dispatch. The
        ///             default is 1. If the Task has a MetaPort whose meta-function is of type
        ///             MF_SIMPLE_ITERATOR, then blocks received on that port will be interpreted as an
        ///             integer-valued iteration count. Dispatch will call GetIterationCount before
        ///             dispatching, and will clear the count (reset to 1) after.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ResetDispatchIterationCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has explicit meta channel bindings. </summary>
        ///
        /// <remarks>   crossbac, 4/26/2012. </remarks>
        ///
        /// <returns>   true if explicit meta channel bindings, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL GetHasImplicitMetaChannelBindings();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the has explicit meta channel bindings. </summary>
        ///
        /// <remarks>   crossbac, 4/26/2012. </remarks>
        ///
        /// <param name="b">    true to b. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetHasImplicitMetaChannelBindings(BOOL b);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   ToString utility. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="os">       [in,out] The operating system. </param>
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   The shifted result. </returns>
        ///-------------------------------------------------------------------------------------------------

        friend std::ostream& operator<<(std::ostream &os, Task * pTask); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Shuts down this object and frees any resources it is using. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Shutdown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the graph. </summary>
        ///
        /// <remarks>   crossbac, 4/26/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the graph. </returns>
        ///-------------------------------------------------------------------------------------------------

        Graph * GetGraph();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a graph. </summary>
        ///
        /// <remarks>   crossbac, 4/26/2012. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetGraph(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases sticky blocks in a quiescent graph. </summary>
        ///
        /// <remarks>   Crossbac, 3/4/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ReleaseStickyBlocks();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has dependent affinities. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   true if dependent affinities, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasDependentAffinities();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Returns the current dispatch count for the task. This is a diagnostic tool; we
        ///             return the value without taking a lock or requiring the task to be in a quiescent
        ///             state, so the return value has no consistency guarantees. Use
        ///             GetDispatchStatistics when you need a consistent view. Use this when you need a
        ///             reasonable estimate (e.g. when debugging!)
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <returns>   The current dispatch count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetCurrentDispatchCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets this object to its initial state. </summary>
        ///
        /// <remarks>   Crossbac, 5/2/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Reset();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this task is a "terminal". Generally, a task is a terminal if it has
        ///             exposed input or output channels. More precisely, a task is terminal if it is a
        ///             sink or source for any channels whose ability to produce or consume blocks is
        ///             under control of the user program. For input and output channels, this is an
        ///             obvious property. Initializer channels with predicates may behave just like an
        ///             input channel since their ability to produce blocks for a task is controlled by
        ///             predicates on blocks produced by the user program. Consequently, a task with
        ///             predicated initializers but no actual exposed input channels is also a terminal.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <returns>   true if terminal, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsTerminal();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the super-set of all "control signals of interest" for this graph object.  
        ///             A control signal is "of interest" if the behavior of this object is is predicated
        ///             in some way by the presence or absence of a given signal. This function returns
        ///             the bit-wise OR of all such signals.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <returns>   The bitwise OR of all found control signals of interest. </returns>
        ///-------------------------------------------------------------------------------------------------

        CONTROLSIGNAL 
        GetControlSignalsOfInterest(
            __in std::map<UINT, Port*>& pPortMap
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check if memory pressure warrants a pre-dispatch GC sweep. Currently this
        ///             only occurs for tasks that are explicitly marked as allocators of temporary
        ///             buffers in user code (ie. allocation that is invisible to ptask), and is an
        ///             attempt to avert OOMs that we cannot catch and remedy after the fact (OOM
        ///             in user-code will cause an exception that we can catch, but which cannot
        ///             recover from).
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void PreDispatchMemoryPressureCheck();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a platform-specific asynchronous context for the task. </summary>
        ///
        /// <remarks>   crossbac, 12/20/2011.
        ///             
        ///             This method is required of all subclasses, and abstracts the work associated with
        ///             managing whatever framework-level asynchrony abstractions are supported by the
        ///             backend target. For example, CUDA supports the "stream", while DirectX supports
        ///             an ID3D11ImmediateContext, OpenCL has command queues, and so on.
        ///             </remarks>
        ///
        /// <param name="pAccelerator">         [in,out] If non-null, the accelerator. </param>
        /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
        ///
        /// <returns>   null if it fails, else the new async context object. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual AsyncContext * 
        CreateAsyncContext(
            __in Accelerator * pAccelerator,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an asynchronous context for the task. </summary>
        ///
        /// <remarks>   crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] If non-null, the accelerator. </param>
        /// <param name="vContextMap">  [out] If non-null, the context map. </param>
        ///
        /// <returns>   true for success, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        __createDispatchAsyncContext(
            __in  Accelerator * pAccelerator
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force sealed. </summary>
        ///
        /// <remarks>   Crossbac, 9/28/2012. </remarks>
        ///
        /// <param name="pBlock">   If non-null, the block. </param>
        /// <param name="pPort">    If non-null, the port. </param>
        ///-------------------------------------------------------------------------------------------------

        void ForceSealed(Datablock * pBlock, Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize downstream views. </summary>
        ///
        /// <remarks>   Crossbac, 9/28/2012. </remarks>
        ///
        /// <param name="pBlock">   If non-null, the block. </param>
        /// <param name="pPort">    If non-null, the port. </param>
        ///-------------------------------------------------------------------------------------------------

        void MaterializeDownstreamViews(Datablock * pBlock, Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind meta ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL BindMetaPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the platform-specific work required to bind an
        /// 			individual input parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificBindInput(Port * pPort, 
                                               int ordinal, 
                                               UINT uiActualIndex, 
                                               PBuffer * pBuffer
                                               )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the platform-specific work required to bind an
        /// 			individual output parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificBindOutput(Port * pPort, 
                                                int ordinal, 
                                                UINT uiActualIndex, 
                                                PBuffer * pBuffer
                                                )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the platform-specific work required to bind an
        /// 			individual input parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificBindConstant(Port * pPort, 
                                                  int ordinal, 
                                                  UINT uiActualIndex, 
                                                  PBuffer * pBuffer
                                                  )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform specific finalize bindings. </summary>
        ///
        /// <remarks>   Crossbac, 1/5/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificFinalizeBindings()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind inputs. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL BindInputs(int &ordinal);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind outputs. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL BindOutputs(int &ordinal);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind constants. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL BindConstants(int &ordinal);   

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind accelerator executable. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL BindExecutable()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind shader. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindExecutable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind inputs. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindInputs();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind outputs. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindOutputs();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind constants. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindConstants();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind meta ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindMetaPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind a port from this task. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="map">      [in,out] non-null, the port map. </param>
        /// <param name="index">    Zero-based index of the variable to unbind. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Port *      UnbindPort(std::map<UINT, Port*> &map, PORTINDEX index);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind a port. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="map">      [in,out] non-null, the port map. </param>
        /// <param name="pPort">    [in] non-null, the port. </param>
        ///
        /// <returns>   HRESULT. </returns>
        ///-------------------------------------------------------------------------------------------------

        HRESULT     UnbindPort(std::map<UINT, Port*> &map, Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind an output port to this task. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        /// <param name="index">    Zero-based index of the variable to bind. </param>
        ///
        /// <returns>   HRESULT </returns>
        ///-------------------------------------------------------------------------------------------------

        HRESULT     BindOutputPort(Port * pPort, PORTINDEX index);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   If this task had recieved a control block, 
        /// 			return it. A control block is one with a
        /// 			non-zero control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   null if no control block has appeared on an input
        /// 			port, else the first such block to be received. 
        /// 			</returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock*  ReceivedControlBlock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Propagate control blocks. 
        /// 			On receiving a control block, propagate it to all outputs.
        /// 			OBSOLETE.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pCtlBlock">    [in,out] If non-null, the control block. </param>
        ///-------------------------------------------------------------------------------------------------

        void        PropagateControlBlocks(Datablock * pCtlBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Attempt a lightweight wait for outstanding async deps on incoming blocks. The
        ///             general strategy in PTask for dealing with RW dependences amongst tasks is to
        ///             detect sharing conflicts and use backend async APIs to explicitly order
        ///             conflicting operations on shared blocks. Where such APIs are present, maximal
        ///             asychrony is possible. However, there is a difficult case: tasks that require
        ///             only dispatch resources that support no explicit asynchrony managment, which
        ///             consume data produced by tasks using resources that do. Such cases fundamentally
        ///             require synchrony (the consumer cannot proceed until all conflicting outstanding
        ///             operations are known to have resolved), which in turn requires performance-
        ///             sapping calls that synchronize with devices and harm the performance of other
        ///             tasks.
        ///             
        ///             The challenge is to perform such synchronization with minimal impact--making
        ///             backend framework calls typically requires a lock on an accelerator instance that
        ///             encapsulate a device context, so while such waits should not impact the
        ///             performance of other tasks from a threading standpoint, locked accelerators cause
        ///             a defacto serialization because no other task can use the encapsulated device for
        ///             the duration of the wait. Async APIs differ by framework though--in particular,
        ///             it appears that the CUDA API we use to deal with these dependences
        ///             (cuEventSynchronize) does not require the device context that created the event
        ///             to be current, and is purportedly threadsafe, so it is plausible that we can
        ///             perform such waits on independent threads without acquiring locks that block
        ///             forward progress of other tasks.
        ///             
        ///             This method (called at the beginning of dispatch), if the runtime setting
        ///             PTask::Runtime::SetTaskDispatchLocklessIncomingDepWait(TRUE) has been called,
        ///             will attempt all such required synchronous waits without acquiring accelerator
        ///             locks *before* the normal protocol for pre-dispatch lock acquisition is run.
        ///             
        ///             This method is experimental--the CUDA APIs lack sufficient detail to predict
        ///             whether this is truly safe, so we're finding out empirically.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void        AttemptLocklessWaitOutstandingDeps();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Acquire locks on all resources that will need to be locked during task dispatch.
        ///             We acquire these locks all at once because we need to do it in order to prevent
        ///             deadlock, making it difficult to defer acquisition for some objects until the
        ///             time the resource is actually accessed. The following resources must be locked
        ///             for dispatch:
        ///             
        ///             1. The dispatch accelerator.   
        ///             2. Any datablocks that will be written by accelerator code during dispatch.  
        ///             3. Any datablocks that require migration--that is, any datablock for  
        ///                that will be bound as input or in/out whose most recent view exists on an
        ///                accelerator other than the dispatch accelerator.
        ///             
        ///             Note that this does not include datablocks whose view is either up to date, or
        ///             whose most recent view is in host memory. These blocks can be locked as
        ///             encountered and unlocked after updates are performed because we do not need any
        ///             locks other than the dispatch accelerator lock and the datablock lock to perform
        ///             these operations. And any concurrent writers attempting to write such blocks will
        ///             hold locks for the duration of dispatch, which will block updates of shared views.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void        AcquireDispatchResourceLocks();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Acquires any locks required to deal with incoming asynchronous dependences. 
        ///             If the block has outstanding dependences we may require an accelerator lock
        ///             that would not otherwise be required, and since we must acquire accelerator
        ///             locks up front, we have to check this condition in advance. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///
        /// <param name="pBlock">                   [in,out] If non-null, the block. </param>
        /// <param name="pDispatchAccelerator">     [in,out] If non-null, the dispatch accelerator. </param>
        /// <param name="pBlockTargetAccelerator">  [in,out] If non-null, the block target accelerator. </param>
        /// <param name="eProposedBlockOperation">  The proposed block operation. </param>
        ///-------------------------------------------------------------------------------------------------

        void        
        AcquireIncomingAsyncDependenceLocks(
            __in Datablock * pBlock,
            __in Accelerator * pDispatchAccelerator,
            __in Accelerator * pBlockTargetAccelerator,
            __in ASYNCHRONOUS_OPTYPE eProposedBlockOperation
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the dispatch resource locks aqcuired
        /// 			in the call to AcquireDispatchResourceLocks, but which were
        /// 			not already released during another phase of dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void        ReleaseDispatchResourceLocks();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Release any locks that were required *only* to deal with incoming asynchronous 
        ///             dependences. If we needed an accelarator lock to make a backend frame work call
        ///             to wait for outanding operations on a block to complete, but did not need it
        ///             for any other reason release it. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void        
        ReleaseIncomingAsyncDependenceLocks(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the actual data transfers associated with datablock migration. All such
        ///             blocks and the accelerator objects required to complete the migration should be
        ///             locked already. The work of migration is performed by the source accelerator.
        ///             Accelerator subclasses can implement Migrate if their underlying APIs have
        ///             support for transfer paths other than through host memory. If the source
        ///             accelerator subclass does not implement migrate, the default implemention
        ///             Accelerator::Migrate will handle transfering data through host memory.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        MigrateInputs();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Early release for exclusive dispatch access to pAccelerator. </summary>
        ///
        /// <remarks>   Crossbac, 1/5/2012. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void        ReleaseExclusiveDispatchAccess(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a block to the inflight list. </summary>
        ///
        /// <remarks>   Crossbac, 1/5/2012. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void        AddToInflightList(Port * pPort, Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Estimate dispatch dimensions. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void EstimateDispatchDimensions(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Estimate dispatch dimensions. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void EstimateDispatchDimensions();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   is this task ready for dispatch?. </summary>
        ///
        /// <remarks>   Crossbac, 2/2/2012. </remarks>
        ///
        /// <param name="pGraphState">  [in,out] If non-null, state of the graph. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------
    public:

        virtual BOOL IsReadyForDispatch(GRAPHSTATE * pGraphState);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   "Estimates" the readiness of this task for dispatch. When the runtime is using
        ///             thread pools (or single-thread mode), the graph runner threads share a ready
        ///             queue, and tasks are enqueued in response to port status signals. This means that
        ///             tasks will almost always visit the ready queue more than once on their way to
        ///             dispatch because we are effectively using the graph runner procs to both test
        ///             readiness and dispatch, which are activities that almost certainly have different
        ///             optimal policies for the order of the queue (Note this is in constrast to the 1:1
        ///             thread:task mapping where non-ready tasks *attempting* to dispatch cannot impede
        ///             progress for actually ready task by clogging the ready queue). The *obvious*
        ///             design response is to avoid actually queuing tasks if they are not ready. There
        ///             is problem with this approach: tasks are signaled as candidates for the ready
        ///             queue when events like dispatch or user-space block pushing happen, both of which
        ///             are activities that typically involve non-trivial locking. Checking a task for
        ///             readiness can also involve locking particularly if there is predication on any of
        ///             the channels or ports that can cause a "ready" conclusion to become stale.
        ///             
        ///             The solution: compromise. Estimate the readiness of tasks and don't queue them if
        ///             we can be sure they aren't ready based on that estimate. The strategy is: for
        ///             tasks whose ready state can be checked without acquiring locks (because only a
        ///             subsequent dispatch can cause a transition out of the ready state), return the
        ///             actual ready state by traversing the state of the ports and channels bound to the
        ///             task. If we cannot make a conclusion because locks would be required, ASSUME the
        ///             task is ready. We'll occasionally try to dispatch something that is not ready.
        ///             But most of the time, we'll avoid clogging the ready queues and forcing the graph
        ///             runner threads to waste work.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/20/2013. 
        ///             WARNING: THIS METHOD returns an UNRELIABLE ANSWER. It elides locks and may 
        ///             claim a task is ready when it is not. Don't use it without 
        ///             that caveat in mind.
        ///             </remarks>
        ///
        /// <returns>   FALSE if we can be *sure* the task is not ready without taking locks.
        ///             TRUE if it is ready, or if we can't be sure without locking graph structures.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL EstimateReadyStatus();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets control propagation paths. </summary>
        ///
        /// <remarks>   crossbac, 6/26/2014. </remarks>
        ///
        /// <param name="vTaskSignalSources">   [in,out] [in,out] If non-null, the task signal sources. </param>
        /// <param name="vPaths">               [in,out] On exit, a map from src port to the set of dest
        ///                                     ports reachable by control signals on the src port. </param>
        /// <param name="vSignalChannels">      [in,out] the set of channels that can carry outbound
        ///                                     control signals. </param>
        ///
        /// <returns>   The number of control propagation pairs. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetControlPropagationPaths(
            __inout std::map<Task*, std::set<Port*>>& vTaskSignalSources,
            __inout std::map<Port*, std::set<Port*>>& vPaths,
            __inout std::set<Channel*>& vSignalChannels
            );
        
    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pGraphState' is ready atomic. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="pGraphState">          [in,out] If non-null, state of the graph. </param>
        /// <param name="bExitWithLocksHeld">   true to exit with locks held. </param>
        ///
        /// <returns>   true if ready atomic, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsReadyAtomic(GRAPHSTATE * pGraphState, BOOL bExitWithLocksHeld);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pGraphState' is ready no lock. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="pGraphState">  [in,out] If non-null, state of the graph. </param>
        ///
        /// <returns>   true if ready no lock, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsReadyNoLock(GRAPHSTATE * pGraphState);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has predicated control flow. </summary>
        ///
        /// <remarks>   Crossbac, 2/2/2012. </remarks>
        ///
        /// <returns>   true if predicated control flow, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasPredicatedControlFlow();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the given port map has all ready ports. </summary>
        ///
        /// <remarks>   Crossbac, 2/2/2012. </remarks>
        ///
        /// <returns>   true if port map ready, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsPortMapReady(std::map<UINT, Port*>* pPortMap, GRAPHSTATE * pGraphState);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the given port map has all ready ports. No lock required. </summary>
        ///
        /// <remarks>   Crossbac, 2/2/2012. </remarks>
        ///
        /// <returns>   true if port map ready, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsPortMapReadyNoLock(std::map<UINT, Port*>* pPortMap);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assemble port lock sets. </summary>
        ///
        /// <remarks>   crossbac, 6/21/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void AssemblePortLockSets();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assemble i/o lock list. </summary>
        ///
        /// <remarks>   Crossbac, 2/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void AssembleIOLockList();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Schedules the given p graph state. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="pGraphState">  [in,out] If non-null, state of the graph. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator * Schedule(GRAPHSTATE * pGraphState);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Propagate outputs. </summary>
        ///
        /// <remarks>   Crossbac, 2/2/2012. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in] non-null, the dispatch accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void PropagateDataflow(Accelerator * pDispatchAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enter dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void MarkInflightDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Complete dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CompleteInflightDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resolve dependent accelerator port bindings. Before dispatch, tasks with
        ///             dependent accelerators, (e.g. host tasks using cuda) must be bound to both
        ///             dispatch and dependent accelerators. In order to bind the appropriate platform-
        ///             specific buffer instance before the platform-specific dispatch, we need to
        ///             resolve the resource assignments from the scheduler to the actual per-port
        ///             bindings. This method looks at the dependent assignments and distributes them
        ///             accordingly across the ports which have dependent bindings.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/21/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ResolveDependentPortBindings();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases any locks held on accelerators that were taken because the 
        ///             accelerator was a migration source. When the dispatch accelerator (or
        ///             dependent port accelerator) has a non-null async context, we can release
        ///             these locks after queing the copy, because we can make the target wait
        ///             for the source without re-acquiring the lock on the source accelerator. 
        ///             When there is a null async context, waiting for those copies requires us
        ///             to drain the source command queue for the given stream, so if we do not 
        ///             hold the lock until this is done, we risk deadlock for such cases. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/28/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ReleaseMigrationSourceLocks();

        /// <summary> The accelerator class of this task </summary>
        ACCELERATOR_CLASS       m_eAcceleratorClass;
        
        /// <summary> Name of the task </summary>
        char *					m_lpszTaskName;

        /// <summary>   The graph this task is a member of. </summary>
        Graph *                 m_pGraph;

        /// <summary>   The CompiledKernel associated with this task. </summary>
        CompiledKernel *        m_pCompiledKernel;
        
        /// <summary>   Handle of the port status event. The port status event is signaled every time a
        ///             block arrives on an input port or leaves  
        ///             on an output port. These events are the only ones that can cause a transition
        ///             from queued to ready state for the task, so execution threads wait on port status
        ///             events, rather than busy waiting on the ports directly.
        ///             </summary>
        HANDLE					m_hPortStatusEvent;

        /// <summary>   Handle of the dispatch event. When a task enqueues itself on the scheduler's
        ///             ready queue, it's execution thread then waits on the dispatch event. When the
        ///             scheduler has assigned a dispatch accelerator, it will signal this event,
        ///             allowing the execution thread to complete the dispatch.
        ///             </summary>
        HANDLE                  m_hDispatchEvent;

        /// <summary>   The dispatch accelerator. When the scheduler moves the task from ready to running,
        ///             it must assign an accelerator to use for the dispatch. The task's dispatch thread
        ///             will expect to wake up after waiting on the dispatch event with the dispatch
        ///             accelerator field set to the accelerator object to use.
        ///             </summary>
        Accelerator *           m_pDispatchAccelerator;

        /// <summary>   Handle of the dispatch thread. The dispatch thread is the thread assigned to
        ///             execute the ptask, and is either a static assignment, or a shared thread
        ///             depending on the runtime mode. See PTask::Runtime::Initialize for more detail on
        ///             dispatch thread management.
        ///             </summary>
        HANDLE                  m_hDispatchThread;

        /// <summary>   The dispatch number. Incremented every time we get scheduled. May differ
        /// 			from the *actual dispatch count* because we support a form of iteration
        /// 			which may dispatch the underlying kernel multiple times for each logical
        /// 			dispatch. A logical dispatch corresponds to a call to schedule.
        /// 			</summary>
        int                     m_nDispatchNumber;

        /// <summary>   The proxy operating system prio of the ptask. Proxy operating system prio is a
        ///             tool to avoid priority laundering (where a task runs at the priority of the
        ///             runtime, rather than the priority of the thread or application being served). The
        ///             proxy priority is the inherited priority of the task's client.
        ///             </summary>
        int                     m_nProxyOSPrio;
        
        /// <summary> The task's priority </summary>
        int                     m_nPriority;
        
        /// <summary> The task's effective priority, which is updated on
        /// 		  every scheduler transition while the task is on the
        /// 		  ready queue. See ComputeEffectivePriority for details.
        /// 		  </summary>
        int                     m_nEffectivePriority;
        
        /// <summary> The usage timer </summary>
        CHighResolutionTimer *  m_pUsageTimer;
        
        /// <summary> Time the task spent waiting on the ready queue
        /// 		  for dispatch on the *previous* dispatch, in 
        /// 		  milliseconds. 
        /// 		  </summary>
        double                  m_dLastWaitTime;
        
        /// <summary> Exponential moving average of wait times
        /// 		  (time spent on the ready queue) for all
        /// 		  previous dispatches of this task.
        /// 		  </summary>
        double                  m_dAverageWaitTime;
        
        /// <summary> Latency of the task's last dispatch, in milliseconds. </summary>
        double                  m_dLastDispatchTime;
        
        /// <summary> Exponential moving average of dispatch latencies. </summary>
        double                  m_dAverageDispatchTime;

        /// <summary>   Handle of the global terminate event. When a task's dispatch thread enters a wait
        ///             (e.g. on a port status event or dispatch event), it must also wait on the global
        ///             terminate event to ensure that dispatch threads are awakened and can exit before
        ///             the runtime's data structures are cleaned up.
        ///             </summary>
        HANDLE                  m_hRuntimeTerminateEvent;

        /// <summary>   Handle of the graph's teardown event. When a task's dispatch thread enters a wait
        ///             (e.g. on a port status event or dispatch event), it must also wait on the graph's
        ///             teardown event to ensure that dispatch threads are awakened and can exit before
        ///             the runtime's data structures are cleaned up.
        ///             </summary>
        HANDLE                  m_hGraphTeardownEvent;

        /// <summary>   Handle of the graph's stop event. When a task's dispatch thread enters a wait
        ///             (e.g. on a port status event or dispatch event), it must also wait on the stop
        ///             to ensure that dispatch threads can exit before the runtime's data structures are 
        ///             cleaned up.
        ///             </summary>
        HANDLE                  m_hGraphStopEvent;

        /// <summary>   Handle of the graph's running event.  </summary>
        HANDLE                  m_hGraphRunningEvent;
        
        /// <summary> Type of the canonical geometry estimator </summary>
        GEOMETRYESTIMATORTYPE   m_tEstimatorType;
        
        /// <summary> A user defined geometry estimator function. </summary>
        LPFNGEOMETRYESTIMATOR   m_lpfnEstimator;

        /// <summary>   The number of elements per thread to use in the
        /// 			canonical geometry estimator. </summary>
        int             m_nEstimatorElemsPerThread;

		/// <summary>	The group size in x dimension for dispatch size estimator function. </summary>
		int				m_nEstimatorGroupSizeX;		

		/// <summary>	The group size in x dimension for dispatch size estimator function. </summary>
		int				m_nEstimatorGroupSizeY;		

		/// <summary>	The group size in x dimension for dispatch size estimator function. </summary>
		int				m_nEstimatorGroupSizeZ;		

        /// <summary> true if the task has any port that produces unmigratable 
        /// 		  data (e.g. data with pointers that would be invalid 
        /// 		  if the block is migrated to another device). 
        /// 		  </summary>
        BOOL            m_bProducesUnmigratableData;

        /// <summary>   The scheduler partition hint: if this is set, the scheduler
        ///             will treat it as the identifier of a partition of the graph, 
        ///             where all tasks in that partition should have mandatory affinity
        ///             for the same accelerator compute resources. 
        ///             </summary>
        UINT            m_uiSchedulerPartitionHint;

        /// <summary>   Opaque pointer to any scheduler-specific state. </summary>
        void *          m_lpvSchedulerCookie;

        /// <summary>   true if a scheduler partition hint was set for this task. </summary>
        BOOL            m_bSchedulerPartitionHintSet;

        /// <summary>   true if the task's user code allocates memory through APIs not
        ///             visible to PTask. Use this to avert OOMs by forcing GC sweep
        ///             before such tasks execute.
        ///             </summary>
        BOOL            m_bUserCodeAllocatesMemory;

        /// <summary> The input port map </summary>
        std::map<UINT, Port*>	m_mapInputPorts;
        
        /// <summary> The output port map </summary>
        std::map<UINT, Port*>	m_mapOutputPorts;
        
        /// <summary> The constant port map </summary>
        std::map<UINT, Port*>	m_mapConstantPorts;
        
        /// <summary> The meta port map </summary>
        std::map<UINT, Port*>   m_mapMetaPorts;       

        /// <summary>   The meta port dispatch order: some graph structures induce an ordering
        ///             requirement on the binding of meta ports (allocator ports that collaborate for
        ///             output ports sharing a block through the descriptor port mechanism. This vector
        ///             contains a traversal order for bind time, and is updated by SortMetaPorts() just
        ///             before we put the graph in running state.
        ///             </summary>
        std::vector<Port*>      m_vMetaPortDispatchOrder;

        /// <summary> A map of accelerator-affinity types </summary>
        std::map<Accelerator*, AFFINITYTYPE>    m_vAffinities;

        /// <summary>   The set of dependent accelerator classes. Some tasks make calls that actually use
        ///             other accelerators through interfaces that that are not visible to PTask. The
        ///             canonical (and currently only example) of this is HostTasks which use CUDA+thrust
        ///             to perform operations that are more easily expressed using those interfaces. When
        ///             this occurs, the scheduler must find not only an available accelerator of this
        ///             task's class, but additional accelerators of the classes in the dependences list.
        ///             Failure to do this can introduce synchronization errors when the PTask scheduler
        ///             thinks a device is available but is actually being used. This map is used by the
        ///             scheduler to ensure that all dependences can be satisfied. The map is from
        ///             accelerator class to the number of accelerators of that class required for
        ///             dispatch.
        ///             ----------------------------------------------------------------------
        ///             XXX: TODO: currently we lack an interface for the programmer to tell is which
        ///             port maps to which instance of each class if more than one item of a given class
        ///             is required. Frankly, I'm not sure we want to support this, but at the moment, we
        ///             will assume that all ports bound to a dependent accelerator class are bound to the
        ///             same instance. I.e. the first one. 
        ///             </summary>
        std::map<ACCELERATOR_CLASS, int>                        m_vDependentAcceleratorRequirements;

        /// <summary>   The set of dependent accelerator classes. Some tasks make calls that actually use
        ///             other accelerators through interfaces that that are not visible to PTask. The
        ///             canonical (and currently only example) of this is HostTasks which use CUDA+thrust
        ///             to perform operations that are more easily expressed using those interfaces. When
        ///             this occurs, the scheduler must find not only an available accelerator of this
        ///             task's class, but additional accelerators of the classes in the dependences list.
        ///             Failure to do this can introduce synchronization errors when the PTask scheduler
        ///             thinks a device is available but is actually being used. This map is used at dispatch
        ///             to map ports with dependences to the actual accelerator object. To which they 
        ///             are bound. Because the task may claim more than one item from that class the map
        ///             must be from type to vector. 
        ///             ----------------------------------------------------------------------
        ///             XXX: TODO: See above: we lack an interface for the programmer to tell is which
        ///             port maps to which instance of each class. The code currently assumes (and asserts)
        ///             that the vector is always of length 1. 
        ///             </summary>
        std::map<ACCELERATOR_CLASS, std::vector<Accelerator*>* > m_vDependentAcceleratorAssignments;


        /// <summary>   Per-port accelerator dependencies. Some tasks make calls that actually use other
        ///             accelerators through interfaces that that are not visible to PTask. The canonical
        ///             (and currently only example) of this is HostTasks which use CUDA+thrust to
        ///             perform operations that are more easily expressed using those interfaces. When
        ///             this occurs, the scheduler must find not only an available accelerator of this
        ///             task's class, but additional accelerators of the classes in the dependences list.
        ///             Failure to do this can introduce synchronization errors when the PTask scheduler
        ///             thinks a device is available but is actually being used. This map is from Port to
        ///             the class of the dependent accelerator on which it's input is materialized. This is
        ///             used by Task::Bind*() to make sure that platform specific bindings match the
        ///             accelerator on which the tasks's code will expect it's in/outs to be bound.
        ///             </summary>
        std::map<Port*, ACCELERATOR_CLASS>          m_vPortDependentAcceleratorRequirements;

        /// <summary>   Instantiated per-port accelerator dependencies. Some tasks make calls that actually use other
        ///             accelerators through interfaces that that are not visible to PTask. The canonical
        ///             (and currently only example) of this is HostTasks which use CUDA+thrust to
        ///             perform operations that are more easily expressed using those interfaces. When
        ///             this occurs, the scheduler must find not only an available accelerator of this
        ///             task's class, but additional accelerators of the classes in the dependences list.
        ///             Failure to do this can introduce synchronization errors when the PTask scheduler
        ///             thinks a device is available but is actually being used. This map is from Port to
        ///             the actual dependent accelerator on which it's input is materialized. This is
        ///             used by Task::Bind*() to make sure that platform specific bindings match the
        ///             accelerator on which the tasks's code will expect it's in/outs to be bound.
        ///             </summary>
        std::map<Port*, Accelerator*>          m_vPortDependentAcceleratorAssignments;

        /// <summary>   Sometimes we just want to check whether a given accelerator has been
        ///             assigned to *any* dependent port. Traversing the map structure above is
        ///             in-efficient. THis is a short-cut list to speedup that check. 
        ///             </summary>
        std::set<Accelerator*>                 m_vAllDependentAssignments;

        /// <summary>   The set of accelerators for which we require dispatch locks because 
        ///             data must be migrated to the dispatch (or dependent) memory space. 
        ///             We can release locks on these accelerators at different times, depending
        ///             on whether the target of a migration is the host or not (more precisely,
        ///             whether or not the target has a non-null async context). If the migration target
        ///             has a null async context, we need to be able to drain the source stream, which
        ///             requires a lock on the source accelerator. (Note that when we have non-null target
        ///             the API calls we do to sync the two command streams do not require a lock at the time
        ///             the wait call on the dependence is made. We need this list so that we can release
        ///             these locks accordingly. 
        ///             </summary>
        std::set<Accelerator*>                 m_vMigrationSources;

        /// <summary>   If the user specifies an accelerator with mandatory affinity, cache it here so
        ///             that we can avoid repeated work finding a constraint that trumps all other
        ///             possible scheduling constraints.
        ///             </summary>
        Accelerator *   m_pMandatoryAccelerator;

        /// <summary>   A cache of the result of traversing the port structure in search of
        ///             mandatory dependent affinities. </summary>
        Accelerator *   m_pMandatoryDependentAccelerator;

        /// <summary>   true if the m_pMandatoryDependentAccelerator pointer is valid. </summary>
        BOOL            m_bMandatoryDependentAcceleratorValid;

        /// <summary> The record cardinality of the current dispatch </summary>
        UINT            m_nRecordCardinality;

        /// <summary> Number of actual input resources that must be bound </summary>
        UINT            m_nActualInputCount;
        
        /// <summary> Number of actual output resources that must be bound </summary>
        UINT            m_nActualOutputCount;
        
        /// <summary> Number of actual constant resources that must be bound </summary>
        UINT            m_nActualConstantCount;

        /// <summary> Number of dispatch iterations. Used to build simple iteration.
        /// 		  The default value is one. If a MetaPort with type MF_SIMPLE_ITERATOR
        /// 		  is an input on this task node, then blocks received on that
        /// 		  port will be used to set this member at meta-port bind time. 
        /// 		  The iteration count will be reset to the default after a single
        /// 		  dispatch.
        /// 		  </summary>
        UINT            m_nDispatchIterationCount;    

        /// <summary>   true if this task is inflight. </summary>
        BOOL            m_bInflight;

        /// <summary>   true to shutdown. </summary>
        BOOL            m_bShutdown;

        /// <summary>   dispatch lock. </summary>
        CRITICAL_SECTION m_csDispatchLock;

        /// <summary>   The asynchronous contexts, 1 per accelerator.
        ///             cjr: 7/8/2013--subsuming dependent contexts member into this data structure--if
        ///             there are entries for multiple classes it means this is a host task with
        ///             dependent bindings. There is no meaningful combination that actually requires a
        ///             separate map to track dependent contexts.
        ///             </summary>
        std::map<Accelerator*, AsyncContext*>      m_vDispatchAsyncContexts;

        /// <summary>   true if platform-specific context objects should
        ///             be provided on the call into platform-specific
        ///             dispatch for dependent accelerator bindings. </summary>
        BOOL m_bRequestDependentPSObjects;

        /// <summary>   true if this object has scheduler affinities set for dependent port bindings. The
        ///             data structure traversals required to determine this are non-trivial, deciding
        ///             this condition is on the scheduler's critical path, *and* it is the uncommon
        ///             case. Therefore, we cache this property when the programmer sets up such bindings
        ///             to make sure we can almost always avoid the overheads incurred by depenendent
        ///             affinities.
        ///             </summary>
        BOOL m_bHasDependentAffinities;

        /// <summary>   The number of dependent binding classes. This is a cached value to 
        ///             speed checks for dependent binding needs. </summary>
        UINT m_uiDependentBindingClasses;

        /// <summary>   A complete list of dependent ports. Compiled once at finalize time. </summary>
        std::set<Port*> m_vDependentPorts;

        /// <summary>   true if the dependent port set is valid (has been populated). </summary>
        BOOL            m_bDependentPortSetValid;

        /// <summary>   A map of mandatory dispatch constraints from port to assignment. </summary>
        std::map<Port*, Accelerator*> m_vMandatoryConstraintMap;

        /// <summary>   The set of mandatory accelerator constraints for this task. </summary>
        std::set<Accelerator*> m_vMandatoryConstraintSet;

        /// <summary>   true if mandatory constraints cache is valid. (constraint map and set populated) </summary>
        BOOL                   m_bMandatoryConstraintsCacheValid;

        /// <summary>   A map of soft affinities, from port->acc. </summary>
        std::map<Port*, Accelerator*> m_vPreferenceConstraintMap;

        /// <summary>   the set of accelerators for which soft affinity is configured. </summary>
        std::set<Accelerator*> m_vPreferenceConstraintSet;

        /// <summary>   true if soft constraints cache is valid. (constraint map and set populated) </summary>
        BOOL                   m_bPreferenceConstraintsCacheValid;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Populate dependent port set. </summary>
        ///
        /// <remarks>   Crossbac, 11/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void            PopulateDependentPortSet();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Populate caches of scheduling constraints. </summary>
        ///
        /// <remarks>   Crossbac, 11/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void            PopulateConstraintsCaches();

    private:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record pending buffer asynchronous dependences. Save the ones for which we must
        ///             wait before dispatch so that we can wait them en masse before the dispatch calls,
        ///             and save the ones which will be outstanding after dispatch so we can create new
        ///             dependences after dispatch.  
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/24/2013. </remarks>
        ///
        /// <param name="pBuffer">              [in] non-null, the buffer. </param>
        /// <param name="pPortBindAccelerator"> [in] non-null, the accelerator. </param>
        /// <param name="eOperationType">       Type of the operation. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        RecordPendingBufferAsyncDependences(
            __in PBuffer * pBuffer,
            __in Accelerator * pPortBindAccelerator, 
            __in ASYNCHRONOUS_OPTYPE eOperationType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record pending buffer asynchronous dependences. Save the ones
        ///             for which we must wait before dispatch so that we can wait them
        ///             en masse before the dispatch calls, and save the ones which will
        ///             be outstanding after dispatch so we can create new dependences
        ///             after dispatch.  
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/24/2013. </remarks>
        ///
        /// <param name="pBuffer">          [in] non-null, the buffer. </param>
        /// <param name="pAccelerator">     [in] non-null, the accelerator. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="eOperationType">   Type of the operation. </param>
        ///-------------------------------------------------------------------------------------------------

        void WaitIncomingBufferAsyncDependences();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates the buffer asynchronous dependences. </summary>
        ///
        /// <remarks>   Crossbac, 5/26/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CreateBufferAsyncDependences();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the target memory space for blocks bound to the given port. </summary>
        ///
        /// <remarks>   crossbac, 5/21/2012. </remarks>
        ///
        /// <param name="pPort">    [in,out] The port. </param>
        ///
        /// <returns>   null if it fails, else the target memory space. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * GetTargetMemorySpace(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps information dispatch for the current dispatch. </summary>
        ///
        /// <remarks>   crossbac, 6/6/2012. </remarks>
        ///
        /// <param name="nIteration">       The iteration. </param>
        /// <param name="nMaxIterations">   The maximum iterations. </param>
        ///-------------------------------------------------------------------------------------------------

        void DumpDispatchInfo(UINT nIteration, UINT nMaxIterations);

        /// <summary>   map from port to block for datablocks that are recieved from an input port. This
        ///             list is filled before dispatch, and used afterward to release and/or
        ///             propagate blocks through outputs.
        ///             </summary>
        std::map<Port*, Datablock*>	m_vInflightDatablockMap;

        /// <summary>   Set of blocks inflight as inputs. Tracked in addition to
        ///             the map above to ensure that duplicate bindings for the
        ///             same block occur only for read only inputs. </summary>
        std::set<Datablock*> m_vInflightBlockSet;

        /// <summary>   The blind write inflight blocks. This set is the subset of blocks in flight that
        ///             will be updated by the next dispatch, but the port has out semantics so write is blind. 
        ///             We care about blind writes because unless an output port is part of an in/out pair, the writes
        ///             to blocks on that port are blind, with the side-effect that materializing a view
        ///             on the dispatch accelerator for that block need not actually transfer any data:
        ///             the only requirement is that the block be backed by a device-side buffer. This is
        ///             a very common case and the savings in transfer latency can be significant, so it
        ///             is worth tracking these to enable the optimization.
        ///             </summary>
        std::set<Datablock*> m_vBlindWriteInflightBlocks;

        /// <summary>   The inflight blocks that required exclusive permissions. </summary>
        std::set<Datablock*> m_vWriteableInflightBlocks;

        /// <summary> Datablocks requiring migration before dispatch. This means
        /// 		  any block whose most recent view is on another accelerator.
		///           This structure is a map, rather than a set because the memory space
		///           to which we need to migrate the block may be different from the dispatch 
		///           accelerator. For example, if a port on a host-task is bound to a dependent
		///           CUDA accelerator, the migration target is not host memory, but the memory
		///           space associated with the CUDA device to which the dependent port is bound
		///           before dispatch. 
        /// 		  </summary>
        std::map<Datablock*, Accelerator*> m_vBlocksRequiringMigration;

        /// <summary>   The list of accelerators for which we acquired locks purely so that we could wait
        ///             for outstanding operations queued by other tasks. This set should only have
        ///             members when a task context and dependent contexts are incapable of asynchrony,
        ///             e.g. a pure host task, and an upstream task provides inputs with outstanding
        ///             async ops. We'll need an accelerator lock to do a blocking wait in these cases.
        ///             </summary>

        std::set<Accelerator*> m_vOutstandingAsyncOpAccelerators;

        /// <summary> The required accelerator locks. An accelerator lock
        /// 		  is required for dispatch if it is either:
        /// 		  a) the dispatch accelerator, or  
        /// 		  b) an accelerator from which some block must be   
        /// 		     migrated before dispatch. </summary>
        std::set<Accelerator*, compare_accelerators_by_memspace_id> m_vRequiredAcceleratorLocks;

        /// <summary> A list of all datablocks that must be locked 
        /// 		  during dispatch. </summary>
        std::set<Datablock*> m_vRequiredDatablockLocks;

        /// <summary>   The input blocks. </summary>
        std::map<Datablock*, Port*> m_vInputDatablockLocks;

        /// <summary>   The set of buffers that have been touched during dispatch, and which require
        ///             annotation with a syncpoint object to ensure downstream consumers can wait for
        ///             async operations queued by this tasks dispatch complete before using the buffers.
        ///             </summary>
#ifdef DEBUG
        std::map<PBuffer*, std::set<AsyncDependence*>> m_vIncomingDepBuffers;
        std::map<AsyncDependence*, BOOL> m_vIncomingDuplicateDeps;
#endif
        std::map<AsyncContext*, std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>*> m_vIncomingPendingBufferOps;
        std::map<AsyncContext*, std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>*> m_vOutgoingPendingBufferOps;

        /// <summary>   Set of all ports and upstream channels that must
        /// 			be locked to ensure the ready check is atomic. 
        /// 			</summary>
        std::set<Lockable*> m_vLockableIOResources;

        /// <summary>   true if this object has predicated data flow. 
        /// 			This flag determines whether we require locking
        /// 			protocols to check for dispatch readiness. </summary>
        BOOL m_bHasPredicatedDataFlow;

        /// <summary>   true if this task has implicit meta channel bindings. 
        /// 			This controls whether input channel bindings are
        /// 			applied to more than one actual input port using the
        /// 			meta and template channels in the datablock. </summary>
        BOOL m_bHasImplicitMetaChannelBindings;

        /// <summary>   The task profile. </summary>
        TaskProfile *       m_pTaskProfile;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the task profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///
        /// <param name="bTabularOutput">   true to tabular output. </param>
        ///-------------------------------------------------------------------------------------------------

        void Task::InitializeTaskProfiling(BOOL bTabularOutput);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitializes the task profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///
        /// <param name="bTabularOutput">   true to tabular output. </param>
        ///-------------------------------------------------------------------------------------------------

        void Task::DeinitializeTaskProfiling();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the task instance profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void InitializeTaskInstanceProfile();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitialize task instance profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DeinitializeTaskInstanceProfile();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps a task instance profile statistics. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <param name="ss">   [in,out] The ss. </param>
        ///-------------------------------------------------------------------------------------------------

        void DumpTaskInstanceProfile(std::ostream& ss);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Merge task instance statistics. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void MergeTaskInstanceStatistics();

        /// <summary>   The dispatch counter. </summary>
        DispatchCounter *   m_pDispatchCounter;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the instance dispatch counter. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void InitializeInstanceDispatchCounter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initialises the invocation counting diagnostics tool. This facility
        /// 			allows us to track the number of invocations per task and compare
        /// 			optionally against specified expected number. Useful for finding
        /// 			races or situations where tasks are firing when they shouldn't.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void InitializeDispatchCounting();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Verify dispatch counts against a prediction for every task in the graph. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="pvInvocationCounts">   [in,out] If non-null, the pv invocation counts. </param>
        ///
        /// <returns>   true if the actual and predicted match, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL VerifyDispatchCounts(std::map<std::string, UINT> * pvInvocationCounts);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Task::RecordDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets expected dispatch count. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///
        /// <param name="uiCount">  Number of. </param>
        ///-------------------------------------------------------------------------------------------------

        void Task::SetExpectedDispatchCount(UINT uiCount);

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
        /// <summary>   Check that block pools contain only datablocks with no control signals. </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CheckBlockPoolStates();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find all ports within this given map containing non-zero control codes. </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///
        /// <param name="vPortMap"> [in,out] [in,out] If non-null, the port map. </param>
        ///-------------------------------------------------------------------------------------------------

        void CheckPortControlCodes(std::map<UINT, Port*> &vPortMap);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Executes the static initializers on a different thread, and waits for the
		/// 			result.
		/// 			</summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///
		/// <param name="dwThreadId">	Identifier for the thread. </param>
		///-------------------------------------------------------------------------------------------------

		void InvokeStaticInitializers(DWORD dwThreadId);

        friend class Graph;

    };

};
#endif