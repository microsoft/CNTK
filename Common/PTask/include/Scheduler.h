///-------------------------------------------------------------------------------------------------
// file:	Scheduler.h
//
// summary:	Declares the scheduler class
///-------------------------------------------------------------------------------------------------

#pragma once
#ifndef __PTASK_SCHEDULER_H__
#define __PTASK_SCHEDULER_H__

#include <deque>
#include <vector>
#include <set>
#include "oclhdr.h"
#include "ptdxhdr.h"
#include "accelerator.h"
#include "PhysicalDevice.h"
#include "AcceleratorManager.h"
#include "PTaskRuntime.h"
#include "cuhdr.h"
#include <map>

namespace PTask {

    class Task;

    typedef struct adapterrec_t {
        IDXGIAdapter1 * pAdapter;
        DXGI_ADAPTER_DESC desc;
    } ADAPTERRECORD;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class Scheduler
    {
    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        Scheduler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Scheduler(void);

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pause and report graph state: another diagnostics tool. Reset the graph
        ///             running event, and set the probe graph event. This will cause the monitor
        ///             thread to dump the graph and diagnostics DOT file. The event synchronization
        ///             require to do this right (that is, to ensure the graph quiesces before we dump)
        ///             is non-trivial; so we do it in the most unprincipled way possible; Sleep!
        ///             TODO: fix this so we can actually probe the graph from other processes.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void 
        PauseAndReportGraphStates(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the scheduler. </summary>
        ///
        /// <remarks>   crossbac, 5/10/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Initialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Frees all scheduler data structures. </summary>
        ///
        /// <remarks>   crossbac, 5/10/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Destroy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Shuts down the scheduler. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Shutdown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Shuts down the scheduler asynchronously. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static HANDLE ShutdownAsync();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets live graph count. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <returns>   The live graph count. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT GetLiveGraphCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the running graph count. Locks are released before return, so
        /// 			the result is not guaranteed fresh--use only to inform heuristics. 
        /// 			</summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <returns>   The running graph count. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT GetRunningGraphCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies a new graph. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        static void
        NotifyGraphCreate(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notify the scheduler that there is a live graphs for which
        ///             it is ok to schedule tasks. Take note of what backend frameworks
        ///             are represented so we can figure out if we need to do cross
        ///             framework sharing checks on underlying devices.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void 
        NotifyGraphRunning(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notify the scheduler that there are no live graphs. Block until there is one.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void 
        NotifyGraphTeardown(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies the scheduler that a graph is being destroyed. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        NotifyGraphDestroy(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queisce the scheduler. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void GlobalQuiesce();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queisce the scheduler with respect only to a given graph. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void 
        QuiesceGraph(
            __in Graph * pGraph, 
            __in HANDLE hQuiescentEvent
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queisce the scheduler with respect only to a given graph. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static HANDLE
        QuiesceGraphAsync(
            __in Graph * pGraph, 
            __in HANDLE hQuiescentEvent
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the scheduler is globally quiescent. There is no way to lock
        ///             the scheduler from outside the scheduler class so this result
        ///             can be arbitrarily stale: use only as a hint or debug aid!</summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <returns>   true if quiescent, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        IsGlobalQuiescent(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if pGraph is quiescent. Requires no lock, so the result
        ///             can be arbitrarily stale: use only as a hint or debug aid!</summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///
        /// <returns>   true if quiescent, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        IsGraphQuiescent(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks the scheduler: by locking the task queues and accelerator lists,
        ///             ensuring no scheduling can occur. Use with caution. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void LockScheduler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the scheduler by unlocking the task queues and accelerator lists,
        ///             ensuring no scheduling can occur. Use with caution. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void UnlockScheduler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Quiesce the scheduler asynchronously. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static HANDLE GlobalQuiesceAsync();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Leave the quiescent state if we are in it. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void EndGlobalQuiescence();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Leave the quiescent state with respect only to a given graph. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void EndGraphQuiescence(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Abandon any dispatches for this task, with no commitment
        ///             to reschedule the task or defer its execution. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        static void AbandonDispatches(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Abandon dispatches for all tasks from the given graph, with 
        ///             no commitment to run them in the future. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        static void AbandonDispatches(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Move any dispatches for pTask from the run queue to the deferred queue. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        static void DeferDispatches(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Move all tasks dispatches from the given graph from the run queue to the
        ///             deferred queue. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        static void DeferDispatches(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   scheduler thread procedure. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        DWORD ScheduleThreadProc();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases an assigned accelerator described by pAccelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        static void ReleaseAccelerator(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for all capable accelerators for a given accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="acc">      The accelerator class. </param>
        /// <param name="acclist">  [in,out] [in,out] If non-null, the acclist. </param>
        ///
        /// <returns>   The found capable accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        static int FindAllCapableAccelerators(ACCELERATOR_CLASS acc, std::set<Accelerator*>& acclist);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for all capable accelerators for a given accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="acc">      The accelerator class. </param>
        /// <param name="acclist">  [in,out] [in,out] If non-null, the acclist. </param>
        ///
        /// <returns>   The found capable accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        static int FindEnabledCapableAccelerators(ACCELERATOR_CLASS acc, std::set<Accelerator*>& acclist);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return a map from class to accelerator objects for all
        ///             accelerators which have a unique memory space. This function is
        ///             not to be used when the graph is running, but rather, should be 
        ///             used at block pool creation to simplify finding candidate memory
        ///             spaces for block pooling.  
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <param name="accmap">   [in,out] [in,out] If non-null, the accmap. </param>
        ///
        /// <returns>   The number of found accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        static std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>* 
        EnumerateBlockPoolAccelerators();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables/disables the use of the given accelerator by the scheduler: this version
        ///             adds/removes entries from a static black list. On startup the scheduler checks
        ///             devices against this black list to enable/disable them. Do not call after
        ///             PTask::Runtime::Initialized is called.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="eClass">       If non-null, the accelerator. </param>
        /// <param name="nPSDeviceID">  platform specific device id. </param>
        /// <param name="bEnable">      true to enable, false to disable. </param>
        ///
        /// <returns>   PTASK_OK for successful addition/removal to/from the black list. 
        ///             PTASK_ERR_ALREADY_INITIALIZED if the runtime is already initialized.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        static PTRESULT EnableAccelerator(ACCELERATOR_CLASS eClass, int nPSDeviceID, BOOL bEnable);


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables/disables the use of the given accelerator by the scheduler: use to
        ///             restrict/manage the pool of devices actually used by PTask to dispatch tasks.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        /// <param name="bEnabled">     true to enable, false to disable. </param>
        ///
        /// <returns>   success/failure </returns>
        ///-------------------------------------------------------------------------------------------------

        static PTRESULT SetAcceleratorEnabled(Accelerator * pAccelerator, BOOL bEnabled);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pAccelerator' is enabled. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if enabled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL IsEnabled(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pTask' has outstanding dispatches. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if outstanding dispatches, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL HasUnscheduledDispatches(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if pGraph has outstanding dispatches. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if outstanding dispatches, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL HasUnscheduledDispatches(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   complete dispatch for a task, but without assuming it has actually dispatched.
        ///             This is called in an error condition. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task that is completing. </param>
        ///-------------------------------------------------------------------------------------------------

        static void AbandonCurrentDispatch(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   The given task has completed dispatch. We want to move its dispatch accelerator
        ///             and any dependent accelerator objects from the inflight list to the available
        ///             list, signal that dispatch has completed, and that accelerators are available.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task that is completing. </param>
        ///-------------------------------------------------------------------------------------------------

        static void CompleteDispatch(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Begins a dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        static BOOL BeginDispatch(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   schedules or enqueues this task. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        static BOOL ScheduleOrEnqueue(Task * pTask, BOOL& bQueued, BOOL bBypassQ);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the global scheduling mode. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="mode"> The mode. </param>
        ///-------------------------------------------------------------------------------------------------

        static void SetSchedulingMode(SCHEDULINGMODE mode);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the global scheduling mode. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The scheduling mode. </returns>
        ///-------------------------------------------------------------------------------------------------

        static SCHEDULINGMODE GetSchedulingMode(VOID);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find the accelerator with the given identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="uiAcceleratorId">  Identifier for the accelerator. </param>
        ///
        /// <returns>   null if no such accelerator is found, 
        /// 			else the accelerator object with the specified id. </returns>
        ///-------------------------------------------------------------------------------------------------
        
        static Accelerator * GetAcceleratorById(UINT uiAcceleratorId);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerate available accelerators. This is a test method, and provides a list of
        ///             currently available accelerator objects by class; the method can be called
        ///             without a lock, which means that any list returned is not guaranteed to be
        ///             consistent with the scheduler's current state--in other words, the results are
        ///             unactionable unless the runtime is known to be in a quiescent state (e.g. during
        ///             test scenarios!)
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="accClass"> The accumulate class. </param>
        /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        EnumerateAvailableAccelerators(
            ACCELERATOR_CLASS accClass, 
            std::set<Accelerator*>& vaccs
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerate enabled accelerators for the given class. This provides a list of
        ///             currently enabled accelerator objects by class; the method can be called
        ///             without a lock, so if the caller requires a consistent view (with no lock, 
        ///             another thread may enable or disable an accelerator after this list returned),
        ///             LockScheduler should be called.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/8/2013. </remarks>
        ///
        /// <param name="accClass"> The accumulate class. </param>
        /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        EnumerateEnabledAccelerators(
            ACCELERATOR_CLASS accClass, 
            std::set<Accelerator*>& vaccs
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerate all accelerators. This provides a list of
        ///             all known accelerator objects by class, whether the accelerators are actually
        ///             in use by the scheduler or not. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="accClass"> The accumulate class. </param>
        /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        EnumerateAllAccelerators(
            ACCELERATOR_CLASS accClass, 
            std::set<Accelerator*>& vaccs
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerate all accelerators. This provides a list of
        ///             all known accelerator objects by class, whether the accelerators are actually
        ///             in use by the scheduler or not. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="accClass"> The accumulate class. </param>
        /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        EnumerateAllAccelerators(
            ACCELERATOR_CLASS accClass, 
            std::vector<Accelerator*>& vaccs
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the last dispatch timestamp. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <returns>   The last dispatch timestamp. </returns>
        ///-------------------------------------------------------------------------------------------------

        static DWORD GetLastDispatchTimestamp();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the last dispatch timestamp. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <returns>   The last dispatch timestamp. </returns>
        ///-------------------------------------------------------------------------------------------------

        static void UpdateLastDispatchTimestamp();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Is the given accelerator on the available list? </summary>
        ///
        /// <remarks>   crossbac, 6/28/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if available, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsAvailable(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks the scheduler: by locking the task queues and accelerator lists,
        ///             ensuring no scheduling can occur. Use with caution. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __LockScheduler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the scheduler by unlocking the task queues and accelerator lists,
        ///             ensuring no scheduling can occur. Use with caution. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __UnlockScheduler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pGraph' is quiescent. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///
        /// <returns>   true if quiescent, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsGraphQuiescent(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if the scheduler is globally quiescent. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <returns>   true if globally quiescent, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsGlobalQuiescent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for the "strongest" available accelerator of the given accelerator class.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="acc">  The accelerator class. </param>
        ///
        /// <returns>   null if it fails, else the found available accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * FindStrongestAvailableAccelerator(ACCELERATOR_CLASS acc);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for the first available accelerator for the given ptask. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">                    [in] non-null, the task. </param>
        /// <param name="pnDependentAccelerators">  [in,out] If non-null, the pn dependent accelerators. </param>
        /// <param name="pppDependentAccelerators"> [in,out] If non-null, the ppp dependent accelerators. </param>
        ///
        /// <returns>   null if it fails, else the found available accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * 
        FindStrongestAvailableAccelerators(
            __in  Task* pTask,
            __out int * pnDependentAccelerators,
            __out Accelerator *** pppDependentAccelerators
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   If there is a scheduling choice that must be made based on affinity the inability
        ///             of the system to migrate unmarshallable datablocks, find an accelerator that fits
        ///             the constraints, if possible.  Call with accelerator lock held.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///
        /// <returns>   null if it fails, else the found affinitized accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * FindAffinitizedAccelerator(Task* pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Given a list of candidates, find the one that is the best fit, where "best"
        ///             adheres to the data-aware policy. We want the accelerator where the most inputs
        ///             are already up-to- date.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="histogram">    [in] histogram: accelerator->(materialized input count). </param>
        /// <param name="candidates">   [in] non-null, the candidates. </param>
        ///
        /// <returns>   null if it fails, else the found accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * FindDataAwareBestFit(std::map<Accelerator*, UINT> &histogram,
                                           std::set<Accelerator*> &candidates);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Look at the input blocks for the given ptask, and build a histogram mapping
        ///             accelerators to the number of inputs that are currently up to date in that memory
        ///             domain. Return the number of histogram buckets, (a zero return value indicates
        ///             there is no way to make a good assignment based on locality alone because no  
        ///             inputs are up to date on any accelerators).
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pTask">        [in] non-null, the task. </param>
        /// <param name="histogram">    [in,out] non-null, the histogram. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT BuildInputHistogram(Task* pTask, std::map<Accelerator*, UINT> &histogram);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for all accelerators capable of executing a given ptask. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        /// <param name="acclist">  [in,out] [in,out] If non-null, the acclist. </param>
        ///
        /// <returns>   The found capable accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        int FindCapableEnabledAccelerators(Task* pTask, std::vector<Accelerator*>& acclist);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Part of initialization. 
        /// 			Creates accelerator objects for all devices in the system. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CreateAccelerators();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Spawn scheduler threads. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void SpawnSchedulerThreads();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sort run queue. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void SortRunQueue();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks the quiescence state. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void LockQuiescenceState();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the quiescence state. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void UnlockQuiescenceState();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks the task queues. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void LockTaskQueues();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the task queues. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void UnlockTaskQueues();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks the accelerator list. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void LockAcceleratorMaps();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the accelerator list. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void UnlockAcceleratorMaps();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates dispatch statistics. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator">             [in] non-null, the accelerator. </param>
        /// <param name="pTask">                    [in] non-null, the task. </param>
        /// <param name="nDependentAccelerators">   The dependent accelerators. </param>
        /// <param name="ppDependentAccelerators">  [in,out] If non-null, the dependent accelerators. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        UpdateDispatchStatistics(
            __in Accelerator * pAccelerator, 
            __in Task * pTask,
            __in int nDependentAccelerators,
            __in Accelerator** ppDependentAccelerators
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates dispatch statistics for accelerators scheduled as dependent accelerators
        ///             (not the dispatch accelerator, and used by a task through APIs other than PTask:
        ///             e.g. HostTasks that use CUDA).
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
        /// <param name="pTask">        [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        void UpdateDependentDispatchStatistics(Accelerator * pAccelerator, Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps dispatch statistics. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DumpDispatchStatistics();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Match an accelerator to a ptask based on current scheduling mode. In the general
        ///             case, matching returns a single accelerator, and leaves the dependent accelerator
        ///             list set to NULL. If the task has dependences on other accelerators (e.g. as many
        ///             CUDA+thrust-based HostTasks do), the dependency list will come back non-empty as
        ///             well).
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">                    [in] non-null, the task. </param>
        /// <param name="pnDependentAccelerators">  [in,out] non-null, the number of dependent
        ///                                         accelerators. </param>
        /// <param name="pppDependentAccelerators"> [out] If non-null, the dependent accelerators. The
        ///                                         caller is *NOT* responsible for freeing this list, as
        ///                                         it will be freed by the runtime after dispatch
        ///                                         completes. </param>
        ///
        /// <returns>   null if it fails, else a pointer to the dispatch accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * MatchAccelerators(Task * pTask, 
                                        int * pnDependentAccelerators,
                                        Accelerator *** pppDependentAccelerators);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Match an accelerator to a ptask using the "data aware" policy. In the general
        ///             case, matching returns a single accelerator, and leaves the dependent accelerator
        ///             list set to NULL. If the task has dependences on other accelerators (e.g. as many
        ///             CUDA+thrust-based HostTasks do), the dependency list will come back non-empty as
        ///             well).
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">                    [in] non-null, the task. </param>
        /// <param name="pnDependentAccelerators">  [in,out] non-null, the number of dependent
        ///                                         accelerators. </param>
        /// <param name="pppDependentAccelerators"> [out] If non-null, the dependent accelerators. The
        ///                                         caller is *NOT* responsible for freeing this list, as
        ///                                         it will be freed by the runtime after dispatch
        ///                                         completes. </param>
        ///
        /// <returns>   null if it fails, else a pointer to the dispatch accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * MatchAcceleratorsDataAware(Task * pTask, 
                                                 int * pnDependentAccelerators,
                                                 Accelerator *** pppDependentAccelerators);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Select the "best" accelerator from a list of candidates.
        ///             "Best" is determined by the following sort order (somewhat arbitrarily) is:
        ///             * highest runtime version support
        ///             * support for concurrent kernels
        ///             * highest core count
        ///             * fastest core clock
        ///             * biggest memory
        ///             * device enumeration order (ensuring we will usually choose   
        ///               the same physical device across multiple back ends)        
        ///               </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="candidates">   [in,out] the candidate list. </param>
        /// <param name="target">       Target class of the task being matched. </param>
        ///
        /// <returns>   null if it fails, else a pointer to the dispatch accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * SelectBestAccelerator(
                        std::vector<Accelerator*>& candidates, 
                        ACCELERATOR_CLASS target);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assign dependent accelerators. If this task requires resources in addition to the
        ///             dispatch accelerator, assign them. Return true if an assignment can be made
        ///             (indicating that this task can be dispatched).
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/9/2012. </remarks>
        ///
        /// <param name="pTask">                    [in] non-null, the task. </param>
        /// <param name="pnDependentAccelerators">  [in,out] If non-null, the pn dependent accelerators. </param>
        /// <param name="pppDependentAccelerators"> [in,out] If non-null, the ppp dependent accelerators. </param>
        /// <param name="pDispatchAccelerator">     [in,out] If non-null, the assignment. </param>
        /// <param name="pInputHistogram">          [in,out] If non-null, a histogram mapping
        ///                                         accelerators to accelerators to number of inputs for
        ///                                         this task which are already materialized there. When
        ///                                         this is null, ignore it. When it is available, if it
        ///                                         is possible to choose dependent assignments to
        ///                                         maximize locality based on this histogram, do it. 
        ///                                         </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        AssignDependentAccelerators(
            Task * pTask,
            int * pnDependentAccelerators,
            Accelerator *** pppDependentAccelerators,
            Accelerator * pDispatchAccelerator,
            std::map<Accelerator*, UINT>* pInputHistogram
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds all the accelerator objects in the list to the schedulers accelator-tracking
        ///             data structures.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="eClass">       The class. </param>
        /// <param name="accelerators"> [in,out] [in,out] If non-null, the accelerators. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        AddAccelerators(
            __in ACCELERATOR_CLASS eClass,
            __in std::vector<Accelerator*> &accelerators
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Top level scheduling function. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL Schedule();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Schedules for the given task. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        /// <param name="bBypassQ"> (Optional) the bypass q. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL AssignDispatchAccelerator(Task * pTask, BOOL bBypassQ=FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for all capable accelerators for a given accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="acc">      The accelerator class. </param>
        /// <param name="acclist">  [in,out] [in,out] If non-null, the acclist. </param>
        ///
        /// <returns>   The found capable accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        int __FindAllCapableAccelerators(ACCELERATOR_CLASS acc, std::vector<Accelerator*>& acclist);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for all capable accelerators for a given accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="acc">      The accelerator class. </param>
        /// <param name="acclist">  [in,out] [in,out] If non-null, the acclist. </param>
        ///
        /// <returns>   The found capable accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        int __FindAllCapableAccelerators(ACCELERATOR_CLASS acc, std::set<Accelerator*>& acclist);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for all capable accelerators for a given accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="acc">      The accelerator class. </param>
        /// <param name="acclist">  [in,out] [in,out] If non-null, the acclist. </param>
        ///
        /// <returns>   The found capable accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        int __FindEnabledCapableAccelerators(ACCELERATOR_CLASS acc, std::vector<Accelerator*>& acclist);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for all capable accelerators for a given accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="acc">      The accelerator class. </param>
        /// <param name="acclist">  [in,out] [in,out] If non-null, the acclist. </param>
        ///
        /// <returns>   The found capable accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        int __FindEnabledCapableAccelerators(ACCELERATOR_CLASS acc, std::set<Accelerator*>& acclist);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return a map from class to accelerator objects for all
        ///             accelerators which have a unique memory space. This function is
        ///             not to be used when the graph is running, but rather, should be 
        ///             used at block pool creation to simplify finding candidate memory
        ///             spaces for block pooling.  
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <param name="accmap">   [in,out] [in,out] If non-null, the accmap. </param>
        ///
        /// <returns>   The number of found accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        int __EnumerateBlockPoolAccelerators(std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>& accmap);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   abandon any dispatches for the task by removing it from the run queue
        ///             and deferred queue. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task that is completing. </param>
        ///-------------------------------------------------------------------------------------------------

        void __AbandonDispatches(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Abandon dispatches for the graph--get any outstanding scheduling 
        ///             attempts out of both the run queue and deferred queue. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        void __AbandonDispatches(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   abandon a dispatch attempt
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task that is completing. </param>
        ///-------------------------------------------------------------------------------------------------

        void __AbandonCurrentDispatch(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   The given task has completed dispatch. We want to move its dispatch accelerator
        ///             and any dependent accelerator objects from the inflight list to the available
        ///             list, signal that dispatch has completed, and that accelerators are available.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task that is completing. </param>
        ///-------------------------------------------------------------------------------------------------

        void __CompleteDispatch(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Begins a dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL __BeginDispatch(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Attempt to schedule a task. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL __ScheduleOrEnqueue(Task * pTask, BOOL& bQueued, BOOL bBypassQ);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Move pTask from the run queue to the deferred queue. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        void __DeferDispatches(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Moves all tasks from the given graph from the run queue to the deferred queue. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        void __DeferDispatches(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find the accelerator with the given identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="uiAcceleratorId">  Identifier for the accelerator. </param>
        ///
        /// <returns>   null if no such accelerator is found, 
        /// 			else the accelerator object with the specified id. </returns>
        ///-------------------------------------------------------------------------------------------------
        
        Accelerator * __GetAcceleratorById(UINT uiAcceleratorId);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the global scheduling mode. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="mode"> The mode. </param>
        ///-------------------------------------------------------------------------------------------------

        void __SetSchedulingMode(SCHEDULINGMODE mode);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the global scheduling mode. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The scheduling mode. </returns>
        ///-------------------------------------------------------------------------------------------------

        SCHEDULINGMODE __GetSchedulingMode(VOID);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases an assigned accelerator described by pAccelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void __ClaimAccelerator(Accelerator * pAccelerator, Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases an assigned accelerator described by pAccelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void __ReleaseAccelerator(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the dispatch resources held by a task on a normal 
        ///             dispatch completion. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        void __ReleaseDispatchResources(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases any dispatch resources held by a task on a failed dispatch completion
        ///             (abandoned dispatch can leave things in the inflight list for a task that the
        ///             task doesn't know about).
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        void __ReleaseDanglingDispatchResources(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   On dispatch completion or abandonment, if there are graphs waiting for quiescence
        ///             or the scheduler is waiting for global quiescence, update our view of such
        ///             outstanding requests, signaling waiters if we have achieved quiescence.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        void __UpdateQuiescenceWaiters(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Shuts down the scheduler. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __Shutdown();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Shuts down the scheduler asynchronously. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        HANDLE __ShutdownAsync();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets live graph count. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <returns>   The live graph count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __GetLiveGraphCount();

		///-------------------------------------------------------------------------------------------------
		/// <summary>   Gets the running graph count. Locks are released before return, so
		/// 			the result is not guaranteed fresh--use only to inform heuristics. 
		/// 			</summary>
		///
		/// <remarks>   crossbac, 6/24/2013. </remarks>
		///
		/// <returns>   The running graph count. </returns>
		///-------------------------------------------------------------------------------------------------

		UINT __GetRunningGraphCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies a new graph. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        __NotifyGraphCreate(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notify the scheduler that there is a live graphs for which
        ///             it is ok to schedule tasks. Take note of what backend frameworks
        ///             are represented so we can figure out if we need to do cross
        ///             framework sharing checks on underlying devices.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        __NotifyGraphRunning(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notify the scheduler that there are no live graphs. Block until there is one.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        __NotifyGraphTeardown(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies the scheduler that a graph is being destroyed. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        __NotifyGraphDestroy(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pause and report graph state: another diagnostics tool. Reset the graph
        ///             running event, and set the probe graph event. This will cause the monitor
        ///             thread to dump the graph and diagnostics DOT file. The event synchronization
        ///             require to do this right (that is, to ensure the graph quiesces before we dump)
        ///             is non-trivial; so we do it in the most unprincipled way possible; Sleep!
        ///             TODO: fix this so we can actually probe the graph from other processes.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        __PauseAndReportGraphStates(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Is this task dispatchable based on the graph state? </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if dispatchable, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        __IsDispatchable(
            __in Task * pTask
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Is this task dispatch deferrable based on the graph state? </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if dispatchable, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        __IsDeferrable(
            __in Task * pTask
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queisce the scheduler. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __GlobalQuiesce();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queisce the scheduler with respect to a particular graph. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __QuiesceGraph(Graph * pGraph, HANDLE hQuiescentEvent);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queisce the scheduler with respect to a particular graph. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        HANDLE __QuiesceGraphAsync(Graph * pGraph, HANDLE hQuiescentEvent);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Quiesce the scheduler asynchronously. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        HANDLE __GlobalQuiesceAsync();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Release the scheduler if it has been queisced. (enqueue any
        ///             deferred tasks and unblock scheduler threads). </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __EndGlobalQuiescence();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   End a quiescent period with respect to a particular graph. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __EndGraphQuiescence(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assign an accelerator to a ptask. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTask">    [in] non-null, the task. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * __AssignAccelerator(Task* pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables/disables the use of the given accelerator by the scheduler: use to
        ///             restrict/manage the pool of devices actually used by PTask to dispatch tasks.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        /// <param name="bEnabled">     true to enable, false to disable. </param>
        ///
        /// <returns>   success/failure </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT __SetAcceleratorEnabled(Accelerator * pAccelerator, BOOL bEnabled);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pAccelerator' is enabled. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if enabled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsEnabled(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pTask' has outstanding dispatches. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if outstanding dispatches, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __HasUnscheduledDispatches(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if pGraph has outstanding dispatches. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if outstanding dispatches, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __HasUnscheduledDispatches(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pTask' has any deferred dispatches. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if outstanding dispatches, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __HasDeferredDispatches(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if pGraph has Deferred dispatches. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if outstanding dispatches, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __HasDeferredDispatches(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pTask' has any in flight dispatches. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if outstanding dispatches, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __HasInflightDispatches(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if pGraph has in flight dispatches. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if outstanding dispatches, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __HasInflightDispatches(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the last dispatch timestamp. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <returns>   The last dispatch timestamp. </returns>
        ///-------------------------------------------------------------------------------------------------

        DWORD __GetLastDispatchTimestamp();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the last dispatch timestamp. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <returns>   The last dispatch timestamp. </returns>
        ///-------------------------------------------------------------------------------------------------

        void __UpdateLastDispatchTimestamp();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the required accelerator availability. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void UpdateRequiredAcceleratorAvailability();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Choose best based on histogram. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pCandidates">      [in,out] If non-null, the candidates. </param>
        /// <param name="pInputHistogram">  [in,out] If non-null, a histogram mapping accelerators to
        ///                                 accelerators to number of inputs for this task which are
        ///                                 already materialized there. When this is null, ignore it.
        ///                                 When it is available, if it is possible to choose dependent
        ///                                 assignments to maximize locality based on this histogram, do
        ///                                 it. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * SelectMaxLocalityAccelerator(std::set<Accelerator*>* pCandidates,
                                                   std::map<Accelerator*, UINT>* pInputHistogram);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check whether there are multiple back-end GPU frameworks
        ///             in use in this graph, and if so, notify the scheduler so it
        ///             can avoid over committing devices that are shared through 
        ///             different accelerator objects (with different subtypes). 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void FindCrossRuntimeDependences(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if we can required accelerators available. </summary>
        ///
        /// <remarks>   crossbac, 6/14/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL RequiredAcceleratorsAvailable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the tasks ready view. </summary>
        ///
        /// <remarks>   crossbac, 6/24/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __UpdateTaskQueueView();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the accelerator queue view. </summary>
        ///
        /// <remarks>   crossbac, 7/7/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __UpdateAcceleratorQueueView();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates our view of the "ok to schedule" state. 
        ///             We can schedule if there are runnable graphs or quiescing graphs,
        ///             and we are not in a wait for global quiescence.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/9/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __UpdateOKToScheduleView();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the quiescence hint. </summary>
        ///
        /// <remarks>   Crossbac, 7/9/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __UpdateQuiescenceHint();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check dispatch constraints. Debug utility--checks whether dispatch
        ///             and dependent accelerator assignments conform to the affinity 
        ///             the user has specified for the given task. </summary>
        ///
        /// <remarks>   Crossbac, 11/15/2013. </remarks>
        ///
        /// <param name="pTask">                    [in,out] If non-null, the task. </param>
        /// <param name="pDispatchAssignment">      [in,out] If non-null, the dispatch assignment. </param>
        /// <param name="pnDependentAccelerators">  [in,out] If non-null, the pn dependent accelerators. </param>
        /// <param name="pppDependentAccelerators"> [in,out] If non-null, the PPP dependent accelerators. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        __CheckDispatchConstraints(
            __in Task * pTask,
            __in Accelerator * pDispatchAssignment,
            __in int * pnDependentAccelerators,
            __in Accelerator *** pppDependentAccelerators
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pAccelerator' is on a black or white restriction list. this is a helper
        ///             function for deciding whether to enable an accelerator at init time.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2014. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        /// <param name="pList">        [in,out] If non-null, the list. </param>
        ///
        /// <returns>   true if on the list, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        IsOnRestrictionList(
            __in Accelerator * pAccelerator, 
            __in std::set<int>* pList
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   get the black and white lists relevant to the given accelerator. 
        ///             helper function. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2014. </remarks>
        ///
        /// <param name="eClass">           the accelerator class. </param>
        /// <param name="pWhiteList">       [in,out] If non-null, list of whites. </param>
        /// <param name="pBlackList">       [in,out] If non-null, list of blacks. </param>
        /// <param name="bCreateIfAbsent">  The create if absent. </param>
        ///-------------------------------------------------------------------------------------------------

        static void
        GetRestrictionLists(
            __in ACCELERATOR_CLASS eClass,
            __out std::set<int>** pWhiteList,
            __out std::set<int>** pBlackList,
            __in BOOL bCreateIfAbsent
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pAccelerator' is black listed. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2014. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if black listed, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL IsBlackListed(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pAccelerator' is white listed. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2014. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if black listed, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL IsWhiteListed(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pAccelerator' is an enablable accelerator based
        ///             on white-list/black-list static informations configured by 
        ///             user code before the scheduler is created (PTask::Runtime::Initialize is called).
        ///             If a white list exists for the given class, the accelerator must be on the white
        ///             list. If a black list exists, it must not be on the black list. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2014. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if enablable candidate, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL IsEnablable(Accelerator* pAccelerator);

        /// <summary> The scheduling mode </summary>
        SCHEDULINGMODE m_eSchedulingMode;

        /// <summary>   Number of threads. </summary>
        UINT m_uiThreadCount;
        
        /// <summary> Handle of the scheduler thread </summary>
        HANDLE * m_phThreads;
        
        /// <summary> Handle of event signalling state change for task queue.</summary>
        HANDLE m_hTaskQueue;

        /// <summary>   Are tasks ready?  </summary>
        BOOL m_bTasksAvailable;

        /// <summary>   are there live graphs in the system? </summary>
        BOOL m_bLiveGraphs;

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

        /// <summary>   The set of live accelerator classes--keep track!
        ///             We encapsulate devices as accelerators *per-runtime*
        ///             as accelerator objects, which means it's actually possible
        ///             for a DX accelerator to be unavailable because the same GPU
        ///             is being used through a CU Accelerator. This complicates
        ///             scheduling quite a bit and is not the common case, so
        ///             we assume we don't need those checks unless we encounter
        ///             a graph that requires them.
        ///             </summary>
        std::set<ACCELERATOR_CLASS> m_vLiveAcceleratorClasses;
        
        /// <summary> Handle of the accelerators available event </summary>
        std::map<ACCELERATOR_CLASS, HANDLE> m_vhAcceleratorsAvailable;

        /// <summary>   Event signalling state change for ready accelerator map. Blocked threads may wish
        ///             to update the list of resources they are blocked for.
        ///             </summary>
        HANDLE m_hAcceleratorQueue;

        /// <summary>   The global terminate event. </summary>
        HANDLE m_hRuntimeTerminateEvent;

        /// <summary>   The scheduler termination event. </summary>
        HANDLE m_hSchedulerTerminateEvent;

        /// <summary>   per graph shut-down events. </summary>
        std::map<Graph*, HANDLE> m_vhShutdownEvents;

        /// <summary>   per graph quiescent events. </summary>
        std::map<Graph*, HANDLE> m_vhQuiescentEvents;

        /// <summary>   The set of graphs that are new but have never entered the running state.
        ///             </summary>
        std::set<Graph*> m_vNascentGraphs;

        /// <summary>   The set of graphs that are "live": ie they have ostensibly runnable tasks. This
        ///             can be out of date with the graph state of course--it's just a hint.
        ///             </summary>
        std::set<Graph*> m_vRunningGraphs;

        /// <summary>   Graphs in the quiescing state: we are draining outstanding dispatches
        ///             for these graphs. Tasks for these graphs should not be
        ///             dispatched, but deferred, since the graph can potentially
        ///             enter the running state again. 
        ///             </summary>
        std::set<Graph*> m_vGraphsWaitingForQuiescence;

        /// <summary>   Quiesced graphs. Tasks for these graphs should not be
        ///             dispatched, but deferred, since the graph can potentially
        ///             enter the running state again. 
        ///             </summary>
        std::set<Graph*> m_vQuiescedGraphs;

        /// <summary>   live graphs: any graph that has entered the run state and
        ///             is still running or has the potential to become
        ///             runnable in the future (quiescing or quiescent). </summary>
        std::set<Graph*> m_vLiveGraphs;

        /// <summary>   Graphs torn down or tearing down but not yet deleted. 
        ///             Tasks from these graphs should never be dispatched
        ///             because the graph is going to be deleted in the future.
        ///             </summary>
        std::set<Graph*> m_vRetiredGraphs;

        /// <summary>   The quiescent event. </summary>
        HANDLE m_hGlobalQuiescentEvent;

        /// <summary>   True if the scheduler is "running"--that is there are graphs
        ///             for which it is OK to schedule tasks. If we are quiescing 
        ///             the scheduler, or if there are no live graphs in the 
        ///             system, this will be reset. 
        ///             </summary>
        HANDLE m_hOKToSchedule;

        /// <summary>   true if this it's ok for the scheduler to do it's job.
        ///             This is a (debug-aid) snapshot of the state we believe the 
        ///             m_hOKToSchedule event to be in. </summary>
        BOOL   m_bOKToSchedule;

        /// <summary>   Is the scheduler quiescent? </summary>
        BOOL m_bInGlobalQuiescentState;

        /// <summary>   Is the scheduler waiting for quiescence? </summary>
        BOOL m_bWaitingForGlobalQuiescence;

        /// <summary>   are accelerators available? </summary>
        std::map<ACCELERATOR_CLASS, BOOL> m_vbAcceleratorsAvailable; 
        
        /// <summary> scheduler alive flag </summary>
        BOOL m_bAlive;
        
        /// <summary> run queue lock </summary>
        CRITICAL_SECTION m_csQ;
        
        /// <summary> Depth of the q lock </summary>
        int m_nQueueLockDepth;
        
        /// <summary> accelerator list lock </summary>
        CRITICAL_SECTION m_csAccelerators;

        /// <summary>   Lock protecting flag indication the presence of quiescent states that require
        ///             update. Most of the data structures related to quiescence are protected by the
        ///             queue lock, which is typically contended. Quiescence-state is typically empty.
        ///             Forcing every task state change to contend for the queue lock to check for
        ///             usually absent quiescence update needs is a bummer. Protect quiescent state with
        ///             an additional critical section that is ordered before the queue lock, allowing
        ///             common case updates to avoid taking the queue lock if it's unnecessary.
        ///             </summary>
        CRITICAL_SECTION m_csQuiescentState;

        /// <summary>   true if ther is live quiescence state. </summary>
        BOOL             m_bQuiescenceInProgress;

        /// <summary>   Depth of the quiescence lock. </summary>
        int             m_nQuiescenceLockDepth;
        
        /// <summary> Depth of the accelerator lock </summary>
        int m_nAcceleratorMapLockDepth;
        
        /// <summary> Manager for physical devices </summary>
        AcceleratorManager * m_pDeviceManager;
        
        /// <summary> List of all accelerator objects in the system </summary>
        std::set<Accelerator*> m_vMasterAcceleratorList;

        /// <summary> List of all accelerator objects in the system </summary>
        std::map<ACCELERATOR_CLASS, std::set<Accelerator*>*> m_vEnabledAccelerators;

        /// <summary> map of available accelerators (not assign to dispatch) </summary>
        std::map<ACCELERATOR_CLASS, std::set<Accelerator*>*> m_vAvailableAccelerators;
        
        /// <summary> The in-flight accelerators (currently assigned to dispatch) </summary>
        std::map<Accelerator*, Task*> m_vInflightAccelerators;

        /// <summary>   A map from class to enabled accelerators. </summary>
        std::map<ACCELERATOR_CLASS, std::set<Accelerator*>> m_accmap;

        /// <summary> lock for statistics data structures</summary>
        CRITICAL_SECTION m_csStatistics;
        
        /// <summary> Stats: per-accelerator dispatch statistics </summary>
        std::map<Accelerator*, int> m_vAcceleratorDispatches;

        /// <summary> Stats: per-accelerator dependent dispatch statistics </summary>
        std::map<Accelerator*, int> m_vDependentAcceleratorDispatches;
        
        /// <summary> per-graph, per-task dispatch statistics </summary>
        std::map<std::string, std::map<std::string, std::map<Accelerator*, int>*>*> m_vDispatches;

        /// <summary> per-graph, per-task dependent dispatch binding statistics </summary>
        std::map<std::string, std::map<std::string, std::map<Accelerator*, int>*>*> m_vDependentDispatches;

        /// <summary> map from accelerator id to accelerator object. </summary>
        std::map<UINT, Accelerator*> m_vAcceleratorMap;
        
        /// <summary> The dispatch total </summary>
        int nDispatchTotal;

        /// <summary>   The dependent dispatch total. </summary>
        int nDependentDispatchTotal;

        /// <summary>   The dependent accelerator dispatch total. </summary>
        int nDependentAcceleratorDispatchTotal;
        
        /// <summary> The run queue</summary>
        std::deque<Task*> m_vReadyQ;

        /// <summary> The deferred run queue</summary>
        std::deque<Task*> m_vDeferredQ;

        /// <summary>   The last dispatch timestamp. </summary>
        DWORD m_dwLastDispatchTimestamp;

        /// <summary>   The create struct dispatch timestamp. </summary>
        CRITICAL_SECTION m_csDispatchTimestamp;

        /// <summary> black/white lists of platform specific device ids whose
        ///           entries represent DisableAccelerator calls before 
        ///           runtime initialization. 
        ///           </summary>
        static std::map<ACCELERATOR_CLASS, std::set<int>*> s_vBlackListDeviceIDs;
        static std::map<ACCELERATOR_CLASS, std::set<int>*> s_vWhiteListDeviceIDs;

#ifdef OPENCL_SUPPORT
        /// <summary> The Open cl devices </summary>
        std::set <cl_device_id> vCLDevices;
#endif

#ifdef CUDA_SUPPORT
        /// <summary> The cuda devices </summary>
        std::set <CUdevice> vCUDADevices;
        
        /// <summary> List of names of the cuda devices </summary>
        std::set<std::string> vCUDADeviceNames;
#endif
    };

};

#endif // __PTASK_SCHEDULER_H__