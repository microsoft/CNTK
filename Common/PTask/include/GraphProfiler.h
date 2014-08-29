///-------------------------------------------------------------------------------------------------
// file:	GraphProfiler.h
//
// summary:	Declares the graph profiler class
///-------------------------------------------------------------------------------------------------

#ifndef __GRAPH_PROFILER_H__
#define __GRAPH_PROFILER_H__

#include "primitive_types.h"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <iostream>

namespace PTask {

    class Task;
    class Graph;

    class GraphProfiler 
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        GraphProfiler(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~GraphProfiler();

    protected:

        Graph *         m_pGraph;

        /// <summary>   Lock for ad hoc graph stats. </summary>
        CRITICAL_SECTION m_csGraphStats;

        /// <summary>   The minimum number of concurrent inflight task threads. </summary>
        UINT            m_uiMinConcurrentInflightThreads;

        /// <summary>   The maximum number of concurrent inflight task threads. </summary>
        UINT            m_uiMaxConcurrentInflightThreads;

        /// <summary>   The concurrent inflight thread accumulator. </summary>
        UINT            m_uiConcurrentInflightThreadAccumulator;

        /// <summary>   The minimum number of concurrent inflight dispatch attempts. </summary>
        UINT            m_uiMinConcurrentInflightDispatches;

        /// <summary>   The maximum number of concurrent inflight dispatch attempts. </summary>
        UINT            m_uiMaxConcurrentInflightDispatches;

        /// <summary>   The maximum concurrent inflight dispatch accumulator. </summary>
        UINT            m_uiConcurrentInflightDispatchAccumulator;

        /// <summary>   The minimum task queue occupancy. </summary>
        UINT            m_uiMinTaskQueueOccupancy;

        /// <summary>   The maximum task queue occupancy. </summary>
        UINT            m_uiMaxTaskQueueOccupancy;

        /// <summary>   The task queue occupancy accumulator. </summary>
        UINT            m_uiTaskQueueOccupancyAccumulator;

        /// <summary>   The task queue samples. </summary>
        UINT            m_uiTaskQueueSamples;

        /// <summary>   The current number of inflight threads. </summary>
        UINT            m_uiAliveThreads;

        /// <summary>   The awake threads. </summary>
        UINT            m_uiAwakeThreads;

        /// <summary>   The blocked threads. </summary>
        UINT            m_uiBlockedRunningThreads;

        /// <summary>   The blocked threads. </summary>
        UINT            m_uiBlockedTaskAvailableThreads;

        /// <summary>   The exited threads. </summary>
        UINT            m_uiExitedThreads;

        /// <summary>   The current number of inflight threads. </summary>
        UINT            m_uiInflightThreads;

        /// <summary>   The current number of inflight dispatches. </summary>
        UINT            m_uiInflightDispatchAttempts;

        /// <summary>   The number of updates to the inflight thread count. </summary>
        UINT            m_uiInflightThreadUpdates;

        /// <summary>   The number of updates to the inflight dispatch count. </summary>
        UINT            m_uiInflightDispatchUpdates;

        /// <summary>   The total number of dispatch attempts. </summary>
        UINT            m_uiDispatchAttempts;

        /// <summary>   The successful dispatch attempts. </summary>
        UINT            m_uiSuccessfulDispatchAttempts;

        /// <summary>   The total number of dequeue attempts. </summary>
        UINT            m_uiDequeueAttempts;

        /// <summary>   The successful dequeu attempts. </summary>
        UINT            m_uiSuccessfulDequeueAttempts;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initialises the graph statistics. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void            Initialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initialises the graph statistics. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void            Destroy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Print graph statistics. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void            Report(std::ostream& ss);

        void            OnTaskThreadAlive();
        void            OnTaskThreadExit();
        void            OnTaskThreadBlockRunningGraph();
        void            OnTaskThreadWakeRunningGraph();
        void            OnTaskThreadBlockTasksAvailable();
        void            OnTaskThreadWakeTasksAvailable();
        void            OnTaskThreadDequeueAttempt();
        void            OnTaskThreadDequeueComplete(Task * pTask);
        void            OnTaskThreadDispatchAttempt();
        void            OnTaskThreadDispatchComplete(BOOL bSuccess);

        friend class Graph;
    };

};
#endif