///-------------------------------------------------------------------------------------------------
// file:	dispatchcounter.h
//
// summary:	Declares the dispatchcounter class
///-------------------------------------------------------------------------------------------------

#ifndef _DISPATCH_COUNTER_H_
#define _DISPATCH_COUNTER_H_

#include "primitive_types.h"
#include <vector>
#include <map>
#include <set>

class CHighResolutionTimer;
class CSharedPerformanceTimer;

namespace PTask {

    class Task;
    class Port;

    class DispatchCounter {

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        DispatchCounter(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~DispatchCounter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initialises the invocation counting diagnostics tool. This facility
        /// 			allows us to track the number of invocations per task and compare
        /// 			optionally against specified expected number. Useful for finding
        /// 			races or situations where tasks are firing when they shouldn't.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Initialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitialises the invocation counting diagnostics tool. This facility
        /// 			allows us to track the number of invocations per task and compare
        /// 			optionally against specified expected number. Useful for finding
        /// 			races or situations where tasks are firing when they shouldn't.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Deinitialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the dispatch counts for every task in the graph. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Report(std::ostream& ss);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Verify dispatch counts against a prediction for every task in the graph. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="pvInvocationCounts">   [in,out] If non-null, the pv invocation counts. </param>
        ///
        /// <returns>   true if the actual and predicted match, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Verify(std::map<std::string, UINT> * pvInvocationCounts);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record the fact that a task dispatch has occurred. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void RecordDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the expected dispatch count for the given task. The runtime will assert if
        ///             the actual number of dispatches for the task exceeds this number.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="nDispatchCount">   Number of dispatches. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetExpectedDispatchCount(UINT nDispatchCount);

    protected:

        /// <summary>   Lock for the dispatch count map. </summary>
        static CRITICAL_SECTION m_csDispatchMap;

        /// <summary>   Number of dispatches per task. Keyed by name to
        /// 			be robust to graph deletion/runtime-cleanup </summary>
        static std::map<std::string, UINT> m_vDispatchMap;

        /// <summary>   true if dispatch counting initialized. </summary>
        static BOOL m_bDispatchCountingInitialized;

        /// <summary>   The task. </summary>
        Task * m_pTask;
        
        /// <summary>   The expected number of dispatches for this task. </summary>
        UINT m_nExpectedDispatches;
        
        /// <summary>   The actual number of times this task has been dispatched. </summary>
        UINT m_nActualDispatches;

    };
};
#endif