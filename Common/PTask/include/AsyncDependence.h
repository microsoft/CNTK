///-------------------------------------------------------------------------------------------------
// file:	AsyncDependence.h
//
// summary:	Declares the asynchronous dependence class
///-------------------------------------------------------------------------------------------------

#ifndef __ASYNC_DEPENDENCE_H__
#define __ASYNC_DEPENDENCE_H__

#include "ReferenceCounted.h"

namespace PTask {

    class SyncPoint;
    class AsyncContext;
    class PBuffer;

	class AsyncDependence : public ReferenceCounted {
	
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the outstanding
        ///                                 asynchronous operations. </param>
        /// <param name="pSyncPoint">       [in,out] If non-null, the sync point on which to depend. </param>
        /// <param name="eOperationType">   Type of the operation. </param>
        ///-------------------------------------------------------------------------------------------------

		AsyncDependence(
            __in AsyncContext * pAsyncContext,
            __in SyncPoint * pSyncPoint,
            __in ASYNCHRONOUS_OPTYPE eOperationType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

		virtual ~AsyncDependence();
                
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the context. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the context. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual AsyncContext * GetContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform context object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the platform context object. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * GetPlatformContextObject();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform wait object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the platform wait object. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * GetPlatformWaitObject();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the synchronise point. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the synchronise point. </returns>
        ///-------------------------------------------------------------------------------------------------

        SyncPoint * GetSyncPoint();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets operation type. </summary>
        ///
        /// <remarks>   crossbac, 5/1/2013. </remarks>
        ///
        /// <returns>   The operation type. </returns>
        ///-------------------------------------------------------------------------------------------------

        ASYNCHRONOUS_OPTYPE GetOperationType();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is outstanding. </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///
        /// <returns>   true if outstanding, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsOutstanding();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Blocking wait complete. </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        BOOL SynchronousExclusiveWait();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Lockless wait outstanding: without acquiring any locks attempt to perform a
        ///             synchronous wait for any outstanding async dependences on this buffer that
        ///             conflict with an operation of the given type. This is an experimental API,
        ///             enable/disable with PTask::Runtime::*etTaskDispatchLocklessIncomingDepWait(),
        ///             attempting to leverage the fact that CUDA apis for waiting on events (which
        ///             appear to be thread-safe and decoupled from a particular device context)
        ///             to minimize serialization associated with outstanding dependences on data
        ///             consumed by tasks that do not require accelerators for any other reason than to
        ///             wait for such operations to complete.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL LocklessWaitOutstanding();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if the dependence is outstanding without acquiring device
        ///             and context locks required to react to resolution if we detect it. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL NonblockingQueryOutstanding();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if the sync point this dependence encapsulates has been
        ///             marked resolved or not. The transition from outstanding to resolved
        ///             is monotonic, so we can make this check without a lock, provided
        ///             that only a FALSE return value is considered actionable.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL QueryOutstandingFlag();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Blocking wait until complete--locks already held. </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        BOOL __SynchronousWaitLocksHeld();

        AsyncContext *      m_pAsyncContext;
        SyncPoint *         m_pSyncPoint;
        ASYNCHRONOUS_OPTYPE m_eOperationType;

        friend class PBuffer;

    };

};

#endif