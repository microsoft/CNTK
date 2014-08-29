///-------------------------------------------------------------------------------------------------
// file:	AsyncContext.h
//
// summary:	Declares the asynchronous context class
///-------------------------------------------------------------------------------------------------

#ifndef __ASYNC_CONTEXT_H__
#define __ASYNC_CONTEXT_H__

#include <stdio.h>
#include <crtdbg.h>
#include <deque>
#include <set>
#include "ReferenceCounted.h"

namespace PTask {

    class Task;
    class SyncPoint;
    class AsyncDependence;
    class Accelerator;

    class AsyncContext : public ReferenceCounted {
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDeviceContext">       [in] non-null, context for the device. </param>
        /// <param name="pTaskContext">         [in] non-null, context for the task. </param>
        /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
        ///-------------------------------------------------------------------------------------------------

        AsyncContext(
            __in Accelerator * pDeviceContext,
            __in Task * pTaskContext,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~AsyncContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a dependence on the synchronization point. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual AsyncDependence * 
        CreateDependence(
            __in ASYNCHRONOUS_OPTYPE eOperationType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a dependence on the synchronization point. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual AsyncDependence * 
        CreateDependence(
            __in SyncPoint * pSyncPoint,
            __in ASYNCHRONOUS_OPTYPE eOperationType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a synchronization point. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual SyncPoint * CreateSyncPoint(void * pPSSyncObject);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroys a synchronization point. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL DestroySyncPoint(SyncPoint * pSyncPoint);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synchronizes the context. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL SynchronizeContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies the device synchronized. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void NotifyDeviceSynchronized();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence--asynchronous; puts a fence in the command queue
        ///             for this context. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        OrderSubsequentOperationsAfter(
            __in AsyncDependence * pDependence
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence--synchronous </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        SynchronousWait(
            __in AsyncDependence * pDependence
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synchronous wait for dependence resolution. </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///
        /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        SynchronousWait(
            __in SyncPoint * pSyncPoint
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pDependence' is dependence resolved. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">  [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if dependence resolved, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        QueryDependenceOutstanding(
            __in AsyncDependence * pDependence
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the device context. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the device context. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator * GetDeviceContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the task context. </summary>
        ///
        /// <remarks>   Crossbac, 7/13/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the task context. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Task * GetTaskContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform context object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the platform context object. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void * GetPlatformContextObject()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes this object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL Initialize()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this accelerator backing this context encapsulates a backend
        ///             framework that provides explicit APIs for managing outstanding (Asynchronous)
        ///             operations. When this is the case, the corresponding AsyncContext subclass can
        ///             manage outstanding dependences explicitly to increase concurrency and avoid
        ///             syncing with the device. When it is *not* the case, we must synchronize when we
        ///             data to and from this accelerator context and contexts that *do* support an
        ///             explicit async API. For example, CUDA supports the stream and event API to
        ///             explicitly manage dependences and we use this feature heavily to allow task
        ///             dispatch to get far ahead of device- side dispatch. However when data moves
        ///             between CUAccelerators and other accelerator classes, we must use synchronous
        ///             operations or provide a way to wait for outstanding dependences from those
        ///             contexts to resolve. This method is used to tell us whether we can create an
        ///             outstanding dependence after making calls that queue work, or whether we need to
        ///             synchronize.
        ///             
        ///             This method simply calls the method of the same name on the (device context)
        ///             accelerator, and is only provided for convenience.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsExplicitAsyncOperations(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks the accelerator. </summary>
        ///
        /// <remarks>   crossbac, 6/26/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual VOID        LockAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the accelerator. </summary>
        ///
        /// <remarks>   crossbac, 6/26/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual VOID        UnlockAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Non-blocking check whether the dependence is still outstanding. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="pDep"> [in,out] If non-null, the dep. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        
        NonblockingQueryOutstanding(
            __inout AsyncDependence * pDep
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synchronous wait for outstanding async op--do not acquire locks
        ///             required to update async and device context state in response 
        ///             to a successful query or wait. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="pDep"> [in,out] If non-null, the dep. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        LocklessWaitOutstanding(
            __inout AsyncDependence * pDep
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the type (dedicated purpose) of the asynchronous context. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2013. </remarks>
        ///
        /// <returns>   The asynchronous context type. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual ASYNCCONTEXTTYPE 
        GetAsyncContextType(
            VOID
            );

    protected:


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence--synchronously. Because we may have to make backend
        ///             framework calls (e.g. to wait or check CUDA event states) we may require
        ///             a number of fairly coarse locks, including an accelerator lock. When calling
        ///             this from task dispatch context, the caller must acquire all locks up front
        ///             since there are lock ordering displines such as (Accelerator->Datablock) that 
        ///             are there to prevent deadlock for concurrent tasks. 
        ///             
        ///             This version assumes (or rather only asserts) that accelerator locks are held
        ///             already, so it can be called from dispatch context: Task is a friend class
        ///             to enable this while minimizing the potential for abuse.
        ///
        ///             This is a building block for the public version, which first collects locks,
        ///             but which cannot be called from a dispatch context as a result.
        ///              </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        __SynchronousWaitLocksHeld(
            __in AsyncDependence * pDependence
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence--synchronously. Because we may have to make backend
        ///             framework calls (e.g. to wait or check CUDA event states) we may require
        ///             a number of fairly coarse locks, including an accelerator lock. When calling
        ///             this from task dispatch context, the caller must acquire all locks up front
        ///             since there are lock ordering displines such as (Accelerator->Datablock) that 
        ///             are there to prevent deadlock for concurrent tasks. 
        ///             
        ///             This version assumes (or rather only asserts) that accelerator locks are held
        ///             already, so it can be called from dispatch context: Task is a friend class
        ///             to enable this while minimizing the potential for abuse.
        ///
        ///             This is a building block for the public version, which first collects locks,
        ///             but which cannot be called from a dispatch context as a result.
        ///              </summary>
        ///              
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///
        /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        __SynchronousWaitLocksHeld(
            __in SyncPoint * pSyncPoint
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sync points, once marked resolved, can never return to the outstanding state.
        ///             Consequently, if a lock-free check of the oustanding flag returns false, there is
        ///             no danger of a race. Conversely, checking if the state is unknown requires
        ///             accelerator and context locks which restrict concurrency and have lock ordering
        ///             disciplines that make it difficult to *always* have these locks when this check
        ///             is required. So a quick check without a lock that can avoid locks when they are
        ///             unnecessary is a handy tool.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/26/2013. </remarks>
        ///
        /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
        ///
        /// <returns>   true if synchronise point resolved no lock, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        __IsSyncPointResolvedNoLock(
            __in SyncPoint * pSyncPoint
            );


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   DEBUG instrumentation for analyzing the composition of outstanding dependences
        ///             on this async context. How many are flagged as resolved, how many are *actually*
        ///             resolved, is the queue monotonic?
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/31/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        AnalyzeOutstandingQueue(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   garbage collect the outstanding queue. Anything no longer outstanding
        ///             can be removed from the queue. The original version is very 
        ///             conservative about how much it actually cleans up--it only checks flags
        ///             (and thus avoids back-end API calls to check event status), which is
        ///             good for performance until the number of outstanding deps piles up.
        ///             This version attempts to balance these effects by making API calls
        ///             if the number of outstanding deps goes beyond a threshold. This version 
        ///             can be reinstated with a static member variable s_bUseConservativeGC. 
        ///             The threshold at which to start making API calls is controlled by 
        ///             PTask::Runtime::[Get|Set]AsyncContextGCQueryThreshold().
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/31/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        GarbageCollectOutstandingQueue(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   garbage collect the outstanding queue. Anything no longer outstanding
        ///             can be removed from the queue. This (old, obsolete) version is very 
        ///             conservative about how much it actually cleans up--it only checks flags
        ///             (and thus avoids back-end API calls to check event status), which is
        ///             good for performance until the number of outstanding deps piles up.
        ///             The new version attempts to balance these effects by making API calls
        ///             if the number of outstanding deps goes beyond a threshold. This version 
        ///             can be reinstated with a static member variable s_bUseConservativeGC. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        GarbageCollectOutstandingQueueConservatively(
            VOID
            );


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Truncate queue. Only to be called when the context is known to be synchronized!
        ///             Marks all outstanding sync points as resolved, and removes them from the
        ///             outstanding queue.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        TruncateOutstandingQueue(
            __in BOOL bContextSynchronized
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Truncate queue. Only to be called when the context is known to be synchronized!
        ///             Marks all outstanding sync points as resolved, and removes them from the
        ///             outstanding queue.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void TruncateOutstandingQueueFrom(
            __in SyncPoint * pSyncPoint
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence asynchronously by inserting a dependence
        ///             in the current context (stream) on the event in the sync point. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        InsertFence(
            __in SyncPoint * pSyncPoint
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform specific create synchronization point. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual SyncPoint *
        PlatformSpecificCreateSyncPoint(
            void * pPSSyncObject
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform specific destroy synchronization point. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        PlatformSpecificDestroySynchronizationPoint(
            __in SyncPoint * pSyncPoint
            )=0;        

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence asynchronously. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificInsertFence(
            __in SyncPoint * pSyncPoint 
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence synchronously. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificSynchronousWait(
            __in SyncPoint * pSyncPoint 
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence synchronously without locking the async context
        ///             or underlying accelerator: this simplifies lock acquisition for such
        ///             waits, but at the expense of leaving live dependences that are
        ///             actually resolved.  </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificLocklessSynchronousWait(
            __in SyncPoint * pSyncPoint 
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if we can platform specific synchronize context. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificSynchronizeContext(
            VOID
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if the sync point is resolved (and marks it if so). </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificQueryOutstanding(
            __inout SyncPoint * pSyncPoint
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform specific nonblocking check whether the event remains outstanding. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        PlatformSpecificNonblockingQueryOutstanding(
            __inout SyncPoint * pSyncPoint
            )=0;

        std::deque<SyncPoint*>                    m_qOutstanding;
        Accelerator *                             m_pDeviceContext;
        Task *                                    m_pTaskContext;
        ASYNCCONTEXTTYPE                          m_eAsyncContextType;
        static BOOL                               s_bUseConservativeGC;

        friend class AsyncDependence;
        friend class SyncPoint;
        friend class Task;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a string describing this refcount object. Allows subclasses to
        ///             provide overrides that make leaks easier to find when detected by the
        ///             rc profiler. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/9/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the rectangle profile descriptor. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual std::string GetRCProfileDescriptor();

    };

};

#endif