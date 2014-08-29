///-------------------------------------------------------------------------------------------------
// file:	SyncPoint.h
//
// summary:	Declares the synchronise point class
///-------------------------------------------------------------------------------------------------

#ifndef __SYNC_POINT_H__
#define __SYNC_POINT_H__

#include "primitive_types.h"
#include "ReferenceCounted.h"

namespace PTask {

    class AsyncContext;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   A synchronization point, on which dependences may be created, so that other
    ///             threads/downstream operations can wait until dependences on previous operations
    ///             in this context have resolved.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class SyncPoint : public ReferenceCounted {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="_pAsyncContext">               [in] If non-null, context for the asynchronous. </param>
        /// <param name="_pPlatformAsyncContextObject"> [in] non-null, the platform-specific asynchronous
        ///                                             context object. E.g. the stream in CUDA, the
        ///                                             ID3D11ImmediateContext object in DirectX and so
        ///                                             on. </param>
        /// <param name="_pPlatformAsyncWaitObject">    [in] non-null, a platform-specific asynchronous
        ///                                             wait object. E.g. a windows event or a cuda event
        ///                                             object, etc. </param>
        /// <param name="_pPlatformParentSyncObject">   The platform parent synchronise object. </param>
        ///-------------------------------------------------------------------------------------------------

        SyncPoint(
            __in AsyncContext *  _pAsyncContext,
            __in void *          _pPlatformAsyncContextObject,
            __in void *          _pPlatformAsyncWaitObject,
            __in void *          _pPlatformParentSyncObject
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~SyncPoint();

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
        /// <summary>   Gets the platform wait object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the platform wait object. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * GetPlatformParentObject();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this sync point is *definitely* resolved. If this returns false, then
        ///             the sync point represents completed work and no lock is required to check this
        ///             since the transition is monotonic. If it returns TRUE indicating the work is
        ///             still outstanding, that doesn't mean the sync point hasn't resolved. It just
        ///             means the caller should acquire locks and call QueryOutstanding to get a higher
        ///             fidelity answer.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if outstanding, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        QueryOutstandingFlag(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this sync point represents outstanding work or work that has been
        ///             completed.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        ///
        /// <returns>   true if outstanding, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        QueryOutstanding(
            __in AsyncContext * pAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this sync point represents outstanding work or work that has been
        ///             completed without blocking to acquire the locks needed to update async context
        ///             and accelerator state when a state change on this sync point is detected.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if outstanding, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        NonblockingQueryOutstanding(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Marks this sync point as retired, meaning all the ops preceding it
        /// 			are known to be complete. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        MarkRetired(
            __in BOOL bContextSynchronized,
            __in BOOL bStatusQueried
            );

        /////-------------------------------------------------------------------------------------------------
        ///// <summary>   Release by decrementing the refcount. We override the implementation inherited
        /////             from ReferenceCounted so that we can figure out if the outstanding list
        /////             for the containing async context can be garbage collected. If the refcount
        /////             goes from 2 to 1, that *should* mean that its async context holds the only
        /////             reference, and therefor we can retire it. 
        /////             </summary>
        /////
        ///// <remarks>   Crossbac, 12/19/2011. </remarks>
        /////
        ///// <returns>   . </returns>
        /////-------------------------------------------------------------------------------------------------

        //virtual LONG Release();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets asynchronous context. </summary>
        ///
        /// <remarks>   crossbac, 5/1/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the asynchronous context. </returns>
        ///-------------------------------------------------------------------------------------------------

        AsyncContext * GetAsyncContext();
    
    protected:        

        /// <summary>   The platform-specific asynchronous context object. 
        /// 			Maps loosely to the abstraction of an independent command
        /// 			queue for a given device context. 
        /// 			</summary>
        void * m_pPlatformAsyncContextObject;

        /// <summary>   The platform-specific asynchronous wait object. </summary>
        void * m_pPlatformAsyncWaitObject;

        /// <summary>   The platform parent synchronisation object--not used by all platforms. </summary>
        void * m_pPlatformParentSyncObject;

        /// <summary>   Context for the outstanding asynchronous operations. </summary>
        AsyncContext *  m_pAsyncContext;

        /// <summary>   true if ops preceding this sync-point are known to
        /// 			be outstanding (or rather, conservatively, not known
        /// 			to be complete). </summary>
        BOOL            m_bOutstanding;

        /// <summary>   true if we queried the underlying event to figure out
        ///             that the sync point was no longer outstanding. </summary>
        BOOL            m_bStatusQueried;

        /// <summary>   true if the context was synchronized, causing the
        ///             sync point to be no longer outstanding.
        ///             </summary>
        BOOL            m_bContextSynchronized;

        friend class AsyncContext;
        friend class AsyncDependence;

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