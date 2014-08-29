///-------------------------------------------------------------------------------------------------
// file:	CUAsyncContext.h
//
// summary:	Declares the cu asynchronous context class
///-------------------------------------------------------------------------------------------------

#ifndef __CUDA_ASYNC_CONTEXT_H__
#define __CUDA_ASYNC_CONTEXT_H__
#ifdef CUDA_SUPPORT 

#include "primitive_types.h"
#include "accelerator.h"
#include "AsyncContext.h"
#include "cuhdr.h"
#include <map>
#include <vector>
#include <list>

namespace PTask {

    class Task;
    class SyncPoint;
    class Accelerator;
    class AsyncDependence;

    class CUAsyncContext : public AsyncContext {
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDeviceContext">       [in,out] If non-null, context for the device. </param>
        /// <param name="pTaskContext">         [in,out] If non-null, context for the task. </param>
        /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
        ///-------------------------------------------------------------------------------------------------

        CUAsyncContext(
            __in Accelerator * pDeviceContext,
            __in Task * pTaskContext,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~CUAsyncContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes this object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL Initialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies the device synchronized. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void NotifyDeviceSynchronized();

    protected:

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
            );

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
            );        

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
            );

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
            );

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
            );

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
            );

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
            );

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
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform context object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the platform context object. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void *
        GetPlatformContextObject();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets stream priority. </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///
        /// <returns>   The stream priority. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetStreamPriority();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets stream priority. </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///
        /// <param name="nPriority">    The priority. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetStreamPriority(int nPriority);

        /// <summary>   The stream. </summary>
        CUstream    m_hStream;

        /// <summary>   The last fence. </summary>
        CUevent     m_hLastFence;

        /// <summary>   The event. </summary>
        CUevent     m_hEvent;

        /// <summary>   The stream priority. </summary>
        int         m_nStreamPriority;

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
#endif // CUDA_SUPPORT
#endif