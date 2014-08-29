///-------------------------------------------------------------------------------------------------
// file:	DXAsyncContext.h
//
// summary:	Declares the DirectX asynchronous context class
///-------------------------------------------------------------------------------------------------

#ifndef __DX_ASYNC_CONTEXT_H__
#define __DX_ASYNC_CONTEXT_H__

#include "primitive_types.h"
#include "accelerator.h"
#include "dxaccelerator.h"
#include "task.h"
#include "channel.h"
#include <map>
#include <vector>
#include <list>
#include "hrperft.h"
#include "AsyncContext.h"
#include "AsyncDependence.h"

namespace PTask {

    class DXAsyncContext : public AsyncContext {
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

        DXAsyncContext(
            __in Accelerator * pDeviceContext,
            __in Task * pTaskContext,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~DXAsyncContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes this object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL Initialize();

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

        ID3D11DeviceContext * m_pDXContext;
    };

};
#endif