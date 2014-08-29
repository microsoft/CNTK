//--------------------------------------------------------------------------------------
// File: pcubuffer.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PCUBUFFER_H_
#define _PCUBUFFER_H_
#include <stdio.h>
#include <crtdbg.h>
#include <cuda.h>
#include "pbuffer.h"
#include "ptaskutils.h"
#include <map>

namespace PTask {

    class CUAccelerator;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform-specific buffer for CUDA. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

	class PCUBuffer :
		public PBuffer
	{
	public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pParent">                  [in,out] If non-null, the parent. </param>
        /// <param name="bufferAccessFlags">        The buffer access flags. </param>
        /// <param name="nChannelIndex">            Zero-based index of the n channel. </param>
        /// <param name="pAccelerator">             (optional) [in,out] If non-null, the accelerator. </param>
        /// <param name="pAllocatorAccelerator">    (optional) [in,out] If non-null, the allocator
        ///                                         accelerator. </param>
        /// <param name="uiUID">                    (optional) the uid. </param>
        ///-------------------------------------------------------------------------------------------------

		PCUBuffer(Datablock * pParent, 
            BUFFERACCESSFLAGS bufferAccessFlags, 
            UINT nChannelIndex, 
            Accelerator * pAccelerator=NULL, 
            Accelerator * pAllocatorAccelerator=NULL, 
            UINT uiUID=ptaskutils::nextuid()
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

		virtual ~PCUBuffer(void);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force synchronize. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void        ForceSynchronize(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Device to device transfer. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pDstBuffer">       [in,out] If non-null, the accelerator. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        
        DeviceToDeviceTransfer(
            __inout PBuffer *       pDstBuffer,
            __in    AsyncContext *  pAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Device memcpy. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pDstBuffer">       If non-null, the accelerator. </param>
        /// <param name="pSrcBuffer">       If non-null, buffer for source data. </param>
        /// <param name="pAsyncContext">    If non-null, context for the asynchronous. </param>
        /// <param name="uiCopyBytes">      The copy in bytes. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        
        Copy(
            __inout PBuffer *       pDstBuffer,
            __inout PBuffer *       pSrcBuffer,
            __in    AsyncContext *  pAsyncContext,
            __in    UINT            uiCopyBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Device memcpy. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pDstBuffer">       [in,out] If non-null, the accelerator. </param>
        /// <param name="pSrcBuffer">       [in,out] If non-null, buffer for source data. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        
        Copy(
            __inout PBuffer *       pDstBuffer,
            __inout PBuffer *       pSrcBuffer,
            __in    AsyncContext *  pAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the derived class supports a memset API. </summary>
        ///
        /// <remarks>   crossbac, 8/14/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsMemset();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   memset. </summary>
        ///
        /// <remarks>   crossbac, 8/14/2013. </remarks>
        ///
        /// <param name="nValue">           The value. </param>
        /// <param name="szExtentBytes">    The extent in bytes. </param>
        ///
        /// <returns>   the number of bytes set </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual size_t       
        FillExtent(
            __in int nValue, 
            __in size_t szExtentBytes=0
            );

	protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize host view. </summary>
        ///
        /// <remarks>   crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, information describing the lpv. </param>
        /// <param name="pBuffer">              [in,out] The data. </param>
        /// <param name="bForceSynchronous">    (optional) the elide synchronization. </param>
        /// <param name="bRequestOutstanding">  [in,out] The request outstanding. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT		
        __populateHostView(
            __in  AsyncContext * pAsyncContext,
            __in  HOSTMEMORYEXTENT * pBuffer,
            __in  BOOL bForceSynchronous,
            __out BOOL &bRequestOutstanding
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize mutable accelerator view. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiBufferSizeBytes">    The buffer size in bytes. </param>
        /// <param name="pInitialData">         [in,out] If non-null, the data. </param>
        /// <param name="bOutstanding">         [in,out] The outstanding. </param>
        /// <param name="pModule">              [in,out] (optional)  If non-null, the module. </param>
        /// <param name="lpszBinding">          (optional) the binding. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

	    virtual UINT		
        __populateMutableAcceleratorView(
            __in AsyncContext *     pAsyncContext,
            __in UINT               uiBufferSizeBytes,
            __in HOSTMEMORYEXTENT * pInitialData,
            __out BOOL&             bOutstanding,
            __in void *             pModule,
            __in const char *       lpszBinding
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize immutable accelerator view. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiBufferSizeBytes">    If non-null, the data. </param>
        /// <param name="pInitialData">         [in,out] The bytes. </param>
        /// <param name="pModule">              [in,out] (optional)  If non-null, the module. </param>
        /// <param name="lpszBinding">          (optional) the binding. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT		
        __populateImmutableAcceleratorView(
            __in AsyncContext *     pAsyncContext,
            __in UINT               uiBufferSizeBytes,
            __in HOSTMEMORYEXTENT * pInitialData,
            __out BOOL&             bOutstanding,
            __in void *             pModule,
            __in const char *       lpszBinding
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes a device-side buffer that is expected to be bound to mutable device
        ///             resources (not in constant memory).
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <param name="pAsyncContext">            [in,out] (optional)  If non-null, context for the
        ///                                         asynchronous. </param>
        /// <param name="uiBufferSizeBytes">        The buffer size in bytes. </param>
        /// <param name="pInitialBufferContents">   (optional) [in] If non-null, the initial buffer
        ///                                         contents. </param>
        /// <param name="strDebugBufferName">       (optional) [in] If non-null, a name to assign to the
        ///                                         buffer which will be used to label runtime- specific
        ///                                         objects to aid in debugging. Ignored on release
        ///                                         builds. </param>
        /// <param name="bByteAddressable">         (optional) true if the buffer should be byte
        ///                                         addressable. </param>
        ///
        /// <returns>   PTRESULT (use PTSUCCESS/PTFAILED macros) </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT	
        CreateMutableBuffer(
            __in AsyncContext *     pAsyncContext,
            __in UINT               uiBufferSizeBytes,
            __in HOSTMEMORYEXTENT * pInitialBufferContents, 
            __in char *             strDebugBufferName=NULL, 
            __in bool               bByteAddressable=true                                                    
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes a device-side buffer that is expected to be bound to immutable device
        ///             resources (i.e. those in constant memory).
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <param name="pAsyncContext">                [in,out] If non-null, context for the
        ///                                             asynchronous. </param>
        /// <param name="pInitialBufferContents">       (optional) [in] If non-null, the initial buffer
        ///                                             contents. </param>
        /// <param name="uiInitialContentsSizeBytes">   (optional) the initial contents size in bytes. </param>
        /// <param name="strDebugBufferName">           (optional) [in] If non-null, a name to assign to
        ///                                             the buffer which will be used to label runtime-
        ///                                             specific objects to aid in debugging. Ignored on
        ///                                             release builds. </param>
        /// <param name="bByteAddressable">             (optional) true if the buffer should be byte
        ///                                             addressable. </param>
        ///
        /// <returns>   PTRESULT (use PTSUCCESS/PTFAILED macros) </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT	
        CreateImmutableBuffer(
            __in AsyncContext *     pAsyncContext,
            __in UINT               uiBufferSizeBytes,
            __in HOSTMEMORYEXTENT * pInitialBufferContents, 
            __in char *             strDebugBufferName=NULL, 
            __in bool               bByteAddressable=true
            );


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates readable bindable objects if the access flags indicate they will be
        ///             required at dispatch time.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
        ///                         used for debugging. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT	CreateBindableObjectsReadable(char * szname = NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates writable bindable objects if the access flags indicate they will be
        ///             required at dispatch time.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
        ///                         used for debugging. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT	CreateBindableObjectsWriteable(char * szname = NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates immutable bindable objects if needed for dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
        ///                         used for debugging. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT	CreateBindableObjectsImmutable(char * szname = NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if a device-side view of this data can be materialized
        /// 			using memset APIs rather than memcpy APIs. </summary>
        ///
        /// <remarks>   crossbac, 7/10/2012. </remarks>
        ///
        /// <param name="uiBufferBytes">        The buffer in bytes. </param>
        /// <param name="pInitialData">         [in,out] If non-null, the data. </param>
        /// <param name="uiInitialDataBytes">   The bytes. </param>
        ///
        /// <returns>   true if device view memsettable, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        IsDeviceViewMemsettable(
            __in UINT uiBufferBytes,
            __in void * pInitialData,
            __in UINT uiInitialDataBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a device memset stride. </summary>
        ///
        /// <remarks>   crossbac, 7/10/2012. </remarks>
        ///
        /// <param name="uiBufferBytes">        The buffer in bytes. </param>
        /// <param name="uiInitialDataBytes">   The bytes. </param>
        ///
        /// <returns>   The device memset stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT
        GetDeviceMemsetStride(
            __in UINT uiBufferBytes,
            __in UINT uiInitialDataBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a device memset count. </summary>
        ///
        /// <remarks>   crossbac, 7/10/2012. </remarks>
        ///
        /// <param name="uiBufferBytes">        The buffer in bytes. </param>
        /// <param name="uiInitialDataBytes">   The bytes. </param>
        ///
        /// <returns>   The device memset count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT
        GetDeviceMemsetCount(
            __in UINT uiBufferBytes,
            __in UINT uiInitialDataBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a device memset value. </summary>
        ///
        /// <remarks>   crossbac, 7/10/2012. </remarks>
        ///
        /// <param name="pInitialValue">    [in,out] The buffer in bytes. </param>
        ///
        /// <returns>   The device memset count. </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID *
        GetDeviceMemsetValue(
            __in void * pInitialValue
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the CUDA stream. </summary>
        ///
        /// <remarks>   crossbac, 7/10/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        ///
        /// <returns>   The stream. </returns>
        ///-------------------------------------------------------------------------------------------------

        CUstream GetStream(AsyncContext * pAsyncContext);

        /// <summary>   The platform-specific accelerator. For convenience--we could typecast 
        /// 			m_pAccelerator inherited from the super-class every time we want one,
        /// 			but it's ugly, and happens alot. </summary>
        CUAccelerator * m_pPSAcc;

        /// <summary>   Buffer for page locked allocations. Asynchronous transfers in CUDA 
        /// 			require the host-side to be page-locked. When we create a buffer that
        /// 			requires asynchronous transfers, we will page-lock the initial data if it is 
        /// 			provided, remembering to un-pin it at delete time, or allocate a page-locked
        /// 			buffer if it is not provided. </summary>
        void *          m_pPageLockedBuffer;

        /// <summary>   true if the page locked buffer is owned by this object, and must
        /// 			therefore be freed (instead of un-pinned) at deletion time. </summary>
        BOOL            m_bPageLockedBufferOwned; 

        /// <summary>   true if the device buffer was created using cuMemAlloc and we 
        /// 			are responsible for freeing it. If the device buffer was
        /// 			created by finding the device-side mapping for a page-locked
        /// 			buffer, then it shares the fate of the page-locked buffer
        /// 			and we must be careful not to free it.
        /// 			and should not free it. </summary>
        BOOL            m_bDeviceBufferOwned;

	};
};
#endif

