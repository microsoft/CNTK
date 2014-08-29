//--------------------------------------------------------------------------------------
// File: pclbuffer.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PCLBUFFER_H_
#define _PCLBUFFER_H_
#ifdef OPENCL_SUPPORT
#include <stdio.h>
#include <crtdbg.h>
#include "pbuffer.h"
#include "ptaskutils.h"
#include <map>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform-specific buffer class for OpenCL runtime access to 
    /// 			. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class PCLBuffer :
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

		PCLBuffer(Datablock * pParent, 
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

		virtual ~PCLBuffer(void);


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force synchronize. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void        ForceSynchronize();

	protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize host view. </summary>
        ///
        /// <remarks>   crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, information describing the lpv. </param>
        /// <param name="pHostSourceBuffer">    [in,out] If non-null, buffer for host source data. </param>
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

        /// <summary> true if this buffer is going to bound to a device-side
        /// 		  scalar variable. </summary>
        BOOL                m_bScalarBinding;
    };

};
#endif
#endif
