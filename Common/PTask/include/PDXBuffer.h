//--------------------------------------------------------------------------------------
// File: pdxbuffer.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PDXBUFFER_H_
#define _PDXBUFFER_H_
#include <stdio.h>
#include <crtdbg.h>
#include "ptdxhdr.h"
#include "pbuffer.h"
#include "ptaskutils.h"
#include <map>

namespace PTask {
    
    class PDXBuffer :
        public PBuffer
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pParentDatablock">         [in,out] If non-null, the parent datablock. </param>
        /// <param name="bufferAccessFlags">        The buffer access flags. </param>
        /// <param name="nChannelIndex">            Zero-based index of the datablock channel this
        ///                                         PBuffer is backing. </param>
        /// <param name="pAccelerator">             (optional) [in,out] If non-null, the accelerator. </param>
        /// <param name="pAllocatingAccelerator">   (optional) [in,out] If non-null, the allocating
        ///                                         accelerator. </param>
        /// <param name="uiUniqueIdentifier">       (optional) unique identifier. </param>
        ///-------------------------------------------------------------------------------------------------

        PDXBuffer(Datablock * pParent, 
            BUFFERACCESSFLAGS f, 
            UINT nChannelIndex, 
            Accelerator * p=NULL, 
            Accelerator * pAllocator=NULL, 
            UINT uiUID=ptaskutils::nextuid()
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~PDXBuffer(void);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force synchronize. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void        ForceSynchronize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Complete any outstanding ops. </summary>
        ///
        /// <remarks>   Crossbac, 3/12/2014. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL CompleteOutstandingOps();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check for any outstanding ops. </summary>
        ///
        /// <remarks>   Crossbac, 3/12/2014. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasOutstandingOps();

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
        /// <summary>   Acquires the synchronise. </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2014. </remarks>
        ///
        /// <param name="uiAcquireKey"> The acquire key. </param>
        ///-------------------------------------------------------------------------------------------------

        PDXBuffer *
        PlatformSpecificAcquireSync(
            __in UINT64 uiAcquireKey
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the synchronise. </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2014. </remarks>
        ///
        /// <param name="uiReleaseKey"> The release key. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        PlatformSpecificReleaseSync(
            __in UINT64 uiReleaseKey
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
        /// <param name="bOutstanding">         [in,out] The outstanding. </param>
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
        /// <summary>   Creates device to host staging buffer. </summary>
        ///
        /// <remarks>   Crossbac, 3/11/2014. </remarks>
        ///
        /// <param name="pDevice">  [in,out] If non-null, the device. </param>
        ///
        /// <returns>   The new hto d stage buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        HRESULT 
        PDXBuffer::CreateStagingBuffer(
            __in ID3D11Device * pDevice 
            );


        HANDLE             m_hDXGIHandle;
        IDXGIKeyedMutex *  m_pDXGIKeyedMutex;
        IDXGIResource *    m_pDXGIResource;
        ID3D11Query *      m_pOutstandingQuery;
        ID3D11Buffer *     m_pStageBuffer;
        ID3D11Buffer *     m_pOutstandingOpBuffer;
        HOSTMEMORYEXTENT * m_pOutstandingHtoDTarget;
        HOSTMEMORYEXTENT * m_pOutstandingDtoHTarget;
        BOOL               m_bHtoDStagePopulated;
        BOOL               m_bDtoHStagePopulated;
        BOOL               m_bP2PShareable;
        BOOL               m_bP2PLocked;

    };

};
#endif

