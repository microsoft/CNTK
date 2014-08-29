//--------------------------------------------------------------------------------------
// File: dxaccelerator.h
// direct x based accelerator
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _DX_ACCELERATOR_H_
#define _DX_ACCELERATOR_H_

#include "primitive_types.h"
#include "ptdxhdr.h"
#include "datablocktemplate.h"
#include "dxcodecache.h"
#include "accelerator.h"
#include "CompiledKernel.h"
#include <vector>

namespace PTask {

    class DXAccelerator : public Accelerator {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        DXAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~DXAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the open. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        HRESULT					Open();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Opens a DXAccelerator by 
        /// 			associating the DXAccelerator object with an adapter
        /// 			and a live D3D11 device context </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAdapter">             [in] If non-null, the adapter. </param>
        /// <param name="uiEnumerationIndex">   Zero-based index of the adapter when 
        /// 									the OS enumerates it. This is necessary because
        /// 									the D3D11 APIs for creating a device are
        /// 									idiosyncratic in the presence of multiple 
        /// 									adapters.</param>
        ///
        /// <returns>   S_OK on success, E_FAIL otherwise.
        /// 			Use windows SUCCEEDED() and FAILED() macros </returns>
        ///-------------------------------------------------------------------------------------------------

        HRESULT					Open(IDXGIAdapter * pAdapter, UINT uiEnumerationIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Opens a reference device. 
        /// 			Should only be called if the programmer wants to work
        /// 			with the runtime in an environment where no DX11 hardware
        /// 			is present, since the reference device is very very slow.
        /// 			Use PTask::Runtime::SetUseReferenceDevices() to enable
        /// 			this feature.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        HRESULT                 OpenReferenceDevice();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return a pointer to the platform-specific device object. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the device. </returns>
        ///-------------------------------------------------------------------------------------------------

        void*					GetDevice();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return a pointer to the platform-specific device context. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the context. </returns>
        ///-------------------------------------------------------------------------------------------------

        void*					GetContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an asynchronous context for the task. Create the cuda stream for this
        ///             ptask.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 12/20/2011.
        ///             
        ///             This method is required of all subclasses, and abstracts the work associated with
        ///             managing whatever framework-level asynchrony abstractions are supported by the
        ///             backend target. For example, CUDA supports the "stream", while DirectX supports
        ///             an ID3D11ImmediateContext, OpenCL has command queues, and so on.
        ///             </remarks>
        ///
        /// <param name="pTask">                [in] non-null, the CUDA-capable acclerator to which the
        ///                                     stream is bound. </param>
        /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual AsyncContext * 
        PlatformSpecificCreateAsyncContext(
            __in Task * pTask,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Cache the DX objects created when a shader is compiled
        /// 			so that subsequent calls are made to compile the
        /// 			same function, we can reuse the existing binaries. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="szFile">   [in] non-null, the file name. </param>
        /// <param name="szFunc">   [in] non-null, the function name </param>
        /// <param name="p">        [in] non-null, a pointer to a ID3D11ComputeShader. </param>
        ///-------------------------------------------------------------------------------------------------

        void					CachePutShader(char * szFile, char * szFunc, ID3D11ComputeShader*p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check the shader cache for an existing binary made from the
        /// 			given HLSL file and function name. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="szFile">   [in] If non-null, the file. </param>
        /// <param name="szFunc">   [in] If non-null, the func. </param>
        ///
        /// <returns>   null if it fails, else the shader binary. </returns>
        ///-------------------------------------------------------------------------------------------------

        ID3D11ComputeShader*	CacheGetShader(char * szFile, char * szFunc);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Compiles accelerator source code to create a PTask binary. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. 
        /// 			The function accepts a file name and an operation in the file
        /// 			to build a binary for. For example, "foo.hlsl" and "vectoradd" will
        /// 			compile the vectoradd() shader in foo.hlsl. On success the function
        /// 			will create platform-specific binary and module objects that can be
        /// 			later used by the runtime to invoke the shader code. The caller can
        /// 			provide a buffer for compiler output, which if present, the runtime
        /// 			will fill *iff* the compilation fails. 
        /// 			***
        /// 			NB: Thread group dimensions are optional parameters here but
        /// 			*must* be used for optimal performance because DirectX requires
        /// 			statically specified thread group sizes, and the default values
        /// 			of 1, 1, 1 are not likely to be a good performance combination.
        /// 			</remarks>
        ///
        /// <param name="lpszFileName">             [in] filename+path of source. cannot be null.</param>
        /// <param name="lpszOperation">            [in] Function name in source file. cannot be null.</param>
        /// <param name="ppPlatformSpecificBinary"> [out] On success, a platform specific binary. </param>
        /// <param name="ppPlatformSpecificModule"> [out] On success, a platform specific module handle. </param>
        /// <param name="lpszCompilerOutput">       (optional) [in,out] On failure, the compiler output. </param>
        /// <param name="uiCompilerOutput">         (optional) [in] length of buffer supplied for 
        /// 										compiler output. </param>
        /// <param name="tgx">                      (optional) thread group X dimensions. (see remarks)</param>
        /// <param name="tgy">                      (optional) thread group Y dimensions. (see remarks)</param>
        /// <param name="tgz">                      (optional) thread group Z dimensions. (see remarks)</param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            Compile(
                                    char * lpszFileName, 
                                    char * lpszOperation, 
                                    void ** ppPlatformSpecificBinary,
                                    void ** ppPlatformSpecificModule,
                                    char * lpszCompilerOutput=NULL,
                                    int uiCompilerOutput=0,
                                    int tgx=1, 
                                    int tgy=1, 
                                    int tgz=1 
                                    );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Compiles accelerator source code to create a PTask binary. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011.
        ///             
        ///             The function accepts a string of source code and an operation in that source to
        ///             build a binary for.  This is a convenience for source code that may not be stored
        ///             in files (e.g. dynamically generated code). On success the function will create
        ///             platform- specific binary and module objects that can be later used by the
        ///             runtime to invoke the shader code. The caller can provide a buffer for compiler
        ///             output, which if present, the runtime will fill *iff* the compilation fails.
        ///             
        ///             NB: Thread group dimensions are optional parameters here. This is because some
        ///             runtimes require them statically, and some do not. DirectX requires thread-group
        ///             sizes to be specified statically to enable compiler optimizations that cannot be
        ///             used otherwise. CUDA and OpenCL allow runtime specification of these parameters.
        ///             </remarks>
        ///
        /// <param name="lpszShaderCode">           [in] actual source. cannot be null. </param>
        /// <param name="uiShaderCodeSize">         Size of the shader code. </param>
        /// <param name="lpszOperation">            [in] Function name in source file. cannot be null. </param>
        /// <param name="ppPlatformSpecificBinary"> [out] On success, a platform specific binary. </param>
        /// <param name="ppPlatformSpecificModule"> [out] On success, a platform specific module handle. </param>
        /// <param name="lpszCompilerOutput">       (optional) [in,out] On failure, the compiler output. </param>
        /// <param name="uiCompilerOutput">         (optional) [in] length of buffer supplied for
        ///                                         compiler output. </param>
        /// <param name="nThreadGroupSizeX">        (optional) thread group X dimensions. (see remarks) </param>
        /// <param name="nThreadGroupSizeY">        (optional) thread group Y dimensions. (see remarks) </param>
        /// <param name="nThreadGroupSizeZ">        (optional) thread group Z dimensions. (see remarks) </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL        
        Compile(
            __in char *  lpszShaderCode, 
            __in UINT    uiShaderCodeSize,
            __in char *  lpszOperation, 
            __in void ** ppPlatformSpecificBinary,
            __in void ** ppPlatformSpecificModule,
            __in char *  lpszCompilerOutput,
            __in int     uiCompilerOutput,
            __in int     nThreadGroupSizeX, 
            __in int     nThreadGroupSizeY, 
            __in int     nThreadGroupSizeZ 
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this accelerator's device context is current. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   true if the context is current. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsDeviceContextCurrent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Makes the context current. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            MakeDeviceContextCurrent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the current context. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ReleaseCurrentDeviceContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the D3D feature level for the hardware 
        /// 			behind this accelerator object. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The feature level. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual D3D_FEATURE_LEVEL GetFeatureLevel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this accelerator has some support for device to device transfer
        /// 			with the given accelerator. This allows us to skip a trip through host memory
        /// 			in many cases.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsDeviceToDeviceTransfer(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Device to device transfer. </summary>
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
        DeviceToDeviceTransfer(
            __inout PBuffer *       pDstBuffer,
            __in    PBuffer *       pSrcBuffer,
            __in    AsyncContext *  pAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the supports device memcy. </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsDeviceMemcpy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the front-end programming model
        /// 			supports function arguments for top-level kernel
        /// 			invocations. DirectX requires
        /// 			top-level invocations to find their inputs
        /// 			at global scope in constant buffers and 
        /// 			*StructuredBuffers, etc. so this function 
        /// 			always returns false for this class.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   FALSE. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            SupportsFunctionArguments();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the underlying platform supports byval arguments
        /// 			for kernel invocations. If the platform does support this,
        /// 			PTask can elide explicit creation and population of 
        /// 			buffers to back these arguments, which is a performance
        /// 			win when it is actually supported. DirectX does not
        /// 			support this sort of thing so we always return false.</summary>
        /// 
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   FALSE </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            SupportsByvalArguments();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synchronizes the context.
        /// 			We could force a synchronization using 
        /// 			ID3D11Device functions (flush, end), but 
        /// 			there is no need because any attempt to reference
        /// 			output from a PTask executed by a DXAccelerator will
        /// 			force the completion of any predecessor operations.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="ctxt">     [in] non-null, the device ctxt. </param>
        /// <param name="pTask">    (optional) [in] If non-null, the task. </param>
        ///
        /// <returns>   true. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            Synchronize(Task*pTask=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check whether the given accelerator has a memory 
        /// 			space that is accessible from this accelerator without
        /// 			copying explictly through host memory space. Currently,
        /// 			CUDA interop APIs make it the case that we should be able
        /// 			to migrate between CUDA and DirectX devices without 
        /// 			necessarily going through the host. 
        /// 			TODO: take advantage of these APIs.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="p">    [in] non-null, a second accelerator. </param>
        ///
        /// <returns>   true if accessible memory space, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasAccessibleMemorySpace(Accelerator*p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the runtime for this accelerator
        /// 			supports pinned host memory. DirectX does not expose this
        /// 			functionality through the API, so we always return false
        /// 			from DXAccelerator.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   false </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            SupportsPinnedHostMemory();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate memory on the host. Some runtimes (esp. earlier versions of CUDA)
        ///             require that CUDA APIs be used to allocate host-side buffers, or support
        ///             specialized host allocators that can help improve DMA performance.
        ///             AllocatePagelockedHostMemory wraps these APIs for accelerators that have runtime support
        ///             for this, and uses normal system services (VirtualAlloc on Windows, malloc
        ///             elsewhere) to satisfy requests.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="uiBytes">              Number of bytes to allocate. </param>
        /// <param name="pbResultPageLocked">   [in,out] If non-null, the result of whether the
        /// 									allocated memory is page-locked is provided here. </param>
        ///
        /// <returns>   byte pointer on success, null on failure. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void *      
        AllocatePagelockedHostMemory(
            UINT uiBytes, 
            BOOL * pbResultPageLocked
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Free host memory. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="pBuffer">      If non-null, the buffer. </param>
        /// <param name="bPageLocked">  true if the memory was allocated in the page-locked area. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void        
        FreeHostMemory(
            void * pBuffer,
            BOOL bPageLocked
            );


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the adapter for this accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the adapter. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual IDXGIAdapter*   GetAdapter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the adapter description. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the adapter description. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual DXGI_ADAPTER_DESC* GetAdapterDesc();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerate accelerators present on the current machine
        /// 			and populate a vector with opened Accelerator objects.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="candidates">   [in] candidates list to populate </param>
        ///-------------------------------------------------------------------------------------------------

        static void             EnumerateAccelerators(std::vector<Accelerator*> &candidates);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a device. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAdapter">             [in,out] If non-null, the adapter. </param>
        /// <param name="DriverType">           Type of the driver. </param>
        /// <param name="Software">             external software rasterizer (always NULL!). </param>
        /// <param name="Flags">                creation flags to pass to DX runtime. </param>
        /// <param name="pFeatureLevels">       Acceptable DX feature levels list. </param>
        /// <param name="FeatureLevels">        Number of entries in feature levels list. </param>
        /// <param name="SDKVersion">           The sdk version. </param>
        /// <param name="ppDevice">             [out] If non-null, the device. </param>
        /// <param name="pFeatureLevel">        [out] If non-null, the feature level of the device </param>
        /// <param name="ppImmediateContext">   [out] If non-null, context for the device. </param>
        ///
        /// <returns>   HRESULT--use SUCCEEDED() or FAILED() macros</returns>
        ///-------------------------------------------------------------------------------------------------

        static HRESULT WINAPI   CreateDevice(
                                    IDXGIAdapter* pAdapter,
                                    D3D_DRIVER_TYPE DriverType,
                                    HMODULE Software,
                                    UINT32 Flags,
                                    CONST D3D_FEATURE_LEVEL* pFeatureLevels,
                                    UINT FeatureLevels,
                                    UINT32 SDKVersion,
                                    ID3D11Device** ppDevice,
                                    D3D_FEATURE_LEVEL* pFeatureLevel,
                                    ID3D11DeviceContext** ppImmediateContext 
                                    );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this accelerator encapsulates a backend framework that provides
        ///             explicit APIs for managing outstanding (Asynchronous) operations. When this is
        ///             the case, the corresponding AsyncContext subclass can manage outstanding
        ///             dependences explicitly to increase concurrency and avoid syncing with the device.
        ///             When it is *not* the case, we must synchronize when we data to and from this
        ///             accelerator context and contexts that *do* support an explicit async API. For
        ///             example, CUDA supports the stream and event API to explicitly manage dependences
        ///             and we use this feature heavily to allow task dispatch to get far ahead of device-
        ///             side dispatch. However when data moves between CUAccelerators and other
        ///             accelerator classes, we must use synchronous operations or provide a way to wait
        ///             for outstanding dependences from those contexts to resolve. This method is used
        ///             to tell us whether we can create an outstanding dependence after making calls
        ///             that queue work, or whether we need to synchronize.
        ///             
        ///             The function is not abstract because most accelerator classes don't support async
        ///             operations yet. In DirectX it is unnecessary because the DX runtime manages these
        ///             dependences under the covers, and in OpenCL the API is present, but we do not
        ///             yet take advantage of it.  So it's simpler to override a default implementation
        ///             that returns FALSE.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsExplicitAsyncOperations(); 

    protected:
        /// <summary> A cache of compiled shader objects</summary>
        DXCodeCache *			m_pCache;
        
        
        /// <summary> The ID3D11Device for this accelerator </summary>
        ID3D11Device*           m_pDevice;
        
        /// <summary> The device context for this accelerator</summary>
        ID3D11DeviceContext*    m_pContext;        
        
        /// <summary> The 3d feature level of the backing device</summary>
        D3D_FEATURE_LEVEL		m_d3dFeatureLevel;
                
        /// <summary> The adapter backing this device</summary>
        IDXGIAdapter *          m_pAdapter;
        
        /// <summary> The description of the adapter provided by the OS</summary>
        DXGI_ADAPTER_DESC       m_desc;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find a shader file to compile. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="strDestPath"> shader file path </param>
        /// <param name="cchDest"> size of path buffer </param>
        /// <param name="strDestPath"> file name </param>
        ///
        /// <returns>   The found dxsdk shader file cch. </returns>
        ///-------------------------------------------------------------------------------------------------

        static HRESULT 
            FindDXSDKShaderFileCch( 
                __in_ecount(cchDest) WCHAR* strDestPath,
                int cchDest, 
                __in LPCWSTR strFilename );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a new platform specific buffer. This routine is called by CreateBuffer to
        ///             get a new instance of whatever buffer type corresponds to the platform
        ///             implementing this interface. For example, DXAccelerator will return a new
        ///             PDXBuffer object, where PDXBuffer is a subclass of PBuffer. The Accelerator super-
        ///             class can then perform the rest of the work required to initialize the PBuffer.
        ///             
        ///             We only create PBuffers to provide 'physical' views of the 'logical' buffer
        ///             abstraction provided by the Datablock. Datablocks can have up to three different
        ///             channels (data, metadata, template), so consequently, each of which must be
        ///             backed by its own PBuffer. A PBuffer should not have to know what channel it is
        ///             backing, but we include that information in it's creation to simplify the
        ///             materialization of views between different subclasses of PBuffer.
        ///             
        ///             The "proxy allocator" is present as parameter to handle two corner cases:
        ///             
        ///             1. Allocation of host-side buffers by the host-specific subclass of PBuffer
        ///                (PHBuffer)--for example, we prefer to use a CUDA accelerator object to
        ///                allocate host memory when a block will be touched by a CUDA-based PTask,
        ///                because we can use the faster async APIs with memory we allocate using CUDA
        ///                host allocation APIs. This requires that the HostAccelerator defer the host-
        ///                side memory allocation to the CUDA accelerator.
        ///             
        ///             2. Communication between runtimes that provide some interop support (e.g. CUDA
        ///                and DirectX can actually share texture objects, meaning there is no need to
        ///                actually allocate a new buffer to back a CUDA view that already has a DirectX
        ///                view, but the two accelerators must cooperate to assemble a PBuffer that
        ///                shares the underlying shared object.
        ///             
        ///             Case 1 is implemented, while case 2 is largely unimplemented. If no proxy
        ///             accelerator is provided, allocation will proceed using the accelerator object
        ///             whose member function is being called to allocate the PBuffer.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pLogicalParent">           [in] If non-null, the datablock that is the logical
        ///                                         buffer using this 'physical' buffer to back a particular
        ///                                         channel on this accelerator. </param>
        /// <param name="nDatblockChannelIndex">    Zero-based index of the channel being backed. Must be:
        ///                                         * DBDATA_IDX = 0, OR
        ///                                         * DBMETADATA_IDX = 1, OR
        ///                                         * DBTEMPLATE_IDX = 2. </param>
        /// <param name="uiBufferAccessFlags">      Access flags determining what views to create. </param>
        /// <param name="pProxyAllocator">          [in,out] If non-null, the proxy allocator. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PBuffer*	NewPlatformSpecificBuffer(Datablock * pLogicalParent, 
                                                      UINT nDatblockChannelIndex, 
                                                      BUFFERACCESSFLAGS uiBufferAccessFlags, 
                                                      Accelerator * pProxyAllocator
                                                      );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Compile with macros. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="lpszShaderCode">           [in] filename+path of source. cannot be null. </param>
        /// <param name="uiShaderCodeSize">         Size of the shader code. </param>
        /// <param name="lpszOperation">            [in] Function name in source file. cannot be null. </param>
        /// <param name="ppPlatformSpecificBinary"> [out] On success, a platform specific binary. </param>
        /// <param name="ppPlatformSpecificModule"> [out] On success, a platform specific module handle. </param>
        /// <param name="lpszCompilerOutput">       (optional) [in,out] On failure, the compiler output. </param>
        /// <param name="uiCompilerOutput">         (optional) [in] length of buffer supplied for
        ///                                         compiler output. </param>
        /// <param name="pMacroDefs">               (optional) the macro defs. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///
        /// ### <param name="uiCompilerOutput"> (optional) the compiler output. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            
        CompileWithMacros(
            __in char *                  lpszShaderCode, 
            __in UINT                    uiShaderCodeSize,                                    
            __in char *                  lpszOperation, 
            __out void **                ppPlatformSpecificBinary,
            __out void **                ppPlatformSpecificModule,
            __inout char *               lpszCompilerOutput,
            __in int                     uiCompilerOutput,
            __in const void *            pMacroDefs=NULL // const D3D_SHADER_MACRO* 
            );

    private:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Warmup pipeline. </summary>
        ///
        /// <remarks>   Crossbac, 1/28/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void WarmupPipeline();

        /// <summary>   true to enable, false to disable code paths that
        ///             directly leverage direct x asyncrony. </summary>
        static BOOL     s_bEnableDirectXAsyncrony;

        /// <summary>   true to enable, false to disable code paths that
        ///             try to use resource sharing support in DX11. </summary>
        static BOOL     s_bEnableDirectXP2PAPIs;

    };

};
#endif
