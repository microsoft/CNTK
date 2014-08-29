//--------------------------------------------------------------------------------------
// File: cuaccelerator.h
// cuda-based accelerator
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _CUDA_ACCELERATOR_H_
#define _CUDA_ACCELERATOR_H_
#ifdef CUDA_SUPPORT 

#include "primitive_types.h"
#include "accelerator.h"
#include "task.h"
#include "cuhdr.h"
#include <vector>
#include <set>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	CUDA specific device attributes. </summary>
    ///
    /// <remarks>	Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct cudevparms_t {

		/// <summary> The major version number for the device</summary>
        int major;
        
		/// <summary> The minor version number for the device</summary>
        int minor;

		/// <summary> Device id returned by cuda initialization </summary>
        CUdevice dev;

        /// <summary> The driver version </summary>
        int driverVersion;
        
		/// <summary> The total global memory in MB</summary>
        size_t totalGlobalMem;
        
		/// <summary> Number of multi processors </summary>
        int multiProcessorCount;
        
		/// <summary> The total constant memory in KB</summary>
        int totalConstantMemory;

        /// <summary> The shared memory per block </summary>
        int sharedMemPerBlock;

        /// <summary> The number of registers per block </summary>
        int regsPerBlock;

        /// <summary> Size of the warp </summary>
        int warpSize;

        /// <summary> The maximum threads per block </summary>
        int maxThreadsPerBlock;

        /// <summary> The maximum block dimensions </summary>
        int maxBlockDim[3];

        /// <summary> The maximum grid dimensions </summary>
        int maxGridDim[3];

        /// <summary> The memory pitch </summary>
        int memPitch;

        /// <summary> The texture alignment </summary>
        int textureAlign;

        /// <summary> The clock rate </summary>
        int clockRate;
        
		/// <summary> True if the device can overlap gpu 
		/// 		  computation with data transfer
		/// 		  </summary>
        int gpuOverlap;

        /// <summary> True if kernel execute timeout is enabled </summary>
        int kernelExecTimeoutEnabled;

        /// <summary> True if the device is integrated, 
        /// 		  false if the device is connected on PCIe
        /// 		  </summary>
        int integrated;

        /// <summary> True if the runtime can map host memory
        /// 		  for data transfers to/from this device
        /// 		  </summary>
        int canMapHostMemory;

        /// <summary> True if the device supports 
        /// 		  concurrent execution of multiple different 
        /// 		  kernels 
        /// 		  </summary>
        int concurrentKernels;

        /// <summary> True if ecc is enabled for the device memory </summary>
        int eccEnabled;

        /// <summary> The if the tcc driver is in use for this device </summary>
        int tccDriver;

        /// <summary>   True if the device supports unified addressing. 
        /// 			Unified addressing means device and host pointers are equal
        /// 			for page-locked host-allocations. 
        /// 			</summary>
        int unifiedAddressing;

        /// <summary> Name of the device </summary>
        char deviceName[256];

    } CUDA_DEVICE_ATTRIBUTES;

    static const int MAXCTXTS = 16;
    static const int MAXCTXDEPTH = 32;

	class CUAccelerator : 
        public Accelerator        
    {
	public:

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Constructor. </summary>
		///
		/// <remarks>	Crossbac, 12/23/2011. </remarks>
		///
		/// <param name="attribs">	[in,out] If non-null, the attributes. </param>
		///-------------------------------------------------------------------------------------------------

		CUAccelerator(CUDA_DEVICE_ATTRIBUTES * attribs);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Destructor. </summary>
		///
		/// <remarks>	Crossbac, 12/23/2011. </remarks>
		///-------------------------------------------------------------------------------------------------

		virtual ~CUAccelerator();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Open the CUAccelerator. </summary>
		///
		/// <remarks>	Crossbac, 12/23/2011. </remarks>
		///
		/// <returns>	HRESULT--use SUCCEEDED() and FAILED() macros to check. </returns>
		///-------------------------------------------------------------------------------------------------

		HRESULT					Open();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Opens a CUAccelerator object for the CUDA
		/// 			device specified. </summary>
		///
		/// <remarks>	Crossbac, 12/23/2011. </remarks>
		///
		/// <param name="dev">	The device id. </param>
		///
		/// <returns>	HRESULT--use SUCCEEDED() and FAILED() macros to check. </returns>
		///-------------------------------------------------------------------------------------------------

		HRESULT					Open(CUdevice dev);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Gets the device. </summary>
		///
		/// <remarks>	Crossbac, 12/23/2011. </remarks>
		///
		/// <returns>	null if it fails, else the device. </returns>
		///-------------------------------------------------------------------------------------------------

		void*					GetDevice();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Gets the context. </summary>
		///
		/// <remarks>	Crossbac, 12/23/2011. </remarks>
		///
		/// <returns>	null if it fails, else the context. </returns>
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
        /// <summary>   Cache the shader and module objects associated with 
        /// 			successful compilation of szFunction in szFile. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="szFile">           [in] non-null, the file name. </param>
        /// <param name="szFunction">       [in] non-null, the function. </param>
        /// <param name="pCUDAFunction">    The cuda function. </param>
        /// <param name="pCUDAModule">      The cuda module. </param>
        ///-------------------------------------------------------------------------------------------------

		void					CachePutShader(char * szFile, 
											   char * szFunction, 
											   CUfunction pCUDAFunction, 
											   CUmodule pCUDAModule
											   );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check the cache for a compiled version of the
        /// 			function szFunction in the file szFile. If it
        /// 			is present, compilation can be skipped. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="szFile">           [in] non-null, the file name. </param>
        /// <param name="szFunction">       [in] non-null, the function. </param>
        /// <param name="pCUDAFunction">    [out] The cuda function. </param>
        /// <param name="pCUDAModule">      [out] The cuda module. </param>
        ///
        /// <returns>   true if the shader is present in the cache, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

		BOOL    				CacheGetShader(char * szFile, 
											   char * szFunction, 
                                               CUfunction &pCUDAFunction, 
                                               CUmodule &pCUDAModule
                                               );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Compiles CUDA code to create a new binary 
        /// 			and module.  </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpszFileName">             [in,out] If non-null, filename of the file. </param>
        /// <param name="lpszOperation">            [in,out] If non-null, the operation. </param>
        /// <param name="ppPlatformSpecificBinary"> [in,out] If non-null, the platform specific binary. </param>
        /// <param name="ppPlatformSpecificModule"> [in,out] If non-null, the platform specific module. </param>
        /// <param name="lpszCompilerOutput">       (optional) [in,out] If non-null, the compiler output. </param>
        /// <param name="uiCompilerOutput">         (optional) the compiler output. </param>
        /// <param name="nThreadGroupSizeX">        (optional) the thread group size x coordinate. </param>
        /// <param name="nThreadGroupSizeY">        (optional) the thread group size y coordinate. </param>
        /// <param name="nThreadGroupSizeZ">        (optional) The thread group size z coordinate. </param>
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
                                    int nThreadGroupSizeX=1, 
                                    int nThreadGroupSizeY=1, 
                                    int nThreadGroupSizeZ=1
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

        virtual BOOL        
        Compile(
            __in char *  lpszShaderCode, 
            __in UINT    uiShaderCodeSize,
            __in char *  lpszOperation, 
            __in void ** ppPlatformSpecificBinary,
            __in void ** ppPlatformSpecificModule,
            __in char *  lpszCompilerOutput=NULL,
            __in int     uiCompilerOutput=0,
            __in int     nThreadGroupSizeX=1, 
            __in int     nThreadGroupSizeY=1, 
            __in int     nThreadGroupSizeZ=1 
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the context of this accelerator is current. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsDeviceContextCurrent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Makes the context current. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            MakeDeviceContextCurrent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the current context. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ReleaseCurrentDeviceContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synchronizes the context. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="ctxt">     [in,out] If non-null, the ctxt. </param>
        /// <param name="pTask">    (optional) [in,out] If non-null, the task. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            Synchronize(Task*pTask=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the cuda runtime has been initialized. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if cuda initialized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL             IsCUDAInitialized();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a cuda initialized. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="bCUDAInitialized"> true to indicate CUinit has been called. </param>
        ///-------------------------------------------------------------------------------------------------

        static void             SetCUDAInitialized(BOOL bCUDAInitialized); 

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
        /// <summary>   Return true if this accelerator has support for unified addressing. Unified
        ///             addressing means there is no distinction between device and host pointers (for
        ///             page-locked memory). This is important because the datablock abstraction
        ///             maintains a buffer per logical memory space, and if two memory spaces are
        ///             logically the same (unified), but only for pointers to page-locked memory, a
        ///             number of special cases arise for allocation, freeing, ownership, etc. Sadly,
        ///             this complexity is required in the common case, because asynchronous transfers
        ///             only work in CUDA when the host pointers are page-locked. We need to be able to
        ///             tell when a page-locked buffer in the host-memory space is different from a
        ///             device pointer in a CUAccelerator memory space.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        /// 
        /// <returns>   true if the device supports unified addressing. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsUnifiedAddressing();

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
        /// <summary>   Gets the supports device memcy. </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsDeviceMemcpy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this accelerator supports top-level 
        /// 			function arguments. This will always return true
        /// 			for CUDA accelerators.</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            SupportsFunctionArguments();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this platform supports passing 
        /// 			structs by value as arguments to top-level kernel
        /// 			entry points. This will always return true for
        /// 			CUDA accelerators. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            SupportsByvalArguments();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pOtherAccelerator' has an accessible memory space. 
        /// 			The other accelerator's memory space is accessible if there 
        /// 			is a way to transfer data between the two other than by 
        /// 			copying to host-memory as a waypoint. For example, some
        /// 			CUDA accelerators support peer-to-peer copy over PCI, 
        /// 			and DirectX has interop APIs with CUDA.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pOtherAccelerator">    [in,out] If non-null, the other accelerator. </param>
        ///
        /// <returns>   true if accessible memory space, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasAccessibleMemorySpace(Accelerator * pOtherAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the accelerator supports pinned host memory. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
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

        virtual void * AllocatePagelockedHostMemory(UINT uiBytes, BOOL * pbResultPageLocked);

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
        /// <summary>   Gets the device identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The device identifier. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual int             GetDeviceId();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the device attributes. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the device attributes. </returns>
        ///-------------------------------------------------------------------------------------------------

        CUDA_DEVICE_ATTRIBUTES* GetDeviceAttributes();

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
        ///             This override returns TRUE since this is the CUDA encapsulation class.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsExplicitAsyncOperations(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerate accelerators. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="devices">  [out] non-null, the acclerator objects supporting CUDA. </param>
        ///-------------------------------------------------------------------------------------------------

        static void             EnumerateAccelerators(std::vector<Accelerator*> &devices);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the thread local context. </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///
        /// <param name="eRole">        The role. </param>
        /// <param name="bMakeDefault"> Device is the default for the thread. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        InitializeTLSContextManagement(
            __in Accelerator * pDefaultAccelerator,
            __in PTTHREADROLE eRole,
            __in BOOL bPooledThread
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitializes the thread local context. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2014. </remarks>
        ///
        /// <param name="eRole">    The role. </param>
        ///-------------------------------------------------------------------------------------------------

        static void DeinitializeTLSContextManagement();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if we can requires thread local context initialization. </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL UsesTLSContextManagement();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the thread local context. </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///
        /// <param name="eRole">        The role. </param>
        /// <param name="bMakeDefault"> This device should be the default for the thread. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        InitializeTLSContext(
            __in PTTHREADROLE eRole,
            __in BOOL bMakeDefault,
            __in BOOL bPooledThread
            );

	protected:

        void CheckContextInvariants();
        void CheckContextTLSInitialized();

        /// <summary> true to cuda initialized </summary>
        static BOOL             s_bCUDAInitialized;

        /// <summary>   context of primary device. This is essentially the "primary" CUDA context, but
        ///             might not actually be the primary if user code is also managing device contexts
        ///             or using cuda runtime API calls.
        ///             </summary>
        static CUcontext        s_pRootContext;

        /// <summary> true if the root context is valid </summary>
        static BOOL             s_bRootContextValid;

        /// <summary>   device id for the root context. </summary>
        static CUdevice         s_nRootContext;

        /// <summary>   known ptask contexts. these should be created
        ///             at init time in single threaded code (Scheduler::CreateAccelerators),
        ///             so we consider them immutable once the runtime is going, so
        ///             we needn't synchronize them or make them TLS. </summary>
        static CUcontext        s_vKnownPTaskContexts[MAXCTXTS];
        static UINT             s_nKnownPTaskContexts;
        static CUcontext        s_vKnownUserContexts[MAXCTXTS];
        static UINT             s_nKnownUserContexts;

        /// <summary>   Thread-local storage for caching device contexts,
        ///             enabling some heuristics to avoid unnecessary and occasionally
        ///             expensive calls to cuCtx[Push|Pop]Current. 
        ///             Additional book-keeping is necessary to keep track of 
        ///             contexts that don't belong to us (e.g. "primary" contexts
        ///             created in user code as a side effect of invoking cuda runtime
        ///             apis like cudaFree()).
        ///             </summary>
        __declspec(thread) static CUAccelerator *  s_pDefaultDeviceCtxt;
        __declspec(thread) static CUAccelerator *  s_pCurrentDeviceCtxt;
        __declspec(thread) static int              s_vContextDepthMap[MAXCTXTS];
        __declspec(thread) static CUAccelerator ** s_pContextChangeMap[MAXCTXTS];
        __declspec(thread) static CUAccelerator *  s_vContextChangeMap[MAXCTXTS*MAXCTXDEPTH];
        __declspec(thread) static CUcontext        s_pUserStackTop;
        __declspec(thread) static BOOL             s_bContextTLSInit;
        __declspec(thread) static BOOL             s_bThreadPoolThread;
        __declspec(thread) static PTTHREADROLE     s_eThreadRole;

        static BOOL IsKnownContext(CUcontext ctx); 
        static BOOL IsKnownContext(CUcontext ctx, CUcontext * pContexts, UINT uiCtxCount); 
        static BOOL AddKnownContext(CUcontext ctx, CUcontext * pContexts, UINT * puiCtxCount);
        static BOOL IsUserContext(CUcontext ctx);
        static BOOL IsPTaskContext(CUcontext ctx);
        static BOOL IsKnownPTaskContext(CUcontext ctx);
        static BOOL IsKnownUserContext(CUcontext ctx);
        static BOOL AddKnownPTaskContext(CUcontext ctx);
        static BOOL AddKnownUserContext(CUcontext ctx);
        static BOOL CheckContextProvenance(CUcontext ctx);
		
        /// <summary> The device </summary>
        CUdevice				m_pDevice;
		
        /// <summary> The context </summary>
        CUcontext				m_pContext;

        /// <summary>   true if this is also an application-level primary context. 
        ///             This means that PTask shares it with user code, does not
        ///             own the context, and cannot make assumptions about context
        ///             state on entry to PTask APIs on *application* threads.
        ///             </summary>
        BOOL                    m_bApplicationPrimaryContext;
        
        /// <summary> Identifier for the device </summary>
        int                     m_nDeviceId;

        /// <summary> The device attributes </summary>
        CUDA_DEVICE_ATTRIBUTES *m_pDeviceAttributes;

        /// <summary> The attributes </summary>
        CUDA_DEVICE_ATTRIBUTES  m_attrs;

        /// <summary>   The set of accelerators that are known accessible for P2P transfers. </summary>
        std::set<Accelerator*>  m_vP2PAccessible;

        /// <summary>   The set of accelerators that are known enabled for P2P transfers. </summary>
        std::set<Accelerator*>  m_vP2PEnabled;
        
        /// <summary>   The set of accelerators that are *known inaccessible* for P2P transfers. </summary>
        std::set<Accelerator*>  m_vP2PInaccessible;

        /// <summary>   The minimum stream priority. </summary>
        int                     m_nMinStreamPriority;

        /// <summary>   The maximum stream priority. </summary>
        int                     m_nMaxStreamPriority;

        /// <summary>   The maximum outstading launches. </summary>
        int                     m_nMaxOutstadingLaunches;

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
        /// <summary>   Determine if we should attempt page locked allocation. </summary>
        ///
        /// <remarks>   Crossbac, 9/24/2012. </remarks>
        ///
        /// <param name="uiAllocBytes"> The allocate in bytes. </param>
        ///
        /// <returns>   true if we should page-lock the requested buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL ShouldAttemptPageLockedAllocation(UINT uiAllocBytes);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determine if we can access a peer device through explicit peer APIs. 
        ///             Cache the result. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if we can access peer, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL CanAccessPeer(Accelerator * pAccelerator);

        friend class PCUBuffer;
	};
};
#endif
#endif

