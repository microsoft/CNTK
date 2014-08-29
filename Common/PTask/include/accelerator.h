///-------------------------------------------------------------------------------------------------
// file:	accelerator.h
//
// summary:	Declares the accelerator class, which is an abstract
//          superclass encapsulating different accelerator front-ends. 
///-------------------------------------------------------------------------------------------------

#ifndef _ACCELERATOR_H_
#define _ACCELERATOR_H_

#include "primitive_types.h"
#include "Lockable.h"
#include "PBuffer.h"
#include <map>
#include <vector>

namespace PTask {

    static const UINT USER_OUTPUT_BUF_SIZE = 512;

    // forward decls
    class Datablock;
    class CompiledKernel;
    class PhysicalDevice;
    class DatablockTemplate;
    class PBuffer;
    class Task;
    class MemorySpace;
    class AllocationStatistics;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Function signature for callbacks used to cleanup TLS device context
    ///             management data structures used by subclasses of Accelerator. 
    ///-------------------------------------------------------------------------------------------------

    typedef void (_cdecl *LPFNTLSCTXTDEINITIALIZER)(void);
    const int MAXTLSCTXTCALLBACKS = 32; // arbitrary

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Accelerator abstract superclass. </summary>
    ///
    /// <remarks>   Crossbac, 12/29/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class Accelerator : public Lockable
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        Accelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Accelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synchronizes all default contexts for accelerators supporting async
        ///             operations. </summary>
        ///
        /// <remarks>   crossbac, 6/26/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void SynchronizeDefaultContexts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitializes any TLS data structures installed on the calling thread
        ///             by accelerator subclasses (in InitializedTLSContextManagement). 
        ///             Note that these data structures are *per-thread*, not per accelerator. 
        ///             Repeated calls are idempotent--it is safe to call this method on a thread
        ///             for which it has already been called. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void DeinitializeTLSContextManagement();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the thread local contexts. 
        ///             This static version traverses all available accelerator
        ///             objects, and for those whose sub-class actually uses TLS device-context
        ///             management data structures, it uses the object to call the virtual
        ///             InitializeTLSContext function. We use this technique for getting to
        ///             sub-class implementations, but TLS initialization is *per-thread*, not per
        ///             accelerator. 
        ///             
        ///             Subclasses that do setup TLS structures are required to install
        ///             a cleanup callback as part of that initialization so that the 
        ///             DeinitializeTLSContextManagement function can cleanup without having
        ///             to find live accelerator objects of the proper class to find the proper
        ///             cleanup function.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///
        /// <param name="eRole">                (Optional) the role. </param>
        /// <param name="uiThreadRoleIndex">    (Optional) zero-based index of the thread role. </param>
        /// <param name="uiThreadRoleCount">    (Optional) number of thread roles. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        InitializeTLSContextManagement(
            __in PTTHREADROLE eRole,
            __in UINT uiThreadRoleIndex,
            __in UINT uiThreadRoleCount,
            __in BOOL bPooledThread
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determine if we should have default context. </summary>
        ///
        /// <remarks>   crossbac, 6/20/2014. </remarks>
        ///
        /// <param name="eRole">    The role. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL ShouldHaveDefaultContext(PTTHREADROLE eRole);

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

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if we can requires thread local context initialization. </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL UsesTLSContextManagement();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a PBuffer: platform specific buffer encapsulating buffer management on
        ///             the accelerator side. Each subclass of Accelerator will also implement a subclass
        ///             of PBuffer to provide a uniform interface on buffer management. The buffer is
        ///             considered to be either:
        ///             
        ///             1) A 3-dimensional array of fixed-stride elements, 2) An undimensioned extent of
        ///             byte-addressable memory.
        ///             
        ///             Buffer geometry is inferred from the datablock which is the logical buffer the
        ///             resulting PBuffer will back on this accelerator.
        ///             
        ///             The proxy accelerator is provided to handle interoperation corner between
        ///             different types of accelerator objects, where deferring buffer allocation to the
        ///             proxy accelerator can enable better performance the parent is backed on both
        ///             accelerator types. See documentation for NewPlatformSpecificBuffer for more
        ///             details.
        ///             
        ///             This method is implemented by the PBuffer superclass, and actual allocation of
        ///             the PBuffer subtype is deferred to NewPlatformSpecificBuffer.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="pAsyncContext">            [in,out] (optional)  If non-null, context for the
        ///                                         asynchronous. </param>
        /// <param name="pLogicalBufferParent">     [in] If non-null, the datablock that is the logical
        ///                                         buffer using this 'physical' buffer to back a
        ///                                         particular channel on this accelerator. </param>
        /// <param name="nDatablockChannelIndex">   Access flags determining what views to create. </param>
        /// <param name="eBufferAccessFlags">       Zero-based index of the channel being backed. Must be:
        ///                                         * DBDATA_IDX = 0, OR
        ///                                         * DBMETADATA_IDX = 1, OR
        ///                                         * DBTEMPLATE_IDX = 2. </param>
        /// <param name="pProxyAccelerator">        [in,out] If non-null, the proxy allocator. </param>
        /// <param name="pInitialBufferContents">   Number of elements in Z dimension. </param>
        /// <param name="strDebugName">             (optional) [in] If non-null, the a name for the
        ///                                         object (helps with debugging). </param>
        /// <param name="bByteAddressable">         (optional) Specifies whether the created buffer
        ///                                         should be geometry-less ("raw") or not. This flag is
        ///                                         not required for all platforms: for example, DirectX
        ///                                         buffers must created with special descriptors if the
        ///                                         HLSL code accessing them is going to use byte-
        ///                                         addressing. This concept is absent from CUDA and
        ///                                         OpenCL. </param>
        ///
        /// <returns>   null if it fails, else a new PBUffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PBuffer*	
        CreateBuffer( 
            __in AsyncContext * pAsyncContext,
            __in Datablock * pLogicalBufferParent,
            __in UINT nDatablockChannelIndex, 
            __in BUFFERACCESSFLAGS eBufferAccessFlags, 
            __in Accelerator * pProxyAccelerator,
            __in HOSTMEMORYEXTENT * pInitialBufferContents=NULL, 
            __in char * strDebugName=NULL, 
            __in bool bByteAddressable=false
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a pagelocked PBuffer: platform specific buffer encapsulating buffer management on
        ///             the accelerator side. Each subclass of Accelerator will also implement a subclass
        ///             of PBuffer to provide a uniform interface on buffer management. The buffer is
        ///             considered to be either:
        ///             
        ///             1) A 3-dimensional array of fixed-stride elements, 2) An undimensioned extent of
        ///             byte-addressable memory.
        ///             
        ///             Buffer geometry is inferred from the datablock which is the logical buffer the
        ///             resulting PBuffer will back on this accelerator.
        ///             
        ///             The proxy accelerator is provided to handle interoperation corner between
        ///             different types of accelerator objects, where deferring buffer allocation to the
        ///             proxy accelerator can enable better performance the parent is backed on both
        ///             accelerator types. See documentation for NewPlatformSpecificBuffer for more
        ///             details.
        ///             
        ///             This method is implemented by the PBuffer superclass, and actual allocation of
        ///             the PBuffer subtype is deferred to NewPlatformSpecificBuffer.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="pAsyncContext">            [in,out] (optional)  If non-null, context for the
        ///                                         asynchronous. </param>
        /// <param name="pLogicalBufferParent">     [in] If non-null, the datablock that is the logical
        ///                                         buffer using this 'physical' buffer to back a
        ///                                         particular channel on this accelerator. </param>
        /// <param name="nDatablockChannelIndex">   Access flags determining what views to create. </param>
        /// <param name="eBufferAccessFlags">       Zero-based index of the channel being backed. Must be:
        ///                                         * DBDATA_IDX = 0, OR
        ///                                         * DBMETADATA_IDX = 1, OR
        ///                                         * DBTEMPLATE_IDX = 2. </param>
        /// <param name="pProxyAccelerator">        [in,out] If non-null, the proxy allocator. </param>
        /// <param name="pInitialBufferContents">   Number of elements in Z dimension. </param>
        /// <param name="strDebugName">             (optional) [in] If non-null, the a name for the
        ///                                         object (helps with debugging). </param>
        /// <param name="bByteAddressable">         (optional) Specifies whether the created buffer
        ///                                         should be geometry-less ("raw") or not. This flag is
        ///                                         not required for all platforms: for example, DirectX
        ///                                         buffers must created with special descriptors if the
        ///                                         HLSL code accessing them is going to use byte-
        ///                                         addressing. This concept is absent from CUDA and
        ///                                         OpenCL. </param>
        ///
        /// <returns>   null if it fails, else a new PBUffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PBuffer*	
        CreatePagelockedBuffer( 
            __in AsyncContext * pAsyncContext,
            __in Datablock * pLogicalBufferParent,
            __in UINT nDatablockChannelIndex, 
            __in BUFFERACCESSFLAGS eBufferAccessFlags, 
            __in Accelerator * pProxyAccelerator,
            __in HOSTMEMORYEXTENT * pInitialBufferContents=NULL, 
            __in char * strDebugName=NULL, 
            __in bool bByteAddressable=false
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Open an accelerator object. Create any device contexts, initialize backend
        ///             runtimes. At the end of this call it should be possible to use this accelerator
        ///             to compile and dispatch PTasks.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   S_OK on success or E_FAIL. Use Windows "SUCCEEDED()" and "FAILED()" macros to
        ///             check the return status. A failed open will not de-allocate the accelerator
        ///             object, obviously, but the object should be deleted by the caller on failure,
        ///             since behavior of its methods are undefined in this circumstance.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT		Open()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return a pointer to the platform-specific device object. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the device. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void*		GetDevice()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return a pointer to the platform-specific device context. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the context. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void*		GetContext()=0;

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
        CreateAsyncContext(
            __in Task * pTask,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   destroys the asynchronous context for the task. Since accelerators keep a reference to
        ///             all async contexts, along with some maps, the demise of a task must be visible to the
        ///             accelerator objects for which it has created task async contexts.
        ///             </summary>
        ///
        /// <param name="pTask">                [in] non-null, the CUDA-capable acclerator to which the
        ///                                     stream is bound. </param>
        /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        ReleaseTaskAsyncContext(
            __in Task * pTask,
            __in AsyncContext * pAsyncContext
            );
        
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the accelerator object is initialized and usable by the caller.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   true if initialized. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL		Initialized();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is enabled. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   true if enabled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        IsEnabled();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is a host accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   true if enabled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        IsHost();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables/disables the accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="bEnable">  true to enable, false to disable. </param>
        ///
        /// <returns>   success/failure </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT    Enable(BOOL bEnable);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform-specific subclass of this accelerator object. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   The ACCELERATOR_CLASS. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual ACCELERATOR_CLASS GetClass();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Compiles accelerator source code to create a PTask binary. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011.
        ///             
        ///             The function accepts a file name and an operation in the file to build a binary
        ///             for. For example, "foo.hlsl" and "vectoradd" will compile the vectoradd() shader
        ///             in foo.hlsl. On success the function will create platform- specific binary and
        ///             module objects that can be later used by the runtime to invoke the shader code.
        ///             The caller can provide a buffer for compiler output, which if present, the
        ///             runtime will fill *iff* the compilation fails.
        ///             
        ///             NB: Thread group dimensions are optional parameters here. This is because some
        ///             runtimes require them statically, and some do not. DirectX requires thread-group
        ///             sizes to be specified statically to enable compiler optimizations that cannot be
        ///             used otherwise. CUDA and OpenCL allow runtime specification of these parameters.
        ///             </remarks>
        ///
        /// <param name="lpszFileName">             [in] filename+path of source. cannot be null. </param>
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
            __in char * lpszFileName, 
            __in char * lpszOperation, 
            __in void ** ppPlatformSpecificBinary,
            __in void ** ppPlatformSpecificModule,
            __in char * lpszCompilerOutput=NULL,
            __in int uiCompilerOutput=0,
            __in int nThreadGroupSizeX=1, 
            __in int nThreadGroupSizeY=1, 
            __in int nThreadGroupSizeZ=1 
            )=0;

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
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synchronizes the device context. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="pTask">    [in,out] (optional)  If non-null, the task. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        Synchronize(Task*pTask=NULL)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this accelerator's device context is current. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   true if the context is current. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        IsDeviceContextCurrent()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Makes the context current. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        MakeDeviceContextCurrent()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the current context. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void        ReleaseCurrentDeviceContext()=0;

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

        virtual BOOL        SupportsDeviceToDeviceTransfer(Accelerator * pAccelerator)=0; 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the supports device memcy. </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsDeviceMemcpy()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the front-end programming model supports function arguments for
        ///             top-level kernel invocations. For example, CUDA and OpenCL allow code of the form:
        ///             
        ///               kernel(float *p, int a, int b)
        ///             
        ///             where all inputs are bound as formal parameters to a function call. DirectX, in
        ///             contrast, requires top-level invocations to find their inputs at global scope in
        ///             constant buffers and (*)StructuredBuffers, etc.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   true if it formal parameters are supported. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsFunctionArguments()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the underlying platform supports byval arguments for kernel
        ///             invocations. If the platform does support this, PTask can elide explicit creation
        ///             and population of buffers to back these arguments, which is a performance win
        ///             when it is actually supported.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   true if supported, false otherwise. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsByvalArguments()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check whether the given accelerator has a memory space that is accessible from
        ///             this accelerator without copying explictly through host memory space. For example,
        ///             Some Tesla cards support peer-to-peer copy across PCIe, some accelerators such as
        ///             Tegra, Larrabbee actually have coherent access to host memory. When one
        ///             accelerator consumes data that is most up to date on another accelerator must
        ///             effect the migration of that data, and in the general case must do so by copying
        ///             first to the host, and then from the host to the second accelerator. When a more
        ///             direct route is supported the runtime must take advantage of it.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="p">    [in] non-null, a second accelerator. </param>
        ///
        /// <returns>   true if accessible memory space, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        HasAccessibleMemorySpace(Accelerator*p)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the runtime for this accelerator supports pinned host memory. When
        ///             pinned host memory is available, the runtime can take advantage of it to avoid
        ///             explicitly backing device buffers with another host buffer. Of course, we have to
        ///             use this feature sparingly to avoid penalizing DMA performance.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   true if host pinned memory is supported, false otherwise. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsPinnedHostMemory()=0;

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

        virtual void * AllocatePagelockedHostMemory(UINT uiBytes, BOOL * pbResultPageLocked)=0;

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
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the device name. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the device name. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual char *      GetDeviceName();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the core count on the physical accelerator backing this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   The core count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        GetCoreCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the core clock rate on the physical accelerator backing this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   The core clock rate. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        GetCoreClockRate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform specific runtime version on the physical accelerator backing
        ///             this object.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   The platform specific runtime version. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        GetPlatformSpecificRuntimeVersion();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the global memory size on the physical accelerator backing this object.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   The global memory size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        GetGlobalMemorySize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the physical accelerator backing this object supports concurrent
        ///             kernels.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   true if it support is present, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        SupportsConcurrentKernels();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the core count. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="nCoreCount">   The core count. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void        SetCoreCount(UINT nCoreCount);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a core clock rate. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="nClockRate">   The clock rate. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void        SetCoreClockRate(UINT nClockRate);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a platform specific runtime version. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="nVersion"> The version. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void        SetPlatformSpecificRuntimeVersion(UINT nVersion);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a global memory size. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="nMemorySize">  The memory size in MB. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void        SetGlobalMemorySize(UINT nMemorySize);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the supports concurrent kernels. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="bSupportsConcurrentKernels">   true if the accelerator supports
        /// 											concurrent kernel execution. 
        /// 											</param>
        ///-------------------------------------------------------------------------------------------------

        virtual void        SetSupportsConcurrentKernels(BOOL bSupportsConcurrentKernels);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the physical device. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the physical device. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PhysicalDevice * GetPhysicalDevice();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a physical device. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="pPhysicalDevice">  [in,out] If non-null, the physical device. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void        SetPhysicalDevice(PhysicalDevice* pPhysicalDevice);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform index. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   The platform index. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        GetPlatformIndex();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a platform index. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="nPlatformIndex">   The platform index. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void        SetPlatformIndex(UINT nPlatformIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Less-than comparison operator. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="rhs">  The right hand side. </param>
        ///
        /// <returns>   true if the first parameter is less than the second. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual bool operator <(const Accelerator &rhs) const;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Select best accelerators. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <param name="nConcurrencyLimit">    The concurrency limit. </param>
        /// <param name="candidates">           [in] non-null, the candidates. </param>
        /// <param name="selected">             [out] non-null, the selected. </param>
        /// <param name="rejected">             [out] non-null, the rejected. </param>
        ///-------------------------------------------------------------------------------------------------

        static void         SelectBestAccelerators(
                                UINT nConcurrencyLimit,
                                std::vector<Accelerator*> &candidates,
                                std::vector<Accelerator*> &selected,
                                std::vector<Accelerator*> &rejected
                                );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerator id. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. </remarks>
        ///
        /// <returns>   The platform index. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        GetAcceleratorId();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the memory space identifier for this accelerator. For now, we do not attempt
        ///             to coalesce memory spaces based on whether they share the same physical device.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   The memory space identifier. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        GetMemorySpaceId();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the memory space identifier for this accelerator. For now, we do not attempt
        ///             to coalesce memory spaces based on whether they share the same physical device.
        ///             Memory space identifiers must be contiguous starting with zero, where zero is
        ///             reserved for host memory.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pMemorySpace">     [in,out] If non-null, the memory space. </param>
        /// <param name="uiMemorySpaceId">  Identifier for the memory space. </param>
        ///-------------------------------------------------------------------------------------------------

        void                SetMemorySpace(MemorySpace * pMemorySpace, UINT uiMemorySpaceId);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the asynchronous context corresponding to the given type of operation. </summary>
        ///
        /// <remarks>   Crossbac, 7/13/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the default asynchronous context. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual AsyncContext * GetAsyncContext(ASYNCCONTEXTTYPE eAsyncContextType);  

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the page locked allocation count. </summary>
        ///
        /// <remarks>   Crossbac, 9/24/2012. </remarks>
        ///
        /// <returns>   The page locked allocation count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual unsigned __int64 GetPageLockedAllocationCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the page locked allocation bytes. </summary>
        ///
        /// <remarks>   Crossbac, 9/24/2012. </remarks>
        ///
        /// <returns>   The page locked allocation bytes. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual unsigned __int64 GetPageLockedAllocationBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the page locked allocation count. </summary>
        ///
        /// <remarks>   Crossbac, 9/24/2012. </remarks>
        ///
        /// <returns>   The page locked allocation count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual unsigned __int64 GetTotalAllocationBuffers();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the page locked allocation bytes. </summary>
        ///
        /// <remarks>   Crossbac, 9/24/2012. </remarks>
        ///
        /// <returns>   The page locked allocation bytes. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual unsigned __int64 GetTotalAllocationBytes();
            
    protected:    

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
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory allocation. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">        [in,out] If non-null, extent of the memory. </param>
        /// <param name="bPinnedAllocation">    true to pinned allocation. </param>
        /// <param name="uiBytes">              The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        RecordAllocation(
            __in void * pMemoryExtent, 
            __in BOOL bPinnedAllocation, 
            __in size_t uiBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory deallocation. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">        [in,out] If non-null, extent of the memory. </param>
        /// <param name="bPinnedAllocation">    true to pinned allocation. </param>
        /// <param name="uiBytes">              The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        RecordDeallocation(
            __in void * pMemoryExtent
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets the allocation statistics. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ResetAllocationStatistics();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the allocation statistics. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DumpAllocationStatistics(std::ostream &ios);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the asynchronous contexts. </summary>
        ///
        /// <remarks>   Crossbac, 3/20/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ReleaseAsyncContexts();

        /// <summary> true if the accelerator object
        /// 		  was successfully initialized 
        /// 		  </summary>
        BOOL                    m_bInitialized;

        /// <summary> The actual accelerator class of the object
        /// 		  implementing this abstract interface
        /// 		  </summary>
        ACCELERATOR_CLASS       m_class;
    
        /// <summary> Unique identifier for the accelerator </summary>
        UINT                    m_uiAcceleratorId;

        /// <summary> Identifier for the memory space associated with this accelerator. </summary>
        UINT                    m_uiMemorySpaceId;

        /// <summary>   The memory space. </summary>
        MemorySpace *           m_pMemorySpace;
    
        /// <summary> Number of cores </summary>
        UINT					m_nCoreCount;
    
        /// <summary> The runtime version </summary>
        UINT                    m_nRuntimeVersion;
    
        /// <summary> Size of the memory </summary>
        UINT                    m_nMemorySize;
    
        /// <summary> The clock rate </summary>
        UINT                    m_nClockRate;
    
        /// <summary> Zero-based index of the m n platform </summary>
        UINT                    m_nPlatformIndex;
    
        /// <summary> true to supports concurrent kernels </summary>
        BOOL                    m_bSupportsConcurrentKernels;
    
        /// <summary> The device </summary>
        PhysicalDevice *        m_pDevice;

        /// <summary> Name of the device </summary>
        char *                  m_lpszDeviceName;

        /// <summary>   Default asynchronous contexts: dedicated copy/xfer contexts,
        ///             wrapper for default device context. </summary>
        std::map<ASYNCCONTEXTTYPE, AsyncContext*> m_vDeviceAsyncContexts;

        /// <summary>   Async context objects created for task execution. 
        ///             Good to have around so we can let them know when we sync
        ///             the whole device. 
        ///             </summary>
        std::map<AsyncContext*, Task*> m_vTaskAsyncContexts;
        std::map<Task*, AsyncContext*> m_vTaskAsyncContextMap;
    
        /// <summary> The user messages </summary>
        char                    m_lpszUserMessages[USER_OUTPUT_BUF_SIZE];

        /// <summary>   true if asynchronous contexts released. </summary>
        BOOL                    m_bAsyncContextsReleased;

        /// <summary> Counter for assigning unique identifiers
        /// 		  to Accelerator objects. 
        /// 		  </summary>
        static UINT m_uiUIDCounter;

        static _declspec(thread) LPFNTLSCTXTDEINITIALIZER   s_vTLSDeinitializers[MAXTLSCTXTCALLBACKS];
        static _declspec(thread) int                        s_nTLSDeinitializers;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assigns a new unique identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   UINT, unique across Accelerator ids. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT AssignUniqueAcceleratorIdentifier();           

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
                                                      )=0;   

    };

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compare accelerators by memspace identifier. Used for
    /// 			sorting accelerators before lock acquisition.
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    struct compare_accelerators_by_memspace_id {     
        bool operator() (const Accelerator * lhs, const Accelerator * rhs) { 
            return ((Accelerator*)lhs)->GetMemorySpaceId() <
                ((Accelerator*)rhs)->GetMemorySpaceId();
        } 
    };


};
#endif
