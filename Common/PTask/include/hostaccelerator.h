//--------------------------------------------------------------------------------------
// File: hostaccelerator.h
// host "accelerator"
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _HOST_ACCELERATOR_H_
#define _HOST_ACCELERATOR_H_

#include "primitive_types.h"
#include "datablocktemplate.h"
#include "dxcodecache.h"
#include "accelerator.h"
#include "CompiledKernel.h"
#include <string>
#include <map>
#include <vector>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Host accelerator. 
    /// 			
    /// 			The host accelerator provides a way to execute ptask nodes
    /// 			on the CPU. Currently, where an accelerator-based ptask 
    /// 			accepts source code and "compiles" a node with the resulting
    /// 			binary, the host accelerator expects a dll with a 
    /// 			
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class HostAccelerator : public Accelerator {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="cpuid">    The cpuid. </param>
        /// <param name="lpszName"> [in,out] If non-null, the name. </param>
        ///-------------------------------------------------------------------------------------------------

        HostAccelerator(int cpuid, char * lpszName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~HostAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Open the host accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT		    Open();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the device. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the device. </returns>
        ///-------------------------------------------------------------------------------------------------

        void*					GetDevice();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the context. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
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
        /// <summary>   Cache a binary. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="szFile">   [in,out] If non-null, the file. </param>
        /// <param name="szFunc">   [in,out] If non-null, the func. </param>
        /// <param name="lpfn">     The lpfn. </param>
        /// <param name="hModule">  The module. </param>
        ///-------------------------------------------------------------------------------------------------

        void					CachePutShader(char * szFile, char * szFunc, FARPROC lpfn, HMODULE hModule);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check the cache for a binary. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="szFile">       [in,out] If non-null, the file. </param>
        /// <param name="szFunc">       [in,out] If non-null, the func. </param>
        /// <param name="ppFunction">   [in,out] The function. </param>
        /// <param name="pModule">      [in,out] The module. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL    				CacheGetShader(char * szFile, char * szFunc, FARPROC &ppFunction, HMODULE &pModule);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Compiles accelerator source code to create a PTask binary. </summary>
        ///
        /// <remarks>   Crossbac, 12/17/2011. 
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

        virtual BOOL        Compile(
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
        ///             build a binary for.  
        ///             
        ///             Currently, this is not implemented for host tasks because this involves
        ///             setting up infrastructure to choose a compiler and target a DLL, etc. 
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
        /// <summary>   Gets the context current. </summary>
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
        /// <summary>   Gets the supports function arguments. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            SupportsFunctionArguments();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the supports byval arguments. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            SupportsByvalArguments();

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
        /// <summary>   Query if 'p' has accessible memory space. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the p. </param>
        ///
        /// <returns>   true if accessible memory space, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasAccessibleMemorySpace(Accelerator*p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the supports pinned host memory. </summary>
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
        /// <summary>   Enumerate accelerators. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="devices">  [in,out] [in,out] If non-null, the devices. </param>
        ///-------------------------------------------------------------------------------------------------

        static void             EnumerateAccelerators(std::vector<Accelerator*> &devices);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate memory extent. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="ulNumberOfBytes">  The ul number of in bytes. </param>
        /// <param name="ulFlags">          The ul flags. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static void *           __stdcall AllocateMemoryExtent(ULONG ulNumberOfBytes, ULONG ulFlags);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate memory extent. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="ulNumberOfBytes">  The ul number of in bytes. </param>
        /// <param name="ulFlags">          The ul flags. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static void             __stdcall DeallocateMemoryExtent(void* pvMemoryExtent);

    protected:

        /// <summary> Identifier for the device </summary>
        int                             m_nDeviceId;

        /// <summary> The code cache </summary>
        std::map<std::string, FARPROC>  m_pCodeCache;

        /// <summary> The module cache </summary>
        std::map<std::string, HMODULE>  m_pModuleCache;

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

    };
};

#endif