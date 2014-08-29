//--------------------------------------------------------------------------------------
// File: pbuffer.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PBUFFER_H_
#define _PBUFFER_H_
#include <map>

#include "ptaskutils.h"
#include "primitive_types.h"
#include <deque>
#include <set>
#include "hrperft.h"

namespace PTask {

    /// <summary>   Sometimes empty datablocks must be created. Binding views of such
    /// 			blocks to device-side resources requires an actual object though, and 
    /// 			zero-size device-side buffers are not allocatable in general. When this
    /// 			occurs we must allocate something even though we do not use it. We choose
    /// 			16 because that is max over the Platform-specific alloc mins of the various
    /// 			backends supported by PTask. </summary>
    static const UINT EMPTY_BUFFER_ALLOC_SIZE = 16;

    /// <summary>   The default alignment. </summary>
    static const UINT DEFAULT_ALIGNMENT = 1;

    /// <summary>   The default size bytes. </summary>
    static const UINT PBUFFER_DEFAULT_SIZE = 0;

    ///-------------------------------------------------------------------------------------------------
    /// Forward declarations
    ///-------------------------------------------------------------------------------------------------

    class Accelerator;
    class DatablockTemplate;
    class Datablock;
    class AsyncContext;
    class AsyncDependence;
    class SyncPoint;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Keys values for buffer view types. These provide a layer of indirection
    ///             between platform-specific objects and view requesters. The view map associated
    ///             with a PBuffer object will have entries at each view type that has been created.
    ///             DirectX uses ShaderResourceView objects to provide readable views,
    ///             UnorderedAccessViews to support writable views, and constant buffers for
    ///             immutable data, resulting in a different object type at each entry in the view
    ///             map. In contrast OpenCL and CUDA   do not use different object types for
    ///             different access types, so the view map can put the same object at both readable
    ///             and writable keys.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum bindingtype_t {
        
        /// <summary> default view type. not used </summary>
        BVT_DEFAULT = 0,
        
        /// <summary> an object at this key entry supports device-side
        /// 		  read access, for example a ShaderResourceView object
        /// 		  in DirectX 11 </summary>
        BVT_ACCELERATOR_READABLE = 1,

        /// <summary> an object at this key entry supports device-side
        /// 		  write access, for example a ShaderResourceView object
        /// 		  in DirectX 11 </summary>
        BVT_ACCELERATOR_WRITEABLE = 2,
        
        /// <summary> an object at this key entry supports device-side
        /// 		  read only access, backed by specialized constant
        /// 		  memory on the device. </summary>
        BVT_ACCELERATOR_IMMUTABLE = 3
         
    } BINDINGTYPE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   bindable object map entry type. Which member of the union 
    /// 			is valid is determined by the key in the map entry
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef union bv_t {
        
        /// <summary> DX11 read access: actual type is ID3D11ShaderResourceView </summary>
        void * psrv;
        
        /// <summary> DX11 write access: actual type is ID3D11UnorderedAccessView </summary>
        void * puav;
        
        /// <summary> DX11 constant memory access (immutable): actual type is ID3D11Buffer </summary>
        void * pconst;
        
        /// <summary> OpenCL read/write/immutable access: actual type is cl_mem </summary>
        void * pclmem;

        /// <summary> OpenCL immutable access.</summary>
        void * pclconst;
        
        /// <summary> CUDA read/write/immutable access.
        /// 		  Currently we don't use constant memory explicitly
        /// 		  in the CUDA back end. </summary>
        void * vptr;

    } BINDABLEOBJECT, *PBINDABLEOBJECT, **PPBINDABLEOBJECT;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Abstract super-class for platform specific buffer objects. PTask encapsulates
    ///             several GPU runtimes, all of which use different object types to support device-
    ///             side buffer management. Each runtime that PTask supports requires a subclass of
    ///             PBuffer.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class PBuffer {

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

        PBuffer(
            __in Datablock * pParentDatablock,
            __in BUFFERACCESSFLAGS bufferAccessFlags, 
            __in UINT nChannelIndex,
            __in Accelerator * pAccelerator=NULL, 
            __in Accelerator * pAllocatingAccelerator=NULL,
            __in UINT uiUniqueIdentifier=ptaskutils::nextuid()
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~PBuffer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes a new PBuffer. Post-condition: IsInitialized returns true if the
        ///             initialization succeeded.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiBufferSizeBytes">        The buffer size in bytes. </param>
        /// <param name="pInitialBufferContents">   [in] If non-null, the initial buffer contents. </param>
        /// <param name="strDebugBufferName">       (optional) [in] If non-null, a name to assign to the
        ///                                         buffer which will be used to label runtime- specific
        ///                                         objects to aid in debugging. Ignored on release
        ///                                         builds. </param>
        /// <param name="bIsByteAddressable">       (optional) true if the resulting PBuffer must be byte
        ///                                         addressable by the device. </param>
        /// <param name="bPageLock">                (optional) the page lock. </param>
        ///
        /// <returns>   PTRESULT--use PTSUCCESS/PTFAILED macros to determine success or failure. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT    
        Initialize(
            __in AsyncContext *     pAsyncContext,
            __in UINT               uiBufferSizeBytes,
            __in HOSTMEMORYEXTENT * pInitialBufferContents,
            __in char *             strDebugBufferName=NULL, 
            __in bool               bIsByteAddressable=true,
            __in bool               bPageLock=false
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the parameter type. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The parameter type. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTASK_PARM_TYPE	    GetParameterType();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is scalar parameter. </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <returns>   true if scalar parameter, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        IsScalarParameter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has its platform-specific buffer created (device-side
        /// 			in the common case). </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <returns>   true if bindable objects, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        IsPlatformBufferInitialized();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has its platform-specific buffer populated (device-side
        /// 			in the common case). </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <returns>   true if bindable objects, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        IsPlatformBufferPopulated();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has its platform-specific bindable objects created. </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <returns>   true if bindable objects, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        IsBindable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has had its dimensions finalized. </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <returns>   true if dimensions finalized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        IsDimensionsFinalized();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the device-side buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void*		GetBuffer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the parent of this item. </summary>
        ///
        /// <remarks>   Crossbac, 1/5/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the parent. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock*  GetParent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force synchronize. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void        ForceSynchronize()=0;

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

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait oustanding dependences. What we have to wait for depends on the type of
        ///             operation we plan to queue. Specifically, if the operation is a read, we can add
        ///             it to the read frontier and wait on the last write. If it is a write, we must
        ///             wait for the most recent reads.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="eOpType">              Type of the operation. </param>
        /// <param name="pDependences">         [in,out] (Optional) If non-null, (Optional) the
        ///                                     dependences. </param>
        /// <param name="pbAlreadyResolved">    [in,out] (Optional) If non-null, (Optional) the pb
        ///                                     already resolved. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        
        WaitOutstandingAsyncOperations(
            __in  AsyncContext * pAsyncContext,
            __in  ASYNCHRONOUS_OPTYPE eOpType,
            __in  std::set<AsyncDependence*>* pDependences=NULL,
            __out BOOL * pbAlreadyResolved=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Lockless wait outstanding: without acquiring any locks attempt to perform a
        ///             synchronous wait for any outstanding async dependences on this buffer that
        ///             conflict with an operation of the given type. This is an experimental API,
        ///             enable/disable with PTask::Runtime::*etTaskDispatchLocklessIncomingDepWait(),
        ///             attempting to leverage the fact that CUDA apis for waiting on events (which
        ///             appear to be thread-safe and decoupled from a particular device context)
        ///             to minimize serialization associated with outstanding dependences on data
        ///             consumed by tasks that do not require accelerators for any other reason than to
        ///             wait for such operations to complete.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///
        /// <param name="eOpType">  Type of the operation. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        LocklessWaitOutstanding(
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this buffer has any outstanding asynchronous ops: this is mostly a debug
        ///             tool for asserting that after a buffer has been involved in a synchronous
        ///             operation, all its outstanding conflicting operations have completed.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="eOpType">  Type of the operation. </param>
        ///
        /// <returns>   true if outstanding asynchronous ops, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        HasOutstandingAsyncOps(
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record outstanding dependences. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="eOperationType">   Type of the operation. </param>
        /// <param name="pSyncPoint">       (optional) [in,out] If non-null, the synchronise point. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        
        AddOutstandingDependence(
            __in AsyncContext * pAsyncContext,
            __in ASYNCHRONOUS_OPTYPE eOperationType,
            __in SyncPoint * pSyncPoint=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Mark dirty. </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void MarkDirty(BOOL bDirty);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is dirty. </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <returns>   true if dirty, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsDirty();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a platform-specific object corresponding to the target
        /// 			binding type. In some platforms, the buffer object can be bound
        /// 			directly to kernel parameters and global data (e.g. CUDA) and in others,
        /// 			different objects are required depending on the resource to which
        /// 			the buffer will be bound. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="t">    (optional) the t. </param>
        ///
        /// <returns>   The view. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BINDABLEOBJECT			GetBindableObject(BINDINGTYPE t=BVT_DEFAULT);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the access flags. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   The access flags. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BUFFERACCESSFLAGS	GetAccessFlags();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the access flags. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="f">    The f. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void                SetAccessFlags(BUFFERACCESSFLAGS f);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the uid. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   The uid. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT				GetUID();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a uid. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="ui">   The user interface. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void				SetUID(UINT ui);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the template. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the template. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual DatablockTemplate*	GetTemplate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the elements. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   The elements. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT				GetElementCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the stride. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   The stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT				GetElementStride();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the row pitch. </summary>
        ///
        /// <remarks>   Crossbac . </remarks>
        ///
        /// <returns>   The pitch. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT				GetPitch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the size of allocated backing buffers in bytes. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <returns>   The size bytes. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT                GetAllocationExtentBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the logical size of the buffer in bytes. This may differ from the allocation
        ///             size due to things such as alignment, or the need to allocate non-zero size
        ///             buffers to back buffers that are logically empty.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <returns>   The size bytes. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT                GetLogicalExtentBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize a host view in the given buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="bForceSynchronous">    [in,out] (optional) true to block caller until transfer
        ///                                     completes. </param>
        ///
        /// <returns>   number of bytes written. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT				
        PopulateHostViewSynchronous(
            __in AsyncContext * pAsyncContext,
            __in HOSTMEMORYEXTENT * pBuffer
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize accelerator view. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] (optional)  If non-null, the stream. </param>
        /// <param name="pSourceHostBuffer">    [in,out] If non-null, buffer for source host data. </param>
        /// <param name="pBuffer">              [in,out] If non-null, the data. </param>
        /// <param name="pModule">              (optional) [in,out] If non-null, the module. </param>
        /// <param name="lpszBinding">          (optional) the binding. </param>
        ///
        /// <returns>   number of bytes transferred. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT				
        PopulateAcceleratorView(
            __in AsyncContext * pAsyncContext,
            __in PBuffer * pSourceHostBuffer,
            __in HOSTMEMORYEXTENT * pBuffer,
            __in void * pModule=NULL, 
            __in const char * lpszBinding=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize a host view in the given buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="pBuffer">              [in,out] If non-null, buffer for lpv data. </param>
        /// <param name="bForceSynchronous">    (optional) true to block caller until transfer completes. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT				
        PopulateHostView(
            __in AsyncContext * pAsyncContext,
            __in PBuffer * pBuffer,
            __in BOOL bForceSynchronous
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize accelerator view. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] (optional)  If non-null, the stream. </param>
        /// <param name="pBuffer">          [in,out] If non-null, the data. </param>
        /// <param name="pModule">          (optional) [in,out] If non-null, the module. </param>
        /// <param name="lpszBinding">      (optional) the binding. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT				
        PopulateAcceleratorView(
            __in AsyncContext * pAsyncContext,
            __in PBuffer * pBuffer,
            __in void * pModule=NULL, 
            __in const char * lpszBinding=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is a "logically" empty buffer: in many cases we must
        ///             allocate a non-zero size buffer so that we have a block that can be bound as a
        ///             dispatch parameter, even when the buffer is logically empty. A side-effect of
        ///             this is that we must carefully track the fact that this buffer is actually empty
        ///             so that ports bound as descriptor ports (which infer other parameters such as
        ///             record count automatically) do the right thing when the buffer size is non-zero
        ///             but the logical size is zero.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/11/2012. </remarks>
        ///
        /// <returns>   true if empty buffer, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsEmptyBuffer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is physical buffer pinned. </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if physical buffer pinned, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsPhysicalBufferPinned();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Debug dump. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="t">                    (optional) the t. </param>
        /// <param name="nDumpStartElement">    (optional) the dump start element. </param>
        /// <param name="nDumpEndElement">      (optional) the dump end element. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void                
        DebugDump(
            __in DEBUGDUMPTYPE t=dt_raw, 
            __in UINT nDumpStartElement=0, 
            __in UINT nDumpEndElement=0,
            __in UINT nStride=1
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Debug dump. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="ssOut">        [in,out] If non-null, the ss out. </param>
        /// <param name="pcsOutLock">   [in,out] If non-null, the pcs out lock. </param>
        /// <param name="szTaskLabel">  [in,out] (optional) the t. </param>
        /// <param name="szPortLabel">  [in,out] (optional) the dump start element. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void                
        DebugDump(
            __in std::ostream* ssOut,
            __in CRITICAL_SECTION* pcsOutLock,
            __in char * szTaskLabel,
            __in char * szPortLabel
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator *       GetAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerator that allocated this buffer. Returns NULL, unless an
        ///             accelerator other than m_pAccelerator allocated the buffer (e.g. when a CUDA
        ///             accelerator has AllocatePagelockedHostMemory() called to allocate host-memory in a PBuffer).
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator *       GetAllocatingAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Context requires synchronise. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="eOperationType">   Type of the operation. </param>
        /// <param name="pDependences">     [in,out] (Optional) If non-null, (Optional) the dependences. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        ContextRequiresSync(
            __in  AsyncContext * pAsyncContext,
            __in  ASYNCHRONOUS_OPTYPE eOperationType,
            __out std::set<AsyncDependence*>* pDependences=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if there are outstanding dependences that would
        ///             need to resolve before an operation of the given type
        ///             can occur. </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if outstanding dependence, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        HasConflictingOutstandingDependences(
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the set of accelerators involved in dependences that must
        ///             be resolved before an operation of the given type can be executed
        ///             without incurring a read-write hazard. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/26/2013. </remarks>
        ///
        /// <param name="eOpType">  Type of the operation. </param>
        ///
        /// <returns>   null if it fails, else the outstanding accelerator dependences. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::set<Accelerator*>*
        GetOutstandingAcceleratorDependences(
            __in ASYNCHRONOUS_OPTYPE eOpType
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
        /// <summary>   We rely heavily on asynchrony, but we release blocks when we are done
        ///             queueing dispatches that use them. Consequently, it is entirely probable that
        ///             we wind up attempting to free Datablocks or return them to their block pools
        ///             before the operations we've queued on them have actually completed.
        ///             With block pools we rely on leaving the outstanding dependences queued
        ///             on the buffers in the datablock. However, for blocks that actually get
        ///             deleted, we need to be sure that any dangling operations have actually
        ///             completed on the GPU. This method is for precisely that--it should only
        ///             be called from the release/dtor code of Datablock. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL SynchronousWaitOutstandingOperations();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the retired dependences. </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        ReleaseRetiredDependences(
            __in AsyncContext * pAsyncContext
            );

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return the number of dependences in the given frontier that
        ///             are actually still outstanding. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="vFrontier">    [in,out] [in,out] If non-null, the frontier. </param>
        ///
        /// <returns>   The frontier outstanding. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT
        QueryFrontierOutstanding(
            __in std::deque<AsyncDependence*>& vFrontier
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Clears the dependences. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        ClearDependences(
            __in AsyncContext * pAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this context is already on the dependence frontier for
        ///             the given operation type. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="eOpType">          Type of the operation. </param>
        ///
        /// <returns>   true if outstanding context, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL  
        IsSingletonOutstandingContext(
            __in AsyncContext * pAsyncContext,
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

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
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize mutable accelerator view. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiBufferSizeBytes">    The buffer size in bytes. </param>
        /// <param name="pInitialData">         [in,out] If non-null, the data. </param>
        /// <param name="bRequestOutstanding">  [in,out] The request outstanding. </param>
        /// <param name="pModule">              [in,out] (optional)  If non-null, the module. </param>
        /// <param name="lpszBinding">          (optional) the binding. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

	    virtual UINT		
        __populateMutableAcceleratorView(
            __in  AsyncContext *     pAsyncContext,
            __in  UINT               uiBufferSizeBytes,
            __in  HOSTMEMORYEXTENT * pInitialData,
            __out BOOL&              bRequestOutstanding,
            __in  void *             pModule,
            __in  const char *       lpszBinding
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize immutable accelerator view. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiBufferSizeBytes">    If non-null, the data. </param>
        /// <param name="pInitialData">         [in,out] The bytes. </param>
        /// <param name="bRequestOutstanding">  [in,out] The request outstanding. </param>
        /// <param name="pModule">              [in,out] (optional)  If non-null, the module. </param>
        /// <param name="lpszBinding">          (optional) the binding. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT		
        __populateImmutableAcceleratorView(
            __in  AsyncContext *     pAsyncContext,
            __in  UINT               uiBufferSizeBytes,
            __in  HOSTMEMORYEXTENT * pInitialData,
            __out BOOL&              bRequestOutstanding,
            __in  void *             pModule,
            __in  const char *       lpszBinding
            )=0;

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
            )=0;

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
            )=0;

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

        virtual PTRESULT	CreateBindableObjectsReadable(char * szname = NULL)=0;

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

        virtual PTRESULT	CreateBindableObjectsWriteable(char * szname = NULL)=0;

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

        virtual PTRESULT	CreateBindableObjectsImmutable(char * szname = NULL)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Create all necessary bindable objects for the buffer based on it's access flags.
        ///             In some platforms, notably directX, view objects exist for different types of
        ///             access. Read access for the GPU requires a shader resource view, write access
        ///             requires an unordered access view, etc. Other platforms (e.g. CUDA) treat all
        ///             device-visible buffers as a single array of flat memory. This function calls into
        ///             it the PBuffer's specializing type to create any views needed for the underlying
        ///             platform. platform-specific work is encapsulated in InitializeMutableBuffer and
        ///             InitializeImmutableBuffer.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
        ///                         used for debugging. </param>
        ///
        /// <returns>   HRESULT. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT	CreateBindableObjects(char * szname = NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalize the dimensions of the device buffer that will be created to back this
        ///             PBuffer.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <param name="bByteAddressable">     [out] (optional) true if the buffer should be byte
        ///                                     addressable. </param>
        /// <param name="uiBufferSizeBytes">    (optional) the buffer size in bytes. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        
        FinalizeDimensions(
            __out bool &bByteAddressable,
            __in UINT uiBufferSizeBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalize the dimensions of the device buffer that will be created to back this
        ///             PBuffer: the bool bRequireSealedParent indicates whether or not the parent datablock
        ///             must be sealed for the block to have its dimensions finalized: for platform buffers
        ///             in any space other than the host, we have to finalize because there will be traffic
        ///             between memory spaces that requires a known, immutable allocation size. In the
        ///             host space however, to let the programmer fill a buffer and subsequently seal it,
        ///             we must be willing to allocate buffers to back an unsealed data block. Exposing a
        ///             parameter for this requirement makes it easier to override FinalizeDimensinons for
        ///             the host-specific pbuffer class. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac </remarks>
        ///
        /// <param name="bByteAddressable">     [out] (optional) true if the buffer should be byte
        ///                                     addressable. </param>
        /// <param name="uiBufferSizeBytes">    (optional) the buffer size in bytes. </param>
        /// <param name="bRequireSealedParent"> The require sealed parent. </param>
        ///
        /// <returns>   bytes allocated. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT        
        FinalizeDimensions(
            __out bool &bByteAddressable,
            __in  UINT uiBufferSizeBytes,
            __in  BOOL bRequireSealedParent
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform specific finalize dimension. </summary>
        ///
        /// <remarks>   crossbac, 7/10/2012. </remarks>
        ///
        /// <param name="uiStride">         The stride. </param>
        /// <param name="uiElementCount">   Number of elements. </param>
        /// <param name="uiDimension">      The dimension. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT
        PlatformSpecificFinalizeDimension(
            __in UINT uiStride,
            __in UINT uiElementCount,
            __in UINT uiDimension
            );        

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Retire dependence frontier. </summary>
        ///
        /// <remarks>   crossbac, 5/1/2013. </remarks>
        ///
        /// <param name="deps">             [in,out] [in,out] If non-null, the deps. </param>
        /// <param name="acclist">          [in,out] [in,out] If non-null, the acclist. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        RetireDependenceFrontier(
            __inout std::deque<AsyncDependence*>& deps,
            __inout std::set<Accelerator*>& acclist,
            __in    AsyncContext * pAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Retire any entries in the dependence frontier that 
        ///             have been resolved synchronously, either through 
        ///             context sync or event query. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void
        RetireResolvedFrontierEntries(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps an initialise data. </summary>
        ///
        /// <remarks>   Crossbac, 1/4/2012. </remarks>
        ///
        /// <param name="pInitData">    [in,out] If non-null, information describing the initialise. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			DumpInitData(VOID* pInitData);

        /// <summary> Unique identifier </summary>
        UINT						m_uiId;
        
        /// <summary> Zero-based index of the channel this PBuffer is backing
        /// 		  in its parent datablock. Here for convenience.
        /// 		  </summary>
        UINT                        m_nChannelIndex;
        
        /// <summary> The access flags, indicating what kind of resources
        /// 		  this PBuffer will need to be bound to and accessed
        /// 		  from. </summary>
        BUFFERACCESSFLAGS			m_eAccessFlags;
        
        /// <summary> The datablock this PBuffer is backing
        /// 		  in a specific memory domain. </summary>
        Datablock *                 m_pParent;
        
        /// <summary> The accelerator in whose memory domain this
        /// 		  PBuffer provides a physical view. </summary>
        Accelerator *				m_pAccelerator;
        
        /// <summary> The accelerator objects whose specialized allocator
        /// 		  was used to create this PBuffer. NULL if allocation 
        /// 		  was not deferred to another accelerator object.
        /// 		  </summary>
        Accelerator *               m_pAllocatingAccelerator;

        /// <summary>   The *requested* buffer dimensions. </summary>
        BUFFERDIMENSIONS            m_vDimensionsRequested;
        
        /// <summary>   true if the requested stride value is valid. </summary>
        BOOL                        m_bRequestedStrideValid;

        /// <summary>   true if the requested element counts are valid. </summary>
        BOOL                        m_bRequestedElementsValid;
        
        /// <summary> The actual dimensions of the buffer. </summary>
        BUFFERDIMENSIONS            m_vDimensionsFinalized;       

        /// <summary> The device-specific buffer object. </summary>
        void *                      m_pBuffer;
        
        /// <summary> true if the device buffer has been populated. </summary>
        BOOL                        m_bPopulated;
        
        /// <summary> true if buffer dimensions are finalized </summary>
        BOOL                        m_bDimensionsFinalized;
        
        /// <summary> A map from binding type to objects that can be
        /// 		  bound to device-side resources of that type. 
        /// 		  </summary>
        std::map<BINDINGTYPE, BINDABLEOBJECT> m_mapBindableObjects;

        /// <summary>   The write frontier. </summary>
        std::deque<AsyncDependence*> m_vWriteFrontier;

        /// <summary>   accelerators represented in the write frontier. </summary>
        std::set<Accelerator*> m_vOutstandingWriteAccelerators;

        /// <summary>   The read frontier. </summary>
        std::deque<AsyncDependence*> m_vReadFrontier;

        /// <summary>   accelerators represented in the write frontier. </summary>
        std::set<Accelerator*> m_vOutstandingReadAccelerators;

        std::deque<AsyncDependence*> m_vRetired;

        /// <summary>   true if this object is logically empty. in many cases we must allocate a non-zero
        ///             size buffer so that we have a block that can be bound as a dispatch parameter,
        ///             even when the buffer is logically empty. A side-effect of this is that we must
        ///             carefully track the fact that this buffer is actually empty so that ports bound
        ///             as descriptor ports (which infer other parameters such as record count
        ///             automatically) do the right thing when the buffer size is non-zero but the
        ///             logical size is zero.
        ///             </summary>

        BOOL                        m_bIsLogicallyEmpty;

        /// <summary>   The byte alignment requirements for the buffer. </summary>
        UINT                        m_nAlignment;

        /// <summary>   true if physical buffer is page-locked. </summary>
        BOOL                        m_bPhysicalBufferPinned;

        /// <summary>   true if buffer request was for page-locked memory--no guarantees!. </summary>
        BOOL                        m_bPinnedBufferRequested;

        /// <summary>   true if the buffer is dirty with respect to some initial state. </summary>
        BOOL                        m_bDirty;

        void CheckDependenceInvariants();

    protected:

        static void * m_pProfiler;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initialises the allocation profiler. </summary>
        ///
        /// <remarks>   Crossbac, 9/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void InitializeProfiler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinit allocation profiler. </summary>
        ///
        /// <remarks>   Crossbac, 9/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void DeinitializeProfiler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the allocation profiler data. </summary>
        ///
        /// <remarks>   Crossbac, 9/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void ProfilerReport(std::ostream &ios);

    };

};
#endif

