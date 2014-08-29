//--------------------------------------------------------------------------------------
// File: primitive_types.h
//
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PRIMITIVE_TYPES_H_
#define _PRIMITIVE_TYPES_H_

#include <stdio.h>
#ifdef DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#endif
#ifdef DEBUG
#include <crtdbg.h>
#endif
#include <d3dcommon.h>
 
#ifndef PTSRELEASE
#define PTSRELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#endif

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Defines an alias for PTask error codes.
    /// 			O or below is a successful outcome. 
    /// 			Positive values indicate failure.</summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef long PTRESULT;

    static const long PTASK_OK = 0;

    //-------------------------------------------------------------------------------------------------
    // result codes > 0 mean failure.
    //-------------------------------------------------------------------------------------------------
    
    /// <summary> Generic/catch-all failure code </summary>
    static const long PTASK_ERR = PTASK_OK + 1;
    
    /// <summary> The requested object was not found </summary>
    static const long PTASK_ERR_NOT_FOUND = PTASK_ERR + 1;
    
    /// <summary> An invalid parameter was given </summary>
    static const long PTASK_ERR_INVALID_PARAMETER = PTASK_ERR + 2;

    /// <summary> Attempt to create or initialize an object
    /// 		  that already exists.
    /// 		  </summary>
    static const long PTASK_ERR_EXISTS = PTASK_ERR + 3;

    /// <summary> Warning: the graph is malformed in some way. For example, some ports are not bound
    /// 		  to channels, or some nodes have no inputs or outputs. Note this error code 
    /// 		  exists as a warning too, to differentiate between cases where the malformation
    /// 		  may be recoverable or can be tolerated by the runtime from those which are not.
    /// 		  </summary>
    static const long PTASK_ERR_GRAPH_MALFORMED = PTASK_OK + 4;

    /// <summary> The runtime has not been initialized yet, and an API call 
    ///           that requires an initialized runtime was made.
    /// 		  </summary>
    static const long PTASK_ERR_UNINITIALIZED = PTASK_OK + 5;

    /// <summary> The caller requested a state change for a ptask object such
    ///           as an accelerator that is currently in use for dispatch; the state
    ///           change cannot be performed until the dispatch completes.
    /// 		  </summary>
    static const long PTASK_ERR_INFLIGHT = PTASK_OK + 6;

    /// <summary> The caller attempted to enable an accelerator that would
    ///           cause the runtime to exceed the limit imposed by a call to 
    ///           PTask::Runtime::SetMaximumConcurrency.
    /// 		  </summary>    
    static const long PTASK_ERR_TOO_MANY_ACCELERATORS = PTASK_OK + 7;

    /// <summary>   The caller made an API call after initialization 
    ///             that is only valid before init.  </summary>
    static const long PTASK_ERR_ALREADY_INITIALIZED = PTASK_OK + 8;


    //-------------------------------------------------------------------------------------------------
    // result codes < 0 mean success, but with a warning
    //-------------------------------------------------------------------------------------------------
       
    /// <summary> Generic warning </summary>
    static const long PTASK_WARNING = PTASK_OK - 1;


    /// <summary> Warning: the graph is malformed in some way. For example, some ports are not bound
    /// 		  to channels, or some nodes have no inputs or outputs.
    /// 		  </summary>
    static const long PTASK_WARNING_GRAPH_MALFORMED = PTASK_OK - 2;

    #define PTSUCCESS(x) ((x) <= PTASK_OK)
    #define PTFAILED(x) ((x) > PTASK_OK)

    typedef unsigned int PORTINDEX;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent scheduling affinity
    /// 			between Tasks and Accelerators. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum affinity_type_t {
        
        /// <summary> no affinity relation exists between
        /// 		  the ptask and a given accelerator.</summary>
        AFFINITYTYPE_NONE = 676,        
        
        /// <summary> given the opportunity to prefer a particular
        /// 		  accelerator, do so, but when a non-matching
        /// 		  accelerator is available and the desired one
        /// 		  is not, use what is available. </summary>
        AFFINITYTYPE_WEAK = 677,
        
        /// <summary> given the opportunity to prefer a particular
        /// 		  accelerator, do so. Only use a non-matching
        /// 		  accelerator after a ptask has failed to
        /// 		  be scheduled on its preferred accelerator
        /// 		  repeatedly. See strong-affinity-threshold.
        /// 		  </summary>
        AFFINITYTYPE_STRONG = 678,

        
        /// <summary> Mandatory affinity means that a given ptask
        /// 		  can run only on the affinitized accelerator.
        /// 		  </summary>
        AFFINITYTYPE_MANDATORY = 679

    } AFFINITYTYPE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent accelerator types. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum acctype_t {

        /// <summary> Accelerator based on DirectX 11 backend </summary>
        ACCELERATOR_CLASS_DIRECT_X = 0,   

        /// <summary> Accelerator based on OpenCL backend </summary>
        ACCELERATOR_CLASS_OPEN_CL = 1,
        
        /// <summary> Accelerator based on CUDA backend </summary>
        ACCELERATOR_CLASS_CUDA = 2,
        
        /// <summary> Accelerator based on DirectX reference driver. 
        /// 		  This is a software-emulated implemention of DX11
        /// 		  hardware, so should not be used in a deployed environment.
        /// 		  We include it to enable debugging on machines without
        /// 		  proper hardware. See Runtime::SetUseReferenceDrivers()
        /// 		   </summary>
        ACCELERATOR_CLASS_REFERENCE = 3,
        
        /// <summary> Host-based accelerator. Technically, not an accelerator,
        /// 		  but a computation resource. Enables PTasks that run 
        /// 		  on the CPU to become part of a ptask graph. </summary>
        ACCELERATOR_CLASS_HOST = 4,
        
        /// <summary> Unknown accelerator type. Should never be used. </summary>
        ACCELERATOR_CLASS_UNKNOWN = 5,

        /// <summary>   An enum constant representing the accelerator class of super-tasks. </summary>
        ACCELERATOR_CLASS_SUPER = 6

    } ACCELERATOR_CLASS;

    static const char * g_lpszAccClassStrings[] = {
        "ACCELERATOR_CLASS_DIRECT_X",
        "ACCELERATOR_CLASS_OPEN_CL",
        "ACCELERATOR_CLASS_CUDA",
        "ACCELERATOR_CLASS_REFERENCE",
        "ACCELERATOR_CLASS_HOST",
        "ACCELERATOR_CLASS_UNKNOWN",
        "ACCELERATOR_CLASS_SUPER"
    };
    #define AccClassString(cls) (g_lpszAccClassStrings[(int)cls])

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Accelerator descriptor
    /// 			Data structure returned to user programs when 
    /// 			accelerators are enumerated in service of 
    /// 			setting ptask affinity.
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct ACCELERATOR_DESCRIPTOR_t {

        /// <summary> Unique identifier for the accelerator </summary>
        UINT                    uiAcceleratorId;
        
        /// <summary> The accelerator class </summary>
        ACCELERATOR_CLASS       accClass;

        /// <summary>   true if this accelerator is enabled in the scheduler. 
        ///             (can be false when devices are explicitly disabled e.g. using
        ///             Runtime::SetMaximumConcurrency)
        ///             </summary>
        BOOL                    bEnabled;
        
        /// <summary> Number of cores </summary>
        UINT					nCoreCount;
        
        /// <summary> The runtime version </summary>
        UINT                    nRuntimeVersion;
        
        /// <summary> Size of the memory </summary>
        UINT                    nMemorySize;
        
        /// <summary> The clock rate </summary>
        UINT                    nClockRate;
        
        /// <summary> Zero-based index of the m n platform </summary>
        UINT                    nPlatformIndex;
        
        /// <summary> true to supports concurrent kernels </summary>
        BOOL                    bSupportsConcurrentKernels;
        
        /// <summary> Description of the accelerator </summary>
        char                    szDescription[256];

        /// <summary>   The accelerator object itself. </summary>
        void *                  pAccelerator;
    
    } ACCELERATOR_DESCRIPTOR, * PACCELERATOR_DESCRIPTOR;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Access flags for buffers 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef unsigned int BUFFERACCESSFLAGS;

    /// <summary>   Flags that mean use default permissions for datablock access. </summary>
    static const unsigned int PT_ACCESS_DEFAULT = 0x0;

    /// <summary> The datablock or buffer will be read from host code</summary>
    static const unsigned int PT_ACCESS_HOST_READ = 0x1;
    
    /// <summary> The datablock or buffer will be written from host code </summary>
    static const unsigned int PT_ACCESS_HOST_WRITE = 0x2;
    
    /// <summary> The datablock or buffer will be read from accelerator code </summary>
    static const unsigned int PT_ACCESS_ACCELERATOR_READ = 0x4;
    
    /// <summary> The datablock or buffer will be written from accelerator code </summary>
    static const unsigned int PT_ACCESS_ACCELERATOR_WRITE = 0x8;
    
    /// <summary> The datablock or buffer should be bound to constant memory on the accelerator  </summary>
    static const unsigned int PT_ACCESS_IMMUTABLE = 0x10;
    
    /// <summary> The datablock will be accessed at byte-granularity by accelerator code </summary>
    static const unsigned int PT_ACCESS_BYTE_ADDRESSABLE = 0x20;

    /// <summary> The datablock may be accessed by the application from multiple disjoint
    ///           memory spaces. This flag is a hint--for most backends we can respond
    ///           dynamically to cross device-sharing, but for DirectX, we have to create
    ///           buffers with the right flags up front to enable API-level D->D transfers.
    ///           If this hint is actually set, there is no need for allocator code to 
    ///           try to make guesses based on the environment or graph. If this hint is
    ///           absent, we can still be correct if we fail to detect the sharing opportunity--
    ///           it just means we have to go through CPU memory explicitly. 
    ///           </summary>
    static const unsigned int PT_ACCESS_SHARED_HINT = 0x40;

    /// <summary>   The datablock may be part of a block pool. Knowing when this
    ///             is the case can help the runtime detect cases when it may need
    ///             to read/write GPU memory more than once, which may argue for
    ///             CPU-access regimes that differ from those that are derived just
    ///             from the semantics of the ports to which they are bound. In particular,
    ///             when allocating a new GPU buffer, some back-ends (DX) can provide
    ///             the initial data in the allocation function--for blocks that are not 
    ///             reused, no CPU-side view is ever needed again. For blocks that are
    ///             pooled, we can do better if we know up front that the CPU will write
    ///             the block again. This is purely a performance hint. 
    ///             </summary>
    static const unsigned int PT_ACCESS_POOLED_HINT = 0x80;

    #define NEEDS_HOST_VIEW(flags)        ((flags & PT_ACCESS_HOST_READ) || (flags & PT_ACCESS_HOST_WRITE))
    #define NEEDS_ACCELERATOR_VIEW(flags) ((flags & PT_ACCESS_ACCELERATOR_READ) || (flags & PT_ACCESS_ACCELERATOR_WRITE))

    /// <summary> The maximum datablock dimensions: XYZ</summary>
    static const int MAX_DATABLOCK_DIMENSIONS = 3;

    /// <summary> Index of X dimension in an array of 3 dimensions </summary>
    static const int XDIM = 0;

    /// <summary> Index of Y dimension in an array of 3 dimensions </summary>
    static const int YDIM = 1;

    /// <summary> Index of Z dimension in an array of 3 dimensions </summary>
    static const int ZDIM = 2;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent geometry dimension bindings for port/datablock pairs.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum estdim_t {
        GD_NONE=-1,
        GD_X = 0,
        GD_Y = 1,
        GD_Z = 2
    } GEOMETRYESTIMATORDIMENSION;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent accelerator code parameter types Required because back-end
    ///             runtimes supporting ptask often have different interfaces for handling scalar vs.
    ///             non-scalar data, or have no support function argument binding.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum ptask_parm_t {

        /// <summary> 4-byte integer scalar </summary>
        PTPARM_INT = 0,

        /// <summary> 4-byte floating-point scalar </summary>
        PTPARM_FLOAT = 1,

        /// <summary> 8-byte floating-point scalar </summary>
        PTPARM_DOUBLE = 2,      
        
        /// <summary> structure passed by value, making it
        /// 		  a scalar from the perspective of the backend runtime  
        /// 		  APIs. 
        /// 		  </summary>
        PTPARM_BYVALSTRUCT = 3,  // structs passed by value
        
        /// <summary> structure passed by reference. Handled 
        /// 		  the same as opaque buffers by most
        /// 		  APIs.
        /// 		  </summary>
        PTPARM_BYREFSTRUCT = 4,  // structs passed by ref are same as buffer
        
        /// <summary> opaque buffer (possibly with element-wise structure),
        /// 		  passed by reference. 
        /// 		  </summary>
        PTPARM_BUFFER = 5,       
                
        /// <summary> better not see this one! </summary>
        PTPARM_NONE = 6

    } PTASK_PARM_TYPE;

    const UINT DANDELION_DEFAULT_METADATA_SIZE = 16384;
    const UINT DANDELION_DEFAULT_TEMPLATE_SIZE = 16384;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Defines an type representing the datablock control code. 
    ///             These signals can be bit-wise or-ed to encode multiple
    ///             signals on a block, for example if a block is both the first
    ///             and last in a stream, it will require BOF | EOF, and so on.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef unsigned __int64 CONTROLSIGNAL;

    /// <summary> The datablock carries no control signal </summary>
    const CONTROLSIGNAL DBCTLC_NONE = 0x00000000; 

    /// <summary>   Code indicating beginning of stream. </summary>
    const CONTROLSIGNAL DBCTLC_BOF = 0x1;
    
    /// <summary> The datablock with this code is the last in a stream </summary>
    const CONTROLSIGNAL DBCTLC_EOF = 0x2;

    /// <summary> a block carrying this signal is the first block
    /// 		  in an iteration run. 
    /// 		  </summary>
    const CONTROLSIGNAL DBCTLC_BEGINITERATION = 0x4;

    /// <summary> a block carrying this signal is the final block
    /// 		  in an iteration run. 
    /// 		  </summary>
    const CONTROLSIGNAL DBCTLC_ENDITERATION = 0x8;

    /// <summary> A runtime error has occurred, and the datablock carrying 
    /// 		  this code is propagating that fact to downstream tasks.
    /// 		  </summary>
    const CONTROLSIGNAL DBCTLC_ERR = 0x10;
  
    /// <summary>   Use this code to intentionally create outputs
    /// 			with values that will never pass the predicate. 
    /// 			Think: "myDataSource > /dev/null" . </summary>
    const CONTROLSIGNAL DBCTLC_DEVNULL = 0x20;

    static const char * g_lpszControlSignalStrings[] = {
        "DBCTLC_NONE",
        "DBCTLC_BOF",
        "DBCTLC_EOF",
        "DBCTLC_BEGINITERATION",
        "DBCTLC_ENDITERATION",
        "DBCTLC_ERR",
        "DBCTLC_DEVNULL"
    };
    
    #define ControlSignalString(eSignal) (eSignal?g_lpszControlSignalStrings[ptaskutils::GetFirstSignalIndex(eSignal)+1]:g_lpszControlSignalStrings[0])

    #define TESTSIGNAL(x,y) (((x)&(y))!=0)
    #define HASSIGNAL(x)    ((x)!=DBCTLC_NONE)

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Defines an iterator for signal words with multiple control signals. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct __ctlsig_iter_t { 
        unsigned __int64 m_uiPos;
        CONTROLSIGNAL * pCtlSignalWord;
        __ctlsig_iter_t(CONTROLSIGNAL &luiSignal) {
            m_uiPos = 0;
            pCtlSignalWord = &luiSignal;
            CONTROLSIGNAL luiTestBit = (1i64<<m_uiPos);
            while(m_uiPos<sizeof(CONTROLSIGNAL) && !TESTSIGNAL(*pCtlSignalWord, luiTestBit)) {
                m_uiPos++;
                luiTestBit = (1i64<<m_uiPos);
            }
        }
        CONTROLSIGNAL begin() {
            if(m_uiPos >= sizeof(CONTROLSIGNAL)) return DBCTLC_NONE;
            CONTROLSIGNAL luiTestBit = (1i64<<m_uiPos);
            if(!TESTSIGNAL(*pCtlSignalWord, luiTestBit)) return DBCTLC_NONE;
            return luiTestBit;
        }
        CONTROLSIGNAL end() {
            return DBCTLC_NONE;
        }
        CONTROLSIGNAL operator ++() {
            do {
                m_uiPos++;
                if(m_uiPos>=sizeof(CONTROLSIGNAL)) return DBCTLC_NONE;
                CONTROLSIGNAL luiTestBit = (1i64<<m_uiPos);
                if(TESTSIGNAL(*pCtlSignalWord, luiTestBit)) return luiTestBit;
            } while(m_uiPos<sizeof(CONTROLSIGNAL));
            return DBCTLC_NONE;
        }
    } CTLSIGITERATOR;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent buffer coherence state across different accelerator memory
    ///             spaces. These states implement a simple MSI protocol, where we distinguish
    ///             between invalid state (which is out-of-date) and "no-entry" which is more akin to
    ///             the traditional I state of coherence protocols.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum buffer_coh_state_t {

        /// <summary> no buffer has been created for this memory space. </summary>
        BSTATE_NO_ENTRY = 0,

        /// <summary> a buffer exists at this entry but is out-of-date
        /// 		  due to accesses that have occurred in other memory
        /// 		  spaces. </summary>
        BSTATE_INVALID = 1,

        /// <summary> a buffer exists at this entry, and it is up-to-date,
        /// 		  however, other up-to-date copies exists that require
        /// 		  invalidation if this buffer is modified. 
        /// 		  </summary>
        BSTATE_SHARED = 2,

        /// <summary> a buffer exists at this entry, and is the only
        /// 		  valid version of the data. </summary>
        BSTATE_EXCLUSIVE = 3

    } BUFFER_COHERENCE_STATE;

    /// <summary> Datablocks have at most 3 channels </summary>
    const UINT NUM_DATABLOCK_CHANNELS = 3;

    /// <summary> Zero-based index of the datablock data channel </summary>
    const UINT DBDATA_IDX = 0;
    
    /// <summary> Zero-based index of the datablock metadata channel </summary>
    const UINT DBMETADATA_IDX = 1;
    
    /// <summary> Zero-based index of the datablock template channel </summary>
    const UINT DBTEMPLATE_IDX = 2;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent different port types. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum porttyp_t {

        /// <summary> An input port binds buffers passed by
        /// 		  reference to input kernel parameters and global
        /// 		  data that is read only on the device, but too big
        /// 		  to put in constant memory. 
        /// 		  </summary>
        INPUT_PORT,

        /// <summary> An output port binds buffers passed by reference to
        /// 		  kernel parameters or global buffers that are written
        /// 		  and potentially read by device code. Parameters that have
        /// 		  in/out semantics in kernel code require both an InputPort
        /// 		  and an OutputPort. 
        /// 		  </summary>
        OUTPUT_PORT,
        
        /// <summary> A port that is bound to scalar values in kernel code, 
        /// 		  with read-only semantics. Typically these values are
        /// 		  bound to constant memory on a device where specialized memories
        /// 		  are available. A sticky port also retains its last value: 
        /// 		  if no new datablock is available on its incoming channel it 
        /// 		  will redeliver the last datablock pulled from it on the next
        /// 		  call to Pull.  
        /// 		  </summary>
        STICKY_PORT,
        
        /// <summary> A MetaPort carries datablocks that contain information consumed
        /// 		  by the runtime rather than being bound to accelerator code. 
        /// 		  There are several canonical "meta-functions" provided by 
        /// 		  MetaPorts, for example, describing output size allocation for
        /// 		  a downstream output port, or describing loop iteration bounds  
        /// 		  for a-cyclic graph structures. 
        /// 		  </summary>
        META_PORT,
        
        /// <summary> An InitializerPort requires no bound channel, but rather,
        /// 		  synthesizes new datablocks based on it's DatablockTemplate's
        /// 		  InitialValue member when Pulled. </summary>
        INITIALIZER_PORT
    
    } PORTTYPE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent canonical descriptor functions for ports that are bound to
    ///             descriptor ports. A descriptor port is used to automatically generate datablocks
    ///             that describe other datablocks. The canonical example is kernel code such as:
    ///             "void op(float * vector, int N)" where N describes the number of entries in the
    ///             vector. Because N can be derived from the datablock for vector, we can bind N's
    ///             port as a descriptor port for vector, alleviating the burden of managing extra
    ///             channels and allocating extra datablocks in user code.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum descfun_t {
        
        /// <summary> Derive and integer-valued buffer size
        /// 		  from the described port and push it into the
        /// 		  descriptor port when ever a block is pushed into the
        /// 		  described port. 
        /// 		  </summary>
        DF_SIZE,
        
        /// <summary> Determine whether the described block is an EOF block
        /// 		  an push an EOF block into the descriptor port if so. 
        /// 		  </summary>
        DF_EOF,

        /// <summary> Determine whether the described block is a BOF block
        /// 		  and push a BOF block into the descriptor port if so. 
        /// 		  </summary>
        DF_BOF,

        /// <summary> Split a block's meta-data from it's data: a new block
        /// 		  with the described block's meta-data in its data channel
        /// 		  is pushed into the descriptor port. Gives a kernel direct
        /// 		  access to block meta data, which is typically only used
        /// 		  by the runtime. 
        /// 		  </summary>
		DF_METADATA_SPLITTER,

        /// <summary> Push a block into the descriptor port with the verbatim
        /// 		  control code in the data channel buffer.
        /// 		  </summary>
        DF_CONTROL_CODE,

        /// <summary> Push a block into the descriptor port with the datablock's
        /// 		  unique identifier number in the data channel buffer. Since
        /// 		  datablock UIDs are monotonically increasing (and unmutable),
        /// 		  this descriptor function gives ptask kernel code access that
        /// 		  can help it determine a block's relative position in a stream
        /// 		  that has been chunked across multiple datablocks.
        /// 		  </summary>
		DF_BLOCK_UID,

        /// <summary>   The descriptor port provides the data for the data
        ///             channel of the described port's output block. Valid
        ///             only for output ports!
        ///             </summary>
        DF_DATA_SOURCE,

        /// <summary>   The descriptor port provides the data for the metadata
        ///             channel of the described port's output block. Valid
        ///             only for output ports!
        ///             </summary>
        DF_METADATA_SOURCE,

        /// <summary>   The descriptor port provides the data for the template
        ///             channel of the described port's output block. Valid
        ///             only for output ports!
        ///             </summary>
        DF_TEMPLATE_SOURCE,

        /// <summary>   The descriptor port provides the the buffer dimensions
        ///             structure corresponding to the data channel in the
        ///             block.
        ///             </summary>        
        DF_DATA_DIMENSIONS

    } DESCRIPTORFUNC;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent canonical meta function types. Meta-functions specify
    /// 			how the runtime should interpret blocks received on MetaPorts. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum metafun_t {
        
        /// <summary> Not used </summary>
        MF_NONE = 0,

        /// <summary> The received block should be interpreted as an
        /// 		  allocation size for a downstream output port. The
        /// 		  allocation port must be specified explicitly by the
        /// 		  programmer. </summary>
        MF_ALLOCATION_SIZE = 1,

        /// <summary> The received block should be interpreted as an integer-valued
        /// 		  upper bound for simple-iteration dispatches. Simple iteration
        /// 		  is where the iteration involves only a single Task node, and
        /// 		  no acyclic graph structures. 
        /// 		  </summary>
        MF_SIMPLE_ITERATOR = 2,

        /// <summary> The received block should be interpreted as an integer-valued
        /// 		  upper bound for general-iteration dispatches. General iteration
        /// 		  is where the iteration involves acyclic graph structures, so control
        /// 		  signal propagation paths must also be specified by the programmer.
        /// 		  </summary>
        MF_GENERAL_ITERATOR = 3,

        /// <summary> The received block should be interpreted as an
        /// 		  allocation size for a downstream output port--however,
        ///           the allocaiton </summary>
        MF_PADDED_ALLOCATION_SIZE = 4,

        /// <summary> A user defined meta-function. Currently not used. </summary>
        MF_USER_DEFINED = 5
    
    } METAFUNCTION;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Host memory extent. Used for intializers. </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct HOSTMEMORYEXTENT_t {

        /// <summary>   The address of the memory extent. </summary>
        void *      lpvAddress;

        /// <summary>   The size of the memory extent in bytes. </summary>
        UINT        uiSizeBytes;

        /// <summary>   true if the physical memory backing the extent is pinned. </summary>
        BOOL        bPinned;            

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        HOSTMEMORYEXTENT_t(
            VOID
            ) : lpvAddress(NULL),       
                uiSizeBytes(0),
                bPinned(FALSE) {}

        HOSTMEMORYEXTENT_t(
            VOID * _lpvAddress,
            UINT _uiSizeBytes,
            BOOL _bPinned
            ) : lpvAddress(_lpvAddress),       
                uiSizeBytes(_uiSizeBytes),
                bPinned(_bPinned) {}

    } HOSTMEMORYEXTENT;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   a structure describing the logical meta data associated with a
    ///             buffer. </summary>
    ///
    /// <remarks>   Crossbac, 2/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct __bufferDims_t {

        /// <summary>   The number of elements in X. </summary>
        UINT uiXElements;

        /// <summary>   The number of elements in Y. </summary>
        UINT uiYElements;

        /// <summary>   The number of elements in Z. </summary>
        UINT uiZElements;

        /// <summary>   The number of bytes per element (stride). </summary>
        UINT cbElementStride;

        /// <summary>   The pitch: length in bytes of each row, if padding
        ///             is required to optimize traversal of the data. 
        ///             </summary>
        UINT cbPitch;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        __bufferDims_t(
            void
            ) { Initialize(0, 0, 0, 0, 0); }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   copy constructor. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        __bufferDims_t(
            const __bufferDims_t& o
            ) 
        { 
            Initialize(o); 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor for un-dimensioned buffers </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiBytes">  The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        __bufferDims_t(
            UINT uiBytes
            ) { Initialize(uiBytes, 1, 1, 1, uiBytes); }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiX">      The x coordinate. </param>
        /// <param name="uiY">      The y coordinate. </param>
        /// <param name="uiZ">      The z coordinate. </param>
        /// <param name="cbStride"> The stride. </param>
        ///-------------------------------------------------------------------------------------------------

        __bufferDims_t(
            UINT uiX,
            UINT uiY,
            UINT uiZ,
            UINT cbStride
            ) { Initialize(uiX, uiY, uiZ, cbStride, cbStride*uiX); }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiX">          The x coordinate. </param>
        /// <param name="uiY">          The y coordinate. </param>
        /// <param name="uiZ">          The z coordinate. </param>
        /// <param name="cbStride">     The stride. </param>
        /// <param name="cbPitchReq">   The pitch request. </param>
        ///-------------------------------------------------------------------------------------------------

        __bufferDims_t(
            UINT uiX,
            UINT uiY,
            UINT uiZ,
            UINT cbStride,
            UINT cbPitchReq
            ) { Initialize(uiX, uiY, uiZ, cbStride, cbPitchReq); }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default initializer. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void
        Initialize(
            void
            ) 
        {
            uiXElements = 0;
            uiYElements = 0;
            uiZElements = 0;
            cbElementStride = 0;
            cbPitch = 0;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes this object. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="o">    [in,out] The __bufferDims_t&amp; to process. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        Initialize(
            const __bufferDims_t& o
            ) 
        {
            uiXElements = o.uiXElements;
            uiYElements = o.uiYElements;
            uiZElements = o.uiZElements;
            cbElementStride = o.cbElementStride;
            cbPitch = o.cbPitch;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializer for un-dimensioned buffers </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiBytes">  The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        Initialize(
            UINT uiBytes
            ) 
        {
            uiXElements = uiBytes;
            uiYElements = 1;
            uiZElements = 1;
            cbElementStride = 1;
            cbPitch = uiBytes;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializer. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiX">      The x coordinate. </param>
        /// <param name="uiY">      The y coordinate. </param>
        /// <param name="uiZ">      The z coordinate. </param>
        /// <param name="cbStride"> The stride. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        Initialize(
            UINT uiX,
            UINT uiY,
            UINT uiZ,
            UINT cbStride
            ) 
        {
            uiXElements = uiX;
            uiYElements = uiY;
            uiZElements = uiZ;
            cbElementStride = cbStride;
            cbPitch = cbStride*uiX;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializer. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiX">          The x coordinate. </param>
        /// <param name="uiY">          The y coordinate. </param>
        /// <param name="uiZ">          The z coordinate. </param>
        /// <param name="cbStride">     The stride. </param>
        /// <param name="cbPitchReq">   The pitch request. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        Initialize(
            UINT uiX,
            UINT uiY,
            UINT uiZ,
            UINT cbStride,
            UINT cbPitchReq
            ) 
        {
            uiXElements = uiX;
            uiYElements = uiY;
            uiZElements = uiZ;
            cbElementStride = cbStride;
            cbPitch = cbPitchReq;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the allocation size in bytes. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT AllocationSizeBytes(
            void
            ) 
        {
            return cbPitch * uiYElements * uiZElements;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of logical elements. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT TotalElements(
            void
            ) 
        {
            return uiXElements * uiYElements * uiZElements;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has padded pitch. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <returns>   true if padded pitch, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasPaddedPitch(
            void
            )
        {
            return uiXElements * cbElementStride != cbPitch;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'oBufDims' is an exact (member-wise) match. </summary>
        ///
        /// <remarks>   crossbac, 4/3/2013. </remarks>
        ///
        /// <param name="oBufDims"> The buffer dims. </param>
        ///
        /// <returns>   true if exact match, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsExactMatch(
            const __bufferDims_t& oBufDims
            )
        {
            return uiXElements == oBufDims.uiXElements && 
                   uiYElements == oBufDims.uiYElements && 
                   uiZElements == oBufDims.uiZElements && 
                   cbElementStride == oBufDims.cbElementStride && 
                   cbPitch == oBufDims.cbPitch;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'oBufDims' has the same allocation size as this. </summary>
        ///
        /// <remarks>   crossbac, 4/3/2013. </remarks>
        ///
        /// <param name="oBufDims"> The buffer dims. </param>
        ///
        /// <returns>   true if allocation size match, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsAllocationSizeMatch(
            __bufferDims_t& oBufDims
            )
        {
            UINT uiOtherDims = oBufDims.AllocationSizeBytes();
            return AllocationSizeBytes() == uiOtherDims; 
        }


    } BUFFERDIMENSIONS;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Defines types for outstanding async operations--this enables
    ///             us to provide a single-writer multiple reader serialization,
    ///             which enables more concurrency than simply forcing all
    ///             subsequent PBuffer users to be ordered after the most
    ///             recent outstanding operation, because the most recent
    ///             operation is not guaranteed to conflict. </summary>
    ///
    /// <remarks>   crossbac, 5/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum __outstanding_op_type {
        OT_LAUNCH_TARGET_READABLE,      // concurrent readers ok
        OT_LAUNCH_TARGET_WRITABLE,      // all other readers and writers must serialize after this dep
        OT_MEMCPY_TARGET,               // all other readers and writers must serialize after this dep
        OT_MEMCPY_SOURCE                // concurrent readers ok
    } ASYNCHRONOUS_OPTYPE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enum for different dedicated async context usage scenarios. In general, where
    ///             back-end platforms that support explicity async management apis, the
    ///             device/driver provide multiple copy and execute engines that can execute
    ///             concurrently. We want to avoid performing transfers on contexts we use to queue
    ///             GPU exec, since it puts data copy on the critical path when it may not need to be
    ///             on the critical path (because it prevents a subsequent pipelined invocation of
    ///             the same kernel from executing despite the lack of a data dependence).
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum __async_ctxt_type_t {
        ASYNCCTXT_DEFAULT,              // default async context for a given accelerator
        ASYNCCTXT_TASK,                 // execution context for a task on a given accelerator
        ASYNCCTXT_XFERHTOD,             // dedicated context for H to D transfers
        ASYNCCTXT_XFERDTOH,             // dedicated context for D to H transfers
        ASYNCCTXT_XFERDTOD,             // dedicated context for D to D transfers 
        ASYNCCTXT_UNDEFINED             // shouldn't be used--this is the type assigned for accelerators
                                        // that have no explicity async API.
    } ASYNCCONTEXTTYPE;

    #define ASYNCCTXT_NUMTYPES      6
    #define ASYNCCTXT_ISDEFAULT(x)  ((x) == ASYNCCTXT_DEFAULT)
    #define ASYNCCTXT_ISEXECCTXT(x) ((x) == ASYNCCTXT_TASK)
    #define ASYNCCTXT_ISXFERCTXT(x) (((x)==ASYNCCTXT_XFERHTOD) || ((x)==ASYNCCTXT_XFERDTOH) || ((x)==ASYNCCTXT_XFERDTOD))

    static const char * g_lpszAsyncContextTypeStrings[] = {
        "DEFAULT",
        "TASK",
        "XFERHTOD",
        "XFERDTOH",
        "XFERDTOD",
        "UNDEFINED"
    };

    #define AsyncCtxtTypeToString(eCtxtType) (g_lpszAsyncContextTypeStrings[eCtxtType])

    typedef enum _graph_state_t {

        /// <summary>   An enum constant representing the initializing state. 
        ///             A graph is in the initializing state from the time it enters
        ///             its constructor, until the time it has been finalized and
        ///             its OnGraphComplete method has been called. It can exit the
        ///             initializing state by entering the runnable state, indicating
        ///             it is can be run but is "stopped". In such a case, all of its'
        ///             graph runner threads are up and ready, but should be blocked
        ///             on the run signal. It can also transition straight to the tearing
        ///             down state. No other exits are legal.  
        ///             </summary>

        PTGS_INITIALIZING = 0,

        /// <summary>   An enum constant representing the graph's runnable state. 
        ///             All data structures are initialized, and the graph runner threads
        ///             are up, but blocked waiting for a signal to enter the run state. 
        ///             A graph can go back and forth from runnable --> running --> quiescing
        ///             any number of times during its life cycle. Legal exits from this
        ///             state: RUNNING, TEARINGDOWN.
        ///             </summary>

        PTGS_RUNNABLE = 1,

        /// <summary>   An enum constant representing the graph's running state. 
        ///             All graph runner threads are up, and tasks from this graph
        ///             can be dispatched as they become ready. The only legal exit 
        ///             from this state is through the quiescing state. The only legal
        ///             entry is through the RUNNABLE state.
        ///             </summary>

        PTGS_RUNNING = 2,

        /// <summary>   An enum constant representing the graph quiescing state. 
        ///             A graph is leaving the RUNNING state, but there may be 
        ///             outstanding dispatches in the scheduler's run queue or
        ///             in flight. This state represents the period between the call
        ///             to stop the graph and the point when all outstanding task
        ///             dispatches from this graph are either complete or are
        ///             in the scheduler's deferred queue. Legal entry only through
        ///             the RUNNING state, legal exit only to the RUNNABLE state.
        ///             </summary>

        PTGS_QUIESCING = 3,

        /// <summary>   An enum constant representing the graph tearingdown state. 
        ///             The teardown method of the graph has been called but the work
        ///             to collect all outstanding objects for this graph is not yet
        ///             complete. Legal entry is only through the RUNNABLE state, 
        ///             legal exit only to the TEARDOWNCOMPLETE method. 
        ///             </summary>

        PTGS_TEARINGDOWN = 4,

        /// <summary>   An enum constant representing the teardowncomplete state. 
        ///             The graph has been cleaned up and is ready for its destructor
        ///             to be called. Only legal entry is through TEARINGDOWN.
        ///             </summary>

        PTGS_TEARDOWNCOMPLETE = 5

    } GRAPHSTATE;

    static const char * g_lpszGraphStateStrings[] = {
        "INITIALIZING",
        "RUNNABLE",
        "RUNNING",
        "QUIESCING",
        "TEARINGDOWN",
        "TEARDOWNCOMPLETE"
    };

    #define IsGraphAlive(pState) (((*pState) != PTGS_TEARINGDOWN) && ((*pState) != PTGS_TEARDOWNCOMPLETE))
    #define IsGraphRunning(pState) ((*pState) == PTGS_RUNNING)
    #define GraphStateToString(eState) (g_lpszGraphStateStrings[(int)eState]);

    #define ASYNCOP_ISREAD(x) ((x==OT_LAUNCH_TARGET_READABLE) || (x==OT_MEMCPY_SOURCE))
    #define ASYNCOP_ISWRITE(x) ((x==OT_LAUNCH_TARGET_WRITABLE) || (x==OT_MEMCPY_TARGET))
    #define ASYNCOP_ISXFER(x) ((x==OT_MEMCPY_SOURCE) || (x==OT_MEMCPY_TARGET))
    #define ASYNCOP_ISEXEC(x) ((x==OT_LAUNCH_TARGET_READABLE) || (x==OT_LAUNCH_TARGET_WRITABLE))

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Values that represent view materialization policies for a channel. 
    ///     1. VIEWMATERIALIZATIONPOLICY_ON_DEMAND:  means the runtime will not attempt to start
    ///        materializing host-side views of data produced on output channels until the consumer 
    ///        requests the data.
    ///     2. VIEWMATERIALIZATIONPOLICY_EAGER: means the runtime will start materializing host-side
    ///        views of data produced on an output channel assuming it can hide some latency before 
    ///        the user requests the data.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum viewmaterializationpolicy_t {
        
        /// <summary> no data transfer will be started as a side-effect of
        /// 		  pushing datablocks into this channel.
        /// 		   </summary>
        VIEWMATERIALIZATIONPOLICY_ON_DEMAND = 0,
        
        /// <summary> start data transfer on output channels when
        /// 		  dispatch completes.
        /// 		  </summary>
        VIEWMATERIALIZATIONPOLICY_EAGER = 1
    
    } VIEWMATERIALIZATIONPOLICY;

    static const char * g_lpszMaterializationPolicyStrings[] = {
        "DEMAND",
        "EAGER",
    };
    #define ViewPolicyString(ePolicy) (g_lpszMaterializationPolicyStrings[(int)ePolicy])

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   3D block/grid dimension struct. </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct ptask_dim3_t {
        int x;
        int y;
        int z;
        ptask_dim3_t(int xx=0, int yy=0, int zz=0) :x(xx), y(yy), z(zz) { }
    } PTASKDIM3;

    typedef struct ptask_usage_stats_t {
        double  dLastWaitTime;
        double  dAverageWaitTime;
        double  dLastDispatchTime;
        double  dAverageDispatchTime;
        double  dAverageOSPrio;
     } PTASKUSAGESTATS, *PPTASKUSAGESTATS;

    typedef enum schedmode_t {
        SCHEDMODE_COOPERATIVE = 0,
        SCHEDMODE_PRIORITY = 1,
        SCHEDMODE_DATADRIVEN = 2,
        SCHEDMODE_FIFO = 3
    } SCHEDULINGMODE;

    static const char * g_lpszSchedulingModeStrings[] = {
        "Cooperative",
        "Priority",
        "Data-Aware",
        "FIFO"
    };
    #define SchedulingModeString(ePolicy) (g_lpszSchedulingModeStrings[(int)ePolicy])

    typedef enum graphpartitionmode_t {
        GRAPHPARTITIONINGMODE_NONE = 0,
        GRAPHPARTITIONINGMODE_HINTED = 1,
        GRAPHPARTITIONINGMODE_HEURISTIC = 2,
        GRAPHPARTITIONINGMODE_OPTIMAL = 3
    } GRAPHPARTITIONINGMODE;

    static const char * g_lpszGraphPartitioningModeStrings[] = {
        "None",
        "Hinted",
        "Heuristic",
        "Optimal"
    };
    #define GraphPartitioningModeString(eMode) (g_lpszGraphPartitioningModeStrings[(int)eMode])

    typedef enum threadpoolpolicy_t {
        TPP_AUTOMATIC,    // let the scheduler decide how many threads to use
        TPP_EXPLICIT,     // the programmer must provide the task pool size
        TPP_THREADPERTASK // always use 1:1 thread:task mapping (original PTask policy) 
    } THREADPOOLPOLICY;

    static const char * g_lpszThreadPoolPolicyStrings[] = {
        "AUTO",
        "EXPLICIT",
        "THREADPERTASK"
    };
    #define ThreadPoolPolicyString(ePolicy) (g_lpszThreadPoolPolicyStrings[(int)ePolicy])

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Policy for how to handle pool re-entry blocks that are drawn from a pool 
    ///             and then resized. Should such blocks just return to their pool, enter
    ///             another pool, etc.?</summary>
    ///
    /// <remarks>   crossbac, 8/30/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum blockpoolresizepolicy_t {

        /// <summary>   pooled blocks that are resized leave the block pool, and do not enter a new pool, 
        ///             so the release that returned the block to the pool causes it to be queued for
        ///             GC instead. </summary>
        
        BPRSP_EXIT_POOL = 0,              
        
        /// <summary>   Pooled blocks that are resized retain their current pool membership. 
        ///             This policy should be used with caution because subsequent requests for block 
        ///             with a size that matches the returned block will not be able to be satisfied with 
        ///             this block, since it will be effictively hiding in a pool with a smaller
        ///             size. We support this policy for convenience--it's clearly the 
        ///             simplest to implement. </summary>

        BPRSP_REMAIN_IN_POOL = 1,         

        /// <summary>   pooled blocks that are resized leave the pool, and join another existing pool,
        ///             if one can be found--otherwise the block is queued for GC. </summary>

        BPRSP_FIND_EXISTING_POOL = 2,

        /// <summary>   pooled blocks that are resized leave the pool, and join a new one, entering an 
        ///             existing pool if there is a good fit, or by creating a new pool to enter. This
        ///             is likely the most rational policy to support, but is also the most difficult
        ///             to support, since it has the potential to put a lot work and memory allocation
        ///             on the critical path by hiding it in Datablock::Release() calls. 
        ///             </summary>
        
        BPRSP_FIND_OR_CREATE_POOL = 3     

    } BLOCKPOOLRESIZEPOLICY;

    static const char * g_lpszBlockPoolResizePolicyStrings[] = {
        "EXIT_POOL",
        "REMAIN_IN_POOL",
        "MIGRATE_EXISTING_POOL",
        "MIGRATE_EXISTING_OR_CREATE"
    };
    #define BlockPoolResizePolicyString(ePolicy) (g_lpszBlockPoolResizePolicyStrings[(int)ePolicy])

    typedef enum blockresizememspacepolicy_t {

        /// <summary>   If a block is resized and has buffers in other memory spaces, just invalidate
        ///             those buffers (potentially syncing the host view) and invalidate/release 
        ///              </summary>

        BRSMSP_RELEASE_DEVICE_BUFFERS = 0,

        /// <summary>   If a block is resized and has buffers in other memory spaces, grow 
        ///             those buffers too (potentially syncing the host view) to make sure
        ///             we don't have to do a bunch of device-side copy work with the realloc
        ///             </summary>

        BRSMSP_GROW_DEVICE_BUFFERS = 1,

    } BLOCKRESIZEMEMSPACEPOLICY;

    static const char * g_lpszBlockResizeMemspacePolicyStrings[] = {
        "RELEASE_DEVICE_BUFFERS",
        "GROW_DEVICE_BUFFERS"
    };
    #define BlockResizeMemspacePolicyString(ePolicy) (g_lpszBlockResizeMemspacePolicyStrings[(int)ePolicy])

    typedef enum graphassigpolicy_t {
        GMP_USER_DEFINED = 0,
        GMP_ROUND_ROBIN = 1
    } GRAPHASSIGNMENTPOLICY;

    typedef enum datablockaffinitypolicy_t {
        DAP_IGNORE_AFFINITY = 0,
        DAP_BOUND_TASK_AFFINITY = 1,
        DAP_TRANSITIVE_AFFINITY = 2
    } DATABLOCKAFFINITYPOLICY;

    static const char * g_lpszGraphAssignmentPolicyStrings[] = {
        "USER_DEFINED",
        "ROUND_ROBIN"
    };
    #define GraphAssignmentPolicyString(ePolicy) (g_lpszGraphAssignmentPolicyStrings[(int)ePolicy])

    typedef enum threadrole_t {
        PTTR_UNKNOWN = 0,
        PTTR_GRAPHRUNNER = 1,
        PTTR_SCHEDULER = 2,
        PTTR_GC = 3,
        PTTR_APPLICATION = 4
    } PTTHREADROLE;

    static const char * g_lpszThreadRoleStrings[] = {
        "UNKNOWN_THREAD_ROLE",
        "GRAPHRUNNER_THREAD",
        "SCHEDULER_THREAD",
        "PTASK_GC_THREAD",
        "APPLICATION_THREAD"
    };
    #define ThreadRoleString(eRole) (g_lpszThreadRoleStrings[(int)eRole])

};

#endif