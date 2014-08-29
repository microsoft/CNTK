///-------------------------------------------------------------------------------------------------
// file:	PTaskRuntime.h
//
// summary:	Declares the task runtime namespace
///-------------------------------------------------------------------------------------------------

#ifndef _PTASK_RUNTIME_API_H_
#define _PTASK_RUNTIME_API_H_

#include <Windows.h>
#include "primitive_types.h"
#include "ptaskutils.h"
#include <iostream>
#include <sstream>

class CSharedPerformanceTimer;

namespace PTask { 

    class CompiledKernel;
    class Graph;
    class Channel;
    class Port;
    class Task;
    class Datablock;    
    class DatablockTemplate;

    namespace Runtime {
    
    using namespace PTask;

	typedef enum instrumentationmetric_t {
		NONE = 0,
		ACTIVITY_FACTOR = 1,
		MEMORY_EFFICIENCY = 2,
		BRANCH_DIVERGENCE = 3,
		CLOCK_CYCLE_COUNT = 4
	} INSTRUMENTATIONMETRIC;

    typedef enum ptasksubsystem_t {
        PTSYS_TASKS = 0,
        PTSYS_TASK_MIGRATION = 1,
        PTSYS_PBUFFERS = 2,
        PTSYS_DATABLOCKS = 3,
        PTSYS_COHERENCE = 4,
        PTSYS_CHANNELS = 5,
        PTSYS_DISPATCH = 6,
        PTSYS_REFCOUNT_OBJECTS = 7,
        PTSYS_ADHOC_INSTRUMENTATION = 8
    } PTASKSUBSYSTEM;
    
    BOOL SubsystemReport(PTASKSUBSYSTEM eSubSystem);
    BOOL SubsystemReport(PTASKSUBSYSTEM eSubSystem, std::ostream& ios);
    void PrintRuntimeConfiguration();
    void PrintRuntimeConfiguration(std::ostream& ios);
    BOOL GetApplicationThreadsManagePrimaryContext();
    UINT GetGCSweepThresholdPercent();
    BOOL GetAggressiveReleaseMode();
    BOOL GetBlockPoolsEnabled();
    BOOL GetCriticalPathAllocMode();
    BOOL GetSignalProfilingEnabled();
    BOOL GetRCProfilingEnabled();
    BOOL GetDBProfilingEnabled();
    BOOL GetCTProfilingEnabled();
    BOOL GetTPProfilingEnabled();
    BOOL GetPBufferProfilingEnabled();
    BOOL GetInvocationCountingEnabled();
    BOOL GetChannelProfilingEnabled();
    BOOL GetBlockPoolProfilingEnabled();
    BOOL GetEnableDebugLogging();
    BOOL GetAdhocInstrumentationEnabled();
    BOOL GetPBufferClearOnCreatePolicy();
    BOOL GetProfilePSDispatch();
    UINT GetSizeDescriptorPoolSize();
    BOOL HasGlobalThreadPool();
    UINT GetGlobalThreadPoolSize();
    BOOL GetPrimeGlobalThreadPool();  
    BOOL GetGlobalThreadPoolGrowable();
    BLOCKPOOLRESIZEPOLICY GetBlockPoolBlockResizePolicy();
    BLOCKRESIZEMEMSPACEPOLICY GetBlockResizeMemorySpacePolicy();
    GRAPHASSIGNMENTPOLICY GetGraphAssignmentPolicy();
    BOOL GetScheduleChecksClassAvailability();
    DATABLOCKAFFINITYPOLICY GetDatablockAffinitiyPolicy();
    BOOL GetHarmonizeInitialValueCoherenceState();
    UINT GetOptimalPartitionerEdgeWeightScheme();
    UINT GetAsyncContextGCQueryThreshold();
    BOOL GetGraphMutabilityMode();

    void SetGraphMutabilityMode(BOOL bGraphsAreMutable);
    void SetApplicationThreadsManagePrimaryContext(BOOL bUseCtxt);
    void SetScheduleChecksClassAvailability(BOOL bCheck);
    void SetGCSweepThresholdPercent(UINT uiPercent);
    void SetAggressiveReleaseMode(BOOL bEnable);
    void SetBlockPoolsEnabled(BOOL bEnable);
    void SetCriticalPathAllocMode(BOOL bEnable);
    void SetSignalProfilingEnabled(BOOL bEnable);
    void RegisterSignalForProfiling(CONTROLSIGNAL luiControlSignal);
    void UnregisterSignalForProfiling(CONTROLSIGNAL luiControlSignal);
    void SetRCProfilingEnabled(BOOL bEnable);
    void SetDBProfilingEnabled(BOOL bEnable);
    void SetCTProfilingEnabled(BOOL bEnable);
    void SetTPProfilingEnabled(BOOL bEnable);
    void SetPBufferProfilingEnabled(BOOL bEnable);
    void SetInvocationCountingEnabled(BOOL bEnable);
    void SetChannelProfilingEnabled(BOOL bEnable);
    void SetBlockPoolProfilingEnabled(BOOL bEnable);
    void SetEnableDebugLogging(BOOL bEnable);
    void SetAdhocInstrumentationEnabled(BOOL bEnable);
    BOOL SetPBufferClearOnCreatePolicy(BOOL bClearOnCreate);
    void SetProfilePSDispatch(BOOL bProfile);
    void RequireBlockPool(int nDataSize, int nMetaSize, int nTemplateSize, int nBlocks=0);
    void RequireBlockPool(DatablockTemplate * pTemplate, int nBlocks);
    void SetSizeDescriptorPoolSize(UINT uiPoolSize);
    void SetGlobalThreadPoolSize(UINT uiThreads);
    void SetPrimeGlobalThreadPool(BOOL b);   
    void SetGlobalThreadPoolGrowable(BOOL b);
    void SetBlockPoolBlockResizePolicy(BLOCKPOOLRESIZEPOLICY policy);
    void SetBlockResizeMemorySpacePolicy(BLOCKRESIZEMEMSPACEPOLICY policy);
    void SetGraphAssignmentPolicy(GRAPHASSIGNMENTPOLICY policy);
    void SetDatablockAffinitiyPolicy(DATABLOCKAFFINITYPOLICY policy);
    void SetHarmonizeInitialValueCoherenceState(BOOL b);
    void SetOptimalPartitionerEdgeWeightScheme(UINT s);
    void SetAsyncContextGCQueryThreshold(UINT uiThreshold);

    extern BOOL g_bScheduleChecksClassAvailability;
    extern VIEWMATERIALIZATIONPOLICY g_eDefaultViewMaterializationPolicy;
    extern VIEWMATERIALIZATIONPOLICY g_eDefaultOutputViewMaterializationPolicy;
    extern BOOL                      g_bRCProfilingEnabled;
    extern BOOL                      g_bDBProfilingEnabled;
    extern BOOL                      g_bCTProfilingEnabled;
    extern BOOL                      g_bTPProfilingEnabled;
    extern BOOL                      g_bPBufferProfilingEnabled;
    extern BOOL                      g_bInvocationCountingEnabled;
    extern BOOL                      g_bChannelProfilingEnabled;
    extern BOOL                      g_bBlockPoolProfilingEnabled;
    extern BOOL                      g_bEnableDebugLogging;
    extern BOOL                      g_bTaskDispatchLocksIncomingAsyncSources;
    extern BOOL                      g_bThreadPoolSignalPerThread;
    extern BOOL                      g_bTaskDispatchReadyCheckIncomingAsyncDeps;
    extern BOOL                      g_bTaskDispatchLocklessIncomingDepWait;
    extern BOOL                      g_bExtremeTrace;
    extern BOOL                      g_bCoherenceProfile;
    extern BOOL                      g_bTaskProfile;
    extern BOOL                      g_bTaskProfileVerbose;
    extern BOOL                      g_bPageLockingEnabled;
    extern BOOL                      g_bAggressivePageLocking;
    extern BOOL                      g_bDebugAsynchrony;
    extern BOOL                      g_bProfilePBuffers;
    extern BOOL                      g_bEagerMetaPorts;
    extern BOOL                      g_bTrackDeviceAllocation;
    extern BOOL                      g_bUseGraphWatchdog;
    extern DWORD                     g_dwGraphWatchdogThreshold;
    extern BOOL                      g_bUserDefinedCUDAHeapSize;
    extern BOOL                      g_bInitCublas;
    extern BOOL                      g_bSortThreadPoolQueues;
    extern BOOL                      g_bProvisionBlockPoolsForCapacity;
    extern BOOL                      g_bAdhocInstrumentationEnabled;
    extern BOOL                      g_bSetPBufferClearOnCreatePolicy;
    extern BOOL                      g_bProfilePSDispatch;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the PTask runtime.
    ///             
    ///             Check success or failure using PTSUCCESS() macro.
    ///             
    ///             The *intended* idiom for Initialize/Teardown is that these APIs will be called at
    ///             most once per address-space. PTask will create a multitude of threads to manage
    ///             graphs and accelerators, but it expects a singleton instance of the runtime, so
    ///             while calls to init and teardown are thread-safe with respect to the singleton
    ///             runtime instance, concurrent calls to initialize or concurrent calls to teardown
    ///             are not guaranteed to leave PTask in a consistent state. Multiple init-teardown
    ///             cycles per address space are supported provided there is no concurrency amongst
    ///             those cycles.
    ///             
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT Initialize();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the runtime is initialized. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <returns>   see summary </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL IsInitialized();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Terminate the PTask runtime. 
    ///
    ///             Check success or failure using PTSUCCESS() macro.
    ///
    ///             The *intended* idiom for Initialize/Teardown is that these APIs will be called at
    ///             most once per address-space. PTask will create a multitude of threads to manage
    ///             graphs and accelerators, but it expects a singleton instance of the runtime, so
    ///             while calls to init and teardown are thread-safe with respect to the singleton
    ///             runtime instance, concurrent calls to initialize or concurrent calls to teardown
    ///             are not guaranteed to leave PTask in a consistent state. Multiple init-teardown
    ///             cycles per address space are supported provided there is no concurrency amongst
    ///             those cycles.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT Terminate();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the force GC. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT ForceGC(BOOL bCollectDanglingRefs=FALSE);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Given the name of a file containing accelerator
    /// 			code, return the class of an accelerator
    /// 			capable of running it. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="szAcceleratorCodeFileName">    Filename of the accelerator code file. </param>
    ///
    /// <returns>   On success, the ACCELERATOR_CLASS of a matching accelerator.
    /// 			On failure, ACCELERATOR_CLASS unknown. </returns>
    ///-------------------------------------------------------------------------------------------------

    ACCELERATOR_CLASS
    GetAcceleratorClass(
        const char * szAcceleratorCodeFileName
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queries if we can execute the kernel 'szAcceleratorCodeFileName' (or think we
    ///             can). This boils down to checking for platform support for the accelerator class
    ///             that would be chosen to compile and run the file. The runtime may or may not be
    ///             started with that support enabled by the programmer, and the runtime environment
    ///             may or may not be able to find an accelerator device for it even if the support
    ///             is enabled in PTask. For example, clients using RDP typically can only use host
    ///             tasks. Machines with nVidia Tesla cards cannot run DirectX programs without the
    ///             reference driver, etc. This should be called after the runtime is initialized,
    ///             and will check whether an appropriate device could be found by the scheduler.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///
    /// <param name="szAcceleratorCodeFileName">    Filename of the accelerator code file. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CanExecuteKernel(
        const char * szAcceleratorCodeFileName
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate accelerators of a given class. 
    /// 			Call this function from user code to find out what accelerators
    /// 			are available to run a particular piece of accelerator code.
    /// 			To enumerate all available accelerators, pass ACCELERATOR_CLASS_UNKNOWN.
    /// 			Caller is responsible for:
    /// 			1. Incrementing the enumeration index.  
    /// 			2. Freeing the returned descriptor using free().  
    /// 			The function returns PTASK_OK until no more accelerators
    /// 			are found.
    /// 		    </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="eAcceleratorClass">        The accelerator class. Pass ACCELERATOR_CLASS_UNKNOWN
    /// 										to enumerate accelerators of all types.</param>
    /// <param name="uiAcceleratorIndex">       Index of the accelerator being enumerated. </param>
    /// <param name="ppAcceleratorDescriptor">  [out] If non-null, an ACCELERATOR_DESCRIPTOR
    /// 										describing the accelerator at that index. The 
    /// 										caller must free the descriptor. If null, the function
    /// 										return code indicates whether an accelerator exists
    /// 										at that index, but does not provide a descriptor.
    /// 										</param>
    ///
    /// <returns>   PTASK_OK if enumeration succeeds.
    ///             PTASK_ERR_UNINITIALIZED if the runtime is not initialized.
    /// 			</returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    EnumerateAccelerators(
        ACCELERATOR_CLASS eAcceleratorClass,
        UINT uiEnumerationIndex,
        ACCELERATOR_DESCRIPTOR ** ppAcceleratorDescriptor
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform specific runtime version for a given accelerator class.
    ///             Currently assumes that all devices of a particular class have the same version,
    ///             which is sufficient for all the needs we currently have (this API is used to
    ///             select compiler settings for autogenerated code), but when there are GPUs with
    ///             e.g. different compute capabilities, this API will assert and return failure. In
    ///             the future, we may need to enumerate all available versions, but at the moment
    ///             that is a lot of needless complexity to support a use case we don't have.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 8/29/2013. </remarks>
    ///
    /// <param name="eAcceleratorClass">    The accelerator class. </param>
    /// <param name="uiPSRuntimeVersion">   [in,out] The ps runtime version. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    GetPlatformSpecificRuntimeVersion(
        ACCELERATOR_CLASS eAcceleratorClass,
        UINT& uiPSRuntimeVersion
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Disables an accelerator before PTask initialization: this means
    ///             PTask will black list it, and if it encounters the accelerator
    ///             at initialization time, it immediately disables it. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2014. </remarks>
    ///
    /// <param name="eAcceleratorClass">    The accelerator class. </param>
    /// <param name="nPSDeviceID">          Identifier for the ps device. </param>
    ///
    /// <returns>   PTASK_OK for successful addition to the black list. 
    ///             PTASK_ERR_ALREADY_INITIALIZED if the runtime is already initialized.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    DisableAccelerator(
        ACCELERATOR_CLASS eAcceleratorClass,
        int  nPSDeviceID
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enable/disables an accelerator before PTask initialization: on disable, this means
    ///             PTask will black list it, and if it encounters the accelerator
    ///             at initialization time, it immediately calls the dynamic API to disable it. 
    ///             Enable is a NO-OP unless there is already a black list entry for it. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2014. </remarks>
    ///
    /// <param name="eAcceleratorClass">    The accelerator class. </param>
    /// <param name="nPSDeviceID">          Identifier for the ps device. </param>
    /// <param name="bEnable">              (Optional) the enable. </param>
    ///
    /// <returns>   PTASK_OK for successful addition/removal to/from the black list. 
    ///             PTASK_ERR_ALREADY_INITIALIZED if the runtime is already initialized.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    EnableAccelerator(
        ACCELERATOR_CLASS eAcceleratorClass,
        int  nPSDeviceID,
        BOOL bEnable=TRUE
        );


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Disables the accelerator indicated by the descriptor. "Disabled" means the
    ///             scheduler will not dispatch work on that accelerator.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pAcceleratorDescriptor">   [in,out] If non-null, information describing the
    ///                                         accelerator. </param>
    ///
    /// <returns>   PTASK_OK for successful disable.
    ///             PTASK_ERR_UNINITIALIZED if the runtime is not yet initialized.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    DynamicDisableAccelerator(
        ACCELERATOR_DESCRIPTOR * pAcceleratorDescriptor
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables the accelerator indicated by the descriptor. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pAcceleratorDescriptor">   [in,out] If non-null, information describing the
    ///                                         accelerator. </param>
    ///
    /// <returns>   PTASK_OK for successful enable.
    ///             PTASK_ERR_UNINITIALIZED if the runtime is not yet initialized.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    DynamicEnableAccelerator(
        ACCELERATOR_DESCRIPTOR * pAcceleratorDescriptor
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Disables the accelerator. "Disabled" means the scheduler will not
    ///             dispatch work on that accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   PTASK_OK for successful disable.
    ///             PTASK_ERR_UNINITIALIZED if the runtime is not yet initialized.
    ///             it should be impossible to get an accelerator pointer before
    ///             the runtime is initialized, FWIW.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    DynamicDisableAccelerator(
        void * pAccelerator
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables the accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   PTASK_OK for successful enable.
    ///             PTASK_ERR_UNINITIALIZED if the runtime is not yet initialized.
    ///             it should be impossible to get an accelerator pointer before
    ///             the runtime is initialized, FWIW.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    DynamicEnableAccelerator(
        void * pAccelerator
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets task-accelerator affinity. Given an accelerator id, set the affinity between the ptask
    ///     and that accelerator to the given affinity type.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pTask">            [in] non-null, the task. </param>
    /// <param name="uiAcceleratorId">  [in] accelerator identifier. </param>
    /// <param name="eAffinityType">   [in] affinity type. </param>
    ///
    /// <returns>
    ///     PTRESULT--use PTSUCCESS macro to check success. return PTASK_OK on success. returns
    ///     PTASK_ERR_INVALID_PARAMETER if the affinity combination requested cannot be provided by
    ///     the runtime.
    /// </returns>
    ///
    ///-------------------------------------------------------------------------------------------------

    PTRESULT SetTaskAffinity(
                Task * pTask,
                UINT uiAcceleratorId,
                AFFINITYTYPE eAffinityType
                );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets task-accelerator affinity. Given a list of accelerator ids, set the
    /// 			affinity between the ptask and each accelerator in the list
    /// 			to the affinity type at the same index in the affinity type
    /// 			list. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pTask">            [in] non-null, the task. </param>
    /// <param name="pvAcceleratorIds"> [in] non-null, list of accelerator identifiers</param>
    /// <param name="pvAffinityTypes">  [in] non-null, list of affinity types. </param>
    /// <param name="nAcceleratorIds">  List of identifiers for the accelerators. </param>
    ///
    /// <returns>   PTRESULT--use PTSUCCESS macro to check success. 
    /// 			return PTASK_OK on success.
    /// 			returns PTASK_ERR_INVALID_PARAMETER if the affinity
    /// 			combination requested cannot be provided by the runtime.</returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT SetTaskAffinity(
                Task * pTask,
                UINT * pvAcceleratorIds,
                AFFINITYTYPE * pvAffinityTypes,
                UINT nAcceleratorIds
                );

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
    /// 			NB: Thread group dimensions are optional parameters here.
    /// 			This is because some runtimes require them statically,
    /// 			and some do not. DirectX requires thread-group sizes to be
    /// 			specified statically to enable compiler optimizations that
    /// 			cannot be used otherwise. CUDA and OpenCL allow runtime
    /// 			specification of these parameters.
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
    /// <returns>	a new compiled kernel object if it succeeds, null if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    CompiledKernel * 
    GetCompiledKernel(
        char * lpszFile, 
        char * lpszOperation, 
        char * lpszCompilerOutput=NULL,
        int uiCompilerOutput=0,
        int tgx=1, 
        int tgy=1, 
        int tgz=1
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Compiles accelerator source code to create a PTask binary. This variant
    /// 			accomodates tasks that may have a global initializer routine that must be called
    /// 			before the graph enters the run state (e.g. creation of block pools).
    /// 			</summary>
    ///
    /// <remarks>	The function accepts a file name and an operation in the file to build a binary
    /// 			for. For example, "foo.hlsl" and "vectoradd" will compile the vectoradd() shader
    /// 			in foo.hlsl. On success the function will create platform-specific binary and
    /// 			module objects that can be later used by the runtime to invoke the shader code.
    /// 			The caller can provide a buffer for compiler output, which if present, the
    /// 			runtime will fill *iff* the compilation fails.
    /// 			***
    /// 			NB: Thread group dimensions are optional parameters here. This is because some
    /// 			runtimes require them statically, and some do not. DirectX requires thread-group
    /// 			sizes to be specified statically to enable compiler optimizations that cannot be
    /// 			used otherwise. CUDA and OpenCL allow runtime specification of these parameters.
    /// 			***
    /// 			If an initializer file and entry point are provided, the runtime will load the
    /// 			corresponding binary and call the entry point upon graph completion.
    /// 			</remarks>
    ///
    /// <param name="lpszFile">						[in] filename+path of source. cannot be null. </param>
    /// <param name="lpszOperation">				[in] Function name in source file. non-null. </param>
    /// <param name="lpszInitializerBinary">		[in] filename+path for initializer DLL. null OK. </param>
    /// <param name="lpszInitializerEntryPoint">	[in] entry point for initializer code. null OK. </param>
    /// <param name="eInitializerPSClass">			The initializer ps class. </param>
    /// <param name="lpszCompilerOutput">			[in,out] (optional)  On failure, the compiler
    /// 											output. </param>
    /// <param name="uiCompilerOutput">				(optional) [in] length of buffer supplied for
    /// 											compiler output. </param>
    /// <param name="tgx">							(optional) thread group X dimensions. (see
    /// 											remarks) </param>
    /// <param name="tgy">							(optional) thread group Y dimensions. (see
    /// 											remarks) </param>
    /// <param name="tgz">							(optional) thread group Z dimensions. (see
    /// 											remarks) </param>
    ///
    /// <returns>	a new compiled kernel object if it succeeds, null if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    CompiledKernel * 
    GetCompiledKernelEx(
        char * lpszFile, 
        char * lpszOperation, 
		char * lpszInitializerBinary,
		char * lpszInitializerEntryPoint,
		ACCELERATOR_CLASS eInitializerPSClass,
        char * lpszCompilerOutput=NULL,
        int uiCompilerOutput=0,
        int tgx=1, 
        int tgy=1, 
        int tgz=1
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a compiled kernel object for a provided host function. </summary>
    ///
    /// <remarks>   jcurrey, 2/24/2014. </remarks>
    ///
    /// <param name="lpszOperation">   [in] Function name, used only for identification. Cannot be null.</param>
    /// <param name="lpfn">            [in] Function pointer. Cannot be null.</param>
    ///
    /// <returns>	A new compiled kernel object if it succeeds, null if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    CompiledKernel * 
    GetHostFunctionCompiledKernel(
        char * lpszOperation,
        FARPROC lpfn);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a new port. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. 
    /// 			Valid port types include:
    /// 			    INPUT_PORT:  a port that encapsulates generic input to a ptask
    ///                 OUTPUT_PORT: encapsulates generic output.
    ///                 STICKY_PORT: a port that can reuse the last input on every
    ///                              invocation of a ptask until a new one arrives. 
    ///                              Generally used to implement constant parameters.
    ///                 META_PORT:   a port that is actually an input to the runtime, 
    ///                              not an input to the ptask binary. A meta port
    ///                              reads its input and uses it to allocate output
    ///                              buffers for an associated output port
    ///                 INITIALIZER_PORT:
    ///                              a port that can provide an initial value without  
    ///                              requiring the programmer to explicitly push 
    ///                              a datablock into an associated channel.    
    ///                              
    ///             When associating ports with eachother for in/out pairs or
    ///             metaports, please bear in mind that the parameter indeces
    ///             are 0-based and have a unique sequence for each type of port.
    ///             For example, output "1" means the second output port for a ptask,
    ///             and does not necessarily mean parameter "1" in the function signature
    ///             of the PTask. This convention is necessary because not all runtimes
    ///             provide a function signature-based abstraction for binding data
    ///             to kernel arguments. 
    /// 			</remarks>
    ///
    /// <param name="type">                     The type of port to create. </param>
    /// <param name="pTemplate">                [in] a datablock template for the port
    /// 										constraining the geometry of datablocks
    /// 										that can flow through the port. Can be null.
    /// 										</param>
    /// <param name="uiId">                     An integer identifier for the port. 
    /// 										Needn't be unique--this is a debugging tool
    /// 										for the programmer.
    /// 										</param>
    /// <param name="lpszVariableBinding">      (optional) [in] If non-null, the name of
    /// 										a variable in PTask code to which this port  
    /// 										will be bound. 
    /// 										</param>
    /// <param name="nKernelParameterIndex">    (optional) zero-based index of the parameter. 
    /// 										in kernel code to which this port will be bound.
    /// 									    See remarks above for caveats/subtleties.
    /// 										</param>
    /// <param name="nInOutParmOutputIndex">    (optional) zero-based index of an output port
    /// 										which is paired with this port to create an
    /// 										in/out variable for a ptask.
    ///                                         </param>
    ///
    /// <returns>   null if it fails, else the port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port * CreatePort(
        PORTTYPE type, 
        DatablockTemplate * pTemplate, 
        UINT uiId, 
        char * lpszVariableBinding=NULL, 
        UINT nKernelParameterIndex=-1,      // other than -1 means this port is bound to a formal parameter with the given 0-based index
        UINT nInOutParmOutputIndex=-1);     // other than -1 means this port describes a ref parameter, meaning a modified input block should 
                                            // be pushed to the output at the given index after kernel dispatch

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the default block pool size. PTask pools blocks on output ports to speed
    ///             performance and to avoid unnecessary allocations on internal channels. When a
    ///             datablock is produced at an output port and consumed on internal channel
    ///             connecting to another input port, without block pooling, that datablock is
    ///             allocated and released on every invocation of the producing task. When a block is
    ///             pooled, it's final Release (refcount==0) returns it to the block pool that
    ///             produced it, rather than deleting it. This way internal channels reuse the same
    ///             blocks and never require allocation/release for data moving along that channel.
    ///             This setting is configurable because there are tradeoffs arising from memory
    ///             pressure arising from block pooling.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The block pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetICBlockPoolSize();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the default block pool size. PTask pools blocks on output ports to speed
    ///             performance and to avoid unnecessary allocations on internal channels. When a
    ///             datablock is produced at an output port and consumed on internal channel
    ///             connecting to another input port, without block pooling, that datablock is
    ///             allocated and released on every invocation of the producing task. When a block is
    ///             pooled, it's final Release (refcount==0) returns it to the block pool that
    ///             produced it, rather than deleting it. This way internal channels reuse the same
    ///             blocks and never require allocation/release for data moving along that channel.
    ///             This setting is configurable because there are tradeoffs arising from memory
    ///             pressure arising from block pooling.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="n">    The n. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetICBlockPoolSize(int n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Gets the gc batch size. PTask frees datablocks on a different thread using a garbage
    ///     collector. When a Datablock's refcount goes to zero, it is added to the GC's list, and a
    ///     GC thread will eventually free it. To minimize CPU consumption by the GC, it will only
    ///     run when the pending free list hits a given batch size, which is configured/queried using
    ///     this API.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The gc batch size. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetGCBatchSize();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets the gc batch size. PTask frees datablocks on a different thread using a garbage
    ///     collector. When a Datablock's refcount goes to zero, it is added to the GC's list, and a
    ///     GC thread will eventually free it. To minimize CPU consumption by the GC, it will only
    ///     run when the pending free list hits a given batch size, which is configured/queried using
    ///     this API.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="n">    The desired batch size. </param>
    ///
    ///-------------------------------------------------------------------------------------------------

    void SetGCBatchSize(int n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Gets the default channel capacity. Channels have configurable capacity, but when the
    ///     programmer does not explicitly specify, channels take the default capacity for the
    ///     runtime. This API manages that default size. Large defaults are good for allowing
    ///     producers to run way ahead of consumers, but can introduce memory pressure.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The default channel capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetDefaultChannelCapacity();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets the default channel capacity. Channels have configurable capacity, but when the
    ///     programmer does not explicitly specify, channels take the default capacity for the
    ///     runtime. This API manages that default size. Large defaults are good for allowing
    ///     producers to run way ahead of consumers, but can introduce memory pressure.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="n">    The desired default channel capacity. </param>
    ///
    ///-------------------------------------------------------------------------------------------------

    void SetDefaultChannelCapacity(int n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets default view materialization policy for channels. </summary>
    ///
    /// <remarks>   Crossbac, 7/9/2013. </remarks>
    ///
    /// <returns>   The default view materialization policy. </returns>
    ///-------------------------------------------------------------------------------------------------

    VIEWMATERIALIZATIONPOLICY GetDefaultViewMaterializationPolicy();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the default view materialization policy for exposed output channels. </summary>
    ///
    /// <remarks>   Crossbac, 7/9/2013. </remarks>
    ///
    /// <returns>   The default output view materialization policy. </returns>
    ///-------------------------------------------------------------------------------------------------

    VIEWMATERIALIZATIONPOLICY GetDefaultOutputViewMaterializationPolicy();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets default view materialization policy for channels. </summary>
    ///
    /// <remarks>   Crossbac, 7/9/2013. </remarks>
    ///
    /// <param name="ePolicy">  The policy. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    SetDefaultViewMaterializationPolicy(
        __in VIEWMATERIALIZATIONPOLICY ePolicy 
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets default view materialization policy for output channels. </summary>
    ///
    /// <remarks>   Crossbac, 7/9/2013. </remarks>
    ///
    /// <param name="ePolicy">  The policy. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    SetDefaultOutputViewMaterializationPolicy(
        __in VIEWMATERIALIZATIONPOLICY ePolicy
        );


    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets the default channel capacity. Channels have configurable capacity, but when the
    ///     programmer does not explicitly specify, channels take the default capacity for the
    ///     runtime. This API manages that default size. Large defaults are good for allowing
    ///     producers to run way ahead of consumers, but can introduce memory pressure.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="n">    The desired default channel capacity. </param>
    ///
    ///-------------------------------------------------------------------------------------------------

    void SetDefaultChannelCapacity(int n);



    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the default size of block pools for initializer channels.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The default channel capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT GetDefaultInitChannelBlockPoolSize();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the default size for block pools on initializer channels. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="n">    The desired pool size. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDefaultInitChannelBlockPoolSize(UINT n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the default grow increment for block pools.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The default channel capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT GetDefaultBlockPoolGrowIncrement();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the default grow increment for block pools. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="n">    The desired pool size. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDefaultBlockPoolGrowIncrement(UINT n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the default size of block pools for input channels.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The default channel capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT GetDefaultInputChannelBlockPoolSize();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the default size for block pools on input channels. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="n">    The desired pool size. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDefaultInputChannelBlockPoolSize(UINT n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Gets the scheduling mode. Will return one of the following values:
    ///       SCHEDMODE_COOPERATIVE = 0:
    ///         free-for all. When PTasks are ready for dispatch they contend on the accelerator
    ///         lock. First to acquire the lock is the first to run.
    ///       SCHEDMODE_PRIORITY = 1:
    ///         When ptasks are ready for dispatch they are queued in descending order of ptask
    ///         priority.
    ///       SCHEDMODE_DATADRIVEN = 2:
    ///         tries to schedule ready ptasks on accelerators where their data is already
    ///         materialized, defaulting to priority policy when this does not unambiguously
    ///         determine the scheduling decision.
    ///       SCHEDMODE_FIFO = 3:
    ///         ready ptasks are queued in the order they arrive.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The scheduling mode. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetSchedulingMode();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets the scheduling mode. Should pass only values in the range [0..3], and query the mode
    ///     to check whether the set succeeded.
    ///       SCHEDMODE_COOPERATIVE = 0:
    ///         free-for all. When PTasks are ready for dispatch they contend on the accelerator
    ///         lock. First to acquire the lock is the first to run.
    ///       SCHEDMODE_PRIORITY = 1:
    ///         When ptasks are ready for dispatch they are queued in descending order of ptask
    ///         priority.
    ///       SCHEDMODE_DATADRIVEN = 2:
    ///         tries to schedule ready ptasks on accelerators where their data is already
    ///         materialized, defaulting to priority policy when this does not unambiguously
    ///         determine the scheduling decision.
    ///       SCHEDMODE_FIFO = 3:
    ///         ready ptasks are queued in the order they arrive.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="mode"> The mode. </param>
    ///
    ///-------------------------------------------------------------------------------------------------

    void SetSchedulingMode(int mode);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the scheduler's thread count. </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///
    /// <returns>   The scheduler thread count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT GetSchedulerThreadCount();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the scheduler thread count. </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///
    /// <param name="uiThreads">    The threads. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetSchedulerThreadCount(UINT uiThreads);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the eager meta port mode. </summary>
    ///
    /// <remarks>   Crossbac, 9/28/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetEagerMetaPortMode();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an eager meta port mode. </summary>
    ///
    /// <remarks>   Crossbac, 9/28/2012. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetEagerMetaPortMode(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Return true if the PTask runtime is in debug mode In debug mode, data blocks traveling
    ///     along internal channels have host-side views materialized at all times, enabling
    ///     debugging when data is moving through parts of the graph that would normally not be
    ///     visible from the host.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if in debug mode, false otherwise. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetDebugMode();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Control the PTask runtime debug mode. In debug mode, data blocks traveling along internal
    ///     channels have host-side views materialized at all times, enabling debugging when data is
    ///     moving through parts of the graph that would normally not be visible from the host.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="b">    true if the runtime should be in debug mode, false otherwise. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDebugMode(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Get the PTask runtime mode for introducing synchrony after sensitive
    ///             operations like dispatch and dependence waiting. When the runtime is in
    ///             "ForceSynchonous" mode, backend API calls are inserted to synchronize device
    ///             contexts with the host. This is intended as a debug tool to rule out race-
    ///             conditions in PTask when faced with undesirable or difficult-to-understand
    ///             results from programs written for PTask.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/12. </remarks>
    ///
    /// <param name="b">    true if the runtime should be in force-sync mode, false otherwise. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetForceSynchronous();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Control the PTask runtime mode for introducing synchrony after sensitive
    ///             operations like dispatch and dependence waiting. When the runtime is in
    ///             "ForceSynchonous" mode, backend API calls are inserted to synchronize device
    ///             contexts with the host. This is intended as a debug tool to rule out race-
    ///             conditions in PTask when faced with undesirable or difficult-to-understand
    ///             results from programs written for PTask.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/12. </remarks>
    ///
    /// <param name="b">    true if the runtime should be in force-sync mode, false otherwise. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetForceSynchronous(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the extreme trace mode. Extreme trace mode logs every API call to the trace
    ///             provider. For the mode to work PTASK must be built with EXTREME_TRACE
    ///             preprocessor macro! Otherwise, the trace mode is always off.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetExtremeTraceMode();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the extreme trace mode. Extreme trace mode logs every API call to the trace
    ///             provider. For the mode to work PTASK must be built with EXTREME_TRACE
    ///             preprocessor macro! Otherwise, the trace mode is always off.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    void SetExtremeTraceMode(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the coherence profile mode. The mode logs every coherence transition
    /// 			and task/port binding for every datablock. For the mode to work PTASK must be built 
    /// 			with PROFILE_MIGRATION. Otherwise the mode is always off.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetCoherenceProfileMode();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the coherence profile mode. The mode logs every coherence transition
    /// 			and task/port binding for every datablock. For the mode to work PTASK must be built 
    /// 			with PROFILE_MIGRATION. Otherwise the mode is always off.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    void SetCoherenceProfileMode(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the task profile mode. The mode captures per-dispatch timings for various
    ///             task-critical activities such as buffer allocation, binding, transfer, dispatch,
    ///             etc. For the mode to work, PTask must be built with PROFILE_TASKS. This API is
    ///             available regardless but does not affect the behavior of the runtime if the build
    ///             doesn't support the profiling option.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///
    /// <param name="bEnable">  true to enable, false to disable. </param>
    /// <param name="bConcise"> (optional) emit a more concise version (no per-task histories). </param>
    ///-------------------------------------------------------------------------------------------------
    
    void SetTaskProfileMode(BOOL bEnable, BOOL bConcise=TRUE);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the task profile mode. The mode captures per-dispatch timings for various
    ///             task-critical activities such as buffer allocation, binding, transfer, dispatch,
    ///             etc. For the mode to work, PTask must be built with PROFILE_TASKS. This API is
    ///             available regardless but always returns false if the build doesn't support the
    ///             profiling option.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetTaskProfileMode();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the debug asynchrony mode. When this is on, PTask emits copious text 
    /// 			when synchronous transfers occur. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    void SetDebugAsynchronyMode(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the debug asynchrony mode. When this is on, PTask emits copious text 
    /// 			when synchronous transfers occur. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    BOOL GetDebugAsynchronyMode();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Controls whether PTask profiles management of platform-specific buffer objects
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    void SetProfilePlatformBuffers(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Controls whether PTask profiles management of platform-specific buffer objects
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    BOOL GetProfilePlatformBuffers();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Controls whether PTask uses page-locked buffers to back host views of datablocks
    ///             or not. If this is turned off, even programmer requests for page-locked views 
    ///             are ignorned. If it is turned on, the behavior is determined by the aggressive
    ///             page locking setting. Non-aggressive leaves things in the hands of the programmer,
    ///             while aggressive attempts to page lock backing buffers for all potentially 
    ///             communicating blocks.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    void SetPageLockingEnabled(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Controls whether PTask uses page-locked buffers to back host views of datablocks
    ///             or not. If this is turned off, even programmer requests for page-locked views 
    ///             are ignorned. If it is turned on, the behavior is determined by the aggressive
    ///             page locking setting. Non-aggressive leaves things in the hands of the programmer,
    ///             while aggressive attempts to page lock backing buffers for all potentially 
    ///             communicating blocks.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    BOOL GetPageLockingEnabled();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Controls whether PTask attempts to always use page-locked host buffers
    /// 			for devices that support page-locked buffer allocation. If it
    /// 			is set to true, PTask will allocate page-locked buffers whenever it detects 
    /// 			that it may be possible to use the resulting buffer in an async API call. 
    ///             When it is off, PTask will only allocate a page-locked host-buffer when 
    ///             the programmer requests it explicitly for a given block. Generally speaking,
    ///             setting this to true is profitable for workloads with moderate memory traffic, 
    ///             but since performance drops off quickly when too much page-locked memory is allocated,
    ///             this is not a setting that should be used without careful consideration. 
    ///             </summary>
    ///             
    /// <remarks>   crossbac, 4/28/2013. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetAggressivePageLocking(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Controls whether PTask attempts to always use page-locked host buffers
    /// 			for devices that support page-locked buffer allocation. If it
    /// 			is set to true, PTask will allocate page-locked buffers whenever it detects 
    /// 			that it may be possible to use the resulting buffer in an async API call. 
    ///             When it is off, PTask will only allocate a page-locked host-buffer when 
    ///             the programmer requests it explicitly for a given block. Generally speaking,
    ///             setting this to true is profitable for workloads with moderate memory traffic, 
    ///             but since performance drops off quickly when too much page-locked memory is allocated,
    ///             this is not a setting that should be used without careful consideration. 
    ///             </summary>
    ///             
    /// <remarks>   crossbac, 4/28/2013. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------
    
    BOOL GetAggressivePageLocking();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dispatch logging enabled property. </summary>
    ///
    /// <remarks>   crossbac, 6/12/2012. </remarks>
    ///
    /// <returns>   true if dispatch logging is enabled. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetDispatchLoggingEnabled();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the dispatch logging enabled property. </summary>
    ///
    /// <remarks>   crossbac, 6/12/2012. </remarks>
    ///
    /// <param name="bEnabled"> true to enable, false to disable. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDispatchLoggingEnabled(BOOL bEnabled);        

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dispatch tracing enabled property. </summary>
    ///
    /// <remarks>   jcurrey, 6/15/2012. </remarks>
    ///
    /// <returns>   true if dispatch logging is enabled. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetDispatchTracingEnabled();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the dispatch tracing enabled property. </summary>
    ///
    /// <remarks>   jcurrey, 6/15/2012. </remarks>
    ///
    /// <param name="bEnabled"> true to enable, false to disable. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDispatchTracingEnabled(BOOL bEnabled);        

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the maximum concurrency for the runtime. The maximum concurrency is the
    ///             maximum number of hardware accelerators the runtime will use. For example, On a
    ///             machine with 2 GPU cards, setting this to one will ensure the runtime uses only
    ///             one of them. The runtime will attempt to select the most powerful card when
    ///             pruning. A setting of zero means use all available accelerators.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="nGPUs">    The maximum number of GPUs. Zero means use all available. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetMaximumConcurrency(int nGPUs);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the maximum concurrency. The maximum concurrency is the maximum number of
    ///             hardware accelerators the runtime will use. For example, On a machine with 2 GPU
    ///             cards, setting this to one will ensure the runtime uses only one of them. The
    ///             runtime will attempt to select the most powerful card when pruning. A setting of
    ///             zero means use all available accelerators.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The maximum concurrency, or zero if it is unconstrained. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetMaximumConcurrency();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the maximum number of host accelerator objects.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="nGPUs">    The maximum number of GPUs. Zero means use all available. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetMaximumHostConcurrency(int nGPUs);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the maximum number of host accelerator objects
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The maximum concurrency, or zero if it is unconstrained. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetMaximumHostConcurrency();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables/Disables fine-grain memory allocation tracking per memory space.
    ///             </summary>
    ///-------------------------------------------------------------------------------------------------

    void SetTrackDeviceMemory(BOOL bTrack);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the enables/disables state of fine-grain memory allocation tracking per
    ///             memory space.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetTrackDeviceMemory();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables/Disables fine-grain memory allocation tracking per memory space.
    ///             </summary>
    ///-------------------------------------------------------------------------------------------------

    void SetTrackDeviceMemory(BOOL bTrack);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the size of the task thread pool. A return value of zero means the
    ///             scheduler is using a thread per-task, which is the traditional PTask
    ///             approach. However, this approach is non-performant for very graphs because
    ///             windows doesn't (can't?) handle large thread counts well. Consequently,
    ///             we provide a switch for the programmer to manage the thread pool size explicitly
    ///             as well as switches to cause the runtime to choose the size automatically
    ///             based on a tunable parameter. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT GetTaskThreadPoolSize();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Set the size of the task thread pool. A return value of zero means the scheduler
    ///             is using a thread per-task, which is the traditional PTask approach. However,
    ///             this approach is non-performant for very graphs because windows doesn't (can't?)
    ///             handle large thread counts well. Consequently, we provide a switch for the
    ///             programmer to manage the thread pool size explicitly as well as switches to cause
    ///             the runtime to choose the size automatically based on a tunable parameter.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="uiThreadPoolSize"> Size of the thread pool. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetTaskThreadPoolSize(UINT uiThreadPoolSize);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the size of the task thread pool. A return value of zero means the
    ///             scheduler is using a thread per-task, which is the traditional PTask
    ///             approach. However, this approach is non-performant for very graphs because
    ///             windows doesn't (can't?) handle large thread counts well. Consequently,
    ///             we provide a switch for the programmer to manage the thread pool size explicitly
    ///             as well as switches to cause the runtime to choose the size automatically
    ///             based on a tunable parameter. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    THREADPOOLPOLICY GetTaskThreadPoolPolicy();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Set the size of the task thread pool. A return value of zero means the scheduler
    ///             is using a thread per-task, which is the traditional PTask approach. However,
    ///             this approach is non-performant for very graphs because windows doesn't (can't?)
    ///             handle large thread counts well. Consequently, we provide a switch for the
    ///             programmer to manage the thread pool size explicitly as well as switches to cause
    ///             the runtime to choose the size automatically based on a tunable parameter.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="uiThreadPoolSize"> Size of the thread pool. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetTaskThreadPoolPolicy(THREADPOOLPOLICY ePolicy);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the threshold at which the runtime will change the thread:task
    ///             cardinality from 1:1 to 1:N. For small graphs the former is more performant,
    ///             while for large graphs, the latter is. The knee of the curve is likely 
    ///             platform-dependent, so we need an API to control this. 
    ///             
    ///             TODO: FIXME: Make PTask choose a good initial value based on CPU count
    ///             rather than taking a hard-coded default.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT GetSchedulerThreadPerTaskThreshold();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Set the threshold at which the runtime will change the thread:task
    ///             cardinality from 1:1 to 1:N. For small graphs the former is more performant,
    ///             while for large graphs, the latter is. The knee of the curve is likely 
    ///             platform-dependent, so we need an API to control this. 
    ///             
    ///             TODO: FIXME: Make PTask choose a good initial value based on CPU count
    ///             rather than taking a hard-coded default.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    void SetSchedulerThreadPerTaskThreshold(UINT uiMaxTasks);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets use graph monitor watchdog. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <param name="bUseWatchdog"> true to use watchdog. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetUseGraphMonitorWatchdog(BOOL bUseWatchdog);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the (ostensible) cuda heap size. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <returns>   size in bytes. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT GetCUDAHeapSize();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the cuda heap size. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <param name="uiSizeBytes">  heap size </param>
    ///-------------------------------------------------------------------------------------------------

    void SetCUDAHeapSize(UINT uiSizeBytes);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Cublas has pathological start up cost, so lazy initialization 
    ///             winds up accruing to ptask execution time. This can be avoided
    ///             by forcing PTask to initialize cublas early, at the cost of some
    ///             unwanted dependences on it.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void SetInitializeCublas(BOOL bInitialize);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   When we use thread pooling, a lot of the scheduler's policy becomes
    ///             ineffective because tasks are queued up waiting not for GPUs but for
    ///             task threads. Technically, we need the same logic we have dealing with
    ///             the scheduler run queue to be present in the graph runner procs.
    ///             For now, just make it possible to sort by priority so we don't wind up
    ///             defaulting to pure FIFO behavior when thread pooling is in use.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void SetThreadPoolPriorityQueues(BOOL bSortQueue);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   When we use thread pooling, a lot of the scheduler's policy becomes
    ///             ineffective because tasks are queued up waiting not for GPUs but for
    ///             task threads. Technically, we need the same logic we have dealing with
    ///             the scheduler run queue to be present in the graph runner procs.
    ///             For now, just make it possible to sort by priority so we don't wind up
    ///             defaulting to pure FIFO behavior when thread pooling is in use.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetThreadPoolPriorityQueues();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Set the default partitioning mode assigned to subsequently created Graph instances.
    ///     Should pass only one of the following values:
    ///       GRAPHPARTITIONINGMODE_NONE = 0:
    ///         The runtime will not partition graphs across multiple available accelerators.
    ///       GRAPHPARTITIONINGMODE_HINTED = 1:
    ///         The runtime will partition graphs across multiple available accelerators,
    ///         according to hints given explicitly by the application via PTask::SetSchedulerPartitionHint().
    ///       GRAPHPARTITIONINGMODE_HEURISTIC = 2:
    ///         The runtime will partition graphs across multiple available accelerators,
    ///         available accelerators, using a set of experimental heuristics.
    ///       AUTOPARTITIONMODE_OPTIMAL = 2:
    ///         The runtime will attempt to auto-partition graphs across multiple
    ///         available accelerators, using a graph cut algorithm that finds the min-cut.
    ///
    ///     The default at runtime initialization is GRAPHPARTITIONINGMODE_NONE.
    /// </summary>
    ///
    /// <remarks>   jcurrey, 1/27/2014. </remarks>
    ///
    /// <param name="mode"> The default graph partitioning mode. </param>
    ///
    ///-------------------------------------------------------------------------------------------------

    void SetDefaultGraphPartitioningMode(int mode);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Get the default partitioning mode assigned to subsequently created Graph instances. 
    ///     Will return one of the following values:
    ///       GRAPHPARTITIONINGMODE_NONE = 0:
    ///         The runtime will not partition graphs across multiple available accelerators.
    ///       GRAPHPARTITIONINGMODE_HINTED = 1:
    ///         The runtime will partition graphs across multiple available accelerators,
    ///         according to hints given explicitly by the application via PTask::SetSchedulerPartitionHint().
    ///       GRAPHPARTITIONINGMODE_HEURISTIC = 2:
    ///         The runtime will partition graphs across multiple available accelerators,
    ///         available accelerators, using a set of experimental heuristics.
    ///       AUTOPARTITIONMODE_OPTIMAL = 2:
    ///         The runtime will attempt to auto-partition graphs across multiple
    ///         available accelerators, using a graph cut algorithm that finds the min-cut.
    ///
    ///     The default at runtime initialization is GRAPHPARTITIONINGMODE_NONE.
    /// </summary>
    ///
    /// <remarks>   jcurrey, 1/27/2014. </remarks>
    ///
    /// <returns>   The graph partitioning mode. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetDefaultGraphPartitioningMode();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets use graph monitor watchdog. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetUseGraphMonitorWatchdog();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets dispatch watchdog threshold. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <returns>   The dispatch watchdog threshold. </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD GetDispatchWatchdogThreshold();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets dispatch watchdog threshold. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <param name="dwThreshold">  The threshold. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDispatchWatchdogThreshold(DWORD dwThreshold);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets provision block pools for capacity. When this mode is set
    ///             the runtime will try to allocate block pools such that they
    ///             can satisfy all downstream request without allocation. This
    ///             requires being able to fill the capacity of all downstream channels
    ///             along inout consumer paths. This is a handy tool, but should not
    ///             be used unless you are actually attempting tune channel capacities
    ///             because the default channel capacity is very large. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/21/2013. </remarks>
    ///
    /// <param name="bProvision">   true to provision. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetProvisionBlockPoolsForCapacity(BOOL bProvision);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets provision block pools for capacity. When this mode is set
    ///             the runtime will try to allocate block pools such that they
    ///             can satisfy all downstream request without allocation. This
    ///             requires being able to fill the capacity of all downstream channels
    ///             along inout consumer paths. This is a handy tool, but should not
    ///             be used unless you are actually attempting tune channel capacities
    ///             because the default channel capacity is very large. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/21/2013. </remarks>
    ///
    /// <param name="bProvision">   true to provision. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetProvisionBlockPoolsForCapacity();

    BOOL GetTaskDispatchLocksIncomingAsyncSources();
    BOOL GetThreadPoolSignalPerThread();
    BOOL GetTaskDispatchReadyCheckIncomingAsyncDeps();
    BOOL GetTaskDispatchLocklessIncomingDepWait();
    void SetTaskDispatchLocksIncomingAsyncSources(BOOL b);
    void SetThreadPoolSignalPerThread(BOOL b);
    void SetTaskDispatchReadyCheckIncomingAsyncDeps(BOOL b);
    void SetTaskDispatchLocklessIncomingDepWait(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the runtime is in a multi gpu environment. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if more than one GPU is present and in use. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL MultiGPUEnvironment();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets a threshold for the runtime to ignore locality as a primary scheduling constraint
    ///     when running in DATA_AWARE scheduling mode.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="n">    The threshold. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetIgnoreLocalityThreshold(int n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Gets the current threshold for the runtime to ignore locality as a primary scheduling
    ///     constraint when running in DATA_AWARE scheduling mode.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The ignore locality threshold. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetIgnoreLocalityThreshold();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the runtime is configured to use Host PTasks. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if host ptasks are enabled. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetUseHost();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Return true if the runtime is configured to use CUDA ptasks. Restricting the number of
    ///     runtimes in use will reduce the size of scheduler data structures and can improve
    ///     performance for workloads that use a single backend homogenously by reducing the latency
    ///     of scheduling decisions.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if CUDA ptasks are enabled. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetUseCUDA(); 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Return true if the runtime is configured to use OpenCL ptasks. Restricting the number of
    ///     runtimes in use will reduce the size of scheduler data structures and can improve
    ///     performance for workloads that use a single backend homogenously by reducing the latency
    ///     of scheduling decisions.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if OpenCL ptasks are enabled. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetUseOpenCL();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Return true if the runtime is configured to use DirectX ptasks. Restricting the number of
    ///     runtimes in use will reduce the size of scheduler data structures and can improve
    ///     performance for workloads that use a single backend homogenously by reducing the latency
    ///     of scheduling decisions.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if DirectX ptasks are enabled. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetUseDirectX();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Control the PTask runtimes use of host PTasks. Disabling task types for workloads that do
    ///     not use them can improve performance by reducing the latency of scheduling decisions.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="b">    true to enable host tasks, false to disable. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetUseHost(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Control the PTask runtimes use of CUDA PTasks. Disabling task types for workloads that do
    ///     not use them can improve performance by reducing the latency of scheduling decisions.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="b">    true to enable CUDA tasks, false to disable. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetUseCUDA(BOOL b); 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Control the PTask runtimes use of OpenCL PTasks. Disabling task types for workloads that do
    ///     not use them can improve performance by reducing the latency of scheduling decisions.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="b">    true to enable OpenCL tasks, false to disable. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetUseOpenCL(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Control the PTask runtimes use of DirectX PTasks. Disabling task types for workloads that do
    ///     not use them can improve performance by reducing the latency of scheduling decisions.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="b">    true to enable DirectX tasks, false to disable. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetUseDirectX(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Gets the minimum direct x feature level. By default, PTask will fail attempts to create
    ///     accelerator objects for hardware that does not support at least DX11. However, this can
    ///     be overly restrictive since DX10 hardware is sufficient to run a large body of code.
    ///     Disabling it by default is important because it makes certain runtime failures more
    ///     obvious. However, we defer control of the setting to the programmer.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The minimum direct x feature level, as an int. E.g. 
    /// 			are return value of "11" means DirectX 11. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetMinimumDirectXFeatureLevel();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets a minimum direct x feature level. By default, PTask will fail attempts to create
    ///     accelerator objects for hardware that does not support at least DX11. However, this can
    ///     be overly restrictive since DX10 hardware is sufficient to run a large body of code.
    ///     Disabling it by default is important because it makes certain runtime failures more
    ///     obvious. However, we defer control of the setting to the programmer.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="nLevel">   The level as int (11 -> DirectX 11 and so on.) </param>
    ///-------------------------------------------------------------------------------------------------

    void SetMinimumDirectXFeatureLevel(int nLevel);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Return true if reference driver use is enabled. Allowing the runtime to use DirectX
    ///     reference drivers can enable debugging on platforms that do not have the requisite
    ///     hardware (laptops!). It is disabled by default because the reference drivers emulate
    ///     hardware and are consequently very slow--should never be used in production.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetUseReferenceDrivers();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Control the use of reference drivers. Allowing the runtime to use DirectX reference
    ///     drivers can enable debugging on platforms that do not have the requisite hardware
    ///     (laptops!). It is disabled by default because the reference drivers emulate hardware and
    ///     are consequently very slow--should never be used in production.
    ///     Currently we only support DirectX reference drivers. 
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="b">    true to enable use of reference drivers. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetUseReferenceDrivers(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Turn tracing on for a given subsystem. </summary>
    ///
    /// <remarks>   crossbac, 5/21/2012. </remarks>
    ///
    /// <param name="lpszSubsystemName">    [in,out] If non-null, name of the subsystem. </param>
    /// <param name="bTrace">               (optional) the trace. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    TraceSubsystem(
        __in char * lpszSubsystemName, 
        __in BOOL bTrace=TRUE
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Logs. </summary>
    ///
    /// <remarks>	Rossbach, 2/15/2012. </remarks>
    ///
    /// <param name="fmt">	[in,out] If non-null, describes the format to use. </param>
    ///
    /// <returns>	. </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID 
    Trace(
        char * szSubsystemName,
        char* fmt, 
        ...
        );


    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Query if the runtime is in verbose mode. In verbose mode the runtime will emit extra
    ///     debug output, controlled by a logging level parameter.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if verbose, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL IsVerbose();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Control runtime verbose mode. In verbose mode the runtime will emit extra debug output,
    ///     controlled by a logging level parameter.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="b">    true to enable verbosity. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetVerbose(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Gets the logging level. If the runtime is in verbose mode it will emit messages provided
    ///     if the log level of call is >= the current logging level. This feature can help tailor
    ///     the amount of debug output.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The logging level. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT GetLoggingLevel();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets the logging level. If the runtime is in verbose mode it will emit messages provided
    ///     if the log level of call is >= the current logging level. This feature can help tailor
    ///     the amount of debug output.
    /// </summary>    
    /// 
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="nLevel">   The level. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetLoggingLevel(UINT nLevel);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Let the programmer control what PTask does when it encounters an internal failure
    ///             that it is unsure of how to handle. In debug mode, the default behavior is to
    ///             exit, to ensure the programmer cannot overlook the problem.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/9/2012. </remarks>
    ///
    /// <param name="bExit">    true to exit. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetExitOnRuntimeFailure(BOOL bExit);
    BOOL GetExitOnRuntimeFailure();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Display an error message. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="szMessage">    The message. </param>
    /// <param name="nLogLevel">    (optional) the log level. </param>
    ///-------------------------------------------------------------------------------------------------

    void ErrorMessage(const char * szMessage, UINT nLogLevel=0xFFFFFFFF);
    void ErrorMessage(const WCHAR * szMessage, UINT nLogLevel=0xFFFFFFFF);
    void ErrorMessage(const std::string &szMessage, UINT nLogLevel=0xFFFFFFFF); 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Provide information on the console that is not an error or warning, but
    /// 			which must be emitted regardless of logging/tracing state.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="szMessage">    The message. </param>
    /// <param name="nLogLevel">    (optional) the log level. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL MandatoryInform(const char * szMessage, ...);
    BOOL MandatoryInform(const WCHAR * szMessage, ...);
    BOOL MandatoryInform(const std::string &szMessage);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Display a warning. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="szMessage">    The message. </param>
    /// <param name="nLogLevel">    (optional) the log level. </param>
    ///-------------------------------------------------------------------------------------------------

    void Warning(const char * szMessage, UINT nLogLevel=0xFFFFFFFF);
    void Warning(const WCHAR * szMessage, UINT nLogLevel=0xFFFFFFFF);
    void Warning(const std::string &szMessage, UINT nLogLevel=0xFFFFFFFF);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Provide information on the console, if verbose mode is on and the log level is >=
    ///             to current runtime log-level.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="szMessage">    The message. </param>
    /// <param name="nLogLevel">    (optional) the log level. </param>
    ///-------------------------------------------------------------------------------------------------

    void Inform(const WCHAR * szMessage);
    void Inform(const std::string &szMessage);
    void Inform(const char * szMessage, ...);
    void Inform(const WCHAR * szMessage, ...);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Trace. Add the given string to a trace output. For runtime debugging. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="szMessage">    The message. </param>
    /// <param name="nLogLevel">    (optional) the log level. </param>
    ///-------------------------------------------------------------------------------------------------

    void Trace(const char * szMessage, UINT nLogLevel=0xFFFFFFFF);
    void Trace(const WCHAR * szMessage, UINT nLogLevel=0xFFFFFFFF);
    void Trace(const std::string &szMessage, UINT nLogLevel=0xFFFFFFFF);

    /// <summary> The default dump type </summary>
    extern DEBUGDUMPTYPE g_nDefaultDumpType;
    
    /// <summary> The default dump stride </summary>
    extern int g_nDefaultDumpStride;
    
    /// <summary> The default dump length </summary>
    extern int g_nDefaultDumpLength;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the default dump type. </summary>
    ///
    /// <remarks>   Crossbac, 1/9/2012. </remarks>
    ///
    /// <param name="typ">  The typ. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDumpType(DEBUGDUMPTYPE typ);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a dump stride. </summary>
    ///
    /// <remarks>   Crossbac, 1/9/2012. </remarks>
    ///
    /// <param name="n">    The n. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDumpStride(int n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a dump length. </summary>
    ///
    /// <remarks>   Crossbac, 1/9/2012. </remarks>
    ///
    /// <param name="n">    The n. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDumpLength(int n);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dump type. </summary>
    ///
    /// <remarks>   Crossbac, 1/9/2012. </remarks>
    ///
    /// <returns>   The dump type. </returns>
    ///-------------------------------------------------------------------------------------------------

    DEBUGDUMPTYPE GetDumpType();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dump stride. </summary>
    ///
    /// <remarks>   Crossbac, 1/9/2012. </remarks>
    ///
    /// <returns>   The dump stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetDumpStride();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dump length. </summary>
    ///
    /// <remarks>   Crossbac, 1/9/2012. </remarks>
    ///
    /// <returns>   The dump length. </returns>
    ///-------------------------------------------------------------------------------------------------

    int GetDumpLength();

#define SET_DUMP_TYPE(x)   PTask::Runtime::SetDumpType(x)
#define SET_DUMP_STRIDE(x) PTask::Runtime::SetDumpStride(x)
#define SET_DUMP_LENGTH(x) PTask::Runtime::SetDumpLength(x)

#if (defined(DANDELION_DEBUG) || defined(DUMP_INTERMEDIATE_BLOCKS))
    #ifdef TRACE_STRINGSTREAM
    extern std::stringstream g_ss;
#else
    extern std::ostream& g_ss;
#endif
    extern CRITICAL_SECTION g_sslock;
#endif

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a datablock template. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="lpszType">         [in] If non-null a string describing the type. </param>
    /// <param name="uiStride">         The stride of elements. </param>
    /// <param name="x">                The number of X elements. </param>
    /// <param name="y">                The number of Y elements. </param>
    /// <param name="z">                The number of Z elements. </param>
    /// <param name="bRecordStream">    (optional) true if this template describes a record stream. </param>
    /// <param name="bRaw">             (optional) true if this template describes raw (byte-
    ///                                 addressable) blocks. </param>
    ///
    /// <returns>   null if it fails, else a new datablock template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate * 
    GetDatablockTemplate(
        char * lpszType, 
        unsigned int uiStride, 
        unsigned int x, 
        unsigned int y, 
        unsigned int z,
        bool bRecordStream = false,
        bool bRaw = false
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a datablock template. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
    /// <param name="uiElementStride">      [in] The element stride in bytes. </param>
    /// <param name="uiElementsX">          [in] Number of elements in X dimension. </param>
    /// <param name="uiElementsY">          [in] Number of elements in Y dimension. </param>
    /// <param name="uiElementsZ">          [in] Number of elements in Z dimension. </param>
    /// <param name="uiPitch">              [in] The row pitch. </param>
    /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
    /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate * 
    GetDatablockTemplate(
        __in char *       lpszTemplateName, 
        __in unsigned int uiElementStride, 
        __in unsigned int uiElementsX, 
        __in unsigned int uiElementsY, 
        __in unsigned int uiElementsZ,
        __in unsigned int uiPitch,
        __in bool         bIsRecordStream,
        __in bool         bIsByteAddressable
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a datablock template. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
    /// <param name="pBufferDims">          [in] The element stride in bytes. </param>
    /// <param name="uiNumBufferDims">      [in] Number of elements in X dimension. </param>
    /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
    /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate * 
    GetDatablockTemplate(
        __in char *             lpszTemplateName, 
        __in BUFFERDIMENSIONS * pBufferDims, 
        __in unsigned int       uiNumBufferDims, 
        __in bool               bIsRecordStream,
        __in bool               bIsByteAddressable
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a datablock template for scalar parameter ports </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="lpszType"> [in] If non-null, the type. </param>
    /// <param name="uiStride"> The stride. </param>
    /// <param name="pttype">   The PTASK_PARM_TYPE. </param>
    ///
    /// <returns>   null if it fails, else the new datablock template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate * 
    GetDatablockTemplate(
        char * lpszType, 
        unsigned int uiStride, 
        PTASK_PARM_TYPE pttype
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets an existing datablock template by type name. </summary>
    ///
    /// <remarks>   jcurrey, 5/8/2013. </remarks>
    ///
    /// <param name="lpszType"> The name of the template type looking for. </param>
    ///
    /// <returns>   null if it fails, else the datablock template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate * 
    GetDatablockTemplate(
        char * lpszType
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate a new datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pTemplate">            [in] If non-null, a template describing geometry of the
    ///                                     allocated block. </param>
    /// <param name="pInitData">            [in,out] If non-null, information describing
    ///                                     the initialise. </param>
    /// <param name="cbInitData">           count of bytes in pInitData if it is provided. </param>
    /// <param name="pDestChannel">         [in] If non-null, the channel this block will
    ///                                     be pushed into. </param>
    /// <param name="flags">                Buffer access flags for the block. </param>
    /// <param name="uiBlockControlCode">   a block control code (such as DBCTL_EOF). </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    AllocateDatablock(
        DatablockTemplate * pTemplate,
        void * pInitData,
        UINT cbInitData,
        Channel * pDestChannel,
        BUFFERACCESSFLAGS flags=0,
        CONTROLSIGNAL uiBlockControlCode=0
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate a new datablock, with an effort to ensure that it is possible
    ///             to use the resulting block with async APIs for whatever backend is the target. 
    ///             In particular, when there are multiple accelerators that can be bound to the
    ///             target port, the runtime has a tough decision with respect to materializing
    ///             on the device: we don't know which device will be bound to execute the target
    ///             task at the time we are allocating the datablock. If the bMaterializeAll parameter
    ///             is true, we will actually create potentially many device-side buffers: one for
    ///             every possible execution context. If it is false, *and* there is a choice about
    ///             the device target, we defer the device copy, but ensure that we allocate a
    ///             host-side copy that enables async copy later (ie, make sure the host buffer
    ///             is pinned!). 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pTemplate">            [in] If non-null, a template describing geometry of the
    ///                                     allocated block. </param>
    /// <param name="pInitData">            [in,out] If non-null, information describing the
    ///                                     initialise. </param>
    /// <param name="cbInitData">           count of bytes in pInitData if it is provided. </param>
    /// <param name="pDestChannel">         [in] If non-null, the channel this block will be pushed
    ///                                     into. </param>
    /// <param name="flags">                Buffer access flags for the block. </param>
    /// <param name="uiBlockControlCode">   a block control code (such as DBCTL_EOF). </param>
    /// <param name="bMaterializeAll">      (optional) the materialize all. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    AllocateDatablockAsync(
        DatablockTemplate * pTemplate,
        void * pInitData,
        UINT cbInitData,
        Channel * pDestChannel,
        BUFFERACCESSFLAGS flags=0,
        CONTROLSIGNAL uiBlockControlCode=0,
        BOOL bMaterializeAll=FALSE
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Allocate a datablock with unknown final geometry. Use this allocator to create a new
    ///     block into which a variable length record stream can be written. The programmer must call
    ///     Datablock::Seal before using the block.
    ///     Note that datablocks have three channels:
    ///     1. The data channel contains raw data, assumed to be a sequence of records.
    ///     2. The meta data contains per record entry metadata that the runtime and kernel  
    ///        code can use to interpret the data channel.
    ///     3. A template channel, containing metadata that applies to all records  
    ///        in the data channel. 
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pTemplate">            [in] If non-null, a template. </param>
    /// <param name="uiDataSize">           Initial size of the data channel. </param>
    /// <param name="uiMetaSize">           Initial size of the meta data channel. </param>
    /// <param name="uiTemplateSize">       Initial size of the template channel. </param>
    /// <param name="uiBlockControlCode">   (optional) a block control code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    AllocateDatablock(
        DatablockTemplate * pTemplate,
        UINT uiDataSize,
        UINT uiMetaSize,
        UINT uiTemplateSize,
        CONTROLSIGNAL uiBlockControlCode=0
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate a control datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="uiBlockControlCode">   The block control code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    AllocateDatablock(
        CONTROLSIGNAL uiBlockControlCode
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free a datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pBlock">   [in] the block to free. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    FreeDatablock(
        Datablock * pBlock
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Handle runtime error. A runtime error is one that cannot be handled by the
    ///             programmer. Currently we respond by complaining vociferously and exiting.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="szMessage">    [in] If non-null, description of the error. </param>
    ///
    /// <returns>   true if the error is recoverable. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL HandleError(const char * szMessage, ...);
    BOOL HandleError(const std::string &strError);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Check a (supposed) invariant, and complain if the invariant does not hold. </summary>
    ///
    /// <remarks>	Crossbac, 10/2/2012. </remarks>
    ///
    /// <param name="bCondition">   	condition to check. </param>
    /// <param name="lpszErrorText">	If non-null, the error text to emit if the invariant fails. </param>
    ///
    /// <returns>	true if the invariant holds. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    CheckInvariantCondition(
		__in BOOL bCondition,
        __in char * lpszErrorText=NULL
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Check a (supposed) invariant, and complain if the invariant does not hold. </summary>
    ///
    /// <remarks>	Crossbac, 10/2/2012. </remarks>
    ///
    /// <param name="bCondition">  	true to condition. </param>
    /// <param name="strErrorText">	The error text. </param>
    ///
    /// <returns>	true if the invariant holds. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CheckInvariantCondition(
		BOOL bCondition,
        std::string strErrorText
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Gets the number of physical accelerators present on the machine. This number is likely
    ///     different from the number of accelerator objects, since a given accelerator will
    ///     typically support multiple runtimes.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The physical accelerator count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    GetPhysicalAcceleratorCount(
        VOID
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the physical accelerator count. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="nAcceleratorCount">    Number of accelerators. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    SetPhysicalAcceleratorCount(
        UINT nAcceleratorCount
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the meta function for the given port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pPort">                    [in,out] If non-null, the port. </param>
    /// <param name="eCanonicalMetaFunction">   The canonical meta function. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    SetPortMetaFunction(
        Port * pPort, 
        METAFUNCTION eCanonicalMetaFunction
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a task. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pGraph">           [in,out] If non-null, the graph. </param>
    /// <param name="pKernel">          [in,out] If non-null, the kernel. </param>
    /// <param name="uiInputPortCount"> Number of input ports. </param>
    /// <param name="pvInputPorts">     [in,out] If non-null, the pv input ports. </param>
    /// <param name="uiOutputPorts">    The output ports. </param>
    /// <param name="pvOutputPorts">    [in,out] If non-null, the pv output ports. </param>
    /// <param name="lpszTaskName">     [in,out] If non-null, name of the task. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task*
    AddTask(
        Graph *             pGraph,
        CompiledKernel *	pKernel,
        UINT				uiInputPortCount,
        Port**				pvInputPorts,
        UINT				uiOutputPorts,
        Port**				pvOutputPorts,
        char *				lpszTaskName
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind a derived port. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pGraph">           [in,out] If non-null, the graph. </param>
    /// <param name="pDescribedPort">   [in,out] If non-null, the described port. </param>
    /// <param name="pDescriberPort">   [in,out] If non-null, the describer port. </param>
    /// <param name="func">             (optional) the func. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    BindDerivedPort(
        Graph * pGraph,
        Port * pDescribedPort, 
        Port* pDescriberPort, 
        DESCRIPTORFUNC func=DF_SIZE
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Put the given graph in the running state. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    RunGraph(
        Graph * pGraph
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes a datablock into the given channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Push(
        Channel * pChannel,
        Datablock * pBlock
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pulls the next datablock from the given channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pChannel">     [in,out] If non-null, the channel. </param>
    /// <param name="dwTimeout">    (optional) the timeout. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Pull(
        Channel * pChannel,
        DWORD dwTimeout=INFINITE
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check to see if the graph is well-formed. This is not an exhaustive check, but a
    ///             collection of obvious sanity checks. If the bFailOnWarning flag is set, then the
    ///             runtime will exit the process if it finds anything wrong with the graph.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="pGraph">           [in] non-null, the graph. </param>
    /// <param name="bVerbose">         (optional) verbose output?. </param>
    /// <param name="bFailOnWarning">   (optional) fail on warning flag: if set, exit the process
    ///                                 when malformed graph elements are found. </param>
    ///
    /// <returns>   PTRESULT--use PTSUCCESS/PTFAILED macros: PTASK_OK:   the graph is well-formed
    ///             PTASK_ERR_GRAPH_MALFORMED: the graph is malformed in a way that cannot be
    ///                                        tolerated by the runtime. Or the the issue may be
    ///                                        tolerable but the user requested fail on warning.
    ///             PTASK_WARNING_GRAPH_MALFORMED: the graph is malformed in a way that can be
    ///                                        tolerated by the runtime.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    CheckGraphSemantics(
        Graph * pGraph,
        BOOL bVerbose=TRUE,
        BOOL bFailOnWarning=FALSE);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Get the GPU Lynx metric specified for instrumenting GPU kernels.
	/// </summary>
    ///
    /// <remarks>   t-nailaf, 5/21/13. </remarks>
    ///
    /// <param name="metric">    the specified instrumentation metric. </param>
    ///-------------------------------------------------------------------------------------------------

	INSTRUMENTATIONMETRIC GetInstrumentationMetric();


	///-------------------------------------------------------------------------------------------------
    /// <summary>   Control the PTask runtime mode for instrumenting GPU kernels. The 
	///				instrumentation metric specifies one of the default metrics made available by GPU Lynx 
	///				for instrumenting kernels (such as activityFactor, memoryEfficiency, etc).
    ///             </summary>
    ///
    /// <remarks>   t-nailaf, 5/21/13. </remarks>
    ///
    /// <param name="metric"> specifies the metric used to instrument kernels.    </param>
    ///-------------------------------------------------------------------------------------------------

	void SetInstrumentationMetric(INSTRUMENTATIONMETRIC metric);

	///-------------------------------------------------------------------------------------------------
        /// <summary>   Instrument the PTX module, given the PTX file location and the 
        ///             instrumented PTX file location.
        /// </summary>
        ///
        /// <remarks>   t-nailaf, 5/21/13. </remarks>
        ///
        /// <param name="ptxFile"> specifies the path to the original PTX file.    </param>
        /// <param name="instrumentedPTXFile"> specifies the path to the instrumented PTX file.    </param>
        ///-------------------------------------------------------------------------------------------------
	void
	Instrument(
		const char * ptxFile,
		const char *instrumentedPTXFile);

   	void CleanupInstrumentation(void);

    	BOOL Instrumented(void);

    	void SetInstrumented(BOOL b);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a handle for the runtime terminate event--handy for any 
    ///             internal services or user code that may need to know about 
    ///             runtime shutdown calls that occur on other threads of control. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   The runtime terminate event. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE GetRuntimeTerminateEvent();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a handle for the runtime initialization complete event--handy for any 
    ///             internal services or user code that may need to know whether
    ///             runtime initialization is complete.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   The runtime terminate event. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE GetRuntimeInitializedEvent();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a graph. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <param name="lpszGraphName">    (optional) [in,out] If non-null, name of the graph. </param>
    ///
    /// <returns>   The new graph. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTask::Graph * CreateGraph();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a graph. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <param name="lpszGraphName">    (optional) [in,out] If non-null, name of the graph. </param>
    ///
    /// <returns>   The new graph. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTask::Graph * CreateGraph(char * lpszGraphName);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the kernels. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    static void DestroyKernels();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates runtime synchronise objects. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    static void CreateRuntimeSyncObjects(BOOL bLockedOnExit);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the runtime synchronise objects. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    static void DestroyRuntimeSyncObjects(BOOL bLockedOnEntry);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the runtime. </summary>
    ///
    /// <remarks>   Crossbac, 7/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    static void LockRuntime();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the runtime. </summary>
    ///
    /// <remarks>   Crossbac, 7/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    static void UnlockRuntime();

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Collect instrumentation data. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///
	/// <param name="szEventName">	   	[in,out] If non-null, name of the event. </param>
	///
	/// <returns>	. </returns>
	///-------------------------------------------------------------------------------------------------

	double CollectInstrumentationData(char * szEventName);

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Collect instrumentation data. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///
	/// <param name="szEventName">	   	[in,out] If non-null, name of the event. </param>
	///
	/// <returns>	. </returns>
	///-------------------------------------------------------------------------------------------------

	double CollectInstrumentationData(char * szEventName, UINT& uiSamples, double& dMin, double& dMax);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets disable device to host xfer. This is a debug/instrumentation
    ///             setting that simply elides backend DtoH xfer calls. The only use
    ///             is estimating DtoH xfer impact on performance--it will clearly 
    ///             result in wrong answers for any workload that wants results back
    ///             from a GPU! Use with caution.</summary>
    ///
    /// <remarks>   Crossbac, 3/25/2014. </remarks>
    ///
    /// <param name="bDisable"> true to disable, false to enable. </param>
    ///-------------------------------------------------------------------------------------------------

    void SetDisableDeviceToHostXfer(BOOL bDisable);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets disable device to host xfer. This is a debug/instrumentation
    ///             setting that simply elides backend DtoH xfer calls. The only use
    ///             is estimating DtoH xfer impact on performance--it will clearly
    ///             result in wrong answers for any workload that wants results back
    ///             from a GPU! Use with caution.</summary>
    ///
    /// <remarks>   Crossbac, 3/25/2014. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL GetDisableDeviceToHostXfer();

}; 
};

#endif
