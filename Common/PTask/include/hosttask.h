//--------------------------------------------------------------------------------------
// File: HostTask.h
// Host based task
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _HOST_TASK_H_
#define _HOST_TASK_H_

#include "primitive_types.h"
#include "accelerator.h"
#include "cuaccelerator.h"
#include "task.h"
#include "channel.h"
#include "datablock.h"
#include "CompiledKernel.h"
#include <map>
#include <vector>
#include <list>

using namespace PTask;

///-------------------------------------------------------------------------------------------------
/// <summary>   function signature for simple host tasks. </summary>
///
/// <remarks>   Crossbac, 5/16/2012. </remarks>
///-------------------------------------------------------------------------------------------------

typedef void (__stdcall *LPFNHOSTTASK)(
    UINT nArguments, 
    void **ppArguments
    );

///-------------------------------------------------------------------------------------------------
/// <summary>   function signature for host tasks that have dependences on other accelerators.
///             The BOOL array contains entries which are true if that entry corresponds to an
///             input already materialized on the dependent device, false otherwise. The
///             pvDeviceBindings array contains entries which are meaningful when the entry at
///             the same index in the BOOL array is true, and is a platform-specific device id.
///             Generated code must know how to use these IDs.
///             </summary>
///
/// <remarks>   Crossbac, 5/16/2012. </remarks>
///-------------------------------------------------------------------------------------------------

typedef void (__stdcall *LPFNDEPHOSTTASK)(
    UINT nArguments, 
    void **ppArguments, 
    BOOL * pbIsDependentBinding, 
    void ** pvDeviceBindings, 
    UINT nDeps, 
    void ** pDeps);

///-------------------------------------------------------------------------------------------------
/// <summary>   Defines a structure for providing dependent accelerator context information
///             to a host task. Moving from LPFNHOSTTASK and LPFNDEPHOSTTASK approach 
///             because we have to change the signature every time there is a new requirement.
///             Using a descriptor struct instead allows us to grow the structure as needed
///             without having to change a bunch of code. </summary>
///
/// <remarks>   Crossbac, 2/6/2013. </remarks>
///-------------------------------------------------------------------------------------------------

typedef struct _dependent_context_t {
    /*
pbDependentBindings: 
pvDependentBindings: 
nDeps: 
pDepDevs: 
pStreams: a vector of length nDeps (always 1 for you), each member of which can be typecast (in your case) to type CUstream_t. 

    */

    /// <summary>   The number of bytes in the dependent context
    ///             descriptor structure. 
    ///             </summary>
    UINT cbDependentContext; 

    /// <summary>   The number of arguments in the task argument list. </summary>
    UINT nArguments;

    /// <summary>   The number of dependent accelerators assigned. </summary>
    UINT nDependentAccelerators;

    /// <summary>   Reserved, pad to 16 bytes before pointer types. </summary>
    UINT uiReserved0;

    /// <summary>   The arguments, to be typecast according to what the
    ///             task knows implicitly as well as the dependent accelerator
    ///             binding information provided in the subsequent members
    ///             of this structure. 
    ///             </summary>
    void **ppArguments;

    /// <summary>   A vector of length nArguments, specifying the datablock
    ///             that each argument is associated with.
    ///             </summary>
    Datablock ** ppDatablocks;

    /// <summary>   a vector of BOOL, of length nArguments. If a given member is TRUE, you can expect
    ///             the data for the argument in question to be pre-materialized in device space.
    ///             </summary>
    BOOL * pbIsDependentBinding;

    /// <summary>   a vector of length nArguments, whose members can be typecast to platform-specic
    ///             device objects (e.g. CUdevice): if pbDependentBindings[i] is TRUE, then
    ///             pvDependentBindings[i] is a valid platform specific object.
    ///             </summary>
    void ** pvDeviceBindings;

    /// <summary>   a vector of length nDeps (always 1 for you), each member of which can be typecast
    ///             (e.g. type CUdevice).
    ///             </summary>
    void ** pDependentDevices;

    /// <summary>   The streams: a vector of length nDependentAccelerators each member of which can
    ///             be typecast to a platform-specific asynchronous context object (e.g. type
    ///             CUstream_t).
    ///             </summary>
    void ** pStreams;

    /// <summary>   A pointer to the PTask-assigned task name. Enables less ambiguous debug
    ///             output for graphs that use the same host entry point in multiple
    ///             places in the graph. 
    ///             </summary>
    char *  lpszTaskName;

} DEPENDENTCONTEXT, *LPDEPENDENTCONTEXT;

///-------------------------------------------------------------------------------------------------
/// <summary>   function signature for host tasks that have dependences on other accelerators.
///             The structure contains members which allow the task dispatch code to determine
///             whether entries are already materialized on the dependent device, as well as
///             enabling the code to get platform specific objects such as device ids and stream
///             handles where needed. Generated code must know how to use this structure. 
///             Currently, the the task's BindDependentAcceleratorClass member is called
///             with the bRequestPSObjects parameter == TRUE, the code assumes the host task
///             entry point follows this form; otherwise the legacy versions above 
///             (LPFNDEPHOSTTASK, LPFNHOSTTASK) are used for backward compatibility.
///             </summary>
///
/// <remarks>   Crossbac, 5/16/2012. </remarks>
///-------------------------------------------------------------------------------------------------

typedef void (__stdcall *LPFNDEPHOSTTASKEX)(LPDEPENDENTCONTEXT);

namespace PTask {

    static const int MAXARGS=64;

    class HostTask : public Task {

        friend class XMLReader;
        friend class XMLWriter;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="hRuntimeTerminateEvt"> Handle of the graph terminate event. </param>
        /// <param name="hGraphTeardownEvent">  Handle of the stop event. </param>
        /// <param name="hGraphStopEvent">      Handle of the running event. </param>
        /// <param name="hGraphRunningEvent">   The graph running event. </param>
        /// <param name="pCompiledKernel">      The CompiledKernel associated with this task. </param>
        ///-------------------------------------------------------------------------------------------------

        HostTask(
            __in HANDLE hRuntimeTerminateEvt, 
            __in HANDLE hGraphTeardownEvent, 
            __in HANDLE hGraphStopEvent, 
            __in HANDLE hGraphRunningEvent,
            __in CompiledKernel * pCompiledKernel
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~HostTask();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates this task. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerators">    [in,out] [in,out] If non-null, the accelerators. </param>
        /// <param name="pKernel">          [in,out] If non-null, the kernel. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT Create( std::set<Accelerator*>& pAccelerators, CompiledKernel * pKernel );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dispatches this task. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificDispatch();


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the compute geometry. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="nThreadGroupsX">   (optional) the thread groups in x. </param>
        /// <param name="nThreadGroupsY">   (optional) the thread groups in y. </param>
        /// <param name="nThreadGroupsZ">   (optional) the thread groups in z. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetComputeGeometry(int nThreadGroupsX=1, int nThreadGroupsY=1, int nThreadGroupsZ=1);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a block and grid size. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="grid">     The grid. </param>
        /// <param name="block">    The block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetBlockAndGridSize(PTASKDIM3 grid, PTASKDIM3 block);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a synchronization timestamp. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the p. </param>
        ///
        /// <returns>   The synchronization timestamp. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetSynchronizationTimestamp(Accelerator * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Increment synchronise timestamp. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the p. </param>
        ///-------------------------------------------------------------------------------------------------

        void IncrementSyncTimestamp(Accelerator * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   When the graph is complete, (indicated because Graph.Run was called), this method
        ///             is called on every task to allow tasks to perform and one-time initializations
        ///             that cannot be performed without knowing that the structure of the graph is now
        ///             static. For example, computing parameter offset maps for dispatch.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/5/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
        
        virtual void PlatformSpecificOnGraphComplete();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the platform-specific work required to bind an individual input parameter.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pPort">            [in,out] If non-null, the port. </param>
        /// <param name="ordinal">          The ordinal. </param>
        /// <param name="uiActualIndex">    Zero-based index of the user interface actual. </param>
        /// <param name="pBuffer">          [in,out] If non-null, the buffer. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificBindInput(Port * pPort, 
                                               int ordinal, 
                                               UINT uiActualIndex, 
                                               PBuffer * pBuffer
                                               );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the platform-specific work required to bind an individual output
        ///             parameter.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pPort">            [in,out] If non-null, the port. </param>
        /// <param name="ordinal">          The ordinal. </param>
        /// <param name="uiActualIndex">    Zero-based index of the user interface actual. </param>
        /// <param name="pBuffer">          [in,out] If non-null, the buffer. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificBindOutput(Port * pPort, 
                                                int ordinal, 
                                                UINT uiActualIndex, 
                                                PBuffer * pBuffer);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the platform-specific work required to bind an individual input parameter.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pPort">            [in,out] If non-null, the port. </param>
        /// <param name="ordinal">          The ordinal. </param>
        /// <param name="uiActualIndex">    Zero-based index of the user interface actual. </param>
        /// <param name="pBuffer">          [in,out] If non-null, the buffer. </param>
        /// <param name="bScalarBinding">   true to scalar binding. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificBindConstant(Port * pPort, 
                                            int ordinal, 
                                            UINT uiActualIndex, 
                                            PBuffer * pBuffer
                                            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform specific finalize bindings. </summary>
        ///
        /// <remarks>   Crossbac, 1/5/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificFinalizeBindings();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind accelerator executable. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL BindExecutable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Collect migration resources. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="vblocks">  [in,out] [in,out] If non-null, the vblocks. </param>
        /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        CollectMigrationResources(
            __inout std::list<Datablock*> &vblocks,
            __inout std::list<Accelerator*> &vaccs
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform-specific dispatch if the task has no dependences on other accelerators.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pCS">  The function pointer address for dispatch. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        PlatformSpecificDispatchNoDependences(
            __in FARPROC pCS
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform-specific dispatch if the task has dependences on other accelerators.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pCS">      The function pointer address for dispatch. </param>
        /// <param name="nDeps">    The number dependent assignments. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        PlatformSpecificDispatchWithDependences(
            __in FARPROC pCS,
            __in UINT nDeps
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform-specific dispatch if the task has dependences on other accelerators.
        ///             This version extends the PlatformSpecificDispatchWithDependences version
        ///             with the ability to provide other platform-specific objects such as stream
        ///             handles through a struct/descriptor based interface. Currently, this is
        ///             called if m_bRequestDependentPSObjects is true, otherwise, legacy versions
        ///             are called.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pCS">      The function pointer address for dispatch. </param>
        /// <param name="nDeps">    The number dependent assignments. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        PlatformSpecificDispatchWithDependencesEx(
            __in FARPROC pCS,
            __in UINT nDeps
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes the ps dispatch enter action. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL OnPSDispatchEnter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes the ps dispatch exit action. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL OnPSDispatchExit();

        /// <summary> map of host-task invocation parameter index to value </summary>
        std::map<int, void*>                m_pParameters;

        /// <summary> map of host-task invocation parameter index to source port </summary>
        std::map<int, Port*>                m_pParameterPorts;

        /// <summary> map of host-task invocation parameter index to datablock </summary>
        std::map<int, Datablock*>           m_pParameterDatablockMap;

        /// <summary> map of function pointers </summary>
        std::map<Accelerator*, FARPROC>     m_pCSMap;	
        
        /// <summary> map of HMODULE handles </summary>
        std::map<Accelerator*, HMODULE>     m_pModuleMap;	
        
        /// <summary> The preferred x size </summary>
        UINT						        m_nPreferredXDim;
        
        /// <summary> The preferred y size </summary>
        UINT						        m_nPreferredYDim;
        
        /// <summary> The preferred z size </summary>
        UINT						        m_nPreferredZDim;
        
        /// <summary> true if the user set the geometry 
        /// 		  explicitly with a call to
        /// 		  SetComputeGeometry.</summary>
        BOOL                                m_bGeometryExplicit;
        
        /// <summary> true if the user set the thread block
        /// 		  sizes explicitly.
        /// 		  </summary>
        BOOL								m_bThreadBlockSizesExplicit;
        
        /// <summary> Size of the thread block </summary>
        PTASKDIM3							m_pThreadBlockSize;
        
        /// <summary> Size of the dispatch grid </summary>
        PTASKDIM3							m_pGridSize;
        
        void*                               m_ppArgs[MAXARGS];
        Datablock*                          m_ppDatablocks[MAXARGS];
        void*                               m_ppDeps[MAXARGS];
        BOOL                                m_pbIsDependentBinding[MAXARGS];
        void*                               m_pvDeviceBindings[MAXARGS];
        void*                               m_ppStreams[MAXARGS];
    };
};
#endif 

