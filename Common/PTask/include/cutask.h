//--------------------------------------------------------------------------------------
// File: CUTask.h
// CUDA based task
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _CUDA_TASK_H_
#define _CUDA_TASK_H_
#ifdef CUDA_SUPPORT 

#include "primitive_types.h"
#include "accelerator.h"
#include "task.h"
#include "cuhdr.h"
#include <map>
#include <vector>
#include <list>


namespace PTask {

    class CompiledKernel;

	class CUTask : public Task {

    friend class GeometryEstimator;
    friend class XMLReader;
    friend class XMLWriter;
	
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="hRuntimeTerminateEvt"> Handle of the global terminate event. </param>
        /// <param name="hGraphTeardownEvent">  Handle of the stop event. </param>
        /// <param name="hGraphStopEvent">      Handle of the running event. </param>
        /// <param name="hGraphRunningEvent">   The graph running event. </param>
        /// <param name="pCompiledKernel">  The CompiledKernel associated with this task. </param>
        ///-------------------------------------------------------------------------------------------------

		CUTask(
            __in HANDLE hRuntimeTerminateEvt, 
            __in HANDLE hGraphTeardownEvent, 
            __in HANDLE hGraphStopEvent, 
            __in HANDLE hGraphRunningEvent,
            __in CompiledKernel * pCompiledKernel
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

		virtual ~CUTask();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a PTask. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerators">    [in] non-null, list of accelerators this task might run on. </param>
        /// <param name="pCompiledKernel">  [in,out] If non-null, the compiled kernel. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT Create( 
                            std::set<Accelerator*>& pAccelerators, 
                            CompiledKernel * pCompiledKernel 
                            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Runs this ptask. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

		virtual BOOL PlatformSpecificDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes instrumentation. </summary>
        ///
        /// <remarks>   t-nailaf, 06/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

		virtual void InitializeInstrumentation();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalizes instrumentation. </summary>
        ///
        /// <remarks>   t-nailaf, 06/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

		virtual void FinalizeInstrumentation();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a compute geometry. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="tgx">  (optional) the thread group X dimensions. </param>
        /// <param name="tgy">  (optional) the thread group Y dimensions. </param>
        /// <param name="tgz">  (optional) the thread group Z dimensions. </param>
        ///-------------------------------------------------------------------------------------------------

		virtual void SetComputeGeometry(int tgx=1, int tgy=1, int tgz=1 );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a block and grid size. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="grid">     The grid. </param>
        /// <param name="block">    The block. </param>
        ///-------------------------------------------------------------------------------------------------

		virtual void SetBlockAndGridSize(PTASKDIM3 grid, PTASKDIM3 block);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a synchronization timestamp. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the p. </param>
        ///
        /// <returns>   The synchronization timestamp. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetSynchronizationTimestamp(Accelerator * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Increment synchronise timestamp. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
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

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Estimate dispatch dimensions. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void EstimateDispatchDimensions();

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
        /// <summary>   Calculates the parameter offsets. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ComputeParameterOffsets();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a parameter indeces to 'indexmap'. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="portmap">  [in,out] [in,out] If non-null, the portmap. </param>
        /// <param name="indexmap"> [in,out] [in,out] If non-null, the indexmap. </param>
        ///-------------------------------------------------------------------------------------------------

        void AddParameterIndeces(
            std::map<UINT, Port*>& portmap,
            std::map<UINT, Port*>& indexmap); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Collect migration resources. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="vblocks">  [in,out] [in,out] If non-null, the vblocks. </param>
        /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
        /// <param name="vstreams"> [in,out] The vstreams. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL CollectMigrationResources(
                std::list<Datablock*> &vblocks,
                std::list<Accelerator*> &vaccs,
                std::list<CUstream> &vstreams);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes the ps dispatch enter action. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="pContext"> The context. </param>
        /// <param name="hStream">  The stream. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL OnPSDispatchEnter(CUstream hStream);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes the ps dispatch exit action. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="pContext"> The context. </param>
        /// <param name="hStream">  The stream. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL OnPSDispatchExit(CUstream hStream);

        std::map<Accelerator*, CUfunction>  m_pCSMap;	
        std::map<Accelerator*, CUmodule>    m_pModuleMap;	
        std::map<Port*, UINT>               m_pParameterOffsets;
        UINT                                m_uiParameterSize;
        BOOL                                m_bParameterOffsetsInitialized;
		UINT						        m_nPreferredXDim;
		UINT						        m_nPreferredYDim;
		UINT						        m_nPreferredZDim;
        BOOL                                m_bGeometryExplicit;
		BOOL								m_bThreadBlockSizesExplicit;
		PTASKDIM3							m_pThreadBlockSize;
		PTASKDIM3							m_pGridSize;
        CUevent                             m_hPSDispatchStart;
        CUevent                             m_hPSDispatchEnd;
        BOOL                                m_bPSDispatchEventsValid;
	};
};
#endif // CUDA_SUPPORT
#endif