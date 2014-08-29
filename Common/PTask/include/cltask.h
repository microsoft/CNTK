//--------------------------------------------------------------------------------------
// File: CLTask.h
//
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _CL_PTASK_H_
#define _CL_PTASK_H_
#ifdef OPENCL_SUPPORT

#include "primitive_types.h"
#include "cuaccelerator.h"
#include "task.h"
#include "channel.h"
#include "CompiledKernel.h"
#include "oclhdr.h"
#include <map>
#include <vector>

namespace PTask {

	class CLTask : public Task {

        friend class XMLReader;
        friend class XMLWriter;

	public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="hRuntimeTerminateEvt"> Handle of the terminate. </param>
        /// <param name="hGraphTeardownEvent">  Handle of the stop event. </param>
        /// <param name="hGraphStopEvent">      Handle of the running event. </param>
        /// <param name="hGraphRunningEvent">   The graph running event. </param>
        /// <param name="pCompiledKernel">  The CompiledKernel associated with this task. </param>
        ///-------------------------------------------------------------------------------------------------

		CLTask(
            __in HANDLE hRuntimeTerminateEvt, 
            __in HANDLE hGraphTeardownEvent, 
            __in HANDLE hGraphStopEvent, 
            __in HANDLE hGraphRunningEvent,
            __in CompiledKernel * pCompiledKernel
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

		virtual ~CLTask();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pAccelerators">    [in] non-null, the accelerators to compile for. </param>
        /// <param name="pKernel">          [in,out] If non-null, the kernel. </param>
        ///
        /// <returns>   HRESULT (use SUCCEEDED/FAILED macros) </returns>
        ///-------------------------------------------------------------------------------------------------

		virtual HRESULT Create(std::set<Accelerator*>& pAccelerators, 
                               CompiledKernel * pKernel 
                               );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Runs this CLTask. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a compute geometry. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="tgx">  (optional) the thread group x dimensions. </param>
        /// <param name="tgy">  (optional) the thread group y dimensions. </param>
        /// <param name="tgz">  (optional) the thread group z dimensions. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetComputeGeometry(int tgx=1, int tgy=1, int tgz=1 );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a block and grid size. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="grid">     The grid. </param>
        /// <param name="block">    The block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetBlockAndGridSize(PTASKDIM3 grid, PTASKDIM3 block);

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
        /// <summary>   Bind parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pCS">      The create struct. </param>
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

		void BindParameter(cl_kernel pCS, PBuffer * pBuffer, Port * pPort, int &ordinal);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the estimate global size. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

		UINT EstimateGlobalSize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Loads source code from a file before compiling. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="cFilename">        Filename of the file. </param>
        /// <param name="cPreamble">        The preamble. </param>
        /// <param name="szFinalLength">    [in,out] If non-null, length of the final. </param>
        ///
        /// <returns>   null if it fails, else the source. </returns>
        ///-------------------------------------------------------------------------------------------------

		char* CLTask::LoadSource(
			const char* cFilename, 
			const char* cPreamble, 
			size_t* szFinalLength
			);

        /// <summary> A map from accelerator to compiled kernel object, 
        /// 		  allowing the system to dispatch on arbitrary
        /// 		  accelerators by selecting the right object 
        /// 		  for the dispatch accelerator.
        /// 		  </summary>
        std::map<Accelerator*, cl_kernel>   m_pCSMap;	
        
        /// <summary> A map from accelerator to module, 
        /// 		  allowing the system to dispatch on arbitrary
        /// 		  accelerators by selecting the right object 
        /// 		  for the dispatch accelerator.
        /// 		  </summary>
        std::map<Accelerator*, cl_program>  m_pModuleMap;			
		
        /// <summary> The preferred x thread group size </summary>
        UINT						        m_nPreferredXDim;
		
        /// <summary> The preferred y thread group size </summary>
		UINT						        m_nPreferredYDim;

        /// <summary> The preferred z thread group size </summary>
		UINT						        m_nPreferredZDim;
        
        /// <summary> true if the user explicitly set the thread
        /// 		  group geometry with a call to 
        /// 		  Task->SetGeometry.
        /// 		  </summary>
        BOOL                                m_bGeometryExplicit;
    };
};
#endif // OPENCL_SUPPORT
#endif // _CLTask_H_
