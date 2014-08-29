//--------------------------------------------------------------------------------------
// File: dxtask.h
// directx based task
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _DX_TASK_H_
#define _DX_TASK_H_

#include "primitive_types.h"
#include "ptdxhdr.h"
#include "accelerator.h"
#include "dxaccelerator.h"
#include "task.h"
#include "channel.h"
#include "CompiledKernel.h"
#include <map>
#include <vector>
#include <set>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Task running accelerator code that
    /// 			is supported by the DirectX 11 runtime. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class DXTask : public Task {

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
        /// <param name="pCompiledKernel">  The CompiledKernel associated with this task. </param>
        ///-------------------------------------------------------------------------------------------------

        DXTask(
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

        virtual ~DXTask();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerators">    [in,out] [in,out] If non-null, the accelerators. </param>
        /// <param name="pKernel">          [in,out] If non-null, the kernel. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT 
        Create( 
            __in std::set<Accelerator*>& pAccelerators, 
            __in CompiledKernel * pKernel 
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform platform-specific calls to dispatch the task. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificDispatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a compute geometry. </summary>
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
        /// <summary>   Perform the platform-specific work required to bind an
        /// 			individual input parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificBindInput(Port * pPort, int ordinal, UINT uiActualIndex, PBuffer * pBuffer);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the platform-specific work required to bind an
        /// 			individual output parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PlatformSpecificBindOutput(Port * pPort, int ordinal, UINT uiActualIndex, PBuffer * pBuffer);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the platform-specific work required to bind an
        /// 			individual input parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="ordinal">  [in,out] The ordinal. </param>
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
        /// <summary>   Bind shader. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL BindExecutable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind shader. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindExecutable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind inputs. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindInputs();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind outputs. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindOutputs();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind constants. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnbindConstants();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for a channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="uiUID">    The uid. </param>
        /// <param name="p">        [in,out] If non-null, the p. </param>
        /// <param name="siz">      The siz. </param>
        ///
        /// <returns>   null if it fails, else the found channel. </returns>
        ///-------------------------------------------------------------------------------------------------

        Channel * FindChannel(UINT uiUID, Channel ** p, int siz);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for index of a given channe. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="uiUID">    The uid. </param>
        /// <param name="p">        [in,out] If non-null, the p. </param>
        /// <param name="siz">      The siz. </param>
        ///
        /// <returns>   The found channel index. </returns>
        ///-------------------------------------------------------------------------------------------------

        int FindChannelIndex(UINT uiUID, Channel ** p, int siz);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Removes the channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="index">    Zero-based index of the. </param>
        /// <param name="p">        [in,out] If non-null, the p. </param>
        /// <param name="psiz">     [in,out] If non-null, the psiz. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL RemoveChannel(int index, Channel ** p, UINT * psiz);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the channels. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="p">            [in,out] If non-null, the p. </param>
        /// <param name="psiz">         [in,out] If non-null, the psiz. </param>
        /// <param name="bDeallocate">  true to deallocate. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL ReleaseChannels(Channel ** p, UINT * psiz, BOOL bDeallocate);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Estimate dispatch dimensions. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void EstimateDispatchDimensions(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Estimate dispatch dimensions helper function. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        /// <param name="x">        [in,out] The x coordinate. </param>
        /// <param name="y">        [in,out] The y coordinate. </param>
        /// <param name="z">        [in,out] The z coordinate. </param>
        ///-------------------------------------------------------------------------------------------------

        void __estimateDispatchDimensions(Datablock * pBlock, UINT& x, UINT& y, UINT& z);

        /// <summary> The compute shader map </summary>
        std::map<Accelerator*, ID3D11ComputeShader*>        m_pCSMap;	
        
        /// <summary> The preferred number of thread 
        /// 		  groups to spawn in the X dimension 
        /// 		  </summary>
        UINT            m_nPreferredXDim;
        
        /// <summary> The preferred number of thread 
        /// 		  groups to spawn in the Y dimension 
        /// 		  </summary>
        UINT			m_nPreferredYDim;
        
        /// <summary> The preferred number of thread 
        /// 		  groups to spawn in the Z dimension 
        /// 		  </summary>
        UINT			m_nPreferredZDim;
                
        /// <summary> true if the compute geometry was 
        /// 		  explicitly set by a call from a
        /// 		  user program. </summary>
        BOOL            m_bGeometryExplicit;

        /// <summary> true if we estimated the 
        /// 		  geometry based on datablock template
        /// 		  or datablock properties. 
        /// 		  </summary>
        BOOL            m_bGeometryEstimated;

        /// <summary> Platform specific objects: a list of ShaderResourceView
        /// 		  pointers, reused for binding inputs on every dispatch.
        /// 		  </summary>
        ID3D11ShaderResourceView**      m_ppInputSRVs;

        /// <summary> Platform specific objects: a list of ID3D11UnorderedAccessView
        /// 		  pointers, reused for binding outputs on every dispatch.
        /// 		  </summary>
        ID3D11UnorderedAccessView **    m_ppOutputUAVs;

        /// <summary> Platform specific objects: a list of ID3D11Buffer
        /// 		  pointers, reused for binding constants on every dispatch.
        /// 		  </summary>        
        ID3D11Buffer**                  m_ppConstantBuffers;

        /// <summary>   The p 2 p dispatch input locks. </summary>
        std::set<PBuffer*>            m_vP2PDispatchInputLocks;

        /// <summary>   The p 2 p dispatch output locks. </summary>
        std::set<PBuffer*>            m_vP2PDispatchOutputLocks;

    };

};
#endif
