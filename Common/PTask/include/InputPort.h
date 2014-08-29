//--------------------------------------------------------------------------------------
// File: InputPort.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _INPUT_PORT_H_
#define _INPUT_PORT_H_

#include "primitive_types.h"
#include "port.h"

namespace PTask {

    class Channel;
    class Datablock;
    class DatablockTemplate;
    class Accelerator;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   InputPort: a port subclass specialized to handle binding to input resources in
    ///             Task nodes.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class InputPort : public Port {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        InputPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~InputPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is occupied. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   true if occupied, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			IsOccupied();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Pulls a datablock from this port, potentially blocking until one becomes available.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		Pull();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets an iteration source. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetIterationSource(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the iteration source. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the iteration source. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port *          GetIterationSource();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Returns the datablock occupying this port without removing it. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the current block. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *     Peek();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pushes a datablock into this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pDatablockToPush"> [in,out] If non-null, the Datablock* to push. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			Push(Datablock* pDatablockToPush);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind control channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pChannelToBind">   [in,out] If non-null, the channel to bind. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			BindControlChannel(Channel * pChannelToBind);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind control channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void			UnbindControlChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the destination buffer. Should be a no-op for InputPort.</summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   null if it fails, else the destination buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		GetDestinationBuffer(Accelerator * pAccelerator=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a destination buffer. No-op for InputPort. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pDatablock">   [in,out] If non-null, the datablock. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual	void			SetDestinationBuffer(Datablock * pDatablock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets an in out consumer. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pInOutConsumerPort">   [in,out] If non-null, the in out consumer port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetInOutConsumer(Port* pInOutConsumerPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a block to be the permanently sticky block for this port. Obviously, only
        ///             valid for certain kinds of ports (input varieties). Use for blocks that will have
        ///             only one value for the lifetime of the graph, to avoid creating and manageing an
        ///             exposed channel or initializer channel that will only every be used once. Do not
        ///             connect an upstream channel to ports that have been configured with a permanent
        ///             block.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    If non-null, the Datablock* to push. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			SetPermanentBlock(Datablock * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the in out consumer. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the in out consumer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port*           GetInOutConsumer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the replayable block. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ReleaseReplayableBlock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Start iteration. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="uiIterations"> The iterations. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            BeginIterationScope(UINT uiIterations);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   complete scoped iteration. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="uiIterations"> The iterations. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            EndIterationScope();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pDatablockTemplate">   [in,out] If non-null, the datablock template. </param>
        /// <param name="uiUniqueIdentifier">   Unique identifier. </param>
        /// <param name="lpszVariableBinding">  [in,out] If non-null, the variable binding. </param>
        /// <param name="nParameterIndex">      Zero-based index of the parameter. </param>
        /// <param name="nInOutRouteIdx">       Zero-based index of the in out route. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Port *			Create(DatablockTemplate * pDatablockTemplate, 
                                       UINT uiUniqueIdentifier, 
                                       char * lpszVariableBinding, 
                                       int nParameterIndex, 
                                       int nInOutRouteIdx
                                       );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has block pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if block pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasBlockPool();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force block pooling for a port that has an up-stream allocator. In general, when
        ///             we have an upstream allocator (meta) port, the runtime will not create a block
        ///             pool for the corresponding output port. This turns out to put device-side
        ///             allocation on the critical path in some cases, so we provide a way to override
        ///             that behavior and allow a port to create a pool based on some size hints. When
        ///             there is a block available with sufficient space in the pool, the meta port can
        ///             avoid the allocation and draw from the pool.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 9/25/2012. </remarks>
        ///
        /// <param name="nPoolSize">                Size of the block pool. </param>
        /// <param name="nStride">                  The stride. </param>
        /// <param name="nDataBytes">               The data in bytes. </param>
        /// <param name="nMetaBytes">               The meta in bytes. </param>
        /// <param name="nTemplateBytes">           The template in bytes. </param>
        /// <param name="bPageLockHostViews">       (optional) the page lock host views. </param>
        /// <param name="bEagerDeviceMaterialize">  (optional) the eager device materialize. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            
        ForceBlockPoolHint(
            __in UINT nPoolSize,
            __in UINT nStride,
            __in UINT nDataBytes,
            __in UINT nMetaBytes,
            __in UINT nTemplateBytes,
            __in BOOL bPageLockHostViews=FALSE,
            __in BOOL bEagerDeviceMaterialize=FALSE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   add a new block to the pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void AddNewBlock(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return a block to the pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void ReturnToPool(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   The pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetPoolSize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets request page locked. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="bPageLocked">  true to lock, false to unlock the page. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetRequestsPageLocked(BOOL bPageLocked);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets request page locked. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL GetRequestsPageLocked();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
        ///             
        ///             Allocation of data-blocks and platform-specific buffers can be a signficant
        ///             latency expense at dispatch time. We can actually preallocate output datablocks
        ///             and create device- side buffers at graph construction time. For each node in the
        ///             graph, allocate data blocks on any output ports, and create device-specific
        ///             buffers for all accelerators capable of executing the node.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/15/2012. </remarks>
        ///
        /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
        ///                                 allocated in the pool may be required. </param>
        /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
        /// 								Runtime::GetICBlockPoolSize() will be used to determine the
        /// 								size of the pool. </param>
        /// 								
        /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
        ///             pooling, return false as well.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            
        AllocateBlockPool(
            __in std::vector<Accelerator*>* pAccelerators,
            __in unsigned int               uiPoolSize=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroys the block pool. AddRef everything in the bool, set its owner
        ///             to null, and then release it. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void
        DestroyBlockPool(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queries if a block pool is active and able to deliver/return blocks. </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <returns>   true if a block pool is active, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        IsBlockPoolActive(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
        ///             Asynchronous version. Only allocates device-space buffers
        ///             in the first pass. Second pass queues all the copies.
        ///             This function handles only the first pass.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/15/2012. </remarks>
        ///
        /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
        ///                                 allocated in the pool may be required. </param>
        /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
        /// 								Runtime::GetICBlockPoolSize() will be used to determine the
        /// 								size of the pool. </param>
        /// 								
        /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
        ///             pooling, return false as well.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            
        AllocateBlockPoolAsync(
            __in std::vector<Accelerator*>* pAccelerators,
            __in unsigned int               uiPoolSize=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
        ///             Asynchronous version. Only allocates device-space buffers
        ///             in the first pass. Second pass queues all the copies.
        ///             This function handles the second pass.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/15/2012. </remarks>
        ///
        /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
        ///                                 allocated in the pool may be required. </param>
        /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
        /// 								Runtime::GetICBlockPoolSize() will be used to determine the
        /// 								size of the pool. </param>
        /// 								
        /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
        ///             pooling, return false as well.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        FinalizeBlockPoolAsync(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find the maximal capacity downstream port/channel path starting at this port.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 1/3/2014. </remarks>
        ///
        /// <param name="vTasksVisited">    [in,out] [in,out] If non-null, the tasks visited. </param>
        /// <param name="vPath">            [in,out] [in,out] If non-null, full pathname of the file. </param>
        ///
        /// <returns>   The found maximal downstream capacity. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT 
        FindMaximalDownstreamCapacity(
            __inout std::set<Task*>& vTasksVisited,
            __inout std::vector<Channel*>& vPath
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is an explicit memory space transition point. 
        ///             We return true only when we know for certain that this task 
        ///             executes on one GPU and at least one downstream tasks definitely
        ///             needs a view of our outputs on another GPU. In general we can only
        ///             tell this with high precision when there is task affinity involved.
        ///             We use this to set the sharing hint on the access flags for blocks
        ///             allocated, which in turn allows some back ends to better optimize GPU-side
        ///             buffer allocation and data transfer. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/13/2014. </remarks>
        ///
        /// <returns>   true if explicit memory space transition point, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsExplicitMemorySpaceTransitionPoint();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check type-specific semantics. Return true if all the structures are initialized
        ///             for this port in a way that is consistent with a well-formed graph. Called by
        ///             CheckSemantics()
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="pos">      [in,out] output string stream. </param>
        /// <param name="pGraph">   [in,out] non-null, the graph. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            CheckTypeSpecificSemantics(std::ostream * pos,
                                                           PTask::Graph * pGraph);
    protected:

        /// <summary> The output port that is the consumer
        /// 		  if this port is part of an in/out pair
        /// 		  </summary>
        Port *                  m_pInOutConsumer;

    };

};
#endif
