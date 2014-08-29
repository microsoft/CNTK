//--------------------------------------------------------------------------------------
// File: InitializerChannel.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _INITIALIZER_CHANNEL_H_
#define _INITIALIZER_CHANNEL_H_

#include "primitive_types.h"
#include "channel.h"
#include "BlockPoolOwner.h"
#include <deque>

namespace PTask {

    class BlockPool;
    class Datablock;
    class DatablockTemplate;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   InitializerChannel. Channel subclass specialized to allocate data based
    /// 			on downstream Port template when pulled. Push is meaningless. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class InitializerChannel : public Channel, public BlockPoolOwner {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pGraph">                   [in,out] If non-null, the graph. </param>
        /// <param name="pDatablockTemplate">       [in,out] If non-null, the datablock template. </param>
        /// <param name="hRuntimeTerminateEvent">   Handle of the graph terminate event. </param>
        /// <param name="hGraphTeardownEvt">        The graph teardown event. </param>
        /// <param name="hGraphStopEvent">          Handle of the graph stop event. </param>
        /// <param name="lpszChannelName">          [in,out] If non-null, name of the channel. </param>
        /// <param name="bHasBlockPool">            the has block pool. </param>
        ///-------------------------------------------------------------------------------------------------

        InitializerChannel(
            __in Graph * pGraph,
            __in DatablockTemplate * pDatablockTemplate, 
            __in HANDLE hRuntimeTerminateEvent,
            __in HANDLE hGraphTeardownEvt, 
            __in HANDLE hGraphStopEvent, 
            __in char * lpszChannelName,
            __in BOOL bHasBlockPool
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~InitializerChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the channel is (or can be) connected to a data source or sink that can be
        ///             streamed. Generally speaking, this is a property of the primitive whose IO
        ///             resources are being exposed by this port; consequently this property must be set
        ///             explicitly by the programmer when graph structures that are stateful are
        ///             constructured. For example, in a sort primitive, the main input can be streamed
        ///             (broken into multiple blocks) only if there is a merge network downstream of the
        ///             node performing the sort. Code that feeds the main input port needs to know this
        ///             to decide whether to grow blocks until all data is present, or two push partial
        ///             input.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if the port can stream data, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            CanStream();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if channel is ready. This has a different meaning depending on the channel
        ///             subtype in question, but in general means "is the channel ready to produce or
        ///             consume datablocks?".
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="type"> (optional) the type of the channel. </param>
        ///
        /// <returns>   true if ready, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsReady(CHANNELENDPOINTTYPE type=CE_DST);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pulls a datablock from the channel, potentially timing out after dwTimeout
        ///             milliseconds.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="dwTimeout">    (optional) the timeout in milliseconds. Use 0xFFFFFFFF for no
        ///                             timeout. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock * Pull(DWORD dwTimeout=0xFFFFFFFF);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Returns the first available datablock on the channel without removing it. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the currently available datablock object. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock * Peek();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pushes a datablock into this channel, blocking until there is capacity
        /// 			for an optional timeout in milliseconds. Default timeout is infinite. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pBlock">       [in,out] If non-null, the block. </param>
        /// <param name="dwTimeout">    (optional) the timeout in milliseconds. Use 0xFFFFFFFF for no
        ///                             timeout. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL Push(Datablock* pBlock, DWORD dwTimeout=0xFFFFFFFF);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is block pool candidate. </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <returns>   true if block pool candidate, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsBlockPoolCandidate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is block pool candidate. </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <returns>   true if block pool candidate, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsAcceleratorOnlyBlockPoolCandidate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is block pool candidate. </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <returns>   true if block pool candidate, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsPagelockedBlockPoolCandidate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has block pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if block pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasBlockPool();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is global pool. </summary>
        ///
        /// <remarks>   crossbac, 8/30/2013. </remarks>
        ///
        /// <returns>   true if global pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL BlockPoolIsGlobal();

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
        /// <summary>   Gets the owner name. </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the owner name. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual char *
        GetPoolOwnerName(
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
        /// <summary>   Gets high water mark. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetHighWaterMark();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the total number of blocks owned by the pool. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The total number of blocks owned by the pool (whether they are queued or not). </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetOwnedBlockCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the low water mark. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetLowWaterMark();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the currently available count. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetAvailableBlockCount();

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
        /// <summary>   Query if this channel has downstream writers. An output channel is
        ///             considered a writer because we must conservatively assume consumed
        ///             blocks will be written.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///
        /// <returns>   true if downstream writers, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasDownstreamWriters();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check type-specific semantics. Return true if all the structures are initialized
        ///             for this chanell in a way that is consistent with a well-formed graph. Called by
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

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate a datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		AllocateBlock(AsyncContext * pAsyncContext);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the block that is (or would be) produced in demand to a pull call
        ///             passes all/any predicates.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 2/1/2012. </remarks>
        ///
        /// <param name="ppDemandAllocatedBlock">   [out] If non-null, on exit, the demand allocated
        ///                                         block if all predicates are passed. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            PassesPredicates(Datablock ** ppDemandAllocatedBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a destination buffer for a block with an upstream
        /// 			allocator. Succeeds only if the pool happens to have blocks
        /// 			backed by sufficient resources in all channels that are backed. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   null if it fails, else the destination buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		
        GetBlockFromPool(
            __in Accelerator * pAccelerator=NULL,
            __in UINT uiDataBytes=0,
            __in UINT uiMetaBytes=0,
            __in UINT uiTemplateBytes=0
            );

        /// <summary>   The peeked control propagation signal source. </summary>
        Datablock *             m_pPeekedControlPropagationSignalSrc;

        /// <summary>   true if a data block was peeked to derive a control propagation signal. </summary>
        BOOL                    m_bControlBlockPeeked;

        /// <summary>   The code for the peeked control signal. </summary>
        CONTROLSIGNAL           m_luiPeekedControlSignal;

        /// <summary>   The block pool. </summary>
        BlockPool *             m_pBlockPool;

    };

};
#endif