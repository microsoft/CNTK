//--------------------------------------------------------------------------------------
// File: OutputPort.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _OUTPUT_PORT_H_
#define _OUTPUT_PORT_H_

#include "primitive_types.h"
#include "port.h"

namespace PTask {

    class Channel;
    class Datablock;
    class DatablockTemplate;
    class Accelerator;
    class BlockPool;
    class BlockPoolOwner;

    class OutputPort : public Port {
        
        friend class XMLWriter;
        friend class XMLReader;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        OutputPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~OutputPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is occupied. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if occupied, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			IsOccupied();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pulls this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		Pull();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Returns a block occupying this port without removing it. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the current datablock. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		Peek();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pushes a datablock into this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			Push(Datablock* p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a destination buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   null if it fails, else the destination buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		GetDestinationBuffer(Accelerator * pAccelerator=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return the channel index on the target datablock to which
        ///             this port/variable should be bound. Typically this is the
        ///             DATABLOCK_DATA_CHANNEL. However, if this output port is a
        ///             descriptor of another output port, we may want to bind to
        ///             a different buffer in the block at dispatch time.  
        ///             </summary>
        ///
        /// <remarks>   Crossbac, </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   null if it fails, destination channel index. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT		GetDestinationChannel(Accelerator * pAccelerator=NULL);

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
        GetPooledDestinationBuffer(
            __in Accelerator * pAccelerator,
            __in UINT uiDataBytes,
            __in UINT uiMetaBytes=0,
            __in UINT uiTemplateBytes=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a destination buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pDatablock">   [in,out] If non-null, the datablock. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual	void			SetDestinationBuffer(Datablock * pDatablock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a destination buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pDatablock">   [in,out] If non-null, the datablock. </param>
        /// <param name="bAddToPool">   true to add to pool. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual	void			SetDestinationBuffer(Datablock * pDatablock, BOOL bAddToPool);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the destination buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ReleaseDestinationBuffer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is block pool candidate. </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <returns>   true if block pool candidate, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsBlockPoolCandidate();

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
        /// <summary>   Query if this object has block pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if block pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasBlockPool();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
        /// <param name="uiId">                 The identifier. </param>
        /// <param name="lpszVariableBinding">  [in,out] If non-null, the variable binding. </param>
        /// <param name="nParmIdx">             Zero-based index of the n parm. </param>
        /// <param name="nInOutRouteIdx">       Zero-based index of the n in out route. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Port *			
        Create(
            __in DatablockTemplate * pTemplate, 
            __in UINT uiId, 
            __in char * lpszVariableBinding, 
            __in int nParmIdx, 
            __in int nInOutRouteIdx
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate block. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator">     [in,out] (optional)  If non-null, the accelerator. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="bPooled">          true to pooled. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		
        AllocateBlock(
            __in Accelerator * pAccelerator, 
            __in AsyncContext * pAsyncContext, 
            __in BOOL bPooled
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate block. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] (optional)  If non-null, the async context where the
        ///                                 block will be used. </param>
        /// <param name="bPooled">          true to pooled. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		
        AllocateBlock(
            __in AsyncContext * pAsyncContext, 
            __in BOOL bPooled
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate destination block. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] (optional)  If non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			
        AllocateDestinationBlock(
            __in Accelerator * pAccelerator
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind descriptor port to this port </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        /// <param name="func">     The func. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			BindDescriptorPort(Port * pPort, DESCRIPTORFUNC func);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets an in out producer. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetInOutProducer(Port* p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the in out producer. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the in out producer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port *          GetInOutProducer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has output channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if output channel, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasOutputChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets an allocator port. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetAllocatorPort(Port * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the allocator port. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the allocator port. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port *          GetAllocatorPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has allocator input port. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if allocator input port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasAllocatorInputPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a control port. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="p">                [in,out] If non-null, the Datablock* to push. </param>
        /// <param name="bInitiallyOpen">   (optional) the initially open. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetControlPort(Port * p, BOOL bInitiallyOpen=FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the control port. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the control port. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port *          GetControlPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has control port. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if control port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasControlPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Raises a Gate signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SignalGate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has downstream writer ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if downstream writer ports, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasDownstreamWriterPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has downstream readonly ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if downstream readonly ports, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasDownstreamReadonlyPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the downstream writer port count. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, a block that would be pushed into the
        /// 						channel. This allows us to check whether we are dealing with
        /// 						a predicated channel that would release the block. </param>  
        /// 						
        /// <returns>   The downstream writer port count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT            GetDownstreamWriterPortCount(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the downstream readonly port count. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, a block that would be pushed into the
        /// 						channel. This allows us to check whether we are dealing with
        /// 						a predicated channel that would release the block. </param>
        ///
        /// <returns>   The downstream readonly port count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT            GetDownstreamReadonlyPortCount(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has downstream host consumer. </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if downstream host consumer, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasDownstreamHostConsumer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has downstream host consumer. </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <param name="vVisitSet">    [in,out] If non-null, set the visit belongs to. </param>
        ///
        /// <returns>   true if downstream host consumer, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasDownstreamHostConsumer(std::set<OutputPort*> &vVisitSet);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases any datablock held by a given port. We need to do this for all ports
        ///             before deleting ports because it is possible that a block owned by one port's
        ///             block pool can become the owned block (m_pDatablock) for another port. We need to
        ///             ensure that if release calls occur for such a block, they do not attempt to
        ///             return the block to the pool of a port that has been deleted.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ReleaseDatablock();

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
        /// <summary>   Sets the suppress clones property, which allows a user to suppress output cloning
        ///             for blocks on ports with multiple (R/W conflicting) downstream consumers, if the
        ///             programmer happens to know something about the structure of the graph that the
        ///             runtime cannot (or does not detect) and that makes it safe to do so.
        ///             </summary>
        ///
        /// <remarks>   Output port is currently the only meaningful implementer of this method. Crossbac,
        ///             2/29/2012.
        ///             </remarks>
        ///
        /// <param name="bSuppressClones">  true to suppress clones. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetSuppressClones(BOOL bSuppressClones);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the suppress clones property, which allows a user to suppress output cloning
        ///             for blocks on ports with multiple (R/W conflicting) downstream consumers, if the
        ///             programmer happens to know something about the structure of the graph that the
        ///             runtime cannot (or does not detect) and that makes it safe to do so.  Note that
        ///             we do not require a lock to query this property because it is assumed this method
        ///             is used only during graph construction and is not used while a graph is running.
        ///             </summary>
        ///
        /// <remarks>   Output port is currently the only meaningful implementer of this method. Crossbac,
        ///             2/29/2012.
        ///             </remarks>
        ///
        /// <returns>   the value of the suppress clones property. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            GetSuppressClones();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is descriptor port. </summary>
        ///
        /// <remarks>   Crossbac, 2/14/2013. </remarks>
        ///
        /// <returns>   true if descriptor port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsDescriptorPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets descriptor port for the given descriptor function (if there is one). </summary>
        ///
        /// <remarks>   Crossbac, 2/15/2013. </remarks>
        ///
        /// <param name="eFunc">    The function. </param>
        ///
        /// <returns>   null if it fails, else the descriptor port. </returns>
        ///-------------------------------------------------------------------------------------------------

        OutputPort * GetDescriptorPort(DESCRIPTORFUNC eFunc);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets pending allocation. </summary>
        ///
        /// <remarks>   Crossbac, 2/15/2013. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        /// <param name="uiSizeBytes">          The size in bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetPendingAllocationSize(
            __in Accelerator * pDispatchAccelerator, 
            __in UINT          uiSizeBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets pending the allocation size. Should only be called if this port is a
        ///             descriptor port for another output port, and both ports have meta-port allocators
        ///             that determine the buffer sizes required at dispatch. In such a case, we defer
        ///             the block allocation and binding until all sizes are available. This method
        ///             returns the pending size, which should have been stashed in a call to
        ///             SetPendingAllocationSize.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 2/15/2013. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        UINT GetPendingAllocationSize(
            __in Accelerator * pDispatchAccelerator
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Complete pending bindings. </summary>
        ///
        /// <remarks>   Crossbac, 2/14/2013. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void CompletePendingAllocation(
            __in Accelerator * pAccelerator, 
            __in Datablock * pBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets this port to be the scope terminus for a subgraph. Generally speaking, this
        ///             means that it is responsible for popping the control signal context on outbound
        ///             datablocks. Less generally speaking, since the control signal stack is not fully
        ///             used yet, this means the port is responsible for setting specified control signal
        ///             on outbound blocks (without overwriting other existing control signals). The
        ///             default super-class implementation of this method fails because only output ports
        ///             can terminate a scope in a well-formed graph.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="luiSignal">    true to trigger. </param>
        /// <param name="bTerminus">    true to terminus. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL        
        SetScopeTerminus(
            __in CONTROLSIGNAL luiSignal, 
            __in BOOL bTerminus
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is the scope terminus for a subgraph. If it is, 
        ///             it is responsible for appending a control signal to outbound blocks. 
        ///              </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <returns>   true if scope terminus port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsScopeTerminus();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is the scope terminus for a subgraph. If it is, 
        ///             it is responsible for appending a control signal to outbound blocks. 
        ///              </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <returns>   true if scope terminus port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsScopeTerminus(CONTROLSIGNAL luiControlSignal);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the propagated control code. We override this in output port
        ///             because an output port may also be a scope terminus, which means
        ///             that propagated control signals need to include any that are also
        ///             mandated by that role. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The propagated control code. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual CONTROLSIGNAL            GetPropagatedControlSignals();

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

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate a block. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator">     (optional)  If non-null, the accelerator. </param>
        /// <param name="uiDataBytes">      The data in bytes. </param>
        /// <param name="uiMetaBytes">      The meta in bytes. </param>
        /// <param name="uiTemplateBytes">  The template in bytes. </param>
        ///
        /// <returns>   null if it fails, else the new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *
        AllocateBlock(
            __in Accelerator * pAccelerator,
            __in UINT uiDataBytes,
            __in UINT uiMetaBytes,
            __in UINT uiTemplateBytes,
            __in BOOL bPooled
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind described port. </summary>
        ///
        /// <remarks>   Crossbac, 2/14/2013. </remarks>
        ///
        /// <param name="eFunc">    [in,out] If non-null, the function. </param>
        ///-------------------------------------------------------------------------------------------------

        void BindDescribedPort(
            __in Port * pPort,
            __in DESCRIPTORFUNC eFunc
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets block permissions flags for blocks allocated as "destination blocks" on this
        ///             port.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/28/2014. </remarks>
        ///
        /// <param name="bPooledBlock"> True if the block will be added to a pool. </param>
        ///
        /// <returns>   The destination block permissions. </returns>
        ///-------------------------------------------------------------------------------------------------

        BUFFERACCESSFLAGS 
        GetAllocationAccessFlags(
            __in BOOL bPooledBlock
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

        /// <summary> Size of the maximum block pool </summary>
        int                     m_nMaxPoolSize;
        /// <summary> The block pool </summary>
        std::deque<Datablock*>  m_pBlockPool;
        /// <summary>   Blocks in the pool that require reset. </summary>
        std::set<Datablock*>    m_vDirtySet;
        /// <summary> The in out producer if this is an in/out pair</summary>
        Port *                  m_pInOutProducer;
        /// <summary> The output datablock </summary>
        Datablock *             m_pDatablock;
        /// <summary> The allocator port if this port's outputs are sized by a meta port</summary>
        Port *                  m_pAllocatorPort;
        /// <summary> The control port controlling any gating on this port. XML serialized. </summary>
        Port *                  m_pControlPort;
        /// <summary> true if port is open. Set to value of m_bInitialPortStateOpen when XML serialized. </summary>
        BOOL                    m_bPortOpen;
        /// <summary> true if initial port state is open. XML serialized. </summary>
        BOOL                    m_bInitialPortStateOpen;
        /// <summary>   true if the block pool is active. </summary>
        BOOL                    m_bBlockPoolActive;
        /// <summary>   True if we have provided hints for block pool management
        /// 			that are not present in the template. 
        /// 			</summary>
        BOOL                    m_bPoolHintsSet;
        /// <summary>   If the m_bPoolHintsSet member is true, this member
        /// 			controls the size of the block pool. 
        /// 			</summary>
        UINT                    m_nPoolHintPoolSize;
        /// <summary>   If the m_bPoolHintsSet member is true, this member
        /// 			controls the stride of the block pool. 
        /// 			</summary>
        UINT                    m_nPoolHintStride;
        /// <summary>   If the m_bPoolHintsSet member is true, this member
        /// 			controls the data channel size of the block pool. 
        /// 			</summary>
        UINT                    m_nPoolHintDataBytes;
        /// <summary>   If the m_bPoolHintsSet member is true, this member
        /// 			controls the meta channel size of the block pool. 
        /// 			</summary>
        UINT                    m_nPoolHintMetaBytes;
        /// <summary>   If the m_bPoolHintsSet member is true, this member
        /// 			controls the template channel size of the block pool. 
        /// 			</summary>
        UINT                    m_nPoolHintTemplateBytes;
        /// <summary>   The described port, if this port is bound
        ///             as a descriptor of another output port
        ///             </summary>
        Port *                  m_pDescribedPort;
        /// <summary>   The describer function, if this port is bound
        ///             as a descriptor of another output port
        ///             </summary>
        DESCRIPTORFUNC  m_eDescriptorFunc;
        /// <summary>   true if there is a pending allocation: this occurs when an attempt to allocate an
        ///             output block is made, but descriptor ports have not yet been bound for dispatch:
        ///             when this occurs we may not know all the required sizes for metadata and template
        ///             data buffers, and so must defer the allocation until such time as we do know.
        ///             </summary>
        BOOL            m_bPendingAllocation;

        /// <summary>   The accelerators on which pending allocations
        ///             are to be done.
        ///             </summary>
        Accelerator *   m_pPendingAllocationAccelerator;

        /// <summary>   Size of the pending allocation. </summary>
        UINT            m_uiPendingAllocationSize;

        /// <summary>   True if host buffers for datablocks in this pool
        ///             should be allocated from page-locked memory
        ///             </summary>
        BOOL            m_bPageLockHostViews;

        /// <summary>   The pool high water mark. </summary>
        UINT            m_uiPoolHighWaterMark;

        /// <summary>   Number of owned blocks. </summary>
        UINT            m_uiOwnedBlockCount;

        /// <summary>   The low water mark. </summary>
        UINT            m_uiLowWaterMark;

        /// <summary>   true to explicit memory space transition point. </summary>
        BOOL            m_bExplicitMemSpaceTransitionPoint;

        /// <summary>   true to explicit memory space transition point set. </summary>
        BOOL            m_bExplicitMemSpaceTransitionPointSet;

        BOOL PoolContainsBlock(Datablock * pBlock);

        void ReleasePooledBlocks();

        friend class Port;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check that the block pool contain only datablocks with no control signals. </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CheckBlockPoolStates();

    };

};
#endif
