///-------------------------------------------------------------------------------------------------
// file:	BlockPoolOwner.h
//
// summary:	Declares the block pool owner class
///-------------------------------------------------------------------------------------------------

#ifndef __BLOCK_POOL_OWNER_H__
#define __BLOCK_POOL_OWNER_H__

#include <stdio.h>
#include <crtdbg.h>
#include <deque>
#include <vector>
#include <map>

namespace PTask {

    class Graph;
    class Datablock;
    class DatablockTemplate;

    class BlockPoolOwner {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the block pool manager. Because PTask objects are 
        ///             reference counted, it is difficult to enforce life-cycle relationships
        ///             that appear to be implied by member containment. For block pools, it
        ///             is entirely possible that user code (or internal code) keeps a reference to a datablock 
        ///             after the block pool from which it came is destroyed or deleted. Consequently,
        ///             the block pool owner pointer is not guaranteed to be valid when a block is released,
        ///             and we must keep a global list of what block pool objects are actually valid and
        ///             active to avoid attempting to return a block to a pool that has been deleted.
        ///             This method creates the data structures pertinent to maintaining that information.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void 
        InitializeBlockPoolManager(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroy the block pool manager. Because PTask objects are 
        ///             reference counted, it is difficult to enforce life-cycle relationships
        ///             that appear to be implied by member containment. For block pools, it
        ///             is entirely possible that user code (or internal code) keeps a reference to a datablock 
        ///             after the block pool from which it came is destroyed or deleted. Consequently,
        ///             the block pool owner pointer is not guaranteed to be valid when a block is released,
        ///             and we must keep a global list of what block pool objects are actually valid and
        ///             active to avoid attempting to return a block to a pool that has been deleted.
        ///             This method cleans up the data structures pertinent to maintaining that information.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void                                  
        DestroyBlockPoolManager(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Is a block pool owner pointer valid? Because PTask objects are reference counted,
        ///             it is difficult to enforce life-cycle relationships that appear to be implied by
        ///             member containment. For block pools, it is entirely possible that user code (or
        ///             internal code) keeps a reference to a datablock after the block pool from which
        ///             it came is destroyed or deleted. Consequently, the block pool owner pointer is
        ///             not guaranteed to be valid when a block is released, and we must keep a global
        ///             list of what block pool objects are actually valid and active to avoid attempting
        ///             to return a block to a pool that has been deleted.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <param name="pOwner">   [in,out] If non-null, the owner. </param>
        ///
        /// <returns>   true if a pool owner is active, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL  
        IsPoolOwnerActive(
            __in BlockPoolOwner * pOwner
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Add a new block pool owner to the global list. Because PTask objects are
        ///             reference counted, it is difficult to enforce life-cycle relationships that
        ///             appear to be implied by member containment. For block pools, it is entirely
        ///             possible that user code (or internal code) keeps a reference to a datablock after
        ///             the block pool from which it came is destroyed or deleted. Consequently, the
        ///             block pool owner pointer is not guaranteed to be valid when a block is released,
        ///             and we must keep a global list of what block pool objects are actually valid and
        ///             active to avoid attempting to return a block to a pool that has been deleted.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        /// <param name="pOwner">   [in,out] If non-null, the owner. </param>
        ///-------------------------------------------------------------------------------------------------

        static void  
        RegisterActivePoolOwner(
            __in Graph * pGraph,
            __in BlockPoolOwner * pOwner
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Retire a block pool owner from the global list. Because PTask objects are
        ///             reference counted, it is difficult to enforce life-cycle relationships that
        ///             appear to be implied by member containment. For block pools, it is entirely
        ///             possible that user code (or internal code) keeps a reference to a datablock after
        ///             the block pool from which it came is destroyed or deleted. Consequently, the
        ///             block pool owner pointer is not guaranteed to be valid when a block is released,
        ///             and we must keep a global list of what block pool objects are actually valid and
        ///             active to avoid attempting to return a block to a pool that has been deleted.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <param name="pOwner">   [in,out] If non-null, the owner. </param>
        ///-------------------------------------------------------------------------------------------------

        static void                             
        RetirePoolOwner(
            __in BlockPoolOwner * pOwner
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Retire all block pool owner from the given graph. Because PTask objects are
        ///             reference counted, it is difficult to enforce life-cycle relationships that
        ///             appear to be implied by member containment. For block pools, it is entirely
        ///             possible that user code (or internal code) keeps a reference to a datablock after
        ///             the block pool from which it came is destroyed or deleted. Consequently, the
        ///             block pool owner pointer is not guaranteed to be valid when a block is released,
        ///             and we must keep a global list of what block pool objects are actually valid and
        ///             active to avoid attempting to return a block to a pool that has been deleted.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <param name="pOwner">   [in,out] If non-null, the owner. </param>
        ///-------------------------------------------------------------------------------------------------

        static void                                  
        RetireGraph(
            __in Graph * pGraph
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   add a new block to the pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void AddNewBlock(Datablock * pBlock)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return a block to the pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void ReturnToPool(Datablock * pBlock)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the owned pool is a global pool. </summary>
        ///
        /// <remarks>   crossbac, 8/30/2013. </remarks>
        ///
        /// <returns>   true if global pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL BlockPoolIsGlobal()=0;

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
            )=0;

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
            )=0;

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
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroys the block pool. AddRef everything in the bool, set its owner
        ///             to null, and then release it. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void
        DestroyBlockPool(
            VOID
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has block pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if block pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasBlockPool()=0;

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
            )=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   The pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT
        GetPoolSize()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets request page locked. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="bPageLocked">  true to lock, false to unlock the page. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetRequestsPageLocked(BOOL bPageLocked)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets request page locked. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL GetRequestsPageLocked()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the owner name. </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the owner name. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual char * GetPoolOwnerName()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queries if a block pool is active and able to deliver/return blocks. </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <returns>   true if a block pool is active, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsBlockPoolActive()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets high water mark. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetHighWaterMark()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the total number of blocks owned by the pool. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The total number of blocks owned by the pool (whether they are queued or not). </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetOwnedBlockCount()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the low water mark. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetLowWaterMark()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the currently available count. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetAvailableBlockCount()=0;

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
            )=0;

        /// <summary>   The lock for the block pool owners. </summary>
        static CRITICAL_SECTION                      s_csBlockPoolOwners;

        /// <summary>   true if block pool owner managment is initialized. </summary>
        static LONG                                  s_bPoolOwnersInit;

        /// <summary>   The active pool owners. </summary>
        static std::map<BlockPoolOwner*, Graph*>     s_vActivePoolOwners;

        /// <summary>   The dead pool owners. </summary>
        static std::map<BlockPoolOwner*, Graph*>     s_vDeadPoolOwners;

    };

};

#endif
