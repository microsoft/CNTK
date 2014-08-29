///-------------------------------------------------------------------------------------------------
// file:	GlobalBlockPool.h
//
// summary:	Declares the global block pool class
///-------------------------------------------------------------------------------------------------

#ifndef __GLOBAL_BLOCK_POOL_H__
#define __GLOBAL_BLOCK_POOL_H__

#include <stdio.h>
#include <crtdbg.h>
#include <Windows.h>
#include "datablock.h"
#include "BlockPoolOwner.h"
#include "BlockPool.h"
#include <deque>

namespace PTask {

    class GlobalBlockPool : public BlockPoolOwner, public Lockable {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pDatablockTemplate">	[in,out] If non-null, the p. </param>
        /// <param name="eAcceleratorClass"> 	acc class. </param>
        ///-------------------------------------------------------------------------------------------------

        GlobalBlockPool(
            __in DatablockTemplate * pDatablockTemplate,
			__in ACCELERATOR_CLASS eAcceleratorClass,
			__in BUFFERACCESSFLAGS ePermissions
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pDatablockTemplate">	[in,out] If non-null, the p. </param>
        /// <param name="eAcceleratorClass"> 	acc class. </param>
        ///-------------------------------------------------------------------------------------------------

        GlobalBlockPool(
			__in UINT nDataBytes,
			__in UINT nMetaBytes,
			__in UINT nTemplateBytes,
			__in ACCELERATOR_CLASS eAcceleratorClass,
			__in BUFFERACCESSFLAGS ePermissions
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~GlobalBlockPool();

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
        /// <summary>   Configure a block pool, but do not perform allocations on it yet. 
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
        ConfigureBlockPool(
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
        /// <summary>   Gets pooled block. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the pooled block. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock * GetPooledBlock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this request matches what is present in this pool. </summary>
        ///
        /// <remarks>   crossbac, 8/14/2013. </remarks>
        ///
        /// <param name="nDataBytes">       The data in bytes. </param>
        /// <param name="nMetaBytes">       The meta in bytes. </param>
        /// <param name="nTemplateBytes">   The template in bytes. </param>
        ///
        /// <returns>   true if matching request, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

		virtual BOOL 
		IsMatchingRequest(			
			__in UINT nDataBytes,
			__in UINT nMetaBytes,
			__in UINT nTemplateBytes
			);

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

    protected:

        /// <summary>   The block pool. </summary>
        BlockPool *             m_pBlockPool;

        /// <summary>   true if this object has block pool. </summary>
        BOOL                    m_bHasBlockPool;

		/// <summary>	The accelerator class. </summary>
		ACCELERATOR_CLASS		m_eAcceleratorClass;

		/// <summary>	The permissions. </summary>
		BUFFERACCESSFLAGS		m_ePermissions;

		/// <summary>	The template. </summary>
		DatablockTemplate *		m_pTemplate;

		/// <summary>	The data in bytes. </summary>
		UINT					m_nDataBytes;

		/// <summary>	The meta in bytes. </summary>
        UINT					m_nMetaBytes;
        
		/// <summary>	The template in bytes. </summary>
        UINT					m_nTemplateBytes;
    };

};
#endif