///-------------------------------------------------------------------------------------------------
// file:	BlockPool.h
//
// summary:	Declares the block pool class
///-------------------------------------------------------------------------------------------------

#ifndef __BLOCK_POOL_H__
#define __BLOCK_POOL_H__

#include <stdio.h>
#include <crtdbg.h>

#include "datablocktemplate.h"
#include "channel.h"
#include "port.h"
#include "PBuffer.h"
#include <deque>
#include <vector>
#include "BlockPoolOwner.h"

namespace PTask {

    class BlockPool : public Lockable {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        BlockPool(
            __in DatablockTemplate * pTemplate,
            __in BUFFERACCESSFLAGS   ePermissions,
            __in UINT                uiPoolSize,
            __in BlockPoolOwner *    pPoolOwner
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~BlockPool();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets request page locked. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="bPageLocked">  true to lock, false to unlock the page. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetRequestsPageLocked(BOOL bPageLocked);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets request page locked. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL GetRequestsPageLocked();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a growable. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="bGrowable">    true if growable. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetGrowable(BOOL bGrowable);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is growable. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if growable, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsGrowable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets eager device materialize. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="bEager">   true to eager. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetEagerDeviceMaterialize(BOOL bEager);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets eager device materialize. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL GetEagerDeviceMaterialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="uiSize">   The size. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetPoolSize(UINT uiSize);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   The pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetPoolSize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a view memory space. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void AddViewMemorySpace(Accelerator* pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a view memory space. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void AddViewMemorySpace(UINT uiMemorySpace);

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
        GetPooledBlock(
            __in Accelerator * pAccelerator=NULL,
            __in UINT uiDataBytes=0,
            __in UINT uiMetaBytes=0,
            __in UINT uiTemplateBytes=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds to the pool. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            AddNewBlock(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return to the pool. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ReturnBlock(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has block pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if block pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasBlockPool();

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
        /// <param name="nPoolSize">            Size of the block pool. </param>
        /// <param name="nStride">              The stride. </param>
        /// <param name="nDataBytes">           The data in bytes. </param>
        /// <param name="nMetaBytes">           The meta in bytes. </param>
        /// <param name="nTemplateBytes">       The template in bytes. </param>
        /// <param name="bPageLockHostViews">   (optional) the page lock host views. </param>
        /// <param name="bEagerMaterialize">    (optional) the eager materialize. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            
        ForceBlockPoolHint(
            __in UINT nPoolSize,
            __in UINT nStride,
            __in UINT nDataBytes,
            __in UINT nMetaBytes,
            __in UINT nTemplateBytes,
            __in BOOL bPageLockHostViews=FALSE,
            __in BOOL bEagerMaterialize=FALSE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
        ///             Synchronous version: allocates buffers and populates any device side
        ///             views in one go. If graph construction performance matters, this is
        ///             not a good way to do it, since memory allocation causes synchronization. 
        ///             The asynchronous variant does it in several passes, allowing us
        ///             to overlap the copy. 
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
        /// <summary>   Query if this object is enabled. </summary>
        ///
        /// <remarks>   crossbac, 6/18/2013. </remarks>
        ///
        /// <returns>   true if enabled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsEnabled();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets high water mark. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetHighWaterMark();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the low water mark. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetLowWaterMark();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the currently available count. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetAvailableBlockCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the total number of blocks owned by the pool. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2013. </remarks>
        ///
        /// <returns>   The high water mark. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetOwnedBlockCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the number of blocks by which the pool should grow if 
        ///             it grows in response to dynamic demand. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/20/2013. </remarks>
        ///
        /// <param name="uiBlockCount"> Number of blocks. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetGrowIncrement(UINT uiBlockCount);

        UINT GetGrowIncrement();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate a block based on the hint size (rather than the template!). 
        ///             We do not support an async variant of this yet.
        ///             </summary>
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

        Datablock *
        AllocateBlockWithPoolHint(
            __in UINT uiDataBytes,
            __in UINT uiMetaBytes,
            __in UINT uiTemplateBytes
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

        Datablock *		
        AllocateBlockForPool(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate a block as part of asynchronous pool construction. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <param name="bFinalized">   [in,out] The finalized. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock *	
        AllocateBlockForPoolAsync(
            __out BOOL &bFinalized
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalize a block allocated with the async variant. Basically
        ///             we need to populate any views on this pass.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void FinalizeBlock(
            __in Datablock * pBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Grows the pool by the given number of blocks. </summary>
        ///
        /// <remarks>   crossbac, 6/20/2013. </remarks>
        ///
        /// <param name="uiBlockCount"> Number of blocks. </param>
        ///-------------------------------------------------------------------------------------------------

        void Grow(UINT uiBlockCount);

        /// <summary>   The template. </summary>
        DatablockTemplate * m_pTemplate;
        /// <summary> Size of the maximum block pool </summary>
        int                     m_nMaxPoolSize;
        /// <summary> The block pool </summary>
        std::deque<Datablock*>  m_pBlockPool;
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
        /// <summary>   True if host buffers for datablocks in this pool
        ///             should be allocated from page-locked memory
        ///             </summary>
        BOOL                    m_bPageLockHostViews;
        /// <summary>   true to eager device materialize. </summary>
        BOOL                    m_bEagerDeviceMaterialize;
        /// <summary>   The memory spaces in which these blocks can reasonably
        ///             require a view. </summary>
        std::set<Accelerator*>   m_vAccelerators;
        /// <summary>   The permissions for blocks in this pool. </summary>
        BUFFERACCESSFLAGS       m_ePermissions;
        /// <summary>   true if growable. </summary>
        BOOL                    m_bGrowable;
        /// <summary>   true if this object has initial value. </summary>
        BOOL                    m_bHasInitialValue;
        /// <summary>   The initial value. </summary>
        HOSTMEMORYEXTENT        m_vInitialValue;
        /// <summary>   The owner of the pool. </summary>
        BlockPoolOwner *        m_pPoolOwner;
        /// <summary>   blocks allocated with async variant that require finalization. </summary>
        std::vector<Datablock*>    m_vOutstandingBlocks;
        /// <summary>   The dirty. </summary>
        std::set<Datablock*>    m_vDirty;
        /// <summary>   The block count high water mark. </summary>
        UINT                    m_uiHighWaterMark;
        /// <summary>   The block count low water mark. </summary>
        UINT                    m_uiLowWaterMark;
        /// <summary>   The owned blocks. </summary>
        UINT                    m_uiOwnedBlocks;
        /// <summary>   The grow increment. </summary>
        UINT                    m_uiGrowIncrement;
        /// <summary>   true to enable, false to disable. </summary>
        BOOL                    m_bEnabled;

        BOOL Contains(Datablock * pBlock);
        void ReleaseBlocks();
        void LockTargetAccelerators();
        void UnlockTargetAccelerators();

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
