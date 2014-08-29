//--------------------------------------------------------------------------------------
// File: StickyPort.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _STICKY_PORT_H_
#define _STICKY_PORT_H_

#include "primitive_types.h"
#include "port.h"
#include <vector>
#include <sstream>

namespace PTask {

    class Channel;
    class Datablock;
    class Accelerator;
    class DatablockTemplate;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sticky port. A port that is bound to scalar values in kernel code, with read-only
    ///             semantics. Typically these values are bound to constant memory on a device where
    ///             specialized memories are available. A sticky port also retains its last value: if
    ///             no new datablock is available on its incoming channel it will redeliver the last
    ///             datablock pulled from it on the next call to Pull.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class StickyPort : public Port {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        StickyPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~StickyPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is occupied. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <returns>   true if occupied, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			IsOccupied();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pulls the next datablock from this port. Return the last datablock if no new
        ///             block is available on the incoming channel.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		Pull();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Returns the datablock that would be returned by the next call to Pull, without
        ///             removing it from the port.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the current top-of-stack object. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		Peek();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   The push method is required by the abstract class Port, but has no meaning for
        ///             sticky ports. This method is a no-op for StickyPort.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			Push(Datablock* p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the destination datablock for this port. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   null if it fails, else the destination buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		GetDestinationBuffer(Accelerator * pAccelerator=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a destination buffer. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name=""> [in,out] If non-null, the. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual	void			SetDestinationBuffer(Datablock *);

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
        /// <summary>   Creates this object. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
        /// <param name="uiId">                 The identifier. </param>
        /// <param name="lpszVariableBinding">  [in,out] If non-null, the variable binding. </param>
        /// <param name="nParmIdx">             Zero-based index of the n parm. </param>
        /// <param name="nInOutRouteIdx">       Zero-based index of the n in out route. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Port *			Create(DatablockTemplate * pTemplate, 
                                       UINT uiId, 
                                       char * lpszVariableBinding, 
                                       int nParmIdx, 
                                       int nInOutRouteIdx
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
        
        /// <summary> The sticky datablock </summary>
        Datablock *				m_pStickyDatablock;
    };

};
#endif
