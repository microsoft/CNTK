//--------------------------------------------------------------------------------------
// File: InitializerPort.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _INITIALIZER_PORT_H_
#define _INITIALIZER_PORT_H_

#include "primitive_types.h"
#include "InputPort.h"

namespace PTask {

    class Task;
    class Graph;
    class Datablock;
    class DatablockTemplate;
    class AsyncContext;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Initializer port. An initializer port is always full, and when pulled, will return a new
    ///     datablock with the value derived from the datablock template with which this port was
    ///     created. When peeked, an initializer port always returns NULL.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class InitializerPort : public InputPort {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        InitializerPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~InitializerPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is occupied. Initializer ports are always occupied, by
        ///             definition.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if occupied, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			IsOccupied();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pulls on this port to read the next available datablock. Return a new datablock
        ///             initialized according to the DatablockTemplate with which this port was created.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		Pull();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Peek at the next datablock on this port. Peek on an InitializerPort always
        ///             returns NULL, because datablocks are created on demand in response to a pull.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the current top-of-stack object. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		Peek();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pushes an object into this port. This is a NO-OP for InitializerPorts, since this
        ///             port type is designed only to be bound to input resources.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			Push(Datablock* p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate a datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAsyncContext">    [in] (optional)  If non-null, the async context where the
        ///                                 block will be first used. </param>
        /// <param name="bPooled">          [in] true to pooled. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		
        AllocateBlock(
            __in AsyncContext * pAsyncContext, 
            __in BOOL bPooled
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a new InitializerPort. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
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
        /// <summary>   Check type-specific semantics. Return true if all the structures are initialized
        ///             for this port in a way that is consistent with a well-formed graph. Called by
        ///             CheckSemantics()
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="pos">      [in,out] output string stream. </param>
        /// <param name="pGraph">   [in] non-null, the graph. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            
        CheckTypeSpecificSemantics(
            __inout std::ostream * pos,
            __in    PTask::Graph * pGraph
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

    };

};
#endif
