//--------------------------------------------------------------------------------------
// File: MetaPort.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _META_PORT_H_
#define _META_PORT_H_

#include "primitive_types.h"
#include "port.h"

namespace PTask {

    class Channel;
    class Datablock;
    class DatablockTemplate;
    class Accelerator;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Meta port. A meta port is a port that consumes datablocks but does not bind them
    ///             to Task inputs. Rather, the runtime uses the contained information to perform
    ///             operations on behalf of the Task for which the MetaPort is an input.  Currently,
    ///             the only operation of this class is allocation of Datablocks on OutputPorts,
    ///             although the mechanism will be generalized in the future. A MetaPort consumes a
    ///             datablock, expecting it to contain a single integer value, which is the
    ///             interpreted as the allocation size for the OutputPort specified in the
    ///             m_pAllocatorPort member.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class MetaPort : public Port {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        MetaPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~MetaPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is occupied. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   true if occupied, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			IsOccupied();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pulls the next datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
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
        /// <summary>   Pushes an object into this port.  </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pDatablock">   [in,out] If non-null, the Datablock* to push. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			Push(Datablock* pDatablock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind control channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			BindControlChannel(Channel * pChannel);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind control channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void			UnbindControlChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the allocation port. This port must be an output port, and is the port on
        ///             which a new datablock will be allocated when a block is consumed from this
        ///             MetaPort.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetAllocationPort(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the allocation port. This port must be an output port, and is the port on
        ///             which a new datablock will be allocated when a block is consumed from this
        ///             MetaPort.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the allocation port. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port *          GetAllocationPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an iteration target to the list. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            BindIterationTarget(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Configure iteration targets. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ConfigureIterationTargets(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets an integer value from a block consumed from this MetaPort. Should not be
        ///             called when the port is unoccupied because it will block on a Pull call. On exit,
        ///             bControlBlock is TRUE if the consumed block carried a control signal;
        ///             uiControlCode will be set accordingly if this is the case. The integer value can
        ///             be used by iteration control or output allocation meta functions.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="bControlBlock">    [out] True on exit if the block pulled to compute the
        ///                                 allocation size carried a control signal. </param>
        /// <param name="luiControlSignal"> [out] If the block pulled to compute the allocation size
        ///                                 carried a control signal, the control code from that block. </param>
        ///
        /// <returns>   The integer value at offset 0 in the datablock's data channel. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT            
        GetIntegerValue(
            BOOL &bControlBlock, 
            CONTROLSIGNAL &luiControlSignal
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a new MetaPort. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pDatablockTemplate">   [in] If non-null, the datablock template. </param>
        /// <param name="uiUniqueIdentifier">   Unique identifier (caller-supplied, uniqueness not
        ///                                     enforced). </param>
        /// <param name="lpszVariableBinding">  [in] If non-null, the variable binding. </param>
        /// <param name="nBoundParameterIndex"> Zero-based index of the n bound parameter. </param>
        /// <param name="nInOutRouteIdx">       Zero-based index of the n in out route. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Port *			
        Create(
            __in DatablockTemplate * pDatablockTemplate, 
            __in UINT                uiUniqueIdentifier, 
            __in char *              lpszVariableBinding, 
            __in int                 nBoundParameterIndex, 
            __in int                 nInOutRouteIdx
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

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a destination buffer occupying this output port. Meaningless for MetaPorts,
        ///             but required by the abstract superclass Port.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in] If non-null, an accelerator object to assist
        ///                             creating a datablock if none is available. </param>
        ///
        /// <returns>   null if it fails, else the destination buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		GetDestinationBuffer(Accelerator * pAccelerator=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a destination buffer. Meaningless for MetaPorts, but required by the
        ///             abstract superclass Port.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			SetDestinationBuffer(Datablock * p);

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
        /// <summary>   Sets a meta function. </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///
        /// <param name="eMetaFunctionSpecifier">   Information describing the meta function. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetMetaFunction(METAFUNCTION eMetaFunctionSpecifier);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the meta function. </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///
        /// <returns>   The meta function. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual METAFUNCTION    GetMetaFunction();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform the work associated with this port's meta function. For example, if the
        ///             port is an allocator, allocate a block for the downstream output port. If it is
        ///             an iterator, set the iteration count on the Task.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void
        PerformMetaFunction(
            __in Accelerator * pDispatchAccelerator
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform any post-dispatch work associated with this port's meta function. For
        /// 			example, if the port is an iteration construct, reset the loop bounds and 
        /// 			propagate any control signals associated with the iteration. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            FinalizeMetaFunction(Accelerator * pDispatchAccelerator);

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
        /// <summary>   Searches for collaborating meta ports: if this port is an allocator
        ///             for output ports with descriptor ports, block allocation may have 
        ///             dependences on other meta ports for the bound task. We need to know this
        ///             at dispatch time, but it is a static property of the graph, so
        ///             we pre-compute it as a side-effect of OnGraphComplete(). 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 2/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void FindCollaboratingMetaPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets an allocation hint. </summary>
        ///
        /// <remarks>   crossbac, 8/21/2013. </remarks>
        ///
        /// <param name="uiAllocationHint"> The allocation hint. </param>
        /// <param name="bForceAllocHint">  true to force allocate hint. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        SetAllocationHint(
            __in UINT uiAllocationHint,
            __in BOOL bForceAllocHint
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port has been configured with a statically known allocation size. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 8/21/2013. </remarks>
        ///
        /// <returns>   true if static allocation size, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsStaticAllocationSize();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the channel allocation size when this meta port is an allocator for an
        ///             output port with descriptor ports (meaning another meta port is responsible for
        ///             computing that allocation size). If this meta port is not involved in such a
        ///             graph structure, return 0.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 2/15/2013. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        /// <param name="eFunc">                The function. </param>
        /// <param name="ppPortTemplate">       [out] on exit the template for the related collaborative
        ///                                     port, if one is available. These are needed when initial
        ///                                     values are supplied by the template. </param>
        ///
        /// <returns>   The meta buffer allocation size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetCollaborativeAllocationSize(
            __in  Accelerator *        pDispatchAccelerator, 
            __in  DESCRIPTORFUNC       eFunc,
            __out DatablockTemplate ** ppPortTemplate
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalize collaborative allocations. If this port has completed a collaborative
        ///             allocation (where other meta ports determine meta/template channel sizes)
        ///             we need to finish the binding of an output block at those ports. </summary>
        ///
        /// <remarks>   Crossbac, 2/15/2013. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in] non-null, the dispatch accelerator. </param>
        /// <param name="pBlock">               [in,out] non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void FinalizeCollaborativeAllocations(
            __in    Accelerator * pDispatchAccelerator, 
            __inout Datablock *   pBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Perform allocation.  In this case, a datablock on a metaport provides an integer-
        ///             valued allocation size for another output port on the ptask. Hence, this function
        ///             looks at all metaports, and performs output datablock allocation as needed.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void  
        PerformAllocation(
            __in Accelerator * pDispatchAccelerator
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Configure simple iteration. Simple iteration is distinguished from general
        ///             iteration because it involves iterative invocation of a single PTask node. The
        ///             mechanisms required to build this are so much simpler than those required to
        ///             build general iteration over arbitrary subgraphs that it is worth bothering to
        ///             distinguish the case. Here, the datablock recieved on this port contains an
        ///             integer-valued iteration count, which we set on the task directly. Task::Dispatch
        ///             is responsible for clearing the iteration count after dispatch.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    ConfigureSimpleIteration();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Configure general iteration. </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void                    ConfigureGeneralIteration();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalize general iteration. (Update iteration state after task dispatch,
        /// 			and propagate control signals where appropriate). </summary>
        ///
        /// <remarks>   Crossbac, 1/10/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void                    FinalizeGeneralIteration();

        /// <summary> The allocation port.  This port must be an output port, and is the port on
        ///           which a new datablock will be allocated when a block is consumed from this
        ///           MetaPort.
        ///           </summary>
        Port *                  m_pAllocationPort;

        /// <summary> The meta function </summary>
        METAFUNCTION            m_eMetaFunction;

        /// <summary> The general iteration block </summary>
        Datablock *             m_pGeneralIterationBlock;

        /// <summary> Number of general iterations </summary>
        UINT                    m_nGeneralIterationCount;

        /// <summary> The general iteration maximum </summary>
        UINT                    m_nGeneralIterationMax;

        /// <summary>   if this object is collaborative allocator and another meta port is responsible
        ///             for computing the allocation size of the metadata buffer channel on the block
        ///             allocated by *this* meta-port, we keep a pointer to that other port. Since
        ///             deciding requires traversing part of the graph structure, we set this once so we
        ///             don't have to do it again.
        ///             </summary>
        Port *          m_pCollaborativeMetaAllocator;

        /// <summary>   if this object is collaborative allocator and another meta port is responsible
        ///             for computing the allocation size of the template buffer channel on the block
        ///             allocated by *this* meta-port, we keep a pointer to that other port. Since
        ///             deciding requires traversing part of the graph structure, we set this once so we
        ///             don't have to do it again.
        ///             </summary>
        Port *          m_pCollaborativeTemplateAllocator;

        /// <summary>   An allocation size hint. </summary>
        UINT                    m_uiAllocHint;

        /// <summary>   true if the allocation hint takes precedence over the value
        ///             received on the incoming channel for this port. </summary>
        BOOL                    m_bForceAllocHint;

    };

};
#endif
