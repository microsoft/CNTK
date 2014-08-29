//--------------------------------------------------------------------------------------
// File: port.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PORT_H_
#define _PORT_H_

#include <stdio.h>
#include <crtdbg.h>
#include "AsyncContext.h"
#include "datablocktemplate.h"
#include "channel.h"
#include "PBuffer.h"
#include "Lockable.h"
#include "accelerator.h"
#include <vector>
#include <set>
#include "Datablock.h"
#include "BlockPool.h"
#include "BlockPoolOwner.h"

namespace PTask {

    class Task;
    class Graph;
    class Port;
    class Channel;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deferred port descriptor record. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct defportrec_t {
        Port * pPort;
        DESCRIPTORFUNC func;
    } DEFERREDPORTDESC;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deferred channel descriptor record. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct defchanrec_t {
        Channel * pChannel;
        DESCRIPTORFUNC func;
    } DEFERREDCHANNELDESC;

    /// <summary> default kernel parameter index: invalid when unspecified. </summary>
    static const int PT_DEFAULT_KERNEL_PARM_IDX = -1;

    /// <summary> default in/out routing index: invalid when unspecified. </summary>
    static const int PT_DEFAULT_INOUT_ROUTING_IDX = -1;

    /// <summary> Unbound port code</summary>
    static const UINT UNBOUND_PORT = 0xFFFFFFFF;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Port class. Ports provide a way to expose variables in device side code (as well
    ///             as runtime parameters that are only available dynamically).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class Port : public Lockable, public BlockPoolOwner {

        friend class XMLWriter;
        friend class XMLReader;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        Port();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Port();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes a port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pTemplate">        [in] If non-null, the datablock template. </param>
        /// <param name="uiId">             An identifier for the port (programmer-supplied). </param>
        /// <param name="lpszBinding">      [in] If non-null, the variable binding. </param>
        /// <param name="nParmIndex">       Zero-based index of the parameter this port binds. </param>
        /// <param name="nInOutRouting">    The index of the out param for in/out routing. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual HRESULT			
        Initialize(
            __in DatablockTemplate * pTemplate, 
            __in UINT uiId, 
            __in char * lpszBinding, 
            __in int nParmIndex, 
            __in int nInOutRouting
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets this object. </summary>
        ///
        /// <remarks>   Crossbac, 5/2/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void Reset();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the uid. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The uid. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual const UINT			GetUID();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is bound to a read-only input. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2014. </remarks>
        ///
        /// <returns>   true if constant semantics, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsConstantSemantics();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is input parameter. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2014. </remarks>
        ///
        /// <returns>   true if input parameter, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsInputParameter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is output parameter. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2014. </remarks>
        ///
        /// <returns>   true if output parameter, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsOutputParameter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is initializer parameter. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2014. </remarks>
        ///
        /// <returns>   true if initializer parameter, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsInitializerParameter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is occupied. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if occupied, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			IsOccupied()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind a channel to this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pChannel"> [in] non-null, the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			BindChannel(Channel * pChannel);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind a task to this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pTask">        [in] non-null, the task. </param>
        /// <param name="nPortIndex">   Zero-based index of the port/parameter to bind. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			BindTask(Task* pTask, UINT nPortIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind a channel from this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			UnbindChannel(int nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind a control channel to this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pChannel"> [in] non-null, the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			BindControlChannel(Channel * pChannel);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind a control channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void			UnbindControlChannel();

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
        /// <summary>   Unbind descriptor ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void			UnbindDescriptorPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind a deferred channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pChannel"> [in] non-null, the channel. </param>
        /// <param name="func">     The func. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			BindDeferredChannel(Channel * pChannel, DESCRIPTORFUNC func);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind deferred channels. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void			UnbindDeferredChannels();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind a task from this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void			UnbindTask();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is dispatch dimensions hint. </summary>
        ///
        /// <remarks>   Crossbac, 1/16/2014. </remarks>
        ///
        /// <returns>   true if dispatch dimensions hint, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsDispatchDimensionsHint();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets dispatch dimensions hint. </summary>
        ///
        /// <remarks>   Crossbac, 1/16/2014. </remarks>
        ///
        /// <param name="bIsHintSource">    true if this object is hint source. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetDispatchDimensionsHint(BOOL bIsHintSource);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Returns the datablock occupying this port without removing it. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the current block. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *     Peek()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pulls a datablock from this port, potentially blocking until one becomes
        ///             available.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		Pull()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pushes a datablock into this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL			Push(Datablock* p)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is bound as descriptor port. </summary>
        ///
        /// <remarks>   crossbac, 4/19/2012. </remarks>
        ///
        /// <returns>   true if descriptor port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsDescriptorPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the port type. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The port type. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PORTTYPE		GetPortType();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the super-set of all "control signals of interest" for this graph object.  
        ///             A control signal is "of interest" if the behavior of this object is is predicated
        ///             in some way by the presence or absence of a given signal. This function returns
        ///             the bit-wise OR of all such signals.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <returns>   The bitwise OR of all found control signals of interest. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual CONTROLSIGNAL GetControlSignalsOfInterest();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a destination buffer occupying this output port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in] If non-null, an accelerator object to assist
        ///                             creating a datablock if none is available. </param>
        ///
        /// <returns>   null if it fails, else the destination buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *		GetDestinationBuffer(Accelerator * pAccelerator=NULL)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has block pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if block pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasBlockPool()=0;

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
            )=0;

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
        /// <summary>   Gets pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   The pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetPoolSize()=0;

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
            )=0;

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
        /// <summary>   Sets a destination buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void			SetDestinationBuffer(Datablock * p)=0;

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
        /// <summary>   Gets the task to which this port is bound </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the task. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Task *	GetTask();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the channel bound to this port at the given index. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        ///
        /// <returns>   null if it fails, else the channel. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Channel *		GetChannel(UINT nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the channel count. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <returns>   The channel count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT            GetChannelCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the control channel bound to this port at the given index. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        ///
        /// <returns>   null if it fails, else the channel. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Channel *		GetControlChannel(UINT nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the control channel count. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <returns>   The channel count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT            GetControlChannelCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the datablock template describing this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the template. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual DatablockTemplate * GetTemplate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the name of the ptask variable to which this port is bound. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the variable binding. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual const char *	GetVariableBinding();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is bound to a ptask formal parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if formal parameter, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsFormalParameter(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is bound to a ptask scalar parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if scalar parameter, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsScalarParameter(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the index of this port in the array it was in when passed 
        ///             into AddTask. </summary>
        ///
        /// <remarks>   jcurrey, 5/6/2013. </remarks>
        ///
        /// <param name="index">   The original index. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetOriginalIndex(UINT index);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the index of this port in the array it was in when passed 
        ///             into AddTask. </summary>
        ///
        /// <remarks>   jcurrey, 5/6/2013. </remarks>
        ///
        /// <returns>   The original index. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT            GetOriginalIndex();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the formal parameter index of this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The formal parameter index. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual int             GetFormalParameterIndex();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is part of an in/out parameter pair. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if in out parameter, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsInOutParameter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the in out routing index for an in/out port pair. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The in out routing index. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual int             GetInOutRoutingIndex();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the parameter type for this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The parameter type. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTASK_PARM_TYPE GetParameterType();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the list of deferred channels associated with this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the deferred channels. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual std::vector<DEFERREDCHANNELDESC*>* GetDeferredChannels();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a gated port to this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    [in] non-null, a the gated port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            AddGatedPort(Port * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a control propagation port to this port </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    [in] non-null, a the control propagation port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            AddControlPropagationPort(Port * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a control propagation port to the given Channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pChannel"> [in] non-null, a the control propagation channel. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            AddControlPropagationChannel(Channel * pChannel);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a control propagation source for this port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    [in] non-null, a the port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetControlPropagationSource(Port * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the control propagation source for this port </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the control propagation source. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port*           GetControlPropagationSource();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port has gated ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if gated ports, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasGatedPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Raises a Gated ports signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SignalGatedPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Propagate a control code from this port to output ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiCode">   The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            PropagateControlSignal(CONTROLSIGNAL uiCode);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a propagated control code signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiCode">   The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetPropagatedControlSignal(CONTROLSIGNAL uiCode);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an initial propagated control code signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiCode">   The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetInitialPropagatedControlSignal(CONTROLSIGNAL uiCode);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a propagated control code signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiCode">   The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ClearPropagatedControlSignal(CONTROLSIGNAL uiCode);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a propagated control code signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiCode">   The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            ClearAllPropagatedControlSignals();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the propagated control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The propagated control code. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual CONTROLSIGNAL            GetPropagatedControlSignals();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Mark this port as Stick/not-Sticky. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="bSticky">  true to sticky. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetSticky(BOOL bSticky);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a control signal which allows a sticky port (either of class "StickyPort",
        ///             or another class with the sticky property set) to release its sticky datablock
        ///             safely. Being able to release such blocks without cleaning up the entire graph
        ///             is an important memory optimization for some Dandelion workloads running
        ///             at scale. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <param name="luiControlSignal"> The lui control signal. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetStickyReleaseSignal(CONTROLSIGNAL luiControlSignal);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has a sticky release signal configured. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <returns>   true if sticky release signal, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            HasStickyReleaseSignal();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is sticky. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if sticky, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsSticky();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Mark this port as Destructive/not-Destructive. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="bSticky">  true to mark the port as destructive. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetDestructive(BOOL bDestructive);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port is destructive, meaning it writes datablocks even though it is
        ///             an input port without a bound inout consumer.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if destructive, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsDestructive();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the port is (or can be) connected to a data source or sink that can be
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
        /// <summary>   Sets whether the port is (or can be) connected to a data source or sink that can be
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
        /// <param name="bIsMarshallable">  true if port is marshallable. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetCanStream(BOOL bCanStream);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this port works with Marshallable output. Unmarshallable data is any
        ///             that contains pointers to dynamically allocated device-side buffers, which are
        ///             (by construction) invisible to ptask. For example, a hash-table cannot be
        ///             migrated because pointers will be invalid on another device, and no facility
        ///             exists to marshal the hashtable by chasing the pointers and flattening the data
        ///             structure. If PTask does not know that a datablock is unmarshallable, migration
        ///             will cause havoc.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if marshallable output, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsMarshallable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Marks this port as producing output that is marshallable. Unmarshallable data is
        ///             any that contains pointers to dynamically allocated device-side buffers, which
        ///             are (by construction) invisible to ptask. For example, a hash-table cannot be
        ///             migrated because pointers will be invalid on another device, and no facility
        ///             exists to marshal the hashtable by chasing the pointers and flattening the data
        ///             structure. If PTask does not know that a datablock is unmarshallable, migration
        ///             will cause havoc.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="bIsMarshallable">  true if port is marshallable. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetMarshallable(BOOL bIsMarshallable);

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
        /// <summary>   Sets an iteration source. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetIterationSource(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an iteration target to the list. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <param name="pPort">    [in,out] If non-null, the port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            BindIterationTarget(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind dependent accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 5/16/2012. </remarks>
        ///
        /// <param name="accClass"> The acc class. </param>
        /// <param name="nIndex">   Zero-based index of the accelerator in the dependent accelerator
        ///                         list. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            BindDependentAccelerator(ACCELERATOR_CLASS accClass, int nIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind dependent accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 5/16/2012. </remarks>
        ///
        /// <param name="accClass"> The acc class. </param>
        /// <param name="nIndex">   Zero-based index of the accelerator in the dependent accelerator
        ///                         list. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            
        SetDependentAffinity(Accelerator* pAccelerator, AFFINITYTYPE affinityType);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets task-accelerator affinity. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="vAccelerators">    [in,out] non-null, the accelerators. </param>
        /// <param name="pvAffinityTypes">  [in,out] List of types of affinities. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        SetDependentAffinity(
            std::vector<Accelerator*> &vAccelerators, 
            std::vector<AFFINITYTYPE> &pvAffinityTypes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has dependent accelerator binding. </summary>
        ///
        /// <remarks>   Crossbac, 5/16/2012. </remarks>
        ///
        /// <returns>   true if dependent accelerator binding, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL              HasDependentAcceleratorBinding();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a dependent accelerator class. </summary>
        ///
        /// <remarks>   Crossbac, 5/16/2012. </remarks>
        ///
        /// <param name="nIndex">   Zero-based index of the accelerator in the dependent accelerator
        ///                         list. </param>
        ///
        /// <returns>   The dependent accelerator class. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual ACCELERATOR_CLASS GetDependentAcceleratorClass(int nIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the dependent accelerator index. </summary>
        ///
        /// <remarks>   Crossbac, 5/16/2012. </remarks>
        ///
        /// <returns>   The dependent accelerator index. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual int               GetDependentAcceleratorIndex();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the iteration source. </summary>
        ///
        /// <remarks>   Crossbac, 2/28/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the iteration source. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port *          GetIterationSource();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the suppress clones property, which allows a user to suppress output cloning
        ///             for blocks on ports with multiple (R/W conflicting) downstream consumers, if the
        ///             programmer happens to know something about the structure of the graph that the
        ///             runtime cannot (or does not detect) and that makes it safe to do so.
        ///             </summary>
        ///
        /// <remarks>   Subclasses must implement this method, since it is not meaningful for all port
        ///             types. Crossbac, 2/29/2012.
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
        /// <remarks>   Subclasses must implement this method, since it is not meaningful for all port
        ///             types. Crossbac, 2/29/2012.
        ///             </remarks>
        ///
        /// <returns>   the value of the suppress clones property. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            GetSuppressClones();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind this port to a particular dimension for geometry estimation. </summary>
        ///
        /// <remarks>   crossbac, 5/1/2012. </remarks>
        ///
        /// <param name="eGeoDimension">    The geo dimension. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            BindToEstimatorDimension(GEOMETRYESTIMATORDIMENSION eGeoDimension);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the estimator dimension binding. </summary>
        ///
        /// <remarks>   crossbac, 5/1/2012. </remarks>
        ///
        /// <returns>   The estimator dimension binding. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual GEOMETRYESTIMATORDIMENSION GetEstimatorDimensionBinding();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check semantics. Return true if all the structures are initialized for this port
        ///             in a way that is consistent with a well-formed graph.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="pos">      [in,out] output string stream. </param>
        /// <param name="pGraph">   [in,out] non-null, the graph. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            CheckSemantics(std::ostream * pos, PTask::Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a trigger port. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        /// <param name="bTrigger"> true to trigger. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetTriggerPort(Graph * pGraph, BOOL bTrigger);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is trigger port. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <returns>   true if trigger port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsTriggerPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Handle trigger. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="uiCode">   The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            HandleTriggers(CONTROLSIGNAL uiCode);

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
        /// <summary>   Query if this object is the scope terminus for a subgraph, and the given
        ///             scope terminal control signal.
        ///              </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <returns>   true if scope terminus port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsScopeTerminus(CONTROLSIGNAL luiControlSignal);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets mandatory dependent accelerator if one has been specified. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the mandatory dependent accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * GetMandatoryDependentAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the dependent affinities map. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the dependent affinities. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::map<Accelerator*, AFFINITYTYPE> * GetDependentAffinities();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets upstream channel pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="uiPoolSize">       Size of the pool. </param>
        /// <param name="bGrowable">        The growable. </param>
        /// <param name="uiGrowIncrement">  The grow increment. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetUpstreamChannelPool(
            __in UINT uiPoolSize, 
            __in BOOL bGrowable=FALSE,
            __in UINT uiGrowIncrement=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has upstream channel pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if upstream channel pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasUpstreamChannelPool();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has a growable upstream channel pool. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if upstream channel pool, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsUpstreamChannelPoolGrowable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets upstream channel pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   The upstream channel pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetUpstreamChannelPoolSize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the upstream channel pool grow increment. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   The upstream channel pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetUpstreamChannelPoolGrowIncrement();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assemble channel lock set. </summary>
        ///
        /// <remarks>   crossbac, 6/21/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void AssembleChannelLockSet();

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
        /// <summary>   Query if 'pAccelerator' is a block pool view accelerator:
        ///             this means that blocks in the pool should eagerly materialize
        ///             buffers/initval views for blocks when they are allocated
        ///             at graph finalization. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/13/2014. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if block pool view accelerator, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        IsBlockPoolViewAccelerator(
            __in Accelerator* pAccelerator
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
        /// <summary>   toString() implementation for std::strings. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="os">   [in,out] The operating system. </param>
        /// <param name="port"> The port. </param>
        ///
        /// <returns>   The shifted result. </returns>
        ///-------------------------------------------------------------------------------------------------

        friend std::ostream& operator<<(std::ostream& os, Port * pPort); 

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks any bound channels. For some operations it is necessary
        ///             to enforce a lock ordering discipline between channels and ports.
        ///             For example, naively, pushing into a channel locks the channel
        ///             and then finds the attached port, and locks it, while the dispatch
        ///             ready checker encounters ports first and traverses them to get to
        ///             channels, which naively encourages the opposite order and admits
        ///             the possibility of deadlock. Consequently, we require a channel->port
        ///             ordering when both locks are required. This utility allows a port
        ///             to lock it attached channels before acquiring its own lock when necessary.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/21/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void LockBoundChannels();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks any bound channels. For some operations it is necessary
        ///             to enforce a lock ordering discipline between channels and ports.
        ///             For example, naively, pushing into a channel locks the channel
        ///             and then finds the attached port, and locks it, while the dispatch
        ///             ready checker encounters ports first and traverses them to get to
        ///             channels, which naively encourages the opposite order and admits
        ///             the possibility of deadlock. Consequently, we require a channel->port
        ///             ordering when both locks are required. This utility allows a port
        ///             to lock it attached channels before acquiring its own lock when necessary.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/21/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void UnlockBoundChannels();

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
                                                           PTask::Graph * pGraph)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the gated ports. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the gated ports. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual std::vector<Port*>*  GetGatedPorts();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the channel bound to this port at the given index. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pChannelList">     [in,out] If non-null, list of channels. </param>
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        ///
        /// <returns>   null if it fails, else the channel. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Channel *		GetChannel(std::vector<Channel*>* pChannelList, UINT nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a channel count. </summary>
        ///
        /// <remarks>   Crossbac, 2/2/2012. </remarks>
        ///
        /// <param name="pChannelList"> [in,out] If non-null, list of channels. </param>
        ///
        /// <returns>   The channel count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT            GetChannelCount(std::vector<Channel*>* pChannelList);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pulls from the given pChannelList. </summary>
        ///
        /// <remarks>   Crossbac, 2/2/2012. </remarks>
        ///
        /// <param name="pChannelList"> [in,out] If non-null, list of channels. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Datablock *     AttemptPull(std::vector<Channel*>* pChannelList);

        /// <summary> Unique port identifier. </summary>
        UINT					            m_uiId;
        /// <summary> The channels </summary>
        std::vector<Channel*>               m_vChannels;
        /// <summary> The control channels </summary>
        std::vector<Channel*>               m_vControlChannels;
        /// <summary>   the channel lock set: std set gives a sorted order (despite the name "set")
        ///             which gives us a lock acquisition order as a side-effect, enabling
        ///             us to avoid deadlocking when there are multiple channels bound to 
        ///             ports. </summary>
        std::set<Channel*>                  m_vChannelLockSet;
        /// <summary> The descriptor ports </summary>
        std::vector<DEFERREDPORTDESC*>      m_vDescriptorPorts;
        /// <summary> The deferred channels </summary>
        std::vector<DEFERREDCHANNELDESC*>   m_vDeferredChannels;
        /// <summary> The gated ports. </summary>
        std::vector<Port *>                 m_pGatedPorts;
        /// <summary> The control propagation ports. </summary>
        std::vector<Port *>                 m_pControlPropagationPorts;
        /// <summary> The control propagation channels. </summary>
        std::set<Channel *>                 m_pControlPropagationChannels;
        /// <summary> The deferred blocks </summary>
        std::vector<Datablock*>             m_vDeferredBlocks;
        /// <summary> The control propagation source. </summary>
        Port *                              m_pControlPropagationSource;
        /// <summary> true if port data is marshallable (can be migrated) </summary>
        BOOL                                m_bMarshallable;
        /// <summary>   true if the port is (or can be) connected to a data source or sink that can be
        ///             streamed. Generally speaking, this is a property of the primitive whose IO
        ///             resources are being exposed by this port;
        ///             consequently this property must be set explicitly by the programmer when graph
        ///             structures that are stateful are constructured. For example, in a sort primitive,
        ///             the main input can be streamed (broken into multiple blocks) only if there is a
        ///             merge network downstream of the node performing the sort. Code that feeds the
        ///             main input port needs to know this to decide whether to grow blocks until all
        ///             data is present, or two push partial input.
        ///             </summary>
        BOOL                                m_bCanStream;
        /// <summary>   The geometry dim binding. </summary>
        GEOMETRYESTIMATORDIMENSION          m_eGeometryDimBinding;

        /// <summary>   The template </summary>
        DatablockTemplate *		m_pTemplate;
        /// <summary>   Type of the port </summary>
        PORTTYPE		m_ePortType;
        /// <summary>   The bound task </summary>
        Task *			m_pBoundTask;
        /// <summary>   The graph. </summary>
        Graph *         m_pGraph;
        /// <summary>   Zero-based index of this port in the array it was in when passed into AddTask. </summary>
        UINT            m_uiOriginalIndex;
        /// <summary>   Zero-based index of the m user interface bound port </summary>
        UINT			m_uiBoundPortIndex;
        /// <summary>   Zero-based index of the m n formal parameter </summary>
        int             m_nFormalParameterIndex;
        /// <summary>   Zero-based index of the m n in out routing </summary>
        int             m_nInOutRoutingIndex;
        /// <summary>   The variable binding </summary>
        char *			m_lpszVariableBinding;
        /// <summary>   The propagated control code </summary>
        CONTROLSIGNAL   m_luiPropagatedControlCode;
        /// <summary>   The initial propagated control code, if there is one </summary>
        CONTROLSIGNAL   m_luiInitialPropagatedControlCode;
        /// <summary>   true if sticky </summary>
        BOOL            m_bSticky;
        /// <summary>   true if destructive. </summary>
        BOOL            m_bDestructive;
        /// <summary>   true if the port carries data that can be used to
        ///             infer the thread group dispatch dimensions. </summary>
        BOOL            m_bDispatchDimensionsHint;
        /// <summary> The replayable block </summary>
        Datablock *     m_pReplayableBlock;
        /// <summary>   Number of replays. </summary>
        UINT            m_uiIterationUpperBound;
        /// <summary>   Number of replays. </summary>
        UINT            m_uiStickyIterationUpperBound;
        /// <summary>   Current replay index. </summary>
        UINT            m_uiIterationIndex;
        /// <summary>   true if iterated. </summary>
        BOOL            m_bIterated;
        /// <summary>   The iteration source. </summary>
        Port *          m_pIterationSource;
        /// <summary>   The iteration targets. </summary>
        std::vector<Port*>      m_vIterationTargets;
        /// <summary>   true to active iteration scope. </summary>
        BOOL            m_bActiveIterationScope;
        /// <summary>   true to suppress clone. </summary>
        BOOL            m_bSuppressClones;
        /// <summary>   true if this is a global trigger port. </summary>
        BOOL            m_bTriggerPort;
        /// <summary>   true if this is scope terminus for some subgraph. </summary>
        BOOL            m_bScopeTerminus;
        /// <summary>   The scope terminal signal, valid if m_bScopeTerminus is TRUE </summary>
        CONTROLSIGNAL   m_luiScopeTerminalSignal;
        /// <summary>   The dependent accelerator binding. </summary>
        int             m_nDependentAcceleratorBinding;
        /// <summary>   The acc dependent accelerator class. </summary>
        ACCELERATOR_CLASS       m_accDependentAcceleratorClass;
        /// <summary>   true if this port has a permanent block
        /// 			bound to it. </summary>
        BOOL            m_bPermanentBlock;
        /// <summary> A map of dependent accelerator-affinity types </summary>
        std::map<Accelerator*, AFFINITYTYPE>    m_vAffinities;
        /// <summary>   If the user specifies an accelerator with mandatory affinity for a dependent
        ///             binding on this port, we cache it here so that we can avoid repeated work finding
        ///             a constraint that trumps all other possible scheduling constraints.
        ///             </summary>
        Accelerator *   m_pMandatoryAccelerator;
        /// <summary>   true if this object has upstream channel pool. </summary>
        BOOL            m_bHasUpstreamChannelPool;
        /// <summary>   true if upstream channel pool is growable. </summary>
        BOOL            m_bUpstreamChannelPoolGrowable;
        /// <summary>   Size of the upstream channel pool. </summary>
        UINT            m_uiUpstreamChannelPoolSize;
        /// <summary>   The upstream channel pool grow increment. </summary>
        UINT            m_uiUpstreamChannelPoolGrowIncrement;
        /// <summary>   The lui sticky release signal. </summary>
        CONTROLSIGNAL   m_luiStickyReleaseSignal;
        /// <summary>   true if a sticky release signal configured. </summary>
        BOOL            m_bStickyReleaseSignalConfigured; 
        /// <summary>   a global map of sticky release ports. </summary>
        static std::map<Graph*, std::set<Port*>> m_vReleaseableStickyPorts;
        /// <summary>   The lock for the releasable sticky ports map. </summary>
        static CRITICAL_SECTION m_csReleasableStickyPorts;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check if control signals are present for this port, either as propagated codes,
        /// 			or as codes carried by any sticky or replayable blocks.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/2/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CheckControlCodes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force sticky port release. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void ForceStickyDeviceBufferRelease();

        static void InitializeGlobal();

        static void DestroyGlobal();
    };

};
#endif
