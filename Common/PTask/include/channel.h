//--------------------------------------------------------------------------------------
// File: channel.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _CHANNEL_H_
#define _CHANNEL_H_

#include "primitive_types.h"
#include "ReferenceCounted.h"
#include <vector>
#include <deque>
#include <map>

namespace PTask {

    class Task;
    class Port;
    class Graph;
    class Channel;
    class Datablock;
    class DatablockTemplate;
    class AsyncContext;
    class Accelerator;
    class ChannelProfiler;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent channel endpoint types. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum endpt_type_t {
        
        /// <summary> Source end of a channel. </summary>
        CE_SRC = 0,
        
        /// <summary> Destination end of a channel. </summary>
        CE_DST = 1

    } CHANNELENDPOINTTYPE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   The number of possible channel endpoint types. This seems obvious, but 1:N, N:1,
    ///             N:N are possible, and we may wish to do something in the future for which it is
    ///             meaningful to be able to tell the difference between single component channels
    ///             and the aggregate of all channels for the N-ary ends.
    ///             </summary>
    ///-------------------------------------------------------------------------------------------------

    static const int NUMENDPOINTTYPES = CE_DST - CE_SRC + 1;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent channel types. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum enum_ct_t {
        
        /// <summary> Input-only channel. The destination endpoint 
        /// 		  can be bound to a port but the source is 
        /// 		  expected to be fed by user code. 
        /// 		  </summary>
        CT_GRAPH_INPUT,

        /// <summary> Ouput-only channel. The source endpoint 
        /// 		  can be bound to a port but the destinatoin is 
        /// 		  expected to be drained by user code. 
        /// 		  </summary>
        CT_GRAPH_OUTPUT,

        /// <summary> Internal channel. Both ends bound to ports. </summary>
        CT_INTERNAL,

        /// <summary>  channel with single input, multiple outputs </summary>
        CT_MULTI,

        /// <summary>   channel that can produce data in response to a pull,
        /// 			without requiring an upstream data source. 
        /// 			</summary>
        CT_INITIALIZER

    } CHANNELTYPE;

    static const DWORD DEFAULT_CHANNEL_TIMEOUT = INFINITE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent actions to take when a predicate fails
    /// 			for a push or pull on a predicated channel. Currently, we have
    /// 			two possibilities: either release the block, or fail the operation. 
    /// 			TODO: these semantics need to be ironed out more thoroughly. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 2/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum predfailaction_t {

        /// <summary>   If the predicate fails, call release on the block,
        /// 			and do not proceed with the push/pull operation. 
        /// 			</summary>
        PFA_RELEASE_BLOCK = 0,

        /// <summary>   If the predicate fails, the push or pull fails,
        /// 			but no action is taken that can affect the state
        /// 			of the channel or block. 
        /// 			</summary>
        PFA_FAIL_OPERATION = 1

    } PREDICATE_FAILURE_ACTION;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   User-supplied channel predication functions must be of type LPFNCHANNELPREDICATE,
    ///             which accepts a channel pointer and a datablock and returns true if the block
    ///             passes the predicate. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef BOOL (__stdcall *LPFNCHANNELPREDICATE)(Channel * pChannel, 
                                                   Datablock * pBlock,
                                                   PREDICATE_FAILURE_ACTION eFailAction);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent canonical predication functions. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum _gate_predicate_fns {
        
        /// <summary> no predication on this channel </summary>
        CGATEFN_NONE = 0,

        /// <summary> The channel should close when a datablock marked EOF is encountered,
        /// 		  meaning the channel predicate is true for all datablocks *without*
        /// 		  EOF control signal. 
        /// 		  </summary>
        CGATEFN_CLOSE_ON_EOF = 1,
        
        /// <summary> The channel should open when a datablock marked EOF is encountered,
        /// 		  meaning the channel predicate is true for all datablocks *with*
        /// 		  EOF control signal. 
        /// 		  </summary>
        CGATEFN_OPEN_ON_EOF = 2,

        /// <summary> The channel should open when a datablock marked begin-iteration is encountered,
        /// 		  meaning the channel predicate is true for all datablocks *with*
        /// 		  begin-iteration control signal. 
        /// 		  </summary>
        CGATEFN_OPEN_ON_BEGINITERATION = 3,

        /// <summary> The channel should close when a datablock marked begin-iteration is encountered,
        /// 		  meaning the channel predicate is false for all datablocks *with*
        /// 		  begin-iteration control signal. 
        /// 		  </summary>
        CGATEFN_CLOSE_ON_BEGINITERATION = 4,

        /// <summary> The channel should open when a datablock marked end-iteration is encountered,
        /// 		  meaning the channel predicate is true for all datablocks *with*
        /// 		  end-iteration control signal. 
        /// 		  </summary>
        CGATEFN_OPEN_ON_ENDITERATION = 5,

        /// <summary> The channel should close when a datablock marked end-iteration is encountered,
        /// 		  meaning the channel predicate is false for all datablocks *with*
        /// 		  end-iteration control signal. 
        /// 		  </summary>
        CGATEFN_CLOSE_ON_ENDITERATION = 6,

        /// <summary>   Use this predicator to intentionally create outputs
        /// 			with values that will never pass the predicate. 
        /// 			Think: "myDataSource > /dev/null" . </summary>
        CGATEFN_DEVNULL = 7,

        /// <summary> The channel should close when a datablock marked EOF is encountered,
        /// 		  meaning the channel predicate is true for all datablocks *without*
        /// 		  EOF control signal. 
        /// 		  </summary>
        CGATEFN_CLOSE_ON_BOF = 8,
        
        /// <summary> The channel should open when a datablock marked EOF is encountered,
        /// 		  meaning the channel predicate is true for all datablocks *with*
        /// 		  EOF control signal. 
        /// 		  </summary>
        CGATEFN_OPEN_ON_BOF = 9,
        
        /// <summary> The user has supplied a callback for a predication function.
        /// 		  </summary>
        CGATEFN_USER_DEFINED = 10
        
        // ....
        // TODO: other canonical functions
        // 
    } CHANNELPREDICATE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Channel predication descriptor. </summary>
    ///
    /// <remarks>   Crossbac, 2/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct CHANNELPREDICATIONDESC_t  {

        /// <summary>   The endpoint to which the described predicate applies.
        /// 			This is redundant, as we will store a descriptor per
        /// 			endpoint, but is helpful for debugging. 
        /// 			</summary>
        CHANNELENDPOINTTYPE         eEndpoint; 

        /// <summary>   Is the predicator being described a canonical predicate and if so, which
        ///             canonical predicate is it? If not, the function pointer entries below must be
        ///             set. Some predicates are very common, such as open or close when an DBCTL_EOF
        ///             control signal is present on the block. For common predicates we allow the user
        ///             to specify the predicate function without providing a callback.
        ///             </summary>
        CHANNELPREDICATE            eCanonicalPredicate;

        /// <summary>   A function pointer to a user-defined predicate function. Can be null. If non-null,
        ///             this function will be called when datablocks are pushed into this channel. If the
        ///             predicate holds, the datablock will be pushed, otherwise it is released. This
        ///             functions is a user-supplied callback, and if non-null, m_ePredicate should be
        ///             set to USER_DEFINED.
        ///             </summary>       
        LPFNCHANNELPREDICATE        lpfnPredicate;

        /// <summary>   The predicate failure action: what should happen
        /// 			if a test of this predicate fails (release the block,
        /// 			or fail the operation?) 
        /// 			</summary>
        PREDICATE_FAILURE_ACTION    ePredicateFailureAction; 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 2/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        CHANNELPREDICATIONDESC_t() : 
            eEndpoint(CE_SRC),
            eCanonicalPredicate(CGATEFN_NONE),
            lpfnPredicate(NULL),
            ePredicateFailureAction(PFA_RELEASE_BLOCK) {}

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   constructor. </summary>
        ///
        /// <remarks>   Crossbac, 2/1/2012. </remarks>
        ///
        /// <param name="eType">    The type. </param>
        ///-------------------------------------------------------------------------------------------------

        CHANNELPREDICATIONDESC_t(CHANNELENDPOINTTYPE eType) : 
            eEndpoint(eType),
            eCanonicalPredicate(CGATEFN_NONE),
            lpfnPredicate(NULL),
            ePredicateFailureAction(PFA_RELEASE_BLOCK) {}

    } CHANNELPREDICATIONDESC;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Channel superclass. Almost all channel functionality is actually implemented in
    ///             the superclass because it is trivial to distinguish channel types by what
    ///             combination of source/dest endpoints are actually bound to a port.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class Channel : public ReferenceCounted
    {
        friend class XMLWriter;
        friend class XMLReader;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets control signals of interest. </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <param name="ePredicate">   The predicate. </param>
        ///
        /// <returns>   The control signals of interest. </returns>
        ///-------------------------------------------------------------------------------------------------

        static CONTROLSIGNAL
        GetControlSignalsOfInterest(
            CHANNELPREDICATE ePredicate
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pGraph">                   [in,out] If non-null, the graph. </param>
        /// <param name="pDatablockTemplate">       [in] If non-null, the datablock template. </param>
        /// <param name="hRuntimeTerminateEvent">   Handle of the runtime terminate event. </param>
        /// <param name="hGraphTeardownEvt">        Handle of the graph teardown event. </param>
        /// <param name="hGraphStopEvent">          Handle of the graph stop event. </param>
        /// <param name="lpszChannelName">          [in] If non-null, name of the channel. </param>
        /// <param name="bHasBlockPool">            the has block pool. </param>
        ///-------------------------------------------------------------------------------------------------

        Channel(
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
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Channel();

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
        /// <summary>   Derives an initial value datablock for this channel based on its template,
        /// 			and pushes that datablock into this channel, blocking until there is capacity
        /// 			for an optional timeout in milliseconds. Default timeout is infinite. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="dwTimeout">    (optional) the timeout in milliseconds. Use 0xFFFFFFFF for no
        ///                             timeout. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL PushInitializer(DWORD dwTimeout=0xFFFFFFFF);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the capacity of the channel, which is the maximum number of datablocks it
        ///             can queue before subsequent calls to push will block.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="nCapacity">    The capacity. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetCapacity(UINT nCapacity);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the capacity of the channel, which is the maximum number of datablocks it
        ///             can queue before subsequent calls to push will block.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The capacity. </returns>
        ///
        /// ### <param name="nCapacity">    The capacity. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetCapacity();

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

        virtual BOOL            CanStream()=0;

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

        virtual BOOL            HasDownstreamWriters()=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind this channel to a port. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pPort">    [in] non-null, the port to bind. </param>
        /// <param name="type">     (optional) the type of the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void BindPort(Port * pPort, CHANNELENDPOINTTYPE type);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unbind a port from this channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="type"> (optional) the type of the channel. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port * UnbindPort(CHANNELENDPOINTTYPE type);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the port to which this channel is bound. Lock not required because we assume
        ///             this is set at creation, rather than after the graph has entered the running
        ///             state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="type"> (optional) the type of the channel. </param>
        ///
        /// <returns>   null if it fails, else the bound port. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port * GetBoundPort(CHANNELENDPOINTTYPE type);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the trigger port. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the trigger port. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port * GetTriggerPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a trigger channel. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        /// <param name="bTrigger"> true to trigger. </param>
        ///-------------------------------------------------------------------------------------------------

        void            
        SetTriggerChannel(
            __in Graph * pGraph, 
            __in BOOL bTrigger
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is a trigger channel. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <returns>   true if trigger port, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL            IsTriggerChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Handle trigger. </summary>
        ///
        /// <remarks>   crossbac, 5/23/2012. </remarks>
        ///
        /// <param name="uiCode">   The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            HandleTriggers(CONTROLSIGNAL luiCode);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the type of this channel. Lock not required because we assume this is set at
        ///             creation, rather than after the graph has entered the running state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The channel type. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual CHANNELTYPE GetType();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the default timeout for this channel, which will be used for Push/Pull calls
        ///             when no timeout parameter is specified. Default is infinite. Lock not required
        ///             because we assume this is set at creation, rather than after the graph has
        ///             entered the running state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="dwTimeoutMilliseconds">    The timeout. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetTimeout(DWORD dwTimeoutMilliseconds);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the datablock template associated with this port. Lock not required because
        ///             we assume this is set at creation, rather than after the graph has entered the
        ///             running state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the template. </returns>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate * GetTemplate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether this channel will be drawn by graph rendering tools drawing the
        ///             current graph. Lock not required because we assume this is set at creation,
        ///             rather than after the graph has entered the running state state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void SetNoDraw();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queries if we should draw this channel when rendering the graph. Lock not
        ///             required because drawing must take place before the graph has entered the running
        ///             state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        bool ShouldDraw();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the user-provided channel name. Lock not required because we assume this is
        ///             set at creation, rather than after the graph has entered the running state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the channel name. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual char * GetName();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the current queue depth. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The queue depth. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual size_t GetQueueDepth();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Drains this channels queue, releasing references to the blocks in the queue.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void Drain();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets this object. </summary>
        ///
        /// <remarks>   Crossbac, 5/2/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual void Reset();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this channel has reached any transit limits. </summary>
        ///
        /// <remarks>   crossbac, 6/27/2013. </remarks>
        ///
        /// <returns>   true if transit limit reached, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsTransitLimitReached();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets block transit limit. Limits this channel to 
        ///             delivering a specified limit before closing. Can be cleared by 
        ///             calling graph::Reset.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/2/2013. </remarks>
        ///
        /// <param name="uiBlockTransitLimit">  The block transit limit. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetBlockTransitLimit(UINT uiBlockTransitLimit);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the block transit limit. Limits this channel to 
        ///             delivering a specified limit before closing. Can be cleared by 
        ///             calling graph::Reset.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 5/2/2013. </remarks>
        ///
        /// <param name="uiBlockTransitLimit">  The block transit limit. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetBlockTransitLimit();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the channel predicator function. Lock not required because we assume this is
        ///             set at creation, rather than after the graph has entered the running state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="eEndpoint">    To which end of the channel does the predicate apply? If CE_SRC
        ///                             is used, pushes have no effect when the predicate does not hold. If
        ///                             CE_DST is used, pulls have no effect, but the upstream producer can
        ///                             still queue data in the channel. </param>
        ///
        /// <returns>   The channel predicator for the given endpoint. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual LPFNCHANNELPREDICATE GetPredicator(CHANNELENDPOINTTYPE eEndpoint);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the channel predication type. Lock not required because we assume this is
        ///             set at creation, rather than after the graph has entered the running state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="eEndpoint">    To which end of the channel does the predicate apply? If CE_SRC
        ///                             is used, pushes have no effect when the predicate does not hold. If
        ///                             CE_DST is used, pulls have no effect, but the upstream producer can
        ///                             still queue data in the channel. </param>
        ///
        /// <returns>   The channel predication type for the given endpoint. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual CHANNELPREDICATE GetPredicationType(CHANNELENDPOINTTYPE eEndpoint);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a channel predicator. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="eEndpoint">    To which end of the channel does the predicate apply? If CE_SRC
        ///                             is used, pushes have no effect when the predicate does not hold. If
        ///                             CE_DST is used, pulls have no effect, but the upstream producer can
        ///                             still queue data in the channel. </param>
        /// <param name="lpfn">         A function pointer to a channel predication function. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetPredicator(CHANNELENDPOINTTYPE eEndpoint, LPFNCHANNELPREDICATE lpfn);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a predication type. </summary>
        ///
        /// <remarks>   Crossbac, 2/1/2012. </remarks>
        ///
        /// <param name="eEndpoint">            To which end of the channel does the predicate apply? If
        ///                                     CE_SRC is used, pushes have no effect when the predicate
        ///                                     does not hold. If CE_DST is used, pulls have no effect,
        ///                                     but the upstream producer can still queue data in the
        ///                                     channel. </param>
        /// <param name="eCanonicalPredicator"> The canonical predicator. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetPredicationType(CHANNELENDPOINTTYPE eEndpoint, 
                                        CHANNELPREDICATE eCanonicalPredicator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a predication type. </summary>
        ///
        /// <remarks>   Crossbac, 2/1/2012. </remarks>
        ///
        /// <param name="eEndpoint">            To which end of the channel does the predicate apply? If
        ///                                     CE_SRC is used, pushes have no effect when the predicate
        ///                                     does not hold. If CE_DST is used, pulls have no effect,
        ///                                     but the upstream producer can still queue data in the
        ///                                     channel. </param>
        /// <param name="eCanonicalPredicator"> The canonical predicator. </param>
        /// <param name="pfa">                  The pfa. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetPredicationType(CHANNELENDPOINTTYPE eEndpoint, 
                                        CHANNELPREDICATE eCanonicalPredicator,
                                        PREDICATE_FAILURE_ACTION pfa);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this channel is predicated at the given end of the channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="eEndpoint">    The endpoint: CE_SRC, or CE_DST </param>
        ///
        /// <returns>   true if predicated, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsPredicated(CHANNELENDPOINTTYPE eEndpoint);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the view materialization policy for this channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="policy">   The policy. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetViewMaterializationPolicy(VIEWMATERIALIZATIONPOLICY policy);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the view materialization policy of this channel. Lock not required because
        ///             we assume this is set at creation, rather than after the graph has entered the
        ///             running state.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The view materialization policy. </returns>
        ///-------------------------------------------------------------------------------------------------

        VIEWMATERIALIZATIONPOLICY GetViewMaterializationPolicy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a control propagation source for this channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    [in] non-null, a the port. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetControlPropagationSource(Port * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the control propagation source for this channel </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the control propagation source. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Port*           GetControlPropagationSource();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets want most recent view. Consumers of data on this channel want only the most
        ///             recent value pushed into it. If this member is true, on a push, any previously
        ///             queued blocks will be drained and released. To be used with caution, as it can
        ///             upset the balance of blocks required to keep a pipeline from stalling: typically
        ///             channels with this property set should either be connected to ports with the
        ///             sticky property set, or should be external (exposed channels e.g. outputs)
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2013. </remarks>
        ///
        /// <param name="bWantMostRecentView">  true to want most recent view. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetWantMostRecentView(BOOL bWantMostRecentView);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Consumers of data on this channel want only the most recent value pushed into it.
        ///             If this member is true, on a push, any previously queued blocks will be drained
        ///             and released. To be used with caution, as it can upset the balance of blocks
        ///             required to keep a pipeline from stalling: typically channels with this property
        ///             set should either be connected to ports with the sticky property set, or should
        ///             be external (exposed channels e.g. outputs)
        ///             
        ///             Gets want most recent view.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/14/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL           GetWantMostRecentView();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a propagated control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="luiCode">  The code. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void            SetPropagatedControlSignal(CONTROLSIGNAL luiCode);

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

        virtual CONTROLSIGNAL GetPropagatedControlSignals();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the given block passes the predicate. The block passes if either
        ///             1) the channel predicate holds for that block, or 2) no channel predicate is in
        ///             force.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="eEndpoint">    The endpoint. </param>
        /// <param name="pBlock">       [in,out] If non-null, the block. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL    PassesPredicate(CHANNELENDPOINTTYPE eEndpoint, Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the given signal passes the predicate.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="eCanonicalPredicate">  The predicate. </param>
        /// <param name="luiSignal">            the control code. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL    
        SignalPassesCanonicalPredicate(
            __in CHANNELPREDICATE eCanonicalPredicate, 
            __in CONTROLSIGNAL luiCtlCode
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Attempt to get an asynchronous context object for the task attached at the
        ///             specified endpoint. If multiple accelerators can execute the attached task, we
        ///             cannot make any assumptions about what contexts will be bound to it. Conversely,
        ///             if the task has a single platform type, and a single device exists in the system
        ///             that can execute it, we can figure out what AsyncContext will be used before it
        ///             is bound.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/11/2012. </remarks>
        ///
        /// <param name="eEndpoint">            To which end of the channel is the sought after context
        ///                                     attached? </param>
        /// <param name="eAsyncContextType">    Type of the asynchronous context on which any pending
        ///                                     operations might be scheduled. For this API, which is used to
        ///                                     try to schedule downstream transfers from the host eagerly,
        ///                                     this should almost always be the HtoD transfer context. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        AsyncContext * 
        FindAsyncContext(
            __in CHANNELENDPOINTTYPE eEndpoint,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Populate a set of tasks that are bound to this channel as consumers. Because a
		/// 			channel may be an output channel or a multi-channel, the range of cardinality of
		/// 			this result is [0..n]. Return the number of such tasks. Note that we cache the
		/// 			result of this call: computing it requires a transitive closure over paths that
		/// 			can include multi-channels and in/out routing, which in turn means traversing the
		/// 			graph recursively. Since the result of this traversal cannot change, and the
		/// 			traversal requires locking parts of the graph, we prefer to avoid repeating work
		/// 			to recompute the same result.
		/// 			</summary>
		///
		/// <remarks>	Crossbac, 10/2/2012. </remarks>
		///
		/// <param name="pvTasks">	[in,out] non-null, the tasks. </param>
		///
		/// <returns>	The number of downstream consuming tasks. </returns>
		///-------------------------------------------------------------------------------------------------

		virtual UINT 
		GetDownstreamTasks(
			__inout  std::set<Task*>* pvTasks
			);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Gets memory spaces downstream of this channel that either *must* consume data
		/// 			that flows through this channel, or *may* consume it. The list is non-trivial
		/// 			because of different channel types and predication. For example, an output
		/// 			channel has no downstream consumers, while a multi-channel can have any number.
		/// 			Enumerating consumers is complicated by the following additional factors:
		/// 			
		/// 			1) The presence of channel predicates can ensure dynamically that a particular
		/// 			bound task never actually consumes a block flowing through it.
		/// 			
		/// 			2) If the channel is bound to In/out ports, then we need to analyze paths of
		/// 			length greater than 1. In fact, we need the transitive closure.
		/// 			
		/// 			3) A task's accelerator class may enable it to be bound to several different
		/// 			accelerators, meaning the list of potential consumers can be greater than 1 even
		/// 			if the channel binding structure is trivial.
		/// 			
		/// 			Note that we cache the result of this call: computing it requires a transitive
		/// 			closure over paths that can include multi-channels and in/out routing, which in
		/// 			turn means traversing the graph recursively. Since the result of this traversal
		/// 			cannot change, and the traversal requires locking parts of the graph, we prefer
		/// 			to avoid repeating work to recompute the same result.
		/// 			</summary>
		///
		/// <remarks>	Crossbac, 10/2/2012. </remarks>
		///
		/// <param name="ppvMandatoryAccelerators">	[in,out] If non-null, the mandatory accelerators. </param>
		/// <param name="ppvPotentialAccelerators">	[in,out] If non-null, the potential accelerators. </param>
		///
		/// <returns>	The downstream memory spaces. </returns>
		///-------------------------------------------------------------------------------------------------

		virtual BOOL
		GetDownstreamMemorySpaces(
			__inout	 std::set<Accelerator*>** ppvMandatoryAccelerators,
			__inout  std::set<Accelerator*>** ppvPotentialAccelerators
			);		

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Gets memory spaces downstream of this channel that either *must* consume data
		/// 			that flows through this channel, or *may* consume it. The list is non-trivial
		/// 			because of different channel types and predication. For example, an output
		/// 			channel has no downstream consumers, while a multi-channel can have any number.
		/// 			Enumerating consumers is complicated by the following additional factors:
		/// 			
		/// 			1) The presence of channel predicates can ensure dynamically that a particular
		/// 			bound task never actually consumes a block flowing through it.
		/// 			
		/// 			2) If the channel is bound to In/out ports, then we need to analyze paths of
		/// 			length greater than 1. In fact, we need the transitive closure.
		/// 			
		/// 			3) A task's accelerator class may enable it to be bound to several different
		/// 			accelerators, meaning the list of potential consumers can be greater than 1 even
		/// 			if the channel binding structure is trivial.
		/// 			
		/// 			Note that we cache the result of this call: computing it requires a transitive
		/// 			closure over paths that can include multi-channels and in/out routing, which in
		/// 			turn means traversing the graph recursively. Since the result of this traversal
		/// 			cannot change, and the traversal requires locking parts of the graph, we prefer
		/// 			to avoid repeating work to recompute the same result.
		/// 			</summary>
		///
		/// <remarks>	Crossbac, 10/2/2012. </remarks>
		///
		/// <param name="ppvMandatoryAccelerators">	[in,out] If non-null, the mandatory accelerators. </param>
		/// <param name="ppvPotentialAccelerators">	[in,out] If non-null, the potential accelerators. </param>
		///
		/// <returns>	The downstream memory spaces. </returns>
		///-------------------------------------------------------------------------------------------------

		virtual BOOL
		EnumerateDownstreamMemorySpaces(
			__inout	 std::set<Accelerator*>* pvMandatoryAccelerators,
			__inout  std::set<Accelerator*>* pvPotentialAccelerators
			);		

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the unambigous downstream memory space if there is one. </summary>
        ///
        /// <remarks>   crossbac, 7/8/2013. </remarks>
        ///
        /// <returns>   null if the downstream memory space for any blocks pushed into this
        ///             channel cannot be determined unambiguously at the time of the call. 
        ///             If such can be determined, return the accelerator object associated with
        ///             that memory space. 
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator * 
        GetUnambigousDownstreamMemorySpace(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets cumulative block transit. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <returns>   The cumulative block transit. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetCumulativeBlockTransit();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets maximum occupancy. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <returns>   The maximum occupancy. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetMaxOccupancy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets cumulative occupancy. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <returns>   The cumulative occupancy. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetCumulativeOccupancy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets occupancy samples. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <returns>   The occupancy samples. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetOccupancySamples();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is pool owner. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <returns>   true if pool owner, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsPoolOwner();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the graph. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the graph. </returns>
        ///-------------------------------------------------------------------------------------------------

        Graph * GetGraph();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has a user specified name, return false if the 
        ///             runtime generated one on demand for it. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <returns>   true if user specified name, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasUserSpecifiedName();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this channel has any non trivial predicates. </summary>
        ///
        /// <remarks>   crossbac, 7/3/2014. </remarks>
        ///
        /// <returns>   true if non trivial predicate, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasNonTrivialPredicate();

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
        /// <summary>   Find the maximal capacity downstream port/channel path starting at this channel.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 1/3/2014. </remarks>
        ///
        /// <param name="vTasksVisited">    [in,out] [in,out] If non-null, the tasks visited. </param>
        /// <param name="vPath">            [in,out] list of channels along the maximal path. </param>
        ///
        /// <returns>   The found maximal downstream capacity. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT 
        FindMaximalDownstreamCapacity(
            __inout std::set<Task*>& vTasksVisited,
            __inout std::vector<Channel*>& vPath
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check semantics. Return true if all the structures are initialized for this
        ///             channel in a way that is consistent with a well-formed graph.
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
        /// <summary>   Channel.toString() </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="os">       [in,out] The operating system. </param>
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        ///
        /// <returns>   The shifted result. </returns>
        ///-------------------------------------------------------------------------------------------------

        friend std::ostream& operator<<(std::ostream &os, Channel * pChannel); 

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Handle any triggers on the destination ports. If the given port has global
        ///             triggers associated with it, check the block for control codes and defer
        ///             execution of those global triggers to the port object.
        ///             </summary>
        ///
        /// <remarks>   channel lock must be held.
        ///             
        ///             crossbac, 6/19/2012.
        ///             </remarks>
        ///
        /// <param name="pBlock">   [in] non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        ConfigurePortTriggers(
            __in Datablock * pBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Handle any triggers on the channel itself. If this channel has global
        ///             triggers associated with it, check the block for control codes and defer
        ///             execution of those global triggers to the channel object.
        ///             </summary>
        ///
        /// <remarks>   channel lock must be held.
        ///             
        ///             crossbac, 6/19/2012.
        ///             </remarks>
        ///
        /// <param name="pBlock">   [in] non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        ConfigureChannelTriggers(
            __in Datablock * pBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Handle any iteration configured for the destination port. If the given port has a
        ///             meta-port with iteration as the meta function, some of that work must be
        ///             performed as part of pushing the block into this channel. Delegate that work to
        ///             the downstream meta port object.
        ///             </summary>
        ///
        /// <remarks>   channel lock must be held.
        ///             
        ///             crossbac, 6/19/2012.
        ///             </remarks>
        ///
        /// <param name="pDstPort"> [in] non-null, destination port. </param>
        /// <param name="pBlock">   [in] non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        ConfigureDownstreamIterationTargets(
            __in Datablock * pBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   If the destination port has deferred channels, derive blocks based on the
        /// 			descriptor functions, and push them into those channels. </summary>
        ///
        /// <remarks>   channel lock must be held.
        /// 			
        /// 			crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pBlock">   [in] non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        PushDescriptorBlocks(
            __in Datablock * pBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Signal any downstream consumers of this channel that something interesting has
        ///             happened (e.g. if tasks should check whether all inputs are now available!)
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void SignalDownstreamConsumers();

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
                                                           PTask::Graph * pGraph)=0;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a unique name for the channel based on whatever data the caller has
        ///             supplied.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        char *                      CreateUniqueName();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check queue invariants--given a queue of blocks and a channel, do some simple
        ///             checks to detect conditions that can lead to incorrect/unexpected results. For
        ///             example, multiple entries of the same object is fine, but only if there are no
        ///             channel predicates (because a control signal change on a block will affect
        ///             multiple queue entries, and therefore has the potential to cause predicates to
        ///             change state at the wrong times). If a channel is a simple cycle with an inout
        ///             port pair, assert that the channel can never have more than one entry.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/27/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CheckQueueInvariants();

        /// <summary> The datablock template for this channel. Currently used to assist in allocation of
        /// 		  Datablocks in user code--if the caller of Runtime::AllocateDatablock() specifies
        /// 		  a destination channel, we can use the template on the channel to determine  
        /// 		  allocation dimensions. 
        /// 		  ----------------------
        /// 		  TODO: FIXME: Implement type some basic type-checks: this template should match
        /// 		  the port as well as Datablocks that are pushed into this channel. Under
        /// 		  some definition of "match" that is.
        /// 		  </summary>
        DatablockTemplate *		    m_pTemplate;

        /// <summary> The Datablock queue for this channel </summary>
        std::deque<Datablock*>	    m_q;

        /// <summary> The channel type (input, output, internal) </summary>
        CHANNELTYPE				    m_type;
        
        /// <summary> The capacity of queue. This is configurable with SetCapacity(). </summary>
        UINT					    m_uiCapacity;
        
        /// <summary> Handle of the empty event. This event is set when the queue is
        /// 		  empty to unblock threads waiting for the queue to be empty. 
        /// 		  </summary>
        HANDLE					    m_hEmpty;
        
        /// <summary> Handle of the "has capacity" event. Set when the
        /// 		  queue has space for additional datablocks. (Number
        /// 		  of queued blocks less than m_uiCapacity). 
        /// 		   </summary>
        HANDLE					    m_hHasCapacity;
        
        /// <summary> Handle of the available event. Set when the queue is
        /// 		  non-empty to unblock waiters who are blocked in a call
        /// 		  to Pull on this channel.
        /// 		  </summary>
        HANDLE					    m_hAvailable;
        
        /// <summary> This channel's copy of the handle to the runtime terminate event.
        /// 		  Blocking calls must wait on this event as well as whatever other
        /// 		  wait object is semantically meaningful for the channel to ensure
        /// 		  that Task dispatch threads can unblock when the graph gets
        /// 		  torn down. 
        /// 		  </summary>
        HANDLE                      m_hRuntimeTerminateEvent;

        /// <summary> This channel's copy of the handle to the graph teardown event.
        /// 		  Some blocking calls must wait on this event as well as whatever other
        /// 		  wait object is semantically meaningful for the channel to ensure
        /// 		  that Task dispatch threads can unblock when the graph gets
        /// 		  stopped. Currently, it is OK to leave threads blocked on a
        ///           push/pull call if the graph is *stopping* or stopped, since
        ///           the user can start the graph again. We only want to unblock
        ///           calls when the caller is actually tearing the graph apart.
        /// 		  </summary>
        HANDLE                      m_hGraphTeardownEvent;
        
        /// <summary> This channel's copy of the handle to the graph stop event.
        /// 		  Some blocking calls must wait on this event as well as whatever other
        /// 		  wait object is semantically meaningful for the channel to ensure
        /// 		  that Task dispatch threads can unblock when the graph gets
        /// 		  stopped. Currently, it is OK to leave threads blocked on a
        ///           push/pull call if the graph is *stopping* or stopped, since
        ///           the user can start the graph again. We only want to unblock
        ///           calls when the caller is actually tearing the graph apart.
        /// 		  </summary>
        HANDLE                      m_hGraphStopEvent;

        /// <summary>   true if this object has block pool. </summary>
        BOOL                        m_bHasBlockPool;
        
        /// <summary> true if the queue is empty </summary>
        BOOL					    m_bEmpty;
        
        /// <summary> true if the queue has capacity </summary>
        BOOL					    m_bHasCapacity;
        
        /// <summary> true if datablocks are available in the queue. </summary>
        BOOL					    m_bAvailable;
        
        /// <summary> Reference count for this channel. </summary>
        ULONG					    m_uiRefCount;
        
        /// <summary> The default timeout for this channel. Calls to wait
        /// 		  will use this value (specified in milliseconds) 
        /// 		  unless those calls specify timeout values themselves.
        /// 		  Usually this should be set to INFINITE.
        /// 		  </summary>
        DWORD					    m_dwTimeout;
        
        /// <summary> The port connected to the source end of this
        /// 		  channel. Will be null for GraphInputChannels. 
        /// 		  </summary>
        Port *					    m_pSrcPort;

        /// <summary> The port connected to the destination end of this
        /// 		  channel. Will be null for GraphOutputChannels. 
        /// 		  </summary>
        Port *					    m_pDstPort;
        
        /// <summary> The name of the channel, user-supplied or a ptask-generated guid. </summary>
        char *					    m_lpszName;

        /// <summary>   true if the user specified a name, false if the runtime generated one for it. </summary>
        BOOL                        m_bUserSpecifiedName;
        
        /// <summary>   Predication descriptors for source and destination ends. </summary>
        CHANNELPREDICATIONDESC      m_vPredicators[NUMENDPOINTTYPES];

        /// <summary>	The view materialization policy for this channel. The runtime tries to start
        /// 			materialization of host- side views of datablocks that are likely are to be
        /// 			needed. The canonical example is when blocks are pushed into GraphOutputChannels,
        /// 			starting the materialization of the host-side view early can hide latency because
        /// 			(presumably) the user is interested in the result of the computation. On the
        /// 			other hand, if the user may discard blocks without looking at them, that work is
        /// 			wasted and adds to the length of the critical path. This member provides a
        /// 			mechanism for forcing the runtime to exercise a specific policy for this channel.
        /// 			</summary>
        VIEWMATERIALIZATIONPOLICY   m_viewMaterializationPolicy;

        /// <summary>   The propagated control code. </summary>
        CONTROLSIGNAL               m_luiPropagatedControlCode;

        /// <summary>   The propagated control code. </summary>
        CONTROLSIGNAL               m_luiInitialPropagatedControlCode;

        /// <summary>   The block throughput limit. </summary>
        UINT                        m_uiBlockTransitLimit;

        /// <summary>   The blocks delivered. </summary>
        UINT                        m_uiBlocksDelivered;

        /// <summary>   The maximum occupancy. </summary>
        UINT                        m_uiMaxOccupancy;

        /// <summary>   The cumulative occupancy. </summary>
        UINT                        m_uiCumulativeOccupancy;

        /// <summary> The control propagation source </summary>
        Port *                      m_pControlPropagationSource;

        /// <summary>   The is a trigger channel? </summary>
        BOOL                        m_bIsTriggerChannel;

        /// <summary>   Consumers of data on this channel want only the most recent
        ///             value pushed into it. If this member is true, on a push, 
        ///             any previously queued blocks will be drained and released.
        ///             To be used with caution, as it can upset the balance of blocks
        ///             required to keep a pipeline from stalling: typically channels with
        ///             this property set should either be connected to ports with the sticky
        ///             property set, or should be external (exposed channels e.g. outputs)
        ///             </summary>
        BOOL                        m_bWantMostRecentView;

        /// <summary>   The graph (set only if the trigger channel flag is on). </summary>
        Graph *                     m_pGraph;

        /// <summary> A uniquifier helper </summary>
        static ULONG                m_nKernelObjUniquifier;  
        
        /// <summary> True if rand() has been seeded with a call to srand.
        /// 		  </summary>
        static ULONG                m_bRandSeeded;

        /// <summary> true if graph-drawing utilities should draw this channel. </summary>
        bool					    m_draw; //Should this channel be drawn in the output graph

		/// <summary>	The set of all downstream consuming tasks. In the common case,
		/// 			this member is null: this is a cache of the result of 
		/// 			the GetDownstreamTasks() member function. </summary>
		std::set<Task*>*			m_pvDownstreamTasks;

		/// <summary>	The set of all downstream consuming accelerators which *must* touch blocks that
		/// 			flow through this channel. In the common case, this member is null: this is a
		/// 			cache of the result of the GetDownstreamMemorySpaces() member function.
		/// 			</summary>
		std::set<Accelerator*>*		m_pvMandatoryDownstreamAccelerators;

		/// <summary>	The set of all downstream consuming accelerators which *may* touch blocks that
		/// 			flow through this channel. In the common case, this member is null: this is a
		/// 			cache of the result of the GetDownstreamMemorySpaces() member function.
		/// 			</summary>
		std::set<Accelerator*>*		m_pvPotentialDownstreamAccelerators;

        /// <summary>   The channel profile. </summary>
        ChannelProfiler *           m_pChannelProfile;

    };

};
#endif