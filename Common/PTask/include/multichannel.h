//--------------------------------------------------------------------------------------
// File: multichannel.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _MULTI_CHANNEL_H_
#define _MULTI_CHANNEL_H_

#include <stdio.h>
#include <crtdbg.h>
#include <Windows.h>
#include "datablock.h"
#include "ReferenceCounted.h"
#include "channel.h"
#include "PTaskRuntime.h"
#include <map>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bundled channel class. Any block pushed into this channel is pushed into
    /// 			multiple bundled channels.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class MultiChannel : public Channel
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pDatablockTemplate">       [in] If non-null, the datablock template. </param>
        /// <param name="hRuntimeTerminateEvent">   Handle of the runtime terminate event. </param>
        /// <param name="hGraphTeardownEvt">        The graph teardown event. </param>
        /// <param name="hGraphStopEvent">          Handle of the graph stop event. </param>
        /// <param name="lpszChannelName">          [in] If non-null, name of the channel. </param>
        /// <param name="bHasBlockPool">            the has block pool. </param>
        ///-------------------------------------------------------------------------------------------------

        MultiChannel(
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

        virtual ~MultiChannel();

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
        /// <summary>
        ///     Sets the capacity of the channel, which is the maximum number of datablocks it can queue
        ///     before subsequent calls to push will block.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="nCapacity">    The capacity. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetCapacity(UINT nCapacity);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the capacity. </summary>
        ///
        /// <remarks>   Crossbac, 7/10/2013. </remarks>
        ///
        /// <returns>   The capacity. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetCapacity();

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
        /// <summary>   Coalesce channel. </summary>
        ///
        /// <remarks>   Crossbac, 1/20/2012. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        void CoalesceChannel(Channel * pChannel);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the coalesced channel map. </summary>
        ///
        /// <remarks>   crossbac, 4/18/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the coalesced channel map. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::map<UINT, Channel*>* GetCoalescedChannelMap();

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
		EnumerateDownstreamMemorySpaces(
			__inout	 std::set<Accelerator*>* pvMandatoryAccelerators,
			__inout  std::set<Accelerator*>* pvPotentialAccelerators
			);		

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
        /// <summary>   Query if this multi-channel has an exposed component channel. </summary>
        ///
        /// <remarks>   crossbac, 7/7/2014. </remarks>
        ///
        /// <returns>   true if exposed component channel, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasExposedComponentChannel();

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
        /// <summary>   Gets the downstream readonly port count. </summary>
        ///
        /// <remarks>   Crossbac, 2/6/2012. </remarks>
        ///
        /// <returns>   The downstream readonly port count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetDownstreamReadonlyPortCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the downstream writer port count. </summary>
        ///
        /// <remarks>   Crossbac, 2/6/2012. </remarks>
        ///
        /// <returns>   The downstream writer port count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetDownstreamWriterPortCount();

        /// <summary>   The channel map. </summary>
        std::map<UINT, Channel*>    m_pChannelMap;

    };

};
#endif