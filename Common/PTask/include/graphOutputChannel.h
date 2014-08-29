//--------------------------------------------------------------------------------------
// File: GraphOutputChannel.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_OUTPUT_CHANNEL_H_
#define _GRAPH_OUTPUT_CHANNEL_H_

#include <stdio.h>
#include <crtdbg.h>
#include <Windows.h>
#include "datablock.h"
#include <deque>

namespace PTask {

    class GraphOutputChannel : public Channel  {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pGraph">                   [in,out] If non-null, the graph. </param>
        /// <param name="pDatablockTemplate">       [in,out] If non-null, the p. </param>
        /// <param name="hRuntimeTerminateEvent">   Handle of the terminate. </param>
        /// <param name="hGraphTeardownEvt">        Handle of the stop. </param>
        /// <param name="hGraphStopEvent">          The graph stop event. </param>
        /// <param name="lpszChannelName">          [in,out] If non-null, name of the channel. </param>
        /// <param name="bHasBlockPool">            the has block pool. </param>
        ///-------------------------------------------------------------------------------------------------

        GraphOutputChannel(
            __in Graph * pGraph,
            __in DatablockTemplate * pDatablockTemplate, 
            __in HANDLE hRuntimeTerminateEvent,
            __in HANDLE hGraphTeardownEvt, 
            __in HANDLE hGraphStopEvent, 
            __in char * lpszChannelName,
            __in BOOL bHasBlockPool
            );

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
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~GraphOutputChannel();

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

    };

};
#endif