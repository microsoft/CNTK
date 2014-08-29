//--------------------------------------------------------------------------------------
// File: InternalChannel.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _INTERNAL_CHANNEL_H_
#define _INTERNAL_CHANNEL_H_

#include <stdio.h>
#include <crtdbg.h>
#include <Windows.h>
#include "datablock.h"
#include "channel.h"
#include <deque>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   InternalChannel. Channel subclass specialized for Task-Task communication. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class InternalChannel : public Channel {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pGraph">                   [in,out] If non-null, the graph. </param>
        /// <param name="pDatablockTemplate">       [in,out] If non-null, the datablock template. </param>
        /// <param name="hRuntimeTerminateEvent">   Handle of the graph terminate event. </param>
        /// <param name="hGraphTeardownEvt">        The graph teardown event. </param>
        /// <param name="hGraphStopEvent">          Handle of the graph stop event. </param>
        /// <param name="lpszChannelName">          [in,out] If non-null, name of the channel. </param>
        /// <param name="bHasBlockPool">            the has block pool. </param>
        ///-------------------------------------------------------------------------------------------------

        InternalChannel(
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
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~InternalChannel();

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