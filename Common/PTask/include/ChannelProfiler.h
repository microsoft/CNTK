///-------------------------------------------------------------------------------------------------
// file:	ChannelProfiler.h
//
// summary:	Declares the channel profiler class
///-------------------------------------------------------------------------------------------------

#ifndef __CHANNEL_PROFILER_H__
#define __CHANNEL_PROFILER_H__

#include "primitive_types.h"
#include <sstream>

namespace PTask {

    class Channel;

    typedef struct __channel_stats_t {

        /// <summary>   The block throughput limit. </summary>
        UINT                        uiBlockTransitLimit;

        /// <summary>   The blocks delivered. </summary>
        UINT                        uiBlocksDelivered;

        /// <summary>   The maximum occupancy. </summary>
        UINT                        uiMaxOccupancy;

        /// <summary>   The cumulative occupancy. </summary>
        UINT                        uiCumulativeOccupancy;

        /// <summary>   The occupancy samples. </summary>
        UINT                        uiOccupancySamples;

        /// <summary>   The capacity. </summary>
        UINT                        uiCapacity;

        /// <summary>   true if the channel is/was a pool owner. </summary>
        BOOL                        bPoolOwner;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets the stats object. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <param name="parameter1">   The first parameter. </param>
        ///-------------------------------------------------------------------------------------------------

        void Reset(
            VOID
            ) 
        {
            uiBlockTransitLimit = 0;
            uiBlocksDelivered = 0;
            uiMaxOccupancy = 0;
            uiCumulativeOccupancy = 0;
            uiOccupancySamples = 0;
            uiCapacity = 0;
            bPoolOwner = 0;
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the stats object with a current snapshot of the channel state. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        void Update(
            Channel * pChannel 
            ) 
        {
            uiBlockTransitLimit = pChannel->GetBlockTransitLimit();
            uiBlocksDelivered = pChannel->GetCumulativeBlockTransit();
            uiMaxOccupancy = pChannel->GetMaxOccupancy();
            uiCumulativeOccupancy = pChannel->GetCumulativeOccupancy();
            uiOccupancySamples = pChannel->GetOccupancySamples();
            uiCapacity = pChannel->GetCapacity();
            bPoolOwner = pChannel->IsPoolOwner();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        __channel_stats_t::__channel_stats_t(
            VOID
            ) 
        {
            Reset();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        __channel_stats_t::__channel_stats_t(
            Channel * pChannel 
            ) 
        {
            Update(pChannel);
        }

    } CHANNELSTATISTICS;

    class ChannelProfiler {

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        ChannelProfiler(Channel * pChannel);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~ChannelProfiler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes this object. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///
        /// <param name="bEnable">  true to enable, false to disable. </param>
        ///-------------------------------------------------------------------------------------------------

        static void Initialize(BOOL bEnable);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   De-initialises this object and frees any resources it is using. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Deinitialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports the given ss. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///
        /// <param name="ss">   [in,out] The ss. </param>
        ///-------------------------------------------------------------------------------------------------

        static void Report(std::ostream& ss);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Merge instance statistics. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void MergeInstanceStatistics();

    protected:

        Channel *                   m_pChannel;
        static BOOL                 m_bChannelProfile;
        static BOOL                 m_bChannelProfileInit;
        static CRITICAL_SECTION     m_csChannelStats;
        static std::map<std::string, std::map<std::string, CHANNELSTATISTICS*>*>  m_vChannelStats;
    };

};
#endif