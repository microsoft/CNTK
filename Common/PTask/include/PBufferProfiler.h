///-------------------------------------------------------------------------------------------------
// file:	PBufferProfiler.h
//
// summary:	Declares the buffer profiler class
///-------------------------------------------------------------------------------------------------

#ifndef _PBUFFER_PROFILER_H_
#define _PBUFFER_PROFILER_H_

#include "ptaskutils.h"
#include "primitive_types.h"
#include <deque>
#include <set>
#include <map>
#include "hrperft.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Buffer profiler. Class encapsulating profiling/statistics tools for PBuffers.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class PBufferProfiler {

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        PBufferProfiler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~PBufferProfiler();

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initialises the allocation profiler. </summary>
        ///
        /// <remarks>   Crossbac, 9/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Initialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinit allocation profiler. </summary>
        ///
        /// <remarks>   Crossbac, 9/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Deinitialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the allocation profiler data. </summary>
        ///
        /// <remarks>   Crossbac, 9/25/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Report(std::ostream &ios);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an allocation data. </summary>
        ///
        /// <remarks>   Crossbac, 9/25/2012. </remarks>
        ///
        /// <param name="uiAllocBytes"> The allocate in bytes. </param>
        /// <param name="uiAccID">      Identifier for the accumulate. </param>
        /// <param name="dLatency">     The latency. </param>
        ///-------------------------------------------------------------------------------------------------

        void Record(UINT uiAllocBytes, UINT uiAccID, double dLatency);

        std::map<UINT, UINT>            m_vAllocationSizes;
        std::map<UINT, UINT>            m_vAllocationDevices;
        std::map<UINT, double>          m_vAllocationLatencies;
        UINT                            m_nAllocations;
        CHighResolutionTimer *          m_pAllocationTimer;
        LPCRITICAL_SECTION              m_pcsAllocProfiler;
        UINT                            m_bAllocProfilerInit;        

    };

};
#endif

