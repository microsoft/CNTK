///-------------------------------------------------------------------------------------------------
// file:	shrperft.h
//
// summary:	Declares a thread-safe high resolution timer utility
///-------------------------------------------------------------------------------------------------

#ifndef _SHRPERFT_H_
#define _SHRPERFT_H_
#include "hrperft.h"

// performance timers are architecture and platform
// specific. The CHighResolutionTimer class defined in 
// hrperft.h is lightweight but not thread-safe. 
// This version is thread-safe, but will have higher 
// overheads due to synchronization...Use this only for
// cases where measurements require a global time line
// across multiple threads. 


///-------------------------------------------------------------------------------------------------
/// <summary>   High resolution timer. 
/// 			For collecting performance measurements.
/// 			</summary>
///
/// <remarks>   Crossbac, 12/23/2011. </remarks>
///-------------------------------------------------------------------------------------------------

class CSharedPerformanceTimer {
public:

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="gran"> The granularity of the timer
    /// 					(seconds, milliseconds, micro-seconds). </param>
    ///-------------------------------------------------------------------------------------------------

    CSharedPerformanceTimer(hpf_granularity gran, bool bStart);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ~CSharedPerformanceTimer(void);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets this timer. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void reset();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the time elapsed since the
    /// 			last reset. For compatibility with hrperft, the reset parameter is
    ///             present, but will assert. Objects of this class should never be reset. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="reset">    true to reset.  (ignored, will assert if true in debug mode)</param>
    ///
    /// <returns>   The elapsed time since the timer started </returns>
    ///-------------------------------------------------------------------------------------------------

    double elapsed(bool reset=false);

protected:

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the tick count. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    __int64 tickcnt();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queries the system time. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="li">   The li. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL query_system_time(PLARGE_INTEGER li);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the tick frequency of the underlying
    /// 			counter primitive. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    double tickfreq();

    /// <summary>   lock. </summary>
    CRITICAL_SECTION m_cs;

    /// <summary> The granularity of the timer,
    /// 		  either seconds or milliseconds 
    /// 		  </summary>
    hpf_granularity m_gran;
    
    /// <summary> the value of the underlying 
    /// 		  timing primitive at the time the 
    /// 		  timer was last reset.</summary>
    __int64 m_start; 
    
    /// <summary> The frequency of the underlying
    /// 		  timing primitive </summary>
    double m_freq;

    /// <summary> Module for windows DLL for querying
    /// 		  system time getting perf counter
    /// 		  frequency. 
    /// 		  </summary>
    HMODULE m_hModule;
    
    /// <summary> Function pointer for querying
    /// 		  system time 
    /// 		  </summary>
    LPFNtQuerySystemTime m_lpfnQuerySystemTime;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free resources allocated to support
    /// 			query of system time. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void free_query_system_time();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialises the query system time. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    LPFNtQuerySystemTime init_query_system_time();
    
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the difference in milliseconds. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lEarly">   The early. </param>
    /// <param name="lLate">    The late. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD delta_milliseconds(LARGE_INTEGER lEarly, LARGE_INTEGER lLate);
};

#endif
