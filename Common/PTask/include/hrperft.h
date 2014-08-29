/********************************************************
* hrperft.h
**********************************************************/

#ifndef _HRPERFT_H_
#define _HRPERFT_H_

// performance timers are architecture and platform
// specific. Need to define a routine to access
// the perf counters on whatever processor is in use here:
#include "windows.h"
typedef double ctrtype;
#define hpfresult(x) x.QuadPart
#define query_hpc(x) QueryPerformanceCounter(x)
#define query_freq(x) QueryPerformanceFrequency(x)
typedef long (__stdcall *LPFNtQuerySystemTime)(PLARGE_INTEGER SystemTime);

typedef enum gran_t {
    gran_nanosec,
    gran_usec,
    gran_msec,
    gran_sec 
} hpf_granularity;

///-------------------------------------------------------------------------------------------------
/// <summary>   High resolution timer. 
/// 			For collecting performance measurements.
/// 			</summary>
///
/// <remarks>   Crossbac, 12/23/2011. </remarks>
///-------------------------------------------------------------------------------------------------

class CHighResolutionTimer {
public:

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="gran"> The granularity of the timer
    /// 					(seconds or milliseconds). </param>
    ///-------------------------------------------------------------------------------------------------

    CHighResolutionTimer(hpf_granularity gran);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ~CHighResolutionTimer(void);

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

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the tick count. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    __int64 tickcnt();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets this timer. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void reset();

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the time elapsed since the
    /// 			last reset. Optionally, reset the timer
    /// 			as a side-effect of the query. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="reset">    true to reset. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    double elapsed(bool reset);

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


protected:

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

    DWORD delta_milliseconds(LARGE_INTEGER lEarly, LARGE_INTEGER lLate);};

#endif
