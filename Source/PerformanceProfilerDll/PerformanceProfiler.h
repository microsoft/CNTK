//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Real-time thread-safe profiler that generates a summary report and a detail profile log.
// The profiler is highly performant and lightweight. Profiling a single event introduces an overhead
// of approximately 100 ns.
//
// Profiler Usage
//
// To initialize and tear down the profiler, call ProfilerInit() and ProfilerClose(). The scoped
// object, ProfilerContext can also be used for managing the lifetime of the profiler. The profiler
// works by accumulating events in a pre-allocated buffer, up until the buffer is full. At the
// time when the profiler is torn down, a summary report and a detailed log file is written to disk.
//
// When profiling code, two types of events can be used - fixed or custom. A fixed event is
// predefined in the ProfilerEvents enum and by the FixedEventDesc struct. A custom event is
// simply a string.
//
// To profile a section of code, call ProfilerTimeBegin() and ProfilerTimeEnd(), or the scoped
// object ScopeProfile. When the GPU sync flag is set in the FixedEventDesc and in ProfilerInit(),
// the GPU will be synced at the time ProfilerTimeEnd() is called, for fixed events only. For
// custom events, ProfilerSyncGpu() can be called to sync the GPU.
//
// When there is a need to profile I/O bandwidth (or throughput), the ProfilerThroughputBegin()
// and ProfilerThroughputEnd() calls should be used. The throughput APIs can only be used
// with fixed events.
//
// CNTK specifics
//
// The profiler is turned off during the very first epoch to avoid polluting profile data with
// times that are typically larger (warm-up).
//

#pragma once

#ifdef CNTK_UWP // UWP does not support performance profiler

#define PROFILE_SCOPE(eventId)      /*nothing*/

#else

#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef _WIN32
#if defined(PERFORMANCE_PROFILER_DLL)
#define PERF_PROFILER_API __declspec(dllexport)
#else
#define PERF_PROFILER_API __declspec(dllimport)
#endif
#else
#define PERF_PROFILER_API
#endif

//
// Fixed/predetermined profiler events. These appear in the summary report.
//
enum ProfilerEvents
{
    // Main thread header (dummy events)
    profilerSepMainThread = 0,
    profilerSepSpace0,

    // Main thread events
    profilerEvtMainEpoch,                   // Train epoch loop time
    profilerEvtMainMinibatch,               // One minibatch loop time
    profilerEvtMainGetMinibatch,            // GetMinibatch() function time
    profilerEvtMainFB,                      // Forward + Backward pass time
    profilerEvtMainGradient,                // Gradient aggregation time
    profilerEvtMainWeights,                 // Weight update time
    profilerEvtMainPost,                    // Remainder time in minibatch loop

    // Data reader header (dummy events)
    profilerSepSpace1,
    profilerSepDataReader,
    profilerSepSpace2,

    // Data reader events
    profilerEvtPrefetchMinibatch,           // Prefetching the next minibatch in a background thread

    profilerEvtMax
};


//
// Initialize all resources to enable profiling.
// profilerDir: Directory where the profiler logs will be saved.
// customEventBufferBytes: Bytes to allocate for the custom event buffer.
// logSuffix: Suffix string to append to log files.
// syncGpu: Wait for GPU to complete processing for each profiling event.
//
void PERF_PROFILER_API ProfilerInit(const std::wstring& profilerDir, const unsigned long long customEventBufferBytes,
    const std::wstring& logSuffix, const bool syncGpu);


//
// Enable/disable profiling.
// By default, profiling is disabled after a ProfilerInit call.
// This can be used to temporarily turn profiling on/off during execution.
//
void PERF_PROFILER_API ProfilerEnable(bool enable);


//
// Measure either a fixed or custom event time.
// ProfilerTimeBegin() returns a stateId that is passed to ProfilerTimeEnd().
// If ProfilerTimeEnd() is not called, the event is not recorded.
//
long long PERF_PROFILER_API ProfilerTimeBegin();
void PERF_PROFILER_API ProfilerTimeEnd(const long long stateId, const int eventId);
void PERF_PROFILER_API ProfilerTimeEnd(const long long stateId, const char* eventDescription);

//
// Conditionally sync the GPU if the syncGPU flag is set. This only needs to be excplicitly
// called for custom events.
//
void PERF_PROFILER_API ProfilerSyncGpu();


//
// Measure throughput given the number of bytes.
// ProfilerThroughputBegin() returns a stateId that is passed to ProfilerThroughputEnd().
// If ProfilerThroughputEnd() is not called, the event is not recorded.
//
long long PERF_PROFILER_API ProfilerThroughputBegin();
void PERF_PROFILER_API ProfilerThroughputEnd(const long long stateId, const int eventId, const long long bytes);


//
// Generate reports and release all resources.
//
void PERF_PROFILER_API ProfilerClose();


//
// Scoped profiler instantiation.
//
struct PERF_PROFILER_API ProfilerContext
{
    void Init(const std::wstring& profilerDir = L"", const unsigned long long customEventBufferBytes = (32 * 1024 * 1024), const std::wstring& logSuffix = L"", const bool syncGpu = false);
    ~ProfilerContext();
};


//
// Scoped time profiling.
//
struct PERF_PROFILER_API ScopeProfile
{
    ScopeProfile(int eventId);
    ScopeProfile(const char* description);
    ~ScopeProfile();

private:
    unsigned long long  m_stateId;
    int                 m_eventId;
    const char*         m_description;
};

#define PROFILE_SCOPE(eventId)      ScopeProfile __sp##eventId(eventId);
#define PROFILE_FUNCTION            ScopeProfile __fsp(__FUNCTION__);



//
// Scoped throughput profiling.
//
struct PERF_PROFILER_API ScopeThroughput
{
    ScopeThroughput(int eventId, long long bytes);
    ~ScopeThroughput();

private:
    unsigned long long  m_stateId;
    int                 m_eventId;
    long long           m_bytes;
};

#define THROUGHPUT_SCOPE(eventId, bytes)    ScopeThroughput __st##eventId(eventId, bytes);

}}}

#endif // CNTK_UWP