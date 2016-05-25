//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Real-time thread-safe profiler that generats a summary report and a detail profile log.
// The profiler is highly performant and light weight and meant to be left on all the time.
//

#pragma once

#include "Basics.h"
#include <stdio.h>

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
    profilerEvtEpoch = 0,
    profilerEvtForwardPass,
    profilerEvtBackwardPass,
    profilerEvtGradientAggregation,
    profilerEvtWeightUpdate,
    profilerEvtInputProcessing,
    profilerEvtImageDecoding,
    profilerEvtMPIProcessing,
    profilerEvtMPIWait,
    profilerEvtMPIThroughput,
    profilerEvtImageReaderThroughput,
    profilerEvtMax,
};

//
// Caller-maintained record to measure throughput events.
//
struct ProfilerThroughputEventRecord
{
    long long       beginClock;
    int             eventId;
};

//
// Initialize all resources to enable profiling.
// profilerDir: Directory where the profiler logs will be saved. nullptr for default location.
// delaySeconds: Number of seconds since this call to wait to start profiling.
// customEventBufferBytes: Bytes to allocate for the custom event buffer.
//
void PERF_PROFILER_API ProfilerInit(const char* profilerDir, const float delaySeconds, const unsigned long long customEventBufferBytes);

//
// Measure time for either a fixed or a custom event.
// The *Begin call returns a stateId that is passed to ProfilerTimeEnd().
// If the fixed event does not need to be recorded, call ProfilerTimeCancel() before ProfilerTimeEnd().
//
unsigned long long PERF_PROFILER_API ProfilerTimeBegin(const int eventId);
void PERF_PROFILER_API ProfilerTimeEnd(const unsigned long long stateId);
void PERF_PROFILER_API ProfilerTimeCancel(unsigned long long* stateId);

unsigned long long PERF_PROFILER_API ProfilerTimeBegin(const char* eventDescription);


//
// CUDA kernel profiling.
// The cuda event calls must be called around the kernel, and the total kernel time provided
// in deltaSeconds.
//
unsigned long long PERF_PROFILER_API ProfilerCudaTimeBegin(const char* eventDescription);
void PERF_PROFILER_API ProfilerCudaTimeEnd(const float deltaSeconds, const unsigned long long stateId);


//
// Measure throughput given a bytes in an *Begin/*End block.
// The ThroughputEventRecord is meaintained by the caller.
// If ProfilerThroughputEnd is not called, the event is not recorded.
//
void PERF_PROFILER_API ProfilerThroughputBegin(const int eventId, ProfilerThroughputEventRecord* profilerThroughputEventRecord);
void PERF_PROFILER_API ProfilerThroughputEnd(const long long bytes, ProfilerThroughputEventRecord* profilerThroughputEventRecord);

//
// Generate reports and release all resources.
//
void PERF_PROFILER_API ProfilerClose();


//
// Scoped profiler instantiation.
//
struct ProfilerContext
{
    void Init(const char* profilerDir = nullptr, const float delaySeconds = 0.0f, const unsigned long long customEventBufferBytes = (32 * 1024 * 1024))
    {
        ProfilerInit(profilerDir, delaySeconds, customEventBufferBytes);
    }

    ~ProfilerContext()
    {
        ProfilerClose();
    }
};


//
// Scoped time profiling.
//
class ScopeProfile
{
    unsigned long long m_stateId;

public:
    ScopeProfile(int eventId)
    {
        m_stateId = ProfilerTimeBegin(eventId);
    }

    ScopeProfile(const char* description)
    {
        m_stateId = ProfilerTimeBegin(description);
    }

    ~ScopeProfile()
    {
        ProfilerTimeEnd(m_stateId);
    }
};

#define PROFILE_SCOPE(eventId)      ScopeProfile __sp(eventId);
#define PROFILE_FUNCTION            ScopeProfile __fsp(__FUNCTION__);



//
// Scoped throughput profiling.
//
class ScopeThroughput
{
    ProfilerThroughputEventRecord   m_throughputEventRecord;
    long long                       m_bytes;

public:
    ScopeThroughput(int eventId, long long bytes)
    {
        m_bytes = bytes;
        ProfilerThroughputBegin(eventId, &m_throughputEventRecord);
    }

    ~ScopeThroughput()
    {
        ProfilerThroughputEnd(m_bytes, &m_throughputEventRecord);
    }
};

#define THROUGHPUT_SCOPE(eventId, bytes)    ScopeThroughput __st(eventId, bytes);


//
// CUDA profiling helpers.
//

void PERF_PROFILER_API SyncCudaScopeSetFlags(bool syncEnabled, bool profilingEnabled);
void PERF_PROFILER_API SyncCudaScopeGetFlags(bool& syncEnabled, bool& profilingEnabled);

struct CudaProfilerTimer
{
    //
    // Setup the CUDA profiler.
    // enable: flag indicating if profiler is enabled
    // syncIterations: Number of iterations between profiling & sync (ex: 100 means that we profile every 100 iterations)
    // maxIterations: Stop syncing & profiling after this many iterations. -1 indicates to never stop.
    //                (ex: 1000 means that we stop after 1000 iterations)
    //
    CudaProfilerTimer(bool enable, int syncIterations, int maxIterations)
    {
        m_enable = enable;
        m_syncIterations = syncIterations;
        m_maxIterations = maxIterations;
        m_iterationCnt = 0;
        SyncCudaScopeSetFlags(true, true);
    }

    //
    // Main function that enables profiling, to be called at the beggining of each iteration.
    //
    void Update()
    {
        if (!m_enable) return;

        if (m_iterationCnt == m_syncIterations && (m_iterationCntTotal < m_maxIterations || m_maxIterations == -1))
        {
            SyncCudaScopeSetFlags(true, true);
            m_iterationCnt = 0;
        }
        else
        {
            SyncCudaScopeSetFlags(false, false);
        }

        m_iterationCnt++;
        m_iterationCntTotal++;
    }

private:
    bool    m_enable;
    int     m_syncIterations;
    int     m_maxIterations;
    int     m_iterationCnt;
    int     m_iterationCntTotal;
};

}}}
