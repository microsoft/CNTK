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
    // Main thread events
    profilerSepMainThread = 0,
    profilerSepSpace0,

    profilerEvtMainEpoch,
    profilerEvtMainMinibatch,
    profilerEvtMainGetMinibatch,
    profilerEvtMainFB,
    profilerEvtMainGradient,
    profilerEvtMainWeights,
    profilerEvtMainPost,

    // MPI/Gradient aggregation multithreaded events
    profilerSepSpace1,
    profilerSepMPI,
    profilerSepSpace2,

    profilerEvtGradient1,
    profilerEvtGradientAsyncComm11,
    profilerEvtGradientWaitGradients1,
    profilerEvtGradientWaitHeaders1,
    profilerEvtGradientAsyncComm21,
    profilerEvtGradientWaitAggGradients1,
    profilerEvtGradientWaitAggHeaders1,
    profilerEvtGradientWaitCompletion1,
    
    profilerEvtGradient32,
    profilerEvtGradientAsyncComm132,
    profilerEvtGradientWaitHeaders32,
    profilerEvtGradientAsyncComm232,
    profilerEvtGradientWaitGradients32,
    profilerEvtGradientWaitCompletion32,

    // Data reader events
    profilerSepSpace3,
    profilerSepDataReader,
    profilerSepSpace4,

    profilerEvtReadMinibatch,
    profilerEvtZipReaderThroughput,

    profilerEvtMax
};


//
// Initialize all resources to enable profiling.
// profilerDir: Directory where the profiler logs will be saved. nullptr for default location.
// customEventBufferBytes: Bytes to allocate for the custom event buffer.
// logSuffix: Suffix string to append to log files.
// syncGpu: Wait for GPU to complete processing for each profiling event.
//
void PERF_PROFILER_API ProfilerInit(const char* profilerDir, const unsigned long long customEventBufferBytes,
    const char* logSuffix, const bool syncGpu);


//
// Enable/disable profiling.
// By default, profiling is disabled after a ProfilerInit call.
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
// CUDA kernel profiling.
// CUDA kernels are profiled using a single call to this function.
//
void PERF_PROFILER_API ProfilerCudaTimeEnd(const float deltaSeconds, const char* eventDescription);


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
struct ProfilerContext
{
    void Init(const char* profilerDir = nullptr, const unsigned long long customEventBufferBytes = (32 * 1024 * 1024), const char* logSuffix = "", const bool syncGpu = false)
    {
        ProfilerInit(profilerDir, customEventBufferBytes, logSuffix, syncGpu);
    }

    ~ProfilerContext()
    {
        ProfilerClose();
    }
};


//
// Scoped time profiling.
//
struct ScopeProfile
{
    ScopeProfile(int eventId)
    {
        m_eventId = eventId;
        m_description = nullptr;
        m_stateId = ProfilerTimeBegin();
    }

    ScopeProfile(const char* description)
    {
        m_description = (char*)description;
        m_stateId = ProfilerTimeBegin();
    }

    ~ScopeProfile()
    {
        if (m_description)
        {
            ProfilerTimeEnd(m_stateId, m_description);
        }
        else
        {
            ProfilerTimeEnd(m_stateId, m_eventId);
        }
    }

private:
    unsigned long long  m_stateId;
    int                 m_eventId;
    char*               m_description;
};

#define PROFILE_SCOPE(eventId)      ScopeProfile __sp##eventId(eventId);
#define PROFILE_FUNCTION            ScopeProfile __fsp(__FUNCTION__);



//
// Scoped throughput profiling.
//
struct ScopeThroughput
{
    ScopeThroughput(int eventId, long long bytes)
    {
        m_bytes = bytes;
        m_eventId = eventId;
        m_stateId = ProfilerThroughputBegin();
    }

    ~ScopeThroughput()
    {
        ProfilerThroughputEnd(m_stateId, m_eventId, m_bytes);
    }

private:
    unsigned long long              m_stateId;
    int                             m_eventId;
    long long                       m_bytes;
};

#define THROUGHPUT_SCOPE(eventId, bytes)    ScopeThroughput __st##eventId(eventId, bytes);


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
        m_iterationCntTotal = 0;
        SyncCudaScopeSetFlags(false, false);
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
