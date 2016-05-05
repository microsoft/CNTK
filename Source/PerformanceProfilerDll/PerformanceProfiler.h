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
    profilerEvtForwardPass = 0,
    profilerEvtBackwardPass,
    profilerEvtInputProcessing,
    profilerEvtMPIProcessing,
    profilerEvtMPIWait,
    profilerEvtMPIThroughput,
    profilerEvtImageReaderThroughput,
    profilerEvtMax,
};


//
// Initialize all resources to enable profiling.
// Optionally provide a directory path where profiling files are saved,
// or nullptr to use the default.
//
void PERF_PROFILER_API ProfilerInit(const char* profilerDir);

//
// Measure time for either a fixed or a custom event.
// The *Begin calls return a stateId that is passed to ProfilerTimeEnd().
//
int PERF_PROFILER_API ProfilerTimeBegin(int eventId);
int PERF_PROFILER_API ProfilerTimeBegin(const char* description);
void PERF_PROFILER_API ProfilerTimeEnd(int stateId);

//
// Measure throughput given a bytes in an *Begin/*End block.
// Works with fixed or custom events.
//
int PERF_PROFILER_API ProfilerThroughputBegin(int eventId);
int PERF_PROFILER_API ProfilerThroughputBegin(const char* description);
void PERF_PROFILER_API ProfilerThroughputEnd(int stateId, long long bytes);

//
// Generate reports and release all resources.
//
void PERF_PROFILER_API ProfilerClose();


//
// Scoped profiler instantiation.
//
struct ProfilerContext
{
    ProfilerContext(const char* profilerDir = nullptr)
    {
        ProfilerInit(profilerDir);
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
    int m_stateId;

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

//
// Function scope profiling.
//
#define PROFILE_SCOPE(eventId)      ScopeProfile sp(eventId);
#define PROFILE_FUNCTION            ScopeProfile sp(__FUNCTION__);

}}}
