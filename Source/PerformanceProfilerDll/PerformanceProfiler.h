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
    /*  1 */ profilerEvtNodeElementTimes,
    /*  2 */ profilerEvtNodeExp,
    /*  3 */ profilerEvtNodeFutureValue,
    /*  4 */ profilerEvtNodeGatherPacked,
    /*  5 */ profilerEvtNodeHardmax,
    /*  6 */ profilerEvtNodeIf,
    /*  7 */ profilerEvtNodeInputValue,
    /*  8 */ profilerEvtNodeLearnableParameter,
    /*  9 */ profilerEvtNodeLog,
    /* 10 */ profilerEvtNodeMinus,
    /* 11 */ profilerEvtNodePackedIndex,
    /* 12 */ profilerEvtNodePass,
    /* 13 */ profilerEvtNodePastValue,
    /* 15 */ profilerEvtNodePlus,
    /* 17 */ profilerEvtNodeReciprocal,
    /* 18 */ profilerEvtNodeReduceElements,
    /* 19 */ profilerEvtNodeReshape,
    /* 20 */ profilerEvtNodeRowStack,
    /* 21 */ profilerEvtSEQTraversalFlowControlNode,
    /* 22 */ profilerEvtScatterPacked,
    /* 23 */ profilerEvtSigmoid,
    /* 24 */ profilerEvtSlice,
    /* 25 */ profilerEvtSoftmax,
    /* 26 */ profilerEvSumColumnElements,
    /* 28 */ profilerEvtTanh,
    /* 29 */ profilerEvtTimes,
    /* 30 */ profilerEvtTransposeTimes,
    /* 31 */ profilerEvtWhere,
    /* 32 */ profilerEvtNodeCrossEntropyWithSoftmax,
    /* 33 */ profilerEvtNodeLogSoftmax,
    profilerEvtNodeUnknown, // Unknown node
    profilerEvtMax, // Last item
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
// Caller maintained record to measure custom events.
//
struct ProfilerCustomTimeEventRecord
{
    long long       beginClock;
    unsigned int    threadId;
    char*           eventDescription;
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
// If the event does not need to be recorded, call ProfilerTimeCancel().
//
int PERF_PROFILER_API ProfilerTimeBegin(int eventId);
void PERF_PROFILER_API ProfilerTimeEnd(int stateId);
void PERF_PROFILER_API ProfilerTimeCancel(int stateId);
unsigned long long PERF_PROFILER_API ProfilerTimeBegin(const char* eventDescription);
void PERF_PROFILER_API ProfilerTimeEnd(unsigned long long stateId);


//
// Measure throughput given a bytes in an *Begin/*End block.
// The ThroughputEventRecord is meaintained by the caller.
// If ProfilerThroughputEnd is not called, the event is not recorded.
//
void PERF_PROFILER_API ProfilerThroughputBegin(int eventId, ProfilerThroughputEventRecord* throughputEventRecord);
void PERF_PROFILER_API ProfilerThroughputEnd(long long bytes, ProfilerThroughputEventRecord* throughputEventRecord);

//
// Generate reports and release all resources.
//
void PERF_PROFILER_API ProfilerClose();


//
// Scoped profiler instantiation.
//
struct ProfilerContext
{
    void Init(const char* profilerDir = nullptr, const float delaySeconds = 0.0f, const unsigned long long customEventBufferBytes = (16 * 1024 * 1024))
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
    int m_stateId;

public:
    ScopeProfile(int eventId)
    {
        m_stateId = ProfilerTimeBegin(eventId);
    }

    ~ScopeProfile()
    {
        ProfilerTimeEnd(m_stateId);
    }
};

//
// Function scope profiling.
//
#define PROFILE_SCOPE(eventId)      ScopeProfile __sp(eventId);


//
// Scoped custom event profiling.
//
class CustomScopeProfile
{
    unsigned long long m_stateId;

public:
    CustomScopeProfile(const char* description)
    {
        m_stateId = ProfilerTimeBegin(description);
    }

    ~CustomScopeProfile()
    {
        ProfilerTimeEnd(m_stateId);
    }
};

//
// Function name scope profiling.
//
#define PROFILE_FUNCTION        CustomScopeProfile __csp(__FUNCTION__);


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

}}}
