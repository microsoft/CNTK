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
    /*  1 */ profilerEvtNodeElementTimesF,
    /*  1 */ profilerEvtNodeElementTimesB,
    /*  2 */ profilerEvtNodeExpF,
    /*  2 */ profilerEvtNodeExpB,
    /*  3 */ profilerEvtNodeFutureValueF,
    /*  3 */ profilerEvtNodeFutureValueB,
    /*  4 */ profilerEvtNodeGatherPackedF,
    /*  4 */ profilerEvtNodeGatherPackedB,
    /*  5 */ profilerEvtNodeHardmaxF,
    /*  5 */ profilerEvtNodeHardmaxB,
    /*  6 */ profilerEvtNodeIfF,
    /*  6 */ profilerEvtNodeIfB,
    /*  7 */ profilerEvtNodeInputValueF,
    /*  7 */ profilerEvtNodeInputValueB,
    /*  8 */ profilerEvtNodeLearnableParameterF,
    /*  8 */ profilerEvtNodeLearnableParameterB,
    /*  9 */ profilerEvtNodeLogF,
    /*  9 */ profilerEvtNodeLogB,
    /* 10 */ profilerEvtNodeMinusF,
    /* 10 */ profilerEvtNodeMinusB,
    /* 11 */ profilerEvtNodePackedIndexF,
    /* 11 */ profilerEvtNodePackedIndexB,
    /* 12 */ profilerEvtNodePassF,
    /* 12 */ profilerEvtNodePassB,
    /* 13 */ profilerEvtNodePastValueF,
    /* 13 */ profilerEvtNodePastValueB,
    /* 15 */ profilerEvtNodePlusF,
    /* 15 */ profilerEvtNodePlusB,
    /* 17 */ profilerEvtNodeReciprocalF,
    /* 17 */ profilerEvtNodeReciprocalB,
    /* 18 */ profilerEvtNodeReduceElementsF,
    /* 18 */ profilerEvtNodeReduceElementsB,
    /* 19 */ profilerEvtNodeReshapeF,
    /* 19 */ profilerEvtNodeReshapeB,
    /* 20 */ profilerEvtNodeRowStackF,
    /* 20 */ profilerEvtNodeRowStackB,
    /* 21 */ profilerEvtSEQTraversalFlowControlNodeF,
    /* 21 */ profilerEvtSEQTraversalFlowControlNodeB,
    /* 22 */ profilerEvtScatterPackedF,
    /* 22 */ profilerEvtScatterPackedB,
    /* 23 */ profilerEvtSigmoidF,
    /* 23 */ profilerEvtSigmoidB,
    /* 24 */ profilerEvtSliceF,
    /* 24 */ profilerEvtSliceB,
    /* 25 */ profilerEvtSoftmaxF,
    /* 25 */ profilerEvtSoftmaxB,
    /* 26 */ profilerEvSumColumnElementsF,
    /* 26 */ profilerEvSumColumnElementsB,
    /* 28 */ profilerEvtTanhF,
    /* 28 */ profilerEvtTanhB,
    /* 29 */ profilerEvtTimesF,
    /* 29 */ profilerEvtTimesB,
    /* 30 */ profilerEvtTransposeTimesF,
    /* 30 */ profilerEvtTransposeTimesB,
    /* 31 */ profilerEvtWhereF,
    /* 31 */ profilerEvtWhereB,
    /* 32 */ profilerEvtNodeCrossEntropyWithSoftmaxF,
    /* 32 */ profilerEvtNodeCrossEntropyWithSoftmaxB,
    /* 33 */ profilerEvtNodeLogSoftmaxF,
    /* 33 */ profilerEvtNodeLogSoftmaxB,
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
