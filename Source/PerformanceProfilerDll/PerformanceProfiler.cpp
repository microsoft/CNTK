//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Real-time thread-safe profiler that generates a summary report and a detail profile log.
// The profiler is highly performant and lightweight. Profiling a single event introduces an overhead
// of approximately 100 ns.
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif
#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _

#include "PerformanceProfiler.h"
#include "Basics.h"
#include "fileutil.h"
#include "TimerUtility.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdio.h>
#ifndef CPUONLY
#include <cuda_runtime_api.h>
#endif

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/syscall.h>
#endif 


namespace Microsoft { namespace MSR { namespace CNTK {

//
// Fixed profiler event descriptions
//
enum FixedEventType
{
    profilerEvtTime = 0,
    profilerEvtThroughput,
    profilerEvtSeparator
};

struct FixedEventDesc
{
    char            eventDescription[64];
    FixedEventType  eventType;
    bool            syncGpu;
};

static const FixedEventDesc c_fixedEvtDesc[profilerEvtMax] = {
    { "Main Thread", profilerEvtSeparator, false },                 // profilerSepMainThread
    { "", profilerEvtSeparator, false },                            // profilerSepSpace0

    { "Epoch", profilerEvtTime, false },                            // profilerEvtMainEpoch
    { "_Minibatch Iteration", profilerEvtTime, false },             // profilerEvtMainMinibatch
    { "__Get Minibatch", profilerEvtTime, true },                   // profilerEvtMainGetMinibatch
    { "__Forward + Backward", profilerEvtTime, true },              // profilerEvtMainFB
    { "__Gradient Aggregation", profilerEvtTime, true },            // profilerEvtMainGradient
    { "__Weight Update", profilerEvtTime, true },                   // profilerEvtMainWeights
    { "__Post Processing", profilerEvtTime, true },                 // profilerEvtMainPost

    { "", profilerEvtSeparator, false },                            // profilerSepSpace1
    { "Data Reader", profilerEvtSeparator, false },                 // profilerSepDataReader
    { "", profilerEvtSeparator, false },                            // profilerSepSpace2

    { "Prefetch Minibatch", profilerEvtTime, false },               // profilerEvtPrefetchMinibatch
};


struct FixedEventRecord
{
    int             cnt;          // event count
    long long       sum;          // time (ns) or throughput (kB/s)
    double          sumsq;        // sum of squares
    long long       min;          // time (ns) or throughput (kB/s)
    long long       max;          // time (ns) or throughput (kB/s)
    long long       totalBytes;   // used only for throughput events
};

//
// The custom event record is a variable size datastructure in memory:
// NULL terminated description string, followed by CustomEventRecord struct
//
struct CustomEventRecord
{
    long long       beginClock;
    long long       endClock;
    unsigned int    threadId;
};


//
// Global state of the profiler
//
struct ProfilerState
{
    bool                    enabled;                     // Profiler enabled (active)
    bool                    syncGpu;                     // Sync GPU per each profiling event
    bool                    cudaSyncEnabled;             // Runtime state of CUDA kernel sync
    std::wstring            profilerDir;                 // Directory where reports/logs are saved
    std::wstring            logSuffix;                   // Suffix to append to report/log file names
    FixedEventRecord        fixedEvents[profilerEvtMax]; // Profiling data for each fixed event
    bool                    customEventBufferFull;       // Is custom event buffer full?
    unsigned long long      customEventBufferBytes;      // Number of bytes allocated for the custom event buffer
    unsigned long long      customEventOffset;           // Offset to current place in buffer
    unique_ptr<char[]>      customEventBuffer;           // Pointer to custom event buffer
};


// We support one global instance of the profiler
static unique_ptr<ProfilerState> g_profilerState;

// Mutex controlling access to g_profilerState
static std::mutex g_mutex;

// Forward declarations
unsigned int GetThreadId();

void ProfilerGenerateReport(const std::wstring& fileName, struct tm* timeInfo);
void FormatTimeStr(char* str, size_t strLen, double value);
void FormatThroughputStr(char* str, size_t strLen, double value);
void FormatBytesStr(char* str, size_t strLen, long long bytes);
void ProfilerGenerateDetailFile(const std::wstring& fileName);


double TicksToSeconds(long long ticks)
{
    long long ticksPerSec = Clock::GetTicksPerSecond();
    return static_cast<double>(ticks) / ticksPerSec;
}

double TicksSqToSecondsSq(double ticksSq)
{
    long long ticksPerSec = Clock::GetTicksPerSecond();
    return ticksSq / ticksPerSec / ticksPerSec;
}


//
// Initialize all resources to enable profiling.
// profilerDir: Directory where the profiler logs will be saved.
// customEventBufferBytes: Size of the custom event buffer.
// logSuffix: Suffix string to append to log file names.
// syncGpu: Wait for GPU to complete processing for each profiling event with syncGpu flag set.
//
void PERF_PROFILER_API ProfilerInit(const std::wstring& profilerDir, const unsigned long long customEventBufferBytes,
    const std::wstring& logSuffix, const bool syncGpu)
{
    if (g_profilerState != nullptr)
    {
        RuntimeError("Error: ProfilerInit: Profiler already initialized.\n");
    }
    g_profilerState.reset(new ProfilerState());

    g_profilerState->profilerDir = profilerDir;
    g_profilerState->logSuffix = logSuffix;

    g_profilerState->customEventBufferFull = false;
    g_profilerState->customEventBufferBytes = customEventBufferBytes;
    g_profilerState->customEventOffset = 0ull;
    g_profilerState->customEventBuffer.reset(new char[customEventBufferBytes]);

    g_profilerState->syncGpu = syncGpu;
    g_profilerState->enabled = false;

    if (_wmkdir(g_profilerState->profilerDir.c_str()) == -1 && errno != EEXIST)
    {
        RuntimeError("Error: ProfilerInit: Cannot create directory <%ls>.\n", g_profilerState->profilerDir.c_str());
    }
}

//
// Enable/disable profiling.
// By default, profiling is disabled after a ProfilerInit call.
// This can be used to temporarily turn profiling on/off during execution.
//
void PERF_PROFILER_API ProfilerEnable(bool enable)
{
    // A nullptr state indicates that the profiler is globally disabled, and not initialized
    if (g_profilerState == nullptr)
        return;

    g_profilerState->enabled = enable;
}


//
// Internal helper functions to record fixed and custom profiling events.
//
void ProfilerTimeRecordFixedEvent(const int eventId, const long long beginClock, const long long endClock)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_profilerState->enabled)
        return;

    long long delta = endClock - beginClock;
    if (g_profilerState->fixedEvents[eventId].cnt == 0)
    {
        g_profilerState->fixedEvents[eventId].min = delta;
        g_profilerState->fixedEvents[eventId].max = delta;
    }
    g_profilerState->fixedEvents[eventId].min = std::min(delta, g_profilerState->fixedEvents[eventId].min);
    g_profilerState->fixedEvents[eventId].max = std::max(delta, g_profilerState->fixedEvents[eventId].max);
    g_profilerState->fixedEvents[eventId].sum += delta;
    g_profilerState->fixedEvents[eventId].sumsq += (double)delta * (double)delta;
    g_profilerState->fixedEvents[eventId].cnt++;
}

void ProfilerTimeRecordToBuffer(const char* eventDescription, const long long beginClock, const long long endClock)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_profilerState->enabled)
        return;

    auto eventDescriptionBytes = strlen(eventDescription) + 1;
    auto requiredBufferBytes = eventDescriptionBytes + sizeof(CustomEventRecord);
    if ((g_profilerState->customEventOffset + requiredBufferBytes) > g_profilerState->customEventBufferBytes)
    {
        if (!g_profilerState->customEventBufferFull)
        {
            fprintf(stderr, "Warning: Performance Profiler: Buffer is full, no more events will be recorded.\n");
            g_profilerState->customEventBufferFull = true;
        }
        return;
    }

    strcpy(g_profilerState->customEventBuffer.get() + g_profilerState->customEventOffset, eventDescription);
    g_profilerState->customEventOffset += eventDescriptionBytes;

    CustomEventRecord eventRecord;
    eventRecord.beginClock = beginClock;
    eventRecord.endClock = endClock;
    eventRecord.threadId = GetThreadId();

    memcpy(g_profilerState->customEventBuffer.get() + g_profilerState->customEventOffset, &eventRecord, sizeof(CustomEventRecord));
    g_profilerState->customEventOffset += sizeof(CustomEventRecord);
}


//
// Measure either a fixed or custom event time.
// ProfilerTimeBegin() returns a stateId that is passed to ProfilerTimeEnd().
// If ProfilerTimeEnd() is not called, the event is not recorded.
//
long long PERF_PROFILER_API ProfilerTimeBegin()
{
    return Clock::GetTimeStamp();
}


void PERF_PROFILER_API ProfilerTimeEnd(const long long stateId, const int eventId)
{
    // A nullptr state indicates that the profiler is globally disabled, and not initialized
    if (g_profilerState == nullptr)
        return;

    if (c_fixedEvtDesc[eventId].syncGpu)
        ProfilerSyncGpu();

    long long endClock = Clock::GetTimeStamp();
    ProfilerTimeRecordFixedEvent(eventId, stateId, endClock);
    ProfilerTimeRecordToBuffer(c_fixedEvtDesc[eventId].eventDescription, stateId, endClock);
}


void PERF_PROFILER_API ProfilerTimeEnd(const long long stateId, const char* eventDescription)
{
    // A nullptr state indicates that the profiler is globally disabled, and not initialized
    if (g_profilerState == nullptr)
        return;

    ProfilerTimeRecordToBuffer(eventDescription, stateId, Clock::GetTimeStamp());
}


//
// Conditionally sync the GPU if the syncGPU flag is set. This only needs to be excplicitly
// called for custom events.
//
void PERF_PROFILER_API ProfilerSyncGpu()
{
#ifndef CPUONLY
    // A nullptr state indicates that the profiler is globally disabled, and not initialized
    if (g_profilerState == nullptr)
        return;

    if(!g_profilerState->enabled)
        return;

    if (g_profilerState->syncGpu)
        cudaDeviceSynchronize();
#endif
}


//
// Measure throughput given the number of bytes.
// ProfilerThroughputBegin() returns a stateId that is passed to ProfilerThroughputEnd().
// If ProfilerThroughputEnd() is not called, the event is not recorded.
//
long long PERF_PROFILER_API ProfilerThroughputBegin()
{
    return Clock::GetTimeStamp();
}


void PERF_PROFILER_API ProfilerThroughputEnd(const long long stateId, const int eventId, const long long bytes)
{
    long long endClock = Clock::GetTimeStamp();

    // A nullptr state indicates that the profiler is globally disabled, and not initialized
    if (g_profilerState == nullptr)
        return;

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_profilerState->enabled)
        return;

    auto beginClock = stateId;
    if (endClock == beginClock)
        return;

    // Use kB rather than bytes to prevent overflow
    long long kBytesPerSec = Clock::GetTicksPerSecond() * bytes / 1000 / (endClock - beginClock);
    if (g_profilerState->fixedEvents[eventId].cnt == 0)
    {
        g_profilerState->fixedEvents[eventId].min = kBytesPerSec;
        g_profilerState->fixedEvents[eventId].max = kBytesPerSec;
    }
    g_profilerState->fixedEvents[eventId].min = std::min(kBytesPerSec, g_profilerState->fixedEvents[eventId].min);
    g_profilerState->fixedEvents[eventId].max = std::max(kBytesPerSec, g_profilerState->fixedEvents[eventId].max);
    g_profilerState->fixedEvents[eventId].sum += kBytesPerSec;
    g_profilerState->fixedEvents[eventId].sumsq += (double)kBytesPerSec * (double)kBytesPerSec;
    g_profilerState->fixedEvents[eventId].totalBytes += bytes;
    g_profilerState->fixedEvents[eventId].cnt++;
}


//
// Generate reports and release all resources.
//
void PERF_PROFILER_API ProfilerClose()
{
    // A nullptr state indicates that the profiler is globally disabled, and not initialized
    if (g_profilerState == nullptr)
        return;

    // Get current time as yyyy-mm-dd_hh-mm-ss
    time_t currentTime;
    time(&currentTime);
    struct tm* timeInfo = localtime(&currentTime);
    wchar_t timeStr[32];
    wcsftime(timeStr, sizeof(timeStr) / sizeof(timeStr[0]), L"%Y-%m-%d_%H-%M-%S", timeInfo);

    // Generate summary report
    std::wstring fileName = g_profilerState->profilerDir + L"/" + std::wstring(timeStr) + L"_summary_" + g_profilerState->logSuffix + L".txt";
    ProfilerGenerateReport(fileName, timeInfo);

    // Generate detailed event file
    fileName = g_profilerState->profilerDir + L"/" + std::wstring(timeStr) + L"_detail_" + g_profilerState->logSuffix + L".csv";
    ProfilerGenerateDetailFile(fileName);

    g_profilerState.reset();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility functions.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Get current thread id
//
unsigned int GetThreadId()
{
#ifdef _WIN32
    return (unsigned int)GetCurrentThreadId();
#else
    return (unsigned int)syscall(SYS_gettid);
#endif
}


//
// Generate summary report.
//
void ProfilerGenerateReport(const std::wstring& fileName, struct tm* timeInfo)
{
    FILE* f = _wfopen(fileName.c_str(), L"wt");
    if (f == NULL)
    {
        RuntimeError("Error: ProfilerGenerateReport: Cannot create file <%ls>.\n", fileName.c_str());
    }

    fprintfOrDie(f, "CNTK Performance Profiler Summary Report\n\n");
    char timeStr[32];
    strftime(timeStr, sizeof(timeStr), "%Y/%m/%d %H:%M:%S", timeInfo);
    fprintfOrDie(f, "Time Stamp: %s\n\n", timeStr);

    fprintfOrDie(f, "Description................ ............Mean ..........StdDev .............Min .............Max ...........Count ...........Total\n\n");

    for (int evtIdx = 0; evtIdx < profilerEvtMax; evtIdx++)
    {
        bool printLine = false;

        switch (c_fixedEvtDesc[evtIdx].eventType)
        {
        case profilerEvtTime:
            if (g_profilerState->fixedEvents[evtIdx].cnt > 0)
            {
                printLine = true;
                fprintfOrDie(f, "%-26s: ", c_fixedEvtDesc[evtIdx].eventDescription);

                char str[32];

                double mean = TicksToSeconds(g_profilerState->fixedEvents[evtIdx].sum) / g_profilerState->fixedEvents[evtIdx].cnt;
                FormatTimeStr(str, sizeof(str), mean);
                fprintfOrDie(f, "%s ", str);

                double sum = TicksToSeconds(g_profilerState->fixedEvents[evtIdx].sum);
                double sumsq = TicksSqToSecondsSq(g_profilerState->fixedEvents[evtIdx].sumsq);
                double stdDev = sumsq - (pow(sum, 2.0) / g_profilerState->fixedEvents[evtIdx].cnt);
                if (stdDev < 0.0) stdDev = 0.0;
                stdDev = sqrt(stdDev / (double)g_profilerState->fixedEvents[evtIdx].cnt);
                FormatTimeStr(str, sizeof(str), stdDev);
                fprintfOrDie(f, "%s ", str);

                FormatTimeStr(str, sizeof(str), TicksToSeconds(g_profilerState->fixedEvents[evtIdx].min));
                fprintfOrDie(f, "%s ", str);

                FormatTimeStr(str, sizeof(str), TicksToSeconds(g_profilerState->fixedEvents[evtIdx].max));
                fprintfOrDie(f, "%s ", str);

                fprintfOrDie(f, "%16d ", g_profilerState->fixedEvents[evtIdx].cnt);

                FormatTimeStr(str, sizeof(str), TicksToSeconds(g_profilerState->fixedEvents[evtIdx].sum));
                fprintfOrDie(f, "%s", str);
            }
            break;

        case profilerEvtThroughput:
            if (g_profilerState->fixedEvents[evtIdx].cnt > 0)
            {
                printLine = true;
                fprintfOrDie(f, "%-26s: ", c_fixedEvtDesc[evtIdx].eventDescription);

                char str[32];

                double mean = ((double)g_profilerState->fixedEvents[evtIdx].sum / (double)g_profilerState->fixedEvents[evtIdx].cnt);
                FormatThroughputStr(str, sizeof(str), mean);
                fprintfOrDie(f, "%s ", str);

                double stdDev = g_profilerState->fixedEvents[evtIdx].sumsq - (pow((double)g_profilerState->fixedEvents[evtIdx].sum, 2.0) / (double)g_profilerState->fixedEvents[evtIdx].cnt);
                if (stdDev < 0.0) stdDev = 0.0;
                stdDev = sqrt(stdDev / (double)g_profilerState->fixedEvents[evtIdx].cnt);
                FormatThroughputStr(str, sizeof(str), stdDev);
                fprintfOrDie(f, "%s ", str);

                FormatThroughputStr(str, sizeof(str), (double)g_profilerState->fixedEvents[evtIdx].min);
                fprintfOrDie(f, "%s ", str);

                FormatThroughputStr(str, sizeof(str), (double)g_profilerState->fixedEvents[evtIdx].max);
                fprintfOrDie(f, "%s ", str);

                fprintfOrDie(f, "%16d ", g_profilerState->fixedEvents[evtIdx].cnt);

                FormatBytesStr(str, sizeof(str), g_profilerState->fixedEvents[evtIdx].totalBytes);
                fprintfOrDie(f, "%s", str);
            }
            break;
        
        case profilerEvtSeparator:
            printLine = true;
            fprintfOrDie(f, "%s", c_fixedEvtDesc[evtIdx].eventDescription);
            break;
        }

        if (printLine) fprintfOrDie(f, "\n");
    }

    fclose(f);
}

//
// String formatting helpers for reporting.
//
void FormatTimeStr(char* str, size_t strLen, double seconds)
{
    if (seconds < 60.0)
    {
        sprintf_s(str, strLen, "%13.3f ms", seconds * 1000.0);
    }
    else
    {
        sprintf_s(str, strLen, "    %02d:%02d:%06.3f", (int)seconds / 3600, ((int)seconds / 60) % 60, fmod(seconds, 60.0));
    }
}

void FormatThroughputStr(char* str, size_t strLen, double kbps)
{
    // MBps = 1000000 bytes per second
    sprintf_s(str, strLen, "%11.3f MBps", kbps / 1000.0);
}

void FormatBytesStr(char* str, size_t strLen, long long bytes)
{
    // kB = 1024 bytes, MB = 1024*1024 bytes
    if (bytes < (1024ll * 1024ll))
    {
        sprintf_s(str, strLen, "%13lld kB", bytes >> 10);
    }
    else
    {
        sprintf_s(str, strLen, "%13lld MB", bytes >> 20);
    }
}



//
// Generate detail event file.
//
void ProfilerGenerateDetailFile(const std::wstring& fileName)
{
    FILE* f = _wfopen(fileName.c_str(), L"wt");
    if (f == NULL)
    {
        RuntimeError("Error: ProfilerGenerateDetailFile: Cannot create file <%ls>.\n", fileName.c_str());
    }

    fprintfOrDie(f, "EventDescription,ThreadId,BeginTimeStamp(ms),EndTimeStamp(ms)\n");

    char* eventPtr = g_profilerState->customEventBuffer.get();

    while (eventPtr < (g_profilerState->customEventBuffer.get() + g_profilerState->customEventOffset))
    {
        char* descriptionStr = eventPtr;
        eventPtr += strlen(descriptionStr) + 1;

        CustomEventRecord* eventRecord = (CustomEventRecord*)eventPtr;
        eventPtr += sizeof(CustomEventRecord);

        fprintfOrDie(f, "\"%s\",%u,%.8f,%.8f\n", descriptionStr, eventRecord->threadId, 
            1000.0 * TicksToSeconds(eventRecord->beginClock),
            1000.0 * TicksToSeconds(eventRecord->endClock));
    }

    fclose(f);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scoped helpers.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ProfilerContext::Init(const std::wstring& profilerDir, const unsigned long long customEventBufferBytes, const std::wstring& logSuffix, const bool syncGpu)
{
    ProfilerInit(profilerDir, customEventBufferBytes, logSuffix, syncGpu);
}

ProfilerContext::~ProfilerContext()
{
    ProfilerClose();
}


ScopeProfile::ScopeProfile(int eventId)
{
    m_eventId = eventId;
    m_description = nullptr;
    m_stateId = ProfilerTimeBegin();
}

ScopeProfile::ScopeProfile(const char* description)
{
    m_description = description;
    m_stateId = ProfilerTimeBegin();
}

ScopeProfile::~ScopeProfile()
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


ScopeThroughput::ScopeThroughput(int eventId, long long bytes)
{
    m_bytes = bytes;
    m_eventId = eventId;
    m_stateId = ProfilerThroughputBegin();
}

ScopeThroughput::~ScopeThroughput()
{
    ProfilerThroughputEnd(m_stateId, m_eventId, m_bytes);
}

}}}
