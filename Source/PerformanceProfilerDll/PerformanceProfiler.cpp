//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Real-time thread-safe profiler that generats a summary report and a detail profile log.
// The profiler is highly performant and light weight and meant to be left on all the time.
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif
#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _

#include "Basics.h"
#include "fileutil.h"
#include <stdio.h>
#include <time.h>
#ifndef CPUONLY
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#ifdef _WIN32
#include <direct.h>
#else
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/syscall.h>
#endif 

#ifndef CPUONLY
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#endif

#include "PerformanceProfiler.h"


namespace Microsoft { namespace MSR { namespace CNTK {

#define PROFILER_DIR        "./profiler"

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
	char			eventDescription[64];
	FixedEventType	eventType;
	bool			syncGpu;
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
    { "Gradient Aggregation Thread & MPI", profilerEvtSeparator, false }, // profilerSepMPI
    { "", profilerEvtSeparator, false },                            // profilerSepSpace2

    { "Gradient Aggregation (1b)", profilerEvtTime, false },        // profilerEvtGradient1
    { "_Async Communcation 1", profilerEvtTime, false },            // profilerEvtGradientAsyncComm11
    { "_Wait for Gradients", profilerEvtTime, false },              // profilerEvtGradientWaitGradients1
    { "_Wait for Headers", profilerEvtTime, false },                // profilerEvtGradientWaitHeaders1
    { "_Async Communication 2", profilerEvtTime, false },           // profilerEvtGradientAsyncComm21
    { "_Wait for Agg. Gradients", profilerEvtTime, false },         // profilerEvtGradientWaitAggGradients1
    { "_Wait for Agg. Headers", profilerEvtTime, false },           // profilerEvtGradientWaitAggHeaders1
    { "_Wait for Completion", profilerEvtTime, false },             // profilerEvtGradientWaitCompletion1

    { "Gradient Aggregation (32b)", profilerEvtTime, false },       // profilerEvtGradient32
    { "_Async Communication 1", profilerEvtTime, false },           // profilerEvtGradientAsyncComm132
    { "_Wait for Headers", profilerEvtTime, false },                // profilerEvtGradientWaitHeaders32
    { "_Async Communication 2", profilerEvtTime, false },           // profilerEvtGradientAsyncComm232
    { "_Wait for Gradients", profilerEvtTime, false },              // profilerEvtGradientWaitGradients32
    { "_Wait for Completion", profilerEvtTime, false },             // profilerEvtGradientWaitCompletion32

    { "", profilerEvtSeparator, false },                            // profilerSepSpace3
    { "Data Reader", profilerEvtSeparator, false },                 // profilerSepDataReader
    { "", profilerEvtSeparator, false },                            // profilerSepSpace4

    { "Read Minibatch Task", profilerEvtTime, false },              // profilerEvtReadMinibatch
    { "ImageReader Throughput", profilerEvtThroughput, false },     // profilerEvtZipReaderThroughput
};


struct FixedEventRecord
{
    int             cnt;
    long long       sum;
    double          sumsq;
    long long       min;
    long long       max;
    long long       totalBytes;
};

//
// The custom event record is a variable size datastructure in memory:
// NULL terminated description string, followed by CustomEventRecord struct
//
struct CustomEventRecord
{
    long long               beginClock;
    long long               endClock;
    unsigned int            threadId;
};


//
// Global state of the profiler
//
struct ProfilerState
{
    bool                    init = false;
    bool                    enabled = false;
    bool                    syncGpu = false;
    char                    profilerDir[256];
    char                    logSuffix[64];
    long long               clockFrequency;
    FixedEventRecord        fixedEvents[profilerEvtMax];
    unsigned long long      customEventBufferBytes;     // Number of bytes allocated for the custom event buffer
    unsigned long long      customEventPtr;             // Pointer to current place in buffer
    char*                   customEventBuffer;
};


// We support one global instance of the profiler
static ProfilerState g_profilerState;

// Forward declarations
unsigned int GetThreadId();

long long GetClockFrequency();
long long GetClock();

void LockInit();
inline void LockEnter();
inline void LockLeave();
void LockClose();

void ProfilerGenerateReport(const char* fileName, struct tm* timeInfo);
void FormatTimeStr(char* str, size_t strLen, double value);
void FormatThroughputStr(char* str, size_t strLen, double value);
void FormatBytesStr(char* str, size_t strLen, long long bytes);
void ProfilerGenerateDetailFile(const char* fileName);

//
// Convenience scope lock.
//
struct ScopeLock
{
    ScopeLock() { LockEnter(); }
    ~ScopeLock() { LockLeave(); }
};
#define LOCK    ScopeLock sl;

//
// Initialize all resources to enable profiling.
// profilerDir: Directory where the profiler logs will be saved. nullptr for default location.
// customEventBufferBytes: Bytes to allocate for the custom event buffer.
// logSuffix: Suffix string to append to log files.
// syncGpu: Wait for GPU to complete processing for each profiling event.
//
void PERF_PROFILER_API ProfilerInit(const char* profilerDir, const unsigned long long customEventBufferBytes, 
                                    const char* logSuffix, const bool syncGpu)
{
    memset(&g_profilerState, 0, sizeof(g_profilerState));

    LockInit();

    if (profilerDir)
    {
        strncpy(g_profilerState.profilerDir, profilerDir, sizeof(g_profilerState.profilerDir) - 1);
    }
    else
    {
        strncpy(g_profilerState.profilerDir, PROFILER_DIR, sizeof(g_profilerState.profilerDir) - 1);
    }
    g_profilerState.profilerDir[sizeof(g_profilerState.profilerDir) - 1] = 0;

    strncpy(g_profilerState.logSuffix, logSuffix, sizeof(g_profilerState.logSuffix) - 1);
    g_profilerState.logSuffix[sizeof(g_profilerState.logSuffix) - 1] = 0;

    g_profilerState.customEventBufferBytes = customEventBufferBytes;
    g_profilerState.customEventBuffer = new char[customEventBufferBytes];

    g_profilerState.clockFrequency = GetClockFrequency();

    g_profilerState.syncGpu = syncGpu;
    g_profilerState.enabled = false;
    g_profilerState.init = true;
}

//
// Enable/disable profiling.
// By default, profiling is disabled after a ProfilerInit call.
//
void PERF_PROFILER_API ProfilerEnable(bool enable)
{
    if (g_profilerState.init)
        g_profilerState.enabled = enable;
    else
        g_profilerState.enabled = false;
}


//
// Internal helper functions to record fixed and custom profiling events.
//
void ProfilerTimeEndInt(const int eventId, const long long beginClock, const long long endClock)
{
    if (!g_profilerState.init || !g_profilerState.enabled) return;

    LOCK

    long long delta = endClock - beginClock;
    if (g_profilerState.fixedEvents[eventId].cnt == 0)
    {
        g_profilerState.fixedEvents[eventId].min = delta;
        g_profilerState.fixedEvents[eventId].max = delta;
    }
    g_profilerState.fixedEvents[eventId].min = min(delta, g_profilerState.fixedEvents[eventId].min);
    g_profilerState.fixedEvents[eventId].max = max(delta, g_profilerState.fixedEvents[eventId].max);
    g_profilerState.fixedEvents[eventId].sum += delta;
    g_profilerState.fixedEvents[eventId].sumsq += (double)delta * (double)delta;
    g_profilerState.fixedEvents[eventId].cnt++;
}

void ProfilerTimeEndInt(const char* eventDescription, const long long beginClock, const long long endClock)
{
    if (!g_profilerState.init || !g_profilerState.enabled) return;

    LOCK

    auto eventDescriptionBytes = strlen(eventDescription) + 1;
    auto requiredBufferBytes = eventDescriptionBytes + sizeof(CustomEventRecord);
    if ((g_profilerState.customEventPtr + requiredBufferBytes) > g_profilerState.customEventBufferBytes) return;

    strcpy(g_profilerState.customEventBuffer + g_profilerState.customEventPtr, eventDescription);
    g_profilerState.customEventPtr += eventDescriptionBytes;

    CustomEventRecord eventRecord;
    eventRecord.beginClock = beginClock;
    eventRecord.endClock = endClock;
    eventRecord.threadId = GetThreadId();

    memcpy(g_profilerState.customEventBuffer + g_profilerState.customEventPtr, &eventRecord, sizeof(CustomEventRecord));
    g_profilerState.customEventPtr += sizeof(CustomEventRecord);
}


//
// Measure either a fixed or custom event time.
// ProfilerTimeBegin() returns a stateId that is passed to ProfilerTimeEnd().
// If ProfilerTimeEnd() is not called, the event is not recorded.
//
long long PERF_PROFILER_API ProfilerTimeBegin()
{
    return GetClock();
}


void PERF_PROFILER_API ProfilerTimeEnd(const long long stateId, const int eventId)
{
    if (c_fixedEvtDesc[eventId].syncGpu) ProfilerSyncGpu();
    long long endClock = GetClock();
    ProfilerTimeEndInt(eventId, stateId, endClock);
    ProfilerTimeEndInt(c_fixedEvtDesc[eventId].eventDescription, stateId, endClock);
}


void PERF_PROFILER_API ProfilerTimeEnd(const long long stateId, const char* eventDescription)
{
    ProfilerTimeEndInt(eventDescription, stateId, GetClock());
}


//
// Conditionally sync the GPU if the syncGPU flag is set. This only needs to be excplicitly
// called for custom events.
//
void PERF_PROFILER_API ProfilerSyncGpu()
{
#ifndef CPUONLY
	if (!g_profilerState.init || !g_profilerState.enabled) return;
    if (g_profilerState.syncGpu) cudaDeviceSynchronize();
#endif
}


//
// CUDA kernel profiling.
// CUDA kernels are profiled using a single call to this function.
//
void PERF_PROFILER_API ProfilerCudaTimeEnd(const float deltaSeconds, const char* eventDescription)
{
    ProfilerTimeEndInt(eventDescription, 0ll, (long long)((double)deltaSeconds * (double)g_profilerState.clockFrequency));
}


//
// Measure throughput given the number of bytes.
// ProfilerThroughputBegin() returns a stateId that is passed to ProfilerThroughputEnd().
// If ProfilerThroughputEnd() is not called, the event is not recorded.
//
long long PERF_PROFILER_API ProfilerThroughputBegin()
{
    return GetClock();
}


void PERF_PROFILER_API ProfilerThroughputEnd(const long long stateId, const int eventId, const long long bytes)
{
    long long endClock = GetClock();
    if (!g_profilerState.init || !g_profilerState.enabled) return;

    LOCK

    auto beginClock = stateId;
    if (endClock == beginClock) return;

    // Use KB rather than bytes to prevent overflow
    long long KBytesPerSec = ((bytes * g_profilerState.clockFrequency) / 1000) / (endClock - beginClock);
    if (g_profilerState.fixedEvents[eventId].cnt == 0)
    {
        g_profilerState.fixedEvents[eventId].min = KBytesPerSec;
        g_profilerState.fixedEvents[eventId].max = KBytesPerSec;
    }
    g_profilerState.fixedEvents[eventId].min = min(KBytesPerSec, g_profilerState.fixedEvents[eventId].min);
    g_profilerState.fixedEvents[eventId].max = max(KBytesPerSec, g_profilerState.fixedEvents[eventId].max);
    g_profilerState.fixedEvents[eventId].sum += KBytesPerSec;
    g_profilerState.fixedEvents[eventId].sumsq += (double)KBytesPerSec * (double)KBytesPerSec;
    g_profilerState.fixedEvents[eventId].totalBytes += bytes;
    g_profilerState.fixedEvents[eventId].cnt++;
}


//
// Generate reports and release all resources.
//
void PERF_PROFILER_API ProfilerClose()
{
    if (!g_profilerState.init) return;
    g_profilerState.init = false;

    LockClose();

    // Generate summary report
    _wmkdir(s2ws(g_profilerState.profilerDir).c_str());

    time_t currentTime;
    time(&currentTime);
    struct tm* timeInfo = localtime(&currentTime);

    char timeStr[32];
    strftime(timeStr, sizeof(timeStr), "%Y-%m-%d_%H-%M-%S", timeInfo);

    char fileName[256];
    sprintf_s(fileName, sizeof(fileName)-1, "%s/%s_summary_%s.txt", g_profilerState.profilerDir, timeStr, g_profilerState.logSuffix);
    ProfilerGenerateReport(fileName, timeInfo);

    // Generate detailed event file
    sprintf_s(fileName, sizeof(fileName) - 1, "%s/%s_detail_%s.csv", g_profilerState.profilerDir, timeStr, g_profilerState.logSuffix);
    ProfilerGenerateDetailFile(fileName);

    delete[] g_profilerState.customEventBuffer;
}


//
// Helpers for CUDA call profiling.
//

#ifdef NO_SYNC
static bool g_SyncCudaScope_syncEnabled = false;
#else
static bool g_SyncCudaScope_syncEnabled = true;
#endif
static bool g_SyncCudaScope_profilingEnabled = false;

// Helper functions to set/get Sync flags.
void PERF_PROFILER_API SyncCudaScopeSetFlags(bool syncEnabled, bool profilingEnabled)
{
#ifdef NO_SYNC
    g_SyncCudaScope_syncEnabled = syncEnabled;
#else
    g_SyncCudaScope_syncEnabled = true;
#endif
    g_SyncCudaScope_profilingEnabled = profilingEnabled;
}

void PERF_PROFILER_API SyncCudaScopeGetFlags(bool& syncEnabled, bool& profilingEnabled)
{
#ifdef NO_SYNC
    syncEnabled = g_SyncCudaScope_syncEnabled;
#else
    syncEnabled = true;
#endif
    profilingEnabled = g_SyncCudaScope_profilingEnabled;
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
// Return freqeuncy in Hz of clock.
//
long long GetClockFrequency()
{
    long long frequency;

#ifdef _WIN32
    QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
#else
    timespec res;
    clock_getres(CLOCK_REALTIME, &res);
    frequency = 1000000000ll / ((long long)res.tv_nsec + 1000000000ll * (long long)res.tv_sec);
#endif

    return frequency;
}

//
// Get current timestamp.
//
long long GetClock()
{
    long long tm;
#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER*)&tm);
#else
    timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    tm = (long long)t.tv_nsec + 1000000000ll * (long long)t.tv_sec;
#endif

    return tm;
}


//
// Locking primitives.
// We avoid unnecessary memory allocations or initializations.
//

#ifdef _WIN32
static CRITICAL_SECTION g_critSec;
#else
static pthread_mutex_t g_mutex;
#endif

//
// Initialize lock object, to be called once at startup.
//
void LockInit()
{
#ifdef _WIN32
    InitializeCriticalSection(&g_critSec);
#else
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_mutex, &attr);
    pthread_mutexattr_destroy(&attr);
#endif
}

//
// Enter the lock. This can block indefinetly.
//
inline void LockEnter()
{
#ifdef _WIN32
    EnterCriticalSection(&g_critSec);
#else
    pthread_mutex_lock(&g_mutex);
#endif
}

//
// Leave the lock.
//
inline void LockLeave()
{
#ifdef _WIN32
    LeaveCriticalSection(&g_critSec);
#else
    pthread_mutex_unlock(&g_mutex);
#endif
}

//
// Release lock resources, to be called once when cleaning up.
//
void LockClose()
{
#ifdef _WIN32
    DeleteCriticalSection(&g_critSec);
#else
    pthread_mutex_destroy(&g_mutex);
#endif
}



//
// Generate summary report.
//
void ProfilerGenerateReport(const char* fileName, struct tm* timeInfo)
{
    FILE* f = fopen(fileName, "wt");
    if (f == NULL)
    {
        fprintf(stderr, "Error: ProfilerGenerateReport: Cannot create file <%s>.\n", fileName);
        return;
    }

    fprintfOrDie(f, "CNTK Performance Profiler Summary Report\n\n");
    char timeStr[32];
    strftime(timeStr, sizeof(timeStr), "%Y/%m/%d %H:%M:%S", timeInfo);
    fprintfOrDie(f, "Time Stamp: %s\n\n", timeStr);

    fprintfOrDie(f, "Description................ ............Mean ..........StdDev .............Min .............Max ...........Count ...........Total\n");

    for (int evtIdx = 0; evtIdx < profilerEvtMax; evtIdx++)
    {
        bool printLine = false;

        switch (c_fixedEvtDesc[evtIdx].eventType)
        {
        case profilerEvtTime:
            if (g_profilerState.fixedEvents[evtIdx].cnt > 0)
            {
                printLine = true;
                fprintfOrDie(f, "%-26s: ", c_fixedEvtDesc[evtIdx].eventDescription);

                char str[32];

                double mean = ((double)g_profilerState.fixedEvents[evtIdx].sum / (double)g_profilerState.fixedEvents[evtIdx].cnt) / (double)g_profilerState.clockFrequency;
                FormatTimeStr(str, sizeof(str), mean);
                fprintfOrDie(f, "%s ", str);

                double sum = (double)g_profilerState.fixedEvents[evtIdx].sum / (double)g_profilerState.clockFrequency;
                double sumsq = g_profilerState.fixedEvents[evtIdx].sumsq / (double)g_profilerState.clockFrequency / (double)g_profilerState.clockFrequency;
                double stdDev = sumsq - (pow(sum, 2.0) / (double)g_profilerState.fixedEvents[evtIdx].cnt);
                if (stdDev < 0.0) stdDev = 0.0;
                stdDev = sqrt(stdDev / (double)g_profilerState.fixedEvents[evtIdx].cnt);
                FormatTimeStr(str, sizeof(str), stdDev);
                fprintfOrDie(f, "%s ", str);

                FormatTimeStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].min / (double)g_profilerState.clockFrequency);
                fprintfOrDie(f, "%s ", str);

                FormatTimeStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].max / (double)g_profilerState.clockFrequency);
                fprintfOrDie(f, "%s ", str);

                fprintfOrDie(f, "%16d ", g_profilerState.fixedEvents[evtIdx].cnt);

                FormatTimeStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].sum / (double)g_profilerState.clockFrequency);
                fprintfOrDie(f, "%s", str);
            }
            break;

        case profilerEvtThroughput:
            if (g_profilerState.fixedEvents[evtIdx].cnt > 0)
            {
                printLine = true;
                fprintfOrDie(f, "%-26s: ", c_fixedEvtDesc[evtIdx].eventDescription);

                char str[32];

                double mean = ((double)g_profilerState.fixedEvents[evtIdx].sum / (double)g_profilerState.fixedEvents[evtIdx].cnt);
                FormatThroughputStr(str, sizeof(str), mean);
                fprintfOrDie(f, "%s ", str);

                double stdDev = g_profilerState.fixedEvents[evtIdx].sumsq - (pow((double)g_profilerState.fixedEvents[evtIdx].sum, 2.0) / (double)g_profilerState.fixedEvents[evtIdx].cnt);
                if (stdDev < 0.0) stdDev = 0.0;
                stdDev = sqrt(stdDev / (double)g_profilerState.fixedEvents[evtIdx].cnt);
                FormatThroughputStr(str, sizeof(str), stdDev);
                fprintfOrDie(f, "%s ", str);

                FormatThroughputStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].min);
                fprintfOrDie(f, "%s ", str);

                FormatThroughputStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].max);
                fprintfOrDie(f, "%s ", str);

                fprintfOrDie(f, "%16d ", g_profilerState.fixedEvents[evtIdx].cnt);

                FormatBytesStr(str, sizeof(str), g_profilerState.fixedEvents[evtIdx].totalBytes);
                fprintfOrDie(f, "%s", str);
            }
            break;
        
        case profilerEvtSeparator:
            printLine = true;
            fprintfOrDie(f, "%s", c_fixedEvtDesc[evtIdx].eventDescription);
            break;
        }

        if(printLine) fprintfOrDie(f, "\n");
    }

    fclose(f);
}

//
// String formatting helpers for reporting.
//
void FormatTimeStr(char* str, size_t strLen, double value)
{
    if (value < 60.0)
    {
        sprintf_s(str, strLen, "%13.3f ms", value * 1000.0);
    }
    else
    {
        sprintf_s(str, strLen, "    %02d:%02d:%06.3f", (int)value / 3600, ((int)value / 60) % 60, fmod(value, 60.0));
    }
}

void FormatThroughputStr(char* str, size_t strLen, double value)
{
    sprintf_s(str, strLen, "%11.3f MBps", value / 1000.0);
}

void FormatBytesStr(char* str, size_t strLen, long long bytes)
{
    if (bytes < (1024ll * 1024ll))
    {
        sprintf_s(str, strLen, "%13lld KB", bytes >> 10);
    }
    else
    {
        sprintf_s(str, strLen, "%13lld MB", bytes >> 20);
    }
}



//
// Generate detail event file.
//
void ProfilerGenerateDetailFile(const char* fileName)
{
    FILE* f = fopen(fileName, "wt");
    if (f == NULL)
    {
        fprintf(stderr, "Error: ProfilerGenerateDetailFile: Cannot create file <%s>.\n", fileName);
        return;
    }

    fprintfOrDie(f, "EventDescription,ThreadId,BeginTimeStamp(ms),EndTimeStamp(ms)\n");

    char* eventPtr = g_profilerState.customEventBuffer;

    while (eventPtr < (g_profilerState.customEventBuffer + g_profilerState.customEventPtr))
    {
        char* descriptionStr = eventPtr;
        eventPtr += strlen(descriptionStr) + 1;

        CustomEventRecord* eventRecord = (CustomEventRecord*)eventPtr;
        eventPtr += sizeof(CustomEventRecord);

        fprintfOrDie(f, "\"%s\",%u,%.8f,%.8f\n", descriptionStr, eventRecord->threadId, 
            1000.0 * ((double)eventRecord->beginClock / (double)g_profilerState.clockFrequency),
            1000.0 * ((double)eventRecord->endClock / (double)g_profilerState.clockFrequency));
    }

    fclose(f);
}


}}}
