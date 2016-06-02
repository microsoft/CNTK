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

#ifdef _WIN32
#include <direct.h>
#else
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/syscall.h>
#endif 

#include "PerformanceProfiler.h"


namespace Microsoft { namespace MSR { namespace CNTK {

#define PROFILER_DIR        "./profiler"

enum ProfilerEvtType
{
    profilerEvtTime = 0,
    profilerEvtThroughput,
    profilerEvtSeparator
};

static const char* c_profilerEvtDesc[profilerEvtMax] = {  
    "Main Thread",
    "",
    "Epoch",
    "-Minibatch Iteration",
    "--Get Minibatch",
    "--Forward + Backward",
    "--Gradient Aggregation",
    "--Weight Update",
    "--Post Processing",
    "",
    "Gradient Aggregation Thread & MPI",
    "",
    "Gradient Aggregation (1bit)",
    "-Async Communcation 1",
    "-Wait for Gradients",
    "-Wait for Headers",
    "-Async Communication 2",
    "-Wait for Agg. Gradients",
    "-Wait for Agg. Headers",
    "-Wait for Completion",
    "Gradient Aggregation (32bit)",
    "-Async Communication 1",
    "-Wait for Headers",
    "-Async Communication 2",
    "-Wait for Gradients",
    "-Wait for Completion",
    "",
    "ImageReader Thread",
    "",
    "Image Decoding",
    "ImageReader Throughput"
};

static const ProfilerEvtType c_profilerEvtType[profilerEvtMax] = {
    profilerEvtSeparator,
    profilerEvtSeparator,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtSeparator,
    profilerEvtSeparator,
    profilerEvtSeparator,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtSeparator,
    profilerEvtSeparator,
    profilerEvtSeparator,
    profilerEvtTime,
    profilerEvtThroughput
};

struct FixedEventRecord
{
    int             cnt;
    long long       sum;
    double          sumsq;
    long long       min;
    long long       max;
    long long       totalBytes;
    long long       beginClock;
    int             refCnt;
};

struct CustomEventRecordBegin
{
    unsigned long long      uniqueId;       // Unique sequential id for this event
    long long               beginClock;     // Beginning time stamp
    unsigned int            threadId;       // Current thread id
};

struct CustomEventRecordEnd
{
    unsigned long long      uniqueId;       // Unique sequential id for this event
    long long               endClock;       // Ending time stamp
};

#define CUSTOM_EVT_BEGIN_TYPE       0
#define CUSTOM_EVT_END_TYPE         1

//
// The custom event record is a variable size datastructure in memory:
// char: custom event type (#defined above)
// if begin event:
// NULL terminated description string, followed by CustomEventRecordBegin struct
// if end event:
// CustomEventRecordEnd struct
//


struct ProfilerState
{
    bool                    init = false;
    bool                    enabled = false;
    char                    profilerDir[256];
    char                    logSuffix[64];
    long long               clockFrequency;
    FixedEventRecord        fixedEvents[profilerEvtMax];
    unsigned long long      uniqueId;
    unsigned long long      customEventBufferBytes;     // Number of bytes allocated for the custom event buffer
    unsigned long long      customEventPtr;             // Pointer to current place in buffer
    unsigned long long      customEventCommittedBytes;  // Number of bytes committed in the event buffer
    char*                   customEventBuffer;
};

// State id is packed into a 64 bit int as follows:
// bits 63-16: unique custom event id
// bits 15-0: fixed event id
#define INVALID_STATE_ID        0xffffffffffffffffull
#define INVALID_FIXED_EVENT_ID  0xffffull

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
//
void PERF_PROFILER_API ProfilerInit(const char* profilerDir, const unsigned long long customEventBufferBytes, const char* logSuffix)
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
// Measure time for either a fixed or a custom event.
// The *Begin call returns a stateId that is passed to ProfilerTimeEnd().
// If the event does not need to be recorded, call ProfilerTimeCancel() before ProfilerTimeEnd().
//
unsigned long long PERF_PROFILER_API ProfilerTimeBegin(const int eventId)
{
    if (!g_profilerState.init || !g_profilerState.enabled) return INVALID_STATE_ID;

    unsigned long long stateId = ProfilerTimeBegin(c_profilerEvtDesc[eventId]);
    if (stateId == INVALID_STATE_ID) return INVALID_STATE_ID;

    LOCK

    if (g_profilerState.fixedEvents[eventId].refCnt == 0)
    {
        g_profilerState.fixedEvents[eventId].beginClock = GetClock();
    }
    g_profilerState.fixedEvents[eventId].refCnt++;

    return (stateId & ~0xffffull) | (unsigned long long)eventId;
}

void ProfilerTimeEndInt(const unsigned long long stateId, const long long customClock = -1ll)
{
    long long endClock = (customClock == -1ll ? GetClock() : customClock);
    if (!g_profilerState.init || (stateId & 0xffffffffffff0000ull) == (INVALID_STATE_ID & 0xffffffffffff0000ull)) return;

    LOCK

    g_profilerState.customEventBuffer[g_profilerState.customEventPtr++] = CUSTOM_EVT_END_TYPE;

    CustomEventRecordEnd endRecord;
    endRecord.endClock = endClock;
    endRecord.uniqueId = (stateId & ~0xffffull) >> 16;

    memcpy(g_profilerState.customEventBuffer + g_profilerState.customEventPtr, &endRecord, sizeof(CustomEventRecordEnd));
    g_profilerState.customEventPtr += sizeof(CustomEventRecordEnd);

    int eventId = stateId & 0xffff;
    if (eventId != INVALID_FIXED_EVENT_ID)
    {
        g_profilerState.fixedEvents[eventId].refCnt--;
        if (g_profilerState.fixedEvents[eventId].refCnt == 0)
        {
            long long delta = endClock - g_profilerState.fixedEvents[eventId].beginClock;
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
    }
}

void PERF_PROFILER_API ProfilerTimeEnd(const unsigned long long stateId)
{
    ProfilerTimeEndInt(stateId);
}

void PERF_PROFILER_API ProfilerTimeCancel(unsigned long long* stateId)
{
    if (!g_profilerState.init || ((*stateId) & 0xffffffffffff0000ull) == (INVALID_STATE_ID & 0xffffffffffff0000ull)) return;

    LOCK

    int eventId = (*stateId) & 0xffff;
    if (eventId != INVALID_FIXED_EVENT_ID)
    {
        g_profilerState.fixedEvents[eventId].refCnt = 0;
    }

    *stateId = INVALID_STATE_ID;
}

unsigned long long ProfilerTimeBeginInt(const char* eventDescription, const long long customClock = -1ll)
{
    if (!g_profilerState.init || !g_profilerState.enabled) return INVALID_STATE_ID;

    LOCK

    // TODO: Perf: We are iterating through eventDescription twice, for length and the copy, this could be optimized.
    unsigned long long eventDescriptionBytes = strlen(eventDescription) + 1;
    unsigned long long requiredBufferBytes = 2 /* Allow room for the custom event type */ + eventDescriptionBytes + sizeof(CustomEventRecordBegin) + sizeof(CustomEventRecordEnd);
    if ((g_profilerState.customEventCommittedBytes + requiredBufferBytes) > g_profilerState.customEventBufferBytes) return INVALID_STATE_ID;
    g_profilerState.customEventCommittedBytes += requiredBufferBytes;

    // Increment id with wrapping around, skipping the INVALID_STATE_ID
    g_profilerState.uniqueId++;
    g_profilerState.uniqueId &= 0xffffffffffffull;
    if (g_profilerState.uniqueId == (INVALID_STATE_ID >> 16))
    {
        g_profilerState.uniqueId++;
        g_profilerState.uniqueId &= 0xffffffffffffull;
    }

    g_profilerState.customEventBuffer[g_profilerState.customEventPtr++] = CUSTOM_EVT_BEGIN_TYPE;

    strcpy(g_profilerState.customEventBuffer + g_profilerState.customEventPtr, eventDescription);
    g_profilerState.customEventPtr += eventDescriptionBytes;

    CustomEventRecordBegin beginRecord;
    beginRecord.uniqueId = g_profilerState.uniqueId;
    beginRecord.threadId = GetThreadId();
    beginRecord.beginClock = (customClock == -1ll ? GetClock() : customClock);

    memcpy(g_profilerState.customEventBuffer + g_profilerState.customEventPtr, &beginRecord, sizeof(CustomEventRecordBegin));
    g_profilerState.customEventPtr += sizeof(CustomEventRecordBegin);

    return (g_profilerState.uniqueId << 16) | INVALID_FIXED_EVENT_ID;
}

unsigned long long PERF_PROFILER_API ProfilerTimeBegin(const char* eventDescription)
{
    return ProfilerTimeBeginInt(eventDescription);
}


//
// CUDA kernel profiling.
// The cuda event calls must be called around the kernel, and the total kernel time provided
// in deltaSeconds.
//
unsigned long long PERF_PROFILER_API ProfilerCudaTimeBegin(const char* eventDescription)
{
    return ProfilerTimeBeginInt(eventDescription, 0ll);
}


void PERF_PROFILER_API ProfilerCudaTimeEnd(const float deltaSeconds, const unsigned long long stateId)
{
    ProfilerTimeEndInt(stateId, (long long)((double)deltaSeconds * (double)g_profilerState.clockFrequency));
}


//
// Measure throughput given a bytes in an *Begin/*End block.
// The ThroughputEventRecord is meaintained by the caller.
// If ProfilerThroughputEnd is not called, the event is not recorded.
//
void PERF_PROFILER_API ProfilerThroughputBegin(const int eventId, ProfilerThroughputEventRecord* profilerThroughputEventRecord)
{
    if (!g_profilerState.init) return;

    profilerThroughputEventRecord->eventId = eventId;
    profilerThroughputEventRecord->beginClock = GetClock();
}

void PERF_PROFILER_API ProfilerThroughputEnd(const long long bytes, ProfilerThroughputEventRecord* profilerThroughputEventRecord)
{
    long long endClock = GetClock();
    if (!g_profilerState.init || !g_profilerState.enabled) return;

    LOCK

    if (endClock == profilerThroughputEventRecord->beginClock) return;
    // Use KB rather than bytes to prevent overflow
    long long KBytesPerSec = ((bytes * g_profilerState.clockFrequency) / 1000) / (endClock - profilerThroughputEventRecord->beginClock);
    int eventId = profilerThroughputEventRecord->eventId;
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
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
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

        switch (c_profilerEvtType[evtIdx])
        {
        case profilerEvtTime:
            if (g_profilerState.fixedEvents[evtIdx].refCnt == 0 && g_profilerState.fixedEvents[evtIdx].cnt > 0)
            {
                printLine = true;
                fprintfOrDie(f, "%-26s: ", c_profilerEvtDesc[evtIdx]);

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
                fprintfOrDie(f, "%-26s: ", c_profilerEvtDesc[evtIdx]);

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
            fprintfOrDie(f, "%s", c_profilerEvtDesc[evtIdx]);
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

    fprintfOrDie(f, "EventId,Description,ThreadId,TimeStamp(ms)\n");

    char* eventPtr = g_profilerState.customEventBuffer;

    while (eventPtr < (g_profilerState.customEventBuffer + g_profilerState.customEventPtr))
    {
        if (*eventPtr == CUSTOM_EVT_BEGIN_TYPE)
        {
            eventPtr++;

            char* descriptionStr = eventPtr;
            eventPtr += strlen(descriptionStr) + 1;

            CustomEventRecordBegin beginRecord;
            memcpy(&beginRecord, eventPtr, sizeof(CustomEventRecordBegin));
            eventPtr += sizeof(CustomEventRecordBegin);

            fprintfOrDie(f, "%llu,\"%s\",%u,%.8f\n", beginRecord.uniqueId, descriptionStr, beginRecord.threadId, 1000.0 * ((double)beginRecord.beginClock / (double)g_profilerState.clockFrequency));
        }
        else if (*eventPtr == CUSTOM_EVT_END_TYPE)
        {
            eventPtr++;

            CustomEventRecordEnd endRecord;
            memcpy(&endRecord, eventPtr, sizeof(CustomEventRecordEnd));
            eventPtr += sizeof(CustomEventRecordEnd);

            fprintfOrDie(f, "%llu,,,%.8f\n", endRecord.uniqueId, 1000.0 * ((double)endRecord.endClock / (double)g_profilerState.clockFrequency));
        }
        else
        {
            assert(*eventPtr == CUSTOM_EVT_BEGIN_TYPE || *eventPtr == CUSTOM_EVT_END_TYPE);
            break;
        }
    }

    fclose(f);
}


}}}
