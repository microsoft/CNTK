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
#include "PerformanceProfiler.h"
#include "fileutil.h"
#include <stdio.h>
#include <time.h>


#ifdef _WIN32
#include <direct.h>
#else
#include <pthread.h>
#include <sys/stat.h>
#endif 


namespace Microsoft { namespace MSR { namespace CNTK {

#define PROFILER_DIR        "./profiler"
#define PROFILER_SUMMARY    "summary.txt"
#define PROFILER_LOG        "log.csv"

enum ProfilerEvtType
{
    profilerEvtTime = 0,
    profilerEvtThroughput
};

static const char* c_profilerEvtDesc[profilerEvtMax] = {  
    "Forward Pass Time.........:",
    "Backward Pass Time........:",
    "Input Data Processing Time:",
    "MPI Processing Time.......:",
    "MPI Throughput............:",
    "Disk Throughput...........:"
};

static const ProfilerEvtType c_profilerEvtType[profilerEvtMax] = {
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtTime,
    profilerEvtThroughput,
    profilerEvtThroughput
};

struct FixedEventRecord
{
    int             cnt;
    long long       sum;
    long long       sumsq;
    long long       min;
    long long       max;
    long long       beginClock;
    int             refCnt;
};

#define THROUGHPUT_EVENT_MAX   (profilerEvtMax * 16)

struct ThroughputEventRecord
{
    long long       beginClock;
    int             eventId;
};

struct ProfilerState
{
    bool                    init = false;
    char                    profilerDir[256];
    long long               clockFrequency;
    FixedEventRecord        fixedEvents[profilerEvtMax];
    int                     throughputIdx;
    ThroughputEventRecord   throughputEvents[THROUGHPUT_EVENT_MAX];
};

// We support one global instance of the profiler
static ProfilerState g_profilerState;

// Forward declarations
long long GetClockFrequency();
long long GetClock();

void LockInit();
inline void LockEnter();
inline void LockLeave();
void LockClose();

void ProfilerGenerateReport(const char* fileName, const char* timeStamp);
void FormatTimeStr(char* str, size_t strLen, double value);
void FormatThroughputStr(char* str, size_t strLen, double value);

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
// Optionally provide a directory path where profiling files are saved, or nullptr.
//
void PERF_PROFILER_API ProfilerInit(const char* profilerDir)
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
    g_profilerState.clockFrequency = GetClockFrequency();
    g_profilerState.throughputIdx = -1;

    g_profilerState.init = true;
}

//
// Measure time for either a fixed or a custom event.
// The *Begin calls return a stateId that is passed to ProfilerTimeEnd().
//
int PERF_PROFILER_API ProfilerTimeBegin(int eventId)
{
    LOCK
    if (!g_profilerState.init) return 0;

    if (g_profilerState.fixedEvents[eventId].refCnt == 0)
    {
        g_profilerState.fixedEvents[eventId].beginClock = GetClock();
    }
    g_profilerState.fixedEvents[eventId].refCnt++;

    return eventId;
}

int PERF_PROFILER_API ProfilerTimeBegin(const char* description)
{
    UNUSED(description);
    return 0;
}

void PERF_PROFILER_API ProfilerTimeEnd(int stateId)
{
    long long endClock = GetClock();
    LOCK
    if (!g_profilerState.init) return;
    g_profilerState.fixedEvents[stateId].refCnt--;
    if (g_profilerState.fixedEvents[stateId].refCnt == 0)
    {
        long long delta = endClock - g_profilerState.fixedEvents[stateId].beginClock;
        if (g_profilerState.fixedEvents[stateId].cnt == 0)
        {
            g_profilerState.fixedEvents[stateId].min = delta;
            g_profilerState.fixedEvents[stateId].max = delta;
        }
        g_profilerState.fixedEvents[stateId].min = min(delta, g_profilerState.fixedEvents[stateId].min);
        g_profilerState.fixedEvents[stateId].max = max(delta, g_profilerState.fixedEvents[stateId].max);
        g_profilerState.fixedEvents[stateId].sum += delta;
        g_profilerState.fixedEvents[stateId].sumsq += delta * delta;
        g_profilerState.fixedEvents[stateId].cnt++;
    }
}

//
// Measure throughput given a bytes in an *Begin/*End block.
// Works with fixed or custom events.
//
int PERF_PROFILER_API ProfilerThroughputBegin(int eventId)
{
    LOCK
    if (!g_profilerState.init) return 0;
    g_profilerState.throughputIdx++;
    if (g_profilerState.throughputIdx >= THROUGHPUT_EVENT_MAX)
    {
        RuntimeError("Profiler error: Out of profiler slots.");
    }
    int stateId = g_profilerState.throughputIdx;
    g_profilerState.throughputEvents[stateId].eventId = eventId;
    g_profilerState.throughputEvents[stateId].beginClock = GetClock();

    return stateId;
}

int PERF_PROFILER_API ProfilerThroughputBegin(const char* description)
{
    UNUSED(description);
    return 0;
}

void PERF_PROFILER_API ProfilerThroughputEnd(int stateId, long long bytes)
{
    long long endClock = GetClock();
    LOCK
    if (!g_profilerState.init) return;
    if (stateId == g_profilerState.throughputIdx)
    {
        g_profilerState.throughputIdx--;
    }
    long long KBytesPerSec = ((bytes * g_profilerState.clockFrequency) / 1000) / (endClock - g_profilerState.throughputEvents[stateId].beginClock);
    int eventId = g_profilerState.throughputEvents[stateId].eventId;
    if (g_profilerState.fixedEvents[eventId].cnt == 0)
    {
        g_profilerState.fixedEvents[eventId].min = KBytesPerSec;
        g_profilerState.fixedEvents[eventId].max = KBytesPerSec;
    }
    g_profilerState.fixedEvents[eventId].min = min(KBytesPerSec, g_profilerState.fixedEvents[eventId].min);
    g_profilerState.fixedEvents[eventId].max = max(KBytesPerSec, g_profilerState.fixedEvents[eventId].max);
    g_profilerState.fixedEvents[eventId].sum += KBytesPerSec;
    g_profilerState.fixedEvents[eventId].sumsq += KBytesPerSec * KBytesPerSec;
    g_profilerState.fixedEvents[eventId].cnt++;
}

//
// Generate reports and release all resources.
//
void PERF_PROFILER_API ProfilerClose()
{
    { LOCK g_profilerState.init = false; }
    LockClose();

    _wmkdir(s2ws(g_profilerState.profilerDir).c_str());

    time_t currentTime;
    time(&currentTime);
    struct tm* timeInfo = localtime(&currentTime);

    char timeStr[32];
    strftime(timeStr, sizeof(timeStr), "%Y-%m-%d_%H-%M-%S", timeInfo);

    char fileName[256];
    sprintf_s(fileName, sizeof(fileName)-1, "%s/%s_" PROFILER_SUMMARY, g_profilerState.profilerDir, timeStr);
    ProfilerGenerateReport(fileName, timeStr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility functions.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Return freqeuncy in Hz of clock.
//
long long GetClockFrequency()
{
    long frequency;

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
void ProfilerGenerateReport(const char* fileName, const char* timeStamp)
{
    FILE* f = fopenOrDie(fileName, "wt");

    fprintfOrDie(f, "CNTK Performance Profiler Summary Report\n\n");
    fprintfOrDie(f, "Time Stamp: %s\n\n", timeStamp);

    fprintfOrDie(f, "Description................ ............Mean ..........StdDev .............Min .............Max ...........Count\n");

    for (int evtIdx = 0; evtIdx < profilerEvtMax; evtIdx++)
    {
        fprintfOrDie(f, "%s ", c_profilerEvtDesc[evtIdx]);

        switch (c_profilerEvtType[evtIdx])
        {
        case profilerEvtTime:
            {
                if (g_profilerState.fixedEvents[evtIdx].refCnt == 0 && g_profilerState.fixedEvents[evtIdx].cnt > 0)
                {
                    char str[32];

                    double mean = ((double)g_profilerState.fixedEvents[evtIdx].sum / (double)g_profilerState.fixedEvents[evtIdx].cnt) / (double)g_profilerState.clockFrequency;
                    FormatTimeStr(str, sizeof(str), mean);
                    fprintfOrDie(f, "%s ", str);

                    double sum = (double)g_profilerState.fixedEvents[evtIdx].sum / (double)g_profilerState.clockFrequency;
                    double sumsq = (double)g_profilerState.fixedEvents[evtIdx].sumsq / (double)g_profilerState.clockFrequency / (double)g_profilerState.clockFrequency;
                    double stdDev = sqrt((sumsq - (pow(sum, 2.0) / (double)g_profilerState.fixedEvents[evtIdx].cnt)) / (double)g_profilerState.fixedEvents[evtIdx].cnt);
                    FormatTimeStr(str, sizeof(str), stdDev);
                    fprintfOrDie(f, "%s ", str);

                    FormatTimeStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].min / (double)g_profilerState.clockFrequency);
                    fprintfOrDie(f, "%s ", str);

                    FormatTimeStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].max / (double)g_profilerState.clockFrequency);
                    fprintfOrDie(f, "%s ", str);
                }
            }
            break;

        case profilerEvtThroughput:
            {
                if (g_profilerState.throughputIdx == -1 && g_profilerState.fixedEvents[evtIdx].cnt > 0)
                {
                    char str[32];

                    double mean = ((double)g_profilerState.fixedEvents[evtIdx].sum / (double)g_profilerState.fixedEvents[evtIdx].cnt);
                    FormatThroughputStr(str, sizeof(str), mean);
                    fprintfOrDie(f, "%s ", str);

                    double stdDev = sqrt(((double)g_profilerState.fixedEvents[evtIdx].sumsq - (pow((double)g_profilerState.fixedEvents[evtIdx].sum, 2.0) / (double)g_profilerState.fixedEvents[evtIdx].cnt)) / (double)g_profilerState.fixedEvents[evtIdx].cnt);
                    FormatThroughputStr(str, sizeof(str), stdDev);
                    fprintfOrDie(f, "%s ", str);

                    FormatThroughputStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].min);
                    fprintfOrDie(f, "%s ", str);

                    FormatThroughputStr(str, sizeof(str), (double)g_profilerState.fixedEvents[evtIdx].max);
                    fprintfOrDie(f, "%s ", str);
                }
            }
            break;
        }

        fprintfOrDie(f, "%16d\n", g_profilerState.fixedEvents[evtIdx].cnt);
    }

    fclose(f);
}

//
// String formatting helpers for reporting.
//
void FormatTimeStr(char* str, size_t strLen, double value)
{
    // Format seconds value appropriatelly to ms or us.
    if (value < 0.001)
    {
        sprintf_s(str, strLen, "%13.3g us", value * 1000000.0);
    }
    else if (value < 1.0)
    {
        sprintf_s(str, strLen, "%13.3g ms", value * 1000.0);
    }
    else
    {
        sprintf_s(str, strLen, "%14.3g s", value);
    }
}

void FormatThroughputStr(char* str, size_t strLen, double value)
{
    if (value < 1000.0)
    {
        sprintf_s(str, strLen, "%11.3g KBps", value);
    }
    else if (value < 1000000.0)
    {
        sprintf_s(str, strLen, "%11.3g MBps", value / 1000.0);
    }
    else
    {
        sprintf_s(str, strLen, "%11.3g GBps", value / 1000000.0);
    }
}


}}}
