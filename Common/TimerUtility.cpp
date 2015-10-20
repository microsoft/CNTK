#include "TimerUtility.h"
#include <assert.h>
#ifdef WIN32
#include <Windows.h>
static LARGE_INTEGER s_ticksPerSecond;
static BOOL s_setFreq = QueryPerformanceFrequency(&s_ticksPerSecond);
#else
#include <time.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {
    long long Timer::GetStamp()
    {
#ifdef WIN32
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        return li.QuadPart;
#else
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts); // Works on Linux

        long long ret = ts.tv_sec * NANO_PER_SEC + ts.tv_nsec;

        return ret;
#endif
    }

    void Timer::Start()
    {
        m_start = GetStamp();
    }

    void Timer::Restart()
    {
        m_start = m_end = 0;
        Start();
    }

    void Timer::Stop()
    {
        m_end = GetStamp();
    }

    long long Timer::ElapsedMicroseconds()
    {
        assert(m_start != 0);
        long long diff = 0;

        if (m_end != 0)
        {
            diff = m_end - m_start;
        }
        else
        {
            diff = GetStamp() - m_start;
        }

        if (diff < 0)
        {
            diff = 0;
        }

#ifdef WIN32
        assert(s_setFreq == TRUE);
        return (diff * MICRO_PER_SEC) / s_ticksPerSecond.QuadPart;
#else
        return diff / MICRO_PER_NANO;
#endif
    }

}}}
