#include "TimerUtility.h"
#include <chrono>
using namespace std::chrono;

namespace Microsoft { namespace MSR { namespace CNTK {

void Timer::Start()
{
    m_start = Clock::GetTimeStamp();
}

void Timer::Restart()
{
    m_start = m_end = 0;
    Start();
}

void Timer::Stop()
{
    m_end = Clock::GetTimeStamp();
}

double Timer::ElapsedSeconds()
{
    if (m_start == 0)
    {
        return 0.0; // the timer hasn't been started yet
    }

    long long diff = 0;

    if (m_end != 0)
    {
        diff = m_end - m_start;
    }
    else
    {
        diff = Clock::GetTimeStamp() - m_start;
    }

    if (diff < 0)
    {
        diff = 0;
    }

    high_resolution_clock::duration dur(diff);
    long long nsec = duration_cast<nanoseconds>(dur).count();
    return static_cast<double>(nsec) / 1e9;
}

long long Clock::GetTimeStamp()
{
    // This takes 20-30 ns.
    return high_resolution_clock::now().time_since_epoch().count();
}

long long Clock::GetTicksPerSecond()
{
    typedef high_resolution_clock::period TickPeriod;
    static_assert(TickPeriod::den % TickPeriod::num == 0, "Ticks per second is not an integer");
    return TickPeriod::den / TickPeriod::num;
}

}}}
