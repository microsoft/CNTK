#pragma once

#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

class Timer
{
public:
    Timer()
        : m_start(0), m_end(0)
    {
    }

    void Start();
    void Stop();
    void Restart();

    double ElapsedSeconds();

private:
    long long m_start;
    long long m_end;
};

class ScopeTimer
{
    Timer m_aggregateTimer;
    size_t m_verbosity;
    std::string m_message;

public:
    ScopeTimer(size_t verbosity, const std::string& message)
        : m_verbosity(verbosity), m_message(message)
    {
        if (m_verbosity > 2)
        {
            m_aggregateTimer.Start();
        }
    }

    ~ScopeTimer()
    {
        if (m_verbosity > 2)
        {
            m_aggregateTimer.Stop();
            double time = m_aggregateTimer.ElapsedSeconds();
            fprintf(stderr, m_message.c_str(), time);
        }
    }
};

class Clock
{
public:
    static long long GetTimeStamp();
    static long long GetTicksPerSecond();
};

}}}
