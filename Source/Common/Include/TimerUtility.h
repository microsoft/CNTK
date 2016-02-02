#pragma once

#include <string>

#define MILLI_PER_SEC 1000
#define MICRO_PER_SEC 1000000
#define NANO_PER_SEC 1000000000
#define MILLI_PER_NANO 1000000
#define MICRO_PER_NANO 1000

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: Replace with std::chrono once VS2015 ships (current std::chrono implementation in VS2013 uses low precision clock)
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

    double ElapsedSeconds()
    {
        return ElapsedMicroseconds() / static_cast<double>(MICRO_PER_SEC);
    }

private:
    long long ElapsedMicroseconds();
    long long GetStamp();

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

} } }
