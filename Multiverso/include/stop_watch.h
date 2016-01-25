#ifndef _MULTIVERSO_STOP_WATCH_H_
#define _MULTIVERSO_STOP_WATCH_H_

#include<cstdint>

namespace multiverso
{
    class StopWatch
    {
    public:
        StopWatch();
        void Start();
        void Stop();

        double ElapsedSecondsToNow();
        double ElapsedMillisecondsToNow();
        double ElapsedSecondsToStop();
        double ElapsedMillisecondsToStop();

    private:
        int64_t tick_per_sec_;
        int64_t start_tick_;
        int64_t stop_tick_;
    };
}

#endif // _MULTIVERSO_TIMER_H_ 