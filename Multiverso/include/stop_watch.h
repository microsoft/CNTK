#ifndef _MULTIVERSO_STOP_WATCH_H_
#define _MULTIVERSO_STOP_WATCH_H_

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
        long long tick_per_sec_;
        long long start_tick_;
        long long stop_tick_;
    };
}

#endif // _MULTIVERSO_TIMER_H_ 