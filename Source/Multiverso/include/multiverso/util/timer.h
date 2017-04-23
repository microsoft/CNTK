#ifndef MULTIVERSO_TIMER_H_
#define MULTIVERSO_TIMER_H_

#include <chrono>
#include <string>

namespace multiverso {

class Timer {
public:
  Timer();

  // Restart the timer
  void Start();

  // Get elapsed milliseconds since last Timer::Start
  double elapse();

private:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  TimePoint start_point_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_TIMER_H_
