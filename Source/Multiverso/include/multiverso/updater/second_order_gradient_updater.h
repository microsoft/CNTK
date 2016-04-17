#ifndef MULTIVERSO_UPDATER_SECOND_ORDER_GRADIENT_UPDATER_H_
#define MULTIVERSO_UPDATER_SECOND_ORDER_GRADIENT_UPDATER_H_

#include <cmath>
#include <vector>

#include "updater.h"


namespace multiverso {

// [TODO(qiwye)]:rename the class to Shuxin Zheng's algorithms
template <typename T>
class SecondOrderUpdater : public Updater<T> {
public:
  explicit SecondOrderUpdater(size_t size) :
    size_(size) {
    Log::Debug("[SecondOrderUpdater] Init with size = %d. \n", size_);
    shadow_copies_.resize(MV_NumWorkers(), std::vector<T>(size_));
    historic_g_sqr_.resize(MV_NumWorkers(), std::vector<T>(size_));
  }
  void Update(size_t num_element, T*data, T*delta,
    UpdateOption* option, size_t offset) override {
    auto g_sqr_data_ = historic_g_sqr_.at(option->worker_id());
    auto copies_data_ = shadow_copies_.at(option->worker_id());
    for (size_t index = 0; index < num_element; ++index) {
      // gsqr = (1 - r) * g^2 + r * gsqr
      g_sqr_data_[index + offset] =
        (1 - option->rho()) * delta[index] * delta[index]
        / option->learning_rate() / option->learning_rate() +
        option->rho() * g_sqr_data_[index + offset];

      data[index + offset] -= delta[index] + option->lambda() *
        std::sqrt(g_sqr_data_[index + offset]) *
        (data[index + offset] - copies_data_[index + offset]);

      // caching each worker's latest version of parameter
      copies_data_[index + offset] = data[index + offset];
    }
  }
  ~SecondOrderUpdater() {
    shadow_copies_.clear();
    historic_g_sqr_.clear();
  }

private:
  float QuakeRsqrt(float number) {
    float x = number * 0.5f, y = number;
    std::uint32_t i;
    std::memcpy(&i, &y, sizeof(float));
    i = 0x5f3759df - (i >> 1);
    std::memcpy(&y, &i, sizeof(float));
    return y * (1.5f - (x * y * y));
  }

protected:
  std::vector< std::vector<T>> shadow_copies_;
  std::vector< std::vector<T>> historic_g_sqr_;

  size_t size_;
};
}  // namespace multiverso

#endif  // MULTIVERSO_UPDATER_SECOND_ORDER_GRADIENT_UPDATER_H_
