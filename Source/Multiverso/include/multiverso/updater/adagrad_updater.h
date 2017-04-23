#ifndef MULTIVERSO_UPDATER_ADAGRAD_UPDATER_H_
#define MULTIVERSO_UPDATER_ADAGRAD_UPDATER_H_

#include "multiverso/updater/updater.h"
#include "multiverso/util/log.h"

#include <vector>
#include <cmath>
#include <cstdint>


namespace multiverso {

template <typename T>
class AdaGradUpdater : public Updater<T> {
public:
  explicit AdaGradUpdater(size_t size):
    e(1e-6f), size_(size) {  
    historic_g_sqr_.resize(MV_NumWorkers(), std::vector<T>(size_));
    Log::Debug("[AdaGradUpdater] Init with size = %d, e = %f. historic_size = %d\n", size_, e, historic_g_sqr_.size());
  }

  void Update(size_t num_element, T* data, T* delta, 
              AddOption* option, size_t offset) override {

    auto g_sqr_data_ = historic_g_sqr_.at(option->worker_id());
    for (size_t index = 0; index < num_element; ++index) {
      g_sqr_data_[index + offset] -=
        delta[index] * delta[index] / option->learning_rate() / 
        option->learning_rate();

      //[TODO(qiwye)] sqrt take too much time
      data[index + offset] -= option->rho() /
        std::sqrt(g_sqr_data_[index + offset] + e) *
        delta[index] / option->learning_rate();

      //data[index + offset] -= option->rho() *
      //  QuakeRsqrt(g_sqr_data_[index + offset] + e) *
      //  delta[index] / option->learning_rate();
    }
  }


private:

  float QuakeRsqrt(float number){
    float x = number * 0.5f, y = number;
    std::uint32_t i;
    std::memcpy(&i, &y, sizeof(float));
    i = 0x5f3759df - (i >> 1);
    std::memcpy(&y, &i, sizeof(float));
    return y * (1.5f - (x * y * y));
  }

protected:
    std::vector< std::vector<T>> historic_g_sqr_;
    float e;
    size_t size_;
};

}

#endif // MULTIVERSO_UPDATER_ADAGRAD_UPDATER_H_
