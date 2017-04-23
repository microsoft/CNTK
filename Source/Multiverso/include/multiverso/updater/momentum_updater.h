#ifndef MULTIVERSO_UPDATER_MOMENTUM_UPDATER_H_
#define MULTIVERSO_UPDATER_MOMENTUM_UPDATER_H_

#include "updater.h"
#include <vector>

namespace multiverso {

template <typename T>
class MomentumUpdater : public Updater<T> {
public:
  explicit MomentumUpdater(size_t size) : size_(size) {
    Log::Debug("[SmoothGradientUpdater] Init with size = %d. \n", size_);
    smooth_gradient_.resize(size_);
  }

  void Update(size_t num_element, T* data, T* delta, 
              AddOption* option, size_t offset) override {
    for (size_t index = 0; index < num_element; ++index) {
      smooth_gradient_[index + offset] = 
        option->momentum() * smooth_gradient_[index + offset] 
        + (1 - option->momentum()) * delta[index];
      data[index + offset] -= smooth_gradient_[index + offset];
    }
  }

  ~MomentumUpdater() { smooth_gradient_.clear(); }
protected:
  std::vector<T> smooth_gradient_;
  size_t size_;
};

}

#endif // MULTIVERSO_UPDATER_MOMENTUM_UPDATER_H_