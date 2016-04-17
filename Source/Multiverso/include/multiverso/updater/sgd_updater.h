#ifndef MULTIVERSO_UPDATER_SGD_UPDATER_H_
#define MULTIVERSO_UPDATER_SGD_UPDATER_H_

#include "updater.h"

namespace multiverso {

template <typename T>
class SGDUpdater : public Updater<T> {
public:
  explicit SGDUpdater(size_t){
    Log::Debug("[SGDUpdater] Init. \n");
  }
  void Update(size_t num_element, T*data, T*delta,
    UpdateOption*, size_t offset) override {
    for (size_t index = 0; index < num_element; ++index) {
      data[index + offset] -= delta[index];
    }
  }
  ~SGDUpdater(){}
};

}

#endif // MULTIVERSO_UPDATER_ASGD_UPDATER_H_