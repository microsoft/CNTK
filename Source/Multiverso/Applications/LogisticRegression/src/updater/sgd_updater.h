#ifndef LOGREG_UPDATER_SGD_UPDATER_H_
#define LOGREG_UPDATER_SGD_UPDATER_H_

#include "updater.h"

namespace logreg {

template <typename EleType>
class SGDUpdater : public Updater<EleType> {
public:
  explicit SGDUpdater(const Configure& config);
  void Process(DataBlock<EleType>* delta);

private:
  double initial_learning_rate_;
  double learning_rate_;
  double learning_rate_coef_;
  size_t update_count_;
  int minibatch_size_;
};

}  // namespace logreg

#endif  // LOGREG_UPDATER_SGD_UPDATER_H_
