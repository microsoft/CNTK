#ifndef LOGREG_OBJECTIVE_SOFTMAX_OBJECTIVE_H_
#define LOGREG_OBJECTIVE_SOFTMAX_OBJECTIVE_H_

#include "objective.h"

namespace logreg {

template <typename EleType>
class SoftmaxObjective : public Objective<EleType> {
public:
  explicit SoftmaxObjective(const Configure& config);

  virtual float Predict(Sample<EleType>*sample,
    DataBlock<EleType>* model, EleType* predict);

protected:
  float Sigmoid(Sample<EleType>* sample,
    DataBlock<EleType>*model, EleType*sigmoid);
  float Loss(Sample<EleType>*sample, EleType* predict);
};

}  // namespace logreg 

#endif  // LOGREG_OBJECTIVE_SOFTMAX_OBJECTIVE_H
