#ifndef LOGREG_OBJECTIVE_SIGMOID_OBJECTIVE_H_
#define LOGREG_OBJECTIVE_SIGMOID_OBJECTIVE_H_

#include "objective.h"

namespace logreg {

template<typename EleType>
class SigmoidObjective : public Objective<EleType> {
public:
  explicit SigmoidObjective(const Configure& config);

  float Gradient(Sample<EleType>* sample,
    DataBlock<EleType>* model,
    DataBlock<EleType>* gradient);

  float Predict(Sample<EleType>*sample,
    DataBlock<EleType>* model, EleType* predict);

private:
  float Sigmoid(Sample<EleType>* sample,
    DataBlock<EleType>*model);
  float Loss(Sample<EleType>*sample, EleType* predict);
};

}  // namespace logreg

#endif  // LOGREG_OBJECTIVE_SIGMOID_OBJECTIVE_H_
