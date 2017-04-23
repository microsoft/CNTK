#ifndef LOGREG_OBJECTIVE_FTRL_OBJECTIVE_H_
#define LOGREG_OBJECTIVE_FTRL_OBJECTIVE_H_

#include "objective.h"

namespace logreg {

template <typename EleType>
class FTRLObjective : public Objective<EleType> {
public:
  explicit FTRLObjective(const Configure& config);

  ~FTRLObjective();

  float Gradient(Sample<EleType>* sample,
    DataBlock<EleType>* model,
    DataBlock<EleType>* gradient);

  float Predict(Sample<EleType>*sample,
    DataBlock<EleType>* model, EleType* predict);

private:
  float Predict(Sample<EleType>*sample,
    DataBlock<EleType>* model, EleType* predict, DataBlock<EleType>* w);
  EleType sgn(const EleType x);
  
private:
  Objective<EleType> *objective_;

  double lambda1_;
  double lambda2_;
  double alpha_;
  double beta_;
};

}  // namespace logreg 

#endif  // LOGREG_OBJECTIVE_FTRL_OBJECTIVE_H_
