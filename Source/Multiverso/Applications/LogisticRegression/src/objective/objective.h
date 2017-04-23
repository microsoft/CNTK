#ifndef LOGREG_OBJECTIVE_OBJECTIVE_H_
#define LOGREG_OBJECTIVE_OBJECTIVE_H_

#include <string>

#include "data_type.h"
#include "configure.h"
#include "regular/regular.h"

namespace logreg {

// provide methods for predict and calculate gradient 
template<typename EleType>
class Objective {
public:
  // \param config should provide:
  //  input size
  //  output size
  //  regular type
  explicit Objective(const Configure& config);
  virtual ~Objective();
  // return train loss
  virtual float Gradient(Sample<EleType>* sample,
    DataBlock<EleType>* model,
    DataBlock<EleType>* gradient);
  // return test loss
  virtual float Predict(Sample<EleType>*sample,
    DataBlock<EleType>* model, EleType* predict);

  virtual bool Correct(const int label, EleType*predict);
  
  // factory method to get a new instance
  // \param config should contain objective type
  //  and params for Objective initialization
  static Objective<EleType>* Get(const Configure& config);

protected:
  // diff -= (label == i)
  virtual void Diff(int label, EleType*diff);
  virtual void AddRegularization(Sample<EleType>*sample,
    DataBlock<EleType>* model,
    EleType* loss,
    DataBlock<EleType>* gradient);
  virtual float Loss(Sample<EleType>*sample, EleType* predict);

protected:
  Regular<EleType> *regular_;

  size_t input_size_;
  int output_size_;
};

}  // namespace logreg

#endif  // LOGREG_OBJECTIVE_OBJECTIVE_H_
