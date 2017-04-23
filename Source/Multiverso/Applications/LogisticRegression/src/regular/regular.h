#ifndef LOGREG_REGULAR_REGULAR_H_
#define LOGREG_REGULAR_REGULAR_H_

#include <string>

#include "data_type.h"
#include "configure.h"

namespace logreg {
  
// provide regularization term
template <typename EleType>
class Regular {
public:
  // \param config should provide:
  //  input size
  //  output size
  explicit Regular(const Configure& config);
  virtual ~Regular() = default;
  // get regularization term
  virtual EleType Calculate(
    size_t key,
    DataBlock<EleType>*model);
  
  // factory method to get a new instance 
  // \param config should provide regular type
  //  and needed params for Regular initialization
  static Regular<EleType>* Get(const Configure& config);

protected:
  size_t input_size_;
  int output_size_;

  double regular_coef_;
};

}  // namespace logreg

#endif  // LOGREG_REGULAR_REGULAR_H_
