#include "regular/regular.h"

#include <math.h>

#include "regular/l1_regular.h"
#include "regular/l2_regular.h"

#include "util/common.h"
#include "util/log.h"

namespace logreg {

template<typename EleType>
Regular<EleType>::Regular(const Configure& config) {
  this->input_size_ = config.input_size;
  this->output_size_ = config.output_size;
  this->regular_coef_ = config.regular_coef;
}

template<typename EleType>
EleType Regular<EleType>::Calculate(
  size_t key,
  DataBlock<EleType>*model) {
  return 0;
}

template<typename EleType>
L1Regular<EleType>::L1Regular(const Configure& config) :
Regular<EleType>(config) {
}

template<typename EleType>
EleType L1Regular<EleType>::Calculate(
  size_t key,
  DataBlock<EleType>*model) {
  EleType* pval = model->Get(key);
  // sgn(x) * regular_coef
  return (pval == nullptr || *pval == 0) ? 0
    : (EleType)(*pval > 0 ? this->regular_coef_ : -this->regular_coef_);
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(L1Regular);

template<typename EleType>
L2Regular<EleType>::L2Regular(const Configure& config) :
Regular<EleType>(config) {
}

template<typename EleType>
EleType L2Regular<EleType>::Calculate(
  size_t key,
  DataBlock<EleType>*model) {
  EleType* pval = model->Get(key);
  // abs(x) * regular_coef
  return pval == nullptr ? 0 : (EleType)(abs(*pval) * this->regular_coef_);
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(L2Regular);

template<typename EleType>
Regular<EleType>* Regular<EleType>::Get(const Configure& config) {
  const std::string &type = config.regular_type;
  Log::Write(Info, "Regular type %s\n", type.c_str());
  if (type == "L1") {
    return new L1Regular<EleType>(config);
  } else if (type == "L2") {
    return new L2Regular<EleType>(config);
  }
  return new Regular<EleType>(config);
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(Regular);
}  // namespace logreg 
