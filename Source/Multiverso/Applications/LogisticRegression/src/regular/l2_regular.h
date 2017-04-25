#ifndef LOGREG_REGULAR_L2_REGULAR_H_
#define LOGREG_REGULAR_L2_REGULAR_H_

#include "regular.h"

namespace logreg {

template <typename EleType>
class L2Regular : public Regular<EleType> {
public:
  explicit L2Regular(const Configure& config);
  virtual ~L2Regular() = default;

  EleType Calculate(
    size_t key,
    DataBlock<EleType>*model);
};

}  // namespace logreg

#endif  // LOGREG_REGULAR_L2_REGULAR_H_
