#ifndef LOGREG_REGULAR_L1_REGULAR_H_
#define LOGREG_REGULAR_L1_REGULAR_H_

#include "regular.h"

namespace logreg {

template <typename EleType>
class L1Regular : public Regular<EleType> {
public:
  explicit L1Regular(const Configure& config);
  virtual ~L1Regular() = default;

  EleType Calculate(
    size_t key,
    DataBlock<EleType>*model);
};

}  // namespace logreg

#endif  // LOGREG_REGULAR_L1_REGULAR_H_
