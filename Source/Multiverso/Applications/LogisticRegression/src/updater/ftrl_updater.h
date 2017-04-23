#ifndef LOGREG_UPDATER_FTRL_UPDATER_
#define LOGREG_UPDATER_FTRL_UPDATER_

#include "updater.h"

namespace logreg {

template <typename EleType>
class FTRLUpdater : public Updater<EleType> {
public:
  explicit FTRLUpdater(const Configure& config);
  void Update(DataBlock<EleType>* data, DataBlock<EleType>* delta);
};

}  // namespace logreg

#endif  // LOGREG_UPDATER_FTRL_UPDATER_
