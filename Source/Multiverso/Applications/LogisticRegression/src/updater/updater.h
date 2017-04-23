#ifndef LOGREG_UPDATER_UPDATER_H_
#define LOGREG_UPDATER_UPDATER_H_

#include <string>

#include "data_type.h"
#include "configure.h"

namespace logreg {

template <typename EleType>
class Updater {
public:
  virtual ~Updater() = default;

  virtual void Update(DataBlock<EleType>* data, 
    DataBlock<EleType>* delta);

  virtual void Process(DataBlock<EleType>* delta) {}

  // factory method to get a new instance
  // \param config should provide updater type and 
  //  params for updater initiate
  static Updater<EleType>* Get(const Configure& config);

protected:
  int row_size_;
  int num_row_;
};

}  // namespace logreg

#endif  // LOGREG_UPDATER_UPDATER_H_
