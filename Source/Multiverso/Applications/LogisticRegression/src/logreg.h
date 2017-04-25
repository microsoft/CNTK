#ifndef LOGREG_LOGREG_H_
#define LOGREG_LOGREG_H_

#include <string>

#include "data_type.h"
#include "model/model.h"
#include "configure.h"

namespace logreg {

// only support EleType = int/float/double
template <typename EleType>
class LogReg {
public:
  // \param config_file each line as: key=value
  explicit LogReg(const std::string &config_file);
  ~LogReg();
  
  void Train(const std::string& train_file);
  // config file should provide
  //  train file
  void Train();

  // will save output in result if result != nullptr
  // return test error
  double Test(const std::string& test_file, EleType**result = nullptr);
  // config file should provide 
  //  test file
  double Test(EleType**result = nullptr);
  // When model is too large, the program may crash...
  void SaveModel();
  void SaveModel(const std::string& model_file);

  // return the data block of model data
  DataBlock<EleType>* model() const;

private:
  Model<EleType> *model_;
  Configure* config_;
};

}  // namespace logreg

#endif  // LOGREG_LOGREG_H_
