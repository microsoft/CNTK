#ifndef LOGREG_MODEL_H_
#define LOGREG_MODEL_H_

#include <vector>
#include <string>

#include "data_type.h"
#include "configure.h"
#include "updater/updater.h"
#include "objective/objective.h"
#include "regular/regular.h"

#include "util/timer.h"
#include "multiverso/util/mt_queue.h"

namespace logreg {

// class for model data management
// local model
template<typename EleType>
class Model {
public:
  // initiate with config data
  // \param config should provide:
  //  objective type
  //  updater type
  //  input size
  //  output size
  explicit Model(Configure& config);
  virtual ~Model();
  // update model with #count samples
  // \return sum of train loss of every sample
  virtual float Update(int count, Sample<EleType>**samples);
  // \param input one input
  // \return correct number
  virtual int Predict(int count, Sample<EleType>**samples, EleType**predicts);
  // load model data from a binary file
  virtual void Load(const std::string& model_file);
  // write model data in binary method
  virtual void Store(const std::string& model_file);
  virtual void SetKeys(multiverso::MtQueue<SparseBlock<bool>*> *keys) {}
  virtual void DisplayTime();
  DataBlock<EleType>* table() const { return table_; }
  // factory method to get a new instance
  // \param config should contain model needed configs
  //    when use_ps=true, return a distributed model
  //    default use a local version
  static Model<EleType>* Get(Configure& config);

protected:
  // compute update delta
  virtual float GetGradient(Sample<EleType>* sample, DataBlock<EleType>* delta);
  // update table
  virtual void UpdateTable(DataBlock<EleType>* delta);

protected: 
  bool ftrl_;

  Objective<EleType>* objective_;
  Updater<EleType>* updater_;
  // local cache
  DataBlock<EleType>* table_;

  int num_row_;

  int minibatch_size_;

  DataBlock<EleType>* delta_;

  Timer timer_;
  double computation_time_;
  double compute_count_;
};

}  // namespace logreg

#endif  // LOGREG_MODEL_H_
