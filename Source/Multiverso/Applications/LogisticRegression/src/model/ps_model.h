#ifndef LOGREG_MODEL_PS_MODEL_H_
#define LOGREG_MODEL_PS_MODEL_H_

#include <string>
#include <queue>

#include "model.h"

#include "multiverso/multiverso.h"
#include "multiverso/table_interface.h"
#include "multiverso/table/array_table.h"
#include "multiverso/util/async_buffer.h"

#include "util/timer.h"

namespace logreg {

template <typename EleType>
class PSModel : public Model<EleType> {
public:
  explicit PSModel(Configure& config);
  ~PSModel();
  int Predict(int count, Sample<EleType>**samples, EleType**predicts);
  void Load(const std::string& model_file);
  void Store(const std::string& model_file);
  void SetKeys(multiverso::MtQueue<SparseBlock<bool>*> *keys);
  void DisplayTime();

private:
  // use multiverso table add interface
  void UpdateTable(DataBlock<EleType>* delta);
  void PullModel();
  // sync table if needed
  void DoesNeedSync();
  void PullWholeModel();
  void GetPipelineTable();

private:
  // multiverso table
  multiverso::WorkerTable* worker_table_;
  // for pipeline sync
  DataBlock<EleType>* buffer_[2]; 
  int wait_id_;
  int buffer_index_;
  // works when not pipeline
  int count_sample_;
  int sync_frequency_;

  multiverso::MtQueue<SparseBlock<bool>*> *keys_;

  Timer network_timer_;
  double push_time_;
  double pull_time_;
  size_t pull_count_;
  size_t push_count_;
};

}  // namespace logreg

#endif  // LOGREG_MODEL_PS_MODEL_H_
