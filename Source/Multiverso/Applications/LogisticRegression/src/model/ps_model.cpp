#include "model/ps_model.h"

#include "util/common.h"
#include "util/log.h"

#include "util/sparse_table.h"
#include "util/ftrl_sparse_table.h"
#include "multiverso/util/configure.h"

namespace logreg {

template<typename EleType>
PSModel<EleType>::PSModel(Configure& config) :
  Model<EleType>(config),
  wait_id_(-1),
  buffer_index_(-1),
  count_sample_(0),
  push_time_(0.0),
  pull_time_(0.0),
  pull_count_(0),
  push_count_(0) {

  // set multiverso updater type
  multiverso::SetCMDFlag<std::string>("updater_type", "sgd");
  // start multiverso
  multiverso::MV_Init();
  // create table
  size_t size = (size_t)config.output_size * config.input_size;
  bool sparse = config.sparse;
  if (sparse) {
    if (this->ftrl_) {
      this->worker_table_ = static_cast<multiverso::WorkerTable*>(
        multiverso::MV_CreateTable(FTRLTableOption<EleType>(size)));
    } else {
      this->worker_table_ = static_cast<multiverso::WorkerTable*>(
        multiverso::MV_CreateTable(SparseTableOption<EleType>(size)));
    }
  } else {
    this->worker_table_ = static_cast<multiverso::WorkerTable*>(
      multiverso::MV_CreateTable(multiverso::ArrayTableOption<EleType>(size)));
  }
  
  std::stringstream ss;
  ss << "-";
  ss << multiverso::MV_WorkerId();
  config.output_model_file += ss.str();
  config.output_file += ss.str();

  Log::Write(Info, "Init ps model, size [%d, %lld]\n", 
    config.output_size, config.input_size);

  // init pipeline related members
  sync_frequency_ = config.sync_frequency;
  if (config.pipeline) {
    buffer_[0] = this->table_;
    buffer_index_ = 0;
    if (this->ftrl_) {
      buffer_[1] = (DataBlock<EleType>*)DataBlock<FTRLEntry<EleType>>
        ::GetBlock(true, size);
    } else {
      buffer_[1] = DataBlock<EleType>::GetBlock(config.sparse, size);
    }

    Log::Write(Info, "ps model using pipeline\n");
  }
  Log::Write(Info, "ps model with sync frequency %d\n", this->sync_frequency_);
}

template<typename EleType>
PSModel<EleType>::~PSModel() {
  if (buffer_index_ != -1) {
    if (wait_id_ != -1) {
      worker_table_->Wait(wait_id_);
    }
    delete buffer_[buffer_index_];
  }
  delete worker_table_;

  multiverso::MV_ShutDown();
}

template<typename EleType>
void PSModel<EleType>::DisplayTime() {
  if (push_count_ == 0 || pull_count_ == 0) {
    return;
  }
  Log::Write(Info, "worker %d average communication time : %fms push, \
    %fms pull\n", multiverso::MV_WorkerId(), push_time_ / push_count_, 
    pull_time_ / pull_count_);

  Model<EleType>::DisplayTime();

  pull_count_ = push_count_ = 0;
  push_time_ = pull_time_ = 0;
}

template<typename EleType>
int PSModel<EleType>::Predict(int count, Sample<EleType>**samples, 
  EleType**predicts) {
  int correct = 0;
  for (int i = 0; i < count; i += this->minibatch_size_) {
    int upper = i + this->minibatch_size_;
    upper = upper > count ? count : upper;
    for (int j = i; j < upper; ++j) {
      this->objective_->Predict(samples[j], this->table_, predicts[j]);
      if (this->objective_->Correct(samples[j]->label, predicts[j])) {
        ++correct;
      }
    }
    DoesNeedSync();
  }
  return correct;
}

template<typename EleType>
void PSModel<EleType>::Load(const std::string& model_file) {
  Model<EleType>::Load(model_file);
  // only load in one machine
  if (multiverso::MV_WorkerId() == 0) {
    if (this->table_->sparse()) {
      if (this->ftrl_) {
        SparseBlockIter<FTRLEntry<EleType>> iter(
          (DataBlock<FTRLEntry<EleType>>*)this->table_);
        while (iter.Next()) {
          ((DataBlock<FTRLGradient<EleType>>*)this->delta_)->
            Set(iter.Key(), FTRLGradient<EleType>(iter.Value()->z, 
            iter.Value()->n));
        }
        UpdateTable(this->delta_);
      } else {
        SparseBlockIter<EleType> iter(this->table_);
        while (iter.Next()) {
          *iter.Value() = -*iter.Value();
        }
        UpdateTable(this->table_);
        iter.Reset();
        while (iter.Next()) {
          *iter.Value() = -*iter.Value();
        }
      }
    } else {
      size_t size = this->table_->size();
      EleType* raw = static_cast<EleType*>(this->table_->raw());
      for (size_t i = 0; i < size; ++i) {
        raw[i] = -raw[i];
      }
      UpdateTable(this->table_);
      for (size_t i = 0; i < size; ++i) {
        raw[i] = -raw[i];
      }
    }
  }
  multiverso::MV_Barrier();
}

template<typename EleType>
void PSModel<EleType>::Store(const std::string& model_file) {
  // assure server process all add done
  if (buffer_index_ != -1 && wait_id_ != -1) {
    worker_table_->Wait(wait_id_);
    wait_id_ = -1;
  }

  multiverso::MV_Barrier();
  // without this, model will not the same in each worker
  PullWholeModel();

  Model<EleType>::Store(model_file);
}

template<typename EleType>
inline void PSModel<EleType>::DoesNeedSync() {
  if (++count_sample_ >= sync_frequency_) {
    if (buffer_index_ != -1) {
      GetPipelineTable();
    } else {
      PullModel();
    }
    
    count_sample_ -= sync_frequency_;
  }
}

template<typename EleType>
inline void PSModel<EleType>::UpdateTable(DataBlock<EleType>* delta) {
  ++push_count_;
  network_timer_.Start();
  if (delta->sparse()) {
    if (this->ftrl_) {
      ((FTRLWorkerTable<EleType>*)worker_table_)->AddAsync(
        (DataBlock<FTRLGradient<EleType>>*)delta);
    } else {
      this->updater_->Process(delta);
      ((SparseWorkerTable<EleType>*)worker_table_)->AddAsync(delta);
    }
  } else {
    this->updater_->Process(delta);
    ((multiverso::ArrayWorker<EleType>*)worker_table_)
      ->AddAsync(static_cast<EleType*>(delta->raw()), delta->size());
  }
  push_time_ += network_timer_.ElapseMilliSeconds();
  DoesNeedSync();
}

template<typename EleType>
inline void PSModel<EleType>::PullModel() {
  network_timer_.Start();
  if (this->table_->sparse()) {
    if (keys_->Size() < 2) {
      return;
    }
    SparseBlock<bool>* key;
    this->keys_->TryPop(key);
    delete key;
    this->keys_->Front(key);
    if (this->ftrl_) {
      ((FTRLWorkerTable<EleType>*)worker_table_)->Get(
        key,
        (DataBlock<FTRLEntry<EleType>>*)this->table_);
    } else {
      ((SparseWorkerTable<EleType>*)worker_table_)->Get(
        key,
        this->table_);
    }
  } else {
    ((multiverso::ArrayWorker<EleType>*)worker_table_)
      ->Get(static_cast<EleType*>(this->table_->raw()), this->table_->size());
  }
  Log::Write(Debug, "worker PULL model time %f\n",
    network_timer_.ElapseMilliSeconds());
  pull_time_ += network_timer_.ElapseMilliSeconds();
  ++pull_count_;
}

template<typename EleType>
inline void PSModel<EleType>::GetPipelineTable() {
  network_timer_.Start();
  if (wait_id_ != -1) {
    worker_table_->Wait(wait_id_);
  }
  this->table_ = buffer_[buffer_index_];
  buffer_index_ = 1 - buffer_index_;
  wait_id_ = -1;
  
  if (this->table_->sparse()) {
    if (keys_->Size() < 2) {
      return;
    }
    SparseBlock<bool>* key;
    this->keys_->TryPop(key);
    delete key;
      this->keys_->Front(key);
      if (this->ftrl_) {
        wait_id_ = ((FTRLWorkerTable<EleType>*)worker_table_)->GetAsync(
          key,
          (DataBlock<FTRLEntry<EleType>>*)buffer_[buffer_index_]);
      } else {
        wait_id_ = ((SparseWorkerTable<EleType>*)worker_table_)->GetAsync(
          key,
          buffer_[buffer_index_]);
      }
  } else {
    wait_id_ = ((multiverso::ArrayWorker<EleType>*)worker_table_)
      ->GetAsync(static_cast<EleType*>(buffer_[buffer_index_]->raw()),
      buffer_[buffer_index_]->size());
  }
  Log::Write(Debug, "worker pipeline PULL model time %f\n", 
    network_timer_.ElapseMilliSeconds());
  pull_time_ += network_timer_.ElapseMilliSeconds();
  ++pull_count_;
}

template<typename EleType>
void PSModel<EleType>::PullWholeModel() {
  if (this->table_->sparse()) {
    if (this->ftrl_) {
      ((FTRLWorkerTable<EleType>*)worker_table_)->Get(
        (DataBlock<FTRLEntry<EleType>>*)this->table_);
    } else {
      ((SparseWorkerTable<EleType>*)worker_table_)->Get(
        this->table_);
    }
  } else {
    ((multiverso::ArrayWorker<EleType>*)worker_table_)
      ->Get(static_cast<EleType*>(this->table_->raw()), this->table_->size());
  }
}

template<typename EleType>
void PSModel<EleType>::SetKeys(multiverso::MtQueue<SparseBlock<bool>*> *keys) {
  keys_ = keys;
  count_sample_ = 0;
  if (buffer_index_ != -1) {
    int count = 0;
    while (keys->Size() < 2 && count++ < 10) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    GetPipelineTable();
  }
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(PSModel);

}  // namespace logreg 
