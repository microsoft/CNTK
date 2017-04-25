#include "model/model.h"
#include "model/ps_model.h"

#include "util/log.h"
#include "util/common.h"
#include "updater/updater.h"
#include "multiverso/io/io.h"

namespace logreg {

template<typename EleType>
Model<EleType>::Model(Configure& config) :
  updater_(nullptr),
  computation_time_(0.0),
  compute_count_(0) {
  num_row_ = config.output_size;
  minibatch_size_ = config.minibatch_size;

  size_t size = (size_t)config.input_size * num_row_;

  // a little trick
  if (config.objective_type == "ftrl") {
    ftrl_ = true;
    table_ = (DataBlock<EleType>*)DataBlock<FTRLEntry<EleType>>
      ::GetBlock(true, size);
    delta_ = (DataBlock<EleType>*)DataBlock<FTRLGradient<EleType>>
      ::GetBlock(true, size);
  } else {
    ftrl_ = false;
    table_ = DataBlock<EleType>::GetBlock(config.sparse, size);
    delta_ = DataBlock<EleType>::GetBlock(config.sparse, size);
  }

  table_->Clear();  // will set value to zero when dense

  if (config.init_model_file != "") {
    Load(config.init_model_file);
  }

  updater_ = Updater<EleType>::Get(config);

  objective_ = Objective<EleType>::Get(config);

  Log::Write(Info, "Init local model, size [%d, %d]\n", 
    num_row_, config.input_size);
}

template<typename EleType>
Model<EleType>::~Model() {
  delete objective_;
  delete updater_;

  delete table_;
  delete delta_;
}

template<typename EleType>
inline float Model<EleType>::GetGradient(Sample<EleType>* sample, 
  DataBlock<EleType>* delta) {
  return objective_->Gradient(sample, table_, delta);
}

template<typename EleType>
float Model<EleType>::Update(int count, Sample<EleType>** samples) {
  float train_loss = 0.0f;
  // process each batch
  for (int i = 0; i < count; i += minibatch_size_) {
    ++compute_count_;
    timer_.Start();
    // compute delta
    delta_->Clear();
    int upper = i + minibatch_size_;
    upper = upper > count ? count : upper;
    for (int j = i; j < upper; ++j) {
      train_loss += GetGradient(samples[j], delta_);
    }
    
    // calculate and average delta
    int batch_size = upper - i;
    if (batch_size > 1) {
      if (delta_->sparse()) {
        if (ftrl_) {
          SparseBlockIter<FTRLGradient<EleType>> iter
            ((DataBlock<FTRLGradient<EleType>>*)delta_);
          while (iter.Next()) {
            iter.Value()->delta_z = (EleType)(iter.Value()->delta_z 
              / static_cast<double>(batch_size));
            iter.Value()->delta_n = (EleType)(iter.Value()->delta_n 
              / static_cast<double>(batch_size));
          }
        } else {
          SparseBlockIter<EleType> iter(delta_);
          while (iter.Next()) {
            (*iter.Value()) = (EleType)(*iter.Value() 
              / static_cast<double>(batch_size)); 
          }
        }
      } else {
        EleType* raw = static_cast<EleType*>(delta_->raw());
        for (size_t i = 0; i < delta_->size(); ++i) {
          raw[i] = (EleType)(raw[i] / static_cast<double>(batch_size));
        }
      }
    }

    computation_time_ += timer_.ElapseMilliSeconds();
    // update delta
    UpdateTable(delta_);
  }
  return train_loss;
}

template<typename EleType>
void Model<EleType>::DisplayTime() {
  if (compute_count_ == 0) {
    return;
  }
  Log::Write(Info, "average computation time: %fms\n", 
    computation_time_ / compute_count_);

  computation_time_ = 0;
  compute_count_ = 0;
}

template<typename EleType>
inline void Model<EleType>::UpdateTable(DataBlock<EleType>* delta) {
  // Log::Write(Debug, "Local model updating %d rows\n", update_idx_.size());
  timer_.Start();
  updater_->Update(table_, delta);
  computation_time_ += timer_.ElapseMilliSeconds();
}

template<typename EleType>
int Model<EleType>::Predict(int count, Sample<EleType>**samples, 
  EleType**predicts) {
  int correct(0);
  for (int i = 0; i < count; ++i) {
    this->objective_->Predict(samples[i], this->table_, predicts[i]);
    if (objective_->Correct(samples[i]->label, predicts[i])) {
      ++correct;
    }
  }
  return correct;
}

template<typename EleType>
void Model<EleType>::Load(const std::string& model_file) {
  auto stream = multiverso::StreamFactory::GetStream(
    multiverso::URI(model_file),
    multiverso::FileOpenMode::BinaryRead);
  if (table_->sparse()) {
    size_t size;
    size_t key;
    stream->Read(&size, sizeof(size_t));
    if (ftrl_) {
      FTRLEntry<EleType> val;
      for (size_t i = 0; i < size; ++i) {
        stream->Read(&key, sizeof(size_t));
        stream->Read(&val, sizeof(FTRLEntry<EleType>));
        ((DataBlock<FTRLEntry<EleType>>*)table_)->Set(key, &val);
      }
    } else {
      EleType val;
      for (size_t i = 0; i < size; ++i) {
        stream->Read(&key, sizeof(size_t));
        stream->Read(&val, sizeof(EleType));
        table_->Set(key, &val);
      }
    }
  } else {
    stream->Read(table_->raw(), table_->size() * sizeof(EleType));
  }
  delete stream;
  Log::Write(Info, "Load model from file %s\n", model_file.c_str());
}

template<typename EleType>
void Model<EleType>::Store(const std::string& model_file) {
  auto stream = multiverso::StreamFactory::GetStream(
    multiverso::URI(model_file),
    multiverso::FileOpenMode::BinaryWrite);
  if (table_->sparse()) {
    size_t tmp = table_->size();
    stream->Write(&tmp, sizeof(size_t));
    if (ftrl_) {
      SparseBlockIter<FTRLEntry<EleType>> iter(
        (DataBlock<FTRLEntry<EleType>>*)table_);
      while (iter.Next()) {
        tmp = iter.Key();
        stream->Write(&tmp, sizeof(size_t));
        stream->Write(iter.Value(), sizeof(FTRLEntry<EleType>));
      }
    } else {
      SparseBlockIter<EleType> iter(table_);
      while (iter.Next()) {
        tmp = iter.Key();
        stream->Write(&tmp, sizeof(size_t));
        stream->Write(iter.Value(), sizeof(EleType));
      }
    }
  } else {
    stream->Write(table_->raw(), table_->size() * sizeof(EleType));
  }
  delete stream;
}

template<typename EleType>
Model<EleType>* Model<EleType>::Get(Configure& config) {
  if (config.use_ps) {
    return new PSModel<EleType>(config);
  } else {
    return new Model<EleType>(config);
  }
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(Model);

} // namespace logreg
