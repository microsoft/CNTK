#include "objective/objective.h"

#include <math.h>

#include "util/common.h"
#include "util/log.h"

#include "objective/sigmoid_objective.h"
#include "objective/softmax_objective.h"
#include "objective/ftrl_objective.h"

#include "multiverso/multiverso.h"

namespace logreg {

inline float MathLog(float x) {
  return log(x < 0.000001f ? 0.000001f : x);
}

inline int Round(float x) {
  return x < 0.5 ? 0 : 1;
}

template<typename EleType>
Objective<EleType>::Objective(const Configure &config) {
  this->input_size_ = config.input_size;
  this->output_size_ = config.output_size;
  regular_ = Regular<EleType>::Get(config);
}

template<typename EleType>
Objective<EleType>::~Objective() {
  delete regular_;
}

template<typename EleType>
inline float Objective<EleType>::Gradient(Sample<EleType>* sample,
  DataBlock<EleType>*model, DataBlock<EleType>* gradient) {
  EleType* loss = new EleType[this->output_size_];
  float train_loss = Predict(sample, model, loss);
  
  Diff(sample->label, loss);
  AddRegularization(sample, model, loss, gradient);
  delete []loss;

  return train_loss;
}

template<typename EleType>
inline float Objective<EleType>::Loss(Sample<EleType>*sample,
  EleType* predict) {
  if (output_size_ == 1) {
    return pow(static_cast<float>(*predict - (sample->label == 1 ? 1.0 : 0.0)), 2);
  }

  float ret = 0.0f;
  for (int i = 0; i < output_size_; ++i) {
    ret += pow(static_cast<float>(predict[i] - (sample->label == i ? 1 : 0)), 2);
  }
  return ret / output_size_;
}

template<typename EleType>
inline void Objective<EleType>::AddRegularization(Sample<EleType>*sample,
  DataBlock<EleType>* model,
  EleType* loss,
  DataBlock<EleType>* gradient) {
  if (model->sparse()) {
    size_t offset = 0;
    size_t size = sample->values.size();
    for (int i = 0; i < this->output_size_; ++i) {
      // each input
      for (size_t j = 0; j < size; ++j) {
        size_t key = sample->keys[j] + offset;
        EleType val = static_cast<EleType>(sample->values[j] * loss[i])
          + regular_->Calculate(key, model);
        
        EleType* pval = gradient->Get(key);
        if (pval == nullptr) {
          gradient->Set(key, val);
        } else {
          *pval += val;
        }
      }
      offset += this->input_size_;
    }
  } else {
    EleType* rawgrad = static_cast<EleType*>(gradient->raw());
    EleType* rawinput = sample->values.data();

    size_t index = 0;
    for (int i = 0; i < this->output_size_; ++i) {
      for (size_t j = 0; j < this->input_size_; ++j) {
        rawgrad[index] += static_cast<EleType>(rawinput[j] * loss[i]
          + regular_->Calculate(index, model));
        ++index;
      }
    }
  }
}

template<typename EleType>
inline void Objective<EleType>::Diff(int label, EleType*diff) {
  if (this->output_size_ == 1) {
    *diff -= static_cast<int>(label == 1);
  } else {
    for (int i = 0; i < this->output_size_; ++i) {
      diff[i] -= static_cast<int>(label == i);
    }
  }
}

template<typename EleType>
float Objective<EleType>::Predict(Sample<EleType>*sample,
  DataBlock<EleType>*model, EleType* predict) {
  for (int i = 0; i < this->output_size_; ++i) {
    predict[i] = Dot((size_t)i * this->input_size_, model, sample);
  }
  return this->Loss(sample, predict);
}

template<typename EleType>
bool Objective<EleType>::Correct(const int label, EleType*output) {
  if (this->output_size_ == 1) {
    return (Round(static_cast<float>(*output)) - static_cast<int>(label == 1)) == 0;
  }

  EleType max = *(output++);
  int idx = 0;
  for (int i = 1; i < this->output_size_; ++i) {
    if (*output > max) {
      idx = i;
      max = *output;
    }
    ++output;
  }

  return idx == label;
}

template<typename EleType>
SigmoidObjective<EleType>::SigmoidObjective(const Configure& config) :
Objective<EleType>(config) {
  if (config.output_size != 1) {
    Log::Write(Fatal, "SigmoidObjective should be used for \
                      output size = 1, with tag = [0/1]\n");
  }
}

template<typename EleType>
inline float SigmoidObjective<EleType>::Gradient(Sample<EleType>* sample,
  DataBlock<EleType>*model, DataBlock<EleType>* gradient) {
  EleType loss = static_cast<EleType>(Sigmoid(sample, model));
  float train_loss = this->Loss(sample, &loss);
  this->Diff(sample->label, &loss);
  this->AddRegularization(sample, model, &loss, gradient);
  return train_loss;
}

template<typename EleType>
inline float SigmoidObjective<EleType>::Predict(Sample<EleType>* sample,
  DataBlock<EleType>* model, EleType* predict) {
  *predict = static_cast<EleType>(Sigmoid(sample, model));
  return this->Loss(sample, predict);
}

template<typename EleType>
inline float SigmoidObjective<EleType>::Sigmoid(Sample<EleType>* sample,
  DataBlock<EleType>*model) {
  return static_cast<float>(1.0f / (1.0f + exp(-Dot(0, model, sample))));
}

template<typename EleType>
inline float SigmoidObjective<EleType>::Loss(Sample<EleType>*sample,
  EleType* predict) {
  if (sample->label == 1) {
    return -MathLog(static_cast<float>(*predict));
  }
  return -MathLog(1.0f - static_cast<float>(*predict));
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(SigmoidObjective);

template<typename EleType>
SoftmaxObjective<EleType>::SoftmaxObjective(const Configure& config) :
Objective<EleType>(config) {
  if (config.output_size < 2) {
    Log::Write(Fatal, "SoftmaxObjective should be used for output size > 1\n");
  }
}

template<typename EleType>
inline float SoftmaxObjective<EleType>::Predict(Sample<EleType>* sample,
  DataBlock<EleType>* model, EleType* predict) {
  float sum = Sigmoid(sample, model, predict);
  for (int i = 0; i < this->output_size_; ++i) {
    predict[i] = static_cast<EleType>(predict[i] / sum);
  }
   return this->Loss(sample, predict);
}

template<typename EleType>
float SoftmaxObjective<EleType>::Sigmoid(Sample<EleType>* sample,
  DataBlock<EleType>*model, EleType*sigmoid) {
  for (int i = 0; i < this->output_size_; ++i) {
    sigmoid[i] = Dot(i*this->input_size_, model, sample);
  }
  float max = static_cast<float>(sigmoid[0]);
  for (int i = 1; i < this->output_size_; ++i) {
    max = static_cast<float>(max < sigmoid[i] ? sigmoid[i] : max);
  }
  float sum = 0.0f;
  for (int i = 0; i < this->output_size_; ++i) {
    sigmoid[i] = static_cast<EleType>(exp(sigmoid[i] - max));
    sum += static_cast<float>(sigmoid[i]);
  }
  return sum;
}

template<typename EleType>
inline float SoftmaxObjective<EleType>::Loss(Sample<EleType>*sample,
  EleType* predict) {
  float ret = 0.0f;
  for (int i = 0; i < this->output_size_; ++i) {
    if (sample->label == i){
      ret -= MathLog(static_cast<float>(predict[i]));
    }
    else {
      ret -= MathLog(1.0f - static_cast<float>(predict[i]));
    }
  }
  return ret / this->output_size_;
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(SoftmaxObjective)

template<typename EleType>
FTRLObjective<EleType>::FTRLObjective(const Configure& config) :
Objective<EleType>(config) {
  LR_CHECK(config.sparse == true);

  if (config.output_size == 1) {
    objective_ = new SigmoidObjective<EleType>(config);
  } else {
    objective_ = new SoftmaxObjective<EleType>(config);
  }
  // initiate from config
  lambda1_ = config.lambda1;
  lambda2_ = config.lambda2;
  beta_ = config.beta;
  // avoid further computing
  alpha_ = 1.0 / config.alpha;
}

template<typename EleType>
FTRLObjective<EleType>::~FTRLObjective() {
  delete objective_;
}

template<typename EleType>
float FTRLObjective<EleType>::Gradient(Sample<EleType>* sample,
  DataBlock<EleType>* model,
  DataBlock<EleType>* gradient) {
  EleType* loss = new EleType[this->output_size_];
  auto w_ = DataBlock<EleType>::GetBlock(true, model->size());

  float train_loss = Predict(sample, model, loss, w_);

  this->Diff(sample->label , loss);

  auto g = (DataBlock<FTRLGradient<EleType>>*)gradient;
  auto entry = (DataBlock<FTRLEntry<EleType>>*)model;
  size_t offset = 0;
  for (int i = 0; i < this->output_size_; ++i) {
    size_t size = sample->keys.size();
    for (size_t j = 0; j < size; ++j) {
      EleType delta_z;

      EleType delta_g = sample->values[j] * loss[i];
      EleType square_g = static_cast<EleType>(pow(delta_g, 2));

      size_t key = sample->keys[j] + offset;
      EleType *w = w_->Get(key);
      if (w == nullptr) {
        delta_z = -delta_g;
      } else {
        FTRLEntry<EleType> *en = entry->Get(key);
        if (en == nullptr) {
          delta_z = static_cast<EleType>(alpha_ * delta_g);
        } else {
          delta_z = static_cast<EleType>(alpha_ * (sqrt(en->n + square_g) - en->sqrtn));
        }
        delta_z = delta_z * (*w) - delta_g;
      }
      // delta_n
      delta_g = -square_g;
      g->Set(key, FTRLGradient<EleType>(static_cast<EleType>(delta_z), 
        static_cast<EleType>(delta_g)));
    }
    offset += this->input_size_;
  }
  delete[]loss;
  delete w_;
  return train_loss;
}

template<typename EleType>
float FTRLObjective<EleType>::Predict(Sample<EleType>* sample,
  DataBlock<EleType>* model, EleType* predict) {
  auto w = DataBlock<EleType>::GetBlock(true, model->size());
  float test_loss = Predict(sample, model, predict, w);
  delete w;
  return test_loss;
}

template<typename EleType>
float FTRLObjective<EleType>::Predict(Sample<EleType>*sample,
  DataBlock<EleType>* model, EleType* predict, DataBlock<EleType>* w) {
  auto entry = (DataBlock<FTRLEntry<EleType>>*)model;
  w->Clear();

  size_t offset = 0;
  for (size_t i = 0; i < this->output_size_; ++i) {
    for (size_t j = 0; j < sample->values.size(); ++j) {
      FTRLEntry<EleType> *en = entry->Get(sample->keys[j] + offset);
      if (en != nullptr && abs(en->z) > lambda1_) {
        EleType val = static_cast<EleType>((sgn(en->z) * lambda1_ - en->z)
          / ((beta_ + en->sqrtn) * alpha_ + lambda2_));
        w->Set(sample->keys[j] + offset, val);
      }
    }
    offset += this->input_size_;
  }

  return objective_->Predict(sample, w, predict);
}

template<typename EleType>
EleType FTRLObjective<EleType>::sgn(const EleType x) {
  return static_cast<EleType>(x > 0 ? 1 : (x < 0 ? -1 : 0));
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(FTRLObjective);

template<typename EleType>
Objective<EleType>* Objective<EleType>::Get(
  const Configure &config) {
  const std::string& type = config.objective_type;
  Log::Write(Info, "Objective type %s\n", type.c_str());

  if (type == "sigmoid") {
    return new SigmoidObjective<EleType>(config);
  } else if (type == "softmax") {
    return new SoftmaxObjective<EleType>(config);
  } else if (type == "ftrl") {
    return new FTRLObjective<EleType>(config);
  }

  return new Objective<EleType>(config);
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(Objective);
}  // namespace logreg 
