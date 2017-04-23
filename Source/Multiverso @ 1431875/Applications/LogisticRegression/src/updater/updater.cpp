#include "updater/updater.h"

#include "updater/sgd_updater.h"
#include "updater/ftrl_updater.h"

#include "util/common.h"
#include "util/log.h"

namespace logreg {

template <typename EleType>
void Updater<EleType>::Update(DataBlock<EleType>* data, 
  DataBlock<EleType>* delta) {
  Process(delta);

  if (data->sparse()) {
    DEBUG_CHECK(delta->sparse());
    SparseBlockIter<EleType> iter(delta);
    EleType* pval;
    while (iter.Next()) {
      pval = data->Get(iter.Key());
      if (pval == nullptr) {
        data->Set(iter.Key(), -(*iter.Value()));
      } else {
        *pval -= (*iter.Value());
      }
    }
  } else {
    EleType* rawdata = static_cast<EleType*>(data->raw());
    EleType* rawdelta = static_cast<EleType*>(delta->raw());
    size_t size = delta->size();
    DEBUG_CHECK(delta->size() == data->size());
    for (size_t j = 0; j < size; ++j) {
      rawdata[j] -= rawdelta[j];
    }
  }
}

inline double max(double a, double b) {
  return a > b ? a : b;
}

template <typename EleType>
SGDUpdater<EleType>::SGDUpdater(const Configure& config) :
initial_learning_rate_(config.learning_rate),
learning_rate_(config.learning_rate),
learning_rate_coef_(config.learning_rate_coef),
update_count_(0),
minibatch_size_(config.minibatch_size) {
}

template <typename EleType>
void SGDUpdater<EleType>::Process(DataBlock<EleType>* delta) {
  if (delta->sparse()) {
    SparseBlockIter<EleType> iter(delta);
    while (iter.Next()) {
      *iter.Value() = (EleType)(*iter.Value() * learning_rate_);
    }
  } else {
    EleType* rawdelta = static_cast<EleType*>(delta->raw());
    size_t size = delta->size();
    for (size_t j = 0; j < size; ++j) {
      rawdelta[j] = (EleType)(rawdelta[j] * learning_rate_);
    }
  }
  ++update_count_;
  learning_rate_ = max(1e-3,
    initial_learning_rate_ - (update_count_ / 
    (learning_rate_coef_ * minibatch_size_)));
  Log::Write(Debug, "SGD learning rate : %f\n", learning_rate_);
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(SGDUpdater);

template <typename EleType>
FTRLUpdater<EleType>::FTRLUpdater(const Configure& config) {
}

template <typename EleType>
void FTRLUpdater<EleType>::Update(DataBlock<EleType>* data,
  DataBlock<EleType>* delta) {
  DEBUG_CHECK(delta->sparse());

  SparseBlockIter<FTRLGradient<EleType>> iter(
    (DataBlock<FTRLGradient<EleType>>*)delta);
  auto d = (DataBlock<FTRLEntry<EleType>> *)data;
  FTRLEntry<EleType> *pval;
  FTRLGradient<EleType> *g;
  while (iter.Next()) {
    pval = d->Get(iter.Key());
    g = iter.Value();
    if (pval == nullptr) {
      d->Set(iter.Key(), FTRLEntry<EleType>(-g->delta_z, -g->delta_n));
    } else {
      pval->n -= g->delta_n;
      pval->z -= g->delta_z;
      DEBUG_CHECK(pval->n >= 0);
      pval->sqrtn = (EleType)sqrt(pval->n);
    }
  }
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(FTRLUpdater);

template <typename EleType>
Updater<EleType>* Updater<EleType>::Get(const Configure& config) {
  const std::string& type = config.updater_type;
  Log::Write(Info, "updater type %s\n", type.c_str());
  if (config.objective_type == "ftrl" || type == "ftrl") {
    return new FTRLUpdater<EleType>(config);
  } else if (type == "sgd") {
    return new SGDUpdater<EleType>(config);
  }
  // default updater
  return new Updater<EleType>();
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(Updater);
}  // namespace logreg
