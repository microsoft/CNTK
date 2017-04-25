#ifndef LOGREG_UTIL_FTRL_SPARSE_TABLE_H_
#define LOGREG_UTIL_FTRL_SPARSE_TABLE_H_

#include "util/sparse_table.h"

namespace logreg {

template<typename EleType>
struct FTRLTableOption;

template<typename EleType>
class FTRLWorkerTable : public SparseWorkerTable<FTRLGradient<EleType>> {
protected:
  using Blob = multiverso::Blob;

public:
  explicit FTRLWorkerTable(size_t size) :
    SparseWorkerTable<FTRLGradient<EleType>>(size) {}

  explicit FTRLWorkerTable(const FTRLTableOption<EleType> &option) :
    FTRLWorkerTable(option.size) {}

  int GetAsync(DataBlock<FTRLEntry<EleType>>* data) {
    LR_CHECK(data != nullptr && data->sparse());
    this->data_ = (DataBlock<FTRLGradient<EleType>>*)data;
    size_t all_key = -1;
    Blob whole_table(&all_key, sizeof(size_t));
    return multiverso::WorkerTable::GetAsync(whole_table);
  }
  void Get(DataBlock<FTRLEntry<EleType>>* data) {
    this->Wait(GetAsync(data));
  }
  int GetAsync(SparseBlock<bool>* keys, DataBlock<FTRLEntry<EleType>>* data) {
    LR_CHECK(keys != nullptr && data != nullptr && data->sparse());
    data->Clear();
    this->data_ = (DataBlock<FTRLGradient<EleType>>*)data;

    size_t size = keys->size();
    Blob key(size * sizeof(size_t));
    size_t* pkey = reinterpret_cast<size_t*>(key.data());

    SparseBlockIter<bool> iter(keys);
    while (iter.Next()) {
      *(pkey++) = iter.Key();
    }

    return multiverso::WorkerTable::GetAsync(key);
  }
  void Get(SparseBlock<bool>* keys, DataBlock<FTRLEntry<EleType>>* data) {
    this->Wait(GetAsync(keys, data));
  }

  void ProcessReplyGet(std::vector<multiverso::Blob>& reply_data) {
    DEBUG_CHECK(reply_data.size() == 2 || reply_data.size() == 1);
    DEBUG_CHECK(this->data_ != nullptr);
    // no data
    if (reply_data.size() == 1) {
      return;
    }
    size_t *keys = reinterpret_cast<size_t*>(reply_data[0].data());
    auto vals = reinterpret_cast<FTRLGradient<EleType>*>(reply_data[1].data());
    size_t size = reply_data[0].size<size_t>();
    auto data = (DataBlock<FTRLEntry<EleType>>*)this->data_;
    for (size_t i = 0; i < size; ++i) {
      data->Set(keys[i], FTRLEntry<EleType>(vals+i));
    }
  }
};

template<typename EleType>
class FTRLServerTable : public SparseServerTable<FTRLGradient<EleType>> {
public:
  explicit FTRLServerTable(size_t size) : 
    SparseServerTable<FTRLGradient<EleType>>(size) { }

  explicit FTRLServerTable(const FTRLTableOption<EleType> &option) :
    FTRLServerTable(option.size) {}
};

template<typename EleType>
struct FTRLTableOption {
  explicit FTRLTableOption(size_t size) :
  size(size) {}
  size_t size;
  DEFINE_TABLE_TYPE(EleType, FTRLWorkerTable, FTRLServerTable);
};

}  // namespace logreg

#endif  // LOGREG_UTIL_FTRL_SPARSE_TABLE_H_
