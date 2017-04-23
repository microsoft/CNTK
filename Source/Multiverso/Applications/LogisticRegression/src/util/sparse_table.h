#ifndef LOGREG_UTIL_SPARSE_TABLE_H_
#define LOGREG_UTIL_SPARSE_TABLE_H_

#include "data_type.h"
#include "multiverso/util/allocator.h"
#include "multiverso/table_interface.h"
#include "multiverso/updater/updater.h"

#include "multiverso/io/io.h"

namespace logreg {

template<typename EleType>
struct SparseTableOption;

template<typename EleType>
class SparseWorkerTable : public multiverso::WorkerTable {
protected:
  using Blob = multiverso::Blob;

public:
  explicit SparseWorkerTable(size_t size) :
  size_(size) {
    num_server_ = multiverso::MV_NumServers();

    Log::Write(Info, "SparseWorker %d create SparseTable with %lld elements\n",
      multiverso::MV_WorkerId(), size_);
  }
  explicit SparseWorkerTable(const SparseTableOption<EleType> &option) :
  SparseWorkerTable(option.size) {}
  virtual ~SparseWorkerTable() = default;
  // get all
  virtual int GetAsync(DataBlock<EleType>* data) {
    LR_CHECK(data != nullptr && data->sparse());
    data_ = data;
    size_t all_key = -1;
    Blob whole_table(&all_key, sizeof(size_t));
    return WorkerTable::GetAsync(whole_table);
  }
  virtual void Get(DataBlock<EleType>* data) {
    Wait(GetAsync(data));
  }
  virtual int GetAsync(SparseBlock<bool>* keys, DataBlock<EleType>* data) {
    data_ = data;
    data_->Clear();

    size_t size = keys->size();
    Blob key(size * sizeof(size_t));
    size_t* pkey = reinterpret_cast<size_t*>(key.data());

    SparseBlockIter<bool> iter(keys);
    while (iter.Next()) {
      *(pkey++) = iter.Key();
    }

    return WorkerTable::GetAsync(key);
  }
  virtual void Get(SparseBlock<bool>* keys, DataBlock<EleType>* data) {
    Wait(GetAsync(keys, data));
  }
  
  virtual int AddAsync(DataBlock<EleType>* data) {
    size_t size = data->size();
    if (size == 0) {
      return -1;
    }

    Blob key(size * sizeof(size_t));
    Blob val(sizeof(EleType) * size);

    size_t* keys = reinterpret_cast<size_t*>(key.data());
    EleType* vals = reinterpret_cast<EleType*>(val.data());

    SparseBlockIter<EleType> iter(data);
    while (iter.Next()) {
      *(keys++) = iter.Key();
      *(vals++) = *iter.Value();
    }

    return WorkerTable::AddAsync(key, val);
  }

  virtual int Partition(const std::vector<Blob>& kv, multiverso::MsgType,
    std::unordered_map<int, std::vector<Blob> >* out) {
    DEBUG_CHECK(kv.size() == 1 || kv.size() == 2);
    // get all
    if (kv.size() == 1 && kv[0].size<size_t>() == 1
      && kv[0].As<size_t>() == (size_t)-1) {
      for (int i = 0; i < num_server_; ++i) (*out)[i].push_back(kv[0]);
      return num_server_;
    }
    
    size_t size = kv[0].size<size_t>();
    DEBUG_CHECK(kv.size() == 1 || size * sizeof(EleType) == kv[1].size());

    size_t* keys = reinterpret_cast<size_t*>(kv[0].data());

    int *dsts = reinterpret_cast<int*>(multiverso::Allocator::Get()
      ->Alloc(size * sizeof(int)));
    // calculate destination
    size_t size_each_server = size_ / num_server_;
    for (size_t i = 0; i < size; ++i) {
      int dst = static_cast<int>(keys[i] / size_each_server);
      dst = dst >= num_server_ ? (num_server_ - 1) : dst;
      dsts[i] = multiverso::MV_ServerIdToRank(dst);
    }
    size_t *idx = reinterpret_cast<size_t*>(multiverso::Allocator::Get()
      ->Alloc(size * sizeof(size_t)));
    size_t *counts = reinterpret_cast<size_t*>(multiverso::Allocator::Get()
      ->Alloc(sizeof(size_t)* num_server_));
    memset(counts, 0, num_server_ * sizeof(size_t));

    for (size_t i = 0; i < size; ++i) {
      idx[i] = counts[dsts[i]]++;
    }

    // Allocate memory
    for (int i = 0; i < num_server_; ++i) {
      if (counts[i] == 0) {
        continue;
      }
      (*out)[i].push_back(Blob(counts[i] * sizeof(size_t)));
      if (kv.size() == 2) {
        (*out)[i].push_back(Blob(counts[i] * sizeof(EleType)));
      }
    }

    multiverso::Allocator::Get()->Free(reinterpret_cast<char*>(counts));
    // copy key and value
    
    for (size_t i = 0; i < size; ++i) {
      int dst = dsts[i];
      (reinterpret_cast<size_t*>((*out)[dst][0].data()))[idx[i]] = keys[i];
    }
    if (kv.size() == 2) {
      EleType* vals = reinterpret_cast<EleType*>(kv[1].data());
      for (size_t i = 0; i < size; ++i) {
        int dst = dsts[i];
        (reinterpret_cast<EleType*>((*out)[dst][1].data()))[idx[i]] = vals[i];
      }
    }
    multiverso::Allocator::Get()->Free(reinterpret_cast<char*>(dsts));
    multiverso::Allocator::Get()->Free(reinterpret_cast<char*>(idx));
    
    return static_cast<int>(out->size());
  }

  virtual void ProcessReplyGet(std::vector<Blob>& reply_data) {
    DEBUG_CHECK(reply_data.size() == 2 || reply_data.size() == 1);
    DEBUG_CHECK(data_ != nullptr);
    // no data
    if (reply_data.size() == 1) {
      return;
    }
    size_t *keys = reinterpret_cast<size_t*>(reply_data[0].data());
    EleType *vals = reinterpret_cast<EleType*>(reply_data[1].data());
    size_t size = reply_data[0].size() / sizeof(size_t);
    for (size_t i = 0; i < size; ++i) {
      data_->Set(keys[i], vals + i);
    }
  }

protected:
  DataBlock<EleType>* data_;  // not owned
  size_t size_;
  int num_server_;
};

template<typename EleType>
class SparseServerTable : public multiverso::ServerTable {
protected:
  using Blob = multiverso::Blob;

public:
  explicit SparseServerTable(size_t size) :
  count_(0) {
    int server_id = multiverso::MV_ServerId();
    int num_server = multiverso::MV_NumServers();

    if (size >= num_server) {
      size_ = size / num_server;
      offset_ = size_ * server_id;
      if (server_id == num_server - 1) {  // last server 
        size_ = size - offset_;
      }
    } else {
      offset_ = server_id;
      size_ = server_id < size ? 1 : 0;
    }

    storage_.resize(size_);
    keys_.resize(size_);

    Log::Write(Info, "SparseServer %d create table with %lld elements, \
      offset %lld\n", multiverso::MV_ServerId(), size_, offset_);
  }
  explicit SparseServerTable(const SparseTableOption<EleType> &option) :
    SparseServerTable(option.size) {}

  virtual ~SparseServerTable() = default;

  virtual void ProcessAdd(const std::vector<Blob>& data) {
    DEBUG_CHECK(data.size() == 2);
    // Timer timer;
    // timer.Start();
    size_t* keys = reinterpret_cast<size_t*>(data[0].data());
    EleType* vals = reinterpret_cast<EleType*>(data[1].data());
    size_t size = data[0].size<size_t>();
    
    for (size_t i = 0; i < size; ++i) {
      keys[i] -= offset_;
      DEBUG_CHECK(keys[i] < storage_.size() && keys[i] >= 0);
      storage_[keys[i]] -= vals[i];
      if (!keys_[keys[i]]) {
        ++count_;
        keys_[keys[i]] = true;
      }
    }
  }

  virtual void ProcessGet(const std::vector<Blob>& data,
    std::vector<Blob>* result) {
    DEBUG_CHECK(data.size() == 1);
    result->reserve(2);
    if (count_ == 0) {
      size_t none = -1;
      result->push_back(Blob(&none, sizeof(size_t)));
      return;
    }

    // request whole table
    if (data[0].size() == sizeof(size_t) && data[0].As<size_t>() == -1) {
      result->push_back(Blob(sizeof(size_t)* count_));
      result->push_back(Blob(sizeof(EleType)* count_));

      size_t *keys = reinterpret_cast<size_t*>((*result)[0].data());
      EleType*vals = reinterpret_cast<EleType*>((*result)[1].data());
      for (size_t i = 0; i < size_; ++i) {
        if (keys_[i]) {
          keys[i] = i + offset_;
          vals[i] = storage_[i];
        }
      }
    } else {
      size_t size = data[0].size<size_t>();
      size_t* keys = reinterpret_cast<size_t*>(data[0].data());
      result->push_back(Blob(size * sizeof(size_t)));
      result->push_back(Blob(size * sizeof(EleType)));
      size_t *pkeys = reinterpret_cast<size_t*>((*result)[0].data());
      EleType*pvals = reinterpret_cast<EleType*>((*result)[1].data());
      for (size_t i = 0; i < size; ++i) {
        DEBUG_CHECK(keys[i] - offset_ < storage_.size());
        pkeys[i] = keys[i];
        pvals[i] = storage_[keys[i] - offset_];
      }
    }
  }

  virtual void Store(multiverso::Stream* s) {
    // save size
    s->Write(&count_, sizeof(size_t));
    // save key
    for (size_t i = 0; i < size_; ++i) {
      if (keys_[i]) {
        s->Write(&i, sizeof(size_t));
      }
    }
    // save value
    s->Write(storage_.data(), sizeof(EleType)* size_);
  }

  virtual void Load(multiverso::Stream* s) {
    // read size
    s->Read(&count_, sizeof(size_t));
    // set key
    for (size_t i = 0; i < size_; ++i) {
      keys_[i] = false;
    }       
    size_t index;
    for (size_t i = 0; i < count_; ++i) {
      s->Read(&index, sizeof(size_t));
      keys_[index] = true;
    }
    // load value
    s->Read(storage_.data(), sizeof(EleType)* size_);
  }

protected:
  size_t size_;
  size_t offset_;
  size_t count_;
  std::vector<bool> keys_;
  std::vector<EleType> storage_;
};

template<typename EleType>
struct SparseTableOption {
  explicit SparseTableOption(size_t size) :
  size(size) {}
  size_t size;
  DEFINE_TABLE_TYPE(EleType, SparseWorkerTable, SparseServerTable);
};

}  // namespace logreg

#endif  // LOGREG_UTIL_SPARSE_TABLE_H_
