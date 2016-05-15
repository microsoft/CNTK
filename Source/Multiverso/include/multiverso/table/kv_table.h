#ifndef MULTIVERSO_KV_TABLE_H_
#define MULTIVERSO_KV_TABLE_H_

#include "multiverso/table_interface.h"
#include "multiverso/util/log.h"

#include <unordered_map>
#include <vector>

namespace multiverso {

// A distributed shared std::unordered_map<Key, Val> table

// Key, Val should be the basic type
template <typename Key, typename Val>
class KVWorkerTable : public WorkerTable {
public:
  void Get(Key key) { WorkerTable::Get(Blob(&key, sizeof(Key))); }

  void Get(std::vector<Key>& keys) {
    WorkerTable::Get(Blob(&keys[0], sizeof(Key) * keys.size()));
  }

  void Add(Key key, Val value) {
    WorkerTable::Add(Blob(&key, sizeof(Key)), Blob(&value, sizeof(Val)));
  }

  void Add(std::vector<Key>& keys, std::vector<Val>& vals) {
    CHECK(keys.size() == vals.size());
    Blob keys_blob(&keys[0], sizeof(Key) * keys.size());
    Blob vals_blob(&vals[0], sizeof(Val) * vals.size());
    WorkerTable::Add(keys_blob, vals_blob);
  }

  std::unordered_map<Key, Val>& raw() { return table_; }
   
  int Partition(const std::vector<Blob>& kv, 
    std::unordered_map<int, std::vector<Blob> >* out) override {
    CHECK(kv.size() == 1 || kv.size() == 2);
    CHECK_NOTNULL(out);
    std::unordered_map<int, int> counts;
    Blob keys = kv[0];
    for (int i = 0; i < keys.size<Key>(); ++i) { // iterate as type Key
      int dst = static_cast<int>(keys.As<Key>(i) % MV_NumServers());
      ++counts[dst];
    }
    for (auto& it : counts) { // Allocate memory
      std::vector<Blob>& vec = (*out)[it.first];
      vec.push_back(Blob(it.second * sizeof(Key)));
      if (kv.size() == 2) vec.push_back(Blob(it.second * sizeof(Val)));
    }
    counts.clear();
    for (int i = 0; i < keys.size<Key>(); ++i) {
      int dst = static_cast<int>(keys.As<Key>(i) % MV_NumServers());
      (*out)[dst][0].As<Key>(counts[dst]) = keys.As<Key>(i);
      if (kv.size() == 2) 
        (*out)[dst][1].As<Val>(counts[dst]) = kv[1].As<Val>(i);
      ++counts[dst];
    }
    return static_cast<int>(out->size());
  }

  void ProcessReplyGet(std::vector<Blob>& data) override {
    CHECK(data.size() == 2);
    Blob keys = data[0], vals = data[1];
    CHECK(keys.size<Key>() == vals.size<Val>());
    for (int i = 0; i < keys.size<Key>(); ++i) {
      table_[keys.As<Key>(i)] = vals.As<Val>(i);
    }
  }

private:
  std::unordered_map<Key, Val> table_;
};

template <typename Key, typename Val>
class KVServerTable : public ServerTable {
public:
  void ProcessGet(const std::vector<Blob>& data, 
                  std::vector<Blob>* result) override {
    CHECK(data.size() == 1);
    CHECK_NOTNULL(result);
    Blob keys = data[0];
    result->push_back(keys); // also push the key
    result->push_back(Blob(keys.size<Key>() * sizeof(Val)));
    Blob& vals = (*result)[1];
    for (int i = 0; i < keys.size<Key>(); ++i) {
      vals.As<Val>(i) = table_[keys.As<Key>(i)];
    }
  }

  void ProcessAdd(const std::vector<Blob>& data) override {
    CHECK(data.size() == 2);
    Blob keys = data[0], vals = data[1];
    CHECK(keys.size<Key>() == vals.size<Val>());
    for (int i = 0; i < keys.size<Key>(); ++i) {
      table_[keys.As<Key>(i)] += vals.As<Val>(i);
    }
  }

  void Store(Stream* s) override{
    size_t size = table_.size();
    s->Write(&size, sizeof(size_t));
    for (auto& i : table_){
      s->Write(&i.first, sizeof(Key));
      s->Write(&i.second, sizeof(Val));
    }
  }
  void Load(Stream* s) override{
    size_t count;
    Key k;
    Val v;
    s->Read(&count, sizeof(size_t));
    for (int i = 0; i < count; ++i){
      s->Read(&k, sizeof(Key));
      s->Read(&v, sizeof(Val));
      table_[k] = v;
    }
  }

private:
  std::unordered_map<Key, Val> table_;
};

}

#endif // MULTIVERSO_KV_TABLE_H_
