#include "multiverso/table/array_table.h"

#include "multiverso/io/io.h"
#include "multiverso/multiverso.h"
#include "multiverso/util/log.h"
#include "multiverso/updater/updater.h"

namespace multiverso {

template <typename T>
ArrayWorker<T>::ArrayWorker(size_t size) : WorkerTable(), size_(size) {
  num_server_ = MV_NumServers();
  server_offsets_.push_back(0);
  CHECK(size_ > MV_NumServers());
  integer_t length = static_cast<integer_t>(size_) / MV_NumServers();
  for (auto i = 1; i < MV_NumServers(); ++i) {
    server_offsets_.push_back(i * length); // may not balance
  }
  server_offsets_.push_back(size_);
  Log::Debug("worker %d create arrayTable with %d elements.\n", MV_Rank(), size);
}

template <typename T>
ArrayWorker<T>::ArrayWorker(const ArrayTableOption<T> &option)
: ArrayWorker<T>(option.size) {
}

template <typename T>
void ArrayWorker<T>::Get(T* data, size_t size) {
  CHECK(size == size_);
  data_ = data;
  integer_t all_key = -1;
  Blob whole_table(&all_key, sizeof(integer_t));
  WorkerTable::Get(whole_table);
  Log::Debug("worker %d getting all parameters.\n", MV_Rank());
}

template <typename T>
int ArrayWorker<T>::GetAsync(T* data, size_t size) {
  CHECK(size == size_);
  data_ = data;
  integer_t all_key = -1;
  Blob whole_table(&all_key, sizeof(integer_t));
  return WorkerTable::GetAsync(whole_table);
}

template <typename T>
void ArrayWorker<T>::Add(T* data, size_t size, const AddOption* option) {
  CHECK(size == size_);
  integer_t all_key = -1;

  Blob key(&all_key, sizeof(integer_t));
  Blob val(data, sizeof(T) * size);
  WorkerTable::Add(key, val, option);
  Log::Debug("worker %d adding parameters with size of %d.\n", MV_Rank(), size);
}

template <typename T>
int ArrayWorker<T>::AddAsync(T* data, size_t size, const AddOption* option) {
  CHECK(size == size_);
  integer_t all_key = -1;

  Blob key(&all_key, sizeof(integer_t));
  Blob val(data, sizeof(T) * size);
  return WorkerTable::AddAsync(key, val, option);
}

template <typename T>
int ArrayWorker<T>::Partition(const std::vector<Blob>& kv,
  MsgType,
  std::unordered_map<int, std::vector<Blob> >* out) {
  CHECK(kv.size() == 1 || kv.size() == 2 || kv.size() == 3);
  for (int i = 0; i < num_server_; ++i) (*out)[i].push_back(kv[0]);
  if (kv.size() >= 2) {
    CHECK(kv[1].size() == size_ * sizeof(T));
    for (int i = 0; i < num_server_; ++i) {
      Blob blob(kv[1].data() + server_offsets_[i] * sizeof(T),
        (server_offsets_[i + 1] - server_offsets_[i]) * sizeof(T));
      (*out)[i].push_back(blob);
      if (kv.size() == 3) {// update option blob
        (*out)[i].push_back(kv[2]);
      }
    }
  }
  return num_server_;
}

template <typename T>
void ArrayWorker<T>::ProcessReplyGet(std::vector<Blob>& reply_data) {
  CHECK(reply_data.size() == 2);
  int id = (reply_data[0]).As<int>();
  CHECK(reply_data[1].size<T>() == (server_offsets_[id + 1] - server_offsets_[id]));

  memcpy(data_ + server_offsets_[id], reply_data[1].data(), reply_data[1].size());
}

template <typename T>
ArrayServer<T>::ArrayServer(size_t size) : ServerTable() {
  server_id_ = MV_Rank();
  size_ = size / MV_NumServers();
  if (server_id_ == MV_NumServers() - 1) { // last server 
    size_ += size % MV_NumServers();
  }
  storage_.resize(size_);
  updater_ = Updater<T>::GetUpdater(size_);
  Log::Debug("server %d create arrayTable with %d elements of %d elements.\n", 
             server_id_, size_, size);
}

template <typename T>
ArrayServer<T>::ArrayServer(const ArrayTableOption<T> &option) 
: ArrayServer<T>(option.size) {
}

template <typename T>
void ArrayServer<T>::ProcessAdd(const std::vector<Blob>& data) {
  Blob keys = data[0], values = data[1];
  AddOption* option = nullptr;
  if (data.size() == 3)
    option = new AddOption(data[2].data(), data[2].size());
  // Always request whole table
  CHECK(keys.size<integer_t>() == 1 && keys.As<integer_t>() == -1); 
  CHECK(values.size() == size_ * sizeof(T));
  T* pvalues = reinterpret_cast<T*>(values.data());
  updater_->Update(size_, storage_.data(), pvalues, option);
  delete option;
}

template <typename T>
void ArrayServer<T>::ProcessGet(const std::vector<Blob>& data,
  std::vector<Blob>* result) {
  size_t key_size = data[0].size<integer_t>();
  CHECK(key_size == 1 && data[0].As<integer_t>() == -1); 
  // Always request the whole table
  Blob key(sizeof(integer_t)); key.As<integer_t>() = server_id_;
  Blob values(sizeof(T) * size_);
  T* pvalues = reinterpret_cast<T*>(values.data());
  updater_->Access(size_, storage_.data(), pvalues);
  result->push_back(key);
  result->push_back(values);
}

template <typename T>
void ArrayServer<T>::Store(Stream* s) {
  s->Write(storage_.data(), storage_.size() * sizeof(T));
}

template <typename T>
void ArrayServer<T>::Load(Stream* s) {
  s->Read(storage_.data(), storage_.size() * sizeof(T));
}

MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(ArrayWorker);
MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(ArrayServer);

}

