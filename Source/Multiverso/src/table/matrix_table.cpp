#include "multiverso/table/matrix_table.h"

#include <vector>

#include "multiverso/io/io.h"
#include "multiverso/multiverso.h"
#include "multiverso/util/log.h"
#include "multiverso/util/quantization_util.h"
#include "multiverso/updater/updater.h"

namespace multiverso {

template <typename T>
MatrixWorkerTable<T>::MatrixWorkerTable(const MatrixTableOption<T>& option) :
MatrixWorkerTable(option.num_row, option.num_col) {}

template <typename T>
MatrixWorkerTable<T>::MatrixWorkerTable(integer_t num_row, integer_t num_col) :
  WorkerTable(), num_row_(num_row), num_col_(num_col) {
  row_size_ = num_col * sizeof(T);
  get_reply_count_ = 0;

  num_server_ = MV_NumServers();
  //  compute row offsets in all servers
  server_offsets_.push_back(0);
  integer_t length = num_row / num_server_;
  integer_t offset = length;
  if (length > 0) {
    int i = 0;
    while (length > 0 && offset < num_row && ++i < num_server_) {
      server_offsets_.push_back(offset);
      offset += length;
    }
    server_offsets_.push_back(num_row);
  } else {
    int i = 0;
    offset += 1;
    while (offset < num_row && ++i < num_server_) {
      server_offsets_.push_back(offset);
      offset += 1;
    }
    server_offsets_.push_back(num_row);
  }
  // using actual number of servers
  num_server_ = static_cast<int>(server_offsets_.size() - 1);

  Log::Debug("[Init] worker =  %d, type = matrixTable, size =  [ %d x %d ].\n",
    MV_Rank(), num_row, num_col);
  row_index_ = new T*[num_row_ + 1];
}

template <typename T>
MatrixWorkerTable<T>::~MatrixWorkerTable() {
  server_offsets_.clear();
  delete[]row_index_;
}

template <typename T>
void MatrixWorkerTable<T>::Get(T* data, size_t size) {
  CHECK(size == num_col_ * num_row_);
  integer_t whole_table = -1;
  Get(whole_table, data, size);
}

template <typename T>
void MatrixWorkerTable<T>::Get(integer_t row_id, T* data, size_t size) {
  if (row_id >= 0) CHECK(size == num_col_);
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  if (row_id == -1) {
    row_index_[num_row_] = data;
  } else {
    row_index_[row_id] = data;  // data_ = data;
  }
  WorkerTable::Get(Blob(&row_id, sizeof(integer_t)));
  Log::Debug("[Get] worker = %d, #row = %d\n", MV_Rank(), row_id);
}

template <typename T>
void MatrixWorkerTable<T>::Get(const std::vector<integer_t>& row_ids,
  const std::vector<T*>& data_vec,
  size_t size) {
  CHECK(size == num_col_);
  CHECK(row_ids.size() == data_vec.size());
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  for (auto i = 0; i < row_ids.size(); ++i) {
    row_index_[row_ids[i]] = data_vec[i];
  }
  WorkerTable::Get(Blob(row_ids.data(), sizeof(integer_t)* row_ids.size()));
  Log::Debug("[Get] worker = %d, #rows_set = %d\n", MV_Rank(), row_ids.size());
}

template <typename T>
void MatrixWorkerTable<T>::Get(T* data, size_t size, integer_t* row_ids,
  integer_t row_ids_size) {
  CHECK(size == num_col_ * row_ids_size);
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  for (auto i = 0; i < row_ids_size; ++i) {
    row_index_[row_ids[i]] = &data[i * num_col_];
  }
  Blob ids_blob(row_ids, sizeof(integer_t) * row_ids_size);
  WorkerTable::Get(ids_blob);
  Log::Debug("[Get] worker = %d, #rows_set = %d\n", MV_Rank(), row_ids_size);
}

template <typename T>
void MatrixWorkerTable<T>::Add(T* data, size_t size, const AddOption* option) {
  CHECK(size == num_col_ * num_row_);
  integer_t whole_table = -1;
  Add(whole_table, data, size, option);
}

template <typename T>
void MatrixWorkerTable<T>::Add(integer_t row_id, T* data, size_t size,
                                              const AddOption* option) {
  if (row_id >= 0) CHECK(size == num_col_);
  Blob ids_blob(&row_id, sizeof(integer_t));
  Blob data_blob(data, size * sizeof(T));
  WorkerTable::Add(ids_blob, data_blob, option);
  Log::Debug("[Add] worker = %d, #row = %d\n", MV_Rank(), row_id);
}

template <typename T>
void MatrixWorkerTable<T>::Add(const std::vector<integer_t>& row_ids,
                               const std::vector<T*>& data_vec,
                               size_t size,
                               const AddOption* option) {
  CHECK(size == num_col_);
  Blob ids_blob(&row_ids[0], sizeof(integer_t)* row_ids.size());
  Blob data_blob(row_ids.size() * row_size_);
  // copy each row
  for (auto i = 0; i < row_ids.size(); ++i) {
    memcpy(data_blob.data() + i * row_size_, data_vec[i], row_size_);
  }
  WorkerTable::Add(ids_blob, data_blob, option);
  Log::Debug("[Add] worker = %d, #rows_set = %d\n", MV_Rank(), row_ids.size());
}

template <typename T>
void MatrixWorkerTable<T>::Add(T* data, size_t size, integer_t* row_ids,
  integer_t row_ids_size,
  const AddOption* option) {
  CHECK(size == num_col_ * row_ids_size);
  Blob ids_blob(row_ids, sizeof(integer_t) * row_ids_size);
  Blob data_blob(data, row_ids_size * row_size_);
  WorkerTable::Add(ids_blob, data_blob, option);
  Log::Debug("[Add] worker = %d, #rows_set = %d\n", MV_Rank(), row_ids_size);
}

template <typename T>
int MatrixWorkerTable<T>::GetAsync(T* data, size_t size) {
  CHECK(size == num_col_ * num_row_);
  integer_t whole_table = -1;
  return GetAsync(whole_table, data, size);
}

template <typename T>
int MatrixWorkerTable<T>::GetAsync(integer_t row_id, T* data, size_t size) {
  if (row_id >= 0) CHECK(size == num_col_);
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  if (row_id == -1) {
    row_index_[num_row_] = data;
  } else {
    row_index_[row_id] = data;  // data_ = data;
  }
  return WorkerTable::GetAsync(Blob(&row_id, sizeof(integer_t)));
}

template <typename T>
int MatrixWorkerTable<T>::GetAsync(const std::vector<integer_t>& row_ids,
  const std::vector<T*>& data_vec,
  size_t size) {
  CHECK(size == num_col_);
  CHECK(row_ids.size() == data_vec.size());
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  for (auto i = 0; i < row_ids.size(); ++i) {
    row_index_[row_ids[i]] = data_vec[i];
  }
  return WorkerTable::GetAsync(Blob(row_ids.data(), sizeof(integer_t)* row_ids.size()));
}

template <typename T>
int MatrixWorkerTable<T>::GetAsync(T* data, size_t size, integer_t* row_ids,
  integer_t row_ids_size) {
  CHECK(size == num_col_ * row_ids_size);
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  for (auto i = 0; i < row_ids_size; ++i) {
    row_index_[row_ids[i]] = &data[i * num_col_];
  }
  Blob ids_blob(row_ids, sizeof(integer_t) * row_ids_size);
  return WorkerTable::GetAsync(ids_blob);
}

template <typename T>
int MatrixWorkerTable<T>::AddAsync(T* data, size_t size, const AddOption* option) {
  CHECK(size == num_col_ * num_row_);
  integer_t whole_table = -1;
  return AddAsync(whole_table, data, size, option);
}

template <typename T>
int MatrixWorkerTable<T>::AddAsync(integer_t row_id, T* data, size_t size,
                                              const AddOption* option) {
  if (row_id >= 0) CHECK(size == num_col_);
  Blob ids_blob(&row_id, sizeof(integer_t));
  Blob data_blob(data, size * sizeof(T));
  return WorkerTable::AddAsync(ids_blob, data_blob, option);
}

template <typename T>
int MatrixWorkerTable<T>::AddAsync(const std::vector<integer_t>& row_ids,
                               const std::vector<T*>& data_vec,
                               size_t size,
                               const AddOption* option) {
  CHECK(size == num_col_);
  Blob ids_blob(&row_ids[0], sizeof(integer_t)* row_ids.size());
  Blob data_blob(row_ids.size() * row_size_);
  // copy each row
  for (auto i = 0; i < row_ids.size(); ++i) {
    memcpy(data_blob.data() + i * row_size_, data_vec[i], row_size_);
  }
  return WorkerTable::AddAsync(ids_blob, data_blob, option);
}

template <typename T>
int MatrixWorkerTable<T>::AddAsync(T* data, size_t size, integer_t* row_ids,
  integer_t row_ids_size,
  const AddOption* option) {
  CHECK(size == num_col_ * row_ids_size);
  Blob ids_blob(row_ids, sizeof(integer_t) * row_ids_size);
  Blob data_blob(data, row_ids_size * row_size_);
  return WorkerTable::AddAsync(ids_blob, data_blob, option);
}

template <typename T>
int MatrixWorkerTable<T>::Partition(const std::vector<Blob>& kv,
  MsgType, std::unordered_map<int, std::vector<Blob>>* out) {
  CHECK(kv.size() == 1 || kv.size() == 2 || kv.size() == 3);
  CHECK_NOTNULL(out);

  size_t keys_size = kv[0].size<integer_t>();
  integer_t *keys = reinterpret_cast<integer_t*>(kv[0].data());
  if (keys_size == 1 && keys[0] == -1) {
    // using actual number of servers, so that one don't send message
    // to empty servers.
    for (auto i = 0; i < num_server_; ++i) {
      int rank = MV_ServerIdToRank(i);
      (*out)[rank].push_back(kv[0]);
    }
    if (kv.size() >= 2) {  // process add values
      for (integer_t i = 0; i < num_server_; ++i){
        int rank = MV_ServerIdToRank(i);
        Blob blob(kv[1].data() + server_offsets_[i] * row_size_,
          (server_offsets_[i + 1] - server_offsets_[i]) * row_size_);
        (*out)[rank].push_back(blob);
        if (kv.size() == 3) {  // update option blob
          (*out)[rank].push_back(kv[2]);
        }
      }
    } else {
      CHECK(get_reply_count_ == 0);
      get_reply_count_ = static_cast<int>(out->size());
    }
    return static_cast<int>(out->size());
  }

  //count row number in each server
  std::vector<int> dest;
  std::vector<integer_t> count;
  count.resize(num_server_, 0);
  integer_t num_row_each = num_row_ / num_server_;
  for (auto i = 0; i < keys_size; ++i){
    int dst = keys[i] / num_row_each;
    dst = (dst >= num_server_ ? num_server_ - 1 : dst);
    dest.push_back(dst);
    ++count[dst];
  }
  for (auto i = 0; i < num_server_; i++) { // allocate memory for blobs
    int rank = MV_ServerIdToRank(i);
    if (count[i] != 0) {
      std::vector<Blob>& vec = (*out)[rank];
      vec.push_back(Blob(count[i] * sizeof(integer_t)));
      if (kv.size() >= 2) vec.push_back(Blob(count[i] * row_size_));
    }
  }
  count.clear();
  count.resize(num_server_, 0);

  integer_t offset = 0;
  for (auto i = 0; i < keys_size; ++i) {
    int dst = dest[i];
    int rank = MV_ServerIdToRank(dst);
    (*out)[rank][0].As<integer_t>(count[dst]) = keys[i];
    if (kv.size() >= 2){ // copy add values
      memcpy(&((*out)[rank][1].As<T>(count[dst] * num_col_)),
        kv[1].data() + offset, row_size_);
      offset += row_size_;
    }
    ++count[dst];
  }
  for (int i = 0; i < num_server_; ++i){
    int rank = MV_ServerIdToRank(i);
    if (count[i] != 0) {
      if (kv.size() == 3) {// update option blob
        (*out)[rank].push_back(kv[2]);
      }
    }
  }

  if (kv.size() == 1){
    CHECK(get_reply_count_ == 0);
    get_reply_count_ = static_cast<int>(out->size());
  }
  return static_cast<int>(out->size());
}

template <typename T>
void MatrixWorkerTable<T>::ProcessReplyGet(std::vector<Blob>& reply_data) {
  CHECK(reply_data.size() == 2 || reply_data.size() == 3); //3 for get all rows

  size_t keys_size = reply_data[0].size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(reply_data[0].data());
  T* data = reinterpret_cast<T*>(reply_data[1].data());

  //get all rows, only happen in T*
  if (keys_size == 1 && keys[0] == -1) {
    int server_id = reply_data[2].As<int>();
    CHECK_NOTNULL(row_index_[num_row_]);
    CHECK(server_id < server_offsets_.size() - 1);
    memcpy(row_index_[num_row_] + server_offsets_[server_id] * num_col_,
      data, reply_data[1].size());
  } else {
    CHECK(reply_data[1].size() == keys_size * row_size_);
    integer_t offset = 0;
    for (auto i = 0; i < keys_size; ++i) {
      CHECK_NOTNULL(row_index_[keys[i]]);
      memcpy(row_index_[keys[i]], data + offset, row_size_);
      offset += num_col_;
    }
  }
  --get_reply_count_;
}

template <typename T>
MatrixServerTable<T>::MatrixServerTable(const MatrixTableOption<T>& option) :
MatrixServerTable(option.num_row, option.num_col) {}

template <typename T>
MatrixServerTable<T>::MatrixServerTable(integer_t num_row, integer_t num_col) :
  ServerTable(), num_col_(num_col) {

  server_id_ = MV_ServerId();
  CHECK(server_id_ != -1);

  integer_t size = num_row / MV_NumServers();
  if (size > 0) {
    row_offset_ = size * server_id_; // Zoo::Get()->rank();
    if (server_id_ == MV_NumServers() - 1) {
      size = num_row - row_offset_;
    }
  } else {
    size = server_id_ < num_row ? 1 : 0;
    row_offset_ = server_id_;
  }
  my_num_row_ = size;
  storage_.resize(my_num_row_ * num_col);
  updater_ = Updater<T>::GetUpdater(my_num_row_ * num_col);
  Log::Debug("[Init] Server =  %d, type = matrixTable, size =  [ %d x %d ], total =  [ %d x %d ].\n",
    server_id_, size, num_col, num_row, num_col);
}

template <typename T>
MatrixServerTable<T>::MatrixServerTable(integer_t num_row, integer_t num_col, float min_value,float max_value) :
MatrixServerTable<T>::MatrixServerTable(num_row, num_col) {
  if (typeid(T) == typeid(float)){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_value,max_value);

    for (int i = 0; i<storage_.size(); i++)
    {
      storage_[i] = static_cast<T>(dis(gen));
    }
  }
}

template <typename T>
void MatrixServerTable<T>::ProcessAdd(const std::vector<Blob>& data) {
  CHECK(data.size() == 2 || data.size() == 3);
  size_t keys_size = data[0].size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(data[0].data());
  T *values = reinterpret_cast<T*>(data[1].data());
  AddOption* option = nullptr;
  if (data.size() == 3) {
    option = new AddOption(data[2].data(), data[2].size());
  }
  // add all values
  if (keys_size == 1 && keys[0] == -1){
    size_t ssize = storage_.size();
    CHECK(ssize == data[1].size<T>());
    updater_->Update(ssize, storage_.data(), values, option);
    Log::Debug("[ProcessAdd] Server = %d, adding all rows offset = %d, #rows = %d\n",
      server_id_, row_offset_, ssize / num_col_);
  } else {
    CHECK(data[1].size() == keys_size * sizeof(T) * num_col_);

    integer_t offset_v = 0;
    CHECK(storage_.size() >= keys_size * num_col_);
    for (auto i = 0; i < keys_size; ++i) {
      integer_t offset_s = (keys[i] - row_offset_) * num_col_;
      updater_->Update(num_col_, storage_.data(), values + offset_v, option, offset_s);
      offset_v += num_col_;
    }
    Log::Debug("[ProcessAdd] Server = %d, adding #rows = %d\n",
      server_id_, keys_size);
  }
  delete option;
}

template <typename T>
void MatrixServerTable<T>::ProcessGet(const std::vector<Blob>& data,
  std::vector<Blob>* result) {
  CHECK(data.size() == 1);
  CHECK_NOTNULL(result);

  result->push_back(data[0]); // also push the key

  size_t keys_size = data[0].size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(data[0].data());

  //get all rows
  if (keys_size == 1 && keys[0] == -1){
    Blob value(sizeof(T) * storage_.size());
    T* pvalues = reinterpret_cast<T*>(value.data());
    updater_->Access(storage_.size(), storage_.data(), pvalues);
    result->push_back(value);
    result->push_back(Blob(&server_id_, sizeof(int)));
    Log::Debug("[ProcessGet] Server = %d, getting all rows offset = %d, #rows = %d\n",
      server_id_, row_offset_, storage_.size() / num_col_);
    return;
  }

  integer_t row_size = sizeof(T)* num_col_;
  result->push_back(Blob(keys_size * row_size));
  T* vals = reinterpret_cast<T*>((*result)[1].data());
  integer_t offset_v = 0;
  for (auto i = 0; i < keys_size; ++i) {
    integer_t offset_s = (keys[i] - row_offset_) * num_col_;
    updater_->Access(num_col_, storage_.data(), vals + offset_v, offset_s);
    offset_v += num_col_;
  }
  Log::Debug("[ProcessGet] Server = %d, getting row #rows = %d\n",
    server_id_, keys_size);
  return;
}

template <typename T>
void MatrixServerTable<T>::Store(Stream* s) {
  s->Write(storage_.data(), storage_.size() * sizeof(T));
}

template <typename T>
void MatrixServerTable<T>::Load(Stream* s) {
  s->Read(storage_.data(), storage_.size() * sizeof(T));
}

MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(MatrixWorkerTable);
MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(MatrixServerTable);

}
