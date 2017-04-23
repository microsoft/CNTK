#include "multiverso/table/matrix.h"

#include <vector>
#include <algorithm>

#include "multiverso/io/io.h"
#include "multiverso/multiverso.h"
#include "multiverso/util/log.h"
#include "multiverso/util/quantization_util.h"
#include "multiverso/updater/updater.h"

namespace multiverso {

template <typename T>
MatrixWorker<T>::MatrixWorker(const MatrixOption<T>& option) :
MatrixWorker(option.num_row, option.num_col, option.is_sparse) {}

template <typename T>
MatrixWorker<T>::MatrixWorker(integer_t num_row, integer_t num_col, bool is_sparse) :
WorkerTable(), num_row_(num_row), num_col_(num_col), is_sparse_(is_sparse) {
  row_size_ = num_col * sizeof(T);
  get_reply_count_ = 0;

  num_server_ = MV_NumServers();
  // compute row offsets in all servers
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
  }
  else {
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
  if (is_sparse_)
    Log::Debug("[Init] worker = %d, with sparse updater.\n", MV_Rank());
  row_index_ = new T*[num_row_ + 1];
}

template <typename T>
MatrixWorker<T>::~MatrixWorker() {
  server_offsets_.clear();
  delete[]row_index_;
}

template <typename T>
void MatrixWorker<T>::Get(T* data, size_t size,
  const GetOption* option) {
  CHECK(size == num_col_ * num_row_);
  integer_t whole_table = -1;
  Get(whole_table, data, size, option);
}

template <typename T>
void MatrixWorker<T>::Get(integer_t row_id, T* data, size_t size,
  const GetOption* option) {
  if (row_id >= 0) CHECK(size == num_col_);
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  // row_index are used to hold the address that from user code
  //    so that multiverso can write back to user code.
  if (row_id == -1) {
    row_index_[num_row_] = data;
  }
  else {
    row_index_[row_id] = data;  // data_ = data;
  }

  bool is_option_mine = false;
  if (is_sparse_ && option == nullptr) {
    // the get option is required by the sparse update logic.
    is_option_mine = true;
    option = new GetOption();
  }

  WorkerTable::Get(Blob(&row_id, sizeof(integer_t)), option);
  Log::Debug("[Get] worker = %d, #row = %d\n", MV_Rank(), row_id);

  if (is_option_mine) delete option;
}

template <typename T>
void MatrixWorker<T>::Get(const std::vector<integer_t>& row_ids,
  const std::vector<T*>& data_vec,
  size_t size, const GetOption* option) {
  CHECK(size == num_col_);
  CHECK(row_ids.size() == data_vec.size());
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  for (auto i = 0; i < row_ids.size(); ++i) {
    row_index_[row_ids[i]] = data_vec[i];
  }

  bool is_option_mine = false;
  if (is_sparse_ && option == nullptr) {
    // the get option is required by the sparse update logic.
    is_option_mine = true;
    option = new GetOption();
  }

  WorkerTable::Get(Blob(row_ids.data(), sizeof(integer_t)* row_ids.size()), option);
  Log::Debug("[Get] worker = %d, #rows_set = %d / %d\n",
    MV_Rank(), row_ids.size(), num_row_);

  if (is_option_mine) delete option;
}

template <typename T>
void MatrixWorker<T>::Get(T* data, size_t size, integer_t* row_ids,
  integer_t row_ids_size,
  const GetOption* option) {
  CHECK(size == num_col_ * row_ids_size);
  for (auto i = 0; i < num_row_ + 1; ++i) row_index_[i] = nullptr;
  for (auto i = 0; i < row_ids_size; ++i) {
    row_index_[row_ids[i]] = &data[i * num_col_];
  }
  Blob ids_blob(row_ids, sizeof(integer_t) * row_ids_size);

  bool is_option_mine = false;
  if (is_sparse_ && option == nullptr) {
    // the get option is required by the sparse update logic.
    is_option_mine = true;
    option = new GetOption();
  }

  WorkerTable::Get(ids_blob, option);
  Log::Debug("[Get] worker = %d, #rows_set = %d / %d\n",
    MV_Rank(), row_ids_size, num_row_);
  if (is_option_mine) delete option;
}

template <typename T>
void MatrixWorker<T>::Add(T* data, size_t size, const AddOption* option) {
  CHECK(size == num_col_ * num_row_);
  if (is_sparse_ && true) {
    // REVIEW[qiwye] does this pre-optimization bring too much overhead?
    std::vector<integer_t> row_ids;
    for (auto i = 0; i < num_row_; i++) {
      auto zero_count = std::count(data + (i * num_col_), data + ((i + 1) * num_col_), (T)0);
      if (zero_count != num_col_) {
        row_ids.push_back(i);
      }
    }
    Blob ids_blob(row_ids.data(), sizeof(integer_t)* row_ids.size());
    Blob data_blob(row_ids.size() * row_size_);

    for (auto i = 0; i < row_ids.size(); ++i) {
      memcpy(data_blob.data() + i * row_size_,
        data + row_ids[i] * num_col_, row_size_);
    }

    bool is_option_mine = false;
    if (option == nullptr) {
      is_option_mine = true;
      option = new AddOption();
    }

    WorkerTable::Add(ids_blob, data_blob, option);
    Log::Debug("[Add] Sparse: worker = %d, #rows_set = %d / %d\n",
      MV_Rank(), row_ids.size(), num_row_);
    if (is_option_mine) delete option;

  }
  else {
    integer_t whole_table = -1;
    Add(whole_table, data, size, option);
  }
}

template <typename T>
void MatrixWorker<T>::Add(integer_t row_id, T* data, size_t size,
  const AddOption* option) {
  if (row_id >= 0) CHECK(size == num_col_);
  Blob ids_blob(&row_id, sizeof(integer_t));
  Blob data_blob(data, size * sizeof(T));

  bool is_option_mine = false;
  if (is_sparse_ && option == nullptr) {
    // add option is required by the sparse update logic.
    is_option_mine = true;
    option = new AddOption();
  }

  WorkerTable::Add(ids_blob, data_blob, option);
  Log::Debug("[Add] worker = %d, #row = %d\n", MV_Rank(), row_id);
  if (is_option_mine) delete option;
}

template <typename T>
void MatrixWorker<T>::Add(const std::vector<integer_t>& row_ids,
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

  bool is_option_mine = false;
  if (is_sparse_ && option == nullptr) {
    // add option is required by the sparse update logic.
    is_option_mine = true;
    option = new AddOption();
  }

  WorkerTable::Add(ids_blob, data_blob, option);
  Log::Debug("[Add] worker = %d, #rows_set = %d / %d\n", MV_Rank(), row_ids.size(), num_row_);
  if (is_option_mine) delete option;
}

template <typename T>
void MatrixWorker<T>::Add(T* data, size_t size, integer_t* row_ids,
  integer_t row_ids_size,
  const AddOption* option) {
  CHECK(size == num_col_ * row_ids_size);
  Blob ids_blob(row_ids, sizeof(integer_t) * row_ids_size);
  Blob data_blob(data, row_ids_size * row_size_);

  bool is_option_mine = false;
  if (is_sparse_ && option == nullptr) {
    // add option is required by the sparse update logic.
    is_option_mine = true;
    option = new AddOption();
  }

  WorkerTable::Add(ids_blob, data_blob, option);
  Log::Debug("[Add] worker = %d, #rows_set = %d / %d\n", MV_Rank(), row_ids_size, num_row_);
  if (is_option_mine) delete option;
}

template <typename T>
int MatrixWorker<T>::Partition(const std::vector<Blob>& kv,
  MsgType partition_type,
  std::unordered_map<int, std::vector<Blob>>* out) {
  CHECK(kv.size() == 1 || kv.size() == 2 || kv.size() == 3);
  CHECK_NOTNULL(out);

  size_t keys_size = kv[0].size<integer_t>();
  integer_t *keys = reinterpret_cast<integer_t*>(kv[0].data());


  if (keys_size == 1 && keys[0] == -1) {
    for (auto i = 0; i < num_server_; ++i) {
      int rank = MV_ServerIdToRank(i);
      (*out)[rank].push_back(kv[0]);
    }

    if (partition_type == MsgType::Request_Add) {
      for (integer_t i = 0; i < num_server_; ++i) {
        int rank = MV_ServerIdToRank(i);
        Blob blob(kv[1].data() + server_offsets_[i] * row_size_,
          (server_offsets_[i + 1] - server_offsets_[i]) * row_size_);
        (*out)[rank].push_back(blob);
        if (kv.size() == 3) {  // adding update options
          (*out)[rank].push_back(kv[2]);
        }
      }
    }
    else if (partition_type == MsgType::Request_Get) {
      for (auto i = 0; i < num_server_; ++i) {
        int rank = MV_ServerIdToRank(i);
        if (kv.size() == 2) {  // adding update options
          (*out)[rank].push_back(kv[1]);
        }
      }
      CHECK(get_reply_count_ == 0);
      get_reply_count_ = static_cast<int>(out->size());
    }
    return static_cast<int>(out->size());
  }

  // count row number in each server
  std::vector<int> dest;
  std::vector<integer_t> count;
  count.resize(num_server_, 0);
  integer_t num_row_each = num_row_ / num_server_;
  for (auto i = 0; i < keys_size; ++i) {
    int dst = keys[i] / num_row_each;
    dst = (dst >= num_server_ ? num_server_ - 1 : dst);
    dest.push_back(dst);
    ++count[dst];
  }
  for (auto i = 0; i < num_server_; i++) {  // allocate memory for blobs
    int rank = MV_ServerIdToRank(i);
    if (count[i] != 0) {
      std::vector<Blob>& vec = (*out)[rank];
      vec.push_back(Blob(count[i] * sizeof(integer_t)));  // row indices
      if (partition_type == MsgType::Request_Add)
        vec.push_back(Blob(count[i] * row_size_));  // row values
    }
  }
  count.clear();
  count.resize(num_server_, 0);

  integer_t offset = 0;
  for (auto i = 0; i < keys_size; ++i) {
    int dst = dest[i];
    int rank = MV_ServerIdToRank(dst);
    (*out)[rank][0].As<integer_t>(count[dst]) = keys[i];
    if (partition_type == MsgType::Request_Add) { // copy add values
      memcpy(&((*out)[rank][1].As<T>(count[dst] * num_col_)),
        kv[1].data() + offset, row_size_);
      offset += row_size_;
    }
    ++count[dst];
  }

  for (int i = 0; i < num_server_; ++i) {
    int rank = MV_ServerIdToRank(i);
    if (count[i] != 0) {
      if (partition_type == MsgType::Request_Add && kv.size() == 3) {
        (*out)[rank].push_back(kv[2]);
      }
      else if (partition_type == MsgType::Request_Get && kv.size() == 2) {
        (*out)[rank].push_back(kv[1]);
      }
    }
  }

  if (partition_type == MsgType::Request_Get) {
    CHECK(get_reply_count_ == 0);
    get_reply_count_ = static_cast<int>(out->size());
  }

  // TODO(qiwye): adding logic for filtering
  return static_cast<int>(out->size());
}

template <typename T>
void MatrixWorker<T>::ProcessReplyGet(std::vector<Blob>& reply_data) {
  size_t keys_size = reply_data[0].size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(reply_data[0].data());
  T* data = reinterpret_cast<T*>(reply_data[1].data());

  if (is_sparse_) {
    if (row_index_[num_row_] != nullptr) {
      for (auto i = 0; i < keys_size; ++i) {
        row_index_[keys[i]] = row_index_[num_row_] + keys[i] * num_col_;
      }
    }
  }
  // get all rows, only happen in T*
  if (keys_size == 1 && keys[0] == -1) {
    int server_id = reply_data[2].As<int>();
    CHECK_NOTNULL(row_index_[num_row_]);
    CHECK(server_id < server_offsets_.size() - 1);
    memcpy(row_index_[num_row_] + server_offsets_[server_id] * num_col_,
      data, reply_data[1].size());
  }
  else {
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
MatrixServer<T>::MatrixServer(const MatrixOption<T>& option) :
MatrixServer(option.num_row, option.num_col, option.is_sparse, option.is_pipeline) {}

template <typename T>
MatrixServer<T>::MatrixServer(integer_t num_row, integer_t num_col,
  bool is_sparse, bool is_use_pipeline) :
  ServerTable(), num_col_(num_col), is_sparse_(is_sparse) {
  server_id_ = MV_ServerId();
  CHECK(server_id_ != -1);

  integer_t size = num_row / MV_NumServers();
  if (size > 0) {
    row_offset_ = size * server_id_;
    if (server_id_ == MV_NumServers() - 1) {
      size = num_row - row_offset_;
    }
  }
  else {
    size = server_id_ < num_row ? 1 : 0;
    row_offset_ = server_id_;
  }
  my_num_row_ = size;
  storage_.resize(my_num_row_ * num_col);
  updater_ = Updater<T>::GetUpdater(my_num_row_ * num_col);
  Log::Info("[Init] Server =  %d, type = matrixTable, size =  [ %d x %d ], total =  [ %d x %d ].\n",
    server_id_, size, num_col, num_row, num_col);

  if (is_sparse_) {
    workers_nums_ = multiverso::MV_NumWorkers();
    if (is_use_pipeline) {
      workers_nums_ *= 2;
    }
    up_to_date_ = new bool*[workers_nums_];
    for (auto i = 0; i < workers_nums_; ++i) {
      up_to_date_[i] = new bool[my_num_row_];
      memset(up_to_date_[i], 0, sizeof(bool) * my_num_row_);
    }
    Log::Info("[Init] Server = %d, with sparse updater.\n", server_id_);
  }
}

template <typename T>
void MatrixServer<T>::ProcessAdd(const std::vector<Blob>& data) {
  CHECK(data.size() == 2 || data.size() == 3);
  // TODO(qiwye): Adding filter logic
  size_t keys_size = data[0].size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(data[0].data());
  T *values = reinterpret_cast<T*>(data[1].data());
  AddOption* option = nullptr;
  if (data.size() == 3) {
    option = new AddOption(data[2].data(), data[2].size());
  }
  if (is_sparse_) {
    CHECK_NOTNULL(option);
    UpdateAddState(option->worker_id(), data[0]);
  }
  // add all values
  if (keys_size == 1 && keys[0] == -1) {
    size_t ssize = storage_.size();
    CHECK(ssize == data[1].size<T>());
    updater_->Update(ssize, storage_.data(), values, option);
    Log::Debug("[ProcessAdd] Server = %d, adding all rows offset = %d, #rows = %d\n",
      server_id_, row_offset_, ssize / num_col_);
  }
  else {
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
void MatrixServer<T>::ProcessGet(const std::vector<Blob>& data,
  std::vector<Blob>* result) {
  // TODO(qiwye): Adding filter logic
  CHECK(data.size() == 1 || data.size() == 2);
  CHECK_NOTNULL(result);

  size_t keys_size = data[0].size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(data[0].data());
  std::vector<integer_t>* outdated_rows;

  GetOption* option = nullptr;
  if (data.size() == 2) {
    option = new GetOption(data[1].data(), data[1].size());
  }
  if (is_sparse_) {
    CHECK_NOTNULL(option);
    outdated_rows = new std::vector<integer_t>();
    UpdateGetState(option->worker_id(), keys, keys_size, outdated_rows);

    keys_size = outdated_rows->size();
    keys = reinterpret_cast<integer_t*>(outdated_rows->data());
    result->push_back(Blob(outdated_rows->data(), keys_size * sizeof(T)));
  }
  else {
    result->push_back(data[0]);
  }


  // get all rows
  if (keys_size == 1 && keys[0] == -1) {
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
    integer_t offset_s = GetPhysicalRow(keys[i]) * num_col_;
    updater_->Access(num_col_, storage_.data(), vals + offset_v, offset_s);
    offset_v += num_col_;
  }
  Log::Debug("[ProcessGet] Server = %d, getting row #rows = %d\n",
    server_id_, keys_size);

  delete option;
}

template <typename T>
void MatrixServer<T>::UpdateAddState(int,
  Blob keys_blob) {
  size_t keys_size = keys_blob.size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(keys_blob.data());

  if (keys_size == 1 && keys[0] == -1) {
    for (auto id = 0; id < workers_nums_; ++id) {
      for (auto local_row_id = 0; local_row_id < this->my_num_row_; ++local_row_id) {
        up_to_date_[id][local_row_id] = false;
      }
    }
  }
  else {
    for (auto id = 0; id < workers_nums_; ++id) {
      for (auto i = 0; i < keys_size; ++i) {
        auto local_row_id = GetPhysicalRow(keys[i]);
        up_to_date_[id][local_row_id] = false;
      }
    }
  }
}

template <typename T>
void MatrixServer<T>::UpdateGetState(int worker_id, integer_t* keys,
  size_t key_size, std::vector<integer_t>* out_rows) {
  if (worker_id == -1) {
    for (auto local_row_id = 0; local_row_id < this->my_num_row_; ++local_row_id)  {
      out_rows->push_back(GetLogicalRow(local_row_id));
    }
    return;
  }

  if (key_size == 1 && keys[0] == -1) {
    for (auto local_row_id = 0; local_row_id < this->my_num_row_; ++local_row_id)  {
      if (!up_to_date_[worker_id][local_row_id]) {
        out_rows->push_back(GetLogicalRow(local_row_id));
        up_to_date_[worker_id][local_row_id] = true;
      }
    }
  }
  else {
    for (auto i = 0; i < key_size; ++i)  {
      auto global_row_id = keys[i];
      auto local_row_id = GetPhysicalRow(global_row_id);
      if (!up_to_date_[worker_id][local_row_id]) {
        up_to_date_[worker_id][local_row_id] = true;
        out_rows->push_back(global_row_id);
      }
    }
  }

  // if all rows are up-to-date, then send the first row
  if (out_rows->size() == 0) {
    out_rows->push_back(GetLogicalRow(0));
  }
}

template <typename T>
void MatrixServer<T>::Store(Stream* s) {
  s->Write(storage_.data(), storage_.size() * sizeof(T));
}

template <typename T>
void MatrixServer<T>::Load(Stream* s) {
  s->Read(storage_.data(), storage_.size() * sizeof(T));
}

MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(MatrixWorker);
MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(MatrixServer);

} // namespace multiverso
