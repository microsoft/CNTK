// Copyright 2015 Microsoft
#include "multiverso/table/sparse_matrix_table.h"
#include <vector>
#include <cctype>

#include "multiverso/multiverso.h"
#include "multiverso/util/log.h"
#include "multiverso/util/quantization_util.h"
#include "multiverso/updater/updater.h"

namespace multiverso {

// get whole table, data is user-allocated memory
template <typename T>
void SparseMatrixWorkerTable<T>::Get(T* data, size_t size,
  const GetOption* option) {
  CHECK(size == this->num_col_ * this->num_row_);
  integer_t whole_table = -1;
  Get(whole_table, data, size, option);
}

// data is user-allocated memory
template <typename T>
void SparseMatrixWorkerTable<T>::Get(integer_t row_id, T* data, size_t size,
  const GetOption* option) {
  if (row_id >= 0) CHECK(size == this->num_col_);
  for (auto i = 0; i < this->num_row_ + 1; ++i) this->row_index_[i] = nullptr;
  if (row_id == -1) {
    this->row_index_[this->num_row_] = data;
  } else {
    this->row_index_[row_id] = data; // data_ = data;
  }
  Blob keys(&row_id, sizeof(integer_t) * 1);

  bool is_option_mine = false;
  if (option == nullptr){
    is_option_mine = true;
    option = new GetOption();
  }

  WorkerTable::Get(keys, option);
  Log::Debug("[Get] worker = %d, #row = %d\n", MV_Rank(), row_id);
  if (is_option_mine) delete option;
}

template <typename T>
void SparseMatrixWorkerTable<T>::Get(const std::vector<integer_t>& row_ids,
  const std::vector<T*>& data_vec, size_t size, 
  const GetOption* option) {
  for (auto i = 0; i < this->num_row_ + 1; ++i) this->row_index_[i] = nullptr;
  CHECK(size == this->num_col_);
  CHECK(row_ids.size() == data_vec.size());
  for (integer_t i = 0; i < row_ids.size(); ++i) {
    this->row_index_[row_ids[i]] = data_vec[i];
  }
  Blob keys(row_ids.data(), sizeof(integer_t) * row_ids.size());

  bool is_option_mine = false;
  if (option == nullptr){
    is_option_mine = true;
    option = new GetOption();
  }

  WorkerTable::Get(keys, option);
  Log::Debug("[Get] worker = %d, #rows_set = %d\n", MV_Rank(),
    row_ids.size());
  if (is_option_mine) delete option;
}

template <typename T>
int SparseMatrixWorkerTable<T>::Partition(const std::vector<Blob>& kv,
  MsgType partition_type,
  std::unordered_map<int, std::vector<Blob>>* out) {
  int res;
  CHECK(kv.size() == 1 || kv.size() == 2 || kv.size() == 3);
  CHECK_NOTNULL(out);

  if (kv.size() == 2) {  // processing Get()
    size_t keys_size = kv[0].size<integer_t>();
    integer_t* keys = reinterpret_cast<integer_t*>(kv[0].data());
    if (keys[0] == -1) {
      for (auto i = 0; i < this->num_server_; ++i) {
        int rank = MV_ServerIdToRank(i);
        (*out)[rank].push_back(kv[0]);
      }

      for (auto i = 0; i < this->num_server_; ++i){
        int rank = MV_ServerIdToRank(i);
        if (kv.size() == 2) {// general option blob
          (*out)[rank].push_back(kv[1]);
        }
      }

      CHECK(this->get_reply_count_ == 0);
      this->get_reply_count_ = static_cast<int>(out->size());
      res = static_cast<int>(out->size());
    } else {
      // count row number in each server
      //std::unordered_map<int, integer_t> count;
      std::vector<integer_t> count;
      std::vector<int> dest;
      count.resize(this->num_server_, 0);
      integer_t num_row_each = this->num_row_ / this->num_server_;  //  num_server_;
      for (auto i = 0; i < keys_size; ++i) {
        int dst = keys[i] / num_row_each;
        dst = (dst >= this->num_server_ ? this->num_server_ - 1 : dst);
        dest.push_back(dst);
        ++count[dst];
      }

      for (auto i = 0; i < this->num_server_; i++) { // allocate memory for blobs
        int rank = MV_ServerIdToRank(i);
        if (count[i] != 0) {
          std::vector<Blob>& vec = (*out)[rank];
          vec.push_back(Blob(count[i] * sizeof(integer_t)));
        }
      }
      count.clear();
      count.resize(this->num_server_, 0);

      for (auto i = 0; i < keys_size; ++i) {
        int dst = dest[i];
        int rank = MV_ServerIdToRank(dst);
        (*out)[rank][0].As<integer_t>(count[dst]) = keys[i];
        ++count[dst];
      }


      for (auto i = 0; i < this->num_server_; ++i){
        int rank = MV_ServerIdToRank(i);
        if (count[i] != 0) {
          if (kv.size() == 2) {// add option blob
            (*out)[rank].push_back(kv[1]);
          }
        }
      }

      CHECK(this->get_reply_count_ == 0);
      this->get_reply_count_ = static_cast<int>(out->size());
      res = static_cast<int>(out->size());
    }
  } else {  // processing Add()
    // call base class's Partition
    res = MatrixWorkerTable<T>::Partition(kv, partition_type, out);
  }

   // only have effect when adding elements
  SparseFilter<T, int32_t>  filter(0, true);
  for (auto& pair : *out) {
    std::vector<Blob> compressed_blobs;
    filter.FilterIn(pair.second, &compressed_blobs);
    pair.second.swap(compressed_blobs);
  }

  return res;
}

template <typename T>
void SparseMatrixWorkerTable<T>::ProcessReplyGet(
  std::vector<Blob>& reply_data) {
  // replace row_index when original key == -1
  if (this->row_index_[this->num_row_] != nullptr) {
    size_t keys_size = reply_data[0].size<integer_t>();
    Log::Debug("[SparseMatrixWorkerTable:ProcessReplyGet] worker = %d, #keys_size = %d\n", MV_Rank(),
      keys_size);
    integer_t* keys = reinterpret_cast<integer_t*>(reply_data[0].data());
    for (auto i = 0; i < keys_size; ++i) {
      this->row_index_[keys[i]] = this->row_index_[this->num_row_] + keys[i] * this->num_col_;
    }
  }

  MatrixWorkerTable<T>::ProcessReplyGet(reply_data);
}

template <typename T>
SparseMatrixServerTable<T>::~SparseMatrixServerTable() {
  for (auto i = 0; i < workers_nums_; ++i) {
    delete[]up_to_date_[i];
  }
  delete[]up_to_date_;
}

template <typename T>
SparseMatrixServerTable<T>::SparseMatrixServerTable(integer_t num_row, integer_t num_col,
  bool using_pipeline) : MatrixServerTable<T>(num_row, num_col) {
   workers_nums_ = multiverso::MV_NumWorkers();
  if (using_pipeline) {
    workers_nums_ *= 2;
  }
        
  up_to_date_ = new bool*[workers_nums_];
  for (auto i = 0; i < workers_nums_; ++i) {
    up_to_date_[i] = new bool[this->my_num_row_];
    memset(up_to_date_[i], 0, sizeof(bool) * this->my_num_row_);
  }
  Log::Debug("[SparseMatrixServerTable] workers_nums_= %d .\n", workers_nums_);
}

template <typename T>
void SparseMatrixServerTable<T>::UpdateAddState(int worker_id,
  Blob keys_blob) {
  size_t keys_size = keys_blob.size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(keys_blob.data());
  // add all values
  if (keys_size == 1 && keys[0] == -1) {
    for (auto id = 0; id < workers_nums_; ++id) {
      if (id == worker_id) continue;
      for (auto local_row_id = 0; local_row_id < this->my_num_row_; ++local_row_id) {
        // if other worker doesn't update the row, we can marked it as the updated.
        up_to_date_[id][local_row_id] = false;
      }
    }
  } else {
    for (auto id = 0; id < workers_nums_; ++id) {
      if (id == worker_id) continue;
      for (auto i = 0; i < keys_size; ++i) {
        // if other worker doesn't update the row, we can marked it as the updated.
        auto local_row_id = GetPhysicalRow(keys[i]);
        up_to_date_[id][local_row_id] = false;
      }
    }
  }
}

template <typename T>
void SparseMatrixServerTable<T>::UpdateGetState(int worker_id, integer_t* keys,
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
  } else {
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
void SparseMatrixServerTable<T>::ProcessAdd(
  const std::vector<Blob>& compressed_data) {
  if (compressed_data.size() == 0) return;
  std::vector<Blob> data;
  SparseFilter<T, int32_t> filter(0, true);
  filter.FilterOut(compressed_data, &data);

  // the AddOption option is needed for the sparse update
  CHECK(data.size() == 3);
  AddOption* option = nullptr;
  option = new AddOption(data[2].data(), data[2].size());
  UpdateAddState(option->worker_id(), data[0]);

  MatrixServerTable<T>::ProcessAdd(data);
  delete option;
}

template <typename T>
void SparseMatrixServerTable<T>::ProcessGet(
  const std::vector<Blob>& compressed_data,
  std::vector<Blob>* result) {
  if (compressed_data.size() == 0) return;
  std::vector<Blob> data;
  SparseFilter<T, int32_t> filter(0, true);
  filter.FilterOut(compressed_data, &data);

  // the GetOption is needed for the sparse update
  CHECK(data.size() == 2);
  CHECK_NOTNULL(result);

  size_t keys_size = data[0].size<integer_t>();
  integer_t* keys = reinterpret_cast<integer_t*>(data[0].data());
  GetOption* option = nullptr;
  if (data.size() == 2) {
    option = new GetOption(data[1].data(), data[1].size());
  }
  std::vector<integer_t> outdated_rows;

  UpdateGetState(option->worker_id(), keys, keys_size, &outdated_rows);

  Blob outdated_rows_blob(sizeof(integer_t) * outdated_rows.size());
  for (auto i = 0; i < outdated_rows.size(); ++i) {
    outdated_rows_blob.As<integer_t>(i) = outdated_rows[i];
  }

  std::vector<Blob> blobs{ outdated_rows_blob };
  MatrixServerTable<T>::ProcessGet(blobs, result);
  delete option;
}

MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(SparseMatrixWorkerTable);
MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(SparseMatrixServerTable);

}   //  namespace multiverso
