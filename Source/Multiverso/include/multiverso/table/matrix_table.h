#ifndef MULTIVERSO_MATRIX_TABLE_H_
#define MULTIVERSO_MATRIX_TABLE_H_

#include "multiverso/multiverso.h"
#include "multiverso/table_interface.h"
#include "multiverso/util/log.h"

#include <vector>

namespace multiverso {

template <typename T>
class MatrixWorkerTable : public WorkerTable {
public:
  MatrixWorkerTable(integer_t num_row, integer_t num_col);
  ~MatrixWorkerTable();

  // get whole table, data is user-allocated memory
  void Get(T* data, size_t size);

  // data is user-allocated memory
  void Get(integer_t row_id, T* data, size_t size);

  void Get(const std::vector<integer_t>& row_ids,
           const std::vector<T*>& data_vec, size_t size);

  // Add whole table
  void Add(T* data, size_t size, const AddOption* option = nullptr);

  void Add(integer_t row_id, T* data, size_t size, 
           const AddOption* option = nullptr);

  void Add(const std::vector<integer_t>& row_ids,
           const std::vector<T*>& data_vec, size_t size, 
           const AddOption* option = nullptr);

  int Partition(const std::vector<Blob>& kv,
    std::unordered_map<int, std::vector<Blob>>* out) override;

  void ProcessReplyGet(std::vector<Blob>& reply_data) override;

protected:
  T** row_index_;
  int get_reply_count_;                    // number of unprocessed get reply
  integer_t num_row_;
  integer_t num_col_;
  integer_t row_size_;                           // equals to sizeof(T) * num_col_
  int num_server_;
  std::vector<integer_t> server_offsets_;        // row id offset
};

template <typename T>
class Updater;

template <typename T>
class MatrixServerTable : public ServerTable {
public:
  MatrixServerTable(integer_t num_row, integer_t num_col);

  void ProcessAdd(const std::vector<Blob>& data) override;

  void ProcessGet(const std::vector<Blob>& data,
                  std::vector<Blob>* result) override;

  void Store(Stream* s) override;
  void Load(Stream* s) override;

protected:
  int server_id_;
  integer_t my_num_row_;
  integer_t num_col_;
  integer_t row_offset_;
  Updater<T>* updater_;
  std::vector<T> storage_;
};

//older implementation
template <typename T>
class MatrixTableHelper : public TableHelper {
public:
  MatrixTableHelper(integer_t num_row, integer_t num_col) : num_row_(num_row), num_col_(num_col){}
  ~MatrixTableHelper() {}

protected:
  WorkerTable* CreateWorkerTable() override{
    return new MatrixWorkerTable<T>(num_row_, num_col_);
  }
  ServerTable* CreateServerTable() override{
    return new MatrixServerTable<T>(num_row_, num_col_);
  }
  integer_t num_row_;
  integer_t num_col_;
};

}

#endif // MULTIVERSO_MATRIX_TABLE_H_
