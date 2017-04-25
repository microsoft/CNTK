#ifndef MULTIVERSO_MATRIX_H_
#define MULTIVERSO_MATRIX_H_

#include "multiverso/multiverso.h"
#include "multiverso/table_interface.h"

#include <vector>

namespace multiverso {

  template <typename T>
  struct MatrixOption;

  template <typename T>
  class MatrixWorker : public WorkerTable {
  public:
    explicit MatrixWorker(const MatrixOption<T>& option);

    MatrixWorker(integer_t num_row, integer_t num_col, bool is_sparse = false);

    ~MatrixWorker();

    // get whole table, data is user-allocated memory
    void Get(T* data, size_t size,
      const GetOption* option = nullptr);

    // data is user-allocated memory
    void Get(integer_t row_id, T* data, size_t size,
      const GetOption* option = nullptr);

    void Get(const std::vector<integer_t>& row_ids,
      const std::vector<T*>& data_vec, size_t size,
      const GetOption* option = nullptr);

    // Get specific rows.
    void Get(T* data, size_t size, integer_t* row_ids,
      integer_t row_ids_size,
      const GetOption* option = nullptr);

    // Add whole table
    void Add(T* data, size_t size,
      const AddOption* option = nullptr);

    void Add(integer_t row_id, T* data, size_t size,
      const AddOption* option = nullptr);

    void Add(const std::vector<integer_t>& row_ids,
      const std::vector<T*>& data_vec, size_t size,
      const AddOption* option = nullptr);

    // Add specific rows.
    void Add(T* data, size_t size, integer_t* row_ids, integer_t row_ids_size,
      const AddOption* option = nullptr);

    int Partition(const std::vector<Blob>& kv,
      MsgType partition_type,
      std::unordered_map<int, std::vector<Blob>>* out) override;

    void ProcessReplyGet(std::vector<Blob>& reply_data) override;

  protected:
    T** row_index_;
    int get_reply_count_;                    // number of unprocessed get reply
    integer_t num_row_;
    integer_t num_col_;
    integer_t row_size_;                           // equals to sizeof(T) * num_col_
    int num_server_;                         // the number of running servers
    std::vector<integer_t> server_offsets_;        // offset for the row in each servers
    bool is_sparse_;
  };

  template <typename T>
  class Updater;

  template <typename T>
  class MatrixServer : public ServerTable {
  public:
    explicit MatrixServer(const MatrixOption<T>& option);

    MatrixServer(integer_t num_row, integer_t num_col, bool is_sparse = false,
                                            bool is_pipeline = false);

    void ProcessAdd(const std::vector<Blob>& data) override;

    void ProcessGet(const std::vector<Blob>& data,
      std::vector<Blob>* result) override;

    void Store(Stream* s) override;
    void Load(Stream* s) override;

  protected:
    void UpdateAddState(int worker_id, Blob keys);
    void UpdateGetState(int worker_id, integer_t* keys, size_t key_size,
      std::vector<integer_t>* out_rows);

    inline integer_t GetLogicalRow(integer_t physical_row) {
      return row_offset_ + physical_row;
    }

    inline integer_t GetPhysicalRow(integer_t logical_row) {
      return logical_row - row_offset_;
    }
    int server_id_;
    integer_t my_num_row_;
    integer_t num_col_;
    integer_t row_offset_;
    Updater<T>* updater_;
    std::vector<T> storage_;

    // following attibutes are used by sparse update
    bool is_sparse_;
    bool** up_to_date_;
    int workers_nums_;
  };

  template <typename T>
  struct MatrixOption {
    integer_t num_row;
    integer_t num_col;
    bool is_sparse;
    bool is_pipeline;
    DEFINE_TABLE_TYPE(T, MatrixWorker, MatrixServer);
  };

}

#endif // MULTIVERSO_MATRIX_H_
