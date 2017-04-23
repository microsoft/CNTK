#ifndef WORDEMBEDDING_COMMUNICATOR_H_
#define WORDEMBEDDING_COMMUNICATOR_H_

#include <multiverso/table/matrix_table.h>
#include <multiverso/table/kv_table.h>
#include <multiverso/updater/updater.h>

#include "memory_manager.h"
#include "block_queue.h"

namespace wordembedding {

  class Communicator {
  public:
    Communicator(Option* option, MemoryManager* memory_mamanger);
    ~Communicator();

    void RequestParameter(DataBlock *data_block);
    void AddDeltaParameter(DataBlock *data_block);

    int64 const GetWordCount();
    void AddWordCount(int64 word_count_num);

    void GetWorkerTableRows(std::vector<int> &row_nums,
      std::vector<real*> &blocks, int embeding_size);

    void PrepareParameterTables(int row_size, int column_size);

  private:
    Option* option_ = nullptr;
    MemoryManager* memory_mamanger_ = nullptr;
    int process_id_ = -1;
    int process_count_ = -1;

    multiverso::MatrixWorkerTable<real>* worker_input_table_ = nullptr;
    multiverso::MatrixWorkerTable<real>* worker_output_table_ = nullptr;
    multiverso::MatrixServerTable<real>* server_input_table_ = nullptr;
    multiverso::MatrixServerTable<real>* server_output_table_ = nullptr;

    multiverso::MatrixWorkerTable<real>* worker_input_gradient_table_ = nullptr;
    multiverso::MatrixWorkerTable<real>* worker_output_gradient_table_ = nullptr;
    multiverso::MatrixServerTable<real>* server_input_gradient_table_ = nullptr;
    multiverso::MatrixServerTable<real>* server_output_gradient_table_ = nullptr;

    multiverso::KVWorkerTable<int, int64>* worker_wordcount_table_ = nullptr;
    multiverso::KVServerTable<int, int64>* server_wordcount_table_ = nullptr;

    void ClearParameterTables();

    void GetRows(multiverso::MatrixWorkerTable<real>* table_, std::vector<int> &row_ids,
      std::vector<real *> &ptrs, int size);

    void RequestParameterByTableId(DataBlock *data_block, int table_id,
      std::vector<int> &nodes, std::vector<real*> &blocks);

    void SetDataBlockEmbedding(DataBlock *data_block, std::vector<real*> &blocks,
      std::vector<int> &nodes, std::function<void(int, real*)> get_function);

    void AddRows(multiverso::MatrixWorkerTable<real>* table, std::vector<int> &row_ids,
      std::vector<real *> &ptrs, int size);

    void AddParameterByTableId(DataBlock *data_block, int table_id,
      std::vector<int> &nodes, std::vector<real*> &blocks,
      std::vector<real*> &recycle_blocks);

    void GetDeltaLoop(DataBlock *data_block, std::vector<real*> &blocks,
      std::vector<int> &nodes, std::vector<real*> &recycle_blocks,
      std::function<real*(int)> get_function);
  };
}
#endif