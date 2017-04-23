#include <iostream>
#include <thread>
#include <random>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <memory>
#include <cassert>

#include <mpi.h>

#include <multiverso/multiverso.h>
#include <multiverso/net.h>
#include <multiverso/util/log.h>
#include <multiverso/util/net_util.h>
#include <multiverso/util/configure.h>
#include <multiverso/util/timer.h>
#include <multiverso/dashboard.h>

#include <multiverso/table/array_table.h>
#include <multiverso/table/kv_table.h>
#include <multiverso/table/matrix_table.h>             
#include <multiverso/table/matrix.h>
#include <multiverso/table/sparse_matrix_table.h>
#include <multiverso/updater/updater.h>
#include <multiverso/table_factory.h>

namespace multiverso {
namespace test {

template<typename WT, typename ST>
void TestmatrixPerformance(int argc, char* argv[],
  std::function<std::shared_ptr<WT>(int num_row, int num_col)>CreateWorkerTable,
  std::function<std::shared_ptr<ST>(int num_row, int num_col)>CreateServerTable,
  std::function<void(const std::shared_ptr<WT>& worker_table, const std::vector<int>& row_ids, const std::vector<float*>& data_vec, size_t size, const AddOption* option, int worker_id)> Add,
  std::function<void(const std::shared_ptr<WT>& worker_table, float* data, size_t size, int worker_id)> Get) {

  Log::ResetLogLevel(LogLevel::Info);
  Log::Info("Test Matrix\n");
  Timer timmer;

  //multiverso::SetCMDFlag("sync", true);
  MV_Init(&argc, argv);
  int num_row = 1000000, num_col = 50;
  if (argc == 3){
    num_row = atoi(argv[2]);
  }

  int size = num_row * num_col;
  int worker_id = MV_Rank();
  int worker_num = MV_Size();

  // test data
  float* data = new float[size];
  float* delta = new float[size];
  for (auto row = 0; row < num_row; ++row) {
    for (auto col = 0; col < num_col; ++col) {
      delta[row * num_col + col] = static_cast<float>(row * num_col + col);
    }
  }

  AddOption option;
  option.set_worker_id(worker_id);

  for (auto percent = 0; percent < 10; ++percent)
    for (auto turn = 0; turn < 10; ++turn)
    {
      //std::shuffle(unique_index.begin(), unique_index.end(), eng);
      if (worker_id == 0) {
        std::cout << "\nTesting: Get All Rows => Add "
          << percent + 1 << "0% Rows to Server => Get All Rows" << std::endl;
      }


      auto worker_table = CreateWorkerTable(num_row, num_col);
      auto server_table = CreateServerTable(num_row, num_col);
      MV_Barrier();

      timmer.Start();
      Get(worker_table, data, size, worker_id);
      std::cout << " " << 1.0 * timmer.elapse() / 1000 << "s:\t" << "get all rows first time, worker id: " << worker_id << std::endl;
      MV_Barrier();

      std::vector<int> row_ids;
      std::vector<float*> data_vec;
      for (auto i = 0; i < num_row; ++i) {
        if (i % 10 <= percent && i % worker_num == worker_id) {
          row_ids.push_back(i);
          data_vec.push_back(delta + i * num_col);
        }
      }

      if (worker_id == 0) {
        std::cout << "adding " << percent + 1 << " /10 rows to matrix server" << std::endl;
      }

      if (row_ids.size() > 0) {
        Add(worker_table, row_ids, data_vec, num_col, &option, worker_id);
      }
      Get(worker_table, data, size, -1);
      MV_Barrier();

      timmer.Start();
      Get(worker_table, data, size, worker_id);
      std::cout << " " << 1.0 * timmer.elapse() / 1000 << "s:\t" << "get all rows after adding to rows, worker id: " << worker_id << std::endl;

      for (auto i = 0; i < num_row; ++i) {
        auto row_start = data + i * num_col;
        for (auto col = 0; col < num_col; ++col) {
          float expected = (float)i * num_col + col;
          float actual = *(row_start + col);
          if (i % 10 <= percent) {
            CHECK(expected == actual); 
          }
          else {
            CHECK(0 == *(row_start + col)); 
          }
        }
      }
    }

  MV_Barrier();
  Log::ResetLogLevel(LogLevel::Info);
  Dashboard::Display();
  Log::ResetLogLevel(LogLevel::Error);
  MV_ShutDown();
}

void TestSparsePerf(int argc, char* argv[]) {
  TestmatrixPerformance<MatrixWorker<float>, MatrixServer<float>>(argc,
    argv,
    [](int num_row, int num_col) {
    return std::shared_ptr<MatrixWorker<float>>(
      new MatrixWorker<float>(num_row, num_col, true));
  },
    [](int num_row, int num_col) {
    return std::shared_ptr<MatrixServer<float>>(
      new MatrixServer<float>(num_row, num_col, true, false));
  },
    [](const std::shared_ptr<MatrixWorker<float>>& worker_table, const std::vector<int>& row_ids, const std::vector<float*>& data_vec, size_t size, const AddOption* option, const int) {
    worker_table->Add(row_ids, data_vec, size, option);
  },

    [](const std::shared_ptr<MatrixWorker<float>>& worker_table, float* data, size_t size, int worker_id) {
    GetOption get_option;
    get_option.set_worker_id(worker_id);
    worker_table->Get(data, size, &get_option);
  });
}


void TestDensePerf(int argc, char* argv[]) {
  TestmatrixPerformance<MatrixWorkerTable<float>, MatrixServerTable<float>>(argc,
    argv,
    [](int num_row, int num_col) {
    return std::shared_ptr<MatrixWorkerTable<float>>(
      new MatrixWorkerTable<float>(num_row, num_col));
  },
    [](int num_row, int num_col) {
    return std::shared_ptr<MatrixServerTable<float>>(
      new MatrixServerTable<float>(num_row, num_col));
  },
    [](const std::shared_ptr<MatrixWorkerTable<float>>& worker_table, const std::vector<int>& row_ids, const std::vector<float*>& data_vec, size_t size, const AddOption* option, const int) {
    worker_table->Add(row_ids, data_vec, size, option);
  },

    [](const std::shared_ptr<MatrixWorkerTable<float>>& worker_table, float* data, size_t size, int) {
    worker_table->Get(data, size);
  });
}

}  // namespace test
}  // namespace multiverso