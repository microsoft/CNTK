#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include <multiverso/util/configure.h>
#include <multiverso/table/matrix.h>

namespace multiverso {
namespace test {

void TestMatrix(int argc, char* argv[]) {
  multiverso::SetCMDFlag("sync", true);
  MV_Init(&argc, argv);

  int num_row = 11, num_col = 3592;
  int num_tables = 2;
  std::vector<int> num_table_size;
  std::vector<MatrixOption<int>* > table_options;

  std::vector<MatrixWorker<int>* > worker_tables;

  for (auto i = 0; i < num_tables - 1; i++)
  {
    table_options.push_back(new MatrixOption<int>());
    table_options[i]->num_col = num_col;
    table_options[i]->num_row = num_row + i;
    table_options[i]->is_sparse = true;
    num_table_size.push_back(num_col * (num_row + i));
    worker_tables.push_back(MV_CreateTable(*table_options[i]));
  }

  table_options.push_back(new MatrixOption<int>());
  table_options[num_tables - 1]->num_col = num_col;
  table_options[num_tables - 1]->num_row = 1;
  num_table_size.push_back(num_col * (1));
  worker_tables.push_back(MV_CreateTable(*table_options[num_tables - 1]));

  int count = 0;

  while (count < 10000) {
    count++;
    std::vector<int> v = { 0, 1, 3, 7 };

    // test data
    std::vector<std::vector<int>> delta(num_tables);
    std::vector<std::vector<int>> data(num_tables);
    for (auto j = 0; j < num_tables; j++) {
      delta[j].resize(num_table_size[j]);
      data[j].resize(num_table_size[j], 0);
      for (auto i = 0; i < num_table_size[j]; ++i)
        delta[j][i] = (int)i + 1;
    }

    for (auto j = 0; j < num_tables; j++) {
      worker_tables[j]->Add(delta[j].data(), num_table_size[j]);
      worker_tables[j]->Get(data[j].data(), num_table_size[j]);
    }

    if (count % 1000 == 0) {
      printf("Dense Add/Get, #test: %d.\n", count);
      fflush(stdout);
    }

    std::vector<int*> data_rows = { &data[0][0], 
                                    &data[0][num_col], 
                                    &data[0][3 * num_col], 
                                    &data[0][7 * num_col] 
                                  };
    std::vector<int*> delta_rows = { &delta[0][0], 
                                     &delta[0][num_col], 
                                     &delta[0][3 * num_col], 
                                     &delta[0][7 * num_col] 
                                   };

    for (auto j = 0; j < num_tables - 1; j++) {
      worker_tables[j]->Add(v, delta_rows, num_col);
      worker_tables[j]->Get(v, data_rows, num_col);
    }

    if (count % 1000 == 0) {
      printf("Sparse Add/Get, #test: %d.\n", count);
      fflush(stdout);
    }

    for (auto i = 0; i < num_row; ++i) {
      for (auto j = 0; j < num_col; ++j) {
        int expected = (int)(i * num_col + j + 1) * count * MV_NumWorkers();
        if (i == 0 || i == 1 || i == 3 || i == 7) {
          expected += (int)(i * num_col + j + 1) * count * MV_NumWorkers();
        }
        int actual = data[0][i* num_col + j];
        CHECK(expected == actual); 
      }
    }
  }

  for (auto table : worker_tables) delete table;
  worker_tables.clear();

  MV_ShutDown();
}

}  // namespace test
}  // namespace multiverso
