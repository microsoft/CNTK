#include <multiverso/multiverso.h>

#include <multiverso/util/log.h>
#include <multiverso/util/configure.h>
#include <multiverso/net.h>
#include <multiverso/table/array_table.h>

namespace multiverso {
namespace test {

void TestArray(int argc, char* argv[]) {
  Log::Info("Test Array \n");

  multiverso::SetCMDFlag("sync", true);
  MV_Init(&argc, argv);

  size_t array_size = 500;

  auto shared_array = MV_CreateTable(ArrayTableOption<int>(array_size));

  Log::Info("Create tables OK. Rank = %d, worker_id = %d\n",
    MV_Rank(), MV_WorkerId());

  std::vector<int> delta(array_size);
  for (int i = 0; i < array_size; ++i)
    delta[i] = static_cast<int>(i);

  int* data = new int[array_size];

  int iter = 10 * (MV_Rank() + 10);
  for (int i = 0; i < iter; ++i) {
    shared_array->Add(delta.data(), array_size);
    shared_array->Add(delta.data(), array_size);
    shared_array->Add(delta.data(), array_size);
    shared_array->Get(data, array_size);
    shared_array->Get(data, array_size);
    shared_array->Get(data, array_size);
    if (iter < 100) {
      for (int k = 0; k < array_size; ++k) {
        CHECK (data[k] != delta[k] * (i + 1) * MV_NumWorkers()) ;
      }
    }
  }
  delete[] data;

  MV_ShutDown();
}

}  // namespace test
}  // namespace multiverso