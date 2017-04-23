#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include <multiverso/table/kv_table.h>

namespace multiverso {
namespace test {

void TestKV(int argc, char* argv[]) {
  Log::Info("Test KV map \n");
  // 1. Start the Multiverso engine
  MV_Init(&argc, argv);

  // 2. To create the shared table
  KVTableOption<int, int> option;
  auto dht = MV_CreateTable(option);

  // 3. User program
  // access the local cache
  std::unordered_map<int, int>& kv = dht->raw();

  // Get from the server
  dht->Get(0);
  // Check the result
  Log::Info("Get 0 from kv server: result = %d\n", kv[0]);

  // Add 1 to the server
  dht->Add(0, 1);
  // Check the result
  dht->Get(0);
  Log::Info("Get 0 from kv server after add 1: result = %d\n", kv[0]);

  // 4. Shutdown the Multiverso engine
  MV_ShutDown();
}

}  // namespace test
}  // namespace multiverso