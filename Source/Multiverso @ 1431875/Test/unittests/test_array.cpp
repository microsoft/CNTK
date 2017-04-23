#include <vector>
#include <boost/test/unit_test.hpp>
#include <multiverso/table/array_table.h>

#include "multiverso_env.h"

namespace multiverso {
namespace test {

struct ArrayTableEnv : public MultiversoEnv {
  ArrayWorker<int>* table;

  ArrayTableEnv() : MultiversoEnv() {
    ArrayTableOption<int> option(10);
    table = MV_CreateTable(option);
  }

  ~ArrayTableEnv() {
    delete table;
    table = nullptr;
  }
};

BOOST_FIXTURE_TEST_SUITE(array_test, ArrayTableEnv)

BOOST_AUTO_TEST_CASE(array_access) {
  std::vector<int> delta(10);
  std::vector<int> model(10);
  for (int i = 0; i < 10; ++i) delta[i] = i;
  table->Add(delta.data(), delta.size());
  table->Get(model.data(), model.size());

  for (int i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(model[i], delta[i]);
  }

  table->AddAsync(delta.data(), delta.size());
  int handle = table->GetAsync(model.data(), model.size());
  table->Wait(handle);
  
  for (int i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(model[i], 2 * delta[i]);
  }
}

BOOST_AUTO_TEST_CASE(array_partition) {
  std::unordered_map<int, std::vector<Blob>> result;
  std::vector<Blob> kv;
  int key = -1; 
  Blob key_blob(&key, sizeof(key));
  std::vector<int> value(10); 
  Blob value_blob(value.data(), sizeof(int) * value.size());
  kv.push_back(key_blob);
  kv.push_back(value_blob);

  table->Partition(kv, MsgType::Request_Get, &result);

  BOOST_CHECK_EQUAL(result.size(), 1);
  BOOST_CHECK(result.find(0) != result.end());
  BOOST_CHECK_EQUAL(result[0].size(), 2);
  BOOST_CHECK_EQUAL(result[0][0].As<int>(), key);
  int* vec = reinterpret_cast<int*>(result[0][1].data());
  for (int i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(vec[i], value[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace test
}  // namespace multiverso