#include <boost/test/unit_test.hpp>
#include <multiverso/table/kv_table.h>

#include "multiverso_env.h"

namespace multiverso {
namespace test {

struct KVTableEnv : public MultiversoEnv {
  KVWorkerTable<int, int>* table;

  KVTableEnv() : MultiversoEnv() {
    KVTableOption<int, int> option;
    table = MV_CreateTable(option);
  }

  ~KVTableEnv() {
    delete table;
    table = nullptr;
  }
};

BOOST_FIXTURE_TEST_SUITE(test_kv, KVTableEnv) 

BOOST_AUTO_TEST_CASE(access) {
  auto& map = table->raw();
  table->Get(0);
  BOOST_CHECK_EQUAL(map[0], 0);

  table->Add(0, 3);

  table->Get(0);
  BOOST_CHECK_EQUAL(map[0], 3);

  table->Add(0, -4);

  table->Get(0);
  BOOST_CHECK_EQUAL(map[0], -1);
}


BOOST_AUTO_TEST_SUITE_END()

}  // namespace test
}  // namespace multiverso
