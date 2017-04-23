#include <boost/test/unit_test.hpp>
#include <multiverso/table/array_table.h>

#include "multiverso_env.h"

namespace multiverso {
namespace test {

struct SyncArrayTableEnv : public SyncMultiversoEnv {
  ArrayWorker<int>* table;

  SyncArrayTableEnv() : SyncMultiversoEnv() {
    ArrayTableOption<int> option(10);
    table = MV_CreateTable(option);
  }

  ~SyncArrayTableEnv() {
    delete table;
    table = nullptr;
  }
};

BOOST_FIXTURE_TEST_SUITE(test_sync, SyncArrayTableEnv)

BOOST_AUTO_TEST_CASE(sync) {
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


BOOST_AUTO_TEST_SUITE_END()

}  // namespace test
}  // namespace multiverso
