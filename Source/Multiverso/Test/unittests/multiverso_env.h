#ifndef MULTIVERSO_TEST_UNITTEST_MULTIVERSO_EVN_H_
#define MULTIVERSO_TEST_UNITTEST_MULTIVERSO_EVN_H_

#include <multiverso/multiverso.h>

namespace multiverso {
namespace test {

struct MultiversoEnv {
  MultiversoEnv() {
    MV_SetFlag("sync", false);
    MV_Init();
  }

  ~MultiversoEnv() {
    MV_ShutDown(false);
  }
};

struct SyncMultiversoEnv {
  SyncMultiversoEnv() {
    MV_SetFlag("sync", true);
    MV_Init();
  }

  ~SyncMultiversoEnv() {
    MV_ShutDown(false);
  }
};

}  // namespace test
}  // namespace multiverso

#endif  // MULTIVERSO_TEST_UNITTEST_MULTIVERSO_EVN_H_