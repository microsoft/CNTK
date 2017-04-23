#include <multiverso/multiverso.h>
#include <multiverso/net.h>
#include <multiverso/util/configure.h>
#include <multiverso/util/log.h>
#include <multiverso/util/net_util.h>

namespace multiverso {
namespace test {

void TestAllreduce(int argc, char* argv[]) {
  multiverso::SetCMDFlag("ma", true);
  MV_Init(&argc, argv);
  int a = 1;
  MV_Aggregate(&a, 1);

  CHECK(a == MV_Size());

  MV_ShutDown();
}

}  // namespace test
}  // namespace multiverso