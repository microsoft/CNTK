#include <cmath>
#include <thread>
#include <string>
#include <new>

#include <multiverso/util/log.h>
#include <multiverso/util/log.h>
#include <multiverso/multiverso.h>

#include "distributed_wordembedding.h"
#include "memory_manager.h"
#include "util.h"

using namespace wordembedding;

int main(int argc, char *argv[]) {
  try {
    DistributedWordembedding dwe;
    dwe.Run(argc, argv);
  }
  catch (std::bad_alloc &memExp) {
    multiverso::Log::Info("Something wrong with new() %s\n", memExp.what());
  }
  catch (...) {
    multiverso::Log::Info("Something wrong with other reason!\n");
  }
  return 0;
}

