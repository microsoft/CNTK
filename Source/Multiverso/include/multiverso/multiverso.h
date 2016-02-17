#ifndef MULTIVERSO_INCLUDE_MULTIVERSO_H_
#define MULTIVERSO_INCLUDE_MULTIVERSO_H_

namespace multiverso {


enum Role {
  kNull = 0,
  kWorker = 1,
  kServer = 2,
  kAll = 3
};

void MultiversoInit(int role = kAll);

void MultiversoBarrier();

void MultiversoShutDown(bool finalize_mpi = true);

}

#endif // MULTIVERSO_INCLUDE_MULTIVERSO_H_