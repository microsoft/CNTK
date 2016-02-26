#ifndef MULTIVERSO_INCLUDE_MULTIVERSO_H_
#define MULTIVERSO_INCLUDE_MULTIVERSO_H_

namespace multiverso {


enum Role {
  kNull = 0,
  kWorker = 1,
  kServer = 2,
  kAll = 3
};

void MV_Init(int* argc = nullptr, 
             char* argv[] = nullptr, 
             int role = kAll);

void MV_Barrier();

void MV_ShutDown(bool finalize_mpi = true);

int  MV_Rank();

// will deprecate the following function name
void MultiversoInit(int* argc = nullptr, 
                    char* argv[] = nullptr, 
                    int role = kAll);

void MultiversoBarrier();

void MultiversoShutDown(bool finalize_mpi = true);

int MultiversoRank();
}

#endif // MULTIVERSO_INCLUDE_MULTIVERSO_H_