#ifndef MULTIVERSO_INCLUDE_MULTIVERSO_H_
#define MULTIVERSO_INCLUDE_MULTIVERSO_H_

namespace multiverso {


enum Role {
  Null = 0,
  Worker = 1,
  Server = 2,
  All = 3
};

void MV_Init(int* argc = nullptr, 
             char* argv[] = nullptr, 
             int role = All);

void MV_Barrier();

void MV_ShutDown(bool finalize_mpi = true);

int  MV_Rank();
int  MV_Size();

int  MV_Worker_Id();
int  MV_Server_Id();
}

#endif // MULTIVERSO_INCLUDE_MULTIVERSO_H_