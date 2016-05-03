#ifndef MULTIVERSO_INCLUDE_MULTIVERSO_H_
#define MULTIVERSO_INCLUDE_MULTIVERSO_H_

#include <string>

namespace multiverso {

void MV_Init(int* argc = nullptr, char* argv[] = nullptr);

void MV_Barrier();

void MV_ShutDown(bool finalize_net = true);

int  MV_Rank();
int  MV_Size();

int  MV_NumWorkers();
int  MV_NumServers();

int  MV_WorkerId();
int  MV_ServerId();

int  MV_WorkerIdToRank(int worker_id);
int  MV_ServerIdToRank(int server_id);

// inplace sum by allreduce
template <typename ElemType>
void MV_Aggregate(ElemType* data, int size);

// --- Net API -------------------------------------------------------------- //
// NOTE(feiga): these API is only used for specific situation.
// Init Multiverso Net with the provided endpoint. Multiverso Net will bind
// the provided endpoint and use this endpoint to listen and recv message
// \param rank the rank of this MV process
// \param endpoint endpoint with format ip:port, e.g., localhost:9999
// \return  0 SUCCESS
// \return -1 FAIL
int  MV_NetBind(int rank, char* endpoint);

// Connect Multiverso Net with other processes in the system. Multiverso Net
// will connect these endpoints and send msgs
// \param ranks array of rank
// \param endpoints endpoints for each rank
// \param size size of the array
// \return  0 SUCCESS
// \return -1 FAIL
int  MV_NetConnect(int* rank, char* endpoint[], int size);

}  // namespace multiverso

#endif  // MULTIVERSO_INCLUDE_MULTIVERSO_H_

