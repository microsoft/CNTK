#include "multiverso/multiverso.h"

#include "multiverso/dashboard.h"
#include "multiverso/net.h"
#include "multiverso/zoo.h"
#include "multiverso/table_factory.h"
#include "multiverso/util/configure.h"

namespace multiverso {

void MV_Init(int* argc, char* argv[]) {
  Zoo::Get()->Start(argc, argv);
}

void MV_ShutDown(bool finalize_net) {
  Zoo::Get()->Stop(finalize_net);
  table_factory::FreeServerTables();
}

void MV_Barrier() { Zoo::Get()->Barrier(); }

int  MV_Rank() { return Zoo::Get()->rank(); }

int  MV_Size() { return Zoo::Get()->size(); }

int  MV_WorkerId() {
  return Zoo::Get()->worker_rank();
}
int  MV_ServerId() {
  return Zoo::Get()->server_rank();
}

int  MV_NumWorkers() {
  return Zoo::Get()->num_workers();
}
int  MV_NumServers() {
  return Zoo::Get()->num_servers();
}

int  MV_WorkerIdToRank(int worker_id) {
  return Zoo::Get()->worker_id_to_rank(worker_id);
}

int  MV_ServerIdToRank(int server_id) {
  return Zoo::Get()->server_id_to_rank(server_id);
}

template <typename T>
void MV_SetFlag(const std::string& name, const T& value) {
  SetCMDFlag(name, value);
}

template <typename ElemType>
void MV_Aggregate(ElemType* data, int size) {
  net::Allreduce(data, size);
}

int  MV_NetBind(int rank, char* endpoint) {
  return NetInterface::Get()->Bind(rank, endpoint);
}

int  MV_NetConnect(int* ranks, char* endpoints[], int size) {
  return NetInterface::Get()->Connect(ranks, endpoints, size);
}

void MV_NetFinalize() {
  NetInterface::Get()->Finalize();
}

template void MV_Aggregate<char>(char*, int);
template void MV_Aggregate<int>(int*, int);
template void MV_Aggregate<float>(float*, int);
template void MV_Aggregate<double>(double*, int);

template void MV_SetFlag<int>(const std::string&, const int&);
template void MV_SetFlag<bool>(const std::string&, const bool&);
template void MV_SetFlag<std::string>(const std::string&, const std::string&);
template void MV_SetFlag<double>(const std::string&, const double&);

}  // namespace multiverso
