#include "multiverso/net.h"

#include <limits>
#include <mutex>
#include "multiverso/message.h"
#include "multiverso/util/log.h"

#include "multiverso/net/zmq_net.h"
#include "multiverso/net/mpi_net.h"

namespace multiverso {

NetInterface* NetInterface::Get() {
#ifdef MULTIVERSO_USE_ZMQ
  static ZMQNetWrapper net_impl;
  return &net_impl;
#else
// #ifdef MULTIVERSO_USE_MPI
  // Use MPI by default
  static MPINetWrapper net_impl;
  return &net_impl;
// #endif
#endif
}

namespace net {
template <typename Typename>
void Allreduce(Typename* data, size_t elem_count) {
#ifdef MULTIVERSO_USE_MPI
  CHECK(NetInterface::Get()->active());
  MPINetWrapper::Allreduce(data, elem_count);
#else
  Log::Fatal("Not implemented yet");
#endif
}

template void Allreduce<char>(char*, size_t);
template void Allreduce<int>(int*, size_t);
template void Allreduce<float>(float*, size_t);
template void Allreduce<double>(double*, size_t);

}  // namespace net


}  // namespace multiverso
