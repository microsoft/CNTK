#ifdef MULTIVERSO_USE_MPI

#include "multiverso/net/mpi_net.h"

namespace multiverso {

template void MPINetWrapper::Allreduce<char>(char*, size_t);
template void MPINetWrapper::Allreduce<int>(int*, size_t);
template void MPINetWrapper::Allreduce<float>(float*, size_t);
template void MPINetWrapper::Allreduce<double>(double*, size_t);

}  // namespace multiverso

#endif
