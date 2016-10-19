#include "Include/Basics.h"
#include "Include/MPIWrapper.h"

using namespace Microsoft::MSR::CNTK;
int MPIWrapper::s_myRank = -1;
std::shared_ptr<MPIWrapper> Microsoft::MSR::CNTK::MPIWrapper::s_mpi = nullptr;
bool MPIWrapper::s_initialized = false;
