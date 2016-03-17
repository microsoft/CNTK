#include "Include/Basics.h"
#include "Include/MPIWrapper.h"

int Microsoft::MSR::CNTK::MPIWrapper::s_myRank = -1;
std::shared_ptr<Microsoft::MSR::CNTK::MPIWrapper> Microsoft::MSR::CNTK::MPIWrapper::s_mpi = nullptr;
