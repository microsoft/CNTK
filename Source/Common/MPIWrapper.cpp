//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Include/Basics.h"
#include "Include/MPIWrapper.h"

using namespace Microsoft::MSR::CNTK;
int MPIWrapper::s_myRank = -1;
std::shared_ptr<MPIWrapper> Microsoft::MSR::CNTK::MPIWrapper::s_mpi = nullptr;
