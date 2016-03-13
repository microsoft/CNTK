//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Basics.h"

#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// ===========================================================================
// ComputationEnvironment -- computation graph and operations
// ===========================================================================

enum class NetworkOperationMode
{
    unspecified,
    training,
    inferring,
    precomputing
};

struct ComputationEnvironment
{
    NetworkOperationMode networkOperationMode = NetworkOperationMode::unspecified;
};
typedef shared_ptr<ComputationEnvironment> ComputationEnvironmentPtr;

}}}
