//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "NetworkDescriptionLanguage.h"
#include "ComputationNetwork.h"
//#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class IExecutionEngine
{
public:
    virtual ComputationNetworkPtr GetComputationNetwork() = 0;

    virtual NDLNodeEvaluator<ElemType>& GetNodeEvaluator() = 0;

    virtual ~IExecutionEngine(){};
};
} } }
