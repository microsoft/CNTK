//
// <copyright file="IExecutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "NetworkDescriptionLanguage.h"
#include "ComputationNetwork.h"
//#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {
    template<class ElemType>
    class IExecutionEngine
    {
    public:
        virtual ComputationNetworkPtr GetComputationNetwork() = 0;

        virtual NDLNodeEvaluator<ElemType> & GetNodeEvaluator() = 0;

        virtual ~IExecutionEngine() {};
    };
}}}
