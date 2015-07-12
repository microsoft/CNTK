//
// <copyright file="IComputationNetBuilder.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
#include <string>

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            template<class ElemType>
            class IComputationNetBuilder //Abstract Class that cannot be instantiated
            {
            public:
                virtual ComputationNetwork<ElemType>* LoadNetworkFromFile(const std::wstring& modelFileName, bool forceLoad = true,
                    bool bAllowNoCriterion = false, ComputationNetwork<ElemType>* = nullptr) = 0;
                virtual ComputationNetwork<ElemType>* BuildNetworkFromDescription(ComputationNetwork<ElemType>* = nullptr) = 0;
                virtual ~IComputationNetBuilder() {};
            };
        }
    }
}