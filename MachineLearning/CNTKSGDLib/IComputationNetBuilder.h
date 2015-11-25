#if 1   // only needed for some unused code in MultiNetworksSGD.h
//
// <copyright file="IComputationNetBuilder.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

    // This interface provides only one method: BuildNetworkFromDescription().
    // There are two variants currently:
    //  - SimpleNetworkBuilder: standard networks built from a few parameters
    //  - NDLNetworkBuilder: networks built using the old CNTK NDL
    // The use of this interface is very local (eventually it will be local to DoTrain() only), so there will no longer be a need to even have this interface.
    // Models created through BrainScript (or Python) do not go through this interface.

    template<class ElemType>
    /*interface*/ struct IComputationNetBuilder
    {
        virtual ComputationNetworkPtr BuildNetworkFromDescription(ComputationNetwork* = nullptr) = 0;
        virtual ~IComputationNetBuilder() {};
    };

}}}
#endif
