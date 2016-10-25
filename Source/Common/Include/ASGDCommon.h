//
// <copyright file="ASGDCommon.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

enum class AdjustLearningRateatBeginning : int
{
    None = 0,
    Linearly = 1,
    Staircase = (1 << 1),
};

}}}
