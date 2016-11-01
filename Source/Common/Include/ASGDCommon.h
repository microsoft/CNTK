//
// <copyright file="ASGDCommon.h" company="Microsoft">
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// </copyright>
//
#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// class AdjustLearningRateAtBeginning
//       Providing option for DataParallelASGD training. so that every nodes
//       could adjust learning rate every minibatch at first N epochs.
// -----------------------------------------------------------------------


// TODO: We can removed these options once we can adjust learning rate at minibatchs level

enum class AdjustLearningRateAtBeginning : int
{
    None = 0,  // default, don't adjust learning rate
    Linearly = 1, // using linear adjustment, learning rate will from 0 to learningRatesPerMB
    Staircase = (1 << 1), // using staircased adjustment, learning rate will from 0 to learningRatesPerMB every adjustNbMinibatch
};

}}}
