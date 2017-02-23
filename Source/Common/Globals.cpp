//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Globals.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    // TODO: get rid of this source file once static initializers in methods are thread-safe (VS 2015)
    std::atomic<bool> Globals::m_forceDeterministicAlgorithms(false);
    std::atomic<bool> Globals::m_forceConstantRandomSeed(false);

    std::atomic<bool> Globals::m_enableShareNodeValueMatrices(true);
    std::atomic<bool> Globals::m_optimizeGradientAccumulation(true);

}}}
