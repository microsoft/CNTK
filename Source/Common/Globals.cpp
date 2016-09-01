//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Globals.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    atomic<bool> Globals::m_forceDeterministicAlgorithms = atomic<bool>(false);

}}}