//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

    // Class containing global configuration for CNTK.
    class Globals
    {
    public:
        static void ForceDeterministicAlgorithms()
        {
            m_forceDeterministicAlgorithms = true;
        }

        static bool ShouldForceDeterministicAlgorithms()
        {
            return m_forceDeterministicAlgorithms;
        }

    private:
        static bool m_forceDeterministicAlgorithms;
    };
}}}
