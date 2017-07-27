//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"

namespace CNTK {

    // Stream (input) metadata. This text-reader specific descriptor adds two
    // additional fields: stream alias (name prefix in each sample) and expected
    // sample dimension.
    struct StreamDescriptor : StreamInformation
    {
        std::string m_alias; // sample name prefix used in the input data
        size_t m_sampleDimension; // expected number of elements in a sample
                                  // (can be omitted for sparse input)
    };

}
