//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class Transformer;
typedef std::shared_ptr<Transformer> TransformerPtr;

// Defines a data transformation interface.
// Transformers are responsible for doing custom transformation of sequences.
// For example for images, there could be scale, crop, or median transformation.
class Transformer
{
public:
    // Starts a new epoch. Some transformers have to change their configuration
    // based on the epoch.
    virtual void StartEpoch(const EpochConfiguration &config) = 0;

    // Transforms input stream into output stream.
    virtual StreamDescription Transform(const StreamDescription& inputStream) = 0;

    // Transforms input sequences into output sequence.
    virtual SequenceDataPtr Transform(SequenceDataPtr sequence) = 0;

    virtual ~Transformer()
    {
    }
};

}}}
