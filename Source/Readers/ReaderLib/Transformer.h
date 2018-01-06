//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "Reader.h"

namespace CNTK {

// Defines a data transformation interface.
// Transformers are responsible for doing custom transformation of sequences.
// For example for images, there could be scale, crop, or median transformation.
class Transformer
{
public:
    // Starts a new epoch. Some transformers have to change their configuration
    // based on the epoch.
    virtual void StartEpoch(const EpochConfiguration &config) = 0;

    // Transformers are applied on a particular input stream - this method should describe
    // how inputStream is transformed to the output stream (return value)
    virtual StreamInformation Transform(const StreamInformation& inputStream) = 0;

    // This method should describe how input sequences is transformed to the output sequence.
    virtual SequenceDataPtr Transform(SequenceDataPtr inputSequence, int indexInBatch=0) = 0;

    virtual ~Transformer()
    {
    }
};

typedef std::shared_ptr<Transformer> TransformerPtr;
}
