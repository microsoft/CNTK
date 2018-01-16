//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "Packer.h"
#include "SequenceEnumerator.h"

namespace CNTK {

    // Base class for the reader.
    // Used for code sharing between different readers.
    // In the end there will be only composite reader, but we have to support other types of configuration 
    // currently as well.
    class ReaderBase : public Reader
    {
    public:
        // Description of streams that this reader provides.
        std::vector<StreamInformation> GetStreamDescriptions() override;

        // Starts a new epoch with the provided configuration.
        void StartEpoch(const EpochConfiguration& config, const std::map<std::wstring, int>& requiredStreams) override;

        // Reads a single minibatch.
        Minibatch ReadMinibatch() override;

        // Returns current position in the global timeline. The returned value is in samples.
        std::map<std::wstring, size_t> GetState() override;

        void SetState(const std::map<std::wstring, size_t>& state) override;

        void SetConfiguration(const ReaderConfiguration& config, const std::map<std::wstring, int>& inputDescriptions) override;

        virtual ~ReaderBase() = 0;

    protected:
        // Deserializer.
        DataDeserializerPtr m_deserializer;

        // Sequence provider.
        SequenceEnumeratorPtr m_sequenceEnumerator;

        // Packer.
        PackerPtr m_packer;

        // Required inputs for the epoch.
        std::map<std::wstring, int> m_requiredInputs;

        // Memory provider per input.
        std::vector<MemoryProviderPtr> m_memoryProviders;
    };
}
