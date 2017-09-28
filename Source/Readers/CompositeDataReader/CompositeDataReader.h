//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <map>
#include <string>
#include <future>
#include "DataReader.h"
#include "ReaderBase.h"
#include "Transformer.h"
#include "TransformController.h"

namespace CNTK {

class DataDeserializer;
typedef std::shared_ptr<DataDeserializer> DataDeserializerPtr;

class Transformer;
typedef std::shared_ptr<Transformer> TransformerPtr;

class Packer;
typedef std::shared_ptr<Packer> PackerPtr;

class MemoryProvider;
typedef std::shared_ptr<MemoryProvider> MemoryProviderPtr;

class CorpusDescriptor;
typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

struct EpochConfiguration;
struct Minibatch;

// The whole CompositeDataReader is meant as a stopgap to allow deserializers/transformers composition until SGD talkes 
// directly to the new Reader API. The example of the cntk configuration that this reader supports can be found at
//     Tests/EndToEndTests/Speech/HtkDeserializers/LSTM/FullUtterance/cntk.cntk
// CompositeDataReader is a factory for the new readers. Its main responsibility is to read the configuration and create the
// corresponding set of deserializers, the corpus descriptor, transformers, randomizer and packer, providing the following functionality:
//     - all input sequences are defined by the corpus descriptor
//     - deserializers provide sequences according to the corpus descriptor
//     - sequences can be transformed by the transformers applied on top of deserializer
//     - deserializers are bound together using the bundler - it bundles sequences with the same sequence id retrieved from different deserializers
//     - packer is used to pack randomized sequences into the minibatch
// The composite reader is currently also responsible for asynchronous prefetching of the minibatch data.

// In order not to break existing configs and allow deserializers composition it exposes the same interface as the old readers, but it is not exposed
// to external developers. The actual "reader developer" now has to provide deserializer(s) only.
// TODO: Implement proper corpus descriptor.
// TODO: Change this interface when SGD is changed.
class CompositeDataReader : public ReaderBase, protected Microsoft::MSR::CNTK::Plugin
{
public:
    CompositeDataReader(const Microsoft::MSR::CNTK::ConfigParameters& parameters);

    // Describes the streams this reader produces.
    std::vector<StreamInformation> GetStreamDescriptions() override;

    // Starts a new epoch with the provided configuration
    void StartEpoch(const EpochConfiguration& config, const std::map<std::wstring, int>& inputDescriptions) override;

private:
    bool CreateDeserializers(const Microsoft::MSR::CNTK::ConfigParameters& readerConfig);
    void CreateTransforms(const Microsoft::MSR::CNTK::ConfigParameters& deserializerConfig);

    DataDeserializerPtr CreateDeserializer(const Microsoft::MSR::CNTK::ConfigParameters& readerConfig, bool primary);
    TransformerPtr CreateTransformer(const Microsoft::MSR::CNTK::ConfigParameters& config, const std::string& defaultModule, const std::wstring& transformerType);

    bool ContainsDeserializer(const Microsoft::MSR::CNTK::ConfigParameters& readerConfig, const wstring& type);

    enum class PackingMode
    {
        sample,
        sequence,
        truncated
    };

    // Packing mode.
    PackingMode m_packingMode;

    // A list of deserializers.
    std::vector<DataDeserializerPtr> m_deserializers;

    // A list of transformers.
    std::vector<Transformation> m_transforms;

    // Corpus descriptor that is shared between deserializers.
    CorpusDescriptorPtr m_corpus;

    // Precision - "float" or "double".
    std::string m_precision;

    // Truncation length for BPTT mode.
    size_t m_truncationLength;
};

}
