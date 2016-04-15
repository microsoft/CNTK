//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <map>
#include <string>
#include <future>
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class IDataDeserializer;
typedef std::shared_ptr<IDataDeserializer> IDataDeserializerPtr;

class Transformer;
typedef std::shared_ptr<Transformer> TransformerPtr;

class Packer;
typedef std::shared_ptr<Packer> PackerPtr;

class MemoryProvider;
typedef std::shared_ptr<MemoryProvider> MemoryProviderPtr;

class CorpusDescriptor;
typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

struct StreamDescription;
typedef std::shared_ptr<StreamDescription> StreamDescriptionPtr;

struct EpochConfiguration;
struct Minibatch;

// TODO: In order not to break existing configs and allow deserializers composition, this is a temporary shim for the new readers
// It will be removed and responsibilities will be moved to different parts of CNTK.
// TODO: Currently binds together several deserializers, packer and randomizer. So that the actual reader developer has to provide deserializer(s) only.
// TODO: Same code as in ReaderLib shim, the one in the ReaderLib will be deleted as the next step.
class CompositeDataReader : public IDataReader, protected Plugin, public ScriptableObjects::Object
{
public:
    CompositeDataReader(const std::string& precision);

    // Currently we do not support BS configuration.
    virtual void Init(const ScriptableObjects::IConfigRecord& /*config*/) override
    {
        assert(false);
    }

    virtual void Init(const ConfigParameters& config) override;

    virtual void Destroy() override
    {
        delete this;
    }

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize) override;
    virtual void StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples) override;

    virtual bool SupportsDistributedMBRead() const override
    {
        return true;
    }

    virtual bool GetMinibatch(StreamMinibatchInputs& matrices) override;
    virtual bool DataEnd() override;
    void CopyMBLayoutTo(MBLayoutPtr) override;
    virtual size_t GetNumParallelSequences() override;

private:
    void CreateDeserializers(const ConfigParameters& readerConfig);
    IDataDeserializerPtr CreateDeserializer(const ConfigParameters& readerConfig, bool primary);
    void StartEpoch(const EpochConfiguration& config);

    enum class PackingMode
    {
        sample,
        sequence,
        truncated
    };

    // Packing mode.
    PackingMode m_packingMode;

    // Pre-fetch task.
    std::future<Minibatch> m_prefetchTask;

    // Launch type of prefetch - async or sync.
    launch m_launchType;

    // Flag indicating end of the epoch.
    bool m_endOfEpoch;

    // MBLayout of the reader. 
    // TODO: Will be taken from the StreamMinibatchInputs.
    MBLayoutPtr m_layout;

    // Stream name to id mapping.
    std::map<std::wstring, size_t> m_nameToStreamId;

    // All streams this reader provides.
    std::vector<StreamDescriptionPtr> m_streams;

    // A list of deserializers.
    std::vector<IDataDeserializerPtr> m_deserializers;

    // Randomizer.
    // TODO: remove Transformer interface from randomizer.
    TransformerPtr m_randomizer;

    // TODO: Should be removed. We already have matrices on this level.
    // Should just get the corresponding pinned memory.
    MemoryProviderPtr m_provider;

    // Corpus descriptor that is shared between deserializers.
    CorpusDescriptorPtr m_corpus;

    // Packer.
    PackerPtr m_packer;

    // Precision - "float" or "double".
    std::string m_precision;

    // Truncation length for BPTT mode.
    size_t m_truncationLength;
};

}}}
