//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ReaderShim.h: Currently we are preserving the old interface in SGD. So this shim exposes the old interface and calls into the 
// reader implemented with the new interfaces (reader/packer/transforms/serializers)
//

#pragma once

#include <map>
#include <string>
#include "DataReader.h"
#include <future>
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

typedef ReaderPtr (*ReaderFactory)(const ConfigParameters& parameters);

template <class ElemType>
class ReaderShim : public IDataReader
{
public:
    explicit ReaderShim(ReaderFactory factory);
    virtual ~ReaderShim() { }

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
    std::future<Minibatch> m_prefetchTask;
    ReaderPtr m_reader;
    ReaderFactory m_factory;
    bool m_endOfEpoch;

    MBLayoutPtr m_layout;

    std::map<std::wstring, size_t> m_nameToStreamId;
    std::vector<StreamDescriptionPtr> m_streams;
    launch m_launchType;

    void FillMatrixFromStream(StorageType type, Matrix<ElemType>* matrix, size_t numRows, const StreamMinibatchPtr& stream);
};

}}}
