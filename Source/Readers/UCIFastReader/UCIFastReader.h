//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// UCIFastReader.h - Include file for the MTK and MLF format of features and samples
//
#pragma once
#include "stdafx.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "Config.h"
#include "RandomOrdering.h"
#include <future>
#include "UCIParser.h"
#include <string>
#include <map>
#include <vector>
#include "CUDAPageLockedMemAllocator.h"

static inline size_t RoundUp(size_t m, size_t n)
{
    if (m % n == 0)
        return m / n;
    else
        return m / n + 1;
}

namespace Microsoft { namespace MSR { namespace CNTK {

enum LabelKind
{
    labelNone = 0,       // no labels to worry about
    labelCategory = 1,   // category labels, creates mapping tables
    labelRegression = 2, // regression labels
    labelOther = 3,      // some other type of label
};

template <class ElemType>
class UCIFastReader : public DataReaderBase
{
    shared_ptr<UCIParser<ElemType, LabelType>> m_parser;
    size_t m_mbSize;                 // size of minibatch requested
    LabelIdType m_labelIdMax;        // maximum label ID we have encountered so far
    LabelIdType m_labelDim;          // maximum label ID we will ever see (used for array dimensions)
    size_t m_mbStartSample;          // starting sample # of the next minibatch
    size_t m_epochSize;              // size of an epoch
    size_t m_epoch;                  // which epoch are we on
    size_t m_epochStartSample;       // the starting sample for the epoch
    size_t m_totalSamples;           // number of samples in the dataset
    size_t m_randomizeRange;         // randomization range
    size_t m_featureCount;           // feature count
    size_t m_readNextSample;         // next sample to read
    bool m_labelFirst;               // the label is the first element in a line
    bool m_partialMinibatch;         // a partial minibatch is allowed
    LabelKind m_labelType;           // labels are categories, create mapping table
    RandomOrdering m_randomordering; // randomizing class

    std::wstring m_labelsName;
    std::wstring m_featuresName;
    std::wstring m_labelsCategoryName;
    std::wstring m_labelsMapName;
    std::shared_ptr<ElemType> m_featuresBuffer;
    std::shared_ptr<ElemType> m_labelsBuffer;
    std::shared_ptr<LabelIdType> m_labelsIdBuffer;
    std::wstring m_labelFileToWrite; // set to the path if we need to write out the label file

    // Prefetching related fields
    bool m_prefetchEnabled;
    std::future<bool> m_pendingAsyncGetMinibatch;
    StreamMinibatchInputs m_prefetchMatrices;

    // Distributed reading related fields
    size_t m_subsetNum;
    size_t m_numSubsets;

    bool m_endReached;
    int m_traceLevel;

    // feature and label data are parallel arrays
    std::vector<ElemType> m_featureData;
    std::vector<LabelIdType> m_labelIdData;
    std::vector<LabelType> m_labelData;
    MBLayoutPtr m_pMBLayout;

    // map is from ElemType to LabelType
    // For UCI, we really only need an int for label data, but we have to transmit in Matrix, so use ElemType instead
    std::map<LabelIdType, LabelType> m_mapIdToLabel;
    std::map<LabelType, LabelIdType> m_mapLabelToId;

    /**
        for reading one line per file, i.e., a file has only one line of data
        */
    bool mOneLinePerFile;

    unique_ptr<CUDAPageLockedMemAllocator> m_cudaAllocator;

    // caching support
    DataReader* m_cachingReader;
    DataWriter* m_cachingWriter;
    ConfigParameters m_readerConfig;
    void InitCache(const ConfigParameters& config);
    void InitCache(const ScriptableObjects::IConfigRecord& config);

    size_t RandomizeSweep(size_t epochSample);
    bool Randomize()
    {
        return m_randomizeRange != randomizeNone;
    }
    size_t UpdateDataVariables(size_t mbStartSample);
    void SetupEpoch();
    void StoreLabel(ElemType& labelStore, const LabelType& labelValue);
    size_t RecordsToRead(size_t mbStartSample, bool tail = false);
    void ReleaseMemory();
    void WriteLabelFile();

    virtual bool EnsureDataAvailable(size_t mbStartSample, bool endOfDataCheck = false);
    virtual bool ReadRecord(size_t readSample);

    // Helper functions
    unique_ptr<CUDAPageLockedMemAllocator>& GetCUDAAllocator(int deviceID);
    std::shared_ptr<ElemType> AllocateIntermediateBuffer(int deviceID, size_t numElements);

public:
    template <class ConfigRecordType>
    void InitFromConfig(const ConfigRecordType&);
    virtual void Init(const ConfigParameters& config) override
    {
        InitFromConfig(config);
    }
    virtual void Init(const ScriptableObjects::IConfigRecord& config) override
    {
        InitFromConfig(config);
    }
    virtual void Destroy();
    UCIFastReader()
    {
        m_pMBLayout = make_shared<MBLayout>();
    }
    virtual ~UCIFastReader();

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize)
    {
        return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
    }

    virtual bool SupportsDistributedMBRead() const override
    {
        return ((m_cachingReader == nullptr) || m_cachingReader->SupportsDistributedMBRead());
    }

    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize) override;

    virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);

    bool GetMinibatchImpl(StreamMinibatchInputs& matrices);

    size_t GetNumParallelSequences()
    {
        return m_pMBLayout->GetNumParallelSequences();
    }
    void CopyMBLayoutTo(MBLayoutPtr pMBLayout)
    {
        pMBLayout->CopyFrom(m_pMBLayout);
    };
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);

    virtual bool DataEnd();
    void SetSentenceSegBatch(Matrix<float>&, Matrix<ElemType>&){};

    void SetNumParallelSequences(const size_t sz);

    void SetRandomSeed(int)
    {
        NOT_IMPLEMENTED;
    }
};
} } }
