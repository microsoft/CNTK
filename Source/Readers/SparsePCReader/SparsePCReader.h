//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// SparsePCReader.h - Include file for the Sparse Parallel Corpus reader.
//
#pragma once
#include "DataReader.h"
#include "DataWriter.h"
#include "Config.h"
#include "RandomOrdering.h"
#include <string>
#include <map>
#include <vector>

// Windows or Posix? Originally the reader was done only for Windows. Keep it this way for now when running on Windows.
#ifdef __WINDOWS__
#define SPARSE_PCREADER_USE_WINDOWS_API
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class SparsePCReader : public DataReaderBase
{
    ConfigParameters m_readerConfig;
    std::wstring m_file;
    size_t m_featureCount;
    std::vector<std::wstring> m_featureNames;
    std::vector<size_t> m_dims;
    std::wstring m_labelName;
    size_t m_miniBatchSize;
    size_t m_microBatchSize;
    int64_t m_maxReadData; // For early exit during debugging
    bool m_doGradientCheck;
    bool m_returnDense;
    size_t m_sparsenessFactor;
    int32_t m_verificationCode;
    std::vector<ElemType*> m_values;
    std::vector<int32_t*> m_rowIndices;
    std::vector<int32_t*> m_colIndices;
    ElemType* m_labelsBuffer;
    MBLayoutPtr m_pMBLayout;

#ifdef SPARSE_PCREADER_USE_WINDOWS_API
    HANDLE m_hndl;
    HANDLE m_filemap;
#else
    int m_hndl;
#endif
    void* m_dataBuffer;
   
    int64_t m_filePositionMax;
    int64_t m_currOffset;
    int m_traceLevel;

    std::map<LabelIdType, LabelType> m_mapIdToLabel;
    std::map<LabelType, LabelIdType> m_mapLabelToId;

public:
    SparsePCReader()
        : m_pMBLayout(make_shared<MBLayout>()){};
    virtual ~SparsePCReader();
    virtual void Destroy();
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
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
    virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);

    size_t GetNumParallelSequences()
    {
        return m_pMBLayout->GetNumParallelSequences();
    }
    void SetNumParallelSequences(const size_t){};
    void CopyMBLayoutTo(MBLayoutPtr pMBLayout)
    {
        pMBLayout->CopyFrom(m_pMBLayout);
    }
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& /*sectionName*/, size_t /*numRecords*/, void* /*data*/, size_t& /*dataBufferSize*/, size_t /*recordStart*/)
    {
        RuntimeError("GetData not supported in SparsePCReader");
    };
    virtual bool DataEnd();
};
} } }
