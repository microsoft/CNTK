//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NDRMReader.h - Include file for the reader for the Neural Document Ranking model.
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
class NDRMReader : public DataReaderBase
{
    ConfigParameters m_readerConfig;
    std::wstring m_file;
    std::wstring m_qEmbeddingsFile;
    std::wstring m_dEmbeddingsFile;
    std::wstring m_idfFile;
    size_t m_numDocs;
    size_t m_numWordsPerQuery;
    size_t m_numWordsPerDoc;
    size_t m_vocabSize;
    size_t m_vectorSize;
    size_t m_miniBatchSize;
    size_t m_bytesPerSample;
    size_t m_bytesPerVector;
    char* m_dIdValues;
    char* m_qEmbValues;
    char* m_dEmbValues;
    char* m_labels;
    MBLayoutPtr m_pMBLayout;

#ifdef SPARSE_PCREADER_USE_WINDOWS_API
    HANDLE m_hndl;
    HANDLE m_qEmbHndl;
    HANDLE m_dEmbHndl;
    HANDLE m_idfHndl;
    HANDLE m_filemap;
    HANDLE m_qEmbFilemap;
    HANDLE m_dEmbFilemap;
    HANDLE m_idfFilemap;
#else
    int m_hndl;
    int m_qEmbHndl;
    int m_dEmbHndl;
    int m_idfHndl;
#endif
    void* m_dataBuffer;
    void* m_qEmbDataBuffer;
    void* m_dEmbDataBuffer;
    void* m_idfDataBuffer;
   
    int64_t m_filePositionMax;
    int64_t m_qEmbFilePositionMax;
    int64_t m_dEmbFilePositionMax;
    int64_t m_idfFilePositionMax;
    int64_t m_currOffset;
    size_t m_numSamplesPerEpoch;
    size_t m_numSamplesCurrEpoch;
    int m_traceLevel;

    std::map<LabelIdType, LabelType> m_mapIdToLabel;
    std::map<LabelType, LabelIdType> m_mapLabelToId;

public:
    NDRMReader()
        : m_pMBLayout(make_shared<MBLayout>())
    {
        m_pMBLayout->SetUniqueAxisName(L"NDRMReader");
    };
    virtual ~NDRMReader();
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

    size_t GetNumParallelSequencesForFixingBPTTMode()
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
        RuntimeError("GetData not supported in NDRMReader");
    };
    virtual bool DataEnd();
};
} } }
