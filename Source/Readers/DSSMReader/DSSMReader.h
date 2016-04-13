//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DSSMReader.h - Include file for the MTK and MLF format of features and samples
//
#pragma once
#include "DataReader.h"
#include "DataWriter.h"
#include "Config.h"
#include "RandomOrdering.h"
#include <string>
#include <map>
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

enum LabelKind
{
    labelNone = 0,       // no labels to worry about
    labelCategory = 1,   // category labels, creates mapping tables
    labelRegression = 2, // regression labels
    labelOther = 3,      // some other type of label
};

template <class ElemType>
class DSSM_BinaryInput
{
private:
    HANDLE m_hndl;
    HANDLE m_filemap;
    HANDLE m_header;
    HANDLE m_offsets;
    HANDLE m_data;

    // void* header_orig; // Don't need this since the header is at the start of the file
    void* offsets_orig;
    void* data_orig;

    void* header_buffer;
    void* offsets_buffer;
    void* data_buffer;

    size_t m_dim;
    size_t mbSize;
    size_t MAX_BUFFER = 300;

    ElemType* values;    // = (ElemType*)malloc(sizeof(float)* 230 * 1024);
    int64_t* offsets;    // = (int*)malloc(sizeof(int)* 230 * 1024);
    int32_t* colIndices; // = (int*)malloc(sizeof(int)* (batchsize + 1));
    int32_t* rowIndices; // = (int*)malloc(sizeof(int)* MAX_BUFFER * batchsize);

public:
    int64_t numRows;
    int32_t numCols;
    int64_t totalNNz;

    DSSM_BinaryInput();
    ~DSSM_BinaryInput();
    void Init(std::wstring fileName, size_t dim);
    bool SetupEpoch(size_t minibatchSize);
    bool Next_Batch(Matrix<ElemType>& matrices, size_t cur, size_t numToRead, int* ordering);
    void Dispose();
};

template <class ElemType>
class DSSMReader : public DataReaderBase
{
    // public:
    //    typedef std::string LabelType;
    //    typedef unsigned LabelIdType;
private:
    int* read_order; // array to shuffle to reorder the dataset
    std::wstring m_featuresNameQuery;
    std::wstring m_featuresNameDoc;
    size_t m_featuresDimQuery;
    size_t m_featuresDimDoc;
    DSSM_BinaryInput<ElemType> dssm_queryInput;
    DSSM_BinaryInput<ElemType> dssm_docInput;

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
    MBLayoutPtr m_pMBLayout;

    std::wstring m_labelsName;
    std::wstring m_featuresName;
    std::wstring m_labelsCategoryName;
    std::wstring m_labelsMapName;
    ElemType* m_qfeaturesBuffer;
    ElemType* m_dfeaturesBuffer;
    ElemType* m_labelsBuffer;
    LabelIdType* m_labelsIdBuffer;
    std::wstring m_labelFileToWrite; // set to the path if we need to write out the label file

    bool m_endReached;
    int m_traceLevel;

    // feature and label data are parallel arrays
    std::vector<ElemType> m_featureData;
    std::vector<LabelIdType> m_labelIdData;
    std::vector<LabelType> m_labelData;

    // map is from ElemType to LabelType
    // For DSSM, we really only need an int for label data, but we have to transmit in Matrix, so use ElemType instead
    std::map<LabelIdType, LabelType> m_mapIdToLabel;
    std::map<LabelType, LabelIdType> m_mapLabelToId;

    // caching support
    DataReader* m_cachingReader;
    DataWriter* m_cachingWriter;
    ConfigParameters m_readerConfig;

    size_t RandomizeSweep(size_t epochSample);
    // bool Randomize() {return m_randomizeRange != randomizeNone;}
    bool Randomize()
    {
        return false;
    }
    void SetupEpoch();
    void StoreLabel(ElemType& labelStore, const LabelType& labelValue);
    size_t RecordsToRead(size_t mbStartSample, bool tail = false);
    void ReleaseMemory();
    void WriteLabelFile();

    virtual bool ReadRecord(size_t readSample);

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
    DSSMReader()
        : m_pMBLayout(make_shared<MBLayout>())
    {
        m_qfeaturesBuffer = NULL;
        m_dfeaturesBuffer = NULL;
        m_labelsBuffer = NULL;
    }
    virtual ~DSSMReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
    virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);

    size_t GetNumParallelSequences()
    {
        return 1;
    }
    void SetNumParallelSequences(const size_t){};
    void CopyMBLayoutTo(MBLayoutPtr pMBLayout)
    {
        pMBLayout->CopyFrom(m_pMBLayout);
        NOT_IMPLEMENTED;
    }

    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);

    virtual bool DataEnd();

    void SetRandomSeed(int)
    {
        NOT_IMPLEMENTED;
    }
};
} } }
