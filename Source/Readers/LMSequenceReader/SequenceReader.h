//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// SequenceReader.h - Include file for the MTK and MLF format of features and samples
//
#pragma once
//#define LEAKDETECT

#include "DataReader.h"
#include "DataWriter.h"
#include "Config.h"
#include "SequenceParser.h"
#include "RandomOrdering.h"
#include <string>
#include <map>
#include <vector>
#include <random>

namespace Microsoft { namespace MSR { namespace CNTK {

#define STRIDX2CLS L"idx2cls"
#define CLASSINFO L"classinfo"

#define STRIDX2PROB L"idx2prob"
#define MAX_STRING 2048

enum LabelKind
{
    labelNone = 0,     // no labels to worry about
    labelCategory = 1, // category labels, creates mapping tables
    labelNextWord = 2, // sentence mapping (predicts next word)
    labelOther = 3,    // some other type of label
};

enum ReaderMode
{
    Softmax = 0, // no labels to worry about
    Class = 1,   // category labels, creates mapping tables
    NCE = 2,     // sentence mapping (predicts next word)
    Unnormalize = 3,
    None = 4, // some other type of label
};

template <typename Count>
class noiseSampler
{
    std::vector<double> m_prob, m_log_prob;
    std::uniform_int_distribution<Count> unif_int;
    bool uniform_sampling;
    double uniform_prob;
    double uniform_log_prob;
    std::piecewise_constant_distribution<double> d;
    std::mt19937 rng;

public:
    noiseSampler()
    {
    }
    noiseSampler(const std::vector<double>& counts, bool xuniform_sampling = false)
        : uniform_sampling(xuniform_sampling), rng(1234)
    {
        size_t k = counts.size();
        uniform_prob = 1.0 / k;
        uniform_log_prob = std::log(uniform_prob);
        std::vector<double> vn(counts.size() + 1);
        for (int i = 0; i < vn.size(); i++)
            vn[i] = i;
        d = std::piecewise_constant_distribution<double>(vn.begin(), vn.end(), counts.begin());
        unif_int = std::uniform_int_distribution<Count>(0, (long) counts.size() - 1);
        m_prob = d.densities();
        m_log_prob.resize(m_prob.size());
        for (int i = 0; i < k; i++)
            m_log_prob[i] = std::log(m_prob[i]);
    }
    int size() const
    {
        return m_prob.size();
    }
    double prob(int i) const
    {
        if (uniform_sampling)
            return uniform_prob;
        else
            return m_prob[i];
    }
    double logprob(int i) const
    {
        if (uniform_sampling)
            return uniform_log_prob;
        else
            return m_log_prob[i];
    }

    template <typename Engine>
    int sample(Engine& eng)
    {
        int m = unif_int(eng);
        if (uniform_sampling)
            return m;
        return (int) d(eng);
    }

    int sample()
    {
        return sample(this->rng);
    }
};

// Note: This class is deprecated for standalone use, only used as a base for BatchSequenceReader which overrides most of the functions.
template <class ElemType>
class SequenceReader : public DataReaderBase
{
protected:
    bool m_idx2clsRead;
    bool m_clsinfoRead;

    bool m_idx2probRead;

public:
    map<string, int> word4idx;
    map<int, string> idx4word;
    map<int, int> idx4class;
    map<int, size_t> idx4cnt;
    int nwords, dims, nsamps, nglen, nmefeats;
    Matrix<ElemType>* m_id2classLocal;  // CPU version
    Matrix<ElemType>* m_classInfoLocal; // CPU version

    Matrix<ElemType>* m_id2Prob; // CPU version
    int m_classSize;
    map<int, vector<int>> class_words;

    int m_noiseSampleSize;
    noiseSampler<long> m_noiseSampler;

    ReaderMode readerMode;
    int eos_idx, unk_idx;

    string mUnk; // unk symbol

public:
    //    typedef std::string LabelType;
    //    typedef unsigned LabelIdType;
protected:
    //    SequenceParser<ElemType, LabelType> m_parser;
    LMSequenceParser<ElemType, LabelType> m_parser;
    //    LMBatchSequenceParser<ElemType, LabelType> m_parser;
    size_t m_mbSize;           // size of minibatch requested
    size_t m_mbStartSample;    // starting sample # of the next minibatch
    size_t m_epochSize;        // size of an epoch
    size_t m_epoch;            // which epoch are we on
    size_t m_epochStartSample; // the starting sample for the epoch
    size_t m_totalSamples;     // number of samples in the dataset
    size_t m_featureDim;       // feature dimensions for extra features
    size_t m_featureCount;     // total number of non-zero features (in labelsDim + extra features dim)
    // for language modeling, the m_featureCount = 1, since there is only one nonzero element
    size_t m_readNextSampleLine; // next sample to read Line
    size_t m_readNextSample;     // next sample to read
    size_t m_seqIndex;           // index into the m_sequence array
    bool m_labelFirst;           // the label is the first element in a line

    enum LabelInfoType
    {
        labelInfoIn = 0,
        labelInfoOut,
        labelInfoNum
    };

    std::wstring m_labelsName[labelInfoNum];
    std::wstring m_featuresName;
    std::wstring m_labelsCategoryName[labelInfoNum];
    std::wstring m_labelsMapName[labelInfoNum];
    std::wstring m_sequenceName;

    size_t m_cacheBlockSize = 50000;

    ElemType* m_featuresBuffer;
    ElemType* m_labelsBuffer;
    LabelIdType* m_labelsIdBuffer;
    size_t* m_sequenceBuffer;

    //size_t* m_featuresBufferRow;
    //size_t* m_featuresBufferRowIdx;

    bool m_endReached;
    int m_traceLevel;

    // feature and label data are parallel arrays
    std::vector<ElemType>    m_featureData; // [j] input label index (seems only 1 dimension, category)
    std::vector<LabelIdType> m_labelIdData; // [j] output label index
    //std::vector<ElemType> m_labelData;
    std::vector<size_t>      m_sequence;
    std::map<size_t, size_t> m_indexer; // feature or label indexer

    // we have two one for input and one for output
    struct LabelInfo
    {
        LabelKind type; // labels are categories, create mapping table
        std::map<LabelIdType, LabelType> mapIdToLabel;
        std::map<LabelType, LabelIdType> mapLabelToId;
        LabelIdType numIds;        // maximum label ID we have encountered so far
        LabelIdType dim;           // maximum label ID we will ever see (used for array dimensions)
        std::string beginSequence; // starting sequence string (i.e. <s>)
        std::string endSequence;   // ending sequence string (i.e. </s>)
        std::wstring mapName;
        std::wstring fileToWrite; // set to the path if we need to write out the label file
    } m_labelInfo[labelInfoNum];

    // caching support
    DataReader* m_cachingReader;
    DataWriter* m_cachingWriter;
    ConfigParameters m_readerConfig;
    void InitCache(const ConfigParameters& config);

    void UpdateDataVariables();
    void SetupEpoch();
    void LMSetupEpoch();
    size_t RecordsToRead(size_t mbStartSample, bool tail = false);
    void ReleaseMemory();
    void WriteLabelFile();

    LabelIdType GetIdFromLabel(const std::string& label, LabelInfo& labelInfo);
    bool CheckIdFromLabel(const std::string& labelValue, const LabelInfo& labelInfo, unsigned& labelId);

    virtual bool EnsureDataAvailable(size_t mbStartSample, bool endOfDataCheck = false);
    virtual bool ReadRecord(size_t readSample);
    bool SentenceEnd();

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
    static void ReadClassInfo(const wstring& vocfile, int& classSize,
                              map<string, int>& word4idx,
                              map<int, string>& idx4word,
                              map<int, int>& idx4class,
                              map<int, size_t>& idx4cnt,
                              int nwords,
                              string mUnk,
                              noiseSampler<long>& m_noiseSampler,
                              bool flatten);
    //static void ReadWord(char* wrod, FILE* fin);

    void GetLabelOutput(StreamMinibatchInputs& matrices, size_t m_mbStartSample, size_t actualmbsize);
    void GetInputToClass(StreamMinibatchInputs& matrices);

    void GetInputProb(StreamMinibatchInputs& matrices);
    void GetClassInfo();

    virtual void Destroy();
    SequenceReader()
    {
        m_featuresBuffer = NULL;
        m_labelsBuffer = NULL;
        m_clsinfoRead = false;
        m_idx2clsRead = false;
        m_cachingReader = NULL;
        m_cachingWriter = NULL;
        m_labelsIdBuffer = NULL;
        readerMode = ReaderMode::Class;
        /*
        delete m_featuresBufferRow;
        delete m_featuresBufferRowIdx;

        delete m_labelsIdBufferRow;
        delete m_labelsBlock2Id;
        delete m_labelsBlock2UniqId;
        */
    }
    virtual ~SequenceReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
    virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);

    // void SetSentenceSegBatch(std::vector<size_t> &/*sentenceEnd*/) {};
    // TODO: ^^ should this be   void CopyMBLayoutTo(MBLayoutPtr pMBLayout);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);

    virtual bool DataEnd();

    //int GetSentenceEndIdFromOutputLabel() { return -1; };
};

template <class ElemType>
class BatchSequenceReader : public SequenceReader<ElemType>
{
    typedef SequenceReader<ElemType> Base;
public:
    using LabelType   = typename Base::LabelType;
    using LabelIdType = typename Base::LabelIdType;
    using LabelInfo   = typename Base::LabelInfo;
    using Base::m_cachingReader;
    using Base::m_cachingWriter;
    using Base::m_featuresName;
    using Base::labelInfoNum;
    using Base::m_labelsName;
    using Base::m_featureDim;
    using Base::m_classSize;
    using Base::m_labelInfo;
    using Base::labelInfoIn;
    using Base::nwords;
    using Base::ReadClassInfo;
    using Base::word4idx;
    using Base::idx4word;
    using Base::idx4cnt;
    using Base::mUnk;
    using Base::m_mbStartSample;
    using Base::m_epoch;
    using Base::m_totalSamples;
    using Base::m_epochStartSample;
    using Base::m_seqIndex;
    using Base::m_readNextSampleLine;
    using Base::m_readNextSample;
    using Base::m_traceLevel;
    using Base::m_featureCount;
    using Base::m_endReached;
    using Base::InitCache;
    using Base::m_readerConfig;
    using Base::ReleaseMemory;
    using Base::m_featuresBuffer;
    //using Base::m_featuresBufferRow;
    using Base::m_labelsBuffer;
    using Base::m_labelsIdBuffer;
    using Base::m_id2classLocal;
    using Base::m_classInfoLocal;
    using Base::m_cacheBlockSize;
    using Base::m_mbSize;
    using Base::m_epochSize;
    using Base::m_featureData;
    using Base::labelInfoOut;
    //using Base::m_labelData;
    using Base::m_labelIdData;
    using Base::LMSetupEpoch;
    using Base::m_clsinfoRead;
    using Base::m_idx2clsRead;
    //using Base::m_featuresBufferRowIdx;
    using Base::m_sequence;
    using Base::idx4class;
    using Base::m_indexer;
    using Base::m_noiseSampleSize;
    using Base::m_noiseSampler;
    using Base::readerMode;
    using Base::GetIdFromLabel;
    using Base::GetInputToClass;
    using Base::GetClassInfo;
    using Base::mRequestedNumParallelSequences; // IDataReader<ElemType>

private:
    unsigned int m_randomSeed = 0; // deterministic random seed

    size_t mLastProcessedSentenceId;

    size_t mNumRead;               // number of sentences in current cache block
    vector<bool> mProcessed;       // [mNumRead] true if sequence has already been returned in this cache block
    size_t m_epochSamplesReturned; // number of samples returned in this epoch

    vector<size_t> mToProcess;     // [] current set of sequences (gets updated each minibatch except if they are too long)

    size_t mPosInSentence;
    size_t mLastPosInSentence;
    size_t m_truncationLength;     // sequences longer than this get chopped up

    std::vector<ElemType> m_featureTemp;
    std::vector<LabelType> m_labelTemp;

    bool mSentenceEnd;
    //bool mSentenceBegin;

    MBLayoutPtr m_pMBLayout;

public:
    LMBatchSequenceParser<ElemType, LabelType> m_parser;
    BatchSequenceReader()
        : m_pMBLayout(make_shared<MBLayout>())
    {
        mLastProcessedSentenceId = 0;
        mRequestedNumParallelSequences = 1;
        mLastPosInSentence = 0;
        mNumRead = 0;
        mSentenceEnd = false;
    }

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
private:
    void Reset();
    size_t DetermineSequencesToProcess();
    bool GetMinibatchData(size_t& firstPosInSentence);
    void GetLabelOutput(StreamMinibatchInputs& matrices, size_t m_mbStartSample, size_t actualmbsize);

public:
    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize) override;
    bool TryGetMinibatch(StreamMinibatchInputs& matrices) override;
    bool DataEnd() override;

    void CopyMBLayoutTo(MBLayoutPtr pMBLayout) { assert(mToProcess.size() == m_pMBLayout->GetNumParallelSequences()); pMBLayout->CopyFrom(m_pMBLayout); }
    size_t GetNumParallelSequences() override { return mToProcess.size(); } // TODO: or get it from MBLayout? Can this ever be called before GetMinibatch()?

    // TODO: what are these?
    //bool RequireSentenceSeg() const override { return true; }
    //int GetSentenceEndIdFromOutputLabel() override;
};

}}}
