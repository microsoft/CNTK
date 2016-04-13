//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// LUSequenceReader.h - Include file for the MTK and MLF format of features and samples
//
#pragma once
//#define LEAKDETECT

#include "Basics.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "LUSequenceParser.h"
#include "Config.h" // for intargvector
#include "ScriptableObjects.h"
#include <string>
#include <map>
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef DBG_SMT
#define CACHE_BLOCK_SIZE 2
#else
#define CACHE_BLOCK_SIZE 50000
#endif

#define STRIDX2CLS L"idx2cls"
#define CLASSINFO L"classinfo"

#define MAX_STRING 100000

#define NULLLABEL 65532

enum LabelKind
{
    labelNone = 0,     // no labels to worry about
    labelCategory = 1, // category labels, creates mapping tables
    labelNextWord = 2, // sentence mapping (predicts next word)
    labelOther = 3,    // some other type of label
};

enum ReaderMode
{
    Plain = 0, // no class info
    Class = 1, // category labels, creates mapping tables
};

template <class ElemType>
class LUSequenceReader : public DataReaderBase
{
protected:
    bool m_idx2clsRead;
    bool m_clsinfoRead;

public:
    using LabelType = wstring;
    using LabelIdType = long;
    long nwords, dims, nsamps, nglen, nmefeats;

    int m_seed;
    bool mRandomize;

public:
    // deal with OOV
    map<LabelType, LabelType> mWordMapping;
    wstring mWordMappingFn;
    LabelType mUnkStr;

public:
    // accumulated number of sentneces read so far
    unsigned long mTotalSentenceSofar;

protected:
    BatchLUSequenceParser<ElemType, LabelType> m_parser;
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
    intargvector m_wordContext;

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

    ElemType* m_featuresBuffer;
    ElemType* m_labelsBuffer;
    LabelIdType* m_labelsIdBuffer;
    size_t* m_sequenceBuffer;

    bool m_endReached;
    int m_traceLevel;

    // feature and label data are parallel arrays
    // The following two hold the actual MB data internally, created by EnsureDataAvailable().
    std::vector<std::vector<vector<LabelIdType>>> m_featureWordContext; // [parSeq + t * numParSeq] word n-tuple in order of storage in m_value matrix
    std::vector<LabelIdType> m_labelIdData;

    std::vector<ElemType> m_labelData;
    std::vector<size_t> m_sequence;

    // we have two one for input and one for output
    struct LabelInfo
    {
        LabelKind type; // labels are categories, create mapping table
        map<LabelType, LabelIdType> word4idx;
        map<LabelIdType, LabelType> idx4word;
        long dim;                // maximum label ID we will ever see (used for array dimensions)
        LabelType beginSequence; // starting sequence string (i.e. <s>)
        LabelType endSequence;   // ending sequence string (i.e. </s>)
        bool busewordmap;        // whether using wordmap to map unseen words to unk
        std::wstring mapName;
        std::wstring fileToWrite; // set to the path if we need to write out the label file

        bool isproposal; // whether this is for proposal generation

        ReaderMode readerMode;
        /**
        word class info saved in file in format below
        ! 29
        # 58
        $ 26
        where the first column is the word and the second column is the class id, base 0
        */
        map<wstring, long> word4cls;
        map<long, long> idx4class;
        Matrix<ElemType>* m_id2classLocal;  // CPU version
        Matrix<ElemType>* m_classInfoLocal; // CPU version
        int mNbrClasses;
        bool m_clsinfoRead;
    } m_labelInfo[labelInfoNum];

    // caching support
    DataReader* m_cachingReader;
    DataWriter* m_cachingWriter;
    ConfigParameters m_readerConfig;
    void InitCache(const ConfigParameters& config);

    void UpdateDataVariables();
    void LMSetupEpoch();
    size_t RecordsToRead(size_t mbStartSample, bool tail = false);
    void ReleaseMemory();
    void WriteLabelFile();
    void LoadLabelFile(const std::wstring& filePath, std::vector<LabelType>& retLabels);

    LabelIdType GetIdFromLabel(const LabelType& label, LabelInfo& labelInfo);
    bool GetIdFromLabel(const vector<LabelIdType>& label, vector<LabelIdType>& val);
    bool CheckIdFromLabel(const LabelType& labelValue, const LabelInfo& labelInfo, unsigned& labelId);

    bool SentenceEnd();

public:
    void Init(const ConfigParameters&){};
    void Init(const ScriptableObjects::IConfigRecord&){};
    void ChangeMaping(const map<LabelType, LabelType>& maplist,
                      const LabelType& unkstr,
                      map<LabelType, LabelIdType>& word4idx);

    void Destroy(){};

    LUSequenceReader()
    {
        m_featuresBuffer = NULL;
        m_labelsBuffer = NULL;
        m_clsinfoRead = false;
        m_idx2clsRead = false;
    }
    ~LUSequenceReader(){};
    void StartMinibatchLoop(size_t, size_t, size_t = requestDataSize){};

    void SetNumParallelSequences(const size_t /*mz*/){};
    void SentenceEnd(std::vector<size_t>& /*sentenceEnd*/){};

    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);

//public:
//    int GetSentenceEndIdFromOutputLabel();
};

template <class ElemType>
class BatchLUSequenceReader : public LUSequenceReader<ElemType>
{
public:
    using LabelType = wstring;
    using LabelIdType = long;
    using LUSequenceReader<ElemType>::mWordMappingFn;
    using LUSequenceReader<ElemType>::m_cachingReader;
    using LUSequenceReader<ElemType>::mWordMapping;
    using LUSequenceReader<ElemType>::mUnkStr;
    using LUSequenceReader<ElemType>::m_cachingWriter;
    using LUSequenceReader<ElemType>::m_featuresName;
    using LUSequenceReader<ElemType>::m_labelsName;
    using LUSequenceReader<ElemType>::labelInfoNum;
    using LUSequenceReader<ElemType>::m_featureDim;
    using LUSequenceReader<ElemType>::m_labelInfo;
    //  using LUSequenceReader<ElemType>::m_labelInfoIn;
    using LUSequenceReader<ElemType>::m_mbStartSample;
    using LUSequenceReader<ElemType>::m_epoch;
    using LUSequenceReader<ElemType>::m_totalSamples;
    using LUSequenceReader<ElemType>::m_epochStartSample;
    using LUSequenceReader<ElemType>::m_seqIndex;
    using LUSequenceReader<ElemType>::m_endReached;
    using LUSequenceReader<ElemType>::m_readNextSampleLine;
    using LUSequenceReader<ElemType>::m_readNextSample;
    using LUSequenceReader<ElemType>::m_traceLevel;
    using LUSequenceReader<ElemType>::m_wordContext;
    using LUSequenceReader<ElemType>::m_featureCount;
    using typename LUSequenceReader<ElemType>::LabelInfo;
    using LUSequenceReader<ElemType>::labelInfoIn;
    using LUSequenceReader<ElemType>::labelInfoOut;
    //  using LUSequenceReader<ElemType>::arrayLabels;
    using LUSequenceReader<ElemType>::m_readerConfig;
    using LUSequenceReader<ElemType>::m_featuresBuffer;
    using LUSequenceReader<ElemType>::m_labelsBuffer;
    using LUSequenceReader<ElemType>::m_labelsIdBuffer;
    using LUSequenceReader<ElemType>::m_mbSize;
    using LUSequenceReader<ElemType>::m_epochSize;
    using LUSequenceReader<ElemType>::m_sequence;
    using LUSequenceReader<ElemType>::m_labelData;
    using LUSequenceReader<ElemType>::m_labelIdData;
    using LUSequenceReader<ElemType>::m_idx2clsRead;
    using LUSequenceReader<ElemType>::m_clsinfoRead;
    using LUSequenceReader<ElemType>::m_featureWordContext;
    using LUSequenceReader<ElemType>::LoadLabelFile;
    using LUSequenceReader<ElemType>::ReleaseMemory;
    using LUSequenceReader<ElemType>::LMSetupEpoch;
    using LUSequenceReader<ElemType>::ChangeMaping;
    using LUSequenceReader<ElemType>::GetIdFromLabel;
    using LUSequenceReader<ElemType>::InitCache;
    using LUSequenceReader<ElemType>::mRandomize;
    using LUSequenceReader<ElemType>::m_seed;
    using LUSequenceReader<ElemType>::mTotalSentenceSofar;
    //using LUSequenceReader<ElemType>::GetSentenceEndIdFromOutputLabel;

private:
    size_t mLastProcessedSentenceId;
    size_t mRequestedNumParallelSequences;
    size_t mPosInSentence;
    vector<size_t> mToProcess; // [seqIndex] utterance id of utterance in this minibatch's position [seqIndex]
    size_t mLastPosInSentence; // BPTT cursor
    size_t mNumRead;

    std::vector<vector<LabelIdType>> m_featureTemp;
    std::vector<LabelIdType> m_labelTemp;

    bool mSentenceEnd;
    bool mSentenceBegin;

public:
    vector<bool> mProcessed;
    BatchLUSequenceParser<ElemType, LabelType> m_parser;
    BatchLUSequenceReader()
        : m_pMBLayout(make_shared<MBLayout>())
    {
        mLastProcessedSentenceId = 0;
        mRequestedNumParallelSequences = 1;
        mLastPosInSentence = 0;
        mNumRead = 0;
        mSentenceEnd = false;
        mSentenceBegin = true;
        mIgnoreSentenceBeginTag = false;
    }

    ~BatchLUSequenceReader();

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
    void Reset();

    // return length of sentences size
    size_t FindNextSentences(size_t numSentences);
    bool DataEnd();
    void SetSentenceEnd(int wrd, int pos, int actualMbSize);
    void SetSentenceBegin(int wrd, int pos, int actualMbSize);
    void SetSentenceBegin(int wrd, size_t pos, size_t actualMbSize)
    {
        SetSentenceBegin(wrd, (int) pos, (int) actualMbSize);
    } // TODO: clean this up
    void SetSentenceEnd(int wrd, size_t pos, size_t actualMbSize)
    {
        SetSentenceEnd(wrd, (int) pos, (int) actualMbSize);
    }
    void SetSentenceBegin(size_t wrd, size_t pos, size_t actualMbSize)
    {
        SetSentenceBegin((int) wrd, (int) pos, (int) actualMbSize);
    }
    void SetSentenceEnd(size_t wrd, size_t pos, size_t actualMbSize)
    {
        SetSentenceEnd((int) wrd, (int) pos, (int) actualMbSize);
    }

    size_t GetLabelOutput(StreamMinibatchInputs& matrices, LabelInfo& labelInfo, size_t actualmbsize);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
    bool TryGetMinibatch(StreamMinibatchInputs& matrices);

    bool EnsureDataAvailable(size_t mbStartSample);
    size_t GetNumParallelSequences();
    void SetNumParallelSequences(const size_t mz);

    void CopyMBLayoutTo(MBLayoutPtr pMBLayout);

public:
    void GetClassInfo(LabelInfo& lblInfo);
    void ReadLabelInfo(const wstring& vocfile,
                       map<wstring, long>& word4idx,
                       bool readClass,
                       map<wstring, long>& word4cls,
                       map<long, wstring>& idx4word,
                       map<long, long>& idx4class,
                       int& mNbrCls);

    template <class ConfigRecordType>
    void LoadWordMapping(const ConfigRecordType& config);
    bool CanReadFor(wstring nodeName); // return true if this reader can output for a node with name nodeName

    vector<size_t> ReturnToProcessId()
    {
        return mToProcess;
    }
    void SetToProcessId(const vector<size_t>& tp)
    {
        mToProcess = tp;
    }

    void SetRandomSeed(int seed)
    {
        m_seed = seed;
    }

public:
    /**
    for sequential reading data, useful for beam search decoding
    */
    // this is for frame-by-frame reading of data.
    // data is first read into these matrices and then if needed is column-by-column retrieved
    map<wstring, std::shared_ptr<Matrix<ElemType>>> mMatrices;
    bool GetFrame(StreamMinibatchInputs& matrices, const size_t tidx, vector<size_t>& history);

    // create proposals
    void InitProposals(StreamMinibatchInputs& pMat);

public:
    bool mEqualLengthOutput;
    bool mAllowMultPassData;

    // return length of sentences size
    vector<size_t> mSentenceLengths; // [seqIndex] lengths of all sentences in a minibatch
    size_t mMaxSentenceLength;       // max over mSentenceLength[]  --TODO: why not compute on the fly?
    vector<int> mSentenceBeginAt;    // [seqIndex] index of first token
    const int NO_INPUT = -2;
    vector<int> mSentenceEndAt; // [seqIndex] index of last token

    MBLayoutPtr m_pMBLayout;

    // if true, reader will set to ((int) MinibatchPackingFlags::None) for time positions that are orignally correspond to ((int) MinibatchPackingFlags::SequenceStart)
    // set to true so that a current minibatch can uses state activities from the previous minibatch.
    // default will have truncated BPTT, which only does BPTT inside a minibatch
    // by default it is false
    bool mIgnoreSentenceBeginTag;
};

template <class ElemType>
class MultiIOBatchLUSequenceReader : public BatchLUSequenceReader<ElemType>
{
private:
    map<wstring, BatchLUSequenceReader<ElemType>*> mReader;

    bool mCheckDictionaryKeys;
    std::map<std::wstring, BatchLUSequenceReader<ElemType>*> nameToReader;

public:
    MultiIOBatchLUSequenceReader()
    {
        mCheckDictionaryKeys = true;
        nameToReader.clear();
    }

    ~MultiIOBatchLUSequenceReader()
    {
        for (typename map<wstring, BatchLUSequenceReader<ElemType>*>::iterator p = mReader.begin(); p != mReader.end(); p++)
        {
            delete[] p->second;
        }
    };

    bool TryGetMinibatch(StreamMinibatchInputs& matrices);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples);

    void CopyMBLayoutTo(MBLayoutPtr pMBLayout);

    size_t GetNumParallelSequences();

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

public:
    void SetRandomSeed(int);

public:
    //int GetSentenceEndIdFromOutputLabel();
    bool DataEnd();

    // create proposals
    void InitProposals(StreamMinibatchInputs& pMat);
    bool GetProposalObs(StreamMinibatchInputs& matrices, const size_t tidx, vector<size_t>& history);
};
} } }
