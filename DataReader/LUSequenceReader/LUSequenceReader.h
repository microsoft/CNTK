//
// <copyright file="LUSequenceReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// LUSequenceReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
//#define LEAKDETECT

#include "DataReader.h"
#include "DataWriter.h"
#include "LUSequenceParser.h"
#include <string>
#include <map>
#include <vector>
#include "minibatchsourcehelpers.h"


namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef DBG_SMT
#define CACHE_BLOG_SIZE 100
#else
#define CACHE_BLOG_SIZE 50000
#endif

#define STRIDX2CLS L"idx2cls"
#define CLASSINFO  L"classinfo"
    
#define MAX_STRING  100000

#define NULLLABEL 65532

enum LabelKind
{
    labelNone = 0,  // no labels to worry about
    labelCategory = 1, // category labels, creates mapping tables
    labelNextWord = 2,  // sentence mapping (predicts next word)
    labelOther = 3, // some other type of label
};

template<class ElemType>
class LUSequenceReader : public IDataReader<ElemType>
{
protected:
    bool   m_idx2clsRead; 
    bool   m_clsinfoRead;

    std::wstring m_file; 
public:
    using LabelType = wstring;
    using LabelIdType = long;
    long nwords, dims, nsamps, nglen, nmefeats;

    int m_seed; 
    bool mRandomize;

public:
    /// deal with OOV
    map<LabelType, LabelType> mWordMapping;
    string mWordMappingFn;
    LabelType mUnkStr;

public:
    /// accumulated number of sentneces read so far
    unsigned long mTotalSentenceSofar;

protected:

    LUBatchLUSequenceParser<ElemType, LabelType> m_parser;
    size_t m_mbSize;    // size of minibatch requested
    size_t m_mbStartSample; // starting sample # of the next minibatch
    size_t m_epochSize; // size of an epoch
    size_t m_epoch; // which epoch are we on
    size_t m_epochStartSample; // the starting sample for the epoch
    size_t m_totalSamples;  // number of samples in the dataset
    size_t m_featureDim; // feature dimensions for extra features
    size_t m_featureCount; // total number of non-zero features (in labelsDim + extra features dim)
    /// for language modeling, the m_featureCount = 1, since there is only one nonzero element
    size_t m_readNextSampleLine; // next sample to read Line
    size_t m_readNextSample; // next sample to read
    size_t m_seqIndex; // index into the m_sequence array
    bool m_labelFirst;  // the label is the first element in a line
    intargvector m_wordContext;

    enum LabelInfoType
    {
        labelInfoMin = 0,
        labelInfoIn = labelInfoMin,
        labelInfoOut,
        labelInfoMax
    };

    std::wstring m_labelsName[labelInfoMax];
    std::wstring m_featuresName;
    std::wstring m_labelsCategoryName[labelInfoMax];
    std::wstring m_labelsMapName[labelInfoMax];
    std::wstring m_sequenceName;

    ElemType* m_featuresBuffer;
    ElemType* m_labelsBuffer;
    LabelIdType* m_labelsIdBuffer;
    size_t* m_sequenceBuffer;

    bool m_endReached;
    int m_traceLevel;
   
    // feature and label data are parallel arrays
    std::vector<std::vector<vector<LabelIdType>>> m_featureWordContext;
    std::vector<vector<LabelIdType>> m_featureData;
    std::vector<LabelIdType> m_labelIdData;
    std::vector<ElemType> m_labelData;
    std::vector<size_t> m_sequence;

    // we have two one for input and one for output
    struct LabelInfo
    {
        LabelKind type;  // labels are categories, create mapping table
        std::map<LabelIdType, LabelType> mapIdToLabel;
        std::map<LabelType, LabelIdType> mapLabelToId;
        map<LabelType, LabelIdType> word4idx;
        map<LabelIdType, LabelType> idx4word;
        LabelIdType idMax; // maximum label ID we have encountered so far
        long dim; // maximum label ID we will ever see (used for array dimensions)
        LabelType beginSequence; // starting sequence string (i.e. <s>)
        LabelType endSequence; // ending sequence string (i.e. </s>)
        bool busewordmap; /// whether using wordmap to map unseen words to unk
        std::wstring mapName;
        std::wstring fileToWrite;  // set to the path if we need to write out the label file

        bool isproposal; /// whether this is for proposal generation

    } m_labelInfo[labelInfoMax];

    // caching support
    DataReader<ElemType>* m_cachingReader;
    DataWriter<ElemType>* m_cachingWriter;
    ConfigParameters m_readerConfig;
    void InitCache(const ConfigParameters& config);

    void UpdateDataVariables();
    void LMSetupEpoch();
    size_t RecordsToRead(size_t mbStartSample, bool tail=false);
    void ReleaseMemory();
    void WriteLabelFile();
    void LoadLabelFile(const std::wstring &filePath, std::vector<LabelType>& retLabels);

    LabelIdType GetIdFromLabel(const LabelType& label, LabelInfo& labelInfo);
    bool GetIdFromLabel(const vector<LabelIdType>& label, vector<LabelIdType>& val);
    bool CheckIdFromLabel(const LabelType& labelValue, const LabelInfo& labelInfo, unsigned & labelId);

    bool SentenceEnd();

public:
    void Init(const ConfigParameters& ){};
    void ReadLabelInfo(const wstring & vocfile,  map<LabelType, LabelIdType> & word4idx,
        map<LabelIdType, LabelType>& idx4word);
    void ChangeMaping(const map<LabelType, LabelType>& maplist,
        const LabelType& unkstr,
        map<LabelType, LabelIdType> & word4idx);

    void Destroy() {};

    LUSequenceReader() {
        m_featuresBuffer=NULL; m_labelsBuffer=NULL; m_clsinfoRead = false; m_idx2clsRead = false; 
    }
    ~LUSequenceReader(){};
    void StartMinibatchLoop(size_t , size_t , size_t = requestDataSize) {};

    void SetNbrSlicesEachRecurrentIter(const size_t /*mz*/) {};
    void SentenceEnd(std::vector<size_t> &/*sentenceEnd*/) {};

    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, typename LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);

public:
    int GetSentenceEndIdFromOutputLabel();
};

template<class ElemType>
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
	using LUSequenceReader<ElemType>::labelInfoMin;
	using LUSequenceReader<ElemType>::labelInfoMax;
	using LUSequenceReader<ElemType>::m_featureDim;
	using LUSequenceReader<ElemType>::m_labelInfo;
//	using LUSequenceReader<ElemType>::m_labelInfoIn;
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
//	using LUSequenceReader<ElemType>::arrayLabels;
	using LUSequenceReader<ElemType>::m_readerConfig;
	using LUSequenceReader<ElemType>::m_featuresBuffer;
	using LUSequenceReader<ElemType>::m_labelsBuffer;
	using LUSequenceReader<ElemType>::m_labelsIdBuffer;
	using LUSequenceReader<ElemType>::m_mbSize;
	using LUSequenceReader<ElemType>::m_epochSize;
	using LUSequenceReader<ElemType>::m_featureData;
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
	using LUSequenceReader<ElemType>::ReadLabelInfo;
	using LUSequenceReader<ElemType>::mRandomize;
	using LUSequenceReader<ElemType>::m_seed;
    using LUSequenceReader<ElemType>::mTotalSentenceSofar;
    using LUSequenceReader<ElemType>::GetSentenceEndIdFromOutputLabel;
private:
    size_t mLastProcssedSentenceId ; 
    size_t mBlgSize; 
    size_t mPosInSentence;
    vector<size_t> mToProcess;
    size_t mLastPosInSentence; 
    size_t mNumRead ;

    std::vector<vector<LabelIdType>>  m_featureTemp;
    std::vector<LabelIdType> m_labelTemp;

    bool   mSentenceEnd; 
    bool   mSentenceBegin;

public:
    vector<bool> mProcessed; 
    LUBatchLUSequenceParser<ElemType, LabelType> m_parser;
    BatchLUSequenceReader() {
        mLastProcssedSentenceId  = 0;
        mBlgSize = 1;
        mLastPosInSentence = 0;
        mNumRead = 0;
        mSentenceEnd = false; 
        mSentenceBegin = true; 
        mIgnoreSentenceBeginTag = false;
    }

    ~BatchLUSequenceReader() {
        if (m_labelTemp.size() > 0)
            m_labelTemp.clear();
        if (m_featureTemp.size() > 0)
            m_featureTemp.clear();
    };
   
    void Init(const ConfigParameters& readerConfig);
    void Reset();

    /// return length of sentences size
    size_t FindNextSentences(size_t numSentences); 
    bool   DataEnd(EndDataType endDataType);
    void   SetSentenceEnd(int wrd, int pos, int actualMbSize);
    void   SetSentenceBegin(int wrd, int pos, int actualMbSize);
    void   SetSentenceBegin(int wrd, size_t pos, size_t actualMbSize) { SetSentenceBegin(wrd, (int)pos, (int)actualMbSize); }   // TODO: clean this up
    void   SetSentenceEnd(int wrd, size_t pos, size_t actualMbSize) { SetSentenceEnd(wrd, (int)pos, (int)actualMbSize); }
    void   SetSentenceBegin(size_t wrd, size_t pos, size_t actualMbSize) { SetSentenceBegin((int)wrd, (int)pos, (int)actualMbSize); }
    void   SetSentenceEnd(size_t wrd, size_t pos, size_t actualMbSize) { SetSentenceEnd((int)wrd, (int)pos, (int)actualMbSize); }

    size_t GetLabelOutput(std::map<std::wstring, Matrix<ElemType>*>& matrices, size_t actualmbsize);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);

    bool EnsureDataAvailable(size_t mbStartSample);
    size_t NumberSlicesInEachRecurrentIter();
    void SetNbrSlicesEachRecurrentIter(const size_t mz);

    void SetSentenceSegBatch(Matrix<ElemType>& sentenceBegin);

public:
    void LoadWordMapping(const ConfigParameters& readerConfig);
    bool CanReadFor(wstring nodeName);  /// return true if this reader can output for a node with name nodeName

    vector<size_t> ReturnToProcessId() { return mToProcess; }
    void SetToProcessId(const vector<size_t>& tp) { mToProcess = tp; }

    void SetRandomSeed(int seed) 
    {
        m_seed = seed;
    }

public:
    /**
    for sequential reading data, useful for beam search decoding
    */
    /// this is for frame-by-frame reading of data.
    /// data is first read into these matrices and then if needed is column-by-column retrieved
    map<wstring, Matrix<ElemType>> mMatrices;
    bool GetFrame(std::map<std::wstring, Matrix<ElemType>*>& matrices, const size_t tidx, vector<size_t>& history);

    /// create proposals
    void InitProposals(map<wstring, Matrix<ElemType>*>& pMat);

public:
    bool mbEncodingForDecoding;

    bool mEqualLengthOutput;
    bool mAllowMultPassData;

    /// return length of sentences size
    vector<size_t> mSentenceLength;
    size_t mMaxSentenceLength;
    vector<int> mSentenceBeginAt;
    vector<int> mSentenceEndAt;
    
    /// a matrix of n_stream x n_length
    /// n_stream is the number of streams
    /// n_length is the maximum lenght of each stream
    /// for example, two sentences used in parallel in one minibatch would be
    /// [2 x 5] if the max length of one of the sentences is 5
    /// the elements of the matrix is 0, 1, or -1, defined as SENTENCE_BEGIN, SENTENCE_MIDDLE, NO_OBSERVATION in cbasetype.h 
    /// 0 1 1 0 1
    /// 1 0 1 0 0 
    /// for two parallel data streams. The first has two sentences, with 0 indicating begining of a sentence
    /// the second data stream has two sentences, with 0 indicating begining of sentences
    /// you may use 1 even if a sentence begins at that position, in this case, the trainer will carry over hidden states to the following
    /// frame. 
    Matrix<ElemType> mtSentenceBegin;

    /// by default it is false
    /// if true, reader will set to SENTENCE_MIDDLE for time positions that are orignally correspond to SENTENCE_BEGIN
    /// set to true so that a current minibatch can uses state activities from the previous minibatch. 
    /// default will have truncated BPTT, which only does BPTT inside a minibatch
    bool mIgnoreSentenceBeginTag;
};

template<class ElemType>
class MultiIOBatchLUSequenceReader : public BatchLUSequenceReader<ElemType>
{
private:
    map<wstring, BatchLUSequenceReader<ElemType>*> mReader;

    bool   mCheckDictionaryKeys;
    std::map<std::wstring, BatchLUSequenceReader<ElemType>*> nameToReader;
public:
    MultiIOBatchLUSequenceReader() {
        mCheckDictionaryKeys = true;
        nameToReader.clear();
    }

    ~MultiIOBatchLUSequenceReader() {
        for (map<wstring, BatchLUSequenceReader<ElemType>*>::iterator p = mReader.begin(); p != mReader.end(); p++)
        {
            delete[] p->second;
        }
    };


    bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples);

    void SetSentenceSegBatch(Matrix<ElemType> & sentenceBegin);

    size_t NumberSlicesInEachRecurrentIter();

    void Init(const ConfigParameters& readerConfig);

public:
    void SetRandomSeed(int);

public:
    int GetSentenceEndIdFromOutputLabel();
    bool DataEnd(EndDataType endDataType);

    /// create proposals
    void InitProposals(map<wstring, Matrix<ElemType>*>& pMat);
    bool GetProposalObs(std::map<std::wstring, Matrix<ElemType>*>& matrices, const size_t tidx, vector<size_t>& history);

};


}}}
