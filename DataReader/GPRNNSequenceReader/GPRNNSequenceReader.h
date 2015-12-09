//
// <copyright file="GPRNNSequenceReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// GPRNNSequenceReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
//#define LEAKDETECT

#include "Basics.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "GPRNNSequenceParser.h"
#include "Config.h" // for intargvector
#include "ScriptableObjects.h"
#include <string>
#include <map>
#include <vector>


namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef DBG_SMT
#define CACHE_BLOG_SIZE 2
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

enum ReaderMode
{
    Plain = 0,  // no class info
    Class = 1, // category labels, creates mapping tables
};

template<class ElemType>
class GPRNNSequenceReader : public IDataReader<ElemType>
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
    wstring mWordMappingFn;
    LabelType mUnkStr;

public:
    /// accumulated number of sentneces read so far
    unsigned long mTotalSentenceSofar;

protected:

    BatchGPRNNSequenceParser<ElemType, LabelType> m_parser;
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
	std::wstring m_auxfeaturesName;
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
	std::vector<std::vector<std::pair<LabelIdType, LabelIdType> > > m_auxfeatureData;
    std::vector<LabelIdType> m_labelIdData;
    std::vector<ElemType> m_labelData;
    std::vector<size_t> m_sequence;

    // we have two one for input and one for output
    struct LabelInfo
    {
        LabelKind type;  // labels are categories, create mapping table
        map<LabelType, LabelIdType> word4idx;
        map<LabelIdType, LabelType> idx4word;
        LabelIdType idMax; // maximum label ID we have encountered so far
        long dim; // maximum label ID we will ever see (used for array dimensions)
		long auxdim;
        LabelType beginSequence; // starting sequence string (i.e. <s>)
        LabelType endSequence; // ending sequence string (i.e. </s>)
        bool busewordmap; /// whether using wordmap to map unseen words to unk
        std::wstring mapName;
        std::wstring fileToWrite;  // set to the path if we need to write out the label file

        bool isproposal; /// whether this is for proposal generation

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
        Matrix<ElemType>* m_id2classLocal; // CPU version
        Matrix<ElemType>* m_classInfoLocal; // CPU version
        int  mNbrClasses;
        bool m_clsinfoRead; 
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
    void Init(const ScriptableObjects::IConfigRecord&){};
    void ChangeMaping(const map<LabelType, LabelType>& maplist,
        const LabelType& unkstr,
        map<LabelType, LabelIdType> & word4idx);

    void Destroy() {};

    GPRNNSequenceReader() {
        m_featuresBuffer=NULL; m_labelsBuffer=NULL; m_clsinfoRead = false; m_idx2clsRead = false; 
    }
    ~GPRNNSequenceReader(){};
    void StartMinibatchLoop(size_t , size_t , size_t = requestDataSize) {};

    void SetNumParallelSequences(const size_t /*mz*/) {};
    void SentenceEnd(std::vector<size_t> &/*sentenceEnd*/) {};

    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);

public:
    int GetSentenceEndIdFromOutputLabel();
};

template<class ElemType>
class BatchGPRNNSequenceReader : public GPRNNSequenceReader<ElemType>
{
public:
    using LabelType = wstring;
    using LabelIdType = long;
    using GPRNNSequenceReader<ElemType>::mWordMappingFn;
    using GPRNNSequenceReader<ElemType>::m_cachingReader;
    using GPRNNSequenceReader<ElemType>::mWordMapping;
    using GPRNNSequenceReader<ElemType>::mUnkStr;
    using GPRNNSequenceReader<ElemType>::m_cachingWriter;
    using GPRNNSequenceReader<ElemType>::m_featuresName;
    using GPRNNSequenceReader<ElemType>::m_labelsName;
    using GPRNNSequenceReader<ElemType>::labelInfoMin;
    using GPRNNSequenceReader<ElemType>::labelInfoMax;
    using GPRNNSequenceReader<ElemType>::m_featureDim;
    using GPRNNSequenceReader<ElemType>::m_labelInfo;
//  using GPRNNSequenceReader<ElemType>::m_labelInfoIn;
    using GPRNNSequenceReader<ElemType>::m_mbStartSample;
    using GPRNNSequenceReader<ElemType>::m_epoch;
    using GPRNNSequenceReader<ElemType>::m_totalSamples;
    using GPRNNSequenceReader<ElemType>::m_epochStartSample;
    using GPRNNSequenceReader<ElemType>::m_seqIndex;
    using GPRNNSequenceReader<ElemType>::m_endReached;
    using GPRNNSequenceReader<ElemType>::m_readNextSampleLine;
    using GPRNNSequenceReader<ElemType>::m_readNextSample;
    using GPRNNSequenceReader<ElemType>::m_traceLevel;
    using GPRNNSequenceReader<ElemType>::m_wordContext;
    using GPRNNSequenceReader<ElemType>::m_featureCount;
    using typename GPRNNSequenceReader<ElemType>::LabelInfo;
    using GPRNNSequenceReader<ElemType>::labelInfoIn;
    using GPRNNSequenceReader<ElemType>::labelInfoOut;
//  using GPRNNSequenceReader<ElemType>::arrayLabels;
    using GPRNNSequenceReader<ElemType>::m_readerConfig;
    using GPRNNSequenceReader<ElemType>::m_featuresBuffer;
    using GPRNNSequenceReader<ElemType>::m_labelsBuffer;
    using GPRNNSequenceReader<ElemType>::m_labelsIdBuffer;
    using GPRNNSequenceReader<ElemType>::m_mbSize;
    using GPRNNSequenceReader<ElemType>::m_epochSize;
    using GPRNNSequenceReader<ElemType>::m_featureData;
    using GPRNNSequenceReader<ElemType>::m_sequence;
    using GPRNNSequenceReader<ElemType>::m_labelData;
    using GPRNNSequenceReader<ElemType>::m_labelIdData;
    using GPRNNSequenceReader<ElemType>::m_idx2clsRead;
    using GPRNNSequenceReader<ElemType>::m_clsinfoRead;
    using GPRNNSequenceReader<ElemType>::m_featureWordContext;
    using GPRNNSequenceReader<ElemType>::LoadLabelFile;
    using GPRNNSequenceReader<ElemType>::ReleaseMemory;
    using GPRNNSequenceReader<ElemType>::LMSetupEpoch;
    using GPRNNSequenceReader<ElemType>::ChangeMaping;
    using GPRNNSequenceReader<ElemType>::GetIdFromLabel;
    using GPRNNSequenceReader<ElemType>::InitCache;
    using GPRNNSequenceReader<ElemType>::mRandomize;
    using GPRNNSequenceReader<ElemType>::m_seed;
    using GPRNNSequenceReader<ElemType>::mTotalSentenceSofar;
    using GPRNNSequenceReader<ElemType>::GetSentenceEndIdFromOutputLabel;
private:
    size_t mLastProcessedSentenceId;
    size_t mRequestedNumParallelSequences; 
    size_t mPosInSentence;
    vector<size_t> mToProcess;      // [seqIndex] utterance id of utterance in this minibatch's position [seqIndex]
    size_t mLastPosInSentence;      // BPTT cursor
    size_t mNumRead ;

	// store the cached input feature & output label
	// the feature contains two parts: first part is contex feature, second feature is auxiliary feature
    std::vector<std::pair<std::vector<long>, std::vector<std::pair<LabelIdType, LabelIdType> > > >  m_featureTemp;
    std::vector<LabelIdType> m_labelTemp;

    bool   mSentenceEnd; 
    bool   mSentenceBegin;

public:
    vector<bool> mProcessed; 
    BatchGPRNNSequenceParser<ElemType, LabelType> m_parser;
    BatchGPRNNSequenceReader() : m_pMBLayout(make_shared<MBLayout>()){
        mLastProcessedSentenceId = 0;
        mRequestedNumParallelSequences = 1;
        mLastPosInSentence = 0;
        mNumRead = 0;
        mSentenceEnd = false; 
        mSentenceBegin = true; 
        mIgnoreSentenceBeginTag = false;
    }

    ~BatchGPRNNSequenceReader();
   
    template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);
    virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
    virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }
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

    size_t GetLabelOutput(std::map<std::wstring,
        Matrix<ElemType>*>& matrices, LabelInfo& labelInfo, size_t actualmbsize);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);

    bool EnsureDataAvailable(size_t mbStartSample);
    size_t GetNumParallelSequences();
    void SetNumParallelSequences(const size_t mz);

    void CopyMBLayoutTo(MBLayoutPtr pMBLayout);

public:
    void GetClassInfo(LabelInfo& lblInfo);
    void ReadLabelInfo(const wstring & vocfile,
        map<wstring, long> & word4idx,
        bool readClass,
        map<wstring, long>& word4cls,
        map<long, wstring>& idx4word,
        map<long, long>& idx4class,
        int & mNbrCls);

    template<class ConfigRecordType>
    void LoadWordMapping(const ConfigRecordType& config);
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

    bool mEqualLengthOutput;
    bool mAllowMultPassData;

    /// return length of sentences size
    vector<size_t> mSentenceLengths;    // [seqIndex] lengths of all sentences in a minibatch
    size_t mMaxSentenceLength;          // max over mSentenceLength[]  --TODO: why not compute on the fly?

	/// return the line number of BoS and EoS.
    vector<int> mSentenceBeginAt;       // [seqIndex] index of first token
    const int NO_INPUT = -2;
    vector<int> mSentenceEndAt;         // [seqIndex] index of last token
    
    /// a matrix of n_stream x n_length
    /// n_stream is the number of streams
    /// n_length is the maximum lenght of each stream
    /// for example, two sentences used in parallel in one minibatch would be
    /// [2 x 5] if the max length of one of the sentences is 5
    /// the elements of the matrix is 0, 1, or -1, defined as SEQUENCE_START, SEQUENCE_MIDDLE, NO_INPUT in cbasetype.h 
    /// 0 1 1 0 1
    /// 1 0 1 0 0 
    /// for two parallel data streams. The first has two sentences, with 0 indicating begining of a sentence
    /// the second data stream has two sentences, with 0 indicating begining of sentences
    /// you may use 1 even if a sentence begins at that position, in this case, the trainer will carry over hidden states to the following
    /// frame.
	/// For LU Slot annotation task, n_stream will be the number of utterence of each minibatch.
    MBLayoutPtr m_pMBLayout;

    // if true, reader will set to ((int) MinibatchPackingFlags::None) for time positions that are orignally correspond to ((int) MinibatchPackingFlags::SequenceStart)
    // set to true so that a current minibatch can uses state activities from the previous minibatch. 
    // default will have truncated BPTT, which only does BPTT inside a minibatch
    // by default it is false
    bool mIgnoreSentenceBeginTag;
};


//// BatchReader support multiple IOFiles. The real work is done by BatchGPRNNSequenceReader.
template<class ElemType>
class MultiIOBatchGPRNNSequenceReader : public BatchGPRNNSequenceReader<ElemType>
{
private:
    map<wstring, BatchGPRNNSequenceReader<ElemType>*> mReader;

    bool   mCheckDictionaryKeys;
    std::map<std::wstring, BatchGPRNNSequenceReader<ElemType>*> nameToReader;
public:
    MultiIOBatchGPRNNSequenceReader() {
        mCheckDictionaryKeys = true;
        nameToReader.clear();
    }

    ~MultiIOBatchGPRNNSequenceReader() {
        for (typename map<wstring, BatchGPRNNSequenceReader<ElemType>*>::iterator p = mReader.begin(); p != mReader.end(); p++)
        {
            delete[] p->second;
        }
    };


    bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples);

    void CopyMBLayoutTo(MBLayoutPtr pMBLayout);

    size_t GetNumParallelSequences();

    template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);
    virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
    virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }

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
