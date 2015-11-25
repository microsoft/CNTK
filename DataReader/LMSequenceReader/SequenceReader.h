//
// <copyright file="SequenceReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// SequenceReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
//#define LEAKDETECT

#include "DataReader.h"
#include "DataWriter.h"
#include "commandArgUtil.h"
#include "SequenceParser.h"
#include "RandomOrdering.h"
#include <string>
#include <map>
#include <vector>
#include <random>

namespace Microsoft { namespace MSR { namespace CNTK {

#define CACHE_BLOG_SIZE 50000

#define STRIDX2CLS L"idx2cls"
#define CLASSINFO  L"classinfo"

#define STRIDX2PROB L"idx2prob"
#define MAX_STRING  2048

enum LabelKind
{
    labelNone = 0,  // no labels to worry about
    labelCategory = 1, // category labels, creates mapping tables
    labelNextWord = 2,  // sentence mapping (predicts next word)
    labelOther = 3, // some other type of label
};
            
enum ReaderMode
{
    Softmax = 0,  // no labels to worry about
    Class = 1, // category labels, creates mapping tables
    NCE = 2,  // sentence mapping (predicts next word)
    Unnormalize = 3,
    None = 4, // some other type of label
};

template <typename Count>
class noiseSampler {
    std::vector<double> m_prob, m_log_prob;
    std::uniform_int_distribution<Count> unif_int;
    bool uniform_sampling;
    double uniform_prob;
    double uniform_log_prob;
    std::piecewise_constant_distribution<double> d;
    std::mt19937 rng;
public:
    noiseSampler(){ }
    noiseSampler(const std::vector<double> &counts, bool xuniform_sampling = false)
        :uniform_sampling(xuniform_sampling), rng(1234)
    {
        size_t k = counts.size();
        uniform_prob = 1.0 / k;
        uniform_log_prob = std::log(uniform_prob);
        std::vector<double> vn(counts.size() + 1);
        for (int i = 0; i < vn.size(); i++)
            vn[i] = i;
        d = std::piecewise_constant_distribution<double>(vn.begin(), vn.end(), counts.begin());
        unif_int = std::uniform_int_distribution<Count>(0,(long) counts.size() - 1);
        m_prob = d.densities();
        m_log_prob.resize(m_prob.size());
        for (int i = 0; i < k; i++)
            m_log_prob[i] = std::log(m_prob[i]);
    }
    int size() const{ return m_prob.size(); }
    double prob(int i) const { if (uniform_sampling) return uniform_prob; else return m_prob[i]; }
    double logprob(int i) const { if (uniform_sampling) return uniform_log_prob; else return m_log_prob[i]; }

    template <typename Engine>
    int sample(Engine &eng)
    {
        int m = unif_int(eng);
        if (uniform_sampling)
            return m;
        return (int)d(eng);
    }
    
    int sample()
    {
        return sample(this->rng);
    }
};

template<class ElemType>
class SequenceReader : public IDataReader<ElemType>
{
protected:
    bool   m_idx2clsRead; 
    bool   m_clsinfoRead;

    bool   m_idx2probRead;
    std::wstring m_file; 
public:
    using LabelType   = typename IDataReader<ElemType>::LabelType;
    using LabelIdType = typename IDataReader<ElemType>::LabelIdType;

    map<string, int> word4idx;
    map<int, string> idx4word;
    map<int, int> idx4class;
    map<int, size_t> idx4cnt;
    int nwords, dims, nsamps, nglen, nmefeats;
    Matrix<ElemType>* m_id2classLocal; // CPU version
    Matrix<ElemType>* m_classInfoLocal; // CPU version
    
    Matrix<ElemType>* m_id2Prob; // CPU version
    int class_size;
    map<int, vector<int>> class_words;

    int noise_sample_size;
    noiseSampler<long> m_noiseSampler;

    ReaderMode readerMode;
    int eos_idx, unk_idx;

    string mUnk; /// unk symbol

public:
//    typedef std::string LabelType;
//    typedef unsigned LabelIdType;
protected:
//    SequenceParser<ElemType, LabelType> m_parser;
    LMSequenceParser<ElemType, LabelType> m_parser;
//    LMBatchSequenceParser<ElemType, LabelType> m_parser;
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

    size_t* m_featuresBufferRow;
    size_t* m_featuresBufferRowIdx;


    CPUSPARSE_INDEX_TYPE* m_labelsIdBufferRow;
    size_t* m_labelsBlock2Id;
    size_t* m_labelsBlock2UniqId;

    bool m_endReached;
    int m_traceLevel;
   
    // feature and label data are parallel arrays
    std::vector<ElemType> m_featureData;
    std::vector<LabelIdType> m_labelIdData;
    std::vector<ElemType> m_labelData;    
    std::vector<size_t> m_sequence;
    std::map<size_t, size_t> m_indexer; // feature or label indexer

    // we have two one for input and one for output
    struct LabelInfo
    {
        LabelKind type;  // labels are categories, create mapping table
        std::map<LabelIdType, LabelType> mapIdToLabel;
        std::map<LabelType, LabelIdType> mapLabelToId;
        LabelIdType idMax; // maximum label ID we have encountered so far
        LabelIdType dim; // maximum label ID we will ever see (used for array dimensions)
        std::string beginSequence; // starting sequence string (i.e. <s>)
        std::string endSequence; // ending sequence string (i.e. </s>)
        std::wstring mapName;
        std::wstring fileToWrite;  // set to the path if we need to write out the label file
    } m_labelInfo[labelInfoMax];

    // caching support
    DataReader<ElemType>* m_cachingReader;
    DataWriter<ElemType>* m_cachingWriter;
    ConfigParameters m_readerConfig;
    void InitCache(const ConfigParameters& config);

    void UpdateDataVariables();
    void SetupEpoch();
    void LMSetupEpoch();
    size_t RecordsToRead(size_t mbStartSample, bool tail=false);
    void ReleaseMemory();
    void WriteLabelFile();
    void LoadLabelFile(const std::wstring &filePath, std::vector<LabelType>& retLabels);

    LabelIdType GetIdFromLabel(const std::string& label, LabelInfo& labelInfo);
    bool CheckIdFromLabel(const std::string& labelValue, const LabelInfo& labelInfo, unsigned & labelId);

    virtual bool EnsureDataAvailable(size_t mbStartSample, bool endOfDataCheck=false);
    virtual bool ReadRecord(size_t readSample);
    bool SentenceEnd();

public:
    template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);
    virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
    virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }
    static void ReadClassInfo(const wstring & vocfile, int& class_size,
        map<string, int>& word4idx,
        map<int, string>& idx4word,
        map<int, int>& idx4class,
        map<int, size_t> & idx4cnt, 
        int nwords,
        string mUnk,
        noiseSampler<long>& m_noiseSampler,
        bool flatten);
    static void ReadWord(char *wrod, FILE *fin);

    void GetLabelOutput(std::map<std::wstring, Matrix<ElemType>*>& matrices, 
                       size_t m_mbStartSample, size_t actualmbsize);
    void GetInputToClass(std::map<std::wstring, Matrix<ElemType>*>& matrices);

    void GetInputProb(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    void GetClassInfo();

    virtual void Destroy();
    SequenceReader() {
        m_featuresBuffer=NULL; m_labelsBuffer=NULL; m_clsinfoRead = false; m_idx2clsRead = false;             
        m_cachingReader=NULL; m_cachingWriter=NULL; m_labelsIdBuffer = NULL;
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
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);

    //void SetSentenceSegBatch(std::vector<size_t> &/*sentenceEnd*/) {};
    // TODO: ^^ should this be   void CopyMBLayoutTo(MBLayoutPtr pMBLayout);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart=0);
    
    virtual bool DataEnd(EndDataType endDataType);

    int GetSentenceEndIdFromOutputLabel(){ return -1; };

};

template<class ElemType>
class BatchSequenceReader : public SequenceReader<ElemType>
{
public:
    using LabelType = typename SequenceReader<ElemType>::LabelType;
    using LabelIdType = typename SequenceReader<ElemType>::LabelIdType;
    using LabelInfo = typename SequenceReader<ElemType>::LabelInfo;
    using SequenceReader<ElemType>::m_cachingReader;
    using SequenceReader<ElemType>::m_cachingWriter;
    using SequenceReader<ElemType>::m_featuresName;
    using SequenceReader<ElemType>::labelInfoMin;
    using SequenceReader<ElemType>::labelInfoMax;
    using SequenceReader<ElemType>::m_labelsName;
    using SequenceReader<ElemType>::m_featureDim;
    using SequenceReader<ElemType>::class_size;
    using SequenceReader<ElemType>::m_labelInfo;
    using SequenceReader<ElemType>::labelInfoIn;
    using SequenceReader<ElemType>::nwords;
    using SequenceReader<ElemType>::ReadClassInfo;
    using SequenceReader<ElemType>::LoadLabelFile;
    using SequenceReader<ElemType>::word4idx;
    using SequenceReader<ElemType>::idx4word;
    using SequenceReader<ElemType>::idx4cnt;
    using SequenceReader<ElemType>::mUnk;
    using SequenceReader<ElemType>::m_mbStartSample;
    using SequenceReader<ElemType>::m_epoch;
    using SequenceReader<ElemType>::m_totalSamples;
    using SequenceReader<ElemType>::m_epochStartSample;
    using SequenceReader<ElemType>::m_seqIndex;
    using SequenceReader<ElemType>::m_readNextSampleLine;
    using SequenceReader<ElemType>::m_readNextSample;
    using SequenceReader<ElemType>::m_traceLevel;
    using SequenceReader<ElemType>::m_featureCount;
    using SequenceReader<ElemType>::m_endReached;
    //	using IDataReader<ElemType>::labelIn;
    //	using IDataReader<ElemType>::labelOut;
    using SequenceReader<ElemType>::InitCache;
    using SequenceReader<ElemType>::m_readerConfig;
    using SequenceReader<ElemType>::ReleaseMemory;
    using SequenceReader<ElemType>::m_featuresBuffer;
    using SequenceReader<ElemType>::m_featuresBufferRow;
    using SequenceReader<ElemType>::m_labelsBuffer;
    using SequenceReader<ElemType>::m_labelsIdBuffer;
    //	using IDataReader<ElemType>::labelInfo;
    //	using SequenceReader<ElemType>::m_featuresBufferRowIndex;
    using SequenceReader<ElemType>::m_labelsIdBufferRow;
    using SequenceReader<ElemType>::m_labelsBlock2Id;
    using SequenceReader<ElemType>::m_labelsBlock2UniqId;
    using SequenceReader<ElemType>::m_id2classLocal;
    using SequenceReader<ElemType>::m_classInfoLocal;
    using SequenceReader<ElemType>::m_mbSize;
    using SequenceReader<ElemType>::m_epochSize;
    using SequenceReader<ElemType>::m_featureData;
    using SequenceReader<ElemType>::labelInfoOut;
    using SequenceReader<ElemType>::m_labelData;
    using SequenceReader<ElemType>::m_labelIdData;
    using SequenceReader<ElemType>::LMSetupEpoch;
    using SequenceReader<ElemType>::m_clsinfoRead;
    using SequenceReader<ElemType>::m_idx2clsRead;
    using SequenceReader<ElemType>::m_featuresBufferRowIdx;
    using SequenceReader<ElemType>::m_sequence;
    using SequenceReader<ElemType>::idx4class;
    using SequenceReader<ElemType>::m_indexer;
    using SequenceReader<ElemType>::m_noiseSampler;
    using SequenceReader<ElemType>::readerMode;
    using SequenceReader<ElemType>::GetIdFromLabel;
    using SequenceReader<ElemType>::GetInputToClass;
    using SequenceReader<ElemType>::GetClassInfo;
    using IDataReader<ElemType>::mBlgSize;

private:
    size_t mLastProcssedSentenceId ; 

    size_t mPosInSentence;
    vector<size_t> mToProcess;
    size_t mLastPosInSentence; 
    size_t mNumRead ;

    std::vector<ElemType>  m_featureTemp;
    std::vector<LabelType> m_labelTemp;

    bool   mSentenceEnd; 
    bool   mSentenceBegin; 

    MBLayoutPtr m_pMBLayout;

public:
    vector<bool> mProcessed; 
    LMBatchSequenceParser<ElemType, LabelType> m_parser;
    BatchSequenceReader() : m_pMBLayout(make_shared<MBLayout>())
    {
        mLastProcssedSentenceId  = 0;
        mBlgSize = 1;
        mLastPosInSentence = 0;
        mNumRead = 0;
        mSentenceEnd = false; 
    }
    ~BatchSequenceReader() {
        if (m_labelTemp.size() > 0)
            m_labelTemp.clear();
        if (m_featureTemp.size() > 0)
            m_featureTemp.clear();
    };
   
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
    void GetLabelOutput(std::map<std::wstring, Matrix<ElemType>*>& matrices,
                       size_t m_mbStartSample, size_t actualmbsize);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    bool EnsureDataAvailable(size_t mbStartSample);
    size_t GetNumParallelSequences();

    void SetSentenceSegBatch(std::vector<size_t> &sentenceEnd);
    void CopyMBLayoutTo(MBLayoutPtr);
    bool RequireSentenceSeg() const override { return true; }

    int GetSentenceEndIdFromOutputLabel();
};

}}}
