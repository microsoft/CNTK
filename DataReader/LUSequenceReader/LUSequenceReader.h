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

#define CACHE_BLOG_SIZE 50000

#define STRIDX2CLS L"idx2cls"
#define CLASSINFO  L"classinfo"
    
#define MAX_STRING  2048

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
    int nwords, dims, nsamps, nglen, nmefeats;

    int class_size;
    map<int, vector<int>> class_words;
    vector<int>class_cn;

public:
    /// deal with OOV
    map<string, string> mWordMapping;
    string mWordMappingFn;
    string mUnkStr;

protected:

    LULUSequenceParser<ElemType, LabelType> m_parser;
//    LUBatchLUSequenceParser<ElemType, LabelType> m_parser;
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
        map<string, int> word4idx;
        map<int, string> idx4word;
        LabelIdType idMax; // maximum label ID we have encountered so far
        LabelIdType dim; // maximum label ID we will ever see (used for array dimensions)
        std::string beginSequence; // starting sequence string (i.e. <s>)
        std::string endSequence; // ending sequence string (i.e. </s>)
        bool busewordmap; /// whether using wordmap to map unseen words to unk
        std::wstring mapName;
        std::wstring fileToWrite;  // set to the path if we need to write out the label file
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

    LabelIdType GetIdFromLabel(const std::string& label, LabelInfo& labelInfo);
    bool GetIdFromLabel(const vector<string>& label, LabelInfo& labelInfo, vector<LabelIdType>& val);
    bool CheckIdFromLabel(const std::string& labelValue, const LabelInfo& labelInfo, unsigned & labelId);

    virtual bool ReadRecord(size_t readSample);
    bool SentenceEnd();

public:
    virtual void Init(const ConfigParameters& config);
    void ReadLabelInfo(const wstring & vocfile,  map<string, int> & word4idx,
                                                map<int, string>& idx4word) ;
    void ChangeMaping(const map<string, string>& maplist, 
            const string & unkstr , 
            map<string, int> & word4idx);

    void ReadWord(char *wrod, FILE *fin);

    virtual void Destroy();
    LUSequenceReader() {
        m_featuresBuffer=NULL; m_labelsBuffer=NULL; m_clsinfoRead = false; m_idx2clsRead = false; 
    }
    virtual ~LUSequenceReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);

    void SetNbrSlicesEachRecurrentIter(const size_t mz) {};
	void SentenceEnd(std::vector<size_t> &sentenceEnd) {};
    void SetSentenceEndInBatch(std::vector<size_t> &sentenceEnd) {};

    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, typename LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart=0);
    
};

template<class ElemType>
class BatchLUSequenceReader : public LUSequenceReader<ElemType>
{
private:
    size_t mLastProcssedSentenceId ; 
    size_t mBlgSize; 
    size_t mPosInSentence;
    vector<size_t> mToProcess;
    size_t mLastPosInSentence; 
    size_t mNumRead ;

    std::vector<vector<LabelType>>  m_featureTemp;
    std::vector<LabelType> m_labelTemp;

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

    void GetLabelOutput(std::map<std::wstring, Matrix<ElemType>*>& matrices, 
                       size_t m_mbStartSample, size_t actualmbsize);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    bool EnsureDataAvailable(size_t mbStartSample);
    size_t NumberSlicesInEachRecurrentIter();
    void SetNbrSlicesEachRecurrentIter(const size_t mz);

    void SetSentenceEndInBatch(std::vector<size_t> &sentenceEnd);

public:
    void LoadWordMapping(const ConfigParameters& readerConfig);


};

}}}