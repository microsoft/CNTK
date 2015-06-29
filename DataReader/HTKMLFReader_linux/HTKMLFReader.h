//
// <copyright file="HTKMLFReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
class HTKMLFReader : public IDataReader<ElemType>
{
private:
    msra::dbn::minibatchiterator* m_mbiter;
    msra::dbn::minibatchsource* m_frameSource;
    //msra::dbn::minibatchreadaheadsource* m_readAheadSource;
     msra::dbn::FileEvalSource* m_fileEvalSource; 
    msra::dbn::latticesource* m_lattices;
    map<wstring,msra::lattices::lattice::htkmlfwordsequence> m_latticeMap;
    
    vector<bool> m_sentenceEnd;
    bool m_readAhead;
    bool m_truncated;
    vector<size_t> m_processedFrame;
    size_t m_numberOfuttsPerMinibatch;
    size_t m_actualnumberOfuttsPerMinibatch;
    size_t m_mbSize;
    vector<size_t> m_toProcess;
    vector<size_t> m_switchFrame;
    bool m_noData;

    bool m_trainOrTest; // if false, in file writing mode
	using LabelType = typename IDataReader<ElemType>::LabelType;
	using LabelIdType = typename IDataReader<ElemType>::LabelIdType;
 
    std::map<LabelIdType, LabelType> m_idToLabelMap;
    
    bool m_partialMinibatch; // allow partial minibatches?
    
    std::vector<ElemType*> m_featuresBufferMultiUtt;
    std::vector<size_t> m_featuresBufferAllocatedMultiUtt;
    std::vector<ElemType*> m_labelsBufferMultiUtt;
    std::vector<size_t> m_labelsBufferAllocatedMultiUtt;
    std::vector<size_t> m_featuresStartIndexMultiUtt;
    std::vector<size_t> m_labelsStartIndexMultiUtt;

    std::vector<ElemType*> m_featuresBufferMultiIO;
    std::vector<size_t> m_featuresBufferAllocatedMultiIO;
    std::vector<ElemType*> m_labelsBufferMultiIO;
    std::vector<size_t> m_labelsBufferAllocatedMultiIO;

    std::map<std::wstring,size_t> m_featureNameToIdMap;
    std::map<std::wstring,size_t> m_labelNameToIdMap;
    std::map<std::wstring,size_t> m_nameToTypeMap;
    std::map<std::wstring,size_t> m_featureNameToDimMap;
    std::map<std::wstring,size_t> m_labelNameToDimMap;
    // for writing outputs to files (standard single input/output network) - deprecate eventually
    bool m_checkDictionaryKeys;
    bool m_convertLabelsToTargets;
    std::vector <bool> m_convertLabelsToTargetsMultiIO;
    std::vector<std::vector<std::wstring>> m_inputFilesMultiIO;
 
    size_t m_inputFileIndex;
    std::vector<size_t> m_featDims;
    std::vector<size_t> m_labelDims;

    std::vector<std::vector<std::vector<ElemType>>>m_labelToTargetMapMultiIO;
     
    void PrepareForTrainingOrTesting(const ConfigParameters& config);
    void PrepareForWriting(const ConfigParameters& config);
    
    bool GetMinibatchToTrainOrTest(std::map<std::wstring, Matrix<ElemType>*>&matrices);
    bool GetMinibatchToWrite(std::map<std::wstring, Matrix<ElemType>*>&matrices);
    
    void StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    void StartMinibatchLoopToWrite(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);

    bool ReNewBufferForMultiIO(size_t i);

    size_t NumberSlicesInEachRecurrentIter() { return m_numberOfuttsPerMinibatch ;} 
    void SetNbrSlicesEachRecurrentIter(const size_t) { };

     void GetDataNamesFromConfig(const ConfigParameters& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels);

    
    size_t ReadLabelToTargetMappingFile (const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap);
    enum InputOutputTypes
    {
        real,
        category,
    };



public:
    /// a matrix of n_stream x n_length
    /// n_stream is the number of streams
    /// n_length is the maximum lenght of each stream
    /// for example, two sentences used in parallel in one minibatch would be
    /// [2 x 5] if the max length of one of the sentences is 5
    /// the elements of the matrix is 0, 1, or -1, defined as SENTENCE_BEGIN, SENTENCE_MIDDLE, NO_LABELS in cbasetype.h 
    /// 0 1 1 0 1
    /// 1 0 1 0 0 
    /// for two parallel data streams. The first has two sentences, with 0 indicating begining of a sentence
    /// the second data stream has two sentences, with 0 indicating begining of sentences
    /// you may use 1 even if a sentence begins at that position, in this case, the trainer will carry over hidden states to the following
    /// frame. 
	Matrix<ElemType> m_sentenceBegin;

    /// a matrix of 1 x n_length
    /// 1 denotes the case that there exists sentnece begin or no_labels case in this frame
    /// 0 denotes such case is not in this frame


	vector<MinibatchPackingFlag> m_minibatchPackingFlag;

    /// by default it is false
    /// if true, reader will set to SENTENCE_MIDDLE for time positions that are orignally correspond to SENTENCE_BEGIN
    /// set to true so that a current minibatch can uses state activities from the previous minibatch. 
    /// default will have truncated BPTT, which only does BPTT inside a minibatch

	bool mIgnoreSentenceBeginTag;
	HTKMLFReader() : m_sentenceBegin(CPUDEVICE) {
	}

    virtual void Init(const ConfigParameters& config);
    virtual void Destroy() {delete this;}
    virtual ~HTKMLFReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart=0);

    virtual bool DataEnd(EndDataType endDataType);
    void SetSentenceEndInBatch(vector<size_t> &/*sentenceEnd*/);
    void SetSentenceEnd(int /*actualMbSize*/){};
    void SetSentenceSegBatch(Matrix<ElemType> &sentenceBegin, vector<MinibatchPackingFlag>& sentenceExistsBeginOrNoLabels);

};

}}}
