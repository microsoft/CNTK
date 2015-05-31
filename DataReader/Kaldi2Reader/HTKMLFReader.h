//
// <copyright file="HTKMLFReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
#include "DataReader.h"
#include "KaldiSequenceTrainingIO.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
class HTKMLFReader : public IDataReader<ElemType>
{
private:
    msra::dbn::minibatchiterator* m_mbiter;
    msra::dbn::minibatchsource* m_frameSource;
    vector<msra::asr::FeatureSection *> m_trainingOrTestingFeatureSections;
    //msra::dbn::minibatchreadaheadsource* m_readAheadSource;
    msra::dbn::FileEvalSource* m_fileEvalSource; 
    vector<msra::asr::FeatureSection *> m_writingFeatureSections;
    msra::dbn::latticesource* m_lattices;
    map<wstring,msra::lattices::lattice::htkmlfwordsequence> m_latticeMap;

    // Sequence training related. Note that for now we only support single
    // utterance in sequence training. But the utterance information holders
    // are designed as if they support multiple utterances -- in case we will
    // extend this soon.
    bool m_doSeqTrain;
    wstring m_seqTrainCriterion;
    KaldiSequenceTrainingIO<ElemType>* m_sequenceTrainingIO;
    std::vector<std::vector<std::pair<wstring, size_t>>> m_uttInfo;
    //-std::vector<size_t> m_uttInfoCurrentIndex;
    //-std::vector<size_t> m_uttInfoCurrentLength;
    
    vector<bool> m_sentenceEnd;
    bool m_readAhead;
    bool m_truncated;
    bool m_framemode;
    bool m_noMix;
    vector<size_t> m_processedFrame;
    size_t m_numberOfuttsPerMinibatch;
    size_t m_actualnumberOfuttsPerMinibatch;
    size_t m_mbSize;
    vector<size_t> m_currentBufferFrames;
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
    void PrepareForSequenceTraining(const ConfigParameters& config);
    
    bool GetMinibatchToTrainOrTest(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    bool GetMinibatchToWrite(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    bool PopulateUtteranceInMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices, size_t uttIndex, size_t startFrame, size_t endFrame, size_t mbSize);

    //-void GetCurrentUtteranceInfo(size_t uttIndex, size_t startFrame, size_t endFrame, wstring& uttID, size_t& startFrameInUtt, size_t& endFrameInUtt);

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
        seqTrainDeriv, /*sequence training derivative, computed in the reader*/
        seqTrainObj,   /*sequence training objective, computed in the reader*/
    };



public:
    virtual void Init(const ConfigParameters& config);
    virtual void Destroy() {delete this;}
    virtual ~HTKMLFReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart=0);

    virtual bool GetForkedUtterance(std::wstring& uttID, std::map<std::wstring, Matrix<ElemType>*>& matrices);
    virtual bool ComputeDerivativeFeatures(const std::wstring& uttID, const Matrix<ElemType>& outputs);
    

    virtual bool DataEnd(EndDataType endDataType);
    void SetSentenceEndInBatch(vector<size_t> &/*sentenceEnd*/);
    void SetSentenceEnd(int /*actualMbSize*/){};
};

}}}
