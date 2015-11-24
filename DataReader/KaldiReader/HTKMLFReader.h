//
// <copyright file="HTKMLFReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
#include "DataReader.h"
#include "commandArgUtil.h" // for intargvector

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
    bool m_framemode;
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

    size_t GetNumParallelSequences() { return m_numberOfuttsPerMinibatch; } 
    void SetNumParallelSequences(const size_t) { };

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
    MBLayoutPtr m_pMBLayout;

    /// by default it is false
    /// if true, reader will set to SEQUENCE_MIDDLE for time positions that are orignally correspond to SEQUENCE_START
    /// set to true so that a current minibatch can uses state activities from the previous minibatch. 
    /// default will have truncated BPTT, which only does BPTT inside a minibatch

	bool mIgnoreSentenceBeginTag;
     HTKMLFReader() : m_pMBLayout(make_shared<MBLayout>())
     {
     }

    template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);
    virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
    virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }
    virtual void Destroy() {delete this;}
    virtual ~HTKMLFReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart=0);

    virtual bool DataEnd(EndDataType endDataType);
    void CopyMBLayoutTo(MBLayoutPtr);
    void SetSentenceEndInBatch(vector<size_t> &/*sentenceEnd*/);
    void SetSentenceEnd(int /*actualMbSize*/){};
    void SetRandomSeed(int) { NOT_IMPLEMENTED };

    bool RequireSentenceSeg() { return !m_framemode; }; 
};

}}}
