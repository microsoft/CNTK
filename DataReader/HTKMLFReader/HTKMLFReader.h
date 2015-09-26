//
// <copyright file="HTKMLFReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
#include "DataReader.h"
#include "commandArgUtil.h" // for intargvector
#include "CUDAPageLockedMemAllocator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
class HTKMLFReader : public IDataReader<ElemType>
{
private:

    const static size_t m_htkRandomizeAuto = 0;
    const static size_t m_htkRandomizeDisable = (size_t)-1;

    msra::dbn::minibatchiterator* m_mbiter;
    msra::dbn::minibatchsource* m_frameSource;
#ifdef _WIN32
    msra::dbn::minibatchreadaheadsource* m_readAheadSource;
#endif
    msra::dbn::FileEvalSource* m_fileEvalSource; 
    msra::dbn::latticesource* m_lattices;
    map<wstring,msra::lattices::lattice::htkmlfwordsequence> m_latticeMap;
    
    vector<bool> m_sentenceEnd;
    bool m_readAhead;
    bool m_truncated;
	bool m_fullutt;              //read full utterance every time
    bool m_framemode;
    vector<size_t> m_processedFrame;
    intargvector m_numberOfuttsPerMinibatchForAllEpochs;
    size_t m_numberOfuttsPerMinibatch;
    size_t m_actualnumberOfuttsPerMinibatch;
    size_t m_mbSize;
    vector<size_t> m_toProcess;
    vector<size_t> m_switchFrame;
	vector<size_t> m_validFrame;       //valid frame number in each channel
	vector<size_t> m_extraUttsPerMinibatch;
	size_t m_extraUttNum;
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

    CUDAPageLockedMemAllocator* m_cudaAllocator;
    std::vector<std::shared_ptr<ElemType>> m_featuresBufferMultiIO;
    std::vector<size_t> m_featuresBufferAllocatedMultiIO;
    std::vector<std::shared_ptr<ElemType>> m_labelsBufferMultiIO;
    std::vector<size_t> m_labelsBufferAllocatedMultiIO;
	//for lattice uids and phoneboundaries
	std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>>  m_latticeBufferMultiUtt;
	std::vector<std::vector<size_t>> m_labelsIDBufferMultiUtt;
	std::vector<std::vector<size_t>> m_phoneboundaryIDBufferMultiUtt;
	std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>>  m_extraLatticeBufferMultiUtt;
	std::vector<std::vector<size_t>> m_extraLabelsIDBufferMultiUtt;
	std::vector<std::vector<size_t>> m_extraPhoneboundaryIDBufferMultiUtt;
	//hmm 
	msra::asr::simplesenonehmm m_hset;

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
	bool GetMinibatch4SEToTrainOrTest(std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> & latticeinput, vector<size_t> &uids, vector<size_t> &boundaries, std::vector<size_t> &extrauttmap);
	void fillOneUttDataforParallelmode(std::map<std::wstring, Matrix<ElemType>*>& matrices, size_t startFr, size_t framenum, size_t channelIndex, size_t sourceChannelIndex);
    bool GetMinibatchToWrite(std::map<std::wstring, Matrix<ElemType>*>&matrices);
    
    void StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize);
    void StartMinibatchLoopToWrite(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);

    bool ReNewBufferForMultiIO(size_t i);

    size_t GetNumParallelSequences() { return m_numberOfuttsPerMinibatch; } 
    void SetNumParallelSequences(const size_t) { };

     void GetDataNamesFromConfig(const ConfigParameters& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels,
		 std::vector<std::wstring>& hmms, std::vector<std::wstring>& lattices);
    
    size_t ReadLabelToTargetMappingFile (const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap);
    
    void ExpandDotDotDot(wstring & featPath, const wstring & scpPath, wstring & scpDirCached);

    
    enum InputOutputTypes
    {
        real,
        category,
    };

private:
    CUDAPageLockedMemAllocator* GetCUDAAllocator(int deviceID)
    {
        if (m_cudaAllocator != nullptr)
        {
            if (m_cudaAllocator->GetDeviceId() != deviceID)
            {
                delete m_cudaAllocator;
                m_cudaAllocator = nullptr;
            }
        }

        if (m_cudaAllocator == nullptr)
        {
            m_cudaAllocator = new CUDAPageLockedMemAllocator(deviceID);
        }

        return m_cudaAllocator;
    }

    std::shared_ptr<ElemType> AllocateIntermediateBuffer(int deviceID, size_t numElements)
    {
        if (deviceID >= 0)
        {
            // Use pinned memory for GPU devices for better copy performance
            size_t totalSize = sizeof(ElemType) * numElements;
            return std::shared_ptr<ElemType>((ElemType*)GetCUDAAllocator(deviceID)->Malloc(totalSize), [this, deviceID](ElemType* p) {
                this->GetCUDAAllocator(deviceID)->Free((char*)p);
            });
        }
        else
        {
            return std::shared_ptr<ElemType>(new ElemType[numElements], [](ElemType* p) {
                delete[] p;
            });
        }
    }

public:
    MBLayoutPtr m_pMBLayout;

    /// by default it is false
    /// if true, reader will set to ((int) MinibatchPackingFlags::None) for time positions that are orignally correspond to ((int) MinibatchPackingFlags::SequenceStart)
    /// set to true so that a current minibatch can uses state activities from the previous minibatch. 
    /// default will have truncated BPTT, which only does BPTT inside a minibatch
    bool mIgnoreSentenceBeginTag;
    // TODO: this ^^ does not seem to belong here.

    HTKMLFReader() : m_pMBLayout(make_shared<MBLayout>())
    {
    }
    virtual void Init(const ConfigParameters& config);
    virtual void Destroy() {delete this;}
    virtual ~HTKMLFReader();

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize)
    {
        return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
    }

    virtual bool SupportsDistributedMBRead() const override
    {
        return m_frameSource && m_frameSource->supportsbatchsubsetting();
    }

    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize) override;

    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart=0);
	virtual bool GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> & latticeinput, vector<size_t> &uids, vector<size_t> &boundaries, vector<size_t> &extrauttmap);
	virtual bool GetHmmData(msra::asr::simplesenonehmm * hmm);

    virtual bool DataEnd(EndDataType endDataType);
    void CopyMBLayoutTo(MBLayoutPtr);
    void SetSentenceEndInBatch(vector<size_t> &/*sentenceEnd*/);
    void SetSentenceEnd(int /*actualMbSize*/){};
    void SetRandomSeed(int){ NOT_IMPLEMENTED };

    bool RequireSentenceSeg() const override { return !m_framemode; }; 
};

}}}
