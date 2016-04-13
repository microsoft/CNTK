//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// HTKMLFReader.h - Include file for the MTK and MLF format of features and samples
//
#pragma once
#include "DataReader.h"
#include "KaldiSequenceTrainingDerivative.h"
#include "UtteranceDerivativeBuffer.h"
#include "Config.h" // for intargvector

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class HTKMLFReader : public DataReaderBase
{
private:
    msra::dbn::minibatchiterator* m_mbiter;
    msra::dbn::minibatchsource* m_frameSource;
    vector<msra::asr::FeatureSection*> m_trainingOrTestingFeatureSections;
    // msra::dbn::minibatchreadaheadsource* m_readAheadSource;
    msra::dbn::FileEvalSource* m_fileEvalSource;
    vector<msra::asr::FeatureSection*> m_writingFeatureSections;
    msra::dbn::latticesource* m_lattices;
    map<wstring, msra::lattices::lattice::htkmlfwordsequence> m_latticeMap;

    // Sequence training realted members.
    bool m_doSeqTrain;
    wstring m_seqTrainCriterion;
    KaldiSequenceTrainingDerivative<ElemType>* m_seqTrainDeriv;

    // Minibatch buffering.
    struct MinibatchBufferUnit
    {
        std::vector<std::vector<ElemType>> features;
        std::vector<std::vector<ElemType>> labels;
        MBLayoutPtr pMBLayout;
        std::vector<std::vector<std::pair<wstring, size_t>>> minibatchUttInfo;
        size_t currentMBSize;
        MinibatchBufferUnit()
            : pMBLayout(make_shared<MBLayout>()), currentMBSize(0)
        {
        }
    };
    bool m_doMinibatchBuffering;
    bool m_getMinibatchCopy;
    bool m_doMinibatchBufferTruncation;
    size_t m_minibatchBufferIndex;
    std::deque<MinibatchBufferUnit> m_minibatchBuffer;
    UtteranceDerivativeBuffer<ElemType>* m_uttDerivBuffer;
    unordered_map<wstring, bool> m_hasUttInCurrentMinibatch;

    // Utterance information.
    std::vector<std::vector<std::pair<wstring, size_t>>> m_uttInfo;
    std::vector<std::vector<std::pair<wstring, size_t>>> m_minibatchUttInfo;

    vector<bool> m_sentenceEnd;
    bool m_readAhead;
    bool m_truncated;
    bool m_framemode;
    vector<size_t> m_processedFrame;
    size_t m_maxUtteranceLength;
    size_t m_numberOfuttsPerMinibatch;
    size_t m_actualnumberOfuttsPerMinibatch;
    size_t m_mbSize;
    size_t m_currentMBSize;
    vector<size_t> m_currentBufferFrames;
    vector<size_t> m_toProcess;
    vector<size_t> m_switchFrame;
    bool m_noData;

    bool m_trainOrTest; // if false, in file writing mode

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

    std::map<std::wstring, size_t> m_featureNameToIdMap;
    std::map<std::wstring, size_t> m_labelNameToIdMap;
    std::map<std::wstring, size_t> m_nameToTypeMap;
    std::map<std::wstring, size_t> m_featureNameToDimMap;
    std::map<std::wstring, size_t> m_labelNameToDimMap;
    std::vector<std::wstring> m_featureIdToNameMap;
    std::vector<std::wstring> m_labelIdToNameMap;
    // for writing outputs to files (standard single input/output network) - deprecate eventually
    bool m_checkDictionaryKeys;
    bool m_convertLabelsToTargets;
    std::vector<bool> m_convertLabelsToTargetsMultiIO;
    std::vector<std::vector<std::wstring>> m_inputFilesMultiIO;

    size_t m_inputFileIndex;
    std::vector<size_t> m_featDims;
    std::vector<size_t> m_labelDims;

    std::vector<std::vector<std::vector<ElemType>>> m_labelToTargetMapMultiIO;

    template <class ConfigRecordType>
    void PrepareForTrainingOrTesting(const ConfigRecordType& config);
    template <class ConfigRecordType>
    void PrepareForWriting(const ConfigRecordType& config);
    template <class ConfigRecordType>
    void PrepareForSequenceTraining(const ConfigRecordType& config);

    bool GetMinibatchToTrainOrTest(StreamMinibatchInputs& matrices);
    bool GetOneMinibatchToTrainOrTestDataBuffer(const StreamMinibatchInputs& matrices);
    bool GetMinibatchToWrite(StreamMinibatchInputs& matrices);
    bool PopulateUtteranceInMinibatch(const StreamMinibatchInputs& matrices, size_t uttIndex, size_t startFrame, size_t endFrame, size_t mbSize, size_t mbOffset = 0);

    // If we have to read the current minibatch from buffer, return true,
    // otherwise return false.
    bool ShouldCopyMinibatchFromBuffer();

    // Copys the current minibatch to buffer.
    void CopyMinibatchToBuffer();

    // Copys one minibatch from buffer to matrix.
    void CopyMinibatchFromBufferToMatrix(size_t index, StreamMinibatchInputs& matrices);

    // Copys one minibatch from <m_featuresBufferMultiIO> to matrix.
    void CopyMinibatchToMatrix(size_t size, const std::vector<ElemType*>& featureBuffer, const std::vector<ElemType*>& labelBuffer, StreamMinibatchInputs& matrices) const;

    void StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
    void StartMinibatchLoopToWrite(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);

    bool ReNewBufferForMultiIO(size_t i);

    size_t NumberSlicesInEachRecurrentIter()
    {
        return m_numberOfuttsPerMinibatch;
    }
    void SetNbrSlicesEachRecurrentIter(const size_t){};

    template <class ConfigRecordType>
    void GetDataNamesFromConfig(const ConfigRecordType& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels);

    size_t ReadLabelToTargetMappingFile(const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap);
    enum InputOutputTypes
    {
        real,
        category,
        readerDeriv, /*derivative computed in the reader*/
        readerObj,   /*objective computed in the reader*/
    };

public:
    MBLayoutPtr m_pMBLayout;

    // by default it is false
    // if true, reader will set to SEQUENCE_MIDDLE for time positions that are orignally correspond to SEQUENCE_START
    // set to true so that a current minibatch can uses state activities from the previous minibatch.
    // default will have truncated BPTT, which only does BPTT inside a minibatch
    bool mIgnoreSentenceBeginTag;
    HTKMLFReader()
        : m_pMBLayout(make_shared<MBLayout>())
    {
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
    virtual void Destroy()
    {
        delete this;
    }
    virtual ~HTKMLFReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
    virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);
    virtual size_t GetNumParallelSequences()
    {
        return m_numberOfuttsPerMinibatch;
    }

    virtual bool GetMinibatchCopy(
        std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        StreamMinibatchInputs& matrices,
        MBLayoutPtr pMBLayout);
    virtual bool SetNetOutput(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const MatrixBase& outputs,
        const MBLayoutPtr pMBLayout);

    virtual bool DataEnd();
    void SetSentenceEndInBatch(vector<size_t>& /*sentenceEnd*/);
    void SetSentenceEnd(int /*actualMbSize*/){};

    void CopyMBLayoutTo(MBLayoutPtr pMBLayout)
    {
        pMBLayout->CopyFrom(m_pMBLayout);
    }
    //bool RequireSentenceSeg() const override { return !m_framemode; };
};
} } }
