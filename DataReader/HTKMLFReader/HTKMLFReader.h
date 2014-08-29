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
    msra::dbn::minibatchreadaheadsource* m_readAheadSource;
    msra::dbn::latticesource* m_lattices;
    map<wstring,msra::lattices::lattice::htkmlfwordsequence> m_latticeMap;
	//std::vector<msra::dbn::latticesource> m_latticesMultiIO;
    std::vector<map<wstring,msra::lattices::lattice::htkmlfwordsequence>> m_latticeMapMultiIO;
	//std::vector<unique_ptr<msra::dbn::latticesource>> m_latticesMultiIO;
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
    
    msra::dbn::minibatchframesourcemulti* m_frameSourceMultiIO;
    msra::dbn::chunkevalsource* m_chunkEvalSource;
    msra::dbn::chunkevalsourcemulti* m_chunkEvalSourceMultiIO;
	msra::dbn::FileEvalSource* m_fileEvalSource; 
	
    size_t m_udim;    // dimensions of label matrix, need to generate this many columns
	size_t fdim;
    std::vector<size_t> vdims;
    std::vector<size_t> udims;
	std::vector<size_t> featDims;
	std::vector<size_t> labelDims;

    std::map<LabelIdType, LabelType> m_idToLabelMap;
    
    bool m_partialMinibatch; // allow partial minibatches?
    ElemType* m_featuresBuffer;
    size_t m_featuresBufferAllocated;
    ElemType* m_labelsBuffer;
    size_t m_labelsBufferAllocated;
    
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


    std::map<std::wstring,size_t> featureNameToIdMap;
    std::map<std::wstring,size_t> labelNameToIdMap;
    std::map<std::wstring,size_t> outputNameToIdMap;
    std::map<std::wstring,size_t> nameToTypeMap;
    std::map<std::wstring,size_t> featureNameToDimMap;
    std::map<std::wstring,size_t> labelNameToDimMap;
    // for writing outputs to files (standard single input/output network) - deprecate eventually
    std::wstring outputPath;
    std::wstring outputExtension;
    ElemType *m_outputBuffer;
    size_t m_outputBufferAllocated;
    std::wstring outputNodeName;
    std::vector<std::wstring> inputFiles;
    std::vector<std::wstring> outfiles; // shared between input and output files
    std::vector<size_t> numFrames;
    size_t inputFileIndex;
    std::wstring outputScp;
    bool scaleByPrior;
    bool checkDictionaryKeys;
    bool convertLabelsToTargets;
	std::vector <bool> convertLabelsToTargetsMultiIO;
    std::vector<std::vector<ElemType>> labelToTargetMap;
	std::vector<std::vector<std::vector<ElemType>>>labelToTargetMapMultiIO;
    // for writing outputs to files (arbitrary multiple input/output network)
    std::vector<std::wstring> outputPaths;
    std::vector<std::wstring> outputExtensions;
    std::vector<ElemType*> m_outputBufferMultiIO;
    std::vector<size_t> m_outputBufferAllocatedMultiIO;
    std::vector<std::wstring> outputNodeNames;
    std::vector<std::vector<std::wstring>> inputFilesMultiIO;
    std::vector<std::wstring> outputScps;
    std::vector<bool>scaleByPriorMultiIO;

    bool multiIO;
	bool m_legacyMode;
    void InitToTrainOrTest(const ConfigParameters& config);
    void InitToWriteOutput(const ConfigParameters& config);
    void InitSingleIO(const ConfigParameters& config);
    void InitMultiIO(const ConfigParameters& config);
    void InitEvalSingleIO(const ConfigParameters& config);
    void InitEvalMultiIO(const ConfigParameters& config);    
	void PrepareForTrainingOrTesting(const ConfigParameters& config);
	void PrepareForWriting(const ConfigParameters& config);
	void InitLegacy(const ConfigParameters& readerConfig);

	void InitEvalReader(const ConfigParameters& readerConfig);
    
	bool GetMinibatchSingleIO(std::map<std::wstring, Matrix<ElemType>*>& matrices); // assumes default naming of "features" and "labels"
    bool GetMinibatchMultiIO(std::map<std::wstring, Matrix<ElemType>*>& matrices); 

	bool ReNewBuffer(size_t i);
	bool ReNewBufferForMultiIO(size_t i);

    bool GetMinibatchEvalSingleIO(std::map<std::wstring, Matrix<ElemType>*>& matrices); // assumes default naming of "features" and "labels"
    bool GetMinibatchEvalMultiIO(std::map<std::wstring, Matrix<ElemType>*>& matrices); 
	bool GetMinibatchEval(std::map<std::wstring, Matrix<ElemType>*>&matrices);
	bool GetMinibatchToTrainOrTest(std::map<std::wstring, Matrix<ElemType>*>&matrices);
	bool GetMinibatchToWrite(std::map<std::wstring, Matrix<ElemType>*>&matrices);

    size_t NumberSlicesInEachRecurrentIter() { return m_numberOfuttsPerMinibatch ;} 
    void SetNbrSlicesEachRecurrentIter(const size_t) { };

    bool trainOrTest; // not write output
    void StartMinibatchLoopTrain(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    void StartMinibatchLoopEval(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    void StartMinibatchLoopToTrainOrTest(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    void StartMinibatchLoopToWrite(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    std::wstring MakeOutPath (const std::wstring & outdir, std::wstring file, const std::wstring & outext);
	void GetDataNamesFromConfig(const ConfigParameters& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels);

    void WriteOutputScp();
    static const std::wstring DefaultFeaturesName() {return L"features";}
    static const std::wstring DefaultLabelsName() {return L"labels";}
    static const std::wstring DefaultInputsName() {return L"input";}
    static const std::wstring DefaultOutputsName() {return L"output";}
    static const std::wstring DefaultPriorName() {return L"Prior";}
    
    size_t ReadLabelToTargetMappingFile (const std::wstring& labelToTargetMappingFile, const std::wstring& labelListFile, std::vector<std::vector<ElemType>>& labelToTargetMap);
    enum InputOutputTypes
    {
		real,
		category,
        inputReal, 
        inputCategory,
        outputReal,
        outputCategory,
        networkOutputs,
    };

public:
    virtual void Init(const ConfigParameters& config);
    virtual void Destroy() {delete this;}
    virtual ~HTKMLFReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<unsigned, LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart=0);

    virtual bool DataEnd(EndDataType endDataType);
	void SetSentenceEndInBatch(vector<size_t> &sentenceEnd);
	void SetSentenceEnd(int actualMbSize){};
};

}}}