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

    /******************************************************************
	START OF FUNCTIONS AND VARIABLES THAT HAVE BEEN DEPRECATED 
	******************************************************************/	
	// msra::dbn::minibatchframesourcemulti* m_frameSourceMultiIO;
    // msra::dbn::chunkevalsource* m_chunkEvalSource;
    // msra::dbn::chunkevalsourcemulti* m_chunkEvalSourceMultiIO;
	/*
	ElemType* m_featuresBuffer;
    size_t m_featuresBufferAllocated;
    ElemType* m_labelsBuffer;
    size_t m_labelsBufferAllocated;
    std::wstring outputScp;
    bool scaleByPrior;
    std::wstring outputPath;
    std::wstring outputExtension;
     ElemType *m_outputBuffer;
    size_t m_outputBufferAllocated;
    std::wstring outputNodeName;
	//std::vector<msra::dbn::latticesource> m_latticesMultiIO;
    std::vector<map<wstring,msra::lattices::lattice::htkmlfwordsequence>> m_latticeMapMultiIO;
	//std::vector<unique_ptr<msra::dbn::latticesource>> m_latticesMultiIO;
   std::map<std::wstring,size_t> m_outputNameToIdMap;
    std::vector<std::wstring> outputPaths;
    std::vector<std::wstring> outputExtensions;
    std::vector<ElemType*> m_outputBufferMultiIO;
    std::vector<size_t> m_outputBufferAllocatedMultiIO;
 size_t fdim;
    std::vector<size_t> vdims;
    std::vector<size_t> udims;

    size_t m_udim;    // dimensions of label matrix, need to generate this many columns

	bool multiIO;
	bool m_legacyMode;
    std::vector<std::wstring> outputNodeNames;
    std::vector<std::wstring> outputScps;
    std::vector<bool>scaleByPriorMultiIO;
	std::vector<std::vector<ElemType>> m_labelToTargetMap;

	void InitLegacy(const ConfigParameters& readerConfig);
	void InitToTrainOrTest(const ConfigParameters& config);
    void InitToWriteOutput(const ConfigParameters& config);
    void InitSingleIO(const ConfigParameters& config);
    void InitMultiIO(const ConfigParameters& config);
    void InitEvalSingleIO(const ConfigParameters& config);
    void InitEvalMultiIO(const ConfigParameters& config);    
	void InitEvalReader(const ConfigParameters& readerConfig);
    
    void StartMinibatchLoopTrain(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    void StartMinibatchLoopEval(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);

	bool GetMinibatchSingleIO(std::map<std::wstring, Matrix<ElemType>*>& matrices); // assumes default naming of "features" and "labels"
    bool GetMinibatchMultiIO(std::map<std::wstring, Matrix<ElemType>*>& matrices); 


    bool GetMinibatchEvalSingleIO(std::map<std::wstring, Matrix<ElemType>*>& matrices); // assumes default naming of "features" and "labels"
    bool GetMinibatchEvalMultiIO(std::map<std::wstring, Matrix<ElemType>*>& matrices); 
	bool GetMinibatchEval(std::map<std::wstring, Matrix<ElemType>*>&matrices);
	   void WriteOutputScp();
 
    std::vector<std::wstring> inputFiles;
    std::vector<std::wstring> outfiles; // shared between input and output files
   std::wstring MakeOutPath (const std::wstring & outdir, std::wstring file, const std::wstring & outext);


	bool ReNewBuffer(size_t i);
    static const std::wstring DefaultFeaturesName() {return L"features";}
    static const std::wstring DefaultLabelsName() {return L"labels";}
    static const std::wstring DefaultInputsName() {return L"input";}
    static const std::wstring DefaultOutputsName() {return L"output";}
    static const std::wstring DefaultPriorName() {return L"Prior";}
	*/


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