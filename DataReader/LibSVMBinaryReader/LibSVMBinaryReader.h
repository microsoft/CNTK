//
// <copyright file="LibSVMBinaryReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// LibSVMBinaryReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
#include "DataReader.h"
#include "DataWriter.h"
#include <string>
#include "commandArgUtil.h"
#include <map>
#include <vector>
#include "minibatchsourcehelpers.h"

namespace Microsoft { namespace MSR { namespace CNTK {

enum LabelKind
{
    labelNone = 0,  // no labels to worry about
    labelCategory = 1, // category labels, creates mapping tables
    labelRegression = 2,  // regression labels
    labelOther = 3, // some other type of label
};


template<class ElemType>
class LibSVM_BinaryInput {
private:
    HANDLE m_hndl;
    HANDLE m_filemap;
    HANDLE m_header;
    HANDLE m_offsets;
    HANDLE m_data;

    //void* header_orig; // Don't need this since the header is at the start of the file
    void* offsets_orig;
    void* data_orig;

    void* header_buffer;
    void* offsets_buffer;
    void* data_buffer;

    size_t m_dim;
    size_t mbSize;
    size_t MAX_BUFFER = 400;
    size_t m_labelDim;

    ElemType* values; // = (ElemType*)malloc(sizeof(ElemType)* 230 * 1024);
    int64_t* offsets; // = (int*)malloc(sizeof(int)* 230 * 1024);
    int32_t* colIndices; // = (int*)malloc(sizeof(int) * (batchsize + 1));
    int32_t* rowIndices; // = (int*)malloc(sizeof(int) * MAX_BUFFER * batchsize);
    int32_t* classIndex; // = (int*)malloc(sizeof(int) * batchsize);
    ElemType* classWeight; // = (ElemType*)malloc(sizeof(ElemType) * batchsize);

    ElemType* m_labelsBuffer;
public:
    int64_t numRows;
    int64_t numBatches;
    int32_t numCols;
    int64_t totalNNz;

    LibSVM_BinaryInput();
    ~LibSVM_BinaryInput();
    void Init(std::wstring fileName, size_t dim);
    bool SetupEpoch( size_t minibatchSize);
    bool Next_Batch(Matrix<ElemType>& features, Matrix<ElemType>& labels, size_t actualmbsize, int batchIndex);
    void Dispose();
};

template<class ElemType>
class LibSVMBinaryReader : public IDataReader<ElemType>
{
//public:
//    typedef std::string LabelType;
//    typedef unsigned LabelIdType;
private:
    int* read_order; // array to shuffle to reorder the dataset
    std::wstring m_featuresName;
    size_t m_featuresDim;
    LibSVM_BinaryInput<ElemType> featuresInput;
    int64_t m_processedMinibatches;

    size_t m_mbSize;    // size of minibatch requested
    LabelIdType m_labelIdMax; // maximum label ID we have encountered so far
    LabelIdType m_labelDim; // maximum label ID we will ever see (used for array dimensions)
    size_t m_mbStartSample; // starting sample # of the next minibatch
    size_t m_epochSize; // size of an epoch
    size_t m_epoch; // which epoch are we on
    size_t m_epochStartSample; // the starting sample for the epoch
    size_t m_totalSamples;  // number of samples in the dataset
    size_t m_randomizeRange; // randomization range
    size_t m_featureCount; // feature count
    size_t m_readNextSample; // next sample to read
    bool m_labelFirst;  // the label is the first element in a line
    bool m_partialMinibatch;    // a partial minibatch is allowed
    LabelKind m_labelType;  // labels are categories, create mapping table
    msra::dbn::randomordering m_randomordering;   // randomizing class

    std::wstring m_labelsName;
    std::wstring m_labelsCategoryName;
    std::wstring m_labelsMapName;
    ElemType* m_qfeaturesBuffer;
    ElemType* m_dfeaturesBuffer;
    ElemType* m_labelsBuffer;
    LabelIdType* m_labelsIdBuffer;
    std::wstring m_labelFileToWrite;  // set to the path if we need to write out the label file

    bool m_endReached;
    int m_traceLevel;
   
    // feature and label data are parallel arrays
    std::vector<ElemType> m_featureData;
    std::vector<LabelIdType> m_labelIdData;
    std::vector<LabelType> m_labelData;

    // map is from ElemType to LabelType
    // For LibSVMBinary, we really only need an int for label data, but we have to transmit in Matrix, so use ElemType instead
    std::map<LabelIdType, LabelType> m_mapIdToLabel;
    std::map<LabelType, LabelIdType> m_mapLabelToId;

    // caching support
    DataReader<ElemType>* m_cachingReader;
    DataWriter<ElemType>* m_cachingWriter;
    ConfigParameters m_readerConfig;

    size_t RandomizeSweep(size_t epochSample);
    //bool Randomize() {return m_randomizeRange != randomizeNone;}
    bool Randomize() { return false; }
    void SetupEpoch();
    void StoreLabel(ElemType& labelStore, const LabelType& labelValue);
    size_t RecordsToRead(size_t mbStartSample, bool tail=false);
    void ReleaseMemory();
    void WriteLabelFile();


    virtual bool ReadRecord(size_t readSample);
public:
    virtual void Init(const ConfigParameters& config);
    virtual void Destroy();
    LibSVMBinaryReader() { m_qfeaturesBuffer = NULL; m_dfeaturesBuffer = NULL;  m_labelsBuffer = NULL; }
    virtual ~LibSVMBinaryReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);

    size_t NumberSlicesInEachRecurrentIter() { return 1 ;} 
    void SetNbrSlicesEachRecurrentIter(const size_t) { };
	void SetSentenceSegBatch(Matrix<ElemType> &, vector<MinibatchPackingFlag>& ){};
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, typename LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart=0);

    virtual bool DataEnd(EndDataType endDataType);
    void SetRandomSeed(int) { NOT_IMPLEMENTED; }
};
}}}
