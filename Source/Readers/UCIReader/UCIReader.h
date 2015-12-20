//
// <copyright file="UCIReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// UCIReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
#include "DataReader.h"
#include <string>
#include <map>
#include <vector>
#include "minibatchsourcehelpers.h"
#include "readaheadsource.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class IReadAhead
{
public:
    virtual bool EnsureDataAvailable(size_t mbStartSample, size_t& numberRead) = 0;
    virtual bool ReadRecord(size_t readSample) = 0;
};

enum LabelKind
{
    labelNone = 0,  // no labels to worry about
    labelCategory = 1, // category labels, creates mapping tables
    labelRegression = 2,  // regression labels
    labelOther = 3, // some other type of label
};

template<class ElemType, typename LabelType=int>
class UCIReader : public IDataReader<ElemType, LabelType>, IReadAhead
{
private:
    typedef unsigned LabelIdType;
    File* m_file;   // file class we are read/writing from
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

    bool m_endReached;
    minibatchreadaheadsource<ElemType, LabelType> m_readahead; // readahead class
    
    // feature and label data are parallel arrays
    std::vector<ElemType> m_featureData;
    std::vector<LabelIdType> m_labelIdData;
    std::vector<LabelType> m_labelData;
    // map is from ElemType to LabelType
    // For UCI, we really only need an int for label data, but we have to transmit in Matrix, so use ElemType instead
    std::map<LabelIdType, LabelType> m_mapIdToLabel;
    std::map<LabelType, LabelIdType> m_mapLabelToId;

    size_t RandomizeSweep(size_t epochSample);
    bool Randomize() {return m_randomizeRange != randomizeNone;}
    void UpdateDataVariables();
    void SetupEpoch();
    virtual bool EnsureDataAvailable(size_t mbStartSample, size_t& numberRead);
    virtual bool ReadRecord(size_t readSample);
public:
    virtual void Init(size_t& vdim, size_t& udim, const std::vector<std::wstring>& filepaths, const ConfigParameters& config);
    virtual void Destroy() {}
    UCIReader() : m_readahead((IDataReader<ElemType, LabelType>*)this) {}
    virtual ~UCIReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    virtual bool GetMinibatch(Matrix<ElemType>& features, Matrix<ElemType>& labels);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping( );
    virtual void SetLabelMapping(const std::map<LabelIdType, LabelType>& labelMapping);
};

}}}