//
// <copyright file="ConcurrentReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// ConncurrentReader.h - Include file for a wrapper reader to do concurrent read-ahead of samples 
#pragma once
#include "DataReader.h"
#include <string>
#include <map>
#include <vector>
#include "readaheadsource.h"

namespace Microsoft { namespace MSR { namespace CNTK {


template<class ElemType, typename LabelType=int>
class DATAREADER_API ConcurrentReader : public IDataReader<ElemType, LabelType>
{
private:
    typedef unsigned LabelIdType;
    IDataReader* m_dataReader;
    size_t m_mbSize;    // size of minibatch requested
    size_t m_mbStartSample; // starting sample # of the next minibatch

    // handle readahead
    minibatchreadaheadsource<ElemType, LabelType> m_readahead; // readahead class
    
public:
    virtual void Init(size_t& vdim, size_t& udim, const std::vector<std::wstring>& filepaths, const ConfigParameters& config);
    virtual void Destroy();
    ConcurrentReader(IDataReader<ElemType, LabelType>* dataReader) : m_dataReader(dataReader), m_readahead(dataReader) {}
    virtual ~ConcurrentReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    virtual bool GetMinibatch(Matrix<ElemType>& features, Matrix<ElemType>& labels);
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping( );
    virtual void SetLabelMapping(const std::map<LabelIdType, LabelType>& labelMapping);
};

}}}