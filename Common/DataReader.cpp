//
// <copyright file="DataReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_LOCAL
#include "basetypes.h"
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
std::string GetReaderName(ElemType)
{std::string empty; return empty;}

template<> std::string GetReaderName(float) {std::string name = "GetReaderF"; return name;}
template<> std::string GetReaderName(double) {std::string name = "GetReaderD"; return name;}

template<class ElemType>
void DataReader<ElemType>::Init(const ConfigParameters& /*config*/)
{
    RuntimeError("Init shouldn't be called, use constructor");
    // not implemented, calls the underlying class instead
}


// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template<class ElemType>
void DataReader<ElemType>::Destroy()
{
    m_dataReader->Destroy();
}

// DataReader Constructor
// config - [in] string  of options (i.e. "filename=data.txt") data reader specific 
template<class ElemType>
void DataReader<ElemType>::GetDataReader(const ConfigParameters& config)
{
    typedef void (*GetReaderProc)(IDataReader<ElemType>** preader);

    // initialize just in case
    m_dataReader = NULL;

    // get the name for the reader we want to use, default to UCIFastReader
    // create a variable of each type just to call the proper templated version
    ElemType elemType = ElemType();
    GetReaderProc getReaderProc = (GetReaderProc)Plugin::Load(config("readerType", "UCIFastReader"), GetReaderName(elemType).c_str());
    getReaderProc(&m_dataReader);
}

// DataReader Constructor
// options - [in] string  of options (i.e. "-windowsize:11 -addenergy") data reader specific 
template<class ElemType>
DataReader<ElemType>::DataReader(const ConfigParameters& config)
{
    GetDataReader(config);
    // now pass that to concurrent reader so we can read ahead
    //m_dataReader = new ConcurrentReader<ElemType>(m_dataReader);
    // NOW we can init
    m_dataReader->Init(config);
}


// destructor - cleanup temp files, etc. 
template<class ElemType>
DataReader<ElemType>::~DataReader()
{
    // free up resources
    if (m_dataReader != NULL)
    {
        m_dataReader->Destroy();
        m_dataReader = NULL;
    }
}

//StartMinibatchLoop - Startup a minibatch loop 
// mbSize - [in] size of the minibatch (number of frames, etc.)
// epoch - [in] epoch number for this loop
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template<class ElemType>
void DataReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    m_dataReader->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
//             [out] each matrix resized if necessary containing data. 
// returns - true if there are more minibatches, false if no more minibatchs remain
template<class ElemType>
bool DataReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    return m_dataReader->GetMinibatch(matrices);
}

template<class ElemType>
size_t DataReader<ElemType>::NumberSlicesInEachRecurrentIter()
{
    return m_dataReader->NumberSlicesInEachRecurrentIter();
}

template<class ElemType>
void DataReader<ElemType>::SetNbrSlicesEachRecurrentIter(const size_t sz)
{
    m_dataReader->SetNbrSlicesEachRecurrentIter(sz);
}
template<class ElemType>
void DataReader<ElemType>::SetSentenceEndInBatch(std::vector<size_t> &sentenceEnd)
{
    m_dataReader->SetSentenceEndInBatch(sentenceEnd);
}
// GetLabelMapping - Gets the label mapping from integer index to label type 
// returns - a map from numeric datatype to native label type 
template<class ElemType>
const map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& DataReader<ElemType>::GetLabelMapping(const std::wstring& sectionName)
{
    return m_dataReader->GetLabelMapping(sectionName);
}

// SetLabelMapping - Sets the label mapping from integer index to label 
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
template<class ElemType>
void DataReader<ElemType>::SetLabelMapping(const std::wstring& sectionName, const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& labelMapping)
{
    m_dataReader->SetLabelMapping(sectionName, labelMapping);
}

// GetData - Gets metadata from the specified section (into CPU memory) 
// sectionName - section name to retrieve data from
// numRecords - number of records to read
// data - pointer to data buffer, if NULL, dataBufferSize will be set to size of required buffer to accomidate request
// dataBufferSize - [in] size of the databuffer in bytes
//                  [out] size of buffer filled with data
// recordStart - record to start reading from, defaults to zero (start of data)
// returns: true if data remains to be read, false if the end of data was reached
template<class ElemType>
bool DataReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    return m_dataReader->GetData(sectionName, numRecords, data, dataBufferSize, recordStart);
}

template<class ElemType>
bool DataReader<ElemType>::DataEnd(EndDataType endDataType)
{
    return m_dataReader->DataEnd(endDataType); 
}

//The explicit instantiation
template class DataReader<double>; 
template class DataReader<float>;

}}}