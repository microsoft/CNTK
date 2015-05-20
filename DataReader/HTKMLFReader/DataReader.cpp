//
// <copyright file="DataReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "basetypes.h"

#include "htkfeatio.h"                  // for reading HTK features
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "simplesenonehmm.h"            // for MMI scoring
#include "msra_mgram.h"                 // for unigram scores of ground-truth path in sequence training

#include "rollingwindowsource.h"        // minibatch sources
#include "utterancesource.h"
#include "readaheadsource.h"
#include "chunkevalsource.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "HTKMLFReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
void DATAREADER_API GetReader(IDataReader<ElemType>** preader)
{
    *preader = new HTKMLFReader<ElemType>();
}

extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
{
    GetReader(preader);
}
extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
{
    GetReader(preader);
}


// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
template<class ElemType>
void DataReader<ElemType>::Init(const ConfigParameters& readerConfig)
{
    m_dataReader = new HTKMLFReader<ElemType>();
    m_dataReader->Init(readerConfig);
}

template<class ElemType>
void DataReader<ElemType>::GetDataReader(const ConfigParameters& /*config*/)
{
    NOT_IMPLEMENTED;
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template<class ElemType>
void DataReader<ElemType>::Destroy()
{
    delete m_dataReader;
    m_dataReader = NULL;
}

// DataReader Constructor
// config - string  of options (i.e. "-windowsize:11 -addenergy") data reader specific 
template<class ElemType>
DataReader<ElemType>::DataReader(const ConfigParameters& config)
{
    Init(config);
}


// destructor - cleanup temp files, etc. 
template<class ElemType>
DataReader<ElemType>::~DataReader()
{
    delete m_dataReader;
    m_dataReader = NULL;
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
int DataReader<ElemType>::GetSentenceEndIdFromOutputLabel()
{
    return m_dataReader->GetSentenceEndIdFromOutputLabel();
}

template<class ElemType>
void DataReader<ElemType>::InitProposals(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    m_dataReader->InitProposals(matrices);
}

template<class ElemType>
bool DataReader<ElemType>::GetProposalObs(std::map<std::wstring, Matrix<ElemType>*>& matrices, const size_t tidx, vector<size_t>& history)
{
    return m_dataReader->GetProposalObs(matrices, tidx, history);
}

template<class ElemType>
void DataReader<ElemType>::SetNbrSlicesEachRecurrentIter(const size_t sz)
{
    m_dataReader->SetNbrSlicesEachRecurrentIter(sz);
}

template<class ElemType>
void DataReader<ElemType>::SetSentenceSegBatch(Matrix<ElemType>& sentenceEnd, Matrix<ElemType>& sentenceExistBeginOrNoLabels)
{
    m_dataReader->SetSentenceSegBatch(sentenceEnd, sentenceExistBeginOrNoLabels);
}

template<class ElemType>
void DataReader<ElemType>::SetRandomSeed(int seed)
{
    m_dataReader->SetRandomSeed(seed);
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

template class DataReader<float>;
template class DataReader<double>;

// Utility function, in ConfigFile.cpp, but HTKMLFReader doesn't need that code...

// Trim - trim white space off the start and end of the string
// str - string to trim
// NOTE: if the entire string is empty, then the string will be set to an empty string
void Trim(std::string& str)
{
    auto found = str.find_first_not_of(" \t");
    if (found == npos)
    {
        str.erase(0);
        return;
    }
    str.erase(0, found);
    found = str.find_last_not_of(" \t");
    if (found != npos)
        str.erase(found+1);
}


}}}