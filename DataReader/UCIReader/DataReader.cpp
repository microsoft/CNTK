//
// <copyright file="DataReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "UCIReader.h"
#include "ConcurrentReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType, typename LabelType>
void DATAREADER_API GetReader(IDataReader<ElemType, LabelType>** preader)
{
    *preader = new UCIReader<ElemType, LabelType>();
}

extern "C" DATAREADER_API void GetReaderFI(IDataReader<float, int>** preader)
{
    GetReader(preader);
}
extern "C" DATAREADER_API void GetReaderDI(IDataReader<double, int>** preader)
{
    GetReader(preader);
}
//extern "C" DATAREADER_API void GetReaderFS(IDataReader<float, std::string>** preader)
//{
//    GetReader(preader);
//}
//extern "C" DATAREADER_API void GetReaderDS(IDataReader<double, std::string>** preader)
//{
//    GetReader(preader);
//}
extern "C" DATAREADER_API void GetReaderFF(IDataReader<float, float>** preader)
{
    GetReader(preader);
}
extern "C" DATAREADER_API void GetReaderDD(IDataReader<double, double>** preader)
{
    GetReader(preader);
}

// Init - Initialize the reader (see the constructor)
template<class ElemType, typename LabelType>
void DataReader<ElemType, LabelType>::Init(size_t& vdim, size_t& udim, const vector<std::wstring>& filepaths, const ConfigParameters& config)
{
    // not implemented, calls the underlying class instead
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template<class ElemType, typename LabelType>
void DataReader<ElemType, LabelType>::Destroy()
{
    delete m_dataReader;
    m_dataReader = NULL;
}

// DataReader Constructor
// vdim - [out] number of elements in a single frame of feature values (single precision values) 
// udim - [out] number of columns in the label matrix 
// filepaths - [in] and array of file paths to necessary files, it is variable depending on the reader 
// options - [in] string  of options (i.e. "-windowsize:11 -addenergy") data reader specific 
template<class ElemType, typename LabelType>
DataReader<ElemType, LabelType>::DataReader(size_t& vdim, size_t& udim, const vector<std::wstring>& filepaths, const ConfigParameters& config)
{
    // create a UCIReader
    m_dataReader = new UCIReader<ElemType, LabelType>();
    // now pass that to concurrent reader so we can read ahead
    //m_dataReader = new ConcurrentReader<ElemType, LabelType>(m_dataReader);
    // NOW we can init
    m_dataReader->Init(vdim, udim, filepaths, config);
}

// destructor - cleanup temp files, etc. 
template<class ElemType, typename LabelType>
DataReader<ElemType, LabelType>::~DataReader()
{
    delete m_dataReader;
    m_dataReader = NULL;
}

//StartMinibatchLoop - Startup a minibatch loop 
// mbSize - [in] size of the minibatch (number of frames, etc.)
// epoch - [in] epoch number for this loop
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template<class ElemType, typename LabelType>
void DataReader<ElemType, LabelType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    m_dataReader->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
}

// GetMinibatch - Get the next minibatch (features and labels)
// features - [out] returns minibatch in passed in matrix, will resize and replace existing data. 
//      Number of columns returned may be less than requested mbSize if dataset size is smaller than requested  has been reached.
// labels - [out] returns matrix of label values as normalized integers (0-x) for a classification tasks, and will be a matrix of (0 or 1) values for regression tasks. Existing data will be replaced. 
//      in non-supervised training, labels will not be used
// returns - true if there are more minibatches, false if no more minibatchs remain
template<class ElemType, typename LabelType>
bool DataReader<ElemType, LabelType>::GetMinibatch(Matrix<ElemType>& features, Matrix<ElemType>& labels)
{
    return m_dataReader->GetMinibatch(features, labels);
}

// GetLabelMapping - Gets the label mapping from integer index to label type 
// returns - a map from numeric datatype to native label type 
template<class ElemType, typename LabelType>
const map<unsigned, LabelType>& DataReader<ElemType, LabelType>::GetLabelMapping( )
{
    return m_dataReader->GetLabelMapping();
}

// SetLabelMapping - Sets the label mapping from integer index to label 
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
template<class ElemType, typename LabelType>
void DataReader<ElemType, LabelType>::SetLabelMapping(const std::map<unsigned, LabelType>& labelMapping)
{
    m_dataReader->SetLabelMapping(labelMapping);
}

//The explicit instantiation
//template class DataReader<double, std::wstring>; 
//template class DataReader<float, std::wstring>;
template class DataReader<float, int>;
template class DataReader<double, int>;
template class DataReader<float, float>;
template class DataReader<double, double>;

}}}