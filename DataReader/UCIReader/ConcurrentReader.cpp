//
// <copyright file="ConcurrentReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// ConcurrentReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "ConcurrentReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Init - Initialize the reader (see the constructor)
template<class ElemType, typename LabelType>
void ConcurrentReader<ElemType, LabelType>::Init(size_t& vdim, size_t& udim, const vector<std::wstring>& filepaths, const ConfigParameters& config)
{
    // not implemented, calls the underlying class instead
    m_dataReader->Init(vdim, udim, filepaths, config);
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template<class ElemType, typename LabelType>
void ConcurrentReader<ElemType, LabelType>::Destroy()
{
    delete m_dataReader;
    m_dataReader = NULL;
}

// destructor - cleanup temp files, etc. 
template<class ElemType, typename LabelType>
ConcurrentReader<ElemType, LabelType>::~ConcurrentReader()
{   // we own the underlying datareader, so release it when we are going away
    delete m_dataReader;
    m_dataReader = NULL;
}

//StartMinibatchLoop - Startup a minibatch loop 
// mbSize - [in] size of the minibatch (number of frames, etc.)
// epoch - [in] epoch number for this loop
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template<class ElemType, typename LabelType>
void ConcurrentReader<ElemType, LabelType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    m_mbSize = mbSize;
    m_mbStartSample = epoch*requestedEpochSamples;
    // let readahead know how many samples in an epoch
    m_dataReader->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
    m_readahead.Init(m_mbStartSample, m_mbSize, epoch, requestedEpochSamples);
}

// GetMinibatch - Get the next minibatch (features and labels)
// features - [out] returns minibatch in passed in matrix, will resize and replace existing data. 
//      Number of columns returned may be less than requested mbSize if dataset size is smaller than requested  has been reached.
// labels - [out] returns matrix of label values as normalized integers (0-x) for a classification tasks, and will be a matrix of (0 or 1) values for regression tasks. Existing data will be replaced. 
//      in non-supervised training, labels will not be used
// returns - true if there are more minibatches, false if no more minibatchs remain
template<class ElemType, typename LabelType>
bool ConcurrentReader<ElemType, LabelType>::GetMinibatch(Matrix<ElemType>& features, Matrix<ElemType>& labels)
{
    // get the next batch from the readahead buffers
    bool ret = m_readahead.getbatch(m_mbStartSample, m_mbSize, features, labels);
    if (ret)
        m_mbStartSample += features.GetNumCols();
    return ret;
}

// GetLabelMapping - Gets the label mapping from integer index to label type 
// returns - a map from numeric datatype to native label type 
template<class ElemType, typename LabelType>
const map<unsigned, LabelType>& ConcurrentReader<ElemType, LabelType>::GetLabelMapping( )
{
    return m_dataReader->GetLabelMapping();
}

// SetLabelMapping - Sets the label mapping from integer index to label 
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
template<class ElemType, typename LabelType>
void ConcurrentReader<ElemType, LabelType>::SetLabelMapping(const std::map<unsigned, LabelType>& labelMapping)
{
    m_dataReader->SetLabelMapping(labelMapping);
}

//The explicit instantiation
//template class ConcurrentReader<double, std::wstring>; 
//template class ConcurrentReader<float, std::wstring>;
template class ConcurrentReader<float, int>;
template class ConcurrentReader<double, int>;
template class ConcurrentReader<float, float>;
template class ConcurrentReader<double, double>;

}}}