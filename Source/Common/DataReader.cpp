//
// <copyright file="DataReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#define DATAREADER_LOCAL
#include "Basics.h"
#include "DataReader.h"
#include "Config.h"
#include "ScriptableObjects.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
std::string GetReaderName(ElemType)
{
    return std::string();
}
template <>
std::string GetReaderName(float)
{
    std::string name = "GetReaderF";
    return name;
}
template <>
std::string GetReaderName(double)
{
    std::string name = "GetReaderD";
    return name;
}

template <class ElemType>
template <class ConfigRecordType>
void DataReader<ElemType>::InitFromConfig(const ConfigRecordType& /*config*/)
{
    RuntimeError("Init shouldn't be called, use constructor");
    // not implemented, calls the underlying class instead
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <class ElemType>
void DataReader<ElemType>::Destroy()
{
    /// newer code that explicitly place multiple streams for inputs
    foreach_index (i, m_ioNames) // inputNames should map to node names
    {
        m_dataReaders[m_ioNames[i]]->Destroy();
    }
}

// DataReader Constructor
// options - [in] string  of options (i.e. "-windowsize:11 -addenergy") data reader specific
template <class ElemType>
template <class ConfigRecordType>
DataReader<ElemType>::DataReader(const ConfigRecordType& config)
{
    typedef void (*GetReaderProc)(IDataReader<ElemType>** preader);

    assert(m_dataReaders.empty());

    bool hasMultipleReaders = config.Exists(L"readers");
    if (hasMultipleReaders)
    {
        vector<wstring> ioNames = config(L"readers", ConfigRecordType::Array(stringargvector()));
        // newer code that explicitly place multiple streams for inputs
        for (const auto& ioName : ioNames) // inputNames should map to node names
        {
            const ConfigRecordType& thisIO = config(ioName);
            // get the name for the reader we want to use, default to UCIFastReader
            GetReaderProc getReaderProc = (GetReaderProc) Plugin::Load(thisIO(L"readerType", L"UCIFastReader"), GetReaderName((ElemType) 0));
            m_ioNames.push_back(ioName);
            getReaderProc(&m_dataReaders[ioName]); // instantiates the reader with the default constructor (no config processed at this point)
        }
    }
    else // legacy
    {
        wstring ioName = L"ioName";
        // backward support to use only one type of data reader
        // get the name for the reader we want to use, default to UCIFastReader
        GetReaderProc getReaderProc = (GetReaderProc) Plugin::Load(config(L"readerType", L"UCIFastReader"), GetReaderName((ElemType) 0));
        m_ioNames.push_back(ioName);
        getReaderProc(&m_dataReaders[ioName]);
    }

    // now pass that to concurrent reader so we can read ahead
    //m_DataReader = new ConcurrentReader<ElemType>(m_DataReader);
    // NOW we can init
    // TODO: merge with the code above, but we first need to get the nbrUttPerMinibatch initialized inside each reader
    for (const auto& ioName : m_ioNames)
    {
        const ConfigRecordType& thisIO = hasMultipleReaders ? config(ioName) : config /*legacy*/;
        m_dataReaders[ioName]->Init(thisIO);

        // pass on some global option    --TODO: Why is this not done inside each reader??
        size_t nbrUttPerMinibatch = config(L"nbruttsineachrecurrentiter", (size_t) 1);
        m_dataReaders[ioName]->SetNumParallelSequences(nbrUttPerMinibatch);
    }
}

template DataReader<float>::DataReader(const ConfigParameters&);
template DataReader<double>::DataReader(const ConfigParameters&);
template DataReader<float>::DataReader(const ScriptableObjects::IConfigRecord&);
template DataReader<double>::DataReader(const ScriptableObjects::IConfigRecord&);

// destructor - cleanup temp files, etc.
template <class ElemType>
DataReader<ElemType>::~DataReader()
{
    // free up resources
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->Destroy();
}

// StartMinibatchLoop - Startup a minibatch loop
//  mbSize - [in] size of the minibatch (number of frames, etc.)
//  epoch - [in] epoch number for this loop
//  requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template <class ElemType>
void DataReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
}

//SupportsDistributedMBRead - Tells if the reader supports distributed minibatch reading for parallel training
template <class ElemType>
bool DataReader<ElemType>::SupportsDistributedMBRead() const
{
    bool supportsDistributedMBRead = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        auto currReaderIter = m_dataReaders.find(m_ioNames[i]);
        assert(currReaderIter != m_dataReaders.end());

        supportsDistributedMBRead &= currReaderIter->second->SupportsDistributedMBRead();
    }

    return supportsDistributedMBRead;
}

//StartDistributedMinibatchLoop - Startup a distributed minibatch loop for parallel training
// mbSize - [in] size of the minibatch (number of frames, etc.)
// epoch - [in] epoch number for this loop
// subsetNum - [in] the subset number of the current node in a group of parallel training nodes
// numSubsets - [in] total number of nodes participating in the parallel training
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template <class ElemType>
void DataReader<ElemType>::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /* = requestDataSize*/)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        m_dataReaders[m_ioNames[i]]->StartDistributedMinibatchLoop(mbSize, epoch, subsetNum, numSubsets, requestedEpochSamples);
    }
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool DataReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    bool bRet = true;
    vector<size_t> vNbrSentences;
    size_t nbr = 0;
    size_t thisNbr = 0;
    /**
    each reader reads data with number of columns as  nbr_utterances_per_minibatch * mbSize
    notice that readers may differ in their actual mbsize, though it is supposedly to be nbr_utterances_per_minibatch * mbSize.
    To handle with this, readers use their getminibatch function and then return their exact number of utterance in each minbatch. 
    This exact number, which is specified for the next reader, is passed to the next reader. 
    The next reader then returns the exact number of utterances per minibatch, after calling its getminibatch function.
    Then this returned number is compared against the specified number. If these two numbers are not consistent, return with logic error.
    The logic error can be avoided usually with an exchange of reading orders. 
    */
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        if (nbr > 0)
            m_dataReaders[m_ioNames[i]]->SetNumParallelSequences(nbr);
        bRet &= m_dataReaders[m_ioNames[i]]->GetMinibatch(matrices);
        thisNbr = m_dataReaders[m_ioNames[i]]->GetNumParallelSequences();
        if (nbr > 0 && thisNbr != nbr)
            LogicError("DataReader<ElemType>::GetMinibatch: The specified number of utterances per minibatch is not consistent to the actual number of utterances per minibatch");
        nbr = thisNbr;
    }
    return bRet;
}

// GetMinibatch4SE - Get the next minibatch for SE training, including lattice, labels and phone boundary
// latticeinput - lattice for each utterances in this minibatch
// uids - lables stored in size_t vector instead of ElemType matrix
// boundary - phone boundaries
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool DataReader<ElemType>::GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>>& latticeinput, vector<size_t>& uids, vector<size_t>& boundaries, vector<size_t>& extrauttmap)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        bRet &= m_dataReaders[m_ioNames[i]]->GetMinibatch4SE(latticeinput, uids, boundaries, extrauttmap);
    }
    return bRet;
}

// GetHmmData - Get the HMM definition for SE training
// hmm - HMM definition
// returns - true if succeed
template <class ElemType>
bool DataReader<ElemType>::GetHmmData(msra::asr::simplesenonehmm* hmm)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        bRet &= m_dataReaders[m_ioNames[i]]->GetHmmData(hmm);
    }
    return bRet;
}

template <class ElemType>
size_t DataReader<ElemType>::GetNumParallelSequences()
{
    size_t nNbr = 0;
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        IDataReader<ElemType>* ptr = m_dataReaders[m_ioNames[i]];
        if (nNbr == 0)
            nNbr = ptr->GetNumParallelSequences();
        if (nNbr != ptr->GetNumParallelSequences())
            LogicError("GetNumParallelSequences: number of slices in each minibatch not consistent for these streams");
    }
    return nNbr;
}

template <class ElemType>
bool DataReader<ElemType>::RequireSentenceSeg() const
{
    bool ans = false;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        ans = ans || m_dataReaders.find(m_ioNames[i])->second->RequireSentenceSeg(); // can't say m_dataReaders[] since that is non-const...
    return ans;
}

template <class ElemType>
void DataReader<ElemType>::InitProposals(std::map<std::wstring, Matrix<ElemType>*>* matrices)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->InitProposals(matrices);
}

template <class ElemType>
int DataReader<ElemType>::GetSentenceEndIdFromOutputLabel()
{
    int iRet = -1;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        iRet = m_dataReaders[m_ioNames[i]]->GetSentenceEndIdFromOutputLabel();
    return iRet;
}

template <class ElemType>
bool DataReader<ElemType>::GetProposalObs(std::map<std::wstring, Matrix<ElemType>*>* matrices, const size_t tidx, vector<size_t>& history)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        bRet &= m_dataReaders[m_ioNames[i]]->GetProposalObs(matrices, tidx, history);
    return bRet;
}

template <class ElemType>
void DataReader<ElemType>::CopyMBLayoutTo(MBLayoutPtr pMBLayout)
{
    // BUGBUG: This copies all data reader's layout info on top of each other, keeping only the last one; likely not what was intended.
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->CopyMBLayoutTo(pMBLayout);
}

template <class ElemType>
void DataReader<ElemType>::SetRandomSeed(int seed)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->SetRandomSeed(seed);
}

template <class ElemType>
bool DataReader<ElemType>::GetMinibatchCopy(
    std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    std::map<std::wstring, Matrix<ElemType>*>& matrices,
    MBLayoutPtr pMBLayout)
{
    bool ans = false;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        ans = (m_dataReaders[m_ioNames[i]]->GetMinibatchCopy(uttInfo, matrices, pMBLayout) || ans);
    return ans;
}

template <class ElemType>
bool DataReader<ElemType>::SetNetOutput(
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    const Matrix<ElemType>& outputs,
    const MBLayoutPtr pMBLayout)
{
    bool ans = false;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        ans = (m_dataReaders[m_ioNames[i]]->SetNetOutput(uttInfo, outputs, pMBLayout) || ans);
    return ans;
}

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
template <class ElemType>
const std::map<typename DataReader<ElemType>::LabelIdType, typename DataReader<ElemType>::LabelType>& DataReader<ElemType>::GetLabelMapping(const std::wstring&)
{
    NOT_IMPLEMENTED;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void DataReader<ElemType>::SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->SetLabelMapping(sectionName, labelMapping);
}

// GetData - Gets metadata from the specified section (into CPU memory)
// sectionName - section name to retrieve data from
// numRecords - number of records to read
// data - pointer to data buffer, if NULL, dataBufferSize will be set to size of required buffer to accomidate request
// dataBufferSize - [in] size of the databuffer in bytes
//                  [out] size of buffer filled with data
// recordStart - record to start reading from, defaults to zero (start of data)
// returns: true if data remains to be read, false if the end of data was reached
template <class ElemType>
bool DataReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        bRet &= m_dataReaders[m_ioNames[i]]->GetData(sectionName, numRecords, data, dataBufferSize, recordStart);
    return bRet;
}

template <class ElemType>
bool DataReader<ElemType>::DataEnd(EndDataType endDataType)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        bRet &= m_dataReaders[m_ioNames[i]]->DataEnd(endDataType);
    return bRet;
}

//The explicit instantiation
template class DataReader<double>;
template class DataReader<float>;

// register SGD<> with the ScriptableObject system
ScriptableObjects::ConfigurableRuntimeTypeRegister::AddFloatDouble<DataReader<float>, DataReader<double>> registerDataReaderPlugin(L"DataReaderPlugin");
} } }
