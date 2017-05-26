//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
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
#include <string>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

static const char* GetReaderName(const string& precision)
{
    if (precision == "float")
        return "GetReaderF";
    else if (precision == "double")
        return "GetReaderD";
    else
        InvalidArgument("DataReader: The 'precision' parameter must be 'float' or 'double'.");
}

void DataReaderBase::SetMinibatchLayout(StreamMinibatchInputs& minibatch)
{
    assert(minibatch.begin() != minibatch.end());

    auto& pMBLayout = minibatch.begin()->second.pMBLayout;
    // This is only allowed for old readers, which support a single layout for all inputs.
    for (const auto& iter : minibatch)
    {
        assert(iter.second.pMBLayout == pMBLayout);
        // TODO: This should be a runtime check, not an assert() that only runs in Debug.
        UNUSED(iter);
    }

    CopyMBLayoutTo(pMBLayout);
}

bool DataReaderBase::GetMinibatch(StreamMinibatchInputs& minibatch)
{
    if (TryGetMinibatch(minibatch))
    {
        SetMinibatchLayout(minibatch);
        return true;
    }

    return false;
}


template <class ConfigRecordType>
void DataReader::InitFromConfig(const ConfigRecordType& /*config*/)
{
    RuntimeError("Init shouldn't be called, use constructor");
    // not implemented, calls the underlying class instead
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
void DataReader::Destroy()
{
    // newer code that explicitly place multiple streams for inputs
    foreach_index (i, m_ioNames) // inputNames should map to node names
    {
        m_dataReaders[m_ioNames[i]]->Destroy();
    }
}

// DataReader Constructor
// options - [in] string  of options (i.e. "-windowsize:11 -addenergy") data reader specific
#pragma optimize("", off) // TODO work around potential VS2015 code optimization bug, replacing virtual- by non-virtual call in Init() below
template <class ConfigRecordType>
DataReader::DataReader(const ConfigRecordType& config)
{
    typedef void (*GetReaderProc)(IDataReader** preader);

    assert(m_dataReaders.empty());

    string precision = config(L"precision", "float");

    bool hasMultipleReaders = config.Exists(L"readers");
    // In case when deserializers are specified, use the new logic to compose them.
    bool hasDeserializers = config.Exists(L"deserializers");
    if (hasMultipleReaders)
    {
        vector<wstring> ioNames = config(L"readers", ConfigRecordType::Array(stringargvector()));
        // newer code that explicitly place multiple streams for inputs
        for (const auto& ioName : ioNames) // inputNames should map to node names
        {
            const ConfigRecordType& thisIO = config(ioName);
            wstring readerType = thisIO(L"readerType", L"Cntk.Deserializers.TextFormat");

            // get the name for the reader we want to use, default to CNTKTextFormatReader
            GetReaderProc getReaderProc = (GetReaderProc) Plugin::Load(readerType, GetReaderName(precision));
            m_ioNames.push_back(ioName);
            assert(getReaderProc != nullptr);
            getReaderProc(&m_dataReaders[ioName]); // instantiates the reader with the default constructor (no config processed at this point)
        }
    }
    else if (hasDeserializers)
    {
        wstring readerType = config(L"readerType", L"Cntk.Composite");

        // Creating Composite Data Reader that allow to combine deserializers.
        // This should be changed to link statically when SGD uses the new interfaces.
        wstring ioName = L"ioName";
        GetReaderProc getReaderProc = (GetReaderProc)Plugin::Load(readerType, GetReaderName(precision));
        m_ioNames.push_back(ioName);
        assert(getReaderProc != nullptr);
        getReaderProc(&m_dataReaders[ioName]);
    }
    else
    {
        wstring readerType = config(L"readerType", L"Cntk.Deserializers.TextFormat");
        wstring ioName = L"ioName";
        // backward support to use only one type of data reader
        // get the name for the reader we want to use, default to CNTKTextFormatReader
        GetReaderProc getReaderProc = (GetReaderProc)Plugin::Load(readerType, GetReaderName(precision));
        m_ioNames.push_back(ioName);
        assert(getReaderProc != nullptr);
        getReaderProc(&m_dataReaders[ioName]);
    }

    // now pass that to concurrent reader so we can read ahead
    // m_DataReader = new ConcurrentReader<ElemType>(m_DataReader);
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

template DataReader::DataReader(const ConfigParameters&);
template DataReader::DataReader(const ScriptableObjects::IConfigRecord&);

// destructor - cleanup temp files, etc.
DataReader::~DataReader()
{
    // free up resources
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->Destroy();
}

// StartMinibatchLoop - Startup a minibatch loop
//  mbSize - [in] size of the minibatch (number of frames, etc.)
//  epoch - [in] epoch number for this loop
//  requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
void DataReader::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
}

// Same as above but with additional information about required streams.
void DataReader::StartMinibatchLoop(size_t mbSize, size_t epoch, const std::unordered_set<InputStreamDescription>& streamDescriptions, size_t requestedEpochSamples)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->StartMinibatchLoop(mbSize, epoch, streamDescriptions, requestedEpochSamples);
}

//SupportsDistributedMBRead - Tells if the reader supports distributed minibatch reading for parallel training
bool DataReader::SupportsDistributedMBRead() const
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

//IsLegacyReader - Returns true if one of the readers is a legacy reader, false otherwise.
bool DataReader::IsLegacyReader() const
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        auto currReaderIter = m_dataReaders.find(m_ioNames[i]);
        assert(currReaderIter != m_dataReaders.end());

        if (currReaderIter->second->IsLegacyReader())
        {
            return true;
        }
    }

    return false;
}

//StartDistributedMinibatchLoop - Startup a distributed minibatch loop for parallel training
// mbSize - [in] size of the minibatch (number of frames, etc.)
// epoch - [in] epoch number for this loop
// subsetNum - [in] the subset number of the current node in a group of parallel training nodes
// numSubsets - [in] total number of nodes participating in the parallel training
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
void DataReader::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /* = requestDataSize*/)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        m_dataReaders[m_ioNames[i]]->StartDistributedMinibatchLoop(mbSize, epoch, subsetNum, numSubsets, requestedEpochSamples);
    }
}

// Same as above but with additional information about required streams.
void DataReader::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, const std::unordered_set<InputStreamDescription>& streamDescriptions, size_t requestedEpochSamples /* = requestDataSize*/)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        m_dataReaders[m_ioNames[i]]->StartDistributedMinibatchLoop(mbSize, epoch, subsetNum, numSubsets, streamDescriptions, requestedEpochSamples);
    }
}

size_t DataReader::GetCurrentSamplePosition()
{
    // BUGBUG: composition of old readers is not supported.
    // Returning just for the last reader.
    return m_dataReaders[m_ioNames.back()]->GetCurrentSamplePosition();
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
bool DataReader::GetMinibatch(StreamMinibatchInputs& matrices)
{
    /**
    each reader reads data with number of columns as  nbr_utterances_per_minibatch * mbSize
    notice that readers may differ in their actual mbsize, though it is supposedly to be nbr_utterances_per_minibatch * mbSize.
    To handle with this, readers use their getminibatch function and then return their exact number of utterance in each minbatch.
    This exact number, which is specified for the next reader, is passed to the next reader.
    The next reader then returns the exact number of utterances per minibatch, after calling its getminibatch function.
    Then this returned number is compared against the specified number. If these two numbers are not consistent, return with logic error.
    The logic error can be avoided usually with an exchange of reading orders.
    */
    bool bRet = true;
    //vector<size_t> vNbrSentences;
    size_t nbr = 0;
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        if (nbr > 0)
            m_dataReaders[m_ioNames[i]]->SetNumParallelSequences(nbr); // the first one determines the param of all others --TODO: This is flimsy.
        bRet &= m_dataReaders[m_ioNames[i]]->GetMinibatch(matrices);
        size_t thisNbr = m_dataReaders[m_ioNames[i]]->GetNumParallelSequencesForFixingBPTTMode();
        if (nbr == 0)
            nbr = thisNbr;
        else if (thisNbr != nbr)
            LogicError("DataReader::GetMinibatch: The specified number of utterances per minibatch is not consistent to the actual number of utterances per minibatch");
    }
    return bRet;
}

// GetMinibatch4SE - Get the next minibatch for SE training, including lattice, labels and phone boundary
// latticeinput - lattice for each utterances in this minibatch
// uids - lables stored in size_t vector instead of ElemType matrix
// boundary - phone boundaries
// returns - true if there are more minibatches, false if no more minibatchs remain
bool DataReader::GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>>& latticeinput, vector<size_t>& uids, vector<size_t>& boundaries, vector<size_t>& extrauttmap)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        bRet &= m_dataReaders[m_ioNames[i]]->GetMinibatch4SE(latticeinput, uids, boundaries, extrauttmap);
    return bRet;
}

// GetHmmData - Get the HMM definition for SE training
// hmm - HMM definition
// returns - true if succeed
bool DataReader::GetHmmData(msra::asr::simplesenonehmm* hmm)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        bRet &= m_dataReaders[m_ioNames[i]]->GetHmmData(hmm);
    return bRet;
}

size_t DataReader::GetNumParallelSequencesForFixingBPTTMode()
{
    size_t nNbr = 0;
    for (size_t i = 0; i < m_ioNames.size(); i++)
    {
        IDataReader* ptr = m_dataReaders[m_ioNames[i]];
        if (nNbr == 0)
            nNbr = ptr->GetNumParallelSequencesForFixingBPTTMode();
        else if (nNbr != ptr->GetNumParallelSequencesForFixingBPTTMode())
            LogicError("GetNumParallelSequences: number of slices in each minibatch not consistent for these streams");
    }
    return nNbr;
}

void DataReader::InitProposals(StreamMinibatchInputs* matrices)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->InitProposals(matrices);
}

bool DataReader::GetProposalObs(StreamMinibatchInputs* matrices, const size_t tidx, vector<size_t>& history)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        bRet &= m_dataReaders[m_ioNames[i]]->GetProposalObs(matrices, tidx, history);
    return bRet;
}

void DataReader::CopyMBLayoutTo(MBLayoutPtr pMBLayout)
{
    // BUGBUG: This copies all data reader's layout info on top of each other, keeping only the last one; likely not what was intended.
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->CopyMBLayoutTo(pMBLayout);
}

void DataReader::SetRandomSeed(int seed)
{
    for (size_t i = 0; i < m_ioNames.size(); i++)
        m_dataReaders[m_ioNames[i]]->SetRandomSeed(seed);
}

bool DataReader::GetMinibatchCopy(
    std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    StreamMinibatchInputs& matrices,
    MBLayoutPtr pMBLayout)
{
    bool ans = false;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        ans = (m_dataReaders[m_ioNames[i]]->GetMinibatchCopy(uttInfo, matrices, pMBLayout) || ans);
    return ans;
}

bool DataReader::SetNetOutput(
    const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
    const MatrixBase& outputs,
    const MBLayoutPtr pMBLayout)
{
    bool ans = false;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        ans = (m_dataReaders[m_ioNames[i]]->SetNetOutput(uttInfo, outputs, pMBLayout) || ans);
    return ans;
}

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
const std::map<typename DataReader::LabelIdType, typename DataReader::LabelType>& DataReader::GetLabelMapping(const std::wstring&)
{
    NOT_IMPLEMENTED;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
void DataReader::SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping)
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
bool DataReader::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        bRet &= m_dataReaders[m_ioNames[i]]->GetData(sectionName, numRecords, data, dataBufferSize, recordStart);
    return bRet;
}

bool DataReader::DataEnd()
{
    bool bRet = true;
    for (size_t i = 0; i < m_ioNames.size(); i++)
        bRet &= m_dataReaders[m_ioNames[i]]->DataEnd();
    return bRet;
}

// register SGD<> with the ScriptableObject system
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<DataReader> registerDataReaderPlugin(L"DataReaderPlugin");

}}}
