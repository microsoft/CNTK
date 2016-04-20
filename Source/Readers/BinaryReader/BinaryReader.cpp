//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// BinaryReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "BinaryReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <class ElemType>
void BinaryReader<ElemType>::Destroy()
{
    delete this;
}

// LoadSections - Load in all the sections in the file
// sectionRoot - section whose children we will enumrate
// NOTE: called recursively to obtain full section heirarchy
template <class ElemType>
void BinaryReader<ElemType>::LoadSections(Section* sectionRoot, MappingType mapping, size_t windowSize)
{
    int sectionCount = sectionRoot->GetSectionCount();
    for (int i = 0; i < sectionCount; ++i)
    {
        Section* newSection = sectionRoot->ReadSection(i, mapping, windowSize);
        auto found = m_sections.find(newSection->GetName());
        if (found != m_sections.end())
        {
            RuntimeError("LoadSections, duplicate section name %ls already defined previously", newSection->GetName().c_str());
        }
        m_sections[newSection->GetName()] = newSection;
        // load in any subsections
        LoadSections(newSection, mapping, windowSize);
    }
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
// Sample format below:
//# Parameter values for the reader
//reader=[
//  # reader to use
//  readerType=BinaryReader
//  miniBatchMode=Partial
//  file={,
//    c:\speech\mnist\mnist_features.bin
//      c:\speech\mnist\mnist_labels.bin
//  }
//]

static vector<wstring> GetCommaSeparatedItems(const ConfigParameters& config, wstring key)
{
    ConfigArray files(config(key), ',');
    return (stringargvector) files;
}
static vector<wstring> GetCommaSeparatedItems(const ScriptableObjects::IConfigRecord& config, wstring key)
{
    return config(key, vector<wstring>());
}

template <class ElemType>
template <class ConfigRecordType>
void BinaryReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    // load in all the file config info for the root level
    std::vector<std::wstring> fileConfigs;
    vector<wstring> files = GetCommaSeparatedItems(readerConfig, L"file");
    mOneLinePerFile = false;
    mOneLinePerFile = readerConfig(L"onelineperfile", false);
    m_dim = readerConfig(L"dim", (size_t) 20);
    m_totalSamples = 0;
    size_t windowSize = readerConfig(L"windowSize", (size_t) 0);
    MappingType mapping = windowSize ? mappingElementWindow : mappingSection;

    if (mOneLinePerFile)
    {
        for (int i = 0; i < files.size(); ++i)
        {
            FILE* secFile = _wfopen(files[i].c_str(), L"rt");
            if (secFile != nullptr)
                m_fStream.push_back(secFile);
            else
                LogicError("BinaryReader::init cannot find %ls to read", files[i].c_str());
        }
    }
    else
    {
        for (int i = 0; i < files.size(); ++i)
        {
            SectionFile* secFile = new SectionFile(files[i], fileOptionsRead, 0);
            size_t records = secFile->FileSection()->GetRecordCount();

            // if we haven't set the total records yet, set it
            SectionFlags flags = secFile->FileSection()->GetFlags();
            if (m_totalSamples == 0 && !(flags & flagAuxilarySection))
            {
                m_totalSamples = records;
            }
            else // otherwise we want to check to make sure it's the same
            {
                if (records != m_totalSamples && !(flags & flagAuxilarySection))
                    RuntimeError("multiple files have different record counts, cannot be used together!");
            }

            m_secFiles.push_back(secFile);

            // now get all the sections out of the file
            Section* section = secFile->FileSection();
            LoadSections(section, mapping, windowSize);
        }
    }
    // initialize all the variables
    m_mbStartSample = m_epoch = m_epochStartSample = 0;
    m_partialMinibatch = false;
    m_traceLevel = readerConfig(L"traceLevel", 0);

    // determine if partial minibatches are desired
    std::string minibatchMode(readerConfig(L"minibatchMode", "Partial"));
    m_partialMinibatch = EqualCI(minibatchMode, "Partial");

    // Initial load is complete
    DisplayProperties();
}

template <class ElemType>
void BinaryReader<ElemType>::DisplayProperties()
{
    if (m_traceLevel == 0)
        return;

    fprintf(stderr, "BinaryReader in use...\n");
    // enumerate the files
    for (SectionFile* file : m_secFiles)
    {
        Section* section = file->FileSection();
        fprintf(stderr, "File: %ls, Records: %lld\n", file->GetName().c_str(), section->GetElementCount());
    }

    for (auto pair : m_sections)
    {
        Section* section = pair.second;
        fprintf(stderr, "Section: %ls, Elements: %zd, ElementsPerRecord: %zd, ElementSize: %zd\n", pair.first.c_str(), section->GetElementCount(), section->GetElementsPerRecord(), section->GetElementSize());
        if (section->GetSectionType() == sectionTypeStats)
        {
            vector<NumericStatistics> stats;
            stats.resize(section->GetElementCount());
            size_t size = sizeof(NumericStatistics) * stats.size();
            GetData(pair.first, stats.size(), &stats[0], size);
            fprintf(stderr, "  *Stats*: ");
            for (NumericStatistics stat : stats)
            {
                fprintf(stderr, "%s:%lf, ", stat.statistic, stat.value);
            }
            fprintf(stderr, "\n");
        }
    }
}

// destructor - virtual so it gets called properly
template <class ElemType>
BinaryReader<ElemType>::~BinaryReader()
{
    // clear the section references, they will be deleted by the sectionFile destructors
    m_sections.clear();

    for (SectionFile* secFile : m_secFiles)
    {
        delete secFile;
    }
    m_secFiles.clear();

    for (size_t i = 0; i < m_fStream.size(); i++)
    {
        // TODO: Check for error code and throw if !std::uncaught_exception()
        fclose(m_fStream[i]);
    }
}

//StartMinibatchLoop - Startup a minibatch loop
// mbSize - [in] size of the minibatch (number of Samples, etc.)
// epoch - [in] epoch number for this loop, if > 0 the requestedEpochSamples must be specified (unless epoch zero was completed this run)
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
//   this value must be a multiple of mbSize, if it is not, it will be rounded up to one.
template <class ElemType>
void BinaryReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    m_mbSize = mbSize;
    if (requestedEpochSamples == requestDataSize)
    {
        // if they want the dataset size, get the first file, and return the dataset size
        if (m_totalSamples == 0)
            RuntimeError("no section files contain total samples, can't determine dataset size");
        requestedEpochSamples = m_totalSamples;
    }
    m_epochSize = requestedEpochSamples;
    m_epoch = epoch;

    SetupEpoch();
}

// CheckEndDataset - Check to see if we have arrived at the end of the dataset
// actualmbsize - [in] the actual size of the dataset we are requesting,
// returns - true if there we hit dataset end, false otherwise
template <class ElemType>
bool BinaryReader<ElemType>::CheckEndDataset(size_t actualmbsize)
{
    size_t epochEnd = m_epochSize;
    size_t epochSample = m_mbStartSample % m_epochSize;

    // check for an odd sized last minibatch
    if (epochSample + actualmbsize > epochEnd)
    {
        actualmbsize = epochEnd - epochSample;
    }

    // make sure we take into account hitting the end of the dataset (not wrapping around)
    actualmbsize = min(m_totalSamples - (m_mbStartSample % m_totalSamples), actualmbsize);

    if (actualmbsize == 0)
    {
        return true;
    }

    // if they don't want partial minibatches, skip and return
    if (actualmbsize < m_mbSize && !m_partialMinibatch)
    {
        m_mbStartSample += actualmbsize;
        return true;
    }
    return false;
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool BinaryReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
{
    // get out if they didn't call StartMinibatchLoop() first
    if (m_mbSize == 0)
        return false;

    // check to see if we have changed epochs, if so we are done with this one.
    if (m_mbStartSample / m_epochSize != m_epoch)
    {
        return false;
    }

    // actual size is either what requested, or total number of samples read so far
    size_t actualmbsize = min(m_totalSamples, m_mbSize); // it may still return less if at end of sweep
    size_t epochStartSample = m_mbStartSample % m_totalSamples;

    bool endOfDataset = CheckEndDataset(actualmbsize);
    if (endOfDataset)
        return false;

    for (auto value : matrices)
    {
        const auto& matrixName = value.first;
        auto& gpuData = matrices.GetInputMatrix<ElemType>(matrixName);
        Section* section = m_sections[matrixName];
        size_t rows = section->GetElementsPerRecord();
        SectionData dataType;
        size_t dataSize;
        section->GetDataTypeSize(dataType, dataSize);

        // if the data types are not as expected
        if (dataType != sectionDataFloat || dataSize != sizeof(ElemType))
        {
            // if it's a label type, may need to get a subsection
            if (section->GetSectionType() == sectionTypeLabel)
            {
                bool categoryLabelFound = false;
                SectionLabel* sectionLabel = (SectionLabel*) section;
                // category type, check to see if we saved it
                if (sectionLabel->GetLabelKind() == labelCategory)
                {
                    Section* sectionCategory = NULL;
                    for (int i = 0; i < section->GetSectionCount(); ++i)
                    {
                        sectionCategory = section->ReadSection(i);

                        // found the category labels, so pass them on
                        if (sectionCategory->GetSectionType() == sectionTypeCategoryLabel)
                        {
                            section = sectionCategory;
                            rows = section->GetElementsPerRecord();
                            section->GetDataTypeSize(dataType, dataSize);
                            categoryLabelFound = true;
                            break;
                        }
                    }
                    if (!categoryLabelFound)
                    {
                        RuntimeError("Category Labels not saved in file, either save, or support creation in BinaryReader");
                    }
                }
            }

            // make sure we are good now
            if (dataType != sectionDataFloat || dataSize != sizeof(ElemType))
            {
                RuntimeError("Category Labels not saved in file, either save, or support creation in BinaryReader");
            }
        }
        size_t size = rows * dataSize * actualmbsize;
        size_t index = epochStartSample * section->GetElementsPerRecord();
        ElemType* data = (ElemType*) section->EnsureElements(index, size);
        // ElemType* data = section->GetElement<ElemType>(epochStartSample*section->GetElementsPerRecord());
        // data = (ElemType*)section->EnsureMapped(data, size);

        // make sure that the data is as expected
        if (!!(section->GetFlags() & flagAuxilarySection) || section->GetElementSize() != sizeof(ElemType))
        {
            RuntimeError("GetMinibatch: Section %ls Auxilary section specified, and/or element size %lld mismatch", section->GetName().c_str(), section->GetElementSize());
        }
        gpuData.SetValue(rows, actualmbsize, gpuData.GetDeviceId(), data);
    }

    // advance to the next minibatch
    m_mbStartSample += actualmbsize;

    // we read some records, so process them
    return true;
}

//SetupEpoch - Setup the proper position in the file, and other variable settings to start a particular epoch
template <class ElemType>
void BinaryReader<ElemType>::SetupEpoch()
{
    // if we are starting fresh (epoch zero and no data read), init everything
    assert(m_totalSamples != 0);

    // make sure we are in the correct location for mid-dataset epochs
    size_t mbStartSample = m_epoch * m_epochSize;

    size_t fileRecord = mbStartSample % m_totalSamples;
    fprintf(stderr, "starting epoch %lld at record count %lld, and file position %lld\n", m_epoch, mbStartSample, fileRecord);

    // reset the original read sample
    m_epochStartSample = m_mbStartSample = mbStartSample;
}

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
template <class ElemType>
const std::map<typename BinaryReader<ElemType>::LabelIdType, typename BinaryReader<ElemType>::LabelType>& BinaryReader<ElemType>::GetLabelMapping(const std::wstring& sectionName)
{
    auto iter = m_sections.find(sectionName);
    if (iter == m_sections.end())
    {
        RuntimeError("GetLabelMapping: requested section name not found\n");
    }
    Section* section = iter->second;
    if (section->GetSectionType() != sectionTypeLabelMapping)
    {
        RuntimeError("section specified is not label mapping section.\n");
    }

    // get it from the correct section
    SectionString* sectionLabelMapping = (SectionString*) section;
    return sectionLabelMapping->GetLabelMapping();
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void BinaryReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<typename BinaryReader<ElemType>::LabelIdType, typename BinaryReader<ElemType>::LabelType>& /*labelMapping*/)
{
    RuntimeError("Binary reader does not support setting the mapping table.\n");
}

// GetData - Gets metadata from the specified section (into CPU memory)
// sectionName - section name to retrieve data from
// numRecords - number of records to read
// data - pointer to data buffer, if NULL, dataBufferSize will be set to size of required buffer to accomidate request
// dataBufferSize - [in] size of the databuffer in bytes
//                  [out] size of buffer filled with data
// recordStart - record to start reading from, defaults to zero (start of data)
// returns: true if data remains to be read, false if the end of data was reached, or buffer is insufficent
template <class ElemType>
bool BinaryReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    auto iter = m_sections.find(sectionName);
    if (iter == m_sections.end())
    {
        RuntimeError("GetData: requested section name not found\n");
    }
    Section* section = iter->second;
    size_t sizeData = section->GetElementsPerRecord() * section->GetElementSize();
    if ((numRecords + recordStart) * section->GetElementsPerRecord() > section->GetElementCount())
    {
        RuntimeError("GetData: requested invalid record number\n");
    }
    sizeData *= numRecords;
    // of buffer isn't large enough, or they didn't pass one
    if (sizeData > dataBufferSize || data == NULL)
    {
        dataBufferSize = sizeData;
        return false;
    }
    char* copyFrom = section->EnsureElements(section->GetElementsPerRecord() * recordStart, sizeData);
    memcpy_s((char*) data, dataBufferSize, copyFrom, sizeData);
    return true;
}

template <class ElemType>
bool BinaryReader<ElemType>::DataEnd() { return true; }

// instantiate all the combinations we expect to be used
template class BinaryReader<double>;
template class BinaryReader<float>;

}}}
