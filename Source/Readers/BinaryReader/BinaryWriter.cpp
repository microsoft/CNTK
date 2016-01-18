//
// <copyright file="BinaryWriter.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// BinaryWriter.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"
#define DATAWRITER_EXPORTS // creating the exports here
#include "DataWriter.h"
#include "BinaryReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <class ElemType>
void BinaryWriter<ElemType>::Destroy()
{
    delete this;
}

// destructor - virtual so it gets called properly
template <class ElemType>
BinaryWriter<ElemType>::~BinaryWriter()
{
    // clear the section references, they will be delted by the sectionFile destructors
    m_sections.clear();

    // delete all the sectionfiles
    for (auto pair : m_secFiles)
    {
        delete pair.second;
    }
    m_secFiles.clear();
}

// Init - Writer Initialize for multiple data sets/sections
// config - [in] configuration parameters for the datawriter
// Sample format below for BinaryWriter:
//writer=[
//  # writer to use, can implement both reader and writer
//  writerType=BinaryWriter
//  miniBatchMode=Partial
//  randomize=None
//  wfile=c:\speech\mnist\mnist_test.bin
//  #wsize - inital size of the file in MB default to 256
//  # has to be large enough for your dataset. the file will shrink to the actual size when closed.
//  #wsize=256
//  #wrecords - number of records we should allocate space for in the file
//  # files cannot be expanded, so this should be large enough. If known modify this element in config before creating file
//  wrecords=50000
//  features=[
//    dim=784
//    start=1
//    sectionType=data
//    stats=[
//      sectionType=stats
//      elementSize=32
//      compute={sum:count:mean:variance:stddev:max:min:range}
//    ]
//  ]
//  labels=[
//    dim=1
//    # sizeof(unsigned) which is the label index type
//    elementSize=4
//    wref=features
//    sectionType=labels
//    mapping=[
//      #redefine number of records for this section, since we don't need to save it for each data record
//      wrecords=10
//      #variable size so use an average string size
//      elementSize=10
//      sectionType=labelMapping
//    ]
//    category=[
//      dim=10
//      #elementSize=sizeof(ElemType) is default
//      sectionType=categoryLabels
//    ]
//      labelType=Category
//  ]
//
//]

template <class ElemType>
Section* BinaryWriter<ElemType>::CreateSection(const ScriptableObjects::IConfigRecord&, Section*, size_t, size_t)
{
    InvalidArgument("BinaryWriter currently not implemented for BrainScript.");
    // ...the reason being that the BinaryWriter needs the ConfigPath, which is also not available in BrainScript.
}

template <class ElemType>
Section* BinaryWriter<ElemType>::CreateSection(const ConfigParameters& config, Section* parentSection, size_t p_records, size_t p_windowSize)
{
    // first check if we need to open a new section file
    std::vector<std::wstring> sections;

    // determine the element size, default to ElemType size
    size_t elementSize = sizeof(ElemType);
    if (config.ExistsCurrent(L"elementSize"))
    {
        elementSize = config(L"elementSize");
    }

    // get the number of records we should expect (max)
    // if defined in previous levels same number will be used
    size_t records = p_records;
    if (config.ExistsCurrent(L"wrecords"))
    {
        records = config(L"wrecords");
    }
    if (records == 0)
    {
        InvalidArgument("Required config variable 'wrecords' missing from BinaryWriter configuration.");
    }

    size_t dim = 1; // default dimension (single item)
    if (config.ExistsCurrent(L"dim"))
    {
        dim = config(L"dim");
    }

    // get the section type (used for caching)
    SectionType sectionType = sectionTypeNull;
    if (config.ExistsCurrent(L"sectionType"))
    {
        SectionType foundType = sectionTypeNull;
        wstring type = config(L"sectionType");
        for (int i = 0; i < sectionTypeMax; i++)
        {
            if (!_wcsicmp(type.c_str(), SectionTypeStrings[i]))
            {
                foundType = SectionType(i);
                break;
            }
        }

        // check to make sure it matched something
        if (foundType == sectionTypeNull)
        {
            InvalidArgument("Invalid value for 'sectionType' in BinaryWriter configuration: %ls", type.c_str());
        }
        sectionType = foundType;
    }

    // calculate number of bytes = dim*elementSize*records
    size_t dataOnlySize = records * elementSize * dim;
    size_t dataSize = dataOnlySize + sectionHeaderMin;

    // filename to use the one defined at this level, if there is none use the parent file
    SectionFile* file = NULL;
    if (config.ExistsCurrent(L"wfile"))
    {
        std::wstring wfile = config(L"wfile");
        auto secFile = m_secFiles.find(wfile);
        if (secFile != m_secFiles.end())
        {
            file = secFile->second;
        }
        else
        {
            // TODO: sanity check and use records as a clue of how big to make it
            size_t initialSize = config(L"wsize", (size_t) 256); // default to 256MB if not provided
            initialSize *= 1024 * 1024;                          // convert MB to bytes
            if (initialSize < dataSize)
                initialSize = dataSize * 5 / 4; // make the initalSize slightly larger than needed for data
            file = new SectionFile(wfile, fileOptionsReadWrite, initialSize);
            m_secFiles[wfile] = file;
            parentSection = file->FileSection();
            parentSection->SetElementCount(records);
            parentSection->SetFileUniqueId(this->m_uniqueID);
        }
    }
    else
    { // no file defined at this config level, use parent file
        if (parentSection != NULL && parentSection->GetSectionFile() != NULL)
        {
            file = parentSection->GetSectionFile();
        }
        else if (sectionType != sectionTypeNull)
        {
            InvalidArgument("No filename (wfile) defined in BinaryWriter configuration.");
        }
    }

    // determine file position if needed
    size_t filePositionLast = 0;
    size_t filePositionNext = 0;

    if (file != NULL)
    {
        // get the next available position in the file (always on the end)
        filePositionLast = file->GetFilePositionMax();
        filePositionNext = file->RoundUp(filePositionLast);

        // we have a gap, zero it out to keep the file clean
        if (filePositionLast != filePositionNext)
        {
            size_t size = filePositionNext - filePositionLast;
            size_t roundDown = file->RoundUp(filePositionLast - file->GetViewAlignment() - 1);
            // need to get a veiw to zero out non-used bytes
            void* view = file->GetView(roundDown, file->GetViewAlignment());
            char* ptr = (char*) view + filePositionLast % file->GetViewAlignment();
            memset(ptr, 0, size);
            file->ReleaseView(view);
        }
    }

    // get the new section name
    std::string sectionName = config.ConfigName();

    // get the window size, to see if we want to do separate element mapping
    size_t windowSize = p_windowSize;
    if (config.ExistsCurrent(L"windowSize"))
    {
        windowSize = config(L"windowSize");
    }
    MappingType mappingMain = windowSize ? mappingElementWindow : mappingParent;
    MappingType mappingAux = windowSize ? mappingSection : mappingParent;

    // now create the new section
    Section* section = NULL;
    switch (sectionType)
    {
    case sectionTypeNull:
        // this happens for the original file header, nothing to do
        // also used when multiple files are defined, but none at the base level
        break;
    case sectionTypeFile: // file header
        // shouldn't occur, but same case as above
        break;
    case sectionTypeData: // data section
        section = new Section(file, parentSection, filePositionNext, mappingMain, dataSize);
        section->InitHeader(sectionTypeData, sectionName + ":Data Section", sectionDataFloat, sizeof(ElemType));
        break;
    case sectionTypeLabel: // label data
    {
        size_t elementSize = sizeof(LabelIdType);
        dataSize = records * elementSize + sectionHeaderMin;
        auto sectionLabel = new SectionLabel(file, parentSection, filePositionNext, mappingMain, dataSize);
        SectionData dataType = sectionDataInt;
        LabelKind labelKind = labelCategory; // default
        if (config.Match(L"labelType", L"Regression"))
        {
            labelKind = labelRegression;
            dataType = sectionDataFloat;
            elementSize = sizeof(ElemType);
        }
        else if (config.Match(L"labelType", L"Category"))
        {
            // everything set already, default value
        }
        else
        {
            RuntimeError("Invalid type 'labelType' or missing in BinaryWriter configuration.");
        }

        // initialize the section header
        sectionLabel->InitHeader(sectionTypeLabel, sectionName + ":Labels", dataType, (WORD) elementSize);

        // initialize the special label header items
        sectionLabel->SetLabelKind(labelKind);
        sectionLabel->SetLabelDim(config(L"labelDim"));
        section = sectionLabel;
        break;
    }
    case sectionTypeLabelMapping: // label mapping table (array of strings)
        section = new SectionString(file, parentSection, filePositionNext, mappingAux, dataSize);
        section->InitHeader(sectionTypeLabelMapping, sectionName + ":Label Map", sectionDataStrings, 0); // declare variable length strings
        section->SetFlags(flagAuxilarySection);
        section->SetFlags(flagVariableSized);
        break;
    case sectionTypeStats: // data statistics
    {
        ConfigArray calcStats = config(L"compute");
        records = calcStats.size();
        elementSize = sizeof(NumericStatistics);
        dataOnlySize = records * elementSize;
        dataSize = dataOnlySize + sectionHeaderMin;
        auto sectionStats = new SectionStats(file, parentSection, filePositionNext, mappingAux, dataSize);
        sectionStats->InitHeader(sectionTypeStats, sectionName + ":Data Statistics", sectionDataStruct, sizeof(NumericStatistics)); // declare variable length strings
        sectionStats->SetFlags(flagAuxilarySection);
        section = sectionStats;
        break;
    }
    case sectionTypeCategoryLabel:
        section = new Section(file, parentSection, filePositionNext, mappingMain, dataSize);
        section->InitHeader(sectionTypeCategoryLabel, sectionName + ":Category Labels", sectionDataFloat, sizeof(ElemType)); // declare variable length strings
        break;
    }

    // set the rest of the header variables necessary
    if (section == NULL)
    {
        // NULL or file section/already created
        section = parentSection;
    }
    else
    {
        section->SetElementSize(elementSize);
        section->SetElementsPerRecord(dim);
        section->SetElementCount(records * dim);
        section->SetSize(dataSize);
        section->SetSizeAll(dataSize);

        // windowSize is in records, convert to bytes
        size_t dataWindowSize = windowSize ? windowSize * elementSize * dim : dataOnlySize;
        // clamp it down to actual data size
        dataWindowSize = min(dataOnlySize, dataWindowSize);

        // now get the data pointer setup and allocate the view as necessary
        bool auxSection = !!(section->GetFlags() & flagAuxilarySection);
        section->EnsureElements(0, auxSection ? dataOnlySize : dataWindowSize);

        // update the max file position for the next section
        file->SetFilePositionMax(section->GetFilePosition() + dataSize);

        // Add new section to parent
        parentSection->AddSection(section);
    }

    // From here on down we have a fully usable section object

    // now find the subsections and repeat
    vector<std::wstring> subsections;
    FindConfigNames(config, "sectionType", subsections);

    // look for any children and create them as well
    for (std::wstring subsection : subsections)
    {
        CreateSection(config(subsection), section, records, windowSize);
    }

    // wait until here so everything is mapped and valid in the object
    if (sectionType == sectionTypeStats)
    {
        ConfigArray calcStats = config(L"compute");
        ((SectionStats*) section)->InitCompute(calcStats);
    }

    // add to section map
    if (sectionType != sectionTypeFile && sectionType != sectionTypeNull)
    {
        std::wstring wsectionName = msra::strfun::utf16(sectionName);
        // can't have identical names in a write configuration
        if (m_sections.find(wsectionName) != m_sections.end())
        {
            RuntimeError("Identical section name appears twice:%s", sectionName.c_str());
        }
        m_sections[wsectionName] = section;
    }

    // validate the header (make sure it's sane)
    if (section && file && !section->ValidateHeader(file->Writing()))
    {
        RuntimeError("Invalid header in file %ls, in header %ls\n", file->GetName().c_str(), section->GetName().c_str());
    }

    // return the now complete section
    return section;
}

// Init - initialize the Binary writer
// config - the configuration for the binary writer
template <class ElemType>
template <class ConfigRecordType>
void BinaryWriter<ElemType>::InitFromConfig(const ConfigRecordType& config)
{
    // initialize all the variables
    m_recordCurrent = 0;
    m_recordMax = config(L"wrecords", (size_t) 0);
    m_traceLevel = config(L"traceLevel", 0);
    m_uniqueID = (WORD) GetTickCount();

    // get the configuration, this will recursively go down and create all subfiles/sections as well
    CreateSection(config, NULL, m_recordMax);
}

// GetSections - Get the sections of the file
// sections - a map of section name to section. Data sepcifications from config file will be used to determine where and how to save data
template <class ElemType>
void BinaryWriter<ElemType>::GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections)
{
    for (auto pair : m_sections)
    {
        sections[pair.first] = pair.second->GetSectionType();
    }
}

// SaveData - save data in the file/files
// recordStart - Starting record number, will be > datasetSize for multiple passes
// matricies - a map of section name (section:subsection) to data pointer. Data sepcifications from config file will be used to determine where and how to save data
// numRecords - number of records we are saving, can be zero if not applicable
// datasetSize - Size of the dataset
// byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
// returns: true if more data is desired, false if no more data is required (signal that file may be closed)
template <class ElemType>
bool BinaryWriter<ElemType>::SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized)
{
    // allow restarting a writing session. This is used primarily for writing entire sections at once
    if (recordStart == 0)
    {
        m_recordCurrent = recordStart;
    }

    // make sure there are no gaps in the writing
    if (m_recordCurrent != recordStart)
    {
        RuntimeError("Caching with binary writer, records skip from %ld to %ld", m_recordCurrent, recordStart);
    }
    bool written = false;
    for (auto pair : m_sections)
    {
        Section* section = pair.second;
        written = section->SaveData(recordStart, matrices, numRecords, datasetSize, byteVariableSized) || written;
    }
    // the current record we expect to write next
    m_recordCurrent = recordStart + numRecords;

    return written;
}

// SaveMapping - save a map into the file
// saveId - name of the section to save into
// labelMapping - map we are saving to the file
template <class ElemType>
void BinaryWriter<ElemType>::SaveMapping(std::wstring saveId, const std::map<typename BinaryWriter<ElemType>::LabelIdType, typename BinaryWriter<ElemType>::LabelType>& labelMapping)
{
    Section* section = m_sections[saveId];
    if (section->GetSectionType() == sectionTypeLabelMapping)
    {
        auto sectionString = (SectionString*) section;
        sectionString->SetLabelMapping(labelMapping);
    }
}

// instantiate all the combinations we expect to be used
template class BinaryWriter<double>;
template class BinaryWriter<float>;
}
}
}