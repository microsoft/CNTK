//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataWriter.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAWRITER_LOCAL
#include "DataWriter.h"

namespace Microsoft { namespace MSR { namespace CNTK {

static const char* GetWriterName(const string& precision)
{
    if (precision == "float")
        return "GetWriterF";
    else if (precision == "double")
        return "GetWriterD";
    else
        InvalidArgument("DataWriter: The 'precision' parameter must be 'float' or 'double'.");
}

template <class ConfigRecordType>
void DataWriter::InitFromConfig(const ConfigRecordType& /*config*/)
{
    RuntimeError("Init shouldn't be called, use constructor");
    // not implemented, calls the underlying class instead
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
void DataWriter::Destroy()
{
    m_dataWriter->Destroy();
}

// DataWriter Constructor
// config - [in] configuration data for the data writer
template <class ConfigRecordType>
DataWriter::DataWriter(const ConfigRecordType& config)
{
    typedef void (*GetWriterProc)(IDataWriter** pwriter);

    // get the name for the writer we want to use, default to BinaryWriter
    wstring writerType = config(L"writerType", L"Cntk.Reader.Binary.Deprecated");

    string precision = config(L"precision", "float");

    GetWriterProc getWriterProc = (GetWriterProc)Plugin::Load(writerType, GetWriterName(precision));
    m_dataWriter = NULL;
    getWriterProc(&m_dataWriter);

    m_dataWriter->Init(config);
}

template DataWriter::DataWriter(const ConfigParameters&);
template DataWriter::DataWriter(const ScriptableObjects::IConfigRecord&);

// destructor - cleanup temp files, etc.
DataWriter::~DataWriter()
{
    // free up resources
    if (m_dataWriter)
        m_dataWriter->Destroy();
}

// GetSections - Get the sections of the file
// sections - a map of section name to section. Data sepcifications from config file will be used to determine where and how to save data
void DataWriter::GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections)
{
    m_dataWriter->GetSections(sections);
}

// SaveData - save data in the file/files
// recordStart - Starting record number
// matricies - a map of section name (section:subsection) to data pointer. Data sepcifications from config file will be used to determine where and how to save data
// numRecords - number of records we are saving, can be zero if not applicable
// datasetSize - Size of the dataset
// byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
bool DataWriter::SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized)
{
    return m_dataWriter->SaveData(recordStart, matrices, numRecords, datasetSize, byteVariableSized);
}

// SaveMapping - save a map into the file
// saveId - name of the section to save into (section:subsection format)
// labelMapping - map we are saving to the file
void DataWriter::SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& labelMapping)
{
    m_dataWriter->SaveMapping(saveId, labelMapping);
}

}}}
