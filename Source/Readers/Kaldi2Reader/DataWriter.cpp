//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataWriter.cpp : Defines the exported functions for the DLL application.
//
// TODO: This is similar but not identical to Common/DataWriter.cpp. Why is this DataWriter different? Can it be reconciled?
//

#include "stdafx.h"
#include "Basics.h"

#include "htkfeatio.h" // for reading HTK features

#define DATAWRITER_EXPORTS
#include "DataWriter.h"
#include "HTKMLFWriter.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: move these to Exports.cpp
extern "C" DATAWRITER_API void GetWriterF(IDataWriter** pwriter)
{
    *pwriter = new HTKMLFWriter<float>();
}
extern "C" DATAWRITER_API void GetWriterD(IDataWriter** pwriter)
{
    *pwriter = new HTKMLFWriter<double>();
}

template <class ConfigRecordType>
void DataWriter::InitFromConfig(const ConfigRecordType& writerConfig)
{
    wstring precision = writerConfig(L"precision", L"float");
    if (precision == L"float")
        m_dataWriter = new HTKMLFWriter<float>();
    else if (precision == L"double")
        m_dataWriter = new HTKMLFWriter<double>();
    else
        InvalidArgument("DataWriter (Kaldi HTKMLFWriter): The 'precision' parameter must be 'float' or 'double'.");

    m_dataWriter->Init(writerConfig);
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
void DataWriter::Destroy()
{
    delete m_dataWriter;
}

// DataWriter Constructor
// config - [in] configuration data for the data writer
template <class ConfigRecordType>
DataWriter::DataWriter(const ConfigRecordType& config)
{
    Init(config);
}

// destructor - cleanup temp files, etc.
DataWriter::~DataWriter()
{
    Destroy();
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
