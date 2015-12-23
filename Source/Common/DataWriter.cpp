//
// <copyright file="DataWriter.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// DataWriter.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAWRITER_LOCAL
#include "DataWriter.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
std::string GetWriterName(ElemType)
{std::string empty; return empty;}

template<> std::string GetWriterName(float) {std::string name = "GetWriterF"; return name;}
template<> std::string GetWriterName(double) {std::string name = "GetWriterD"; return name;}

template<class ElemType>
template<class ConfigRecordType>
void DataWriter<ElemType>::InitFromConfig(const ConfigRecordType & /*config*/)
{
    RuntimeError("Init shouldn't be called, use constructor");
    // not implemented, calls the underlying class instead
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template<class ElemType>
void DataWriter<ElemType>::Destroy()
{
    m_dataWriter->Destroy();
}

// DataWriter Constructor
// config - [in] configuration data for the data writer
template<class ElemType>
template<class ConfigRecordType>
DataWriter<ElemType>::DataWriter(const ConfigRecordType & config)
{
    typedef void(*GetWriterProc)(IDataWriter<ElemType>** pwriter);

    // initialize just in case
    m_dataWriter = NULL;

    // get the name for the writer we want to use, default to BinaryWriter (which is in BinaryReader.dll)
    // TODO: This seems like a find-replace operation?
    wstring writerType = config(L"writerType", L"BinaryReader");
    if (writerType == L"HTKMLFWriter" || writerType == L"HTKMLFReader")
    {
        writerType = L"HTKMLFReader";
    }
    else if (writerType == L"BinaryWriter" || writerType == L"BinaryReader")
    {
        writerType = L"BinaryReader";
    }
    else if (writerType == L"LUSequenceWriter" || writerType == L"LUSequenceReader")
    {
        writerType = L"LUSequenceReader";
    }
    else if (writerType == L"LMSequenceWriter" || writerType == L"LMSequenceReader")
    {
        writerType = L"LMSequenceReader";
    }
    else if (writerType == L"KaldiReader" || writerType == L"KaldiWriter")
    {
        writerType = L"KaldiReader";
    }

    ElemType elemType = ElemType();
    GetWriterProc getWriterProc = (GetWriterProc)Plugin::Load(writerType, GetWriterName(elemType));
    getWriterProc(&m_dataWriter);

    m_dataWriter->Init(config);
}

template DataWriter<float >::DataWriter(const ConfigParameters &);
template DataWriter<double>::DataWriter(const ConfigParameters &);
template DataWriter<float >::DataWriter(const ScriptableObjects::IConfigRecord &);
template DataWriter<double>::DataWriter(const ScriptableObjects::IConfigRecord &);

// destructor - cleanup temp files, etc. 
template<class ElemType>
DataWriter<ElemType>::~DataWriter()
{
    // free up resources
    if (m_dataWriter)
        m_dataWriter->Destroy();
}

// GetSections - Get the sections of the file
// sections - a map of section name to section. Data sepcifications from config file will be used to determine where and how to save data
template<class ElemType>
void DataWriter<ElemType>::GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections)
{
    m_dataWriter->GetSections(sections);
}

// SaveData - save data in the file/files 
// recordStart - Starting record number
// matricies - a map of section name (section:subsection) to data pointer. Data sepcifications from config file will be used to determine where and how to save data
// numRecords - number of records we are saving, can be zero if not applicable
// datasetSize - Size of the dataset
// byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
template<class ElemType>
bool DataWriter<ElemType>::SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized)
{
    return m_dataWriter->SaveData(recordStart, matrices, numRecords, datasetSize, byteVariableSized);
}

// SaveMapping - save a map into the file
// saveId - name of the section to save into (section:subsection format)
// labelMapping - map we are saving to the file
template<class ElemType>
void DataWriter<ElemType>::SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& labelMapping)
{
    m_dataWriter->SaveMapping(saveId, labelMapping);
}

//The explicit instantiation
template class DataWriter<double>; 
template class DataWriter<float>;

}}}
