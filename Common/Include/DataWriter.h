//
// <copyright file="DataWriter.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the DATAWRITER_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// DATAWRITER_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#if defined(DATAWRITER_EXPORTS)
#define DATAWRITER_API __declspec(dllexport)
#elif defined(DATAWRITER_LOCAL)
#define DATAWRITER_API
#else
#define DATAWRITER_API __declspec(dllimport)
#endif
#include "Matrix.h"
#include <map>
#include <string>
#include "basetypes.h"
#include "commandArgUtil.h"


namespace Microsoft { namespace MSR { namespace CNTK {

// type of data in this section
enum SectionType
{
    sectionTypeNull = 0,
    sectionTypeFile = 1, // file header
    sectionTypeData = 2, // data section
    sectionTypeLabel = 3, // label data
    sectionTypeLabelMapping = 4, // label mapping table (array of strings)
    sectionTypeStats = 5, // data statistics
    sectionTypeCategoryLabel = 6, // labels in category format (float type, all zeros with a single 1.0 per column)
    sectionTypeMax
};

// Data Writer interface
// implemented by some DataWriters
template<class ElemType>
class DATAWRITER_API IDataWriter
{
public:
    typedef std::string LabelType;
    typedef unsigned LabelIdType;

    virtual void Init(const ConfigParameters& config) = 0;
    virtual void Destroy() = 0;
    virtual void GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections) = 0;
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized) = 0;
    virtual void SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& labelMapping) = 0;
};


// GetWriter - get a reader type from the DLL
// since we have 2 writerr types based on template parameters, exposes 2 exports
// could be done directly the templated name, but that requires mangled C++ names
template<class ElemType>
void DATAWRITER_API GetWriter(IDataWriter<ElemType>** pwriter);
extern "C" DATAWRITER_API void GetWriterF(IDataWriter<float>** pwriter);
extern "C" DATAWRITER_API void GetWriterD(IDataWriter<double>** pwriter);

// Data Writer class
// interface for clients of the Data Writer
// mirrors the IDataWriter interface, except the Init method is private (use the constructor)
template<class ElemType>
class DataWriter : public IDataWriter<ElemType>, public Plugin
{
private:
    IDataWriter<ElemType> *m_dataWriter;  // writer

    // Init - Writer Initialize for multiple data sets
    // config - [in] configuration parameters for the datawriter
    // Sample format below for BinaryWriter:
    //writer=[
    //  # writer to use, can implement both reader and writer
    //  writerType=BinaryWriter
    //  miniBatchMode=Partial
    //  randomize=None
    //  wfile=c:\speech\mnist\mnist_test.bin
    //  #wrecords - number of records we should allocate space for in the file
    //  # files cannot be expanded, so this should be large enough. If known modify this element in config before creating file
    //  wrecords=50000
    //  features=[
    //    dim=784
    //    start=1
    //    sectionType=data
    //    stats=[
    //      sectionType=stats
    //      elementSize=8
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
    //      sectionType=stringMap
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
    virtual void Init(const ConfigParameters& config);

    void GetDataWriter(const ConfigParameters& config);

    // Destroy - cleanup and remove this class
    // NOTE: this destroys the object, and it can't be used past this point
    virtual void Destroy();

public:
    // DataWriter Constructor
    // config - [in] configuration parameters for the datareader 
    DataWriter(const ConfigParameters& config);
    virtual ~DataWriter();

    virtual void GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections);

    // SaveData - save data in the file/files 
    // recordStart - Starting record number
    // matricies - a map of section name (section:subsection) to data pointer. Data sepcifications from config file will be used to determine where and how to save data
    // numRecords - number of records we are saving, can be zero if not applicable
    // datasetSize - size of the dataset (in records)
    // byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized=0);

    // SaveMapping - save a map into the file
    // saveId - name of the section to save into (section:subsection format)
    // labelMapping - map we are saving to the file
    virtual void SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& labelMapping);
};

}}}
