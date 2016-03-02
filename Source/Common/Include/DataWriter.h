//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the DATAWRITER_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// DATAWRITER_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef _WIN32
#if defined(DATAWRITER_EXPORTS)
#define DATAWRITER_API __declspec(dllexport)
#elif defined(DATAWRITER_LOCAL)
#define DATAWRITER_API
#else
#define DATAWRITER_API __declspec(dllimport)
#endif
#else
#define DATAWRITER_API
#endif

#include "Basics.h"
#include "Matrix.h"
#include "Config.h" // for ConfigParameters
#include "ScriptableObjects.h"
#include <map>
#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

// type of data in this section
enum SectionType
{
    sectionTypeNull = 0,
    sectionTypeFile = 1,          // file header
    sectionTypeData = 2,          // data section
    sectionTypeLabel = 3,         // label data
    sectionTypeLabelMapping = 4,  // label mapping table (array of strings)
    sectionTypeStats = 5,         // data statistics
    sectionTypeCategoryLabel = 6, // labels in category format (float type, all zeros with a single 1.0 per column)
    sectionTypeMax
};

// Data Writer interface
// implemented by some DataWriters
class DATAWRITER_API IDataWriter
{
public:
    typedef std::string LabelType;
    typedef unsigned int LabelIdType;

    virtual void Init(const ConfigParameters& writerConfig) = 0;
    virtual void Init(const ScriptableObjects::IConfigRecord& writerConfig) = 0;
    virtual void Destroy() = 0;
protected: public: // BUGBUG: This is accessed by a wrapper class.
    virtual ~IDataWriter() { }
public:
    virtual void GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections) = 0;
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized) = 0;
    virtual void SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& labelMapping) = 0;
    virtual bool SupportMultiUtterances() const = 0;
};
typedef std::shared_ptr<IDataWriter> IDataWriterPtr;

// GetWriter - get a reader type from the DLL
// The F version gets the 'float' version, and D gets 'double'.
extern "C" DATAWRITER_API void GetWriterF(IDataWriter** pwriter);
extern "C" DATAWRITER_API void GetWriterD(IDataWriter** pwriter);

// Data Writer class
// interface for clients of the Data Writer
// mirrors the IDataWriter interface, except the Init method is private (use the constructor)
class DataWriter : public IDataWriter, protected Plugin
{
    IDataWriter* m_dataWriter; // writer

    // Init - Writer Initialize for multiple data sets
    // config - [in] configuration parameters for the datawriter
    // Sample format below for BinaryWriter:
    // writer=[
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
    //    sectionType="data"
    //    stats=[
    //      sectionType="stats"
    //      elementSize=8
    //      compute={"sum":"count":"mean:"v"ariance":"stddev":"max":"min":"range"}
    //    ]
    //  ]
    //  labels=[
    //    dim=1
    //    # sizeof(unsigned) which is the label index type
    //    elementSize=4
    //    wref="features"
    //    sectionType="labels"
    //    mapping=[
    //      #redefine number of records for this section, since we don't need to save it for each data record
    //      wrecords=10
    //      #variable size so use an average string size
    //      elementSize=10
    //      sectionType="stringMap"
    //    ]
    //    category=[
    //      dim=10
    //      #elementSize=sizeof(ElemType) is default
    //      sectionType="categoryLabels"
    //    ]
    //    labelType="category"
    //  ]
    //]
    template <class ConfigRecordType>
    void InitFromConfig(const ConfigRecordType&);
    virtual void Init(const ConfigParameters& config) override
    {
        InitFromConfig(config);
    }
    virtual void Init(const ScriptableObjects::IConfigRecord& config) override
    {
        InitFromConfig(config);
    }

    // Destroy - cleanup and remove this class
    // NOTE: this destroys the object, and it can't be used past this point
    virtual void Destroy();

public:
    // DataWriter Constructor
    // config - [in] configuration parameters for the datareader
    template <class ConfigRecordType>
    DataWriter(const ConfigRecordType& config);
    // constructor from Scripting
    DataWriter(const ScriptableObjects::IConfigRecordPtr configp)
        : DataWriter(*configp)
    {
    }
    virtual ~DataWriter();

    virtual void GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections);

    // SaveData - save data in the file/files
    // recordStart - Starting record number
    // matricies - a map of section name (section:subsection) to data pointer. Data sepcifications from config file will be used to determine where and how to save data
    // numRecords - number of records we are saving, can be zero if not applicable
    // datasetSize - size of the dataset (in records)
    // byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized = 0);

    // SaveMapping - save a map into the file
    // saveId - name of the section to save into (section:subsection format)
    // labelMapping - map we are saving to the file
    virtual void SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& labelMapping);
    virtual bool SupportMultiUtterances() const 
    {
        return false;
    };
};

} } }
