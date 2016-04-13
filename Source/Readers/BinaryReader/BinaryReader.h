//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// BinaryReader.h - Include file for the MTK and MLF format of features and samples
//
#pragma once
#include "DataReader.h"
#include "DataWriter.h"
#include "Config.h"
#include <string>
#include <map>
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

enum SectionData
{
    // section data type
    sectionDataNone = 0,    // no data in this section
    sectionDataFloat = 1,   // floating point values (single or double)
    sectionDataStrings = 2, // string values in a mapping table format
    sectionDataInt = 3,     // integer values (short, long, size_t)
    sectionDataStruct = 4,  // custom structure
    sectionDataMax
};

static const wchar_t* SectionTypeStrings[sectionTypeMax] =
    {
        L"Null",
        L"File",           // file header
        L"Data",           // data section
        L"Labels",         // label data
        L"LabelMapping",   // label mapping table (array of strings)
        L"Stats",          // data statistics
        L"CategoryLabels", // labels in category format (float type, all zeros with a single 1.0 per column)
};

enum CustomStructure
{
    customStructureNone = 0,  // not a custom structure
    customStructureStats = 1, // custom structure to store statistics
    customStructureMax
};

// flags for this section, bitflags
enum SectionFlags
{
    flagNone = 0,
    flagRandomized = 1,      // dataset has already been randomized
    flagAuxilarySection = 2, // a section that contains non record based data (i.e. stats, mapping table, etc)
    flagVariableSized = 4,   // a section that contains variable sized data (i.e. strings)
    flagMax
};

struct NumericStatistics
{
    char statistic[24]; // string identifier for the statistic (i.e. average, stddev, etc.)
    double value;
};

// magic numbers for headers
const WORD magicFile = 0xACE9;       // file header, only expected at beginning of file
const WORD magicSection = 0x4ACE;    // section headers, all other header types
const WORD magicIncomplete = 0xBAD1; // use a section header for the file that isn't valid until we close it properly

// get offset in a structure macro
#define offset(class, member) ((size_t) & (((class*) 0)->member))
const size_t descriptionSize = 128;

#pragma warning(push)
#pragma warning(disable : 4200) // warning C4200: nonstandard extension used : zero-sized array in struct/union
// Section Header on Disk
struct SectionHeader
{
    WORD wMagic;            // magic number ACE9
    WORD version;           // version number ##.## in hex
    WORD sizeHeader;        // size of this header (rounded up to mappable boundary)
    WORD dataSections;      // number of data sub-sections (for nesting)
    WORD sectionType;       // what is the type of the data in this section
    WORD sectionData;       // type of section (SectionData enum)
    WORD bytesPerElement;   // number of bytes per element, (0 means variable)
    WORD customStructureID; // ID for custom structure
    WORD elementsPerRecord; // number of elements per Record
    WORD flags;             // bit flags, dependent on sectionType
    WORD writtenID;         // unique ID so files written at the same time can be identified
    WORD unusedWords[5];
    size_t elementsCount; // number of total elements stored
    // * section specific data goes below here * //
    WORD labelKind;                                                    // kind of label (LabelKind type)
    WORD labelDim;                                                     // number of possible states for labels (category type)
    char unused[descriptionSize - 18 * sizeof(WORD) - sizeof(size_t)]; // space for future expansion (zero out in current versions)
    char nameDescription[descriptionSize];                             // name and description of section contents in this format (name: description) (string, with extra bytes zeroed out, at least one null terminator required)
    size_t size;                                                       // size of this section (including header)
    size_t sizeAll;                                                    // size of this section (including header and all sub-sections)
    size_t sectionFilePosition[];                                      // sub-section file offsets (if needed), assumed to be in File Position order
};
#pragma warning(pop)

const int sectionHeaderMin = ((sizeof(SectionHeader) + 64 - 1) / 64) * 64;

enum LabelKind
{
    labelNone = 0,       // no labels to worry about
    labelCategory = 1,   // category labels, creates mapping tables
    labelRegression = 2, // regression labels
    labelOther = 3,      // some other type of label
};

// store the position and size of a view
struct ViewPosition
{
    void* view;
    size_t filePosition;
    size_t size;
    int refCount;
    ViewPosition()
        : view(NULL), filePosition(0), size(0), refCount(0)
    {
    }
    ViewPosition(void* view, size_t filePosition, size_t size)
        : view(view), filePosition(filePosition), size(size), refCount(0)
    {
    }
};

// BinaryFile - class that will read/write a Binary file to a local or network path
// for local paths, the disk file will be memory mapped for best performance
// if a network path is used, it still works fine, but consistency between processes is not guaranteed
class BinaryFile
{
protected:
    HANDLE m_hndFile;         // handle to the file
    HANDLE m_hndMapped;       // handle to the mapped file object
    size_t m_mappedSize;      // size of mapped file (zero for size of file being read)
    size_t m_maxViewSize;     // maximum size we want a single view to contain
    size_t m_viewAlignment;   // address alignment required by views
    size_t m_filePositionMax; // current maximum file position in the file
    bool m_writeFile;
    std::wstring m_name;          // name of this
    vector<ViewPosition> m_views; // keep track of all the views into the file
public:
    BinaryFile(std::wstring fileName, FileOptions options = fileOptionsRead, size_t size = 0);
    virtual ~BinaryFile();
    void* GetView(size_t filePosition, size_t size);
    void ReleaseView(void* view);
    vector<ViewPosition>::iterator NotFound()
    {
        return m_views.end();
    }
    vector<ViewPosition>::iterator FindView(void* view);
    vector<ViewPosition>::iterator FindView(size_t filePosition);
    vector<ViewPosition>::iterator FindDataView(void* data);
    vector<ViewPosition>::iterator ReleaseView(vector<ViewPosition>::iterator iter, bool force = false);
    void* ReallocateView(void* view, size_t size);
    void* EnsureViewSize(void* view, size_t size);
    void* EnsureMapped(void* data, size_t size);
    vector<ViewPosition>::iterator Mapped(size_t filePosition, size_t& size);
    size_t RoundUp(size_t filePosition);
    size_t GetViewAlignment()
    {
        return m_viewAlignment;
    }
    bool Writing()
    {
        return m_writeFile;
    }
    size_t GetFilePositionMax()
    {
        return m_filePositionMax;
    }
    void SetFilePositionMax(size_t filePositionMax);
    const std::wstring& GetName()
    {
        return m_name;
    }
};

class SectionFile;

enum MappingType
{
    mappingNone = 0,          // not a mapped file
    mappingParent = 1,        // parent owns view mapping for this section
    mappingFile = 2,          // map the entire file in one view
    mappingSectionAll = 3,    // each section has it's own view (and all subsections)
    mappingSection = 4,       // each section has it's own view (but doesn't include subsections)
    mappingElementWindow = 5, // each subsection has it's own view, and this section has a separate view window for elements
};

// Section - Describes a section of the file
// This class keeps track of the current mapping of the memory mapped file
// It also wraps the headers and creates an easier way to navagate the section tree
class Section
{
protected:
    SectionFile* m_file;            // file we belong to
    Section* m_parent;              // parent pointer
    SectionHeader* m_sectionHeader; // always mapped valid header
    size_t m_filePosition;          // filePosition of the header (section)
    MappingType m_mappingType;      // type of mapping we desire for this section
    size_t m_mappedSize;            // size of the current mapping (if parent mapped, size from the beginning of this header)
    void* m_elementBuffer;          // pointer to the beginning of the buffer, may not be mapped for mappingElementWindow mappingType

    // extra variables needed only for mappingElementWindow mapping
    void* m_elementView;        // pointer to beginning of the valid view
    size_t m_mappedElementSize; // size of mapped element view, only used for mappingElementWindow mappingType
    size_t m_elemMin;           // currently mapped minimum value (first valid)
    size_t m_elemMax;           // currently mapped maximum value (one past last valid)
    bool CheckBounds(size_t index)
    {
        return index >= m_elemMin && index < m_elemMax;
    }

    // vector of sub-sections of this section
    vector<Section*> m_sections; // sub-sections of this section
    std::wstring m_name;         // name of this section, if empty use first part of description (before colon)

    Section* SectionMappingOwner();
    void EnsureMappingSize(size_t size); // ensure the appropriate view is at least this size
    void RemapHeader(void* view, size_t filePosition);
    char* GetElementBuffer(size_t element = 0, size_t windowSize = 0);
    void Init(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType, size_t size);

public:
    Section(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType = mappingParent, size_t size = 0);
    Section(SectionFile* file, Section* parentSection, size_t filePosition, MappingType mappingType = mappingParent, size_t size = 0);
    virtual ~Section();

    void InitHeader(SectionType type, std::string description, SectionData dataType = sectionDataFloat, WORD elementSize = sizeof(float));
    bool ValidateHeader(bool writing = false) const; // ensure header is sane, if not, throw an exception

    // element accessors
    template <typename T>
    T& operator[](size_t index)
    {
        return *(T*) GetElement(index);
    }
    template <typename T>
    T* GetElement(size_t index) const
    {
        return (T*) (GetElement(index));
    }
    virtual void* GetElement(size_t index) const
    {
        return (char*) m_elementBuffer + index * GetElementSize();
    }
    virtual char* EnsureElements(size_t element, size_t bytesRequested = 0);

    SectionHeader* GetSectionHeader(size_t filePosition, MappingType& mappingType, size_t& size);

    void* EnsureMapped(void* start, size_t size);
    // subsection functions
    Section* ReadSection(size_t index, MappingType mappingType = mappingParent, size_t size = 0);
    Section* AddSection(Section* sectionToAdd);

    // accessors
    SectionFile* GetSectionFile() const
    {
        return m_file;
    }
    SectionHeader* GetHeader() const
    {
        return m_sectionHeader;
    }
    size_t GetFilePosition() const
    {
        return m_filePosition;
    }
    void SetFilePosition(size_t filePosition)
    {
        m_filePosition = filePosition;
    }
    MappingType GetMappingType() const
    {
        return m_mappingType;
    }
    void SetMappingType(MappingType mappingType)
    {
        m_mappingType = mappingType;
    }
    size_t GetSize() const
    {
        return m_sectionHeader->size;
    }
    void SetSize(size_t size)
    {
        m_sectionHeader->size = size;
    }
    size_t GetSizeAll() const
    {
        return m_sectionHeader->sizeAll;
    }
    void SetSizeAll(size_t size)
    {
        m_sectionHeader->sizeAll = size;
    }
    size_t GetHeaderSize() const
    {
        return m_sectionHeader->sizeHeader;
    }
    void SetHeaderSize(size_t size)
    {
        assert(size < 0x10000);
        m_sectionHeader->sizeHeader = (WORD) size;
    }
    size_t GetMappedSize() const
    {
        return m_mappedSize;
    }
    void SetMappedSize(size_t mappedSize)
    {
        m_mappedSize = mappedSize;
    }
    void SetDescription(const std::string& description)
    {
        assert(description.size() < _countof(m_sectionHeader->nameDescription));
        strcpy(m_sectionHeader->nameDescription, description.c_str());
    }
    char* GetDescription() const
    {
        return m_sectionHeader->nameDescription;
    }
    void SetDataTypeSize(SectionData dataType, size_t size)
    {
        assert(size < 0x10000);
        m_sectionHeader->bytesPerElement = (WORD) size;
        m_sectionHeader->sectionData = (WORD) dataType;
    }
    void GetDataTypeSize(SectionData& dataType, size_t& size) const
    {
        size = m_sectionHeader->bytesPerElement;
        dataType = (SectionData) m_sectionHeader->sectionData;
    }
    // ElementSize - number of bytes per element, (0 means variable)
    void SetElementSize(size_t elementSize)
    {
        assert(elementSize < 0x10000);
        m_sectionHeader->bytesPerElement = (WORD) elementSize;
    }
    size_t GetElementSize() const
    {
        return (size_t) m_sectionHeader->bytesPerElement;
    }
    // ElementsPerRecord - number of elements per record
    void SetElementsPerRecord(size_t elementsPerRecord)
    {
        assert(elementsPerRecord < 0x10000);
        m_sectionHeader->elementsPerRecord = (WORD) elementsPerRecord;
    }
    size_t GetElementsPerRecord() const
    {
        return (size_t) m_sectionHeader->elementsPerRecord;
    }
    // ElementCount - number of total elements stored
    void SetElementCount(size_t elementCount)
    {
        m_sectionHeader->elementsCount = elementCount;
    }
    size_t GetElementCount() const
    {
        return m_sectionHeader->elementsCount;
    }

    void SetFlags(SectionFlags flags)
    {
        m_sectionHeader->flags |= flags;
    }
    void ClearFlags(SectionFlags flags)
    {
        m_sectionHeader->flags &= ~flags;
    }
    SectionFlags GetFlags() const
    {
        return (SectionFlags) m_sectionHeader->flags;
    }
    void SetCustomStruct(CustomStructure customStruct)
    {
        m_sectionHeader->customStructureID = (WORD) customStruct;
    }
    CustomStructure GetCustomStruct() const
    {
        return (CustomStructure) m_sectionHeader->customStructureID;
    }
    void SetSectionType(SectionType sectionType)
    {
        m_sectionHeader->sectionType = (WORD) sectionType;
    }
    SectionType GetSectionType() const
    {
        return (SectionType) m_sectionHeader->sectionType;
    }
    size_t GetFileUniqueId() const
    {
        return m_sectionHeader->writtenID;
    }
    void SetFileUniqueId(WORD fileId)
    {
        m_sectionHeader->writtenID = fileId;
    }

    void SetParent(Section* parent)
    {
        m_parent = parent;
    }
    Section* GetParent() const
    {
        return m_parent;
    }

    void SetName(const std::wstring& name)
    {
        m_name = name;
    }
    const std::wstring& GetName();
    int GetSectionCount() const
    {
        return m_sectionHeader->dataSections;
    }
    size_t GetRecordCount() const
    {
        return m_sectionHeader->elementsPerRecord != 0 ? m_sectionHeader->elementsCount / m_sectionHeader->elementsPerRecord : m_sectionHeader->elementsCount;
    }

    // Save the data into this section,
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized);
};

// SectionStats - section to hold statistics for a featureSet
class SectionStats : public Section
{
private:
    // single pass measures
    size_t m_count; // number of elements
    double m_max;   // maximum value we have seen
    double m_min;   // minimum value we have seen
    double m_sum;   // sum of all numbers we have seen
    double m_sum2;  // sum of the squares of all numbers we have seen

    // compute after single pass
    double m_mean; // mean of all values
    double m_rms;  // root mean square

    // progressive statistics
    double m_pmean;
    double m_pvariance;

    // second pass measures
    double m_varSum; // accumulated sum of difference between the mean and and the value squared

    // compute after second pass
    double m_variance;
    double m_stddev;

    template <typename ElemType>
    bool AccumulateData(ElemType* dataSource, size_t numRecords, size_t recordStart, size_t datasetSize);

    // Save the data into this section,
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized);

public:
    SectionStats(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType = mappingParent, size_t size = 0);
    SectionStats(SectionFile* file, Section* parentSection, size_t filePosition, MappingType mappingType = mappingParent, size_t size = 0);
    void InitCompute(const ConfigArray& compute);
    void SetCompute(const std::string& name, double value);
    double GetCompute(const std::string& name);
    void Store();
};

// SectionString - section to hold variable length zero terminated UTF8 strings
// supports mapping tables
// for faster access, a section with offsets to the beginning of the strings can be saved as well
class SectionString : public Section
{
public:
    typedef std::string LabelType; // TODO: are these supposed to be the same as the DataReader's?
    typedef unsigned LabelIdType;

private:
    std::map<LabelIdType, LabelType> m_mapIdToLabel;
    std::map<LabelType, LabelIdType> m_mapLabelToId;

protected:
    virtual char* EnsureElements(size_t element, size_t bytesRequested = 0);

public:
    SectionString(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType = mappingParent, size_t size = 0);
    SectionString(SectionFile* file, Section* parentSection, size_t filePosition, MappingType mappingType = mappingParent, size_t size = 0);

    // override this method to provide variable sized string behavior
    virtual void* GetElement(size_t index) const;

    // mapping table case
    const std::map<LabelIdType, LabelType>& GetLabelMapping();
    void SetLabelMapping(const std::map<LabelIdType, LabelType>& labelMapping);
};

// SectionLabel - class for handling labels
class SectionLabel : public Section
{
public:
    typedef unsigned LabelIdType;

public:
    SectionLabel(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType = mappingParent, size_t size = 0);
    SectionLabel(SectionFile* file, Section* parentSection, size_t filePosition, MappingType mappingType = mappingParent, size_t size = 0);

    // accessors for header members
    void SetLabelKind(LabelKind labelKind)
    {
        m_sectionHeader->labelKind = (WORD) labelKind;
    }
    LabelKind GetLabelKind()
    {
        return (enum LabelKind)(m_sectionHeader->labelKind);
    }
    void SetLabelDim(LabelIdType labelDim)
    {
        m_sectionHeader->labelDim = (WORD) labelDim;
    }
    LabelIdType GetLabelDim()
    {
        return m_sectionHeader->labelDim;
    }
};

// SectionFile - A binary file that is organized as sections
class SectionFile : public BinaryFile
{
    Section* m_fileSection;

public:
    SectionFile(std::wstring fileName, FileOptions options = fileOptionsRead, size_t size = 0);
    virtual ~SectionFile();
    Section* FileSection()
    {
        return m_fileSection;
    }
    SectionHeader* GetSection(size_t filePosition, size_t& size);
    void ReleaseSection(SectionHeader* section);
};

template <class ElemType>
class BinaryReader : public DataReaderBase
{
    size_t m_mbSize;           // size of minibatch requested
    size_t m_mbStartSample;    // starting sample # of the next minibatch
    size_t m_epochSize;        // size of an epoch
    size_t m_epoch;            // which epoch are we on
    size_t m_epochStartSample; // the starting sample for the epoch
    size_t m_totalSamples;     // number of samples in the dataset
    bool m_partialMinibatch;   // a partial minibatch is allowed
    MBLayoutPtr m_pMBLayout;

    int m_traceLevel;
    vector<SectionFile*> m_secFiles;
    std::map<std::wstring, Section*, nocase_compare> m_sections;

    /**
    for reading one line per file, i.e., a file has only one line of data
    */
    bool mOneLinePerFile;
    size_t m_dim;
    vector<FILE*> m_fStream;

    void SetupEpoch();
    void LoadSections(Section* parentSection, MappingType mapping, size_t windowSize);
    void DisplayProperties();
    bool CheckEndDataset(size_t actualmbsize);

public:
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
    virtual void Destroy();
    BinaryReader()
        : m_pMBLayout(make_shared<MBLayout>())
    {
    }
    virtual ~BinaryReader();
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
    virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);

    size_t GetNumParallelSequences()
    {
        return 1;
    }
    void SetNumParallelSequences(const size_t){};
    void CopyMBLayoutTo(MBLayoutPtr pMBLayout)
    {
        pMBLayout->CopyFrom(m_pMBLayout);
        NOT_IMPLEMENTED;
    }
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<typename BinaryReader<ElemType>::LabelIdType, typename BinaryReader<ElemType>::LabelType>& labelMapping);
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);

    virtual bool DataEnd();

    void SetRandomSeed(int)
    {
        NOT_IMPLEMENTED;
    };
};

template <class ElemType>
class BinaryWriter : public IDataWriter
{
    int m_traceLevel; // trace level to output the
    size_t m_recordCurrent;
    size_t m_recordMax;
    WORD m_uniqueID; // ID to identify all files as written at the same time

    // all the section files in the dataset
    std::map<std::wstring, SectionFile*, nocase_compare> m_secFiles;

    // all the sections in the save configuration
    std::map<std::wstring, Section*, nocase_compare> m_sections;

    // create the section map for the API
    std::map<std::wstring, SectionType, nocase_compare> m_sectionInfo;

    // create a section from config parameters
    Section* CreateSection(const ConfigParameters& config, Section* parentSection, size_t p_records, size_t p_windowSize = 0);
    Section* CreateSection(const ScriptableObjects::IConfigRecord& config, Section* parentSection, size_t p_records, size_t p_windowSize = 0);

public:
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
    // DataWriter Constructor
    // config - [in] configuration parameters for the datareader
    BinaryWriter(const ConfigParameters& config)
    {
        Init(config);
    }
    BinaryWriter()
    {
    }
    // Destroy - cleanup and remove this class
    // NOTE: this destroys the object, and it can't be used past this point
    virtual void Destroy();
    virtual ~BinaryWriter();

    // GetSections - Get the sections of the file
    // sections - a map of section name to section. Data sepcifications from config file will be used to determine where and how to save data
    virtual void GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections);

    // SaveData - save data in the file/files
    // recordStart - Starting record number
    // matricies - a map of section name (section:subsection) to data pointer. Data sepcifications from config file will be used to determine where and how to save data
    // numRecords - number of records we are saving, can be zero if not applicable
    // datasetSize - Size of the dataset
    // byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
    virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized = 0);

    // SaveMapping - save a map into the file
    // saveId - name of the section to save into (section:subsection format)
    // labelMapping - map we are saving to the file
    virtual void SaveMapping(std::wstring saveId, const std::map<typename BinaryWriter<ElemType>::LabelIdType, typename BinaryWriter<ElemType>::LabelType>& labelMapping);
    virtual bool SupportMultiUtterances() const 
    {
        return false;
    };
};

// utility function to round an integer up to a multiple of size
size_t RoundUp(size_t value, size_t size);
// HIGH and LOW DWORD functions
DWORD HIDWORD(size_t size);
DWORD LODWORD(size_t size);
} } }
