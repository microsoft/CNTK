//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// BinaryFile.cpp : Define the Binary File class
//

#include "stdafx.h"
#include "DataReader.h"
#include "BinaryReader.h"
#include <limits.h>
#include <stdint.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// HIGH and LOW DWORD functions
DWORD HIDWORD(size_t size)
{
    return size >> 32;
}
DWORD LODWORD(size_t size)
{
    return size & 0xFFFFFFFF;
}

// BinaryFile Constructor
// fileName - file to read or create (if it doesn't exist)
// options - file options, (fileOptionsReadWrite and fileOptionsRead are accepted)
// size - size of the file to map, will expand/contract existing files to given size. zero means keep current size
BinaryFile::BinaryFile(std::wstring fileName, FileOptions options, size_t size)
{

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    m_viewAlignment = sysInfo.dwAllocationGranularity;
    /* If file created, continue to map file. */

    m_writeFile = options == fileOptionsReadWrite;
    m_name = fileName;
    m_maxViewSize = 0x10000000; // 256MB initial max size
    m_hndFile = CreateFile(fileName.c_str(), m_writeFile ? (GENERIC_WRITE | GENERIC_READ) : GENERIC_READ,
                           FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m_hndFile == INVALID_HANDLE_VALUE)
    {
        char message[256];
        sprintf_s(message, "Unable to Open/Create file %ls, error %x", fileName.c_str(), GetLastError());
        RuntimeError(message);
    }

    // code to detect type of file (network/local)
    // std::wstring path;
    // auto found = fileName.find_last_of(L"\\/");
    // if (found == npos)
    //    path = fileName + L"\\";
    // else
    //    path = fileName.substr(0, found);
    // auto driveType = GetDriveType(path.c_str());

    // get the actual size of the file
    if (size == 0)
    {
        GetFileSizeEx(m_hndFile, (PLARGE_INTEGER) &size);
    }
    m_filePositionMax = size;

    m_hndMapped =
        CreateFileMapping(m_hndFile, NULL,
                          m_writeFile ? PAGE_READWRITE : PAGE_READONLY,
                          HIDWORD(size), LODWORD(size),
                          NULL);
    if (m_hndMapped == NULL)
    {
        char message[256];
        sprintf_s(message, "Unable to map file %ls, error 0x%x", fileName.c_str(), GetLastError());
        RuntimeError(message);
    }
    m_mappedSize = size;

    // if writing the file, the inital size of the file is zero
    if (m_writeFile)
    {
        m_filePositionMax = 0;
    }
}

// Destructor - destroy the BinaryFile
BinaryFile::~BinaryFile()
{
    for (auto iter = m_views.begin(); iter != m_views.end();)
    {
        // the view
        iter = ReleaseView(iter, true);
    }
    int rc = CloseHandle(m_hndMapped);
    if ((rc == 0) && !std::uncaught_exception())
    {
        RuntimeError("BinaryFile: Failed to close handle: %d", ::GetLastError());
    }

    // if we are writing the file, truncate to actual size
    if (m_writeFile)
    {
        SetFilePointerEx(m_hndFile, *(LARGE_INTEGER*) &m_filePositionMax, NULL, FILE_BEGIN);
        SetEndOfFile(m_hndFile);
    }
    rc = CloseHandle(m_hndFile);
    if ((rc == 0) && !std::uncaught_exception())
    {
        RuntimeError("BinaryFile: Failed to close handle: %d", ::GetLastError());
    }
}

void BinaryFile::SetFilePositionMax(size_t filePositionMax)
{
    m_filePositionMax = filePositionMax;
    if (m_filePositionMax > m_mappedSize)
    {
        char message[128];
        sprintf_s(message, "Setting max position larger than mapped file size: %ld > %ld", m_filePositionMax, m_mappedSize);
        RuntimeError(message);
    }
}

// Mapped - Check to see if the current address is mapped
// filePosition - file position to search for
// size - [in]size of the mapped region
//        [out] size of region that is actually mapped starting at filePosition
// returns - true if the address is already mapped completely
vector<ViewPosition>::iterator BinaryFile::Mapped(size_t filePosition, size_t& size)
{
    auto iter = m_views.begin();
    for (; iter != m_views.end(); ++iter)
    {
        if (filePosition >= iter->filePosition &&
            filePosition + size < iter->filePosition + iter->size)
        {
            // this is safe to put in a size_t (unsigned type)
            // because we know the answer will be positive (from the if above)
            size = iter->filePosition + iter->size - filePosition;
            break;
        }
    }
    return iter;
}

// FindDataView - Find view that contains the pointer value
// data - data pointer to search for
// returns - iterator to the found view, or end() if not found
vector<ViewPosition>::iterator BinaryFile::FindDataView(void* data)
{
    // search for the view we want
    auto iter = m_views.begin();
    for (; iter != m_views.end(); ++iter)
    {
        byte* viewBegin = (byte*) iter->view;
        if (viewBegin <= data && viewBegin + iter->size > data)
            break;
    }
    return iter;
}

// FindView - Find view by mapped pointer (pointer to beginning)
// view - pointer to view to search for
// returns - iterator to the found view, or end() if not found
vector<ViewPosition>::iterator BinaryFile::FindView(void* view)
{
    auto iter = m_views.begin();
    for (; iter != m_views.end(); ++iter)
    {
        if (iter->view == view)
            break;
    }
    return iter;
}

// FindView - Find view by file Position
// filePosition - file position to search for
// returns - iterator to the found view, or end() if not found
vector<ViewPosition>::iterator BinaryFile::FindView(size_t filePosition)
{
    auto iter = m_views.begin();
    for (; iter != m_views.end(); ++iter)
    {
        if (iter->filePosition == filePosition)
            break;
    }
    return iter;
}

// ReleaseView - release a view, and return the next iterator
// iter - iterator into the view array
// returns - next element in the view vector that is valid
vector<ViewPosition>::iterator BinaryFile::ReleaseView(vector<ViewPosition>::iterator iter, bool force)
{
    if (!force && iter->refCount > 1)
    {
        iter->refCount--;
        iter++;
    }
    else
    {
        if (m_writeFile)
            FlushViewOfFile(iter->view, iter->size);
        bool ret = UnmapViewOfFile(iter->view) != FALSE;
        ret;
        iter = m_views.erase(iter);
    }
    return iter;
}

// GetView - Get a view of the file
// filePosition - file position where the view starts
// size - size of the view (in bytes)
// returns - pointer to the view
void* BinaryFile::GetView(size_t filePosition, size_t size)
{
    void* pBuf = MapViewOfFile(m_hndMapped,                                  // handle to map object
                               m_writeFile ? FILE_MAP_WRITE : FILE_MAP_READ, // get correct permissions
                               HIDWORD(filePosition),
                               LODWORD(filePosition),
                               size);
    if (pBuf == NULL)
    {
        char message[128];
        sprintf_s(message, "Unable to map file %ls @ %lld, error %x", m_name.c_str(), filePosition, GetLastError());
        RuntimeError(message);
    }
    m_views.push_back(ViewPosition(pBuf, filePosition, size));

    // update file position max if neccesary
    size_t filePositionEnd = filePosition + size;
    if (m_filePositionMax < filePositionEnd)
        m_filePositionMax = filePositionEnd;
    return pBuf;
}

// ReleaseView - Release a view of the file
// view - view to release (must have been returned from GetView() previously
void BinaryFile::ReleaseView(void* view)
{
    bool found = false;
    for (auto iter = m_views.begin(); iter != m_views.end(); ++iter)
    {
        if (iter->view == view)
        {
            ReleaseView(iter);
            found = true;
            break;
        }
    }
    // view not released, could not be found
    assert(found);
}

// ReallocateView - Reallocate View
// view - pointer to the view
// size - size of the view
// returns - pointer to the reallocated view
// NOTE: on reallocate, the view address may change
void* BinaryFile::ReallocateView(void* view, size_t size)
{
    auto viewPos = FindView(view);
    if (viewPos != m_views.end())
    {
        size_t filePosition = viewPos->filePosition;
        ReleaseView(viewPos);
        view = GetView(filePosition, size);
    }
    else
    {
        // if we can't find the view, can't reallocate it, return NULL
        assert(false);
        view = NULL;
    }
    return view;
}

// EnsureViewSize - Ensure that a view is at least a particular size
// view - pointer to the view
// size - minimum size of the view
// returns - pointer to the view that is at least size bytes large
// NOTE: on reallocate, the view address may change
void* BinaryFile::EnsureViewSize(void* view, size_t size)
{
    auto viewPos = FindView(view);
    if (viewPos != m_views.end())
    {
        if (viewPos->size < size)
        {
            size_t filePosition = viewPos->filePosition;
            ReleaseView(viewPos);
            view = GetView(filePosition, size);
        }
    }
    else
    {
        // if we can't find the view, can't reallocate it, return NULL
        assert(false);
        view = NULL;
    }
    return view;
}

// EnsureMapped - Ensure that a view is at least a particular size
// dataPointer - pointer to the data
// size - allocated size of the data
// returns - pointer to the data location that is at least size bytes large
// NOTE: on reallocate, the view address may change
void* BinaryFile::EnsureMapped(void* data, size_t size)
{
    auto viewPos = FindDataView(data);
    if (viewPos != m_views.end())
    {
        int64_t offset = (byte*) data - (byte*) viewPos->view;
        int64_t dataEnd = offset + size;

        // if our end of data is beyond the size of the view, need to reallocate
        if (dataEnd > (int64_t) viewPos->size)
        {
            // TODO: this view change only accomidates this request
            size_t filePosition = viewPos->filePosition;
            ReleaseView(viewPos);
            byte* view = (byte*) GetView(filePosition, dataEnd);
            data = view + offset;
        }
    }
    else
    {
        // if we can't find the view, can't reallocate it, return NULL
        assert(false);
        data = NULL;
    }
    return data;
}

// RoundUp - round the file position to the next mappable location if we intend on mapping the location separately
// filePosition - position in the file we want to round up
size_t BinaryFile::RoundUp(size_t filePosition)
{
    size_t roundTo = m_viewAlignment;
    return ((filePosition + roundTo - 1) / roundTo) * roundTo;
}

// SectionFile Constructor
// fileName - file to read or create (if it doesn't exist)
// options - file options, (fileOptionsReadWrite and fileOptionsRead are accepted)
// size - size of the file to map, will expand/contract existing files to given size. zero means keep current size
SectionFile::SectionFile(std::wstring fileName, FileOptions options, size_t size)
    : BinaryFile(fileName, options, size)
{
    m_fileSection = new Section(this, 0, NULL, mappingFile, sectionHeaderMin);
    if (m_writeFile)
    {
        m_fileSection->InitHeader(sectionTypeFile, string("Binary Data File"), sectionDataNone, 0);
    }

    // check for a file header
    if (!m_fileSection->ValidateHeader(m_writeFile))
    {
        char message[128];
        sprintf_s(message, "Invalid File format for binary file %ls", fileName.c_str());
        RuntimeError(message);
    }
}

// SectionFile Destructor
SectionFile::~SectionFile()
{
    // remove each section
    SectionHeader* header = m_fileSection->GetHeader();

    // finishing up the file, put the magic number on it now. (but only if
    if (m_writeFile)
    {
        header->wMagic = magicFile;
    }

    // now delete the file section
    delete m_fileSection;
    m_fileSection = NULL;
    // after this BinaryFile destructor will take care of cleaning up sections mappings, memory mapped file, etc.
}

void Section::InitHeader(SectionType type, std::string description, SectionData dataType, WORD elementSize)
{

    memset(m_sectionHeader, 0, sectionHeaderMin);
    m_sectionHeader->wMagic = type == sectionTypeFile ? magicIncomplete : magicSection; // magic number ACE9/4ACE
    m_sectionHeader->version = 0x0100;                                                  // version number ##.## in hex
    m_sectionHeader->sizeHeader = sectionHeaderMin;                                     // size of this header (rounded up to nearest 64 byte boundary)
    m_sectionHeader->dataSections = 0;                                                  // number of data sub-sections (for nesting)
    m_sectionHeader->sectionType = (WORD) type;                                         // what is the type of the data in this section
    m_sectionHeader->sectionData = (WORD) dataType;                                     // type of section (SectionData enum)
    m_sectionHeader->bytesPerElement = elementSize;                                     // number of bytes per element, (0 means variable)
    m_sectionHeader->customStructureID = customStructureNone;                           // ID for custom structure
    m_sectionHeader->elementsPerRecord = 0;                                             // number of elements per Record
    m_sectionHeader->flags = flagNone;                                                  // bit flags, dependent on sectionType
    m_sectionHeader->elementsCount = 0;                                                 // number of total elements stored
    memset(m_sectionHeader->nameDescription, 0, descriptionSize);                       // clear out the string buffer to all zeros first
    strcpy_s(m_sectionHeader->nameDescription, description.c_str());                    // name and description of section contents in this format (name: description) (string, with extra bytes zeroed out, at least one null terminator required)
    m_sectionHeader->size = sectionHeaderMin;                                           // size of this section (including header)
    m_sectionHeader->sizeAll = sectionHeaderMin;                                        // size of this section (including header and all sub-sections)
    m_sectionHeader->sectionFilePosition[0] = 0;                                        // sub-section file offsets (if needed), assumed to be in File Position order
    return;
}

// ValidateHeader - make sure everything in the header is valid
bool Section::ValidateHeader(bool writing) const
{
    bool valid = true;
    valid &= (m_sectionHeader->wMagic == magicFile || m_sectionHeader->wMagic == magicSection || (writing && m_sectionHeader->wMagic == magicIncomplete)); // magic number ACE9/4ACE
    valid &= (m_sectionHeader->version >= 0x0100);                                                                                                         // version number ##.## in hex
    valid &= (m_sectionHeader->sizeHeader >= sectionHeaderMin && m_sectionHeader->sizeHeader < sectionHeaderMin * 2);                                      // size of this header (rounded up to nearest 64 byte boundary)
    valid &= (m_sectionHeader->dataSections < (m_sectionHeader->sizeHeader - offset(SectionHeader, sectionFilePosition)) / sizeof(size_t));                // number of data sub-sections (for nesting)
    valid &= (m_sectionHeader->sectionType < sectionTypeMax * 2);                                                                                          // what is the type of the data in this section
    valid &= (m_sectionHeader->sectionData < sectionDataMax * 2);                                                                                          // type of section (SectionData enum)
    // m_sectionHeader->bytesPerElement = elementSize;  // number of bytes per element, (0 means variable)
    valid &= (m_sectionHeader->customStructureID < customStructureMax * 2); // ID for custom structure
    // m_sectionHeader->elementsPerRecord = 0;  // number of elements per Record
    valid &= (m_sectionHeader->flags < flagMax * 2); // bit flags, dependent on sectionType
    // m_sectionHeader->elementsCount = 0; // number of total elements stored
    // strcpy_s(m_sectionHeader->nameDescription, description.c_str()); // name and description of section contents in this format (name: description) (string, with extra bytes zeroed out, at least one null terminator required)

    // the size of the section must at least accomidate the header and all it's elements
    valid &= (m_sectionHeader->size >= m_sectionHeader->sizeHeader + m_sectionHeader->elementsCount * m_sectionHeader->bytesPerElement);    // size of this section (including header and all sub-sections)
    valid &= (m_sectionHeader->sizeAll >= m_sectionHeader->sizeHeader + m_sectionHeader->elementsCount * m_sectionHeader->bytesPerElement); // size of this section (including header and all sub-sections)
    // There should be an integral number of records in the file (elementsCount/elementsPerRecord))
    valid &= (m_sectionHeader->elementsPerRecord == 0 || m_sectionHeader->elementsCount % m_sectionHeader->elementsPerRecord == 0);

    // file positions must be in order
    size_t filePositionLast = 0;
    for (int i = 0; i < m_sectionHeader->dataSections; ++i)
    {
        valid &= (m_sectionHeader->sectionFilePosition[i] > filePositionLast); // sub-section file offsets (if needed), assumed to be in File Position order
    }
    assert(valid);
    return valid;
}

// GetName - Get the name of this section, value can be overridden by user
const std::wstring& Section::GetName()
{
    if (!m_name.empty())
        return m_name;

    // if name is not set yet, get it from the description header
    std::wstring nameDescription(msra::strfun::utf16(m_sectionHeader->nameDescription));
    auto firstColon = nameDescription.find_first_of(L':');
    if (firstColon != npos && nameDescription.size() >= firstColon)
    {
        m_name = nameDescription.substr(0, firstColon);
    }

    // return name (or empty string if description had no colon)
    return m_name;
}

// GetSection - Get a sub section of this section
// index - index to this section
// mapping - type of mapping desired for this section
// sizeElements - size of the mapping window (records) if using mappingElementWindow
Section* Section::ReadSection(size_t index, MappingType mapping, size_t sizeElements)
{
    // check for valid index
    if (m_sections.size() <= index)
    {
        RuntimeError("ReadSection:Invalid index");
    }

    // already loaded, so return
    if (m_sections[index] != NULL)
        return m_sections[index];

    // load it if it hasn't been loaded (mapped)
    size_t filePosition = m_sectionHeader->sectionFilePosition[index];

    Section* section = NULL;
    size_t loadSize = 0;
    SectionHeader* header = GetSectionHeader(filePosition, mapping, loadSize);

    // makes sure the requested records isn't more than we have
    sizeElements = min(header->elementsCount / header->elementsPerRecord, sizeElements);
    // convert from records to bytes
    sizeElements *= header->elementsPerRecord * header->bytesPerElement;

    switch (header->sectionType)
    {
    default:
        section = new Section(m_file, this, header, filePosition, mapping, loadSize);
        break;
    case sectionTypeLabelMapping:
        section = new SectionString(m_file, this, header, filePosition, mapping, loadSize);
        // variable sized section, size is just the size of the section in bytes
        // for now we don't support separate window for variable sized data
        sizeElements = section->GetSize() - section->GetHeaderSize();
        break;
    case sectionTypeLabel:
        section = new SectionLabel(m_file, this, header, filePosition, mapping, loadSize);
        break;
    case sectionTypeStats:
        section = new SectionStats(m_file, this, header, filePosition, mapping, loadSize);
        break;
    }

    // make sure the header is valid
    if (!section->ValidateHeader())
    {
        char message[256];
        sprintf_s(message, "Invalid header in file %ls, in header %s\n", m_file->GetName().c_str(), section->GetName().c_str());
        RuntimeError(message);
    }

    // setup the element mapping and pointers as needed
    section->EnsureElements(0, sizeElements);

    // add section to parent section array
    m_sections[index] = section;
    return section;
}

// Destructor - destroy the section type
Section::~Section()
{
    // go through the sections array and delete all the sections
    for (auto iter = m_sections.begin(); iter != m_sections.end(); ++iter)
    {
        Section* section = *iter;
        // a section that is not mapped is skipped
        if (section == NULL)
            continue;
        delete section;
    }
    m_sections.clear();
}

// EnsureElements - Make sure that the elements range is available to read/write
// element - beginning element to access
// bytesRequested - bytes requested
// returns: pointer to the element requested
char* Section::EnsureElements(size_t element, size_t bytesRequested)
{
    // everything is already setup, so just return the pointer
    assert(!(GetFlags() & flagVariableSized)); // does not support variable size entries, those should be handled by Section subclasses
    size_t elementsRequested = bytesRequested / GetElementSize();
    if (element + elementsRequested > GetElementCount())
    {
        char message[256];
        sprintf_s(message, "Element out of range, error accesing element %lld, size=%lld\n", element, bytesRequested);
        RuntimeError(message);
    }

    // make sure we have the buffer in the range to handle the request
    if (element < m_elemMin || element + elementsRequested >= m_elemMax || m_elementBuffer == NULL)
    {
        GetElementBuffer(element, bytesRequested);
    }

    return (char*) m_elementBuffer + (element - m_elemMin) * GetElementSize();
}

// GetElementBuffer - get the element buffer for the passed element and size
// element - element we want the elementBuffer to start from
// windowSize - minimum size of the window in bytes for Element Window (will not resize smaller)
// returns: pointer to the element buffer
char* Section::GetElementBuffer(size_t element, size_t windowSize)
{
    // check element range
    if (!m_file->Writing() && element >= GetElementCount())
    {
        char message[256];
        sprintf_s(message, "Element out of range, error accesing element %lld, max element=%lld\n", element, GetElementCount());
        RuntimeError(message);
    }

    // section is mapped as a whole, so no separate mapping for element buffer
    if (m_mappingType != mappingElementWindow)
    {
        // mapped together with header, just offset
        m_elementBuffer = (char*) m_sectionHeader + m_sectionHeader->sizeHeader;
        m_elemMin = 0;
        if (m_file->Writing())
        {
            m_elemMax = windowSize;
            // variable sized sections won't have an element size, so just use bytes
            if (GetElementSize())
                m_elemMax /= GetElementSize();
        }
        else
        {
            m_elemMax = GetElementCount();
        }
        m_mappedElementSize = 0; // only used for element window mapping

        // save off the header size so we can reestablish pointers if necessary
        // size_t headerSize = m_sectionHeader->sizeHeader;
        // assert((char*)m_elementBuffer - (char*)m_sectionHeader);
        // void* elementBuffer =
        EnsureMapped(m_elementBuffer, windowSize);

        // check to see if the mapping changed, if so update pointers
        // if (elementBuffer != m_elementBuffer)
        // {
        //    m_elementBuffer = elementBuffer;
        //    m_sectionHeader -= headerSize;
        // }

        return (char*) m_elementBuffer;
    }

    // if the window size is larger, increase mapped size
    if (m_mappedElementSize < windowSize)
        m_mappedElementSize = windowSize;

    // if we have no element size specified us a default size
    else if (m_mappedElementSize == 0)
        m_mappedElementSize = 16 * 1024 * 1024; // default size 16Meg for now

    // get number of elements mapped
    size_t elementMappedCount = m_mappedElementSize / GetElementSize();
    elementMappedCount = min(elementMappedCount, GetElementCount());
    m_mappedElementSize = elementMappedCount * GetElementSize();

    // get the file position of the element
    if (element + elementMappedCount > GetElementCount())
    {
        m_elemMin = GetElementCount() - elementMappedCount;
        m_elemMax = GetElementCount();
    }
    else
    {
        m_elemMin = element;
        m_elemMax = m_elemMin + elementMappedCount;
    }
    size_t sectionPosition = GetHeaderSize() + m_elemMin * GetElementSize();

    // now get the aligned view location
    size_t alignBoundary = m_file->GetViewAlignment();
    size_t viewPosition = ((sectionPosition + m_filePosition) / alignBoundary) * alignBoundary;
    size_t offset = (sectionPosition + m_filePosition) % alignBoundary;

    // release the old view if we have a current view
    if (m_elementView != NULL)
    {
        m_file->ReleaseView(m_elementView);
    }
    m_elementView = m_file->GetView(viewPosition, m_mappedElementSize + offset);
    m_elementBuffer = (char*) m_elementView + offset;
    return (char*) m_elementBuffer;
}

// RemapHeader - Update view pointers if view changed for all sections in that view
// view - new view pointer to the passed filePosition
// filePosition - filePosition view refers to
void Section::RemapHeader(void* view, size_t filePosition)
{
    for (auto iter = m_sections.begin(); iter != m_sections.end(); ++iter)
    {
        Section* section = *iter;
        // a section that is not mapped is skipped
        if (section == NULL)
            continue;
        if (section->m_mappingType == mappingParent)
        {
            int64_t offset = section->m_filePosition - filePosition;
            if (offset < 0)
                RuntimeError("RemapHeader:Invalid mapping, children must follow parent in mapping space");

            // update our header location
            SectionHeader* header = section->m_sectionHeader = (SectionHeader*) ((char*) view + offset);
            section->m_elementBuffer = (char*) header + header->sizeHeader;
            // now update any children
            section->RemapHeader(view, filePosition);
        }
    }
}

// EnsureMappingSize - make sure the mapping for this section is at least this size
// size - size to ensure we have
void Section::EnsureMappingSize(size_t size)
{
    // only call on mapping owners, that are mapped at least by section
    assert(m_mappingType != mappingElementWindow);
    assert(this == SectionMappingOwner());
    void* view = m_file->EnsureViewSize(m_sectionHeader, size);

    // check for remap
    if (view != m_sectionHeader)
    {
        m_sectionHeader = (SectionHeader*) view;
        m_elementBuffer = (char*) m_sectionHeader + m_sectionHeader->sizeHeader;
        RemapHeader(view, m_filePosition);
    }
}

// EnsureMapped - make sure the pointer passed is mapped appropriately
// dataStart - starting pointer of the location we want mapped
// size - size to ensure we have starting at this pointer
// returns: new mapping if applicable
void* Section::EnsureMapped(void* dataStart, size_t size)
{
    // only call on mapping owners
    void* view = m_file->EnsureMapped(dataStart, size);

    // check for remap, if this section isn't mapping itself, we may need to
    // check to see if the mapping owner is different
    if (view != dataStart)
    {
        // Element Window is mapped separately so won't no need to remap
        if (m_mappingType != mappingElementWindow)
        {
            int64_t offset = (byte*) view - (byte*) dataStart;
            m_sectionHeader = (SectionHeader*) ((char*) m_sectionHeader + offset);
            m_elementBuffer = (char*) m_sectionHeader + m_sectionHeader->sizeHeader;
            RemapHeader(m_sectionHeader, m_filePosition);
        }
    }
    return view;
}

// GetSectionHeader - Retrieve a section header (and possibly entire section)
// filePosition - position in the file where the section starts
// mappingType - [in,out] mapping type to use for this header, may be changed depending on the header
// size - [in] max size to read, if 0, read the entire section (size in the header)
//        [out] size actually mapped for this section
// returns pointer to the section header
SectionHeader* Section::GetSectionHeader(size_t filePosition, MappingType& mappingType, size_t& size)
{
    SectionHeader* sectionHeader = NULL;
    switch (mappingType)
    {
    case mappingParent:
    {
        auto sectionOwner = SectionMappingOwner();
        if (filePosition < sectionOwner->GetFilePosition())
            RuntimeError("invalid fileposition, cannot be earlier in the file than mapping parent");
        size_t offset = filePosition - sectionOwner->GetFilePosition();
        size_t totalSize = offset + max(size, (size_t) sectionHeaderMin);

        // make sure we can at least get to the header
        if (sectionOwner->GetMappedSize() < totalSize)
        {
            sectionOwner->EnsureMappingSize(totalSize);
        }
        sectionHeader = (SectionHeader*) ((char*) sectionOwner->GetHeader() + offset);

        // if we are reading we can actually see the REAL header size now, realloc if needed to get that as well
        if (!m_file->Writing())
        {
            size = sectionHeader->size;
            totalSize = offset + size;
            if (sectionOwner->GetMappedSize() < totalSize)
            {
                sectionOwner->EnsureMappingSize(totalSize);
            }
        }
        break;
    }
    case mappingFile:
        if (filePosition != 0)
            RuntimeError("invalid fileposition, file mapping sections must start at filePostion zero");
    // intentional fall-through - same case, just at beginning of file
    case mappingSectionAll:
        sectionHeader = m_file->GetSection(filePosition, size);
        if (!m_file->Writing())
        {
            size_t newSize = sectionHeader->sizeAll;
            if (newSize > size)
                sectionHeader = (SectionHeader*) m_file->ReallocateView(sectionHeader, newSize);
            size = newSize;
        }
        break;
    case mappingSection:
    {
        sectionHeader = m_file->GetSection(filePosition, size);
        break;
    }
    case mappingElementWindow:
        size_t sizeHeader = sectionHeaderMin;
        sectionHeader = m_file->GetSection(filePosition, sizeHeader);
        size = sizeHeader;
        // we don't allocate the element view at this time, we will later
        // This allows us to not map the areas of the file we don't care about...

        // if we are reading we can actually see the REAL header size now, realloc if needed to get that as well
        if (!m_file->Writing())
        {
            // however we always map auxiliary Sections, now that we know we have one switch over to mappingSection and map it.
            if (sectionHeader->flags & (flagAuxilarySection | flagVariableSized))
            {
                size = sectionHeader->size;
                m_file->ReleaseSection(sectionHeader);
                sectionHeader = m_file->GetSection(filePosition, size);
                mappingType = mappingSection;
            }
        }
        break;
    }

    // Now return the header
    return sectionHeader;
}

// GetSection - Retrieve a section header (and possibly entire section)
// filePosition - position in the file where the section starts
// size - [in] max size to read, if 0, read the entire section (size in the header)
//        [out] size actually mapped for this section
// returns pointer to the section header
SectionHeader* SectionFile::GetSection(size_t filePosition, size_t& size)
{
    bool determineSize = size == 0;
    if (determineSize)
        size = sectionHeaderMin;
    SectionHeader* binHeader = (SectionHeader*) GetView(filePosition, size);

    // if we aren't writing a file figure out what size we need
    if (!m_writeFile)
    {
        // mapped size not large enough for this section, and they want us to allocate for the entire thing
        if (binHeader->size > size && determineSize)
        {
            size = binHeader->size;
            binHeader = (SectionHeader*) ReallocateView(binHeader, size);
        }
        // size not big enough for the header, need at least that much
        else if (binHeader->sizeHeader > size)
        {
            size = binHeader->sizeHeader;
            binHeader = (SectionHeader*) ReallocateView(binHeader, size);
        }
        // max size is bigger than entire section, reduce to actual size mapped
        else if (binHeader->sizeAll < size)
        {
            size = binHeader->size;
            binHeader = (SectionHeader*) ReallocateView(binHeader, size);
        }
    }
    else
    {
        memset(binHeader, 0, sectionHeaderMin);
        binHeader->sectionType = sectionTypeData;
        binHeader->wMagic = magicSection;
        binHeader->version = 0x0100;
        binHeader->sizeHeader = sectionHeaderMin;
        binHeader->size = size;    // set the currently mapped size for this header
        binHeader->sizeAll = size; // set size with all sub-elements to this size too
    }
    return binHeader;
}

// ReleaseSection - Release a Section of the file
// section - Section to release (must have been returned from GetSection() previously)
void SectionFile::ReleaseSection(SectionHeader* section)
{
    ReleaseView(section);
}

// SectionMappingOwner - find the section that owns the mapping view for this section
// returns - section that owns the mapping
Section* Section::SectionMappingOwner()
{
    Section* section = this;
    // if the current section is not a parent mapping, then this is the mapping owner
    if (section->m_mappingType != mappingParent)
    {
        assert(section->m_mappingType == mappingSectionAll || section->m_mappingType == mappingFile);
        return this;
    }

    // search backwards looking for a mapping owner
    while (section->m_parent != NULL)
    {
        section = section->m_parent;
        if (section->m_mappingType == mappingSectionAll || section->m_mappingType == mappingFile)
        {
            return section;
        }
    }

    // oops, nothing found, mapping error
    assert("mapping error, no mapping parent found");
    return this;
}

// Section Constructor
// initializes all the section variables
// for writing, it does NOT initialize the header, this must be done by the caller
Section::Section(SectionFile* sectionFile, Section* parentSection, size_t filePosition, MappingType mappingType, size_t size)
{
    Init(sectionFile, parentSection, NULL, filePosition, mappingType, size);
}

// Section Constructor
// initializes all the section variables
// for writing, it does NOT initialize the header, this must be done by the caller
Section::Section(SectionFile* sectionFile, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType, size_t size)
{
    Init(sectionFile, parentSection, sectionHeader, filePosition, mappingType, size);
}

// Init Section
// file - section file to which we belong
// parentSection - parent section
// sectionHeader - sectionHeader pointer to pre-mapped section, or NULL if constructor should do mapping
// filePosition - position in file where header starts
// mappingType - type of mapping we are using for this section
// size - size for this section, for reading this should be zero so actual size of section will be used
// NOTE: for writing, it does NOT initialize the header, call InitHeader() for header initalization
void Section::Init(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType, size_t size)
{
    m_file = file;
    m_filePosition = file->RoundUp(filePosition);
    m_parent = parentSection;
    m_mappingType = mappingType;

    if (size < sectionHeaderMin && file->Writing())
        RuntimeError("Insufficient size requested for section in write mode");

    // clear out element window mapping variables
    m_elementView = NULL;    // pointer to beginning of the valid view
    m_elementBuffer = NULL;  // no buffer pointer yet.
    m_mappedElementSize = 0; // size of mapped element view, only used for mappingElementWindow mappingType
    m_elemMin = 0;           // currently mapped minimum value (first valid)
    m_elemMax = 0;           // currently mapped maximum value (one past last valid)

    if (sectionHeader == NULL)
    {
        sectionHeader = GetSectionHeader(filePosition, mappingType, size);
        m_mappingType = mappingType; // in case it was modified in GetSectionHeader
    }
    m_sectionHeader = sectionHeader;

    // resize the number of sections so it is correct, initialize with zero to show we haven't actually mapped that area yet
    if (m_sectionHeader->dataSections > 0)
        m_sections.resize(m_sectionHeader->dataSections, 0);

    m_mappedSize = size;
}

// AddSection - Add a new subsection to this section.
//   This handles any allocation that may be necessary as well as updating headers with valid file positions
//   Currently limited to adding sections at the end of a parent section
// sectionToAdd - subsection we are adding on this section
Section* Section::AddSection(Section* sectionToAdd)
{
    // Get file position
    Section* section = this;

    size_t filePosition = sectionToAdd->GetFilePosition(); // get file position
    size_t sizeRounded = m_file->RoundUp(section->GetSize());

    // set parent
    sectionToAdd->SetParent(this);

    // add to parents header array of sections if they are in the same file
    if (sectionToAdd->GetSectionFile() == section->GetSectionFile())
    {
        SectionHeader* header = section->GetHeader();
        header->sectionFilePosition[header->dataSections++] = filePosition;
        if (header->dataSections && header->sectionFilePosition[header->dataSections - 1] < filePosition)
            RuntimeError("invalid fileposition for section, subsection cannot start earlier in the file than parent section");

        // now update size for all pervious ancestor sections
        do
        { // add in the size of the new section to parent setion, always assumed to go on the end
            sizeRounded = m_file->RoundUp(header->sizeAll);
            sizeRounded += sectionToAdd->GetSize(); // put it on the end of this section
            header->sizeAll = sizeRounded;

            // if this section is the file level, we don't need to update past here
            if (section->GetMappingType() == mappingFile)
                break;
            section = section->GetParent();
            header = section->GetHeader();
        } while (section != NULL && sectionToAdd->GetSectionFile() == section->GetSectionFile());
    }

    // add to sections array (in memory only)
    m_sections.push_back(sectionToAdd);

    return sectionToAdd;
}

// SaveData - save data in the file/files
// recordStart - Starting record number, will be > datasetSize for multiple passes
// matricies - a map of section name (section:subsection) to data pointer. Data sepcifications from config file will be used to determine where and how to save data
// numRecords - number of records we are saving, can be zero if not applicable
// datasetSize - Size of the dataset
// byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
// returns: true if more data is desired, false if no more data is required (signal that file may be closed)
bool Section::SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized)
{
    Section* section = this;
    // if the section not in the data stream, by default we don't save anything
    // (override for implementations that do stats, and other processing)
    auto iterator = matrices.find(GetName());
    if (iterator == matrices.end())
    {
        return false;
    }
    void* dataSource = iterator->second;

    // only want to save data on first iteration
    if (recordStart >= datasetSize)
    {
        return false;
    }

    size_t index = recordStart * section->GetElementsPerRecord();
    size_t size;
    if (byteVariableSized > 0 && (GetFlags() & flagVariableSized))
        size = byteVariableSized;
    else if (section->GetElementSize() > 0)
        size = section->GetElementSize() * section->GetElementsPerRecord() * numRecords;
    else
    {
        // hum, is this allowable?
        RuntimeError("Invalid size in Binary Writer, variable sized data with no length");
    }

    char* data = section->EnsureElements(index, size);
    // void* data = section->GetElement(index);
    memcpy_s(data, size, dataSource, size);
    return recordStart + numRecords < GetRecordCount();
}

SectionString::SectionString(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType, size_t size)
    : Section(file, parentSection, sectionHeader, filePosition, mappingType, size)
{
}
SectionString::SectionString(SectionFile* file, Section* parentSection, size_t filePosition, MappingType mappingType, size_t size)
    : Section(file, parentSection, filePosition, mappingType, size)
{
}

// GetLabelMapping - Get the label mapping from the current file
// translates from a string representation in the file
// returns - a map from LabelId to Label string
const std::map<SectionString::LabelIdType, SectionString::LabelType>& SectionString::GetLabelMapping()
{
    if (m_mapIdToLabel.size() == 0)
    {
        char* str = (char*) GetElement(0);
        SectionString::LabelIdType elements = (SectionString::LabelIdType) GetElementCount();
        for (SectionString::LabelIdType element = 0; element < elements; ++element)
        {
            std::string stringLabel(str);
            m_mapIdToLabel[element] = stringLabel;
            m_mapLabelToId[stringLabel] = element;

            // advance to the next string (stops on the NULL)
            while (*str++)
                ;
        }
    }
    return m_mapIdToLabel;
}

// SetLabelMapping - set the mapping data
void SectionString::SetLabelMapping(const std::map<SectionString::LabelIdType, SectionString::LabelType>& labelMapping)
{
    Section* section = this;
    if (labelMapping.size() != GetElementCount())
        RuntimeError("SetLabelMapping called with mapping table that doesn't match file defined element count");
    if (GetSectionType() != sectionTypeLabelMapping)
        RuntimeError("label type invalid");

    // free the old mapping tables
    m_mapLabelToId.clear();
    m_mapIdToLabel.clear();

    // now write out the passed mapping table
    char* curStr = section->GetElement<char>(0);
    size_t size = section->GetSize() - section->GetHeaderSize();
    size_t originalSize = size;
    for (unsigned i = 0; i < section->GetElementCount(); ++i)
    {
        auto iter = labelMapping.find(i);
        if (iter == labelMapping.end())
        {
            char message[256];
            sprintf_s(message, "Mapping table doesn't contain an entry for label Id#%d\n", i);
            RuntimeError(message);
        }

        // add to reverse mapping table
        m_mapLabelToId[iter->second] = i;

        const string& str = iter->second;
        errno_t err = strcpy_s(curStr, size, str.c_str());
        if (err)
        {
            char message[256];
            sprintf_s(message, "Not enough room in mapping buffer, %lld bytes insufficient for string %d - %s\n", originalSize, i, str.c_str());
            RuntimeError(message);
        }
        size_t len = str.length() + 1; // don't forget the null
        size -= len;
        curStr += len;
    }

    // now zero out the rest of the free space
    memset(curStr, 0, size);

    // store the resulting map
    m_mapIdToLabel = labelMapping;
}

// GetElement - Get the element at the given index
void* SectionString::GetElement(size_t index) const
{
    char* str = (char*) m_elementBuffer;
    if (index >= GetElementCount())
    {
        char message[256];
        sprintf_s(message, "GetElement: invalid index, %lld requested when there are only %lld elements\n", index, GetElementCount());
        RuntimeError(message);
    }

    // now skip all the strings before the one that we want
    for (SectionString::LabelIdType element = 0; element < index; ++element)
    {
        // advance to the next string (stops on the NULL)
        while (*str++)
            ;
    }
    return str;
}

// EnsureElements - Make sure that the elements range is available to read/write
// element - beginning element to access
// bytesRequested - bytes requested
// returns: pointer to the element requested
char* SectionString::EnsureElements(size_t element, size_t bytesRequested)
{
    // everything is already setup, so just return the pointer
    assert((GetFlags() & flagVariableSized));         // does support variable size entries
    assert(GetMappingType() != mappingElementWindow); // not supported for string tables currently
    if (element >= GetElementCount())
    {
        char message[256];
        sprintf_s(message, "Element out of range, error accesing element %lld, size=%lld\n", element, bytesRequested);
        RuntimeError(message);
    }

    // make sure we have the buffer in the range to handle the request
    if (m_elementBuffer == NULL)
    {
        GetElementBuffer(element, bytesRequested);
    }

    return (char*) GetElement(element);
}

SectionLabel::SectionLabel(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType, size_t size)
    : Section(file, parentSection, sectionHeader, filePosition, mappingType, size)
{
}
SectionLabel::SectionLabel(SectionFile* file, Section* parentSection, size_t filePosition, MappingType mappingType, size_t size)
    : Section(file, parentSection, filePosition, mappingType, size)
{
}

// Store - store the stats in the file
void SectionStats::Store()
{
    for (int i = 0; i < GetElementCount(); i++)
    {
        auto stat = GetElement<NumericStatistics>(i);
        if (EqualCI(stat->statistic, "sum"))
        {
            stat->value = m_sum;
        }
        else if (EqualCI(stat->statistic, "count"))
        {
            stat->value = (double) m_count;
        }
        else if (EqualCI(stat->statistic, "mean"))
        {
            stat->value = m_mean;
        }
        else if (EqualCI(stat->statistic, "max"))
        {
            stat->value = m_max;
        }
        else if (EqualCI(stat->statistic, "min"))
        {
            stat->value = m_min;
        }
        else if (EqualCI(stat->statistic, "range"))
        {
            stat->value = abs(m_max - m_min);
        }
        else if (EqualCI(stat->statistic, "rootmeansquare"))
        {
            stat->value = m_rms;
        }
        else if (EqualCI(stat->statistic, "variance"))
        {
            stat->value = m_variance;
        }
        else if (EqualCI(stat->statistic, "stddev"))
        {
            stat->value = m_stddev;
        }
        else
        {
            fprintf(stderr, "Unknown statistic requested in file %s\n", stat->statistic);
        }
    }
}

SectionStats::SectionStats(SectionFile* file, Section* parentSection, size_t filePosition, MappingType mappingType, size_t size)
    : Section(file, parentSection, filePosition, mappingType, size)
{
}

SectionStats::SectionStats(SectionFile* file, Section* parentSection, SectionHeader* sectionHeader, size_t filePosition, MappingType mappingType, size_t size)
    : Section(file, parentSection, sectionHeader, filePosition, mappingType, size)
{
}

// SetCompute - Initialize the statistics gathering locations
// compute - config array of the names of the stats we want recorded
void SectionStats::InitCompute(const ConfigArray& compute)
{
    SetElementCount(compute.size());
    for (int i = 0; i < compute.size(); i++)
    {
        std::string name = compute[i];
        auto stat = GetElement<NumericStatistics>(i);
        strcpy_s(stat->statistic, name.c_str());
        stat->value = 0.0;
    }

    // initialize all the internal variables
    m_count = 0;      // number of elements
    m_max = -DBL_MAX; // maximum value we have seen
    m_min = DBL_MAX;  // minimum value we have seen
    m_sum = 0.0;      // sum of all numbers we have seen
    m_sum2 = 0.0;     // sum of the squares of all numbers we have seen

    // compute after single pass
    m_mean = 0.0; // mean of all values
    m_rms = 0.0;  // root mean square

    // second pass measures
    m_varSum = 0.0; // accumulated sum of difference between the mean and and the value squared

    // compute after second pass
    m_variance = 0.0;
    m_stddev = 0.0;

    // progressive variables
    m_pmean = 0.0;
    m_pvariance = 0.0;
}

// SetCompute - Set a compute variable to a value
// name - name of the compute variable
// value - value to set
void SectionStats::SetCompute(const std::string& name, double value)
{
    for (int i = 0; i < GetElementCount(); i++)
    {
        auto stat = GetElement<NumericStatistics>(i);
        if (EqualCI(stat->statistic, name.c_str()))
        {
            stat->value = value;
            break;
        }
    }
}

// GetCompute - Get a compute variable based on a name
// name - name of the compute variable
// returns - value of the variable
double SectionStats::GetCompute(const std::string& name)
{
    for (int i = 0; i < GetElementCount(); i++)
    {
        auto stat = GetElement<NumericStatistics>(i);
        if (EqualCI(stat->statistic, name.c_str()))
        {
            return stat->value;
        }
    }
    return 0.0;
}

// SaveData - save data in the file/files
// recordStart - Starting record number, will be > datasetSize for multiple passes
// matricies - a map of section name (section:subsection) to data pointer. Data sepcifications from config file will be used to determine where and how to save data
// numRecords - number of records we are saving, can be zero if not applicable
// datasetSize - Size of the dataset
// byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
// returns: true if more data is desired, false if no more data is required (signal that file may be closed)
bool SectionStats::SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t /*byteVariableSized*/)
{
    // if the section is not the data stream we are a child of ignore it
    auto iterator = matrices.find(m_parent->GetName());
    if (iterator == matrices.end())
    {
        return false;
    }
    void* dataSource = iterator->second;

    size_t elementSize = m_parent->GetElementSize();
    bool retVal = false;
    if (elementSize == sizeof(float))
        retVal = AccumulateData((float*) dataSource, numRecords, recordStart, datasetSize);
    else if (elementSize == sizeof(double))
        retVal = AccumulateData((double*) dataSource, numRecords, recordStart, datasetSize);
    return retVal;
}

#define ONE_PASS
// AccumulateData - accumulate the data for the dataset
// Templated to handle both double and float values
// dataSource - pointer to the data
// numRecords - number of records in the dataSource
// recordStart - record we are starting on
// datasetSize - size of the dataset
template <typename ElemType>
bool SectionStats::AccumulateData(ElemType* dataSource, size_t numRecords, size_t recordStart, size_t datasetSize)
{
    // on first pass we want to calculate sums and squared sums
    if (recordStart < datasetSize)
    {

        ElemType* elemP = (ElemType*) dataSource;
        ElemType* elemPMax = elemP + m_parent->GetElementsPerRecord() * numRecords;

        // sum up all the values
        while (elemP < elemPMax)
        {
            ElemType elem = *elemP;
            // keep current and next count for easier math below
            size_t curCount = m_count;
            m_count++;

            // calculate progressive mean and variance
            // see: http://mathworld.wolfram.com/SampleVarianceComputation.html
            double prevMean = m_pmean;
            m_pmean = m_pmean + (elem - m_pmean) / (m_count);
            if (curCount)
            {
                double meanDifference = (m_pmean - prevMean);
                m_pvariance = (1.0 - 1.0 / curCount) * m_pvariance + m_count * meanDifference * meanDifference;
            }

            // accumulate sums
            m_sum += elem;
            m_sum2 += elem * elem;
            if (elem < m_min)
                m_min = elem;
            if (elem > m_max)
                m_max = elem;
            elemP++;
        }

        // check for the last chunk in the first pass
        if (recordStart + numRecords == datasetSize)
        {
            m_mean = m_sum / m_count;
            m_rms = m_sum2 / m_count;
            m_rms = sqrt(m_rms);
            m_variance = m_pvariance;
            m_stddev = sqrt(m_variance);

            double error = m_mean - m_pmean;
            if (error > 0.00001)
                printf("substantial difference in progressive mean\n");
#if defined(ONE_PASS)
            // since we are done now we can store the final values
            Store();
            return false;
#endif
        }
        return true;
    }
#if !defined(ONE_PASS)
    else if (recordStart < 2 * datasetSize)
    {
        ElemType* elemP = (ElemType*) dataSource;
        ElemType* elemPMax = elemP + m_parent->GetElementsPerRecord() * numRecords;

        // sum up all the values
        while (elemP < elemPMax)
        {
            ElemType elem = *elemP++;
            double diff = elem - m_mean;
            m_varSum += diff * diff;
        }

        // check for the last chunk in the first pass
        if (recordStart + numRecords == 2 * datasetSize)
        {
            m_variance = m_varSum / m_count;
            m_stddev = sqrt(m_variance);
            double error = m_pvariance - m_variance;
            if (error > 0.00001)
                printf("substantial difference in progressive variance\n");

            // since we are done now we can store the final values
            Store();
            return false;
        }
        return true;
    }
#endif
    // done with data
    return false;
}
} } }
