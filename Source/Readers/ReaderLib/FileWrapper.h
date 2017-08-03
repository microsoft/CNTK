//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _
#pragma warning(disable : 4996)   // ^^ this does not seem to work--TODO: make it work
#define _FILE_OFFSET_BITS 64      // to force fseeko() and ftello() 64 bit in Linux

#include <stdio.h>
#ifdef __WINDOWS__
#endif
#ifdef __unix__
#include <sys/types.h>
#include <sys/stat.h>
#endif
#include <errno.h>
#include <memory>
#include "fileutil.h"
#include <type_traits>


namespace CNTK {

class FileWrapper
{
public:

    FileWrapper(const std::wstring& filename, const wchar_t* mode)
        : m_filename(filename), 
        m_file(_wfopen(filename.c_str(), mode), std::fclose)
    {}

    // This variant of the FileWrapper does not own the file pointer
    // and does not close correspoinding file in the dtor.
    FileWrapper(const std::wstring& filename, FILE* file)
        : m_filename(filename),
        m_file(file, [](FILE*){})
    {}

    inline bool TryFlush()
    {
        return (fflush(m_file.get()) == 0);
    }

    inline void FlushOrDie()
    {
        if (!TryFlush())
            RuntimeError("Error flushing file '%ls': %s.", m_filename.c_str(), strerror(errno));
    }

    inline bool TrySeek(int64_t offset, int mode)
    {
        int rc;
#ifdef __WINDOWS__
        rc = _fseeki64(m_file.get(), offset, mode);
#else
        rc = fseeko(m_file.get(), offset, mode);
#endif
        return (rc == 0);
    }

    inline void SeekOrDie(int64_t offset, int mode)
    {
        if (!TrySeek(offset, mode))
            RuntimeError("Error seeking to position '%zu' in file '%ls': %s",
                offset, m_filename.c_str(), strerror(errno));
    }


    inline bool TryTell(uint64_t& offset)
    {
        int64_t rc;
#ifdef __WINDOWS__
        rc = _ftelli64(m_file.get());
#else
        rc = ftello(m_file.get());
#endif
        if (rc < 0)
            return false;

        offset = rc;
        return true;
    }

    inline int64_t TellOrDie()
    {
        uint64_t offset;

        if (!TryTell(offset))
            RuntimeError("Error retrieving current position in file '%ls': %s.",
                m_filename.c_str(), strerror(errno));

        return offset;
    }

    inline bool ReachedEOF() 
    {
        return feof(m_file.get()) != 0;
    }

    inline size_t Filesize() const
    {
        return filesize(m_file.get());
    }

    inline size_t Read(void* ptr, size_t size, size_t count)
    {
        return fread(ptr, size, count, m_file.get());
    }

    inline bool TryRead(void* ptr, size_t size, size_t count)
    {
        return (count == Read(ptr, size, count));
    }

    // This method should not be used if T has bare pointers as its members.
    template <typename T, typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
    inline bool TryRead(T& value)
    {
        return TryRead(&value, sizeof(value), 1);
    }

    inline void ReadOrDie(void* ptr, size_t size, size_t count)
    {
        if (!TryRead(ptr, size, count))
            RuntimeError("Error reading file '%ls': %s.", m_filename.c_str(), strerror(errno));
    }

    // This method should not be used if T has bare pointers as its members.
    template <typename T, typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
    inline bool ReadOrDie(T& value)
    {
        if (!TryRead(value))
            RuntimeError("Error reading file '%ls': %s.", m_filename.c_str(), strerror(errno));
    }

    inline bool TryWrite(const void* ptr, size_t size, size_t count)
    {
        size_t rc;
        rc = fwrite(ptr, size, count, m_file.get());
        if (size != 0)
            return (rc == count);
        return (rc == 0);
    }

    // This method should not be used if T has bare pointers as its members.
    template <typename T, typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
    inline bool TryWrite(const T& value)
    {
        return TryWrite(&value, sizeof(value), 1);
    }

    inline void WriteOrDie(const void* ptr, size_t size, size_t count)
    {
        if (!TryWrite(ptr, size, count))
            RuntimeError("Error writing to file '%ls': %s.", m_filename.c_str(), strerror(errno));
    }

    // This method should not be used if T has bare pointers as its members.
    template <typename T, typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
    inline bool WriteOrDie(const T& value)
    {
        WriteOrDie(&value, sizeof(value), 1);
    }

    static inline FileWrapper OpenOrDie(const std::wstring& filename, const wchar_t* mode)
    {
        FileWrapper wrapper(filename, mode);
        if (!wrapper.IsOpen())
            RuntimeError("Error opening file '%ls': %s.", filename.c_str(), strerror(errno));
        return wrapper;
    }

    inline bool IsOpen() const
    {
        return m_file != nullptr;
    }

    inline void CheckIsOpenOrDie() const
    {
        if (!IsOpen())
            RuntimeError("Input file '%ls' is not open.", Filename().c_str());
    }

    inline FILE* File() const
    {
        return m_file.get();
    }

    inline const std::wstring& Filename() const
    {
        return m_filename;
    }

private:

    std::wstring m_filename;
    std::shared_ptr<FILE> m_file;
};



}