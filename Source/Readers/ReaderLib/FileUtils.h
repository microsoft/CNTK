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
#include "Basics.h"


namespace Microsoft { namespace MSR { namespace CNTK {

using FilePtr = std::unique_ptr<std::FILE, int(*)(std::FILE*)>;

inline FilePtr Open(const std::wstring& pathname, const wchar_t* mode)
{
    return FilePtr(_wfopen(pathname.c_str(), mode), std::fclose);
}

inline bool TryOpen(FilePtr& f, const std::wstring& pathname, const wchar_t* mode)
{
    f.reset(_wfopen(pathname.c_str(), mode));
    return f != nullptr;
}

inline FilePtr OpenOrDie(const std::wstring& pathname, const wchar_t* mode)
{
    FilePtr f(nullptr, std::fclose);
    if (!TryOpen(f, pathname, mode))
        RuntimeError("Error opening file '%ls': %s.", pathname.c_str(), strerror(errno));
    return std::move(f);
}

inline bool TryFlush(FilePtr& f)
{
    return (fflush(f.get()) == 0);
}

inline void FlushOrDie(FilePtr& f)
{
    if (!TryFlush(f))
        RuntimeError("Error flushing: %s.", strerror(errno));
}

inline bool TrySeek(FilePtr& f, int64_t offset, int mode)
{
    int rc;
#ifdef __WINDOWS__
    rc = _fseeki64(f.get(), offset, mode);
#else
    rc = fseeko(f, offset, mode);
#endif
    return (rc == 0);
}

inline void SeekOrDie(FilePtr& f, int64_t offset, int mode)
{
    if (!TrySeek(f, offset, mode))
        RuntimeError("Error seeking: %s.", strerror(errno));
}


inline bool TryTell(FilePtr& f, uint64_t& offset)
{
    int64_t rc;
#ifdef __WINDOWS__
    rc = _ftelli64(f.get());
#else
    rc = ftello(f);
#endif
    if (rc < 0)
        return false;

    offset = rc;
    return true;
}

inline int64_t TellOrDie(FilePtr& f)
{
    uint64_t offset;
    
    if (!TryTell(f, offset))
        RuntimeError("Error telling: %s.", strerror(errno));

    return offset;
}

inline bool TryRead(void* ptr, size_t size, size_t count, FilePtr& f)
{
    size_t rc;
    rc = fread(ptr, size, count, f.get());
    return (rc == count);
}

template <typename T>
inline bool TryRead(T& value, FilePtr& f)
{
    return TryRead(&value, sizeof(value), 1, f);
}

inline void ReadOrDie(void* ptr, size_t size, size_t count, FilePtr& f)
{
    if (!TryRead(ptr, size, count, f))
        RuntimeError("Error reading: %s.", strerror(errno));
}

inline bool TryWrite(const void* ptr, size_t size, size_t count, const FilePtr& f)
{
    size_t rc;
    rc = fwrite(ptr, size, count, f.get());
    if (size != 0)
        return (rc == count);
    return (rc == 0);
}

template <typename T>
inline bool TryWrite(const T& value, const  FilePtr& f)
{
    return TryWrite(&value, sizeof(value), 1, f);
}

inline void WriteOrDie(const void* ptr, size_t size, size_t count, const  FilePtr& f)
{
    if (!TryWrite(ptr, size, count, f))
        RuntimeError("Error writing: %s.", strerror(errno));
}

template <typename T>
inline bool WriteOrDie(const T& value, const FilePtr& f)
{
    WriteOrDie(&value, sizeof(value), 1, f);
}




}}}