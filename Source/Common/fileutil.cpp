//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _
#pragma warning(disable : 4996)   // ^^ this does not seem to work--TODO: make it work
#define _FILE_OFFSET_BITS 64      // to force fseeko() and ftello() 64 bit in Linux

#ifdef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES // fixed-buffer overloads for strcpy() etc.
#undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#endif
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#include "Basics.h"
#include "basetypes.h" //for attemp()
#include "fileutil.h"
#include "ProgressTracing.h"

#ifdef __unix__
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <glob.h>
#include <dirent.h>
#endif
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#ifdef _WIN32
#define NOMINMAX
#include "Windows.h" // for FILETIME
#endif
#include <algorithm> // for std::find
#include <limits.h>
#include <memory>
#include <cwctype>
#ifndef UNDER_CE // some headers don't exist under winCE - the appropriate definitions seem to be in stdlib.h
#if defined(_WIN32) || defined(__CYGWIN__)
#include <fcntl.h> // for _O_BINARY/TEXT - not needed for wince
#include <io.h>    // for _setmode()
#define SET_BINARY_MODE(handle) setmode(handle, _O_BINARY)
#define SET_TEXT_MODE(handle) setmode(handle, _O_TEXT)
#else
#define SET_BINARY_MODE(handle) ((int) 0)
#define SET_TEXT_MODE(handle) ((int) 0)
#endif
#endif

#define __out_z_cap(x) // a fake SAL annotation; this may come in handy some day if we try static code analysis, so I don't want to delete it
#define FINDCLOSE_ERROR 0

#include <errno.h>

using namespace std;
using namespace Microsoft::MSR::CNTK;

// All sizes are in bytes
const int BUF_SIZE = 1000000;                       // Default buffer size 
const int LARGE_BUF_SIZE = 10 * BUF_SIZE;           // Used by fopenOrDie
const DWORD READ_SIZE_LIMIT = 15 * 1024 * 1024;     // Used by freadOrDie
const DWORD WRITE_SIZE_LIMIT = 16 * 1024 * 1024;    // Used by fwriteOrDie

// ----------------------------------------------------------------------------
// some mappings for non-Windows builds
// ----------------------------------------------------------------------------

template <>
const wchar_t* GetScanFormatString(char)
{
    return L" %hc";
}
template <>
const wchar_t* GetScanFormatString(wchar_t)
{
    return L" %lc";
}
template <>
const wchar_t* GetScanFormatString(short)
{
    return L" %hi";
}
template <>
const wchar_t* GetScanFormatString(int)
{
    return L" %i";
}
template <>
const wchar_t* GetScanFormatString(long)
{
    return L" %li";
}
template <>
const wchar_t* GetScanFormatString(unsigned short)
{
    return L" %hu";
}
template <>
const wchar_t* GetScanFormatString(unsigned int)
{
    return L" %u";
}
template <>
const wchar_t* GetScanFormatString(unsigned long) 
{
    return L" %lu";
}
template <>
const wchar_t* GetScanFormatString(float)
{
    return L" %g";
}
template <>
const wchar_t* GetScanFormatString(double)
{
    return L" %lg";
}
template <>
const wchar_t* GetScanFormatString(unsigned long long)
{
    return L" %llu";
}
template <>
const wchar_t* GetScanFormatString(long long)
{
    return L" %lli";
}

template <>
const wchar_t* GetFormatString(char)
{
    return L" %hc";
}
template <>
const wchar_t* GetFormatString(wchar_t)
{
    return L" %lc";
}
template <>
const wchar_t* GetFormatString(short)
{
    return L" %hi";
}
template <>
const wchar_t* GetFormatString(int)
{
    return L" %i";
}
template <>
const wchar_t* GetFormatString(long)
{
    return L" %li";
}
template <>
const wchar_t* GetFormatString(unsigned short)
{
    return L" %hu";
}
template <>
const wchar_t* GetFormatString(unsigned int)
{
    return L" %u";
}

template <>
const wchar_t* GetFormatString(unsigned long)
{
    return L" %lu";
}

template <>
const wchar_t* GetFormatString(float)
{
    return L" %.9g";
}
template <>
const wchar_t* GetFormatString(double)
{
    return L" %.17g";
}
template <>
const wchar_t* GetFormatString(unsigned long long)
{
    return L" %llu";
}
template <>
const wchar_t* GetFormatString(long long)
{
    return L" %lli";
}
template <>
const wchar_t* GetFormatString(const char*)
{
    return L" %hs";
}
template <>
const wchar_t* GetFormatString(const wchar_t*)
{
    return L" %ls";
}

// ----------------------------------------------------------------------------
// fgetText() specializations for fwscanf differences: get a value from a text file
// ----------------------------------------------------------------------------
void fgetText(FILE* f, char& v)
{
    const wchar_t* formatString = GetFormatString(v);
    int rc = fwscanf(f, formatString, &v);
    if (rc == 0)
        RuntimeError("error reading value from file (invalid format): %ls", formatString);
    else if (rc == EOF)
        RuntimeError("error reading from file: %s", strerror(errno));
    assert(rc == 1);
}
void fgetText(FILE* f, wchar_t& v)
{
    const wchar_t* formatString = GetFormatString(v);
    int rc = fwscanf(f, formatString, &v);
    if (rc == 0)
        RuntimeError("error reading value from file (invalid format): %ls", formatString);
    else if (rc == EOF)
        RuntimeError("error reading from file: %s", strerror(errno));
    assert(rc == 1);
}

// ----------------------------------------------------------------------------
// fopenOrDie(): like fopen() but terminate with err msg in case of error.
// A pathname of "-" returns stdout or stdin, depending on mode, and it will
// change the binary mode if 'b' or 't' are given. If you use this, make sure
// not to fclose() such a handle.
// ----------------------------------------------------------------------------

static const wchar_t* strchr(const wchar_t* s, wchar_t v)
{
    return wcschr(s, v);
}

// pathname is "-" -- open stdin or stdout. Changes bin mode if 'b' or 't' given.
template <class _T>
FILE* fopenStdHandle(const _T* mode)
{
    FILE* f = strchr(mode, 'r') ? stdin : stdout;
    if (strchr(mode, 'b') || strchr(mode, 't')) // change binary mode
        fsetmode(f, strchr(mode, 'b') ? 'b' : 't');
    return f;
}

FILE* fopenOrDie(const string& pathname, const char* mode)
{
    FILE* f = (pathname[0] == '-') ? fopenStdHandle(mode) : fopen(pathname.c_str(), mode);
    if (f == NULL)
    {
        RuntimeError("error opening file '%s': %s", pathname.c_str(), strerror(errno));
    }
    if (strchr(mode, 'S'))
    {
        // If optimized for sequential access, then use large buffer. OK if it fails
        setvbuf(f, NULL, _IOFBF, LARGE_BUF_SIZE);
    }
    return f;
}

FILE* fopenOrDie(const wstring& pathname, const wchar_t* mode)
{
    FILE* f = (pathname[0] == '-') ? fopenStdHandle(mode) : _wfopen(pathname.c_str(), mode);
    if (f == NULL)
    {
        RuntimeError("error opening file '%ls': %s", pathname.c_str(), strerror(errno));
    }
    if (strchr(mode, 'S'))
    {
        // If optimized for sequential access, then use large buffer. OK if it fails
        setvbuf(f, NULL, _IOFBF, LARGE_BUF_SIZE);
    }
    return f;
}

// ----------------------------------------------------------------------------
// set mode to binary or text (pass 'b' or 't')
// ----------------------------------------------------------------------------

void fsetmode(FILE* f, char type)
{
    if (type != 'b' && type != 't')
    {
        RuntimeError("fsetmode: invalid type '%c'", type);
    }
#ifdef UNDER_CE           // winCE and win32 have different return types for _fileno
    FILE* fd = fileno(f); // note: no error check possible
#else
    int fd = fileno(f); // note: no error check possible
#endif
    int rc = (type == 'b' ? SET_BINARY_MODE(fd) : SET_TEXT_MODE(fd));
    if (rc == -1)
    {
        RuntimeError("error changing file mode: %s", strerror(errno));
    }
}

// ----------------------------------------------------------------------------
// freadOrDie(): like fread() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void freadOrDie(void* ptr, size_t size, size_t count, FILE* f)
{
    size_t limit = max(READ_SIZE_LIMIT / size, (size_t)1);  // Normalize by size, as fread() expects units, not bytes

    // \\XXX\C$ reads are limited, with some randomness (e.g. 48 MB), on Windows 7 32 bit, so we break this into chunks of some MB. Meh.
    while (count > 0)
    {
        size_t chunkn = min(count, limit);
        size_t n = fread(ptr, size, chunkn, f);
        if (n != chunkn)
            RuntimeError("error reading from file: %s", strerror(errno));
        count -= n;
        ptr = n * size + (char*) ptr;
    }
}

#ifdef _WIN32
void freadOrDie(void* ptr, size_t size, size_t count, const HANDLE f)
{
    // \\XXX\C$ reads are limited, with some randomness (e.g. 48 MB), on Windows 7 32 bit, so we break this into chunks of some MB. Meh.
    while (count > 0)
    {
        DWORD chunkn = min((DWORD)(count * size), READ_SIZE_LIMIT);
        DWORD n;
        ReadFile(f, ptr, chunkn, &n, NULL);
        if (n != chunkn)
            RuntimeError("error number for reading from file: %s", GetLastError());
        count -= (size_t)(n / size);
        ptr = n + (char*) ptr;
    }
}
#endif

// ----------------------------------------------------------------------------
// fwriteOrDie(): like fwrite() but terminate with err msg in case of error;
// Windows C std lib fwrite() has problems writing >100 MB at a time (fails
// with Invalid Argument error), so we break it into chunks (yak!!)
// ----------------------------------------------------------------------------

void fwriteOrDie(const void* ptr, size_t size, size_t count, FILE* f)
{
    const char* p1 = (const char*) ptr;
    size_t totalBytes = size * count;
    while (totalBytes > 0)
    {
        size_t wantWrite = totalBytes;
        if (wantWrite > WRITE_SIZE_LIMIT)
        {
            wantWrite = WRITE_SIZE_LIMIT;
        }
        size_t n = fwrite((const void*) p1, 1, wantWrite, f);
        if (n != wantWrite)
        {
            RuntimeError("error writing to file (ptr=0x%08lx, size=%d, count=%d, writing %d bytes after %d): %s",
                         (unsigned long) (size_t) ptr, (int) size, (int) count, (int) wantWrite,
                         (int) (size * count - totalBytes),
                         strerror(errno));
        }
        totalBytes -= wantWrite;
        p1 += wantWrite;
    }
}

#ifdef _WIN32
void fwriteOrDie(const void* ptr, size_t size, size_t count, const HANDLE f)
{
    const char* p1 = (const char*) ptr;
    DWORD totalBytes = (DWORD)(size * count);
    while (totalBytes > 0)
    {
        DWORD wantWrite = totalBytes;
        if (wantWrite > WRITE_SIZE_LIMIT)
        {
            wantWrite = WRITE_SIZE_LIMIT;
        }
        DWORD byteWritten = 0;
        if (WriteFile(f, (const void*) p1, wantWrite, &byteWritten, NULL) == false)
        {
            RuntimeError("error writing to file (ptr=0x%08lx, size=%d,"
                         " count=%d, writing %d bytes after %d): %s",
                         ptr, size, count, (int) wantWrite,
                         (int) (size * count - totalBytes),
                         strerror(errno));
        }
        totalBytes -= wantWrite;
        p1 += wantWrite;
    }
}
#endif

long fseekOrDie(FILE* f, long offset, int mode)
{
    long curPos = ftell(f);
    if (curPos == -1L)
    {
        RuntimeError("error seeking: %s", strerror(errno));
    }
    int rc = fseek(f, offset, mode);
    if (rc != 0)
    {
        RuntimeError("error seeking: %s", strerror(errno));
    }
    return curPos;
}

// ----------------------------------------------------------------------------
// fprintfOrDie(): like fprintf() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

#pragma warning(push)
#pragma warning(disable : 4793) // 'vararg' : causes native code generation
void fprintfOrDie(FILE* f, const char* fmt, ...)
{
    va_list arg_ptr;
    va_start(arg_ptr, fmt);
    int rc = vfprintf(f, fmt, arg_ptr);
    if (rc < 0)
    {
        RuntimeError("error writing to file: %s", strerror(errno));
    }
}
#pragma warning(pop)

// ----------------------------------------------------------------------------
// fsyncOrDie(): like fsync() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void fsyncOrDie(FILE* f)
{
    int fd = fileno(f);
    if (fd == -1)
    {
        RuntimeError("unable to convert file handle to file descriptor: %s", strerror(errno));
    }

    // Ensure that all data is synced before returning from this function
#ifdef _WIN32
    if (!FlushFileBuffers((HANDLE)_get_osfhandle(fd)))
    {
        RuntimeError("error syncing to file: %d", (int) ::GetLastError());
    }
#else
    int rc = fsync(fd);
    if (rc != 0)
    {
        RuntimeError("error syncing to file: %s", strerror(errno));
    }
#endif
}

// ----------------------------------------------------------------------------
// fflushOrDie(): like fflush() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void fflushOrDie(FILE* f)
{
    int rc = fflush(f);
    if (rc != 0)
    {
        RuntimeError("error flushing to file: %s", strerror(errno));
    }
}

// ----------------------------------------------------------------------------
// filesize(): determine size of the file in bytes (with open file)
// ----------------------------------------------------------------------------
size_t filesize(FILE* f)
{
#ifdef _WIN32
    size_t curPos = _ftelli64(f);
    if (curPos == -1L)
    {
        RuntimeError("error determining file position: %s", strerror(errno));
    }
    int rc = _fseeki64(f, 0, SEEK_END);
    if (rc != 0)
        RuntimeError("error seeking to end of file: %s", strerror(errno));
    size_t len = _ftelli64(f);
    if (len == -1L)
        RuntimeError("error determining file position: %s", strerror(errno));
    rc = _fseeki64(f, curPos, SEEK_SET);
    if (rc != 0)
        RuntimeError("error resetting file position: %s", strerror(errno));
    return len;
#else // TODO: test this
    struct stat stat_buf;
    int rc = fstat(fileno(f), &stat_buf);
    if (rc != 0)
        RuntimeError("error determining length of file: %s", strerror(errno));
    static_assert(sizeof(stat_buf.st_size) >= sizeof(uint64_t), "struct stat not compiled for 64-bit mode");
    return stat_buf.st_size;
#endif
}

// filesize(): determine size of the file in bytes (with pathname)
size_t filesize(const wchar_t* pathname)
{
    FILE* f = fopenOrDie(pathname, L"rb");
    try
    {
        size_t len = filesize(f);
        fclose(f);
        return (size_t) len;
    }
    catch (...)
    {
        fclose(f);
        throw;
    }
}

#ifndef UNDER_CE // no 64-bit under winCE

// filesize64(): determine size of the file in bytes (with pathname)
int64_t filesize64(const wchar_t* pathname)
{
#ifdef _WIN32
    struct _stat64 fileinfo;
    if (_wstat64(pathname, &fileinfo) == -1)
        return 0;
    else
        return fileinfo.st_size;
#else
    return filesize(pathname);
#endif
}
#endif

// ----------------------------------------------------------------------------
// fget/setpos(): seek functions with error handling
// ----------------------------------------------------------------------------

uint64_t fgetpos(FILE* f)
{
#ifdef _MSC_VER // standard does not allow to cast between fpos_t and integer numbers, and indeed it does not work on Linux (but on Windows and GCC)
    fpos_t post;
    int rc = ::fgetpos(f, &post);
    if (rc != 0)
        RuntimeError("error getting file position: %s", strerror(errno));
#else
    auto pos = ftello(f);
    uint64_t post = (uint64_t) pos;
    static_assert(sizeof(post) >= sizeof(pos), "64-bit file offsets not enabled");
    if ((decltype(pos)) post != pos)
        LogicError("64-bit file offsets not enabled");
#endif
    return post;
}

void fsetpos(FILE* f, uint64_t reqpos)
{
#ifdef _MSC_VER // standard does not allow to cast between fpos_t and integer numbers, and indeed it does not work on Linux (but on Windows and GCC)
#if (_MSC_VER <= 1800) // Note: this does not trigger if loaded in vs2013 mode in vs2015!
    // Visual Studio's ::fsetpos() flushes the read buffer. This conflicts with a situation where
    // we generally read linearly but skip a few bytes or KB occasionally, as is
    // the case in speech recognition tools. This requires a number of optimizations.

    uint64_t curpos = fgetpos(f);
    uint64_t cureob = curpos + f->_cnt; // UGH: we mess with an internal structure here
    while (reqpos >= curpos && reqpos < cureob)
    {
        // if we made it then do not call fsetpos()
        if (reqpos == fgetpos(f))
            return;

        // if we seek within the existing buffer, then just move to the position by dummy reads
        char buf[65536];
        size_t n = min((size_t) reqpos - (size_t) curpos, _countof(buf));
        fread(buf, sizeof(buf[0]), n, f); // (this may fail, but really shouldn't)
        curpos += n;

        // since we mess with f->_cnt, if something unexpected happened to the buffer then back off
        if (curpos != fgetpos(f) || curpos + f->_cnt != cureob)
            break; // oops
    }
#else
    // special hack for VS CRT (for VS2015)
    // Visual Studio's ::fsetpos() flushes the read buffer. This conflicts with a situation where
    // we generally read linearly but skip a few bytes or KB occasionally, as is
    // the case in speech recognition tools. This requires a number of optimizations.
#define MAX_FREAD_SKIP 65536

    // forward seeks up to 64KiB are simulated
    // through a dummy read instead of fsetpos to
    // the new position.
    uint64_t curpos = fgetpos(f);
    size_t n = min((size_t)reqpos - (size_t)curpos, (size_t)MAX_FREAD_SKIP);

    // TODO: if we only skip a limited number of bytes, fread() them
    //       instead of fsetpos() to the new position since the vs2015
    //       libraries might drop the internal buffer and thus have to re-read
    //       from the new position, somthing that costs performance.
    if (n < MAX_FREAD_SKIP)
    {
        // in case we  stay in the internal buffer, no fileio is needed for this operation.
        char buf[MAX_FREAD_SKIP];
        fread(buf, sizeof(buf[0]), n, f); // (this may fail, but really shouldn't)

        // if we made it then do not call fsetpos()
        if (reqpos == fgetpos(f))
            return;
    }
#undef MAX_FREAD_SKIP
#endif // end special hack for VS CRT

    // actually perform the seek
    fpos_t post = reqpos;
    int rc = ::fsetpos(f, &post);
#else // assuming __unix__
    off_t post = (off_t) reqpos;
    static_assert(sizeof(off_t) >= sizeof(reqpos), "64-bit file offsets not enabled");
    if ((decltype(reqpos)) post != reqpos)
        LogicError("64-bit file offsets not enabled");
    int rc = fseeko(f, post, SEEK_SET);
#endif
    if (rc != 0)
        RuntimeError("error setting file position: %s", strerror(errno));
}

// ----------------------------------------------------------------------------
// unlinkOrDie(): unlink() with error handling
// ----------------------------------------------------------------------------

void unlinkOrDie(const std::string& pathname)
{
    if (unlink(pathname.c_str()) != 0 && errno != ENOENT) // if file is missing that's what we want
        RuntimeError("error deleting file '%s': %s", pathname.c_str(), strerror(errno));
}
void unlinkOrDie(const std::wstring& pathname)
{
    if (_wunlink(pathname.c_str()) != 0 && errno != ENOENT) // if file is missing that's what we want
        RuntimeError("error deleting file '%ls': %s", pathname.c_str(), strerror(errno));
}

// ----------------------------------------------------------------------------
// renameOrDie(): rename() with error handling
// ----------------------------------------------------------------------------

void renameOrDie(const std::string& from, const std::string& to)
{
#ifdef _WIN32
    // deleting destination file if exits (to match Linux semantic)
    if (fexists(to.c_str()) && !DeleteFileA(to.c_str()))
        RuntimeError("error deleting file: '%s': %d", to.c_str(), GetLastError());

    if (!MoveFileA(from.c_str(), to.c_str()))
        RuntimeError("error renaming file '%s': %d", from.c_str(), GetLastError());
#else
    // Delete destination file if it exists
    // WORKAROUND: "rename" should do this but this is a workaround
    // to the HDFS FUSE implementation's bug of failing to do so
    // workaround for FUSE rename when running on Philly
    unlinkOrDie(to);
    if (rename(from.c_str(), to.c_str()) != 0)
    {
        RuntimeError("error renaming file '%s': %s", from.c_str(), strerror(errno));
    }
#endif
}

void renameOrDie(const std::wstring& from, const std::wstring& to)
{
#ifdef _WIN32
    // deleting destination file if exits (to match Linux semantic)
    if (fexists(to.c_str()) && !DeleteFileW(to.c_str()))
        RuntimeError("error deleting file '%ls': %d", to.c_str(), GetLastError());

    if (!MoveFileW(from.c_str(), to.c_str()))
        RuntimeError("error renaming file '%ls': %d", from.c_str(), GetLastError());
#else
    renameOrDie(wtocharpath(from.c_str()).c_str(), wtocharpath(to.c_str()).c_str());
#endif
}

// ----------------------------------------------------------------------------
// fputstring(): write a 0-terminated string
// ----------------------------------------------------------------------------

void fputstring(FILE* f, const char* str)
{
    fwriteOrDie((void*) str, sizeof(*str), strnlen(str, SIZE_MAX) + 1, f); // SECURITY NOTE: string use has been reviewed
}

void fputstring(FILE* f, const std::string& str)
{
    fputstring(f, str.c_str());
}

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4127)
#endif
void fputstring(FILE* f, const wchar_t* str)
{
    if (sizeof(*str) == 2)
    {
        fwriteOrDie((void*) str, sizeof(*str), wcsnlen(str, SIZE_MAX) + 1, f); // SECURITY NOTE: string use has been reviewed
    }
    else if (sizeof(*str) == 4)
    {
        size_t strLen = wcsnlen(str, SIZE_MAX);
        std::unique_ptr<char16_t[]> str16(new char16_t[strLen + 1]);
        for (int i = 0; i < strLen; i++)
        {
            str16[i] = (char16_t) str[i];
        }
        str16[strLen] = 0;
        fwriteOrDie((void*) str16.get(), sizeof(*str) / 2, strLen + 1, f); // SECURITY NOTE: string use has been reviewed
    }
    else
    {
        RuntimeError("error: unknown encoding\n");
    }
}
#ifdef _WIN32
#pragma warning(pop)
#endif

void fputstring(FILE* f, const std::wstring& str)
{
    fputstring(f, str.c_str());
}

// ----------------------------------------------------------------------------
// fexists(): test if a file exists
// ----------------------------------------------------------------------------

bool fexists(const wchar_t* pathname)
{
#ifdef _MSC_VER
    WIN32_FIND_DATAW findFileData;
    HANDLE hFind = FindFirstFileW(pathname, &findFileData);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        FindClose(hFind);
        return true;
    }
    else
    {
        return false;
    }
#else
    auto_file_ptr f(_wfopen(pathname, L"r"));
    return f != nullptr;
#endif
}

bool fexists(const char* pathname)
{
#ifdef _MSC_VER
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA(pathname, &findFileData);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        FindClose(hFind);
        return true;
    }
    else
    {
        return false;
    }
#else
    auto_file_ptr f(fopen(pathname, "r"));
    return f != nullptr;
#endif
}

// ----------------------------------------------------------------------------
// funicode(): test if a file uses unicode by reading its BOM
// ----------------------------------------------------------------------------

bool funicode(FILE* f)
{
    unsigned short testCode;
    if (fread(&testCode, sizeof(short), 1, f) == 1 &&
        (int) testCode == 0xFEFF)
        return true;
    fseek(f, 0, SEEK_SET);
    // rewind (f);
    return false;
}

// ----------------------------------------------------------------------------
// fgetline(): like fgets() but terminate with err msg in case of error;
// removes the newline character at the end (like gets());
// Returns 'buf' (always). buf guaranteed to be 0-terminated.
// ----------------------------------------------------------------------------

#ifdef __CYGWIN__ // strnlen() is somehow missing in Cygwin, which we use to quick-check GCC builds under Windows (although it is not a real target platform)
static inline size_t strnlen(const char* s, size_t n)
{
    return std::find(s, s + n, '\0') - s;
}
#endif

#ifdef UNDER_CE // strlen for char * not defined in winCE
static inline size_t strnlen(const char* s, size_t n)
{
    return std::find(s, s + n, '\0') - s;
}
#endif

static inline wchar_t* fgets(wchar_t* buf, int n, FILE* f)
{
    return fgetws(buf, n, f);
}
static inline size_t strnlen(wchar_t* s, size_t n)
{
    return wcsnlen(s, n);
}

template <class CHAR>
CHAR* fgetline(FILE* f, CHAR* buf, int size)
{
    // TODO: we should redefine this to write UTF-16 (which matters on GCC which defines wchar_t as 32 bit)
    CHAR* p = fgets(buf, size, f);
    if (p == NULL) // EOF reached: next time feof() = true
    {
        if (ferror(f))
            RuntimeError("error reading line: %s", strerror(errno));
        buf[0] = 0;
        return buf;
    }
    size_t n = strnlen(p, size);

    // check for buffer overflow

    if (n >= (size_t) size - 1)
    {
        basic_string<CHAR> example(p, n < 100 ? n : 100);
        uint64_t filepos = fgetpos(f); // (for error message only)
        RuntimeError("input line too long at file offset %d (max. %d characters allowed) [%s ...]", (int) filepos, (int) size - 1, msra::strfun::utf8(example).c_str());
    }

    // remove newline at end

    if (n > 0 && p[n - 1] == '\n') // UNIX and Windows style
    {
        n--;
        p[n] = 0;
        if (n > 0 && p[n - 1] == '\r') // Windows style
        {
            n--;
            p[n] = 0;
        }
    }
    else if (n > 0 && p[n - 1] == '\r') // Mac style
    {
        n--;
        p[n] = 0;
    }

    return buf;
}

// STL string version
std::string fgetline(FILE* f)
{
    vector<char> buf(BUF_SIZE);
    return fgetline(f, &buf[0], (int) buf.size());
}

// STL string version
std::wstring fgetlinew(FILE* f)
{
    vector<wchar_t> buf(BUF_SIZE);
    return fgetline(f, &buf[0], (int) buf.size());
}

// STL string version avoiding most memory allocations
void fgetline(FILE* f, std::string& s, std::vector<char>& buf)
{
    buf.resize(BUF_SIZE);
    const char* p = fgetline(f, &buf[0], (int) buf.size());
    s.assign(p);
}

void fgetline(FILE* f, std::wstring& s, std::vector<wchar_t>& buf)
{
    buf.resize(BUF_SIZE);
    const wchar_t* p = fgetline(f, &buf[0], (int) buf.size());
    s.assign(p);
}

// char buffer version
void fgetline(FILE* f, std::vector<char>& buf)
{
    buf.resize(BUF_SIZE);
    fgetline(f, &buf[0], (int) buf.size());
    buf.resize(strnlen(&buf[0], BUF_SIZE) + 1); // SECURITY NOTE: string use has been reviewed
}

void fgetline(FILE* f, std::vector<wchar_t>& buf)
{
    buf.resize(BUF_SIZE);
    fgetline(f, &buf[0], (int) buf.size());
    buf.resize(wcsnlen(&buf[0], BUF_SIZE) + 1); // SECURITY NOTE: string use has been reviewed
}

// read a 0-terminated string
const char* fgetstring(FILE* f, __out_z_cap(size) char* buf, int size)
{
    int i;
    for (i = 0;; i++)
    {
        int c = fgetc(f);
        if (c == EOF)
            RuntimeError("error reading string or missing 0: %s", strerror(errno));
        if (c == 0)
            break;
        if (i >= size - 1)
            RuntimeError("input line too long (max. %d characters allowed)", size - 1);
        buf[i] = (char) c;
    }
    assert(i < size);
    buf[i] = 0;
    return buf;
}

// read a 0-terminated wstring
string fgetstring(FILE* f)
{
    string res;
    for (;;)
    {
        int c = fgetc(f);
        if (c == EOF)
            RuntimeError("error reading string or missing 0: %s", strerror(errno));
        if (c == 0)
            break;
        res.push_back((char) c);
    }
    return res;
}

// read a 0-terminated string
const wchar_t* fgetstring(FILE* f, __out_z_cap(size) wchar_t* buf, int size)
{
    int i;
    for (i = 0;; i++)
    {
        // TODO: we should redefine this to write UTF-16 (which matters on GCC which defines wchar_t as 32 bit)
        wint_t c = fgetwc(f);
        if (c == WEOF)
            RuntimeError("error reading string or missing 0: %s", strerror(errno));
        if (c == 0)
            break;
        if (i >= size - 1)
        {
            RuntimeError("input line too long (max. %d wchar_tacters allowed)", size - 1);
        }
        buf[i] = (wchar_t) c;
    }
    assert(i < size);
    buf[i] = 0;
    return buf;
}

#if (_MSC_VER < 1800)
// read a 0-terminated wstring
wstring fgetwstring(FILE* f)
{
    // TODO: we should redefine this to write UTF-16 (which matters on GCC which defines wchar_t as 32 bit)
    wstring res;
    for (;;)
    {
        //
        // there is a known vc++ runtime bug: Microsoft Connect 768113
        // fgetwc can skip a byte in certain condition
        // this is already fixed in update release to VS 2012
        // for now the workaround is to use fgetc twice to simulate fgetwc
        //
        // wint_t c = fgetwc (f);
        int c1 = fgetc(f);
        int c2 = fgetc(f);

        // synthetic fgetc output to simulate fgetwc
        // note the order below works only for little endian
        wint_t c = (wint_t)((c2 << 8) | c1);
        if (c == WEOF)
            RuntimeError("error reading string or missing 0: %s", strerror(errno));
        if (c == 0)
            break;
        res.push_back((wchar_t) c);
    }
    return res;
}

#else
// read a 0-terminated wstring
wstring fgetwstring(FILE* f)
{
    // TODO: we should redefine this to write UTF-16 (which matters on GCC which defines wchar_t as 32 bit)
    wstring res;
    for (;;)
    {
        wint_t c = fgetwc(f);
        if (c == WEOF)
            RuntimeError("error reading string or missing 0: %s", strerror(errno));
        if (c == 0)
            break;
        res.push_back((wchar_t) c);
    }
    return res;
}
#endif

bool fskipspace(FILE* f)
{
    int count = 0;
    for (;; count++)
    {
        int c = fgetc(f);
        if (c == EOF) // hit the end
        {
            if (ferror(f))
                RuntimeError("error reading from file: %s", strerror(errno));
            break;
        }
        if (!isspace(c)) // end of space: undo getting that character
        {
            int rc = ungetc(c, f);
            if (rc != c)
                RuntimeError("error in ungetc(): %s", strerror(errno));
            break;
        }
    }
    return count > 0;
}

bool fskipwspace(FILE* f)
{
    // TODO: we should redefine this to write UTF-16 (which matters on GCC which defines wchar_t as 32 bit)
    int count = 0;
    for (;; count++)
    {
        wint_t c = fgetwc(f);
        if (c == WEOF) // hit the end
        {
            if (ferror(f))
                RuntimeError("error reading from file: %s", strerror(errno));
            break;
        }
        if (!iswspace(c)) // end of space: undo getting that character
        {
            wint_t rc = ungetwc(c, f);
            if (rc != c)
                RuntimeError("error in ungetc(): %s", strerror(errno));
            break;
        }
    }
    return count > 0;
}

// fskipNewLine(): skip all white space until end of line incl. the newline
// skip - skip the end of line if true, otherwise leave the end of line (but eat any leading space)
// returns false, true, or EOF
int fskipNewline(FILE* f, bool skip)
{
    int c;
    bool found = false;

    // skip white space

    do
    {
        c = fgetc(f);
    } while (c == ' ' || c == '\t');

    if (c == '\r' || c == '\n') // Accept any type of newline
    {
        found = true;
        if (skip)
            c = fgetc(f);
    }

    if ((found && !skip) ||
        !(c == '\r' || c == '\n'))
    {
        // if we found an EOF, return that unless there was a newline before the EOF
        if (c == EOF)
            return found ? (int) true : EOF;
        int rc = ungetc(c, f);
        if (rc != c)
            RuntimeError("error in ungetc(): %s", strerror(errno));
        return (int) found;
    }
    // if we get here we saw a newline
    return (int) true;
}

// read a space-terminated token
// ...TODO: eat trailing space like fscanf() doessurrounding space)
const char* fgettoken(FILE* f, __out_z_cap(size) char* buf, int size)
{
    fskipspace(f); // skip leading space
    int c = -1;
    int i;
    for (i = 0;; i++)
    {
        c = fgetc(f);
        if (c == EOF)
            break;
        if (isspace(c))
            break;
        if (i >= size - 1)
            RuntimeError("input token too long (max. %d characters allowed)", size - 1);
        buf[i] = (char) c;
    }
    // ... TODO: while (IsWhiteSpace (c)) c = fgetc (f);      // skip trailing space
    if (c != EOF)
    {
        int rc = ungetc(c, f);
        if (rc != c)
            RuntimeError("error in ungetc(): %s", strerror(errno));
    }
    assert(i < size);
    buf[i] = 0;
    return buf;
}

string fgettoken(FILE* f)
{
    char buf[80];
    return fgettoken(f, buf, sizeof(buf) / sizeof(*buf));
}

// read a space-terminated token
const wchar_t* fgettoken(FILE* f, __out_z_cap(size) wchar_t* buf, int size)
{
    // TODO: we should redefine this to write UTF-16 (which matters on GCC which defines wchar_t as 32 bit)
    fskipwspace(f); // skip leading space
    wint_t c = WEOF;
    int i;
    for (i = 0;; i++)
    {
        c = fgetwc(f);
        if (c == WEOF)
            break;
        if (iswspace(c))
            break;
        if (i >= size - 1)
            RuntimeError("input token too long (max. %d wchar_tacters allowed)", size - 1);
        buf[i] = (wchar_t) c;
    }
    // ... TODO: while (IsWhiteSpace (c)) c = fgetc (f);      // skip trailing space
    if (c != WEOF)
    {
        int rc = ungetwc(c, f);
        if (rc != c)
            RuntimeError("error in ungetwc(): %s", strerror(errno));
    }
    assert(i < size);
    buf[i] = 0;
    return buf;
}

wstring fgetwtoken(FILE* f)
{
    wchar_t buf[80];
    return fgettoken(f, buf, sizeof(buf) / sizeof(*buf));
}

template <>
int ftrygetText<bool>(FILE* f, bool& v)
{
    wchar_t c;
    int rc = ftrygetText(f, c);
    v = (c == L'T');
    return rc;
}

// ----------------------------------------------------------------------------
// fputText(): write a bool out as character
// ----------------------------------------------------------------------------
template <>
void fputText<bool>(FILE* f, bool v)
{
    fputText(f, v ? L'T' : L'F');
}

// ----------------------------------------------------------------------------
// fgetTag(): read a 4-byte tag & return as a string
// ----------------------------------------------------------------------------

std::string fgetTag(FILE* f)
{
    char tag[5];
    freadOrDie(&tag[0], sizeof(tag[0]), 4, f);
    tag[4] = 0;
    return std::string(tag);
}

// ----------------------------------------------------------------------------
// fcheckTag(): read a 4-byte tag & verify it; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcheckTag(FILE* f, const char* expectedTag)
{
    fcompareTag(fgetTag(f), expectedTag);
}

void fcheckTag_ascii(FILE* f, const string& expectedTag)
{
    char buf[20]; // long enough for a tag
    fskipspace(f);
    fgettoken(f, buf, sizeof(buf) / sizeof(*buf));
    if (expectedTag != buf)
    {
        RuntimeError("invalid tag '%s' found; expected '%s'", buf, expectedTag.c_str());
    }
}

// ----------------------------------------------------------------------------
// fcompareTag(): compare two tags; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcompareTag(const string& readTag, const string& expectedTag)
{
    if (readTag != expectedTag)
    {
        RuntimeError("invalid tag '%s' found; expected '%s'",
                     readTag.c_str(), expectedTag.c_str());
    }
}

// ----------------------------------------------------------------------------
// fputTag(): write a 4-byte tag
// ----------------------------------------------------------------------------

void fputTag(FILE* f, const char* tag)
{
    const int TAG_LEN = 4;
    assert(strnlen(tag, TAG_LEN + 1) == TAG_LEN);
    fwriteOrDie((void*) tag, sizeof(*tag), strnlen(tag, TAG_LEN), f);
}

// ----------------------------------------------------------------------------
// fskipstring(): skip a 0-terminated string, such as a pad string
// ----------------------------------------------------------------------------

void fskipstring(FILE* f)
{
    char c;
    do
    {
        freadOrDie(&c, sizeof(c), 1, f);
    } while (c);
}

// ----------------------------------------------------------------------------
// fpad(): write a 0-terminated string to pad file to a n-byte boundary
// (note: file must be opened in binmode to work properly on DOS/Windows!!!)
// ----------------------------------------------------------------------------
void fpad(FILE* f, int n)
{
    // get current writing position
    int pos = ftell(f);
    if (pos == -1)
    {
        RuntimeError("error in ftell(): %s", strerror(errno));
    }
    // determine how many bytes are needed (at least 1 for the 0-terminator)
    // and create a dummy string of that length incl. terminator
    int len = n - (pos % n);
    const char dummyString[] = "MSR-Asia: JL+FS";
    size_t offset = sizeof(dummyString) / sizeof(dummyString[0]) - len;
    assert(offset >= 0);
    fputstring(f, dummyString + offset);
}

// ----------------------------------------------------------------------------
// fgetbyte(): read a byte value
// ----------------------------------------------------------------------------

char fgetbyte(FILE* f)
{
    char v;
    freadOrDie(&v, sizeof(v), 1, f);
    return v;
}

// ----------------------------------------------------------------------------
// fgetshort(): read a short value
// ----------------------------------------------------------------------------

short fgetshort(FILE* f)
{
    short v;
    freadOrDie(&v, sizeof(v), 1, f);
    return v;
}

short fgetshort_bigendian(FILE* f)
{
    unsigned char b[2];
    freadOrDie(&b, sizeof(b), 1, f);
    return (short) ((b[0] << 8) + b[1]);
}

// ----------------------------------------------------------------------------
// fgetint24(): read a 3-byte (24-bit) int value
// ----------------------------------------------------------------------------

int fgetint24(FILE* f)
{
    int v;
    assert(sizeof(v) == 4);
    freadOrDie(&v, sizeof(v) - 1, 1, f); // only read 3 lower-order bytes
    v <<= 8;                             // shift up (upper 8 bits uninit'ed)
    v >>= 8;                             // shift down 8 bits with sign-extend
    return v;
}

// ----------------------------------------------------------------------------
// fgetint(): read an int value
// ----------------------------------------------------------------------------

int fgetint(FILE* f)
{
    int v;
    freadOrDie(&v, sizeof(v), 1, f);
    return v;
}

int fgetint_bigendian(FILE* f)
{
    unsigned char b[4];
    freadOrDie(&b, sizeof(b), 1, f);
    return (int) (((((b[0] << 8) + b[1]) << 8) + b[2]) << 8) + b[3];
}

int fgetint_ascii(FILE* f)
{
    fskipspace(f);
    int res = 0;
    char c;
    freadOrDie(&c, sizeof(c), 1, f);
    while (isdigit((unsigned char) c))
    {
        res = (10 * res) + (c - '0');
        freadOrDie(&c, sizeof(c), 1, f);
    }
    int rc = ungetc(c, f);
    if (rc != c)
    {
        RuntimeError("error in ungetc(): %s", strerror(errno));
    }
    return res;
}

// ----------------------------------------------------------------------------
// fgetlong(): read an long value
// ----------------------------------------------------------------------------

long fgetlong(FILE* f)
{
    long v;
    freadOrDie(&v, sizeof(v), 1, f);
    return v;
}

// ----------------------------------------------------------------------------
// fgetfloat(): read a float value
// ----------------------------------------------------------------------------

float fgetfloat(FILE* f)
{
    float v;
    freadOrDie(&v, sizeof(v), 1, f);
    return v;
}

float fgetfloat_bigendian(FILE* f)
{
    int bitpattern = fgetint_bigendian(f);
    return *((float*) &bitpattern);
}

float fgetfloat_ascii(FILE* f)
{
    float val;
    fskipspace(f);
    int rc = fscanf(f, "%f", &val); // security hint: safe overloads
    if (rc == 0)
        RuntimeError("error reading float value from file (invalid format): %s", strerror(errno));
    else if (rc == EOF)
        RuntimeError("error reading from file: %s", strerror(errno));
    assert(rc == 1);
    return val;
}

// ----------------------------------------------------------------------------
// fgetdouble(): read a double value
// ----------------------------------------------------------------------------

double fgetdouble(FILE* f)
{
    double v;
    freadOrDie(&v, sizeof(v), 1, f);
    return v;
}

#ifdef _WIN32

// ----------------------------------------------------------------------------
// fgetwav(): read an entire .wav file
// ----------------------------------------------------------------------------

void WAVEHEADER::prepareRest(int sampleCount)
{
    FmtLength = 16;

    wFormatTag = 1;
    nAvgBytesPerSec = nSamplesPerSec * nBlockAlign;

    riffchar[0] = 'R';
    riffchar[1] = 'I';
    riffchar[2] = 'F';
    riffchar[3] = 'F';
    if (sampleCount != -1)
    {
        DataLength = sampleCount * nBlockAlign;
        RiffLength = 36 + DataLength;
    }
    else
    {
        DataLength = 0xffffffff;
        RiffLength = 0xffffffff;
    }

    wavechar[0] = 'W';
    wavechar[1] = 'A';
    wavechar[2] = 'V';
    wavechar[3] = 'E';
    wavechar[4] = 'f';
    wavechar[5] = 'm';
    wavechar[6] = 't';
    wavechar[7] = ' ';

    datachar[0] = 'd';
    datachar[1] = 'a';
    datachar[2] = 't';
    datachar[3] = 'a';
}

void WAVEHEADER::prepare(unsigned int Fs, int Bits, int Channels, int SampleCount)
{
    nChannels = (short) Channels;
    nSamplesPerSec = Fs;
    nBlockAlign = (short) (Channels * (Bits / 8));
    nAvgBytesPerSec = Fs * nBlockAlign;
    wBitsPerSample = (short) Bits;

    prepareRest(SampleCount);
}

void WAVEHEADER::prepare(const WAVEFORMATEX& wfx, int sampleCount /* -1 for unknown */)
{
    nChannels = wfx.nChannels;
    nSamplesPerSec = wfx.nSamplesPerSec;
    nBlockAlign = wfx.nBlockAlign;
    wBitsPerSample = wfx.wBitsPerSample;

    prepareRest(sampleCount);
}

void WAVEHEADER::write(FILE* f)
{
    fputTag(f, "RIFF");
    fputint(f, RiffLength);
    fputTag(f, "WAVE");
    fputTag(f, "fmt ");
    fputint(f, FmtLength);
    fputshort(f, wFormatTag);
    fputshort(f, nChannels);
    fputint(f, nSamplesPerSec);
    fputint(f, nAvgBytesPerSec);
    fputshort(f, nBlockAlign);
    fputshort(f, wBitsPerSample);
    assert(FmtLength == 16);
    assert(wFormatTag == 1);
    fputTag(f, "data");
    fputint(f, DataLength);
    fflushOrDie(f);
}

/*static*/ void WAVEHEADER::update(FILE* f)
{
    long curPos = ftell(f);
    if (curPos == -1L)
    {
        RuntimeError("error determining file position: %s", strerror(errno));
    }
    unsigned int len = (unsigned int) filesize(f);
    unsigned int RiffLength = len - 8;
    unsigned int DataLength = RiffLength - 36;
    fseekOrDie(f, 4, SEEK_SET);
    fputint(f, RiffLength);
    fseekOrDie(f, 40, SEEK_SET);
    fputint(f, DataLength);
    fseekOrDie(f, curPos, SEEK_SET);
}

#endif

// ----------------------------------------------------------------------------
// fputbyte(): write a byte value
// ----------------------------------------------------------------------------

void fputbyte(FILE* f, char v)
{
    fwriteOrDie(&v, sizeof(v), 1, f);
}

// ----------------------------------------------------------------------------
// fputshort(): write a short value
// ----------------------------------------------------------------------------

void fputshort(FILE* f, short v)
{
    fwriteOrDie(&v, sizeof(v), 1, f);
}

// ----------------------------------------------------------------------------
// fputint24(): write a 3-byte (24-bit) int value
// ----------------------------------------------------------------------------

void fputint24(FILE* f, int v)
{
    assert(sizeof(v) == 4);
    fwriteOrDie(&v, sizeof(v) - 1, 1, f); // write low-order 3 bytes
}

// ----------------------------------------------------------------------------
// fputint(): write an int value
// ----------------------------------------------------------------------------

void fputint(FILE* f, int v)
{
    fwriteOrDie(&v, sizeof(v), 1, f);
}

// ----------------------------------------------------------------------------
// fputlong(): write an long value
// ----------------------------------------------------------------------------

void fputlong(FILE* f, long v)
{
    fwriteOrDie(&v, sizeof(v), 1, f);
}

// ----------------------------------------------------------------------------
// fputfloat(): write a float value
// ----------------------------------------------------------------------------

void fputfloat(FILE* f, float v)
{
    fwriteOrDie(&v, sizeof(v), 1, f);
}

// ----------------------------------------------------------------------------
// fputdouble(): write a double value
// ----------------------------------------------------------------------------

void fputdouble(FILE* f, double v)
{
    fwriteOrDie(&v, sizeof(v), 1, f);
}

// ----------------------------------------------------------------------------
// fputfile(): write a binary block or a string as a file
// ----------------------------------------------------------------------------

void fputfile(const wstring& pathname, const std::vector<char>& buffer)
{
    FILE* f = fopenOrDie(pathname, L"wb");
    try
    {
        if (buffer.size() > 0)
        { // ^^ otherwise buffer[0] is an illegal expression
            fwriteOrDie(&buffer[0], sizeof(buffer[0]), buffer.size(), f);
        }
        fcloseOrDie(f);
    }
    catch (...)
    {
        fclose(f);
        throw;
    }
}

void fputfile(const wstring& pathname, const std::wstring& string)
{
    FILE* f = fopenOrDie(pathname, L"wb");
    try
    {
        if (string.length() > 0)
        { // ^^ otherwise buffer[0] is an illegal expression
            fwriteOrDie(string.c_str(), sizeof(string[0]), string.length(), f);
        }
        fcloseOrDie(f);
    }
    catch (...)
    {
        fclose(f);
        throw;
    }
}

void fputfile(const wstring& pathname, const std::string& string)
{
    FILE* f = fopenOrDie(pathname, L"wb");
    try
    {
        if (string.length() > 0)
        { // ^^ otherwise buffer[0] is an illegal expression
            fwriteOrDie(string.c_str(), sizeof(string[0]), string.length(), f);
        }
        fcloseOrDie(f);
    }
    catch (...)
    {
        fclose(f);
        throw;
    }
}

// ----------------------------------------------------------------------------
// fgetfile(): load a file as a binary block
// ----------------------------------------------------------------------------

void fgetfile(const wstring& pathname, std::vector<char>& buffer)
{
    FILE* f = fopenOrDie(pathname, L"rb");
    size_t len = filesize(f);
    buffer.resize(len);
    if (buffer.size() > 0)
    { // ^^ otherwise buffer[0] is an illegal expression
        freadOrDie(&buffer[0], sizeof(buffer[0]), buffer.size(), f);
    }
    fclose(f);
}

void fgetfile(FILE* f, std::vector<char>& buffer)
{ // this version reads until eof
    buffer.resize(0);
    buffer.reserve(1000000); // avoid too many reallocations
    std::vector<char> inbuf;
    inbuf.resize(65536); // read in chunks of this size
    while (!feof(f))     // read until eof
    {
        size_t n = fread(&inbuf[0], sizeof(inbuf[0]), inbuf.size(), f);
        if (ferror(f))
        {
            RuntimeError("fgetfile: error reading from file: %s", strerror(errno));
        }
        buffer.insert(buffer.end(), inbuf.begin(), inbuf.begin() + n);
    }
    buffer.reserve(buffer.size());
}

// load it into RAM in one huge chunk
static size_t fgetfilechars(const std::wstring& path, vector<char>& buffer)
{
    auto_file_ptr f(fopenOrDie(path, L"rb"));
    size_t len = filesize(f);
    buffer.reserve(len + 1);
    freadOrDie(buffer, len, f);
    buffer.push_back(0); // this makes it a proper C string
    return len;
}

static void fgetfilechars(const std::wstring& path, vector<char>& buffer, size_t& len)
{
    len = fgetfilechars(path, buffer);
}

template <class LINES>
static void strtoklines(char* s, LINES& lines)
{
    for (char* p = strtok(s, "\r\n"); p; p = strtok(NULL, "\r\n"))
        lines.push_back(p);
}

void msra::files::fgetfilelines(const std::wstring& path, vector<char>& buffer, std::vector<std::string>& lines, int numberOfTries)
{
    size_t len = 0;
    msra::util::attempt(numberOfTries, [&]() // (can be reading from network)
    {
        // load it into RAM in one huge chunk
        fgetfilechars(path, buffer, len);
    });

    // parse into lines
    lines.resize(0);
    lines.reserve(len / 20);
    strtoklines(&buffer[0], lines);
}

// same as above but returning const char* (avoiding the memory allocation)
vector<char*> msra::files::fgetfilelines(const wstring& path, vector<char>& buffer, int numberOfTries)
{
    size_t len = 0;
    msra::util::attempt(numberOfTries, [&]() // (can be reading from network)
    {
        // load it into RAM in one huge chunk
        fgetfilechars(path, buffer, len);
    });
    
    // parse into lines
    vector<char*> lines;
    lines.reserve(len / 20);
    strtoklines(&buffer[0], lines);
    return lines;
}

// ----------------------------------------------------------------------------
// getfiletime(): access modification time
// ----------------------------------------------------------------------------

#ifndef _FILETIME_
//typedef struct _FILETIME { DWORD dwLowDateTime; DWORD dwHighDateTime; };    // from minwindef.h
typedef time_t FILETIME;
#else
bool operator>=(const FILETIME& targettime, const FILETIME& inputtime) // for use in fuptodate()
{
    return (targettime.dwHighDateTime > inputtime.dwHighDateTime) ||
           (targettime.dwHighDateTime == inputtime.dwHighDateTime && targettime.dwLowDateTime >= inputtime.dwLowDateTime);
}
#endif

#ifdef _WIN32
class auto_find_handle
{
    HANDLE h;
    auto_find_handle operator=(const auto_find_handle&);
    auto_find_handle(const auto_find_handle&);

public:
    auto_find_handle(HANDLE p_h)
        : h(p_h)
    {
    }
    ~auto_find_handle()
    {
        if (h != INVALID_HANDLE_VALUE)
        {
            int rc = ::FindClose(h);
            if ((rc == FINDCLOSE_ERROR) && !std::uncaught_exception())
            {
                RuntimeError("Release: Failed to close handle: %d", ::GetLastError());
            }
        }
    }
    operator HANDLE() const
    {
        return h;
    }
};
#endif

bool getfiletime(const wstring& path, FILETIME& time)
{ // return file modification time, false if cannot be determined
#ifdef _WIN32
    WIN32_FIND_DATAW findFileData;
    auto_find_handle hFind(FindFirstFileW(path.c_str(), &findFileData));
    if (hFind != INVALID_HANDLE_VALUE)
    {
        time = findFileData.ftLastWriteTime;
        return true;
    }
    else
        return false;
#else // TODO: test this; e.g. does st_mtime have the desired resolution?
    struct stat buf;
    int result;

    // Get data associated with "crt_stat.c":
    result = stat(wtocharpath(path.c_str()).c_str(), &buf);
    // Check if statistics are valid:
    if (result != 0)
        return false;

    time = buf.st_mtime;
    return true;
#endif
}

// ----------------------------------------------------------------------------
// expand_wildcards -- wildcard expansion of a path, including directories.
// ----------------------------------------------------------------------------

#ifdef _WIN32
// Win32-style variant of this function (in case we want to use it some day)
// Returns 0 in case of failure. May throw in case of bad_alloc.
static BOOL ExpandWildcards(wstring path, vector<wstring>& paths)
{
    // convert root to DOS filename convention
    for (size_t k = 0; k < path.length(); k++)
        if (path[k] == '/')
            path[k] = '\\';

    // remove terminating backslash
    size_t last = path.length() - 1;
    if (last >= 0 && path[last] == '\\')
        path.erase(last);

    // convert root to long filename convention
    // if (path.find (L"\\\\?\\") != 0)
    //    path = L"\\\\?\\" + root;

    // split off everything after first wildcard
    size_t wpos = path.find_first_of(L"*?");
    if (wpos == 2 && path[0] == '\\' && path[1] == '\\')
        wpos = path.find_first_of(L"*?", 4); // 4=skip "\\?\"
    if (wpos == wstring::npos)
    { // no wildcard: just return it
        paths.push_back(path);
        return TRUE;
    }

    // split off everything afterwards if any
    wstring rest; // remaining path after this directory
    size_t spos = path.find_first_of(L"\\", wpos + 1);
    if (spos != wstring::npos)
    {
        rest = path.substr(spos + 1);
        path.erase(spos);
    }

    // crawl folder
    WIN32_FIND_DATAW ffdata;
    auto_find_handle hFind(::FindFirstFileW(path.c_str(), &ffdata));
    if (hFind == INVALID_HANDLE_VALUE)
    {
        DWORD err = ::GetLastError();
        if (rest.empty() && err == 2)
            return TRUE; // no matching file: empty
        return FALSE;    // another error
    }
    size_t pos = path.find_last_of(L"\\");
    if (pos == wstring::npos)
        LogicError("unexpected missing \\ in path");
    wstring parent = path.substr(0, pos);
    do
    {
        // skip this and parent directory
        bool isDir = ((ffdata.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0);
        if (isDir && ffdata.cFileName[0] == '.')
            continue;

        wstring filename = parent + L"\\" + ffdata.cFileName;
        if (rest.empty())
        {
            paths.push_back(filename);
        }
        else if (isDir) // multi-wildcards: further expand
        {
            BOOL rc = ExpandWildcards(filename + L"\\" + rest, paths);
            rc; // error here means no match, e.g. Access Denied to one subfolder
        }
    } while (::FindNextFileW(hFind, &ffdata) != 0);
    return TRUE;
}
#endif

void expand_wildcards(const wstring& path, vector<wstring>& paths)
{
#ifdef _WIN32
    BOOL rc = ExpandWildcards(path, paths);
    if (!rc)
        RuntimeError("error in expanding wild cards '%ls': Win32 error %d", path.c_str(), (int) ::GetLastError());
#else
    // On Linux we have just the function for the job: glob
    glob_t globResult;
    if (glob(wtocharpath(path.c_str()).c_str(), GLOB_TILDE, NULL, &globResult) != 0)
    {
        RuntimeError("error in expanding wild cards '%ls': %s", path.c_str(), strerror(errno));
    }

    for (unsigned int i = 0; i < globResult.gl_pathc; ++i)
    {
        paths.push_back(msra::strfun::utf16(globResult.gl_pathv[i]));
    }
    globfree(&globResult);
#endif
}

// ----------------------------------------------------------------------------
// make_intermediate_dirs() -- make all intermediate dirs on a path
// ----------------------------------------------------------------------------

static void mkdir(const wstring& path)
{
    int rc = _wmkdir(path.c_str());
    if (rc >= 0 || errno == EEXIST)
        return; // no error or already existing --ok
#ifdef _WIN32   // bug in _wmkdir(): returns access_denied if folder exists but read-only --check existence
    if (errno == EACCES)
    {
        DWORD att = ::GetFileAttributesW(path.c_str());
        if (att != INVALID_FILE_ATTRIBUTES || (att & FILE_ATTRIBUTE_DIRECTORY) != 0)
            return; // ok
    }
#endif
    RuntimeError("mkdir: error creating intermediate directory %ls", path.c_str());
}

// make subdir of a file including parents
void msra::files::make_intermediate_dirs(const wstring& filepath)
{
    vector<wchar_t> buf;
    buf.resize(filepath.length() + 1, 0);
    wcscpy(&buf[0], filepath.c_str());
    wstring subpath;
    int skip = 0;
#ifdef _WIN32
    // On windows, if share (\\) then the first two levels (machine, share name) cannot be made.
    if ((buf[0] == '/' && buf[1] == '/') || (buf[0] == '\\' && buf[1] == '\\'))
    {
        subpath = L"/";
        skip = 2; // skip two levels (machine, share)
    }
#else
    // On unix, if the filepath starts with '/' then it is absolute
    // path and the created sub-paths should also start with '/'
    if (buf[0] == '/')
    {
        subpath = L"/";
    }
#endif
    // make all constituents except the filename (to make a dir, include a trailing slash)
    wchar_t* context = nullptr;
    for (const wchar_t* p = wcstok_s(&buf[0], L"/\\", &context); p; p = wcstok_s(NULL, L"/\\", &context))
    {
        if (subpath != L"" && subpath != L"/" && subpath != L"\\" && skip == 0)
        {
            mkdir(subpath);
        }
        else if (skip > 0)
            skip--; // skip this level
        // rebuild the final path
        if (subpath != L"")
            subpath += L"/";
        subpath += p;
    }
}

std::vector<std::wstring> msra::files::get_all_files_from_directory(const std::wstring& directory)
{
    std::vector<std::wstring> result;
#ifdef _WIN32
    WIN32_FIND_DATA ffd = {};
    HANDLE hFind = FindFirstFile(directory.c_str(), &ffd);
    if (INVALID_HANDLE_VALUE == hFind)
        RuntimeError("Cannot get information about directory '%ls'.", directory.c_str());

    do
    {
        if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        {
            result.push_back(ffd.cFileName);
        }
    } while (FindNextFile(hFind, &ffd) != 0);

    auto dwError = GetLastError();
    FindClose(hFind);

    if (dwError != ERROR_NO_MORE_FILES)
        RuntimeError("Error iterating directory '%ls'", directory.c_str());
#else
    std::string d = msra::strfun::utf8(directory);
    auto dirp = opendir(d.c_str());
    dirent *dp = nullptr;
    struct stat st = {};
    while ((dp = readdir(dirp)) != NULL)
    {
        const std::string fileName = dp->d_name;
        const std::string fullFileName = d + "/" + fileName;

        if (fileName == "." || fileName == "..")
            continue;

        if (stat(fullFileName.c_str(), &st) == -1)
            continue;

        if ((st.st_mode & S_IFDIR) != 0)
            continue;

        result.push_back(msra::strfun::utf16(fileName));
    }
    closedir(dirp);
#endif
    return result;
}

// ----------------------------------------------------------------------------
// fuptodate() -- test whether an output file is at least as new as an input file
// ----------------------------------------------------------------------------

// test if file 'target' is not older than 'input' --used for make mode
// 'input' must exist if 'inputrequired'; otherweise if 'target' exists, it is considered up to date
// 'target' may or may not exist
bool msra::files::fuptodate(const wstring& target, const wstring& input, bool inputrequired)
{
    FILETIME targettime;
    if (!getfiletime(target, targettime))
        return false; // target missing: need to update
    FILETIME inputtime;
    if (!getfiletime(input, inputtime))
        return !inputrequired; // input missing: if required, pretend to be out of date as to force caller to fail
    // up to date if target has higher time stamp
    return targettime >= inputtime; // note: uses an overload for WIN32 FILETIME (in Linux, FILETIME=time_t=size_t)
}

// separate string by separator
template<class String>
vector<String> SplitString(const String& str, const String& sep)
{
    vector<String> vstr;
    String csub;
    size_t ifound = 0;
    size_t ifoundlast = ifound;
    ifound = str.find_first_of(sep, ifound);
    while (ifound != String::npos)
    {
        csub = str.substr(ifoundlast, ifound - ifoundlast);
        if (!csub.empty())
            vstr.push_back(csub);

        ifoundlast = ifound + 1;
        ifound = str.find_first_of(sep, ifoundlast);
    }
    ifound = str.length();
    csub = str.substr(ifoundlast, ifound - ifoundlast);
    if (!csub.empty())
        vstr.push_back(csub);

    return vstr;
}

template vector<string>  SplitString(const  string& istr, const  string& sep);
template vector<wstring> SplitString(const wstring& istr, const wstring& sep);

static inline std::string wcstombs(const std::wstring& p) // output: MBCS
{
    size_t len = p.length();
    vector<char> buf(2 * len + 1); // max: 1 wchar => 2 mb chars
    fill(buf.begin(), buf.end(), 0);
    ::wcstombs(&buf[0], p.c_str(), 2 * len + 1);
    return std::string(&buf[0]);
}
static inline std::wstring mbstowcs(const std::string& p) // input: MBCS
{
    size_t len = p.length();
    vector<wchar_t> buf(len + 1); // max: >1 mb chars => 1 wchar
    fill(buf.begin(), buf.end(), (wchar_t) 0);
    // OACR_WARNING_SUPPRESS(UNSAFE_STRING_FUNCTION, "Reviewed OK. size checked. [rogeryu 2006/03/21]");
    ::mbstowcs(&buf[0], p.c_str(), len + 1);
    return std::wstring(&buf[0]);
}

wstring s2ws(const string& str)
{
#ifdef __unix__
    return mbstowcs(str);
#else
    typedef std::codecvt_utf8<wchar_t> convert_typeX;
    std::wstring_convert<convert_typeX, wchar_t> converterX;
    return converterX.from_bytes(str);

#endif
}

string ws2s(const wstring& wstr)
{
#ifdef __unix__
    return wcstombs(wstr);
#else
    typedef codecvt_utf8<wchar_t> convert_typeX;
    wstring_convert<convert_typeX, wchar_t> converterX;
    return converterX.to_bytes(wstr);
#endif
}
