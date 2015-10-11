//
// <copyright file="FileUtil.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//


#include "stdafx.h"

#ifndef UNDER_CE    // fixed-buffer overloads not available for wince
#ifdef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES  // fixed-buffer overloads for strcpy() etc.
#undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#endif
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif

#include "Basics.h"
#include "fileutil.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#ifndef __unix__
#include "windows.h"    // for FILETIME
#endif
#include <algorithm>    // for std::find

#ifndef UNDER_CE  // some headers don't exist under winCE - the appropriate definitions seem to be in stdlib.h
#include <fcntl.h>      // for _O_BINARY/TEXT - not needed for wince
#ifndef __unix__
#include <io.h>         // for _setmode()
#endif
#endif

#include <errno.h>

using namespace std;

// ----------------------------------------------------------------------------
// fopenOrDie(): like fopen() but terminate with err msg in case of error.
// A pathname of "-" returns stdout or stdin, depending on mode, and it will
// change the binary mode if 'b' or 't' are given. If you use this, make sure
// not to fclose() such a handle.
// ----------------------------------------------------------------------------

static const wchar_t * strchr (const wchar_t * s, wchar_t v) { return wcschr (s, v); }

// pathname is "-" -- open stdin or stdout. Changes bin mode if 'b' or 't' given.
template<class _T> FILE * fopenStdHandle (const _T * mode)
{
    FILE * f = strchr (mode, 'r') ? stdin : stdout;
#ifndef __unix__ // don't need binary/text distinction on unix
    if (strchr(mode, 'b') || strchr(mode, 't'))   // change binary mode
    {
        // switch to binary mode if not yet (in case it is stdin)
        int rc = _setmode (_fileno (f), strchr (mode, 'b') ? _O_BINARY : _O_TEXT);
        if (rc == -1)
            RuntimeError ("error switching stream to binary mode: %s", strerror (errno));
    }
#endif
    return f;
}

FILE * fopenOrDie (const STRING & pathname, const char * mode)
{
    FILE * f = (pathname[0] == '-') ? fopenStdHandle (mode) : fopen (pathname.c_str(), mode);
    if (f == NULL)
    {
        RuntimeError("error opening file '%s': %s", pathname.c_str(), strerror(errno));
    }
    if (strchr (mode, 'S'))
    {   // if optimized for sequential access then use large buffer
    setvbuf (f, NULL, _IOFBF, 10000000);    // OK if it fails
    }
    return f;
}

FILE * fopenOrDie (const WSTRING & pathname, const wchar_t * mode)
{
    FILE * f = (pathname[0] == '-') ? fopenStdHandle (mode) : _wfopen (pathname.c_str(), mode);
    if (f == NULL)
    {
        RuntimeError ("error opening file '%S': %s", pathname.c_str(), strerror (errno));
    }
    if (strchr (mode, 'S'))
    {   // if optimized for sequential access then use large buffer
        setvbuf (f, NULL, _IOFBF, 10000000);    // OK if it fails
    }
    return f;
}

// ----------------------------------------------------------------------------
// set mode to binary or text (pass 'b' or 't')
// ----------------------------------------------------------------------------

#ifndef __unix__ // don't need binary/text distinction on unix
void fsetmode(FILE * f, char type)
{
    if (type != 'b' && type != 't')
    {
        RuntimeError ("fsetmode: invalid type '%c'");
    }
#ifdef UNDER_CE // winCE and win32 have different return types for _fileno
    FILE *fd = _fileno (f);   // note: no error check possible
#else
    int fd = _fileno (f);   // note: no error check possible
#endif
    int mode = type == 'b' ? _O_BINARY : _O_TEXT;
    int rc = _setmode (fd, mode);
    if (rc == -1)
    {
    RuntimeError ("error changing file mode: %s", strerror (errno));
    }
}
#endif

// ----------------------------------------------------------------------------
// freadOrDie(): like fread() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void freadOrDie (void * ptr, size_t size, size_t count, FILE * f)
{
    // \\XXX\C$ reads are limited, with some randomness (e.g. 48 MB), on Windows 7 32 bit, so we break this into chunks of some MB. Meh.
    while (count > 0)
    {
        size_t chunkn = min (count, 15*1024*1024);  // BUGBUG: I surely meant this limit to be bytes, not units of 'size'...
        size_t n = fread (ptr, size, chunkn, f);
        if (n != chunkn)
            RuntimeError ("error reading from file: %s", strerror (errno));
        count -= n;
        ptr = n * size + (char*) ptr;
    }
}

void freadOrDie (void * ptr, size_t size, size_t count, const HANDLE f)
{
    // \\XXX\C$ reads are limited, with some randomness (e.g. 48 MB), on Windows 7 32 bit, so we break this into chunks of some MB. Meh.
    while (count > 0)
    {
        size_t chunkn = min (count * size, 15*1024*1024);  
        DWORD n ;
        ReadFile(f, ptr, (DWORD) chunkn, &n, NULL);
        if (n != chunkn)
            RuntimeError ("error number for reading from file: %s", GetLastError());
        count -= (size_t) (n / size);
        ptr = n + (char*) ptr;
    }
}

// ----------------------------------------------------------------------------
// fwriteOrDie(): like fwrite() but terminate with err msg in case of error;
// Windows C std lib fwrite() has problems writing >100 MB at a time (fails
// with Invalid Argument error), so we break it into chunks (yak!!)
// ----------------------------------------------------------------------------

void fwriteOrDie (const void * ptr, size_t size, size_t count, FILE * f)
{
    const char * p1 = (const char *) ptr;
    size_t totalBytes = size * count;
    while (totalBytes > 0)
    {
        size_t wantWrite = totalBytes;
#define LIMIT (16*1024*1024)    // limit to 16 MB at a time
        if (wantWrite > LIMIT)
        {
            wantWrite = LIMIT;
        }
        size_t n = fwrite ((const void *) p1, 1, wantWrite, f);
        if (n != wantWrite)
        {
            RuntimeError ("error writing to file (ptr=0x%08lx, size=%d,"
                " count=%d, writing %d bytes after %d): %s",
                ptr, size, count, (int) wantWrite,
                (int) (size * count - totalBytes),
                strerror (errno));
        }
        totalBytes -= wantWrite;
        p1 += wantWrite;
    }
}

void fwriteOrDie (const void * ptr, size_t size, size_t count, const HANDLE f)
{
    const char * p1 = (const char *) ptr;
    DWORD totalBytes = (DWORD) (size * count);
    while (totalBytes > 0)
    {
        DWORD wantWrite = totalBytes;
#define LIMIT (16*1024*1024)    // limit to 16 MB at a time
        if (wantWrite > LIMIT)
        {
            wantWrite = LIMIT;
        }
        DWORD byteWritten = 0 ;
        if (WriteFile(f, (const void *) p1, wantWrite, &byteWritten, NULL) == false)
        {
            RuntimeError ("error writing to file (ptr=0x%08lx, size=%d,"
                " count=%d, writing %d bytes after %d): %s",
                ptr, size, count, (int) wantWrite,
                (int) (size * count - totalBytes),
                strerror (errno));
        }
        totalBytes -= wantWrite;
        p1 += wantWrite;
    }
}


// ----------------------------------------------------------------------------
// fprintfOrDie(): like fprintf() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

#pragma warning(push)
#pragma warning(disable : 4793) // 'vararg' : causes native code generation
void fprintfOrDie (FILE * f, const char * fmt, ...)
{
    va_list arg_ptr;
    va_start (arg_ptr, fmt);
    int rc = vfprintf (f, fmt, arg_ptr);
    if (rc < 0)
    {
        RuntimeError ("error writing to file: %s", strerror (errno));
    }
}
#pragma warning(pop)

// ----------------------------------------------------------------------------
// fflushOrDie(): like fflush() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void fflushOrDie (FILE * f)
{
    int rc = fflush (f);
    if (rc != 0)
    {
    RuntimeError ("error flushing to file: %s", strerror (errno));
    }
}

// ----------------------------------------------------------------------------
// filesize(): determine size of the file in bytes (with open file)
// BUGBUG: how about files > 4 GB? 
//  [2/10/2015 erw]  update with _ftelli64 and _fseeki64, still a possible BUG in some platform, 
//  as size(size_t) is not guaranteed to be 64bit under 64bit OS 
//  but in practice, this is quite rare 
// ----------------------------------------------------------------------------
size_t filesize (FILE * f)
{
#ifdef _WIN32
    size_t curPos = _ftelli64 (f);
    if (curPos == -1L)
    {
    RuntimeError ("error determining file position: %s", strerror (errno));
    }
    int rc = _fseeki64 (f, 0, SEEK_END);
    if (rc != 0)
    {
    RuntimeError ("error seeking to end of file: %s", strerror (errno));
    }
    size_t len = _ftelli64 (f);
    if (len == -1L)
    {
    RuntimeError ("error determining file position: %s", strerror (errno));
    }
    rc = _fseeki64 (f, curPos, SEEK_SET);
    if (rc != 0)
    {
    RuntimeError ("error resetting file position: %s", strerror (errno));
    }
    return len;
#else
	// linux version 
    long curPos = ftell (f);
    if (curPos == -1L)
    {
        RuntimeError ("error determining file position: %s", strerror (errno));
    }
    int rc = fseek (f, 0, SEEK_END);
    if (rc != 0)
    {
        RuntimeError ("error seeking to end of file: %s", strerror (errno));
    }
    long len = ftell (f);
    if (len == -1L)
    {
        RuntimeError ("error determining file position: %s", strerror (errno));
    }
    rc = fseek (f, curPos, SEEK_SET);
    if (rc != 0)
    {
        RuntimeError ("error resetting file position: %s", strerror (errno));
    }
    return (size_t) len;
#endif 
}

// filesize(): determine size of the file in bytes (with pathname)
size_t filesize (const wchar_t * pathname)
{
    FILE * f = fopenOrDie (pathname, L"rb");
    try
    {
        size_t len = filesize (f);
        fclose (f);
        return (size_t) len;
    }
    catch (...)
    {
        fclose (f);
        throw;
    }
}

#ifndef UNDER_CE    // no 64-bit under winCE

// filesize64(): determine size of the file in bytes (with pathname)
int64_t filesize64 (const wchar_t * pathname)
{
    __stat64 fileinfo;
    if (_wstat64 (pathname,&fileinfo) == -1) 
        return 0;
    else
        return fileinfo.st_size;
}
#endif

// ----------------------------------------------------------------------------
// fseekOrDie(),ftellOrDie(), fget/setpos(): seek functions with error handling
// ----------------------------------------------------------------------------

long fseekOrDie (FILE * f, long offset, int mode)
{
    long curPos = ftell (f);
    if (curPos == -1L)
    {
    RuntimeError ("error seeking: %s", strerror (errno));
    }
    int rc = fseek (f, offset, mode);
    if (rc != 0)
    {
    RuntimeError ("error seeking: %s", strerror (errno));
    }
    return curPos;
}

uint64_t fgetpos (FILE * f)
{
    fpos_t post;
    int rc = ::fgetpos (f, &post);
    if (rc != 0)
        RuntimeError ("error getting file position: %s", strerror (errno));
    return post;
}

void fsetpos (FILE * f, uint64_t reqpos)
{
    // ::fsetpos() flushes the read buffer. This conflicts with a situation where
    // we generally read linearly but skip a few bytes or KB occasionally, as is
    // the case in speech recognition tools. This requires a number of optimizations.

    uint64_t curpos = fgetpos (f);
    uint64_t cureob = curpos + f->_cnt; // UGH: we mess with an internal structure here
    while (reqpos >= curpos && reqpos < cureob)
    {
        // if we made it then do not call fsetpos()
        if (reqpos == fgetpos (f))
            return;

        // if we seek within the existing buffer, then just move to the position by dummy reads
        char buf[65536];
        size_t n = min ((size_t) reqpos - (size_t) curpos, _countof (buf));
        fread (buf, sizeof (buf[0]), n, f);     // (this may fail, but really shouldn't)
        curpos += n;

        // since we mess with f->_cnt, if something unexpected happened to the buffer then back off
        if (curpos != fgetpos (f) || curpos + f->_cnt != cureob)
            break;                              // oops
    }

    // actually perform the seek
    fpos_t post = reqpos;
    int rc = ::fsetpos (f, &post);
    if (rc != 0)
        RuntimeError ("error setting file position: %s", strerror (errno));
}

// ----------------------------------------------------------------------------
// unlinkOrDie(): unlink() with error handling
// ----------------------------------------------------------------------------

void unlinkOrDie (const std::string & pathname)
{
    if (_unlink (pathname.c_str()) != 0 && errno != ENOENT)     // if file is missing that's what we want
    RuntimeError ("error deleting file '%s': %s", pathname.c_str(), strerror (errno));
}
void unlinkOrDie (const std::wstring & pathname)
{
    if (_wunlink (pathname.c_str()) != 0 && errno != ENOENT)    // if file is missing that's what we want
    RuntimeError ("error deleting file '%S': %s", pathname.c_str(), strerror (errno));
}

// ----------------------------------------------------------------------------
// renameOrDie(): rename() with error handling
// ----------------------------------------------------------------------------

#ifndef UNDER_CE // CE only supports Unicode APIs
void renameOrDie (const std::string & from, const std::string & to)
{
    if (!MoveFileA (from.c_str(),to.c_str()))
    RuntimeError ("error renaming: %s", GetLastError());
}
#endif

void renameOrDie (const std::wstring & from, const std::wstring & to)
{
    if (!MoveFileW (from.c_str(),to.c_str()))
    RuntimeError ("error renaming: %s", GetLastError());
}

// ----------------------------------------------------------------------------
// fexists(): test if a file exists
// ----------------------------------------------------------------------------

bool fexists (const wchar_t * pathname)
{
    WIN32_FIND_DATAW findFileData;
    HANDLE hFind = FindFirstFileW (pathname, &findFileData);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        FindClose (hFind);
        return true;
    }
    else
    {
        return false;
    }
}

#ifndef UNDER_CE // CE only supports Unicode APIs
bool fexists (const char * pathname)
{
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA (pathname, &findFileData);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        FindClose (hFind);
        return true;
    }
    else
    {
        return false;
    }
}
#endif

// ----------------------------------------------------------------------------
// funicode(): test if a file uses unicode by reading its BOM
// ----------------------------------------------------------------------------

bool funicode (FILE * f)
{
    unsigned short testCode;
    if (fread (&testCode, sizeof(short), 1, f) == 1 &&
        (int)testCode == 0xFEFF)
        return true;
    fseek (f,0,SEEK_SET);
    //rewind (f);
    return false;
}

// ----------------------------------------------------------------------------
// fgetline(): like fgets() but terminate with err msg in case of error;
// removes the newline character at the end (like gets());
// Returns 'buf' (always). buf guaranteed to be 0-terminated.
// ----------------------------------------------------------------------------

static inline wchar_t * fgets (wchar_t * buf, int n, FILE * f) { return fgetws (buf, n, f); }
static inline string _utf8 (const string & s) { return s; }
static inline string _utf8 (const wstring & s) { return msra::strfun::utf8 (s); }
static inline size_t strnlen (wchar_t * s, size_t n) { return wcsnlen (s, n); }

#ifdef UNDER_CE     // strlen for char * not defined in winCE
static inline size_t strnlen (const char *s, size_t n) { return std::find (s,s+n,'\0') - s; }
#endif

template<class CHAR>
CHAR * fgetline (FILE * f, CHAR * buf, int size)
{

    uint64_t filepos = fgetpos (f); // (for error message only)
    CHAR * p = fgets (buf, size, f);
    if (p == NULL)            // EOF reached: next time feof() = true
    {
        if (ferror (f))
            RuntimeError ("error reading line: %s", strerror (errno));
        buf[0] = 0;
        return buf;
    }
    size_t n = strnlen (p, size);

    // check for buffer overflow

    if (n >= (size_t) size -1)
    {
        basic_string<CHAR> example (p, n < 100 ? n : 100);
        RuntimeError ("input line too long at file offset %I64d (max. %d characters allowed) [%s ...]",
               filepos, size -1, _utf8 (example).c_str());
    }

    // remove newline at end

    if (n > 0 && p[n-1] == '\n')    // UNIX and Windows style
    {
        n--;
        p[n] = 0;
        if (n > 0 && p[n-1] == '\r')    // Windows style
        {
            n--;
            p[n] = 0;
        }
    }
    else if (n > 0 && p[n-1] == '\r')    // Mac style
    {
        n--;
        p[n] = 0;
    }

    return buf;
}

#if 0
const wchar_t * fgetline (FILE * f, wchar_t * buf, int size)
{
    wchar_t * p = fgetws (buf, size, f);
    if (p == NULL)            // EOF reached: next time feof() = true
    {
        if (ferror (f))
            RuntimeError ("error reading line: %s", strerror (errno));
        buf[0] = 0;
        return buf;
    }
    size_t n = wcsnlen (p, size); // SECURITY NOTE: string use has been reviewed

    // check for buffer overflow

    if (n >= (size_t) size -1)
    {
        wstring example (buf, min (n, 100));
        RuntimeError ("input line too long at file offset %U64d (max. %d characters allowed) [%S ...]",
               fgetpos (f), size -1, example.c_str());
    }

    // remove newline at end

    if (n > 0 && p[n-1] == L'\n')    // UNIX and Windows style
    {
        n--;
        p[n] = 0;
        if (n > 0 && p[n-1] == L'\r')    // Windows style
        {
            n--;
            p[n] = 0;
        }
    }
    else if (n > 0 && p[n-1] == L'\r')    // Mac style
    {
        n--;
        p[n] = 0;
    }

    return buf;
}
#endif

// STL string version
std::string fgetline (FILE * f)
{
    fixed_vector<char> buf (1000000);
    return fgetline (f, &buf[0], (int) buf.size());
}

// STL string version
std::wstring fgetlinew (FILE * f)
{
    fixed_vector<wchar_t> buf (1000000);
    return fgetline (f, &buf[0], (int) buf.size());
}

// STL string version avoiding most memory allocations
void fgetline (FILE * f, std::string & s, ARRAY<char> & buf)
{
    buf.resize (1000000);    // enough? // KIT: increased to 1M to be safe
    const char * p = fgetline (f, &buf[0], (int) buf.size());
    s.assign (p);
}

void fgetline (FILE * f, std::wstring & s, ARRAY<wchar_t> & buf)
{
    buf.resize (1000000);    // enough? // KIT: increased to 1M to be safe
    const wchar_t * p = fgetline (f, &buf[0], (int) buf.size());
    s.assign (p);
}

// char buffer version
void fgetline (FILE * f, ARRAY<char> & buf)
{
    const int BUF_SIZE = 1000000;    // enough? // KIT: increased to 1M to be safe
    buf.resize (BUF_SIZE);
    fgetline (f, &buf[0], (int) buf.size());
    buf.resize (strnlen (&buf[0], BUF_SIZE) +1); // SECURITY NOTE: string use has been reviewed
}

void fgetline (FILE * f, ARRAY<wchar_t> & buf)
{
    const int BUF_SIZE = 1000000;    // enough? // KIT: increased to 1M to be safe
    buf.resize (BUF_SIZE);
    fgetline (f, &buf[0], (int) buf.size());
    buf.resize (wcsnlen (&buf[0], BUF_SIZE) +1); // SECURITY NOTE: string use has been reviewed
}

// read a 0-terminated string
const char * fgetstring (FILE * f, __out_z_cap(size) char * buf, int size)
{
    int i;
    for (i = 0; ; i++)
    {
    int c = fgetc (f);
    if (c == EOF)
            RuntimeError ("error reading string or missing 0: %s", strerror (errno));
    if (c == 0) break;
    if (i >= size -1)
    {
        RuntimeError ("input line too long (max. %d characters allowed)", size -1);
    }
    buf[i] = (char) c;
    }
    ASSERT (i < size);
    buf[i] = 0;
    return buf;
}

const char * fgetstring (const HANDLE f, __out_z_cap(size) char * buf, int size)
{
    int i;
    for (i = 0; ; i++)
    {
        char c; 
        freadOrDie((void*) &c, sizeof(char), 1, f);
        if (c == (char) 0) break;
        if (i >= size -1)
        {
            RuntimeError ("input line too long (max. %d characters allowed)", size -1);
        }
        buf[i] = (char) c;
    }
    ASSERT (i < size);
    buf[i] = 0;
    return buf;
}

// read a 0-terminated wstring
wstring fgetwstring (FILE * f)
{
    wstring res;
    for (;;)
    {
    int c = fgetwc (f);
    if (c == EOF)
            RuntimeError ("error reading string or missing 0: %s", strerror (errno));
    if (c == 0) break;
        res.push_back ((wchar_t) c);
    }
    return res;
}

void fskipspace (FILE * f)
{
    for (;;)
    {
    int c = fgetc (f);
    if (c == EOF)       // hit the end
        {
            if (ferror (f))
                RuntimeError ("error reading from file: %s", strerror (errno));
            break;
        }
    if (!isspace (c))    // end of space: undo getting that character
        {
            int rc = ungetc (c, f);
            if (rc != c)
                RuntimeError ("error in ungetc(): %s", strerror (errno));
            break;
        }
    }
}

// fskipNewLine(): skip all white space until end of line incl. the newline
void fskipNewline (FILE * f)
{
    char c;

    // skip white space
    
    do
    {
    freadOrDie (&c, sizeof (c), 1, f);
    } while (c == ' ' || c == '\t');

    if (c == '\r')            // Windows-style CR-LF
    {
    freadOrDie (&c, sizeof (c), 1, f);
    }

    if (c != '\n')
    {
    RuntimeError ("unexpected garbage at end of line");
    }
}

// read a space-terminated token
// ...TODO: eat trailing space like fscanf() doessurrounding space)
const char * fgettoken (FILE * f, __out_z_cap(size) char * buf, int size)
{
    fskipspace (f);                         // skip leading space
    int c = -1;
    int i;
    for (i = 0; ; i++)
    {
    c = fgetc (f);
    if (c == EOF) break;
    if (isspace (c)) break;
    if (i >= size -1)
        RuntimeError ("input token too long (max. %d characters allowed)", size -1);
    buf[i] = (char) c;
    }
    // ... TODO: while (isspace (c)) c = fgetc (f);      // skip trailing space
    if (c != EOF)
    {
    int rc = ungetc (c, f);
    if (rc != c)
        RuntimeError ("error in ungetc(): %s", strerror (errno));
    }
    ASSERT (i < size);
    buf[i] = 0;
    return buf;
}

STRING fgettoken (FILE * f)
{
    char buf[80];
    return fgettoken (f, buf, sizeof(buf)/sizeof(*buf));
}

// ----------------------------------------------------------------------------
// fputstring(): write a 0-terminated string
// ----------------------------------------------------------------------------

void fputstring (FILE * f, const char * str)
{
    fwriteOrDie ((void *) str, sizeof (*str), strnlen (str, SIZE_MAX)+1, f); // SECURITY NOTE: string use has been reviewed
}

void fputstring (const HANDLE f, const char * str)
{
    fwriteOrDie ((void *) str, sizeof (*str), strnlen (str, SIZE_MAX)+1, f); // SECURITY NOTE: string use has been reviewed
}

void fputstring (FILE * f, const std::string & str)
{
    fputstring (f, str.c_str());
}

void fputstring (FILE * f, const wchar_t * str)
{
    fwriteOrDie ((void *) str, sizeof (*str), wcsnlen (str, SIZE_MAX)+1, f); // SECURITY NOTE: string use has been reviewed
}

void fputstring (FILE * f, const std::wstring & str)
{
    fputstring (f, str.c_str());
}


// ----------------------------------------------------------------------------
// fgetTag(): read a 4-byte tag & return as a string
// ----------------------------------------------------------------------------

std::string fgetTag (FILE * f)
{
    char tag[5];
    freadOrDie (&tag[0], sizeof (tag[0]), 4, f);
    tag[4] = 0;
    return std::string (tag);
}

std::string fgetTag (const HANDLE f)
{
    char tag[5];
    freadOrDie (&tag[0], sizeof (tag[0]), 4, f);
    tag[4] = 0;
    return std::string (tag);
}

// ----------------------------------------------------------------------------
// fcheckTag(): read a 4-byte tag & verify it; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcheckTag (FILE * f, const char * expectedTag)
{
    fcompareTag (fgetTag (f), expectedTag);
}


void fcheckTag (const HANDLE f, const char * expectedTag)
{
    fcompareTag (fgetTag (f), expectedTag);
}

void fcheckTag_ascii (FILE * f, const STRING & expectedTag)
{
    char buf[20];    // long enough for a tag
    fskipspace (f);
    fgettoken (f, buf, sizeof(buf)/sizeof(*buf));
    if (expectedTag != buf)
    {
        RuntimeError ("invalid tag '%s' found; expected '%s'", buf, expectedTag.c_str());
    }
}

// ----------------------------------------------------------------------------
// fcompareTag(): compare two tags; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcompareTag (const STRING & readTag, const STRING & expectedTag)
{
    if (readTag != expectedTag)
    {
        RuntimeError ("invalid tag '%s' found; expected '%s'", 
               readTag.c_str(), expectedTag.c_str());
    }
}

// ----------------------------------------------------------------------------
// fputTag(): write a 4-byte tag
// ----------------------------------------------------------------------------

void fputTag (FILE * f, const char * tag)
{
    const int TAG_LEN = 4;
    ASSERT (strnlen (tag, TAG_LEN + 1) == TAG_LEN);
    fwriteOrDie ((void *) tag, sizeof (*tag), strnlen (tag, TAG_LEN), f);
}

void fputTag(const HANDLE f, const char * tag)
{
    const int TAG_LEN = 4;
    ASSERT (strnlen (tag, TAG_LEN + 1) == TAG_LEN);
    fwriteOrDie ((void *) tag, sizeof (*tag), strnlen (tag, TAG_LEN), f);
}

// ----------------------------------------------------------------------------
// fskipstring(): skip a 0-terminated string, such as a pad string
// ----------------------------------------------------------------------------

void fskipstring (FILE * f)
{
    char c;
    do
    {
    freadOrDie (&c, sizeof (c), 1, f);
    }
    while (c);
}

// ----------------------------------------------------------------------------
// fpad(): write a 0-terminated string to pad file to a n-byte boundary
// (note: file must be opened in binmode to work properly on DOS/Windows!!!)
// ----------------------------------------------------------------------------
void fpad (FILE * f, int n)
{
    // get current writing position
    int pos = ftell (f);
    if (pos == -1)
    {
    RuntimeError ("error in ftell(): %s", strerror (errno));
    }
    // determine how many bytes are needed (at least 1 for the 0-terminator)
    // and create a dummy string of that length incl. terminator
    int len = n - (pos % n);
    const char dummyString[] = "MSR-Asia: JL+FS";
    size_t offset = sizeof(dummyString)/sizeof(dummyString[0]) - len;
    ASSERT (offset >= 0);
    fputstring (f, dummyString + offset);
}
// ----------------------------------------------------------------------------
// fgetbyte(): read a byte value
// ----------------------------------------------------------------------------

char fgetbyte (FILE * f)
{
    char v;
    freadOrDie (&v, sizeof (v), 1, f);
    return v;
}

// ----------------------------------------------------------------------------
// fgetshort(): read a short value
// ----------------------------------------------------------------------------

short fgetshort (FILE * f)
{
    short v;
    freadOrDie (&v, sizeof (v), 1, f);
    return v;
}

short fgetshort_bigendian (FILE * f)
{
    unsigned char b[2];
    freadOrDie (&b, sizeof (b), 1, f);
    return (short) ((b[0] << 8) + b[1]);
}

// ----------------------------------------------------------------------------
// fgetint24(): read a 3-byte (24-bit) int value
// ----------------------------------------------------------------------------

int fgetint24 (FILE * f)
{
    int v;
    ASSERT (sizeof (v) == 4);
    freadOrDie (&v, sizeof (v) -1, 1, f);   // only read 3 lower-order bytes
    v <<= 8;                                // shift up (upper 8 bits uninit'ed)
    v >>= 8;                                // shift down 8 bits with sign-extend
    return v;
}

// ----------------------------------------------------------------------------
// fgetint(): read an int value
// ----------------------------------------------------------------------------

int fgetint (FILE * f)
{
    int v;
    freadOrDie (&v, sizeof (v), 1, f);
    return v;
}

int fgetint (const HANDLE f)
{
    int v;
    freadOrDie (&v, sizeof (v), 1, f);
    return v;
}

int fgetint_bigendian (FILE * f)
{
    unsigned char b[4];
    freadOrDie (&b, sizeof (b), 1, f);
    return (int) (((((b[0] << 8) + b[1]) << 8) + b[2]) << 8) + b[3];
}

int fgetint_ascii (FILE * f)
{
    fskipspace (f);
    int res = 0;
    char c;
    freadOrDie (&c, sizeof (c), 1, f);
    while (isdigit ((unsigned char)c))
    {
    res = (10 * res) + (c - '0');
    freadOrDie (&c, sizeof (c), 1, f);
    }
    int rc = ungetc (c, f);
    if (rc != c)
    {
    RuntimeError ("error in ungetc(): %s", strerror (errno));
    }
    return res;
}

// ----------------------------------------------------------------------------
// fgetfloat(): read a float value
// ----------------------------------------------------------------------------

float fgetfloat (FILE * f)
{
    float v;
    freadOrDie (&v, sizeof (v), 1, f);
    return v;
}

float fgetfloat_bigendian (FILE * f)
{
    int bitpattern = fgetint_bigendian (f);
    return *((float*) &bitpattern);
}

float fgetfloat_ascii (FILE * f)
{
    float val;
    fskipspace (f);
    int rc = fscanf (f, "%f", &val); // security hint: safe overloads
    if (rc == 0)
    RuntimeError ("error reading float value from file (invalid format): %s");
    else if (rc == EOF)
    RuntimeError ("error reading from file: %s", strerror (errno));
    ASSERT (rc == 1);
    return val;
}

// ----------------------------------------------------------------------------
// fgetdouble(): read a double value
// ----------------------------------------------------------------------------

double fgetdouble (FILE * f)
{
    double v;
    freadOrDie (&v, sizeof (v), 1, f);
    return v;
}

// ----------------------------------------------------------------------------
// fgetwav(): read an entire .wav file
// ----------------------------------------------------------------------------

void WAVEHEADER::prepareRest (int sampleCount)
{
    FmtLength   = 16; 

    wFormatTag      = 1;
    nAvgBytesPerSec = nSamplesPerSec * nBlockAlign;

    riffchar[0] = 'R';
    riffchar[1] = 'I';
    riffchar[2] = 'F';
    riffchar[3] = 'F';
    if (sampleCount != -1)
    {
        DataLength  = sampleCount * nBlockAlign;
        RiffLength  = 36 + DataLength;
    }
    else
    {
        DataLength  = 0xffffffff;
        RiffLength  = 0xffffffff;
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

void WAVEHEADER::prepare (unsigned int Fs, int Bits, int Channels, int SampleCount)
{
    nChannels       = (short) Channels; 
    nSamplesPerSec  = Fs; 
    nBlockAlign     = (short) (Channels * (Bits/8));
    nAvgBytesPerSec = Fs * nBlockAlign;
    wBitsPerSample  = (short) Bits;

    prepareRest (SampleCount);
}

void WAVEHEADER::prepare (const WAVEFORMATEX & wfx, int sampleCount /* -1 for unknown */)
{
    nChannels       = wfx.nChannels;
    nSamplesPerSec  = wfx.nSamplesPerSec;
    nBlockAlign     = wfx.nBlockAlign;
    wBitsPerSample  = wfx.wBitsPerSample;

    prepareRest (sampleCount);
}

void WAVEHEADER::write (FILE * f)
{
    fputTag (f, "RIFF");
    fputint (f, RiffLength);
    fputTag (f, "WAVE");
    fputTag (f, "fmt ");
    fputint (f, FmtLength);
    fputshort (f, wFormatTag);
    fputshort (f, nChannels);
    fputint (f, nSamplesPerSec);
    fputint (f, nAvgBytesPerSec);
    fputshort (f, nBlockAlign);
    fputshort (f, wBitsPerSample);
    ASSERT (FmtLength == 16);
    ASSERT (wFormatTag == 1);
    fputTag (f, "data");
    fputint (f, DataLength);
    fflushOrDie (f);
}

/*static*/ void WAVEHEADER::update (FILE * f)
{
    long curPos = ftell (f);
    if (curPos == -1L)
    {
    RuntimeError ("error determining file position: %s", strerror (errno));
    }
    unsigned int len = (unsigned int) filesize (f);
    unsigned int RiffLength = len - 8;
    unsigned int DataLength = RiffLength - 36;
    fseekOrDie (f, 4, SEEK_SET);
    fputint (f, RiffLength);
    fseekOrDie (f, 40, SEEK_SET);
    fputint (f, DataLength);
    fseekOrDie (f, curPos, SEEK_SET);
}

#if 0
unsigned int WAVEHEADER::read (FILE * f, signed short & wRealFormatTag, int & bytesPerSample)
{
    // read header
    fcheckTag (f, "RIFF");
    /*unsigned int riffLen = */ fgetint (f);
    fcheckTag (f, "WAVE");
    fcheckTag (f, "fmt ");
    unsigned int fmtLen = fgetint (f);
    wRealFormatTag = fgetshort (f);
    if (wRealFormatTag == -2)   // MARecorder.exe [Ivan Tashev] puts a -2 for
    {                           // 8-channel recordings (meaning unknown).
        wRealFormatTag = 1;     // Workaround: pretend it is 1 (seems safe)
    }
    (wRealFormatTag == 1 || wRealFormatTag == 7)
        || RuntimeError ("WAVEHEADER::read: wFormatTag=%d not supported for now", wRealFormatTag);
    unsigned short wChannels = fgetshort (f);
    unsigned long dwSamplesPerSec = fgetint (f);
    unsigned int sampleRate = dwSamplesPerSec;
    /*unsigned long dwAvgBytesPerSec = */ fgetint (f);
    unsigned short wBlockAlign = fgetshort (f);
    unsigned short wBitsPerSample = fgetshort (f);
    (wBitsPerSample <= 16) || RuntimeError ("WAVEHEADER::read: invalid wBitsPerSample %d", wBitsPerSample);
    bytesPerSample = wBitsPerSample / 8;
    (wBlockAlign == wChannels * bytesPerSample)
        || RuntimeError ("WAVEHEADER::read: wBlockAlign != wChannels*bytesPerSample not supported");
    while (fmtLen > 16) // unused extra garbage in header
    {
        fgetbyte (f);
        fmtLen--;
    }
    if (wRealFormatTag == 7)
    {
        (bytesPerSample == 1) || RuntimeError ("WAVEHEADER::read: invalid wBitsPerSample %d for mulaw", wBitsPerSample);
        fcheckTag (f, "fact");
        unsigned int factLen = fgetint (f);
        while (factLen > 0)
        {
            fgetbyte (f);
            factLen--;
        }
    }
    fcheckTag (f, "data");
    unsigned int dataLen = fgetint (f);
    unsigned int numSamples = dataLen / wBlockAlign;

    // prepare a nice wave header without junk (44 bytes, 16-bit PCM)
    prepare (sampleRate, wBitsPerSample, wChannels, numSamples);

    return numSamples;
}

static short toolULawToLinear(unsigned char p_ucULawByte)
{
    static short anExpLut[8] = { 0, 132, 396, 924, 1980, 4092, 8316, 16764 };
    short nSign, nExponent, nMantissa, nSample;

    p_ucULawByte=~p_ucULawByte;
    nSign=(p_ucULawByte & 0x80);
    nExponent=(p_ucULawByte >> 4) & 0x07;
    nMantissa=p_ucULawByte & 0x0F;
    nSample=anExpLut[nExponent]+(nMantissa<<(nExponent+3));
    if(nSign != 0) 
        nSample = -nSample;

    return nSample;
}

// fgetwavraw(): only read data of .wav file. For multi-channel data, samples
// are kept interleaved.
static void fgetwavraw(FILE * f, ARRAY<short> & wav, const WAVEHEADER & wavhd)
{
    int bytesPerSample = wavhd.wBitsPerSample / 8;  // (sample size on one channel)
    wav.resize (wavhd.DataLength / bytesPerSample);
    if (wavhd.wFormatTag == 7)    // mulaw
    {
        (wavhd.nChannels == 1) || RuntimeError ("fgetwav: wChannels=%d not supported for mulaw", wavhd.nChannels);
        ARRAY<unsigned char> data;
        int numSamples = wavhd.DataLength/wavhd.nBlockAlign;
        data.resize (numSamples);
        freadOrDie (&data[0], sizeof (data[0]), numSamples, f);
        for (int i = 0; i < numSamples; i++)
        {
            wav[i] = toolULawToLinear (data[i]);
        }
    }
    else if (bytesPerSample == 2)
    {   // note: we may be reading an interleaved multi-channel signal.
        freadOrDie (&wav[0], sizeof (wav[0]), wav.size(), f);
    }
    // ... TODO: support 8 bit linear PCM samples (implement when needed; samples scaled to 'short')
    else
    {
        RuntimeError ("bytesPerSample != 2 is not supported except mulaw format!\n");
    }
}

// ----------------------------------------------------------------------------
// fgetwav(): read an entire .wav file. Stereo is mapped to mono.
// ----------------------------------------------------------------------------

void fgetwav (FILE * f, ARRAY<short> & wav, int & sampleRate)
{
    WAVEHEADER wavhd;           // will be filled in for 16-bit PCM!!
    signed short wFormatTag;    // real format tag as found in data
    int bytesPerSample;         // bytes per sample as found in data

    unsigned int numSamples = wavhd.read (f, wFormatTag, bytesPerSample);
    sampleRate = (int) wavhd.nSamplesPerSec;

    if (wavhd.nChannels == 1)
    {
        fgetwavraw (f, wav, wavhd);
    }
    else if (wavhd.nChannels == 2)
    {
        //read raw data        
        ARRAY<short> buf;
        buf.resize(numSamples * 2);
        fgetwavraw(f, buf, wavhd);
        
        //map to mono
        wav.resize (numSamples);
        const short * p = &buf[0];
        for (int i = 0; i < (int) numSamples; i++)
        {
            int l = *p++;
            int r = *p++;
            int mono = ((l + r) + 1) >> 1;
            wav[i] = (short) mono;
        }
    }
    else
    {
        RuntimeError ("bytesPerSample/wChannels != 2 needs to be implemented");
    }
}

void fgetwav (const wstring & fn, ARRAY<short> & wav, int & sampleRate)
{
    auto_file_ptr f = fopenOrDie (fn, L"rbS");
    fgetwav (f, wav, sampleRate);
}

// ----------------------------------------------------------------------------
// ... TODO:
//  - rename this function!!
//  - also change to read header itself and return sample rate and channels
// fgetraw(): read data of multi-channel .wav file, and separate data of multiple channels. 
//            For example, data[i][j]: i is channel index, 0 means the first 
//            channel. j is sample index.
// ----------------------------------------------------------------------------

void fgetraw (FILE *f, ARRAY< ARRAY<short> > & data, const WAVEHEADER & wavhd)
{
    ARRAY<short> wavraw;
    fgetwavraw (f, wavraw, wavhd);
    data.resize (wavhd.nChannels);
    int numSamples = wavhd.DataLength/wavhd.nBlockAlign;
    ASSERT (numSamples == (int) wavraw.size() / wavhd.nChannels);

    for (int i = 0; i < wavhd.nChannels; i++)
    {
        data[i].resize (numSamples);

        for (int j = 0; j < numSamples; j++)
        {
            data[i][j] = wavraw[wavhd.nChannels*j + i];
        }
    }
}

// ----------------------------------------------------------------------------
// fgetwfx(), fputwfx(): direct access to simple WAV headers
// ----------------------------------------------------------------------------

// read header and skip to first data byte; return #samples
unsigned int fgetwfx (FILE * f, WAVEFORMATEX & wfx)
{
    // read header
    fcheckTag (f, "RIFF");
    /*unsigned int riffLen = */ fgetint (f);
    fcheckTag (f, "WAVE");
    fcheckTag (f, "fmt ");
    wfx.cbSize = sizeof (wfx);
    int fmtLen = fgetint (f);
    wfx.wFormatTag = fgetshort (f);
    if (wfx.wFormatTag == -2)   // MARecorder.exe [Ivan Tashev] puts a -2 for
    {                           // 8-channel recordings (meaning unknown).
        wfx.wFormatTag = 1;     // Workaround: pretend it is 1 (seems safe)
    }
    (wfx.wFormatTag == 1 || wfx.wFormatTag == 3 || wfx.wFormatTag == 7)
        || RuntimeError ("WAVEHEADER::read: wFormatTag=%d not supported for now", wfx.wFormatTag);
    wfx.nChannels = fgetshort (f);
    wfx.nSamplesPerSec = fgetint (f);
    wfx.nAvgBytesPerSec = fgetint (f);
    wfx.nBlockAlign = fgetshort (f);
    wfx.wBitsPerSample = fgetshort (f);
    // unused extra garbage in header
    for ( ; fmtLen > 16; fmtLen--) fgetbyte (f);
    fcheckTag (f, "data");
    unsigned int dataLen = fgetint (f);
    unsigned int numSamples = dataLen / wfx.nBlockAlign;
    return numSamples;
}

void fputwfx (FILE *f, const WAVEFORMATEX & wfx, unsigned int numSamples)
{
    unsigned int DataLength = numSamples * wfx.nBlockAlign;
    (DataLength / wfx.nBlockAlign == numSamples)
        || RuntimeError ("fputwfx: data size exceeds WAV header 32-bit range");
    unsigned int RiffLength = 36 + DataLength;
    unsigned int FmtLength  = 16; 
    // file header
    ASSERT (wfx.cbSize == 0 || wfx.cbSize == FmtLength + 2);
    fputTag (f, "RIFF");
    fputint (f, RiffLength);
    fputTag (f, "WAVE");
    // 'fmt ' chunk (to hold wfx)
    fputTag (f, "fmt ");
    fputint (f, FmtLength);
    fputshort (f, wfx.wFormatTag);
    fputshort (f, wfx.nChannels);
    fputint (f, wfx.nSamplesPerSec);
    fputint (f, wfx.nAvgBytesPerSec);
    fputshort (f, wfx.nBlockAlign);
    fputshort (f, wfx.wBitsPerSample);
    // data chunk
    fputTag (f, "data");
    fputint (f, DataLength);
    fflushOrDie (f);
}

// ----------------------------------------------------------------------------
// fputwav(): write an entire .wav file (16 bit PCM)
// ----------------------------------------------------------------------------

void fputwav (FILE * f, const vector<short> & wav, int sampleRate, int nChannels)
{
    f;wav;sampleRate;nChannels;
    // construct WAVEFORMATEX
    WAVEFORMATEX wfx;
    wfx.cbSize = 16 + 2;  //fmt data + extra data
    wfx.nAvgBytesPerSec = (DWORD)(sampleRate * nChannels * 2); //short: 2 bytes per sample
    wfx.nBlockAlign = (WORD)nChannels * 2; //short: 2bytes per sample
    wfx.nChannels = (WORD)nChannels;
    wfx.nSamplesPerSec = sampleRate;
    wfx.wBitsPerSample = 16;
    wfx.wFormatTag = WAVE_FORMAT_PCM;
    //putwfx
    fputwfx (f, wfx, (unsigned int) wav.size());
    // wrtie the data
    fwriteOrDie (&wav[0], sizeof(wav[0]), wav.size(), f);
}

void fputwav (const wstring & fn, const vector<short> & wav, int sampleRate, int nChannels)
{
    auto_file_ptr f = fopenOrDie (fn, L"wbS");
    fputwav (f, wav, sampleRate, nChannels);
    fflushOrDie (f);    // after this, fclose() (in destructor of f) cannot fail
}
#endif

// ----------------------------------------------------------------------------
// fputbyte(): write a byte value
// ----------------------------------------------------------------------------

void fputbyte (FILE * f, char v)
{
    fwriteOrDie (&v, sizeof (v), 1, f);
}

// ----------------------------------------------------------------------------
// fputshort(): write a short value
// ----------------------------------------------------------------------------

void fputshort (FILE * f, short v)
{
    fwriteOrDie (&v, sizeof (v), 1, f);
}

// ----------------------------------------------------------------------------
// fputint24(): write a 3-byte (24-bit) int value
// ----------------------------------------------------------------------------

void fputint24 (FILE * f, int v)
{
    ASSERT (sizeof (v) == 4);
    fwriteOrDie (&v, sizeof (v) -1, 1, f);  // write low-order 3 bytes
}

// ----------------------------------------------------------------------------
// fputint(): write an int value
// ----------------------------------------------------------------------------

void fputint (FILE * f, int v)
{
    fwriteOrDie (&v, sizeof (v), 1, f);
}

void fputint (const HANDLE f, int v)
{
    fwriteOrDie (&v, sizeof (v), 1, f);
}

// ----------------------------------------------------------------------------
// fputfloat(): write a float value
// ----------------------------------------------------------------------------

void fputfloat (FILE * f, float v)
{
    fwriteOrDie (&v, sizeof (v), 1, f);
}

// ----------------------------------------------------------------------------
// fputdouble(): write a double value
// ----------------------------------------------------------------------------

void fputdouble (FILE * f, double v)
{
    fwriteOrDie (&v, sizeof (v), 1, f);
}

// ----------------------------------------------------------------------------
// fputfile(): write a binary block or a string as a file
// ----------------------------------------------------------------------------

void fputfile (const WSTRING & pathname, const ARRAY<char> & buffer)
{
    FILE * f = fopenOrDie (pathname, L"wb");
    try
    {
        if (buffer.size() > 0)
        {   // ^^ otherwise buffer[0] is an illegal expression
            fwriteOrDie (&buffer[0], sizeof (buffer[0]), buffer.size(), f);
        }
        fcloseOrDie (f);
    }
    catch (...)
    {
        fclose (f);
        throw;
    }
}

void fputfile (const WSTRING & pathname, const std::wstring & string)
{
    FILE * f = fopenOrDie (pathname, L"wb");
    try
    {
        if (string.length() > 0)
        {   // ^^ otherwise buffer[0] is an illegal expression
            fwriteOrDie (string.c_str(), sizeof (string[0]), string.length(), f);
        }
        fcloseOrDie (f);
    }
    catch (...)
    {
        fclose (f);
        throw;
    }
}

void fputfile (const WSTRING & pathname, const std::string & string)
{
    FILE * f = fopenOrDie (pathname, L"wb");
    try
    {
        if (string.length() > 0)
        {   // ^^ otherwise buffer[0] is an illegal expression
            fwriteOrDie (string.c_str(), sizeof (string[0]), string.length(), f);
        }
        fcloseOrDie (f);
    }
    catch (...)
    {
        fclose (f);
        throw;
    }
}

// ----------------------------------------------------------------------------
// fgetfile(): load a file as a binary block
// ----------------------------------------------------------------------------

void fgetfile (const WSTRING & pathname, ARRAY<char> & buffer)
{
    FILE * f = fopenOrDie (pathname, L"rb");
    size_t len = filesize (f);
    buffer.resize (len);
    if (buffer.size() > 0)
    {   // ^^ otherwise buffer[0] is an illegal expression
        freadOrDie (&buffer[0], sizeof (buffer[0]), buffer.size(), f);
    }
    fclose (f);
}

void fgetfile (FILE * f, ARRAY<char> & buffer)
{   // this version reads until eof
    buffer.resize (0);
    buffer.reserve (1000000);   // avoid too many reallocations
    ARRAY<char> inbuf;
    inbuf.resize (65536);         // read in chunks of this size
    while (!feof (f))           // read until eof
    {
        size_t n = fread (&inbuf[0], sizeof (inbuf[0]), inbuf.size(), f);
        if (ferror (f))
        {
            RuntimeError ("fgetfile: error reading from file: %s", strerror (errno));
        }
        buffer.insert (buffer.end(), inbuf.begin(), inbuf.begin() + n);
    }
    buffer.reserve (buffer.size());
}

// load it into RAM in one huge chunk
static size_t fgetfilechars (const std::wstring & path, vector<char> & buffer)
{
    auto_file_ptr f = fopenOrDie (path, L"rb");
    size_t len = filesize (f);
    buffer.reserve (len +1);
    freadOrDie (buffer, len, f);
    buffer.push_back (0);           // this makes it a proper C string
    return len;
}

template<class LINES> static void strtoklines (char * s, LINES & lines)
{
    char * context;
    for (char * p = strtok_s (s, "\r\n", &context); p; p = strtok_s (NULL, "\r\n", &context))
        lines.push_back (p);
}

void msra::files::fgetfilelines (const std::wstring & path, vector<char> & buffer, std::vector<std::string> & lines)
{
    // load it into RAM in one huge chunk
    const size_t len = fgetfilechars (path, buffer);

    // parse into lines
    lines.resize (0);
    lines.reserve (len / 20);
    strtoklines (&buffer[0], lines);
}

// same as above but returning const char* (avoiding the memory allocation)
vector<char*> msra::files::fgetfilelines (const wstring & path, vector<char> & buffer)
{
    // load it into RAM in one huge chunk
    const size_t len = fgetfilechars (path, buffer);

    // parse into lines
    vector<char *> lines;
    lines.reserve (len / 20);
    strtoklines (&buffer[0], lines);
    return lines;
}

// ----------------------------------------------------------------------------
// getfiletime(), setfiletime(): access modification time
// ----------------------------------------------------------------------------

bool getfiletime (const wstring & path, FILETIME & time)
{   // return file modification time, false if cannot be determined
    WIN32_FIND_DATAW findFileData;
    auto_handle hFind (FindFirstFileW (path.c_str(), &findFileData), ::FindClose);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        time = findFileData.ftLastWriteTime;
        return true;
    }
    else
    {
        return false;
    }
}

void setfiletime (const wstring & path, const FILETIME & time)
{   // update the file modification time of an existing file
    auto_handle h (CreateFileW (path.c_str(), FILE_WRITE_ATTRIBUTES,
                                FILE_SHARE_READ|FILE_SHARE_WRITE, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL));
    if (h == INVALID_HANDLE_VALUE)
    {
        RuntimeError ("setfiletime: error opening file: %d", GetLastError());
    }
    BOOL rc = SetFileTime (h, NULL, NULL, &time);
    if (!rc)
    {
        RuntimeError ("setfiletime: error setting file time information: %d", GetLastError());
    }
}

// ----------------------------------------------------------------------------
// expand_wildcards -- wildcard expansion of a path, including directories.
// ----------------------------------------------------------------------------

// Win32-style variant of this function (in case we want to use it some day)
// Returns 0 in case of failure. May throw in case of bad_alloc.
static BOOL ExpandWildcards (wstring path, vector<wstring> & paths)
{
    // convert root to DOS filename convention
    for (size_t k = 0; k < path.length(); k++) if (path[k] == '/') path[k] = '\\';

    // remove terminating backslash
    size_t last = path.length() -1;
    if (last >= 0 && path[last] == '\\') path.erase (last);

    // convert root to long filename convention
    //if (path.find (L"\\\\?\\") != 0)
    //    path = L"\\\\?\\" + root;

    // split off everything after first wildcard
    size_t wpos = path.find_first_of (L"*?");
    if (wpos == 2 && path[0] == '\\' && path[1] == '\\')
        wpos = path.find_first_of (L"*?", 4);   // 4=skip "\\?\"
    if (wpos == wstring::npos)
    {   // no wildcard: just return it
        paths.push_back (path);
        return TRUE;
    }

    // split off everything afterwards if any
    wstring rest;   // remaining path after this directory
    size_t spos = path.find_first_of (L"\\", wpos +1);
    if (spos != wstring::npos)
    {
        rest = path.substr (spos +1);
        path.erase (spos);
    }

    // crawl folder
    WIN32_FIND_DATAW ffdata;
    auto_handle hFind (::FindFirstFileW (path.c_str(), &ffdata), ::FindClose);
    if (hFind == INVALID_HANDLE_VALUE) 
    {
        DWORD err = ::GetLastError();
        if (rest.empty() && err == 2) return TRUE;  // no matching file: empty
        return FALSE;                   // another error
    }
    size_t pos = path.find_last_of (L"\\");
    if (pos == wstring::npos) LogicError("unexpected missing \\ in path");
    wstring parent = path.substr (0, pos);
    do
    {
        // skip this and parent directory
        bool isDir = ((ffdata.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0);
        if (isDir && ffdata.cFileName[0] == '.') continue;

        wstring filename = parent + L"\\" + ffdata.cFileName;
        if (rest.empty())
        {
            paths.push_back (filename);
        }
        else if (isDir)     // multi-wildcards: further expand
        {
            BOOL rc = ExpandWildcards (filename + L"\\" + rest, paths);
            rc; // error here means no match, e.g. Access Denied to one subfolder
        }
    } while (::FindNextFileW(hFind, &ffdata) != 0);
    return TRUE;
}

void expand_wildcards (const wstring & path, vector<wstring> & paths)
{
    BOOL rc = ExpandWildcards (path, paths);
    if (!rc)
        RuntimeError ("error in expanding wild cards '%S': %S", path.c_str(), FormatWin32Error (::GetLastError()).c_str());
}

// ----------------------------------------------------------------------------
// make_intermediate_dirs() -- make all intermediate dirs on a path
// ----------------------------------------------------------------------------

static void mkdir (const wstring & path)
{
    int rc = _wmkdir (path.c_str());
    if (rc >= 0 || errno == EEXIST)
        return;     // no error or already existing --ok
    if (errno == EACCES)
    {
        // bug in _wmkdir(): returns access_denied if folder exists but read-only --check existence
        DWORD att = ::GetFileAttributesW (path.c_str());
        if (att != INVALID_FILE_ATTRIBUTES || (att & FILE_ATTRIBUTE_DIRECTORY) != 0)
            return; // ok
    }
    RuntimeError ("make_intermediate_dirs: error creating intermediate directory %S", path.c_str());
}

// make subdir of a file including parents
void msra::files::make_intermediate_dirs (const wstring & filepath)
{
    vector<wchar_t> buf;
    buf.resize (filepath.length() +1, 0);
    wcscpy_s (&buf[0], buf.size(), filepath.c_str());
    wstring subpath;
    int skip = 0;
    // if share (\\) then the first two levels (machine, share name) cannot be made
    if ((buf[0] == '/' && buf[1] == '/') || (buf[0] == '\\' && buf[1] == '\\'))
    {
        subpath = L"/";
        skip = 2;           // skip two levels (machine, share)
    }
    // make all constituents except the filename (to make a dir, include a trailing slash)
    for (const wchar_t * p = wcstok (&buf[0], L"/\\"); p; p = wcstok (NULL, L"/\\"))
    {
        if (subpath != L"" && subpath != L"/" && subpath != L"\\" && skip == 0)
        {
            mkdir (subpath);
        }
        else if (skip > 0) skip--;  // skip this level
        // rebuild the final path
        if (subpath != L"") subpath += L"/";
        subpath += p;
    }
}

// ----------------------------------------------------------------------------
// fuptodate() -- test whether an output file is at least as new as an input file
// ----------------------------------------------------------------------------

// test if file 'target' is not older than 'input' --used for make mode
// 'input' must exist if 'inputrequired'; otherweise if 'target' exists, it is considered up to date
// 'target' may or may not exist
bool msra::files::fuptodate (const wstring & target, const wstring & input, bool inputrequired)
{
    FILETIME targettime;
    if (!getfiletime (target, targettime)) return false;        // target missing: need to update
    FILETIME inputtime;
    if (!getfiletime (input, inputtime)) return !inputrequired; // input missing: if required, pretend to be out of date as to force caller to fail
    ULARGE_INTEGER targett, inputt;
    memcpy (&targett, &targettime, sizeof (targett));
    memcpy (&inputt,  &inputtime, sizeof (inputt));
    return !(targett.QuadPart < inputt.QuadPart);               // up to date if target not older than input
}
