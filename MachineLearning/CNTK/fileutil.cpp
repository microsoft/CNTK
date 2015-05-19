//
// fileutil.cpp - file I/O with error checking
//
//     Copyright (c) Microsoft Corporation.  All rights reserved.
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/fileutil.cpp $
// 
// 125   1/03/13 8:53p Kaisheny
// Asynchronous SGD using data pipe.
// 
// 124   9/30/12 10:46a Fseide
// new optional parameter to fuptodate()--caller can now choose whether a
// missing input file, with target file present, will cause a failure or
// considers the target up-to-date
// 
// 123   8/20/12 12:29p V-hansu
// fixed a major bug in freadOrDie() for chunks > 15M units (breaking into
// chunks was broken)
// 
// 122   4/01/12 12:02p Fseide
// (expanded an error message)
// 
// 121   11/09/11 10:01 Fseide
// added a new overload for fgetfilelines() that returns an array of char*
// instead of strings, to avoid mem alloc
// 
// 120   10/27/11 18:52 Fseide
// updated freadOrDie() to smaller chunk size
// 
// 119   10/27/11 13:40 Fseide
// freadOrDie() now explicitly breaks up large reads because CRT fread()
// does not handle them (due to a Windows bug)
// 
// 118   6/10/11 9:49 Fseide
// new function fgetfilelines() for reading text files
// 
// 117   3/07/11 12:13 Fseide
// actually implemented unlinkOrDie() (was a dummy)
// 
// 116   12/07/10 10:03 Fseide
// (corrected the buffer size in fsetpos() fro 65336 to 65536)
// 
// 115   12/03/10 10:53 Fseide
// fsetpos() optimization when seeking forward within the current read
// buffer
// 
// 114   11/18/10 4:32p Kit
// added missing header for errno
// 
// 113   11/18/10 9:20 Fseide
// a basic optimization in fsetpos() to avoid rereading the buffer if
// fsetpos() does not actually move the file pointer
// 
// 112   11/17/10 15:00 Fseide
// new function fuptodate();
// make_intermediate_dirs() moved to namespace msra::files (all new
// functions should be put in there)
// 
// 111   11/12/10 16:43 Fseide
// bug in getfiletime(), totally broken
// 
// 110   11/09/10 8:56 Fseide
// some cleanup of make_intermediate_dirs()
// 
// 109   11/08/10 17:07 Fseide
// new function make_intermediate_dirs()
// 
// 108   11/30/09 1:32p Kit
// 
// 107   2/05/09 19:05 Fseide
// fgetline() now returns a non-const pointer, because user may want to
// post-process the line, and the returned value is a user-specified
// buffer anyway
// 
// 106   1/16/09 8:59 Fseide
// exported fskipspace()
// 
// 105   1/16/09 8:47 Fseide
// (a comment added)
// 
// 104   1/15/09 7:38 Fseide
// some magic to unify fgetstring() for char and wchar_t to a single
// template function
// 
// 103   1/14/09 19:27 Fseide
// new functions fsetpos() and fgetpos();
// added missing read-error checks to fget(w)string()
// 
// 102   1/14/09 12:38 Fseide
// bug fix in fgetline(): missed an error check
// 
// 101   1/09/09 7:40 Fseide
// (fixed a warning)
// 
// 100   1/08/09 16:38 Fseide
// fopenOrDie() now supports "-" as the pathname, referring to stdin or
// stdout
// 
// 99    1/08/09 15:32 Fseide
// new funtion expand_wildcards()
// 
// 98    1/05/09 8:44 Fseide
// (added comments)
// 
// 97    12/24/08 14:44 Fseide
// added an overflow check to fputwfx()
// 
// 96    12/12/08 10:11a Qiluo
// (change marker of banned APIs)
// 
// 95    12/11/08 7:40p Qiluo
// (change marker of banned APIs)
// 
// 94    12/09/08 6:59p Qiluo
// reverted stringerror => strerror
// 
// 93    12/09/08 6:37p Qiluo
// fixed a few compilation bugs
// 
// 92    12/09/08 6:28p Qiluo
// strerror => stringerror
// 
// 91    12/01/08 2:43p Qiluo
// add markers for banned APIs, and refine the api fixing
// 
// 90    11/11/08 7:34p Qiluo
// fix bug in strnlen
// 
// 89    11/11/08 18:27 Fseide
// no longer disables C4996
// 
// 88    11/11/08 6:04p Qiluo
// recover the old fputstring functions
// 
// 87    11/10/08 2:34p Qiluo
// remove the dependency of header "StringUtil.h"
// 
// 86    10/31/08 5:08p Qiluo
// remove banned APIs
// 
// 85    6/24/08 19:03 Fseide
// added fgetwstring() and fputstring() for wstrings
// 
// 84    6/02/08 14:11 Fseide
// fgetwfx() and wputwfx() now a bit more tolerant
// 
// 83    08-05-29 18:18 Llu
// fix the interface of fputwav
// 
// 82    08-05-29 14:53 Llu
// 
// 81    08-05-29 13:53 Llu
// add fputwav revise fgetwav using stl instead of short *
// 
// 80    3/19/08 16:13 Fseide
// (better solution to prev. problem)
// 
// 79    3/19/08 16:07 Fseide
// (#ifdef'ed out fprintfOrDie() in _MANAGED builds)
// 
// 78    10/30/07 16:46 Fseide
// 
// 77    3/27/07 13:54 Fseide
// added 'using namespace std;' (was removed from message.h as it does not
// belong there)
// 
// 76    1/30/07 1:59p Kit
// Undid updates to fgetline error handling
// 
// 70    12/20/06 10:48a Kit
// increased size of line buffer for fgetline because we seem to be
// getting large strings in some rss feeds
// 
// 69    06-12-04 18:30 Llu
// (fixed an unnecessary "deprecated string function" warning under VS
// 2005)
// 
// 68    11/27/06 11:40 Fseide
// new methods fgetwfx() and fputwfx() for direct access to simple PCM WAV
// files
// 
// 67    10/14/06 18:31 Fseide
// added char* version of fexists()
// 
// 66    5/14/06 19:58 Fseide
// new function fsetmode()
// 
// 65    3/29/06 16:10 Fseide
// increased buffer size in fgetfile() to 64k
// 
// 64    3/29/06 15:36 Fseide
// changed to reading entire file instead of line-by-line, not changing
// newlines anymore
// 
// 63    3/24/06 4:40p Rogeryu
// workaround a VC 2003 header bug (va_start macro for references) in
// MESSAGE/ERROR functions
// 
// 62    3/22/06 3:31p Rogeryu
// (comments changed)
// 
// 61    3/21/06 5:21p Rogeryu
// review and fix level2_security OACR warnings
// 
// 60    3/21/06 9:26a Rogeryu
// review and fix OACR warnings
// 
// 59    06-03-15 15:41 Yushli
// Suppress C4996 Warning per function
// 
// 58    06-03-14 12:11 Yushli
// Suppress C4996 Warning on strerror per function
// 
// 57    06-03-14 10:33 Yushli
// Suppress C4996 Warning per function.
// 
// 56    2/28/06 1:49p Kjchen
// suppress oacr warning
// 
// 55    2/24/06 8:03p Kjchen
// depress oacr warnings
// 
// 54    2/21/06 11:32a Kit
// aadded filesize64 to support large files
// 
// 53    1/10/06 8:23p Rogeryu
// fix a warning
// 
// 52    1/09/06 7:12p Rogeryu
// wide version of fgetline
// 
// 51    12/20/05 21:15 Fseide
// changed CreateFile() to CreateFileW()
// 
// 50    12/19/05 22:50 Fseide
// setfiletime() fixed, now actually works
// 
// 49    12/19/05 21:52 Fseide
// fputfile() added in 8-bit string version
// 
// 48    12/18/05 17:01 Fseide
// fixed file-handle leaks in error conditions
// 
// 47    12/15/05 20:25 Fseide
// added getfiletime(), setfiletime(), and fputfile() for strings
// 
// 46    9/27/05 12:22 Fseide
// added wstring version of renameOrDie()
// 
// 45    9/22/05 12:26 Fseide
// new method fexists()
// 
// 44    9/15/05 11:33 Fseide
// new version of fgetline() that avoids buffer allocations, since this
// seems very expensive esp. when reading a file line by line with
// fgetline()
// 
// 43    9/05/05 4:57p F-xyzhao
// renameOrDie(): changed string to std::string
// 
// 42    9/05/05 11:00 Fseide
// new method renameOrDie()
// 
// 41    8/19/05 18:19 Fseide
// bugfixes in WAVEHEADER::write and prepare
// 
// 40    8/19/05 18:02 Fseide
// WAVEHEADER::write() now flushes
// 
// 39    8/19/05 17:56 Fseide
// extended WAVEHEADER with write() and update()
// 
// 38    8/14/05 16:56 Fseide
// fopenOrDie() now sets large buffer if 'S' option
// 
// 37    8/13/05 15:37 Fseide
// added new version of fgetline that takes a buffer
// 
// 36    7/28/05 18:04 Fseide
// bug fix in fgetin24 and fputint24
// 
// 35    7/26/05 18:54 Fseide
// new functions fgetint24() and fputint24()
// 
// 34    5/10/05 14:12 Fseide
// (level-4 warning fixed)
// 
// 33    5/10/05 11:57 Fseide
// (level-4 warnings removed)
// 
// 32    5/09/05 12:07 Fseide
// fixed for-loop conformance issues
// 
// 31    2/27/05 17:41 Fseide
// recovered v29 that somehow got overwritten
// 
// 29    2/12/05 15:21 Fseide
// fgetdouble() and fputdouble() added
// 
// 28    2/05/05 12:38 Fseide
// new methods fputfile(), fgetfile();
// new overload for filesize()
// 
// 27    2/03/05 22:34 Fseide
// added new version of fgetline() that returns an STL string
// 
// 26    5/31/04 10:06 Fseide
// new methods fseekOrDie(), ftellOrDie(), unlinkOrDie(), renameOrDie()
// 
// 25    3/19/04 4:01p Fseide
// fwriteOrDie(): first argument changed to const
// 
// 24    2/21/04 10:26 Fseide
// (compiler warnings eliminated)
// 
// 23    2/19/04 9:46p V-xlshi
// 
// 22    2/19/04 3:44p V-xlshi
// fgetwavraw and fgetraw function is added, fgetwav is changed but its
// functionality is the same with the old one.
// 
// 21    2/03/04 8:17p V-xlshi
// 
// 20    9/08/03 22:55 Fseide
// fgetwav() can now read stereo PCM files
// 
// 19    8/15/03 15:40 Fseide
// new method filesize()
// 
// 18    8/13/03 21:06 Fseide
// new function fputbyte()
// 
// 17    8/13/03 15:37 Fseide
// an error msg corrected
// 
// 16    8/07/03 22:04 Fseide
// fprintfOrDie() now really dies in case of error
// 
// 15    7/30/03 5:09p Fseide
// (eliminated a compiler warning)
// 
// 14    03-07-30 14:17 I-rogery
// 
// 13    7/25/03 6:07p Fseide
// new functions fgetbyte() and fgetwav()
// 
// 12    6/03/03 5:23p Fseide
// (some compiler warnings related to size_t eliminated)
// 
// 11    3/27/03 3:42p Fseide
// fwriteOrDie() rewritten to break huge blocks into chunks of 16 MB
// because Windows std C lib can't handle fwrite() with e.g. 100 MB in one
// call
// 
// 10    7/23/02 9:00p Jlzhou
// 
// 9     7/03/02 9:25p Fseide
// fcompareTag() now uses STRING type for both of its arguments (before,
// it used const char * for one of them)
// 
// 8     6/10/02 3:14p Fseide
// new functions fgettoken(), fgetfloat_ascii(), fskipNewline()
// 
// 7     6/07/02 7:26p Fseide
// new functions fcheckTag_ascii() and fgetint_ascii()
// 
// 6     6/03/02 10:58a Jlzhou
// 
// 5     4/15/02 1:12p Fseide
// void fputstring (FILE * f, const TSTRING & str) and fpad() added
// 
// 4     4/03/02 3:56p Fseide
// VSS keyword and copyright added
//
// F. Seide 5 Mar 2002
//

#ifndef UNDER_CE    // fixed-buffer overloads not available for wince
#ifdef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES  // fixed-buffer overloads for strcpy() etc.
#undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#endif
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif

#include "basetypes.h"
#include "fileutil.h"
#include "message.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "windows.h"    // for FILETIME
#include <algorithm>    // for std::find

#ifndef UNDER_CE  // some headers don't exist under winCE - the appropriate definitions seem to be in stdlib.h
#include <fcntl.h>      // for _O_BINARY/TEXT - not needed for wince
#include <io.h>         // for _setmode()
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
    if (strchr (mode, 'b') || strchr (mode, 't'))   // change binary mode
    {
        // switch to binary mode if not yet (in case it is stdin)
        int rc = _setmode (_fileno (f), strchr (mode, 'b') ? _O_BINARY : _O_TEXT);
        if (rc == -1)
            ERROR ("error switching stream to binary mode: %s", strerror (errno));
    }
    return f;
}

FILE * fopenOrDie (const STRING & pathname, const char * mode)
{
    FILE * f = (pathname[0] == '-') ? fopenStdHandle (mode) : fopen (pathname.c_str(), mode);
    if (f == NULL)
    {
    ERROR ("error opening file '%s': %s", pathname.c_str(), strerror (errno));
        return NULL;    // keep OACR happy
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
        ERROR ("error opening file '%S': %s", pathname.c_str(), strerror (errno));
        return NULL;    // keep OACR happy
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

void fsetmode (FILE * f, char type)
{
    if (type != 'b' && type != 't')
    {
        ERROR ("fsetmode: invalid type '%c'");
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
    ERROR ("error changing file mode: %s", strerror (errno));
    }
}

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
            ERROR ("error reading from file: %s", strerror (errno));
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
            ERROR ("error number for reading from file: %s", GetLastError());
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
            ERROR ("error writing to file (ptr=0x%08lx, size=%d,"
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
            ERROR ("error writing to file (ptr=0x%08lx, size=%d,"
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
        ERROR ("error writing to file: %s", strerror (errno));
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
    ERROR ("error flushing to file: %s", strerror (errno));
    }
}

// ----------------------------------------------------------------------------
// filesize(): determine size of the file in bytes (with open file)
// BUGBUG: how about files > 4 GB?
// ----------------------------------------------------------------------------
size_t filesize (FILE * f)
{
    long curPos = ftell (f);
    if (curPos == -1L)
    {
    ERROR ("error determining file position: %s", strerror (errno));
    }
    int rc = fseek (f, 0, SEEK_END);
    if (rc != 0)
    {
    ERROR ("error seeking to end of file: %s", strerror (errno));
    }
    long len = ftell (f);
    if (len == -1L)
    {
    ERROR ("error determining file position: %s", strerror (errno));
    }
    rc = fseek (f, curPos, SEEK_SET);
    if (rc != 0)
    {
    ERROR ("error resetting file position: %s", strerror (errno));
    }
    return (size_t) len;
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
__int64 filesize64 (const wchar_t * pathname)
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
    ERROR ("error seeking: %s", strerror (errno));
    }
    int rc = fseek (f, offset, mode);
    if (rc != 0)
    {
    ERROR ("error seeking: %s", strerror (errno));
    }
    return curPos;
}

unsigned __int64 fgetpos (FILE * f)
{
    fpos_t post;
    int rc = ::fgetpos (f, &post);
    if (rc != 0)
        ERROR ("error getting file position: %s", strerror (errno));
    return post;
}

void fsetpos (FILE * f, unsigned __int64 reqpos)
{
    // ::fsetpos() flushes the read buffer. This conflicts with a situation where
    // we generally read linearly but skip a few bytes or KB occasionally, as is
    // the case in speech recognition tools. This requires a number of optimizations.

    unsigned __int64 curpos = fgetpos (f);
    unsigned __int64 cureob = curpos + f->_cnt; // UGH: we mess with an internal structure here
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
        ERROR ("error setting file position: %s", strerror (errno));
}

// ----------------------------------------------------------------------------
// unlinkOrDie(): unlink() with error handling
// ----------------------------------------------------------------------------

void unlinkOrDie (const std::string & pathname)
{
    if (_unlink (pathname.c_str()) != 0 && errno != ENOENT)     // if file is missing that's what we want
    ERROR ("error deleting file '%s': %s", pathname.c_str(), strerror (errno));
}
void unlinkOrDie (const std::wstring & pathname)
{
    if (_wunlink (pathname.c_str()) != 0 && errno != ENOENT)    // if file is missing that's what we want
    ERROR ("error deleting file '%S': %s", pathname.c_str(), strerror (errno));
}

// ----------------------------------------------------------------------------
// renameOrDie(): rename() with error handling
// ----------------------------------------------------------------------------

#ifndef UNDER_CE // CE only supports Unicode APIs
void renameOrDie (const std::string & from, const std::string & to)
{
    if (!MoveFileA (from.c_str(),to.c_str()))
    ERROR ("error renaming: %s", GetLastError());
}
#endif

void renameOrDie (const std::wstring & from, const std::wstring & to)
{
    if (!MoveFileW (from.c_str(),to.c_str()))
    ERROR ("error renaming: %s", GetLastError());
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

    unsigned __int64 filepos = fgetpos (f); // (for error message only)
    CHAR * p = fgets (buf, size, f);
    if (p == NULL)            // EOF reached: next time feof() = true
    {
        if (ferror (f))
            ERROR ("error reading line: %s", strerror (errno));
        buf[0] = 0;
        return buf;
    }
    size_t n = strnlen (p, size);

    // check for buffer overflow

    if (n >= (size_t) size -1)
    {
        basic_string<CHAR> example (p, n < 100 ? n : 100);
        ERROR ("input line too long at file offset %I64d (max. %d characters allowed) [%s ...]",
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
            ERROR ("error reading line: %s", strerror (errno));
        buf[0] = 0;
        return buf;
    }
    size_t n = wcsnlen (p, size); // SECURITY NOTE: string use has been reviewed

    // check for buffer overflow

    if (n >= (size_t) size -1)
    {
        wstring example (buf, min (n, 100));
        ERROR ("input line too long at file offset %U64d (max. %d characters allowed) [%S ...]",
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
            ERROR ("error reading string or missing 0: %s", strerror (errno));
    if (c == 0) break;
    if (i >= size -1)
    {
        ERROR ("input line too long (max. %d characters allowed)", size -1);
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
            ERROR ("input line too long (max. %d characters allowed)", size -1);
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
            ERROR ("error reading string or missing 0: %s", strerror (errno));
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
                ERROR ("error reading from file: %s", strerror (errno));
            break;
        }
    if (!isspace (c))    // end of space: undo getting that character
        {
            int rc = ungetc (c, f);
            if (rc != c)
                ERROR ("error in ungetc(): %s", strerror (errno));
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
    ERROR ("unexpected garbage at end of line");
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
        ERROR ("input token too long (max. %d characters allowed)", size -1);
    buf[i] = (char) c;
    }
    // ... TODO: while (isspace (c)) c = fgetc (f);      // skip trailing space
    if (c != EOF)
    {
    int rc = ungetc (c, f);
    if (rc != c)
        ERROR ("error in ungetc(): %s", strerror (errno));
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
        ERROR ("invalid tag '%s' found; expected '%s'", buf, expectedTag.c_str());
    }
}

// ----------------------------------------------------------------------------
// fcompareTag(): compare two tags; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcompareTag (const STRING & readTag, const STRING & expectedTag)
{
    if (readTag != expectedTag)
    {
        ERROR ("invalid tag '%s' found; expected '%s'", 
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
    ERROR ("error in ftell(): %s", strerror (errno));
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
    ERROR ("error in ungetc(): %s", strerror (errno));
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
    ERROR ("error reading float value from file (invalid format): %s");
    else if (rc == EOF)
    ERROR ("error reading from file: %s", strerror (errno));
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
    ERROR ("error determining file position: %s", strerror (errno));
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
        || ERROR ("WAVEHEADER::read: wFormatTag=%d not supported for now", wRealFormatTag);
    unsigned short wChannels = fgetshort (f);
    unsigned long dwSamplesPerSec = fgetint (f);
    unsigned int sampleRate = dwSamplesPerSec;
    /*unsigned long dwAvgBytesPerSec = */ fgetint (f);
    unsigned short wBlockAlign = fgetshort (f);
    unsigned short wBitsPerSample = fgetshort (f);
    (wBitsPerSample <= 16) || ERROR ("WAVEHEADER::read: invalid wBitsPerSample %d", wBitsPerSample);
    bytesPerSample = wBitsPerSample / 8;
    (wBlockAlign == wChannels * bytesPerSample)
        || ERROR ("WAVEHEADER::read: wBlockAlign != wChannels*bytesPerSample not supported");
    while (fmtLen > 16) // unused extra garbage in header
    {
        fgetbyte (f);
        fmtLen--;
    }
    if (wRealFormatTag == 7)
    {
        (bytesPerSample == 1) || ERROR ("WAVEHEADER::read: invalid wBitsPerSample %d for mulaw", wBitsPerSample);
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
        (wavhd.nChannels == 1) || ERROR ("fgetwav: wChannels=%d not supported for mulaw", wavhd.nChannels);
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
        ERROR ("bytesPerSample != 2 is not supported except mulaw format!\n");
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
        ERROR ("bytesPerSample/wChannels != 2 needs to be implemented");
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
        || ERROR ("WAVEHEADER::read: wFormatTag=%d not supported for now", wfx.wFormatTag);
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
        || ERROR ("fputwfx: data size exceeds WAV header 32-bit range");
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
            ERROR ("fgetfile: error reading from file: %s", strerror (errno));
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
        ERROR ("setfiletime: error opening file: %d", GetLastError());
    }
    BOOL rc = SetFileTime (h, NULL, NULL, &time);
    if (!rc)
    {
        ERROR ("setfiletime: error setting file time information: %d", GetLastError());
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
    if (pos == wstring::npos) throw std::logic_error ("unexpected missing \\ in path");
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
        ERROR ("error in expanding wild cards '%S': %S", path.c_str(), FormatWin32Error (::GetLastError()).c_str());
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
    ERROR ("make_intermediate_dirs: error creating intermediate directory %S", path.c_str());
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
