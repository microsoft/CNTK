//
// fileutil.h - file I/O with error checking
//
//     Copyright (c) Microsoft Corporation.  All rights reserved.
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/fileutil.h $
// 
// 71    1/03/13 8:53p Kaisheny
// Asynchronous SGD using data pipe.
// 
// 70    9/30/12 10:46a Fseide
// new optional parameter to fuptodate()--caller can now choose whether a
// missing input file, with target file present, will cause a failure or
// considers the target up-to-date
// 
// 69    11/09/11 10:01 Fseide
// added a new overload for fgetfilelines() that returns an array of char*
// instead of strings, to avoid mem alloc
// 
// 68    6/10/11 9:50 Fseide
// (fixed a missing 'inline')
// 
// 67    6/10/11 9:49 Fseide
// new function fgetfilelines() for reading text files
// 
// 66    6/09/11 15:18 Fseide
// added overloads to fexists() that accept STL strings
// 
// 65    3/07/11 12:13 Fseide
// actually implemented unlinkOrDie() (was a dummy)
// 
// 64    11/17/10 15:00 Fseide
// new function fuptodate();
// make_intermediate_dirs() moved to namespace msra::files (all new
// functions should be put in there)
// 
// 63    11/15/10 7:04p Fseide
// added an overload for freadOrDie (vector) that takes size as a size_t
// instead of an int, to pleasr the x64 compiler
// 
// 62    11/08/10 17:07 Fseide
// new function make_intermediate_dirs()
// 
// 61    11/08/10 11:43 Fseide
// (minor cleanup)
// 
// 60    2/05/09 19:06 Fseide
// fgetline() now returns a non-const pointer, because user may want to
// post-process the line, and the returned value is a user-specified
// buffer anyway
// 
// 59    1/16/09 17:34 Fseide
// relpath() and splitpath() moved to fileutil.h
// 
// 58    1/16/09 8:59 Fseide
// exported fskipspace()
// 
// 57    1/15/09 7:38 Fseide
// some magic to unify fgetstring() for char and wchar_t to a single
// template function
// 
// 56    1/15/09 7:26 Fseide
// corrected the #include order of basetypes.h
// 
// 55    1/14/09 19:26 Fseide
// new functions fsetpos() and fgetpos();
// new fixed-buffer size overload for fgetstring() and fgettoken()
// 
// 54    1/08/09 16:14 Fseide
// fopenOrDie() now supports "-" as the pathname, referring to stdin or
// stdout
// 
// 53    1/08/09 15:32 Fseide
// new funtion expand_wildcards()
// 
// 52    1/05/09 8:44 Fseide
// (added comments)
// 
// 51    11/11/08 6:04p Qiluo
// recover the old fputstring functions
// 
// 50    10/31/08 5:09p Qiluo
// remove banned APIs
// 
// 49    7/17/08 7:22p V-spwang
// undid changes - back to version 47
// 
// 47    6/24/08 19:03 Fseide
// added fgetwstring() and fputstring() for wstrings;
// added templates for freadOrDie() and fwriteOrDie() for STL vectors
// 
// 46    6/18/08 11:41 Fseide
// added #pragma once
// 
// 45    08-05-29 18:18 Llu
// fix the interface of fputwav
// 
// 44    08-05-29 13:54 Llu
// add fputwav revise fgetwav using stl instead of short *
// 
// 43    11/27/06 11:40 Fseide
// new methods fgetwfx() and fputwfx() for direct access to simple PCM WAV
// files
// 
// 42    10/14/06 18:31 Fseide
// added char* version of fexists()
// 
// 41    5/22/06 9:34 Fseide
// (experimental auto_file class checked in)
// 
// 40    5/14/06 19:59 Fseide
// new function fsetmode()
// 
// 39    3/29/06 15:36 Fseide
// changed to reading entire file instead of line-by-line, not changing
// newlines anymore
// 
// 38    2/21/06 12:39p Kit
// Added filesize64 function
// 
// 37    1/09/06 7:12p Rogeryu
// wide version of fgetline
// 
// 36    12/19/05 21:52 Fseide
// fputfile() added in 8-bit string version
// 
// 35    12/15/05 20:25 Fseide
// added getfiletime(), setfiletime(), and fputfile() for strings
// 
// 34    9/27/05 12:22 Fseide
// added wstring version of renameOrDie()
// 
// 33    9/22/05 12:26 Fseide
// new method fexists()
// 
// 32    9/15/05 11:33 Fseide
// new version of fgetline() that avoids buffer allocations, since this
// seems very expensive esp. when reading a file line by line with
// fgetline()
// 
// 31    9/05/05 4:57p F-xyzhao
// added #include <windows.h> for #include <mmreg.h> -- ugh
// 
// 30    9/05/05 11:00 Fseide
// new method renameOrDie()
// 
// 29    8/24/05 5:45p Kjchen
// merge changes in OneNote
// 
// 28    8/19/05 17:56 Fseide
// extended WAVEHEADER with write() and update()
// 
// 27    8/13/05 15:37 Fseide
// added new version of fgetline that takes a buffer
// 
// 26    7/26/05 18:54 Fseide
// new functions fgetint24() and fputint24()
// 
// 25    2/12/05 15:21 Fseide
// fgetdouble() and fputdouble() added
// 
// 24    2/05/05 12:38 Fseide
// new methods fputfile(), fgetfile();
// new overload for filesize()
// 
// 23    2/03/05 22:34 Fseide
// added new version of fgetline() that returns an STL string
// 
// 22    5/31/04 10:06 Fseide
// new methods fseekOrDie(), ftellOrDie(), unlinkOrDie(), renameOrDie()
// 
// 21    3/19/04 4:01p Fseide
// fwriteOrDie(): first argument changed to const
// 
// 20    2/27/04 10:04a V-xlshi
// 
// 19    2/19/04 3:45p V-xlshi
// fgetraw function is added.
// 
// 18    2/19/04 1:49p V-xlshi
// 
// 17    2/03/04 8:17p V-xlshi
// 
// 16    2/03/04 6:20p V-xlshi
// WAVEHEADER.prepare() added
// 
// 15    2/03/04 5:58p V-xlshi
// WAVEHEADER structure added
// 
// 14    8/15/03 15:40 Fseide
// new method filesize()
// 
// 13    8/13/03 21:06 Fseide
// new function fputbyte()
// 
// 12    8/13/03 15:37 Fseide
// prototype of fOpenOrDie() Unicode version changed
// 
// 11    8/07/03 22:04 Fseide
// fprintfOrDie() now really dies in case of error
// 
// 10    03-07-30 12:06 I-rogery
// enable both unicode and non-unicode version
// 
// 9     7/25/03 6:07p Fseide
// new functions fgetbyte() and fgetwav()
// 
// 8     7/03/02 9:25p Fseide
// fcompareTag() now uses string type for both of its arguments (before,
// it used const char * for one of them)
// 
// 7     6/10/02 3:14p Fseide
// new functions fgettoken(), fgetfloat_ascii(), fskipNewline()
// 
// 6     6/07/02 7:26p Fseide
// new functions fcheckTag_ascii() and fgetint_ascii()
// 
// 5     4/15/02 1:12p Fseide
// void fputstring (FILE * f, const TSTRING & str) and fpad() added
// 
// 4     4/03/02 3:58p Fseide
// VSS keyword and copyright added
//
// F. Seide 5 Mar 2002
//

#pragma once
#ifndef _FILEUTIL_
#define _FILEUTIL_

#include <stdio.h>
#include <windows.h>    // for mmreg.h and FILETIME
#include <mmreg.h>
#include <algorithm>    // for std::find
#include <vector>
#include <map>
#include <functional>
#include <cctype>
#include <errno.h>
#include <stdint.h>
#include <assert.h>

using namespace std;

#define SAFE_CLOSE(f) (((f) == NULL) || (fcloseOrDie ((f)), (f) = NULL))

// ----------------------------------------------------------------------------
// fopenOrDie(): like fopen() but terminate with err msg in case of error.
// A pathname of "-" returns stdout or stdin, depending on mode, and it will
// change the binary mode if 'b' or 't' are given. If you use this, make sure
// not to fclose() such a handle.
// ----------------------------------------------------------------------------

FILE * fopenOrDie (const string & pathname, const char * mode);
FILE * fopenOrDie (const wstring & pathname, const wchar_t * mode);

// ----------------------------------------------------------------------------
// fsetmode(): set mode to binary or text
// ----------------------------------------------------------------------------

void fsetmode (FILE * f, char type);

// ----------------------------------------------------------------------------
// freadOrDie(): like fread() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void freadOrDie (void * ptr, size_t size, size_t count, FILE * f);
void freadOrDie (void * ptr, size_t size, size_t count, const HANDLE f);

template<class _T>
void freadOrDie (_T & data, int num, FILE * f)    // template for vector<>
{ data.resize (num); if (data.size() > 0) freadOrDie (&data[0], sizeof (data[0]), data.size(), f); }
template<class _T>
void freadOrDie (_T & data, size_t num, FILE * f)    // template for vector<>
{ data.resize (num); if (data.size() > 0) freadOrDie (&data[0], sizeof (data[0]), data.size(), f); }

template<class _T>
void freadOrDie (_T & data, int num, const HANDLE f)    // template for vector<>
{ data.resize (num); if (data.size() > 0) freadOrDie (&data[0], sizeof (data[0]), data.size(), f); }
template<class _T>
void freadOrDie (_T & data, size_t num, const HANDLE f)    // template for vector<>
{ data.resize (num); if (data.size() > 0) freadOrDie (&data[0], sizeof (data[0]), data.size(), f); }


// ----------------------------------------------------------------------------
// fwriteOrDie(): like fwrite() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void fwriteOrDie (const void * ptr, size_t size, size_t count, FILE * f);
void fwriteOrDie (const void * ptr, size_t size, size_t count, const HANDLE f);

template<class _T>
void fwriteOrDie (const _T & data, FILE * f)    // template for vector<>
{ if (data.size() > 0) fwriteOrDie (&data[0], sizeof (data[0]), data.size(), f); }

template<class _T>
void fwriteOrDie (const _T & data, const HANDLE f)    // template for vector<>
{ if (data.size() > 0) fwriteOrDie (&data[0], sizeof (data[0]), data.size(), f); }


// ----------------------------------------------------------------------------
// fprintfOrDie(): like fprintf() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void fprintfOrDie (FILE * f, const char *format, ...);

// ----------------------------------------------------------------------------
// fcloseOrDie(): like fclose() but terminate with err msg in case of error
// not yet implemented, but we should
// ----------------------------------------------------------------------------

#define fcloseOrDie fclose

// ----------------------------------------------------------------------------
// fflushOrDie(): like fflush() but terminate with err msg in case of error
// ----------------------------------------------------------------------------

void fflushOrDie (FILE * f);

// ----------------------------------------------------------------------------
// filesize(): determine size of the file in bytes
// ----------------------------------------------------------------------------

size_t filesize (const wchar_t * pathname);
size_t filesize (FILE * f);
int64_t filesize64 (const wchar_t * pathname);

// ----------------------------------------------------------------------------
// fseekOrDie(),ftellOrDie(), fget/setpos(): seek functions with error handling
// ----------------------------------------------------------------------------

size_t fseekOrDie (FILE * f, size_t offset, int mode = SEEK_SET);
#define ftellOrDie _ftelli64
uint64_t fgetpos (FILE * f);
void fsetpos (FILE * f, uint64_t pos);

// ----------------------------------------------------------------------------
// unlinkOrDie(): unlink() with error handling
// ----------------------------------------------------------------------------

void unlinkOrDie (const std::string & pathname);
void unlinkOrDie (const std::wstring & pathname);

// ----------------------------------------------------------------------------
// renameOrDie(): rename() with error handling
// ----------------------------------------------------------------------------

void renameOrDie (const std::string & from, const std::string & to);
void renameOrDie (const std::wstring & from, const std::wstring & to);

// ----------------------------------------------------------------------------
// fexists(): test if a file exists
// ----------------------------------------------------------------------------

bool fexists (const char * pathname);
bool fexists (const wchar_t * pathname);
inline bool fexists (const std::string & pathname) { return fexists (pathname.c_str()); }
inline bool fexists (const std::wstring & pathname) { return fexists (pathname.c_str()); }

// ----------------------------------------------------------------------------
// funicode(): test if a file uses unicode
// ----------------------------------------------------------------------------

bool funicode (FILE * f);

// ----------------------------------------------------------------------------
// fskipspace(): skip space characters
// ----------------------------------------------------------------------------

bool fskipspace (FILE * F);
bool fskipwspace (FILE * F);

// ----------------------------------------------------------------------------
// fgetline(): like fgets() but terminate with err msg in case of error;
//  removes the newline character at the end (like gets()), returned buffer is
//  always 0-terminated; has second version that returns an STL string instead
// fgetstring(): read a 0-terminated string (terminate if error)
// fgetword(): read a space-terminated token (terminate if error)
// fskipNewLine(): skip all white space until end of line incl. the newline
// ----------------------------------------------------------------------------

template<class CHAR> CHAR * fgetline (FILE * f, CHAR * buf, int size);
template<class CHAR, size_t n> CHAR * fgetline (FILE * f, CHAR (& buf)[n]) { return fgetline (f, buf, n); }
string fgetline (FILE * f);
wstring fgetlinew (FILE * f);
void fgetline (FILE * f, std::string & s, std::vector<char> & buf);
void fgetline (FILE * f, std::wstring & s, std::vector<char> & buf);
void fgetline (FILE * f, std::vector<char> & buf);
void fgetline (FILE * f, std::vector<wchar_t> & buf);

const char * fgetstring (FILE * f, char * buf, int size);
template<size_t n> const char * fgetstring (FILE * f, char (& buf)[n]) { return fgetstring (f, buf, n); }
const char * fgetstring (const HANDLE f, char * buf, int size);
template<size_t n> const char * fgetstring (const HANDLE f, char (& buf)[n]) { return fgetstring (f, buf, n); }
const wchar_t * fgetstring (FILE * f, wchar_t * buf, int size);
wstring fgetwstring (FILE * f);
string fgetstring (FILE * f);

const char * fgettoken (FILE * f, char * buf, int size);
template<size_t n> const char * fgettoken (FILE * f, char (& buf)[n]) { return fgettoken (f, buf, n); }
string fgettoken (FILE * f);
const wchar_t * fgettoken (FILE * f, wchar_t * buf, int size);
wstring fgetwtoken (FILE * f);

int fskipNewline (FILE * f, bool skip = true);
int fskipwNewline (FILE * f, bool skip = true);

// ----------------------------------------------------------------------------
// fputstring(): write a 0-terminated string (terminate if error)
// ----------------------------------------------------------------------------

void fputstring (FILE * f, const char *);
void fputstring (const HANDLE f, const char * str);
void fputstring (FILE * f, const std::string &);
void fputstring (FILE * f, const wchar_t *);
void fputstring (FILE * f, const std::wstring &);

// ----------------------------------------------------------------------------
// fgetTag(): read a 4-byte tag & return as a string
// ----------------------------------------------------------------------------

string fgetTag (FILE * f);

// ----------------------------------------------------------------------------
// fcheckTag(): read a 4-byte tag & verify it; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcheckTag (FILE * f, const char * expectedTag);
void fcheckTag (const HANDLE f, const char * expectedTag);
void fcheckTag_ascii (FILE * f, const string & expectedTag);

// ----------------------------------------------------------------------------
// fcompareTag(): compare two tags; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcompareTag (const string & readTag, const string & expectedTag);

// ----------------------------------------------------------------------------
// fputTag(): write a 4-byte tag
// ----------------------------------------------------------------------------

void fputTag (FILE * f, const char * tag);
void fputTag(const HANDLE f, const char * tag);

// ----------------------------------------------------------------------------
// fskipstring(): skip a 0-terminated string, such as a pad string
// ----------------------------------------------------------------------------

void fskipstring (FILE * f);

// ----------------------------------------------------------------------------
// fpad(): write a 0-terminated string to pad file to a n-byte boundary
// ----------------------------------------------------------------------------

void fpad (FILE * f, int n);

// ----------------------------------------------------------------------------
// fgetbyte(): read a byte value
// ----------------------------------------------------------------------------

char fgetbyte (FILE * f);

// ----------------------------------------------------------------------------
// fgetshort(): read a short value
// ----------------------------------------------------------------------------

short fgetshort (FILE * f);
short fgetshort_bigendian (FILE * f);

// ----------------------------------------------------------------------------
// fgetint24(): read a 3-byte (24-bit) int value
// ----------------------------------------------------------------------------

int fgetint24 (FILE * f);

// ----------------------------------------------------------------------------
// fgetint(): read an int value
// ----------------------------------------------------------------------------

int fgetint (FILE * f);
int fgetint (const HANDLE f);
int fgetint_bigendian (FILE * f);
int fgetint_ascii (FILE * f);

// ----------------------------------------------------------------------------
// fgetlong(): read an long value
// ----------------------------------------------------------------------------
long fgetlong (FILE * f);
long fgetlong (const HANDLE f);

// ----------------------------------------------------------------------------
// fgetfloat(): read a float value
// ----------------------------------------------------------------------------

float fgetfloat (FILE * f);
float fgetfloat_bigendian (FILE * f);
float fgetfloat_ascii (FILE * f);

// ----------------------------------------------------------------------------
// fgetdouble(): read a double value
// ----------------------------------------------------------------------------

double fgetdouble (FILE * f);

#if 0   // TODO: remove these in CNTK
// ----------------------------------------------------------------------------
// fgetwav(): read an entire .wav file
// ----------------------------------------------------------------------------

void fgetwav (FILE * f, std::vector<short> & wav, int & sampleRate);
void fgetwav (const wstring & fn, std::vector<short> & wav, int & sampleRate);

// ----------------------------------------------------------------------------
// fputwav(): save data into a .wav file
// ----------------------------------------------------------------------------

void fputwav (FILE * f, const vector<short> & wav, int sampleRate, int nChannels = 1); 
void fputwav (const wstring & fn, const vector<short> & wav, int sampleRate, int nChannels = 1); 
#endif

// ----------------------------------------------------------------------------
// fputbyte(): write a byte value
// ----------------------------------------------------------------------------

void fputbyte (FILE * f, char val);

// ----------------------------------------------------------------------------
// fputshort(): write a short value
// ----------------------------------------------------------------------------

void fputshort (FILE * f, short val);

// ----------------------------------------------------------------------------
// fputint24(): write a 3-byte (24-bit) int value
// ----------------------------------------------------------------------------

void fputint24 (FILE * f, int v);

// ----------------------------------------------------------------------------
// fputint(): write an int value
// ----------------------------------------------------------------------------

void fputint (FILE * f, int val);
void fputint (const HANDLE f, int v);

// ----------------------------------------------------------------------------
// fputlong(): write an long value
// ----------------------------------------------------------------------------

void fputlong (FILE * f, long val);
void fputlong (const HANDLE f, long v);

// ----------------------------------------------------------------------------
// fputfloat(): write a float value
// ----------------------------------------------------------------------------

void fputfloat (FILE * f, float val);

// ----------------------------------------------------------------------------
// fputdouble(): write a double value
// ----------------------------------------------------------------------------

void fputdouble (FILE * f, double val);


// template versions of put/get functions for binary files
template <typename T>
void fput(FILE * f, T v)
{
    fwriteOrDie (&v, sizeof (v), 1, f);
}

template <typename T>
void fput(const HANDLE f, T v)
{
    fwriteOrDie (&v, sizeof (v), 1, f);
}


// template versions of put/get functions for binary files
template <typename T>
void fget(FILE * f, T& v)
{
    freadOrDie ((void *)&v, sizeof (v), 1, f);
}

template <typename T>
long fget(const HANDLE f)
{
    T v;
    freadOrDie (&v, sizeof (v), 1, f);
    return v;
}


// GetFormatString - get the format string for a particular type
template <typename T>
const wchar_t* GetFormatString(T /*t*/)
{
    // if this _ASSERT goes off it means that you are using a type that doesn't have
    // a read and/or write routine. 
    // If the type is a user defined class, you need to create some global functions that handles file in/out.
    // for example: 
    //File& operator>>(File& stream, MyClass& test);
    //File& operator<<(File& stream, MyClass& test);
    //
    // in your class you will probably want to add these functions as friends so you can access any private members
    // friend File& operator>>(File& stream, MyClass& test);
    // friend File& operator<<(File& stream, MyClass& test);
    //
    // if you are using wchar_t* or char* types, these use other methods because they require buffers to be passed
    // either use std::string and std::wstring, or use the WriteString() and ReadString() methods
    assert(false);  // need a specialization
    return NULL;
}

// GetFormatString - specalizations to get the format string for a particular type
template <>             const wchar_t* GetFormatString(char);
template <>          const wchar_t* GetFormatString(wchar_t);
template <>            const wchar_t* GetFormatString(short);
template <>              const wchar_t* GetFormatString(int);
template <>             const wchar_t* GetFormatString(long);
template <>   const wchar_t* GetFormatString(unsigned short);
template <>     const wchar_t* GetFormatString(unsigned int);
template <>    const wchar_t* GetFormatString(unsigned long);
template <>            const wchar_t* GetFormatString(float);
template <>           const wchar_t* GetFormatString(double);
template <>           const wchar_t* GetFormatString(size_t);
template <>        const wchar_t* GetFormatString(long long);

// GetFormatString - get the format string for a particular type
template <typename T>
const wchar_t* GetScanFormatString(T t)
{
    // if this _ASSERT goes off it means that you are using a type that doesn't have
    // a read and/or write routine. 
    // If the type is a user defined class, you need to create some global functions that handles file in/out.
    // for example: 
    //File& operator>>(File& stream, MyClass& test);
    //File& operator<<(File& stream, MyClass& test);
    //
    // in your class you will probably want to add these functions as friends so you can access any private members
    // friend File& operator>>(File& stream, MyClass& test);
    // friend File& operator<<(File& stream, MyClass& test);
    //
    // if you are using wchar_t* or char* types, these use other methods because they require buffers to be passed
    // either use std::string and std::wstring, or use the WriteString() and ReadString() methods
    assert(false);  // need a specialization
    return NULL;
}

// GetFormatString - specalizations to get the format string for a particular type
template <>             const wchar_t* GetScanFormatString(char);
template <>          const wchar_t* GetScanFormatString(wchar_t);
template <>            const wchar_t* GetScanFormatString(short);
template <>              const wchar_t* GetScanFormatString(int);
template <>             const wchar_t* GetScanFormatString(long);
template <>   const wchar_t* GetScanFormatString(unsigned short);
template <>     const wchar_t* GetScanFormatString(unsigned int);
template <>    const wchar_t* GetScanFormatString(unsigned long);
template <>            const wchar_t* GetScanFormatString(float);
template <>           const wchar_t* GetScanFormatString(double);
template <>           const wchar_t* GetScanFormatString(size_t);
template <>        const wchar_t* GetScanFormatString(long long);


// ----------------------------------------------------------------------------
// fgetText(): get a value from a text file
// ----------------------------------------------------------------------------
template <typename T>
void fgetText(FILE * f, T& v)
{
    int rc = ftrygetText(f, v);
    if (rc == 0)
        throw std::runtime_error("error reading value from file (invalid format)");
    else if (rc == EOF)
        throw std::runtime_error(std::string("error reading from file: ") + strerror(errno));
    assert(rc == 1);
}

// version to try and get a string, and not throw exceptions if contents don't match
template <typename T>
int ftrygetText(FILE * f, T& v)
{
    const wchar_t* formatString = GetScanFormatString<T>(v);
    int rc = fwscanf (f, formatString, &v);
    assert(rc == 1 || rc == 0);
    return rc;
}

template <> int ftrygetText<bool>(FILE * f, bool& v);
// ----------------------------------------------------------------------------
// fgetText() specializations for fwscanf_s differences: get a value from a text file
// ----------------------------------------------------------------------------
void fgetText(FILE * f, char& v);
void fgetText(FILE * f, wchar_t& v);


// ----------------------------------------------------------------------------
// fputText(): write a value out as text
// ----------------------------------------------------------------------------
template <typename T>
void fputText(FILE * f, T v)
{
    const wchar_t* formatString = GetFormatString(v);
    int rc = fwprintf(f, formatString, v);
    if (rc == 0)
        throw std::runtime_error("error writing value to file, no values written");
    else if (rc < 0)
        throw std::runtime_error(std::string("error writing to file: ") + strerror(errno));
}

// ----------------------------------------------------------------------------
// fputText(): write a bool out as character
// ----------------------------------------------------------------------------
template <> void fputText<bool>(FILE * f, bool v);

// ----------------------------------------------------------------------------
// fputfile(): write a binary block or a string as a file
// ----------------------------------------------------------------------------

void fputfile (const wstring & pathname, const std::vector<char> & buffer);
void fputfile (const wstring & pathname, const std::wstring & string);
void fputfile (const wstring & pathname, const std::string & string);

// ----------------------------------------------------------------------------
// fgetfile(): load a file as a binary block
// ----------------------------------------------------------------------------

void fgetfile (const wstring & pathname, std::vector<char> & buffer);
void fgetfile (FILE * f, std::vector<char> & buffer);
namespace msra { namespace files {
    void fgetfilelines (const std::wstring & pathname, vector<char> & readbuffer, std::vector<std::string> & lines);
    static inline std::vector<std::string> fgetfilelines (const std::wstring & pathname) { vector<char> buffer; std::vector<std::string> lines; fgetfilelines (pathname, buffer, lines); return lines; }
    vector<char*> fgetfilelines (const wstring & pathname, vector<char> & readbuffer);
};};

// ----------------------------------------------------------------------------
// getfiletime(), setfiletime(): access modification time
// ----------------------------------------------------------------------------

bool getfiletime (const std::wstring & path, FILETIME & time);
void setfiletime (const std::wstring & path, const FILETIME & time);

// ----------------------------------------------------------------------------
// expand_wildcards() -- expand a path with wildcards (also intermediate ones)
// ----------------------------------------------------------------------------

void expand_wildcards (const wstring & path, vector<wstring> & paths);

// ----------------------------------------------------------------------------
// make_intermediate_dirs() -- make all intermediate dirs on a path
// ----------------------------------------------------------------------------

namespace msra { namespace files {
    void make_intermediate_dirs (const wstring & filepath);
};};

// ----------------------------------------------------------------------------
// fuptodate() -- test whether an output file is at least as new as an input file
// ----------------------------------------------------------------------------

namespace msra { namespace files {
    bool fuptodate (const wstring & target, const wstring & input, bool inputrequired = true);
};};

// ----------------------------------------------------------------------------
// simple support for WAV file I/O
// ----------------------------------------------------------------------------

typedef struct wavehder{
    char          riffchar[4];
    unsigned int  RiffLength;
    char          wavechar[8];
    unsigned int  FmtLength; 
    signed short  wFormatTag; 
    signed short  nChannels;    
    unsigned int  nSamplesPerSec; 
    unsigned int  nAvgBytesPerSec; 
    signed short  nBlockAlign; 
    signed short  wBitsPerSample;
    char          datachar[4];
    unsigned int  DataLength;
private:
    void prepareRest (int SampleCount);
public:
    void prepare (unsigned int Fs, int Bits, int Channels, int SampleCount);
    void prepare (const WAVEFORMATEX & wfx, int SampleCount);
    unsigned int read (FILE * f, signed short & wRealFormatTag, int & bytesPerSample);
    void write (FILE * f);
    static void update (FILE * f);
} WAVEHEADER;

// ----------------------------------------------------------------------------
// fgetwfx(), fputwfx(): I/O of wave file headers only
// ----------------------------------------------------------------------------
unsigned int fgetwfx (FILE *f, WAVEFORMATEX & wfx);
void fputwfx (FILE *f, const WAVEFORMATEX & wfx, unsigned int numSamples);

// ----------------------------------------------------------------------------
// fgetraw(): read data of .wav file, and separate data of multiple channels. 
//            For example, data[i][j]: i is channel index, 0 means the first 
//            channel. j is sample index.
// ----------------------------------------------------------------------------
void fgetraw (FILE *f,std::vector< std::vector<short> > & data,const WAVEHEADER & wavhd);

// ----------------------------------------------------------------------------
// temp functions -- clean these up
// ----------------------------------------------------------------------------

// split a pathname into directory and filename
static inline void splitpath (const wstring & path, wstring & dir, wstring & file)
{
    size_t pos = path.find_last_of (L"\\:/");    // DOS drives, UNIX, Windows
    if (pos == path.npos)   // no directory found
    {
        dir.clear();
        file = path;
    }
    else
    {
        dir = path.substr (0, pos);
        file = path.substr (pos +1);
    }
}

// test if a pathname is a relative path
// A relative path is one that can be appended to a directory.
// Drive-relative paths, such as D:file, are considered non-relative.
static inline bool relpath (const wchar_t * path)
{   // this is a wild collection of pathname conventions in Windows
    if (path[0] == '/' || path[0] == '\\')  // e.g. \WINDOWS
        return false;
    if (path[0] && path[1] == ':')          // drive syntax
        return false;
    // ... TODO: handle long NT paths
    return true;                            // all others
}
template<class CHAR>
static inline bool relpath (const std::basic_string<CHAR> & s) { return relpath (s.c_str()); }

// trim from start
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}

vector<string> sep_string(const string & str, const string & sep);

#endif	// _FILEUTIL_
