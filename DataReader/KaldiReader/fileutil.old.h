// TODO: this is a dup; use the one in Include/ instead

//
// <copyright file="fileutil.old.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once
#ifndef _FILEUTIL_
#define _FILEUTIL_

#include "basetypes.h"
#include <stdio.h>
#ifdef __WINDOWS__
#include <windows.h>    // for mmreg.h and FILETIME
#include <mmreg.h>
#endif
#include <stdint.h>
using namespace std;

#define SAFE_CLOSE(f) (((f) == NULL) || (fcloseOrDie ((f)), (f) = NULL))

// ----------------------------------------------------------------------------
// fopenOrDie(): like fopen() but terminate with err msg in case of error.
// A pathname of "-" returns stdout or stdin, depending on mode, and it will
// change the binary mode if 'b' or 't' are given. If you use this, make sure
// not to fclose() such a handle.
// ----------------------------------------------------------------------------

FILE * fopenOrDie (const STRING & pathname, const char * mode);
FILE * fopenOrDie (const WSTRING & pathname, const wchar_t * mode);

#ifndef __unix__ // don't need binary/text distinction on unix
// ----------------------------------------------------------------------------
// fsetmode(): set mode to binary or text
// ----------------------------------------------------------------------------

void fsetmode (FILE * f, char type);
#endif

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

// 32-bit offsets only
long fseekOrDie (FILE * f, long offset, int mode = SEEK_SET);
#define ftellOrDie ftell
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

void fskipspace (FILE * F);

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
STRING fgetline (FILE * f);
WSTRING fgetlinew (FILE * f);
void fgetline (FILE * f, std::string & s, std::vector<char> & buf);
void fgetline (FILE * f, std::wstring & s, std::vector<char> & buf);
void fgetline (FILE * f, std::vector<char> & buf);
void fgetline (FILE * f, std::vector<wchar_t> & buf);

const char * fgetstring (FILE * f, char * buf, int size);
template<size_t n> const char * fgetstring (FILE * f, char (& buf)[n]) { return fgetstring (f, buf, n); }
const char * fgetstring (const HANDLE f, char * buf, int size);
template<size_t n> const char * fgetstring (const HANDLE f, char (& buf)[n]) { return fgetstring (f, buf, n); }
wstring fgetwstring (FILE * f);

const char * fgettoken (FILE * f, char * buf, int size);
template<size_t n> const char * fgettoken (FILE * f, char (& buf)[n]) { return fgettoken (f, buf, n); }
STRING fgettoken (FILE * f);

void fskipNewline (FILE * f);

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

STRING fgetTag (FILE * f);

// ----------------------------------------------------------------------------
// fcheckTag(): read a 4-byte tag & verify it; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcheckTag (FILE * f, const char * expectedTag);
void fcheckTag (const HANDLE f, const char * expectedTag);
void fcheckTag_ascii (FILE * f, const STRING & expectedTag);

// ----------------------------------------------------------------------------
// fcompareTag(): compare two tags; terminate if wrong tag
// ----------------------------------------------------------------------------

void fcompareTag (const STRING & readTag, const STRING & expectedTag);

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
// fgetfloat(): read a float value
// ----------------------------------------------------------------------------

float fgetfloat (FILE * f);
float fgetfloat_bigendian (FILE * f);
float fgetfloat_ascii (FILE * f);

// ----------------------------------------------------------------------------
// fgetdouble(): read a double value
// ----------------------------------------------------------------------------

double fgetdouble (FILE * f);

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
// fputfloat(): write a float value
// ----------------------------------------------------------------------------

void fputfloat (FILE * f, float val);

// ----------------------------------------------------------------------------
// fputdouble(): write a double value
// ----------------------------------------------------------------------------

void fputdouble (FILE * f, double val);

// ----------------------------------------------------------------------------
// fputfile(): write a binary block or a string as a file
// ----------------------------------------------------------------------------

void fputfile (const WSTRING & pathname, const std::vector<char> & buffer);
void fputfile (const WSTRING & pathname, const std::wstring & string);
void fputfile (const WSTRING & pathname, const std::string & string);

// ----------------------------------------------------------------------------
// fgetfile(): load a file as a binary block
// ----------------------------------------------------------------------------

void fgetfile (const WSTRING & pathname, std::vector<char> & buffer);
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

#endif    // _FILEUTIL_
