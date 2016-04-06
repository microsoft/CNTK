//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif
#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _

#include "Basics.h"
#define FORMAT_SPECIALIZE // to get the specialized version of the format routines
#include "File.h"
#include <string>
#include <stdint.h>
#include <locale>
#ifdef _WIN32
#define NOMINMAX
#include "Windows.h"
#include <VersionHelpers.h>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
#endif
#ifdef __unix__
#include <unistd.h>
#include <linux/limits.h> // for PATH_MAX
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

// File creation
// filename - the path
// fileOptions - options to open the file
File::File(const std::wstring& filename, int fileOptions)
{
    Init(filename.c_str(), fileOptions);
}

File::File(const std::string& filename, int fileOptions)
{
    // this converts from string to wstring, and then to wchar_t*
    Init(msra::strfun::utf16(filename).c_str(), fileOptions);
}

File::File(const wchar_t* filename, int fileOptions)
{
    Init(filename, fileOptions);
}

template<class String>
static bool IsNonFilePath(const String& filename)
{
    return
        filename.front() == '|' ||                    // "| command": output pipe
        filename.back()  == '|' ||                    // "command |": input pipe
        (filename.size() == 1 && filename[0] == '-'); // "-": stdin/stdout
}

// test if a file exists
// If the pathname is a pipe, it is considered to exist.
template<class String>
/*static*/ bool File::Exists(const String& filename)
{
    return IsNonFilePath(filename) || fexists(filename);
}

template /*static*/ bool File::Exists<string> (const string&  filename);
template /*static*/ bool File::Exists<wstring>(const wstring& filename);

template<class String>
/*static*/ void File::MakeIntermediateDirs(const String& filename)
{
    if (!IsNonFilePath(filename))
        msra::files::make_intermediate_dirs(filename);
}

//template /*static*/ void File::MakeIntermediateDirs<string> (const string&  filename); // implement this if needed
template /*static*/ void File::MakeIntermediateDirs<wstring>(const wstring& filename);

// all constructors call this
void File::Init(const wchar_t* filename, int fileOptions)
{
    m_filename = filename;
    m_options = fileOptions;
    if (m_filename.empty())
        RuntimeError("File: filename is empty");
    const auto outputPipe = (m_filename.front() == '|');
    const auto inputPipe  = (m_filename.back()  == '|');
    // translate the options string into a string for fopen()
    const auto reading = !!(fileOptions & fileOptionsRead);
    const auto writing = !!(fileOptions & fileOptionsWrite);
    if (!reading && !writing)
        RuntimeError("File: either fileOptionsRead or fileOptionsWrite must be specified");
    // convert fileOptions to fopen()'s mode string
    wstring options = reading ? L"r" : L"";
    if (writing)
    {
        // if we already are reading the file, change to read/write
        options.clear();
        options.append(L"w");
        if (!outputPipe && m_filename != L"-")
        {
            options.append(L"+");
            msra::files::make_intermediate_dirs(m_filename.c_str()); // writing to regular file -> also create the intermediate directories as a convenience
        }
    }
    if (fileOptions & fileOptionsBinary)
        options += L"b";
    else
        options += L"t";
    // add sequential flag to allocate big read buffer
    if (fileOptions & fileOptionsSequential)
        options += L"S";
    // now open the file
    // Special path syntax understood here:
    //  - "-" refers to stdin or stdout
    //  - "|cmd" writes to a pipe
    //  - "cmd|" reads from a pipe
    m_pcloseNeeded = false;
    m_seekable = false;
    if (m_filename == L"-") // stdin/stdout
    {
        if (writing && reading)
            RuntimeError("File: cannot specify fileOptionsRead and fileOptionsWrite at once with path '-'");
        m_file = writing ? stdout : stdin;
    }
    else if (outputPipe || inputPipe) // pipe syntax
    {
        if (inputPipe && outputPipe)
            RuntimeError("File: pipes cannot specify fileOptionsRead and fileOptionsWrite at once");
        if (inputPipe != reading)
            RuntimeError("File: pipes must use consistent fileOptionsRead/fileOptionsWrite");
        const auto command = inputPipe ? m_filename.substr(0, m_filename.size() - 1) : m_filename.substr(1);
        m_file = _wpopen(command.c_str(), options.c_str());
        if (!m_file)
            RuntimeError("File: error exexuting pipe command '%S': %s", command.c_str(), strerror(errno));
        m_pcloseNeeded = true;
    }
    else
        attempt([=]() // regular file: use a retry loop
                {
                    m_file = fopenOrDie(filename, options.c_str());
                    m_seekable = true;
                });
}

// determine the directory for a given pathname
// (wstring only for now; feel free to make this a template if needed)
/*static*/ wstring File::DirectoryPathOf(wstring path)
{
#if WIN32
    if (IsWindows8OrGreater())
    {
        typedef HRESULT(*PathCchRemoveFileSpecProc)(_Inout_updates_(_Inexpressible_(cchPath)) PWSTR, _In_ size_t);

        HINSTANCE hinstLib;
        PathCchRemoveFileSpecProc ProcAdd;
        BOOL fFreeResult = FALSE;

        hinstLib = LoadLibrary(TEXT("api-ms-win-core-path-l1-1-0.dll"));
        if (hinstLib != nullptr)
        {
            ProcAdd = reinterpret_cast<PathCchRemoveFileSpecProc>(GetProcAddress(hinstLib, "PathCchRemoveFileSpec"));
            if (NULL != ProcAdd)
            {
                auto hr = (ProcAdd)(&path[0], path.size());
                if (hr == S_OK) // done
                    path.resize(wcslen(&path[0]));
                else if (hr == S_FALSE) // nothing to remove: use .
                    path = L".";
            }
            else
            {
                LogicError("DirectoryPathOf: GetProcAddress() unexpectedly failed.");
            }

            fFreeResult = FreeLibrary(hinstLib);
        }
        else
        {
            LogicError("DirectoryPathOf: LoadLibrary() unexpectedly failed.");
        }
    }
    else
    {
        auto hr = PathRemoveFileSpec(&path[0]);
        if (hr != 0) // done
            path.resize(wcslen(&path[0]));
        else
            path = L".";
    }
#else
    auto pos = path.find_last_of(L"/");
    if (pos != path.npos)
        path.erase(pos);
    else // if no directory path at all, use current directory
        return L".";
#endif
    return path;
}

// determine the file name for a given pathname
// (wstring only for now; feel free to make this a template if needed)
/*static*/ wstring File::FileNameOf(wstring path)
{
#if WIN32
    static const wstring delim = L"\\:/";
#else
    static const wstring delim = L"/";
#endif
    auto pos = path.find_last_of(delim);
    if (pos != path.npos)
        return path.substr(pos + 1);
    else // no directory path
        return path;
}

// get path of current executable
/*static*/ wstring File::GetExecutablePath()
{
#if WIN32
    wchar_t path[33000];
    if (GetModuleFileNameW(NULL, path, _countof(path)) == 0)
        LogicError("GetExecutablePath: GetModuleFileNameW() unexpectedly failed.");
    return path;
#else
    // from http://stackoverflow.com/questions/4025370/can-an-executable-discover-its-own-path-linux
    pid_t pid = getpid();
    char path[PATH_MAX + 1] = { 0 };
    sprintf(path, "/proc/%d/exe", pid);
    char dest[PATH_MAX + 1] = { 0 };
    if (readlink(path, dest, PATH_MAX) == -1)
        RuntimeError("GetExecutableDirectory: readlink() call failed.");
    else
        return msra::strfun::utf16(dest);
#endif
}

// skip to given delimiter character
void File::SkipToDelimiter(int delim)
{
    int ch = 0;

    while (ch != delim)
    {
        ch = fgetc(m_file);
        if (feof(m_file))
        {
            printf("Unexpected end of file\n");
            LogicError("Unexpected end of file\n");
        }
    }
}

bool File::IsTextBased()
{
    return !!(m_options & fileOptionsText);
}

// File Destructor
// closes the file
// Note: this does not check for errors when the File corresponds to pipe stream. In this case, use Flush() before closing a file you are writing.
File::~File(void)
{
    if (m_pcloseNeeded)
    {
        // TODO: Check for error code and throw if !std::uncaught_exception()     
        _pclose(m_file);
    }
    else if (m_file != stdin && m_file != stdout && m_file != stderr)
    {
        int rc = fclose(m_file);
        if ((rc != 0) && !std::uncaught_exception())
            RuntimeError("File: failed to close file at %S", m_filename.c_str());
    }
}

void File::Flush()
{
    fflushOrDie(m_file);
}

// read a line
// End of line is denoted by one of these, i.e. we don't support the old Mac OS convention of CR
//  - LF
//  - CR+LF
//  - EOF
static bool fgetc(char& c, FILE * f) { int ci = getc(f); c = (char) ci; return ci != EOF; }

static inline bool BeginsWithUnicodeBOM(const char * s)
{
    return ((unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF);
}

// read a 8-bit string until newline is hit
template<class STRING>
static void fgets(STRING & s, FILE * f)
{
    s.resize(0);
    char c;
    while (fgetc(c, f))
    {
        if (c == '\n' || c == '\r')
        {
            if (c == '\r' && (!fgetc(c, f) || c != '\n'))
                RuntimeError("fgets: malformed text file, CR without LF");
            break;
        }
        s.push_back(c);
        // strip Unicode BOM
        // We strip it from any string, not just at the start.
        // This allows to UNIX-'cat' multiple UTF-8 files with BOMs.
        // Since the BOM is otherwise invalid within a file, this is well-defined and upwards compatible.
        if (s.size() == 3 && BeginsWithUnicodeBOM(s.c_str()))
            s.clear();
    }
}

// GetLine - get a line from the file
// str - string
void File::GetLine(string& str)
{
    fgets(str, m_file);
}

static void PushBackString(vector<string>& lines,  const string& s) { lines.push_back(s); }
static void PushBackString(vector<wstring>& lines, string& s)       { lines.push_back(msra::strfun::utf16(s)); }

// GetLines - get all lines from a file
template <typename STRING>
static void FileGetLines(File& file, /*out*/ std::vector<STRING>& lines)
{
    lines.clear();
    string line;
    while (!file.IsEOF())
    {
        file.GetLine(line);
        PushBackString(lines, line);
    }
}
void File::GetLines(std::vector<std::wstring>& lines)
{
    FileGetLines(*this, lines);
};
void File::GetLines(std::vector<std::string>& lines)
{
    FileGetLines(*this, lines);
}

// Put a zero/space terminated wstring into a file
// val - value to write to the file
File& File::operator<<(const std::wstring& val)
{
    WriteString(val.c_str());
    return *this;
}

// Put a zero/space terminated string into a file
// val - value to write to the file
File& File::operator<<(const std::string& val)
{
    WriteString(val.c_str());
    return *this;
}

// Put a marker in the file, the marker depends on the file type
// marker - marker to place in the file
File& File::operator<<(FileMarker marker)
{
    File& file = *this;
    switch (marker)
    {
    case fileMarkerBeginFile: // beginning of file marker
        // TODO: why not write a BOM?
        break;
    case fileMarkerEndFile: // end of file marker
        // use ^Z for end of file for text files
        // TODO: What??
        if (m_options & fileOptionsText)
            file << char(26);
        break;
    case fileMarkerBeginList: // Beginning of list marker
        // no marker written for either
        break;
    case fileMarkerListSeparator: // separate elements of a list
        // do nothing for now, built in space deliminter for all types (before type)
        // future: make this customizable, so you can specify a separator (i.e. ',')
        break;
    case fileMarkerEndList: // end of line/list marker
        if (m_options & fileOptionsText)
            file.WriteString("\r\n");
        break;
    case fileMarkerBeginSection: // beginning of section
    case fileMarkerEndSection:   // end of section
        assert(false);           // sections should use a string modifier
        break;
    }
    return file;
}

// PutMarker for beginning of list support (lists with a count)
// count - [in] the number of elements in the list
File& File::PutMarker(FileMarker marker, size_t count)
{
    assert(marker == fileMarkerBeginList);
    marker; // only beginning of list supported for count  markers
    *this << count;
    return *this;
}

// PutMarker for section beginning and ending tags
// section - [in]name of section
File& File::PutMarker(FileMarker marker, const std::string& section)
{
    File& file = *this;
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection);
    marker;
    file << section;
    return file;
}

// PutMarker for section beginning and ending tags
// section - [in]name of section
File& File::PutMarker(FileMarker marker, const std::wstring& section)
{
    File& file = *this;
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection);
    marker;
    file << section;
    return file;
}

// Get a zero terminated wstring from a file
// val - value to read from the file
File& File::operator>>(std::wstring& val)
{
    if (IsTextBased())
        val = fgetwtoken(m_file);
    else
        val = fgetwstring(m_file);
    return *this;
}

// Get a zero terminated string from a file
// val - value to read from the file
File& File::operator>>(std::string& val)
{
    if (IsTextBased())
        val = fgettoken(m_file);
    else
        val = fgetstring(m_file);
    return *this;
}

// ReadChars - read a specified number of characters, and reset read pointer if requested
// val - [in,out] return value will be returned here
// cnt - number of characters to read
// reset - reset the read pointer
void File::ReadChars(std::string& val, size_t cnt, bool reset)
{
    size_t pos = 0; // (initialize to keep compiler happy)
    if (reset)
        pos = GetPosition();
    val.resize(cnt);
    char* str = const_cast<char*>(val.c_str());
    for (int i = 0; i < cnt; ++i)
        *this >> str[i];
    if (reset)
        SetPosition(pos);
}

// ReadChars - read a specified number of characters, and reset read pointer if requested
// val - [in,out] return value will be returned here
// cnt - number of characters to read
// reset - reset the read pointer
void File::ReadChars(std::wstring& val, size_t cnt, bool reset)
{
    size_t pos = 0; // (initialize to keep compiler happy)
    if (reset)
        pos = GetPosition();
    val.resize(cnt);
    wchar_t* str = const_cast<wchar_t*>(val.c_str());
    for (int i = 0; i < cnt; ++i)
        *this >> str[i];
    if (reset)
        SetPosition(pos);
}

// WriteString - outputs a string into the file
// str - the string to output
// size - size of the string to output, if zero null terminated
void File::WriteString(const char* str, int size)
{
    if (size > 0)
    {
        fwprintf(m_file, L" %.*hs", size, str);
    }
    else
    {
        if (IsTextBased())
            fwprintf(m_file, L" %hs", str);
        else
            fputstring(m_file, str);
    }
}

// ReadString - reads a string into the file
// str - the string buffer to read the string into
// size - size of the string buffer incl. zero terminator (we fail if input is too long)
void File::ReadString(char* str, int size)
{
    if (IsTextBased())
    {
        fgettoken(m_file, str, size);
        if (BeginsWithUnicodeBOM(str))
            for (; str[3]; str++)
                str[0] = str[3];    // delete it from start of line
    }
    else
        fgetstring(m_file, str, size);
}

// WriteString - outputs a string into the file
//   if writing to text based file and spaces are embedded, writes quotes around string
//   BUGBUG: This should be consistent between char and wchar_t versions
// str - the string to output
// size - size of the string to output, if zero null terminated
void File::WriteString(const wchar_t* str, int size)
{
#ifdef EMBEDDED_SPACES
    // start of implementation of embedded space support with quoting
    // not complete, not sure if we need it
    bool spacefound = false;
    wchar_t quote = 0;
    if (IsTextBased())
    {
        // search for embedded spaces and quotes
        wstring searchString = L" \"'~";
        const wchar_t* result = NULL;
        while (result = wcspbrk(str, searchString.c_str()))
        {
            if (IsWhiteSpace(*result))
                spacefound = true;
            searchString.find(*result, 0);
        }
    }
#endif
    if (size > 0)
    {
        fwprintf(m_file, L" %.*ls", size, str);
    }
    else
    {
        if (IsTextBased())
            fwprintf(m_file, L" %ls", str);
        else
            fputstring(m_file, str);
    }
}

// ReadString - reads a string from the file
// str - the string buffer to read the string into
// size - size of the string string buffer
void File::ReadString(wchar_t* str, int size)
{
    if (IsTextBased())
        fgettoken(m_file, str, size);
    else
        fgetstring(m_file, str, size);
}

// IsUnicodeBOM - is the next characters the Unicode Byte Order Mark?
// skip - skip the BOM mark if found (defaults to false)
// returns - true if on a unicode BOM
bool File::IsUnicodeBOM(bool skip)
{
    File& file = *this;
    uint64_t pos = GetPosition(); // Note: This is where we will fail for non-seekable streams.
    // if we aren't at the beginning of the file, it can't be the byte order mark
    if (pos != 0)
        return false;

    // only exists for UNICODE files
    bool found = false;
    if (m_options & fileOptionsText)
    {
        char val[3] = { 0 };
        for (size_t i = 0; i < _countof(val) && !file.IsEOF(); i++)
            val[i] = (char) getc(m_file);
        found = BeginsWithUnicodeBOM(val);
    }
    // restore pointer if no BOM or we aren't skipping it
    if (!found || !skip)
    {
        SetPosition(pos);
    }
    return found;
}

//Size - return the size of the file
// WARNING: calling this will reset the EOF marker, so do so with care
size_t File::Size()
{
    if (!CanSeek())
        RuntimeError("File: attempted to get Size() on non-seekable stream");
    return filesize(m_file);
}

// IsEOF - if we have read past the end of the file
// return - true if end of file has been found
bool File::IsEOF()
{
    return !!feof(m_file);
}

// IsWhiteSpace - are the next characters whitespace (space, \t, \r, \n, etc.)?
// skip - skip the whitespace if found (defaults to false)
// returns - true if whitespace found
// TODO: This function actually consumes the white-space characters. Document that behavior.
bool File::IsWhiteSpace(bool skip)
{
    bool spaceFound = false;
    bool spaceCur = false;
    int c;
    do
    {
        c = fgetc(m_file);
        if (c == EOF) // hit the end
            return spaceFound;
        spaceCur = !!isspace(c);
        spaceFound = spaceFound || spaceCur;
    } while (spaceCur && skip);
    // put back the last character (EOF is ignored)
    ungetc(c, m_file);

    return spaceFound;
}

// EndOfLineOrEOF - are the next characters an end of line sequence ('\r\n') possibly preceeded by (space, \t)? EOF detected too
// skip - skip the end of line if found (defaults to false)
// returns - true if end of line found, EOF if end of file found, or false if nothing found, in which case any leading space will have been stripped
int File::EndOfLineOrEOF(bool skip)
{
    if (IsTextBased())
        return fskipNewline(m_file, skip);
    else
        return false;
}

// Get a marker from the file
// some are ignored others are expecting characters
// must use GetMarker methods for those that require parameters
File& File::operator>>(FileMarker marker)
{
    File& file = *this;

    switch (marker)
    {
    case fileMarkerBeginFile: // beginning of file marker
        // check for Unicode BOM marker
        if (IsTextBased() && CanSeek()) // files from a pipe cannot begin with Unicode BOM, sorry
            IsUnicodeBOM(true);
        break;
    case fileMarkerEndFile: // end of file marker, should we throw if it's not the end of the file?
        if (!IsEOF())
            RuntimeError("fileMarkerEndFile not found");
        break;
    case fileMarkerBeginList: // Beginning of list marker
        // no marker written unless an list with a count header
        break;
    case fileMarkerListSeparator: // separate elements of a list
        // do nothing for now, built in space deliminter for all types (before type)
        // future: make this customizable, so you can specify a separator (i.e. ',')
        break;
    case fileMarkerEndList: // end of line/list marker
        if (IsTextBased())
        {
            int found = EndOfLineOrEOF(true);
            if (found != (int) true) // EOF can also be returned
                RuntimeError("Newline not found");
        }
        break;
    case fileMarkerBeginSection: // beginning of section
    case fileMarkerEndSection:   // end of section
        assert(false);           // sections should use a string modifier
        break;
    }
    return file;
}

// Get a marker from the file
// some are ignored others are expecting characters
// must use GetMarker methods for those that require parameters
// This function will fail for non-seekable streams.
bool File::IsMarker(FileMarker marker, bool skip)
{
    bool retval = false;
    switch (marker)
    {
    case fileMarkerBeginFile: // beginning of file marker
        // check for Unicode BOM marker
        retval = IsUnicodeBOM(skip);
        break;
    case fileMarkerEndFile: // end of file marker, should we throw if it's not the end of the file?
        retval = IsEOF();
        break;
    case fileMarkerBeginList: // Beginning of list marker
        // no marker written unless an list with a count header
        // should we try to validate BOL header (just know it's an int, not negative, etc.)
        break;
    case fileMarkerListSeparator: // separate elements of a list
        // do nothing for now, built in space deliminter for all types (before type)
        // future: make this customizable, so you can specify a separator (i.e. ',')
        break;
    case fileMarkerEndList: // end of line/list marker
        if (IsTextBased())
        {
            int eolSeen = false;
            eolSeen = EndOfLineOrEOF(skip);
            retval = (eolSeen == (int) true);
        }
        break;
    case fileMarkerBeginSection: // beginning of section
    case fileMarkerEndSection:   // end of section
        // can't destinquish from a string currently
        break;
    }
    return retval;
}

// GetMarker for beginning of list support (lists with a count)
// count - [out] returns the number of elements in the list
File& File::GetMarker(FileMarker marker, size_t& count)
{
    assert(marker == fileMarkerBeginList);
    marker; // only beginning of list supported for count file markers
    // use text based try, so it can fail without an exception
    if (IsTextBased())
        ftrygetText(m_file, count);
    else
        fget(m_file, count);
    return *this;
}

// GetMarker for section beginning and ending tags
// section - [in]name of section that is expected
File& File::GetMarker(FileMarker marker, const std::string& section)
{
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection);
    marker;
    string str;
    *this >> str;
    if (str != section)
        RuntimeError("section name mismatch %s != %s", str.c_str(), section.c_str());
    return *this;
}

// GetMarker for section beginning and ending tags
// section - [in]name of section that is expected
File& File::GetMarker(FileMarker marker, const std::wstring& section)
{
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection);
    marker;
    wstring str;
    *this >> str;
    if (str != section)
        RuntimeError("section name mismatch %ls != %ls", str.c_str(), section.c_str());
    return *this;
}

// TryGetMarker for section beginning and ending tags
// section - [in]name of section that is expected
bool File::TryGetMarker(FileMarker marker, const std::wstring& section)
{
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection);
    marker;
    size_t pos = GetPosition();
    std::wstring str;
    try
    {
        *this >> str;
        if (str == section)
            return true;
    }
    catch (...)
    {
        // eat
    }
    SetPosition(pos);
    return false;
}

// TryGetMarker for section beginning and ending tags
// section - [in]name of section that is expected
bool File::TryGetMarker(FileMarker marker, const std::string& section)
{
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection);
    marker;
    size_t pos = GetPosition();
    std::string str;
    try
    {
        *this >> str;
        if (str == section)
            return true;
    }
    catch (...)
    {
        return false;
    }
    SetPosition(pos);
    return false;
}

// GetPosition - Get position in a file
uint64_t File::GetPosition()
{
    if (!CanSeek())
        RuntimeError("File: attempted to GetPosition() on non-seekable stream");
    return fgetpos(m_file);
}

// Set the position in the file
// pos - position in the file
void File::SetPosition(uint64_t pos)
{
    if (!CanSeek())
        RuntimeError("File: attempted to SetPosition() on non-seekable stream");
    fsetpos(m_file, pos);
}

// helper to load a matrix from a stream (file or string literal)
// The input string is expected to contain one line per matrix row (natural printing order for humans).
// Inputs:
//  - getLineFn: a lambda that fills a string with the next input line (=next matrix row)
//               The lambda returns an empty string to denote the end.
// Outputs:
//  - numRows, numCols: matrix dimensions inferred from newlines
//  - array: matrix values in column-major order (ready for SetValue())
template<class ElemType, class F>
static void LoadMatrixFromLambda(const F& getLineFn, const wstring& locationForMsg, vector<ElemType>& array, size_t& /*out*/ numRows, size_t& /*out*/ numCols)
{
    // load matrix into vector of vectors (since we don't know the size in advance)
    vector<ElemType> vec;
    std::vector<std::vector<ElemType>> elements;
    size_t numColsInFirstRow = 0;

    std::string line;
    for(;;)
    {
        // get next input line
        getLineFn(line);
        if (line.empty())
            break;

        // tokenize and parse
        vec.clear();
        const char * p = line.c_str();
        for (;;)
        {
            while (isspace((unsigned char)*p))
                p++;
            if (!*p)
                break;
            char* ep; // will be set to point to first character that failed parsing
            double value = strtod(p, &ep);
            if (*ep != 0 && !isspace((unsigned char)*ep))
                RuntimeError("LoadMatrixFromTextFile: Malformed number '%.15s...' in row %d of %ls", p, (int)elements.size(), locationForMsg.c_str());
            p = ep;
            vec.push_back((ElemType)value);
        }

        size_t numElementsInRow = vec.size();
        if (elements.empty())
            numColsInFirstRow = numElementsInRow;
        else if (numElementsInRow != numColsInFirstRow)
            RuntimeError("Row %d has column dimension %d, inconsistent with previous dimension %d: %ls", (int)elements.size(), (int)numElementsInRow, (int)numColsInFirstRow, locationForMsg.c_str());

        elements.push_back(vec);
    }

    numRows = elements.size();
    numCols = numColsInFirstRow;

    // Perform transpose when copying elements from vectors to ElemType[],
    // in order to store in column-major format.
    array.resize(numRows * numCols);
    for (int i = 0; i < numCols; i++)
        for (int j = 0; j < numRows; j++)
            array[i * numRows + j] = elements[j][i];
}

// Load matrix from file. The file is a simple text file consisting of one line per matrix row, where each line contains the elements of the row separated by white space.
template <class ElemType>
/*static*/ vector<ElemType> File::LoadMatrixFromTextFile(const std::wstring& filePath, size_t& /*out*/ numRows, size_t& /*out*/ numCols)
{
    File myfile(filePath, FileOptions::fileOptionsText | FileOptions::fileOptionsRead);

    // LoadMatrixFromLambda() reads its input lines from the following lambda
    // return the next input line, or empty string when the end is reached
    auto getLineFn = [&](string& line)
    {
        while (!myfile.IsEOF())
        {
            myfile.GetLine(line);
            if (!line.empty())
                return; // got the next line to return
            // End of file manifests as an empty line at the end.
            // Also, we allow empty lines within the file, as that may help to visually structure matrices that really are >2D tensors.
        }
        line.clear(); // empty line indicates end of file
    };

    vector<ElemType> array;
    LoadMatrixFromLambda(getLineFn, filePath, array, numRows, numCols);
    return array;
}

// Load matrix from file. The file is a simple text file consisting of one line per matrix row, where each line contains the elements of the row separated by white space.
template <class ElemType>
/*static*/ vector<ElemType> File::LoadMatrixFromStringLiteral(const std::string& literal, size_t& /*out*/ numRows, size_t& /*out*/ numCols)
{
    // LoadMatrixFromLambda() reads its input lines from the following lambda
    // return the next input line, or empty string when the end is reached
    size_t pos = 0; // cursor for traversing the string. The lambda takes this by reference and modifies it.
    auto getLineFn = [&](string& line)
    {
        // find first non-blank character of line
        pos = literal.find_first_not_of(" \r\n", pos); // skip previous line end and any leading spaces
        if (pos == string::npos)
            return line.clear(); // hit the end: return empty line
        // find end of line
        auto endPos = literal.find_first_of("\r\n", pos + 1); // find line end
        if (endPos == string::npos)
            endPos = literal.size(); // no LF required at very end, so that it looks pretty in BS source code
        line = literal.substr(pos, endPos - pos);
        pos = endPos; // and advance cursor (we position it on the LF, which is skipped in next round)
        return;
    };

    vector<ElemType> array;
    LoadMatrixFromLambda(getLineFn, L"string literal", array, numRows, numCols);
    return array;
}

template vector<float>  File::LoadMatrixFromTextFile<float> (const std::wstring& filePath, size_t& /*out*/ numRows, size_t& /*out*/ numCols);
template vector<double> File::LoadMatrixFromTextFile<double>(const std::wstring& filePath, size_t& /*out*/ numRows, size_t& /*out*/ numCols);

template vector<float>  File::LoadMatrixFromStringLiteral<float> (const std::string& literal, size_t& /*out*/ numRows, size_t& /*out*/ numCols);
template vector<double> File::LoadMatrixFromStringLiteral<double>(const std::string& literal, size_t& /*out*/ numRows, size_t& /*out*/ numCols);

}}}
