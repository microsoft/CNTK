//
// <copyright file="File.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif
#define _CRT_NONSTDC_NO_DEPRECATE   // make VS accept POSIX functions without _

#include "Basics.h"
#define FORMAT_SPECIALIZE // to get the specialized version of the format routines
#include "File.h"
#include <string>
#include <stdint.h>
#include <locale>
#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef __unix__
#include <unistd.h>
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
            msra::files::make_intermediate_dirs(m_filename.c_str());    // writing to regular file -> also create the intermediate directories as a convenience
        }
    }
    if (fileOptions&fileOptionsBinary)
    {
        options += L"b";
    }
    else
    {
        if (fileOptions & fileOptionsUnicode)
            options += L"b";    
        else
            options += L"t";
        // I attempted to use the translated characterset modes, but encountered strange errors
        //options += L"t, ccs=";
        //options += (fileOptions & fileOptionsUnicode)?L"UNICODE":L"UTF-8";
    }
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
    if (m_filename == L"-")                                 // stdin/stdout
    {
        if (writing && reading)
            RuntimeError("File: cannot specify fileOptionsRead and fileOptionsWrite at once with path '-'");
        m_file = writing ? stdout : stdin;
    }
    else if (outputPipe || inputPipe)                       // pipe syntax
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
    else attempt([=]()                                      // regular file: use a retry loop
    {
        m_file = fopenOrDie(filename, options.c_str());
        m_seekable = true;
    });
}

// skip to given delimiter character
void File::SkipToDelimiter(int delim)
{
    int ch=0;

    while (ch!=delim) {
        ch=fgetc(m_file);
        if (feof(m_file)) {
            printf("Unexpected end of file\n");
            LogicError("Unexpected end of file\n");
        }
    }
}

bool File::IsTextBased()
{
    return !!(m_options & (fileOptionsText|fileOptionsUnicode));
}

// File Destructor
// closes the file
// Note: this does not check for errors. Use Flush() before closing a file you are writing.
File::~File(void)
{
    if (m_pcloseNeeded)
        _pclose(m_file);
    else if (m_file != stdin && m_file != stdout && m_file != stderr)
        fclose(m_file); // (since destructors may not throw, we ignore the return code here)
}

void File::Flush()
{
    fflushOrDie(m_file);
}

// GetLine - get a line from the file
// str - string to store the line
void File::GetLine(wstring& str)
{
    str = fgetlinew(m_file);
}

// GetLine - get a line from the file
// str - string 
void File::GetLine(string& str)
{
    str = fgetline(m_file);
}

// GetLines - get all lines from a file
template<typename STRING> static void FileGetLines(File & file, std::vector<STRING>& lines)
{
    STRING line;
    while (!file.IsEOF())
    {
        file.GetLine(line);
        lines.push_back(line);
    }
}
void File::GetLines(std::vector<std::wstring>& lines) { FileGetLines(*this, lines); };
void File::GetLines(std::vector<std::string>&  lines) { FileGetLines(*this, lines); }


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
    switch(marker)
    {
    case fileMarkerBeginFile: // beginning of file marker
        // only exists for UNICODE files
        if (m_options & fileOptionsUnicode)
            file << (unsigned int)0xfeff; // byte order mark
        break;
    case fileMarkerEndFile: // end of file marker
        // use ^Z for end of file for text files
        if (m_options & fileOptionsUnicode)
            file << wchar_t(26); // ^Z
        else if (m_options & fileOptionsText)
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
        if (m_options & fileOptionsUnicode)
            file.WriteString(L"\r\n"); // carriage return/life feed
        else if (m_options & fileOptionsText)
            file.WriteString("\r\n");
        break;
    case fileMarkerBeginSection: // beginning of section
    case fileMarkerEndSection: // end of section
        assert(false);  // sections should use a string modifier 
        break;
    }
    return file;
}

// PutMarker for beginning of list support (lists with a count)
// count - [in] the number of elements in the list
File& File::PutMarker(FileMarker marker, size_t count)
{
    assert(marker == fileMarkerBeginList); marker; // only beginning of list supported for count  markers
    *this << count;
    return *this;
}

// PutMarker for section beginning and ending tags
// section - [in]name of section
File& File::PutMarker(FileMarker marker, const std::string& section)
{
    File& file = *this;
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection); marker;
    file << section;
    return file;
}

// PutMarker for section beginning and ending tags
// section - [in]name of section
File& File::PutMarker(FileMarker marker, const std::wstring& section)
{
    File& file = *this;
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection); marker;
    file << section;
    return file;
}

// Get a zero terminated wstring from a file
// val - value to read from the file
File& File::operator>>(std::wstring& val)
{
    attempt([&]
    {
        if (IsTextBased())
            val = fgetwtoken(m_file);
        else
            val = fgetwstring(m_file);
    });
    return *this;
}

// Get a zero terminated string from a file
// val - value to read from the file
File& File::operator>>(std::string& val)
{
    attempt([&]
    {
        if (IsTextBased())
            val = fgettoken(m_file);
        else
            val = fgetstring(m_file);
    });
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
    char *str = const_cast<char *>(val.c_str());
    for (int i=0;i < cnt;++i)
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
    wchar_t *str = const_cast<wchar_t *>(val.c_str());
    for (int i=0;i < cnt;++i)
        *this >> str[i];
    if (reset)
        SetPosition(pos);
}

// WriteString - outputs a string into the file
// str - the string to output
// size - size of the string to output, if zero null terminated
void File::WriteString(const char* str, int size)
{
    attempt([&]{
        if (size > 0)
        {
            fwprintf(m_file, L" %.*hs", size, str);
        }
        else
        {
            if (IsTextBased())
                fwprintf(m_file, L" %hs", str);
            else
                fputstring (m_file, str);
        }
    });
}

// ReadString - reads a string into the file
// str - the string buffer to read the string into
// size - size of the string string buffer
void File::ReadString(char* str, int size)
{
    attempt([&]{
        if (IsTextBased())
            fgettoken(m_file, str, size);
        else
            fgetstring (m_file, str, size);
    });
}

// WriteString - outputs a string into the file
//   if writing to text based file and spaces are embedded, writes quotes around string
// str - the string to output
// size - size of the string to output, if zero null terminated
void File::WriteString(const wchar_t* str, int size)
{
    attempt([&]{
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
                fputstring (m_file, str);
        }
    });
}

// ReadString - reads a string into the file
// str - the string buffer to read the string into
// size - size of the string string buffer
void File::ReadString(wchar_t* str, int size)
{
    attempt([&]
    {
        if (IsTextBased())
            fgettoken(m_file, str, size);
        else
            fgetstring (m_file, str, size);
    });
}

// IsUnicodeBOM - is the next characters the Unicode Byte Order Mark?
// skip - skip the BOM mark if found (defaults to false)
// returns - true if on a unicode BOM
bool File::IsUnicodeBOM(bool skip)
{
    File& file = *this;
    uint64_t pos = GetPosition();
    // if we aren't at the beginning of the file, it can't be the byte order mark
    if (pos != 0)
        return false;

    // only exists for UNICODE files
    bool found = false;
    if (m_options & fileOptionsUnicode)
    {
        unsigned int bom=0;
        if (IsTextBased())
            ftrygetText(m_file, bom);
        else
            fget(m_file, bom);
        // future: one reason for the BOM is to detect other-endian files, should we support?
        found = (bom == 0xfeff);
    }
    else if (m_options & fileOptionsText)
    {
        char val[3];
        file.ReadString(val, 3);
        found = (val[0] == 0xEF && val[1] == 0xBB && val[2] == 0xBF);
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
bool File::IsWhiteSpace(bool skip)
{
    bool spaceFound = false;
    bool spaceCur = false;
    if (m_options & fileOptionsUnicode)
    {
        wint_t c;
        do
        {
            c = fgetwc (m_file);
            if (c == WEOF)       // hit the end
                return spaceFound;
            spaceCur = !!iswspace(c);
            spaceFound = spaceFound || spaceCur;
        } while (spaceCur && skip);
        // put back the last character (WEOF is ignored)
        ungetwc(c, m_file);
    }
    else
    {
        int c;
        do
        {
            c = fgetc (m_file);
            if (c == EOF)       // hit the end
                return spaceFound;
            spaceCur = !!isspace(c);
            spaceFound = spaceFound || spaceCur;
        } while (spaceCur && skip);
        // put back the last character (EOF is ignored)
        ungetc(c, m_file);
    }

    return spaceFound;
}

// EndOfLineOrEOF - are the next characters an end of line sequence ('\r\n') possibly preceeded by (space, \t)? EOF detected too
// skip - skip the end of line if found (defaults to false)
// returns - true if end of line found, EOF if end of file found, or false if nothing found, in which case any leading space will have been stripped
int File::EndOfLineOrEOF(bool skip)
{
    int found = false;
    if (m_options & fileOptionsUnicode)
        found = fskipwNewline(m_file,skip);
    else if (m_options & fileOptionsText)
        found = fskipNewline(m_file, skip);
    return found;
}


// Get a marker from the file
// some are ignored others are expecting characters
// must use GetMarker methods for those that require parameters
File& File::operator>>(FileMarker marker)
{
    File& file = *this;

    switch(marker)
    {
    case fileMarkerBeginFile: // beginning of file marker
        // check for Unicode BOM marker
        if (IsTextBased())
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
            if (found != (int)true) // EOF can also be returned
                RuntimeError("Newline not found");
        }
        break;
    case fileMarkerBeginSection: // beginning of section
    case fileMarkerEndSection: // end of section
        assert(false);  // sections should use a string modifier 
        break;
    }
    return file;
}

// Get a marker from the file
// some are ignored others are expecting characters
// must use GetMarker methods for those that require parameters
bool File::IsMarker(FileMarker marker, bool skip)
{
    bool retval = false;
    switch(marker)
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
            retval = (eolSeen == (int)true);
        }
        break;
    case fileMarkerBeginSection: // beginning of section
    case fileMarkerEndSection: // end of section
        // can't destinquish from a string currently
        break;
    }
    return retval;
}


// GetMarker for beginning of list support (lists with a count)
// count - [out] returns the number of elements in the list
File& File::GetMarker(FileMarker marker, size_t& count)
{
    assert(marker == fileMarkerBeginList); marker; // only beginning of list supported for count file markers
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
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection); marker;
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
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection); marker;
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
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection); marker;
    size_t pos = GetPosition();
    std::wstring str;
    try
    {
        *this >> str;
        if (str == section)
            return true;
    }
    catch(...)
    {
        //eat
    }
    SetPosition(pos);
    return false;
}

// TryGetMarker for section beginning and ending tags
// section - [in]name of section that is expected
bool File::TryGetMarker(FileMarker marker, const std::string& section)
{
    // only the section markers take a string parameter
    assert(marker == fileMarkerBeginSection || marker == fileMarkerEndSection); marker;
    size_t pos = GetPosition();
    std::string str;
    try
    {
        *this >> str;
        if (str == section)
            return true;
    }
    catch(...)
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

}}}
