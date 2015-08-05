// ParseConfig.cpp : tool for developing and testing the config parser
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "File.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <deque>
#include <stdexcept>
#include <algorithm>

#ifndef let
#define let const auto
#endif

namespace Microsoft{ namespace MSR { namespace CNTK {

using namespace std;

struct SourceFile               // content of one source file
{
    /*const*/ wstring path;                     // where it came from
    /*const*/ vector<wstring> lines;            // source code lines
    SourceFile(wstring location, wstring text) : path(location), lines(msra::strfun::split(text, L"\r\n")) { }  // from string, e.g. command line
    SourceFile(wstring path) : path(path)       // from file
    {
        File(path, fileOptionsRead).GetLines(lines);
    }
};

struct TextLocation                 // position in the text. Lightweight value struct that we can copy around, even into dictionaries etc., for error messages
{
    // source-code locations are given by line number, character position, and the source file
    size_t lineNo, charPos;         // line number and character index (0-based)
    const SourceFile & GetSourceFile() const { return sourceFileMap[sourceFileAsIndex]; }    // get the corresponding source-code line

    // register a new source file and return a TextPosition that points to its start
    static TextLocation NewSourceFile(SourceFile && sourceFile)
    {
        TextLocation loc;
        loc.lineNo = 0;
        loc.charPos = 0;
        loc.sourceFileAsIndex = sourceFileMap.size();   // index under which we store the source file
        sourceFileMap.push_back(move(sourceFile));      // take ownership of the source file and give it a numeric index
        return loc;
    }
    TextLocation() : lineNo(SIZE_MAX), charPos(SIZE_MAX), sourceFileAsIndex(SIZE_MAX) { }   // default: location

    // helper for pretty-printing errors: Show source-code line with ...^ under it to mark up the point of error
    wstring FormatErroneousLine() const
    {
        let lines = GetSourceFile().lines;
        let line = (lineNo == lines.size()) ? L"(end)" : lines[lineNo].c_str();
        return wstring(line) + L"\n" + wstring(charPos, L'.') + L"^";
    }

    void PrintIssue(const char * errorKind, const char * kind, const char * what) const
    {
        fprintf(stderr, "%ls(%d): %s %s: %s\n%ls\n", GetSourceFile().path.c_str(), lineNo+1/*report 1-based*/, errorKind, kind, what, FormatErroneousLine().c_str());
    }

private:
    size_t sourceFileAsIndex;                   // source file is remembered in the value struct as an index into the static sourceFileMap[]
    // the meaning of the 'sourceFile' index is global, stored in this static map
    static vector<SourceFile> sourceFileMap;
};
/*static*/ vector<SourceFile> TextLocation::sourceFileMap;

// all errors from processing the config files are reported as ConfigError
class ConfigError : public runtime_error
{
    TextLocation location;
public:
    TextLocation where() const { return location; }
    virtual const char * kind() const = 0;
    ConfigError(const string & msg, TextLocation where) : location(where), runtime_error(msg) { }

    // pretty-print this as an error message
    void PrintError() const { location.PrintIssue("error", kind(), what()); }
};

// ---------------------------------------------------------------------------
// reader -- reads source code, including loading from disk
// ---------------------------------------------------------------------------

class CodeSource
{
    vector<TextLocation> locationStack; // parent locations in case of included files
    TextLocation cursor;                // current location
    const wchar_t * currentLine;        // cache of cursor.GetSourceFile().lines[cursor.lineNo]
    void CacheCurrentLine()             // update currentLine from cursor
    {
        let lines = cursor.GetSourceFile().lines;
        if (cursor.lineNo == lines.size())
            currentLine = nullptr;
        else
            currentLine = lines[cursor.lineNo].c_str();
    }
public:

    class CodeSourceError : public ConfigError
    {
    public:
        CodeSourceError(const string & msg, TextLocation where) : ConfigError(msg, where) { }
        /*implement*/ const char * kind() const { return "reading source"; }
    };

    TextLocation GetCursor() const { return cursor; }
    void Fail(string msg, TextLocation where) { throw CodeSourceError(msg, where); }

    // enter a source file, at start or as a result of an include statement
    void PushSourceFile(SourceFile && sourceFile)
    {
        locationStack.push_back(cursor);
        cursor = TextLocation::NewSourceFile(move(sourceFile));   // save source file and set the cursor to its start
        CacheCurrentLine();             // re-cache current line
    }

    // done with a source file
    void PopSourceFile()
    {
        if (locationStack.empty()) LogicError("PopSourceFile: location stack empty");
        cursor = locationStack.back();  // restore cursor we came from
        CacheCurrentLine();             // re-cache current line
        locationStack.pop_back();
    }

    // get character at current position.
    // Special cases:
    //  - end of line is returned as '\n'
    //  - end of file is returned as 0
    wchar_t GotChar() const
    {
        if (!currentLine) return 0;                             // end of file
        else if (!currentLine[cursor.charPos]) return '\n';     // end of line
        else return currentLine[cursor.charPos];
    }

    // we chan also return the address of the current character, e.g. for passing it to a C stdlib funcion such as wcstod()
    const wchar_t * GotCharPtr() const
    {
        return currentLine + cursor.charPos;
    }

    // advance cursor by #chars (but across line boundaries)
    void Consume(size_t chars)
    {
        let ch = GotChar();
        if (!ch) LogicError("Consume: cannot run beyond end of source file");
        if (ch == '\n')
        {
            if (chars != 1) LogicError("Consume: cannot run beyond end of line");
            cursor.lineNo++;
            CacheCurrentLine(); // line no has changed: re-cache the line ptr
            cursor.charPos = 0;
        }
        else
            cursor.charPos += chars;
    }

    // get the next character
    wchar_t GetChar()
    {
        Consume(1);
        return GotChar();
    }
};

// ---------------------------------------------------------------------------
// lexer -- iterates over the source code and returns token by token
// ---------------------------------------------------------------------------

class Lexer : CodeSource
{
};

// ---------------------------------------------------------------------------
// parser -- parses configurations
// ---------------------------------------------------------------------------

class Parser : Lexer
{
};

}}}   // namespaces

using namespace Microsoft::MSR::CNTK;

int wmain(int /*argc*/, wchar_t* /*argv*/[])
{
    try
    {
        CodeSource source;
        source.PushSourceFile(SourceFile(L"(command line)", L"this is a test\nand another line"));
        source.GotChar();
        source.GetChar();
        source.Fail("error test", source.GetCursor());
    }
    catch (const ConfigError & err)
    {
        err.PrintError();
    }
    return EXIT_SUCCESS;
}
