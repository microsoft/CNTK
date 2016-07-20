// ConfigParser.h -- config parser (syntactic only, that is, source -> Expression tree)

#pragma once

#include "Basics.h"
#include "ScriptableObjects.h"
#include "File.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace Microsoft { namespace MSR { namespace BS {

using namespace std;

// ---------------------------------------------------------------------------
// TextLocation -- holds a pointer into a source file
// ---------------------------------------------------------------------------

struct SourceFile // content of one source file  (only in this header because TextLocation's private member uses it)
{
    /*const*/ wstring path;                     // where it came from
    /*const*/ vector<wstring> lines;            // source code lines
    SourceFile(wstring location, wstring text); // from string, e.g. command line
    SourceFile(wstring path);                   // from file
};

struct TextLocation // position in the text. Lightweight value struct that we can copy around, even into dictionaries etc., for error messages
{
    // source-code locations are given by line number, character position, and the source file
    size_t lineNo, charPos; // line number and character index (0-based)
    const SourceFile& GetSourceFile() const
    {
        return sourceFileMap[sourceFileAsIndex];
    } // get the corresponding source-code line

    // helpers for pretty-printing errors: Show source-code line with ...^ under it to mark up the point of error
    static void PrintIssue(const vector<TextLocation>& locations, const wchar_t* errorKind, const wchar_t* kind, const wchar_t* what);
    static std::wstring CreateIssueMessage(const vector<TextLocation>& locations, const wchar_t* errorKind, const wchar_t* kind, const wchar_t* what);
    static void Trace(TextLocation, const wchar_t* traceKind, const wchar_t* op, const wchar_t* exprPath);

    // construction
    TextLocation()
        : lineNo(SIZE_MAX), charPos(SIZE_MAX), sourceFileAsIndex(SIZE_MAX)
    {
    } // default constructor constructs an unmissably invalid object
    bool IsValid() const;

    // register a new source file and return a TextPosition that points to its start
    static TextLocation NewSourceFile(SourceFile&& sourceFile);

private:
    size_t sourceFileAsIndex; // source file is remembered in the value struct as an index into the static sourceFileMap[]
    // the meaning of the 'sourceFile' index is global, stored in this static map
    static vector<SourceFile> sourceFileMap;
};

// ---------------------------------------------------------------------------
// ConfigException -- all errors from processing the config files are reported as ConfigException
// ---------------------------------------------------------------------------

class ConfigException : public Microsoft::MSR::ScriptableObjects::ScriptingException
{
    vector<TextLocation> locations; // error location (front()) and evaluation parents (upper)
public:
    // Note: All our Error objects use wide strings, which we round-trip through runtime_error as utf8.
    ConfigException(const wstring& msg, TextLocation where)
        : Microsoft::MSR::ScriptableObjects::ScriptingException(msra::strfun::utf8(msg))
    {
        locations.push_back(where);
    }

    // these are used in pretty-printing
    TextLocation where() const
    {
        return locations.front();
    }                                        // where the error happened
    virtual const wchar_t* kind() const = 0; // e.g. "warning" or "error"

    wstring GetError(const std::wstring& linePrefix) const override
    {
        return TextLocation::CreateIssueMessage(locations, linePrefix.c_str(), kind(), msra::strfun::utf16(what()).c_str());
    }
    // pretty-print this as an error message
    void /*ScriptingException::*/ PrintError(const std::wstring& linePrefix) const override
    {
        TextLocation::PrintIssue(locations, linePrefix.c_str(), kind(), msra::strfun::utf16(what()).c_str());
    }
    void AddLocation(TextLocation where)
    {
        locations.push_back(where);
    }
};

// ---------------------------------------------------------------------------
// Expression -- the entire config is a tree of Expression types
// We don't use polymorphism here because C++ is so verbose...
// ---------------------------------------------------------------------------

struct Expression
{
    wstring op; // operation, encoded as a string; 'symbol' for punctuation and keywords, otherwise used in constructors below ...TODO: use constexpr
    wstring id; // identifier;      op == "id", "new", "array", and "." (if macro then it also has args)
    wstring s;  // string literal;  op == "s"
    double d;   // numeric literal; op == "d"
    bool b;     // boolean literal; op == "b"
    typedef shared_ptr<struct Expression> ExpressionPtr;
    vector<ExpressionPtr> args;                                // position-dependent expression/function args
    map<wstring, pair<TextLocation, ExpressionPtr>> namedArgs; // named expression/function args; also dictionary members (loc is of the identifier)
    TextLocation location;                                     // where in the source code (for downstream error reporting)
    // constructors
    Expression(TextLocation location)
        : location(location), d(0.0), b(false)
    {
    }
    Expression(TextLocation location, wstring op)
        : location(location), d(0.0), b(false), op(op)
    {
    }
    Expression(TextLocation location, wstring op, double d, wstring s, bool b)
        : location(location), d(d), s(s), b(b), op(op)
    {
    }
    Expression(TextLocation location, wstring op, ExpressionPtr arg)
        : location(location), d(0.0), b(false), op(op)
    {
        args.push_back(arg);
    }
    Expression(TextLocation location, wstring op, ExpressionPtr arg1, ExpressionPtr arg2)
        : location(location), d(0.0), b(false), op(op)
    {
        args.push_back(arg1);
        args.push_back(arg2);
    }
    // diagnostics helper: print the content
    void Dump(int indent = 0) const;
    void DumpToStream(wstringstream & treeStream, int indent = 0);
};
typedef Expression::ExpressionPtr ExpressionPtr; // circumvent some circular definition problem

// access the parser through one of these functions
ExpressionPtr ParseConfigDictFromString(wstring text, wstring location, vector<wstring>&& includePaths);          // parses a list of dictionary members, returns a dictionary expression
// TODO: These rvalue references are no longer adding value, change to const<>&
//ExpressionPtr ParseConfigDictFromFile(wstring path, vector<wstring> includePaths);              // likewise, but from a file path
ExpressionPtr ParseConfigExpression(const wstring& sourceText, vector<wstring>&& includePaths); // parses a single expression from sourceText, which is meant to contain an include statement, hence includePaths

}}}
