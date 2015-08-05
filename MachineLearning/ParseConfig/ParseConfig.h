// ParseConfig.h -- config parser

#pragma once

#include "Basics.h"
#include "File.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;
    using namespace msra::strfun;

    struct SourceFile               // content of one source file  (only in this header because TextLocation's private member uses it)
    {
        /*const*/ wstring path;                     // where it came from
        /*const*/ vector<wstring> lines;            // source code lines
        SourceFile(wstring location, wstring text) : path(location), lines(split(text, L"\r\n")) { }  // from string, e.g. command line
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
            const auto & lines = GetSourceFile().lines;
            const auto line = (lineNo == lines.size()) ? L"(end)" : lines[lineNo].c_str();
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

    struct Expression
    {
        Expression(TextLocation location) : location(location), d(0.0), b(false) { }
        wstring op;                 // operation, encoded as a string; 'symbol' for punctuation and keywords, otherwise used in constructors below ...TODO: use constexpr
        wstring id;                 // identifier;      op == "id", "new", "array", and "." (if macro then it also has args)
        wstring s;                  // string literal;  op == "s"
        double d;                   // numeric literal; op == "d"
        bool b;                     // boolean literal; op == "b"
        typedef shared_ptr<struct Expression> ExpressionRef;
        vector<ExpressionRef> args;             // position-dependent expression/function args
        map<wstring, ExpressionRef> namedArgs;  // named expression/function args; also dictionary members
        TextLocation location;      // where in the source code (for downstream error reporting)
        //Expression() : d(0.0), b(false) { }
        // diagnostics helper: print the content
        void Dump(int indent = 0) const
        {
            fprintf(stderr, "%*s", indent, "", op.c_str());
            if (op == L"s") fprintf(stderr, "'%ls' ", s.c_str());
            else if (op == L"d") fprintf(stderr, "%.f ", d);
            else if (op == L"b") fprintf(stderr, "%s ", b ? "true" : "false");
            else if (op == L"id") fprintf(stderr, "%ls ", id.c_str());
            else if (op == L"new" || op == L"array" || op == L".") fprintf(stderr, "%ls %ls ", op.c_str(), id.c_str());
            else fprintf(stderr, "%ls ", op.c_str());
            if (!args.empty())
            {
                fprintf(stderr, "\n");
                for (const auto & arg : args)
                    arg->Dump(indent+2);
            }
            if (!namedArgs.empty())
            {
                fprintf(stderr, "\n");
                for (const auto & arg : namedArgs)
                {
                    fprintf(stderr, "%*s%ls =\n", indent+2, "", arg.first.c_str());
                    arg.second->Dump(indent + 4);
                }
            }
            fprintf(stderr, "\n");
        }
    };
    typedef Expression::ExpressionRef ExpressionRef;    // circumvent some circular definition problem

    // access the parser through one of these two functions
    ExpressionRef ParseConfigString(wstring text);
    ExpressionRef ParseConfigFile(wstring path);

}}} // namespaces
