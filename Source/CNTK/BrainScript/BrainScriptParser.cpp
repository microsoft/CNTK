// ConfigParser.cpp -- config parser (syntactic only, that is, source -> Expression tree)

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "BrainScriptParser.h"
#include "File.h"
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>
#include <deque>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace BS {

using namespace std;
using namespace msra::strfun;
using namespace Microsoft::MSR::CNTK;

// ---------------------------------------------------------------------------
// source files and text references (location) into them
// ---------------------------------------------------------------------------

// SourceFile constructors
SourceFile::SourceFile(wstring location, wstring text)
    : path(location), lines(split(text, L"\r\n"))
{
} // from string, e.g. command line
SourceFile::SourceFile(wstring path)
    : path(path) // from file
{
    File(path, fileOptionsRead | fileOptionsText).GetLines(lines);
}

bool TextLocation::IsValid() const
{
    return sourceFileAsIndex != SIZE_MAX;
}

// register a new source file and return a TextPosition that points to its start
/*static*/ TextLocation TextLocation::NewSourceFile(SourceFile&& sourceFile)
{
    TextLocation loc;
    loc.lineNo = 0;
    loc.charPos = 0;
    loc.sourceFileAsIndex = sourceFileMap.size(); // index under which we store the source file
    sourceFileMap.push_back(move(sourceFile));    // take ownership of the source file and give it a numeric index
    return loc;
}

// helper for pretty-printing errors: Show source-code line with ...^ under it to mark up the point of error
struct Issue
{
    TextLocation location; // using lineno and source file; char position only for printing the overall error loc
    wstring markup;        // string with markup symbols at char positions and dots inbetween
    void AddMarkup(wchar_t symbol, size_t charPos)
    {
        if (charPos >= markup.size())
            markup.resize(charPos + 1, L' '); // fill with '.' up to desired position if the string is not that long yet
        if (markup[charPos] == L' ')          // don't overwrite
            markup[charPos] = symbol;
    }
    Issue(TextLocation location)
        : location(location)
    {
    }
};

// trace
/*static*/ void TextLocation::Trace(TextLocation location, const wchar_t* traceKind, const wchar_t* op, const wchar_t* exprPath)
{
    fprintf(stderr, "%ls: %ls (path %ls)\n", traceKind, op, exprPath);
    const auto& lines = location.GetSourceFile().lines;
    const auto line = (location.lineNo == lines.size()) ? L"(end)" : lines[location.lineNo].c_str();
    Issue issue(location);
    issue.AddMarkup(L'^', location.charPos);
    fprintf(stderr, "  %ls\n  %ls\n", line, issue.markup.c_str());
}

// report an error
// The source line is shown, and the position is marked as '^'.
// Because it is often hard to recognize an issue only from the point where it occurred, we also report the history in compact visual form.
// Since often multiple contexts are on the same source line, we only print each source line once in a consecutive row of contexts.
/*static*/ void TextLocation::PrintIssue(const vector<TextLocation>& locations, const wchar_t* errorKind, const wchar_t* kind, const wchar_t* what)
{
    wstring error = CreateIssueMessage(locations, errorKind, kind, what);
    fprintf(stderr, "%ls", error.c_str());
    fflush(stderr);
}

/*static*/ wstring TextLocation::CreateIssueMessage(const vector<TextLocation>& locations, const wchar_t* errorKind, const wchar_t* kind, const wchar_t* what)
{
    vector<Issue> issues; // tracing the error backwards
    size_t symbolIndex = 0;
    wstring message;

    for (size_t n = 0; n < locations.size(); n++)
    {
        let& location = locations[n];
        if (!location.IsValid()) // means thrower has no location, go up one context
            continue;
        // build the array
        if (symbolIndex == 0 || location.lineNo != issues.back().location.lineNo || location.sourceFileAsIndex != issues.back().location.sourceFileAsIndex)
        {
            if (issues.size() == 10)
                break;
            else
                issues.push_back(location);
        }
        // get the symbol to indicate how many steps back, in this sequence: ^ 0..9 a..z A..Z (we don't go further than this)
        wchar_t symbol;
        if (symbolIndex == 0)
            symbol = '^';
        else if (symbolIndex < 1 + 10)
            symbol = '0' + (wchar_t) symbolIndex - 1;
        else if (symbolIndex < 1 + 10 + 26)
            symbol = 'a' + (wchar_t) symbolIndex - (1 + 10);
        else if (symbolIndex < 1 + 10 + 26 + 26)
            symbol = 'A' + (wchar_t) symbolIndex - (1 + 10 + 26);
        else
            break;
        symbolIndex++;
        // insert the markup
        issues.back().AddMarkup(symbol, location.charPos);
    }
    // print it backwards
    if (!locations.empty()) // (be resilient to some throwers not having a TextLocation; to be avoided)
    {
        let& firstLoc = issues.front().location;
        message += wstrprintf(L"[CALL STACK]\n");
        for (auto i = issues.rbegin(); i != issues.rend(); i++)
        {
            let& issue = *i;
            auto& where = issue.location;
            const auto& lines = where.GetSourceFile().lines;
            const auto line = (where.lineNo == lines.size()) ? L"(end)" : lines[where.lineNo].c_str();
            message += wstrprintf(L"  %ls\n  %ls\n", line, issue.markup.c_str());
        }
        message += wstrprintf(L"%ls while %ls: %ls(%d)", errorKind, kind, firstLoc.GetSourceFile().path.c_str(), (int)firstLoc.lineNo + 1 /*report 1-based*/);
    }
    else
    {
        message += wstrprintf(L"%ls while %ls", errorKind, kind);
    }
    message += wstrprintf(L": %ls\n", what);
    return message;
}
/*static*/ vector<SourceFile> TextLocation::sourceFileMap;

// ---------------------------------------------------------------------------
// reader -- reads source code, including loading from disk
// ---------------------------------------------------------------------------

class CodeSource
{
    vector<TextLocation> locationStack; // parent locations in case of included files
    TextLocation cursor;                // current location
    const wchar_t* currentLine;         // cache of cursor.GetSourceFile().lines[cursor.lineNo]
    // update currentLine from cursor
    void CacheCurrentLine()
    {
        let& lines = cursor.GetSourceFile().lines;
        if (cursor.lineNo == lines.size())
            currentLine = nullptr;
        else
            currentLine = lines[cursor.lineNo].c_str();
    }

protected:
    // set a source file; only do that from constructor or inside PushSourceFile()
    void SetSourceFile(SourceFile&& sourceFile)
    {
        cursor = TextLocation::NewSourceFile(move(sourceFile)); // save source file and set the cursor to its start
        CacheCurrentLine();                                     // re-cache current line
    }

public:
    class CodeSourceException : public ConfigException
    {
    public:
        CodeSourceException(const wstring& msg, TextLocation where)
            : ConfigException(msg, where)
        {
        }
        /*ConfigException::*/ const wchar_t* kind() const
        {
            return L"reading source";
        }
    };

    __declspec_noreturn static void Fail(wstring msg, TextLocation where)
    {
        //Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
        throw CodeSourceException(msg, where);
    }

    // enter a source file, at start or as a result of an include statement
    void PushSourceFile(SourceFile&& sourceFile)
    {
        locationStack.push_back(cursor);
        SetSourceFile(move(sourceFile));
    }

    // are we inside an include file?
    bool IsInInclude()
    {
        return locationStack.size() > 0;
    }

    // done with a source file. Only call this for nested files; the outermost one must not be popped.
    void PopSourceFile()
    {
        if (!IsInInclude())
            LogicError("PopSourceFile: location stack empty");
        cursor = locationStack.back(); // restore cursor we came from
        CacheCurrentLine();            // re-cache current line
        locationStack.pop_back();
    }

    // get current cursor; this is remembered for each token, and also used when throwing errors
    TextLocation GetCursor() const
    {
        return cursor;
    }

    // get character at current position.
    // Special cases:
    //  - end of line is returned as '\n'
    //  - end of file is returned as 0
    wchar_t GotChar() const
    {
        if (!currentLine)
            return 0; // end of file
        else if (!currentLine[cursor.charPos])
            return '\n'; // end of line
        else
            return currentLine[cursor.charPos];
    }

    // we chan also return the address of the current character, e.g. for passing it to a C stdlib funcion such as wcstod()
    const wchar_t* GotCharPtr() const
    {
        return currentLine + cursor.charPos;
    }

    // advance cursor by #chars (but across line boundaries)
    void ConsumeChars(size_t chars)
    {
        let ch = GotChar();
        if (!ch)
            LogicError("Consume: cannot run beyond end of source file");
        if (ch == '\n' && chars > 0)
        {
            if (chars != 1)
                LogicError("Consume: cannot run beyond end of line");
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
        ConsumeChars(1);
        return GotChar();
    }
};

// ---------------------------------------------------------------------------
// lexer -- iterates over the source code and returns token by token
// ---------------------------------------------------------------------------

class Lexer : public CodeSource
{
    set<wstring> keywords;
    set<wstring> punctuations;
    vector<wstring> includePaths;

public:
    Lexer(vector<wstring>&& includePaths)
        : CodeSource(), includePaths(includePaths), currentToken(TextLocation())
    {
        keywords = set<wstring>{
            L"include",
            L"new", L"with", L"true", L"false",
            L"if", L"then", L"else",
            L"array",
        };
        punctuations = set<wstring>{
            L"=", L";", L",", L"\n",
            L"[", L"]", L"(", L")",
            L"+", L"-", L"*", L"/", L"**", L".*", L"%", L"||", L"&&", L"^",
            L"!",
            L"==", L"!=", L"<", L"<=", L">", L">=",
            L":", L"=>",
            L"..", L".",
            L"//", L"#", L"/*"};
    }

    enum TokenKind
    {
        invalid,
        punctuation,
        numberliteral,
        stringliteral,
        booleanliter,
        identifier,
        keyword,
        eof // TODO: what are true and false? Literals or identifiers?
    };

    struct Token
    {
        wstring symbol; // identifier, keyword, punctuation, or string literal
        double number;  // number
        TokenKind kind;
        TextLocation beginLocation; // text loc of first character of this token
        bool isLineInitial;         // this token is the first on the line (ignoring comments)
        Token(TextLocation loc)
            : beginLocation(loc), kind(invalid), number(0.0), isLineInitial(false)
        {
        }
        // diagnostic helper
        static wstring TokenKindToString(TokenKind kind)
        {
            switch (kind)
            {
            case invalid:
                return L"invalid";
            case punctuation:
                return L"punctuation";
            case numberliteral:
                return L"numberliteral";
            case stringliteral:
                return L"stringliteral";
            case identifier:
                return L"identifier";
            case keyword:
                return L"keyword";
            case eof:
                return L"eof";
            default:
                return L"(unknown?)";
            }
        }
        wstring ToString() const // string to show the content of token for debugging
        {
            let kindStr = TokenKindToString(kind);
            switch (kind)
            {
            case numberliteral:
                return kindStr + wstrprintf(L" %f", number);
            case stringliteral:
                return kindStr + L" '" + symbol + L"'";
            case identifier:
            case keyword:
            case punctuation:
                return kindStr + L" " + symbol;
            default:
                return kindStr;
            }
        }
    };

    class LexerException : public ConfigException
    {
    public:
        LexerException(const wstring& msg, TextLocation where)
            : ConfigException(msg, where)
        {
        }
        /*ConfigException::*/ const wchar_t* kind() const
        {
            return L"tokenizing";
        }
    };

private:
    __declspec_noreturn static void Fail(wstring msg, Token where)
    {
        //Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
        throw LexerException(msg, where.beginLocation);
    }

    // find a file either at given location or traverse include paths
    // TODO: also allow ... syntax, where ... refers to the directory of the enclosing file
    static wstring FindSourceFile(const wstring& path, const vector<wstring>& includePaths)
    {
        if (File::Exists(path))
            return path;
        // non-existent path: scan include paths
        // TODO: This is a little weird. Rather, this should be done by the call site.
        let fileName = File::FileNameOf(path);
        for (let& dir : includePaths)
        {
            // TODO: We should use the separator that matches the include path.
            let newPath = dir + L"/" + fileName;
            if (File::Exists(newPath))
                return newPath;
        }
        // not in include path: try EXE directory
        let dir = File::DirectoryPathOf(File::GetExecutablePath());
        let newPath = dir + L"/" + fileName;
        if (File::Exists(newPath))
            return newPath;
        // not found: return unmodified, let caller fail
        return path;
    }

    Token currentToken;
    // consume input characters to form a next token
    //  - this function mutates the cursor, but does not set currentToken
    //  - white space and comments are skipped
    //  - including files is handled here
    //  - the cursor is left on the first character that does not belong to the token
    // TODO: need to know whether we want to see '\n' or not
    Token NextToken()
    {
        auto ch = GotChar();
        // skip white space
        // We remember whether we crossed a line end. Dictionary assignments end at newlines if syntactically acceptable.
        bool crossedLineEnd = (GetCursor().lineNo == 0 && GetCursor().charPos == 0);
        while (iswblank(ch) || ch == '\n' || ch == '\r')
        {
            crossedLineEnd |= (ch == '\n' || ch == '\r');
            ch = GetChar();
        }
        Token t(GetCursor());
        t.isLineInitial = crossedLineEnd;
        // handle end of (include) file
        if (ch == 0)
        {
            if (IsInInclude())
            {
                includePaths.erase(includePaths.begin()); // pop dir of current include file
                PopSourceFile();
                t = NextToken();        // tail call--the current 't' gets dropped/ignored
                t.isLineInitial = true; // eof is a line end
                return t;
            }
            // really end of all source code: we are done. If calling this function multiple times, we will keep returning this.
            t.kind = eof;
        }
        else if (iswdigit(ch) || (ch == L'.' && iswdigit(GotCharPtr()[1]))) // --- number
        {
            let beginPtr = GotCharPtr();
            wchar_t* endPtr = nullptr;
            t.number = wcstod(beginPtr, &endPtr); // BUGBUG: this seems to honor locale settings. We need one that doesn't. With this, CNTK won't parse right in Germany.
            if (endPtr == beginPtr)
                Fail(L"parsing number", t); // should not really happen!
            t.kind = numberliteral;
            if (endPtr[0] == L'.' && endPtr[-1] == L'.') // prevent 1..2 from begin tokenized 1. .2
                endPtr--;
            ConsumeChars(endPtr - beginPtr);
        }
        else if (iswalpha(ch) || ch == L'_') // --- identifier or keyword
        {
            while (iswalpha(ch) || ch == L'_' || iswdigit(ch)) // inside we also allow digits
            {
                t.symbol.push_back(ch);
                ch = GetChar();
            }
            // check against keyword list
            if (keywords.find(t.symbol) != keywords.end())
                t.kind = keyword;
            else
                t.kind = identifier;
            // special case: include "path"
            if (t.symbol == L"include")
            {
                let nameTok = NextToken(); // must be followed by a string literal
                if (nameTok.kind != stringliteral)
                    Fail(L"'include' must be followed by a quoted string", nameTok);
                let path = FindSourceFile(nameTok.symbol, includePaths);
                PushSourceFile(SourceFile(path)); // current cursor is right after the pathname; that's where we will pick up later
                includePaths.insert(includePaths.begin(), File::DirectoryPathOf(path));
                return NextToken();
            }
        }
        else if (ch == L'"' || ch == 0x27) // --- string literal
        {
            t.kind = stringliteral;
            let q = ch;                // remember quote character
            ch = GetChar();            // consume the quote character
            while (ch != 0 && ch != q) // note: our strings do not have any escape characters to consider
            {
                t.symbol.append(1, ch);
                ch = GetChar();
            }
            if (ch == 0) // runaway string
                Fail(L"string without closing quotation mark", t);
            GetChar(); // consume the closing quote
        }
        else // --- punctuation
        {
            t.kind = punctuation;
            t.symbol = ch;
            t.symbol.append(1, GetChar()); // first try two-char punctuation
            if (punctuations.find(t.symbol) != punctuations.end())
                GetChar(); // it is a two-char one: need to consume the second one of them
            else           // try single-char one
            {
                t.symbol.pop_back();                                   // drop the last one & try again
                if (punctuations.find(t.symbol) == punctuations.end()) // unknown
                    Fail(L"unexpected character: " + t.symbol, t);
            }
            // special case: comments
            if (t.symbol == L"#" || t.symbol == L"//")
            {
                ConsumeChars(wcslen(GotCharPtr()));
                return NextToken();
            }
            else if (t.symbol == L"/*")
            {
                ch = GotChar();
                while (ch != 0 && !(ch == L'*' && GetChar() == L'/')) // note: this test leverages short-circuit evaluation semantics of C
                    ch = GetChar();
                if (ch == 0)
                    Fail(L"comment without closing */", t);
                GetChar();          // consume the final '/'
                return NextToken(); // and return the next token
            }
        }
        return t;
    }

public:
    const Token& GotToken()
    {
        return currentToken;
    }
    void ConsumeToken()
    {
        currentToken = NextToken();
    }
    const Token& GetToken()
    {
        ConsumeToken();
        return GotToken();
    }

    // some simple test function
    void Test()
    {
        let lexerTest = L"new CNTK [ do = (train:eval) # main\ntrain=/*test * */if eval include 'c:/me/test.txt' then 13 else array[1..10](i=>i*i); eval=\"a\"+'b'  // line-end\n ] 'a\nb\nc' new";
        PushSourceFile(SourceFile(L"(command line)", lexerTest));
        while (GotToken().kind != Lexer::TokenKind::eof)
        {
            let& token = GotToken(); // get first token
            fprintf(stderr, "%ls\n", token.ToString().c_str());
            ConsumeToken();
        }
        Fail(L"error test", GetCursor());
    }
};

// ---------------------------------------------------------------------------
// parser -- parses configurations
// ---------------------------------------------------------------------------

// diagnostics helper: print the content
void Expression::DumpToStream(wstringstream & treeStream, int indent)
{
    treeStream << std::setfill(L' ') << std::setw(indent) << L" ";
    treeStream << std::setw(0);

    if (op == L"s")
        treeStream << "'" << s.c_str() << "'";
    else if (op == L"d")
        treeStream << std::fixed << std::setprecision(0) << d;
    else if (op == L"b")
        treeStream << b ? "true" : "false";
    else if (op == L"id")
        treeStream << id.c_str();
    else if (op == L"new" || op == L"array" || op == L".")
        treeStream << op.c_str() << " " << id.c_str();
    else
        treeStream << op.c_str();

    if (!args.empty())
    {
        treeStream << std::endl;
        for (const auto& arg : args)
        {
            arg->DumpToStream(treeStream, indent + 1);
        }
    }
    if (!namedArgs.empty())
    {
        treeStream << std::endl;
        for (const auto& arg : namedArgs)
        {
            treeStream << std::setfill(L' ') << std::setw(indent + 1) << L"";
            treeStream << arg.first.c_str() << L" =" << std::endl;
            arg.second.second->DumpToStream(treeStream, indent + 2);
        }
    }
    treeStream << std::endl;
}

class Parser : public Lexer
{
    // errors
    class ParseException : public ConfigException
    {
    public:
        ParseException(const wstring& msg, TextLocation where)
            : ConfigException(msg, where)
        {
        }
        /*ConfigException::*/ const wchar_t* kind() const
        {
            return L"parsing";
        }
    };

    __declspec_noreturn static void Fail(const wstring& msg, Token where)
    {
        //Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
        throw ParseException(msg, where.beginLocation);
    }

    //void Expected(const wstring & what) { Fail(strprintf("%ls expected", what.c_str()), GotToken().beginLocation); }  // I don't know why this does not work
    void Expected(const wstring& what)
    {
        Fail(what + L" expected", GotToken().beginLocation);
    }

    // this token must be punctuation 's'; check and get the next
    void ConsumePunctuation(const wchar_t* s)
    {
        let& tok = GotToken();
        if (tok.kind != punctuation || tok.symbol != s)
            Expected(L"'" + wstring(s) + L"'");
        ConsumeToken();
    }

    // this token must be keyword 's'; check and get the next
    void ConsumeKeyword(const wchar_t* s)
    {
        let& tok = GotToken();
        if (tok.kind != keyword || tok.symbol != s)
            Expected(L"'" + wstring(s) + L"'");
        ConsumeToken();
    }

    // this token must be an identifier; check and get the next token. Return the identifier.
    wstring ConsumeIdentifier()
    {
        let& tok = GotToken();
        if (tok.kind != identifier)
            Expected(L"identifier");
        let id = tok.symbol;
        ConsumeToken();
        return id;
    }

    map<wstring, int> infixPrecedence; // precedence level of infix operators
public:
    Parser(SourceFile&& sourceFile, vector<wstring>&& includePaths)
        : Lexer(move(includePaths))
    {
        infixPrecedence = map<wstring, int>{
            {L".", 99}, {L"[", 99}, {L"(",   99}, // also sort-of infix operands...
            {L"*", 10}, {L"/", 10}, {L".*",  10}, {L"**", 10}, {L"%", 10},
            {L"+",  9}, {L"-",  9}, {L"with", 9}, {L"==",  8},
            {L"!=", 8}, {L"<",  8}, {L"<=",   8}, {L">",   8}, {L">=", 8},
            {L"&&", 7},
            {L"||", 6},
            {L":",  5},
            {L"=>", 0},
        };
        SetSourceFile(move(sourceFile));
        ConsumeToken(); // get the very first token
    }
    ExpressionPtr OperandFromTokenSymbol(const Token& tok) // helper to make an Operand expression with op==tok.symbol and then consume it
    {
        auto operand = make_shared<Expression>(tok.beginLocation, tok.symbol);
        ConsumeToken();
        return operand;
    }
    ExpressionPtr ParseOperand(bool stopAtNewline)
    {
        let& tok = GotToken();
        ExpressionPtr operand;
        if (tok.kind == numberliteral) // === numeral literal
        {
            operand = make_shared<Expression>(tok.beginLocation, L"d", tok.number, wstring(), false);
            ConsumeToken();
        }
        else if (tok.kind == stringliteral) // === string literal
        {
            operand = make_shared<Expression>(tok.beginLocation, L"s", 0.0, tok.symbol, false);
            ConsumeToken();
        }
        else if (tok.symbol == L"true" || tok.symbol == L"false") // === boolean literal
        {
            operand = make_shared<Expression>(tok.beginLocation, L"b", 0.0, wstring(), (tok.symbol == L"true"));
            ConsumeToken();
        }
        else if (tok.kind == identifier) // === dict member (unqualified)
        {
            operand = make_shared<Expression>(tok.beginLocation, L"id");
            operand->id = ConsumeIdentifier();
        }
        else if (tok.symbol == L"+" || tok.symbol == L"-" // === unary operators
                 || tok.symbol == L"!")
        {
            operand = make_shared<Expression>(tok.beginLocation, tok.symbol + L"("); // encoded as +( -( !(
            ConsumeToken();
            operand->args.push_back(ParseExpression(100, stopAtNewline));
        }
        else if (tok.symbol == L"new") // === new class instance
        {
            operand = OperandFromTokenSymbol(tok);
            operand->id = ConsumeIdentifier();
            operand->args.push_back(ParseOperand(stopAtNewline));
        }
        else if (tok.symbol == L"if") // === conditional expression
        {
            operand = OperandFromTokenSymbol(tok);
            operand->args.push_back(ParseExpression(0, false)); // [0] condition
            ConsumeKeyword(L"then");
            operand->args.push_back(ParseExpression(0, false)); // [1] then expression
            ConsumeKeyword(L"else");
            operand->args.push_back(ParseExpression(0, false)); // [2] else expression
        }
        else if (tok.symbol == L"(") // === nested parentheses
        {
            ConsumeToken();
            operand = ParseExpression(0, false /*go across newlines*/); // just return the content of the parens (they do not become part of the expression tree)
            ConsumePunctuation(L")");
        }
        else if (tok.symbol == L"[") // === dictionary constructor
        {
            operand = make_shared<Expression>(tok.beginLocation, L"[]");
            ConsumeToken();
            operand->namedArgs = ParseRecordMembers();
            ConsumePunctuation(L"]");
        }
        else if (tok.symbol == L"array") // === array constructor
        {
            operand = OperandFromTokenSymbol(tok);
            ConsumePunctuation(L"[");
            operand->args.push_back(ParseExpression(0, false)); // [0] first index
            ConsumePunctuation(L"..");
            operand->args.push_back(ParseExpression(0, false)); // [1] last index
            ConsumePunctuation(L"]");
            ConsumePunctuation(L"(");
            operand->args.push_back(ParseExpression(0, false)); // [2] one-argument lambda to initialize
            ConsumePunctuation(L")");
        }
        else
            Expected(L"operand");
        return operand; // not using returns above to avoid "not all control paths return a value"
    }
    ExpressionPtr ParseExpression(int requiredPrecedence, bool stopAtNewline)
    {
        auto left = ParseOperand(stopAtNewline); // get first operand
        for (;;)
        {
            let& opTok = GotToken();
            // BUGBUG: 'stopAtNewline' is broken.
            // It does not prevent "a = 13 b = 42" from being accepted.
            // On the other hand, it would prevent the totally valid "dict \n with dict2".
            // A correct solution should require "a = 13 ; b = 42", i.e. a semicolon or newline,
            // while continuing to parse across newlines when syntactically meaningful (there is no ambiguity in BrainScript).
            //if (stopAtNewline && opTok.isLineInitial)
            //    break;
            let opIter = infixPrecedence.find(opTok.symbol);
            if (opIter == infixPrecedence.end()) // not an infix operator: we are done here, 'left' is our expression
                break;
            let opPrecedence = opIter->second;
            if (opPrecedence < requiredPrecedence) // operator below required precedence level: does not belong to this sub-expression
                break;
            let op = opTok.symbol;
            auto operation = make_shared<Expression>(opTok.beginLocation, op, left); // [0] is left operand; we will add [1] except for macro application
            // deal with special cases first
            // We treat member lookup (.), macro application (a()), and indexing (a[i]) together with the true infix operators.
            if (op == L".") // === reference of a dictionary item
            {
                ConsumeToken();
                operation->location = GotToken().beginLocation; // location of the identifier after the .
                operation->id = ConsumeIdentifier();
            }
            else if (op == L"=>")
            {
                if (left->op != L"id") // currently only allow for a single argument
                    Expected(L"identifier");
                ConsumeToken();
                let macroArgs = make_shared<Expression>(left->location, L"()", left); // wrap identifier in a '()' macro-args expression
                // TODO: test parsing of i => j => i*j
                let body = ParseExpression(opPrecedence, stopAtNewline); // pass same precedence; this makes '=>' right-associative  e.g.i=>j=>i*j
                operation->args[0] = macroArgs;                          // [0]: parameter list
                operation->args.push_back(body);                         // [1]: right operand
            }
            else if (op == L"(") // === macro application
            {
                // op = "("   means 'apply'
                // args[0] = lambda expression (lambda: op="=>", args[0] = param list, args[1] = expression with unbound vars)
                // args[1] = arguments    (arguments: op="(), args=vector of expressions, one per arg; and namedArgs)
                operation->args.push_back(ParseMacroArgs(false)); // [1]: all arguments
            }
            else if (op == L"[") // === array index
            {
                ConsumeToken();
                operation->args.push_back(ParseExpression(0, false)); // [1]: index
                ConsumePunctuation(L"]");
            }
            else if (op == L":")
            {
                // special case: (a : b : c) gets flattened into :(a,b,c) i.e. an operation with possibly >2 operands
                ConsumeToken();
                let right = ParseExpression(opPrecedence + 1, stopAtNewline); // get right operand, or entire multi-operand expression with higher precedence
                if (left->op == L":")                                         // appending to a list: flatten it
                {
                    operation->args = left->args;
                    operation->location = left->location; // location of first ':' (we need to choose some location)
                }
                operation->args.push_back(right); // form a list of multiple operands (not just two)
            }
            else // === regular infix operator
            {
                ConsumeToken();
                let right = ParseExpression(opPrecedence + 1, stopAtNewline); // get right operand, or entire multi-operand expression with higher precedence
                operation->args.push_back(right);                             // [1]: right operand
            }
            left = operation;
        }
        return left;
    }
    // a macro-args expression lists position-dependent and optional parameters
    // This is used both for defining macros (LHS) and using macros (RHS).
    // Result:
    //  op = "()"
    //  args = vector of arguments (which are given comma-separated)
    //         In case of macro definition, all arguments must be of type "id". Pass 'defining' to check for that.
    //  namedArgs = dictionary of optional args
    //         In case of macro definition, dictionary values are default values that are used if the argument is not given
    ExpressionPtr ParseMacroArgs(bool defining)
    {
        ConsumePunctuation(L"(");
        auto macroArgs = make_shared<Expression>(GotToken().beginLocation, L"()");
        if (GotToken().symbol != L")") // x() defines an empty argument list
        {
            for (;;)
            {
                let expr = ParseExpression(0, false); // this could be an optional arg (var = val)
                if (defining && expr->op != L"id")    // when defining we only allow a single identifier
                    Fail(L"argument identifier expected", expr->location);
                if (expr->op == L"id" && GotToken().symbol == L"=")
                {
                    let id = expr->id; // 'expr' gets resolved (to 'id') and forgotten
                    ConsumeToken();
                    let defValueExpr = ParseExpression(0, false); // default value
                    let res = macroArgs->namedArgs.insert(make_pair(id, make_pair(expr->location, defValueExpr)));
                    if (!res.second)
                        Fail(L"duplicate optional parameter '" + id + L"'", expr->location);
                }
                else
                    macroArgs->args.push_back(expr); // [0..]: position args
                if (GotToken().symbol != L",")
                    break;
                ConsumeToken();
            }
        }
        ConsumePunctuation(L")");
        return macroArgs;
    }
    map<wstring, pair<TextLocation, ExpressionPtr>> ParseRecordMembers()
    {
        // A dictionary is a map
        //  member identifier -> expression
        // Macro declarations are translated into lambdas, e.g.
        //  F(A,B) = expr(A,B)
        // gets represented in the dictionary as
        //  F = (A,B) => expr(A,B)
        // where a lambda expression has this structure:
        //  op="=>"
        //  args[0] = parameter list (op="()", with args (all of op="id") and namedArgs)
        //  args[1] = expression with unbound arguments
        // An array constructor of the form
        //  V[i:from..to] = expression of i
        // gets mapped to the explicit array operator
        //  V = array[from..to] (i => expression of i)
        map<wstring, pair<TextLocation, ExpressionPtr>> members;
        auto idTok = GotToken();
        while (idTok.kind == identifier)
        {
            let location = idTok.beginLocation; // for error message
            let id = ConsumeIdentifier();       // the member's name
            // optional array constructor
            ExpressionPtr arrayIndexExpr, fromExpr, toExpr;
            if (GotToken().symbol == L"[")
            {
                // X[i:from..to]
                ConsumeToken();
                arrayIndexExpr = ParseOperand(false); // 'i' name of index variable
                if (arrayIndexExpr->op != L"id")
                    Expected(L"identifier");
                ConsumePunctuation(L":");
                fromExpr = ParseExpression(0, false); // 'from' start index
                ConsumePunctuation(L"..");
                toExpr = ParseExpression(0, false); // 'to' end index
                ConsumePunctuation(L"]");
            }
            // optional macro args
            let parameters = (GotToken().symbol == L"(") ? ParseMacroArgs(true /*defining*/) : ExpressionPtr(); // optionally, macro arguments
            ConsumePunctuation(L"=");
            auto rhs = ParseExpression(0, true /*can end at newline*/); // and the right-hand side
            // if macro then rewrite it as an assignment of a lambda expression
            if (parameters)
                rhs = make_shared<Expression>(parameters->location, L"=>", parameters, rhs);
            // if array then rewrite it as an assignment of a array-constructor expression
            if (arrayIndexExpr)
            {
                // create a lambda expression over the index variable
                let macroArgs = make_shared<Expression>(arrayIndexExpr->location, L"()", arrayIndexExpr);      // wrap identifier in a '()' macro-args expression
                let initLambdaExpr = make_shared<Expression>(arrayIndexExpr->location, L"=>", macroArgs, rhs); // [0] is id, [1] is body
                rhs = make_shared<Expression>(location, L"array");
                rhs->args.push_back(fromExpr);       // [0] first index
                rhs->args.push_back(toExpr);         // [1] last index
                rhs->args.push_back(initLambdaExpr); // [2] one-argument lambda to initialize
            }
            // insert
            let res = members.insert(make_pair(id, make_pair(location, rhs)));
            if (!res.second)
                Fail(L"duplicate member definition '" + id + L"'", location);
            // advance
            idTok = GotToken();
            if (idTok.symbol == L";")
                idTok = GetToken();
        }
        return members;
    }
    void VerifyAtEnd()
    {
        if (GotToken().kind != eof)
            Fail(L"junk at end of source", GetCursor());
    }
    // top-level parse function parses dictonary members without enclosing [ ... ] and returns it as a dictionary
    ExpressionPtr ParseRecordMembersToDict()
    {
        let topMembers = ParseRecordMembers();
        VerifyAtEnd();
        ExpressionPtr topDict = make_shared<Expression>(GetCursor(), L"[]");
        topDict->namedArgs = topMembers;
        return topDict;
    }
};

// globally exported functions to execute the parser
static ExpressionPtr Parse(SourceFile&& sourceFile, vector<wstring>&& includePaths)
{
    return Parser(move(sourceFile), move(includePaths)).ParseRecordMembersToDict();
}
ExpressionPtr ParseConfigDictFromString(wstring text, wstring location, vector<wstring>&& includePaths)
{
    return Parse(SourceFile(location, text), move(includePaths));
}
//ExpressionPtr ParseConfigDictFromFile(wstring path, vector<wstring> includePaths)
//{
//    auto sourceFile = SourceFile(path); // note: no resolution against include paths done here
//    includePaths.insert(includePaths.begin(), File::DirectoryPathOf(path)); // must include our own path for nested include statements
//    return Parse(move(sourceFile), move(includePaths));
//}
ExpressionPtr ParseConfigExpression(const wstring& sourceText, vector<wstring>&& includePaths)
{
    auto parser = Parser(SourceFile(L"(command line)", sourceText), move(includePaths));
    auto expr = parser.ParseExpression(0, true /*can end at newline*/);
    parser.VerifyAtEnd();
    return expr;
}

}}}
