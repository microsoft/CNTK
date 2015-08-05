// ParseConfig.cpp -- config parser

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ParseConfig.h"
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>
#include <deque>
#include <set>
#include <stdexcept>
#include <algorithm>

#ifndef let
#define let const auto
#endif

namespace Microsoft{ namespace MSR { namespace CNTK {

using namespace std;
using namespace msra::strfun;

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
    // update currentLine from cursor
    void CacheCurrentLine()
    {
        let & lines = cursor.GetSourceFile().lines;
        if (cursor.lineNo == lines.size())
            currentLine = nullptr;
        else
            currentLine = lines[cursor.lineNo].c_str();
    }
protected:
    // set a source file; only do that from constructor or inside PushSourceFile()
    void SetSourceFile(SourceFile && sourceFile)
    {
        cursor = TextLocation::NewSourceFile(move(sourceFile)); // save source file and set the cursor to its start
        CacheCurrentLine();             // re-cache current line
    }
public:
    class CodeSourceError : public ConfigError
    {
    public:
        CodeSourceError(const string & msg, TextLocation where) : ConfigError(msg, where) { }
        /*implement*/ const char * kind() const { return "reading source"; }
    };

    void Fail(string msg, TextLocation where) { throw CodeSourceError(msg, where); }

    // enter a source file, at start or as a result of an include statement
    void PushSourceFile(SourceFile && sourceFile)
    {
        locationStack.push_back(cursor);
        SetSourceFile(move(sourceFile));
    }

    // are we inside an include file?
    bool IsInInclude() { return locationStack.size() > 0; }

    // done with a source file. Only call this for nested files; the outermost one must not be popped.
    void PopSourceFile()
    {
        if (!IsInInclude())
            LogicError("PopSourceFile: location stack empty");
        cursor = locationStack.back();  // restore cursor we came from
        CacheCurrentLine();             // re-cache current line
        locationStack.pop_back();
    }

    // get current cursor; this is remembered for each token, and also used when throwing errors
    TextLocation GetCursor() const { return cursor; }

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
    void ConsumeChars(size_t chars)
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
public:
    Lexer() : CodeSource(), currentToken(TextLocation())
    {
        keywords = set<wstring>
        {
            L"include",
            L"new", L"true", L"false",
            L"if", L"then", L"else",
            L"array",
        };
        punctuations = set<wstring>
        {
            L"=", L";", L"\n",
            L"[", L"]", L"(", L")",
            L"+", L"-", L"*", L"/", L"**", L".*", L"%", L"||", L"&&", L"^",
            L"!",
            L"==", L"!=", L"<", L"<=", L">", L">=",
            L":", L"=>",
            L"..", L".",
            L"//", L"#", L"/*"
        };
    }

    enum TokenKind
    {
        invalid, punctuation, numberliteral, stringliteral, booleanliter, identifier, keyword, eof  // TODO: what are true and false? Literals or identifiers?
    };

    struct Token
    {
        wstring symbol;             // identifier, keyword, punctuation, or string literal
        double number;              // number
        TokenKind kind;
        TextLocation beginLocation; // text loc of first character of this token
        Token(TextLocation loc) : beginLocation(loc), kind(invalid), number(0.0) { }
        // diagnostic helper
        static wstring TokenKindToString(TokenKind kind)
        {
            switch (kind)
            {
            case invalid: return L"invalid";
            case punctuation: return L"punctuation";
            case numberliteral: return L"numberliteral";
            case stringliteral: return L"stringliteral";
            case identifier: return L"identifier";
            case keyword: return L"keyword";
            case eof: return L"eof";
            default: return L"(unknown?)";
            }
        }
        wstring ToString() const    // string to show the content of token for debugging
        {
            let kindStr = TokenKindToString(kind);
            switch (kind)
            {
            case numberliteral: return kindStr + wstrprintf(L" %f", number);
            case stringliteral: return kindStr + L" '" + symbol + L"'";
            case identifier: case keyword: case punctuation: return kindStr + L" " + symbol;
            default: return kindStr;
            }
        }
    };

    class LexerError : public CodeSourceError
    {
    public:
        LexerError(const string & msg, TextLocation where) : CodeSourceError(msg, where) { }
        /*implement*/ const char * kind() const { return "tokenizing"; }
    };

private:
    void Fail(string msg, Token where) { throw LexerError(msg, where.beginLocation); }

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
        while (iswblank(ch) || (ch == '\n' /* and ...*/))    // TODO: need to be newline-sensitive
            ch = GetChar();
        Token t(GetCursor());
        // handle end of (include) file
        if (ch == 0)
        {
            if (IsInInclude())
            {
                PopSourceFile();
                return NextToken();      // tail call--the current 't' gets dropped/ignored
            }
            // really end of all source code: we are done. If calling this function multiple times, we will keep returning this.
            t.kind = eof;
        }
        else if (iswdigit(ch) || (ch == L'.' && iswdigit(GotCharPtr()[1])))  // --- number
        {
            let beginPtr = GotCharPtr();
            wchar_t * endPtr = nullptr;
            t.number = wcstod(beginPtr, &endPtr);   // BUGBUG: this seems to honor locale settings. We need one that doesn't. With this, CNTK won't parse right in Germany.
            if (endPtr == beginPtr) Fail("parsing number", t);  // should not really happen!
            t.kind = numberliteral;
            if (endPtr[0] == L'.' && endPtr[-1] == L'.')    // prevent 1..2 from begin tokenized 1. .2
                endPtr--;
            ConsumeChars(endPtr - beginPtr);
        }
        else if (iswalpha(ch) || ch == L'_')                            // --- identifier or keyword
        {
            while (iswalpha(ch) || ch == L'_' || iswdigit(ch))          // inside we also allow digits
            {
                t.symbol.push_back(ch);
                ch = GetChar();
            }
            // check against keyword list
            if (keywords.find(t.symbol) != keywords.end()) t.kind = keyword;
            else t.kind = identifier;
            // special case: include "path"
            if (t.symbol == L"include")
            {
                let nameTok = NextToken();       // must be followed by a string literal
                if (nameTok.kind != stringliteral) Fail("'include' must be followed by a quoted string", nameTok);
                let path = nameTok.symbol;          // TODO: some massaging of the path
                PushSourceFile(SourceFile(path));   // current cursor is right after the pathname; that's where we will pick up later
                return NextToken();
            }
        }
        else if (ch == L'"' || ch == 0x27)                              // --- string literal
        {
            t.kind = stringliteral;
            let q = ch;     // remember quote character
            ch = GetChar(); // consume the quote character
            while (ch != 0 && ch != q)  // note: our strings do not have any escape characters to consider
            {
                t.symbol.append(1, ch);
                ch = GetChar();
            }
            if (ch == 0)    // runaway string
                Fail("string without closing quotation mark", t);
            GetChar();  // consume the closing quote
        }
        else                                                            // --- punctuation
        {
            t.kind = punctuation;
            t.symbol = ch;
            t.symbol.append(1, GetChar());                              // first try two-char punctuation
            if (punctuations.find(t.symbol) != punctuations.end())
                GetChar();                                              // it is a two-char one: need to consume the second one of them
            else                                                        // try single-char one
            {
                t.symbol.pop_back();                                    // drop the last one & try again
                if (punctuations.find(t.symbol) == punctuations.end())  // unknown
                    Fail("unexpected character", t);
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
                while (ch != 0 && !(ch == L'*' && GetChar() == L'/'))   // note: this test leverages short-circuit evaluation semantics of C
                    ch = GetChar();
                if (ch == 0)
                    Fail("comment without closing */", t);
                GetChar();  // consume the final '/'
                return NextToken();  // and return the next token
            }
        }
        return t;
    }
public:
    const Token & GotToken() { return currentToken; }
    void ConsumeToken() { currentToken = NextToken(); }
    const Token & GetToken()
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
            let & token = GotToken();   // get first token
            fprintf(stderr, "%ls\n", token.ToString().c_str());
            ConsumeToken();
        }
        Fail("error test", GetCursor());
    }
};

// ---------------------------------------------------------------------------
// parser -- parses configurations
// ---------------------------------------------------------------------------

class Parser : public Lexer
{
    class ParseError : public LexerError
    {
    public:
        ParseError(const string & msg, TextLocation where) : LexerError(msg, where) { }
        /*implement*/ const char * kind() const { return "parsing"; }
    };

    void Fail(const string & msg, Token where) { throw LexerError(msg, where.beginLocation); }

    //void Expected(const wstring & what) { Fail(strprintf("%ls expected", what.c_str()), GotToken().beginLocation); }  // I don't know why this does not work
    void Expected(const wstring & what) { Fail(utf8(what) + " expected", GotToken().beginLocation); }

    void ConsumePunctuation(const wchar_t * s)
    {
        let & tok = GotToken();
        if (tok.kind != punctuation || tok.symbol != s)
            Expected(L"'" + wstring(s) + L"'");
        ConsumeToken();
    }

    wstring ConsumeIdentifier()
    {
        let & tok = GotToken();
        if (tok.kind != identifier)
            Expected(L"identifier");
        let id = tok.symbol;
        ConsumeToken();
        return id;
    }

    map<wstring, int> infixPrecedence;      // precedence level of infix operators
public:
    Parser(SourceFile && sourceFile) : Lexer()
    {
        infixPrecedence = map<wstring, int>
        {
            { L".", 11 }, { L"[", 11 }, { L"(", 11 },     // also sort-of infix operands...
            { L"*", 10 }, { L"/", 10 }, { L".*", 10 }, { L"**", 10 }, { L"%", 10 },
            { L"+", 9 }, { L"-", 9 },
            { L"==", 8 }, { L"!=", 8 }, { L"<", 8 }, { L"<=", 8 }, { L">", 8 }, { L">=", 8 },
            { L"&&", 7 },
            { L"||", 6 },
            { L":", 5 },
        };
        SetSourceFile(move(sourceFile));
        ConsumeToken();     // get the very first token
    }
    ExpressionRef ParseOperand()
    {
        let & tok = GotToken();
        ExpressionRef operand = make_shared<Expression>(tok.beginLocation);
        if (tok.kind == numberliteral)                                  // === numeral literal
        {
            operand->op = L"d";
            operand->d = tok.number;
            ConsumeToken();
        }
        else if (tok.kind == stringliteral)                             // === string literal
        {
            operand->op = L"s";
            operand->s = tok.symbol;
            ConsumeToken();
        }
        else if (tok.symbol == L"true" || tok.symbol == L"false")       // === boolean literal
        {
            operand->op = L"b";
            operand->b = (tok.symbol == L"true");
            ConsumeToken();
        }
        else if (tok.kind == identifier)                                // === dict member (unqualified)
        {
            operand->op = L"id";
            operand->id = ConsumeIdentifier();
        }
        else if (tok.symbol == L"+" || tok.symbol == L"-"               // === unary operators
            || tok.symbol == L"!")
        {
            operand->op = tok.symbol;
            ConsumeToken();
            operand->args.push_back(ParseOperand());
        }
        else if (tok.symbol == L"new")                                  // === new class instance
        {
            operand->op = tok.symbol;
            ConsumeToken();
            operand->id = ConsumeIdentifier();
            operand->args.push_back(ParseOperand());
        }
        else if (tok.symbol == L"(")                                    // === nested parentheses
        {
            ConsumeToken();
            operand = ParseExpression(0, false/*go across newlines*/);  // note: we abandon the current operand object
            ConsumePunctuation(L")");
        }
        else if (tok.symbol == L"[")                                    // === dictionary constructor
        {
            operand->op = L"[]";
            ConsumeToken();
#if 1
            let namedArgs = ParseDictMembers();  // ...CONTINUE HERE
            for (const auto & arg : namedArgs)
                operand->namedArgs.insert(make_pair(arg.first->id, arg.second));
#endif
            /*operand->namedArgs = */ParseDictMembers();  // ...CONTINUE HERE
            ConsumePunctuation(L"]");
        }
        else if (tok.symbol == L"array")                                // === array constructor
        {
            operand->op = tok.symbol;
            ConsumeToken();
            ConsumePunctuation(L"[");
            operand->args.push_back(ParseExpression(0, false));         // [0] first index
            ConsumePunctuation(L"..");
            operand->args.push_back(ParseExpression(0, false));         // [1] last index
            ConsumePunctuation(L"]");
            ConsumePunctuation(L"(");
            // Note: needs a new local scope for this
            operand->id = ConsumeIdentifier();                          // identifier kept here
            ConsumePunctuation(L"=>");
            operand->args.push_back(ParseExpression(0, false));         // [2] function expression
            ConsumePunctuation(L")");
        }
        else
            Expected(L"operand");
        return operand; // not using returns above to avoid "not all control paths return a value"
    }
    ExpressionRef ParseExpression(int requiredPrecedence, bool stopAtNewline)
    {
        auto left = ParseOperand();                 // get first operand
        for (;;)
        {
            let & opTok = GotToken();
            let opIter = infixPrecedence.find(opTok.symbol);
            if (opIter == infixPrecedence.end())    // not an infix operator: we are done here, 'left' is our expression
                break;
            let opPrecedence = opIter->second;
            if (opPrecedence < requiredPrecedence)  // operator below required precedence level: does not belong to this sub-expression
                break;
            let op = opTok.symbol;
            ExpressionRef operation = make_shared<Expression>(opTok.beginLocation);
            operation->op = op;
            operation->args.push_back(left);        // [0] is left operand; [1] is right except for macro application
            // deal with special cases first
            // We treat member lookup (.), macro application (a()), and indexing (a[i]) together with the true infix operators.
            if (op == L".")                                 // === reference of a dictionary item
            {
                ConsumeToken();
                operation->id = ConsumeIdentifier();
            }
            else if (op == L"(")                            // === macro application
            {
                operation->args.push_back(ParseMacroArgs(false));    // [1]: all arguments
            }
            else if (op == L"[")                            // === array index
            {
                ConsumeToken();
                operation->args.push_back(ParseExpression(0, false));    // [1]: index
                ConsumePunctuation(L"]");
            }
            else                                            // === regular infix operator
            {
                ConsumeToken();
                let right = ParseExpression(opPrecedence + 1, stopAtNewline);   // get right operand, or entire multi-operand expression with higher precedence
                operation->args.push_back(right);           // [1]: right operand
            }
            left = operation;
        }
        return left;
    }
    // a macro-args expression lists position-dependent and optional parameters
    // This is used both for defining macros (LHS) and using macros (RHS).
    ExpressionRef ParseMacroArgs(bool defining)
    {
        ConsumePunctuation(L"(");
        ExpressionRef macroArgs = make_shared<Expression>(GotToken().beginLocation);
        macroArgs->op = L"()";
        for (;;)
        {
            let expr = ParseExpression(0, false);   // this could be an optional arg (var = val)
            if (defining && expr->op != L"id")      // when defining we only allow a single identifier
                Fail("argument identifier expected", expr->location);
            if (expr->op == L"id" && GotToken().symbol == L"=")
            {
                ConsumeToken();
                let valueExpr = ParseExpression(0, false);
                let res = macroArgs->namedArgs.insert(make_pair(expr->id, valueExpr));
                if (!res.second)
                    Fail(strprintf("duplicate optional argument '%ls'", expr->id.c_str()), expr->location);
            }
            else
                macroArgs->args.push_back(expr);    // [0..]: position args
            if (GotToken().symbol != L",")
                break;
            ConsumeToken();
        }
        ConsumePunctuation(L")");
        return macroArgs;
    }
    map<ExpressionRef, ExpressionRef> ParseDictMembers()
    {
        map<ExpressionRef, ExpressionRef> members;
        auto idTok = GotToken();
        while (idTok.kind == identifier)
        {
            ExpressionRef var = make_shared<Expression>(idTok.beginLocation);
            // parse
            var->op = L"id";
            var->id = ConsumeIdentifier();                          // left-hand side
            if (GotToken().symbol == L"(")                          // optionally, macro arguments
                var->args.push_back(ParseMacroArgs(true/*defining*/));
            ConsumePunctuation(L"=");
            let valueExpr = ParseExpression(0, false);              // and the right-hand side
            // insert
            let res = members.insert(make_pair(var, valueExpr));
            if (!res.second)
                Fail(strprintf("duplicate member definition '%ls'", var->id.c_str()), var->location);
            // advance
            idTok = GotToken();
            if (idTok.symbol == L";")
                idTok = GetToken();
        }
        return members;
    }
    ExpressionRef Parse()
    {
        let topDict = ParseExpression(0, true);
        if (GotToken().kind != eof)
            Fail("junk at end of source", GetCursor());
        return topDict;
    }
    static void Test()
    {
        let parserTest = L"[ do = (print:train:eval) ; x = array[1..13] (i=>1+i*print.message==13*42) ; print = new PrintAction [ message = 'Hello World' ] ]";
        ParseConfigString(parserTest)->Dump();
    }
};

// globally exported functions to execute the parser
static ExpressionRef Parse(SourceFile && sourceFile) { return Parser(move(sourceFile)).Parse(); }
ExpressionRef ParseConfigString(wstring text) { return Parse(SourceFile(L"(command line)", text)); }
ExpressionRef ParseConfigFile(wstring path) { return Parse(SourceFile(path)); }

}}}   // namespaces

using namespace Microsoft::MSR::CNTK;

int wmain(int /*argc*/, wchar_t* /*argv*/[])
{
    try
    {
        Parser::Test();
    }
    catch (const ConfigError & err)
    {
        err.PrintError();
    }
    return EXIT_SUCCESS;
}
