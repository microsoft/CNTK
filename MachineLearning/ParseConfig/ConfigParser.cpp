// ConfigParser.cpp -- config parser (syntactic only, that is, source -> Expression tree)

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ConfigParser.h"
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

// ---------------------------------------------------------------------------
// source files and text references (location) into them
// ---------------------------------------------------------------------------

// SourceFile constructors
SourceFile::SourceFile(wstring location, wstring text) : path(location), lines(split(text, L"\r\n")) { }  // from string, e.g. command line
SourceFile::SourceFile(wstring path) : path(path)       // from file
{
    File(path, fileOptionsRead).GetLines(lines);
}

// default constructor constructs an unmissably invalid object
TextLocation::TextLocation() : lineNo(SIZE_MAX), charPos(SIZE_MAX), sourceFileAsIndex(SIZE_MAX) { }

// register a new source file and return a TextPosition that points to its start
/*static*/ TextLocation TextLocation::NewSourceFile(SourceFile && sourceFile)
{
    TextLocation loc;
    loc.lineNo = 0;
    loc.charPos = 0;
    loc.sourceFileAsIndex = sourceFileMap.size();   // index under which we store the source file
    sourceFileMap.push_back(move(sourceFile));      // take ownership of the source file and give it a numeric index
    return loc;
}

// helper for pretty-printing errors: Show source-code line with ...^ under it to mark up the point of error
wstring TextLocation::FormatErroneousLine() const
{
    const auto & lines = GetSourceFile().lines;
    const auto line = (lineNo == lines.size()) ? L"(end)" : lines[lineNo].c_str();
    return wstring(line) + L"\n" + wstring(charPos, L'.') + L"^";
}

void TextLocation::PrintIssue(const wchar_t * errorKind, const wchar_t * kind, const wchar_t * what) const
{
    fprintf(stderr, "%ls(%d): %ls %ls: %ls\n%ls\n", GetSourceFile().path.c_str(), lineNo + 1/*report 1-based*/, errorKind, kind, what, FormatErroneousLine().c_str());
}
/*static*/ vector<SourceFile> TextLocation::sourceFileMap;

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
        CodeSourceError(const wstring & msg, TextLocation where) : ConfigError(msg, where) { }
        /*implement*/ const wchar_t * kind() const { return L"reading source"; }
    };

    void Fail(wstring msg, TextLocation where) { throw CodeSourceError(msg, where); }

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
    const wchar_t * GotCharPtr() const { return currentLine + cursor.charPos; }

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
            L"=", L";", L",", L"\n",
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
        bool isLineInitial;         // this token is the first on the line (ignoring comments)
        Token(TextLocation loc) : beginLocation(loc), kind(invalid), number(0.0), isLineInitial(false) { }
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

    class LexerError : public ConfigError
    {
    public:
        LexerError(const wstring & msg, TextLocation where) : ConfigError(msg, where) { }
        /*implement*/ const wchar_t * kind() const { return L"tokenizing"; }
    };

private:
    void Fail(wstring msg, Token where) { throw LexerError(msg, where.beginLocation); }

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
                PopSourceFile();
                t = NextToken();            // tail call--the current 't' gets dropped/ignored
                t.isLineInitial = true;     // eof is a line end
                return t;
            }
            // really end of all source code: we are done. If calling this function multiple times, we will keep returning this.
            t.kind = eof;
        }
        else if (iswdigit(ch) || (ch == L'.' && iswdigit(GotCharPtr()[1])))  // --- number
        {
            let beginPtr = GotCharPtr();
            wchar_t * endPtr = nullptr;
            t.number = wcstod(beginPtr, &endPtr);   // BUGBUG: this seems to honor locale settings. We need one that doesn't. With this, CNTK won't parse right in Germany.
            if (endPtr == beginPtr) Fail(L"parsing number", t);  // should not really happen!
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
                if (nameTok.kind != stringliteral) Fail(L"'include' must be followed by a quoted string", nameTok);
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
                Fail(L"string without closing quotation mark", t);
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
                while (ch != 0 && !(ch == L'*' && GetChar() == L'/'))   // note: this test leverages short-circuit evaluation semantics of C
                    ch = GetChar();
                if (ch == 0)
                    Fail(L"comment without closing */", t);
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
        Fail(L"error test", GetCursor());
    }
};

// ---------------------------------------------------------------------------
// parser -- parses configurations
// ---------------------------------------------------------------------------

// diagnostics helper: print the content
void Expression::Dump(int indent) const
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
            arg->Dump(indent + 2);
    }
    if (!namedArgs.empty())
    {
        fprintf(stderr, "\n");
        for (const auto & arg : namedArgs)
        {
            fprintf(stderr, "%*s%ls =\n", indent + 2, "", arg.first.c_str());
            arg.second.second->Dump(indent + 4);
        }
    }
    fprintf(stderr, "\n");
}

class Parser : public Lexer
{
    // errors
    class ParseError : public ConfigError
    {
    public:
        ParseError(const wstring & msg, TextLocation where) : ConfigError(msg, where) { }
        /*implement*/ const wchar_t * kind() const { return L"parsing"; }
    };

    void Fail(const wstring & msg, Token where) { throw ParseError(msg, where.beginLocation); }

    //void Expected(const wstring & what) { Fail(strprintf("%ls expected", what.c_str()), GotToken().beginLocation); }  // I don't know why this does not work
    void Expected(const wstring & what) { Fail(what + L" expected", GotToken().beginLocation); }

    // this token must be punctuation 's'; check and get the next
    void ConsumePunctuation(const wchar_t * s)
    {
        let & tok = GotToken();
        if (tok.kind != punctuation || tok.symbol != s)
            Expected(L"'" + wstring(s) + L"'");
        ConsumeToken();
    }

    // this token must be keyword 's'; check and get the next
    void ConsumeKeyword(const wchar_t * s)
    {
        let & tok = GotToken();
        if (tok.kind != keyword || tok.symbol != s)
            Expected(L"'" + wstring(s) + L"'");
        ConsumeToken();
    }

    // this token must be an identifier; check and get the next token. Return the identifier.
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
            { L".", 100 }, { L"[", 100 }, { L"(", 100 },     // also sort-of infix operands...
            { L"*", 10 }, { L"/", 10 }, { L".*", 10 }, { L"**", 10 }, { L"%", 10 },
            { L"+", 9 }, { L"-", 9 },
            { L"==", 8 }, { L"!=", 8 }, { L"<", 8 }, { L"<=", 8 }, { L">", 8 }, { L">=", 8 },
            { L"&&", 7 },
            { L"||", 6 },
            { L":", 5 },
            { L"=>", 0 },
        };
        SetSourceFile(move(sourceFile));
        ConsumeToken();     // get the very first token
    }
    ExpressionPtr OperandFromTokenSymbol(const Token & tok)   // helper to make an Operand expression with op==tok.symbol and then consume it
    {
        auto operand = make_shared<Expression>(tok.beginLocation, tok.symbol);
        ConsumeToken();
        return operand;
    }
    ExpressionPtr ParseOperand()
    {
        let & tok = GotToken();
        ExpressionPtr operand;
        if (tok.kind == numberliteral)                                  // === numeral literal
        {
            operand = make_shared<Expression>(tok.beginLocation, L"d", tok.number, wstring(), false);
            ConsumeToken();
        }
        else if (tok.kind == stringliteral)                             // === string literal
        {
            operand = make_shared<Expression>(tok.beginLocation, L"s", 0.0, tok.symbol, false);
            ConsumeToken();
        }
        else if (tok.symbol == L"true" || tok.symbol == L"false")       // === boolean literal
        {
            operand = make_shared<Expression>(tok.beginLocation, L"b", 0.0, wstring(), (tok.symbol == L"true"));
            ConsumeToken();
        }
        else if (tok.kind == identifier)                                // === dict member (unqualified)
        {
            operand = make_shared<Expression>(tok.beginLocation, L"id");
            operand->id = ConsumeIdentifier();
        }
        else if (tok.symbol == L"+" || tok.symbol == L"-"               // === unary operators
            || tok.symbol == L"!")
        {
            // BUGBUG: fails for -F(x); it parses it as (-F)(x) which fails
            operand = make_shared<Expression>(tok.beginLocation, tok.symbol + L"(");    // encoded as +( -( !(
            ConsumeToken();
            operand->args.push_back(ParseOperand());
        }
        else if (tok.symbol == L"new")                                  // === new class instance
        {
            operand = OperandFromTokenSymbol(tok);
            if (GotToken().symbol == L"!")                              // new! class [ ] will initialize the class delayed (this is specifically used for the Delay node to break circular references)
            {
                operand->op = L"new!";
                ConsumeToken();
            }
            operand->id = ConsumeIdentifier();
            operand->args.push_back(ParseOperand());
        }
        else if (tok.symbol == L"if")                                   // === conditional expression
        {
            operand = OperandFromTokenSymbol(tok);
            operand->args.push_back(ParseExpression(0, false));         // [0] condition
            ConsumeKeyword(L"then");
            operand->args.push_back(ParseExpression(0, false));         // [1] then expression
            ConsumeKeyword(L"else");
            operand->args.push_back(ParseExpression(0, false));         // [2] else expression
        }
        else if (tok.symbol == L"(")                                    // === nested parentheses
        {
            ConsumeToken();
            operand = ParseExpression(0, false/*go across newlines*/);  // just return the content of the parens (they do not become part of the expression tree)
            ConsumePunctuation(L")");
        }
        else if (tok.symbol == L"[")                                    // === dictionary constructor
        {
            operand = make_shared<Expression>(tok.beginLocation, L"[]");
            ConsumeToken();
            operand->namedArgs = ParseDictMembers();
            ConsumePunctuation(L"]");
        }
        else if (tok.symbol == L"array")                                // === array constructor
        {
            operand = OperandFromTokenSymbol(tok);
            ConsumePunctuation(L"[");
            operand->args.push_back(ParseExpression(0, false));         // [0] first index
            ConsumePunctuation(L"..");
            operand->args.push_back(ParseExpression(0, false));         // [1] last index
            ConsumePunctuation(L"]");
            ConsumePunctuation(L"(");
            operand->args.push_back(ParseExpression(0, false));         // [2] one-argument lambda to initialize
            ConsumePunctuation(L")");
        }
        else
            Expected(L"operand");
        return operand; // not using returns above to avoid "not all control paths return a value"
    }
    ExpressionPtr ParseExpression(int requiredPrecedence, bool stopAtNewline)
    {
        auto left = ParseOperand();                 // get first operand
        for (;;)
        {
            let & opTok = GotToken();
            if (stopAtNewline && opTok.isLineInitial)
                break;
            let opIter = infixPrecedence.find(opTok.symbol);
            if (opIter == infixPrecedence.end())    // not an infix operator: we are done here, 'left' is our expression
                break;
            let opPrecedence = opIter->second;
            if (opPrecedence < requiredPrecedence)  // operator below required precedence level: does not belong to this sub-expression
                break;
            let op = opTok.symbol;
            auto operation = make_shared<Expression>(opTok.beginLocation, op, left);    // [0] is left operand; we will add [1] except for macro application
            // deal with special cases first
            // We treat member lookup (.), macro application (a()), and indexing (a[i]) together with the true infix operators.
            if (op == L".")                                 // === reference of a dictionary item
            {
                ConsumeToken();
                operation->location = GotToken().beginLocation; // location of the identifier after the .
                operation->id = ConsumeIdentifier();
            }
            else if (op == L"=>")
            {
                if (left->op != L"id")      // currently only allow for a single argument
                    Expected(L"identifier");
                ConsumeToken();
                let macroArgs = make_shared<Expression>(left->location, L"()", left); // wrap identifier in a '()' macro-args expression
                // TODO: test parsing of i => j => i*j
                let body = ParseExpression(opPrecedence, stopAtNewline);   // pass same precedence; this makes '=>' right-associative  e.g.i=>j=>i*j
                operation->args[0] = macroArgs;             // [0]: parameter list
                operation->args.push_back(body);            // [1]: right operand
            }
            else if (op == L"(")                            // === macro application
            {
                // op = "("   means 'apply'
                // args[0] = lambda expression (lambda: op="=>", args[0] = param list, args[1] = expression with unbound vars)
                // args[1] = arguments    (arguments: op="(), args=vector of expressions, one per arg; and namedArgs)
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
        for (;;)
        {
            let expr = ParseExpression(0, false);   // this could be an optional arg (var = val)
            if (defining && expr->op != L"id")      // when defining we only allow a single identifier
                Fail(L"argument identifier expected", expr->location);
            if (expr->op == L"id" && GotToken().symbol == L"=")
            {
                let id = expr->id;                  // 'expr' gets resolved (to 'id') and forgotten
                ConsumeToken();
                let defValueExpr = ParseExpression(0, false);  // default value
                let res = macroArgs->namedArgs.insert(make_pair(id, make_pair(expr->location, defValueExpr)));
                if (!res.second)
                    Fail(L"duplicate optional parameter '" + id + L"'", expr->location);
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
    map<wstring, pair<TextLocation,ExpressionPtr>> ParseDictMembers()
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
        map<wstring, pair<TextLocation,ExpressionPtr>> members;
        auto idTok = GotToken();
        while (idTok.kind == identifier)
        {
            let location = idTok.beginLocation; // for error message
            let id = ConsumeIdentifier();       // the member's name    --TODO: do we need to keep its location?
            let parameters = (GotToken().symbol == L"(") ? ParseMacroArgs(true/*defining*/) : ExpressionPtr();  // optionally, macro arguments
            ConsumePunctuation(L"=");
            let rhs = ParseExpression(0, true/*can end at newline*/);   // and the right-hand side
            let val = parameters ? make_shared<Expression>(parameters->location, L"=>", parameters, rhs) : rhs;  // rewrite to lambda if it's a macro
            // insert
            let res = members.insert(make_pair(id, make_pair(location, val)));
            if (!res.second)
                Fail(L"duplicate member definition '" + id + L"'", location);
            // advance
            idTok = GotToken();
            if (idTok.symbol == L";")
                idTok = GetToken();
        }
        return members;
    }
    // set the parent pointer in the entire tree (we don't need them inside here, so this is a final step)
    void SetParents(ExpressionPtr us, ExpressionPtr parent)
    {
        us->parent = parent;                // this is our parent
        for (auto & child : us->args)       // now tell our children about ourselves
            SetParents(child, us);
        for (auto & child : us->namedArgs)
            SetParents(child.second.second, us);
    }
    // top-level parse function parses dictonary members
    ExpressionPtr Parse()
    {
        let topMembers = ParseDictMembers();
        if (GotToken().kind != eof)
            Fail(L"junk at end of source", GetCursor());
        ExpressionPtr topDict = make_shared<Expression>(GetCursor(), L"[]");
        topDict->namedArgs = topMembers;
        SetParents(topDict, nullptr);    // set all parent pointer
        return topDict;
    }
    // simple test function for use during development
    static void Test()
    {
        let parserTest = L"a=1\na1_=13;b=2 // cmt\ndo = (print\n:train:eval) ; x = array[1..13] (i=>1+i*print.message==13*42) ; print = new PrintAction [ message = 'Hello World' ]";
        ParseConfigString(parserTest)->Dump();
    }
};

// globally exported functions to execute the parser
static ExpressionPtr Parse(SourceFile && sourceFile) { return Parser(move(sourceFile)).Parse(); }
ExpressionPtr ParseConfigString(wstring text) { return Parse(SourceFile(L"(command line)", text)); }
ExpressionPtr ParseConfigFile(wstring path) { return Parse(SourceFile(path)); }

}}}     // namespaces
