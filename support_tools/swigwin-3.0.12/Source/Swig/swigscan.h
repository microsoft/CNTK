/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * swigscan.h
 *
 * C/C++ scanner. 
 * ----------------------------------------------------------------------------- */

typedef struct Scanner Scanner;

extern Scanner     *NewScanner(void);
extern void         DelScanner(Scanner *);
extern void         Scanner_clear(Scanner *);
extern void         Scanner_push(Scanner *, String *);
extern void         Scanner_pushtoken(Scanner *, int, const_String_or_char_ptr value);
extern int          Scanner_token(Scanner *);
extern String      *Scanner_text(Scanner *);
extern void         Scanner_skip_line(Scanner *);
extern int          Scanner_skip_balanced(Scanner *, int startchar, int endchar);
extern String      *Scanner_get_raw_text_balanced(Scanner *, int startchar, int endchar);
extern void         Scanner_set_location(Scanner *, String *file, int line);
extern String      *Scanner_file(Scanner *);
extern int          Scanner_line(Scanner *);
extern int          Scanner_start_line(Scanner *);
extern void         Scanner_idstart(Scanner *, const char *idchar);
extern String      *Scanner_errmsg(Scanner *);
extern int          Scanner_errline(Scanner *);
extern int          Scanner_isoperator(int tokval);
extern void         Scanner_locator(Scanner *, String *loc);

/* Note: Tokens in range 100+ are for C/C++ operators */

#define   SWIG_MAXTOKENS          200
#define   SWIG_TOKEN_LPAREN       1        /* ( */
#define   SWIG_TOKEN_RPAREN       2        /* ) */
#define   SWIG_TOKEN_SEMI         3        /* ; */
#define   SWIG_TOKEN_LBRACE       4        /* { */
#define   SWIG_TOKEN_RBRACE       5        /* } */
#define   SWIG_TOKEN_LBRACKET     6        /* [ */
#define   SWIG_TOKEN_RBRACKET     7        /* ] */
#define   SWIG_TOKEN_BACKSLASH    8        /* \ */
#define   SWIG_TOKEN_ENDLINE      9        /* \n */
#define   SWIG_TOKEN_STRING       10       /* "str" */
#define   SWIG_TOKEN_POUND        11       /* # */
#define   SWIG_TOKEN_COLON        12       /* : */
#define   SWIG_TOKEN_DCOLON       13       /* :: */
#define   SWIG_TOKEN_DCOLONSTAR   14       /* ::* */
#define   SWIG_TOKEN_ID           15       /* identifer */
#define   SWIG_TOKEN_FLOAT        16       /* 3.1415F */
#define   SWIG_TOKEN_DOUBLE       17       /* 3.1415 */
#define   SWIG_TOKEN_INT          18       /* 314 */
#define   SWIG_TOKEN_UINT         19       /* 314U */
#define   SWIG_TOKEN_LONG         20       /* 314L */
#define   SWIG_TOKEN_ULONG        21       /* 314UL */
#define   SWIG_TOKEN_CHAR         22       /* 'charconst' */
#define   SWIG_TOKEN_PERIOD       23       /* . */
#define   SWIG_TOKEN_AT           24       /* @ */
#define   SWIG_TOKEN_DOLLAR       25       /* $ */
#define   SWIG_TOKEN_CODEBLOCK    26       /* %{ ... %} ... */
#define   SWIG_TOKEN_RSTRING      27       /* `charconst` */
#define   SWIG_TOKEN_LONGLONG     28       /* 314LL */
#define   SWIG_TOKEN_ULONGLONG    29       /* 314ULL */
#define   SWIG_TOKEN_QUESTION     30       /* ? */
#define   SWIG_TOKEN_COMMENT      31       /* C or C++ comment */
#define   SWIG_TOKEN_BOOL         32       /* true or false */
#define   SWIG_TOKEN_WSTRING      33       /* L"str" */
#define   SWIG_TOKEN_WCHAR        34       /* L'c' */

#define   SWIG_TOKEN_ILLEGAL      99
#define   SWIG_TOKEN_ERROR        -1

#define   SWIG_TOKEN_COMMA        101      /* , */
#define   SWIG_TOKEN_STAR         102      /* * */
#define   SWIG_TOKEN_TIMES        102      /* * */
#define   SWIG_TOKEN_EQUAL        103      /* = */
#define   SWIG_TOKEN_EQUALTO      104      /* == */
#define   SWIG_TOKEN_NOTEQUAL     105      /* != */
#define   SWIG_TOKEN_PLUS         106      /* + */
#define   SWIG_TOKEN_MINUS        107      /* - */
#define   SWIG_TOKEN_AND          108      /* & */
#define   SWIG_TOKEN_LAND         109      /* && */
#define   SWIG_TOKEN_OR           110      /* | */
#define   SWIG_TOKEN_LOR          111      /* || */
#define   SWIG_TOKEN_XOR          112      /* ^ */
#define   SWIG_TOKEN_LESSTHAN     113      /* < */
#define   SWIG_TOKEN_GREATERTHAN  114      /* > */
#define   SWIG_TOKEN_LTEQUAL      115      /* <= */
#define   SWIG_TOKEN_GTEQUAL      116      /* >= */
#define   SWIG_TOKEN_NOT          117      /* ~ */
#define   SWIG_TOKEN_LNOT         118      /* ! */
#define   SWIG_TOKEN_SLASH        119      /* / */
#define   SWIG_TOKEN_DIVIDE       119      /* / */
#define   SWIG_TOKEN_PERCENT      120      /* % */
#define   SWIG_TOKEN_MODULO       120      /* % */
#define   SWIG_TOKEN_LSHIFT       121      /* << */
#define   SWIG_TOKEN_RSHIFT       122      /* >> */
#define   SWIG_TOKEN_PLUSPLUS     123      /* ++ */
#define   SWIG_TOKEN_MINUSMINUS   124      /* -- */
#define   SWIG_TOKEN_PLUSEQUAL    125      /* += */
#define   SWIG_TOKEN_MINUSEQUAL   126      /* -= */
#define   SWIG_TOKEN_TIMESEQUAL   127      /* *= */
#define   SWIG_TOKEN_DIVEQUAL     128      /* /= */
#define   SWIG_TOKEN_ANDEQUAL     129      /* &= */
#define   SWIG_TOKEN_OREQUAL      130      /* |= */
#define   SWIG_TOKEN_XOREQUAL     131      /* ^= */
#define   SWIG_TOKEN_LSEQUAL      132      /* <<= */
#define   SWIG_TOKEN_RSEQUAL      133      /* >>= */
#define   SWIG_TOKEN_MODEQUAL     134      /* %= */
#define   SWIG_TOKEN_ARROW        135      /* -> */
#define   SWIG_TOKEN_ARROWSTAR    136      /* ->* */
