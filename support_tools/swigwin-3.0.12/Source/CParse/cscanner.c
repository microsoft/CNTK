/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * scanner.c
 *
 * SWIG tokenizer.  This file is a wrapper around the generic C scanner
 * found in Swig/scanner.c.   Extra logic is added both to accommodate the
 * bison-based grammar and certain peculiarities of C++ parsing (e.g.,
 * operator overloading, typedef resolution, etc.).  This code also splits
 * C identifiers up into keywords and SWIG directives.
 * ----------------------------------------------------------------------------- */

#include "cparse.h"
#include "parser.h"
#include <string.h>
#include <ctype.h>

/* Scanner object */
static Scanner *scan = 0;

/* Global string containing C code. Used by the parser to grab code blocks */
String *scanner_ccode = 0;

/* The main file being parsed */
static String *main_input_file = 0;

/* Error reporting/location information */
int     cparse_line = 1;
String *cparse_file = 0;
int     cparse_start_line = 0;

/* C++ mode */
int cparse_cplusplus = 0;

/* Generate C++ compatible code when wrapping C code */
int cparse_cplusplusout = 0;

/* To allow better error reporting */
String *cparse_unknown_directive = 0;

/* Private vars */
static int scan_init = 0;
static int num_brace = 0;
static int last_brace = 0;
static int last_id = 0;
static int rename_active = 0;

/* -----------------------------------------------------------------------------
 * Swig_cparse_cplusplus()
 * ----------------------------------------------------------------------------- */

void Swig_cparse_cplusplus(int v) {
  cparse_cplusplus = v;
}

/* -----------------------------------------------------------------------------
 * Swig_cparse_cplusplusout()
 * ----------------------------------------------------------------------------- */

void Swig_cparse_cplusplusout(int v) {
  cparse_cplusplusout = v;
}

/* ----------------------------------------------------------------------------
 * scanner_init()
 *
 * Initialize buffers
 * ------------------------------------------------------------------------- */

void scanner_init() {
  scan = NewScanner();
  Scanner_idstart(scan,"%");
  scan_init = 1;
  scanner_ccode = NewStringEmpty();
}

/* ----------------------------------------------------------------------------
 * scanner_file(DOHFile *f)
 *
 * Start reading from new file
 * ------------------------------------------------------------------------- */
void scanner_file(DOHFile * f) {
  if (!scan_init) scanner_init();
  Scanner_clear(scan);
  Scanner_push(scan,f);
}

/* ----------------------------------------------------------------------------
 * start_inline(char *text, int line)
 *
 * Take a chunk of text and recursively feed it back into the scanner.  Used
 * by the %inline directive.
 * ------------------------------------------------------------------------- */

void start_inline(char *text, int line) {
  String *stext = NewString(text);

  Seek(stext,0,SEEK_SET);
  Setfile(stext,cparse_file);
  Setline(stext,line);
  Scanner_push(scan,stext);
  Delete(stext);
}

/* -----------------------------------------------------------------------------
 * skip_balanced()
 *
 * Skips a piece of code enclosed in begin/end symbols such as '{...}' or
 * (...).  Ignores symbols inside comments or strings.
 * ----------------------------------------------------------------------------- */

void skip_balanced(int startchar, int endchar) {
  int start_line = Scanner_line(scan);
  Clear(scanner_ccode);

  if (Scanner_skip_balanced(scan,startchar,endchar) < 0) {
    Swig_error(cparse_file, start_line, "Missing '%c'. Reached end of input.\n", endchar);
    return;
  }

  cparse_line = Scanner_line(scan);
  cparse_file = Scanner_file(scan);

  Append(scanner_ccode, Scanner_text(scan));
  if (endchar == '}')
    num_brace--;
  return;
}

/* -----------------------------------------------------------------------------
 * get_raw_text_balanced()
 *
 * Returns raw text between 2 braces
 * ----------------------------------------------------------------------------- */

String *get_raw_text_balanced(int startchar, int endchar) {
  return Scanner_get_raw_text_balanced(scan, startchar, endchar);
}

/* ----------------------------------------------------------------------------
 * void skip_decl(void)
 *
 * This tries to skip over an entire declaration.   For example
 *
 *  friend ostream& operator<<(ostream&, const char *s);
 *
 * or
 *  friend ostream& operator<<(ostream&, const char *s) { }
 *
 * ------------------------------------------------------------------------- */

void skip_decl(void) {
  int tok;
  int done = 0;
  int start_line = Scanner_line(scan);

  while (!done) {
    tok = Scanner_token(scan);
    if (tok == 0) {
      if (!Swig_error_count()) {
	Swig_error(cparse_file, start_line, "Missing semicolon. Reached end of input.\n");
      }
      return;
    }
    if (tok == SWIG_TOKEN_LBRACE) {
      if (Scanner_skip_balanced(scan,'{','}') < 0) {
	Swig_error(cparse_file, start_line, "Missing '}'. Reached end of input.\n");
      }
      break;
    }
    if (tok == SWIG_TOKEN_SEMI) {
      done = 1;
    }
  }
  cparse_file = Scanner_file(scan);
  cparse_line = Scanner_line(scan);
}

/* ----------------------------------------------------------------------------
 * int yylook()
 *
 * Lexical scanner.
 * ------------------------------------------------------------------------- */

static int yylook(void) {

  int tok = 0;

  while (1) {
    if ((tok = Scanner_token(scan)) == 0)
      return 0;
    if (tok == SWIG_TOKEN_ERROR)
      return 0;
    cparse_start_line = Scanner_start_line(scan);
    cparse_line = Scanner_line(scan);
    cparse_file = Scanner_file(scan);

    switch(tok) {
    case SWIG_TOKEN_ID:
      return ID;
    case SWIG_TOKEN_LPAREN: 
      return LPAREN;
    case SWIG_TOKEN_RPAREN: 
      return RPAREN;
    case SWIG_TOKEN_SEMI:
      return SEMI;
    case SWIG_TOKEN_COMMA:
      return COMMA;
    case SWIG_TOKEN_STAR:
      return STAR;
    case SWIG_TOKEN_RBRACE:
      num_brace--;
      if (num_brace < 0) {
	Swig_error(cparse_file, cparse_line, "Syntax error. Extraneous '}'\n");
	num_brace = 0;
      } else {
	return RBRACE;
      }
      break;
    case SWIG_TOKEN_LBRACE:
      last_brace = num_brace;
      num_brace++;
      return LBRACE;
    case SWIG_TOKEN_EQUAL:
      return EQUAL;
    case SWIG_TOKEN_EQUALTO:
      return EQUALTO;
    case SWIG_TOKEN_PLUS:
      return PLUS;
    case SWIG_TOKEN_MINUS:
      return MINUS;
    case SWIG_TOKEN_SLASH:
      return SLASH;
    case SWIG_TOKEN_AND:
      return AND;
    case SWIG_TOKEN_LAND:
      return LAND;
    case SWIG_TOKEN_OR:
      return OR;
    case SWIG_TOKEN_LOR:
      return LOR;
    case SWIG_TOKEN_XOR:
      return XOR;
    case SWIG_TOKEN_NOT:
      return NOT;
    case SWIG_TOKEN_LNOT:
      return LNOT;
    case SWIG_TOKEN_NOTEQUAL:
      return NOTEQUALTO;
    case SWIG_TOKEN_LBRACKET:
      return LBRACKET;
    case SWIG_TOKEN_RBRACKET:
      return RBRACKET;
    case SWIG_TOKEN_QUESTION:
      return QUESTIONMARK;
    case SWIG_TOKEN_LESSTHAN:
      return LESSTHAN;
    case SWIG_TOKEN_LTEQUAL:
      return LESSTHANOREQUALTO;
    case SWIG_TOKEN_LSHIFT:
      return LSHIFT;
    case SWIG_TOKEN_GREATERTHAN:
      return GREATERTHAN;
    case SWIG_TOKEN_GTEQUAL:
      return GREATERTHANOREQUALTO;
    case SWIG_TOKEN_RSHIFT:
      return RSHIFT;
    case SWIG_TOKEN_ARROW:
      return ARROW;
    case SWIG_TOKEN_PERIOD:
      return PERIOD;
    case SWIG_TOKEN_MODULO:
      return MODULO;
    case SWIG_TOKEN_COLON:
      return COLON;
    case SWIG_TOKEN_DCOLONSTAR:
      return DSTAR;
      
    case SWIG_TOKEN_DCOLON:
      {
	int nexttok = Scanner_token(scan);
	if (nexttok == SWIG_TOKEN_STAR) {
	  return DSTAR;
	} else if (nexttok == SWIG_TOKEN_NOT) {
	  return DCNOT;
	} else {
	  Scanner_pushtoken(scan,nexttok,Scanner_text(scan));
	  if (!last_id) {
	    scanner_next_token(DCOLON);
	    return NONID;
	  } else {
	    return DCOLON;
	  }
	}
      }
      break;
      
      /* Look for multi-character sequences */
      
    case SWIG_TOKEN_RSTRING:
      yylval.type = NewString(Scanner_text(scan));
      return TYPE_RAW;
      
    case SWIG_TOKEN_STRING:
      yylval.id = Swig_copy_string(Char(Scanner_text(scan)));
      return STRING;

    case SWIG_TOKEN_WSTRING:
      yylval.id = Swig_copy_string(Char(Scanner_text(scan)));
      return WSTRING;
      
    case SWIG_TOKEN_CHAR:
      yylval.str = NewString(Scanner_text(scan));
      if (Len(yylval.str) == 0) {
	Swig_error(cparse_file, cparse_line, "Empty character constant\n");
      }
      return CHARCONST;

    case SWIG_TOKEN_WCHAR:
      yylval.str = NewString(Scanner_text(scan));
      if (Len(yylval.str) == 0) {
	Swig_error(cparse_file, cparse_line, "Empty character constant\n");
      }
      return WCHARCONST;

      /* Numbers */
      
    case SWIG_TOKEN_INT:
      return NUM_INT;
      
    case SWIG_TOKEN_UINT:
      return NUM_UNSIGNED;
      
    case SWIG_TOKEN_LONG:
      return NUM_LONG;
      
    case SWIG_TOKEN_ULONG:
      return NUM_ULONG;
      
    case SWIG_TOKEN_LONGLONG:
      return NUM_LONGLONG;
      
    case SWIG_TOKEN_ULONGLONG:
      return NUM_ULONGLONG;
      
    case SWIG_TOKEN_DOUBLE:
    case SWIG_TOKEN_FLOAT:
      return NUM_FLOAT;
      
    case SWIG_TOKEN_BOOL:
      return NUM_BOOL;
      
    case SWIG_TOKEN_POUND:
      Scanner_skip_line(scan);
      yylval.id = Swig_copy_string(Char(Scanner_text(scan)));
      return POUND;
      break;
      
    case SWIG_TOKEN_CODEBLOCK:
      yylval.str = NewString(Scanner_text(scan));
      return HBLOCK;
      
    case SWIG_TOKEN_COMMENT:
      {
	String *cmt = Scanner_text(scan);
	char *loc = Char(cmt);
	if ((strncmp(loc,"/*@SWIG",7) == 0) && (loc[Len(cmt)-3] == '@')) {
	  Scanner_locator(scan, cmt);
	}
      }
      break;
    case SWIG_TOKEN_ENDLINE:
      break;
    case SWIG_TOKEN_BACKSLASH:
      break;
    default:
      Swig_error(cparse_file, cparse_line, "Illegal token '%s'.\n", Scanner_text(scan));
      return (ILLEGAL);
    }
  }
}

static int check_typedef = 0;

void scanner_set_location(String *file, int line) {
  Scanner_set_location(scan,file,line-1);
}

void scanner_check_typedef() {
  check_typedef = 1;
}

void scanner_ignore_typedef() {
  check_typedef = 0;
}

void scanner_last_id(int x) {
  last_id = x;
}

void scanner_clear_rename() {
  rename_active = 0;
}

/* Used to push a fictitious token into the scanner */
static int next_token = 0;
void scanner_next_token(int tok) {
  next_token = tok;
}

void scanner_set_main_input_file(String *file) {
  main_input_file = file;
}

String *scanner_get_main_input_file() {
  return main_input_file;
}

/* ----------------------------------------------------------------------------
 * int yylex()
 *
 * Gets the lexene and returns tokens.
 * ------------------------------------------------------------------------- */

int yylex(void) {

  int l;
  char *yytext;

  if (!scan_init) {
    scanner_init();
  }

  if (next_token) {
    l = next_token;
    next_token = 0;
    return l;
  }

  l = yylook();

  /*   Swig_diagnostic(cparse_file, cparse_line, ":::%d: '%s'\n", l, Scanner_text(scan)); */

  if (l == NONID) {
    last_id = 1;
  } else {
    last_id = 0;
  }

  /* We got some sort of non-white space object.  We set the start_line
     variable unless it has already been set */

  if (!cparse_start_line) {
    cparse_start_line = cparse_line;
  }

  /* Copy the lexene */

  switch (l) {

  case NUM_INT:
  case NUM_FLOAT:
  case NUM_ULONG:
  case NUM_LONG:
  case NUM_UNSIGNED:
  case NUM_LONGLONG:
  case NUM_ULONGLONG:
  case NUM_BOOL:
    if (l == NUM_INT)
      yylval.dtype.type = T_INT;
    if (l == NUM_FLOAT)
      yylval.dtype.type = T_DOUBLE;
    if (l == NUM_ULONG)
      yylval.dtype.type = T_ULONG;
    if (l == NUM_LONG)
      yylval.dtype.type = T_LONG;
    if (l == NUM_UNSIGNED)
      yylval.dtype.type = T_UINT;
    if (l == NUM_LONGLONG)
      yylval.dtype.type = T_LONGLONG;
    if (l == NUM_ULONGLONG)
      yylval.dtype.type = T_ULONGLONG;
    if (l == NUM_BOOL)
      yylval.dtype.type = T_BOOL;
    yylval.dtype.val = NewString(Scanner_text(scan));
    yylval.dtype.bitfield = 0;
    yylval.dtype.throws = 0;
    return (l);

  case ID:
    yytext = Char(Scanner_text(scan));
    if (yytext[0] != '%') {
      /* Look for keywords now */

      if (strcmp(yytext, "int") == 0) {
	yylval.type = NewSwigType(T_INT);
	return (TYPE_INT);
      }
      if (strcmp(yytext, "double") == 0) {
	yylval.type = NewSwigType(T_DOUBLE);
	return (TYPE_DOUBLE);
      }
      if (strcmp(yytext, "void") == 0) {
	yylval.type = NewSwigType(T_VOID);
	return (TYPE_VOID);
      }
      if (strcmp(yytext, "char") == 0) {
	yylval.type = NewSwigType(T_CHAR);
	return (TYPE_CHAR);
      }
      if (strcmp(yytext, "wchar_t") == 0) {
	yylval.type = NewSwigType(T_WCHAR);
	return (TYPE_WCHAR);
      }
      if (strcmp(yytext, "short") == 0) {
	yylval.type = NewSwigType(T_SHORT);
	return (TYPE_SHORT);
      }
      if (strcmp(yytext, "long") == 0) {
	yylval.type = NewSwigType(T_LONG);
	return (TYPE_LONG);
      }
      if (strcmp(yytext, "float") == 0) {
	yylval.type = NewSwigType(T_FLOAT);
	return (TYPE_FLOAT);
      }
      if (strcmp(yytext, "signed") == 0) {
	yylval.type = NewSwigType(T_INT);
	return (TYPE_SIGNED);
      }
      if (strcmp(yytext, "unsigned") == 0) {
	yylval.type = NewSwigType(T_UINT);
	return (TYPE_UNSIGNED);
      }
      if (strcmp(yytext, "bool") == 0) {
	yylval.type = NewSwigType(T_BOOL);
	return (TYPE_BOOL);
      }

      /* Non ISO (Windows) C extensions */
      if (strcmp(yytext, "__int8") == 0) {
	yylval.type = NewString(yytext);
	return (TYPE_NON_ISO_INT8);
      }
      if (strcmp(yytext, "__int16") == 0) {
	yylval.type = NewString(yytext);
	return (TYPE_NON_ISO_INT16);
      }
      if (strcmp(yytext, "__int32") == 0) {
	yylval.type = NewString(yytext);
	return (TYPE_NON_ISO_INT32);
      }
      if (strcmp(yytext, "__int64") == 0) {
	yylval.type = NewString(yytext);
	return (TYPE_NON_ISO_INT64);
      }

      /* C++ keywords */
      if (cparse_cplusplus) {
	if (strcmp(yytext, "and") == 0)
	  return (LAND);
	if (strcmp(yytext, "or") == 0)
	  return (LOR);
	if (strcmp(yytext, "not") == 0)
	  return (LNOT);
	if (strcmp(yytext, "class") == 0)
	  return (CLASS);
	if (strcmp(yytext, "private") == 0)
	  return (PRIVATE);
	if (strcmp(yytext, "public") == 0)
	  return (PUBLIC);
	if (strcmp(yytext, "protected") == 0)
	  return (PROTECTED);
	if (strcmp(yytext, "friend") == 0)
	  return (FRIEND);
	if (strcmp(yytext, "constexpr") == 0)
	  return (CONSTEXPR);
	if (strcmp(yytext, "thread_local") == 0)
	  return (THREAD_LOCAL);
	if (strcmp(yytext, "decltype") == 0)
	  return (DECLTYPE);
	if (strcmp(yytext, "virtual") == 0)
	  return (VIRTUAL);
	if (strcmp(yytext, "static_assert") == 0)
	  return (STATIC_ASSERT);
	if (strcmp(yytext, "operator") == 0) {
	  int nexttok;
	  String *s = NewString("operator ");

	  /* If we have an operator, we have to collect the operator symbol and attach it to
             the operator identifier.   To do this, we need to scan ahead by several tokens.
             Cases include:

             (1) If the next token is an operator as determined by Scanner_isoperator(),
                 it means that the operator applies to one of the standard C++ mathematical,
                 assignment, or logical operator symbols (e.g., '+','<=','==','&', etc.)
                 In this case, we merely append the symbol text to the operator string above.

             (2) If the next token is (, we look for ).  This is operator ().
             (3) If the next token is [, we look for ].  This is operator [].
	     (4) If the next token is an identifier.  The operator is possibly a conversion operator.
                      (a) Must check for special case new[] and delete[]

             Error handling is somewhat tricky here.  We'll try to back out gracefully if we can.
 
	  */

	  do {
	    nexttok = Scanner_token(scan);
	  } while (nexttok == SWIG_TOKEN_ENDLINE || nexttok == SWIG_TOKEN_COMMENT);

	  if (Scanner_isoperator(nexttok)) {
	    /* One of the standard C/C++ symbolic operators */
	    Append(s,Scanner_text(scan));
	    yylval.str = s;
	    return OPERATOR;
	  } else if (nexttok == SWIG_TOKEN_LPAREN) {
	    /* Function call operator.  The next token MUST be a RPAREN */
	    nexttok = Scanner_token(scan);
	    if (nexttok != SWIG_TOKEN_RPAREN) {
	      Swig_error(Scanner_file(scan),Scanner_line(scan),"Syntax error. Bad operator name.\n");
	    } else {
	      Append(s,"()");
	      yylval.str = s;
	      return OPERATOR;
	    }
	  } else if (nexttok == SWIG_TOKEN_LBRACKET) {
	    /* Array access operator.  The next token MUST be a RBRACKET */
	    nexttok = Scanner_token(scan);
	    if (nexttok != SWIG_TOKEN_RBRACKET) {
	      Swig_error(Scanner_file(scan),Scanner_line(scan),"Syntax error. Bad operator name.\n");	      
	    } else {
	      Append(s,"[]");
	      yylval.str = s;
	      return OPERATOR;
	    }
	  } else if (nexttok == SWIG_TOKEN_STRING) {
	    /* Operator "" or user-defined string literal ""_suffix */
	    Append(s,"\"\"");
	    yylval.str = s;
	    return OPERATOR;
	  } else if (nexttok == SWIG_TOKEN_ID) {
	    /* We have an identifier.  This could be any number of things. It could be a named version of
               an operator (e.g., 'and_eq') or it could be a conversion operator.   To deal with this, we're
               going to read tokens until we encounter a ( or ;.  Some care is needed for formatting. */
	    int needspace = 1;
	    int termtoken = 0;
	    const char *termvalue = 0;

	    Append(s,Scanner_text(scan));
	    while (1) {

	      nexttok = Scanner_token(scan);
	      if (nexttok <= 0) {
		Swig_error(Scanner_file(scan),Scanner_line(scan),"Syntax error. Bad operator name.\n");	      
	      }
	      if (nexttok == SWIG_TOKEN_LPAREN) {
		termtoken = SWIG_TOKEN_LPAREN;
		termvalue = "(";
		break;
              } else if (nexttok == SWIG_TOKEN_CODEBLOCK) {
                termtoken = SWIG_TOKEN_CODEBLOCK;
                termvalue = Char(Scanner_text(scan));
                break;
              } else if (nexttok == SWIG_TOKEN_LBRACE) {
                termtoken = SWIG_TOKEN_LBRACE;
                termvalue = "{";
                break;
              } else if (nexttok == SWIG_TOKEN_SEMI) {
		termtoken = SWIG_TOKEN_SEMI;
		termvalue = ";";
		break;
              } else if (nexttok == SWIG_TOKEN_STRING) {
		termtoken = SWIG_TOKEN_STRING;
                termvalue = Swig_copy_string(Char(Scanner_text(scan)));
		break;
	      } else if (nexttok == SWIG_TOKEN_ID) {
		if (needspace) {
		  Append(s," ");
		}
		Append(s,Scanner_text(scan));
	      } else if (nexttok == SWIG_TOKEN_ENDLINE) {
	      } else if (nexttok == SWIG_TOKEN_COMMENT) {
	      } else {
		Append(s,Scanner_text(scan));
		needspace = 0;
	      }
	    }
	    yylval.str = s;
	    if (!rename_active) {
	      String *cs;
	      char *t = Char(s) + 9;
	      if (!((strcmp(t, "new") == 0)
		    || (strcmp(t, "delete") == 0)
		    || (strcmp(t, "new[]") == 0)
		    || (strcmp(t, "delete[]") == 0)
		    || (strcmp(t, "and") == 0)
		    || (strcmp(t, "and_eq") == 0)
		    || (strcmp(t, "bitand") == 0)
		    || (strcmp(t, "bitor") == 0)
		    || (strcmp(t, "compl") == 0)
		    || (strcmp(t, "not") == 0)
		    || (strcmp(t, "not_eq") == 0)
		    || (strcmp(t, "or") == 0)
		    || (strcmp(t, "or_eq") == 0)
		    || (strcmp(t, "xor") == 0)
		    || (strcmp(t, "xor_eq") == 0)
		    )) {
		/*              retract(strlen(t)); */

		/* The operator is a conversion operator.   In order to deal with this, we need to feed the
                   type information back into the parser.  For now this is a hack.  Needs to be cleaned up later. */
		cs = NewString(t);
		if (termtoken) Append(cs,termvalue);
		Seek(cs,0,SEEK_SET);
		Setline(cs,cparse_line);
		Setfile(cs,cparse_file);
		Scanner_push(scan,cs);
		Delete(cs);
		return CONVERSIONOPERATOR;
	      }
	    }
	    if (termtoken)
              Scanner_pushtoken(scan, termtoken, termvalue);
	    return (OPERATOR);
	  }
	}
	if (strcmp(yytext, "throw") == 0)
	  return (THROW);
	if (strcmp(yytext, "noexcept") == 0)
	  return (NOEXCEPT);
	if (strcmp(yytext, "try") == 0)
	  return (yylex());
	if (strcmp(yytext, "catch") == 0)
	  return (CATCH);
	if (strcmp(yytext, "inline") == 0)
	  return (yylex());
	if (strcmp(yytext, "mutable") == 0)
	  return (yylex());
	if (strcmp(yytext, "explicit") == 0)
	  return (EXPLICIT);
	if (strcmp(yytext, "auto") == 0)
	  return (AUTO);
	if (strcmp(yytext, "export") == 0)
	  return (yylex());
	if (strcmp(yytext, "typename") == 0)
	  return (TYPENAME);
	if (strcmp(yytext, "template") == 0) {
	  yylval.intvalue = cparse_line;
	  return (TEMPLATE);
	}
	if (strcmp(yytext, "delete") == 0)
	  return (DELETE_KW);
	if (strcmp(yytext, "default") == 0)
	  return (DEFAULT);
	if (strcmp(yytext, "using") == 0)
	  return (USING);
	if (strcmp(yytext, "namespace") == 0)
	  return (NAMESPACE);
	if (strcmp(yytext, "override") == 0)
	  return (OVERRIDE);
	if (strcmp(yytext, "final") == 0)
	  return (FINAL);
      } else {
	if (strcmp(yytext, "class") == 0) {
	  Swig_warning(WARN_PARSE_CLASS_KEYWORD, cparse_file, cparse_line, "class keyword used, but not in C++ mode.\n");
	}
	if (strcmp(yytext, "complex") == 0) {
	  yylval.type = NewSwigType(T_COMPLEX);
	  return (TYPE_COMPLEX);
	}
	if (strcmp(yytext, "restrict") == 0)
	  return (yylex());
      }

      /* Misc keywords */

      if (strcmp(yytext, "extern") == 0)
	return (EXTERN);
      if (strcmp(yytext, "const") == 0)
	return (CONST_QUAL);
      if (strcmp(yytext, "static") == 0)
	return (STATIC);
      if (strcmp(yytext, "struct") == 0)
	return (STRUCT);
      if (strcmp(yytext, "union") == 0)
	return (UNION);
      if (strcmp(yytext, "enum") == 0)
	return (ENUM);
      if (strcmp(yytext, "sizeof") == 0)
	return (SIZEOF);

      if (strcmp(yytext, "typedef") == 0) {
	yylval.intvalue = 0;
	return (TYPEDEF);
      }

      /* Ignored keywords */

      if (strcmp(yytext, "volatile") == 0)
	return (VOLATILE);
      if (strcmp(yytext, "register") == 0)
	return (REGISTER);
      if (strcmp(yytext, "inline") == 0)
	return (yylex());

    } else {
      Delete(cparse_unknown_directive);
      cparse_unknown_directive = NULL;

      /* SWIG directives */
      if (strcmp(yytext, "%module") == 0)
	return (MODULE);
      if (strcmp(yytext, "%insert") == 0)
	return (INSERT);
      if (strcmp(yytext, "%name") == 0)
	return (NAME);
      if (strcmp(yytext, "%rename") == 0) {
	rename_active = 1;
	return (RENAME);
      }
      if (strcmp(yytext, "%namewarn") == 0) {
	rename_active = 1;
	return (NAMEWARN);
      }
      if (strcmp(yytext, "%includefile") == 0)
	return (INCLUDE);
      if (strcmp(yytext, "%beginfile") == 0)
	return (BEGINFILE);
      if (strcmp(yytext, "%endoffile") == 0)
	return (ENDOFFILE);
      if (strcmp(yytext, "%val") == 0) {
	Swig_warning(WARN_DEPRECATED_VAL, cparse_file, cparse_line, "%%val directive deprecated (ignored).\n");
	return (yylex());
      }
      if (strcmp(yytext, "%out") == 0) {
	Swig_warning(WARN_DEPRECATED_OUT, cparse_file, cparse_line, "%%out directive deprecated (ignored).\n");
	return (yylex());
      }
      if (strcmp(yytext, "%constant") == 0)
	return (CONSTANT);
      if (strcmp(yytext, "%typedef") == 0) {
	yylval.intvalue = 1;
	return (TYPEDEF);
      }
      if (strcmp(yytext, "%native") == 0)
	return (NATIVE);
      if (strcmp(yytext, "%pragma") == 0)
	return (PRAGMA);
      if (strcmp(yytext, "%extend") == 0)
	return (EXTEND);
      if (strcmp(yytext, "%fragment") == 0)
	return (FRAGMENT);
      if (strcmp(yytext, "%inline") == 0)
	return (INLINE);
      if (strcmp(yytext, "%typemap") == 0)
	return (TYPEMAP);
      if (strcmp(yytext, "%feature") == 0) {
        /* The rename_active indicates we don't need the information of the 
         * following function's return type. This applied for %rename, so do
         * %feature. 
         */
        rename_active = 1;
	return (FEATURE);
      }
      if (strcmp(yytext, "%except") == 0)
	return (EXCEPT);
      if (strcmp(yytext, "%importfile") == 0)
	return (IMPORT);
      if (strcmp(yytext, "%echo") == 0)
	return (ECHO);
      if (strcmp(yytext, "%apply") == 0)
	return (APPLY);
      if (strcmp(yytext, "%clear") == 0)
	return (CLEAR);
      if (strcmp(yytext, "%types") == 0)
	return (TYPES);
      if (strcmp(yytext, "%parms") == 0)
	return (PARMS);
      if (strcmp(yytext, "%varargs") == 0)
	return (VARARGS);
      if (strcmp(yytext, "%template") == 0) {
	return (SWIGTEMPLATE);
      }
      if (strcmp(yytext, "%warn") == 0)
	return (WARN);

      /* Note down the apparently unknown directive for error reporting. */
      cparse_unknown_directive = NewString(yytext);
    }
    /* Have an unknown identifier, as a last step, we'll do a typedef lookup on it. */

    /* Need to fix this */
    if (check_typedef) {
      if (SwigType_istypedef(yytext)) {
	yylval.type = NewString(yytext);
	return (TYPE_TYPEDEF);
      }
    }
    yylval.id = Swig_copy_string(yytext);
    last_id = 1;
    return (ID);
  case POUND:
    return yylex();
  default:
    return (l);
  }
}
