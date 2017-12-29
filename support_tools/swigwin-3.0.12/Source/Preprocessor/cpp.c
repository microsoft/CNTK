/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * cpp.c
 *
 * An implementation of a C preprocessor plus some support for additional
 * SWIG directives.
 *
 * - SWIG directives such as %include, %extern, and %import are handled
 * - A new macro %define ... %enddef can be used for multiline macros
 * - No preprocessing is performed in %{ ... %} blocks
 * - Lines beginning with %# are stripped down to #... and passed through.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "preprocessor.h"
#include <ctype.h>

static Hash *cpp = 0;		/* C preprocessor data */
static int include_all = 0;	/* Follow all includes */
static int ignore_missing = 0;
static int import_all = 0;	/* Follow all includes, but as %import statements */
static int imported_depth = 0;	/* Depth of %imported files */
static int single_include = 1;	/* Only include each file once */
static Hash *included_files = 0;
static List *dependencies = 0;
static Scanner *id_scan = 0;
static int error_as_warning = 0;	/* Understand the cpp #error directive as a special #warning */
static int expand_defined_operator = 0;
static int macro_level = 0;
static int macro_start_line = 0;
static const String * macro_start_file = 0;

/* Test a character to see if it starts an identifier */
#define isidentifier(c) ((isalpha(c)) || (c == '_') || (c == '$'))

/* Test a character to see if it valid in an identifier (after the first letter) */
#define isidchar(c) ((isalnum(c)) || (c == '_') || (c == '$'))

static DOH *Preprocessor_replace(DOH *);

/* Skip whitespace */
static void skip_whitespace(String *s, String *out) {
  int c;
  while ((c = Getc(s)) != EOF) {
    if (!isspace(c)) {
      Ungetc(c, s);
      break;
    } else if (out)
      Putc(c, out);
  }
}

/* Skip to a specified character taking line breaks into account */
static int skip_tochar(String *s, int ch, String *out) {
  int c;
  while ((c = Getc(s)) != EOF) {
    if (out)
      Putc(c, out);
    if (c == ch)
      break;
    if (c == '\\') {
      c = Getc(s);
      if ((c != EOF) && (out))
	Putc(c, out);
    }
  }
  if (c == EOF)
    return -1;
  return 0;
}

static void copy_location(const DOH *s1, DOH *s2) {
  Setfile(s2, Getfile((DOH *) s1));
  Setline(s2, Getline((DOH *) s1));
}

static String *cpp_include(const_String_or_char_ptr fn, int sysfile) {
  String *s = sysfile ? Swig_include_sys(fn) : Swig_include(fn);
  if (s && single_include) {
    String *file = Getfile(s);
    if (Getattr(included_files, file)) {
      Delete(s);
      return 0;
    }
    Setattr(included_files, file, file);
  }
  if (!s) {
    if (ignore_missing) {
      Swig_warning(WARN_PP_MISSING_FILE, Getfile(fn), Getline(fn), "Unable to find '%s'\n", fn);
    } else {
      Swig_error(Getfile(fn), Getline(fn), "Unable to find '%s'\n", fn);
    }
  } else {
    String *lf;
    Seek(s, 0, SEEK_SET);
    if (!dependencies) {
      dependencies = NewList();
    }
    lf = Copy(Swig_last_file());
    Append(dependencies, lf);
    Delete(lf);
  }
  return s;
}

List *Preprocessor_depend(void) {
  return dependencies;
}

/* -----------------------------------------------------------------------------
 * void Preprocessor_cpp_init() - Initialize the preprocessor
 * ----------------------------------------------------------------------------- */
static String *kpp_args = 0;
static String *kpp_define = 0;
static String *kpp_defined = 0;
static String *kpp_elif = 0;
static String *kpp_else = 0;
static String *kpp_endif = 0;
static String *kpp_expanded = 0;
static String *kpp_if = 0;
static String *kpp_ifdef = 0;
static String *kpp_ifndef = 0;
static String *kpp_name = 0;
static String *kpp_swigmacro = 0;
static String *kpp_symbols = 0;
static String *kpp_undef = 0;
static String *kpp_value = 0;
static String *kpp_varargs = 0;
static String *kpp_error = 0;
static String *kpp_warning = 0;
static String *kpp_line = 0;
static String *kpp_include = 0;
static String *kpp_pragma = 0;
static String *kpp_level = 0;

static String *kpp_dline = 0;
static String *kpp_ddefine = 0;
static String *kpp_dinclude = 0;
static String *kpp_dimport = 0;
static String *kpp_dbeginfile = 0;
static String *kpp_dextern = 0;

static String *kpp_LINE = 0;
static String *kpp_FILE = 0;

static String *kpp_hash_if = 0;
static String *kpp_hash_elif = 0;

void Preprocessor_init(void) {
  Hash *s;

  kpp_args = NewString("args");
  kpp_define = NewString("define");
  kpp_defined = NewString("defined");
  kpp_else = NewString("else");
  kpp_elif = NewString("elif");
  kpp_endif = NewString("endif");
  kpp_expanded = NewString("*expanded*");
  kpp_if = NewString("if");
  kpp_ifdef = NewString("ifdef");
  kpp_ifndef = NewString("ifndef");
  kpp_name = NewString("name");
  kpp_swigmacro = NewString("swigmacro");
  kpp_symbols = NewString("symbols");
  kpp_undef = NewString("undef");
  kpp_value = NewString("value");
  kpp_error = NewString("error");
  kpp_warning = NewString("warning");
  kpp_pragma = NewString("pragma");
  kpp_level = NewString("level");
  kpp_line = NewString("line");
  kpp_include = NewString("include");
  kpp_varargs = NewString("varargs");

  kpp_dinclude = NewString("%include");
  kpp_dimport = NewString("%import");
  kpp_dbeginfile = NewString("%beginfile");
  kpp_dextern = NewString("%extern");
  kpp_ddefine = NewString("%define");
  kpp_dline = NewString("%line");


  kpp_LINE = NewString("__LINE__");
  kpp_FILE = NewString("__FILE__");

  kpp_hash_if = NewString("#if");
  kpp_hash_elif = NewString("#elif");

  cpp = NewHash();
  s = NewHash();
  Setattr(cpp, kpp_symbols, s);
  Delete(s);
  Preprocessor_expr_init();	/* Initialize the expression evaluator */
  included_files = NewHash();

  id_scan = NewScanner();

}

void Preprocessor_delete(void) {
  Delete(kpp_args);
  Delete(kpp_define);
  Delete(kpp_defined);
  Delete(kpp_else);
  Delete(kpp_elif);
  Delete(kpp_endif);
  Delete(kpp_expanded);
  Delete(kpp_if);
  Delete(kpp_ifdef);
  Delete(kpp_ifndef);
  Delete(kpp_name);
  Delete(kpp_swigmacro);
  Delete(kpp_symbols);
  Delete(kpp_undef);
  Delete(kpp_value);
  Delete(kpp_error);
  Delete(kpp_warning);
  Delete(kpp_pragma);
  Delete(kpp_level);
  Delete(kpp_line);
  Delete(kpp_include);
  Delete(kpp_varargs);

  Delete(kpp_dinclude);
  Delete(kpp_dimport);
  Delete(kpp_dbeginfile);
  Delete(kpp_dextern);
  Delete(kpp_ddefine);
  Delete(kpp_dline);

  Delete(kpp_LINE);
  Delete(kpp_FILE);

  Delete(kpp_hash_if);
  Delete(kpp_hash_elif);

  Delete(cpp);
  Delete(included_files);
  Preprocessor_expr_delete();
  DelScanner(id_scan);

  Delete(dependencies);

  Delete(Swig_add_directory(0));
}

/* -----------------------------------------------------------------------------
 * void Preprocessor_include_all() - Instruct preprocessor to include all files
 * ----------------------------------------------------------------------------- */
void Preprocessor_include_all(int a) {
  include_all = a;
}

void Preprocessor_import_all(int a) {
  import_all = a;
}

void Preprocessor_ignore_missing(int a) {
  ignore_missing = a;
}

void Preprocessor_error_as_warning(int a) {
  error_as_warning = a;
}


/* -----------------------------------------------------------------------------
 * Preprocessor_define()
 *
 * Defines a new C preprocessor symbol.   swigmacro specifies whether or not the macro has
 * SWIG macro semantics.
 * ----------------------------------------------------------------------------- */


String *Macro_vararg_name(const_String_or_char_ptr str, const_String_or_char_ptr line) {
  String *argname;
  String *varargname;
  char *s, *dots;

  argname = Copy(str);
  s = Char(argname);
  dots = strchr(s, '.');
  if (!dots) {
    Delete(argname);
    return NULL;
  }

  if (strcmp(dots, "...") != 0) {
    Swig_error(Getfile(line), Getline(line), "Illegal macro argument name '%s'\n", str);
    Delete(argname);
    return NULL;
  }
  if (dots == s) {
    varargname = NewString("__VA_ARGS__");
  } else {
    *dots = '\0';
    varargname = NewString(s);
  }
  Delete(argname);
  return varargname;
}

Hash *Preprocessor_define(const_String_or_char_ptr _str, int swigmacro) {
  String *macroname = 0, *argstr = 0, *macrovalue = 0, *file = 0, *s = 0;
  Hash *macro = 0, *symbols = 0, *m1;
  List *arglist = 0;
  int c, line;
  int varargs = 0;
  String *str;

  assert(cpp);
  assert(_str);

  /* First make sure that string is actually a string */
  if (DohCheck(_str)) {
    s = Copy(_str);
    copy_location(_str, s);
    str = s;
  } else {
    str = NewString((char *) _str);
  }
  Seek(str, 0, SEEK_SET);
  line = Getline(str);
  file = Getfile(str);

  /* Skip over any leading whitespace */
  skip_whitespace(str, 0);

  /* Now look for a macro name */
  macroname = NewStringEmpty();
  copy_location(str, macroname);
  while ((c = Getc(str)) != EOF) {
    if (c == '(') {
      argstr = NewStringEmpty();
      copy_location(str, argstr);
      /* It is a macro.  Go extract its argument string */
      while ((c = Getc(str)) != EOF) {
	if (c == ')')
	  break;
	else
	  Putc(c, argstr);
      }
      if (c != ')') {
	Swig_error(Getfile(argstr), Getline(argstr), "Missing \')\' in macro parameters\n");
	goto macro_error;
      }
      break;
    } else if (isidchar(c) || (c == '%')) {
      Putc(c, macroname);
    } else if (isspace(c)) {
      break;
    } else if (c == '\\') {
      c = Getc(str);
      if (c != '\n') {
	Ungetc(c, str);
	Ungetc('\\', str);
	break;
      }
    } else {
      Ungetc(c, str);
      break;
    }
  }
  if (!swigmacro)
    skip_whitespace(str, 0);
  macrovalue = NewStringEmpty();
  copy_location(str, macrovalue);
  while ((c = Getc(str)) != EOF) {
    Putc(c, macrovalue);
  }

  /* If there are any macro arguments, convert into a list */
  if (argstr) {
    String *argname, *varargname;
    arglist = NewList();
    Seek(argstr, 0, SEEK_SET);
    argname = NewStringEmpty();
    while ((c = Getc(argstr)) != EOF) {
      if (c == ',') {
	varargname = Macro_vararg_name(argname, argstr);
	if (varargname) {
	  Delete(varargname);
	  Swig_error(Getfile(argstr), Getline(argstr), "Variable length macro argument must be last parameter\n");
	} else {
	  Append(arglist, argname);
	}
	Delete(argname);
	argname = NewStringEmpty();
      } else if (isidchar(c) || (c == '.')) {
	Putc(c, argname);
      } else if (!(isspace(c) || (c == '\\'))) {
	Delete(argname);
	Swig_error(Getfile(argstr), Getline(argstr), "Illegal character in macro argument name\n");
	goto macro_error;
      }
    }
    if (Len(argname)) {
      /* Check for varargs */
      varargname = Macro_vararg_name(argname, argstr);
      if (varargname) {
	Append(arglist, varargname);
	Delete(varargname);
	varargs = 1;
      } else {
	Append(arglist, argname);
      }
    }
    Delete(argname);
  }

  if (!swigmacro) {
    Replace(macrovalue, "\\\n", " ", DOH_REPLACE_NOQUOTE);
  }

  /* Look for special # substitutions.   We only consider # that appears
     outside of quotes and comments */

  {
    int state = 0;
    char *cc = Char(macrovalue);
    while (*cc) {
      switch (state) {
      case 0:
	if (*cc == '#')
	  *cc = '\001';
	else if (*cc == '/')
	  state = 10;
	else if (*cc == '\'')
	  state = 20;
	else if (*cc == '\"')
	  state = 30;
	break;
      case 10:
	if (*cc == '*')
	  state = 11;
	else if (*cc == '/')
	  state = 15;
	else {
	  state = 0;
	  cc--;
	}
	break;
      case 11:
	if (*cc == '*')
	  state = 12;
	break;
      case 12:
	if (*cc == '/')
	  state = 0;
	else if (*cc != '*')
	  state = 11;
	break;
      case 15:
	if (*cc == '\n')
	  state = 0;
	break;
      case 20:
	if (*cc == '\'')
	  state = 0;
	if (*cc == '\\')
	  state = 21;
	break;
      case 21:
	state = 20;
	break;
      case 30:
	if (*cc == '\"')
	  state = 0;
	if (*cc == '\\')
	  state = 31;
	break;
      case 31:
	state = 30;
	break;
      default:
	break;
      }
      cc++;
    }
  }

  /* Get rid of whitespace surrounding # */
  /*  Replace(macrovalue,"#","\001",DOH_REPLACE_NOQUOTE); */
  while (strstr(Char(macrovalue), "\001 ")) {
    Replace(macrovalue, "\001 ", "\001", DOH_REPLACE_ANY);
  }
  while (strstr(Char(macrovalue), " \001")) {
    Replace(macrovalue, " \001", "\001", DOH_REPLACE_ANY);
  }
  /* Replace '##' with a special token */
  Replace(macrovalue, "\001\001", "\002", DOH_REPLACE_ANY);
  /* Replace '#@' with a special token */
  Replace(macrovalue, "\001@", "\004", DOH_REPLACE_ANY);
  /* Replace '##@' with a special token */
  Replace(macrovalue, "\002@", "\005", DOH_REPLACE_ANY);

  /* Go create the macro */
  macro = NewHash();
  Setattr(macro, kpp_name, macroname);

  if (arglist) {
    Setattr(macro, kpp_args, arglist);
    Delete(arglist);
    if (varargs) {
      Setattr(macro, kpp_varargs, "1");
    }
  }
  Setattr(macro, kpp_value, macrovalue);
  Setline(macro, line);
  Setfile(macro, file);
  if (swigmacro) {
    Setattr(macro, kpp_swigmacro, "1");
  }
  symbols = Getattr(cpp, kpp_symbols);
  if ((m1 = Getattr(symbols, macroname))) {
    if (!Checkattr(m1, kpp_value, macrovalue)) {
      Swig_error(Getfile(macroname), Getline(macroname), "Macro '%s' redefined,\n", macroname);
      Swig_error(Getfile(m1), Getline(m1), "previous definition of '%s'.\n", macroname);
      goto macro_error;
    }
  } else {
    Setattr(symbols, macroname, macro);
    Delete(macro);
  }

  Delete(macroname);
  Delete(macrovalue);

  Delete(str);
  Delete(argstr);
  return macro;

macro_error:
  Delete(str);
  Delete(argstr);
  Delete(arglist);
  Delete(macroname);
  Delete(macrovalue);
  return 0;
}

/* -----------------------------------------------------------------------------
 * Preprocessor_undef()
 *
 * Undefines a macro.
 * ----------------------------------------------------------------------------- */
void Preprocessor_undef(const_String_or_char_ptr str) {
  Hash *symbols;
  assert(cpp);
  symbols = Getattr(cpp, kpp_symbols);
  Delattr(symbols, str);
}

/* -----------------------------------------------------------------------------
 * find_args()
 *
 * Isolates macro arguments and returns them in a list.   For each argument,
 * leading and trailing whitespace is stripped (ala K&R, pg. 230).
 * ----------------------------------------------------------------------------- */
static List *find_args(String *s, int ismacro, String *macro_name) {
  List *args;
  String *str;
  int c, level;
  long pos;

  /* Create a new list */
  args = NewList();
  copy_location(s, args);

  /* First look for a '(' */
  pos = Tell(s);
  skip_whitespace(s, 0);

  /* Now see if the next character is a '(' */
  c = Getc(s);
  if (c != '(') {
    /* Not a macro, bail out now! */
    assert(pos != -1);
    (void)Seek(s, pos, SEEK_SET);
    Delete(args);
    return 0;
  }
  c = Getc(s);
  /* Okay.  This appears to be a macro so we will start isolating arguments */
  while (c != EOF) {
    if (isspace(c)) {
      skip_whitespace(s, 0);	/* Skip leading whitespace */
      c = Getc(s);
    }
    str = NewStringEmpty();
    copy_location(s, str);
    level = 0;
    while (c != EOF) {
      if (c == '\"') {
	Putc(c, str);
	skip_tochar(s, '\"', str);
	c = Getc(s);
	continue;
      } else if (c == '\'') {
	Putc(c, str);
	skip_tochar(s, '\'', str);
	c = Getc(s);
	continue;
      }
      if ((c == ',') && (level == 0))
	break;
      if ((c == ')') && (level == 0))
	break;
      Putc(c, str);
      if (c == '(')
	level++;
      if (c == ')')
	level--;
      c = Getc(s);
    }
    if (level > 0) {
      goto unterm;
    }
    Chop(str);
    if (Len(args) || Len(str))
      Append(args, str);
    Delete(str);

    /*    if (Len(str) && (c != ')'))
       Append(args,str); */

    if (c == ')')
      return args;
    c = Getc(s);
  }
unterm:
  if (ismacro)
    Swig_error(Getfile(args), Getline(args), "Unterminated call invoking macro '%s'\n", macro_name);
  else
    Swig_error(Getfile(args), Getline(args), "Unterminated call to '%s'\n", macro_name);
  return args;
}

/* -----------------------------------------------------------------------------
 * DOH *get_filename()
 *
 * Read a filename from str.   A filename can be enclosed in quotes, angle brackets,
 * or bare.
 * ----------------------------------------------------------------------------- */

static String *get_filename(String *str, int *sysfile) {
  String *fn;
  int c;

  fn = NewStringEmpty();
  copy_location(str, fn);
  c = Getc(str);
  *sysfile = 0;
  if (c == '\"') {
    while (((c = Getc(str)) != EOF) && (c != '\"'))
      Putc(c, fn);
  } else if (c == '<') {
    *sysfile = 1;
    while (((c = Getc(str)) != EOF) && (c != '>'))
      Putc(c, fn);
  } else {
    String *preprocessed_str;
    Putc(c, fn);
    while (((c = Getc(str)) != EOF) && (!isspace(c)))
      Putc(c, fn);
    if (isspace(c))
      Ungetc(c, str);
    preprocessed_str = Preprocessor_replace(fn);
    Seek(preprocessed_str, 0, SEEK_SET);
    Delete(fn);

    fn = NewStringEmpty();
    copy_location(preprocessed_str, fn);
    c = Getc(preprocessed_str);
    if (c == '\"') {
      while (((c = Getc(preprocessed_str)) != EOF) && (c != '\"'))
	Putc(c, fn);
    } else if (c == '<') {
      *sysfile = 1;
      while (((c = Getc(preprocessed_str)) != EOF) && (c != '>'))
	Putc(c, fn);
    } else {
      fn = Copy(preprocessed_str);
    }
    Delete(preprocessed_str);
  }
  Swig_filename_unescape(fn);
  Swig_filename_correct(fn);
  Seek(fn, 0, SEEK_SET);
  return fn;
}

static String *get_options(String *str) {
  int c;
  c = Getc(str);
  if (c == '(') {
    String *opt;
    int level = 1;
    opt = NewString("(");
    while (((c = Getc(str)) != EOF)) {
      Putc(c, opt);
      if (c == ')') {
	level--;
	if (!level)
	  return opt;
      }
      if (c == '(')
	level++;
    }
    Delete(opt);
    return 0;
  } else {
    Ungetc(c, str);
    return 0;
  }
}

/* -----------------------------------------------------------------------------
 * expand_macro()
 *
 * Perform macro expansion and return a new string.  Returns NULL if some sort
 * of error occurred.
 * name - name of the macro
 * args - arguments passed to the macro
 * line_file - only used for line/file name when reporting errors
 * ----------------------------------------------------------------------------- */

static String *expand_macro(String *name, List *args, String *line_file) {
  String *ns;
  DOH *symbols, *macro, *margs, *mvalue, *temp, *tempa, *e;
  int i, l;
  int isvarargs = 0;

  symbols = Getattr(cpp, kpp_symbols);
  if (!symbols)
    return 0;

  /* See if the name is actually defined */
  macro = Getattr(symbols, name);
  if (!macro)
    return 0;

  if (macro_level == 0) {
    /* Store the start of the macro should the macro contain __LINE__ and __FILE__ for expansion */
    macro_start_line = Getline(args ? args : line_file);
    macro_start_file = Getfile(args ? args : line_file);
  }
  macro_level++;

  if (Getattr(macro, kpp_expanded)) {
    ns = NewStringEmpty();
    Append(ns, name);
    if (args) {
      int lenargs = Len(args);
      if (lenargs)
	Putc('(', ns);
      for (i = 0; i < lenargs; i++) {
	Append(ns, Getitem(args, i));
	if (i < (lenargs - 1))
	  Putc(',', ns);
      }
      if (i)
	Putc(')', ns);
    }
    macro_level--;
    return ns;
  }

  /* Get macro arguments and value */
  mvalue = Getattr(macro, kpp_value);
  assert(mvalue);
  margs = Getattr(macro, kpp_args);

  if (args && Getattr(macro, kpp_varargs)) {
    isvarargs = 1;
    /* Variable length argument macro.  We need to collect all of the extra arguments into a single argument */
    if (Len(args) >= (Len(margs) - 1)) {
      int i;
      int vi, na;
      String *vararg = NewStringEmpty();
      vi = Len(margs) - 1;
      na = Len(args);
      for (i = vi; i < na; i++) {
	Append(vararg, Getitem(args, i));
	if ((i + 1) < na) {
	  Append(vararg, ",");
	}
      }
      /* Remove arguments */
      for (i = vi; i < na; i++) {
	Delitem(args, vi);
      }
      Append(args, vararg);
      Delete(vararg);
    }
  }
  /* If there are arguments, see if they match what we were given */
  if (args && (margs) && (Len(margs) != Len(args))) {
    if (Len(margs) > (1 + isvarargs))
      Swig_error(macro_start_file, macro_start_line, "Macro '%s' expects %d arguments\n", name, Len(margs) - isvarargs);
    else if (Len(margs) == (1 + isvarargs))
      Swig_error(macro_start_file, macro_start_line, "Macro '%s' expects 1 argument\n", name);
    else
      Swig_error(macro_start_file, macro_start_line, "Macro '%s' expects no arguments\n", name);
    macro_level--;
    return 0;
  }

  /* If the macro expects arguments, but none were supplied, we leave it in place */
  if (!args && (margs) && Len(margs) > 0) {
    macro_level--;
    return NewString(name);
  }

  /* Copy the macro value */
  ns = Copy(mvalue);
  copy_location(mvalue, ns);

  /* Tag the macro as being expanded.   This is to avoid recursion in
     macro expansion */

  temp = NewStringEmpty();
  tempa = NewStringEmpty();
  if (args && margs) {
    l = Len(margs);
    for (i = 0; i < l; i++) {
      DOH *arg, *aname;
      String *reparg;
      arg = Getitem(args, i);	/* Get an argument value */
      reparg = Preprocessor_replace(arg);
      aname = Getitem(margs, i);	/* Get macro argument name */
      if (strstr(Char(ns), "\001")) {
	/* Try to replace a quoted version of the argument */
	Clear(temp);
	Clear(tempa);
	Printf(temp, "\001%s", aname);
	Printf(tempa, "\"%s\"", arg);
	Replace(ns, temp, tempa, DOH_REPLACE_ID_END);
      }
      if (strstr(Char(ns), "\002")) {
	/* Look for concatenation tokens */
	Clear(temp);
	Clear(tempa);
	Printf(temp, "\002%s", aname);
	Append(tempa, "\002\003");
	Replace(ns, temp, tempa, DOH_REPLACE_ID_END);
	Clear(temp);
	Clear(tempa);
	Printf(temp, "%s\002", aname);
	Append(tempa, "\003\002");
	Replace(ns, temp, tempa, DOH_REPLACE_ID_BEGIN);
      }

      /* Non-standard macro expansion.   The value `x` is replaced by a quoted
         version of the argument except that if the argument is already quoted
         nothing happens */

      if (strchr(Char(ns), '`')) {
	String *rep;
	char *c;
	Clear(temp);
	Printf(temp, "`%s`", aname);
	c = Char(arg);
	if (*c == '\"') {
	  rep = arg;
	} else {
	  Clear(tempa);
	  Printf(tempa, "\"%s\"", arg);
	  rep = tempa;
	}
	Replace(ns, temp, rep, DOH_REPLACE_ANY);
      }

      /* Non-standard mangle expansions.  
         The #@Name is replaced by mangle_arg(Name). */
      if (strstr(Char(ns), "\004")) {
	String *marg = Swig_string_mangle(arg);
	Clear(temp);
	Printf(temp, "\004%s", aname);
	Replace(ns, temp, marg, DOH_REPLACE_ID_END);
	Delete(marg);
      }
      if (strstr(Char(ns), "\005")) {
	String *marg = Swig_string_mangle(arg);
	Clear(temp);
	Clear(tempa);
	Printf(temp, "\005%s", aname);
	Printf(tempa, "\"%s\"", marg);
	Replace(ns, temp, tempa, DOH_REPLACE_ID_END);
	Delete(marg);
      }

      if (isvarargs && i == l - 1 && Len(arg) == 0) {
	/* Zero length varargs macro argument.   We search for commas that might appear before and nuke them */
	char *a, *s, *t, *name;
	int namelen;
	s = Char(ns);
	name = Char(aname);
	namelen = Len(aname);
	a = strstr(s, name);
	while (a) {
	  char ca = a[namelen + 1];
	  if (!isidchar((int) ca)) {
	    /* Matched the entire vararg name, not just a prefix */
	    t = a - 1;
	    if (*t == '\002') {
	      t--;
	      while (t >= s) {
		if (isspace((int) *t))
		  t--;
		else if (*t == ',') {
		  *t = ' ';
		} else
		  break;
	      }
	    }
	  }
	  a = strstr(a + namelen, name);
	}
      }
      /*      Replace(ns, aname, arg, DOH_REPLACE_ID); */
      Replace(ns, aname, reparg, DOH_REPLACE_ID);	/* Replace expanded args */
      Replace(ns, "\003", arg, DOH_REPLACE_ANY);	/* Replace unexpanded arg */
      Delete(reparg);
    }
  }
  Replace(ns, "\002", "", DOH_REPLACE_ANY);	/* Get rid of concatenation tokens */
  Replace(ns, "\001", "#", DOH_REPLACE_ANY);	/* Put # back (non-standard C) */
  Replace(ns, "\004", "#@", DOH_REPLACE_ANY);	/* Put # back (non-standard C) */

  /* Expand this macro even further */
  Setattr(macro, kpp_expanded, "1");

  e = Preprocessor_replace(ns);

  Delattr(macro, kpp_expanded);
  Delete(ns);

  if (Getattr(macro, kpp_swigmacro)) {
    String *g;
    String *f = NewStringEmpty();
    Seek(e, 0, SEEK_SET);
    copy_location(macro, e);
    g = Preprocessor_parse(e);

#if 0
    /* Drop the macro in place, but with a marker around it */
    Printf(f, "/*@%s,%d,%s@*/%s/*@@*/", Getfile(macro), Getline(macro), name, g);
#else
    /* Use simplified around markers to properly count lines in cscanner.c */
    if (strchr(Char(g), '\n')) {
      Printf(f, "/*@SWIG:%s,%d,%s@*/%s/*@SWIG@*/", Getfile(macro), Getline(macro), name, g);
#if 0
      Printf(f, "/*@SWIG:%s@*/%s/*@SWIG@*/", name, g);
#endif
    } else {
      Append(f, g);
    }
#endif

    Delete(g);
    Delete(e);
    e = f;
  }
  macro_level--;
  Delete(temp);
  Delete(tempa);
  return e;
}

/* -----------------------------------------------------------------------------
 * DOH *Preprocessor_replace(DOH *s)
 *
 * Performs a macro substitution on a string s.  Returns a new string with
 * substitutions applied.   This function works by walking down s and looking
 * for identifiers.   When found, a check is made to see if they are macros
 * which are then expanded.
 * ----------------------------------------------------------------------------- */

/* #define SWIG_PUT_BUFF  */

static DOH *Preprocessor_replace(DOH *s) {
  DOH *ns, *symbols, *m;
  int c, i, state = 0;
  String *id = NewStringEmpty();

  assert(cpp);
  symbols = Getattr(cpp, kpp_symbols);

  ns = NewStringEmpty();
  copy_location(s, ns);
  Seek(s, 0, SEEK_SET);

  /* Try to locate identifiers in s and replace them with macro replacements */
  while ((c = Getc(s)) != EOF) {
    switch (state) {
    case 0:
      if (isidentifier(c)) {
	Clear(id);
	Putc(c, id);
	state = 4;
      } else if (c == '%') {
	Clear(id);
	Putc(c, id);
	state = 2;
      } else if (c == '#') {
	Clear(id);
	Putc(c, id);
	state = 4;
      } else if (c == '\"') {
	Putc(c, ns);
	skip_tochar(s, '\"', ns);
      } else if (c == '\'') {
	Putc(c, ns);
	skip_tochar(s, '\'', ns);
      } else if (c == '/') {
	Putc(c, ns);
	state = 10;
      } else if (c == '\\') {
	Putc(c, ns);
	c = Getc(s);
	if (c == '\n') {
	  Putc(c, ns);
	} else {
	  Ungetc(c, s);
	}
      } else if (c == '\n') {
	Putc(c, ns);
	expand_defined_operator = 0;
      } else {
	Putc(c, ns);
      }
      break;
    case 2:
      /* Found '%#' */
      if (c == '#') {
	Putc(c, id);
	state = 4;
      } else {
	Ungetc(c, s);
	state = 4;
      }
      break;
    case 4:			/* An identifier */
      if (isidchar(c)) {
	Putc(c, id);
	state = 4;
      } else {
	/* We found the end of a valid identifier */
	Ungetc(c, s);
	/* See if this is the special "defined" operator */
       	if (Equal(kpp_defined, id)) {
	  if (expand_defined_operator) {
	    int lenargs = 0;
	    DOH *args = 0;
	    /* See whether or not a parenthesis has been used */
	    skip_whitespace(s, 0);
	    c = Getc(s);
	    if (c == '(') {
	      Ungetc(c, s);
	      args = find_args(s, 0, kpp_defined);
	    } else if (isidchar(c)) {
	      DOH *arg = NewStringEmpty();
	      args = NewList();
	      Putc(c, arg);
	      while (((c = Getc(s)) != EOF)) {
		if (!isidchar(c)) {
		  Ungetc(c, s);
		  break;
		}
		Putc(c, arg);
	      }
	      if (Len(arg))
		Append(args, arg);
	      Delete(arg);
	    } else {
	      Seek(s, -1, SEEK_CUR);
	    }
	    lenargs = Len(args);
	    if ((!args) || (!lenargs)) {
	      /* This is not a defined() operator. */
	      Append(ns, id);
	      state = 0;
	      break;
	    }
	    for (i = 0; i < lenargs; i++) {
	      DOH *o = Getitem(args, i);
	      if (!Getattr(symbols, o)) {
		break;
	      }
	    }
	    if (i < lenargs)
	      Putc('0', ns);
	    else
	      Putc('1', ns);
	    Delete(args);
	  } else {
	    Append(ns, id);
	  }
	  state = 0;
	  break;
	} else if (Equal(kpp_LINE, id)) {
	  Printf(ns, "%d", macro_level > 0 ? macro_start_line : Getline(s));
	  state = 0;
	  break;
	} else if (Equal(kpp_FILE, id)) {
	  String *fn = Copy(macro_level > 0 ? macro_start_file : Getfile(s));
	  Replaceall(fn, "\\", "\\\\");
	  Printf(ns, "\"%s\"", fn);
	  Delete(fn);
	  state = 0;
	  break;
	} else if (Equal(kpp_hash_if, id) || Equal(kpp_hash_elif, id)) {
	  expand_defined_operator = 1;
	  Append(ns, id);
	  /*
	} else if (Equal("%#if", id) || Equal("%#ifdef", id)) {
	  Swig_warning(998, Getfile(s), Getline(s), "Found: %s preprocessor directive.\n", id);
	  Append(ns, id);
	} else if (Equal("#ifdef", id) || Equal("#ifndef", id)) {
	  Swig_warning(998, Getfile(s), Getline(s), "The %s preprocessor directive does not work in macros, try #if instead.\n", id);
	  Append(ns, id);
	  */
	} else if ((m = Getattr(symbols, id))) {
	  /* See if the macro is defined in the preprocessor symbol table */
	  DOH *args = 0;
	  DOH *e;
	  int macro_additional_lines = 0;
	  /* See if the macro expects arguments */
	  if (Getattr(m, kpp_args)) {
	    /* Yep.  We need to go find the arguments and do a substitution */
	    int line = Getline(s);
	    args = find_args(s, 1, id);
	    macro_additional_lines = Getline(s) - line;
	    assert(macro_additional_lines >= 0);
	    if (!Len(args)) {
	      Delete(args);
	      args = 0;
	    }
	  } else {
	    args = 0;
	  }
	  e = expand_macro(id, args, s);
	  if (e) {
	    Append(ns, e);
	  }
	  while (macro_additional_lines--) {
	    Putc('\n', ns);
	  }
	  Delete(e);
	  Delete(args);
	} else {
	  Append(ns, id);
	}
	state = 0;
      }
      break;
    case 10:
      if (c == '/')
	state = 11;
      else if (c == '*')
	state = 12;
      else {
	Ungetc(c, s);
	state = 0;
	break;
      }
      Putc(c, ns);
      break;
    case 11:
      /* in C++ comment */
      Putc(c, ns);
      if (c == '\n') {
	expand_defined_operator = 0;
	state = 0;
      }
      break;
    case 12:
      /* in C comment */
      Putc(c, ns);
      if (c == '*')
	state = 13;
      break;
    case 13:
      Putc(c, ns);
      if (c == '/')
	state = 0;
      else if (c != '*')
	state = 12;
      break;
    default:
      state = 0;
      break;
    }
  }

  /* Identifier at the end */
  if (state == 2 || state == 4) {
    /* See if this is the special "defined" operator */
    if (Equal(kpp_defined, id)) {
      Swig_error(Getfile(s), Getline(s), "No arguments given to defined()\n");
    } else if (Equal(kpp_LINE, id)) {
      Printf(ns, "%d", macro_level > 0 ? macro_start_line : Getline(s));
    } else if (Equal(kpp_FILE, id)) {
      String *fn = Copy(macro_level > 0 ? macro_start_file : Getfile(s));
      Replaceall(fn, "\\", "\\\\");
      Printf(ns, "\"%s\"", fn);
      Delete(fn);
    } else if (Getattr(symbols, id)) {
      DOH *e;
      /* Yes.  There is a macro here */
      /* See if the macro expects arguments */
      e = expand_macro(id, 0, s);
      if (e)
	Append(ns, e);
      Delete(e);
    } else {
      Append(ns, id);
    }
  }
  Delete(id);
  return ns;
}


/* -----------------------------------------------------------------------------
 * int checkpp_id(DOH *s)
 *
 * Checks the string s to see if it contains any unresolved identifiers.  This
 * function contains the heuristic that determines whether or not a macro
 * definition passes through the preprocessor as a constant declaration.
 * ----------------------------------------------------------------------------- */
static int checkpp_id(DOH *s) {
  int c;
  int hastok = 0;
  Scanner *scan = id_scan;

  Seek(s, 0, SEEK_SET);

  Scanner_clear(scan);
  s = Copy(s);
  Seek(s, SEEK_SET, 0);
  Scanner_push(scan, s);
  while ((c = Scanner_token(scan))) {
    hastok = 1;
    if ((c == SWIG_TOKEN_ID) || (c == SWIG_TOKEN_LBRACE) || (c == SWIG_TOKEN_RBRACE))
      return 1;
  }
  if (!hastok)
    return 1;
  return 0;
}

/* addline().  Utility function for adding lines to a chunk */
static void addline(DOH *s1, DOH *s2, int allow) {
  if (allow) {
    Append(s1, s2);
  } else {
    char *c = Char(s2);
    while (*c) {
      if (*c == '\n')
	Putc('\n', s1);
      c++;
    }
  }
}

static void add_chunk(DOH *ns, DOH *chunk, int allow) {
  DOH *echunk;
  Seek(chunk, 0, SEEK_SET);
  if (allow) {
    echunk = Preprocessor_replace(chunk);
    addline(ns, echunk, allow);
    Delete(echunk);
  } else {
    addline(ns, chunk, 0);
  }
  Clear(chunk);
}

/*
  push/pop_imported(): helper functions for defining and undefining
  SWIGIMPORTED (when %importing a file).
 */
static void push_imported() {
  if (imported_depth == 0) {
    Preprocessor_define("SWIGIMPORTED 1", 0);
  }
  ++imported_depth;
}

static void pop_imported() {
  --imported_depth;
  if (imported_depth == 0) {
    Preprocessor_undef("SWIGIMPORTED");
  }
}


/* -----------------------------------------------------------------------------
 * Preprocessor_parse()
 *
 * Parses the string s.  Returns a new string containing the preprocessed version.
 *
 * Parsing rules :
 *       1.  Lines starting with # are C preprocessor directives
 *       2.  Macro expansion inside strings is not allowed
 *       3.  All code inside false conditionals is changed to blank lines
 *       4.  Code in %{, %} is not parsed because it may need to be
 *           included inline (with all preprocessor directives included).
 * ----------------------------------------------------------------------------- */

String *Preprocessor_parse(String *s) {
  String *ns;			/* New string containing the preprocessed text */
  String *chunk, *decl;
  Hash *symbols;
  String *id = 0, *value = 0, *comment = 0;
  int i, state, e, c;
  int start_line = 0;
  int allow = 1;
  int level = 0;
  int dlevel = 0;
  int filelevel = 0;
  int mask = 0;
  int start_level = 0;
  int cpp_lines = 0;
  int cond_lines[256];

  /* Blow away all carriage returns */
  Replace(s, "\015", "", DOH_REPLACE_ANY);

  ns = NewStringEmpty();	/* Return result */

  decl = NewStringEmpty();
  id = NewStringEmpty();
  value = NewStringEmpty();
  comment = NewStringEmpty();
  chunk = NewStringEmpty();
  copy_location(s, chunk);
  copy_location(s, ns);
  symbols = Getattr(cpp, kpp_symbols);

  state = 0;
  while ((c = Getc(s)) != EOF) {
    switch (state) {
    case 0:			/* Initial state - in first column */
      /* Look for C preprocessor directives.   Otherwise, go directly to state 1 */
      if (c == '#') {
	copy_location(s, chunk);
	add_chunk(ns, chunk, allow);
	cpp_lines = 1;
	state = 40;
      } else if (isspace(c)) {
	Putc(c, chunk);
	skip_whitespace(s, chunk);
      } else {
	state = 1;
	Ungetc(c, s);
      }
      break;
    case 1:			/* Non-preprocessor directive */
      /* Look for SWIG directives */
      if (c == '%') {
	state = 100;
	break;
      }
      Putc(c, chunk);
      if (c == '\n')
	state = 0;
      else if (c == '\"') {
	start_line = Getline(s);
	if (skip_tochar(s, '\"', chunk) < 0) {
	  Swig_error(Getfile(s), start_line, "Unterminated string constant\n");
	}
      } else if (c == '\'') {
	start_line = Getline(s);
	if (skip_tochar(s, '\'', chunk) < 0) {
	  Swig_error(Getfile(s), start_line, "Unterminated character constant\n");
	}
      } else if (c == '/')
	state = 30;		/* Comment */
      break;

    case 30:			/* Possibly a comment string of some sort */
      start_line = Getline(s);
      Putc(c, chunk);
      if (c == '/')
	state = 31;
      else if (c == '*')
	state = 32;
      else
	state = 1;
      break;
    case 31:
      Putc(c, chunk);
      if (c == '\n')
	state = 0;
      break;
    case 32:
      Putc(c, chunk);
      if (c == '*')
	state = 33;
      break;
    case 33:
      Putc(c, chunk);
      if (c == '/')
	state = 1;
      else if (c != '*')
	state = 32;
      break;

    case 40:			/* Start of a C preprocessor directive */
      if (c == '\n') {
	Putc('\n', chunk);
	state = 0;
      } else if (isspace(c)) {
	state = 40;
      } else {
	/* Got the start of a preprocessor directive */
	Ungetc(c, s);
	Clear(id);
	copy_location(s, id);
	state = 41;
      }
      break;

    case 41:			/* Build up the name of the preprocessor directive */
      if ((isspace(c) || (!isalpha(c)))) {
	Clear(value);
	Clear(comment);
	if (c == '\n') {
	  Ungetc(c, s);
	  state = 50;
	} else {
	  state = 42;
	  if (!isspace(c)) {
	    Ungetc(c, s);
	  }
	}

	copy_location(s, value);
	break;
      }
      Putc(c, id);
      break;

    case 42:			/* Strip any leading space before preprocessor value */
      if (isspace(c)) {
	if (c == '\n') {
	  Ungetc(c, s);
	  state = 50;
	}
	break;
      }
      state = 43;
      /* FALL THRU */

    case 43:
      /* Get preprocessor value */
      if (c == '\n') {
	Ungetc(c, s);
	state = 50;
      } else if (c == '/') {
	state = 45;
      } else if (c == '\"') {
	Putc(c, value);
	skip_tochar(s, '\"', value);
      } else if (c == '\'') {
	Putc(c, value);
	skip_tochar(s, '\'', value);
      } else {
	Putc(c, value);
	if (c == '\\')
	  state = 44;
      }
      break;

    case 44:
      if (c == '\n') {
	Putc(c, value);
	cpp_lines++;
      } else {
	Ungetc(c, s);
      }
      state = 43;
      break;

      /* States 45-48 are used to remove, but retain comments from macro values.  The comments
         will be placed in the output in an alternative form */

    case 45:
      if (c == '/')
	state = 46;
      else if (c == '*')
	state = 47;
      else if (c == '\n') {
	Putc('/', value);
	Ungetc(c, s);
	state = 50;
      } else {
	Putc('/', value);
	Putc(c, value);
	state = 43;
      }
      break;
    case 46: /* in C++ comment */
      if (c == '\n') {
	Ungetc(c, s);
	state = 50;
      } else
	Putc(c, comment);
      break;
    case 47: /* in C comment */
      if (c == '*')
	state = 48;
      else
	Putc(c, comment);
      break;
    case 48:
      if (c == '/')
	state = 43;
      else if (c == '*')
	Putc(c, comment);
      else {
	Putc('*', comment);
	Putc(c, comment);
	state = 47;
      }
      break;
    case 50:
      /* Check for various preprocessor directives */
      Chop(value);
      if (Equal(id, kpp_define)) {
	if (allow) {
	  DOH *m, *v, *v1;
	  Seek(value, 0, SEEK_SET);
	  m = Preprocessor_define(value, 0);
	  if ((m) && !(Getattr(m, kpp_args))) {
	    v = Copy(Getattr(m, kpp_value));
	    copy_location(m, v);
	    if (Len(v)) {
	      Swig_error_silent(1);
	      v1 = Preprocessor_replace(v);
	      Swig_error_silent(0);
	      /*              Printf(stdout,"checking '%s'\n", v1); */
	      if (!checkpp_id(v1)) {
		if (Len(comment) == 0)
		  Printf(ns, "%%constant %s = %s;\n", Getattr(m, kpp_name), v1);
		else
		  Printf(ns, "%%constant %s = %s; /*%s*/\n", Getattr(m, kpp_name), v1, comment);
		cpp_lines--;
	      }
	      Delete(v1);
	    }
	    Delete(v);
	  }
	}
      } else if (Equal(id, kpp_undef)) {
	if (allow)
	  Preprocessor_undef(value);
      } else if (Equal(id, kpp_ifdef)) {
	cond_lines[level] = Getline(id);
	level++;
	if (allow) {
	  start_level = level;
	  if (Len(value) > 0) {
	    /* See if the identifier is in the hash table */
	    if (!Getattr(symbols, value))
	      allow = 0;
	  } else {
	    Swig_error(Getfile(s), Getline(id), "Missing identifier for #ifdef.\n");
	    allow = 0;
	  }
	  mask = 1;
	}
      } else if (Equal(id, kpp_ifndef)) {
	cond_lines[level] = Getline(id);
	level++;
	if (allow) {
	  start_level = level;
	  if (Len(value) > 0) {
	    /* See if the identifier is in the hash table */
	    if (Getattr(symbols, value))
	      allow = 0;
	  } else {
	    Swig_error(Getfile(s), Getline(id), "Missing identifier for #ifndef.\n");
	    allow = 0;
	  }
	  mask = 1;
	}
      } else if (Equal(id, kpp_else)) {
	if (level <= 0) {
	  Swig_error(Getfile(s), Getline(id), "Misplaced #else.\n");
	} else {
	  cond_lines[level - 1] = Getline(id);
	  if (Len(value) != 0)
	    Swig_warning(WARN_PP_UNEXPECTED_TOKENS, Getfile(s), Getline(id), "Unexpected tokens after #else directive.\n");
	  if (allow) {
	    allow = 0;
	    mask = 0;
	  } else if (level == start_level) {
	    allow = 1 * mask;
	  }
	}
      } else if (Equal(id, kpp_endif)) {
	level--;
	if (level < 0) {
	  Swig_error(Getfile(id), Getline(id), "Extraneous #endif.\n");
	  level = 0;
	} else {
	  if (level < start_level) {
	    if (Len(value) != 0)
	      Swig_warning(WARN_PP_UNEXPECTED_TOKENS, Getfile(s), Getline(id), "Unexpected tokens after #endif directive.\n");
	    allow = 1;
	    start_level--;
	  }
	}
      } else if (Equal(id, kpp_if)) {
	cond_lines[level] = Getline(id);
	level++;
	if (allow) {
	  int val;
	  String *sval;
	  expand_defined_operator = 1;
	  sval = Preprocessor_replace(value);
	  start_level = level;
	  Seek(sval, 0, SEEK_SET);
	  /*      Printf(stdout,"Evaluating '%s'\n", sval); */
	  if (Len(sval) > 0) {
	    val = Preprocessor_expr(sval, &e);
	    if (e) {
	      const char *msg = Preprocessor_expr_error();
	      Seek(value, 0, SEEK_SET);
	      Swig_warning(WARN_PP_EVALUATION, Getfile(value), Getline(value), "Could not evaluate expression '%s'\n", value);
	      if (msg)
		Swig_warning(WARN_PP_EVALUATION, Getfile(value), Getline(value), "Error: '%s'\n", msg);
	      allow = 0;
	    } else {
	      if (val == 0)
		allow = 0;
	    }
	  } else {
	    Swig_error(Getfile(s), Getline(id), "Missing expression for #if.\n");
	    allow = 0;
	  }
	  expand_defined_operator = 0;
	  mask = 1;
	}
      } else if (Equal(id, kpp_elif)) {
	if (level == 0) {
	  Swig_error(Getfile(s), Getline(id), "Misplaced #elif.\n");
	} else {
	  cond_lines[level - 1] = Getline(id);
	  if (allow) {
	    allow = 0;
	    mask = 0;
	  } else if (level == start_level) {
	    int val;
	    String *sval;
	    expand_defined_operator = 1;
	    sval = Preprocessor_replace(value);
	    Seek(sval, 0, SEEK_SET);
	    if (Len(sval) > 0) {
	      val = Preprocessor_expr(sval, &e);
	      if (e) {
		const char *msg = Preprocessor_expr_error();
		Seek(value, 0, SEEK_SET);
		Swig_warning(WARN_PP_EVALUATION, Getfile(value), Getline(value), "Could not evaluate expression '%s'\n", value);
		if (msg)
		  Swig_warning(WARN_PP_EVALUATION, Getfile(value), Getline(value), "Error: '%s'\n", msg);
		allow = 0;
	      } else {
		if (val)
		  allow = 1 * mask;
		else
		  allow = 0;
	      }
	    } else {
	      Swig_error(Getfile(s), Getline(id), "Missing expression for #elif.\n");
	      allow = 0;
	    }
	    expand_defined_operator = 0;
	  }
	}
      } else if (Equal(id, kpp_warning)) {
	if (allow) {
	  Swig_warning(WARN_PP_CPP_WARNING, Getfile(s), Getline(id), "CPP #warning, \"%s\".\n", value);
	}
      } else if (Equal(id, kpp_error)) {
	if (allow) {
	  if (error_as_warning) {
	    Swig_warning(WARN_PP_CPP_ERROR, Getfile(s), Getline(id), "CPP #error \"%s\".\n", value);
	  } else {
	    Swig_error(Getfile(s), Getline(id), "CPP #error \"%s\". Use the -cpperraswarn option to continue swig processing.\n", value);
	  }
	}
      } else if (Equal(id, kpp_line)) {
      } else if (Equal(id, kpp_include)) {
	if (((include_all) || (import_all)) && (allow)) {
	  String *s1, *s2, *fn;
	  String *dirname;
	  int sysfile = 0;
	  if (include_all && import_all) {
	    Swig_warning(WARN_PP_INCLUDEALL_IMPORTALL, Getfile(s), Getline(id), "Both includeall and importall are defined: using includeall.\n");
	    import_all = 0;
	  }
	  Seek(value, 0, SEEK_SET);
	  fn = get_filename(value, &sysfile);
	  s1 = cpp_include(fn, sysfile);
	  if (s1) {
	    if (include_all)
	      Printf(ns, "%%includefile \"%s\" %%beginfile\n", Swig_filename_escape(Swig_last_file()));
	    else if (import_all) {
	      Printf(ns, "%%importfile \"%s\" %%beginfile\n", Swig_filename_escape(Swig_last_file()));
	      push_imported();
	    }

	    /* See if the filename has a directory component */
	    dirname = Swig_file_dirname(Swig_last_file());
	    if (sysfile || !Len(dirname)) {
	      Delete(dirname);
	      dirname = 0;
	    }
	    if (dirname) {
	      int len = Len(dirname);
	      Delslice(dirname, len - 1, len); /* Kill trailing directory delimiter */
	      Swig_push_directory(dirname);
	    }
	    s2 = Preprocessor_parse(s1);
	    addline(ns, s2, allow);
	    Append(ns, "%endoffile");
	    if (dirname) {
	      Swig_pop_directory();
	    }
	    if (import_all) {
	      pop_imported();
	    }
	    Delete(s2);
	    Delete(dirname);
	    Delete(s1);
	  }
	  Delete(fn);
	}
      } else if (Equal(id, kpp_pragma)) {
	if (Strncmp(value, "SWIG ", 5) == 0) {
	  char *c = Char(value) + 5;
	  while (*c && (isspace((int) *c)))
	    c++;
	  if (*c) {
	    if (strncmp(c, "nowarn=", 7) == 0) {
	      String *val = NewString(c + 7);
	      String *nowarn = Preprocessor_replace(val);
	      Swig_warnfilter(nowarn, 1);
	      Delete(nowarn);
	      Delete(val);
	    } else if (strncmp(c, "cpperraswarn=", 13) == 0) {
	      error_as_warning = atoi(c + 13);
	    } else {
              Swig_error(Getfile(s), Getline(id), "Unknown SWIG pragma: %s\n", c);
            }
	  }
	}
      } else if (Equal(id, kpp_level)) {
	Swig_error(Getfile(s), Getline(id), "cpp debug: level = %d, startlevel = %d\n", level, start_level);
      } else if (Equal(id, "")) {
	/* Null directive */
      } else {
	/* Ignore unknown preprocessor directives which are inside an inactive
	 * conditional (github issue #394). */
	if (allow)
	  Swig_error(Getfile(s), Getline(id), "Unknown SWIG preprocessor directive: %s (if this is a block of target language code, delimit it with %%{ and %%})\n", id);
      }
      for (i = 0; i < cpp_lines; i++)
	Putc('\n', ns);
      state = 0;
      break;

      /* SWIG directives  */
    case 100:
      /* %{,%} block  */
      if (c == '{') {
	start_line = Getline(s);
	copy_location(s, chunk);
	add_chunk(ns, chunk, allow);
	Putc('%', chunk);
	Putc(c, chunk);
	state = 105;
      }
      /* %#cpp -  an embedded C preprocessor directive (we strip off the %)  */
      else if (c == '#') {
	add_chunk(ns, chunk, allow);
	Putc(c, chunk);
	state = 107;
      } else if (isidentifier(c)) {
	Clear(decl);
	Putc('%', decl);
	Putc(c, decl);
	state = 110;
      } else {
	Putc('%', chunk);
	Putc(c, chunk);
	state = 1;
      }
      break;

    case 105:
      Putc(c, chunk);
      if (c == '%')
	state = 106;
      break;

    case 106:
      Putc(c, chunk);
      if (c == '}') {
	state = 1;
	addline(ns, chunk, allow);
	Clear(chunk);
	copy_location(s, chunk);
      } else {
	state = 105;
      }
      break;

    case 107:
      Putc(c, chunk);
      if (c == '\n') {
	addline(ns, chunk, allow);
	Clear(chunk);
	state = 0;
      } else if (c == '\\') {
	state = 108;
      }
      break;

    case 108:
      Putc(c, chunk);
      state = 107;
      break;

    case 110:
      if (!isidchar(c)) {
	Ungetc(c, s);
	/* Look for common SWIG directives  */
	if (Equal(decl, kpp_dinclude) || Equal(decl, kpp_dimport) || Equal(decl, kpp_dextern)) {
	  /* Got some kind of file inclusion directive, eg: %import(option1="value1") "filename" */
	  if (allow) {
	    DOH *s1, *s2, *fn, *opt;
	    String *options_whitespace = NewStringEmpty();
	    String *filename_whitespace = NewStringEmpty();
	    int sysfile = 0;

	    if (Equal(decl, kpp_dextern)) {
	      Swig_warning(WARN_DEPRECATED_EXTERN, Getfile(s), Getline(s), "%%extern is deprecated. Use %%import instead.\n");
	      Clear(decl);
	      Append(decl, "%%import");
	    }
	    skip_whitespace(s, options_whitespace);
	    opt = get_options(s);

	    skip_whitespace(s, filename_whitespace);
	    fn = get_filename(s, &sysfile);
	    s1 = cpp_include(fn, sysfile);
	    if (s1) {
	      String *dirname;
	      copy_location(s, chunk);
	      add_chunk(ns, chunk, allow);
	      Printf(ns, "%sfile%s%s%s\"%s\" %%beginfile\n", decl, options_whitespace, opt, filename_whitespace, Swig_filename_escape(Swig_last_file()));
	      if (Equal(decl, kpp_dimport)) {
		push_imported();
	      }
	      dirname = Swig_file_dirname(Swig_last_file());
	      if (sysfile || !Len(dirname)) {
		Delete(dirname);
		dirname = 0;
	      }
	      if (dirname) {
		int len = Len(dirname);
		Delslice(dirname, len - 1, len); /* Kill trailing directory delimiter */
		Swig_push_directory(dirname);
	      }
	      s2 = Preprocessor_parse(s1);
	      if (dirname) {
		Swig_pop_directory();
	      }
	      if (Equal(decl, kpp_dimport)) {
		pop_imported();
	      }
	      addline(ns, s2, allow);
	      Append(ns, "%endoffile");
	      Delete(s2);
	      Delete(dirname);
	      Delete(s1);
	    }
	    Delete(fn);
	    Delete(filename_whitespace);
	    Delete(options_whitespace);
	  }
	  state = 1;
	} else if (Equal(decl, kpp_dbeginfile)) {
          /* Got an internal directive marking the beginning of an included file: %beginfile ... %endoffile */
          filelevel++;
          start_line = Getline(s);
          copy_location(s, chunk);
          add_chunk(ns, chunk, allow);
          Append(chunk, decl);
          state = 120;
	} else if (Equal(decl, kpp_dline)) {
	  /* Got a line directive  */
	  state = 1;
	} else if (Equal(decl, kpp_ddefine)) {
	  /* Got a define directive  */
	  dlevel++;
	  copy_location(s, chunk);
	  add_chunk(ns, chunk, allow);
	  Clear(value);
	  copy_location(s, value);
	  state = 150;
	} else {
	  Append(chunk, decl);
	  state = 1;
	}
      } else {
	Putc(c, decl);
      }
      break;

      /* Searching for the end of a %beginfile block */
    case 120:
      Putc(c, chunk);
      if (c == '%') {
        const char *bf = "beginfile";
        const char *ef = "endoffile";
        char statement[10];
        int i = 0;
        for (i = 0; i < 9;) {
          c = Getc(s);
          Putc(c, chunk);
          statement[i++] = (char)c;
	  if (strncmp(statement, bf, i) && strncmp(statement, ef, i))
	    break;
	}
	c = Getc(s);
	Ungetc(c, s);
        if ((i == 9) && (isspace(c))) {
	  if (strncmp(statement, bf, i) == 0) {
	    ++filelevel;
	  } else if (strncmp(statement, ef, i) == 0) {
            --filelevel;
            if (!filelevel) {
              /* Reached end of included file */
              addline(ns, chunk, allow);
              Clear(chunk);
              copy_location(s, chunk);
              state = 1;
            }
          }
        }
      }
      break;

      /* Searching for the end of a %define statement  */
    case 150:
      Putc(c, value);
      if (c == '%') {
	const char *ed = "enddef";
	const char *df = "define";
	char statement[7];
	int i = 0;
	for (i = 0; i < 6;) {
	  c = Getc(s);
	  Putc(c, value);
	  statement[i++] = (char)c;
	  if (strncmp(statement, ed, i) && strncmp(statement, df, i))
	    break;
	}
	c = Getc(s);
	Ungetc(c, s);
	if ((i == 6) && (isspace(c))) {
	  if (strncmp(statement, df, i) == 0) {
	    ++dlevel;
	  } else {
	    if (strncmp(statement, ed, i) == 0) {
	      --dlevel;
	      if (!dlevel) {
		/* Got the macro  */
		for (i = 0; i < 7; i++) {
		  Delitem(value, DOH_END);
		}
		if (allow) {
		  Seek(value, 0, SEEK_SET);
		  Preprocessor_define(value, 1);
		}
		addline(ns, value, 0);
		state = 0;
	      }
	    }
	  }
	}
      }
      break;
    default:
      Printf(stderr, "cpp: Invalid parser state %d\n", state);
      abort();
      break;
    }
  }
  while (level > 0) {
    Swig_error(Getfile(s), cond_lines[level - 1], "Missing #endif for conditional starting here\n");
    level--;
  }
  if (state == 120) {
    Swig_error(Getfile(s), start_line, "Missing %%endoffile for file inclusion block starting here\n");
  }
  if (state == 150) {
    Seek(value, 0, SEEK_SET);
    Swig_error(Getfile(s), Getline(value), "Missing %%enddef for macro starting here\n", Getline(value));
  }
  if ((state >= 105) && (state < 107)) {
    Swig_error(Getfile(s), start_line, "Unterminated %%{ ... %%} block\n");
  }
  if ((state >= 30) && (state < 40)) {
    Swig_error(Getfile(s), start_line, "Unterminated comment\n");
  }

  copy_location(s, chunk);
  add_chunk(ns, chunk, allow);

  /*  DelScope(scp); */
  Delete(decl);
  Delete(id);
  Delete(value);
  Delete(comment);
  Delete(chunk);

  return ns;
}
