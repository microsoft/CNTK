/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * s-exp.cxx
 *
 * A parse tree represented as Lisp s-expressions.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "dohint.h"

static const char *usage = "\
S-Exp Options (available with -sexp)\n\
     -typemaplang <lang> - Typemap language\n\n";

//static Node *view_top = 0;
static File *out = 0;

class Sexp:public Language {
  int indent_level;
  DOHHash *print_circle_hash;
  int print_circle_count;
  int hanging_parens;
  bool need_whitespace;
  bool need_newline;

public:
  Sexp():
    indent_level(0),
    print_circle_hash(0),
    print_circle_count(0),
    hanging_parens(0),
    need_whitespace(0),
    need_newline(0) {
  }
  
  virtual ~ Sexp() {
  }

  virtual void main(int argc, char *argv[]) {
    // Add a symbol to the parser for conditional compilation
    Preprocessor_define("SWIGSEXP 1", 0);

    SWIG_typemap_lang("sexp");
    for (int iX = 0; iX < argc; iX++) {
      if (strcmp(argv[iX], "-typemaplang") == 0) {
	Swig_mark_arg(iX);
	iX++;
	SWIG_typemap_lang(argv[iX]);
	Swig_mark_arg(iX);
	continue;
      }
      if (strcmp(argv[iX], "-help") == 0) {
	fputs(usage, stdout);
      }
    }
  }

  /* Top of the parse tree */
  virtual int top(Node *n) {
    if (out == 0) {
      String *outfile = Getattr(n, "outfile");
      Replaceall(outfile, "_wrap.cxx", ".lisp");
      Replaceall(outfile, "_wrap.c", ".lisp");
      out = NewFile(outfile, "w", SWIG_output_files());
      if (!out) {
	FileErrorDisplay(outfile);
	SWIG_exit(EXIT_FAILURE);
      }
    }
    String *f_sink = NewString("");
    Swig_register_filebyname("header", f_sink);
    Swig_register_filebyname("wrapper", f_sink);
    Swig_register_filebyname("begin", f_sink);
    Swig_register_filebyname("runtime", f_sink);
    Swig_register_filebyname("init", f_sink);

    Swig_banner_target_lang(out, ";;;");

    Language::top(n);
    Printf(out, "\n");
    Printf(out, ";;; Lisp parse tree produced by SWIG\n");
    print_circle_hash = NewHash();
    print_circle_count = 0;
    hanging_parens = 0;
    need_whitespace = 0;
    need_newline = 0;
    Sexp_print_node(n);
    flush_parens();
    return SWIG_OK;
  }

  void print_indent() {
    int i;
    for (i = 0; i < indent_level; i++) {
      Printf(out, " ");
    }
  }

  void open_paren(const String *oper) {
    flush_parens();
    Printf(out, "(");
    if (oper)
      Printf(out, "%s ", oper);
    indent_level += 2;
  }

  void close_paren(bool neednewline = false) {
    hanging_parens++;
    if (neednewline)
      print_lazy_whitespace();
    indent_level -= 2;
  }

  void flush_parens() {
    int i;
    if (hanging_parens) {
      for (i = 0; i < hanging_parens; i++)
	Printf(out, ")");
      hanging_parens = 0;
      need_newline = true;
      need_whitespace = true;
    }
    if (need_newline) {
      Printf(out, "\n");
      print_indent();
      need_newline = false;
      need_whitespace = false;
    } else if (need_whitespace) {
      Printf(out, " ");
      need_whitespace = false;
    }
  }

  void print_lazy_whitespace() {
    need_whitespace = 1;
  }

  void print_lazy_newline() {
    need_newline = 1;
  }

  bool internal_key_p(DOH *key) {
    return ((Cmp(key, "nodeType") == 0)
	    || (Cmp(key, "firstChild") == 0)
	    || (Cmp(key, "lastChild") == 0)
	    || (Cmp(key, "parentNode") == 0)
	    || (Cmp(key, "nextSibling") == 0)
	    || (Cmp(key, "previousSibling") == 0)
	    || (Cmp(key, "csym:nextSibling") == 0)
	    || (Cmp(key, "csym:previousSibling") == 0)
	    || (Cmp(key, "typepass:visit") == 0)
	    || (Cmp(key, "allocate:visit") == 0)
	    || (*(Char(key)) == '$'));
  }

  bool boolean_key_p(DOH *key) {
    return ((Cmp(key, "allocate:default_constructor") == 0)
	    || (Cmp(key, "allocate:default_destructor") == 0)
	    || (Cmp(key, "allows_typedef") == 0)
	    || (Cmp(key, "feature:immutable") == 0));
  }

  bool list_key_p(DOH *key) {
    return ((Cmp(key, "parms") == 0)
	    || (Cmp(key, "baselist") == 0));
  }

  bool plist_key_p(DOH *key)
      // true if KEY is the name of data that is a mapping from keys to
      // values, which should be printed as a plist.
  {
    return ((Cmp(key, "typescope") == 0));
  }

  bool maybe_plist_key_p(DOH *key) {
    return (Strncmp(key, "tmap:", 5) == 0);
  }

  bool print_circle(DOH *obj, bool list_p)
      // We have a complex object, which might be referenced several
      // times, or even recursively.  Use Lisp's reader notation for
      // circular structures (#n#, #n=).
      //
      // An object can be printed in list-mode or object-mode; LIST_P toggles.
      // return TRUE if OBJ still needs to be printed
  {
    flush_parens();
    // Following is a silly hack.  It works around the limitation of
    // DOH's hash tables that only work with string keys!
    char address[32];
    sprintf(address, "%p%c", obj, list_p ? 'L' : 'O');
    DOH *placeholder = Getattr(print_circle_hash, address);
    if (placeholder) {
      Printv(out, placeholder, NIL);
      return false;
    } else {
      String *placeholder = NewStringf("#%d#", ++print_circle_count);
      Setattr(print_circle_hash, address, placeholder);
      Printf(out, "#%d=", print_circle_count);
      return true;
    }
  }

  void Sexp_print_value_of_key(DOH *value, DOH *key) {
    if ((Cmp(key, "parms") == 0) || (Cmp(key, "wrap:parms") == 0)
	|| (Cmp(key, "kwargs") == 0) || (Cmp(key, "pattern") == 0))
      Sexp_print_parms(value);
    else if (plist_key_p(key))
      Sexp_print_plist(value);
    else if (maybe_plist_key_p(key)) {
      if (DohIsMapping(value))
	Sexp_print_plist(value);
      else
	Sexp_print_doh(value);
    } else if (list_key_p(key))
      Sexp_print_list(value);
    else if (boolean_key_p(key))
      Sexp_print_boolean(value);
    else
      Sexp_print_doh(value);
  }

  void Sexp_print_boolean(DOH *obj) {
    flush_parens();
    /* See DOH/Doh/base.c, DohGetInt() */
    if (DohIsString(obj)) {
      if (atoi(Char(obj)) != 0)
	Printf(out, "t");
      else
	Printf(out, "nil");
    } else
      Printf(out, "nil");
  }

  void Sexp_print_list(DOH *obj) {
    if (print_circle(obj, true)) {
      open_paren(NIL);
      for (; obj; obj = nextSibling(obj)) {
	Sexp_print_doh(obj);
	print_lazy_whitespace();
      }
      close_paren(true);
    }
  }

  void Sexp_print_parms(DOH *obj) {
    // print it as a list of plists
    if (print_circle(obj, true)) {
      open_paren(NIL);
      for (; obj; obj = nextSibling(obj)) {
	if (DohIsMapping(obj)) {
	  Iterator k;
	  open_paren(NIL);
	  for (k = First(obj); k.key; k = Next(k)) {
	    if (!internal_key_p(k.key)) {
	      DOH *value = Getattr(obj, k.key);
	      Sexp_print_as_keyword(k.key);
	      Sexp_print_value_of_key(value, k.key);
	      print_lazy_whitespace();
	    }
	  }
	  close_paren(true);
	} else
	  Sexp_print_doh(obj);
	print_lazy_whitespace();
      }
      close_paren(true);
    }
  }

  void Sexp_print_doh(DOH *obj) {
    flush_parens();
    if (DohIsString(obj)) {
      String *o = Str(obj);
      Replaceall(o, "\\", "\\\\");
      Replaceall(o, "\"", "\\\"");
      Printf(out, "\"%s\"", o);
      Delete(o);
    } else {
      if (print_circle(obj, false)) {
	// Dispatch type
	if (nodeType(obj)) {
	  Sexp_print_node(obj);
	}

	else if (DohIsMapping(obj)) {
	  Iterator k;
	  open_paren(NIL);
	  for (k = First(obj); k.key; k = Next(k)) {
	    if (!internal_key_p(k.key)) {
	      DOH *value = Getattr(obj, k.key);
	      flush_parens();
	      open_paren(NIL);
	      Sexp_print_doh(k.key);
	      Printf(out, " . ");
	      Sexp_print_value_of_key(value, k.key);
	      close_paren();
	    }
	  }
	  close_paren();
	} else if (strcmp(ObjType(obj)->objname, "List") == 0) {
	  int i;
	  open_paren(NIL);
	  for (i = 0; i < Len(obj); i++) {
	    DOH *item = Getitem(obj, i);
	    Sexp_print_doh(item);
	  }
	  close_paren();
	} else {
	  // What is it?
	  Printf(out, "#<DOH %s %p>", ObjType(obj)->objname, obj);
	}
      }
    }
  }

  void Sexp_print_as_keyword(const DOH *k) {
    /* Print key, replacing ":" with "-" because : is CL's package prefix */
    flush_parens();
    String *key = NewString(k);
    Replaceall(key, ":", "-");
    Replaceall(key, "_", "-");
    Printf(out, ":%s ", key);
    Delete(key);
  }

  void Sexp_print_plist_noparens(DOH *obj) {
    /* attributes map names to objects */
    Iterator k;
    bool first;
    for (k = First(obj), first = true; k.key; k = Next(k), first = false) {
      if (!internal_key_p(k.key)) {
	DOH *value = Getattr(obj, k.key);
	flush_parens();
	if (!first) {
	  Printf(out, " ");
	}
	Sexp_print_as_keyword(k.key);
	/* Print value */
	Sexp_print_value_of_key(value, k.key);
      }
    }
  }

  void Sexp_print_plist(DOH *obj) {
    flush_parens();
    if (print_circle(obj, true)) {
      open_paren(NIL);
      Sexp_print_plist_noparens(obj);
      close_paren();
    }
  }

  void Sexp_print_attributes(Node *obj) {
    Sexp_print_plist_noparens(obj);
  }

  void Sexp_print_node(Node *obj) {
    Node *cobj;
    open_paren(nodeType(obj));
    /* A node has an attribute list... */
    Sexp_print_attributes(obj);
    /* ... and child nodes. */
    cobj = firstChild(obj);
    if (cobj) {
      print_lazy_newline();
      flush_parens();
      Sexp_print_as_keyword("children");
      open_paren(NIL);
      for (; cobj; cobj = nextSibling(cobj)) {
	Sexp_print_node(cobj);
      }
      close_paren();
    }
    close_paren();
  }


  virtual int functionWrapper(Node *n) {
    ParmList *l = Getattr(n, "parms");
    Wrapper *f = NewWrapper();
    emit_attach_parmmaps(l, f);
    Setattr(n, "wrap:parms", l);
    DelWrapper(f);
    return SWIG_OK;
  }

};


static Language *new_swig_sexp() {
  return new Sexp();
}
extern "C" Language *swig_sexp(void) {
  return new_swig_sexp();
}
