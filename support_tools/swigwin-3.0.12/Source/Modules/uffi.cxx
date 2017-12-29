/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * uffi.cxx
 *
 * Uffi language module for SWIG.
 * ----------------------------------------------------------------------------- */

// TODO: remove remnants of lisptype

#include "swigmod.h"

static const char *usage = "\
UFFI Options (available with -uffi)\n\
     -identifier-converter <type or funcname> - \n\
                       Specifies the type of conversion to do on C identifiers\n\
                       to convert them to symbols. There are two built-in\n\
                       converters: 'null' and 'lispify'. The default is\n\
                       'null'. If you supply a name other than one of the\n\
                       built-ins, then a function by that name will be\n\
                       called to convert identifiers to symbols.\n\
";

class UFFI:public Language {
public:

  virtual void main(int argc, char *argv[]);
  virtual int top(Node *n);
  virtual int functionWrapper(Node *n);
  virtual int constantWrapper(Node *n);
  virtual int classHandler(Node *n);
  virtual int membervariableHandler(Node *n);

};

static File *f_cl = 0;

static struct {
  int count;
  String **entries;
} defined_foreign_types;

static String *identifier_converter = NewString("identifier-convert-null");

static int any_varargs(ParmList *pl) {
  Parm *p;

  for (p = pl; p; p = nextSibling(p)) {
    if (SwigType_isvarargs(Getattr(p, "type")))
      return 1;
  }

  return 0;
}


/* utilities */
/* returns new string w/ parens stripped */
static String *strip_parens(String *string) {
  char *s = Char(string), *p;
  int len = Len(string);
  String *res;

  if (len == 0 || s[0] != '(' || s[len - 1] != ')') {
    return NewString(string);
  }

  p = (char *) malloc(len - 2 + 1);
  if (!p) {
    Printf(stderr, "Malloc failed\n");
    SWIG_exit(EXIT_FAILURE);
  }

  strncpy(p, s + 1, len - 1);
  p[len - 2] = 0;		/* null terminate */

  res = NewString(p);
  free(p);

  return res;
}


static String *convert_literal(String *num_param, String *type) {
  String *num = strip_parens(num_param), *res;
  char *s = Char(num);

  /* Make sure doubles use 'd' instead of 'e' */
  if (!Strcmp(type, "double")) {
    String *updated = Copy(num);
    if (Replace(updated, "e", "d", DOH_REPLACE_ANY) > 1) {
      Printf(stderr, "Weird!! number %s looks invalid.\n", num);
      SWIG_exit(EXIT_FAILURE);
    }
    Delete(num);
    return updated;
  }

  if (SwigType_type(type) == T_CHAR) {
    /* Use CL syntax for character literals */
    return NewStringf("#\\%s", num_param);
  } else if (SwigType_type(type) == T_STRING) {
    /* Use CL syntax for string literals */
    return NewStringf("\"%s\"", num_param);
  }

  if (Len(num) < 2 || s[0] != '0') {
    return num;
  }

  /* octal or hex */

  res = NewStringf("#%c%s", s[1] == 'x' ? 'x' : 'o', s + 2);
  Delete(num);

  return res;
}

static void add_defined_foreign_type(String *type) {
  if (!defined_foreign_types.count) {
    /* Make fresh */
    defined_foreign_types.count = 1;
    defined_foreign_types.entries = (String **) malloc(sizeof(String *));
  } else {
    /* make room */
    defined_foreign_types.count++;
    defined_foreign_types.entries = (String **)
	realloc(defined_foreign_types.entries, defined_foreign_types.count * sizeof(String *));
  }

  if (!defined_foreign_types.entries) {
    Printf(stderr, "Out of memory\n");
    SWIG_exit(EXIT_FAILURE);
  }

  /* Fill in the new data */
  defined_foreign_types.entries[defined_foreign_types.count - 1] = Copy(type);

}


static String *get_ffi_type(Node *n, SwigType *ty, const_String_or_char_ptr name) {
  Node *node = NewHash();
  Setattr(node, "type", ty);
  Setattr(node, "name", name);
  Setfile(node, Getfile(n));
  Setline(node, Getline(n));
  const String *tm = Swig_typemap_lookup("ffitype", node, "", 0);
  Delete(node);

  if (tm) {
    return NewString(tm);
  } else {
    SwigType *tr = SwigType_typedef_resolve_all(ty);
    char *type_reduced = Char(tr);
    int i;

    //Printf(stdout,"convert_type %s\n", ty);
    if (SwigType_isconst(tr)) {
      SwigType_pop(tr);
      type_reduced = Char(tr);
    }

    if (SwigType_ispointer(type_reduced) || SwigType_isarray(ty) || !strncmp(type_reduced, "p.f", 3)) {
      return NewString(":pointer-void");
    }

    for (i = 0; i < defined_foreign_types.count; i++) {
      if (!Strcmp(ty, defined_foreign_types.entries[i])) {
	return NewStringf("#.(%s \"%s\" :type :type)", identifier_converter, ty);
      }
    }

    if (!Strncmp(type_reduced, "enum ", 5)) {
      return NewString(":int");
    }

    Printf(stderr, "Unsupported data type: %s (was: %s)\n", type_reduced, ty);
    SWIG_exit(EXIT_FAILURE);
  }
  return 0;
}

static String *get_lisp_type(Node *n, SwigType *ty, const_String_or_char_ptr name) {
  Node *node = NewHash();
  Setattr(node, "type", ty);
  Setattr(node, "name", name);
  Setfile(node, Getfile(n));
  Setline(node, Getline(n));
  const String *tm = Swig_typemap_lookup("lisptype", node, "", 0);
  Delete(node);

  return tm ? NewString(tm) : NewString("");
}

void UFFI::main(int argc, char *argv[]) {
  int i;

  Preprocessor_define("SWIGUFFI 1", 0);
  SWIG_library_directory("uffi");
  SWIG_config_file("uffi.swg");


  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-identifier-converter")) {
      char *conv = argv[i + 1];

      if (!conv)
	Swig_arg_error();

      Swig_mark_arg(i);
      Swig_mark_arg(i + 1);
      i++;

      /* check for built-ins */
      if (!strcmp(conv, "lispify")) {
	Delete(identifier_converter);
	identifier_converter = NewString("identifier-convert-lispify");
      } else if (!strcmp(conv, "null")) {
	Delete(identifier_converter);
	identifier_converter = NewString("identifier-convert-null");
      } else {
	/* Must be user defined */
	Delete(identifier_converter);
	identifier_converter = NewString(conv);
      }
    }

    if (!strcmp(argv[i], "-help")) {
      Printf(stdout, "%s\n", usage);
    }
  }
}

int UFFI::top(Node *n) {
  String *module = Getattr(n, "name");
  String *output_filename = NewString("");
  File *f_null = NewString("");

  Printf(output_filename, "%s%s.cl", SWIG_output_directory(), module);


  f_cl = NewFile(output_filename, "w", SWIG_output_files());
  if (!f_cl) {
    FileErrorDisplay(output_filename);
    SWIG_exit(EXIT_FAILURE);
  }

  Swig_register_filebyname("header", f_null);
  Swig_register_filebyname("begin", f_null);
  Swig_register_filebyname("runtime", f_null);
  Swig_register_filebyname("wrapper", f_cl);

  Swig_banner_target_lang(f_cl, ";;");

  Printf(f_cl, "\n"
	 ";; -*- Mode: Lisp; Syntax: ANSI-Common-Lisp; Base: 10; package: %s -*-\n\n(defpackage :%s\n  (:use :common-lisp :uffi))\n\n(in-package :%s)\n",
	 module, module, module);
  Printf(f_cl, "(eval-when (compile load eval)\n  (defparameter *swig-identifier-converter* '%s))\n", identifier_converter);

  Language::top(n);

  Delete(f_cl);			// Delete the handle, not the file
  Delete(f_null);

  return SWIG_OK;
}

int UFFI::functionWrapper(Node *n) {
  String *funcname = Getattr(n, "sym:name");
  ParmList *pl = Getattr(n, "parms");
  Parm *p;
  int argnum = 0, first = 1;
//  int varargs = 0;

  //Language::functionWrapper(n);

  Printf(f_cl, "(swig-defun \"%s\"\n", funcname);
  Printf(f_cl, "  (");

  /* Special cases */

  if (ParmList_len(pl) == 0) {
    Printf(f_cl, ":void");
  } else if (any_varargs(pl)) {
    Printf(f_cl, "#| varargs |#");
//    varargs = 1;
  } else {
    for (p = pl; p; p = nextSibling(p), argnum++) {
      String *argname = Getattr(p, "name");
      SwigType *argtype = Getattr(p, "type");
      String *ffitype = get_ffi_type(n, argtype, argname);
      String *lisptype = get_lisp_type(n, argtype, argname);
      int tempargname = 0;

      if (!argname) {
	argname = NewStringf("arg%d", argnum);
	tempargname = 1;
      }

      if (!first) {
	Printf(f_cl, "\n   ");
      }
      Printf(f_cl, "(%s %s %s)", argname, ffitype, lisptype);
      first = 0;

      Delete(ffitype);
      Delete(lisptype);
      if (tempargname)
	Delete(argname);

    }
  }
  Printf(f_cl, ")\n");		/* finish arg list */
  Printf(f_cl, "  :returning %s\n"
	 //"  :strings-convert t\n"
	 //"  :call-direct %s\n"
	 //"  :optimize-for-space t"
	 ")\n", get_ffi_type(n, Getattr(n, "type"), Swig_cresult_name())
	 //,varargs ? "nil"  : "t"
      );


  return SWIG_OK;
}

int UFFI::constantWrapper(Node *n) {
  String *type = Getattr(n, "type");
  String *converted_value = convert_literal(Getattr(n, "value"), type);
  String *name = Getattr(n, "sym:name");

#if 0
  Printf(stdout, "constant %s is of type %s. value: %s\n", name, type, converted_value);
#endif

  Printf(f_cl, "(swig-defconstant \"%s\" %s)\n", name, converted_value);

  Delete(converted_value);

  return SWIG_OK;
}

// Includes structs
int UFFI::classHandler(Node *n) {

  String *name = Getattr(n, "sym:name");
  String *kind = Getattr(n, "kind");
  Node *c;

  if (Strcmp(kind, "struct")) {
    Printf(stderr, "Don't know how to deal with %s kind of class yet.\n", kind);
    Printf(stderr, " (name: %s)\n", name);
    SWIG_exit(EXIT_FAILURE);
  }

  Printf(f_cl, "(swig-def-struct \"%s\"\n \n", name);

  for (c = firstChild(n); c; c = nextSibling(c)) {
    SwigType *type = Getattr(c, "type");
    SwigType *decl = Getattr(c, "decl");
    type = Copy(type);
    SwigType_push(type, decl);
    String *lisp_type;

    if (Strcmp(nodeType(c), "cdecl")) {
      Printf(stderr, "Structure %s has a slot that we can't deal with.\n", name);
      Printf(stderr, "nodeType: %s, name: %s, type: %s\n", nodeType(c), Getattr(c, "name"), Getattr(c, "type"));
      SWIG_exit(EXIT_FAILURE);
    }


    /* Printf(stdout, "Converting %s in %s\n", type, name); */
    lisp_type = get_ffi_type(n, type, Getattr(c, "sym:name"));

    Printf(f_cl, "  (#.(%s \"%s\" :type :slot) %s)\n", identifier_converter, Getattr(c, "sym:name"), lisp_type);

    Delete(lisp_type);
  }

  // Language::classHandler(n);

  Printf(f_cl, " )\n");

  /* Add this structure to the known lisp types */
  //Printf(stdout, "Adding %s foreign type\n", name);
  add_defined_foreign_type(name);

  return SWIG_OK;
}

int UFFI::membervariableHandler(Node *n) {
  Language::membervariableHandler(n);
  return SWIG_OK;
}


extern "C" Language *swig_uffi(void) {
  return new UFFI();
}
