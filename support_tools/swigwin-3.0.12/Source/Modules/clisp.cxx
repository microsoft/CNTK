/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * clisp.cxx
 *
 * clisp language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

static const char *usage = "\
CLISP Options (available with -clisp)\n\
     -extern-all       - Create clisp definitions for all the functions and\n\
                         global variables otherwise only definitions for\n\
                         externed functions and variables are created.\n\
     -generate-typedef - Use def-c-type to generate shortcuts according to the\n\
                         typedefs in the input.\n\
";

class CLISP:public Language {
public:
  File *f_cl;
  String *module;
  virtual void main(int argc, char *argv[]);
  virtual int top(Node *n);
  virtual int functionWrapper(Node *n);
  virtual int variableWrapper(Node *n);
  virtual int constantWrapper(Node *n);
  virtual int classDeclaration(Node *n);
  virtual int enumDeclaration(Node *n);
  virtual int typedefHandler(Node *n);
  List *entries;
private:
  String *get_ffi_type(Node *n, SwigType *ty);
  String *convert_literal(String *num_param, String *type);
  String *strip_parens(String *string);
  int extern_all_flag;
  int generate_typedef_flag;
  int is_function;
};

void CLISP::main(int argc, char *argv[]) {
  int i;

  Preprocessor_define("SWIGCLISP 1", 0);
  SWIG_library_directory("clisp");
  SWIG_config_file("clisp.swg");
  generate_typedef_flag = 0;
  extern_all_flag = 0;

  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-help")) {
      Printf(stdout, "%s\n", usage);
    } else if ((Strcmp(argv[i], "-extern-all") == 0)) {
      extern_all_flag = 1;
      Swig_mark_arg(i);
    } else if ((Strcmp(argv[i], "-generate-typedef") == 0)) {
      generate_typedef_flag = 1;
      Swig_mark_arg(i);
    }
  }
}

int CLISP::top(Node *n) {

  File *f_null = NewString("");
  module = Getattr(n, "name");
  String *output_filename;
  entries = NewList();

  /* Get the output file name */
  String *outfile = Getattr(n, "outfile");

  if (!outfile) {
    Printf(stderr, "Unable to determine outfile\n");
    SWIG_exit(EXIT_FAILURE);
  }

  output_filename = NewStringf("%s%s.lisp", SWIG_output_directory(), module);

  f_cl = NewFile(output_filename, "w+", SWIG_output_files());
  if (!f_cl) {
    FileErrorDisplay(output_filename);
    SWIG_exit(EXIT_FAILURE);
  }

  Swig_register_filebyname("header", f_null);
  Swig_register_filebyname("begin", f_null);
  Swig_register_filebyname("runtime", f_null);
  Swig_register_filebyname("wrapper", f_null);

  String *header = NewString("");

  Swig_banner_target_lang(header, ";;");

  Printf(header, "\n(defpackage :%s\n  (:use :common-lisp :ffi)", module);

  Language::top(n);

  Iterator i;

  long len = Len(entries);
  if (len > 0) {
    Printf(header, "\n  (:export");
  }
  //else nothing to export

  for (i = First(entries); i.item; i = Next(i)) {
    Printf(header, "\n\t:%s", i.item);
  }

  if (len > 0) {
    Printf(header, ")");
  }

  Printf(header, ")\n");
  Printf(header, "\n(in-package :%s)\n", module);
  Printf(header, "\n(default-foreign-language :stdc)\n");

  len = Tell(f_cl);

  Printf(f_cl, "%s", header);

  long end = Tell(f_cl);

  for (len--; len >= 0; len--) {
    end--;
    (void)Seek(f_cl, len, SEEK_SET);
    int ch = Getc(f_cl);
    (void)Seek(f_cl, end, SEEK_SET);
    Putc(ch, f_cl);
  }

  Seek(f_cl, 0, SEEK_SET);
  Write(f_cl, Char(header), Len(header));

  Delete(f_cl);

  return SWIG_OK;
}


int CLISP::functionWrapper(Node *n) {
  is_function = 1;
  String *storage = Getattr(n, "storage");
  if (!extern_all_flag && (!storage || (!Swig_storage_isextern(n) && !Swig_storage_isexternc(n))))
    return SWIG_OK;

  String *func_name = Getattr(n, "sym:name");

  ParmList *pl = Getattr(n, "parms");

  int argnum = 0, first = 1;

  Printf(f_cl, "\n(ffi:def-call-out %s\n\t(:name \"%s\")\n", func_name, func_name);

  Append(entries, func_name);

  if (ParmList_len(pl) != 0) {
    Printf(f_cl, "\t(:arguments ");
  }
  for (Parm *p = pl; p; p = nextSibling(p), argnum++) {

    String *argname = Getattr(p, "name");
    //    SwigType *argtype;

    String *ffitype = get_ffi_type(n, Getattr(p, "type"));

    int tempargname = 0;

    if (!argname) {
      argname = NewStringf("arg%d", argnum);
      tempargname = 1;
    }

    if (!first) {
      Printf(f_cl, "\n\t\t");
    }
    Printf(f_cl, "(%s %s)", argname, ffitype);
    first = 0;

    Delete(ffitype);

    if (tempargname)
      Delete(argname);
  }
  if (ParmList_len(pl) != 0) {
    Printf(f_cl, ")\n");	/* finish arg list */
  }
  String *ffitype = get_ffi_type(n, Getattr(n, "type"));
  if (Strcmp(ffitype, "NIL")) {	//when return type is not nil
    Printf(f_cl, "\t(:return-type %s)\n", ffitype);
  }
  Printf(f_cl, "\t(:library +library-name+))\n");

  return SWIG_OK;
}


int CLISP::constantWrapper(Node *n) {
  is_function = 0;
  String *type = Getattr(n, "type");
  String *converted_value = convert_literal(Getattr(n, "value"), type);
  String *name = Getattr(n, "sym:name");

  Printf(f_cl, "\n(defconstant %s %s)\n", name, converted_value);
  Append(entries, name);
  Delete(converted_value);

  return SWIG_OK;
}

int CLISP::variableWrapper(Node *n) {
  is_function = 0;
  String *storage = Getattr(n, "storage");

  if (!extern_all_flag && (!storage || (!Swig_storage_isextern(n) && !Swig_storage_isexternc(n))))
    return SWIG_OK;

  String *var_name = Getattr(n, "sym:name");
  String *lisp_type = get_ffi_type(n, Getattr(n, "type"));
  Printf(f_cl, "\n(ffi:def-c-var %s\n (:name \"%s\")\n (:type %s)\n", var_name, var_name, lisp_type);
  Printf(f_cl, "\t(:library +library-name+))\n");
  Append(entries, var_name);

  Delete(lisp_type);
  return SWIG_OK;
}

int CLISP::typedefHandler(Node *n) {
  if (generate_typedef_flag) {
    is_function = 0;
    Printf(f_cl, "\n(ffi:def-c-type %s %s)\n", Getattr(n, "name"), get_ffi_type(n, Getattr(n, "type")));
  }

  return Language::typedefHandler(n);
}

int CLISP::enumDeclaration(Node *n) {
  if (getCurrentClass() && (cplus_mode != PUBLIC))
    return SWIG_NOWRAP;

  is_function = 0;
  String *name = Getattr(n, "sym:name");

  Printf(f_cl, "\n(ffi:def-c-enum %s ", name);

  for (Node *c = firstChild(n); c; c = nextSibling(c)) {

    String *slot_name = Getattr(c, "name");
    String *value = Getattr(c, "enumvalue");

    Printf(f_cl, "(%s %s)", slot_name, value);

    Append(entries, slot_name);

    Delete(value);
  }

  Printf(f_cl, ")\n");
  return SWIG_OK;
}


// Includes structs
int CLISP::classDeclaration(Node *n) {
  is_function = 0;
  String *name = Getattr(n, "sym:name");
  String *kind = Getattr(n, "kind");

  if (Strcmp(kind, "struct")) {
    Printf(stderr, "Don't know how to deal with %s kind of class yet.\n", kind);
    Printf(stderr, " (name: %s)\n", name);
    SWIG_exit(EXIT_FAILURE);
  }


  Printf(f_cl, "\n(ffi:def-c-struct %s", name);

  Append(entries, NewStringf("make-%s", name));

  for (Node *c = firstChild(n); c; c = nextSibling(c)) {

    if (Strcmp(nodeType(c), "cdecl")) {
      Printf(stderr, "Structure %s has a slot that we can't deal with.\n", name);
      Printf(stderr, "nodeType: %s, name: %s, type: %s\n", nodeType(c), Getattr(c, "name"), Getattr(c, "type"));
      SWIG_exit(EXIT_FAILURE);
    }

    String *temp = Copy(Getattr(c, "decl"));
    if (temp) {
      Append(temp, Getattr(c, "type"));	//appending type to the end, otherwise wrong type
      String *lisp_type = get_ffi_type(n, temp);
      Delete(temp);

      String *slot_name = Getattr(c, "sym:name");
      Printf(f_cl, "\n\t(%s %s)", slot_name, lisp_type);

      Append(entries, NewStringf("%s-%s", name, slot_name));

      Delete(lisp_type);
    }
  }

  Printf(f_cl, ")\n");

  /* Add this structure to the known lisp types */
  //Printf(stdout, "Adding %s foreign type\n", name);
  //  add_defined_foreign_type(name);

  return SWIG_OK;
}

/* utilities */
/* returns new string w/ parens stripped */
String *CLISP::strip_parens(String *string) {
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

String *CLISP::convert_literal(String *num_param, String *type) {
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

String *CLISP::get_ffi_type(Node *n, SwigType *ty) {
  Node *node = NewHash();
  Setattr(node, "type", ty);
  Setfile(node, Getfile(n));
  Setline(node, Getline(n));
  const String *tm = Swig_typemap_lookup("in", node, "", 0);
  Delete(node);

  if (tm) {
    return NewString(tm);
  } else if (SwigType_ispointer(ty)) {
    SwigType *cp = Copy(ty);
    SwigType_del_pointer(cp);
    String *inner_type = get_ffi_type(n, cp);

    if (SwigType_isfunction(cp)) {
      return inner_type;
    }

    SwigType *base = SwigType_base(ty);
    String *base_name = SwigType_str(base, 0);

    String *str;
    if (!Strcmp(base_name, "int") || !Strcmp(base_name, "float") || !Strcmp(base_name, "short")
	|| !Strcmp(base_name, "double") || !Strcmp(base_name, "long") || !Strcmp(base_name, "char")) {

      str = NewStringf("(ffi:c-ptr %s)", inner_type);
    } else {
      str = NewStringf("(ffi:c-pointer %s)", inner_type);
    }
    Delete(base_name);
    Delete(base);
    Delete(cp);
    Delete(inner_type);
    return str;
  } else if (SwigType_isarray(ty)) {
    SwigType *cp = Copy(ty);
    String *array_dim = SwigType_array_getdim(ty, 0);

    if (!Strcmp(array_dim, "")) {	//dimension less array convert to pointer
      Delete(array_dim);
      SwigType_del_array(cp);
      SwigType_add_pointer(cp);
      String *str = get_ffi_type(n, cp);
      Delete(cp);
      return str;
    } else {
      SwigType_pop_arrays(cp);
      String *inner_type = get_ffi_type(n, cp);
      Delete(cp);

      int ndim = SwigType_array_ndim(ty);
      String *dimension;
      if (ndim == 1) {
	dimension = array_dim;
      } else {
	dimension = array_dim;
	for (int i = 1; i < ndim; i++) {
	  array_dim = SwigType_array_getdim(ty, i);
	  Append(dimension, " ");
	  Append(dimension, array_dim);
	  Delete(array_dim);
	}
	String *temp = dimension;
	dimension = NewStringf("(%s)", dimension);
	Delete(temp);
      }
      String *str;
      if (is_function)
	str = NewStringf("(ffi:c-ptr (ffi:c-array %s %s))", inner_type, dimension);
      else
	str = NewStringf("(ffi:c-array %s %s)", inner_type, dimension);

      Delete(inner_type);
      Delete(dimension);
      return str;
    }
  } else if (SwigType_isfunction(ty)) {
    SwigType *cp = Copy(ty);
    SwigType *fn = SwigType_pop_function(cp);
    String *args = NewString("");
    ParmList *pl = SwigType_function_parms(fn, n);
    if (ParmList_len(pl) != 0) {
      Printf(args, "(:arguments ");
    }
    int argnum = 0, first = 1;
    for (Parm *p = pl; p; p = nextSibling(p), argnum++) {
      String *argname = Getattr(p, "name");
      SwigType *argtype = Getattr(p, "type");
      String *ffitype = get_ffi_type(n, argtype);

      int tempargname = 0;

      if (!argname) {
	argname = NewStringf("arg%d", argnum);
	tempargname = 1;
      }
      if (!first) {
	Printf(args, "\n\t\t");
      }
      Printf(args, "(%s %s)", argname, ffitype);
      first = 0;
      Delete(ffitype);
      if (tempargname)
	Delete(argname);
    }
    if (ParmList_len(pl) != 0) {
      Printf(args, ")\n");	/* finish arg list */
    }
    String *ffitype = get_ffi_type(n, cp);
    String *str = NewStringf("(ffi:c-function %s \t\t\t\t(:return-type %s))", args, ffitype);
    Delete(fn);
    Delete(args);
    Delete(cp);
    Delete(ffitype);
    return str;
  }
  String *str = SwigType_str(ty, 0);
  if (str) {
    char *st = Strstr(str, "struct");
    if (st) {
      st += 7;
      return NewString(st);
    }
    char *cl = Strstr(str, "class");
    if (cl) {
      cl += 6;
      return NewString(cl);
    }
  }
  return str;
}

extern "C" Language *swig_clisp(void) {
  return new CLISP();
}
