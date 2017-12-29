/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * pike.cxx
 *
 * Pike language module for SWIG.
 * ----------------------------------------------------------------------------- */

/*
 * Notes:
 *
 * - The current approach used for "out" typemaps is inconsistent with
 *   how "out" typemaps are handled by other language modules. Instead
 *   of converting the C/C++ type ($1) to a Pike object type (e.g. a
 *   struct svalue), we're just calling the appropriate push_XXX
 *   (e.g. push_int) to push the return value onto the stack.
 *
 * - Pike classes can't have static member functions or data, so we need
 *   to find some other appropriate mapping for C++ static member functions
 *   and data.
 *
 * - Pike doesn't seem to provide any default way to print the memory
 *   address, etc. for extension objects. Should we do something here?
 *
 */

#include "swigmod.h"

#include <ctype.h>		// for isalnum()

static const char *usage = "\
Pike Options (available with -pike)\n\
     [no additional options]\n\
\n";

class PIKE:public Language {
private:

  File *f_begin;
  File *f_runtime;
  File *f_header;
  File *f_wrappers;
  File *f_init;
  File *f_classInit;

  String *PrefixPlusUnderscore;
  int current;

  // Wrap modes
  enum {
    NO_CPP,
    MEMBER_FUNC,
    CONSTRUCTOR,
    DESTRUCTOR,
    MEMBER_VAR,
    CLASS_CONST,
    STATIC_FUNC,
    STATIC_VAR
  };

public:

  /* ---------------------------------------------------------------------
   * PIKE()
   *
   * Initialize member data
   * --------------------------------------------------------------------- */

   PIKE() {
    f_begin = 0;
    f_runtime = 0;
    f_header = 0;
    f_wrappers = 0;
    f_init = 0;
    f_classInit = 0;
    PrefixPlusUnderscore = 0;
    current = NO_CPP;
  }

  /* ---------------------------------------------------------------------
   * main()
   *
   * Parse command line options and initializes variables.
   * --------------------------------------------------------------------- */
  
   virtual void main(int argc, char *argv[]) {

    /* Set location of SWIG library */
    SWIG_library_directory("pike");

    /* Look for certain command line options */
    for (int i = 1; i < argc; i++) {
      if (argv[i]) {
	if (strcmp(argv[i], "-help") == 0) {
	  fputs(usage, stdout);
	}
      }
    }

    /* Add a symbol to the parser for conditional compilation */
    Preprocessor_define("SWIGPIKE 1", 0);

    /* Set language-specific configuration file */
    SWIG_config_file("pike.swg");

    /* Set typemap language */
    SWIG_typemap_lang("pike");

    /* Enable overloaded methods support */
    allow_overloading();
  }

  /* ---------------------------------------------------------------------
   * top()
   * --------------------------------------------------------------------- */

  virtual int top(Node *n) {
    /* Get the module name */
    String *module = Getattr(n, "name");

    /* Get the output file name */
    String *outfile = Getattr(n, "outfile");

    /* Open the output file */
    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      SWIG_exit(EXIT_FAILURE);
    }
    f_runtime = NewString("");
    f_init = NewString("");
    f_classInit = NewString("");
    f_header = NewString("");
    f_wrappers = NewString("");

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("header", f_header);
    Swig_register_filebyname("wrapper", f_wrappers);
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", f_init);
    Swig_register_filebyname("classInit", f_classInit);

    /* Standard stuff for the SWIG runtime section */
    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGPIKE\n#define SWIGPIKE\n#endif\n\n");

    Printf(f_header, "#define SWIG_init    pike_module_init\n");
    Printf(f_header, "#define SWIG_name    \"%s\"\n\n", module);

    /* Change naming scheme for constructors and destructors */
    Swig_name_register("construct", "%n%c_create");
    Swig_name_register("destroy", "%n%c_destroy");

    /* Current wrap type */
    current = NO_CPP;

    /* Emit code for children */
    Language::top(n);

    /* Close the initialization function */
    Printf(f_init, "}\n");
    SwigType_emit_type_table(f_runtime, f_wrappers);

    /* Close all of the files */
    Dump(f_runtime, f_begin);
    Dump(f_header, f_begin);
    Dump(f_wrappers, f_begin);
    Wrapper_pretty_print(f_init, f_begin);

    Delete(f_header);
    Delete(f_wrappers);
    Delete(f_init);
    Delete(f_classInit);
    Delete(f_runtime);
    Delete(f_begin);

    /* Done */
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * validIdentifier()
   * ------------------------------------------------------------ */

  virtual int validIdentifier(String *s) {
    char *c = Char(s);
    const char *c0 = c;
    const char *c1 = c0 + 1;
    while (*c) {
      if (*c == '`' && c == c0) {
	c++;
	continue;
      }
      if ((*c == '+' || *c == '-' || *c == '*' || *c == '/') && c == c1) {
	c++;
	continue;
      }
      if (!(isalnum(*c) || (*c == '_')))
	return 0;
      c++;
    }
    return 1;
  }

  /* ------------------------------------------------------------
   * importDirective()
   * ------------------------------------------------------------ */

  virtual int importDirective(Node *n) {
    String *modname = Getattr(n, "module");
    if (modname) {
      Printf(f_init, "pike_require(\"%s\");\n", modname);
    }
    return Language::importDirective(n);
  }

  /* ------------------------------------------------------------
   * strip()
   *
   * For names that begin with the current class prefix plus an
   * underscore (e.g. "Foo_enum_test"), return the base function
   * name (i.e. "enum_test").
   * ------------------------------------------------------------ */

  String *strip(const DOHconst_String_or_char_ptr name) {
    String *s = Copy(name);
    if (Strncmp(name, PrefixPlusUnderscore, Len(PrefixPlusUnderscore)) != 0) {
      return s;
    }
    Replaceall(s, PrefixPlusUnderscore, "");
    return s;
  }

  /* ------------------------------------------------------------
   * add_method()
   * ------------------------------------------------------------ */

  void add_method(const DOHconst_String_or_char_ptr name, const DOHconst_String_or_char_ptr function, const DOHconst_String_or_char_ptr description) {
    String *rename = NULL;
    switch (current) {
    case NO_CPP:
      rename = NewString(name);
      Printf(f_init, "ADD_FUNCTION(\"%s\", %s, tFunc(%s), 0);\n", rename, function, description);
      break;
    case STATIC_FUNC:
    case STATIC_VAR:
      rename = NewString(name);
      Printf(f_init, "ADD_FUNCTION(\"%s\", %s, tFunc(%s), 0);\n", rename, function, description);
      break;
    case CONSTRUCTOR:
    case DESTRUCTOR:
    case MEMBER_FUNC:
    case MEMBER_VAR:
      rename = strip(name);
      Printf(f_classInit, "ADD_FUNCTION(\"%s\", %s, tFunc(%s), 0);\n", rename, function, description);
      break;
    case CLASS_CONST: // shouldn't have gotten here for CLASS_CONST nodes
    default: // what is this?
      assert(false);
    }
    Delete(rename);
  }

  /* ---------------------------------------------------------------------
   * functionWrapper()
   *
   * Create a function declaration and register it with the interpreter.
   * --------------------------------------------------------------------- */

  virtual int functionWrapper(Node *n) {

    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *d = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");

    Parm *p;
    String *tm;
    int i;

    String *overname = 0;
    if (Getattr(n, "sym:overloaded")) {
      overname = Getattr(n, "sym:overname");
    } else {
      if (!addSymbol(iname, n))
	return SWIG_ERROR;
    }

    Wrapper *f = NewWrapper();

    // Emit all of the local variables for holding arguments.
    emit_parameter_variables(l, f);

    /* Attach the standard typemaps */
    emit_attach_parmmaps(l, f);
    Setattr(n, "wrap:parms", l);

    /* Get number of required and total arguments */
    int num_arguments = emit_num_arguments(l);
    int varargs = emit_isvarargs(l);

    /* Which input argument to start with? */
    int start = (current == MEMBER_FUNC || current == MEMBER_VAR || current == DESTRUCTOR) ? 1 : 0;

    /* Offset to skip over the attribute name */
    // int offset = (current == MEMBER_VAR) ? 1 : 0;
    int offset = 0;

    String *wname = Swig_name_wrapper(iname);
    if (overname) {
      Append(wname, overname);
    }
    Setattr(n, "wrap:name", wname);

    Printv(f->def, "static void ", wname, "(INT32 args) {", NIL);

    /* Generate code for argument marshalling */
    String *description = NewString("");
    char source[64];
    for (i = 0, p = l; i < num_arguments; i++) {

      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");
      String *ln = Getattr(p, "lname");

      if (i < start) {
	String *lstr = SwigType_lstr(pt, 0);
	Printf(f->code, "%s = (%s) THIS;\n", ln, lstr);
	Delete(lstr);
      } else {
	/* Look for an input typemap */
	sprintf(source, "Pike_sp[%d-args]", i - start + offset);
	if ((tm = Getattr(p, "tmap:in"))) {
	  Replaceall(tm, "$source", source);
	  Replaceall(tm, "$target", ln);
	  Replaceall(tm, "$input", source);
	  Setattr(p, "emit:input", source);
	  Printf(f->code, "%s\n", tm);
	  String *pikedesc = Getattr(p, "tmap:in:pikedesc");
	  if (pikedesc) {
	    Printv(description, pikedesc, " ", NIL);
	  }
	  p = Getattr(p, "tmap:in:next");
	  continue;
	} else {
	  Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
	  break;
	}
      }
      p = nextSibling(p);
    }

    /* Check for trailing varargs */
    if (varargs) {
      if (p && (tm = Getattr(p, "tmap:in"))) {
	Replaceall(tm, "$input", "varargs");
	Printv(f->code, tm, "\n", NIL);
      }
    }

    /* Insert constraint checking code */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:check"))) {
	Replaceall(tm, "$target", Getattr(p, "lname"));
	Printv(f->code, tm, "\n", NIL);
	p = Getattr(p, "tmap:check:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert cleanup code */
    String *cleanup = NewString("");
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:freearg"))) {
	Replaceall(tm, "$source", Getattr(p, "lname"));
	Printv(cleanup, tm, "\n", NIL);
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert argument output code */
    String *outarg = NewString("");
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:argout"))) {
	Replaceall(tm, "$source", Getattr(p, "lname"));
	Replaceall(tm, "$target", "resultobj");
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(outarg, tm, "\n", NIL);
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Emit the function call */
    String *actioncode = emit_action(n);

    /* Clear the return stack */
    Printf(actioncode, "pop_n_elems(args);\n");

    /* Return the function value */
    if (current == CONSTRUCTOR) {
      Printv(actioncode, "THIS = (void *) ", Swig_cresult_name(), ";\n", NIL);
      Printv(description, ", tVoid", NIL);
    } else if (current == DESTRUCTOR) {
      Printv(description, ", tVoid", NIL);
    } else {
      Printv(description, ", ", NIL);
      if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
        actioncode = 0;
	Replaceall(tm, "$source", Swig_cresult_name());
	Replaceall(tm, "$target", "resultobj");
	Replaceall(tm, "$result", "resultobj");
	if (GetFlag(n, "feature:new")) {
	  Replaceall(tm, "$owner", "1");
	} else {
	  Replaceall(tm, "$owner", "0");
	}
	String *pikedesc = Getattr(n, "tmap:out:pikedesc");
	if (pikedesc) {
	  Printv(description, pikedesc, NIL);
	}
	Printf(f->code, "%s\n", tm);
      } else {
	Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(d, 0), name);
      }
    }
    if (actioncode) {
      Append(f->code, actioncode);
      Delete(actioncode);
    }
    emit_return_variable(n, d, f);

    /* Output argument output code */
    Printv(f->code, outarg, NIL);

    /* Output cleanup code */
    Printv(f->code, cleanup, NIL);

    /* Look to see if there is any newfree cleanup code */
    if (GetFlag(n, "feature:new")) {
      if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
	Replaceall(tm, "$source", Swig_cresult_name());
	Printf(f->code, "%s\n", tm);
      }
    }

    /* See if there is any return cleanup code */
    if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
      Replaceall(tm, "$source", Swig_cresult_name());
      Printf(f->code, "%s\n", tm);
    }

    /* Close the function */
    Printf(f->code, "}\n");

    /* Substitute the cleanup code */
    Replaceall(f->code, "$cleanup", cleanup);

    /* Substitute the function name */
    Replaceall(f->code, "$symname", iname);
    Replaceall(f->code, "$result", "resultobj");

    /* Dump the function out */
    Wrapper_print(f, f_wrappers);

    /* Now register the function with the interpreter. */
    if (!Getattr(n, "sym:overloaded")) {
      add_method(iname, wname, description);
    } else {
      if (!Getattr(n, "sym:nextSibling")) {
	dispatchFunction(n);
      }
    }

    Delete(cleanup);
    Delete(outarg);
    Delete(description);
    Delete(wname);
    DelWrapper(f);

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * dispatchFunction()
   *
   * Emit overloading dispatch function
   * ------------------------------------------------------------ */

  void dispatchFunction(Node *n) {
    /* Last node in overloaded chain */

    int maxargs;
    String *tmp = NewString("");
    String *dispatch = Swig_overload_dispatch(n, "%s(args); return;", &maxargs);

    /* Generate a dispatch wrapper for all overloaded functions */

    Wrapper *f = NewWrapper();
    String *symname = Getattr(n, "sym:name");
    String *wname = Swig_name_wrapper(symname);

    Printf(f->def, "static void %s(INT32 args) {", wname);

    Wrapper_add_local(f, "argc", "INT32 argc");
    Printf(tmp, "struct svalue argv[%d]", maxargs);
    Wrapper_add_local(f, "argv", tmp);
    Wrapper_add_local(f, "ii", "INT32 ii");

    Printf(f->code, "argc = args;\n");
    Printf(f->code, "for (ii = 0; (ii < argc) && (ii < %d); ii++) {\n", maxargs);
    Printf(f->code, "argv[ii] = Pike_sp[ii-args];\n");
    Printf(f->code, "}\n");

    Replaceall(dispatch, "$args", "self, args");
    Printv(f->code, dispatch, "\n", NIL);
    Printf(f->code, "Pike_error(\"No matching function for overloaded '%s'.\");\n", symname);
    Printv(f->code, "}\n", NIL);

    Wrapper_print(f, f_wrappers);

    String *description = NewString("");
    Printf(description, "tAny,");
    if (current == CONSTRUCTOR || current == DESTRUCTOR) {
      Printf(description, " tVoid");
    } else {
      String *pd = Getattr(n, "tmap:out:pikedesc");
      if (pd)
	Printf(description, " %s", pd);
    }
    add_method(symname, wname, description);
    Delete(description);

    DelWrapper(f);
    Delete(dispatch);
    Delete(tmp);
    Delete(wname);
  }

  /* ------------------------------------------------------------
   * variableWrapper()
   * ------------------------------------------------------------ */

  virtual int variableWrapper(Node *n) {
    return Language::variableWrapper(n);
  }

  /* ------------------------------------------------------------
   * constantWrapper()
   * ------------------------------------------------------------ */

  virtual int constantWrapper(Node *n) {

    Swig_require("constantWrapper", n, "*sym:name", "type", "value", NIL);

    String *symname = Getattr(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    String *value = Getattr(n, "value");
    bool is_enum_item = (Cmp(nodeType(n), "enumitem") == 0);

    if (SwigType_type(type) == T_MPOINTER) {
      /* Special hook for member pointer */
      String *wname = Swig_name_wrapper(symname);
      Printf(f_header, "static %s = %s;\n", SwigType_str(type, wname), value);
      value = wname;
    } else if (SwigType_type(type) == T_CHAR && is_enum_item) {
      type = NewSwigType(T_INT);
      Setattr(n, "type", type);
    }

    /* Perform constant typemap substitution */
    String *tm = Swig_typemap_lookup("constant", n, value, 0);
    if (tm) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", symname);
      Replaceall(tm, "$symname", symname);
      Replaceall(tm, "$value", value);
      Printf(f_init, "%s\n", tm);
    } else {
      Swig_warning(WARN_TYPEMAP_CONST_UNDEF, input_file, line_number, "Unsupported constant value %s = %s\n", SwigType_str(type, 0), value);
    }

    Swig_restore(n);

    return SWIG_OK;
  }

  /* ------------------------------------------------------------ 
   * nativeWrapper()
   * ------------------------------------------------------------ */

  virtual int nativeWrapper(Node *n) {
    //   return Language::nativeWrapper(n);
    String *name = Getattr(n, "sym:name");
    String *wrapname = Getattr(n, "wrap:name");

    if (!addSymbol(wrapname, n))
      return SWIG_ERROR;

    add_method(name, wrapname, 0);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * enumDeclaration()
   * ------------------------------------------------------------ */

  virtual int enumDeclaration(Node *n) {
    return Language::enumDeclaration(n);
  }

  /* ------------------------------------------------------------
   * enumvalueDeclaration()
   * ------------------------------------------------------------ */

  virtual int enumvalueDeclaration(Node *n) {
    return Language::enumvalueDeclaration(n);
  }

  /* ------------------------------------------------------------
   * classDeclaration()
   * ------------------------------------------------------------ */

  virtual int classDeclaration(Node *n) {
    return Language::classDeclaration(n);
  }

  /* ------------------------------------------------------------
   * classHandler()
   * ------------------------------------------------------------ */

  virtual int classHandler(Node *n) {

    String *symname = Getattr(n, "sym:name");
    if (!addSymbol(symname, n))
      return SWIG_ERROR;

    PrefixPlusUnderscore = NewStringf("%s_", getClassPrefix());

    Printf(f_classInit, "start_new_program();\n");

    /* Handle inheritance */
    List *baselist = Getattr(n, "bases");
    if (baselist && Len(baselist) > 0) {
      Iterator base = First(baselist);
      while (base.item) {
	String *basename = Getattr(base.item, "name");
	SwigType *basetype = NewString(basename);
	SwigType_add_pointer(basetype);
	SwigType_remember(basetype);
	String *basemangle = SwigType_manglestr(basetype);
	Printf(f_classInit, "low_inherit((struct program *) SWIGTYPE%s->clientdata, 0, 0, 0, 0, 0);\n", basemangle);
	Delete(basemangle);
	Delete(basetype);
	base = Next(base);
      }
    } else {
      Printf(f_classInit, "ADD_STORAGE(swig_object_wrapper);\n");
    }

    Language::classHandler(n);

    /* Accessors for member variables */
    /*
       List *membervariables = Getattr(n,"membervariables");
       if (membervariables && Len(membervariables) > 0) {
       membervariableAccessors(membervariables);
       }
     */

    /* Done, close the class and dump its definition to the init function */
    Printf(f_classInit, "add_program_constant(\"%s\", pr = end_program(), 0);\n", symname);
    Dump(f_classInit, f_init);
    Clear(f_classInit);

    SwigType *tt = NewString(symname);
    SwigType_add_pointer(tt);
    SwigType_remember(tt);
    String *tm = SwigType_manglestr(tt);
    Printf(f_init, "SWIG_TypeClientData(SWIGTYPE%s, (void *) pr);\n", tm);
    Delete(tm);
    Delete(tt);

    Delete(PrefixPlusUnderscore);
    PrefixPlusUnderscore = 0;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberfunctionHandler()
   *
   * Method for adding C++ member function
   * ------------------------------------------------------------ */

  virtual int memberfunctionHandler(Node *n) {
    current = MEMBER_FUNC;
    Language::memberfunctionHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constructorHandler()
   *
   * Method for adding C++ member constructor
   * ------------------------------------------------------------ */

  virtual int constructorHandler(Node *n) {
    current = CONSTRUCTOR;
    Language::constructorHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * destructorHandler()
   * ------------------------------------------------------------ */

  virtual int destructorHandler(Node *n) {
    current = DESTRUCTOR;
    Language::destructorHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * membervariableAccessors()
   * ------------------------------------------------------------ */

  void membervariableAccessors(List *membervariables) {
    String *name;
    Iterator i;
    bool need_setter;
    String *funcname;

    /* If at least one of them is mutable, we need a setter */
    need_setter = false;
    i = First(membervariables);
    while (i.item) {
      if (!GetFlag(i.item, "feature:immutable")) {
	need_setter = true;
	break;
      }
      i = Next(i);
    }

    /* Create a function to set the values of the (mutable) variables */
    if (need_setter) {
      Wrapper *wrapper = NewWrapper();
      String *setter = Swig_name_member(NSPACE_TODO, getClassPrefix(), "`->=");
      String *wname = Swig_name_wrapper(setter);
      Printv(wrapper->def, "static void ", wname, "(INT32 args) {", NIL);
      Printf(wrapper->locals, "char *name = (char *) STR0(Pike_sp[0-args].u.string);\n");

      i = First(membervariables);
      while (i.item) {
	if (!GetFlag(i.item, "feature:immutable")) {
	  name = Getattr(i.item, "name");
	  funcname = Swig_name_wrapper(Swig_name_set(NSPACE_TODO, Swig_name_member(NSPACE_TODO, getClassPrefix(), name)));
	  Printf(wrapper->code, "if (!strcmp(name, \"%s\")) {\n", name);
	  Printf(wrapper->code, "%s(args);\n", funcname);
	  Printf(wrapper->code, "return;\n");
	  Printf(wrapper->code, "}\n");
	  Delete(funcname);
	}
	i = Next(i);
      }

      /* Close the function */
      Printf(wrapper->code, "pop_n_elems(args);\n");
      Printf(wrapper->code, "}\n");

      /* Dump wrapper code to the output file */
      Wrapper_print(wrapper, f_wrappers);

      /* Register it with Pike */
      String *description = NewString("tStr tFloat, tVoid");
      add_method("`->=", wname, description);
      Delete(description);

      /* Clean up */
      Delete(wname);
      Delete(setter);
      DelWrapper(wrapper);
    }

    /* Create a function to get the values of the (mutable) variables */
    Wrapper *wrapper = NewWrapper();
    String *getter = Swig_name_member(NSPACE_TODO, getClassPrefix(), "`->");
    String *wname = Swig_name_wrapper(getter);
    Printv(wrapper->def, "static void ", wname, "(INT32 args) {", NIL);
    Printf(wrapper->locals, "char *name = (char *) STR0(Pike_sp[0-args].u.string);\n");

    i = First(membervariables);
    while (i.item) {
      name = Getattr(i.item, "name");
      funcname = Swig_name_wrapper(Swig_name_get(NSPACE_TODO, Swig_name_member(NSPACE_TODO, getClassPrefix(), name)));
      Printf(wrapper->code, "if (!strcmp(name, \"%s\")) {\n", name);
      Printf(wrapper->code, "%s(args);\n", funcname);
      Printf(wrapper->code, "return;\n");
      Printf(wrapper->code, "}\n");
      Delete(funcname);
      i = Next(i);
    }

    /* Close the function */
    Printf(wrapper->code, "pop_n_elems(args);\n");
    Printf(wrapper->code, "}\n");

    /* Dump wrapper code to the output file */
    Wrapper_print(wrapper, f_wrappers);

    /* Register it with Pike */
    String *description = NewString("tStr, tMix");
    add_method("`->", wname, description);
    Delete(description);

    /* Clean up */
    Delete(wname);
    Delete(getter);
    DelWrapper(wrapper);
  }

  /* ------------------------------------------------------------
   * membervariableHandler()
   * ------------------------------------------------------------ */

  virtual int membervariableHandler(Node *n) {
    List *membervariables = Getattr(getCurrentClass(), "membervariables");
    if (!membervariables) {
      membervariables = NewList();
      Setattr(getCurrentClass(), "membervariables", membervariables);
    }
    Append(membervariables, n);
    current = MEMBER_VAR;
    Language::membervariableHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------
   * staticmemberfunctionHandler()
   *
   * Wrap a static C++ function
   * ---------------------------------------------------------------------- */

  virtual int staticmemberfunctionHandler(Node *n) {
    current = STATIC_FUNC;
    Language::staticmemberfunctionHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberconstantHandler()
   *
   * Create a C++ constant
   * ------------------------------------------------------------ */

  virtual int memberconstantHandler(Node *n) {
    current = CLASS_CONST;
    constantWrapper(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------
   * staticmembervariableHandler()
   * --------------------------------------------------------------------- */

  virtual int staticmembervariableHandler(Node *n) {
    current = STATIC_VAR;
    Language::staticmembervariableHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }
};

/* -----------------------------------------------------------------------------
 * swig_pike()    - Instantiate module
 * ----------------------------------------------------------------------------- */

static Language *new_swig_pike() {
  return new PIKE();
}
extern "C" Language *swig_pike(void) {
  return new_swig_pike();
}
