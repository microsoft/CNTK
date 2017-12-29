/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * tcl8.cxx
 *
 * Tcl8 language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"

static const char *usage = "\
Tcl 8 Options (available with -tcl)\n\
     -itcl           - Enable ITcl support\n\
     -nosafe         - Leave out SafeInit module function.\n\
     -prefix <name>  - Set a prefix <name> to be prepended to all names\n\
     -namespace      - Build module into a Tcl 8 namespace\n\
     -pkgversion     - Set package version\n\n";

static String *cmd_tab = 0;	/* Table of command names    */
static String *var_tab = 0;	/* Table of global variables */
static String *const_tab = 0;	/* Constant table            */
static String *methods_tab = 0;	/* Methods table             */
static String *attr_tab = 0;	/* Attribute table           */
static String *prefix = 0;
static String *module = 0;
static int namespace_option = 0;
static String *init_name = 0;
static String *ns_name = 0;
static int have_constructor;
static String *constructor_name;
static int have_destructor;
static int have_base_classes;
static String *destructor_action = 0;
static String *version = (String *) "0.0";
static String *class_name = 0;

static int have_attributes;
static int have_methods;
static int nosafe = 0;

static File *f_header = 0;
static File *f_wrappers = 0;
static File *f_init = 0;
static File *f_begin = 0;
static File *f_runtime = 0;


//  Itcl support
static int itcl = 0;
static File *f_shadow = 0;
static File *f_shadow_stubs = 0;

static String *constructor = 0;
static String *destructor = 0;
static String *base_classes = 0;
static String *base_class_init = 0;
static String *methods = 0;
static String *imethods = 0;
static String *attributes = 0;
static String *attribute_traces = 0;
static String *iattribute_traces = 0;



class TCL8:public Language {
public:

  /* ------------------------------------------------------------
   * TCL8::main()
   * ------------------------------------------------------------ */

  virtual void main(int argc, char *argv[]) {
    int cppcast = 1;

     SWIG_library_directory("tcl");

    for (int i = 1; i < argc; i++) {
      if (argv[i]) {
	if (strcmp(argv[i], "-prefix") == 0) {
	  if (argv[i + 1]) {
	    prefix = NewString(argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else
	     Swig_arg_error();
	} else if (strcmp(argv[i], "-pkgversion") == 0) {
	  if (argv[i + 1]) {
	    version = NewString(argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  }
	} else if (strcmp(argv[i], "-namespace") == 0) {
	  namespace_option = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-itcl") == 0) {
	  itcl = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-nosafe") == 0) {
	  nosafe = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-cppcast") == 0) {
	  cppcast = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-nocppcast") == 0) {
	  cppcast = 0;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-help") == 0) {
	  fputs(usage, stdout);
	}
      }
    }

    if (cppcast) {
      Preprocessor_define((DOH *) "SWIG_CPLUSPLUS_CAST", 0);
    }

    Preprocessor_define("SWIGTCL 1", 0);
    // SWIGTCL8 is deprecated, and no longer documented.
    Preprocessor_define("SWIGTCL8 1", 0);
    SWIG_typemap_lang("tcl8");
    SWIG_config_file("tcl8.swg");
    allow_overloading();
  }

  /* ------------------------------------------------------------
   * top()
   * ------------------------------------------------------------ */

  virtual int top(Node *n) {

    /* Initialize all of the output files */
    String *outfile = Getattr(n, "outfile");

    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      SWIG_exit(EXIT_FAILURE);
    }
    f_runtime = NewString("");
    f_init = NewString("");
    f_header = NewString("");
    f_wrappers = NewString("");

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("header", f_header);
    Swig_register_filebyname("wrapper", f_wrappers);
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", f_init);

    /* Initialize some variables for the object interface */

    cmd_tab = NewString("");
    var_tab = NewString("");
    methods_tab = NewString("");
    const_tab = NewString("");

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGTCL\n#define SWIGTCL\n#endif\n\n");

    /* Set the module name, namespace, and prefix */

    module = NewStringf("%(lower)s", Getattr(n, "name"));
    init_name = NewStringf("%(title)s_Init", module);

    ns_name = prefix ? Copy(prefix) : Copy(module);
    if (prefix)
      Append(prefix, "_");


    /* If shadow classing is enabled, we're going to change the module name to "_module" */
    if (itcl) {
      String *filen;
      filen = NewStringf("%s%s.itcl", SWIG_output_directory(), module);

      Insert(module, 0, "_");

      if ((f_shadow = NewFile(filen, "w", SWIG_output_files())) == 0) {
	FileErrorDisplay(filen);
	SWIG_exit(EXIT_FAILURE);
      }
      f_shadow_stubs = NewString("");

      Swig_register_filebyname("shadow", f_shadow);
      Swig_register_filebyname("itcl", f_shadow);

      Swig_banner_target_lang(f_shadow, "#");

      Printv(f_shadow, "\npackage require Itcl\n\n", NIL);
      Delete(filen);
    }

    /* Generate some macros used throughout code generation */

    Printf(f_header, "#define SWIG_init    %s\n", init_name);
    Printf(f_header, "#define SWIG_name    \"%s\"\n", module);
    if (namespace_option) {
      Printf(f_header, "#define SWIG_prefix  \"%s::\"\n", ns_name);
      Printf(f_header, "#define SWIG_namespace \"%s\"\n\n", ns_name);
    } else {
      Printf(f_header, "#define SWIG_prefix  \"%s\"\n", prefix);
    }
    Printf(f_header, "#define SWIG_version \"%s\"\n", version);

    Printf(cmd_tab, "\nstatic swig_command_info swig_commands[] = {\n");
    Printf(var_tab, "\nstatic swig_var_info swig_variables[] = {\n");
    Printf(const_tab, "\nstatic swig_const_info swig_constants[] = {\n");

    Printf(f_wrappers, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n");

    /* Start emitting code */
    Language::top(n);

    /* Done.  Close up the module */
    Printv(cmd_tab, tab4, "{0, 0, 0}\n", "};\n", NIL);
    Printv(var_tab, tab4, "{0,0,0,0}\n", "};\n", NIL);
    Printv(const_tab, tab4, "{0,0,0,0,0,0}\n", "};\n", NIL);

    Printv(f_wrappers, cmd_tab, var_tab, const_tab, NIL);

    /* Dump the pointer equivalency table */
    SwigType_emit_type_table(f_runtime, f_wrappers);

    Printf(f_wrappers, "#ifdef __cplusplus\n}\n#endif\n");

    /* Close the init function and quit */
    Printf(f_init, "return TCL_OK;\n}\n");

    if (!nosafe) {
      Printf(f_init, "SWIGEXPORT int %(title)s_SafeInit(Tcl_Interp *interp) {\n", module);
      Printf(f_init, "    return SWIG_init(interp);\n");
      Printf(f_init, "}\n");
    }

    if (itcl) {
      Printv(f_shadow, f_shadow_stubs, "\n", NIL);
      Delete(f_shadow);
    }

    /* Close all of the files */
    Dump(f_runtime, f_begin);
    Printv(f_begin, f_header, f_wrappers, NIL);
    Wrapper_pretty_print(f_init, f_begin);
    Delete(f_header);
    Delete(f_wrappers);
    Delete(f_init);
    Delete(f_runtime);
    Delete(f_begin);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * functionWrapper()
   * ------------------------------------------------------------ */

  virtual int functionWrapper(Node *n) {
    String *name = Getattr(n, "name");	/* Like to get rid of this */
    String *iname = Getattr(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    ParmList *parms = Getattr(n, "parms");
    String *overname = 0;

    Parm *p;
    int i;
    String *tm;
    Wrapper *f;
    String *incode, *cleanup, *outarg, *argstr, *args;
    int num_arguments = 0;
    int num_required = 0;
    int varargs = 0;

    char source[64];

    if (Getattr(n, "sym:overloaded")) {
      overname = Getattr(n, "sym:overname");
    } else {
      if (!addSymbol(iname, n))
	return SWIG_ERROR;
    }

    incode = NewString("");
    cleanup = NewString("");
    outarg = NewString("");
    argstr = NewString("\"");
    args = NewString("");

    f = NewWrapper();

#ifdef SWIG_USE_RESULTOBJ
    Wrapper_add_local(f, "resultobj", "Tcl_Obj *resultobj = NULL");
#endif


    String *wname = Swig_name_wrapper(iname);
    if (overname) {
      Append(wname, overname);
    }
    Setattr(n, "wrap:name", wname);

    Printv(f->def, "SWIGINTERN int\n ", wname, "(ClientData clientData SWIGUNUSED, Tcl_Interp *interp, int objc, Tcl_Obj *CONST objv[]) {", NIL);

    // Emit all of the local variables for holding arguments.
    emit_parameter_variables(parms, f);

    /* Attach standard typemaps */
    emit_attach_parmmaps(parms, f);
    Setattr(n, "wrap:parms", parms);

    /* Get number of require and total arguments */
    num_arguments = emit_num_arguments(parms);
    num_required = emit_num_required(parms);
    varargs = emit_isvarargs(parms);

    /* Unmarshal parameters */

    for (i = 0, p = parms; i < num_arguments; i++) {
      /* Skip ignored arguments */

      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");
      String *ln = Getattr(p, "lname");

      /* Produce string representations of the source and target arguments */
      sprintf(source, "objv[%d]", i + 1);

      if (i == num_required)
	Putc('|', argstr);
      if ((tm = Getattr(p, "tmap:in"))) {
	String *parse = Getattr(p, "tmap:in:parse");
	if (!parse) {
	  Replaceall(tm, "$target", ln);
	  Replaceall(tm, "$source", source);
	  Replaceall(tm, "$input", source);
	  Setattr(p, "emit:input", source);

	  if (Getattr(p, "wrap:disown") || (Getattr(p, "tmap:in:disown"))) {
	    Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
	  } else {
	    Replaceall(tm, "$disown", "0");
	  }

	  Putc('o', argstr);
	  Printf(args, ",(void *)0");
	  if (i >= num_required) {
	    Printf(incode, "if (objc > %d) {\n", i + 1);
	  }
	  Printf(incode, "%s\n", tm);
	  if (i >= num_required) {
	    Printf(incode, "}\n");
	  }
	} else {
	  Printf(argstr, "%s", parse);
	  Printf(args, ",&%s", ln);
	  if (Strcmp(parse, "p") == 0) {
	    SwigType *lt = SwigType_ltype(pt);
	    SwigType_remember(pt);
	    if (Cmp(lt, "p.void") == 0) {
	      Printf(args, ",(void *)0");
	    } else {
	      Printf(args, ",SWIGTYPE%s", SwigType_manglestr(pt));
	    }
	    Delete(lt);
	  }
	}
	p = Getattr(p, "tmap:in:next");
	continue;
      } else {
	Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
      }
      p = nextSibling(p);
    }

    if (!varargs) {
      Putc(':', argstr);
    } else {
      Putc(';', argstr);
      /* If variable length arguments we need to emit the in typemap here */
      if (p && (tm = Getattr(p, "tmap:in"))) {
	sprintf(source, "objv[%d]", i + 1);
	Printf(incode, "if (objc > %d) {\n", i);
	Replaceall(tm, "$input", source);
	Printv(incode, tm, "\n", NIL);
	Printf(incode, "}\n");
      }
    }

    Printf(argstr, "%s\"", usage_string(Char(iname), type, parms));

    Printv(f->code, "if (SWIG_GetArgs(interp, objc, objv,", argstr, args, ") == TCL_ERROR) SWIG_fail;\n", NIL);

    Printv(f->code, incode, NIL);

    /* Insert constraint checking code */
    for (p = parms; p;) {
      if ((tm = Getattr(p, "tmap:check"))) {
	Replaceall(tm, "$target", Getattr(p, "lname"));
	Printv(f->code, tm, "\n", NIL);
	p = Getattr(p, "tmap:check:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert cleanup code */
    for (i = 0, p = parms; p; i++) {
      if (!checkAttribute(p, "tmap:in:numinputs", "0")
	  && !Getattr(p, "tmap:in:parse") && (tm = Getattr(p, "tmap:freearg"))) {
	if (Len(tm) != 0) {
	  Replaceall(tm, "$source", Getattr(p, "lname"));
	  Printv(cleanup, tm, "\n", NIL);
	}
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert argument output code */
    for (i = 0, p = parms; p; i++) {
      if ((tm = Getattr(p, "tmap:argout"))) {
	Replaceall(tm, "$source", Getattr(p, "lname"));
#ifdef SWIG_USE_RESULTOBJ
	Replaceall(tm, "$target", "resultobj");
	Replaceall(tm, "$result", "resultobj");
#else
	Replaceall(tm, "$target", "(Tcl_GetObjResult(interp))");
	Replaceall(tm, "$result", "(Tcl_GetObjResult(interp))");
#endif
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(outarg, tm, "\n", NIL);
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Now write code to make the function call */
    String *actioncode = emit_action(n);

    /* Need to redo all of this code (eventually) */

    /* Return value if necessary  */
    if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
      Replaceall(tm, "$source", Swig_cresult_name());
#ifdef SWIG_USE_RESULTOBJ
      Replaceall(tm, "$target", "resultobj");
      Replaceall(tm, "$result", "resultobj");
#else
      Replaceall(tm, "$target", "(Tcl_GetObjResult(interp))");
      Replaceall(tm, "$result", "(Tcl_GetObjResult(interp))");
#endif
      if (GetFlag(n, "feature:new")) {
	Replaceall(tm, "$owner", "SWIG_POINTER_OWN");
      } else {
	Replaceall(tm, "$owner", "0");
      }
      Printf(f->code, "%s\n", tm);
    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(type, 0), name);
    }
    emit_return_variable(n, type, f);

    /* Dump output argument code */
    Printv(f->code, outarg, NIL);

    /* Dump the argument cleanup code */
    Printv(f->code, cleanup, NIL);

    /* Look for any remaining cleanup */
    if (GetFlag(n, "feature:new")) {
      if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
	Replaceall(tm, "$source", Swig_cresult_name());
	Printf(f->code, "%s\n", tm);
      }
    }

    if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
      Replaceall(tm, "$source", Swig_cresult_name());
      Printf(f->code, "%s\n", tm);
    }
#ifdef SWIG_USE_RESULTOBJ
    Printv(f->code, "if (resultobj) Tcl_SetObjResult(interp, resultobj);\n", NIL);
#endif
    Printv(f->code, "return TCL_OK;\n", NIL);
    Printv(f->code, "fail:\n", cleanup, "return TCL_ERROR;\n", NIL);
    Printv(f->code, "}\n", NIL);

    /* Substitute the cleanup code */
    Replaceall(f->code, "$cleanup", cleanup);
    Replaceall(f->code, "$symname", iname);

    /* Dump out the function */
    Wrapper_print(f, f_wrappers);

    if (!Getattr(n, "sym:overloaded")) {
      /* Register the function with Tcl */
      Printv(cmd_tab, tab4, "{ SWIG_prefix \"", iname, "\", (swig_wrapper_func) ", Swig_name_wrapper(iname), ", NULL},\n", NIL);
    } else {
      if (!Getattr(n, "sym:nextSibling")) {
	/* Emit overloading dispatch function */

	int maxargs;
	String *dispatch = Swig_overload_dispatch(n, "return %s(clientData, interp, objc, argv - 1);", &maxargs);

	/* Generate a dispatch wrapper for all overloaded functions */

	Wrapper *df = NewWrapper();
	String *dname = Swig_name_wrapper(iname);

	Printv(df->def, "SWIGINTERN int\n", dname, "(ClientData clientData SWIGUNUSED, Tcl_Interp *interp, int objc, Tcl_Obj *CONST objv[]) {", NIL);
	Printf(df->code, "Tcl_Obj *CONST *argv = objv+1;\n");
	Printf(df->code, "int argc = objc-1;\n");
	Printv(df->code, dispatch, "\n", NIL);
	Node *sibl = n;
	while (Getattr(sibl, "sym:previousSibling"))
	  sibl = Getattr(sibl, "sym:previousSibling");	// go all the way up
	String *protoTypes = NewString("");
	do {
	  String *fulldecl = Swig_name_decl(sibl);
	  Printf(protoTypes, "\n\"    %s\\n\"", fulldecl);
	  Delete(fulldecl);
	} while ((sibl = Getattr(sibl, "sym:nextSibling")));
	Printf(df->code, "Tcl_SetResult(interp,(char *) "
	       "\"Wrong number or type of arguments for overloaded function '%s'.\\n\""
	       "\n\"  Possible C/C++ prototypes are:\\n\"%s, TCL_STATIC);\n", iname, protoTypes);
	Delete(protoTypes);
	Printf(df->code, "return TCL_ERROR;\n");
	Printv(df->code, "}\n", NIL);
	Wrapper_print(df, f_wrappers);
	Printv(cmd_tab, tab4, "{ SWIG_prefix \"", iname, "\", (swig_wrapper_func) ", dname, ", NULL},\n", NIL);
	DelWrapper(df);
	Delete(dispatch);
	Delete(dname);
      }
    }

    Delete(incode);
    Delete(cleanup);
    Delete(outarg);
    Delete(argstr);
    Delete(args);
    DelWrapper(f);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * variableWrapper()
   * ------------------------------------------------------------ */

  virtual int variableWrapper(Node *n) {

    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *t = Getattr(n, "type");

    String *setname = 0;
    String *setfname = 0;
    Wrapper *setf = 0, *getf = 0;
    int readonly = 0;
    String *tm;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    /* Create a function for getting a variable */
    int addfail = 0;
    getf = NewWrapper();
    String *getname = Swig_name_get(NSPACE_TODO, iname);
    String *getfname = Swig_name_wrapper(getname);
    Setattr(n, "wrap:name", getfname);
    Printv(getf->def, "SWIGINTERN const char *", getfname, "(ClientData clientData SWIGUNUSED, Tcl_Interp *interp, char *name1, char *name2, int flags) {", NIL);
    Wrapper_add_local(getf, "value", "Tcl_Obj *value = 0");
    if ((tm = Swig_typemap_lookup("varout", n, name, 0))) {
      Replaceall(tm, "$source", name);
      Replaceall(tm, "$target", "value");
      Replaceall(tm, "$result", "value");
      /* Printf(getf->code, "%s\n",tm); */
      addfail = emit_action_code(n, getf->code, tm);
      Printf(getf->code, "if (value) {\n");
      Printf(getf->code, "Tcl_SetVar2(interp,name1,name2,Tcl_GetStringFromObj(value,NULL), flags);\n");
      Printf(getf->code, "Tcl_DecrRefCount(value);\n");
      Printf(getf->code, "}\n");
      Printf(getf->code, "return NULL;\n");
      if (addfail) {
	Append(getf->code, "fail:\n");
	Printf(getf->code, "return \"%s\";\n", iname);
      }
      Printf(getf->code, "}\n");
      Wrapper_print(getf, f_wrappers);
    } else {
      Swig_warning(WARN_TYPEMAP_VAROUT_UNDEF, input_file, line_number, "Unable to read variable of type %s\n", SwigType_str(t, 0));
      DelWrapper(getf);
      return SWIG_NOWRAP;
    }
    DelWrapper(getf);

    /* Try to create a function setting a variable */
    if (is_assignable(n)) {
      setf = NewWrapper();
      setname = Swig_name_set(NSPACE_TODO, iname);
      setfname = Swig_name_wrapper(setname);
      Setattr(n, "wrap:name", setfname);
      if (setf) {
        Printv(setf->def, "SWIGINTERN const char *", setfname,
	     "(ClientData clientData SWIGUNUSED, Tcl_Interp *interp, char *name1, char *name2 SWIGUNUSED, int flags) {", NIL);
        Wrapper_add_local(setf, "value", "Tcl_Obj *value = 0");
        Wrapper_add_local(setf, "name1o", "Tcl_Obj *name1o = 0");

        if ((tm = Swig_typemap_lookup("varin", n, name, 0))) {
	  Replaceall(tm, "$source", "value");
	  Replaceall(tm, "$target", name);
	  Replaceall(tm, "$input", "value");
	  Printf(setf->code, "name1o = Tcl_NewStringObj(name1,-1);\n");
	  Printf(setf->code, "value = Tcl_ObjGetVar2(interp, name1o, 0, flags);\n");
	  Printf(setf->code, "Tcl_DecrRefCount(name1o);\n");
	  Printf(setf->code, "if (!value) SWIG_fail;\n");
	  /* Printf(setf->code,"%s\n", tm); */
	  emit_action_code(n, setf->code, tm);
	  Printf(setf->code, "return NULL;\n");
	  Printf(setf->code, "fail:\n");
	  Printf(setf->code, "return \"%s\";\n", iname);
	  Printf(setf->code, "}\n");
	  Wrapper_print(setf, f_wrappers);
        } else {
	  Swig_warning(WARN_TYPEMAP_VARIN_UNDEF, input_file, line_number, "Unable to set variable of type %s.\n", SwigType_str(t, 0));
	  readonly = 1;
        }
      }
      DelWrapper(setf);
    } else {
      readonly = 1;
    }


    Printv(var_tab, tab4, "{ SWIG_prefix \"", iname, "\", 0, (swig_variable_func) ", getfname, ",", NIL);
    if (readonly) {
      static int readonlywrap = 0;
      if (!readonlywrap) {
	Wrapper *ro = NewWrapper();
	Printf(ro->def,
	       "SWIGINTERN const char *swig_readonly(ClientData clientData SWIGUNUSED, Tcl_Interp *interp SWIGUNUSED, char *name1 SWIGUNUSED, char *name2 SWIGUNUSED, int flags SWIGUNUSED) {");
	Printv(ro->code, "return \"Variable is read-only\";\n", "}\n", NIL);
	Wrapper_print(ro, f_wrappers);
	readonlywrap = 1;
	DelWrapper(ro);
      }
      Printf(var_tab, "(swig_variable_func) swig_readonly},\n");
    } else {
      Printv(var_tab, "(swig_variable_func) ", setfname, "},\n", NIL);
    }
    Delete(getfname);
    Delete(setfname);
    Delete(setname);
    Delete(getname);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constantWrapper()
   * ------------------------------------------------------------ */

  virtual int constantWrapper(Node *n) {
    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    String *nsname = !namespace_option ? Copy(iname) : NewStringf("%s::%s", ns_name, iname);
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");
    String *tm;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;
    if (namespace_option)
      Setattr(n, "sym:name", nsname);

    /* Special hook for member pointer */
    if (SwigType_type(type) == T_MPOINTER) {
      String *wname = Swig_name_wrapper(iname);
      Printf(f_wrappers, "static %s = %s;\n", SwigType_str(type, wname), value);
      value = Char(wname);
    }

    if ((tm = Swig_typemap_lookup("consttab", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", name);
      Replaceall(tm, "$value", value);
      Replaceall(tm, "$nsname", nsname);
      Printf(const_tab, "%s,\n", tm);
    } else if ((tm = Swig_typemap_lookup("constcode", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", name);
      Replaceall(tm, "$value", value);
      Replaceall(tm, "$nsname", nsname);
      Printf(f_init, "%s\n", tm);
    } else {
      Delete(nsname);
      Swig_warning(WARN_TYPEMAP_CONST_UNDEF, input_file, line_number, "Unsupported constant value.\n");
      return SWIG_NOWRAP;
    }
    Delete(nsname);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * nativeWrapper()
   * ------------------------------------------------------------ */

  virtual int nativeWrapper(Node *n) {
    String *name = Getattr(n, "sym:name");
    String *funcname = Getattr(n, "wrap:name");
    if (!addSymbol(funcname, n))
      return SWIG_ERROR;

    Printf(f_init, "\t Tcl_CreateObjCommand(interp, SWIG_prefix \"%s\", (swig_wrapper_func) %s, (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);\n", name,
	   funcname);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * classHandler()
   * ------------------------------------------------------------ */

  virtual int classHandler(Node *n) {
    static Hash *emitted = NewHash();
    String *mangled_classname = 0;
    String *real_classname = 0;

    have_constructor = 0;
    have_destructor = 0;
    destructor_action = 0;
    constructor_name = 0;

    if (itcl) {
      constructor = NewString("");
      destructor = NewString("");
      base_classes = NewString("");
      base_class_init = NewString("");
      methods = NewString("");
      imethods = NewString("");
      attributes = NewString("");
      attribute_traces = NewString("");
      iattribute_traces = NewString("");

      have_base_classes = 0;
      have_methods = 0;
      have_attributes = 0;
    }

    class_name = Getattr(n, "sym:name");
    if (!addSymbol(class_name, n))
      return SWIG_ERROR;

    real_classname = Getattr(n, "name");
    mangled_classname = Swig_name_mangle(real_classname);

    if (Getattr(emitted, mangled_classname))
      return SWIG_NOWRAP;
    Setattr(emitted, mangled_classname, "1");

    attr_tab = NewString("");
    Printf(attr_tab, "static swig_attribute swig_");
    Printv(attr_tab, mangled_classname, "_attributes[] = {\n", NIL);

    methods_tab = NewStringf("");
    Printf(methods_tab, "static swig_method swig_");
    Printv(methods_tab, mangled_classname, "_methods[] = {\n", NIL);

    /* Generate normal wrappers */
    Language::classHandler(n);

    SwigType *t = Copy(Getattr(n, "name"));
    SwigType_add_pointer(t);

    // Catch all: eg. a class with only static functions and/or variables will not have 'remembered'
    // SwigType_remember(t);
    String *wrap_class = NewStringf("&_wrap_class_%s", mangled_classname);
    SwigType_remember_clientdata(t, wrap_class);

    String *rt = Copy(getClassType());
    SwigType_add_pointer(rt);

    // Register the class structure with the type checker
    /*    Printf(f_init,"SWIG_TypeClientData(SWIGTYPE%s, (void *) &_wrap_class_%s);\n", SwigType_manglestr(t), mangled_classname); */
    if (have_destructor) {
      Printv(f_wrappers, "SWIGINTERN void swig_delete_", class_name, "(void *obj) {\n", NIL);
      if (destructor_action) {
	Printv(f_wrappers, SwigType_str(rt, "arg1"), " = (", SwigType_str(rt, 0), ") obj;\n", NIL);
	Printv(f_wrappers, destructor_action, "\n", NIL);
      } else {
	if (CPlusPlus) {
	  Printv(f_wrappers, "    delete (", SwigType_str(rt, 0), ") obj;\n", NIL);
	} else {
	  Printv(f_wrappers, "    free((char *) obj);\n", NIL);
	}
      }
      Printf(f_wrappers, "}\n");
    }

    Printf(methods_tab, "    {0,0}\n};\n");
    Printv(f_wrappers, methods_tab, NIL);

    Printf(attr_tab, "    {0,0,0}\n};\n");
    Printv(f_wrappers, attr_tab, NIL);

    /* Handle inheritance */

    String *base_class = NewString("");
    String *base_class_names = NewString("");

    if (itcl) {
      base_classes = NewString("");
    }

    List *baselist = Getattr(n, "bases");
    if (baselist && Len(baselist)) {
      Iterator b;
      int index = 0;
      b = First(baselist);
      while (b.item) {
	String *bname = Getattr(b.item, "name");
	if ((!bname) || GetFlag(b.item, "feature:ignore") || (!Getattr(b.item, "module"))) {
	  b = Next(b);
	  continue;
	}
	if (itcl) {
	  have_base_classes = 1;
	  Printv(base_classes, bname, " ", NIL);
	  Printv(base_class_init, "    ", bname, "Ptr::constructor $ptr\n", NIL);
	}
	String *bmangle = Swig_name_mangle(bname);
	//      Printv(f_wrappers,"extern swig_class _wrap_class_", bmangle, ";\n", NIL);
	//      Printf(base_class,"&_wrap_class_%s",bmangle);
	Printf(base_class, "0");
	Printf(base_class_names, "\"%s *\",", SwigType_namestr(bname));
	/* Put code to register base classes in init function */

	//Printf(f_init,"/* Register base : %s */\n", bmangle);
	//Printf(f_init,"swig_%s_bases[%d] = (swig_class *) SWIG_TypeQuery(\"%s *\")->clientdata;\n",  mangled_classname, index, SwigType_namestr(bname));
	b = Next(b);
	index++;
	Putc(',', base_class);
	Delete(bmangle);
      }
    }

    if (itcl) {
      String *ptrclass = NewString("");

      // First, build the pointer base class
      Printv(ptrclass, "itcl::class ", class_name, "Ptr {\n", NIL);
      if (have_base_classes)
	Printv(ptrclass, "  inherit ", base_classes, "\n", NIL);

      //  Define protected variables for SWIG object pointer
      Printv(ptrclass, "  protected variable swigobj\n", "  protected variable thisown\n", NIL);

      //  Define public variables
      if (have_attributes) {
	Printv(ptrclass, attributes, NIL);

	// base class swig_getset was being called for complex inheritance trees
	if (namespace_option) {

	  Printv(ptrclass, "  protected method ", class_name, "_swig_getset {var name1 name2 op} {\n", NIL);

	  Printv(ptrclass,
		 "    switch -exact -- $op {\n",
		 "      r {set $var [", ns_name, "::", class_name, "_[set var]_get $swigobj]}\n",
		 "      w {", ns_name, "::", class_name, "_${var}_set $swigobj [set $var]}\n", "    }\n", "  }\n", NIL);
	} else {
	  Printv(ptrclass,
		 "  protected method ", class_name, "_swig_getset {var name1 name2 op} {\n",
		 "    switch -exact -- $op {\n",
		 "      r {set $var [", class_name, "_[set var]_get $swigobj]}\n",
		 "      w {", class_name, "_${var}_set $swigobj [set $var]}\n", "    }\n", "  }\n", NIL);
	}
      }
      //  Add the constructor, which may include
      //  calls to base class class constructors

      Printv(ptrclass, "  constructor { ptr } {\n", NIL);
      if (have_base_classes) {
	Printv(ptrclass, base_class_init, NIL);
	Printv(ptrclass, "  } {\n", NIL);
      }

      Printv(ptrclass, "    set swigobj $ptr\n", "    set thisown 0\n", NIL);

      if (have_attributes) {
	Printv(ptrclass, attribute_traces, NIL);
      }
      Printv(ptrclass, "  }\n", NIL);


      //  Add destructor
      Printv(ptrclass, "  destructor {\n",
	     "    set d_func delete_", class_name, "\n",
	     "    if { $thisown && ([info command $d_func] != \"\") } {\n" "      $d_func $swigobj\n", "    }\n", "  }\n", NIL);

      //  Add methods
      if (have_methods) {
	Printv(ptrclass, imethods, NIL);
      };

      //  Close out the pointer class
      Printv(ptrclass, "}\n\n", NIL);
      Printv(f_shadow, ptrclass, NIL);
      // pointer class end


      //  Create the "real" class.
      Printv(f_shadow, "itcl::class ", class_name, " {\n", NIL);
      Printv(f_shadow, "  inherit ", class_name, "Ptr\n", NIL);

      //  If we have a constructor, then use it.
      //  If not, then we must have an abstract class without
      //  any constructor.  So we create a class constructor
      //  which will fail for this class (but not for inherited
      //  classes).  Note that the constructor must fail before
      //  calling the ptrclass constructor.

      if (have_constructor) {
	Printv(f_shadow, constructor, NIL);
      } else {
	Printv(f_shadow, "  constructor { } {\n", NIL);
	Printv(f_shadow, "    # This constructor will fail if called directly\n", NIL);
	Printv(f_shadow, "    if { [info class] == \"::", class_name, "\" } {\n", NIL);
	Printv(f_shadow, "      error \"No constructor for class ", class_name, (Getattr(n, "abstracts") ? " - class is abstract" : ""), "\"\n", NIL);
	Printv(f_shadow, "    }\n", NIL);
	Printv(f_shadow, "  }\n", NIL);
      }

      Printv(f_shadow, "}\n\n", NIL);
    }

    Printv(f_wrappers, "static swig_class *swig_", mangled_classname, "_bases[] = {", base_class, "0};\n", NIL);
    Printv(f_wrappers, "static const char * swig_", mangled_classname, "_base_names[] = {", base_class_names, "0};\n", NIL);
    Delete(base_class);
    Delete(base_class_names);

    Printv(f_wrappers, "static swig_class _wrap_class_", mangled_classname, " = { \"", class_name, "\", &SWIGTYPE", SwigType_manglestr(t), ",", NIL);

    if (have_constructor) {
      Printf(f_wrappers, "%s", Swig_name_wrapper(Swig_name_construct(NSPACE_TODO, constructor_name)));
      Delete(constructor_name);
      constructor_name = 0;
    } else {
      Printf(f_wrappers, "0");
    }
    if (have_destructor) {
      Printv(f_wrappers, ", swig_delete_", class_name, NIL);
    } else {
      Printf(f_wrappers, ",0");
    }
    Printv(f_wrappers, ", swig_", mangled_classname, "_methods, swig_", mangled_classname, "_attributes, swig_", mangled_classname, "_bases,",
	   "swig_", mangled_classname, "_base_names, &swig_module, SWIG_TCL_HASHTABLE_INIT };\n", NIL);

    if (!itcl) {
      Printv(cmd_tab, tab4, "{ SWIG_prefix \"", class_name, "\", (swig_wrapper_func) SWIG_ObjectConstructor, (ClientData)&_wrap_class_", mangled_classname,
	     "},\n", NIL);
    };

    Delete(t);
    Delete(mangled_classname);
    return SWIG_OK;
  }


  /* ------------------------------------------------------------
   * memberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int memberfunctionHandler(Node *n) {
    String *name = Getattr(n, "name");
    String *iname = GetChar(n, "sym:name");

    String *realname, *rname;

    Language::memberfunctionHandler(n);

    realname = iname ? iname : name;
    rname = Swig_name_wrapper(Swig_name_member(NSPACE_TODO, class_name, realname));
    if (!Getattr(n, "sym:nextSibling")) {
      Printv(methods_tab, tab4, "{\"", realname, "\", ", rname, "}, \n", NIL);
    }

    if (itcl) {
      ParmList *l = Getattr(n, "parms");
      Parm *p = 0;
      String *pname = NewString("");

      // Add this member to our class handler function
      Printv(imethods, tab2, "method ", realname, " [list ", NIL);

      int pnum = 0;
      for (p = l; p; p = nextSibling(p)) {

	String *pn = Getattr(p, "name");
	String *dv = Getattr(p, "value");
	SwigType *pt = Getattr(p, "type");

	Printv(pname, ",(", pt, ")", NIL);
	Clear(pname);

	/* Only print an argument if not void */
	if (Cmp(pt, "void") != 0) {
	  if (Len(pn) > 0) {
	    Printv(pname, pn, NIL);
	  } else {
	    Printf(pname, "p%d", pnum);
	  }

	  if (Len(dv) > 0) {
	    String *defval = NewString(dv);
	    if (namespace_option) {
	      Insert(defval, 0, "::");
	      Insert(defval, 0, ns_name);
	    }
	    if (Strncmp(dv, "(", 1) == 0) {
	      Insert(defval, 0, "$");
	      Replaceall(defval, "(", "");
	      Replaceall(defval, ")", "");
	    }
	    Printv(imethods, "[list ", pname, " ", defval, "] ", NIL);
	  } else {
	    Printv(imethods, pname, " ", NIL);
	  }
	}
	++pnum;
      }
      Printv(imethods, "] ", NIL);

      if (namespace_option) {
	Printv(imethods, "{ ", ns_name, "::", class_name, "_", realname, " $swigobj", NIL);
      } else {
	Printv(imethods, "{ ", class_name, "_", realname, " $swigobj", NIL);
      };

      pnum = 0;
      for (p = l; p; p = nextSibling(p)) {

	String *pn = Getattr(p, "name");
	SwigType *pt = Getattr(p, "type");
	Clear(pname);

	/* Only print an argument if not void */
	if (Cmp(pt, "void") != 0) {
	  if (Len(pn) > 0) {
	    Printv(pname, pn, NIL);
	  } else {
	    Printf(pname, "p%d", pnum);
	  }
	  Printv(imethods, " $", pname, NIL);
	}
	++pnum;
      }
      Printv(imethods, " }\n", NIL);
      have_methods = 1;
    }

    Delete(rname);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * membervariableHandler()
   * ------------------------------------------------------------ */

  virtual int membervariableHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");
    String *rname;

    Language::membervariableHandler(n);
    Printv(attr_tab, tab4, "{ \"-", symname, "\",", NIL);
    rname = Swig_name_wrapper(Swig_name_get(NSPACE_TODO, Swig_name_member(NSPACE_TODO, class_name, symname)));
    Printv(attr_tab, rname, ", ", NIL);
    Delete(rname);
    if (!GetFlag(n, "feature:immutable")) {
      rname = Swig_name_wrapper(Swig_name_set(NSPACE_TODO, Swig_name_member(NSPACE_TODO, class_name, symname)));
      Printv(attr_tab, rname, "},\n", NIL);
      Delete(rname);
    } else {
      Printf(attr_tab, "0 },\n");
    }

    if (itcl) {
      Printv(attributes, "  public variable ", symname, "\n", NIL);

      Printv(attribute_traces, "    trace variable ", symname, " rw [list ", class_name, "_swig_getset ", symname, "]\n", NIL);
      Printv(attribute_traces, "    set ", symname, "\n", NIL);

      have_attributes = 1;
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constructorHandler()
   * ------------------------------------------------------------ */

  virtual int constructorHandler(Node *n) {
    Language::constructorHandler(n);

    if (itcl) {
      String *name = Getattr(n, "name");
      String *iname = GetChar(n, "sym:name");

      String *realname;

      ParmList *l = Getattr(n, "parms");
      Parm *p = 0;

      String *pname = NewString("");

      realname = iname ? iname : name;

      if (!have_constructor) {
	// Add this member to our class handler function
	Printf(constructor, "  constructor { ");

	//  Add parameter list
	int pnum = 0;
	for (p = l; p; p = nextSibling(p)) {

	  SwigType *pt = Getattr(p, "type");
	  String *pn = Getattr(p, "name");
	  String *dv = Getattr(p, "value");
	  Clear(pname);

	  /* Only print an argument if not void */
	  if (Cmp(pt, "void") != 0) {
	    if (Len(pn) > 0) {
	      Printv(pname, pn, NIL);
	    } else {
	      Printf(pname, "p%d", pnum);
	    }

	    if (Len(dv) > 0) {
	      Printv(constructor, "{", pname, " {", dv, "} } ", NIL);
	    } else {
	      Printv(constructor, pname, " ", NIL);
	    }
	  }
	  ++pnum;
	}
	Printf(constructor, "} { \n");

	// [BRE] 08/17/00 Added test to see if we are instantiating this object
	// type, or, if this constructor is being called as part of the itcl
	// inheritance hierarchy.
	// In the former case, we need to call the C++ constructor, in the
	// latter we don't, or we end up with two C++ objects.
	// Check to see if we are instantiating a 'realname' or something 
	// derived from it.
	//
	Printv(constructor, "    if { [string equal -nocase \"", realname, "\" \"[namespace tail [info class]]\" ] } {\n", NIL);

	// Call to constructor wrapper and parent Ptr class
	// [BRE] add -namespace/-prefix support

	if (namespace_option) {
	  Printv(constructor, "      ", realname, "Ptr::constructor [", ns_name, "::new_", realname, NIL);
	} else {
	  Printv(constructor, "      ", realname, "Ptr::constructor [new_", realname, NIL);
	}

	pnum = 0;
	for (p = l; p; p = nextSibling(p)) {

	  SwigType *pt = Getattr(p, "type");
	  String *pn = Getattr(p, "name");
	  Clear(pname);

	  /* Only print an argument if not void */
	  if (Cmp(pt, "void") != 0) {
	    if (Len(pn) > 0) {
	      Printv(pname, pn, NIL);
	    } else {
	      Printf(pname, "p%d", pnum);
	    }
	    Printv(constructor, " $", pname, NIL);
	  }
	  ++pnum;
	}

	Printv(constructor, "]\n", "    }\n", "  } {\n", "    set thisown 1\n", "  }\n", NIL);
      }
    }

    if (!have_constructor)
      constructor_name = NewString(Getattr(n, "sym:name"));
    have_constructor = 1;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * destructorHandler()
   * ------------------------------------------------------------ */

  virtual int destructorHandler(Node *n) {
    Language::destructorHandler(n);
    have_destructor = 1;
    destructor_action = Getattr(n, "wrap:action");
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * validIdentifier()
   * ------------------------------------------------------------ */

  virtual int validIdentifier(String *s) {
    if (Strchr(s, ' '))
      return 0;
    return 1;
  }

  /* ------------------------------------------------------------
   * usage_string()
   * ------------------------------------------------------------ */

  char *usage_string(char *iname, SwigType *, ParmList *l) {
    static String *temp = 0;
    Parm *p;
    int i, numopt, pcount;

    if (!temp)
      temp = NewString("");
    Clear(temp);
    if (namespace_option) {
      Printf(temp, "%s::%s ", ns_name, iname);
    } else {
      Printf(temp, "%s ", iname);
    }
    /* Now go through and print parameters */
    i = 0;
    pcount = emit_num_arguments(l);
    numopt = pcount - emit_num_required(l);
    for (p = l; p; p = nextSibling(p)) {

      SwigType *pt = Getattr(p, "type");
      String *pn = Getattr(p, "name");
      /* Only print an argument if not ignored */
      if (!checkAttribute(p, "tmap:in:numinputs", "0")) {
	if (i >= (pcount - numopt))
	  Putc('?', temp);
	if (Len(pn) > 0) {
	  Printf(temp, "%s", pn);
	} else {
	  Printf(temp, "%s", SwigType_str(pt, 0));
	}
	if (i >= (pcount - numopt))
	  Putc('?', temp);
	Putc(' ', temp);
	i++;
      }
    }
    return Char(temp);
  }

  String *runtimeCode() {
    String *s = NewString("");
    String *serrors = Swig_include_sys("tclerrors.swg");
    if (!serrors) {
      Printf(stderr, "*** Unable to open 'tclerrors.swg'\n");
    } else {
      Append(s, serrors);
      Delete(serrors);
    }
    String *sapi = Swig_include_sys("tclapi.swg");
    if (!sapi) {
      Printf(stderr, "*** Unable to open 'tclapi.swg'\n");
    } else {
      Append(s, sapi);
      Delete(sapi);
    }
    String *srun = Swig_include_sys("tclrun.swg");
    if (!srun) {
      Printf(stderr, "*** Unable to open 'tclrun.swg'\n");
    } else {
      Append(s, srun);
      Delete(srun);
    }

    return s;
  }

  String *defaultExternalRuntimeFilename() {
    return NewString("swigtclrun.h");
  }
};

/* ----------------------------------------------------------------------
 * swig_tcl()    - Instantiate module
 * ---------------------------------------------------------------------- */

static Language *new_swig_tcl() {
  return new TCL8();
}
extern "C" Language *swig_tcl(void) {
  return new_swig_tcl();
}
