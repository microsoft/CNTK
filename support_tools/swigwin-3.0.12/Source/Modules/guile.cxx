/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * guile.cxx
 *
 * Guile language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

#include <ctype.h>

// Note string broken in half for compilers that can't handle long strings
static const char *usage = "\
Guile Options (available with -guile)\n\
     -emitsetters            - Emit procedures-with-setters for variables\n\
                               and structure slots.\n\
     -emitslotaccessors      - Emit accessor methods for all GOOPS slots\n" "\
     -exportprimitive        - Add the (export ...) code from scmstub into the\n\
                               GOOPS file.\n\
     -goopsprefix <prefix>   - Prepend <prefix> to all goops identifiers\n\
     -Linkage <lstyle>       - Use linkage protocol <lstyle> (default `simple')\n\
                               Use `module' for native Guile module linking\n\
                               (requires Guile >= 1.5.0).  Use `passive' for\n\
                               passive linking (no C-level module-handling code),\n\
                               `ltdlmod' for Guile's old dynamic module\n\
                               convention (Guile <= 1.4), or `hobbit' for hobbit\n\
                               modules.\n\
     -onlysetters            - Don't emit traditional getter and setter\n\
                               procedures for structure slots,\n\
                               only emit procedures-with-setters.\n\
     -package <name>         - Set the path of the module to <name>\n\
                               (default NULL)\n\
     -prefix <name>          - Use <name> as prefix [default \"gswig_\"]\n\
     -procdoc <file>         - Output procedure documentation to <file>\n\
     -procdocformat <format> - Output procedure documentation in <format>;\n\
                               one of `guile-1.4', `plain', `texinfo'\n\
     -proxy                  - Export GOOPS class definitions\n\
     -primsuffix <suffix>    - Name appended to primitive module when exporting\n\
                               GOOPS classes. (default = \"primitive\")\n\
     -scmstub                - Output Scheme file with module declaration and\n\
                               exports; only with `passive' and `simple' linkage\n\
     -useclassprefix         - Prepend the class name to all goops identifiers\n\
\n";

static File *f_begin = 0;
static File *f_runtime = 0;
static File *f_header = 0;
static File *f_wrappers = 0;
static File *f_init = 0;


static String *prefix = NewString("gswig_");
static char *module = 0;
static String *package = 0;
static enum {
  GUILE_LSTYLE_SIMPLE,		// call `SWIG_init()'
  GUILE_LSTYLE_PASSIVE,		// passive linking (no module code)
  GUILE_LSTYLE_MODULE,		// native guile module linking (Guile >= 1.4.1)
  GUILE_LSTYLE_LTDLMOD_1_4,	// old (Guile <= 1.4) dynamic module convention
  GUILE_LSTYLE_HOBBIT		// use (hobbit4d link)
} linkage = GUILE_LSTYLE_SIMPLE;

static File *procdoc = 0;
static bool scmstub = false;
static String *scmtext;
static bool goops = false;
static String *goopstext;
static String *goopscode;
static String *goopsexport;

static enum {
  GUILE_1_4,
  PLAIN,
  TEXINFO
} docformat = GUILE_1_4;

static int emit_setters = 0;
static int only_setters = 0;
static int emit_slot_accessors = 0;
static int struct_member = 0;

static String *beforereturn = 0;
static String *return_nothing_doc = 0;
static String *return_one_doc = 0;
static String *return_multi_doc = 0;

static String *exported_symbols = 0;

static int exporting_destructor = 0;
static String *swigtype_ptr = 0;

/* GOOPS stuff */
static String *primsuffix = 0;
static String *class_name = 0;
static String *short_class_name = 0;
static String *goops_class_methods;
static int in_class = 0;
static int have_constructor = 0;
static int useclassprefix = 0;	// -useclassprefix argument
static String *goopsprefix = 0;	// -goopsprefix argument
static int primRenamer = 0;	// if (use-modules ((...) :renamer ...) is exported to GOOPS file
static int exportprimitive = 0;	// -exportprimitive argument
static String *memberfunction_name = 0;

extern "C" {
  static int has_classname(Node *class_node) {
    return Getattr(class_node, "guile:goopsclassname") ? 1 : 0;
  }
}

class GUILE:public Language {
public:

  /* ------------------------------------------------------------
   * main()
   * ------------------------------------------------------------ */

  virtual void main(int argc, char *argv[]) {
    int i;

     SWIG_library_directory("guile");
     SWIG_typemap_lang("guile");

    // Look for certain command line options
    for (i = 1; i < argc; i++) {
      if (argv[i]) {
	if (strcmp(argv[i], "-help") == 0) {
	  fputs(usage, stdout);
	  SWIG_exit(EXIT_SUCCESS);
	} else if (strcmp(argv[i], "-prefix") == 0) {
	  if (argv[i + 1]) {
	    prefix = NewString(argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-package") == 0) {
	  if (argv[i + 1]) {
	    package = NewString(argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-Linkage") == 0 || strcmp(argv[i], "-linkage") == 0) {
	  if (argv[i + 1]) {
	    if (0 == strcmp(argv[i + 1], "ltdlmod"))
	      linkage = GUILE_LSTYLE_LTDLMOD_1_4;
	    else if (0 == strcmp(argv[i + 1], "hobbit"))
	      linkage = GUILE_LSTYLE_HOBBIT;
	    else if (0 == strcmp(argv[i + 1], "simple"))
	      linkage = GUILE_LSTYLE_SIMPLE;
	    else if (0 == strcmp(argv[i + 1], "passive"))
	      linkage = GUILE_LSTYLE_PASSIVE;
	    else if (0 == strcmp(argv[i + 1], "module"))
	      linkage = GUILE_LSTYLE_MODULE;
	    else
	      Swig_arg_error();
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-procdoc") == 0) {
	  if (argv[i + 1]) {
	    procdoc = NewFile(argv[i + 1], "w", SWIG_output_files());
	    if (!procdoc) {
	      FileErrorDisplay(argv[i + 1]);
	      SWIG_exit(EXIT_FAILURE);
	    }
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-procdocformat") == 0) {
	  if (strcmp(argv[i + 1], "guile-1.4") == 0)
	    docformat = GUILE_1_4;
	  else if (strcmp(argv[i + 1], "plain") == 0)
	    docformat = PLAIN;
	  else if (strcmp(argv[i + 1], "texinfo") == 0)
	    docformat = TEXINFO;
	  else
	    Swig_arg_error();
	  Swig_mark_arg(i);
	  Swig_mark_arg(i + 1);
	  i++;
	} else if (strcmp(argv[i], "-emit-setters") == 0 || strcmp(argv[i], "-emitsetters") == 0) {
	  emit_setters = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-only-setters") == 0 || strcmp(argv[i], "-onlysetters") == 0) {
	  emit_setters = 1;
	  only_setters = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-emit-slot-accessors") == 0 || strcmp(argv[i], "-emitslotaccessors") == 0) {
	  emit_slot_accessors = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-scmstub") == 0) {
	  scmstub = true;
	  Swig_mark_arg(i);
	} else if ((strcmp(argv[i], "-shadow") == 0) || ((strcmp(argv[i], "-proxy") == 0))) {
	  goops = true;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-gh") == 0) {
	  Printf(stderr, "Deprecated command line option: -gh. Wrappers are always generated for the SCM interface. See documentation for more information regarding the deprecated GH interface.\n");
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-scm") == 0) {
	  Printf(stderr, "Deprecated command line option: -scm. Wrappers are always generated for the SCM interface. See documentation for more information regarding the deprecated GH interface.\n");
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-primsuffix") == 0) {
	  if (argv[i + 1]) {
	    primsuffix = NewString(argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-goopsprefix") == 0) {
	  if (argv[i + 1]) {
	    goopsprefix = NewString(argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-useclassprefix") == 0) {
	  useclassprefix = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-exportprimitive") == 0) {
	  exportprimitive = 1;
	  // should use Swig_warning() here?
	  Swig_mark_arg(i);
	}
      }
    }

    // set default value for primsuffix
    if (!primsuffix)
      primsuffix = NewString("primitive");

    //goops support can only be enabled if passive or module linkage is used
    if (goops) {
      if (linkage != GUILE_LSTYLE_PASSIVE && linkage != GUILE_LSTYLE_MODULE) {
	Printf(stderr, "guile: GOOPS support requires passive or module linkage\n");
	exit(1);
      }
    }

    if (goops) {
      // -proxy implies -emit-setters
      emit_setters = 1;
    }

    if ((linkage == GUILE_LSTYLE_PASSIVE && scmstub) || linkage == GUILE_LSTYLE_MODULE)
      primRenamer = 1;

    if (exportprimitive && primRenamer) {
      // should use Swig_warning() ?
      Printf(stderr, "guile: Warning: -exportprimitive only makes sense with passive linkage without a scmstub.\n");
    }

    // Make sure `prefix' ends in an underscore
    if (prefix) {
      const char *px = Char(prefix);
      if (px[Len(prefix) - 1] != '_')
	Printf(prefix, "_");
    }

    /* Add a symbol for this module */
    Preprocessor_define("SWIGGUILE 1", 0);
    /* Read in default typemaps */
    SWIG_config_file("guile_scm.swg");
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

    scmtext = NewString("");
    Swig_register_filebyname("scheme", scmtext);
    exported_symbols = NewString("");
    goopstext = NewString("");
    Swig_register_filebyname("goops", goopstext);
    goopscode = NewString("");
    goopsexport = NewString("");

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGGUILE\n#define SWIGGUILE\n#endif\n\n");

    /* Write out directives and declarations */

    module = Swig_copy_string(Char(Getattr(n, "name")));

    switch (linkage) {
    case GUILE_LSTYLE_SIMPLE:
      /* Simple linkage; we have to export the SWIG_init function. The user can
         rename the function by a #define. */
      Printf(f_runtime, "#define SWIG_GUILE_INIT_STATIC extern\n");
      break;
    default:
      /* Other linkage; we make the SWIG_init function static */
      Printf(f_runtime, "#define SWIG_GUILE_INIT_STATIC static\n");
      break;
    }

    if (CPlusPlus) {
      Printf(f_runtime, "extern \"C\" {\n\n");
    }
    Printf(f_runtime, "SWIG_GUILE_INIT_STATIC void\nSWIG_init (void);\n");
    if (CPlusPlus) {
      Printf(f_runtime, "\n}\n");
    }

    Printf(f_runtime, "\n");

    Language::top(n);

    /* Close module */

    Printf(f_wrappers, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n");

    SwigType_emit_type_table(f_runtime, f_wrappers);

    Printf(f_init, "}\n\n");
    Printf(f_init, "#ifdef __cplusplus\n}\n#endif\n");

    String *module_name = NewString("");

    if (!module)
      Printv(module_name, "swig", NIL);
    else {
      if (package)
	Printf(module_name, "%s/%s", package, module);
      else
	Printv(module_name, module, NIL);
    }
    emit_linkage(module_name);

    Delete(module_name);

    if (procdoc) {
      Delete(procdoc);
      procdoc = NULL;
    }
    Delete(goopscode);
    Delete(goopsexport);
    Delete(goopstext);

    /* Close all of the files */
    Dump(f_runtime, f_begin);
    Dump(f_header, f_begin);
    Dump(f_wrappers, f_begin);
    Wrapper_pretty_print(f_init, f_begin);
    Delete(f_header);
    Delete(f_wrappers);
    Delete(f_init);
    Delete(f_runtime);
    Delete(f_begin);
    return SWIG_OK;
  }

  void emit_linkage(String *module_name) {
    String *module_func = NewString("");

    if (CPlusPlus) {
      Printf(f_init, "extern \"C\" {\n\n");
    }

    Printv(module_func, module_name, NIL);
    Replaceall(module_func, "-", "_");

    switch (linkage) {
    case GUILE_LSTYLE_SIMPLE:
      Printf(f_init, "\n/* Linkage: simple */\n");
      break;
    case GUILE_LSTYLE_PASSIVE:
      Printf(f_init, "\n/* Linkage: passive */\n");
      Replaceall(module_func, "/", "_");
      Insert(module_func, 0, "scm_init_");
      Append(module_func, "_module");

      Printf(f_init, "SCM\n%s (void)\n{\n", module_func);
      Printf(f_init, "  SWIG_init();\n");
      Printf(f_init, "  return SCM_UNSPECIFIED;\n");
      Printf(f_init, "}\n");
      break;
    case GUILE_LSTYLE_LTDLMOD_1_4:
      Printf(f_init, "\n/* Linkage: ltdlmod */\n");
      Replaceall(module_func, "/", "_");
      Insert(module_func, 0, "scm_init_");
      Append(module_func, "_module");
      Printf(f_init, "SCM\n%s (void)\n{\n", module_func);
      {
	String *mod = NewString(module_name);
	Replaceall(mod, "/", " ");
	Printf(f_init, "    scm_register_module_xxx (\"%s\", (void *) SWIG_init);\n", mod);
	Printf(f_init, "    return SCM_UNSPECIFIED;\n");
	Delete(mod);
      }
      Printf(f_init, "}\n");
      break;
    case GUILE_LSTYLE_MODULE:
      Printf(f_init, "\n/* Linkage: module */\n");
      Replaceall(module_func, "/", "_");
      Insert(module_func, 0, "scm_init_");
      Append(module_func, "_module");

      Printf(f_init, "static void SWIG_init_helper(void *data)\n");
      Printf(f_init, "{\n    SWIG_init();\n");
      if (Len(exported_symbols) > 0)
	Printf(f_init, "    scm_c_export(%sNULL);", exported_symbols);
      Printf(f_init, "\n}\n\n");

      Printf(f_init, "SCM\n%s (void)\n{\n", module_func);
      {
	String *mod = NewString(module_name);
	if (goops)
	  Printv(mod, "-", primsuffix, NIL);
	Replaceall(mod, "/", " ");
	Printf(f_init, "    scm_c_define_module(\"%s\",\n", mod);
	Printf(f_init, "      SWIG_init_helper, NULL);\n");
	Printf(f_init, "    return SCM_UNSPECIFIED;\n");
	Delete(mod);
      }
      Printf(f_init, "}\n");
      break;
    case GUILE_LSTYLE_HOBBIT:
      Printf(f_init, "\n/* Linkage: hobbit */\n");
      Replaceall(module_func, "/", "_slash_");
      Insert(module_func, 0, "scm_init_");
      Printf(f_init, "SCM\n%s (void)\n{\n", module_func);
      {
	String *mod = NewString(module_name);
	Replaceall(mod, "/", " ");
	Printf(f_init, "    scm_register_module_xxx (\"%s\", (void *) SWIG_init);\n", mod);
	Printf(f_init, "    return SCM_UNSPECIFIED;\n");
	Delete(mod);
      }
      Printf(f_init, "}\n");
      break;
    default:
      abort();			// for now
    }

    if (scmstub) {
      /* Emit Scheme stub if requested */
      String *primitive_name = NewString(module_name);
      if (goops)
	Printv(primitive_name, "-", primsuffix, NIL);

      String *mod = NewString(primitive_name);
      Replaceall(mod, "/", " ");

      String *fname = NewStringf("%s%s.scm",
				 SWIG_output_directory(),
				 primitive_name);
      Delete(primitive_name);
      File *scmstubfile = NewFile(fname, "w", SWIG_output_files());
      if (!scmstubfile) {
	FileErrorDisplay(fname);
	SWIG_exit(EXIT_FAILURE);
      }
      Delete(fname);

      Swig_banner_target_lang(scmstubfile, ";;;");
      Printf(scmstubfile, "\n");
      if (linkage == GUILE_LSTYLE_SIMPLE || linkage == GUILE_LSTYLE_PASSIVE)
	Printf(scmstubfile, "(define-module (%s))\n\n", mod);
      Delete(mod);
      Printf(scmstubfile, "%s", scmtext);
      if ((linkage == GUILE_LSTYLE_SIMPLE || linkage == GUILE_LSTYLE_PASSIVE)
	  && Len(exported_symbols) > 0) {
	String *ex = NewString(exported_symbols);
	Replaceall(ex, ", ", "\n        ");
	Replaceall(ex, "\"", "");
	Chop(ex);
	Printf(scmstubfile, "\n(export %s)\n", ex);
	Delete(ex);
      }
      Delete(scmstubfile);
    }

    if (goops) {
      String *mod = NewString(module_name);
      Replaceall(mod, "/", " ");

      String *fname = NewStringf("%s%s.scm", SWIG_output_directory(),
				 module_name);
      File *goopsfile = NewFile(fname, "w", SWIG_output_files());
      if (!goopsfile) {
	FileErrorDisplay(fname);
	SWIG_exit(EXIT_FAILURE);
      }
      Delete(fname);
      Swig_banner_target_lang(goopsfile, ";;;");
      Printf(goopsfile, "\n");
      Printf(goopsfile, "(define-module (%s))\n", mod);
      Printf(goopsfile, "%s\n", goopstext);
      Printf(goopsfile, "(use-modules (oop goops) (Swig common))\n");
      if (primRenamer) {
	Printf(goopsfile, "(use-modules ((%s-%s) :renamer (symbol-prefix-proc 'primitive:)))\n", mod, primsuffix);
      }
      Printf(goopsfile, "%s\n(export %s)", goopscode, goopsexport);
      if (exportprimitive) {
	String *ex = NewString(exported_symbols);
	Replaceall(ex, ", ", "\n        ");
	Replaceall(ex, "\"", "");
	Chop(ex);
	Printf(goopsfile, "\n(export %s)", ex);
	Delete(ex);
      }
      Delete(mod);
      Delete(goopsfile);
    }

    Delete(module_func);
    if (CPlusPlus) {
      Printf(f_init, "\n}\n");
    }
  }

  /* Return true iff T is a pointer type */

  int is_a_pointer(SwigType *t) {
    return SwigType_ispointer(SwigType_typedef_resolve_all(t));
  }

  /* Report an error handling the given type. */

  void throw_unhandled_guile_type_error(SwigType *d) {
    Swig_warning(WARN_TYPEMAP_UNDEF, input_file, line_number, "Unable to handle type %s.\n", SwigType_str(d, 0));
  }

  /* Write out procedure documentation */

  void write_doc(const String *proc_name, const String *signature, const String *doc, const String *signature2 = NULL) {
    switch (docformat) {
    case GUILE_1_4:
      Printv(procdoc, "\f\n", NIL);
      Printv(procdoc, "(", signature, ")\n", NIL);
      if (signature2)
	Printv(procdoc, "(", signature2, ")\n", NIL);
      Printv(procdoc, doc, "\n", NIL);
      break;
    case PLAIN:
      Printv(procdoc, "\f", proc_name, "\n\n", NIL);
      Printv(procdoc, "(", signature, ")\n", NIL);
      if (signature2)
	Printv(procdoc, "(", signature2, ")\n", NIL);
      Printv(procdoc, doc, "\n\n", NIL);
      break;
    case TEXINFO:
      Printv(procdoc, "\f", proc_name, "\n", NIL);
      Printv(procdoc, "@deffn primitive ", signature, "\n", NIL);
      if (signature2)
	Printv(procdoc, "@deffnx primitive ", signature2, "\n", NIL);
      Printv(procdoc, doc, "\n", NIL);
      Printv(procdoc, "@end deffn\n\n", NIL);
      break;
    }
  }

  /* returns false if the typemap is an empty string */
  bool handle_documentation_typemap(String *output,
				    const String *maybe_delimiter, Parm *p, const String *typemap, const String *default_doc, const String *name = NULL) {
    String *tmp = NewString("");
    String *tm;
    if (!(tm = Getattr(p, typemap))) {
      Printf(tmp, "%s", default_doc);
      tm = tmp;
    }
    bool result = (Len(tm) > 0);
    if (maybe_delimiter && Len(output) > 0 && Len(tm) > 0) {
      Printv(output, maybe_delimiter, NIL);
    }
    const String *pn = !name ? (const String *) Getattr(p, "name") : name;
    String *pt = Getattr(p, "type");
    Replaceall(tm, "$name", pn);	// legacy for $parmname
    Replaceall(tm, "$type", SwigType_str(pt, 0));
    /* $NAME is like $name, but marked-up as a variable. */
    String *ARGNAME = NewString("");
    if (docformat == TEXINFO)
      Printf(ARGNAME, "@var{%s}", pn);
    else
      Printf(ARGNAME, "%(upper)s", pn);
    Replaceall(tm, "$NAME", ARGNAME);
    Replaceall(tm, "$PARMNAME", ARGNAME);
    Printv(output, tm, NIL);
    Delete(tmp);
    return result;
  }

  /* ------------------------------------------------------------
   * functionWrapper()
   * Create a function declaration and register it with the interpreter.
   * ------------------------------------------------------------ */

  virtual int functionWrapper(Node *n) {
    String *iname = Getattr(n, "sym:name");
    SwigType *d = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    Parm *p;
    String *proc_name = 0;
    char source[256];
    Wrapper *f = NewWrapper();
    String *cleanup = NewString("");
    String *outarg = NewString("");
    String *signature = NewString("");
    String *doc_body = NewString("");
    String *returns = NewString("");
    String *method_signature = NewString("");
    String *primitive_args = NewString("");
    Hash *scheme_arg_names = NewHash();
    int num_results = 1;
    String *tmp = NewString("");
    String *tm;
    int i;
    int numargs = 0;
    int numreq = 0;
    String *overname = 0;
    int args_passed_as_array = 0;
    int scheme_argnum = 0;
    bool any_specialized_arg = false;

    // Make a wrapper name for this
    String *wname = Swig_name_wrapper(iname);
    if (Getattr(n, "sym:overloaded")) {
      overname = Getattr(n, "sym:overname");
      args_passed_as_array = 1;
    } else {
      if (!addSymbol(iname, n)) {
        DelWrapper(f);
	return SWIG_ERROR; 
      }
    }
    if (overname) {
      Append(wname, overname);
    }
    Setattr(n, "wrap:name", wname);

    // Build the name for scheme.
    proc_name = NewString(iname);
    Replaceall(proc_name, "_", "-");

    /* Emit locals etc. into f->code; figure out which args to ignore */
    emit_parameter_variables(l, f);

    /* Attach the standard typemaps */
    emit_attach_parmmaps(l, f);
    Setattr(n, "wrap:parms", l);

    /* Get number of required and total arguments */
    numargs = emit_num_arguments(l);
    numreq = emit_num_required(l);

    /* Declare return variable */

    Wrapper_add_local(f, "gswig_result", "SCM gswig_result");
    Wrapper_add_local(f, "gswig_list_p", "SWIGUNUSED int gswig_list_p = 0");

    /* Open prototype and signature */

    Printv(f->def, "static SCM\n", wname, " (", NIL);
    if (args_passed_as_array) {
      Printv(f->def, "int argc, SCM *argv", NIL);
    }
    Printv(signature, proc_name, NIL);

    /* Now write code to extract the parameters */

    for (i = 0, p = l; i < numargs; i++) {

      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");
      int opt_p = (i >= numreq);

      // Produce names of source and target
      if (args_passed_as_array)
	sprintf(source, "argv[%d]", i);
      else
	sprintf(source, "s_%d", i);
      String *target = Getattr(p, "lname");

      if (!args_passed_as_array) {
	if (i != 0)
	  Printf(f->def, ", ");
	Printf(f->def, "SCM s_%d", i);
      }
      if (opt_p) {
	Printf(f->code, "    if (%s != SCM_UNDEFINED) {\n", source);
      }
      if ((tm = Getattr(p, "tmap:in"))) {
	Replaceall(tm, "$source", source);
	Replaceall(tm, "$target", target);
	Replaceall(tm, "$input", source);
	Setattr(p, "emit:input", source);
	Printv(f->code, tm, "\n", NIL);

	SwigType *pb = SwigType_typedef_resolve_all(SwigType_base(pt));
	SwigType *pn = Getattr(p, "name");
	String *argname;
	scheme_argnum++;
	if (pn && !Getattr(scheme_arg_names, pn))
	  argname = pn;
	else {
	  /* Anonymous arg or re-used argument name -- choose a name that cannot clash */
	  argname = NewStringf("%%arg%d", scheme_argnum);
	}

	if (procdoc) {
	  if (i == numreq) {
	    /* First optional argument */
	    Printf(signature, " #:optional");
	  }
	  /* Add to signature (arglist) */
	  handle_documentation_typemap(signature, " ", p, "tmap:in:arglist", "$name", argname);
	  /* Document the type of the arg in the documentation body */
	  handle_documentation_typemap(doc_body, ", ", p, "tmap:in:doc", "$NAME is of type <$type>", argname);
	}

	if (goops) {
	  if (i < numreq) {
	    if (strcmp("void", Char(pt)) != 0) {
	      Node *class_node = Swig_symbol_clookup_check(pb, Getattr(n, "sym:symtab"),
							   has_classname);
	      String *goopsclassname = !class_node ? NULL : Getattr(class_node, "guile:goopsclassname");
	      /* do input conversion */
	      if (goopsclassname) {
		Printv(method_signature, " (", argname, " ", goopsclassname, ")", NIL);
		any_specialized_arg = true;
	      } else {
		Printv(method_signature, " ", argname, NIL);
	      }
	      Printv(primitive_args, " ", argname, NIL);
	      Setattr(scheme_arg_names, argname, p);
	    }
	  }
	}

	if (!pn) {
	  Delete(argname);
	}
	p = Getattr(p, "tmap:in:next");
      } else {
	throw_unhandled_guile_type_error(pt);
	p = nextSibling(p);
      }
      if (opt_p)
	Printf(f->code, "    }\n");
    }
    if (Len(doc_body) > 0)
      Printf(doc_body, ".\n");

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
    /* Pass output arguments back to the caller. */

    /* Insert argument output code */
    String *returns_argout = NewString("");
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:argout"))) {
	Replaceall(tm, "$source", Getattr(p, "lname"));
	Replaceall(tm, "$target", Getattr(p, "lname"));
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(outarg, tm, "\n", NIL);
	if (procdoc) {
	  if (handle_documentation_typemap(returns_argout, ", ", p, "tmap:argout:doc", "$NAME (of type $type)")) {
	    /* A documentation typemap that is not the empty string
	       indicates that a value is returned to Scheme. */
	    num_results++;
	  }
	}
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert cleanup code */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:freearg"))) {
	Replaceall(tm, "$target", Getattr(p, "lname"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(cleanup, tm, "\n", NIL);
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }

    if (exporting_destructor) {
      /* Mark the destructor's argument as destroyed. */
      String *tm = NewString("SWIG_Guile_MarkPointerDestroyed($input);");
      Replaceall(tm, "$input", Getattr(l, "emit:input"));
      Printv(cleanup, tm, "\n", NIL);
      Delete(tm);
    }

    /* Close prototype */

    Printf(f->def, ")\n{\n");

    /* Define the scheme name in C. This define is used by several Guile
       macros. */
    Printv(f->def, "#define FUNC_NAME \"", proc_name, "\"", NIL);

    // Now write code to make the function call
    String *actioncode = emit_action(n);

    // Now have return value, figure out what to do with it.
    if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
      Replaceall(tm, "$result", "gswig_result");
      Replaceall(tm, "$target", "gswig_result");
      Replaceall(tm, "$source", Swig_cresult_name());
      if (GetFlag(n, "feature:new"))
	Replaceall(tm, "$owner", "1");
      else
	Replaceall(tm, "$owner", "0");
      Printv(f->code, tm, "\n", NIL);
    } else {
      throw_unhandled_guile_type_error(d);
    }
    emit_return_variable(n, d, f);

    // Documentation
    if ((tm = Getattr(n, "tmap:out:doc"))) {
      Printv(returns, tm, NIL);
      if (Len(tm) > 0)
	num_results = 1;
      else
	num_results = 0;
    } else {
      String *s = SwigType_str(d, 0);
      Chop(s);
      Printf(returns, "<%s>", s);
      Delete(s);
      num_results = 1;
    }
    Append(returns, returns_argout);


    // Dump the argument output code
    Printv(f->code, outarg, NIL);

    // Dump the argument cleanup code
    Printv(f->code, cleanup, NIL);

    // Look for any remaining cleanup

    if (GetFlag(n, "feature:new")) {
      if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
	Replaceall(tm, "$source", Swig_cresult_name());
	Printv(f->code, tm, "\n", NIL);
      }
    }
    // Free any memory allocated by the function being wrapped..
    if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
      Replaceall(tm, "$source", Swig_cresult_name());
      Printv(f->code, tm, "\n", NIL);
    }
    // Wrap things up (in a manner of speaking)

    if (beforereturn)
      Printv(f->code, beforereturn, "\n", NIL);
    Printv(f->code, "return gswig_result;\n", NIL);

    /* Substitute the function name */
    Replaceall(f->code, "$symname", iname);
    // Undefine the scheme name

    Printf(f->code, "#undef FUNC_NAME\n");
    Printf(f->code, "}\n");

    Wrapper_print(f, f_wrappers);

    if (!Getattr(n, "sym:overloaded")) {
      if (numargs > 10) {
	int i;
	/* gh_new_procedure would complain: too many args */
	/* Build a wrapper wrapper */
	Printv(f_wrappers, "static SCM\n", wname, "_rest (SCM rest)\n", NIL);
	Printv(f_wrappers, "{\n", NIL);
	Printf(f_wrappers, "SCM arg[%d];\n", numargs);
	Printf(f_wrappers, "SWIG_Guile_GetArgs (arg, rest, %d, %d, \"%s\");\n", numreq, numargs - numreq, proc_name);
	Printv(f_wrappers, "return ", wname, "(", NIL);
	Printv(f_wrappers, "arg[0]", NIL);
	for (i = 1; i < numargs; i++)
	  Printf(f_wrappers, ", arg[%d]", i);
	Printv(f_wrappers, ");\n", NIL);
	Printv(f_wrappers, "}\n", NIL);
	/* Register it */
	Printf(f_init, "scm_c_define_gsubr(\"%s\", 0, 0, 1, (swig_guile_proc) %s_rest);\n", proc_name, wname);
      } else if (emit_setters && struct_member && strlen(Char(proc_name)) > 3) {
	int len = Len(proc_name);
	const char *pc = Char(proc_name);
	/* MEMBER-set and MEMBER-get functions. */
	int is_setter = (pc[len - 3] == 's');
	if (is_setter) {
	  Printf(f_init, "SCM setter = ");
	  struct_member = 2;	/* have a setter */
	} else
	  Printf(f_init, "SCM getter = ");
	/* GOOPS support uses the MEMBER-set and MEMBER-get functions,
	   so ignore only_setters in this case. */
	if (only_setters && !goops)
	  Printf(f_init, "scm_c_make_gsubr(\"%s\", %d, %d, 0, (swig_guile_proc) %s);\n", proc_name, numreq, numargs - numreq, wname);
	else
	  Printf(f_init, "scm_c_define_gsubr(\"%s\", %d, %d, 0, (swig_guile_proc) %s);\n", proc_name, numreq, numargs - numreq, wname);

	if (!is_setter) {
	  /* Strip off "-get" */
	  char *pws_name = (char *) malloc(sizeof(char) * (len - 3));
	  strncpy(pws_name, pc, len - 3);
	  pws_name[len - 4] = 0;
	  if (struct_member == 2) {
	    /* There was a setter, so create a procedure with setter */
	    Printf(f_init, "scm_c_define");
	    Printf(f_init, "(\"%s\", " "scm_make_procedure_with_setter(getter, setter));\n", pws_name);
	  } else {
	    /* There was no setter, so make an alias to the getter */
	    Printf(f_init, "scm_c_define");
	    Printf(f_init, "(\"%s\", getter);\n", pws_name);
	  }
	  Printf(exported_symbols, "\"%s\", ", pws_name);
	  free(pws_name);
	}
      } else {
	/* Register the function */
	if (exporting_destructor) {
	  Printf(f_init, "((swig_guile_clientdata *)(SWIGTYPE%s->clientdata))->destroy = (guile_destructor) %s;\n", swigtype_ptr, wname);
	  //Printf(f_init, "SWIG_TypeClientData(SWIGTYPE%s, (void *) %s);\n", swigtype_ptr, wname);
	}
	Printf(f_init, "scm_c_define_gsubr(\"%s\", %d, %d, 0, (swig_guile_proc) %s);\n", proc_name, numreq, numargs - numreq, wname);
      }
    } else {			/* overloaded function; don't export the single methods */
      if (!Getattr(n, "sym:nextSibling")) {
	/* Emit overloading dispatch function */

	int maxargs;
	String *dispatch = Swig_overload_dispatch(n, "return %s(argc,argv);", &maxargs);

	/* Generate a dispatch wrapper for all overloaded functions */

	Wrapper *df = NewWrapper();
	String *dname = Swig_name_wrapper(iname);

	Printv(df->def, "static SCM\n", dname, "(SCM rest)\n{\n", NIL);
	Printf(df->code, "#define FUNC_NAME \"%s\"\n", proc_name);
	Printf(df->code, "SCM argv[%d];\n", maxargs);
	Printf(df->code, "int argc = SWIG_Guile_GetArgs (argv, rest, %d, %d, \"%s\");\n", 0, maxargs, proc_name);
	Printv(df->code, dispatch, "\n", NIL);
	Printf(df->code, "scm_misc_error(\"%s\", \"No matching method for generic function `%s'\", SCM_EOL);\n", proc_name, iname);
	Printf(df->code, "#undef FUNC_NAME\n");
	Printv(df->code, "}\n", NIL);
	Wrapper_print(df, f_wrappers);
	Printf(f_init, "scm_c_define_gsubr(\"%s\", 0, 0, 1, (swig_guile_proc) %s);\n", proc_name, dname);
	DelWrapper(df);
	Delete(dispatch);
	Delete(dname);
      }
    }
    Printf(exported_symbols, "\"%s\", ", proc_name);

    if (!in_class || memberfunction_name) {
      // export wrapper into goops file
      String *method_def = NewString("");
      String *goops_name;
      if (in_class)
	goops_name = NewString(memberfunction_name);
      else
	goops_name = goopsNameMapping(proc_name, "");
      String *primitive_name = NewString("");
      if (primRenamer)
	Printv(primitive_name, "primitive:", proc_name, NIL);
      else
	Printv(primitive_name, proc_name, NIL);
      Replaceall(method_signature, "_", "-");
      Replaceall(primitive_args, "_", "-");
      if (!any_specialized_arg) {
	/* If there would not be any specialized argument in
	   the method declaration, we simply re-export the
	   function.  This is a performance optimization. */
	Printv(method_def, "(define ", goops_name, " ", primitive_name, ")\n", NIL);
      } else if (numreq == numargs) {
	Printv(method_def, "(define-method (", goops_name, method_signature, ")\n", NIL);
	Printv(method_def, "  (", primitive_name, primitive_args, "))\n", NIL);
      } else {
	/* Handle optional args. For the rest argument, use a name
	   that cannot clash. */
	Printv(method_def, "(define-method (", goops_name, method_signature, " . %args)\n", NIL);
	Printv(method_def, "  (apply ", primitive_name, primitive_args, " %args))\n", NIL);
      }
      if (in_class) {
	/* Defer method definition till end of class definition. */
	Printv(goops_class_methods, method_def, NIL);
      } else {
	Printv(goopscode, method_def, NIL);
      }
      Printf(goopsexport, "%s ", goops_name);
      Delete(primitive_name);
      Delete(goops_name);
      Delete(method_def);
    }

    if (procdoc) {
      String *returns_text = NewString("");
      if (num_results == 0)
	Printv(returns_text, return_nothing_doc, NIL);
      else if (num_results == 1)
	Printv(returns_text, return_one_doc, NIL);
      else
	Printv(returns_text, return_multi_doc, NIL);
      /* Substitute documentation variables */
      static const char *numbers[] = { "zero", "one", "two", "three",
	"four", "five", "six", "seven",
	"eight", "nine", "ten", "eleven",
	"twelve"
      };
      if (num_results <= 12)
	Replaceall(returns_text, "$num_values", numbers[num_results]);
      else {
	String *num_results_str = NewStringf("%d", num_results);
	Replaceall(returns_text, "$num_values", num_results_str);
	Delete(num_results_str);
      }
      Replaceall(returns_text, "$values", returns);
      Printf(doc_body, "\n%s", returns_text);
      write_doc(proc_name, signature, doc_body);
      Delete(returns_text);
    }

    Delete(proc_name);
    Delete(outarg);
    Delete(cleanup);
    Delete(signature);
    Delete(method_signature);
    Delete(primitive_args);
    Delete(doc_body);
    Delete(returns_argout);
    Delete(returns);
    Delete(tmp);
    Delete(scheme_arg_names);
    DelWrapper(f);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * variableWrapper()
   *
   * Create a link to a C variable.
   * This creates a single function PREFIX_var_VARNAME().
   * This function takes a single optional argument.   If supplied, it means
   * we are setting this variable to some value.  If omitted, it means we are
   * simply evaluating this variable.  Either way, we return the variables
   * value.
   * ------------------------------------------------------------ */

  virtual int variableWrapper(Node *n) {

    char *name = GetChar(n, "name");
    char *iname = GetChar(n, "sym:name");
    SwigType *t = Getattr(n, "type");

    String *proc_name;
    Wrapper *f;
    String *tm;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    f = NewWrapper();
    // evaluation function names

    String *var_name = Swig_name_wrapper(iname);

    // Build the name for scheme.
    proc_name = NewString(iname);
    Replaceall(proc_name, "_", "-");
    Setattr(n, "wrap:name", proc_name);

    if (1 || (SwigType_type(t) != T_USER) || (is_a_pointer(t))) {

      Printf(f->def, "static SCM\n%s(SCM s_0)\n{\n", var_name);

      /* Define the scheme name in C. This define is used by several Guile
         macros. */
      Printv(f->def, "#define FUNC_NAME \"", proc_name, "\"", NIL);

      Wrapper_add_local(f, "gswig_result", "SCM gswig_result");

      if (!GetFlag(n, "feature:immutable")) {
	/* Check for a setting of the variable value */
	Printf(f->code, "if (s_0 != SCM_UNDEFINED) {\n");
	if ((tm = Swig_typemap_lookup("varin", n, name, 0))) {
	  Replaceall(tm, "$source", "s_0");
	  Replaceall(tm, "$input", "s_0");
	  Replaceall(tm, "$target", name);
	  /* Printv(f->code,tm,"\n",NIL); */
	  emit_action_code(n, f->code, tm);
	} else {
	  throw_unhandled_guile_type_error(t);
	}
	Printf(f->code, "}\n");
      }
      // Now return the value of the variable (regardless
      // of evaluating or setting)

      if ((tm = Swig_typemap_lookup("varout", n, name, 0))) {
	Replaceall(tm, "$source", name);
	Replaceall(tm, "$target", "gswig_result");
	Replaceall(tm, "$result", "gswig_result");
	/* Printv(f->code,tm,"\n",NIL); */
	emit_action_code(n, f->code, tm);
      } else {
	throw_unhandled_guile_type_error(t);
      }
      Printf(f->code, "\nreturn gswig_result;\n");
      Printf(f->code, "#undef FUNC_NAME\n");
      Printf(f->code, "}\n");

      Wrapper_print(f, f_wrappers);

      // Now add symbol to the Guile interpreter

      if (!emit_setters || GetFlag(n, "feature:immutable")) {
	/* Read-only variables become a simple procedure returning the
	   value; read-write variables become a simple procedure with
	   an optional argument. */

	if (!goops && GetFlag(n, "feature:constasvar")) {
	  /* need to export this function as a variable instead of a procedure */
	  if (scmstub) {
	    /* export the function in the wrapper, and (set!) it in scmstub */
	    Printf(f_init, "scm_c_define_gsubr(\"%s\", 0, %d, 0, (swig_guile_proc) %s);\n", proc_name, !GetFlag(n, "feature:immutable"), var_name);
	    Printf(scmtext, "(set! %s (%s))\n", proc_name, proc_name);
	  } else {
	    /* export the variable directly */
	    Printf(f_init, "scm_c_define(\"%s\", %s(SCM_UNDEFINED));\n", proc_name, var_name);
	  }

	} else {
	  /* Export the function as normal */
	  Printf(f_init, "scm_c_define_gsubr(\"%s\", 0, %d, 0, (swig_guile_proc) %s);\n", proc_name, !GetFlag(n, "feature:immutable"), var_name);
	}

      } else {
	/* Read/write variables become a procedure with setter. */
	Printf(f_init, "{ SCM p = scm_c_define_gsubr(\"%s\", 0, 1, 0, (swig_guile_proc) %s);\n", proc_name, var_name);
	Printf(f_init, "scm_c_define");
	Printf(f_init, "(\"%s\", " "scm_make_procedure_with_setter(p, p)); }\n", proc_name);
      }
      Printf(exported_symbols, "\"%s\", ", proc_name);

      // export wrapper into goops file
      if (!in_class) {		// only if the variable is not part of a class
	String *class_name = SwigType_typedef_resolve_all(SwigType_base(t));
	String *goops_name = goopsNameMapping(proc_name, "");
	String *primitive_name = NewString("");
	if (primRenamer)
	  Printv(primitive_name, "primitive:", NIL);
	Printv(primitive_name, proc_name, NIL);
	/* Simply re-export the procedure */
	if ((!emit_setters || GetFlag(n, "feature:immutable"))
	    && GetFlag(n, "feature:constasvar")) {
	  Printv(goopscode, "(define ", goops_name, " (", primitive_name, "))\n", NIL);
	} else {
	  Printv(goopscode, "(define ", goops_name, " ", primitive_name, ")\n", NIL);
	}
	Printf(goopsexport, "%s ", goops_name);
	Delete(primitive_name);
	Delete(class_name);
	Delete(goops_name);
      }

      if (procdoc) {
	/* Compute documentation */
	String *signature = NewString("");
	String *signature2 = NULL;
	String *doc = NewString("");

	if (GetFlag(n, "feature:immutable")) {
	  Printv(signature, proc_name, NIL);
	  if (GetFlag(n, "feature:constasvar")) {
	    Printv(doc, "Is constant ", NIL);
	  } else {
	    Printv(doc, "Returns constant ", NIL);
	  }
	  if ((tm = Getattr(n, "tmap:varout:doc"))) {
	    Printv(doc, tm, NIL);
	  } else {
	    String *s = SwigType_str(t, 0);
	    Chop(s);
	    Printf(doc, "<%s>", s);
	    Delete(s);
	  }
	} else if (emit_setters) {
	  Printv(signature, proc_name, NIL);
	  signature2 = NewString("");
	  Printv(signature2, "set! (", proc_name, ") ", NIL);
	  handle_documentation_typemap(signature2, NIL, n, "tmap:varin:arglist", "new-value");
	  Printv(doc, "Get or set the value of the C variable, \n", NIL);
	  Printv(doc, "which is of type ", NIL);
	  handle_documentation_typemap(doc, NIL, n, "tmap:varout:doc", "$1_type");
	  Printv(doc, ".");
	} else {
	  Printv(signature, proc_name, " #:optional ", NIL);
	  if ((tm = Getattr(n, "tmap:varin:doc"))) {
	    Printv(signature, tm, NIL);
	  } else {
	    String *s = SwigType_str(t, 0);
	    Chop(s);
	    Printf(signature, "new-value <%s>", s);
	    Delete(s);
	  }

	  Printv(doc, "If NEW-VALUE is provided, " "set C variable to this value.\n", NIL);
	  Printv(doc, "Returns variable value ", NIL);
	  if ((tm = Getattr(n, "tmap:varout:doc"))) {
	    Printv(doc, tm, NIL);
	  } else {
	    String *s = SwigType_str(t, 0);
	    Chop(s);
	    Printf(doc, "<%s>", s);
	    Delete(s);
	  }
	}
	write_doc(proc_name, signature, doc, signature2);
	Delete(signature);
	if (signature2)
	  Delete(signature2);
	Delete(doc);
      }

    } else {
      Swig_warning(WARN_TYPEMAP_VAR_UNDEF, input_file, line_number, "Unsupported variable type %s (ignored).\n", SwigType_str(t, 0));
    }
    Delete(var_name);
    Delete(proc_name);
    DelWrapper(f);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constantWrapper()
   *
   * We create a read-only variable.
   * ------------------------------------------------------------ */

  virtual int constantWrapper(Node *n) {
    char *name = GetChar(n, "name");
    char *iname = GetChar(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");
    int constasvar = GetFlag(n, "feature:constasvar");


    String *proc_name;
    String *var_name;
    Wrapper *f;
    SwigType *nctype;
    String *tm;

    f = NewWrapper();

    // Make a static variable;
    var_name = NewStringf("%sconst_%s", prefix, iname);

    // Strip const qualifier from type if present

    nctype = NewString(type);
    if (SwigType_isconst(nctype)) {
      Delete(SwigType_pop(nctype));
    }
    // Build the name for scheme.
    proc_name = NewString(iname);
    Replaceall(proc_name, "_", "-");

    if ((SwigType_type(nctype) == T_USER) && (!is_a_pointer(nctype))) {
      Swig_warning(WARN_TYPEMAP_CONST_UNDEF, input_file, line_number, "Unsupported constant value.\n");
      Delete(var_name);
      DelWrapper(f);
      return SWIG_NOWRAP;
    }
    // See if there's a typemap

    if ((tm = Swig_typemap_lookup("constant", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$value", value);
      Replaceall(tm, "$target", name);
      Printv(f_header, tm, "\n", NIL);
    } else {
      // Create variable and assign it a value
      Printf(f_header, "static %s = (%s)(%s);\n", SwigType_str(type, var_name), SwigType_str(type, 0), value);
    }
    {
      /* Hack alert: will cleanup later -- Dave */
      Node *nn = NewHash();
      Setfile(nn, Getfile(n));
      Setline(nn, Getline(n));
      Setattr(nn, "name", var_name);
      Setattr(nn, "sym:name", iname);
      Setattr(nn, "type", nctype);
      SetFlag(nn, "feature:immutable");
      if (constasvar) {
	SetFlag(nn, "feature:constasvar");
      }
      variableWrapper(nn);
      Delete(nn);
    }
    Delete(var_name);
    Delete(nctype);
    Delete(proc_name);
    DelWrapper(f);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * classDeclaration()
   * ------------------------------------------------------------ */
  virtual int classDeclaration(Node *n) {
    String *class_name = NewStringf("<%s>", Getattr(n, "sym:name"));
    Setattr(n, "guile:goopsclassname", class_name);
    return Language::classDeclaration(n);
  }

  /* ------------------------------------------------------------
   * classHandler()
   * ------------------------------------------------------------ */
  virtual int classHandler(Node *n) {
    /* Create new strings for building up a wrapper function */
    have_constructor = 0;

    class_name = NewString("");
    short_class_name = NewString("");
    Printv(class_name, "<", Getattr(n, "sym:name"), ">", NIL);
    Printv(short_class_name, Getattr(n, "sym:name"), NIL);
    Replaceall(class_name, "_", "-");
    Replaceall(short_class_name, "_", "-");

    if (!addSymbol(class_name, n))
      return SWIG_ERROR;

    /* Handle inheritance */
    String *base_class = NewString("<");
    List *baselist = Getattr(n, "bases");
    if (baselist && Len(baselist)) {
      Iterator i = First(baselist);
      while (i.item) {
	Printv(base_class, Getattr(i.item, "sym:name"), NIL);
	i = Next(i);
	if (i.item) {
	  Printf(base_class, "> <");
	}
      }
    }
    Printf(base_class, ">");
    Replaceall(base_class, "_", "-");

    Printv(goopscode, "(define-class ", class_name, " ", NIL);
    Printf(goopsexport, "%s ", class_name);

    if (Len(base_class) > 2) {
      Printv(goopscode, "(", base_class, ")\n", NIL);
    } else {
      Printv(goopscode, "(<swig>)\n", NIL);
    }
    SwigType *ct = NewStringf("p.%s", Getattr(n, "name"));
    swigtype_ptr = SwigType_manglestr(ct);

    String *mangled_classname = Swig_name_mangle(Getattr(n, "sym:name"));
    /* Export clientdata structure */
    Printf(f_runtime, "static swig_guile_clientdata _swig_guile_clientdata%s = { NULL, SCM_EOL };\n", mangled_classname);

    Printv(f_init, "SWIG_TypeClientData(SWIGTYPE", swigtype_ptr, ", (void *) &_swig_guile_clientdata", mangled_classname, ");\n", NIL);
    SwigType_remember(ct);
    Delete(ct);

    /* Emit all of the members */
    goops_class_methods = NewString("");

    in_class = 1;
    Language::classHandler(n);
    in_class = 0;

    Printv(goopscode, "  #:metaclass <swig-metaclass>\n", NIL);

    if (have_constructor)
      Printv(goopscode, "  #:new-function ", primRenamer ? "primitive:" : "", "new-", short_class_name, "\n", NIL);

    Printf(goopscode, ")\n%s\n", goops_class_methods);
    Delete(goops_class_methods);
    goops_class_methods = 0;


    /* export class initialization function */
    if (goops) {
      /* export the wrapper function */
      String *funcName = NewString(mangled_classname);
      Printf(funcName, "_swig_guile_setgoopsclass");
      String *guileFuncName = NewString(funcName);
      Replaceall(guileFuncName, "_", "-");

      Printv(f_wrappers, "static SCM ", funcName, "(SCM cl) \n", NIL);
      Printf(f_wrappers, "#define FUNC_NAME %s\n{\n", guileFuncName);
      Printv(f_wrappers, "  ((swig_guile_clientdata *)(SWIGTYPE", swigtype_ptr, "->clientdata))->goops_class = cl;\n", NIL);
      Printf(f_wrappers, "  return SCM_UNSPECIFIED;\n");
      Printf(f_wrappers, "}\n#undef FUNC_NAME\n\n");

      Printf(f_init, "scm_c_define_gsubr(\"%s\", 1, 0, 0, (swig_guile_proc) %s);\n", guileFuncName, funcName);
      Printf(exported_symbols, "\"%s\", ", guileFuncName);

      /* export the call to the wrapper function */
      Printf(goopscode, "(%s%s %s)\n\n", primRenamer ? "primitive:" : "", guileFuncName, class_name);

      Delete(guileFuncName);
      Delete(funcName);
    }

    Delete(mangled_classname);

    Delete(swigtype_ptr);
    swigtype_ptr = 0;

    Delete(class_name);
    Delete(short_class_name);
    class_name = 0;
    short_class_name = 0;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberfunctionHandler()
   * ------------------------------------------------------------ */
  int memberfunctionHandler(Node *n) {
    String *iname = Getattr(n, "sym:name");
    String *proc = NewString(iname);
    Replaceall(proc, "_", "-");

    memberfunction_name = goopsNameMapping(proc, short_class_name);
    Language::memberfunctionHandler(n);
    Delete(memberfunction_name);
    memberfunction_name = NULL;
    Delete(proc);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * membervariableHandler()
   * ------------------------------------------------------------ */
  int membervariableHandler(Node *n) {
    String *iname = Getattr(n, "sym:name");

    if (emit_setters) {
      struct_member = 1;
      Printf(f_init, "{\n");
    }

    Language::membervariableHandler(n);

    if (emit_setters) {
      Printf(f_init, "}\n");
      struct_member = 0;
    }

    String *proc = NewString(iname);
    Replaceall(proc, "_", "-");
    String *goops_name = goopsNameMapping(proc, short_class_name);

    /* The slot name is never qualified with the class,
       even if useclassprefix is true. */
    Printv(goopscode, "  (", proc, " #:allocation #:virtual", NIL);
    /* GOOPS (at least in Guile 1.6.3) only accepts closures, not
       primitive procedures for slot-ref and slot-set. */
    Printv(goopscode, "\n   #:slot-ref (lambda (obj) (", primRenamer ? "primitive:" : "", short_class_name, "-", proc, "-get", " obj))", NIL);
    if (!GetFlag(n, "feature:immutable")) {
      Printv(goopscode, "\n   #:slot-set! (lambda (obj value) (", primRenamer ? "primitive:" : "", short_class_name, "-", proc, "-set", " obj value))", NIL);
    } else {
      Printf(goopscode, "\n   #:slot-set! (lambda (obj value) (error \"Immutable slot\"))");
    }
    if (emit_slot_accessors) {
      if (GetFlag(n, "feature:immutable")) {
	Printv(goopscode, "\n   #:getter ", goops_name, NIL);
      } else {
	Printv(goopscode, "\n   #:accessor ", goops_name, NIL);
      }
      Printf(goopsexport, "%s ", goops_name);
    }
    Printv(goopscode, ")\n", NIL);
    Delete(proc);
    Delete(goops_name);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constructorHandler()
   * ------------------------------------------------------------ */
  int constructorHandler(Node *n) {
    Language::constructorHandler(n);
    have_constructor = 1;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * destructorHandler()
   * ------------------------------------------------------------ */
  virtual int destructorHandler(Node *n) {
    exporting_destructor = true;
    Language::destructorHandler(n);
    exporting_destructor = false;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * pragmaDirective()
   * ------------------------------------------------------------ */

  virtual int pragmaDirective(Node *n) {
    if (!ImportMode) {
      String *lang = Getattr(n, "lang");
      String *cmd = Getattr(n, "name");
      String *value = Getattr(n, "value");

#     define store_pragma(PRAGMANAME)			\
        if (Strcmp(cmd, #PRAGMANAME) == 0) {		\
	  if (PRAGMANAME) Delete(PRAGMANAME);		\
	  PRAGMANAME = value ? NewString(value) : NULL;	\
	}

      if (Strcmp(lang, "guile") == 0) {
	store_pragma(beforereturn)
	    store_pragma(return_nothing_doc)
	    store_pragma(return_one_doc)
	    store_pragma(return_multi_doc);
#     undef store_pragma
      }
    }
    return Language::pragmaDirective(n);
  }


  /* ------------------------------------------------------------
   * goopsNameMapping()
   * Maps the identifier from C++ to the GOOPS based * on command 
   * line parameters and such.
   * If class_name = "" that means the mapping is for a function or
   * variable not attached to any class.
   * ------------------------------------------------------------ */
  String *goopsNameMapping(String *name, const_String_or_char_ptr class_name) {
    String *n = NewString("");

    if (Strcmp(class_name, "") == 0) {
      // not part of a class, so no class name to prefix
      if (goopsprefix) {
	Printf(n, "%s%s", goopsprefix, name);
      } else {
	Printf(n, "%s", name);
      }
    } else {
      if (useclassprefix) {
	Printf(n, "%s-%s", class_name, name);
      } else {
	if (goopsprefix) {
	  Printf(n, "%s%s", goopsprefix, name);
	} else {
	  Printf(n, "%s", name);
	}
      }
    }
    return n;
  }


  /* ------------------------------------------------------------
   * validIdentifier()
   * ------------------------------------------------------------ */

  virtual int validIdentifier(String *s) {
    char *c = Char(s);
    /* Check whether we have an R5RS identifier.  Guile supports a
       superset of R5RS identifiers, but it's probably a bad idea to use
       those. */
    /* <identifier> --> <initial> <subsequent>* | <peculiar identifier> */
    /* <initial> --> <letter> | <special initial> */
    if (!(isalpha(*c) || (*c == '!') || (*c == '$') || (*c == '%')
	  || (*c == '&') || (*c == '*') || (*c == '/') || (*c == ':')
	  || (*c == '<') || (*c == '=') || (*c == '>') || (*c == '?')
	  || (*c == '^') || (*c == '_') || (*c == '~'))) {
      /* <peculiar identifier> --> + | - | ... */
      if ((strcmp(c, "+") == 0)
	  || strcmp(c, "-") == 0 || strcmp(c, "...") == 0)
	return 1;
      else
	return 0;
    }
    /* <subsequent> --> <initial> | <digit> | <special subsequent> */
    while (*c) {
      if (!(isalnum(*c) || (*c == '!') || (*c == '$') || (*c == '%')
	    || (*c == '&') || (*c == '*') || (*c == '/') || (*c == ':')
	    || (*c == '<') || (*c == '=') || (*c == '>') || (*c == '?')
	    || (*c == '^') || (*c == '_') || (*c == '~') || (*c == '+')
	    || (*c == '-') || (*c == '.') || (*c == '@')))
	return 0;
      c++;
    }
    return 1;
  }

  String *runtimeCode() {
    String *s;
    s = Swig_include_sys("guile_scm_run.swg");
    if (!s) {
      Printf(stderr, "*** Unable to open 'guile_scm_run.swg");
      s = NewString("");
    }
    return s;
  }

  String *defaultExternalRuntimeFilename() {
    return NewString("swigguilerun.h");
  }
};

/* -----------------------------------------------------------------------------
 * swig_guile()    - Instantiate module
 * ----------------------------------------------------------------------------- */

static Language *new_swig_guile() {
  return new GUILE();
}
extern "C" Language *swig_guile(void) {
  return new_swig_guile();
}
