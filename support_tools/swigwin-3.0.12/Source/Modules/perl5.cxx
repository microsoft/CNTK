/* ----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * perl5.cxx
 *
 * Perl5 language module for SWIG.
 * ------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"
#include <ctype.h>

static const char *usage = "\
Perl5 Options (available with -perl5)\n\
     -compat         - Compatibility mode\n\
     -const          - Wrap constants as constants and not variables (implies -proxy)\n\
     -cppcast        - Enable C++ casting operators\n\
     -nocppcast      - Disable C++ casting operators, useful for generating bugs\n\
     -nopm           - Do not generate the .pm file\n\
     -noproxy        - Don't create proxy classes\n\
     -proxy          - Create proxy classes\n\
     -static         - Omit code related to dynamic loading\n\
\n";

static int compat = 0;

static int no_pmfile = 0;

static int export_all = 0;

/*
 * pmfile
 *   set by the -pm flag, overrides the name of the .pm file
 */
static String *pmfile = 0;

/*
 * module
 *   set by the %module directive, e.g. "Xerces". It will determine
 *   the name of the .pm file, and the dynamic library, and the name
 *   used by any module wanting to %import the module.
 */
static String *module = 0;

/*
 * namespace_module
 *   the fully namespace qualified name of the module. It will be used
 *   to set the package namespace in the .pm file, as well as the name
 *   of the initialization methods in the glue library. This will be
 *   the same as module, above, unless the %module directive is given
 *   the 'package' option, e.g. %module(package="Foo::Bar") "baz"
 */
static String       *namespace_module = 0;

/*
 * cmodule
 *   the namespace of the internal glue code, set to the value of
 *   module with a 'c' appended
 */
static String *cmodule = 0;

/*
 * dest_package
 *   an optional namespace to put all classes into. Specified by using
 *   the %module(package="Foo::Bar") "baz" syntax
 */
static String       *dest_package = 0;

static String *command_tab = 0;
static String *constant_tab = 0;
static String *variable_tab = 0;

static File *f_begin = 0;
static File *f_runtime = 0;
static File *f_runtime_h = 0;
static File *f_header = 0;
static File *f_wrappers = 0;
static File *f_directors = 0;
static File *f_directors_h = 0;
static File *f_init = 0;
static File *f_pm = 0;
static String *pm;		/* Package initialization code */
static String *magic;		/* Magic variable wrappers     */

static int staticoption = 0;

// controlling verbose output
static int          verbose = 0;

/* The following variables are used to manage Perl5 classes */

static int blessed = 1;		/* Enable object oriented features */
static int do_constants = 0;	/* Constant wrapping */
static List *classlist = 0;	/* List of classes */
static int have_constructor = 0;
static int have_destructor = 0;
static int have_data_members = 0;
static String *class_name = 0;	/* Name of the class (what Perl thinks it is) */
static String *real_classname = 0;	/* Real name of C/C++ class */
static String *fullclassname = 0;

static String *pcode = 0;	/* Perl code associated with each class */
						  /* static  String   *blessedmembers = 0;     *//* Member data associated with each class */
static int member_func = 0;	/* Set to 1 when wrapping a member function */
static String *func_stubs = 0;	/* Function stubs */
static String *const_stubs = 0;	/* Constant stubs */
static int num_consts = 0;	/* Number of constants */
static String *var_stubs = 0;	/* Variable stubs */
static String *exported = 0;	/* Exported symbols */
static String *pragma_include = 0;
static String *additional_perl_code = 0;	/* Additional Perl code from %perlcode %{ ... %} */
static Hash *operators = 0;
static int have_operators = 0;

class PERL5:public Language {
public:

  PERL5():Language () {
    Clear(argc_template_string);
    Printv(argc_template_string, "items", NIL);
    Clear(argv_template_string);
    Printv(argv_template_string, "ST(%d)", NIL);
    director_language = 1;
  }

  /* Test to see if a type corresponds to something wrapped with a shadow class */
  Node *is_shadow(SwigType *t) {
    Node *n;
    n = classLookup(t);
    /*  Printf(stdout,"'%s' --> '%p'\n", t, n); */
    if (n) {
      if (!Getattr(n, "perl5:proxy")) {
	setclassname(n);
      }
      return Getattr(n, "perl5:proxy");
    }
    return 0;
  }

  /* ------------------------------------------------------------
   * main()
   * ------------------------------------------------------------ */

  virtual void main(int argc, char *argv[]) {
    int i = 1;
    int cppcast = 1;

    SWIG_library_directory("perl5");

    for (i = 1; i < argc; i++) {
      if (argv[i]) {
	if (strcmp(argv[i], "-package") == 0) {
	  Printv(stderr,
		 "*** -package is no longer supported\n*** use the directive '%module A::B::C' in your interface file instead\n*** see the Perl section in the manual for details.\n", NIL);
	  SWIG_exit(EXIT_FAILURE);
	} else if (strcmp(argv[i], "-interface") == 0) {
	  Printv(stderr,
		 "*** -interface is no longer supported\n*** use the directive '%module A::B::C' in your interface file instead\n*** see the Perl section in the manual for details.\n", NIL);
	  SWIG_exit(EXIT_FAILURE);
	} else if (strcmp(argv[i], "-exportall") == 0) {
	  export_all = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-static") == 0) {
	  staticoption = 1;
	  Swig_mark_arg(i);
	} else if ((strcmp(argv[i], "-shadow") == 0) || ((strcmp(argv[i], "-proxy") == 0))) {
	  blessed = 1;
	  Swig_mark_arg(i);
	} else if ((strcmp(argv[i], "-noproxy") == 0)) {
	  blessed = 0;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-const") == 0) {
	  do_constants = 1;
	  blessed = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-nopm") == 0) {
	  no_pmfile = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-pm") == 0) {
	  Swig_mark_arg(i);
	  i++;
	  pmfile = NewString(argv[i]);
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i],"-v") == 0) {
	    Swig_mark_arg(i);
	    verbose++;
	} else if (strcmp(argv[i], "-cppcast") == 0) {
	  cppcast = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-nocppcast") == 0) {
	  cppcast = 0;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-compat") == 0) {
	  compat = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-help") == 0) {
	  fputs(usage, stdout);
	}
      }
    }

    if (cppcast) {
      Preprocessor_define((DOH *) "SWIG_CPLUSPLUS_CAST", 0);
    }

    Preprocessor_define("SWIGPERL 1", 0);
    // SWIGPERL5 is deprecated, and no longer documented.
    Preprocessor_define("SWIGPERL5 1", 0);
    SWIG_typemap_lang("perl5");
    SWIG_config_file("perl5.swg");
    allow_overloading();
  }

  /* ------------------------------------------------------------
   * top()
   * ------------------------------------------------------------ */

  virtual int top(Node *n) {
    /* check if directors are enabled for this module.  note: this 
     * is a "master" switch, without which no director code will be
     * emitted.  %feature("director") statements are also required
     * to enable directors for individual classes or methods.
     *
     * use %module(directors="1") modulename at the start of the 
     * interface file to enable director generation.
     *
     * TODO: directors are disallowed in conjunction with many command
     * line options.  Some of them are probably safe, but it will take 
     * some effort to validate each one.
     */
    {
      Node *mod = Getattr(n, "module");
      if (mod) {
	Node *options = Getattr(mod, "options");
	if (options) {
	  int dirprot = 0;
	  if (Getattr(options, "dirprot"))
	    dirprot = 1;
	  if (Getattr(options, "nodirprot"))
	    dirprot = 0;
	  if (Getattr(options, "directors")) {
	    int allow = 1;
	    if (export_all) {
	      Printv(stderr, "*** directors are not supported with -exportall\n", NIL);
	      allow = 0;
	    }
	    if (staticoption) {
	      Printv(stderr, "*** directors are not supported with -static\n", NIL);
	      allow = 0;
	    }
	    if (!blessed) {
	      Printv(stderr, "*** directors are not supported with -noproxy\n", NIL);
	      allow = 0;
	    }
	    if (no_pmfile) {
	      Printv(stderr, "*** directors are not supported with -nopm\n", NIL);
	      allow = 0;
	    }
	    if (compat) {
	      Printv(stderr, "*** directors are not supported with -compat\n", NIL);
	      allow = 0;
	    }
	    if (allow) {
	      allow_directors();
	      if (dirprot)
		allow_dirprot();
	    }
	  }
	}
      }
    }

    /* Initialize all of the output files */
    String *outfile = Getattr(n, "outfile");
    String *outfile_h = Getattr(n, "outfile_h");

    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      SWIG_exit(EXIT_FAILURE);
    }
    f_runtime = NewString("");
    f_init = NewString("");
    f_header = NewString("");
    f_wrappers = NewString("");
    f_directors_h = NewString("");
    f_directors = NewString("");

    if (directorsEnabled()) {
      f_runtime_h = NewFile(outfile_h, "w", SWIG_output_files());
      if (!f_runtime_h) {
	FileErrorDisplay(outfile_h);
	SWIG_exit(EXIT_FAILURE);
      }
    }

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("header", f_header);
    Swig_register_filebyname("wrapper", f_wrappers);
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", f_init);
    Swig_register_filebyname("director", f_directors);
    Swig_register_filebyname("director_h", f_directors_h);

    classlist = NewList();

    pm = NewString("");
    func_stubs = NewString("");
    var_stubs = NewString("");
    const_stubs = NewString("");
    exported = NewString("");
    magic = NewString("");
    pragma_include = NewString("");
    additional_perl_code = NewString("");

    command_tab = NewString("static swig_command_info swig_commands[] = {\n");
    constant_tab = NewString("static swig_constant_info swig_constants[] = {\n");
    variable_tab = NewString("static swig_variable_info swig_variables[] = {\n");

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGPERL\n#define SWIGPERL\n#endif\n\n");

    if (directorsEnabled()) {
      Printf(f_runtime, "#define SWIG_DIRECTORS\n");
    }
    Printf(f_runtime, "#define SWIG_CASTRANK_MODE\n");
    Printf(f_runtime, "\n");

    // Is the imported module in another package?  (IOW, does it use the
    // %module(package="name") option and it's different than the package
    // of this module.)
    Node *mod = Getattr(n, "module");
    Node *options = Getattr(mod, "options");
    module = Copy(Getattr(n,"name"));

    String *underscore_module = Copy(module);
    Replaceall(underscore_module,":","_");

    if (verbose > 0) {
      fprintf(stdout, "top: using namespace_module: %s\n", Char(namespace_module));
    }

    if (directorsEnabled()) {
      Swig_banner(f_directors_h);
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "#ifndef SWIG_%s_WRAP_H_\n", underscore_module);
      Printf(f_directors_h, "#define SWIG_%s_WRAP_H_\n\n", underscore_module);
      if (dirprot_mode()) {
	Printf(f_directors_h, "#include <map>\n");
	Printf(f_directors_h, "#include <string>\n\n");
      }

      Printf(f_directors, "\n\n");
      Printf(f_directors, "/* ---------------------------------------------------\n");
      Printf(f_directors, " * C++ director class methods\n");
      Printf(f_directors, " * --------------------------------------------------- */\n\n");
      if (outfile_h) {
	String *filename = Swig_file_filename(outfile_h);
	Printf(magic, "#include \"%s\"\n\n", filename);
	Delete(filename);
      }
    }

    if (verbose > 0) {
      fprintf(stdout, "top: using module: %s\n", Char(module));
    }

    dest_package = options ? Getattr(options, "package") : 0;
    if (dest_package) {
      namespace_module = Copy(dest_package);
      if (verbose > 0) {
	fprintf(stdout, "top: Found package: %s\n",Char(dest_package));
      }
    } else {
      namespace_module = Copy(module);
      if (verbose > 0) {
	fprintf(stdout, "top: No package found\n");
      }
    }
    /* If we're in blessed mode, change the package name to "packagec" */

    if (blessed) {
      cmodule = NewStringf("%sc",namespace_module);
    } else {
      cmodule = NewString(namespace_module);
    }

    /* Create a .pm file
     * Need to strip off any prefixes that might be found in
     * the module name */

    if (no_pmfile) {
      f_pm = NewString(0);
    } else {
      if (!pmfile) {
	char *m = Char(module) + Len(module);
	while (m != Char(module)) {
	  if (*m == ':') {
	    m++;
	    break;
	  }
	  m--;
	}
	pmfile = NewStringf("%s.pm", m);
      }
      String *filen = NewStringf("%s%s", SWIG_output_directory(), pmfile);
      if ((f_pm = NewFile(filen, "w", SWIG_output_files())) == 0) {
	FileErrorDisplay(filen);
	SWIG_exit(EXIT_FAILURE);
      }
      Delete(filen);
      filen = NULL;
      Swig_register_filebyname("pm", f_pm);
      Swig_register_filebyname("perl", f_pm);
    }
    {
      String *boot_name = NewStringf("boot_%s", underscore_module);
      Printf(f_header,"#define SWIG_init    %s\n\n", boot_name);
      Printf(f_header,"#define SWIG_name   \"%s::%s\"\n", cmodule, boot_name);
      Printf(f_header,"#define SWIG_prefix \"%s::\"\n", cmodule);
      Delete(boot_name);
    }

    Swig_banner_target_lang(f_pm, "#");
    Printf(f_pm, "\n");

    Printf(f_pm, "package %s;\n", module);

    /* 
     * If the package option has been given we are placing our
     *   symbols into some other packages namespace, so we do not
     *   mess with @ISA or require for that package
     */
    if (dest_package) {
      Printf(f_pm,"use base qw(DynaLoader);\n");
    } else {
      Printf(f_pm,"use base qw(Exporter);\n");
      if (!staticoption) {
	Printf(f_pm,"use base qw(DynaLoader);\n");
      }
    }

    /* Start creating magic code */

    Printv(magic,
           "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n",
	   "#ifdef PERL_OBJECT\n",
	   "#define MAGIC_CLASS _wrap_", underscore_module, "_var::\n",
	   "class _wrap_", underscore_module, "_var : public CPerlObj {\n",
	   "public:\n",
	   "#else\n",
	   "#define MAGIC_CLASS\n",
	   "#endif\n",
	   "SWIGCLASS_STATIC int swig_magic_readonly(pTHX_ SV *SWIGUNUSEDPARM(sv), MAGIC *SWIGUNUSEDPARM(mg)) {\n",
	   tab4, "MAGIC_PPERL\n", tab4, "croak(\"Value is read-only.\");\n", tab4, "return 0;\n", "}\n", NIL);

    Printf(f_wrappers, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n");

    /* emit wrappers */
    Language::top(n);

    if (directorsEnabled()) {
      // Insert director runtime into the f_runtime file (make it occur before %header section)
      Swig_insert_file("director_common.swg", f_runtime);
      Swig_insert_file("director.swg", f_runtime);
    }

    String *base = NewString("");

    /* Dump out variable wrappers */

    Printv(magic, "\n\n#ifdef PERL_OBJECT\n", "};\n", "#endif\n", NIL);
    Printv(magic, "\n#ifdef __cplusplus\n}\n#endif\n", NIL);

    Printf(f_header, "%s\n", magic);

    String *type_table = NewString("");

    /* Patch the type table to reflect the names used by shadow classes */
    if (blessed) {
      Iterator cls;
      for (cls = First(classlist); cls.item; cls = Next(cls)) {
	String *pname = Getattr(cls.item, "perl5:proxy");
	if (pname) {
	  SwigType *type = Getattr(cls.item, "classtypeobj");
	  if (!type)
	    continue;		/* If unnamed class, no type will be found */
	  type = Copy(type);

	  SwigType_add_pointer(type);
	  String *mangled = SwigType_manglestr(type);
	  SwigType_remember_mangleddata(mangled, NewStringf("\"%s\"", pname));
	  Delete(type);
	  Delete(mangled);
	}
      }
    }
    SwigType_emit_type_table(f_runtime, type_table);

    Printf(f_wrappers, "%s", type_table);
    Delete(type_table);

    Printf(constant_tab, "{0,0,0,0,0,0}\n};\n");
    Printv(f_wrappers, constant_tab, NIL);

    Printf(f_wrappers, "#ifdef __cplusplus\n}\n#endif\n");

    Printf(f_init, "\t ST(0) = &PL_sv_yes;\n");
    Printf(f_init, "\t XSRETURN(1);\n");
    Printf(f_init, "}\n");

    /* Finish off tables */
    Printf(variable_tab, "{0,0,0,0}\n};\n");
    Printv(f_wrappers, variable_tab, NIL);

    Printf(command_tab, "{0,0}\n};\n");
    Printv(f_wrappers, command_tab, NIL);


    Printf(f_pm, "package %s;\n", cmodule);

    if (!staticoption) {
      Printf(f_pm,"bootstrap %s;\n", module);
    } else {
      Printf(f_pm,"package %s;\n", cmodule);
      Printf(f_pm,"boot_%s();\n", underscore_module);
    }

    Printf(f_pm, "package %s;\n", module);
    /* 
     * If the package option has been given we are placing our
     *   symbols into some other packages namespace, so we do not
     *   mess with @EXPORT
     */
    if (!dest_package) {
      Printf(f_pm,"@EXPORT = qw(%s);\n", exported);
    }

    Printf(f_pm, "%s", pragma_include);

    if (blessed) {

      /*
       * These methods will be duplicated if package 
       *   has been specified, so we do not output them
       */
      if (!dest_package) {
	Printv(base, "\n# ---------- BASE METHODS -------------\n\n", "package ", namespace_module, ";\n\n", NIL);

	/* Write out the TIE method */

	Printv(base, "sub TIEHASH {\n", tab4, "my ($classname,$obj) = @_;\n", tab4, "return bless $obj, $classname;\n", "}\n\n", NIL);

	/* Output a CLEAR method.   This is just a place-holder, but by providing it we
	 * can make declarations such as
	 *     %$u = ( x => 2, y=>3, z =>4 );
	 *
	 * Where x,y,z are the members of some C/C++ object. */

	Printf(base, "sub CLEAR { }\n\n");

	/* Output default firstkey/nextkey methods */

	Printf(base, "sub FIRSTKEY { }\n\n");
	Printf(base, "sub NEXTKEY { }\n\n");

	/* Output a FETCH method.  This is actually common to all classes */
	Printv(base,
	       "sub FETCH {\n",
	       tab4, "my ($self,$field) = @_;\n", tab4, "my $member_func = \"swig_${field}_get\";\n", tab4, "$self->$member_func();\n", "}\n\n", NIL);

	/* Output a STORE method.   This is also common to all classes (might move to base class) */

	Printv(base,
	       "sub STORE {\n",
	       tab4, "my ($self,$field,$newval) = @_;\n",
	       tab4, "my $member_func = \"swig_${field}_set\";\n", tab4, "$self->$member_func($newval);\n", "}\n\n", NIL);

	/* Output a 'this' method */

	Printv(base, "sub this {\n", tab4, "my $ptr = shift;\n", tab4, "return tied(%$ptr);\n", "}\n\n", NIL);

	Printf(f_pm, "%s", base);
      }

      /* Emit function stubs for stand-alone functions */
      Printf(f_pm, "\n# ------- FUNCTION WRAPPERS --------\n\n");
      Printf(f_pm, "package %s;\n\n", namespace_module);
      Printf(f_pm, "%s", func_stubs);

      /* Emit package code for different classes */
      Printf(f_pm, "%s", pm);

      if (num_consts > 0) {
	/* Emit constant stubs */
	Printf(f_pm, "\n# ------- CONSTANT STUBS -------\n\n");
	Printf(f_pm, "package %s;\n\n", namespace_module);
	Printf(f_pm, "%s", const_stubs);
      }

      /* Emit variable stubs */

      Printf(f_pm, "\n# ------- VARIABLE STUBS --------\n\n");
      Printf(f_pm, "package %s;\n\n", namespace_module);
      Printf(f_pm, "%s", var_stubs);
    }

    /* Add additional Perl code at the end */
    Printf(f_pm, "%s", additional_perl_code);

    Printf(f_pm, "1;\n");
    Delete(f_pm);
    Delete(base);
    Delete(dest_package);
    Delete(underscore_module);

    /* Close all of the files */
    Dump(f_runtime, f_begin);
    Dump(f_header, f_begin);

    if (directorsEnabled()) {
      Dump(f_directors_h, f_runtime_h);
      Printf(f_runtime_h, "\n");
      Printf(f_runtime_h, "#endif\n");
      Dump(f_directors, f_begin);
    }

    Dump(f_wrappers, f_begin);
    Wrapper_pretty_print(f_init, f_begin);
    Delete(f_header);
    Delete(f_wrappers);
    Delete(f_init);
    Delete(f_directors);
    Delete(f_directors_h);
    Delete(f_runtime);
    Delete(f_begin);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * importDirective(Node *n)
   * ------------------------------------------------------------ */

  virtual int importDirective(Node *n) {
    if (blessed) {
      String *modname = Getattr(n, "module");
      if (modname) {
	Printf(f_pm, "require %s;\n", modname);
      }
    }
    return Language::importDirective(n);
  }

  /* ------------------------------------------------------------
   * functionWrapper()
   * ------------------------------------------------------------ */

  virtual int functionWrapper(Node *n) {
    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *d = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    String *overname = 0;
    int director_method = 0;

    Parm *p;
    int i;
    Wrapper *f;
    char source[256], temp[256];
    String *tm;
    String *cleanup, *outarg;
    int num_saved = 0;
    int num_arguments, num_required;
    int varargs = 0;

    if (Getattr(n, "sym:overloaded")) {
      overname = Getattr(n, "sym:overname");
    } else {
      if (!addSymbol(iname, n))
	return SWIG_ERROR;
    }

    f = NewWrapper();
    cleanup = NewString("");
    outarg = NewString("");

    String *wname = Swig_name_wrapper(iname);
    if (overname) {
      Append(wname, overname);
    }
    Setattr(n, "wrap:name", wname);
    Printv(f->def, "XS(", wname, ") {\n", "{\n",	/* scope to destroy C++ objects before croaking */
	   NIL);

    emit_parameter_variables(l, f);
    emit_attach_parmmaps(l, f);
    Setattr(n, "wrap:parms", l);

    num_arguments = emit_num_arguments(l);
    num_required = emit_num_required(l);
    varargs = emit_isvarargs(l);

    Wrapper_add_local(f, "argvi", "int argvi = 0");

    /* Check the number of arguments */
    if (!varargs) {
      Printf(f->code, "    if ((items < %d) || (items > %d)) {\n", num_required, num_arguments);
    } else {
      Printf(f->code, "    if (items < %d) {\n", num_required);
    }
    Printf(f->code, "        SWIG_croak(\"Usage: %s\");\n", usage_func(Char(iname), d, l));
    Printf(f->code, "}\n");

    /* Write code to extract parameters. */
    for (i = 0, p = l; i < num_arguments; i++) {

      /* Skip ignored arguments */

      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");

      /* Produce string representation of source and target arguments */
      sprintf(source, "ST(%d)", i);
      String *target = Getattr(p, "lname");

      if (i >= num_required) {
	Printf(f->code, "    if (items > %d) {\n", i);
      }
      if ((tm = Getattr(p, "tmap:in"))) {
	Replaceall(tm, "$target", target);
	Replaceall(tm, "$source", source);
	Replaceall(tm, "$input", source);
	Setattr(p, "emit:input", source);	/* Save input location */

	if (Getattr(p, "wrap:disown") || (Getattr(p, "tmap:in:disown"))) {
	  Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
	} else {
	  Replaceall(tm, "$disown", "0");
	}

	Printf(f->code, "%s\n", tm);
	p = Getattr(p, "tmap:in:next");
      } else {
	Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
	p = nextSibling(p);
      }
      if (i >= num_required) {
	Printf(f->code, "    }\n");
      }
    }

    if (varargs) {
      if (p && (tm = Getattr(p, "tmap:in"))) {
	sprintf(source, "ST(%d)", i);
	Replaceall(tm, "$input", source);
	Setattr(p, "emit:input", source);
	Printf(f->code, "if (items >= %d) {\n", i);
	Printv(f->code, tm, "\n", NIL);
	Printf(f->code, "}\n");
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
    for (i = 0, p = l; p; i++) {
      if ((tm = Getattr(p, "tmap:freearg"))) {
	Replaceall(tm, "$source", Getattr(p, "lname"));
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(cleanup, tm, "\n", NIL);
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert argument output code */
    num_saved = 0;
    for (i = 0, p = l; p; i++) {
      if ((tm = Getattr(p, "tmap:argout"))) {
	SwigType *t = Getattr(p, "type");
	Replaceall(tm, "$source", Getattr(p, "lname"));
	Replaceall(tm, "$target", "ST(argvi)");
	Replaceall(tm, "$result", "ST(argvi)");
	if (is_shadow(t)) {
	  Replaceall(tm, "$shadow", "SWIG_SHADOW");
	} else {
	  Replaceall(tm, "$shadow", "0");
	}

	String *in = Getattr(p, "emit:input");
	if (in) {
	  sprintf(temp, "_saved[%d]", num_saved);
	  Replaceall(tm, "$arg", temp);
	  Replaceall(tm, "$input", temp);
	  Printf(f->code, "_saved[%d] = %s;\n", num_saved, in);
	  num_saved++;
	}
	Printv(outarg, tm, "\n", NIL);
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* If there were any saved arguments, emit a local variable for them */
    if (num_saved) {
      sprintf(temp, "_saved[%d]", num_saved);
      Wrapper_add_localv(f, "_saved", "SV *", temp, NIL);
    }

    director_method = is_member_director(n) && !is_smart_pointer() && 0 != Cmp(nodeType(n), "destructor");
    if (director_method) {
      Wrapper_add_local(f, "director", "Swig::Director *director = 0");
      Append(f->code, "director = SWIG_DIRECTOR_CAST(arg1);\n");
      if (dirprot_mode() && !is_public(n)) {
	Printf(f->code, "if (!director || !(director->swig_get_inner(\"%s\"))) {\n", name);
	Printf(f->code, "SWIG_exception_fail(SWIG_RuntimeError, \"accessing protected member %s\");\n", name);
	Append(f->code, "}\n");
      }
      Wrapper_add_local(f, "upcall", "bool upcall = false");
      Printf(f->code, "upcall = director && SvSTASH(SvRV(ST(0))) == gv_stashpv(director->swig_get_class(), 0);\n");
    }

    /* Emit the function call */
    if (director_method) {
      Append(f->code, "try {\n");
    }

    /* Now write code to make the function call */

    Swig_director_emit_dynamic_cast(n, f);
    String *actioncode = emit_action(n);

    if (director_method) {
      Append(actioncode, "} catch (Swig::DirectorException& swig_err) {\n");
      Append(actioncode, "  sv_setsv(ERRSV, swig_err.getNative());\n");
      Append(actioncode, "  SWIG_fail;\n");
      Append(actioncode, "}\n");
    }

    if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
      SwigType *t = Getattr(n, "type");
      Replaceall(tm, "$source", Swig_cresult_name());
      Replaceall(tm, "$target", "ST(argvi)");
      Replaceall(tm, "$result", "ST(argvi)");
      if (is_shadow(t)) {
	Replaceall(tm, "$shadow", "SWIG_SHADOW");
      } else {
	Replaceall(tm, "$shadow", "0");
      }
      if (GetFlag(n, "feature:new")) {
	Replaceall(tm, "$owner", "SWIG_OWNER");
      } else {
	Replaceall(tm, "$owner", "0");
      }
      Printf(f->code, "%s\n", tm);
    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(d, 0), name);
    }
    emit_return_variable(n, d, f);

    /* If there were any output args, take care of them. */

    Printv(f->code, outarg, NIL);

    /* If there was any cleanup, do that. */

    Printv(f->code, cleanup, NIL);

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

    Printv(f->code, "XSRETURN(argvi);\n", "fail:\n", cleanup, "SWIG_croak_null();\n" "}\n" "}\n", NIL);

    /* Add the dXSARGS last */

    Wrapper_add_local(f, "dXSARGS", "dXSARGS");

    /* Substitute the cleanup code */
    Replaceall(f->code, "$cleanup", cleanup);
    Replaceall(f->code, "$symname", iname);

    /* Dump the wrapper function */

    Wrapper_print(f, f_wrappers);

    /* Now register the function */

    if (!Getattr(n, "sym:overloaded")) {
      Printf(command_tab, "{\"%s::%s\", %s},\n", cmodule, iname, wname);
    } else if (!Getattr(n, "sym:nextSibling")) {
      /* Generate overloaded dispatch function */
      int maxargs;
      String *dispatch = Swig_overload_dispatch_cast(n, "PUSHMARK(MARK); SWIG_CALLXS(%s); return;", &maxargs);

      /* Generate a dispatch wrapper for all overloaded functions */

      Wrapper *df = NewWrapper();
      String *dname = Swig_name_wrapper(iname);

      Printv(df->def, "XS(", dname, ") {\n", NIL);

      Wrapper_add_local(df, "dXSARGS", "dXSARGS");
      Printv(df->code, dispatch, "\n", NIL);
      Printf(df->code, "croak(\"No matching function for overloaded '%s'\");\n", iname);
      Printf(df->code, "XSRETURN(0);\n");
      Printv(df->code, "}\n", NIL);
      Wrapper_print(df, f_wrappers);
      Printf(command_tab, "{\"%s::%s\", %s},\n", cmodule, iname, dname);
      DelWrapper(df);
      Delete(dispatch);
      Delete(dname);
    }
    if (!Getattr(n, "sym:nextSibling")) {
      if (export_all) {
	Printf(exported, "%s ", iname);
      }

      /* --------------------------------------------------------------------
       * Create a stub for this function, provided it's not a member function
       * -------------------------------------------------------------------- */

      if ((blessed) && (!member_func)) {
	Printv(func_stubs, "*", iname, " = *", cmodule, "::", iname, ";\n", NIL);
      }

    }
    Delete(cleanup);
    Delete(outarg);
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
    Wrapper *getf, *setf;
    String *tm;
    String *getname = Swig_name_get(NSPACE_TODO, iname);
    String *setname = Swig_name_set(NSPACE_TODO, iname);

    String *get_name = Swig_name_wrapper(getname);
    String *set_name = Swig_name_wrapper(setname);

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    getf = NewWrapper();
    setf = NewWrapper();

    /* Create a Perl function for setting the variable value */

    if (!GetFlag(n, "feature:immutable")) {
      Setattr(n, "wrap:name", set_name);
      Printf(setf->def, "SWIGCLASS_STATIC int %s(pTHX_ SV* sv, MAGIC * SWIGUNUSEDPARM(mg)) {\n", set_name);
      Printv(setf->code, tab4, "MAGIC_PPERL\n", NIL);

      /* Check for a few typemaps */
      tm = Swig_typemap_lookup("varin", n, name, 0);
      if (tm) {
	Replaceall(tm, "$source", "sv");
	Replaceall(tm, "$target", name);
	Replaceall(tm, "$input", "sv");
	/* Printf(setf->code,"%s\n", tm); */
	emit_action_code(n, setf->code, tm);
      } else {
	Swig_warning(WARN_TYPEMAP_VARIN_UNDEF, input_file, line_number, "Unable to set variable of type %s.\n", SwigType_str(t, 0));
	DelWrapper(setf);
	DelWrapper(getf);
	return SWIG_NOWRAP;
      }
      Printf(setf->code, "fail:\n");
      Printf(setf->code, "    return 1;\n}\n");
      Replaceall(setf->code, "$symname", iname);
      Wrapper_print(setf, magic);
    }

    /* Now write a function to evaluate the variable */
    Setattr(n, "wrap:name", get_name);
    int addfail = 0;
    Printf(getf->def, "SWIGCLASS_STATIC int %s(pTHX_ SV *sv, MAGIC *SWIGUNUSEDPARM(mg)) {\n", get_name);
    Printv(getf->code, tab4, "MAGIC_PPERL\n", NIL);

    if ((tm = Swig_typemap_lookup("varout", n, name, 0))) {
      Replaceall(tm, "$target", "sv");
      Replaceall(tm, "$result", "sv");
      Replaceall(tm, "$source", name);
      if (is_shadow(t)) {
	Replaceall(tm, "$shadow", "SWIG_SHADOW");
      } else {
	Replaceall(tm, "$shadow", "0");
      }
      /* Printf(getf->code,"%s\n", tm); */
      addfail = emit_action_code(n, getf->code, tm);
    } else {
      Swig_warning(WARN_TYPEMAP_VAROUT_UNDEF, input_file, line_number, "Unable to read variable of type %s\n", SwigType_str(t, 0));
      DelWrapper(setf);
      DelWrapper(getf);
      return SWIG_NOWRAP;
    }
    Printf(getf->code, "    return 1;\n");
    if (addfail) {
      Append(getf->code, "fail:\n");
      Append(getf->code, "  return 0;\n");
    }
    Append(getf->code, "}\n");


    Replaceall(getf->code, "$symname", iname);
    Wrapper_print(getf, magic);

    String *tt = Getattr(n, "tmap:varout:type");
    if (tt) {
      tt = NewStringf("&%s", tt);
    } else {
      tt = NewString("0");
    }
    /* Now add symbol to the PERL interpreter */
    if (GetFlag(n, "feature:immutable")) {
      Printv(variable_tab, tab4, "{ \"", cmodule, "::", iname, "\", MAGIC_CLASS swig_magic_readonly, MAGIC_CLASS ", get_name, ",", tt, " },\n", NIL);

    } else {
      Printv(variable_tab, tab4, "{ \"", cmodule, "::", iname, "\", MAGIC_CLASS ", set_name, ", MAGIC_CLASS ", get_name, ",", tt, " },\n", NIL);
    }

    /* If we're blessed, try to figure out what to do with the variable
       1.  If it's a Perl object of some sort, create a tied-hash
       around it.
       2.  Otherwise, just hack Perl's symbol table */

    if (blessed) {
      if (is_shadow(t)) {
	Printv(var_stubs,
	       "\nmy %__", iname, "_hash;\n",
	       "tie %__", iname, "_hash,\"", is_shadow(t), "\", $",
	       cmodule, "::", iname, ";\n", "$", iname, "= \\%__", iname, "_hash;\n", "bless $", iname, ", ", is_shadow(t), ";\n", NIL);
      } else {
	Printv(var_stubs, "*", iname, " = *", cmodule, "::", iname, ";\n", NIL);
      }
    }
    if (export_all)
      Printf(exported, "$%s ", iname);

    Delete(tt);
    DelWrapper(setf);
    DelWrapper(getf);
    Delete(getname);
    Delete(setname);
    Delete(set_name);
    Delete(get_name);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constantWrapper()
   * ------------------------------------------------------------ */

  virtual int constantWrapper(Node *n) {
    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");
    String *tm;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

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
      if (is_shadow(type)) {
	Replaceall(tm, "$shadow", "SWIG_SHADOW");
      } else {
	Replaceall(tm, "$shadow", "0");
      }
      Printf(constant_tab, "%s,\n", tm);
    } else if ((tm = Swig_typemap_lookup("constcode", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", name);
      Replaceall(tm, "$value", value);
      if (is_shadow(type)) {
	Replaceall(tm, "$shadow", "SWIG_SHADOW");
      } else {
	Replaceall(tm, "$shadow", "0");
      }
      Printf(f_init, "%s\n", tm);
    } else {
      Swig_warning(WARN_TYPEMAP_CONST_UNDEF, input_file, line_number, "Unsupported constant value.\n");
      return SWIG_NOWRAP;
    }

    if (blessed) {
      if (is_shadow(type)) {
	Printv(var_stubs,
	       "\nmy %__", iname, "_hash;\n",
	       "tie %__", iname, "_hash,\"", is_shadow(type), "\", $",
	       cmodule, "::", iname, ";\n", "$", iname, "= \\%__", iname, "_hash;\n", "bless $", iname, ", ", is_shadow(type), ";\n", NIL);
      } else if (do_constants) {
	Printv(const_stubs, "sub ", name, " () { $", cmodule, "::", name, " }\n", NIL);
	num_consts++;
      } else {
	Printv(var_stubs, "*", iname, " = *", cmodule, "::", iname, ";\n", NIL);
      }
    }
    if (export_all) {
      if (do_constants && !is_shadow(type)) {
	Printf(exported, "%s ", name);
      } else {
	Printf(exported, "$%s ", iname);
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * usage_func()
   * ------------------------------------------------------------ */
  char *usage_func(char *iname, SwigType *, ParmList *l) {
    static String *temp = 0;
    Parm *p;
    int i;

    if (!temp)
      temp = NewString("");
    Clear(temp);
    Printf(temp, "%s(", iname);

    /* Now go through and print parameters */
    p = l;
    i = 0;
    while (p != 0) {
      SwigType *pt = Getattr(p, "type");
      String *pn = Getattr(p, "name");
      if (!checkAttribute(p,"tmap:in:numinputs","0")) {
	/* If parameter has been named, use that.   Otherwise, just print a type  */
	if (SwigType_type(pt) != T_VOID) {
	  if (Len(pn) > 0) {
	    Printf(temp, "%s", pn);
	  } else {
	    Printf(temp, "%s", SwigType_str(pt, 0));
	  }
	}
	i++;
	p = nextSibling(p);
	if (p)
	  if (!checkAttribute(p,"tmap:in:numinputs","0"))
	    Putc(',', temp);
      } else {
	p = nextSibling(p);
	if (p)
	  if ((i > 0) && (!checkAttribute(p,"tmap:in:numinputs","0")))
	    Putc(',', temp);
      }
    }
    Printf(temp, ");");
    return Char(temp);
  }

  /* ------------------------------------------------------------
   * nativeWrapper()
   * ------------------------------------------------------------ */

  virtual int nativeWrapper(Node *n) {
    String *name = Getattr(n, "sym:name");
    String *funcname = Getattr(n, "wrap:name");

    if (!addSymbol(funcname, n))
      return SWIG_ERROR;

    Printf(command_tab, "{\"%s::%s\", %s},\n", cmodule, name, funcname);
    if (export_all)
      Printf(exported, "%s ", name);
    if (blessed) {
      Printv(func_stubs, "*", name, " = *", cmodule, "::", name, ";\n", NIL);
    }
    return SWIG_OK;
  }

/* ----------------------------------------------------------------------------
 *                      OBJECT-ORIENTED FEATURES
 *
 * These extensions provide a more object-oriented interface to C++
 * classes and structures.    The code here is based on extensions
 * provided by David Fletcher and Gary Holt.
 *
 * I have generalized these extensions to make them more general purpose
 * and to resolve object-ownership problems.
 *
 * The approach here is very similar to the Python module :
 *       1.   All of the original methods are placed into a single
 *            package like before except that a 'c' is appended to the
 *            package name.
 *
 *       2.   All methods and function calls are wrapped with a new
 *            perl function.   While possibly inefficient this allows
 *            us to catch complex function arguments (which are hard to
 *            track otherwise).
 *
 *       3.   Classes are represented as tied-hashes in a manner similar
 *            to Gary Holt's extension.   This allows us to access
 *            member data.
 *
 *       4.   Stand-alone (global) C functions are modified to take
 *            tied hashes as arguments for complex datatypes (if
 *            appropriate).
 *
 *       5.   Global variables involving a class/struct is encapsulated
 *            in a tied hash.
 *
 * ------------------------------------------------------------------------- */


  void setclassname(Node *n) {
    String *symname = Getattr(n, "sym:name");
    String *fullname;
    String *actualpackage;
    Node *clsmodule = Getattr(n, "module");

    if (!clsmodule) {
      /* imported module does not define a module name.   Oh well */
      return;
    }

    /* Do some work on the class name */
    if (verbose > 0) {
      String *modulename = Getattr(clsmodule, "name");
      fprintf(stdout, "setclassname: Found sym:name: %s\n", Char(symname));
      fprintf(stdout, "setclassname: Found module: %s\n", Char(modulename));
      fprintf(stdout, "setclassname: No package found\n");
    }

    if (dest_package) {
      fullname = NewStringf("%s::%s", namespace_module, symname);
    } else {
      actualpackage = Getattr(clsmodule,"name");

      if (verbose > 0) {
	fprintf(stdout, "setclassname: Found actualpackage: %s\n", Char(actualpackage));
      }
      if ((!compat) && (!Strchr(symname,':'))) {
	fullname = NewStringf("%s::%s",actualpackage,symname);
      } else {
	fullname = NewString(symname);
      }
    }
    if (verbose > 0) {
      fprintf(stdout, "setclassname: setting proxy: %s\n", Char(fullname));
    }
    Setattr(n, "perl5:proxy", fullname);
  }

  /* ------------------------------------------------------------
   * classDeclaration()
   * ------------------------------------------------------------ */
  virtual int classDeclaration(Node *n) {
    /* Do some work on the class name */
    if (!Getattr(n, "feature:onlychildren")) {
      if (blessed) {
	setclassname(n);
	Append(classlist, n);
      }
    }

    return Language::classDeclaration(n);
  }

  /* ------------------------------------------------------------
   * classHandler()
   * ------------------------------------------------------------ */

  virtual int classHandler(Node *n) {

    if (blessed) {
      have_constructor = 0;
      have_operators = 0;
      have_destructor = 0;
      have_data_members = 0;
      operators = NewHash();

      class_name = Getattr(n, "sym:name");

      if (!addSymbol(class_name, n))
	return SWIG_ERROR;

      /* Use the fully qualified name of the Perl class */
      if (!compat) {
	fullclassname = NewStringf("%s::%s", namespace_module, class_name);
      } else {
	fullclassname = NewString(class_name);
      }
      real_classname = Getattr(n, "name");
      pcode = NewString("");
      // blessedmembers = NewString("");
    }

    /* Emit all of the members */
    Language::classHandler(n);


    /* Finish the rest of the class */
    if (blessed) {
      /* Generate a client-data entry */
      SwigType *ct = NewStringf("p.%s", real_classname);
      Printv(f_init, "SWIG_TypeClientData(SWIGTYPE", SwigType_manglestr(ct), ", (void*) \"", fullclassname, "\");\n", NIL);
      SwigType_remember(ct);
      Delete(ct);

      Printv(pm, "\n############# Class : ", fullclassname, " ##############\n", "\npackage ", fullclassname, ";\n", NIL);

      if (have_operators) {
	Printf(pm, "use overload\n");
	Iterator ki;
	for (ki = First(operators); ki.key; ki = Next(ki)) {
	  char *name = Char(ki.key);
	  //        fprintf(stderr,"found name: <%s>\n", name);
	  if (strstr(name, "__eq__")) {
	    Printv(pm, tab4, "\"==\" => sub { $_[0]->__eq__($_[1])},\n",NIL);
	  } else if (strstr(name, "__ne__")) {
	    Printv(pm, tab4, "\"!=\" => sub { $_[0]->__ne__($_[1])},\n",NIL);
	    // there are no tests for this in operator_overload_runme.pl
	    // it is likely to be broken
	    //	  } else if (strstr(name, "__assign__")) {
	    //	    Printv(pm, tab4, "\"=\" => sub { $_[0]->__assign__($_[1])},\n",NIL);
	  } else if (strstr(name, "__str__")) {
	    Printv(pm, tab4, "'\"\"' => sub { $_[0]->__str__()},\n",NIL);
	  } else if (strstr(name, "__plusplus__")) {
	    Printv(pm, tab4, "\"++\" => sub { $_[0]->__plusplus__()},\n",NIL);
	  } else if (strstr(name, "__minmin__")) {
	    Printv(pm, tab4, "\"--\" => sub { $_[0]->__minmin__()},\n",NIL);
	  } else if (strstr(name, "__add__")) {
	    Printv(pm, tab4, "\"+\" => sub { $_[0]->__add__($_[1])},\n",NIL);
	  } else if (strstr(name, "__sub__")) {
	    Printv(pm, tab4, "\"-\" => sub {  if( not $_[2] ) { $_[0]->__sub__($_[1]) }\n",NIL);
	    Printv(pm, tab8, "elsif( $_[0]->can('__rsub__') ) { $_[0]->__rsub__($_[1]) }\n",NIL);
	    Printv(pm, tab8, "else { die(\"reverse subtraction not supported\") }\n",NIL);
	    Printv(pm, tab8, "},\n",NIL);
	  } else if (strstr(name, "__mul__")) {
	    Printv(pm, tab4, "\"*\" => sub { $_[0]->__mul__($_[1])},\n",NIL);
	  } else if (strstr(name, "__div__")) {
	    Printv(pm, tab4, "\"/\" => sub { $_[0]->__div__($_[1])},\n",NIL);
	  } else if (strstr(name, "__mod__")) {
	    Printv(pm, tab4, "\"%\" => sub { $_[0]->__mod__($_[1])},\n",NIL);
	    // there are no tests for this in operator_overload_runme.pl
	    // it is likely to be broken
	    //	  } else if (strstr(name, "__and__")) {
	    //	    Printv(pm, tab4, "\"&\" => sub { $_[0]->__and__($_[1])},\n",NIL);

	    // there are no tests for this in operator_overload_runme.pl
	    // it is likely to be broken
	    //	  } else if (strstr(name, "__or__")) {
	    //	    Printv(pm, tab4, "\"|\" => sub { $_[0]->__or__($_[1])},\n",NIL);
	  } else if (strstr(name, "__gt__")) {
	    Printv(pm, tab4, "\">\" => sub { $_[0]->__gt__($_[1])},\n",NIL);
          } else if (strstr(name, "__ge__")) {
            Printv(pm, tab4, "\">=\" => sub { $_[0]->__ge__($_[1])},\n",NIL);
	  } else if (strstr(name, "__not__")) {
	    Printv(pm, tab4, "\"!\" => sub { $_[0]->__not__()},\n",NIL);
	  } else if (strstr(name, "__lt__")) {
	    Printv(pm, tab4, "\"<\" => sub { $_[0]->__lt__($_[1])},\n",NIL);
          } else if (strstr(name, "__le__")) {
            Printv(pm, tab4, "\"<=\" => sub { $_[0]->__le__($_[1])},\n",NIL);
	  } else if (strstr(name, "__pluseq__")) {
	    Printv(pm, tab4, "\"+=\" => sub { $_[0]->__pluseq__($_[1])},\n",NIL);
	  } else if (strstr(name, "__mineq__")) {
	    Printv(pm, tab4, "\"-=\" => sub { $_[0]->__mineq__($_[1])},\n",NIL);
	  } else if (strstr(name, "__neg__")) {
	    Printv(pm, tab4, "\"neg\" => sub { $_[0]->__neg__()},\n",NIL);
	  } else {
	    fprintf(stderr,"Unknown operator: %s\n", name);
	  }
	}
	Printv(pm, tab4,
               "\"=\" => sub { my $class = ref($_[0]); $class->new($_[0]) },\n", NIL);
	Printv(pm, tab4, "\"fallback\" => 1;\n", NIL);
      }
      // make use strict happy
      Printv(pm, "use vars qw(@ISA %OWNER %ITERATORS %BLESSEDMEMBERS);\n", NIL);

      /* If we are inheriting from a base class, set that up */

      Printv(pm, "@ISA = qw(", NIL);

      /* Handle inheritance */
      List *baselist = Getattr(n, "bases");
      if (baselist && Len(baselist)) {
	Iterator b;
	b = First(baselist);
	while (b.item) {
	  String *bname = Getattr(b.item, "perl5:proxy");
	  if (!bname) {
	    b = Next(b);
	    continue;
	  }
	  Printv(pm, " ", bname, NIL);
	  b = Next(b);
	}
      }

      /* Module comes last */
      if (!compat || Cmp(namespace_module, fullclassname)) {
	Printv(pm, " ", namespace_module, NIL);
      }

      Printf(pm, " );\n");

      /* Dump out a hash table containing the pointers that we own */
      Printf(pm, "%%OWNER = ();\n");
      if (have_data_members || have_destructor)
	Printf(pm, "%%ITERATORS = ();\n");

      /* Dump out the package methods */

      Printv(pm, pcode, NIL);
      Delete(pcode);

      /* Output methods for managing ownership */

      String *director_disown;
      if (Getattr(n, "perl5:directordisown")) {
	director_disown = NewStringf("%s%s($self);\n", tab4, Getattr(n, "perl5:directordisown"));
      } else {
	director_disown = NewString("");
      }
      Printv(pm,
	     "sub DISOWN {\n",
	     tab4, "my $self = shift;\n",
	     director_disown,
	     tab4, "my $ptr = tied(%$self);\n",
	     tab4, "delete $OWNER{$ptr};\n",
	     "}\n\n", "sub ACQUIRE {\n", tab4, "my $self = shift;\n", tab4, "my $ptr = tied(%$self);\n", tab4, "$OWNER{$ptr} = 1;\n", "}\n\n", NIL);
      Delete(director_disown);

      /* Only output the following methods if a class has member data */

      Delete(operators);
      operators = 0;
      if (Swig_directorclass(n)) {
	/* director classes need a way to recover subclass instance attributes */
	Node *get_attr = NewHash();
	String *mrename;
	String *symname = Getattr(n, "sym:name");
	mrename = Swig_name_disown(NSPACE_TODO, symname);
	Replaceall(mrename, "disown", "swig_get_attr");
	String *type = NewString(getClassType());
	String *name = NewString("self");
	SwigType_add_pointer(type);
	Parm *p = NewParm(type, name, n);
	Delete(name);
	Delete(type);
	type = NewString("SV");
	SwigType_add_pointer(type);
	String *action = NewString("");
	Printv(action, "{\n", "  Swig::Director *director = SWIG_DIRECTOR_CAST(arg1);\n",
	       "  result = sv_newmortal();\n" "  if (director) sv_setsv(result, director->swig_get_self());\n", "}\n", NIL);
	Setfile(get_attr, Getfile(n));
	Setline(get_attr, Getline(n));
	Setattr(get_attr, "wrap:action", action);
	Setattr(get_attr, "name", mrename);
	Setattr(get_attr, "sym:name", mrename);
	Setattr(get_attr, "type", type);
	Setattr(get_attr, "parms", p);
	Delete(action);
	Delete(type);
	Delete(p);

	member_func = 1;
	functionWrapper(get_attr);
	member_func = 0;
	Delete(get_attr);

	Printv(pm, "sub FETCH {\n", tab4, "my ($self,$field) = @_;\n", tab4, "my $member_func = \"swig_${field}_get\";\n", tab4,
	       "if (not $self->can($member_func)) {\n", tab8, "my $h = ", cmodule, "::", mrename, "($self);\n", tab8, "return $h->{$field} if $h;\n",
	       tab4, "}\n", tab4, "return $self->$member_func;\n", "}\n", "\n", "sub STORE {\n", tab4, "my ($self,$field,$newval) = @_;\n", tab4,
	       "my $member_func = \"swig_${field}_set\";\n", tab4, "if (not $self->can($member_func)) {\n", tab8, "my $h = ", cmodule, "::", mrename,
	       "($self);\n", tab8, "return $h->{$field} = $newval if $h;\n", tab4, "}\n", tab4, "return $self->$member_func($newval);\n", "}\n", NIL);

	Delete(mrename);
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int memberfunctionHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");

    member_func = 1;
    Language::memberfunctionHandler(n);
    member_func = 0;

    if ((blessed) && (!Getattr(n, "sym:nextSibling"))) {

      if (Strstr(symname, "__eq__")) {
	DohSetInt(operators, "__eq__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__ne__")) {
	DohSetInt(operators, "__ne__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__assign__")) {
	DohSetInt(operators, "__assign__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__str__")) {
	DohSetInt(operators, "__str__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__add__")) {
	DohSetInt(operators, "__add__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__sub__")) {
	DohSetInt(operators, "__sub__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__mul__")) {
	DohSetInt(operators, "__mul__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__div__")) {
	DohSetInt(operators, "__div__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__mod__")) {
	DohSetInt(operators, "__mod__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__and__")) {
	DohSetInt(operators, "__and__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__or__")) {
	DohSetInt(operators, "__or__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__not__")) {
	DohSetInt(operators, "__not__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__gt__")) {
	DohSetInt(operators, "__gt__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__ge__")) {
	DohSetInt(operators, "__ge__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__lt__")) {
	DohSetInt(operators, "__lt__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__le__")) {
	DohSetInt(operators, "__le__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__neg__")) {
	DohSetInt(operators, "__neg__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__plusplus__")) {
	DohSetInt(operators, "__plusplus__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__minmin__")) {
	DohSetInt(operators, "__minmin__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__mineq__")) {
	DohSetInt(operators, "__mineq__", 1);
	have_operators = 1;
      } else if (Strstr(symname, "__pluseq__")) {
	DohSetInt(operators, "__pluseq__", 1);
	have_operators = 1;
      }

      if (Getattr(n, "feature:shadow")) {
	String *plcode = perlcode(Getattr(n, "feature:shadow"), 0);
	String *plaction = NewStringf("%s::%s", cmodule, Swig_name_member(NSPACE_TODO, class_name, symname));
	Replaceall(plcode, "$action", plaction);
	Delete(plaction);
	Printv(pcode, plcode, NIL);
      } else {
	Printv(pcode, "*", symname, " = *", cmodule, "::", Swig_name_member(NSPACE_TODO, class_name, symname), ";\n", NIL);
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * membervariableHandler()
   *
   * Adds an instance member.
   * ----------------------------------------------------------------------------- */

  virtual int membervariableHandler(Node *n) {

    String *symname = Getattr(n, "sym:name");
    /* SwigType *t  = Getattr(n,"type"); */

    /* Emit a pair of get/set functions for the variable */

    member_func = 1;
    Language::membervariableHandler(n);
    member_func = 0;

    if (blessed) {

      Printv(pcode, "*swig_", symname, "_get = *", cmodule, "::", Swig_name_get(NSPACE_TODO, Swig_name_member(NSPACE_TODO, class_name, symname)), ";\n", NIL);
      Printv(pcode, "*swig_", symname, "_set = *", cmodule, "::", Swig_name_set(NSPACE_TODO, Swig_name_member(NSPACE_TODO, class_name, symname)), ";\n", NIL);

      /* Now we need to generate a little Perl code for this */

      /* if (is_shadow(t)) {

       *//* This is a Perl object that we have already seen.  Add an
         entry to the members list *//*
         Printv(blessedmembers,
         tab4, symname, " => '", is_shadow(t), "',\n",
         NIL);

         }
       */
    }
    have_data_members++;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constructorDeclaration()
   *
   * Emits a blessed constructor for our class.    In addition to our construct
   * we manage a Perl hash table containing all of the pointers created by
   * the constructor.   This prevents us from accidentally trying to free
   * something that wasn't necessarily allocated by malloc or new
   * ------------------------------------------------------------ */

  virtual int constructorHandler(Node *n) {

    String *symname = Getattr(n, "sym:name");

    member_func = 1;

    Swig_save("perl5:constructorHandler", n, "parms", NIL);
    if (Swig_directorclass(n)) {
      Parm *parms = Getattr(n, "parms");
      Parm *self;
      String *name = NewString("self");
      String *type = NewString("SV");
      SwigType_add_pointer(type);
      self = NewParm(type, name, n);
      Delete(type);
      Delete(name);
      Setattr(self, "lname", "O");
      if (parms)
	set_nextSibling(self, parms);
      Setattr(n, "parms", self);
      Setattr(n, "wrap:self", "1");
      Setattr(n, "hidden", "1");
      Delete(self);
    }

    String *saved_nc = none_comparison;
    none_comparison = NewStringf("strcmp(SvPV_nolen(ST(0)), \"%s::%s\") != 0", module, class_name);
    String *saved_director_prot_ctor_code = director_prot_ctor_code;
    director_prot_ctor_code = NewStringf("if ($comparison) { /* subclassed */\n" "  $director_new\n" "} else {\n"
					 "SWIG_exception_fail(SWIG_RuntimeError, \"accessing abstract class or protected constructor\");\n" "}\n");
    Language::constructorHandler(n);
    Delete(none_comparison);
    none_comparison = saved_nc;
    Delete(director_prot_ctor_code);
    director_prot_ctor_code = saved_director_prot_ctor_code;
    Swig_restore(n);

    if ((blessed) && (!Getattr(n, "sym:nextSibling"))) {
      if (Getattr(n, "feature:shadow")) {
	String *plcode = perlcode(Getattr(n, "feature:shadow"), 0);
	String *plaction = NewStringf("%s::%s", module, Swig_name_member(NSPACE_TODO, class_name, symname));
	Replaceall(plcode, "$action", plaction);
	Delete(plaction);
	Printv(pcode, plcode, NIL);
      } else {
	if ((Cmp(symname, class_name) == 0)) {
	  /* Emit a blessed constructor  */
	  Printf(pcode, "sub new {\n");
	} else {
	  /* Constructor doesn't match classname so we'll just use the normal name  */
	  Printv(pcode, "sub ", Swig_name_construct(NSPACE_TODO, symname), " {\n", NIL);
	}

	const char *pkg = getCurrentClass() && Swig_directorclass(getCurrentClass())? "$_[0]" : "shift";
	Printv(pcode,
	       tab4, "my $pkg = ", pkg, ";\n",
	       tab4, "my $self = ", cmodule, "::", Swig_name_construct(NSPACE_TODO, symname), "(@_);\n", tab4, "bless $self, $pkg if defined($self);\n", "}\n\n", NIL);

	have_constructor = 1;
      }
    }
    member_func = 0;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------ 
   * destructorHandler()
   * ------------------------------------------------------------ */

  virtual int destructorHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");
    member_func = 1;
    Language::destructorHandler(n);
    if (blessed) {
      if (Getattr(n, "feature:shadow")) {
	String *plcode = perlcode(Getattr(n, "feature:shadow"), 0);
	String *plaction = NewStringf("%s::%s", module, Swig_name_member(NSPACE_TODO, class_name, symname));
	Replaceall(plcode, "$action", plaction);
	Delete(plaction);
	Printv(pcode, plcode, NIL);
      } else {
	Printv(pcode,
	       "sub DESTROY {\n",
	       tab4, "return unless $_[0]->isa('HASH');\n",
	       tab4, "my $self = tied(%{$_[0]});\n",
	       tab4, "return unless defined $self;\n",
	       tab4, "delete $ITERATORS{$self};\n",
	       tab4, "if (exists $OWNER{$self}) {\n",
	       tab8, cmodule, "::", Swig_name_destroy(NSPACE_TODO, symname), "($self);\n", tab8, "delete $OWNER{$self};\n", tab4, "}\n}\n\n", NIL);
	have_destructor = 1;
      }
    }
    member_func = 0;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * staticmemberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int staticmemberfunctionHandler(Node *n) {
    member_func = 1;
    Language::staticmemberfunctionHandler(n);
    member_func = 0;
    if ((blessed) && (!Getattr(n, "sym:nextSibling"))) {
      String *symname = Getattr(n, "sym:name");
      Printv(pcode, "*", symname, " = *", cmodule, "::", Swig_name_member(NSPACE_TODO, class_name, symname), ";\n", NIL);
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * staticmembervariableHandler()
   * ------------------------------------------------------------ */

  virtual int staticmembervariableHandler(Node *n) {
    Language::staticmembervariableHandler(n);
    if (blessed) {
      String *symname = Getattr(n, "sym:name");
      Printv(pcode, "*", symname, " = *", cmodule, "::", Swig_name_member(NSPACE_TODO, class_name, symname), ";\n", NIL);
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberconstantHandler()
   * ------------------------------------------------------------ */

  virtual int memberconstantHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");
    int oldblessed = blessed;

    /* Create a normal constant */
    blessed = 0;
    Language::memberconstantHandler(n);
    blessed = oldblessed;

    if (blessed) {
      Printv(pcode, "*", symname, " = *", cmodule, "::", Swig_name_member(NSPACE_TODO, class_name, symname), ";\n", NIL);
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * pragma()
   *
   * Pragma directive.
   *
   * %pragma(perl5) code="String"              # Includes a string in the .pm file
   * %pragma(perl5) include="file.pl"          # Includes a file in the .pm file
   * ------------------------------------------------------------ */

  virtual int pragmaDirective(Node *n) {
    String *lang;
    String *code;
    String *value;
    if (!ImportMode) {
      lang = Getattr(n, "lang");
      code = Getattr(n, "name");
      value = Getattr(n, "value");
      if (Strcmp(lang, "perl5") == 0) {
	if (Strcmp(code, "code") == 0) {
	  /* Dump the value string into the .pm file */
	  if (value) {
	    Printf(pragma_include, "%s\n", value);
	  }
	} else if (Strcmp(code, "include") == 0) {
	  /* Include a file into the .pm file */
	  if (value) {
	    FILE *f = Swig_include_open(value);
	    if (!f) {
	      Swig_error(input_file, line_number, "Unable to locate file %s\n", value);
	    } else {
	      char buffer[4096];
	      while (fgets(buffer, 4095, f)) {
		Printf(pragma_include, "%s", buffer);
	      }
	      fclose(f);
	    }
	  }
	} else {
	  Swig_error(input_file, line_number, "Unrecognized pragma.\n");
	}
      }
    }
    return Language::pragmaDirective(n);
  }

  /* ------------------------------------------------------------
   * perlcode()     - Output perlcode code into the shadow file
   * ------------------------------------------------------------ */

  String *perlcode(String *code, const String *indent) {
    String *out = NewString("");
    String *temp;
    char *t;
    if (!indent)
      indent = "";

    temp = NewString(code);

    t = Char(temp);
    if (*t == '{') {
      Delitem(temp, 0);
      Delitem(temp, DOH_END);
    }

    /* Split the input text into lines */
    List *clist = SplitLines(temp);
    Delete(temp);
    int initial = 0;
    String *s = 0;
    Iterator si;
    /* Get the initial indentation */

    for (si = First(clist); si.item; si = Next(si)) {
      s = si.item;
      if (Len(s)) {
	char *c = Char(s);
	while (*c) {
	  if (!isspace(*c))
	    break;
	  initial++;
	  c++;
	}
	if (*c && !isspace(*c))
	  break;
	else {
	  initial = 0;
	}
      }
    }
    while (si.item) {
      s = si.item;
      if (Len(s) > initial) {
	char *c = Char(s);
	c += initial;
	Printv(out, indent, c, "\n", NIL);
      } else {
	Printv(out, "\n", NIL);
      }
      si = Next(si);
    }
    Delete(clist);
    return out;
  }

  /* ------------------------------------------------------------
   * insertDirective()
   * 
   * Hook for %insert directive.
   * ------------------------------------------------------------ */

  virtual int insertDirective(Node *n) {
    String *code = Getattr(n, "code");
    String *section = Getattr(n, "section");

    if ((!ImportMode) && (Cmp(section, "perl") == 0)) {
      Printv(additional_perl_code, code, NIL);
    } else {
      Language::insertDirective(n);
    }
    return SWIG_OK;
  }

  String *runtimeCode() {
    String *s = NewString("");
    String *shead = Swig_include_sys("perlhead.swg");
    if (!shead) {
      Printf(stderr, "*** Unable to open 'perlhead.swg'\n");
    } else {
      Append(s, shead);
      Delete(shead);
    }
    String *serrors = Swig_include_sys("perlerrors.swg");
    if (!serrors) {
      Printf(stderr, "*** Unable to open 'perlerrors.swg'\n");
    } else {
      Append(s, serrors);
      Delete(serrors);
    }
    String *srun = Swig_include_sys("perlrun.swg");
    if (!srun) {
      Printf(stderr, "*** Unable to open 'perlrun.swg'\n");
    } else {
      Append(s, srun);
      Delete(srun);
    }
    return s;
  }

  String *defaultExternalRuntimeFilename() {
    return NewString("swigperlrun.h");
  }

  virtual int classDirectorInit(Node *n) {
    String *declaration = Swig_director_declaration(n);
    Printf(f_directors_h, "\n");
    Printf(f_directors_h, "%s\n", declaration);
    Printf(f_directors_h, "public:\n");
    Delete(declaration);
    return Language::classDirectorInit(n);
  }

  virtual int classDirectorEnd(Node *n) {
    if (dirprot_mode()) {
      /*
         This implementation uses a std::map<std::string,int>.

         It should be possible to rewrite it using a more elegant way,
         like copying the Java approach for the 'override' array.

         But for now, this seems to be the least intrusive way.
       */
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "/* Internal director utilities */\n");
      Printf(f_directors_h, "public:\n");
      Printf(f_directors_h, "    bool swig_get_inner(const char *swig_protected_method_name) const {\n");
      Printf(f_directors_h, "      std::map<std::string, bool>::const_iterator iv = swig_inner.find(swig_protected_method_name);\n");
      Printf(f_directors_h, "      return (iv != swig_inner.end() ? iv->second : false);\n");
      Printf(f_directors_h, "    }\n");

      Printf(f_directors_h, "    void swig_set_inner(const char *swig_protected_method_name, bool swig_val) const {\n");
      Printf(f_directors_h, "      swig_inner[swig_protected_method_name] = swig_val;\n");
      Printf(f_directors_h, "    }\n");
      Printf(f_directors_h, "private:\n");
      Printf(f_directors_h, "    mutable std::map<std::string, bool> swig_inner;\n");
    }
    Printf(f_directors_h, "};\n");
    return Language::classDirectorEnd(n);
  }

  virtual int classDirectorConstructor(Node *n) {
    Node *parent = Getattr(n, "parentNode");
    String *sub = NewString("");
    String *decl = Getattr(n, "decl");
    String *supername = Swig_class_name(parent);
    String *classname = NewString("");
    Printf(classname, "SwigDirector_%s", supername);

    /* insert self parameter */
    Parm *p;
    ParmList *superparms = Getattr(n, "parms");
    ParmList *parms = CopyParmList(superparms);
    String *type = NewString("SV");
    SwigType_add_pointer(type);
    p = NewParm(type, NewString("self"), n);
    set_nextSibling(p, parms);
    parms = p;

    if (!Getattr(n, "defaultargs")) {
      /* constructor */
      {
	Wrapper *w = NewWrapper();
	String *call;
	String *basetype = Getattr(parent, "classtype");
	String *target = Swig_method_decl(0, decl, classname, parms, 0, 0);
	call = Swig_csuperclass_call(0, basetype, superparms);
	Printf(w->def, "%s::%s: %s, Swig::Director(self) { \n", classname, target, call);
	Printf(w->def, "   SWIG_DIRECTOR_RGTR((%s *)this, this); \n", basetype);
	Append(w->def, "}\n");
	Delete(target);
	Wrapper_print(w, f_directors);
	Delete(call);
	DelWrapper(w);
      }

      /* constructor header */
      {
	String *target = Swig_method_decl(0, decl, classname, parms, 0, 1);
	Printf(f_directors_h, "    %s;\n", target);
	Delete(target);
      }
    }

    Delete(sub);
    Delete(classname);
    Delete(supername);
    Delete(parms);
    return Language::classDirectorConstructor(n);
  }

  virtual int classDirectorMethod(Node *n, Node *parent, String *super) {
    int is_void = 0;
    int is_pointer = 0;
    String *decl = Getattr(n, "decl");
    String *name = Getattr(n, "name");
    String *classname = Getattr(parent, "sym:name");
    String *c_classname = Getattr(parent, "name");
    String *symname = Getattr(n, "sym:name");
    String *declaration = NewString("");
    ParmList *l = Getattr(n, "parms");
    Wrapper *w = NewWrapper();
    String *tm;
    String *wrap_args = NewString("");
    String *returntype = Getattr(n, "type");
    String *value = Getattr(n, "value");
    String *storage = Getattr(n, "storage");
    bool pure_virtual = false;
    int status = SWIG_OK;
    int idx;
    bool ignored_method = GetFlag(n, "feature:ignore") ? true : false;

    if (Cmp(storage, "virtual") == 0) {
      if (Cmp(value, "0") == 0) {
	pure_virtual = true;
      }
    }

    /* determine if the method returns a pointer */
    is_pointer = SwigType_ispointer_return(decl);
    is_void = (!Cmp(returntype, "void") && !is_pointer);

    /* virtual method definition */
    String *target;
    String *pclassname = NewStringf("SwigDirector_%s", classname);
    String *qualified_name = NewStringf("%s::%s", pclassname, name);
    SwigType *rtype = Getattr(n, "conversion_operator") ? 0 : Getattr(n, "classDirectorMethods:type");
    target = Swig_method_decl(rtype, decl, qualified_name, l, 0, 0);
    Printf(w->def, "%s", target);
    Delete(qualified_name);
    Delete(target);
    /* header declaration */
    target = Swig_method_decl(rtype, decl, name, l, 0, 1);
    Printf(declaration, "    virtual %s", target);
    Delete(target);

    // Get any exception classes in the throws typemap
    ParmList *throw_parm_list = 0;

    if ((throw_parm_list = Getattr(n, "throws")) || Getattr(n, "throw")) {
      Parm *p;
      int gencomma = 0;

      Append(w->def, " throw(");
      Append(declaration, " throw(");

      if (throw_parm_list)
	Swig_typemap_attach_parms("throws", throw_parm_list, 0);
      for (p = throw_parm_list; p; p = nextSibling(p)) {
	if (Getattr(p, "tmap:throws")) {
	  if (gencomma++) {
	    Append(w->def, ", ");
	    Append(declaration, ", ");
	  }
	  String *str = SwigType_str(Getattr(p, "type"), 0);
	  Append(w->def, str);
	  Append(declaration, str);
	  Delete(str);
	}
      }

      Append(w->def, ")");
      Append(declaration, ")");
    }

    Append(w->def, " {");
    Append(declaration, ";\n");

    /* declare method return value 
     * if the return value is a reference or const reference, a specialized typemap must
     * handle it, including declaration of c_result ($result).
     */
    if (!is_void) {
      if (!(ignored_method && !pure_virtual)) {
	String *cres = SwigType_lstr(returntype, "c_result");
	Printf(w->code, "%s;\n", cres);
	Delete(cres);
      }
      if (!ignored_method) {
	String *pres = NewStringf("SV *%s", Swig_cresult_name());
	Wrapper_add_local(w, Swig_cresult_name(), pres);
	Delete(pres);
      }
    }

    if (ignored_method) {
      if (!pure_virtual) {
	if (!is_void)
	  Printf(w->code, "return ");
	String *super_call = Swig_method_call(super, l);
	Printf(w->code, "%s;\n", super_call);
	Delete(super_call);
      } else {
	Printf(w->code, "Swig::DirectorPureVirtualException::raise(\"Attempted to invoke pure virtual method %s::%s\");\n", SwigType_namestr(c_classname),
	       SwigType_namestr(name));
      }
    } else {
      /* attach typemaps to arguments (C/C++ -> Perl) */
      String *parse_args = NewString("");
      String *pstack = NewString("");

      Swig_director_parms_fixup(l);

      /* remove the wrapper 'w' since it was producing spurious temps */
      Swig_typemap_attach_parms("in", l, 0);
      Swig_typemap_attach_parms("directorin", l, 0);
      Swig_typemap_attach_parms("directorargout", l, w);

      Wrapper_add_local(w, "SP", "dSP");

      {
	String *ptype = Copy(getClassType());
	SwigType_add_pointer(ptype);
	String *mangle = SwigType_manglestr(ptype);

	Wrapper_add_local(w, "swigself", "SV *swigself");
	Printf(w->code, "swigself = SWIG_NewPointerObj(SWIG_as_voidptr(this), SWIGTYPE%s, SWIG_SHADOW);\n", mangle);
	Printf(w->code, "sv_bless(swigself, gv_stashpv(swig_get_class(), 0));\n");
	Delete(mangle);
	Delete(ptype);
	Append(pstack, "XPUSHs(swigself);\n");
      }

      Parm *p;
      char source[256];

      int outputs = 0;
      if (!is_void)
	outputs++;

      /* build argument list and type conversion string */
      idx = 0;
      p = l;
      while (p) {
	if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	  p = Getattr(p, "tmap:in:next");
	  continue;
	}

	/* old style?  caused segfaults without the p!=0 check
	   in the for() condition, and seems dangerous in the
	   while loop as well.
	   while (Getattr(p, "tmap:ignore")) {
	   p = Getattr(p, "tmap:ignore:next");
	   }
	 */

	if (Getattr(p, "tmap:directorargout") != 0)
	  outputs++;

	String *pname = Getattr(p, "name");
	String *ptype = Getattr(p, "type");

	if ((tm = Getattr(p, "tmap:directorin")) != 0) {
	  sprintf(source, "obj%d", idx++);
	  String *input = NewString(source);
	  Setattr(p, "emit:directorinput", input);
	  Replaceall(tm, "$input", input);
	  Delete(input);
	  Replaceall(tm, "$owner", "0");
	  Replaceall(tm, "$shadow", "0");
	  /* Wrapper_add_localv(w, source, "SV *", source, "= 0", NIL); */
	  Printv(wrap_args, "SV *", source, ";\n", NIL);

	  Printv(wrap_args, tm, "\n", NIL);
	  Putc('O', parse_args);
	  Printv(pstack, "XPUSHs(", source, ");\n", NIL);
	  p = Getattr(p, "tmap:directorin:next");
	  continue;
	} else if (Cmp(ptype, "void")) {
	  /* special handling for pointers to other C++ director classes.
	   * ideally this would be left to a typemap, but there is currently no
	   * way to selectively apply the dynamic_cast<> to classes that have
	   * directors.  in other words, the type "SwigDirector_$1_lname" only exists
	   * for classes with directors.  we avoid the problem here by checking
	   * module.wrap::directormap, but it's not clear how to get a typemap to
	   * do something similar.  perhaps a new default typemap (in addition
	   * to SWIGTYPE) called DIRECTORTYPE?
	   */
	  if (SwigType_ispointer(ptype) || SwigType_isreference(ptype)) {
	    Node *module = Getattr(parent, "module");
	    Node *target = Swig_directormap(module, ptype);
	    sprintf(source, "obj%d", idx++);
	    String *nonconst = 0;
	    /* strip pointer/reference --- should move to Swig/stype.c */
	    String *nptype = NewString(Char(ptype) + 2);
	    /* name as pointer */
	    String *ppname = Copy(pname);
	    if (SwigType_isreference(ptype)) {
	      Insert(ppname, 0, "&");
	    }
	    /* if necessary, cast away const since Perl doesn't support it! */
	    if (SwigType_isconst(nptype)) {
	      nonconst = NewStringf("nc_tmp_%s", pname);
	      String *nonconst_i = NewStringf("= const_cast< %s >(%s)", SwigType_lstr(ptype, 0), ppname);
	      Wrapper_add_localv(w, nonconst, SwigType_lstr(ptype, 0), nonconst, nonconst_i, NIL);
	      Delete(nonconst_i);
	      Swig_warning(WARN_LANG_DISCARD_CONST, input_file, line_number,
			   "Target language argument '%s' discards const in director method %s::%s.\n",
			   SwigType_str(ptype, pname), SwigType_namestr(c_classname), SwigType_namestr(name));
	    } else {
	      nonconst = Copy(ppname);
	    }
	    Delete(nptype);
	    Delete(ppname);
	    String *mangle = SwigType_manglestr(ptype);
	    if (target) {
	      String *director = NewStringf("director_%s", mangle);
	      Wrapper_add_localv(w, director, "Swig::Director *", director, "= 0", NIL);
	      Wrapper_add_localv(w, source, "SV *", source, "= 0", NIL);
	      Printf(wrap_args, "%s = SWIG_DIRECTOR_CAST(%s);\n", director, nonconst);
	      Printf(wrap_args, "if (!%s) {\n", director);
	      Printf(wrap_args, "%s = SWIG_NewPointerObj(%s, SWIGTYPE%s, 0);\n", source, nonconst, mangle);
	      Append(wrap_args, "} else {\n");
	      Printf(wrap_args, "%s = %s->swig_get_self();\n", source, director);
	      Printf(wrap_args, "SvREFCNT_inc((SV *)%s);\n", source);
	      Append(wrap_args, "}\n");
	      Delete(director);
	    } else {
	      Wrapper_add_localv(w, source, "SV *", source, "= 0", NIL);
	      Printf(wrap_args, "%s = SWIG_NewPointerObj(%s, SWIGTYPE%s, 0);\n", source, nonconst, mangle);
	      Printf(pstack, "XPUSHs(sv_2mortal(%s));\n", source);
	    }
	    Putc('O', parse_args);
	    Delete(mangle);
	    Delete(nonconst);
	  } else {
	    Swig_warning(WARN_TYPEMAP_DIRECTORIN_UNDEF, input_file, line_number,
			 "Unable to use type %s as a function argument in director method %s::%s (skipping method).\n", SwigType_str(ptype, 0),
			 SwigType_namestr(c_classname), SwigType_namestr(name));
	    status = SWIG_NOWRAP;
	    break;
	  }
	}
	p = nextSibling(p);
      }

      /* add the method name as a PyString */
      String *pyname = Getattr(n, "sym:name");

      /* wrap complex arguments to PyObjects */
      Printv(w->code, wrap_args, NIL);

      /* pass the method call on to the Python object */
      if (dirprot_mode() && !is_public(n)) {
	Printf(w->code, "swig_set_inner(\"%s\", true);\n", name);
      }

      Append(w->code, "ENTER;\n");
      Append(w->code, "SAVETMPS;\n");
      Append(w->code, "PUSHMARK(SP);\n");
      Append(w->code, pstack);
      Delete(pstack);
      Append(w->code, "PUTBACK;\n");
      Printf(w->code, "call_method(\"%s\", G_EVAL | G_SCALAR);\n", pyname);

      if (dirprot_mode() && !is_public(n))
	Printf(w->code, "swig_set_inner(\"%s\", false);\n", name);

      /* exception handling */
      tm = Swig_typemap_lookup("director:except", n, Swig_cresult_name(), 0);
      if (!tm) {
	tm = Getattr(n, "feature:director:except");
	if (tm)
	  tm = Copy(tm);
      }
      Append(w->code, "if (SvTRUE(ERRSV)) {\n");
      Append(w->code, "  PUTBACK;\n  FREETMPS;\n  LEAVE;\n");
      if ((tm) && Len(tm) && (Strcmp(tm, "1") != 0)) {
	Replaceall(tm, "$error", "ERRSV");
	Printv(w->code, Str(tm), "\n", NIL);
      } else {
	Printf(w->code, "  Swig::DirectorMethodException::raise(ERRSV);\n", classname, pyname);
      }
      Append(w->code, "}\n");
      Delete(tm);

      /*
       * Python method may return a simple object, or a tuple.
       * for in/out aruments, we have to extract the appropriate PyObjects from the tuple,
       * then marshal everything back to C/C++ (return value and output arguments).
       *
       */

      /* marshal return value and other outputs (if any) from PyObject to C/C++ type */

      String *cleanup = NewString("");
      String *outarg = NewString("");

      if (outputs > 1) {
	Wrapper_add_local(w, "output", "SV *output");
	Printf(w->code, "if (count != %d) {\n", outputs);
	Printf(w->code, "  Swig::DirectorTypeMismatchException::raise(\"Perl method %s.%sfailed to return a list.\");\n", classname, pyname);
	Append(w->code, "}\n");
      }

      idx = 0;

      /* marshal return value */
      if (!is_void) {
	Append(w->code, "SPAGAIN;\n");
	Printf(w->code, "%s = POPs;\n", Swig_cresult_name());
	tm = Swig_typemap_lookup("directorout", n, Swig_cresult_name(), w);
	if (tm != 0) {
	  if (outputs > 1) {
	    Printf(w->code, "output = POPs;\n");
	    Replaceall(tm, "$input", "output");
	  } else {
	    Replaceall(tm, "$input", Swig_cresult_name());
	  }
	  char temp[24];
	  sprintf(temp, "%d", idx);
	  Replaceall(tm, "$argnum", temp);

	  /* TODO check this */
	  if (Getattr(n, "wrap:disown")) {
	    Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
	  } else {
	    Replaceall(tm, "$disown", "0");
	  }
	  Replaceall(tm, "$result", "c_result");
	  Printv(w->code, tm, "\n", NIL);
	  Delete(tm);
	} else {
	  Swig_warning(WARN_TYPEMAP_DIRECTOROUT_UNDEF, input_file, line_number,
		       "Unable to use return type %s in director method %s::%s (skipping method).\n", SwigType_str(returntype, 0),
		       SwigType_namestr(c_classname), SwigType_namestr(name));
	  status = SWIG_ERROR;
	}
      }

      /* marshal outputs */
      for (p = l; p;) {
	if ((tm = Getattr(p, "tmap:directorargout")) != 0) {
	  if (outputs > 1) {
	    Printf(w->code, "output = POPs;\n");
	    Replaceall(tm, "$result", "output");
	  } else {
	    Replaceall(tm, "$result", Swig_cresult_name());
	  }
	  Replaceall(tm, "$input", Getattr(p, "emit:directorinput"));
	  Printv(w->code, tm, "\n", NIL);
	  p = Getattr(p, "tmap:directorargout:next");
	} else {
	  p = nextSibling(p);
	}
      }

      Delete(parse_args);
      Delete(cleanup);
      Delete(outarg);
    }

    if (!ignored_method) {
      Append(w->code, "PUTBACK;\n");
      Append(w->code, "FREETMPS;\n");
      Append(w->code, "LEAVE;\n");
    }

    if (!is_void) {
      if (!(ignored_method && !pure_virtual)) {
	String *rettype = SwigType_str(returntype, 0);
	if (!SwigType_isreference(returntype)) {
	  Printf(w->code, "return (%s) c_result;\n", rettype);
	} else {
	  Printf(w->code, "return (%s) *c_result;\n", rettype);
	}
	Delete(rettype);
      }
    }

    Append(w->code, "}\n");

    // We expose protected methods via an extra public inline method which makes a straight call to the wrapped class' method
    String *inline_extra_method = NewString("");
    if (dirprot_mode() && !is_public(n) && !pure_virtual) {
      Printv(inline_extra_method, declaration, NIL);
      String *extra_method_name = NewStringf("%sSwigPublic", name);
      Replaceall(inline_extra_method, name, extra_method_name);
      Replaceall(inline_extra_method, ";\n", " {\n      ");
      if (!is_void)
	Printf(inline_extra_method, "return ");
      String *methodcall = Swig_method_call(super, l);
      Printv(inline_extra_method, methodcall, ";\n    }\n", NIL);
      Delete(methodcall);
      Delete(extra_method_name);
    }

    /* emit the director method */
    if (status == SWIG_OK) {
      if (!Getattr(n, "defaultargs")) {
	Replaceall(w->code, "$symname", symname);
	Wrapper_print(w, f_directors);
	Printv(f_directors_h, declaration, NIL);
	Printv(f_directors_h, inline_extra_method, NIL);
      }
    }

    /* clean up */
    Delete(wrap_args);
    Delete(pclassname);
    DelWrapper(w);
    return status;
  }
  int classDirectorDisown(Node *n) {
    int rv;
    member_func = 1;
    rv = Language::classDirectorDisown(n);
    member_func = 0;
    if (rv == SWIG_OK && Swig_directorclass(n)) {
      String *symname = Getattr(n, "sym:name");
      String *disown = Swig_name_disown(NSPACE_TODO, symname);
      Setattr(n, "perl5:directordisown", NewStringf("%s::%s", cmodule, disown));
    }
    return rv;
  }
  int classDirectorDestructor(Node *n) {
    /* TODO: it would be nice if this didn't have to copy the body of Language::classDirectorDestructor() */
    String *DirectorClassName = directorClassName(getCurrentClass());
    String *body = NewString("\n");

    String *ptype = Copy(getClassType());
    SwigType_add_pointer(ptype);
    String *mangle = SwigType_manglestr(ptype);

    Printv(body, tab4, "dSP;\n", tab4, "SV *self = SWIG_NewPointerObj(SWIG_as_voidptr(this), SWIGTYPE", mangle, ", SWIG_SHADOW);\n", tab4, "\n", tab4,
	   "sv_bless(self, gv_stashpv(swig_get_class(), 0));\n", tab4, "ENTER;\n", tab4, "SAVETMPS;\n", tab4, "PUSHMARK(SP);\n", tab4,
	   "XPUSHs(self);\n", tab4, "XPUSHs(&PL_sv_yes);\n", tab4, "PUTBACK;\n", tab4, "call_method(\"DESTROY\", G_EVAL | G_VOID);\n", tab4,
	   "FREETMPS;\n", tab4, "LEAVE;\n", NIL);

    Delete(mangle);
    Delete(ptype);

    if (Getattr(n, "throw")) {
      Printf(f_directors_h, "    virtual ~%s() throw ();\n", DirectorClassName);
      Printf(f_directors, "%s::~%s() throw () {%s}\n\n", DirectorClassName, DirectorClassName, body);
    } else {
      Printf(f_directors_h, "    virtual ~%s();\n", DirectorClassName);
      Printf(f_directors, "%s::~%s() {%s}\n\n", DirectorClassName, DirectorClassName, body);
    }
    return SWIG_OK;
  }
};

/* -----------------------------------------------------------------------------
 * swig_perl5()    - Instantiate module
 * ----------------------------------------------------------------------------- */

static Language *new_swig_perl5() {
  return new PERL5();
}
extern "C" Language *swig_perl5(void) {
  return new_swig_perl5();
}
