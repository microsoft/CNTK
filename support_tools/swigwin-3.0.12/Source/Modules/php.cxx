/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * php.cxx
 *
 * PHP language module for SWIG.
 * -----------------------------------------------------------------------------
 */

/* FIXME: PHP5 OO wrapping TODO list:
 *
 * Medium term:
 *
 * Handle default parameters on overloaded methods in PHP where possible.
 *   (Mostly done - just need to handle cases of overloaded methods with
 *   default parameters...)
 *   This is an optimisation - we could handle this case using a PHP
 *   default value, but currently we treat it as we would for a default
 *   value which is a compound C++ expression (i.e. as if we had a
 *   method with two overloaded forms instead of a single method with
 *   a default parameter value).
 *
 * Long term:
 *
 * Sort out locale-dependent behaviour of strtod() - it's harmless unless
 *   SWIG ever sets the locale and DOH/base.c calls atof, so we're probably
 *   OK currently at least.
 */

/*
 * TODO: Replace remaining stderr messages with Swig_error or Swig_warning
 * (may need to add more WARN_PHP_xxx codes...)
 */

#include "swigmod.h"

#include <ctype.h>
#include <errno.h>

static const char *usage = "\
PHP Options (available with -php7)\n\
     -noproxy         - Don't generate proxy classes.\n\
     -prefix <prefix> - Prepend <prefix> to all class names in PHP wrappers\n\
\n";

/* The original class wrappers for PHP stored the pointer to the C++ class in
 * the object property _cPtr.  If we use the same name for the member variable
 * which we put the pointer to the C++ class in, then the flat function
 * wrappers will automatically pull it out without any changes being required.
 * FIXME: Isn't using a leading underscore a bit suspect here?
 */
#define SWIG_PTR "_cPtr"

/* This is the name of the hash where the variables existing only in PHP
 * classes are stored.
 */
#define SWIG_DATA "_pData"

static int constructors = 0;
static String *NOTCLASS = NewString("Not a class");
static Node *classnode = 0;
static String *module = 0;
static String *cap_module = 0;
static String *prefix = 0;

static String *shadow_classname = 0;

static File *f_begin = 0;
static File *f_runtime = 0;
static File *f_runtime_h = 0;
static File *f_h = 0;
static File *f_phpcode = 0;
static File *f_directors = 0;
static File *f_directors_h = 0;
static String *phpfilename = 0;

static String *s_header;
static String *s_wrappers;
static String *s_init;
static String *r_init;		// RINIT user code
static String *s_shutdown;	// MSHUTDOWN user code
static String *r_shutdown;	// RSHUTDOWN user code
static String *s_vinit;		// varinit initialization code.
static String *s_vdecl;
static String *s_cinit;		// consttab initialization code.
static String *s_oinit;
static String *s_arginfo;
static String *s_entry;
static String *cs_entry;
static String *all_cs_entry;
static String *pragma_incl;
static String *pragma_code;
static String *pragma_phpinfo;
static String *s_oowrappers;
static String *s_fakeoowrappers;
static String *s_phpclasses;

/* To reduce code size (generated and compiled) we only want to emit each
 * different arginfo once, so we need to track which have been used.
 */
static Hash *arginfo_used;

/* Variables for using PHP classes */
static Node *current_class = 0;

static Hash *shadow_get_vars;
static Hash *shadow_set_vars;
static Hash *zend_types = 0;

static int shadow = 1;

static bool class_has_ctor = false;
static String *wrapping_member_constant = NULL;

// These static variables are used to pass some state from Handlers into functionWrapper
static enum {
  standard = 0,
  memberfn,
  staticmemberfn,
  membervar,
  staticmembervar,
  constructor,
  directorconstructor
} wrapperType = standard;

extern "C" {
  static void (*r_prevtracefunc) (const SwigType *t, String *mangled, String *clientdata) = 0;
}

static void SwigPHP_emit_resource_registrations() {
  Iterator ki;
  bool emitted_default_dtor = false;

  if (!zend_types)
    return;

  ki = First(zend_types);
  if (ki.key)
    Printf(s_oinit, "\n/* Register resource destructors for pointer types */\n");
  while (ki.key) {
    DOH *key = ki.key;
    Node *class_node = ki.item;
    String *human_name = key;
    String *rsrc_dtor_name = NULL;

    // write out body
    if (class_node != NOTCLASS) {
      String *destructor = Getattr(class_node, "destructor");
      human_name = Getattr(class_node, "sym:name");
      if (!human_name)
        human_name = Getattr(class_node, "name");
      // Do we have a known destructor for this type?
      if (destructor) {
	rsrc_dtor_name = NewStringf("_wrap_destroy%s", key);
	// Write out custom destructor function
	Printf(s_wrappers, "static ZEND_RSRC_DTOR_FUNC(%s) {\n", rsrc_dtor_name);
        Printf(s_wrappers, "  %s(res, SWIGTYPE%s->name);\n", destructor, key);
	Printf(s_wrappers, "}\n");
      }
    }

    if (!rsrc_dtor_name) {
      rsrc_dtor_name = NewString("_swig_default_rsrc_destroy");
      if (!emitted_default_dtor) {
	// Write out custom destructor function
	Printf(s_wrappers, "static ZEND_RSRC_DTOR_FUNC(%s) {\n", rsrc_dtor_name);
	Printf(s_wrappers, "  efree(res->ptr);\n");
	Printf(s_wrappers, "}\n");
	emitted_default_dtor = true;
      }
    }

    // declare le_swig_<mangled> to store php registration
    Printf(s_vdecl, "static int le_swig_%s=0; /* handle for %s */\n", key, human_name);

    // register with php
    Printf(s_oinit, "le_swig_%s=zend_register_list_destructors_ex"
		    "(%s, NULL, SWIGTYPE%s->name, module_number);\n", key, rsrc_dtor_name, key);

    // store php type in class struct
    Printf(s_oinit, "SWIG_TypeClientData(SWIGTYPE%s,&le_swig_%s);\n", key, key);

    Delete(rsrc_dtor_name);

    ki = Next(ki);
  }
}

class PHP : public Language {
public:
  PHP() {
    director_language = 1;
  }

  /* ------------------------------------------------------------
   * main()
   * ------------------------------------------------------------ */

  virtual void main(int argc, char *argv[]) {
    SWIG_library_directory("php");

    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-prefix") == 0) {
	if (argv[i + 1]) {
	  prefix = NewString(argv[i + 1]);
	  Swig_mark_arg(i);
	  Swig_mark_arg(i + 1);
	  i++;
	} else {
	  Swig_arg_error();
	}
      } else if ((strcmp(argv[i], "-noshadow") == 0) || (strcmp(argv[i], "-noproxy") == 0)) {
	shadow = 0;
	Swig_mark_arg(i);
      } else if (strcmp(argv[i], "-help") == 0) {
	fputs(usage, stdout);
      }
    }

    Preprocessor_define("SWIGPHP 1", 0);
    Preprocessor_define("SWIGPHP7 1", 0);
    SWIG_typemap_lang("php");
    SWIG_config_file("php.swg");
    allow_overloading();
  }

  /* ------------------------------------------------------------
   * top()
   * ------------------------------------------------------------ */

  virtual int top(Node *n) {

    String *filen;

    /* Check if directors are enabled for this module. */
    Node *mod = Getattr(n, "module");
    if (mod) {
      Node *options = Getattr(mod, "options");
      if (options && Getattr(options, "directors")) {
	allow_directors();
      }
    }

    /* Set comparison with null for ConstructorToFunction */
    setSubclassInstanceCheck(NewString("Z_TYPE_P($arg) != IS_NULL"));

    /* Initialize all of the output files */
    String *outfile = Getattr(n, "outfile");
    String *outfile_h = Getattr(n, "outfile_h");

    /* main output file */
    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      SWIG_exit(EXIT_FAILURE);
    }
    f_runtime = NewStringEmpty();

    /* sections of the output file */
    s_init = NewStringEmpty();
    r_init = NewStringEmpty();
    s_shutdown = NewStringEmpty();
    r_shutdown = NewStringEmpty();
    s_header = NewString("/* header section */\n");
    s_wrappers = NewString("/* wrapper section */\n");
    /* subsections of the init section */
    s_vinit = NewStringEmpty();
    s_vdecl = NewString("/* vdecl subsection */\n");
    s_cinit = NewString("/* cinit subsection */\n");
    s_oinit = NewString("/* oinit subsection */\n");
    pragma_phpinfo = NewStringEmpty();
    s_phpclasses = NewString("/* PHP Proxy Classes */\n");
    f_directors_h = NewStringEmpty();
    f_directors = NewStringEmpty();

    if (directorsEnabled()) {
      f_runtime_h = NewFile(outfile_h, "w", SWIG_output_files());
      if (!f_runtime_h) {
	FileErrorDisplay(outfile_h);
	SWIG_exit(EXIT_FAILURE);
      }
    }

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", s_init);
    Swig_register_filebyname("rinit", r_init);
    Swig_register_filebyname("shutdown", s_shutdown);
    Swig_register_filebyname("rshutdown", r_shutdown);
    Swig_register_filebyname("header", s_header);
    Swig_register_filebyname("wrapper", s_wrappers);
    Swig_register_filebyname("director", f_directors);
    Swig_register_filebyname("director_h", f_directors_h);

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGPHP\n#define SWIGPHP\n#endif\n\n");

    if (directorsEnabled()) {
      Printf(f_runtime, "#define SWIG_DIRECTORS\n");
    }

    /* Set the module name */
    module = Copy(Getattr(n, "name"));
    cap_module = NewStringf("%(upper)s", module);
    if (!prefix)
      prefix = NewStringEmpty();

    Printf(f_runtime, "#define SWIG_PREFIX \"%s\"\n", prefix);
    Printf(f_runtime, "#define SWIG_PREFIX_LEN %lu\n", (unsigned long)Len(prefix));

    if (directorsEnabled()) {
      Swig_banner(f_directors_h);
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "#ifndef SWIG_%s_WRAP_H_\n", cap_module);
      Printf(f_directors_h, "#define SWIG_%s_WRAP_H_\n\n", cap_module);

      String *filename = Swig_file_filename(outfile_h);
      Printf(f_directors, "\n#include \"%s\"\n\n", filename);
      Delete(filename);
    }

    /* PHP module file */
    filen = NewStringEmpty();
    Printv(filen, SWIG_output_directory(), module, ".php", NIL);
    phpfilename = NewString(filen);

    f_phpcode = NewFile(filen, "w", SWIG_output_files());
    if (!f_phpcode) {
      FileErrorDisplay(filen);
      SWIG_exit(EXIT_FAILURE);
    }

    Printf(f_phpcode, "<?php\n\n");

    Swig_banner(f_phpcode);

    Printf(f_phpcode, "\n");
    Printf(f_phpcode, "// Try to load our extension if it's not already loaded.\n");
    Printf(f_phpcode, "if (!extension_loaded('%s')) {\n", module);
    Printf(f_phpcode, "  if (strtolower(substr(PHP_OS, 0, 3)) === 'win') {\n");
    Printf(f_phpcode, "    if (!dl('php_%s.dll')) return;\n", module);
    Printf(f_phpcode, "  } else {\n");
    Printf(f_phpcode, "    // PHP_SHLIB_SUFFIX gives 'dylib' on MacOS X but modules are 'so'.\n");
    Printf(f_phpcode, "    if (PHP_SHLIB_SUFFIX === 'dylib') {\n");
    Printf(f_phpcode, "      if (!dl('%s.so')) return;\n", module);
    Printf(f_phpcode, "    } else {\n");
    Printf(f_phpcode, "      if (!dl('%s.'.PHP_SHLIB_SUFFIX)) return;\n", module);
    Printf(f_phpcode, "    }\n");
    Printf(f_phpcode, "  }\n");
    Printf(f_phpcode, "}\n\n");

    /* sub-sections of the php file */
    pragma_code = NewStringEmpty();
    pragma_incl = NewStringEmpty();

    /* Initialize the rest of the module */

    Printf(s_oinit, "ZEND_INIT_MODULE_GLOBALS(%s, %s_init_globals, NULL);\n", module, module);

    /* start the header section */
    Printf(s_header, "ZEND_BEGIN_MODULE_GLOBALS(%s)\n", module);
    Printf(s_header, "const char *error_msg;\n");
    Printf(s_header, "int error_code;\n");
    Printf(s_header, "ZEND_END_MODULE_GLOBALS(%s)\n", module);
    Printf(s_header, "ZEND_DECLARE_MODULE_GLOBALS(%s)\n", module);
    Printf(s_header, "#define SWIG_ErrorMsg() (%s_globals.error_msg)\n", module);
    Printf(s_header, "#define SWIG_ErrorCode() (%s_globals.error_code)\n", module);

    /* The following can't go in Lib/php/phprun.swg as it uses SWIG_ErrorMsg(), etc
     * which has to be dynamically generated as it depends on the module name.
     */
    Append(s_header, "#ifdef __GNUC__\n");
    Append(s_header, "static void SWIG_FAIL(void) __attribute__ ((__noreturn__));\n");
    Append(s_header, "#endif\n\n");
    Append(s_header, "static void SWIG_FAIL(void) {\n");
    Append(s_header, "    zend_error(SWIG_ErrorCode(), \"%s\", SWIG_ErrorMsg());\n");
    // zend_error() should never return with the parameters we pass, but if it
    // does, we really don't want to let SWIG_FAIL() return.  This also avoids
    // a warning about returning from a function marked as "__noreturn__".
    Append(s_header, "    abort();\n");
    Append(s_header, "}\n\n");

    Printf(s_header, "static void %s_init_globals(zend_%s_globals *globals ) {\n", module, module);
    Printf(s_header, "  globals->error_msg = default_error_msg;\n");
    Printf(s_header, "  globals->error_code = default_error_code;\n");
    Printf(s_header, "}\n");

    Printf(s_header, "static void SWIG_ResetError(void) {\n");
    Printf(s_header, "  SWIG_ErrorMsg() = default_error_msg;\n");
    Printf(s_header, "  SWIG_ErrorCode() = default_error_code;\n");
    Printf(s_header, "}\n");

    Append(s_header, "\n");
    Printf(s_header, "ZEND_NAMED_FUNCTION(_wrap_swig_%s_alter_newobject) {\n", module);
    Append(s_header, "  zval args[2];\n");
    Append(s_header, "  swig_object_wrapper *value;\n");
    Append(s_header, "\n");
    Append(s_header, "  SWIG_ResetError();\n");
    Append(s_header, "  if(ZEND_NUM_ARGS() != 2 || zend_get_parameters_array_ex(2, args) != SUCCESS) {\n");
    Append(s_header, "    WRONG_PARAM_COUNT;\n");
    Append(s_header, "  }\n");
    Append(s_header, "\n");
    Append(s_header, "  value = (swig_object_wrapper *) Z_RES_VAL(args[0]);\n");
    Append(s_header, "  value->newobject = zval_is_true(&args[1]);\n");
    Append(s_header, "\n");
    Append(s_header, "  return;\n");
    Append(s_header, "}\n");
    Printf(s_header, "ZEND_NAMED_FUNCTION(_wrap_swig_%s_get_newobject) {\n", module);
    Append(s_header, "  zval args[1];\n");
    Append(s_header, "  swig_object_wrapper *value;\n");
    Append(s_header, "\n");
    Append(s_header, "  SWIG_ResetError();\n");
    Append(s_header, "  if(ZEND_NUM_ARGS() != 1 || zend_get_parameters_array_ex(1, args) != SUCCESS) {\n");
    Append(s_header, "    WRONG_PARAM_COUNT;\n");
    Append(s_header, "  }\n");
    Append(s_header, "\n");
    Append(s_header, "  value = (swig_object_wrapper *) Z_RES_VAL(args[0]);\n");
    Append(s_header, "  RETVAL_LONG(value->newobject);\n");
    Append(s_header, "\n");
    Append(s_header, "  return;\n");
    Append(s_header, "}\n");

    Printf(s_header, "#define SWIG_name  \"%s\"\n", module);
    Printf(s_header, "#ifdef __cplusplus\n");
    Printf(s_header, "extern \"C\" {\n");
    Printf(s_header, "#endif\n");
    Printf(s_header, "#include \"php.h\"\n");
    Printf(s_header, "#include \"php_ini.h\"\n");
    Printf(s_header, "#include \"ext/standard/info.h\"\n");
    Printf(s_header, "#include \"php_%s.h\"\n", module);
    Printf(s_header, "#ifdef __cplusplus\n");
    Printf(s_header, "}\n");
    Printf(s_header, "#endif\n\n");

    if (directorsEnabled()) {
      // Insert director runtime
      Swig_insert_file("director_common.swg", s_header);
      Swig_insert_file("director.swg", s_header);
    }

    /* Create the .h file too */
    filen = NewStringEmpty();
    Printv(filen, SWIG_output_directory(), "php_", module, ".h", NIL);
    f_h = NewFile(filen, "w", SWIG_output_files());
    if (!f_h) {
      FileErrorDisplay(filen);
      SWIG_exit(EXIT_FAILURE);
    }

    Swig_banner(f_h);

    Printf(f_h, "\n");
    Printf(f_h, "#ifndef PHP_%s_H\n", cap_module);
    Printf(f_h, "#define PHP_%s_H\n\n", cap_module);
    Printf(f_h, "extern zend_module_entry %s_module_entry;\n", module);
    Printf(f_h, "#define phpext_%s_ptr &%s_module_entry\n\n", module, module);
    Printf(f_h, "#ifdef PHP_WIN32\n");
    Printf(f_h, "# define PHP_%s_API __declspec(dllexport)\n", cap_module);
    Printf(f_h, "#else\n");
    Printf(f_h, "# define PHP_%s_API\n", cap_module);
    Printf(f_h, "#endif\n\n");

    /* start the arginfo section */
    s_arginfo = NewString("/* arginfo subsection */\n");
    arginfo_used = NewHash();

    /* start the function entry section */
    s_entry = NewString("/* entry subsection */\n");

    /* holds all the per-class function entry sections */
    all_cs_entry = NewString("/* class entry subsection */\n");
    cs_entry = NULL;

    Printf(s_entry, "/* Every non-class user visible function must have an entry here */\n");
    Printf(s_entry, "static zend_function_entry %s_functions[] = {\n", module);

    /* Emit all of the code */
    Language::top(n);

    SwigPHP_emit_resource_registrations();

    /* start the init section */
    {
      String * s_init_old = s_init;
      s_init = NewString("/* init section */\n");
      Printv(s_init, "zend_module_entry ", module, "_module_entry = {\n", NIL);
      Printf(s_init, "    STANDARD_MODULE_HEADER,\n");
      Printf(s_init, "    \"%s\",\n", module);
      Printf(s_init, "    %s_functions,\n", module);
      Printf(s_init, "    PHP_MINIT(%s),\n", module);
      if (Len(s_shutdown) > 0) {
	Printf(s_init, "    PHP_MSHUTDOWN(%s),\n", module);
      } else {
	Printf(s_init, "    NULL, /* No MSHUTDOWN code */\n");
      }
      if (Len(r_init) > 0 || Len(s_vinit) > 0) {
	Printf(s_init, "    PHP_RINIT(%s),\n", module);
      } else {
	Printf(s_init, "    NULL, /* No RINIT code */\n");
      }
      if (Len(r_shutdown) > 0) {
	Printf(s_init, "    PHP_RSHUTDOWN(%s),\n", module);
      } else {
	Printf(s_init, "    NULL, /* No RSHUTDOWN code */\n");
      }
      if (Len(pragma_phpinfo) > 0) {
	Printf(s_init, "    PHP_MINFO(%s),\n", module);
      } else {
	Printf(s_init, "    NULL, /* No MINFO code */\n");
      }
      Printf(s_init, "    NO_VERSION_YET,\n");
      Printf(s_init, "    STANDARD_MODULE_PROPERTIES\n");
      Printf(s_init, "};\n");
      Printf(s_init, "zend_module_entry* SWIG_module_entry = &%s_module_entry;\n\n", module);

      Printf(s_init, "#ifdef __cplusplus\n");
      Printf(s_init, "extern \"C\" {\n");
      Printf(s_init, "#endif\n");
      // We want to write "SWIGEXPORT ZEND_GET_MODULE(%s)" but ZEND_GET_MODULE
      // in PHP5 has "extern "C" { ... }" around it so we can't do that.
      Printf(s_init, "SWIGEXPORT zend_module_entry *get_module(void) { return &%s_module_entry; }\n", module);
      Printf(s_init, "#ifdef __cplusplus\n");
      Printf(s_init, "}\n");
      Printf(s_init, "#endif\n\n");

      Printf(s_init, "#define SWIG_php_minit PHP_MINIT_FUNCTION(%s)\n\n", module);

      Printv(s_init, s_init_old, NIL);
      Delete(s_init_old);
    }

    /* We have to register the constants before they are (possibly) used
     * by the pointer typemaps. This all needs re-arranging really as
     * things are being called in the wrong order
     */

    //    Printv(s_init,s_resourcetypes,NIL);
    /* We need this after all classes written out by ::top */
    Printf(s_oinit, "CG(active_class_entry) = NULL;\n");
    Printf(s_oinit, "/* end oinit subsection */\n");
    Printf(s_init, "%s\n", s_oinit);

    /* Constants generated during top call */
    Printf(s_cinit, "/* end cinit subsection */\n");
    Printf(s_init, "%s\n", s_cinit);
    Clear(s_cinit);
    Delete(s_cinit);

    Printf(s_init, "    return SUCCESS;\n");
    Printf(s_init, "}\n\n");

    // Now do REQUEST init which holds any user specified %rinit, and also vinit
    if (Len(r_init) > 0 || Len(s_vinit) > 0) {
      Printf(f_h, "PHP_RINIT_FUNCTION(%s);\n", module);

      Printf(s_init, "PHP_RINIT_FUNCTION(%s)\n{\n", module);
      if (Len(r_init) > 0) {
	Printv(s_init,
	       "/* rinit section */\n",
	       r_init, "\n",
	       NIL);
      }

      if (Len(s_vinit) > 0) {
	/* finish our init section which will have been used by class wrappers */
	Printv(s_init,
	       "/* vinit subsection */\n",
	       s_vinit, "\n"
	       "/* end vinit subsection */\n",
	       NIL);
	Clear(s_vinit);
      }
      Delete(s_vinit);

      Printf(s_init, "    return SUCCESS;\n");
      Printf(s_init, "}\n\n");
    }

    Printf(f_h, "PHP_MINIT_FUNCTION(%s);\n", module);

    if (Len(s_shutdown) > 0) {
      Printf(f_h, "PHP_MSHUTDOWN_FUNCTION(%s);\n", module);

      Printv(s_init, "PHP_MSHUTDOWN_FUNCTION(", module, ")\n"
		     "/* shutdown section */\n"
		     "{\n",
		     s_shutdown,
		     "    return SUCCESS;\n"
		     "}\n\n", NIL);
    }

    if (Len(r_shutdown) > 0) {
      Printf(f_h, "PHP_RSHUTDOWN_FUNCTION(%s);\n", module);

      Printf(s_init, "PHP_RSHUTDOWN_FUNCTION(%s)\n{\n", module);
      Printf(s_init, "/* rshutdown section */\n");
      Printf(s_init, "%s\n", r_shutdown);
      Printf(s_init, "    return SUCCESS;\n");
      Printf(s_init, "}\n\n");
    }

    if (Len(pragma_phpinfo) > 0) {
      Printf(f_h, "PHP_MINFO_FUNCTION(%s);\n", module);

      Printf(s_init, "PHP_MINFO_FUNCTION(%s)\n{\n", module);
      Printf(s_init, "%s", pragma_phpinfo);
      Printf(s_init, "}\n");
    }

    Printf(s_init, "/* end init section */\n");

    Printf(f_h, "\n#endif /* PHP_%s_H */\n", cap_module);

    Delete(f_h);

    String *type_table = NewStringEmpty();
    SwigType_emit_type_table(f_runtime, type_table);
    Printf(s_header, "%s", type_table);
    Delete(type_table);

    /* Oh dear, more things being called in the wrong order. This whole
     * function really needs totally redoing.
     */

    if (directorsEnabled()) {
      Dump(f_directors_h, f_runtime_h);
      Printf(f_runtime_h, "\n");
      Printf(f_runtime_h, "#endif\n");
      Delete(f_runtime_h);
    }

    Printf(s_header, "/* end header section */\n");
    Printf(s_wrappers, "/* end wrapper section */\n");
    Printf(s_vdecl, "/* end vdecl subsection */\n");

    Dump(f_runtime, f_begin);
    Printv(f_begin, s_header, NIL);
    if (directorsEnabled()) {
      Dump(f_directors, f_begin);
    }
    Printv(f_begin, s_vdecl, s_wrappers, NIL);
    Printv(f_begin, all_cs_entry, "\n\n", s_arginfo, "\n\n", s_entry,
	" SWIG_ZEND_NAMED_FE(swig_", module, "_alter_newobject,_wrap_swig_", module, "_alter_newobject,NULL)\n"
	" SWIG_ZEND_NAMED_FE(swig_", module, "_get_newobject,_wrap_swig_", module, "_get_newobject,NULL)\n"
	" ZEND_FE_END\n};\n\n", NIL);
    Printv(f_begin, s_init, NIL);
    Delete(s_header);
    Delete(s_wrappers);
    Delete(s_init);
    Delete(s_vdecl);
    Delete(all_cs_entry);
    Delete(s_entry);
    Delete(s_arginfo);
    Delete(f_runtime);
    Delete(f_begin);
    Delete(arginfo_used);

    Printf(f_phpcode, "%s\n%s\n", pragma_incl, pragma_code);
    if (s_fakeoowrappers) {
      Printf(f_phpcode, "abstract class %s {", Len(prefix) ? prefix : module);
      Printf(f_phpcode, "%s", s_fakeoowrappers);
      Printf(f_phpcode, "}\n\n");
      Delete(s_fakeoowrappers);
      s_fakeoowrappers = NULL;
    }
    Printf(f_phpcode, "%s\n?>\n", s_phpclasses);
    Delete(f_phpcode);

    return SWIG_OK;
  }

  /* Just need to append function names to function table to register with PHP. */
  void create_command(String *cname, String *iname, Node *n) {
    // This is for the single main zend_function_entry record
    Printf(f_h, "ZEND_NAMED_FUNCTION(%s);\n", iname);

    // We want to only emit each different arginfo once, as that reduces the
    // size of both the generated source code and the compiled extension
    // module.  To do this, we name the arginfo to encode the number of
    // parameters and which (if any) are passed by reference by using a
    // sequence of 0s (for non-reference) and 1s (for by references).
    ParmList *l = Getattr(n, "parms");
    String * arginfo_code = NewStringEmpty();
    for (Parm *p = l; p; p = Getattr(p, "tmap:in:next")) {
      /* Ignored parameters */
      if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	continue;
      }
      Append(arginfo_code, GetFlag(p, "tmap:in:byref") ? "1" : "0");
    }

    if (!GetFlag(arginfo_used, arginfo_code)) {
      // Not had this one before, so emit it.
      SetFlag(arginfo_used, arginfo_code);
      Printf(s_arginfo, "ZEND_BEGIN_ARG_INFO_EX(swig_arginfo_%s, 0, 0, 0)\n", arginfo_code);
      for (const char * p = Char(arginfo_code); *p; ++p) {
	Printf(s_arginfo, " ZEND_ARG_PASS_INFO(%c)\n", *p);
      }
      Printf(s_arginfo, "ZEND_END_ARG_INFO()\n");
    }

    String * s = cs_entry;
    if (!s) s = s_entry;
    Printf(s, " SWIG_ZEND_NAMED_FE(%(lower)s,%s,swig_arginfo_%s)\n", cname, iname, arginfo_code);
    Delete(arginfo_code);
  }

  /* ------------------------------------------------------------
   * dispatchFunction()
   * ------------------------------------------------------------ */
  void dispatchFunction(Node *n) {
    /* Last node in overloaded chain */

    int maxargs;
    String *tmp = NewStringEmpty();
    if (Swig_directorclass(n) && wrapperType == directorconstructor) {
      /* We have an extra 'this' parameter. */
      SetFlag(n, "wrap:this");
    }
    String *dispatch = Swig_overload_dispatch(n, "%s(INTERNAL_FUNCTION_PARAM_PASSTHRU); return;", &maxargs);

    /* Generate a dispatch wrapper for all overloaded functions */

    Wrapper *f = NewWrapper();
    String *symname = Getattr(n, "sym:name");
    String *wname = Swig_name_wrapper(symname);

    create_command(symname, wname, n);
    Printv(f->def, "ZEND_NAMED_FUNCTION(", wname, ") {\n", NIL);

    Wrapper_add_local(f, "argc", "int argc");

    Printf(tmp, "zval argv[%d]", maxargs);
    Wrapper_add_local(f, "argv", tmp);

    Printf(f->code, "argc = ZEND_NUM_ARGS();\n");

    Printf(f->code, "zend_get_parameters_array_ex(argc, argv);\n");

    Replaceall(dispatch, "$args", "self,args");

    Printv(f->code, dispatch, "\n", NIL);

    Printf(f->code, "SWIG_ErrorCode() = E_ERROR;\n");
    Printf(f->code, "SWIG_ErrorMsg() = \"No matching function for overloaded '%s'\";\n", symname);
    Printv(f->code, "SWIG_FAIL();\n", NIL);

    Printv(f->code, "}\n", NIL);
    Wrapper_print(f, s_wrappers);

    DelWrapper(f);
    Delete(dispatch);
    Delete(tmp);
    Delete(wname);
  }

  /* ------------------------------------------------------------
   * functionWrapper()
   * ------------------------------------------------------------ */

  /* Helper method for PHP::functionWrapper */
  bool is_class(SwigType *t) {
    Node *n = classLookup(t);
    if (n) {
      String *r = Getattr(n, "php:proxy");	// Set by classDeclaration()
      if (!r)
	r = Getattr(n, "sym:name");	// Not seen by classDeclaration yet, but this is the name
      if (r)
	return true;
    }
    return false;
  }

  virtual int functionWrapper(Node *n) {
    String *name = GetChar(n, "name");
    String *iname = GetChar(n, "sym:name");
    SwigType *d = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    String *nodeType = Getattr(n, "nodeType");
    int newobject = GetFlag(n, "feature:new");
    int constructor = (Cmp(nodeType, "constructor") == 0);

    Parm *p;
    int i;
    int numopt;
    String *tm;
    Wrapper *f;

    String *wname;
    int overloaded = 0;
    String *overname = 0;

    if (Cmp(nodeType, "destructor") == 0) {
      // We just generate the Zend List Destructor and let Zend manage the
      // reference counting.  There's no explicit destructor, but the user can
      // just do `$obj = null;' to remove a reference to an object.
      return CreateZendListDestructor(n);
    }
    // Test for overloading;
    if (Getattr(n, "sym:overloaded")) {
      overloaded = 1;
      overname = Getattr(n, "sym:overname");
    } else {
      if (!addSymbol(iname, n))
	return SWIG_ERROR;
    }

    wname = Swig_name_wrapper(iname);
    if (overname) {
      Printf(wname, "%s", overname);
    }

    f = NewWrapper();

    String *outarg = NewStringEmpty();
    String *cleanup = NewStringEmpty();

    Printv(f->def, "ZEND_NAMED_FUNCTION(", wname, ") {\n", NIL);

    emit_parameter_variables(l, f);
    /* Attach standard typemaps */

    emit_attach_parmmaps(l, f);
    // Not issued for overloaded functions.
    if (!overloaded) {
      create_command(iname, wname, n);
    }

    // wrap:parms is used by overload resolution.
    Setattr(n, "wrap:parms", l);

    int num_arguments = emit_num_arguments(l);
    int num_required = emit_num_required(l);
    numopt = num_arguments - num_required;

    if (wrapperType == directorconstructor)
      num_arguments++;

    if (num_arguments > 0) {
      String *args = NewStringEmpty();
      if (wrapperType == directorconstructor)
        Wrapper_add_local(f, "arg0", "zval * arg0");
      Printf(args, "zval args[%d]", num_arguments);
      Wrapper_add_local(f, "args", args);
      Delete(args);
      args = NULL;
    }

    // This generated code may be called:
    // 1) as an object method, or
    // 2) as a class-method/function (without a "this_ptr")
    // Option (1) has "this_ptr" for "this", option (2) needs it as
    // first parameter

    // NOTE: possible we ignore this_ptr as a param for native constructor

    Printf(f->code, "SWIG_ResetError();\n");

    if (numopt > 0) {		// membervariable wrappers do not have optional args
      Wrapper_add_local(f, "arg_count", "int arg_count");
      Printf(f->code, "arg_count = ZEND_NUM_ARGS();\n");
      Printf(f->code, "if(arg_count<%d || arg_count>%d ||\n", num_required, num_arguments);
      Printf(f->code, "   zend_get_parameters_array_ex(arg_count,args)!=SUCCESS)\n");
      Printf(f->code, "\tWRONG_PARAM_COUNT;\n\n");
    } else {
      if (num_arguments == 0) {
	Printf(f->code, "if(ZEND_NUM_ARGS() != 0) {\n");
      } else {
	Printf(f->code, "if(ZEND_NUM_ARGS() != %d || zend_get_parameters_array_ex(%d, args) != SUCCESS) {\n", num_arguments, num_arguments);
      }
      Printf(f->code, "WRONG_PARAM_COUNT;\n}\n\n");
    }
    if (wrapperType == directorconstructor)
      Printf(f->code, "arg0 = &args[0];\n  \n");

    /* Now convert from PHP to C variables */
    // At this point, argcount if used is the number of deliberately passed args
    // not including this_ptr even if it is used.
    // It means error messages may be out by argbase with error
    // reports.  We can either take argbase into account when raising
    // errors, or find a better way of dealing with _thisptr.
    // I would like, if objects are wrapped, to assume _thisptr is always
    // _this and not the first argument.
    // This may mean looking at Language::memberfunctionHandler

    int limit = num_arguments;
    if (wrapperType == directorconstructor)
      limit--;
    for (i = 0, p = l; i < limit; i++) {
      String *source;

      /* Skip ignored arguments */
      //while (Getattr(p,"tmap:ignore")) { p = Getattr(p,"tmap:ignore:next");}
      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");

      if (wrapperType == directorconstructor) {
	source = NewStringf("args[%d]", i+1);
      } else {
	source = NewStringf("args[%d]", i);
      }

      String *ln = Getattr(p, "lname");

      /* Check if optional */
      if (i >= num_required) {
	Printf(f->code, "\tif(arg_count > %d) {\n", i);
      }

      if ((tm = Getattr(p, "tmap:in"))) {
	Replaceall(tm, "$source", &source);
	Replaceall(tm, "$target", ln);
	Replaceall(tm, "$input", source);
	Setattr(p, "emit:input", source);
	Printf(f->code, "%s\n", tm);
	if (i == 0 && Getattr(p, "self")) {
	  Printf(f->code, "\tif(!arg1) SWIG_PHP_Error(E_ERROR, \"this pointer is NULL\");\n");
	}
	p = Getattr(p, "tmap:in:next");
	if (i >= num_required) {
	  Printf(f->code, "}\n");
	}
	continue;
      } else {
	Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
      }
      if (i >= num_required) {
	Printf(f->code, "\t}\n");
      }
      Delete(source);
    }

    if (is_member_director(n)) {
      Wrapper_add_local(f, "upcall", "bool upcall = false");
      Printf(f->code, "upcall = !Swig::Director::swig_is_overridden_method(\"%s%s\", \"%s\");\n",
	  prefix, Swig_class_name(Swig_methodclass(n)), name);
    }

    Swig_director_emit_dynamic_cast(n, f);

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
	Printv(cleanup, tm, "\n", NIL);
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert argument output code */
    bool hasargout = false;
    for (i = 0, p = l; p; i++) {
      if ((tm = Getattr(p, "tmap:argout"))) {
	hasargout = true;
	Replaceall(tm, "$source", Getattr(p, "lname"));
	//      Replaceall(tm,"$input",Getattr(p,"lname"));
	Replaceall(tm, "$target", "return_value");
	Replaceall(tm, "$result", "return_value");
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(outarg, tm, "\n", NIL);
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    Setattr(n, "wrap:name", wname);

    /* emit function call */
    String *actioncode = emit_action(n);

    if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
      Replaceall(tm, "$input", Swig_cresult_name());
      Replaceall(tm, "$source", Swig_cresult_name());
      Replaceall(tm, "$target", "return_value");
      Replaceall(tm, "$result", "return_value");
      Replaceall(tm, "$owner", newobject ? "1" : "0");
      Printf(f->code, "%s\n", tm);
    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(d, 0), name);
    }
    emit_return_variable(n, d, f);

    if (outarg) {
      Printv(f->code, outarg, NIL);
    }

    if (cleanup) {
      Printv(f->code, cleanup, NIL);
    }

    /* Look to see if there is any newfree cleanup code */
    if (GetFlag(n, "feature:new")) {
      if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
	Printf(f->code, "%s\n", tm);
	Delete(tm);
      }
    }

    /* See if there is any return cleanup code */
    if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
      Printf(f->code, "%s\n", tm);
      Delete(tm);
    }

    Printf(f->code, "thrown:\n");
    Printf(f->code, "return;\n");

    /* Error handling code */
    Printf(f->code, "fail:\n");
    Printv(f->code, cleanup, NIL);
    Append(f->code, "SWIG_FAIL();\n");

    Printf(f->code, "}\n");

    Replaceall(f->code, "$cleanup", cleanup);
    Replaceall(f->code, "$symname", iname);

    Wrapper_print(f, s_wrappers);
    DelWrapper(f);
    f = NULL;

    if (overloaded && !Getattr(n, "sym:nextSibling")) {
      dispatchFunction(n);
    }

    Delete(wname);
    wname = NULL;

    if (!shadow) {
      return SWIG_OK;
    }

    // Handle getters and setters.
    if (wrapperType == membervar) {
      const char *p = Char(iname);
      if (strlen(p) > 4) {
	p += strlen(p) - 4;
	String *varname = Getattr(n, "membervariableHandler:sym:name");
	if (strcmp(p, "_get") == 0) {
	  Setattr(shadow_get_vars, varname, Getattr(n, "type"));
	} else if (strcmp(p, "_set") == 0) {
	  Setattr(shadow_set_vars, varname, iname);
	}
      }
      return SWIG_OK;
    }

    // Only look at non-overloaded methods and the last entry in each overload
    // chain (we check the last so that wrap:parms and wrap:name have been set
    // for them all).
    if (overloaded && Getattr(n, "sym:nextSibling") != 0)
      return SWIG_OK;

    if (!s_oowrappers)
      s_oowrappers = NewStringEmpty();

    if (newobject || wrapperType == memberfn || wrapperType == staticmemberfn || wrapperType == standard || wrapperType == staticmembervar) {
      bool handle_as_overload = false;
      String **arg_names;
      String **arg_values;
      unsigned char * byref;
      // Method or static method or plain function.
      const char *methodname = 0;
      String *output = s_oowrappers;
      if (constructor) {
	class_has_ctor = true;
	// Skip the Foo:: prefix.
	char *ptr = strrchr(GetChar(Swig_methodclass(n), "sym:name"), ':');
	if (ptr) {
	  ptr++;
	} else {
	  ptr = GetChar(Swig_methodclass(n), "sym:name");
	}
	if (strcmp(ptr, GetChar(n, "constructorHandler:sym:name")) == 0) {
	  methodname = "__construct";
	} else {
	  // The class has multiple constructors and this one is
	  // renamed, so this will be a static factory function
	  methodname = GetChar(n, "constructorHandler:sym:name");
	}
      } else if (wrapperType == memberfn) {
	methodname = Char(Getattr(n, "memberfunctionHandler:sym:name"));
      } else if (wrapperType == staticmemberfn) {
	methodname = Char(Getattr(n, "staticmemberfunctionHandler:sym:name"));
      } else if (wrapperType == staticmembervar) {
	// Static member variable, wrapped as a function due to PHP limitations.
	methodname = Char(Getattr(n, "staticmembervariableHandler:sym:name"));
      } else {			// wrapperType == standard
	methodname = Char(iname);
	if (!s_fakeoowrappers)
	  s_fakeoowrappers = NewStringEmpty();
	output = s_fakeoowrappers;
      }

      bool really_overloaded = overloaded ? true : false;
      int min_num_of_arguments = emit_num_required(l);
      int max_num_of_arguments = emit_num_arguments(l);

      Hash *ret_types = NewHash();
      Setattr(ret_types, d, d);

      bool non_void_return = (Cmp(d, "void") != 0);

      if (overloaded) {
	// Look at all the overloaded versions of this method in turn to
	// decide if it's really an overloaded method, or just one where some
	// parameters have default values.
	Node *o = Getattr(n, "sym:overloaded");
	while (o) {
	  if (o == n) {
	    o = Getattr(o, "sym:nextSibling");
	    continue;
	  }

	  SwigType *d2 = Getattr(o, "type");
	  if (!d2) {
	    assert(constructor);
	  } else if (!Getattr(ret_types, d2)) {
	    Setattr(ret_types, d2, d2);
	    non_void_return = non_void_return || (Cmp(d2, "void") != 0);
	  }

	  ParmList *l2 = Getattr(o, "wrap:parms");
	  int num_arguments = emit_num_arguments(l2);
	  int num_required = emit_num_required(l2);
	  if (num_required < min_num_of_arguments)
	    min_num_of_arguments = num_required;

	  if (num_arguments > max_num_of_arguments) {
	    max_num_of_arguments = num_arguments;
	  }
	  o = Getattr(o, "sym:nextSibling");
	}

	o = Getattr(n, "sym:overloaded");
	while (o) {
	  if (o == n) {
	    o = Getattr(o, "sym:nextSibling");
	    continue;
	  }

	  ParmList *l2 = Getattr(o, "wrap:parms");
	  Parm *p = l, *p2 = l2;
	  if (wrapperType == memberfn) {
	    p = nextSibling(p);
	    p2 = nextSibling(p2);
	  }
	  while (p && p2) {
	    if (Cmp(Getattr(p, "type"), Getattr(p2, "type")) != 0)
	      break;
	    if (Cmp(Getattr(p, "name"), Getattr(p2, "name")) != 0)
	      break;
	    String *value = Getattr(p, "value");
	    String *value2 = Getattr(p2, "value");
	    if (value && !value2)
	      break;
	    if (!value && value2)
	      break;
	    if (value) {
	      if (Cmp(value, value2) != 0)
		break;
	    }
	    p = nextSibling(p);
	    p2 = nextSibling(p2);
	  }
	  if (p && p2)
	    break;
	  // One parameter list is a prefix of the other, so check that all
	  // remaining parameters of the longer list are optional.
	  if (p2)
	    p = p2;
	  while (p && Getattr(p, "value"))
	    p = nextSibling(p);
	  if (p)
	    break;
	  o = Getattr(o, "sym:nextSibling");
	}
	if (!o) {
	  // This "overloaded method" is really just one with default args.
	  really_overloaded = false;
	}
      }

      if (wrapperType == memberfn) {
	// Allow for the "this" pointer.
	--min_num_of_arguments;
	--max_num_of_arguments;
      }

      arg_names = (String **) malloc(max_num_of_arguments * sizeof(String *));
      if (!arg_names) {
	/* FIXME: How should this be handled?  The rest of SWIG just seems
	 * to not bother checking for malloc failing! */
	fprintf(stderr, "Malloc failed!\n");
	exit(1);
      }
      for (i = 0; i < max_num_of_arguments; ++i) {
	arg_names[i] = NULL;
      }

      arg_values = (String **) malloc(max_num_of_arguments * sizeof(String *));
      byref = (unsigned char *) malloc(max_num_of_arguments);
      if (!arg_values || !byref) {
	/* FIXME: How should this be handled?  The rest of SWIG just seems
	 * to not bother checking for malloc failing! */
	fprintf(stderr, "Malloc failed!\n");
	exit(1);
      }
      for (i = 0; i < max_num_of_arguments; ++i) {
	arg_values[i] = NULL;
	byref[i] = false;
      }

      Node *o;
      if (overloaded) {
	o = Getattr(n, "sym:overloaded");
      } else {
	o = n;
      }
      while (o) {
	int argno = 0;
	Parm *p = Getattr(o, "wrap:parms");
	if (wrapperType == memberfn)
	  p = nextSibling(p);
	while (p) {
	  if (GetInt(p, "tmap:in:numinputs") == 0) {
	    p = nextSibling(p);
	    continue;
	  }
	  assert(0 <= argno && argno < max_num_of_arguments);
	  byref[argno] = GetFlag(p, "tmap:in:byref");
	  String *&pname = arg_names[argno];
	  const char *pname_cstr = GetChar(p, "name");
	  // Just get rid of the C++ namespace part for now.
	  const char *ptr = NULL;
	  if (pname_cstr && (ptr = strrchr(pname_cstr, ':'))) {
	    pname_cstr = ptr + 1;
	  }
	  if (!pname_cstr) {
	    // Unnamed parameter, e.g. int foo(int);
	  } else if (!pname) {
	    pname = NewString(pname_cstr);
	  } else {
	    size_t len = strlen(pname_cstr);
	    size_t spc = 0;
	    size_t len_pname = strlen(Char(pname));
	    while (spc + len <= len_pname) {
	      if (strncmp(pname_cstr, Char(pname) + spc, len) == 0) {
		char ch = ((char *) Char(pname))[spc + len];
		if (ch == '\0' || ch == ' ') {
		  // Already have this pname_cstr.
		  pname_cstr = NULL;
		  break;
		}
	      }
	      char *p = strchr(Char(pname) + spc, ' ');
	      if (!p)
		break;
	      spc = (p + 4) - Char(pname);
	    }
	    if (pname_cstr) {
	      Printf(pname, " or_%s", pname_cstr);
	    }
	  }
	  String *value = NewString(Getattr(p, "value"));
	  if (Len(value)) {
	    /* Check that value is a valid constant in PHP (and adjust it if
	     * necessary, or replace it with "?" if it's just not valid). */
	    SwigType *type = Getattr(p, "type");
	    switch (SwigType_type(type)) {
	      case T_BOOL: {
		if (Strcmp(value, "true") == 0 || Strcmp(value, "false") == 0)
		  break;
		char *p;
		errno = 0;
		long n = strtol(Char(value), &p, 0);
	        Clear(value);
		if (errno || *p) {
		  Append(value, "?");
		} else if (n) {
		  Append(value, "true");
		} else {
		  Append(value, "false");
		}
		break;
	      }
	      case T_CHAR:
	      case T_SCHAR:
	      case T_SHORT:
	      case T_INT:
	      case T_LONG:
	      case T_LONGLONG: {
		char *p;
		errno = 0;
		long n = strtol(Char(value), &p, 0);
		(void) n;
		if (errno || *p) {
		  Clear(value);
		  Append(value, "?");
		}
		break;
	      }
	      case T_UCHAR:
	      case T_USHORT:
	      case T_UINT:
	      case T_ULONG:
	      case T_ULONGLONG: {
		char *p;
		errno = 0;
		unsigned int n = strtoul(Char(value), &p, 0);
		(void) n;
		if (errno || *p) {
		  Clear(value);
		  Append(value, "?");
		}
		break;
	      }
	      case T_FLOAT:
	      case T_DOUBLE:
	      case T_LONGDOUBLE: {
		char *p;
		errno = 0;
		/* FIXME: strtod is locale dependent... */
		double val = strtod(Char(value), &p);
		if (errno || *p) {
		  Clear(value);
		  Append(value, "?");
		} else if (strchr(Char(value), '.') == 0) {
		  // Ensure value is a double constant, not an integer one.
		  Append(value, ".0");
		  double val2 = strtod(Char(value), &p);
		  if (errno || *p || val != val2) {
		    Clear(value);
		    Append(value, "?");
		  }
		}
		break;
	      }
	      case T_STRING:
		if (Len(value) < 2) {
		  // How can a string (including "" be less than 2 characters?)
		  Clear(value);
		  Append(value, "?");
		} else {
		  const char *v = Char(value);
		  if (v[0] != '"' || v[Len(value) - 1] != '"') {
		    Clear(value);
		    Append(value, "?");
		  }
		  // Strings containing "$" require special handling, but we do
		  // that later.
		}
		break;
	      case T_VOID:
		assert(false);
		break;
	      case T_POINTER: {
		const char *v = Char(value);
		if (v[0] == '(') {
		  // Handle "(void*)0", "(TYPE*)0", "(char*)NULL", etc.
		  v += strcspn(v + 1, "*()") + 1;
		  if (*v == '*') {
		    do {
		      v++;
		      v += strspn(v, " \t");
		    } while (*v == '*');
		    if (*v++ == ')') {
		      v += strspn(v, " \t");
		      String * old = value;
		      value = NewString(v);
		      Delete(old);
		    }
		  }
		}
		if (Strcmp(value, "NULL") == 0 ||
		    Strcmp(value, "nullptr") == 0 ||
		    Strcmp(value, "0") == 0 ||
		    Strcmp(value, "0L") == 0) {
		  Clear(value);
		  Append(value, "null");
		} else {
		  Clear(value);
		  Append(value, "?");
		}
		break;
	      }
	      default:
		/* Safe default */
		Clear(value);
		Append(value, "?");
		break;
	    }

	    if (!arg_values[argno]) {
	      arg_values[argno] = value;
	      value = NULL;
	    } else if (Cmp(arg_values[argno], value) != 0) {
	      // If a parameter has two different default values in
	      // different overloaded forms of the function, we can't
	      // set its default in PHP.  Flag this by setting its
	      // default to `?'.
	      Delete(arg_values[argno]);
	      arg_values[argno] = NewString("?");
	    }
	  } else if (arg_values[argno]) {
	    // This argument already has a default value in another overloaded
	    // form, but doesn't in this form.  So don't try to do anything
	    // clever, just let the C wrappers resolve the overload and set the
	    // default values.
	    //
	    // This handling is safe, but I'm wondering if it may be overly
	    // conservative (FIXME) in some cases.  It seems it's only bad when
	    // there's an overloaded form with the appropriate number of
	    // parameters which doesn't want the default value, but I need to
	    // think about this more.
	    Delete(arg_values[argno]);
	    arg_values[argno] = NewString("?");
	  }
	  Delete(value);
	  p = nextSibling(p);
	  ++argno;
	}
	if (!really_overloaded)
	  break;
	o = Getattr(o, "sym:nextSibling");
      }

      /* Clean up any parameters which haven't yet got names, or whose
       * names clash. */
      Hash *seen = NewHash();
      /* We need $this to refer to the current class, so can't allow it
       * to be used as a parameter. */
      Setattr(seen, "this", seen);
      /* We use $r to store the return value, so disallow that as a parameter
       * name in case the user uses the "call-time pass-by-reference" feature
       * (it's deprecated and off by default in PHP5, but we want to be
       * maximally portable).  Similarly we use $c for the classname or new
       * stdClass object.
       */
      Setattr(seen, "r", seen);
      Setattr(seen, "c", seen);

      for (int argno = 0; argno < max_num_of_arguments; ++argno) {
	String *&pname = arg_names[argno];
	if (pname) {
	  Replaceall(pname, " ", "_");
	} else {
	  /* We get here if the SWIG .i file has "int foo(int);" */
	  pname = NewStringEmpty();
	  Printf(pname, "arg%d", argno + 1);
	}
	// Check if we've already used this parameter name.
	while (Getattr(seen, pname)) {
	  // Append "_" to clashing names until they stop clashing...
	  Printf(pname, "_");
	}
	Setattr(seen, Char(pname), seen);

	if (arg_values[argno] && Cmp(arg_values[argno], "?") == 0) {
	  handle_as_overload = true;
	}
      }
      Delete(seen);
      seen = NULL;

      String *invoke = NewStringEmpty();
      String *prepare = NewStringEmpty();
      String *args = NewStringEmpty();

      if (!handle_as_overload && !(really_overloaded && max_num_of_arguments > min_num_of_arguments)) {
	Printf(invoke, "%s(", iname);
	if (wrapperType == memberfn) {
	  Printf(invoke, "$this->%s", SWIG_PTR);
	}
	for (int i = 0; i < max_num_of_arguments; ++i) {
	  if (i)
	    Printf(args, ",");
	  if (i || wrapperType == memberfn)
	    Printf(invoke, ",");
	  if (byref[i]) Printf(args, "&");
	  String *value = arg_values[i];
	  if (value) {
	    const char *v = Char(value);
	    if (v[0] == '"') {
	      /* In a PHP double quoted string, $ needs to be escaped as \$. */
	      Replaceall(value, "$", "\\$");
	    }
	    Printf(args, "$%s=%s", arg_names[i], value);
	  } else if (constructor && i >= 1 && i < min_num_of_arguments) {
	    // We need to be able to call __construct($resource).
	    Printf(args, "$%s=null", arg_names[i]);
	  } else {
	    Printf(args, "$%s", arg_names[i]);
	  }
	  Printf(invoke, "$%s", arg_names[i]);
	}
	Printf(invoke, ")");
      } else {
	int i;
	for (i = 0; i < min_num_of_arguments; ++i) {
	  if (i)
	    Printf(args, ",");
	  Printf(args, "$%s", arg_names[i]);
	}
	String *invoke_args = NewStringEmpty();
	if (wrapperType == memberfn) {
	  Printf(invoke_args, "$this->%s", SWIG_PTR);
	  if (min_num_of_arguments > 0)
	    Printf(invoke_args, ",");
	}
	Printf(invoke_args, "%s", args);
	if (constructor && min_num_of_arguments > 1) {
	  // We need to be able to call __construct($resource).
	  Clear(args);
	  Printf(args, "$%s", arg_names[0]);
	  for (i = 1; i < min_num_of_arguments; ++i) {
	    Printf(args, ",");
	    Printf(args, "$%s=null", arg_names[i]);
	  }
	}
	bool had_a_case = false;
	int last_handled_i = i - 1;
	for (; i < max_num_of_arguments; ++i) {
	  if (i)
	    Printf(args, ",");
	  const char *value = Char(arg_values[i]);
	  // FIXME: (really_overloaded && handle_as_overload) is perhaps a
	  // little conservative, but it doesn't hit any cases that it
	  // shouldn't for Xapian at least (and we need it to handle
	  // "Enquire::get_mset()" correctly).
	  bool non_php_default = ((really_overloaded && handle_as_overload) ||
				  !value || strcmp(value, "?") == 0);
	  if (non_php_default)
	    value = "null";
	  Printf(args, "$%s=%s", arg_names[i], value);
	  if (non_php_default) {
	    if (!had_a_case) {
	      Printf(prepare, "\t\tswitch (func_num_args()) {\n");
	      had_a_case = true;
	    }
	    Printf(prepare, "\t\t");
	    while (last_handled_i < i) {
	      Printf(prepare, "case %d: ", ++last_handled_i);
	    }
	    if (non_void_return) {
	      if ((!directorsEnabled() || !Swig_directorclass(n)) && !newobject) {
		Append(prepare, "$r=");
	      } else {
		Printf(prepare, "$this->%s=", SWIG_PTR);
	      }
	    }
	    if (!directorsEnabled() || !Swig_directorclass(n) || !newobject) {
	      Printf(prepare, "%s(%s); break;\n", iname, invoke_args);
	    } else if (!i) {
	      Printf(prepare, "%s($_this%s); break;\n", iname, invoke_args);
	    } else {
	      Printf(prepare, "%s($_this, %s); break;\n", iname, invoke_args);
	    }
	  }
	  if (i || wrapperType == memberfn)
	    Printf(invoke_args, ",");
	  Printf(invoke_args, "$%s", arg_names[i]);
	}
	Printf(prepare, "\t\t");
	if (had_a_case)
	  Printf(prepare, "default: ");
	if (non_void_return) {
	  if ((!directorsEnabled() || !Swig_directorclass(n)) && !newobject) {
	    Append(prepare, "$r=");
	  } else {
	    Printf(prepare, "$this->%s=", SWIG_PTR);
	  }
	}

	if (!directorsEnabled() || !Swig_directorclass(n) || !newobject) {
	  Printf(prepare, "%s(%s);\n", iname, invoke_args);
	} else {
	  Printf(prepare, "%s($_this, %s);\n", iname, invoke_args);
	}
	if (had_a_case)
	  Printf(prepare, "\t\t}\n");
	Delete(invoke_args);
	Printf(invoke, "$r");
      }

      Printf(output, "\n");
      // If it's a member function or a class constructor...
      if (wrapperType == memberfn || (constructor && current_class)) {
	String *acc = NewString(Getattr(n, "access"));
	// If a base has the same method with public access, then PHP
	// requires to have it here as public as well
	Node *bases = Getattr(Swig_methodclass(n), "bases");
	if (bases && Strcmp(acc, "public") != 0) {
	  String *warnmsg = 0;
	  int haspublicbase = 0;
	  Iterator i = First(bases);
	  while (i.item) {
	    Node *j = firstChild(i.item);
	    while (j) {
	      String *jname = Getattr(j, "name");
	      if (!jname || Strcmp(jname, Getattr(n, "name")) != 0) {
		j = nextSibling(j);
		continue;
	      }
	      if (Strcmp(nodeType(j), "cdecl") == 0) {
		if (!Getattr(j, "access") || checkAttribute(j, "access", "public")) {
		  haspublicbase = 1;
		}
	      } else if (Strcmp(nodeType(j), "using") == 0 && firstChild(j) && Strcmp(nodeType(firstChild(j)), "cdecl") == 0) {
		if (!Getattr(firstChild(j), "access") || checkAttribute(firstChild(j), "access", "public")) {
		  haspublicbase = 1;
		}
	      }
	      if (haspublicbase) {
		  warnmsg = NewStringf("Modifying the access of '%s::%s' to public, as the base '%s' has it as public as well.\n", Getattr(current_class, "classtype"), Getattr(n, "name"), Getattr(i.item, "classtype"));
		  break;
	      }
	      j = nextSibling(j);
	    }
	    i = Next(i);
	    if (haspublicbase) {
	      break;
	    }
	  }
	  if (Getattr(n, "access") && haspublicbase) {
	    Delete(acc);
	    acc = NewStringEmpty(); // implicitly public
	    Swig_warning(WARN_PHP_PUBLIC_BASE, input_file, line_number, Char(warnmsg));
	    Delete(warnmsg);
	  }
	}

	if (Cmp(acc, "public") == 0) {
	  // The default visibility for methods is public, so don't specify
	  // that explicitly to keep the wrapper size down.
	  Delete(acc);
	  acc = NewStringEmpty();
	} else if (Cmp(acc, "") != 0) {
	  Append(acc, " ");
	}

	if (constructor) {
	  const char * arg0;
	  if (max_num_of_arguments > 0) {
	    arg0 = Char(arg_names[0]);
	  } else {
	    arg0 = "res";
	    Delete(args);
	    args = NewString("$res=null");
	  }
	  String *mangled_type = SwigType_manglestr(Getattr(n, "type"));
	  Printf(output, "\t%sfunction %s(%s) {\n", acc, methodname, args);
	  Printf(output, "\t\tif (is_resource($%s) && get_resource_type($%s) === '%s') {\n", arg0, arg0, mangled_type);
	  Printf(output, "\t\t\t$this->%s=$%s;\n", SWIG_PTR, arg0);
	  Printf(output, "\t\t\treturn;\n");
	  Printf(output, "\t\t}\n");
	} else {
	  Printf(output, "\t%sfunction %s(%s) {\n", acc, methodname, args);
	}
	Delete(acc);
      } else if (wrapperType == staticmembervar) {
	// We're called twice for a writable static member variable - first
	// with "foo_set" and then with "foo_get" - so generate half the
	// wrapper function each time.
	//
	// For a const static member, we only get called once.
	static bool started = false;
	if (!started) {
	  Printf(output, "\tstatic function %s() {\n", methodname);
	  if (max_num_of_arguments) {
	    // Setter.
	    Printf(output, "\t\tif (func_num_args()) {\n");
	    Printf(output, "\t\t\t%s(func_get_arg(0));\n", iname);
	    Printf(output, "\t\t\treturn;\n");
	    Printf(output, "\t\t}\n");
	    started = true;
	    goto done;
	  }
	}
	started = false;
      } else {
	Printf(output, "\tstatic function %s(%s) {\n", methodname, args);
      }

      if (!newobject)
	Printf(output, "%s", prepare);
      if (constructor) {
	if (!directorsEnabled() || !Swig_directorclass(n)) {
	  if (!Len(prepare)) {
	    if (strcmp(methodname, "__construct") == 0) {
	      Printf(output, "\t\t$this->%s=%s;\n", SWIG_PTR, invoke);
	    } else {
	      String *classname = Swig_class_name(current_class);
	      Printf(output, "\t\treturn new %s%s(%s);\n", prefix, classname, invoke);
	    }
	  }
	} else {
	  Node *parent = Swig_methodclass(n);
	  String *classname = Swig_class_name(parent);
	  Printf(output, "\t\tif (get_class($this) === '%s%s') {\n", prefix, classname);
	  Printf(output, "\t\t\t$_this = null;\n");
	  Printf(output, "\t\t} else {\n");
	  Printf(output, "\t\t\t$_this = $this;\n");
	  Printf(output, "\t\t}\n");
	  if (!Len(prepare)) {
	    if (num_arguments > 1) {
	      Printf(output, "\t\t$this->%s=%s($_this, %s);\n", SWIG_PTR, iname, args);
	    } else {
	      Printf(output, "\t\t$this->%s=%s($_this);\n", SWIG_PTR, iname);
	    }
	  }
	}
	Printf(output, "%s", prepare);
      } else if (!non_void_return && !hasargout) {
	if (Cmp(invoke, "$r") != 0)
	  Printf(output, "\t\t%s;\n", invoke);
      } else if (is_class(d)) {
	if (Cmp(invoke, "$r") != 0)
	  Printf(output, "\t\t$r=%s;\n", invoke);
	if (Len(ret_types) == 1) {
	  /* If d is abstract we can't create a new wrapper type d. */
	  Node *d_class = classLookup(d);
	  int is_abstract = 0;
	  if (Getattr(d_class, "abstracts")) {
	    is_abstract = 1;
	  }
	  if (newobject || !is_abstract) {
	    Printf(output, "\t\tif (is_resource($r)) {\n");
	    if (Getattr(classLookup(Getattr(n, "type")), "module")) {
	      /*
	       * _p_Foo -> Foo, _p_ns__Bar -> Bar
	       * TODO: do this in a more elegant way
	       */
	      if (Len(prefix) == 0) {
		Printf(output, "\t\t\t$c=substr(get_resource_type($r), (strpos(get_resource_type($r), '__') ? strpos(get_resource_type($r), '__') + 2 : 3));\n");
	      } else {
		Printf(output, "\t\t\t$c='%s'.substr(get_resource_type($r), (strpos(get_resource_type($r), '__') ? strpos(get_resource_type($r), '__') + 2 : 3));\n", prefix);
	      }
	      Printf(output, "\t\t\tif (class_exists($c)) return new $c($r);\n");
	      Printf(output, "\t\t\treturn new %s%s($r);\n", prefix, Getattr(classLookup(d), "sym:name"));
	    } else {
	      Printf(output, "\t\t\t$c = new stdClass();\n");
	      Printf(output, "\t\t\t$c->" SWIG_PTR " = $r;\n");
	      Printf(output, "\t\t\treturn $c;\n");
	    }
	    Printf(output, "\t\t}\n\t\treturn $r;\n");
	  } else {
	    Printf(output, "\t\t$this->%s = $r;\n", SWIG_PTR);
	    Printf(output, "\t\treturn $this;\n");
	  }
	} else {
	  Printf(output, "\t\tif (!is_resource($r)) return $r;\n");
	  String *wrapobj = NULL;
	  String *common = NULL;
	  Iterator i = First(ret_types);
	  while (i.item) {
	    SwigType *ret_type = i.item;
	    i = Next(i);
	    String *mangled = NewString("_p");
	    Printf(mangled, "%s", SwigType_manglestr(ret_type));
	    Node *class_node = Getattr(zend_types, mangled);
	    if (!class_node) {
	      /* This is needed when we're returning a pointer to a type
	       * rather than returning the type by value or reference. */
	      Delete(mangled);
	      mangled = NewString(SwigType_manglestr(ret_type));
	      class_node = Getattr(zend_types, mangled);
	      if (!class_node) {
		// Return type isn't an object, so will be handled by the
		// !is_resource() check before the switch.
		continue;
	      }
	    }
	    const char *classname = GetChar(class_node, "sym:name");
	    if (!classname)
	      classname = GetChar(class_node, "name");
	    String * action = NewStringEmpty();
	    if (classname)
	      Printf(action, "return new %s%s($r);\n", prefix, classname);
            else
	      Printf(action, "return $r;\n");
	    if (!wrapobj) {
		wrapobj = NewString("\t\tswitch (get_resource_type($r)) {\n");
		common = action;
	    } else {
		if (common && Cmp(common, action) != 0) {
		    Delete(common);
		    common = NULL;
		}
	    }
	    Printf(wrapobj, "\t\t");
	    if (i.item) {
	      Printf(wrapobj, "case '%s': ", mangled);
	    } else {
	      Printf(wrapobj, "default: ");
	    }
	    Printv(wrapobj, action, NIL);
	    if (action != common) Delete(action);
	    Delete(mangled);
	  }
	  Printf(wrapobj, "\t\t}\n");
	  if (common) {
	      // All cases have the same action, so eliminate the switch
	      // wrapper.
	      Printf(output, "\t\t%s", common);
	      Delete(common);
	  } else {
	      Printv(output, wrapobj, NIL);
	  }
	  Delete(wrapobj);
	}
      } else {
	if (non_void_return) {
	  Printf(output, "\t\treturn %s;\n", invoke);
	} else if (Cmp(invoke, "$r") != 0) {
	  Printf(output, "\t\t%s;\n", invoke);
	}
      }
      Printf(output, "\t}\n");

done:
      Delete(prepare);
      Delete(invoke);
      free(arg_values);

      Delete(args);
      args = NULL;

      for (int i = 0; i < max_num_of_arguments; ++i) {
	Delete(arg_names[i]);
      }
      free(arg_names);
      arg_names = NULL;
    }

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * globalvariableHandler()
   * ------------------------------------------------------------ */

  virtual int globalvariableHandler(Node *n) {
    char *name = GetChar(n, "name");
    char *iname = GetChar(n, "sym:name");
    SwigType *t = Getattr(n, "type");
    String *tm;

    /* First do the wrappers such as name_set(), name_get()
     * as provided by the baseclass's implementation of variableWrapper
     */
    if (Language::globalvariableHandler(n) == SWIG_NOWRAP) {
      return SWIG_NOWRAP;
    }

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    /* First link C variables to PHP */

    tm = Swig_typemap_lookup("varinit", n, name, 0);
    if (tm) {
      Replaceall(tm, "$target", name);
      Printf(s_vinit, "%s\n", tm);
    } else {
      Swig_error(input_file, line_number, "Unable to link with type %s\n", SwigType_str(t, 0));
    }

    /* Now generate PHP -> C sync blocks */
    /*
       tm = Swig_typemap_lookup("varin", n, name, 0);
       if(tm) {
       Replaceall(tm, "$symname", iname);
       Printf(f_c->code, "%s\n", tm);
       } else {
       Swig_error(input_file, line_number, "Unable to link with type %s\n", SwigType_str(t, 0));
       }
     */
    /* Now generate C -> PHP sync blocks */
    /*
       if(!GetFlag(n,"feature:immutable")) {

       tm = Swig_typemap_lookup("varout", n, name, 0);
       if(tm) {
       Replaceall(tm, "$symname", iname);
       Printf(f_php->code, "%s\n", tm);
       } else {
       Swig_error(input_file, line_number, "Unable to link with type %s\n", SwigType_str(t, 0));
       }
       }
     */
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constantWrapper()
   * ------------------------------------------------------------ */

  virtual int constantWrapper(Node *n) {
    String *name = GetChar(n, "name");
    String *iname = GetChar(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");
    String *tm;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    SwigType_remember(type);

    if ((tm = Swig_typemap_lookup("consttab", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", name);
      Replaceall(tm, "$value", value);
      Printf(s_cinit, "%s\n", tm);
    }

    if (shadow) {
      String *enumvalue = GetChar(n, "enumvalue");
      String *set_to = iname;

      if (!enumvalue) {
	enumvalue = GetChar(n, "enumvalueex");
      }

      if (enumvalue && *Char(enumvalue)) {
	// Check for a simple constant expression which is valid in PHP.
	// If we find one, initialise the const member with it; otherwise
	// we initialise it using the C/C++ wrapped constant.
	const char *p;
	for (p = Char(enumvalue); *p; ++p) {
	  if (!isdigit((unsigned char)*p) && !strchr(" +-", *p)) {
	    // FIXME: enhance to handle `<previous_enum> + 1' which is what
	    // we get for enums that don't have an explicit value set.
	    break;
	  }
	}
	if (!*p)
	  set_to = enumvalue;
      }

      if (wrapping_member_constant) {
	if (!s_oowrappers)
	  s_oowrappers = NewStringEmpty();
	Printf(s_oowrappers, "\n\tconst %s = %s;\n", wrapping_member_constant, set_to);
      } else {
	if (!s_fakeoowrappers)
	  s_fakeoowrappers = NewStringEmpty();
	Printf(s_fakeoowrappers, "\n\tconst %s = %s;\n", iname, set_to);
      }
    }

    return SWIG_OK;
  }

  /*
   * PHP::pragma()
   *
   * Pragma directive.
   *
   * %pragma(php) code="String"         # Includes a string in the .php file
   * %pragma(php) include="file.php"    # Includes a file in the .php file
   */

  virtual int pragmaDirective(Node *n) {
    if (!ImportMode) {
      String *lang = Getattr(n, "lang");
      String *type = Getattr(n, "name");
      String *value = Getattr(n, "value");

      if (Strcmp(lang, "php") == 0 || Strcmp(lang, "php4") == 0) {
	if (Strcmp(type, "code") == 0) {
	  if (value) {
	    Printf(pragma_code, "%s\n", value);
	  }
	} else if (Strcmp(type, "include") == 0) {
	  if (value) {
	    Printf(pragma_incl, "include '%s';\n", value);
	  }
	} else if (Strcmp(type, "phpinfo") == 0) {
	  if (value) {
	    Printf(pragma_phpinfo, "%s\n", value);
	  }
	} else {
	  Swig_warning(WARN_PHP_UNKNOWN_PRAGMA, input_file, line_number, "Unrecognized pragma <%s>.\n", type);
	}
      }
    }
    return Language::pragmaDirective(n);
  }

  /* ------------------------------------------------------------
   * classDeclaration()
   * ------------------------------------------------------------ */

  virtual int classDeclaration(Node *n) {
    if (!Getattr(n, "feature:onlychildren")) {
      String *symname = Getattr(n, "sym:name");
      Setattr(n, "php:proxy", symname);
    }

    return Language::classDeclaration(n);
  }

  /* ------------------------------------------------------------
   * classHandler()
   * ------------------------------------------------------------ */

  virtual int classHandler(Node *n) {
    constructors = 0;
    current_class = n;

    if (shadow) {
      char *rename = GetChar(n, "sym:name");

      if (!addSymbol(rename, n))
	return SWIG_ERROR;
      shadow_classname = NewString(rename);

      shadow_get_vars = NewHash();
      shadow_set_vars = NewHash();

      /* Deal with inheritance */
      List *baselist = Getattr(n, "bases");
      if (baselist) {
	Iterator base = First(baselist);
	while (base.item && GetFlag(base.item, "feature:ignore")) {
	  base = Next(base);
	}
	base = Next(base);
	if (base.item) {
	  /* Warn about multiple inheritance for additional base class(es) */
	  while (base.item) {
	    if (GetFlag(base.item, "feature:ignore")) {
	      base = Next(base);
	      continue;
	    }
	    String *proxyclassname = SwigType_str(Getattr(n, "classtypeobj"), 0);
	    String *baseclassname = SwigType_str(Getattr(base.item, "name"), 0);
	    Swig_warning(WARN_PHP_MULTIPLE_INHERITANCE, input_file, line_number,
			 "Warning for %s, base %s ignored. Multiple inheritance is not supported in PHP.\n", proxyclassname, baseclassname);
	    base = Next(base);
	  }
	}
      }
    }

    classnode = n;
    Language::classHandler(n);
    classnode = 0;

    if (shadow) {
      List *baselist = Getattr(n, "bases");
      Iterator ki, base;

      if (baselist) {
	base = First(baselist);
	while (base.item && GetFlag(base.item, "feature:ignore")) {
	  base = Next(base);
	}
      } else {
	base.item = NULL;
      }

      if (Getattr(n, "abstracts") && !GetFlag(n, "feature:notabstract")) {
	Printf(s_phpclasses, "abstract ");
      }

      Printf(s_phpclasses, "class %s%s ", prefix, shadow_classname);
      String *baseclass = NULL;
      if (base.item && Getattr(base.item, "module")) {
	baseclass = Getattr(base.item, "sym:name");
	if (!baseclass)
	  baseclass = Getattr(base.item, "name");
	Printf(s_phpclasses, "extends %s%s ", prefix, baseclass);
      } else if (GetFlag(n, "feature:exceptionclass")) {
	Append(s_phpclasses, "extends Exception ");
      }
      {
	Node *node = NewHash();
	Setattr(node, "type", Getattr(n, "name"));
	Setfile(node, Getfile(n));
	Setline(node, Getline(n));
	String * interfaces = Swig_typemap_lookup("phpinterfaces", node, "", 0);
	if (interfaces) {
	  Printf(s_phpclasses, "implements %s ", interfaces);
	}
	Delete(node);
      }
      Printf(s_phpclasses, "{\n\tpublic $%s=null;\n", SWIG_PTR);
      if (!baseclass) {
	// Only store this in the base class (NB !baseclass means we *are*
	// a base class...)
	Printf(s_phpclasses, "\tprotected $%s=array();\n", SWIG_DATA);
      }

      // Write property SET handlers
      ki = First(shadow_set_vars);
      if (ki.key) {
	// This class has setters.
	Printf(s_phpclasses, "\n\tfunction __set($var,$value) {\n");
	// FIXME: tune this threshold...
	if (Len(shadow_set_vars) <= 2) {
	  // Not many setters, so avoid call_user_func.
	  for (; ki.key; ki = Next(ki)) {
	    DOH *key = ki.key;
	    String *iname = ki.item;
	    Printf(s_phpclasses, "\t\tif ($var === '%s') return %s($this->%s,$value);\n", key, iname, SWIG_PTR);
	  }
	} else {
	  Printf(s_phpclasses, "\t\t$func = '%s_'.$var.'_set';\n", shadow_classname);
	  Printf(s_phpclasses, "\t\tif (function_exists($func)) return call_user_func($func,$this->%s,$value);\n", SWIG_PTR);
	}
	Printf(s_phpclasses, "\t\tif ($var === 'thisown') return swig_%s_alter_newobject($this->%s,$value);\n", module, SWIG_PTR);
	if (baseclass) {
	  Printf(s_phpclasses, "\t\t%s%s::__set($var,$value);\n", prefix, baseclass);
	} else {
	  Printf(s_phpclasses, "\t\t$this->%s[$var] = $value;\n", SWIG_DATA);
	}
	Printf(s_phpclasses, "\t}\n");
      } else {
	Printf(s_phpclasses, "\n\tfunction __set($var,$value) {\n");
	Printf(s_phpclasses, "\t\tif ($var === 'thisown') return swig_%s_alter_newobject($this->%s,$value);\n", module, SWIG_PTR);
	if (baseclass) {
	  Printf(s_phpclasses, "\t\t%s%s::__set($var,$value);\n", prefix, baseclass);
	} else {
	  Printf(s_phpclasses, "\t\t$this->%s[$var] = $value;\n", SWIG_DATA);
	}
	Printf(s_phpclasses, "\t}\n");
      }

      // Write property GET handlers
      ki = First(shadow_get_vars);
      if (ki.key) {
	// This class has getters.
	Printf(s_phpclasses, "\n\tfunction __get($var) {\n");
	int non_class_getters = 0;
	for (; ki.key; ki = Next(ki)) {
	  DOH *key = ki.key;
	  SwigType *d = ki.item;
	  if (!is_class(d)) {
	    ++non_class_getters;
	    continue;
	  }
	  Printv(s_phpclasses, "\t\tif ($var === '", key, "') return new ", prefix, Getattr(classLookup(d), "sym:name"), "(", shadow_classname, "_", key, "_get($this->", SWIG_PTR, "));\n", NIL);
	}
	// FIXME: tune this threshold...
	if (non_class_getters <= 2) {
	  // Not many non-class getters, so avoid call_user_func.
	  for (ki = First(shadow_get_vars); non_class_getters && ki.key;  ki = Next(ki)) {
	    DOH *key = ki.key;
	    SwigType *d = ki.item;
	    if (is_class(d)) continue;
	    Printv(s_phpclasses, "\t\tif ($var === '", key, "') return ", shadow_classname, "_", key, "_get($this->", SWIG_PTR, ");\n", NIL);
	    --non_class_getters;
	  }
	} else {
	  Printf(s_phpclasses, "\t\t$func = '%s_'.$var.'_get';\n", shadow_classname);
	  Printf(s_phpclasses, "\t\tif (function_exists($func)) return call_user_func($func,$this->%s);\n", SWIG_PTR);
	}
	Printf(s_phpclasses, "\t\tif ($var === 'thisown') return swig_%s_get_newobject($this->%s);\n", module, SWIG_PTR);
	if (baseclass) {
	  Printf(s_phpclasses, "\t\treturn %s%s::__get($var);\n", prefix, baseclass);
	} else {
	  // Reading an unknown property name gives null in PHP.
	  Printf(s_phpclasses, "\t\treturn $this->%s[$var];\n", SWIG_DATA);
	}
	Printf(s_phpclasses, "\t}\n");

	/* Create __isset for PHP 5.1 and later; PHP 5.0 will just ignore it. */
	/* __isset() should return true for read-only properties, so check for
	 * *_get() not *_set(). */
	Printf(s_phpclasses, "\n\tfunction __isset($var) {\n");
	Printf(s_phpclasses, "\t\tif (function_exists('%s_'.$var.'_get')) return true;\n", shadow_classname);
	Printf(s_phpclasses, "\t\tif ($var === 'thisown') return true;\n");
	if (baseclass) {
	  Printf(s_phpclasses, "\t\treturn %s%s::__isset($var);\n", prefix, baseclass);
	} else {
	  Printf(s_phpclasses, "\t\treturn array_key_exists($var, $this->%s);\n", SWIG_DATA);
	}
	Printf(s_phpclasses, "\t}\n");
      } else {
	Printf(s_phpclasses, "\n\tfunction __get($var) {\n");
	Printf(s_phpclasses, "\t\tif ($var === 'thisown') return swig_%s_get_newobject($this->%s);\n", module, SWIG_PTR);
	if (baseclass) {
	  Printf(s_phpclasses, "\t\treturn %s%s::__get($var);\n", prefix, baseclass);
	} else {
	  Printf(s_phpclasses, "\t\treturn $this->%s[$var];\n", SWIG_DATA);
	}
	Printf(s_phpclasses, "\t}\n");
	Printf(s_phpclasses, "\n\tfunction __isset($var) {\n");
	Printf(s_phpclasses, "\t\tif ($var === 'thisown') return true;\n");
	if (baseclass) {
	  Printf(s_phpclasses, "\t\treturn %s%s::__isset($var);\n", prefix, baseclass);
	} else {
	  Printf(s_phpclasses, "\t\treturn array_key_exists($var, $this->%s);\n", SWIG_DATA);
	}
	Printf(s_phpclasses, "\t}\n");
      }

      if (!class_has_ctor) {
	Printf(s_phpclasses, "\tfunction __construct($h) {\n");
	Printf(s_phpclasses, "\t\t$this->%s=$h;\n", SWIG_PTR);
	Printf(s_phpclasses, "\t}\n");
      }

      if (s_oowrappers) {
	Printf(s_phpclasses, "%s", s_oowrappers);
	Delete(s_oowrappers);
	s_oowrappers = NULL;
      }
      class_has_ctor = false;

      Printf(s_phpclasses, "}\n\n");

      Delete(shadow_classname);
      shadow_classname = NULL;

      Delete(shadow_set_vars);
      shadow_set_vars = NULL;
      Delete(shadow_get_vars);
      shadow_get_vars = NULL;
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int memberfunctionHandler(Node *n) {
    wrapperType = memberfn;
    Language::memberfunctionHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * membervariableHandler()
   * ------------------------------------------------------------ */

  virtual int membervariableHandler(Node *n) {
    wrapperType = membervar;
    Language::membervariableHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * staticmembervariableHandler()
   * ------------------------------------------------------------ */

  virtual int staticmembervariableHandler(Node *n) {
    wrapperType = staticmembervar;
    Language::staticmembervariableHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * staticmemberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int staticmemberfunctionHandler(Node *n) {
    wrapperType = staticmemberfn;
    Language::staticmemberfunctionHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  int abstractConstructorHandler(Node *) {
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constructorHandler()
   * ------------------------------------------------------------ */

  virtual int constructorHandler(Node *n) {
    constructors++;
    if (Swig_directorclass(n)) {
      String *name = GetChar(Swig_methodclass(n), "name");
      String *ctype = GetChar(Swig_methodclass(n), "classtype");
      String *sname = GetChar(Swig_methodclass(n), "sym:name");
      String *args = NewStringEmpty();
      ParmList *p = Getattr(n, "parms");
      int i;

      for (i = 0; p; p = nextSibling(p), i++) {
	if (i) {
	  Printf(args, ", ");
	}
	if (Strcmp(GetChar(p, "type"), SwigType_str(GetChar(p, "type"), 0))) {
	  SwigType *t = Getattr(p, "type");
	  Printf(args, "%s", SwigType_rcaststr(t, 0));
	  if (SwigType_isreference(t)) {
	    Append(args, "*");
	  }
	}
	Printf(args, "arg%d", i+1);
      }

      /* director ctor code is specific for each class */
      Delete(director_ctor_code);
      director_ctor_code = NewStringEmpty();
      director_prot_ctor_code = NewStringEmpty();
      Printf(director_ctor_code, "if (Z_TYPE_P(arg0) == IS_NULL) { /* not subclassed */\n");
      Printf(director_prot_ctor_code, "if (Z_TYPE_P(arg0) == IS_NULL) { /* not subclassed */\n");
      Printf(director_ctor_code, "  %s = (%s *)new %s(%s);\n", Swig_cresult_name(), ctype, ctype, args);
      Printf(director_prot_ctor_code, "  SWIG_PHP_Error(E_ERROR, \"accessing abstract class or protected constructor\");\n", name, name, args);
      if (i) {
	Insert(args, 0, ", ");
      }
      Printf(director_ctor_code, "} else {\n  %s = (%s *)new SwigDirector_%s(arg0%s);\n}\n", Swig_cresult_name(), ctype, sname, args);
      Printf(director_prot_ctor_code, "} else {\n  %s = (%s *)new SwigDirector_%s(arg0%s);\n}\n", Swig_cresult_name(), ctype, sname, args);
      Delete(args);

      wrapperType = directorconstructor;
    } else {
      wrapperType = constructor;
    }
    Language::constructorHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * CreateZendListDestructor()
   * ------------------------------------------------------------ */
  //virtual int destructorHandler(Node *n) {
  //}
  int CreateZendListDestructor(Node *n) {
    String *name = GetChar(Swig_methodclass(n), "name");
    String *iname = GetChar(n, "sym:name");
    ParmList *l = Getattr(n, "parms");

    String *destructorname = NewStringEmpty();
    Printf(destructorname, "_%s", Swig_name_wrapper(iname));
    Setattr(classnode, "destructor", destructorname);

    Wrapper *f = NewWrapper();
    Printf(f->def, "/* This function is designed to be called by the zend list destructors */\n");
    Printf(f->def, "/* to typecast and do the actual destruction */\n");
    Printf(f->def, "static void %s(zend_resource *res, const char *type_name) {\n", destructorname);

    Wrapper_add_localv(f, "value", "swig_object_wrapper *value=(swig_object_wrapper *) res->ptr", NIL);
    Wrapper_add_localv(f, "ptr", "void *ptr=value->ptr", NIL);
    Wrapper_add_localv(f, "newobject", "int newobject=value->newobject", NIL);

    emit_parameter_variables(l, f);
    emit_attach_parmmaps(l, f);

    // Get type of first arg, thing to be destructed
    // Skip ignored arguments
    Parm *p = l;
    //while (Getattr(p,"tmap:ignore")) {p = Getattr(p,"tmap:ignore:next");}
    while (checkAttribute(p, "tmap:in:numinputs", "0")) {
      p = Getattr(p, "tmap:in:next");
    }
    SwigType *pt = Getattr(p, "type");

    Printf(f->code, "  efree(value);\n");
    Printf(f->code, "  if (! newobject) return; /* can't delete it! */\n");
    Printf(f->code, "  arg1 = (%s)SWIG_ConvertResourceData(ptr, type_name, SWIGTYPE%s);\n", SwigType_lstr(pt, 0), SwigType_manglestr(pt));
    Printf(f->code, "  if (! arg1) zend_error(E_ERROR, \"%s resource already free'd\");\n", Char(name));

    Setattr(n, "wrap:name", destructorname);

    String *actioncode = emit_action(n);
    Append(f->code, actioncode);
    Delete(actioncode);

    Printf(f->code, "thrown:\n");
    Append(f->code, "return;\n");
    Append(f->code, "fail:\n");
    Append(f->code, "SWIG_FAIL();\n");
    Printf(f->code, "}\n");

    Wrapper_print(f, s_wrappers);
    DelWrapper(f);

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberconstantHandler()
   * ------------------------------------------------------------ */

  virtual int memberconstantHandler(Node *n) {
    wrapping_member_constant = Getattr(n, "sym:name");
    Language::memberconstantHandler(n);
    wrapping_member_constant = NULL;
    return SWIG_OK;
  }

  int classDirectorInit(Node *n) {
    String *declaration = Swig_director_declaration(n);
    Printf(f_directors_h, "%s\n", declaration);
    Printf(f_directors_h, "public:\n");
    Delete(declaration);
    return Language::classDirectorInit(n);
  }

  int classDirectorEnd(Node *n) {
    Printf(f_directors_h, "};\n");
    return Language::classDirectorEnd(n);
  }

  int classDirectorConstructor(Node *n) {
    Node *parent = Getattr(n, "parentNode");
    String *decl = Getattr(n, "decl");
    String *supername = Swig_class_name(parent);
    String *classname = NewStringEmpty();
    Printf(classname, "SwigDirector_%s", supername);

    /* insert self parameter */
    Parm *p;
    ParmList *superparms = Getattr(n, "parms");
    ParmList *parms = CopyParmList(superparms);
    String *type = NewString("zval");
    SwigType_add_pointer(type);
    p = NewParm(type, NewString("self"), n);
    set_nextSibling(p, parms);
    parms = p;

    if (!Getattr(n, "defaultargs")) {
      // There should always be a "self" parameter first.
      assert(ParmList_len(parms) > 0);

      /* constructor */
      {
	Wrapper *w = NewWrapper();
	String *call;
	String *basetype = Getattr(parent, "classtype");

	String *target = Swig_method_decl(0, decl, classname, parms, 0, 0);
	call = Swig_csuperclass_call(0, basetype, superparms);
	Printf(w->def, "%s::%s: %s, Swig::Director(self) {", classname, target, call);
	Append(w->def, "}");
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
    return Language::classDirectorConstructor(n);
  }

  int classDirectorMethod(Node *n, Node *parent, String *super) {
    int is_void = 0;
    int is_pointer = 0;
    String *decl = Getattr(n, "decl");
    String *returntype = Getattr(n, "type");
    String *name = Getattr(n, "name");
    String *classname = Getattr(parent, "sym:name");
    String *c_classname = Getattr(parent, "name");
    String *symname = Getattr(n, "sym:name");
    String *declaration = NewStringEmpty();
    ParmList *l = Getattr(n, "parms");
    Wrapper *w = NewWrapper();
    String *tm;
    String *wrap_args = NewStringEmpty();
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
    is_void = (Cmp(returntype, "void") == 0 && !is_pointer);

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
      /* attach typemaps to arguments (C/C++ -> PHP) */
      String *parse_args = NewStringEmpty();

      Swig_director_parms_fixup(l);

      /* remove the wrapper 'w' since it was producing spurious temps */
      Swig_typemap_attach_parms("in", l, 0);
      Swig_typemap_attach_parms("directorin", l, 0);
      Swig_typemap_attach_parms("directorargout", l, w);

      Parm *p;

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

	if (Getattr(p, "tmap:directorargout") != 0)
	  outputs++;

	String *pname = Getattr(p, "name");
	String *ptype = Getattr(p, "type");

	if ((tm = Getattr(p, "tmap:directorin")) != 0) {
	  String *parse = Getattr(p, "tmap:directorin:parse");
	  if (!parse) {
	    String *input = NewStringf("&args[%d]", idx++);
	    Setattr(p, "emit:directorinput", input);
	    Replaceall(tm, "$input", input);
	    Delete(input);
	    Replaceall(tm, "$owner", "0");
	    Printv(wrap_args, tm, "\n", NIL);
	    Putc('O', parse_args);
	  } else {
	    Append(parse_args, parse);
	    Setattr(p, "emit:directorinput", pname);
	    Replaceall(tm, "$input", pname);
	    Replaceall(tm, "$owner", "0");
	    if (Len(tm) == 0)
	      Append(tm, pname);
	  }
	  p = Getattr(p, "tmap:directorin:next");
	  continue;
	} else if (Cmp(ptype, "void")) {
	  Swig_warning(WARN_TYPEMAP_DIRECTORIN_UNDEF, input_file, line_number,
	      "Unable to use type %s as a function argument in director method %s::%s (skipping method).\n", SwigType_str(ptype, 0),
	      SwigType_namestr(c_classname), SwigType_namestr(name));
	  status = SWIG_NOWRAP;
	  break;
	}
	p = nextSibling(p);
      }

      /* exception handling */
      bool error_used_in_typemap = false;
      tm = Swig_typemap_lookup("director:except", n, Swig_cresult_name(), 0);
      if (!tm) {
	tm = Getattr(n, "feature:director:except");
	if (tm)
	  tm = Copy(tm);
      }
      if ((tm) && Len(tm) && (Strcmp(tm, "1") != 0)) {
	if (Replaceall(tm, "$error", "error")) {
	  /* Only declare error if it is used by the typemap. */
	  error_used_in_typemap = true;
	  Append(w->code, "int error;\n");
	}
      } else {
	Delete(tm);
	tm = NULL;
      }

      if (!idx) {
	Printf(w->code, "zval *args = NULL;\n");
      } else {
	Printf(w->code, "zval args[%d];\n", idx);
      }
      // typemap_directorout testcase requires that 0 can be assigned to the
      // variable named after the result of Swig_cresult_name(), so that can't
      // be a zval - make it a pointer to one instead.
      Printf(w->code, "zval swig_zval_result, swig_funcname;\n", Swig_cresult_name());
      Printf(w->code, "zval * SWIGUNUSED %s = &swig_zval_result;\n", Swig_cresult_name());
      const char * funcname = GetChar(n, "sym:name");
      Printf(w->code, "ZVAL_STRINGL(&swig_funcname, \"%s\", %d);\n", funcname, strlen(funcname));

      /* wrap complex arguments to zvals */
      Printv(w->code, wrap_args, NIL);

      if (error_used_in_typemap) {
	Append(w->code, "error = ");
      }
      Append(w->code, "call_user_function(EG(function_table), &swig_self, &swig_funcname,");
      Printf(w->code, " &swig_zval_result, %d, args);\n", idx);

      if (tm) {
	Printv(w->code, Str(tm), "\n", NIL);
	Delete(tm);
      }

      /* marshal return value from PHP to C/C++ type */

      String *cleanup = NewStringEmpty();
      String *outarg = NewStringEmpty();

      idx = 0;

      /* marshal return value */
      if (!is_void) {
	tm = Swig_typemap_lookup("directorout", n, Swig_cresult_name(), w);
	if (tm != 0) {
	  Replaceall(tm, "$input", Swig_cresult_name());
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
	      "Unable to use return type %s in director method %s::%s (skipping method).\n", SwigType_str(returntype, 0), SwigType_namestr(c_classname),
	      SwigType_namestr(name));
	  status = SWIG_ERROR;
	}
      }

      /* marshal outputs */
      for (p = l; p;) {
	if ((tm = Getattr(p, "tmap:directorargout")) != 0) {
	  Replaceall(tm, "$result", Swig_cresult_name());
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

    Append(w->code, "thrown:\n");
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
    } else {
      Append(w->code, "return;\n");
    }

    Append(w->code, "fail:\n");
    Append(w->code, "SWIG_FAIL();\n");
    Append(w->code, "}\n");

    // We expose protected methods via an extra public inline method which makes a straight call to the wrapped class' method
    String *inline_extra_method = NewStringEmpty();
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

  int classDirectorDisown(Node *) {
    return SWIG_OK;
  }
};				/* class PHP */

static PHP *maininstance = 0;

// We use this function to be able to write out zend_register_list_destructor_ex
// lines for most things in the type table
// NOTE: it's a function NOT A PHP::METHOD
extern "C" {
static void typetrace(const SwigType *ty, String *mangled, String *clientdata) {
  Node *class_node;
  if (!zend_types) {
    zend_types = NewHash();
  }
  // we want to know if the type which reduced to this has a constructor
  if ((class_node = maininstance->classLookup(ty))) {
    if (!Getattr(zend_types, mangled)) {
      // OK it may have been set before by a different SwigType but it would
      // have had the same underlying class node I think
      // - it is certainly required not to have different originating class
      // nodes for the same SwigType
      Setattr(zend_types, mangled, class_node);
    }
  } else {			// a non-class pointer
    Setattr(zend_types, mangled, NOTCLASS);
  }
  if (r_prevtracefunc)
    (*r_prevtracefunc) (ty, mangled, (String *) clientdata);
}
}

/* -----------------------------------------------------------------------------
 * new_swig_php()    - Instantiate module
 * ----------------------------------------------------------------------------- */

static Language *new_swig_php() {
  maininstance = new PHP;
  if (!r_prevtracefunc) {
    r_prevtracefunc = SwigType_remember_trace(typetrace);
  } else {
    Printf(stderr, "php Typetrace vector already saved!\n");
    assert(0);
  }
  return maininstance;
}

extern "C" Language *swig_php(void) {
  return new_swig_php();
}
