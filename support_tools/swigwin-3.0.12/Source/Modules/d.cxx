/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * d.cxx
 *
 * D language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"
#include <ctype.h>

// Hash type used for storing information about director callbacks for a class.
typedef DOH UpcallData;

class D : public Language {
  static const char *usage;
  const String *empty_string;
  const String *public_string;
  const String *protected_string;

  /*
   * Files and file sections containing C/C++ code.
   */
  File *f_begin;
  File *f_runtime;
  File *f_runtime_h;
  File *f_header;
  File *f_wrappers;
  File *f_init;
  File *f_directors;
  File *f_directors_h;
  List *filenames_list;

  /*
   * Command line-set modes of operation.
   */
  // Whether a single proxy D module is generated or classes and enums are
  // written to their own files.
  bool split_proxy_dmodule;

  // The major D version targeted (currently 1 or 2).
  unsigned short d_version;

  /*
   * State variables which indicate what is being wrapped at the moment.
   * This is probably not the most elegant way of handling state, but it has
   * proven to work in the C# and Java modules.
   */
  // Indicates if wrapping a native function.
  bool native_function_flag;

  // Indicates if wrapping a static functions or member variables
  bool static_flag;

  // Indicates if wrapping a nonstatic member variable
  bool variable_wrapper_flag;

  // Indicates if wrapping a member variable/enum/const.
  bool wrapping_member_flag;

  // Indicates if wrapping a global variable.
  bool global_variable_flag;

  // Name of a variable being wrapped.
  String *variable_name;

  /*
   * Variables temporarily holding the generated C++ code.
   */
  // C++ code for the generated wrapper functions for casts up the C++
  // for inheritance hierarchies.
  String *upcasts_code;

  // Function pointer typedefs for handling director callbacks on the C++ side.
  String *director_callback_typedefs;

  // Variables for storing the function pointers to the director callbacks on
  // the C++ side.
  String *director_callback_pointers;

  /*
   * Names of generated D entities.
   */
  // The name of the D module containing the interface to the C wrapper.
  String *im_dmodule_name;

  // The fully qualified name of the wrap D module (package name included).
  String *im_dmodule_fq_name;

  // The name of the proxy module which exposes the (SWIG) module contents as a
  // D module.
  String *proxy_dmodule_name;

  // The fully qualified name of the proxy D module.
  String *proxy_dmodule_fq_name;

  // Optional: Package the D modules are placed in (set via the -package
  // command line option).
  String *package;

  // The directory the generated D module files are written to. Is constructed
  // from the package path if a target package is set, points to the general
  // output directory otherwise.
  String *dmodule_directory;

  // The name of the library which contains the C wrapper (used when generating
  // the dynamic library loader). Can be overridden via the -wrapperlibrary
  // command line flag.
  String *wrap_library_name;

  /*
   * Variables temporarily holding the generated D code.
   */
  // Import statements written to the intermediary D module header set via
  // %pragma(d) imdmoduleimports.
  String *im_dmodule_imports;

  // The code for the intermediary D module body.
  String *im_dmodule_code;

  // Import statements for all proxy modules (the main proxy module and, if in
  // split proxy module mode, the proxy class modules) from
  // %pragma(d) globalproxyimports.
  String *global_proxy_imports;

  // The D code for the main proxy modules. nspace_proxy_dmodules is a hash from
  // the namespace name as key to an {"imports", "code"}. If the nspace feature
  // is not active, only proxy_dmodule_imports and proxy_dmodule_code are used,
  // which contain the code for the root proxy module.
  //
  // These variables should not be accessed directly but rather via the
  // proxy{Imports, Code}Buffer)() helper functions which return the right
  // buffer for a given namespace. If not in split proxy mode, they contain the
  // whole proxy code.
  String *proxy_dmodule_imports;
  String *proxy_dmodule_code;
  Hash *nspace_proxy_dmodules;

  // The D code generated for the currently processed enum.
  String *proxy_enum_code;

  /*
   * D data for the current proxy class.
   *
   * These strings are mainly used to temporarily accumulate code from the
   * various member handling functions while a single class is processed and are
   * no longer relevant once that class has been finished, i.e. after
   * classHandler() has returned.
   */
  // The unqualified name of the current proxy class.
  String *proxy_class_name;

  // The name of the current proxy class, qualified with the name of the
  // namespace it is in, if any.
  String *proxy_class_qname;

  // The import directives for the current proxy class. They are written to the
  // same D module the proxy class is written to.
  String *proxy_class_imports;

  // Code for enumerations nested in the current proxy class. Is emitted earlier
  // than the rest of the body to work around forward referencing-issues.
  String *proxy_class_enums_code;

  // The generated D code making up the body of the current proxy class.
  String *proxy_class_body_code;

  // D code which is emitted right after the proxy class.
  String *proxy_class_epilogue_code;

  // The full code for the current proxy class, including the epilogue.
  String* proxy_class_code;

  // Contains a D call to the function wrapping C++ the destructor of the
  // current class (if there is a public C++ destructor).
  String *destructor_call;

  // D code for the director callbacks generated for the current class.
  String *director_dcallbacks_code;

  /*
   * Code for dynamically loading the wrapper library on the D side.
   */
  // D code which is inserted into the im D module if dynamic linking is used.
  String *wrapper_loader_code;

  // The D code to bind a function pointer to a library symbol.
  String *wrapper_loader_bind_command;

  // The cumulated binding commands binding all the functions declared in the
  // intermediary D module to the C/C++ library symbols.
  String *wrapper_loader_bind_code;

  /*
   * Director data.
   */
  List *dmethods_seq;
  Hash *dmethods_table;
  int n_dmethods;
  int first_class_dmethod;
  int curr_class_dmethod;

  /*
   * SWIG types data.
   */
  // Collects information about encountered types SWIG does not know about (e.g.
  // incomplete types). This is used later to generate type wrapper proxy
  // classes for the unknown types.
  Hash *unknown_types;


public:
  /* ---------------------------------------------------------------------------
   * D::D()
   * --------------------------------------------------------------------------- */
   D():empty_string(NewString("")),
      public_string(NewString("public")),
      protected_string(NewString("protected")),
      f_begin(NULL),
      f_runtime(NULL),
      f_runtime_h(NULL),
      f_header(NULL),
      f_wrappers(NULL),
      f_init(NULL),
      f_directors(NULL),
      f_directors_h(NULL),
      filenames_list(NULL),
      split_proxy_dmodule(false),
      d_version(1),
      native_function_flag(false),
      static_flag(false),
      variable_wrapper_flag(false),
      wrapping_member_flag(false),
      global_variable_flag(false),
      variable_name(NULL),
      upcasts_code(NULL),
      director_callback_typedefs(NULL),
      director_callback_pointers(NULL),
      im_dmodule_name(NULL),
      im_dmodule_fq_name(NULL),
      proxy_dmodule_name(NULL),
      proxy_dmodule_fq_name(NULL),
      package(NULL),
      dmodule_directory(NULL),
      wrap_library_name(NULL),
      im_dmodule_imports(NULL),
      im_dmodule_code(NULL),
      global_proxy_imports(NULL),
      proxy_dmodule_imports(NULL),
      proxy_dmodule_code(NULL),
      nspace_proxy_dmodules(NULL),
      proxy_enum_code(NULL),
      proxy_class_name(NULL),
      proxy_class_qname(NULL),
      proxy_class_imports(NULL),
      proxy_class_enums_code(NULL),
      proxy_class_body_code(NULL),
      proxy_class_epilogue_code(NULL),
      proxy_class_code(NULL),
      destructor_call(NULL),
      director_dcallbacks_code(NULL),
      wrapper_loader_code(NULL),
      wrapper_loader_bind_command(NULL),
      wrapper_loader_bind_code(NULL),
      dmethods_seq(NULL),
      dmethods_table(NULL),
      n_dmethods(0),
      first_class_dmethod(0),
      curr_class_dmethod(0),
      unknown_types(NULL) {

    // For now, multiple inheritance with directors is not possible. It should be
    // easy to implement though.
    director_multiple_inheritance = 0;
    director_language = 1;

    // Not used:
    Delete(none_comparison);
    none_comparison = NewString("");
  }

  /* ---------------------------------------------------------------------------
   * D::main()
   * --------------------------------------------------------------------------- */
  virtual void main(int argc, char *argv[]) {
    SWIG_library_directory("d");

    // Look for certain command line options
    for (int i = 1; i < argc; i++) {
      if (argv[i]) {
	if ((strcmp(argv[i], "-d2") == 0)) {
      	  Swig_mark_arg(i);
      	  d_version = 2;
      	} else if (strcmp(argv[i], "-wrapperlibrary") == 0) {
	  if (argv[i + 1]) {
	    wrap_library_name = NewString("");
	    Printf(wrap_library_name, argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-package") == 0) {
	  if (argv[i + 1]) {
	    package = NewString("");
	    Printf(package, argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if ((strcmp(argv[i], "-splitproxy") == 0)) {
	  Swig_mark_arg(i);
	  split_proxy_dmodule = true;
	} else if (strcmp(argv[i], "-help") == 0) {
	  Printf(stdout, "%s\n", usage);
	}
      }
    }

    // Add a symbol to the parser for conditional compilation
    Preprocessor_define("SWIGD 1", 0);

    // Also make the target D version available as preprocessor symbol for
    // use in our library files.
    String *version_define = NewStringf("SWIG_D_VERSION %u", d_version);
    Preprocessor_define(version_define, 0);
    Delete(version_define);

    // Add typemap definitions
    SWIG_typemap_lang("d");
    SWIG_config_file("d.swg");

    allow_overloading();
  }

  /* ---------------------------------------------------------------------------
   * D::top()
   * --------------------------------------------------------------------------- */
  virtual int top(Node *n) {
    // Get any options set in the module directive
    Node *optionsnode = Getattr(Getattr(n, "module"), "options");

    if (optionsnode) {
      if (Getattr(optionsnode, "imdmodulename")) {
	im_dmodule_name = Copy(Getattr(optionsnode, "imdmodulename"));
      }

      if (Getattr(optionsnode, "directors")) {
	// Check if directors are enabled for this module. Note: This is a
	// "master switch", if it is not set, not director code will be emitted
	// at all. %feature("director") statements are also required to enable
	// directors for individual classes or methods.
	//
	// Use the »directors« attributte of the %module directive to enable
	// director generation (e.g. »%module(directors="1") modulename«).
	allow_directors();
      }

      if (Getattr(optionsnode, "dirprot")) {
	allow_dirprot();
      }

      allow_allprotected(GetFlag(optionsnode, "allprotected"));
    }

    /* Initialize all of the output files */
    String *outfile = Getattr(n, "outfile");
    String *outfile_h = Getattr(n, "outfile_h");

    if (!outfile) {
      Printf(stderr, "Unable to determine outfile\n");
      SWIG_exit(EXIT_FAILURE);
    }

    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      SWIG_exit(EXIT_FAILURE);
    }

    if (directorsEnabled()) {
      if (!outfile_h) {
	Printf(stderr, "Unable to determine outfile_h\n");
	SWIG_exit(EXIT_FAILURE);
      }
      f_runtime_h = NewFile(outfile_h, "w", SWIG_output_files());
      if (!f_runtime_h) {
	FileErrorDisplay(outfile_h);
	SWIG_exit(EXIT_FAILURE);
      }
    }

    f_runtime = NewString("");
    f_init = NewString("");
    f_header = NewString("");
    f_wrappers = NewString("");
    f_directors_h = NewString("");
    f_directors = NewString("");

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("header", f_header);
    Swig_register_filebyname("wrapper", f_wrappers);
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", f_init);
    Swig_register_filebyname("director", f_directors);
    Swig_register_filebyname("director_h", f_directors_h);

    unknown_types = NewHash();
    filenames_list = NewList();

    // Make the package name and the resulting module output path.
    if (package) {
      // Append a dot so we can prepend the package variable directly to the
      // module names in the rest of the code.
      Printv(package, ".", NIL);
    } else {
      // Write the generated D modules to the »root« package by default.
      package = NewString("");
    }

    dmodule_directory = Copy(SWIG_output_directory());
    if (Len(package) > 0) {
      String *package_directory = Copy(package);
      Replaceall(package_directory, ".", SWIG_FILE_DELIMITER);
      Printv(dmodule_directory, package_directory, NIL);
      Delete(package_directory);
    }

    // Make the wrap and proxy D module names.
    // The wrap module name can be set in the module directive.
    if (!im_dmodule_name) {
      im_dmodule_name = NewStringf("%s_im", Getattr(n, "name"));
    }
    im_dmodule_fq_name = NewStringf("%s%s", package, im_dmodule_name);
    proxy_dmodule_name = Copy(Getattr(n, "name"));
    proxy_dmodule_fq_name = NewStringf("%s%s", package, proxy_dmodule_name);

    im_dmodule_code = NewString("");
    proxy_class_imports = NewString("");
    proxy_class_enums_code = NewString("");
    proxy_class_body_code = NewString("");
    proxy_class_epilogue_code = NewString("");
    proxy_class_code = NewString("");
    destructor_call = NewString("");
    proxy_dmodule_code = NewString("");
    proxy_dmodule_imports = NewString("");
    nspace_proxy_dmodules = NewHash();
    im_dmodule_imports = NewString("");
    upcasts_code = NewString("");
    global_proxy_imports = NewString("");
    wrapper_loader_code = NewString("");
    wrapper_loader_bind_command = NewString("");
    wrapper_loader_bind_code = NewString("");
    dmethods_seq = NewList();
    dmethods_table = NewHash();
    n_dmethods = 0;

    // By default, expect the dynamically loaded wrapper library to be named
    // [lib]<module>_wrap[.so/.dll].
    if (!wrap_library_name)
      wrap_library_name = NewStringf("%s_wrap", Getattr(n, "name"));

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGD\n#define SWIGD\n#endif\n\n");

    if (directorsEnabled()) {
      Printf(f_runtime, "#define SWIG_DIRECTORS\n");

      /* Emit initial director header and director code: */
      Swig_banner(f_directors_h);
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "#ifndef SWIG_%s_WRAP_H_\n", proxy_dmodule_name);
      Printf(f_directors_h, "#define SWIG_%s_WRAP_H_\n\n", proxy_dmodule_name);

      Printf(f_directors, "\n\n");
      Printf(f_directors, "/* ---------------------------------------------------\n");
      Printf(f_directors, " * C++ director class methods\n");
      Printf(f_directors, " * --------------------------------------------------- */\n\n");
      if (outfile_h) {
	String *filename = Swig_file_filename(outfile_h);
	Printf(f_directors, "#include \"%s\"\n\n", filename);
	Delete(filename);
      }
    }

    Printf(f_runtime, "\n");

    Swig_name_register("wrapper", "D_%f");

    Printf(f_wrappers, "\n#ifdef __cplusplus\n");
    Printf(f_wrappers, "extern \"C\" {\n");
    Printf(f_wrappers, "#endif\n\n");

    // Emit all the wrapper code.
    Language::top(n);

    if (directorsEnabled()) {
      // Insert director runtime into the f_runtime file (before %header section).
      Swig_insert_file("director_common.swg", f_runtime);
      Swig_insert_file("director.swg", f_runtime);
    }

    // Generate the wrap D module.
    // TODO: Add support for »static« linking.
    {
      String *filen = NewStringf("%s%s.d", dmodule_directory, im_dmodule_name);
      File *im_d_file = NewFile(filen, "w", SWIG_output_files());
      if (!im_d_file) {
	FileErrorDisplay(filen);
	SWIG_exit(EXIT_FAILURE);
      }
      Append(filenames_list, Copy(filen));
      Delete(filen);
      filen = NULL;

      // Start writing out the intermediary class file.
      emitBanner(im_d_file);

      Printf(im_d_file, "module %s;\n", im_dmodule_fq_name);

      Printv(im_d_file, im_dmodule_imports, "\n", NIL);

      Replaceall(wrapper_loader_code, "$wraplibrary", wrap_library_name);
      Replaceall(wrapper_loader_code, "$wrapperloaderbindcode", wrapper_loader_bind_code);
      Replaceall(wrapper_loader_code, "$module", proxy_dmodule_name);
      Printf(im_d_file, "%s\n", wrapper_loader_code);

      // Add the wrapper function declarations.
      replaceModuleVariables(im_dmodule_code);
      Printv(im_d_file, im_dmodule_code, NIL);

      Delete(im_d_file);
    }

    // Generate the main D proxy module.
    {
      String *filen = NewStringf("%s%s.d", dmodule_directory, proxy_dmodule_name);
      File *proxy_d_file = NewFile(filen, "w", SWIG_output_files());
      if (!proxy_d_file) {
	FileErrorDisplay(filen);
	SWIG_exit(EXIT_FAILURE);
      }
      Append(filenames_list, Copy(filen));
      Delete(filen);
      filen = NULL;

      emitBanner(proxy_d_file);

      Printf(proxy_d_file, "module %s;\n", proxy_dmodule_fq_name);
      Printf(proxy_d_file, "\nstatic import %s;\n", im_dmodule_fq_name);
      Printv(proxy_d_file, global_proxy_imports, NIL);
      Printv(proxy_d_file, proxy_dmodule_imports, NIL);
      Printv(proxy_d_file, "\n", NIL);

      // Write a D type wrapper class for each SWIG type to the proxy module code.
      for (Iterator swig_type = First(unknown_types); swig_type.key; swig_type = Next(swig_type)) {
	writeTypeWrapperClass(swig_type.key, swig_type.item);
      }

      // Add the proxy functions (and classes, if they are not written to a separate file).
      replaceModuleVariables(proxy_dmodule_code);
      Printv(proxy_d_file, proxy_dmodule_code, NIL);

      Delete(proxy_d_file);
    }

    // Generate the additional proxy modules for nspace support.
    for (Iterator it = First(nspace_proxy_dmodules); it.key; it = Next(it)) {
      String *module_name = createLastNamespaceName(it.key);

      String *filename = NewStringf("%s%s.d", outputDirectory(it.key), module_name);
      File *file = NewFile(filename, "w", SWIG_output_files());
      if (!file) {
	FileErrorDisplay(filename);
	SWIG_exit(EXIT_FAILURE);
      }
      Delete(filename);

      emitBanner(file);

      Printf(file, "module %s%s.%s;\n", package, it.key, module_name);
      Printf(file, "\nstatic import %s;\n", im_dmodule_fq_name);
      Printv(file, global_proxy_imports, NIL);
      Printv(file, Getattr(it.item, "imports"), NIL);
      Printv(file, "\n", NIL);

      String *code = Getattr(it.item, "code");
      replaceModuleVariables(code);
      Printv(file, code, NIL);

      Delete(file);
      Delete(module_name);
    }

    if (upcasts_code)
      Printv(f_wrappers, upcasts_code, NIL);

    Printf(f_wrappers, "#ifdef __cplusplus\n");
    Printf(f_wrappers, "}\n");
    Printf(f_wrappers, "#endif\n");

    // Check for overwriting file problems on filesystems that are case insensitive
    Iterator it1;
    Iterator it2;
    for (it1 = First(filenames_list); it1.item; it1 = Next(it1)) {
      String *item1_lower = Swig_string_lower(it1.item);
      for (it2 = Next(it1); it2.item; it2 = Next(it2)) {
	String *item2_lower = Swig_string_lower(it2.item);
	if (it1.item && it2.item) {
	  if (Strcmp(item1_lower, item2_lower) == 0) {
	    Swig_warning(WARN_LANG_PORTABILITY_FILENAME, input_file, line_number,
			 "Portability warning: File %s will be overwritten by %s on case insensitive filesystems such as "
			 "Windows' FAT32 and NTFS unless the class/module name is renamed\n", it1.item, it2.item);
	  }
	}
	Delete(item2_lower);
      }
      Delete(item1_lower);
    }

    Delete(unknown_types);
    unknown_types = NULL;
    Delete(filenames_list);
    filenames_list = NULL;
    Delete(im_dmodule_name);
    im_dmodule_name = NULL;
    Delete(im_dmodule_fq_name);
    im_dmodule_fq_name = NULL;
    Delete(im_dmodule_code);
    im_dmodule_code = NULL;
    Delete(proxy_class_imports);
    proxy_class_imports = NULL;
    Delete(proxy_class_enums_code);
    proxy_class_enums_code = NULL;
    Delete(proxy_class_body_code);
    proxy_class_body_code = NULL;
    Delete(proxy_class_epilogue_code);
    proxy_class_epilogue_code = NULL;
    Delete(proxy_class_code);
    proxy_class_code = NULL;
    Delete(destructor_call);
    destructor_call = NULL;
    Delete(proxy_dmodule_name);
    proxy_dmodule_name = NULL;
    Delete(proxy_dmodule_fq_name);
    proxy_dmodule_fq_name = NULL;
    Delete(proxy_dmodule_code);
    proxy_dmodule_code = NULL;
    Delete(proxy_dmodule_imports);
    proxy_dmodule_imports = NULL;
    Delete(nspace_proxy_dmodules);
    nspace_proxy_dmodules = NULL;
    Delete(im_dmodule_imports);
    im_dmodule_imports = NULL;
    Delete(upcasts_code);
    upcasts_code = NULL;
    Delete(global_proxy_imports);
    global_proxy_imports = NULL;
    Delete(wrapper_loader_code);
    wrapper_loader_code = NULL;
    Delete(wrapper_loader_bind_code);
    wrapper_loader_bind_code = NULL;
    Delete(wrapper_loader_bind_command);
    wrapper_loader_bind_command = NULL;
    Delete(dmethods_seq);
    dmethods_seq = NULL;
    Delete(dmethods_table);
    dmethods_table = NULL;
    Delete(package);
    package = NULL;
    Delete(dmodule_directory);
    dmodule_directory = NULL;
    n_dmethods = 0;

    // Merge all the generated C/C++ code and close the output files.
    Dump(f_runtime, f_begin);
    Dump(f_header, f_begin);

    if (directorsEnabled()) {
      Dump(f_directors, f_begin);
      Dump(f_directors_h, f_runtime_h);

      Printf(f_runtime_h, "\n");
      Printf(f_runtime_h, "#endif\n");

      Delete(f_runtime_h);
      f_runtime_h = NULL;
      Delete(f_directors);
      f_directors = NULL;
      Delete(f_directors_h);
      f_directors_h = NULL;
    }

    Dump(f_wrappers, f_begin);
    Wrapper_pretty_print(f_init, f_begin);
    Delete(f_header);
    Delete(f_wrappers);
    Delete(f_init);
    Delete(f_runtime);
    Delete(f_begin);

    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::insertDirective()
   * --------------------------------------------------------------------------- */
  virtual int insertDirective(Node *n) {
    int ret = SWIG_OK;
    String *code = Getattr(n, "code");
    String *section = Getattr(n, "section");
    replaceModuleVariables(code);

    if (!ImportMode && (Cmp(section, "proxycode") == 0)) {
      if (proxy_class_body_code) {
	Swig_typemap_replace_embedded_typemap(code, n);
	Printv(proxy_class_body_code, code, NIL);
      }
    } else {
      ret = Language::insertDirective(n);
    }
    return ret;
  }

  /* ---------------------------------------------------------------------------
   * D::pragmaDirective()
   *
   * Valid Pragmas:
   * imdmodulecode      - text (D code) is copied verbatim to the wrap module
   * imdmoduleimports   - import statements for the im D module
   *
   * proxydmodulecode     - text (D code) is copied verbatim to the proxy module
   *                        (the main proxy module if in split proxy mode).
   * globalproxyimports   - import statements inserted into _all_ proxy modules.
   *
   * wrapperloadercode    - D code for loading the wrapper library (is copied to
   *                        the im D module).
   * wrapperloaderbindcommand - D code for binding a symbol from the wrapper
   *                        library to the declaration in the im D module.
   * --------------------------------------------------------------------------- */
  virtual int pragmaDirective(Node *n) {
    if (!ImportMode) {
      String *lang = Getattr(n, "lang");
      String *code = Getattr(n, "name");
      String *value = Getattr(n, "value");

      if (Strcmp(lang, "d") == 0) {
	String *strvalue = NewString(value);
	Replaceall(strvalue, "\\\"", "\"");

	if (Strcmp(code, "imdmodulecode") == 0) {
	  Printf(im_dmodule_code, "%s\n", strvalue);
	} else if (Strcmp(code, "imdmoduleimports") == 0) {
	  replaceImportTypeMacros(strvalue);
	  Chop(strvalue);
	  Printf(im_dmodule_imports, "%s\n", strvalue);
	} else if (Strcmp(code, "proxydmodulecode") == 0) {
	  Printf(proxyCodeBuffer(0), "%s\n", strvalue);
	} else if (Strcmp(code, "globalproxyimports") == 0) {
	  replaceImportTypeMacros(strvalue);
	  Chop(strvalue);
	  Printf(global_proxy_imports, "%s\n", strvalue);
	} else if (Strcmp(code, "wrapperloadercode") == 0) {
	  Delete(wrapper_loader_code);
	  wrapper_loader_code = Copy(strvalue);
	} else if (Strcmp(code, "wrapperloaderbindcommand") == 0) {
	  Delete(wrapper_loader_bind_command);
	  wrapper_loader_bind_command = Copy(strvalue);
	} else {
	  Swig_error(input_file, line_number, "Unrecognized pragma.\n");
	}
	Delete(strvalue);
      }
    }
    return Language::pragmaDirective(n);
  }

  /* ---------------------------------------------------------------------------
   * D::enumDeclaration()
   *
   * Wraps C/C++ enums as D enums.
   * --------------------------------------------------------------------------- */
  virtual int enumDeclaration(Node *n) {
    if (ImportMode)
      return SWIG_OK;

    if (getCurrentClass() && (cplus_mode != PUBLIC))
      return SWIG_NOWRAP;

    proxy_enum_code = NewString("");
    String *symname = Getattr(n, "sym:name");
    String *typemap_lookup_type = Getattr(n, "name");

    // Emit the enum declaration.
    if (typemap_lookup_type) {
      const String *enummodifiers = lookupCodeTypemap(n, "dclassmodifiers", typemap_lookup_type, WARN_D_TYPEMAP_CLASSMOD_UNDEF);
      Printv(proxy_enum_code, "\n", enummodifiers, " ", symname, " {\n", NIL);
    } else {
      // Handle anonymous enums.
      Printv(proxy_enum_code, "\nenum {\n", NIL);
    }

    // Emit each enum item.
    Language::enumDeclaration(n);

    if (GetFlag(n, "nonempty")) {
      // Finish the enum.
      if (typemap_lookup_type) {
	Printv(proxy_enum_code,
	  lookupCodeTypemap(n, "dcode", typemap_lookup_type, WARN_NONE), // Extra D code
	  "\n}\n", NIL);
      } else {
	// Handle anonymous enums.
	Printv(proxy_enum_code, "\n}\n", NIL);
      }
      Replaceall(proxy_enum_code, "$dclassname", symname);
    } else {
      // D enum declarations must have at least one member to be legal, so emit
      // an alias to int instead (their ctype/imtype is always int).
      Delete(proxy_enum_code);
      proxy_enum_code = NewStringf("\nalias int %s;\n", symname);
    }

    const String* imports =
      lookupCodeTypemap(n, "dimports", typemap_lookup_type, WARN_NONE);
    String* imports_trimmed;
    if (Len(imports) > 0) {
      imports_trimmed = Copy(imports);
      Chop(imports_trimmed);
      replaceImportTypeMacros(imports_trimmed);
      Printv(imports_trimmed, "\n", NIL);
    } else {
      imports_trimmed = NewString("");
    }

    if (is_wrapping_class()) {
      // Enums defined within the C++ class are written into the proxy
      // class.
      Printv(proxy_class_imports, imports_trimmed, NIL);
      Printv(proxy_class_enums_code, proxy_enum_code, NIL);
    } else {
      // Write non-anonymous enums to their own file if in split proxy module
      // mode.
      if (split_proxy_dmodule && typemap_lookup_type) {
	assertClassNameValidity(proxy_class_name);

	String *nspace = Getattr(n, "sym:nspace");
	String *output_directory = outputDirectory(nspace);
	String *filename = NewStringf("%s%s.d", output_directory, symname);
	Delete(output_directory);

	File *class_file = NewFile(filename, "w", SWIG_output_files());
	if (!class_file) {
	  FileErrorDisplay(filename);
	  SWIG_exit(EXIT_FAILURE);
	}
	Append(filenames_list, Copy(filename));
	Delete(filename);

	emitBanner(class_file);
	if (nspace) {
	  Printf(class_file, "module %s%s.%s;\n", package, nspace, symname);
	} else {
	  Printf(class_file, "module %s%s;\n", package, symname);
	}
	Printv(class_file, imports_trimmed, NIL);

	Printv(class_file, proxy_enum_code, NIL);

	Delete(class_file);
      } else {
	String *nspace = Getattr(n, "sym:nspace");
	Printv(proxyImportsBuffer(nspace), imports, NIL);
	Printv(proxyCodeBuffer(nspace), proxy_enum_code, NIL);
      }
    }

    Delete(imports_trimmed);

    Delete(proxy_enum_code);
    proxy_enum_code = NULL;
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::enumvalueDeclaration()
   * --------------------------------------------------------------------------- */
  virtual int enumvalueDeclaration(Node *n) {
    if (getCurrentClass() && (cplus_mode != PUBLIC))
      return SWIG_NOWRAP;

    Swig_require("enumvalueDeclaration", n, "*name", "?value", NIL);
    String *value = Getattr(n, "value");
    String *name = Getattr(n, "name");
    Node *parent = parentNode(n);
    String *tmpValue;

    // Strange hack from parent method.
    // RESEARCH: What is this doing?
    if (value)
      tmpValue = NewString(value);
    else
      tmpValue = NewString(name);
    // Note that this is used in enumValue() amongst other places
    Setattr(n, "value", tmpValue);

    // Deal with enum values that are not int
    int swigtype = SwigType_type(Getattr(n, "type"));
    if (swigtype == T_BOOL) {
      const char *val = Equal(Getattr(n, "enumvalue"), "true") ? "1" : "0";
      Setattr(n, "enumvalue", val);
    } else if (swigtype == T_CHAR) {
      String *val = NewStringf("'%(escape)s'", Getattr(n, "enumvalue"));
      Setattr(n, "enumvalue", val);
      Delete(val);
    }

    // Emit the enum item.
    {
      if (!GetFlag(n, "firstenumitem"))
	Printf(proxy_enum_code, ",\n");

      Printf(proxy_enum_code, "  %s", Getattr(n, "sym:name"));

      // Check for the %dconstvalue feature
      String *value = Getattr(n, "feature:d:constvalue");

      // Note that in D, enum values must be compile-time constants. Thus,
      // %dmanifestconst(0) (getting the enum values at runtime) is not supported.
      value = value ? value : Getattr(n, "enumvalue");
      if (value) {
	Printf(proxy_enum_code, " = %s", value);
      }

      // Keep track that the currently processed enum has at least one value.
      SetFlag(parent, "nonempty");
    }

    Delete(tmpValue);
    Swig_restore(n);
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::memberfunctionHandler()
   * --------------------------------------------------------------------------- */
  virtual int memberfunctionHandler(Node *n) {
    Language::memberfunctionHandler(n);

    String *overloaded_name = getOverloadedName(n);
    String *intermediary_function_name =
      Swig_name_member(getNSpace(), proxy_class_name, overloaded_name);
    Setattr(n, "imfuncname", intermediary_function_name);

    String *proxy_func_name = Getattr(n, "sym:name");
    Setattr(n, "proxyfuncname", proxy_func_name);
    if (split_proxy_dmodule &&
      Len(Getattr(n, "parms")) == 0 &&
      Strncmp(proxy_func_name, package, Len(proxy_func_name)) == 0) {
      // If we are in split proxy mode and the function is named like the
      // target package, the D compiler is unable to resolve the ambiguity
      // between the package name and an argument-less function call.
      // TODO: This might occur with nspace as well, augment the check.
      Swig_warning(WARN_D_NAME_COLLISION, input_file, line_number,
	"%s::%s might collide with the package name, consider using %%rename to resolve the ambiguity.\n",
	proxy_class_name, proxy_func_name);
    }

    writeProxyClassFunction(n);

    Delete(overloaded_name);

    // For each function, look if we have to alias in the parent class function
    // for the overload resolution process to work as expected from C++
    // (http://www.digitalmars.com/d/2.0/function.html#function-inheritance).
    // For multiple overloads, only emit the alias directive once (for the
    // last method, »sym:nextSibling« is null then).
    // Smart pointer classes do not mirror the inheritance hierarchy of the
    // underlying types, so aliasing the base class methods in is not required
    // for them.
    // DMD BUG: We have to emit the alias after the last function becasue
    // taking a delegate in the overload checking code fails otherwise
    // (http://d.puremagic.com/issues/show_bug.cgi?id=4860).
    if (!Getattr(n, "sym:nextSibling") && !is_smart_pointer() &&
	!areAllOverloadsOverridden(n)) {
      String *name = Getattr(n, "sym:name");
      Printf(proxy_class_body_code, "\nalias $dbaseclass.%s %s;\n", name, name);
    }
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::staticmemberfunctionHandler()
   * --------------------------------------------------------------------------- */
  virtual int staticmemberfunctionHandler(Node *n) {
    static_flag = true;

    Language::staticmemberfunctionHandler(n);

    String *overloaded_name = getOverloadedName(n);
    String *intermediary_function_name =
      Swig_name_member(getNSpace(), proxy_class_name, overloaded_name);
    Setattr(n, "proxyfuncname", Getattr(n, "sym:name"));
    Setattr(n, "imfuncname", intermediary_function_name);
    writeProxyClassFunction(n);
    Delete(overloaded_name);

    static_flag = false;
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::globalvariableHandler()
   * --------------------------------------------------------------------------- */
  virtual int globalvariableHandler(Node *n) {
    variable_name = Getattr(n, "sym:name");
    global_variable_flag = true;
    int ret = Language::globalvariableHandler(n);
    global_variable_flag = false;

    return ret;
  }

  /* ---------------------------------------------------------------------------
   * D::membervariableHandler()
   * --------------------------------------------------------------------------- */
  virtual int membervariableHandler(Node *n) {
    variable_name = Getattr(n, "sym:name");
    wrapping_member_flag = true;
    variable_wrapper_flag = true;
    Language::membervariableHandler(n);
    wrapping_member_flag = false;
    variable_wrapper_flag = false;

    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::staticmembervariableHandler()
   * --------------------------------------------------------------------------- */
  virtual int staticmembervariableHandler(Node *n) {
    if (GetFlag(n, "feature:d:manifestconst") != 1) {
      Delattr(n, "value");
    }

    variable_name = Getattr(n, "sym:name");
    wrapping_member_flag = true;
    static_flag = true;
    Language::staticmembervariableHandler(n);
    wrapping_member_flag = false;
    static_flag = false;

    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::memberconstantHandler()
   * --------------------------------------------------------------------------- */
  virtual int memberconstantHandler(Node *n) {
    variable_name = Getattr(n, "sym:name");
    wrapping_member_flag = true;
    Language::memberconstantHandler(n);
    wrapping_member_flag = false;
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::constructorHandler()
   * --------------------------------------------------------------------------- */
  virtual int constructorHandler(Node *n) {
    Language::constructorHandler(n);

    // Wrappers not wanted for some methods where the parameters cannot be overloadedprocess in D.
    if (Getattr(n, "overload:ignore")) {
      return SWIG_OK;
    }

    ParmList *l = Getattr(n, "parms");
    String *tm;
    String *proxy_constructor_code = NewString("");
    int i;

    // Holds code for the constructor helper method generated only when the din
    // typemap has code in the pre or post attributes.
    String *helper_code = NewString("");
    String *helper_args = NewString("");
    String *pre_code = NewString("");
    String *post_code = NewString("");
    String *terminator_code = NewString("");
    NewString("");

    String *overloaded_name = getOverloadedName(n);
    String *mangled_overname = Swig_name_construct(getNSpace(), overloaded_name);
    String *imcall = NewString("");

    const String *methodmods = Getattr(n, "feature:d:methodmodifiers");
    methodmods = methodmods ? methodmods : (is_public(n) ? public_string : protected_string);

    // Typemaps were attached earlier to the node, get the return type of the
    // call to the C++ constructor wrapper.
    const String *wrapper_return_type = lookupDTypemap(n, "imtype", true);

    String *imtypeout = Getattr(n, "tmap:imtype:out");
    if (imtypeout) {
      // The type in the imtype typemap's out attribute overrides the type in
      // the typemap itself.
      wrapper_return_type = imtypeout;
    }

    Printf(proxy_constructor_code, "\n%s this(", methodmods);
    Printf(helper_code, "static private %s SwigConstruct%s(",
      wrapper_return_type, proxy_class_name);

    Printv(imcall, im_dmodule_fq_name, ".", mangled_overname, "(", NIL);

    /* Attach the non-standard typemaps to the parameter list */
    Swig_typemap_attach_parms("in", l, NULL);
    Swig_typemap_attach_parms("dtype", l, NULL);
    Swig_typemap_attach_parms("din", l, NULL);

    emit_mark_varargs(l);

    int gencomma = 0;

    /* Output each parameter */
    Parm *p = l;
    for (i = 0; p; i++) {
      if (checkAttribute(p, "varargs:ignore", "1")) {
	// Skip ignored varargs.
	p = nextSibling(p);
	continue;
      }

      if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	// Skip ignored parameters.
	p = Getattr(p, "tmap:in:next");
	continue;
      }

      SwigType *pt = Getattr(p, "type");
      String *param_type = NewString("");

      // Get the D parameter type.
      if ((tm = lookupDTypemap(p, "dtype", true))) {
	const String *inattributes = Getattr(p, "tmap:dtype:inattributes");
	Printf(param_type, "%s%s", inattributes ? inattributes : empty_string, tm);
      } else {
	Swig_warning(WARN_D_TYPEMAP_DTYPE_UNDEF, input_file, line_number,
	  "No dtype typemap defined for %s\n", SwigType_str(pt, 0));
      }

      if (gencomma)
	Printf(imcall, ", ");

      String *arg = makeParameterName(n, p, i, false);
      String *parmtype = 0;

      // Get the D code to convert the parameter value to the type used in the
      // intermediary D module.
      if ((tm = lookupDTypemap(p, "din"))) {
	Replaceall(tm, "$dinput", arg);
	String *pre = Getattr(p, "tmap:din:pre");
	if (pre) {
	  replaceClassname(pre, pt);
	  Replaceall(pre, "$dinput", arg);
	  if (Len(pre_code) > 0)
	    Printf(pre_code, "\n");
	  Printv(pre_code, pre, NIL);
	}
	String *post = Getattr(p, "tmap:din:post");
	if (post) {
	  replaceClassname(post, pt);
	  Replaceall(post, "$dinput", arg);
	  if (Len(post_code) > 0)
	    Printf(post_code, "\n");
	  Printv(post_code, post, NIL);
	}
	String *terminator = Getattr(p, "tmap:din:terminator");
	if (terminator) {
	  replaceClassname(terminator, pt);
	  Replaceall(terminator, "$dinput", arg);
	  if (Len(terminator_code) > 0)
	    Insert(terminator_code, 0, "\n");
	  Insert(terminator_code, 0, terminator);
	}
	parmtype = Getattr(p, "tmap:din:parmtype");
	if (parmtype)
	  Replaceall(parmtype, "$dinput", arg);
	Printv(imcall, tm, NIL);
      } else {
	Swig_warning(WARN_D_TYPEMAP_DIN_UNDEF, input_file, line_number,
	  "No din typemap defined for %s\n", SwigType_str(pt, 0));
      }

      /* Add parameter to proxy function */
      if (gencomma) {
	Printf(proxy_constructor_code, ", ");
	Printf(helper_code, ", ");
	Printf(helper_args, ", ");
      }
      Printf(proxy_constructor_code, "%s %s", param_type, arg);
      Printf(helper_code, "%s %s", param_type, arg);
      Printf(helper_args, "%s", parmtype ? parmtype : arg);
      ++gencomma;

      Delete(parmtype);
      Delete(arg);
      Delete(param_type);
      p = Getattr(p, "tmap:in:next");
    }

    Printf(imcall, ")");

    Printf(proxy_constructor_code, ")");
    Printf(helper_code, ")");

    // Insert the dconstructor typemap (replacing $directorconnect as needed).
    Hash *attributes = NewHash();
    String *construct_tm = Copy(lookupCodeTypemap(n, "dconstructor",
      Getattr(n, "name"), WARN_D_TYPEMAP_DCONSTRUCTOR_UNDEF, attributes));
    if (construct_tm) {
      const bool use_director = (parentNode(n) && Swig_directorclass(n));
      if (!use_director) {
	Replaceall(construct_tm, "$directorconnect", "");
      } else {
	String *connect_attr = Getattr(attributes, "tmap:dconstructor:directorconnect");

	if (connect_attr) {
	  Replaceall(construct_tm, "$directorconnect", connect_attr);
	} else {
	  Swig_warning(WARN_D_NO_DIRECTORCONNECT_ATTR, input_file, line_number,
	    "\"directorconnect\" attribute missing in %s \"dconstructor\" typemap.\n",
	    Getattr(n, "name"));
	  Replaceall(construct_tm, "$directorconnect", "");
	}
      }

      Printv(proxy_constructor_code, " ", construct_tm, NIL);
    }

    replaceExcode(n, proxy_constructor_code, "dconstructor", attributes);

    bool is_pre_code = Len(pre_code) > 0;
    bool is_post_code = Len(post_code) > 0;
    bool is_terminator_code = Len(terminator_code) > 0;
    if (is_pre_code || is_post_code || is_terminator_code) {
      Printf(helper_code, " {\n");
      if (is_pre_code) {
	Printv(helper_code, pre_code, "\n", NIL);
      }
      if (is_post_code) {
	Printf(helper_code, "  try {\n");
	Printv(helper_code, "    return ", imcall, ";\n", NIL);
	Printv(helper_code, "  } finally {\n", post_code, "\n    }", NIL);
      } else {
	Printv(helper_code, "  return ", imcall, ";", NIL);
      }
      if (is_terminator_code) {
	Printv(helper_code, "\n", terminator_code, NIL);
      }
      Printf(helper_code, "\n}\n");
      String *helper_name = NewStringf("%s.SwigConstruct%s(%s)",
	proxy_class_name, proxy_class_name, helper_args);
      Replaceall(proxy_constructor_code, "$imcall", helper_name);
      Delete(helper_name);
    } else {
      Replaceall(proxy_constructor_code, "$imcall", imcall);
    }

    Printv(proxy_class_body_code, proxy_constructor_code, "\n", NIL);

    Delete(helper_args);
    Delete(pre_code);
    Delete(post_code);
    Delete(terminator_code);
    Delete(construct_tm);
    Delete(attributes);
    Delete(overloaded_name);
    Delete(imcall);

    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::destructorHandler()
   * --------------------------------------------------------------------------- */
  virtual int destructorHandler(Node *n) {
    Language::destructorHandler(n);
    String *symname = Getattr(n, "sym:name");
    Printv(destructor_call, im_dmodule_fq_name, ".", Swig_name_destroy(getNSpace(),symname), "(cast(void*)swigCPtr)", NIL);
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::classHandler()
   * --------------------------------------------------------------------------- */
  virtual int classHandler(Node *n) {
    String *nspace = getNSpace();
    File *class_file = NULL;

    proxy_class_name = Copy(Getattr(n, "sym:name"));
    if (nspace) {
      proxy_class_qname = NewStringf("%s.%s", nspace, proxy_class_name);
    } else {
      proxy_class_qname = Copy(proxy_class_name);
    }

    if (!addSymbol(proxy_class_name, n, nspace)) {
      return SWIG_ERROR;
    }

    assertClassNameValidity(proxy_class_name);

    if (split_proxy_dmodule) {
      String *output_directory = outputDirectory(nspace);
      String *filename = NewStringf("%s%s.d", output_directory, proxy_class_name);
      class_file = NewFile(filename, "w", SWIG_output_files());
      Delete(output_directory);
      if (!class_file) {
	FileErrorDisplay(filename);
	SWIG_exit(EXIT_FAILURE);
      }
      Append(filenames_list, Copy(filename));
      Delete(filename);

      emitBanner(class_file);
      if (nspace) {
        Printf(class_file, "module %s%s.%s;\n", package, nspace, proxy_class_name);
      } else {
        Printf(class_file, "module %s%s;\n", package, proxy_class_name);
      }
      Printf(class_file, "\nstatic import %s;\n", im_dmodule_fq_name);
    }

    Clear(proxy_class_imports);
    Clear(proxy_class_enums_code);
    Clear(proxy_class_body_code);
    Clear(proxy_class_epilogue_code);
    Clear(proxy_class_code);
    Clear(destructor_call);


    // Traverse the tree for this class, using the *Handler()s to generate code
    // to the proxy_class_* variables.
    Language::classHandler(n);


    writeProxyClassAndUpcasts(n);
    writeDirectorConnectWrapper(n);

    Replaceall(proxy_class_code, "$dclassname", proxy_class_name);

    String *dclazzname = Swig_name_member(getNSpace(), proxy_class_name, "");
    Replaceall(proxy_class_code, "$dclazzname", dclazzname);
    Delete(dclazzname);

    if (split_proxy_dmodule) {
      Printv(class_file, global_proxy_imports, NIL);
      Printv(class_file, proxy_class_imports, NIL);

      replaceModuleVariables(proxy_class_code);
      Printv(class_file, proxy_class_code, NIL);

      Delete(class_file);
    } else {
      Printv(proxyImportsBuffer(getNSpace()), proxy_class_imports, NIL);
      Printv(proxyCodeBuffer(getNSpace()), proxy_class_code, NIL);
    }

    Delete(proxy_class_qname);
    proxy_class_qname = NULL;
    Delete(proxy_class_name);
    proxy_class_name = NULL;

    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::constantWrapper()
   *
   * Used for wrapping constants declared by #define or %constant and also for
   * (primitive) static member constants initialised inline.
   *
   * If the %dmanifestconst feature is used, the C/C++ constant value is used to
   * initialize a D »const«. If not, a »getter« method is generated which
   * retrieves the value via a call to the C wrapper. However, if there is a
   * %dconstvalue specified, it overrides all other settings.
   * --------------------------------------------------------------------------- */
  virtual int constantWrapper(Node *n) {
    String *symname = Getattr(n, "sym:name");
    if (!addSymbol(symname, n))
      return SWIG_ERROR;

    // The %dmanifestconst feature determines if a D manifest constant
    // (const/enum) or a getter function is created.
    if (GetFlag(n, "feature:d:manifestconst") != 1) {
      // Default constant handling will work with any type of C constant. It
      // generates a getter function (which is the same as a read only property
      // in D) which retrieves the value via by calling the C wrapper.
      // Note that this is only called for global constants, static member
      // constants are already handled in staticmemberfunctionHandler().

      Swig_save("constantWrapper", n, "value", NIL);
      Swig_save("constantWrapper", n, "tmap:ctype:out", "tmap:imtype:out", "tmap:dtype:out", "tmap:out:null", "tmap:imtype:outattributes", "tmap:dtype:outattributes", NIL);

      // Add the stripped quotes back in.
      String *old_value = Getattr(n, "value");
      SwigType *t = Getattr(n, "type");
      if (SwigType_type(t) == T_STRING) {
	Setattr(n, "value", NewStringf("\"%s\"", old_value));
	Delete(old_value);
      } else if (SwigType_type(t) == T_CHAR) {
	Setattr(n, "value", NewStringf("\'%s\'", old_value));
	Delete(old_value);
      }

      SetFlag(n, "feature:immutable");
      int result = globalvariableHandler(n);

      Swig_restore(n);
      return result;
    }

    String *constants_code = NewString("");
    SwigType *t = Getattr(n, "type");
    SwigType *valuetype = Getattr(n, "valuetype");
    ParmList *l = Getattr(n, "parms");

    // Attach the non-standard typemaps to the parameter list.
    Swig_typemap_attach_parms("dtype", l, NULL);

    // Get D return type.
    String *return_type = NewString("");
    String *tm;
    if ((tm = lookupDTypemap(n, "dtype"))) {
      String *dtypeout = Getattr(n, "tmap:dtype:out");
      if (dtypeout) {
	// The type in the out attribute of the typemap overrides the type
	// in the dtype typemap.
	tm = dtypeout;
	replaceClassname(tm, t);
      }
      Printf(return_type, "%s", tm);
    } else {
      Swig_warning(WARN_D_TYPEMAP_DTYPE_UNDEF, input_file, line_number,
	"No dtype typemap defined for %s\n", SwigType_str(t, 0));
    }

    const String *itemname = wrapping_member_flag ? variable_name : symname;

    String *attributes = Getattr(n, "feature:d:methodmodifiers");
    if (attributes) {
      attributes = Copy(attributes);
    } else {
      attributes = Copy(is_public(n) ? public_string : protected_string);
    }

    if (d_version == 1) {
      if (static_flag) {
	Printv(attributes, " static", NIL);
      }
      Printf(constants_code, "\n%s const %s %s = ", attributes, return_type, itemname);
    } else {
      Printf(constants_code, "\n%s enum %s %s = ", attributes, return_type, itemname);
    }
    Delete(attributes);

    // Retrive the override value set via %dconstvalue, if any.
    String *override_value = Getattr(n, "feature:d:constvalue");
    if (override_value) {
      Printf(constants_code, "%s;\n", override_value);
    } else {
      // Just take the value from the C definition and hope it compiles in D.
      if (Getattr(n, "wrappedasconstant")) {
	if (SwigType_type(valuetype) == T_CHAR)
          Printf(constants_code, "\'%(escape)s\';\n", Getattr(n, "staticmembervariableHandler:value"));
	else
          Printf(constants_code, "%s;\n", Getattr(n, "staticmembervariableHandler:value"));
      } else {
	// Add the stripped quotes back in.
	String* value = Getattr(n, "value");
	if (SwigType_type(t) == T_STRING) {
	  Printf(constants_code, "\"%s\";\n", value);
	} else if (SwigType_type(t) == T_CHAR) {
	  Printf(constants_code, "\'%s\';\n", value);
	} else {
	  Printf(constants_code, "%s;\n", value);
	}
      }
    }

    // Emit the generated code to appropriate place.
    if (wrapping_member_flag) {
      Printv(proxy_class_body_code, constants_code, NIL);
    } else {
      Printv(proxyCodeBuffer(getNSpace()), constants_code, NIL);
    }

    // Cleanup.
    Delete(return_type);
    Delete(constants_code);

    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::functionWrapper()
   *
   * Generates the C wrapper code for a function and the corresponding
   * declaration in the wrap D module.
   * --------------------------------------------------------------------------- */
  virtual int functionWrapper(Node *n) {
    String *symname = Getattr(n, "sym:name");
    SwigType *t = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    String *tm;
    Parm *p;
    int i;
    String *c_return_type = NewString("");
    String *im_return_type = NewString("");
    String *cleanup = NewString("");
    String *outarg = NewString("");
    String *body = NewString("");
    int num_arguments = 0;
    bool is_void_return;
    String *overloaded_name = getOverloadedName(n);

    if (!Getattr(n, "sym:overloaded")) {
      if (!addSymbol(Getattr(n, "sym:name"), n))
	return SWIG_ERROR;
    }

    // A new wrapper function object
    Wrapper *f = NewWrapper();

    // Make a wrapper name for this function
    String *wname = Swig_name_wrapper(overloaded_name);

    /* Attach the non-standard typemaps to the parameter list. */
    Swig_typemap_attach_parms("ctype", l, f);
    Swig_typemap_attach_parms("imtype", l, f);

    /* Get return types */
    if ((tm = lookupDTypemap(n, "ctype"))) {
      String *ctypeout = Getattr(n, "tmap:ctype:out");
      if (ctypeout) {
	// The type in the ctype typemap's out attribute overrides the type in
	// the typemap itself.
	tm = ctypeout;
      }
      Printf(c_return_type, "%s", tm);
    } else {
      Swig_warning(WARN_D_TYPEMAP_CTYPE_UNDEF, input_file, line_number,
	"No ctype typemap defined for %s\n", SwigType_str(t, 0));
    }

    if ((tm = lookupDTypemap(n, "imtype"))) {
      String *imtypeout = Getattr(n, "tmap:imtype:out");
      if (imtypeout) {
	// The type in the imtype typemap's out attribute overrides the type in
	// the typemap itself.
	tm = imtypeout;
      }
      Printf(im_return_type, "%s", tm);
    } else {
      Swig_warning(WARN_D_TYPEMAP_IMTYPE_UNDEF, input_file, line_number, "No imtype typemap defined for %s\n", SwigType_str(t, 0));
    }

    is_void_return = (Cmp(c_return_type, "void") == 0);
    if (!is_void_return)
      Wrapper_add_localv(f, "jresult", c_return_type, "jresult", NIL);

    Printv(f->def, " SWIGEXPORT ", c_return_type, " ", wname, "(", NIL);

    // Emit all of the local variables for holding arguments.
    emit_parameter_variables(l, f);

    /* Attach the standard typemaps */
    emit_attach_parmmaps(l, f);

    // Parameter overloading
    Setattr(n, "wrap:parms", l);
    Setattr(n, "wrap:name", wname);

    // Wrappers not wanted for some methods where the parameters cannot be overloaded in D
    if (Getattr(n, "sym:overloaded")) {
      // Emit warnings for the few cases that can't be overloaded in D and give up on generating wrapper
      Swig_overload_check(n);
      if (Getattr(n, "overload:ignore")) {
	DelWrapper(f);
	return SWIG_OK;
      }
    }

    // Collect the parameter list for the intermediary D module declaration of
    // the generated wrapper function.
    String *im_dmodule_parameters = NewString("(");

    /* Get number of required and total arguments */
    num_arguments = emit_num_arguments(l);
    int gencomma = 0;

    // Now walk the function parameter list and generate code to get arguments
    for (i = 0, p = l; i < num_arguments; i++) {

      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");
      String *ln = Getattr(p, "lname");
      String *im_param_type = NewString("");
      String *c_param_type = NewString("");
      String *arg = NewString("");

      Printf(arg, "j%s", ln);

      /* Get the ctype types of the parameter */
      if ((tm = lookupDTypemap(p, "ctype", true))) {
	Printv(c_param_type, tm, NIL);
      } else {
	Swig_warning(WARN_D_TYPEMAP_CTYPE_UNDEF, input_file, line_number, "No ctype typemap defined for %s\n", SwigType_str(pt, 0));
      }

      /* Get the intermediary class parameter types of the parameter */
      if ((tm = lookupDTypemap(p, "imtype", true))) {
	const String *inattributes = Getattr(p, "tmap:imtype:inattributes");
	Printf(im_param_type, "%s%s", inattributes ? inattributes : empty_string, tm);
      } else {
	Swig_warning(WARN_D_TYPEMAP_IMTYPE_UNDEF, input_file, line_number, "No imtype typemap defined for %s\n", SwigType_str(pt, 0));
      }

      /* Add parameter to intermediary class method */
      if (gencomma)
	Printf(im_dmodule_parameters, ", ");
      Printf(im_dmodule_parameters, "%s %s", im_param_type, arg);

      // Add parameter to C function
      Printv(f->def, gencomma ? ", " : "", c_param_type, " ", arg, NIL);

      gencomma = 1;

      // Get typemap for this argument
      if ((tm = Getattr(p, "tmap:in"))) {
	canThrow(n, "in", p);
	Replaceall(tm, "$input", arg);
	Setattr(p, "emit:input", arg);
	Printf(f->code, "%s\n", tm);
	p = Getattr(p, "tmap:in:next");
      } else {
	Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
	p = nextSibling(p);
      }
      Delete(im_param_type);
      Delete(c_param_type);
      Delete(arg);
    }

    /* Insert constraint checking code */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:check"))) {
	canThrow(n, "check", p);
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(f->code, tm, "\n", NIL);
	p = Getattr(p, "tmap:check:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert cleanup code */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:freearg"))) {
	canThrow(n, "freearg", p);
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(cleanup, tm, "\n", NIL);
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert argument output code */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:argout"))) {
	canThrow(n, "argout", p);
	Replaceall(tm, "$result", "jresult");
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(outarg, tm, "\n", NIL);
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    // Look for usage of throws typemap and the canthrow flag
    ParmList *throw_parm_list = NULL;
    if ((throw_parm_list = Getattr(n, "catchlist"))) {
      Swig_typemap_attach_parms("throws", throw_parm_list, f);
      for (p = throw_parm_list; p; p = nextSibling(p)) {
	if (Getattr(p, "tmap:throws")) {
	  canThrow(n, "throws", p);
	}
      }
    }

    String *null_attribute = 0;
    // Now write code to make the function call
    if (!native_function_flag) {

      Swig_director_emit_dynamic_cast(n, f);
      String *actioncode = emit_action(n);

      /* Return value if necessary  */
      if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
	canThrow(n, "out", n);
	Replaceall(tm, "$result", "jresult");

	if (GetFlag(n, "feature:new"))
	  Replaceall(tm, "$owner", "1");
	else
	  Replaceall(tm, "$owner", "0");

	Printf(f->code, "%s", tm);
	null_attribute = Getattr(n, "tmap:out:null");
	if (Len(tm))
	  Printf(f->code, "\n");
      } else {
	Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(t, 0), Getattr(n, "name"));
      }
      emit_return_variable(n, t, f);
    }

    /* Output argument output code */
    Printv(f->code, outarg, NIL);

    /* Output cleanup code */
    Printv(f->code, cleanup, NIL);

    /* Look to see if there is any newfree cleanup code */
    if (GetFlag(n, "feature:new")) {
      if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
	canThrow(n, "newfree", n);
	Printf(f->code, "%s\n", tm);
      }
    }

    /* See if there is any return cleanup code */
    if (!native_function_flag) {
      if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
	canThrow(n, "ret", n);
	Printf(f->code, "%s\n", tm);
      }
    }

    // Complete D im parameter list and emit the declaration/binding code.
    Printv(im_dmodule_parameters, ")", NIL);
    writeImDModuleFunction(overloaded_name, im_return_type,
      im_dmodule_parameters, wname);
    Delete(im_dmodule_parameters);

    // Finish C function header.
    Printf(f->def, ") {");

    if (!is_void_return)
      Printv(f->code, "    return jresult;\n", NIL);
    Printf(f->code, "}\n");

    /* Substitute the cleanup code */
    Replaceall(f->code, "$cleanup", cleanup);

    /* Substitute the function name */
    Replaceall(f->code, "$symname", symname);

    /* Contract macro modification */
    if (Replaceall(f->code, "SWIG_contract_assert(", "SWIG_contract_assert($null, ") > 0) {
      Setattr(n, "d:canthrow", "1");
    }

    if (!null_attribute)
      Replaceall(f->code, "$null", "0");
    else
      Replaceall(f->code, "$null", null_attribute);

    /* Dump the function out */
    if (!native_function_flag) {
      Wrapper_print(f, f_wrappers);

      // Handle %exception which sets the canthrow attribute.
      if (Getattr(n, "feature:except:canthrow")) {
	Setattr(n, "d:canthrow", "1");
      }

      // A very simple check (it is not foolproof) to assist typemap writers
      // with setting the correct features when the want to throw D exceptions
      // from C++ code. It checks for the common methods which set
      // a pending D exception and issues a warning if one of them has been found
      // in the typemap, but the »canthrow« attribute/feature is not set.
      if (!Getattr(n, "d:canthrow")) {
	if (Strstr(f->code, "SWIG_exception")) {
	  Swig_warning(WARN_D_CANTHROW_MISSING, input_file, line_number,
	  "C code contains a call to SWIG_exception and D code does not handle pending exceptions via the canthrow attribute.\n");
	} else if (Strstr(f->code, "SWIG_DSetPendingException")) {
	  Swig_warning(WARN_D_CANTHROW_MISSING, input_file, line_number,
	  "C code contains a call to a SWIG_DSetPendingException method and D code does not handle pending exceptions via the canthrow attribute.\n");
	}
      }
    }

    // If we are not processing an enum or constant, and we were not generating
    // a wrapper function which will be accessed via a proxy class, write a
    // function to the proxy D module.
    if (!is_wrapping_class()) {
      writeProxyDModuleFunction(n);
    }

    // If we are processing a public member variable, write the property-style
    // member function to the proxy class.
    if (wrapping_member_flag) {
      Setattr(n, "proxyfuncname", variable_name);
      Setattr(n, "imfuncname", symname);

      writeProxyClassFunction(n);
    }

    Delete(c_return_type);
    Delete(im_return_type);
    Delete(cleanup);
    Delete(outarg);
    Delete(body);
    Delete(overloaded_name);
    DelWrapper(f);
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::nativeWrapper()
   * --------------------------------------------------------------------------- */
  virtual int nativeWrapper(Node *n) {
    String *wrapname = Getattr(n, "wrap:name");

    if (!addSymbol(wrapname, n))
      return SWIG_ERROR;

    if (Getattr(n, "type")) {
      Swig_save("nativeWrapper", n, "name", NIL);
      Setattr(n, "name", wrapname);
      native_function_flag = true;
      functionWrapper(n);
      Swig_restore(n);
      native_function_flag = false;
    } else {
      Swig_error(input_file, line_number, "No return type for %%native method %s.\n", Getattr(n, "wrap:name"));
    }

    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::classDirector()
   * --------------------------------------------------------------------------- */
  virtual int classDirector(Node *n) {
    String *nspace = Getattr(n, "sym:nspace");
    proxy_class_name = NewString(Getattr(n, "sym:name"));
    if (nspace) {
      proxy_class_qname = NewStringf("%s.%s", nspace, proxy_class_name);
    } else {
      proxy_class_qname = Copy(proxy_class_name);
    }

    int success = Language::classDirector(n);

    Delete(proxy_class_qname);
    proxy_class_qname = NULL;
    Delete(proxy_class_name);
    proxy_class_name = NULL;

    return success;
  }


  /* ---------------------------------------------------------------------------
   * D::classDirectorInit()
   * --------------------------------------------------------------------------- */
  virtual int classDirectorInit(Node *n) {
    Delete(director_ctor_code);
    director_ctor_code = NewString("$director_new");

    // Write C++ director class declaration, for example:
    // class SwigDirector_myclass : public myclass, public Swig::Director {
    String *classname = Swig_class_name(n);
    String *directorname = directorClassName(n);
    String *declaration = Swig_class_declaration(n, directorname);
    const String *base = Getattr(n, "classtype");

    Printf(f_directors_h,
      "%s : public %s, public Swig::Director {\n", declaration, base);
    Printf(f_directors_h, "\npublic:\n");

    Delete(declaration);
    Delete(directorname);
    Delete(classname);

    // Stash for later.
    Setattr(n, "director:ctor", NewString("Swig::Director()"));

    // Keep track of the director methods for this class.
    first_class_dmethod = curr_class_dmethod = n_dmethods;

    director_callback_typedefs = NewString("");
    director_callback_pointers = NewString("");
    director_dcallbacks_code = NewString("");

    return Language::classDirectorInit(n);
  }

  /* ---------------------------------------------------------------------------
   * D::classDirectorMethod()
   *
   * Emit a virtual director method to pass a method call on to the
   * underlying D object.
   * --------------------------------------------------------------------------- */
  virtual int classDirectorMethod(Node *n, Node *parent, String *super) {
    String *classname = Getattr(parent, "sym:name");
    String *c_classname = Getattr(parent, "name");
    String *name = Getattr(n, "name");
    String *symname = Getattr(n, "sym:name");
    SwigType *returntype = Getattr(n, "type");
    String *overloaded_name = getOverloadedName(n);
    String *storage = Getattr(n, "storage");
    String *value = Getattr(n, "value");
    String *decl = Getattr(n, "decl");
    String *declaration = NewString("");
    String *tm;
    Parm *p;
    int i;
    Wrapper *w = NewWrapper();
    ParmList *l = Getattr(n, "parms");
    bool is_void = !(Cmp(returntype, "void"));
    String *qualified_return = 0;
    bool pure_virtual = (!(Cmp(storage, "virtual")) && !(Cmp(value, "0")));
    int status = SWIG_OK;
    bool output_director = true;
    String *dirclassname = directorClassName(parent);
    String *qualified_name = NewStringf("%s::%s", dirclassname, name);
    SwigType *c_ret_type = NULL;
    String *dcallback_call_args = NewString("");
    String *imclass_dmethod;
    String *callback_typedef_parms = NewString("");
    String *delegate_parms = NewString("");
    String *proxy_method_param_list = NewString("");
    String *proxy_callback_return_type = NewString("");
    String *callback_def = NewString("");
    String *callback_code = NewString("");
    String *imcall_args = NewString("");
    bool ignored_method = GetFlag(n, "feature:ignore") ? true : false;

    // Kludge Alert: functionWrapper sets sym:overload properly, but it
    // isn't at this point, so we have to manufacture it ourselves. At least
    // we're consistent with the sym:overload name in functionWrapper. (?? when
    // does the overloaded method name get set?)

    imclass_dmethod = NewStringf("SwigDirector_%s", Swig_name_member(getNSpace(), classname, overloaded_name));

    qualified_return = SwigType_rcaststr(returntype, "c_result");

    if (!is_void && !ignored_method) {
      if (!SwigType_isclass(returntype)) {
	if (!(SwigType_ispointer(returntype) || SwigType_isreference(returntype))) {
	  String *construct_result = NewStringf("= SwigValueInit< %s >()", SwigType_lstr(returntype, 0));
	  Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), construct_result, NIL);
	  Delete(construct_result);
	} else {
	  String *base_typename = SwigType_base(returntype);
	  String *resolved_typename = SwigType_typedef_resolve_all(base_typename);
	  Symtab *symtab = Getattr(n, "sym:symtab");
	  Node *typenode = Swig_symbol_clookup(resolved_typename, symtab);

	  if (SwigType_ispointer(returntype) || (typenode && Getattr(typenode, "abstracts"))) {
	    /* initialize pointers to something sane. Same for abstract
	       classes when a reference is returned. */
	    Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), "= 0", NIL);
	  } else {
	    /* If returning a reference, initialize the pointer to a sane
	       default - if a D exception occurs, then the pointer returns
	       something other than a NULL-initialized reference. */
	    String *non_ref_type = Copy(returntype);

	    /* Remove reference and const qualifiers */
	    Replaceall(non_ref_type, "r.", "");
	    Replaceall(non_ref_type, "q(const).", "");
	    Wrapper_add_localv(w, "result_default", "static", SwigType_str(non_ref_type, "result_default"), "=", SwigType_str(non_ref_type, "()"), NIL);
	    Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), "= &result_default", NIL);

	    Delete(non_ref_type);
	  }

	  Delete(base_typename);
	  Delete(resolved_typename);
	}
      } else {
	SwigType *vt;

	vt = cplus_value_type(returntype);
	if (!vt) {
	  Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), NIL);
	} else {
	  Wrapper_add_localv(w, "c_result", SwigType_lstr(vt, "c_result"), NIL);
	  Delete(vt);
	}
      }
    }

    /* Create the intermediate class wrapper */
    tm = lookupDTypemap(n, "imtype");
    if (tm) {
      String *imtypeout = Getattr(n, "tmap:imtype:out");
      if (imtypeout) {
	// The type in the imtype typemap's out attribute overrides the type
	// in the typemap.
	tm = imtypeout;
      }
      Printf(callback_def, "\nprivate extern(C) %s swigDirectorCallback_%s_%s(void* dObject", tm, classname, overloaded_name);
      Printv(proxy_callback_return_type, tm, NIL);
    } else {
      Swig_warning(WARN_D_TYPEMAP_IMTYPE_UNDEF, input_file, line_number,
	"No imtype typemap defined for %s\n", SwigType_str(returntype, 0));
    }

    if ((c_ret_type = Swig_typemap_lookup("ctype", n, "", 0))) {
      if (!is_void && !ignored_method) {
	String *jretval_decl = NewStringf("%s jresult", c_ret_type);
	Wrapper_add_localv(w, "jresult", jretval_decl, "= 0", NIL);
	Delete(jretval_decl);
      }
    } else {
      Swig_warning(WARN_D_TYPEMAP_CTYPE_UNDEF, input_file, line_number,
	"No ctype typemap defined for %s for use in %s::%s (skipping director method)\n",
	SwigType_str(returntype, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
      output_director = false;
    }

    Swig_director_parms_fixup(l);

    // Attach the standard typemaps.
    Swig_typemap_attach_parms("out", l, 0);
    Swig_typemap_attach_parms("ctype", l, 0);
    Swig_typemap_attach_parms("imtype", l, 0);
    Swig_typemap_attach_parms("dtype", l, 0);
    Swig_typemap_attach_parms("directorin", l, 0);
    Swig_typemap_attach_parms("ddirectorin", l, 0);
    Swig_typemap_attach_parms("directorargout", l, w);

    // Preamble code.
    if (!ignored_method)
      Printf(w->code, "if (!swig_callback_%s) {\n", overloaded_name);

    if (!pure_virtual) {
      String *super_call = Swig_method_call(super, l);
      if (is_void) {
	Printf(w->code, "%s;\n", super_call);
	if (!ignored_method)
	  Printf(w->code, "return;\n");
      } else {
	Printf(w->code, "return %s;\n", super_call);
      }
      Delete(super_call);
    } else {
      Printf(w->code, " throw Swig::DirectorPureVirtualException(\"%s::%s\");\n", SwigType_namestr(c_classname), SwigType_namestr(name));
    }

    if (!ignored_method)
      Printf(w->code, "} else {\n");

    // Go through argument list.
    for (i = 0, p = l; p; ++i) {
      /* Is this superfluous? */
      while (checkAttribute(p, "tmap:directorin:numinputs", "0")) {
	p = Getattr(p, "tmap:directorin:next");
      }

      SwigType *pt = Getattr(p, "type");
      String *ln = makeParameterName(n, p, i, false);
      String *c_param_type = NULL;
      String *c_decl = NewString("");
      String *arg = NewString("");

      Printf(arg, "j%s", ln);

      // Add each parameter to the D callback invocation arguments.
      Printf(dcallback_call_args, ", %s", arg);

      /* Get parameter's intermediary C type */
      if ((c_param_type = lookupDTypemap(p, "ctype", true))) {
	String *ctypeout = Getattr(p, "tmap:ctype:out");
	if (ctypeout) {
	  // The type in the ctype typemap's out attribute overrides the type
	  // in the typemap itself.
	  c_param_type = ctypeout;
	}

	/* Add to local variables */
	Printf(c_decl, "%s %s", c_param_type, arg);
	if (!ignored_method)
	  Wrapper_add_localv(w, arg, c_decl, (!(SwigType_ispointer(pt) || SwigType_isreference(pt)) ? "" : "= 0"), NIL);

	/* Add input marshalling code */
	if ((tm = Getattr(p, "tmap:directorin"))) {

	  Setattr(p, "emit:directorinput", arg);
	  Replaceall(tm, "$input", arg);
	  Replaceall(tm, "$owner", "0");

	  if (Len(tm))
	    if (!ignored_method)
	      Printf(w->code, "%s\n", tm);

	  // Add parameter type to the C typedef for the D callback function.
	  Printf(callback_typedef_parms, ", %s", c_param_type);

	  /* Add parameter to the intermediate class code if generating the
	   * intermediate's upcall code */
	  if ((tm = lookupDTypemap(p, "imtype", true))) {
	    String *imtypeout = Getattr(p, "tmap:imtype:out");
	    if (imtypeout) {
	      // The type in the imtype typemap's out attribute overrides the
	      // type in the typemap itself.
	      tm = imtypeout;
	    }
	    const String *im_directorinattributes = Getattr(p, "tmap:imtype:directorinattributes");

	    // TODO: Is this copy really needed?
	    String *din = Copy(lookupDTypemap(p, "ddirectorin", true));

	    if (din) {
	      Replaceall(din, "$winput", ln);

	      Printf(delegate_parms, ", ");
	      if (i > 0) {
		Printf(proxy_method_param_list, ", ");
		Printf(imcall_args, ", ");
	      }
	      Printf(delegate_parms, "%s%s %s", im_directorinattributes ? im_directorinattributes : empty_string, tm, ln);

	      if (Cmp(din, ln)) {
		Printv(imcall_args, din, NIL);
	      } else {
		Printv(imcall_args, ln, NIL);
	      }

	      Delete(din);

	      // Get the parameter type in the proxy D class (used later when
	      // generating the overload checking code for the directorConnect
	      // function).
	      if ((tm = lookupDTypemap(p, "dtype", true))) {
		Printf(proxy_method_param_list, "%s", tm);
	      } else {
		Swig_warning(WARN_D_TYPEMAP_DTYPE_UNDEF, input_file, line_number,
	          "No dtype typemap defined for %s\n", SwigType_str(pt, 0));
	      }
	    } else {
	      Swig_warning(WARN_D_TYPEMAP_DDIRECTORIN_UNDEF, input_file, line_number,
	        "No ddirectorin typemap defined for %s for use in %s::%s (skipping director method)\n",
		SwigType_str(pt, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	      output_director = false;
	    }
	  } else {
	    Swig_warning(WARN_D_TYPEMAP_IMTYPE_UNDEF, input_file, line_number,
	      "No imtype typemap defined for %s for use in %s::%s (skipping director method)\n",
	      SwigType_str(pt, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	    output_director = false;
	  }

	  p = Getattr(p, "tmap:directorin:next");
	} else {
	  Swig_warning(WARN_D_TYPEMAP_DDIRECTORIN_UNDEF, input_file, line_number,
	    "No or improper directorin typemap defined for argument %s for use in %s::%s (skipping director method)\n",
	    SwigType_str(pt, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	  p = nextSibling(p);
	  output_director = false;
	}
      } else {
	Swig_warning(WARN_D_TYPEMAP_CTYPE_UNDEF, input_file, line_number,
	  "No ctype typemap defined for %s for use in %s::%s (skipping director method)\n",
	  SwigType_str(pt, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	output_director = false;
	p = nextSibling(p);
      }

      Delete(arg);
      Delete(c_decl);
      Delete(c_param_type);
      Delete(ln);
    }

    /* header declaration, start wrapper definition */
    String *target;
    SwigType *rtype = Getattr(n, "conversion_operator") ? 0 : Getattr(n, "classDirectorMethods:type");
    target = Swig_method_decl(rtype, decl, qualified_name, l, 0, 0);
    Printf(w->def, "%s", target);
    Delete(qualified_name);
    Delete(target);
    target = Swig_method_decl(rtype, decl, name, l, 0, 1);
    Printf(declaration, "    virtual %s", target);
    Delete(target);

    // Add any exception specifications to the methods in the director class
    ParmList *throw_parm_list = NULL;
    if ((throw_parm_list = Getattr(n, "throws")) || Getattr(n, "throw")) {
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
	  Printf(w->def, "%s", SwigType_str(Getattr(p, "type"), 0));
	  Printf(declaration, "%s", SwigType_str(Getattr(p, "type"), 0));
	}
      }

      Append(w->def, ")");
      Append(declaration, ")");
    }

    Append(w->def, " {");
    Append(declaration, ";\n");

    // Finish the callback function declaraction.
    Printf(callback_def, "%s)", delegate_parms);
    Printf(callback_def, " {\n");

    /* Emit the intermediate class's upcall to the actual class */

    String *upcall = NewStringf("(cast(%s)dObject).%s(%s)", classname, symname, imcall_args);

    if (!is_void) {
      if ((tm = lookupDTypemap(n, "ddirectorout"))) {
	Replaceall(tm, "$dcall", upcall);
	Printf(callback_code, "  return %s;\n", tm);
      }
    } else {
      Printf(callback_code, "  %s;\n", upcall);
    }

    Printf(callback_code, "}\n");
    Delete(upcall);

    if (!ignored_method) {
      if (!is_void)
	Printf(w->code, "jresult = (%s) ", c_ret_type);

      Printf(w->code, "swig_callback_%s(d_object%s);\n", overloaded_name, dcallback_call_args);

      if (!is_void) {
	String *jresult_str = NewString("jresult");
	String *result_str = NewString("c_result");

	/* Copy jresult into c_result... */
	if ((tm = Swig_typemap_lookup("directorout", n, result_str, w))) {
	  Replaceall(tm, "$input", jresult_str);
	  Replaceall(tm, "$result", result_str);
	  Printf(w->code, "%s\n", tm);
	} else {
	  Swig_warning(WARN_TYPEMAP_DIRECTOROUT_UNDEF, input_file, line_number,
		       "Unable to use return type %s used in %s::%s (skipping director method)\n",
		       SwigType_str(returntype, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	  output_director = false;
	}

	Delete(jresult_str);
	Delete(result_str);
      }

      /* Marshal outputs */
      for (p = l; p;) {
	if ((tm = Getattr(p, "tmap:directorargout"))) {
	  canThrow(n, "directorargout", p);
	  Replaceall(tm, "$result", "jresult");
	  Replaceall(tm, "$input", Getattr(p, "emit:directorinput"));
	  Printv(w->code, tm, "\n", NIL);
	  p = Getattr(p, "tmap:directorargout:next");
	} else {
	  p = nextSibling(p);
	}
      }

      /* Terminate wrapper code */
      Printf(w->code, "}\n");
      if (!is_void)
	Printf(w->code, "return %s;", qualified_return);
    }

    Printf(w->code, "}");

    // We expose virtual protected methods via an extra public inline method which makes a straight call to the wrapped class' method
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
    if (status == SWIG_OK && output_director) {
      if (!is_void) {
	Replaceall(w->code, "$null", qualified_return);
      } else {
	Replaceall(w->code, "$null", "");
      }
      if (!ignored_method)
	Printv(director_dcallbacks_code, callback_def, callback_code, NIL);
      if (!Getattr(n, "defaultargs")) {
	Replaceall(w->code, "$symname", symname);
	Wrapper_print(w, f_directors);
	Printv(f_directors_h, declaration, NIL);
	Printv(f_directors_h, inline_extra_method, NIL);
      }
    }

    if (!ignored_method) {
      // Register the upcall method so that the callback registering code can
      // be written later.

      // We cannot directly use n here because its »type« attribute does not
      // the full return type any longer after Language::functionHandler has
      // returned.
      String *dp_return_type = lookupDTypemap(n, "dtype");
      if (dp_return_type) {
	String *dtypeout = Getattr(n, "tmap:dtype:out");
	if (dtypeout) {
	  // The type in the dtype typemap's out attribute overrides the type
	  // in the typemap itself.
	  dp_return_type = dtypeout;
  	replaceClassname(dp_return_type, returntype);
	}
      } else {
	Swig_warning(WARN_D_TYPEMAP_DTYPE_UNDEF, input_file, line_number,
	  "No dtype typemap defined for %s\n", SwigType_str(returntype, 0));
	dp_return_type = NewString("");
      }

      UpcallData *udata = addUpcallMethod(imclass_dmethod, symname, decl, overloaded_name, dp_return_type, proxy_method_param_list);
      Delete(dp_return_type);

      // Write the global callback function pointer on the C code.
      String *methid = Getattr(udata, "class_methodidx");

      Printf(director_callback_typedefs, "    typedef %s (* SWIG_Callback%s_t)", c_ret_type, methid);
      Printf(director_callback_typedefs, "(void *dobj%s);\n", callback_typedef_parms);
      Printf(director_callback_pointers, "    SWIG_Callback%s_t swig_callback_%s;\n", methid, overloaded_name);

      // Write the type alias for the callback to the intermediary D module.
      String *proxy_callback_type = NewString("");
      String *dirClassName = directorClassName(parent);
      Printf(proxy_callback_type, "%s_Callback%s", dirClassName, methid);
      Printf(im_dmodule_code, "alias extern(C) %s function(void*%s) %s;\n", proxy_callback_return_type, delegate_parms, proxy_callback_type);
      Delete(proxy_callback_type);
      Delete(dirClassName);
    }

    Delete(qualified_return);
    Delete(c_ret_type);
    Delete(declaration);
    Delete(callback_typedef_parms);
    Delete(delegate_parms);
    Delete(proxy_method_param_list);
    Delete(callback_def);
    Delete(callback_code);
    DelWrapper(w);

    return status;
  }

  /* ---------------------------------------------------------------------------
   * D::classDirectorConstructor()
   * --------------------------------------------------------------------------- */
  virtual int classDirectorConstructor(Node *n) {
    Node *parent = parentNode(n);
    String *decl = Getattr(n, "decl");;
    String *supername = Swig_class_name(parent);
    String *dirclassname = directorClassName(parent);
    String *sub = NewString("");
    Parm *p;
    ParmList *superparms = Getattr(n, "parms");
    ParmList *parms;
    int argidx = 0;

    /* Assign arguments to superclass's parameters, if not already done */
    for (p = superparms; p; p = nextSibling(p)) {
      String *pname = Getattr(p, "name");

      if (!pname) {
	pname = NewStringf("arg%d", argidx++);
	Setattr(p, "name", pname);
      }
    }

    // TODO: Is this copy needed?
    parms = CopyParmList(superparms);

    if (!Getattr(n, "defaultargs")) {
      /* constructor */
      {
	String *basetype = Getattr(parent, "classtype");
	String *target = Swig_method_decl(0, decl, dirclassname, parms, 0, 0);
	String *call = Swig_csuperclass_call(0, basetype, superparms);
	String *classtype = SwigType_namestr(Getattr(n, "name"));

	Printf(f_directors, "%s::%s : %s, %s {\n", dirclassname, target, call, Getattr(parent, "director:ctor"));
	Printf(f_directors, "  swig_init_callbacks();\n");
	Printf(f_directors, "}\n\n");

	Delete(classtype);
	Delete(target);
	Delete(call);
      }

      /* constructor header */
      {
	String *target = Swig_method_decl(0, decl, dirclassname, parms, 0, 1);
	Printf(f_directors_h, "    %s;\n", target);
	Delete(target);
      }
    }

    Delete(sub);
    Delete(supername);
    Delete(parms);
    Delete(dirclassname);
    return Language::classDirectorConstructor(n);
  }

  /* ---------------------------------------------------------------------------
   * D::classDirectorDefaultConstructor()
   * --------------------------------------------------------------------------- */
  virtual int classDirectorDefaultConstructor(Node *n) {
    String *dirclassname = directorClassName(n);
    String *classtype = SwigType_namestr(Getattr(n, "name"));
    Wrapper *w = NewWrapper();

    Printf(w->def, "%s::%s() : %s {", dirclassname, dirclassname, Getattr(n, "director:ctor"));
    Printf(w->code, "}\n");
    Wrapper_print(w, f_directors);

    Printf(f_directors_h, "    %s();\n", dirclassname);
    DelWrapper(w);
    Delete(classtype);
    Delete(dirclassname);
    return Language::classDirectorDefaultConstructor(n);
  }

  /* ---------------------------------------------------------------------------
   * D::classDirectorDestructor()
   * --------------------------------------------------------------------------- */
  virtual int classDirectorDestructor(Node *n) {
    Node *current_class = getCurrentClass();
    String *dirclassname = directorClassName(current_class);
    Wrapper *w = NewWrapper();

    if (Getattr(n, "throw")) {
      Printf(f_directors_h, "    virtual ~%s() throw ();\n", dirclassname);
      Printf(w->def, "%s::~%s() throw () {\n", dirclassname, dirclassname);
    } else {
      Printf(f_directors_h, "    virtual ~%s();\n", dirclassname);
      Printf(w->def, "%s::~%s() {\n", dirclassname, dirclassname);
    }

    Printv(w->code, "}\n", NIL);

    Wrapper_print(w, f_directors);

    DelWrapper(w);
    Delete(dirclassname);
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::classDirectorEnd()
   * --------------------------------------------------------------------------- */
  virtual int classDirectorEnd(Node *n) {
    int i;
    String *director_classname = directorClassName(n);

    Wrapper *w = NewWrapper();

    if (Len(director_callback_typedefs) > 0) {
      Printf(f_directors_h, "\n%s", director_callback_typedefs);
    }

    Printf(f_directors_h, "    void swig_connect_director(void* dobj");

    Printf(w->def, "void %s::swig_connect_director(void* dobj", director_classname);
    Printf(w->code, "d_object = dobj;");

    for (i = first_class_dmethod; i < curr_class_dmethod; ++i) {
      UpcallData *udata = Getitem(dmethods_seq, i);
      String *methid = Getattr(udata, "class_methodidx");
      String *overname = Getattr(udata, "overname");

      Printf(f_directors_h, ", SWIG_Callback%s_t callback%s", methid, overname);
      Printf(w->def, ", SWIG_Callback%s_t callback_%s", methid, overname);
      Printf(w->code, "swig_callback_%s = callback_%s;\n", overname, overname);
    }

    Printf(f_directors_h, ");\n");
    Printf(w->def, ") {");

    Printf(f_directors_h, "\nprivate:\n");
    Printf(f_directors_h, "    void swig_init_callbacks();\n");
    Printf(f_directors_h, "    void *d_object;\n");
    if (Len(director_callback_pointers) > 0) {
      Printf(f_directors_h, "%s", director_callback_pointers);
    }
    Printf(f_directors_h, "};\n\n");
    Printf(w->code, "}\n\n");

    Printf(w->code, "void %s::swig_init_callbacks() {\n", director_classname);
    for (i = first_class_dmethod; i < curr_class_dmethod; ++i) {
      UpcallData *udata = Getitem(dmethods_seq, i);
      String *overname = Getattr(udata, "overname");
      Printf(w->code, "swig_callback_%s = 0;\n", overname);
    }
    Printf(w->code, "}");

    Wrapper_print(w, f_directors);

    DelWrapper(w);

    return Language::classDirectorEnd(n);
  }

  /* ---------------------------------------------------------------------------
   * D::classDirectorDisown()
   * --------------------------------------------------------------------------- */
  virtual int classDirectorDisown(Node *n) {
    (void) n;
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------------
   * D::replaceSpecialVariables()
   * --------------------------------------------------------------------------- */
  virtual void replaceSpecialVariables(String *method, String *tm, Parm *parm) {
    (void)method;
    SwigType *type = Getattr(parm, "type");

    // Just assume that this goes to the proxy class, we cannot know.
    replaceClassname(tm, type);
  }

protected:
  /* ---------------------------------------------------------------------------
   * D::extraDirectorProtectedCPPMethodsRequired()
   * --------------------------------------------------------------------------- */
  virtual bool extraDirectorProtectedCPPMethodsRequired() const {
    return false;
  }

private:
  /* ---------------------------------------------------------------------------
   * D::writeImDModuleFunction()
   *
   * Writes a function declaration for the given (C) wrapper function to the
   * intermediary D module.
   *
   * d_name - The name the function in the intermediary D module will get.
   * return type - The return type of the function in the C wrapper.
   * parameters - The parameter list of the C wrapper function.
   * wrapper_function_name - The name of the exported function in the C wrapper
   *                         (usually d_name prefixed by »D_«).
   * --------------------------------------------------------------------------- */
  void writeImDModuleFunction(const_String_or_char_ptr d_name,
    const_String_or_char_ptr return_type, const_String_or_char_ptr parameters,
    const_String_or_char_ptr wrapper_function_name) {

    // TODO: Add support for static linking here.
    Printf(im_dmodule_code, "SwigExternC!(%s function%s) %s;\n", return_type,
      parameters, d_name);
    Printv(wrapper_loader_bind_code, wrapper_loader_bind_command, NIL);
    Replaceall(wrapper_loader_bind_code, "$function", d_name);
    Replaceall(wrapper_loader_bind_code, "$symbol", wrapper_function_name);
  }

  /* ---------------------------------------------------------------------------
   * D::writeProxyClassFunction()
   *
   * Creates a D proxy function for a C++ function in the wrapped class. Used
   * for both static and non-static C++ class functions.
   *
   * The Node must contain two extra attributes.
   *  - "proxyfuncname": The name of the D proxy function.
   *  - "imfuncname": The corresponding function in the intermediary D module.
   * --------------------------------------------------------------------------- */
  void writeProxyClassFunction(Node *n) {
    SwigType *t = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    String *intermediary_function_name = Getattr(n, "imfuncname");
    String *proxy_function_name = Getattr(n, "proxyfuncname");
    String *tm;
    Parm *p;
    int i;
    String *imcall = NewString("");
    String *return_type = NewString("");
    String *function_code = NewString("");
    bool setter_flag = false;
    String *pre_code = NewString("");
    String *post_code = NewString("");
    String *terminator_code = NewString("");

    // Wrappers not wanted for some methods where the parameters cannot be
    // overloaded in D.
    if (Getattr(n, "overload:ignore"))
      return;

    // Don't generate proxy method for additional explicitcall method used in
    // directors.
    if (GetFlag(n, "explicitcall"))
      return;

    // RESEARCH: What is this good for?
    if (l) {
      if (SwigType_type(Getattr(l, "type")) == T_VOID) {
	l = nextSibling(l);
      }
    }

    /* Attach the non-standard typemaps to the parameter list */
    Swig_typemap_attach_parms("in", l, NULL);
    Swig_typemap_attach_parms("dtype", l, NULL);
    Swig_typemap_attach_parms("din", l, NULL);

    // Get return types.
    if ((tm = lookupDTypemap(n, "dtype"))) {
      String *dtypeout = Getattr(n, "tmap:dtype:out");
      if (dtypeout) {
	// The type in the dtype typemap's out attribute overrides the type in
	// the typemap.
	tm = dtypeout;
        replaceClassname(tm, t);
      }
      Printf(return_type, "%s", tm);
    } else {
      Swig_warning(WARN_D_TYPEMAP_DTYPE_UNDEF, input_file, line_number,
	"No dtype typemap defined for %s\n", SwigType_str(t, 0));
    }

    if (wrapping_member_flag) {
      // Check if this is a setter method for a public member.
      const String *setter_name = Swig_name_set(getNSpace(),
        Swig_name_member(0, proxy_class_name, variable_name));

      if (Cmp(Getattr(n, "sym:name"), setter_name) == 0) {
        setter_flag = true;
      }
    }

    // Write function modifiers.
    {
      String *modifiers;

      const String *mods_override = Getattr(n, "feature:d:methodmodifiers");
      if (mods_override) {
	modifiers = Copy(mods_override);
      } else {
	modifiers = Copy(is_public(n) ? public_string : protected_string);

	if (Getattr(n, "override")) {
	  Printf(modifiers, " override");
	}
      }

      if (is_smart_pointer()) {
	// Smart pointer classes do not mirror the inheritance hierarchy of the
	// underlying pointer type, so no override required.
	Replaceall(modifiers, "override", "");
      }

      Chop(modifiers);

      if (static_flag) {
	Printf(modifiers, " static");
      }

      Printf(function_code, "%s ", modifiers);
      Delete(modifiers);
    }

    // Complete the function declaration up to the parameter list.
    Printf(function_code, "%s %s(", return_type, proxy_function_name);

    // Write the wrapper function call up to the parameter list.
    Printv(imcall, im_dmodule_fq_name, ".$imfuncname(", NIL);
    if (!static_flag) {
      Printf(imcall, "cast(void*)swigCPtr");
    }

    String *proxy_param_types = NewString("");

    // Write the parameter list for the proxy function declaration and the
    // wrapper function call.
    emit_mark_varargs(l);
    int gencomma = !static_flag;
    for (i = 0, p = l; p; i++) {
      // Ignored varargs.
      if (checkAttribute(p, "varargs:ignore", "1")) {
	p = nextSibling(p);
	continue;
      }

      // Ignored parameters.
      if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
	continue;
      }

      // Ignore the 'this' argument for variable wrappers.
      if (!(variable_wrapper_flag && i == 0)) {
	String *param_name = makeParameterName(n, p, i, setter_flag);
	SwigType *pt = Getattr(p, "type");

	// Write the wrapper function call argument.
	{
	  if (gencomma) {
	    Printf(imcall, ", ");
	  }

	  if ((tm = lookupDTypemap(p, "din", true))) {
	    Replaceall(tm, "$dinput", param_name);
	    String *pre = Getattr(p, "tmap:din:pre");
	    if (pre) {
	      replaceClassname(pre, pt);
	      Replaceall(pre, "$dinput", param_name);
	      if (Len(pre_code) > 0)
		Printf(pre_code, "\n");
	      Printv(pre_code, pre, NIL);
	    }
	    String *post = Getattr(p, "tmap:din:post");
	    if (post) {
	      replaceClassname(post, pt);
	      Replaceall(post, "$dinput", param_name);
	      if (Len(post_code) > 0)
		Printf(post_code, "\n");
	      Printv(post_code, post, NIL);
	    }
	    String *terminator = Getattr(p, "tmap:din:terminator");
	    if (terminator) {
	      replaceClassname(terminator, pt);
	      Replaceall(terminator, "$dinput", param_name);
	      if (Len(terminator_code) > 0)
		Insert(terminator_code, 0, "\n");
	      Insert(terminator_code, 0, terminator);
	    }
	    Printv(imcall, tm, NIL);
	  } else {
	    Swig_warning(WARN_D_TYPEMAP_DIN_UNDEF, input_file, line_number,
	      "No din typemap defined for %s\n", SwigType_str(pt, 0));
	  }
	}

	// Write the D proxy function parameter.
	{
	  String *proxy_type = NewString("");

	  if ((tm = lookupDTypemap(p, "dtype"))) {
	    const String *inattributes = Getattr(p, "tmap:dtype:inattributes");
	    Printf(proxy_type, "%s%s", inattributes ? inattributes : empty_string, tm);
	  } else {
	    Swig_warning(WARN_D_TYPEMAP_DTYPE_UNDEF, input_file, line_number,
	      "No dtype typemap defined for %s\n", SwigType_str(pt, 0));
	  }

	  if (gencomma >= 2) {
	    Printf(function_code, ", ");
	    Printf(proxy_param_types, ", ");
	  }
	  gencomma = 2;
	  Printf(function_code, "%s %s", proxy_type, param_name);
	  Append(proxy_param_types, proxy_type);

	  Delete(proxy_type);
	}

	Delete(param_name);
      }
      p = Getattr(p, "tmap:in:next");
    }

    Printf(imcall, ")");
    Printf(function_code, ") ");

    if (d_version > 1 && wrapping_member_flag) {
      Printf(function_code, "@property ");
    }

    if (wrapMemberFunctionAsDConst(n)) {
      Printf(function_code, "const ");
    }

    // Lookup the code used to convert the wrapper return value to the proxy
    // function return type.
    if ((tm = lookupDTypemap(n, "dout"))) {
      replaceExcode(n, tm, "dout", n);
      bool is_pre_code = Len(pre_code) > 0;
      bool is_post_code = Len(post_code) > 0;
      bool is_terminator_code = Len(terminator_code) > 0;
      if (is_pre_code || is_post_code || is_terminator_code) {
	if (is_post_code) {
	  Insert(tm, 0, "\n  try ");
	  Printv(tm, " finally {\n", post_code, "\n  }", NIL);
	} else {
	  Insert(tm, 0, "\n  ");
	}
	if (is_pre_code) {
	  Insert(tm, 0, pre_code);
	  Insert(tm, 0, "\n");
	}
	if (is_terminator_code) {
	  Printv(tm, "\n", terminator_code, NIL);
	}
	Insert(tm, 0, "{");
	Printv(tm, "}", NIL);
      }
      if (GetFlag(n, "feature:new"))
	Replaceall(tm, "$owner", "true");
      else
	Replaceall(tm, "$owner", "false");
      replaceClassname(tm, t);

      // For director methods: generate code to selectively make a normal
      // polymorphic call or an explicit method call. Needed to prevent infinite
      // recursion when calling director methods.
      Node *explicit_n = Getattr(n, "explicitcallnode");
      if (explicit_n && Swig_directorclass(getCurrentClass())) {
	String *ex_overloaded_name = getOverloadedName(explicit_n);
	String *ex_intermediary_function_name = Swig_name_member(getNSpace(), proxy_class_name, ex_overloaded_name);

	String *ex_imcall = Copy(imcall);
	Replaceall(ex_imcall, "$imfuncname", ex_intermediary_function_name);
	Replaceall(imcall, "$imfuncname", intermediary_function_name);

	String *excode = NewString("");
	if (!Cmp(return_type, "void"))
	  Printf(excode, "if (swigIsMethodOverridden!(%s delegate(%s), %s function(%s), %s)()) %s; else %s",
	    return_type, proxy_param_types, return_type, proxy_param_types, proxy_function_name, ex_imcall, imcall);
	else
	  Printf(excode, "((swigIsMethodOverridden!(%s delegate(%s), %s function(%s), %s)()) ? %s : %s)",
	    return_type, proxy_param_types, return_type, proxy_param_types, proxy_function_name, ex_imcall, imcall);

	Clear(imcall);
	Printv(imcall, excode, NIL);
	Delete(ex_overloaded_name);
	Delete(excode);
      } else {
	Replaceall(imcall, "$imfuncname", intermediary_function_name);
      }
      Replaceall(tm, "$imcall", imcall);
    } else {
      Swig_warning(WARN_D_TYPEMAP_DOUT_UNDEF, input_file, line_number,
	"No dout typemap defined for %s\n", SwigType_str(t, 0));
    }

    Delete(proxy_param_types);

    // The whole function body is now in stored tm (if there was a matching type
    // map, of course), so simply append it to the code buffer. The braces are
    // included in the typemap.
    Printv(function_code, tm, NIL);

    // Write function code buffer to the class code.
    Printv(proxy_class_body_code, "\n", function_code, "\n", NIL);

    Delete(pre_code);
    Delete(post_code);
    Delete(terminator_code);
    Delete(function_code);
    Delete(return_type);
    Delete(imcall);
  }

  /* ---------------------------------------------------------------------------
   * D::writeProxyDModuleFunction()
   * --------------------------------------------------------------------------- */
  void writeProxyDModuleFunction(Node *n) {
    SwigType *t = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    String *tm;
    Parm *p;
    int i;
    String *imcall = NewString("");
    String *return_type = NewString("");
    String *function_code = NewString("");
    int num_arguments = 0;
    String *overloaded_name = getOverloadedName(n);
    String *func_name = NULL;
    String *pre_code = NewString("");
    String *post_code = NewString("");
    String *terminator_code = NewString("");

    // RESEARCH: What is this good for?
    if (l) {
      if (SwigType_type(Getattr(l, "type")) == T_VOID) {
	l = nextSibling(l);
      }
    }

    /* Attach the non-standard typemaps to the parameter list */
    Swig_typemap_attach_parms("dtype", l, NULL);
    Swig_typemap_attach_parms("din", l, NULL);

    /* Get return types */
    if ((tm = lookupDTypemap(n, "dtype"))) {
      String *dtypeout = Getattr(n, "tmap:dtype:out");
      if (dtypeout) {
	// The type in the dtype typemap's out attribute overrides the type in
	// the typemap.
	tm = dtypeout;
	replaceClassname(tm, t);
      }
      Printf(return_type, "%s", tm);
    } else {
      Swig_warning(WARN_D_TYPEMAP_DTYPE_UNDEF, input_file, line_number,
	"No dtype typemap defined for %s\n", SwigType_str(t, 0));
    }

    /* Change function name for global variables */
    if (global_variable_flag) {
      // RESEARCH: Is the Copy() needed here?
      func_name = Copy(variable_name);
    } else {
      func_name = Copy(Getattr(n, "sym:name"));
    }

    /* Start generating the function */
    const String *outattributes = Getattr(n, "tmap:dtype:outattributes");
    if (outattributes)
      Printf(function_code, "  %s\n", outattributes);

    const String *methodmods = Getattr(n, "feature:d:methodmodifiers");
    // TODO: Check if is_public(n) could possibly make any sense here
    // (private global functions would be useless anyway?).
    methodmods = methodmods ? methodmods : empty_string;

    Printf(function_code, "\n%s%s %s(", methodmods, return_type, func_name);
    Printv(imcall, im_dmodule_fq_name, ".", overloaded_name, "(", NIL);

    /* Get number of required and total arguments */
    num_arguments = emit_num_arguments(l);

    int gencomma = 0;

    /* Output each parameter */
    for (i = 0, p = l; i < num_arguments; i++) {

      /* Ignored parameters */
      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");
      String *param_type = NewString("");

      // Get the D parameter type.
      if ((tm = lookupDTypemap(p, "dtype", true))) {
	const String *inattributes = Getattr(p, "tmap:dtype:inattributes");
	Printf(param_type, "%s%s", inattributes ? inattributes : empty_string, tm);
      } else {
	Swig_warning(WARN_D_TYPEMAP_DTYPE_UNDEF, input_file, line_number,
	  "No dtype typemap defined for %s\n", SwigType_str(pt, 0));
      }

      if (gencomma)
	Printf(imcall, ", ");

      const bool generating_setter = global_variable_flag || wrapping_member_flag;
      String *arg = makeParameterName(n, p, i, generating_setter);

      // Get the D code to convert the parameter value to the type used in the
      // wrapper D module.
      if ((tm = lookupDTypemap(p, "din", true))) {
	Replaceall(tm, "$dinput", arg);
	String *pre = Getattr(p, "tmap:din:pre");
	if (pre) {
	  replaceClassname(pre, pt);
	  Replaceall(pre, "$dinput", arg);
	  if (Len(pre_code) > 0)
	    Printf(pre_code, "\n");
	  Printv(pre_code, pre, NIL);
	}
	String *post = Getattr(p, "tmap:din:post");
	if (post) {
	  replaceClassname(post, pt);
	  Replaceall(post, "$dinput", arg);
	  if (Len(post_code) > 0)
	    Printf(post_code, "\n");
	  Printv(post_code, post, NIL);
	}
	String *terminator = Getattr(p, "tmap:din:terminator");
	if (terminator) {
	  replaceClassname(terminator, pt);
	  Replaceall(terminator, "$dinput", arg);
	  if (Len(terminator_code) > 0)
	    Insert(terminator_code, 0, "\n");
	  Insert(terminator_code, 0, terminator);
	}
	Printv(imcall, tm, NIL);
      } else {
	Swig_warning(WARN_D_TYPEMAP_DIN_UNDEF, input_file, line_number,
	  "No din typemap defined for %s\n", SwigType_str(pt, 0));
      }

      /* Add parameter to module class function */
      if (gencomma >= 2)
	Printf(function_code, ", ");
      gencomma = 2;
      Printf(function_code, "%s %s", param_type, arg);

      p = Getattr(p, "tmap:in:next");
      Delete(arg);
      Delete(param_type);
    }

    Printf(imcall, ")");
    Printf(function_code, ") ");

    if (global_variable_flag && (d_version > 1)) {
      Printf(function_code, "@property ");
    }

    // Lookup the code used to convert the wrapper return value to the proxy
    // function return type.
    if ((tm = lookupDTypemap(n, "dout"))) {
      replaceExcode(n, tm, "dout", n);
      bool is_pre_code = Len(pre_code) > 0;
      bool is_post_code = Len(post_code) > 0;
      bool is_terminator_code = Len(terminator_code) > 0;
      if (is_pre_code || is_post_code || is_terminator_code) {
	if (is_post_code) {
	  Insert(tm, 0, "\n  try ");
	  Printv(tm, " finally {\n", post_code, "\n  }", NIL);
	} else {
	  Insert(tm, 0, "\n  ");
	}
	if (is_pre_code) {
	  Insert(tm, 0, pre_code);
	  Insert(tm, 0, "\n");
	}
	if (is_terminator_code) {
	  Printv(tm, "\n", terminator_code, NIL);
	}
	Insert(tm, 0, " {");
	Printf(tm, "\n}");
      }
      if (GetFlag(n, "feature:new"))
	Replaceall(tm, "$owner", "true");
      else
	Replaceall(tm, "$owner", "false");
      replaceClassname(tm, t);
      Replaceall(tm, "$imcall", imcall);
    } else {
      Swig_warning(WARN_D_TYPEMAP_DOUT_UNDEF, input_file, line_number,
	"No dout typemap defined for %s\n", SwigType_str(t, 0));
    }

    // The whole function code is now stored in tm (if there was a matching
    // type map, of course), so simply append it to the code buffer.
    Printf(function_code, "%s\n", tm ? (const String *) tm : empty_string);
    Printv(proxyCodeBuffer(getNSpace()), function_code, NIL);

    Delete(pre_code);
    Delete(post_code);
    Delete(terminator_code);
    Delete(function_code);
    Delete(return_type);
    Delete(imcall);
    Delete(func_name);
  }

  /* ---------------------------------------------------------------------------
   * D::writeProxyClassAndUpcasts()
   *
   * Collects all the code fragments generated by the handler function while
   * traversing the tree from the proxy_class_* variables and writes the
   * class definition (including any epilogue code) to proxy_class_code.
   *
   * Also writes the upcast function to the wrapper layer when processing a
   * derived class.
   *
   * Inputs:
   *  n – The class node currently processed.
   * --------------------------------------------------------------------------- */
  void writeProxyClassAndUpcasts(Node *n) {
    SwigType *typemap_lookup_type = Getattr(n, "classtypeobj");

    /*
     * Handle inheriting from D and C++ classes.
     */

    String *c_classname = SwigType_namestr(Getattr(n, "name"));
    String *c_baseclass = NULL;
    Node *basenode = NULL;
    String *basename = NULL;
    String *c_baseclassname = NULL;

    // Inheritance from pure D classes.
    Node *attributes = NewHash();
    const String *pure_baseclass =
      lookupCodeTypemap(n, "dbase", typemap_lookup_type, WARN_NONE, attributes);
    bool purebase_replace = GetFlag(attributes, "tmap:dbase:replace") ? true : false;
    bool purebase_notderived = GetFlag(attributes, "tmap:dbase:notderived") ? true : false;
    Delete(attributes);

    // C++ inheritance.
    if (!purebase_replace) {
      List *baselist = Getattr(n, "bases");
      if (baselist) {
	Iterator base = First(baselist);
	while (base.item) {
	  if (!GetFlag(base.item, "feature:ignore")) {
	    String *baseclassname = Getattr(base.item, "name");
	    if (!c_baseclassname) {
	      basenode = base.item;
	      c_baseclassname = baseclassname;
	      basename = createProxyName(c_baseclassname);
	      if (basename)
		c_baseclass = SwigType_namestr(baseclassname);
	    } else {
	      /* Warn about multiple inheritance for additional base class(es) */
	      String *proxyclassname = Getattr(n, "classtypeobj");
	      Swig_warning(WARN_D_MULTIPLE_INHERITANCE, Getfile(n), Getline(n),
		  "Base %s of class %s ignored: multiple inheritance is not supported in D.\n", SwigType_namestr(baseclassname), SwigType_namestr(proxyclassname));
	    }
	  }
	  base = Next(base);
	}
      }
    }

    bool derived = (basename != NULL);

    if (derived && purebase_notderived) {
      pure_baseclass = empty_string;
    }
    const String *wanted_base = basename ? basename : pure_baseclass;

    if (purebase_replace) {
      wanted_base = pure_baseclass;
      derived = false;
      basenode = NULL;
      Delete(basename);
      basename = NULL;
      if (purebase_notderived) {
	Swig_error(Getfile(n), Getline(n),
	  "The dbase typemap for proxy %s must contain just one of the 'replace' or 'notderived' attributes.\n",
	  typemap_lookup_type);
      }
    } else if (basename && Len(pure_baseclass) > 0) {
      Swig_warning(WARN_D_MULTIPLE_INHERITANCE, Getfile(n), Getline(n),
	"Warning for %s, base class %s ignored. Multiple inheritance is not supported in D. "
	"Perhaps you need one of the 'replace' or 'notderived' attributes in the dbase typemap?\n", typemap_lookup_type, pure_baseclass);
    }

    // Add code to do C++ casting to base class (only for classes in an inheritance hierarchy)
    if (derived) {
      writeClassUpcast(n, proxy_class_name, c_classname, c_baseclass);
    }

    /*
     * Write needed imports.
     */
    // If this class is derived from a C++ class, we need to have the D class
    // generated for it in scope.
    if (derived) {
      requireDType(Getattr(basenode, "sym:nspace"), Getattr(basenode, "sym:name"));
    }

    // Write any custom import statements to the proxy module header.
    const String *imports = lookupCodeTypemap(n, "dimports", typemap_lookup_type, WARN_NONE);
    if (Len(imports) > 0) {
      String* imports_trimmed = Copy(imports);
      Chop(imports_trimmed);
      replaceImportTypeMacros(imports_trimmed);
      Printv(proxy_class_imports, imports_trimmed, "\n", NIL);
      Delete(imports_trimmed);
    }

    /*
     * Write the proxy class header.
     */
    // Class modifiers.
    const String *modifiers =
      lookupCodeTypemap(n, "dclassmodifiers", typemap_lookup_type, WARN_D_TYPEMAP_CLASSMOD_UNDEF);

    // User-defined interfaces.
    const String *interfaces =
      lookupCodeTypemap(n, derived ? "dinterfaces_derived" : "dinterfaces", typemap_lookup_type, WARN_NONE);

    Printv(proxy_class_code,
      "\n",
      modifiers,
      " $dclassname",
      (*Char(wanted_base) || *Char(interfaces)) ? " : " : "", wanted_base,
      (*Char(wanted_base) && *Char(interfaces)) ? ", " : "", interfaces, " {",
      NIL);

    /*
     * Write the proxy class body.
     */
    String* body = NewString("");

    // Default class body.
    const String *dbody;
    if (derived) {
      dbody = lookupCodeTypemap(n, "dbody_derived", typemap_lookup_type, WARN_D_TYPEMAP_DBODY_UNDEF);
    } else {
      dbody = lookupCodeTypemap(n, "dbody", typemap_lookup_type, WARN_D_TYPEMAP_DBODY_UNDEF);
    }

    Printv(body, dbody, NIL);

    // Destructor and dispose().
    // If the C++ destructor is accessible (public), it is wrapped by the
    // dispose() method which is also called by the emitted D constructor. If it
    // is not accessible, no D destructor is written and the generated dispose()
    // method throws an exception.
    // This enables C++ classes with protected or private destructors to be used
    // in D as it would be used in C++ (GC finalization is a no-op then because
    // of the empty D destructor) while preventing usage in »scope« variables.
    // The method name for the dispose() method is specified in a typemap
    // attribute called »methodname«.
    const String *tm = NULL;

    String *dispose_methodname;
    String *dispose_methodmodifiers;
    attributes = NewHash();
    if (derived) {
      tm = lookupCodeTypemap(n, "ddispose_derived", typemap_lookup_type, WARN_NONE, attributes);
      dispose_methodname = Getattr(attributes, "tmap:ddispose_derived:methodname");
      dispose_methodmodifiers = Getattr(attributes, "tmap:ddispose_derived:methodmodifiers");
    } else {
      tm = lookupCodeTypemap(n, "ddispose", typemap_lookup_type, WARN_NONE, attributes);
      dispose_methodname = Getattr(attributes, "tmap:ddispose:methodname");
      dispose_methodmodifiers = Getattr(attributes, "tmap:ddispose:methodmodifiers");
    }

    if (tm && *Char(tm)) {
      if (!dispose_methodname) {
	Swig_error(Getfile(n), Getline(n),
	  "No methodname attribute defined in the ddispose%s typemap for %s\n",
	  (derived ? "_derived" : ""), proxy_class_name);
      }
      if (!dispose_methodmodifiers) {
	Swig_error(Getfile(n), Getline(n),
	  "No methodmodifiers attribute defined in ddispose%s typemap for %s.\n",
	  (derived ? "_derived" : ""), proxy_class_name);
      }
    }

    if (tm) {
      // Write the destructor if the C++ one is accessible.
      if (*Char(destructor_call)) {
	Printv(body,
	  lookupCodeTypemap(n, "ddestructor", typemap_lookup_type, WARN_NONE), NIL);
      }

      // Write the dispose() method.
      String *dispose_code = NewString("");
      Printv(dispose_code, tm, NIL);

      if (*Char(destructor_call)) {
	Replaceall(dispose_code, "$imcall", destructor_call);
      } else {
	Replaceall(dispose_code, "$imcall", "throw new object.Exception(\"C++ destructor does not have public access\")");
      }

      if (*Char(dispose_code)) {
	Printv(body, "\n", dispose_methodmodifiers,
	  (derived ? " override" : ""), " void ", dispose_methodname, "() ",
	  dispose_code, "\n", NIL);
      }
    }

    if (Swig_directorclass(n)) {
      // If directors are enabled for the current class, generate the
      // director connect helper function which is called from the constructor
      // and write it to the class body.
      writeDirectorConnectProxy(n);
    }

    // Write all constants and enumerations first to prevent forward reference
    // errors.
    Printv(body, proxy_class_enums_code, NIL);

    // Write the code generated in other methods to the class body.
    Printv(body, proxy_class_body_code, NIL);

    // Append extra user D code to the class body.
    Printv(body,
      lookupCodeTypemap(n, "dcode", typemap_lookup_type, WARN_NONE), "\n", NIL);

    // Write the class body and the curly bracket closing the class definition
    // to the proxy module.
    indentCode(body);
    Replaceall(body, "$dbaseclass", basename);
    Delete(basename);

    Printv(proxy_class_code, body, "\n}\n", NIL);
    Delete(body);

    // Write the epilogue code if there is any.
    Printv(proxy_class_code, proxy_class_epilogue_code, NIL);
  }


  /* ---------------------------------------------------------------------------
   * D::writeClassUpcast()
   * --------------------------------------------------------------------------- */
  void writeClassUpcast(Node *n, const String* d_class_name, String* c_class_name, String* c_base_name) {

    SwigType *smart = Swig_cparse_smartptr(n);
    String *upcast_name = Swig_name_member(getNSpace(), d_class_name, (smart != 0 ? "SmartPtrUpcast" : "Upcast"));
    String *upcast_wrapper_name = Swig_name_wrapper(upcast_name);

    writeImDModuleFunction(upcast_name, "void*", "(void* objectRef)",
      upcast_wrapper_name);

    if (smart) {
      SwigType *bsmart = Copy(smart);
      SwigType *rclassname = SwigType_typedef_resolve_all(c_class_name);
      SwigType *rbaseclass = SwigType_typedef_resolve_all(c_base_name);
      Replaceall(bsmart, rclassname, rbaseclass);
      Delete(rclassname);
      Delete(rbaseclass);
      String *smartnamestr = SwigType_namestr(smart);
      String *bsmartnamestr = SwigType_namestr(bsmart);
      Printv(upcasts_code,
	"SWIGEXPORT ", bsmartnamestr, " * ", upcast_wrapper_name,
	  "(", smartnamestr, " *objectRef) {\n",
	"    return objectRef ? new ", bsmartnamestr, "(*objectRef) : 0;\n"
	"}\n",
	"\n", NIL);
      Delete(bsmartnamestr);
      Delete(smartnamestr);
      Delete(bsmart);
    } else {
      Printv(upcasts_code,
	"SWIGEXPORT ", c_base_name, " * ", upcast_wrapper_name,
	  "(", c_base_name, " *objectRef) {\n",
	"    return (", c_base_name, " *)objectRef;\n"
	"}\n",
	"\n", NIL);
    }

    Replaceall(upcasts_code, "$cclass", c_class_name);
    Replaceall(upcasts_code, "$cbaseclass", c_base_name);

    Delete(upcast_name);
    Delete(upcast_wrapper_name);
    Delete(smart);
  }

  /* ---------------------------------------------------------------------------
   * D::writeTypeWrapperClass()
   * --------------------------------------------------------------------------- */
  void writeTypeWrapperClass(String *classname, SwigType *type) {
    Node *n = NewHash();
    Setfile(n, input_file);
    Setline(n, line_number);

    assertClassNameValidity(classname);

    String* imports_target;
    String* code_target;
    File *class_file = NULL;
    if (split_proxy_dmodule) {
      String *filename = NewStringf("%s%s.d", dmodule_directory, classname);
      class_file = NewFile(filename, "w", SWIG_output_files());
      if (!class_file) {
	FileErrorDisplay(filename);
	SWIG_exit(EXIT_FAILURE);
      }
      Append(filenames_list, Copy(filename));
      Delete(filename);

      emitBanner(class_file);
      Printf(class_file, "module %s%s;\n", package, classname);
      Printf(class_file, "\nstatic import %s;\n", im_dmodule_fq_name);

      imports_target = NewString("");
      code_target = NewString("");
    } else {
      imports_target = proxyImportsBuffer(0);
      code_target = proxyCodeBuffer(0);
    }

    // Import statements.
    const String *imports = lookupCodeTypemap(n, "dimports", type, WARN_NONE);
    if (Len(imports) > 0) {
      String *imports_trimmed = Copy(imports);
      Chop(imports_trimmed);
      replaceImportTypeMacros(imports_trimmed);
      Printv(imports_target, imports_trimmed, "\n", NIL);
      Delete(imports_trimmed);
    }

    // Pure D baseclass and interfaces (no C++ inheritance possible.
    const String *pure_baseclass = lookupCodeTypemap(n, "dbase", type, WARN_NONE);
    const String *pure_interfaces = lookupCodeTypemap(n, "dinterfaces", type, WARN_NONE);

    // Emit the class.
    Printv(code_target,
      "\n",
      lookupCodeTypemap(n, "dclassmodifiers", type, WARN_D_TYPEMAP_CLASSMOD_UNDEF),
      " $dclassname",
      (*Char(pure_baseclass) || *Char(pure_interfaces)) ? " : " : "", pure_baseclass,
      ((*Char(pure_baseclass)) && *Char(pure_interfaces)) ? ", " : "", pure_interfaces,
      " {", NIL);

    String* body = NewString("");
    Printv(body, lookupCodeTypemap(n, "dbody", type, WARN_D_TYPEMAP_DBODY_UNDEF),
      lookupCodeTypemap(n, "dcode", type, WARN_NONE), NIL);
    indentCode(body);
    Printv(code_target, body, "\n}\n", NIL);
    Delete(body);

    Replaceall(code_target, "$dclassname", classname);

    if (split_proxy_dmodule) {
      Printv(class_file, imports_target, NIL);
      Delete(imports_target);

      replaceModuleVariables(code_target);
      Printv(class_file, code_target, NIL);
      Delete(code_target);

      Delete(class_file);
    }

    Delete(n);
  }

  /* ---------------------------------------------------------------------------
   * D::writeDirectorConnectProxy(Node *classNode)
   *
   * Writes the helper method which registers the director callbacks by calling
   * the director connect function from the D side to the proxy class.
   * --------------------------------------------------------------------------- */
  void writeDirectorConnectProxy(Node* classNode) {
    String *dirClassName = directorClassName(classNode);
    String *connect_name = Swig_name_member(getNSpace(),
      proxy_class_name, "director_connect");
    Printf(proxy_class_body_code, "\nprivate void swigDirectorConnect() {\n");

    int i;
    for (i = first_class_dmethod; i < curr_class_dmethod; ++i) {
      UpcallData *udata = Getitem(dmethods_seq, i);
      String *method = Getattr(udata, "method");
      String *overloaded_name = Getattr(udata, "overname");
      String *return_type = Getattr(udata, "return_type");
      String *param_list = Getattr(udata, "param_list");
      String *methid = Getattr(udata, "class_methodidx");
      Printf(proxy_class_body_code, "  %s.%s_Callback%s callback%s;\n", im_dmodule_fq_name, dirClassName, methid, methid);
      Printf(proxy_class_body_code, "  if (swigIsMethodOverridden!(%s delegate(%s), %s function(%s), %s)()) {\n", return_type, param_list, return_type, param_list, method);
      Printf(proxy_class_body_code, "    callback%s = &swigDirectorCallback_%s_%s;\n", methid, proxy_class_name, overloaded_name);
      Printf(proxy_class_body_code, "  }\n\n");
    }
    Printf(proxy_class_body_code, "  %s.%s(cast(void*)swigCPtr, cast(void*)this", im_dmodule_fq_name, connect_name);
    for (i = first_class_dmethod; i < curr_class_dmethod; ++i) {
      UpcallData *udata = Getitem(dmethods_seq, i);
      String *methid = Getattr(udata, "class_methodidx");
      Printf(proxy_class_body_code, ", callback%s", methid);
    }
    Printf(proxy_class_body_code, ");\n");
    Printf(proxy_class_body_code, "}\n");

    // Helper function to determine if a method has been overridden in a
    // subclass of the wrapped class. If not, we just pass null to the
    // director_connect_function since the method from the C++ class should
    // be called as usual (see above).
    // Only emit it if the proxy class has at least one method.
    if (first_class_dmethod < curr_class_dmethod) {
      Printf(proxy_class_body_code, "\n");
      Printf(proxy_class_body_code, "private bool swigIsMethodOverridden(DelegateType, FunctionType, alias fn)() %s{\n", (d_version > 1) ? "const " : "");
      Printf(proxy_class_body_code, "  DelegateType dg = &fn;\n");
      Printf(proxy_class_body_code, "  return dg.funcptr != SwigNonVirtualAddressOf!(FunctionType, fn);\n");
      Printf(proxy_class_body_code, "}\n");
      Printf(proxy_class_body_code, "\n");
      Printf(proxy_class_body_code, "private static Function SwigNonVirtualAddressOf(Function, alias fn)() {\n");
      Printf(proxy_class_body_code, "  return cast(Function) &fn;\n");
      Printf(proxy_class_body_code, "}\n");
    }

    if (Len(director_dcallbacks_code) > 0) {
      Printv(proxy_class_epilogue_code, director_dcallbacks_code, NIL);
    }

    Delete(director_callback_typedefs);
    director_callback_typedefs = NULL;
    Delete(director_callback_pointers);
    director_callback_pointers = NULL;
    Delete(director_dcallbacks_code);
    director_dcallbacks_code = NULL;
    Delete(dirClassName);
    Delete(connect_name);
  }

  /* ---------------------------------------------------------------------------
   * D::writeDirectorConnectWrapper()
   *
   * Writes the director connect function and the corresponding declaration to
   * the C++ wrapper respectively the D wrapper.
   * --------------------------------------------------------------------------- */
  void writeDirectorConnectWrapper(Node *n) {
    if (!Swig_directorclass(n))
      return;

    // Output the director connect method.
    String *norm_name = SwigType_namestr(Getattr(n, "name"));
    String *connect_name = Swig_name_member(getNSpace(),
      proxy_class_name, "director_connect");
    String *dirClassName = directorClassName(n);
    Wrapper *code_wrap;

    Printv(wrapper_loader_bind_code, wrapper_loader_bind_command, NIL);
    Replaceall(wrapper_loader_bind_code, "$function", connect_name);
    Replaceall(wrapper_loader_bind_code, "$symbol", Swig_name_wrapper(connect_name));

    Printf(im_dmodule_code, "extern(C) void function(void* cObject, void* dObject");

    code_wrap = NewWrapper();
    Printf(code_wrap->def, "SWIGEXPORT void D_%s(void *objarg, void *dobj", connect_name);

    Printf(code_wrap->code, "  %s *obj = (%s *)objarg;\n", norm_name, norm_name);
    Printf(code_wrap->code, "  %s *director = dynamic_cast<%s *>(obj);\n", dirClassName, dirClassName);

    Printf(code_wrap->code, "  if (director) {\n");
    Printf(code_wrap->code, "    director->swig_connect_director(dobj");

    for (int i = first_class_dmethod; i < curr_class_dmethod; ++i) {
      UpcallData *udata = Getitem(dmethods_seq, i);
      String *methid = Getattr(udata, "class_methodidx");

      Printf(code_wrap->def, ", %s::SWIG_Callback%s_t callback%s", dirClassName, methid, methid);
      Printf(code_wrap->code, ", callback%s", methid);
      Printf(im_dmodule_code, ", %s_Callback%s callback%s", dirClassName, methid, methid);
    }

    Printf(code_wrap->def, ") {\n");
    Printf(code_wrap->code, ");\n");
    Printf(im_dmodule_code, ") %s;\n", connect_name);
    Printf(code_wrap->code, "  }\n");
    Printf(code_wrap->code, "}\n");

    Wrapper_print(code_wrap, f_wrappers);
    DelWrapper(code_wrap);

    Delete(connect_name);
    Delete(dirClassName);
  }

  /* ---------------------------------------------------------------------------
   * D::requireDType()
   *
   * If the given type is not already in scope in the current module, adds an
   * import statement for it. The name is considered relative to the global root
   * package if one is set.
   *
   * This is only used for dependencies created in generated code, user-
   * (i.e. typemap-) specified import statements are handled separately.
   * --------------------------------------------------------------------------- */
  void requireDType(const String *nspace, const String *symname) {
    String *dmodule = createModuleName(nspace, symname);

    if (!inProxyModule(dmodule)) {
      String *import = createImportStatement(dmodule);
      Append(import, "\n");
      if (is_wrapping_class()) {
	addImportStatement(proxy_class_imports, import);
      } else {
	addImportStatement(proxyImportsBuffer(getNSpace()), import);
      }
      Delete(import);
    }
    Delete(dmodule);
  }

  /* ---------------------------------------------------------------------------
   * D::addImportStatement()
   *
   * Adds the given import statement to the given list of import statements if
   * there is no statement importing that module present yet.
   * --------------------------------------------------------------------------- */
  void addImportStatement(String *target, const String *import) const {
    char *position = Strstr(target, import);
    if (position) {
      // If the import statement has been found in the target string, we have to
      // check if the previous import was static, which would lead to problems
      // if this import is not.
      // Thus, we check if the seven characters in front of the occurrence are
      // »static «. If the import string passed is also static, the checks fail
      // even if the found statement is also static because the last seven
      // characters would be part of the previous import statement then.

      if (position - Char(target) < 7) {
	return;
      }
      if (strncmp(position - 7, "static ", 7)) {
	return;
      }
    }

    Printv(target, import, NIL);
  }

  /* ---------------------------------------------------------------------------
   * D::createImportStatement()
   *
   * Creates a string containing an import statement for the given module.
   * --------------------------------------------------------------------------- */
  String *createImportStatement(const String *dmodule_name,
    bool static_import = true) const {

    if (static_import) {
      return NewStringf("static import %s%s;", package, dmodule_name);
    } else {
      return NewStringf("import %s%s;", package, dmodule_name);
    }
  }

  /* ---------------------------------------------------------------------------
   * D::inProxyModule()
   *
   * Determines if the specified proxy type is declared in the currently
   * processed proxy D module.
   *
   * This function is used to determine if fully qualified type names have to
   * be used (package, module and type name). If the split proxy mode is not
   * used, this solely depends on whether the type is in the current namespace.
   * --------------------------------------------------------------------------- */
  bool inProxyModule(const String *type_name) const {
    if (!split_proxy_dmodule) {
      String *nspace = createOuterNamespaceNames(type_name);

      // Check if strings are either both null (no namespace) or are both
      // non-null and have the same contents. Cannot use Strcmp for this
      // directly because of its strange way of handling the case where only
      // one argument is 0 ("<").
      bool result = !nspace && !getNSpace();
      if (nspace && getNSpace())
	result = (Strcmp(nspace, getNSpace()) == 0);

      Delete(nspace);
      return result;
    }

    if (!is_wrapping_class()) {
      return false;
    }

    return (Strcmp(proxy_class_qname, type_name) == 0);
  }

  /* ---------------------------------------------------------------------------
   * D::addUpcallMethod()
   *
   * Adds new director upcall signature.
   * --------------------------------------------------------------------------- */
  UpcallData *addUpcallMethod(String *imclass_method, String *class_method,
    String *decl, String *overloaded_name, String *return_type, String *param_list) {

    String *key = NewStringf("%s|%s", imclass_method, decl);

    ++curr_class_dmethod;

    String *class_methodidx = NewStringf("%d", n_dmethods - first_class_dmethod);
    n_dmethods++;

    Hash *new_udata = NewHash();
    Append(dmethods_seq, new_udata);
    Setattr(dmethods_table, key, new_udata);

    Setattr(new_udata, "method", Copy(class_method));
    Setattr(new_udata, "class_methodidx", class_methodidx);
    Setattr(new_udata, "decl", Copy(decl));
    Setattr(new_udata, "overname", Copy(overloaded_name));
    Setattr(new_udata, "return_type", Copy(return_type));
    Setattr(new_udata, "param_list", Copy(param_list));

    Delete(key);
    return new_udata;
  }

  /* ---------------------------------------------------------------------------
   * D::assertClassNameValidity()
   * --------------------------------------------------------------------------- */
  void assertClassNameValidity(const String* class_name) const {
    // TODO: With nspace support, there could arise problems also when not in
    // split proxy mode, warnings for these should be added.
    if (split_proxy_dmodule) {
      if (Cmp(class_name, im_dmodule_name) == 0) {
	Swig_error(input_file, line_number,
	  "Class name cannot be equal to intermediary D module name: %s\n",
	  class_name);
	SWIG_exit(EXIT_FAILURE);
      }

      String *nspace = getNSpace();
      if (nspace) {
	// Check the root package/outermost namespace (a class A in module
	// A.B leads to problems if another module A.C is also imported)
	if (Len(package) > 0) {
	  String *dotless_package = NewStringWithSize(package, Len(package) - 1);
	  if (Cmp(class_name, dotless_package) == 0) {
	    Swig_error(input_file, line_number,
	      "Class name cannot be the same as the root package it is in: %s\n",
	      class_name);
	    SWIG_exit(EXIT_FAILURE);
	  }
	  Delete(dotless_package);
	} else {
	  String *outer = createFirstNamespaceName(nspace);
	  if (Cmp(class_name, outer) == 0) {
	    Swig_error(input_file, line_number,
	      "Class name cannot be the same as the outermost namespace it is in: %s\n",
	      class_name);
	    SWIG_exit(EXIT_FAILURE);
	  }
	  Delete(outer);
	}

	// … and the innermost one (because of the conflict with the main proxy
	// module named like the namespace).
	String *inner = createLastNamespaceName(nspace);
	if (Cmp(class_name, inner) == 0) {
	  Swig_error(input_file, line_number,
	    "Class name cannot be the same as the innermost namespace it is in: %s\n",
	    class_name);
	  SWIG_exit(EXIT_FAILURE);
	}
	Delete(inner);
      } else {
	if (Cmp(class_name, proxy_dmodule_name) == 0) {
	  Swig_error(input_file, line_number,
	    "Class name cannot be equal to proxy D module name: %s\n",
	    class_name);
	  SWIG_exit(EXIT_FAILURE);
	}
      }
    }
  }

  /* ---------------------------------------------------------------------------
   * D::getPrimitiveDptype()
   *
   * Returns the D proxy type for the passed type if it is a primitive type in
   * both C and D.
   * --------------------------------------------------------------------------- */
  String *getPrimitiveDptype(Node *node, SwigType *type) {
    SwigType *stripped_type = SwigType_typedef_resolve_all(type);

    // A reference can only be the »outermost element« of a type.
    bool mutable_ref = false;
    if (SwigType_isreference(stripped_type)) {
      SwigType_del_reference(stripped_type);

      if (SwigType_isconst(stripped_type)) {
	SwigType_del_qualifier(stripped_type);
      } else {
	mutable_ref = true;
      }
    }

    // Strip all the pointers from the type.
    int indirection_count = 0;
    while (SwigType_ispointer(stripped_type)) {
      ++indirection_count;
      SwigType_del_pointer(stripped_type);
    }

    // Now that we got rid of the pointers, see if we are dealing with a
    // primitive type.
    String *dtype = 0;
    if (SwigType_isfunction(stripped_type) && indirection_count > 0) {
      // type was a function pointer, split it up.
      SwigType_add_pointer(stripped_type);
      --indirection_count;

      SwigType *return_type = Copy(stripped_type);
      SwigType *params_type = SwigType_functionpointer_decompose(return_type);
      String *return_dtype = getPrimitiveDptype(node, return_type);
      Delete(return_type);
      if (!return_dtype) {
	return 0;
      }

      List *parms = SwigType_parmlist(params_type);
      List *param_dtypes = NewList();
      for (Iterator it = First(parms); it.item; it = Next(it)) {
	String *current_dtype = getPrimitiveDptype(node, it.item);
	if (Cmp(current_dtype, "void") == 0) {
	  // void somefunc(void) is legal syntax in C, but not in D, so simply
	  // skip the void parameter.
	  Delete(current_dtype);
	  continue;
	}
	if (!current_dtype) {
	  Delete(return_dtype);
	  Delete(param_dtypes);
	  return 0;
	}
	Append(param_dtypes, current_dtype);
      }

      String *param_list = NewString("");
      {
	bool gen_comma = false;
	for (Iterator it = First(param_dtypes); it.item; it = Next(it)) {
	  if (gen_comma) {
	    Append(param_list, ", ");
	  }
	  Append(param_list, it.item);
	  Delete(it.item);
	  gen_comma = true;
	}
      }

      dtype = NewStringf("%s.SwigExternC!(%s function(%s))", im_dmodule_fq_name,
        return_dtype, param_list);
      Delete(param_list);
      Delete(param_dtypes);
      Delete(return_dtype);
    } else {
      Hash *attributes = NewHash();
      const String *tm =
	lookupCodeTypemap(node, "dtype", stripped_type, WARN_NONE, attributes);
      if(!GetFlag(attributes, "tmap:dtype:cprimitive")) {
	dtype = 0;
      } else {
	dtype = Copy(tm);

	// We need to call replaceClassname here with the stripped type to avoid
	// $dclassname in the enum typemaps being replaced later with the full
	// type.
	replaceClassname(dtype, stripped_type);
      }
      Delete(attributes);
    }
    Delete(stripped_type);

    if (!dtype) {
      // The type passed is no primitive type.
      return 0;
    }

    // The type is ultimately a primitive type, now append the right number of
    // indirection levels (pointers).
    for (int i = 0; i < indirection_count; ++i) {
      Append(dtype, "*");
    }

    // Add a level of indirection for a mutable reference since it is wrapped
    // as a pointer.
    if (mutable_ref) {
      Append(dtype, "*");
    }

    return dtype;
  }

  /* ---------------------------------------------------------------------------
   * D::lookupCodeTypemap()
   *
   * Looks up a D code fragment for generating the wrapper class for the given
   * type.
   *
   * n - for input only and must contain info for Getfile(n) and Getline(n) to work
   * tmap_method - typemap method name
   * type - typemap type to lookup
   * warning - warning number to issue if no typemaps found
   * typemap_attributes - the typemap attributes are attached to this node and will
   *   also be used for temporary storage if non null
   * return is never NULL, unlike Swig_typemap_lookup()
   * --------------------------------------------------------------------------- */
  const String *lookupCodeTypemap(Node *n, const_String_or_char_ptr tmap_method,
    SwigType *type, int warning, Node *typemap_attributes = 0) const {

    Node *node = !typemap_attributes ? NewHash() : typemap_attributes;
    Setattr(node, "type", type);
    Setfile(node, Getfile(n));
    Setline(node, Getline(n));
    const String *tm = Swig_typemap_lookup(tmap_method, node, "", 0);
    if (!tm) {
      tm = empty_string;
      if (warning != WARN_NONE) {
	Swig_warning(warning, Getfile(n), Getline(n),
	  "No %s typemap defined for %s\n", tmap_method, SwigType_str(type, 0));
      }
    }
    if (!typemap_attributes) {
      Delete(node);
    }

    return tm;
  }

  /* ---------------------------------------------------------------------------
   * D::lookupDTypemap()
   *
   * Looks up a D typemap for the given node, replacing D-specific special
   * variables as needed.
   *
   * The method parameter specifies the typemap method to use. If attached is
   * true, the value is just fetched from the tmap:<method> node attribute,
   * Swig_typemap_lookup is used otherwise.
   * --------------------------------------------------------------------------- */
  String *lookupDTypemap(Node *n, const_String_or_char_ptr method, bool attached = false) {
    String *result = 0;

    if (attached) {
      String *attr_name = NewStringf("tmap:%s", method);
      result = Copy(Getattr(n, attr_name));
      Delete(attr_name);
    } else {
      // FIXME: As a workaround for a bug so far only surfacing in the
      // smart_pointer_const_overload test case, remove the nativepointer
      // typemap attribute since it seems to be already there from a dout
      // typemap of a different type in that test.
      String *np_key = NewStringf("tmap:%s:nativepointer", method);
      Delattr(n, np_key);
      Delete(np_key);

      result = Swig_typemap_lookup(method, n, "", 0);
    }

    if (!result) {
      return 0;
    }

    // Check if the passed node actually has type information attached. This
    // is not the case e.g. in constructorWrapper.
    SwigType *type = Getattr(n, "type");
    if (type) {
      String *np_key = NewStringf("tmap:%s:nativepointer", method);
      String *np_value = Getattr(n, np_key);
      Delete(np_key);
      String *dtype;
      if (np_value && (dtype = getPrimitiveDptype(n, type))) {
        // If the typemap in question has a »nativepointer« attribute and we
        // are dealing with a primitive type, use it instead.
        result = Copy(np_value);
        Replaceall(result, "$dtype", dtype);
      }

      replaceClassname(result, type);
    }

    return result;
  }

  /* ---------------------------------------------------------------------------
   * D::replaceClassname()
   *
   * Replaces the special variable $dclassname with the proxy class name for
   * classes/structs/unions SWIG knows about. Also substitutes the enumeration
   * name for non-anonymous enums. Otherwise, $classname is replaced with a
   * $descriptor(type)-like name.
   *
   * $*dclassname and $&classname work like with descriptors (see manual section
   * 10.4.3), they remove a prointer from respectively add a pointer to the type.
   *
   * Inputs:
   *   tm - String to perform the substitution at (will usually come from a
   *        typemap.
   *   pt - The type to substitute for the variables.
   * Outputs:
   *   tm - String with the variables substituted.
   * Return:
   *   substitution_performed - flag indicating if a substitution was performed
   * --------------------------------------------------------------------------- */
  bool replaceClassname(String *tm, SwigType *pt) {
    bool substitution_performed = false;
    SwigType *type = Copy(SwigType_typedef_resolve_all(pt));
    SwigType *strippedtype = SwigType_strip_qualifiers(type);

    if (Strstr(tm, "$dclassname")) {
      SwigType *classnametype = Copy(strippedtype);
      replaceClassnameVariable(tm, "$dclassname", classnametype);
      substitution_performed = true;
      Delete(classnametype);
    }
    if (Strstr(tm, "$*dclassname")) {
      SwigType *classnametype = Copy(strippedtype);
      Delete(SwigType_pop(classnametype));
      replaceClassnameVariable(tm, "$*dclassname", classnametype);
      substitution_performed = true;
      Delete(classnametype);
    }
    if (Strstr(tm, "$&dclassname")) {
      SwigType *classnametype = Copy(strippedtype);
      SwigType_add_pointer(classnametype);
      replaceClassnameVariable(tm, "$&dclassname", classnametype);
      substitution_performed = true;
      Delete(classnametype);
    }

    Delete(strippedtype);
    Delete(type);

    return substitution_performed;
  }

  /* ---------------------------------------------------------------------------
   * D::replaceClassnameVariable()
   *
   * See D::replaceClassname().
   * --------------------------------------------------------------------------- */
  void replaceClassnameVariable(String *target, const char *variable, SwigType *type) {
    // TODO: Fix const-correctness of methods called in here and make type const.

    // We make use of the fact that this function is called at least once for
    // every type encountered which is written to a separate file, which allows
    // us to handle imports here.
    // When working in split proxy module mode, each generated proxy class/enum
    // is written to a separate module. This requires us to add a corresponding
    // import when a type is used in another generated module. If we are not
    // working in split proxy module mode, this is not relevant and the
    // generated module name is discarded.
    String *type_name;

    if (SwigType_isenum(type)) {
      // RESEARCH: Make sure that we really cannot get here for anonymous enums.
      Node *n = enumLookup(type);
      if (n) {
	String *enum_name = Getattr(n, "sym:name");

	Node *p = parentNode(n);
	if (p && !Strcmp(nodeType(p), "class")) {
	  // This is a nested enum.
	  String *parent_name = Getattr(p, "sym:name");
	  String *nspace = Getattr(p, "sym:nspace");

	  // An enum nested in a class is not written to a separate module (this
	  // would not even be possible in D), so just import the parent.
	  requireDType(nspace, parent_name);

	  String *module = createModuleName(nspace, parent_name);
	  if (inProxyModule(module)) {
	    type_name = NewStringf("%s.%s", parent_name, enum_name);
	  } else {
	    type_name = NewStringf("%s%s.%s.%s", package, module, parent_name, enum_name);
	  }
	} else {
	  // A non-nested enum is written to a separate module, import it.
	  String *nspace = Getattr(n, "sym:nspace");
	  requireDType(nspace, enum_name);

	  String *module = createModuleName(nspace, enum_name);
	  if (inProxyModule(module)) {
	    type_name = Copy(enum_name);
	  } else {
	    type_name = NewStringf("%s%s.%s", package, module, enum_name);
	  }
	}
      } else {
	type_name = NewStringf("int");
      }
    } else {
      Node *n = classLookup(type);
      if (n) {
	String *class_name = Getattr(n, "sym:name");
	String *nspace = Getattr(n, "sym:nspace");
	requireDType(nspace, class_name);

	String *module = createModuleName(nspace, class_name);
	if (inProxyModule(module)) {
	  type_name = Copy(class_name);
	} else {
	  type_name = NewStringf("%s%s.%s", package, module, class_name);
	}
        Delete(module);
      } else {
	// SWIG does not know anything about the type (after resolving typedefs).
	// Just mangle the type name string like $descriptor(type) would do.
	String *descriptor = NewStringf("SWIGTYPE%s", SwigType_manglestr(type));
	requireDType(NULL, descriptor);

	String *module = createModuleName(NULL, descriptor);
	if (inProxyModule(module)) {
	  type_name = Copy(descriptor);
	} else {
	  type_name = NewStringf("%s%s.%s", package, module, descriptor);
	}
	Delete(module);

	// Add to hash table so that a type wrapper class can be created later.
	Setattr(unknown_types, descriptor, type);

	Delete(descriptor);
      }
    }

    Replaceall(target, variable, type_name);
    Delete(type_name);
  }

  /* ---------------------------------------------------------------------------
   * D::createModuleName()
   *
   * Returns a string holding the name of the module to import to bring the
   * given type in scope.
   * --------------------------------------------------------------------------- */
  String *createModuleName(const String *nspace, const String *type_name) const {
    String *module;
    if (nspace) {
      module = NewStringf("%s.", nspace);
      if (split_proxy_dmodule) {
	Printv(module, type_name, NIL);
      } else {
	String *inner = createLastNamespaceName(nspace);
	Printv(module, inner, NIL);
	Delete(inner);
      }
    } else {
      if (split_proxy_dmodule) {
	module = Copy(type_name);
      } else {
	module = Copy(proxy_dmodule_name);
      }
    }
    return module;
  }

  /* ---------------------------------------------------------------------------
   * D::replaceModuleVariables()
   *
   * Replaces the $imdmodule and $module variables with their values in the
   * target string.
   * --------------------------------------------------------------------------- */
  void replaceModuleVariables(String *target) const {
    Replaceall(target, "$imdmodule", im_dmodule_fq_name);
    Replaceall(target, "$module", proxy_dmodule_name);
  }

  /* ---------------------------------------------------------------------------
   * D::replaceExcode()
   *
   * If a C++ method can throw a exception, additional code is added to the
   * proxy method to check if an exception is pending so that it can be
   * rethrown on the D side.
   *
   * This method replaces the $excode variable with the exception handling code
   * in the excode typemap attribute if it »canthrow« an exception.
   * --------------------------------------------------------------------------- */
  void replaceExcode(Node *n, String *code, const String *typemap, Node *parameter) const {
    String *excode_attribute = NewStringf("tmap:%s:excode", typemap);
    String *excode = Getattr(parameter, excode_attribute);
    if (Getattr(n, "d:canthrow")) {
      int count = Replaceall(code, "$excode", excode);
      if (count < 1 || !excode) {
	Swig_warning(WARN_D_EXCODE_MISSING, input_file, line_number,
	  "D exception may not be thrown – no $excode or excode attribute in '%s' typemap.\n",
	  typemap);
      }
    } else {
      Replaceall(code, "$excode", "");
    }
    Delete(excode_attribute);
  }

  /* ---------------------------------------------------------------------------
   * D::replaceImportTypeMacros()
   *
   * Replaces the $importtype(SomeDClass) macro with an import statement if it
   * is required to get SomeDClass in scope for the currently generated proxy
   * D module.
   * --------------------------------------------------------------------------- */
  void replaceImportTypeMacros(String *target) const {
    // Code from replace_embedded_typemap.
    char *start = 0;
    while ((start = Strstr(target, "$importtype("))) {
      char *end = 0;
      char *param_start = 0;
      char *param_end = 0;
      int level = 0;
      char *c = start;
      while (*c) {
	if (*c == '(') {
	  if (level == 0) {
	    param_start = c + 1;
	  }
	  level++;
	}
	if (*c == ')') {
	  level--;
	  if (level == 0) {
	    param_end = c;
	    end = c + 1;
	    break;
	  }
	}
	c++;
      }

      if (end) {
	String *current_macro = NewStringWithSize(start, (int)(end - start));
	String *current_param = NewStringWithSize(param_start, (int)(param_end - param_start));


	if (inProxyModule(current_param)) {
	  Replace(target, current_macro, "", DOH_REPLACE_ANY);
	} else {
	  String *import = createImportStatement(current_param, false);
	  Replace(target, current_macro, import, DOH_REPLACE_ANY);
	  Delete(import);
	}

	Delete(current_param);
	Delete(current_macro);
      } else {
	String *current_macro = NewStringWithSize(start, (int)(c - start));
	Swig_error(Getfile(target), Getline(target), "Syntax error in: %s\n", current_macro);
	Replace(target, current_macro, "<error in $importtype macro>", DOH_REPLACE_ANY);
	Delete(current_macro);
      }
    }
  }

  /* ---------------------------------------------------------------------------
   * D::getOverloadedName()
   * --------------------------------------------------------------------------- */
  String *getOverloadedName(Node *n) const {
    // A void* parameter is used for all wrapped classes in the wrapper code.
    // Thus, the wrapper function names for overloaded functions are postfixed
    // with a counter string to make them unique.
    String *overloaded_name = Copy(Getattr(n, "sym:name"));

    if (Getattr(n, "sym:overloaded")) {
      Append(overloaded_name, Getattr(n, "sym:overname"));
    }

    return overloaded_name;
  }

  /* ---------------------------------------------------------------------------
   * D::createProxyName()
   *
   * Returns the D class name if a type corresponds to something wrapped with a
   * proxy class, NULL otherwise.
   * --------------------------------------------------------------------------- */
  String *createProxyName(SwigType *t) {
    String *proxyname = NULL;
    Node *n = classLookup(t);
    if (n) {
      String *nspace = Getattr(n, "sym:nspace");
      String *symname = Getattr(n, "sym:name");

      String *module = createModuleName(nspace, symname);
      if (inProxyModule(module)) {
	proxyname = Copy(symname);
      } else {
	proxyname = NewStringf("%s%s.%s", package, module, symname);
      }
    }
    return proxyname;
  }

  String *makeParameterName(Node *n, Parm *p, int arg_num, bool setter) const {
    String *arg = Language::makeParameterName(n, p, arg_num, setter);

    if (split_proxy_dmodule && Strncmp(arg, package, Len(arg)) == 0) {
      // If we are in split proxy mode and the argument is named like the target
      // package, we append an underscore to its name to avoid clashes.
      Append(arg, "_");
    }

    return arg;
  }

  /* ---------------------------------------------------------------------------
   * D::canThrow()
   *
   * Determines whether the code in the typemap can throw a D exception.
   * If so, note it for later when excodeSubstitute() is called.
   * --------------------------------------------------------------------------- */
  void canThrow(Node *n, const String *typemap, Node *parameter) const {
    String *canthrow_attribute = NewStringf("tmap:%s:canthrow", typemap);
    String *canthrow = Getattr(parameter, canthrow_attribute);
    if (canthrow)
      Setattr(n, "d:canthrow", "1");
    Delete(canthrow_attribute);
  }

  /* ---------------------------------------------------------------------------
   * D::wrapMemberFunctionAsDConst()
   *
   * Determines whether the member function represented by the passed node is
   * wrapped as D »const« or not.
   * --------------------------------------------------------------------------- */
  bool wrapMemberFunctionAsDConst(Node *n) const {
    if (d_version == 1) return false;
    if (static_flag) return false; // Never emit »const« for static member functions.
    return GetFlag(n, "memberget") || SwigType_isconst(Getattr(n, "decl"));
  }

  /* ---------------------------------------------------------------------------
   * D::areAllOverloadsOverridden()
   *
   * Determines whether the class the passed function node belongs to overrides
   * all the overlaods for the passed function node defined somewhere up the
   * inheritance hierachy.
   * --------------------------------------------------------------------------- */
  bool areAllOverloadsOverridden(Node *n) const {
    List *base_list = Getattr(parentNode(n), "bases");
    if (!base_list) {
      // If the class which contains n is not derived from any other class,
      // there cannot be any not-overridden overloads.
      return true;
    }

    // In case of multiple base classes, skip to the one which has not been
    // ignored.
    // RESEARCH: Also emit a warning in case of multiple inheritance here?
    Iterator it = First(base_list);
    while (it.item && GetFlag(it.item, "feature:ignore")) {
      it = Next(it);
    }
    Node *base_class = it.item;

    if (!base_class) {
      // If all base classes have been ignored, there cannot be one either.
      return true;
    }

    // We try to find at least a single overload which exists in the base class
    // so we can progress up the inheritance hierachy even if there have been
    // new overloads introduced after the topmost class.
    Node *base_function = NULL;
    String *symname = Getattr(n, "sym:name");
    if (symname) {
      for (Node *tmp = firstChild(base_class); tmp; tmp = nextSibling(tmp)) {
	String *child_symname = Getattr(tmp, "sym:name");
	if (child_symname && (Strcmp(child_symname, symname) == 0)) {
	  base_function = tmp;
	  break;
	}
      }
    }

    if (!base_function) {
      // If there is no overload which also exists in the super class, there
      // cannot be any base class overloads not overridden.
      return true;
    }

    size_t base_overload_count = 0;
    for (Node *tmp = firstSibling(base_function); tmp; tmp = Getattr(tmp, "sym:nextSibling")) {
      if (is_protected(base_function) &&
	  !(Swig_director_mode() && Swig_director_protected_mode() && Swig_all_protected_mode())) {
	// If the base class function is »protected« and were are not in
	// director mode, it is not emitted to the base class and thus we do
	// not count it. Otherwise, we would run into issues if the visiblity
	// of some functions was changed from protected to public in a child
	// class with the using directive.
	continue;
      }
      ++base_overload_count;
    }

    return ((base_overload_count <= overridingOverloadCount(n)) &&
      areAllOverloadsOverridden(base_function));
  }

  /* ---------------------------------------------------------------------------
   * D::overridingOverloadCount()
   *
   * Given a member function node, this function counts how many of the
   * overloads of the function (including itself) override a function in the
   * base class.
   * --------------------------------------------------------------------------- */
  size_t overridingOverloadCount(Node *n) const {
    size_t result = 0;

    Node *tmp = firstSibling(n);
    do {
      // KLUDGE: We also have to count the function if the access attribute is
      // not present, since this means that it has been promoted into another
      // protection level in the base class with the C++ »using« directive, and
      // is thus taken into account when counting the base class overloads, even
      // if it is not marked as »override« by the SWIG parser.
      if (Getattr(n, "override") || !Getattr(n, "access")) {
	++result;
      }
    } while((tmp = Getattr(tmp, "sym:nextSibling")));

    return result;
  }

  /* ---------------------------------------------------------------------------
   * D::firstSibling()
   *
   * Returns the first sibling of the passed node.
   * --------------------------------------------------------------------------- */
  Node *firstSibling(Node *n) const {
    Node *result = n;
    while (Node *tmp = Getattr(result, "sym:previousSibling")) {
      result = tmp;
    }
    return result;
  }

  /* ---------------------------------------------------------------------------
   * D::indentCode()
   *
   * Helper function to indent a code (string) by one level.
   * --------------------------------------------------------------------------- */
  void indentCode(String* code) const {
    Replaceall(code, "\n", "\n  ");
    Replaceall(code, "  \n", "\n");
    Chop(code);
  }

  /* ---------------------------------------------------------------------------
   * D::emitBanner()
   * --------------------------------------------------------------------------- */
  void emitBanner(File *f) const {
    Printf(f, "/* ----------------------------------------------------------------------------\n");
    Swig_banner_target_lang(f, " *");
    Printf(f, " * ----------------------------------------------------------------------------- */\n\n");
  }

  /* ---------------------------------------------------------------------------
   * D::outputDirectory()
   *
   * Returns the directory to write the D modules for the given namespace to and
   * and creates the subdirectory if it doesn't exist.
   * --------------------------------------------------------------------------- */
  String *outputDirectory(String *nspace) {
    String *output_directory = Copy(dmodule_directory);
    if (nspace) {
      String *nspace_subdirectory = Copy(nspace);
      Replaceall(nspace_subdirectory, ".", SWIG_FILE_DELIMITER);
      String *newdir_error = Swig_new_subdirectory(output_directory, nspace_subdirectory);
      if (newdir_error) {
	Printf(stderr, "%s\n", newdir_error);
	Delete(newdir_error);
	SWIG_exit(EXIT_FAILURE);
      }
      Printv(output_directory, nspace_subdirectory, SWIG_FILE_DELIMITER, 0);
      Delete(nspace_subdirectory);
    }
    return output_directory;
  }

  /* ---------------------------------------------------------------------------
   * D::proxyCodeBuffer()
   *
   * Returns the buffer to write proxy code for the given namespace to.
   * --------------------------------------------------------------------------- */
  String *proxyCodeBuffer(String *nspace) {
    if (!nspace) {
      return proxy_dmodule_code;
    }

    Hash *hash = Getattr(nspace_proxy_dmodules, nspace);
    if (!hash) {
      hash = NewHash();
      Setattr(hash, "code", NewString(""));
      Setattr(hash, "imports", NewString(""));
      Setattr(nspace_proxy_dmodules, nspace, hash);
    }
    return Getattr(hash, "code");
  }

  /* ---------------------------------------------------------------------------
   * D::proxyCodeBuffer()
   *
   * Returns the buffer to write imports for the proxy code for the given
   * namespace to.
   * --------------------------------------------------------------------------- */
  String *proxyImportsBuffer(String *nspace) {
    if (!nspace) {
      return proxy_dmodule_imports;
    }

    Hash *hash = Getattr(nspace_proxy_dmodules, nspace);
    if (!hash) {
      hash = NewHash();
      Setattr(hash, "code", NewString(""));
      Setattr(hash, "imports", NewString(""));
      Setattr(nspace_proxy_dmodules, nspace, hash);
    }
    return Getattr(hash, "imports");
  }

  /* ---------------------------------------------------------------------------
   * D::createFirstNamespaceName()
   *
   * Returns a new string containing the name of the outermost namespace, e.g.
   * »A« for the argument »A.B.C«.
   * --------------------------------------------------------------------------- */
  String *createFirstNamespaceName(const String *nspace) const {
    char *tmp = Char(nspace);
    char *c = tmp;
    char *co = 0;
    if (!strstr(c, "."))
      return 0;

    co = c + Len(nspace);

    while (*c && (c != co)) {
      if (*c == '.') {
	break;
      }
      c++;
    }
    if (!*c || (c == tmp)) {
      return NULL;
    }
    return NewStringWithSize(tmp, (int)(c - tmp));
  }

  /* ---------------------------------------------------------------------------
   * D::createLastNamespaceName()
   *
   * Returns a new string containing the name of the innermost namespace, e.g.
   * »C« for the argument »A.B.C«.
   * --------------------------------------------------------------------------- */
  String *createLastNamespaceName(const String *nspace) const {
    if (!nspace) return NULL;
    char *c = Char(nspace);
    char *cc = c;
    if (!strstr(c, "."))
      return NewString(nspace);

    while (*c) {
      if (*c == '.') {
	cc = c;
      }
      ++c;
    }
    return NewString(cc + 1);
  }

  /* ---------------------------------------------------------------------------
   * D::createOuterNamespaceNames()
   *
   * Returns a new string containing the name of the outer namespace, e.g.
   * »A.B« for the argument »A.B.C«.
   * --------------------------------------------------------------------------- */
  String *createOuterNamespaceNames(const String *nspace) const {
    if (!nspace) return NULL;
    char *tmp = Char(nspace);
    char *c = tmp;
    char *cc = c;
    if (!strstr(c, "."))
      return NULL;

    while (*c) {
      if (*c == '.') {
	cc = c;
      }
      ++c;
    }
    if (cc == tmp) {
      return NULL;
    }
    return NewStringWithSize(tmp, (int)(cc - tmp));
  }
};

static Language *new_swig_d() {
  return new D();
}

/* -----------------------------------------------------------------------------
 * swig_d()    - Instantiate module
 * ----------------------------------------------------------------------------- */
extern "C" Language *swig_d(void) {
  return new_swig_d();
}

/* -----------------------------------------------------------------------------
 * Usage information displayed at the command line.
 * ----------------------------------------------------------------------------- */
const char *D::usage = "\
D Options (available with -d)\n\
     -d2                  - Generate code for D2/Phobos (default: D1/Tango)\n\
     -package <pkg>       - Write generated D modules into package <pkg>\n\
     -splitproxy          - Write each D type to a dedicated file instead of\n\
                            generating a single proxy D module.\n\
     -wrapperlibrary <wl> - Set the name of the wrapper library to <wl>\n\
\n";
