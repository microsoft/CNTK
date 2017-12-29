/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * csharp.cxx
 *
 * C# language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include <limits.h>		// for INT_MAX
#include "cparse.h"
#include <ctype.h>

/* Hash type used for upcalls from C/C++ */
typedef DOH UpcallData;

class CSHARP:public Language {
  static const char *usage;
  const String *empty_string;
  const String *public_string;
  const String *protected_string;

  Hash *swig_types_hash;
  File *f_begin;
  File *f_runtime;
  File *f_runtime_h;
  File *f_header;
  File *f_wrappers;
  File *f_init;
  File *f_directors;
  File *f_directors_h;
  File *f_single_out;
  List *filenames_list;

  bool proxy_flag;		// Flag for generating proxy classes
  bool native_function_flag;	// Flag for when wrapping a native function
  bool enum_constant_flag;	// Flag for when wrapping an enum or constant
  bool static_flag;		// Flag for when wrapping a static functions or member variables
  bool variable_wrapper_flag;	// Flag for when wrapping a nonstatic member variable
  bool wrapping_member_flag;	// Flag for when wrapping a member variable/enum/const
  bool global_variable_flag;	// Flag for when wrapping a global variable
  bool old_variable_names;	// Flag for old style variable names in the intermediary class
  bool generate_property_declaration_flag;	// Flag for generating properties

  String *imclass_name;		// intermediary class name
  String *module_class_name;	// module class name
  String *imclass_class_code;	// intermediary class code
  String *proxy_class_def;
  String *proxy_class_code;
  String *interface_class_code; // if %feature("interface") was declared for a class, here goes the interface declaration
  String *module_class_code;
  String *proxy_class_name;	// proxy class name
  String *full_imclass_name;	// fully qualified intermediary class name when using nspace feature, otherwise same as imclass_name
  String *variable_name;	//Name of a variable being wrapped
  String *proxy_class_constants_code;
  String *module_class_constants_code;
  String *enum_code;
  String *dllimport;		// DllImport attribute name
  String *namespce;		// Optional namespace name
  String *imclass_imports;	//intermediary class imports from %pragma
  String *module_imports;	//module imports from %pragma
  String *imclass_baseclass;	//inheritance for intermediary class class from %pragma
  String *module_baseclass;	//inheritance for module class from %pragma
  String *imclass_interfaces;	//interfaces for intermediary class class from %pragma
  String *module_interfaces;	//interfaces for module class from %pragma
  String *imclass_class_modifiers;	//class modifiers for intermediary class overriden by %pragma
  String *module_class_modifiers;	//class modifiers for module class overriden by %pragma
  String *upcasts_code;		//C++ casts for inheritance hierarchies C++ code
  String *imclass_cppcasts_code;	//C++ casts up inheritance hierarchies intermediary class code
  String *director_callback_typedefs;	// Director function pointer typedefs for callbacks
  String *director_callbacks;	// Director callback function pointer member variables
  String *director_delegate_callback;	// Director callback method that delegates are set to call
  String *director_delegate_definitions;	// Director delegates definitions in proxy class
  String *director_delegate_instances;	// Director delegates member variables in proxy class
  String *director_method_types;	// Director method types
  String *director_connect_parms;	// Director delegates parameter list for director connect call
  String *destructor_call;	//C++ destructor call if any
  String *output_file;		// File name for single file mode. If set all generated code will be written to this file

  // Director method stuff:
  List *dmethods_seq;
  Hash *dmethods_table;
  int n_dmethods;
  int n_directors;
  int first_class_dmethod;
  int curr_class_dmethod;
  int nesting_depth;

  enum EnumFeature { SimpleEnum, TypeunsafeEnum, TypesafeEnum, ProperEnum };

public:

  /* -----------------------------------------------------------------------------
   * CSHARP()
   * ----------------------------------------------------------------------------- */

   CSHARP():empty_string(NewString("")),
      public_string(NewString("public")),
      protected_string(NewString("protected")),
      swig_types_hash(NULL),
      f_begin(NULL),
      f_runtime(NULL),
      f_runtime_h(NULL),
      f_header(NULL),
      f_wrappers(NULL),
      f_init(NULL),
      f_directors(NULL),
      f_directors_h(NULL),
      f_single_out(NULL),
      filenames_list(NULL),
      proxy_flag(true),
      native_function_flag(false),
      enum_constant_flag(false),
      static_flag(false),
      variable_wrapper_flag(false),
      wrapping_member_flag(false),
      global_variable_flag(false),
      old_variable_names(false),
      generate_property_declaration_flag(false),
      imclass_name(NULL),
      module_class_name(NULL),
      imclass_class_code(NULL),
      proxy_class_def(NULL),
      proxy_class_code(NULL),
      interface_class_code(NULL),
      module_class_code(NULL),
      proxy_class_name(NULL),
      full_imclass_name(NULL),
      variable_name(NULL),
      proxy_class_constants_code(NULL),
      module_class_constants_code(NULL),
      enum_code(NULL),
      dllimport(NULL),
      namespce(NULL),
      imclass_imports(NULL),
      module_imports(NULL),
      imclass_baseclass(NULL),
      module_baseclass(NULL),
      imclass_interfaces(NULL),
      module_interfaces(NULL),
      imclass_class_modifiers(NULL),
      module_class_modifiers(NULL),
      upcasts_code(NULL),
      imclass_cppcasts_code(NULL),
      director_callback_typedefs(NULL),
      director_callbacks(NULL),
      director_delegate_callback(NULL),
      director_delegate_definitions(NULL),
      director_delegate_instances(NULL),
      director_method_types(NULL),
      director_connect_parms(NULL),
      destructor_call(NULL),
      output_file(NULL),
      dmethods_seq(NULL),
      dmethods_table(NULL),
      n_dmethods(0),
      n_directors(0),
      first_class_dmethod(0),
      curr_class_dmethod(0),
      nesting_depth(0){
    /* for now, multiple inheritance in directors is disabled, this
       should be easy to implement though */
    director_multiple_inheritance = 0;
    director_language = 1;
  }

  /* -----------------------------------------------------------------------------
   * getProxyName()
   *
   * Test to see if a type corresponds to something wrapped with a proxy class.
   * Return NULL if not otherwise the proxy class name, fully qualified with
   * a namespace if the nspace feature is used.
   * ----------------------------------------------------------------------------- */
  
   String *getProxyName(SwigType *t) {
     String *proxyname = NULL;
     if (proxy_flag) {
       Node *n = classLookup(t);
       if (n) {
	 proxyname = Getattr(n, "proxyname");
	 if (!proxyname) {
	   String *nspace = Getattr(n, "sym:nspace");
	   String *symname = Copy(Getattr(n, "sym:name"));
	   if (symname && !GetFlag(n, "feature:flatnested")) {
	     for (Node *outer_class = Getattr(n, "nested:outer"); outer_class; outer_class = Getattr(outer_class, "nested:outer")) {
               if (String* name = Getattr(outer_class, "sym:name")) {
                 Push(symname, ".");
                 Push(symname, name);
               }
               else
                 return NULL;
	     }
	   }
	   if (nspace) {
	     if (namespce)
	       proxyname = NewStringf("%s.%s.%s", namespce, nspace, symname);
	     else
	       proxyname = NewStringf("%s.%s", nspace, symname);
	   } else {
	     proxyname = Copy(symname);
	   }
	   Setattr(n, "proxyname", proxyname);
	   Delete(proxyname);
	   Delete(symname);
	 }
       }
     }
     return proxyname;
   }

  /* ------------------------------------------------------------
   * main()
   * ------------------------------------------------------------ */

  virtual void main(int argc, char *argv[]) {

    SWIG_library_directory("csharp");

    // Look for certain command line options
    for (int i = 1; i < argc; i++) {
      if (argv[i]) {
	if (strcmp(argv[i], "-dllimport") == 0) {
	  if (argv[i + 1]) {
	    dllimport = NewString("");
	    Printf(dllimport, argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-namespace") == 0) {
	  if (argv[i + 1]) {
	    namespce = NewString("");
	    Printf(namespce, argv[i + 1]);
	    if (Len(namespce) == 0) {
	      Delete(namespce);
	      namespce = 0;
	    }
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if ((strcmp(argv[i], "-noproxy") == 0)) {
	  Swig_mark_arg(i);
	  proxy_flag = false;
	} else if (strcmp(argv[i], "-oldvarnames") == 0) {
	  Swig_mark_arg(i);
	  old_variable_names = true;
	} else if (strcmp(argv[i], "-outfile") == 0) {
	  if (argv[i + 1]) {
	    output_file = NewString("");
	    Printf(output_file, argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-help") == 0) {
	  Printf(stdout, "%s\n", usage);
	}
      }
    }

    // Add a symbol to the parser for conditional compilation
    Preprocessor_define("SWIGCSHARP 1", 0);

    // Add typemap definitions
    SWIG_typemap_lang("csharp");
    SWIG_config_file("csharp.swg");

    allow_overloading();
    Swig_interface_feature_enable();
  }

  /* ---------------------------------------------------------------------
   * top()
   * --------------------------------------------------------------------- */

  virtual int top(Node *n) {

    // Get any options set in the module directive
    Node *optionsnode = Getattr(Getattr(n, "module"), "options");

    if (optionsnode) {
      if (Getattr(optionsnode, "imclassname"))
	imclass_name = Copy(Getattr(optionsnode, "imclassname"));
      /* check if directors are enabled for this module.  note: this 
       * is a "master" switch, without which no director code will be
       * emitted.  %feature("director") statements are also required
       * to enable directors for individual classes or methods.
       *
       * use %module(directors="1") modulename at the start of the 
       * interface file to enable director generation.
       */
      if (Getattr(optionsnode, "directors")) {
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

    swig_types_hash = NewHash();
    filenames_list = NewList();

    // Make the intermediary class and module class names. The intermediary class name can be set in the module directive.
    if (!imclass_name) {
      imclass_name = NewStringf("%sPINVOKE", Getattr(n, "name"));
      module_class_name = Copy(Getattr(n, "name"));
    } else {
      // Rename the module name if it is the same as intermediary class name - a backwards compatibility solution
      if (Cmp(imclass_name, Getattr(n, "name")) == 0)
	module_class_name = NewStringf("%sModule", Getattr(n, "name"));
      else
	module_class_name = Copy(Getattr(n, "name"));
    }

    // module class and intermediary classes are always created
    if (!addSymbol(imclass_name, n))
      return SWIG_ERROR;
    if (!addSymbol(module_class_name, n))
      return SWIG_ERROR;

    imclass_class_code = NewString("");
    proxy_class_def = NewString("");
    proxy_class_code = NewString("");
    module_class_constants_code = NewString("");
    imclass_baseclass = NewString("");
    imclass_interfaces = NewString("");
    imclass_class_modifiers = NewString("");
    module_class_code = NewString("");
    module_baseclass = NewString("");
    module_interfaces = NewString("");
    module_imports = NewString("");
    module_class_modifiers = NewString("");
    imclass_imports = NewString("");
    imclass_cppcasts_code = NewString("");
    director_connect_parms = NewString("");
    upcasts_code = NewString("");
    dmethods_seq = NewList();
    dmethods_table = NewHash();
    n_dmethods = 0;
    n_directors = 0;
    if (!dllimport)
      dllimport = Copy(module_class_name);

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGCSHARP\n#define SWIGCSHARP\n#endif\n\n");

    if (directorsEnabled()) {
      Printf(f_runtime, "#define SWIG_DIRECTORS\n");

      /* Emit initial director header and director code: */
      Swig_banner(f_directors_h);
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "#ifndef SWIG_%s_WRAP_H_\n", module_class_name);
      Printf(f_directors_h, "#define SWIG_%s_WRAP_H_\n\n", module_class_name);

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
    if (namespce) {
      String *wrapper_name = NewStringf("");
      Printf(wrapper_name, "CSharp_%s_%%f", namespce);
      Swig_name_register("wrapper", wrapper_name);
      Delete(wrapper_name);
    }
    else {
      Swig_name_register("wrapper", "CSharp_%f");
    }

    if (old_variable_names) {
      Swig_name_register("set", "set_%n%v");
      Swig_name_register("get", "get_%n%v");
    }

    Printf(f_wrappers, "\n#ifdef __cplusplus\n");
    Printf(f_wrappers, "extern \"C\" {\n");
    Printf(f_wrappers, "#endif\n\n");

    /* Emit code */
    Language::top(n);

    if (directorsEnabled()) {
      // Insert director runtime into the f_runtime file (make it occur before %header section)
      Swig_insert_file("director_common.swg", f_runtime);
      Swig_insert_file("director.swg", f_runtime);
    }
    // Generate the intermediary class
    {
      File *f_im = getOutputFile(SWIG_output_directory(), imclass_name);

      addOpenNamespace(0, f_im);

      if (imclass_imports)
	Printf(f_im, "%s\n", imclass_imports);

      if (Len(imclass_class_modifiers) > 0)
	Printf(f_im, "%s ", imclass_class_modifiers);
      Printf(f_im, "%s ", imclass_name);

      if (imclass_baseclass && *Char(imclass_baseclass))
	Printf(f_im, ": %s ", imclass_baseclass);
      if (Len(imclass_interfaces) > 0)
	Printv(f_im, "implements ", imclass_interfaces, " ", NIL);
      Printf(f_im, "{\n");

      // Add the intermediary class methods
      Replaceall(imclass_class_code, "$module", module_class_name);
      Replaceall(imclass_class_code, "$imclassname", imclass_name);
      Replaceall(imclass_class_code, "$dllimport", dllimport);
      Printv(f_im, imclass_class_code, NIL);
      Printv(f_im, imclass_cppcasts_code, NIL);

      // Finish off the class
      Printf(f_im, "}\n");
      addCloseNamespace(0, f_im);

      if (f_im != f_single_out)
	Delete(f_im);
      f_im = NULL;
    }

    // Generate the C# module class
    {
      File *f_module = getOutputFile(SWIG_output_directory(), module_class_name);

      addOpenNamespace(0, f_module);

      if (module_imports)
	Printf(f_module, "%s\n", module_imports);

      if (Len(module_class_modifiers) > 0)
	Printf(f_module, "%s ", module_class_modifiers);
      Printf(f_module, "%s ", module_class_name);

      if (module_baseclass && *Char(module_baseclass))
	Printf(f_module, ": %s ", module_baseclass);
      if (Len(module_interfaces) > 0)
	Printv(f_module, "implements ", module_interfaces, " ", NIL);
      Printf(f_module, "{\n");

      Replaceall(module_class_code, "$module", module_class_name);
      Replaceall(module_class_constants_code, "$module", module_class_name);

      Replaceall(module_class_code, "$imclassname", imclass_name);
      Replaceall(module_class_constants_code, "$imclassname", imclass_name);

      Replaceall(module_class_code, "$dllimport", dllimport);
      Replaceall(module_class_constants_code, "$dllimport", dllimport);

      // Add the wrapper methods
      Printv(f_module, module_class_code, NIL);

      // Write out all the global constants
      Printv(f_module, module_class_constants_code, NIL);

      // Finish off the class
      Printf(f_module, "}\n");
      addCloseNamespace(0, f_module);

      if (f_module != f_single_out)
	Delete(f_module);
      f_module = NULL;
    }

    if (upcasts_code)
      Printv(f_wrappers, upcasts_code, NIL);

    Printf(f_wrappers, "#ifdef __cplusplus\n");
    Printf(f_wrappers, "}\n");
    Printf(f_wrappers, "#endif\n");

    // Output a C# type wrapper class for each SWIG type
    for (Iterator swig_type = First(swig_types_hash); swig_type.key; swig_type = Next(swig_type)) {
      emitTypeWrapperClass(swig_type.key, swig_type.item);
    }

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

    Delete(swig_types_hash);
    swig_types_hash = NULL;
    Delete(filenames_list);
    filenames_list = NULL;
    Delete(imclass_name);
    imclass_name = NULL;
    Delete(imclass_class_code);
    imclass_class_code = NULL;
    Delete(proxy_class_def);
    proxy_class_def = NULL;
    Delete(proxy_class_code);
    proxy_class_code = NULL;
    Delete(module_class_constants_code);
    module_class_constants_code = NULL;
    Delete(imclass_baseclass);
    imclass_baseclass = NULL;
    Delete(imclass_interfaces);
    imclass_interfaces = NULL;
    Delete(imclass_class_modifiers);
    imclass_class_modifiers = NULL;
    Delete(module_class_name);
    module_class_name = NULL;
    Delete(module_class_code);
    module_class_code = NULL;
    Delete(module_baseclass);
    module_baseclass = NULL;
    Delete(module_interfaces);
    module_interfaces = NULL;
    Delete(module_imports);
    module_imports = NULL;
    Delete(module_class_modifiers);
    module_class_modifiers = NULL;
    Delete(imclass_imports);
    imclass_imports = NULL;
    Delete(imclass_cppcasts_code);
    imclass_cppcasts_code = NULL;
    Delete(upcasts_code);
    upcasts_code = NULL;
    Delete(dmethods_seq);
    dmethods_seq = NULL;
    Delete(dmethods_table);
    dmethods_table = NULL;
    Delete(namespce);
    namespce = NULL;
    n_dmethods = 0;

    /* Close all of the files */
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

    if (f_single_out) {
      Dump(f_single_out, f_begin);
      Delete(f_single_out);
      f_single_out = NULL;
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

  /* -----------------------------------------------------------------------------
   * emitBanner()
   * ----------------------------------------------------------------------------- */

  void emitBanner(File *f) {
    Printf(f, "//------------------------------------------------------------------------------\n");
    Printf(f, "// <auto-generated />\n");
    Printf(f, "//\n");
    Swig_banner_target_lang(f, "//");
    Printf(f, "//------------------------------------------------------------------------------\n\n");
  }

  /* -----------------------------------------------------------------------------
   * getOutputFile()
   *
   * Prepares a File object by creating the file in the file system and
   * writing the banner for auto-generated files to it (emitBanner).
   * If '-outfile' is provided (single file mode) the supplied parameters will
   * be ignored and the returned file will always be:
   *  <outdir>/<outfile>
   * Otherwise the file will be:
   *  <dir>/<name>.cs
   * ----------------------------------------------------------------------------- */

  File *getOutputFile(const String *dir, const String *name) {
    if (output_file) {
      if (!f_single_out) {
	String *filen = NewStringf("%s%s", SWIG_output_directory(), output_file);
	f_single_out = NewFile(filen, "w", SWIG_output_files());
	if (!f_single_out) {
	  FileErrorDisplay(filen);
	  SWIG_exit(EXIT_FAILURE);
	}
	Append(filenames_list, Copy(filen));
	Delete(filen);
	filen = NULL;

	emitBanner(f_single_out);
      }
      return f_single_out;
    } else {
      String *filen = NewStringf("%s%s.cs", dir, name);
      File *f = NewFile(filen, "w", SWIG_output_files());
      if (!f) {
	FileErrorDisplay(filen);
	SWIG_exit(EXIT_FAILURE);
      }
      Append(filenames_list, Copy(filen));
      Delete(filen);
      filen = NULL;

      emitBanner(f);
      return f;
    }
  }

  /*-----------------------------------------------------------------------
   * Add new director upcall signature
   *----------------------------------------------------------------------*/

  UpcallData *addUpcallMethod(String *imclass_method, String *class_method, String *decl, String *overloaded_name) {
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

    Delete(key);
    return new_udata;
  }

  /*-----------------------------------------------------------------------
   * Get director upcall signature
   *----------------------------------------------------------------------*/

  /*
     UpcallData * getUpcallMethodData(String *director_class, String *decl) {
     String             *key = NewStringf("%s|%s", director_class, decl);
     UpcallData         *udata = Getattr(dmethods_table, key);

     Delete(key);
     return udata;
     }
   */

  /* ----------------------------------------------------------------------
   * nativeWrapper()
   * ---------------------------------------------------------------------- */

  virtual int nativeWrapper(Node *n) {
    String *wrapname = Getattr(n, "wrap:name");

    if (!addSymbol(wrapname, n, imclass_name))
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

  /* ----------------------------------------------------------------------
   * functionWrapper()
   * ---------------------------------------------------------------------- */

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
    String *im_outattributes = 0;
    int num_arguments = 0;
    bool is_void_return;
    String *overloaded_name = getOverloadedName(n);

    if (!Getattr(n, "sym:overloaded")) {
      if (!addSymbol(symname, n, imclass_name))
	return SWIG_ERROR;
    }

    /*
       The rest of this function deals with generating the intermediary class wrapper function (that wraps
       a c/c++ function) and generating the PInvoke c code. Each C# wrapper function has a 
       matching PInvoke c function call.
     */

    // A new wrapper function object
    Wrapper *f = NewWrapper();

    // Make a wrapper name for this function
    String *wname = Swig_name_wrapper(overloaded_name);

    /* Attach the non-standard typemaps to the parameter list. */
    Swig_typemap_attach_parms("ctype", l, f);
    Swig_typemap_attach_parms("imtype", l, f);

    /* Get return types */
    if ((tm = Swig_typemap_lookup("ctype", n, "", 0))) {
      String *ctypeout = Getattr(n, "tmap:ctype:out");	// the type in the ctype typemap's out attribute overrides the type in the typemap
      if (ctypeout)
	tm = ctypeout;
      Printf(c_return_type, "%s", tm);
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CTYPE_UNDEF, input_file, line_number, "No ctype typemap defined for %s\n", SwigType_str(t, 0));
    }

    if ((tm = Swig_typemap_lookup("imtype", n, "", 0))) {
      String *imtypeout = Getattr(n, "tmap:imtype:out");	// the type in the imtype typemap's out attribute overrides the type in the typemap
      if (imtypeout)
	tm = imtypeout;
      Printf(im_return_type, "%s", tm);
      im_outattributes = Getattr(n, "tmap:imtype:outattributes");
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CSTYPE_UNDEF, input_file, line_number, "No imtype typemap defined for %s\n", SwigType_str(t, 0));
    }

    is_void_return = (Cmp(c_return_type, "void") == 0);
    if (!is_void_return)
      Wrapper_add_localv(f, "jresult", c_return_type, "jresult", NIL);

    Printv(f->def, " SWIGEXPORT ", c_return_type, " SWIGSTDCALL ", wname, "(", NIL);

    // Emit all of the local variables for holding arguments.
    emit_parameter_variables(l, f);

    /* Attach the standard typemaps */
    emit_attach_parmmaps(l, f);

    // Parameter overloading
    Setattr(n, "wrap:parms", l);
    Setattr(n, "wrap:name", wname);

    // Wrappers not wanted for some methods where the parameters cannot be overloaded in C#
    if (Getattr(n, "sym:overloaded")) {
      // Emit warnings for the few cases that can't be overloaded in C# and give up on generating wrapper
      Swig_overload_check(n);
      if (Getattr(n, "overload:ignore")) {
	DelWrapper(f);
	return SWIG_OK;
      }
    }

    Printv(imclass_class_code, "\n  [global::System.Runtime.InteropServices.DllImport(\"", dllimport, "\", EntryPoint=\"", wname, "\")]\n", NIL);

    if (im_outattributes)
      Printf(imclass_class_code, "  %s\n", im_outattributes);

    Printf(imclass_class_code, "  public static extern %s %s(", im_return_type, overloaded_name);


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
      if ((tm = Getattr(p, "tmap:ctype"))) {
	Printv(c_param_type, tm, NIL);
      } else {
	Swig_warning(WARN_CSHARP_TYPEMAP_CTYPE_UNDEF, input_file, line_number, "No ctype typemap defined for %s\n", SwigType_str(pt, 0));
      }

      /* Get the intermediary class parameter types of the parameter */
      if ((tm = Getattr(p, "tmap:imtype"))) {
	const String *inattributes = Getattr(p, "tmap:imtype:inattributes");
	Printf(im_param_type, "%s%s", inattributes ? inattributes : empty_string, tm);
      } else {
	Swig_warning(WARN_CSHARP_TYPEMAP_CSTYPE_UNDEF, input_file, line_number, "No imtype typemap defined for %s\n", SwigType_str(pt, 0));
      }

      /* Add parameter to intermediary class method */
      if (gencomma)
	Printf(imclass_class_code, ", ");
      Printf(imclass_class_code, "%s %s", im_param_type, arg);

      // Add parameter to C function
      Printv(f->def, gencomma ? ", " : "", c_param_type, " ", arg, NIL);

      gencomma = 1;

      // Get typemap for this argument
      if ((tm = Getattr(p, "tmap:in"))) {
	canThrow(n, "in", p);
	Replaceall(tm, "$source", arg);	/* deprecated */
	Replaceall(tm, "$target", ln);	/* deprecated */
	Replaceall(tm, "$arg", arg);	/* deprecated? */
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
	Replaceall(tm, "$target", Getattr(p, "lname"));	/* deprecated */
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));	/* deprecated? */
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
	Replaceall(tm, "$source", Getattr(p, "emit:input"));	/* deprecated */
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));	/* deprecated? */
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
	Replaceall(tm, "$source", Getattr(p, "emit:input"));	/* deprecated */
	Replaceall(tm, "$target", Getattr(p, "lname"));	/* deprecated */
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));	/* deprecated? */
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
	Replaceall(tm, "$source", Swig_cresult_name());	/* deprecated */
	Replaceall(tm, "$target", "jresult");	/* deprecated */
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
	Replaceall(tm, "$source", Swig_cresult_name());	/* deprecated */
	Printf(f->code, "%s\n", tm);
      }
    }

    /* See if there is any return cleanup code */
    if (!native_function_flag) {
      if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
	canThrow(n, "ret", n);
	Replaceall(tm, "$source", Swig_cresult_name());	/* deprecated */
	Printf(f->code, "%s\n", tm);
      }
    }

    /* Finish C function and intermediary class function definitions */
    Printf(imclass_class_code, ")");
    Printf(imclass_class_code, ";\n");

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
      Setattr(n, "csharp:canthrow", "1");
    }

    if (!null_attribute)
      Replaceall(f->code, "$null", "0");
    else
      Replaceall(f->code, "$null", null_attribute);

    /* Dump the function out */
    if (!native_function_flag) {
      Wrapper_print(f, f_wrappers);

      // Handle %csexception which sets the canthrow attribute
      if (Getattr(n, "feature:except:canthrow"))
	Setattr(n, "csharp:canthrow", "1");

      // A very simple check (it is not foolproof) to help typemap/feature writers for
      // throwing C# exceptions from unmanaged code. It checks for the common methods which
      // set a pending C# exception... the 'canthrow' typemap/feature attribute must be set
      // so that code which checks for pending exceptions is added in the C# proxy method.
      if (!Getattr(n, "csharp:canthrow")) {
	if (Strstr(f->code, "SWIG_exception")) {
	  Swig_warning(WARN_CSHARP_CANTHROW, input_file, line_number,
		       "Unmanaged code contains a call to SWIG_exception and C# code does not handle pending exceptions via the canthrow attribute.\n");
	} else if (Strstr(f->code, "SWIG_CSharpSetPendingException")) {
	  Swig_warning(WARN_CSHARP_CANTHROW, input_file, line_number,
		       "Unmanaged code contains a call to a SWIG_CSharpSetPendingException method and C# code does not handle pending exceptions via the canthrow attribute.\n");
	}
      }
    }

    if (!(proxy_flag && is_wrapping_class()) && !enum_constant_flag) {
      moduleClassFunctionHandler(n);
    }

    /* 
     * Generate the proxy class properties for public member variables.
     * Not for enums and constants.
     */
    if (proxy_flag && wrapping_member_flag && !enum_constant_flag) {
      // Capitalize the first letter in the variable in the getter/setter function name
      bool getter_flag = Cmp(symname, Swig_name_set(getNSpace(), Swig_name_member(0, getClassPrefix(), variable_name))) != 0;

      String *getter_setter_name = NewString("");
      if (!getter_flag)
	Printf(getter_setter_name, "set");
      else
	Printf(getter_setter_name, "get");
      Putc(toupper((int) *Char(variable_name)), getter_setter_name);
      Printf(getter_setter_name, "%s", Char(variable_name) + 1);

      Setattr(n, "proxyfuncname", getter_setter_name);
      Setattr(n, "imfuncname", symname);

      proxyClassFunctionHandler(n);
      Delete(getter_setter_name);
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

  /* -----------------------------------------------------------------------
   * variableWrapper()
   * ----------------------------------------------------------------------- */

  virtual int variableWrapper(Node *n) {
    Language::variableWrapper(n);
    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------
   * globalvariableHandler()
   * ------------------------------------------------------------------------ */

  virtual int globalvariableHandler(Node *n) {

    generate_property_declaration_flag = true;
    variable_name = Getattr(n, "sym:name");
    global_variable_flag = true;
    int ret = Language::globalvariableHandler(n);
    global_variable_flag = false;
    generate_property_declaration_flag = false;

    if (proxy_flag) {
      Printf(module_class_code, "\n  }\n\n");
    }

    return ret;
  }
  
  String *getCurrentScopeName(String *nspace) {
    String *scope = 0;
    if (nspace || getCurrentClass()) {
      scope = NewString("");
      if (nspace)
	Printf(scope, "%s", nspace);
      if (Node *cls = getCurrentClass()) {
	if (Node *outer = Getattr(cls, "nested:outer")) {
	  String *outerClassesPrefix = Copy(Getattr(outer, "sym:name"));
	  for (outer = Getattr(outer, "nested:outer"); outer != 0; outer = Getattr(outer, "nested:outer")) {
	    Push(outerClassesPrefix, ".");
	    Push(outerClassesPrefix, Getattr(outer, "sym:name"));
	  }
	  Printv(scope, nspace ? "." : "", outerClassesPrefix, ".", proxy_class_name, NIL);
	  Delete(outerClassesPrefix);
	} else
	  Printv(scope, nspace ? "." : "", proxy_class_name, NIL);
      }
    }
    return scope;
  }

  /* ----------------------------------------------------------------------
   * enumDeclaration()
   *
   * C/C++ enums can be mapped in one of 4 ways, depending on the cs:enum feature specified:
   * 1) Simple enums - simple constant within the proxy class or module class
   * 2) Typeunsafe enums - simple constant in a C# class (class named after the c++ enum name)
   * 3) Typesafe enum - typesafe enum pattern (class named after the c++ enum name)
   * 4) Proper enums - proper C# enum
   * Anonymous enums always default to 1)
   * ---------------------------------------------------------------------- */

  virtual int enumDeclaration(Node *n) {

    if (!ImportMode) {
      if (getCurrentClass() && (cplus_mode != PUBLIC))
	return SWIG_NOWRAP;

      String *nspace = Getattr(n, "sym:nspace"); // NSpace/getNSpace() only works during Language::enumDeclaration call
      if (proxy_flag && !is_wrapping_class()) {
	// Global enums / enums in a namespace
	assert(!full_imclass_name);

	if (!nspace) {
	  full_imclass_name = NewStringf("%s", imclass_name);
	} else {
	  if (namespce) {
	    full_imclass_name = NewStringf("%s.%s", namespce, imclass_name);
	  } else {
	    full_imclass_name = NewStringf("%s", imclass_name);
	  }
	}
      }

      enum_code = NewString("");
      String *symname = Getattr(n, "sym:name");
      String *constants_code = (proxy_flag && is_wrapping_class())? proxy_class_constants_code : module_class_constants_code;
      EnumFeature enum_feature = decodeEnumFeature(n);
      String *typemap_lookup_type = Getattr(n, "name");

      if ((enum_feature != SimpleEnum) && symname && typemap_lookup_type) {
	// Wrap (non-anonymous) C/C++ enum within a typesafe, typeunsafe or proper C# enum

	String *scope = getCurrentScopeName(nspace);
	if (!addSymbol(symname, n, scope))
	  return SWIG_ERROR;

	// Pure C# baseclass and interfaces
	const String *pure_baseclass = typemapLookup(n, "csbase", typemap_lookup_type, WARN_NONE);
	const String *pure_interfaces = typemapLookup(n, "csinterfaces", typemap_lookup_type, WARN_NONE);

	// Class attributes
	const String *csattributes = typemapLookup(n, "csattributes", typemap_lookup_type, WARN_NONE);
	if (csattributes && *Char(csattributes))
	  Printf(enum_code, "%s\n", csattributes);

	// Emit the enum
	Printv(enum_code, typemapLookup(n, "csclassmodifiers", typemap_lookup_type, WARN_CSHARP_TYPEMAP_CLASSMOD_UNDEF),	// Class modifiers (enum modifiers really)
	       " ", symname, (*Char(pure_baseclass) || *Char(pure_interfaces)) ? " : " : "", pure_baseclass, ((*Char(pure_baseclass)) && *Char(pure_interfaces)) ?	// Interfaces
	       ", " : "", pure_interfaces, " {\n", NIL);
	Delete(scope);
      } else {
	// Wrap C++ enum with integers - just indicate start of enum with a comment, no comment for anonymous enums of any sort
	if (symname && !Getattr(n, "unnamedinstance"))
	  Printf(constants_code, "  // %s \n", symname);
      }

      // Emit each enum item
      Language::enumDeclaration(n);

      if ((enum_feature != SimpleEnum) && symname && typemap_lookup_type) {
	// Wrap (non-anonymous) C/C++ enum within a typesafe, typeunsafe or proper C# enum
	// Finish the enum declaration
	// Typemaps are used to generate the enum definition in a similar manner to proxy classes.
	Printv(enum_code, (enum_feature == ProperEnum) ? "\n" : typemapLookup(n, "csbody", typemap_lookup_type, WARN_CSHARP_TYPEMAP_CSBODY_UNDEF),	// main body of class
	       typemapLookup(n, "cscode", typemap_lookup_type, WARN_NONE),	// extra C# code
	       "}", NIL);

	Replaceall(enum_code, "$csclassname", symname);

	// Substitute $enumvalues - intended usage is for typesafe enums
	if (Getattr(n, "enumvalues"))
	  Replaceall(enum_code, "$enumvalues", Getattr(n, "enumvalues"));
	else
	  Replaceall(enum_code, "$enumvalues", "");

	if (proxy_flag && is_wrapping_class()) {
	  // Enums defined within the C++ class are defined within the proxy class

	  // Add extra indentation
	  Replaceall(enum_code, "\n", "\n  ");
	  Replaceall(enum_code, "  \n", "\n");

	  Printv(proxy_class_constants_code, "  ", enum_code, "\n\n", NIL);
	} else {
	  // Global enums are defined in their own file
	  String *output_directory = outputDirectory(nspace);
	  File *f_enum = getOutputFile(output_directory, symname);

	  addOpenNamespace(nspace, f_enum);

	  Printv(f_enum, typemapLookup(n, "csimports", typemap_lookup_type, WARN_NONE), // Import statements
		 "\n", enum_code, "\n", NIL);

	  addCloseNamespace(nspace, f_enum);
	  if (f_enum != f_single_out)
	    Delete(f_enum);
	  f_enum = NULL;
	  Delete(output_directory);
	}
      } else {
	// Wrap C++ enum with simple constant
	Printf(enum_code, "\n");
	if (proxy_flag && is_wrapping_class())
	  Printv(proxy_class_constants_code, enum_code, NIL);
	else
	  Printv(module_class_constants_code, enum_code, NIL);
      }

      Delete(enum_code);
      enum_code = NULL;

      if (proxy_flag && !is_wrapping_class()) {
	Delete(full_imclass_name);
	full_imclass_name = 0;
      }
    }
    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * enumvalueDeclaration()
   * ---------------------------------------------------------------------- */

  virtual int enumvalueDeclaration(Node *n) {
    if (getCurrentClass() && (cplus_mode != PUBLIC))
      return SWIG_NOWRAP;

    Swig_require("enumvalueDeclaration", n, "*name", "?value", NIL);
    String *symname = Getattr(n, "sym:name");
    String *value = Getattr(n, "value");
    String *name = Getattr(n, "name");
    Node *parent = parentNode(n);
    int unnamedinstance = GetFlag(parent, "unnamedinstance");
    String *parent_name = Getattr(parent, "name");
    String *nspace = getNSpace();
    String *newsymname = 0;
    String *tmpValue;

    // Strange hack from parent method
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
      String *val = NewStringf("'%(hexescape)s'", Getattr(n, "enumvalue"));
      Setattr(n, "enumvalue", val);
      Delete(val);
    }

    {
      EnumFeature enum_feature = decodeEnumFeature(parent);

      if ((enum_feature == SimpleEnum) && GetFlag(parent, "scopedenum")) {
	newsymname = Swig_name_member(0, Getattr(parent, "sym:name"), symname);
	symname = newsymname;
      }

      // Add to language symbol table
      String *scope = 0;
      if (unnamedinstance || !parent_name || enum_feature == SimpleEnum) {
	String *enumClassPrefix = getEnumClassPrefix();
	if (enumClassPrefix) {
	  scope = NewString("");
	  if (nspace)
	    Printf(scope, "%s.", nspace);
	  Printf(scope, "%s", enumClassPrefix);
	} else {
	  scope = Copy(module_class_name);
	}
      } else {
	scope = getCurrentScopeName(nspace);
	if (!scope)
	  scope = Copy(Getattr(parent, "sym:name"));
	else
	  Printf(scope, ".%s", Getattr(parent, "sym:name"));
      }
      if (!addSymbol(symname, n, scope))
	return SWIG_ERROR;

      const String *csattributes = Getattr(n, "feature:cs:attributes");

      if ((enum_feature == ProperEnum) && parent_name && !unnamedinstance) {
	// Wrap (non-anonymous) C/C++ enum with a proper C# enum
	// Emit the enum item.
	if (!GetFlag(n, "firstenumitem"))
	  Printf(enum_code, ",\n");

	if (csattributes)
	  Printf(enum_code, "  %s\n", csattributes);

	Printf(enum_code, "  %s", symname);

	// Check for the %csconstvalue feature
	String *value = Getattr(n, "feature:cs:constvalue");

	// Note that the enum value must be a true constant and cannot be set from a PINVOKE call, thus no support for %csconst(0)
	value = value ? value : Getattr(n, "enumvalue");
	if (value) {
	  Printf(enum_code, " = %s", value);
	}
      } else {
	// Wrap C/C++ enums with constant integers or use the typesafe enum pattern
	SwigType *typemap_lookup_type = parent_name ? parent_name : NewString("enum ");
	Setattr(n, "type", typemap_lookup_type);
	const String *tm = typemapLookup(n, "cstype", typemap_lookup_type, WARN_CSHARP_TYPEMAP_CSTYPE_UNDEF);

	String *return_type = Copy(tm);
	substituteClassname(typemap_lookup_type, return_type);
        const String *methodmods = Getattr(n, "feature:cs:methodmodifiers");
        methodmods = methodmods ? methodmods : (is_public(n) ? public_string : protected_string);

	if (csattributes)
	  Printf(enum_code, "  %s\n", csattributes);

	if ((enum_feature == TypesafeEnum) && parent_name && !unnamedinstance) {
	  // Wrap (non-anonymous) enum using the typesafe enum pattern
	  if (Getattr(n, "enumvalue")) {
	    String *value = enumValue(n);
	    Printf(enum_code, "  %s static readonly %s %s = new %s(\"%s\", %s);\n", methodmods, return_type, symname, return_type, symname, value);
	    Delete(value);
	  } else {
	    Printf(enum_code, "  %s static readonly %s %s = new %s(\"%s\");\n", methodmods, return_type, symname, return_type, symname);
	  }
	} else {
	  // Simple integer constants
	  // Note these are always generated for anonymous enums, no matter what enum_feature is specified
	  // Code generated is the same for SimpleEnum and TypeunsafeEnum -> the class it is generated into is determined later

	  // The %csconst feature determines how the constant value is obtained
	  int const_feature_flag = GetFlag(n, "feature:cs:const");

	  const char *const_readonly = const_feature_flag ? "const" : "static readonly";
	  String *value = enumValue(n);
	  Printf(enum_code, "  %s %s %s %s = %s;\n", methodmods, const_readonly, return_type, symname, value);
	  Delete(value);
	}
	Delete(return_type);
      }

      // Add the enum value to the comma separated list being constructed in the enum declaration.
      String *enumvalues = Getattr(parent, "enumvalues");
      if (!enumvalues)
	Setattr(parent, "enumvalues", Copy(symname));
      else
	Printv(enumvalues, ", ", symname, NIL);
      Delete(scope);
    }

    Delete(newsymname);
    Delete(tmpValue);
    Swig_restore(n);
    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------
   * constantWrapper()
   * Used for wrapping constants - #define or %constant.
   * Also for inline initialised const static primitive type member variables (short, int, double, enums etc).
   * C# static const variables are generated for these.
   * If the %csconst(1) feature is used then the C constant value is used to initialise the C# const variable.
   * If not, a PINVOKE method is generated to get the C constant value for initialisation of the C# const variable.
   * However, if the %csconstvalue feature is used, it overrides all other ways to generate the initialisation.
   * Also note that this method might be called for wrapping enum items (when the enum is using %csconst(0)).
   * ------------------------------------------------------------------------ */

  virtual int constantWrapper(Node *n) {
    String *symname = Getattr(n, "sym:name");
    SwigType *t = Getattr(n, "type");
    SwigType *valuetype = Getattr(n, "valuetype");
    ParmList *l = Getattr(n, "parms");
    String *tm;
    String *return_type = NewString("");
    String *constants_code = NewString("");
    Swig_save("constantWrapper", n, "value", NIL);
    Swig_save("constantWrapper", n, "tmap:ctype:out", "tmap:imtype:out", "tmap:cstype:out", "tmap:out:null", "tmap:imtype:outattributes", "tmap:cstype:outattributes", NIL);

    bool is_enum_item = (Cmp(nodeType(n), "enumitem") == 0);

    const String *itemname = (proxy_flag && wrapping_member_flag) ? variable_name : symname;
    if (!is_enum_item) {
      String *scope = 0;
      if (proxy_class_name) {
	String *nspace = getNSpace();
	scope = NewString("");
	if (nspace)
	  Printf(scope, "%s.", nspace);
	Printf(scope, "%s", proxy_class_name);
      } else {
	scope = Copy(module_class_name);
      }
      if (!addSymbol(itemname, n, scope))
	return SWIG_ERROR;
      Delete(scope);
    }

    // The %csconst feature determines how the constant value is obtained
    int const_feature_flag = GetFlag(n, "feature:cs:const");

    /* Adjust the enum type for the Swig_typemap_lookup.
     * We want the same jstype typemap for all the enum items so we use the enum type (parent node). */
    if (is_enum_item) {
      t = Getattr(parentNode(n), "enumtype");
      Setattr(n, "type", t);
    }

    /* Attach the non-standard typemaps to the parameter list. */
    Swig_typemap_attach_parms("cstype", l, NULL);

    /* Get C# return types */
    bool classname_substituted_flag = false;

    if ((tm = Swig_typemap_lookup("cstype", n, "", 0))) {
      String *cstypeout = Getattr(n, "tmap:cstype:out");	// the type in the cstype typemap's out attribute overrides the type in the typemap
      if (cstypeout)
	tm = cstypeout;
      classname_substituted_flag = substituteClassname(t, tm);
      Printf(return_type, "%s", tm);
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CSWTYPE_UNDEF, input_file, line_number, "No cstype typemap defined for %s\n", SwigType_str(t, 0));
    }

    // Default (octal) escaping is no good - change to hex escaped value
    String *hexescaped_value = Getattr(n, "rawvalue") ? NewStringf("%(hexescape)s", Getattr(n, "rawvalue")) : 0;
    // Add the stripped quotes back in
    String *new_value = NewString("");
    if (SwigType_type(t) == T_STRING) {
      Printf(new_value, "\"%s\"", hexescaped_value ? hexescaped_value : Copy(Getattr(n, "value")));
      Setattr(n, "value", new_value);
    } else if (SwigType_type(t) == T_CHAR) {
      Printf(new_value, "\'%s\'", hexescaped_value ? hexescaped_value : Copy(Getattr(n, "value")));
      Setattr(n, "value", new_value);
    }

    const String *outattributes = Getattr(n, "tmap:cstype:outattributes");
    if (outattributes)
      Printf(constants_code, "  %s\n", outattributes);

    const String *methodmods = Getattr(n, "feature:cs:methodmodifiers");
    methodmods = methodmods ? methodmods : (is_public(n) ? public_string : protected_string);

    Printf(constants_code, "  %s %s %s %s = ", methodmods, (const_feature_flag ? "const" : "static readonly"), return_type, itemname);

    // Check for the %csconstvalue feature
    String *value = Getattr(n, "feature:cs:constvalue");

    if (value) {
      Printf(constants_code, "%s;\n", value);
    } else if (!const_feature_flag) {
      // Default enum and constant handling will work with any type of C constant and initialises the C# variable from C through a PINVOKE call.

      if (classname_substituted_flag) {
	if (SwigType_isenum(t)) {
	  // This handles wrapping of inline initialised const enum static member variables (not when wrapping enum items - ignored later on)
	  Printf(constants_code, "(%s)%s.%s();\n", return_type, full_imclass_name, Swig_name_get(getNSpace(), symname));
	} else {
	  // This handles function pointers using the %constant directive
	  Printf(constants_code, "new %s(%s.%s(), false);\n", return_type, full_imclass_name ? full_imclass_name : imclass_name, Swig_name_get(getNSpace(), symname));
	}
      } else {
	Printf(constants_code, "%s.%s();\n", full_imclass_name ? full_imclass_name : imclass_name, Swig_name_get(getNSpace(), symname));
      }

      // Each constant and enum value is wrapped with a separate PInvoke function call
      SetFlag(n, "feature:immutable");
      enum_constant_flag = true;
      variableWrapper(n);
      enum_constant_flag = false;
    } else {
      // Alternative constant handling will use the C syntax to make a true C# constant and hope that it compiles as C# code
      if (Getattr(n, "wrappedasconstant")) {
	if (SwigType_type(t) == T_CHAR) {
	  if (SwigType_type(valuetype) == T_CHAR)
	    Printf(constants_code, "\'%(hexescape)s\';\n", Getattr(n, "staticmembervariableHandler:value"));
	  else
	    Printf(constants_code, "(char)%s;\n", Getattr(n, "staticmembervariableHandler:value"));
	} else {
          Printf(constants_code, "%s;\n", Getattr(n, "staticmembervariableHandler:value"));
	}
      } else {
        Printf(constants_code, "%s;\n", Getattr(n, "value"));
      }
    }

    // Emit the generated code to appropriate place
    // Enums only emit the intermediate and PINVOKE methods, so no proxy or module class wrapper methods needed
    if (!is_enum_item) {
      if (proxy_flag && wrapping_member_flag)
	Printv(proxy_class_constants_code, constants_code, NIL);
      else
	Printv(module_class_constants_code, constants_code, NIL);
    }
    // Cleanup
    Swig_restore(n);
    Delete(new_value);
    Delete(return_type);
    Delete(constants_code);
    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------------
   * insertDirective()
   * ----------------------------------------------------------------------------- */

  virtual int insertDirective(Node *n) {
    int ret = SWIG_OK;
    String *code = Getattr(n, "code");
    String *section = Getattr(n, "section");
    Replaceall(code, "$module", module_class_name);
    Replaceall(code, "$imclassname", imclass_name);
    Replaceall(code, "$dllimport", dllimport);

    if (!ImportMode && (Cmp(section, "proxycode") == 0)) {
      if (proxy_class_code) {
	Swig_typemap_replace_embedded_typemap(code, n);
	int offset = Len(code) > 0 && *Char(code) == '\n' ? 1 : 0;
	Printv(proxy_class_code, Char(code) + offset, "\n", NIL);
      }
    } else {
      ret = Language::insertDirective(n);
    }
    return ret;
  }

  /* -----------------------------------------------------------------------------
   * pragmaDirective()
   *
   * Valid Pragmas:
   * imclassbase            - base (extends) for the intermediary class
   * imclassclassmodifiers  - class modifiers for the intermediary class
   * imclasscode            - text (C# code) is copied verbatim to the intermediary class
   * imclassimports         - import statements for the intermediary class
   * imclassinterfaces      - interface (implements) for the intermediary class
   *
   * modulebase              - base (extends) for the module class
   * moduleclassmodifiers    - class modifiers for the module class
   * modulecode              - text (C# code) is copied verbatim to the module class
   * moduleimports           - import statements for the module class
   * moduleinterfaces        - interface (implements) for the module class
   *
   * ----------------------------------------------------------------------------- */

  virtual int pragmaDirective(Node *n) {
    if (!ImportMode) {
      String *lang = Getattr(n, "lang");
      String *code = Getattr(n, "name");
      String *value = Getattr(n, "value");

      if (Strcmp(lang, "csharp") == 0) {

	String *strvalue = NewString(value);
	Replaceall(strvalue, "\\\"", "\"");

	if (Strcmp(code, "imclassbase") == 0) {
	  Delete(imclass_baseclass);
	  imclass_baseclass = Copy(strvalue);
	} else if (Strcmp(code, "imclassclassmodifiers") == 0) {
	  Delete(imclass_class_modifiers);
	  imclass_class_modifiers = Copy(strvalue);
	} else if (Strcmp(code, "imclasscode") == 0) {
	  Printf(imclass_class_code, "%s\n", strvalue);
	} else if (Strcmp(code, "imclassimports") == 0) {
	  Delete(imclass_imports);
	  imclass_imports = Copy(strvalue);
	} else if (Strcmp(code, "imclassinterfaces") == 0) {
	  Delete(imclass_interfaces);
	  imclass_interfaces = Copy(strvalue);
	} else if (Strcmp(code, "modulebase") == 0) {
	  Delete(module_baseclass);
	  module_baseclass = Copy(strvalue);
	} else if (Strcmp(code, "moduleclassmodifiers") == 0) {
	  Delete(module_class_modifiers);
	  module_class_modifiers = Copy(strvalue);
	} else if (Strcmp(code, "modulecode") == 0) {
	  Printf(module_class_code, "%s\n", strvalue);
	} else if (Strcmp(code, "moduleimports") == 0) {
	  Delete(module_imports);
	  module_imports = Copy(strvalue);
	} else if (Strcmp(code, "moduleinterfaces") == 0) {
	  Delete(module_interfaces);
	  module_interfaces = Copy(strvalue);
	} else {
	  Swig_error(input_file, line_number, "Unrecognized pragma.\n");
	}
	Delete(strvalue);
      }
    }
    return Language::pragmaDirective(n);
  }

  /* -----------------------------------------------------------------------------
   * getQualifiedInterfaceName()
   * ----------------------------------------------------------------------------- */

  String *getQualifiedInterfaceName(Node *n) {
    String *ret = Getattr(n, "interface:qname");
    if (!ret) {
      String *nspace = Getattr(n, "sym:nspace");
      String *interface_name = Getattr(n, "interface:name");
      if (nspace) {
	if (namespce)
	  ret = NewStringf("%s.%s.%s", namespce, nspace, interface_name);
	else
	  ret = NewStringf("%s.%s", nspace, interface_name);
      } else {
	ret = Copy(interface_name);
      }
      Setattr(n, "interface:qname", ret);
    }
    return ret;
  }

  /* -----------------------------------------------------------------------------
   * getInterfaceName()
   * ----------------------------------------------------------------------------- */

  String *getInterfaceName(SwigType *t, bool qualified) {
    String *interface_name = NULL;
    if (proxy_flag) {
      Node *n = classLookup(t);
      if (n && Getattr(n, "interface:name"))
	interface_name = qualified ? getQualifiedInterfaceName(n) : Getattr(n, "interface:name");
    }
    return interface_name;
  }

  /* -----------------------------------------------------------------------------
   * addInterfaceNameAndUpcasts()
   * ----------------------------------------------------------------------------- */

  void addInterfaceNameAndUpcasts(SwigType *smart, String *interface_list, String *interface_upcasts, Hash *base_list, String *c_classname) {
    List *keys = Keys(base_list);
    for (Iterator it = First(keys); it.item; it = Next(it)) {
      Node *base = Getattr(base_list, it.item);
      String *c_baseclass = SwigType_namestr(Getattr(base, "name"));
      String *interface_name = Getattr(base, "interface:name");
      if (Len(interface_list))
	Append(interface_list, ", ");
      Append(interface_list, interface_name);

      Node *attributes = NewHash();
      String *interface_code = Copy(typemapLookup(base, "csinterfacecode", Getattr(base, "classtypeobj"), WARN_CSHARP_TYPEMAP_INTERFACECODE_UNDEF, attributes));
      String *cptr_method_name = 0;
      if (interface_code) {
        Replaceall(interface_code, "$interfacename", interface_name);
	Printv(interface_upcasts, interface_code, NIL);
	cptr_method_name = Copy(Getattr(attributes, "tmap:csinterfacecode:cptrmethod"));
      }
      if (!cptr_method_name)
	cptr_method_name = NewStringf("%s_GetInterfaceCPtr", interface_name);
      Replaceall(cptr_method_name, ".", "_");
      Replaceall(cptr_method_name, "$interfacename", interface_name);

      String *upcast_method_name = Swig_name_member(getNSpace(), proxy_class_name, cptr_method_name);
      upcastsCode(smart, upcast_method_name, c_classname, c_baseclass);

      Delete(upcast_method_name);
      Delete(cptr_method_name);
      Delete(interface_code);
      Delete(c_baseclass);
    }
    Delete(keys);
  }

  /* -----------------------------------------------------------------------------
   * upcastsCode()
   *
   * Add code for C++ casting to base class
   * ----------------------------------------------------------------------------- */

  void upcastsCode(SwigType *smart, String *upcast_method_name, String *c_classname, String *c_baseclass) {
    String *wname = Swig_name_wrapper(upcast_method_name);

    Printv(imclass_cppcasts_code, "\n  [global::System.Runtime.InteropServices.DllImport(\"", dllimport, "\", EntryPoint=\"", wname, "\")]\n", NIL);
    Printf(imclass_cppcasts_code, "  public static extern global::System.IntPtr %s(global::System.IntPtr jarg1);\n", upcast_method_name);

    Replaceall(imclass_cppcasts_code, "$csclassname", proxy_class_name);

    if (smart) {
      SwigType *bsmart = Copy(smart);
      SwigType *rclassname = SwigType_typedef_resolve_all(c_classname);
      SwigType *rbaseclass = SwigType_typedef_resolve_all(c_baseclass);
      Replaceall(bsmart, rclassname, rbaseclass);
      Delete(rclassname);
      Delete(rbaseclass);
      String *smartnamestr = SwigType_namestr(smart);
      String *bsmartnamestr = SwigType_namestr(bsmart);
      Printv(upcasts_code,
	  "SWIGEXPORT ", bsmartnamestr, " * SWIGSTDCALL ", wname, "(", smartnamestr, " *jarg1) {\n",
	  "    return jarg1 ? new ", bsmartnamestr, "(*jarg1) : 0;\n"
	  "}\n", "\n", NIL);
      Delete(bsmartnamestr);
      Delete(smartnamestr);
      Delete(bsmart);
    } else {
      Printv(upcasts_code,
	  "SWIGEXPORT ", c_baseclass, " * SWIGSTDCALL ", wname, "(", c_classname, " *jarg1) {\n",
	  "    return (", c_baseclass, " *)jarg1;\n"
	  "}\n", "\n", NIL);
    }
    Delete(wname);
  }

  /* -----------------------------------------------------------------------------
   * emitProxyClassDefAndCPPCasts()
   * ----------------------------------------------------------------------------- */

  void emitProxyClassDefAndCPPCasts(Node *n) {
    String *c_classname = SwigType_namestr(Getattr(n, "name"));
    String *c_baseclass = NULL;
    String *baseclass = NULL;
    String *c_baseclassname = NULL;
    String *interface_list = NewStringEmpty();
    String *interface_upcasts = NewStringEmpty();
    SwigType *typemap_lookup_type = Getattr(n, "classtypeobj");
    bool feature_director = Swig_directorclass(n) ? true : false;
    bool has_outerclass = Getattr(n, "nested:outer") != 0 && !GetFlag(n, "feature:flatnested");
    SwigType *smart = Swig_cparse_smartptr(n);

    // Inheritance from pure C# classes
    Node *attributes = NewHash();
    const String *pure_baseclass = typemapLookup(n, "csbase", typemap_lookup_type, WARN_NONE, attributes);
    bool purebase_replace = GetFlag(attributes, "tmap:csbase:replace") ? true : false;
    bool purebase_notderived = GetFlag(attributes, "tmap:csbase:notderived") ? true : false;
    Delete(attributes);

    // C++ inheritance
    if (!purebase_replace) {
      List *baselist = Getattr(n, "bases");
      if (baselist) {
	Iterator base = First(baselist);
	while (base.item) {
	  if (!(GetFlag(base.item, "feature:ignore") || Getattr(base.item, "feature:interface"))) {
	    String *baseclassname = Getattr(base.item, "name");
	    if (!c_baseclassname) {
	      c_baseclassname = baseclassname;
	      baseclass = Copy(getProxyName(baseclassname));
	      if (baseclass)
		c_baseclass = SwigType_namestr(baseclassname);
	    } else {
	      /* Warn about multiple inheritance for additional base class(es) */
	      String *proxyclassname = Getattr(n, "classtypeobj");
	      Swig_warning(WARN_CSHARP_MULTIPLE_INHERITANCE, Getfile(n), Getline(n),
		  "Warning for %s, base %s ignored. Multiple inheritance is not supported in C#.\n", SwigType_namestr(proxyclassname), SwigType_namestr(baseclassname));
	    }
	  }
	  base = Next(base);
	}
      }
    }
    Hash *interface_bases = Getattr(n, "interface:bases");
    if (interface_bases)
      addInterfaceNameAndUpcasts(smart, interface_list, interface_upcasts, interface_bases, c_classname);

    bool derived = baseclass && getProxyName(c_baseclassname);
    if (derived && purebase_notderived)
      pure_baseclass = empty_string;
    const String *wanted_base = baseclass ? baseclass : pure_baseclass;

    if (purebase_replace) {
      wanted_base = pure_baseclass;
      derived = false;
      Delete(baseclass);
      baseclass = NULL;
      if (purebase_notderived)
        Swig_error(Getfile(n), Getline(n), "The csbase typemap for proxy %s must contain just one of the 'replace' or 'notderived' attributes.\n", typemap_lookup_type);
    } else if (Len(pure_baseclass) > 0 && Len(baseclass) > 0) {
      Swig_warning(WARN_CSHARP_MULTIPLE_INHERITANCE, Getfile(n), Getline(n),
		   "Warning for %s, base %s ignored. Multiple inheritance is not supported in C#. "
		   "Perhaps you need one of the 'replace' or 'notderived' attributes in the csbase typemap?\n", typemap_lookup_type, pure_baseclass);
    }

    // Pure C# interfaces
    const String *pure_interfaces = typemapLookup(n, derived ? "csinterfaces_derived" : "csinterfaces", typemap_lookup_type, WARN_NONE);
    if (*Char(interface_list) && *Char(pure_interfaces))
      Append(interface_list, ", ");
    Append(interface_list, pure_interfaces);
    // Start writing the proxy class
    if (!has_outerclass)
      Printv(proxy_class_def, typemapLookup(n, "csimports", typemap_lookup_type, WARN_NONE),	// Import statements
	   "\n", NIL);

    // Class attributes
    const String *csattributes = typemapLookup(n, "csattributes", typemap_lookup_type, WARN_NONE);
    if (csattributes && *Char(csattributes))
      Printf(proxy_class_def, "%s\n", csattributes);

    Printv(proxy_class_def, typemapLookup(n, "csclassmodifiers", typemap_lookup_type, WARN_CSHARP_TYPEMAP_CLASSMOD_UNDEF),	// Class modifiers
	   " $csclassname",	// Class name and base class
	   (*Char(wanted_base) || *Char(interface_list)) ? " : " : "", wanted_base, (*Char(wanted_base) && *Char(interface_list)) ?	// Interfaces
	   ", " : "", interface_list, " {", derived ? typemapLookup(n, "csbody_derived", typemap_lookup_type, WARN_CSHARP_TYPEMAP_CSBODY_UNDEF) :	// main body of class
	   typemapLookup(n, "csbody", typemap_lookup_type, WARN_CSHARP_TYPEMAP_CSBODY_UNDEF),	// main body of class
	   NIL);

    // C++ destructor is wrapped by the Dispose method
    // Note that the method name is specified in a typemap attribute called methodname
    String *destruct = NewString("");
    const String *tm = NULL;
    attributes = NewHash();
    String *destruct_methodname = NULL;
    String *destruct_methodmodifiers = NULL;
    if (derived) {
      tm = typemapLookup(n, "csdestruct_derived", typemap_lookup_type, WARN_NONE, attributes);
      destruct_methodname = Getattr(attributes, "tmap:csdestruct_derived:methodname");
      destruct_methodmodifiers = Getattr(attributes, "tmap:csdestruct_derived:methodmodifiers");
    } else {
      tm = typemapLookup(n, "csdestruct", typemap_lookup_type, WARN_NONE, attributes);
      destruct_methodname = Getattr(attributes, "tmap:csdestruct:methodname");
      destruct_methodmodifiers = Getattr(attributes, "tmap:csdestruct:methodmodifiers");
    }
    if (tm && *Char(tm)) {
      if (!destruct_methodname) {
	Swig_error(Getfile(n), Getline(n), "No methodname attribute defined in csdestruct%s typemap for %s\n", (derived ? "_derived" : ""), proxy_class_name);
      }
      if (!destruct_methodmodifiers) {
	Swig_error(Getfile(n), Getline(n),
		   "No methodmodifiers attribute defined in csdestruct%s typemap for %s.\n", (derived ? "_derived" : ""), proxy_class_name);
      }
    }
    // Emit the Finalize and Dispose methods
    if (tm) {
      // Finalize method
      if (*Char(destructor_call)) {
	Printv(proxy_class_def, typemapLookup(n, "csfinalize", typemap_lookup_type, WARN_NONE), NIL);
      }
      // Dispose method
      Printv(destruct, tm, NIL);
      if (*Char(destructor_call))
	Replaceall(destruct, "$imcall", destructor_call);
      else
	Replaceall(destruct, "$imcall", "throw new global::System.MethodAccessException(\"C++ destructor does not have public access\")");
      if (*Char(destruct))
	Printv(proxy_class_def, "\n  ", destruct_methodmodifiers, " ", derived ? "override" : "virtual", " void ", destruct_methodname, "() ", destruct, "\n",
	       NIL);
    }
    if (*Char(interface_upcasts))
      Printv(proxy_class_def, interface_upcasts, NIL);

    if (feature_director) {
      // Generate director connect method
      // put this in classDirectorEnd ???
      Printf(proxy_class_code, "  private void SwigDirectorConnect() {\n");

      int i;
      for (i = first_class_dmethod; i < curr_class_dmethod; ++i) {
	UpcallData *udata = Getitem(dmethods_seq, i);
	String *method = Getattr(udata, "method");
	String *methid = Getattr(udata, "class_methodidx");
	String *overname = Getattr(udata, "overname");
	Printf(proxy_class_code, "    if (SwigDerivedClassHasMethod(\"%s\", swigMethodTypes%s))\n", method, methid);
	Printf(proxy_class_code, "      swigDelegate%s = new SwigDelegate%s_%s(SwigDirector%s);\n", methid, proxy_class_name, methid, overname);
      }
      String *director_connect_method_name = Swig_name_member(getNSpace(), getClassPrefix(), "director_connect");
      Printf(proxy_class_code, "    %s.%s(swigCPtr", imclass_name, director_connect_method_name);
      for (i = first_class_dmethod; i < curr_class_dmethod; ++i) {
	UpcallData *udata = Getitem(dmethods_seq, i);
	String *methid = Getattr(udata, "class_methodidx");
	Printf(proxy_class_code, ", swigDelegate%s", methid);
      }
      Printf(proxy_class_code, ");\n");
      Printf(proxy_class_code, "  }\n");

      if (first_class_dmethod < curr_class_dmethod) {
	// Only emit if there is at least one director method
	Printf(proxy_class_code, "\n");
	Printf(proxy_class_code, "  private bool SwigDerivedClassHasMethod(string methodName, global::System.Type[] methodTypes) {\n");
	Printf(proxy_class_code,
	       "    global::System.Reflection.MethodInfo methodInfo = this.GetType().GetMethod(methodName, global::System.Reflection.BindingFlags.Public | global::System.Reflection.BindingFlags.NonPublic | global::System.Reflection.BindingFlags.Instance, null, methodTypes, null);\n");
	Printf(proxy_class_code, "    bool hasDerivedMethod = methodInfo.DeclaringType.IsSubclassOf(typeof(%s));\n", proxy_class_name);
	/* Could add this code to cover corner case where the GetMethod() returns a method which allows type
	 * promotion, eg it will return foo(double), if looking for foo(int).
	 if (hasDerivedMethod) {
	 hasDerivedMethod = false;
	 if (methodInfo != null)
	 {
	 hasDerivedMethod = true;
	 ParameterInfo[] parameterArray1 = methodInfo.GetParameters();
	 for (int i=0; i<methodTypes.Length; i++)
	 {
	 if (parameterArray1[0].ParameterType != methodTypes[0])
	 {
	 hasDerivedMethod = false;
	 break;
	 }
	 }
	 }
	 }
	 */
	Printf(proxy_class_code, "    return hasDerivedMethod;\n");
	Printf(proxy_class_code, "  }\n");
      }

      if (Len(director_delegate_callback) > 0)
	Printv(proxy_class_code, director_delegate_callback, NIL);
      if (Len(director_delegate_definitions) > 0)
	Printv(proxy_class_code, "\n", director_delegate_definitions, NIL);
      if (Len(director_delegate_instances) > 0)
	Printv(proxy_class_code, "\n", director_delegate_instances, NIL);
      if (Len(director_method_types) > 0)
	Printv(proxy_class_code, "\n", director_method_types, NIL);

      Delete(director_callback_typedefs);
      director_callback_typedefs = NULL;
      Delete(director_callbacks);
      director_callbacks = NULL;
      Delete(director_delegate_callback);
      director_delegate_callback = NULL;
      Delete(director_delegate_definitions);
      director_delegate_definitions = NULL;
      Delete(director_delegate_instances);
      director_delegate_instances = NULL;
      Delete(director_method_types);
      director_method_types = NULL;
      Delete(director_connect_parms);
      director_connect_parms = NULL;
      Delete(director_connect_method_name);
    }

    Delete(interface_upcasts);
    Delete(interface_list);
    Delete(attributes);
    Delete(destruct);

    // Emit extra user code
    Printv(proxy_class_def, typemapLookup(n, "cscode", typemap_lookup_type, WARN_NONE),	// extra C# code
	   "\n", NIL);

    if (derived) {
      String *upcast_method_name = Swig_name_member(getNSpace(), getClassPrefix(), smart != 0 ? "SWIGSmartPtrUpcast" : "SWIGUpcast");
      upcastsCode(smart, upcast_method_name, c_classname, c_baseclass);
      Delete(upcast_method_name);
    }

    Delete(smart);
    Delete(baseclass);
  }

  /* ----------------------------------------------------------------------
   * emitInterfaceDeclaration()
   * ---------------------------------------------------------------------- */

  void emitInterfaceDeclaration(Node *n, String *interface_name, File *f_interface) {
    Printv(f_interface, typemapLookup(n, "csimports", Getattr(n, "classtypeobj"), WARN_NONE), "\n", NIL);
    Printf(f_interface, "public interface %s", interface_name);
    if (List *baselist = Getattr(n, "bases")) {
      String *bases = 0;
      for (Iterator base = First(baselist); base.item; base = Next(base)) {
	if (GetFlag(base.item, "feature:ignore") || !Getattr(base.item, "feature:interface"))
	  continue; // TODO: warn about skipped non-interface bases
	String *base_iname = Getattr(base.item, "interface:name");
	if (!bases)
	  bases = NewStringf(" : %s", base_iname);
	else {
	  Append(bases, ", ");
	  Append(bases, base_iname);
	}
      }
      if (bases) {
	Printv(f_interface, bases, NIL);
	Delete(bases);
      }
    }
    Printf(f_interface, " {\n");

    Node *attributes = NewHash();
    String *interface_code = Copy(typemapLookup(n, "csinterfacecode", Getattr(n, "classtypeobj"), WARN_CSHARP_TYPEMAP_INTERFACECODE_UNDEF, attributes));
    if (interface_code) {
      String *interface_declaration = Copy(Getattr(attributes, "tmap:csinterfacecode:declaration"));
      if (interface_declaration) {
        Replaceall(interface_declaration, "$interfacename", interface_name);
	Printv(f_interface, interface_declaration, NIL);
	Delete(interface_declaration);
      }
      Delete(interface_code);
    }
  }

  /* ----------------------------------------------------------------------
   * classHandler()
   * ---------------------------------------------------------------------- */

  virtual int classHandler(Node *n) {
    String *nspace = getNSpace();
    File *f_proxy = NULL;
    File *f_interface = NULL;
    // save class local variables
    String *old_proxy_class_name = proxy_class_name;
    String *old_full_imclass_name = full_imclass_name;
    String *old_destructor_call = destructor_call;
    String *old_proxy_class_constants_code = proxy_class_constants_code;
    String *old_proxy_class_def = proxy_class_def;
    String *old_proxy_class_code = proxy_class_code;
    bool has_outerclass = Getattr(n, "nested:outer") && !GetFlag(n, "feature:flatnested");
    String *old_interface_class_code = interface_class_code;
    interface_class_code = 0;

    if (proxy_flag) {
      proxy_class_name = NewString(Getattr(n, "sym:name"));
      String *interface_name = Getattr(n, "feature:interface") ? Getattr(n, "interface:name") : 0;
      if (Node *outer = Getattr(n, "nested:outer")) {
	String *outerClassesPrefix = Copy(Getattr(outer, "sym:name"));
	for (outer = Getattr(outer, "nested:outer"); outer != 0; outer = Getattr(outer, "nested:outer")) {
	  Push(outerClassesPrefix, ".");
	  Push(outerClassesPrefix, Getattr(outer, "sym:name"));
	}
	String *fnspace = nspace ? NewStringf("%s.%s", nspace, outerClassesPrefix) : outerClassesPrefix;
	if (!addSymbol(proxy_class_name, n, fnspace))
	  return SWIG_ERROR;
	if (interface_name && !addInterfaceSymbol(interface_name, n, fnspace))
	  return SWIG_ERROR;
	if (nspace)
	  Delete(fnspace);
	Delete(outerClassesPrefix);
      } else {
	if (!addSymbol(proxy_class_name, n, nspace))
	  return SWIG_ERROR;
	if (interface_name && !addInterfaceSymbol(interface_name, n, nspace))
	  return SWIG_ERROR;
      }

      if (!nspace) {
	full_imclass_name = NewStringf("%s", imclass_name);
	if (Cmp(proxy_class_name, imclass_name) == 0) {
	  Printf(stderr, "Class name cannot be equal to intermediary class name: %s\n", proxy_class_name);
	  SWIG_exit(EXIT_FAILURE);
	}

	if (Cmp(proxy_class_name, module_class_name) == 0) {
	  Printf(stderr, "Class name cannot be equal to module class name: %s\n", proxy_class_name);
	  SWIG_exit(EXIT_FAILURE);
	}
      } else {
	if (namespce) {
	  full_imclass_name = NewStringf("%s.%s", namespce, imclass_name);
	} else {
	  full_imclass_name = NewStringf("%s", imclass_name);
	}
      }

      if (!has_outerclass) {
	String *output_directory = outputDirectory(nspace);
	f_proxy = getOutputFile(output_directory, proxy_class_name);

	addOpenNamespace(nspace, f_proxy);
        Delete(output_directory);
      }
      else
	++nesting_depth;

      proxy_class_def = NewString("");
      proxy_class_code = NewString("");
      destructor_call = NewString("");
      proxy_class_constants_code = NewString("");

      if (Getattr(n, "feature:interface")) {
	interface_class_code = NewString("");
	String *output_directory = outputDirectory(nspace);
	f_interface = getOutputFile(output_directory, interface_name);
	addOpenNamespace(nspace, f_interface);
	emitInterfaceDeclaration(n, interface_name, interface_class_code);
        Delete(output_directory);
      }
    }

    Language::classHandler(n);

    if (proxy_flag) {

      emitProxyClassDefAndCPPCasts(n);

      String *csclazzname = Swig_name_member(getNSpace(), getClassPrefix(), ""); // mangled full proxy class name

      Replaceall(proxy_class_def, "$csclassname", proxy_class_name);
      Replaceall(proxy_class_code, "$csclassname", proxy_class_name);
      Replaceall(proxy_class_constants_code, "$csclassname", proxy_class_name);
      Replaceall(interface_class_code, "$csclassname", proxy_class_name);

      Replaceall(proxy_class_def, "$csclazzname", csclazzname);
      Replaceall(proxy_class_code, "$csclazzname", csclazzname);
      Replaceall(proxy_class_constants_code, "$csclazzname", csclazzname);
      Replaceall(interface_class_code, "$csclazzname", csclazzname);

      Replaceall(proxy_class_def, "$module", module_class_name);
      Replaceall(proxy_class_code, "$module", module_class_name);
      Replaceall(proxy_class_constants_code, "$module", module_class_name);
      Replaceall(interface_class_code, "$module", module_class_name);

      Replaceall(proxy_class_def, "$imclassname", full_imclass_name);
      Replaceall(proxy_class_code, "$imclassname", full_imclass_name);
      Replaceall(proxy_class_constants_code, "$imclassname", full_imclass_name);
      Replaceall(interface_class_code, "$imclassname", full_imclass_name);

      Replaceall(proxy_class_def, "$dllimport", dllimport);
      Replaceall(proxy_class_code, "$dllimport", dllimport);
      Replaceall(proxy_class_constants_code, "$dllimport", dllimport);
      Replaceall(interface_class_code, "$dllimport", dllimport);

      if (!has_outerclass)
	Printv(f_proxy, proxy_class_def, proxy_class_code, NIL);
      else {
	Swig_offset_string(proxy_class_def, nesting_depth);
	Append(old_proxy_class_code, proxy_class_def);
	Swig_offset_string(proxy_class_code, nesting_depth);
	Append(old_proxy_class_code, proxy_class_code);
      }

      // Write out all the constants
      if (Len(proxy_class_constants_code) != 0) {
	if (!has_outerclass)
	  Printv(f_proxy, proxy_class_constants_code, NIL);
	else {
	  Swig_offset_string(proxy_class_constants_code, nesting_depth);
	  Append(old_proxy_class_code, proxy_class_constants_code);
	}
      }
      if (!has_outerclass) {
	Printf(f_proxy, "}\n");
	addCloseNamespace(nspace, f_proxy);
	if (f_proxy != f_single_out)
	  Delete(f_proxy);
	f_proxy = NULL;
      } else {
	for (int i = 0; i < nesting_depth; ++i)
	  Append(old_proxy_class_code, "  ");
	Append(old_proxy_class_code, "}\n\n");
	--nesting_depth;
      }

      /* Output the downcast method, if necessary. Note: There's no other really
         good place to put this code, since Abstract Base Classes (ABCs) can and should have 
         downcasts, making the constructorHandler() a bad place (because ABCs don't get to
         have constructors emitted.) */
      if (GetFlag(n, "feature:csdowncast")) {
	String *downcast_method = Swig_name_member(getNSpace(), proxy_class_name, "SWIGDowncast");
	String *wname = Swig_name_wrapper(downcast_method);

	String *norm_name = SwigType_namestr(Getattr(n, "name"));

	Printf(imclass_class_code, "  public final static native %s %s(long cPtrBase, boolean cMemoryOwn);\n", proxy_class_name, downcast_method);

	Wrapper *dcast_wrap = NewWrapper();

	Printf(dcast_wrap->def, "SWIGEXPORT jobject SWIGSTDCALL %s(JNIEnv *jenv, jclass jcls, jlong jCPtrBase, jboolean cMemoryOwn) {", wname);
	Printf(dcast_wrap->code, "  Swig::Director *director = (Swig::Director *) 0;\n");
	Printf(dcast_wrap->code, "  jobject jresult = (jobject) 0;\n");
	Printf(dcast_wrap->code, "  %s *obj = *((%s **)&jCPtrBase);\n", norm_name, norm_name);
	Printf(dcast_wrap->code, "  if (obj) director = dynamic_cast<Swig::Director *>(obj);\n");
	Printf(dcast_wrap->code, "  if (director) jresult = director->swig_get_self(jenv);\n");
	Printf(dcast_wrap->code, "  return jresult;\n");
	Printf(dcast_wrap->code, "}\n");

	Wrapper_print(dcast_wrap, f_wrappers);
	DelWrapper(dcast_wrap);

	Delete(norm_name);
	Delete(wname);
	Delete(downcast_method);
      }

      if (f_interface) {
	Printv(f_interface, interface_class_code, "}\n", NIL);
	addCloseNamespace(nspace, f_interface);
	if (f_interface != f_single_out)
	  Delete(f_interface);
	f_interface = 0;
      }

      emitDirectorExtraMethods(n);

      Delete(interface_class_code);
      interface_class_code = old_interface_class_code;
      Delete(csclazzname);
      Delete(proxy_class_name);
      proxy_class_name = old_proxy_class_name;
      Delete(full_imclass_name);
      full_imclass_name = old_full_imclass_name;
      Delete(destructor_call);
      destructor_call = old_destructor_call;
      Delete(proxy_class_constants_code);
      proxy_class_constants_code = old_proxy_class_constants_code;
      Delete(proxy_class_def);
      proxy_class_def = old_proxy_class_def;
      Delete(proxy_class_code);
      proxy_class_code = old_proxy_class_code;
    }
    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * memberfunctionHandler()
   * ---------------------------------------------------------------------- */

  virtual int memberfunctionHandler(Node *n) {
    Language::memberfunctionHandler(n);

    if (proxy_flag) {
      String *overloaded_name = getOverloadedName(n);
      String *intermediary_function_name = Swig_name_member(getNSpace(), getClassPrefix(), overloaded_name);
      Setattr(n, "proxyfuncname", Getattr(n, "sym:name"));
      Setattr(n, "imfuncname", intermediary_function_name);
      proxyClassFunctionHandler(n);
      Delete(overloaded_name);
    }
    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * staticmemberfunctionHandler()
   * ---------------------------------------------------------------------- */

  virtual int staticmemberfunctionHandler(Node *n) {

    static_flag = true;
    Language::staticmemberfunctionHandler(n);

    if (proxy_flag) {
      String *overloaded_name = getOverloadedName(n);
      String *intermediary_function_name = Swig_name_member(getNSpace(), getClassPrefix(), overloaded_name);
      Setattr(n, "proxyfuncname", Getattr(n, "sym:name"));
      Setattr(n, "imfuncname", intermediary_function_name);
      proxyClassFunctionHandler(n);
      Delete(overloaded_name);
    }
    static_flag = false;

    return SWIG_OK;
  }


  /* -----------------------------------------------------------------------------
   * proxyClassFunctionHandler()
   *
   * Function called for creating a C# wrapper function around a c++ function in the 
   * proxy class. Used for both static and non-static C++ class functions.
   * C++ class static functions map to C# static functions.
   * Two extra attributes in the Node must be available. These are "proxyfuncname" - 
   * the name of the C# class proxy function, which in turn will call "imfuncname" - 
   * the intermediary (PInvoke) function name in the intermediary class.
   * ----------------------------------------------------------------------------- */

  void proxyClassFunctionHandler(Node *n) {
    SwigType *t = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    String *intermediary_function_name = Getattr(n, "imfuncname");
    String *proxy_function_name = Getattr(n, "proxyfuncname");
    String *tm;
    Parm *p;
    Parm *last_parm = 0;
    int i;
    String *imcall = NewString("");
    String *return_type = NewString("");
    String *function_code = NewString("");
    bool setter_flag = false;
    String *pre_code = NewString("");
    String *post_code = NewString("");
    String *terminator_code = NewString("");
    bool is_interface = Getattr(parentNode(n), "feature:interface") != 0 
      && !static_flag && Getattr(n, "interface:owner") == 0;

    if (!proxy_flag)
      return;

    // Wrappers not wanted for some methods where the parameters cannot be overloaded in C#
    if (Getattr(n, "overload:ignore"))
      return;

    // Don't generate proxy method for additional explicitcall method used in directors
    if (GetFlag(n, "explicitcall"))
      return;

    if (l) {
      if (SwigType_type(Getattr(l, "type")) == T_VOID) {
	l = nextSibling(l);
      }
    }

    /* Attach the non-standard typemaps to the parameter list */
    Swig_typemap_attach_parms("in", l, NULL);
    Swig_typemap_attach_parms("cstype", l, NULL);
    Swig_typemap_attach_parms("csin", l, NULL);

    /* Get return types */
    if ((tm = Swig_typemap_lookup("cstype", n, "", 0))) {
      // Note that in the case of polymorphic (covariant) return types, the method's return type is changed to be the base of the C++ return type
      SwigType *covariant = Getattr(n, "covariant");
      String *cstypeout = Getattr(n, "tmap:cstype:out");	// the type in the cstype typemap's out attribute overrides the type in the typemap
      if (cstypeout)
	tm = cstypeout;
      substituteClassname(covariant ? covariant : t, tm);
      Printf(return_type, "%s", tm);
      if (covariant)
	Swig_warning(WARN_CSHARP_COVARIANT_RET, input_file, line_number,
		     "Covariant return types not supported in C#. Proxy method will return %s.\n", SwigType_str(covariant, 0));
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CSWTYPE_UNDEF, input_file, line_number, "No cstype typemap defined for %s\n", SwigType_str(t, 0));
    }

    if (wrapping_member_flag && !enum_constant_flag) {
      // Properties
      setter_flag = (Cmp(Getattr(n, "sym:name"), Swig_name_set(getNSpace(), Swig_name_member(0, getClassPrefix(), variable_name))) == 0);
      if (setter_flag)
        Swig_typemap_attach_parms("csvarin", l, NULL);
    }

    /* Start generating the proxy function */
    const String *outattributes = Getattr(n, "tmap:cstype:outattributes");
    if (outattributes)
      Printf(function_code, "  %s\n", outattributes);
    const String *csattributes = Getattr(n, "feature:cs:attributes");
    if (csattributes)
      Printf(function_code, "  %s\n", csattributes);
    const String *methodmods = Getattr(n, "feature:cs:methodmodifiers");
    if (methodmods) {
      if (is_smart_pointer()) {
	// Smart pointer classes do not mirror the inheritance hierarchy of the underlying pointer type, so no virtual/override/new required.
	String *mmods = Copy(methodmods);
	Replaceall(mmods, "override", "");
	Replaceall(mmods, "virtual", "");
	Replaceall(mmods, "new", "");
	Chop(mmods);		// remove trailing whitespace
	Printf(function_code, "  %s ", mmods);
	Delete(mmods);
      } else {
	Printf(function_code, "  %s ", methodmods);
      }
    } else {
      methodmods = (is_public(n) ? public_string : protected_string);
      Printf(function_code, "  %s ", methodmods);
      if (!is_smart_pointer()) {
	// Smart pointer classes do not mirror the inheritance hierarchy of the underlying pointer type, so no virtual/override/new required.
	if (Getattr(n, "override"))
	    Printf(function_code, "override ");
	else if (checkAttribute(n, "storage", "virtual"))
	  Printf(function_code, "virtual ");
	if (Getattr(n, "hides"))
	  Printf(function_code, "new ");
      }
    }
    if (static_flag)
      Printf(function_code, "static ");
    Printf(function_code, "%s %s(", return_type, proxy_function_name);
    if (is_interface)
      Printf(interface_class_code, "  %s %s(", return_type, proxy_function_name);
    

    Printv(imcall, full_imclass_name, ".$imfuncname(", NIL);
    if (!static_flag)
      Printf(imcall, "swigCPtr");

    emit_mark_varargs(l);

    int gencomma = !static_flag;

    /* Output each parameter */
    for (i = 0, p = l; p; i++) {

      /* Ignored varargs */
      if (checkAttribute(p, "varargs:ignore", "1")) {
	p = nextSibling(p);
	continue;
      }

      /* Ignored parameters */
      if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
	continue;
      }

      /* Ignore the 'this' argument for variable wrappers */
      if (!(variable_wrapper_flag && i == 0)) {
	SwigType *pt = Getattr(p, "type");
	String *param_type = NewString("");
        if (setter_flag)
          last_parm = p;

	/* Get the C# parameter type */
	if ((tm = Getattr(p, "tmap:cstype"))) {
	  substituteClassname(pt, tm);
	  const String *inattributes = Getattr(p, "tmap:cstype:inattributes");
	  Printf(param_type, "%s%s", inattributes ? inattributes : empty_string, tm);
	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSWTYPE_UNDEF, input_file, line_number, "No cstype typemap defined for %s\n", SwigType_str(pt, 0));
	}

	if (gencomma)
	  Printf(imcall, ", ");

	String *arg = makeParameterName(n, p, i, setter_flag);

	// Use typemaps to transform type used in C# wrapper function (in proxy class) to type used in PInvoke function (in intermediary class)
	if ((tm = Getattr(p, "tmap:csin"))) {
	  substituteClassname(pt, tm);
	  Replaceall(tm, "$csinput", arg);
          String *pre = Getattr(p, "tmap:csin:pre");
          if (pre) {
            substituteClassname(pt, pre);
            Replaceall(pre, "$csinput", arg);
            if (Len(pre_code) > 0)
              Printf(pre_code, "\n");
            Printv(pre_code, pre, NIL);
          }
          String *post = Getattr(p, "tmap:csin:post");
          if (post) {
            substituteClassname(pt, post);
            Replaceall(post, "$csinput", arg);
            if (Len(post_code) > 0)
              Printf(post_code, "\n");
            Printv(post_code, post, NIL);
          }
          String *terminator = Getattr(p, "tmap:csin:terminator");
          if (terminator) {
            substituteClassname(pt, terminator);
            Replaceall(terminator, "$csinput", arg);
            if (Len(terminator_code) > 0)
              Insert(terminator_code, 0, "\n");
            Insert(terminator_code, 0, terminator);
          }
	  Printv(imcall, tm, NIL);
	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSIN_UNDEF, input_file, line_number, "No csin typemap defined for %s\n", SwigType_str(pt, 0));
	}

	/* Add parameter to proxy function */
	if (gencomma >= 2) {
	  Printf(function_code, ", ");
	  if (is_interface)
	    Printf(interface_class_code, ", ");
	}
	gencomma = 2;
	Printf(function_code, "%s %s", param_type, arg);
	if (is_interface)
	  Printf(interface_class_code, "%s %s", param_type, arg);

	Delete(arg);
	Delete(param_type);
      }
      p = Getattr(p, "tmap:in:next");
    }

    Printf(imcall, ")");
    Printf(function_code, ")");
    if (is_interface)
      Printf(interface_class_code, ");\n");

    // Transform return type used in PInvoke function (in intermediary class) to type used in C# wrapper function (in proxy class)
    if ((tm = Swig_typemap_lookup("csout", n, "", 0))) {
      excodeSubstitute(n, tm, "csout", n);
      bool is_pre_code = Len(pre_code) > 0;
      bool is_post_code = Len(post_code) > 0;
      bool is_terminator_code = Len(terminator_code) > 0;
      if (is_pre_code || is_post_code || is_terminator_code) {
        Replaceall(tm, "\n ", "\n   "); // add extra indentation to code in typemap
        if (is_post_code) {
          Insert(tm, 0, "\n    try ");
          Printv(tm, " finally {\n", post_code, "\n    }", NIL);
        } else {
          Insert(tm, 0, "\n    ");
        }
        if (is_pre_code) {
          Insert(tm, 0, pre_code);
          Insert(tm, 0, "\n");
        }
        if (is_terminator_code) {
          Printv(tm, "\n", terminator_code, NIL);
        }
	Insert(tm, 0, "{");
	Printf(tm, "\n  }");
      }
      if (GetFlag(n, "feature:new"))
	Replaceall(tm, "$owner", "true");
      else
	Replaceall(tm, "$owner", "false");
      substituteClassname(t, tm);

      // For director methods: generate code to selectively make a normal polymorphic call or 
      // an explicit method call - needed to prevent infinite recursion calls in director methods.
      Node *explicit_n = Getattr(n, "explicitcallnode");
      if (explicit_n) {
	String *ex_overloaded_name = getOverloadedName(explicit_n);
	String *ex_intermediary_function_name = Swig_name_member(getNSpace(), getClassPrefix(), ex_overloaded_name);

	String *ex_imcall = Copy(imcall);
	Replaceall(ex_imcall, "$imfuncname", ex_intermediary_function_name);
	Replaceall(imcall, "$imfuncname", intermediary_function_name);
	String *excode = NewString("");
	Node *directorNode = Getattr(n, "directorNode");
	if (directorNode) {
	  UpcallData *udata = Getattr(directorNode, "upcalldata");
	  String *methid = Getattr(udata, "class_methodidx");

	  if (!Cmp(return_type, "void"))
	    Printf(excode, "if (SwigDerivedClassHasMethod(\"%s\", swigMethodTypes%s)) %s; else %s", proxy_function_name, methid, ex_imcall, imcall);
	  else
	    Printf(excode, "(SwigDerivedClassHasMethod(\"%s\", swigMethodTypes%s) ? %s : %s)", proxy_function_name, methid, ex_imcall, imcall);

	  Clear(imcall);
	  Printv(imcall, excode, NIL);
	} else {
	  // probably an ignored method or nodirector
	}
	Delete(excode);
	Delete(ex_overloaded_name);
      } else {
	Replaceall(imcall, "$imfuncname", intermediary_function_name);
      }
      Replaceall(tm, "$imcall", imcall);
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CSOUT_UNDEF, input_file, line_number, "No csout typemap defined for %s\n", SwigType_str(t, 0));
    }

    if (wrapping_member_flag && !enum_constant_flag) {
      // Properties
      if (generate_property_declaration_flag) {	// Ensure the declaration is generated just once should the property contain both a set and get
	// Get the C# variable type - obtained differently depending on whether a setter is required.
	String *variable_type = return_type;
	if (setter_flag) {
	  assert(last_parm);	// (last parameter is the only parameter for properties)
	  /* Get variable type - ensure the variable name is fully resolved during typemap lookup via the symbol table set in NewParmNode */
	  SwigType *cvariable_type = Getattr(last_parm, "type");
	  Parm *variable_parm = NewParmNode(cvariable_type, n);
	  if ((tm = Swig_typemap_lookup("cstype", variable_parm, "", 0))) {
	    String *cstypeout = Getattr(variable_parm, "tmap:cstype:out");	// the type in the cstype typemap's out attribute overrides the type in the typemap
	    if (cstypeout)
	      tm = cstypeout;
	    substituteClassname(cvariable_type, tm);
	    variable_type = tm;
	  } else {
	    Swig_warning(WARN_CSHARP_TYPEMAP_CSOUT_UNDEF, input_file, line_number, "No cstype typemap defined for %s\n", SwigType_str(cvariable_type, 0));
	  }
	}
	const String *csattributes = Getattr(n, "feature:cs:attributes");
	if (csattributes)
	  Printf(proxy_class_code, "  %s\n", csattributes);
	const String *methodmods = Getattr(n, "feature:cs:methodmodifiers");
	if (!methodmods)
	  methodmods = (is_public(n) ? public_string : protected_string);
	Printf(proxy_class_code, "  %s %s%s %s {", methodmods, static_flag ? "static " : "", variable_type, variable_name);
      }
      generate_property_declaration_flag = false;

      if (setter_flag) {
	// Setter method
	assert(last_parm);	// (last parameter is the only parameter for properties)
	SwigType *cvariable_type = Getattr(last_parm, "type");
	Parm *variable_parm = NewParmNode(cvariable_type, n);
	if ((tm = Swig_typemap_lookup("csvarin", variable_parm, "", 0))) {
	  substituteClassname(cvariable_type, tm);
	  Replaceall(tm, "$csinput", "value");
	  Replaceall(tm, "$imcall", imcall);
	  excodeSubstitute(n, tm, "csvarin", variable_parm);
	  Printf(proxy_class_code, "%s", tm);
	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSOUT_UNDEF, input_file, line_number, "No csvarin typemap defined for %s\n", SwigType_str(cvariable_type, 0));
	}
      } else {
	// Getter method
	if ((tm = Swig_typemap_lookup("csvarout", n, "", 0))) {
	  if (GetFlag(n, "feature:new"))
	    Replaceall(tm, "$owner", "true");
	  else
	    Replaceall(tm, "$owner", "false");
	  substituteClassname(t, tm);
	  Replaceall(tm, "$imcall", imcall);
	  excodeSubstitute(n, tm, "csvarout", n);
	  Printf(proxy_class_code, "%s", tm);
	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSOUT_UNDEF, input_file, line_number, "No csvarout typemap defined for %s\n", SwigType_str(t, 0));
	}
      }
    } else {
      // Normal function call
      Printf(function_code, " %s\n\n", tm ? (const String *) tm : empty_string);
      Printv(proxy_class_code, function_code, NIL);
    }

    Delete(pre_code);
    Delete(post_code);
    Delete(terminator_code);
    Delete(function_code);
    Delete(return_type);
    Delete(imcall);
  }

  /* ----------------------------------------------------------------------
   * constructorHandler()
   * ---------------------------------------------------------------------- */

  virtual int constructorHandler(Node *n) {

    ParmList *l = Getattr(n, "parms");
    String *tm;
    Parm *p;
    int i;
    String *function_code = NewString("");
    String *helper_code = NewString(""); // Holds code for the constructor helper method generated only when the csin typemap has code in the pre or post attributes
    String *helper_args = NewString("");
    String *pre_code = NewString("");
    String *post_code = NewString("");
    String *terminator_code = NewString("");
    String *im_return_type = NewString("");
    bool feature_director = (parentNode(n) && Swig_directorclass(n));

    Language::constructorHandler(n);

    // Wrappers not wanted for some methods where the parameters cannot be overloaded in C#
    if (Getattr(n, "overload:ignore"))
      return SWIG_OK;

    if (proxy_flag) {
      String *overloaded_name = getOverloadedName(n);
      String *mangled_overname = Swig_name_construct(getNSpace(), overloaded_name);
      String *imcall = NewString("");

      const String *csattributes = Getattr(n, "feature:cs:attributes");
      if (csattributes) {
	Printf(function_code, "  %s\n", csattributes);
	Printf(helper_code, "  %s\n", csattributes);
      }
      const String *methodmods = Getattr(n, "feature:cs:methodmodifiers");
      methodmods = methodmods ? methodmods : (is_public(n) ? public_string : protected_string);

      tm = Getattr(n, "tmap:imtype"); // typemaps were attached earlier to the node
      String *imtypeout = Getattr(n, "tmap:imtype:out");	// the type in the imtype typemap's out attribute overrides the type in the typemap
      if (imtypeout)
	tm = imtypeout;
      Printf(im_return_type, "%s", tm);

      Printf(function_code, "  %s %s(", methodmods, proxy_class_name);
      Printf(helper_code, "  static private %s SwigConstruct%s(", im_return_type, proxy_class_name);

      Printv(imcall, full_imclass_name, ".", mangled_overname, "(", NIL);

      /* Attach the non-standard typemaps to the parameter list */
      Swig_typemap_attach_parms("in", l, NULL);
      Swig_typemap_attach_parms("cstype", l, NULL);
      Swig_typemap_attach_parms("csin", l, NULL);

      emit_mark_varargs(l);

      int gencomma = 0;

      /* Output each parameter */
      for (i = 0, p = l; p; i++) {

	/* Ignored varargs */
	if (checkAttribute(p, "varargs:ignore", "1")) {
	  p = nextSibling(p);
	  continue;
	}

	/* Ignored parameters */
	if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	  p = Getattr(p, "tmap:in:next");
	  continue;
	}

	SwigType *pt = Getattr(p, "type");
	String *param_type = NewString("");

	/* Get the C# parameter type */
	if ((tm = Getattr(p, "tmap:cstype"))) {
	  substituteClassname(pt, tm);
	  const String *inattributes = Getattr(p, "tmap:cstype:inattributes");
	  Printf(param_type, "%s%s", inattributes ? inattributes : empty_string, tm);
	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSWTYPE_UNDEF, input_file, line_number, "No cstype typemap defined for %s\n", SwigType_str(pt, 0));
	}

	if (gencomma)
	  Printf(imcall, ", ");

	String *arg = makeParameterName(n, p, i, false);
        String *cshin = 0;

	// Use typemaps to transform type used in C# wrapper function (in proxy class) to type used in PInvoke function (in intermediary class)
	if ((tm = Getattr(p, "tmap:csin"))) {
	  substituteClassname(pt, tm);
	  Replaceall(tm, "$csinput", arg);
          String *pre = Getattr(p, "tmap:csin:pre");
          if (pre) {
            substituteClassname(pt, pre);
            Replaceall(pre, "$csinput", arg);
            if (Len(pre_code) > 0)
              Printf(pre_code, "\n");
            Printv(pre_code, pre, NIL);
          }
          String *post = Getattr(p, "tmap:csin:post");
          if (post) {
            substituteClassname(pt, post);
            Replaceall(post, "$csinput", arg);
            if (Len(post_code) > 0)
              Printf(post_code, "\n");
            Printv(post_code, post, NIL);
          }
          String *terminator = Getattr(p, "tmap:csin:terminator");
          if (terminator) {
            substituteClassname(pt, terminator);
            Replaceall(terminator, "$csinput", arg);
            if (Len(terminator_code) > 0)
              Insert(terminator_code, 0, "\n");
            Insert(terminator_code, 0, terminator);
          }
          cshin = Getattr(p, "tmap:csin:cshin");
          if (cshin)
            Replaceall(cshin, "$csinput", arg);
	  Printv(imcall, tm, NIL);
	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSIN_UNDEF, input_file, line_number, "No csin typemap defined for %s\n", SwigType_str(pt, 0));
	}

	/* Add parameter to proxy function */
	if (gencomma) {
	  Printf(function_code, ", ");
	  Printf(helper_code, ", ");
	  Printf(helper_args, ", ");
        }
	Printf(function_code, "%s %s", param_type, arg);
	Printf(helper_code, "%s %s", param_type, arg);
	Printf(helper_args, "%s", cshin ? cshin : arg);
	++gencomma;

	Delete(cshin);
	Delete(arg);
	Delete(param_type);
	p = Getattr(p, "tmap:in:next");
      }

      Printf(imcall, ")");

      Printf(function_code, ")");
      Printf(helper_code, ")");

      /* Insert the csconstruct typemap, doing the replacement for $directorconnect, as needed */
      Hash *attributes = NewHash();
      String *construct_tm = Copy(typemapLookup(n, "csconstruct", Getattr(n, "name"),
						WARN_CSHARP_TYPEMAP_CSCONSTRUCT_UNDEF, attributes));
      if (construct_tm) {
	if (!feature_director) {
	  Replaceall(construct_tm, "$directorconnect", "");
	} else {
	  String *connect_attr = Getattr(attributes, "tmap:csconstruct:directorconnect");

	  if (connect_attr) {
	    Replaceall(construct_tm, "$directorconnect", connect_attr);
	  } else {
	    Swig_warning(WARN_CSHARP_NO_DIRECTORCONNECT_ATTR, input_file, line_number, "\"directorconnect\" attribute missing in %s \"csconstruct\" typemap.\n",
			 Getattr(n, "name"));
	    Replaceall(construct_tm, "$directorconnect", "");
	  }
	}

	Printv(function_code, " ", construct_tm, NIL);
      }

      excodeSubstitute(n, function_code, "csconstruct", attributes);

      bool is_pre_code = Len(pre_code) > 0;
      bool is_post_code = Len(post_code) > 0;
      bool is_terminator_code = Len(terminator_code) > 0;
      if (is_pre_code || is_post_code || is_terminator_code) {
        Printf(helper_code, " {\n");
        if (is_pre_code) {
          Printv(helper_code, pre_code, "\n", NIL);
        }
        if (is_post_code) {
          Printf(helper_code, "    try {\n");
          Printv(helper_code, "      return ", imcall, ";\n", NIL);
          Printv(helper_code, "    } finally {\n", post_code, "\n    }", NIL);
        } else {
          Printv(helper_code, "    return ", imcall, ";", NIL);
        }
        if (is_terminator_code) {
          Printv(helper_code, "\n", terminator_code, NIL);
        }
        Printf(helper_code, "\n  }\n");
        String *helper_name = NewStringf("%s.SwigConstruct%s(%s)", proxy_class_name, proxy_class_name, helper_args);
        String *im_outattributes = Getattr(n, "tmap:imtype:outattributes");
        if (im_outattributes)
          Printf(proxy_class_code, "  %s\n", im_outattributes);
        Printv(proxy_class_code, helper_code, "\n", NIL);
        Replaceall(function_code, "$imcall", helper_name);
        Delete(helper_name);
      } else {
        Replaceall(function_code, "$imcall", imcall);
      }

      Printv(proxy_class_code, function_code, "\n", NIL);

      Delete(helper_args);
      Delete(im_return_type);
      Delete(pre_code);
      Delete(post_code);
      Delete(terminator_code);
      Delete(construct_tm);
      Delete(attributes);
      Delete(overloaded_name);
      Delete(imcall);
    }

    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * destructorHandler()
   * ---------------------------------------------------------------------- */

  virtual int destructorHandler(Node *n) {
    Language::destructorHandler(n);
    String *symname = Getattr(n, "sym:name");

    if (proxy_flag) {
      Printv(destructor_call, full_imclass_name, ".", Swig_name_destroy(getNSpace(), symname), "(swigCPtr)", NIL);
    }
    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * membervariableHandler()
   * ---------------------------------------------------------------------- */

  virtual int membervariableHandler(Node *n) {

    generate_property_declaration_flag = true;
    variable_name = Getattr(n, "sym:name");
    wrapping_member_flag = true;
    variable_wrapper_flag = true;
    Language::membervariableHandler(n);
    wrapping_member_flag = false;
    variable_wrapper_flag = false;
    generate_property_declaration_flag = false;

    Printf(proxy_class_code, "\n  }\n\n");

    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * staticmembervariableHandler()
   * ---------------------------------------------------------------------- */

  virtual int staticmembervariableHandler(Node *n) {

    bool static_const_member_flag = (Getattr(n, "value") == 0);

    generate_property_declaration_flag = true;
    variable_name = Getattr(n, "sym:name");
    wrapping_member_flag = true;
    static_flag = true;
    Language::staticmembervariableHandler(n);
    wrapping_member_flag = false;
    static_flag = false;
    generate_property_declaration_flag = false;

    if (static_const_member_flag)
      Printf(proxy_class_code, "\n  }\n\n");

    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * memberconstantHandler()
   * ---------------------------------------------------------------------- */

  virtual int memberconstantHandler(Node *n) {
    variable_name = Getattr(n, "sym:name");
    wrapping_member_flag = true;
    Language::memberconstantHandler(n);
    wrapping_member_flag = false;
    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------------
   * getOverloadedName()
   * ----------------------------------------------------------------------------- */

  String *getOverloadedName(Node *n) {

    /* A C# HandleRef is used for all classes in the SWIG intermediary class.
     * The intermediary class methods are thus mangled when overloaded to give
     * a unique name. */
    String *overloaded_name = NewStringf("%s", Getattr(n, "sym:name"));

    if (Getattr(n, "sym:overloaded")) {
      Printv(overloaded_name, Getattr(n, "sym:overname"), NIL);
    }

    return overloaded_name;
  }

  /* -----------------------------------------------------------------------------
   * moduleClassFunctionHandler()
   * ----------------------------------------------------------------------------- */

  void moduleClassFunctionHandler(Node *n) {
    SwigType *t = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    String *tm;
    Parm *p;
    Parm *last_parm = 0;
    int i;
    String *imcall = NewString("");
    String *return_type = NewString("");
    String *function_code = NewString("");
    int num_arguments = 0;
    String *overloaded_name = getOverloadedName(n);
    String *func_name = NULL;
    bool setter_flag = false;
    String *pre_code = NewString("");
    String *post_code = NewString("");
    String *terminator_code = NewString("");

    if (l) {
      if (SwigType_type(Getattr(l, "type")) == T_VOID) {
	l = nextSibling(l);
      }
    }

    /* Attach the non-standard typemaps to the parameter list */
    Swig_typemap_attach_parms("cstype", l, NULL);
    Swig_typemap_attach_parms("csin", l, NULL);

    /* Get return types */
    if ((tm = Swig_typemap_lookup("cstype", n, "", 0))) {
      String *cstypeout = Getattr(n, "tmap:cstype:out");	// the type in the cstype typemap's out attribute overrides the type in the typemap
      if (cstypeout)
	tm = cstypeout;
      substituteClassname(t, tm);
      Printf(return_type, "%s", tm);
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CSWTYPE_UNDEF, input_file, line_number, "No cstype typemap defined for %s\n", SwigType_str(t, 0));
    }

    /* Change function name for global variables */
    if (proxy_flag && global_variable_flag) {
      // Capitalize the first letter in the variable to create the getter/setter function name
      func_name = NewString("");
      setter_flag = (Cmp(Getattr(n, "sym:name"), Swig_name_set(getNSpace(), variable_name)) == 0);
      if (setter_flag)
	Printf(func_name, "set");
      else
	Printf(func_name, "get");
      Putc(toupper((int) *Char(variable_name)), func_name);
      Printf(func_name, "%s", Char(variable_name) + 1);
      if (setter_flag)
        Swig_typemap_attach_parms("csvarin", l, NULL);
    } else {
      func_name = Copy(Getattr(n, "sym:name"));
    }

    /* Start generating the function */
    const String *outattributes = Getattr(n, "tmap:cstype:outattributes");
    if (outattributes)
      Printf(function_code, "  %s\n", outattributes);
    const String *csattributes = Getattr(n, "feature:cs:attributes");
    if (csattributes)
      Printf(function_code, "  %s\n", csattributes);
    const String *methodmods = Getattr(n, "feature:cs:methodmodifiers");
    methodmods = methodmods ? methodmods : (is_public(n) ? public_string : protected_string);
    Printf(function_code, "  %s static %s %s(", methodmods, return_type, func_name);
    Printv(imcall, imclass_name, ".", overloaded_name, "(", NIL);

    /* Get number of required and total arguments */
    num_arguments = emit_num_arguments(l);

    bool global_or_member_variable = global_variable_flag || (wrapping_member_flag && !enum_constant_flag);
    int gencomma = 0;

    /* Output each parameter */
    for (i = 0, p = l; i < num_arguments; i++) {

      /* Ignored parameters */
      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");
      String *param_type = NewString("");
      last_parm = p;

      /* Get the C# parameter type */
      if ((tm = Getattr(p, "tmap:cstype"))) {
	substituteClassname(pt, tm);
	const String *inattributes = Getattr(p, "tmap:cstype:inattributes");
	Printf(param_type, "%s%s", inattributes ? inattributes : empty_string, tm);
      } else {
	Swig_warning(WARN_CSHARP_TYPEMAP_CSWTYPE_UNDEF, input_file, line_number, "No cstype typemap defined for %s\n", SwigType_str(pt, 0));
      }

      if (gencomma)
	Printf(imcall, ", ");

      String *arg = makeParameterName(n, p, i, global_or_member_variable);

      // Use typemaps to transform type used in C# wrapper function (in proxy class) to type used in PInvoke function (in intermediary class)
      if ((tm = Getattr(p, "tmap:csin"))) {
	substituteClassname(pt, tm);
	Replaceall(tm, "$csinput", arg);
	String *pre = Getattr(p, "tmap:csin:pre");
	if (pre) {
	  substituteClassname(pt, pre);
	  Replaceall(pre, "$csinput", arg);
          if (Len(pre_code) > 0)
            Printf(pre_code, "\n");
	  Printv(pre_code, pre, NIL);
	}
	String *post = Getattr(p, "tmap:csin:post");
	if (post) {
	  substituteClassname(pt, post);
	  Replaceall(post, "$csinput", arg);
          if (Len(post_code) > 0)
            Printf(post_code, "\n");
	  Printv(post_code, post, NIL);
	}
        String *terminator = Getattr(p, "tmap:csin:terminator");
        if (terminator) {
          substituteClassname(pt, terminator);
          Replaceall(terminator, "$csinput", arg);
          if (Len(terminator_code) > 0)
            Insert(terminator_code, 0, "\n");
          Insert(terminator_code, 0, terminator);
        }
	Printv(imcall, tm, NIL);
      } else {
	Swig_warning(WARN_CSHARP_TYPEMAP_CSIN_UNDEF, input_file, line_number, "No csin typemap defined for %s\n", SwigType_str(pt, 0));
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
    Printf(function_code, ")");

    // Transform return type used in PInvoke function (in intermediary class) to type used in C# wrapper function (in module class)
    if ((tm = Swig_typemap_lookup("csout", n, "", 0))) {
      excodeSubstitute(n, tm, "csout", n);
      bool is_pre_code = Len(pre_code) > 0;
      bool is_post_code = Len(post_code) > 0;
      bool is_terminator_code = Len(terminator_code) > 0;
      if (is_pre_code || is_post_code || is_terminator_code) {
        Replaceall(tm, "\n ", "\n   "); // add extra indentation to code in typemap
        if (is_post_code) {
          Insert(tm, 0, "\n    try ");
          Printv(tm, " finally {\n", post_code, "\n    }", NIL);
        } else {
          Insert(tm, 0, "\n    ");
        }
        if (is_pre_code) {
          Insert(tm, 0, pre_code);
          Insert(tm, 0, "\n");
        }
        if (is_terminator_code) {
          Printv(tm, "\n", terminator_code, NIL);
        }
	Insert(tm, 0, "{");
	Printf(tm, "\n  }");
      }
      if (GetFlag(n, "feature:new"))
	Replaceall(tm, "$owner", "true");
      else
	Replaceall(tm, "$owner", "false");
      substituteClassname(t, tm);
      Replaceall(tm, "$imcall", imcall);
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CSOUT_UNDEF, input_file, line_number, "No csout typemap defined for %s\n", SwigType_str(t, 0));
    }

    if (proxy_flag && global_variable_flag) {
      // Properties
      if (generate_property_declaration_flag) {	// Ensure the declaration is generated just once should the property contain both a set and get
	// Get the C# variable type - obtained differently depending on whether a setter is required.
	String *variable_type = return_type;
	if (setter_flag) {
	  p = last_parm;	// (last parameter is the only parameter for properties)
	  SwigType *pt = Getattr(p, "type");
	  if ((tm = Getattr(p, "tmap:cstype"))) {
	    substituteClassname(pt, tm);
            String *cstypeout = Getattr(p, "tmap:cstype:out");	// the type in the cstype typemap's out attribute overrides the type in the typemap
	    variable_type = cstypeout ? cstypeout : tm;
	  } else {
	    Swig_warning(WARN_CSHARP_TYPEMAP_CSOUT_UNDEF, input_file, line_number, "No csvarin typemap defined for %s\n", SwigType_str(pt, 0));
	  }
	}
	const String *csattributes = Getattr(n, "feature:cs:attributes");
	if (csattributes)
	  Printf(module_class_code, "  %s\n", csattributes);
	const String *methodmods = Getattr(n, "feature:cs:methodmodifiers");
	if (!methodmods)
	  methodmods = (is_public(n) ? public_string : protected_string);
	Printf(module_class_code, "  %s static %s %s {", methodmods, variable_type, variable_name);
      }
      generate_property_declaration_flag = false;

      if (setter_flag) {
	// Setter method
	p = last_parm;		// (last parameter is the only parameter for properties)
	SwigType *pt = Getattr(p, "type");
	if ((tm = Getattr(p, "tmap:csvarin"))) {
	  substituteClassname(pt, tm);
	  Replaceall(tm, "$csinput", "value");
	  Replaceall(tm, "$imcall", imcall);
	  excodeSubstitute(n, tm, "csvarin", p);
	  Printf(module_class_code, "%s", tm);
	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSOUT_UNDEF, input_file, line_number, "No csvarin typemap defined for %s\n", SwigType_str(pt, 0));
	}
      } else {
	// Getter method
	if ((tm = Swig_typemap_lookup("csvarout", n, "", 0))) {
	  if (GetFlag(n, "feature:new"))
	    Replaceall(tm, "$owner", "true");
	  else
	    Replaceall(tm, "$owner", "false");
	  substituteClassname(t, tm);
	  Replaceall(tm, "$imcall", imcall);
	  excodeSubstitute(n, tm, "csvarout", n);
	  Printf(module_class_code, "%s", tm);
	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSOUT_UNDEF, input_file, line_number, "No csvarout typemap defined for %s\n", SwigType_str(t, 0));
	}
      }
    } else {
      // Normal function call
      Printf(function_code, " %s\n\n", tm ? (const String *) tm : empty_string);
      Printv(module_class_code, function_code, NIL);
    }

    Delete(pre_code);
    Delete(post_code);
    Delete(terminator_code);
    Delete(function_code);
    Delete(return_type);
    Delete(imcall);
    Delete(func_name);
  }

  /*----------------------------------------------------------------------
   * replaceSpecialVariables()
   *--------------------------------------------------------------------*/

  virtual void replaceSpecialVariables(String *method, String *tm, Parm *parm) {
    (void)method;
    SwigType *type = Getattr(parm, "type");
    substituteClassname(type, tm);
  }

  /*----------------------------------------------------------------------
   * decodeEnumFeature()
   * Decode the possible enum features, which are one of:
   *   %csenum(simple)
   *   %csenum(typeunsafe) - default
   *   %csenum(typesafe)
   *   %csenum(proper)
   *--------------------------------------------------------------------*/

  EnumFeature decodeEnumFeature(Node *n) {
    EnumFeature enum_feature = TypeunsafeEnum;
    String *feature = Getattr(n, "feature:cs:enum");
    if (feature) {
      if (Cmp(feature, "simple") == 0)
	enum_feature = SimpleEnum;
      else if (Cmp(feature, "typesafe") == 0)
	enum_feature = TypesafeEnum;
      else if (Cmp(feature, "proper") == 0)
	enum_feature = ProperEnum;
    }
    return enum_feature;
  }

  /* -----------------------------------------------------------------------
   * enumValue()
   * This method will return a string with an enum value to use in C# generated
   * code. If the %csconst feature is not used, the string will contain the intermediary
   * class call to obtain the enum value. The intermediary class and PINVOKE methods to obtain
   * the enum value will be generated. Otherwise the C/C++ enum value will be used if there
   * is one and hopefully it will compile as C# code - e.g. 20 as in: enum E{e=20};
   * The %csconstvalue feature overrides all other ways to generate the constant value.
   * The caller must delete memory allocated for the returned string.
   * ------------------------------------------------------------------------ */

  String *enumValue(Node *n) {
    String *symname = Getattr(n, "sym:name");

    // Check for the %csconstvalue feature
    String *value = Getattr(n, "feature:cs:constvalue");

    if (!value) {
      // The %csconst feature determines how the constant value is obtained
      int const_feature_flag = GetFlag(n, "feature:cs:const");

      if (const_feature_flag) {
	// Use the C syntax to make a true C# constant and hope that it compiles as C# code
	value = Getattr(n, "enumvalue") ? Copy(Getattr(n, "enumvalue")) : Copy(Getattr(n, "enumvalueex"));
      } else {
	String *newsymname = 0;
	if (!getCurrentClass() || !proxy_flag) {
	  String *enumClassPrefix = getEnumClassPrefix();
	  if (enumClassPrefix) {
	    // A global scoped enum
	    newsymname = Swig_name_member(0, enumClassPrefix, symname);
	    symname = newsymname;
	  }
	}

	// Get the enumvalue from a PINVOKE call
	if (!getCurrentClass() || !cparse_cplusplus || !proxy_flag) {
	  // Strange hack to change the name
	  Setattr(n, "name", Getattr(n, "value"));	/* for wrapping of enums in a namespace when emit_action is used */
	  constantWrapper(n);
	  value = NewStringf("%s.%s()", full_imclass_name ? full_imclass_name : imclass_name, Swig_name_get(getNSpace(), symname));
	} else {
	  memberconstantHandler(n);
	  value = NewStringf("%s.%s()", full_imclass_name ? full_imclass_name : imclass_name, Swig_name_get(getNSpace(), Swig_name_member(0, getEnumClassPrefix(), symname)));
	}
      }
    }
    return value;
  }

  /* -----------------------------------------------------------------------------
   * getEnumName()
   * ----------------------------------------------------------------------------- */

  String *getEnumName(SwigType *t) {
    Node *enumname = NULL;
    Node *n = enumLookup(t);
    if (n) {
      enumname = Getattr(n, "enumname");
      if (!enumname) {
	String *symname = Getattr(n, "sym:name");
	if (symname) {
	  // Add in class scope when referencing enum if not a global enum
	  String *scopename_prefix = Swig_scopename_prefix(Getattr(n, "name"));
	  String *proxyname = 0;
	  if (scopename_prefix) {
	    proxyname = getProxyName(scopename_prefix);
	  }
	  if (proxyname) {
	    enumname = NewStringf("%s.%s", proxyname, symname);
	  } else {
	    // global enum or enum in a namespace
	    String *nspace = Getattr(n, "sym:nspace");
	    if (nspace) {
	      if (namespce)
		enumname = NewStringf("%s.%s.%s", namespce, nspace, symname);
	      else
		enumname = NewStringf("%s.%s", nspace, symname);
	    } else {
	      enumname = Copy(symname);
	    }
	  }
	  Setattr(n, "enumname", enumname);
	  Delete(enumname);
	  Delete(scopename_prefix);
	}
      }
    }

    return enumname;
  }

  /* -----------------------------------------------------------------------------
   * substituteClassname()
   *
   * Substitute the special variable $csclassname with the proxy class name for classes/structs/unions 
   * that SWIG knows about. Also substitutes enums with enum name.
   * Otherwise use the $descriptor name for the C# class name. Note that the $&csclassname substitution
   * is the same as a $&descriptor substitution, ie one pointer added to descriptor name.
   * Inputs:
   *   pt - parameter type
   *   tm - typemap contents that might contain the special variable to be replaced
   * Outputs:
   *   tm - typemap contents complete with the special variable substitution
   * Return:
   *   substitution_performed - flag indicating if a substitution was performed
   * ----------------------------------------------------------------------------- */

  bool substituteClassname(SwigType *pt, String *tm) {
    bool substitution_performed = false;
    SwigType *type = Copy(SwigType_typedef_resolve_all(pt));
    SwigType *strippedtype = SwigType_strip_qualifiers(type);

    if (Strstr(tm, "$csclassname")) {
      SwigType *classnametype = Copy(strippedtype);
      substituteClassnameSpecialVariable(classnametype, tm, "$csclassname");
      substitution_performed = true;
      Delete(classnametype);
    }
    if (Strstr(tm, "$*csclassname")) {
      SwigType *classnametype = Copy(strippedtype);
      Delete(SwigType_pop(classnametype));
      if (Len(classnametype) > 0) {
	substituteClassnameSpecialVariable(classnametype, tm, "$*csclassname");
	substitution_performed = true;
      }
      Delete(classnametype);
    }
    if (Strstr(tm, "$&csclassname")) {
      SwigType *classnametype = Copy(strippedtype);
      SwigType_add_pointer(classnametype);
      substituteClassnameSpecialVariable(classnametype, tm, "$&csclassname");
      substitution_performed = true;
      Delete(classnametype);
    }
    if (Strstr(tm, "$csinterfacename")) {
      SwigType *interfacenametype = Copy(strippedtype);
      substituteInterfacenameSpecialVariable(interfacenametype, tm, "$csinterfacename", true);
      substitution_performed = true;
      Delete(interfacenametype);
    }
    if (Strstr(tm, "$*csinterfacename")) {
      SwigType *interfacenametype = Copy(strippedtype);
      Delete(SwigType_pop(interfacenametype));
      if (Len(interfacenametype) > 0) {
	substituteInterfacenameSpecialVariable(interfacenametype, tm, "$*csinterfacename", true);
	substitution_performed = true;
      }
      Delete(interfacenametype);
    }
    if (Strstr(tm, "$&csinterfacename")) {
      SwigType *interfacenametype = Copy(strippedtype);
      SwigType_add_pointer(interfacenametype);
      substituteInterfacenameSpecialVariable(interfacenametype, tm, "$&csinterfacename", true);
      substitution_performed = true;
      Delete(interfacenametype);
    }
    if (Strstr(tm, "$interfacename")) {
      SwigType *interfacenametype = Copy(strippedtype);
      substituteInterfacenameSpecialVariable(interfacenametype, tm, "$interfacename", false);
      substitution_performed = true;
      Delete(interfacenametype);
    }
    if (Strstr(tm, "$*interfacename")) {
      SwigType *interfacenametype = Copy(strippedtype);
      Delete(SwigType_pop(interfacenametype));
      if (Len(interfacenametype) > 0) {
	substituteInterfacenameSpecialVariable(interfacenametype, tm, "$*interfacename", false);
	substitution_performed = true;
      }
      Delete(interfacenametype);
    }
    if (Strstr(tm, "$&interfacename")) {
      SwigType *interfacenametype = Copy(strippedtype);
      SwigType_add_pointer(interfacenametype);
      substituteInterfacenameSpecialVariable(interfacenametype, tm, "$&interfacename", false);
      substitution_performed = true;
      Delete(interfacenametype);
    }

    Delete(strippedtype);
    Delete(type);

    return substitution_performed;
  }

  /* -----------------------------------------------------------------------------
   * substituteClassnameSpecialVariable()
   * ----------------------------------------------------------------------------- */

  void substituteClassnameSpecialVariable(SwigType *classnametype, String *tm, const char *classnamespecialvariable) {
    String *replacementname;
    if (SwigType_isenum(classnametype)) {
      String *enumname = getEnumName(classnametype);
      if (enumname) {
	replacementname = Copy(enumname);
      } else {
	bool anonymous_enum = (Cmp(classnametype, "enum ") == 0);
	if (anonymous_enum) {
	  replacementname = NewString("int");
	} else {
	  // An unknown enum - one that has not been parsed (neither a C enum forward reference nor a definition) or an ignored enum
	  replacementname = NewStringf("SWIGTYPE%s", SwigType_manglestr(classnametype));
	  Replace(replacementname, "enum ", "", DOH_REPLACE_ANY);
	  Setattr(swig_types_hash, replacementname, classnametype);
	}
      }
    } else {
      String *classname = getProxyName(classnametype); // getProxyName() works for pointers to classes too
      if (classname) {
	replacementname = Copy(classname);
      } else {
	// use $descriptor if SWIG does not know anything about this type. Note that any typedefs are resolved.
	replacementname = NewStringf("SWIGTYPE%s", SwigType_manglestr(classnametype));

	// Add to hash table so that the type wrapper classes can be created later
	Setattr(swig_types_hash, replacementname, classnametype);
      }
    }
    Replaceall(tm, classnamespecialvariable, replacementname);

    Delete(replacementname);
  }

  /* -----------------------------------------------------------------------------
   * substituteInterfacenameSpecialVariable()
   * ----------------------------------------------------------------------------- */

  void substituteInterfacenameSpecialVariable(SwigType *interfacenametype, String *tm, const char *interfacenamespecialvariable, bool qualified) {

    String *interfacename = getInterfaceName(interfacenametype, qualified);
    if (interfacename) {
      String *replacementname = Copy(interfacename);
      Replaceall(tm, interfacenamespecialvariable, replacementname);
      Delete(replacementname);
    }
  }

  /* -----------------------------------------------------------------------------
   * emitTypeWrapperClass()
   * ----------------------------------------------------------------------------- */

  void emitTypeWrapperClass(String *classname, SwigType *type) {
    Node *n = NewHash();
    Setfile(n, input_file);
    Setline(n, line_number);

    String *swigtype = NewString("");
    File *f_swigtype = getOutputFile(SWIG_output_directory(), classname);

    addOpenNamespace(0, f_swigtype);

    // Pure C# baseclass and interfaces
    const String *pure_baseclass = typemapLookup(n, "csbase", type, WARN_NONE);
    const String *pure_interfaces = typemapLookup(n, "csinterfaces", type, WARN_NONE);

    // Emit the class
    Printv(swigtype, typemapLookup(n, "csimports", type, WARN_NONE),	// Import statements
	   "\n", NIL);

    // Class attributes
    const String *csattributes = typemapLookup(n, "csattributes", type, WARN_NONE);
    if (csattributes && *Char(csattributes))
      Printf(swigtype, "%s\n", csattributes);

    Printv(swigtype, typemapLookup(n, "csclassmodifiers", type, WARN_CSHARP_TYPEMAP_CLASSMOD_UNDEF),	// Class modifiers
	   " $csclassname",	// Class name and base class
	   (*Char(pure_baseclass) || *Char(pure_interfaces)) ? " : " : "", pure_baseclass, ((*Char(pure_baseclass)) && *Char(pure_interfaces)) ?	// Interfaces
	   ", " : "", pure_interfaces, " {", typemapLookup(n, "csbody", type, WARN_CSHARP_TYPEMAP_CSBODY_UNDEF),	// main body of class
	   typemapLookup(n, "cscode", type, WARN_NONE),	// extra C# code
	   "}\n", NIL);

    Replaceall(swigtype, "$csclassname", classname);
    Replaceall(swigtype, "$module", module_class_name);
    Replaceall(swigtype, "$imclassname", imclass_name);
    Replaceall(swigtype, "$dllimport", dllimport);

    // For unknown enums
    Replaceall(swigtype, "$enumvalues", "");

    Printv(f_swigtype, swigtype, NIL);

    addCloseNamespace(0, f_swigtype);
    if (f_swigtype != f_single_out)
      Delete(f_swigtype);
    f_swigtype = NULL;
    Delete(swigtype);
    Delete(n);
  }

  /* -----------------------------------------------------------------------------
   * typemapLookup()
   * n - for input only and must contain info for Getfile(n) and Getline(n) to work
   * tmap_method - typemap method name
   * type - typemap type to lookup
   * warning - warning number to issue if no typemaps found
   * typemap_attributes - the typemap attributes are attached to this node and will 
   *   also be used for temporary storage if non null
   * return is never NULL, unlike Swig_typemap_lookup()
   * ----------------------------------------------------------------------------- */

  const String *typemapLookup(Node *n, const_String_or_char_ptr tmap_method, SwigType *type, int warning, Node *typemap_attributes = 0) {
    Node *node = !typemap_attributes ? NewHash() : typemap_attributes;
    Setattr(node, "type", type);
    Setfile(node, Getfile(n));
    Setline(node, Getline(n));
    const String *tm = Swig_typemap_lookup(tmap_method, node, "", 0);
    if (!tm) {
      tm = empty_string;
      if (warning != WARN_NONE)
	Swig_warning(warning, Getfile(n), Getline(n), "No %s typemap defined for %s\n", tmap_method, SwigType_str(type, 0));
    }
    if (!typemap_attributes)
      Delete(node);
    return tm;
  }

  /* -----------------------------------------------------------------------------
   * canThrow()
   * Determine whether the code in the typemap can throw a C# exception.
   * If so, note it for later when excodeSubstitute() is called.
   * ----------------------------------------------------------------------------- */

  void canThrow(Node *n, const String *typemap, Node *parameter) {
    String *canthrow_attribute = NewStringf("tmap:%s:canthrow", typemap);
    String *canthrow = Getattr(parameter, canthrow_attribute);
    if (canthrow)
      Setattr(n, "csharp:canthrow", "1");
    Delete(canthrow_attribute);
  }

  /* -----------------------------------------------------------------------------
   * excodeSubstitute()
   * If a method can throw a C# exception, additional exception code is added to
   * check for the pending exception so that it can then throw the exception. The
   * $excode special variable is replaced by the exception code in the excode
   * typemap attribute.
   * ----------------------------------------------------------------------------- */

  void excodeSubstitute(Node *n, String *code, const String *typemap, Node *parameter) {
    String *excode_attribute = NewStringf("tmap:%s:excode", typemap);
    String *excode = Getattr(parameter, excode_attribute);
    if (Getattr(n, "csharp:canthrow")) {
      int count = Replaceall(code, "$excode", excode);
      if (count < 1 || !excode) {
	Swig_warning(WARN_CSHARP_EXCODE, input_file, line_number,
		     "C# exception may not be thrown - no $excode or excode attribute in '%s' typemap.\n", typemap);
      }
    } else {
      Replaceall(code, "$excode", empty_string);
    }
    Delete(excode_attribute);
  }

  /* -----------------------------------------------------------------------------
   * addOpenNamespace()
   * ----------------------------------------------------------------------------- */

  void addOpenNamespace(const String *nspace, File *file) {
    if (namespce || nspace) {
      Printf(file, "namespace ");
      if (namespce)
	Printv(file, namespce, nspace ? "." : "", NIL);
      if (nspace)
	Printv(file, nspace, NIL);
      Printf(file, " {\n");
    }
  }

  /* -----------------------------------------------------------------------------
   * addCloseNamespace()
   * ----------------------------------------------------------------------------- */

  void addCloseNamespace(const String *nspace, File *file) {
    if (namespce || nspace)
      Printf(file, "\n}\n");
  }

  /* -----------------------------------------------------------------------------
   * outputDirectory()
   *
   * Return the directory to use for generating C# classes/enums and create the
   * subdirectory (does not create if language specific outdir does not exist).
   * ----------------------------------------------------------------------------- */

  String *outputDirectory(String *nspace) {
    String *output_directory = Copy(SWIG_output_directory());
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

  /*----------------------------------------------------------------------
   * Start of director methods
   *--------------------------------------------------------------------*/

#if 0
  /*----------------------------------------------------------------------
   * emitDirectorUpcalls()
   *--------------------------------------------------------------------*/

  void emitDirectorUpcalls() {
    if (n_dmethods) {
      Wrapper *w = NewWrapper();
      String *dmethod_data = NewString("");
      int n_methods = 0;
      Iterator udata_iter;

      udata_iter = First(dmethods_seq);
      while (udata_iter.item) {
	UpcallData *udata = udata_iter.item;
	Printf(dmethod_data, "  { \"%s\", \"%s\" }", Getattr(udata, "imclass_method"), Getattr(udata, "imclass_fdesc"));
	++n_methods;

	udata_iter = Next(udata_iter);

	if (udata_iter.item)
	  Putc(',', dmethod_data);
	Putc('\n', dmethod_data);
      }


      Wrapper_print(w, f_wrappers);
      Delete(dmethod_data);
      Delete(swig_module_init);
      DelWrapper(w);
    }
  }
#endif

  /*----------------------------------------------------------------------
   * emitDirectorExtraMethods()
   *
   * This is where the director connect method is generated.
   *--------------------------------------------------------------------*/
  void emitDirectorExtraMethods(Node *n) {
    if (!Swig_directorclass(n))
      return;

    // Output the director connect method:
    String *norm_name = SwigType_namestr(Getattr(n, "name"));
    String *dirclassname = directorClassName(n);
    String *swig_director_connect = Swig_name_member(getNSpace(), getClassPrefix(), "director_connect");
    String *wname = Swig_name_wrapper(swig_director_connect);
    String *sym_name = Getattr(n, "sym:name");
    String *qualified_classname = Copy(sym_name);
    String *nspace = getNSpace();
    String *dirClassName = directorClassName(n);
    String *smartptr = Getattr(n, "feature:smartptr");
    if (!GetFlag(n, "feature:flatnested")) {
      for (Node *outer_class = Getattr(n, "nested:outer"); outer_class; outer_class = Getattr(outer_class, "nested:outer")) {

	Push(qualified_classname, ".");
	Push(qualified_classname, Getattr(outer_class, "sym:name"));
      }
    }
    if (nspace)
      Insert(qualified_classname, 0, NewStringf("%s.", nspace));

    Printv(imclass_class_code, "\n  [global::System.Runtime.InteropServices.DllImport(\"", dllimport, "\", EntryPoint=\"", wname, "\")]\n", NIL);
    Printf(imclass_class_code, "  public static extern void %s(global::System.Runtime.InteropServices.HandleRef jarg1", swig_director_connect);

    Wrapper *code_wrap = NewWrapper();
    Printf(code_wrap->def, "SWIGEXPORT void SWIGSTDCALL %s(void *objarg", wname);

    if (smartptr) {
      Printf(code_wrap->code, "  %s *obj = (%s *)objarg;\n", smartptr, smartptr);
      Printf(code_wrap->code, "  // Keep a local instance of the smart pointer around while we are using the raw pointer\n");
      Printf(code_wrap->code, "  // Avoids using smart pointer specific API.\n");
      Printf(code_wrap->code, "  %s *director = dynamic_cast<%s *>(obj->operator->());\n", dirClassName, dirClassName);
    }
    else {
      Printf(code_wrap->code, "  %s *obj = (%s *)objarg;\n", norm_name, norm_name);
      Printf(code_wrap->code, "  %s *director = dynamic_cast<%s *>(obj);\n", dirClassName, dirClassName);
    }

    // TODO: if statement not needed?? - Java too
    Printf(code_wrap->code, "  if (director) {\n");
    Printf(code_wrap->code, "    director->swig_connect_director(");

    for (int i = first_class_dmethod; i < curr_class_dmethod; ++i) {
      UpcallData *udata = Getitem(dmethods_seq, i);
      String *methid = Getattr(udata, "class_methodidx");

      Printf(code_wrap->def, ", ");
      if (i != first_class_dmethod)
	Printf(code_wrap->code, ", ");
      Printf(code_wrap->def, "%s::SWIG_Callback%s_t callback%s", dirclassname, methid, methid);
      Printf(code_wrap->code, "callback%s", methid);
      Printf(imclass_class_code, ", %s.SwigDelegate%s_%s delegate%s", qualified_classname, sym_name, methid, methid);
    }

    Printf(code_wrap->def, ") {\n");
    Printf(code_wrap->code, ");\n");
    Printf(imclass_class_code, ");\n");
    Printf(code_wrap->code, "  }\n");
    Printf(code_wrap->code, "}\n");

    Wrapper_print(code_wrap, f_wrappers);
    DelWrapper(code_wrap);

    Delete(wname);
    Delete(swig_director_connect);
    Delete(qualified_classname);
    Delete(dirclassname);
  }

  /* ---------------------------------------------------------------
   * classDirectorMethod()
   *
   * Emit a virtual director method to pass a method call on to the 
   * underlying C# object.
   *
   * --------------------------------------------------------------- */

  int classDirectorMethod(Node *n, Node *parent, String *super) {
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
    String *pre_code = NewString("");
    String *post_code = NewString("");
    String *terminator_code = NewString("");
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
    String *jupcall_args = NewString("");
    String *imclass_dmethod;
    String *callback_typedef_parms = NewString("");
    String *delegate_parms = NewString("");
    String *proxy_method_types = NewString("");
    String *callback_def = NewString("");
    String *callback_code = NewString("");
    String *imcall_args = NewString("");
    bool ignored_method = GetFlag(n, "feature:ignore") ? true : false;

    // Kludge Alert: functionWrapper sets sym:overload properly, but it
    // isn't at this point, so we have to manufacture it ourselves. At least
    // we're consistent with the sym:overload name in functionWrapper. (?? when
    // does the overloaded method name get set?)

    imclass_dmethod = NewStringf("SwigDirector_%s", Swig_name_member(getNSpace(), getClassPrefix(), overloaded_name));

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
	       default - if a C# exception occurs, then the pointer returns
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
    tm = Swig_typemap_lookup("imtype", n, "", 0);
    if (tm) {
      String *imtypeout = Getattr(n, "tmap:imtype:out");	// the type in the imtype typemap's out attribute overrides the type in the typemap
      if (imtypeout)
	tm = imtypeout;
      const String *im_directoroutattributes = Getattr(n, "tmap:imtype:directoroutattributes");
      if (im_directoroutattributes) {
	Printf(callback_def, "  %s\n", im_directoroutattributes);
	if (!ignored_method)
	  Printf(director_delegate_definitions, "  %s\n", im_directoroutattributes);
      }

      Printf(callback_def, "  private %s SwigDirector%s(", tm, overloaded_name);
      if (!ignored_method) {
	const String *csdirectordelegatemodifiers = Getattr(n, "feature:csdirectordelegatemodifiers");
	String *modifiers = (csdirectordelegatemodifiers ? NewStringf("%s%s", csdirectordelegatemodifiers, Len(csdirectordelegatemodifiers) > 0 ? " " : "") : NewStringf("public "));
	Printf(director_delegate_definitions, "  %sdelegate %s", modifiers, tm);
	Delete(modifiers);
      }
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CSTYPE_UNDEF, input_file, line_number, "No imtype typemap defined for %s\n", SwigType_str(returntype, 0));
    }

    if ((c_ret_type = Swig_typemap_lookup("ctype", n, "", 0))) {
      if (!is_void && !ignored_method) {
	String *jretval_decl = NewStringf("%s jresult", c_ret_type);
	Wrapper_add_localv(w, "jresult", jretval_decl, "= 0", NIL);
	Delete(jretval_decl);
      }
    } else {
      Swig_warning(WARN_CSHARP_TYPEMAP_CTYPE_UNDEF, input_file, line_number, "No ctype typemap defined for %s for use in %s::%s (skipping director method)\n", 
	  SwigType_str(returntype, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
      output_director = false;
    }

    Swig_director_parms_fixup(l);

    /* Attach the standard typemaps */
    Swig_typemap_attach_parms("out", l, 0);
    Swig_typemap_attach_parms("ctype", l, 0);
    Swig_typemap_attach_parms("imtype", l, 0);
    Swig_typemap_attach_parms("cstype", l, 0);
    Swig_typemap_attach_parms("directorin", l, 0);
    Swig_typemap_attach_parms("csdirectorin", l, 0);
    Swig_typemap_attach_parms("directorargout", l, w);

    /* Preamble code */
    if (!ignored_method)
      Printf(w->code, "if (!swig_callback%s) {\n", overloaded_name);

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

    /* Go through argument list, convert from native to C# */
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

      /* And add to the upcall args */
      if (i > 0)
	Printf(jupcall_args, ", ");
      Printf(jupcall_args, "%s", arg);

      /* Get parameter's intermediary C type */
      if ((c_param_type = Getattr(p, "tmap:ctype"))) {
	String *ctypeout = Getattr(p, "tmap:ctype:out");	// the type in the ctype typemap's out attribute overrides the type in the typemap
	if (ctypeout)
	  c_param_type = ctypeout;

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

	  /* Add C type to callback typedef */
	  if (i > 0)
	    Printf(callback_typedef_parms, ", ");
	  Printf(callback_typedef_parms, "%s", c_param_type);

	  /* Add parameter to the intermediate class code if generating the
	   * intermediate's upcall code */
	  if ((tm = Getattr(p, "tmap:imtype"))) {

	    String *imtypeout = Getattr(p, "tmap:imtype:out");	// the type in the imtype typemap's out attribute overrides the type in the typemap
	    if (imtypeout)
	      tm = imtypeout;
            const String *im_directorinattributes = Getattr(p, "tmap:imtype:directorinattributes");

	    String *din = Copy(Getattr(p, "tmap:csdirectorin"));

	    if (din) {
	      Replaceall(din, "$module", module_class_name);
	      Replaceall(din, "$imclassname", imclass_name);
	      substituteClassname(pt, din);
	      Replaceall(din, "$iminput", ln);

	      // pre and post attribute support
	      String *pre = Getattr(p, "tmap:csdirectorin:pre");
	      if (pre) {
		substituteClassname(pt, pre);
		Replaceall(pre, "$iminput", ln);
		if (Len(pre_code) > 0)
		  Printf(pre_code, "\n");
		Printv(pre_code, pre, NIL);
	      }
	      String *post = Getattr(p, "tmap:csdirectorin:post");
	      if (post) {
		substituteClassname(pt, post);
		Replaceall(post, "$iminput", ln);
		if (Len(post_code) > 0)
		  Printf(post_code, "\n");
		Printv(post_code, post, NIL);
	      }
	      String *terminator = Getattr(p, "tmap:csdirectorin:terminator");
	      if (terminator) {
		substituteClassname(pt, terminator);
		Replaceall(terminator, "$iminput", ln);
		if (Len(terminator_code) > 0)
		Insert(terminator_code, 0, "\n");
		Insert(terminator_code, 0, terminator);
	      }

	      if (i > 0) {
		Printf(delegate_parms, ", ");
		Printf(proxy_method_types, ", ");
		Printf(imcall_args, ", ");
	      }
	      Printf(delegate_parms, "%s%s %s", im_directorinattributes ? im_directorinattributes : empty_string, tm, ln);

	      if (Cmp(din, ln)) {
		Printv(imcall_args, din, NIL);
	      } else
		Printv(imcall_args, ln, NIL);

	      /* Get the C# parameter type */
	      if ((tm = Getattr(p, "tmap:cstype"))) {
		substituteClassname(pt, tm);
		if (Strncmp(tm, "ref ", 4) == 0) {
		  Replace(tm, "ref ", "", DOH_REPLACE_FIRST);
		  Printf(proxy_method_types, "typeof(%s).MakeByRefType()", tm);
		} else if (Strncmp(tm, "out ", 4) == 0) {
		  Replace(tm, "out ", "", DOH_REPLACE_FIRST);
		  Printf(proxy_method_types, "typeof(%s).MakeByRefType()", tm);
		} else {
		  Printf(proxy_method_types, "typeof(%s)", tm);
		}
	      } else {
		Swig_warning(WARN_CSHARP_TYPEMAP_CSWTYPE_UNDEF, input_file, line_number, "No cstype typemap defined for %s\n", SwigType_str(pt, 0));
	      }
	    } else {
	      Swig_warning(WARN_CSHARP_TYPEMAP_CSDIRECTORIN_UNDEF, input_file, line_number, "No csdirectorin typemap defined for %s for use in %s::%s (skipping director method)\n", 
		  SwigType_str(pt, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	      output_director = false;
	    }
	  } else {
	    Swig_warning(WARN_CSHARP_TYPEMAP_CSTYPE_UNDEF, input_file, line_number, "No imtype typemap defined for %s for use in %s::%s (skipping director method)\n", 
		SwigType_str(pt, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	    output_director = false;
	  }

	  p = Getattr(p, "tmap:directorin:next");

	} else {
	  Swig_warning(WARN_CSHARP_TYPEMAP_CSDIRECTORIN_UNDEF, input_file, line_number,
		       "No or improper directorin typemap defined for argument %s for use in %s::%s (skipping director method)\n", 
		       SwigType_str(pt, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	  p = nextSibling(p);
	  output_director = false;
	}
      } else {
	Swig_warning(WARN_CSHARP_TYPEMAP_CTYPE_UNDEF, input_file, line_number, "No ctype typemap defined for %s for use in %s::%s (skipping director method)\n", 
	    SwigType_str(pt, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
	output_director = false;
	p = nextSibling(p);
      }

      Delete(ln);
      Delete(arg);
      Delete(c_decl);
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

    /* Finish off the inherited upcall's definition */

    Printf(callback_def, "%s)", delegate_parms);
    Printf(callback_def, " {\n");

    /* Emit the intermediate class's upcall to the actual class */

    String *upcall = NewStringf("%s(%s)", symname, imcall_args);

    if ((tm = Swig_typemap_lookup("csdirectorout", n, "", 0))) {
      substituteClassname(returntype, tm);
      Replaceall(tm, "$cscall", upcall);
      if (!is_void)
	Insert(tm, 0, "return ");
      Replaceall(tm, "\n ", "\n   "); // add extra indentation to code in typemap

      // pre and post attribute support
      bool is_pre_code = Len(pre_code) > 0;
      bool is_post_code = Len(post_code) > 0;
      bool is_terminator_code = Len(terminator_code) > 0;
      if (is_pre_code && is_post_code)
	Printf(callback_code, "%s\n    try {\n      %s;\n    } finally {\n%s\n    }\n", pre_code, tm, post_code);
      else if (is_pre_code)
	Printf(callback_code, "%s\n    %s;\n", pre_code, tm);
      else if (is_post_code)
	Printf(callback_code, "    try {\n      %s;\n    } finally {\n%s\n    }\n", tm, post_code);
      else
	Printf(callback_code, "    %s;\n", tm);
      if (is_terminator_code)
	Printv(callback_code, "\n", terminator_code, NIL);
    }

    Printf(callback_code, "  }\n");
    Delete(upcall);

    if (!ignored_method) {
      if (!is_void)
	Printf(w->code, "jresult = (%s) ", c_ret_type);

      Printf(w->code, "swig_callback%s(%s);\n", overloaded_name, jupcall_args);

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
	Printv(director_delegate_callback, "\n", callback_def, callback_code, NIL);
      if (!Getattr(n, "defaultargs")) {
	Replaceall(w->code, "$symname", symname);
	Wrapper_print(w, f_directors);
	Printv(f_directors_h, declaration, NIL);
	Printv(f_directors_h, inline_extra_method, NIL);
      }
    }

    if (!ignored_method) {
      /* Emit the actual upcall through */
      UpcallData *udata = addUpcallMethod(imclass_dmethod, symname, decl, overloaded_name);
      String *methid = Getattr(udata, "class_methodidx");
      Setattr(n, "upcalldata", udata);
      /*
      Printf(stdout, "setting upcalldata, nodeType: %s %s::%s %p\n", nodeType(n), classname, Getattr(n, "name"), n);
      */

      Printf(director_callback_typedefs, "    typedef %s (SWIGSTDCALL* SWIG_Callback%s_t)(", c_ret_type, methid);
      Printf(director_callback_typedefs, "%s);\n", callback_typedef_parms);
      Printf(director_callbacks, "    SWIG_Callback%s_t swig_callback%s;\n", methid, overloaded_name);

      Printf(director_delegate_definitions, " SwigDelegate%s_%s(%s);\n", classname, methid, delegate_parms);
      Printf(director_delegate_instances, "  private SwigDelegate%s_%s swigDelegate%s;\n", classname, methid, methid);
      Printf(director_method_types, "  private static global::System.Type[] swigMethodTypes%s = new global::System.Type[] { %s };\n", methid, proxy_method_types);
      Printf(director_connect_parms, "SwigDirector%s%s delegate%s", classname, methid, methid);
    }

    Delete(pre_code);
    Delete(post_code);
    Delete(terminator_code);
    Delete(qualified_return);
    Delete(declaration);
    Delete(callback_typedef_parms);
    Delete(delegate_parms);
    Delete(proxy_method_types);
    Delete(callback_def);
    Delete(callback_code);
    Delete(dirclassname);
    DelWrapper(w);

    return status;
  }

  /* ------------------------------------------------------------
   * classDirectorConstructor()
   * ------------------------------------------------------------ */

  int classDirectorConstructor(Node *n) {
    Node *parent = parentNode(n);
    String *decl = Getattr(n, "decl");
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

	Printf(f_directors, "%s::%s : %s, %s {\n", dirclassname, target, call, Getattr(parent, "director:ctor"));
	Printf(f_directors, "  swig_init_callbacks();\n");
	Printf(f_directors, "}\n\n");

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

  /* ------------------------------------------------------------
   * classDirectorDefaultConstructor()
   * ------------------------------------------------------------ */

  int classDirectorDefaultConstructor(Node *n) {
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


  /* ------------------------------------------------------------
   * classDirectorInit()
   * ------------------------------------------------------------ */

  int classDirectorInit(Node *n) {
    Delete(none_comparison);
    none_comparison = NewString("");	// not used

    Delete(director_ctor_code);
    director_ctor_code = NewString("$director_new");

    directorDeclaration(n);

    Printf(f_directors_h, "%s {\n", Getattr(n, "director:decl"));
    Printf(f_directors_h, "\npublic:\n");

    /* Keep track of the director methods for this class */
    first_class_dmethod = curr_class_dmethod = n_dmethods;

    director_callback_typedefs = NewString("");
    director_callbacks = NewString("");
    director_delegate_callback = NewString("");
    director_delegate_definitions = NewString("");
    director_delegate_instances = NewString("");
    director_method_types = NewString("");
    director_connect_parms = NewString("");

    return Language::classDirectorInit(n);
  }

  int classDeclaration(Node *n) {
    String *old_director_callback_typedefs = director_callback_typedefs;
    String *old_director_callbacks = director_callbacks;
    String *old_director_delegate_callback = director_delegate_callback;
    String *old_director_delegate_definitions = director_delegate_definitions;
    String *old_director_delegate_instances = director_delegate_instances;
    String *old_director_method_types = director_method_types;
    String *old_director_connect_parms = director_connect_parms;

    int ret = Language::classDeclaration(n);

    // these variables are deleted in emitProxyClassDefAndCPPCasts, hence no Delete here
    director_callback_typedefs = old_director_callback_typedefs;
    director_callbacks = old_director_callbacks;
    director_delegate_callback = old_director_delegate_callback;
    director_delegate_definitions = old_director_delegate_definitions;
    director_delegate_instances = old_director_delegate_instances;
    director_method_types = old_director_method_types;
    director_connect_parms = old_director_connect_parms;

    return ret;
  }

  /* ----------------------------------------------------------------------
   * classDirectorDestructor()
   * ---------------------------------------------------------------------- */

  int classDirectorDestructor(Node *n) {
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

  /* ------------------------------------------------------------
   * classDirectorEnd()
   * ------------------------------------------------------------ */

  int classDirectorEnd(Node *n) {
    int i;
    String *dirclassname = directorClassName(n);

    Wrapper *w = NewWrapper();

    if (Len(director_callback_typedefs) > 0) {
      Printf(f_directors_h, "\n%s", director_callback_typedefs);
    }

    Printf(f_directors_h, "    void swig_connect_director(");

    Printf(w->def, "void %s::swig_connect_director(", dirclassname);

    for (i = first_class_dmethod; i < curr_class_dmethod; ++i) {
      UpcallData *udata = Getitem(dmethods_seq, i);
      String *methid = Getattr(udata, "class_methodidx");
      String *overname = Getattr(udata, "overname");

      Printf(f_directors_h, "SWIG_Callback%s_t callback%s", methid, overname);
      Printf(w->def, "SWIG_Callback%s_t callback%s", methid, overname);
      Printf(w->code, "swig_callback%s = callback%s;\n", overname, overname);
      if (i != curr_class_dmethod - 1) {
	Printf(f_directors_h, ", ");
	Printf(w->def, ", ");
      }
    }

    Printf(f_directors_h, ");\n");
    Printf(w->def, ") {");


    if (Len(director_callbacks) > 0) {
      Printf(f_directors_h, "\nprivate:\n%s", director_callbacks);
    }
    Printf(f_directors_h, "    void swig_init_callbacks();\n");
    Printf(f_directors_h, "};\n\n");
    Printf(w->code, "}\n\n");

    Printf(w->code, "void %s::swig_init_callbacks() {\n", dirclassname);
    for (i = first_class_dmethod; i < curr_class_dmethod; ++i) {
      UpcallData *udata = Getitem(dmethods_seq, i);
      String *overname = Getattr(udata, "overname");
      Printf(w->code, "swig_callback%s = 0;\n", overname);
    }
    Printf(w->code, "}");

    Wrapper_print(w, f_directors);

    DelWrapper(w);
    Delete(dirclassname);

    return Language::classDirectorEnd(n);
  }

  /* --------------------------------------------------------------------
   * classDirectorDisown()
   * ------------------------------------------------------------------*/
  virtual int classDirectorDisown(Node *n) {
    (void) n;
    return SWIG_OK;
  }

  /*----------------------------------------------------------------------
   * extraDirectorProtectedCPPMethodsRequired()
   *--------------------------------------------------------------------*/

  bool extraDirectorProtectedCPPMethodsRequired() const {
    return false;
  }

  /*----------------------------------------------------------------------
   * directorDeclaration()
   *
   * Generate the director class's declaration
   * e.g. "class SwigDirector_myclass : public myclass, public Swig::Director {"
   *--------------------------------------------------------------------*/
 
  void directorDeclaration(Node *n) {

    String *base = Getattr(n, "classtype");
    String *class_ctor = NewString("Swig::Director()");

    String *dirclassname = directorClassName(n);
    String *declaration = Swig_class_declaration(n, dirclassname);

    Printf(declaration, " : public %s, public Swig::Director", base);

    // Stash stuff for later.
    Setattr(n, "director:decl", declaration);
    Setattr(n, "director:ctor", class_ctor);

    Delete(dirclassname);
  }

  /*----------------------------------------------------------------------
   * nestedClassesSupport()
   *--------------------------------------------------------------------*/

  NestedClassSupport nestedClassesSupport() const {
    return NCS_Full;
  }
};				/* class CSHARP */

/* -----------------------------------------------------------------------------
 * swig_csharp()    - Instantiate module
 * ----------------------------------------------------------------------------- */

static Language *new_swig_csharp() {
  return new CSHARP();
}
extern "C" Language *swig_csharp(void) {
  return new_swig_csharp();
}

/* -----------------------------------------------------------------------------
 * Static member variables
 * ----------------------------------------------------------------------------- */

const char *CSHARP::usage = "\
C# Options (available with -csharp)\n\
     -dllimport <dl> - Override DllImport attribute name to <dl>\n\
     -namespace <nm> - Generate wrappers into C# namespace <nm>\n\
     -noproxy        - Generate the low-level functional interface instead\n\
                       of proxy classes\n\
     -oldvarnames    - Old intermediary method names for variable wrappers\n\
     -outfile <file> - Write all C# into a single <file> located in the output directory\n\
\n";
