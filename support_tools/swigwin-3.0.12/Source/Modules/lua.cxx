/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * lua.cxx
 *
 * Lua language module for SWIG.
 * ----------------------------------------------------------------------------- */

/* NEW LANGUAGE NOTE:
 * ver001
   this is simply a copy of tcl8.cxx, which has been renamed
 * ver002
   all non essential code commented out, program now does virtually nothing
   it prints to stderr the list of functions to wrap, but does not create
   the XXX_wrap.c file
 * ver003
   added back top(), still prints the list of fns to stderr
   but now creates a rather empty XXX_wrap.c with some basic boilerplate code
 * ver004
   very basic version of functionWrapper()
   also uncommented usage_string() to keep compiler happy
   this will start producing proper looking code soon (I hope)
   produced the wrapper code, but without any type conversion (in or out)
   generates a few warning because of no wrappering
   does not generate SWIG_init()
   reason for this is that lua.swg is empty
   we will need to add code into this to make it work
 * ver005/6
   massive rework, basing work on the pike module instead of tcl
   (pike module it only 1/3 of the size)(though not as complete)
 * ver007
   added simple type checking
 * ver008
   INPUT, OUTPUT, INOUT typemaps handled (though not all types yet)
 * ver009
   class support: ok for basic types, but methods still TDB
   (code is VERY messed up & needs to be cleaned)
 * ver010
   Added support for embedded Lua. Try swig -lua -help for more information
*/

#include "swigmod.h"
#include "cparse.h"

/**** Diagnostics:
  With the #define REPORT(), you can change the amount of diagnostics given
  This helps me search the parse tree & figure out what is going on inside SWIG
  (because its not clear or documented)
*/
#define REPORT(T,D)		// no info:
//#define REPORT(T,D)   {Printf(stdout,T"\n");} // only title
//#define REPORT(T,D)           {Printf(stdout,T" %p\n",n);} // title & pointer
//#define REPORT(T,D)   {Printf(stdout,T"\n");display_mapping(D);}      // the works
//#define REPORT(T,D)   {Printf(stdout,T"\n");if(D)Swig_print_node(D);}      // the works

void display_mapping(DOH *d) {
  if (d == 0 || !DohIsMapping(d))
    return;
  for (Iterator it = First(d); it.item; it = Next(it)) {
    if (DohIsString(it.item))
      Printf(stdout, "  %s = %s\n", it.key, it.item);
    else if (DohIsMapping(it.item))
      Printf(stdout, "  %s = <mapping>\n", it.key);
    else if (DohIsSequence(it.item))
      Printf(stdout, "  %s = <sequence>\n", it.key);
    else
      Printf(stdout, "  %s = <unknown>\n", it.key);
  }
}

extern "C"
{
  static int compareByLen(const DOH *f, const DOH *s) {
    return Len(s) - Len(f);
  }
}


/* NEW LANGUAGE NOTE:***********************************************
 most of the default options are handled by SWIG
 you can add new ones here
 (though for now I have not bothered)
NEW LANGUAGE NOTE:END ************************************************/
static const char *usage = "\
Lua Options (available with -lua)\n\
     -elua           - Generates LTR compatible wrappers for smaller devices running elua\n\
     -eluac          - LTR compatible wrappers in \"crass compress\" mode for elua\n\
     -elua-emulate   - Emulates behaviour of eLua. Useful only for testing.\n\
                       Incompatible with -elua/-eluac options.\n\
     -nomoduleglobal - Do not register the module name as a global variable \n\
                       but return the module table from calls to require.\n\
     -no-old-metatable-bindings\n\
                     - Disable support for old-style bindings name generation, some\n\
                       old-style members scheme etc.\n\
     -squash-bases   - Squashes symbols from all inheritance tree of a given class\n\
                       into itself. Emulates pre-SWIG3.0 inheritance. Insignificantly\n\
                       speeds things up, but increases memory consumption.\n\
\n";

static int nomoduleglobal = 0;
static int elua_ltr = 0;
static int eluac_ltr = 0;
static int elua_emulate = 0;
static int squash_bases = 0;
/* The new metatable bindings were introduced in SWIG 3.0.0.
 * old_metatable_bindings in v2: 
 *                    1. static methods will be put into the scope their respective class
 *                    belongs to as well as into the class scope itself. (only for classes without %nspace given)
 *                    2. The layout in elua mode is somewhat different
 */
static int old_metatable_bindings = 1;
static int old_compatible_names = 1; // This flag can temporarily disable backward compatible names generation if old_metatable_bindings is enabled

/* NEW LANGUAGE NOTE:***********************************************
 To add a new language, you need to derive your class from
 Language and the overload various virtual functions
 (more on this as I figure it out)
NEW LANGUAGE NOTE:END ************************************************/

class LUA:public Language {
private:

  File *f_begin;
  File *f_runtime;
  File *f_header;
  File *f_wrappers;
  File *f_init;
  File *f_initbeforefunc;
  String *s_luacode;		// luacode to be called during init
  String *module;		//name of the module

  // Parameters for current class. NIL if not parsing class
  int have_constructor;
  int have_destructor;
  String *destructor_action;
  // This variable holds the name of the current class in Lua. Usually it is
  // the same as C++ class name, but rename directives can change it.
  String *proxy_class_name;
  // This is a so calld fully qualified symname - the above proxy class name
  // prepended with class namespace. If class Lua name is the same as class C++ name,
  // then it is basically C++ fully qualified name with colons replaced with dots.
  String *full_proxy_class_name;	
  // All static methods and/or variables are treated as if they were in the
  // special C++ namespace $(classname).SwigStatic. This is internal mechanism only
  // and is not visible to user in any manner. This variable holds the name
  // of such pseudo-namespace a.k.a the result of above expression evaluation
  String *class_static_nspace;
  // This variable holds the name of generated C function that acts as a constructor
  // for the currently parsed class.
  String *constructor_name;

  // Many wrappers forward calls to each other, for example staticmembervariableHandler
  // forwards calls to variableHandler, which, in turn, makes to call to functionWrapper.
  // In order to access information about whether it is a static member of class or just 
  // a plain old variable, the current array is kept and used as a 'log' of the call stack.
  enum TState {
    NO_CPP,
    VARIABLE,
    GLOBAL_FUNC,
    GLOBAL_VAR,
    MEMBER_FUNC,
    CONSTRUCTOR,
    DESTRUCTOR,
    MEMBER_VAR,
    STATIC_FUNC,
    STATIC_VAR,
    STATIC_CONST,		// enums and things like static const int x = 5;
    ENUM_CONST, // This is only needed for backward compatibility in C mode

    STATES_COUNT
  };
  bool current[STATES_COUNT];

public:

  /* ---------------------------------------------------------------------
   * LUA()
   *
   * Initialize member data
   * --------------------------------------------------------------------- */

   LUA():
      f_begin(0),
      f_runtime(0),
      f_header(0),
      f_wrappers(0),
      f_init(0),
      f_initbeforefunc(0),
      s_luacode(0),
      module(0),
      have_constructor(0),
      have_destructor(0),
      destructor_action(0),
      proxy_class_name(0),
      full_proxy_class_name(0),
      class_static_nspace(0),
      constructor_name(0) {
    for (int i = 0; i < STATES_COUNT; i++)
      current[i] = false;
  }

  /* NEW LANGUAGE NOTE:***********************************************
     This is called to initialise the system & read any command line args
     most of this is boilerplate code, except the command line args
     which depends upon what args your code supports
     NEW LANGUAGE NOTE:END *********************************************** */

  /* ---------------------------------------------------------------------
   * main()
   *
   * Parse command line options and initializes variables.
   * --------------------------------------------------------------------- */

  virtual void main(int argc, char *argv[]) {

    /* Set location of SWIG library */
    SWIG_library_directory("lua");

    /* Look for certain command line options */
    for (int i = 1; i < argc; i++) {
      if (argv[i]) {
	if (strcmp(argv[i], "-help") == 0) {	// usage flags
	  fputs(usage, stdout);
	} else if (strcmp(argv[i], "-nomoduleglobal") == 0) {
	  nomoduleglobal = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-elua") == 0) {
	  elua_ltr = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-eluac") == 0) {
	  eluac_ltr = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-no-old-metatable-bindings") == 0) {
	  Swig_mark_arg(i);
	  old_metatable_bindings = 0;
	} else if (strcmp(argv[i], "-squash-bases") == 0) {
	  Swig_mark_arg(i);
	  squash_bases = 1;
	} else if (strcmp(argv[i], "-elua-emulate") == 0) {
	  Swig_mark_arg(i);
	  elua_emulate = 1;
	}
      }
    }

    if (elua_emulate && (eluac_ltr || elua_ltr )) {
      Printf(stderr, "Cannot have -elua-emulate with either -eluac or -elua\n");
      Swig_arg_error();
    }

    // Set elua_ltr if elua_emulate is requested
    if(elua_emulate)
      elua_ltr = 1;

    /* NEW LANGUAGE NOTE:***********************************************
       This is the boilerplate code, setting a few #defines
       and which lib directory to use
       the SWIG_library_directory() is also boilerplate code
       but it always seems to be the first line of code
       NEW LANGUAGE NOTE:END *********************************************** */
    /* Add a symbol to the parser for conditional compilation */
    Preprocessor_define("SWIGLUA 1", 0);

    /* Set language-specific configuration file */
    SWIG_config_file("lua.swg");

    /* Set typemap language */
    SWIG_typemap_lang("lua");

    /* Enable overloaded methods support */
    allow_overloading();
  }




  /* NEW LANGUAGE NOTE:***********************************************
     After calling main, SWIG parses the code to wrap (I believe)
     then calls top()
     in this is more boilerplate code to set everything up
     and a call to Language::top()
     which begins the code generations by calling the member fns
     after all that is more boilerplate code to close all down
     (overall there is virtually nothing here that needs to be edited
     just use as is)
     NEW LANGUAGE NOTE:END *********************************************** */
  /* ---------------------------------------------------------------------
   * top()
   * --------------------------------------------------------------------- */

  virtual int top(Node *n) {
    /* Get the module name */
    module = Getattr(n, "name");

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
    f_header = NewString("");
    f_wrappers = NewString("");
    f_initbeforefunc = NewString("");

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("header", f_header);
    Swig_register_filebyname("wrapper", f_wrappers);
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", f_init);
    Swig_register_filebyname("initbeforefunc", f_initbeforefunc);


    s_luacode = NewString("");
    Swig_register_filebyname("luacode", s_luacode);

    current[NO_CPP] = true;

    /* Standard stuff for the SWIG runtime section */
    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGLUA\n#define SWIGLUA\n#endif\n\n");

    emitLuaFlavor(f_runtime);

    if (nomoduleglobal) {
      Printf(f_runtime, "#define SWIG_LUA_NO_MODULE_GLOBAL\n");
    } else {
      Printf(f_runtime, "#define SWIG_LUA_MODULE_GLOBAL\n");
    }
    if (squash_bases)
      Printf(f_runtime, "#define SWIG_LUA_SQUASH_BASES\n");

    //    if (NoInclude) {
    //      Printf(f_runtime, "#define SWIG_NOINCLUDE\n");
    //    }

    Printf(f_runtime, "\n");

    //String *init_name = NewStringf("%(title)s_Init", module);
    //Printf(f_header, "#define SWIG_init    %s\n", init_name);
    //Printf(f_header, "#define SWIG_name    \"%s\"\n", module);
    /* SWIG_import is a special function name for importing within Lua5.1 */
    //Printf(f_header, "#define SWIG_import  luaopen_%s\n\n", module);
    Printf(f_header, "#define SWIG_name      \"%s\"\n", module);
    Printf(f_header, "#define SWIG_init      luaopen_%s\n", module);
    Printf(f_header, "#define SWIG_init_user luaopen_%s_user\n\n", module);
    Printf(f_header, "#define SWIG_LUACODE   luaopen_%s_luacode\n", module);

    Printf(f_wrappers, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n");

    /* %init code inclusion, effectively in the SWIG_init function */
    Printf(f_init, "void SWIG_init_user(lua_State* L)\n{\n");
    Language::top(n);
    Printf(f_init, "/* exec Lua code if applicable */\nSWIG_Lua_dostring(L,SWIG_LUACODE);\n");
    Printf(f_init, "}\n");

    // Done.  Close up the module & write to the wrappers
    closeNamespaces(f_wrappers);
    Printf(f_wrappers, "#ifdef __cplusplus\n}\n#endif\n");

    SwigType_emit_type_table(f_runtime, f_wrappers);

    /* NEW LANGUAGE NOTE:***********************************************
       this basically combines several of the strings together
       and then writes it all to a file
       NEW LANGUAGE NOTE:END *********************************************** */
    Dump(f_runtime, f_begin);
    Dump(f_header, f_begin);
    Dump(f_wrappers, f_begin);
    Dump(f_initbeforefunc, f_begin);
    /* for the Lua code it needs to be properly escaped to be added into the C/C++ code */
    escapeCode(s_luacode);
    Printf(f_begin, "const char* SWIG_LUACODE=\n  \"%s\";\n\n", s_luacode);
    Wrapper_pretty_print(f_init, f_begin);
    /* Close all of the files */
    Delete(s_luacode);
    Delete(f_header);
    Delete(f_wrappers);
    Delete(f_init);
    Delete(f_initbeforefunc);
    Delete(f_runtime);
    Delete(f_begin);

    /* Done */
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * importDirective()
   * ------------------------------------------------------------ */

  virtual int importDirective(Node *n) {
    return Language::importDirective(n);
  }

  /* ------------------------------------------------------------
   * cDeclaration()
   * It copies sym:name to lua:name to preserve its original value
   * ------------------------------------------------------------ */

  virtual int cDeclaration(Node *n) {
    // class 'Language' is messing with symname in a really heavy way.
    // Although documentation states that sym:name is a name in
    // the target language space, it is not true. sym:name and
    // its derivatives are used in various places, including
    // behind-the-scene C code generation. The best way is not to
    // touch it at all.
    // But we need to know what was the name of function/variable
    // etc that user desired, that's why we store correct symname
    // as lua:name
    String *symname = Getattr(n, "sym:name");
    if (symname)
      Setattr(n, "lua:name", symname);
    return Language::cDeclaration(n);
  }
  virtual int constructorDeclaration(Node *n) {
    Setattr(n, "lua:name", Getattr(n, "sym:name"));
    return Language::constructorDeclaration(n);
  }
  virtual int destructorDeclaration(Node *n) {
    Setattr(n, "lua:name", Getattr(n, "sym:name"));
    return Language::destructorDeclaration(n);
  }
  /* NEW LANGUAGE NOTE:***********************************************
     This is it!
     you get this one right, and most of your work is done
     but its going to take some file to get it working right
     quite a bit of this is generally boilerplate code
     (or stuff I don't understand)
     that which matters will have extra added comments
     NEW LANGUAGE NOTE:END *********************************************** */
  /* ---------------------------------------------------------------------
   * functionWrapper()
   *
   * Create a function declaration and register it with the interpreter.
   * --------------------------------------------------------------------- */

  /* -----------------------------------------------------------------------
   * registerMethod()
   *
   * Determines wrap name of a method, its scope etc and calls
   * registerMethod overload with correct arguments
   * Overloaded variant adds method to the "methods" array of specified lua scope/class
   * ---------------------------------------------------------------------- */

  void registerMethod(Node *n, bool overwrite = false, String *overwriteLuaScope = 0) {
    String *symname = Getattr(n, "sym:name");
    assert(symname);

    if (Getattr(n, "sym:nextSibling"))
      return;

    // Lua scope. It is not symbol NSpace, it is the actual key to retrieve getCArraysHash.
    String *luaScope = luaCurrentSymbolNSpace();
    if (overwrite)
      luaScope = overwriteLuaScope;

    String *wrapname = 0;
    String *mrename;
    if (current[NO_CPP] || !getCurrentClass()) {
      mrename = symname;
    } else {
      assert(!current[NO_CPP]);
      if (current[STATIC_FUNC] || current[MEMBER_FUNC]) {
	mrename = Swig_name_member(getNSpace(), getClassPrefix(), symname);
      } else {
	mrename = symname;
      }
    }
    wrapname = Swig_name_wrapper(mrename);
    registerMethod(n, wrapname, luaScope);
  }

  /* -----------------------------------------------------------------------
   * registerMethod()
   *
   * Add method to the "methods" C array of given namespace/class
   * ---------------------------------------------------------------------- */

  void registerMethod(Node *n, String* wname, String *luaScope) {
    assert(n);
    Hash *nspaceHash = getCArraysHash(luaScope);
    String *s_ns_methods_tab = Getattr(nspaceHash, "methods");
    String *lua_name = Getattr(n, "lua:name");
    if (elua_ltr || eluac_ltr)
      Printv(s_ns_methods_tab, tab4, "{LSTRKEY(\"", lua_name, "\")", ", LFUNCVAL(", wname, ")", "},\n", NIL);
    else
      Printv(s_ns_methods_tab, tab4, "{ \"", lua_name, "\", ", wname, "},\n", NIL);
    // Add to the metatable if method starts with '__'
    const char * tn = Char(lua_name);
    if (tn[0]=='_' && tn[1] == '_' && !eluac_ltr) {
      String *metatable_tab = Getattr(nspaceHash, "metatable");
      assert(metatable_tab);
      if (elua_ltr)
	Printv(metatable_tab, tab4, "{LSTRKEY(\"", lua_name, "\")", ", LFUNCVAL(", wname, ")", "},\n", NIL);
      else
	Printv(metatable_tab, tab4, "{ \"", lua_name, "\", ", wname, "},\n", NIL);
    }
  }

  virtual int functionWrapper(Node *n) {
    REPORT("functionWrapper", n);

    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    String *lua_name = Getattr(n, "lua:name");
    assert(lua_name);
    SwigType *d = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    Parm *p;
    String *tm;
    int i;
    //Printf(stdout,"functionWrapper %s %s %d\n",name,iname,current);

    String *overname = 0;
    if (Getattr(n, "sym:overloaded")) {
      overname = Getattr(n, "sym:overname");
    } else {
      if (!luaAddSymbol(lua_name, n)) {
	return SWIG_ERROR;
      }
    }

    /* NEW LANGUAGE NOTE:***********************************************
       the wrapper object holds all the wrapper code
       we need to add a couple of local variables
       NEW LANGUAGE NOTE:END *********************************************** */
    Wrapper *f = NewWrapper();
    Wrapper_add_local(f, "SWIG_arg", "int SWIG_arg = 0");


    String *wname = Swig_name_wrapper(iname);
    if (overname) {
      Append(wname, overname);
    }
    if (current[CONSTRUCTOR]) {
      if (constructor_name != 0)
	Delete(constructor_name);
      constructor_name = Copy(wname);
    }

    /* NEW LANGUAGE NOTE:***********************************************
       the format of a lua fn is:
       static int wrap_XXX(lua_State* L){...}
       this line adds this into the wrapper code
       NEW LANGUAGE NOTE:END *********************************************** */
    Printv(f->def, "static int ", wname, "(lua_State* L) {", NIL);

    /* NEW LANGUAGE NOTE:***********************************************
       this prints the list of args, eg for a C fn
       int gcd(int x,int y);
       it will print
       int arg1;
       int arg2;
       NEW LANGUAGE NOTE:END *********************************************** */
    /* Write code to extract function parameters. */
    emit_parameter_variables(l, f);

    /* Attach the standard typemaps */
    emit_attach_parmmaps(l, f);
    Setattr(n, "wrap:parms", l);

    /* Get number of required and total arguments */
    int num_arguments = emit_num_arguments(l);
    int num_required = emit_num_required(l);
    int varargs = emit_isvarargs(l);

    // Check if we have to ignore arguments that are passed by LUA.
    // Needed for unary minus, where lua passes two arguments and
    // we have to ignore the second.

    int args_to_ignore = 0;
    if (Getattr(n, "lua:ignore_args")) {
      args_to_ignore = GetInt(n, "lua:ignore_args");
    }


    /* NEW LANGUAGE NOTE:***********************************************
       from here on in, it gets rather hairy
       this is the code to convert from the scripting language to C/C++
       some of the stuff will refer to the typemaps code written in your swig file
       (lua.swg), and some is done in the code here
       I suppose you could do all the conversion in C, but it would be a nightmare to do
       NEW LANGUAGE NOTE:END *********************************************** */
    /* Generate code for argument marshalling */
    //    String *description = NewString("");
    /* NEW LANGUAGE NOTE:***********************************************
       argument_check is a new feature I added to check types of arguments:
       eg for int gcd(int,int)
       I want to check that arg1 & arg2 really are integers
       NEW LANGUAGE NOTE:END *********************************************** */
    String *argument_check = NewString("");
    String *argument_parse = NewString("");
    String *checkfn = NULL;
    char source[64];
    Printf(argument_check, "SWIG_check_num_args(\"%s\",%d,%d)\n", Swig_name_str(n), num_required + args_to_ignore, num_arguments + args_to_ignore);

    for (i = 0, p = l; i < num_arguments; i++) {

      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");
      String *ln = Getattr(p, "lname");

      /* Look for an input typemap */
      sprintf(source, "%d", i + 1);
      if ((tm = Getattr(p, "tmap:in"))) {
	Replaceall(tm, "$source", source);
	Replaceall(tm, "$target", ln);
	Replaceall(tm, "$input", source);
	Setattr(p, "emit:input", source);
	if (Getattr(p, "wrap:disown") || (Getattr(p, "tmap:in:disown"))) {
	  Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
	} else {
	  Replaceall(tm, "$disown", "0");
	}
	/* NEW LANGUAGE NOTE:***********************************************
	   look for a 'checkfn' typemap
	   this an additional parameter added to the in typemap
	   if found the type will be tested for
	   this will result in code either in the
	   argument_check or argument_parse string
	   NEW LANGUAGE NOTE:END *********************************************** */
	if ((checkfn = Getattr(p, "tmap:in:checkfn"))) {
	  if (i < num_required) {
	    Printf(argument_check, "if(!%s(L,%s))", checkfn, source);
	  } else {
	    Printf(argument_check, "if(lua_gettop(L)>=%s && !%s(L,%s))", source, checkfn, source);
	  }
	  Printf(argument_check, " SWIG_fail_arg(\"%s\",%s,\"%s\");\n", Swig_name_str(n), source, SwigType_str(pt, 0));
	}
	/* NEW LANGUAGE NOTE:***********************************************
	   lua states the number of arguments passed to a function using the fn
	   lua_gettop()
	   we can use this to deal with default arguments
	   NEW LANGUAGE NOTE:END *********************************************** */
	if (i < num_required) {
	  Printf(argument_parse, "%s\n", tm);
	} else {
	  Printf(argument_parse, "if(lua_gettop(L)>=%s){%s}\n", source, tm);
	}
	p = Getattr(p, "tmap:in:next");
	continue;
      } else {
	/* NEW LANGUAGE NOTE:***********************************************
	   // why is this code not called when I don't have a typemap?
	   // instead of giving a warning, no code is generated
	   NEW LANGUAGE NOTE:END *********************************************** */
	Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
	break;
      }
    }

    // add all argcheck code
    Printv(f->code, argument_check, argument_parse, NIL);

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
	//          // managing the number of returning variables
	//        if (numoutputs=Getattr(p,"tmap:argout:numoutputs")){
	//                      int i=GetInt(p,"tmap:argout:numoutputs");
	//                      printf("got argout:numoutputs of %d\n",i);
	//                      returnval+=GetInt(p,"tmap:argout:numoutputs");
	//        }
	//        else returnval++;
	Replaceall(tm, "$source", Getattr(p, "lname"));
	Replaceall(tm, "$target", Swig_cresult_name());
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(outarg, tm, "\n", NIL);
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    // Remember C name of the wrapping function
    Setattr(n, "wrap:name", wname);

    /* Emit the function call */
    String *actioncode = emit_action(n);

    /* NEW LANGUAGE NOTE:***********************************************
       FIXME:
       returns 1 if there is a void return type
       this is because there is a typemap for void
       NEW LANGUAGE NOTE:END *********************************************** */
    // Return value if necessary
    if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
      // managing the number of returning variables
      //      if (numoutputs=Getattr(tm,"numoutputs")){
      //              int i=GetInt(tm,"numoutputs");
      //              printf("return numoutputs %d\n",i);
      //              returnval+=GetInt(tm,"numoutputs");
      //      }
      //        else returnval++;
      Replaceall(tm, "$source", Swig_cresult_name());
      if (GetFlag(n, "feature:new")) {
	Replaceall(tm, "$owner", "1");
      } else {
	Replaceall(tm, "$owner", "0");
      }
      Printf(f->code, "%s\n", tm);
      //      returnval++;
    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(d, 0), name);
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
    Printv(f->code, "return SWIG_arg;\n", NIL);
    // add the failure cleanup code:
    Printv(f->code, "\nif(0) SWIG_fail;\n", NIL);
    Printv(f->code, "\nfail:\n", NIL);
    Printv(f->code, "$cleanup", "lua_error(L);\n", NIL);
    Printv(f->code, "return SWIG_arg;\n", NIL);
    Printf(f->code, "}\n");

    /* Substitute the cleanup code */
    Replaceall(f->code, "$cleanup", cleanup);

    /* Substitute the function name */
    Replaceall(f->code, "$symname", iname);
    Replaceall(f->code, "$result", Swig_cresult_name());

    /* Dump the function out */
    /* in Lua we will not emit the destructor as a wrapper function,
       Lua will automatically call the destructor when the object is free'd
       However: you cannot just skip this function as it will not emit
       any custom destructor (using %extend), as you need to call emit_action()
       Therefore we go though the whole function, 
       but do not write the code into the wrapper
     */
    if (!current[DESTRUCTOR]) {
      Wrapper_print(f, f_wrappers);
    }

    /* NEW LANGUAGE NOTE:***********************************************
       register the function in SWIG
       different language mappings seem to use different ideas
       NEW LANGUAGE NOTE:END *********************************************** */
    /* Now register the function with the interpreter. */
    int result = SWIG_OK;
    if (Getattr(n, "sym:overloaded")) {
      if (!Getattr(n, "sym:nextSibling")) {
	result = dispatchFunction(n);
      }
    }

    Delete(argument_check);
    Delete(argument_parse);

    Delete(cleanup);
    Delete(outarg);
    //    Delete(description);
    DelWrapper(f);

    return result;
  }

  /* ------------------------------------------------------------
   * dispatchFunction()
   *
   * Emit overloading dispatch function
   * ------------------------------------------------------------ */

  /* NEW LANGUAGE NOTE:***********************************************
     This is an extra function used for overloading of functions
     it checks the args & then calls the relevant fn
     most of the real work in again typemaps:
     look for %typecheck(SWIG_TYPECHECK_*) in the .swg file
     NEW LANGUAGE NOTE:END *********************************************** */
  int dispatchFunction(Node *n) {
    //REPORT("dispatchFunction", n);
    /* Last node in overloaded chain */

    int maxargs;
    String *tmp = NewString("");
    String *dispatch = Swig_overload_dispatch(n, "return %s(L);", &maxargs);

    /* Generate a dispatch wrapper for all overloaded functions */

    Wrapper *f = NewWrapper();
    String *symname = Getattr(n, "sym:name");
    String *lua_name = Getattr(n, "lua:name");
    assert(lua_name);
    String *wname = Swig_name_wrapper(symname);

    //Printf(stdout,"Swig_overload_dispatch %s %s '%s' %d\n",symname,wname,dispatch,maxargs);

    if (!luaAddSymbol(lua_name, n)) {
      DelWrapper(f);
      Delete(dispatch);
      Delete(tmp);
      return SWIG_ERROR;
    }

    Printv(f->def, "static int ", wname, "(lua_State* L) {", NIL);
    Wrapper_add_local(f, "argc", "int argc");
    Printf(tmp, "int argv[%d]={1", maxargs + 1);
    for (int i = 1; i <= maxargs; i++) {
      Printf(tmp, ",%d", i + 1);
    }
    Printf(tmp, "}");
    Wrapper_add_local(f, "argv", tmp);
    Printf(f->code, "argc = lua_gettop(L);\n");

    Replaceall(dispatch, "$args", "self,args");
    Printv(f->code, dispatch, "\n", NIL);

    Node *sibl = n;
    while (Getattr(sibl, "sym:previousSibling"))
      sibl = Getattr(sibl, "sym:previousSibling");	// go all the way up
    String *protoTypes = NewString("");
    do {
      String *fulldecl = Swig_name_decl(sibl);
      Printf(protoTypes, "\n\"    %s\\n\"", fulldecl);
      Delete(fulldecl);
    } while ((sibl = Getattr(sibl, "sym:nextSibling")));
    Printf(f->code, "SWIG_Lua_pusherrstring(L,\"Wrong arguments for overloaded function '%s'\\n\"\n"
	   "\"  Possible C/C++ prototypes are:\\n\"%s);\n", symname, protoTypes);
    Delete(protoTypes);

    Printf(f->code, "lua_error(L);return 0;\n");
    Printv(f->code, "}\n", NIL);
    Wrapper_print(f, f_wrappers);

    // Remember C name of the wrapping function
    Setattr(n, "wrap:name", wname);

    if (current[CONSTRUCTOR]) {
      if (constructor_name != 0)
	Delete(constructor_name);
      constructor_name = Copy(wname);
    }

    DelWrapper(f);
    Delete(dispatch);
    Delete(tmp);

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * Add variable to "attributes"  C arrays of given namespace or class.
   * Input is node. Based on the state of "current" array it determines
   * the name of the getter function, setter function etc and calls
   * registerVariable overload with necessary params.
   * Lua scope could be overwritten. (Used only for backward compatibility)
   * ------------------------------------------------------------ */

  void registerVariable(Node *n, bool overwrite = false, String *overwriteLuaScope = 0) {
    int assignable = is_assignable(n);
    String *symname = Getattr(n, "sym:name");
    assert(symname);

    // Lua scope. It is not symbol NSpace, it is the actual key to retrieve getCArraysHash.
    String *luaScope = luaCurrentSymbolNSpace();
    if (overwrite)
      luaScope = overwriteLuaScope;

    // Getter and setter
    String *getName = 0;
    String *setName = 0;
    String *mrename = 0;
    if (current[NO_CPP] || !getCurrentClass()) {
      // Global variable
      getName = Swig_name_get(getNSpace(), symname);
      if (assignable)
        setName = Swig_name_set(getNSpace(), symname);
    } else {
        assert(!current[NO_CPP]);
        if (current[STATIC_VAR] ) {
          mrename = Swig_name_member(getNSpace(), getClassPrefix(), symname);
          getName = Swig_name_get(0, mrename);
          if (assignable)
            setName = Swig_name_set(0, mrename);
        } else if (current[MEMBER_VAR]) {
          mrename = Swig_name_member(0, getClassPrefix(), symname);
          getName = Swig_name_get(getNSpace(), mrename);
          if (assignable)
            setName = Swig_name_set(getNSpace(), mrename);
        } else {
          assert(false);
        }
    }

    getName = Swig_name_wrapper(getName);
    if (setName)
      setName = Swig_name_wrapper(setName);
    registerVariable(luaScope, n, getName, setName);
  }

  /* ------------------------------------------------------------
   * registerVariable()
   *
   * Add variable to the "attributes" (or "get"/"set" in
   * case of elua_ltr) C arrays of given namespace or class
   * ------------------------------------------------------------ */

  void registerVariable(String *lua_nspace_or_class_name, Node *n, String *getName, String *setName) {
    String *unassignable = NewString("SWIG_Lua_set_immutable");
    if (setName == 0 || GetFlag(n, "feature:immutable")) {
      setName = unassignable;
    }
    Hash *nspaceHash = getCArraysHash(lua_nspace_or_class_name);
    String *s_ns_methods_tab = Getattr(nspaceHash, "methods");
    String *s_ns_var_tab = Getattr(nspaceHash, "attributes");
    String *lua_name = Getattr(n, "lua:name");
    if (elua_ltr) {
      String *s_ns_dot_get = Getattr(nspaceHash, "get");
      String *s_ns_dot_set = Getattr(nspaceHash, "set");
      Printf(s_ns_dot_get, "%s{LSTRKEY(\"%s\"), LFUNCVAL(%s)},\n", tab4, lua_name, getName);
      Printf(s_ns_dot_set, "%s{LSTRKEY(\"%s\"), LFUNCVAL(%s)},\n", tab4, lua_name, setName);
    } else if (eluac_ltr) {
      Printv(s_ns_methods_tab, tab4, "{LSTRKEY(\"", lua_name, "_get", "\")", ", LFUNCVAL(", getName, ")", "},\n", NIL);
      Printv(s_ns_methods_tab, tab4, "{LSTRKEY(\"", lua_name, "_set", "\")", ", LFUNCVAL(", setName, ")", "},\n", NIL);
    } else {
      Printf(s_ns_var_tab, "%s{ \"%s\", %s, %s },\n", tab4, lua_name, getName, setName);
    }
  }

  /* ------------------------------------------------------------
   * variableWrapper()
   * ------------------------------------------------------------ */

  virtual int variableWrapper(Node *n) {
    /* NEW LANGUAGE NOTE:***********************************************
       Language::variableWrapper(n) will generate two wrapper fns
       Foo_get & Foo_set by calling functionWrapper()
       so we will just add these into the variable lists
       ideally we should not have registered these as functions,
       only WRT this variable will look into this later.
       NEW LANGUAGE NOTE:END *********************************************** */
    //    REPORT("variableWrapper", n);
    String *lua_name = Getattr(n, "lua:name");
    assert(lua_name);
    current[VARIABLE] = true;
    // let SWIG generate the wrappers
    int result = Language::variableWrapper(n);
    
    // It is impossible to use registerVariable, because sym:name of the Node is currently
    // in an undefined state - the callees of this function may have modified it.
    // registerVariable should be used from respective callees.*
    current[VARIABLE] = false;
    return result;
  }


  /* ------------------------------------------------------------
   * Add constant to appropriate C array. constantRecord is an array record.
   * Actually, in current implementation it is resolved consttab typemap
   * ------------------------------------------------------------ */

  void registerConstant(String *nspace, String *constantRecord) {
    Hash *nspaceHash = getCArraysHash(nspace);
    String *s_const_tab = 0;
    if (eluac_ltr || elua_ltr)
      // In elua everything goes to "methods" tab
      s_const_tab = Getattr(nspaceHash, "methods");
    else
      s_const_tab = Getattr(nspaceHash, "constants");

    assert(s_const_tab);
    Printf(s_const_tab, "    %s,\n", constantRecord);

    if ((eluac_ltr || elua_ltr) && old_metatable_bindings) {
      s_const_tab = Getattr(nspaceHash, "constants");
      assert(s_const_tab);
      Printf(s_const_tab, "    %s,\n", constantRecord);
    }

  }

  /* ------------------------------------------------------------
   * constantWrapper()
   * ------------------------------------------------------------ */

  virtual int constantWrapper(Node *n) {
    REPORT("constantWrapper", n);
    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    String *lua_name = Getattr(n, "lua:name");
    if (lua_name == 0)
      lua_name = iname;
    String *nsname = Copy(iname);
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");
    String *tm;
    String *lua_name_v2 = 0;
    String *tm_v2 = 0;
    String *iname_v2 = 0;
    Node *n_v2 = 0;

    if (!luaAddSymbol(lua_name, n))
      return SWIG_ERROR;

    Swig_save("lua_constantMember", n, "sym:name", NIL);
    Setattr(n, "sym:name", lua_name);
    /* Special hook for member pointer */
    if (SwigType_type(type) == T_MPOINTER) {
      String *wname = Swig_name_wrapper(iname);
      Printf(f_wrappers, "static %s = %s;\n", SwigType_str(type, wname), value);
      value = Char(wname);
    }

    if ((tm = Swig_typemap_lookup("consttab", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", lua_name);
      Replaceall(tm, "$value", value);
      Replaceall(tm, "$nsname", nsname);
      registerConstant(luaCurrentSymbolNSpace(), tm);
    } else if ((tm = Swig_typemap_lookup("constcode", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", lua_name);
      Replaceall(tm, "$value", value);
      Replaceall(tm, "$nsname", nsname);
      Printf(f_init, "%s\n", tm);
    } else {
      Delete(nsname);
      nsname = 0;
      Swig_warning(WARN_TYPEMAP_CONST_UNDEF, input_file, line_number, "Unsupported constant value.\n");
      Swig_restore(n);
      return SWIG_NOWRAP;
    }

    bool make_v2_compatible = old_metatable_bindings && getCurrentClass() && old_compatible_names;

    if (make_v2_compatible) {
      // Don't do anything for enums in C mode - they are already
      // wrapped correctly
      if (CPlusPlus || !current[ENUM_CONST]) {
	lua_name_v2 = Swig_name_member(0, proxy_class_name, lua_name);
	iname_v2 = Swig_name_member(0, proxy_class_name, iname);
        n_v2 = Copy(n);
        if (!luaAddSymbol(iname_v2, n, getNSpace())) {
          Swig_restore(n);
          return SWIG_ERROR;
        }

        Setattr(n_v2, "sym:name", lua_name_v2);
        tm_v2 = Swig_typemap_lookup("consttab", n_v2, name, 0);
        if (tm_v2) {
          Replaceall(tm_v2, "$source", value);
          Replaceall(tm_v2, "$target", lua_name_v2);
          Replaceall(tm_v2, "$value", value);
          Replaceall(tm_v2, "$nsname", nsname);
          registerConstant(getNSpace(), tm_v2);
        } else {
          tm_v2 = Swig_typemap_lookup("constcode", n_v2, name, 0);
          if (!tm_v2) {
            // This can't be.
            assert(false);
            Swig_restore(n);
            return SWIG_ERROR;
          }
          Replaceall(tm_v2, "$source", value);
          Replaceall(tm_v2, "$target", lua_name_v2);
          Replaceall(tm_v2, "$value", value);
          Replaceall(tm_v2, "$nsname", nsname);
          Printf(f_init, "%s\n", tm_v2);
        }
        Delete(n_v2);
      }
    }

    Swig_restore(n);
    Delete(nsname);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * nativeWrapper()
   * ------------------------------------------------------------ */

  virtual int nativeWrapper(Node *n) {
    //    REPORT("nativeWrapper", n);
    String *symname = Getattr(n, "sym:name");
    String *wrapname = Getattr(n, "wrap:name");
    if (!luaAddSymbol(wrapname, n))
      return SWIG_ERROR;

    Hash *nspaceHash = getCArraysHash(getNSpace());
    String *s_ns_methods_tab = Getattr(nspaceHash, "methods");
    Printv(s_ns_methods_tab, tab4, "{ \"", symname, "\",", wrapname, "},\n", NIL);
    //   return Language::nativeWrapper(n); // this does nothing...
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * enumDeclaration()
   * ------------------------------------------------------------ */

  virtual int enumDeclaration(Node *n) {
    if (getCurrentClass() && (cplus_mode != PUBLIC))
      return SWIG_NOWRAP;

    current[STATIC_CONST] = true;
    current[ENUM_CONST] = true;
    // There is some slightly specific behaviour with enums. Basically,
    // their NSpace may be tracked separately. The code below tries to work around
    // this issue to some degree.
    // The idea is the same as in classHandler - to drop old names generation if
    // enum is in class in namespace.
    const int old_compatible_names_saved = old_compatible_names;
    if (getNSpace() || ( Getattr(n, "sym:nspace") != 0 && Len(Getattr(n, "sym:nspace")) > 0 ) ) {
      old_compatible_names = 0;
    }
    int result = Language::enumDeclaration(n);
    current[STATIC_CONST] = false;
    current[ENUM_CONST] = false;
    old_compatible_names = old_compatible_names_saved;
    return result;
  }

  /* ------------------------------------------------------------
   * enumvalueDeclaration()
   * ------------------------------------------------------------ */

  virtual int enumvalueDeclaration(Node *n) {
    if (getCurrentClass() && (cplus_mode != PUBLIC))
      return SWIG_NOWRAP;

    Swig_require("enumvalueDeclaration", n, "*name", "?value", "*sym:name", NIL);
    String *symname = Getattr(n, "sym:name");
    String *value = Getattr(n, "value");
    String *name = Getattr(n, "name");
    String *tmpValue;
    Node *parent = parentNode(n);

    if (value)
      tmpValue = NewString(value);
    else
      tmpValue = NewString(name);
    Setattr(n, "value", tmpValue);

    Setattr(n, "name", tmpValue);	/* for wrapping of enums in a namespace when emit_action is used */

    if (GetFlag(parent, "scopedenum")) {
      symname = Swig_name_member(0, Getattr(parent, "sym:name"), symname);
      Setattr(n, "sym:name", symname);
      Delete(symname);
    }

    int result = constantWrapper(n);

    Delete(tmpValue);
    Swig_restore(n);
    return result;
  }

  /* ------------------------------------------------------------
   * classDeclaration()
   * ------------------------------------------------------------ */

  virtual int classDeclaration(Node *n) {
    return Language::classDeclaration(n);
  }


  /* ------------------------------------------------------------
   * Helper function that adds record to appropriate C arrays
   * ------------------------------------------------------------ */

  void registerClass(String *scope, String *wrap_class) {
    assert(wrap_class);
    Hash *nspaceHash = getCArraysHash(scope);
    String *ns_classes = Getattr(nspaceHash, "classes");
    Printv(ns_classes, "&", wrap_class, ",\n", NIL);
    if (elua_ltr || eluac_ltr) {
      String *ns_methods = Getattr(nspaceHash, "methods");
      Hash *class_hash = getCArraysHash(class_static_nspace);
      assert(class_hash);
      String *cls_methods = Getattr(class_hash, "methods:name");
      assert(cls_methods);
      Printv(ns_methods, tab4, "{LSTRKEY(\"", proxy_class_name, "\")", ", LROVAL(", cls_methods, ")", "},\n", NIL);
    }
  }

  /* ------------------------------------------------------------
   * classHandler()
   * ------------------------------------------------------------ */

  virtual int classHandler(Node *n) {
    //REPORT("classHandler", n);

    String *mangled_full_proxy_class_name = 0;
    String *destructor_name = 0;
    String *nspace = getNSpace();

    constructor_name = 0;
    have_constructor = 0;
    have_destructor = 0;
    destructor_action = 0;
    assert(class_static_nspace == 0);
    assert(full_proxy_class_name == 0);
    assert(proxy_class_name == 0);

    current[NO_CPP] = false;

    proxy_class_name = Getattr(n, "sym:name");
    // We have to enforce nspace here, because technically we are already
    // inside class parsing (getCurrentClass != 0), but we should register
    // class in its parent namespace
    if (!luaAddSymbol(proxy_class_name, n, nspace))
      return SWIG_ERROR;

    if (nspace == 0)
      full_proxy_class_name = NewStringf("%s", proxy_class_name);
    else
      full_proxy_class_name = NewStringf("%s.%s", nspace, proxy_class_name);

    assert(full_proxy_class_name);
    mangled_full_proxy_class_name = Swig_name_mangle(full_proxy_class_name);

    SwigType *t = Copy(Getattr(n, "name"));
    SwigType *fr_t = SwigType_typedef_resolve_all(t);	/* Create fully resolved type */
    SwigType *t_tmp = 0;
    t_tmp = SwigType_typedef_qualified(fr_t);	// Temporal variable
    Delete(fr_t);
    fr_t = SwigType_strip_qualifiers(t_tmp);
    String *mangled_fr_t = 0;
    mangled_fr_t = SwigType_manglestr(fr_t);
    // not sure exactly how this works,
    // but tcl has a static hashtable of all classes emitted and then only emits code for them once.
    // this fixes issues in test suites: template_default2 & template_specialization

    // * if i understand correctly, this is a bug.
    // * consider effect on template_specialization_defarg

    static Hash *emitted = NewHash();
    if (GetFlag(emitted, mangled_fr_t)) {
      full_proxy_class_name = 0;
      proxy_class_name = 0;
      return SWIG_NOWRAP;
    }
    SetFlag(emitted, mangled_fr_t);

    // We treat class T as both 'class' and 'namespace'. All static members, attributes
    // and constants are considered part of namespace T, all members - part of the 'class'
    // Now, here is a trick. Static methods, attributes and non-static methods and attributes
    // are described with same structures - swig_lua_attribute/swig_lua_method. Instead of calling
    // getCArraysHash(class name) to initialize things for static methods/attributes and then
    // manually doing same initialization for non-static methods, we call getCArraysHash 2 times:
    // 1) With name "class name" + "." + "SwigStatic" to initialize static things
    // 2) With "class name" to initialize non-static things
    Hash *instance_cls = getCArraysHash(full_proxy_class_name, false);
    assert(instance_cls);
    String *s_attr_tab_name = Getattr(instance_cls, "attributes:name");
    String *s_methods_tab_name = Getattr(instance_cls, "methods:name");
    SetFlag(instance_cls, "lua:no_namespaces");
    SetFlag(instance_cls, "lua:no_classes");
    SetFlag(instance_cls, "lua:class_instance");

    /* There is no use for "constants", "classes" and "namespaces" arrays.
     * All constants are considered part of static part of class.
     */

    class_static_nspace = NewStringf("%s%sSwigStatic", full_proxy_class_name, NSPACE_SEPARATOR);
    Hash *static_cls = getCArraysHash(class_static_nspace, false);
    assert(static_cls);
    SetFlag(static_cls, "lua:no_namespaces");
    SetFlag(static_cls, "lua:class_static");

    // Notifying instance_cls and static_cls hashes about each other
    Setattr(instance_cls, "lua:class_instance:static_hash", static_cls);
    Setattr(static_cls, "lua:class_static:instance_hash", instance_cls);

    const int old_compatible_names_saved = old_compatible_names;
    // If class has %nspace enabled, then generation of backward compatible names
    // should be disabled
    if (getNSpace()) {
      old_compatible_names = 0;
    }

    /* There is no use for "classes" and "namespaces" arrays. Subclasses are not supported
     * by SWIG and namespaces couldn't be nested inside classes (C++ Standard)
     */
    // Generate normal wrappers
    Language::classHandler(n);

    old_compatible_names = old_compatible_names_saved;

    SwigType_add_pointer(t);

    // Catch all: eg. a class with only static functions and/or variables will not have 'remembered'
    String *wrap_class_name = Swig_name_wrapper(NewStringf("class_%s", mangled_full_proxy_class_name));
    String *wrap_class = NewStringf("&%s", wrap_class_name);
    SwigType_remember_clientdata(t, wrap_class);

    String *rt = Copy(getClassType());
    SwigType_add_pointer(rt);

    // Adding class to apropriate namespace
    registerClass(nspace, wrap_class_name);
    Hash *nspaceHash = getCArraysHash(nspace);

    // Register the class structure with the type checker
    //    Printf(f_init,"SWIG_TypeClientData(SWIGTYPE%s, (void *) &_wrap_class_%s);\n", SwigType_manglestr(t), mangled_full_proxy_class_name);

    // emit a function to be called to delete the object 
    if (have_destructor) {
      destructor_name = NewStringf("swig_delete_%s", mangled_full_proxy_class_name);
      Printv(f_wrappers, "static void ", destructor_name, "(void *obj) {\n", NIL);
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
    // Wrap constructor wrapper into one more proxy function. It will be used as class namespace __call method, thus
    // allowing both
    // Module.ClassName.StaticMethod to access static method/variable/constant
    // Module.ClassName() to create new object
    if (have_constructor) {
      assert(constructor_name);
      String *constructor_proxy_name = NewStringf("_proxy_%s", constructor_name);
      Printv(f_wrappers, "static int ", constructor_proxy_name, "(lua_State *L) {\n", NIL);
      Printv(f_wrappers,
	     tab4, "assert(lua_istable(L,1));\n",
	     tab4, "lua_pushcfunction(L,", constructor_name, ");\n",
	     tab4, "assert(!lua_isnil(L,-1));\n",
	     tab4, "lua_replace(L,1); /* replace our table with real constructor */\n",
	     tab4, "lua_call(L,lua_gettop(L)-1,1);\n",
	     tab4, "return 1;\n}\n", NIL);
      Delete(constructor_name);
      constructor_name = constructor_proxy_name;
      if (elua_ltr) {
	String *static_cls_metatable_tab = Getattr(static_cls, "metatable");
	assert(static_cls_metatable_tab);
	Printf(static_cls_metatable_tab, "    {LSTRKEY(\"__call\"), LFUNCVAL(%s)},\n", constructor_name);
      } else if (eluac_ltr) {
	String *ns_methods_tab = Getattr(nspaceHash, "methods");
	assert(ns_methods_tab);
	Printv(ns_methods_tab, tab4, "{LSTRKEY(\"", "new_", proxy_class_name, "\")", ", LFUNCVAL(", constructor_name, ")", "},\n", NIL);
      }
    }
    if (have_destructor) {
      if (eluac_ltr) {
	String *ns_methods_tab = Getattr(nspaceHash, "methods");
	assert(ns_methods_tab);
	Printv(ns_methods_tab, tab4, "{LSTRKEY(\"", "free_", mangled_full_proxy_class_name, "\")", ", LFUNCVAL(", destructor_name, ")", "},\n", NIL);
      }
    }

    closeCArraysHash(full_proxy_class_name, f_wrappers);
    closeCArraysHash(class_static_nspace, f_wrappers);


    // Handle inheritance
    // note: with the idea of class hierarchies spread over multiple modules
    // cf test-suite: imports.i
    // it is not possible to just add the pointers to the base classes to the code
    // (as sometimes these classes are not present)
    // therefore we instead hold the name of the base class and a null pointer
    // at runtime: we can query the swig type manager & see if the class exists
    // if so, we can get the pointer to the base class & replace the null pointer
    // if the type does not exist, then we cannot...
    String *base_class = NewString("");
    String *base_class_names = NewString("");

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
	// stores a null pointer & the name
	Printf(base_class, "0,");
	Printf(base_class_names, "\"%s *\",", SwigType_namestr(bname));

	b = Next(b);
	index++;
      }
    }
    // First, print class static part
    printCArraysDefinition(class_static_nspace, proxy_class_name, f_wrappers);

    assert(mangled_full_proxy_class_name);
    assert(base_class);
    assert(base_class_names);
    assert(proxy_class_name);
    assert(full_proxy_class_name);
    
    // Then print class isntance part
    Printv(f_wrappers, "static swig_lua_class *swig_", mangled_full_proxy_class_name, "_bases[] = {", base_class, "0};\n", NIL);
    Delete(base_class);
    Printv(f_wrappers, "static const char *swig_", mangled_full_proxy_class_name, "_base_names[] = {", base_class_names, "0};\n", NIL);
    Delete(base_class_names);

    Printv(f_wrappers, "static swig_lua_class _wrap_class_", mangled_full_proxy_class_name, " = { \"", proxy_class_name, "\", \"", full_proxy_class_name, "\", &SWIGTYPE",
	   SwigType_manglestr(t), ",", NIL);

    if (have_constructor) {
      Printv(f_wrappers, constructor_name, NIL);
      Delete(constructor_name);
      constructor_name = 0;
    } else {
      Printf(f_wrappers, "0");
    }

    if (have_destructor) {
      Printv(f_wrappers, ", ", destructor_name, NIL);
    } else {
      Printf(f_wrappers, ",0");
    }
    Printf(f_wrappers, ", %s, %s, &%s", s_methods_tab_name, s_attr_tab_name, Getattr(static_cls, "cname"));
    
    if (!eluac_ltr) {
      Printf(f_wrappers, ", %s", Getattr(instance_cls,"metatable:name"));
    }
    else
      Printf(f_wrappers, ", 0");

    Printf(f_wrappers, ", swig_%s_bases, swig_%s_base_names };\n\n", mangled_full_proxy_class_name, mangled_full_proxy_class_name);

    current[NO_CPP] = true;
    Delete(class_static_nspace);
    class_static_nspace = 0;
    Delete(mangled_full_proxy_class_name);
    mangled_full_proxy_class_name = 0;
    Delete(destructor_name);
    destructor_name = 0;
    Delete(full_proxy_class_name);
    full_proxy_class_name = 0;
    proxy_class_name = 0;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int memberfunctionHandler(Node *n) {
    String *symname = GetChar(n, "sym:name");
    //Printf(stdout,"memberfunctionHandler %s %s\n",name,iname);

    // Special case unary minus: LUA passes two parameters for the
    // wrapper function while we want only one. Tell our
    // functionWrapper to ignore a parameter.

    if (Cmp(symname, "__unm") == 0) {
      //Printf(stdout, "unary minus: ignore one argument\n");
      SetInt(n, "lua:ignore_args", 1);
    }

    current[MEMBER_FUNC] = true;
    Language::memberfunctionHandler(n);

    registerMethod(n);
    current[MEMBER_FUNC] = false;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * membervariableHandler()
   * ------------------------------------------------------------ */

  virtual int membervariableHandler(Node *n) {
    //    REPORT("membervariableHandler",n);
    current[MEMBER_VAR] = true;
    Language::membervariableHandler(n);
    registerVariable(n);
    current[MEMBER_VAR] = false;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constructorHandler()
   *
   * Method for adding C++ member constructor
   * ------------------------------------------------------------ */

  virtual int constructorHandler(Node *n) {
    //    REPORT("constructorHandler", n);
    current[CONSTRUCTOR] = true;
    Language::constructorHandler(n);
    current[CONSTRUCTOR] = false;
    //constructor_name = NewString(Getattr(n, "sym:name"));
    have_constructor = 1;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * destructorHandler()
   * ------------------------------------------------------------ */

  virtual int destructorHandler(Node *n) {
    REPORT("destructorHandler", n);
    current[DESTRUCTOR] = true;
    Language::destructorHandler(n);
    current[DESTRUCTOR] = false;
    have_destructor = 1;
    destructor_action = Getattr(n, "wrap:action");
    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * globalfunctionHandler()
   *
   * It can be called:
   * 1. Usual C/C++ global function.
   * 2. During class parsing for functions declared/defined as friend
   * 3. During class parsing from staticmemberfunctionHandler
   * ---------------------------------------------------------------------- */

  virtual int globalfunctionHandler(Node *n) {
    bool oldVal = current[NO_CPP];
    if (!current[STATIC_FUNC])	// If static function, don't switch to NO_CPP
      current[NO_CPP] = true;
    const int result = Language::globalfunctionHandler(n);
    
    if (!current[STATIC_FUNC]) // Register only if not called from static function handler
      registerMethod(n);
    current[NO_CPP] = oldVal;
    return result;
  }

  /* ----------------------------------------------------------------------
   * globalvariableHandler()
   *
   * Sets "current" array correctly
   * ---------------------------------------------------------------------- */

  virtual int globalvariableHandler(Node *n) {
    bool oldVal = current[NO_CPP];
    current[GLOBAL_VAR] = true;
    current[NO_CPP] = true;

    const int result = Language::globalvariableHandler(n);
    registerVariable(n);

    current[GLOBAL_VAR] = false;
    current[NO_CPP] = oldVal;
    return result;
  }


  /* -----------------------------------------------------------------------
   * staticmemberfunctionHandler()
   *
   * Wrap a static C++ function
   * ---------------------------------------------------------------------- */

  virtual int staticmemberfunctionHandler(Node *n) {
    REPORT("staticmemberfunctionHandler", n);
    current[STATIC_FUNC] = true;

    const int result = Language::staticmemberfunctionHandler(n);
    registerMethod(n);

    if (old_metatable_bindings && result == SWIG_OK && old_compatible_names) {
      Swig_require("lua_staticmemberfunctionHandler", n, "*lua:name", NIL);
      String *lua_name = Getattr(n, "lua:name");
      // Although this function uses Swig_name_member, it actually generates the Lua name,
      // not the C++ name. This is because an earlier version used such a scheme for static function
      // name generation and we have to maintain backward compatibility.
      String *compat_name = Swig_name_member(0, proxy_class_name, lua_name);
      Setattr(n, "lua:name", compat_name);
      registerMethod(n, true, getNSpace());
      Delete(compat_name);
      Swig_restore(n);
    }

    current[STATIC_FUNC] = false;;
    return result;
  }

  /* ------------------------------------------------------------
   * memberconstantHandler()
   *
   * Create a C++ constant
   * ------------------------------------------------------------ */

  virtual int memberconstantHandler(Node *n) {
    REPORT("memberconstantHandler", n);
    int result = Language::memberconstantHandler(n);

    return result;
  }

  /* ---------------------------------------------------------------------
   * staticmembervariableHandler()
   * --------------------------------------------------------------------- */

  virtual int staticmembervariableHandler(Node *n) {
    REPORT("staticmembervariableHandler", n);
    current[STATIC_VAR] = true;
    //String *symname = Getattr(n, "sym:name");
    int result = Language::staticmembervariableHandler(n);
    if (!GetFlag(n, "wrappedasconstant")) {
      registerVariable(n);
    }

    if (result == SWIG_OK) {
      // This will add static member variable to the class namespace with name ClassName_VarName
      if (old_metatable_bindings && old_compatible_names) {
	Swig_save("lua_staticmembervariableHandler", n, "lua:name", NIL);
	String *lua_name = Getattr(n, "lua:name");
	// Although this function uses Swig_name_member, it actually generates the Lua name,
	// not the C++ name. This is because an earlier version used such a scheme for static function
	// name generation and we have to maintain backward compatibility.
	String *v2_name = Swig_name_member(NIL, proxy_class_name, lua_name);
	if (!GetFlag(n, "wrappedasconstant")) {
	  Setattr(n, "lua:name", v2_name);
          // Registering static var in the class parent nspace
	  registerVariable(n, true, getNSpace());
	}
	// If static member variable was wrapped as a constant, then
	// constant wrapper has already performed all actions necessary for old_metatable_bindings
	Delete(v2_name);
	Swig_restore(n);
      }
    }
    current[STATIC_VAR] = false;

    return result;
  }


  /* ---------------------------------------------------------------------
   * external runtime generation
   * --------------------------------------------------------------------- */

  /* This is to support the usage
     SWIG -external-runtime <filename>
     The code consists of two functions:
     String *runtimeCode()  // returns a large string with all the runtimes in
     String *defaultExternalRuntimeFilename() // returns the default filename
     I am writing a generic solution, even though SWIG-Lua only has one file right now...
   */
  String *runtimeCode() {
    String *s = NewString("");
    const char *filenames[] = { "luarun.swg", 0 };	// must be 0 terminated

    emitLuaFlavor(s);

    String *sfile = 0;
    for (int i = 0; filenames[i] != 0; i++) {
      sfile = Swig_include_sys(filenames[i]);
      if (!sfile) {
	Printf(stderr, "*** Unable to open '%s'\n", filenames[i]);
      } else {
	Append(s, sfile);
	Delete(sfile);
      }
    }

    return s;
  }

  String *defaultExternalRuntimeFilename() {
    return NewString("swigluarun.h");
  }

  /* ---------------------------------------------------------------------
   * helpers
   * --------------------------------------------------------------------- */

  void emitLuaFlavor(String *s) {
    if (elua_emulate) {
      Printf(s, "/*This is only emulation!*/\n");
      Printf(s, "#define SWIG_LUA_TARGET SWIG_LUA_FLAVOR_ELUA\n");
      Printf(s, "#define SWIG_LUA_ELUA_EMULATE\n");
    } else if (elua_ltr)
      Printf(s, "#define SWIG_LUA_TARGET SWIG_LUA_FLAVOR_ELUA\n");
    else if (eluac_ltr)
      Printf(s, "#define SWIG_LUA_TARGET SWIG_LUA_FLAVOR_ELUAC\n");
    else
      Printf(s, "#define SWIG_LUA_TARGET SWIG_LUA_FLAVOR_LUA\n");
  }


  /* -----------------------------------------------------------------------------
   * escapeCode()
   *
   * This is to convert the string of Lua code into a proper string, which can then be
   * emitted into the C/C++ code.
   * Basically is is a lot of search & replacing of odd sequences
   * ---------------------------------------------------------------------------- */

  void escapeCode(String *str) {
    //Printf(f_runtime,"/* original luacode:[[[\n%s\n]]]\n*/\n",str);
    Chop(str);			// trim
    Replace(str, "\\", "\\\\", DOH_REPLACE_ANY);	// \ to \\ (this must be done first)
    Replace(str, "\"", "\\\"", DOH_REPLACE_ANY);	// " to \"
    Replace(str, "\n", "\\n\"\n  \"", DOH_REPLACE_ANY);	// \n to \n"\n" (ie quoting every line)
    //Printf(f_runtime,"/* hacked luacode:[[[\n%s\n]]]\n*/\n",str);
  }

  /* -----------------------------------------------------------------------------
   * rawGetCArraysHash(String *name)
   *
   * A small helper to hide impelementation of how CArrays hashes are stored
   * ---------------------------------------------------------------------------- */

  Hash *rawGetCArraysHash(const_String_or_char_ptr name) {
    Hash *scope = symbolScopeLookup( name ? name : "" );
    if(!scope)
      return 0;

    Hash *carrays_hash = Getattr(scope, "lua:cdata");
    return carrays_hash;
  }
   
  /* -----------------------------------------------------------------------------
   * getCArraysHash()
   *
   * Each namespace can be described with a hash that stores C arrays
   * where members of the namespace should be added. All these hashes are stored
   * inside the symbols table, in pseudo-symbol for every namespace.
   * nspace could be NULL (NSPACE_TODO), that means functions and variables and classes
   * that are not in any namespace (this is default for SWIG unless %nspace feature is used).
   * You can later set some attributes that will affect behaviour of functions that use this hash:
   * "lua:no_namespaces" will disable "namespaces" array.
   * "lua:no_classes" will disable "classes" array.
   * For every component ("attributes", "methods", etc) there are subcomponents:
   *   XXX:name - name of the C array that stores data for component
   *   XXX:decl - statement with forward declaration of this array;
   * Namespace could be automatically registered to its parent if 'reg' == true. This can only be
   * done during the first call (a.k.a when nspace is created).
   * ---------------------------------------------------------------------------- */

  Hash *getCArraysHash(String *nspace, bool reg = true) {
    Hash *scope = symbolScopeLookup(nspace ? nspace : "");
    if(!scope) {
      symbolAddScope( nspace ? nspace : "" );
      scope = symbolScopeLookup(nspace ? nspace : "");
      assert(scope);
    }
    Hash *carrays_hash = Getattr(scope, "lua:cdata");
    if (carrays_hash != 0)
      return carrays_hash;
    carrays_hash = NewHash();
    String *mangled_name = 0;
    if (nspace == 0 || Len(nspace) == 0)
      mangled_name = NewString("SwigModule");
    else
      mangled_name = Swig_name_mangle(nspace);
    String *cname = NewStringf("swig_%s", mangled_name);

    Setattr(carrays_hash, "cname", cname);

    String *attr_tab = NewString("");
    String *attr_tab_name = NewStringf("swig_%s_attributes", mangled_name);
    String *attr_tab_decl = NewString("");
    Printv(attr_tab, "static swig_lua_attribute ", NIL);
    Printv(attr_tab, attr_tab_name, "[]", NIL);
    Printv(attr_tab_decl, attr_tab, ";\n", NIL);
    Printv(attr_tab, " = {\n", NIL);
    Setattr(carrays_hash, "attributes", attr_tab);
    Setattr(carrays_hash, "attributes:name", attr_tab_name);
    Setattr(carrays_hash, "attributes:decl", attr_tab_decl);

    String *methods_tab = NewString("");
    String *methods_tab_name = NewStringf("swig_%s_methods", mangled_name);
    String *methods_tab_decl = NewString("");
    if (elua_ltr || eluac_ltr)	// In this case methods array also acts as namespace rotable
      Printf(methods_tab, "const LUA_REG_TYPE ");
    else
      Printf(methods_tab, "static swig_lua_method ");
    Printv(methods_tab, methods_tab_name, "[]", NIL);
    Printv(methods_tab_decl, methods_tab, ";\n", NIL);
    Printv(methods_tab, "= {\n", NIL);
    Setattr(carrays_hash, "methods", methods_tab);
    Setattr(carrays_hash, "methods:name", methods_tab_name);
    Setattr(carrays_hash, "methods:decl", methods_tab_decl);

    String *const_tab = NewString("");
    String *const_tab_name = NewStringf("swig_%s_constants", mangled_name);
    String *const_tab_decl = NewString("");
    if (elua_ltr || eluac_ltr)	// In this case const array holds rotable with namespace constants
      Printf(const_tab, "const LUA_REG_TYPE ");
    else
      Printf(const_tab, "static swig_lua_const_info ");
    Printv(const_tab, const_tab_name, "[]", NIL);
    Printv(const_tab_decl, const_tab, ";", NIL);
    Printv(const_tab, "= {\n", NIL);
    Setattr(carrays_hash, "constants", const_tab);
    Setattr(carrays_hash, "constants:name", const_tab_name);
    Setattr(carrays_hash, "constants:decl", const_tab_decl);

    String *classes_tab = NewString("");
    String *classes_tab_name = NewStringf("swig_%s_classes", mangled_name);
    String *classes_tab_decl = NewString("");
    Printf(classes_tab, "static swig_lua_class* ");
    Printv(classes_tab, classes_tab_name, "[]", NIL);
    Printv(classes_tab_decl, classes_tab, ";", NIL);
    Printv(classes_tab, "= {\n", NIL);
    Setattr(carrays_hash, "classes", classes_tab);
    Setattr(carrays_hash, "classes:name", classes_tab_name);
    Setattr(carrays_hash, "classes:decl", classes_tab_decl);

    String *namespaces_tab = NewString("");
    String *namespaces_tab_name = NewStringf("swig_%s_namespaces", mangled_name);
    String *namespaces_tab_decl = NewString("");
    Printf(namespaces_tab, "static swig_lua_namespace* ");
    Printv(namespaces_tab, namespaces_tab_name, "[]", NIL);
    Printv(namespaces_tab_decl, namespaces_tab, ";", NIL);
    Printv(namespaces_tab, " = {\n", NIL);
    Setattr(carrays_hash, "namespaces", namespaces_tab);
    Setattr(carrays_hash, "namespaces:name", namespaces_tab_name);
    Setattr(carrays_hash, "namespaces:decl", namespaces_tab_decl);

    if (elua_ltr) {
      String *get_tab = NewString("");
      String *get_tab_name = NewStringf("swig_%s_get", mangled_name);
      String *get_tab_decl = NewString("");
      Printv(get_tab, "const LUA_REG_TYPE ", get_tab_name, "[]", NIL);
      Printv(get_tab_decl, get_tab, ";", NIL);
      Printv(get_tab, " = {\n", NIL);
      Setattr(carrays_hash, "get", get_tab);
      Setattr(carrays_hash, "get:name", get_tab_name);
      Setattr(carrays_hash, "get:decl", get_tab_decl);

      String *set_tab = NewString("");
      String *set_tab_name = NewStringf("swig_%s_set", mangled_name);
      String *set_tab_decl = NewString("");
      Printv(set_tab, "const LUA_REG_TYPE ", set_tab_name, "[]", NIL);
      Printv(set_tab_decl, set_tab, ";", NIL);
      Printv(set_tab, " = {\n", NIL);
      Setattr(carrays_hash, "set", set_tab);
      Setattr(carrays_hash, "set:name", set_tab_name);
      Setattr(carrays_hash, "set:decl", set_tab_decl);

    }
    if (!eluac_ltr) {
      String *metatable_tab = NewString("");
      String *metatable_tab_name = NewStringf("swig_%s_meta", mangled_name);
      String *metatable_tab_decl = NewString("");
      if (elua_ltr) // In this case const array holds rotable with namespace constants
	Printf(metatable_tab, "const LUA_REG_TYPE ");
      else
	Printf(metatable_tab, "static swig_lua_method ");
      Printv(metatable_tab, metatable_tab_name, "[]", NIL);
      Printv(metatable_tab_decl, metatable_tab, ";", NIL);
      Printv(metatable_tab, " = {\n", NIL);
      Setattr(carrays_hash, "metatable", metatable_tab);
      Setattr(carrays_hash, "metatable:name", metatable_tab_name);
      Setattr(carrays_hash, "metatable:decl", metatable_tab_decl);
    }

    Setattr(scope, "lua:cdata", carrays_hash);
    assert(rawGetCArraysHash(nspace));

    if (reg && nspace != 0 && Len(nspace) != 0 && GetFlag(carrays_hash, "lua:no_reg") == 0) {
      // Split names into components
      List *components = Split(nspace, '.', -1);
      String *parent_path = NewString("");
      int len = Len(components);
      String *name = Copy(Getitem(components, len - 1));
      for (int i = 0; i < len - 1; i++) {
	if (i > 0)
	  Printv(parent_path, NSPACE_SEPARATOR, NIL);
	String *item = Getitem(components, i);
	Printv(parent_path, item, NIL);
      }
      Hash *parent = getCArraysHash(parent_path, true);
      String *namespaces_tab = Getattr(parent, "namespaces");
      Printv(namespaces_tab, "&", cname, ",\n", NIL);
      if (elua_ltr || eluac_ltr) {
	String *methods_tab = Getattr(parent, "methods");
	Printv(methods_tab, tab4, "{LSTRKEY(\"", name, "\")", ", LROVAL(", methods_tab_name, ")", "},\n", NIL);
      }
      Setattr(carrays_hash, "name", name);

      Delete(components);
      Delete(parent_path);
    } else if (!reg)		// This namespace shouldn't be registered. Lets remember it.
      SetFlag(carrays_hash, "lua:no_reg");

    Delete(mangled_name);
    mangled_name = 0;
    return carrays_hash;
  }

  /* -----------------------------------------------------------------------------
   * closeCArraysHash()
   *
   * Functions add end markers {0,0,...,0} to all arrays, prints them to
   * output and marks hash as closed (lua:closed). Consequent attempts to
   * close the same hash will result in an error.
   * closeCArraysHash DOES NOT print structure that describes namespace, it only
   * prints array. You can use printCArraysDefinition to print structure.
   * if "lua:no_namespaces" is set, then array for "namespaces" won't be printed
   * if "lua:no_classes" is set, then array for "classes" won't be printed
   * ----------------------------------------------------------------------------- */

  void closeCArraysHash(String *nspace, File *output) {
    Hash *carrays_hash = rawGetCArraysHash(nspace);
    assert(carrays_hash);
    assert(GetFlag(carrays_hash, "lua:closed") == 0);

    SetFlag(carrays_hash, "lua:closed");

    // Do arrays describe class instance part or class static part
    const int is_instance = GetFlag(carrays_hash, "lua:class_instance");


    String *attr_tab = Getattr(carrays_hash, "attributes");
    Printf(attr_tab, "    {0,0,0}\n};\n");
    Printv(output, attr_tab, NIL);

    String *const_tab = Getattr(carrays_hash, "constants");
    String *const_tab_name = Getattr(carrays_hash, "constants:name");
    if (elua_ltr || eluac_ltr)
      Printv(const_tab, tab4, "{LNILKEY, LNILVAL}\n", "};\n", NIL);
    else
      Printf(const_tab, "    {0,0,0,0,0,0}\n};\n");
    
    // For the sake of compiling with -Wall -Werror we print constants
    // only when necessary
    int need_constants = 0;
    if ( (elua_ltr || eluac_ltr) && (old_metatable_bindings) )
      need_constants = 1;
    else if (!is_instance) // static part need constants tab
      need_constants = 1;

    if (need_constants)
      Printv(output, const_tab, NIL);

    if (elua_ltr) {
      // Put forward declaration of metatable array
      Printv(output, "extern ", Getattr(carrays_hash, "metatable:decl"), "\n", NIL);
    }
    String *methods_tab = Getattr(carrays_hash, "methods");
    String *metatable_tab_name = Getattr(carrays_hash, "metatable:name");
    if (elua_ltr || eluac_ltr) {
      if (old_metatable_bindings)
	Printv(methods_tab, tab4, "{LSTRKEY(\"const\"), LROVAL(", const_tab_name, ")},\n", NIL);
      if (elua_ltr) {
	Printv(methods_tab, tab4, "{LSTRKEY(\"__metatable\"), LROVAL(", metatable_tab_name, ")},\n", NIL);
      }

      Printv(methods_tab, tab4, "{LSTRKEY(\"__disown\"), LFUNCVAL(SWIG_Lua_class_disown)},\n", NIL);
      Printv(methods_tab, tab4, "{LNILKEY, LNILVAL}\n};\n", NIL);
    } else
      Printf(methods_tab, "    {0,0}\n};\n");
    Printv(output, methods_tab, NIL);

    if (!GetFlag(carrays_hash, "lua:no_classes")) {
      String *classes_tab = Getattr(carrays_hash, "classes");
      Printf(classes_tab, "    0\n};\n");
      Printv(output, classes_tab, NIL);
    }

    if (!GetFlag(carrays_hash, "lua:no_namespaces")) {
      String *namespaces_tab = Getattr(carrays_hash, "namespaces");
      Printf(namespaces_tab, "    0\n};\n");
      Printv(output, namespaces_tab, NIL);
    }
    if (elua_ltr) {
      String *get_tab = Getattr(carrays_hash, "get");
      String *set_tab = Getattr(carrays_hash, "set");
      Printv(get_tab, tab4, "{LNILKEY, LNILVAL}\n};\n", NIL);
      Printv(set_tab, tab4, "{LNILKEY, LNILVAL}\n};\n", NIL);
      Printv(output, get_tab, NIL);
      Printv(output, set_tab, NIL);
    }

    // Heuristic whether we need to print metatable or not.
    // For the sake of compiling with -Wall -Werror we don't print
    // metatable for static part.
    int need_metatable = 0;
    if (eluac_ltr)
      need_metatable = 0;
    else if(!is_instance)
      need_metatable = 0;
    else
      need_metatable = 1;

    if (need_metatable) {
      String *metatable_tab = Getattr(carrays_hash, "metatable");
      assert(metatable_tab);
      if (elua_ltr) {
	String *get_tab_name = Getattr(carrays_hash, "get:name");
	String *set_tab_name = Getattr(carrays_hash, "set:name");

	if (GetFlag(carrays_hash, "lua:class_instance")) {
	  Printv(metatable_tab, tab4, "{LSTRKEY(\"__index\"), LFUNCVAL(SWIG_Lua_class_get)},\n", NIL);
	  Printv(metatable_tab, tab4, "{LSTRKEY(\"__newindex\"), LFUNCVAL(SWIG_Lua_class_set)},\n", NIL);
	} else {
	  Printv(metatable_tab, tab4, "{LSTRKEY(\"__index\"), LFUNCVAL(SWIG_Lua_namespace_get)},\n", NIL);
	  Printv(metatable_tab, tab4, "{LSTRKEY(\"__newindex\"), LFUNCVAL(SWIG_Lua_namespace_set)},\n", NIL);
	}

	Printv(metatable_tab, tab4, "{LSTRKEY(\"__gc\"), LFUNCVAL(SWIG_Lua_class_destruct)},\n", NIL);
	Printv(metatable_tab, tab4, "{LSTRKEY(\".get\"), LROVAL(", get_tab_name, ")},\n", NIL);
	Printv(metatable_tab, tab4, "{LSTRKEY(\".set\"), LROVAL(", set_tab_name, ")},\n", NIL);
	Printv(metatable_tab, tab4, "{LSTRKEY(\".fn\"), LROVAL(", Getattr(carrays_hash, "methods:name"), ")},\n", NIL);

	if (GetFlag(carrays_hash, "lua:class_instance")) {
	  String *static_cls = Getattr(carrays_hash, "lua:class_instance:static_hash");
	  assert(static_cls);
	  // static_cls is swig_lua_namespace. This structure can't be use with eLua(LTR)
	  // Instead structure describing its methods isused
	  String *static_cls_cname = Getattr(static_cls, "methods:name");
	  assert(static_cls_cname);
	  Printv(metatable_tab, tab4, "{LSTRKEY(\".static\"), LROVAL(", static_cls_cname, ")},\n", NIL);
	  // Put forward declaration of this array
	  Printv(output, "extern ", Getattr(static_cls, "methods:decl"), "\n", NIL);
	} else if (GetFlag(carrays_hash, "lua:class_static")) {
	  Hash *instance_cls = Getattr(carrays_hash, "lua:class_static:instance_hash");
	  assert(instance_cls);
	  String *instance_cls_metatable_name = Getattr(instance_cls, "metatable:name");
	  assert(instance_cls_metatable_name);
	  Printv(metatable_tab, tab4, "{LSTRKEY(\".instance\"), LROVAL(", instance_cls_metatable_name, ")},\n", NIL);
	}

	Printv(metatable_tab, tab4, "{LNILKEY, LNILVAL}\n};\n", NIL);
      } else {
	Printf(metatable_tab, "    {0,0}\n};\n");
      }

      Printv(output, metatable_tab, NIL);
    }

    Printv(output, "\n", NIL);
  }

  /* -----------------------------------------------------------------------------
   * closeNamespaces()
   *
   * Recursively close all non-closed namespaces. Prints data to dataOutput.
   * ----------------------------------------------------------------------------- */

  void closeNamespaces(File *dataOutput) {
    // Special handling for empty module.
    if (symbolScopeLookup("") == 0 || rawGetCArraysHash("") == 0) {
      // Module is empty. Create hash for global scope in order to have swig_SwigModule
      // variable in resulting file
      getCArraysHash(0);
    }
    // Because we cant access directly 'symtabs', instead we access
    // top-level scope and look on all scope pseudo-symbols in it.
    Hash *top_scope = symbolScopeLookup("");
    assert(top_scope);
    Iterator ki = First(top_scope);
    List *to_close = NewList();
    while (ki.key) {
      assert(ki.item);
      if (Getattr(ki.item, "sym:scope")) {
        // We have a pseudo symbol. Lets get actual scope for this pseudo symbol
        Hash *carrays_hash = rawGetCArraysHash(ki.key);
        assert(carrays_hash);
        if (GetFlag(carrays_hash, "lua:closed") == 0)
          Append(to_close, ki.key);
      }
      ki = Next(ki);
    }
    SortList(to_close, &compareByLen);
    int len = Len(to_close);
    for (int i = 0; i < len; i++) {
      String *key = Getitem(to_close, i);
      closeCArraysHash(key, dataOutput);
      Hash *carrays_hash = rawGetCArraysHash(key);
      String *name = 0;		// name - name of the namespace as it should be visible in Lua
      if (DohLen(key) == 0)	// This is global module
	name = module;
      else
	name = Getattr(carrays_hash, "name");
      assert(name);
      printCArraysDefinition(key, name, dataOutput);
    }
    Delete(to_close);
  }

  /* -----------------------------------------------------------------------------
   * printCArraysDefinition()
   *
   * This function prints to output a definition of namespace in form
   *   swig_lua_namespace $cname =  { attr_array, methods_array, ... , namespaces_array };
   * You can call this function as many times as is necessary.
   * 'name' is a user-visible name that this namespace will have in Lua. It shouldn't
   * be a fully qualified name, just its own name.
   * ----------------------------------------------------------------------------- */

  void printCArraysDefinition(String *nspace, String *name, File *output) {
    Hash *carrays_hash = getCArraysHash(nspace, false);

    String *cname = Getattr(carrays_hash, "cname");	// cname - name of the C structure that describes namespace
    assert(cname);
    Printv(output, "static swig_lua_namespace ", cname, " = ", NIL);

    String *null_string = NewString("0");
    String *attr_tab_name = Getattr(carrays_hash, "attributes:name");
    String *methods_tab_name = Getattr(carrays_hash, "methods:name");
    String *const_tab_name = Getattr(carrays_hash, "constants:name");
    String *classes_tab_name = Getattr(carrays_hash, "classes:name");
    String *namespaces_tab_name = Getattr(carrays_hash, "namespaces:name");
    bool has_classes = GetFlag(carrays_hash, "lua:no_classes") == 0;
    bool has_namespaces = GetFlag(carrays_hash, "lua:no_namespaces") == 0;

    Printv(output, "{\n",
	   tab4, "\"", name, "\",\n",
	   tab4, methods_tab_name, ",\n",
	   tab4, attr_tab_name, ",\n",
	   tab4, const_tab_name, ",\n",
	   tab4, (has_classes) ? classes_tab_name : null_string, ",\n",
	   tab4, (has_namespaces) ? namespaces_tab_name : null_string, "\n};\n", NIL);
    Delete(null_string);
  }

  /* -----------------------------------------------------------------------------
   * luaCurrentSymbolNSpace()
   *
   * This function determines actual namespace/scope where any symbol at the
   * current moment should be placed. It looks at the 'current' array
   * and depending on where are we - static class member/function,
   * instance class member/function or just global functions decides
   * where symbol should be put.
   * The namespace/scope doesn't depend from symbol, only from 'current'
   * ----------------------------------------------------------------------------- */

  String *luaCurrentSymbolNSpace() {
    String *scope = 0;
    // If ouside class, than NSpace is used.
    // If inside class, but current[NO_CPP], then this is friend function. It belongs to NSpace
    if (!getCurrentClass() || current[NO_CPP]) {
      scope = getNSpace();
    } else if (current[ENUM_CONST] && !CPlusPlus ) {
        // Enums in C mode go to NSpace
        scope = getNSpace();
    } else {
      // If inside class, then either class static namespace or class fully qualified name is used
      assert(!current[NO_CPP]);
      if (current[STATIC_FUNC] || current[STATIC_VAR] || current[STATIC_CONST]) {
	scope = class_static_nspace;
      } else if (current[MEMBER_VAR] || current[CONSTRUCTOR] || current[DESTRUCTOR]
		 || current[MEMBER_FUNC]) {
	scope = full_proxy_class_name;
      } else {			// Friend functions are handled this way
	scope = class_static_nspace;
      }
      assert(scope);
    }
    return scope;
  }

  /* -----------------------------------------------------------------------------
   * luaAddSymbol()
   *
   * Our implementation of addSymbol. Determines scope correctly, then 
   * forwards to Language::addSymbol
   * ----------------------------------------------------------------------------- */

  int luaAddSymbol(const String *s, const Node *n) {
    String *scope = luaCurrentSymbolNSpace();
    return luaAddSymbol(s, n, scope);
  }

  /* -----------------------------------------------------------------------------
   * luaAddSymbol()
   *
   * Overload. Enforces given scope. Actually, it simply forwards call to Language::addSymbol
   * ----------------------------------------------------------------------------- */

  int luaAddSymbol(const String *s, const Node *n, const_String_or_char_ptr scope) {
    int result = Language::addSymbol(s, n, scope);
    if (!result)
      Printf(stderr, "addSymbol(%s to scope %s) failed\n", s, scope);
    return result;
  }

};

/* NEW LANGUAGE NOTE:***********************************************
 in order to add you language into swig, you need to make the following changes:
 - write this file (obviously)
 - add into the makefile (not 100% clear on how to do this)
 - edit swigmain.cxx to add your module
 
near the top of swigmain.cxx, look for this code & add you own codes
======= begin change ==========
extern "C" {
  Language *swig_tcl(void);
  Language *swig_python(void);
  //etc,etc,etc...
  Language *swig_lua(void);	// this is my code
}
 
  //etc,etc,etc...
 
swig_module  modules[] = {
  {"-guile",     swig_guile,     "Guile"},
  {"-java",      swig_java,      "Java"},
  //etc,etc,etc...
  {"-lua",       swig_lua,       "Lua"},	// this is my code
  {NULL, NULL, NULL}	// this must come at the end of the list
};
======= end change ==========
 
This is all that is needed
 
NEW LANGUAGE NOTE:END ************************************************/

/* -----------------------------------------------------------------------------
 * swig_lua()    - Instantiate module
 * ----------------------------------------------------------------------------- */

extern "C" Language *swig_lua(void) {
  return new LUA();
}
