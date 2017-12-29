/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * r.cxx
 *
 * R language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

static const double DEFAULT_NUMBER = .0000123456712312312323;

static String* replaceInitialDash(const String *name)
{
  String *retval;
  if (!Strncmp(name, "_", 1)) {
    retval = Copy(name);
    Insert(retval, 0, "s");
  } else {
    retval = Copy(name);
  }
  return retval;
}

static String * getRTypeName(SwigType *t, int *outCount = NULL) {
  String *b = SwigType_base(t);
  List *els = SwigType_split(t);
  int count = 0;
  int i;
  
  if(Strncmp(b, "struct ", 7) == 0) 
    Replace(b, "struct ", "", DOH_REPLACE_FIRST);
  
  /* Printf(stdout, "<getRTypeName> %s,base = %s\n", t, b);
     for(i = 0; i < Len(els); i++) 
     Printf(stdout, "%d) %s, ", i, Getitem(els,i));
     Printf(stdout, "\n"); */
  
  for(i = 0; i < Len(els); i++) {
    String *el = Getitem(els, i);
    if(Strcmp(el, "p.") == 0 || Strncmp(el, "a(", 2) == 0) {
      count++;
      Append(b, "Ref");
    }
  }
  if(outCount)
    *outCount = count;
  
  String *tmp = NewString("");
  char *retName = Char(SwigType_manglestr(t));
  Insert(tmp, 0, retName);
  return tmp;
  
  /*
  if(count)
    return(b);
  
  Delete(b);
  return(NewString(""));
  */
}

/*********************
 Tries to get the name of the R class corresponding  to the given type
  e.g. struct A * is ARef,  struct A**  is  ARefRef.
  Now handles arrays, i.e. struct A[2]
****************/

static String *getRClassName(String *retType, int /*addRef*/ = 1, int upRef=0) {
  String *tmp = NewString("");
  SwigType *resolved = SwigType_typedef_resolve_all(retType);
  char *retName = Char(SwigType_manglestr(resolved));
  if (upRef) {
    Printf(tmp, "_p%s", retName);
  } else{
    Insert(tmp, 0, retName);
  }
  
  return tmp;
/*
#if 1
  List *l = SwigType_split(retType);
  int n = Len(l);
  if(!l || n == 0) {
#ifdef R_SWIG_VERBOSE
    if (debugMode)
      Printf(stdout, "SwigType_split return an empty list for %s\n", 
	     retType);
#endif
    return(tmp);
  }
  
  
  String *el = Getitem(l, n-1);
  char *ptr = Char(el);
  if(strncmp(ptr, "struct ", 7) == 0)
    ptr += 7;
  
  Printf(tmp, "%s", ptr);
  
  if(addRef) {
    for(int i = 0; i < n; i++) {
      if(Strcmp(Getitem(l, i), "p.") == 0 || 
	 Strncmp(Getitem(l, i), "a(", 2) == 0)
	Printf(tmp, "Ref");
    }
  }
  
#else
  char *retName = Char(SwigType_manglestr(retType));
  if(!retName)
    return(tmp);
  
  if(addRef) {
    while(retName && strlen(retName) > 1 && strncmp(retName, "_p", 2) == 0)  {
      retName += 2;
      Printf(tmp, "Ref");
    }
  }
  if(retName[0] == '_')
    retName ++;
  Insert(tmp, 0, retName);
#endif
  
  return tmp;
*/
}

/*********************
 Tries to get the name of the R class corresponding  to the given type
  e.g. struct A * is ARef,  struct A**  is  ARefRef.
  Now handles arrays, i.e. struct A[2]
****************/

static String * getRClassNameCopyStruct(String *retType, int addRef) {
  String *tmp = NewString("");
  
#if 1
  List *l = SwigType_split(retType);
  int n = Len(l);
  if(!l || n == 0) {
#ifdef R_SWIG_VERBOSE
    Printf(stdout, "SwigType_split return an empty list for %s\n", retType);
#endif
    return(tmp);
  }
  
  
  String *el = Getitem(l, n-1);
  char *ptr = Char(el);
  if(strncmp(ptr, "struct ", 7) == 0)
    ptr += 7;
  
  Printf(tmp, "%s", ptr);
  
  if(addRef) {
    for(int i = 0; i < n; i++) {
      if(Strcmp(Getitem(l, i), "p.") == 0 || 
	 Strncmp(Getitem(l, i), "a(", 2) == 0)
	Printf(tmp, "Ref");
    }
  }
  
#else
  char *retName = Char(SwigType_manglestr(retType));
  if(!retName)
    return(tmp);
  
  if(addRef) {
    while(retName && strlen(retName) > 1 && 
	  strncmp(retName, "_p", 2) == 0)  {
      retName += 2;
      Printf(tmp, "Ref");
    }
  }
  
  if(retName[0] == '_')
    retName ++;
  Insert(tmp, 0, retName);
#endif

  return tmp;
}


/*********************************
  Write the elements of a list to the File*, one element per line.
  If quote  is true, surround the element with "element".
  This takes care of inserting a tab in front of each line and also
  a comma after each element, except the last one.
**********************************/

static void writeListByLine(List *l, File *out, bool quote = 0) {
  int i, n = Len(l);
  for(i = 0; i < n; i++) 
    Printf(out, "%s%s%s%s%s\n", tab8, 
	   quote ? "\"" :"",  
	   Getitem(l, i), 
	   quote ? "\"" :"", i < n-1 ? "," : "");
}


static const char *usage = "\
R Options (available with -r)\n\
     -copystruct      - Emit R code to copy C structs (on by default)\n\
     -cppcast         - Enable C++ casting operators (default) \n\
     -debug           - Output debug\n\
     -dll <name>      - Name of the DLL (without the .dll or .so suffix).\n\
                        Default is the module name.\n\
     -gc              - Aggressive garbage collection\n\
     -memoryprof      - Add memory profile\n\
     -namespace       - Output NAMESPACE file\n\
     -no-init-code    - Turn off the generation of the R_init_<pkgname> code\n\
                        (registration information still generated)\n\
     -package <name>  - Package name for the PACKAGE argument of the R .Call()\n\
                        invocations. Default is the module name.\n\
";



/************
 Display the help for this module on the screen/console.
*************/
static void showUsage() {
  fputs(usage, stdout);
}

static bool expandTypedef(SwigType *t) {
  if (SwigType_isenum(t)) return false;
  String *prefix = SwigType_prefix(t);
  if (Strncmp(prefix, "f", 1)) return false;
  if (Strncmp(prefix, "p.f", 3)) return false;
  return true;
}


/*****
      Determine whether  we should add a .copy argument to the S function
      that wraps/interfaces to the routine that returns the given type.
*****/
static int addCopyParameter(SwigType *type) {
  int ok = 0;
  ok = Strncmp(type, "struct ", 7) == 0 || Strncmp(type, "p.struct ", 9) == 0;
  if(!ok) {
    ok = Strncmp(type, "p.", 2);
  }

  return(ok);
}

static void replaceRClass(String *tm, SwigType *type) {
  String *tmp = getRClassName(type);
  String *tmp_base = getRClassName(type, 0);
  String *tmp_ref = getRClassName(type, 1, 1);
  Replaceall(tm, "$R_class", tmp);
  Replaceall(tm, "$*R_class", tmp_base);
  Replaceall(tm, "$&R_class", tmp_ref);
  Delete(tmp); Delete(tmp_base); Delete(tmp_ref);
}

static double getNumber(String *value) {
  double d = DEFAULT_NUMBER;
  if(Char(value)) {
    if(sscanf(Char(value), "%lf", &d) != 1)
      return(DEFAULT_NUMBER);
  }
  return(d);
}

class R : public Language {
public:
  R();
  void registerClass(Node *n);
  void main(int argc, char *argv[]);
  int top(Node *n);
  
  void dispatchFunction(Node *n);
  int functionWrapper(Node *n);
  int constantWrapper(Node *n);
  int variableWrapper(Node *n);

  int classDeclaration(Node *n);
  int enumDeclaration(Node *n);

  int membervariableHandler(Node *n);

  int typedefHandler(Node *n);
  static List *Swig_overload_rank(Node *n,
			   bool script_lang_wrapping);

  int memberfunctionHandler(Node *n) {
    if (debugMode)
      Printf(stdout, "<memberfunctionHandler> %s %s\n", 
	     Getattr(n, "name"),
	     Getattr(n, "type"));
    member_name = Getattr(n, "sym:name");
    processing_class_member_function = 1;
    int status = Language::memberfunctionHandler(n);    
    processing_class_member_function = 0;
    return status;
  }

  /* Grab the name of the current class being processed so that we can 
     deal with members of that class. */
  int classHandler(Node *n){
    if(!ClassMemberTable) 
      ClassMemberTable = NewHash();
    
    class_name = Getattr(n, "name");
    int status = Language::classHandler(n);
    
    class_name = NULL;
    return status;
  }

  // Not used:
  String *runtimeCode();
  
protected:
  int addRegistrationRoutine(String *rname, int nargs);
  int outputRegistrationRoutines(File *out);
  
  int outputCommandLineArguments(File *out);
  int generateCopyRoutines(Node *n); 
  int DumpCode(Node *n);
  
  int OutputMemberReferenceMethod(String *className, int isSet, List *el, File *out);
  int OutputArrayMethod(String *className, List *el, File *out);
  int OutputClassMemberTable(Hash *tb, File *out);
  int OutputClassMethodsTable(File *out);
  int OutputClassAccessInfo(Hash *tb, File *out);
  
  int defineArrayAccessors(SwigType *type);
  
  void addNamespaceFunction(String *name) {
    if(!namespaceFunctions)
      namespaceFunctions = NewList();
    Append(namespaceFunctions, name);
  }

  void addNamespaceMethod(String *name) {
    if(!namespaceMethods)
      namespaceMethods = NewList();
    Append(namespaceMethods, name);
  }
  
  String* processType(SwigType *t, Node *n, int *nargs = NULL);
  String *createFunctionPointerHandler(SwigType *t, Node *n, int *nargs);
  int addFunctionPointerProxy(String *name, Node *n, SwigType *t, String *s_paramTypes) {
    /*XXX Do we need to put the t in there to get the return type later. */
    if(!functionPointerProxyTable) 
      functionPointerProxyTable = NewHash();
    
    Setattr(functionPointerProxyTable, name, n);
    
    Setattr(SClassDefs, name, name);
    Printv(s_classes, "setClass('", 
	   name,
	   "',\n", tab8, 
	   "prototype = list(parameterTypes = c(", s_paramTypes, "),\n",
	   tab8, tab8, tab8,
	   "returnType = '", SwigType_manglestr(t), "'),\n", tab8, 
	   "contains = 'CRoutinePointer')\n\n##\n", NIL);
    
    return SWIG_OK;
  }
  

  void addSMethodInfo(String *name, 
		      String *argType, int nargs);
  // Simple initialization such as constant strings that can be reused. 
  void init(); 
  
  
  void addAccessor(String *memberName, Wrapper *f, 
		   String *name, int isSet = -1);
  
  static int getFunctionPointerNumArgs(Node *n, SwigType *tt);

protected: 
  bool copyStruct;
  bool memoryProfile;
  bool aggressiveGc;

  // Strings into which we cumulate the generated code that is to be written
  //vto the files.
  String *sfile;
  String *f_init;
  String *s_classes;
  String *f_begin;
  String *f_runtime;
  String *f_wrapper;
  String *s_header;
  String *f_wrappers;
  String *s_init;
  String *s_init_routine;
  String *s_namespace;
  
  // State variables that carry information across calls to functionWrapper() 
  // from  member accessors and class declarations. 
  String *opaqueClassDeclaration;
  int processing_variable;
  int processing_member_access_function;
  String *member_name;
  String *class_name;
  
  
  int processing_class_member_function;
  List *class_member_functions;
  List *class_member_set_functions;
  
  /* */
  Hash *ClassMemberTable;
  Hash *ClassMethodsTable;
  Hash *SClassDefs;
  Hash *SMethodInfo;
  
  // Information about routines that are generated and to be registered with 
  // R for dynamic lookup. 
  Hash *registrationTable;
  Hash *functionPointerProxyTable;
  
  List *namespaceFunctions;
  List *namespaceMethods;
  List *namespaceClasses; // Probably can do this from ClassMemberTable.
  
  
  // Store a copy of the command line. 
  // Need only keep a string that has it formatted. 
  char **Argv;
  int    Argc;
  bool inCPlusMode;
  
  // State variables that we remember from the command line settings
  // potentially that govern the code we generate.
  String *DllName;
  String *Rpackage;
  bool    noInitializationCode;
  bool    outputNamespaceInfo;
  
  String *UnProtectWrapupCode;

  // Static members
  static bool debugMode;
};

R::R() :
  copyStruct(false),
  memoryProfile(false),
  aggressiveGc(false),
  sfile(0),
  f_init(0),
  s_classes(0),
  f_begin(0),
  f_runtime(0),
  f_wrapper(0),
  s_header(0),
  f_wrappers(0),
  s_init(0),
  s_init_routine(0),
  s_namespace(0),
  opaqueClassDeclaration(0),
  processing_variable(0),
  processing_member_access_function(0),
  member_name(0),
  class_name(0),
  processing_class_member_function(0),
  class_member_functions(0),
  class_member_set_functions(0),
  ClassMemberTable(0),
  ClassMethodsTable(0),
  SClassDefs(0),
  SMethodInfo(0),
  registrationTable(0),
  functionPointerProxyTable(0),
  namespaceFunctions(0),
  namespaceMethods(0),
  namespaceClasses(0),
  Argv(0),
  Argc(0),
  inCPlusMode(false),
  DllName(0),
  Rpackage(0),
  noInitializationCode(false),
  outputNamespaceInfo(false),
  UnProtectWrapupCode(0) {
}

bool R::debugMode = false;

int R::getFunctionPointerNumArgs(Node *n, SwigType *tt) {
  (void) tt;
  n = Getattr(n, "type");
  if (debugMode)
    Printf(stdout, "type: %s\n", n);

  ParmList *parms = Getattr(n, "parms");
  if (debugMode)
    Printf(stdout, "parms = %p\n", parms);
  return ParmList_len(parms);
}


void R::addSMethodInfo(String *name, String *argType, int nargs) {
  (void) argType;
  
  if(!SMethodInfo)
    SMethodInfo = NewHash();
  if (debugMode)
    Printf(stdout, "[addMethodInfo] %s\n", name);

  Hash *tb = Getattr(SMethodInfo, name);

  if(!tb) {
    tb = NewHash();
    Setattr(SMethodInfo, name, tb);
  }

  String *str = Getattr(tb, "max");
  int max = -1;
  if(str)
    max = atoi(Char(str));
  if(max < nargs) {
    if(str)  Delete(str);
    str = NewStringf("%d", max);
    Setattr(tb, "max", str);
  }
}
 
/*
Returns the name of the new routine.
*/
String * R::createFunctionPointerHandler(SwigType *t, Node *n, int *numArgs) {
  String *funName = SwigType_manglestr(t);
  
  /* See if we have already processed this one. */
  if(functionPointerProxyTable && Getattr(functionPointerProxyTable, funName))
    return funName;
  
  if (debugMode)
    Printf(stdout, "<createFunctionPointerHandler> Defining %s\n",  t);
  
  SwigType *rettype = Copy(Getattr(n, "type"));
  SwigType *funcparams = SwigType_functionpointer_decompose(rettype);
  String *rtype = SwigType_str(rettype, 0);

  //   ParmList *parms = Getattr(n, "parms");
  // memory leak
  ParmList *parms = SwigType_function_parms(SwigType_del_pointer(Copy(t)), n);


  if (debugMode) {
    Printf(stdout, "Type: %s\n", t);
    Printf(stdout, "Return type: %s\n", SwigType_base(t));
  }
  
  bool isVoidType = Strcmp(rettype, "void") == 0;
  if (debugMode)
    Printf(stdout, "%s is void ? %s  (%s)\n", funName, isVoidType ? "yes" : "no", rettype);
  
  Wrapper *f = NewWrapper();
  
  /* Go through argument list, attach lnames for arguments */
  int i = 0;
  Parm *p = parms;
  for (i = 0; p; p = nextSibling(p), ++i) {
    String *arg = Getattr(p, "name");
    String *lname;

    if (!arg && Cmp(Getattr(p, "type"), "void")) {
      lname = NewStringf("s_arg%d", i+1);
      Setattr(p, "name", lname);
    } else
      lname = arg;

    Setattr(p, "lname", lname);
  }
  
  Swig_typemap_attach_parms("out", parms, f);
  Swig_typemap_attach_parms("scoerceout", parms, f);
  Swig_typemap_attach_parms("scheck", parms, f);

  Printf(f->def, "%s %s(", rtype, funName);

  emit_parameter_variables(parms, f);
  emit_return_variable(n, rettype, f);
//  emit_attach_parmmaps(parms,f);

  /*  Using weird name and struct to avoid potential conflicts. */
  Wrapper_add_local(f, "r_swig_cb_data", "RCallbackFunctionData *r_swig_cb_data = R_SWIG_getCallbackFunctionData()");
  String *lvar = NewString("r_swig_cb_data");

  Wrapper_add_local(f, "r_tmp", "SEXP r_tmp"); // for use in converting arguments to R objects for call.
  Wrapper_add_local(f, "r_nprotect", "int r_nprotect = 0"); // for use in converting arguments to R objects for call.
  Wrapper_add_local(f, "r_vmax", "char * r_vmax= 0"); // for use in converting arguments to R objects for call.

  // Add local for error code in return value.  This is not in emit_return_variable because that assumes an out typemap
  // whereas the type makes are reverse
  Wrapper_add_local(f, "ecode", "int ecode = 0");

  p = parms;
  int nargs = ParmList_len(parms);
  if(numArgs) {
    *numArgs = nargs;
    if (debugMode)
      Printf(stdout, "Setting number of parameters to %d\n", *numArgs);
  } 
  String *setExprElements = NewString("");
  
  String *s_paramTypes = NewString("");
  for(i = 0; p; i++) {
    SwigType *tt = Getattr(p, "type");
    SwigType *name = Getattr(p, "name");
    String *tm = Getattr(p, "tmap:out");
    Printf(f->def,  "%s %s", SwigType_str(tt, 0), name);
     if(tm) {
      Replaceall(tm, "$1", name);
      if (SwigType_isreference(tt)) {
	String *tmp = NewString("");
        Append(tmp, "*");
	Append(tmp, name);
	Replaceall(tm, tmp, name);
      }
      Replaceall(tm, "$result", "r_tmp");
      replaceRClass(tm, Getattr(p,"type"));
      Replaceall(tm,"$owner", "R_SWIG_EXTERNAL");
    } 
    
    Printf(setExprElements, "%s\n", tm);
    Printf(setExprElements, "SETCAR(r_swig_cb_data->el, %s);\n", "r_tmp");
    Printf(setExprElements, "r_swig_cb_data->el = CDR(r_swig_cb_data->el);\n\n");
    
    Printf(s_paramTypes, "'%s'", SwigType_manglestr(tt));
    
    
    p = nextSibling(p);
    if(p) {
      Printf(f->def, ", ");
      Printf(s_paramTypes, ", ");
    }
  }
  
  Printf(f->def,  ") {\n");
  
  Printf(f->code, "Rf_protect(%s->expr = Rf_allocVector(LANGSXP, %d));\n", lvar, nargs + 1);
  Printf(f->code, "r_nprotect++;\n");
  Printf(f->code, "r_swig_cb_data->el = r_swig_cb_data->expr;\n\n");
  
  Printf(f->code, "SETCAR(r_swig_cb_data->el, r_swig_cb_data->fun);\n");
  Printf(f->code, "r_swig_cb_data->el = CDR(r_swig_cb_data->el);\n\n");
  
  Printf(f->code, "%s\n\n", setExprElements);
  
  Printv(f->code, "r_swig_cb_data->retValue = R_tryEval(", 
	 "r_swig_cb_data->expr,",
	 " R_GlobalEnv,",
	 " &r_swig_cb_data->errorOccurred",
	 ");\n", 
	 NIL);
  
  Printv(f->code, "\n",
	 "if(r_swig_cb_data->errorOccurred) {\n",
	 "R_SWIG_popCallbackFunctionData(1);\n",
	 "Rf_error(\"error in calling R function as a function pointer (",
	 funName,
	 ")\");\n",
	 "}\n",
	 NIL);
   
   
   
  if(!isVoidType) {
    /* Need to deal with the return type of the function pointer, not the function pointer itself. 
       So build a new node that has the relevant pieces.
       XXX  Have to be a little more clever so that we can deal with struct A * - the * is getting lost.
       Is this still true? If so, will a SwigType_push() solve things?
    */
    Parm *bbase = NewParmNode(rettype, n);
    String *returnTM = Swig_typemap_lookup("in", bbase, Swig_cresult_name(), f);
    if(returnTM) {
      String *tm = returnTM;
      Replaceall(tm,"$input", "r_swig_cb_data->retValue");
      Replaceall(tm,"$target", Swig_cresult_name());
      replaceRClass(tm, rettype);
      Replaceall(tm,"$owner", "R_SWIG_EXTERNAL");
      Replaceall(tm,"$disown","0");
      Printf(f->code, "%s\n", tm);
    }
    Delete(bbase);
  }
  
  Printv(f->code, "R_SWIG_popCallbackFunctionData(1);\n", NIL);
  Printv(f->code, "\n", UnProtectWrapupCode, NIL);

  if (SwigType_isreference(rettype)) {  
    Printv(f->code,  "return *", Swig_cresult_name(), ";\n", NIL);
  } else if(!isVoidType)
    Printv(f->code,  "return ", Swig_cresult_name(), ";\n", NIL);
  
  Printv(f->code, "\n}\n", NIL);
  Replaceall(f->code, "SWIG_exception_fail", "SWIG_exception_noreturn");
  
  /* To coerce correctly in S, we really want to have an extra/intermediate
     function that handles the scoerceout. 
     We need to check if any of the argument types have an entry in
     that map. If none do, the ignore and call the function straight.
     Otherwise, generate the a marshalling function.
     Need to be able to find it in S. Or use an entirely generic one
     that evaluates the expressions.
     Handle errors in the evaluation of the function by restoring
     the stack, if there is one in use for this function (i.e. no 
     userData).
  */
  
  Wrapper_print(f, f_wrapper);
  
  addFunctionPointerProxy(funName, n, t, s_paramTypes);
  Delete(s_paramTypes);
  Delete(rtype);
  Delete(rettype);
  Delete(funcparams);
  DelWrapper(f);
  
  return funName;
}

void R::init() {
  UnProtectWrapupCode =  
    NewStringf("%s", "vmaxset(r_vmax);\nif(r_nprotect)  Rf_unprotect(r_nprotect);\n\n");
  
  SClassDefs = NewHash();
  
  sfile = NewString("");
  f_init = NewString("");
  s_header = NewString("");
  f_begin = NewString("");
  f_runtime = NewString("");
  f_wrapper = NewString("");
  s_classes = NewString("");
  s_init = NewString("");
  s_init_routine = NewString("");
}



#if 0
int R::cDeclaration(Node *n) {
  SwigType *t = Getattr(n, "type");
  SwigType *name = Getattr(n, "name");
  if (debugMode)
    Printf(stdout, "cDeclaration (%s): %s\n", name, SwigType_lstr(t, 0));
  return Language::cDeclaration(n);
}
#endif


/**
   Method from Language that is called to start the entire
   processing off, i.e. the generation of the code. 
   It is called after the input has been read and parsed.
   Here we open the output streams and generate the code.
***/
int R::top(Node *n) {
  String *module = Getattr(n, "name");
  if(!Rpackage) 
    Rpackage = Copy(module);
  if(!DllName)
    DllName = Copy(module);

  if(outputNamespaceInfo) {
    s_namespace = NewString("");
    Swig_register_filebyname("snamespace", s_namespace);
    Printf(s_namespace, "useDynLib(%s)\n", DllName);
  }

  /* Associate the different streams with names so that they can be used in %insert directives by the
     typemap code. */
  Swig_register_filebyname("sinit", s_init);
  Swig_register_filebyname("sinitroutine", s_init_routine);

  Swig_register_filebyname("begin", f_begin);
  Swig_register_filebyname("runtime", f_runtime);
  Swig_register_filebyname("init", f_init);
  Swig_register_filebyname("header", s_header);
  Swig_register_filebyname("wrapper", f_wrapper);
  Swig_register_filebyname("s", sfile);
  Swig_register_filebyname("sclasses", s_classes);

  Swig_banner(f_begin);

  Printf(f_runtime, "\n\n#ifndef SWIGR\n#define SWIGR\n#endif\n\n");

  
  Swig_banner_target_lang(s_init, "#");
  outputCommandLineArguments(s_init);

  Printf(f_wrapper, "#ifdef __cplusplus\n");
  Printf(f_wrapper, "extern \"C\" {\n");
  Printf(f_wrapper, "#endif\n\n");

  Language::top(n);

  Printf(f_wrapper, "#ifdef __cplusplus\n");
  Printf(f_wrapper, "}\n");
  Printf(f_wrapper, "#endif\n");

  String *type_table = NewString("");
  SwigType_emit_type_table(f_runtime,f_wrapper);
  Delete(type_table);

  if(ClassMemberTable) {
    //XXX OutputClassAccessInfo(ClassMemberTable, sfile);
    Delete(ClassMemberTable);
    ClassMemberTable = NULL;
  }

  Printf(f_init,"}\n");
  if(registrationTable)
    outputRegistrationRoutines(f_init);

  /* Now arrange to write the 2 files - .S and .c. */

  DumpCode(n);

  Delete(sfile);
  Delete(s_classes);
  Delete(s_init);
  Delete(f_wrapper);
  Delete(f_init);

  Delete(s_header);
  Delete(f_runtime);
  Delete(f_begin);

  return SWIG_OK;
}


/*****************************************************
  Write the generated code  to the .S and the .c files.
****************************************************/
int R::DumpCode(Node *n) {
  String *output_filename = NewString("");
  
  
  /* The name of the file in which we will generate the S code. */
  Printf(output_filename, "%s%s.R", SWIG_output_directory(), Rpackage);
  
#ifdef R_SWIG_VERBOSE
  Printf(stdout, "Writing S code to %s\n", output_filename);
#endif
  
  File *scode = NewFile(output_filename, "w", SWIG_output_files());
  if (!scode) {
    FileErrorDisplay(output_filename);
    SWIG_exit(EXIT_FAILURE);
  }
  Delete(output_filename);
  
  
  Printf(scode, "%s\n\n", s_init);
  Printf(scode, "%s\n\n", s_classes);
  Printf(scode, "%s\n", sfile);
  
  Delete(scode);
  String *outfile = Getattr(n,"outfile");
  File *runtime = NewFile(outfile,"w", SWIG_output_files());
  if (!runtime) {
    FileErrorDisplay(outfile);
    SWIG_exit(EXIT_FAILURE);
  }
  
  Printf(runtime, "%s", f_begin);
  Printf(runtime, "%s\n", f_runtime);
  Printf(runtime, "%s\n", s_header);
  Printf(runtime, "%s\n", f_wrapper);
  Printf(runtime, "%s\n", f_init);

  Delete(runtime);

  if(outputNamespaceInfo) {
    output_filename = NewString("");
    Printf(output_filename, "%sNAMESPACE", SWIG_output_directory());
    File *ns = NewFile(output_filename, "w", SWIG_output_files());
    if (!ns) {
      FileErrorDisplay(output_filename);
      SWIG_exit(EXIT_FAILURE);
    }
    Delete(output_filename);
   
    Printf(ns, "%s\n", s_namespace);

    Printf(ns, "\nexport(\n");
    writeListByLine(namespaceFunctions, ns);
    Printf(ns, ")\n");
    Printf(ns, "\nexportMethods(\n");
    writeListByLine(namespaceFunctions, ns, 1);
    Printf(ns, ")\n");
    Delete(ns);
    Delete(s_namespace);
  }

  return SWIG_OK;
}



/*
  We may need to do more.... so this is left as a 
  stub for the moment.
*/
int R::OutputClassAccessInfo(Hash *tb, File *out) {
  int n = OutputClassMemberTable(tb, out);
  OutputClassMethodsTable(out);
  return n;
}

/************************************************************************
  Currently this just writes the information collected about the
  different methods of the C++ classes that have been processed
  to the console. 
  This will be used later to define S4 generics and methods.
**************************************************************************/
int R::OutputClassMethodsTable(File *) {
  Hash *tb = ClassMethodsTable;
  
  if(!tb)
    return SWIG_OK;
  
  List *keys = Keys(tb);
  String *key;
  int i, n = Len(keys);
  if (debugMode) {
    for(i = 0; i < n ; i++ ) {
      key = Getitem(keys, i);
      Printf(stdout, "%d) %s\n", i, key);
      List *els = Getattr(tb, key);
      int nels = Len(els);
      Printf(stdout, "\t");
      for(int j = 0; j < nels; j+=2) {
	Printf(stdout, "%s%s", Getitem(els, j), j < nels - 1 ? ", " : "");
	Printf(stdout, "%s\n", Getitem(els, j+1));
      }
      Printf(stdout, "\n");
    }
  }

  return SWIG_OK;
}


/*
  Iterate over the <class name>_set and <>_get 
  elements and generate the $ and $<- functions
  that provide constrained access to the member
  fields in these elements.

  tb - a hash table that is built up in functionWrapper
  as we process each membervalueHandler.
  The entries are indexed by <class name>_set and 
  <class_name>_get. Each entry is a List *.
   
  out - the stram where the code is to be written. This is the S
  code stream as we generate only S code here..
*/
int R::OutputClassMemberTable(Hash *tb, File *out) {
  List *keys = Keys(tb), *el;
  
  String *key;
  int i, n = Len(keys);
  /* Loop over all the  <Class>_set and <Class>_get entries in the table. */
  /* This function checks for names ending in _set - perhaps it should */
  /* use attributes of some other form, as it potentially clashes with */
  /* methods ending in _set */
  if(n && outputNamespaceInfo) {
    Printf(s_namespace, "exportClasses(");
  }
  for(i = 0; i < n; i++) {
    key = Getitem(keys, i);
    el = Getattr(tb, key);
    
    String *className = Getitem(el, 0);
    char *ptr = Char(key);
    int klen = Len(key);
    int isSet = 0;
    if (klen > 4) {
      ptr = &ptr[klen - 4];
      isSet = strcmp(ptr, "_set") == 0;
    }

    //        OutputArrayMethod(className, el, out);        
    OutputMemberReferenceMethod(className, isSet, el, out);
    
    if(outputNamespaceInfo) 
      Printf(s_namespace, "\"%s\"%s", className, i < n-1 ? "," : "");
  }
  if(n && outputNamespaceInfo) { 
    Printf(s_namespace, ")\n");
  }
  
  return n;
}

/*******************************************************************
 Write the methods for $ or $<- for accessing a member field in an 
 struct or union (or class).
 className - the name of the struct or union (e.g. Bar for struct Bar)
 isSet - a logical value indicating whether the method is for 
           modifying ($<-) or accessing ($) the member field.
 el - a list of length  2 * # accessible member elements  + 1.
      The first element is the name of the class. 
      The other pairs are  member name and the name of the R function to access it.
 out - the stream where we write the code.
********************************************************************/
int R::OutputMemberReferenceMethod(String *className, int isSet, 
				   List *el, File *out) {
  int numMems = Len(el), j;
  int varaccessor = 0;
  if (numMems == 0) 
    return SWIG_OK;
  
  Wrapper *f = NewWrapper(), *attr = NewWrapper();
  
  Printf(f->def, "function(x, name%s)", isSet ? ", value" : "");
  Printf(attr->def, "function(x, i, j, ...%s)", isSet ? ", value" : "");
  
  Printf(f->code, "{\n");
  Printf(f->code, "%saccessorFuns = list(", tab8);

  Node *itemList = NewHash();
  bool has_prev = false;
  for(j = 0; j < numMems; j+=3) {
    String *item = Getitem(el, j);
    if (Getattr(itemList, item)) 
      continue;
    Setattr(itemList, item, "1");
    
    String *dup = Getitem(el, j + 1);
    char *ptr = Char(dup);
    ptr = &ptr[Len(dup) - 3];
    
    if (!strcmp(ptr, "get"))
      varaccessor++;

    String *pitem;
    if (!Strcmp(item, "operator ()")) {
      pitem = NewString("call");
    } else if (!Strcmp(item, "operator ->")) {
      pitem = NewString("deref");
    } else if (!Strcmp(item, "operator +")) {
      pitem = NewString("add");
    } else if (!Strcmp(item, "operator -")) {
      pitem = NewString("sub");
    } else {
      pitem = Copy(item);
    }
    if (has_prev) 
      Printf(f->code, ", ");
    Printf(f->code, "'%s' = %s", pitem, dup);
    has_prev = true;
    Delete(pitem);
  }
  Delete(itemList);
  Printf(f->code, ");\n");
  
  if (!isSet && varaccessor > 0) {
    Printf(f->code, "%svaccessors = c(", tab8);
    int first = 1;
    for(j = 0; j < numMems; j+=3) {
      String *item = Getitem(el, j);
      String *dup = Getitem(el, j + 1);
      char *ptr = Char(dup);
      ptr = &ptr[Len(dup) - 3];
      
      if (!strcmp(ptr, "get")) {
	Printf(f->code, "%s'%s'", first ? "" : ", ", item);
	first = 0;
      }
    }
    Printf(f->code, ");\n");
  }
  
  
  /*    Printv(f->code, tab8,
	"idx = pmatch(name, names(accessorFuns))\n",
	tab8,
	"if(is.na(idx)) {\n",
	tab8, tab4, 
	"stop(\"No ", (isSet ? "modifiable" : "accessible"), " field named \", name, \" in ", className,
	": fields are \", paste(names(accessorFuns), sep = \", \")", 
	")", "\n}\n", NIL); */
  Printv(f->code, ";", tab8,
	 "idx = pmatch(name, names(accessorFuns));\n",
	 tab8,
	 "if(is.na(idx)) \n",
	 tab8, tab4, NIL);
  Printf(f->code, "return(callNextMethod(x, name%s));\n",
	 isSet ? ", value" : "");
  Printv(f->code, tab8, "f = accessorFuns[[idx]];\n", NIL);
  if(isSet) {
    Printv(f->code, tab8, "f(x, value);\n", NIL);
    Printv(f->code, tab8, "x;\n", NIL); // make certain to return the S value.
  } else {
    if (varaccessor) {
      Printv(f->code, tab8,
	     "if (is.na(match(name, vaccessors))) function(...){f(x, ...)} else f(x);\n", NIL);
    } else {
      Printv(f->code, tab8, "function(...){f(x, ...)};\n", NIL);
    }
  }
  Printf(f->code, "}\n");
  
  
  Printf(out, "# Start of accessor method for %s\n", className);
  Printf(out, "setMethod('$%s', '_p%s', ",
	 isSet ? "<-" : "", 
	 getRClassName(className)); 
  Wrapper_print(f, out);
  Printf(out, ");\n");
  
  if(isSet) {
    Printf(out, "setMethod('[[<-', c('_p%s', 'character'),", 
	   getRClassName(className)); 
    Insert(f->code, 2, "name = i;\n");
    Printf(attr->code, "%s", f->code);
    Wrapper_print(attr, out);
    Printf(out, ");\n");
  }
  
  DelWrapper(attr);
  DelWrapper(f);
  
  Printf(out, "# end of accessor method for %s\n", className);
  
  return SWIG_OK;
}

/*******************************************************************
 Write the methods for [ or [<- for accessing a member field in an 
 struct or union (or class).
 className - the name of the struct or union (e.g. Bar for struct Bar)
 el - a list of length  2 * # accessible member elements  + 1.
      The first element is the name of the class. 
      The other pairs are  member name and the name of the R function to access it.
 out - the stream where we write the code.
********************************************************************/
int R::OutputArrayMethod(String *className, List *el, File *out) {
  int numMems = Len(el), j;
  
  if(!el || numMems == 0)
    return(0);
  
  Printf(out, "# start of array methods for %s\n", className);
  for(j = 0; j < numMems; j+=3) {
    String *item = Getitem(el, j);
    String *dup = Getitem(el, j + 1);
    if (!Strcmp(item, "__getitem__")) {
      Printf(out, 
	     "setMethod('[', '_p%s', function(x, i, j, ..., drop =TRUE) ", 
	     getRClassName(className));
      Printf(out, "  sapply(i, function (n)  %s(x, as.integer(n-1))))\n\n", dup);
    }
    if (!Strcmp(item, "__setitem__")) {
      Printf(out, "setMethod('[<-', '_p%s', function(x, i, j, ..., value)", 
	     getRClassName(className));
      Printf(out, "  sapply(1:length(i), function(n) %s(x, as.integer(i[n]-1), value[n])))\n\n", dup);
    }
    
  }
  
  Printf(out, "# end of array methods for %s\n", className);
  
  return SWIG_OK;
}


/************************************************************
 Called when a enumeration is to be processed.
 We want to call the R function defineEnumeration().
 tdname is the typedef of the enumeration, i.e. giving its name.
*************************************************************/
int R::enumDeclaration(Node *n) {
  if (getCurrentClass() && (cplus_mode != PUBLIC))
    return SWIG_NOWRAP;

  String *name = Getattr(n, "name");
  String *tdname = Getattr(n, "tdname");
  
  /* Using name if tdname is empty. */
  
  if(Len(tdname) == 0)
    tdname = name;


  if(!tdname || Strcmp(tdname, "") == 0) {
    Language::enumDeclaration(n);
    return SWIG_OK;
  }
  
  String *mangled_tdname = SwigType_manglestr(tdname);
  String *scode = NewString("");
  
  Printv(scode, "defineEnumeration('", mangled_tdname, "'", 
	 ",\n",  tab8, tab8, tab4, ".values = c(\n", NIL);
  
  Node *c;
  int value = -1; // First number is zero
  for (c = firstChild(n); c; c = nextSibling(c)) {
    //      const char *tag = Char(nodeType(c));
    //      if (Strcmp(tag,"cdecl") == 0) {        
    name = Getattr(c, "name");
    String *val = Getattr(c, "enumvalue");
    if(val && Char(val)) {
      int inval = (int) getNumber(val);
      if(inval == DEFAULT_NUMBER) 
	value++;
      else 
	value = inval;
    } else
      value++;
    
    Printf(scode, "%s%s%s'%s' = %d%s\n", tab8, tab8, tab8, name, value,
	   nextSibling(c) ? ", " : "");
    //      }
  }
  
  Printv(scode, "))", NIL);
  Printf(sfile, "%s\n", scode);
  
  Delete(scode);
  Delete(mangled_tdname);
  
  return SWIG_OK;
}


/*************************************************************
**************************************************************/
int R::variableWrapper(Node *n) {
  String *name = Getattr(n, "sym:name");
  
  processing_variable = 1;
  Language::variableWrapper(n); // Force the emission of the _set and _get function wrappers.
  processing_variable = 0;
  
  
  SwigType *ty = Getattr(n, "type");
  int addCopyParam = addCopyParameter(ty);
  
  //XXX
  processType(ty, n);
  
  if(!SwigType_isconst(ty)) {
    Wrapper *f = NewWrapper();
    Printf(f->def, "%s = \nfunction(value%s)\n{\n", 
	   name, addCopyParam ? ", .copy = FALSE" : "");
    Printv(f->code, "if(missing(value)) {\n", 
	   name, "_get(", addCopyParam ? ".copy" : "", ")\n}", NIL);
    Printv(f->code, " else {\n", 
	   name, "_set(value)\n}\n}", NIL);
    
    Wrapper_print(f, sfile);
    DelWrapper(f);
  } else {
    Printf(sfile, "%s = %s_get\n", name, name);
  }

  return SWIG_OK;
}


void R::addAccessor(String *memberName, Wrapper *wrapper, String *name, 
		    int isSet) {
  if(isSet < 0) {
    int n = Len(name);
    char *ptr = Char(name);
    if (n>4) {
      isSet = Strcmp(NewString(&ptr[n-4]), "_set") == 0;
    }
  }
  
  List *l = isSet ? class_member_set_functions : class_member_functions;
  
  if(!l) {
    l = NewList();
    if(isSet)
      class_member_set_functions = l;
    else
      class_member_functions = l;
  }
  
  Append(l, memberName);
  Append(l, name);
  
  String *tmp = NewString("");
  Wrapper_print(wrapper, tmp);
  Append(l, tmp);
  // if we could put the wrapper in directly:       Append(l, Copy(sfun));
  if (debugMode)
    Printf(stdout, "Adding accessor: %s (%s) => %s\n", memberName, name, tmp);
}

#define MAX_OVERLOAD 256

struct Overloaded {
  Node      *n;          /* Node                               */
  int        argc;       /* Argument count                     */
  ParmList  *parms;      /* Parameters used for overload check */
  int        error;      /* Ambiguity error                    */
};


List * R::Swig_overload_rank(Node *n, 
				 bool script_lang_wrapping) {
  Overloaded  nodes[MAX_OVERLOAD];
  int         nnodes = 0;
  Node *o = Getattr(n,"sym:overloaded");


  if (!o) return 0;

  Node *c = o;
  while (c) {
    if (Getattr(c,"error")) {
      c = Getattr(c,"sym:nextSibling");
      continue;
    }
    /*    if (SmartPointer && Getattr(c,"cplus:staticbase")) {
	  c = Getattr(c,"sym:nextSibling");
	  continue;
	  } */

    /* Make a list of all the declarations (methods) that are overloaded with
     * this one particular method name */

    if (Getattr(c,"wrap:name")) {
      nodes[nnodes].n = c;
      nodes[nnodes].parms = Getattr(c,"wrap:parms");
      nodes[nnodes].argc = emit_num_required(nodes[nnodes].parms);
      nodes[nnodes].error = 0;
      nnodes++;
    }
    c = Getattr(c,"sym:nextSibling");
  }
  
  /* Sort the declarations by required argument count */
  {
    int i,j;
    for (i = 0; i < nnodes; i++) {
      for (j = i+1; j < nnodes; j++) {
	if (nodes[i].argc > nodes[j].argc) {
	  Overloaded t = nodes[i];
	  nodes[i] = nodes[j];
	  nodes[j] = t;
	}
      }
    }
  }

  /* Sort the declarations by argument types */
  {
    int i,j;
    for (i = 0; i < nnodes-1; i++) {
      if (nodes[i].argc == nodes[i+1].argc) {
	for (j = i+1; (j < nnodes) && (nodes[j].argc == nodes[i].argc); j++) {
	  Parm *p1 = nodes[i].parms;
	  Parm *p2 = nodes[j].parms;
	  int   differ = 0;
	  int   num_checked = 0;
	  while (p1 && p2 && (num_checked < nodes[i].argc)) {
	    if (debugMode) {
	      Printf(stdout,"p1 = '%s', p2 = '%s'\n", Getattr(p1,"type"), Getattr(p2,"type"));
	    }
	    if (checkAttribute(p1,"tmap:in:numinputs","0")) {
	      p1 = Getattr(p1,"tmap:in:next");
	      continue;
	    }
	    if (checkAttribute(p2,"tmap:in:numinputs","0")) {
	      p2 = Getattr(p2,"tmap:in:next");
	      continue;
	    }
	    String *t1 = Getattr(p1,"tmap:typecheck:precedence");
	    String *t2 = Getattr(p2,"tmap:typecheck:precedence");
	    if (debugMode) {
	      Printf(stdout,"t1 = '%s', t2 = '%s'\n", t1, t2);
	    }
	    if ((!t1) && (!nodes[i].error)) {
	      Swig_warning(WARN_TYPEMAP_TYPECHECK, Getfile(nodes[i].n), Getline(nodes[i].n),
			   "Overloaded method %s not supported (incomplete type checking rule - no precedence level in typecheck typemap for '%s').\n",
			   Swig_name_decl(nodes[i].n), SwigType_str(Getattr(p1, "type"), 0));
	      nodes[i].error = 1;
	    } else if ((!t2) && (!nodes[j].error)) {
	      Swig_warning(WARN_TYPEMAP_TYPECHECK, Getfile(nodes[j].n), Getline(nodes[j].n),
			   "Overloaded method %s not supported (incomplete type checking rule - no precedence level in typecheck typemap for '%s').\n",
			   Swig_name_decl(nodes[j].n), SwigType_str(Getattr(p2, "type"), 0));
	      nodes[j].error = 1;
	    }
	    if (t1 && t2) {
	      int t1v, t2v;
	      t1v = atoi(Char(t1));
	      t2v = atoi(Char(t2));
	      differ = t1v-t2v;
	    }
	    else if (!t1 && t2) differ = 1;
	    else if (t1 && !t2) differ = -1;
	    else if (!t1 && !t2) differ = -1;
	    num_checked++;
	    if (differ > 0) {
	      Overloaded t = nodes[i];
	      nodes[i] = nodes[j];
	      nodes[j] = t;
	      break;
	    } else if ((differ == 0) && (Strcmp(t1,"0") == 0)) {
	      t1 = Getattr(p1,"ltype");
	      if (!t1) {
		t1 = SwigType_ltype(Getattr(p1,"type"));
		if (Getattr(p1,"tmap:typecheck:SWIGTYPE")) {
		  SwigType_add_pointer(t1);
		}
		Setattr(p1,"ltype",t1);
	      }
	      t2 = Getattr(p2,"ltype");
	      if (!t2) {
		t2 = SwigType_ltype(Getattr(p2,"type"));
		if (Getattr(p2,"tmap:typecheck:SWIGTYPE")) {
		  SwigType_add_pointer(t2);
		}
		Setattr(p2,"ltype",t2);
	      }

	      /* Need subtype check here.  If t2 is a subtype of t1, then we need to change the
                 order */

	      if (SwigType_issubtype(t2,t1)) {
		Overloaded t = nodes[i];
		nodes[i] = nodes[j];
		nodes[j] = t;
	      }

	      if (Strcmp(t1,t2) != 0) {
		differ = 1;
		break;
	      }
	    } else if (differ) {
	      break;
	    }
	    if (Getattr(p1,"tmap:in:next")) {
	      p1 = Getattr(p1,"tmap:in:next");
	    } else {
	      p1 = nextSibling(p1);
	    }
	    if (Getattr(p2,"tmap:in:next")) {
	      p2 = Getattr(p2,"tmap:in:next");
	    } else {
	      p2 = nextSibling(p2);
	    }
	  }
	  if (!differ) {
	    /* See if declarations differ by const only */
	    String *d1 = Getattr(nodes[i].n, "decl");
	    String *d2 = Getattr(nodes[j].n, "decl");
	    if (d1 && d2) {
	      String *dq1 = Copy(d1);
	      String *dq2 = Copy(d2);
	      if (SwigType_isconst(d1)) {
		Delete(SwigType_pop(dq1));
	      }
	      if (SwigType_isconst(d2)) {
		Delete(SwigType_pop(dq2));
	      }
	      if (Strcmp(dq1, dq2) == 0) {

		if (SwigType_isconst(d1) && !SwigType_isconst(d2)) {
		  if (script_lang_wrapping) {
		    // Swap nodes so that the const method gets ignored (shadowed by the non-const method)
		    Overloaded t = nodes[i];
		    nodes[i] = nodes[j];
		    nodes[j] = t;
		  }
		  differ = 1;
		  if (!nodes[j].error) {
		    if (script_lang_wrapping) {
		      Swig_warning(WARN_LANG_OVERLOAD_CONST, Getfile(nodes[j].n), Getline(nodes[j].n),
				   "Overloaded method %s ignored,\n", Swig_name_decl(nodes[j].n));
		      Swig_warning(WARN_LANG_OVERLOAD_CONST, Getfile(nodes[i].n), Getline(nodes[i].n),
				   "using non-const method %s instead.\n", Swig_name_decl(nodes[i].n));
		    } else {
		      if (!Getattr(nodes[j].n, "overload:ignore")) {
			Swig_warning(WARN_LANG_OVERLOAD_IGNORED, Getfile(nodes[j].n), Getline(nodes[j].n),
				     "Overloaded method %s ignored,\n", Swig_name_decl(nodes[j].n));
			Swig_warning(WARN_LANG_OVERLOAD_IGNORED, Getfile(nodes[i].n), Getline(nodes[i].n),
				     "using %s instead.\n", Swig_name_decl(nodes[i].n));
		      }
		    }
		  }
		  nodes[j].error = 1;
		} else if (!SwigType_isconst(d1) && SwigType_isconst(d2)) {
		  differ = 1;
		  if (!nodes[j].error) {
		    if (script_lang_wrapping) {
		      Swig_warning(WARN_LANG_OVERLOAD_CONST, Getfile(nodes[j].n), Getline(nodes[j].n),
				   "Overloaded method %s ignored,\n", Swig_name_decl(nodes[j].n));
		      Swig_warning(WARN_LANG_OVERLOAD_CONST, Getfile(nodes[i].n), Getline(nodes[i].n),
				   "using non-const method %s instead.\n", Swig_name_decl(nodes[i].n));
		    } else {
		      if (!Getattr(nodes[j].n, "overload:ignore")) {
			Swig_warning(WARN_LANG_OVERLOAD_IGNORED, Getfile(nodes[j].n), Getline(nodes[j].n),
				     "Overloaded method %s ignored,\n", Swig_name_decl(nodes[j].n));
			Swig_warning(WARN_LANG_OVERLOAD_IGNORED, Getfile(nodes[i].n), Getline(nodes[i].n),
				     "using %s instead.\n", Swig_name_decl(nodes[i].n));
		      }
		    }
		  }
		  nodes[j].error = 1;
		}
	      }
	      Delete(dq1);
	      Delete(dq2);
	    }
	  }
	  if (!differ) {
	    if (!nodes[j].error) {
	      if (script_lang_wrapping) {
		Swig_warning(WARN_LANG_OVERLOAD_SHADOW, Getfile(nodes[j].n), Getline(nodes[j].n),
			     "Overloaded method %s effectively ignored,\n", Swig_name_decl(nodes[j].n));
		Swig_warning(WARN_LANG_OVERLOAD_SHADOW, Getfile(nodes[i].n), Getline(nodes[i].n),
			     "as it is shadowed by %s.\n", Swig_name_decl(nodes[i].n));
	      } else {
		if (!Getattr(nodes[j].n, "overload:ignore")) {
		  Swig_warning(WARN_LANG_OVERLOAD_IGNORED, Getfile(nodes[j].n), Getline(nodes[j].n),
			       "Overloaded method %s ignored,\n", Swig_name_decl(nodes[j].n));
		  Swig_warning(WARN_LANG_OVERLOAD_IGNORED, Getfile(nodes[i].n), Getline(nodes[i].n),
			       "using %s instead.\n", Swig_name_decl(nodes[i].n));
		}
	      }
	      nodes[j].error = 1;
	    }
	  }
	}
      }
    }
  }
  List *result = NewList();
  {
    int i;
    for (i = 0; i < nnodes; i++) {
      if (nodes[i].error)
        Setattr(nodes[i].n, "overload:ignore", "1");
      Append(result,nodes[i].n);
      //      Printf(stdout,"[ %d ] %s\n", i, ParmList_errorstr(nodes[i].parms));
      //      Swig_print_node(nodes[i].n);
    }
  }
  return result;
}

void R::dispatchFunction(Node *n) {
  Wrapper *f = NewWrapper();
  String *symname = Getattr(n, "sym:name");
  String *nodeType = Getattr(n, "nodeType");
  bool constructor = (!Cmp(nodeType, "constructor")); 

  String *sfname = NewString(symname);

  if (constructor)
    Replace(sfname, "new_", "", DOH_REPLACE_FIRST);

  Printf(f->def,
	 "`%s` <- function(...) {", sfname);
  if (debugMode) {
    Swig_print_node(n);
  }
  List *dispatch = Swig_overload_rank(n, true);
  int   nfunc = Len(dispatch);
  Printv(f->code, 
	 "argtypes <- mapply(class, list(...));\n",
	 "argv <- list(...);\n",
	 "argc <- length(argtypes);\n", NIL );

  Printf(f->code, "# dispatch functions %d\n", nfunc);
  int cur_args = -1;
  bool first_compare = true;
  for (int i=0; i < nfunc; i++) {
    Node *ni = Getitem(dispatch,i);
    Parm *pi = Getattr(ni,"wrap:parms");
    int num_arguments = emit_num_arguments(pi);

    String *overname = Getattr(ni,"sym:overname");      
    if (cur_args != num_arguments) {
      if (cur_args != -1) {
	Printv(f->code, "} else ", NIL);
      }
      Printf(f->code, "if (argc == %d) {", num_arguments);
      cur_args = num_arguments;
      first_compare = true;
    }
    Parm *p;
    int j;
    if (num_arguments > 0) {
      if (!first_compare) {
	Printv(f->code, " else ", NIL);
      } else {
	first_compare = false;
      }
      Printv(f->code, "if (", NIL);
      for (p =pi, j = 0 ; j < num_arguments ; j++) {
	if (debugMode) {
	  Swig_print_node(p);
	}
	String *tm = Swig_typemap_lookup("rtype", p, "", 0);
	if(tm) {
	  replaceRClass(tm, Getattr(p, "type"));
	}

	String *tmcheck = Swig_typemap_lookup("rtypecheck", p, "", 0);
	if (tmcheck) {
	  String *tmp = NewString("");
	  Printf(tmp, "argv[[%d]]", j+1);
	  Replaceall(tmcheck, "$arg", tmp);
	  Printf(tmp, "argtype[%d]", j+1);
	  Replaceall(tmcheck, "$argtype", tmp);
	  if (tm) {
	    Replaceall(tmcheck, "$rtype", tm);
	  }
	  if (debugMode) {
	    Printf(stdout, "<rtypecheck>%s\n", tmcheck);
	  }
	  Printf(f->code, "%s(%s)",
		 j == 0? "" : " && ",
		 tmcheck);
	  p = Getattr(p, "tmap:in:next");
	  continue;
	}
	if (tm) {
	  if (Strcmp(tm,"numeric")==0) {
	    Printf(f->code, "%sis.numeric(argv[[%d]])",
		j == 0 ? "" : " && ",
		j+1);
	  }
	  else if (Strcmp(tm,"integer")==0) {
	    Printf(f->code, "%s(is.integer(argv[[%d]]) || is.numeric(argv[[%d]]))",
		j == 0 ? "" : " && ",
		j+1, j+1);
	  }
	  else if (Strcmp(tm,"character")==0) {
	    Printf(f->code, "%sis.character(argv[[%d]])",
		j == 0 ? "" : " && ",
		j+1);
	  }
	  else {
	    Printf(f->code, "%sextends(argtypes[%d], '%s')",
		j == 0 ? "" : " && ",
		j+1,
		tm);
	  }
	}
	if (!SwigType_ispointer(Getattr(p, "type"))) {
	  Printf(f->code, " && length(argv[[%d]]) == 1",
	      j+1);
	}
	p = Getattr(p, "tmap:in:next");
      }
      Printf(f->code, ") { f <- %s%s; }\n", sfname, overname);
    } else {
      Printf(f->code, "f <- %s%s; ", sfname, overname);
    }
  }
  if (cur_args != -1) {
    Printf(f->code, "} else {\n"
	   "stop(\"cannot find overloaded function for %s with argtypes (\","
	   "toString(argtypes),\")\");\n"
	   "}", sfname);
  }
  Printv(f->code, ";\nf(...)", NIL);
  Printv(f->code, ";\n}", NIL);
  Wrapper_print(f, sfile);
  Printv(sfile, "# Dispatch function\n", NIL);
  DelWrapper(f);
}

/******************************************************************

*******************************************************************/
int R::functionWrapper(Node *n) {
  String *fname = Getattr(n, "name");
  String *iname = Getattr(n, "sym:name");
  String *type = Getattr(n, "type"); 
  
  if (debugMode) {
    Printf(stdout, 
	   "<functionWrapper> %s %s %s\n", fname, iname, type);
  }
  String *overname = 0;
  String *nodeType = Getattr(n, "nodeType");
  bool constructor = (!Cmp(nodeType, "constructor")); 
  bool destructor = (!Cmp(nodeType, "destructor")); 
  
  String *sfname = NewString(iname);
  
  if (constructor)
    Replace(sfname, "new_", "", DOH_REPLACE_FIRST);
  
  if (Getattr(n,"sym:overloaded")) {
    overname = Getattr(n,"sym:overname");      
    Append(sfname, overname);
  }
  
  if (debugMode) 
    Printf(stdout, 
	   "<functionWrapper> processing parameters\n");
  
  
  ParmList *l = Getattr(n, "parms");
  Parm *p;
  String *tm;
  
  p = l;
  while(p) {
    SwigType *resultType = Getattr(p, "type");
    if (expandTypedef(resultType) && 
	SwigType_istypedef(resultType)) {
      SwigType *resolved =
	SwigType_typedef_resolve_all(resultType);
      if (expandTypedef(resolved)) {
	Setattr(p, "type", Copy(resolved));
      }
    }
    p = nextSibling(p);
  } 

  String *unresolved_return_type = 
    Copy(type);
  if (expandTypedef(type) &&
      SwigType_istypedef(type)) {
    SwigType *resolved = 
      SwigType_typedef_resolve_all(type);
    if (expandTypedef(resolved)) {
      type = Copy(resolved);
      Setattr(n, "type", type);
    }
  }
  if (debugMode) 
    Printf(stdout, "<functionWrapper> unresolved_return_type %s\n",
	   unresolved_return_type);
  if(processing_member_access_function) {
    if (debugMode)
      Printf(stdout, "<functionWrapper memberAccess> '%s' '%s' '%s' '%s'\n", 
	     fname, iname, member_name, class_name);
    
    if(opaqueClassDeclaration)
      return SWIG_OK;
      
      
    /* Add the name of this member to a list for this class_name. 
       We will dump all these at the end. */
    
    int n = Len(iname);
    char *ptr = Char(iname);
    bool isSet(0);
    if (n > 4) isSet = Strcmp(NewString(&ptr[n-4]), "_set") == 0;
    
    
    String *tmp = NewString("");
    Printf(tmp, "%s_%s", class_name, isSet ? "set" : "get");
    
    List *memList = Getattr(ClassMemberTable, tmp);
    if(!memList) {
      memList = NewList();
      Append(memList, class_name);
      Setattr(ClassMemberTable, tmp, memList);
    }
    Delete(tmp);
    Append(memList, member_name);
    Append(memList, iname);
  }
  
  int i;
  int nargs;
  
  String *wname = Swig_name_wrapper(iname);
  Replace(wname, "_wrap", "R_swig", DOH_REPLACE_FIRST);
  if(overname) 
    Append(wname, overname);
  Setattr(n,"wrap:name", wname);

  Wrapper *f = NewWrapper();
  Wrapper *sfun = NewWrapper();
  
  int isVoidReturnType = (Strcmp(type, "void") == 0);
  // Need to use the unresolved return type since 
  // typedef resolution removes the const which causes a 
  // mismatch with the function action
  emit_return_variable(n, unresolved_return_type, f);

  SwigType *rtype = Getattr(n, "type");
  int addCopyParam = 0;

  if(!isVoidReturnType) 
    addCopyParam = addCopyParameter(rtype);


  // Can we get the nodeType() of the type node! and see if it is a struct.
  //    int addCopyParam = SwigType_isclass(rtype);

  //    if(addCopyParam)
  if (debugMode)
    Printf(stdout, "Adding a .copy argument to %s for %s = %s\n", 
	   iname, type, addCopyParam ? "yes" : "no");

  Printv(f->def, "SWIGEXPORT SEXP\n", wname, " ( ", NIL);

  Printf(sfun->def, "# Start of %s\n", iname);         
  Printv(sfun->def, "\n`", sfname, "` = function(", NIL);

  if(outputNamespaceInfo) //XXX Need to be a little more discriminating
    addNamespaceFunction(iname);

  Swig_typemap_attach_parms("scoercein", l, f);
  Swig_typemap_attach_parms("scoerceout", l, f);
  Swig_typemap_attach_parms("scheck", l, f);

  emit_parameter_variables(l, f);
  emit_attach_parmmaps(l,f);
  Setattr(n,"wrap:parms",l);

  nargs = emit_num_arguments(l);

  Wrapper_add_local(f, "r_nprotect", "unsigned int r_nprotect = 0");
  Wrapper_add_localv(f, "r_ans", "SEXP", "r_ans = R_NilValue", NIL);
  Wrapper_add_localv(f, "r_vmax", "VMAXTYPE", "r_vmax = vmaxget()", NIL);

  String *sargs = NewString("");


  String *s_inputTypes = NewString("");
  String *s_inputMap = NewString("");
  bool inFirstArg = true;
  bool inFirstType = true;
  Parm *curP;
  for (p =l, i = 0 ; i < nargs ; i++) {

    while (checkAttribute(p, "tmap:in:numinputs", "0")) {
      p = Getattr(p, "tmap:in:next");
    }

    SwigType *tt = Getattr(p, "type");
    int nargs = -1;
    String *funcptr_name = processType(tt, p, &nargs);

    //      SwigType *tp = Getattr(p, "type");
    String   *name  = Getattr(p,"name");
    String   *lname  = Getattr(p,"lname");

    // R keyword renaming
    if (name) {
      if (Swig_name_warning(p, 0, name, 0)) {
	name = 0;
      } else {
	/* If we have a :: in the parameter name because we are accessing a static member of a class, say, then
	   we need to remove that prefix. */
	while (Strstr(name, "::")) {
	  //XXX need to free.
	  name = NewStringf("%s", Strchr(name, ':') + 2);
	  if (debugMode)
	    Printf(stdout, "+++  parameter name with :: in it %s\n", name);
	}
      }
    }
    if (!name || Len(name) == 0)
      name = NewStringf("s_arg%d", i+1);

    name = replaceInitialDash(name);

    if (!Strncmp(name, "arg", 3)) {
      name = Copy(name);
      Insert(name, 0, "s_");
    }
      
    if(processing_variable) {
      name = Copy(name);
      Insert(name, 0, "s_");
    }

    if(!Strcmp(name, fname)) {
      name = Copy(name);
      Insert(name, 0, "s_");
    }

    Printf(sargs, "%s, ", name);

    String *tm;
    if((tm = Getattr(p, "tmap:scoercein"))) {
      Replaceall(tm, "$input", name);
      replaceRClass(tm, Getattr(p, "type"));

      if(funcptr_name) {
	//XXX need to get this to return non-zero
	if(nargs == -1)
	  nargs = getFunctionPointerNumArgs(p, tt);

	String *snargs = NewStringf("%d", nargs);
	Printv(sfun->code, "if(is.function(", name, ")) {", "\n",
	       "assert('...' %in% names(formals(", name, 
	       ")) || length(formals(", name, ")) >= ", snargs, ");\n} ", NIL);
	Delete(snargs);

	Printv(sfun->code, "else {\n",
	       "if(is.character(", name, ")) {\n",
	       name, " = getNativeSymbolInfo(", name, ");",
	       "\n};\n",
	       "if(is(", name, ", \"NativeSymbolInfo\")) {\n",
	       name, " = ", name, "$address", ";\n}\n",
	       "if(is(", name, ", \"ExternalReference\")) {\n",
	       name, " = ", name, "@ref;\n}\n",
	       "}; \n",
	       NIL);
      } else {
	Printf(sfun->code, "%s\n", tm);
      }
    }

    Printv(sfun->def, inFirstArg ? "" : ", ", name, NIL);

    if ((tm = Getattr(p,"tmap:scheck"))) {

      Replaceall(tm,"$target", lname);
      Replaceall(tm,"$source", name);
      Replaceall(tm,"$input", name);
      replaceRClass(tm, Getattr(p, "type"));
      Printf(sfun->code,"%s\n",tm);
    }



    curP = p;
    if ((tm = Getattr(p,"tmap:in"))) {

      Replaceall(tm,"$target", lname);
      Replaceall(tm,"$source", name);
      Replaceall(tm,"$input", name);

      if (Getattr(p,"wrap:disown") || (Getattr(p,"tmap:in:disown"))) {
	Replaceall(tm,"$disown","SWIG_POINTER_DISOWN");
      } else {
	Replaceall(tm,"$disown","0");
      }

      if(funcptr_name) {
	/* have us a function pointer */
	Printf(f->code, "if(TYPEOF(%s) != CLOSXP) {\n", name);
	Replaceall(tm,"$R_class", "");
      } else {
	replaceRClass(tm, Getattr(p, "type"));
      }


      Printf(f->code,"%s\n",tm);
      if(funcptr_name) 
	Printf(f->code, "} else {\n%s = %s;\nR_SWIG_pushCallbackFunctionData(%s, NULL);\n}\n", 
	       lname, funcptr_name, name);
      Printv(f->def, inFirstArg ? "" : ", ", "SEXP ", name, NIL);
      if (Len(name) != 0) 
	inFirstArg = false;
      p = Getattr(p,"tmap:in:next");

    } else {
      p = nextSibling(p);
    }


    tm = Swig_typemap_lookup("rtype", curP, "", 0);
    if(tm) {
      replaceRClass(tm, Getattr(curP, "type"));
    }
    Printf(s_inputTypes, "%s'%s'", inFirstType ? "" : ", ", tm);
    Printf(s_inputMap, "%s%s='%s'", inFirstType ? "" : ", ", name, tm);
    inFirstType = false;

    if(funcptr_name) 
      Delete(funcptr_name);
  } /* end of looping over parameters. */

  if(addCopyParam) {
    Printf(sfun->def, "%s.copy = FALSE", nargs > 0 ? ", " : "");
    Printf(f->def, "%sSEXP s_swig_copy", nargs > 0 ? ", " : "");

    Printf(sargs, "as.logical(.copy), ");
  }

  Printv(f->def, ")\n{\n", NIL);
  Printv(sfun->def, ")\n{\n", NIL);


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

  String *outargs = NewString("");
  int numOutArgs = isVoidReturnType ? -1 : 0;
  for(p = l, i = 0; p; i++) {
    if((tm = Getattr(p, "tmap:argout"))) {
      //       String *lname =  Getattr(p, "lname");
      numOutArgs++;
      String *pos = NewStringf("%d", numOutArgs);
      Replaceall(tm,"$source", Getattr(p, "lname"));
      Replaceall(tm,"$result", "r_ans");
      Replaceall(tm,"$n", pos); // The position into which to store the answer.
      Replaceall(tm,"$arg", Getattr(p, "emit:input"));
      Replaceall(tm,"$input", Getattr(p, "emit:input"));
      Replaceall(tm,"$owner", "R_SWIG_EXTERNAL");


      Printf(outargs, "%s\n", tm);
      p = Getattr(p,"tmap:argout:next");
    } else
      p = nextSibling(p);
  }

  String *actioncode = emit_action(n);

  /* Deal with the explicit return value. */
  if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) { 
    SwigType *retType = Getattr(n, "type");
    //Printf(stdout, "Return Value for %s, array? %s\n", retType, SwigType_isarray(retType) ? "yes" : "no");     
    /*      if(SwigType_isarray(retType)) {
	    defineArrayAccessors(retType);
	    } */


    Replaceall(tm,"$1", Swig_cresult_name());
    Replaceall(tm,"$result", "r_ans");
    replaceRClass(tm, retType);

    if (GetFlag(n,"feature:new")) {
      Replaceall(tm, "$owner", "R_SWIG_OWNER");
    } else {
      Replaceall(tm,"$owner", "R_SWIG_EXTERNAL");
    }

#if 0
    if(addCopyParam) {
      Printf(f->code, "if(LOGICAL(s_swig_copy)[0]) {\n");
      Printf(f->code, "/* Deal with returning a reference. */\nr_ans = R_NilValue;\n");
      Printf(f->code, "}\n else {\n");
    } 
#endif
    Printf(f->code, "%s\n", tm);
#if 0
    if(addCopyParam) 
      Printf(f->code, "}\n"); /* end of if(s_swig_copy) ... else { ... } */
#endif

  } else {
    Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number,
		 "Unable to use return type %s in function %s.\n", SwigType_str(type, 0), fname);
  }


  if(Len(outargs)) {
    Wrapper_add_local(f, "R_OutputValues", "SEXP R_OutputValues");

    String *tmp = NewString("");
    if(!isVoidReturnType)
      Printf(tmp, "Rf_protect(r_ans);\n");

    Printf(tmp, "Rf_protect(R_OutputValues = Rf_allocVector(VECSXP,%d));\nr_nprotect += %d;\n", 
	   numOutArgs + !isVoidReturnType, 
	   isVoidReturnType ? 1 : 2);

    if(!isVoidReturnType)
      Printf(tmp, "SET_VECTOR_ELT(R_OutputValues, 0, r_ans);\n");
    Printf(tmp, "r_ans = R_OutputValues;\n");

    Insert(outargs, 0, tmp);
    Delete(tmp); 



    Printv(f->code, outargs, NIL);
    Delete(outargs);

  }

  /* Output cleanup code */
  Printv(f->code, cleanup, NIL);
  Delete(cleanup);

  /* Look to see if there is any newfree cleanup code */
  if (GetFlag(n, "feature:new")) {
    if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
      Replaceall(tm, "$source", Swig_cresult_name());	/* deprecated */
      Printf(f->code, "%s\n", tm);
    }
  }

  /* See if there is any return cleanup code */
  if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
    Replaceall(tm, "$source", Swig_cresult_name());
    Printf(f->code, "%s\n", tm);
    Delete(tm);
  }

  Printv(f->code, UnProtectWrapupCode, NIL);

  /*If the user gave us something to convert the result in  */
  if ((tm = Swig_typemap_lookup("scoerceout", n, Swig_cresult_name(), sfun))) {
    Replaceall(tm,"$source","ans");
    Replaceall(tm,"$result","ans");
    replaceRClass(tm, Getattr(n, "type"));
    Chop(tm);
  }


  Printv(sfun->code, ";", (Len(tm) ? "ans = " : ""), ".Call('", wname, 
	 "', ", sargs, "PACKAGE='", Rpackage, "');\n", NIL);
  if(Len(tm))
    {
      Printf(sfun->code, "%s\n\n", tm); 
      if (constructor)
	{ 
	  String *finalizer = NewString(iname);
	  Replace(finalizer, "new_", "", DOH_REPLACE_FIRST);
	  Printf(sfun->code, "reg.finalizer(ans@ref, delete_%s)\n", finalizer);
	}                                                                      
      Printf(sfun->code, "ans\n");
    }

  if (destructor)
    Printv(f->code, "R_ClearExternalPtr(self);\n", NIL);

  Printv(f->code, "return r_ans;\n}\n", NIL);
  Printv(sfun->code, "\n}", NIL);

  /* Substitute the function name */
  Replaceall(f->code,"$symname",iname);

  Wrapper_print(f, f_wrapper);
  Wrapper_print(sfun, sfile);

  Printf(sfun->code, "\n# End of %s\n", iname);
  tm = Swig_typemap_lookup("rtype", n, "", 0);
  if(tm) {
    SwigType *retType = Getattr(n, "type");
    replaceRClass(tm, retType);
  }  
    
  Printv(sfile, "attr(`", sfname, "`, 'returnType') = '", 
	 isVoidReturnType ? "void" : (tm ? tm : ""), 
	 "'\n", NIL); 
    
  if(nargs > 0)
    Printv(sfile, "attr(`", sfname, "`, \"inputTypes\") = c(",
	   s_inputTypes, ")\n", NIL);
  Printv(sfile, "class(`", sfname, "`) = c(\"SWIGFunction\", class('", 
	 sfname, "'))\n\n", NIL); 

  if (memoryProfile) {
    Printv(sfile, "memory.profile()\n", NIL);
  }
  if (aggressiveGc) {
    Printv(sfile, "gc()\n", NIL);
  }

  // Printv(sfile, "setMethod('", name, "', '", name, "', ", iname, ")\n\n\n");



  /* If we are dealing with a method in an C++ class, then 
     add the name of the R function and its definition. 
     XXX need to figure out how to store the Wrapper if possible in the hash/list.
     Would like to be able to do this so that we can potentially insert
  */
  if(processing_member_access_function || processing_class_member_function) {
    addAccessor(member_name, sfun, iname);
  }

  if (Getattr(n, "sym:overloaded") &&
      !Getattr(n, "sym:nextSibling")) {
    dispatchFunction(n);
  }

  addRegistrationRoutine(wname, addCopyParam ? nargs +1 : nargs);

  DelWrapper(f);
  DelWrapper(sfun);

  Delete(sargs);
  Delete(sfname);
  return SWIG_OK;
}

/* ----------------------------------------------------------------------
 * R::constantWrapper()
 * ---------------------------------------------------------------------- */

int R::constantWrapper(Node *n) {
  (void) n;
  // TODO
  return SWIG_OK;
}

/*****************************************************
 Add the specified routine name to the collection of 
 generated routines that are called from R functions.
 This is used to register the routines with R for 
 resolving symbols.

 rname - the name of the routine
 nargs - the number of arguments it expects. 
******************************************************/
int R::addRegistrationRoutine(String *rname, int nargs) {
  if(!registrationTable) 
    registrationTable = NewHash();

  String *el = 
    NewStringf("{\"%s\", (DL_FUNC) &%s, %d}", rname, rname, nargs);
  
  Setattr(registrationTable, rname, el);

  return SWIG_OK;
}

/*****************************************************
 Write the registration information to an array and
 create the initialization routine for registering
 these.
******************************************************/
int R::outputRegistrationRoutines(File *out) {
  int i, n;
  if(!registrationTable) 
    return(0);
  if(inCPlusMode) 
    Printf(out, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");

  Printf(out, "#include <R_ext/Rdynload.h>\n\n");
  if(inCPlusMode) 
    Printf(out, "#ifdef __cplusplus\n}\n#endif\n\n");

  Printf(out, "SWIGINTERN R_CallMethodDef CallEntries[] = {\n");
    
  List *keys = Keys(registrationTable);
  n = Len(keys);
  for(i = 0; i < n; i++)
    Printf(out, "   %s,\n", Getattr(registrationTable, Getitem(keys, i)));

  Printf(out, "   {NULL, NULL, 0}\n};\n\n");

  if(!noInitializationCode) {
    if (inCPlusMode)
      Printv(out, "extern \"C\" ", NIL);
    Printf(out, "SWIGEXPORT void R_init_%s(DllInfo *dll) {\n", Rpackage);
    Printf(out, "%sR_registerRoutines(dll, NULL, CallEntries, NULL, NULL);\n", tab4);
    if(Len(s_init_routine)) {
      Printf(out, "\n%s\n", s_init_routine);
    }
    Printf(out, "}\n");
  }

  return n;
}



/****************************************************************************
  Process a struct, union or class declaration in the source code,
  or an anonymous typedef struct
 
*****************************************************************************/
//XXX What do we need to do here - 
// Define an S4 class to refer to this.

void R::registerClass(Node *n) {
  String *name = Getattr(n, "name");    
  String *kind = Getattr(n, "kind");    

  if (debugMode)
    Swig_print_node(n);
  String *sname = NewStringf("_p%s", SwigType_manglestr(name));
  if(!Getattr(SClassDefs, sname)) {
    Setattr(SClassDefs, sname, sname);
    String *base;

    if(Strcmp(kind, "class") == 0) {
      base = NewString("");
      List *l = Getattr(n, "bases");
      if(Len(l)) {
	Printf(base, "c(");
	for(int i = 0; i < Len(l); i++) {
	  registerClass(Getitem(l, i));
	  Printf(base, "'_p%s'%s", 
		 SwigType_manglestr(Getattr(Getitem(l, i), "name")), 
		 i < Len(l)-1 ? ", " : "");                
	}
	Printf(base, ")");
      } else {
	base = NewString("'C++Reference'");
      }
    } else 
      base = NewString("'ExternalReference'");

    Printf(s_classes, "setClass('%s', contains = %s)\n", sname, base);
    Delete(base);
  }
  
}

int R::classDeclaration(Node *n) {

  String *name = Getattr(n, "name");    
  String *kind = Getattr(n, "kind");    

  if (debugMode)
    Swig_print_node(n);
  registerClass(n);

    
  /* If we have a typedef union { ... } U, then we never get to see the typedef
     via a regular call to typedefHandler. Instead, */
  if(Getattr(n, "unnamed") && Getattr(n, "storage") && Strcmp(Getattr(n, "storage"), "typedef") == 0
     && Getattr(n, "tdname") && Strcmp(Getattr(n, "tdname"), name) == 0) {
    if (debugMode)
      Printf(stdout, "Typedef in the class declaration for %s\n", name);
    //        typedefHandler(n);
  }

  bool opaque = GetFlag(n, "feature:opaque") ? true : false;

  if(opaque)
    opaqueClassDeclaration = name;

  int status = Language::classDeclaration(n);

  opaqueClassDeclaration = NULL;


  // OutputArrayMethod(name, class_member_functions, sfile);        
  if (class_member_functions)
    OutputMemberReferenceMethod(name, 0, class_member_functions, sfile);
  if (class_member_set_functions)
    OutputMemberReferenceMethod(name, 1, class_member_set_functions, sfile);

  if(class_member_functions) {
    Delete(class_member_functions);
    class_member_functions = NULL;
  }
  if(class_member_set_functions) {
    Delete(class_member_set_functions);
    class_member_set_functions = NULL;
  }
  if (Getattr(n, "has_destructor")) {
    Printf(sfile, "setMethod('delete', '_p%s', function(obj) {delete%s(obj)})\n",
	   getRClassName(Getattr(n, "name")),
	   getRClassName(Getattr(n, "name")));

  }
  if(!opaque && !Strcmp(kind, "struct") && copyStruct) {

    String *def = 
      NewStringf("setClass(\"%s\",\n%srepresentation(\n", name, tab4);
    bool firstItem = true;

    for(Node *c = firstChild(n); c; ) {
      String *elName;
      String *tp;

      elName = Getattr(c, "name");
 
      String *elKind = Getattr(c, "kind");
      if (!Equal(elKind, "variable")) {
	c = nextSibling(c);
	continue;
      }
      if (!Len(elName)) {
	c = nextSibling(c);
	continue;
      }
#if 0
      tp = getRType(c);
#else
      tp = Swig_typemap_lookup("rtype", c, "", 0);
      if(!tp) {
	c = nextSibling(c);
	continue;
      }
      if (Strstr(tp, "R_class")) {
	c = nextSibling(c);
	continue;
      }
      if (Strcmp(tp, "character") &&
	  Strstr(Getattr(c, "decl"), "p.")) {
	c = nextSibling(c);
	continue;
      }

      if (!firstItem) {
	Printf(def, ",\n");
      } 
      //	    else 
      //XXX How can we tell if this is already done.
      //	      SwigType_push(elType, elDecl);
	    
	    
      // returns ""  tp = processType(elType, c, NULL);
      //	    Printf(stdout, "<classDeclaration> elType %p\n", elType);
      //	    tp = getRClassNameCopyStruct(Getattr(c, "type"), 1);
#endif
      String *elNameT = replaceInitialDash(elName);
      Printf(def, "%s%s = \"%s\"", tab8, elNameT, tp);
      firstItem = false;
      Delete(tp);
      Delete(elNameT);
      c = nextSibling(c);
    }
    Printf(def, "),\n%scontains = \"RSWIGStruct\")\n", tab8);
    Printf(s_classes, "%s\n\n# End class %s\n\n", def, name);

    generateCopyRoutines(n);

    Delete(def);
  }

  return status;
}



/***************************************************************
 Create the C routines that copy an S object of the class given
 by the given struct definition in Node *n to the C value
 and also the routine that goes from the C routine to an object
 of this S class.
****************************************************************/
/*XXX
  Clean up the toCRef - make certain the names are correct for the types, etc.
  in all cases.
*/

int R::generateCopyRoutines(Node *n) {
  Wrapper *copyToR = NewWrapper();
  Wrapper *copyToC = NewWrapper();
  
  String *name = Getattr(n, "name");
  String *tdname = Getattr(n, "tdname");
  String *kind = Getattr(n, "kind");
  String *type;

  if(Len(tdname)) {
    type = Copy(tdname);
  } else {
    type = NewStringf("%s %s", kind, name);
  }

  String *mangledName = SwigType_manglestr(name);

  if (debugMode)
    Printf(stdout, "generateCopyRoutines:  name = %s, %s\n", name, type);

  Printf(copyToR->def, "CopyToR%s = function(value, obj = new(\"%s\"))\n{\n", 
	 mangledName, name);
  Printf(copyToC->def, "CopyToC%s = function(value, obj)\n{\n", 
	 mangledName);

  Node *c = firstChild(n);

  for(; c; c = nextSibling(c)) {
    String *elName = Getattr(c, "name");
    if (!Len(elName)) {
      continue;
    }
    String *elKind = Getattr(c, "kind");
    if (!Equal(elKind, "variable")) {
      continue;
    }

    String *tp = Swig_typemap_lookup("rtype", c, "", 0);
    if(!tp) {
      continue;
    }
    if (Strstr(tp, "R_class")) {
      continue;
    }
    if (Strcmp(tp, "character") &&
	Strstr(Getattr(c, "decl"), "p.")) {
      continue;
    }


    /* The S functions to get and set the member value. */
    String *elNameT = replaceInitialDash(elName);
    Printf(copyToR->code, "obj@%s = value$%s;\n", elNameT, elNameT);
    Printf(copyToC->code, "obj$%s = value@%s;\n", elNameT, elNameT);
    Delete(elNameT);
  }
  Printf(copyToR->code, "obj;\n}\n\n");
  String *rclassName = getRClassNameCopyStruct(type, 0); // without the Ref.
  Printf(sfile, "# Start definition of copy functions & methods for %s\n", rclassName);  
  
  Wrapper_print(copyToR, sfile);
  Printf(copyToC->code, "obj\n}\n\n");
  Wrapper_print(copyToC, sfile);
  
  
  Printf(sfile, "# Start definition of copy methods for %s\n", rclassName);  
  Printf(sfile, "setMethod('copyToR', '_p_%s', CopyToR%s);\n", rclassName, 
	 mangledName);
  Printf(sfile, "setMethod('copyToC', '%s', CopyToC%s);\n\n", rclassName, 
	 mangledName);
  
  Printf(sfile, "# End definition of copy methods for %s\n", rclassName);  
  Printf(sfile, "# End definition of copy functions & methods for %s\n", rclassName);  
      
  String *m = NewStringf("%sCopyToR", name);
  addNamespaceMethod(m);
  char *tt = Char(m);  tt[Len(m)-1] = 'C';
  addNamespaceMethod(m);
  Delete(m);
  Delete(rclassName);
  Delete(mangledName);
  DelWrapper(copyToR);
  DelWrapper(copyToC);

  return SWIG_OK;
}



/*****
      Called when there is a typedef to be invoked. 

      XXX Needs to be enhanced or split to handle the case where we have a 
      typedef within a classDeclaration emission because the struct/union/etc.
      is anonymous.
******/
int R::typedefHandler(Node *n) {
  SwigType *tp = Getattr(n, "type");
  String *type = Getattr(n, "type");
  if (debugMode)
    Printf(stdout, "<typedefHandler> %s\n", Getattr(n, "name"));

  processType(tp, n);

  if(Strncmp(type, "struct ", 7) == 0) {
    String *name = Getattr(n, "name");
    char *trueName = Char(type);
    trueName += 7;
    if (debugMode)
      Printf(stdout, "<typedefHandler> Defining S class %s\n", trueName);
    Printf(s_classes, "setClass('_p%s', contains = 'ExternalReference')\n", 
	   SwigType_manglestr(name));
  }

  return Language::typedefHandler(n);
}



/*********************
  Called when processing a field in a "class", i.e. struct, union or
  actual class.  We set a state variable so that we can correctly
  interpret the resulting functionWrapper() call and understand that 
  it is for a field element.
**********************/
int R::membervariableHandler(Node *n) {
  SwigType *t = Getattr(n, "type");
  processType(t, n, NULL);
  processing_member_access_function = 1;
  member_name = Getattr(n,"sym:name");
  if (debugMode)
    Printf(stdout, "<membervariableHandler> name = %s, sym:name = %s\n", 
	   Getattr(n, "name"), member_name);

  int status(Language::membervariableHandler(n));

  if(!opaqueClassDeclaration && debugMode)
    Printf(stdout, "<membervariableHandler> %s %s\n", Getattr(n, "name"), Getattr(n, "type"));

  processing_member_access_function = 0;
  member_name = NULL;

  return status;
}


/*
  This doesn't seem to get used so leave it out for the moment.
*/
String * R::runtimeCode() {
  String *s = Swig_include_sys("rrun.swg");
  if (!s) {
    Printf(stdout, "*** Unable to open 'rrun.swg'\n");
    s = NewString("");
  }
  return s;
}


/**
   Called when SWIG wants to initialize this 
   We initialize anythin we want here.
   Most importantly, tell SWIG where to find the files (e.g. r.swg) for this module.
   Use Swig_mark_arg() to tell SWIG that it is understood and not to throw an error.
**/
void R::main(int argc, char *argv[]) {
  bool cppcast = true;
  init();
  Preprocessor_define("SWIGR 1", 0);
  SWIG_library_directory("r");
  SWIG_config_file("r.swg");
  debugMode = false;
  copyStruct = true;
  memoryProfile = false;
  aggressiveGc = false;
  inCPlusMode = false;
  outputNamespaceInfo = false;
  noInitializationCode = false;

  this->Argc = argc;
  this->Argv = argv;

  allow_overloading();// can we support this?    

  for(int i = 0; i < argc; i++) {
    if(strcmp(argv[i], "-package") == 0) {
      Swig_mark_arg(i);
      i++;
      Swig_mark_arg(i);
      Rpackage = argv[i];
    } else if(strcmp(argv[i], "-dll") == 0) {
      Swig_mark_arg(i);
      i++;
      Swig_mark_arg(i);
      DllName = argv[i];
    } else if(strcmp(argv[i], "-help") == 0) {
      showUsage();
    } else if(strcmp(argv[i], "-namespace") == 0) {
      outputNamespaceInfo = true;
      Swig_mark_arg(i);
    } else if(!strcmp(argv[i], "-no-init-code")) {
      noInitializationCode = true;
      Swig_mark_arg(i);
    } else if(!strcmp(argv[i], "-c++")) {
      inCPlusMode = true;
      Swig_mark_arg(i);
      Printf(s_classes, "setClass('C++Reference', contains = 'ExternalReference')\n");
    } else if(!strcmp(argv[i], "-debug")) {
      debugMode = true;
      Swig_mark_arg(i);
    }  else if (!strcmp(argv[i],"-cppcast")) {
      cppcast = true;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i],"-nocppcast")) {
      cppcast = false;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i],"-copystruct")) {
      copyStruct = true;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i], "-nocopystruct")) {
      copyStruct = false;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i], "-memoryprof")) {
      memoryProfile = true;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i], "-nomemoryprof")) {
      memoryProfile = false;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i], "-aggressivegc")) {
      aggressiveGc = true;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i], "-noaggressivegc")) {
      aggressiveGc = false;
      Swig_mark_arg(i);
    }

    if (cppcast) {
      Preprocessor_define((DOH *) "SWIG_CPLUSPLUS_CAST", 0);
    }

    if (debugMode) {
      Swig_typemap_search_debug_set();
      Swig_typemap_used_debug_set();
      Swig_typemap_register_debug_set();
      Swig_file_debug_set();
    }
    /// copyToR copyToC functions.

  }
}

/*
  Could make this work for String or File and then just store the resulting string
  rather than the collection of arguments and argc.
*/
int R::outputCommandLineArguments(File *out)
{
  if(Argc < 1 || !Argv || !Argv[0])
    return(-1);

  Printf(out, "\n##   Generated via the command line invocation:\n##\t");
  for(int i = 0; i < Argc ; i++) {
    Printf(out, " %s", Argv[i]);
  }
  Printf(out, "\n\n\n");

  return Argc;
}



/* How SWIG instantiates an object from this module. 
   See swigmain.cxx */
extern "C" 
Language *swig_r(void) {
  return new R();
}



/*************************************************************************************/

/*
  Needs to be reworked.
*/
String * R::processType(SwigType *t, Node *n, int *nargs) {
  //XXX Need to handle typedefs, e.g.
  //  a type which is a typedef  to a function pointer.

  SwigType *tmp = Getattr(n, "tdname");
  if (debugMode)
    Printf(stdout, "processType %s (tdname = %s)\n", Getattr(n, "name"), tmp);
  
  SwigType *td = t;
  if (expandTypedef(t) &&
      SwigType_istypedef(t)) {
    SwigType *resolved = 
      SwigType_typedef_resolve_all(t);
    if (expandTypedef(resolved)) {
      td = Copy(resolved);
    }
  }

  if(!td) {
    int count = 0;
    String *b = getRTypeName(t, &count);
    if(count && b && !Getattr(SClassDefs, b)) {
      if (debugMode)
	Printf(stdout, "<processType> Defining class %s\n",  b);

      Printf(s_classes, "setClass('%s', contains = 'ExternalReference')\n", b);       
      Setattr(SClassDefs, b, b);
    }
     
  }


  if(td)
    t = td;

  if(SwigType_isfunctionpointer(t)) {
    if (debugMode)
      Printf(stdout, 
	     "<processType> Defining pointer handler %s\n",  t);
       
    String *tmp = createFunctionPointerHandler(t, n, nargs);
    return tmp;
  }

#if 0
  SwigType_isfunction(t) && SwigType_ispointer(t)
#endif

    return NULL;
}









/*************************************************************************************/





