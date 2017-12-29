/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * swigmod.h
 *
 * Main header file for SWIG modules.
 * ----------------------------------------------------------------------------- */

#ifndef SWIG_SWIGMOD_H_
#define SWIG_SWIGMOD_H_

#include "swig.h"
#include "preprocessor.h"
#include "swigwarn.h"

#define NOT_VIRTUAL     0
#define PLAIN_VIRTUAL   1
#define PURE_VIRTUAL    2

extern String *input_file;
extern int line_number;
extern int start_line;
extern int CPlusPlus;		// C++ mode
extern int Extend;		// Extend mode
extern int Verbose;
extern int IsVirtual;
extern int ImportMode;
extern int NoExcept;		// -no_except option
extern int Abstract;		// abstract base class
extern int SmartPointer;	// smart pointer methods being emitted
extern int SwigRuntime;

/* Overload "argc" and "argv" */
extern String *argv_template_string;
extern String *argc_template_string;

/* Miscellaneous stuff */

#define  tab2   "  "
#define  tab4   "    "
#define  tab8   "        "

class Dispatcher {
public:

  Dispatcher ():cplus_mode(PUBLIC) {
  }
  virtual ~ Dispatcher () {
  }

  virtual int emit_one(Node *n);
  virtual int emit_children(Node *n);
  virtual int defaultHandler(Node *n);

  /* Top of the parse tree */
  virtual int top(Node *n) = 0;

  /* SWIG directives */

  virtual int applyDirective(Node *n);
  virtual int clearDirective(Node *n);
  virtual int constantDirective(Node *n);
  virtual int extendDirective(Node *n);
  virtual int fragmentDirective(Node *n);
  virtual int importDirective(Node *n);
  virtual int includeDirective(Node *n);
  virtual int insertDirective(Node *n);
  virtual int moduleDirective(Node *n);
  virtual int nativeDirective(Node *n);
  virtual int pragmaDirective(Node *n);
  virtual int typemapDirective(Node *n);
  virtual int typemapitemDirective(Node *n);
  virtual int typemapcopyDirective(Node *n);
  virtual int typesDirective(Node *n);

  /* C/C++ parsing */

  virtual int cDeclaration(Node *n);
  virtual int externDeclaration(Node *n);
  virtual int enumDeclaration(Node *n);
  virtual int enumvalueDeclaration(Node *n);
  virtual int enumforwardDeclaration(Node *n);
  virtual int classDeclaration(Node *n);
  virtual int classforwardDeclaration(Node *n);
  virtual int constructorDeclaration(Node *n);
  virtual int destructorDeclaration(Node *n);
  virtual int accessDeclaration(Node *n);
  virtual int usingDeclaration(Node *n);
  virtual int namespaceDeclaration(Node *n);
  virtual int templateDeclaration(Node *n);
  virtual int lambdaDeclaration(Node *n);

  enum AccessMode { PUBLIC, PRIVATE, PROTECTED };

protected:
  AccessMode cplus_mode;
};

/* ----------------------------------------------------------------------------
 * class language:
 *
 * This class defines the functions that need to be supported by the
 * scripting language being used.    The translator calls these virtual
 * functions to output different types of code for different languages.
 * ------------------------------------------------------------------------- */

class Language:public Dispatcher {
public:
  Language();
  virtual ~Language();
  virtual int emit_one(Node *n);

  String *directorClassName(Node *n);

  /* Parse command line options */

  virtual void main(int argc, char *argv[]);

  /* Top of the parse tree */

  virtual int top(Node *n);

  /* SWIG directives */


  virtual int applyDirective(Node *n);
  virtual int clearDirective(Node *n);
  virtual int constantDirective(Node *n);
  virtual int extendDirective(Node *n);
  virtual int fragmentDirective(Node *n);
  virtual int importDirective(Node *n);
  virtual int includeDirective(Node *n);
  virtual int insertDirective(Node *n);
  virtual int moduleDirective(Node *n);
  virtual int nativeDirective(Node *n);
  virtual int pragmaDirective(Node *n);
  virtual int typemapDirective(Node *n);
  virtual int typemapcopyDirective(Node *n);
  virtual int typesDirective(Node *n);

  /* C/C++ parsing */

  virtual int cDeclaration(Node *n);
  virtual int externDeclaration(Node *n);
  virtual int enumDeclaration(Node *n);
  virtual int enumvalueDeclaration(Node *n);
  virtual int enumforwardDeclaration(Node *n);
  virtual int classDeclaration(Node *n);
  virtual int classforwardDeclaration(Node *n);
  virtual int constructorDeclaration(Node *n);
  virtual int destructorDeclaration(Node *n);
  virtual int accessDeclaration(Node *n);
  virtual int namespaceDeclaration(Node *n);
  virtual int usingDeclaration(Node *n);

  /* Function handlers */

  virtual int functionHandler(Node *n);
  virtual int globalfunctionHandler(Node *n);
  virtual int memberfunctionHandler(Node *n);
  virtual int staticmemberfunctionHandler(Node *n);
  virtual int callbackfunctionHandler(Node *n);

  /* Variable handlers */

  virtual int variableHandler(Node *n);
  virtual int globalvariableHandler(Node *n);
  virtual int membervariableHandler(Node *n);
  virtual int staticmembervariableHandler(Node *n);

  /* C++ handlers */

  virtual int memberconstantHandler(Node *n);
  virtual int constructorHandler(Node *n);
  virtual int copyconstructorHandler(Node *n);
  virtual int destructorHandler(Node *n);
  virtual int classHandler(Node *n);

  /* Miscellaneous */

  virtual int typedefHandler(Node *n);

  /* Low-level code generation */

  virtual int constantWrapper(Node *n);
  virtual int variableWrapper(Node *n);
  virtual int functionWrapper(Node *n);
  virtual int nativeWrapper(Node *n);

  /* C++ director class generation */
  virtual int classDirector(Node *n);
  virtual int classDirectorInit(Node *n);
  virtual int classDirectorEnd(Node *n);
  virtual int unrollVirtualMethods(Node *n, Node *parent, List *vm, int default_director, int &virtual_destructor, int protectedbase = 0);
  virtual int classDirectorConstructor(Node *n);
  virtual int classDirectorDefaultConstructor(Node *n);
  virtual int classDirectorMethod(Node *n, Node *parent, String *super);
  virtual int classDirectorConstructors(Node *n);
  virtual int classDirectorDestructor(Node *n);
  virtual int classDirectorMethods(Node *n);
  virtual int classDirectorDisown(Node *n);

  /* Miscellaneous */
  virtual int validIdentifier(String *s);	/* valid identifier? */
  virtual int addSymbol(const String *s, const Node *n, const_String_or_char_ptr scope = "");	/* Add symbol        */
  virtual int addInterfaceSymbol(const String *interface_name, Node *n, const_String_or_char_ptr scope = "");
  virtual void dumpSymbols();
  virtual Node *symbolLookup(const String *s, const_String_or_char_ptr scope = ""); /* Symbol lookup */
  virtual Hash* symbolAddScope(const_String_or_char_ptr scope);
  virtual Hash* symbolScopeLookup(const_String_or_char_ptr scope);
  virtual Hash* symbolScopePseudoSymbolLookup(const_String_or_char_ptr scope);
  virtual Node *classLookup(const SwigType *s) const; /* Class lookup      */
  virtual Node *enumLookup(SwigType *s);	/* Enum lookup       */
  virtual int abstractClassTest(Node *n);	/* Is class really abstract? */
  virtual int is_assignable(Node *n);	/* Is variable assignable? */
  virtual String *runtimeCode();	/* returns the language specific runtime code */
  virtual String *defaultExternalRuntimeFilename();	/* the default filename for the external runtime */
  virtual void replaceSpecialVariables(String *method, String *tm, Parm *parm); /* Language specific special variable substitutions for $typemap() */

  /* Runtime is C++ based, so extern "C" header section */
  void enable_cplus_runtime_mode();

  /* Returns the cplus_runtime mode */
  int cplus_runtime_mode();

  /* Allow director related code generation */
  void allow_directors(int val = 1);

  /* Return true if directors are enabled */
  int directorsEnabled() const;

  /* Allow director protected members related code generation */
  void allow_dirprot(int val = 1);

  /* Allow all protected members code generation (for directors) */
  void allow_allprotected(int val = 0);

  /* Returns the dirprot mode */
  int dirprot_mode() const;

  /* Check if the non public constructor is  needed (for directors) */
  int need_nonpublic_ctor(Node *n);

  /* Check if the non public member is  needed (for directors) */
  int need_nonpublic_member(Node *n);

  /* Set none comparison string */
  void setSubclassInstanceCheck(String *s);

  /* Set overload variable templates argc and argv */
  void setOverloadResolutionTemplates(String *argc, String *argv);

  /* Language instance is a singleton - get instance */
  static Language* instance();

protected:
  /* Allow multiple-input typemaps */
  void allow_multiple_input(int val = 1);

  /* Allow overloaded functions */
  void allow_overloading(int val = 1);

  /* Wrapping class query */
  int is_wrapping_class() const;

  /* Return the node for the current class */
  Node *getCurrentClass() const;

  /* Return C++ mode */
  int getCPlusMode() const;

  /* Return the namespace for the class/enum - the nspace feature */
  String *getNSpace() const;

  /* Return the real name of the current class */
  String *getClassName() const;

  /* Return the classes hash */
  Hash *getClassHash() const;

  /* Return the current class prefix */
  String *getClassPrefix() const;

  /* Return the current enum class prefix */
  String *getEnumClassPrefix() const;

  /* Fully qualified type name to use */
  String *getClassType() const;

  /* Return true if the current method is part of a smart-pointer */
  int is_smart_pointer() const;

  /* Return the name to use for the given parameter. */
  virtual String *makeParameterName(Node *n, Parm *p, int arg_num, bool setter = false) const;

  /* Some language modules require additional wrappers for virtual methods not declared in sub-classes */
  virtual bool extraDirectorProtectedCPPMethodsRequired() const;

public:
  enum NestedClassSupport {
    NCS_None, // Target language does not have an equivalent to nested classes
    NCS_Full, // Target language does have an equivalent to nested classes and is fully implemented
    NCS_Unknown // Target language may or may not have an equivalent to nested classes. If it does, it has not been implemented yet.
  };
  /* Does target language support nested classes? Default is NCS_Unknown. 
    If NCS_Unknown is returned, then the nested classes will be ignored unless 
    %feature "flatnested" is applied to them, in which case they will appear in global space.
    If the target language does not support the notion of class
    nesting, the language module should return NCS_None from this function, and 
    the nested classes will be moved to the global scope (like implicit global %feature "flatnested").
  */
  virtual NestedClassSupport nestedClassesSupport() const;

  /* Returns true if the target language supports key word arguments (kwargs) */
  virtual bool kwargsSupport() const;

protected:
  /* Identifies if a protected members that are generated when the allprotected option is used.
     This does not include protected virtual methods as they are turned on with the dirprot option. */
  bool isNonVirtualProtectedAccess(Node *n) const;

  /* Identify if a wrapped global or member variable n should use the naturalvar feature */
  int use_naturalvar_mode(Node *n) const;

  /* Director subclass comparison test */
  String *none_comparison;

  /* Director constructor "template" code */
  String *director_ctor_code;

  /* Director 'protected' constructor "template" code */
  String *director_prot_ctor_code;

  /* Director allows multiple inheritance */
  int director_multiple_inheritance;

  /* Director language module */
  int director_language;

private:
  Hash *symtabs; /* symbol tables */
  Hash *classtypes;
  Hash *enumtypes;
  int overloading;
  int multiinput;
  int cplus_runtime;
  int directors;
  static Language *this_;
};

int SWIG_main(int, char **, Language *);
void emit_parameter_variables(ParmList *l, Wrapper *f);
void emit_return_variable(Node *n, SwigType *rt, Wrapper *f);
void SWIG_exit(int);		/* use EXIT_{SUCCESS,FAILURE} */
void SWIG_config_file(const_String_or_char_ptr );
const String *SWIG_output_directory();
void SWIG_config_cppext(const char *ext);
void Swig_print_xml(Node *obj, String *filename);

/* get the list of generated files */
List *SWIG_output_files();

void SWIG_library_directory(const char *);
int emit_num_arguments(ParmList *);
int emit_num_required(ParmList *);
int emit_isvarargs(ParmList *);
void emit_attach_parmmaps(ParmList *, Wrapper *f);
void emit_mark_varargs(ParmList *l);
String *emit_action(Node *n);
int emit_action_code(Node *n, String *wrappercode, String *action);
void Swig_overload_check(Node *n);
String *Swig_overload_dispatch(Node *n, const_String_or_char_ptr fmt, int *);
String *Swig_overload_dispatch_cast(Node *n, const_String_or_char_ptr fmt, int *);
String *Swig_overload_dispatch_fast(Node *n, const_String_or_char_ptr fmt, int *);
List *Swig_overload_rank(Node *n, bool script_lang_wrapping);
SwigType *cplus_value_type(SwigType *t);

/* directors.cxx start */
String *Swig_csuperclass_call(String *base, String *method, ParmList *l);
String *Swig_class_declaration(Node *n, String *name);
String *Swig_class_name(Node *n);
String *Swig_method_call(const_String_or_char_ptr name, ParmList *parms);
String *Swig_method_decl(SwigType *rtype, SwigType *decl, const_String_or_char_ptr id, List *args, int strip, int values);
String *Swig_director_declaration(Node *n);
void Swig_director_emit_dynamic_cast(Node *n, Wrapper *f);
void Swig_director_parms_fixup(ParmList *parms);
/* directors.cxx end */

extern "C" {
  void SWIG_typemap_lang(const char *);
  typedef Language *(*ModuleFactory) (void);
} 

void Swig_register_module(const char *name, ModuleFactory fac);
ModuleFactory Swig_find_module(const char *name);

/* Utilities */

int is_public(Node *n);
int is_private(Node *n);
int is_protected(Node *n);
int is_member_director(Node *parentnode, Node *member);
int is_member_director(Node *member);
int is_non_virtual_protected_access(Node *n); /* Check if the non-virtual protected members are required (for directors) */

void Wrapper_virtual_elimination_mode_set(int);
void Wrapper_fast_dispatch_mode_set(int);
void Wrapper_cast_dispatch_mode_set(int);
void Wrapper_naturalvar_mode_set(int);

void clean_overloaded(Node *n);

extern "C" {
  const char *Swig_to_string(DOH *object, int count = -1);
  const char *Swig_to_string_with_location(DOH *object, int count = -1);
  void Swig_print(DOH *object, int count = -1);
  void Swig_print_with_location(DOH *object, int count = -1);
}

/* Contracts */
void Swig_contracts(Node *n);
void Swig_contract_mode_set(int flag);
int Swig_contract_mode_get();

/* Browser */
void Swig_browser(Node *n, int);
void Swig_default_allocators(Node *n);
void Swig_process_types(Node *n);

/* Nested classes */
void Swig_nested_process_classes(Node *n);
void Swig_nested_name_unnamed_c_structs(Node *n);

/* Interface feature */
void Swig_interface_feature_enable();
void Swig_interface_propagate_methods(Node *n);

/* Miscellaneous */
template <class T> class save_value {
  T _value;
  T& _value_ptr;
  save_value(const save_value&);
  save_value& operator=(const save_value&);

public:
  save_value(T& value) : _value(value), _value_ptr(value) {}
  save_value(T& value, T new_val) : _value(value), _value_ptr(value) { value = new_val; }
  ~save_value() { _value_ptr = _value; }
};

#endif
