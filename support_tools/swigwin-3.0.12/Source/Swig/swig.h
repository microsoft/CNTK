/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * swig.h
 *
 * Header file for the SWIG core.
 * ----------------------------------------------------------------------------- */

#ifndef SWIGCORE_H_
#define SWIGCORE_H_

#ifndef MACSWIG
#include "swigconfig.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "doh.h"

/* Status codes */

#define SWIG_OK         1
#define SWIG_ERROR      0
#define SWIG_NOWRAP     0

/* Global macros */
#define NSPACE_SEPARATOR "." /* Namespace separator for the nspace feature - this should be changed to a target language configurable variable */
#define NSPACE_TODO 0 /* Languages that still need to implement and test the nspace feature use this */

/* Short names for common data types */

  typedef DOH String;
  typedef DOH Hash;
  typedef DOH List;
  typedef DOH String_or_char;
  typedef DOH File;
  typedef DOH Parm;
  typedef DOH ParmList;
  typedef DOH Node;
  typedef DOH Symtab;
  typedef DOH Typetab;
  typedef DOH SwigType;

/* --- Legacy DataType interface.  These type codes are provided solely 
       for backwards compatibility with older modules --- */

/* --- The ordering of type values is used to determine type-promotion 
       in the parser.  Do not change */

/* Numeric types */

#define   T_BOOL       1
#define   T_SCHAR      2
#define   T_UCHAR      3
#define   T_SHORT      4
#define   T_USHORT     5
#define   T_ENUM       6
#define   T_INT        7
#define   T_UINT       8
#define   T_LONG       9
#define   T_ULONG      10
#define   T_LONGLONG   11
#define   T_ULONGLONG  12
#define   T_FLOAT      20
#define   T_DOUBLE     21
#define   T_LONGDOUBLE 22
#define   T_FLTCPLX    23
#define   T_DBLCPLX    24
#define   T_NUMERIC    25
#define   T_AUTO       26

#define   T_COMPLEX    T_DBLCPLX

/* non-numeric */

#define   T_CHAR       29
#define   T_WCHAR      30
#define   T_USER       31
#define   T_VOID       32
#define   T_STRING     33
#define   T_POINTER    34
#define   T_REFERENCE  35
#define   T_ARRAY      36
#define   T_FUNCTION   37
#define   T_MPOINTER   38
#define   T_VARARGS    39
#define   T_RVALUE_REFERENCE  40
#define   T_WSTRING    41

#define   T_SYMBOL     98
#define   T_ERROR      99



/* --- File interface --- */

#include "swigfile.h"

/* --- Command line parsing --- */

#include "swigopt.h"

/* --- Scanner Interface --- */

#include "swigscan.h"

/* --- Functions for manipulating the string-based type encoding --- */

  extern SwigType *NewSwigType(int typecode);
  extern SwigType *SwigType_del_element(SwigType *t);
  extern SwigType *SwigType_add_pointer(SwigType *t);
  extern SwigType *SwigType_add_memberpointer(SwigType *t, const_String_or_char_ptr qual);
  extern SwigType *SwigType_del_memberpointer(SwigType *t);
  extern SwigType *SwigType_del_pointer(SwigType *t);
  extern SwigType *SwigType_add_array(SwigType *t, const_String_or_char_ptr size);
  extern SwigType *SwigType_del_array(SwigType *t);
  extern SwigType *SwigType_pop_arrays(SwigType *t);
  extern SwigType *SwigType_add_reference(SwigType *t);
  extern SwigType *SwigType_del_reference(SwigType *t);
  extern SwigType *SwigType_add_rvalue_reference(SwigType *t);
  extern SwigType *SwigType_del_rvalue_reference(SwigType *t);
  extern SwigType *SwigType_add_qualifier(SwigType *t, const_String_or_char_ptr qual);
  extern SwigType *SwigType_del_qualifier(SwigType *t);
  extern SwigType *SwigType_add_function(SwigType *t, ParmList *parms);
  extern SwigType *SwigType_add_template(SwigType *t, ParmList *parms);
  extern SwigType *SwigType_pop_function(SwigType *t);
  extern ParmList *SwigType_function_parms(const SwigType *t, Node *file_line_node);
  extern List *SwigType_split(const SwigType *t);
  extern String *SwigType_pop(SwigType *t);
  extern void SwigType_push(SwigType *t, String *s);
  extern List *SwigType_parmlist(const SwigType *p);
  extern String *SwigType_parm(const SwigType *p);
  extern String *SwigType_str(const SwigType *s, const_String_or_char_ptr id);
  extern String *SwigType_lstr(const SwigType *s, const_String_or_char_ptr id);
  extern String *SwigType_rcaststr(const SwigType *s, const_String_or_char_ptr id);
  extern String *SwigType_lcaststr(const SwigType *s, const_String_or_char_ptr id);
  extern String *SwigType_manglestr(const SwigType *t);
  extern SwigType *SwigType_ltype(const SwigType *t);
  extern int SwigType_ispointer(const SwigType *t);
  extern int SwigType_ispointer_return(const SwigType *t);
  extern int SwigType_isfunctionpointer(const SwigType *t);
  extern int SwigType_ismemberpointer(const SwigType *t);
  extern int SwigType_isreference(const SwigType *t);
  extern int SwigType_isreference_return(const SwigType *t);
  extern int SwigType_isrvalue_reference(const SwigType *t);
  extern int SwigType_isarray(const SwigType *t);
  extern int SwigType_prefix_is_simple_1D_array(const SwigType *t);
  extern int SwigType_isfunction(const SwigType *t);
  extern int SwigType_isqualifier(const SwigType *t);
  extern int SwigType_isconst(const SwigType *t);
  extern int SwigType_issimple(const SwigType *t);
  extern int SwigType_ismutable(const SwigType *t);
  extern int SwigType_isvarargs(const SwigType *t);
  extern int SwigType_istemplate(const SwigType *t);
  extern int SwigType_isenum(const SwigType *t);
  extern int SwigType_check_decl(const SwigType *t, const_String_or_char_ptr decl);
  extern SwigType *SwigType_strip_qualifiers(const SwigType *t);
  extern SwigType *SwigType_strip_single_qualifier(const SwigType *t);
  extern SwigType *SwigType_functionpointer_decompose(SwigType *t);
  extern String *SwigType_base(const SwigType *t);
  extern String *SwigType_namestr(const SwigType *t);
  extern String *SwigType_templateprefix(const SwigType *t);
  extern String *SwigType_templatesuffix(const SwigType *t);
  extern String *SwigType_istemplate_templateprefix(const SwigType *t);
  extern String *SwigType_istemplate_only_templateprefix(const SwigType *t);
  extern String *SwigType_templateargs(const SwigType *t);
  extern String *SwigType_prefix(const SwigType *t);
  extern int SwigType_array_ndim(const SwigType *t);
  extern String *SwigType_array_getdim(const SwigType *t, int n);
  extern void SwigType_array_setdim(SwigType *t, int n, const_String_or_char_ptr rep);
  extern SwigType *SwigType_array_type(const SwigType *t);
  extern SwigType *SwigType_default_create(const SwigType *ty);
  extern SwigType *SwigType_default_deduce(const SwigType *t);
  extern void SwigType_typename_replace(SwigType *t, String *pat, String *rep);
  extern SwigType *SwigType_remove_global_scope_prefix(const SwigType *t);
  extern SwigType *SwigType_alttype(const SwigType *t, int ltmap);

/* --- Type-system managment --- */
  extern void SwigType_typesystem_init(void);
  extern int SwigType_typedef(const SwigType *type, const_String_or_char_ptr name);
  extern int SwigType_typedef_class(const_String_or_char_ptr name);
  extern int SwigType_typedef_using(const_String_or_char_ptr qname);
  extern void SwigType_inherit(String *subclass, String *baseclass, String *cast, String *conversioncode);
  extern int SwigType_issubtype(const SwigType *subtype, const SwigType *basetype);
  extern void SwigType_scope_alias(String *aliasname, Typetab *t);
  extern void SwigType_using_scope(Typetab *t);
  extern void SwigType_new_scope(const_String_or_char_ptr name);
  extern void SwigType_inherit_scope(Typetab *scope);
  extern Typetab *SwigType_pop_scope(void);
  extern Typetab *SwigType_set_scope(Typetab *h);
  extern void SwigType_print_scope(void);
  extern SwigType *SwigType_typedef_resolve(const SwigType *t);
  extern SwigType *SwigType_typedef_resolve_all(const SwigType *t);
  extern SwigType *SwigType_typedef_qualified(const SwigType *t);
  extern int SwigType_istypedef(const SwigType *t);
  extern int SwigType_isclass(const SwigType *t);
  extern void SwigType_attach_symtab(Symtab *syms);
  extern void SwigType_remember(const SwigType *t);
  extern void SwigType_remember_clientdata(const SwigType *t, const_String_or_char_ptr clientdata);
  extern void SwigType_remember_mangleddata(String *mangled, const_String_or_char_ptr clientdata);
  extern void (*SwigType_remember_trace(void (*tf) (const SwigType *, String *, String *))) (const SwigType *, String *, String *);
  extern void SwigType_emit_type_table(File *f_headers, File *f_table);
  extern int SwigType_type(const SwigType *t);

/* --- Symbol table module --- */

  extern void Swig_symbol_print_tables(Symtab *symtab);
  extern void Swig_symbol_print_tables_summary(void);
  extern void Swig_symbol_print_symbols(void);
  extern void Swig_symbol_print_csymbols(void);
  extern void Swig_symbol_init(void);
  extern void Swig_symbol_setscopename(const_String_or_char_ptr name);
  extern String *Swig_symbol_getscopename(void);
  extern String *Swig_symbol_qualifiedscopename(Symtab *symtab);
  extern String *Swig_symbol_qualified_language_scopename(Symtab *symtab);
  extern Symtab *Swig_symbol_newscope(void);
  extern Symtab *Swig_symbol_setscope(Symtab *);
  extern Symtab *Swig_symbol_getscope(const_String_or_char_ptr symname);
  extern Symtab *Swig_symbol_global_scope(void);
  extern Symtab *Swig_symbol_current(void);
  extern Symtab *Swig_symbol_popscope(void);
  extern Node *Swig_symbol_add(const_String_or_char_ptr symname, Node *node);
  extern void Swig_symbol_cadd(const_String_or_char_ptr symname, Node *node);
  extern Node *Swig_symbol_clookup(const_String_or_char_ptr symname, Symtab *tab);
  extern Node *Swig_symbol_clookup_check(const_String_or_char_ptr symname, Symtab *tab, int (*check) (Node *));
  extern Node *Swig_symbol_clookup_no_inherit(const_String_or_char_ptr name, Symtab *n);
  extern Symtab *Swig_symbol_cscope(const_String_or_char_ptr symname, Symtab *tab);
  extern Node *Swig_symbol_clookup_local(const_String_or_char_ptr symname, Symtab *tab);
  extern Node *Swig_symbol_clookup_local_check(const_String_or_char_ptr symname, Symtab *tab, int (*check) (Node *));
  extern String *Swig_symbol_qualified(Node *node);
  extern Node *Swig_symbol_isoverloaded(Node *node);
  extern void Swig_symbol_remove(Node *node);
  extern void Swig_symbol_alias(const_String_or_char_ptr aliasname, Symtab *tab);
  extern void Swig_symbol_inherit(Symtab *tab);
  extern SwigType *Swig_symbol_type_qualify(const SwigType *ty, Symtab *tab);
  extern String *Swig_symbol_string_qualify(String *s, Symtab *tab);
  extern SwigType *Swig_symbol_typedef_reduce(const SwigType *ty, Symtab *tab);

  extern ParmList *Swig_symbol_template_defargs(Parm *parms, Parm *targs, Symtab *tscope, Symtab *tsdecl);
  extern SwigType *Swig_symbol_template_deftype(const SwigType *type, Symtab *tscope);
  extern SwigType *Swig_symbol_template_param_eval(const SwigType *p, Symtab *symtab);

/* --- Parameters and Parameter Lists --- */

#include "swigparm.h"

extern String    *ParmList_errorstr(ParmList *);
extern int        ParmList_is_compactdefargs(ParmList *p);

/* --- Parse tree support --- */

#include "swigtree.h"

/* -- Wrapper function Object */

#include "swigwrap.h"

/* --- Naming functions --- */

  extern void Swig_name_register(const_String_or_char_ptr method, const_String_or_char_ptr format);
  extern void Swig_name_unregister(const_String_or_char_ptr method);
  extern String *Swig_name_mangle(const_String_or_char_ptr s);
  extern String *Swig_name_wrapper(const_String_or_char_ptr fname);
  extern String *Swig_name_member(const_String_or_char_ptr nspace, const_String_or_char_ptr classname, const_String_or_char_ptr membername);
  extern String *Swig_name_get(const_String_or_char_ptr nspace, const_String_or_char_ptr vname);
  extern String *Swig_name_set(const_String_or_char_ptr nspace, const_String_or_char_ptr vname);
  extern String *Swig_name_construct(const_String_or_char_ptr nspace, const_String_or_char_ptr classname);
  extern String *Swig_name_copyconstructor(const_String_or_char_ptr nspace, const_String_or_char_ptr classname);
  extern String *Swig_name_destroy(const_String_or_char_ptr nspace, const_String_or_char_ptr classname);
  extern String *Swig_name_disown(const_String_or_char_ptr nspace, const_String_or_char_ptr classname);

  extern void Swig_naming_init(void);
  extern void Swig_name_namewarn_add(String *prefix, String *name, SwigType *decl, Hash *namewrn);
  extern void Swig_name_rename_add(String *prefix, String *name, SwigType *decl, Hash *namewrn, ParmList *declaratorparms);
  extern void Swig_name_inherit(String *base, String *derived);
  extern List *Swig_make_inherit_list(String *clsname, List *names, String *Namespaceprefix);
  extern void Swig_inherit_base_symbols(List *bases);
  extern int Swig_need_protected(Node *n);
  extern int Swig_need_redefined_warn(Node *a, Node *b, int InClass);

  extern String *Swig_name_make(Node *n, String *prefix, const_String_or_char_ptr cname, SwigType *decl, String *oldname);
  extern String *Swig_name_warning(Node *n, String *prefix, String *name, SwigType *decl);
  extern String *Swig_name_str(Node *n);
  extern String *Swig_name_decl(Node *n);
  extern String *Swig_name_fulldecl(Node *n);

/* --- parameterized rename functions --- */

  extern void Swig_name_object_set(Hash *namehash, String *name, SwigType *decl, DOH *object);
  extern DOH *Swig_name_object_get(Hash *namehash, String *prefix, String *name, SwigType *decl);
  extern void Swig_name_object_inherit(Hash *namehash, String *base, String *derived);
  extern void Swig_features_get(Hash *features, String *prefix, String *name, SwigType *decl, Node *n);
  extern void Swig_feature_set(Hash *features, const_String_or_char_ptr name, SwigType *decl, const_String_or_char_ptr featurename, const_String_or_char_ptr value, Hash *featureattribs);

/* --- Misc --- */
  extern char *Swig_copy_string(const char *c);
  extern void Swig_set_fakeversion(const char *version);
  extern const char *Swig_package_version(void);
  extern void Swig_banner(File *f);
  extern void Swig_banner_target_lang(File *f, const_String_or_char_ptr commentchar);
  extern String *Swig_strip_c_comments(const String *s);
  extern String *Swig_new_subdirectory(String *basedirectory, String *subdirectory);
  extern void Swig_filename_correct(String *filename);
  extern String *Swig_filename_escape(String *filename);
  extern void Swig_filename_unescape(String *filename);
  extern int Swig_storage_isextern(Node *n);
  extern int Swig_storage_isexternc(Node *n);
  extern int Swig_storage_isstatic_custom(Node *n, const_String_or_char_ptr storage);
  extern int Swig_storage_isstatic(Node *n);
  extern String *Swig_string_escape(String *s);
  extern String *Swig_string_mangle(const String *s);
  extern void Swig_scopename_split(const String *s, String **prefix, String **last);
  extern String *Swig_scopename_prefix(const String *s);
  extern String *Swig_scopename_last(const String *s);
  extern String *Swig_scopename_first(const String *s);
  extern String *Swig_scopename_suffix(const String *s);
  extern int Swig_scopename_check(const String *s);
  extern String *Swig_string_lower(String *s);
  extern String *Swig_string_upper(String *s);
  extern String *Swig_string_title(String *s);
  extern void Swig_offset_string(String *s, int number);
  extern String *Swig_pcre_version(void);
  extern void Swig_init(void);
  extern int Swig_value_wrapper_mode(int mode);

  typedef enum { EMF_STANDARD, EMF_MICROSOFT } ErrorMessageFormat;

  extern void Swig_warning(int num, const_String_or_char_ptr filename, int line, const char *fmt, ...);
  extern void Swig_error(const_String_or_char_ptr filename, int line, const char *fmt, ...);
  extern int Swig_error_count(void);
  extern void Swig_error_silent(int s);
  extern void Swig_warnfilter(const_String_or_char_ptr wlist, int val);
  extern void Swig_warnall(void);
  extern int Swig_warn_count(void);
  extern void Swig_error_msg_format(ErrorMessageFormat format);
  extern void Swig_diagnostic(const_String_or_char_ptr filename, int line, const char *fmt, ...);
  extern String *Swig_stringify_with_location(DOH *object);

/* --- C Wrappers --- */
  extern void Swig_cresult_name_set(const char *new_name);
  extern const char *Swig_cresult_name(void);
  extern String *Swig_cparm_name(Parm *p, int i);
  extern String *Swig_wrapped_var_type(SwigType *t, int varcref);
  extern int Swig_cargs(Wrapper *w, ParmList *l);
  extern String *Swig_cresult(SwigType *t, const_String_or_char_ptr name, const_String_or_char_ptr decl);

  extern String *Swig_cfunction_call(const_String_or_char_ptr name, ParmList *parms);
  extern String *Swig_cconstructor_call(const_String_or_char_ptr name);
  extern String *Swig_cppconstructor_call(const_String_or_char_ptr name, ParmList *parms);
  extern String *Swig_unref_call(Node *n);
  extern String *Swig_ref_call(Node *n, const String *lname);
  extern String *Swig_cdestructor_call(Node *n);
  extern String *Swig_cppdestructor_call(Node *n);
  extern String *Swig_cmemberset_call(const_String_or_char_ptr name, SwigType *type, String *self, int varcref);
  extern String *Swig_cmemberget_call(const_String_or_char_ptr name, SwigType *t, String *self, int varcref);

  extern int Swig_add_extension_code(Node *n, const String *function_name, ParmList *parms, SwigType *return_type, const String *code, int cplusplus, const String *self);
  extern void Swig_replace_special_variables(Node *n, Node *parentnode, String *code);

/* --- Transformations --- */

  extern int Swig_MethodToFunction(Node *n, const_String_or_char_ptr nspace, String *classname, int flags, SwigType *director_type, int is_director);
  extern int Swig_ConstructorToFunction(Node *n, const_String_or_char_ptr nspace, String *classname, String *none_comparison, String *director_ctor, int cplus, int flags, String *directorname);
  extern int Swig_DestructorToFunction(Node *n, const_String_or_char_ptr nspace, String *classname, int cplus, int flags);
  extern int Swig_MembersetToFunction(Node *n, String *classname, int flags);
  extern int Swig_MembergetToFunction(Node *n, String *classname, int flags);
  extern int Swig_VargetToFunction(Node *n, int flags);
  extern int Swig_VarsetToFunction(Node *n, int flags);

#define  CWRAP_EXTEND                 0x01
#define  CWRAP_SMART_POINTER          0x02
#define  CWRAP_NATURAL_VAR            0x04
#define  CWRAP_DIRECTOR_ONE_CALL      0x08
#define  CWRAP_DIRECTOR_TWO_CALLS     0x10
#define  CWRAP_ALL_PROTECTED_ACCESS   0x20
#define  CWRAP_SMART_POINTER_OVERLOAD 0x40

/* --- Director Helpers --- */
  extern Node *Swig_methodclass(Node *n);
  extern int Swig_directorclass(Node *n);
  extern Node *Swig_directormap(Node *n, String *type);

/* --- Legacy Typemap API (somewhat simplified, ha!) --- */

  extern void Swig_typemap_init(void);
  extern void Swig_typemap_register(const_String_or_char_ptr tmap_method, ParmList *pattern, const_String_or_char_ptr code, ParmList *locals, ParmList *kwargs);
  extern int Swig_typemap_copy(const_String_or_char_ptr tmap_method, ParmList *srcpattern, ParmList *pattern);
  extern void Swig_typemap_clear(const_String_or_char_ptr tmap_method, ParmList *pattern);
  extern int Swig_typemap_apply(ParmList *srcpat, ParmList *destpat);
  extern void Swig_typemap_clear_apply(ParmList *pattern);
  extern void Swig_typemap_replace_embedded_typemap(String *s, Node *file_line_node);
  extern void Swig_typemap_debug(void);
  extern void Swig_typemap_search_debug_set(void);
  extern void Swig_typemap_used_debug_set(void);
  extern void Swig_typemap_register_debug_set(void);

  extern String *Swig_typemap_lookup(const_String_or_char_ptr tmap_method, Node *n, const_String_or_char_ptr lname, Wrapper *f);
  extern String *Swig_typemap_lookup_out(const_String_or_char_ptr tmap_method, Node *n, const_String_or_char_ptr lname, Wrapper *f, String *actioncode);

  extern void Swig_typemap_attach_parms(const_String_or_char_ptr tmap_method, ParmList *parms, Wrapper *f);

/* --- Code fragment support --- */

  extern void Swig_fragment_register(Node *fragment);
  extern void Swig_fragment_emit(String *name);
  extern void Swig_fragment_clear(String *section);

/* --- Extension support --- */

  extern Hash *Swig_extend_hash(void);
  extern void Swig_extend_merge(Node *cls, Node *am);
  extern void Swig_extend_append_previous(Node *cls, Node *am);
  extern void Swig_extend_unused_check(void);

/* hacks defined in C++ ! */
  extern int Swig_director_mode(void);
  extern int Swig_director_protected_mode(void);
  extern int Swig_all_protected_mode(void);
  extern void Wrapper_director_mode_set(int);
  extern void Wrapper_director_protected_mode_set(int);
  extern void Wrapper_all_protected_mode_set(int);
  extern void Language_replace_special_variables(String *method, String *tm, Parm *parm);
  extern void Swig_print(DOH *object, int count);
  extern void Swig_print_with_location(DOH *object, int count);


/* -- template init -- */
  extern void SwigType_template_init(void);


#ifdef __cplusplus
}
#endif
#endif
