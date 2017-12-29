/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * typesys.c
 *
 * SWIG type system management.   These functions are used to manage
 * the C++ type system including typenames, typedef, type scopes, 
 * inheritance, and namespaces.   Generation of support code for the
 * run-time type checker is also handled here.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "cparse.h"

/* -----------------------------------------------------------------------------
 * Synopsis
 *
 * The purpose of this module is to manage type names and scoping issues related
 * to the C++ type system.   The primary use is tracking typenames through typedef
 * and inheritance. 
 *
 * New typenames are introduced by typedef, class, and enum declarations.
 * Each type is declared in a scope.  This is either the global scope, a
 * class, or a namespace.  For example:
 *
 *  typedef int A;         // Typename A, in global scope
 *  namespace Foo {
 *    typedef int A;       // Typename A, in scope Foo::
 *  }
 *  class Bar {            // Typename Bar, in global scope
 *    typedef int A;       // Typename A, in scope Bar::
 *  }
 *
 * To manage scopes, the type system is constructed as a tree of hash tables.  Each 
 * hash table contains the following attributes:
 *
 *    "name"            -  Scope name
 *    "qname"           -  Fully qualified typename
 *    "typetab"         -  Type table containing typenames and typedef information
 *    "symtab"          -  Hash table of symbols defined in a scope
 *    "inherit"         -  List of inherited scopes       
 *    "parent"          -  Parent scope
 * 
 * Typedef information is stored in the "typetab" hash table.  For example,
 * if you have these declarations:
 *
 *      typedef int A;
 *      typedef A B;
 *      typedef B *C;
 *
 * typetab is built as follows:
 *
 *      "A"     : "int"
 *      "B"     : "A"
 *      "C"     : "p.B"
 *
 * To resolve a type back to its root type, one repeatedly expands on the type base.
 * For example:
 *
 *     C *[40]   ---> a(40).p.C       (string type representation, see stype.c)
 *               ---> a(40).p.p.B     (C --> p.B)
 *               ---> a(40).p.p.A     (B --> A)
 *               ---> a(40).p.p.int   (A --> int)
 *
 * For inheritance, SWIG tries to resolve types back to the base class. For instance, if
 * you have this:
 *
 *     class Foo {
 *     public:
 *        typedef int Integer;
 *     };
 *
 *     class Bar : public Foo {
 *           void blah(Integer x);
 *     };
 *
 * The argument type of Bar::blah will be set to Foo::Integer.   
 *
 * The scope-inheritance mechanism is used to manage C++ namespace aliases.
 * For example, if you have this:
 *
 *    namespace Foo {
 *         typedef int Integer;
 *    }
 *
 *   namespace F = Foo;
 *
 * In this case, "F::" is defined as a scope that "inherits" from Foo.  Internally,
 * "F::" will merely be an empty scope that refers to Foo.  SWIG will never 
 * place new type information into a namespace alias---attempts to do so
 * will generate a warning message (in the parser) and will place information into 
 * Foo instead.
 *
 *----------------------------------------------------------------------------- */

static Typetab *current_scope = 0;	/* Current type scope                           */
static Hash *current_typetab = 0;	/* Current type table                           */
static Hash *current_symtab = 0;	/* Current symbol table                         */
static Typetab *global_scope = 0;	/* The global scope                             */
static Hash *scopes = 0;	/* Hash table containing fully qualified scopes */

/* Performance optimization */
#define SWIG_TYPEDEF_RESOLVE_CACHE 
static Hash *typedef_resolve_cache = 0;
static Hash *typedef_all_cache = 0;
static Hash *typedef_qualified_cache = 0;

static Typetab *SwigType_find_scope(Typetab *s, const SwigType *nameprefix);

/* common attribute keys, to avoid calling find_key all the times */

/* 
   Enable this one if your language fully support SwigValueWrapper<T>.
   
   Leaving at '0' keeps the old swig behavior, which is not
   always safe, but is well known.

   Setting at '1' activates the new scheme, which is always safe but
   it requires all the typemaps to be ready for that.
  
*/
static int value_wrapper_mode = 0;
int Swig_value_wrapper_mode(int mode) {
  value_wrapper_mode = mode;
  return mode;
}


static void flush_cache() {
  typedef_resolve_cache = 0;
  typedef_all_cache = 0;
  typedef_qualified_cache = 0;
}

/* Initialize the scoping system */

void SwigType_typesystem_init() {
  if (global_scope)
    Delete(global_scope);
  if (scopes)
    Delete(scopes);

  current_scope = NewHash();
  global_scope = current_scope;

  Setattr(current_scope, "name", "");	/* No name for global scope */
  current_typetab = NewHash();
  Setattr(current_scope, "typetab", current_typetab);

  current_symtab = 0;
  scopes = NewHash();
  Setattr(scopes, "", current_scope);
}


/* -----------------------------------------------------------------------------
 * SwigType_typedef()
 *
 * Defines a new typedef in the current scope. Returns -1 if the type name is 
 * already defined.  
 * ----------------------------------------------------------------------------- */

int SwigType_typedef(const SwigType *type, const_String_or_char_ptr name) {
  if (Getattr(current_typetab, name))
    return -1;			/* Already defined */
  if (Strcmp(type, name) == 0) {	/* Can't typedef a name to itself */
    return 0;
  }

  /* Check if 'type' is already a scope.  If so, we create an alias in the type
     system for it.  This is needed to make strange nested scoping problems work
     correctly.  */
  {
    Typetab *t = SwigType_find_scope(current_scope, type);
    if (t) {
      SwigType_new_scope(name);
      SwigType_inherit_scope(t);
      SwigType_pop_scope();
    }
  }
  Setattr(current_typetab, name, type);
  flush_cache();
  return 0;
}


/* -----------------------------------------------------------------------------
 * SwigType_typedef_class()
 *
 * Defines a class in the current scope. 
 * ----------------------------------------------------------------------------- */

int SwigType_typedef_class(const_String_or_char_ptr name) {
  String *cname;
  /*  Printf(stdout,"class : '%s'\n", name); */
  if (Getattr(current_typetab, name))
    return -1;			/* Already defined */
  cname = NewString(name);
  Setmeta(cname, "class", "1");
  Setattr(current_typetab, cname, cname);
  Delete(cname);
  flush_cache();
  return 0;
}

/* -----------------------------------------------------------------------------
 * SwigType_scope_name()
 *
 * Returns the qualified scope name of a type table
 * ----------------------------------------------------------------------------- */

String *SwigType_scope_name(Typetab *ttab) {
  String *qname = NewString(Getattr(ttab, "name"));
  ttab = Getattr(ttab, "parent");
  while (ttab) {
    String *pname = Getattr(ttab, "name");
    if (Len(pname)) {
      Insert(qname, 0, "::");
      Insert(qname, 0, pname);
    }
    ttab = Getattr(ttab, "parent");
  }
  return qname;
}

/* -----------------------------------------------------------------------------
 * SwigType_new_scope()
 *
 * Creates a new scope
 * ----------------------------------------------------------------------------- */

void SwigType_new_scope(const_String_or_char_ptr name) {
  Typetab *s;
  Hash *ttab;
  String *qname;

  if (!name) {
    name = "<unnamed>";
  }
  s = NewHash();
  Setattr(s, "name", name);
  Setattr(s, "parent", current_scope);
  ttab = NewHash();
  Setattr(s, "typetab", ttab);

  /* Build fully qualified name and */
  qname = SwigType_scope_name(s);
  Setattr(scopes, qname, s);
  Setattr(s, "qname", qname);
  Delete(qname);

  current_scope = s;
  current_typetab = ttab;
  current_symtab = 0;
  flush_cache();
}

/* -----------------------------------------------------------------------------
 * SwigType_inherit_scope()
 *
 * Makes the current scope inherit from another scope.  This is used for both
 * C++ class inheritance, namespaces, and namespace aliases.
 * ----------------------------------------------------------------------------- */

void SwigType_inherit_scope(Typetab *scope) {
  List *inherits;
  int i, len;
  inherits = Getattr(current_scope, "inherit");
  if (!inherits) {
    inherits = NewList();
    Setattr(current_scope, "inherit", inherits);
    Delete(inherits);
  }
  assert(scope != current_scope);

  len = Len(inherits);
  for (i = 0; i < len; i++) {
    Node *n = Getitem(inherits, i);
    if (n == scope)
      return;
  }
  Append(inherits, scope);
}

/* -----------------------------------------------------------------------------
 * SwigType_scope_alias()
 *
 * Creates a scope-alias.
 * ----------------------------------------------------------------------------- */

void SwigType_scope_alias(String *aliasname, Typetab *ttab) {
  String *q;
  /*  Printf(stdout,"alias: '%s' '%p'\n", aliasname, ttab); */
  q = SwigType_scope_name(current_scope);
  if (Len(q)) {
    Append(q, "::");
  }
  Append(q, aliasname);
  Setattr(scopes, q, ttab);
  flush_cache();
}

/* -----------------------------------------------------------------------------
 * SwigType_using_scope()
 *
 * Import another scope into this scope.
 * ----------------------------------------------------------------------------- */

void SwigType_using_scope(Typetab *scope) {
  SwigType_inherit_scope(scope);
  {
    List *ulist;
    int i, len;
    ulist = Getattr(current_scope, "using");
    if (!ulist) {
      ulist = NewList();
      Setattr(current_scope, "using", ulist);
      Delete(ulist);
    }
    assert(scope != current_scope);
    len = Len(ulist);
    for (i = 0; i < len; i++) {
      Typetab *n = Getitem(ulist, i);
      if (n == scope)
	return;
    }
    Append(ulist, scope);
  }
  flush_cache();
}

/* -----------------------------------------------------------------------------
 * SwigType_pop_scope()
 *
 * Pop off the last scope and perform a merge operation.  Returns the hash
 * table for the scope that was popped off.
 * ----------------------------------------------------------------------------- */

Typetab *SwigType_pop_scope() {
  Typetab *t, *old = current_scope;
  t = Getattr(current_scope, "parent");
  if (!t)
    t = global_scope;
  current_scope = t;
  current_typetab = Getattr(t, "typetab");
  current_symtab = Getattr(t, "symtab");
  flush_cache();
  return old;
}

/* -----------------------------------------------------------------------------
 * SwigType_set_scope()
 *
 * Set the scope.  Returns the old scope.
 * ----------------------------------------------------------------------------- */

Typetab *SwigType_set_scope(Typetab *t) {
  Typetab *old = current_scope;
  if (!t)
    t = global_scope;
  current_scope = t;
  current_typetab = Getattr(t, "typetab");
  current_symtab = Getattr(t, "symtab");
  flush_cache();
  return old;
}

/* -----------------------------------------------------------------------------
 * SwigType_attach_symtab()
 *
 * Attaches a symbol table to a type scope
 * ----------------------------------------------------------------------------- */

void SwigType_attach_symtab(Symtab *sym) {
  Setattr(current_scope, "symtab", sym);
  current_symtab = sym;
}

/* -----------------------------------------------------------------------------
 * SwigType_print_scope()
 *
 * Debugging function for printing out current scope
 * ----------------------------------------------------------------------------- */

void SwigType_print_scope(void) {
  Hash *ttab;
  Iterator i, j;

  Printf(stdout, "SCOPES start  =======================================\n");
  for (i = First(scopes); i.key; i = Next(i)) {
    Printf(stdout, "-------------------------------------------------------------\n");
    ttab = Getattr(i.item, "typetab");

    Printf(stdout, "Type scope '%s' (%p)\n", i.key, i.item);
    {
      List *inherit = Getattr(i.item, "inherit");
      if (inherit) {
	Iterator j;
	for (j = First(inherit); j.item; j = Next(j)) {
	  Printf(stdout, "    Inherits from '%s' (%p)\n", Getattr(j.item, "qname"), j.item);
	}
      }
    }
    for (j = First(ttab); j.key; j = Next(j)) {
      Printf(stdout, "%40s -> %s\n", j.key, j.item);
    }
  }
  Printf(stdout, "SCOPES finish =======================================\n");
}

static Typetab *SwigType_find_scope(Typetab *s, const SwigType *nameprefix) {
  Typetab *ss;
  Typetab *s_orig = s;
  String *nnameprefix = 0;
  static int check_parent = 1;

  if (Getmark(s))
    return 0;
  Setmark(s, 1);

  if (SwigType_istemplate(nameprefix)) {
    nnameprefix = SwigType_typedef_resolve_all(nameprefix);
    nameprefix = nnameprefix;
  }

  ss = s;
  while (ss) {
    String *full;
    String *qname = Getattr(ss, "qname");
    if (qname) {
      full = NewStringf("%s::%s", qname, nameprefix);
    } else {
      full = NewString(nameprefix);
    }
    if (Getattr(scopes, full)) {
      s = Getattr(scopes, full);
    } else {
      s = 0;
    }
    Delete(full);
    if (s) {
      if (nnameprefix)
	Delete(nnameprefix);
      Setmark(s_orig, 0);
      return s;
    }
    if (!s) {
      /* Check inheritance */
      List *inherit;
      inherit = Getattr(ss, "using");
      if (inherit) {
	Typetab *ttab;
	int i, len;
	len = Len(inherit);
	for (i = 0; i < len; i++) {
	  int oldcp = check_parent;
	  ttab = Getitem(inherit, i);
	  check_parent = 0;
	  s = SwigType_find_scope(ttab, nameprefix);
	  check_parent = oldcp;
	  if (s) {
	    if (nnameprefix)
	      Delete(nnameprefix);
	    Setmark(s_orig, 0);
	    return s;
	  }
	}
      }
    }
    if (!check_parent)
      break;
    ss = Getattr(ss, "parent");
  }
  if (nnameprefix)
    Delete(nnameprefix);
  Setmark(s_orig, 0);
  return 0;
}

/* ----------------------------------------------------------------------------- 
 * typedef_resolve()
 *
 * Resolves a typedef and returns a new type string.  Returns 0 if there is no
 * typedef mapping.  base is a name without qualification.
 * Internal function.
 * ----------------------------------------------------------------------------- */

static Typetab *resolved_scope = 0;

/* Internal function */

static SwigType *_typedef_resolve(Typetab *s, String *base, int look_parent) {
  Hash *ttab;
  SwigType *type = 0;
  List *inherit;
  Typetab *parent;

  /* if (!s) return 0; *//* now is checked bellow */
  /* Printf(stdout,"Typetab %s : %s\n", Getattr(s,"name"), base);  */

  if (!Getmark(s)) {
    Setmark(s, 1);

    ttab = Getattr(s, "typetab");
    type = Getattr(ttab, base);
    if (type) {
      resolved_scope = s;
      Setmark(s, 0);
    } else {
      /* Hmmm. Not found in my scope.  It could be in an inherited scope */
      inherit = Getattr(s, "inherit");
      if (inherit) {
	int i, len;
	len = Len(inherit);
	for (i = 0; i < len; i++) {
	  type = _typedef_resolve(Getitem(inherit, i), base, 0);
	  if (type) {
	    Setmark(s, 0);
	    break;
	  }
	}
      }
      if (!type) {
	/* Hmmm. Not found in my scope.  check parent */
	if (look_parent) {
	  parent = Getattr(s, "parent");
	  type = parent ? _typedef_resolve(parent, base, 1) : 0;
	}
      }
      Setmark(s, 0);
    }
  }
  return type;
}

/* ----------------------------------------------------------------------------- 
 * template_parameters_resolve()
 *
 * For use with templates only. The template parameters are resolved. If none
 * of the parameters can be resolved, zero is returned.
 * ----------------------------------------------------------------------------- */

static String *template_parameters_resolve(const String *base) {
  List *tparms;
  String *suffix;
  String *type;
  int i, sz;
  int rep = 0;
  type = SwigType_templateprefix(base);
  suffix = SwigType_templatesuffix(base);
  Append(type, "<(");
  tparms = SwigType_parmlist(base);
  sz = Len(tparms);
  for (i = 0; i < sz; i++) {
    SwigType *tpr;
    SwigType *tp = Getitem(tparms, i);
    if (!rep) {
      tpr = SwigType_typedef_resolve(tp);
    } else {
      tpr = 0;
    }
    if (tpr) {
      Append(type, tpr);
      Delete(tpr);
      rep = 1;
    } else {
      Append(type, tp);
    }
    if ((i + 1) < sz)
      Append(type, ",");
  }
  Append(type, ")>");
  Append(type, suffix);
  Delete(suffix);
  Delete(tparms);
  if (!rep) {
    Delete(type);
    type = 0;
  }
  return type;
}

static SwigType *typedef_resolve(Typetab *s, String *base) {
  return _typedef_resolve(s, base, 1);
}


/* ----------------------------------------------------------------------------- 
 * SwigType_typedef_resolve()
 * ----------------------------------------------------------------------------- */

/* #define SWIG_DEBUG */
SwigType *SwigType_typedef_resolve(const SwigType *t) {
  String *base;
  String *type = 0;
  String *r = 0;
  Typetab *s;
  Hash *ttab;
  String *namebase = 0;
  String *nameprefix = 0, *rnameprefix = 0;
  int newtype = 0;

  resolved_scope = 0;

#ifdef SWIG_TYPEDEF_RESOLVE_CACHE
  if (!typedef_resolve_cache) {
    typedef_resolve_cache = NewHash();
  }
  r = Getattr(typedef_resolve_cache, t);
  if (r) {
    resolved_scope = Getmeta(r, "scope");
    return Copy(r);
  }
#endif

  base = SwigType_base(t);

#ifdef SWIG_DEBUG
  Printf(stdout, "base = '%s' t='%s'\n", base, t);
#endif

  if (SwigType_issimple(base)) {
    s = current_scope;
    ttab = current_typetab;
    if (strncmp(Char(base), "::", 2) == 0) {
      s = global_scope;
      ttab = Getattr(s, "typetab");
      Delitem(base, 0);
      Delitem(base, 0);
    }
    /* Do a quick check in the local scope */
    type = Getattr(ttab, base);
    if (type) {
      resolved_scope = s;
    }
    if (!type) {
      /* Didn't find in this scope.   We need to do a little more searching */
      if (Swig_scopename_check(base)) {
	/* A qualified name. */
	Swig_scopename_split(base, &nameprefix, &namebase);
#ifdef SWIG_DEBUG
	Printf(stdout, "nameprefix = '%s'\n", nameprefix);
#endif
	if (nameprefix) {
	  rnameprefix = SwigType_typedef_resolve(nameprefix);
	  if(rnameprefix != NULL) {
#ifdef SWIG_DEBUG
	    Printf(stdout, "nameprefix '%s' is a typedef to '%s'\n", nameprefix, rnameprefix);
#endif
	    type = Copy(namebase);
	    Insert(type, 0, "::");
	    Insert(type, 0, rnameprefix);
	    if (strncmp(Char(type), "::", 2) == 0) {
	      Delitem(type, 0);
	      Delitem(type, 0);
	    }
	    newtype = 1;
	  } else {
	    /* Name had a prefix on it.   See if we can locate the proper scope for it */
	    String *rnameprefix = template_parameters_resolve(nameprefix);
	    nameprefix = rnameprefix ? Copy(rnameprefix) : nameprefix;
	    Delete(rnameprefix);
	    s = SwigType_find_scope(s, nameprefix);

	    /* Couldn't locate a scope for the type.  */
	    if (!s) {
	      Delete(base);
	      Delete(namebase);
	      Delete(nameprefix);
	      r = 0;
	      goto return_result;
	    }
	    /* Try to locate the name starting in the scope */
#ifdef SWIG_DEBUG
	    Printf(stdout, "namebase = '%s'\n", namebase);
#endif
	    type = typedef_resolve(s, namebase);
	    if (type && resolved_scope) {
	      /* we need to look for the resolved type, this will also
	         fix the resolved_scope if 'type' and 'namebase' are
	         declared in different scopes */
	      String *rtype = 0;
	      rtype = typedef_resolve(resolved_scope, type);
	      if (rtype)
	        type = rtype;
	    }
#ifdef SWIG_DEBUG
	    Printf(stdout, "%s type = '%s'\n", Getattr(s, "name"), type);
#endif
	    if ((type) && (!Swig_scopename_check(type)) && resolved_scope) {
	      Typetab *rtab = resolved_scope;
	      String *qname = Getattr(resolved_scope, "qname");
	      /* If qualified *and* the typename is defined from the resolved scope, we qualify */
	      if ((qname) && typedef_resolve(resolved_scope, type)) {
	        type = Copy(type);
	        Insert(type, 0, "::");
	        Insert(type, 0, qname);
#ifdef SWIG_DEBUG
	        Printf(stdout, "qual %s \n", type);
#endif
	        newtype = 1;
	      }
	      resolved_scope = rtab;
	    }
	  }
	} else {
	  /* Name is unqualified. */
	  type = typedef_resolve(s, base);
	}
      } else {
	/* Name is unqualified. */
	type = typedef_resolve(s, base);
      }
    }

    if (type && (Equal(base, type))) {
      if (newtype)
	Delete(type);
      Delete(base);
      Delete(namebase);
      Delete(nameprefix);
      r = 0;
      goto return_result;
    }

    /* If the type is a template, and no typedef was found, we need to check the
       template arguments one by one to see if they can be resolved. */

    if (!type && SwigType_istemplate(base)) {
      newtype = 1;
      type = template_parameters_resolve(base);
    }
    if (namebase)
      Delete(namebase);
    if (nameprefix)
      Delete(nameprefix);
  } else {
    if (SwigType_isfunction(base)) {
      List *parms;
      int i, sz;
      int rep = 0;
      type = NewString("f(");
      newtype = 1;
      parms = SwigType_parmlist(base);
      sz = Len(parms);
      for (i = 0; i < sz; i++) {
	SwigType *tpr;
	SwigType *tp = Getitem(parms, i);
	if (!rep) {
	  tpr = SwigType_typedef_resolve(tp);
	} else {
	  tpr = 0;
	}
	if (tpr) {
	  Append(type, tpr);
	  Delete(tpr);
	  rep = 1;
	} else {
	  Append(type, tp);
	}
	if ((i + 1) < sz)
	  Append(type, ",");
      }
      Append(type, ").");
      Delete(parms);
      if (!rep) {
	Delete(type);
	type = 0;
      }
    } else if (SwigType_ismemberpointer(base)) {
      String *rt;
      String *mtype = SwigType_parm(base);
      rt = SwigType_typedef_resolve(mtype);
      if (rt) {
	type = NewStringf("m(%s).", rt);
	newtype = 1;
	Delete(rt);
      }
      Delete(mtype);
    } else {
      type = 0;
    }
  }
  r = SwigType_prefix(t);
  if (!type) {
    if (r && Len(r)) {
      char *cr = Char(r);
      if ((strstr(cr, "f(") || (strstr(cr, "m(")))) {
	SwigType *rt = SwigType_typedef_resolve(r);
	if (rt) {
	  Delete(r);
	  Append(rt, base);
	  Delete(base);
	  r = rt;
	  goto return_result;
	}
      }
    }
    Delete(r);
    Delete(base);
    r = 0;
    goto return_result;
  }
  Delete(base);

  /* If 'type' is an array, then the right-most qualifier in 'r' should
     be added to 'type' after the array qualifier, so that given
       a(7).q(volatile).double myarray     // typedef volatile double[7] myarray;
     the type
       q(const).myarray                    // const myarray
     becomes
       a(7).q(const volatile).double       // const volatile double[7]
     and NOT
       q(const).a(7).q(volatile).double    // non-sensical type
  */
  if (r && Len(r) && SwigType_isarray(type)) {
    List *r_elem;
    String *r_qual;
    int r_sz;
    r_elem = SwigType_split(r);
    r_sz = Len(r_elem);
    r_qual = Getitem(r_elem, r_sz-1);
    if (SwigType_isqualifier(r_qual)) {
      String *new_r;
      String *new_type;
      List *type_elem;
      String *type_qual;
      String *r_qual_arg;
      int i, type_sz;

      type_elem = SwigType_split(type);
      type_sz = Len(type_elem);

      for (i = 0; i < type_sz; ++i) {
        String *e = Getitem(type_elem, i);
        if (!SwigType_isarray(e))
          break;
      }
      type_qual = Copy(Getitem(type_elem, i));
      r_qual_arg = SwigType_parm(r_qual);
      SwigType_add_qualifier(type_qual, r_qual_arg);
      Delete(r_qual_arg);
      Setitem(type_elem, i, type_qual);

      new_r = NewStringEmpty();
      for (i = 0; i < r_sz-1; ++i) {
        Append(new_r, Getitem(r_elem, i));
      }
      new_type = NewStringEmpty();
      for (i = 0; i < type_sz; ++i) {
        Append(new_type, Getitem(type_elem, i));
      }
#ifdef SWIG_DEBUG
      Printf(stdout, "r+type='%s%s' new_r+new_type='%s%s'\n", r, type, new_r, new_type);
#endif

      Delete(r);
      r = new_r;
      newtype = 1;
      type = new_type;
      Delete(type_elem);
    }
    Delete(r_elem);
  }

  Append(r, type);
  if (newtype) {
    Delete(type);
  }

return_result:
#ifdef SWIG_TYPEDEF_RESOLVE_CACHE
  {
    String *key = NewString(t);
    if (r) {
      SwigType *r1;
      Setattr(typedef_resolve_cache, key, r);
      Setmeta(r, "scope", resolved_scope);
      r1 = Copy(r);
      Delete(r);
      r = r1;
    }
    Delete(key);
  }
#endif
  return r;
}

/* -----------------------------------------------------------------------------
 * SwigType_typedef_resolve_all()
 *
 * Fully resolve a type down to its most basic datatype
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_typedef_resolve_all(const SwigType *t) {
  SwigType *n;
  SwigType *r;
  int count = 0;

  /* Check to see if the typedef resolve has been done before by checking the cache */
  if (!typedef_all_cache) {
    typedef_all_cache = NewHash();
  }
  r = Getattr(typedef_all_cache, t);
  if (r) {
    return Copy(r);
  }

  /* Recursively resolve the typedef */
  r = NewString(t);
  while ((n = SwigType_typedef_resolve(r))) {
    Delete(r);
    r = n;
    if (++count >= 512) {
      Swig_error(Getfile(t), Getline(t), "Recursive typedef detected resolving '%s' to '%s' to '%s' and so on...\n", SwigType_str(t, 0), SwigType_str(SwigType_typedef_resolve(t), 0), SwigType_str(SwigType_typedef_resolve(SwigType_typedef_resolve(t)), 0));
      break;
    }
  }

  /* Add the typedef to the cache for next time it is looked up */
  {
    String *key;
    SwigType *rr = Copy(r);
    key = NewString(t);
    Setattr(typedef_all_cache, key, rr);
    Delete(key);
    Delete(rr);
  }
  return r;
}


/* -----------------------------------------------------------------------------
 * SwigType_typedef_qualified()
 *
 * Given a type declaration, this function tries to fully qualify it according to
 * typedef scope rules.
 * If the unary scope operator (::) is used as a prefix to the type to denote global
 * scope, it is left in place.
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_typedef_qualified(const SwigType *t) {
  List *elements;
  String *result;
  int i, len;

  if (!typedef_qualified_cache)
    typedef_qualified_cache = NewHash();
  result = Getattr(typedef_qualified_cache, t);
  if (result) {
    String *rc = Copy(result);
    return rc;
  }

  result = NewStringEmpty();
  elements = SwigType_split(t);
  len = Len(elements);
  for (i = 0; i < len; i++) {
    String *ty = 0;
    String *e = Getitem(elements, i);
    if (SwigType_issimple(e)) {
      if (!SwigType_istemplate(e)) {
	String *isenum = 0;
	if (SwigType_isenum(e)) {
	  isenum = NewString("enum ");
	  ty = NewString(Char(e) + 5);
	  e = ty;
	}
	resolved_scope = 0;
	if (typedef_resolve(current_scope, e) && resolved_scope) {
	  /* resolved_scope contains the scope that actually resolved the symbol */
	  String *qname = Getattr(resolved_scope, "qname");
	  if (qname) {
	    Insert(e, 0, "::");
	    Insert(e, 0, qname);
	  }
	} else {
	  if (Swig_scopename_check(e)) {
	    String *qlast;
	    String *qname;
	    Swig_scopename_split(e, &qname, &qlast);
	    if (qname) {
	      String *tqname = SwigType_typedef_qualified(qname);
	      Clear(e);
	      Printf(e, "%s::%s", tqname, qlast);
	      Delete(qname);
	      Delete(tqname);
	    }
	    Delete(qlast);

	    /* Automatic template instantiation might go here??? */
	  } else {
	    /* It's a bare name.  It's entirely possible, that the
	       name is part of a namespace. We'll check this by unrolling
	       out of the current scope */

	    Typetab *cs = current_scope;
	    while (cs) {
	      String *qs = SwigType_scope_name(cs);
	      if (Len(qs)) {
		Append(qs, "::");
	      }
	      Append(qs, e);
	      if (Getattr(scopes, qs)) {
		Clear(e);
		Append(e, qs);
		Delete(qs);
		break;
	      }
	      Delete(qs);
	      cs = Getattr(cs, "parent");
	    }
	  }
	}
	if (isenum) {
	  Insert(e, 0, isenum);
	  Delete(isenum);
	}
      } else {
	/* Template.  We need to qualify template parameters as well as the template itself */
	String *tprefix, *qprefix;
	String *tsuffix;
	Iterator pi;
	Parm *p;
	List *parms;
	ty = Swig_symbol_template_deftype(e, current_symtab);
	/*
	String *dt = Swig_symbol_template_deftype(e, current_symtab);
	ty = Swig_symbol_type_qualify(dt, 0);
	*/
	e = ty;
	parms = SwigType_parmlist(e);
	tprefix = SwigType_templateprefix(e);
	tsuffix = SwigType_templatesuffix(e);
	qprefix = SwigType_typedef_qualified(tprefix);
	Append(qprefix, "<(");
	pi = First(parms);
	while ((p = pi.item)) {
	  String *qt = SwigType_typedef_qualified(p);
	  if (Equal(qt, p)) {	/*  && (!Swig_scopename_check(qt))) */
	    /* No change in value.  It is entirely possible that the parameter is an integer value.
	       If there is a symbol table associated with this scope, we're going to check for this */

	    if (current_symtab) {
	      Node *lastnode = 0;
	      String *value = Copy(p);
	      while (1) {
		Node *n = Swig_symbol_clookup(value, current_symtab);
		if (n == lastnode)
		  break;
		lastnode = n;
		if (n) {
		  char *ntype = Char(nodeType(n));
		  if (strcmp(ntype, "enumitem") == 0) {
		    /* An enum item.   Generate a fully qualified name */
		    String *qn = Swig_symbol_qualified(n);
		    if (Len(qn)) {
		      Append(qn, "::");
		      Append(qn, Getattr(n, "name"));
		      Delete(value);
		      value = qn;
		      continue;
		    } else {
		      Delete(qn);
		      break;
		    }
		  } else if ((strcmp(ntype, "cdecl") == 0) && (Getattr(n, "value"))) {
		    Delete(value);
		    value = Copy(Getattr(n, "value"));
		    continue;
		  }
		}
		break;
	      }
	      Append(qprefix, value);
	      Delete(value);
	    } else {
	      Append(qprefix, p);
	    }
	  } else {
	    Append(qprefix, qt);
	  }
	  Delete(qt);
	  pi = Next(pi);
	  if (pi.item) {
	    Append(qprefix, ",");
	  }
	}
	Append(qprefix, ")>");
	Append(qprefix, tsuffix);
	Delete(tsuffix);
	Clear(e);
	Append(e, qprefix);
	Delete(tprefix);
	Delete(qprefix);
	Delete(parms);
	/*
	Delete(dt);
	*/
      }
      Append(result, e);
      Delete(ty);
    } else if (SwigType_isfunction(e)) {
      List *parms = SwigType_parmlist(e);
      String *s = NewString("f(");
      Iterator pi;
      pi = First(parms);
      while (pi.item) {
	String *pq = SwigType_typedef_qualified(pi.item);
	Append(s, pq);
	Delete(pq);
	pi = Next(pi);
	if (pi.item) {
	  Append(s, ",");
	}
      }
      Append(s, ").");
      Append(result, s);
      Delete(s);
      Delete(parms);
    } else if (SwigType_isarray(e)) {
      String *ndim;
      String *dim = SwigType_parm(e);
      ndim = Swig_symbol_string_qualify(dim, 0);
      Printf(result, "a(%s).", ndim);
      Delete(dim);
      Delete(ndim);
    } else {
      Append(result, e);
    }
  }
  Delete(elements);
  {
    String *key, *cresult;
    key = NewString(t);
    cresult = NewString(result);
    Setattr(typedef_qualified_cache, key, cresult);
    Delete(key);
    Delete(cresult);
  }
  return result;
}

/* ----------------------------------------------------------------------------- 
 * SwigType_istypedef()
 *
 * Checks a typename to see if it is a typedef.
 * ----------------------------------------------------------------------------- */

int SwigType_istypedef(const SwigType *t) {
  String *type;

  type = SwigType_typedef_resolve(t);
  if (type) {
    Delete(type);
    return 1;
  } else {
    return 0;
  }
}


/* -----------------------------------------------------------------------------
 * SwigType_typedef_using()
 *
 * Processes a 'using' declaration to import types from one scope into another.
 * Name is a qualified name like A::B.
 * ----------------------------------------------------------------------------- */

int SwigType_typedef_using(const_String_or_char_ptr name) {
  String *base;
  String *td;
  String *prefix;
  Typetab *s;
  Typetab *tt = 0;

  String *defined_name = 0;

  /*  Printf(stdout,"using %s\n", name); */

  if (!Swig_scopename_check(name))
    return -1;			/* Not properly qualified */
  base = Swig_scopename_last(name);

  /* See if the base is already defined in this scope */
  if (Getattr(current_typetab, base)) {
    Delete(base);
    return -1;
  }

  /* See if the using name is a scope */
  /*  tt = SwigType_find_scope(current_scope,name);
     Printf(stdout,"tt = %p, name = '%s'\n", tt, name); */

  /* We set up a typedef  B --> A::B */
  Setattr(current_typetab, base, name);

  /* Find the scope name where the symbol is defined */
  td = SwigType_typedef_resolve(name);
  /*  Printf(stdout,"td = '%s' %p\n", td, resolved_scope); */
  if (resolved_scope) {
    defined_name = Getattr(resolved_scope, "qname");
    if (defined_name) {
      defined_name = Copy(defined_name);
      Append(defined_name, "::");
      Append(defined_name, base);
      /*  Printf(stdout,"defined_name = '%s'\n", defined_name); */
      tt = SwigType_find_scope(current_scope, defined_name);
    }
  }
  if (td)
    Delete(td);


  /* Figure out the scope the using directive refers to */
  {
    prefix = Swig_scopename_prefix(name);
    if (prefix) {
      s = SwigType_find_scope(current_scope, prefix);
      if (s) {
	Hash *ttab = Getattr(s, "typetab");
	if (!Getattr(ttab, base) && defined_name) {
	  Setattr(ttab, base, defined_name);
	}
      }
    }
  }

  if (tt) {
    /* Using directive had its own scope.  We need to create a new scope for it */
    SwigType_new_scope(base);
    SwigType_inherit_scope(tt);
    SwigType_pop_scope();
  }

  if (defined_name)
    Delete(defined_name);
  Delete(prefix);
  Delete(base);
  return 0;
}

/* -----------------------------------------------------------------------------
 * SwigType_isclass()
 *
 * Determines if a type defines a class or not.   A class is defined by
 * its type-table entry maps to itself.  Note: a pointer to a class is not
 * a class.
 * ----------------------------------------------------------------------------- */

int SwigType_isclass(const SwigType *t) {
  SwigType *qty, *qtys;
  int isclass = 0;

  qty = SwigType_typedef_resolve_all(t);
  qtys = SwigType_strip_qualifiers(qty);
  if (SwigType_issimple(qtys)) {
    String *td = SwigType_typedef_resolve(qtys);
    if (td) {
      Delete(td);
    }
    if (resolved_scope) {
      isclass = 1;
    }
    /* Hmmm. Not a class.  If a template, it might be uninstantiated */
    if (!isclass) {
      String *tp = SwigType_istemplate_templateprefix(qtys);
      if (tp && Strcmp(tp, t) != 0) {
	isclass = SwigType_isclass(tp);
      }
      Delete(tp);
    }
  }
  Delete(qty);
  Delete(qtys);
  return isclass;
}

/* -----------------------------------------------------------------------------
 * SwigType_type()
 *
 * Returns an integer code describing the datatype.  This is only used for
 * compatibility with SWIG1.1 language modules and is likely to go away once
 * everything is based on typemaps.
 * ----------------------------------------------------------------------------- */

int SwigType_type(const SwigType *t) {
  char *c;
  /* Check for the obvious stuff */
  c = Char(t);

  if (strncmp(c, "p.", 2) == 0) {
    if (SwigType_type(c + 2) == T_CHAR)
      return T_STRING;
    else if (SwigType_type(c + 2) == T_WCHAR)
      return T_WSTRING;
    else
      return T_POINTER;
  }
  if (strncmp(c, "a(", 2) == 0)
    return T_ARRAY;
  if (strncmp(c, "r.", 2) == 0)
    return T_REFERENCE;
  if (strncmp(c, "z.", 2) == 0)
    return T_RVALUE_REFERENCE;
  if (strncmp(c, "m(", 2) == 0)
    return T_MPOINTER;
  if (strncmp(c, "q(", 2) == 0) {
    while (*c && (*c != '.'))
      c++;
    if (*c)
      return SwigType_type(c + 1);
    return T_ERROR;
  }
  if (strncmp(c, "f(", 2) == 0)
    return T_FUNCTION;

  /* Look for basic types */
  if (strcmp(c, "int") == 0)
    return T_INT;
  if (strcmp(c, "long") == 0)
    return T_LONG;
  if (strcmp(c, "short") == 0)
    return T_SHORT;
  if (strcmp(c, "unsigned") == 0)
    return T_UINT;
  if (strcmp(c, "unsigned short") == 0)
    return T_USHORT;
  if (strcmp(c, "unsigned long") == 0)
    return T_ULONG;
  if (strcmp(c, "unsigned int") == 0)
    return T_UINT;
  if (strcmp(c, "char") == 0)
    return T_CHAR;
  if (strcmp(c, "signed char") == 0)
    return T_SCHAR;
  if (strcmp(c, "unsigned char") == 0)
    return T_UCHAR;
  if (strcmp(c, "wchar_t") == 0)
    return T_WCHAR;
  if (strcmp(c, "float") == 0)
    return T_FLOAT;
  if (strcmp(c, "double") == 0)
    return T_DOUBLE;
  if (strcmp(c, "long double") == 0)
    return T_LONGDOUBLE;
  if (!cparse_cplusplus && (strcmp(c, "float complex") == 0))
    return T_FLTCPLX;
  if (!cparse_cplusplus && (strcmp(c, "double complex") == 0))
    return T_DBLCPLX;
  if (!cparse_cplusplus && (strcmp(c, "complex") == 0))
    return T_COMPLEX;
  if (strcmp(c, "void") == 0)
    return T_VOID;
  if (strcmp(c, "bool") == 0)
    return T_BOOL;
  if (strcmp(c, "long long") == 0)
    return T_LONGLONG;
  if (strcmp(c, "unsigned long long") == 0)
    return T_ULONGLONG;
  if (strncmp(c, "enum ", 5) == 0)
    return T_INT;
  if (strcmp(c, "auto") == 0)
    return T_AUTO;

  if (strcmp(c, "v(...)") == 0)
    return T_VARARGS;
  /* Hmmm. Unknown type */
  if (SwigType_istypedef(t)) {
    int r;
    SwigType *nt = SwigType_typedef_resolve(t);
    r = SwigType_type(nt);
    Delete(nt);
    return r;
  }
  return T_USER;
}

/* -----------------------------------------------------------------------------
 * SwigType_alttype()
 *
 * Returns the alternative value type needed in C++ for class value
 * types. When swig is not sure about using a plain $ltype value,
 * since the class doesn't have a default constructor, or it can't be
 * assigned, you will get back 'SwigValueWrapper<type >'.
 *
 * This is the default behavior unless:
 *
 *  1.- swig detects a default_constructor and 'setallocate:default_constructor'
 *      attribute.
 *
 *  2.- swig doesn't mark 'type' as non-assignable.
 *
 *  3.- the user specifies that the value wrapper is not needed by using
 *      %feature("novaluewrapper") like so:
 *
 *        %feature("novaluewrapper") MyOpaqueClass;
 *        class MyOpaqueClass;
 *
 * The user can also force the use of the value wrapper with
 * %feature("valuewrapper").
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_alttype(const SwigType *t, int local_tmap) {
  Node *n;
  SwigType *w = 0;
  int use_wrapper = 0;
  SwigType *td = 0;

  if (!cparse_cplusplus)
    return 0;

  if (value_wrapper_mode == 0) {
    /* old partial use of SwigValueTypes, it can fail for opaque types */
    if (local_tmap)
      return 0;
    if (SwigType_isclass(t)) {
      SwigType *ftd = SwigType_typedef_resolve_all(t);
      td = SwigType_strip_qualifiers(ftd);
      Delete(ftd);
      n = Swig_symbol_clookup(td, 0);
      if (n) {
	if (GetFlag(n, "feature:valuewrapper")) {
	  use_wrapper = 1;
	} else {
	  if (Checkattr(n, "nodeType", "class")
	      && (!Getattr(n, "allocate:default_constructor")
		  || (Getattr(n, "allocate:noassign")))) {
	    use_wrapper = !GetFlag(n, "feature:novaluewrapper") || GetFlag(n, "feature:nodefault");
	  }
	}
      } else {
	if (SwigType_issimple(td) && SwigType_istemplate(td)) {
	  use_wrapper = !n || !GetFlag(n, "feature:novaluewrapper");
	}
      }
    }
  } else {
    /* safe use of SwigValueTypes, it can fail with some typemaps */
    SwigType *ftd = SwigType_typedef_resolve_all(t);
    td = SwigType_strip_qualifiers(ftd);
    Delete(ftd);
    if (SwigType_type(td) == T_USER) {
      use_wrapper = 1;
      n = Swig_symbol_clookup(td, 0);
      if (n) {
	if ((Checkattr(n, "nodeType", "class")
	     && !Getattr(n, "allocate:noassign")
	     && (Getattr(n, "allocate:default_constructor")))
	    || (GetFlag(n, "feature:novaluewrapper"))) {
	  use_wrapper = GetFlag(n, "feature:valuewrapper");
	}
      }
    }
  }

  if (use_wrapper) {
    /* Need a space before the type in case it starts "::" (since the <:
     * token is a digraph for [ in C++.  Also need a space after the
     * type in case it ends with ">" since then we form the token ">>".
     */
    w = NewStringf("SwigValueWrapper< %s >", td);
  }
  Delete(td);
  return w;
}

/* ----------------------------------------------------------------------------
 *                         * * * WARNING * * *                            ***
 *                                                                        ***
 * Don't even think about modifying anything below this line unless you   ***
 * are completely on top of *EVERY* subtle aspect of the C++ type system  ***
 * and you are prepared to suffer endless hours of agony trying to        ***
 * debug the SWIG run-time type checker after you break it.               ***
 * ------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * SwigType_remember()
 *
 * This function "remembers" a datatype that was used during wrapper code generation
 * so that a type-checking table can be generated later on. It is up to the language
 * modules to actually call this function--it is not done automatically.
 *
 * Type tracking is managed through two separate hash tables.  The hash 'r_mangled'
 * is mapping between mangled type names (used in the target language) and 
 * fully-resolved C datatypes used in the source input.   The second hash 'r_resolved'
 * is the inverse mapping that maps fully-resolved C datatypes to all of the mangled
 * names in the scripting languages.  For example, consider the following set of
 * typedef declarations:
 *
 *      typedef double Real;
 *      typedef double Float;
 *      typedef double Point[3];
 *
 * Now, suppose that the types 'double *', 'Real *', 'Float *', 'double[3]', and
 * 'Point' were used in an interface file and "remembered" using this function.
 * The hash tables would look like this:
 *
 * r_mangled {
 *   _p_double : [ p.double, a(3).double ]
 *   _p_Real   : [ p.double ]
 *   _p_Float  : [ p.double ]
 *   _Point    : [ a(3).double ]
 *
 * r_resolved {
 *   p.double     : [ _p_double, _p_Real, _p_Float ]
 *   a(3).double  : [ _p_double, _Point ]
 * }
 *
 * Together these two hash tables can be used to determine type-equivalency between
 * mangled typenames.  To do this, we view the two hash tables as a large graph and
 * compute the transitive closure.
 * ----------------------------------------------------------------------------- */

static Hash *r_mangled = 0;	/* Hash mapping mangled types to fully resolved types */
static Hash *r_resolved = 0;	/* Hash mapping resolved types to mangled types       */
static Hash *r_ltype = 0;	/* Hash mapping mangled names to their local c type   */
static Hash *r_clientdata = 0;	/* Hash mapping resolved types to client data         */
static Hash *r_mangleddata = 0;	/* Hash mapping mangled types to client data         */
static Hash *r_remembered = 0;	/* Hash of types we remembered already */

static void (*r_tracefunc) (const SwigType *t, String *mangled, String *clientdata) = 0;

void SwigType_remember_mangleddata(String *mangled, const_String_or_char_ptr clientdata) {
  if (!r_mangleddata) {
    r_mangleddata = NewHash();
  }
  Setattr(r_mangleddata, mangled, clientdata);
}


void SwigType_remember_clientdata(const SwigType *t, const_String_or_char_ptr clientdata) {
  String *mt;
  SwigType *lt;
  Hash *h;
  SwigType *fr;
  SwigType *qr;
  String *tkey;
  String *cd;
  Hash *lthash;

  if (!r_mangled) {
    r_mangled = NewHash();
    r_resolved = NewHash();
    r_ltype = NewHash();
    r_clientdata = NewHash();
    r_remembered = NewHash();
  }

  {
    String *last;
    last = Getattr(r_remembered, t);
    if (last && (Cmp(last, clientdata) == 0))
      return;
  }

  tkey = Copy(t);
  cd = clientdata ? NewString(clientdata) : NewStringEmpty();
  Setattr(r_remembered, tkey, cd);
  Delete(tkey);
  Delete(cd);

  mt = SwigType_manglestr(t);	/* Create mangled string */

  if (r_tracefunc) {
    (*r_tracefunc) (t, mt, (String *) clientdata);
  }

  if (SwigType_istypedef(t)) {
    lt = Copy(t);
  } else {
    lt = SwigType_ltype(t);
  }

  lthash = Getattr(r_ltype, mt);
  if (!lthash) {
    lthash = NewHash();
    Setattr(r_ltype, mt, lthash);
  }
  Setattr(lthash, lt, "1");
  Delete(lt);

  fr = SwigType_typedef_resolve_all(t);	/* Create fully resolved type */
  qr = SwigType_typedef_qualified(fr);
  Delete(fr);

  /* Added to deal with possible table bug */
  fr = SwigType_strip_qualifiers(qr);
  Delete(qr);

  /*Printf(stdout,"t = '%s'\n", t);
     Printf(stdout,"fr= '%s'\n\n", fr); */

  if (t) {
    char *ct = Char(t);
    if (strchr(ct, '<') && !(strstr(ct, "<("))) {
      Printf(stdout, "Bad template type passed to SwigType_remember: %s\n", t);
      assert(0);
    }
  }

  h = Getattr(r_mangled, mt);
  if (!h) {
    h = NewHash();
    Setattr(r_mangled, mt, h);
    Delete(h);
  }
  Setattr(h, fr, mt);

  h = Getattr(r_resolved, fr);
  if (!h) {
    h = NewHash();
    Setattr(r_resolved, fr, h);
    Delete(h);
  }
  Setattr(h, mt, fr);

  if (clientdata) {
    String *cd = Getattr(r_clientdata, fr);
    if (cd) {
      if (Strcmp(clientdata, cd) != 0) {
	Printf(stderr, "*** Internal error. Inconsistent clientdata for type '%s'\n", SwigType_str(fr, 0));
	Printf(stderr, "*** '%s' != '%s'\n", clientdata, cd);
	assert(0);
      }
    } else {
      String *cstr = NewString(clientdata);
      Setattr(r_clientdata, fr, cstr);
      Delete(cstr);
    }
  }

  /* If the remembered type is a reference, we also remember the pointer version.
     This is to prevent odd problems with mixing pointers and references--especially
     when different functions are using different typenames (via typedef). */

  if (SwigType_isreference(t)) {
    SwigType *tt = Copy(t);
    SwigType_del_reference(tt);
    SwigType_add_pointer(tt);
    SwigType_remember_clientdata(tt, clientdata);
  } else if (SwigType_isrvalue_reference(t)) {
    SwigType *tt = Copy(t);
    SwigType_del_rvalue_reference(tt);
    SwigType_add_pointer(tt);
    SwigType_remember_clientdata(tt, clientdata);
  }
}

void SwigType_remember(const SwigType *ty) {
  SwigType_remember_clientdata(ty, 0);
}

void (*SwigType_remember_trace(void (*tf) (const SwigType *, String *, String *))) (const SwigType *, String *, String *) {
  void (*o) (const SwigType *, String *, String *) = r_tracefunc;
  r_tracefunc = tf;
  return o;
}

/* -----------------------------------------------------------------------------
 * SwigType_equivalent_mangle()
 *
 * Return a list of all of the mangled typenames that are equivalent to another
 * mangled name.   This works as follows: For each fully qualified C datatype
 * in the r_mangled hash entry, we collect all of the mangled names from the
 * r_resolved hash and combine them together in a list (removing duplicate entries).
 * ----------------------------------------------------------------------------- */

List *SwigType_equivalent_mangle(String *ms, Hash *checked, Hash *found) {
  List *l;
  Hash *h;
  Hash *ch;
  Hash *mh;

  if (found) {
    h = found;
  } else {
    h = NewHash();
  }
  if (checked) {
    ch = checked;
  } else {
    ch = NewHash();
  }
  if (Getattr(ch, ms))
    goto check_exit;		/* Already checked this type */
  Setattr(h, ms, "1");
  Setattr(ch, ms, "1");
  mh = Getattr(r_mangled, ms);
  if (mh) {
    Iterator ki;
    ki = First(mh);
    while (ki.key) {
      Hash *rh;
      if (Getattr(ch, ki.key)) {
	ki = Next(ki);
	continue;
      }
      Setattr(ch, ki.key, "1");
      rh = Getattr(r_resolved, ki.key);
      if (rh) {
	Iterator rk;
	rk = First(rh);
	while (rk.key) {
	  Setattr(h, rk.key, "1");
	  SwigType_equivalent_mangle(rk.key, ch, h);
	  rk = Next(rk);
	}
      }
      ki = Next(ki);
    }
  }
check_exit:
  if (!found) {
    l = Keys(h);
    Delete(h);
    Delete(ch);
    return l;
  } else {
    return 0;
  }
}

/* -----------------------------------------------------------------------------
 * SwigType_clientdata_collect()
 *
 * Returns the clientdata field for a mangled type-string.
 * ----------------------------------------------------------------------------- */

static
String *SwigType_clientdata_collect(String *ms) {
  Hash *mh;
  String *clientdata = 0;

  if (r_mangleddata) {
    clientdata = Getattr(r_mangleddata, ms);
    if (clientdata)
      return clientdata;
  }

  mh = Getattr(r_mangled, ms);
  if (mh) {
    Iterator ki;
    ki = First(mh);
    while (ki.key) {
      clientdata = Getattr(r_clientdata, ki.key);
      if (clientdata)
	break;
      ki = Next(ki);
    }
  }
  return clientdata;
}




/* -----------------------------------------------------------------------------
 * SwigType_inherit()
 *
 * Record information about inheritance.  We keep a hash table that keeps
 * a mapping between base classes and all of the classes that are derived
 * from them.
 *
 * subclass is a hash that maps base-classes to all of the classes derived from them.
 *
 * derived - name of derived class
 * base - name of base class
 * cast - additional casting code when casting from derived to base
 * conversioncode - if set, overrides the default code in the function when casting
 *                from derived to base
 * ----------------------------------------------------------------------------- */

static Hash *subclass = 0;
static Hash *conversions = 0;

void SwigType_inherit(String *derived, String *base, String *cast, String *conversioncode) {
  Hash *h;
  String *dd = 0;
  String *bb = 0;
  if (!subclass)
    subclass = NewHash();

  /* Printf(stdout,"'%s' --> '%s'  '%s'\n", derived, base, cast); */

  if (SwigType_istemplate(derived)) {
    String *ty = SwigType_typedef_resolve_all(derived);
    dd = SwigType_typedef_qualified(ty);
    derived = dd;
    Delete(ty);
  }
  if (SwigType_istemplate(base)) {
    String *ty = SwigType_typedef_resolve_all(base);
    bb = SwigType_typedef_qualified(ty);
    base = bb;
    Delete(ty);
  }

  /* Printf(stdout,"'%s' --> '%s'  '%s'\n", derived, base, cast); */

  h = Getattr(subclass, base);
  if (!h) {
    h = NewHash();
    Setattr(subclass, base, h);
    Delete(h);
  }
  if (!Getattr(h, derived)) {
    Hash *c = NewHash();
    if (cast)
      Setattr(c, "cast", cast);
    if (conversioncode)
      Setattr(c, "convcode", conversioncode);
    Setattr(h, derived, c);
    Delete(c);
  }

  Delete(dd);
  Delete(bb);
}

/* -----------------------------------------------------------------------------
 * SwigType_issubtype()
 *
 * Determines if a t1 is a subtype of t2, ie, is t1 derived from t2
 * ----------------------------------------------------------------------------- */

int SwigType_issubtype(const SwigType *t1, const SwigType *t2) {
  SwigType *ft1, *ft2;
  String *b1, *b2;
  Hash *h;
  int r = 0;

  if (!subclass)
    return 0;

  ft1 = SwigType_typedef_resolve_all(t1);
  ft2 = SwigType_typedef_resolve_all(t2);
  b1 = SwigType_base(ft1);
  b2 = SwigType_base(ft2);

  h = Getattr(subclass, b2);
  if (h) {
    if (Getattr(h, b1)) {
      r = 1;
    }
  }
  Delete(ft1);
  Delete(ft2);
  Delete(b1);
  Delete(b2);
  /* Printf(stdout, "issubtype(%s,%s) --> %d\n", t1, t2, r); */
  return r;
}

/* -----------------------------------------------------------------------------
 * SwigType_inherit_equiv()
 *
 * Modify the type table to handle C++ inheritance
 * ----------------------------------------------------------------------------- */

void SwigType_inherit_equiv(File *out) {
  String *ckey;
  String *prefix, *base;
  String *mprefix, *mkey;
  Hash *sub;
  Hash *rh;
  List *rlist;
  Iterator rk, bk, ck;

  if (!conversions)
    conversions = NewHash();
  if (!subclass)
    subclass = NewHash();

  rk = First(r_resolved);
  while (rk.key) {
    /* rkey is a fully qualified type.  We strip all of the type constructors off of it just to get the base */
    base = SwigType_base(rk.key);
    /* Check to see whether the base is recorded in the subclass table */
    sub = Getattr(subclass, base);
    Delete(base);
    if (!sub) {
      rk = Next(rk);
      continue;
    }

    /* This type has subclasses.  We now need to walk through these subtypes and generate pointer conversion functions */

    rh = Getattr(r_resolved, rk.key);
    rlist = NewList();
    for (ck = First(rh); ck.key; ck = Next(ck)) {
      Append(rlist, ck.key);
    }
    /*    Printf(stdout,"rk.key = '%s'\n", rk.key);
       Printf(stdout,"rh = %p '%s'\n", rh,rh); */

    bk = First(sub);
    while (bk.key) {
      prefix = SwigType_prefix(rk.key);
      Append(prefix, bk.key);
      /*      Printf(stdout,"set %p = '%s' : '%s'\n", rh, SwigType_manglestr(prefix),prefix); */
      mprefix = SwigType_manglestr(prefix);
      Setattr(rh, mprefix, prefix);
      mkey = SwigType_manglestr(rk.key);
      ckey = NewStringf("%s+%s", mprefix, mkey);
      if (!Getattr(conversions, ckey)) {
	String *convname = NewStringf("%sTo%s", mprefix, mkey);
	String *lkey = SwigType_lstr(rk.key, 0);
	String *lprefix = SwigType_lstr(prefix, 0);
        Hash *subhash = Getattr(sub, bk.key);
        String *convcode = Getattr(subhash, "convcode");
        if (convcode) {
          char *newmemoryused = Strstr(convcode, "newmemory"); /* see if newmemory parameter is used in order to avoid unused parameter warnings */
          String *fn = Copy(convcode);
          Replaceall(fn, "$from", "x");
          Printf(out, "static void *%s(void *x, int *%s) {", convname, newmemoryused ? "newmemory" : "SWIGUNUSEDPARM(newmemory)");
          Printf(out, "%s", fn);
        } else {
          String *cast = Getattr(subhash, "cast");
          Printf(out, "static void *%s(void *x, int *SWIGUNUSEDPARM(newmemory)) {", convname);
          Printf(out, "\n    return (void *)((%s) ", lkey);
          if (cast)
            Printf(out, "%s", cast);
          Printf(out, " ((%s) x));\n", lprefix);
        }
	Printf(out, "}\n");
	Setattr(conversions, ckey, convname);
	Delete(ckey);
	Delete(lkey);
	Delete(lprefix);

	/* This inserts conversions for typedefs */
	{
	  Hash *r = Getattr(r_resolved, prefix);
	  if (r) {
	    Iterator rrk;
	    rrk = First(r);
	    while (rrk.key) {
	      Iterator rlk;
	      String *rkeymangle;

	      /* Make sure this name equivalence is not due to inheritance */
	      if (Cmp(prefix, Getattr(r, rrk.key)) == 0) {
		rkeymangle = Copy(mkey);
		ckey = NewStringf("%s+%s", rrk.key, rkeymangle);
		if (!Getattr(conversions, ckey)) {
		  Setattr(conversions, ckey, convname);
		}
		Delete(ckey);
		for (rlk = First(rlist); rlk.item; rlk = Next(rlk)) {
		  ckey = NewStringf("%s+%s", rrk.key, rlk.item);
		  Setattr(conversions, ckey, convname);
		  Delete(ckey);
		}
		Delete(rkeymangle);
		/* This is needed to pick up other alternative names for the same type.
		   Needed to make templates work */
		Setattr(rh, rrk.key, rrk.item);
	      }
	      rrk = Next(rrk);
	    }
	  }
	}
	Delete(convname);
      }
      Delete(prefix);
      Delete(mprefix);
      Delete(mkey);
      bk = Next(bk);
    }
    rk = Next(rk);
    Delete(rlist);
  }
}

/* Helper function to sort the mangled list */
static int SwigType_compare_mangled(const DOH *a, const DOH *b) {
  return strcmp((char *) Data(a), (char *) Data(b));
}

/* -----------------------------------------------------------------------------
 * SwigType_get_sorted_mangled_list()
 *
 * Returns the sorted list of mangled type names that should be exported into the
 * wrapper file.
 * ----------------------------------------------------------------------------- */
List *SwigType_get_sorted_mangled_list() {
  List *l = Keys(r_mangled);
  SortList(l, SwigType_compare_mangled);
  return l;
}


/* -----------------------------------------------------------------------------
 * SwigType_type_table()
 *
 * Generate the type-table for the type-checker.
 * ----------------------------------------------------------------------------- */

void SwigType_emit_type_table(File *f_forward, File *f_table) {
  Iterator ki;
  String *types, *table, *cast, *cast_init, *cast_temp;
  Hash *imported_types;
  List *mangled_list;
  List *table_list = NewList();
  int i = 0;

  if (!r_mangled) {
    r_mangled = NewHash();
    r_resolved = NewHash();
  }

  Printf(f_table, "\n/* -------- TYPE CONVERSION AND EQUIVALENCE RULES (BEGIN) -------- */\n\n");

  SwigType_inherit_equiv(f_table);

   /*#define DEBUG 1*/
#ifdef DEBUG
  Printf(stdout, "---r_mangled---\n");
  Printf(stdout, "%s\n", r_mangled);

  Printf(stdout, "---r_resolved---\n");
  Printf(stdout, "%s\n", r_resolved);

  Printf(stdout, "---r_ltype---\n");
  Printf(stdout, "%s\n", r_ltype);

  Printf(stdout, "---subclass---\n");
  Printf(stdout, "%s\n", subclass);

  Printf(stdout, "---conversions---\n");
  Printf(stdout, "%s\n", conversions);

  Printf(stdout, "---r_clientdata---\n");
  Printf(stdout, "%s\n", r_clientdata);

#endif
  table = NewStringEmpty();
  types = NewStringEmpty();
  cast = NewStringEmpty();
  cast_init = NewStringEmpty();
  imported_types = NewHash();

  Printf(table, "static swig_type_info *swig_type_initial[] = {\n");
  Printf(cast_init, "static swig_cast_info *swig_cast_initial[] = {\n");

  Printf(f_forward, "\n/* -------- TYPES TABLE (BEGIN) -------- */\n\n");

  mangled_list = SwigType_get_sorted_mangled_list();
  for (ki = First(mangled_list); ki.item; ki = Next(ki)) {
    List *el;
    Iterator ei;
    SwigType *lt;
    SwigType *rt = 0;
    String *nt;
    String *ln;
    String *rn;
    const String *cd;
    Hash *lthash;
    Iterator ltiter;
    Hash *nthash;

    cast_temp = NewStringEmpty();

    Printv(types, "static swig_type_info _swigt_", ki.item, " = {", NIL);
    Append(table_list, ki.item);
    Printf(cast_temp, "static swig_cast_info _swigc_%s[] = {", ki.item);
    i++;

    cd = SwigType_clientdata_collect(ki.item);
    if (!cd)
      cd = "0";

    lthash = Getattr(r_ltype, ki.item);
    nt = 0;
    nthash = NewHash();
    ltiter = First(lthash);
    while (ltiter.key) {
      lt = ltiter.key;
      rt = SwigType_typedef_resolve_all(lt);
      /* we save the original type and the fully resolved version */
      ln = SwigType_lstr(lt, 0);
      rn = SwigType_lstr(rt, 0);
      if (Equal(ln, rn)) {
        Setattr(nthash, ln, "1");
      } else {
	Setattr(nthash, rn, "1");
	Setattr(nthash, ln, "1");
      }
      if (SwigType_istemplate(rt)) {
        String *dt = Swig_symbol_template_deftype(rt, 0);
        String *dn = SwigType_lstr(dt, 0);
        if (!Equal(dn, rn) && !Equal(dn, ln)) {
          Setattr(nthash, dn, "1");
        }
        Delete(dt);
        Delete(dn);
      }

      ltiter = Next(ltiter);
    }

    /* now build nt */
    ltiter = First(nthash);
    nt = 0;
    while (ltiter.key) {
      if (nt) {
	 Printf(nt, "|%s", ltiter.key);
      } else {
	 nt = NewString(ltiter.key);
      }
      ltiter = Next(ltiter);
    }
    Delete(nthash);

    Printf(types, "\"%s\", \"%s\", 0, 0, (void*)%s, 0};\n", ki.item, nt, cd);

    el = SwigType_equivalent_mangle(ki.item, 0, 0);
    for (ei = First(el); ei.item; ei = Next(ei)) {
      String *ckey;
      String *conv;
      ckey = NewStringf("%s+%s", ei.item, ki.item);
      conv = Getattr(conversions, ckey);
      if (conv) {
	Printf(cast_temp, "  {&_swigt_%s, %s, 0, 0},", ei.item, conv);
      } else {
	Printf(cast_temp, "  {&_swigt_%s, 0, 0, 0},", ei.item);
      }
      Delete(ckey);

      if (!Getattr(r_mangled, ei.item) && !Getattr(imported_types, ei.item)) {
	Printf(types, "static swig_type_info _swigt_%s = {\"%s\", 0, 0, 0, 0, 0};\n", ei.item, ei.item);
	Append(table_list, ei.item);

	Printf(cast, "static swig_cast_info _swigc_%s[] = {{&_swigt_%s, 0, 0, 0},{0, 0, 0, 0}};\n", ei.item, ei.item);
	i++;

	Setattr(imported_types, ei.item, "1");
      }
    }
    Delete(el);
    Printf(cast, "%s{0, 0, 0, 0}};\n", cast_temp);
    Delete(cast_temp);
    Delete(nt);
    Delete(rt);
  }
  /* print the tables in the proper order */
  SortList(table_list, SwigType_compare_mangled);
  i = 0;
  for (ki = First(table_list); ki.item; ki = Next(ki)) {
    Printf(f_forward, "#define SWIGTYPE%s swig_types[%d]\n", ki.item, i++);
    Printf(table, "  &_swigt_%s,\n", ki.item);
    Printf(cast_init, "  _swigc_%s,\n", ki.item);
  }
  if (i == 0) {
    /* empty arrays are not allowed by ISO C */
    Printf(table, "  NULL\n");
    Printf(cast_init, "  NULL\n");
  }

  Delete(table_list);

  Delete(mangled_list);

  Printf(table, "};\n");
  Printf(cast_init, "};\n");
  Printf(f_table, "%s\n", types);
  Printf(f_table, "%s\n", table);
  Printf(f_table, "%s\n", cast);
  Printf(f_table, "%s\n", cast_init);
  Printf(f_table, "\n/* -------- TYPE CONVERSION AND EQUIVALENCE RULES (END) -------- */\n\n");

  Printf(f_forward, "static swig_type_info *swig_types[%d];\n", i + 1);
  Printf(f_forward, "static swig_module_info swig_module = {swig_types, %d, 0, 0, 0, 0};\n", i);
  Printf(f_forward, "#define SWIG_TypeQuery(name) SWIG_TypeQueryModule(&swig_module, &swig_module, name)\n");
  Printf(f_forward, "#define SWIG_MangledTypeQuery(name) SWIG_MangledTypeQueryModule(&swig_module, &swig_module, name)\n");
  Printf(f_forward, "\n/* -------- TYPES TABLE (END) -------- */\n\n");

  Delete(types);
  Delete(table);
  Delete(cast);
  Delete(cast_init);
  Delete(imported_types);
}
