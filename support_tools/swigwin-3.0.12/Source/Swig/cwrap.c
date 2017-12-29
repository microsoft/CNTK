/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * cwrap.c
 *
 * This file defines a variety of wrapping rules for C/C++ handling including
 * the naming of local variables, calling conventions, and so forth.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "cparse.h"

static const char *cresult_variable_name = "result";

static Parm *nonvoid_parms(Parm *p) {
  if (p) {
    SwigType *t = Getattr(p, "type");
    if (SwigType_type(t) == T_VOID)
      return 0;
  }
  return p;
}

/* -----------------------------------------------------------------------------
 * Swig_cresult_name_set()
 *
 * Change the name of the variable used to hold the return value from C/C++ wrapper functions
 * from the default "result".
 * ----------------------------------------------------------------------------- */

void Swig_cresult_name_set(const char *new_name) {
  cresult_variable_name = new_name;
}

/* -----------------------------------------------------------------------------
 * Swig_cresult_name()
 *
 * Get the name of the variable used to hold the return value from C/C++ wrapper functions
 * ----------------------------------------------------------------------------- */

const char *Swig_cresult_name(void) {
  return cresult_variable_name;
}

/* -----------------------------------------------------------------------------
 * Swig_cparm_name()
 *
 * Generates a name for the ith argument in an argument list
 * ----------------------------------------------------------------------------- */

String *Swig_cparm_name(Parm *p, int i) {
  String *name = NewStringf("arg%d", i + 1);
  if (p) {
    Setattr(p, "lname", name);
  }

  return name;
}

/* -----------------------------------------------------------------------------
 * Swig_clocal()
 *
 * Creates a string that declares a C local variable type.  Converts references
 * and user defined types to pointers.
 * ----------------------------------------------------------------------------- */

static String *Swig_clocal(SwigType *t, const_String_or_char_ptr name, const_String_or_char_ptr value) {
  String *decl;

  decl = NewStringEmpty();

  switch (SwigType_type(t)) {
  case T_REFERENCE:
    if (value) {
      String *lstrname = SwigType_lstr(t, name);
      String *lstr = SwigType_lstr(t, 0);
      Printf(decl, "%s = (%s) &%s_defvalue", lstrname, lstr, name);
      Delete(lstrname);
      Delete(lstr);
    } else {
      String *lstrname = SwigType_lstr(t, name);
      Printf(decl, "%s = 0", lstrname);
      Delete(lstrname);
    }
    break;
  case T_RVALUE_REFERENCE:
    if (value) {
      String *lstrname = SwigType_lstr(t, name);
      String *lstr = SwigType_lstr(t, 0);
      Printf(decl, "%s = (%s) &%s_defrvalue", lstrname, lstr, name);
      Delete(lstrname);
      Delete(lstr);
    } else {
      String *lstrname = SwigType_lstr(t, name);
      Printf(decl, "%s = 0", lstrname);
      Delete(lstrname);
    }
    break;
  case T_VOID:
    break;
  case T_VARARGS:
    Printf(decl, "void *%s = 0", name);
    break;

  default:
    if (value) {
      String *lcaststr = SwigType_lcaststr(t, value);
      String *lstr = SwigType_lstr(t, 0);
      String *lstrn = SwigType_lstr(t, name);
      Printf(decl, "%s = (%s) %s", lstrn, lstr, lcaststr);
      Delete(lcaststr);
      Delete(lstr);
      Delete(lstrn);
    } else {
      String *lstrname = SwigType_lstr(t, name);
      Append(decl, lstrname);
      Delete(lstrname);
    }
  }
  return decl;
}

/* -----------------------------------------------------------------------------
 * Swig_wrapped_var_convert()
 *
 * Converts a member variable for use in the get and set wrapper methods.
 * This function only converts user defined types to pointers.
 * ----------------------------------------------------------------------------- */

String *Swig_wrapped_var_type(SwigType *t, int varcref) {
  SwigType *ty;

  if (!Strstr(t, "enum $unnamed")) {
    ty = Copy(t);
  } else {
    /* Change the type for unnamed enum instance variables */
    ty = NewString("int");
  }

  if (SwigType_isclass(t)) {
    if (varcref) {
      if (cparse_cplusplus) {
	if (!SwigType_isconst(ty))
	  SwigType_add_qualifier(ty, "const");
	SwigType_add_reference(ty);
      } else {
	return Copy(ty);
      }
    } else {
      SwigType_add_pointer(ty);
    }
  }
  return ty;
}

String *Swig_wrapped_member_var_type(SwigType *t, int varcref) {
  SwigType *ty;

  if (!Strstr(t, "enum $unnamed")) {
    ty = Copy(t);
  } else {
    /* Change the type for unnamed enum instance variables */
    ty = NewString("int");
  }
  if (SwigType_isclass(t)) {
    if (varcref) {
      if (cparse_cplusplus) {
	if (!SwigType_isconst(ty))
	  SwigType_add_qualifier(ty, "const");
	SwigType_add_reference(ty);
      } else {
	return Copy(ty);
      }
    } else {
      SwigType_add_pointer(ty);
    }
  }
  return ty;
}


static String *Swig_wrapped_var_deref(SwigType *t, const_String_or_char_ptr name, int varcref) {
  if (SwigType_isclass(t)) {
    if (varcref) {
      if (cparse_cplusplus) {
	return NewStringf("*%s", name);
      } else {
	return NewStringf("%s", name);
      }
    } else {
      return NewStringf("*%s", name);
    }
  } else {
    return SwigType_rcaststr(t, name);
  }
}

static String *Swig_wrapped_var_assign(SwigType *t, const_String_or_char_ptr name, int varcref) {
  if (SwigType_isclass(t)) {
    if (varcref) {
      return NewStringf("%s", name);
    } else {
      return NewStringf("&%s", name);
    }
  } else {
    return SwigType_lcaststr(t, name);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_cargs()
 *
 * Emit all of the local variables for a list of parameters.  Returns the
 * number of parameters.
 * Default values for the local variables are only emitted if the compact default
 * argument behaviour is required.
 * ----------------------------------------------------------------------------- */
int Swig_cargs(Wrapper *w, ParmList *p) {
  int i = 0;
  int compactdefargs = ParmList_is_compactdefargs(p);

  while (p != 0) {
    String *lname = Swig_cparm_name(p, i);
    SwigType *pt = Getattr(p, "type");
    if ((SwigType_type(pt) != T_VOID)) {
      String *local = 0;
      String *type = Getattr(p, "type");
      /* default values only emitted if in compact default args mode */
      String *pvalue = (compactdefargs) ? Getattr(p, "value") : 0;

      /* When using compactdefaultargs, the code generated initialises a variable via a constructor call that accepts the
       * default value as a parameter. The default constructor is not called and therefore SwigValueWrapper is not needed. */
      SwigType *altty = pvalue ? 0 : SwigType_alttype(type, 0);

      int tycode = SwigType_type(type);
      if (tycode == T_REFERENCE) {
	if (pvalue) {
	  SwigType *tvalue;
	  String *defname, *defvalue, *rvalue, *qvalue;
	  rvalue = SwigType_typedef_resolve_all(pvalue);
	  qvalue = SwigType_typedef_qualified(rvalue);
	  defname = NewStringf("%s_defvalue", lname);
	  tvalue = Copy(type);
	  SwigType_del_reference(tvalue);
	  tycode = SwigType_type(tvalue);
	  if (tycode != T_USER) {
	    /* plain primitive type, we copy the def value */
	    String *lstr = SwigType_lstr(tvalue, defname);
	    defvalue = NewStringf("%s = %s", lstr, qvalue);
	    Delete(lstr);
	  } else {
	    /* user type, we copy the reference value */
	    String *str = SwigType_str(type, defname);
	    defvalue = NewStringf("%s = %s", str, qvalue);
	    Delete(str);
	  }
	  Wrapper_add_localv(w, defname, defvalue, NIL);
	  Delete(tvalue);
	  Delete(rvalue);
	  Delete(qvalue);
	  Delete(defname);
	  Delete(defvalue);
	}
      } else if (tycode == T_RVALUE_REFERENCE) {
	if (pvalue) {
	  SwigType *tvalue;
	  String *defname, *defvalue, *rvalue, *qvalue;
	  rvalue = SwigType_typedef_resolve_all(pvalue);
	  qvalue = SwigType_typedef_qualified(rvalue);
	  defname = NewStringf("%s_defrvalue", lname);
	  tvalue = Copy(type);
	  SwigType_del_rvalue_reference(tvalue);
	  tycode = SwigType_type(tvalue);
	  if (tycode != T_USER) {
	    /* plain primitive type, we copy the the def value */
	    String *lstr = SwigType_lstr(tvalue, defname);
	    defvalue = NewStringf("%s = %s", lstr, qvalue);
	    Delete(lstr);
	  } else {
	    /* user type, we copy the reference value */
	    String *str = SwigType_str(type, defname);
	    defvalue = NewStringf("%s = %s", str, qvalue);
	    Delete(str);
	  }
	  Wrapper_add_localv(w, defname, defvalue, NIL);
	  Delete(tvalue);
	  Delete(rvalue);
	  Delete(qvalue);
	  Delete(defname);
	  Delete(defvalue);
	}
      } else if (!pvalue && ((tycode == T_POINTER) || (tycode == T_STRING) || (tycode == T_WSTRING))) {
	pvalue = (String *) "0";
      }
      if (!altty) {
	local = Swig_clocal(pt, lname, pvalue);
      } else {
	local = Swig_clocal(altty, lname, pvalue);
	Delete(altty);
      }
      Wrapper_add_localv(w, lname, local, NIL);
      Delete(local);
      i++;
    }
    Delete(lname);
    p = nextSibling(p);
  }
  return (i);
}

/* -----------------------------------------------------------------------------
 * Swig_cresult()
 *
 * This function generates the C code needed to set the result of a C
 * function call.  
 * ----------------------------------------------------------------------------- */

String *Swig_cresult(SwigType *t, const_String_or_char_ptr name, const_String_or_char_ptr decl) {
  String *fcall;

  fcall = NewStringEmpty();
  switch (SwigType_type(t)) {
  case T_VOID:
    break;
  case T_REFERENCE:
    {
      String *lstr = SwigType_lstr(t, 0);
      Printf(fcall, "%s = (%s) &", name, lstr);
      Delete(lstr);
    }
    break;
  case T_RVALUE_REFERENCE:
    {
      String *const_lvalue_str;
      String *lstr = SwigType_lstr(t, 0);
      SwigType *tt = Copy(t);
      SwigType_del_rvalue_reference(tt);
      SwigType_add_qualifier(tt, "const");
      SwigType_add_reference(tt);
      const_lvalue_str = SwigType_rcaststr(tt, 0);

      Printf(fcall, "%s = (%s) &%s", name, lstr, const_lvalue_str);

      Delete(const_lvalue_str);
      Delete(tt);
      Delete(lstr);
    }
    break;
  case T_USER:
    Printf(fcall, "%s = ", name);
    break;

  default:
    /* Normal return value */
    {
      String *lstr = SwigType_lstr(t, 0);
      Printf(fcall, "%s = (%s)", name, lstr);
      Delete(lstr);
    }
    break;
  }

  /* Now print out function call */
  Append(fcall, decl);

  /* A sick hack */
  {
    char *c = Char(decl) + Len(decl) - 1;
    if (!((*c == ';') || (*c == '}')))
      Append(fcall, ";");
  }

  return fcall;
}

/* -----------------------------------------------------------------------------
 * Swig_cfunction_call()
 *
 * Creates a string that calls a C function using the local variable rules
 * defined above.
 *
 *    name(arg0, arg1, arg2, ... argn)
 *
 * ----------------------------------------------------------------------------- */

String *Swig_cfunction_call(const_String_or_char_ptr name, ParmList *parms) {
  String *func;
  int i = 0;
  int comma = 0;
  Parm *p = parms;
  String *nname;

  func = NewStringEmpty();
  nname = SwigType_namestr(name);

  /*
     SWIGTEMPLATEDISAMBIGUATOR is compiler dependent (swiglabels.swg),
     - SUN Studio 9 requires 'template', 
     - gcc-3.4 forbids the use of 'template'.
     the rest seems not caring very much,
   */
  if (SwigType_istemplate(name)) {
    String *prefix = Swig_scopename_prefix(nname);
    if (!prefix || Len(prefix) == 0) {
      Printf(func, "%s(", nname);
    } else {
      String *last = Swig_scopename_last(nname);
      Printf(func, "%s::SWIGTEMPLATEDISAMBIGUATOR %s(", prefix, last);
      Delete(last);
    }
    Delete(prefix);
  } else {
    Printf(func, "%s(", nname);
  }
  Delete(nname);

  while (p) {
    SwigType *pt = Getattr(p, "type");
    if ((SwigType_type(pt) != T_VOID)) {
      SwigType *rpt = SwigType_typedef_resolve_all(pt);
      String *pname = Swig_cparm_name(p, i);
      String *rcaststr = SwigType_rcaststr(rpt, pname);

      if (comma) {
	Printv(func, ",", rcaststr, NIL);
      } else {
	Append(func, rcaststr);
      }
      Delete(rpt);
      Delete(pname);
      Delete(rcaststr);
      comma = 1;
      i++;
    }
    p = nextSibling(p);
  }
  Append(func, ")");
  return func;
}

/* -----------------------------------------------------------------------------
 * Swig_cmethod_call()
 *
 * Generates a string that calls a C++ method from a list of parameters.
 * 
 *    arg0->name(arg1, arg2, arg3, ..., argn)
 *
 * self is an argument that defines how to handle the first argument. Normally,
 * it should be set to "this->".  With C++ proxy classes enabled, it could be
 * set to "(*this)->" or some similar sequence.
 * ----------------------------------------------------------------------------- */

static String *Swig_cmethod_call(const_String_or_char_ptr name, ParmList *parms, const_String_or_char_ptr self, String *explicit_qualifier, SwigType *director_type) {
  String *func, *nname;
  int i = 0;
  Parm *p = parms;
  SwigType *pt;
  int comma = 0;

  func = NewStringEmpty();
  if (!p)
    return func;

  if (!self)
    self = "(this)->";
  Append(func, self);

  if (SwigType_istemplate(name) && (strncmp(Char(name), "operator ", 9) == 0)) {
    /* fix for template + operators and compilers like gcc 3.3.5 */
    String *tprefix = SwigType_templateprefix(name);
    nname = tprefix;
  } else {
    nname = SwigType_namestr(name);
  }

  if (director_type) {
    const char *pname = "darg";
    String *rcaststr = SwigType_rcaststr(director_type, pname);
    Replaceall(func, "this", rcaststr);
    Delete(rcaststr);
  } else {
    pt = Getattr(p, "type");

    /* If the method is invoked through a dereferenced pointer, we don't add any casts
       (needed for smart pointers).  Otherwise, we cast to the appropriate type */

    if (Strstr(func, "*this")) {
      String *pname = Swig_cparm_name(p, 0);
      Replaceall(func, "this", pname);
      Delete(pname);
    } else {
      String *pname = Swig_cparm_name(p, 0);
      String *rcaststr = SwigType_rcaststr(pt, pname);
      Replaceall(func, "this", rcaststr);
      Delete(rcaststr);
      Delete(pname);
    }

    /*
       SWIGTEMPLATEDESIMBUAGATOR is compiler dependent (swiglabels.swg),
       - SUN Studio 9 requires 'template', 
       - gcc-3.4 forbids the use of 'template' (correctly implementing the ISO C++ standard)
       the others don't seem to care,
     */
    if (SwigType_istemplate(name))
      Printf(func, "SWIGTEMPLATEDISAMBIGUATOR ");

    if (explicit_qualifier) {
      Printv(func, explicit_qualifier, "::", NIL);
    }
  }

  Printf(func, "%s(", nname);

  i++;
  p = nextSibling(p);
  while (p) {
    pt = Getattr(p, "type");
    if ((SwigType_type(pt) != T_VOID)) {
      String *pname = Swig_cparm_name(p, i);
      String *rcaststr = SwigType_rcaststr(pt, pname);
      if (comma)
	Append(func, ",");
      Append(func, rcaststr);
      Delete(rcaststr);
      Delete(pname);
      comma = 1;
      i++;
    }
    p = nextSibling(p);
  }
  Append(func, ")");
  Delete(nname);
  return func;
}

/* -----------------------------------------------------------------------------
 * Swig_cconstructor_call()
 *
 * Creates a string that calls a C constructor function.
 *
 *      calloc(1,sizeof(name));
 * ----------------------------------------------------------------------------- */

String *Swig_cconstructor_call(const_String_or_char_ptr name) {
  DOH *func;

  func = NewStringEmpty();
  Printf(func, "calloc(1, sizeof(%s))", name);
  return func;
}


/* -----------------------------------------------------------------------------
 * Swig_cppconstructor_call()
 *
 * Creates a string that calls a C function using the local variable rules
 * defined above.
 *
 *    name(arg0, arg1, arg2, ... argn)
 *
 * ----------------------------------------------------------------------------- */

String *Swig_cppconstructor_base_call(const_String_or_char_ptr name, ParmList *parms, int skip_self) {
  String *func;
  String *nname;
  int i = 0;
  int comma = 0;
  Parm *p = parms;
  SwigType *pt;
  if (skip_self) {
    if (p)
      p = nextSibling(p);
    i++;
  }
  nname = SwigType_namestr(name);
  func = NewStringEmpty();
  Printf(func, "new %s(", nname);
  while (p) {
    pt = Getattr(p, "type");
    if ((SwigType_type(pt) != T_VOID)) {
      String *rcaststr = 0;
      String *pname = 0;
      if (comma)
	Append(func, ",");
      if (!Getattr(p, "arg:byname")) {
	pname = Swig_cparm_name(p, i);
	i++;
      } else {
        pname = Getattr(p, "value");
	if (pname)
	  pname = Copy(pname);
	else
	  pname = Copy(Getattr(p, "name"));
      }
      rcaststr = SwigType_rcaststr(pt, pname);
      Append(func, rcaststr);
      Delete(rcaststr);
      comma = 1;
      Delete(pname);
    }
    p = nextSibling(p);
  }
  Append(func, ")");
  Delete(nname);
  return func;
}

String *Swig_cppconstructor_call(const_String_or_char_ptr name, ParmList *parms) {
  return Swig_cppconstructor_base_call(name, parms, 0);
}

String *Swig_cppconstructor_nodirector_call(const_String_or_char_ptr name, ParmList *parms) {
  return Swig_cppconstructor_base_call(name, parms, 1);
}

String *Swig_cppconstructor_director_call(const_String_or_char_ptr name, ParmList *parms) {
  return Swig_cppconstructor_base_call(name, parms, 0);
}

/* -----------------------------------------------------------------------------
 * recursive_flag_search()
 *
 * This function searches for the class attribute 'attr' in the class
 * 'n' or recursively in its bases.
 *
 * If you define SWIG_FAST_REC_SEARCH, the method will set the found
 * 'attr' in the target class 'n'. If not, the method will set the
 * 'noattr' one. This prevents of having to navigate the entire
 * hierarchy tree everytime, so, it is an O(1) method...  or something
 * like that. However, it populates all the parsed classes with the
 * 'attr' and/or 'noattr' attributes.
 *
 * If you undefine the SWIG_FAST_REC_SEARCH no attribute will be set
 * while searching. This could be slower for large projects with very
 * large hierarchy trees... or maybe not. But it will be cleaner. 
 *
 * Maybe later a swig option can be added to switch at runtime.
 *
 * ----------------------------------------------------------------------------- */

/* #define SWIG_FAST_REC_SEARCH 1 */
static String *recursive_flag_search(Node *n, const String *attr, const String *noattr) {
  String *f = 0;
  n = Swig_methodclass(n);
  if (GetFlag(n, noattr)) {
    return 0;
  }
  f = GetFlagAttr(n, attr);
  if (f) {
    return f;
  } else {
    List *bl = Getattr(n, "bases");
    if (bl) {
      Iterator bi;
      for (bi = First(bl); bi.item; bi = Next(bi)) {
	f = recursive_flag_search(bi.item, attr, noattr);
	if (f) {
#ifdef SWIG_FAST_REC_SEARCH
	  SetFlagAttr(n, attr, f);
#endif
	  return f;
	}
      }
    }
  }
#ifdef SWIG_FAST_REC_SEARCH
  SetFlag(n, noattr);
#endif
  return 0;
}

/* -----------------------------------------------------------------------------
 * Swig_unref_call()
 *
 * Find the "feature:unref" call, if any.
 * ----------------------------------------------------------------------------- */

String *Swig_unref_call(Node *n) {
  String *unref = recursive_flag_search(n, "feature:unref", "feature:nounref");
  if (unref) {
    String *pname = Swig_cparm_name(0, 0);
    unref = NewString(unref);
    Replaceall(unref, "$this", pname);
    Replaceall(unref, "$self", pname);
    Delete(pname);
  }
  return unref;
}

/* -----------------------------------------------------------------------------
 * Swig_ref_call()
 *
 * Find the "feature:ref" call, if any.
 * ----------------------------------------------------------------------------- */

String *Swig_ref_call(Node *n, const String *lname) {
  String *ref = recursive_flag_search(n, "feature:ref", "feature:noref");
  if (ref) {
    ref = NewString(ref);
    Replaceall(ref, "$this", lname);
    Replaceall(ref, "$self", lname);
  }
  return ref;
}

/* -----------------------------------------------------------------------------
 * Swig_cdestructor_call()
 *
 * Creates a string that calls a C destructor function.
 *
 *      free((char *) arg0);
 * ----------------------------------------------------------------------------- */

String *Swig_cdestructor_call(Node *n) {
  Node *cn = Swig_methodclass(n);
  String *unref = Swig_unref_call(cn);

  if (unref) {
    return unref;
  } else {
    String *pname = Swig_cparm_name(0, 0);
    String *call = NewStringf("free((char *) %s);", pname);
    Delete(pname);
    return call;
  }
}


/* -----------------------------------------------------------------------------
 * Swig_cppdestructor_call()
 *
 * Creates a string that calls a C destructor function.
 *
 *      delete arg1;
 * ----------------------------------------------------------------------------- */

String *Swig_cppdestructor_call(Node *n) {
  Node *cn = Swig_methodclass(n);
  String *unref = Swig_unref_call(cn);
  if (unref) {
    return unref;
  } else {
    String *pname = Swig_cparm_name(0, 0);
    String *call = NewStringf("delete %s;", pname);
    Delete(pname);
    return call;
  }
}

/* -----------------------------------------------------------------------------
 * Swig_cmemberset_call()
 *
 * Generates a string that sets the name of a member in a C++ class or C struct.
 *
 *        arg0->name = arg1
 *
 * ----------------------------------------------------------------------------- */

String *Swig_cmemberset_call(const_String_or_char_ptr name, SwigType *type, String *self, int varcref) {
  String *func;
  String *pname0 = Swig_cparm_name(0, 0);
  String *pname1 = Swig_cparm_name(0, 1);
  func = NewStringEmpty();
  if (!self)
    self = NewString("(this)->");
  else
    self = NewString(self);
  Replaceall(self, "this", pname0);
  if (SwigType_type(type) != T_ARRAY) {
    if (!Strstr(type, "enum $unnamed")) {
      String *dref = Swig_wrapped_var_deref(type, pname1, varcref);
      int extra_cast = 0;
      if (cparse_cplusplusout) {
	/* Required for C nested structs compiled as C++ as a duplicate of the nested struct is put into the global namespace.
	 * We could improve this by adding the extra casts just for nested structs rather than all structs. */
	String *base = SwigType_base(type);
	extra_cast = SwigType_isclass(base);
	Delete(base);
      }
      if (extra_cast) {
	String *lstr;
	SwigType *ptype = Copy(type);
	SwigType_add_pointer(ptype);
	lstr = SwigType_lstr(ptype, 0);
	Printf(func, "if (%s) *(%s)&%s%s = %s", pname0, lstr, self, name, dref);
	Delete(lstr);
	Delete(ptype);
      } else {
        Printf(func, "if (%s) %s%s = %s", pname0, self, name, dref);
      }
      Delete(dref);
    } else {
      Printf(func, "if (%s && sizeof(int) == sizeof(%s%s)) *(int*)(void*)&(%s%s) = %s", pname0, self, name, self, name, pname1);
    }
  }
  Delete(self);
  Delete(pname0);
  Delete(pname1);
  return (func);
}


/* -----------------------------------------------------------------------------
 * Swig_cmemberget_call()
 *
 * Generates a string that sets the name of a member in a C++ class or C struct.
 *
 *        arg0->name
 *
 * ----------------------------------------------------------------------------- */

String *Swig_cmemberget_call(const_String_or_char_ptr name, SwigType *t, String *self, int varcref) {
  String *func;
  String *call;
  String *pname0 = Swig_cparm_name(0, 0);
  if (!self)
    self = NewString("(this)->");
  else
    self = NewString(self);
  Replaceall(self, "this", pname0);
  func = NewStringEmpty();
  call = Swig_wrapped_var_assign(t, "", varcref);
  Printf(func, "%s (%s%s)", call, self, name);
  Delete(self);
  Delete(call);
  Delete(pname0);
  return func;
}

/* -----------------------------------------------------------------------------
 * Swig_replace_special_variables()
 *
 * Replaces special variables with a value from the supplied node
 * ----------------------------------------------------------------------------- */
void Swig_replace_special_variables(Node *n, Node *parentnode, String *code) {
  Node *parentclass = parentnode;
  String *overloaded = Getattr(n, "sym:overloaded");
  Replaceall(code, "$name", Getattr(n, "name"));
  Replaceall(code, "$symname", Getattr(n, "sym:name"));
  Replaceall(code, "$wrapname", Getattr(n, "wrap:name"));
  Replaceall(code, "$overname", overloaded ? Char(Getattr(n, "sym:overname")) : "");

  if (Strstr(code, "$decl")) {
    String *decl = Swig_name_decl(n);
    Replaceall(code, "$decl", decl);
    Delete(decl);
  }
  if (Strstr(code, "$fulldecl")) {
    String *fulldecl = Swig_name_fulldecl(n);
    Replaceall(code, "$fulldecl", fulldecl);
    Delete(fulldecl);
  }

  if (parentclass && !Equal(nodeType(parentclass), "class"))
    parentclass = 0;
  if (Strstr(code, "$parentclasssymname")) {
    String *parentclasssymname = 0;
    if (parentclass)
      parentclasssymname = Getattr(parentclass, "sym:name");
    Replaceall(code, "$parentclasssymname", parentclasssymname ? parentclasssymname : "");
  }
  if (Strstr(code, "$parentclassname")) {
    String *parentclassname = 0;
    if (parentclass)
      parentclassname = Getattr(parentclass, "name");
    Replaceall(code, "$parentclassname", parentclassname ? SwigType_str(parentclassname, "") : "");
  }
}

/* -----------------------------------------------------------------------------
 * extension_code()
 *
 * Generates an extension function (a function defined in %extend)
 *
 *        return_type function_name(parms) code
 *
 * ----------------------------------------------------------------------------- */
static String *extension_code(Node *n, const String *function_name, ParmList *parms, SwigType *return_type, const String *code, int cplusplus, const String *self) {
  String *parms_str = cplusplus ? ParmList_str_defaultargs(parms) : ParmList_str(parms);
  String *sig = NewStringf("%s(%s)", function_name, (cplusplus || Len(parms_str)) ? parms_str : "void");
  String *rt_sig = SwigType_str(return_type, sig);
  String *body = NewStringf("SWIGINTERN %s", rt_sig);
  Printv(body, code, "\n", NIL);
  if (Strstr(body, "$")) {
    Swig_replace_special_variables(n, parentNode(parentNode(n)), body);
    if (self)
      Replaceall(body, "$self", self);
  }
  Delete(parms_str);
  Delete(sig);
  Delete(rt_sig);
  return body;
}

/* -----------------------------------------------------------------------------
 * Swig_add_extension_code()
 *
 * Generates an extension function (a function defined in %extend) and
 * adds it to the "wrap:code" attribute of a node
 *
 * See also extension_code()
 *
 * ----------------------------------------------------------------------------- */
int Swig_add_extension_code(Node *n, const String *function_name, ParmList *parms, SwigType *return_type, const String *code, int cplusplus, const String *self) {
  String *body = extension_code(n, function_name, parms, return_type, code, cplusplus, self);
  Setattr(n, "wrap:code", body);
  Delete(body);
  return SWIG_OK;
}


/* -----------------------------------------------------------------------------
 * Swig_MethodToFunction(Node *n)
 *
 * Converts a C++ method node to a function accessor function.
 * ----------------------------------------------------------------------------- */

int Swig_MethodToFunction(Node *n, const_String_or_char_ptr nspace, String *classname, int flags, SwigType *director_type, int is_director) {
  String *name;
  ParmList *parms;
  SwigType *type;
  Parm *p;
  String *self = 0;
  int is_smart_pointer_overload = 0;
  String *qualifier = Getattr(n, "qualifier");
  String *directorScope = NewString(nspace);

  Replace(directorScope, NSPACE_SEPARATOR, "_", DOH_REPLACE_ANY);
  
  /* If smart pointer without const overload or mutable method, change self dereferencing */
  if (flags & CWRAP_SMART_POINTER) {
    if (flags & CWRAP_SMART_POINTER_OVERLOAD) {
      if (qualifier && strncmp(Char(qualifier), "q(const)", 8) == 0) {
        self = NewString("(*(this))->");
        is_smart_pointer_overload = 1;
      }
      else if (Swig_storage_isstatic(n)) {
	String *cname = Getattr(n, "extendsmartclassname") ? Getattr(n, "extendsmartclassname") : classname;
	String *ctname = SwigType_namestr(cname);
        self = NewStringf("(*(%s const *)this)->", ctname);
        is_smart_pointer_overload = 1;
	Delete(ctname);
      }
      else {
        self = NewString("(*this)->");
      }
    } else {
      self = NewString("(*this)->");
    }
  }

  /* If node is a member template expansion, we don't allow added code */
  if (Getattr(n, "templatetype"))
    flags &= ~(CWRAP_EXTEND);

  name = Getattr(n, "name");
  parms = CopyParmList(nonvoid_parms(Getattr(n, "parms")));

  type = NewString(classname);
  if (qualifier) {
    SwigType_push(type, qualifier);
  }
  SwigType_add_pointer(type);
  p = NewParm(type, "self", n);
  Setattr(p, "self", "1");
  Setattr(p, "hidden","1");
  /*
     Disable the 'this' ownership in 'self' to manage inplace
     operations like:

     A& A::operator+=(int i) { ...; return *this;}

     Here the 'self' parameter ownership needs to be disabled since
     there could be two objects sharing the same 'this' pointer: the
     input and the result one. And worse, the pointer could be deleted
     in one of the objects (input), leaving the other (output) with
     just a seg. fault to happen.

     To avoid the previous problem, use

     %feature("self:disown") *::operator+=;
     %feature("new") *::operator+=;

     These two lines just transfer the ownership of the 'this' pointer
     from the input to the output wrapping object.

     This happens in python, but may also happen in other target
     languages.
   */
  if (GetFlag(n, "feature:self:disown")) {
    Setattr(p, "wrap:disown", "1");
  }
  set_nextSibling(p, parms);
  Delete(type);

  /* Generate action code for the access */
  if (!(flags & CWRAP_EXTEND)) {
    String *explicit_qualifier = 0;
    String *call = 0;
    String *cres = 0;
    String *explicitcall_name = 0;
    int pure_virtual = !(Cmp(Getattr(n, "storage"), "virtual")) && !(Cmp(Getattr(n, "value"), "0"));

    /* Call the explicit method rather than allow for a polymorphic call */
    if ((flags & CWRAP_DIRECTOR_TWO_CALLS) || (flags & CWRAP_DIRECTOR_ONE_CALL)) {
      String *access = Getattr(n, "access");
      if (access && (Cmp(access, "protected") == 0)) {
	/* If protected access (can only be if a director method) then call the extra public accessor method (language module must provide this) */
	String *explicit_qualifier_tmp = SwigType_namestr(Getattr(Getattr(parentNode(n), "typescope"), "qname"));
	explicitcall_name = NewStringf("%sSwigPublic", name);
        if (Len(directorScope) > 0)
	  explicit_qualifier = NewStringf("SwigDirector_%s_%s", directorScope, explicit_qualifier_tmp);
        else
	  explicit_qualifier = NewStringf("SwigDirector_%s", explicit_qualifier_tmp);
	Delete(explicit_qualifier_tmp);
      } else {
	explicit_qualifier = SwigType_namestr(Getattr(Getattr(parentNode(n), "typescope"), "qname"));
      }
    }

    call = Swig_cmethod_call(explicitcall_name ? explicitcall_name : name, p, self, explicit_qualifier, director_type);
    cres = Swig_cresult(Getattr(n, "type"), Swig_cresult_name(), call);

    if (pure_virtual && is_director && (flags & CWRAP_DIRECTOR_TWO_CALLS)) {
      String *qualifier = SwigType_namestr(Getattr(Getattr(parentNode(n), "typescope"), "qname"));
      Delete(cres);
      cres = NewStringf("Swig::DirectorPureVirtualException::raise(\"%s::%s\");", qualifier, name);
      Delete(qualifier);
    }

    if (flags & CWRAP_DIRECTOR_TWO_CALLS) {
      /* Create two method calls, one to call the explicit method, the other a normal polymorphic function call */
      String *cres_both_calls = NewStringf("");
      String *call_extra = Swig_cmethod_call(name, p, self, 0, director_type);
      String *cres_extra = Swig_cresult(Getattr(n, "type"), Swig_cresult_name(), call_extra);
      Printv(cres_both_calls, "if (upcall) {\n", cres, "\n", "} else {", cres_extra, "\n}", NIL);
      Setattr(n, "wrap:action", cres_both_calls);
      Delete(cres_extra);
      Delete(call_extra);
      Delete(cres_both_calls);
    } else {
      Setattr(n, "wrap:action", cres);
    }

    Delete(explicitcall_name);
    Delete(call);
    Delete(cres);
    Delete(explicit_qualifier);
  } else {
    /* Methods with default arguments are wrapped with additional methods for each default argument,
     * however, only one extra %extend method is generated. */

    String *defaultargs = Getattr(n, "defaultargs");
    String *code = Getattr(n, "code");
    String *cname = Getattr(n, "extendsmartclassname") ? Getattr(n, "extendsmartclassname") : classname;
    String *membername = Swig_name_member(nspace, cname, name);
    String *mangled = Swig_name_mangle(membername);
    int is_smart_pointer = flags & CWRAP_SMART_POINTER;

    type = Getattr(n, "type");

    /* Check if the method is overloaded.   If so, and it has code attached, we append an extra suffix
       to avoid a name-clash in the generated wrappers.  This allows overloaded methods to be defined
       in C. */
    if (Getattr(n, "sym:overloaded") && code) {
      Append(mangled, Getattr(defaultargs ? defaultargs : n, "sym:overname"));
    }

    /* See if there is any code that we need to emit */
    if (!defaultargs && code && !is_smart_pointer) {
      Swig_add_extension_code(n, mangled, p, type, code, cparse_cplusplus, "self");
    }
    if (is_smart_pointer) {
      int i = 0;
      Parm *pp = p;
      String *func = NewStringf("%s(", mangled);
      String *cres;

      if (!Swig_storage_isstatic(n)) {
	String *pname = Swig_cparm_name(pp, i);
	String *ctname = SwigType_namestr(cname);
	String *fadd = 0;
	if (is_smart_pointer_overload) {
	  String *nclassname = SwigType_namestr(classname);
	  fadd = NewStringf("(%s const *)((%s const *)%s)->operator ->()", ctname, nclassname, pname);
	  Delete(nclassname);
	}
	else {
	  fadd = NewStringf("(%s*)(%s)->operator ->()", ctname, pname);
	}
	Append(func, fadd);
	Delete(ctname);
	Delete(fadd);
	Delete(pname);
	pp = nextSibling(pp);
	if (pp)
	  Append(func, ",");
      } else {
	pp = nextSibling(pp);
      }
      ++i;
      while (pp) {
	SwigType *pt = Getattr(pp, "type");
	if ((SwigType_type(pt) != T_VOID)) {
	  String *pname = Swig_cparm_name(pp, i++);
	  String *rcaststr = SwigType_rcaststr(pt, pname);
	  Append(func, rcaststr);
	  Delete(rcaststr);
	  Delete(pname);
	  pp = nextSibling(pp);
	  if (pp)
	    Append(func, ",");
	}
      }
      Append(func, ")");
      cres = Swig_cresult(Getattr(n, "type"), Swig_cresult_name(), func);
      Setattr(n, "wrap:action", cres);
      Delete(cres);
    } else {
      String *call = Swig_cfunction_call(mangled, p);
      String *cres = Swig_cresult(Getattr(n, "type"), Swig_cresult_name(), call);
      Setattr(n, "wrap:action", cres);
      Delete(call);
      Delete(cres);
    }

    Delete(membername);
    Delete(mangled);
  }
  Setattr(n, "parms", p);
  Delete(p);
  Delete(self);
  Delete(parms);
  Delete(directorScope);
  return SWIG_OK;
}

/* -----------------------------------------------------------------------------
 * Swig_methodclass()
 *
 * This function returns the class node for a given method or class.
 * ----------------------------------------------------------------------------- */

Node *Swig_methodclass(Node *n) {
  Node *nodetype = nodeType(n);
  if (Cmp(nodetype, "class") == 0)
    return n;
  return GetFlag(n, "feature:extend") ? parentNode(parentNode(n)) : parentNode(n);
}

int Swig_directorclass(Node *n) {
  Node *classNode = Swig_methodclass(n);
  assert(classNode != 0);
  return (Getattr(classNode, "vtable") != 0);
}

Node *Swig_directormap(Node *module, String *type) {
  int is_void = !Cmp(type, "void");
  if (!is_void && module) {
    /* ?? follow the inheritance hierarchy? */

    String *base = SwigType_base(type);

    Node *directormap = Getattr(module, "wrap:directormap");
    if (directormap)
      return Getattr(directormap, base);
  }
  return 0;
}


/* -----------------------------------------------------------------------------
 * Swig_ConstructorToFunction()
 *
 * This function creates a C wrapper for a C constructor function. 
 * ----------------------------------------------------------------------------- */

int Swig_ConstructorToFunction(Node *n, const_String_or_char_ptr nspace, String *classname, String *none_comparison, String *director_ctor, int cplus, int flags, String *directorname) {
  Parm *p;
  ParmList *directorparms;
  SwigType *type;
  int use_director = Swig_directorclass(n);
  ParmList *parms = CopyParmList(nonvoid_parms(Getattr(n, "parms")));
  /* Prepend the list of prefix_args (if any) */
  Parm *prefix_args = Getattr(n, "director:prefix_args");
  if (prefix_args != NIL) {
    Parm *p2, *p3;

    directorparms = CopyParmList(prefix_args);
    for (p = directorparms; nextSibling(p); p = nextSibling(p));
    for (p2 = parms; p2; p2 = nextSibling(p2)) {
      p3 = CopyParm(p2);
      set_nextSibling(p, p3);
      Delete(p3);
      p = p3;
    }
  } else
    directorparms = parms;

  type = NewString(classname);
  SwigType_add_pointer(type);

  if (flags & CWRAP_EXTEND) {
    /* Constructors with default arguments are wrapped with additional constructor methods for each default argument,
     * however, only one extra %extend method is generated. */
    String *call;
    String *cres;
    String *defaultargs = Getattr(n, "defaultargs");
    String *code = Getattr(n, "code");
    String *membername = Swig_name_construct(nspace, classname);
    String *mangled = Swig_name_mangle(membername);

    /* Check if the constructor is overloaded.   If so, and it has code attached, we append an extra suffix
       to avoid a name-clash in the generated wrappers.  This allows overloaded constructors to be defined
       in C. */
    if (Getattr(n, "sym:overloaded") && code) {
      Append(mangled, Getattr(defaultargs ? defaultargs : n, "sym:overname"));
    }

    /* See if there is any code that we need to emit */
    if (!defaultargs && code) {
      Swig_add_extension_code(n, mangled, parms, type, code, cparse_cplusplus, "self");
    }

    call = Swig_cfunction_call(mangled, parms);
    cres = Swig_cresult(type, Swig_cresult_name(), call);
    Setattr(n, "wrap:action", cres);
    Delete(cres);
    Delete(call);
    Delete(membername);
    Delete(mangled);
  } else {
    if (cplus) {
      /* if a C++ director class exists, create it rather than the original class */
      if (use_director) {
	Node *parent = Swig_methodclass(n);
	int abstract = Getattr(parent, "abstracts") != 0;
	String *action = NewStringEmpty();
	String *tmp_none_comparison = Copy(none_comparison);
	String *director_call;
	String *nodirector_call;

	Replaceall(tmp_none_comparison, "$arg", "arg1");

	director_call = Swig_cppconstructor_director_call(directorname, directorparms);
	nodirector_call = Swig_cppconstructor_nodirector_call(classname, parms);

	if (abstract) {
	  /* whether or not the abstract class has been subclassed in python,
	   * create a director instance (there's no way to create a normal
	   * instance).  if any of the pure virtual methods haven't been
	   * implemented in the target language, calls to those methods will
	   * generate Swig::DirectorPureVirtualException exceptions.
	   */
	  String *cres = Swig_cresult(type, Swig_cresult_name(), director_call);
	  Append(action, cres);
	  Delete(cres);
	} else {
	  /* (scottm): The code for creating a new director is now a string
	     template that gets passed in via the director_ctor argument.

	     $comparison : an 'if' comparison from none_comparison
	     $director_new: Call new for director class
	     $nondirector_new: Call new for non-director class
	   */
	  String *cres;
	  Append(action, director_ctor);
	  Replaceall(action, "$comparison", tmp_none_comparison);

	  cres = Swig_cresult(type, Swig_cresult_name(), director_call);
	  Replaceall(action, "$director_new", cres);
	  Delete(cres);

	  cres = Swig_cresult(type, Swig_cresult_name(), nodirector_call);
	  Replaceall(action, "$nondirector_new", cres);
	  Delete(cres);
	}
	Setattr(n, "wrap:action", action);
	Delete(tmp_none_comparison);
	Delete(action);
      } else {
	String *call = Swig_cppconstructor_call(classname, parms);
	String *cres = Swig_cresult(type, Swig_cresult_name(), call);
	Setattr(n, "wrap:action", cres);
	Delete(cres);
	Delete(call);
      }
    } else {
      String *call = Swig_cconstructor_call(classname);
      String *cres = Swig_cresult(type, Swig_cresult_name(), call);
      Setattr(n, "wrap:action", cres);
      Delete(cres);
      Delete(call);
    }
  }
  Setattr(n, "type", type);
  Setattr(n, "parms", parms);
  Delete(type);
  if (directorparms != parms)
    Delete(directorparms);
  Delete(parms);
  return SWIG_OK;
}

/* -----------------------------------------------------------------------------
 * Swig_DestructorToFunction()
 *
 * This function creates a C wrapper for a destructor function.
 * ----------------------------------------------------------------------------- */

int Swig_DestructorToFunction(Node *n, const_String_or_char_ptr nspace, String *classname, int cplus, int flags) {
  SwigType *type;
  Parm *p;

  type = NewString(classname);
  SwigType_add_pointer(type);
  p = NewParm(type, "self", n);
  Setattr(p, "self", "1");
  Setattr(p, "hidden", "1");
  Setattr(p, "wrap:disown", "1");
  Delete(type);
  type = NewString("void");

  if (flags & CWRAP_EXTEND) {
    String *cres;
    String *call;
    String *membername, *mangled, *code;
    membername = Swig_name_destroy(nspace, classname);
    mangled = Swig_name_mangle(membername);
    code = Getattr(n, "code");
    if (code) {
      Swig_add_extension_code(n, mangled, p, type, code, cparse_cplusplus, "self");
    }
    call = Swig_cfunction_call(mangled, p);
    cres = NewStringf("%s;", call);
    Setattr(n, "wrap:action", cres);
    Delete(membername);
    Delete(mangled);
    Delete(call);
    Delete(cres);
  } else {
    if (cplus) {
      String *call = Swig_cppdestructor_call(n);
      String *cres = NewStringf("%s", call);
      Setattr(n, "wrap:action", cres);
      Delete(call);
      Delete(cres);
    } else {
      String *call = Swig_cdestructor_call(n);
      String *cres = NewStringf("%s", call);
      Setattr(n, "wrap:action", cres);
      Delete(call);
      Delete(cres);
    }
  }
  Setattr(n, "type", type);
  Setattr(n, "parms", p);
  Delete(type);
  Delete(p);
  return SWIG_OK;
}

/* -----------------------------------------------------------------------------
 * Swig_MembersetToFunction()
 *
 * This function creates a C wrapper for setting a structure member.
 * ----------------------------------------------------------------------------- */

int Swig_MembersetToFunction(Node *n, String *classname, int flags) {
  String *name;
  ParmList *parms;
  Parm *p;
  SwigType *t;
  SwigType *ty;
  SwigType *type;
  SwigType *void_type = NewString("void");
  String *self = 0;

  int varcref = flags & CWRAP_NATURAL_VAR;

  if (flags & CWRAP_SMART_POINTER) {
    self = NewString("(*this)->");
  }
  if (flags & CWRAP_ALL_PROTECTED_ACCESS) {
    self = NewStringf("darg->");
  }

  name = Getattr(n, "name");
  type = Getattr(n, "type");

  t = NewString(classname);
  SwigType_add_pointer(t);
  parms = NewParm(t, "self", n);
  Setattr(parms, "self", "1");
  Setattr(parms, "hidden","1");
  Delete(t);

  ty = Swig_wrapped_member_var_type(type, varcref);
  p = NewParm(ty, name, n);
  Setattr(parms, "hidden", "1");
  set_nextSibling(parms, p);

  /* If the type is a pointer or reference.  We mark it with a special wrap:disown attribute */
  if (SwigType_check_decl(type, "p.")) {
    Setattr(p, "wrap:disown", "1");
  }
  Delete(p);

  if (flags & CWRAP_EXTEND) {
    String *call;
    String *cres;
    String *code = Getattr(n, "code");

    String *sname = Swig_name_set(0, name);
    String *membername = Swig_name_member(0, classname, sname);
    String *mangled = Swig_name_mangle(membername);

    if (code) {
      /* I don't think this ever gets run - WSF */
      Swig_add_extension_code(n, mangled, parms, void_type, code, cparse_cplusplus, "self");
    }
    call = Swig_cfunction_call(mangled, parms);
    cres = NewStringf("%s;", call);
    Setattr(n, "wrap:action", cres);

    Delete(cres);
    Delete(call);
    Delete(mangled);
    Delete(membername);
    Delete(sname);
  } else {
    String *call = Swig_cmemberset_call(name, type, self, varcref);
    String *cres = NewStringf("%s;", call);
    Setattr(n, "wrap:action", cres);
    Delete(call);
    Delete(cres);
  }
  Setattr(n, "type", void_type);
  Setattr(n, "parms", parms);
  Delete(parms);
  Delete(ty);
  Delete(void_type);
  Delete(self);
  return SWIG_OK;
}

/* -----------------------------------------------------------------------------
 * Swig_MembergetToFunction()
 *
 * This function creates a C wrapper for getting a structure member.
 * ----------------------------------------------------------------------------- */

int Swig_MembergetToFunction(Node *n, String *classname, int flags) {
  String *name;
  ParmList *parms;
  SwigType *t;
  SwigType *ty;
  SwigType *type;
  String *self = 0;

  int varcref = flags & CWRAP_NATURAL_VAR;

  if (flags & CWRAP_SMART_POINTER) {
    if (Swig_storage_isstatic(n)) {
      Node *sn = Getattr(n, "cplus:staticbase");
      String *base = Getattr(sn, "name");
      self = NewStringf("%s::", base);
    } else if (flags & CWRAP_SMART_POINTER_OVERLOAD) {
      String *nclassname = SwigType_namestr(classname);
      self = NewStringf("(*(%s const *)this)->", nclassname);
      Delete(nclassname);
    } else {
      self = NewString("(*this)->");
    }
  }
  if (flags & CWRAP_ALL_PROTECTED_ACCESS) {
    self = NewStringf("darg->");
  }

  name = Getattr(n, "name");
  type = Getattr(n, "type");

  t = NewString(classname);
  SwigType_add_pointer(t);
  parms = NewParm(t, "self", n);
  Setattr(parms, "self", "1");
  Setattr(parms, "hidden","1");
  Delete(t);

  ty = Swig_wrapped_member_var_type(type, varcref);
  if (flags & CWRAP_EXTEND) {
    String *call;
    String *cres;
    String *code = Getattr(n, "code");

    String *gname = Swig_name_get(0, name);
    String *membername = Swig_name_member(0, classname, gname);
    String *mangled = Swig_name_mangle(membername);

    if (code) {
      /* I don't think this ever gets run - WSF */
      Swig_add_extension_code(n, mangled, parms, ty, code, cparse_cplusplus, "self");
    }
    call = Swig_cfunction_call(mangled, parms);
    cres = Swig_cresult(ty, Swig_cresult_name(), call);
    Setattr(n, "wrap:action", cres);

    Delete(cres);
    Delete(call);
    Delete(mangled);
    Delete(membername);
    Delete(gname);
  } else {
    String *call = Swig_cmemberget_call(name, type, self, varcref);
    String *cres = Swig_cresult(ty, Swig_cresult_name(), call);
    Setattr(n, "wrap:action", cres);
    Delete(call);
    Delete(cres);
  }
  Setattr(n, "type", ty);
  Setattr(n, "parms", parms);
  Delete(parms);
  Delete(ty);

  return SWIG_OK;
}

/* -----------------------------------------------------------------------------
 * Swig_VarsetToFunction()
 *
 * This function creates a C wrapper for setting a global variable or static member
 * variable.
 * ----------------------------------------------------------------------------- */

int Swig_VarsetToFunction(Node *n, int flags) {
  String *name, *nname;
  ParmList *parms;
  SwigType *type, *ty;

  int varcref = flags & CWRAP_NATURAL_VAR;

  name = Getattr(n, "name");
  type = Getattr(n, "type");
  nname = SwigType_namestr(name);
  ty = Swig_wrapped_var_type(type, varcref);
  parms = NewParm(ty, name, n);

  if (flags & CWRAP_EXTEND) {
    String *sname = Swig_name_set(0, name);
    String *mangled = Swig_name_mangle(sname);
    String *call = Swig_cfunction_call(mangled, parms);
    String *cres = NewStringf("%s;", call);
    Setattr(n, "wrap:action", cres);
    Delete(cres);
    Delete(call);
    Delete(mangled);
    Delete(sname);
  } else {
    if (!Strstr(type, "enum $unnamed")) {
      String *pname = Swig_cparm_name(0, 0);
      String *dref = Swig_wrapped_var_deref(type, pname, varcref);
      String *call = NewStringf("%s = %s;", nname, dref);
      Setattr(n, "wrap:action", call);
      Delete(call);
      Delete(dref);
      Delete(pname);
    } else {
      String *pname = Swig_cparm_name(0, 0);
      String *call = NewStringf("if (sizeof(int) == sizeof(%s)) *(int*)(void*)&(%s) = %s;", nname, nname, pname);
      Setattr(n, "wrap:action", call);
      Delete(pname);
      Delete(call);
    }
  }
  Setattr(n, "type", "void");
  Setattr(n, "parms", parms);
  Delete(parms);
  Delete(ty);
  Delete(nname);
  return SWIG_OK;
}

/* -----------------------------------------------------------------------------
 * Swig_VargetToFunction()
 *
 * This function creates a C wrapper for getting a global variable or static member
 * variable.
 * ----------------------------------------------------------------------------- */

int Swig_VargetToFunction(Node *n, int flags) {
  String *cres, *call;
  String *name;
  SwigType *type;
  SwigType *ty = 0;

  int varcref = flags & CWRAP_NATURAL_VAR;

  name = Getattr(n, "name");
  type = Getattr(n, "type");
  ty = Swig_wrapped_var_type(type, varcref);

  if (flags & CWRAP_EXTEND) {
    String *sname = Swig_name_get(0, name);
    String *mangled = Swig_name_mangle(sname);
    call = Swig_cfunction_call(mangled, 0);
    cres = Swig_cresult(ty, Swig_cresult_name(), call);
    Setattr(n, "wrap:action", cres);
    Delete(mangled);
    Delete(sname);
  } else {
    String *nname = 0;
    if (Equal(nodeType(n), "constant")) {
      String *rawval = Getattr(n, "rawval");
      String *value = rawval ? rawval : Getattr(n, "value");
      nname = NewStringf("(%s)", value);
    } else {
      nname = SwigType_namestr(name);
    }
    call = Swig_wrapped_var_assign(type, nname, varcref);
    cres = Swig_cresult(ty, Swig_cresult_name(), call);
    Setattr(n, "wrap:action", cres);
    Delete(nname);
  }

  Setattr(n, "type", ty);
  Delattr(n, "parms");
  Delete(cres);
  Delete(call);
  Delete(ty);
  return SWIG_OK;
}
