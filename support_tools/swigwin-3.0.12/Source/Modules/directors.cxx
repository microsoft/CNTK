/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * directors.cxx
 *
 * Director support functions.
 * Not all of these may be necessary, and some may duplicate existing functionality
 * in SWIG.  --MR
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

/* Swig_csuperclass_call()
 *
 * Generates a fully qualified method call, including the full parameter list.
 * e.g. "base::method(i, j)"
 *
 */

String *Swig_csuperclass_call(String *base, String *method, ParmList *l) {
  String *call = NewString("");
  int arg_idx = 0;
  Parm *p;
  if (base) {
    Printf(call, "%s::", base);
  }
  Printf(call, "%s(", method);
  for (p = l; p; p = nextSibling(p)) {
    String *pname = Getattr(p, "name");
    if (!pname && Cmp(Getattr(p, "type"), "void")) {
      pname = NewString("");
      Printf(pname, "arg%d", arg_idx++);
    }
    if (p != l)
      Printf(call, ", ");
    Printv(call, pname, NIL);
  }
  Printf(call, ")");
  return call;
}

/* Swig_class_declaration()
 *
 * Generate the start of a class/struct declaration.
 * e.g. "class myclass"
 *
 */

String *Swig_class_declaration(Node *n, String *name) {
  if (!name) {
    name = Getattr(n, "sym:name");
  }
  String *result = NewString("");
  String *kind = Getattr(n, "kind");
  Printf(result, "%s %s", kind, name);
  return result;
}

String *Swig_class_name(Node *n) {
  String *name;
  name = Copy(Getattr(n, "sym:name"));
  return name;
}

/* Swig_director_declaration()
 *
 * Generate the full director class declaration, complete with base classes.
 * e.g. "class SwigDirector_myclass : public myclass, public Swig::Director {"
 *
 */

String *Swig_director_declaration(Node *n) {
  String *classname = Swig_class_name(n);
  String *directorname = Language::instance()->directorClassName(n);
  String *base = Getattr(n, "classtype");
  String *declaration = Swig_class_declaration(n, directorname);

  Printf(declaration, " : public %s, public Swig::Director {\n", base);
  Delete(classname);
  Delete(directorname);
  return declaration;
}


String *Swig_method_call(const_String_or_char_ptr name, ParmList *parms) {
  String *func;
  int i = 0;
  int comma = 0;
  Parm *p = parms;
  SwigType *pt;
  String *nname;

  func = NewString("");
  nname = SwigType_namestr(name);
  Printf(func, "%s(", nname);
  while (p) {
    String *pname;
    pt = Getattr(p, "type");
    if ((SwigType_type(pt) != T_VOID)) {
      if (comma)
	Printf(func, ",");
      pname = Getattr(p, "name");
      Printf(func, "%s", pname);
      comma = 1;
      i++;
    }
    p = nextSibling(p);
  }
  Printf(func, ")");
  return func;
}

/* Swig_method_decl
 *
 * Misnamed and misappropriated!  Taken from SWIG's type string manipulation utilities
 * and modified to generate full (or partial) type qualifiers for method declarations,
 * local variable declarations, and return value casting.  More importantly, it merges
 * parameter type information with actual parameter names to produce a complete method
 * declaration that fully mirrors the original method declaration.
 *
 * There is almost certainly a saner way to do this.
 *
 * This function needs to be cleaned up and possibly split into several smaller 
 * functions.  For instance, attaching default names to parameters should be done in a 
 * separate function.
 *
 */

String *Swig_method_decl(SwigType *rettype, SwigType *decl, const_String_or_char_ptr id, List *args, int strip, int values) {
  String *result;
  List *elements;
  String *element = 0, *nextelement;
  int is_const = 0;
  int nelements, i;
  int is_func = 0;
  int arg_idx = 0;

  if (id) {
    result = NewString(Char(id));
  } else {
    result = NewString("");
  }

  elements = SwigType_split(decl);
  nelements = Len(elements);
  if (nelements > 0) {
    element = Getitem(elements, 0);
  }
  for (i = 0; i < nelements; i++) {
    if (i < (nelements - 1)) {
      nextelement = Getitem(elements, i + 1);
    } else {
      nextelement = 0;
    }
    if (SwigType_isqualifier(element)) {
      int skip = 0;
      DOH *q = 0;
      if (!strip) {
	q = SwigType_parm(element);
	if (!Cmp(q, "const")) {
	  is_const = 1;
	  is_func = SwigType_isfunction(nextelement);
	  if (is_func)
	    skip = 1;
	  skip = 1;
	}
	if (!skip) {
	  Insert(result, 0, " ");
	  Insert(result, 0, q);
	}
	Delete(q);
      }
    } else if (SwigType_isfunction(element)) {
      Parm *parm;
      String *p;
      Append(result, "(");
      parm = args;
      while (parm != 0) {
	String *type = Getattr(parm, "type");
	String *name = Getattr(parm, "name");
	if (!name && Cmp(type, "void")) {
	  name = NewString("");
	  Printf(name, "arg%d", arg_idx++);
	  Setattr(parm, "name", name);
	}
	if (!name) {
	  name = NewString("");
	}
	p = SwigType_str(type, name);
	Append(result, p);
	String *value = Getattr(parm, "value");
	if (values && (value != 0)) {
	  Printf(result, " = %s", value);
	}
	parm = nextSibling(parm);
	if (parm != 0)
	  Append(result, ", ");
      }
      Append(result, ")");
    } else if (rettype) { // This check is intended for conversion operators to a pointer/reference which needs the pointer/reference ignoring in the declaration
      if (SwigType_ispointer(element)) {
	Insert(result, 0, "*");
	if ((nextelement) && ((SwigType_isfunction(nextelement) || (SwigType_isarray(nextelement))))) {
	  Insert(result, 0, "(");
	  Append(result, ")");
	}
      } else if (SwigType_ismemberpointer(element)) {
	String *q;
	q = SwigType_parm(element);
	Insert(result, 0, "::*");
	Insert(result, 0, q);
	if ((nextelement) && ((SwigType_isfunction(nextelement) || (SwigType_isarray(nextelement))))) {
	  Insert(result, 0, "(");
	  Append(result, ")");
	}
	Delete(q);
      } else if (SwigType_isreference(element)) {
	Insert(result, 0, "&");
      } else if (SwigType_isarray(element)) {
	DOH *size;
	Append(result, "[");
	size = SwigType_parm(element);
	Append(result, size);
	Append(result, "]");
	Delete(size);
      } else {
	if (Strcmp(element, "v(...)") == 0) {
	  Insert(result, 0, "...");
	} else {
	  String *bs = SwigType_namestr(element);
	  Insert(result, 0, " ");
	  Insert(result, 0, bs);
	  Delete(bs);
	}
      }
    }
    element = nextelement;
  }

  Delete(elements);

  if (is_const) {
    if (is_func) {
      Append(result, " ");
      Append(result, "const");
    } else {
      Insert(result, 0, "const ");
    }
  }

  Chop(result);

  if (rettype) {
    Insert(result, 0, " ");
    String *rtype = SwigType_str(rettype, 0);
    Insert(result, 0, rtype);
    Delete(rtype);
  }

  return result;
}

/* -----------------------------------------------------------------------------
 * Swig_director_emit_dynamic_cast()
 *
 * In order to call protected virtual director methods from the target language, we need
 * to add an extra dynamic_cast to call the public C++ wrapper in the director class. 
 * Also for non-static protected members when the allprotected option is on.
 * ----------------------------------------------------------------------------- */
void Swig_director_emit_dynamic_cast(Node *n, Wrapper *f) {
  // TODO: why is the storage element removed in staticmemberfunctionHandler ??
  if ((!is_public(n) && (is_member_director(n) || GetFlag(n, "explicitcall"))) || 
      (is_non_virtual_protected_access(n) && !(Swig_storage_isstatic_custom(n, "staticmemberfunctionHandler:storage") || 
                                               Swig_storage_isstatic(n))
                                          && !Equal(nodeType(n), "constructor"))) {
    Node *parent = Getattr(n, "parentNode");
    String *dirname;
    String *dirdecl;
    dirname = Language::instance()->directorClassName(parent);
    dirdecl = NewStringf("%s *darg = 0", dirname);
    Wrapper_add_local(f, "darg", dirdecl);
    Printf(f->code, "darg = dynamic_cast<%s *>(arg1);\n", dirname);
    Delete(dirname);
    Delete(dirdecl);
  }
}

/* ------------------------------------------------------------
 * Swig_director_parms_fixup()
 *
 * For each parameter in the C++ member function, copy the parameter name
 * to its "lname"; this ensures that Swig_typemap_attach_parms() will do
 * the right thing when it sees strings like "$1" in "directorin" typemaps.
 * ------------------------------------------------------------ */

void Swig_director_parms_fixup(ParmList *parms) {
  Parm *p;
  int i;
  for (i = 0, p = parms; p; p = nextSibling(p), ++i) {
    String *arg = Getattr(p, "name");
    String *lname = 0;

    if (!arg && !Equal(Getattr(p, "type"), "void")) {
      lname = NewStringf("arg%d", i);
      Setattr(p, "name", lname);
    } else
      lname = Copy(arg);

    Setattr(p, "lname", lname);
    Delete(lname);
  }
}

