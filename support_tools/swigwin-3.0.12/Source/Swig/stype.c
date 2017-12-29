/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * stype.c
 *
 * This file provides general support for datatypes that are encoded in
 * the form of simple strings.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "cparse.h"
#include <ctype.h>

/* -----------------------------------------------------------------------------
 * Synopsis
 *
 * The purpose of this module is to provide a general purpose type representation
 * based on simple text strings. 
 *
 * General idea:
 *
 * Types are represented by a base type (e.g., "int") and a collection of
 * type operators applied to the base (e.g., pointers, arrays, etc...).
 *
 * Encoding:
 *
 * Types are encoded as strings of type constructors such as follows:
 *
 *        String Encoding                 C Example
 *        ---------------                 ---------
 *        p.p.int                         int **
 *        a(300).a(400).int               int [300][400]
 *        p.q(const).char                 char const *
 *
 * All type constructors are denoted by a trailing '.':
 * 
 *  'p.'                = Pointer (*)
 *  'r.'                = Reference (&)
 *  'z.'                = Rvalue reference (&&)
 *  'a(n).'             = Array of size n  [n]
 *  'f(..,..).'         = Function with arguments  (args)
 *  'q(str).'           = Qualifier (such as const or volatile) (const, volatile)
 *  'm(qual).'          = Pointer to member (qual::*)
 *
 * The encoding follows the order that you might describe a type in words.
 * For example "p.a(200).int" is "A pointer to array of int's" and
 * "p.q(const).char" is "a pointer to a const char".
 *
 * This representation of types is fairly convenient because ordinary string
 * operations can be used for type manipulation. For example, a type could be
 * formed by combining two strings such as the following:
 *
 *        "p.p." + "a(400).int" = "p.p.a(400).int"
 *
 * Similarly, one could strip a 'const' declaration from a type doing something
 * like this:
 *
 *        Replace(t,"q(const).","",DOH_REPLACE_ANY)
 *
 * For the most part, this module tries to minimize the use of special
 * characters (*, [, <, etc...) in its type encoding.  One reason for this
 * is that SWIG might be extended to encode data in formats such as XML
 * where you might want to do this:
 * 
 *      <function>
 *         <type>p.p.int</type>
 *         ...
 *      </function>
 *
 * Or alternatively,
 *
 *      <function type="p.p.int" ...>blah</function>
 *
 * In either case, it's probably best to avoid characters such as '&', '*', or '<'.
 *
 * Why not use C syntax?  Well, C syntax is fairly complicated to parse
 * and not particularly easy to manipulate---especially for adding, deleting and
 * composing type constructors.  The string representation presented here makes
 * this pretty easy.
 *
 * Why not use a bunch of nested data structures?  Are you kidding? How
 * would that be easier to use than a few simple string operations? 
 * ----------------------------------------------------------------------------- */


SwigType *NewSwigType(int t) {
  switch (t) {
  case T_BOOL:
    return NewString("bool");
    break;
  case T_INT:
    return NewString("int");
    break;
  case T_UINT:
    return NewString("unsigned int");
    break;
  case T_SHORT:
    return NewString("short");
    break;
  case T_USHORT:
    return NewString("unsigned short");
    break;
  case T_LONG:
    return NewString("long");
    break;
  case T_ULONG:
    return NewString("unsigned long");
    break;
  case T_FLOAT:
    return NewString("float");
    break;
  case T_DOUBLE:
    return NewString("double");
    break;
  case T_COMPLEX:
    return NewString("complex");
    break;
  case T_CHAR:
    return NewString("char");
    break;
  case T_SCHAR:
    return NewString("signed char");
    break;
  case T_UCHAR:
    return NewString("unsigned char");
    break;
  case T_STRING: {
      SwigType *t = NewString("char");
      SwigType_add_qualifier(t, "const");
      SwigType_add_pointer(t);
      return t;
      break;
    }
  case T_WCHAR:
    return NewString("wchar_t");
    break;
  case T_WSTRING: {
    SwigType *t = NewString("wchar_t");
    SwigType_add_pointer(t);
    return t;
    break;
  }
  case T_LONGLONG:
    return NewString("long long");
    break;
  case T_ULONGLONG:
    return NewString("unsigned long long");
    break;
  case T_VOID:
    return NewString("void");
    break;
  case T_AUTO:
    return NewString("auto");
    break;
  default:
    break;
  }
  return NewStringEmpty();
}

/* -----------------------------------------------------------------------------
 * SwigType_push()
 *
 * Push a type constructor onto the type
 * ----------------------------------------------------------------------------- */

void SwigType_push(SwigType *t, String *cons) {
  if (!cons)
    return;
  if (!Len(cons))
    return;

  if (Len(t)) {
    char *c = Char(cons);
    if (c[strlen(c) - 1] != '.')
      Insert(t, 0, ".");
  }
  Insert(t, 0, cons);
}

/* -----------------------------------------------------------------------------
 * SwigType_ispointer_return()
 *
 * Testing functions for querying a raw datatype
 * ----------------------------------------------------------------------------- */

int SwigType_ispointer_return(const SwigType *t) {
  char *c;
  int idx;
  if (!t)
    return 0;
  c = Char(t);
  idx = (int)strlen(c) - 4;
  if (idx >= 0) {
    return (strcmp(c + idx, ").p.") == 0);
  }
  return 0;
}

int SwigType_isreference_return(const SwigType *t) {
  char *c;
  int idx;
  if (!t)
    return 0;
  c = Char(t);
  idx = (int)strlen(c) - 4;
  if (idx >= 0) {
    return (strcmp(c + idx, ").r.") == 0);
  }
  return 0;
}

int SwigType_isconst(const SwigType *t) {
  char *c;
  if (!t)
    return 0;
  c = Char(t);
  if (strncmp(c, "q(", 2) == 0) {
    String *q = SwigType_parm(t);
    if (strstr(Char(q), "const")) {
      Delete(q);
      return 1;
    }
    Delete(q);
  }
  /* Hmmm. Might be const through a typedef */
  if (SwigType_issimple(t)) {
    int ret;
    SwigType *td = SwigType_typedef_resolve(t);
    if (td) {
      ret = SwigType_isconst(td);
      Delete(td);
      return ret;
    }
  }
  return 0;
}

int SwigType_ismutable(const SwigType *t) {
  int r;
  SwigType *qt = SwigType_typedef_resolve_all(t);
  if (SwigType_isreference(qt) || SwigType_isrvalue_reference(qt) || SwigType_isarray(qt)) {
    Delete(SwigType_pop(qt));
  }
  r = SwigType_isconst(qt);
  Delete(qt);
  return r ? 0 : 1;
}

int SwigType_isenum(const SwigType *t) {
  char *c = Char(t);
  if (!t)
    return 0;
  if (strncmp(c, "enum ", 5) == 0) {
    return 1;
  }
  return 0;
}

int SwigType_issimple(const SwigType *t) {
  char *c = Char(t);
  if (!t)
    return 0;
  while (*c) {
    if (*c == '<') {
      int nest = 1;
      c++;
      while (*c && nest) {
	if (*c == '<')
	  nest++;
	if (*c == '>')
	  nest--;
	c++;
      }
      c--;
    }
    if (*c == '.')
      return 0;
    c++;
  }
  return 1;
}

/* -----------------------------------------------------------------------------
 * SwigType_default_create()
 *
 * Create the default type for this datatype. This takes a type and strips it
 * down to a generic form first by resolving all typedefs.
 *
 * Rules:
 *     Pointers:              p.SWIGTYPE
 *     References:            r.SWIGTYPE
 *     Arrays no dimension:   a().SWIGTYPE
 *     Arrays with dimension: a(ANY).SWIGTYPE
 *     Member pointer:        m(CLASS).SWIGTYPE
 *     Function pointer:      f(ANY).SWIGTYPE
 *     Enums:                 enum SWIGTYPE
 *     Types:                 SWIGTYPE
 *
 * Examples (also see SwigType_default_deduce):
 *
 *  int [2][4]
 *    a(2).a(4).int
 *    a(ANY).a(ANY).SWIGTYPE
 *
 *  struct A {};
 *  typedef A *Aptr;
 *  Aptr const &
 *    r.q(const).Aptr
 *    r.q(const).p.SWIGTYPE
 *
 *  enum E {e1, e2};
 *  enum E const &
 *    r.q(const).enum E
 *    r.q(const).enum SWIGTYPE
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_default_create(const SwigType *ty) {
  SwigType *r = 0;
  List *l;
  Iterator it;
  int numitems;

  if (!SwigType_isvarargs(ty)) {
    SwigType *t = SwigType_typedef_resolve_all(ty);
    r = NewStringEmpty();
    l = SwigType_split(t);
    numitems = Len(l);

    if (numitems >= 1) {
      String *last_subtype = Getitem(l, numitems-1);
      if (SwigType_isenum(last_subtype))
	Setitem(l, numitems-1, NewString("enum SWIGTYPE"));
      else
	Setitem(l, numitems-1, NewString("SWIGTYPE"));
    }

    for (it = First(l); it.item; it = Next(it)) {
      String *subtype = it.item;
      if (SwigType_isarray(subtype)) {
	if (Equal(subtype, "a()."))
	  Append(r, NewString("a()."));
	else
	  Append(r, NewString("a(ANY)."));
      } else if (SwigType_isfunction(subtype)) {
	Append(r, NewString("f(ANY).SWIGTYPE"));
	break;
      } else if (SwigType_ismemberpointer(subtype)) {
	Append(r, NewString("m(CLASS).SWIGTYPE"));
	break;
      } else {
	Append(r, subtype);
      }
    }

    Delete(l);
    Delete(t);
  }

  return r;
}

/* -----------------------------------------------------------------------------
 * SwigType_default_deduce()
 *
 * This function implements type deduction used in the typemap matching rules
 * and is very close to the type deduction used in partial template class
 * specialization matching in that the most specialized type is always chosen.
 * SWIGTYPE is used as the generic type. The basic idea is to repeatedly call
 * this function to find a deduced type unless until nothing matches.
 *
 * The type t must have already been converted to the default type via a call to
 * SwigType_default_create() before calling this function.
 *
 * Example deductions (matching the examples described in SwigType_default_create),
 * where the most specialized matches are highest in the list:
 *
 *    a(ANY).a(ANY).SWIGTYPE
 *    a(ANY).a().SWIGTYPE
 *    a(ANY).p.SWIGTYPE
 *    a(ANY).SWIGTYPE
 *    a().SWIGTYPE
 *    p.SWIGTYPE
 *    SWIGTYPE
 *
 *    r.q(const).p.SWIGTYPE
 *    r.q(const).SWIGTYPE
 *    r.SWIGTYPE
 *    SWIGTYPE
 *
 *    r.q(const).enum SWIGTYPE
 *    r.enum SWIGTYPE
 *    r.SWIGTYPE
 *    SWIGTYPE
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_default_deduce(const SwigType *t) {
  SwigType *r = NewStringEmpty();
  List *l;
  Iterator it;
  int numitems;

  l = SwigType_split(t);

  numitems = Len(l);
  if (numitems >= 1) {
    String *last_subtype = Getitem(l, numitems-1);
    int is_enum = SwigType_isenum(last_subtype);

    if (numitems >=2 ) {
      String *subtype = Getitem(l, numitems-2); /* last but one */
      if (SwigType_isarray(subtype)) {
	if (is_enum) {
	  /* enum deduction, enum SWIGTYPE => SWIGTYPE */
	  Setitem(l, numitems-1, NewString("SWIGTYPE"));
	} else {
	  /* array deduction, a(ANY). => a(). => p. */
	  String *deduced_subtype = 0;
	  if (Strcmp(subtype, "a().") == 0) {
	    deduced_subtype = NewString("p.");
	  } else if (Strcmp(subtype, "a(ANY).") == 0) {
	    deduced_subtype = NewString("a().");
	  } else {
	    assert(0);
	  }
	  Setitem(l, numitems-2, deduced_subtype);
	}
      } else if (SwigType_ismemberpointer(subtype)) {
	/* member pointer deduction, m(CLASS). => p. */
	Setitem(l, numitems-2, NewString("p."));
      } else if (is_enum && !SwigType_isqualifier(subtype)) {
	/* enum deduction, enum SWIGTYPE => SWIGTYPE */
	Setitem(l, numitems-1, NewString("SWIGTYPE"));
      } else {
	/* simple type deduction, eg, r.p.p. => r.p. */
	/* also function pointers eg, p.f(ANY). => p. */
	Delitem(l, numitems-2);
      }
    } else {
      if (is_enum) {
	/* enum deduction, enum SWIGTYPE => SWIGTYPE */
	Setitem(l, numitems-1, NewString("SWIGTYPE"));
      } else {
	/* delete the only item, we are done with deduction */
	Delitem(l, 0);
      }
    }
  } else {
    assert(0);
  }

  for (it = First(l); it.item; it = Next(it)) {
    Append(r, it.item);
  }

  if (Len(r) == 0) {
    Delete(r);
    r = 0;
  }

  Delete(l);
  return r;
}


/* -----------------------------------------------------------------------------
 * SwigType_namestr()
 *
 * Returns a string of the base type.  Takes care of template expansions
 * ----------------------------------------------------------------------------- */

String *SwigType_namestr(const SwigType *t) {
  String *r;
  String *suffix;
  List *p;
  int i, sz;
  char *d = Char(t);
  char *c = strstr(d, "<(");

  if (!c || !strstr(c + 2, ")>"))
    return NewString(t);

  r = NewStringWithSize(d, (int)(c - d));
  if (*(c - 1) == '<')
    Putc(' ', r);
  Putc('<', r);

  p = SwigType_parmlist(c + 1);
  sz = Len(p);
  for (i = 0; i < sz; i++) {
    String *str = SwigType_str(Getitem(p, i), 0);
    /* Avoid creating a <: token, which is the same as [ in C++ - put a space after '<'. */
    if (i == 0 && Len(str))
      Putc(' ', r);
    Append(r, str);
    if ((i + 1) < sz)
      Putc(',', r);
    Delete(str);
  }
  Putc(' ', r);
  Putc('>', r);
  suffix = SwigType_templatesuffix(t);
  if (Len(suffix) > 0) {
    String *suffix_namestr = SwigType_namestr(suffix);
    Append(r, suffix_namestr);
    Delete(suffix_namestr);
  } else {
    Append(r, suffix);
  }
  Delete(suffix);
  Delete(p);
  return r;
}

/* -----------------------------------------------------------------------------
 * SwigType_str()
 *
 * Create a C string representation of a datatype.
 * ----------------------------------------------------------------------------- */

String *SwigType_str(const SwigType *s, const_String_or_char_ptr id) {
  String *result;
  String *element = 0;
  String *nextelement;
  String *forwardelement;
  List *elements;
  int nelements, i;

  if (id) {
    /* stringify the id expanding templates, for example when the id is a fully qualified templated class name */
    String *id_str = NewString(id); /* unfortunate copy due to current const limitations */
    result = SwigType_str(id_str, 0);
    Delete(id_str);
  } else {
    result = NewStringEmpty();
  }

  elements = SwigType_split(s);
  nelements = Len(elements);

  if (nelements > 0) {
    element = Getitem(elements, 0);
  }
  /* Now, walk the type list and start emitting */
  for (i = 0; i < nelements; i++) {
    if (i < (nelements - 1)) {
      nextelement = Getitem(elements, i + 1);
      forwardelement = nextelement;
      if (SwigType_isqualifier(nextelement)) {
	if (i < (nelements - 2))
	  forwardelement = Getitem(elements, i + 2);
      }
    } else {
      nextelement = 0;
      forwardelement = 0;
    }
    if (SwigType_isqualifier(element)) {
      DOH *q = 0;
      q = SwigType_parm(element);
      Insert(result, 0, " ");
      Insert(result, 0, q);
      Delete(q);
    } else if (SwigType_ispointer(element)) {
      Insert(result, 0, "*");
      if ((forwardelement) && ((SwigType_isfunction(forwardelement) || (SwigType_isarray(forwardelement))))) {
	Insert(result, 0, "(");
	Append(result, ")");
      }
    } else if (SwigType_ismemberpointer(element)) {
      String *q;
      q = SwigType_parm(element);
      Insert(result, 0, "::*");
      Insert(result, 0, q);
      if ((forwardelement) && ((SwigType_isfunction(forwardelement) || (SwigType_isarray(forwardelement))))) {
	Insert(result, 0, "(");
	Append(result, ")");
      }
      Delete(q);
    } else if (SwigType_isreference(element)) {
      Insert(result, 0, "&");
      if ((forwardelement) && ((SwigType_isfunction(forwardelement) || (SwigType_isarray(forwardelement))))) {
	Insert(result, 0, "(");
	Append(result, ")");
      }
    } else if (SwigType_isrvalue_reference(element)) {
      Insert(result, 0, "&&");
      if ((nextelement) && ((SwigType_isfunction(nextelement) || (SwigType_isarray(nextelement))))) {
	Insert(result, 0, "(");
	Append(result, ")");
      }
    } else if (SwigType_isarray(element)) {
      DOH *size;
      Append(result, "[");
      size = SwigType_parm(element);
      Append(result, size);
      Append(result, "]");
      Delete(size);
    } else if (SwigType_isfunction(element)) {
      DOH *parms, *p;
      int j, plen;
      Append(result, "(");
      parms = SwigType_parmlist(element);
      plen = Len(parms);
      for (j = 0; j < plen; j++) {
	p = SwigType_str(Getitem(parms, j), 0);
	Append(result, p);
	if (j < (plen - 1))
	  Append(result, ",");
      }
      Append(result, ")");
      Delete(parms);
    } else {
      if (strcmp(Char(element), "v(...)") == 0) {
	Insert(result, 0, "...");
      } else {
	String *bs = SwigType_namestr(element);
	Insert(result, 0, " ");
	Insert(result, 0, bs);
	Delete(bs);
      }
    }
    element = nextelement;
  }
  Delete(elements);
  Chop(result);
  return result;
}

/* -----------------------------------------------------------------------------
 * SwigType_ltype(const SwigType *ty)
 *
 * Create a locally assignable type
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_ltype(const SwigType *s) {
  String *result;
  String *element;
  SwigType *td, *tc = 0;
  List *elements;
  int nelements, i;
  int firstarray = 1;
  int notypeconv = 0;

  result = NewStringEmpty();
  tc = Copy(s);
  /* Nuke all leading qualifiers */
  while (SwigType_isqualifier(tc)) {
    Delete(SwigType_pop(tc));
  }
  if (SwigType_issimple(tc)) {
    /* Resolve any typedef definitions */
    SwigType *tt = Copy(tc);
    td = 0;
    while ((td = SwigType_typedef_resolve(tt))) {
      if (td && (SwigType_isconst(td) || SwigType_isarray(td) || SwigType_isreference(td) || SwigType_isrvalue_reference(td))) {
	/* We need to use the typedef type */
	Delete(tt);
	break;
      } else if (td) {
	Delete(tt);
	tt = td;
      }
    }
    if (td) {
      Delete(tc);
      tc = td;
    }
  }
  elements = SwigType_split(tc);
  nelements = Len(elements);

  /* Now, walk the type list and start emitting */
  for (i = 0; i < nelements; i++) {
    element = Getitem(elements, i);
    /* when we see a function, we need to preserve the following types */
    if (SwigType_isfunction(element)) {
      notypeconv = 1;
    }
    if (SwigType_isqualifier(element)) {
      /* Do nothing. Ignore */
    } else if (SwigType_ispointer(element)) {
      Append(result, element);
      firstarray = 0;
    } else if (SwigType_ismemberpointer(element)) {
      Append(result, element);
      firstarray = 0;
    } else if (SwigType_isreference(element)) {
      if (notypeconv) {
	Append(result, element);
      } else {
	Append(result, "p.");
      }
      firstarray = 0;
    } else if (SwigType_isrvalue_reference(element)) {
      if (notypeconv) {
	Append(result, element);
      } else {
	Append(result, "p.");
      }
      firstarray = 0;
    } else if (SwigType_isarray(element) && firstarray) {
      if (notypeconv) {
	Append(result, element);
      } else {
	Append(result, "p.");
      }
      firstarray = 0;
    } else if (SwigType_isenum(element)) {
      int anonymous_enum = (Cmp(element, "enum ") == 0);
      if (notypeconv || !anonymous_enum) {
	Append(result, element);
      } else {
	Append(result, "int");
      }
    } else {
      Append(result, element);
    }
  }
  Delete(elements);
  Delete(tc);
  return result;
}

/* -----------------------------------------------------------------------------
 * SwigType_lstr(DOH *s, DOH *id)
 *
 * Produces a type-string that is suitable as a lvalue in an expression.
 * That is, a type that can be freely assigned a value without violating
 * any C assignment rules.
 *
 *      -   Qualifiers such as 'const' and 'volatile' are stripped.
 *      -   Arrays are converted into a *single* pointer (i.e.,
 *          double [][] becomes double *).
 *      -   References are converted into a pointer.
 *      -   Typedef names that refer to read-only types will be replaced
 *          with an equivalent assignable version.
 * -------------------------------------------------------------------- */

String *SwigType_lstr(const SwigType *s, const_String_or_char_ptr id) {
  String *result;
  SwigType *tc;

  tc = SwigType_ltype(s);
  result = SwigType_str(tc, id);
  Delete(tc);
  return result;
}

/* -----------------------------------------------------------------------------
 * SwigType_rcaststr()
 *
 * Produces a casting string that maps the type returned by lstr() to the real 
 * datatype printed by str().
 * ----------------------------------------------------------------------------- */

String *SwigType_rcaststr(const SwigType *s, const_String_or_char_ptr name) {
  String *result, *cast;
  String *element = 0;
  String *nextelement;
  String *forwardelement;
  SwigType *td, *tc = 0;
  const SwigType *rs;
  List *elements;
  int nelements, i;
  int clear = 1;
  int firstarray = 1;
  int isreference = 0;
  int isfunction = 0;

  result = NewStringEmpty();

  if (SwigType_isconst(s)) {
    tc = Copy(s);
    Delete(SwigType_pop(tc));
    rs = tc;
  } else {
    rs = s;
  }

  if ((SwigType_isconst(rs) || SwigType_isarray(rs) || SwigType_isreference(rs) || SwigType_isrvalue_reference(rs))) {
    td = 0;
  } else {
    td = SwigType_typedef_resolve(rs);
  }

  if (td) {
    if ((SwigType_isconst(td) || SwigType_isarray(td) || SwigType_isreference(td) || SwigType_isrvalue_reference(td))) {
      elements = SwigType_split(td);
    } else {
      elements = SwigType_split(rs);
    }
    Delete(td);
  } else {
    elements = SwigType_split(rs);
  }
  nelements = Len(elements);
  if (nelements > 0) {
    element = Getitem(elements, 0);
  }
  /* Now, walk the type list and start emitting */
  for (i = 0; i < nelements; i++) {
    if (i < (nelements - 1)) {
      nextelement = Getitem(elements, i + 1);
      forwardelement = nextelement;
      if (SwigType_isqualifier(nextelement)) {
	if (i < (nelements - 2))
	  forwardelement = Getitem(elements, i + 2);
      }
    } else {
      nextelement = 0;
      forwardelement = 0;
    }
    if (SwigType_isqualifier(element)) {
      DOH *q = 0;
      q = SwigType_parm(element);
      Insert(result, 0, " ");
      Insert(result, 0, q);
      Delete(q);
      clear = 0;
    } else if (SwigType_ispointer(element)) {
      Insert(result, 0, "*");
      if ((forwardelement) && ((SwigType_isfunction(forwardelement) || (SwigType_isarray(forwardelement))))) {
	Insert(result, 0, "(");
	Append(result, ")");
      }
      firstarray = 0;
    } else if (SwigType_ismemberpointer(element)) {
      String *q;
      Insert(result, 0, "::*");
      q = SwigType_parm(element);
      Insert(result, 0, q);
      Delete(q);
      if ((forwardelement) && ((SwigType_isfunction(forwardelement) || (SwigType_isarray(forwardelement))))) {
	Insert(result, 0, "(");
	Append(result, ")");
      }
      firstarray = 0;
    } else if (SwigType_isreference(element)) {
      Insert(result, 0, "&");
      if ((forwardelement) && ((SwigType_isfunction(forwardelement) || (SwigType_isarray(forwardelement))))) {
	Insert(result, 0, "(");
	Append(result, ")");
      }
      if (!isfunction)
	isreference = 1;
    } else if (SwigType_isrvalue_reference(element)) {
      Insert(result, 0, "&&");
      if ((forwardelement) && ((SwigType_isfunction(forwardelement) || (SwigType_isarray(forwardelement))))) {
	Insert(result, 0, "(");
	Append(result, ")");
      }
      if (!isfunction)
	isreference = 1;
      clear = 0;
    } else if (SwigType_isarray(element)) {
      DOH *size;
      if (firstarray && !isreference) {
	Append(result, "(*)");
	firstarray = 0;
      } else {
	Append(result, "[");
	size = SwigType_parm(element);
	Append(result, size);
	Append(result, "]");
	Delete(size);
	clear = 0;
      }
    } else if (SwigType_isfunction(element)) {
      DOH *parms, *p;
      int j, plen;
      Append(result, "(");
      parms = SwigType_parmlist(element);
      plen = Len(parms);
      for (j = 0; j < plen; j++) {
	p = SwigType_str(Getitem(parms, j), 0);
	Append(result, p);
	Delete(p);
	if (j < (plen - 1))
	  Append(result, ",");
      }
      Append(result, ")");
      Delete(parms);
      isfunction = 1;
    } else {
      String *bs = SwigType_namestr(element);
      Insert(result, 0, " ");
      Insert(result, 0, bs);
      Delete(bs);
    }
    element = nextelement;
  }
  Delete(elements);
  if (clear) {
    cast = NewStringEmpty();
  } else {
    cast = NewStringf("(%s)", result);
  }
  if (name) {
    if (isreference) {
      Append(cast, "*");
    }
    Append(cast, name);
  }
  Delete(result);
  Delete(tc);
  return cast;
}


/* -----------------------------------------------------------------------------
 * SwigType_lcaststr()
 *
 * Casts a variable from the real type to the local datatype.
 * ----------------------------------------------------------------------------- */

String *SwigType_lcaststr(const SwigType *s, const_String_or_char_ptr name) {
  String *result;

  result = NewStringEmpty();

  if (SwigType_isarray(s)) {
    String *lstr = SwigType_lstr(s, 0);
    Printf(result, "(%s)%s", lstr, name);
    Delete(lstr);
  } else if (SwigType_isreference(s)) {
    String *str = SwigType_str(s, 0);
    Printf(result, "(%s)", str);
    Delete(str);
    if (name)
      Append(result, name);
  } else if (SwigType_isrvalue_reference(s)) {
    String *str = SwigType_str(s, 0);
    Printf(result, "(%s)", str);
    Delete(str);
    if (name)
      Append(result, name);
  } else if (SwigType_isqualifier(s)) {
    String *lstr = SwigType_lstr(s, 0);
    Printf(result, "(%s)%s", lstr, name);
    Delete(lstr);
  } else {
    if (name)
      Append(result, name);
  }
  return result;
}

#if 0
/* Alternative implementation for manglestr_default. Mangling is similar to the original
   except for a few subtle differences for example in templates:
    namespace foo {
      template<class T> class bar {};
      typedef int Integer;
      void test2(bar<Integer *> *x);
    }
    Mangling is more consistent and changes from 
    _p_foo__barT_int_p_t to 
    _p_foo__barT_p_int_t.
*/
static void mangle_stringcopy(String *destination, const char *source, int count) {
  while (count-- > 0) {
    char newc = '_';
    if (!(*source == '.' || *source == ':' || *source == ' '))
      newc = *source;
    /* TODO: occasionally '*' or numerics need converting to '_', eg in array dimensions and template expressions */
    Putc(newc, destination);
    source++;
  }
}

static void mangle_subtype(String *mangled, SwigType *s);

/* -----------------------------------------------------------------------------
 * mangle_namestr()
 *
 * Mangles a type taking care of template expansions. Similar to SwigType_namestr().
 * The type may include a trailing '.', for example "p."
 * ----------------------------------------------------------------------------- */

static void mangle_namestr(String *mangled, SwigType *t) {
  int length = Len(t);
  if (SwigType_isqualifier(t)) {
    Append(mangled, "q_");
    mangle_stringcopy(mangled, Char(t)+2, length-4);
    Append(mangled, "__");
  } else if (SwigType_ismemberpointer(t)) {
    Append(mangled, "m_");
    mangle_stringcopy(mangled, Char(t)+2, length-4);
    Append(mangled, "__");
  } else if (SwigType_isarray(t)) {
    Append(mangled, "a_");
    mangle_stringcopy(mangled, Char(t)+2, length-4);
    Append(mangled, "__");
  } else if (SwigType_isfunction(t)) {
    List *p = SwigType_parmlist(t);
    int sz = Len(p);
    int i;
    Append(mangled, "f_");
    for (i = 0; i < sz; i++) {
      mangle_subtype(mangled, Getitem(p, i));
      Putc('_', mangled);
    }
    Append(mangled, (sz > 0) ? "_" : "__");
  } else if (SwigType_isvarargs(t)) {
    Append(mangled, "___");
  } else {
    char *d = Char(t);
    char *c = strstr(d, "<(");
    if (!c || !strstr(c + 2, ")>")) {
      /* not a template type */
      mangle_stringcopy(mangled, Char(t), Len(t));
    } else {
      /* a template type */
      String *suffix;
      List *p;
      int i, sz;
      mangle_stringcopy(mangled, d, c-d);
      Putc('T', mangled);
      Putc('_', mangled);

      p = SwigType_parmlist(c + 1);
      sz = Len(p);
      for (i = 0; i < sz; i++) {
	mangle_subtype(mangled, Getitem(p, i));
	Putc('_', mangled);
      }
      Putc('t', mangled);
      suffix = SwigType_templatesuffix(t);
      if (Len(suffix) > 0) {
	mangle_namestr(mangled, suffix);
      } else {
	Append(mangled, suffix);
      }
      Delete(suffix);
      Delete(p);
    }
  }
}

static void mangle_subtype(String *mangled, SwigType *s) {
  List *elements;
  int nelements, i;

  assert(s);
  elements = SwigType_split(s);
  nelements = Len(elements);
  for (i = 0; i < nelements; i++) {
    SwigType *element = Getitem(elements, i);
    mangle_namestr(mangled, element);
  }
  Delete(elements);
}

static String *manglestr_default(const SwigType *s) {
  String *mangled = NewString("_");
  SwigType *sr = SwigType_typedef_resolve_all(s);
  SwigType *sq = SwigType_typedef_qualified(sr);
  SwigType *ss = SwigType_remove_global_scope_prefix(sq);
  SwigType *type = ss;
  SwigType *lt;

  if (SwigType_istemplate(ss)) {
    SwigType *ty = Swig_symbol_template_deftype(ss, 0);
    Delete(ss);
    ss = ty;
    type = ss;
  }

  lt = SwigType_ltype(type);

  Replace(lt, "struct ", "", DOH_REPLACE_ANY);
  Replace(lt, "class ", "", DOH_REPLACE_ANY);
  Replace(lt, "union ", "", DOH_REPLACE_ANY);
  Replace(lt, "enum ", "", DOH_REPLACE_ANY);

  mangle_subtype(mangled, lt);

  Delete(ss);
  Delete(sq);
  Delete(sr);

  return mangled;
}

#else

static String *manglestr_default(const SwigType *s) {
  char *c;
  String *result = 0;
  String *base = 0;
  SwigType *lt;
  SwigType *sr = SwigType_typedef_resolve_all(s);
  SwigType *sq = SwigType_typedef_qualified(sr);
  SwigType *ss = SwigType_remove_global_scope_prefix(sq);
  SwigType *type = ss;

  if (SwigType_istemplate(ss)) {
    SwigType *ty = Swig_symbol_template_deftype(ss, 0);
    Delete(ss);
    ss = ty;
    type = ss;
  }

  lt = SwigType_ltype(type);
  result = SwigType_prefix(lt);
  base = SwigType_base(lt);

  c = Char(result);
  while (*c) {
    if (!isalnum((int) *c))
      *c = '_';
    c++;
  }
  if (SwigType_istemplate(base)) {
    String *b = SwigType_namestr(base);
    Delete(base);
    base = b;
  }

  Replace(base, "struct ", "", DOH_REPLACE_ANY);	/* This might be problematic */
  Replace(base, "class ", "", DOH_REPLACE_ANY);
  Replace(base, "union ", "", DOH_REPLACE_ANY);
  Replace(base, "enum ", "", DOH_REPLACE_ANY);

  c = Char(base);
  while (*c) {
    if (*c == '<')
      *c = 'T';
    else if (*c == '>')
      *c = 't';
    else if (*c == '*')
      *c = 'p';
    else if (*c == '[')
      *c = 'a';
    else if (*c == ']')
      *c = 'A';
    else if (*c == '&')
      *c = 'R';
    else if (*c == '(')
      *c = 'f';
    else if (*c == ')')
      *c = 'F';
    else if (!isalnum((int) *c))
      *c = '_';
    c++;
  }
  Append(result, base);
  Insert(result, 0, "_");
  Delete(lt);
  Delete(base);
  Delete(ss);
  Delete(sq);
  Delete(sr);
  return result;
}
#endif

String *SwigType_manglestr(const SwigType *s) {
#if 0
  /* Debugging checks to ensure a proper SwigType is passed in and not a stringified type */
  String *angle = Strstr(s, "<");
  if (angle && Strncmp(angle, "<(", 2) != 0)
    Printf(stderr, "SwigType_manglestr error: %s\n", s);
  else if (Strstr(s, "*") || Strstr(s, "&") || Strstr(s, "["))
    Printf(stderr, "SwigType_manglestr error: %s\n", s);
#endif
  return manglestr_default(s);
}

/* -----------------------------------------------------------------------------
 * SwigType_typename_replace()
 *
 * Replaces a typename in a type with something else.  Needed for templates.
 * ----------------------------------------------------------------------------- */

void SwigType_typename_replace(SwigType *t, String *pat, String *rep) {
  String *nt;
  int i, ilen;
  List *elem;

  if (!Strstr(t, pat))
    return;

  if (Equal(t, pat)) {
    Replace(t, pat, rep, DOH_REPLACE_ANY);
    return;
  }
  nt = NewStringEmpty();
  elem = SwigType_split(t);
  ilen = Len(elem);
  for (i = 0; i < ilen; i++) {
    String *e = Getitem(elem, i);
    if (SwigType_issimple(e)) {
      if (Equal(e, pat)) {
	/* Replaces a type of the form 'pat' with 'rep<args>' */
	Replace(e, pat, rep, DOH_REPLACE_ANY);
      } else if (SwigType_istemplate(e)) {
	/* Replaces a type of the form 'pat<args>' with 'rep' */
	if (Equal(e, pat)) {
	  String *repbase = SwigType_templateprefix(rep);
	  Replace(e, pat, repbase, DOH_REPLACE_ID | DOH_REPLACE_FIRST);
	  Delete(repbase);
	}
	{
	  String *tsuffix;
	  List *tparms = SwigType_parmlist(e);
	  int j, jlen;
	  String *nt = SwigType_templateprefix(e);
	  Append(nt, "<(");
	  jlen = Len(tparms);
	  for (j = 0; j < jlen; j++) {
	    SwigType_typename_replace(Getitem(tparms, j), pat, rep);
	    Append(nt, Getitem(tparms, j));
	    if (j < (jlen - 1))
	      Putc(',', nt);
	  }
	  tsuffix = SwigType_templatesuffix(e);
	  Printf(nt, ")>%s", tsuffix);
	  Delete(tsuffix);
	  Clear(e);
	  Append(e, nt);
	  Delete(nt);
	  Delete(tparms);
	}
      } else if (Swig_scopename_check(e)) {
	String *first, *rest;
	first = Swig_scopename_first(e);
	rest = Swig_scopename_suffix(e);
	SwigType_typename_replace(rest, pat, rep);
	SwigType_typename_replace(first, pat, rep);
	Clear(e);
	Printv(e, first, "::", rest, NIL);
	Delete(first);
	Delete(rest);
      }
    } else if (SwigType_isfunction(e)) {
      int j, jlen;
      List *fparms = SwigType_parmlist(e);
      Clear(e);
      Append(e, "f(");
      jlen = Len(fparms);
      for (j = 0; j < jlen; j++) {
	SwigType_typename_replace(Getitem(fparms, j), pat, rep);
	Append(e, Getitem(fparms, j));
	if (j < (jlen - 1))
	  Putc(',', e);
      }
      Append(e, ").");
      Delete(fparms);
    } else if (SwigType_isarray(e)) {
      Replace(e, pat, rep, DOH_REPLACE_ID);
    }
    Append(nt, e);
  }
  Clear(t);
  Append(t, nt);
  Delete(nt);
  Delete(elem);
}

/* -----------------------------------------------------------------------------
 * SwigType_remove_global_scope_prefix()
 *
 * Removes the unary scope operator (::) prefix indicating global scope in all 
 * components of the type
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_remove_global_scope_prefix(const SwigType *t) {
  SwigType *result;
  const char *type = Char(t);
  if (strncmp(type, "::", 2) == 0)
    type += 2;
  result = NewString(type);
  Replaceall(result, ".::", ".");
  Replaceall(result, "(::", "(");
  Replaceall(result, "enum ::", "enum ");
  return result;
}

/* -----------------------------------------------------------------------------
 * SwigType_check_decl()
 *
 * Checks type declarators for a match
 * ----------------------------------------------------------------------------- */

int SwigType_check_decl(const SwigType *ty, const SwigType *decl) {
  SwigType *t, *t1, *t2;
  int r;
  t = SwigType_typedef_resolve_all(ty);
  t1 = SwigType_strip_qualifiers(t);
  t2 = SwigType_prefix(t1);
  r = Equal(t2, decl);
  Delete(t);
  Delete(t1);
  Delete(t2);
  return r == 1;
}
