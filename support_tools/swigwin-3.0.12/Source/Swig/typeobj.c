/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * typeobj.c
 *
 * This file provides functions for constructing, manipulating, and testing
 * type objects.   Type objects are merely the raw low-level representation
 * of C++ types.   They do not incorporate high-level type system features
 * like typedef, namespaces, etc.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include <ctype.h>
#include <limits.h>

/* -----------------------------------------------------------------------------
 * Synopsis
 *
 * This file provides a collection of low-level functions for constructing and
 * manipulating C++ data types.   In SWIG, C++ datatypes are encoded as simple
 * text strings.  This representation is compact, easy to debug, and easy to read.
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
 *  The complete type representation for varargs is:
 *  'v(...)'
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
 * For C++, typenames may be parameterized using <(...)>.  Here are some
 * examples:
 *
 *       String Encoding                  C++ Example
 *       ---------------                  ------------
 *       p.vector<(int)>                  vector<int> *
 *       r.foo<(int,p.double)>            foo<int,double *> &
 *
 * Contents of this file:
 *
 * Most of this functions in this file pertain to the low-level manipulation
 * of type objects.   There are constructor functions like this:
 *
 *       SwigType_add_pointer()
 *       SwigType_add_reference()
 *       SwigType_add_rvalue_reference()
 *       SwigType_add_array()
 *
 * These are used to build new types.  There are also functions to undo these
 * operations.  For example:
 *
 *       SwigType_del_pointer()
 *       SwigType_del_reference()
 *       SwigType_del_rvalue_reference()
 *       SwigType_del_array()
 *
 * In addition, there are query functions
 *
 *       SwigType_ispointer()
 *       SwigType_isreference()
 *       SwigType_isrvalue_reference()
 *       SwigType_isarray()
 *
 * Finally, there are some data extraction functions that can be used to
 * extract array dimensions, template arguments, and so forth.
 * 
 * It is very important for developers to realize that the functions in this
 * module do *NOT* incorporate higher-level type system features like typedef.
 * For example, you could have C code like this:
 *
 *        typedef  int  *intptr;
 *       
 * In this case, a SwigType of type 'intptr' will be treated as a simple type and
 * functions like SwigType_ispointer() will evaluate as false.  It is strongly
 * advised that developers use the TypeSys_* interface to check types in a more
 * reliable manner.
 * ----------------------------------------------------------------------------- */


/* -----------------------------------------------------------------------------
 * NewSwigType()
 *
 * Constructs a new type object.   Eventually, it would be nice for this function
 * to accept an initial value in the form a C/C++ abstract type (currently unimplemented).
 * ----------------------------------------------------------------------------- */

#ifdef NEW
SwigType *NewSwigType(const_String_or_char_ptr initial) {
  return NewString(initial);
}

#endif

/* The next few functions are utility functions used in the construction and 
   management of types */

/* -----------------------------------------------------------------------------
 * static element_size()
 *
 * This utility function finds the size of a single type element in a type string.
 * Type elements are always delimited by periods, but may be nested with
 * parentheses.  A nested element is always handled as a single item.
 *
 * Returns the integer size of the element (which can be used to extract a 
 * substring, to chop the element off, or for other purposes).
 * ----------------------------------------------------------------------------- */

static int element_size(char *c) {
  int nparen;
  char *s = c;
  while (*c) {
    if (*c == '.') {
      c++;
      return (int) (c - s);
    } else if (*c == '(') {
      nparen = 1;
      c++;
      while (*c) {
	if (*c == '(')
	  nparen++;
	if (*c == ')') {
	  nparen--;
	  if (nparen == 0)
	    break;
	}
	c++;
      }
    }
    if (*c)
      c++;
  }
  return (int) (c - s);
}

/* -----------------------------------------------------------------------------
 * SwigType_del_element()
 *
 * Deletes one type element from the type.  
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_del_element(SwigType *t) {
  int sz = element_size(Char(t));
  Delslice(t, 0, sz);
  return t;
}

/* -----------------------------------------------------------------------------
 * SwigType_pop()
 * 
 * Pop one type element off the type.
 * Example: t in:  q(const).p.Integer
 *          t out: p.Integer
 *	   result: q(const).
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_pop(SwigType *t) {
  SwigType *result;
  char *c;
  int sz;

  c = Char(t);
  if (!*c)
    return 0;

  sz = element_size(c);
  result = NewStringWithSize(c, sz);
  Delslice(t, 0, sz);
  c = Char(t);
  if (*c == '.') {
    Delitem(t, 0);
  }
  return result;
}

/* -----------------------------------------------------------------------------
 * SwigType_parm()
 *
 * Returns the parameter of an operator as a string
 * ----------------------------------------------------------------------------- */

String *SwigType_parm(const SwigType *t) {
  char *start, *c;
  int nparens = 0;

  c = Char(t);
  while (*c && (*c != '(') && (*c != '.'))
    c++;
  if (!*c || (*c == '.'))
    return 0;
  c++;
  start = c;
  while (*c) {
    if (*c == ')') {
      if (nparens == 0)
	break;
      nparens--;
    } else if (*c == '(') {
      nparens++;
    }
    c++;
  }
  return NewStringWithSize(start, (int) (c - start));
}

/* -----------------------------------------------------------------------------
 * SwigType_split()
 *
 * Splits a type into its component parts and returns a list of string.
 * ----------------------------------------------------------------------------- */

List *SwigType_split(const SwigType *t) {
  String *item;
  List *list;
  char *c;
  int len;

  c = Char(t);
  list = NewList();
  while (*c) {
    len = element_size(c);
    item = NewStringWithSize(c, len);
    Append(list, item);
    Delete(item);
    c = c + len;
    if (*c == '.')
      c++;
  }
  return list;
}

/* -----------------------------------------------------------------------------
 * SwigType_parmlist()
 *
 * Splits a comma separated list of parameters into its component parts
 * The input is expected to contain the parameter list within () brackets
 * Returns 0 if no argument list in the input, ie there are no round brackets ()
 * Returns an empty List if there are no parameters in the () brackets
 * For example:
 *
 *     Foo(std::string,p.f().Bar<(int,double)>)
 *
 * returns 2 elements in the list:
 *    std::string
 *    p.f().Bar<(int,double)>
 * ----------------------------------------------------------------------------- */
 
List *SwigType_parmlist(const String *p) {
  String *item = 0;
  List *list;
  char *c;
  char *itemstart;
  int size;

  assert(p);
  c = Char(p);
  while (*c && (*c != '(') && (*c != '.'))
    c++;
  if (!*c)
    return 0;
  assert(*c != '.'); /* p is expected to contain sub elements of a type */
  c++;
  list = NewList();
  itemstart = c;
  while (*c) {
    if (*c == ',') {
      size = (int) (c - itemstart);
      item = NewStringWithSize(itemstart, size);
      Append(list, item);
      Delete(item);
      itemstart = c + 1;
    } else if (*c == '(') {
      int nparens = 1;
      c++;
      while (*c) {
	if (*c == '(')
	  nparens++;
	if (*c == ')') {
	  nparens--;
	  if (nparens == 0)
	    break;
	}
	c++;
      }
    } else if (*c == ')') {
      break;
    }
    if (*c)
      c++;
  }
  size = (int) (c - itemstart);
  if (size > 0) {
    item = NewStringWithSize(itemstart, size);
    Append(list, item);
  }
  Delete(item);
  return list;
}

/* -----------------------------------------------------------------------------
 *                                 Pointers
 *
 * SwigType_add_pointer()
 * SwigType_del_pointer()
 * SwigType_ispointer()
 *
 * Add, remove, and test if a type is a pointer.  The deletion and query
 * functions take into account qualifiers (if any).
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_add_pointer(SwigType *t) {
  Insert(t, 0, "p.");
  return t;
}

SwigType *SwigType_del_pointer(SwigType *t) {
  char *c, *s;
  c = Char(t);
  s = c;
  /* Skip qualifiers, if any */
  if (strncmp(c, "q(", 2) == 0) {
    c = strchr(c, '.');
    assert(c);
    c++;
  }
  if (strncmp(c, "p.", 2)) {
    printf("Fatal error. SwigType_del_pointer applied to non-pointer.\n");
    abort();
  }
  Delslice(t, 0, (int)((c - s) + 2));
  return t;
}

int SwigType_ispointer(const SwigType *t) {
  char *c;
  if (!t)
    return 0;
  c = Char(t);
  /* Skip qualifiers, if any */
  if (strncmp(c, "q(", 2) == 0) {
    c = strchr(c, '.');
    if (!c)
      return 0;
    c++;
  }
  if (strncmp(c, "p.", 2) == 0) {
    return 1;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 *                                 References
 *
 * SwigType_add_reference()
 * SwigType_del_reference()
 * SwigType_isreference()
 *
 * Add, remove, and test if a type is a reference.  The deletion and query
 * functions take into account qualifiers (if any).
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_add_reference(SwigType *t) {
  Insert(t, 0, "r.");
  return t;
}

SwigType *SwigType_del_reference(SwigType *t) {
  char *c = Char(t);
  int check = strncmp(c, "r.", 2);
  assert(check == 0);
  Delslice(t, 0, 2);
  return t;
}

int SwigType_isreference(const SwigType *t) {
  char *c;
  if (!t)
    return 0;
  c = Char(t);
  if (strncmp(c, "r.", 2) == 0) {
    return 1;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 *                                 Rvalue References
 *
 * SwigType_add_rvalue_reference()
 * SwigType_del_rvalue_reference()
 * SwigType_isrvalue_reference()
 *
 * Add, remove, and test if a type is a rvalue reference.  The deletion and query
 * functions take into account qualifiers (if any).
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_add_rvalue_reference(SwigType *t) {
  Insert(t, 0, "z.");
  return t;
}

SwigType *SwigType_del_rvalue_reference(SwigType *t) {
  char *c = Char(t);
  int check = strncmp(c, "z.", 2);
  assert(check == 0);
  Delslice(t, 0, 2);
  return t;
}

int SwigType_isrvalue_reference(const SwigType *t) {
  char *c;
  if (!t)
    return 0;
  c = Char(t);
  if (strncmp(c, "z.", 2) == 0) {
    return 1;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 *                                  Qualifiers
 *
 * SwigType_add_qualifier()
 * SwigType_del_qualifier()
 * SwigType_is_qualifier()
 *
 * Adds type qualifiers like "const" and "volatile".   When multiple qualifiers
 * are added to a type, they are combined together into a single qualifier.
 * Repeated qualifications have no effect.  Moreover, the order of qualifications
 * is alphabetical---meaning that "const volatile" and "volatile const" are
 * stored in exactly the same way as "q(const volatile)".
 * 'qual' can be a list of multiple qualifiers in any order, separated by spaces.
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_add_qualifier(SwigType *t, const_String_or_char_ptr qual) {
  List *qlist;
  String *allq, *newq;
  int i, sz;
  const char *cqprev = 0;
  const char *c = Char(t);
  const char *cqual = Char(qual);

  /* if 't' has no qualifiers and 'qual' is a single qualifier, simply add it */
  if ((strncmp(c, "q(", 2) != 0) && (strstr(cqual, " ") == 0)) {
    String *temp = NewStringf("q(%s).", cqual);
    Insert(t, 0, temp);
    Delete(temp);
    return t;
  }

  /* create string of all qualifiers */
  if (strncmp(c, "q(", 2) == 0) {
    allq = SwigType_parm(t);
    Append(allq, " ");
    SwigType_del_element(t);     /* delete old qualifier list from 't' */
  } else {
    allq = NewStringEmpty();
  }
  Append(allq, qual);

  /* create list of all qualifiers from string */
  qlist = Split(allq, ' ', INT_MAX);
  Delete(allq);

  /* sort in alphabetical order */
  SortList(qlist, Strcmp);

  /* create new qualifier string from unique elements of list */
  sz = Len(qlist);
  newq = NewString("q(");
  for (i = 0; i < sz; ++i) {
    String *q = Getitem(qlist, i);
    const char *cq = Char(q);
    if (cqprev == 0 || strcmp(cqprev, cq) != 0) {
      if (i > 0) {
        Append(newq, " ");
      }
      Append(newq, q);
      cqprev = cq;
    }
  }
  Append(newq, ").");
  Delete(qlist);

  /* replace qualifier string with new one */
  Insert(t, 0, newq);
  Delete(newq);
  return t;
}

SwigType *SwigType_del_qualifier(SwigType *t) {
  char *c = Char(t);
  int check = strncmp(c, "q(", 2);
  assert(check == 0);
  Delslice(t, 0, element_size(c));
  return t;
}

int SwigType_isqualifier(const SwigType *t) {
  char *c;
  if (!t)
    return 0;
  c = Char(t);
  if (strncmp(c, "q(", 2) == 0) {
    return 1;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 *                                Function Pointers
 * ----------------------------------------------------------------------------- */

int SwigType_isfunctionpointer(const SwigType *t) {
  char *c;
  if (!t)
    return 0;
  c = Char(t);
  if (strncmp(c, "p.f(", 4) == 0) {
    return 1;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 * SwigType_functionpointer_decompose
 *
 * Decompose the function pointer into the parameter list and the return type
 * t - input and on completion contains the return type
 * returns the function's parameters
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_functionpointer_decompose(SwigType *t) {
  String *p;
  assert(SwigType_isfunctionpointer(t));
  p = SwigType_pop(t);
  Delete(p);
  p = SwigType_pop(t);
  return p;
}

/* -----------------------------------------------------------------------------
 *                                Member Pointers
 *
 * SwigType_add_memberpointer()
 * SwigType_del_memberpointer()
 * SwigType_ismemberpointer()
 *
 * Add, remove, and test for C++ pointer to members.
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_add_memberpointer(SwigType *t, const_String_or_char_ptr name) {
  String *temp = NewStringf("m(%s).", name);
  Insert(t, 0, temp);
  Delete(temp);
  return t;
}

SwigType *SwigType_del_memberpointer(SwigType *t) {
  char *c = Char(t);
  int check = strncmp(c, "m(", 2);
  assert(check == 0);
  Delslice(t, 0, element_size(c));
  return t;
}

int SwigType_ismemberpointer(const SwigType *t) {
  char *c;
  if (!t)
    return 0;
  c = Char(t);
  if (strncmp(c, "m(", 2) == 0) {
    return 1;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 *                                    Arrays
 *
 * SwigType_add_array()
 * SwigType_del_array()
 * SwigType_isarray()
 *
 * Utility functions:
 *
 * SwigType_array_ndim()        - Calculate number of array dimensions.
 * SwigType_array_getdim()      - Get array dimension
 * SwigType_array_setdim()      - Set array dimension
 * SwigType_array_type()        - Return array type
 * SwigType_pop_arrays()        - Remove all arrays
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_add_array(SwigType *t, const_String_or_char_ptr size) {
  String *temp = NewString("a(");
  Append(temp, size);
  Append(temp, ").");
  Insert(t, 0, temp);
  Delete(temp);
  return t;
}

SwigType *SwigType_del_array(SwigType *t) {
  char *c = Char(t);
  int check = strncmp(c, "a(", 2);
  assert(check == 0);
  Delslice(t, 0, element_size(c));
  return t;
}

int SwigType_isarray(const SwigType *t) {
  char *c;
  if (!t)
    return 0;
  c = Char(t);
  if (strncmp(c, "a(", 2) == 0) {
    return 1;
  }
  return 0;
}
/*
 * SwigType_prefix_is_simple_1D_array
 *
 * Determine if the type is a 1D array type that is treated as a pointer within SWIG
 * eg Foo[], Foo[3] return true, but Foo[3][3], Foo*[], Foo*[3], Foo**[] return false
 */
int SwigType_prefix_is_simple_1D_array(const SwigType *t) {
  char *c = Char(t);

  if (c && (strncmp(c, "a(", 2) == 0)) {
    c = strchr(c, '.');
    if (c)
      return (*(++c) == 0);
  }
  return 0;
}


/* Remove all arrays */
SwigType *SwigType_pop_arrays(SwigType *t) {
  String *ta;
  assert(SwigType_isarray(t));
  ta = NewStringEmpty();
  while (SwigType_isarray(t)) {
    SwigType *td = SwigType_pop(t);
    Append(ta, td);
    Delete(td);
  }
  return ta;
}

/* Return number of array dimensions */
int SwigType_array_ndim(const SwigType *t) {
  int ndim = 0;
  char *c = Char(t);

  while (c && (strncmp(c, "a(", 2) == 0)) {
    c = strchr(c, '.');
    if (c) {
      c++;
      ndim++;
    }
  }
  return ndim;
}

/* Get nth array dimension */
String *SwigType_array_getdim(const SwigType *t, int n) {
  char *c = Char(t);
  while (c && (strncmp(c, "a(", 2) == 0) && (n > 0)) {
    c = strchr(c, '.');
    if (c) {
      c++;
      n--;
    }
  }
  if (n == 0) {
    String *dim = SwigType_parm(c);
    if (SwigType_istemplate(dim)) {
      String *ndim = SwigType_namestr(dim);
      Delete(dim);
      dim = ndim;
    }

    return dim;
  }

  return 0;
}

/* Replace nth array dimension */
void SwigType_array_setdim(SwigType *t, int n, const_String_or_char_ptr rep) {
  String *result = 0;
  char temp;
  char *start;
  char *c = Char(t);

  start = c;
  if (strncmp(c, "a(", 2))
    abort();

  while (c && (strncmp(c, "a(", 2) == 0) && (n > 0)) {
    c = strchr(c, '.');
    if (c) {
      c++;
      n--;
    }
  }
  if (n == 0) {
    temp = *c;
    *c = 0;
    result = NewString(start);
    Printf(result, "a(%s)", rep);
    *c = temp;
    c = strchr(c, '.');
    Append(result, c);
  }
  Clear(t);
  Append(t, result);
  Delete(result);
}

/* Return base type of an array */
SwigType *SwigType_array_type(const SwigType *ty) {
  SwigType *t;
  t = Copy(ty);
  while (SwigType_isarray(t)) {
    Delete(SwigType_pop(t));
  }
  return t;
}


/* -----------------------------------------------------------------------------
 *                                    Functions
 *
 * SwigType_add_function()
 * SwigType_del_function()
 * SwigType_isfunction()
 * SwigType_pop_function()
 *
 * Add, remove, and test for function types.
 * ----------------------------------------------------------------------------- */

/* Returns the function type, t, constructed from the parameters, parms */
SwigType *SwigType_add_function(SwigType *t, ParmList *parms) {
  String *pstr;
  Parm *p;

  Insert(t, 0, ").");
  pstr = NewString("f(");
  for (p = parms; p; p = nextSibling(p)) {
    if (p != parms)
      Putc(',', pstr);
    Append(pstr, Getattr(p, "type"));
  }
  Insert(t, 0, pstr);
  Delete(pstr);
  return t;
}

SwigType *SwigType_pop_function(SwigType *t) {
  SwigType *f = 0;
  SwigType *g = 0;
  char *c = Char(t);
  if (strncmp(c, "q(", 2) == 0) {
    f = SwigType_pop(t);
    c = Char(t);
  }
  if (strncmp(c, "f(", 2)) {
    printf("Fatal error. SwigType_pop_function applied to non-function.\n");
    abort();
  }
  g = SwigType_pop(t);
  if (f)
    SwigType_push(g, f);
  Delete(f);
  return g;
}

int SwigType_isfunction(const SwigType *t) {
  char *c;
  if (!t) {
    return 0;
  }
  c = Char(t);
  if (strncmp(c, "q(", 2) == 0) {
    /* Might be a 'const' function.  Try to skip over the 'const' */
    c = strchr(c, '.');
    if (c)
      c++;
    else
      return 0;
  }
  if (strncmp(c, "f(", 2) == 0) {
    return 1;
  }
  return 0;
}

/* Create a list of parameters from the type t, using the file_line_node Node for 
 * file and line numbering for the parameters */
ParmList *SwigType_function_parms(const SwigType *t, Node *file_line_node) {
  List *l = SwigType_parmlist(t);
  Hash *p, *pp = 0, *firstp = 0;
  Iterator o;

  for (o = First(l); o.item; o = Next(o)) {
    p = file_line_node ? NewParm(o.item, 0, file_line_node) : NewParmWithoutFileLineInfo(o.item, 0);
    if (!firstp)
      firstp = p;
    if (pp) {
      set_nextSibling(pp, p);
      Delete(p);
    }
    pp = p;
  }
  Delete(l);
  return firstp;
}

int SwigType_isvarargs(const SwigType *t) {
  if (Strcmp(t, "v(...)") == 0)
    return 1;
  return 0;
}

/* -----------------------------------------------------------------------------
 *                                    Templates
 *
 * SwigType_add_template()
 *
 * Template handling.
 * ----------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * SwigType_add_template()
 *
 * Adds a template to a type.   This template is encoded in the SWIG type
 * mechanism and produces a string like this:
 *
 *  vector<int *> ----> "vector<(p.int)>"
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_add_template(SwigType *t, ParmList *parms) {
  Parm *p;

  Append(t, "<(");
  for (p = parms; p; p = nextSibling(p)) {
    String *v;
    if (Getattr(p, "default"))
      continue;
    if (p != parms)
      Append(t, ",");
    v = Getattr(p, "value");
    if (v) {
      Append(t, v);
    } else {
      Append(t, Getattr(p, "type"));
    }
  }
  Append(t, ")>");
  return t;
}


/* -----------------------------------------------------------------------------
 * SwigType_templateprefix()
 *
 * Returns the prefix before the first template definition.
 * Returns the type unmodified if not a template.
 * For example:
 *
 *     Foo<(p.int)>::bar  =>  Foo
 *     r.q(const).Foo<(p.int)>::bar => r.q(const).Foo
 *     Foo => Foo
 * ----------------------------------------------------------------------------- */

String *SwigType_templateprefix(const SwigType *t) {
  const char *s = Char(t);
  const char *c = strstr(s, "<(");
  return c ? NewStringWithSize(s, (int)(c - s)) : NewString(s);
}

/* -----------------------------------------------------------------------------
 * SwigType_templatesuffix()
 *
 * Returns text after a template substitution.  Used to handle scope names
 * for example:
 *
 *        Foo<(p.int)>::bar
 *
 * returns "::bar"
 * ----------------------------------------------------------------------------- */

String *SwigType_templatesuffix(const SwigType *t) {
  const char *c;
  c = Char(t);
  while (*c) {
    if ((*c == '<') && (*(c + 1) == '(')) {
      int nest = 1;
      c++;
      while (*c && nest) {
	if (*c == '<')
	  nest++;
	if (*c == '>')
	  nest--;
	c++;
      }
      return NewString(c);
    }
    c++;
  }
  return NewStringEmpty();
}

/* -----------------------------------------------------------------------------
 * SwigType_istemplate_templateprefix()
 *
 * Combines SwigType_istemplate and SwigType_templateprefix efficiently into one function.
 * Returns the prefix before the first template definition.
 * Returns NULL if not a template.
 * For example:
 *
 *     Foo<(p.int)>::bar  =>  Foo
 *     r.q(const).Foo<(p.int)>::bar => r.q(const).Foo
 *     Foo => NULL
 * ----------------------------------------------------------------------------- */

String *SwigType_istemplate_templateprefix(const SwigType *t) {
  const char *s = Char(t);
  const char *c = strstr(s, "<(");
  return c ? NewStringWithSize(s, (int)(c - s)) : 0;
}

/* -----------------------------------------------------------------------------
 * SwigType_istemplate_only_templateprefix()
 *
 * Similar to SwigType_istemplate_templateprefix() but only returns the template
 * prefix if the type is just the template and not a subtype/symbol within the template.
 * Returns NULL if not a template or is a template with a symbol within the template.
 * For example:
 *
 *     Foo<(p.int)>  =>  Foo
 *     Foo<(p.int)>::bar  =>  NULL
 *     r.q(const).Foo<(p.int)> => r.q(const).Foo
 *     r.q(const).Foo<(p.int)>::bar => NULL
 *     Foo => NULL
 * ----------------------------------------------------------------------------- */

String *SwigType_istemplate_only_templateprefix(const SwigType *t) {
  int len = Len(t);
  const char *s = Char(t);
  if (len >= 4 && strcmp(s + len - 2, ")>") == 0) {
    const char *c = strstr(s, "<(");
    return c ? NewStringWithSize(s, (int)(c - s)) : 0;
  } else {
    return 0;
  }
}

/* -----------------------------------------------------------------------------
 * SwigType_templateargs()
 *
 * Returns the template arguments
 * For example:
 *
 *     Foo<(p.int)>::bar
 *
 * returns "<(p.int)>"
 * ----------------------------------------------------------------------------- */

String *SwigType_templateargs(const SwigType *t) {
  const char *c;
  const char *start;
  c = Char(t);
  while (*c) {
    if ((*c == '<') && (*(c + 1) == '(')) {
      int nest = 1;
      start = c;
      c++;
      while (*c && nest) {
	if (*c == '<')
	  nest++;
	if (*c == '>')
	  nest--;
	c++;
      }
      return NewStringWithSize(start, (int)(c - start));
    }
    c++;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 * SwigType_istemplate()
 *
 * Tests a type to see if it includes template parameters
 * ----------------------------------------------------------------------------- */

int SwigType_istemplate(const SwigType *t) {
  char *ct = Char(t);
  ct = strstr(ct, "<(");
  if (ct && (strstr(ct + 2, ")>")))
    return 1;
  return 0;
}

/* -----------------------------------------------------------------------------
 * SwigType_base()
 *
 * This function returns the base of a type.  For example, if you have a
 * type "p.p.int", the function would return "int".
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_base(const SwigType *t) {
  char *c;
  char *lastop = 0;
  c = Char(t);

  lastop = c;

  /* Search for the last type constructor separator '.' */
  while (*c) {
    if (*c == '.') {
      if (*(c + 1)) {
	lastop = c + 1;
      }
      c++;
      continue;
    }
    if (*c == '<') {
      /* Skip over template---it's part of the base name */
      int ntemp = 1;
      c++;
      while ((*c) && (ntemp > 0)) {
	if (*c == '>')
	  ntemp--;
	else if (*c == '<')
	  ntemp++;
	c++;
      }
      if (ntemp)
	break;
      continue;
    }
    if (*c == '(') {
      /* Skip over params */
      int nparen = 1;
      c++;
      while ((*c) && (nparen > 0)) {
	if (*c == '(')
	  nparen++;
	else if (*c == ')')
	  nparen--;
	c++;
      }
      if (nparen)
	break;
      continue;
    }
    c++;
  }
  return NewString(lastop);
}

/* -----------------------------------------------------------------------------
 * SwigType_prefix()
 *
 * Returns the prefix of a datatype.  For example, the prefix of the
 * type "p.p.int" is "p.p.".
 * ----------------------------------------------------------------------------- */

String *SwigType_prefix(const SwigType *t) {
  char *c, *d;
  String *r = 0;

  c = Char(t);
  d = c + strlen(c);

  /* Check for a type constructor */
  if ((d > c) && (*(d - 1) == '.'))
    d--;

  while (d > c) {
    d--;
    if (*d == '>') {
      int nest = 1;
      d--;
      while ((d > c) && (nest)) {
	if (*d == '>')
	  nest++;
	if (*d == '<')
	  nest--;
	d--;
      }
    }
    if (*d == ')') {
      /* Skip over params */
      int nparen = 1;
      d--;
      while ((d > c) && (nparen)) {
	if (*d == ')')
	  nparen++;
	if (*d == '(')
	  nparen--;
	d--;
      }
    }

    if (*d == '.') {
      char t = *(d + 1);
      *(d + 1) = 0;
      r = NewString(c);
      *(d + 1) = t;
      return r;
    }
  }
  return NewStringEmpty();
}

/* -----------------------------------------------------------------------------
 * SwigType_strip_qualifiers()
 * 
 * Strip all qualifiers from a type and return a new type
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_strip_qualifiers(const SwigType *t) {
  static Hash *memoize_stripped = 0;
  SwigType *r;
  List *l;
  Iterator ei;

  if (!memoize_stripped)
    memoize_stripped = NewHash();
  r = Getattr(memoize_stripped, t);
  if (r)
    return Copy(r);

  l = SwigType_split(t);
  r = NewStringEmpty();

  for (ei = First(l); ei.item; ei = Next(ei)) {
    if (SwigType_isqualifier(ei.item))
      continue;
    Append(r, ei.item);
  }
  Delete(l);
  {
    String *key, *value;
    key = Copy(t);
    value = Copy(r);
    Setattr(memoize_stripped, key, value);
    Delete(key);
    Delete(value);
  }
  return r;
}

/* -----------------------------------------------------------------------------
 * SwigType_strip_single_qualifier()
 * 
 * If the type contains a qualifier, strip one qualifier and return a new type.
 * The left most qualifier is stripped first (when viewed as C source code) but
 * this is the equivalent to the right most qualifier using SwigType notation.
 * Example: 
 *    r.q(const).p.q(const).int => r.q(const).p.int
 *    r.q(const).p.int          => r.p.int
 *    r.p.int                   => r.p.int
 * ----------------------------------------------------------------------------- */

SwigType *SwigType_strip_single_qualifier(const SwigType *t) {
  static Hash *memoize_stripped = 0;
  SwigType *r = 0;
  List *l;
  int numitems;

  if (!memoize_stripped)
    memoize_stripped = NewHash();
  r = Getattr(memoize_stripped, t);
  if (r)
    return Copy(r);

  l = SwigType_split(t);

  numitems = Len(l);
  if (numitems >= 2) {
    int item;
    /* iterate backwards from last but one item */
    for (item = numitems - 2; item >= 0; --item) {
      String *subtype = Getitem(l, item);
      if (SwigType_isqualifier(subtype)) {
	Iterator it;
	Delitem(l, item);
	r = NewStringEmpty();
	for (it = First(l); it.item; it = Next(it)) {
	  Append(r, it.item);
	}
	break;
      }
    }
  }
  if (!r)
    r = Copy(t);

  Delete(l);
  {
    String *key, *value;
    key = Copy(t);
    value = Copy(r);
    Setattr(memoize_stripped, key, value);
    Delete(key);
    Delete(value);
  }
  return r;
}

