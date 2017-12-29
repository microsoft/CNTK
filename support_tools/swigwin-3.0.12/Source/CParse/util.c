/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * util.c
 *
 * Parsing utilities.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "cparse.h"

/* -----------------------------------------------------------------------------
 * Swig_cparse_replace_descriptor()
 *
 * Replaces type descriptor string $descriptor() with the SWIG type descriptor
 * string.
 * ----------------------------------------------------------------------------- */

void Swig_cparse_replace_descriptor(String *s) {
  char tmp[512];
  String *arg = 0;
  SwigType *t;
  char *c = 0;

  while ((c = strstr(Char(s), "$descriptor("))) {
    char *d = tmp;
    int level = 0;
    while (*c) {
      if (*c == '(')
	level++;
      if (*c == ')') {
	level--;
	if (level == 0) {
	  break;
	}
      }
      *d = *c;
      d++;
      c++;
    }
    *d = 0;
    arg = NewString(tmp + 12);
    t = Swig_cparse_type(arg);
    Delete(arg);
    arg = 0;

    if (t) {
      String *mangle;
      String *descriptor;

      mangle = SwigType_manglestr(t);
      descriptor = NewStringf("SWIGTYPE%s", mangle);
      SwigType_remember(t);
      *d = ')';
      d++;
      *d = 0;
      Replace(s, tmp, descriptor, DOH_REPLACE_ANY);
      Delete(mangle);
      Delete(descriptor);
      Delete(t);
    } else {
      Swig_error(Getfile(s), Getline(s), "Bad $descriptor() macro.\n");
      break;
    }
  }
}

/* -----------------------------------------------------------------------------
 * Swig_cparse_smartptr()
 *
 * Parse the type in smartptr feature and convert into a SwigType.
 * Error out if the parsing fails as this is like a parser syntax error.
 * ----------------------------------------------------------------------------- */

SwigType *Swig_cparse_smartptr(Node *n) {
    SwigType *smart = 0;
    String *smartptr = Getattr(n, "feature:smartptr");
    if (smartptr) {
      SwigType *cpt = Swig_cparse_type(smartptr);
      if (cpt) {
	smart = SwigType_typedef_resolve_all(cpt);
	Delete(cpt);
      } else {
	Swig_error(Getfile(n), Getline(n), "Invalid type (%s) in 'smartptr' feature for class %s.\n", smartptr, SwigType_namestr(Getattr(n, "name")));
      }
    }
    return smart;
}

/* -----------------------------------------------------------------------------
 * cparse_normalize_void()
 *
 * This function is used to replace arguments of the form (void) with empty
 * arguments in C++
 * ----------------------------------------------------------------------------- */

void cparse_normalize_void(Node *n) {
  String *decl = Getattr(n, "decl");
  Parm *parms = Getattr(n, "parms");

  if (SwigType_isfunction(decl)) {
    if ((ParmList_len(parms) == 1) && (SwigType_type(Getattr(parms, "type")) == T_VOID)) {
      Replaceall(decl, "f(void).", "f().");
      Delattr(n, "parms");
    }
  }
}

/* -----------------------------------------------------------------------------
 * new_node()
 *
 * Create an empty parse node, setting file and line number information
 * ----------------------------------------------------------------------------- */

Node *new_node(const_String_or_char_ptr tag) {
  Node *n = NewHash();
  set_nodeType(n,tag);
  Setfile(n,cparse_file);
  Setline(n,cparse_line);
  return n;
}
