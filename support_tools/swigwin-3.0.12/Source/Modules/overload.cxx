/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * overload.cxx
 *
 * This file is used to analyze overloaded functions and methods.
 * It looks at signatures and tries to gather information for
 * building a dispatch function.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

#define MAX_OVERLOAD 4096

/* Overload "argc" and "argv" */
String *argv_template_string;
String *argc_template_string;

struct Overloaded {
  Node *n;			/* Node                               */
  int argc;			/* Argument count                     */
  ParmList *parms;		/* Parameters used for overload check */
  int error;			/* Ambiguity error                    */
  bool implicitconv_function;	/* For ordering implicitconv functions*/
};

static int fast_dispatch_mode = 0;
static int cast_dispatch_mode = 0;

/* Set fast_dispatch_mode */
void Wrapper_fast_dispatch_mode_set(int flag) {
  fast_dispatch_mode = flag;
}

void Wrapper_cast_dispatch_mode_set(int flag) {
  cast_dispatch_mode = flag;
}

/* -----------------------------------------------------------------------------
 * mark_implicitconv_function()
 *
 * Mark function if it contains an implicitconv type in the parameter list
 * ----------------------------------------------------------------------------- */
static void mark_implicitconv_function(Overloaded& onode) {
  Parm *parms = onode.parms;
  if (parms) {
    bool is_implicitconv_function = false;
    Parm *p = parms;
    while (p) {
      if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
	continue;
      }
      if (GetFlag(p, "implicitconv")) {
	is_implicitconv_function = true;
	break;
      }
      p = nextSibling(p);
    }
    if (is_implicitconv_function)
      onode.implicitconv_function = true;
  }
}

/* -----------------------------------------------------------------------------
 * Swig_overload_rank()
 *
 * This function takes an overloaded declaration and creates a list that ranks
 * all overloaded methods in an order that can be used to generate a dispatch 
 * function.
 * Slight difference in the way this function is used by scripting languages and
 * statically typed languages. The script languages call this method via 
 * Swig_overload_dispatch() - where wrappers for all overloaded methods are generated,
 * however sometimes the code can never be executed. The non-scripting languages
 * call this method via Swig_overload_check() for each overloaded method in order
 * to determine whether or not the method should be wrapped. Note the slight
 * difference when overloading methods that differ by const only. The
 * scripting languages will ignore the const method, whereas the non-scripting
 * languages ignore the first method parsed.
 * ----------------------------------------------------------------------------- */

List *Swig_overload_rank(Node *n, bool script_lang_wrapping) {
  Overloaded nodes[MAX_OVERLOAD];
  int nnodes = 0;
  Node *o = Getattr(n, "sym:overloaded");
  Node *c;

  if (!o)
    return 0;

  c = o;
  while (c) {
    if (Getattr(c, "error")) {
      c = Getattr(c, "sym:nextSibling");
      continue;
    }
    /*    if (SmartPointer && Getattr(c,"cplus:staticbase")) {
       c = Getattr(c,"sym:nextSibling");
       continue;
       } */

    /* Make a list of all the declarations (methods) that are overloaded with
     * this one particular method name */
    if (Getattr(c, "wrap:name")) {
      assert(nnodes < MAX_OVERLOAD);
      nodes[nnodes].n = c;
      nodes[nnodes].parms = Getattr(c, "wrap:parms");
      nodes[nnodes].argc = emit_num_required(nodes[nnodes].parms);
      nodes[nnodes].error = 0;
      nodes[nnodes].implicitconv_function = false;

      mark_implicitconv_function(nodes[nnodes]);
      nnodes++;
    }
    c = Getattr(c, "sym:nextSibling");
  }

  /* Sort the declarations by required argument count */
  {
    int i, j;
    for (i = 0; i < nnodes; i++) {
      for (j = i + 1; j < nnodes; j++) {
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
    int i, j;
    for (i = 0; i < nnodes - 1; i++) {
      if (nodes[i].argc == nodes[i + 1].argc) {
	for (j = i + 1; (j < nnodes) && (nodes[j].argc == nodes[i].argc); j++) {
	  Parm *p1 = nodes[i].parms;
	  Parm *p2 = nodes[j].parms;
	  int differ = 0;
	  int num_checked = 0;
	  while (p1 && p2 && (num_checked < nodes[i].argc)) {
	    //    Printf(stdout,"p1 = '%s', p2 = '%s'\n", Getattr(p1,"type"), Getattr(p2,"type"));
	    if (checkAttribute(p1, "tmap:in:numinputs", "0")) {
	      p1 = Getattr(p1, "tmap:in:next");
	      continue;
	    }
	    if (checkAttribute(p2, "tmap:in:numinputs", "0")) {
	      p2 = Getattr(p2, "tmap:in:next");
	      continue;
	    }
	    String *t1 = Getattr(p1, "tmap:typecheck:precedence");
	    String *t2 = Getattr(p2, "tmap:typecheck:precedence");
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
	      differ = t1v - t2v;
	    } else if (!t1 && t2)
	      differ = 1;
	    else if (t1 && !t2)
	      differ = -1;
	    else if (!t1 && !t2)
	      differ = -1;
	    num_checked++;
	    if (differ > 0) {
	      Overloaded t = nodes[i];
	      nodes[i] = nodes[j];
	      nodes[j] = t;
	      break;
	    } else if ((differ == 0) && (Strcmp(t1, "0") == 0)) {
	      t1 = Getattr(p1, "ltype");
	      if (!t1) {
		t1 = SwigType_ltype(Getattr(p1, "type"));
		if (Getattr(p1, "tmap:typecheck:SWIGTYPE")) {
		  SwigType_add_pointer(t1);
		}
		Setattr(p1, "ltype", t1);
	      }
	      t2 = Getattr(p2, "ltype");
	      if (!t2) {
		t2 = SwigType_ltype(Getattr(p2, "type"));
		if (Getattr(p2, "tmap:typecheck:SWIGTYPE")) {
		  SwigType_add_pointer(t2);
		}
		Setattr(p2, "ltype", t2);
	      }

	      /* Need subtype check here.  If t2 is a subtype of t1, then we need to change the
	         order */

	      if (SwigType_issubtype(t2, t1)) {
		Overloaded t = nodes[i];
		nodes[i] = nodes[j];
		nodes[j] = t;
	      }

	      if (Strcmp(t1, t2) != 0) {
		differ = 1;
		break;
	      }
	    } else if (differ) {
	      break;
	    }
	    if (Getattr(p1, "tmap:in:next")) {
	      p1 = Getattr(p1, "tmap:in:next");
	    } else {
	      p1 = nextSibling(p1);
	    }
	    if (Getattr(p2, "tmap:in:next")) {
	      p2 = Getattr(p2, "tmap:in:next");
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
    int argc_changed_index = -1;
    for (i = 0; i < nnodes; i++) {
      if (nodes[i].error)
	Setattr(nodes[i].n, "overload:ignore", "1");
      Append(result, nodes[i].n);
      // Printf(stdout,"[ %d ] %d    %s\n", i, nodes[i].implicitconv_function, ParmList_errorstr(nodes[i].parms));
      // Swig_print_node(nodes[i].n);
      if (i == nnodes-1 || nodes[i].argc != nodes[i+1].argc) {
	if (argc_changed_index+2 < nnodes && (nodes[argc_changed_index+1].argc == nodes[argc_changed_index+2].argc)) {
	  // Add additional implicitconv functions in same order as already ranked.
	  // Consider overloaded functions by argument count... only add additional implicitconv functions if
	  // the number of functions with the same arg count > 1, ie, only if overloaded by same argument count.
	  int j;
	  for (j = argc_changed_index + 1; j <= i; j++) {
	    if (nodes[j].implicitconv_function) {
	      SetFlag(nodes[j].n, "implicitconvtypecheckoff");
	      Append(result, nodes[j].n);
	      // Printf(stdout,"[ %d ] %d +  %s\n", j, nodes[j].implicitconv_function, ParmList_errorstr(nodes[j].parms));
	      // Swig_print_node(nodes[j].n);
	    }
	  }
	}
	argc_changed_index = i;
      }
    }
  }
  return result;
}

// /* -----------------------------------------------------------------------------
//  * print_typecheck()
//  * ----------------------------------------------------------------------------- */

static bool print_typecheck(String *f, int j, Parm *pj, bool implicitconvtypecheckoff) {
  char tmp[256];
  sprintf(tmp, Char(argv_template_string), j);
  String *tm = Getattr(pj, "tmap:typecheck");
  if (tm) {
    tm = Copy(tm);
    Replaceid(tm, Getattr(pj, "lname"), "_v");
    String *conv = Getattr(pj, "implicitconv");
    if (conv && !implicitconvtypecheckoff) {
      Replaceall(tm, "$implicitconv", conv);
    } else {
      Replaceall(tm, "$implicitconv", "0");
    }
    Replaceall(tm, "$input", tmp);
    Printv(f, tm, "\n", NIL);
    Delete(tm);
    return true;
  } else
    return false;
}

/* -----------------------------------------------------------------------------
 * ReplaceFormat()
 * ----------------------------------------------------------------------------- */

static String *ReplaceFormat(const_String_or_char_ptr fmt, int j) {
  String *lfmt = NewString(fmt);
  char buf[50];
  sprintf(buf, "%d", j);
  Replaceall(lfmt, "$numargs", buf);
  int i;
  String *commaargs = NewString("");
  for (i = 0; i < j; i++) {
    Printv(commaargs, ", ", NIL);
    Printf(commaargs, Char(argv_template_string), i);
  }
  Replaceall(lfmt, "$commaargs", commaargs);
  return lfmt;
}

/* -----------------------------------------------------------------------------
 * Swig_overload_dispatch()
 *
 * Generate a dispatch function.  argc is assumed to hold the argument count.
 * argv is the argument vector.
 *
 * Note that for C++ class member functions, Swig_overload_dispatch() assumes
 * that argc includes the "self" argument and that the first element of argv[]
 * is the "self" argument. So for a member function:
 *
 *     Foo::bar(int x, int y, int z);
 *
 * the argc should be 4 (not 3!) and the first element of argv[] would be
 * the appropriate scripting language reference to "self". For regular
 * functions (and static class functions) the argc and argv only include
 * the regular function arguments.
 * ----------------------------------------------------------------------------- */

/*
  Cast dispatch mechanism.
*/
String *Swig_overload_dispatch_cast(Node *n, const_String_or_char_ptr fmt, int *maxargs) {
  int i, j;

  *maxargs = 1;

  String *f = NewString("");
  String *sw = NewString("");
  Printf(f, "{\n");
  Printf(f, "unsigned long _index = 0;\n");
  Printf(f, "SWIG_TypeRank _rank = 0; \n");

  /* Get a list of methods ranked by precedence values and argument count */
  List *dispatch = Swig_overload_rank(n, true);
  int nfunc = Len(dispatch);

  /* Loop over the functions */

  bool emitcheck = 1;
  for (i = 0; i < nfunc; i++) {
    int fn = 0;
    Node *ni = Getitem(dispatch, i);
    Parm *pi = Getattr(ni, "wrap:parms");
    bool implicitconvtypecheckoff = GetFlag(ni, "implicitconvtypecheckoff") != 0;
    int num_required = emit_num_required(pi);
    int num_arguments = emit_num_arguments(pi);
    if (num_arguments > *maxargs)
      *maxargs = num_arguments;

    if (num_required == num_arguments) {
      Printf(f, "if (%s == %d) {\n", argc_template_string, num_required);
    } else {
      Printf(f, "if ((%s >= %d) && (%s <= %d)) {\n", argc_template_string, num_required, argc_template_string, num_arguments);
    }
    Printf(f, "SWIG_TypeRank _ranki = 0;\n");
    Printf(f, "SWIG_TypeRank _rankm = 0;\n");
    if (num_arguments)
      Printf(f, "SWIG_TypeRank _pi = 1;\n");

    /* create a list with the wrappers that collide with the
       current one based on argument number */
    List *coll = NewList();
    for (int k = i + 1; k < nfunc; k++) {
      Node *nk = Getitem(dispatch, k);
      Parm *pk = Getattr(nk, "wrap:parms");
      int nrk = emit_num_required(pk);
      int nak = emit_num_arguments(pk);
      if ((nrk >= num_required && nrk <= num_arguments) || (nak >= num_required && nak <= num_arguments) || (nrk <= num_required && nak >= num_arguments))
	Append(coll, nk);
    }

    // printf("overload: %s coll=%d\n", Char(Getattr(n, "sym:name")), Len(coll));

    int num_braces = 0;
    bool test = (num_arguments > 0);
    if (test) {
      int need_v = 1;
      j = 0;
      Parm *pj = pi;
      while (pj) {
	if (checkAttribute(pj, "tmap:in:numinputs", "0")) {
	  pj = Getattr(pj, "tmap:in:next");
	  continue;
	}

	String *tm = Getattr(pj, "tmap:typecheck");
	if (tm) {
	  tm = Copy(tm);
	  /* normalise for comparison later */
	  Replaceid(tm, Getattr(pj, "lname"), "_v");

	  /* if all the wrappers have the same type check on this
	     argument we can optimize it out */
	  for (int k = 0; k < Len(coll) && !emitcheck; k++) {
	    Node *nk = Getitem(coll, k);
	    Parm *pk = Getattr(nk, "wrap:parms");
	    int nak = emit_num_arguments(pk);
	    if (nak <= j)
	      continue;
	    int l = 0;
	    Parm *pl = pk;
	    /* finds arg j on the collider wrapper */
	    while (pl && l <= j) {
	      if (checkAttribute(pl, "tmap:in:numinputs", "0")) {
		pl = Getattr(pl, "tmap:in:next");
		continue;
	      }
	      if (l == j) {
		/* we are at arg j, so we compare the tmaps now */
		String *tml = Getattr(pl, "tmap:typecheck");
		/* normalise it before comparing */
		if (tml)
		  Replaceid(tml, Getattr(pl, "lname"), "_v");
		if (!tml || Cmp(tm, tml))
		  emitcheck = 1;
		//printf("tmap: %s[%d] (%d) => %s\n\n",
		//       Char(Getattr(nk, "sym:name")),
		//       l, emitcheck, tml?Char(tml):0);
	      }
	      Parm *pl1 = Getattr(pl, "tmap:in:next");
	      if (pl1)
		pl = pl1;
	      else
		pl = nextSibling(pl);
	      l++;
	    }
	  }

	  if (emitcheck) {
	    if (need_v) {
	      Printf(f, "int _v = 0;\n");
	      need_v = 0;
	    }
	    if (j >= num_required) {
	      Printf(f, "if (%s > %d) {\n", argc_template_string, j);
	      num_braces++;
	    }
	    String *tmp = NewStringf(argv_template_string, j);

	    String *conv = Getattr(pj, "implicitconv");
	    if (conv && !implicitconvtypecheckoff) {
	      Replaceall(tm, "$implicitconv", conv);
	    } else {
	      Replaceall(tm, "$implicitconv", "0");
	    }
	    Replaceall(tm, "$input", tmp);
	    Printv(f, "{\n", tm, "}\n", NIL);
	    Delete(tm);
	    fn = i + 1;
	    Printf(f, "if (!_v) goto check_%d;\n", fn);
	    Printf(f, "_ranki += _v*_pi;\n");
	    Printf(f, "_rankm += _pi;\n");
	    Printf(f, "_pi *= SWIG_MAXCASTRANK;\n");
	  }
	}
	if (!Getattr(pj, "tmap:in:SWIGTYPE") && Getattr(pj, "tmap:typecheck:SWIGTYPE")) {
	  /* we emit  a warning if the argument defines the 'in' typemap, but not the 'typecheck' one */
	  Swig_warning(WARN_TYPEMAP_TYPECHECK_UNDEF, Getfile(ni), Getline(ni),
		       "Overloaded method %s with no explicit typecheck typemap for arg %d of type '%s'\n",
		       Swig_name_decl(n), j, SwigType_str(Getattr(pj, "type"), 0));
	}
	Parm *pj1 = Getattr(pj, "tmap:in:next");
	if (pj1)
	  pj = pj1;
	else
	  pj = nextSibling(pj);
	j++;
      }
    }

    /* close braces */
    for ( /* empty */ ; num_braces > 0; num_braces--)
      Printf(f, "}\n");

    Printf(f, "if (!_index || (_ranki < _rank)) {\n");
    Printf(f, " _rank = _ranki; _index = %d;\n", i + 1);
    Printf(f, " if (_rank == _rankm) goto dispatch;\n");
    Printf(f, "}\n");
    String *lfmt = ReplaceFormat(fmt, num_arguments);
    Printf(sw, "case %d:\n", i + 1);
    Printf(sw, Char(lfmt), Getattr(ni, "wrap:name"));
    Printf(sw, "\n");

    Printf(f, "}\n");		/* braces closes "if" for this method */
    if (fn)
      Printf(f, "check_%d:\n\n", fn);

    if (implicitconvtypecheckoff)
      Delattr(ni, "implicitconvtypecheckoff");

    Delete(lfmt);
    Delete(coll);
  }
  Delete(dispatch);
  Printf(f, "dispatch:\n");
  Printf(f, "switch(_index) {\n");
  Printf(f, "%s", sw);
  Printf(f, "}\n");

  Printf(f, "}\n");
  return f;
}

/*
  Fast dispatch mechanism, provided by  Salvador Fandi~no Garc'ia (#930586).
*/
String *Swig_overload_dispatch_fast(Node *n, const_String_or_char_ptr fmt, int *maxargs) {
  int i, j;

  *maxargs = 1;

  String *f = NewString("");

  /* Get a list of methods ranked by precedence values and argument count */
  List *dispatch = Swig_overload_rank(n, true);
  int nfunc = Len(dispatch);

  /* Loop over the functions */

  for (i = 0; i < nfunc; i++) {
    int fn = 0;
    Node *ni = Getitem(dispatch, i);
    Parm *pi = Getattr(ni, "wrap:parms");
    bool implicitconvtypecheckoff = GetFlag(ni, "implicitconvtypecheckoff") != 0;
    int num_required = emit_num_required(pi);
    int num_arguments = emit_num_arguments(pi);
    if (num_arguments > *maxargs)
      *maxargs = num_arguments;

    if (num_required == num_arguments) {
      Printf(f, "if (%s == %d) {\n", argc_template_string, num_required);
    } else {
      Printf(f, "if ((%s >= %d) && (%s <= %d)) {\n", argc_template_string, num_required, argc_template_string, num_arguments);
    }

    /* create a list with the wrappers that collide with the
       current one based on argument number */
    List *coll = NewList();
    for (int k = i + 1; k < nfunc; k++) {
      Node *nk = Getitem(dispatch, k);
      Parm *pk = Getattr(nk, "wrap:parms");
      int nrk = emit_num_required(pk);
      int nak = emit_num_arguments(pk);
      if ((nrk >= num_required && nrk <= num_arguments) || (nak >= num_required && nak <= num_arguments) || (nrk <= num_required && nak >= num_arguments))
	Append(coll, nk);
    }

    // printf("overload: %s coll=%d\n", Char(Getattr(n, "sym:name")), Len(coll));

    int num_braces = 0;
    bool test = (Len(coll) > 0 && num_arguments);
    if (test) {
      int need_v = 1;
      j = 0;
      Parm *pj = pi;
      while (pj) {
	if (checkAttribute(pj, "tmap:in:numinputs", "0")) {
	  pj = Getattr(pj, "tmap:in:next");
	  continue;
	}

	String *tm = Getattr(pj, "tmap:typecheck");
	if (tm) {
	  tm = Copy(tm);
	  /* normalise for comparison later */
	  Replaceid(tm, Getattr(pj, "lname"), "_v");

	  /* if all the wrappers have the same type check on this
	     argument we can optimize it out */
	  bool emitcheck = 0;
	  for (int k = 0; k < Len(coll) && !emitcheck; k++) {
	    Node *nk = Getitem(coll, k);
	    Parm *pk = Getattr(nk, "wrap:parms");
	    int nak = emit_num_arguments(pk);
	    if (nak <= j)
	      continue;
	    int l = 0;
	    Parm *pl = pk;
	    /* finds arg j on the collider wrapper */
	    while (pl && l <= j) {
	      if (checkAttribute(pl, "tmap:in:numinputs", "0")) {
		pl = Getattr(pl, "tmap:in:next");
		continue;
	      }
	      if (l == j) {
		/* we are at arg j, so we compare the tmaps now */
		String *tml = Getattr(pl, "tmap:typecheck");
		/* normalise it before comparing */
		if (tml)
		  Replaceid(tml, Getattr(pl, "lname"), "_v");
		if (!tml || Cmp(tm, tml))
		  emitcheck = 1;
		//printf("tmap: %s[%d] (%d) => %s\n\n",
		//       Char(Getattr(nk, "sym:name")),
		//       l, emitcheck, tml?Char(tml):0);
	      }
	      Parm *pl1 = Getattr(pl, "tmap:in:next");
	      if (pl1)
		pl = pl1;
	      else
		pl = nextSibling(pl);
	      l++;
	    }
	  }

	  if (emitcheck) {
	    if (need_v) {
	      Printf(f, "int _v = 0;\n");
	      need_v = 0;
	    }
	    if (j >= num_required) {
	      Printf(f, "if (%s > %d) {\n", argc_template_string, j);
	      num_braces++;
	    }
	    String *tmp = NewStringf(argv_template_string, j);

	    String *conv = Getattr(pj, "implicitconv");
	    if (conv && !implicitconvtypecheckoff) {
	      Replaceall(tm, "$implicitconv", conv);
	    } else {
	      Replaceall(tm, "$implicitconv", "0");
	    }
	    Replaceall(tm, "$input", tmp);
	    Printv(f, "{\n", tm, "}\n", NIL);
	    Delete(tm);
	    fn = i + 1;
	    Printf(f, "if (!_v) goto check_%d;\n", fn);
	  }
	}
	if (!Getattr(pj, "tmap:in:SWIGTYPE") && Getattr(pj, "tmap:typecheck:SWIGTYPE")) {
	  /* we emit  a warning if the argument defines the 'in' typemap, but not the 'typecheck' one */
	  Swig_warning(WARN_TYPEMAP_TYPECHECK_UNDEF, Getfile(ni), Getline(ni),
		       "Overloaded method %s with no explicit typecheck typemap for arg %d of type '%s'\n",
		       Swig_name_decl(n), j, SwigType_str(Getattr(pj, "type"), 0));
	}
	Parm *pj1 = Getattr(pj, "tmap:in:next");
	if (pj1)
	  pj = pj1;
	else
	  pj = nextSibling(pj);
	j++;
      }
    }

    /* close braces */
    for ( /* empty */ ; num_braces > 0; num_braces--)
      Printf(f, "}\n");


    String *lfmt = ReplaceFormat(fmt, num_arguments);
    Printf(f, Char(lfmt), Getattr(ni, "wrap:name"));

    Printf(f, "}\n");		/* braces closes "if" for this method */
    if (fn)
      Printf(f, "check_%d:\n\n", fn);

    if (implicitconvtypecheckoff)
      Delattr(ni, "implicitconvtypecheckoff");

    Delete(lfmt);
    Delete(coll);
  }
  Delete(dispatch);
  return f;
}

String *Swig_overload_dispatch(Node *n, const_String_or_char_ptr fmt, int *maxargs) {

  if (fast_dispatch_mode || GetFlag(n, "feature:fastdispatch")) {
    return Swig_overload_dispatch_fast(n, fmt, maxargs);
  }

  int i, j;

  *maxargs = 1;

  String *f = NewString("");

  /* Get a list of methods ranked by precedence values and argument count */
  List *dispatch = Swig_overload_rank(n, true);
  int nfunc = Len(dispatch);

  /* Loop over the functions */

  for (i = 0; i < nfunc; i++) {
    Node *ni = Getitem(dispatch, i);
    Parm *pi = Getattr(ni, "wrap:parms");
    bool implicitconvtypecheckoff = GetFlag(ni, "implicitconvtypecheckoff") != 0;
    int num_required = emit_num_required(pi);
    int num_arguments = emit_num_arguments(pi);
    if (GetFlag(n, "wrap:this")) {
      num_required++;
      num_arguments++;
    }
    if (num_arguments > *maxargs)
      *maxargs = num_arguments;

    if (num_required == num_arguments) {
      Printf(f, "if (%s == %d) {\n", argc_template_string, num_required);
    } else {
      Printf(f, "if ((%s >= %d) && (%s <= %d)) {\n", argc_template_string, num_required, argc_template_string, num_arguments);
    }

    if (num_arguments) {
      Printf(f, "int _v;\n");
    }

    int num_braces = 0;
    j = 0;
    Parm *pj = pi;
    while (pj) {
      if (checkAttribute(pj, "tmap:in:numinputs", "0")) {
	pj = Getattr(pj, "tmap:in:next");
	continue;
      }
      if (j >= num_required) {
	String *lfmt = ReplaceFormat(fmt, num_arguments);
	Printf(f, "if (%s <= %d) {\n", argc_template_string, j);
	Printf(f, Char(lfmt), Getattr(ni, "wrap:name"));
	Printf(f, "}\n");
	Delete(lfmt);
      }
      if (print_typecheck(f, (GetFlag(n, "wrap:this") ? j + 1 : j), pj, implicitconvtypecheckoff)) {
	Printf(f, "if (_v) {\n");
	num_braces++;
      }
      if (!Getattr(pj, "tmap:in:SWIGTYPE") && Getattr(pj, "tmap:typecheck:SWIGTYPE")) {
	/* we emit  a warning if the argument defines the 'in' typemap, but not the 'typecheck' one */
	Swig_warning(WARN_TYPEMAP_TYPECHECK_UNDEF, Getfile(ni), Getline(ni),
		     "Overloaded method %s with no explicit typecheck typemap for arg %d of type '%s'\n",
		     Swig_name_decl(n), j, SwigType_str(Getattr(pj, "type"), 0));
      }
      Parm *pk = Getattr(pj, "tmap:in:next");
      if (pk)
	pj = pk;
      else
	pj = nextSibling(pj);
      j++;
    }
    String *lfmt = ReplaceFormat(fmt, num_arguments);
    Printf(f, Char(lfmt), Getattr(ni, "wrap:name"));
    Delete(lfmt);
    /* close braces */
    for ( /* empty */ ; num_braces > 0; num_braces--)
      Printf(f, "}\n");
    Printf(f, "}\n");		/* braces closes "if" for this method */
    if (implicitconvtypecheckoff)
      Delattr(ni, "implicitconvtypecheckoff");
  }
  Delete(dispatch);
  return f;
}

/* -----------------------------------------------------------------------------
 * Swig_overload_check()
 * ----------------------------------------------------------------------------- */
void Swig_overload_check(Node *n) {
  Swig_overload_rank(n, false);
}
