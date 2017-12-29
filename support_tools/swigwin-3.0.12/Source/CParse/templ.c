/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * templ.c
 *
 * Expands a template into a specialized version.   
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "cparse.h"

static int template_debug = 0;


const char *baselists[3];

void SwigType_template_init() {
  baselists[0] = "baselist";
  baselists[1] = "protectedbaselist";
  baselists[2] = "privatebaselist";
}


static void add_parms(ParmList *p, List *patchlist, List *typelist) {
  while (p) {
    SwigType *ty = Getattr(p, "type");
    SwigType *val = Getattr(p, "value");
    Append(typelist, ty);
    Append(typelist, val);
    Append(patchlist, val);
    p = nextSibling(p);
  }
}

void Swig_cparse_debug_templates(int x) {
  template_debug = x;
}

/* -----------------------------------------------------------------------------
 * cparse_template_expand()
 *
 * Expands a template node into a specialized version.  This is done by
 * patching typenames and other aspects of the node according to a list of
 * template parameters
 * ----------------------------------------------------------------------------- */

static int cparse_template_expand(Node *n, String *tname, String *rname, String *templateargs, List *patchlist, List *typelist, List *cpatchlist) {
  static int expanded = 0;
  int ret;
  String *nodeType;
  if (!n)
    return 0;
  nodeType = nodeType(n);
  if (Getattr(n, "error"))
    return 0;

  if (Equal(nodeType, "template")) {
    /* Change the node type back to normal */
    if (!expanded) {
      expanded = 1;
      set_nodeType(n, Getattr(n, "templatetype"));
      ret = cparse_template_expand(n, tname, rname, templateargs, patchlist, typelist, cpatchlist);
      expanded = 0;
      return ret;
    } else {
      /* Called when template appears inside another template */
      /* Member templates */

      set_nodeType(n, Getattr(n, "templatetype"));
      ret = cparse_template_expand(n, tname, rname, templateargs, patchlist, typelist, cpatchlist);
      set_nodeType(n, "template");
      return ret;
    }
  } else if (Equal(nodeType, "cdecl")) {
    /* A simple C declaration */
    SwigType *t, *v, *d;
    String *code;
    t = Getattr(n, "type");
    v = Getattr(n, "value");
    d = Getattr(n, "decl");

    code = Getattr(n, "code");

    Append(typelist, t);
    Append(typelist, d);
    Append(patchlist, v);
    Append(cpatchlist, code);

    if (Getattr(n, "conversion_operator")) {
      Append(cpatchlist, Getattr(n, "name"));
      if (Getattr(n, "sym:name")) {
	Append(cpatchlist, Getattr(n, "sym:name"));
      }
    }
    if (checkAttribute(n, "storage", "friend")) {
      String *symname = Getattr(n, "sym:name");
      if (symname) {
	String *stripped_name = SwigType_templateprefix(symname);
	Setattr(n, "sym:name", stripped_name);
	Delete(stripped_name);
      }
      Append(typelist, Getattr(n, "name"));
    }

    add_parms(Getattr(n, "parms"), cpatchlist, typelist);
    add_parms(Getattr(n, "throws"), cpatchlist, typelist);

  } else if (Equal(nodeType, "class")) {
    /* Patch base classes */
    {
      int b = 0;
      for (b = 0; b < 3; ++b) {
	List *bases = Getattr(n, baselists[b]);
	if (bases) {
	  int i;
	  int ilen = Len(bases);
	  for (i = 0; i < ilen; i++) {
	    String *name = Copy(Getitem(bases, i));
	    Setitem(bases, i, name);
	    Append(typelist, name);
	  }
	}
      }
    }
    /* Patch children */
    {
      Node *cn = firstChild(n);
      while (cn) {
	cparse_template_expand(cn, tname, rname, templateargs, patchlist, typelist, cpatchlist);
	cn = nextSibling(cn);
      }
    }
  } else if (Equal(nodeType, "constructor")) {
    String *name = Getattr(n, "name");
    if (!(Getattr(n, "templatetype"))) {
      String *symname;
      String *stripped_name = SwigType_templateprefix(name);
      if (Strstr(tname, stripped_name)) {
	Replaceid(name, stripped_name, tname);
      }
      Delete(stripped_name);
      symname = Getattr(n, "sym:name");
      if (symname) {
	stripped_name = SwigType_templateprefix(symname);
	if (Strstr(tname, stripped_name)) {
	  Replaceid(symname, stripped_name, tname);
	}
	Delete(stripped_name);
      }
      if (strchr(Char(name), '<')) {
	Append(patchlist, Getattr(n, "name"));
      } else {
	Append(name, templateargs);
      }
      name = Getattr(n, "sym:name");
      if (name) {
	if (strchr(Char(name), '<')) {
	  Clear(name);
	  Append(name, rname);
	} else {
	  String *tmp = Copy(name);
	  Replace(tmp, tname, rname, DOH_REPLACE_ANY);
	  Clear(name);
	  Append(name, tmp);
	  Delete(tmp);
	}
      }
      /* Setattr(n,"sym:name",name); */
    }
    Append(cpatchlist, Getattr(n, "code"));
    Append(typelist, Getattr(n, "decl"));
    add_parms(Getattr(n, "parms"), cpatchlist, typelist);
    add_parms(Getattr(n, "throws"), cpatchlist, typelist);
  } else if (Equal(nodeType, "destructor")) {
    String *name = Getattr(n, "name");
    if (name) {
      if (strchr(Char(name), '<'))
        Append(patchlist, Getattr(n, "name"));
      else
        Append(name, templateargs);
    }
    name = Getattr(n, "sym:name");
    if (name) {
      if (strchr(Char(name), '<')) {
        String *sn = Copy(tname);
        Setattr(n, "sym:name", sn);
        Delete(sn);
      } else {
        Replace(name, tname, rname, DOH_REPLACE_ANY);
      }
    }
    /* Setattr(n,"sym:name",name); */
    Append(cpatchlist, Getattr(n, "code"));
  } else if (Equal(nodeType, "using")) {
    String *uname = Getattr(n, "uname");
    if (uname && strchr(Char(uname), '<')) {
      Append(patchlist, uname);
    }
    if (Getattr(n, "namespace")) {
      /* Namespace link.   This is nasty.  Is other namespace defined? */

    }
  } else {
    /* Look for obvious parameters */
    Node *cn;
    Append(cpatchlist, Getattr(n, "code"));
    Append(typelist, Getattr(n, "type"));
    Append(typelist, Getattr(n, "decl"));
    add_parms(Getattr(n, "parms"), cpatchlist, typelist);
    add_parms(Getattr(n, "kwargs"), cpatchlist, typelist);
    add_parms(Getattr(n, "pattern"), cpatchlist, typelist);
    add_parms(Getattr(n, "throws"), cpatchlist, typelist);
    cn = firstChild(n);
    while (cn) {
      cparse_template_expand(cn, tname, rname, templateargs, patchlist, typelist, cpatchlist);
      cn = nextSibling(cn);
    }
  }
  return 0;
}

static
String *partial_arg(String *s, String *p) {
  char *c;
  char *cp = Char(p);
  String *prefix;
  String *newarg;

  /* Find the prefix on the partial argument */

  c = strchr(cp, '$');
  if (!c) {
    return Copy(s);
  }
  prefix = NewStringWithSize(cp, (int)(c - cp));
  newarg = Copy(s);
  Replace(newarg, prefix, "", DOH_REPLACE_ANY | DOH_REPLACE_FIRST);
  Delete(prefix);
  return newarg;
}

/* -----------------------------------------------------------------------------
 * Swig_cparse_template_expand()
 * ----------------------------------------------------------------------------- */

int Swig_cparse_template_expand(Node *n, String *rname, ParmList *tparms, Symtab *tscope) {
  List *patchlist, *cpatchlist, *typelist;
  String *templateargs;
  String *tname;
  String *iname;
  String *tbase;
  patchlist = NewList();
  cpatchlist = NewList();
  typelist = NewList();

  {
    String *tmp = NewStringEmpty();
    if (tparms) {
      SwigType_add_template(tmp, tparms);
    }
    templateargs = Copy(tmp);
    Delete(tmp);
  }

  tname = Copy(Getattr(n, "name"));
  tbase = Swig_scopename_last(tname);

  /* Look for partial specialization matching */
  if (Getattr(n, "partialargs")) {
    Parm *p, *tp;
    ParmList *ptargs = SwigType_function_parms(Getattr(n, "partialargs"), n);
    p = ptargs;
    tp = tparms;
    while (p && tp) {
      SwigType *ptype;
      SwigType *tptype;
      SwigType *partial_type;
      ptype = Getattr(p, "type");
      tptype = Getattr(tp, "type");
      if (ptype && tptype) {
	partial_type = partial_arg(tptype, ptype);
	/*      Printf(stdout,"partial '%s' '%s'  ---> '%s'\n", tptype, ptype, partial_type); */
	Setattr(tp, "type", partial_type);
	Delete(partial_type);
      }
      p = nextSibling(p);
      tp = nextSibling(tp);
    }
    assert(ParmList_len(ptargs) == ParmList_len(tparms));
    Delete(ptargs);
  }

  /*
    Parm *p = tparms;
    while (p) {
      Printf(stdout, "tparm: '%s' '%s' '%s'\n", Getattr(p, "name"), Getattr(p, "type"), Getattr(p, "value"));
      p = nextSibling(p);
    }
  */

  /*  Printf(stdout,"targs = '%s'\n", templateargs);
     Printf(stdout,"rname = '%s'\n", rname);
     Printf(stdout,"tname = '%s'\n", tname);  */
  cparse_template_expand(n, tname, rname, templateargs, patchlist, typelist, cpatchlist);

  /* Set the name */
  {
    String *name = Getattr(n, "name");
    if (name) {
      Append(name, templateargs);
    }
    iname = name;
  }

  /* Patch all of the types */
  {
    Parm *tp = Getattr(n, "templateparms");
    Parm *p = tparms;
    /*    Printf(stdout,"%s\n", ParmList_str_defaultargs(tp)); */

    if (tp) {
      Symtab *tsdecl = Getattr(n, "sym:symtab");
      while (p && tp) {
	String *name, *value, *valuestr, *tmp, *tmpr;
	int sz, i;
	String *dvalue = 0;
	String *qvalue = 0;

	name = Getattr(tp, "name");
	value = Getattr(p, "value");

	if (name) {
	  if (!value)
	    value = Getattr(p, "type");
	  qvalue = Swig_symbol_typedef_reduce(value, tsdecl);
	  dvalue = Swig_symbol_type_qualify(qvalue, tsdecl);
	  if (SwigType_istemplate(dvalue)) {
	    String *ty = Swig_symbol_template_deftype(dvalue, tscope);
	    Delete(dvalue);
	    dvalue = ty;
	  }

	  assert(dvalue);
	  valuestr = SwigType_str(dvalue, 0);
	  /* Need to patch default arguments */
	  {
	    Parm *rp = nextSibling(p);
	    while (rp) {
	      String *rvalue = Getattr(rp, "value");
	      if (rvalue) {
		Replace(rvalue, name, dvalue, DOH_REPLACE_ID);
	      }
	      rp = nextSibling(rp);
	    }
	  }
	  sz = Len(patchlist);
	  for (i = 0; i < sz; i++) {
	    String *s = Getitem(patchlist, i);
	    Replace(s, name, dvalue, DOH_REPLACE_ID);
	  }
	  sz = Len(typelist);
	  for (i = 0; i < sz; i++) {
	    String *s = Getitem(typelist, i);
	    /*      Replace(s,name,value, DOH_REPLACE_ID); */
	    /*      Printf(stdout,"name = '%s', value = '%s', tbase = '%s', iname='%s' s = '%s' --> ", name, dvalue, tbase, iname, s); */
	    SwigType_typename_replace(s, name, dvalue);
	    SwigType_typename_replace(s, tbase, iname);
	    /*      Printf(stdout,"'%s'\n", s); */
	  }

	  tmp = NewStringf("#%s", name);
	  tmpr = NewStringf("\"%s\"", valuestr);

	  sz = Len(cpatchlist);
	  for (i = 0; i < sz; i++) {
	    String *s = Getitem(cpatchlist, i);
	    Replace(s, tmp, tmpr, DOH_REPLACE_ID);
	    Replace(s, name, valuestr, DOH_REPLACE_ID);
	  }
	  Delete(tmp);
	  Delete(tmpr);
	  Delete(valuestr);
	  Delete(dvalue);
	  Delete(qvalue);
	}
	p = nextSibling(p);
	tp = nextSibling(tp);
	if (!p)
	  p = tp;
      }
    } else {
      /* No template parameters at all.  This could be a specialization */
      int i, sz;
      sz = Len(typelist);
      for (i = 0; i < sz; i++) {
	String *s = Getitem(typelist, i);
	SwigType_typename_replace(s, tbase, iname);
      }
    }
  }

  /* Patch bases */
  {
    List *bases = Getattr(n, "baselist");
    if (bases) {
      Iterator b;
      for (b = First(bases); b.item; b = Next(b)) {
	String *qn = Swig_symbol_type_qualify(b.item, tscope);
	Clear(b.item);
	Append(b.item, qn);
	Delete(qn);
      }
    }
  }
  Delete(patchlist);
  Delete(cpatchlist);
  Delete(typelist);
  Delete(tbase);
  Delete(tname);
  Delete(templateargs);

  /*  set_nodeType(n,"template"); */
  return 0;
}

typedef enum { ExactNoMatch = -2, PartiallySpecializedNoMatch = -1, PartiallySpecializedMatch = 1, ExactMatch = 2 } EMatch;

/* -----------------------------------------------------------------------------
 * does_parm_match()
 *
 * Template argument deduction - check if a template type matches a partially specialized 
 * template parameter type. Typedef reduce 'partial_parm_type' to see if it matches 'type'.
 *
 * type - template parameter type to match against
 * partial_parm_type - partially specialized template type - a possible match
 * partial_parm_type_base - base type of partial_parm_type
 * tscope - template scope
 * specialization_priority - (output) contains a value indicating how good the match is 
 *   (higher is better) only set if return is set to PartiallySpecializedMatch or ExactMatch.
 * ----------------------------------------------------------------------------- */

static EMatch does_parm_match(SwigType *type, SwigType *partial_parm_type, const char *partial_parm_type_base, Symtab *tscope, int *specialization_priority) {
  static const int EXACT_MATCH_PRIORITY = 99999; /* a number bigger than the length of any conceivable type */
  int matches;
  int substitutions;
  EMatch match;
  SwigType *ty = Swig_symbol_typedef_reduce(type, tscope);
  String *base = SwigType_base(ty);
  SwigType *t = Copy(partial_parm_type);
  substitutions = Replaceid(t, partial_parm_type_base, base); /* eg: Replaceid("p.$1", "$1", "int") returns t="p.int" */
  matches = Equal(ty, t);
  *specialization_priority = -1;
  if (substitutions == 1) {
    /* we have a non-explicit specialized parameter (in partial_parm_type) because a substitution for $1, $2... etc has taken place */
    SwigType *tt = Copy(partial_parm_type);
    int len;
    /*
       check for match to partial specialization type, for example, all of the following could match the type in the %template:
       template <typename T> struct XX {};
       template <typename T> struct XX<T &> {};         // r.$1
       template <typename T> struct XX<T const&> {};    // r.q(const).$1
       template <typename T> struct XX<T *const&> {};   // r.q(const).p.$1
       %template(XXX) XX<int *const&>;                  // r.q(const).p.int

       where type="r.q(const).p.int" will match either of tt="r.", tt="r.q(const)" tt="r.q(const).p"
    */
    Replaceid(tt, partial_parm_type_base, ""); /* remove the $1, $2 etc, eg tt="p.$1" => "p." */
    len = Len(tt);
    if (Strncmp(tt, ty, len) == 0) {
      match = PartiallySpecializedMatch;
      *specialization_priority = len;
    } else {
      match = PartiallySpecializedNoMatch;
    }
    Delete(tt);
  } else {
    match = matches ? ExactMatch : ExactNoMatch;
    if (matches)
      *specialization_priority = EXACT_MATCH_PRIORITY; /* exact matches always take precedence */
  }
  /*
  Printf(stdout, "      does_parm_match %2d %5d [%s] [%s]\n", match, *specialization_priority, type, partial_parm_type);
  */
  Delete(t);
  Delete(base);
  Delete(ty);
  return match;
}

/* -----------------------------------------------------------------------------
 * template_locate()
 *
 * Search for a template that matches name with given parameters.
 * ----------------------------------------------------------------------------- */

static Node *template_locate(String *name, Parm *tparms, Symtab *tscope) {
  Node *n = 0;
  String *tname = 0;
  Node *templ;
  Symtab *primary_scope = 0;
  List *possiblepartials = 0;
  Parm *p;
  Parm *parms = 0;
  Parm *targs;
  ParmList *expandedparms;
  int *priorities_matrix = 0;
  int max_possible_partials = 0;
  int posslen = 0;

  /* Search for primary (unspecialized) template */
  templ = Swig_symbol_clookup(name, 0);

  if (template_debug) {
    tname = Copy(name);
    SwigType_add_template(tname, tparms);
    Printf(stdout, "\n");
    Swig_diagnostic(cparse_file, cparse_line, "template_debug: Searching for match to: '%s'\n", tname);
    Delete(tname);
    tname = 0;
  }

  if (templ) {
    tname = Copy(name);
    parms = CopyParmList(tparms);

    /* All template specializations must be in the primary template's scope, store the symbol table for this scope for specialization lookups */
    primary_scope = Getattr(templ, "sym:symtab");

    /* Add default values from primary template */
    targs = Getattr(templ, "templateparms");
    expandedparms = Swig_symbol_template_defargs(parms, targs, tscope, primary_scope);

    /* reduce the typedef */
    p = expandedparms;
    while (p) {
      SwigType *ty = Getattr(p, "type");
      if (ty) {
	SwigType *nt = Swig_symbol_type_qualify(ty, tscope);
	Setattr(p, "type", nt);
	Delete(nt);
      }
      p = nextSibling(p);
    }
    SwigType_add_template(tname, expandedparms);

    /* Search for an explicit (exact) specialization. Example: template<> class name<int> { ... } */
    {
      if (template_debug) {
	Printf(stdout, "    searching for : '%s' (explicit specialization)\n", tname);
      }
      n = Swig_symbol_clookup_local(tname, primary_scope);
      if (!n) {
	SwigType *rname = Swig_symbol_typedef_reduce(tname, tscope);
	if (!Equal(rname, tname)) {
	  if (template_debug) {
	    Printf(stdout, "    searching for : '%s' (explicit specialization with typedef reduction)\n", rname);
	  }
	  n = Swig_symbol_clookup_local(rname, primary_scope);
	}
	Delete(rname);
      }
      if (n) {
	Node *tn;
	String *nodeType = nodeType(n);
	if (Equal(nodeType, "template")) {
	  if (template_debug) {
	    Printf(stdout, "    explicit specialization found: '%s'\n", Getattr(n, "name"));
	  }
	  goto success;
	}
	tn = Getattr(n, "template");
	if (tn) {
	  if (template_debug) {
	    Printf(stdout, "    previous instantiation found: '%s'\n", Getattr(n, "name"));
	  }
	  n = tn;
	  goto success;	  /* Previously wrapped by a template instantiation */
	}
	Swig_error(cparse_file, cparse_line, "'%s' is not defined as a template. (%s)\n", name, nodeType(n));
	Delete(tname);
	Delete(parms);
	return 0;	  /* Found a match, but it's not a template of any kind. */
      }
    }

    /* Search for partial specializations.
     * Example: template<typename T> class name<T *> { ... } 

     * There are 3 types of template arguments:
     * (1) Template type arguments
     * (2) Template non type arguments
     * (3) Template template arguments
     * only (1) is really supported for partial specializations
     */

    /* Rank each template parameter against the desired template parameters then build a matrix of best matches */
    possiblepartials = NewList();
    {
      char tmp[32];
      List *partials;

      partials = Getattr(templ, "partials"); /* note that these partial specializations do not include explicit specializations */
      if (partials) {
	Iterator pi;
	int parms_len = ParmList_len(parms);
	int *priorities_row;
	max_possible_partials = Len(partials);
	priorities_matrix = (int *)malloc(sizeof(int) * max_possible_partials * parms_len); /* slightly wasteful allocation for max possible matches */
	priorities_row = priorities_matrix;
	for (pi = First(partials); pi.item; pi = Next(pi)) {
	  Parm *p = parms;
	  int all_parameters_match = 1;
	  int i = 1;
	  Parm *partialparms = Getattr(pi.item, "partialparms");
	  Parm *pp = partialparms;
	  String *templcsymname = Getattr(pi.item, "templcsymname");
	  if (template_debug) {
	    Printf(stdout, "    checking match: '%s' (partial specialization)\n", templcsymname);
	  }
	  if (ParmList_len(partialparms) == parms_len) {
	    while (p && pp) {
	      SwigType *t;
	      sprintf(tmp, "$%d", i);
	      t = Getattr(p, "type");
	      if (!t)
		t = Getattr(p, "value");
	      if (t) {
		EMatch match = does_parm_match(t, Getattr(pp, "type"), tmp, tscope, priorities_row + i - 1);
		if (match < (int)PartiallySpecializedMatch) {
		  all_parameters_match = 0;
		  break;
		}
	      }
	      i++;
	      p = nextSibling(p);
	      pp = nextSibling(pp);
	    }
	    if (all_parameters_match) {
	      Append(possiblepartials, pi.item);
	      priorities_row += parms_len;
	    }
	  }
	}
      }
    }

    posslen = Len(possiblepartials);
    if (template_debug) {
      int i;
      if (posslen == 0)
	Printf(stdout, "    matched partials: NONE\n");
      else if (posslen == 1)
	Printf(stdout, "    chosen partial: '%s'\n", Getattr(Getitem(possiblepartials, 0), "templcsymname"));
      else {
	Printf(stdout, "    possibly matched partials:\n");
	for (i = 0; i < posslen; i++) {
	  Printf(stdout, "      '%s'\n", Getattr(Getitem(possiblepartials, i), "templcsymname"));
	}
      }
    }

    if (posslen > 1) {
      /* Now go through all the possibly matched partial specialization templates and look for a non-ambiguous match.
       * Exact matches rank the highest and deduced parameters are ranked by how specialized they are, eg looking for
       * a match to const int *, the following rank (highest to lowest):
       *   const int * (exact match)
       *   const T *
       *   T *
       *   T
       *
       *   An ambiguous example when attempting to match as either specialization could match: %template() X<int *, double *>;
       *   template<typename T1, typename T2> X class {};  // primary template
       *   template<typename T1> X<T1, double *> class {}; // specialization (1)
       *   template<typename T2> X<int *, T2> class {};    // specialization (2)
       */
      if (template_debug) {
	int row, col;
	int parms_len = ParmList_len(parms);
	Printf(stdout, "      parameter priorities matrix (%d parms):\n", parms_len);
	for (row = 0; row < posslen; row++) {
	  int *priorities_row = priorities_matrix + row*parms_len;
	  Printf(stdout, "        ");
	  for (col = 0; col < parms_len; col++) {
	    Printf(stdout, "%5d ", priorities_row[col]);
	  }
	  Printf(stdout, "\n");
	}
      }
      {
	int row, col;
	int parms_len = ParmList_len(parms);
	/* Printf(stdout, "      parameter priorities inverse matrix (%d parms):\n", parms_len); */
	for (col = 0; col < parms_len; col++) {
	  int *priorities_col = priorities_matrix + col;
	  int maxpriority = -1;
	  /* 
	     Printf(stdout, "max_possible_partials: %d col:%d\n", max_possible_partials, col);
	     Printf(stdout, "        ");
	     */
	  /* determine the highest rank for this nth parameter */
	  for (row = 0; row < posslen; row++) {
	    int *element_ptr = priorities_col + row*parms_len;
	    int priority = *element_ptr;
	    if (priority > maxpriority)
	      maxpriority = priority;
	    /* Printf(stdout, "%5d ", priority); */
	  }
	  /* Printf(stdout, "\n"); */
	  /* flag all the parameters which equal the highest rank */
	  for (row = 0; row < posslen; row++) {
	    int *element_ptr = priorities_col + row*parms_len;
	    int priority = *element_ptr;
	    *element_ptr = (priority >= maxpriority) ? 1 : 0;
	  }
	}
      }
      {
	int row, col;
	int parms_len = ParmList_len(parms);
	Iterator pi = First(possiblepartials);
	Node *chosenpartials = NewList();
	if (template_debug)
	  Printf(stdout, "      priority flags matrix:\n");
	for (row = 0; row < posslen; row++) {
	  int *priorities_row = priorities_matrix + row*parms_len;
	  int highest_count = 0; /* count of highest priority parameters */
	  for (col = 0; col < parms_len; col++) {
	    highest_count += priorities_row[col];
	  }
	  if (template_debug) {
	    Printf(stdout, "        ");
	    for (col = 0; col < parms_len; col++) {
	      Printf(stdout, "%5d ", priorities_row[col]);
	    }
	    Printf(stdout, "\n");
	  }
	  if (highest_count == parms_len) {
	    Append(chosenpartials, pi.item);
	  }
	  pi = Next(pi);
	}
	if (Len(chosenpartials) > 0) {
	  /* one or more best match found */
	  Delete(possiblepartials);
	  possiblepartials = chosenpartials;
	  posslen = Len(possiblepartials);
	} else {
	  /* no best match found */
	  Delete(chosenpartials);
	}
      }
    }

    if (posslen > 0) {
      String *s = Getattr(Getitem(possiblepartials, 0), "templcsymname");
      n = Swig_symbol_clookup_local(s, primary_scope);
      if (posslen > 1) {
	int i;
	if (n) {
	  Swig_warning(WARN_PARSE_TEMPLATE_AMBIG, cparse_file, cparse_line, "Instantiation of template '%s' is ambiguous,\n", SwigType_namestr(tname));
	  Swig_warning(WARN_PARSE_TEMPLATE_AMBIG, Getfile(n), Getline(n), "  instantiation '%s' used,\n", SwigType_namestr(Getattr(n, "name")));
	}
	for (i = 1; i < posslen; i++) {
	  String *templcsymname = Getattr(Getitem(possiblepartials, i), "templcsymname");
	  Node *ignored_node = Swig_symbol_clookup_local(templcsymname, primary_scope);
	  assert(ignored_node);
	  Swig_warning(WARN_PARSE_TEMPLATE_AMBIG, Getfile(ignored_node), Getline(ignored_node), "  instantiation '%s' ignored.\n", SwigType_namestr(Getattr(ignored_node, "name")));
	}
      }
    }

    if (!n) {
      if (template_debug) {
	Printf(stdout, "    chosen primary template: '%s'\n", Getattr(templ, "name"));
      }
      n = templ;
    }
  } else {
    if (template_debug) {
      Printf(stdout, "    primary template not found\n");
    }
    /* Give up if primary (unspecialized) template not found as specializations will only exist if there is a primary template */
    n = 0;
  }

  if (!n) {
    Swig_error(cparse_file, cparse_line, "Template '%s' undefined.\n", name);
  } else if (n) {
    String *nodeType = nodeType(n);
    if (!Equal(nodeType, "template")) {
      Swig_error(cparse_file, cparse_line, "'%s' is not defined as a template. (%s)\n", name, nodeType);
      n = 0;
    }
  }
success:
  Delete(tname);
  Delete(possiblepartials);
  if ((template_debug) && (n)) {
    /*
    Printf(stdout, "Node: %p\n", n);
    Swig_print_node(n);
    */
    Printf(stdout, "    chosen template:'%s'\n", Getattr(n, "name"));
  }
  Delete(parms);
  free(priorities_matrix);
  return n;
}


/* -----------------------------------------------------------------------------
 * Swig_cparse_template_locate()
 *
 * Search for a template that matches name with given parameters.
 * For templated classes finds the specialized template should there be one.
 * For templated functions finds the unspecialized template even if a specialized
 * template exists.
 * ----------------------------------------------------------------------------- */

Node *Swig_cparse_template_locate(String *name, Parm *tparms, Symtab *tscope) {
  Node *n = template_locate(name, tparms, tscope);	/* this function does what we want for templated classes */

  if (n) {
    String *nodeType = nodeType(n);
    int isclass = 0;
    assert(Equal(nodeType, "template"));
    isclass = (Equal(Getattr(n, "templatetype"), "class"));
    if (!isclass) {
      /* If not a templated class we must have a templated function.
         The template found is not necessarily the one we want when dealing with templated
         functions. We don't want any specialized templated functions as they won't have
         the default parameters. Let's look for the unspecialized template. Also make sure
         the number of template parameters is correct as it is possible to overload a
         templated function with different numbers of template parameters. */

      if (template_debug) {
	Printf(stdout, "    Not a templated class, seeking most appropriate templated function\n");
      }

      n = Swig_symbol_clookup_local(name, 0);
      while (n) {
	Parm *tparmsfound = Getattr(n, "templateparms");
	if (ParmList_len(tparms) == ParmList_len(tparmsfound)) {
	  /* successful match */
	  break;
	}
	/* repeat until we find a match with correct number of templated parameters */
	n = Getattr(n, "sym:nextSibling");
      }

      if (!n) {
	Swig_error(cparse_file, cparse_line, "Template '%s' undefined.\n", name);
      }

      if ((template_debug) && (n)) {
	Printf(stdout, "Templated function found: %p\n", n);
	Swig_print_node(n);
      }
    }
  }

  return n;
}
