/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * allocate.cxx
 *
 * This module tries to figure out which classes and structures support
 * default constructors and destructors in C++.   There are several rules that
 * define this behavior including pure abstract methods, private sections,
 * and non-default constructors in base classes.  See the ARM or
 * Doc/Manual/SWIGPlus.html for details.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"

static int virtual_elimination_mode = 0;	/* set to 0 on default */

/* Set virtual_elimination_mode */
void Wrapper_virtual_elimination_mode_set(int flag) {
  virtual_elimination_mode = flag;
}

/* Helper function to assist with abstract class checking.  
   This is a major hack. Sorry.  */

extern "C" {
  static String *search_decl = 0;	/* Declarator being searched */
  static int check_implemented(Node *n) {
    String *decl;
    if (!n)
       return 0;
    while (n) {
      if (Strcmp(nodeType(n), "cdecl") == 0) {
	decl = Getattr(n, "decl");
	if (SwigType_isfunction(decl)) {
	  SwigType *decl1 = SwigType_typedef_resolve_all(decl);
	  SwigType *decl2 = SwigType_pop_function(decl1);
	  if (Strcmp(decl2, search_decl) == 0) {
	    if (!GetFlag(n, "abstract")) {
	      Delete(decl1);
	      Delete(decl2);
	      return 1;
	    }
	  }
	  Delete(decl1);
	  Delete(decl2);
	}
      }
      n = Getattr(n, "csym:nextSibling");
    }
    return 0;
  }
}

class Allocate:public Dispatcher {
  Node *inclass;
  int extendmode;

  /* Checks if a function, n, is the same as any in the base class, ie if the method is polymorphic.
   * Also checks for methods which will be hidden (ie a base has an identical non-virtual method).
   * Both methods must have public access for a match to occur. */
  int function_is_defined_in_bases(Node *n, Node *bases) {

    if (!bases)
      return 0;

    String *this_decl = Getattr(n, "decl");
    if (!this_decl)
       return 0;

    String *name = Getattr(n, "name");
    String *this_type = Getattr(n, "type");
    String *resolved_decl = SwigType_typedef_resolve_all(this_decl);

    // Search all base classes for methods with same signature
    for (int i = 0; i < Len(bases); i++) {
      Node *b = Getitem(bases, i);
      Node *base = firstChild(b);
      while (base) {
	if (Strcmp(nodeType(base), "extend") == 0) {
	  // Loop through all the %extend methods
	  Node *extend = firstChild(base);
	  while (extend) {
	    if (function_is_defined_in_bases_seek(n, b, extend, this_decl, name, this_type, resolved_decl)) {
	      Delete(resolved_decl);
	      return 1;
	    }
	    extend = nextSibling(extend);
	  }
	} else if (Strcmp(nodeType(base), "using") == 0) {
	  // Loop through all the using declaration methods
	  Node *usingdecl = firstChild(base);
	  while (usingdecl) {
	    if (function_is_defined_in_bases_seek(n, b, usingdecl, this_decl, name, this_type, resolved_decl)) {
	      Delete(resolved_decl);
	      return 1;
	    }
	    usingdecl = nextSibling(usingdecl);
	  }
	} else {
	  // normal methods
	  if (function_is_defined_in_bases_seek(n, b, base, this_decl, name, this_type, resolved_decl)) {
	    Delete(resolved_decl);
	    return 1;
	  }
	}
	base = nextSibling(base);
      }
    }
    Delete(resolved_decl);
    resolved_decl = 0;
    for (int j = 0; j < Len(bases); j++) {
      Node *b = Getitem(bases, j);
      if (function_is_defined_in_bases(n, Getattr(b, "allbases")))
	return 1;
    }
    return 0;
  }

  /* Helper function for function_is_defined_in_bases */
  int function_is_defined_in_bases_seek(Node *n, Node *b, Node *base, String *this_decl, String *name, String *this_type, String *resolved_decl) {

    String *base_decl = Getattr(base, "decl");
    SwigType *base_type = Getattr(base, "type");
    if (base_decl && base_type) {
      if (checkAttribute(base, "name", name) && !GetFlag(b, "feature:ignore") /* whole class is ignored */ ) {
	if (SwigType_isfunction(resolved_decl) && SwigType_isfunction(base_decl)) {
	  // We have found a method that has the same name as one in a base class
	  bool covariant_returntype = false;
	  bool returntype_match = Strcmp(base_type, this_type) == 0 ? true : false;
	  bool decl_match = Strcmp(base_decl, this_decl) == 0 ? true : false;
	  if (returntype_match && decl_match) {
	    // Exact match - we have found a method with identical signature
	    // No typedef resolution was done, but skipping it speeds things up slightly
	  } else {
	    // Either we have:
	    //  1) matching methods but are one of them uses a different typedef (return type or parameter) to the one in base class' method
	    //  2) matching polymorphic methods with covariant return type
	    //  3) a non-matching method (ie an overloaded method of some sort)
	    //  4) a matching method which is not polymorphic, ie it hides the base class' method

	    // Check if fully resolved return types match (including
	    // covariant return types)
	    if (!returntype_match) {
	      String *this_returntype = function_return_type(n);
	      String *base_returntype = function_return_type(base);
	      returntype_match = Strcmp(this_returntype, base_returntype) == 0 ? true : false;
	      if (!returntype_match) {
		covariant_returntype = SwigType_issubtype(this_returntype, base_returntype) ? true : false;
		returntype_match = covariant_returntype;
	      }
	      Delete(this_returntype);
	      Delete(base_returntype);
	    }
	    // The return types must match at this point, for the whole method to match
	    if (returntype_match && !decl_match) {
	      // Now need to check the parameter list
	      // First do an inexpensive parameter count
	      ParmList *this_parms = Getattr(n, "parms");
	      ParmList *base_parms = Getattr(base, "parms");
	      if (ParmList_len(this_parms) == ParmList_len(base_parms)) {
		// Number of parameters are the same, now check that all the parameters match
		SwigType *base_fn = NewString("");
		SwigType *this_fn = NewString("");
		SwigType_add_function(base_fn, base_parms);
		SwigType_add_function(this_fn, this_parms);
		base_fn = SwigType_typedef_resolve_all(base_fn);
		this_fn = SwigType_typedef_resolve_all(this_fn);
		if (Strcmp(base_fn, this_fn) == 0) {
		  // Finally check that the qualifiers match
		  int base_qualifier = SwigType_isqualifier(resolved_decl);
		  int this_qualifier = SwigType_isqualifier(base_decl);
		  if (base_qualifier == this_qualifier) {
		    decl_match = true;
		  }
		}
		Delete(base_fn);
		Delete(this_fn);
	      }
	    }
	  }
	  //Printf(stderr,"look %s %s %d %d\n",base_decl, this_decl, returntype_match, decl_match);

	  if (decl_match && returntype_match) {
	    // Found an identical method in the base class
	    bool this_wrapping_protected_members = is_member_director(n) ? true : false;	// This should really check for dirprot rather than just being a director method
	    bool base_wrapping_protected_members = is_member_director(base) ? true : false;	// This should really check for dirprot rather than just being a director method
	    bool both_have_public_access = is_public(n) && is_public(base);
	    bool both_have_protected_access = (is_protected(n) && this_wrapping_protected_members) && (is_protected(base) && base_wrapping_protected_members);
	    bool both_have_private_access = is_private(n) && is_private(base);
	    if (checkAttribute(base, "storage", "virtual")) {
	      // Found a polymorphic method.
	      // Mark the polymorphic method, in case the virtual keyword was not used.
	      Setattr(n, "storage", "virtual");
	      if (!Getattr(b, "feature:interface")) { // interface implementation neither hides nor overrides
		if (both_have_public_access || both_have_protected_access) {
		  if (!is_non_public_base(inclass, b))
		    Setattr(n, "override", base);	// Note C# definition of override, ie access must be the same
		}
		else if (!both_have_private_access) {
		  // Different access
		  if (this_wrapping_protected_members || base_wrapping_protected_members)
		    if (!is_non_public_base(inclass, b))
		      Setattr(n, "hides", base);	// Note C# definition of hiding, ie hidden if access is different
		}
	      }
	      // Try and find the most base's covariant return type
	      SwigType *most_base_covariant_type = Getattr(base, "covariant");
	      if (!most_base_covariant_type && covariant_returntype)
		most_base_covariant_type = function_return_type(base, false);

	      if (!most_base_covariant_type) {
		// Eliminate the derived virtual method.
		if (virtual_elimination_mode && !is_member_director(n))
		  if (both_have_public_access)
		    if (!is_non_public_base(inclass, b))
		      if (!Swig_symbol_isoverloaded(n)) {
			// Don't eliminate if an overloaded method as this hides the method
			// in the scripting languages: the dispatch function will hide the base method if ignored.
			SetFlag(n, "feature:ignore");
		      }
	      } else {
		// Some languages need to know about covariant return types
		Setattr(n, "covariant", most_base_covariant_type);
	      }

	    } else {
	      // Found an identical method in the base class, but it is not polymorphic.
	      if (both_have_public_access || both_have_protected_access)
		if (!is_non_public_base(inclass, b))
		  Setattr(n, "hides", base);
	    }
	    if (both_have_public_access || both_have_protected_access)
	      return 1;
	  }
	}
      }
    }
    return 0;
  }

  /* Determines whether the base class, b, is in the list of private
   * or protected base classes for class n. */
  bool is_non_public_base(Node *n, Node *b) {
    bool non_public_base = false;
    Node *bases = Getattr(n, "privatebases");
    if (bases) {
      for (int i = 0; i < Len(bases); i++) {
	Node *base = Getitem(bases, i);
	if (base == b)
	  non_public_base = true;
      }
    }
    bases = Getattr(n, "protectedbases");
    if (bases) {
      for (int i = 0; i < Len(bases); i++) {
	Node *base = Getitem(bases, i);
	if (base == b)
	  non_public_base = true;
      }
    }
    return non_public_base;
  }

  /* Returns the return type for a function. The node n should be a function.
     If resolve is true the fully returned type is fully resolved.
     Caller is responsible for deleting returned string. */
  String *function_return_type(Node *n, bool resolve = true) {
    String *decl = Getattr(n, "decl");
    SwigType *type = Getattr(n, "type");
    String *ty = NewString(type);
    SwigType_push(ty, decl);
    if (SwigType_isqualifier(ty))
      Delete(SwigType_pop(ty));
    Delete(SwigType_pop_function(ty));
    if (resolve) {
      String *unresolved = ty;
      ty = SwigType_typedef_resolve_all(unresolved);
      Delete(unresolved);
    }
    return ty;
  }

  /* Checks if a class member is the same as inherited from the class bases */
  int class_member_is_defined_in_bases(Node *member, Node *classnode) {
    Node *bases;		/* bases is the closest ancestors of classnode */
    int defined = 0;

    bases = Getattr(classnode, "allbases");
    if (!bases)
      return 0;

    {
      int old_mode = virtual_elimination_mode;
      if (is_member_director(classnode, member))
	virtual_elimination_mode = 0;

      if (function_is_defined_in_bases(member, bases)) {
	defined = 1;
      }

      virtual_elimination_mode = old_mode;
    }

    if (defined)
      return 1;
    else
      return 0;
  }

  /* Checks to see if a class is abstract through inheritance,
     and saves the first node that seems to be abstract.
   */
  int is_abstract_inherit(Node *n, Node *base = 0, int first = 0) {
    if (!first && (base == n))
      return 0;
    if (!base) {
      /* Root node */
      Symtab *stab = Getattr(n, "symtab");	/* Get symbol table for node */
      Symtab *oldtab = Swig_symbol_setscope(stab);
      int ret = is_abstract_inherit(n, n, 1);
      Swig_symbol_setscope(oldtab);
      return ret;
    }
    List *abstracts = Getattr(base, "abstracts");
    if (abstracts) {
      int dabstract = 0;
      int len = Len(abstracts);
      for (int i = 0; i < len; i++) {
	Node *nn = Getitem(abstracts, i);
	String *name = Getattr(nn, "name");
	if (!name)
	  continue;
	if (Strchr(name, '~'))
	  continue;		/* Don't care about destructors */
	String *base_decl = Getattr(nn, "decl");
	if (base_decl)
	  base_decl = SwigType_typedef_resolve_all(base_decl);
	if (SwigType_isfunction(base_decl))
	  search_decl = SwigType_pop_function(base_decl);
	Node *dn = Swig_symbol_clookup_local_check(name, 0, check_implemented);
	Delete(search_decl);
	Delete(base_decl);

	if (!dn) {
	  List *nabstracts = Getattr(n, "abstracts");
	  if (!nabstracts) {
	    nabstracts = NewList();
	    Setattr(n, "abstracts", nabstracts);
	    Delete(nabstracts);
	  }
	  Append(nabstracts, nn);
	  if (!Getattr(n, "abstracts:firstnode")) {
	    Setattr(n, "abstracts:firstnode", nn);
	  }
	  dabstract = base != n;
	}
      }
      if (dabstract)
	return 1;
    }
    List *bases = Getattr(base, "allbases");
    if (!bases)
      return 0;
    for (int i = 0; i < Len(bases); i++) {
      if (is_abstract_inherit(n, Getitem(bases, i))) {
	return 1;
      }
    }
    return 0;
  }


  /* Grab methods used by smart pointers */

  List *smart_pointer_methods(Node *cls, List *methods, int isconst, String *classname = 0) {
    if (!methods) {
      methods = NewList();
    }

    Node *c = firstChild(cls);

    while (c) {
      if (Getattr(c, "error") || GetFlag(c, "feature:ignore")) {
	c = nextSibling(c);
	continue;
      }
      if (!isconst && (Strcmp(nodeType(c), "extend") == 0)) {
	methods = smart_pointer_methods(c, methods, isconst, Getattr(cls, "name"));
      } else if (Strcmp(nodeType(c), "cdecl") == 0) {
	if (!GetFlag(c, "feature:ignore")) {
	  String *storage = Getattr(c, "storage");
	  if (!((Cmp(storage, "typedef") == 0))
	      && !((Cmp(storage, "friend") == 0))) {
	    String *name = Getattr(c, "name");
	    String *symname = Getattr(c, "sym:name");
	    Node *e = Swig_symbol_clookup_local(name, 0);
	    if (e && is_public(e) && !GetFlag(e, "feature:ignore") && (Cmp(symname, Getattr(e, "sym:name")) == 0)) {
	      Swig_warning(WARN_LANG_DEREF_SHADOW, Getfile(e), Getline(e), "Declaration of '%s' shadows declaration accessible via operator->(),\n", name);
	      Swig_warning(WARN_LANG_DEREF_SHADOW, Getfile(c), Getline(c), "previous declaration of '%s'.\n", name);
	    } else {
	      /* Make sure node with same name doesn't already exist */
	      int k;
	      int match = 0;
	      for (k = 0; k < Len(methods); k++) {
		e = Getitem(methods, k);
		if (Cmp(symname, Getattr(e, "sym:name")) == 0) {
		  match = 1;
		  break;
		}
		if (!Getattr(e, "sym:name") && (Cmp(name, Getattr(e, "name")) == 0)) {
		  match = 1;
		  break;
		}
	      }
	      if (!match) {
		Node *cc = c;
		while (cc) {
		  Node *cp = cc;
		  if (classname) {
		    Setattr(cp, "extendsmartclassname", classname);
		  }
		  Setattr(cp, "allocate:smartpointeraccess", "1");
		  /* If constant, we have to be careful */
		  if (isconst) {
		    SwigType *decl = Getattr(cp, "decl");
		    if (decl) {
		      if (SwigType_isfunction(decl)) {	/* If method, we only add if it's a const method */
			if (SwigType_isconst(decl)) {
			  Append(methods, cp);
			}
		      } else {
			Append(methods, cp);
		      }
		    } else {
		      Append(methods, cp);
		    }
		  } else {
		    Append(methods, cp);
		  }
		  cc = Getattr(cc, "sym:nextSibling");
		}
	      }
	    }
	  }
	}
      }

      c = nextSibling(c);
    }
    /* Look for methods in base classes */
    {
      Node *bases = Getattr(cls, "bases");
      int k;
      for (k = 0; k < Len(bases); k++) {
	smart_pointer_methods(Getitem(bases, k), methods, isconst);
      }
    }
    /* Remove protected/private members */
    {
      for (int i = 0; i < Len(methods);) {
	Node *n = Getitem(methods, i);
	if (!is_public(n)) {
	  Delitem(methods, i);
	  continue;
	}
	i++;
      }
    }
    return methods;
  }

  void mark_exception_classes(ParmList *p) {
    while (p) {
      SwigType *ty = Getattr(p, "type");
      SwigType *t = SwigType_typedef_resolve_all(ty);
      if (SwigType_isreference(t) || SwigType_ispointer(t) || SwigType_isarray(t)) {
	Delete(SwigType_pop(t));
      }
      Node *c = Swig_symbol_clookup(t, 0);
      if (c) {
	if (!GetFlag(c, "feature:exceptionclass")) {
	  SetFlag(c, "feature:exceptionclass");
	}
      }
      p = nextSibling(p);
      Delete(t);
    }
  }


  void process_exceptions(Node *n) {
    ParmList *catchlist = 0;
    /* 
       the "catchlist" attribute is used to emit the block

       try {$action;} 
       catch <list of catches>;

       in emit.cxx

       and is either constructued from the "feature:catches" feature
       or copied from the node "throws" list.
     */
    String *scatchlist = Getattr(n, "feature:catches");
    if (scatchlist) {
      catchlist = Swig_cparse_parms(scatchlist, n);
      if (catchlist) {
	Setattr(n, "catchlist", catchlist);
	mark_exception_classes(catchlist);
	Delete(catchlist);
      }
    }
    ParmList *throws = Getattr(n, "throws");
    if (throws) {
      /* if there is no explicit catchlist, we catch everything in the throws list */
      if (!catchlist) {
	Setattr(n, "catchlist", throws);
      }
      mark_exception_classes(throws);
    }
  }

public:
Allocate():
  inclass(NULL), extendmode(0) {
  }

  virtual int top(Node *n) {
    cplus_mode = PUBLIC;
    inclass = 0;
    extendmode = 0;
    emit_children(n);
    return SWIG_OK;
  }

  virtual int importDirective(Node *n) {
    return emit_children(n);
  }
  virtual int includeDirective(Node *n) {
    return emit_children(n);
  }
  virtual int externDeclaration(Node *n) {
    return emit_children(n);
  }
  virtual int namespaceDeclaration(Node *n) {
    return emit_children(n);
  }
  virtual int extendDirective(Node *n) {
    extendmode = 1;
    emit_children(n);
    extendmode = 0;
    return SWIG_OK;
  }

  virtual int classDeclaration(Node *n) {
    Symtab *symtab = Swig_symbol_current();
    Swig_symbol_setscope(Getattr(n, "symtab"));
    save_value<Node*> oldInclass(inclass);
    save_value<AccessMode> oldAcessMode(cplus_mode);
    save_value<int> oldExtendMode(extendmode);
    if (Getattr(n, "template"))
      extendmode = 0;
    if (!CPlusPlus) {
      /* Always have default constructors/destructors in C */
      Setattr(n, "allocate:default_constructor", "1");
      Setattr(n, "allocate:default_destructor", "1");
    }

    if (Getattr(n, "allocate:visit"))
      return SWIG_OK;
    Setattr(n, "allocate:visit", "1");

    /* Always visit base classes first */
    {
      List *bases = Getattr(n, "bases");
      if (bases) {
	for (int i = 0; i < Len(bases); i++) {
	  Node *b = Getitem(bases, i);
	  classDeclaration(b);
	}
      }
    }
    inclass = n;
    String *kind = Getattr(n, "kind");
    if (Strcmp(kind, "class") == 0) {
      cplus_mode = PRIVATE;
    } else {
      cplus_mode = PUBLIC;
    }

    emit_children(n);

    /* Check if the class is abstract via inheritance.   This might occur if a class didn't have
       any pure virtual methods of its own, but it didn't implement all of the pure methods in
       a base class */
    if (!Getattr(n, "abstracts") && is_abstract_inherit(n)) {
      if (((Getattr(n, "allocate:public_constructor") || (!GetFlag(n, "feature:nodefault") && !Getattr(n, "allocate:has_constructor"))))) {
	if (!GetFlag(n, "feature:notabstract")) {
	  Node *na = Getattr(n, "abstracts:firstnode");
	  if (na) {
	    Swig_warning(WARN_TYPE_ABSTRACT, Getfile(n), Getline(n),
			 "Class '%s' might be abstract, " "no constructors generated,\n", SwigType_namestr(Getattr(n, "name")));
	    Swig_warning(WARN_TYPE_ABSTRACT, Getfile(na), Getline(na), "Method %s might not be implemented.\n", Swig_name_decl(na));
	    if (!Getattr(n, "abstracts")) {
	      List *abstracts = NewList();
	      Append(abstracts, na);
	      Setattr(n, "abstracts", abstracts);
	      Delete(abstracts);
	    }
	  }
	}
      }
    }

    if (!Getattr(n, "allocate:has_constructor")) {
      /* No constructor is defined.  We need to check a few things */
      /* If class is abstract.  No default constructor. Sorry */
      if (Getattr(n, "abstracts")) {
	Delattr(n, "allocate:default_constructor");
      }
      if (!Getattr(n, "allocate:default_constructor")) {
	/* Check base classes */
	List *bases = Getattr(n, "allbases");
	int allows_default = 1;

	for (int i = 0; i < Len(bases); i++) {
	  Node *n = Getitem(bases, i);
	  /* If base class does not allow default constructor, we don't allow it either */
	  if (!Getattr(n, "allocate:default_constructor") && (!Getattr(n, "allocate:default_base_constructor"))) {
	    allows_default = 0;
	  }
	}
	if (allows_default) {
	  Setattr(n, "allocate:default_constructor", "1");
	}
      }
    }
    if (!Getattr(n, "allocate:has_copy_constructor")) {
      if (Getattr(n, "abstracts")) {
	Delattr(n, "allocate:copy_constructor");
      }
      if (!Getattr(n, "allocate:copy_constructor")) {
	/* Check base classes */
	List *bases = Getattr(n, "allbases");
	int allows_copy = 1;

	for (int i = 0; i < Len(bases); i++) {
	  Node *n = Getitem(bases, i);
	  /* If base class does not allow copy constructor, we don't allow it either */
	  if (!Getattr(n, "allocate:copy_constructor") && (!Getattr(n, "allocate:copy_base_constructor"))) {
	    allows_copy = 0;
	  }
	}
	if (allows_copy) {
	  Setattr(n, "allocate:copy_constructor", "1");
	}
      }
    }

    if (!Getattr(n, "allocate:has_destructor")) {
      /* No destructor was defined */
      List *bases = Getattr(n, "allbases");
      int allows_destruct = 1;

      for (int i = 0; i < Len(bases); i++) {
	Node *n = Getitem(bases, i);
	/* If base class does not allow default destructor, we don't allow it either */
	if (!Getattr(n, "allocate:default_destructor") && (!Getattr(n, "allocate:default_base_destructor"))) {
	  allows_destruct = 0;
	}
      }
      if (allows_destruct) {
	Setattr(n, "allocate:default_destructor", "1");
      }
    }

    if (!Getattr(n, "allocate:has_assign")) {
      /* No assignment operator was defined */
      List *bases = Getattr(n, "allbases");
      int allows_assign = 1;

      for (int i = 0; i < Len(bases); i++) {
	Node *n = Getitem(bases, i);
	/* If base class does not allow assignment, we don't allow it either */
	if (Getattr(n, "allocate:has_assign")) {
	  allows_assign = !Getattr(n, "allocate:noassign");
	}
      }
      if (!allows_assign) {
	Setattr(n, "allocate:noassign", "1");
      }
    }

    if (!Getattr(n, "allocate:has_new")) {
      /* No new operator was defined */
      List *bases = Getattr(n, "allbases");
      int allows_new = 1;

      for (int i = 0; i < Len(bases); i++) {
	Node *n = Getitem(bases, i);
	/* If base class does not allow new operator, we don't allow it either */
	if (Getattr(n, "allocate:has_new")) {
	  allows_new = !Getattr(n, "allocate:nonew");
	}
      }
      if (!allows_new) {
	Setattr(n, "allocate:nonew", "1");
      }
    }

    /* Check if base classes allow smart pointers, but might be hidden */
    if (!Getattr(n, "allocate:smartpointer")) {
      Node *sp = Swig_symbol_clookup("operator ->", 0);
      if (sp) {
	/* Look for parent */
	Node *p = parentNode(sp);
	if (Strcmp(nodeType(p), "extend") == 0) {
	  p = parentNode(p);
	}
	if (Strcmp(nodeType(p), "class") == 0) {
	  if (GetFlag(p, "feature:ignore")) {
	    Setattr(n, "allocate:smartpointer", Getattr(p, "allocate:smartpointer"));
	  }
	}
      }
    }

    Swig_interface_propagate_methods(n);

    /* Only care about default behavior.  Remove temporary values */
    Setattr(n, "allocate:visit", "1");
    Swig_symbol_setscope(symtab);
    return SWIG_OK;
  }

  virtual int accessDeclaration(Node *n) {
    String *kind = Getattr(n, "kind");
    if (Cmp(kind, "public") == 0) {
      cplus_mode = PUBLIC;
    } else if (Cmp(kind, "private") == 0) {
      cplus_mode = PRIVATE;
    } else if (Cmp(kind, "protected") == 0) {
      cplus_mode = PROTECTED;
    }
    return SWIG_OK;
  }

  virtual int usingDeclaration(Node *n) {

    Node *c = 0;
    for (c = firstChild(n); c; c = nextSibling(c)) {
      if (Strcmp(nodeType(c), "cdecl") == 0) {
	process_exceptions(c);

	if (inclass)
	  class_member_is_defined_in_bases(c, inclass);
      }
    }

    return SWIG_OK;
  }

  virtual int cDeclaration(Node *n) {

    process_exceptions(n);

    if (inclass) {
      /* check whether the member node n is defined in class node in class's bases */
      class_member_is_defined_in_bases(n, inclass);

      /* Check to see if this is a static member or not.  If so, we add an attribute
         cplus:staticbase that saves the current class */

      if (Swig_storage_isstatic(n)) {
	Setattr(n, "cplus:staticbase", inclass);
      }

      String *name = Getattr(n, "name");
      if (cplus_mode != PUBLIC) {
	if (Strcmp(name, "operator =") == 0) {
	  /* Look for a private assignment operator */
	  if (!GetFlag(n, "deleted"))
	    Setattr(inclass, "allocate:has_assign", "1");
	  Setattr(inclass, "allocate:noassign", "1");
	} else if (Strcmp(name, "operator new") == 0) {
	  /* Look for a private new operator */
	  if (!GetFlag(n, "deleted"))
	    Setattr(inclass, "allocate:has_new", "1");
	  Setattr(inclass, "allocate:nonew", "1");
	}
      } else {
	if (Strcmp(name, "operator =") == 0) {
	  if (!GetFlag(n, "deleted"))
	    Setattr(inclass, "allocate:has_assign", "1");
	  else
	    Setattr(inclass, "allocate:noassign", "1");
	} else if (Strcmp(name, "operator new") == 0) {
	  if (!GetFlag(n, "deleted"))
	    Setattr(inclass, "allocate:has_new", "1");
	  else
	    Setattr(inclass, "allocate:nonew", "1");
	}
	/* Look for smart pointer operator */
	if ((Strcmp(name, "operator ->") == 0) && (!GetFlag(n, "feature:ignore"))) {
	  /* Look for version with no parameters */
	  Node *sn = n;
	  while (sn) {
	    if (!Getattr(sn, "parms")) {
	      SwigType *type = SwigType_typedef_resolve_all(Getattr(sn, "type"));
	      SwigType_push(type, Getattr(sn, "decl"));
	      Delete(SwigType_pop_function(type));
	      SwigType *base = SwigType_base(type);
	      Node *sc = Swig_symbol_clookup(base, 0);
	      if ((sc) && (Strcmp(nodeType(sc), "class") == 0)) {
		if (SwigType_check_decl(type, "p.")) {
		  /* Need to check if type is a const pointer */
		  int isconst = 0;
		  Delete(SwigType_pop(type));
		  if (SwigType_isconst(type)) {
		    isconst = !Getattr(inclass, "allocate:smartpointermutable");
		    Setattr(inclass, "allocate:smartpointerconst", "1");
		  }
		  else {
		    Setattr(inclass, "allocate:smartpointermutable", "1");
		  }
		  List *methods = smart_pointer_methods(sc, 0, isconst);
		  Setattr(inclass, "allocate:smartpointer", methods);
		  Setattr(inclass, "allocate:smartpointerpointeeclassname", Getattr(sc, "name"));
		} else {
		  /* Hmmm.  The return value is not a pointer.  If the type is a value
		     or reference.  We're going to chase it to see if another operator->()
		     can be found */
		  if ((SwigType_check_decl(type, "")) || (SwigType_check_decl(type, "r."))) {
		    Node *nn = Swig_symbol_clookup("operator ->", Getattr(sc, "symtab"));
		    if (nn) {
		      Delete(base);
		      Delete(type);
		      sn = nn;
		      continue;
		    }
		  }
		}
	      }
	      Delete(base);
	      Delete(type);
	      break;
	    }
	  }
	}
      }
    }
    return SWIG_OK;
  }

  virtual int constructorDeclaration(Node *n) {
    if (!inclass)
      return SWIG_OK;
    Parm *parms = Getattr(n, "parms");

    process_exceptions(n);
    if (!extendmode) {
      if (!ParmList_numrequired(parms)) {
	/* Class does define a default constructor */
	/* However, we had better see where it is defined */
	if (cplus_mode == PUBLIC) {
	  Setattr(inclass, "allocate:default_constructor", "1");
	} else if (cplus_mode == PROTECTED) {
	  Setattr(inclass, "allocate:default_base_constructor", "1");
	}
      }
      /* Class defines some kind of constructor. May or may not be public */
      Setattr(inclass, "allocate:has_constructor", "1");
      if (cplus_mode == PUBLIC) {
	Setattr(inclass, "allocate:public_constructor", "1");
      }
    } else {
      Setattr(inclass, "allocate:has_constructor", "1");
      Setattr(inclass, "allocate:public_constructor", "1");
    }


    /* See if this is a copy constructor */
    if (parms && (ParmList_numrequired(parms) == 1)) {
      /* Look for a few cases. X(const X &), X(X &), X(X *) */
      int copy_constructor = 0;
      SwigType *type = Getattr(inclass, "name");
      String *tn = NewStringf("r.q(const).%s", type);
      String *cc = SwigType_typedef_resolve_all(tn);
      SwigType *rt = SwigType_typedef_resolve_all(Getattr(parms, "type"));
      if (SwigType_istemplate(type)) {
	String *tmp = Swig_symbol_template_deftype(cc, 0);
	Delete(cc);
	cc = tmp;
	tmp = Swig_symbol_template_deftype(rt, 0);
	Delete(rt);
	rt = tmp;
      }
      if (Strcmp(cc, rt) == 0) {
	copy_constructor = 1;
      } else {
	Delete(cc);
	cc = NewStringf("r.%s", Getattr(inclass, "name"));
	if (Strcmp(cc, Getattr(parms, "type")) == 0) {
	  copy_constructor = 1;
	} else {
	  Delete(cc);
	  cc = NewStringf("p.%s", Getattr(inclass, "name"));
	  String *ty = SwigType_strip_qualifiers(Getattr(parms, "type"));
	  if (Strcmp(cc, ty) == 0) {
	    copy_constructor = 1;
	  }
	  Delete(ty);
	}
      }
      Delete(cc);
      Delete(rt);
      Delete(tn);

      if (copy_constructor) {
	Setattr(n, "copy_constructor", "1");
	Setattr(inclass, "allocate:has_copy_constructor", "1");
	if (cplus_mode == PUBLIC) {
	  Setattr(inclass, "allocate:copy_constructor", "1");
	} else if (cplus_mode == PROTECTED) {
	  Setattr(inclass, "allocate:copy_base_constructor", "1");
	}
      }
    }
    return SWIG_OK;
  }

  virtual int destructorDeclaration(Node *n) {
    (void) n;
    if (!inclass)
      return SWIG_OK;
    if (!extendmode) {
      Setattr(inclass, "allocate:has_destructor", "1");
      if (cplus_mode == PUBLIC) {
	Setattr(inclass, "allocate:default_destructor", "1");
      } else if (cplus_mode == PROTECTED) {
	Setattr(inclass, "allocate:default_base_destructor", "1");
      } else if (cplus_mode == PRIVATE) {
	Setattr(inclass, "allocate:private_destructor", "1");
      }
    } else {
      Setattr(inclass, "allocate:has_destructor", "1");
      Setattr(inclass, "allocate:default_destructor", "1");
    }
    return SWIG_OK;
  }
};

void Swig_default_allocators(Node *n) {
  if (!n)
    return;
  Allocate *a = new Allocate;
  a->top(n);
  delete a;
}
