/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * interface.cxx
 *
 * This module contains support for the interface feature.
 * This feature is used in language modules where the target language does not
 * naturally support C++ style multiple inheritance, but does support inheritance 
 * from multiple interfaces.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

static bool interface_feature_enabled = false;

/* -----------------------------------------------------------------------------
 * collect_interface_methods()
 *
 * Create a list of all the methods from the base classes of class n that are
 * marked as an interface. The resulting list is thus the list of methods that
 * need to be implemented in order for n to be non-abstract.
 * ----------------------------------------------------------------------------- */

static List *collect_interface_methods(Node *n) {
  List *methods = NewList();
  if (Hash *bases = Getattr(n, "interface:bases")) {
    List *keys = Keys(bases);
    for (Iterator base = First(keys); base.item; base = Next(base)) {
      Node *cls = Getattr(bases, base.item);
      if (cls == n)
	continue;
      for (Node *child = firstChild(cls); child; child = nextSibling(child)) {
	if (Cmp(nodeType(child), "cdecl") == 0) {
	  if (GetFlag(child, "feature:ignore") || Getattr(child, "interface:owner"))
	    continue; // skip methods propagated to bases
	  Node *m = Copy(child);
	  set_nextSibling(m, NIL);
	  set_previousSibling(m, NIL);
	  Setattr(m, "interface:owner", cls);
	  Append(methods, m);
	}
      }
    }
    Delete(keys);
  }
  return methods;
}

/* -----------------------------------------------------------------------------
 * collect_interface_bases
 * ----------------------------------------------------------------------------- */

static void collect_interface_bases(Hash *bases, Node *n) {
  if (Getattr(n, "feature:interface")) {
    String *name = Getattr(n, "interface:name");
    if (!Getattr(bases, name))
      Setattr(bases, name, n);
  }

  if (List *baselist = Getattr(n, "bases")) {
    for (Iterator base = First(baselist); base.item; base = Next(base)) {
      if (!GetFlag(base.item, "feature:ignore")) {
	if (Getattr(base.item, "feature:interface"))
	  collect_interface_bases(bases, base.item);
      }
    }
  }
}

/* -----------------------------------------------------------------------------
 * collect_interface_base_classes()
 *
 * Create a hash containing all the classes up the inheritance hierarchy
 * marked with feature:interface (including this class n).
 * Stops going up the inheritance chain as soon as a class is found without
 * feature:interface.
 * The idea is to find all the base interfaces that a class must implement.
 * ----------------------------------------------------------------------------- */

static void collect_interface_base_classes(Node *n) {
  if (Getattr(n, "feature:interface")) {
    // check all bases are also interfaces
    if (List *baselist = Getattr(n, "bases")) {
      for (Iterator base = First(baselist); base.item; base = Next(base)) {
	if (!GetFlag(base.item, "feature:ignore")) {
	  if (!Getattr(base.item, "feature:interface")) {
	    Swig_error(Getfile(n), Getline(n), "Base class '%s' of '%s' is not similarly marked as an interface.\n", SwigType_namestr(Getattr(base.item, "name")), SwigType_namestr(Getattr(n, "name")));
	    SWIG_exit(EXIT_FAILURE);
	  }
	}
      }
    }
  }

  Hash *interface_bases = NewHash();
  collect_interface_bases(interface_bases, n);
  if (Len(interface_bases) == 0)
    Delete(interface_bases);
  else
    Setattr(n, "interface:bases", interface_bases);
}

/* -----------------------------------------------------------------------------
 * process_interface_name()
 * ----------------------------------------------------------------------------- */

static void process_interface_name(Node *n) {
  if (Getattr(n, "feature:interface")) {
    String *interface_name = Getattr(n, "feature:interface:name");
    if (!Len(interface_name)) {
      Swig_error(Getfile(n), Getline(n), "The interface feature for '%s' is missing the name attribute.\n", SwigType_namestr(Getattr(n, "name")));
      SWIG_exit(EXIT_FAILURE);
    }
    if (Strchr(interface_name, '%')) {
      String *name = NewStringf(interface_name, Getattr(n, "sym:name"));
      Setattr(n, "interface:name", name);
    } else {
      Setattr(n, "interface:name", interface_name);
    }
  }
}

/* -----------------------------------------------------------------------------
 * Swig_interface_propagate_methods()
 *
 * Find all the base classes marked as an interface (with feature:interface) for
 * class node n. For each of these, add all of its methods as methods of n so that
 * n is not abstract. If class n is also marked as an interface, it will remain
 * abstract and not have any methods added.
 * ----------------------------------------------------------------------------- */

void Swig_interface_propagate_methods(Node *n) {
  if (interface_feature_enabled) {
    process_interface_name(n);
    collect_interface_base_classes(n);
    List *methods = collect_interface_methods(n);
    bool is_interface = Getattr(n, "feature:interface") != 0;
    for (Iterator mi = First(methods); mi.item; mi = Next(mi)) {
      if (!is_interface && GetFlag(mi.item, "abstract"))
	continue;
      String *this_decl = Getattr(mi.item, "decl");
      String *this_decl_resolved = SwigType_typedef_resolve_all(this_decl);
      bool identically_overloaded_method = false; // true when a base class' method is implemented in n
      if (SwigType_isfunction(this_decl_resolved)) {
	String *name = Getattr(mi.item, "name");
	for (Node *child = firstChild(n); child; child = nextSibling(child)) {
	  if (Getattr(child, "interface:owner"))
	    break; // at the end of the list are newly appended methods
	  if (checkAttribute(child, "name", name)) {
	    String *decl = SwigType_typedef_resolve_all(Getattr(child, "decl"));
	    identically_overloaded_method = Strcmp(decl, this_decl_resolved) == 0;
	    Delete(decl);
	    if (identically_overloaded_method)
	      break;
	  }
	}
      }
      Delete(this_decl_resolved);
      if (!identically_overloaded_method) {
	// TODO: Fix if the method is overloaded with different arguments / has default args
	appendChild(n, mi.item);
      } else {
	Delete(mi.item);
      }
    }
    Delete(methods);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_interface_feature_enable()
 *
 * Turn on interface feature support
 * ----------------------------------------------------------------------------- */

void Swig_interface_feature_enable() {
  interface_feature_enabled = true;
}
