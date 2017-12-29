/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * allegrocl.cxx
 *
 * ALLEGROCL language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"
#include <ctype.h>

// #define ALLEGROCL_DEBUG
// #define ALLEGROCL_WRAP_DEBUG
// #define ALLEGROCL_TYPE_DEBUG
// #define ALLEGROCL_CLASS_DEBUG

static const char *usage = "\
Allegro CL Options (available with -allegrocl)\n\
     -identifier-converter <type or funcname> - \n\
                       Specifies the type of conversion to do on C identifiers to convert\n\
                       them to symbols. There are two built-in converters: 'null' and\n\
                       'lispify'. The default is 'null'. If you supply a name other\n\
                       than one of the built-ins, then a function by that name will be\n\
                       called to convert identifiers to symbols.\n\
     -[no]cwrap      - Turn on or turn off generation of an intermediate C file when\n\
                       creating a C interface. By default this is only done for C++ code.\n\
     -isolate        - Define all SWIG helper functions in a package unique to this\n\
                       module. Avoids redefinition warnings when loading multiple\n\
                       SWIGged modules into the same running Allegro CL image.\n\
";

static File *f_cl = 0;
String *f_clhead = NewString("");
String *f_clwrap = NewString("(swig-in-package ())\n\n");
static File *f_begin;
static File *f_runtime;
static File *f_cxx_header = 0;
static File *f_cxx_wrapper = 0;

static String *module_name = 0;
static String *swig_package = 0;

static String *identifier_converter = NewString("identifier-convert-null");

static bool CWrap = true;	// generate wrapper file for C code by default. most correct.
static bool Generate_Wrapper = false;
static bool unique_swig_package = false;

static SwigType *fwdref_ffi_type = NewString("__SWIGACL_FwdReference");

static String *current_namespace = NewString("");
static String *current_package = NewString("");
static Hash *defined_namespace_packages = NewHash();
static Node *in_class = 0;

static Node *first_linked_type = 0;
static Hash *defined_foreign_types = NewHash();
static Hash *defined_foreign_ltypes = NewHash();

static String *anon_type_name = NewString("anontype");
static int anon_type_count = 0;

// stub
String *convert_literal(String *num_param, String *type, bool try_to_split = true);

class ALLEGROCL:public Language {
public:
  virtual void main(int argc, char *argv[]);
  virtual int top(Node *n);
  virtual int functionWrapper(Node *n);
  virtual int namespaceDeclaration(Node *n);
  virtual int constructorHandler(Node *n);
  virtual int destructorHandler(Node *n);
  virtual int globalvariableHandler(Node *n);
  virtual int variableWrapper(Node *n);
  virtual int constantWrapper(Node *n);
  virtual int memberfunctionHandler(Node *n);
  virtual int membervariableHandler(Node *n);
  virtual int classHandler(Node *n);
  virtual int emit_one(Node *n);
  virtual int enumDeclaration(Node *n);
  virtual int enumvalueDeclaration(Node *n);
  virtual int typedefHandler(Node *n);
  virtual int classforwardDeclaration(Node *n);
  virtual int templateDeclaration(Node *n);
  virtual int validIdentifier(String *s);
private:
  int emit_defun(Node *n, File *f_cl);
  int emit_dispatch_defun(Node *n);
  int emit_buffered_defuns(Node *n);
  int cClassHandler(Node *n);
  int cppClassHandler(Node *n);
};
static ALLEGROCL *allegrocl = 0;

static String *trim(String *str) {
  char *c = Char(str);
  while (*c != '\0' && isspace((int) *c))
    ++c;
  String *result = NewString(c);
  Chop(result);
  return result;
}

int is_integer(String *s) {
  char *c = Char(s);
  if (c[0] == '#' && (c[1] == 'x' || c[1] == 'o'))
    c += 2;

  while (*c) {
    if (!isdigit(*c))
      return 0;
    c++;
  }
  return 1;
}

String *class_from_class_or_class_ref(String *type) {
  SwigType *stripped = SwigType_strip_qualifiers(type);
  if (SwigType_isclass(stripped))
    return stripped;

  if (SwigType_ispointer(stripped) || SwigType_isreference(stripped)) {
    // Printf(stderr,"It is a pointer/reference. Is it a class?\n");
    SwigType_pop(stripped);
    if (SwigType_isclass(stripped)) {
      return stripped;
    }
  }
  return 0;
}

String *lookup_defined_foreign_type(String *k) {

#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "Looking up defined type '%s'.\n  Found: '%s'\n", k, Getattr(defined_foreign_types, k));
#endif

  return Getattr(defined_foreign_types, k);
}

String *listify_namespace(String *namespaze) {
  if (Len(namespaze) == 0)
    return NewString("()");
  String *result = NewStringf("(\"%s\")", namespaze);
  Replaceall(result, "::", "\" \"");
  return result;
}

String *namespaced_name(Node *n, String *ns = current_namespace) {

  return NewStringf("%s%s%s", ns, (Len(ns) != 0) ? "::" : "", Getattr(n, "sym:name"));
}

// "Namespace::Nested::Class2::Baz" -> "Baz"
static String *strip_namespaces(String *str) {
  return Swig_scopename_last(str);
}

void add_linked_type(Node *n) {
#ifdef ALLEGROCL_CLASS_DEBUG
  Printf(stderr, "Adding linked node of type: %s(%s) %s(%p)\n\n", nodeType(n), Getattr(n, "storage"), Getattr(n, "name"), n);
  // Swig_print_node(n);
#endif
  if (!first_linked_type) {
    first_linked_type = n;
    Setattr(n, "allegrocl:last_linked_type", n);
  } else {
    Node *t = Getattr(first_linked_type, "allegrocl:last_linked_type");
    Setattr(t, "allegrocl:next_linked_type", n);
    Setattr(first_linked_type, "allegrocl:last_linked_type", n);
  }
}

void replace_linked_type(Node *old, Node *new_node) {
  Node *prev = Getattr(old, "allegrocl:prev_linked_type");

  Setattr(new_node, "allegrocl:next_linked_type", Getattr(old, "allegrocl:next_linked_type"));
  if (prev)
    Setattr(prev, "allegrocl:next_linked_type", new_node);
  Delattr(old, "allegrocl:next_linked_type");
  Delattr(old, "allegrocl:prev_linked_type");

  // check if we're replacing the first link.
  if (first_linked_type == old) {
    first_linked_type = new_node;
    Setattr(first_linked_type, "allegrocl:last_linked_type", Getattr(old, "allegrocl:last_linked_type"));
  }
  // check if we're replacing the last link.
  if (Getattr(first_linked_type, "allegrocl:last_linked_type") == old)
    Setattr(first_linked_type, "allegrocl:last_linked_type", new_node);
}

void insert_linked_type_at(Node *old, Node *new_node, int before = 1) {
  Node *p = 0;

  if (!first_linked_type) {
    add_linked_type(new_node);
    return;
  }

  if (!before) {
    Setattr(new_node, "allegrocl:next_linked_type", Getattr(old, "allegrocl:next_linked_type"));
    Setattr(old, "allegrocl:next_linked_type", new_node);
    if (Getattr(first_linked_type, "allegrocl:last_linked_type") == old)
      Setattr(first_linked_type, "allegrocl:last_linked_type", new_node);
  } else {
    Node *c = first_linked_type;
    while (c) {
      if (c == old) {
	break;
      } else {
	p = c;
	c = Getattr(c, "allegrocl:next_linked_type");
      }
    }
    if (c == old) {
      Setattr(new_node, "allegrocl:next_linked_type", c);
      if (first_linked_type == c) {
	first_linked_type = new_node;
	Setattr(first_linked_type, "allegrocl:last_linked_type", Getattr(c, "allegrocl:last_linked_type"));
	Delattr(c, "allegrocl:last_linked_type");
      }
      if (p)
	Setattr(p, "allegrocl:next_linked_type", new_node);
    }
  }
}

Node *find_linked_type_by_name(String *name) {
  Node *p = 0;
  Node *c = first_linked_type;

  // Printf(stderr,"in find_linked_type_by_name '%s'...", name);
  while (c) {
    String *key = Getattr(c, "name");
    if (!Strcmp(key, name)) {
      break;
    } else {
      p = c;
      c = Getattr(c, "allegrocl:next_linked_type");
    }
  }
  // Printf(stderr,"exit find_linked_type_by_name.\n");

  if (p && c)
    Setattr(c, "allegrocl:prev_linked_type", p);
  // Printf(stderr,"find_linked_type_by_name: DONE\n");
  return c;
}

Node *get_primary_synonym_of(Node *n) {
  Node *p = Getattr(n, "allegrocl:synonym-of");
  Node *prim = n;

  // Printf(stderr, "getting primary synonym of %p\n", n);
  while (p) {
    // Printf(stderr, "   found one! %p\n", p);
    prim = p;
    p = Getattr(p, "allegrocl:synonym-of");
  }
  // Printf(stderr,"get_primary_syn: DONE. returning %s(%p)\n", Getattr(prim,"name"),prim);
  return prim;
}

void add_forward_referenced_type(Node *n, int overwrite = 0) {
  String *k = Getattr(n, "name");
  String *name = Getattr(n, "sym:name");
  String *ns = listify_namespace(current_namespace);

  String *val = Getattr(defined_foreign_types, k);

  if (!val || overwrite) {
#ifdef ALLEGROCL_TYPE_DEBUG
    Printf(stderr, "Adding forward reference for %s (overwrite=%d)\n", k, overwrite);
#endif
    Setattr(defined_foreign_types, Copy(k), NewString("forward-reference"));

    String *mangled_lname_gen = NewStringf("#.(swig-insert-id \"%s\" %s :type :class)", name, ns);

    Setattr(defined_foreign_ltypes, Copy(k), mangled_lname_gen);
    //    Printf(f_cl, ";; forward reference stub\n"
    //           "(swig-def-foreign-class \"%s\" (ff:foreign-pointer) (:class ))\n\n"
    //     , name);

#ifdef ALLEGROCL_CLASS_DEBUG
    Printf(stderr, "Linking forward reference type = %s(%p)\n", k, n);
#endif
    add_linked_type(n);
  }
}

void add_defined_foreign_type(Node *n, int overwrite = 0, String *k = 0,
			      String *name = 0, String *ns = current_namespace) {

  String *val;
  String *ns_list = listify_namespace(ns);
  String *templated = n ? Getattr(n, "template") : 0;
  String *cDeclName = n ? Getattr(n, "name") : 0;

#ifdef ALLEGROCL_CLASS_DEBUG
  Printf(stderr, "IN A-D-F-T. (n=%p, ow=%d, k=%s, name=%s, ns=%s\n", n, overwrite, k, name, ns);
  Printf(stderr, "    templated = '%p', classDecl = '%p'\n", templated, cDeclName);
#endif
  if (n) {
    if (!name)
      name = Getattr(n, "sym:name");
    if (!name)
      name = strip_namespaces(Getattr(n, "name"));
    if (templated) {
      k = namespaced_name(n);
    } else {
      String *kind_of_type = Getattr(n, "kind");

      /*
         For typedefs of the form:

         typedef struct __xxx { ... } xxx;

	 behavior differs between C mode and C++ mode.

	 C Mode:
         add_defined_foreign_type will be called once via classHandler
         to define the type for 'struct __xxx' and add the mapping from
	 'struct __xxx' -> 'xxx'

	 It will also be called once via typedefHandler to add the
	 mapping 'xxx' -> 'xxx'

	 C++ Mode:
	 add_defined_foreign_type will be called once via classHandler
	 to define the type for 'xxx'. it also adds the mapping from
	 'xxx' -> 'xxx' and also for 'struct xxx' -> 'xxx'

	 In typedefHandler, we again try to add the mapping from
	 'xxx' -> 'xxx', which already exists. This second mapping
	 is ignored.

	 Both modes:

         All references to this typedef'd struct will appear in
         generated lisp code as an objectd of type 'xxx'. For
         non-typedef'd structs, the classHand mapping will be

           struct __xxx -> (swig-insert-id "__xxx")
       */
      // Swig_print_node(n);
      String *unnamed = Getattr(n, "unnamed");
      if (kind_of_type && (!Strcmp(kind_of_type, "struct")
			   || !Strcmp(kind_of_type, "union")) && cDeclName && !unnamed) {
	k = NewStringf("%s %s", kind_of_type, cDeclName);
      } else {
	if (!Strcmp(nodeType(n), "enum") && unnamed) {
	  name = NewStringf("%s%d", anon_type_name, anon_type_count++);
	  k = NewStringf("enum %s", name);
	  Setattr(n, "allegrocl:name", name);

	} else {
	  k = k ? k : Getattr(n, "name");
	}
      }
    }
    // Swig_print_node(n);
  }

  String *tname = SwigType_istemplate_templateprefix(name);
  if (tname) {
    String *temp = strip_namespaces(tname);
    name = NewStringf("%s%s%s", temp, SwigType_templateargs(name), SwigType_templatesuffix(name));
    Delete(temp);
    Delete(tname);
  }

  val = lookup_defined_foreign_type(k);

  int is_fwd_ref = 0;
  if (val)
    is_fwd_ref = !Strcmp(val, "forward-reference");

  if (!val || overwrite || is_fwd_ref) {
#ifdef ALLEGROCL_CLASS_DEBUG
    Printf(stderr, "Adding defined type '%s' = '%s' '%s' (overwrite=%d, in-class=%d)\n", k, ns, name, overwrite, in_class);
#endif
    String *mangled_name_gen = NewStringf("#.(swig-insert-id \"%s\" %s :type :type)", name, ns_list);
    String *mangled_lname_gen = NewStringf("#.(swig-insert-id \"%s\" %s :type :class)", name, ns_list);

    Setattr(defined_foreign_types, Copy(k), Copy(mangled_name_gen));
    Setattr(defined_foreign_ltypes, Copy(k), Copy(mangled_lname_gen));

    if (CPlusPlus) {
      bool cpp_struct = Strstr(k, "struct ") ? true : false;
      bool cpp_union = Strstr(k, "union ") ? true : false;

      String *cpp_type = 0;
      if (cpp_struct) {
	cpp_type = Copy(k);
	Replaceall(cpp_type, "struct ", "");
      } else if (cpp_union) {
	cpp_type = Copy(k);
	Replaceall(cpp_type, "union ", "");
      }

      if (cpp_struct || cpp_union) {
#ifdef ALLEGROCL_CLASS_DEBUG
	Printf(stderr, " Also adding defined type '%s' = '%s' '%s' (overwrite=%d)\n", cpp_type, ns, name, overwrite);
#endif
	Setattr(defined_foreign_types, Copy(cpp_type), Copy(mangled_name_gen));
	Setattr(defined_foreign_ltypes, Copy(cpp_type), Copy(mangled_lname_gen));
      }
    }
#ifdef ALLEGROCL_CLASS_DEBUG
    Printf(stderr, "looking to add %s/%s(%p) to linked_type_list...\n", k, name, n);
#endif
    if (is_fwd_ref) {
      // Printf(stderr,"*** 1\n");
      if (n)
	add_linked_type(n);
    } else {
      // Printf(stderr,"*** 1-a\n");
      if (SwigType_istemplate(k)) {
	SwigType *resolved = SwigType_typedef_resolve_all(k);
	// Printf(stderr,"*** 1-b\n");
	Node *match = find_linked_type_by_name(resolved);
	Node *new_node = 0;
	// Printf(stderr, "*** temp-1\n");
	if (n) {
	  new_node = n;
	} else {
#ifdef ALLEGROCL_CLASS_DEBUG
	  Printf(stderr, "Creating a new templateInst:\n");
	  Printf(stderr, "       name = %s\n", resolved);
	  Printf(stderr, "   sym:name = %s\n", name);
	  Printf(stderr, "  real-name = %s\n", k);
	  Printf(stderr, "       type = %s\n", resolved);
	  Printf(stderr, "         ns = %s\n\n", ns);
#endif
	  new_node = NewHash();
	  Setattr(new_node, "nodeType", "templateInst");
	  Setattr(new_node, "name", Copy(resolved));
	  Setattr(new_node, "sym:name", Copy(name));
	  Setattr(new_node, "real-name", Copy(k));
	  Setattr(new_node, "type", Copy(resolved));
	  Setattr(new_node, "allegrocl:namespace", ns);
	  Setattr(new_node, "allegrocl:package", ns);
	}

	if (!match) {
	  if (!Strcmp(nodeType(new_node), "templateInst") && in_class) {
	    /* this is an implicit template instantiation found while
	       walking a class. need to insert this into the
	       linked_type list before the current class definition */
#ifdef ALLEGROCL_CLASS_DEBUG
	    Printf(stderr, "trying to insert a templateInst before a class\n");
#endif
	    insert_linked_type_at(in_class, new_node);
#ifdef ALLEGROCL_CLASS_DEBUG
	    Printf(stderr, "DID IT!\n");
#endif
	  } else {
	    // Printf(stderr,"*** 3\n");
	    add_linked_type(new_node);
	  }
	  Setattr(new_node, "allegrocl:synonym:is-primary", "1");
	} else {
	  // a synonym type was found (held in variable 'match')
	  // Printf(stderr, "setting primary synonym of %p to %p\n", new_node, match);
	  if (new_node == match)
	    Printf(stderr, "Hey-4 * - '%s' is a synonym of iteself!\n", Getattr(new_node, "name"));
	  Setattr(new_node, "allegrocl:synonym-of", match);
	  // Printf(stderr,"*** 4\n");
	  add_linked_type(new_node);
	}
      } else {
	Node *match;

	if (!Strcmp(nodeType(n), "cdecl") && !Strcmp(Getattr(n, "storage"), "typedef")) {
	  SwigType *type = SwigType_strip_qualifiers(Getattr(n, "type"));
#ifdef ALLEGROCL_CLASS_DEBUG
	  Printf(stderr, "Examining typedef '%s' for class references. (%d)\n", type, SwigType_isclass(type));
#endif
	  if (SwigType_isclass(type)) {
#ifdef ALLEGROCL_CLASS_DEBUG
	    Printf(stderr, "Found typedef of a class '%s'\n", type);
#endif
	    /* 
	       For the following parsed expression:

	       typedef struct __xxx { ... } xxx;

	       if n is of kind "class" (defining the class 'struct __xxx'
	       then we add n to the linked type list.

	       if n is "cdecl" node of storage "typedef" (to note
	       that xxx is equivalent to 'struct __xxx' then we don't
	       want to add this node to the linked type list.
	     */
	    String *defined_type = lookup_defined_foreign_type(type);
	    String *defined_key_type = lookup_defined_foreign_type(k);

	    if ((Strstr(type, "struct ") || Strstr(type, "union "))
		&& defined_type && !Strcmp(defined_type, defined_key_type)) {
	      // mark as a synonym but don't add to linked_type list
	      // Printf(stderr,"*** 4.8\n");
	      Setattr(n, "allegrocl:synonym", "1");
	    } else {
	      SwigType *lookup_type = SwigType_istemplate(type) ? SwigType_typedef_resolve_all(type) : Copy(type);
	      match = find_linked_type_by_name(lookup_type);
	      if (match) {
		Setattr(n, "allegrocl:synonym", "1");
		Setattr(n, "allegrocl:synonym-of", match);
		Setattr(n, "real-name", Copy(lookup_type));

		// Printf(stderr, "*** pre-5: found match of '%s'(%p)\n", Getattr(match,"name"),match);
		// if(n == match) Printf(stderr, "Hey-5 *** setting synonym of %p to %p\n", n, match);
		// Printf(stderr,"*** 5\n");
		add_linked_type(n);
	      } else {
#ifdef ALLEGROCL_CLASS_DEBUG
		Printf(stderr, "Creating classfoward node for struct stub in typedef.\n");
#endif
		Node *new_node = NewHash();
		String *symname = Copy(type);
		Replaceall(symname, "struct ", "");
		Setattr(new_node, "nodeType", "classforward");
		Setattr(new_node, "name", Copy(type));
		Setattr(new_node, "sym:name", symname);
		Setattr(new_node, "allegrocl:namespace", ns);
		Setattr(new_node, "allegrocl:package", ns);

		String *mangled_new_name = NewStringf("#.(swig-insert-id \"%s\" %s)", symname, ns_list);
		String *mangled_new_lname = NewStringf("#.(swig-insert-id \"%s\" %s :type :class)", symname, ns_list);
		Setattr(defined_foreign_types, Copy(symname), Copy(mangled_new_name));
		Setattr(defined_foreign_ltypes, Copy(symname), Copy(mangled_new_lname));

		// Printf(stderr,"Weird! Can't find the type!\n");
		add_forward_referenced_type(new_node);
		add_linked_type(new_node);

		Setattr(n, "allegrocl:synonym", "1");
		Setattr(n, "allegrocl:synonym-of", new_node);

		add_linked_type(n);
	      }
	      Delete(lookup_type);
	    }
	  } else {
	    // check if it's a pointer or reference to a class.
	    // Printf(stderr,"Checking if '%s' is a p. or r. to a class\n", type);
	    String *class_ref = class_from_class_or_class_ref(type);
	    if (class_ref) {
	      match = find_linked_type_by_name(class_ref);
	      Setattr(n, "allegrocl:synonym", "1");
	      Setattr(n, "allegrocl:synonym-of", match);
	      add_linked_type(n);
	    }
	  }
	  Delete(type);
	  // synonym types have already been added.
	  // Printf(stderr,"*** 10\n");
	  if (!Getattr(n, "allegrocl:synonym"))
	    add_linked_type(n);
	} else if (Getattr(n, "template")) {
	  // Printf(stderr, "this is a class template node(%s)\n", nodeType(n));
	  String *resolved = SwigType_typedef_resolve_all(Getattr(n, "name"));

#ifdef ALLEGROCL_CLASS_DEBUG
	  Printf(stderr, "   looking up %s for linked type match with %s...\n", Getattr(n, "sym:name"), resolved);
#endif
	  match = find_linked_type_by_name(resolved);
	  if (!match) {
#ifdef ALLEGROCL_CLASS_DEBUG
	    Printf(stderr, "found no implicit instantiation of %%template node %s(%p)\n", Getattr(n, "name"), n);
#endif
	    add_linked_type(n);
	  } else {
	    Node *primary = get_primary_synonym_of(match);

	    Setattr(n, "allegrocl:synonym:is-primary", "1");
	    Delattr(primary, "allegrocl:synonym:is-primary");
	    if (n == match)
	      Printf(stderr, "Hey-7 * setting synonym of %p to %p\n (match = %p)", primary, n, match);
	    Setattr(primary, "allegrocl:synonym-of", n);
	    // Printf(stderr,"*** 7\n");
	    add_linked_type(n);
	  }
	} else {
#ifdef ALLEGROCL_CLASS_DEBUG
	  Printf(stderr, "linking type '%s'(%p)\n", k, n);
#endif
	  // Printf(stderr,"*** 8\n");
	  add_linked_type(n);
	}
      }
    }
    Delete(mangled_name_gen);
    Delete(mangled_lname_gen);
  } else {
    if (!CPlusPlus || Strcmp(Getattr(n,"kind"),"typedef")) {
       Swig_warning(WARN_TYPE_REDEFINED, Getfile(n), Getline(n),
		    "Attempting to store a foreign type that exists: %s (%s)\n",
		    k, val);
    } 
  }

  Delete(ns_list);

#ifdef ALLEGROCL_CLASS_DEBUG
  Printf(stderr, "OUT A-D-F-T\n");
#endif
}

void note_implicit_template_instantiation(SwigType *t) {
  // the namespace of the implicit instantiation is not necessarily
  // current_namespace. Attempt to cull this from the type.
#ifdef ALLEGROCL_CLASS_DEBUG
  Printf(stderr, "culling namespace of '%s' from '%s'\n", t, SwigType_templateprefix(t));
#endif
  SwigType *type = Copy(t);
  SwigType *tok = SwigType_pop(type);
  String *implicit_ns = SwigType_istemplate(tok) ? Swig_scopename_prefix(SwigType_templateprefix(tok)) : 0;
  add_defined_foreign_type(0, 0, t, t, implicit_ns ? implicit_ns : current_namespace);

  Delete(type);
}

String *get_ffi_type(Node *n, SwigType *ty, const_String_or_char_ptr name) {
  /* lookup defined foreign type.
     if it exists, it will return a form suitable for placing
     into lisp code to generate the def-foreign-type name */

#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "inside g_f_t: looking up '%s' '%s'\n", ty, name);
#endif

  String *found_type = lookup_defined_foreign_type(ty);

  if (found_type) {
#ifdef ALLEGROCL_TYPE_DEBUG
    Printf(stderr, "found_type '%s'\n", found_type);
#endif
    return (Strcmp(found_type, "forward-reference") ? Copy(found_type) : get_ffi_type(n, fwdref_ffi_type, ""));
  } else {
    Node *node = NewHash();
    Setattr(node, "type", ty);
    Setfile(node, Getfile(n));
    Setline(node, Getline(n));
    const String *tm = Swig_typemap_lookup("ffitype", node, name, 0);
    Delete(node);

    if (tm) {
#ifdef ALLEGROCL_TYPE_DEBUG
      Printf(stderr, "g-f-t: found ffitype typemap '%s'\n", tm);
#endif
      return NewString(tm);
    }

    if (SwigType_istemplate(ty)) {
      note_implicit_template_instantiation(ty);
      return Copy(lookup_defined_foreign_type(ty));
    }
  }
  return 0;
}

String *lookup_defined_foreign_ltype(String *l) {

#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "Looking up defined ltype '%s'.\n  Found: '%s'\n", l, Getattr(defined_foreign_ltypes, l));
#endif
  return Getattr(defined_foreign_ltypes, l);
}

/* walk type and return string containing lisp version.
   recursive. */
String *internal_compose_foreign_type(Node *n, SwigType *ty) {

  SwigType *tok;
  String *ffiType = NewString("");

  // for a function type, need to walk the parm list.
  while (Len(ty) != 0) {
    tok = SwigType_pop(ty);

    if (SwigType_isfunction(tok)) {
      // Generate Function wrapper
      Printf(ffiType, "(:function ");
      // walk parm list
      List *pl = SwigType_parmlist(tok);

      Printf(ffiType, "(");	// start parm list
      for (Iterator i = First(pl); i.item; i = Next(i)) {
	SwigType *f_arg = SwigType_strip_qualifiers(i.item);
	Printf(ffiType, "%s ", internal_compose_foreign_type(n, f_arg));
	Delete(f_arg);
      }
      Printf(ffiType, ")");	// end parm list.

      // do function return type.
      Printf(ffiType, " %s)", internal_compose_foreign_type(n, ty));
      break;
    } else if (SwigType_ispointer(tok) || SwigType_isreference(tok)) {
      Printf(ffiType, "(* %s)", internal_compose_foreign_type(n, ty));
    } else if (SwigType_isarray(tok)) {
      Printf(ffiType, "(:array %s", internal_compose_foreign_type(n, ty));
      String *atype = NewString("int");
      String *dim = convert_literal(SwigType_array_getdim(tok, 0), atype);
      Delete(atype);
      if (is_integer(dim)) {
	Printf(ffiType, " %s)", dim);
      } else {
	Printf(ffiType, " #| %s |#)", SwigType_array_getdim(tok, 0));
      }
    } else if (SwigType_ismemberpointer(tok)) {
      // temp
      Printf(ffiType, "(* %s)", internal_compose_foreign_type(n, ty));
    } else {
      String *res = get_ffi_type(n, tok, "");
      if (res) {
	Printf(ffiType, "%s", res);
      } else {
	SwigType *resolved_type = SwigType_typedef_resolve_all(tok);
	if (Cmp(resolved_type, tok) != 0) {
	  res = get_ffi_type(n, resolved_type, "");
	  if (res) {
	  } else {
	    res = internal_compose_foreign_type(n, resolved_type);
	  }
	  if (res)
	    Printf(ffiType, "%s", res);
	}

	if (!res) {
	  String *is_struct = 0;
	  String *tok_remove_text = 0;
	  String *tok_name = Copy(tok);
	  String *tok_key = SwigType_str(tok,0);
	  if ((is_struct = Strstr(tok_key, "struct ")) || Strstr(tok_key, "union ")) {
	    tok_remove_text = NewString(is_struct ? "struct " : "union ");
	  }

	  /* be more permissive of opaque types. This is the swig way.
	     compiles will notice if these types are ultimately not
	     present. */

	  if(tok_remove_text) {
	    Replaceall(tok_name,tok_remove_text,"");
	  }
	  tok_name = strip_namespaces(tok_name);
	  Delete(tok_remove_text);
	  // Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(tok), Getline(tok), "Unable to find definition of '%s', assuming forward reference.\n", tok);

#ifdef ALLEGROCL_TYPE_DEBUG
	  Printf(stderr, "i-c-f-t: adding forward reference for unknown type '%s'. mapping: %s -> %s\n", tok, tok_key, tok_name);
#endif
	  Node *nn = NewHash();
	  Setattr(nn,"nodeType","classforward");
	  Setattr(nn,"kind","class");
	  Setattr(nn,"sym:name",tok_name);
	  Setattr(nn,"name",tok_key);
	  Setattr(nn,"allegrocl:package",current_namespace);

	  add_forward_referenced_type(nn, 0);
	  // tok_name is dangling here, unused. ouch. why?
	  Printf(ffiType, "%s", get_ffi_type(n, tok, ""), tok_name);
	}
      }
    }
  }
  return ffiType;
}

String *compose_foreign_type(Node *n, SwigType *ty, String * /*id*/ = 0) {

#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "compose_foreign_type: ENTER (%s)...\n ", ty);
  // Printf(stderr, "compose_foreign_type: ENTER (%s)(%s)...\n ", ty, (id ? id : 0));
  /* String *id_ref = SwigType_str(ty, id);
  Printf(stderr, "looking up typemap for %s, found '%s'(%p)\n",
	 id_ref, lookup_res ? Getattr(lookup_res, "code") : 0, lookup_res);
  if (lookup_res) Swig_print_node(lookup_res);
  */
#endif

  /* should we allow named lookups in the typemap here? YES! */
  /* unnamed lookups should be found in get_ffi_type, called
     by internal_compose_foreign_type(), below. */

  /* I'm reverting to 'no' for the question above. I can no longer
     remember why I needed it. If a user needed it, I'll find out
     as soon as they upgrade. Sigh. -mutandiz 9/16/2008. */

/*
  if(id && lookup_res) {
#ifdef ALLEGROCL_TYPE_DEBUG
    Printf(stderr, "compose_foreign_type: EXIT-1 (%s)\n ", Getattr(lookup_res, "code"));
#endif
    return NewString(Getattr(lookup_res, "code"));
  }
*/

  SwigType *temp = SwigType_strip_qualifiers(ty);
  String *res = internal_compose_foreign_type(n, temp);
  Delete(temp);

#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "compose_foreign_type: EXIT (%s)\n ", res);
#endif

  return res;
}

void update_package_if_needed(Node *n, File *f = f_clwrap) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "update_package: ENTER... \n");
  Printf(stderr, "  current_package = '%s'\n", current_package);
  Printf(stderr, "     node_package = '%s'\n", Getattr(n, "allegrocl:package"));
  Printf(stderr, "   node(%p) = '%s'\n", n, Getattr(n, "name"));
#endif
  String *node_package = Getattr(n, "allegrocl:package");
  if (Strcmp(current_package, node_package)) {
    String *lispy_package = listify_namespace(node_package);

    Delete(current_package);
    current_package = Copy(node_package);
    Printf(f, "\n(swig-in-package %s)\n", lispy_package);
    Delete(lispy_package);
  }
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "update_package: EXIT.\n");
#endif
}

static String *mangle_name(Node *n, char const *prefix = "ACL", String *ns = current_namespace) {
  String *suffix = Getattr(n, "sym:overname");
  String *pre_mangled_name = NewStringf("%s_%s__%s%s", prefix, ns, Getattr(n, "sym:name"), suffix);
  String *mangled_name = Swig_name_mangle(pre_mangled_name);
  Delete(pre_mangled_name);
  return mangled_name;
}

/* utilities */

/* remove a pointer from ffitype. non-destructive. 
   (* :char) ==> :char
   (* (:array :int 30)) ==> (:array :int 30) */
String *dereference_ffitype(String *ffitype) {
   char *start;
   char *temp = Char(ffitype);
   String *reduced_type = 0;

   if(temp && temp[0] == '(' && temp[1] == '*') {
      temp += 2;

      // walk past start of pointer references
      while(*temp == ' ') temp++;
      start = temp;
      // temp = Char(reduced_type);
      reduced_type = NewString(start);
      temp = Char(reduced_type);
      // walk to end of string. remove closing paren
      while(*temp != '\0') temp++;
      *(--temp) = '\0';
   }

   return reduced_type ? reduced_type : Copy(ffitype);
}

/* returns new string w/ parens stripped */
String *strip_parens(String *string) {
  string = Copy(string);
  Replaceall(string, "(", "");
  Replaceall(string, ")", "");
  return string;
}

int ALLEGROCL::validIdentifier(String *s) {
#ifdef ALLEGROCL_DEBUG
	Printf(stderr, "validIdentifier %s\n", s);
#endif

  char *c = Char(s);

  bool got_dot = false;
  bool only_dots = true;

  /* Check that s is a valid common lisp symbol. There's a lot of leeway here.
     A common lisp symbol is essentially any token that's not a number and
     does not consist of only dots. 

     We are expressly not allowing spaces in identifiers here, but spaces
     could be added via the identifier converter. */
  while (*c) {
    if (*c == '.') {
      got_dot = true;
    } else {
      only_dots = false;
    }
    if (!isgraph(*c))
      return 0;
    c++;
  }

  return (got_dot && only_dots) ? 0 : 1;
}

String *infix_to_prefix(String *val, char split_op, const String *op, String *type) {
  List *ored = Split(val, split_op, -1);

  // some float hackery
  if (((split_op == '+') || (split_op == '-')) && Len(ored) == 2 &&
      (SwigType_type(type) == T_FLOAT || SwigType_type(type) == T_DOUBLE || SwigType_type(type) == T_LONGDOUBLE)) {
    // check that we're not splitting a float
    String *possible_result = convert_literal(val, type, false);
    if (possible_result)
      return possible_result;

  }
  // try parsing the split results. if any part fails, kick out.
  bool part_failed = false;
  if (Len(ored) > 1) {
    String *result = NewStringf("(%s", op);
    for (Iterator i = First(ored); i.item; i = Next(i)) {
      String *converted = convert_literal(i.item, type);
      if (converted) {
	Printf(result, " %s", converted);
	Delete(converted);
      } else {
	part_failed = true;
	break;
      }
    }
    Printf(result, ")");
    Delete(ored);
    return part_failed ? 0 : result;
  }
  Delete(ored);
  return 0;
}

/* To be called by code generating the lisp interface
   Will return a containing the literal based on type.
   Will return null if there are problems.

   try_to_split defaults to true (see stub above).
 */
String *convert_literal(String *literal, String *type, bool try_to_split) {
  String *num_param = Copy(literal);
  String *trimmed = trim(num_param);
  String *num = strip_parens(trimmed), *res = 0;
  char *s = Char(num);

  String *ns = listify_namespace(current_namespace);

  // very basic parsing of infix expressions.
  if (try_to_split && SwigType_type(type) != T_STRING) {
    if ((res = infix_to_prefix(num, '|', "logior", type)))
      return res;
    if ((res = infix_to_prefix(num, '&', "logand", type)))
      return res;
    if ((res = infix_to_prefix(num, '^', "logxor", type)))
      return res;
    if ((res = infix_to_prefix(num, '*', "*", type)))
      return res;
    if ((res = infix_to_prefix(num, '/', "/", type)))
      return res;
    if ((res = infix_to_prefix(num, '+', "+", type)))
      return res;
    if ((res = infix_to_prefix(num, '-', "-", type)))
      return res;
    // if ((res = infix_to_prefix(num, '~', "lognot", type))) return res;
    //  if( (res = infix_to_prefix(num, '<<', "ash", type)) ) return res;  
  }

  // unary complement...
  if (s[0] == '~' && Len(num) >= 2) {
    String *id = NewString(++s);
    String *id_conv = convert_literal(id, type, false);
    Delete(id);
    if (id_conv) 
      return NewStringf("(lognot %s)", id_conv);
    s--;
  }

  if (SwigType_type(type) == T_FLOAT || SwigType_type(type) == T_DOUBLE || SwigType_type(type) == T_LONGDOUBLE) {
    // Use CL syntax for float literals 
    String *oldnum = Copy(num);

    // careful. may be a float identifier or float constant.
    char *num_start = Char(num);
    char *num_end = num_start + strlen(num_start) - 1;

    bool is_literal = isdigit(*num_start) || (*num_start == '.');

    String *lisp_exp = 0;
    if (is_literal) {
      if (*num_end == 'f' || *num_end == 'F') {
	lisp_exp = NewString("f");
      } else {
	lisp_exp = NewString("d");
      }

      if (*num_end == 'l' || *num_end == 'L' || *num_end == 'f' || *num_end == 'F') {
	*num_end = '\0';
	num_end--;
      }

      int exponents = Replaceall(num, "e", lisp_exp) + Replaceall(num, "E", lisp_exp);

      if (!exponents)
	Printf(num, "%s0", lisp_exp);

      if (exponents > 1 || (exponents + Replaceall(num, ".", ".") == 0)) {
	// Printf(stderr, "Can't parse '%s' as type '%s'.\n", oldnum, type);
	Delete(num);
	num = 0;
      }
      Delete(lisp_exp);
    } else {
      String *id = NewStringf("#.(swig-insert-id \"%s\" %s :type :constant)",
			      num, ns);
      Delete(num);
      num = id;
    }

    Delete(oldnum);
    Delete(trimmed);
    Delete(ns);
    return num;
  } else if (SwigType_type(type) == T_CHAR) {
    /* Use CL syntax for character literals */
    Delete(num);
    Delete(trimmed);
    return NewStringf("#\\%s", num_param);
  } else if (SwigType_type(type) == T_STRING) {
    /* Use CL syntax for string literals */
    Delete(num);
    Delete(trimmed);
    return NewStringf("\"%s\"", num_param);
  } else if (Len(num) >= 1 && (isdigit(s[0]) || s[0] == '+' || s[0] == '-')) {
    /* use CL syntax for numbers */
    String *oldnum = Copy(num);
    int usuffixes = Replaceall(num, "u", "") + Replaceall(num, "U", "");
    int lsuffixes = Replaceall(num, "l", "") + Replaceall(num, "L", "");
    if (usuffixes > 1 || lsuffixes > 1) {
      Printf(stderr, "Weird!! number %s looks invalid.\n", oldnum);
      SWIG_exit(EXIT_FAILURE);
    }
    s = Char(num);
    if (s[0] == '0' && Len(num) >= 2) {
      /*octal or hex */
      res = NewStringf("#%c%s", tolower(s[1]) == 'x' ? 'x' : 'o', s + 2);
      Delete(num);
    } else {
      res = num;
    }
    Delete(oldnum);
    Delete(trimmed);
    return res;
  } else if (allegrocl->validIdentifier(num)) {
    /* convert C/C++ identifiers to CL symbols */
    res = NewStringf("#.(swig-insert-id \"%s\" %s :type :constant)", num, ns);
    Delete(num);
    Delete(trimmed);
    Delete(ns);
    return res;
  } else {
    Delete(trimmed);
    return num;
  }
}


void emit_stub_class(Node *n) {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_stub_class: ENTER... '%s'(%p)\n", Getattr(n, "sym:name"), n);
  Swig_print_node(n);
#endif


  String *name = Getattr(n, "sym:name");

  if (Getattr(n, "allegrocl:synonym:already-been-stubbed"))
    return;

  String *tname = SwigType_istemplate_templateprefix(name);
  if (tname) {
    String *temp = strip_namespaces(tname);
    name = NewStringf("%s%s%s", temp, SwigType_templateargs(name), SwigType_templatesuffix(name));
    Delete(temp);
    Delete(tname);
  } else {
    name = strip_namespaces(name);
  }

  // Printf(f_clhead, ";; from emit-stub-class\n");
  update_package_if_needed(n, f_clhead);
  Printf(f_clhead, ";; class template stub.\n");
  Printf(f_clhead, "(swig-def-foreign-stub \"%s\")\n", name);

  Setattr(n, "allegrocl:synonym:already-been-stubbed", "1");

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_stub_class: EXIT\n");
#endif
}

void emit_synonym(Node *synonym) {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_synonym: ENTER... \n");
#endif

  // Printf(stderr,"in emit_synonym for %s(%p)\n", Getattr(synonym,"name"),synonym);
  int is_tempInst = !Strcmp(nodeType(synonym), "templateInst");
  String *synonym_type;

  Node *of = get_primary_synonym_of(synonym);

  if (is_tempInst) {
    // Printf(stderr, "*** using real-name '%s'\n", Getattr(synonym,"real-name"));
    synonym_type = Getattr(synonym, "real-name");
  } else {
    // Printf(stderr, "*** using name '%s'\n", Getattr(synonym,"name"));
    synonym_type = Getattr(synonym, "name");
  }

  String *synonym_ns = listify_namespace(Getattr(synonym, "allegrocl:namespace"));
  String *syn_ltype, *syn_type, *of_ltype;
  // String *of_cdeclname = Getattr(of,"allegrocl:classDeclarationName");
  String *of_ns = Getattr(of, "allegrocl:namespace");
  String *of_ns_list = listify_namespace(of_ns);
  // String *of_name = of_cdeclname ? NewStringf("struct %s", Getattr(of,"name")) : NewStringf("%s::%s", of_ns, Getattr(of,"sym:name"));
  // String *of_name = NewStringf("%s::%s", of_ns, Getattr(of,"sym:name"));
  String *of_name = namespaced_name(of, of_ns);

  if (CPlusPlus && !Strcmp(nodeType(synonym), "cdecl")) {
    String *real_name = Getattr(synonym, "real-name");
    if (!real_name)
      real_name = NewString("Unknown"); // TODO: fix
    syn_ltype = NewStringf("#.(swig-insert-id \"%s\" %s :type :class)", strip_namespaces(real_name), synonym_ns);
    syn_type = NewStringf("#.(swig-insert-id \"%s\" %s :type :type)", strip_namespaces(real_name), synonym_ns);
  } else {
    syn_ltype = lookup_defined_foreign_ltype(synonym_type);
    syn_type = lookup_defined_foreign_type(synonym_type);
  }

  of_ltype = lookup_defined_foreign_ltype(of_name);

  // Printf(stderr,";; from emit-synonym syn='%s' of_ltype='%s'\n", syn_ltype, of_ltype);
  if( of_ltype )
      Printf(f_clhead, "(swig-def-synonym-type %s\n   %s\n   %s)\n", syn_ltype, of_ltype, syn_type);

  Delete(synonym_ns);
  Delete(of_ns_list);
  Delete(of_name);

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_synonym: EXIT\n");
#endif
}

void emit_full_class(Node *n) {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_full_class: ENTER... \n");
#endif

  String *name = Getattr(n, "sym:name");
  String *kind = Getattr(n, "kind");

  // Printf(stderr,"in emit_full_class: '%s'(%p).", Getattr(n,"name"),n);
  if (Getattr(n, "allegrocl:synonym-of")) {
    // Printf(stderr,"but it's a synonym of something.\n");
    update_package_if_needed(n, f_clhead);
    emit_synonym(n);
    return;
  }
  // collect superclasses
  String *bases = Getattr(n, "bases");
  String *supers = NewString("(");
  if (bases) {
    int first = 1;
    for (Iterator i = First(bases); i.item; i = Next(i)) {
      if (!first)
	Printf(supers, " ");
      String *s = lookup_defined_foreign_ltype(Getattr(i.item, "name"));
      // String *name = Getattr(i.item,"name");
      if (s) {
	Printf(supers, "%s", s);
      } else {
#ifdef ALLEGROCL_TYPE_DEBUG
	Printf(stderr, "emit_templ_inst: did not find ltype for base class %s (%s)", Getattr(i.item, "name"), Getattr(n, "allegrocl:namespace"));
#endif
      }
    }
  } else {
    Printf(supers, "ff:foreign-pointer");
  }

  // check for "feature:aclmixins" and add those as well.
  Printf(supers, " %s)", Getattr(n,"feature:aclmixins"));

  // Walk children to generate type definition.
  String *slotdefs = NewString("   ");

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "  walking children...\n");
#endif

  Node *c;
  for (c = firstChild(n); c; c = nextSibling(c)) {
    String *storage_type = Getattr(c, "storage");
    if ((!Strcmp(nodeType(c), "cdecl") && (!storage_type || Strcmp(storage_type, "typedef")))) {
      String *access = Getattr(c, "access");

      // hack. why would decl have a value of "variableHandler" and now "0"?
      String *childDecl = Getattr(c, "decl");
      // Printf(stderr,"childDecl = '%s' (%s)\n", childDecl, Getattr(c,"view"));
      if (!childDecl || !Strcmp(childDecl, "0"))
	childDecl = NewString("");

      SwigType *childType;
      String *cname;

      // don't include types for private slots (yet). spr33959.
      if(access && Strcmp(access,"public")) {
	      childType = NewStringf("int");
	      cname = NewString("nil");
      } else {
	      childType = NewStringf("%s%s", childDecl, Getattr(c, "type"));
	      cname = Copy(Getattr(c, "name"));
      }

      if (!SwigType_isfunction(childType)) {
	// Printf(slotdefs, ";;; member functions don't appear as slots.\n ");
	// Printf(slotdefs, ";; ");
	String *ns = listify_namespace(Getattr(n, "allegrocl:package"));

#ifdef ALLEGROCL_WRAP_DEBUG
	Printf(stderr, "slot name = '%s' ns = '%s' class-of '%s' and type = '%s'\n", cname, ns, name, childType);
#endif
	Printf(slotdefs, "(#.(swig-insert-id \"%s\" %s :type :slot :class \"%s\") %s)", cname, ns, name, compose_foreign_type(n, childType));
	Delete(ns);
	if (access && Strcmp(access, "public"))
	  Printf(slotdefs, " ;; %s member", access);

	Printf(slotdefs, "\n   ");
      }
      Delete(childType);
      Delete(cname);
    }
  }

  String *ns_list = listify_namespace(Getattr(n, "allegrocl:namespace"));
  update_package_if_needed(n, f_clhead);
  Printf(f_clhead, "(swig-def-foreign-class \"%s\"\n %s\n  (:%s\n%s))\n\n", name, supers, kind, slotdefs);

  Delete(supers);
  Delete(ns_list);

  Setattr(n, "allegrocl:synonym:already-been-stubbed", "1");
#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_full_class: EXIT\n");
#endif

}

void emit_class(Node *n) {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_class: ENTER... '%s'(%p)\n", Getattr(n, "sym:name"), n);
#endif

  int is_tempInst = !Strcmp(nodeType(n), "templateInst");

  String *ns_list = listify_namespace(Getattr(n, "allegrocl:namespace"));
  String *name = Getattr(n, is_tempInst ? "real-name" : "name");

  String *tname = SwigType_istemplate_templateprefix(name);
  if (tname) {
    String *temp = strip_namespaces(tname);
    name = NewStringf("%s%s%s", temp, SwigType_templateargs(name), SwigType_templatesuffix(name));
    Delete(temp);
    Delete(tname);
  } else {
    name = strip_namespaces(name);
  }

  if (Getattr(n, "allegrocl:synonym:is-primary")) {
    // Printf(stderr,"  is primary... ");
    if (is_tempInst) {
      emit_stub_class(n);
    } else {
      emit_full_class(n);
    }
  } else {
    // Node *primary = Getattr(n,"allegrocl:synonym-of");
    Node *primary = get_primary_synonym_of(n);
    if (primary && (primary != n)) {
      // Printf(stderr,"  emitting synonym... ");
      emit_stub_class(primary);
      update_package_if_needed(n, f_clhead);
      emit_synonym(n);
    } else {
      emit_full_class(n);
    }
  }
  // Printf(stderr,"DONE\n");
  Delete(name);
  Delete(ns_list);

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_class: EXIT\n");
#endif
}

void emit_typedef(Node *n) {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_typedef: ENTER... \n");
#endif

  String *name;
  String *sym_name = Getattr(n, "sym:name");
  String *type = NewStringf("%s%s", Getattr(n, "decl"), Getattr(n, "type"));
  String *lisp_type = compose_foreign_type(n, type);
  Delete(type);
  Node *in_class = Getattr(n, "allegrocl:typedef:in-class");

  // Printf(stderr,"in emit_typedef: '%s'(%p).",Getattr(n,"name"),n);
  if (Getattr(n, "allegrocl:synonym-of")) {
    // Printf(stderr," but it's a synonym of something.\n");
    emit_synonym(n);
    return;
  }

  if (in_class) {
    String *class_name = Getattr(in_class, "name");
    String *tname = SwigType_istemplate_templateprefix(class_name);
    if (tname) {
      String *temp = strip_namespaces(tname);
      class_name = NewStringf("%s%s%s", temp, SwigType_templateargs(class_name), SwigType_templatesuffix(class_name));
      Delete(temp);
      Delete(tname);
    }

    name = NewStringf("%s__%s", class_name, sym_name);
    Setattr(n, "allegrocl:in-class", in_class);
  } else {
    name = sym_name ? Copy(sym_name) : Copy(Getattr(n, "name"));
  }

  // leave these in for now. might want to change these to def-foreign-class at some point.
//  Printf(f_clhead, ";; %s\n", SwigType_typedef_resolve_all(lisp_type));
  Printf(f_clhead, "(swig-def-foreign-type \"%s\"\n  %s)\n", name, lisp_type);

  Delete(name);

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_typedef: EXIT\n");
#endif
}

void emit_enum_type_no_wrap(Node *n) {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_enum_type_no_wrap: ENTER... \n");
#endif

  String *unnamed = Getattr(n, "unnamed");
  String *name;
  //  SwigType *enumtype;

  name = unnamed ? Getattr(n, "allegrocl:name") : Getattr(n, "sym:name");
  SwigType *tmp = NewStringf("enum %s", unnamed ? unnamed : name);

  Node *node = NewHash();
  Setattr(node, "type", tmp);
  Setfile(node, Getfile(n));
  Setline(node, Getline(n));
  const String *enumtype = Swig_typemap_lookup("ffitype", node, "", 0);
  Delete(node);

  Delete(tmp);

  if (name) {
    String *ns = listify_namespace(current_namespace);

    Printf(f_clhead, "(swig-def-foreign-type \"%s\" %s)\n", name, enumtype);
    Delete(ns);

    // walk children.
    Node *c;
    for (c = firstChild(n); c; c = nextSibling(c)) {
      if (!Getattr(c, "error")) {
	String *val = Getattr(c, "enumvalue");
	if (!val)
	  val = Getattr(c, "enumvalueex");
	String *converted_val = convert_literal(val, Getattr(c, "type"));
	String *valname = Getattr(c, "sym:name");

	if (converted_val) {
	  Printf(f_clhead, "(swig-defconstant \"%s\" %s)\n", valname, converted_val);
	  Delete(converted_val);
	} else {
	  Swig_warning(WARN_LANG_DISCARD_CONST, Getfile(n), Getline(n), "Unable to parse enum value '%s'. Setting to NIL\n", val);
	  Printf(f_clhead, "(swig-defconstant \"%s\" nil #| %s |#)\n", valname, val);
	}
      }
    }
  }
  Printf(f_clhead, "\n");

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_enum_type_no_wrap: EXIT\n");
#endif

}

void emit_enum_type(Node *n) {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_enum_type: ENTER... \n");
#endif

  if (!Generate_Wrapper) {
    emit_enum_type_no_wrap(n);
    return;
  }

  String *unnamed = Getattr(n, "unnamed");
  String *name;
  // SwigType *enumtype;

  name = unnamed ? Getattr(n, "allegrocl:name") : Getattr(n, "sym:name");
  SwigType *tmp = NewStringf("enum %s", unnamed ? unnamed : name);

  Node *node = NewHash();
  Setattr(node, "type", tmp);
  Setfile(node, Getfile(n));
  Setline(node, Getline(n));
  const String *enumtype = Swig_typemap_lookup("ffitype", node, "", 0);
  Delete(node);

  Delete(tmp);

  if (name) {
    String *ns = listify_namespace(current_namespace);

    Printf(f_clhead, "(swig-def-foreign-type \"%s\" %s)\n", name, enumtype);
    Delete(ns);

    // walk children.
    Node *c;
    for(c = firstChild(n); c; c=nextSibling(c)) {
      String *mangled_name = mangle_name(c, "ACL_ENUM", Getattr(c,"allegrocl:package"));
      Printf(f_clhead, "(swig-defvar \"%s\" \"%s\" :type :constant :ftype :signed-long)\n", Getattr(c, "sym:name"), mangled_name);
      Delete(mangled_name);
    }
  }
#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_enum_type: EXIT\n");
#endif

}

void emit_default_linked_type(Node *n) {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_default_linked_type: ENTER... \n");
#endif

  // catchall for non class types.
  if (!Strcmp(nodeType(n), "classforward")) {
    Printf(f_clhead, ";; forward referenced stub.\n");
    Printf(f_clhead, "(swig-def-foreign-class \"%s\" (ff:foreign-pointer) (:class ))\n\n", Getattr(n, "sym:name"));
  } else if (!Strcmp(nodeType(n), "enum")) {
    emit_enum_type(n);
  } else {
    Printf(stderr, "Don't know how to emit node type '%s' named '%s'\n", nodeType(n), Getattr(n, "name"));
  }

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_default_linked_type: EXIT\n");
#endif

}

void dump_linked_types(File *f) {
  Node *n = first_linked_type;
  int i = 0;
  while (n) {
    Printf(f, "%d: (%p) node '%s' name '%s'\n", i++, n, nodeType(n), Getattr(n, "sym:name"));

    Node *t = Getattr(n, "allegrocl:synonym-of");
    if (t)
      Printf(f, "     synonym-of %s(%p)\n", Getattr(t, "name"), t);
    n = Getattr(n, "allegrocl:next_linked_type");
  }
}

void emit_linked_types() {

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_linked_types: ENTER... ");
#endif

  Node *n = first_linked_type;

  while (n) {
    String *node_type = nodeType(n);

    // Printf(stderr,"emitting node %s(%p) of type %s.", Getattr(n,"name"),n, nodeType(n));
    if (!Strcmp(node_type, "class") || !Strcmp(node_type, "templateInst")) {
      // may need to emit a stub, so it will update the package itself.
      // Printf(stderr," Passing to emit_class.");
      emit_class(n);
    } else if (!Strcmp(nodeType(n), "cdecl")) {
      // Printf(stderr," Passing to emit_typedef.");
      update_package_if_needed(n, f_clhead);
      emit_typedef(n);
    } else {
      // Printf(stderr," Passing to default_emitter.");
      update_package_if_needed(n, f_clhead);
      emit_default_linked_type(n);
    }

    n = Getattr(n, "allegrocl:next_linked_type");
    // Printf(stderr,"returned.\n");
  }

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_linked_types: EXIT\n");
#endif
}

extern "C" Language *swig_allegrocl(void) {
  return (allegrocl = new ALLEGROCL());
}

void ALLEGROCL::main(int argc, char *argv[]) {
  int i;

  Preprocessor_define("SWIGALLEGROCL 1", 0);
  SWIG_library_directory("allegrocl");
  SWIG_config_file("allegrocl.swg");

  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-identifier-converter")) {
      char *conv = argv[i + 1];

      if (!conv)
	Swig_arg_error();

      Swig_mark_arg(i);
      Swig_mark_arg(i + 1);
      i++;

      /* check for built-ins */
      if (!strcmp(conv, "lispify")) {
	Delete(identifier_converter);
	identifier_converter = NewString("identifier-convert-lispify");
      } else if (!strcmp(conv, "null")) {
	Delete(identifier_converter);
	identifier_converter = NewString("identifier-convert-null");
      } else {
	/* Must be user defined */
	Delete(identifier_converter);
	identifier_converter = NewString(conv);
      }
    } else if (!strcmp(argv[i], "-cwrap")) {
      CWrap = true;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i], "-nocwrap")) {
      CWrap = false;
      Swig_mark_arg(i);
    } else if (!strcmp(argv[i], "-isolate")) {
      unique_swig_package = true;
      Swig_mark_arg(i);
    }

    if (!strcmp(argv[i], "-help")) {
      Printf(stdout, "%s\n", usage);
    }

  }

  allow_overloading();
}

int ALLEGROCL::top(Node *n) {
  module_name = Getattr(n, "name");
  String *cxx_filename = Getattr(n, "outfile");
  String *cl_filename = NewString("");

  swig_package = unique_swig_package ? NewStringf("swig.%s", module_name) : NewString("swig");

  Printf(cl_filename, "%s%s.cl", SWIG_output_directory(), module_name);

  f_cl = NewFile(cl_filename, "w", SWIG_output_files());
  if (!f_cl) {
    Printf(stderr, "Unable to open %s for writing\n", cl_filename);
    SWIG_exit(EXIT_FAILURE);
  }

  Generate_Wrapper = CPlusPlus || CWrap;

  if (Generate_Wrapper) {
    f_begin = NewFile(cxx_filename, "w", SWIG_output_files());
    if (!f_begin) {
      Delete(f_cl);
      Printf(stderr, "Unable to open %s for writing\n", cxx_filename);
      SWIG_exit(EXIT_FAILURE);
    }
  } else
    f_begin = NewString("");

  f_runtime = NewString("");
  f_cxx_header = f_runtime;
  f_cxx_wrapper = NewString("");

  Swig_register_filebyname("header", f_cxx_header);
  Swig_register_filebyname("wrapper", f_cxx_wrapper);
  Swig_register_filebyname("begin", f_begin);
  Swig_register_filebyname("runtime", f_runtime);
  Swig_register_filebyname("lisp", f_clwrap);
  Swig_register_filebyname("lisphead", f_cl);

  Swig_banner(f_begin);

  Printf(f_runtime, "\n\n#ifndef SWIGALLEGROCL\n#define SWIGALLEGROCL\n#endif\n\n");

  Swig_banner_target_lang(f_cl, ";;");

  Printf(f_cl, "\n"
	 "(defpackage :%s\n"
	 "  (:use :common-lisp :ff :excl)\n"
	 "  (:export #:*swig-identifier-converter* #:*swig-module-name*\n"
	 "           #:*void* #:*swig-export-list*))\n"
	 "(in-package :%s)\n\n"
	 "(eval-when (:compile-toplevel :load-toplevel :execute)\n"
	 "  (defparameter *swig-identifier-converter* '%s)\n"
	 "  (defparameter *swig-module-name* :%s))\n\n", swig_package, swig_package, identifier_converter, module_name);
  Printf(f_cl, "(defpackage :%s\n" "  (:use :common-lisp :%s :ff :excl))\n\n", module_name, swig_package);

  Printf(f_clhead, "(in-package :%s)\n", module_name);

  Language::top(n);

#ifdef ALLEGROCL_TYPE_DEBUG
  dump_linked_types(stderr);
#endif
  emit_linked_types();

  Printf(f_clwrap, "\n(cl::in-package :%s)\n", swig_package);
  Printf(f_clwrap, "\n(macrolet ((swig-do-export ()\n");
  Printf(f_clwrap, "                 `(dolist (s ',*swig-export-list*)\n");
  Printf(f_clwrap, "                    (apply #'export s))))\n");
  Printf(f_clwrap, "   (swig-do-export))\n");
  Printf(f_clwrap, "\n(setq *swig-export-list* nil)\n");

  Printf(f_cl, "%s\n", f_clhead);
  Printf(f_cl, "%s\n", f_clwrap);

  Delete(f_cl);
  Delete(f_clhead);
  Delete(f_clwrap);

  Dump(f_runtime, f_begin);
  Printf(f_begin, "%s\n", f_cxx_wrapper);

  Delete(f_runtime);
  Delete(f_begin);
  Delete(f_cxx_wrapper);

  // Swig_print_tree(n);

  return SWIG_OK;
}

/* very shamelessly 'borrowed' from overload.cxx, which
   keeps the below Swig_overload_rank() code to itself.
   We don't need a dispatch function in the C++ wrapper
   code; we want it over on the lisp side. */

#define Swig_overload_rank Allegrocl_swig_overload_rank

#define MAX_OVERLOAD 256

/* Overload "argc" and "argv" */
// String *argv_template_string;
// String *argc_template_string;

struct Overloaded {
  Node *n;			/* Node                               */
  int argc;			/* Argument count                     */
  ParmList *parms;		/* Parameters used for overload check */
  int error;			/* Ambiguity error                    */
};

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

static List *Swig_overload_rank(Node *n, bool script_lang_wrapping) {
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
      nodes[nnodes].n = c;
      nodes[nnodes].parms = Getattr(c, "wrap:parms");
      nodes[nnodes].argc = emit_num_required(nodes[nnodes].parms);
      nodes[nnodes].error = 0;
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
    for (i = 0; i < nnodes; i++) {
      if (nodes[i].error)
	Setattr(nodes[i].n, "overload:ignore", "1");
      Append(result, nodes[i].n);
      //      Printf(stdout,"[ %d ] %s\n", i, ParmList_errorstr(nodes[i].parms));
      //      Swig_print_node(nodes[i].n);
    }
  }
  return result;
}

/* end shameless borrowing */

int any_varargs(ParmList *pl) {
  Parm *p;

  for (p = pl; p; p = nextSibling(p)) {
    if (SwigType_isvarargs(Getattr(p, "type")))
      return 1;
  }

  return 0;
}

String *get_lisp_type(Node *n, SwigType *ty, const_String_or_char_ptr name) {
  Node *node = NewHash();
  Setattr(node, "type", ty);
  Setattr(node, "name", name);
  Setfile(node, Getfile(n));
  Setline(node, Getline(n));
  const String *tm = Swig_typemap_lookup("lisptype", node, "", 0);
  Delete(node);

  return tm ? NewString(tm) : NewString("");
}

Node *parent_node_skipping_extends(Node *n) {
  Node *result = n;
  do {
    result = parentNode(result);
  }
  while (Cmp("extend", nodeType(result)) == 0);
  return result;
}

/* -----------------------------------------------------------------------------
 * emit_num_lin_arguments()
 *
 * Calculate the total number of arguments.   This function is safe for use
 * with multi-argument typemaps which may change the number of arguments in
 * strange ways.
 * ----------------------------------------------------------------------------- */

int emit_num_lin_arguments(ParmList *parms) {
  Parm *p = parms;
  int nargs = 0;

  while (p) {
    // Printf(stderr,"enla: '%s' lin='%p' numinputs='%s'\n", Getattr(p,"name"), Getattr(p,"tmap:lin"), Getattr(p,"tmap:lin:numinputs"));
    if (Getattr(p, "tmap:lin")) {
      nargs += GetInt(p, "tmap:lin:numinputs");
      p = Getattr(p, "tmap:lin:next");
    } else {
      p = nextSibling(p);
    }
  }

  /* DB 04/02/2003: Not sure this is necessary with tmap:in:numinputs */
  /*
     if (parms && (p = Getattr(parms,"emit:varargs"))) {
     if (!nextSibling(p)) {
     nargs--;
     }
     }
   */
  return nargs;
}

String *id_converter_type(SwigType const *type) {
  SwigType *t = Copy(type);
  String *result = 0;

  if (SwigType_ispointer(t)) {
    SwigType_pop(t);
    String *pointee = id_converter_type(t);
    result = NewStringf("(:* %s)", pointee);
    Delete(pointee);
  } else if (SwigType_ismemberpointer(t)) {
    String *klass = SwigType_parm(t);
    SwigType_pop(t);
    String *member = id_converter_type(t);
    result = NewStringf("(:member \"%s\" %s)", klass, member);
    Delete(klass);
    Delete(member);
  } else if (SwigType_isreference(t)) {
    SwigType_pop(t);
    String *referencee = id_converter_type(t);
    result = NewStringf("(:& %s)", referencee);
    Delete(referencee);
  } else if (SwigType_isarray(t)) {
    String *size = SwigType_parm(t);
    SwigType_pop(t);
    String *element_type = id_converter_type(t);
    result = NewStringf("(:array %s \"%s\")", element_type, size);
    Delete(size);
    Delete(element_type);
  } else if (SwigType_isfunction(t)) {
    result = NewString("(:function (");
    String *parmlist_str = SwigType_parm(t);
    List *parms = SwigType_parmlist(parmlist_str);

    for (Iterator i = First(parms); i.item;) {
      String *parm = id_converter_type((SwigType *) i.item);
      Printf(result, "%s", parm);
      i = Next(i);
      if (i.item)
	Printf(result, " ");
      Delete(parm);
    }
    SwigType_pop(t);
    String *ret = id_converter_type(t);
    Printf(result, ") %s)", ret);

    Delete(parmlist_str);
    Delete(parms);
    Delete(ret);
  } else if (SwigType_isqualifier(t)) {
    result = NewString("(:qualified (");
    String *qualifiers_str = Copy(SwigType_parm(t));	// ?!
    // Replaceall below SEGVs if we don't put the Copy here...
    SwigType_pop(t);
    String *qualifiee = id_converter_type(t);

    Replaceall(qualifiers_str, " ", " :");
    if (Len(qualifiers_str) > 0)
      Printf(result, ":");
    Printf(result, "%s) %s)", qualifiers_str, qualifiee);

    Delete(qualifiers_str);
    Delete(qualifiee);
  } else if (SwigType_istemplate(t)) {
    result = NewStringf("(:template \"%s\")", t);
  } else {			/* if (SwigType_issimple(t)) */

    if (Strstr(Char(t), "::")) {
      result = listify_namespace(t);
    } else {
      result = NewStringf("\"%s\"", t);
    }
  }

  Delete(t);
  return result;
}

static ParmList *parmlist_with_names(ParmList *pl) {
  ParmList *pl2 = CopyParmList(pl);
  for (Parm *p = pl, *p2 = pl2; p2; p = nextSibling(p), p2 = nextSibling(p2)) {
    if (!Getattr(p2, "name"))
      Setattr(p2, "name", Getattr(p2, "lname"));
    Setattr(p2, "name", strip_namespaces(Getattr(p2, "name")));
    Setattr(p2, "tmap:ctype", Getattr(p, "tmap:ctype"));

    String *temp = Getattr(p, "tmap:lin");
    if (temp) {
      Setattr(p2, "tmap:lin", temp);
      Setattr(p2, "tmap:lin:next", Getattr(p, "tmap:lin:next"));
    }
  }
  return pl2;
}

static String *parmlist_str_id_converter(ParmList *pl) {
  String *result = NewString("");
  for (Parm *p = pl; p;) {
    String *lispy_type = id_converter_type(Getattr(p, "type"));
    Printf(result, "(\"%s\" %s)", Getattr(p, "name"), lispy_type);
    Delete(lispy_type);
    if ((p = nextSibling(p)))
      Printf(result, " ");
  }
  return result;
}

String *collect_others_args(Node *overload) {
  String *overloaded_from = Getattr(overload, "sym:overloaded");
  String *others_args = NewString("");
  int first_overload = 1;

  for (Node *overload2 = overloaded_from; overload2; overload2 = Getattr(overload2, "sym:nextSibling")) {
    if (overload2 == overload || GetInt(overload2, "overload:ignore"))
      continue;

    ParmList *opl = parmlist_with_names(Getattr(overload2, "wrap:parms"));
    String *args = parmlist_str_id_converter(opl);
    if (!first_overload)
      Printf(others_args, "\n                           ");
    Printf(others_args, "(%s)", args);
    Delete(args);
    Delete(opl);
    first_overload = 0;
  }
  return others_args;
}

struct IDargs {
  String *name;
  String *type;
  String *klass;
  String *arity;

  IDargs():name(0), type(0), klass(0), arity(0) {
  }

  String *full_quoted_str() {
    String *result = no_others_quoted_str();
    if (arity)
      Printf(result, " :arity %s", arity);
    return result;
  }

  String *no_others_quoted_str() {
    String *result = NewString("");
    Printf(result, "\"%s\" :type :%s", name, type);
    if (klass)
      Printf(result, " :class \"%s\"", klass);
    return result;
  }

  String *noname_str(bool include_class = true) {
    String *result = NewString("");
    Printf(result, " :type :%s", type);
    if (klass && include_class)
      Printf(result, " :class \"%s\"", klass);
    if (arity)
      Printf(result, " :arity %s", arity);
    return result;
  }

  String *noname_no_others_str(bool include_class = true) {
    String *result = NewString("");
    Printf(result, " :type :%s", type);
    if (klass && include_class)
      Printf(result, " :class \"%s\"", klass);
    return result;
  }
};
IDargs *id_converter_arguments(Node *n) {
  IDargs *result = (IDargs *) GetVoid(n, "allegrocl:id-converter-args");
  if (!result)
    result = new IDargs;

  // Base name
  if (!result->name) {
    result->name = Getattr(n, "allegrocl:old-sym:name");
    if (!result->name)
      result->name = Getattr(n, "sym:name");
    result->name = Copy(result->name);
  }
  // :type
  if (result->type)
    Delete(result->type);
  if (!Getattr(n, "allegrocl:kind"))
    Setattr(n, "allegrocl:kind", "function");
  if (Strstr(Getattr(n, "name"), "operator "))
    Replaceall(Getattr(n, "allegrocl:kind"), "function", "operator");
  if (Strstr(Getattr(n, "allegrocl:kind"), "variable")) {
    int name_end = Len(Getattr(n, "sym:name")) - 4;
    char *str = Char(Getattr(n, "sym:name"));
    String *get_set = NewString(str + name_end + 1);
    result->type = Copy(Getattr(n, "allegrocl:kind"));
    Replaceall(result->type, "variable", "");
    Printf(result->type, "%ster", get_set);
    Delete(get_set);
  } else {
    result->type = Copy(Getattr(n, "allegrocl:kind"));
  }

  // :class
  if (Strstr(result->type, "member ")) {
    Replaceall(result->type, "member ", "");
    if (!result->klass) {
      result->klass = Copy(Getattr(parent_node_skipping_extends(n), "sym:name"));
    }
  }
  // :arity
  if (Getattr(n, "sym:overloaded")) {
    if (result->arity)
      Delete(result->arity);
    result->arity = NewStringf("%d",
			       // emit_num_arguments(Getattr(n, "wrap:parms")));
			       emit_num_lin_arguments(Getattr(n, "wrap:parms")));
    // Printf(stderr, "got arity of '%s' node '%s' '%p'\n", result->arity, Getattr(n,"name"), Getattr(n,"wrap:parms"));
  }

  SetVoid(n, "allegrocl:id-converter-args", result);
  return result;
}

int ALLEGROCL::emit_buffered_defuns(Node *n) {

  Node *overloaded_from = Getattr(n, "sym:overloaded");

  String *wrap;

  if (!overloaded_from) {
    wrap = Getattr(n, "allegrocl:lisp-wrap");

    Printf(f_clwrap, "%s\n", wrap);
    Delattr(n, "allegrocl:lisp-wrap");
    Delete(wrap);
  } else {
    for (Node *overload = overloaded_from; overload; overload = Getattr(overload, "sym:nextSibling")) {
      String *others_args = collect_others_args(overload);
      wrap = Getattr(overload, "allegrocl:lisp-wrap");

      Replaceall(wrap, "@@OTHERS-ARGS-GO-HERE@@", others_args);
//        IDargs* id_args = id_converter_arguments(overload);
//        Replaceall(id_args->others_args, "@@OTHERS-ARGS-GO-HERE@@", others_args);

      if (!GetInt(overload, "overload:ignore"))
	Printf(f_clwrap, "%s", wrap);

      Delattr(overload, "allegrocl:lisp-wrap");
      Delete(wrap);
    }
  }
  return SWIG_OK;
}

String *dispatching_type(Node *n, Parm *p) {
  String *result = 0;

  String *parsed = Getattr(p, "type");	//Swig_cparse_type(Getattr(p,"tmap:ctype"));
  String *cl_t = SwigType_typedef_resolve_all(parsed);

  Node *node = NewHash();
  Setattr(node, "type", parsed);
  Setfile(node, Getfile(n));
  Setline(node, Getline(n));
  const String *tm = Swig_typemap_lookup("lispclass", node, Getattr(p, "name"), 0);
  Delete(node);

  if (tm) {
    result = Copy(tm);
  } else {
    String *lookup_type = class_from_class_or_class_ref(parsed);
    if (lookup_type)
      result = lookup_defined_foreign_ltype(lookup_type);
  }

  //  if (!result && SwigType_ispointer(cl_t)) {
  //    SwigType_pop(cl_t);
  //    result = lookup_defined_foreign_ltype(cl_t);
  //  }

  if (!result)
    result = NewStringf("ff:foreign-pointer");

  // Delete(parsed);
  Delete(cl_t);
  return result;
}

int ALLEGROCL::emit_dispatch_defun(Node *n) {
#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_dispatch_defun: ENTER... ");
#endif
  List *overloads = Swig_overload_rank(n, true);

  // Printf(stderr,"\ndispatch node=%p\n\n", n);
  // Swig_print_node(n);

  Node *overloaded_from = Getattr(n,"sym:overloaded");
  bool include_class = Getattr(overloaded_from, "allegrocl:dispatcher:include-class") ? true : false;
  String *id_args = id_converter_arguments(n)->noname_no_others_str(include_class);
  Printf(f_clwrap, "(swig-dispatcher (\"%s\" %s :arities (", Getattr(overloaded_from, "allegrocl:dispatcher:name"), id_args);

  Delattr(overloaded_from, "allegrocl:dispatcher:include-class");
  Delattr(overloaded_from, "allegrocl:dispatcher:name");

  int last_arity = -1;
  for (Iterator i = First(overloads); i.item; i = Next(i)) {
    int arity = emit_num_lin_arguments(Getattr(i.item, "wrap:parms"));
    if (arity == last_arity)
      continue;

    Printf(f_clwrap, "%s%d", last_arity == -1 ? "" : " ", arity);

    last_arity = arity;
  }
  Printf(f_clwrap, ")))\n");

  Delete(id_args);
  Delete(overloads);

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_dispatch_defun: EXIT\n");
#endif

  return SWIG_OK;
}

int ALLEGROCL::emit_defun(Node *n, File *fcl) {
#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_defun: ENTER... ");
#endif

  // avoid name conflicts between smart pointer wrappers and the wrappers for the
  // actual class.
  bool smartmemberwrapper = (!Cmp(Getattr(n, "view"), "memberfunctionHandler") &&
			     Getattr(n,"allocate:smartpointeraccess"));

#ifdef ALLEGROCL_DEBUG
  int auto_generated = Cmp(Getattr(n, "view"), "globalfunctionHandler");
  Printf(stderr, "%s%sfunction %s%s%s\n", auto_generated ? "> " : "", Getattr(n, "sym:overloaded")
	 ? "overloaded " : "", current_namespace, (current_namespace) > 0 ? "::" : "", Getattr(n, "sym:name"));
  Printf(stderr, "  (view: %s)\n", Getattr(n, "view"));
  Swig_print_node(n);
#endif


  String *funcname = Getattr(n, "allegrocl:old-sym:name");
  if (smartmemberwrapper || !funcname)
    funcname = Getattr(n, "sym:name");

  String *mangled_name = Getattr(n, "wrap:name");
  ParmList *pl = parmlist_with_names(Getattr(n, "wrap:parms"));

  // attach typemap info.
  Wrapper *wrap = NewWrapper();
  Swig_typemap_attach_parms("lin", pl, wrap);
  // Swig_typemap_attach_parms("ffitype", pl, wrap);
  Swig_typemap_lookup("lout", n, Swig_cresult_name(), 0);

  SwigType *result_type = Swig_cparse_type(Getattr(n, "tmap:ctype"));
  // prime the pump, with support for OUTPUT, INOUT typemaps.
  Printf(wrap->code,
	 "(cl::let ((ACL_ffresult %s:*void*)\n        ACL_result)\n  $body\n  (cl::if (cl::eq ACL_ffresult %s:*void*)\n    (cl::values-list ACL_result)\n   (cl::values-list (cl::cons ACL_ffresult ACL_result))))",
	 swig_package, swig_package);

  Parm *p;
  int largnum = 0, argnum = 0, first = 1;
  // int varargs=0;
  if (Generate_Wrapper) {
    String *extra_parms = id_converter_arguments(n)->noname_str(smartmemberwrapper ? false : true);
    Node *overloaded_from = Getattr(n,"sym:overloaded");
    if (overloaded_from) {
      if(!GetFlag(overloaded_from,"allegrocl:dispatcher:name")) {
        Setattr(overloaded_from,"allegrocl:dispatcher:name",funcname);
	Setattr(overloaded_from,"allegrocl:dispatcher:include-class", smartmemberwrapper ? 0 : "1");
	// Printf(stderr, "   set a:d:name='%s', a:d:i-c='%s'\n", Getattr(n,"allegrocl:dispatcher:name"), Getattr(n,"allegrocl:dispatcher:include-class"));
      }
      Printf(fcl, "(swig-defmethod (\"%s\" \"%s\"%s)\n", funcname, mangled_name, extra_parms);
    } else
      Printf(fcl, "(swig-defun (\"%s\" \"%s\"%s)\n", funcname, mangled_name, extra_parms);
    Delete(extra_parms);
  }
  // Just C
  else {
    Printf(fcl, "(swig-defun (\"%s\" \"%s\")\n", funcname, Generate_Wrapper ? mangled_name : funcname);
  }

  //////////////////////////////////////
  // Lisp foreign call parameter list //
  //////////////////////////////////////
  Printf(fcl, "  (");

  /* Special cases */

  if (ParmList_len(pl) == 0) {
    Printf(fcl, ":void");
/*  } else if (any_varargs(pl)) {
    Printf(fcl, "#| varargs |#");
    varargs=1; */
  } else {
    String *largs = NewString("");

    for (p = pl; p; p = nextSibling(p), argnum++, largnum++) {
      // SwigType *argtype=Getattr(p, "type");
      SwigType *argtype = Swig_cparse_type(Getattr(p, "tmap:ctype"));
      SwigType *parmtype = Getattr(p,"type");

      if (!first) {
	Printf(fcl, "\n   ");
      }

      /* by default, skip varargs */
      if (!SwigType_isvarargs(parmtype)) {
	String *argname = NewStringf("PARM%d_%s", largnum, Getattr(p, "name"));

	// Printf(stderr,"%s\n", Getattr(p,"tmap:lin"));
	String *ffitype = compose_foreign_type(n, argtype, Getattr(p,"name"));
	String *deref_ffitype = dereference_ffitype(ffitype);
	String *lisptype = get_lisp_type(n, parmtype, Getattr(p, "name"));

#ifdef ALLEGROCL_DEBUG
	Printf(stderr, "lisptype of '%s' '%s' = '%s'\n", parmtype,
	       Getattr(p, "name"), lisptype);
#endif

	// while we're walking the parameters, generating LIN
	// wrapper code...
	Setattr(p, "lname", NewStringf("SWIG_arg%d", largnum));

	String *parm_code = Getattr(p, "tmap:lin");
	if (parm_code) {
	  String *lname = Getattr(p, "lname");

	  Printf(largs, " %s", lname);
	  Replaceall(parm_code, "$in_fftype", ffitype); // must come before $in
	  Replaceall(parm_code, "$in", argname);
	  Replaceall(parm_code, "$out", lname);
	  Replaceall(parm_code, "$*in_fftype", deref_ffitype);
	  Replaceall(wrap->code, "$body", parm_code);
	}

	String *dispatchtype = Getattr(n, "sym:overloaded") ? dispatching_type(n, p) : NewString("");

	// if this parameter has been removed from the C/++ wrapper
	// it shouldn't be in the lisp wrapper either.
	if (!checkAttribute(p, "tmap:in:numinputs", "0")) {
	  Printf(fcl, "(%s %s %s %s %s)",
		 // parms in the ff wrapper, but not in the lisp wrapper.
		 (checkAttribute(p, "tmap:lin:numinputs", "0") ? ":p-" : ":p+"), argname, dispatchtype, ffitype, lisptype);

	  first = 0;
	}

	Delete(argname);
	Delete(ffitype);
	Delete(deref_ffitype);
	Delete(lisptype);
      }
    }

    Printf(wrap->locals, "%s", largs);
  }

  String *lout = Getattr(n, "tmap:lout");
  Replaceall(lout, "$owner", GetFlag(n, "feature:new") ? "t" : "nil");

  Replaceall(wrap->code, "$body", lout);
  // $lclass handling.
  String *lclass = (String *) 0;
  SwigType *parsed = Swig_cparse_type(Getattr(n, "tmap:ctype"));
  //  SwigType *cl_t = SwigType_typedef_resolve_all(parsed);
  SwigType *cl_t = class_from_class_or_class_ref(parsed);
  String *out_ffitype = compose_foreign_type(n, parsed);
  String *deref_out_ffitype;
  String *out_temp = Copy(parsed);

  if (SwigType_ispointer(out_temp)) {
    SwigType_pop(out_temp);
    deref_out_ffitype = compose_foreign_type(n, out_temp);
  } else {
    deref_out_ffitype = Copy(out_ffitype);
  }

  Delete(out_temp);

  Delete(parsed);

  if (cl_t) {
    lclass = lookup_defined_foreign_ltype(cl_t);
  }

  int ff_foreign_ptr = 0;
  if (!lclass) {
    ff_foreign_ptr = 1;
    lclass = NewStringf("ff:foreign-pointer");
  }
#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "for output wrapping %s: type=%s, ctype=%s\n", Getattr(n, "name"),
	 Getattr(n, "type"), Swig_cparse_type(Getattr(n, "tmap:ctype")));
#endif

  if (lclass)
    Replaceall(wrap->code, "$lclass", lclass);
  if (out_ffitype)
    Replaceall(wrap->code, "$out_fftype", out_ffitype);
  if (deref_out_ffitype)
    Replaceall(wrap->code, "$*out_fftype", deref_out_ffitype);

  Replaceall(wrap->code, "$body", NewStringf("(swig-ff-call%s)", wrap->locals));
  String *ldestructor = Copy(lclass);
  if (ff_foreign_ptr)
    Replaceall(ldestructor, ldestructor, "cl::identity");
  else
    Replaceall(ldestructor, ":type :class", ":type :destructor");
  Replaceall(wrap->code, "$ldestructor", ldestructor);
  Delete(ldestructor);

  Printf(fcl, ")\n");		/* finish arg list */

  /////////////////////////////////////////////////////
  // Lisp foreign call return type and optimizations //
  /////////////////////////////////////////////////////
  Printf(fcl, "  (:returning (%s %s)", compose_foreign_type(n, result_type), get_lisp_type(n, Getattr(n, "type"), Swig_cresult_name()));

  for (Iterator option = First(n); option.item; option = Next(option)) {
    if (Strncmp("feature:ffargs:", option.key, 15))
      continue;
    String *option_val = option.item;
    String *option_name = NewString(Char(option.key) + 14);
    Replaceall(option_name, "_", "-");

    // TODO: varargs vs call-direct ?
    Printf(fcl, "\n   %s %s", option_name, option_val);

    Delete(option_name);
  }

  Printf(fcl, ")\n  %s)\n\n", wrap->code);
  // Wrapper_print(wrap, stderr);

  Delete(result_type);
  Delete(mangled_name);
  Delete(pl);
  DelWrapper(wrap);

#ifdef ALLEGROCL_WRAP_DEBUG
  Printf(stderr, "emit_defun: EXIT\n");
#endif

  return SWIG_OK;
}

int ALLEGROCL::functionWrapper(Node *n) {
#ifdef ALLEGROCL_DEBUG
	Printf(stderr, "functionWrapper %s\n", Getattr(n,"name"));
#endif


  ParmList *parms = CopyParmList(Getattr(n, "parms"));
  Wrapper *f = NewWrapper();
  SwigType *t = Getattr(n, "type");
  String *name = Getattr(n, "name");

  String *raw_return_type = Swig_typemap_lookup("ctype", n, "", 0);
  SwigType *return_type = Swig_cparse_type(raw_return_type);
  SwigType *resolved = SwigType_typedef_resolve_all(return_type);
  int is_void_return = (Cmp(resolved, "void") == 0);

  Delete(resolved);

  if (!is_void_return) {
     String *lresult_init =
	     NewStringf("= (%s)0",
			SwigType_str(SwigType_strip_qualifiers(return_type),0));
     Wrapper_add_localv(f, "lresult",
			SwigType_lstr(SwigType_ltype(return_type), "lresult"),
			lresult_init, NIL);
     Delete(lresult_init);
  }
  // Emit all of the local variables for holding arguments.
  emit_parameter_variables(parms, f);

  // Attach the standard typemaps 
  Swig_typemap_attach_parms("ctype", parms, f);
  Swig_typemap_attach_parms("lin", parms, f);
  emit_attach_parmmaps(parms, f);

  String *mangled = mangle_name(n);
  Node *overloaded = Getattr(n, "sym:overloaded");

  // Parameter overloading
  Setattr(n, "wrap:parms", parms);
  Setattr(n, "wrap:name", mangled);

  if (overloaded) {
    // emit warnings when overloading is impossible on the lisp side.
    // basically Swig_overload_check(n), but with script_lang_wrapping
    // set to true.
    Delete(Swig_overload_rank(n, true));
    if (Getattr(n, "overload:ignore")) {
      // if we're the last overload, make sure to force the emit
      // of the rest of the overloads before we leave.
      // Printf(stderr, "ignored overload %s(%p)\n", name, Getattr(n, "sym:nextSibling"));
      if (!Getattr(n, "sym:nextSibling")) {
	update_package_if_needed(n);
	emit_buffered_defuns(n);
	emit_dispatch_defun(n);
      }
      DelWrapper(f);
      return SWIG_OK;
    }
  }
  // Get number of required and total arguments 
  int num_arguments = emit_num_arguments(parms);
  int gencomma = 0;

#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "Walking parameters for %s '%s'\n", Getattr(n, "allegrocl:kind"), name);
#endif
  // Now walk the function parameter list and generate code to get arguments
  String *name_and_parms = NewStringf("%s (", mangled);
  int i;
  Parm *p;
  for (i = 0, p = parms; i < num_arguments; i++) {

#ifdef ALLEGROCL_DEBUG
	  String *temp1 = Getattr(p,"tmap:in");
	  String *temp2 = Getattr(p,"tmap:in:numinputs");
	  Printf(stderr,"  parm %d: %s, tmap:in='%s', tmap:in:numinputs='%s'\n", i, Getattr(p,"name"), temp1 ? temp1 : "", temp2 ? temp2 : "");
#endif

    while (p && checkAttribute(p, "tmap:in:numinputs", "0")) {
      p = Getattr(p, "tmap:in:next");
    }

    if (!p)
      break;

    SwigType *c_parm_type = Swig_cparse_type(Getattr(p, "tmap:ctype"));
    String *arg = NewStringf("l%s", Getattr(p, "lname"));

    // Emit parameter declaration
    if (gencomma)
      Printf(name_and_parms, ", ");
    String *parm_decl = SwigType_str(c_parm_type, arg);
    Printf(name_and_parms, "%s", parm_decl);
#ifdef ALLEGROCL_DEBUG
    Printf(stderr, "  param: %s\n", parm_decl);
#endif
    Delete(parm_decl);
    gencomma = 1;

    // Emit parameter conversion code
    String *parm_code = Getattr(p, "tmap:in");
    //if (!parm_code) {
    //  Swig_warning(...);
    //  p = nextSibling(p);
    /*} else */  {
      // canThrow(n, "in", p);
      Replaceall(parm_code, "$input", arg);
      Setattr(p, "emit:input", arg);
      Printf(f->code, "%s\n", parm_code);
      p = Getattr(p, "tmap:in:next");
    }

    Delete(arg);
  }
  Printf(name_and_parms, ")");

#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "   arity = %d(%d)\n", emit_num_lin_arguments(parms), emit_num_lin_arguments(Getattr(n,"wrap:parms")));
#endif

  // Emit the function definition
  String *signature = SwigType_str(return_type, name_and_parms);
  Printf(f->def, "EXPORT %s {", signature);
  if (CPlusPlus)
    Printf(f->code, "  try {\n");

  String *actioncode = emit_action(n);

  String *tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode);
  if (!is_void_return) {
    if (tm) { 
      Replaceall(tm, "$result", "lresult");
      Printf(f->code, "%s\n", tm);
      Printf(f->code, "    return lresult;\n");
      Delete(tm);
    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number,
		   "Unable to use return type %s in function %s.\n",
		   SwigType_str(t, 0), name);
    }
  }

  /* See if there is any return cleanup code */
  if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
    Replaceall(tm, "$source", Swig_cresult_name());
    Printf(f->code, "%s\n", tm);
    Delete(tm);
  }

  emit_return_variable(n, t, f);

  if (CPlusPlus) {
    Printf(f->code, "  } catch (...) {\n");
    if (!is_void_return)
      Printf(f->code, "    return (%s)0;\n",
	     SwigType_str(SwigType_strip_qualifiers(return_type),0));
    Printf(f->code, "  }\n");
  }
  Printf(f->code, "}\n");

  /* print this when in C mode? make this a command-line arg? */
  if (Generate_Wrapper)
    Wrapper_print(f, f_cxx_wrapper);

  String *f_buffer = NewString("");

  emit_defun(n, f_buffer);
  Setattr(n, "allegrocl:lisp-wrap", f_buffer);

  if (!overloaded || !Getattr(n, "sym:nextSibling")) {
    update_package_if_needed(n);
    emit_buffered_defuns(n);
    // this is the last overload.
    if (overloaded) {
      emit_dispatch_defun(n);
    }
  }

  DelWrapper(f);

  return SWIG_OK;
}

int ALLEGROCL::namespaceDeclaration(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "namespaceDecl: '%s'(%p) (fc=%p)\n", Getattr(n, "sym:name"), n, firstChild(n));
#endif

  /* don't wrap a namespace with no contents. package bloat.
     also, test-suite/namespace_class.i claims an unnamed namespace
     is 'private' and should not be wrapped. Complying...
  */
  if (Getattr(n,"unnamed") || !firstChild(n))
    return SWIG_OK;

  String *name = Getattr(n, "sym:name");

  String *old_namespace = current_namespace;
  if (Cmp(current_namespace, "") == 0)
    current_namespace = NewStringf("%s", name);
  else
    current_namespace = NewStringf("%s::%s", current_namespace, name);

  if (!GetInt(defined_namespace_packages, current_namespace)) {
    SetInt(defined_namespace_packages, current_namespace, 1);
    String *lispy_namespace = listify_namespace(current_namespace);
    Printf(f_clhead, "(swig-defpackage %s)\n", lispy_namespace);
    Delete(lispy_namespace);
  }

  emit_children(n);

  Delete(current_namespace);
  current_namespace = old_namespace;
  return SWIG_OK;
}

int ALLEGROCL::constructorHandler(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "constructorHandler %s\n", Getattr(n, "name"));
#endif
  // Swig_print_node(n);
  Setattr(n, "allegrocl:kind", "constructor");
  Setattr(n, "allegrocl:old-sym:name", Getattr(n, "sym:name"));

  // Let SWIG generate a global forwarding function.
  return Language::constructorHandler(n);
}

int ALLEGROCL::destructorHandler(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "destructorHandler %s\n", Getattr(n, "name"));
#endif

  Setattr(n, "allegrocl:kind", "destructor");
  Setattr(n, "allegrocl:old-sym:name", Getattr(n, "sym:name"));

  // Let SWIG generate a global forwarding function.
  return Language::destructorHandler(n);
}

int ALLEGROCL::constantWrapper(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "constantWrapper %s\n", Getattr(n, "name"));
#endif

  if (Generate_Wrapper) {
    // Setattr(n,"wrap:name",mangle_name(n, "ACLPP"));
    String *const_type = Getattr(n, "type");

    String *const_val = 0;
    String *raw_const = Getattr(n, "value");

    if (SwigType_type(const_type) == T_STRING) {
      const_val = NewStringf("\"%s\"", raw_const);
    } else if (SwigType_type(const_type) == T_CHAR) {
      const_val = NewStringf("'%s'", raw_const);
    } else {
      const_val = Copy(raw_const);
    }

    SwigType_add_qualifier(const_type, "const");

    String *ppcname = NewStringf("ACLppc_%s", Getattr(n, "sym:name"));
    // Printf(f_runtime, "static const %s = %s;\n", SwigType_lstr(const_type, ppcname), const_val);
    Printf(f_runtime, "static %s = %s;\n", SwigType_lstr(const_type, ppcname), const_val);

    Setattr(n, "name", ppcname);
    SetFlag(n, "feature:immutable");

    Delete(const_val);
    return variableWrapper(n);
  }

  String *type = Getattr(n, "type");
  String *value = Getattr(n, "value");
  String *converted_value = convert_literal(value, type);
  String *name = Getattr(n, "sym:name");

  Setattr(n, "allegrocl:kind", "constant");
  Setattr(n, "allegrocl:old-sym:name", Getattr(n, "sym:name"));

#if 0
  Printf(stdout, "constant %s is of type %s. value: %s\n", name, type, converted_value);
#endif

  if (converted_value) {
    Printf(f_clwrap, "(swig-defconstant \"%s\" %s)\n", name, converted_value);
  } else {
    Swig_warning(WARN_LANG_DISCARD_CONST, Getfile(n), Getline(n), "Unable to parse constant value '%s'. Setting to NIL\n", value);
    Printf(f_clwrap, "(swig-defconstant \"%s\" nil #| %s |#)\n", name, value);
  }

  Delete(converted_value);

  return SWIG_OK;
}

int ALLEGROCL::globalvariableHandler(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "globalvariableHandler %s\n", Getattr(n, "name"));
#endif

  if (Generate_Wrapper)
    return Language::globalvariableHandler(n);

  // String *name = Getattr(n, "name");
  SwigType *type = Getattr(n, "type");
  SwigType *rtype = SwigType_typedef_resolve_all(type);

  if (SwigType_isclass(rtype)) {
    SwigType_add_pointer(type);
    SwigType_add_pointer(rtype);
  }

  Printf(f_clwrap, "(swig-defvar \"%s\" \"%s\" :type %s)\n",
	 Getattr(n, "sym:name"), Getattr(n, "sym:name"), ((SwigType_isconst(type)) ? ":constant" : ":variable"));

  return SWIG_OK;
}

int ALLEGROCL::variableWrapper(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "variableWrapper %s\n", Getattr(n, "name"));
#endif
  Setattr(n, "allegrocl:kind", "variable");
  Setattr(n, "allegrocl:old-sym:name", Getattr(n, "sym:name"));

  // Let SWIG generate a get/set function pair.
  if (Generate_Wrapper)
    return Language::variableWrapper(n);

  String *name = Getattr(n, "name");
  SwigType *type = Getattr(n, "type");
  SwigType *ctype;
  SwigType *rtype = SwigType_typedef_resolve_all(type);

  String *mangled_name = mangle_name(n);

  int pointer_added = 0;

  if (SwigType_isclass(rtype)) {
    SwigType_add_pointer(type);
    SwigType_add_pointer(rtype);
    pointer_added = 1;
  }

  ctype = SwigType_str(type, 0);

  // EXPORT <SwigType_str> <mangled_name>;
  // <SwigType_str> <mangled_name> = <name>;
  Printf(f_runtime, "EXPORT %s %s;\n%s %s = %s%s;\n", ctype, mangled_name, ctype, mangled_name, (pointer_added ? "&" : ""), name);

  Printf(f_cl, "(swig-defvar \"%s\" :type %s)\n", mangled_name, ((SwigType_isconst(type)) ? ":constant" : ":variable"));

  Printf(stderr,"***\n");
  Delete(mangled_name);

#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "DONE variable %s\n", Getattr(n, "name"));
#endif

  return SWIG_OK;
}

int ALLEGROCL::memberfunctionHandler(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "memberfunctionHandler %s::%s\n", Getattr(parent_node_skipping_extends(n), "name"), Getattr(n, "name"));
  Swig_print_node(n);
#endif
  Setattr(n, "allegrocl:kind", "member function");
  Setattr(n, "allegrocl:old-sym:name", Getattr(n, "sym:name"));

  // Let SWIG generate a global forwarding function.
  return Language::memberfunctionHandler(n);
}

int ALLEGROCL::membervariableHandler(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "membervariableHandler %s::%s\n", Getattr(parent_node_skipping_extends(n), "name"), Getattr(n, "name"));
#endif
  Setattr(n, "allegrocl:kind", "member variable");
  Setattr(n, "allegrocl:old-sym:name", Getattr(n, "sym:name"));

  // Let SWIG generate a get/set function pair.
  return Language::membervariableHandler(n);
}

int ALLEGROCL::typedefHandler(Node *n) {
#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "In typedefHandler\n");
#endif

  SwigType *typedef_type = Getattr(n,"type");
  // has the side-effect of noting any implicit
  // template instantiations in type.
  String *ff_type = compose_foreign_type(n, typedef_type);

  String *sym_name = Getattr(n, "sym:name");

  String *name;
  String *type_ref;

  if (in_class) {
#ifdef ALLEGROCL_TYPE_DEBUG
    Printf(stderr, "  typedef in class '%s'(%p)\n", Getattr(in_class, "sym:name"), in_class);
#endif
    Setattr(n, "allegrocl:typedef:in-class", in_class);

    String *class_name = Getattr(in_class, "name");
    name = NewStringf("%s__%s", class_name, sym_name);
    type_ref = NewStringf("%s::%s", class_name, sym_name);
    Setattr(n, "allegrocl:in-class", in_class);
  } else {
    name = Copy(sym_name);
    type_ref = Copy(Getattr(n, "name"));
  }

  Setattr(n, "allegrocl:namespace", current_namespace);

  String *lookup = lookup_defined_foreign_type(typedef_type);

#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "** lookup='%s'(%p), typedef_type='%s', strcmp = '%d' strstr = '%d'\n", lookup, lookup, typedef_type, Strcmp(typedef_type,"void"), Strstr(ff_type,"__SWIGACL_FwdReference"));
#endif

  if(lookup || (!lookup && Strcmp(typedef_type,"void")) ||
     (!lookup && Strstr(ff_type,"__SWIGACL_FwdReference"))) {
	  add_defined_foreign_type(n, 0, type_ref, name);
  } else {
     add_forward_referenced_type(n);
  }

#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "Out typedefHandler\n");
#endif

  Delete(ff_type);

  return SWIG_OK;
}

// forward referenced classes are added specially to defined_foreign_types
int ALLEGROCL::classforwardDeclaration(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "classforwardDeclaration %s\n", Getattr(n, "name"));
#endif

  add_forward_referenced_type(n);
  return SWIG_OK;
}

int ALLEGROCL::classHandler(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "classHandler %s::%s\n", current_namespace, Getattr(n, "sym:name"));
#endif

  int result;

  if (Generate_Wrapper)
    result = cppClassHandler(n);
  else
    result = cClassHandler(n);

  return result;
}

int ALLEGROCL::cClassHandler(Node *n) {
#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "In cClassHandler\n");
#endif
  /* Add this structure to the known lisp types */
  // Printf(stderr, "Adding %s foreign type\n", name);
  String *ns = listify_namespace(current_namespace);

  add_defined_foreign_type(n);

  Delete(ns);

#ifdef ALLEGROCL_TYPE_DEBUG
  Printf(stderr, "Out cClassHandler\n");
#endif

  return SWIG_OK;
}

int ALLEGROCL::cppClassHandler(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "cppClassHandler %s\n", Getattr(n, "name"));
#endif

  // String *name=Getattr(n, "sym:name");
  // String *kind = Getattr(n,"kind");

  /* Template instantiation.
     Careful. 
     SWIG does not create instantiations of templated classes whenever
     it sees a templated class reference (say, as a return type, or
     in a parameter list).

     The %template directive results in a templated class instantiation
     that will actually be seen by <LANG> :: classHandler().

     In this case, we don't want to error if the type already exists;
     the point is to force the creation of wrappers for the templated
     class.
   */
  String *templated = Getattr(n, "template");
  String *t_name;
  // String *ns = listify_namespace(current_namespace);

  if (templated) {
    t_name = namespaced_name(n);
  } else {
    t_name = Getattr(n, "name");
  }

  Setattr(n, "allegrocl:namespace", current_namespace);

  /* Add this structure to the known lisp types.
     Class may contain references to the type currently being
     defined */
  if (!templated || !lookup_defined_foreign_type(t_name)) {
#ifdef ALLEGROCL_CLASS_DEBUG
    Printf(stderr, "Adding %s foreign type\n", Getattr(n, "sym:name"));
#endif
    add_defined_foreign_type(n);
  } else {
#ifdef ALLEGROCL_CLASS_DEBUG
    Printf(stderr, "cppClassHand: type %s already exists. Assuming %%template instantiation for wrapping purposes.\n", Getattr(n, "sym:name"));
#endif
    add_defined_foreign_type(n, 1);
  }

  // Generate slot accessors, constructor, and destructor.
  Node *prev_class = in_class;
  in_class = n;

  Node *c;
  // walk all member variables.
#ifdef ALLEGROCL_CLASS_DEBUG
  Printf(stderr, "   MANUALLY walking class members... \n");
#endif
  for (c = firstChild(n); c; c = nextSibling(c)) {
    // ping the types of all children--even protected and private
    // so their types can be added to the linked_type_list.
    SwigType *childType = NewStringf("%s%s", Getattr(c, "decl"),
				     Getattr(c, "type"));
#ifdef ALLEGROCL_CLASS_DEBUG
    Printf(stderr, "looking at child '%p' of type '%s' '%d'\n", c, childType, SwigType_isfunction(childType));
    // Swig_print_node(c);
#endif
    if (!SwigType_isfunction(childType))
      Delete(compose_foreign_type(n, childType));

    Delete(childType);
  }
#ifdef ALLEGROCL_CLASS_DEBUG
  Printf(stderr, "   MANUAL walk DONE.\n");
#endif

  // this will walk all necessary methods.
#ifdef ALLEGROCL_CLASS_DEBUG
  Printf(stderr, "   LANGUAGE walk of children...\n");
#endif
  Language::classHandler(n);
#ifdef ALLEGROCL_CLASS_DEBUG
  Printf(stderr, "   LANGUAGE walk DONE\n");
#endif
  in_class = prev_class;

  return SWIG_OK;
}

int ALLEGROCL::emit_one(Node *n) {
  // When the current package does not correspond with the current
  // namespace we need to generate an IN-PACKAGE form, unless the
  // current node is another namespace node.
  if (Cmp(nodeType(n), "namespace") != 0 && Cmp(current_package, current_namespace) != 0) {
    String *lispy_namespace = listify_namespace(current_namespace);
    Printf(f_clwrap, "(swig-in-package %s)\n", lispy_namespace);
    Delete(lispy_namespace);
    Delete(current_package);
    current_package = NewStringf("%s", current_namespace);
  }

  Setattr(n, "allegrocl:package", current_package);

  return Language::emit_one(n);
}

int ALLEGROCL::enumDeclaration(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "enumDeclaration %s\n", Getattr(n, "name"));
#endif

  if (getCurrentClass() && (cplus_mode != PUBLIC))
    return SWIG_NOWRAP;

  if (Getattr(n, "sym:name")) {
    add_defined_foreign_type(n);
  }
  Node *c;
  for (c = firstChild(n); c; c = nextSibling(c)) {
    ALLEGROCL::enumvalueDeclaration(c);
    // since we walk our own children, we need to add
    // the current package ourselves.
    Setattr(c, "allegrocl:package", current_package);
  }
  return SWIG_OK;
}


int ALLEGROCL::enumvalueDeclaration(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "enumvalueDeclaration %s\n", Getattr(n, "name"));
#endif
  /* print this when in C mode? make this a command-line arg? */
  if (Generate_Wrapper) {
	  SwigType *enum_type = Copy(Getattr(n,"type"));
	  String *mangled_name =
		  mangle_name(n, "ACL_ENUM",
			      in_class ? Getattr(in_class,"name") :
			      current_namespace);
	  
	  SwigType_add_qualifier(enum_type,"const");

	  String *enum_decl = SwigType_str(enum_type, mangled_name);
	  Printf(f_cxx_wrapper, "EXPORT %s;\n", enum_decl);
	  Printf(f_cxx_wrapper, "%s = %s;\n", enum_decl, Getattr(n, "value"));

    Delete(mangled_name);
    Delete(enum_type);
    Delete(enum_decl);
  }
  return SWIG_OK;
}

int ALLEGROCL::templateDeclaration(Node *n) {
#ifdef ALLEGROCL_DEBUG
  Printf(stderr, "templateDeclaration %s\n", Getattr(n, "name"));
#endif

  String *type = Getattr(n, "templatetype");

  // Printf(stderr, "tempDecl: %s %s\n", Getattr(n,"name"),
  //        type);
  // Swig_print_node(n);

  if (!Strcmp(type, "cdecl")) {
    SwigType *ty = NewStringf("%s%s", Getattr(n, "decl"),
			      Getattr(n, "type"));
    Delete(ty);
  }

  Delete(type);

  return SWIG_OK;
}

