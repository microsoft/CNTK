/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * nested.cxx
 *
 * Nested structs support
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"

// Nested classes processing section
static Hash *classhash = 0;

static String *make_name(Node *n, String *name, SwigType *decl) {
  int destructor = name && (*(Char(name)) == '~');
  if (String *yyrename = Getattr(n, "class_rename")) {
    String *s = NewString(yyrename);
    Delattr(n, "class_rename");
    if (destructor && (*(Char(s)) != '~')) {
      Insert(s, 0, "~");
    }
    return s;
  }

  if (!name)
    return 0;
  return Swig_name_make(n, 0, name, decl, 0);
}

// C version of add_symbols()
static void add_symbols_c(Node *n) {
  String *decl;
  String *wrn = 0;
  String *symname = 0;
  int iscdecl = Cmp(nodeType(n), "cdecl") == 0;
  Setattr(n, "ismember", "1");
  Setattr(n, "access", "public");
  if (Getattr(n, "sym:name"))
    return;
  decl = Getattr(n, "decl");
  if (!SwigType_isfunction(decl)) {
    String *name = Getattr(n, "name");
    String *makename = Getattr(n, "parser:makename");
    if (iscdecl) {
      String *storage = Getattr(n, "storage");
      if (Cmp(storage, "typedef") == 0) {
	Setattr(n, "kind", "typedef");
      } else {
	SwigType *type = Getattr(n, "type");
	String *value = Getattr(n, "value");
	Setattr(n, "kind", "variable");
	if (value && Len(value)) {
	  Setattr(n, "hasvalue", "1");
	}
	if (type) {
	  SwigType *ty;
	  SwigType *tmp = 0;
	  if (decl) {
	    ty = tmp = Copy(type);
	    SwigType_push(ty, decl);
	  } else {
	    ty = type;
	  }
	  if (!SwigType_ismutable(ty)) {
	    SetFlag(n, "hasconsttype");
	    SetFlag(n, "feature:immutable");
	  }
	  if (tmp)
	    Delete(tmp);
	}
	if (!type) {
	  Printf(stderr, "notype name %s\n", name);
	}
      }
    }
    Swig_features_get(Swig_cparse_features(), 0, name, 0, n);
    if (makename) {
      symname = make_name(n, makename, 0);
      Delattr(n, "parser:makename");	/* temporary information, don't leave it hanging around */
    } else {
      makename = name;
      symname = make_name(n, makename, 0);
    }

    if (!symname) {
      symname = Copy(Getattr(n, "unnamed"));
    }
    if (symname) {
      wrn = Swig_name_warning(n, 0, symname, 0);
    }
  } else {
    String *name = Getattr(n, "name");
    SwigType *fdecl = Copy(decl);
    SwigType *fun = SwigType_pop_function(fdecl);
    if (iscdecl) {
      Setattr(n, "kind", "function");
    }

    Swig_features_get(Swig_cparse_features(), 0, name, fun, n);

    symname = make_name(n, name, fun);
    wrn = Swig_name_warning(n, 0, symname, fun);

    Delete(fdecl);
    Delete(fun);

  }
  if (!symname)
    return;
  if (GetFlag(n, "feature:ignore")) {
    /* Only add to C symbol table and continue */
    Swig_symbol_add(0, n);
  } else if (strncmp(Char(symname), "$ignore", 7) == 0) {
    char *c = Char(symname) + 7;
    SetFlag(n, "feature:ignore");
    if (strlen(c)) {
      SWIG_WARN_NODE_BEGIN(n);
      Swig_warning(0, Getfile(n), Getline(n), "%s\n", c + 1);
      SWIG_WARN_NODE_END(n);
    }
    Swig_symbol_add(0, n);
  } else {
    Node *c;
    if ((wrn) && (Len(wrn))) {
      String *metaname = symname;
      if (!Getmeta(metaname, "already_warned")) {
	SWIG_WARN_NODE_BEGIN(n);
	Swig_warning(0, Getfile(n), Getline(n), "%s\n", wrn);
	SWIG_WARN_NODE_END(n);
	Setmeta(metaname, "already_warned", "1");
      }
    }
    c = Swig_symbol_add(symname, n);

    if (c != n) {
      /* symbol conflict attempting to add in the new symbol */
      if (Getattr(n, "sym:weak")) {
	Setattr(n, "sym:name", symname);
      } else {
	String *e = NewStringEmpty();
	String *en = NewStringEmpty();
	String *ec = NewStringEmpty();
	int redefined = Swig_need_redefined_warn(n, c, true);
	if (redefined) {
	  Printf(en, "Identifier '%s' redefined (ignored)", symname);
	  Printf(ec, "previous definition of '%s'", symname);
	} else {
	  Printf(en, "Redundant redeclaration of '%s'", symname);
	  Printf(ec, "previous declaration of '%s'", symname);
	}
	if (Cmp(symname, Getattr(n, "name"))) {
	  Printf(en, " (Renamed from '%s')", SwigType_namestr(Getattr(n, "name")));
	}
	Printf(en, ",");
	if (Cmp(symname, Getattr(c, "name"))) {
	  Printf(ec, " (Renamed from '%s')", SwigType_namestr(Getattr(c, "name")));
	}
	Printf(ec, ".");
	SWIG_WARN_NODE_BEGIN(n);
	if (redefined) {
	  Swig_warning(WARN_PARSE_REDEFINED, Getfile(n), Getline(n), "%s\n", en);
	  Swig_warning(WARN_PARSE_REDEFINED, Getfile(c), Getline(c), "%s\n", ec);
	} else {
	  Swig_warning(WARN_PARSE_REDUNDANT, Getfile(n), Getline(n), "%s\n", en);
	  Swig_warning(WARN_PARSE_REDUNDANT, Getfile(c), Getline(c), "%s\n", ec);
	}
	SWIG_WARN_NODE_END(n);
	Printf(e, "%s:%d:%s\n%s:%d:%s\n", Getfile(n), Getline(n), en, Getfile(c), Getline(c), ec);
	Setattr(n, "error", e);
	Delete(e);
	Delete(en);
	Delete(ec);
      }
    }
  }
  Delete(symname);
}

/* Strips C-style and C++-style comments from string in-place. */
static void strip_comments(char *string) {
  int state = 0;
  /*
   * 0 - not in comment
   * 1 - in c-style comment
   * 2 - in c++-style comment
   * 3 - in string
   * 4 - after reading / not in comments
   * 5 - after reading * in c-style comments
   * 6 - after reading \ in strings
   */
  char *c = string;
  while (*c) {
    switch (state) {
    case 0:
      if (*c == '\"')
	state = 3;
      else if (*c == '/')
	state = 4;
      break;
    case 1:
      if (*c == '*')
	state = 5;
      *c = ' ';
      break;
    case 2:
      if (*c == '\n')
	state = 0;
      else
	*c = ' ';
      break;
    case 3:
      if (*c == '\"')
	state = 0;
      else if (*c == '\\')
	state = 6;
      break;
    case 4:
      if (*c == '/') {
	*(c - 1) = ' ';
	*c = ' ';
	state = 2;
      } else if (*c == '*') {
	*(c - 1) = ' ';
	*c = ' ';
	state = 1;
      } else
	state = 0;
      break;
    case 5:
      if (*c == '/')
	state = 0;
      else
	state = 1;
      *c = ' ';
      break;
    case 6:
      state = 3;
      break;
    }
    ++c;
  }
}

// Create a %insert with a typedef to make a new name visible to C
static Node *create_insert(Node *n, bool noTypedef = false) {
  // format a typedef
  String *ccode = Getattr(n, "code");
  Push(ccode, " ");
  if (noTypedef) {
    Push(ccode, Getattr(n, "name"));
    Push(ccode, " ");
    Push(ccode, Getattr(n, "kind"));
  } else {
    Push(ccode, Getattr(n, "kind"));
    Push(ccode, "typedef ");
    Append(ccode, " ");
    Append(ccode, Getattr(n, "tdname"));
  }
  Append(ccode, ";");

  /* Strip comments - further code may break in presence of comments. */
  strip_comments(Char(ccode));

  /* Make all SWIG created typedef structs/unions/classes unnamed else
     redefinition errors occur - nasty hack alert. */
  if (!noTypedef) {
    const char *types_array[3] = { "struct", "union", "class" };
    for (int i = 0; i < 3; i++) {
      char *code_ptr = Char(ccode);
      while (code_ptr) {
	/* Replace struct name (as in 'struct name {...}' ) with whitespace
	   name will be between struct and opening brace */

	code_ptr = strstr(code_ptr, types_array[i]);
	if (code_ptr) {
	  char *open_bracket_pos;
	  code_ptr += strlen(types_array[i]);
	  open_bracket_pos = strchr(code_ptr, '{');
	  if (open_bracket_pos) {
	    /* Make sure we don't have something like struct A a; */
	    char *semi_colon_pos = strchr(code_ptr, ';');
	    if (!(semi_colon_pos && (semi_colon_pos < open_bracket_pos)))
	      while (code_ptr < open_bracket_pos)
		*code_ptr++ = ' ';
	  }
	}
      }
    }
  }
  {
    /* Remove SWIG directive %constant which may be left in the SWIG created typedefs */
    char *code_ptr = Char(ccode);
    while (code_ptr) {
      code_ptr = strstr(code_ptr, "%constant");
      if (code_ptr) {
	char *directive_end_pos = strchr(code_ptr, ';');
	if (directive_end_pos) {
	  while (code_ptr <= directive_end_pos)
	    *code_ptr++ = ' ';
	}
      }
    }
  }
  Node *newnode = NewHash();
  set_nodeType(newnode, "insert");
  Setfile(newnode, Getfile(n));
  Setline(newnode, Getline(n));
  String *code = NewStringEmpty();
  Wrapper_pretty_print(ccode, code);
  Setattr(newnode, "code", code);
  Delete(code);
  Delattr(n, "code");
  return newnode;
}

static void insertNodeAfter(Node *n, Node *c) {
  Node *g = parentNode(n);
  set_parentNode(c, g);
  Node *ns = nextSibling(n);
  if (Node *outer = Getattr(c, "nested:outer")) {
    while (ns && outer == Getattr(ns, "nested:outer")) {
      n = ns;
      ns = nextSibling(n);
    }
  }
  if (!ns) {
    set_lastChild(g, c);
  } else {
    set_nextSibling(c, ns);
    set_previousSibling(ns, c);
  }
  set_nextSibling(n, c);
  set_previousSibling(c, n);
}

void Swig_nested_name_unnamed_c_structs(Node *n) {
  if (!n)
    return;
  if (!classhash)
    classhash = Getattr(n, "classes");
  Node *c = firstChild(n);
  while (c) {
    Node *next = nextSibling(c);
    if (String *declName = Getattr(c, "nested:unnamed")) {
      if (Node *outer = Getattr(c, "nested:outer")) {
	// generate a name
	String *name = NewStringf("%s_%s", Getattr(outer, "name"), declName);
	Delattr(c, "nested:unnamed");
	// set the name to the class and symbol table
	Setattr(c, "tdname", name);
	Setattr(c, "name", name);
	Swig_symbol_setscope(Getattr(c, "symtab"));
	Swig_symbol_setscopename(name);
	// now that we have a name - gather base symbols
	if (List *publicBases = Getattr(c, "baselist")) {
	  List *bases = Swig_make_inherit_list(name, publicBases, 0);
	  Swig_inherit_base_symbols(bases);
	  Delete(bases);
	}
	Setattr(classhash, name, c);

	// Merge the extension into the symbol table
	if (Node *am = Getattr(Swig_extend_hash(), name)) {
	  Swig_extend_merge(c, am);
	  Swig_extend_append_previous(c, am);
	  Delattr(Swig_extend_hash(), name);
	}
	Swig_symbol_popscope();

	// process declarations following this type (assign correct new type)
	SwigType *ty = Copy(name);
	Node *decl = nextSibling(c);
	List *declList = NewList();
	while (decl && Getattr(decl, "nested:unnamedtype") == c) {
	  Setattr(decl, "type", ty);
	  Append(declList, decl);
	  Delattr(decl, "nested:unnamedtype");
	  SetFlag(decl, "feature:immutable");
	  add_symbols_c(decl);
	  decl = nextSibling(decl);
	}
	Delete(ty);
	Swig_symbol_setscope(Swig_symbol_global_scope());
	add_symbols_c(c);

	Node *ins = create_insert(c);
	insertNodeAfter(c, ins);
	removeNode(c);
	insertNodeAfter(n, c);
	Delete(ins);
	Delattr(c, "nested:outer");
      } else {
	// global unnamed struct - ignore it and it's instances
	SetFlag(c, "feature:ignore");
	while (next && Getattr(next, "nested:unnamedtype") == c) {
	  SetFlag(next, "feature:ignore");
	  next = nextSibling(next);
	}
	c = next;
	continue;
      }
    } else if (cparse_cplusplusout) {
      if (Getattr(c, "nested:outer")) {
	Node *ins = create_insert(c, true);
	insertNodeAfter(c, ins);
	Delete(ins);
	Delattr(c, "nested:outer");
      }
    }
    // process children
    Swig_nested_name_unnamed_c_structs(c);
    c = next;
  }
}

static void remove_outer_class_reference(Node *n) {
  for (Node *c = firstChild(n); c; c = nextSibling(c)) {
    if (GetFlag(c, "feature:flatnested") || Language::instance()->nestedClassesSupport() == Language::NCS_None) {
      Delattr(c, "nested:outer");
      remove_outer_class_reference(c);
    }
  }
}

void Swig_nested_process_classes(Node *n) {
  if (!n)
    return;
  Node *c = firstChild(n);
  while (c) {
    Node *next = nextSibling(c);
    if (!Getattr(c, "templatetype")) {
      if (GetFlag(c, "nested") && (GetFlag(c, "feature:flatnested") || Language::instance()->nestedClassesSupport() == Language::NCS_None)) {
	removeNode(c);
	if (!checkAttribute(c, "access", "public"))
	  SetFlag(c, "feature:ignore");
	else if (Strcmp(nodeType(n),"extend") == 0 && Strcmp(nodeType(parentNode(n)),"class") == 0)
	  insertNodeAfter(parentNode(n), c);
	else
	  insertNodeAfter(n, c);
      }
      Swig_nested_process_classes(c);
    }
    c = next;
  }
  remove_outer_class_reference(n);
}

