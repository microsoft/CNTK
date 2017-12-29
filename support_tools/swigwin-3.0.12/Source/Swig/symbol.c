/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * symbol.c
 *
 * This file implements the SWIG symbol table.  See details below.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "swigwarn.h"
#include <ctype.h>

/* #define SWIG_DEBUG*/
/* -----------------------------------------------------------------------------
 * Synopsis
 *
 * This module provides symbol table management for all of SWIG.  In previous
 * releases, the management of symbols was rather haphazard.  This module tries
 * to correct that.
 *
 * All symbols are associated with simple identifiers.  For example, here are some
 * declarations that generate symbol table entries:
 *
 *  decl                                    symbol
 *  --------------                          ------------
 *  void foo(int);                          foo
 *  int  x;                                 x
 *  typedef int *blah;                      blah
 *
 * Associated with each symbol is a Hash table that can contain any set of
 * attributes that make sense for that object.  For example:
 *
 *  typedef int *blah;             ---->    "name" : 'blah'
 *                                          "type" : 'int'
 *                                          "decl" : 'p.'
 *                                       "storage" : 'typedef'          
 * 
 * In some cases, the symbol table needs to manage overloaded entries.  For instance,
 * overloaded functions.  In this case, a linked list is built.  The "sym:nextSibling"
 * attribute is reserved to hold a link to the next entry.  For example:
 *
 * int foo(int);            --> "name" : "foo"         "name" : "foo"
 * int foo(int,double);         "type" : "int"         "type" : "int" 
 *                              "decl" : "f(int)."     "decl" : "f(int,double)."
 *                               ...                    ...
 *                   "sym:nextSibling" :  --------> "sym:nextSibling": --------> ...
 *
 * When more than one symbol has the same name, the symbol declarator is 
 * used to detect duplicates.  For example, in the above case, foo(int) and
 * foo(int,double) are different because their "decl" attribute is different.
 * However, if a third declaration "foo(int)" was made, it would generate a 
 * conflict (due to having a declarator that matches a previous entry).
 *
 * Structures and classes:
 *
 * C/C++ symbol tables are normally managed in a few different spaces.  The
 * most visible namespace is reserved for functions, variables, typedef, enum values
 * and such.  In C, a separate tag-space is reserved for 'struct name', 'class name',
 * and 'union name' declarations.   In SWIG, a single namespace is used for everything
 * this means that certain incompatibilities will arise with some C programs. For instance:
 *
 *        struct Foo {
 *             ...
 *        }
 *
 *        int Foo();       // Error. Name clash.  Works in C though 
 * 
 * Due to the unified namespace for structures, special handling is performed for
 * the following:
 *
 *        typedef struct Foo {
 *
 *        } Foo;
 * 
 * In this case, the symbol table contains an entry for the structure itself.  The
 * typedef is left out of the symbol table.
 *
 * Target language vs C:
 *
 * The symbol tables are normally managed *in the namespace of the target language*.
 * This means that name-collisions can be resolved using %rename and related 
 * directives.   A quirk of this is that sometimes the symbol tables need to
 * be used for C type resolution as well.  To handle this, each symbol table
 * also has a C-symbol table lurking behind the scenes.  This is used to locate 
 * symbols in the C namespace.  However, this symbol table is not used for error 
 * reporting nor is it used for anything else during code generation.
 *
 * Symbol table structure:
 *
 * Symbol tables themselves are a special kind of node that is organized just like
 * a normal parse tree node.  Symbol tables are organized in a tree that can be
 * traversed using the SWIG-DOM API. The following attributes names are reserved.
 *
 *     name           -- Name of the scope defined by the symbol table (if any)
 *                       This name is the C-scope name and is not affected by
 *                       %renaming operations
 *     symtab         -- Hash table mapping identifiers to nodes.
 *     csymtab        -- Hash table mapping C identifiers to nodes.
 *
 * Reserved attributes on symbol objects:
 *
 * When a symbol is placed in the symbol table, the following attributes
 * are set:
 *       
 *     sym:name             -- Symbol name
 *     sym:nextSibling      -- Next symbol (if overloaded)
 *     sym:previousSibling  -- Previous symbol (if overloaded)
 *     sym:symtab           -- Symbol table object holding the symbol
 *     sym:overloaded       -- Set to the first symbol if overloaded
 *
 * These names are modeled after XML namespaces.  In particular, every attribute 
 * pertaining to symbol table management is prefaced by the "sym:" prefix.   
 *
 * An example dump of the parse tree showing symbol table entries for the 
 * following code should clarify this:
 *
 *   namespace OuterNamespace {
 *       namespace InnerNamespace {
 *           class Class {
 *           };
 *           struct Struct {
 *               int Var;
 *           };
 *       }
 *    }
 *
 *   +++ namespace ----------------------------------------
 *   | sym:name     - "OuterNamespace"
 *   | symtab       - 0xa064bf0
 *   | sym:symtab   - 0xa041690
 *   | sym:overname - "__SWIG_0"
 *  
 *         +++ namespace ----------------------------------------
 *         | sym:name     - "InnerNamespace"
 *         | symtab       - 0xa064cc0
 *         | sym:symtab   - 0xa064bf0
 *         | sym:overname - "__SWIG_0"
 *  
 *               +++ class ----------------------------------------
 *               | sym:name     - "Class"
 *               | symtab       - 0xa064d80
 *               | sym:symtab   - 0xa064cc0
 *               | sym:overname - "__SWIG_0"
 *               | 
 *               +++ class ----------------------------------------
 *               | sym:name     - "Struct"
 *               | symtab       - 0xa064f00
 *               | sym:symtab   - 0xa064cc0
 *               | sym:overname - "__SWIG_0"
 *  
 *                     +++ cdecl ----------------------------------------
 *                     | sym:name     - "Var"
 *                     | sym:symtab   - 0xa064f00
 *                     | sym:overname - "__SWIG_0"
 *                     | 
 *  
 *
 * Each class and namespace has its own scope and thus a new symbol table (sym)
 * is created. The sym attribute is only set for the first entry in the symbol
 * table. The sym:symtab entry points to the symbol table in which the symbol
 * exists, so for example, Struct is in the scope OuterNamespace::InnerNamespace
 * so sym:symtab points to this symbol table (0xa064cc0).
 *
 * ----------------------------------------------------------------------------- */

static Hash *current = 0;	/* The current symbol table hash */
static Hash *ccurrent = 0;	/* The current c symbol table hash */
static Hash *current_symtab = 0;	/* Current symbol table node */
static Hash *symtabs = 0;	/* Hash of all symbol tables by fully-qualified name */
static Hash *global_scope = 0;	/* Global scope */

static int use_inherit = 1;

/* common attribute keys, to avoid calling find_key all the times */


/* -----------------------------------------------------------------------------
 * Swig_symbol_print_tables()
 *
 * Debug display of symbol tables
 * ----------------------------------------------------------------------------- */

void Swig_symbol_print_tables(Symtab *symtab) {
  if (!symtab)
    symtab = current_symtab;

  Printf(stdout, "SYMBOL TABLES start  =======================================\n");
  Swig_print_tree(symtab);
  Printf(stdout, "SYMBOL TABLES finish =======================================\n");
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_print_tables_summary()
 *
 * Debug summary display of all symbol tables by fully-qualified name 
 * ----------------------------------------------------------------------------- */

void Swig_symbol_print_tables_summary(void) {
  Printf(stdout, "SYMBOL TABLES SUMMARY start  =======================================\n");
  Swig_print_node(symtabs);
  Printf(stdout, "SYMBOL TABLES SUMMARY finish =======================================\n");
}

/* -----------------------------------------------------------------------------
 * symbol_print_symbols()
 * ----------------------------------------------------------------------------- */

static void symbol_print_symbols(const char *symboltabletype) {
  Node *table = symtabs;
  Iterator ki = First(table);
  while (ki.key) {
    String *k = ki.key;
    Printf(stdout, "===================================================\n");
    Printf(stdout, "%s -\n", k);
    {
      Symtab *symtab = Getattr(Getattr(table, k), symboltabletype);
      Iterator it = First(symtab);
      while (it.key) {
	String *symname = it.key;
	Printf(stdout, "  %s (%s)\n", symname, nodeType(it.item));
	/*
	Printf(stdout, "  %s - %p (%s)\n", symname, it.item, Getattr(it.item, "name"));
	*/
	it = Next(it);
      }
    }
    ki = Next(ki);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_print_symbols()
 *
 * Debug display of all the target language symbols
 * ----------------------------------------------------------------------------- */

void Swig_symbol_print_symbols(void) {
  Printf(stdout, "SYMBOLS start  =======================================\n");
  symbol_print_symbols("symtab");
  Printf(stdout, "SYMBOLS finish =======================================\n");
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_print_csymbols()
 *
 * Debug display of all the C symbols
 * ----------------------------------------------------------------------------- */

void Swig_symbol_print_csymbols(void) {
  Printf(stdout, "CSYMBOLS start  =======================================\n");
  symbol_print_symbols("csymtab");
  Printf(stdout, "CSYMBOLS finish =======================================\n");
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_init()
 *
 * Create a new symbol table object
 * ----------------------------------------------------------------------------- */

void Swig_symbol_init(void) {

  current = NewHash();
  current_symtab = NewHash();
  ccurrent = NewHash();
  set_nodeType(current_symtab, "symboltable");
  Setattr(current_symtab, "symtab", current);
  Delete(current);
  Setattr(current_symtab, "csymtab", ccurrent);
  Delete(ccurrent);

  /* Set the global scope */
  symtabs = NewHash();
  Setattr(symtabs, "", current_symtab);
  Delete(current_symtab);
  global_scope = current_symtab;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_setscopename()
 *
 * Set the C scopename of the current symbol table.
 * ----------------------------------------------------------------------------- */

void Swig_symbol_setscopename(const_String_or_char_ptr name) {
  String *qname;
  /* assert(!Getattr(current_symtab,"name")); */
  Setattr(current_symtab, "name", name);

  /* Set nested scope in parent */

  qname = Swig_symbol_qualifiedscopename(current_symtab);

  /* Save a reference to this scope */
  Setattr(symtabs, qname, current_symtab);
  Delete(qname);
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_getscopename()
 *
 * Get the C scopename of the current symbol table
 * ----------------------------------------------------------------------------- */

String *Swig_symbol_getscopename(void) {
  return Getattr(current_symtab, "name");
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_getscope()
 *
 * Given a fully qualified C scopename, this function returns a symbol table
 * ----------------------------------------------------------------------------- */

Symtab *Swig_symbol_getscope(const_String_or_char_ptr name) {
  if (!symtabs)
    return 0;
  if (Equal("::", (const_String_or_char_ptr ) name))
    name = "";
  return Getattr(symtabs, name);
}

/* ----------------------------------------------------------------------------- 
 * Swig_symbol_qualifiedscopename()
 *
 * Get the fully qualified C scopename of a symbol table.  Note, this only pertains
 * to the C/C++ scope name.  It is not affected by renaming.
 * ----------------------------------------------------------------------------- */

String *Swig_symbol_qualifiedscopename(Symtab *symtab) {
  String *result = 0;
  Hash *parent;
  String *name;
  if (!symtab)
    symtab = current_symtab;
  parent = Getattr(symtab, "parentNode");
  if (parent) {
    result = Swig_symbol_qualifiedscopename(parent);
  }
  name = Getattr(symtab, "name");
  if (name) {
    if (!result) {
      result = NewStringEmpty();
    }
    if (Len(result)) {
      Printv(result, "::", name, NIL);
    } else {
      Append(result, name);
    }
  }
  return result;
}

/* ----------------------------------------------------------------------------- 
 * Swig_symbol_qualified_language_scopename()
 *
 * Get the fully qualified C scopename of a symbol table but using a language
 * specific separator for the scopenames. Basically the same as
 * Swig_symbol_qualifiedscopename() but using the different separator.
 * ----------------------------------------------------------------------------- */

String *Swig_symbol_qualified_language_scopename(Symtab *n) {
  /* TODO: fix for %rename to work */
  String *result = Swig_symbol_qualifiedscopename(n);
  Replaceall(result, "::", NSPACE_SEPARATOR);
  return result;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_newscope()
 *
 * Create a new scope.  Returns the newly created scope.
 * ----------------------------------------------------------------------------- */

Symtab *Swig_symbol_newscope(void) {
  Hash *n;
  Hash *hsyms, *h;

  hsyms = NewHash();
  h = NewHash();

  set_nodeType(h, "symboltable");
  Setattr(h, "symtab", hsyms);
  Delete(hsyms);
  set_parentNode(h, current_symtab);

  n = lastChild(current_symtab);
  if (!n) {
    set_firstChild(current_symtab, h);
  } else {
    set_nextSibling(n, h);
    Delete(h);
  }
  set_lastChild(current_symtab, h);
  current = hsyms;
  ccurrent = NewHash();
  Setattr(h, "csymtab", ccurrent);
  Delete(ccurrent);
  current_symtab = h;
  return h;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_setscope()
 *
 * Set the current scope.  Returns the previous current scope.
 * ----------------------------------------------------------------------------- */

Symtab *Swig_symbol_setscope(Symtab *sym) {
  Symtab *ret = current_symtab;
  current_symtab = sym;
  current = Getattr(sym, "symtab");
  assert(current);
  ccurrent = Getattr(sym, "csymtab");
  assert(ccurrent);
  return ret;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_popscope()
 *
 * Pop out of the current scope.  Returns the popped scope and sets the
 * scope to the parent scope.
 * ----------------------------------------------------------------------------- */

Symtab *Swig_symbol_popscope(void) {
  Hash *h = current_symtab;
  current_symtab = Getattr(current_symtab, "parentNode");
  assert(current_symtab);
  current = Getattr(current_symtab, "symtab");
  assert(current);
  ccurrent = Getattr(current_symtab, "csymtab");
  assert(ccurrent);
  return h;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_global_scope()
 *
 * Return the symbol table for the global scope.
 * ----------------------------------------------------------------------------- */

Symtab *Swig_symbol_global_scope(void) {
  return global_scope;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_current()
 *
 * Return the current symbol table.
 * ----------------------------------------------------------------------------- */

Symtab *Swig_symbol_current(void) {
  return current_symtab;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_alias()
 *
 * Makes an alias for a symbol in the global symbol table.
 * Primarily for namespace aliases such as 'namespace X = Y;'.
 * ----------------------------------------------------------------------------- */

void Swig_symbol_alias(const_String_or_char_ptr aliasname, Symtab *s) {
  String *qname = Swig_symbol_qualifiedscopename(current_symtab);
  if (qname) {
    Printf(qname, "::%s", aliasname);
  } else {
    qname = NewString(aliasname);
  }
  if (!Getattr(symtabs, qname)) {
    Setattr(symtabs, qname, s);
  }
  Delete(qname);
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_inherit()
 *
 * Inherit symbols from another scope. Primarily for C++ inheritance and
 * for using directives, such as 'using namespace X;'
 * but not for using declarations, such as 'using A;'.
 * ----------------------------------------------------------------------------- */

void Swig_symbol_inherit(Symtab *s) {
  int i, ilen;
  List *inherit = Getattr(current_symtab, "inherit");
  if (!inherit) {
    inherit = NewList();
    Setattr(current_symtab, "inherit", inherit);
    Delete(inherit);
  }

  if (s == current_symtab) {
    Swig_warning(WARN_PARSE_REC_INHERITANCE, Getfile(s), Getline(s), "Recursive scope inheritance of '%s'.\n", Getattr(s, "name"));
    return;
  }
  assert(s != current_symtab);
  ilen = Len(inherit);
  for (i = 0; i < ilen; i++) {
    Node *n = Getitem(inherit, i);
    if (n == s)
      return;			/* Already inherited */
  }
  Append(inherit, s);
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_cadd()
 *
 * Adds a node to the C symbol table only.
 * ----------------------------------------------------------------------------- */

void Swig_symbol_cadd(const_String_or_char_ptr name, Node *n) {
  Node *append = 0;
  Node *cn;
  /* There are a few options for weak symbols.  A "weak" symbol 
     is any symbol that can be replaced by another symbol in the C symbol
     table.  An example would be a forward class declaration.  A forward
     class sits in the symbol table until a real class declaration comes along.

     Certain symbols are marked as "sym:typename".  These are important 
     symbols related to the C++ type-system and take precedence in the C
     symbol table.  An example might be code like this:

     template<class T> T foo(T x);
     int foo(int);

     In this case, the template is marked with "sym:typename" so that it
     stays in the C symbol table (so that it can be expanded using %template).
   */

  if (!name)
    return;

  if (SwigType_istemplate(name)) {
    String *cname = NewString(name);
    String *dname = Swig_symbol_template_deftype(cname, 0);
    if (!Equal(dname, name)) {
      Swig_symbol_cadd(dname, n);
    }
    Delete(dname);
    Delete(cname);
  }
#ifdef SWIG_DEBUG
  Printf(stderr, "symbol_cadd %s %p\n", name, n);
#endif
  cn = Getattr(ccurrent, name);

  if (cn && (Getattr(cn, "sym:typename"))) {
    /* The node in the C symbol table is a typename.  Do nothing */
    /* We might append the symbol at the end */
    append = n;
  } else if (cn && (Getattr(cn, "sym:weak"))) {
    /* The node in the symbol table is weak. Replace it */
    if (checkAttribute(cn, "nodeType", "template")
	&& checkAttribute(cn, "templatetype", "classforward")) {
      /* The node is a template classforward declaration, and the
         default template parameters here take precedence. */
      ParmList *pc = Getattr(cn, "templateparms");
      ParmList *pn = Getattr(n, "templateparms");
#ifdef SWIG_DEBUG
      Printf(stderr, "found template classforward %s\n", Getattr(cn, "name"));
#endif
      while (pc && pn) {
	String *value = Getattr(pc, "value");
	if (value) {
#ifdef SWIG_DEBUG
	  Printf(stderr, "add default template value %s %s\n", Getattr(pc, "name"), value);
#endif
	  Setattr(pn, "value", value);
	}
	pc = nextSibling(pc);
	pn = nextSibling(pn);
      }
      Setattr(n, "templateparms", Getattr(cn, "templateparms"));
    }
    Setattr(ccurrent, name, n);

  } else if (cn && (Getattr(n, "sym:weak"))) {
    /* The node being added is weak.  Don't worry about it */
  } else if (cn && (Getattr(n, "sym:typename"))) {
    /* The node being added is a typename.  We definitely add it */
    Setattr(ccurrent, name, n);
    append = cn;
  } else if (cn && (Checkattr(cn, "nodeType", "templateparm"))) {
    Swig_error(Getfile(n), Getline(n), "Declaration of '%s' shadows template parameter,\n", name);
    Swig_error(Getfile(cn), Getline(cn), "previous template parameter declaration '%s'.\n", name);
    return;
  } else if (cn) {
    append = n;
  } else if (!cn) {
    /* No conflict. Add the symbol */
    Setattr(ccurrent, name, n);
  }

  /* Multiple entries in the C symbol table.   We append to to the symbol table */
  if (append) {
    Node *fn, *pn = 0;
    cn = Getattr(ccurrent, name);
    fn = cn;
    while (fn) {
      pn = fn;
      if (fn == append) {
	/* already added. Bail */
	return;
      }
      fn = Getattr(fn, "csym:nextSibling");
    }
    if (pn) {
      Setattr(pn, "csym:nextSibling", append);
    }
  }

  /* Special typedef handling.  When a typedef node is added to the symbol table, we
     might have to add a type alias.   This would occur if the typedef mapped to another
     scope in the system.  For example:

     class Foo {
     };

     typedef Foo OtherFoo;

     In this case, OtherFoo becomes an alias for Foo. */

  {
    Node *td = n;
    while (td && Checkattr(td, "nodeType", "cdecl") && Checkattr(td, "storage", "typedef")) {
      SwigType *type;
      Node *td1;
      type = Copy(Getattr(td, "type"));
      SwigType_push(type, Getattr(td, "decl"));
      td1 = Swig_symbol_clookup(type, 0);

      /* Fix pathetic case #1214313:

         class Foo
         {
         };

         typedef Foo FooBar;

         class CBaz
         {
         public:
         typedef FooBar Foo;
         };

         ie, when Foo -> FooBar -> Foo, jump one scope up when possible.

       */
      if (td1 && Checkattr(td1, "storage", "typedef")) {
	String *st = Getattr(td1, "type");
	String *sn = Getattr(td, "name");
	if (st && sn && Equal(st, sn)) {
	  Symtab *sc = Getattr(current_symtab, "parentNode");
	  if (sc)
	    td1 = Swig_symbol_clookup(type, sc);
	}
      }

      Delete(type);
      if (td1 == td)
	break;
      td = td1;
      if (td) {
	Symtab *st = Getattr(td, "symtab");
	if (st) {
	  Swig_symbol_alias(Getattr(n, "name"), st);
	  break;
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------------- 
 * Swig_symbol_add()
 *
 * Adds a node to the symbol table.  Returns the node itself if successfully
 * added.  Otherwise, it returns the symbol table entry of the conflicting node.
 *
 * Also places the symbol in a behind-the-scenes C symbol table.  This is needed
 * for namespace support, type resolution, and other issues.
 * ----------------------------------------------------------------------------- */

Node *Swig_symbol_add(const_String_or_char_ptr symname, Node *n) {
  Hash *c, *cn, *cl = 0;
  SwigType *decl, *ndecl;
  String *cstorage, *nstorage;
  int nt = 0, ct = 0;
  int pn = 0;
  int u1 = 0, u2 = 0;
  String *name, *overname;

  /* See if the node has a name.  If so, we place in the C symbol table for this
     scope. We don't worry about overloading here---the primary purpose of this
     is to record information for type/name resolution for later. Conflicts
     in C namespaces are errors, but these will be caught by the C++ compiler
     when compiling the wrapper code */


  /* There are a few options for weak symbols.  A "weak" symbol 
     is any symbol that can be replaced by another symbol in the C symbol
     table.  An example would be a forward class declaration.  A forward
     class sits in the symbol table until a real class declaration comes along.

     Certain symbols are marked as "sym:typename".  These are important 
     symbols related to the C++ type-system and take precedence in the C
     symbol table.  An example might be code like this:

     template<class T> T foo(T x);
     int foo(int);

     In this case, the template is marked with "sym:typename" so that it
     stays in the C symbol table (so that it can be expanded using %template).
   */

  name = Getattr(n, "name");
  if (name && Len(name)) {
    Swig_symbol_cadd(name, n);
  }

  /* No symbol name defined.  We return. */
  if (!symname) {
    Setattr(n, "sym:symtab", current_symtab);
    return n;
  }

  /* If node is ignored. We don't proceed any further */
  if (GetFlag(n, "feature:ignore"))
    return n;

  /* See if the symbol already exists in the table */
  c = Getattr(current, symname);

  /* Check for a weak symbol.  A weak symbol is allowed to be in the
     symbol table, but is silently overwritten by other symbols.  An example
     would be a forward class declaration.  For instance:

     class Foo;

     In this case, "Foo" sits in the symbol table.  However, the
     definition of Foo would replace the entry if it appeared later. */

  if (c && Getattr(c, "sym:weak")) {
    c = 0;
  }
  if (c) {
    /* There is a symbol table conflict.  There are a few cases to consider here:
       (1) A conflict between a class/enum and a typedef declaration is okay.
       In this case, the symbol table entry is set to the class/enum declaration
       itself, not the typedef.   

       (2) A conflict between namespaces is okay--namespaces are open

       (3) Otherwise, overloading is only allowed for functions
     */

    /* Check for namespaces */
    String *ntype = Getattr(n, "nodeType");
    if ((Equal(ntype, Getattr(c, "nodeType"))) && ((Equal(ntype, "namespace")))) {
      Node *cl, *pcl = 0;
      cl = c;
      while (cl) {
	pcl = cl;
	cl = Getattr(cl, "sym:nextSibling");
      }
      Setattr(pcl, "sym:nextSibling", n);
      Setattr(n, "sym:symtab", current_symtab);
      Setattr(n, "sym:name", symname);
      Setattr(n, "sym:previousSibling", pcl);
      return n;
    }
    if (Getattr(n, "allows_typedef"))
      nt = 1;
    if (Getattr(c, "allows_typedef"))
      ct = 1;
    if (nt || ct) {
      Node *td, *other;
      String *s;
      /* At least one of the nodes allows typedef overloading.  Make sure that
         both don't--this would be a conflict */

      if (nt && ct)
	return c;

      /* Figure out which node allows the typedef */
      if (nt) {
	td = n;
	other = c;
      } else {
	td = c;
	other = n;
      }
      /* Make sure the other node is a typedef */
      s = Getattr(other, "storage");
      if (!s || (!Equal(s, "typedef")))
	return c;		/* No.  This is a conflict */

      /* Hmmm.  This appears to be okay.  Make sure the symbol table refers to the allow_type node */

      if (td != c) {
	Setattr(current, symname, td);
	Setattr(td, "sym:symtab", current_symtab);
	Setattr(td, "sym:name", symname);
      }
      return n;
    }

    decl = Getattr(c, "decl");
    ndecl = Getattr(n, "decl");

    {
      String *nt1, *nt2;
      nt1 = Getattr(n, "nodeType");
      if (Equal(nt1, "template"))
	nt1 = Getattr(n, "templatetype");
      nt2 = Getattr(c, "nodeType");
      if (Equal(nt2, "template"))
	nt2 = Getattr(c, "templatetype");
      if (Equal(nt1, "using"))
	u1 = 1;
      if (Equal(nt2, "using"))
	u2 = 1;

      if ((!Equal(nt1, nt2)) && !(u1 || u2))
	return c;
    }
    if (!(u1 || u2)) {
      if ((!SwigType_isfunction(decl)) || (!SwigType_isfunction(ndecl))) {
	/* Symbol table conflict */
	return c;
      }
    }

    /* Hmmm. Declarator seems to indicate that this is a function */
    /* Look at storage class to see if compatible */
    cstorage = Getattr(c, "storage");
    nstorage = Getattr(n, "storage");

    /* If either one is declared as typedef, forget it. We're hosed */
    if (Cmp(cstorage, "typedef") == 0) {
      return c;
    }
    if (Cmp(nstorage, "typedef") == 0) {
      return c;
    }

    /* Okay. Walk down the list of symbols and see if we get a declarator match */
    {
      String *nt = Getattr(n, "nodeType");
      int n_template = Equal(nt, "template") && Checkattr(n, "templatetype", "cdecl");
      int n_plain_cdecl = Equal(nt, "cdecl");
      cn = c;
      pn = 0;
      while (cn) {
	decl = Getattr(cn, "decl");
	if (!(u1 || u2)) {
	  if (Cmp(ndecl, decl) == 0) {
	    /* Declarator conflict */
	    /* Now check we don't have a non-templated function overloaded by a templated function with same params,
	     * eg void foo(); template<typename> void foo(); */
	    String *cnt = Getattr(cn, "nodeType");
	    int cn_template = Equal(cnt, "template") && Checkattr(cn, "templatetype", "cdecl");
	    int cn_plain_cdecl = Equal(cnt, "cdecl");
	    if (!((n_template && cn_plain_cdecl) || (cn_template && n_plain_cdecl))) {
	      /* found a conflict */
	      return cn;
	    }
	  }
	}
	cl = cn;
	cn = Getattr(cn, "sym:nextSibling");
	pn++;
      }
    }
    /* Well, we made it this far.  Guess we can drop the symbol in place */
    Setattr(n, "sym:symtab", current_symtab);
    Setattr(n, "sym:name", symname);
    /* Printf(stdout,"%s %p\n", Getattr(n,"sym:overname"), current_symtab); */
    assert(!Getattr(n, "sym:overname"));
    overname = NewStringf("__SWIG_%d", pn);
    Setattr(n, "sym:overname", overname);
    /*Printf(stdout,"%s %s %s\n", symname, Getattr(n,"decl"), Getattr(n,"sym:overname")); */
    Setattr(cl, "sym:nextSibling", n);
    Setattr(n, "sym:previousSibling", cl);
    Setattr(cl, "sym:overloaded", c);
    Setattr(n, "sym:overloaded", c);
    Delete(overname);
    return n;
  }

  /* No conflict.  Just add it */
  Setattr(n, "sym:symtab", current_symtab);
  Setattr(n, "sym:name", symname);
  /* Printf(stdout,"%s\n", Getattr(n,"sym:overname")); */
  overname = NewStringf("__SWIG_%d", pn);
  Setattr(n, "sym:overname", overname);
  Delete(overname);
  /* Printf(stdout,"%s %s %s\n", symname, Getattr(n,"decl"), Getattr(n,"sym:overname")); */
  Setattr(current, symname, n);
  return n;
}

/* -----------------------------------------------------------------------------
 * symbol_lookup()
 *
 * Internal function to handle fully qualified symbol table lookups.  This
 * works from the symbol table supplied in symtab and unwinds its way out
 * towards the global scope. 
 *
 * This function operates in the C namespace, not the target namespace.
 *
 * The check function is an optional callback that can be used to verify a particular
 * symbol match.   This is only used in some of the more exotic parts of SWIG. For instance,
 * verifying that a class hierarchy implements all pure virtual methods.
 * ----------------------------------------------------------------------------- */

static Node *_symbol_lookup(const String *name, Symtab *symtab, int (*check) (Node *n)) {
  Node *n;
  List *inherit;
  Hash *sym = Getattr(symtab, "csymtab");
  if (Getmark(symtab))
    return 0;
  Setmark(symtab, 1);

  n = Getattr(sym, name);

#ifdef SWIG_DEBUG
  Printf(stderr, "symbol_look %s %p %p %s\n", name, n, symtab, Getattr(symtab, "name"));
#endif

  if (n) {
    /* if a check-function is defined.  Call it to determine a match */
    if (check) {
      int c = check(n);
      if (c == 1) {
	Setmark(symtab, 0);
	return n;
      }
      if (c < 0) {
	/* Terminate the search right away */
	Setmark(symtab, 0);
	return 0;
      }
    } else {
      Setmark(symtab, 0);
      return n;
    }
  }

  if (!n && SwigType_istemplate(name)) {
    String *dname = 0;
    Setmark(symtab, 0);
    dname = Swig_symbol_template_deftype(name, symtab);
    if (!Equal(dname, name)) {
      n = _symbol_lookup(dname, symtab, check);
    }
    Delete(dname);
    if (n)
      return n;
    Setmark(symtab, 1);
  }

  inherit = Getattr(symtab, "inherit");
  if (inherit && use_inherit) {
    int i, len;
    len = Len(inherit);
    for (i = 0; i < len; i++) {
      n = _symbol_lookup(name, Getitem(inherit, i), check);
      if (n) {
	Setmark(symtab, 0);
	return n;
      }
    }
  }

  Setmark(symtab, 0);
  return 0;
}

static Node *symbol_lookup(const_String_or_char_ptr name, Symtab *symtab, int (*check) (Node *n)) {
  Node *n = 0;
  if (DohCheck(name)) {
    n = _symbol_lookup(name, symtab, check);
  } else {
    String *sname = NewString(name);
    n = _symbol_lookup(sname, symtab, check);
    Delete(sname);
  }
  return n;
}



/* -----------------------------------------------------------------------------
 * symbol_lookup_qualified()
 * ----------------------------------------------------------------------------- */

static Node *symbol_lookup_qualified(const_String_or_char_ptr name, Symtab *symtab, const String *prefix, int local, int (*checkfunc) (Node *n)) {
  /* This is a little funky, we search by fully qualified names */

  if (!symtab)
    return 0;
  if (!prefix) {
    Node *n;
    String *bname;
    String *prefix;
    Swig_scopename_split(name, &prefix, &bname);
    n = symbol_lookup_qualified(bname, symtab, prefix, local, checkfunc);
    Delete(bname);
    Delete(prefix);
    return n;
  } else {
    Symtab *st;
    Node *n = 0;
    /* Make qualified name of current scope */
    String *qalloc = 0;
    String *qname = Swig_symbol_qualifiedscopename(symtab);
    const String *cqname;
    if (qname) {
      if (Len(qname)) {
	if (prefix && Len(prefix)) {
	  Printv(qname, "::", prefix, NIL);
	}
      } else {
	Append(qname, prefix);
      }
      qalloc = qname;
      cqname = qname;
    } else {
      cqname = prefix;
    }
    st = Getattr(symtabs, cqname);
    /* Found a scope match */
    if (st) {
      if (!name) {
	if (qalloc)
	  Delete(qalloc);
	return st;
      }
      n = symbol_lookup(name, st, checkfunc);
    }
    if (qalloc)
      Delete(qalloc);

    if (!n) {
      if (!local) {
	Node *pn = Getattr(symtab, "parentNode");
	if (pn)
	  n = symbol_lookup_qualified(name, pn, prefix, local, checkfunc);

	/* Check inherited scopes */
	if (!n) {
	  List *inherit = Getattr(symtab, "inherit");
	  if (inherit && use_inherit) {
	    int i, len;
	    len = Len(inherit);
	    for (i = 0; i < len; i++) {
	      Node *prefix_node = symbol_lookup(prefix, Getitem(inherit, i), checkfunc);
	      if (prefix_node) {
		Node *prefix_symtab = Getattr(prefix_node, "symtab");
		if (prefix_symtab) {
		  n = symbol_lookup(name, prefix_symtab, checkfunc);
		  break;
		}
	      }
	    }
	  }
	}
      } else {
	n = 0;
      }
    }
    return n;
  }
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_clookup()
 *
 * Look up a symbol in the symbol table.   This uses the C name, not scripting
 * names.   Note: If we come across a using declaration, we follow it to
 * to get the real node. Any using directives are also followed (but this is
 * implemented in symbol_lookup()).
 * ----------------------------------------------------------------------------- */

Node *Swig_symbol_clookup(const_String_or_char_ptr name, Symtab *n) {
  Hash *hsym = 0;
  Node *s = 0;

  if (!n) {
    hsym = current_symtab;
  } else {
    if (!Checkattr(n, "nodeType", "symboltable")) {
      n = Getattr(n, "sym:symtab");
    }
    assert(n);
    if (n) {
      hsym = n;
    }
  }

  if (Swig_scopename_check(name)) {
    char *cname = Char(name);
    if (strncmp(cname, "::", 2) == 0) {
      String *nname = NewString(cname + 2);
      if (Swig_scopename_check(nname)) {
	s = symbol_lookup_qualified(nname, global_scope, 0, 0, 0);
      } else {
	s = symbol_lookup(nname, global_scope, 0);
      }
      Delete(nname);
    } else {
      String *prefix = Swig_scopename_prefix(name);
      if (prefix) {
	s = symbol_lookup_qualified(name, hsym, 0, 0, 0);
	Delete(prefix);
	if (!s) {
	  return 0;
	}
      }
    }
  }
  if (!s) {
    while (hsym) {
      s = symbol_lookup(name, hsym, 0);
      if (s)
	break;
      hsym = Getattr(hsym, "parentNode");
      if (!hsym)
	break;
    }
  }

  if (!s) {
    return 0;
  }
  /* Check if s is a 'using' node */
  while (s && Checkattr(s, "nodeType", "using")) {
    String *uname = Getattr(s, "uname");
    Symtab *un = Getattr(s, "sym:symtab");
    Node *ss = (!Equal(name, uname) || (un != n)) ? Swig_symbol_clookup(uname, un) : 0;	/* avoid infinity loop */
    if (!ss) {
      Swig_warning(WARN_PARSE_USING_UNDEF, Getfile(s), Getline(s), "Nothing known about '%s'.\n", SwigType_namestr(Getattr(s, "uname")));
    }
    s = ss;
  }
  return s;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_clookup_check()
 *
 * This function is identical to Swig_symbol_clookup() except that it
 * accepts a callback function that is invoked to determine a symbol match.
 * The purpose of this function is to support complicated algorithms that need
 * to examine multiple definitions of the same symbol that might appear in an
 * inheritance hierarchy. 
 * ----------------------------------------------------------------------------- */

Node *Swig_symbol_clookup_check(const_String_or_char_ptr name, Symtab *n, int (*checkfunc) (Node *n)) {
  Hash *hsym = 0;
  Node *s = 0;

  if (!n) {
    hsym = current_symtab;
  } else {
    if (!Checkattr(n, "nodeType", "symboltable")) {
      n = Getattr(n, "sym:symtab");
    }
    assert(n);
    if (n) {
      hsym = n;
    }
  }

  if (Swig_scopename_check(name)) {
    char *cname = Char(name);
    if (strncmp(cname, "::", 2) == 0) {
      String *nname = NewString(cname + 2);
      if (Swig_scopename_check(nname)) {
	s = symbol_lookup_qualified(nname, global_scope, 0, 0, checkfunc);
      } else {
	s = symbol_lookup(nname, global_scope, checkfunc);
      }
      Delete(nname);
    } else {
      String *prefix = Swig_scopename_prefix(name);
      if (prefix) {
	s = symbol_lookup_qualified(name, hsym, 0, 0, checkfunc);
	Delete(prefix);
	if (!s) {
	  return 0;
	}
      }
    }
  }
  if (!s) {
    while (hsym) {
      s = symbol_lookup(name, hsym, checkfunc);
      if (s)
	break;
      hsym = Getattr(hsym, "parentNode");
      if (!hsym)
	break;
    }
  }
  if (!s) {
    return 0;
  }
  /* Check if s is a 'using' node */
  while (s && Checkattr(s, "nodeType", "using")) {
    Node *ss;
    ss = Swig_symbol_clookup(Getattr(s, "uname"), Getattr(s, "sym:symtab"));
    if (!ss && !checkfunc) {
      Swig_warning(WARN_PARSE_USING_UNDEF, Getfile(s), Getline(s), "Nothing known about '%s'.\n", SwigType_namestr(Getattr(s, "uname")));
    }
    s = ss;
  }
  return s;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_clookup_local()
 *
 * Same as Swig_symbol_clookup but parent nodes are not searched, that is, just
 * this symbol table is searched.
 * ----------------------------------------------------------------------------- */

Node *Swig_symbol_clookup_local(const_String_or_char_ptr name, Symtab *n) {
  Hash *hsym;
  Node *s = 0;

  if (!n) {
    hsym = current_symtab;
  } else {
    if (!Checkattr(n, "nodeType", "symboltable")) {
      n = Getattr(n, "sym:symtab");
    }
    assert(n);
    hsym = n;
  }

  if (Swig_scopename_check(name)) {
    char *cname = Char(name);
    if (strncmp(cname, "::", 2) == 0) {
      String *nname = NewString(cname + 2);
      if (Swig_scopename_check(nname)) {
	s = symbol_lookup_qualified(nname, global_scope, 0, 0, 0);
      } else {
	s = symbol_lookup(nname, global_scope, 0);
      }
      Delete(nname);
    } else {
      s = symbol_lookup_qualified(name, hsym, 0, 0, 0);
    }
  }
  if (!s) {
    s = symbol_lookup(name, hsym, 0);
  }
  if (!s)
    return 0;
  /* Check if s is a 'using' node */
  while (s && Checkattr(s, "nodeType", "using")) {
    Node *ss = Swig_symbol_clookup_local(Getattr(s, "uname"), Getattr(s, "sym:symtab"));
    if (!ss) {
      Swig_warning(WARN_PARSE_USING_UNDEF, Getfile(s), Getline(s), "Nothing known about '%s'.\n", SwigType_namestr(Getattr(s, "uname")));
    }
    s = ss;
  }
  return s;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_clookup_local_check()
 * ----------------------------------------------------------------------------- */

Node *Swig_symbol_clookup_local_check(const_String_or_char_ptr name, Symtab *n, int (*checkfunc) (Node *)) {
  Hash *hsym;
  Node *s = 0;

  if (!n) {
    hsym = current_symtab;
  } else {
    if (!Checkattr(n, "nodeType", "symboltable")) {
      n = Getattr(n, "sym:symtab");
    }
    assert(n);
    hsym = n;
  }

  if (Swig_scopename_check(name)) {
    char *cname = Char(name);
    if (strncmp(cname, "::", 2) == 0) {
      String *nname = NewString(cname + 2);
      if (Swig_scopename_check(nname)) {
	s = symbol_lookup_qualified(nname, global_scope, 0, 0, checkfunc);
      } else {
	s = symbol_lookup(nname, global_scope, checkfunc);
      }
      Delete(nname);
    } else {
      s = symbol_lookup_qualified(name, hsym, 0, 0, checkfunc);
    }
  }
  if (!s) {
    s = symbol_lookup(name, hsym, checkfunc);
  }
  if (!s)
    return 0;
  /* Check if s is a 'using' node */
  while (s && Checkattr(s, "nodeType", "using")) {
    Node *ss = Swig_symbol_clookup_local_check(Getattr(s, "uname"), Getattr(s, "sym:symtab"), checkfunc);
    if (!ss && !checkfunc) {
      Swig_warning(WARN_PARSE_USING_UNDEF, Getfile(s), Getline(s), "Nothing known about '%s'.\n", SwigType_namestr(Getattr(s, "uname")));
    }
    s = ss;
  }
  return s;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_clookup_no_inherit()
 *
 * Symbol lookup like Swig_symbol_clookup but does not follow using declarations.
 * ----------------------------------------------------------------------------- */

Node *Swig_symbol_clookup_no_inherit(const_String_or_char_ptr name, Symtab *n) {
  Node *s = 0;
  assert(use_inherit==1);
  use_inherit = 0;
  s = Swig_symbol_clookup(name, n);
  use_inherit = 1;
  return s;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_cscope()
 *
 * Look up a scope name.
 * ----------------------------------------------------------------------------- */

Symtab *Swig_symbol_cscope(const_String_or_char_ptr name, Symtab *symtab) {
  char *cname = Char(name);
  if (strncmp(cname, "::", 2) == 0)
    return symbol_lookup_qualified(0, global_scope, name, 0, 0);
  return symbol_lookup_qualified(0, symtab, name, 0, 0);
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_remove()
 *
 * Remove a symbol. If the symbol is an overloaded function and the symbol removed
 * is not the last in the list of overloaded functions, then the overloaded
 * names (sym:overname attribute) are changed to start from zero, eg __SWIG_0.
 * ----------------------------------------------------------------------------- */

void Swig_symbol_remove(Node *n) {
  Symtab *symtab;
  String *symname;
  String *overname;
  Node *symprev;
  Node *symnext;
  Node *fixovername = 0;
  symtab = Getattr(n, "sym:symtab");	/* Get symbol table object */
  symtab = Getattr(symtab, "symtab");	/* Get actual hash table of symbols */
  symname = Getattr(n, "sym:name");
  symprev = Getattr(n, "sym:previousSibling");
  symnext = Getattr(n, "sym:nextSibling");

  /* If previous symbol, just fix the links */
  if (symprev) {
    if (symnext) {
      Setattr(symprev, "sym:nextSibling", symnext);
      fixovername = symprev;	/* fix as symbol to remove is somewhere in the middle of the linked list */
    } else {
      Delattr(symprev, "sym:nextSibling");
    }
  } else {
    /* If no previous symbol, see if there is a next symbol */
    if (symnext) {
      Setattr(symtab, symname, symnext);
      fixovername = symnext;	/* fix as symbol to remove is at head of linked list */
    } else {
      if (symname)
	Delattr(symtab, symname);
    }
  }
  if (symnext) {
    if (symprev) {
      Setattr(symnext, "sym:previousSibling", symprev);
    } else {
      Delattr(symnext, "sym:previousSibling");
    }
  }
  Delattr(n, "sym:symtab");
  Delattr(n, "sym:previousSibling");
  Delattr(n, "sym:nextSibling");
  Delattr(n, "csym:nextSibling");
  Delattr(n, "sym:overname");
  Delattr(n, "csym:previousSibling");
  Delattr(n, "sym:overloaded");
  n = 0;

  if (fixovername) {
    Node *nn = fixovername;
    Node *head = fixovername;
    int pn = 0;

    /* find head of linked list */
    while (nn) {
      head = nn;
      nn = Getattr(nn, "sym:previousSibling");
    }

    /* adjust all the sym:overname strings to start from 0 and increment by one */
    nn = head;
    while (nn) {
      assert(Getattr(nn, "sym:overname"));
      Delattr(nn, "sym:overname");
      overname = NewStringf("__SWIG_%d", pn);
      Setattr(nn, "sym:overname", overname);
      Delete(overname);
      pn++;
      nn = Getattr(nn, "sym:nextSibling");
    }
  }
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_qualified()
 *
 * Return the qualified name of a symbol
 * ----------------------------------------------------------------------------- */

String *Swig_symbol_qualified(Node *n) {
  Hash *symtab;
  if (Checkattr(n, "nodeType", "symboltable")) {
    symtab = n;
  } else {
    symtab = Getattr(n, "sym:symtab");
  }
  if (!symtab)
    return NewStringEmpty();
#ifdef SWIG_DEBUG
  Printf(stderr, "symbol_qscope %s %p %s\n", Getattr(n, "name"), symtab, Getattr(symtab, "name"));
#endif
  return Swig_symbol_qualifiedscopename(symtab);
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_isoverloaded()
 * 
 * Check if a symbol is overloaded.  Returns the first symbol if so.
 * ----------------------------------------------------------------------------- */

Node *Swig_symbol_isoverloaded(Node *n) {
  return Getattr(n, "sym:overloaded");
}

/* -----------------------------------------------------------------------------
 * symbol_template_qualify()
 *
 * Internal function to create a fully qualified type name for templates
 * ----------------------------------------------------------------------------- */

/* This cache produces problems with OSS, don't active it */
/* #define SWIG_TEMPLATE_QUALIFY_CACHE */
static SwigType *symbol_template_qualify(const SwigType *e, Symtab *st) {
  String *tprefix, *tsuffix;
  SwigType *qprefix;
  List *targs;
  Node *tempn;
  Symtab *tscope;
  Iterator ti;
#ifdef SWIG_TEMPLATE_QUALIFY_CACHE
  static Hash *qualify_cache = 0;
  String *scopetype = st ? NewStringf("%s::%s", Getattr(st, "name"), e)
      : NewStringf("%s::%s", Swig_symbol_getscopename(), e);
  if (!qualify_cache) {
    qualify_cache = NewHash();
  }
  if (scopetype) {
    String *cres = Getattr(qualify_cache, scopetype);
    if (cres) {
      Delete(scopetype);
      return Copy(cres);
    }
  }
#endif

  tprefix = SwigType_templateprefix(e);
  tsuffix = SwigType_templatesuffix(e);
  qprefix = Swig_symbol_type_qualify(tprefix, st);
  targs = SwigType_parmlist(e);
  tempn = Swig_symbol_clookup_local(tprefix, st);
  tscope = tempn ? Getattr(tempn, "sym:symtab") : 0;
  Append(qprefix, "<(");
  for (ti = First(targs); ti.item;) {
    String *vparm;
    String *qparm = Swig_symbol_type_qualify(ti.item, st);
    if (tscope && (tscope != st)) {
      String *ty = Swig_symbol_type_qualify(qparm, tscope);
      Delete(qparm);
      qparm = ty;
    }

    vparm = Swig_symbol_template_param_eval(qparm, st);
    Append(qprefix, vparm);
    ti = Next(ti);
    if (ti.item) {
      Putc(',', qprefix);
    }
    Delete(qparm);
    Delete(vparm);
  }
  Append(qprefix, ")>");
  Append(qprefix, tsuffix);
  Delete(tprefix);
  Delete(tsuffix);
  Delete(targs);
#ifdef SWIG_DEBUG
  Printf(stderr, "symbol_temp_qual %s %s\n", e, qprefix);
#endif
#ifdef SWIG_TEMPLATE_QUALIFY_CACHE
  Setattr(qualify_cache, scopetype, qprefix);
  Delete(scopetype);
#endif

  return qprefix;
}


static int symbol_no_constructor(Node *n) {
  return !Checkattr(n, "nodeType", "constructor");
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_type_qualify()
 *
 * Create a fully qualified type name
 * Note: Does not resolve a constructor if passed in as the 'type'.
 * ----------------------------------------------------------------------------- */

SwigType *Swig_symbol_type_qualify(const SwigType *t, Symtab *st) {
  List *elements;
  String *result = NewStringEmpty();
  int i, len;
  char *c = Char(t);
  if (strncmp(c, "::", 2) == 0) {
    Append(result, t);
    return result;
  }

  elements = SwigType_split(t);

  len = Len(elements);
  for (i = 0; i < len; i++) {
    String *e = Getitem(elements, i);
    if (SwigType_issimple(e)) {
      /* Note: the unary scope operator (::) is being removed from the template parameters here. */
      Node *n = Swig_symbol_clookup_check(e, st, symbol_no_constructor);
      if (n) {
	String *name = Getattr(n, "name");
	Clear(e);
	Append(e, name);
#ifdef SWIG_DEBUG
	Printf(stderr, "symbol_qual_ei %d %s %s %p\n", i, name, e, st);
#endif
	if (!Swig_scopename_check(name)) {
	  String *qname = Swig_symbol_qualified(n);
	  if (qname && Len(qname)) {
	    Insert(e, 0, "::");
	    Insert(e, 0, qname);
	  }
#ifdef SWIG_DEBUG
	  Printf(stderr, "symbol_qual_sc %d %s %s %p\n", i, qname, e, st);
#endif
	  Delete(qname);
	}
      } else if (SwigType_istemplate(e)) {
	SwigType *ty = symbol_template_qualify(e, st);
	Clear(e);
	Append(e, ty);
	Delete(ty);
      }
      if (strncmp(Char(e), "::", 2) == 0) {
	Delitem(e, 0);
	Delitem(e, 0);
      }
      Append(result, e);
    } else if (SwigType_isfunction(e)) {
      List *parms = SwigType_parmlist(e);
      String *s = NewString("f(");
      Iterator pi = First(parms);
      while (pi.item) {
	String *pf = Swig_symbol_type_qualify(pi.item, st);
	Append(s, pf);
	pi = Next(pi);
	if (pi.item) {
	  Append(s, ",");
	}
	Delete(pf);
      }
      Append(s, ").");
      Append(result, s);
      Delete(parms);
      Delete(s);
    } else {
      Append(result, e);
    }
  }
  Delete(elements);
#ifdef SWIG_DEBUG
  Printf(stderr, "symbol_qualify %s %s %p %s\n", t, result, st, st ? Getattr(st, "name") : 0);
#endif

  return result;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_template_reduce()
 * Resolves template parameter types
 * For example:
 *   typedef int Int;
 *   typedef Int Integer;
 * with input:
 *   Foo<(Int,Integer)> 
 * returns:
 *   Foo<(int,int)>
 * ----------------------------------------------------------------------------- */

static
SwigType *Swig_symbol_template_reduce(SwigType *qt, Symtab *ntab) {
  Parm *p;
  String *templateargs = SwigType_templateargs(qt);
  List *parms = SwigType_parmlist(templateargs);
  Iterator pi = First(parms);
  String *tprefix = SwigType_templateprefix(qt);
  String *tsuffix = SwigType_templatesuffix(qt);
  String *qprefix = SwigType_typedef_qualified(tprefix);
  Append(qprefix, "<(");
  while ((p = pi.item)) {
    String *np;
    String *tp = Swig_symbol_typedef_reduce(p, ntab);
    String *qp = Swig_symbol_type_qualify(tp, ntab);
    Node *n = Swig_symbol_clookup(qp, ntab);
    if (n) {
      String *qual = Swig_symbol_qualified(n);
      np = Copy(Getattr(n, "name"));
      Delete(tp);
      tp = np;
      if (qual && Len(qual)) {
	Insert(np, 0, "::");
	Insert(np, 0, qual);
      }
      Delete(qual);
    } else {
      np = qp;
    }
    Append(qprefix, np);
    pi = Next(pi);
    if (pi.item) {
      Append(qprefix, ",");
    }
    Delete(qp);
    Delete(tp);
  }
  Append(qprefix, ")>");
  Append(qprefix, tsuffix);
  Delete(parms);
  Delete(tprefix);
  Delete(tsuffix);
  Delete(templateargs);
  return qprefix;
}


/* -----------------------------------------------------------------------------
 * Swig_symbol_typedef_reduce()
 *
 * Chase a typedef through symbol tables looking for a match.
 * ----------------------------------------------------------------------------- */

SwigType *Swig_symbol_typedef_reduce(const SwigType *ty, Symtab *tab) {
  SwigType *prefix, *base;
  Node *n;
  String *nt;

  base = SwigType_base(ty);
  prefix = SwigType_prefix(ty);

  n = Swig_symbol_clookup(base, tab);
  if (!n) {
    if (SwigType_istemplate(ty)) {
      SwigType *qt = Swig_symbol_template_reduce(base, tab);
      Append(prefix, qt);
      Delete(qt);
#ifdef SWIG_DEBUG
      Printf(stderr, "symbol_reduce (a) %s %s\n", ty, prefix);
#endif
      Delete(base);
      return prefix;
    } else {
      Delete(prefix);
#ifdef SWIG_DEBUG
      Printf(stderr, "symbol_reduce (b) %s %s\n", ty, ty);
#endif
      return Copy(ty);
    }
  }
  nt = Getattr(n, "nodeType");
  if (Equal(nt, "using")) {
    String *uname = Getattr(n, "uname");
    if (uname) {
      n = Swig_symbol_clookup(base, Getattr(n, "sym:symtab"));
      if (!n) {
	Delete(base);
	Delete(prefix);
#ifdef SWIG_DEBUG
	Printf(stderr, "symbol_reduce (c) %s %s\n", ty, ty);
#endif
	return Copy(ty);
      }
    }
  }
  if (Equal(nt, "cdecl")) {
    String *storage = Getattr(n, "storage");
    if (storage && (Equal(storage, "typedef"))) {
      SwigType *decl;
      SwigType *rt;
      SwigType *qt;
      Symtab *ntab;
      SwigType *nt = Copy(Getattr(n, "type"));

      /* Fix for case 'typedef struct Hello hello;' */
      {
	const char *dclass[3] = { "struct ", "union ", "class " };
	int i;
	char *c = Char(nt);
	for (i = 0; i < 3; i++) {
	  if (strstr(c, dclass[i]) == c) {
	    Replace(nt, dclass[i], "", DOH_REPLACE_FIRST);
	  }
	}
      }
      decl = Getattr(n, "decl");
      if (decl) {
	SwigType_push(nt, decl);
      }
      SwigType_push(nt, prefix);
      Delete(base);
      Delete(prefix);
      ntab = Getattr(n, "sym:symtab");
      rt = Swig_symbol_typedef_reduce(nt, ntab);
      qt = Swig_symbol_type_qualify(rt, ntab);
      if (SwigType_istemplate(qt)) {
	SwigType *qtr = Swig_symbol_template_reduce(qt, ntab);
	Delete(qt);
	qt = qtr;
      }
      Delete(nt);
      Delete(rt);
#ifdef SWIG_DEBUG
      Printf(stderr, "symbol_reduce (d) %s %s\n", qt, ty);
#endif
      return qt;
    }
  }
  Delete(base);
  Delete(prefix);
#ifdef SWIG_DEBUG
  Printf(stderr, "symbol_reduce (e) %s %s\n", ty, ty);
#endif
  return Copy(ty);
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_string_qualify()
 *
 * This function takes a string and looks for identifiers.  Identifiers are
 * then qualified according to scope rules.  This function is used in a number
 * of settings including expression evaluation, scoping of conversion operators,
 * and so forth.
 * ----------------------------------------------------------------------------- */

String *Swig_symbol_string_qualify(String *s, Symtab *st) {
  int have_id = 0;
  String *id = NewStringEmpty();
  String *r = NewStringEmpty();
  char *c = Char(s);
  int first_char = 1;
  while (*c) {
    if (isalpha((int) *c) || (*c == '_') || (*c == ':') || (*c == '~' && first_char) || (isdigit((int) *c) && !first_char)) {
      Putc(*c, id);
      have_id = 1;
    } else {
      if (have_id) {
	String *qid = Swig_symbol_type_qualify(id, st);
	Append(r, qid);
	Clear(id);
	Delete(qid);
	have_id = 0;
      }
      Putc(*c, r);
    }
    first_char = (*c == ':');
    c++;
  }
  if (have_id) {
    String *qid = Swig_symbol_type_qualify(id, st);
    Append(r, qid);
    Delete(qid);
  }
  Delete(id);
  return r;
}


/* -----------------------------------------------------------------------------
 * Swig_symbol_template_defargs()
 *
 * Apply default arg from generic template default args 
 * Returns a parameter list which contains missing default arguments (if any)
 * Note side effects: parms will also contain the extra parameters in its list
 * (but only if non-zero).
 * ----------------------------------------------------------------------------- */


ParmList *Swig_symbol_template_defargs(Parm *parms, Parm *targs, Symtab *tscope, Symtab *tsdecl) {
  ParmList *expandedparms = parms;
  if (Len(parms) < Len(targs)) {
    Parm *lp = parms;
    Parm *p = lp;
    Parm *tp = targs;
    while (p && tp) {
      p = nextSibling(p);
      tp = nextSibling(tp);
      if (p)
	lp = p;
    }
    while (tp) {
      String *value = Getattr(tp, "value");
      if (value) {
	Parm *cp;
	Parm *ta = targs;
	Parm *p = parms;
	SwigType *nt = Swig_symbol_string_qualify(value, tsdecl);
	SwigType *ntq = 0;
#ifdef SWIG_DEBUG
	Printf(stderr, "value %s %s %s\n", value, nt, tsdecl ? Getattr(tsdecl, "name") : tsdecl);
#endif
	while (p && ta) {
	  String *name = Getattr(ta, "name");
	  String *pvalue = Getattr(p, "value");
	  String *value = pvalue ? pvalue : Getattr(p, "type");
	  String *ttq = Swig_symbol_type_qualify(value, tscope);
	  /* value = SwigType_typedef_resolve_all(value); */
	  Replaceid(nt, name, ttq);
	  p = nextSibling(p);
	  ta = nextSibling(ta);
	  Delete(ttq);
	}
	ntq = Swig_symbol_type_qualify(nt, tsdecl);
	if (SwigType_istemplate(ntq)) {
	  String *ty = Swig_symbol_template_deftype(ntq, tscope);
	  Delete(ntq);
	  ntq = ty;
	}
	cp = NewParmWithoutFileLineInfo(ntq, 0);
	if (lp) {
	  set_nextSibling(lp, cp);
	  Delete(cp);
	} else {
	  expandedparms = cp;
	}
	lp = cp;
	tp = nextSibling(tp);
	Delete(nt);
	Delete(ntq);
      } else {
	tp = 0;
      }
    }
  }
  return expandedparms;
}

/* -----------------------------------------------------------------------------
 * Swig_symbol_template_deftype()
 *
 * Apply default args to generic template type
 * ----------------------------------------------------------------------------- */

#define SWIG_TEMPLATE_DEFTYPE_CACHE
SwigType *Swig_symbol_template_deftype(const SwigType *type, Symtab *tscope) {
  String *result = NewStringEmpty();
  List *elements = SwigType_split(type);
  int len = Len(elements);
  int i;
#ifdef SWIG_TEMPLATE_DEFTYPE_CACHE
  static Hash *s_cache = 0;
  Hash *scope_cache;
  /* The lookup depends on the current scope and potential namespace qualification.
     Looking up x in namespace y is not the same as looking up x::y in outer scope.
     -> we use a 2-level hash: first scope and then symbol. */
  String *scope_name = tscope
      ? Swig_symbol_qualifiedscopename(tscope)
      : Swig_symbol_qualifiedscopename(current_symtab);
  String *type_name = tscope
      ? NewStringf("%s::%s", Getattr(tscope, "name"), type)
      : NewStringf("%s::%s", Swig_symbol_getscopename(), type);
  if (!scope_name) scope_name = NewString("::");
  if (!s_cache) {
    s_cache = NewHash();
  }
  scope_cache = Getattr(s_cache, scope_name);
  if (scope_cache) {
    String *cres = Getattr(scope_cache, type_name);
    if (cres) {
      Append(result, cres);
#ifdef SWIG_DEBUG
      Printf(stderr, "cached deftype %s(%s) -> %s\n", type, scope_name, result);
#endif
      Delete(type_name);
      Delete(scope_name);
      return result;
    }
  } else {
      scope_cache = NewHash();
      Setattr(s_cache, scope_name, scope_cache);
      Delete(scope_name);
  }
#endif

#ifdef SWIG_DEBUG
  Printf(stderr, "finding deftype %s\n", type);
#endif

  for (i = 0; i < len; i++) {
    String *e = Getitem(elements, i);
    if (SwigType_isfunction(e)) {
      String *s = NewString("f(");
      List *parms = SwigType_parmlist(e);
      Iterator pi = First(parms);
      while (pi.item) {
	String *pf = SwigType_istemplate(e) ? Swig_symbol_template_deftype(pi.item, tscope)
	    : Swig_symbol_type_qualify(pi.item, tscope);
	Append(s, pf);
	pi = Next(pi);
	if (pi.item) {
	  Append(s, ",");
	}
	Delete(pf);
      }
      Append(s, ").");
      Append(result, s);
      Delete(s);
      Delete(parms);
    } else if (SwigType_istemplate(e)) {
      String *prefix = SwigType_prefix(e);
      String *base = SwigType_base(e);
      String *tprefix = SwigType_templateprefix(base);
      String *targs = SwigType_templateargs(base);
      String *tsuffix = SwigType_templatesuffix(base);
      ParmList *tparms = SwigType_function_parms(targs, 0);
      Node *tempn = Swig_symbol_clookup_local(tprefix, tscope);
      if (!tempn && tsuffix && Len(tsuffix)) {
	tempn = Swig_symbol_clookup(tprefix, 0);
      }
#ifdef SWIG_DEBUG
      Printf(stderr, "deftype type %s %s %d\n", e, tprefix, (long) tempn);
#endif
      if (tempn) {
	ParmList *tnargs = Getattr(tempn, "templateparms");
        ParmList *expandedparms;
	Parm *p;
	Symtab *tsdecl = Getattr(tempn, "sym:symtab");

#ifdef SWIG_DEBUG
	Printf(stderr, "deftype type %s %s %s\n", tprefix, targs, tsuffix);
#endif
	Append(tprefix, "<(");
	expandedparms = Swig_symbol_template_defargs(tparms, tnargs, tscope, tsdecl);
	p = expandedparms;
	tscope = tsdecl;
	while (p) {
	  SwigType *ptype = Getattr(p, "type");
	  SwigType *ttr = ptype ? ptype : Getattr(p, "value");
	  SwigType *ttf = Swig_symbol_type_qualify(ttr, tscope);
	  SwigType *ttq = Swig_symbol_template_param_eval(ttf, tscope);
#ifdef SWIG_DEBUG
	  Printf(stderr, "arg type %s\n", ttq);
#endif
	  if (SwigType_istemplate(ttq)) {
	    SwigType *ttd = Swig_symbol_template_deftype(ttq, tscope);
	    Delete(ttq);
	    ttq = ttd;
#ifdef SWIG_DEBUG
	    Printf(stderr, "arg deftype %s\n", ttq);
#endif
	  }
	  Append(tprefix, ttq);
	  p = nextSibling(p);
	  if (p)
	    Putc(',', tprefix);
	  Delete(ttf);
	  Delete(ttq);
	}
	Append(tprefix, ")>");
	Append(tprefix, tsuffix);
	Append(prefix, tprefix);
#ifdef SWIG_DEBUG
	Printf(stderr, "deftype %s %s \n", type, tprefix);
#endif
	Append(result, prefix);
      } else {
	Append(result, e);
      }
      Delete(prefix);
      Delete(base);
      Delete(tprefix);
      Delete(tsuffix);
      Delete(targs);
      Delete(tparms);
    } else {
      Append(result, e);
    }
  }
  Delete(elements);
#ifdef SWIG_TEMPLATE_DEFTYPE_CACHE
  Setattr(scope_cache, type_name, result);
  Delete(type_name);
#endif

  return result;
}

SwigType *Swig_symbol_template_param_eval(const SwigType *p, Symtab *symtab) {
  String *value = Copy(p);
  Node *lastnode = 0;
  while (1) {
    Node *n = Swig_symbol_clookup(value, symtab);
    if (n == lastnode)
      break;
    lastnode = n;
    if (n) {
      String *nt = Getattr(n, "nodeType");
      if (Equal(nt, "enumitem")) {
	/* An enum item.   Generate a fully qualified name */
	String *qn = Swig_symbol_qualified(n);
	if (qn && Len(qn)) {
	  Append(qn, "::");
	  Append(qn, Getattr(n, "name"));
	  Delete(value);
	  value = qn;
	  continue;
	} else {
	  Delete(qn);
	  break;
	}
      } else if ((Equal(nt, "cdecl"))) {
	String *nv = Getattr(n, "value");
	if (nv) {
	  Delete(value);
	  value = Copy(nv);
	  continue;
	}
      }
    }
    break;
  }
  return value;
}
