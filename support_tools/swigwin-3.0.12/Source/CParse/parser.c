/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 25 "parser.y" /* yacc.c:339  */

#define yylex yylex

#include "swig.h"
#include "cparse.h"
#include "preprocessor.h"
#include <ctype.h>

/* We do this for portability */
#undef alloca
#define alloca malloc

/* -----------------------------------------------------------------------------
 *                               Externals
 * ----------------------------------------------------------------------------- */

int  yyparse();

/* NEW Variables */

static Node    *top = 0;      /* Top of the generated parse tree */
static int      unnamed = 0;  /* Unnamed datatype counter */
static Hash    *classes = 0;        /* Hash table of classes */
static Hash    *classes_typedefs = 0; /* Hash table of typedef classes: typedef struct X {...} Y; */
static Symtab  *prev_symtab = 0;
static Node    *current_class = 0;
String  *ModuleName = 0;
static Node    *module_node = 0;
static String  *Classprefix = 0;  
static String  *Namespaceprefix = 0;
static int      inclass = 0;
static Node    *currentOuterClass = 0; /* for nested classes */
static const char *last_cpptype = 0;
static int      inherit_list = 0;
static Parm    *template_parameters = 0;
static int      parsing_template_declaration = 0;
static int      extendmode   = 0;
static int      compact_default_args = 0;
static int      template_reduce = 0;
static int      cparse_externc = 0;
int		ignore_nested_classes = 0;
int		kwargs_supported = 0;
/* -----------------------------------------------------------------------------
 *                            Assist Functions
 * ----------------------------------------------------------------------------- */


 
/* Called by the parser (yyparse) when an error is found.*/
static void yyerror (const char *e) {
  (void)e;
}

/* Copies a node.  Does not copy tree links or symbol table data (except for
   sym:name) */

static Node *copy_node(Node *n) {
  Node *nn;
  Iterator k;
  nn = NewHash();
  Setfile(nn,Getfile(n));
  Setline(nn,Getline(n));
  for (k = First(n); k.key; k = Next(k)) {
    String *ci;
    String *key = k.key;
    char *ckey = Char(key);
    if ((strcmp(ckey,"nextSibling") == 0) ||
	(strcmp(ckey,"previousSibling") == 0) ||
	(strcmp(ckey,"parentNode") == 0) ||
	(strcmp(ckey,"lastChild") == 0)) {
      continue;
    }
    if (Strncmp(key,"csym:",5) == 0) continue;
    /* We do copy sym:name.  For templates */
    if ((strcmp(ckey,"sym:name") == 0) || 
	(strcmp(ckey,"sym:weak") == 0) ||
	(strcmp(ckey,"sym:typename") == 0)) {
      String *ci = Copy(k.item);
      Setattr(nn,key, ci);
      Delete(ci);
      continue;
    }
    if (strcmp(ckey,"sym:symtab") == 0) {
      Setattr(nn,"sym:needs_symtab", "1");
    }
    /* We don't copy any other symbol table attributes */
    if (strncmp(ckey,"sym:",4) == 0) {
      continue;
    }
    /* If children.  We copy them recursively using this function */
    if (strcmp(ckey,"firstChild") == 0) {
      /* Copy children */
      Node *cn = k.item;
      while (cn) {
	Node *copy = copy_node(cn);
	appendChild(nn,copy);
	Delete(copy);
	cn = nextSibling(cn);
      }
      continue;
    }
    /* We don't copy the symbol table.  But we drop an attribute 
       requires_symtab so that functions know it needs to be built */

    if (strcmp(ckey,"symtab") == 0) {
      /* Node defined a symbol table. */
      Setattr(nn,"requires_symtab","1");
      continue;
    }
    /* Can't copy nodes */
    if (strcmp(ckey,"node") == 0) {
      continue;
    }
    if ((strcmp(ckey,"parms") == 0) || (strcmp(ckey,"pattern") == 0) || (strcmp(ckey,"throws") == 0)
	|| (strcmp(ckey,"kwargs") == 0)) {
      ParmList *pl = CopyParmList(k.item);
      Setattr(nn,key,pl);
      Delete(pl);
      continue;
    }
    if (strcmp(ckey,"nested:outer") == 0) { /* don't copy outer classes links, they will be updated later */
      Setattr(nn, key, k.item);
      continue;
    }
    /* defaultargs will be patched back in later in update_defaultargs() */
    if (strcmp(ckey,"defaultargs") == 0) {
      Setattr(nn, "needs_defaultargs", "1");
      continue;
    }
    /* Looks okay.  Just copy the data using Copy */
    ci = Copy(k.item);
    Setattr(nn, key, ci);
    Delete(ci);
  }
  return nn;
}

/* -----------------------------------------------------------------------------
 *                              Variables
 * ----------------------------------------------------------------------------- */

static char  *typemap_lang = 0;    /* Current language setting */

static int cplus_mode  = 0;

/* C++ modes */

#define  CPLUS_PUBLIC    1
#define  CPLUS_PRIVATE   2
#define  CPLUS_PROTECTED 3

/* include types */
static int   import_mode = 0;

void SWIG_typemap_lang(const char *tm_lang) {
  typemap_lang = Swig_copy_string(tm_lang);
}

void SWIG_cparse_set_compact_default_args(int defargs) {
  compact_default_args = defargs;
}

int SWIG_cparse_template_reduce(int treduce) {
  template_reduce = treduce;
  return treduce;  
}

/* -----------------------------------------------------------------------------
 *                           Assist functions
 * ----------------------------------------------------------------------------- */

static int promote_type(int t) {
  if (t <= T_UCHAR || t == T_CHAR) return T_INT;
  return t;
}

/* Perform type-promotion for binary operators */
static int promote(int t1, int t2) {
  t1 = promote_type(t1);
  t2 = promote_type(t2);
  return t1 > t2 ? t1 : t2;
}

static String *yyrename = 0;

/* Forward renaming operator */

static String *resolve_create_node_scope(String *cname);


Hash *Swig_cparse_features(void) {
  static Hash   *features_hash = 0;
  if (!features_hash) features_hash = NewHash();
  return features_hash;
}

/* Fully qualify any template parameters */
static String *feature_identifier_fix(String *s) {
  String *tp = SwigType_istemplate_templateprefix(s);
  if (tp) {
    String *ts, *ta, *tq;
    ts = SwigType_templatesuffix(s);
    ta = SwigType_templateargs(s);
    tq = Swig_symbol_type_qualify(ta,0);
    Append(tp,tq);
    Append(tp,ts);
    Delete(ts);
    Delete(ta);
    Delete(tq);
    return tp;
  } else {
    return NewString(s);
  }
}

static void set_access_mode(Node *n) {
  if (cplus_mode == CPLUS_PUBLIC)
    Setattr(n, "access", "public");
  else if (cplus_mode == CPLUS_PROTECTED)
    Setattr(n, "access", "protected");
  else
    Setattr(n, "access", "private");
}

static void restore_access_mode(Node *n) {
  String *mode = Getattr(n, "access");
  if (Strcmp(mode, "private") == 0)
    cplus_mode = CPLUS_PRIVATE;
  else if (Strcmp(mode, "protected") == 0)
    cplus_mode = CPLUS_PROTECTED;
  else
    cplus_mode = CPLUS_PUBLIC;
}

/* Generate the symbol table name for an object */
/* This is a bit of a mess. Need to clean up */
static String *add_oldname = 0;



static String *make_name(Node *n, String *name,SwigType *decl) {
  String *made_name = 0;
  int destructor = name && (*(Char(name)) == '~');

  if (yyrename) {
    String *s = NewString(yyrename);
    Delete(yyrename);
    yyrename = 0;
    if (destructor  && (*(Char(s)) != '~')) {
      Insert(s,0,"~");
    }
    return s;
  }

  if (!name) return 0;

  if (parsing_template_declaration)
    SetFlag(n, "parsing_template_declaration");
  made_name = Swig_name_make(n, Namespaceprefix, name, decl, add_oldname);
  Delattr(n, "parsing_template_declaration");

  return made_name;
}

/* Generate an unnamed identifier */
static String *make_unnamed() {
  unnamed++;
  return NewStringf("$unnamed%d$",unnamed);
}

/* Return if the node is a friend declaration */
static int is_friend(Node *n) {
  return Cmp(Getattr(n,"storage"),"friend") == 0;
}

static int is_operator(String *name) {
  return Strncmp(name,"operator ", 9) == 0;
}


/* Add declaration list to symbol table */
static int  add_only_one = 0;

static void add_symbols(Node *n) {
  String *decl;
  String *wrn = 0;

  if (inclass && n) {
    cparse_normalize_void(n);
  }
  while (n) {
    String *symname = 0;
    /* for friends, we need to pop the scope once */
    String *old_prefix = 0;
    Symtab *old_scope = 0;
    int isfriend = inclass && is_friend(n);
    int iscdecl = Cmp(nodeType(n),"cdecl") == 0;
    int only_csymbol = 0;
    
    if (inclass) {
      String *name = Getattr(n, "name");
      if (isfriend) {
	/* for friends, we need to add the scopename if needed */
	String *prefix = name ? Swig_scopename_prefix(name) : 0;
	old_prefix = Namespaceprefix;
	old_scope = Swig_symbol_popscope();
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	if (!prefix) {
	  if (name && !is_operator(name) && Namespaceprefix) {
	    String *nname = NewStringf("%s::%s", Namespaceprefix, name);
	    Setattr(n,"name",nname);
	    Delete(nname);
	  }
	} else {
	  Symtab *st = Swig_symbol_getscope(prefix);
	  String *ns = st ? Getattr(st,"name") : prefix;
	  String *base  = Swig_scopename_last(name);
	  String *nname = NewStringf("%s::%s", ns, base);
	  Setattr(n,"name",nname);
	  Delete(nname);
	  Delete(base);
	  Delete(prefix);
	}
	Namespaceprefix = 0;
      } else {
	/* for member functions, we need to remove the redundant
	   class scope if provided, as in
	   
	   struct Foo {
	   int Foo::method(int a);
	   };
	   
	*/
	String *prefix = name ? Swig_scopename_prefix(name) : 0;
	if (prefix) {
	  if (Classprefix && (Equal(prefix,Classprefix))) {
	    String *base = Swig_scopename_last(name);
	    Setattr(n,"name",base);
	    Delete(base);
	  }
	  Delete(prefix);
	}
      }
    }

    if (!isfriend && (inclass || extendmode)) {
      Setattr(n,"ismember","1");
    }

    if (extendmode) {
      if (!Getattr(n, "template"))
        SetFlag(n,"isextendmember");
    }

    if (!isfriend && inclass) {
      if ((cplus_mode != CPLUS_PUBLIC)) {
	only_csymbol = 1;
	if (cplus_mode == CPLUS_PROTECTED) {
	  Setattr(n,"access", "protected");
	  only_csymbol = !Swig_need_protected(n);
	} else {
	  Setattr(n,"access", "private");
	  /* private are needed only when they are pure virtuals - why? */
	  if ((Cmp(Getattr(n,"storage"),"virtual") == 0) && (Cmp(Getattr(n,"value"),"0") == 0)) {
	    only_csymbol = 0;
	  }
	  if (Cmp(nodeType(n),"destructor") == 0) {
	    /* Needed for "unref" feature */
	    only_csymbol = 0;
	  }
	}
      } else {
	  Setattr(n,"access", "public");
      }
    }
    if (Getattr(n,"sym:name")) {
      n = nextSibling(n);
      continue;
    }
    decl = Getattr(n,"decl");
    if (!SwigType_isfunction(decl)) {
      String *name = Getattr(n,"name");
      String *makename = Getattr(n,"parser:makename");
      if (iscdecl) {	
	String *storage = Getattr(n, "storage");
	if (Cmp(storage,"typedef") == 0) {
	  Setattr(n,"kind","typedef");
	} else {
	  SwigType *type = Getattr(n,"type");
	  String *value = Getattr(n,"value");
	  Setattr(n,"kind","variable");
	  if (value && Len(value)) {
	    Setattr(n,"hasvalue","1");
	  }
	  if (type) {
	    SwigType *ty;
	    SwigType *tmp = 0;
	    if (decl) {
	      ty = tmp = Copy(type);
	      SwigType_push(ty,decl);
	    } else {
	      ty = type;
	    }
	    if (!SwigType_ismutable(ty) || (storage && Strstr(storage, "constexpr"))) {
	      SetFlag(n,"hasconsttype");
	      SetFlag(n,"feature:immutable");
	    }
	    if (tmp) Delete(tmp);
	  }
	  if (!type) {
	    Printf(stderr,"notype name %s\n", name);
	  }
	}
      }
      Swig_features_get(Swig_cparse_features(), Namespaceprefix, name, 0, n);
      if (makename) {
	symname = make_name(n, makename,0);
        Delattr(n,"parser:makename"); /* temporary information, don't leave it hanging around */
      } else {
        makename = name;
	symname = make_name(n, makename,0);
      }
      
      if (!symname) {
	symname = Copy(Getattr(n,"unnamed"));
      }
      if (symname) {
	if (parsing_template_declaration)
	  SetFlag(n, "parsing_template_declaration");
	wrn = Swig_name_warning(n, Namespaceprefix, symname,0);
	Delattr(n, "parsing_template_declaration");
      }
    } else {
      String *name = Getattr(n,"name");
      SwigType *fdecl = Copy(decl);
      SwigType *fun = SwigType_pop_function(fdecl);
      if (iscdecl) {	
	Setattr(n,"kind","function");
      }
      
      Swig_features_get(Swig_cparse_features(),Namespaceprefix,name,fun,n);

      symname = make_name(n, name,fun);
      if (parsing_template_declaration)
	SetFlag(n, "parsing_template_declaration");
      wrn = Swig_name_warning(n, Namespaceprefix,symname,fun);
      Delattr(n, "parsing_template_declaration");
      
      Delete(fdecl);
      Delete(fun);
      
    }
    if (!symname) {
      n = nextSibling(n);
      continue;
    }
    if (cparse_cplusplus) {
      String *value = Getattr(n, "value");
      if (value && Strcmp(value, "delete") == 0) {
	/* C++11 deleted definition / deleted function */
        SetFlag(n,"deleted");
        SetFlag(n,"feature:ignore");
      }
    }
    if (only_csymbol || GetFlag(n,"feature:ignore") || strncmp(Char(symname),"$ignore",7) == 0) {
      /* Only add to C symbol table and continue */
      Swig_symbol_add(0, n);
      if (!only_csymbol && !GetFlag(n, "feature:ignore")) {
	/* Print the warning attached to $ignore name, if any */
        char *c = Char(symname) + 7;
	if (strlen(c)) {
	  SWIG_WARN_NODE_BEGIN(n);
	  Swig_warning(0,Getfile(n), Getline(n), "%s\n",c+1);
	  SWIG_WARN_NODE_END(n);
	}
	/* If the symbol was ignored via "rename" and is visible, set also feature:ignore*/
	SetFlag(n, "feature:ignore");
      }
      if (!GetFlag(n, "feature:ignore") && Strcmp(symname,"$ignore") == 0) {
	/* Add feature:ignore if the symbol was explicitely ignored, regardless of visibility */
	SetFlag(n, "feature:ignore");
      }
    } else {
      Node *c;
      if ((wrn) && (Len(wrn))) {
	String *metaname = symname;
	if (!Getmeta(metaname,"already_warned")) {
	  SWIG_WARN_NODE_BEGIN(n);
	  Swig_warning(0,Getfile(n),Getline(n), "%s\n", wrn);
	  SWIG_WARN_NODE_END(n);
	  Setmeta(metaname,"already_warned","1");
	}
      }
      c = Swig_symbol_add(symname,n);

      if (c != n) {
        /* symbol conflict attempting to add in the new symbol */
        if (Getattr(n,"sym:weak")) {
          Setattr(n,"sym:name",symname);
        } else {
          String *e = NewStringEmpty();
          String *en = NewStringEmpty();
          String *ec = NewStringEmpty();
          int redefined = Swig_need_redefined_warn(n,c,inclass);
          if (redefined) {
            Printf(en,"Identifier '%s' redefined (ignored)",symname);
            Printf(ec,"previous definition of '%s'",symname);
          } else {
            Printf(en,"Redundant redeclaration of '%s'",symname);
            Printf(ec,"previous declaration of '%s'",symname);
          }
          if (Cmp(symname,Getattr(n,"name"))) {
            Printf(en," (Renamed from '%s')", SwigType_namestr(Getattr(n,"name")));
          }
          Printf(en,",");
          if (Cmp(symname,Getattr(c,"name"))) {
            Printf(ec," (Renamed from '%s')", SwigType_namestr(Getattr(c,"name")));
          }
          Printf(ec,".");
	  SWIG_WARN_NODE_BEGIN(n);
          if (redefined) {
            Swig_warning(WARN_PARSE_REDEFINED,Getfile(n),Getline(n),"%s\n",en);
            Swig_warning(WARN_PARSE_REDEFINED,Getfile(c),Getline(c),"%s\n",ec);
          } else if (!is_friend(n) && !is_friend(c)) {
            Swig_warning(WARN_PARSE_REDUNDANT,Getfile(n),Getline(n),"%s\n",en);
            Swig_warning(WARN_PARSE_REDUNDANT,Getfile(c),Getline(c),"%s\n",ec);
          }
	  SWIG_WARN_NODE_END(n);
          Printf(e,"%s:%d:%s\n%s:%d:%s\n",Getfile(n),Getline(n),en,
                 Getfile(c),Getline(c),ec);
          Setattr(n,"error",e);
	  Delete(e);
          Delete(en);
          Delete(ec);
        }
      }
    }
    /* restore the class scope if needed */
    if (isfriend) {
      Swig_symbol_setscope(old_scope);
      if (old_prefix) {
	Delete(Namespaceprefix);
	Namespaceprefix = old_prefix;
      }
    }
    Delete(symname);

    if (add_only_one) return;
    n = nextSibling(n);
  }
}


/* add symbols a parse tree node copy */

static void add_symbols_copy(Node *n) {
  String *name;
  int    emode = 0;
  while (n) {
    char *cnodeType = Char(nodeType(n));

    if (strcmp(cnodeType,"access") == 0) {
      String *kind = Getattr(n,"kind");
      if (Strcmp(kind,"public") == 0) {
	cplus_mode = CPLUS_PUBLIC;
      } else if (Strcmp(kind,"private") == 0) {
	cplus_mode = CPLUS_PRIVATE;
      } else if (Strcmp(kind,"protected") == 0) {
	cplus_mode = CPLUS_PROTECTED;
      }
      n = nextSibling(n);
      continue;
    }

    add_oldname = Getattr(n,"sym:name");
    if ((add_oldname) || (Getattr(n,"sym:needs_symtab"))) {
      int old_inclass = -1;
      Node *old_current_class = 0;
      if (add_oldname) {
	DohIncref(add_oldname);
	/*  Disable this, it prevents %rename to work with templates */
	/* If already renamed, we used that name  */
	/*
	if (Strcmp(add_oldname, Getattr(n,"name")) != 0) {
	  Delete(yyrename);
	  yyrename = Copy(add_oldname);
	}
	*/
      }
      Delattr(n,"sym:needs_symtab");
      Delattr(n,"sym:name");

      add_only_one = 1;
      add_symbols(n);

      if (Getattr(n,"partialargs")) {
	Swig_symbol_cadd(Getattr(n,"partialargs"),n);
      }
      add_only_one = 0;
      name = Getattr(n,"name");
      if (Getattr(n,"requires_symtab")) {
	Swig_symbol_newscope();
	Swig_symbol_setscopename(name);
	Delete(Namespaceprefix);
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);
      }
      if (strcmp(cnodeType,"class") == 0) {
	old_inclass = inclass;
	inclass = 1;
	old_current_class = current_class;
	current_class = n;
	if (Strcmp(Getattr(n,"kind"),"class") == 0) {
	  cplus_mode = CPLUS_PRIVATE;
	} else {
	  cplus_mode = CPLUS_PUBLIC;
	}
      }
      if (strcmp(cnodeType,"extend") == 0) {
	emode = cplus_mode;
	cplus_mode = CPLUS_PUBLIC;
      }
      add_symbols_copy(firstChild(n));
      if (strcmp(cnodeType,"extend") == 0) {
	cplus_mode = emode;
      }
      if (Getattr(n,"requires_symtab")) {
	Setattr(n,"symtab", Swig_symbol_popscope());
	Delattr(n,"requires_symtab");
	Delete(Namespaceprefix);
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);
      }
      if (add_oldname) {
	Delete(add_oldname);
	add_oldname = 0;
      }
      if (strcmp(cnodeType,"class") == 0) {
	inclass = old_inclass;
	current_class = old_current_class;
      }
    } else {
      if (strcmp(cnodeType,"extend") == 0) {
	emode = cplus_mode;
	cplus_mode = CPLUS_PUBLIC;
      }
      add_symbols_copy(firstChild(n));
      if (strcmp(cnodeType,"extend") == 0) {
	cplus_mode = emode;
      }
    }
    n = nextSibling(n);
  }
}

/* Add in the "defaultargs" attribute for functions in instantiated templates.
 * n should be any instantiated template (class or start of linked list of functions). */
static void update_defaultargs(Node *n) {
  if (n) {
    Node *firstdefaultargs = n;
    update_defaultargs(firstChild(n));
    n = nextSibling(n);
    /* recursively loop through nodes of all types, but all we really need are the overloaded functions */
    while (n) {
      update_defaultargs(firstChild(n));
      if (!Getattr(n, "defaultargs")) {
	if (Getattr(n, "needs_defaultargs")) {
	  Setattr(n, "defaultargs", firstdefaultargs);
	  Delattr(n, "needs_defaultargs");
	} else {
	  firstdefaultargs = n;
	}
      } else {
	/* Functions added in with %extend (for specialized template classes) will already have default args patched up */
	assert(Getattr(n, "defaultargs") == firstdefaultargs);
      }
      n = nextSibling(n);
    }
  }
}

/* Check a set of declarations to see if any are pure-abstract */

static List *pure_abstracts(Node *n) {
  List *abstracts = 0;
  while (n) {
    if (Cmp(nodeType(n),"cdecl") == 0) {
      String *decl = Getattr(n,"decl");
      if (SwigType_isfunction(decl)) {
	String *init = Getattr(n,"value");
	if (Cmp(init,"0") == 0) {
	  if (!abstracts) {
	    abstracts = NewList();
	  }
	  Append(abstracts,n);
	  SetFlag(n,"abstract");
	}
      }
    } else if (Cmp(nodeType(n),"destructor") == 0) {
      if (Cmp(Getattr(n,"value"),"0") == 0) {
	if (!abstracts) {
	  abstracts = NewList();
	}
	Append(abstracts,n);
	SetFlag(n,"abstract");
      }
    }
    n = nextSibling(n);
  }
  return abstracts;
}

/* Make a classname */

static String *make_class_name(String *name) {
  String *nname = 0;
  String *prefix;
  if (Namespaceprefix) {
    nname= NewStringf("%s::%s", Namespaceprefix, name);
  } else {
    nname = NewString(name);
  }
  prefix = SwigType_istemplate_templateprefix(nname);
  if (prefix) {
    String *args, *qargs;
    args   = SwigType_templateargs(nname);
    qargs  = Swig_symbol_type_qualify(args,0);
    Append(prefix,qargs);
    Delete(nname);
    Delete(args);
    Delete(qargs);
    nname = prefix;
  }
  return nname;
}

/* Use typedef name as class name */

static void add_typedef_name(Node *n, Node *declnode, String *oldName, Symtab *cscope, String *scpname) {
  String *class_rename = 0;
  SwigType *decl = Getattr(declnode, "decl");
  if (!decl || !Len(decl)) {
    String *cname;
    String *tdscopename;
    String *class_scope = Swig_symbol_qualifiedscopename(cscope);
    String *name = Getattr(declnode, "name");
    cname = Copy(name);
    Setattr(n, "tdname", cname);
    tdscopename = class_scope ? NewStringf("%s::%s", class_scope, name) : Copy(name);
    class_rename = Getattr(n, "class_rename");
    if (class_rename && (Strcmp(class_rename, oldName) == 0))
      Setattr(n, "class_rename", NewString(name));
    if (!classes_typedefs) classes_typedefs = NewHash();
    if (!Equal(scpname, tdscopename) && !Getattr(classes_typedefs, tdscopename)) {
      Setattr(classes_typedefs, tdscopename, n);
    }
    Setattr(n, "decl", decl);
    Delete(class_scope);
    Delete(cname);
    Delete(tdscopename);
  }
}

/* If the class name is qualified.  We need to create or lookup namespace entries */

static Symtab *set_scope_to_global() {
  Symtab *symtab = Swig_symbol_global_scope();
  Swig_symbol_setscope(symtab);
  return symtab;
}
 
/* Remove the block braces, { and }, if the 'noblock' attribute is set.
 * Node *kw can be either a Hash or Parmlist. */
static String *remove_block(Node *kw, const String *inputcode) {
  String *modified_code = 0;
  while (kw) {
   String *name = Getattr(kw,"name");
   if (name && (Cmp(name,"noblock") == 0)) {
     char *cstr = Char(inputcode);
     int len = Len(inputcode);
     if (len && cstr[0] == '{') {
       --len; ++cstr; 
       if (len && cstr[len - 1] == '}') { --len; }
       /* we now remove the extra spaces */
       while (len && isspace((int)cstr[0])) { --len; ++cstr; }
       while (len && isspace((int)cstr[len - 1])) { --len; }
       modified_code = NewStringWithSize(cstr, len);
       break;
     }
   }
   kw = nextSibling(kw);
  }
  return modified_code;
}


static Node *nscope = 0;
static Node *nscope_inner = 0;

/* Remove the scope prefix from cname and return the base name without the prefix.
 * The scopes required for the symbol name are resolved and/or created, if required.
 * For example AA::BB::CC as input returns CC and creates the namespace AA then inner 
 * namespace BB in the current scope. If cname is found to already exist as a weak symbol
 * (forward reference) then the scope might be changed to match, such as when a symbol match 
 * is made via a using reference. */
static String *resolve_create_node_scope(String *cname) {
  Symtab *gscope = 0;
  Node *cname_node = 0;
  int skip_lookup = 0;
  nscope = 0;
  nscope_inner = 0;  

  if (Strncmp(cname,"::",2) == 0)
    skip_lookup = 1;

  cname_node = skip_lookup ? 0 : Swig_symbol_clookup_no_inherit(cname, 0);

  if (cname_node) {
    /* The symbol has been defined already or is in another scope.
       If it is a weak symbol, it needs replacing and if it was brought into the current scope
       via a using declaration, the scope needs adjusting appropriately for the new symbol.
       Similarly for defined templates. */
    Symtab *symtab = Getattr(cname_node, "sym:symtab");
    Node *sym_weak = Getattr(cname_node, "sym:weak");
    if ((symtab && sym_weak) || Equal(nodeType(cname_node), "template")) {
      /* Check if the scope is the current scope */
      String *current_scopename = Swig_symbol_qualifiedscopename(0);
      String *found_scopename = Swig_symbol_qualifiedscopename(symtab);
      int len;
      if (!current_scopename)
	current_scopename = NewString("");
      if (!found_scopename)
	found_scopename = NewString("");
      len = Len(current_scopename);
      if ((len > 0) && (Strncmp(current_scopename, found_scopename, len) == 0)) {
	if (Len(found_scopename) > len + 2) {
	  /* A matching weak symbol was found in non-global scope, some scope adjustment may be required */
	  String *new_cname = NewString(Char(found_scopename) + len + 2); /* skip over "::" prefix */
	  String *base = Swig_scopename_last(cname);
	  Printf(new_cname, "::%s", base);
	  cname = new_cname;
	  Delete(base);
	} else {
	  /* A matching weak symbol was found in the same non-global local scope, no scope adjustment required */
	  assert(len == Len(found_scopename));
	}
      } else {
	String *base = Swig_scopename_last(cname);
	if (Len(found_scopename) > 0) {
	  /* A matching weak symbol was found in a different scope to the local scope - probably via a using declaration */
	  cname = NewStringf("%s::%s", found_scopename, base);
	} else {
	  /* Either:
	      1) A matching weak symbol was found in a different scope to the local scope - this is actually a
	      symbol with the same name in a different scope which we don't want, so no adjustment required.
	      2) A matching weak symbol was found in the global scope - no adjustment required.
	  */
	  cname = Copy(base);
	}
	Delete(base);
      }
      Delete(current_scopename);
      Delete(found_scopename);
    }
  }

  if (Swig_scopename_check(cname)) {
    Node   *ns;
    String *prefix = Swig_scopename_prefix(cname);
    String *base = Swig_scopename_last(cname);
    if (prefix && (Strncmp(prefix,"::",2) == 0)) {
/* I don't think we can use :: global scope to declare classes and hence neither %template. - consider reporting error instead - wsfulton. */
      /* Use the global scope */
      String *nprefix = NewString(Char(prefix)+2);
      Delete(prefix);
      prefix= nprefix;
      gscope = set_scope_to_global();
    }
    if (Len(prefix) == 0) {
      /* Use the global scope, but we need to add a 'global' namespace.  */
      if (!gscope) gscope = set_scope_to_global();
      /* note that this namespace is not the "unnamed" one,
	 and we don't use Setattr(nscope,"name", ""),
	 because the unnamed namespace is private */
      nscope = new_node("namespace");
      Setattr(nscope,"symtab", gscope);;
      nscope_inner = nscope;
      return base;
    }
    /* Try to locate the scope */
    ns = Swig_symbol_clookup(prefix,0);
    if (!ns) {
      Swig_error(cparse_file,cparse_line,"Undefined scope '%s'\n", prefix);
    } else {
      Symtab *nstab = Getattr(ns,"symtab");
      if (!nstab) {
	Swig_error(cparse_file,cparse_line, "'%s' is not defined as a valid scope.\n", prefix);
	ns = 0;
      } else {
	/* Check if the node scope is the current scope */
	String *tname = Swig_symbol_qualifiedscopename(0);
	String *nname = Swig_symbol_qualifiedscopename(nstab);
	if (tname && (Strcmp(tname,nname) == 0)) {
	  ns = 0;
	  cname = base;
	}
	Delete(tname);
	Delete(nname);
      }
      if (ns) {
	/* we will try to create a new node using the namespaces we
	   can find in the scope name */
	List *scopes;
	String *sname;
	Iterator si;
	String *name = NewString(prefix);
	scopes = NewList();
	while (name) {
	  String *base = Swig_scopename_last(name);
	  String *tprefix = Swig_scopename_prefix(name);
	  Insert(scopes,0,base);
	  Delete(base);
	  Delete(name);
	  name = tprefix;
	}
	for (si = First(scopes); si.item; si = Next(si)) {
	  Node *ns1,*ns2;
	  sname = si.item;
	  ns1 = Swig_symbol_clookup(sname,0);
	  assert(ns1);
	  if (Strcmp(nodeType(ns1),"namespace") == 0) {
	    if (Getattr(ns1,"alias")) {
	      ns1 = Getattr(ns1,"namespace");
	    }
	  } else {
	    /* now this last part is a class */
	    si = Next(si);
	    /*  or a nested class tree, which is unrolled here */
	    for (; si.item; si = Next(si)) {
	      if (si.item) {
		Printf(sname,"::%s",si.item);
	      }
	    }
	    /* we get the 'inner' class */
	    nscope_inner = Swig_symbol_clookup(sname,0);
	    /* set the scope to the inner class */
	    Swig_symbol_setscope(Getattr(nscope_inner,"symtab"));
	    /* save the last namespace prefix */
	    Delete(Namespaceprefix);
	    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	    /* and return the node name, including the inner class prefix */
	    break;
	  }
	  /* here we just populate the namespace tree as usual */
	  ns2 = new_node("namespace");
	  Setattr(ns2,"name",sname);
	  Setattr(ns2,"symtab", Getattr(ns1,"symtab"));
	  add_symbols(ns2);
	  Swig_symbol_setscope(Getattr(ns1,"symtab"));
	  Delete(Namespaceprefix);
	  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	  if (nscope_inner) {
	    if (Getattr(nscope_inner,"symtab") != Getattr(ns2,"symtab")) {
	      appendChild(nscope_inner,ns2);
	      Delete(ns2);
	    }
	  }
	  nscope_inner = ns2;
	  if (!nscope) nscope = ns2;
	}
	cname = base;
	Delete(scopes);
      }
    }
    Delete(prefix);
  }

  return cname;
}
 
/* look for simple typedef name in typedef list */
static String *try_to_find_a_name_for_unnamed_structure(const char *storage, Node *decls) {
  String *name = 0;
  Node *n = decls;
  if (storage && (strcmp(storage, "typedef") == 0)) {
    for (; n; n = nextSibling(n)) {
      if (!Len(Getattr(n, "decl"))) {
	name = Copy(Getattr(n, "name"));
	break;
      }
    }
  }
  return name;
}

/* traverse copied tree segment, and update outer class links*/
static void update_nested_classes(Node *n)
{
  Node *c = firstChild(n);
  while (c) {
    if (Getattr(c, "nested:outer"))
      Setattr(c, "nested:outer", n);
    update_nested_classes(c);
    c = nextSibling(c);
  }
}

/* -----------------------------------------------------------------------------
 * nested_forward_declaration()
 * 
 * Nested struct handling for C++ code if the nested classes are disabled.
 * Create the nested class/struct/union as a forward declaration.
 * ----------------------------------------------------------------------------- */

static Node *nested_forward_declaration(const char *storage, const char *kind, String *sname, String *name, Node *cpp_opt_declarators) {
  Node *nn = 0;

  if (sname) {
    /* Add forward declaration of the nested type */
    Node *n = new_node("classforward");
    Setattr(n, "kind", kind);
    Setattr(n, "name", sname);
    Setattr(n, "storage", storage);
    Setattr(n, "sym:weak", "1");
    add_symbols(n);
    nn = n;
  }

  /* Add any variable instances. Also add in any further typedefs of the nested type.
     Note that anonymous typedefs (eg typedef struct {...} a, b;) are treated as class forward declarations */
  if (cpp_opt_declarators) {
    int storage_typedef = (storage && (strcmp(storage, "typedef") == 0));
    int variable_of_anonymous_type = !sname && !storage_typedef;
    if (!variable_of_anonymous_type) {
      int anonymous_typedef = !sname && (storage && (strcmp(storage, "typedef") == 0));
      Node *n = cpp_opt_declarators;
      SwigType *type = name;
      while (n) {
	Setattr(n, "type", type);
	Setattr(n, "storage", storage);
	if (anonymous_typedef) {
	  Setattr(n, "nodeType", "classforward");
	  Setattr(n, "sym:weak", "1");
	}
	n = nextSibling(n);
      }
      add_symbols(cpp_opt_declarators);

      if (nn) {
	set_nextSibling(nn, cpp_opt_declarators);
      } else {
	nn = cpp_opt_declarators;
      }
    }
  }

  if (!currentOuterClass || !GetFlag(currentOuterClass, "nested")) {
    if (nn && Equal(nodeType(nn), "classforward")) {
      Node *n = nn;
      if (!GetFlag(n, "feature:ignore")) {
	SWIG_WARN_NODE_BEGIN(n);
	Swig_warning(WARN_PARSE_NAMED_NESTED_CLASS, cparse_file, cparse_line,"Nested %s not currently supported (%s ignored)\n", kind, sname ? sname : name);
	SWIG_WARN_NODE_END(n);
      }
    } else {
      Swig_warning(WARN_PARSE_UNNAMED_NESTED_CLASS, cparse_file, cparse_line, "Nested %s not currently supported (ignored).\n", kind);
    }
  }

  return nn;
}


Node *Swig_cparse(File *f) {
  scanner_file(f);
  top = 0;
  yyparse();
  return top;
}

static void single_new_feature(const char *featurename, String *val, Hash *featureattribs, char *declaratorid, SwigType *type, ParmList *declaratorparms, String *qualifier) {
  String *fname;
  String *name;
  String *fixname;
  SwigType *t = Copy(type);

  /* Printf(stdout, "single_new_feature: [%s] [%s] [%s] [%s] [%s] [%s]\n", featurename, val, declaratorid, t, ParmList_str_defaultargs(declaratorparms), qualifier); */

  /* Warn about deprecated features */
  if (strcmp(featurename, "nestedworkaround") == 0)
    Swig_warning(WARN_DEPRECATED_NESTED_WORKAROUND, cparse_file, cparse_line, "The 'nestedworkaround' feature is deprecated.\n");

  fname = NewStringf("feature:%s",featurename);
  if (declaratorid) {
    fixname = feature_identifier_fix(declaratorid);
  } else {
    fixname = NewStringEmpty();
  }
  if (Namespaceprefix) {
    name = NewStringf("%s::%s",Namespaceprefix, fixname);
  } else {
    name = fixname;
  }

  if (declaratorparms) Setmeta(val,"parms",declaratorparms);
  if (!Len(t)) t = 0;
  if (t) {
    if (qualifier) SwigType_push(t,qualifier);
    if (SwigType_isfunction(t)) {
      SwigType *decl = SwigType_pop_function(t);
      if (SwigType_ispointer(t)) {
	String *nname = NewStringf("*%s",name);
	Swig_feature_set(Swig_cparse_features(), nname, decl, fname, val, featureattribs);
	Delete(nname);
      } else {
	Swig_feature_set(Swig_cparse_features(), name, decl, fname, val, featureattribs);
      }
      Delete(decl);
    } else if (SwigType_ispointer(t)) {
      String *nname = NewStringf("*%s",name);
      Swig_feature_set(Swig_cparse_features(),nname,0,fname,val, featureattribs);
      Delete(nname);
    }
  } else {
    /* Global feature, that is, feature not associated with any particular symbol */
    Swig_feature_set(Swig_cparse_features(),name,0,fname,val, featureattribs);
  }
  Delete(fname);
  Delete(name);
}

/* Add a new feature to the Hash. Additional features are added if the feature has a parameter list (declaratorparms)
 * and one or more of the parameters have a default argument. An extra feature is added for each defaulted parameter,
 * simulating the equivalent overloaded method. */
static void new_feature(const char *featurename, String *val, Hash *featureattribs, char *declaratorid, SwigType *type, ParmList *declaratorparms, String *qualifier) {

  ParmList *declparms = declaratorparms;

  /* remove the { and } braces if the noblock attribute is set */
  String *newval = remove_block(featureattribs, val);
  val = newval ? newval : val;

  /* Add the feature */
  single_new_feature(featurename, val, featureattribs, declaratorid, type, declaratorparms, qualifier);

  /* Add extra features if there are default parameters in the parameter list */
  if (type) {
    while (declparms) {
      if (ParmList_has_defaultargs(declparms)) {

        /* Create a parameter list for the new feature by copying all
           but the last (defaulted) parameter */
        ParmList* newparms = CopyParmListMax(declparms, ParmList_len(declparms)-1);

        /* Create new declaration - with the last parameter removed */
        SwigType *newtype = Copy(type);
        Delete(SwigType_pop_function(newtype)); /* remove the old parameter list from newtype */
        SwigType_add_function(newtype,newparms);

        single_new_feature(featurename, Copy(val), featureattribs, declaratorid, newtype, newparms, qualifier);
        declparms = newparms;
      } else {
        declparms = 0;
      }
    }
  }
}

/* check if a function declaration is a plain C object */
static int is_cfunction(Node *n) {
  if (!cparse_cplusplus || cparse_externc)
    return 1;
  if (Swig_storage_isexternc(n)) {
    return 1;
  }
  return 0;
}

/* If the Node is a function with parameters, check to see if any of the parameters
 * have default arguments. If so create a new function for each defaulted argument. 
 * The additional functions form a linked list of nodes with the head being the original Node n. */
static void default_arguments(Node *n) {
  Node *function = n;

  if (function) {
    ParmList *varargs = Getattr(function,"feature:varargs");
    if (varargs) {
      /* Handles the %varargs directive by looking for "feature:varargs" and 
       * substituting ... with an alternative set of arguments.  */
      Parm     *p = Getattr(function,"parms");
      Parm     *pp = 0;
      while (p) {
	SwigType *t = Getattr(p,"type");
	if (Strcmp(t,"v(...)") == 0) {
	  if (pp) {
	    ParmList *cv = Copy(varargs);
	    set_nextSibling(pp,cv);
	    Delete(cv);
	  } else {
	    ParmList *cv =  Copy(varargs);
	    Setattr(function,"parms", cv);
	    Delete(cv);
	  }
	  break;
	}
	pp = p;
	p = nextSibling(p);
      }
    }

    /* Do not add in functions if kwargs is being used or if user wants old default argument wrapping
       (one wrapped method per function irrespective of number of default arguments) */
    if (compact_default_args 
	|| is_cfunction(function) 
	|| GetFlag(function,"feature:compactdefaultargs") 
	|| (GetFlag(function,"feature:kwargs") && kwargs_supported)) {
      ParmList *p = Getattr(function,"parms");
      if (p) 
        Setattr(p,"compactdefargs", "1"); /* mark parameters for special handling */
      function = 0; /* don't add in extra methods */
    }
  }

  while (function) {
    ParmList *parms = Getattr(function,"parms");
    if (ParmList_has_defaultargs(parms)) {

      /* Create a parameter list for the new function by copying all
         but the last (defaulted) parameter */
      ParmList* newparms = CopyParmListMax(parms,ParmList_len(parms)-1);

      /* Create new function and add to symbol table */
      {
	SwigType *ntype = Copy(nodeType(function));
	char *cntype = Char(ntype);
        Node *new_function = new_node(ntype);
        SwigType *decl = Copy(Getattr(function,"decl"));
        int constqualifier = SwigType_isconst(decl);
	String *ccode = Copy(Getattr(function,"code"));
	String *cstorage = Copy(Getattr(function,"storage"));
	String *cvalue = Copy(Getattr(function,"value"));
	SwigType *ctype = Copy(Getattr(function,"type"));
	String *cthrow = Copy(Getattr(function,"throw"));

        Delete(SwigType_pop_function(decl)); /* remove the old parameter list from decl */
        SwigType_add_function(decl,newparms);
        if (constqualifier)
          SwigType_add_qualifier(decl,"const");

        Setattr(new_function,"name", Getattr(function,"name"));
        Setattr(new_function,"code", ccode);
        Setattr(new_function,"decl", decl);
        Setattr(new_function,"parms", newparms);
        Setattr(new_function,"storage", cstorage);
        Setattr(new_function,"value", cvalue);
        Setattr(new_function,"type", ctype);
        Setattr(new_function,"throw", cthrow);

	Delete(ccode);
	Delete(cstorage);
	Delete(cvalue);
	Delete(ctype);
	Delete(cthrow);
	Delete(decl);

        {
          Node *throws = Getattr(function,"throws");
	  ParmList *pl = CopyParmList(throws);
          if (throws) Setattr(new_function,"throws",pl);
	  Delete(pl);
        }

        /* copy specific attributes for global (or in a namespace) template functions - these are not templated class methods */
        if (strcmp(cntype,"template") == 0) {
          Node *templatetype = Getattr(function,"templatetype");
          Node *symtypename = Getattr(function,"sym:typename");
          Parm *templateparms = Getattr(function,"templateparms");
          if (templatetype) {
	    Node *tmp = Copy(templatetype);
	    Setattr(new_function,"templatetype",tmp);
	    Delete(tmp);
	  }
          if (symtypename) {
	    Node *tmp = Copy(symtypename);
	    Setattr(new_function,"sym:typename",tmp);
	    Delete(tmp);
	  }
          if (templateparms) {
	    Parm *tmp = CopyParmList(templateparms);
	    Setattr(new_function,"templateparms",tmp);
	    Delete(tmp);
	  }
        } else if (strcmp(cntype,"constructor") == 0) {
          /* only copied for constructors as this is not a user defined feature - it is hard coded in the parser */
          if (GetFlag(function,"feature:new")) SetFlag(new_function,"feature:new");
        }

        add_symbols(new_function);
        /* mark added functions as ones with overloaded parameters and point to the parsed method */
        Setattr(new_function,"defaultargs", n);

        /* Point to the new function, extending the linked list */
        set_nextSibling(function, new_function);
	Delete(new_function);
        function = new_function;
	
	Delete(ntype);
      }
    } else {
      function = 0;
    }
  }
}

/* -----------------------------------------------------------------------------
 * mark_nodes_as_extend()
 *
 * Used by the %extend to mark subtypes with "feature:extend".
 * template instances declared within %extend are skipped
 * ----------------------------------------------------------------------------- */

static void mark_nodes_as_extend(Node *n) {
  for (; n; n = nextSibling(n)) {
    if (Getattr(n, "template") && Strcmp(nodeType(n), "class") == 0)
      continue;
    /* Fix me: extend is not a feature. Replace with isextendmember? */
    Setattr(n, "feature:extend", "1");
    mark_nodes_as_extend(firstChild(n));
  }
}


#line 1397 "y.tab.c" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    ID = 258,
    HBLOCK = 259,
    POUND = 260,
    STRING = 261,
    WSTRING = 262,
    INCLUDE = 263,
    IMPORT = 264,
    INSERT = 265,
    CHARCONST = 266,
    WCHARCONST = 267,
    NUM_INT = 268,
    NUM_FLOAT = 269,
    NUM_UNSIGNED = 270,
    NUM_LONG = 271,
    NUM_ULONG = 272,
    NUM_LONGLONG = 273,
    NUM_ULONGLONG = 274,
    NUM_BOOL = 275,
    TYPEDEF = 276,
    TYPE_INT = 277,
    TYPE_UNSIGNED = 278,
    TYPE_SHORT = 279,
    TYPE_LONG = 280,
    TYPE_FLOAT = 281,
    TYPE_DOUBLE = 282,
    TYPE_CHAR = 283,
    TYPE_WCHAR = 284,
    TYPE_VOID = 285,
    TYPE_SIGNED = 286,
    TYPE_BOOL = 287,
    TYPE_COMPLEX = 288,
    TYPE_TYPEDEF = 289,
    TYPE_RAW = 290,
    TYPE_NON_ISO_INT8 = 291,
    TYPE_NON_ISO_INT16 = 292,
    TYPE_NON_ISO_INT32 = 293,
    TYPE_NON_ISO_INT64 = 294,
    LPAREN = 295,
    RPAREN = 296,
    COMMA = 297,
    SEMI = 298,
    EXTERN = 299,
    INIT = 300,
    LBRACE = 301,
    RBRACE = 302,
    PERIOD = 303,
    CONST_QUAL = 304,
    VOLATILE = 305,
    REGISTER = 306,
    STRUCT = 307,
    UNION = 308,
    EQUAL = 309,
    SIZEOF = 310,
    MODULE = 311,
    LBRACKET = 312,
    RBRACKET = 313,
    BEGINFILE = 314,
    ENDOFFILE = 315,
    ILLEGAL = 316,
    CONSTANT = 317,
    NAME = 318,
    RENAME = 319,
    NAMEWARN = 320,
    EXTEND = 321,
    PRAGMA = 322,
    FEATURE = 323,
    VARARGS = 324,
    ENUM = 325,
    CLASS = 326,
    TYPENAME = 327,
    PRIVATE = 328,
    PUBLIC = 329,
    PROTECTED = 330,
    COLON = 331,
    STATIC = 332,
    VIRTUAL = 333,
    FRIEND = 334,
    THROW = 335,
    CATCH = 336,
    EXPLICIT = 337,
    STATIC_ASSERT = 338,
    CONSTEXPR = 339,
    THREAD_LOCAL = 340,
    DECLTYPE = 341,
    AUTO = 342,
    NOEXCEPT = 343,
    OVERRIDE = 344,
    FINAL = 345,
    USING = 346,
    NAMESPACE = 347,
    NATIVE = 348,
    INLINE = 349,
    TYPEMAP = 350,
    EXCEPT = 351,
    ECHO = 352,
    APPLY = 353,
    CLEAR = 354,
    SWIGTEMPLATE = 355,
    FRAGMENT = 356,
    WARN = 357,
    LESSTHAN = 358,
    GREATERTHAN = 359,
    DELETE_KW = 360,
    DEFAULT = 361,
    LESSTHANOREQUALTO = 362,
    GREATERTHANOREQUALTO = 363,
    EQUALTO = 364,
    NOTEQUALTO = 365,
    ARROW = 366,
    QUESTIONMARK = 367,
    TYPES = 368,
    PARMS = 369,
    NONID = 370,
    DSTAR = 371,
    DCNOT = 372,
    TEMPLATE = 373,
    OPERATOR = 374,
    CONVERSIONOPERATOR = 375,
    PARSETYPE = 376,
    PARSEPARM = 377,
    PARSEPARMS = 378,
    CAST = 379,
    LOR = 380,
    LAND = 381,
    OR = 382,
    XOR = 383,
    AND = 384,
    LSHIFT = 385,
    RSHIFT = 386,
    PLUS = 387,
    MINUS = 388,
    STAR = 389,
    SLASH = 390,
    MODULO = 391,
    UMINUS = 392,
    NOT = 393,
    LNOT = 394,
    DCOLON = 395
  };
#endif
/* Tokens.  */
#define ID 258
#define HBLOCK 259
#define POUND 260
#define STRING 261
#define WSTRING 262
#define INCLUDE 263
#define IMPORT 264
#define INSERT 265
#define CHARCONST 266
#define WCHARCONST 267
#define NUM_INT 268
#define NUM_FLOAT 269
#define NUM_UNSIGNED 270
#define NUM_LONG 271
#define NUM_ULONG 272
#define NUM_LONGLONG 273
#define NUM_ULONGLONG 274
#define NUM_BOOL 275
#define TYPEDEF 276
#define TYPE_INT 277
#define TYPE_UNSIGNED 278
#define TYPE_SHORT 279
#define TYPE_LONG 280
#define TYPE_FLOAT 281
#define TYPE_DOUBLE 282
#define TYPE_CHAR 283
#define TYPE_WCHAR 284
#define TYPE_VOID 285
#define TYPE_SIGNED 286
#define TYPE_BOOL 287
#define TYPE_COMPLEX 288
#define TYPE_TYPEDEF 289
#define TYPE_RAW 290
#define TYPE_NON_ISO_INT8 291
#define TYPE_NON_ISO_INT16 292
#define TYPE_NON_ISO_INT32 293
#define TYPE_NON_ISO_INT64 294
#define LPAREN 295
#define RPAREN 296
#define COMMA 297
#define SEMI 298
#define EXTERN 299
#define INIT 300
#define LBRACE 301
#define RBRACE 302
#define PERIOD 303
#define CONST_QUAL 304
#define VOLATILE 305
#define REGISTER 306
#define STRUCT 307
#define UNION 308
#define EQUAL 309
#define SIZEOF 310
#define MODULE 311
#define LBRACKET 312
#define RBRACKET 313
#define BEGINFILE 314
#define ENDOFFILE 315
#define ILLEGAL 316
#define CONSTANT 317
#define NAME 318
#define RENAME 319
#define NAMEWARN 320
#define EXTEND 321
#define PRAGMA 322
#define FEATURE 323
#define VARARGS 324
#define ENUM 325
#define CLASS 326
#define TYPENAME 327
#define PRIVATE 328
#define PUBLIC 329
#define PROTECTED 330
#define COLON 331
#define STATIC 332
#define VIRTUAL 333
#define FRIEND 334
#define THROW 335
#define CATCH 336
#define EXPLICIT 337
#define STATIC_ASSERT 338
#define CONSTEXPR 339
#define THREAD_LOCAL 340
#define DECLTYPE 341
#define AUTO 342
#define NOEXCEPT 343
#define OVERRIDE 344
#define FINAL 345
#define USING 346
#define NAMESPACE 347
#define NATIVE 348
#define INLINE 349
#define TYPEMAP 350
#define EXCEPT 351
#define ECHO 352
#define APPLY 353
#define CLEAR 354
#define SWIGTEMPLATE 355
#define FRAGMENT 356
#define WARN 357
#define LESSTHAN 358
#define GREATERTHAN 359
#define DELETE_KW 360
#define DEFAULT 361
#define LESSTHANOREQUALTO 362
#define GREATERTHANOREQUALTO 363
#define EQUALTO 364
#define NOTEQUALTO 365
#define ARROW 366
#define QUESTIONMARK 367
#define TYPES 368
#define PARMS 369
#define NONID 370
#define DSTAR 371
#define DCNOT 372
#define TEMPLATE 373
#define OPERATOR 374
#define CONVERSIONOPERATOR 375
#define PARSETYPE 376
#define PARSEPARM 377
#define PARSEPARMS 378
#define CAST 379
#define LOR 380
#define LAND 381
#define OR 382
#define XOR 383
#define AND 384
#define LSHIFT 385
#define RSHIFT 386
#define PLUS 387
#define MINUS 388
#define STAR 389
#define SLASH 390
#define MODULO 391
#define UMINUS 392
#define NOT 393
#define LNOT 394
#define DCOLON 395

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 1356 "parser.y" /* yacc.c:355  */

  const char  *id;
  List  *bases;
  struct Define {
    String *val;
    String *rawval;
    int     type;
    String *qualifier;
    String *bitfield;
    Parm   *throws;
    String *throwf;
    String *nexcept;
  } dtype;
  struct {
    const char *type;
    String *filename;
    int   line;
  } loc;
  struct {
    char      *id;
    SwigType  *type;
    String    *defarg;
    ParmList  *parms;
    short      have_parms;
    ParmList  *throws;
    String    *throwf;
    String    *nexcept;
  } decl;
  Parm         *tparms;
  struct {
    String     *method;
    Hash       *kwargs;
  } tmap;
  struct {
    String     *type;
    String     *us;
  } ptype;
  SwigType     *type;
  String       *str;
  Parm         *p;
  ParmList     *pl;
  int           intvalue;
  Node         *node;

#line 1762 "y.tab.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 1779 "y.tab.c" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  60
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   5039

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  141
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  171
/* YYNRULES -- Number of rules.  */
#define YYNRULES  579
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  1128

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   395

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,  1521,  1521,  1533,  1537,  1540,  1543,  1546,  1549,  1554,
    1559,  1564,  1565,  1566,  1567,  1568,  1578,  1594,  1604,  1605,
    1606,  1607,  1608,  1609,  1610,  1611,  1612,  1613,  1614,  1615,
    1616,  1617,  1618,  1619,  1620,  1621,  1622,  1623,  1624,  1631,
    1631,  1713,  1723,  1734,  1755,  1777,  1788,  1797,  1816,  1822,
    1828,  1833,  1840,  1847,  1851,  1864,  1873,  1888,  1901,  1901,
    1957,  1958,  1965,  1984,  2015,  2019,  2029,  2034,  2052,  2095,
    2101,  2114,  2120,  2146,  2152,  2159,  2160,  2163,  2164,  2171,
    2217,  2263,  2274,  2277,  2304,  2310,  2316,  2322,  2330,  2336,
    2342,  2348,  2356,  2357,  2358,  2361,  2366,  2376,  2412,  2413,
    2448,  2465,  2473,  2486,  2511,  2517,  2521,  2524,  2535,  2540,
    2553,  2565,  2863,  2873,  2880,  2881,  2885,  2885,  2910,  2916,
    2926,  2944,  3012,  3070,  3074,  3097,  3101,  3112,  3119,  3126,
    3133,  3142,  3143,  3144,  3148,  3149,  3150,  3161,  3166,  3171,
    3178,  3184,  3189,  3192,  3192,  3205,  3208,  3211,  3220,  3223,
    3230,  3252,  3281,  3379,  3431,  3432,  3433,  3434,  3435,  3436,
    3441,  3441,  3689,  3689,  3836,  3837,  3849,  3867,  3867,  4128,
    4134,  4140,  4143,  4146,  4149,  4152,  4155,  4158,  4163,  4199,
    4203,  4206,  4209,  4214,  4218,  4223,  4233,  4264,  4264,  4293,
    4293,  4315,  4342,  4359,  4364,  4359,  4372,  4373,  4374,  4374,
    4390,  4391,  4408,  4409,  4410,  4411,  4412,  4413,  4414,  4415,
    4416,  4417,  4418,  4419,  4420,  4421,  4422,  4423,  4424,  4433,
    4461,  4488,  4519,  4534,  4551,  4569,  4588,  4607,  4614,  4621,
    4628,  4636,  4644,  4647,  4651,  4654,  4655,  4656,  4657,  4658,
    4659,  4660,  4661,  4664,  4671,  4678,  4687,  4696,  4705,  4717,
    4720,  4723,  4724,  4728,  4730,  4738,  4750,  4751,  4752,  4753,
    4754,  4755,  4756,  4757,  4758,  4759,  4760,  4761,  4762,  4763,
    4764,  4765,  4766,  4767,  4768,  4769,  4776,  4787,  4791,  4794,
    4798,  4802,  4812,  4820,  4828,  4841,  4845,  4848,  4852,  4856,
    4884,  4892,  4904,  4919,  4929,  4938,  4949,  4953,  4957,  4964,
    4981,  4998,  5006,  5014,  5023,  5032,  5036,  5045,  5056,  5067,
    5079,  5089,  5103,  5111,  5120,  5129,  5133,  5142,  5153,  5164,
    5176,  5186,  5196,  5207,  5220,  5227,  5235,  5251,  5259,  5270,
    5281,  5292,  5311,  5319,  5336,  5344,  5351,  5358,  5369,  5380,
    5391,  5411,  5432,  5438,  5444,  5451,  5458,  5467,  5476,  5479,
    5488,  5497,  5504,  5511,  5518,  5528,  5539,  5550,  5561,  5568,
    5575,  5578,  5595,  5605,  5612,  5618,  5623,  5629,  5633,  5639,
    5640,  5641,  5647,  5653,  5657,  5658,  5662,  5669,  5672,  5673,
    5677,  5678,  5680,  5683,  5686,  5691,  5702,  5727,  5730,  5784,
    5788,  5792,  5796,  5800,  5804,  5808,  5812,  5816,  5820,  5824,
    5828,  5832,  5836,  5842,  5842,  5856,  5861,  5864,  5870,  5883,
    5897,  5898,  5901,  5902,  5906,  5912,  5915,  5919,  5924,  5932,
    5944,  5959,  5960,  5979,  5980,  5984,  5989,  5994,  5995,  6000,
    6013,  6028,  6035,  6052,  6059,  6066,  6073,  6081,  6089,  6093,
    6097,  6103,  6104,  6105,  6106,  6107,  6108,  6109,  6110,  6113,
    6117,  6121,  6125,  6129,  6133,  6137,  6141,  6145,  6149,  6153,
    6157,  6161,  6165,  6179,  6183,  6187,  6193,  6197,  6201,  6205,
    6209,  6225,  6230,  6233,  6238,  6243,  6243,  6244,  6247,  6264,
    6273,  6273,  6291,  6291,  6309,  6310,  6311,  6314,  6318,  6322,
    6326,  6332,  6335,  6339,  6345,  6349,  6353,  6359,  6362,  6367,
    6368,  6371,  6374,  6377,  6380,  6385,  6390,  6395,  6400,  6405,
    6412,  6418,  6422,  6426,  6434,  6442,  6450,  6459,  6468,  6475,
    6484,  6485,  6488,  6489,  6490,  6491,  6494,  6506,  6512,  6521,
    6522,  6523,  6526,  6527,  6528,  6531,  6532,  6535,  6540,  6544,
    6547,  6550,  6553,  6556,  6561,  6565,  6568,  6575,  6581,  6584,
    6589,  6592,  6598,  6603,  6607,  6610,  6613,  6616,  6621,  6625,
    6628,  6631,  6637,  6640,  6643,  6651,  6654,  6657,  6661,  6666,
    6679,  6683,  6688,  6694,  6698,  6703,  6707,  6714,  6717,  6722
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "ID", "HBLOCK", "POUND", "STRING",
  "WSTRING", "INCLUDE", "IMPORT", "INSERT", "CHARCONST", "WCHARCONST",
  "NUM_INT", "NUM_FLOAT", "NUM_UNSIGNED", "NUM_LONG", "NUM_ULONG",
  "NUM_LONGLONG", "NUM_ULONGLONG", "NUM_BOOL", "TYPEDEF", "TYPE_INT",
  "TYPE_UNSIGNED", "TYPE_SHORT", "TYPE_LONG", "TYPE_FLOAT", "TYPE_DOUBLE",
  "TYPE_CHAR", "TYPE_WCHAR", "TYPE_VOID", "TYPE_SIGNED", "TYPE_BOOL",
  "TYPE_COMPLEX", "TYPE_TYPEDEF", "TYPE_RAW", "TYPE_NON_ISO_INT8",
  "TYPE_NON_ISO_INT16", "TYPE_NON_ISO_INT32", "TYPE_NON_ISO_INT64",
  "LPAREN", "RPAREN", "COMMA", "SEMI", "EXTERN", "INIT", "LBRACE",
  "RBRACE", "PERIOD", "CONST_QUAL", "VOLATILE", "REGISTER", "STRUCT",
  "UNION", "EQUAL", "SIZEOF", "MODULE", "LBRACKET", "RBRACKET",
  "BEGINFILE", "ENDOFFILE", "ILLEGAL", "CONSTANT", "NAME", "RENAME",
  "NAMEWARN", "EXTEND", "PRAGMA", "FEATURE", "VARARGS", "ENUM", "CLASS",
  "TYPENAME", "PRIVATE", "PUBLIC", "PROTECTED", "COLON", "STATIC",
  "VIRTUAL", "FRIEND", "THROW", "CATCH", "EXPLICIT", "STATIC_ASSERT",
  "CONSTEXPR", "THREAD_LOCAL", "DECLTYPE", "AUTO", "NOEXCEPT", "OVERRIDE",
  "FINAL", "USING", "NAMESPACE", "NATIVE", "INLINE", "TYPEMAP", "EXCEPT",
  "ECHO", "APPLY", "CLEAR", "SWIGTEMPLATE", "FRAGMENT", "WARN", "LESSTHAN",
  "GREATERTHAN", "DELETE_KW", "DEFAULT", "LESSTHANOREQUALTO",
  "GREATERTHANOREQUALTO", "EQUALTO", "NOTEQUALTO", "ARROW", "QUESTIONMARK",
  "TYPES", "PARMS", "NONID", "DSTAR", "DCNOT", "TEMPLATE", "OPERATOR",
  "CONVERSIONOPERATOR", "PARSETYPE", "PARSEPARM", "PARSEPARMS", "CAST",
  "LOR", "LAND", "OR", "XOR", "AND", "LSHIFT", "RSHIFT", "PLUS", "MINUS",
  "STAR", "SLASH", "MODULO", "UMINUS", "NOT", "LNOT", "DCOLON", "$accept",
  "program", "interface", "declaration", "swig_directive",
  "extend_directive", "$@1", "apply_directive", "clear_directive",
  "constant_directive", "echo_directive", "except_directive", "stringtype",
  "fname", "fragment_directive", "include_directive", "$@2", "includetype",
  "inline_directive", "insert_directive", "module_directive",
  "name_directive", "native_directive", "pragma_directive", "pragma_arg",
  "pragma_lang", "rename_directive", "rename_namewarn",
  "feature_directive", "stringbracesemi", "featattr", "varargs_directive",
  "varargs_parms", "typemap_directive", "typemap_type", "tm_list",
  "tm_tail", "typemap_parm", "types_directive", "template_directive",
  "warn_directive", "c_declaration", "$@3", "c_decl", "c_decl_tail",
  "initializer", "cpp_alternate_rettype", "cpp_lambda_decl",
  "lambda_introducer", "lambda_body", "lambda_tail", "$@4", "c_enum_key",
  "c_enum_inherit", "c_enum_forward_decl", "c_enum_decl",
  "c_constructor_decl", "cpp_declaration", "cpp_class_decl", "@5", "@6",
  "cpp_opt_declarators", "cpp_forward_class_decl", "cpp_template_decl",
  "$@7", "cpp_temp_possible", "template_parms", "templateparameters",
  "templateparameter", "templateparameterstail", "cpp_using_decl",
  "cpp_namespace_decl", "$@8", "$@9", "cpp_members", "$@10", "$@11",
  "$@12", "cpp_member", "cpp_constructor_decl", "cpp_destructor_decl",
  "cpp_conversion_operator", "cpp_catch_decl", "cpp_static_assert",
  "cpp_protection_decl", "cpp_swig_directive", "cpp_end", "cpp_vend",
  "anonymous_bitfield", "anon_bitfield_type", "extern_string",
  "storage_class", "parms", "rawparms", "ptail", "parm", "valparms",
  "rawvalparms", "valptail", "valparm", "def_args", "parameter_declarator",
  "plain_declarator", "declarator", "notso_direct_declarator",
  "direct_declarator", "abstract_declarator", "direct_abstract_declarator",
  "pointer", "type_qualifier", "type_qualifier_raw", "type", "rawtype",
  "type_right", "decltype", "primitive_type", "primitive_type_list",
  "type_specifier", "definetype", "$@13", "default_delete",
  "deleted_definition", "explicit_default", "ename",
  "optional_constant_directive", "enumlist", "edecl", "etype", "expr",
  "valexpr", "exprnum", "exprcompound", "ellipsis", "variadic", "inherit",
  "raw_inherit", "$@14", "base_list", "base_specifier", "@15", "@16",
  "access_specifier", "templcpptype", "cpptype", "classkey", "classkeyopt",
  "opt_virtual", "virt_specifier_seq", "exception_specification",
  "cpp_const", "ctor_end", "ctor_initializer", "mem_initializer_list",
  "mem_initializer", "less_valparms_greater", "identifier", "idstring",
  "idstringopt", "idcolon", "idcolontail", "idtemplate",
  "idtemplatetemplate", "idcolonnt", "idcolontailnt", "string", "wstring",
  "stringbrace", "options", "kwargs", "stringnum", "empty", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395
};
# endif

#define YYPACT_NINF -1018

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-1018)))

#define YYTABLE_NINF -580

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     564,  4133,  4205,   188,    63,  3623, -1018, -1018, -1018, -1018,
   -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018,
   -1018, -1018, -1018, -1018, -1018, -1018,   159, -1018, -1018, -1018,
   -1018, -1018,   262,   257,   264,    32, -1018, -1018,   179,   250,
     256,   292,   382,  4850,   728,   129,   728, -1018, -1018, -1018,
    2504, -1018,   292,   256, -1018,    75, -1018,   396,   406,  4567,
   -1018,   293, -1018, -1018, -1018,   416, -1018, -1018,    36,   446,
    4277,   452, -1018, -1018,   446,   457,   461,   465,   530, -1018,
   -1018,   478,   439,   361,    22,   137,   471,   488,   218,   492,
     498,   551,  4638,  4638,   524,   535,   579,   567,   752, -1018,
   -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018,   446,
   -1018, -1018, -1018, -1018, -1018, -1018, -1018,  1559, -1018, -1018,
   -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018,
   -1018, -1018, -1018, -1018, -1018, -1018, -1018,    31,  4709, -1018,
     562, -1018, -1018,   571,   578,   292,    57,   401,  2151, -1018,
   -1018, -1018,   728, -1018,  3295,   585,    85,  2285,  3089,    33,
    1089,  1235,    69,   292, -1018, -1018,   245,   307,   245,   327,
    1652,   525, -1018, -1018, -1018, -1018, -1018,   207,   305, -1018,
   -1018, -1018,   604, -1018,   614, -1018, -1018,   409, -1018, -1018,
     401,    48,   409,   409, -1018,   620,  1654, -1018,     8,   879,
     316,   207,   207, -1018,   409,  4495, -1018, -1018,  4567, -1018,
   -1018, -1018, -1018, -1018,   292,   298, -1018,   120,   621,   207,
   -1018, -1018,   409,   207, -1018, -1018, -1018,   663,  4567,   626,
     479,   653,   634,   409,   579,   663,  4567,  4567,   292,   579,
    2124,  1870,  1873,   409,   532,   594, -1018, -1018,  1654,   292,
    1672,   217, -1018,   675,   677,   660,   207, -1018, -1018,    75,
     639, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018,
   -1018, -1018,  3089,   427,  3089,  3089,  3089,  3089,  3089,  3089,
    3089, -1018,   654, -1018,   695,   705,   374,  2904,    19, -1018,
   -1018,   663,   753, -1018, -1018,  3408,  1382,  1382,   721,   729,
    1161,   656,   734, -1018, -1018, -1018,   731,  3089, -1018, -1018,
   -1018, -1018,  4281, -1018,  2904,   756,  3408,   769,   292,   370,
     327, -1018,   773,   370,   327, -1018,   714, -1018, -1018,  4567,
    2419, -1018,  4567,  2553,   801,  1336,  1838,   370,   327,   736,
    1779, -1018, -1018,    75,   814,  4567, -1018, -1018, -1018, -1018,
     829,   663,   292, -1018, -1018,    38,   832, -1018, -1018,   387,
     245,   286, -1018,   840, -1018, -1018, -1018, -1018,   292, -1018,
     850,   839,   641,   852,   858, -1018,   861,   865, -1018,  4780,
   -1018,   292, -1018,   863,   868, -1018,   869,   870,  4638, -1018,
   -1018, -1018, -1018, -1018,  4638, -1018, -1018, -1018,   871, -1018,
   -1018,   744,   219,   872,   813, -1018,   883, -1018,    56, -1018,
   -1018,    34,   156,   156,   156,   301,   810,   886,   153,   887,
    1897,  2157,   815,  1779,   822,    49,   859,   357, -1018,  3480,
    1081, -1018,   891, -1018,   359, -1018, -1018, -1018, -1018,   256,
   -1018,   401,  2387,  4780,   892,  3172,  1887, -1018, -1018, -1018,
   -1018, -1018, -1018,  2151, -1018, -1018, -1018,  3089,  3089,  3089,
    3089,  3089,  3089,  3089,  3089,  3089,  3089,  3089,  3089,  3089,
    3089,  3089,  3089,  3089, -1018,   379,   379,   405,   828,    27,
   -1018,   559, -1018, -1018,   379,   379,   633,   830,   156,   156,
    3089,  2904, -1018,  4567,  1814,    18,   899, -1018,  4567,  2687,
     903, -1018,   912, -1018,  4642,   913, -1018,  4729,   907,   908,
     370,   327,   914,   370,   327,  1718,   916,   922,  2291,   370,
   -1018, -1018,   614,   235, -1018, -1018,   409,  1964, -1018,   917,
     918, -1018,   926, -1018,   597,  1331,  2376,   932,  4567,  1654,
     928, -1018,   479,  3725,   933, -1018,   713,  4638,   285,   935,
     931,   634,   487,   936,   409,  4567,   124,   889,  4567, -1018,
   -1018, -1018,   156,  1159,  1388,    23, -1018,  2239,  4920,   924,
    4850,   355, -1018,   941,   810,   949,   145,   900,   905,   308,
   -1018,   800, -1018,   245,   919, -1018, -1018,   950, -1018,   292,
    3089,  2821,  2955,  3223,   121,   129,   944,   695,   934,   934,
    1281,  1281,  2787,  3261,  3172,  1618,  2528,  1887,   503,   503,
     678,   678, -1018, -1018, -1018,   830, -1018, -1018, -1018, -1018,
     379,   652,   307,  4854,   959,   770,   830, -1018,  1388,  1388,
     960, -1018,  4866,  1388, -1018, -1018, -1018, -1018,  1388,   954,
     955,   962,   963,  2373,   370,   327,   964,   968,   969,   370,
   -1018, -1018, -1018,   663,  3827, -1018,   965, -1018,   219,   966,
   -1018, -1018, -1018, -1018, -1018,   663, -1018, -1018, -1018,   979,
   -1018,   460,   663, -1018,   970,   182,   793,  1331, -1018,   460,
   -1018,   977, -1018, -1018,  3929,    41,  4780,   442, -1018, -1018,
    4567, -1018, -1018,   882, -1018,   161,   925, -1018,   984,   980,
   -1018,   292,  1233,   883, -1018,   460,   260,  1388, -1018, -1018,
   -1018,  1081, -1018, -1018, -1018, -1018,   214, -1018, -1018,   967,
    1061,  4567,  3089, -1018, -1018, -1018, -1018,  1654, -1018, -1018,
   -1018, -1018,   245, -1018, -1018,   988, -1018,   772, -1018,  2023,
   -1018,   245,  2904,  3089,  3089,  3223,  3550,  3089,   992,   995,
     996,   998, -1018,  3089, -1018, -1018, -1018, -1018,   838,   370,
   -1018, -1018,   370,   370,  1388,  1388,   999,  1001,  1002,   370,
    1388,  1003,  1005, -1018,   409,   409,  2023,  4567,   779, -1018,
     124, -1018,  1964,  1427,   409,  1013, -1018,   460,  1000, -1018,
   -1018,   663,  1654,   118, -1018,  4638, -1018,  1014,   362,   207,
     368, -1018,  2151,   294, -1018,  1008,    36,  1006,   789, -1018,
   -1018, -1018, -1018, -1018, -1018, -1018, -1018,  4348, -1018,  4031,
    1028, -1018,   308,  4567, -1018,   389, -1018,   207,   424, -1018,
    4567,   286,  1018,  1007, -1018,  1015,  2520,  1081, -1018,   919,
   -1018, -1018, -1018,   292, -1018, -1018, -1018,  1029,  1009,  1010,
    1012,   938,   207, -1018, -1018, -1018, -1018, -1018, -1018, -1018,
   -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018, -1018,
   -1018, -1018, -1018, -1018,  1027,  2023, -1018, -1018, -1018, -1018,
   -1018, -1018, -1018, -1018,  4420,  1031,  2023, -1018,  2904,  2904,
    2904,  3089,  3089, -1018,  4780,  3038, -1018,   370,   370,  1388,
    1034,  1041,   370,  1388,  1388, -1018, -1018,  1032,  1049, -1018,
   -1018,   663,  1054, -1018,   460,  1444,   124, -1018,  1056, -1018,
    1059, -1018, -1018,   161, -1018, -1018,   161,  1017, -1018, -1018,
    4780, -1018,  4567,  1654,  4780,  1736, -1018, -1018, -1018,  1072,
   -1018, -1018, -1018,   967,  1063,   967,  1496,  1077,  1084,   286,
     292,   520, -1018, -1018, -1018,   308, -1018,  1080,   919,  2023,
   -1018, -1018, -1018, -1018,   207,  1094,  1512, -1018,  1060,  1067,
    1068,  1071,  1076,   242,  1101,  2904,  2904,   129,   370,  1388,
    1388,   370,   370, -1018,  1110, -1018,  1111, -1018,   460, -1018,
   -1018, -1018, -1018, -1018,  1113,   479,  1055,    72,  3480,   460,
    1115, -1018,  3089,   207, -1018,  1081,   550, -1018,  1120,  1124,
    1119,   234, -1018, -1018, -1018,  1125, -1018, -1018, -1018,   292,
   -1018,  2023,  1128,  4567, -1018, -1018,  1081,  3089, -1018,  1512,
    1131,   370,   370, -1018, -1018,  1132, -1018,  1137, -1018,  4567,
    1134,  1141,    13,  1142,    -3, -1018, -1018,  2904,   967,   308,
   -1018, -1018, -1018,   292,  1143, -1018, -1018,  1144,  1080,  1135,
    4567,  1148,   308,  2653, -1018, -1018, -1018, -1018,  1152,  4567,
    4567,  4567,  1145,  1061,  4780,   389, -1018, -1018,  1150,  1154,
   -1018, -1018, -1018,  1162,   460, -1018, -1018,   460,  1165,  1169,
    1178,  4567, -1018,  1177, -1018,  1176, -1018,  2023,   460, -1018,
     483, -1018,   519,   460,   460,   460,  1184,   389,  1180, -1018,
   -1018, -1018, -1018,   286, -1018, -1018,   286, -1018, -1018, -1018,
     460, -1018, -1018,  1183,  1188, -1018, -1018, -1018
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
     579,     0,     0,     0,     0,     0,    10,     4,   529,   389,
     397,   390,   391,   394,   395,   392,   393,   379,   396,   378,
     398,   381,   399,   400,   401,   402,     0,   369,   370,   371,
     492,   493,   145,   487,   488,     0,   530,   531,     0,     0,
     541,     0,     0,     0,   367,   579,   374,   384,   377,   386,
     387,   491,     0,   548,   382,   539,     6,     0,     0,   579,
       1,    15,    64,    60,    61,     0,   261,    14,   256,   579,
       0,     0,    82,    83,   579,   579,     0,     0,   260,   262,
     263,     0,   264,   265,   270,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     9,
      11,    18,    19,    20,    21,    22,    23,    24,    25,   579,
      26,    27,    28,    29,    30,    31,    32,     0,    33,    34,
      35,    36,    37,    38,    12,   113,   118,   115,   114,    16,
      13,   154,   155,   156,   157,   158,   159,   257,     0,   275,
       0,   147,   146,     0,     0,     0,     0,     0,   579,   542,
     380,     3,   373,   368,   579,     0,   403,     0,     0,   541,
     352,   351,   366,     0,   298,   281,   579,   305,   579,   348,
     342,   332,   295,   375,   388,   383,   549,     0,     0,   537,
       5,     8,     0,   276,   579,   278,    17,     0,   563,   273,
       0,   255,     0,     0,   570,     0,     0,   372,   548,     0,
     579,     0,     0,    78,     0,   579,   268,   272,   579,   266,
     269,   267,   274,   271,     0,     0,   189,   548,     0,     0,
      62,    63,     0,     0,    51,    49,    46,    47,   579,     0,
     579,     0,   579,   579,     0,   112,   579,   579,     0,     0,
       0,     0,     0,     0,     0,   332,   259,   258,     0,   579,
       0,   579,   283,     0,     0,     0,     0,   543,   550,   540,
       0,   565,   429,   430,   441,   442,   443,   444,   445,   446,
     447,   448,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   289,     0,   284,   579,   422,   372,     0,   421,   423,
     427,   424,   428,   286,   376,   579,   352,   351,     0,     0,
     342,   382,     0,   293,   408,   409,   291,     0,   405,   406,
     407,   358,     0,   421,   294,     0,   579,     0,     0,   307,
     350,   324,     0,   306,   349,   364,   365,   333,   296,   579,
       0,   297,   579,     0,     0,   345,   344,   302,   343,   324,
     353,   547,   546,   545,     0,     0,   277,   280,   533,   532,
       0,   534,     0,   562,   116,   573,     0,    68,    45,     0,
     579,   403,    70,     0,   495,   496,   494,   497,     0,   498,
       0,    74,     0,     0,     0,    98,     0,     0,   185,     0,
     579,     0,   187,     0,     0,   103,     0,     0,     0,   107,
     299,   300,   301,    42,     0,   104,   106,   535,     0,   536,
      54,     0,    53,     0,     0,   178,   579,   182,   491,   180,
     169,     0,     0,     0,     0,   532,     0,     0,     0,     0,
       0,     0,   324,     0,     0,   332,   579,   548,   411,   579,
     579,   475,     0,   474,   383,   477,   489,   490,   385,     0,
     538,     0,     0,     0,     0,   439,   438,   467,   466,   440,
     468,   469,   528,     0,   285,   288,   470,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   564,   352,   351,   342,   382,     0,
     332,     0,   362,   360,   345,   344,     0,   332,   353,     0,
       0,   404,   359,   579,   342,   382,     0,   325,   579,     0,
       0,   363,     0,   338,     0,     0,   356,     0,     0,     0,
     304,   347,     0,   303,   346,   354,     0,     0,     0,   308,
     544,     7,   579,     0,   170,   579,     0,     0,   569,     0,
       0,    69,     0,    77,     0,     0,     0,     0,     0,     0,
       0,   186,   579,     0,     0,   579,   579,     0,     0,   108,
       0,   579,     0,     0,     0,     0,     0,   167,     0,   179,
     184,    58,     0,     0,     0,     0,    79,     0,     0,     0,
       0,     0,   149,     0,   382,     0,   506,   501,   502,     0,
     127,   579,   507,   579,   579,   162,   166,     0,   551,     0,
     431,     0,     0,   366,     0,   579,     0,   579,   464,   463,
     461,   462,     0,   460,   459,   455,   456,   454,   457,   458,
     449,   450,   451,   452,   453,     0,   353,   336,   335,   334,
     354,     0,   315,     0,     0,     0,   324,   326,   353,     0,
       0,   329,     0,     0,   340,   339,   361,   357,     0,     0,
       0,     0,     0,     0,   309,   355,     0,     0,     0,   311,
     279,    66,    67,    65,     0,   574,   575,   578,   577,   571,
      44,    43,    39,    76,    73,    75,   568,    93,   567,     0,
      88,   579,   566,    92,     0,   577,     0,     0,    99,   579,
     227,     0,   190,   191,     0,   256,     0,     0,    50,    48,
     579,    41,   105,     0,   556,   554,     0,    57,     0,     0,
     110,     0,   579,   579,   579,   579,     0,     0,   133,   132,
     134,   579,   136,   131,   135,   140,     0,   148,   150,   579,
     579,   579,     0,   508,   504,   503,   126,     0,   123,   125,
     121,   128,   579,   129,   499,   476,   478,   480,   500,     0,
     160,   579,   432,     0,     0,   366,   365,     0,     0,     0,
       0,     0,   287,     0,   337,   292,   341,   327,     0,   317,
     331,   330,   316,   312,     0,     0,     0,     0,     0,   310,
       0,     0,     0,   117,     0,     0,     0,   579,   510,   511,
       0,   513,     0,     0,     0,     0,    90,   579,     0,   119,
     188,   255,     0,   548,   101,     0,   100,     0,     0,     0,
       0,   552,   579,     0,    52,     0,   256,     0,     0,   171,
     172,   176,   175,   168,   173,   177,   174,     0,   183,     0,
       0,    81,     0,   579,   141,     0,   412,   417,     0,   413,
     579,   403,   511,   579,   153,     0,     0,   579,   130,   579,
     485,   484,   486,     0,   482,   198,   218,     0,     0,     0,
       0,   262,     0,   240,   241,   233,   242,   216,   196,   238,
     234,   232,   235,   236,   237,   239,   217,   213,   214,   200,
     208,   207,   211,   210,     0,     0,   201,   202,   206,   212,
     203,   204,   205,   215,     0,   275,     0,   282,   435,   434,
     433,     0,     0,   425,     0,   465,   328,   314,   313,     0,
       0,     0,   318,     0,     0,   576,   572,     0,     0,   512,
      84,   577,    95,    89,   579,     0,     0,    97,     0,    71,
       0,   109,   557,   555,   561,   560,   559,     0,    55,    56,
       0,   228,   579,     0,     0,     0,    59,    80,   122,     0,
     143,   142,   139,   579,   418,   579,     0,     0,     0,     0,
       0,     0,   521,   505,   509,     0,   479,   579,   579,     0,
     193,   230,   229,   231,     0,     0,     0,   192,   379,   378,
     381,     0,   377,   382,     0,   437,   436,   579,   319,     0,
       0,   323,   322,    40,     0,    96,     0,    91,   579,    86,
      72,   102,   553,   558,     0,   579,     0,     0,   579,   579,
       0,   416,     0,   415,   151,   579,     0,   518,     0,   520,
     522,     0,   514,   515,   124,     0,   472,   481,   473,     0,
     199,     0,     0,   579,   164,   163,   579,     0,   209,     0,
       0,   321,   320,    94,    85,     0,   111,     0,   167,   579,
       0,     0,     0,     0,     0,   144,   419,   420,   579,     0,
     516,   517,   519,     0,     0,   526,   527,     0,   579,     0,
     579,     0,     0,     0,   161,   426,    87,   120,     0,   579,
     579,   579,     0,   579,     0,     0,   414,   152,   523,     0,
     471,   483,   194,     0,   579,   165,   249,   579,     0,     0,
       0,   579,   219,     0,   137,     0,   524,     0,   579,   220,
       0,   226,     0,   579,   579,   579,     0,     0,     0,   195,
     221,   243,   245,     0,   246,   248,   403,   224,   223,   222,
     579,   138,   525,     0,     0,   225,   244,   247
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
   -1018, -1018,  -367, -1018, -1018, -1018, -1018,    44,    68,    -4,
      77, -1018,   706, -1018,    82,    84, -1018, -1018, -1018,    97,
   -1018,    99, -1018,   100, -1018, -1018,   105, -1018,   109,  -540,
    -643,   110, -1018,   117, -1018,  -353,   683,   -71,   119,   146,
     152,   174, -1018,   533,  -792,  -692, -1018, -1018, -1018,  -862,
   -1017, -1018,  -134, -1018, -1018, -1018, -1018, -1018,    10, -1018,
   -1018,   208,    24,    46, -1018, -1018,   304, -1018,   682,   542,
     190, -1018, -1018, -1018,  -742, -1018, -1018, -1018, -1018,   545,
   -1018,   553,   199,   554, -1018, -1018, -1018,  -474, -1018, -1018,
   -1018,    45,   -56, -1018,   730,    16,   451, -1018,   661,   804,
     -25,  -584,  -534,   -95,  1102,  -200,  -145,   745,    62,    35,
   -1018,   -68,    54,   -11,   691,  -541,  1211, -1018,  -347, -1018,
    -150, -1018, -1018, -1018,  -867, -1018,   263, -1018,   937,  -136,
    -499, -1018, -1018,   210,   831, -1018, -1018, -1018,   430, -1018,
   -1018, -1018,  -214,   -52, -1018, -1018,   313,   697,  -382,  -614,
     206, -1018, -1018,   231,   -30,   123,  -116, -1018,   946,  -218,
    -139,  1116, -1018,  -393,  1109, -1018,   602,   230,  -196,  -505,
       0
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     4,     5,    99,   100,   101,   776,   853,   854,   855,
     856,   106,   400,   401,   857,   858,   704,   109,   110,   859,
     112,   860,   114,   861,   664,   202,   862,   117,   863,   670,
     537,   864,   374,   865,   384,   231,   395,   232,   866,   867,
     868,   869,   525,   125,   730,   579,   711,   126,   716,   825,
     942,  1000,    41,   571,   127,   128,   129,   130,   870,   886,
     739,  1025,   871,   872,   702,   813,   404,   405,   406,   559,
     873,   135,   545,   380,   874,  1021,  1097,   959,   875,   876,
     877,   878,   879,   880,   881,   882,  1099,  1101,   883,   971,
     137,   884,   298,   183,   346,   184,   282,   283,   454,   284,
     580,   165,   389,   166,   319,   167,   168,   169,   244,    43,
      44,   285,   197,    46,    47,    48,    49,    50,   306,   307,
     348,   309,   310,   426,   827,   828,   943,  1046,   287,   313,
     289,   290,  1016,  1017,   432,   433,   584,   735,   736,   843,
     958,   844,    51,    52,   367,   368,   737,   582,   779,  1102,
     834,   951,  1009,  1010,   176,    53,   355,   398,    54,   179,
      55,   259,   696,   801,   291,   292,   673,   193,   356,   659,
     185
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
       6,   104,   196,   182,   249,   139,   308,   258,   681,   299,
     149,   750,   288,   543,   530,   131,   700,    42,    57,   822,
     164,   229,   243,   408,   230,   230,   385,   713,   657,   132,
     938,   676,   152,   785,   907,   550,   315,   657,   341,   258,
     353,   440,   188,   824,   417,   172,   238,   188,   583,   102,
     138,   133,   246,  1071,   353,    45,    45,   780,  1094,   627,
       8,  -290,   361,    60,   627,   788,   212,   329,   617,   194,
     250,   350,   145,   103,   194,   203,  1001,   357,  1003,   153,
     526,   173,   105,   363,   330,   391,   251,   107,   372,   108,
    1121,   820,   527,   561,   354,   260,   479,   481,  -181,   213,
     486,   360,   111,   569,   113,   115,   833,   170,  1074,   194,
     116,   148,  1039,    45,   118,   119,   247,   397,    27,    28,
      29,   189,   120,  -290,   121,   520,   189,   419,   666,   149,
     188,   303,     8,   967,   628,   390,   148,   786,   352,   707,
       8,   328,  1072,   331,   974,   955,    36,    37,   293,   375,
     299,   122,   376,   424,   190,   430,     8,   123,   654,     8,
    -181,   919,   747,  1014,   281,   423,   172,   667,   172,   154,
     668,   299,   387,   916,   381,   256,   257,   155,   684,   124,
     403,  1076,  1075,   156,   347,   722,   157,   294,   353,    58,
     304,   305,   177,   198,   687,   134,   359,   326,  1040,   732,
     369,  1041,   286,   162,   136,   158,   162,   140,    45,   217,
       8,   308,   479,   481,   486,   178,   300,  1020,    36,    37,
       8,   148,   220,   148,   325,   353,    36,    37,    59,   214,
     392,  1107,   396,   399,   577,   578,  -534,   409,   669,   651,
     910,   188,    36,    37,    38,    36,    37,   748,   159,   428,
     749,   435,    38,   407,   823,   160,    40,  1077,   161,    45,
     824,   164,    45,   162,   221,   555,   353,   163,    38,   985,
    1085,    38,    40,   502,  1055,   159,   505,   912,   799,  1059,
    1056,   652,    45,   657,   455,  1028,   172,   162,   621,   622,
      45,    45,   170,   431,   163,     8,    36,    37,   928,   156,
     986,   800,   414,   821,   200,   143,    36,    37,     8,   726,
     349,   542,   144,  1049,   141,   349,   349,   288,  -254,   146,
     230,   158,   349,   551,   370,   371,   230,   349,   688,   583,
     655,   689,    38,   142,  1062,   529,    40,   819,   832,   239,
     929,   378,   383,   972,   408,   349,   386,   329,   170,    45,
     727,   728,   379,   147,   729,  1109,   349,   477,   698,   148,
     172,   522,   621,   415,   330,     8,   349,   332,   364,   365,
      45,     8,   427,   573,  1035,   595,   989,     8,   494,   439,
       6,    36,    37,    45,   333,  1044,    45,   366,   501,   589,
       8,   304,   305,  1030,    36,    37,   909,   391,   718,    45,
    -410,   719,   586,  -410,   148,  -579,   560,    38,     8,   588,
     498,    40,     8,   186,   154,   188,  -579,  -548,  -548,   295,
     674,   414,   155,   256,   342,   151,   572,   499,   156,   940,
     172,   157,   941,  -410,   435,   431,   157,   624,   210,   180,
     671,  -548,   630,   211,   679,   295,   666,   390,   188,   181,
     158,    36,    37,    30,    31,   583,   187,    36,    37,   833,
     148,  1037,   157,    36,    37,   581,   945,   443,   705,   281,
    1100,   946,    33,    34,     8,   444,    36,    37,  -579,   230,
     564,   922,     8,    45,   948,   794,   192,   925,   668,    38,
       8,   414,   199,   159,    36,    37,   795,   201,    36,    37,
     160,   204,    38,   161,   594,   205,    40,   286,   162,    27,
      28,    29,   163,   412,   304,   305,   413,   216,   208,   154,
      38,   162,   347,   209,    40,     6,  1111,   155,   219,  1112,
     992,   484,   222,   993,   485,     8,   157,  1113,   223,   104,
     575,   224,   392,   139,   225,     6,   139,    45,   576,   577,
     578,   396,    45,   131,   678,   226,   731,   188,   733,   717,
      36,    37,  1114,  1012,   233,  1115,  1013,   132,    36,    37,
     164,   699,   418,  1116,   407,   234,    36,    37,   905,   906,
     334,   172,   787,   172,   738,   188,    38,   102,   138,   133,
      40,   686,    45,  1050,    38,   172,  1051,   455,   159,   329,
     618,   663,   693,   188,   170,   160,   694,   236,   161,    45,
     252,   103,    45,   162,   206,   207,   330,   163,   792,   253,
     105,    36,    37,   583,  1110,   107,   254,   108,   746,  1117,
    1118,  1119,   837,   302,   797,   469,   470,   471,   472,   473,
     111,   340,   113,   115,   583,   344,  1125,    38,   116,   349,
     104,    40,   118,   119,   139,   325,   345,   170,   420,   349,
     120,   421,   121,   358,   131,   835,   288,   382,   674,   353,
     318,   781,   388,   329,   619,   695,   394,   349,   132,   781,
     104,   308,   535,   536,   139,     1,     2,     3,   914,   122,
     330,   832,   329,   754,   131,   123,   393,   918,   102,   138,
     133,   438,   139,   560,     6,   781,   778,   838,   132,   330,
     423,   172,   810,   965,   778,   826,   887,   124,   408,   829,
     781,   908,   103,   436,   920,   437,   811,   230,   102,   138,
     133,   105,   172,   134,    66,   152,   107,   453,   108,   885,
     778,   172,   136,   441,    45,   456,   581,   817,   812,   935,
     249,   111,   103,   113,   115,   778,   238,   685,   452,   116,
     474,   105,   482,   118,   119,   251,   107,   939,   108,  1124,
     483,   120,   488,   121,   947,    45,   885,    27,    28,    29,
     746,   111,   489,   113,   115,   553,   554,   781,   490,   116,
      78,    79,    80,   118,   119,    82,   493,    83,    84,  1008,
     122,   120,   293,   121,    30,    31,   123,   325,   501,   793,
     498,   757,   471,   472,   473,   104,   935,   496,   281,   139,
     988,   500,   778,    33,    34,  1022,   977,   499,   124,   131,
     122,    45,   251,   952,   783,   784,   123,   172,   424,   738,
     430,    30,    31,   132,   134,   840,   841,   842,   162,   508,
     391,  1005,   515,   136,   156,   237,   286,   521,   124,   575,
      33,    34,   995,   102,   138,   133,   997,   576,   577,   578,
     523,  1026,   581,   528,   134,   885,   158,    45,   498,   896,
     575,   531,     8,   136,    45,   188,   885,   103,   576,   577,
     578,   533,   932,   534,   538,   499,   105,   349,   349,   539,
     390,   107,   540,   108,   546,   320,   324,   349,   541,   547,
     548,   549,   552,   556,   781,   338,   111,   557,   113,   115,
     362,   923,   924,   926,   116,   558,   562,   563,   118,   119,
     566,   567,   409,   568,  1026,   570,   120,   585,   121,   826,
     596,   826,  1043,   829,   616,   829,   620,   629,   407,   778,
     944,   633,   164,   634,   636,   638,   639,  1018,   738,   885,
     660,   661,   640,  1123,   646,   122,   308,  1061,    36,    37,
     647,   123,   662,   677,   680,   690,   683,   172,   691,   697,
     701,   715,   720,  1068,   304,   305,    45,   150,   781,   721,
     724,   171,   751,   124,   725,   392,   740,   734,   175,   781,
     756,   760,   764,   765,  1083,   172,  1093,   774,   775,   134,
     766,   767,   770,  1088,  1089,  1090,   771,   772,   136,   777,
     789,   885,   798,   778,   782,   803,   172,   804,   802,    70,
     839,   215,   218,   891,   778,  1106,   892,   893,   894,   170,
     581,   320,   324,   917,   826,   338,   931,   899,   829,   900,
     901,   903,    45,   904,   915,   921,   953,   170,  1018,  1042,
     414,   581,   930,   245,   467,   468,   469,   470,   471,   472,
     473,   937,   949,   781,   966,   960,   964,    45,  -197,   983,
     511,   514,   979,   950,   781,   961,   962,   781,   963,   980,
     984,   255,     8,    45,   312,   314,   784,   885,   781,   990,
     301,   830,   991,   781,   781,   781,   321,   321,   778,   327,
      27,    28,    29,   999,    45,   831,   339,  1002,  1006,   778,
     781,   994,   778,    45,    45,    45,   944,  1007,  1015,   316,
      27,    28,    29,   778,  1023,   156,  -252,   317,   778,   778,
     778,   575,   245,  -251,  -253,    45,   157,  1027,  1029,   576,
     577,   578,  -250,  1033,  1034,   778,  1036,   158,  1045,  1038,
     377,   575,     8,  1052,     8,   188,  1053,  1054,  1060,   576,
     577,   578,  1065,  1057,  1069,  1066,   171,   191,    36,    37,
    1067,  1070,  1082,  1073,   410,  1091,   416,   321,   321,  1084,
     422,  1079,  1080,  1087,   425,   150,   245,   434,  1095,   359,
     227,   154,  1096,  1098,    38,   235,  1103,   155,    40,   442,
    1104,   445,   446,   447,   448,   449,   450,   451,   157,  1105,
     320,   324,   338,   824,  1108,  1120,  1126,   318,  1122,   511,
     514,  1127,   171,   656,   692,   809,   996,  1064,     8,   338,
     703,   478,   480,   480,   491,   818,   487,   814,    36,    37,
      36,    37,   650,   927,    66,   815,   816,   597,   752,   712,
     645,   174,   495,   323,   497,   587,  1048,   504,  1081,   956,
     507,  1019,   337,   723,    38,   316,    38,   806,   159,  1092,
     159,   321,   321,   322,  1078,   241,   321,   484,   242,   796,
     485,     0,   157,   162,   343,     0,   351,   163,   524,   163,
       0,   351,   351,     0,     0,   416,     0,     0,   351,     0,
      78,    79,    80,   351,   532,    82,   807,    83,    84,     0,
       0,     0,     0,     0,    36,    37,     0,   544,     0,     0,
       0,   351,     0,     0,     8,   666,     0,   188,     0,     8,
       0,     0,   351,   402,   323,     0,   337,     0,   411,   351,
      38,   808,   351,     0,    40,     0,     0,     0,   480,   480,
     480,     0,     0,     0,   565,   645,   321,   321,     0,   321,
       0,   359,     0,   318,   667,   574,   316,   668,     0,   155,
       0,     0,     0,     0,   509,     8,     0,     0,   457,   458,
       0,     8,     0,   157,   598,   599,   600,   601,   602,   603,
     604,   605,   606,   607,   608,   609,   610,   611,   612,   613,
     614,   467,   468,   469,   470,   471,   472,   473,     0,     0,
      36,    37,   154,   615,     0,    36,    37,   623,   418,     0,
       8,     0,     0,     0,   480,   480,   632,   510,   513,   157,
     626,     0,   519,     0,     0,   669,    38,     8,     0,     0,
     159,    38,     0,     0,     0,    40,     0,   241,     0,     0,
     242,   321,     0,     0,   321,   162,     0,   359,     0,   163,
     913,    36,    37,     0,   318,   155,     0,    36,    37,     0,
       0,   245,     0,     0,   359,   245,     0,   987,   171,     0,
       0,     0,   155,     0,     0,     0,     0,    38,     0,     8,
       0,   159,     0,    38,     0,     0,     0,    40,   480,   245,
     321,     0,     0,   321,   714,     8,    36,    37,     0,     0,
     163,     0,   510,   513,     0,   519,   318,   742,   604,   607,
     612,     0,     0,    36,    37,   741,   359,     0,     0,  1004,
       0,   171,    38,     0,   155,     0,   159,     0,     0,     0,
       0,     0,   359,   241,     0,  1024,   242,     0,     0,    38,
     155,   162,     8,   159,     0,   163,     0,     0,     0,     0,
     241,     0,     0,   242,   321,   321,     0,     0,   162,   321,
       0,     0,   163,     0,   321,    36,    37,     0,     0,   321,
       0,     0,     0,     0,     0,     0,   625,     0,     0,   240,
       0,    36,    37,     0,     0,     0,     0,   155,     0,     0,
       0,    38,     0,     0,     0,   159,     0,   644,     0,     0,
     649,     0,   241,   245,     0,   242,     0,    38,     0,     0,
     162,   159,   653,     0,   163,   351,   658,     0,   241,     0,
       0,   242,     0,   665,   672,   675,   162,   805,    36,    37,
     163,     0,     0,   321,     0,     8,     0,     8,     0,   836,
       0,     0,     0,   351,     0,   672,   625,     0,     0,   644,
       0,     0,   706,   245,    38,     8,     0,     0,   159,     0,
     888,   889,   449,     0,   890,   241,     0,     0,   242,     0,
     895,     0,   316,   162,   359,     0,     0,   163,     0,     0,
     334,     0,   155,     0,     0,     0,     0,     0,     0,   157,
     321,   321,   429,     0,     0,     0,   321,     0,     0,     0,
     155,     8,     0,     0,     0,   457,   458,   459,   460,   245,
     758,   759,     0,     0,     0,   762,     0,     0,   245,     8,
     763,    36,    37,    36,    37,   769,   465,   466,   467,   468,
     469,   470,   471,   472,   473,     0,     0,     0,   316,     0,
       0,    36,    37,     0,     0,     0,   641,    38,     0,    38,
       0,    40,     0,   159,     0,   157,   998,     0,   335,     0,
     241,   336,     8,   242,   155,     0,   672,    38,   162,   957,
     318,   159,   163,     0,   791,     0,   672,     0,   241,     0,
       0,   242,     0,     0,     0,     0,   162,    36,    37,   758,
     163,     0,     0,     0,     0,     0,     0,     8,     0,   418,
       0,     0,     0,     0,     0,    36,    37,   516,   975,   976,
     973,     0,     0,    38,     0,     0,     0,    40,     0,     0,
       0,     8,     0,     0,   642,   321,     0,   643,     0,   321,
     321,    38,     0,     0,   316,   159,   318,     0,     0,     0,
       0,   245,   241,     0,     0,   242,   897,   898,    36,    37,
     162,   157,   902,     8,   163,     0,     8,     0,   316,   245,
       0,   245,     0,   351,   351,     0,   512,     0,     0,   672,
       0,   911,   245,   351,    38,   157,  1011,     0,    40,     0,
       8,     0,     0,    36,    37,   517,     0,     0,   518,     0,
     418,     0,   245,   418,     0,   791,     0,   318,   317,     0,
       0,   322,     0,   171,     0,   321,   321,    36,    37,    38,
       0,     0,     0,    40,     0,     0,     0,   418,     0,  1047,
     484,   171,     0,   485,   574,   509,     0,     0,     0,     0,
       0,     0,   318,    38,     0,     0,     0,    40,     0,    36,
      37,     0,    36,    37,  1063,  1058,     0,     0,     0,     0,
     188,     0,     0,     0,     0,   245,   318,   264,   265,   266,
     267,   268,   269,   270,   271,    38,    36,    37,    38,    40,
       0,     0,    40,     0,   457,   458,   459,   460,     0,  1011,
       0,   978,     0,     0,     0,   981,   982,     0,   318,     0,
       0,   318,    38,     0,     0,     0,    40,   467,   468,   469,
     470,   471,   472,   473,   845,   672,  -579,    62,     0,     0,
       0,    63,    64,    65,     0,   318,     0,     0,     0,     0,
       0,     0,     0,     0,    66,  -579,  -579,  -579,  -579,  -579,
    -579,  -579,  -579,  -579,  -579,  -579,  -579,     0,  -579,  -579,
    -579,  -579,  -579,     0,     0,     0,   846,    68,     0,     0,
    -579,     0,  -579,  -579,  -579,  -579,  -579,     0,     0,     0,
       0,  1031,  1032,     0,     0,    70,    71,    72,    73,   847,
      75,    76,    77,  -579,  -579,  -579,   848,   849,   850,     0,
      78,   851,    80,     0,    81,    82,   807,    83,    84,  -579,
    -579,     0,  -579,  -579,    85,     0,     0,     0,    89,     0,
      91,    92,    93,    94,    95,    96,     0,     8,     0,     0,
     188,     0,     0,     0,     0,     0,    97,     0,  -579,     0,
       0,    98,  -579,  -579,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     8,     0,     0,   188,   261,     0,
       8,   852,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,     0,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,     0,    21,    22,    23,    24,
      25,   272,     0,     0,     0,     0,     0,   418,     0,    26,
      27,    28,    29,    30,    31,   512,   273,     0,     0,     0,
       0,     0,     0,    36,    37,     0,     0,     0,     0,     0,
       0,    32,    33,    34,     0,     0,     0,     0,     0,   304,
     305,     0,     0,     0,     0,     0,     0,    35,     0,    38,
      36,    37,     8,    40,     0,     0,    36,    37,     0,     0,
     412,     0,     0,   413,     0,     0,     0,     0,   162,     0,
       0,     0,     0,     0,     0,     0,    38,     0,     0,    39,
      40,     0,    38,     0,     0,     0,    40,   274,     0,   418,
     275,     0,     0,   276,   277,   278,     0,   641,     8,   279,
     280,   188,   261,     0,     8,   318,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,     0,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,     0,
      21,    22,    23,    24,    25,   272,     0,     0,    36,    37,
       0,   418,     0,     0,    27,    28,    29,    30,    31,   648,
     273,     0,     0,   311,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    38,    32,    33,    34,    40,     0,
       0,     0,     0,     0,     0,   642,     0,     0,   643,     0,
       0,    35,     0,     0,    36,    37,     8,   318,     0,     8,
      36,    37,   188,     0,     0,     0,     0,     0,     0,   264,
     265,   266,   267,   268,   269,   270,   271,     0,     0,     0,
      38,     0,     0,     0,    40,     0,    38,     0,     0,     0,
      40,   274,     0,   418,   275,     0,     0,   276,   277,   278,
       0,   768,     8,   279,   280,   188,   261,     0,   590,   318,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
       0,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,     0,    21,    22,    23,    24,    25,   272,
       0,     0,    36,    37,     0,    36,    37,     0,    27,    28,
      29,    30,    31,     0,   273,     0,     0,   503,     0,     0,
       0,   304,   305,     0,     0,     0,     0,     0,    38,    32,
      33,    34,    40,     0,   457,   458,   459,   460,     0,   461,
       0,     0,     0,     0,     0,    35,     0,     0,    36,    37,
       0,   318,   462,   591,   464,   465,   592,   467,   468,   469,
     470,   593,   472,   473,     0,     0,     9,    10,    11,    12,
      13,    14,    15,    16,    38,    18,     0,    20,    40,     0,
      22,    23,    24,    25,     0,   274,     0,     0,   275,     0,
       0,   276,   277,   278,     0,     0,     8,   279,   280,   188,
     261,   954,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,     0,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,     0,    21,    22,
      23,    24,    25,   272,     0,     0,     0,     0,     0,     0,
       0,     0,    27,    28,    29,    30,    31,     0,   273,     0,
       0,   506,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    32,    33,    34,     0,   457,   458,   459,
     460,     0,   461,     0,     0,   457,   458,   459,   460,    35,
       0,     0,    36,    37,     0,   462,   463,   464,   465,   466,
     467,   468,   469,   470,   471,   472,   473,   466,   467,   468,
     469,   470,   471,   472,   473,     0,     0,     0,    38,     0,
       0,     0,    40,     0,     0,     0,     0,     0,     0,   274,
       0,     0,   275,     0,     0,   276,   277,   278,     0,     0,
       8,   279,   280,   188,   261,     0,  1086,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,     0,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,     0,    21,    22,    23,    24,    25,   272,     0,     0,
       0,     0,     0,     0,     0,     0,    27,    28,    29,    30,
      31,     0,   273,     0,     0,   631,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    32,    33,    34,
     457,   458,   459,   460,     0,   461,     0,     0,     0,     0,
       0,     0,     0,    35,     0,     0,    36,    37,   462,   463,
     464,   465,   466,   467,   468,   469,   470,   471,   472,   473,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    38,     0,     0,     0,    40,     0,     0,     0,
       0,     0,     0,   274,     0,     0,   275,     0,     0,   276,
     277,   278,     0,     0,     8,   279,   280,   188,   261,     0,
       0,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,     0,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,     0,    21,    22,    23,    24,
      25,   272,   743,   753,     0,     0,     0,     0,     0,     0,
      27,    28,    29,    30,    31,     0,   273,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    32,    33,    34,   457,   458,   459,   460,     0,   461,
       0,     0,     0,     0,     0,     0,     0,    35,     0,     0,
      36,    37,   462,   463,   464,   465,   466,   467,   468,   469,
     470,   471,   472,   473,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    38,     0,     0,     0,
      40,     0,     0,     0,     0,     0,     0,   274,     0,     0,
     275,     0,     0,   276,   277,   278,     0,     0,     8,   279,
     280,   188,   261,     0,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,     0,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,     0,
      21,    22,    23,    24,    25,   272,   744,     0,     0,     0,
       0,     0,     0,     0,    27,    28,    29,    30,    31,     0,
     273,   457,   458,   459,   460,     0,   461,     0,     0,     0,
       0,     0,     0,     0,     0,    32,    33,    34,     0,   462,
     463,   464,   465,   466,   467,   468,   469,   470,   471,   472,
     473,    35,     0,     0,    36,    37,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      38,     0,     0,     0,    40,     0,     0,     0,     0,     0,
       0,   274,     0,     0,   275,     0,     0,   276,   277,   278,
       0,     0,     8,   279,   280,   188,   261,     0,     0,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
       0,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,     0,    21,    22,    23,    24,    25,   272,
       0,     0,     0,     0,     0,     0,     0,     0,    27,    28,
      29,    30,    31,     0,   273,   457,   458,   459,   460,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    32,
      33,    34,     0,   462,   463,   464,   465,   466,   467,   468,
     469,   470,   471,   472,   473,    35,     0,     0,    36,    37,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    38,     0,     0,     0,    40,     0,
       0,     0,     0,     0,     0,   274,     0,     0,   275,     0,
       0,   276,   277,   278,     0,     0,     8,   279,   280,   188,
     261,     0,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,     0,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,     0,    21,    22,
      23,    24,    25,   272,     0,     0,     0,     0,     0,     0,
       0,     0,    27,    28,    29,    30,    31,     0,   273,   457,
     458,   459,   460,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    32,    33,    34,     0,     0,     8,   464,
     465,   466,   467,   468,   469,   470,   471,   472,   473,    35,
       0,     0,    36,    37,     0,     0,     0,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,     0,
      21,    22,    23,    24,    25,   295,     0,     0,    38,     0,
       0,     0,    40,    26,    27,    28,    29,    30,    31,     0,
       0,     0,   157,     0,     0,   276,   277,   745,     0,     0,
       0,   279,   280,     0,     0,    32,    33,    34,   457,   458,
     459,   460,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    35,     0,     0,    36,    37,     0,   463,   464,   465,
     466,   467,   468,   469,   470,   471,   472,   473,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      38,     8,     0,    39,    40,     0,     0,     0,     0,     0,
       0,   296,     0,     0,   297,     0,     0,     0,     0,   162,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,     0,    21,    22,    23,    24,    25,   295,     0,
       0,     0,     0,     0,     0,     0,    26,    27,    28,    29,
      30,    31,     0,     0,     0,   157,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    32,    33,
      34,     0,     0,     8,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    35,     0,     0,    36,    37,     0,
       0,     0,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,     0,    21,    22,    23,    24,    25,
       0,     0,     0,    38,     0,     0,    39,    40,    26,    27,
      28,    29,    30,    31,   475,     0,     0,   476,     0,     0,
       0,     0,   162,     0,     0,     0,     0,     0,     0,     0,
      32,    33,    34,     8,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    35,     0,     0,    36,
      37,     0,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,     0,    21,    22,    23,    24,    25,
       0,     0,     0,     0,     0,    38,     0,     0,    39,    40,
       0,     0,    30,    31,     0,     0,   412,     0,     0,   413,
       0,     0,     0,     0,   162,     0,     0,     0,     0,     0,
      32,    33,    34,    -2,    61,     0,  -579,    62,     0,     0,
       0,    63,    64,    65,     0,     0,    35,     0,     0,    36,
      37,     0,     0,     0,    66,  -579,  -579,  -579,  -579,  -579,
    -579,  -579,  -579,  -579,  -579,  -579,  -579,     0,  -579,  -579,
    -579,  -579,  -579,     0,     0,    38,    67,    68,     0,    40,
       0,     0,  -579,  -579,  -579,  -579,  -579,     0,     0,    69,
       0,     0,     0,     0,   162,    70,    71,    72,    73,    74,
      75,    76,    77,  -579,  -579,  -579,     0,     0,     0,     0,
      78,    79,    80,     0,    81,    82,     0,    83,    84,  -579,
    -579,     0,  -579,  -579,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    61,     0,  -579,    62,
       0,     0,     0,    63,    64,    65,    97,     0,  -579,     0,
       0,    98,  -579,     0,     0,     0,    66,  -579,  -579,  -579,
    -579,  -579,  -579,  -579,  -579,  -579,  -579,  -579,  -579,     0,
    -579,  -579,  -579,  -579,  -579,     0,     0,     0,    67,    68,
       0,     0,   682,     0,  -579,  -579,  -579,  -579,  -579,     0,
       0,    69,     0,     0,     0,     0,     0,    70,    71,    72,
      73,    74,    75,    76,    77,  -579,  -579,  -579,     0,     0,
       0,     0,    78,    79,    80,     0,    81,    82,     0,    83,
      84,  -579,  -579,     0,  -579,  -579,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    61,     0,
    -579,    62,     0,     0,     0,    63,    64,    65,    97,     0,
    -579,     0,     0,    98,  -579,     0,     0,     0,    66,  -579,
    -579,  -579,  -579,  -579,  -579,  -579,  -579,  -579,  -579,  -579,
    -579,     0,  -579,  -579,  -579,  -579,  -579,     0,     0,     0,
      67,    68,     0,     0,   773,     0,  -579,  -579,  -579,  -579,
    -579,     0,     0,    69,     0,     0,     0,     0,     0,    70,
      71,    72,    73,    74,    75,    76,    77,  -579,  -579,  -579,
       0,     0,     0,     0,    78,    79,    80,     0,    81,    82,
       0,    83,    84,  -579,  -579,     0,  -579,  -579,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      61,     0,  -579,    62,     0,     0,     0,    63,    64,    65,
      97,     0,  -579,     0,     0,    98,  -579,     0,     0,     0,
      66,  -579,  -579,  -579,  -579,  -579,  -579,  -579,  -579,  -579,
    -579,  -579,  -579,     0,  -579,  -579,  -579,  -579,  -579,     0,
       0,     0,    67,    68,     0,     0,   790,     0,  -579,  -579,
    -579,  -579,  -579,     0,     0,    69,     0,     0,     0,     0,
       0,    70,    71,    72,    73,    74,    75,    76,    77,  -579,
    -579,  -579,     0,     0,     0,     0,    78,    79,    80,     0,
      81,    82,     0,    83,    84,  -579,  -579,     0,  -579,  -579,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    61,     0,  -579,    62,     0,     0,     0,    63,
      64,    65,    97,     0,  -579,     0,     0,    98,  -579,     0,
       0,     0,    66,  -579,  -579,  -579,  -579,  -579,  -579,  -579,
    -579,  -579,  -579,  -579,  -579,     0,  -579,  -579,  -579,  -579,
    -579,     0,     0,     0,    67,    68,     0,     0,     0,     0,
    -579,  -579,  -579,  -579,  -579,     0,     0,    69,     0,     0,
       0,   936,     0,    70,    71,    72,    73,    74,    75,    76,
      77,  -579,  -579,  -579,     0,     0,     0,     0,    78,    79,
      80,     0,    81,    82,     0,    83,    84,  -579,  -579,     0,
    -579,  -579,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,     7,     0,     8,     0,     0,     0,
       0,     0,     0,     0,    97,     0,  -579,     0,     0,    98,
    -579,     0,     0,     0,     0,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,     0,    21,    22,
      23,    24,    25,     0,     0,     0,     0,     0,     0,     0,
       0,    26,    27,    28,    29,    30,    31,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    32,    33,    34,    56,     0,     8,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    35,
       0,     0,    36,    37,     0,     0,     0,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,     0,
      21,    22,    23,    24,    25,     0,     0,     0,    38,     0,
       0,    39,    40,    26,    27,    28,    29,    30,    31,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    32,    33,    34,   195,     0,
       8,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    35,     0,     0,    36,    37,     0,     0,     0,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,     0,    21,    22,    23,    24,    25,     0,     0,     0,
      38,     0,     0,    39,    40,     0,    27,    28,    29,    30,
      31,     0,     0,     0,     0,     0,     0,     0,     0,   492,
       0,     0,     0,     0,     0,     0,     0,    32,    33,    34,
       0,     8,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    35,     0,     0,    36,    37,     0,     0,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,     0,    21,    22,    23,    24,    25,   457,   458,
     459,   460,    38,   461,     0,     0,    40,    27,    28,    29,
      30,    31,     0,     0,     0,     0,   462,   463,   464,   465,
     466,   467,   468,   469,   470,   471,   472,   473,    32,    33,
      34,     0,     0,     8,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    35,   933,     0,    36,    37,     0,
       0,     0,     9,    10,    11,    12,    13,    14,    15,    16,
     968,    18,   969,    20,     0,   970,    22,    23,    24,    25,
       0,     0,     0,    38,     0,     0,     0,    40,   934,    27,
      28,    29,    30,    31,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      32,    33,    34,     0,     0,     0,     0,     0,     8,     0,
       0,     0,     0,     0,     0,     0,    35,   248,   373,    36,
      37,     0,     0,     0,     0,     0,     0,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,     0,
      21,    22,    23,    24,    25,    38,     0,     0,     0,    40,
     934,     0,     0,    26,    27,    28,    29,    30,    31,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    32,    33,    34,     0,     0,
       8,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    35,     0,     0,    36,    37,     0,     0,     0,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,     0,    21,    22,    23,    24,    25,     0,     0,     0,
      38,     0,     0,    39,    40,    26,    27,    28,    29,    30,
      31,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    32,    33,    34,
       0,     8,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    35,     0,     0,    36,    37,     0,     0,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,     0,    21,    22,    23,    24,    25,   228,     0,
       0,     0,    38,     0,     0,    39,    40,    27,    28,    29,
      30,    31,     0,     0,     0,     0,     0,     0,     0,     0,
     635,     0,     0,     0,     0,     0,     0,     0,    32,    33,
      34,     0,     8,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    35,     0,     0,    36,    37,     0,
       0,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,     0,    21,    22,    23,    24,    25,   457,
     458,   459,   460,    38,   461,     0,     0,    40,    27,    28,
      29,    30,    31,     0,     0,     0,     0,   462,   463,   464,
     465,   466,   467,   468,   469,   470,   471,   472,   473,    32,
      33,    34,     0,     8,     0,     0,     0,   637,     0,     0,
       0,     0,     0,     0,     0,    35,   248,     0,    36,    37,
       0,     0,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,     0,    21,    22,    23,    24,    25,
       0,     0,     0,     0,    38,     0,     0,     0,    40,    27,
      28,    29,    30,    31,     0,     0,   457,   458,   459,   460,
       0,   461,     0,     0,     0,     0,     0,     0,     0,     0,
      32,    33,    34,     8,   462,   463,   464,   465,   466,   467,
     468,   469,   470,   471,   472,   473,    35,     0,     0,    36,
      37,     0,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,     0,    21,    22,    23,    24,    25,
       0,     0,     0,     0,     0,    38,     0,     0,     0,    40,
       0,     0,    30,    31,     0,     0,     0,     0,     0,     0,
       0,     0,   755,     0,     0,     0,     0,     0,     0,     0,
      32,    33,    34,     8,   761,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    35,     0,     0,    36,
      37,     0,     9,    10,    11,    12,    13,    14,    15,    16,
     708,    18,   709,    20,     0,   710,    22,    23,    24,    25,
       0,   457,   458,   459,   460,    38,   461,     0,     0,    40,
       0,     0,     0,   457,   458,   459,   460,     0,   461,   462,
     463,   464,   465,   466,   467,   468,   469,   470,   471,   472,
     473,   462,   463,   464,   465,   466,   467,   468,   469,   470,
     471,   472,   473,     0,     0,     0,    35,     0,     0,    36,
      37,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    38,     0,     0,     0,    40
};

static const yytype_int16 yycheck[] =
{
       0,     5,    70,    59,   138,     5,   156,   146,   542,   154,
      40,   595,   148,   380,   361,     5,   556,     1,     2,   711,
      45,    92,   117,   237,    92,    93,   222,   568,   527,     5,
     822,   536,    43,   676,   776,   388,     3,   536,   177,   178,
       6,   259,     6,    46,   240,    45,    98,     6,   430,     5,
       5,     5,    21,    40,     6,     1,     2,   671,  1075,    41,
       3,    42,    54,     0,    41,   679,    44,    40,    41,    69,
     138,   187,    40,     5,    74,    75,   943,   193,   945,    44,
      42,    46,     5,   199,    57,   230,   138,     5,   204,     5,
    1107,   705,    54,    59,    46,   147,   296,   297,    42,    77,
     300,   196,     5,    54,     5,     5,   720,    45,   111,   109,
       5,   103,    40,    59,     5,     5,    85,   233,    49,    50,
      51,    85,     5,   104,     5,   343,    85,   243,     4,   159,
       6,    46,     3,   875,   116,   230,   103,   677,   190,   116,
       3,   166,   129,   168,   886,   837,    89,    90,   148,   205,
     295,     5,   208,   248,   118,   250,     3,     5,   525,     3,
     104,    43,    41,   955,   148,   116,   166,    43,   168,    40,
      46,   316,   228,   787,    54,   118,   119,    48,   545,     5,
     236,  1048,  1044,    54,   184,    40,    57,   152,     6,     1,
     105,   106,   117,    70,   547,     5,    40,   162,   126,   581,
     200,   129,   148,   134,     5,    76,   134,    48,   154,    86,
       3,   361,   412,   413,   414,   140,   154,   959,    89,    90,
       3,   103,     4,   103,   162,     6,    89,    90,    40,    92,
     230,  1093,   232,   233,    89,    90,    54,   237,   114,     4,
     780,     6,    89,    90,   115,    89,    90,   126,   119,   249,
     129,   251,   115,   237,    40,   126,   119,  1049,   129,   205,
      46,   286,   208,   134,    46,    46,     6,   138,   115,   912,
    1062,   115,   119,   329,    40,   119,   332,   782,   117,  1021,
      46,    46,   228,   782,   284,    43,   286,   134,   488,   489,
     236,   237,   230,    76,   138,     3,    89,    90,     4,    54,
     914,   140,   240,    43,    74,    48,    89,    90,     3,     1,
     187,   379,    48,  1005,    52,   192,   193,   453,    76,   140,
     388,    76,   199,   394,   201,   202,   394,   204,    43,   711,
     526,    46,   115,    71,  1026,   360,   119,   704,   720,   109,
      46,    43,   219,   884,   558,   222,   223,    40,   286,   295,
      42,    43,    54,   103,    46,  1097,   233,   295,   554,   103,
     360,   345,   562,   240,    57,     3,   243,    40,    52,    53,
     316,     3,   249,   429,   988,   443,   916,     3,   316,   256,
     380,    89,    90,   329,    57,   999,   332,    71,   326,   441,
       3,   105,   106,   977,    89,    90,   778,   542,    43,   345,
      43,    46,    43,    46,   103,    46,   406,   115,     3,   439,
      40,   119,     3,   120,    40,     6,    42,   116,   117,    40,
     536,   359,    48,   118,   119,    43,   426,    57,    54,    40,
     430,    57,    43,    76,   434,    76,    57,   493,    77,    43,
     535,   140,   498,    82,   539,    40,     4,   542,     6,    43,
      76,    89,    90,    52,    53,   837,    40,    89,    90,  1073,
     103,   995,    57,    89,    90,   430,    42,    40,   563,   453,
    1084,    47,    71,    72,     3,    48,    89,    90,   104,   547,
     418,   119,     3,   429,   831,    43,    40,   119,    46,   115,
       3,   429,    40,   119,    89,    90,    54,    40,    89,    90,
     126,    40,   115,   129,   442,    40,   119,   453,   134,    49,
      50,    51,   138,   126,   105,   106,   129,    46,    40,    40,
     115,   134,   522,    84,   119,   525,    43,    48,    40,    46,
     923,   126,    40,   926,   129,     3,    57,    54,    40,   543,
      80,    43,   542,   543,    46,   545,   546,   493,    88,    89,
      90,   551,   498,   543,   538,     4,   581,     6,   583,   570,
      89,    90,    43,    43,    40,    46,    46,   543,    89,    90,
     595,   555,    40,    54,   558,    40,    89,    90,   774,   775,
      48,   581,   677,   583,   584,     6,   115,   543,   543,   543,
     119,   546,   538,    43,   115,   595,    46,   597,   119,    40,
      41,     4,   115,     6,   542,   126,   119,    40,   129,   555,
      48,   543,   558,   134,    84,    85,    57,   138,   686,    48,
     543,    89,    90,  1005,  1098,   543,    48,   543,   593,  1103,
    1104,  1105,   727,    48,   690,   132,   133,   134,   135,   136,
     543,   116,   543,   543,  1026,    41,  1120,   115,   543,   526,
     654,   119,   543,   543,   654,   593,    42,   595,   126,   536,
     543,   129,   543,    43,   654,   721,   802,    46,   784,     6,
     138,   671,    46,    40,    41,   552,    42,   554,   654,   679,
     684,   831,    41,    42,   684,   121,   122,   123,   783,   543,
      57,  1073,    40,    41,   684,   543,    43,   792,   654,   654,
     654,    41,   702,   703,   704,   705,   671,   732,   684,    57,
     116,   711,   702,   852,   679,   719,   741,   543,   932,   719,
     720,   777,   654,    48,   795,    48,   702,   795,   684,   684,
     684,   654,   732,   543,    21,   746,   654,    42,   654,   739,
     705,   741,   543,   104,   690,    40,   711,   702,   702,   817,
     884,   654,   684,   654,   654,   720,   808,    44,   104,   654,
       7,   684,    41,   654,   654,   817,   684,   823,   684,  1116,
      41,   654,   116,   654,   830,   721,   776,    49,    50,    51,
     745,   684,    48,   684,   684,    41,    42,   787,    57,   684,
      77,    78,    79,   684,   684,    82,    40,    84,    85,   949,
     654,   684,   802,   684,    52,    53,   654,   745,   746,   686,
      40,    41,   134,   135,   136,   819,   884,    48,   802,   819,
     915,    48,   787,    71,    72,   964,   894,    57,   654,   819,
     684,   777,   884,   833,    41,    42,   684,   837,   933,   839,
     935,    52,    53,   819,   654,    73,    74,    75,   134,    48,
     995,   946,   116,   654,    54,   103,   802,    43,   684,    80,
      71,    72,   930,   819,   819,   819,   934,    88,    89,    90,
      41,   966,   837,    41,   684,   875,    76,   823,    40,    41,
      80,    41,     3,   684,   830,     6,   886,   819,    88,    89,
      90,    41,   103,    54,    42,    57,   819,   774,   775,    41,
     995,   819,    41,   819,    41,   160,   161,   784,    43,    41,
      41,    41,    41,    41,   914,   170,   819,   104,   819,   819,
      41,   798,   799,   800,   819,    42,   116,    41,   819,   819,
      43,   116,   932,   111,  1029,    76,   819,    46,   819,   943,
      48,   945,   998,   943,   116,   945,   116,    48,   932,   914,
     827,    48,   977,    41,    41,    48,    48,   957,   958,   959,
      43,    43,    48,  1113,    48,   819,  1116,  1023,    89,    90,
      48,   819,    46,    41,    46,    40,    43,   977,    47,    43,
      91,    57,    41,  1039,   105,   106,   932,    41,   988,    40,
      90,    45,    48,   819,    89,   995,    46,    78,    52,   999,
      41,    41,    48,    48,  1060,  1005,  1074,    42,    42,   819,
      48,    48,    48,  1069,  1070,  1071,    48,    48,   819,    40,
      43,  1021,   140,   988,    54,    41,  1026,    47,   103,    62,
      42,    85,    86,    41,   999,  1091,    41,    41,    40,   977,
    1005,   296,   297,    43,  1048,   300,    40,    48,  1048,    48,
      48,    48,   998,    48,    41,    41,    41,   995,  1058,   997,
     998,  1026,    54,   117,   130,   131,   132,   133,   134,   135,
     136,    43,    54,  1073,    47,    46,   138,  1023,    47,    47,
     335,   336,    48,    76,  1084,    76,    76,  1087,    76,    48,
      41,   145,     3,  1039,   157,   158,    42,  1097,  1098,    43,
     154,    40,    43,  1103,  1104,  1105,   160,   161,  1073,   163,
      49,    50,    51,    41,  1060,    54,   170,    54,    41,  1084,
    1120,   104,  1087,  1069,  1070,  1071,  1003,    43,    48,    40,
      49,    50,    51,  1098,    40,    54,    76,    48,  1103,  1104,
    1105,    80,   196,    76,    76,  1091,    57,    76,    47,    88,
      89,    90,    76,    43,    43,  1120,    43,    76,    43,   104,
     214,    80,     3,    43,     3,     6,    42,    48,    40,    88,
      89,    90,    41,    48,    40,    43,   230,    68,    89,    90,
      43,    40,    47,    41,   238,    40,   240,   241,   242,    41,
     244,    48,    48,    41,   248,   249,   250,   251,    48,    40,
      91,    40,    48,    41,   115,    96,    41,    48,   119,   272,
      41,   274,   275,   276,   277,   278,   279,   280,    57,    41,
     475,   476,   477,    46,    48,    41,    43,   138,    48,   484,
     485,    43,   286,   527,   551,   702,   932,  1029,     3,   494,
     558,   295,   296,   297,   307,   703,   300,   702,    89,    90,
      89,    90,   522,   802,    21,   702,   702,   453,   597,   568,
     515,    50,   316,   161,   318,   434,  1003,   330,  1058,   839,
     333,   958,   170,   576,   115,    40,   115,    44,   119,  1073,
     119,   335,   336,    48,  1053,   126,   340,   126,   129,   687,
     129,    -1,    57,   134,   178,    -1,   187,   138,   352,   138,
      -1,   192,   193,    -1,    -1,   359,    -1,    -1,   199,    -1,
      77,    78,    79,   204,   368,    82,    83,    84,    85,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,   381,    -1,    -1,
      -1,   222,    -1,    -1,     3,     4,    -1,     6,    -1,     3,
      -1,    -1,   233,   234,   242,    -1,   244,    -1,   239,   240,
     115,   118,   243,    -1,   119,    -1,    -1,    -1,   412,   413,
     414,    -1,    -1,    -1,   418,   620,   420,   421,    -1,   423,
      -1,    40,    -1,   138,    43,   429,    40,    46,    -1,    48,
      -1,    -1,    -1,    -1,    48,     3,    -1,    -1,   107,   108,
      -1,     3,    -1,    57,   457,   458,   459,   460,   461,   462,
     463,   464,   465,   466,   467,   468,   469,   470,   471,   472,
     473,   130,   131,   132,   133,   134,   135,   136,    -1,    -1,
      89,    90,    40,   477,    -1,    89,    90,   490,    40,    -1,
       3,    -1,    -1,    -1,   488,   489,   499,   335,   336,    57,
     494,    -1,   340,    -1,    -1,   114,   115,     3,    -1,    -1,
     119,   115,    -1,    -1,    -1,   119,    -1,   126,    -1,    -1,
     129,   515,    -1,    -1,   518,   134,    -1,    40,    -1,   138,
      43,    89,    90,    -1,   138,    48,    -1,    89,    90,    -1,
      -1,   535,    -1,    -1,    40,   539,    -1,    43,   542,    -1,
      -1,    -1,    48,    -1,    -1,    -1,    -1,   115,    -1,     3,
      -1,   119,    -1,   115,    -1,    -1,    -1,   119,   562,   563,
     564,    -1,    -1,   567,   568,     3,    89,    90,    -1,    -1,
     138,    -1,   420,   421,    -1,   423,   138,   590,   591,   592,
     593,    -1,    -1,    89,    90,   589,    40,    -1,    -1,    43,
      -1,   595,   115,    -1,    48,    -1,   119,    -1,    -1,    -1,
      -1,    -1,    40,   126,    -1,    43,   129,    -1,    -1,   115,
      48,   134,     3,   119,    -1,   138,    -1,    -1,    -1,    -1,
     126,    -1,    -1,   129,   628,   629,    -1,    -1,   134,   633,
      -1,    -1,   138,    -1,   638,    89,    90,    -1,    -1,   643,
      -1,    -1,    -1,    -1,    -1,    -1,   494,    -1,    -1,    40,
      -1,    89,    90,    -1,    -1,    -1,    -1,    48,    -1,    -1,
      -1,   115,    -1,    -1,    -1,   119,    -1,   515,    -1,    -1,
     518,    -1,   126,   677,    -1,   129,    -1,   115,    -1,    -1,
     134,   119,   523,    -1,   138,   526,   527,    -1,   126,    -1,
      -1,   129,    -1,   534,   535,   536,   134,   701,    89,    90,
     138,    -1,    -1,   707,    -1,     3,    -1,     3,    -1,   722,
      -1,    -1,    -1,   554,    -1,   556,   564,    -1,    -1,   567,
      -1,    -1,   563,   727,   115,     3,    -1,    -1,   119,    -1,
     743,   744,   745,    -1,   747,   126,    -1,    -1,   129,    -1,
     753,    -1,    40,   134,    40,    -1,    -1,   138,    -1,    -1,
      48,    -1,    48,    -1,    -1,    -1,    -1,    -1,    -1,    57,
     764,   765,    40,    -1,    -1,    -1,   770,    -1,    -1,    -1,
      48,     3,    -1,    -1,    -1,   107,   108,   109,   110,   783,
     628,   629,    -1,    -1,    -1,   633,    -1,    -1,   792,     3,
     638,    89,    90,    89,    90,   643,   128,   129,   130,   131,
     132,   133,   134,   135,   136,    -1,    -1,    -1,    40,    -1,
      -1,    89,    90,    -1,    -1,    -1,    48,   115,    -1,   115,
      -1,   119,    -1,   119,    -1,    57,    40,    -1,   126,    -1,
     126,   129,     3,   129,    48,    -1,   677,   115,   134,   843,
     138,   119,   138,    -1,   685,    -1,   687,    -1,   126,    -1,
      -1,   129,    -1,    -1,    -1,    -1,   134,    89,    90,   707,
     138,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,    40,
      -1,    -1,    -1,    -1,    -1,    89,    90,    48,   891,   892,
     884,    -1,    -1,   115,    -1,    -1,    -1,   119,    -1,    -1,
      -1,     3,    -1,    -1,   126,   899,    -1,   129,    -1,   903,
     904,   115,    -1,    -1,    40,   119,   138,    -1,    -1,    -1,
      -1,   915,   126,    -1,    -1,   129,   764,   765,    89,    90,
     134,    57,   770,     3,   138,    -1,     3,    -1,    40,   933,
      -1,   935,    -1,   774,   775,    -1,    48,    -1,    -1,   780,
      -1,   782,   946,   784,   115,    57,   950,    -1,   119,    -1,
       3,    -1,    -1,    89,    90,   126,    -1,    -1,   129,    -1,
      40,    -1,   966,    40,    -1,   806,    -1,   138,    48,    -1,
      -1,    48,    -1,   977,    -1,   979,   980,    89,    90,   115,
      -1,    -1,    -1,   119,    -1,    -1,    -1,    40,    -1,  1002,
     126,   995,    -1,   129,   998,    48,    -1,    -1,    -1,    -1,
      -1,    -1,   138,   115,    -1,    -1,    -1,   119,    -1,    89,
      90,    -1,    89,    90,  1027,  1019,    -1,    -1,    -1,    -1,
       6,    -1,    -1,    -1,    -1,  1029,   138,    13,    14,    15,
      16,    17,    18,    19,    20,   115,    89,    90,   115,   119,
      -1,    -1,   119,    -1,   107,   108,   109,   110,    -1,  1053,
      -1,   899,    -1,    -1,    -1,   903,   904,    -1,   138,    -1,
      -1,   138,   115,    -1,    -1,    -1,   119,   130,   131,   132,
     133,   134,   135,   136,     1,   916,     3,     4,    -1,    -1,
      -1,     8,     9,    10,    -1,   138,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    -1,    35,    36,
      37,    38,    39,    -1,    -1,    -1,    43,    44,    -1,    -1,
      47,    -1,    49,    50,    51,    52,    53,    -1,    -1,    -1,
      -1,   979,   980,    -1,    -1,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      77,    78,    79,    -1,    81,    82,    83,    84,    85,    86,
      87,    -1,    89,    90,    91,    -1,    -1,    -1,    95,    -1,
      97,    98,    99,   100,   101,   102,    -1,     3,    -1,    -1,
       6,    -1,    -1,    -1,    -1,    -1,   113,    -1,   115,    -1,
      -1,   118,   119,   120,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     3,    -1,    -1,     6,     7,    -1,
       3,   138,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    -1,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    -1,    35,    36,    37,    38,
      39,    40,    -1,    -1,    -1,    -1,    -1,    40,    -1,    48,
      49,    50,    51,    52,    53,    48,    55,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    -1,    -1,
      -1,    70,    71,    72,    -1,    -1,    -1,    -1,    -1,   105,
     106,    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,   115,
      89,    90,     3,   119,    -1,    -1,    89,    90,    -1,    -1,
     126,    -1,    -1,   129,    -1,    -1,    -1,    -1,   134,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,   118,
     119,    -1,   115,    -1,    -1,    -1,   119,   126,    -1,    40,
     129,    -1,    -1,   132,   133,   134,    -1,    48,     3,   138,
     139,     6,     7,    -1,     3,   138,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    -1,
      35,    36,    37,    38,    39,    40,    -1,    -1,    89,    90,
      -1,    40,    -1,    -1,    49,    50,    51,    52,    53,    48,
      55,    -1,    -1,    58,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   115,    70,    71,    72,   119,    -1,
      -1,    -1,    -1,    -1,    -1,   126,    -1,    -1,   129,    -1,
      -1,    86,    -1,    -1,    89,    90,     3,   138,    -1,     3,
      89,    90,     6,    -1,    -1,    -1,    -1,    -1,    -1,    13,
      14,    15,    16,    17,    18,    19,    20,    -1,    -1,    -1,
     115,    -1,    -1,    -1,   119,    -1,   115,    -1,    -1,    -1,
     119,   126,    -1,    40,   129,    -1,    -1,   132,   133,   134,
      -1,    48,     3,   138,   139,     6,     7,    -1,    41,   138,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    -1,    35,    36,    37,    38,    39,    40,
      -1,    -1,    89,    90,    -1,    89,    90,    -1,    49,    50,
      51,    52,    53,    -1,    55,    -1,    -1,    58,    -1,    -1,
      -1,   105,   106,    -1,    -1,    -1,    -1,    -1,   115,    70,
      71,    72,   119,    -1,   107,   108,   109,   110,    -1,   112,
      -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,
      -1,   138,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,    -1,    -1,    22,    23,    24,    25,
      26,    27,    28,    29,   115,    31,    -1,    33,   119,    -1,
      36,    37,    38,    39,    -1,   126,    -1,    -1,   129,    -1,
      -1,   132,   133,   134,    -1,    -1,     3,   138,   139,     6,
       7,    41,    -1,    -1,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    -1,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    -1,    35,    36,
      37,    38,    39,    40,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    49,    50,    51,    52,    53,    -1,    55,    -1,
      -1,    58,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    70,    71,    72,    -1,   107,   108,   109,
     110,    -1,   112,    -1,    -1,   107,   108,   109,   110,    86,
      -1,    -1,    89,    90,    -1,   125,   126,   127,   128,   129,
     130,   131,   132,   133,   134,   135,   136,   129,   130,   131,
     132,   133,   134,   135,   136,    -1,    -1,    -1,   115,    -1,
      -1,    -1,   119,    -1,    -1,    -1,    -1,    -1,    -1,   126,
      -1,    -1,   129,    -1,    -1,   132,   133,   134,    -1,    -1,
       3,   138,   139,     6,     7,    -1,    43,    -1,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    -1,    35,    36,    37,    38,    39,    40,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    51,    52,
      53,    -1,    55,    -1,    -1,    58,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,
     107,   108,   109,   110,    -1,   112,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    86,    -1,    -1,    89,    90,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   115,    -1,    -1,    -1,   119,    -1,    -1,    -1,
      -1,    -1,    -1,   126,    -1,    -1,   129,    -1,    -1,   132,
     133,   134,    -1,    -1,     3,   138,   139,     6,     7,    -1,
      -1,    -1,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    -1,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    -1,    35,    36,    37,    38,
      39,    40,    41,    76,    -1,    -1,    -1,    -1,    -1,    -1,
      49,    50,    51,    52,    53,    -1,    55,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    70,    71,    72,   107,   108,   109,   110,    -1,   112,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,
      89,    90,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,
     119,    -1,    -1,    -1,    -1,    -1,    -1,   126,    -1,    -1,
     129,    -1,    -1,   132,   133,   134,    -1,    -1,     3,   138,
     139,     6,     7,    -1,    -1,    -1,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    -1,
      35,    36,    37,    38,    39,    40,    41,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    50,    51,    52,    53,    -1,
      55,   107,   108,   109,   110,    -1,   112,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,    86,    -1,    -1,    89,    90,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     115,    -1,    -1,    -1,   119,    -1,    -1,    -1,    -1,    -1,
      -1,   126,    -1,    -1,   129,    -1,    -1,   132,   133,   134,
      -1,    -1,     3,   138,   139,     6,     7,    -1,    -1,    -1,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    -1,    35,    36,    37,    38,    39,    40,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,
      51,    52,    53,    -1,    55,   107,   108,   109,   110,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,
      71,    72,    -1,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,    86,    -1,    -1,    89,    90,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    -1,
      -1,    -1,    -1,    -1,    -1,   126,    -1,    -1,   129,    -1,
      -1,   132,   133,   134,    -1,    -1,     3,   138,   139,     6,
       7,    -1,    -1,    -1,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    -1,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    -1,    35,    36,
      37,    38,    39,    40,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    49,    50,    51,    52,    53,    -1,    55,   107,
     108,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    70,    71,    72,    -1,    -1,     3,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,    86,
      -1,    -1,    89,    90,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    -1,
      35,    36,    37,    38,    39,    40,    -1,    -1,   115,    -1,
      -1,    -1,   119,    48,    49,    50,    51,    52,    53,    -1,
      -1,    -1,    57,    -1,    -1,   132,   133,   134,    -1,    -1,
      -1,   138,   139,    -1,    -1,    70,    71,    72,   107,   108,
     109,   110,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    86,    -1,    -1,    89,    90,    -1,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     115,     3,    -1,   118,   119,    -1,    -1,    -1,    -1,    -1,
      -1,   126,    -1,    -1,   129,    -1,    -1,    -1,    -1,   134,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    -1,    35,    36,    37,    38,    39,    40,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    48,    49,    50,    51,
      52,    53,    -1,    -1,    -1,    57,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    -1,    35,    36,    37,    38,    39,
      -1,    -1,    -1,   115,    -1,    -1,   118,   119,    48,    49,
      50,    51,    52,    53,   126,    -1,    -1,   129,    -1,    -1,
      -1,    -1,   134,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    89,
      90,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    -1,    35,    36,    37,    38,    39,
      -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,   118,   119,
      -1,    -1,    52,    53,    -1,    -1,   126,    -1,    -1,   129,
      -1,    -1,    -1,    -1,   134,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,     0,     1,    -1,     3,     4,    -1,    -1,
      -1,     8,     9,    10,    -1,    -1,    86,    -1,    -1,    89,
      90,    -1,    -1,    -1,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    -1,    35,    36,
      37,    38,    39,    -1,    -1,   115,    43,    44,    -1,   119,
      -1,    -1,    49,    50,    51,    52,    53,    -1,    -1,    56,
      -1,    -1,    -1,    -1,   134,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    -1,    -1,    -1,    -1,
      77,    78,    79,    -1,    81,    82,    -1,    84,    85,    86,
      87,    -1,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,     1,    -1,     3,     4,
      -1,    -1,    -1,     8,     9,    10,   113,    -1,   115,    -1,
      -1,   118,   119,    -1,    -1,    -1,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    -1,
      35,    36,    37,    38,    39,    -1,    -1,    -1,    43,    44,
      -1,    -1,    47,    -1,    49,    50,    51,    52,    53,    -1,
      -1,    56,    -1,    -1,    -1,    -1,    -1,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    -1,    -1,
      -1,    -1,    77,    78,    79,    -1,    81,    82,    -1,    84,
      85,    86,    87,    -1,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,     1,    -1,
       3,     4,    -1,    -1,    -1,     8,     9,    10,   113,    -1,
     115,    -1,    -1,   118,   119,    -1,    -1,    -1,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    -1,    35,    36,    37,    38,    39,    -1,    -1,    -1,
      43,    44,    -1,    -1,    47,    -1,    49,    50,    51,    52,
      53,    -1,    -1,    56,    -1,    -1,    -1,    -1,    -1,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      -1,    -1,    -1,    -1,    77,    78,    79,    -1,    81,    82,
      -1,    84,    85,    86,    87,    -1,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
       1,    -1,     3,     4,    -1,    -1,    -1,     8,     9,    10,
     113,    -1,   115,    -1,    -1,   118,   119,    -1,    -1,    -1,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    -1,    35,    36,    37,    38,    39,    -1,
      -1,    -1,    43,    44,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    -1,    -1,    56,    -1,    -1,    -1,    -1,
      -1,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    -1,    -1,    -1,    -1,    77,    78,    79,    -1,
      81,    82,    -1,    84,    85,    86,    87,    -1,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,     1,    -1,     3,     4,    -1,    -1,    -1,     8,
       9,    10,   113,    -1,   115,    -1,    -1,   118,   119,    -1,
      -1,    -1,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    -1,    35,    36,    37,    38,
      39,    -1,    -1,    -1,    43,    44,    -1,    -1,    -1,    -1,
      49,    50,    51,    52,    53,    -1,    -1,    56,    -1,    -1,
      -1,    60,    -1,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    -1,    -1,    -1,    -1,    77,    78,
      79,    -1,    81,    82,    -1,    84,    85,    86,    87,    -1,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,     1,    -1,     3,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   113,    -1,   115,    -1,    -1,   118,
     119,    -1,    -1,    -1,    -1,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    -1,    35,    36,
      37,    38,    39,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    48,    49,    50,    51,    52,    53,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    70,    71,    72,     1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    86,
      -1,    -1,    89,    90,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    -1,
      35,    36,    37,    38,    39,    -1,    -1,    -1,   115,    -1,
      -1,   118,   119,    48,    49,    50,    51,    52,    53,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    70,    71,    72,     1,    -1,
       3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    86,    -1,    -1,    89,    90,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    -1,    35,    36,    37,    38,    39,    -1,    -1,    -1,
     115,    -1,    -1,   118,   119,    -1,    49,    50,    51,    52,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    58,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,    -1,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    -1,    35,    36,    37,    38,    39,   107,   108,
     109,   110,   115,   112,    -1,    -1,   119,    49,    50,    51,
      52,    53,    -1,    -1,    -1,    -1,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,    70,    71,
      72,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    87,    -1,    89,    90,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    -1,    35,    36,    37,    38,    39,
      -1,    -1,    -1,   115,    -1,    -1,    -1,   119,   120,    49,
      50,    51,    52,    53,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    86,    87,    13,    89,
      90,    -1,    -1,    -1,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    -1,
      35,    36,    37,    38,    39,   115,    -1,    -1,    -1,   119,
     120,    -1,    -1,    48,    49,    50,    51,    52,    53,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,
       3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    86,    -1,    -1,    89,    90,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    -1,    35,    36,    37,    38,    39,    -1,    -1,    -1,
     115,    -1,    -1,   118,   119,    48,    49,    50,    51,    52,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,    -1,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    -1,    35,    36,    37,    38,    39,    40,    -1,
      -1,    -1,   115,    -1,    -1,   118,   119,    49,    50,    51,
      52,    53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      58,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    -1,    35,    36,    37,    38,    39,   107,
     108,   109,   110,   115,   112,    -1,    -1,   119,    49,    50,
      51,    52,    53,    -1,    -1,    -1,    -1,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,    70,
      71,    72,    -1,     3,    -1,    -1,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    86,    87,    -1,    89,    90,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    -1,    35,    36,    37,    38,    39,
      -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    49,
      50,    51,    52,    53,    -1,    -1,   107,   108,   109,   110,
      -1,   112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,     3,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,    86,    -1,    -1,    89,
      90,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    -1,    35,    36,    37,    38,    39,
      -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,   119,
      -1,    -1,    52,    53,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    58,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,     3,    58,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    89,
      90,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    -1,    35,    36,    37,    38,    39,
      -1,   107,   108,   109,   110,   115,   112,    -1,    -1,   119,
      -1,    -1,    -1,   107,   108,   109,   110,    -1,   112,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,    -1,    -1,    -1,    86,    -1,    -1,    89,
      90,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,   119
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   121,   122,   123,   142,   143,   311,     1,     3,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    35,    36,    37,    38,    39,    48,    49,    50,    51,
      52,    53,    70,    71,    72,    86,    89,    90,   115,   118,
     119,   193,   236,   250,   251,   253,   254,   255,   256,   257,
     258,   283,   284,   296,   299,   301,     1,   236,     1,    40,
       0,     1,     4,     8,     9,    10,    21,    43,    44,    56,
      62,    63,    64,    65,    66,    67,    68,    69,    77,    78,
      79,    81,    82,    84,    85,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   113,   118,   144,
     145,   146,   148,   149,   150,   151,   152,   155,   156,   158,
     159,   160,   161,   162,   163,   164,   167,   168,   169,   172,
     174,   179,   180,   181,   182,   184,   188,   195,   196,   197,
     198,   199,   203,   204,   211,   212,   223,   231,   232,   311,
      48,    52,    71,    48,    48,    40,   140,   103,   103,   295,
     299,    43,   254,   250,    40,    48,    54,    57,    76,   119,
     126,   129,   134,   138,   241,   242,   244,   246,   247,   248,
     249,   299,   311,   250,   257,   299,   295,   117,   140,   300,
      43,    43,   233,   234,   236,   311,   120,    40,     6,    85,
     118,   305,    40,   308,   311,     1,   252,   253,   296,    40,
     308,    40,   166,   311,    40,    40,    84,    85,    40,    84,
      77,    82,    44,    77,    92,   299,    46,   296,   299,    40,
       4,    46,    40,    40,    43,    46,     4,   305,    40,   178,
     252,   176,   178,    40,    40,   305,    40,   103,   284,   308,
      40,   126,   129,   244,   249,   299,    21,    85,    87,   193,
     252,   284,    48,    48,    48,   299,   118,   119,   301,   302,
     284,     7,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    40,    55,   126,   129,   132,   133,   134,   138,
     139,   236,   237,   238,   240,   252,   253,   269,   270,   271,
     272,   305,   306,   311,   250,    40,   126,   129,   233,   247,
     249,   299,    48,    46,   105,   106,   259,   260,   261,   262,
     263,    58,   269,   270,   269,     3,    40,    48,   138,   245,
     248,   299,    48,   245,   248,   249,   250,   299,   241,    40,
      57,   241,    40,    57,    48,   126,   129,   245,   248,   299,
     116,   301,   119,   302,    41,    42,   235,   311,   261,   296,
     297,   305,   284,     6,    46,   297,   309,   297,    43,    40,
     244,    54,    41,   297,    52,    53,    71,   285,   286,   311,
     296,   296,   297,    13,   173,   233,   233,   299,    43,    54,
     214,    54,    46,   296,   175,   309,   296,   233,    46,   243,
     244,   247,   311,    43,    42,   177,   311,   297,   298,   311,
     153,   154,   305,   233,   207,   208,   209,   236,   283,   311,
     299,   305,   126,   129,   249,   296,   299,   309,    40,   297,
     126,   129,   299,   116,   244,   299,   264,   296,   311,    40,
     244,    76,   275,   276,   299,   311,    48,    48,    41,   296,
     300,   104,   269,    40,    48,   269,   269,   269,   269,   269,
     269,   269,   104,    42,   239,   311,    40,   107,   108,   109,
     110,   112,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,     7,   126,   129,   249,   299,   246,
     299,   246,    41,    41,   126,   129,   246,   299,   116,    48,
      57,   269,    58,    40,   249,   299,    48,   299,    40,    57,
      48,   249,   233,    58,   269,   233,    58,   269,    48,    48,
     245,   248,    48,   245,   248,   116,    48,   126,   129,   245,
     300,    43,   236,    41,   299,   183,    42,    54,    41,   241,
     259,    41,   299,    41,    54,    41,    42,   171,    42,    41,
      41,    43,   252,   143,   299,   213,    41,    41,    41,    41,
     176,   178,    41,    41,    42,    46,    41,   104,    42,   210,
     311,    59,   116,    41,   249,   299,    43,   116,   111,    54,
      76,   194,   311,   233,   299,    80,    88,    89,    90,   186,
     241,   250,   288,   289,   277,    46,    43,   275,   295,   284,
      41,   126,   129,   134,   249,   252,    48,   240,   269,   269,
     269,   269,   269,   269,   269,   269,   269,   269,   269,   269,
     269,   269,   269,   269,   269,   299,   116,    41,    41,    41,
     116,   246,   246,   269,   233,   245,   299,    41,   116,    48,
     233,    58,   269,    48,    41,    58,    41,    58,    48,    48,
      48,    48,   126,   129,   245,   248,    48,    48,    48,   245,
     235,     4,    46,   305,   143,   309,   153,   271,   305,   310,
      43,    43,    46,     4,   165,   305,     4,    43,    46,   114,
     170,   244,   305,   307,   297,   305,   310,    41,   236,   244,
      46,   243,    47,    43,   143,    44,   232,   176,    43,    46,
      40,    47,   177,   115,   119,   296,   303,    43,   309,   236,
     170,    91,   205,   209,   157,   244,   305,   116,    30,    32,
      35,   187,   255,   256,   299,    57,   189,   254,    43,    46,
      41,    40,    40,   288,    90,    89,     1,    42,    43,    46,
     185,   241,   289,   241,    78,   278,   279,   287,   311,   201,
      46,   299,   269,    41,    41,   134,   250,    41,   126,   129,
     242,    48,   239,    76,    41,    58,    41,    41,   245,   245,
      41,    58,   245,   245,    48,    48,    48,    48,    48,   245,
      48,    48,    48,    47,    42,    42,   147,    40,   250,   289,
     290,   311,    54,    41,    42,   171,   170,   244,   290,    43,
      47,   305,   252,   296,    43,    54,   307,   233,   140,   117,
     140,   304,   103,    41,    47,   299,    44,    83,   118,   184,
     199,   203,   204,   206,   220,   222,   224,   232,   210,   143,
     290,    43,   186,    40,    46,   190,   150,   265,   266,   311,
      40,    54,   289,   290,   291,   233,   269,   244,   241,    42,
      73,    74,    75,   280,   282,     1,    43,    66,    73,    74,
      75,    78,   138,   148,   149,   150,   151,   155,   156,   160,
     162,   164,   167,   169,   172,   174,   179,   180,   181,   182,
     199,   203,   204,   211,   215,   219,   220,   221,   222,   223,
     224,   225,   226,   229,   232,   311,   200,   241,   269,   269,
     269,    41,    41,    41,    40,   269,    41,   245,   245,    48,
      48,    48,   245,    48,    48,   309,   309,   215,   233,   289,
     170,   305,   310,    43,   244,    41,   290,    43,   244,    43,
     178,    41,   119,   296,   296,   119,   296,   237,     4,    46,
      54,    40,   103,    87,   120,   252,    60,    43,   185,   233,
      40,    43,   191,   267,   296,    42,    47,   233,   259,    54,
      76,   292,   311,    41,    41,   186,   279,   299,   281,   218,
      46,    76,    76,    76,   138,   301,    47,   215,    30,    32,
      35,   230,   256,   299,   215,   269,   269,   252,   245,    48,
      48,   245,   245,    47,    41,   171,   290,    43,   244,   170,
      43,    43,   304,   304,   104,   252,   207,   252,    40,    41,
     192,   265,    54,   265,    43,   244,    41,    43,   261,   293,
     294,   299,    43,    46,   185,    48,   273,   274,   311,   287,
     215,   216,   301,    40,    43,   202,   244,    76,    43,    47,
     242,   245,   245,    43,    43,   290,    43,   243,   104,    40,
     126,   129,   249,   233,   290,    43,   268,   269,   267,   186,
      43,    46,    43,    42,    48,    40,    46,    48,   299,   215,
      40,   233,   186,   269,   202,    41,    43,    43,   233,    40,
      40,    40,   129,    41,   111,   190,   265,   185,   294,    48,
      48,   274,    47,   233,    41,   185,    43,    41,   233,   233,
     233,    40,   291,   252,   191,    48,    48,   217,    41,   227,
     290,   228,   290,    41,    41,    41,   233,   190,    48,   215,
     228,    43,    46,    54,    43,    46,    54,   228,   228,   228,
      41,   191,    48,   261,   259,   228,    43,    43
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   141,   142,   142,   142,   142,   142,   142,   142,   143,
     143,   144,   144,   144,   144,   144,   144,   144,   145,   145,
     145,   145,   145,   145,   145,   145,   145,   145,   145,   145,
     145,   145,   145,   145,   145,   145,   145,   145,   145,   147,
     146,   148,   149,   150,   150,   150,   151,   151,   152,   152,
     152,   152,   153,   154,   154,   155,   155,   155,   157,   156,
     158,   158,   159,   159,   160,   160,   160,   160,   161,   162,
     162,   163,   163,   164,   164,   165,   165,   166,   166,   167,
     167,   167,   168,   168,   169,   169,   169,   169,   169,   169,
     169,   169,   170,   170,   170,   171,   171,   172,   173,   173,
     174,   174,   174,   175,   176,   177,   177,   178,   178,   178,
     179,   180,   181,   182,   182,   182,   183,   182,   182,   182,
     182,   184,   184,   185,   185,   185,   185,   186,   186,   186,
     186,   187,   187,   187,   187,   187,   187,   188,   188,   188,
     189,   190,   191,   192,   191,   193,   193,   193,   194,   194,
     195,   196,   196,   197,   198,   198,   198,   198,   198,   198,
     200,   199,   201,   199,   202,   202,   203,   205,   204,   204,
     204,   206,   206,   206,   206,   206,   206,   206,   207,   208,
     208,   209,   209,   210,   210,   211,   211,   213,   212,   214,
     212,   212,   215,   216,   217,   215,   215,   215,   218,   215,
     219,   219,   219,   219,   219,   219,   219,   219,   219,   219,
     219,   219,   219,   219,   219,   219,   219,   219,   219,   220,
     221,   221,   222,   222,   222,   222,   222,   223,   224,   225,
     225,   225,   226,   226,   226,   226,   226,   226,   226,   226,
     226,   226,   226,   227,   227,   227,   228,   228,   228,   229,
     230,   230,   230,   230,   230,   231,   232,   232,   232,   232,
     232,   232,   232,   232,   232,   232,   232,   232,   232,   232,
     232,   232,   232,   232,   232,   232,   233,   234,   234,   235,
     235,   236,   236,   236,   237,   238,   238,   239,   239,   240,
     240,   241,   241,   241,   241,   241,   242,   242,   242,   243,
     243,   243,   244,   244,   244,   244,   244,   244,   244,   244,
     244,   244,   244,   244,   244,   244,   244,   244,   244,   244,
     244,   244,   244,   244,   245,   245,   245,   245,   245,   245,
     245,   245,   246,   246,   246,   246,   246,   246,   246,   246,
     246,   246,   247,   247,   247,   247,   247,   247,   247,   247,
     247,   247,   247,   247,   247,   247,   248,   248,   248,   248,
     248,   248,   248,   249,   249,   249,   249,   250,   250,   251,
     251,   251,   252,   253,   253,   253,   253,   254,   254,   254,
     254,   254,   254,   254,   254,   255,   256,   257,   257,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   260,   259,   259,   261,   261,   262,   263,
     264,   264,   265,   265,   266,   266,   266,   266,   267,   267,
     268,   269,   269,   270,   270,   270,   270,   270,   270,   270,
     270,   270,   270,   270,   270,   270,   270,   270,   270,   270,
     270,   271,   271,   271,   271,   271,   271,   271,   271,   272,
     272,   272,   272,   272,   272,   272,   272,   272,   272,   272,
     272,   272,   272,   272,   272,   272,   272,   272,   272,   272,
     272,   273,   274,   274,   275,   277,   276,   276,   278,   278,
     280,   279,   281,   279,   282,   282,   282,   283,   283,   283,
     283,   284,   284,   284,   285,   285,   285,   286,   286,   287,
     287,   288,   288,   288,   288,   289,   289,   289,   289,   289,
     290,   290,   290,   290,   291,   291,   291,   291,   291,   291,
     292,   292,   293,   293,   293,   293,   294,   294,   295,   296,
     296,   296,   297,   297,   297,   298,   298,   299,   299,   299,
     299,   299,   299,   299,   300,   300,   300,   300,   301,   301,
     302,   302,   303,   303,   303,   303,   303,   303,   304,   304,
     304,   304,   305,   305,   306,   306,   307,   307,   307,   308,
     308,   309,   309,   309,   309,   309,   309,   310,   310,   311
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     3,     2,     3,     2,     5,     3,     2,
       1,     1,     1,     1,     1,     1,     1,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     0,
       8,     5,     3,     5,     5,     3,     2,     2,     5,     2,
       5,     2,     4,     1,     1,     7,     7,     5,     0,     7,
       1,     1,     2,     2,     1,     5,     5,     5,     3,     4,
       3,     7,     8,     5,     3,     1,     1,     3,     1,     4,
       7,     6,     1,     1,     7,     9,     8,    10,     5,     7,
       6,     8,     1,     1,     5,     4,     5,     7,     1,     3,
       6,     6,     8,     1,     2,     3,     1,     2,     3,     6,
       5,     9,     2,     1,     1,     1,     0,     6,     1,     6,
      10,     5,     7,     1,     4,     1,     1,     1,     2,     2,
       3,     1,     1,     1,     1,     1,     1,    11,    13,     7,
       1,     1,     1,     0,     3,     1,     2,     2,     2,     1,
       5,     8,    10,     6,     1,     1,     1,     1,     1,     1,
       0,     9,     0,     8,     1,     3,     4,     0,     6,     3,
       4,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     1,     1,     3,     1,     3,     4,     0,     6,     0,
       5,     5,     2,     0,     0,     7,     1,     1,     0,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     6,
       6,     7,     8,     8,     8,     9,     7,     5,     2,     2,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     4,     2,     2,     4,     2,     5,
       1,     1,     1,     1,     1,     2,     1,     1,     2,     2,
       1,     1,     1,     1,     1,     1,     2,     2,     2,     2,
       1,     2,     2,     2,     2,     1,     1,     2,     1,     3,
       1,     2,     7,     3,     1,     2,     1,     3,     1,     1,
       1,     2,     5,     2,     2,     1,     2,     2,     1,     1,
       1,     1,     2,     3,     3,     1,     2,     2,     3,     4,
       5,     4,     5,     6,     6,     4,     5,     5,     6,     7,
       8,     8,     7,     7,     1,     2,     3,     4,     5,     3,
       4,     4,     1,     2,     4,     4,     4,     5,     3,     4,
       4,     5,     1,     2,     2,     2,     3,     3,     1,     2,
       2,     1,     1,     2,     3,     4,     3,     4,     2,     3,
       3,     4,     3,     3,     2,     2,     1,     1,     2,     1,
       1,     1,     1,     2,     1,     2,     3,     1,     1,     1,
       2,     1,     1,     2,     1,     4,     1,     1,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     0,     2,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     5,     3,     3,     1,     1,     3,
       1,     1,     1,     1,     1,     5,     8,     1,     1,     1,
       1,     3,     4,     5,     5,     5,     6,     6,     2,     2,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     5,     2,     2,     2,     2,
       2,     3,     1,     1,     1,     0,     3,     1,     1,     3,
       0,     4,     0,     6,     1,     1,     1,     1,     1,     4,
       4,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     2,     4,     1,     1,     2,     4,
       1,     1,     2,     1,     3,     3,     4,     4,     3,     4,
       2,     1,     1,     3,     4,     6,     2,     2,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     4,     1,
       3,     1,     2,     3,     3,     2,     2,     2,     1,     2,
       1,     3,     2,     4,     1,     3,     1,     3,     3,     2,
       2,     2,     2,     1,     2,     1,     1,     1,     1,     3,
       1,     3,     5,     1,     3,     3,     5,     1,     1,     0
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 1521 "parser.y" /* yacc.c:1646  */
    {
                   if (!classes) classes = NewHash();
		   Setattr((yyvsp[0].node),"classes",classes); 
		   Setattr((yyvsp[0].node),"name",ModuleName);
		   
		   if ((!module_node) && ModuleName) {
		     module_node = new_node("module");
		     Setattr(module_node,"name",ModuleName);
		   }
		   Setattr((yyvsp[0].node),"module",module_node);
	           top = (yyvsp[0].node);
               }
#line 4493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 1533 "parser.y" /* yacc.c:1646  */
    {
                 top = Copy(Getattr((yyvsp[-1].p),"type"));
		 Delete((yyvsp[-1].p));
               }
#line 4502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 1537 "parser.y" /* yacc.c:1646  */
    {
                 top = 0;
               }
#line 4510 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 1540 "parser.y" /* yacc.c:1646  */
    {
                 top = (yyvsp[-1].p);
               }
#line 4518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 1543 "parser.y" /* yacc.c:1646  */
    {
                 top = 0;
               }
#line 4526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 1546 "parser.y" /* yacc.c:1646  */
    {
                 top = (yyvsp[-2].pl);
               }
#line 4534 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 1549 "parser.y" /* yacc.c:1646  */
    {
                 top = 0;
               }
#line 4542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 1554 "parser.y" /* yacc.c:1646  */
    {  
                   /* add declaration to end of linked list (the declaration isn't always a single declaration, sometimes it is a linked list itself) */
                   appendChild((yyvsp[-1].node),(yyvsp[0].node));
                   (yyval.node) = (yyvsp[-1].node);
               }
#line 4552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 1559 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.node) = new_node("top");
               }
#line 4560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 1564 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 1565 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 1566 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 1567 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 4584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 1568 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.node) = 0;
		  if (cparse_unknown_directive) {
		      Swig_error(cparse_file, cparse_line, "Unknown directive '%s'.\n", cparse_unknown_directive);
		  } else {
		      Swig_error(cparse_file, cparse_line, "Syntax error in input(1).\n");
		  }
		  exit(1);
               }
#line 4598 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 1578 "parser.y" /* yacc.c:1646  */
    { 
                  if ((yyval.node)) {
   		      add_symbols((yyval.node));
                  }
                  (yyval.node) = (yyvsp[0].node); 
	       }
#line 4609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 1594 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.node) = 0;
                  skip_decl();
               }
#line 4618 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 1604 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4624 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 1605 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4630 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 1606 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4636 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 1607 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4642 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 1608 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4648 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 1609 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4654 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 1610 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4660 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 1611 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4666 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 1612 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 1613 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4678 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 1614 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4684 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 1615 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4690 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 1616 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4696 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 1617 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4702 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 1618 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4708 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 1619 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4714 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 1620 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 1621 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 1622 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 1623 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 1624 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 4744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 1631 "parser.y" /* yacc.c:1646  */
    {
               Node *cls;
	       String *clsname;
	       extendmode = 1;
	       cplus_mode = CPLUS_PUBLIC;
	       if (!classes) classes = NewHash();
	       if (!classes_typedefs) classes_typedefs = NewHash();
	       clsname = make_class_name((yyvsp[-1].str));
	       cls = Getattr(classes,clsname);
	       if (!cls) {
	         cls = Getattr(classes_typedefs, clsname);
		 if (!cls) {
		   /* No previous definition. Create a new scope */
		   Node *am = Getattr(Swig_extend_hash(),clsname);
		   if (!am) {
		     Swig_symbol_newscope();
		     Swig_symbol_setscopename((yyvsp[-1].str));
		     prev_symtab = 0;
		   } else {
		     prev_symtab = Swig_symbol_setscope(Getattr(am,"symtab"));
		   }
		   current_class = 0;
		 } else {
		   /* Previous typedef class definition.  Use its symbol table.
		      Deprecated, just the real name should be used. 
		      Note that %extend before the class typedef never worked, only %extend after the class typdef. */
		   prev_symtab = Swig_symbol_setscope(Getattr(cls, "symtab"));
		   current_class = cls;
		   SWIG_WARN_NODE_BEGIN(cls);
		   Swig_warning(WARN_PARSE_EXTEND_NAME, cparse_file, cparse_line, "Deprecated %%extend name used - the %s name '%s' should be used instead of the typedef name '%s'.\n", Getattr(cls, "kind"), SwigType_namestr(Getattr(cls, "name")), (yyvsp[-1].str));
		   SWIG_WARN_NODE_END(cls);
		 }
	       } else {
		 /* Previous class definition.  Use its symbol table */
		 prev_symtab = Swig_symbol_setscope(Getattr(cls,"symtab"));
		 current_class = cls;
	       }
	       Classprefix = NewString((yyvsp[-1].str));
	       Namespaceprefix= Swig_symbol_qualifiedscopename(0);
	       Delete(clsname);
	     }
#line 4790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 1671 "parser.y" /* yacc.c:1646  */
    {
               String *clsname;
	       extendmode = 0;
               (yyval.node) = new_node("extend");
	       Setattr((yyval.node),"symtab",Swig_symbol_popscope());
	       if (prev_symtab) {
		 Swig_symbol_setscope(prev_symtab);
	       }
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
               clsname = make_class_name((yyvsp[-4].str));
	       Setattr((yyval.node),"name",clsname);

	       mark_nodes_as_extend((yyvsp[-1].node));
	       if (current_class) {
		 /* We add the extension to the previously defined class */
		 appendChild((yyval.node), (yyvsp[-1].node));
		 appendChild(current_class,(yyval.node));
	       } else {
		 /* We store the extensions in the extensions hash */
		 Node *am = Getattr(Swig_extend_hash(),clsname);
		 if (am) {
		   /* Append the members to the previous extend methods */
		   appendChild(am, (yyvsp[-1].node));
		 } else {
		   appendChild((yyval.node), (yyvsp[-1].node));
		   Setattr(Swig_extend_hash(),clsname,(yyval.node));
		 }
	       }
	       current_class = 0;
	       Delete(Classprefix);
	       Delete(clsname);
	       Classprefix = 0;
	       prev_symtab = 0;
	       (yyval.node) = 0;

	     }
#line 4831 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 1713 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.node) = new_node("apply");
                    Setattr((yyval.node),"pattern",Getattr((yyvsp[-3].p),"pattern"));
		    appendChild((yyval.node),(yyvsp[-1].p));
               }
#line 4841 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 1723 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = new_node("clear");
		 appendChild((yyval.node),(yyvsp[-1].p));
               }
#line 4850 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 1734 "parser.y" /* yacc.c:1646  */
    {
		   if (((yyvsp[-1].dtype).type != T_ERROR) && ((yyvsp[-1].dtype).type != T_SYMBOL)) {
		     SwigType *type = NewSwigType((yyvsp[-1].dtype).type);
		     (yyval.node) = new_node("constant");
		     Setattr((yyval.node),"name",(yyvsp[-3].id));
		     Setattr((yyval.node),"type",type);
		     Setattr((yyval.node),"value",(yyvsp[-1].dtype).val);
		     if ((yyvsp[-1].dtype).rawval) Setattr((yyval.node),"rawval", (yyvsp[-1].dtype).rawval);
		     Setattr((yyval.node),"storage","%constant");
		     SetFlag((yyval.node),"feature:immutable");
		     add_symbols((yyval.node));
		     Delete(type);
		   } else {
		     if ((yyvsp[-1].dtype).type == T_ERROR) {
		       Swig_warning(WARN_PARSE_UNSUPPORTED_VALUE,cparse_file,cparse_line,"Unsupported constant value (ignored)\n");
		     }
		     (yyval.node) = 0;
		   }

	       }
#line 4875 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 1755 "parser.y" /* yacc.c:1646  */
    {
		 if (((yyvsp[-1].dtype).type != T_ERROR) && ((yyvsp[-1].dtype).type != T_SYMBOL)) {
		   SwigType_push((yyvsp[-3].type),(yyvsp[-2].decl).type);
		   /* Sneaky callback function trick */
		   if (SwigType_isfunction((yyvsp[-3].type))) {
		     SwigType_add_pointer((yyvsp[-3].type));
		   }
		   (yyval.node) = new_node("constant");
		   Setattr((yyval.node),"name",(yyvsp[-2].decl).id);
		   Setattr((yyval.node),"type",(yyvsp[-3].type));
		   Setattr((yyval.node),"value",(yyvsp[-1].dtype).val);
		   if ((yyvsp[-1].dtype).rawval) Setattr((yyval.node),"rawval", (yyvsp[-1].dtype).rawval);
		   Setattr((yyval.node),"storage","%constant");
		   SetFlag((yyval.node),"feature:immutable");
		   add_symbols((yyval.node));
		 } else {
		     if ((yyvsp[-1].dtype).type == T_ERROR) {
		       Swig_warning(WARN_PARSE_UNSUPPORTED_VALUE,cparse_file,cparse_line,"Unsupported constant value\n");
		     }
		   (yyval.node) = 0;
		 }
               }
#line 4902 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 1777 "parser.y" /* yacc.c:1646  */
    {
		 Swig_warning(WARN_PARSE_BAD_VALUE,cparse_file,cparse_line,"Bad constant value (ignored).\n");
		 (yyval.node) = 0;
	       }
#line 4911 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 1788 "parser.y" /* yacc.c:1646  */
    {
		 char temp[64];
		 Replace((yyvsp[0].str),"$file",cparse_file, DOH_REPLACE_ANY);
		 sprintf(temp,"%d", cparse_line);
		 Replace((yyvsp[0].str),"$line",temp,DOH_REPLACE_ANY);
		 Printf(stderr,"%s\n", (yyvsp[0].str));
		 Delete((yyvsp[0].str));
                 (yyval.node) = 0;
	       }
#line 4925 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 1797 "parser.y" /* yacc.c:1646  */
    {
		 char temp[64];
		 String *s = (yyvsp[0].str);
		 Replace(s,"$file",cparse_file, DOH_REPLACE_ANY);
		 sprintf(temp,"%d", cparse_line);
		 Replace(s,"$line",temp,DOH_REPLACE_ANY);
		 Printf(stderr,"%s\n", s);
		 Delete(s);
                 (yyval.node) = 0;
               }
#line 4940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 1816 "parser.y" /* yacc.c:1646  */
    {
                    skip_balanced('{','}');
		    (yyval.node) = 0;
		    Swig_warning(WARN_DEPRECATED_EXCEPT,cparse_file, cparse_line, "%%except is deprecated.  Use %%exception instead.\n");
	       }
#line 4950 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 1822 "parser.y" /* yacc.c:1646  */
    {
                    skip_balanced('{','}');
		    (yyval.node) = 0;
		    Swig_warning(WARN_DEPRECATED_EXCEPT,cparse_file, cparse_line, "%%except is deprecated.  Use %%exception instead.\n");
               }
#line 4960 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 1828 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = 0;
		 Swig_warning(WARN_DEPRECATED_EXCEPT,cparse_file, cparse_line, "%%except is deprecated.  Use %%exception instead.\n");
               }
#line 4969 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 1833 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = 0;
		 Swig_warning(WARN_DEPRECATED_EXCEPT,cparse_file, cparse_line, "%%except is deprecated.  Use %%exception instead.\n");
	       }
#line 4978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 1840 "parser.y" /* yacc.c:1646  */
    {		 
                 (yyval.node) = NewHash();
                 Setattr((yyval.node),"value",(yyvsp[-3].str));
		 Setattr((yyval.node),"type",Getattr((yyvsp[-1].p),"type"));
               }
#line 4988 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 1847 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = NewHash();
                 Setattr((yyval.node),"value",(yyvsp[0].str));
              }
#line 4997 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 1851 "parser.y" /* yacc.c:1646  */
    {
                (yyval.node) = (yyvsp[0].node);
              }
#line 5005 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 1864 "parser.y" /* yacc.c:1646  */
    {
                   Hash *p = (yyvsp[-2].node);
		   (yyval.node) = new_node("fragment");
		   Setattr((yyval.node),"value",Getattr((yyvsp[-4].node),"value"));
		   Setattr((yyval.node),"type",Getattr((yyvsp[-4].node),"type"));
		   Setattr((yyval.node),"section",Getattr(p,"name"));
		   Setattr((yyval.node),"kwargs",nextSibling(p));
		   Setattr((yyval.node),"code",(yyvsp[0].str));
                 }
#line 5019 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 1873 "parser.y" /* yacc.c:1646  */
    {
		   Hash *p = (yyvsp[-2].node);
		   String *code;
                   skip_balanced('{','}');
		   (yyval.node) = new_node("fragment");
		   Setattr((yyval.node),"value",Getattr((yyvsp[-4].node),"value"));
		   Setattr((yyval.node),"type",Getattr((yyvsp[-4].node),"type"));
		   Setattr((yyval.node),"section",Getattr(p,"name"));
		   Setattr((yyval.node),"kwargs",nextSibling(p));
		   Delitem(scanner_ccode,0);
		   Delitem(scanner_ccode,DOH_END);
		   code = Copy(scanner_ccode);
		   Setattr((yyval.node),"code",code);
		   Delete(code);
                 }
#line 5039 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 1888 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.node) = new_node("fragment");
		   Setattr((yyval.node),"value",Getattr((yyvsp[-2].node),"value"));
		   Setattr((yyval.node),"type",Getattr((yyvsp[-2].node),"type"));
		   Setattr((yyval.node),"emitonly","1");
		 }
#line 5050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 1901 "parser.y" /* yacc.c:1646  */
    {
                     (yyvsp[-3].loc).filename = Copy(cparse_file);
		     (yyvsp[-3].loc).line = cparse_line;
		     scanner_set_location((yyvsp[-1].str),1);
                     if ((yyvsp[-2].node)) { 
		       String *maininput = Getattr((yyvsp[-2].node), "maininput");
		       if (maininput)
		         scanner_set_main_input_file(NewString(maininput));
		     }
               }
#line 5065 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 1910 "parser.y" /* yacc.c:1646  */
    {
                     String *mname = 0;
                     (yyval.node) = (yyvsp[-1].node);
		     scanner_set_location((yyvsp[-6].loc).filename,(yyvsp[-6].loc).line+1);
		     if (strcmp((yyvsp[-6].loc).type,"include") == 0) set_nodeType((yyval.node),"include");
		     if (strcmp((yyvsp[-6].loc).type,"import") == 0) {
		       mname = (yyvsp[-5].node) ? Getattr((yyvsp[-5].node),"module") : 0;
		       set_nodeType((yyval.node),"import");
		       if (import_mode) --import_mode;
		     }
		     
		     Setattr((yyval.node),"name",(yyvsp[-4].str));
		     /* Search for the module (if any) */
		     {
			 Node *n = firstChild((yyval.node));
			 while (n) {
			     if (Strcmp(nodeType(n),"module") == 0) {
			         if (mname) {
				   Setattr(n,"name", mname);
				   mname = 0;
				 }
				 Setattr((yyval.node),"module",Getattr(n,"name"));
				 break;
			     }
			     n = nextSibling(n);
			 }
			 if (mname) {
			   /* There is no module node in the import
			      node, ie, you imported a .h file
			      directly.  We are forced then to create
			      a new import node with a module node.
			   */			      
			   Node *nint = new_node("import");
			   Node *mnode = new_node("module");
			   Setattr(mnode,"name", mname);
                           Setattr(mnode,"options",(yyvsp[-5].node));
			   appendChild(nint,mnode);
			   Delete(mnode);
			   appendChild(nint,firstChild((yyval.node)));
			   (yyval.node) = nint;
			   Setattr((yyval.node),"module",mname);
			 }
		     }
		     Setattr((yyval.node),"options",(yyvsp[-5].node));
               }
#line 5115 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 1957 "parser.y" /* yacc.c:1646  */
    { (yyval.loc).type = "include"; }
#line 5121 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 1958 "parser.y" /* yacc.c:1646  */
    { (yyval.loc).type = "import"; ++import_mode;}
#line 5127 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 1965 "parser.y" /* yacc.c:1646  */
    {
                 String *cpps;
		 if (Namespaceprefix) {
		   Swig_error(cparse_file, cparse_start_line, "%%inline directive inside a namespace is disallowed.\n");
		   (yyval.node) = 0;
		 } else {
		   (yyval.node) = new_node("insert");
		   Setattr((yyval.node),"code",(yyvsp[0].str));
		   /* Need to run through the preprocessor */
		   Seek((yyvsp[0].str),0,SEEK_SET);
		   Setline((yyvsp[0].str),cparse_start_line);
		   Setfile((yyvsp[0].str),cparse_file);
		   cpps = Preprocessor_parse((yyvsp[0].str));
		   start_inline(Char(cpps), cparse_start_line);
		   Delete((yyvsp[0].str));
		   Delete(cpps);
		 }
		 
	       }
#line 5151 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 1984 "parser.y" /* yacc.c:1646  */
    {
                 String *cpps;
		 int start_line = cparse_line;
		 skip_balanced('{','}');
		 if (Namespaceprefix) {
		   Swig_error(cparse_file, cparse_start_line, "%%inline directive inside a namespace is disallowed.\n");
		   
		   (yyval.node) = 0;
		 } else {
		   String *code;
                   (yyval.node) = new_node("insert");
		   Delitem(scanner_ccode,0);
		   Delitem(scanner_ccode,DOH_END);
		   code = Copy(scanner_ccode);
		   Setattr((yyval.node),"code", code);
		   Delete(code);		   
		   cpps=Copy(scanner_ccode);
		   start_inline(Char(cpps), start_line);
		   Delete(cpps);
		 }
               }
#line 5177 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 2015 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = new_node("insert");
		 Setattr((yyval.node),"code",(yyvsp[0].str));
	       }
#line 5186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 2019 "parser.y" /* yacc.c:1646  */
    {
		 String *code = NewStringEmpty();
		 (yyval.node) = new_node("insert");
		 Setattr((yyval.node),"section",(yyvsp[-2].id));
		 Setattr((yyval.node),"code",code);
		 if (Swig_insert_file((yyvsp[0].str),code) < 0) {
		   Swig_error(cparse_file, cparse_line, "Couldn't find '%s'.\n", (yyvsp[0].str));
		   (yyval.node) = 0;
		 } 
               }
#line 5201 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 2029 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = new_node("insert");
		 Setattr((yyval.node),"section",(yyvsp[-2].id));
		 Setattr((yyval.node),"code",(yyvsp[0].str));
               }
#line 5211 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 2034 "parser.y" /* yacc.c:1646  */
    {
		 String *code;
                 skip_balanced('{','}');
		 (yyval.node) = new_node("insert");
		 Setattr((yyval.node),"section",(yyvsp[-2].id));
		 Delitem(scanner_ccode,0);
		 Delitem(scanner_ccode,DOH_END);
		 code = Copy(scanner_ccode);
		 Setattr((yyval.node),"code", code);
		 Delete(code);
	       }
#line 5227 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 2052 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = new_node("module");
		 if ((yyvsp[-1].node)) {
		   Setattr((yyval.node),"options",(yyvsp[-1].node));
		   if (Getattr((yyvsp[-1].node),"directors")) {
		     Wrapper_director_mode_set(1);
		     if (!cparse_cplusplus) {
		       Swig_error(cparse_file, cparse_line, "Directors are not supported for C code and require the -c++ option\n");
		     }
		   } 
		   if (Getattr((yyvsp[-1].node),"dirprot")) {
		     Wrapper_director_protected_mode_set(1);
		   } 
		   if (Getattr((yyvsp[-1].node),"allprotected")) {
		     Wrapper_all_protected_mode_set(1);
		   } 
		   if (Getattr((yyvsp[-1].node),"templatereduce")) {
		     template_reduce = 1;
		   }
		   if (Getattr((yyvsp[-1].node),"notemplatereduce")) {
		     template_reduce = 0;
		   }
		 }
		 if (!ModuleName) ModuleName = NewString((yyvsp[0].id));
		 if (!import_mode) {
		   /* first module included, we apply global
		      ModuleName, which can be modify by -module */
		   String *mname = Copy(ModuleName);
		   Setattr((yyval.node),"name",mname);
		   Delete(mname);
		 } else { 
		   /* import mode, we just pass the idstring */
		   Setattr((yyval.node),"name",(yyvsp[0].id));   
		 }		 
		 if (!module_node) module_node = (yyval.node);
	       }
#line 5268 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 2095 "parser.y" /* yacc.c:1646  */
    {
                 Swig_warning(WARN_DEPRECATED_NAME,cparse_file,cparse_line, "%%name is deprecated.  Use %%rename instead.\n");
		 Delete(yyrename);
                 yyrename = NewString((yyvsp[-1].id));
		 (yyval.node) = 0;
               }
#line 5279 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 2101 "parser.y" /* yacc.c:1646  */
    {
		 Swig_warning(WARN_DEPRECATED_NAME,cparse_file,cparse_line, "%%name is deprecated.  Use %%rename instead.\n");
		 (yyval.node) = 0;
		 Swig_error(cparse_file,cparse_line,"Missing argument to %%name directive.\n");
	       }
#line 5289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 2114 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = new_node("native");
		 Setattr((yyval.node),"name",(yyvsp[-4].id));
		 Setattr((yyval.node),"wrap:name",(yyvsp[-1].id));
	         add_symbols((yyval.node));
	       }
#line 5300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 2120 "parser.y" /* yacc.c:1646  */
    {
		 if (!SwigType_isfunction((yyvsp[-1].decl).type)) {
		   Swig_error(cparse_file,cparse_line,"%%native declaration '%s' is not a function.\n", (yyvsp[-1].decl).id);
		   (yyval.node) = 0;
		 } else {
		     Delete(SwigType_pop_function((yyvsp[-1].decl).type));
		     /* Need check for function here */
		     SwigType_push((yyvsp[-2].type),(yyvsp[-1].decl).type);
		     (yyval.node) = new_node("native");
	             Setattr((yyval.node),"name",(yyvsp[-5].id));
		     Setattr((yyval.node),"wrap:name",(yyvsp[-1].decl).id);
		     Setattr((yyval.node),"type",(yyvsp[-2].type));
		     Setattr((yyval.node),"parms",(yyvsp[-1].decl).parms);
		     Setattr((yyval.node),"decl",(yyvsp[-1].decl).type);
		 }
	         add_symbols((yyval.node));
	       }
#line 5322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 2146 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = new_node("pragma");
		 Setattr((yyval.node),"lang",(yyvsp[-3].id));
		 Setattr((yyval.node),"name",(yyvsp[-2].id));
		 Setattr((yyval.node),"value",(yyvsp[0].str));
	       }
#line 5333 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 2152 "parser.y" /* yacc.c:1646  */
    {
		(yyval.node) = new_node("pragma");
		Setattr((yyval.node),"lang",(yyvsp[-1].id));
		Setattr((yyval.node),"name",(yyvsp[0].id));
	      }
#line 5343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 2159 "parser.y" /* yacc.c:1646  */
    { (yyval.str) = (yyvsp[0].str); }
#line 5349 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 2160 "parser.y" /* yacc.c:1646  */
    { (yyval.str) = (yyvsp[0].str); }
#line 5355 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 2163 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (yyvsp[-1].id); }
#line 5361 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 2164 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (char *) "swig"; }
#line 5367 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 2171 "parser.y" /* yacc.c:1646  */
    {
                SwigType *t = (yyvsp[-2].decl).type;
		Hash *kws = NewHash();
		String *fixname;
		fixname = feature_identifier_fix((yyvsp[-2].decl).id);
		Setattr(kws,"name",(yyvsp[-1].id));
		if (!Len(t)) t = 0;
		/* Special declarator check */
		if (t) {
		  if (SwigType_isfunction(t)) {
		    SwigType *decl = SwigType_pop_function(t);
		    if (SwigType_ispointer(t)) {
		      String *nname = NewStringf("*%s",fixname);
		      if ((yyvsp[-3].intvalue)) {
			Swig_name_rename_add(Namespaceprefix, nname,decl,kws,(yyvsp[-2].decl).parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,nname,decl,kws);
		      }
		      Delete(nname);
		    } else {
		      if ((yyvsp[-3].intvalue)) {
			Swig_name_rename_add(Namespaceprefix,(fixname),decl,kws,(yyvsp[-2].decl).parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,(fixname),decl,kws);
		      }
		    }
		    Delete(decl);
		  } else if (SwigType_ispointer(t)) {
		    String *nname = NewStringf("*%s",fixname);
		    if ((yyvsp[-3].intvalue)) {
		      Swig_name_rename_add(Namespaceprefix,(nname),0,kws,(yyvsp[-2].decl).parms);
		    } else {
		      Swig_name_namewarn_add(Namespaceprefix,(nname),0,kws);
		    }
		    Delete(nname);
		  }
		} else {
		  if ((yyvsp[-3].intvalue)) {
		    Swig_name_rename_add(Namespaceprefix,(fixname),0,kws,(yyvsp[-2].decl).parms);
		  } else {
		    Swig_name_namewarn_add(Namespaceprefix,(fixname),0,kws);
		  }
		}
                (yyval.node) = 0;
		scanner_clear_rename();
              }
#line 5418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 2217 "parser.y" /* yacc.c:1646  */
    {
		String *fixname;
		Hash *kws = (yyvsp[-4].node);
		SwigType *t = (yyvsp[-2].decl).type;
		fixname = feature_identifier_fix((yyvsp[-2].decl).id);
		if (!Len(t)) t = 0;
		/* Special declarator check */
		if (t) {
		  if ((yyvsp[-1].dtype).qualifier) SwigType_push(t,(yyvsp[-1].dtype).qualifier);
		  if (SwigType_isfunction(t)) {
		    SwigType *decl = SwigType_pop_function(t);
		    if (SwigType_ispointer(t)) {
		      String *nname = NewStringf("*%s",fixname);
		      if ((yyvsp[-6].intvalue)) {
			Swig_name_rename_add(Namespaceprefix, nname,decl,kws,(yyvsp[-2].decl).parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,nname,decl,kws);
		      }
		      Delete(nname);
		    } else {
		      if ((yyvsp[-6].intvalue)) {
			Swig_name_rename_add(Namespaceprefix,(fixname),decl,kws,(yyvsp[-2].decl).parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,(fixname),decl,kws);
		      }
		    }
		    Delete(decl);
		  } else if (SwigType_ispointer(t)) {
		    String *nname = NewStringf("*%s",fixname);
		    if ((yyvsp[-6].intvalue)) {
		      Swig_name_rename_add(Namespaceprefix,(nname),0,kws,(yyvsp[-2].decl).parms);
		    } else {
		      Swig_name_namewarn_add(Namespaceprefix,(nname),0,kws);
		    }
		    Delete(nname);
		  }
		} else {
		  if ((yyvsp[-6].intvalue)) {
		    Swig_name_rename_add(Namespaceprefix,(fixname),0,kws,(yyvsp[-2].decl).parms);
		  } else {
		    Swig_name_namewarn_add(Namespaceprefix,(fixname),0,kws);
		  }
		}
                (yyval.node) = 0;
		scanner_clear_rename();
              }
#line 5469 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 2263 "parser.y" /* yacc.c:1646  */
    {
		if ((yyvsp[-5].intvalue)) {
		  Swig_name_rename_add(Namespaceprefix,(yyvsp[-1].str),0,(yyvsp[-3].node),0);
		} else {
		  Swig_name_namewarn_add(Namespaceprefix,(yyvsp[-1].str),0,(yyvsp[-3].node));
		}
		(yyval.node) = 0;
		scanner_clear_rename();
              }
#line 5483 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 2274 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.intvalue) = 1;
                }
#line 5491 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 2277 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.intvalue) = 0;
                }
#line 5499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 2304 "parser.y" /* yacc.c:1646  */
    {
                    String *val = (yyvsp[0].str) ? NewString((yyvsp[0].str)) : NewString("1");
                    new_feature((yyvsp[-4].id), val, 0, (yyvsp[-2].decl).id, (yyvsp[-2].decl).type, (yyvsp[-2].decl).parms, (yyvsp[-1].dtype).qualifier);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5510 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 2310 "parser.y" /* yacc.c:1646  */
    {
                    String *val = Len((yyvsp[-4].str)) ? (yyvsp[-4].str) : 0;
                    new_feature((yyvsp[-6].id), val, 0, (yyvsp[-2].decl).id, (yyvsp[-2].decl).type, (yyvsp[-2].decl).parms, (yyvsp[-1].dtype).qualifier);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5521 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 2316 "parser.y" /* yacc.c:1646  */
    {
                    String *val = (yyvsp[0].str) ? NewString((yyvsp[0].str)) : NewString("1");
                    new_feature((yyvsp[-5].id), val, (yyvsp[-4].node), (yyvsp[-2].decl).id, (yyvsp[-2].decl).type, (yyvsp[-2].decl).parms, (yyvsp[-1].dtype).qualifier);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 2322 "parser.y" /* yacc.c:1646  */
    {
                    String *val = Len((yyvsp[-5].str)) ? (yyvsp[-5].str) : 0;
                    new_feature((yyvsp[-7].id), val, (yyvsp[-4].node), (yyvsp[-2].decl).id, (yyvsp[-2].decl).type, (yyvsp[-2].decl).parms, (yyvsp[-1].dtype).qualifier);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5543 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 2330 "parser.y" /* yacc.c:1646  */
    {
                    String *val = (yyvsp[0].str) ? NewString((yyvsp[0].str)) : NewString("1");
                    new_feature((yyvsp[-2].id), val, 0, 0, 0, 0, 0);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 2336 "parser.y" /* yacc.c:1646  */
    {
                    String *val = Len((yyvsp[-2].str)) ? (yyvsp[-2].str) : 0;
                    new_feature((yyvsp[-4].id), val, 0, 0, 0, 0, 0);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 2342 "parser.y" /* yacc.c:1646  */
    {
                    String *val = (yyvsp[0].str) ? NewString((yyvsp[0].str)) : NewString("1");
                    new_feature((yyvsp[-3].id), val, (yyvsp[-2].node), 0, 0, 0, 0);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5576 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 2348 "parser.y" /* yacc.c:1646  */
    {
                    String *val = Len((yyvsp[-3].str)) ? (yyvsp[-3].str) : 0;
                    new_feature((yyvsp[-5].id), val, (yyvsp[-2].node), 0, 0, 0, 0);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5587 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 2356 "parser.y" /* yacc.c:1646  */
    { (yyval.str) = (yyvsp[0].str); }
#line 5593 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 2357 "parser.y" /* yacc.c:1646  */
    { (yyval.str) = 0; }
#line 5599 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 2358 "parser.y" /* yacc.c:1646  */
    { (yyval.str) = (yyvsp[-2].pl); }
#line 5605 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 2361 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.node) = NewHash();
		  Setattr((yyval.node),"name",(yyvsp[-2].id));
		  Setattr((yyval.node),"value",(yyvsp[0].str));
                }
#line 5615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 2366 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.node) = NewHash();
		  Setattr((yyval.node),"name",(yyvsp[-3].id));
		  Setattr((yyval.node),"value",(yyvsp[-1].str));
                  set_nextSibling((yyval.node),(yyvsp[0].node));
                }
#line 5626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 2376 "parser.y" /* yacc.c:1646  */
    {
                 Parm *val;
		 String *name;
		 SwigType *t;
		 if (Namespaceprefix) name = NewStringf("%s::%s", Namespaceprefix, (yyvsp[-2].decl).id);
		 else name = NewString((yyvsp[-2].decl).id);
		 val = (yyvsp[-4].pl);
		 if ((yyvsp[-2].decl).parms) {
		   Setmeta(val,"parms",(yyvsp[-2].decl).parms);
		 }
		 t = (yyvsp[-2].decl).type;
		 if (!Len(t)) t = 0;
		 if (t) {
		   if ((yyvsp[-1].dtype).qualifier) SwigType_push(t,(yyvsp[-1].dtype).qualifier);
		   if (SwigType_isfunction(t)) {
		     SwigType *decl = SwigType_pop_function(t);
		     if (SwigType_ispointer(t)) {
		       String *nname = NewStringf("*%s",name);
		       Swig_feature_set(Swig_cparse_features(), nname, decl, "feature:varargs", val, 0);
		       Delete(nname);
		     } else {
		       Swig_feature_set(Swig_cparse_features(), name, decl, "feature:varargs", val, 0);
		     }
		     Delete(decl);
		   } else if (SwigType_ispointer(t)) {
		     String *nname = NewStringf("*%s",name);
		     Swig_feature_set(Swig_cparse_features(),nname,0,"feature:varargs",val, 0);
		     Delete(nname);
		   }
		 } else {
		   Swig_feature_set(Swig_cparse_features(),name,0,"feature:varargs",val, 0);
		 }
		 Delete(name);
		 (yyval.node) = 0;
              }
#line 5666 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 2412 "parser.y" /* yacc.c:1646  */
    { (yyval.pl) = (yyvsp[0].pl); }
#line 5672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 2413 "parser.y" /* yacc.c:1646  */
    { 
		  int i;
		  int n;
		  Parm *p;
		  n = atoi(Char((yyvsp[-2].dtype).val));
		  if (n <= 0) {
		    Swig_error(cparse_file, cparse_line,"Argument count in %%varargs must be positive.\n");
		    (yyval.pl) = 0;
		  } else {
		    String *name = Getattr((yyvsp[0].p), "name");
		    (yyval.pl) = Copy((yyvsp[0].p));
		    if (name)
		      Setattr((yyval.pl), "name", NewStringf("%s%d", name, n));
		    for (i = 1; i < n; i++) {
		      p = Copy((yyvsp[0].p));
		      name = Getattr(p, "name");
		      if (name)
		        Setattr(p, "name", NewStringf("%s%d", name, n-i));
		      set_nextSibling(p,(yyval.pl));
		      Delete((yyval.pl));
		      (yyval.pl) = p;
		    }
		  }
                }
#line 5701 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 2448 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.node) = 0;
		   if ((yyvsp[-3].tmap).method) {
		     String *code = 0;
		     (yyval.node) = new_node("typemap");
		     Setattr((yyval.node),"method",(yyvsp[-3].tmap).method);
		     if ((yyvsp[-3].tmap).kwargs) {
		       ParmList *kw = (yyvsp[-3].tmap).kwargs;
                       code = remove_block(kw, (yyvsp[0].str));
		       Setattr((yyval.node),"kwargs", (yyvsp[-3].tmap).kwargs);
		     }
		     code = code ? code : NewString((yyvsp[0].str));
		     Setattr((yyval.node),"code", code);
		     Delete(code);
		     appendChild((yyval.node),(yyvsp[-1].p));
		   }
	       }
#line 5723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 2465 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = 0;
		 if ((yyvsp[-3].tmap).method) {
		   (yyval.node) = new_node("typemap");
		   Setattr((yyval.node),"method",(yyvsp[-3].tmap).method);
		   appendChild((yyval.node),(yyvsp[-1].p));
		 }
	       }
#line 5736 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 2473 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.node) = 0;
		   if ((yyvsp[-5].tmap).method) {
		     (yyval.node) = new_node("typemapcopy");
		     Setattr((yyval.node),"method",(yyvsp[-5].tmap).method);
		     Setattr((yyval.node),"pattern", Getattr((yyvsp[-1].p),"pattern"));
		     appendChild((yyval.node),(yyvsp[-3].p));
		   }
	       }
#line 5750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 2486 "parser.y" /* yacc.c:1646  */
    {
		 Hash *p;
		 String *name;
		 p = nextSibling((yyvsp[0].node));
		 if (p && (!Getattr(p,"value"))) {
 		   /* this is the deprecated two argument typemap form */
 		   Swig_warning(WARN_DEPRECATED_TYPEMAP_LANG,cparse_file, cparse_line,
				"Specifying the language name in %%typemap is deprecated - use #ifdef SWIG<LANG> instead.\n");
		   /* two argument typemap form */
		   name = Getattr((yyvsp[0].node),"name");
		   if (!name || (Strcmp(name,typemap_lang))) {
		     (yyval.tmap).method = 0;
		     (yyval.tmap).kwargs = 0;
		   } else {
		     (yyval.tmap).method = Getattr(p,"name");
		     (yyval.tmap).kwargs = nextSibling(p);
		   }
		 } else {
		   /* one-argument typemap-form */
		   (yyval.tmap).method = Getattr((yyvsp[0].node),"name");
		   (yyval.tmap).kwargs = p;
		 }
                }
#line 5778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 2511 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.p) = (yyvsp[-1].p);
		 set_nextSibling((yyval.p),(yyvsp[0].p));
		}
#line 5787 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 2517 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.p) = (yyvsp[-1].p);
		 set_nextSibling((yyval.p),(yyvsp[0].p));
                }
#line 5796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 2521 "parser.y" /* yacc.c:1646  */
    { (yyval.p) = 0;}
#line 5802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 2524 "parser.y" /* yacc.c:1646  */
    {
                  Parm *parm;
		  SwigType_push((yyvsp[-1].type),(yyvsp[0].decl).type);
		  (yyval.p) = new_node("typemapitem");
		  parm = NewParmWithoutFileLineInfo((yyvsp[-1].type),(yyvsp[0].decl).id);
		  Setattr((yyval.p),"pattern",parm);
		  Setattr((yyval.p),"parms", (yyvsp[0].decl).parms);
		  Delete(parm);
		  /*		  $$ = NewParmWithoutFileLineInfo($1,$2.id);
				  Setattr($$,"parms",$2.parms); */
                }
#line 5818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 2535 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.p) = new_node("typemapitem");
		  Setattr((yyval.p),"pattern",(yyvsp[-1].pl));
		  /*		  Setattr($$,"multitype",$2); */
               }
#line 5828 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 2540 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.p) = new_node("typemapitem");
		 Setattr((yyval.p),"pattern", (yyvsp[-4].pl));
		 /*                 Setattr($$,"multitype",$2); */
		 Setattr((yyval.p),"parms",(yyvsp[-1].pl));
               }
#line 5839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 2553 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.node) = new_node("types");
		   Setattr((yyval.node),"parms",(yyvsp[-2].pl));
                   if ((yyvsp[0].str))
		     Setattr((yyval.node),"convcode",NewString((yyvsp[0].str)));
               }
#line 5850 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 2565 "parser.y" /* yacc.c:1646  */
    {
                  Parm *p, *tp;
		  Node *n;
		  Node *outer_class = currentOuterClass;
		  Symtab *tscope = 0;
		  int     specialized = 0;
		  int     variadic = 0;

		  (yyval.node) = 0;

		  tscope = Swig_symbol_current();          /* Get the current scope */

		  /* If the class name is qualified, we need to create or lookup namespace entries */
		  if (!inclass) {
		    (yyvsp[-4].str) = resolve_create_node_scope((yyvsp[-4].str));
		  }
		  if (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0) {
		    outer_class	= nscope_inner;
		  }

		  /*
		    We use the new namespace entry 'nscope' only to
		    emit the template node. The template parameters are
		    resolved in the current 'tscope'.

		    This is closer to the C++ (typedef) behavior.
		  */
		  n = Swig_cparse_template_locate((yyvsp[-4].str),(yyvsp[-2].p),tscope);

		  /* Patch the argument types to respect namespaces */
		  p = (yyvsp[-2].p);
		  while (p) {
		    SwigType *value = Getattr(p,"value");
		    if (!value) {
		      SwigType *ty = Getattr(p,"type");
		      if (ty) {
			SwigType *rty = 0;
			int reduce = template_reduce;
			if (reduce || !SwigType_ispointer(ty)) {
			  rty = Swig_symbol_typedef_reduce(ty,tscope);
			  if (!reduce) reduce = SwigType_ispointer(rty);
			}
			ty = reduce ? Swig_symbol_type_qualify(rty,tscope) : Swig_symbol_type_qualify(ty,tscope);
			Setattr(p,"type",ty);
			Delete(ty);
			Delete(rty);
		      }
		    } else {
		      value = Swig_symbol_type_qualify(value,tscope);
		      Setattr(p,"value",value);
		      Delete(value);
		    }

		    p = nextSibling(p);
		  }

		  /* Look for the template */
		  {
                    Node *nn = n;
                    Node *linklistend = 0;
                    Node *linkliststart = 0;
                    while (nn) {
                      Node *templnode = 0;
                      if (Strcmp(nodeType(nn),"template") == 0) {
                        int nnisclass = (Strcmp(Getattr(nn,"templatetype"),"class") == 0); /* if not a templated class it is a templated function */
                        Parm *tparms = Getattr(nn,"templateparms");
                        if (!tparms) {
                          specialized = 1;
                        } else if (Getattr(tparms,"variadic") && strncmp(Char(Getattr(tparms,"variadic")), "1", 1)==0) {
                          variadic = 1;
                        }
                        if (nnisclass && !variadic && !specialized && (ParmList_len((yyvsp[-2].p)) > ParmList_len(tparms))) {
                          Swig_error(cparse_file, cparse_line, "Too many template parameters. Maximum of %d.\n", ParmList_len(tparms));
                        } else if (nnisclass && !specialized && ((ParmList_len((yyvsp[-2].p)) < (ParmList_numrequired(tparms) - (variadic?1:0))))) { /* Variadic parameter is optional */
                          Swig_error(cparse_file, cparse_line, "Not enough template parameters specified. %d required.\n", (ParmList_numrequired(tparms)-(variadic?1:0)) );
                        } else if (!nnisclass && ((ParmList_len((yyvsp[-2].p)) != ParmList_len(tparms)))) {
                          /* must be an overloaded templated method - ignore it as it is overloaded with a different number of template parameters */
                          nn = Getattr(nn,"sym:nextSibling"); /* repeat for overloaded templated functions */
                          continue;
                        } else {
			  String *tname = Copy((yyvsp[-4].str));
                          int def_supplied = 0;
                          /* Expand the template */
			  Node *templ = Swig_symbol_clookup((yyvsp[-4].str),0);
			  Parm *targs = templ ? Getattr(templ,"templateparms") : 0;

                          ParmList *temparms;
                          if (specialized) temparms = CopyParmList((yyvsp[-2].p));
                          else temparms = CopyParmList(tparms);

                          /* Create typedef's and arguments */
                          p = (yyvsp[-2].p);
                          tp = temparms;
                          if (!p && ParmList_len(p) != ParmList_len(temparms)) {
                            /* we have no template parameters supplied in %template for a template that has default args*/
                            p = tp;
                            def_supplied = 1;
                          }

                          while (p) {
                            String *value = Getattr(p,"value");
                            if (def_supplied) {
                              Setattr(p,"default","1");
                            }
                            if (value) {
                              Setattr(tp,"value",value);
                            } else {
                              SwigType *ty = Getattr(p,"type");
                              if (ty) {
                                Setattr(tp,"type",ty);
                              }
                              Delattr(tp,"value");
                            }
			    /* fix default arg values */
			    if (targs) {
			      Parm *pi = temparms;
			      Parm *ti = targs;
			      String *tv = Getattr(tp,"value");
			      if (!tv) tv = Getattr(tp,"type");
			      while(pi != tp && ti && pi) {
				String *name = Getattr(ti,"name");
				String *value = Getattr(pi,"value");
				if (!value) value = Getattr(pi,"type");
				Replaceid(tv, name, value);
				pi = nextSibling(pi);
				ti = nextSibling(ti);
			      }
			    }
                            p = nextSibling(p);
                            tp = nextSibling(tp);
                            if (!p && tp) {
                              p = tp;
                              def_supplied = 1;
                            } else if (p && !tp) { /* Variadic template - tp < p */
			      SWIG_WARN_NODE_BEGIN(nn);
                              Swig_warning(WARN_CPP11_VARIADIC_TEMPLATE,cparse_file, cparse_line,"Only the first variadic template argument is currently supported.\n");
			      SWIG_WARN_NODE_END(nn);
                              break;
                            }
                          }

                          templnode = copy_node(nn);
			  update_nested_classes(templnode); /* update classes nested within template */
                          /* We need to set the node name based on name used to instantiate */
                          Setattr(templnode,"name",tname);
			  Delete(tname);
                          if (!specialized) {
                            Delattr(templnode,"sym:typename");
                          } else {
                            Setattr(templnode,"sym:typename","1");
                          }
			  /* for now, nested %template is allowed only in the same scope as the template declaration */
                          if ((yyvsp[-6].id) && !(nnisclass && ((outer_class && (outer_class != Getattr(nn, "nested:outer")))
			    ||(extendmode && current_class && (current_class != Getattr(nn, "nested:outer")))))) {
			    /*
			       Comment this out for 1.3.28. We need to
			       re-enable it later but first we need to
			       move %ignore from using %rename to use
			       %feature(ignore).

			       String *symname = Swig_name_make(templnode,0,$3,0,0);
			    */
			    String *symname = NewString((yyvsp[-6].id));
                            Swig_cparse_template_expand(templnode,symname,temparms,tscope);
                            Setattr(templnode,"sym:name",symname);
                          } else {
                            static int cnt = 0;
                            String *nname = NewStringf("__dummy_%d__", cnt++);
                            Swig_cparse_template_expand(templnode,nname,temparms,tscope);
                            Setattr(templnode,"sym:name",nname);
			    Delete(nname);
                            Setattr(templnode,"feature:onlychildren", "typemap,typemapitem,typemapcopy,typedef,types,fragment");
			    if ((yyvsp[-6].id)) {
			      Swig_warning(WARN_PARSE_NESTED_TEMPLATE, cparse_file, cparse_line, "Named nested template instantiations not supported. Processing as if no name was given to %%template().\n");
			    }
                          }
                          Delattr(templnode,"templatetype");
                          Setattr(templnode,"template",nn);
                          Setfile(templnode,cparse_file);
                          Setline(templnode,cparse_line);
                          Delete(temparms);
			  if (outer_class && nnisclass) {
			    SetFlag(templnode, "nested");
			    Setattr(templnode, "nested:outer", outer_class);
			  }
                          add_symbols_copy(templnode);

                          if (Strcmp(nodeType(templnode),"class") == 0) {

                            /* Identify pure abstract methods */
                            Setattr(templnode,"abstracts", pure_abstracts(firstChild(templnode)));

                            /* Set up inheritance in symbol table */
                            {
                              Symtab  *csyms;
                              List *baselist = Getattr(templnode,"baselist");
                              csyms = Swig_symbol_current();
                              Swig_symbol_setscope(Getattr(templnode,"symtab"));
                              if (baselist) {
                                List *bases = Swig_make_inherit_list(Getattr(templnode,"name"),baselist, Namespaceprefix);
                                if (bases) {
                                  Iterator s;
                                  for (s = First(bases); s.item; s = Next(s)) {
                                    Symtab *st = Getattr(s.item,"symtab");
                                    if (st) {
				      Setfile(st,Getfile(s.item));
				      Setline(st,Getline(s.item));
                                      Swig_symbol_inherit(st);
                                    }
                                  }
				  Delete(bases);
                                }
                              }
                              Swig_symbol_setscope(csyms);
                            }

                            /* Merge in %extend methods for this class.
			       This only merges methods within %extend for a template specialized class such as
			         template<typename T> class K {}; %extend K<int> { ... }
			       The copy_node() call above has already added in the generic %extend methods such as
			         template<typename T> class K {}; %extend K { ... } */

			    /* !!! This may be broken.  We may have to add the
			       %extend methods at the beginning of the class */
                            {
                              String *stmp = 0;
                              String *clsname;
                              Node *am;
                              if (Namespaceprefix) {
                                clsname = stmp = NewStringf("%s::%s", Namespaceprefix, Getattr(templnode,"name"));
                              } else {
                                clsname = Getattr(templnode,"name");
                              }
                              am = Getattr(Swig_extend_hash(),clsname);
                              if (am) {
                                Symtab *st = Swig_symbol_current();
                                Swig_symbol_setscope(Getattr(templnode,"symtab"));
                                /*			    Printf(stdout,"%s: %s %p %p\n", Getattr(templnode,"name"), clsname, Swig_symbol_current(), Getattr(templnode,"symtab")); */
                                Swig_extend_merge(templnode,am);
                                Swig_symbol_setscope(st);
				Swig_extend_append_previous(templnode,am);
                                Delattr(Swig_extend_hash(),clsname);
                              }
			      if (stmp) Delete(stmp);
                            }

                            /* Add to classes hash */
			    if (!classes)
			      classes = NewHash();

			    if (Namespaceprefix) {
			      String *temp = NewStringf("%s::%s", Namespaceprefix, Getattr(templnode,"name"));
			      Setattr(classes,temp,templnode);
			      Delete(temp);
			    } else {
			      String *qs = Swig_symbol_qualifiedscopename(templnode);
			      Setattr(classes, qs,templnode);
			      Delete(qs);
			    }
                          }
                        }

                        /* all the overloaded templated functions are added into a linked list */
                        if (!linkliststart)
                          linkliststart = templnode;
                        if (nscope_inner) {
                          /* non-global namespace */
                          if (templnode) {
                            appendChild(nscope_inner,templnode);
			    Delete(templnode);
                            if (nscope) (yyval.node) = nscope;
                          }
                        } else {
                          /* global namespace */
                          if (!linklistend) {
                            (yyval.node) = templnode;
                          } else {
                            set_nextSibling(linklistend,templnode);
			    Delete(templnode);
                          }
                          linklistend = templnode;
                        }
                      }
                      nn = Getattr(nn,"sym:nextSibling"); /* repeat for overloaded templated functions. If a templated class there will never be a sibling. */
                    }
                    update_defaultargs(linkliststart);
		  }
	          Swig_symbol_setscope(tscope);
		  Delete(Namespaceprefix);
		  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
                }
#line 6146 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 2863 "parser.y" /* yacc.c:1646  */
    {
		  Swig_warning(0,cparse_file, cparse_line,"%s\n", (yyvsp[0].str));
		  (yyval.node) = 0;
               }
#line 6155 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 2873 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.node) = (yyvsp[0].node); 
                    if ((yyval.node)) {
   		      add_symbols((yyval.node));
                      default_arguments((yyval.node));
   	            }
                }
#line 6167 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 2880 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 6173 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 2881 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 6179 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 2885 "parser.y" /* yacc.c:1646  */
    {
		  if (Strcmp((yyvsp[-1].str),"C") == 0) {
		    cparse_externc = 1;
		  }
		}
#line 6189 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 2889 "parser.y" /* yacc.c:1646  */
    {
		  cparse_externc = 0;
		  if (Strcmp((yyvsp[-4].str),"C") == 0) {
		    Node *n = firstChild((yyvsp[-1].node));
		    (yyval.node) = new_node("extern");
		    Setattr((yyval.node),"name",(yyvsp[-4].str));
		    appendChild((yyval.node),n);
		    while (n) {
		      SwigType *decl = Getattr(n,"decl");
		      if (SwigType_isfunction(decl) && !Equal(Getattr(n, "storage"), "typedef")) {
			Setattr(n,"storage","externc");
		      }
		      n = nextSibling(n);
		    }
		  } else {
		     Swig_warning(WARN_PARSE_UNDEFINED_EXTERN,cparse_file, cparse_line,"Unrecognized extern type \"%s\".\n", (yyvsp[-4].str));
		    (yyval.node) = new_node("extern");
		    Setattr((yyval.node),"name",(yyvsp[-4].str));
		    appendChild((yyval.node),firstChild((yyvsp[-1].node)));
		  }
                }
#line 6215 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 2910 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.node) = (yyvsp[0].node);
		  SWIG_WARN_NODE_BEGIN((yyval.node));
		  Swig_warning(WARN_CPP11_LAMBDA, cparse_file, cparse_line, "Lambda expressions and closures are not fully supported yet.\n");
		  SWIG_WARN_NODE_END((yyval.node));
		}
#line 6226 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 2916 "parser.y" /* yacc.c:1646  */
    {
		  /* Convert using statement to a typedef statement */
		  (yyval.node) = new_node("cdecl");
		  Setattr((yyval.node),"type",(yyvsp[-2].type));
		  Setattr((yyval.node),"storage","typedef");
		  Setattr((yyval.node),"name",(yyvsp[-4].str));
		  Setattr((yyval.node),"decl",(yyvsp[-1].decl).type);
		  SetFlag((yyval.node),"typealias");
		  add_symbols((yyval.node));
		}
#line 6241 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 2926 "parser.y" /* yacc.c:1646  */
    {
		  /* Convert alias template to a "template" typedef statement */
		  (yyval.node) = new_node("template");
		  Setattr((yyval.node),"type",(yyvsp[-2].type));
		  Setattr((yyval.node),"storage","typedef");
		  Setattr((yyval.node),"name",(yyvsp[-4].str));
		  Setattr((yyval.node),"decl",(yyvsp[-1].decl).type);
		  Setattr((yyval.node),"templateparms",(yyvsp[-7].tparms));
		  Setattr((yyval.node),"templatetype","cdecl");
		  SetFlag((yyval.node),"aliastemplate");
		  add_symbols((yyval.node));
		}
#line 6258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 2944 "parser.y" /* yacc.c:1646  */
    {
              (yyval.node) = new_node("cdecl");
	      if ((yyvsp[-1].dtype).qualifier) SwigType_push((yyvsp[-2].decl).type,(yyvsp[-1].dtype).qualifier);
	      Setattr((yyval.node),"type",(yyvsp[-3].type));
	      Setattr((yyval.node),"storage",(yyvsp[-4].id));
	      Setattr((yyval.node),"name",(yyvsp[-2].decl).id);
	      Setattr((yyval.node),"decl",(yyvsp[-2].decl).type);
	      Setattr((yyval.node),"parms",(yyvsp[-2].decl).parms);
	      Setattr((yyval.node),"value",(yyvsp[-1].dtype).val);
	      Setattr((yyval.node),"throws",(yyvsp[-1].dtype).throws);
	      Setattr((yyval.node),"throw",(yyvsp[-1].dtype).throwf);
	      Setattr((yyval.node),"noexcept",(yyvsp[-1].dtype).nexcept);
	      if ((yyvsp[-1].dtype).val && (yyvsp[-1].dtype).type) {
		/* store initializer type as it might be different to the declared type */
		SwigType *valuetype = NewSwigType((yyvsp[-1].dtype).type);
		if (Len(valuetype) > 0)
		  Setattr((yyval.node),"valuetype",valuetype);
		else
		  Delete(valuetype);
	      }
	      if (!(yyvsp[0].node)) {
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr((yyval.node),"code",code);
		  Delete(code);
		}
	      } else {
		Node *n = (yyvsp[0].node);
		/* Inherit attributes */
		while (n) {
		  String *type = Copy((yyvsp[-3].type));
		  Setattr(n,"type",type);
		  Setattr(n,"storage",(yyvsp[-4].id));
		  n = nextSibling(n);
		  Delete(type);
		}
	      }
	      if ((yyvsp[-1].dtype).bitfield) {
		Setattr((yyval.node),"bitfield", (yyvsp[-1].dtype).bitfield);
	      }

	      /* Look for "::" declarations (ignored) */
	      if (Strstr((yyvsp[-2].decl).id,"::")) {
                /* This is a special case. If the scope name of the declaration exactly
                   matches that of the declaration, then we will allow it. Otherwise, delete. */
                String *p = Swig_scopename_prefix((yyvsp[-2].decl).id);
		if (p) {
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last((yyvsp[-2].decl).id);
		    Setattr((yyval.node),"name",lstr);
		    Delete(lstr);
		    set_nextSibling((yyval.node),(yyvsp[0].node));
		  } else {
		    Delete((yyval.node));
		    (yyval.node) = (yyvsp[0].node);
		  }
		  Delete(p);
		} else {
		  Delete((yyval.node));
		  (yyval.node) = (yyvsp[0].node);
		}
	      } else {
		set_nextSibling((yyval.node),(yyvsp[0].node));
	      }
           }
#line 6329 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 3012 "parser.y" /* yacc.c:1646  */
    {
              (yyval.node) = new_node("cdecl");
	      if ((yyvsp[-1].dtype).qualifier) SwigType_push((yyvsp[-4].decl).type,(yyvsp[-1].dtype).qualifier);
	      Setattr((yyval.node),"type",(yyvsp[-2].node));
	      Setattr((yyval.node),"storage",(yyvsp[-6].id));
	      Setattr((yyval.node),"name",(yyvsp[-4].decl).id);
	      Setattr((yyval.node),"decl",(yyvsp[-4].decl).type);
	      Setattr((yyval.node),"parms",(yyvsp[-4].decl).parms);
	      Setattr((yyval.node),"value",(yyvsp[-1].dtype).val);
	      Setattr((yyval.node),"throws",(yyvsp[-1].dtype).throws);
	      Setattr((yyval.node),"throw",(yyvsp[-1].dtype).throwf);
	      Setattr((yyval.node),"noexcept",(yyvsp[-1].dtype).nexcept);
	      if (!(yyvsp[0].node)) {
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr((yyval.node),"code",code);
		  Delete(code);
		}
	      } else {
		Node *n = (yyvsp[0].node);
		while (n) {
		  String *type = Copy((yyvsp[-2].node));
		  Setattr(n,"type",type);
		  Setattr(n,"storage",(yyvsp[-6].id));
		  n = nextSibling(n);
		  Delete(type);
		}
	      }
	      if ((yyvsp[-1].dtype).bitfield) {
		Setattr((yyval.node),"bitfield", (yyvsp[-1].dtype).bitfield);
	      }

	      if (Strstr((yyvsp[-4].decl).id,"::")) {
                String *p = Swig_scopename_prefix((yyvsp[-4].decl).id);
		if (p) {
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last((yyvsp[-4].decl).id);
		    Setattr((yyval.node),"name",lstr);
		    Delete(lstr);
		    set_nextSibling((yyval.node),(yyvsp[0].node));
		  } else {
		    Delete((yyval.node));
		    (yyval.node) = (yyvsp[0].node);
		  }
		  Delete(p);
		} else {
		  Delete((yyval.node));
		  (yyval.node) = (yyvsp[0].node);
		}
	      } else {
		set_nextSibling((yyval.node),(yyvsp[0].node));
	      }
           }
#line 6388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 3070 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.node) = 0;
                   Clear(scanner_ccode); 
               }
#line 6397 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 3074 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = new_node("cdecl");
		 if ((yyvsp[-1].dtype).qualifier) SwigType_push((yyvsp[-2].decl).type,(yyvsp[-1].dtype).qualifier);
		 Setattr((yyval.node),"name",(yyvsp[-2].decl).id);
		 Setattr((yyval.node),"decl",(yyvsp[-2].decl).type);
		 Setattr((yyval.node),"parms",(yyvsp[-2].decl).parms);
		 Setattr((yyval.node),"value",(yyvsp[-1].dtype).val);
		 Setattr((yyval.node),"throws",(yyvsp[-1].dtype).throws);
		 Setattr((yyval.node),"throw",(yyvsp[-1].dtype).throwf);
		 Setattr((yyval.node),"noexcept",(yyvsp[-1].dtype).nexcept);
		 if ((yyvsp[-1].dtype).bitfield) {
		   Setattr((yyval.node),"bitfield", (yyvsp[-1].dtype).bitfield);
		 }
		 if (!(yyvsp[0].node)) {
		   if (Len(scanner_ccode)) {
		     String *code = Copy(scanner_ccode);
		     Setattr((yyval.node),"code",code);
		     Delete(code);
		   }
		 } else {
		   set_nextSibling((yyval.node),(yyvsp[0].node));
		 }
	       }
#line 6425 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 3097 "parser.y" /* yacc.c:1646  */
    { 
                   skip_balanced('{','}');
                   (yyval.node) = 0;
               }
#line 6434 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 3101 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.node) = 0;
		   if (yychar == RPAREN) {
		       Swig_error(cparse_file, cparse_line, "Unexpected ')'.\n");
		   } else {
		       Swig_error(cparse_file, cparse_line, "Syntax error - possibly a missing semicolon.\n");
		   }
		   exit(1);
               }
#line 6448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 3112 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.dtype) = (yyvsp[0].dtype); 
                   (yyval.dtype).qualifier = 0;
		   (yyval.dtype).throws = 0;
		   (yyval.dtype).throwf = 0;
		   (yyval.dtype).nexcept = 0;
              }
#line 6460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 3119 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.dtype) = (yyvsp[0].dtype); 
		   (yyval.dtype).qualifier = (yyvsp[-1].str);
		   (yyval.dtype).throws = 0;
		   (yyval.dtype).throwf = 0;
		   (yyval.dtype).nexcept = 0;
	      }
#line 6472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 3126 "parser.y" /* yacc.c:1646  */
    { 
		   (yyval.dtype) = (yyvsp[0].dtype); 
                   (yyval.dtype).qualifier = 0;
		   (yyval.dtype).throws = (yyvsp[-1].dtype).throws;
		   (yyval.dtype).throwf = (yyvsp[-1].dtype).throwf;
		   (yyval.dtype).nexcept = (yyvsp[-1].dtype).nexcept;
              }
#line 6484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 3133 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.dtype) = (yyvsp[0].dtype); 
                   (yyval.dtype).qualifier = (yyvsp[-2].str);
		   (yyval.dtype).throws = (yyvsp[-1].dtype).throws;
		   (yyval.dtype).throwf = (yyvsp[-1].dtype).throwf;
		   (yyval.dtype).nexcept = (yyvsp[-1].dtype).nexcept;
              }
#line 6496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 3142 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].type); }
#line 6502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 3143 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].type); }
#line 6508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 3144 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].type); }
#line 6514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 3148 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].type); }
#line 6520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 3149 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].str); }
#line 6526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 3150 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].type); }
#line 6532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 3161 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.node) = new_node("lambda");
		  Setattr((yyval.node),"name",(yyvsp[-8].str));
		  add_symbols((yyval.node));
	        }
#line 6542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 3166 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.node) = new_node("lambda");
		  Setattr((yyval.node),"name",(yyvsp[-10].str));
		  add_symbols((yyval.node));
		}
#line 6552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 3171 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.node) = new_node("lambda");
		  Setattr((yyval.node),"name",(yyvsp[-4].str));
		  add_symbols((yyval.node));
		}
#line 6562 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 3178 "parser.y" /* yacc.c:1646  */
    {
		  skip_balanced('[',']');
		  (yyval.node) = 0;
	        }
#line 6571 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 3184 "parser.y" /* yacc.c:1646  */
    {
		  skip_balanced('{','}');
		  (yyval.node) = 0;
		}
#line 6580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 3189 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.pl) = 0;
		}
#line 6588 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 3192 "parser.y" /* yacc.c:1646  */
    {
		  skip_balanced('(',')');
		}
#line 6596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 3194 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.pl) = 0;
		}
#line 6604 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 3205 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.node) = (char *)"enum";
	      }
#line 6612 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 3208 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.node) = (char *)"enum class";
	      }
#line 6620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 3211 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.node) = (char *)"enum struct";
	      }
#line 6628 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 3220 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.node) = (yyvsp[0].type);
              }
#line 6636 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 3223 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 6642 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 3230 "parser.y" /* yacc.c:1646  */
    {
		   SwigType *ty = 0;
		   int scopedenum = (yyvsp[-2].id) && !Equal((yyvsp[-3].node), "enum");
		   (yyval.node) = new_node("enumforward");
		   ty = NewStringf("enum %s", (yyvsp[-2].id));
		   Setattr((yyval.node),"enumkey",(yyvsp[-3].node));
		   if (scopedenum)
		     SetFlag((yyval.node), "scopedenum");
		   Setattr((yyval.node),"name",(yyvsp[-2].id));
		   Setattr((yyval.node),"inherit",(yyvsp[-1].node));
		   Setattr((yyval.node),"type",ty);
		   Setattr((yyval.node),"sym:weak", "1");
		   add_symbols((yyval.node));
	      }
#line 6661 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 3252 "parser.y" /* yacc.c:1646  */
    {
		  SwigType *ty = 0;
		  int scopedenum = (yyvsp[-5].id) && !Equal((yyvsp[-6].node), "enum");
                  (yyval.node) = new_node("enum");
		  ty = NewStringf("enum %s", (yyvsp[-5].id));
		  Setattr((yyval.node),"enumkey",(yyvsp[-6].node));
		  if (scopedenum)
		    SetFlag((yyval.node), "scopedenum");
		  Setattr((yyval.node),"name",(yyvsp[-5].id));
		  Setattr((yyval.node),"inherit",(yyvsp[-4].node));
		  Setattr((yyval.node),"type",ty);
		  appendChild((yyval.node),(yyvsp[-2].node));
		  add_symbols((yyval.node));      /* Add to tag space */

		  if (scopedenum) {
		    Swig_symbol_newscope();
		    Swig_symbol_setscopename((yyvsp[-5].id));
		    Delete(Namespaceprefix);
		    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  }

		  add_symbols((yyvsp[-2].node));      /* Add enum values to appropriate enum or enum class scope */

		  if (scopedenum) {
		    Setattr((yyval.node),"symtab", Swig_symbol_popscope());
		    Delete(Namespaceprefix);
		    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  }
               }
#line 6695 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 3281 "parser.y" /* yacc.c:1646  */
    {
		 Node *n;
		 SwigType *ty = 0;
		 String   *unnamed = 0;
		 int       unnamedinstance = 0;
		 int scopedenum = (yyvsp[-7].id) && !Equal((yyvsp[-8].node), "enum");

		 (yyval.node) = new_node("enum");
		 Setattr((yyval.node),"enumkey",(yyvsp[-8].node));
		 if (scopedenum)
		   SetFlag((yyval.node), "scopedenum");
		 Setattr((yyval.node),"inherit",(yyvsp[-6].node));
		 if ((yyvsp[-7].id)) {
		   Setattr((yyval.node),"name",(yyvsp[-7].id));
		   ty = NewStringf("enum %s", (yyvsp[-7].id));
		 } else if ((yyvsp[-2].decl).id) {
		   unnamed = make_unnamed();
		   ty = NewStringf("enum %s", unnamed);
		   Setattr((yyval.node),"unnamed",unnamed);
                   /* name is not set for unnamed enum instances, e.g. enum { foo } Instance; */
		   if ((yyvsp[-9].id) && Cmp((yyvsp[-9].id),"typedef") == 0) {
		     Setattr((yyval.node),"name",(yyvsp[-2].decl).id);
                   } else {
                     unnamedinstance = 1;
                   }
		   Setattr((yyval.node),"storage",(yyvsp[-9].id));
		 }
		 if ((yyvsp[-2].decl).id && Cmp((yyvsp[-9].id),"typedef") == 0) {
		   Setattr((yyval.node),"tdname",(yyvsp[-2].decl).id);
                   Setattr((yyval.node),"allows_typedef","1");
                 }
		 appendChild((yyval.node),(yyvsp[-4].node));
		 n = new_node("cdecl");
		 Setattr(n,"type",ty);
		 Setattr(n,"name",(yyvsp[-2].decl).id);
		 Setattr(n,"storage",(yyvsp[-9].id));
		 Setattr(n,"decl",(yyvsp[-2].decl).type);
		 Setattr(n,"parms",(yyvsp[-2].decl).parms);
		 Setattr(n,"unnamed",unnamed);

                 if (unnamedinstance) {
		   SwigType *cty = NewString("enum ");
		   Setattr((yyval.node),"type",cty);
		   SetFlag((yyval.node),"unnamedinstance");
		   SetFlag(n,"unnamedinstance");
		   Delete(cty);
                 }
		 if ((yyvsp[0].node)) {
		   Node *p = (yyvsp[0].node);
		   set_nextSibling(n,p);
		   while (p) {
		     SwigType *cty = Copy(ty);
		     Setattr(p,"type",cty);
		     Setattr(p,"unnamed",unnamed);
		     Setattr(p,"storage",(yyvsp[-9].id));
		     Delete(cty);
		     p = nextSibling(p);
		   }
		 } else {
		   if (Len(scanner_ccode)) {
		     String *code = Copy(scanner_ccode);
		     Setattr(n,"code",code);
		     Delete(code);
		   }
		 }

                 /* Ensure that typedef enum ABC {foo} XYZ; uses XYZ for sym:name, like structs.
                  * Note that class_rename/yyrename are bit of a mess so used this simple approach to change the name. */
                 if ((yyvsp[-2].decl).id && (yyvsp[-7].id) && Cmp((yyvsp[-9].id),"typedef") == 0) {
		   String *name = NewString((yyvsp[-2].decl).id);
                   Setattr((yyval.node), "parser:makename", name);
		   Delete(name);
                 }

		 add_symbols((yyval.node));       /* Add enum to tag space */
		 set_nextSibling((yyval.node),n);
		 Delete(n);

		 if (scopedenum) {
		   Swig_symbol_newscope();
		   Swig_symbol_setscopename((yyvsp[-7].id));
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 }

		 add_symbols((yyvsp[-4].node));      /* Add enum values to appropriate enum or enum class scope */

		 if (scopedenum) {
		   Setattr((yyval.node),"symtab", Swig_symbol_popscope());
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 }

	         add_symbols(n);
		 Delete(unnamed);
	       }
#line 6796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 3379 "parser.y" /* yacc.c:1646  */
    {
                   /* This is a sick hack.  If the ctor_end has parameters,
                      and the parms parameter only has 1 parameter, this
                      could be a declaration of the form:

                         type (id)(parms)

			 Otherwise it's an error. */
                    int err = 0;
                    (yyval.node) = 0;

		    if ((ParmList_len((yyvsp[-2].pl)) == 1) && (!Swig_scopename_check((yyvsp[-4].type)))) {
		      SwigType *ty = Getattr((yyvsp[-2].pl),"type");
		      String *name = Getattr((yyvsp[-2].pl),"name");
		      err = 1;
		      if (!name) {
			(yyval.node) = new_node("cdecl");
			Setattr((yyval.node),"type",(yyvsp[-4].type));
			Setattr((yyval.node),"storage",(yyvsp[-5].id));
			Setattr((yyval.node),"name",ty);

			if ((yyvsp[0].decl).have_parms) {
			  SwigType *decl = NewStringEmpty();
			  SwigType_add_function(decl,(yyvsp[0].decl).parms);
			  Setattr((yyval.node),"decl",decl);
			  Setattr((yyval.node),"parms",(yyvsp[0].decl).parms);
			  if (Len(scanner_ccode)) {
			    String *code = Copy(scanner_ccode);
			    Setattr((yyval.node),"code",code);
			    Delete(code);
			  }
			}
			if ((yyvsp[0].decl).defarg) {
			  Setattr((yyval.node),"value",(yyvsp[0].decl).defarg);
			}
			Setattr((yyval.node),"throws",(yyvsp[0].decl).throws);
			Setattr((yyval.node),"throw",(yyvsp[0].decl).throwf);
			Setattr((yyval.node),"noexcept",(yyvsp[0].decl).nexcept);
			err = 0;
		      }
		    }
		    if (err) {
		      Swig_error(cparse_file,cparse_line,"Syntax error in input(2).\n");
		      exit(1);
		    }
                }
#line 6847 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 3431 "parser.y" /* yacc.c:1646  */
    {  (yyval.node) = (yyvsp[0].node); }
#line 6853 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 3432 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 6859 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 3433 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 6865 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 3434 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 6871 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 3435 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 6877 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 3436 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 6883 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 3441 "parser.y" /* yacc.c:1646  */
    {
                   String *prefix;
                   List *bases = 0;
		   Node *scope = 0;
		   String *code;
		   (yyval.node) = new_node("class");
		   Setline((yyval.node),cparse_start_line);
		   Setattr((yyval.node),"kind",(yyvsp[-3].id));
		   if ((yyvsp[-1].bases)) {
		     Setattr((yyval.node),"baselist", Getattr((yyvsp[-1].bases),"public"));
		     Setattr((yyval.node),"protectedbaselist", Getattr((yyvsp[-1].bases),"protected"));
		     Setattr((yyval.node),"privatebaselist", Getattr((yyvsp[-1].bases),"private"));
		   }
		   Setattr((yyval.node),"allows_typedef","1");

		   /* preserve the current scope */
		   Setattr((yyval.node),"prev_symtab",Swig_symbol_current());
		  
		   /* If the class name is qualified.  We need to create or lookup namespace/scope entries */
		   scope = resolve_create_node_scope((yyvsp[-2].str));
		   /* save nscope_inner to the class - it may be overwritten in nested classes*/
		   Setattr((yyval.node), "nested:innerscope", nscope_inner);
		   Setattr((yyval.node), "nested:nscope", nscope);
		   Setfile(scope,cparse_file);
		   Setline(scope,cparse_line);
		   (yyvsp[-2].str) = scope;
		   Setattr((yyval.node),"name",(yyvsp[-2].str));

		   if (currentOuterClass) {
		     SetFlag((yyval.node), "nested");
		     Setattr((yyval.node), "nested:outer", currentOuterClass);
		     set_access_mode((yyval.node));
		   }
		   Swig_features_get(Swig_cparse_features(), Namespaceprefix, Getattr((yyval.node), "name"), 0, (yyval.node));
		   /* save yyrename to the class attribute, to be used later in add_symbols()*/
		   Setattr((yyval.node), "class_rename", make_name((yyval.node), (yyvsp[-2].str), 0));
		   Setattr((yyval.node), "Classprefix", (yyvsp[-2].str));
		   Classprefix = NewString((yyvsp[-2].str));
		   /* Deal with inheritance  */
		   if ((yyvsp[-1].bases))
		     bases = Swig_make_inherit_list((yyvsp[-2].str),Getattr((yyvsp[-1].bases),"public"),Namespaceprefix);
		   prefix = SwigType_istemplate_templateprefix((yyvsp[-2].str));
		   if (prefix) {
		     String *fbase, *tbase;
		     if (Namespaceprefix) {
		       fbase = NewStringf("%s::%s", Namespaceprefix,(yyvsp[-2].str));
		       tbase = NewStringf("%s::%s", Namespaceprefix, prefix);
		     } else {
		       fbase = Copy((yyvsp[-2].str));
		       tbase = Copy(prefix);
		     }
		     Swig_name_inherit(tbase,fbase);
		     Delete(fbase);
		     Delete(tbase);
		   }
                   if (strcmp((yyvsp[-3].id),"class") == 0) {
		     cplus_mode = CPLUS_PRIVATE;
		   } else {
		     cplus_mode = CPLUS_PUBLIC;
		   }
		   if (!cparse_cplusplus) {
		     set_scope_to_global();
		   }
		   Swig_symbol_newscope();
		   Swig_symbol_setscopename((yyvsp[-2].str));
		   Swig_inherit_base_symbols(bases);
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   cparse_start_line = cparse_line;

		   /* If there are active template parameters, we need to make sure they are
                      placed in the class symbol table so we can catch shadows */

		   if (template_parameters) {
		     Parm *tp = template_parameters;
		     while(tp) {
		       String *tpname = Copy(Getattr(tp,"name"));
		       Node *tn = new_node("templateparm");
		       Setattr(tn,"name",tpname);
		       Swig_symbol_cadd(tpname,tn);
		       tp = nextSibling(tp);
		       Delete(tpname);
		     }
		   }
		   Delete(prefix);
		   inclass = 1;
		   currentOuterClass = (yyval.node);
		   if (cparse_cplusplusout) {
		     /* save the structure declaration to declare it in global scope for C++ to see */
		     code = get_raw_text_balanced('{', '}');
		     Setattr((yyval.node), "code", code);
		     Delete(code);
		   }
               }
#line 6982 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 3534 "parser.y" /* yacc.c:1646  */
    {
		   Node *p;
		   SwigType *ty;
		   Symtab *cscope;
		   Node *am = 0;
		   String *scpname = 0;
		   (void) (yyvsp[-3].node);
		   (yyval.node) = currentOuterClass;
		   currentOuterClass = Getattr((yyval.node), "nested:outer");
		   nscope_inner = Getattr((yyval.node), "nested:innerscope");
		   nscope = Getattr((yyval.node), "nested:nscope");
		   Delattr((yyval.node), "nested:innerscope");
		   Delattr((yyval.node), "nested:nscope");
		   if (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0) { /* actual parent class for this class */
		     Node* forward_declaration = Swig_symbol_clookup_no_inherit(Getattr((yyval.node),"name"), Getattr(nscope_inner, "symtab"));
		     if (forward_declaration) {
		       Setattr((yyval.node), "access", Getattr(forward_declaration, "access"));
		     }
		     Setattr((yyval.node), "nested:outer", nscope_inner);
		     SetFlag((yyval.node), "nested");
                   }
		   if (!currentOuterClass)
		     inclass = 0;
		   cscope = Getattr((yyval.node), "prev_symtab");
		   Delattr((yyval.node), "prev_symtab");
		   
		   /* Check for pure-abstract class */
		   Setattr((yyval.node),"abstracts", pure_abstracts((yyvsp[-2].node)));
		   
		   /* This bit of code merges in a previously defined %extend directive (if any) */
		   {
		     String *clsname = Swig_symbol_qualifiedscopename(0);
		     am = Getattr(Swig_extend_hash(), clsname);
		     if (am) {
		       Swig_extend_merge((yyval.node), am);
		       Delattr(Swig_extend_hash(), clsname);
		     }
		     Delete(clsname);
		   }
		   if (!classes) classes = NewHash();
		   scpname = Swig_symbol_qualifiedscopename(0);
		   Setattr(classes, scpname, (yyval.node));

		   appendChild((yyval.node), (yyvsp[-2].node));
		   
		   if (am) 
		     Swig_extend_append_previous((yyval.node), am);

		   p = (yyvsp[0].node);
		   if (p && !nscope_inner) {
		     if (!cparse_cplusplus && currentOuterClass)
		       appendChild(currentOuterClass, p);
		     else
		      appendSibling((yyval.node), p);
		   }
		   
		   if (nscope_inner) {
		     ty = NewString(scpname); /* if the class is declared out of scope, let the declarator use fully qualified type*/
		   } else if (cparse_cplusplus && !cparse_externc) {
		     ty = NewString((yyvsp[-6].str));
		   } else {
		     ty = NewStringf("%s %s", (yyvsp[-7].id), (yyvsp[-6].str));
		   }
		   while (p) {
		     Setattr(p, "storage", (yyvsp[-8].id));
		     Setattr(p, "type" ,ty);
		     if (!cparse_cplusplus && currentOuterClass && (!Getattr(currentOuterClass, "name"))) {
		       SetFlag(p, "hasconsttype");
		       SetFlag(p, "feature:immutable");
		     }
		     p = nextSibling(p);
		   }
		   if ((yyvsp[0].node) && Cmp((yyvsp[-8].id),"typedef") == 0)
		     add_typedef_name((yyval.node), (yyvsp[0].node), (yyvsp[-6].str), cscope, scpname);
		   Delete(scpname);

		   if (cplus_mode != CPLUS_PUBLIC) {
		   /* we 'open' the class at the end, to allow %template
		      to add new members */
		     Node *pa = new_node("access");
		     Setattr(pa, "kind", "public");
		     cplus_mode = CPLUS_PUBLIC;
		     appendChild((yyval.node), pa);
		     Delete(pa);
		   }
		   if (currentOuterClass)
		     restore_access_mode((yyval.node));
		   Setattr((yyval.node), "symtab", Swig_symbol_popscope());
		   Classprefix = Getattr((yyval.node), "Classprefix");
		   Delattr((yyval.node), "Classprefix");
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   if (cplus_mode == CPLUS_PRIVATE) {
		     (yyval.node) = 0; /* skip private nested classes */
		   } else if (cparse_cplusplus && currentOuterClass && ignore_nested_classes && !GetFlag((yyval.node), "feature:flatnested")) {
		     (yyval.node) = nested_forward_declaration((yyvsp[-8].id), (yyvsp[-7].id), (yyvsp[-6].str), Copy((yyvsp[-6].str)), (yyvsp[0].node));
		   } else if (nscope_inner) {
		     /* this is tricky */
		     /* we add the declaration in the original namespace */
		     if (Strcmp(nodeType(nscope_inner), "class") == 0 && cparse_cplusplus && ignore_nested_classes && !GetFlag((yyval.node), "feature:flatnested"))
		       (yyval.node) = nested_forward_declaration((yyvsp[-8].id), (yyvsp[-7].id), (yyvsp[-6].str), Copy((yyvsp[-6].str)), (yyvsp[0].node));
		     appendChild(nscope_inner, (yyval.node));
		     Swig_symbol_setscope(Getattr(nscope_inner, "symtab"));
		     Delete(Namespaceprefix);
		     Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		     yyrename = Copy(Getattr((yyval.node), "class_rename"));
		     add_symbols((yyval.node));
		     Delattr((yyval.node), "class_rename");
		     /* but the variable definition in the current scope */
		     Swig_symbol_setscope(cscope);
		     Delete(Namespaceprefix);
		     Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		     add_symbols((yyvsp[0].node));
		     if (nscope) {
		       (yyval.node) = nscope; /* here we return recreated namespace tower instead of the class itself */
		       if ((yyvsp[0].node)) {
			 appendSibling((yyval.node), (yyvsp[0].node));
		       }
		     } else if (!SwigType_istemplate(ty) && template_parameters == 0) { /* for tempalte we need the class itself */
		       (yyval.node) = (yyvsp[0].node);
		     }
		   } else {
		     Delete(yyrename);
		     yyrename = 0;
		     if (!cparse_cplusplus && currentOuterClass) { /* nested C structs go into global scope*/
		       Node *outer = currentOuterClass;
		       while (Getattr(outer, "nested:outer"))
			 outer = Getattr(outer, "nested:outer");
		       appendSibling(outer, (yyval.node));
		       add_symbols((yyvsp[0].node));
		       set_scope_to_global();
		       Delete(Namespaceprefix);
		       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		       yyrename = Copy(Getattr((yyval.node), "class_rename"));
		       add_symbols((yyval.node));
		       if (!cparse_cplusplusout)
			 Delattr((yyval.node), "nested:outer");
		       Delattr((yyval.node), "class_rename");
		       (yyval.node) = 0;
		     } else {
		       yyrename = Copy(Getattr((yyval.node), "class_rename"));
		       add_symbols((yyval.node));
		       add_symbols((yyvsp[0].node));
		       Delattr((yyval.node), "class_rename");
		     }
		   }
		   Delete(ty);
		   Swig_symbol_setscope(cscope);
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   Classprefix = currentOuterClass ? Getattr(currentOuterClass, "Classprefix") : 0;
	       }
#line 7139 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 3689 "parser.y" /* yacc.c:1646  */
    {
	       String *unnamed;
	       String *code;
	       unnamed = make_unnamed();
	       (yyval.node) = new_node("class");
	       Setline((yyval.node),cparse_start_line);
	       Setattr((yyval.node),"kind",(yyvsp[-2].id));
	       if ((yyvsp[-1].bases)) {
		 Setattr((yyval.node),"baselist", Getattr((yyvsp[-1].bases),"public"));
		 Setattr((yyval.node),"protectedbaselist", Getattr((yyvsp[-1].bases),"protected"));
		 Setattr((yyval.node),"privatebaselist", Getattr((yyvsp[-1].bases),"private"));
	       }
	       Setattr((yyval.node),"storage",(yyvsp[-3].id));
	       Setattr((yyval.node),"unnamed",unnamed);
	       Setattr((yyval.node),"allows_typedef","1");
	       if (currentOuterClass) {
		 SetFlag((yyval.node), "nested");
		 Setattr((yyval.node), "nested:outer", currentOuterClass);
		 set_access_mode((yyval.node));
	       }
	       Swig_features_get(Swig_cparse_features(), Namespaceprefix, 0, 0, (yyval.node));
	       /* save yyrename to the class attribute, to be used later in add_symbols()*/
	       Setattr((yyval.node), "class_rename", make_name((yyval.node),0,0));
	       if (strcmp((yyvsp[-2].id),"class") == 0) {
		 cplus_mode = CPLUS_PRIVATE;
	       } else {
		 cplus_mode = CPLUS_PUBLIC;
	       }
	       Swig_symbol_newscope();
	       cparse_start_line = cparse_line;
	       currentOuterClass = (yyval.node);
	       inclass = 1;
	       Classprefix = 0;
	       Delete(Namespaceprefix);
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       /* save the structure declaration to make a typedef for it later*/
	       code = get_raw_text_balanced('{', '}');
	       Setattr((yyval.node), "code", code);
	       Delete(code);
	     }
#line 7184 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 3728 "parser.y" /* yacc.c:1646  */
    {
	       String *unnamed;
               List *bases = 0;
	       String *name = 0;
	       Node *n;
	       Classprefix = 0;
	       (void)(yyvsp[-3].node);
	       (yyval.node) = currentOuterClass;
	       currentOuterClass = Getattr((yyval.node), "nested:outer");
	       if (!currentOuterClass)
		 inclass = 0;
	       else
		 restore_access_mode((yyval.node));
	       unnamed = Getattr((yyval.node),"unnamed");
               /* Check for pure-abstract class */
	       Setattr((yyval.node),"abstracts", pure_abstracts((yyvsp[-2].node)));
	       n = (yyvsp[0].node);
	       if (cparse_cplusplus && currentOuterClass && ignore_nested_classes && !GetFlag((yyval.node), "feature:flatnested")) {
		 String *name = n ? Copy(Getattr(n, "name")) : 0;
		 (yyval.node) = nested_forward_declaration((yyvsp[-7].id), (yyvsp[-6].id), 0, name, n);
		 Swig_symbol_popscope();
	         Delete(Namespaceprefix);
		 Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       } else if (n) {
	         appendSibling((yyval.node),n);
		 /* If a proper typedef name was given, we'll use it to set the scope name */
		 name = try_to_find_a_name_for_unnamed_structure((yyvsp[-7].id), n);
		 if (name) {
		   String *scpname = 0;
		   SwigType *ty;
		   Setattr((yyval.node),"tdname",name);
		   Setattr((yyval.node),"name",name);
		   Swig_symbol_setscopename(name);
		   if ((yyvsp[-5].bases))
		     bases = Swig_make_inherit_list(name,Getattr((yyvsp[-5].bases),"public"),Namespaceprefix);
		   Swig_inherit_base_symbols(bases);

		     /* If a proper name was given, we use that as the typedef, not unnamed */
		   Clear(unnamed);
		   Append(unnamed, name);
		   if (cparse_cplusplus && !cparse_externc) {
		     ty = NewString(name);
		   } else {
		     ty = NewStringf("%s %s", (yyvsp[-6].id),name);
		   }
		   while (n) {
		     Setattr(n,"storage",(yyvsp[-7].id));
		     Setattr(n, "type", ty);
		     if (!cparse_cplusplus && currentOuterClass && (!Getattr(currentOuterClass, "name"))) {
		       SetFlag(n,"hasconsttype");
		       SetFlag(n,"feature:immutable");
		     }
		     n = nextSibling(n);
		   }
		   n = (yyvsp[0].node);

		   /* Check for previous extensions */
		   {
		     String *clsname = Swig_symbol_qualifiedscopename(0);
		     Node *am = Getattr(Swig_extend_hash(),clsname);
		     if (am) {
		       /* Merge the extension into the symbol table */
		       Swig_extend_merge((yyval.node),am);
		       Swig_extend_append_previous((yyval.node),am);
		       Delattr(Swig_extend_hash(),clsname);
		     }
		     Delete(clsname);
		   }
		   if (!classes) classes = NewHash();
		   scpname = Swig_symbol_qualifiedscopename(0);
		   Setattr(classes,scpname,(yyval.node));
		   Delete(scpname);
		 } else { /* no suitable name was found for a struct */
		   Setattr((yyval.node), "nested:unnamed", Getattr(n, "name")); /* save the name of the first declarator for later use in name generation*/
		   while (n) { /* attach unnamed struct to the declarators, so that they would receive proper type later*/
		     Setattr(n, "nested:unnamedtype", (yyval.node));
		     Setattr(n, "storage", (yyvsp[-7].id));
		     n = nextSibling(n);
		   }
		   n = (yyvsp[0].node);
		   Swig_symbol_setscopename("<unnamed>");
		 }
		 appendChild((yyval.node),(yyvsp[-2].node));
		 /* Pop the scope */
		 Setattr((yyval.node),"symtab",Swig_symbol_popscope());
		 if (name) {
		   Delete(yyrename);
		   yyrename = Copy(Getattr((yyval.node), "class_rename"));
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   add_symbols((yyval.node));
		   add_symbols(n);
		   Delattr((yyval.node), "class_rename");
		 }else if (cparse_cplusplus)
		   (yyval.node) = 0; /* ignore unnamed structs for C++ */
	         Delete(unnamed);
	       } else { /* unnamed struct w/o declarator*/
		 Swig_symbol_popscope();
	         Delete(Namespaceprefix);
		 Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 add_symbols((yyvsp[-2].node));
		 Delete((yyval.node));
		 (yyval.node) = (yyvsp[-2].node); /* pass member list to outer class/namespace (instead of self)*/
	       }
	       Classprefix = currentOuterClass ? Getattr(currentOuterClass, "Classprefix") : 0;
              }
#line 7295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 3836 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 7301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 3837 "parser.y" /* yacc.c:1646  */
    {
                        (yyval.node) = new_node("cdecl");
                        Setattr((yyval.node),"name",(yyvsp[-2].decl).id);
                        Setattr((yyval.node),"decl",(yyvsp[-2].decl).type);
                        Setattr((yyval.node),"parms",(yyvsp[-2].decl).parms);
			set_nextSibling((yyval.node),(yyvsp[0].node));
                    }
#line 7313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 3849 "parser.y" /* yacc.c:1646  */
    {
              if ((yyvsp[-3].id) && (Strcmp((yyvsp[-3].id),"friend") == 0)) {
		/* Ignore */
                (yyval.node) = 0; 
	      } else {
		(yyval.node) = new_node("classforward");
		Setattr((yyval.node),"kind",(yyvsp[-2].id));
		Setattr((yyval.node),"name",(yyvsp[-1].str));
		Setattr((yyval.node),"sym:weak", "1");
		add_symbols((yyval.node));
	      }
             }
#line 7330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 3867 "parser.y" /* yacc.c:1646  */
    { 
		    if (currentOuterClass)
		      Setattr(currentOuterClass, "template_parameters", template_parameters);
		    template_parameters = (yyvsp[-1].tparms); 
		    parsing_template_declaration = 1;
		  }
#line 7341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 3872 "parser.y" /* yacc.c:1646  */
    {
			String *tname = 0;
			int     error = 0;

			/* check if we get a namespace node with a class declaration, and retrieve the class */
			Symtab *cscope = Swig_symbol_current();
			Symtab *sti = 0;
			Node *ntop = (yyvsp[0].node);
			Node *ni = ntop;
			SwigType *ntype = ni ? nodeType(ni) : 0;
			while (ni && Strcmp(ntype,"namespace") == 0) {
			  sti = Getattr(ni,"symtab");
			  ni = firstChild(ni);
			  ntype = nodeType(ni);
			}
			if (sti) {
			  Swig_symbol_setscope(sti);
			  Delete(Namespaceprefix);
			  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
			  (yyvsp[0].node) = ni;
			}

			(yyval.node) = (yyvsp[0].node);
			if ((yyval.node)) tname = Getattr((yyval.node),"name");
			
			/* Check if the class is a template specialization */
			if (((yyval.node)) && (Strchr(tname,'<')) && (!is_operator(tname))) {
			  /* If a specialization.  Check if defined. */
			  Node *tempn = 0;
			  {
			    String *tbase = SwigType_templateprefix(tname);
			    tempn = Swig_symbol_clookup_local(tbase,0);
			    if (!tempn || (Strcmp(nodeType(tempn),"template") != 0)) {
			      SWIG_WARN_NODE_BEGIN(tempn);
			      Swig_warning(WARN_PARSE_TEMPLATE_SP_UNDEF, Getfile((yyval.node)),Getline((yyval.node)),"Specialization of non-template '%s'.\n", tbase);
			      SWIG_WARN_NODE_END(tempn);
			      tempn = 0;
			      error = 1;
			    }
			    Delete(tbase);
			  }
			  Setattr((yyval.node),"specialization","1");
			  Setattr((yyval.node),"templatetype",nodeType((yyval.node)));
			  set_nodeType((yyval.node),"template");
			  /* Template partial specialization */
			  if (tempn && ((yyvsp[-3].tparms)) && ((yyvsp[0].node))) {
			    List   *tlist;
			    String *targs = SwigType_templateargs(tname);
			    tlist = SwigType_parmlist(targs);
			    /*			  Printf(stdout,"targs = '%s' %s\n", targs, tlist); */
			    if (!Getattr((yyval.node),"sym:weak")) {
			      Setattr((yyval.node),"sym:typename","1");
			    }
			    
			    if (Len(tlist) != ParmList_len(Getattr(tempn,"templateparms"))) {
			      Swig_error(Getfile((yyval.node)),Getline((yyval.node)),"Inconsistent argument count in template partial specialization. %d %d\n", Len(tlist), ParmList_len(Getattr(tempn,"templateparms")));
			      
			    } else {

			    /* This code builds the argument list for the partial template
			       specialization.  This is a little hairy, but the idea is as
			       follows:

			       $3 contains a list of arguments supplied for the template.
			       For example template<class T>.

			       tlist is a list of the specialization arguments--which may be
			       different.  For example class<int,T>.

			       tp is a copy of the arguments in the original template definition.
       
			       The patching algorithm walks through the list of supplied
			       arguments ($3), finds the position in the specialization arguments
			       (tlist), and then patches the name in the argument list of the
			       original template.
			    */

			    {
			      String *pn;
			      Parm *p, *p1;
			      int i, nargs;
			      Parm *tp = CopyParmList(Getattr(tempn,"templateparms"));
			      nargs = Len(tlist);
			      p = (yyvsp[-3].tparms);
			      while (p) {
				for (i = 0; i < nargs; i++){
				  pn = Getattr(p,"name");
				  if (Strcmp(pn,SwigType_base(Getitem(tlist,i))) == 0) {
				    int j;
				    Parm *p1 = tp;
				    for (j = 0; j < i; j++) {
				      p1 = nextSibling(p1);
				    }
				    Setattr(p1,"name",pn);
				    Setattr(p1,"partialarg","1");
				  }
				}
				p = nextSibling(p);
			      }
			      p1 = tp;
			      i = 0;
			      while (p1) {
				if (!Getattr(p1,"partialarg")) {
				  Delattr(p1,"name");
				  Setattr(p1,"type", Getitem(tlist,i));
				} 
				i++;
				p1 = nextSibling(p1);
			      }
			      Setattr((yyval.node),"templateparms",tp);
			      Delete(tp);
			    }
  #if 0
			    /* Patch the parameter list */
			    if (tempn) {
			      Parm *p,*p1;
			      ParmList *tp = CopyParmList(Getattr(tempn,"templateparms"));
			      p = (yyvsp[-3].tparms);
			      p1 = tp;
			      while (p && p1) {
				String *pn = Getattr(p,"name");
				Printf(stdout,"pn = '%s'\n", pn);
				if (pn) Setattr(p1,"name",pn);
				else Delattr(p1,"name");
				pn = Getattr(p,"type");
				if (pn) Setattr(p1,"type",pn);
				p = nextSibling(p);
				p1 = nextSibling(p1);
			      }
			      Setattr((yyval.node),"templateparms",tp);
			      Delete(tp);
			    } else {
			      Setattr((yyval.node),"templateparms",(yyvsp[-3].tparms));
			    }
  #endif
			    Delattr((yyval.node),"specialization");
			    Setattr((yyval.node),"partialspecialization","1");
			    /* Create a specialized name for matching */
			    {
			      Parm *p = (yyvsp[-3].tparms);
			      String *fname = NewString(Getattr((yyval.node),"name"));
			      String *ffname = 0;
			      ParmList *partialparms = 0;

			      char   tmp[32];
			      int    i, ilen;
			      while (p) {
				String *n = Getattr(p,"name");
				if (!n) {
				  p = nextSibling(p);
				  continue;
				}
				ilen = Len(tlist);
				for (i = 0; i < ilen; i++) {
				  if (Strstr(Getitem(tlist,i),n)) {
				    sprintf(tmp,"$%d",i+1);
				    Replaceid(fname,n,tmp);
				  }
				}
				p = nextSibling(p);
			      }
			      /* Patch argument names with typedef */
			      {
				Iterator tt;
				Parm *parm_current = 0;
				List *tparms = SwigType_parmlist(fname);
				ffname = SwigType_templateprefix(fname);
				Append(ffname,"<(");
				for (tt = First(tparms); tt.item; ) {
				  SwigType *rtt = Swig_symbol_typedef_reduce(tt.item,0);
				  SwigType *ttr = Swig_symbol_type_qualify(rtt,0);

				  Parm *newp = NewParmWithoutFileLineInfo(ttr, 0);
				  if (partialparms)
				    set_nextSibling(parm_current, newp);
				  else
				    partialparms = newp;
				  parm_current = newp;

				  Append(ffname,ttr);
				  tt = Next(tt);
				  if (tt.item) Putc(',',ffname);
				  Delete(rtt);
				  Delete(ttr);
				}
				Delete(tparms);
				Append(ffname,")>");
			      }
			      {
				Node *new_partial = NewHash();
				String *partials = Getattr(tempn,"partials");
				if (!partials) {
				  partials = NewList();
				  Setattr(tempn,"partials",partials);
				  Delete(partials);
				}
				/*			      Printf(stdout,"partial: fname = '%s', '%s'\n", fname, Swig_symbol_typedef_reduce(fname,0)); */
				Setattr(new_partial, "partialparms", partialparms);
				Setattr(new_partial, "templcsymname", ffname);
				Append(partials, new_partial);
			      }
			      Setattr((yyval.node),"partialargs",ffname);
			      Swig_symbol_cadd(ffname,(yyval.node));
			    }
			    }
			    Delete(tlist);
			    Delete(targs);
			  } else {
			    /* An explicit template specialization */
			    /* add default args from primary (unspecialized) template */
			    String *ty = Swig_symbol_template_deftype(tname,0);
			    String *fname = Swig_symbol_type_qualify(ty,0);
			    Swig_symbol_cadd(fname,(yyval.node));
			    Delete(ty);
			    Delete(fname);
			  }
			}  else if ((yyval.node)) {
			  Setattr((yyval.node),"templatetype",nodeType((yyvsp[0].node)));
			  set_nodeType((yyval.node),"template");
			  Setattr((yyval.node),"templateparms", (yyvsp[-3].tparms));
			  if (!Getattr((yyval.node),"sym:weak")) {
			    Setattr((yyval.node),"sym:typename","1");
			  }
			  add_symbols((yyval.node));
			  default_arguments((yyval.node));
			  /* We also place a fully parameterized version in the symbol table */
			  {
			    Parm *p;
			    String *fname = NewStringf("%s<(", Getattr((yyval.node),"name"));
			    p = (yyvsp[-3].tparms);
			    while (p) {
			      String *n = Getattr(p,"name");
			      if (!n) n = Getattr(p,"type");
			      Append(fname,n);
			      p = nextSibling(p);
			      if (p) Putc(',',fname);
			    }
			    Append(fname,")>");
			    Swig_symbol_cadd(fname,(yyval.node));
			  }
			}
			(yyval.node) = ntop;
			Swig_symbol_setscope(cscope);
			Delete(Namespaceprefix);
			Namespaceprefix = Swig_symbol_qualifiedscopename(0);
			if (error || (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0)) {
			  (yyval.node) = 0;
			}
			if (currentOuterClass)
			  template_parameters = Getattr(currentOuterClass, "template_parameters");
			else
			  template_parameters = 0;
			parsing_template_declaration = 0;
                }
#line 7600 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 4128 "parser.y" /* yacc.c:1646  */
    {
		  Swig_warning(WARN_PARSE_EXPLICIT_TEMPLATE, cparse_file, cparse_line, "Explicit template instantiation ignored.\n");
                  (yyval.node) = 0; 
		}
#line 7609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 4134 "parser.y" /* yacc.c:1646  */
    {
		  Swig_warning(WARN_PARSE_EXPLICIT_TEMPLATE, cparse_file, cparse_line, "Explicit template instantiation ignored.\n");
                  (yyval.node) = 0; 
                }
#line 7618 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 4140 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.node) = (yyvsp[0].node);
                }
#line 7626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 4143 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.node) = (yyvsp[0].node);
                }
#line 7634 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 4146 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.node) = (yyvsp[0].node);
                }
#line 7642 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 4149 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.node) = (yyvsp[0].node);
                }
#line 7650 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 4152 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.node) = 0;
                }
#line 7658 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 4155 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.node) = (yyvsp[0].node);
                }
#line 7666 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 4158 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.node) = (yyvsp[0].node);
                }
#line 7674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 4163 "parser.y" /* yacc.c:1646  */
    {
		   /* Rip out the parameter names */
		  Parm *p = (yyvsp[0].pl);
		  (yyval.tparms) = (yyvsp[0].pl);

		  while (p) {
		    String *name = Getattr(p,"name");
		    if (!name) {
		      /* Hmmm. Maybe it's a 'class T' parameter */
		      char *type = Char(Getattr(p,"type"));
		      /* Template template parameter */
		      if (strncmp(type,"template<class> ",16) == 0) {
			type += 16;
		      }
		      if ((strncmp(type,"class ",6) == 0) || (strncmp(type,"typename ", 9) == 0)) {
			char *t = strchr(type,' ');
			Setattr(p,"name", t+1);
		      } else 
                      /* Variadic template args */
		      if ((strncmp(type,"class... ",9) == 0) || (strncmp(type,"typename... ", 12) == 0)) {
			char *t = strchr(type,' ');
			Setattr(p,"name", t+1);
			Setattr(p,"variadic", "1");
		      } else {
			/*
			 Swig_error(cparse_file, cparse_line, "Missing template parameter name\n");
			 $$.rparms = 0;
			 $$.parms = 0;
			 break; */
		      }
		    }
		    p = nextSibling(p);
		  }
                 }
#line 7713 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 4199 "parser.y" /* yacc.c:1646  */
    {
                      set_nextSibling((yyvsp[-1].p),(yyvsp[0].pl));
                      (yyval.pl) = (yyvsp[-1].p);
                   }
#line 7722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 4203 "parser.y" /* yacc.c:1646  */
    { (yyval.pl) = 0; }
#line 7728 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 4206 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.p) = NewParmWithoutFileLineInfo(NewString((yyvsp[0].id)), 0);
                  }
#line 7736 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 4209 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.p) = (yyvsp[0].p);
                  }
#line 7744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 4214 "parser.y" /* yacc.c:1646  */
    {
                         set_nextSibling((yyvsp[-1].p),(yyvsp[0].pl));
                         (yyval.pl) = (yyvsp[-1].p);
                       }
#line 7753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 4218 "parser.y" /* yacc.c:1646  */
    { (yyval.pl) = 0; }
#line 7759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 4223 "parser.y" /* yacc.c:1646  */
    {
                  String *uname = Swig_symbol_type_qualify((yyvsp[-1].str),0);
		  String *name = Swig_scopename_last((yyvsp[-1].str));
                  (yyval.node) = new_node("using");
		  Setattr((yyval.node),"uname",uname);
		  Setattr((yyval.node),"name", name);
		  Delete(uname);
		  Delete(name);
		  add_symbols((yyval.node));
             }
#line 7774 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 4233 "parser.y" /* yacc.c:1646  */
    {
	       Node *n = Swig_symbol_clookup((yyvsp[-1].str),0);
	       if (!n) {
		 Swig_error(cparse_file, cparse_line, "Nothing known about namespace '%s'\n", (yyvsp[-1].str));
		 (yyval.node) = 0;
	       } else {

		 while (Strcmp(nodeType(n),"using") == 0) {
		   n = Getattr(n,"node");
		 }
		 if (n) {
		   if (Strcmp(nodeType(n),"namespace") == 0) {
		     Symtab *current = Swig_symbol_current();
		     Symtab *symtab = Getattr(n,"symtab");
		     (yyval.node) = new_node("using");
		     Setattr((yyval.node),"node",n);
		     Setattr((yyval.node),"namespace", (yyvsp[-1].str));
		     if (current != symtab) {
		       Swig_symbol_inherit(symtab);
		     }
		   } else {
		     Swig_error(cparse_file, cparse_line, "'%s' is not a namespace.\n", (yyvsp[-1].str));
		     (yyval.node) = 0;
		   }
		 } else {
		   (yyval.node) = 0;
		 }
	       }
             }
#line 7808 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 4264 "parser.y" /* yacc.c:1646  */
    { 
                Hash *h;
                (yyvsp[-2].node) = Swig_symbol_current();
		h = Swig_symbol_clookup((yyvsp[-1].str),0);
		if (h && ((yyvsp[-2].node) == Getattr(h,"sym:symtab")) && (Strcmp(nodeType(h),"namespace") == 0)) {
		  if (Getattr(h,"alias")) {
		    h = Getattr(h,"namespace");
		    Swig_warning(WARN_PARSE_NAMESPACE_ALIAS, cparse_file, cparse_line, "Namespace alias '%s' not allowed here. Assuming '%s'\n",
				 (yyvsp[-1].str), Getattr(h,"name"));
		    (yyvsp[-1].str) = Getattr(h,"name");
		  }
		  Swig_symbol_setscope(Getattr(h,"symtab"));
		} else {
		  Swig_symbol_newscope();
		  Swig_symbol_setscopename((yyvsp[-1].str));
		}
		Delete(Namespaceprefix);
		Namespaceprefix = Swig_symbol_qualifiedscopename(0);
             }
#line 7832 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 4282 "parser.y" /* yacc.c:1646  */
    {
                Node *n = (yyvsp[-1].node);
		set_nodeType(n,"namespace");
		Setattr(n,"name",(yyvsp[-4].str));
                Setattr(n,"symtab", Swig_symbol_popscope());
		Swig_symbol_setscope((yyvsp[-5].node));
		(yyval.node) = n;
		Delete(Namespaceprefix);
		Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		add_symbols((yyval.node));
             }
#line 7848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 4293 "parser.y" /* yacc.c:1646  */
    {
	       Hash *h;
	       (yyvsp[-1].node) = Swig_symbol_current();
	       h = Swig_symbol_clookup("    ",0);
	       if (h && (Strcmp(nodeType(h),"namespace") == 0)) {
		 Swig_symbol_setscope(Getattr(h,"symtab"));
	       } else {
		 Swig_symbol_newscope();
		 /* we don't use "__unnamed__", but a long 'empty' name */
		 Swig_symbol_setscopename("    ");
	       }
	       Namespaceprefix = 0;
             }
#line 7866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 4305 "parser.y" /* yacc.c:1646  */
    {
	       (yyval.node) = (yyvsp[-1].node);
	       set_nodeType((yyval.node),"namespace");
	       Setattr((yyval.node),"unnamed","1");
	       Setattr((yyval.node),"symtab", Swig_symbol_popscope());
	       Swig_symbol_setscope((yyvsp[-4].node));
	       Delete(Namespaceprefix);
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       add_symbols((yyval.node));
             }
#line 7881 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 4315 "parser.y" /* yacc.c:1646  */
    {
	       /* Namespace alias */
	       Node *n;
	       (yyval.node) = new_node("namespace");
	       Setattr((yyval.node),"name",(yyvsp[-3].id));
	       Setattr((yyval.node),"alias",(yyvsp[-1].str));
	       n = Swig_symbol_clookup((yyvsp[-1].str),0);
	       if (!n) {
		 Swig_error(cparse_file, cparse_line, "Unknown namespace '%s'\n", (yyvsp[-1].str));
		 (yyval.node) = 0;
	       } else {
		 if (Strcmp(nodeType(n),"namespace") != 0) {
		   Swig_error(cparse_file, cparse_line, "'%s' is not a namespace\n",(yyvsp[-1].str));
		   (yyval.node) = 0;
		 } else {
		   while (Getattr(n,"alias")) {
		     n = Getattr(n,"namespace");
		   }
		   Setattr((yyval.node),"namespace",n);
		   add_symbols((yyval.node));
		   /* Set up a scope alias */
		   Swig_symbol_alias((yyvsp[-3].id),Getattr(n,"symtab"));
		 }
	       }
             }
#line 7911 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 4342 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.node) = (yyvsp[-1].node);
                   /* Insert cpp_member (including any siblings) to the front of the cpp_members linked list */
		   if ((yyval.node)) {
		     Node *p = (yyval.node);
		     Node *pp =0;
		     while (p) {
		       pp = p;
		       p = nextSibling(p);
		     }
		     set_nextSibling(pp,(yyvsp[0].node));
		     if ((yyvsp[0].node))
		       set_previousSibling((yyvsp[0].node), pp);
		   } else {
		     (yyval.node) = (yyvsp[0].node);
		   }
             }
#line 7933 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 4359 "parser.y" /* yacc.c:1646  */
    { 
	       extendmode = 1;
	       if (cplus_mode != CPLUS_PUBLIC) {
		 Swig_error(cparse_file,cparse_line,"%%extend can only be used in a public section\n");
	       }
             }
#line 7944 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 4364 "parser.y" /* yacc.c:1646  */
    {
	       extendmode = 0;
	     }
#line 7952 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 4366 "parser.y" /* yacc.c:1646  */
    {
	       (yyval.node) = new_node("extend");
	       mark_nodes_as_extend((yyvsp[-3].node));
	       appendChild((yyval.node),(yyvsp[-3].node));
	       set_nextSibling((yyval.node),(yyvsp[0].node));
	     }
#line 7963 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 4372 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 7969 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 4373 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0;}
#line 7975 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 4374 "parser.y" /* yacc.c:1646  */
    {
	       int start_line = cparse_line;
	       skip_decl();
	       Swig_error(cparse_file,start_line,"Syntax error in input(3).\n");
	       exit(1);
	       }
#line 7986 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 4379 "parser.y" /* yacc.c:1646  */
    { 
		 (yyval.node) = (yyvsp[0].node);
   	     }
#line 7994 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 4390 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8000 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 4391 "parser.y" /* yacc.c:1646  */
    { 
                 (yyval.node) = (yyvsp[0].node); 
		 if (extendmode && current_class) {
		   String *symname;
		   symname= make_name((yyval.node),Getattr((yyval.node),"name"), Getattr((yyval.node),"decl"));
		   if (Strcmp(symname,Getattr((yyval.node),"name")) == 0) {
		     /* No renaming operation.  Set name to class name */
		     Delete(yyrename);
		     yyrename = NewString(Getattr(current_class,"sym:name"));
		   } else {
		     Delete(yyrename);
		     yyrename = symname;
		   }
		 }
		 add_symbols((yyval.node));
                 default_arguments((yyval.node));
             }
#line 8022 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 4408 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8028 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 4409 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8034 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 4410 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8040 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 4411 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8046 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 4412 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8052 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 4413 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8058 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 4414 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8064 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 4415 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 8070 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 4416 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8076 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 4417 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8082 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 4418 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 8088 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 4419 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8094 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 4420 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8100 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 4421 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 8106 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 4422 "parser.y" /* yacc.c:1646  */
    {(yyval.node) = (yyvsp[0].node); }
#line 8112 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 4423 "parser.y" /* yacc.c:1646  */
    {(yyval.node) = (yyvsp[0].node); }
#line 8118 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 4424 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 8124 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 4433 "parser.y" /* yacc.c:1646  */
    {
              if (inclass || extendmode) {
		SwigType *decl = NewStringEmpty();
		(yyval.node) = new_node("constructor");
		Setattr((yyval.node),"storage",(yyvsp[-5].id));
		Setattr((yyval.node),"name",(yyvsp[-4].type));
		Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		SwigType_add_function(decl,(yyvsp[-2].pl));
		Setattr((yyval.node),"decl",decl);
		Setattr((yyval.node),"throws",(yyvsp[0].decl).throws);
		Setattr((yyval.node),"throw",(yyvsp[0].decl).throwf);
		Setattr((yyval.node),"noexcept",(yyvsp[0].decl).nexcept);
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr((yyval.node),"code",code);
		  Delete(code);
		}
		SetFlag((yyval.node),"feature:new");
		if ((yyvsp[0].decl).defarg)
		  Setattr((yyval.node),"value",(yyvsp[0].decl).defarg);
	      } else {
		(yyval.node) = 0;
              }
              }
#line 8153 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 4461 "parser.y" /* yacc.c:1646  */
    {
               String *name = NewStringf("%s",(yyvsp[-4].str));
	       if (*(Char(name)) != '~') Insert(name,0,"~");
               (yyval.node) = new_node("destructor");
	       Setattr((yyval.node),"name",name);
	       Delete(name);
	       if (Len(scanner_ccode)) {
		 String *code = Copy(scanner_ccode);
		 Setattr((yyval.node),"code",code);
		 Delete(code);
	       }
	       {
		 String *decl = NewStringEmpty();
		 SwigType_add_function(decl,(yyvsp[-2].pl));
		 Setattr((yyval.node),"decl",decl);
		 Delete(decl);
	       }
	       Setattr((yyval.node),"throws",(yyvsp[0].dtype).throws);
	       Setattr((yyval.node),"throw",(yyvsp[0].dtype).throwf);
	       Setattr((yyval.node),"noexcept",(yyvsp[0].dtype).nexcept);
	       if ((yyvsp[0].dtype).val)
	         Setattr((yyval.node),"value",(yyvsp[0].dtype).val);
	       add_symbols((yyval.node));
	      }
#line 8182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 4488 "parser.y" /* yacc.c:1646  */
    {
		String *name;
		(yyval.node) = new_node("destructor");
		Setattr((yyval.node),"storage","virtual");
	        name = NewStringf("%s",(yyvsp[-4].str));
		if (*(Char(name)) != '~') Insert(name,0,"~");
		Setattr((yyval.node),"name",name);
		Delete(name);
		Setattr((yyval.node),"throws",(yyvsp[0].dtype).throws);
		Setattr((yyval.node),"throw",(yyvsp[0].dtype).throwf);
		Setattr((yyval.node),"noexcept",(yyvsp[0].dtype).nexcept);
		if ((yyvsp[0].dtype).val)
		  Setattr((yyval.node),"value",(yyvsp[0].dtype).val);
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr((yyval.node),"code",code);
		  Delete(code);
		}
		{
		  String *decl = NewStringEmpty();
		  SwigType_add_function(decl,(yyvsp[-2].pl));
		  Setattr((yyval.node),"decl",decl);
		  Delete(decl);
		}

		add_symbols((yyval.node));
	      }
#line 8214 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 4519 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = new_node("cdecl");
                 Setattr((yyval.node),"type",(yyvsp[-5].type));
		 Setattr((yyval.node),"name",(yyvsp[-6].str));
		 Setattr((yyval.node),"storage",(yyvsp[-7].id));

		 SwigType_add_function((yyvsp[-4].type),(yyvsp[-2].pl));
		 if ((yyvsp[0].dtype).qualifier) {
		   SwigType_push((yyvsp[-4].type),(yyvsp[0].dtype).qualifier);
		 }
		 Setattr((yyval.node),"decl",(yyvsp[-4].type));
		 Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		 Setattr((yyval.node),"conversion_operator","1");
		 add_symbols((yyval.node));
              }
#line 8234 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 4534 "parser.y" /* yacc.c:1646  */
    {
		 SwigType *decl;
                 (yyval.node) = new_node("cdecl");
                 Setattr((yyval.node),"type",(yyvsp[-5].type));
		 Setattr((yyval.node),"name",(yyvsp[-6].str));
		 Setattr((yyval.node),"storage",(yyvsp[-7].id));
		 decl = NewStringEmpty();
		 SwigType_add_reference(decl);
		 SwigType_add_function(decl,(yyvsp[-2].pl));
		 if ((yyvsp[0].dtype).qualifier) {
		   SwigType_push(decl,(yyvsp[0].dtype).qualifier);
		 }
		 Setattr((yyval.node),"decl",decl);
		 Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		 Setattr((yyval.node),"conversion_operator","1");
		 add_symbols((yyval.node));
	       }
#line 8256 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 4551 "parser.y" /* yacc.c:1646  */
    {
		 SwigType *decl;
                 (yyval.node) = new_node("cdecl");
                 Setattr((yyval.node),"type",(yyvsp[-5].type));
		 Setattr((yyval.node),"name",(yyvsp[-6].str));
		 Setattr((yyval.node),"storage",(yyvsp[-7].id));
		 decl = NewStringEmpty();
		 SwigType_add_rvalue_reference(decl);
		 SwigType_add_function(decl,(yyvsp[-2].pl));
		 if ((yyvsp[0].dtype).qualifier) {
		   SwigType_push(decl,(yyvsp[0].dtype).qualifier);
		 }
		 Setattr((yyval.node),"decl",decl);
		 Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		 Setattr((yyval.node),"conversion_operator","1");
		 add_symbols((yyval.node));
	       }
#line 8278 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 4569 "parser.y" /* yacc.c:1646  */
    {
		 SwigType *decl;
                 (yyval.node) = new_node("cdecl");
                 Setattr((yyval.node),"type",(yyvsp[-6].type));
		 Setattr((yyval.node),"name",(yyvsp[-7].str));
		 Setattr((yyval.node),"storage",(yyvsp[-8].id));
		 decl = NewStringEmpty();
		 SwigType_add_pointer(decl);
		 SwigType_add_reference(decl);
		 SwigType_add_function(decl,(yyvsp[-2].pl));
		 if ((yyvsp[0].dtype).qualifier) {
		   SwigType_push(decl,(yyvsp[0].dtype).qualifier);
		 }
		 Setattr((yyval.node),"decl",decl);
		 Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		 Setattr((yyval.node),"conversion_operator","1");
		 add_symbols((yyval.node));
	       }
#line 8301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 4588 "parser.y" /* yacc.c:1646  */
    {
		String *t = NewStringEmpty();
		(yyval.node) = new_node("cdecl");
		Setattr((yyval.node),"type",(yyvsp[-4].type));
		Setattr((yyval.node),"name",(yyvsp[-5].str));
		 Setattr((yyval.node),"storage",(yyvsp[-6].id));
		SwigType_add_function(t,(yyvsp[-2].pl));
		if ((yyvsp[0].dtype).qualifier) {
		  SwigType_push(t,(yyvsp[0].dtype).qualifier);
		}
		Setattr((yyval.node),"decl",t);
		Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		Setattr((yyval.node),"conversion_operator","1");
		add_symbols((yyval.node));
              }
#line 8321 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 4607 "parser.y" /* yacc.c:1646  */
    {
                 skip_balanced('{','}');
                 (yyval.node) = 0;
               }
#line 8330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 4614 "parser.y" /* yacc.c:1646  */
    {
                skip_balanced('(',')');
                (yyval.node) = 0;
              }
#line 8339 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 4621 "parser.y" /* yacc.c:1646  */
    { 
                (yyval.node) = new_node("access");
		Setattr((yyval.node),"kind","public");
                cplus_mode = CPLUS_PUBLIC;
              }
#line 8349 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 4628 "parser.y" /* yacc.c:1646  */
    { 
                (yyval.node) = new_node("access");
                Setattr((yyval.node),"kind","private");
		cplus_mode = CPLUS_PRIVATE;
	      }
#line 8359 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 4636 "parser.y" /* yacc.c:1646  */
    { 
		(yyval.node) = new_node("access");
		Setattr((yyval.node),"kind","protected");
		cplus_mode = CPLUS_PROTECTED;
	      }
#line 8369 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 4644 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8375 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 4647 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8381 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 4651 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8387 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 4654 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8393 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 4655 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8399 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 4656 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8405 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 4657 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8411 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 4658 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8417 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 4659 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8423 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 4660 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8429 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 4661 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 8435 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 4664 "parser.y" /* yacc.c:1646  */
    {
	            Clear(scanner_ccode);
		    (yyval.dtype).val = 0;
		    (yyval.dtype).throws = (yyvsp[-1].dtype).throws;
		    (yyval.dtype).throwf = (yyvsp[-1].dtype).throwf;
		    (yyval.dtype).nexcept = (yyvsp[-1].dtype).nexcept;
               }
#line 8447 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 4671 "parser.y" /* yacc.c:1646  */
    {
	            Clear(scanner_ccode);
		    (yyval.dtype).val = (yyvsp[-1].dtype).val;
		    (yyval.dtype).throws = (yyvsp[-3].dtype).throws;
		    (yyval.dtype).throwf = (yyvsp[-3].dtype).throwf;
		    (yyval.dtype).nexcept = (yyvsp[-3].dtype).nexcept;
               }
#line 8459 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 4678 "parser.y" /* yacc.c:1646  */
    { 
		    skip_balanced('{','}'); 
		    (yyval.dtype).val = 0;
		    (yyval.dtype).throws = (yyvsp[-1].dtype).throws;
		    (yyval.dtype).throwf = (yyvsp[-1].dtype).throwf;
		    (yyval.dtype).nexcept = (yyvsp[-1].dtype).nexcept;
	       }
#line 8471 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 4687 "parser.y" /* yacc.c:1646  */
    { 
                     Clear(scanner_ccode);
                     (yyval.dtype).val = 0;
                     (yyval.dtype).qualifier = (yyvsp[-1].dtype).qualifier;
                     (yyval.dtype).bitfield = 0;
                     (yyval.dtype).throws = (yyvsp[-1].dtype).throws;
                     (yyval.dtype).throwf = (yyvsp[-1].dtype).throwf;
                     (yyval.dtype).nexcept = (yyvsp[-1].dtype).nexcept;
                }
#line 8485 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 4696 "parser.y" /* yacc.c:1646  */
    { 
                     Clear(scanner_ccode);
                     (yyval.dtype).val = (yyvsp[-1].dtype).val;
                     (yyval.dtype).qualifier = (yyvsp[-3].dtype).qualifier;
                     (yyval.dtype).bitfield = 0;
                     (yyval.dtype).throws = (yyvsp[-3].dtype).throws; 
                     (yyval.dtype).throwf = (yyvsp[-3].dtype).throwf; 
                     (yyval.dtype).nexcept = (yyvsp[-3].dtype).nexcept; 
               }
#line 8499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 4705 "parser.y" /* yacc.c:1646  */
    { 
                     skip_balanced('{','}');
                     (yyval.dtype).val = 0;
                     (yyval.dtype).qualifier = (yyvsp[-1].dtype).qualifier;
                     (yyval.dtype).bitfield = 0;
                     (yyval.dtype).throws = (yyvsp[-1].dtype).throws; 
                     (yyval.dtype).throwf = (yyvsp[-1].dtype).throwf; 
                     (yyval.dtype).nexcept = (yyvsp[-1].dtype).nexcept; 
               }
#line 8513 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 4717 "parser.y" /* yacc.c:1646  */
    { }
#line 8519 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 4720 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type);
                  /* Printf(stdout,"primitive = '%s'\n", $$);*/
                }
#line 8527 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 4723 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 8533 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 4724 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 8539 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 4728 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 8545 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 4730 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.type) = (yyvsp[0].str);
               }
#line 8553 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 4738 "parser.y" /* yacc.c:1646  */
    {
                   if (Strcmp((yyvsp[0].str),"C") == 0) {
		     (yyval.id) = "externc";
                   } else if (Strcmp((yyvsp[0].str),"C++") == 0) {
		     (yyval.id) = "extern";
		   } else {
		     Swig_warning(WARN_PARSE_UNDEFINED_EXTERN,cparse_file, cparse_line,"Unrecognized extern type \"%s\".\n", (yyvsp[0].str));
		     (yyval.id) = 0;
		   }
               }
#line 8568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 4750 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "extern"; }
#line 8574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 4751 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (yyvsp[0].id); }
#line 8580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 4752 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "thread_local"; }
#line 8586 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 4753 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "typedef"; }
#line 8592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 4754 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "static"; }
#line 8598 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 4755 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "typedef"; }
#line 8604 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 4756 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "virtual"; }
#line 8610 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 4757 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "friend"; }
#line 8616 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 4758 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "explicit"; }
#line 8622 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 4759 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "constexpr"; }
#line 8628 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 4760 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "explicit constexpr"; }
#line 8634 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 4761 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "explicit constexpr"; }
#line 8640 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 4762 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "static constexpr"; }
#line 8646 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 4763 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "static constexpr"; }
#line 8652 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 4764 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "thread_local"; }
#line 8658 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 4765 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "static thread_local"; }
#line 8664 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 4766 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "static thread_local"; }
#line 8670 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 4767 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "extern thread_local"; }
#line 8676 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 4768 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "extern thread_local"; }
#line 8682 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 4769 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = 0; }
#line 8688 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 4776 "parser.y" /* yacc.c:1646  */
    {
                 Parm *p;
		 (yyval.pl) = (yyvsp[0].pl);
		 p = (yyvsp[0].pl);
                 while (p) {
		   Replace(Getattr(p,"type"),"typename ", "", DOH_REPLACE_ANY);
		   p = nextSibling(p);
                 }
               }
#line 8702 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 4787 "parser.y" /* yacc.c:1646  */
    {
                  set_nextSibling((yyvsp[-1].p),(yyvsp[0].pl));
                  (yyval.pl) = (yyvsp[-1].p);
		}
#line 8711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 4791 "parser.y" /* yacc.c:1646  */
    { (yyval.pl) = 0; }
#line 8717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 4794 "parser.y" /* yacc.c:1646  */
    {
                 set_nextSibling((yyvsp[-1].p),(yyvsp[0].pl));
		 (yyval.pl) = (yyvsp[-1].p);
                }
#line 8726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 4798 "parser.y" /* yacc.c:1646  */
    { (yyval.pl) = 0; }
#line 8732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 4802 "parser.y" /* yacc.c:1646  */
    {
                   SwigType_push((yyvsp[-1].type),(yyvsp[0].decl).type);
		   (yyval.p) = NewParmWithoutFileLineInfo((yyvsp[-1].type),(yyvsp[0].decl).id);
		   Setfile((yyval.p),cparse_file);
		   Setline((yyval.p),cparse_line);
		   if ((yyvsp[0].decl).defarg) {
		     Setattr((yyval.p),"value",(yyvsp[0].decl).defarg);
		   }
		}
#line 8746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 4812 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.p) = NewParmWithoutFileLineInfo(NewStringf("template<class> %s %s", (yyvsp[-2].id),(yyvsp[-1].str)), 0);
		  Setfile((yyval.p),cparse_file);
		  Setline((yyval.p),cparse_line);
                  if ((yyvsp[0].dtype).val) {
                    Setattr((yyval.p),"value",(yyvsp[0].dtype).val);
                  }
                }
#line 8759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 4820 "parser.y" /* yacc.c:1646  */
    {
		  SwigType *t = NewString("v(...)");
		  (yyval.p) = NewParmWithoutFileLineInfo(t, 0);
		  Setfile((yyval.p),cparse_file);
		  Setline((yyval.p),cparse_line);
		}
#line 8770 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 4828 "parser.y" /* yacc.c:1646  */
    {
                 Parm *p;
		 (yyval.p) = (yyvsp[0].p);
		 p = (yyvsp[0].p);
                 while (p) {
		   if (Getattr(p,"type")) {
		     Replace(Getattr(p,"type"),"typename ", "", DOH_REPLACE_ANY);
		   }
		   p = nextSibling(p);
                 }
               }
#line 8786 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 4841 "parser.y" /* yacc.c:1646  */
    {
                  set_nextSibling((yyvsp[-1].p),(yyvsp[0].p));
                  (yyval.p) = (yyvsp[-1].p);
		}
#line 8795 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 4845 "parser.y" /* yacc.c:1646  */
    { (yyval.p) = 0; }
#line 8801 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 4848 "parser.y" /* yacc.c:1646  */
    {
                 set_nextSibling((yyvsp[-1].p),(yyvsp[0].p));
		 (yyval.p) = (yyvsp[-1].p);
                }
#line 8810 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 4852 "parser.y" /* yacc.c:1646  */
    { (yyval.p) = 0; }
#line 8816 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 4856 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.p) = (yyvsp[0].p);
		  {
		    /* We need to make a possible adjustment for integer parameters. */
		    SwigType *type;
		    Node     *n = 0;

		    while (!n) {
		      type = Getattr((yyvsp[0].p),"type");
		      n = Swig_symbol_clookup(type,0);     /* See if we can find a node that matches the typename */
		      if ((n) && (Strcmp(nodeType(n),"cdecl") == 0)) {
			SwigType *decl = Getattr(n,"decl");
			if (!SwigType_isfunction(decl)) {
			  String *value = Getattr(n,"value");
			  if (value) {
			    String *v = Copy(value);
			    Setattr((yyvsp[0].p),"type",v);
			    Delete(v);
			    n = 0;
			  }
			}
		      } else {
			break;
		      }
		    }
		  }

               }
#line 8849 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 4884 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.p) = NewParmWithoutFileLineInfo(0,0);
                  Setfile((yyval.p),cparse_file);
		  Setline((yyval.p),cparse_line);
		  Setattr((yyval.p),"value",(yyvsp[0].dtype).val);
               }
#line 8860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 4892 "parser.y" /* yacc.c:1646  */
    { 
                  (yyval.dtype) = (yyvsp[0].dtype); 
		  if ((yyvsp[0].dtype).type == T_ERROR) {
		    Swig_warning(WARN_PARSE_BAD_DEFAULT,cparse_file, cparse_line, "Can't set default argument (ignored)\n");
		    (yyval.dtype).val = 0;
		    (yyval.dtype).rawval = 0;
		    (yyval.dtype).bitfield = 0;
		    (yyval.dtype).throws = 0;
		    (yyval.dtype).throwf = 0;
		    (yyval.dtype).nexcept = 0;
		  }
               }
#line 8877 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 4904 "parser.y" /* yacc.c:1646  */
    { 
		  (yyval.dtype) = (yyvsp[-3].dtype);
		  if ((yyvsp[-3].dtype).type == T_ERROR) {
		    Swig_warning(WARN_PARSE_BAD_DEFAULT,cparse_file, cparse_line, "Can't set default argument (ignored)\n");
		    (yyval.dtype) = (yyvsp[-3].dtype);
		    (yyval.dtype).val = 0;
		    (yyval.dtype).rawval = 0;
		    (yyval.dtype).bitfield = 0;
		    (yyval.dtype).throws = 0;
		    (yyval.dtype).throwf = 0;
		    (yyval.dtype).nexcept = 0;
		  } else {
		    (yyval.dtype).val = NewStringf("%s[%s]",(yyvsp[-3].dtype).val,(yyvsp[-1].dtype).val); 
		  }		  
               }
#line 8897 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 4919 "parser.y" /* yacc.c:1646  */
    {
		 skip_balanced('{','}');
		 (yyval.dtype).val = NewString(scanner_ccode);
		 (yyval.dtype).rawval = 0;
                 (yyval.dtype).type = T_INT;
		 (yyval.dtype).bitfield = 0;
		 (yyval.dtype).throws = 0;
		 (yyval.dtype).throwf = 0;
		 (yyval.dtype).nexcept = 0;
	       }
#line 8912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 4929 "parser.y" /* yacc.c:1646  */
    { 
		 (yyval.dtype).val = 0;
		 (yyval.dtype).rawval = 0;
		 (yyval.dtype).type = 0;
		 (yyval.dtype).bitfield = (yyvsp[0].dtype).val;
		 (yyval.dtype).throws = 0;
		 (yyval.dtype).throwf = 0;
		 (yyval.dtype).nexcept = 0;
	       }
#line 8926 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 4938 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype).val = 0;
                 (yyval.dtype).rawval = 0;
                 (yyval.dtype).type = T_INT;
		 (yyval.dtype).bitfield = 0;
		 (yyval.dtype).throws = 0;
		 (yyval.dtype).throwf = 0;
		 (yyval.dtype).nexcept = 0;
               }
#line 8940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 4949 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.decl) = (yyvsp[-1].decl);
		 (yyval.decl).defarg = (yyvsp[0].dtype).rawval ? (yyvsp[0].dtype).rawval : (yyvsp[0].dtype).val;
            }
#line 8949 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 4953 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[-1].decl);
	      (yyval.decl).defarg = (yyvsp[0].dtype).rawval ? (yyvsp[0].dtype).rawval : (yyvsp[0].dtype).val;
            }
#line 8958 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 4957 "parser.y" /* yacc.c:1646  */
    {
   	      (yyval.decl).type = 0;
              (yyval.decl).id = 0;
	      (yyval.decl).defarg = (yyvsp[0].dtype).rawval ? (yyvsp[0].dtype).rawval : (yyvsp[0].dtype).val;
            }
#line 8968 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 4964 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.decl) = (yyvsp[0].decl);
		 if (SwigType_isfunction((yyvsp[0].decl).type)) {
		   Delete(SwigType_pop_function((yyvsp[0].decl).type));
		 } else if (SwigType_isarray((yyvsp[0].decl).type)) {
		   SwigType *ta = SwigType_pop_arrays((yyvsp[0].decl).type);
		   if (SwigType_isfunction((yyvsp[0].decl).type)) {
		     Delete(SwigType_pop_function((yyvsp[0].decl).type));
		   } else {
		     (yyval.decl).parms = 0;
		   }
		   SwigType_push((yyvsp[0].decl).type,ta);
		   Delete(ta);
		 } else {
		   (yyval.decl).parms = 0;
		 }
            }
#line 8990 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 4981 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      if (SwigType_isfunction((yyvsp[0].decl).type)) {
		Delete(SwigType_pop_function((yyvsp[0].decl).type));
	      } else if (SwigType_isarray((yyvsp[0].decl).type)) {
		SwigType *ta = SwigType_pop_arrays((yyvsp[0].decl).type);
		if (SwigType_isfunction((yyvsp[0].decl).type)) {
		  Delete(SwigType_pop_function((yyvsp[0].decl).type));
		} else {
		  (yyval.decl).parms = 0;
		}
		SwigType_push((yyvsp[0].decl).type,ta);
		Delete(ta);
	      } else {
		(yyval.decl).parms = 0;
	      }
            }
#line 9012 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 4998 "parser.y" /* yacc.c:1646  */
    {
   	      (yyval.decl).type = 0;
              (yyval.decl).id = 0;
	      (yyval.decl).parms = 0;
	      }
#line 9022 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 5006 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      if ((yyval.decl).type) {
		SwigType_push((yyvsp[-1].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-1].type);
           }
#line 9035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 5014 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      SwigType_add_reference((yyvsp[-2].type));
              if ((yyval.decl).type) {
		SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-2].type);
           }
#line 9049 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 5023 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      SwigType_add_rvalue_reference((yyvsp[-2].type));
              if ((yyval.decl).type) {
		SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-2].type);
           }
#line 9063 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 5032 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      if (!(yyval.decl).type) (yyval.decl).type = NewStringEmpty();
           }
#line 9072 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 5036 "parser.y" /* yacc.c:1646  */
    {
	     (yyval.decl) = (yyvsp[0].decl);
	     (yyval.decl).type = NewStringEmpty();
	     SwigType_add_reference((yyval.decl).type);
	     if ((yyvsp[0].decl).type) {
	       SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
	       Delete((yyvsp[0].decl).type);
	     }
           }
#line 9086 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 5045 "parser.y" /* yacc.c:1646  */
    {
	     /* Introduced in C++11, move operator && */
             /* Adds one S/R conflict */
	     (yyval.decl) = (yyvsp[0].decl);
	     (yyval.decl).type = NewStringEmpty();
	     SwigType_add_rvalue_reference((yyval.decl).type);
	     if ((yyvsp[0].decl).type) {
	       SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
	       Delete((yyvsp[0].decl).type);
	     }
           }
#line 9102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 5056 "parser.y" /* yacc.c:1646  */
    { 
	     SwigType *t = NewStringEmpty();

	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-2].str));
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = t;
	     }
#line 9118 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 5067 "parser.y" /* yacc.c:1646  */
    { 
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-2].str));
	     SwigType_push((yyvsp[-3].type),t);
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-3].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-3].type);
	     Delete(t);
	   }
#line 9135 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 5079 "parser.y" /* yacc.c:1646  */
    { 
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer((yyvsp[-4].type),(yyvsp[-3].str));
	     SwigType_add_reference((yyvsp[-4].type));
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-4].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-4].type);
	   }
#line 9150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 5089 "parser.y" /* yacc.c:1646  */
    { 
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-3].str));
	     SwigType_add_reference(t);
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     } 
	     (yyval.decl).type = t;
	   }
#line 9166 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 5103 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      if ((yyval.decl).type) {
		SwigType_push((yyvsp[-4].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-4].type);
           }
#line 9179 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 5111 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      SwigType_add_reference((yyvsp[-5].type));
              if ((yyval.decl).type) {
		SwigType_push((yyvsp[-5].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-5].type);
           }
#line 9193 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 5120 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      SwigType_add_rvalue_reference((yyvsp[-5].type));
              if ((yyval.decl).type) {
		SwigType_push((yyvsp[-5].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-5].type);
           }
#line 9207 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 5129 "parser.y" /* yacc.c:1646  */
    {
              (yyval.decl) = (yyvsp[0].decl);
	      if (!(yyval.decl).type) (yyval.decl).type = NewStringEmpty();
           }
#line 9216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 5133 "parser.y" /* yacc.c:1646  */
    {
	     (yyval.decl) = (yyvsp[0].decl);
	     (yyval.decl).type = NewStringEmpty();
	     SwigType_add_reference((yyval.decl).type);
	     if ((yyvsp[0].decl).type) {
	       SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
	       Delete((yyvsp[0].decl).type);
	     }
           }
#line 9230 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 5142 "parser.y" /* yacc.c:1646  */
    {
	     /* Introduced in C++11, move operator && */
             /* Adds one S/R conflict */
	     (yyval.decl) = (yyvsp[0].decl);
	     (yyval.decl).type = NewStringEmpty();
	     SwigType_add_rvalue_reference((yyval.decl).type);
	     if ((yyvsp[0].decl).type) {
	       SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
	       Delete((yyvsp[0].decl).type);
	     }
           }
#line 9246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 5153 "parser.y" /* yacc.c:1646  */
    { 
	     SwigType *t = NewStringEmpty();

	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-5].str));
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = t;
	     }
#line 9262 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 5164 "parser.y" /* yacc.c:1646  */
    { 
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-5].str));
	     SwigType_push((yyvsp[-6].type),t);
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-6].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-6].type);
	     Delete(t);
	   }
#line 9279 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 5176 "parser.y" /* yacc.c:1646  */
    { 
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer((yyvsp[-7].type),(yyvsp[-6].str));
	     SwigType_add_reference((yyvsp[-7].type));
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-7].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-7].type);
	   }
#line 9294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 5186 "parser.y" /* yacc.c:1646  */
    { 
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer((yyvsp[-7].type),(yyvsp[-6].str));
	     SwigType_add_rvalue_reference((yyvsp[-7].type));
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-7].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-7].type);
	   }
#line 9309 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 5196 "parser.y" /* yacc.c:1646  */
    { 
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-6].str));
	     SwigType_add_reference(t);
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     } 
	     (yyval.decl).type = t;
	   }
#line 9325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 5207 "parser.y" /* yacc.c:1646  */
    { 
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-6].str));
	     SwigType_add_rvalue_reference(t);
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     } 
	     (yyval.decl).type = t;
	   }
#line 9341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 5220 "parser.y" /* yacc.c:1646  */
    {
                /* Note: This is non-standard C.  Template declarator is allowed to follow an identifier */
                 (yyval.decl).id = Char((yyvsp[0].str));
		 (yyval.decl).type = 0;
		 (yyval.decl).parms = 0;
		 (yyval.decl).have_parms = 0;
                  }
#line 9353 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 5227 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.decl).id = Char(NewStringf("~%s",(yyvsp[0].str)));
                  (yyval.decl).type = 0;
                  (yyval.decl).parms = 0;
                  (yyval.decl).have_parms = 0;
                  }
#line 9364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 5235 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.decl).id = Char((yyvsp[-1].str));
                  (yyval.decl).type = 0;
                  (yyval.decl).parms = 0;
                  (yyval.decl).have_parms = 0;
                  }
#line 9375 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 5251 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl) = (yyvsp[-1].decl);
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-2].type);
                  }
#line 9388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 5259 "parser.y" /* yacc.c:1646  */
    {
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-1].decl);
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t,(yyvsp[-3].str));
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
		    }
#line 9404 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 5270 "parser.y" /* yacc.c:1646  */
    { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-2].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 5281 "parser.y" /* yacc.c:1646  */
    { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,(yyvsp[-1].dtype).val);
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 5292 "parser.y" /* yacc.c:1646  */
    {
		    SwigType *t;
                    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_function(t,(yyvsp[-1].pl));
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-1].pl);
		      (yyval.decl).have_parms = 1;
		    }
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
		  }
#line 9458 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 5311 "parser.y" /* yacc.c:1646  */
    {
                /* Note: This is non-standard C.  Template declarator is allowed to follow an identifier */
                 (yyval.decl).id = Char((yyvsp[0].str));
		 (yyval.decl).type = 0;
		 (yyval.decl).parms = 0;
		 (yyval.decl).have_parms = 0;
                  }
#line 9470 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 5319 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.decl).id = Char(NewStringf("~%s",(yyvsp[0].str)));
                  (yyval.decl).type = 0;
                  (yyval.decl).parms = 0;
                  (yyval.decl).have_parms = 0;
                  }
#line 9481 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 5336 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl) = (yyvsp[-1].decl);
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-2].type);
                  }
#line 9494 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 5344 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.decl) = (yyvsp[-1].decl);
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = NewStringEmpty();
		    }
		    SwigType_add_reference((yyval.decl).type);
                  }
#line 9506 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 5351 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.decl) = (yyvsp[-1].decl);
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = NewStringEmpty();
		    }
		    SwigType_add_rvalue_reference((yyval.decl).type);
                  }
#line 9518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 5358 "parser.y" /* yacc.c:1646  */
    {
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-1].decl);
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t,(yyvsp[-3].str));
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
		    }
#line 9534 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 5369 "parser.y" /* yacc.c:1646  */
    { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-2].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 5380 "parser.y" /* yacc.c:1646  */
    { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,(yyvsp[-1].dtype).val);
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 5391 "parser.y" /* yacc.c:1646  */
    {
		    SwigType *t;
                    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_function(t,(yyvsp[-1].pl));
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-1].pl);
		      (yyval.decl).have_parms = 1;
		    }
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
                 }
#line 9588 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 5411 "parser.y" /* yacc.c:1646  */
    {
		    SwigType *t;
                    Append((yyvsp[-4].str), " "); /* intervening space is mandatory */
                    Append((yyvsp[-4].str), Char((yyvsp[-3].id)));
		    (yyval.decl).id = Char((yyvsp[-4].str));
		    t = NewStringEmpty();
		    SwigType_add_function(t,(yyvsp[-1].pl));
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-1].pl);
		      (yyval.decl).have_parms = 1;
		    }
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
		  }
#line 9612 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 5432 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl).type = (yyvsp[0].type);
                    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
                  }
#line 9623 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 5438 "parser.y" /* yacc.c:1646  */
    { 
                     (yyval.decl) = (yyvsp[0].decl);
                     SwigType_push((yyvsp[-1].type),(yyvsp[0].decl).type);
		     (yyval.decl).type = (yyvsp[-1].type);
		     Delete((yyvsp[0].decl).type);
                  }
#line 9634 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 5444 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl).type = (yyvsp[-1].type);
		    SwigType_add_reference((yyval.decl).type);
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		  }
#line 9646 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 5451 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl).type = (yyvsp[-1].type);
		    SwigType_add_rvalue_reference((yyval.decl).type);
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		  }
#line 9658 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 5458 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl) = (yyvsp[0].decl);
		    SwigType_add_reference((yyvsp[-2].type));
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-2].type);
                  }
#line 9672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 5467 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl) = (yyvsp[0].decl);
		    SwigType_add_rvalue_reference((yyvsp[-2].type));
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-2].type);
                  }
#line 9686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 5476 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl) = (yyvsp[0].decl);
                  }
#line 9694 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 5479 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl) = (yyvsp[0].decl);
		    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_reference((yyval.decl).type);
		    if ((yyvsp[0].decl).type) {
		      SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
		      Delete((yyvsp[0].decl).type);
		    }
                  }
#line 9708 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 5488 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.decl) = (yyvsp[0].decl);
		    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_rvalue_reference((yyval.decl).type);
		    if ((yyvsp[0].decl).type) {
		      SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
		      Delete((yyvsp[0].decl).type);
		    }
                  }
#line 9722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 5497 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.decl).id = 0;
                    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
                    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_reference((yyval.decl).type);
                  }
#line 9734 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 5504 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.decl).id = 0;
                    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
                    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_rvalue_reference((yyval.decl).type);
                  }
#line 9746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 5511 "parser.y" /* yacc.c:1646  */
    { 
		    (yyval.decl).type = NewStringEmpty();
                    SwigType_add_memberpointer((yyval.decl).type,(yyvsp[-1].str));
                    (yyval.decl).id = 0;
                    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
      	          }
#line 9758 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 5518 "parser.y" /* yacc.c:1646  */
    { 
		    SwigType *t = NewStringEmpty();
                    (yyval.decl).type = (yyvsp[-2].type);
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    SwigType_add_memberpointer(t,(yyvsp[-1].str));
		    SwigType_push((yyval.decl).type,t);
		    Delete(t);
                  }
#line 9773 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 5528 "parser.y" /* yacc.c:1646  */
    { 
		    (yyval.decl) = (yyvsp[0].decl);
		    SwigType_add_memberpointer((yyvsp[-3].type),(yyvsp[-2].str));
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-3].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-3].type);
                  }
#line 9787 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 5539 "parser.y" /* yacc.c:1646  */
    { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-2].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 5550 "parser.y" /* yacc.c:1646  */
    { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,(yyvsp[-1].dtype).val);
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9819 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 5561 "parser.y" /* yacc.c:1646  */
    { 
		    (yyval.decl).type = NewStringEmpty();
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    SwigType_add_array((yyval.decl).type,"");
                  }
#line 9831 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 5568 "parser.y" /* yacc.c:1646  */
    { 
		    (yyval.decl).type = NewStringEmpty();
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    SwigType_add_array((yyval.decl).type,(yyvsp[-1].dtype).val);
		  }
#line 9843 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 5575 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.decl) = (yyvsp[-1].decl);
		  }
#line 9851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 5578 "parser.y" /* yacc.c:1646  */
    {
		    SwigType *t;
                    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
                    SwigType_add_function(t,(yyvsp[-1].pl));
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-1].pl);
		      (yyval.decl).have_parms = 1;
		    }
		  }
#line 9873 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 5595 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.decl).type = NewStringEmpty();
                    SwigType_add_function((yyval.decl).type,(yyvsp[-1].pl));
		    (yyval.decl).parms = (yyvsp[-1].pl);
		    (yyval.decl).have_parms = 1;
		    (yyval.decl).id = 0;
                  }
#line 9885 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 5605 "parser.y" /* yacc.c:1646  */
    { 
             (yyval.type) = NewStringEmpty();
             SwigType_add_pointer((yyval.type));
	     SwigType_push((yyval.type),(yyvsp[-1].str));
	     SwigType_push((yyval.type),(yyvsp[0].type));
	     Delete((yyvsp[0].type));
           }
#line 9897 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 5612 "parser.y" /* yacc.c:1646  */
    {
	     (yyval.type) = NewStringEmpty();
	     SwigType_add_pointer((yyval.type));
	     SwigType_push((yyval.type),(yyvsp[0].type));
	     Delete((yyvsp[0].type));
	   }
#line 9908 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 5618 "parser.y" /* yacc.c:1646  */
    { 
	     (yyval.type) = NewStringEmpty();
	     SwigType_add_pointer((yyval.type));
	     SwigType_push((yyval.type),(yyvsp[0].str));
           }
#line 9918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 5623 "parser.y" /* yacc.c:1646  */
    {
	     (yyval.type) = NewStringEmpty();
	     SwigType_add_pointer((yyval.type));
           }
#line 9927 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 5629 "parser.y" /* yacc.c:1646  */
    {
	          (yyval.str) = NewStringEmpty();
	          if ((yyvsp[0].id)) SwigType_add_qualifier((yyval.str),(yyvsp[0].id));
               }
#line 9936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 5633 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.str) = (yyvsp[0].str);
	          if ((yyvsp[-1].id)) SwigType_add_qualifier((yyval.str),(yyvsp[-1].id));
               }
#line 9945 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 5639 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "const"; }
#line 9951 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 5640 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = "volatile"; }
#line 9957 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 5641 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = 0; }
#line 9963 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 5647 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.type) = (yyvsp[0].type);
                   Replace((yyval.type),"typename ","", DOH_REPLACE_ANY);
                }
#line 9972 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 5653 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.type) = (yyvsp[0].type);
	           SwigType_push((yyval.type),(yyvsp[-1].str));
               }
#line 9981 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 5657 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 9987 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 5658 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.type) = (yyvsp[-1].type);
	          SwigType_push((yyval.type),(yyvsp[0].str));
	       }
#line 9996 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 5662 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.type) = (yyvsp[-1].type);
	          SwigType_push((yyval.type),(yyvsp[0].str));
	          SwigType_push((yyval.type),(yyvsp[-2].str));
	       }
#line 10006 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 5669 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type);
                  /* Printf(stdout,"primitive = '%s'\n", $$);*/
               }
#line 10014 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 5672 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 10020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 5673 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 10026 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 5677 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = NewStringf("enum %s", (yyvsp[0].str)); }
#line 10032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 5678 "parser.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 10038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 5680 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.type) = (yyvsp[0].str);
               }
#line 10046 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 5683 "parser.y" /* yacc.c:1646  */
    { 
		 (yyval.type) = NewStringf("%s %s", (yyvsp[-1].id), (yyvsp[0].str));
               }
#line 10054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 5686 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.type) = (yyvsp[0].type);
               }
#line 10062 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 5691 "parser.y" /* yacc.c:1646  */
    {
                 Node *n = Swig_symbol_clookup((yyvsp[-1].str),0);
                 if (!n) {
		   Swig_error(cparse_file, cparse_line, "Identifier %s not defined.\n", (yyvsp[-1].str));
                   (yyval.type) = (yyvsp[-1].str);
                 } else {
                   (yyval.type) = Getattr(n, "type");
                 }
               }
#line 10076 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 5702 "parser.y" /* yacc.c:1646  */
    {
		 if (!(yyvsp[0].ptype).type) (yyvsp[0].ptype).type = NewString("int");
		 if ((yyvsp[0].ptype).us) {
		   (yyval.type) = NewStringf("%s %s", (yyvsp[0].ptype).us, (yyvsp[0].ptype).type);
		   Delete((yyvsp[0].ptype).us);
                   Delete((yyvsp[0].ptype).type);
		 } else {
                   (yyval.type) = (yyvsp[0].ptype).type;
		 }
		 if (Cmp((yyval.type),"signed int") == 0) {
		   Delete((yyval.type));
		   (yyval.type) = NewString("int");
                 } else if (Cmp((yyval.type),"signed long") == 0) {
		   Delete((yyval.type));
                   (yyval.type) = NewString("long");
                 } else if (Cmp((yyval.type),"signed short") == 0) {
		   Delete((yyval.type));
		   (yyval.type) = NewString("short");
		 } else if (Cmp((yyval.type),"signed long long") == 0) {
		   Delete((yyval.type));
		   (yyval.type) = NewString("long long");
		 }
               }
#line 10104 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 5727 "parser.y" /* yacc.c:1646  */
    { 
                 (yyval.ptype) = (yyvsp[0].ptype);
               }
#line 10112 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 5730 "parser.y" /* yacc.c:1646  */
    {
                    if ((yyvsp[-1].ptype).us && (yyvsp[0].ptype).us) {
		      Swig_error(cparse_file, cparse_line, "Extra %s specifier.\n", (yyvsp[0].ptype).us);
		    }
                    (yyval.ptype) = (yyvsp[0].ptype);
                    if ((yyvsp[-1].ptype).us) (yyval.ptype).us = (yyvsp[-1].ptype).us;
		    if ((yyvsp[-1].ptype).type) {
		      if (!(yyvsp[0].ptype).type) (yyval.ptype).type = (yyvsp[-1].ptype).type;
		      else {
			int err = 0;
			if ((Cmp((yyvsp[-1].ptype).type,"long") == 0)) {
			  if ((Cmp((yyvsp[0].ptype).type,"long") == 0) || (Strncmp((yyvsp[0].ptype).type,"double",6) == 0)) {
			    (yyval.ptype).type = NewStringf("long %s", (yyvsp[0].ptype).type);
			  } else if (Cmp((yyvsp[0].ptype).type,"int") == 0) {
			    (yyval.ptype).type = (yyvsp[-1].ptype).type;
			  } else {
			    err = 1;
			  }
			} else if ((Cmp((yyvsp[-1].ptype).type,"short")) == 0) {
			  if (Cmp((yyvsp[0].ptype).type,"int") == 0) {
			    (yyval.ptype).type = (yyvsp[-1].ptype).type;
			  } else {
			    err = 1;
			  }
			} else if (Cmp((yyvsp[-1].ptype).type,"int") == 0) {
			  (yyval.ptype).type = (yyvsp[0].ptype).type;
			} else if (Cmp((yyvsp[-1].ptype).type,"double") == 0) {
			  if (Cmp((yyvsp[0].ptype).type,"long") == 0) {
			    (yyval.ptype).type = NewString("long double");
			  } else if (Cmp((yyvsp[0].ptype).type,"complex") == 0) {
			    (yyval.ptype).type = NewString("double complex");
			  } else {
			    err = 1;
			  }
			} else if (Cmp((yyvsp[-1].ptype).type,"float") == 0) {
			  if (Cmp((yyvsp[0].ptype).type,"complex") == 0) {
			    (yyval.ptype).type = NewString("float complex");
			  } else {
			    err = 1;
			  }
			} else if (Cmp((yyvsp[-1].ptype).type,"complex") == 0) {
			  (yyval.ptype).type = NewStringf("%s complex", (yyvsp[0].ptype).type);
			} else {
			  err = 1;
			}
			if (err) {
			  Swig_error(cparse_file, cparse_line, "Extra %s specifier.\n", (yyvsp[-1].ptype).type);
			}
		      }
		    }
               }
#line 10168 "y.tab.c" /* yacc.c:1646  */
    break;

  case 389:
#line 5784 "parser.y" /* yacc.c:1646  */
    { 
		    (yyval.ptype).type = NewString("int");
                    (yyval.ptype).us = 0;
               }
#line 10177 "y.tab.c" /* yacc.c:1646  */
    break;

  case 390:
#line 5788 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("short");
                    (yyval.ptype).us = 0;
                }
#line 10186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 391:
#line 5792 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("long");
                    (yyval.ptype).us = 0;
                }
#line 10195 "y.tab.c" /* yacc.c:1646  */
    break;

  case 392:
#line 5796 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("char");
                    (yyval.ptype).us = 0;
                }
#line 10204 "y.tab.c" /* yacc.c:1646  */
    break;

  case 393:
#line 5800 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("wchar_t");
                    (yyval.ptype).us = 0;
                }
#line 10213 "y.tab.c" /* yacc.c:1646  */
    break;

  case 394:
#line 5804 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("float");
                    (yyval.ptype).us = 0;
                }
#line 10222 "y.tab.c" /* yacc.c:1646  */
    break;

  case 395:
#line 5808 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("double");
                    (yyval.ptype).us = 0;
                }
#line 10231 "y.tab.c" /* yacc.c:1646  */
    break;

  case 396:
#line 5812 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).us = NewString("signed");
                    (yyval.ptype).type = 0;
                }
#line 10240 "y.tab.c" /* yacc.c:1646  */
    break;

  case 397:
#line 5816 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).us = NewString("unsigned");
                    (yyval.ptype).type = 0;
                }
#line 10249 "y.tab.c" /* yacc.c:1646  */
    break;

  case 398:
#line 5820 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("complex");
                    (yyval.ptype).us = 0;
                }
#line 10258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 399:
#line 5824 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("__int8");
                    (yyval.ptype).us = 0;
                }
#line 10267 "y.tab.c" /* yacc.c:1646  */
    break;

  case 400:
#line 5828 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("__int16");
                    (yyval.ptype).us = 0;
                }
#line 10276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 401:
#line 5832 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("__int32");
                    (yyval.ptype).us = 0;
                }
#line 10285 "y.tab.c" /* yacc.c:1646  */
    break;

  case 402:
#line 5836 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.ptype).type = NewString("__int64");
                    (yyval.ptype).us = 0;
                }
#line 10294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 403:
#line 5842 "parser.y" /* yacc.c:1646  */
    { /* scanner_check_typedef(); */ }
#line 10300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 404:
#line 5842 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.dtype) = (yyvsp[0].dtype);
		   if ((yyval.dtype).type == T_STRING) {
		     (yyval.dtype).rawval = NewStringf("\"%(escape)s\"",(yyval.dtype).val);
		   } else if ((yyval.dtype).type != T_CHAR && (yyval.dtype).type != T_WSTRING && (yyval.dtype).type != T_WCHAR) {
		     (yyval.dtype).rawval = NewStringf("%s", (yyval.dtype).val);
		   }
		   (yyval.dtype).qualifier = 0;
		   (yyval.dtype).bitfield = 0;
		   (yyval.dtype).throws = 0;
		   (yyval.dtype).throwf = 0;
		   (yyval.dtype).nexcept = 0;
		   scanner_ignore_typedef();
                }
#line 10319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 405:
#line 5856 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.dtype) = (yyvsp[0].dtype);
		}
#line 10327 "y.tab.c" /* yacc.c:1646  */
    break;

  case 406:
#line 5861 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.dtype) = (yyvsp[0].dtype);
		}
#line 10335 "y.tab.c" /* yacc.c:1646  */
    break;

  case 407:
#line 5864 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.dtype) = (yyvsp[0].dtype);
		}
#line 10343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 408:
#line 5870 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.dtype).val = NewString("delete");
		  (yyval.dtype).rawval = 0;
		  (yyval.dtype).type = T_STRING;
		  (yyval.dtype).qualifier = 0;
		  (yyval.dtype).bitfield = 0;
		  (yyval.dtype).throws = 0;
		  (yyval.dtype).throwf = 0;
		  (yyval.dtype).nexcept = 0;
		}
#line 10358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 409:
#line 5883 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.dtype).val = NewString("default");
		  (yyval.dtype).rawval = 0;
		  (yyval.dtype).type = T_STRING;
		  (yyval.dtype).qualifier = 0;
		  (yyval.dtype).bitfield = 0;
		  (yyval.dtype).throws = 0;
		  (yyval.dtype).throwf = 0;
		  (yyval.dtype).nexcept = 0;
		}
#line 10373 "y.tab.c" /* yacc.c:1646  */
    break;

  case 410:
#line 5897 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (yyvsp[0].id); }
#line 10379 "y.tab.c" /* yacc.c:1646  */
    break;

  case 411:
#line 5898 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (char *) 0;}
#line 10385 "y.tab.c" /* yacc.c:1646  */
    break;

  case 412:
#line 5901 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = (yyvsp[0].node); }
#line 10391 "y.tab.c" /* yacc.c:1646  */
    break;

  case 413:
#line 5902 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 10397 "y.tab.c" /* yacc.c:1646  */
    break;

  case 414:
#line 5906 "parser.y" /* yacc.c:1646  */
    {
		 Node *leftSibling = Getattr((yyvsp[-4].node),"_last");
		 set_nextSibling(leftSibling,(yyvsp[-1].node));
		 Setattr((yyvsp[-4].node),"_last",(yyvsp[-1].node));
		 (yyval.node) = (yyvsp[-4].node);
	       }
#line 10408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 415:
#line 5912 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = (yyvsp[-2].node);
	       }
#line 10416 "y.tab.c" /* yacc.c:1646  */
    break;

  case 416:
#line 5915 "parser.y" /* yacc.c:1646  */
    {
		 Setattr((yyvsp[-1].node),"_last",(yyvsp[-1].node));
		 (yyval.node) = (yyvsp[-1].node);
	       }
#line 10425 "y.tab.c" /* yacc.c:1646  */
    break;

  case 417:
#line 5919 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = 0;
	       }
#line 10433 "y.tab.c" /* yacc.c:1646  */
    break;

  case 418:
#line 5924 "parser.y" /* yacc.c:1646  */
    {
		   SwigType *type = NewSwigType(T_INT);
		   (yyval.node) = new_node("enumitem");
		   Setattr((yyval.node),"name",(yyvsp[0].id));
		   Setattr((yyval.node),"type",type);
		   SetFlag((yyval.node),"feature:immutable");
		   Delete(type);
		 }
#line 10446 "y.tab.c" /* yacc.c:1646  */
    break;

  case 419:
#line 5932 "parser.y" /* yacc.c:1646  */
    {
		   SwigType *type = NewSwigType((yyvsp[0].dtype).type == T_BOOL ? T_BOOL : ((yyvsp[0].dtype).type == T_CHAR ? T_CHAR : T_INT));
		   (yyval.node) = new_node("enumitem");
		   Setattr((yyval.node),"name",(yyvsp[-2].id));
		   Setattr((yyval.node),"type",type);
		   SetFlag((yyval.node),"feature:immutable");
		   Setattr((yyval.node),"enumvalue", (yyvsp[0].dtype).val);
		   Setattr((yyval.node),"value",(yyvsp[-2].id));
		   Delete(type);
                 }
#line 10461 "y.tab.c" /* yacc.c:1646  */
    break;

  case 420:
#line 5944 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.dtype) = (yyvsp[0].dtype);
		   if (((yyval.dtype).type != T_INT) && ((yyval.dtype).type != T_UINT) &&
		       ((yyval.dtype).type != T_LONG) && ((yyval.dtype).type != T_ULONG) &&
		       ((yyval.dtype).type != T_LONGLONG) && ((yyval.dtype).type != T_ULONGLONG) &&
		       ((yyval.dtype).type != T_SHORT) && ((yyval.dtype).type != T_USHORT) &&
		       ((yyval.dtype).type != T_SCHAR) && ((yyval.dtype).type != T_UCHAR) &&
		       ((yyval.dtype).type != T_CHAR) && ((yyval.dtype).type != T_BOOL)) {
		     Swig_error(cparse_file,cparse_line,"Type error. Expecting an integral type\n");
		   }
                }
#line 10477 "y.tab.c" /* yacc.c:1646  */
    break;

  case 421:
#line 5959 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10483 "y.tab.c" /* yacc.c:1646  */
    break;

  case 422:
#line 5960 "parser.y" /* yacc.c:1646  */
    {
		 Node *n;
		 (yyval.dtype).val = (yyvsp[0].type);
		 (yyval.dtype).type = T_INT;
		 /* Check if value is in scope */
		 n = Swig_symbol_clookup((yyvsp[0].type),0);
		 if (n) {
                   /* A band-aid for enum values used in expressions. */
                   if (Strcmp(nodeType(n),"enumitem") == 0) {
                     String *q = Swig_symbol_qualified(n);
                     if (q) {
                       (yyval.dtype).val = NewStringf("%s::%s", q, Getattr(n,"name"));
                       Delete(q);
                     }
                   }
		 }
               }
#line 10505 "y.tab.c" /* yacc.c:1646  */
    break;

  case 423:
#line 5979 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10511 "y.tab.c" /* yacc.c:1646  */
    break;

  case 424:
#line 5980 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.dtype).val = (yyvsp[0].str);
                    (yyval.dtype).type = T_STRING;
               }
#line 10520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 425:
#line 5984 "parser.y" /* yacc.c:1646  */
    {
		  SwigType_push((yyvsp[-2].type),(yyvsp[-1].decl).type);
		  (yyval.dtype).val = NewStringf("sizeof(%s)",SwigType_str((yyvsp[-2].type),0));
		  (yyval.dtype).type = T_ULONG;
               }
#line 10530 "y.tab.c" /* yacc.c:1646  */
    break;

  case 426:
#line 5989 "parser.y" /* yacc.c:1646  */
    {
		  SwigType_push((yyvsp[-2].type),(yyvsp[-1].decl).type);
		  (yyval.dtype).val = NewStringf("sizeof...(%s)",SwigType_str((yyvsp[-2].type),0));
		  (yyval.dtype).type = T_ULONG;
               }
#line 10540 "y.tab.c" /* yacc.c:1646  */
    break;

  case 427:
#line 5994 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10546 "y.tab.c" /* yacc.c:1646  */
    break;

  case 428:
#line 5995 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.dtype).val = (yyvsp[0].str);
		    (yyval.dtype).rawval = NewStringf("L\"%s\"", (yyval.dtype).val);
                    (yyval.dtype).type = T_WSTRING;
	       }
#line 10556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 429:
#line 6000 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.dtype).val = NewString((yyvsp[0].str));
		  if (Len((yyval.dtype).val)) {
		    (yyval.dtype).rawval = NewStringf("'%(escape)s'", (yyval.dtype).val);
		  } else {
		    (yyval.dtype).rawval = NewString("'\\0'");
		  }
		  (yyval.dtype).type = T_CHAR;
		  (yyval.dtype).bitfield = 0;
		  (yyval.dtype).throws = 0;
		  (yyval.dtype).throwf = 0;
		  (yyval.dtype).nexcept = 0;
	       }
#line 10574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 430:
#line 6013 "parser.y" /* yacc.c:1646  */
    {
		  (yyval.dtype).val = NewString((yyvsp[0].str));
		  if (Len((yyval.dtype).val)) {
		    (yyval.dtype).rawval = NewStringf("L\'%s\'", (yyval.dtype).val);
		  } else {
		    (yyval.dtype).rawval = NewString("L'\\0'");
		  }
		  (yyval.dtype).type = T_WCHAR;
		  (yyval.dtype).bitfield = 0;
		  (yyval.dtype).throws = 0;
		  (yyval.dtype).throwf = 0;
		  (yyval.dtype).nexcept = 0;
	       }
#line 10592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 431:
#line 6028 "parser.y" /* yacc.c:1646  */
    {
   	            (yyval.dtype).val = NewStringf("(%s)",(yyvsp[-1].dtype).val);
		    (yyval.dtype).type = (yyvsp[-1].dtype).type;
   	       }
#line 10601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 432:
#line 6035 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   switch ((yyvsp[-2].dtype).type) {
		     case T_FLOAT:
		     case T_DOUBLE:
		     case T_LONGDOUBLE:
		     case T_FLTCPLX:
		     case T_DBLCPLX:
		       (yyval.dtype).val = NewStringf("(%s)%s", (yyvsp[-2].dtype).val, (yyvsp[0].dtype).val); /* SwigType_str and decimal points don't mix! */
		       break;
		     default:
		       (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-2].dtype).val,0), (yyvsp[0].dtype).val);
		       break;
		   }
		 }
 	       }
#line 10623 "y.tab.c" /* yacc.c:1646  */
    break;

  case 433:
#line 6052 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_push((yyvsp[-3].dtype).val,(yyvsp[-2].type));
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-3].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 10635 "y.tab.c" /* yacc.c:1646  */
    break;

  case 434:
#line 6059 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_add_reference((yyvsp[-3].dtype).val);
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-3].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 10647 "y.tab.c" /* yacc.c:1646  */
    break;

  case 435:
#line 6066 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_add_rvalue_reference((yyvsp[-3].dtype).val);
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-3].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 10659 "y.tab.c" /* yacc.c:1646  */
    break;

  case 436:
#line 6073 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_push((yyvsp[-4].dtype).val,(yyvsp[-3].type));
		   SwigType_add_reference((yyvsp[-4].dtype).val);
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-4].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 10672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 437:
#line 6081 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_push((yyvsp[-4].dtype).val,(yyvsp[-3].type));
		   SwigType_add_rvalue_reference((yyvsp[-4].dtype).val);
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-4].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 10685 "y.tab.c" /* yacc.c:1646  */
    break;

  case 438:
#line 6089 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype) = (yyvsp[0].dtype);
                 (yyval.dtype).val = NewStringf("&%s",(yyvsp[0].dtype).val);
	       }
#line 10694 "y.tab.c" /* yacc.c:1646  */
    break;

  case 439:
#line 6093 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype) = (yyvsp[0].dtype);
                 (yyval.dtype).val = NewStringf("&&%s",(yyvsp[0].dtype).val);
	       }
#line 10703 "y.tab.c" /* yacc.c:1646  */
    break;

  case 440:
#line 6097 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype) = (yyvsp[0].dtype);
                 (yyval.dtype).val = NewStringf("*%s",(yyvsp[0].dtype).val);
	       }
#line 10712 "y.tab.c" /* yacc.c:1646  */
    break;

  case 441:
#line 6103 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10718 "y.tab.c" /* yacc.c:1646  */
    break;

  case 442:
#line 6104 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10724 "y.tab.c" /* yacc.c:1646  */
    break;

  case 443:
#line 6105 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10730 "y.tab.c" /* yacc.c:1646  */
    break;

  case 444:
#line 6106 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10736 "y.tab.c" /* yacc.c:1646  */
    break;

  case 445:
#line 6107 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10742 "y.tab.c" /* yacc.c:1646  */
    break;

  case 446:
#line 6108 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10748 "y.tab.c" /* yacc.c:1646  */
    break;

  case 447:
#line 6109 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10754 "y.tab.c" /* yacc.c:1646  */
    break;

  case 448:
#line 6110 "parser.y" /* yacc.c:1646  */
    { (yyval.dtype) = (yyvsp[0].dtype); }
#line 10760 "y.tab.c" /* yacc.c:1646  */
    break;

  case 449:
#line 6113 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s+%s", COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 10769 "y.tab.c" /* yacc.c:1646  */
    break;

  case 450:
#line 6117 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s-%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 10778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 451:
#line 6121 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s*%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 10787 "y.tab.c" /* yacc.c:1646  */
    break;

  case 452:
#line 6125 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s/%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 10796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 453:
#line 6129 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s%%%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 10805 "y.tab.c" /* yacc.c:1646  */
    break;

  case 454:
#line 6133 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s&%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 10814 "y.tab.c" /* yacc.c:1646  */
    break;

  case 455:
#line 6137 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s|%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 10823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 456:
#line 6141 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s^%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 10832 "y.tab.c" /* yacc.c:1646  */
    break;

  case 457:
#line 6145 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s << %s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote_type((yyvsp[-2].dtype).type);
	       }
#line 10841 "y.tab.c" /* yacc.c:1646  */
    break;

  case 458:
#line 6149 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s >> %s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote_type((yyvsp[-2].dtype).type);
	       }
#line 10850 "y.tab.c" /* yacc.c:1646  */
    break;

  case 459:
#line 6153 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s&&%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 10859 "y.tab.c" /* yacc.c:1646  */
    break;

  case 460:
#line 6157 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s||%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 10868 "y.tab.c" /* yacc.c:1646  */
    break;

  case 461:
#line 6161 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s==%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 10877 "y.tab.c" /* yacc.c:1646  */
    break;

  case 462:
#line 6165 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s!=%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 10886 "y.tab.c" /* yacc.c:1646  */
    break;

  case 463:
#line 6179 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s >= %s", COMPOUND_EXPR_VAL((yyvsp[-2].dtype)), COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 10895 "y.tab.c" /* yacc.c:1646  */
    break;

  case 464:
#line 6183 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s <= %s", COMPOUND_EXPR_VAL((yyvsp[-2].dtype)), COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 10904 "y.tab.c" /* yacc.c:1646  */
    break;

  case 465:
#line 6187 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("%s?%s:%s", COMPOUND_EXPR_VAL((yyvsp[-4].dtype)), COMPOUND_EXPR_VAL((yyvsp[-2].dtype)), COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 /* This may not be exactly right, but is probably good enough
		  * for the purposes of parsing constant expressions. */
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type, (yyvsp[0].dtype).type);
	       }
#line 10915 "y.tab.c" /* yacc.c:1646  */
    break;

  case 466:
#line 6193 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("-%s",(yyvsp[0].dtype).val);
		 (yyval.dtype).type = (yyvsp[0].dtype).type;
	       }
#line 10924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 467:
#line 6197 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype).val = NewStringf("+%s",(yyvsp[0].dtype).val);
		 (yyval.dtype).type = (yyvsp[0].dtype).type;
	       }
#line 10933 "y.tab.c" /* yacc.c:1646  */
    break;

  case 468:
#line 6201 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.dtype).val = NewStringf("~%s",(yyvsp[0].dtype).val);
		 (yyval.dtype).type = (yyvsp[0].dtype).type;
	       }
#line 10942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 469:
#line 6205 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.dtype).val = NewStringf("!%s",COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = T_INT;
	       }
#line 10951 "y.tab.c" /* yacc.c:1646  */
    break;

  case 470:
#line 6209 "parser.y" /* yacc.c:1646  */
    {
		 String *qty;
                 skip_balanced('(',')');
		 qty = Swig_symbol_type_qualify((yyvsp[-1].type),0);
		 if (SwigType_istemplate(qty)) {
		   String *nstr = SwigType_namestr(qty);
		   Delete(qty);
		   qty = nstr;
		 }
		 (yyval.dtype).val = NewStringf("%s%s",qty,scanner_ccode);
		 Clear(scanner_ccode);
		 (yyval.dtype).type = T_INT;
		 Delete(qty);
               }
#line 10970 "y.tab.c" /* yacc.c:1646  */
    break;

  case 471:
#line 6225 "parser.y" /* yacc.c:1646  */
    {
	        (yyval.str) = NewString("...");
	      }
#line 10978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 472:
#line 6230 "parser.y" /* yacc.c:1646  */
    {
	        (yyval.str) = (yyvsp[0].str);
	      }
#line 10986 "y.tab.c" /* yacc.c:1646  */
    break;

  case 473:
#line 6233 "parser.y" /* yacc.c:1646  */
    {
	        (yyval.str) = 0;
	      }
#line 10994 "y.tab.c" /* yacc.c:1646  */
    break;

  case 474:
#line 6238 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.bases) = (yyvsp[0].bases);
               }
#line 11002 "y.tab.c" /* yacc.c:1646  */
    break;

  case 475:
#line 6243 "parser.y" /* yacc.c:1646  */
    { inherit_list = 1; }
#line 11008 "y.tab.c" /* yacc.c:1646  */
    break;

  case 476:
#line 6243 "parser.y" /* yacc.c:1646  */
    { (yyval.bases) = (yyvsp[0].bases); inherit_list = 0; }
#line 11014 "y.tab.c" /* yacc.c:1646  */
    break;

  case 477:
#line 6244 "parser.y" /* yacc.c:1646  */
    { (yyval.bases) = 0; }
#line 11020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 478:
#line 6247 "parser.y" /* yacc.c:1646  */
    {
		   Hash *list = NewHash();
		   Node *base = (yyvsp[0].node);
		   Node *name = Getattr(base,"name");
		   List *lpublic = NewList();
		   List *lprotected = NewList();
		   List *lprivate = NewList();
		   Setattr(list,"public",lpublic);
		   Setattr(list,"protected",lprotected);
		   Setattr(list,"private",lprivate);
		   Delete(lpublic);
		   Delete(lprotected);
		   Delete(lprivate);
		   Append(Getattr(list,Getattr(base,"access")),name);
	           (yyval.bases) = list;
               }
#line 11041 "y.tab.c" /* yacc.c:1646  */
    break;

  case 479:
#line 6264 "parser.y" /* yacc.c:1646  */
    {
		   Hash *list = (yyvsp[-2].bases);
		   Node *base = (yyvsp[0].node);
		   Node *name = Getattr(base,"name");
		   Append(Getattr(list,Getattr(base,"access")),name);
                   (yyval.bases) = list;
               }
#line 11053 "y.tab.c" /* yacc.c:1646  */
    break;

  case 480:
#line 6273 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.intvalue) = cparse_line;
	       }
#line 11061 "y.tab.c" /* yacc.c:1646  */
    break;

  case 481:
#line 6275 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = NewHash();
		 Setfile((yyval.node),cparse_file);
		 Setline((yyval.node),(yyvsp[-2].intvalue));
		 Setattr((yyval.node),"name",(yyvsp[-1].str));
		 Setfile((yyvsp[-1].str),cparse_file);
		 Setline((yyvsp[-1].str),(yyvsp[-2].intvalue));
                 if (last_cpptype && (Strcmp(last_cpptype,"struct") != 0)) {
		   Setattr((yyval.node),"access","private");
		   Swig_warning(WARN_PARSE_NO_ACCESS, Getfile((yyval.node)), Getline((yyval.node)), "No access specifier given for base class '%s' (ignored).\n", SwigType_namestr((yyvsp[-1].str)));
                 } else {
		   Setattr((yyval.node),"access","public");
		 }
		 if ((yyvsp[0].str))
		   SetFlag((yyval.node), "variadic");
               }
#line 11082 "y.tab.c" /* yacc.c:1646  */
    break;

  case 482:
#line 6291 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.intvalue) = cparse_line;
	       }
#line 11090 "y.tab.c" /* yacc.c:1646  */
    break;

  case 483:
#line 6293 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = NewHash();
		 Setfile((yyval.node),cparse_file);
		 Setline((yyval.node),(yyvsp[-3].intvalue));
		 Setattr((yyval.node),"name",(yyvsp[-1].str));
		 Setfile((yyvsp[-1].str),cparse_file);
		 Setline((yyvsp[-1].str),(yyvsp[-3].intvalue));
		 Setattr((yyval.node),"access",(yyvsp[-4].id));
	         if (Strcmp((yyvsp[-4].id),"public") != 0) {
		   Swig_warning(WARN_PARSE_PRIVATE_INHERIT, Getfile((yyval.node)), Getline((yyval.node)), "%s inheritance from base '%s' (ignored).\n", (yyvsp[-4].id), SwigType_namestr((yyvsp[-1].str)));
		 }
		 if ((yyvsp[0].str))
		   SetFlag((yyval.node), "variadic");
               }
#line 11109 "y.tab.c" /* yacc.c:1646  */
    break;

  case 484:
#line 6309 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (char*)"public"; }
#line 11115 "y.tab.c" /* yacc.c:1646  */
    break;

  case 485:
#line 6310 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (char*)"private"; }
#line 11121 "y.tab.c" /* yacc.c:1646  */
    break;

  case 486:
#line 6311 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (char*)"protected"; }
#line 11127 "y.tab.c" /* yacc.c:1646  */
    break;

  case 487:
#line 6314 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.id) = (char*)"class"; 
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11136 "y.tab.c" /* yacc.c:1646  */
    break;

  case 488:
#line 6318 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.id) = (char *)"typename"; 
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11145 "y.tab.c" /* yacc.c:1646  */
    break;

  case 489:
#line 6322 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.id) = (char *)"class..."; 
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11154 "y.tab.c" /* yacc.c:1646  */
    break;

  case 490:
#line 6326 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.id) = (char *)"typename..."; 
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11163 "y.tab.c" /* yacc.c:1646  */
    break;

  case 491:
#line 6332 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.id) = (yyvsp[0].id);
               }
#line 11171 "y.tab.c" /* yacc.c:1646  */
    break;

  case 492:
#line 6335 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.id) = (char*)"struct"; 
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11180 "y.tab.c" /* yacc.c:1646  */
    break;

  case 493:
#line 6339 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.id) = (char*)"union"; 
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11189 "y.tab.c" /* yacc.c:1646  */
    break;

  case 494:
#line 6345 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.id) = (char*)"class";
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11198 "y.tab.c" /* yacc.c:1646  */
    break;

  case 495:
#line 6349 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.id) = (char*)"struct";
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11207 "y.tab.c" /* yacc.c:1646  */
    break;

  case 496:
#line 6353 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.id) = (char*)"union";
		   if (!inherit_list) last_cpptype = (yyval.id);
               }
#line 11216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 497:
#line 6359 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.id) = (yyvsp[0].id);
               }
#line 11224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 498:
#line 6362 "parser.y" /* yacc.c:1646  */
    {
		   (yyval.id) = 0;
               }
#line 11232 "y.tab.c" /* yacc.c:1646  */
    break;

  case 501:
#line 6371 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = 0;
	       }
#line 11240 "y.tab.c" /* yacc.c:1646  */
    break;

  case 502:
#line 6374 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = 0;
	       }
#line 11248 "y.tab.c" /* yacc.c:1646  */
    break;

  case 503:
#line 6377 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = 0;
	       }
#line 11256 "y.tab.c" /* yacc.c:1646  */
    break;

  case 504:
#line 6380 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = 0;
	       }
#line 11264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 505:
#line 6385 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.dtype).throws = (yyvsp[-1].pl);
                    (yyval.dtype).throwf = NewString("1");
                    (yyval.dtype).nexcept = 0;
	       }
#line 11274 "y.tab.c" /* yacc.c:1646  */
    break;

  case 506:
#line 6390 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = NewString("true");
	       }
#line 11284 "y.tab.c" /* yacc.c:1646  */
    break;

  case 507:
#line 6395 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = 0;
	       }
#line 11294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 508:
#line 6400 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = NewString("true");
	       }
#line 11304 "y.tab.c" /* yacc.c:1646  */
    break;

  case 509:
#line 6405 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = (yyvsp[-1].dtype).val;
	       }
#line 11314 "y.tab.c" /* yacc.c:1646  */
    break;

  case 510:
#line 6412 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = 0;
                    (yyval.dtype).qualifier = (yyvsp[0].str);
               }
#line 11325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 511:
#line 6418 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.dtype) = (yyvsp[0].dtype);
                    (yyval.dtype).qualifier = 0;
               }
#line 11334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 512:
#line 6422 "parser.y" /* yacc.c:1646  */
    {
		    (yyval.dtype) = (yyvsp[0].dtype);
                    (yyval.dtype).qualifier = (yyvsp[-1].str);
               }
#line 11343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 513:
#line 6426 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = 0;
                    (yyval.dtype).qualifier = 0; 
               }
#line 11354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 514:
#line 6434 "parser.y" /* yacc.c:1646  */
    { 
                    Clear(scanner_ccode); 
                    (yyval.decl).have_parms = 0; 
                    (yyval.decl).defarg = 0; 
		    (yyval.decl).throws = (yyvsp[-2].dtype).throws;
		    (yyval.decl).throwf = (yyvsp[-2].dtype).throwf;
		    (yyval.decl).nexcept = (yyvsp[-2].dtype).nexcept;
               }
#line 11367 "y.tab.c" /* yacc.c:1646  */
    break;

  case 515:
#line 6442 "parser.y" /* yacc.c:1646  */
    { 
                    skip_balanced('{','}'); 
                    (yyval.decl).have_parms = 0; 
                    (yyval.decl).defarg = 0; 
                    (yyval.decl).throws = (yyvsp[-2].dtype).throws;
                    (yyval.decl).throwf = (yyvsp[-2].dtype).throwf;
                    (yyval.decl).nexcept = (yyvsp[-2].dtype).nexcept;
               }
#line 11380 "y.tab.c" /* yacc.c:1646  */
    break;

  case 516:
#line 6450 "parser.y" /* yacc.c:1646  */
    { 
                    Clear(scanner_ccode); 
                    (yyval.decl).parms = (yyvsp[-2].pl); 
                    (yyval.decl).have_parms = 1; 
                    (yyval.decl).defarg = 0; 
		    (yyval.decl).throws = 0;
		    (yyval.decl).throwf = 0;
		    (yyval.decl).nexcept = 0;
               }
#line 11394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 517:
#line 6459 "parser.y" /* yacc.c:1646  */
    {
                    skip_balanced('{','}'); 
                    (yyval.decl).parms = (yyvsp[-2].pl); 
                    (yyval.decl).have_parms = 1; 
                    (yyval.decl).defarg = 0; 
                    (yyval.decl).throws = 0;
                    (yyval.decl).throwf = 0;
                    (yyval.decl).nexcept = 0;
               }
#line 11408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 518:
#line 6468 "parser.y" /* yacc.c:1646  */
    { 
                    (yyval.decl).have_parms = 0; 
                    (yyval.decl).defarg = (yyvsp[-1].dtype).val; 
                    (yyval.decl).throws = 0;
                    (yyval.decl).throwf = 0;
                    (yyval.decl).nexcept = 0;
               }
#line 11420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 519:
#line 6475 "parser.y" /* yacc.c:1646  */
    {
                    (yyval.decl).have_parms = 0;
                    (yyval.decl).defarg = (yyvsp[-1].dtype).val;
                    (yyval.decl).throws = (yyvsp[-3].dtype).throws;
                    (yyval.decl).throwf = (yyvsp[-3].dtype).throwf;
                    (yyval.decl).nexcept = (yyvsp[-3].dtype).nexcept;
               }
#line 11432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 526:
#line 6494 "parser.y" /* yacc.c:1646  */
    {
		  skip_balanced('(',')');
		  Clear(scanner_ccode);
		}
#line 11441 "y.tab.c" /* yacc.c:1646  */
    break;

  case 527:
#line 6506 "parser.y" /* yacc.c:1646  */
    {
		  skip_balanced('{','}');
		  Clear(scanner_ccode);
		}
#line 11450 "y.tab.c" /* yacc.c:1646  */
    break;

  case 528:
#line 6512 "parser.y" /* yacc.c:1646  */
    {
                     String *s = NewStringEmpty();
                     SwigType_add_template(s,(yyvsp[-1].p));
                     (yyval.id) = Char(s);
		     scanner_last_id(1);
                }
#line 11461 "y.tab.c" /* yacc.c:1646  */
    break;

  case 529:
#line 6521 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (yyvsp[0].id); }
#line 11467 "y.tab.c" /* yacc.c:1646  */
    break;

  case 530:
#line 6522 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = Swig_copy_string("override"); }
#line 11473 "y.tab.c" /* yacc.c:1646  */
    break;

  case 531:
#line 6523 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = Swig_copy_string("final"); }
#line 11479 "y.tab.c" /* yacc.c:1646  */
    break;

  case 532:
#line 6526 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (yyvsp[0].id); }
#line 11485 "y.tab.c" /* yacc.c:1646  */
    break;

  case 533:
#line 6527 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = Char((yyvsp[0].dtype).val); }
#line 11491 "y.tab.c" /* yacc.c:1646  */
    break;

  case 534:
#line 6528 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = Char((yyvsp[0].str)); }
#line 11497 "y.tab.c" /* yacc.c:1646  */
    break;

  case 535:
#line 6531 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = (yyvsp[0].id); }
#line 11503 "y.tab.c" /* yacc.c:1646  */
    break;

  case 536:
#line 6532 "parser.y" /* yacc.c:1646  */
    { (yyval.id) = 0; }
#line 11509 "y.tab.c" /* yacc.c:1646  */
    break;

  case 537:
#line 6535 "parser.y" /* yacc.c:1646  */
    { 
                  (yyval.str) = 0;
		  if (!(yyval.str)) (yyval.str) = NewStringf("%s%s", (yyvsp[-1].str),(yyvsp[0].str));
      	          Delete((yyvsp[0].str));
               }
#line 11519 "y.tab.c" /* yacc.c:1646  */
    break;

  case 538:
#line 6540 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = NewStringf("::%s%s",(yyvsp[-1].str),(yyvsp[0].str));
                 Delete((yyvsp[0].str));
               }
#line 11528 "y.tab.c" /* yacc.c:1646  */
    break;

  case 539:
#line 6544 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = NewString((yyvsp[0].str));
   	       }
#line 11536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 540:
#line 6547 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 541:
#line 6550 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.str) = NewStringf("%s", (yyvsp[0].str));
	       }
#line 11552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 542:
#line 6553 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.str) = NewStringf("%s%s", (yyvsp[-1].str), (yyvsp[0].id));
	       }
#line 11560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 543:
#line 6556 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 544:
#line 6561 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = NewStringf("::%s%s",(yyvsp[-1].str),(yyvsp[0].str));
		   Delete((yyvsp[0].str));
               }
#line 11577 "y.tab.c" /* yacc.c:1646  */
    break;

  case 545:
#line 6565 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 546:
#line 6568 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11593 "y.tab.c" /* yacc.c:1646  */
    break;

  case 547:
#line 6575 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = NewStringf("::~%s",(yyvsp[0].str));
               }
#line 11601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 548:
#line 6581 "parser.y" /* yacc.c:1646  */
    {
		(yyval.str) = NewStringf("%s", (yyvsp[0].id));
	      }
#line 11609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 549:
#line 6584 "parser.y" /* yacc.c:1646  */
    {
		(yyval.str) = NewStringf("%s%s", (yyvsp[-1].id), (yyvsp[0].id));
	      }
#line 11617 "y.tab.c" /* yacc.c:1646  */
    break;

  case 550:
#line 6589 "parser.y" /* yacc.c:1646  */
    {
		(yyval.str) = (yyvsp[0].str);
	      }
#line 11625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 551:
#line 6592 "parser.y" /* yacc.c:1646  */
    {
		(yyval.str) = NewStringf("%s%s", (yyvsp[-1].id), (yyvsp[0].id));
	      }
#line 11633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 552:
#line 6598 "parser.y" /* yacc.c:1646  */
    {
                  (yyval.str) = 0;
		  if (!(yyval.str)) (yyval.str) = NewStringf("%s%s", (yyvsp[-1].id),(yyvsp[0].str));
      	          Delete((yyvsp[0].str));
               }
#line 11643 "y.tab.c" /* yacc.c:1646  */
    break;

  case 553:
#line 6603 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = NewStringf("::%s%s",(yyvsp[-1].id),(yyvsp[0].str));
                 Delete((yyvsp[0].str));
               }
#line 11652 "y.tab.c" /* yacc.c:1646  */
    break;

  case 554:
#line 6607 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = NewString((yyvsp[0].id));
   	       }
#line 11660 "y.tab.c" /* yacc.c:1646  */
    break;

  case 555:
#line 6610 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = NewStringf("::%s",(yyvsp[0].id));
               }
#line 11668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 556:
#line 6613 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.str) = NewString((yyvsp[0].str));
	       }
#line 11676 "y.tab.c" /* yacc.c:1646  */
    break;

  case 557:
#line 6616 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11684 "y.tab.c" /* yacc.c:1646  */
    break;

  case 558:
#line 6621 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = NewStringf("::%s%s",(yyvsp[-1].id),(yyvsp[0].str));
		   Delete((yyvsp[0].str));
               }
#line 11693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 559:
#line 6625 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = NewStringf("::%s",(yyvsp[0].id));
               }
#line 11701 "y.tab.c" /* yacc.c:1646  */
    break;

  case 560:
#line 6628 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11709 "y.tab.c" /* yacc.c:1646  */
    break;

  case 561:
#line 6631 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = NewStringf("::~%s",(yyvsp[0].id));
               }
#line 11717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 562:
#line 6637 "parser.y" /* yacc.c:1646  */
    { 
                   (yyval.str) = NewStringf("%s%s", (yyvsp[-1].str), (yyvsp[0].id));
               }
#line 11725 "y.tab.c" /* yacc.c:1646  */
    break;

  case 563:
#line 6640 "parser.y" /* yacc.c:1646  */
    { (yyval.str) = NewString((yyvsp[0].id));}
#line 11731 "y.tab.c" /* yacc.c:1646  */
    break;

  case 564:
#line 6643 "parser.y" /* yacc.c:1646  */
    {
                   (yyval.str) = NewStringf("%s%s", (yyvsp[-1].str), (yyvsp[0].id));
               }
#line 11739 "y.tab.c" /* yacc.c:1646  */
    break;

  case 565:
#line 6651 "parser.y" /* yacc.c:1646  */
    { (yyval.str) = NewString((yyvsp[0].id));}
#line 11745 "y.tab.c" /* yacc.c:1646  */
    break;

  case 566:
#line 6654 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = (yyvsp[0].str);
               }
#line 11753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 567:
#line 6657 "parser.y" /* yacc.c:1646  */
    {
                  skip_balanced('{','}');
		  (yyval.str) = NewString(scanner_ccode);
               }
#line 11762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 568:
#line 6661 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = (yyvsp[0].str);
              }
#line 11770 "y.tab.c" /* yacc.c:1646  */
    break;

  case 569:
#line 6666 "parser.y" /* yacc.c:1646  */
    {
                  Hash *n;
                  (yyval.node) = NewHash();
                  n = (yyvsp[-1].node);
                  while(n) {
                     String *name, *value;
                     name = Getattr(n,"name");
                     value = Getattr(n,"value");
		     if (!value) value = (String *) "1";
                     Setattr((yyval.node),name, value);
		     n = nextSibling(n);
		  }
               }
#line 11788 "y.tab.c" /* yacc.c:1646  */
    break;

  case 570:
#line 6679 "parser.y" /* yacc.c:1646  */
    { (yyval.node) = 0; }
#line 11794 "y.tab.c" /* yacc.c:1646  */
    break;

  case 571:
#line 6683 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = NewHash();
		 Setattr((yyval.node),"name",(yyvsp[-2].id));
		 Setattr((yyval.node),"value",(yyvsp[0].str));
               }
#line 11804 "y.tab.c" /* yacc.c:1646  */
    break;

  case 572:
#line 6688 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.node) = NewHash();
		 Setattr((yyval.node),"name",(yyvsp[-4].id));
		 Setattr((yyval.node),"value",(yyvsp[-2].str));
		 set_nextSibling((yyval.node),(yyvsp[0].node));
               }
#line 11815 "y.tab.c" /* yacc.c:1646  */
    break;

  case 573:
#line 6694 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = NewHash();
                 Setattr((yyval.node),"name",(yyvsp[0].id));
	       }
#line 11824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 574:
#line 6698 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = NewHash();
                 Setattr((yyval.node),"name",(yyvsp[-2].id));
                 set_nextSibling((yyval.node),(yyvsp[0].node));
               }
#line 11834 "y.tab.c" /* yacc.c:1646  */
    break;

  case 575:
#line 6703 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = (yyvsp[0].node);
		 Setattr((yyval.node),"name",(yyvsp[-2].id));
               }
#line 11843 "y.tab.c" /* yacc.c:1646  */
    break;

  case 576:
#line 6707 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.node) = (yyvsp[-2].node);
		 Setattr((yyval.node),"name",(yyvsp[-4].id));
		 set_nextSibling((yyval.node),(yyvsp[0].node));
               }
#line 11853 "y.tab.c" /* yacc.c:1646  */
    break;

  case 577:
#line 6714 "parser.y" /* yacc.c:1646  */
    {
		 (yyval.str) = (yyvsp[0].str);
               }
#line 11861 "y.tab.c" /* yacc.c:1646  */
    break;

  case 578:
#line 6717 "parser.y" /* yacc.c:1646  */
    {
                 (yyval.str) = Char((yyvsp[0].dtype).val);
               }
#line 11869 "y.tab.c" /* yacc.c:1646  */
    break;


#line 11873 "y.tab.c" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 6724 "parser.y" /* yacc.c:1906  */


SwigType *Swig_cparse_type(String *s) {
   String *ns;
   ns = NewStringf("%s;",s);
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSETYPE);
   yyparse();
   /*   Printf(stdout,"typeparse: '%s' ---> '%s'\n", s, top); */
   return top;
}


Parm *Swig_cparse_parm(String *s) {
   String *ns;
   ns = NewStringf("%s;",s);
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSEPARM);
   yyparse();
   /*   Printf(stdout,"typeparse: '%s' ---> '%s'\n", s, top); */
   Delete(ns);
   return top;
}


ParmList *Swig_cparse_parms(String *s, Node *file_line_node) {
   String *ns;
   char *cs = Char(s);
   if (cs && cs[0] != '(') {
     ns = NewStringf("(%s);",s);
   } else {
     ns = NewStringf("%s;",s);
   }
   Setfile(ns, Getfile(file_line_node));
   Setline(ns, Getline(file_line_node));
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSEPARMS);
   yyparse();
   /*   Printf(stdout,"typeparse: '%s' ---> '%s'\n", s, top); */
   return top;
}

