/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * naming.c
 *
 * Functions for generating various kinds of names during code generation.
 *
 * Swig_name_register is used to register a format string for generating names.
 * The format string makes use of the following format specifiers:
 *
 * %c - class name is substituted
 * %f - function name is substituted
 * %m - member name is substituted
 * %n - namespace is substituted
 * %v - variable name is substituted
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "cparse.h"
#include <ctype.h>

/* Hash table containing naming data */

static Hash *naming_hash = 0;

#if 0
#define SWIG_DEBUG
#endif

/* -----------------------------------------------------------------------------
 * Swig_name_register()
 *
 * Register a new naming format.
 * ----------------------------------------------------------------------------- */

void Swig_name_register(const_String_or_char_ptr method, const_String_or_char_ptr format) {
  if (!naming_hash)
    naming_hash = NewHash();
  Setattr(naming_hash, method, format);
}

void Swig_name_unregister(const_String_or_char_ptr method) {
  if (naming_hash) {
    Delattr(naming_hash, method);
  }
}

/* Return naming format for the specified method or the default format if none was explicitly registered */
static String* get_naming_format_for(const char *method, const char *def_format) {
  String* f = naming_hash ? Getattr(naming_hash, method) : NULL;

  return f ? Copy(f) : NewString(def_format);
}

static int name_mangle(String *r) {
  char *c;
  int special;
  special = 0;
  Replaceall(r, "::", "_");
  c = Char(r);
  while (*c) {
    if (!isalnum((int) *c) && (*c != '_')) {
      special = 1;
      switch (*c) {
      case '+':
	*c = 'a';
	break;
      case '-':
	*c = 's';
	break;
      case '*':
	*c = 'm';
	break;
      case '/':
	*c = 'd';
	break;
      case '<':
	*c = 'l';
	break;
      case '>':
	*c = 'g';
	break;
      case '=':
	*c = 'e';
	break;
      case ',':
	*c = 'c';
	break;
      case '(':
	*c = 'p';
	break;
      case ')':
	*c = 'P';
	break;
      case '[':
	*c = 'b';
	break;
      case ']':
	*c = 'B';
	break;
      case '^':
	*c = 'x';
	break;
      case '&':
	*c = 'A';
	break;
      case '|':
	*c = 'o';
	break;
      case '~':
	*c = 'n';
	break;
      case '!':
	*c = 'N';
	break;
      case '%':
	*c = 'M';
	break;
      case '.':
	*c = 'f';
	break;
      case '?':
	*c = 'q';
	break;
      default:
	*c = '_';
	break;
      }
    }
    c++;
  }
  if (special)
    Append(r, "___");
  return special;
}

/* -----------------------------------------------------------------------------
 * replace_nspace()
 *
 * Mangles in the namespace from nspace by replacing %n in name if nspace feature required.
 * ----------------------------------------------------------------------------- */

static void replace_nspace(String *name, const_String_or_char_ptr nspace) {
  if (nspace) {
    String *namspace = NewStringf("%s_", nspace);
    Replaceall(namspace, NSPACE_SEPARATOR, "_");
    Replace(name, "%n", namspace, DOH_REPLACE_ANY);
    Delete(namspace);
  } else {
    Replace(name, "%n", "", DOH_REPLACE_ANY);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_name_mangle()
 *
 * Converts all of the non-identifier characters of a string to underscores.
 * ----------------------------------------------------------------------------- */

String *Swig_name_mangle(const_String_or_char_ptr s) {
#if 0
  String *r = NewString(s);
  name_mangle(r);
  return r;
#else
  return Swig_string_mangle(s);
#endif
}

/* -----------------------------------------------------------------------------
 * Swig_name_wrapper()
 *
 * Returns the name of a wrapper function.
 * ----------------------------------------------------------------------------- */

String *Swig_name_wrapper(const_String_or_char_ptr fname) {
  String *r = get_naming_format_for("wrapper", "_wrap_%f");

  Replace(r, "%f", fname, DOH_REPLACE_ANY);
  name_mangle(r);
  return r;
}


/* -----------------------------------------------------------------------------
 * Swig_name_member()
 *
 * Returns the name of a class method.
 * ----------------------------------------------------------------------------- */

String *Swig_name_member(const_String_or_char_ptr nspace, const_String_or_char_ptr classname, const_String_or_char_ptr membername) {
  String *r;
  String *rclassname;
  char *cname;

  rclassname = SwigType_namestr(classname);
  r = get_naming_format_for("member", "%n%c_%m");
  cname = Char(rclassname);
  if ((strncmp(cname, "struct ", 7) == 0) || ((strncmp(cname, "class ", 6) == 0)) || ((strncmp(cname, "union ", 6) == 0))) {
    cname = strchr(cname, ' ') + 1;
  }
  replace_nspace(r, nspace);
  Replace(r, "%c", cname, DOH_REPLACE_ANY);
  Replace(r, "%m", membername, DOH_REPLACE_ANY);
  /*  name_mangle(r); */
  Delete(rclassname);
  return r;
}

/* -----------------------------------------------------------------------------
 * Swig_name_get()
 *
 * Returns the name of the accessor function used to get a variable.
 * ----------------------------------------------------------------------------- */

String *Swig_name_get(const_String_or_char_ptr nspace, const_String_or_char_ptr vname) {
  String *r = get_naming_format_for("get", "%n%v_get");

#ifdef SWIG_DEBUG
  Printf(stdout, "Swig_name_get:  '%s'\n", vname);
#endif

  replace_nspace(r, nspace);
  Replace(r, "%v", vname, DOH_REPLACE_ANY);
  /* name_mangle(r); */
  return r;
}

/* ----------------------------------------------------------------------------- 
 * Swig_name_set()
 *
 * Returns the name of the accessor function used to set a variable.
 * ----------------------------------------------------------------------------- */

String *Swig_name_set(const_String_or_char_ptr nspace, const_String_or_char_ptr vname) {
  String *r = get_naming_format_for("set", "%n%v_set");

  replace_nspace(r, nspace);
  Replace(r, "%v", vname, DOH_REPLACE_ANY);
  /* name_mangle(r); */
  return r;
}

/* Common implementation of all Swig_name_<special-method>() functions below. */
static String *make_full_name_for(const char *method, const char *def_format, const_String_or_char_ptr nspace, const_String_or_char_ptr classname) {
  String *r;
  String *rclassname;
  char *cname;

  rclassname = SwigType_namestr(classname);
  r = get_naming_format_for(method, def_format);

  cname = Char(rclassname);
  if ((strncmp(cname, "struct ", 7) == 0) || ((strncmp(cname, "class ", 6) == 0)) || ((strncmp(cname, "union ", 6) == 0))) {
    cname = strchr(cname, ' ') + 1;
  }

  replace_nspace(r, nspace);
  Replace(r, "%c", cname, DOH_REPLACE_ANY);
  Delete(rclassname);
  return r;
}

/* -----------------------------------------------------------------------------
 * Swig_name_construct()
 *
 * Returns the name of the accessor function used to create an object.
 * ----------------------------------------------------------------------------- */

String *Swig_name_construct(const_String_or_char_ptr nspace, const_String_or_char_ptr classname) {
  return make_full_name_for("construct", "new_%n%c", nspace, classname);
}


/* -----------------------------------------------------------------------------
 * Swig_name_copyconstructor()
 *
 * Returns the name of the accessor function used to copy an object.
 * ----------------------------------------------------------------------------- */

String *Swig_name_copyconstructor(const_String_or_char_ptr nspace, const_String_or_char_ptr classname) {
  return make_full_name_for("copy", "copy_%n%c", nspace, classname);
}

/* -----------------------------------------------------------------------------
 * Swig_name_destroy()
 *
 * Returns the name of the accessor function used to destroy an object.
 * ----------------------------------------------------------------------------- */

String *Swig_name_destroy(const_String_or_char_ptr nspace, const_String_or_char_ptr classname) {
  return make_full_name_for("destroy", "delete_%n%c", nspace, classname);
}


/* -----------------------------------------------------------------------------
 * Swig_name_disown()
 *
 * Returns the name of the accessor function used to disown an object.
 * ----------------------------------------------------------------------------- */

String *Swig_name_disown(const_String_or_char_ptr nspace, const_String_or_char_ptr classname) {
  return make_full_name_for("disown", "disown_%n%c", nspace, classname);
}


/* -----------------------------------------------------------------------------
 * Swig_name_object_set()
 *
 * Sets an object associated with a name and optional declarators. 
 * ----------------------------------------------------------------------------- */

void Swig_name_object_set(Hash *namehash, String *name, SwigType *decl, DOH *object) {
  DOH *n;

#ifdef SWIG_DEBUG
  Printf(stdout, "Swig_name_object_set:  '%s', '%s'\n", name, decl);
#endif
  n = Getattr(namehash, name);
  if (!n) {
    n = NewHash();
    Setattr(namehash, name, n);
    Delete(n);
  }
  /* Add an object based on the declarator value */
  if (!decl) {
    Setattr(n, "start", object);
  } else {
    SwigType *cd = Copy(decl);
    Setattr(n, cd, object);
    Delete(cd);
  }
}


/* -----------------------------------------------------------------------------
 * Swig_name_object_get()
 *
 * Return an object associated with an optional class prefix, name, and 
 * declarator.   This function operates according to name matching rules
 * described for the %rename directive in the SWIG manual.
 * ----------------------------------------------------------------------------- */

static DOH *get_object(Hash *n, String *decl) {
  DOH *rn = 0;
  if (!n)
    return 0;
  if (decl) {
    rn = Getattr(n, decl);
  } else {
    rn = Getattr(n, "start");
  }
  return rn;
}

static DOH *name_object_get(Hash *namehash, String *tname, SwigType *decl, SwigType *ncdecl) {
  DOH *rn = 0;
  Hash *n = Getattr(namehash, tname);
  if (n) {
    rn = get_object(n, decl);
    if ((!rn) && ncdecl)
      rn = get_object(n, ncdecl);
    if (!rn)
      rn = get_object(n, 0);
  }
  return rn;
}

DOH *Swig_name_object_get(Hash *namehash, String *prefix, String *name, SwigType *decl) {
  String *tname = NewStringEmpty();
  DOH *rn = 0;
  char *ncdecl = 0;

  if (!namehash)
    return 0;

  /* DB: This removed to more tightly control feature/name matching */
  /*  if ((decl) && (SwigType_isqualifier(decl))) {
     ncdecl = strchr(Char(decl),'.');
     ncdecl++;
     }
   */
#ifdef SWIG_DEBUG
  Printf(stdout, "Swig_name_object_get:  '%s' '%s', '%s'\n", prefix, name, decl);
#endif


  /* Perform a class-based lookup (if class prefix supplied) */
  if (prefix) {
    if (Len(prefix)) {
      Printf(tname, "%s::%s", prefix, name);
      rn = name_object_get(namehash, tname, decl, ncdecl);
      if (!rn) {
	String *cls = Swig_scopename_last(prefix);
	if (!Equal(cls, prefix)) {
	  Clear(tname);
	  Printf(tname, "*::%s::%s", cls, name);
	  rn = name_object_get(namehash, tname, decl, ncdecl);
	}
	Delete(cls);
      }
      /* Lookup a name within a templated-based class */
      if (!rn) {
	String *t_name = SwigType_istemplate_templateprefix(prefix);
	if (t_name) {
	  Clear(tname);
	  Printf(tname, "%s::%s", t_name, name);
	  rn = name_object_get(namehash, tname, decl, ncdecl);
	  Delete(t_name);
	}
      }
      /* Lookup a template-based name within a class */
      if (!rn) {
	String *t_name = SwigType_istemplate_templateprefix(name);
	if (t_name)
	  rn = Swig_name_object_get(namehash, prefix, t_name, decl);
	Delete(t_name);
      }
    }
    /* A wildcard-based class lookup */
    if (!rn) {
      Clear(tname);
      Printf(tname, "*::%s", name);
      rn = name_object_get(namehash, tname, decl, ncdecl);
    }
  } else {
    /* Lookup in the global namespace only */
    Clear(tname);
    Printf(tname, "::%s", name);
    rn = name_object_get(namehash, tname, decl, ncdecl);
  }
  /* Catch-all */
  if (!rn) {
    rn = name_object_get(namehash, name, decl, ncdecl);
  }
  if (!rn && Swig_scopename_check(name)) {
    String *nprefix = NewStringEmpty();
    String *nlast = NewStringEmpty();
    Swig_scopename_split(name, &nprefix, &nlast);
    rn = name_object_get(namehash, nlast, decl, ncdecl);
    Delete(nlast);
    Delete(nprefix);
  }

  Delete(tname);

#ifdef SWIG_DEBUG
  Printf(stdout, "Swig_name_object_get:  found %d\n", rn ? 1 : 0);
#endif

  return rn;
}

/* -----------------------------------------------------------------------------
 * Swig_name_object_inherit()
 *
 * Implements name-based inheritance scheme. 
 * ----------------------------------------------------------------------------- */

void Swig_name_object_inherit(Hash *namehash, String *base, String *derived) {
  Iterator ki;
  Hash *derh;
  String *bprefix;
  String *dprefix;
  char *cbprefix;
  int plen;

  if (!namehash)
    return;

  /* Temporary hash holding all the entries we add while we iterate over
     namehash itself as we can't modify the latter while iterating over it. */
  derh = NULL;
  bprefix = NewStringf("%s::", base);
  dprefix = NewStringf("%s::", derived);
  cbprefix = Char(bprefix);
  plen = (int)strlen(cbprefix);
  for (ki = First(namehash); ki.key; ki = Next(ki)) {
    char *k = Char(ki.key);
    if (strncmp(k, cbprefix, plen) == 0) {
      /* Copy, adjusting name, this element to the derived hash. */
      Iterator oi;
      String *nkey = NewStringf("%s%s", dprefix, k + plen);
      Hash *n = ki.item;
      Hash *newh;

      /* Don't overwrite an existing value for the derived class, if any. */
      newh = Getattr(namehash, nkey);
      if (!newh) {
	if (!derh)
	  derh = NewHash();

	newh = NewHash();
	Setattr(derh, nkey, newh);
	Delete(newh);
      }
      for (oi = First(n); oi.key; oi = Next(oi)) {
	if (!Getattr(newh, oi.key)) {
	  String *ci = Copy(oi.item);
	  Setattr(newh, oi.key, ci);
	  Delete(ci);
	}
      }
      Delete(nkey);
    }
  }

  /* Merge the contents of derived hash into the main hash. */
  if (derh) {
    for (ki = First(derh); ki.key; ki = Next(ki)) {
      Setattr(namehash, ki.key, ki.item);
    }
  }

  Delete(bprefix);
  Delete(dprefix);
  Delete(derh);
}

/* -----------------------------------------------------------------------------
 * merge_features()
 *
 * Given a hash, this function merges the features in the hash into the node.
 * ----------------------------------------------------------------------------- */

static void merge_features(Hash *features, Node *n) {
  Iterator ki;

  if (!features)
    return;
  for (ki = First(features); ki.key; ki = Next(ki)) {
    String *ci = Copy(ki.item);
    Setattr(n, ki.key, ci);
    Delete(ci);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_features_get()
 *
 * Attaches any features in the features hash to the node that matches
 * the declaration, decl.
 * ----------------------------------------------------------------------------- */

static void features_get(Hash *features, const String *tname, SwigType *decl, SwigType *ncdecl, Node *node) {
  Node *n = Getattr(features, tname);
#ifdef SWIG_DEBUG
  Printf(stdout, "  features_get: %s\n", tname);
#endif
  if (n) {
    merge_features(get_object(n, 0), node);
    if (ncdecl)
      merge_features(get_object(n, ncdecl), node);
    merge_features(get_object(n, decl), node);
  }
}

void Swig_features_get(Hash *features, String *prefix, String *name, SwigType *decl, Node *node) {
  char *ncdecl = 0;
  String *rdecl = 0;
  String *rname = 0;
  if (!features)
    return;

  /* MM: This removed to more tightly control feature/name matching */
  /*
     if ((decl) && (SwigType_isqualifier(decl))) {
     ncdecl = strchr(Char(decl),'.');
     ncdecl++;
     }
   */

  /* very specific hack for template constructors/destructors */
  if (name && SwigType_istemplate(name)) {
    String *nodetype = nodeType(node);
    if (nodetype && (Equal(nodetype, "constructor") || Equal(nodetype, "destructor"))) {
      String *nprefix = NewStringEmpty();
      String *nlast = NewStringEmpty();
      String *tprefix;
      Swig_scopename_split(name, &nprefix, &nlast);
      tprefix = SwigType_templateprefix(nlast);
      Delete(nlast);
      if (Len(nprefix)) {
	Append(nprefix, "::");
	Append(nprefix, tprefix);
	Delete(tprefix);
	rname = nprefix;
      } else {
	rname = tprefix;
	Delete(nprefix);
      }
      rdecl = Copy(decl);
      Replaceall(rdecl, name, rname);
      decl = rdecl;
      name = rname;
    }
  }

#ifdef SWIG_DEBUG
  Printf(stdout, "Swig_features_get: '%s' '%s' '%s'\n", prefix, name, decl);
#endif

  /* Global features */
  features_get(features, "", 0, 0, node);
  if (name) {
    String *tname = NewStringEmpty();
    /* add features for 'root' template */
    String *dname = SwigType_istemplate_templateprefix(name);
    if (dname) {
      features_get(features, dname, decl, ncdecl, node);
    }
    /* Catch-all */
    features_get(features, name, decl, ncdecl, node);
    /* Perform a class-based lookup (if class prefix supplied) */
    if (prefix) {
      /* A class-generic feature */
      if (Len(prefix)) {
	Printf(tname, "%s::", prefix);
	features_get(features, tname, decl, ncdecl, node);
      }
      /* A wildcard-based class lookup */
      Clear(tname);
      Printf(tname, "*::%s", name);
      features_get(features, tname, decl, ncdecl, node);
      /* A specific class lookup */
      if (Len(prefix)) {
	/* A template-based class lookup */
	String *tprefix = SwigType_istemplate_templateprefix(prefix);
	if (tprefix) {
	  Clear(tname);
	  Printf(tname, "%s::%s", tprefix, name);
	  features_get(features, tname, decl, ncdecl, node);
	}
	Clear(tname);
	Printf(tname, "%s::%s", prefix, name);
	features_get(features, tname, decl, ncdecl, node);
	Delete(tprefix);
      }
    } else {
      /* Lookup in the global namespace only */
      Clear(tname);
      Printf(tname, "::%s", name);
      features_get(features, tname, decl, ncdecl, node);
    }
    Delete(tname);
    Delete(dname);
  }
  if (name && SwigType_istemplate(name)) {
    /* add features for complete template type */
    String *dname = Swig_symbol_template_deftype(name, 0);
    if (!Equal(dname, name)) {
      Swig_features_get(features, prefix, dname, decl, node);
    }
    Delete(dname);
  }

  if (rname)
    Delete(rname);
  if (rdecl)
    Delete(rdecl);
}


/* -----------------------------------------------------------------------------
 * Swig_feature_set()
 *
 * Sets a feature name and value. Also sets optional feature attributes as
 * passed in by featureattribs. Optional feature attributes are given a full name
 * concatenating the feature name plus ':' plus the attribute name.
 * ----------------------------------------------------------------------------- */

void Swig_feature_set(Hash *features, const_String_or_char_ptr name, SwigType *decl, const_String_or_char_ptr featurename, const_String_or_char_ptr value, Hash *featureattribs) {
  Hash *n;
  Hash *fhash;

#ifdef SWIG_DEBUG
  Printf(stdout, "Swig_feature_set: '%s' '%s' '%s' '%s'\n", name, decl, featurename, value);
#endif

  n = Getattr(features, name);
  if (!n) {
    n = NewHash();
    Setattr(features, name, n);
    Delete(n);
  }
  if (!decl) {
    fhash = Getattr(n, "start");
    if (!fhash) {
      fhash = NewHash();
      Setattr(n, "start", fhash);
      Delete(fhash);
    }
  } else {
    fhash = Getattr(n, decl);
    if (!fhash) {
      String *cdecl_ = Copy(decl);
      fhash = NewHash();
      Setattr(n, cdecl_, fhash);
      Delete(cdecl_);
      Delete(fhash);
    }
  }
  if (value) {
    Setattr(fhash, featurename, value);
  } else {
    Delattr(fhash, featurename);
  }

  {
    /* Add in the optional feature attributes */
    Hash *attribs = featureattribs;
    while (attribs) {
      String *attribname = Getattr(attribs, "name");
      String *featureattribname = NewStringf("%s:%s", featurename, attribname);
      if (value) {
	String *attribvalue = Getattr(attribs, "value");
	Setattr(fhash, featureattribname, attribvalue);
      } else {
	Delattr(fhash, featureattribname);
      }
      attribs = nextSibling(attribs);
      Delete(featureattribname);
    }
  }

  if (name && SwigType_istemplate(name)) {
    String *dname = Swig_symbol_template_deftype(name, 0);
    if (Strcmp(dname, name)) {
      Swig_feature_set(features, dname, decl, featurename, value, featureattribs);
    }
    Delete(dname);
  }
}

/* -----------------------------------------------------------------------------
 * The rename/namewarn engine
 *
 * Code below was in parser.y for a while
 * ----------------------------------------------------------------------------- */

static Hash *namewarn_hash = 0;
static Hash *name_namewarn_hash() {
  if (!namewarn_hash)
    namewarn_hash = NewHash();
  return namewarn_hash;
}

static Hash *rename_hash = 0;
static Hash *name_rename_hash() {
  if (!rename_hash)
    rename_hash = NewHash();
  return rename_hash;
}

static List *namewarn_list = 0;
static List *name_namewarn_list() {
  if (!namewarn_list)
    namewarn_list = NewList();
  return namewarn_list;
}

static List *rename_list = 0;
static List *name_rename_list() {
  if (!rename_list)
    rename_list = NewList();
  return rename_list;
}

/* -----------------------------------------------------------------------------
 * int need_name_warning(Node *n)
 *
 * Detects if a node needs name warnings 
 *
 * ----------------------------------------------------------------------------- */

static int need_name_warning(Node *n) {
  int need = 1;
  /* 
     We don't use name warnings for:
     - class forwards, no symbol is generated at the target language.
     - template declarations, only for real instances using %template(name).
     - typedefs, have no effect at the target language.
     - using declarations and using directives, have no effect at the target language.
   */
  if (checkAttribute(n, "nodeType", "classforward")) {
    need = 0;
  } else if (checkAttribute(n, "nodeType", "using")) {
    need = 0;
  } else if (checkAttribute(n, "storage", "typedef")) {
    need = 0;
  } else if (Getattr(n, "hidden")) {
    need = 0;
  } else if (Getattr(n, "ignore")) {
    need = 0;
  } else if (Getattr(n, "templatetype")) {
    need = 0;
  } else if (GetFlag(n, "parsing_template_declaration")) {
    need = 0;
  }
  return need;
}

/* -----------------------------------------------------------------------------
 * int Swig_need_redefined_warn()
 *
 * Detects when a redefined object needs a warning
 * 
 * ----------------------------------------------------------------------------- */

static int nodes_are_equivalent(Node *a, Node *b, int a_inclass) {
  /* they must have the same type */
  String *ta = nodeType(a);
  String *tb = nodeType(b);
  if (!Equal(ta, tb)) {
    if (!(Equal(ta, "using") && Equal(tb, "cdecl"))) {
      return 0;
    }
  }

  if (Cmp(ta, "cdecl") == 0) {
    /* both cdecl case */
    /* typedef */
    String *a_storage = Getattr(a, "storage");
    String *b_storage = Getattr(b, "storage");

    if ((Cmp(a_storage, "typedef") == 0)
	|| (Cmp(b_storage, "typedef") == 0)) {
      if (Cmp(a_storage, b_storage) == 0) {
	String *a_type = (Getattr(a, "type"));
	String *b_type = (Getattr(b, "type"));
	if (Cmp(a_type, b_type) == 0)
	  return 1;
      }
      return 0;
    }

    /* static functions */
    if (Swig_storage_isstatic(a) || Swig_storage_isstatic(b)) {
      if (Cmp(a_storage, b_storage) != 0)
	return 0;
    }

    /* friend methods */

    if (!a_inclass || (Cmp(a_storage, "friend") == 0)) {
      /* check declaration */

      String *a_decl = (Getattr(a, "decl"));
      String *b_decl = (Getattr(b, "decl"));
      if (Cmp(a_decl, b_decl) == 0) {
	/* check return type */
	String *a_type = (Getattr(a, "type"));
	String *b_type = (Getattr(b, "type"));
	if (Cmp(a_type, b_type) == 0) {
	  /* check parameters */
	  Parm *ap = (Getattr(a, "parms"));
	  Parm *bp = (Getattr(b, "parms"));
	  while (ap && bp) {
	    SwigType *at = Getattr(ap, "type");
	    SwigType *bt = Getattr(bp, "type");
	    if (Cmp(at, bt) != 0)
	      return 0;
	    ap = nextSibling(ap);
	    bp = nextSibling(bp);
	  }
	  if (ap || bp) {
	    return 0;
	  } else {
	    Node *a_template = Getattr(a, "template");
	    Node *b_template = Getattr(b, "template");
	    /* Not equivalent if one is a template instantiation (via %template) and the other is a non-templated function */
	    if ((a_template && !b_template) || (!a_template && b_template))
	      return 0;
	  }
	  return 1;
	}
      }
    }
  } else if (Equal(ta, "using")) {
    /* using and cdecl case */
    String *b_storage = Getattr(b, "storage");
    if (Equal(b_storage, "typedef")) {
      String *a_name = Getattr(a, "name");
      String *b_name = Getattr(b, "name");
      if (Equal(a_name, b_name))
	return 1;
    }
  } else {
    /* both %constant case */
    String *a_storage = Getattr(a, "storage");
    String *b_storage = Getattr(b, "storage");
    if ((Cmp(a_storage, "%constant") == 0)
	|| (Cmp(b_storage, "%constant") == 0)) {
      if (Cmp(a_storage, b_storage) == 0) {
	String *a_type = (Getattr(a, "type"));
	String *b_type = (Getattr(b, "type"));
	if ((Cmp(a_type, b_type) == 0)
	    && (Cmp(Getattr(a, "value"), Getattr(b, "value")) == 0))
	  return 1;
      }
      return 0;
    }
    if (Equal(ta, "template") && Equal(tb, "template")) {
      if (Cmp(a_storage, "friend") == 0 || Cmp(b_storage, "friend") == 0)
	return 1;
    }
  }
  return 0;
}

int Swig_need_redefined_warn(Node *a, Node *b, int InClass) {
  String *a_name = Getattr(a, "name");
  String *b_name = Getattr(b, "name");
  String *a_symname = Getattr(a, "sym:name");
  String *b_symname = Getattr(b, "sym:name");
  /* always send a warning if a 'rename' is involved */
  if ((a_symname && !Equal(a_symname, a_name))
      || (b_symname && !Equal(b_symname, b_name))) {
    if (!Equal(a_name, b_name)) {
      return 1;
    }
  }


  return !nodes_are_equivalent(a, b, InClass);
}


/* -----------------------------------------------------------------------------
 * int Swig_need_protected(Node* n)
 *
 * Detects when we need to fully register the protected member.
 * This is basically any protected members when the allprotected mode is set.
 * Otherwise we take just the protected virtual methods and non-static methods 
 * (potentially virtual methods) as well as constructors/destructors.
 * Also any "using" statements in a class may potentially be virtual.
 * ----------------------------------------------------------------------------- */

int Swig_need_protected(Node *n) {
  String *nodetype = nodeType(n);
  if (checkAttribute(n, "access", "protected")) {
    if ((Equal(nodetype, "cdecl"))) {
      if (Swig_director_mode() && Swig_director_protected_mode() && Swig_all_protected_mode()) {
        return 1;
      }
      if (SwigType_isfunction(Getattr(n, "decl"))) {
        String *storage = Getattr(n, "storage");
        /* The function is declared virtual, or it has no storage. This eliminates typedef, static etc. */
        return !storage || Equal(storage, "virtual");
      }
    } else if (Equal(nodetype, "constructor") || Equal(nodetype, "destructor")) {
      return 1;
    } else if (Equal(nodetype, "using") && !Getattr(n, "namespace")) {
      return 1;
    }
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 * void name_nameobj_add()
 *
 * Add nameobj (rename/namewarn)
 * 
 * ----------------------------------------------------------------------------- */

static List *make_attrlist(const char *ckey) {
  List *list = NewList();
  const char *cattr = strchr(ckey, '$');
  if (cattr) {
    String *nattr;
    const char *rattr = strchr(++cattr, '$');
    while (rattr) {
      nattr = NewStringWithSize(cattr, (int)(rattr - cattr));
      Append(list, nattr);
      Delete(nattr);
      cattr = rattr + 1;
      rattr = strchr(cattr, '$');
    }
    nattr = NewString(cattr);
    Append(list, nattr);
    Delete(nattr);
  } else {
    Append(list, "nodeType");
  }
  return list;
}

static void name_object_attach_keys(const char *keys[], Hash *nameobj) {
  Node *kw = nextSibling(nameobj);
  List *matchlist = 0;
  while (kw) {
    Node *next = nextSibling(kw);
    String *kname = Getattr(kw, "name");
    char *ckey = kname ? Char(kname) : 0;
    if (ckey) {
      const char **rkey;
      int isnotmatch = 0;
      int isregexmatch = 0;
      if ((strncmp(ckey, "match", 5) == 0)
	  || (isnotmatch = (strncmp(ckey, "notmatch", 8) == 0))
	  || (isregexmatch = (strncmp(ckey, "regexmatch", 10) == 0))
	  || (isnotmatch = isregexmatch = (strncmp(ckey, "notregexmatch", 13) == 0))) {
	Hash *mi = NewHash();
	List *attrlist = make_attrlist(ckey);
	if (!matchlist)
	  matchlist = NewList();
	Setattr(mi, "value", Getattr(kw, "value"));
	Setattr(mi, "attrlist", attrlist);
	if (isnotmatch)
	  SetFlag(mi, "notmatch");
	if (isregexmatch)
	  SetFlag(mi, "regexmatch");
	Delete(attrlist);
	Append(matchlist, mi);
	Delete(mi);
	removeNode(kw);
      } else {
	for (rkey = keys; *rkey != 0; ++rkey) {
	  if (strcmp(ckey, *rkey) == 0) {
	    Setattr(nameobj, *rkey, Getattr(kw, "value"));
	    removeNode(kw);
	  }
	}
      }
    }
    kw = next;
  }
  if (matchlist) {
    Setattr(nameobj, "matchlist", matchlist);
    Delete(matchlist);
  }
}

static void name_nameobj_add(Hash *name_hash, List *name_list, String *prefix, String *name, SwigType *decl, Hash *nameobj) {
  String *nname = 0;
  if (name && Len(name)) {
    String *target_fmt = Getattr(nameobj, "targetfmt");
    nname = prefix ? NewStringf("%s::%s", prefix, name) : NewString(name);
    if (target_fmt) {
      String *tmp = NewStringf(target_fmt, nname);
      Delete(nname);
      nname = tmp;
    }
  }

  if (!nname || !Len(nname) || Getattr(nameobj, "fullname") ||	/* any of these options trigger a 'list' nameobj */
      Getattr(nameobj, "sourcefmt") || Getattr(nameobj, "matchlist") || Getattr(nameobj, "regextarget")) {
    if (decl)
      Setattr(nameobj, "decl", decl);
    if (nname && Len(nname))
      Setattr(nameobj, "targetname", nname);
    /* put the new nameobj at the beginnig of the list, such that the
       last inserted rule take precedence */
    Insert(name_list, 0, nameobj);
  } else {
    /* here we add an old 'hash' nameobj, simple and fast */
    Swig_name_object_set(name_hash, nname, decl, nameobj);
  }
  Delete(nname);
}

/* -----------------------------------------------------------------------------
 * int name_match_nameobj()
 *
 * Apply and check the nameobj's math list to the node
 * 
 * ----------------------------------------------------------------------------- */

static DOH *get_lattr(Node *n, List *lattr) {
  DOH *res = 0;
  int ilen = Len(lattr);
  int i;
  for (i = 0; n && (i < ilen); ++i) {
    String *nattr = Getitem(lattr, i);
    res = Getattr(n, nattr);
#ifdef SWIG_DEBUG
    if (!res) {
      Printf(stdout, "missing %s %s %s\n", nattr, Getattr(n, "name"), Getattr(n, "member"));
    } else {
      Printf(stdout, "lattr %d %s %s\n", i, nattr, DohIsString(res) ? res : Getattr(res, "name"));
    }
#endif
    n = res;
  }
  return res;
}

#ifdef HAVE_PCRE
#include <pcre.h>

static int name_regexmatch_value(Node *n, String *pattern, String *s) {
  pcre *compiled_pat;
  const char *err;
  int errpos;
  int rc;

  compiled_pat = pcre_compile(Char(pattern), 0, &err, &errpos, NULL);
  if (!compiled_pat) {
    Swig_error("SWIG", Getline(n),
               "Invalid regex \"%s\": compilation failed at %d: %s\n",
               Char(pattern), errpos, err);
    exit(1);
  }

  rc = pcre_exec(compiled_pat, NULL, Char(s), Len(s), 0, 0, NULL, 0);
  pcre_free(compiled_pat);

  if (rc == PCRE_ERROR_NOMATCH)
    return 0;

  if (rc < 0 ) {
    Swig_error("SWIG", Getline(n),
               "Matching \"%s\" against regex \"%s\" failed: %d\n",
               Char(s), Char(pattern), rc);
    exit(1);
  }

  return 1;
}

#else /* !HAVE_PCRE */

static int name_regexmatch_value(Node *n, String *pattern, String *s) {
  (void)pattern;
  (void)s;
  Swig_error("SWIG", Getline(n),
             "PCRE regex matching is not available in this SWIG build.\n");
  exit(1);
}

#endif /* HAVE_PCRE/!HAVE_PCRE */

static int name_match_value(String *mvalue, String *value) {
#if defined(SWIG_USE_SIMPLE_MATCHOR)
  int match = 0;
  char *cvalue = Char(value);
  char *cmvalue = Char(mvalue);
  char *sep = strchr(cmvalue, '|');
  while (sep && !match) {
    match = strncmp(cvalue, cmvalue, sep - cmvalue) == 0;
#ifdef SWIG_DEBUG
    Printf(stdout, "match_value: %s %s %d\n", cvalue, cmvalue, match);
#endif
    cmvalue = sep + 1;
    sep = strchr(cmvalue, '|');
  }
  if (!match) {
    match = strcmp(cvalue, cmvalue) == 0;
#ifdef SWIG_DEBUG
    Printf(stdout, "match_value: %s %s %d\n", cvalue, cmvalue, match);
#endif
  }
  return match;
#else
  return Equal(mvalue, value);
#endif
}

static int name_match_nameobj(Hash *rn, Node *n) {
  int match = 1;
  List *matchlist = Getattr(rn, "matchlist");
#ifdef SWIG_DEBUG
  Printf(stdout, "name_match_nameobj: %s\n", Getattr(n, "name"));
#endif
  if (matchlist) {
    int ilen = Len(matchlist);
    int i;
    for (i = 0; match && (i < ilen); ++i) {
      Node *mi = Getitem(matchlist, i);
      List *lattr = Getattr(mi, "attrlist");
      String *nval = get_lattr(n, lattr);
      int notmatch = GetFlag(mi, "notmatch");
      int regexmatch = GetFlag(mi, "regexmatch");
      match = 0;
      if (nval) {
	String *kwval = Getattr(mi, "value");
	match = regexmatch ? name_regexmatch_value(n, kwval, nval)
	    : name_match_value(kwval, nval);
#ifdef SWIG_DEBUG
	Printf(stdout, "val %s %s %d %d \n", nval, kwval, match, ilen);
#endif
      }
      if (notmatch)
	match = !match;
    }
  }
#ifdef SWIG_DEBUG
  Printf(stdout, "name_match_nameobj: %d\n", match);
#endif
  return match;
}

/* -----------------------------------------------------------------------------
 * Hash *name_nameobj_lget()
 *
 * Get a nameobj (rename/namewarn) from the list of filters
 * 
 * ----------------------------------------------------------------------------- */

static Hash *name_nameobj_lget(List *namelist, Node *n, String *prefix, String *name, String *decl) {
  Hash *res = 0;
  if (namelist) {
    int len = Len(namelist);
    int i;
    int match = 0;
    for (i = 0; !match && (i < len); i++) {
      Hash *rn = Getitem(namelist, i);
      String *rdecl = Getattr(rn, "decl");
      if (rdecl && (!decl || !Equal(rdecl, decl))) {
	continue;
      } else if (name_match_nameobj(rn, n)) {
	String *tname = Getattr(rn, "targetname");
	if (tname) {
	  String *sfmt = Getattr(rn, "sourcefmt");
	  String *sname = 0;
	  int fullname = GetFlag(rn, "fullname");
	  int regextarget = GetFlag(rn, "regextarget");
	  if (sfmt) {
	    if (fullname && prefix) {
	      String *pname = NewStringf("%s::%s", prefix, name);
	      sname = NewStringf(sfmt, pname);
	      Delete(pname);
	    } else {
	      sname = NewStringf(sfmt, name);
	    }
	  } else {
	    if (fullname && prefix) {
	      sname = NewStringf("%s::%s", prefix, name);
	    } else {
	      sname = name;
	      DohIncref(name);
	    }
	  }
	  match = regextarget ? name_regexmatch_value(n, tname, sname)
	    : name_match_value(tname, sname);
	  Delete(sname);
	} else {
	  /* Applying the renaming rule may fail if it contains a %(regex)s expression that doesn't match the given name. */
	  String *sname = NewStringf(Getattr(rn, "name"), name);
	  if (sname) {
	    if (Len(sname))
	      match = 1;
	    Delete(sname);
	  }
	}
      }
      if (match) {
	res = rn;
	break;
      }
    }
  }
  return res;
}

/* -----------------------------------------------------------------------------
 * Swig_name_namewarn_add
 *
 * Add a namewarn objects
 * 
 * ----------------------------------------------------------------------------- */

void Swig_name_namewarn_add(String *prefix, String *name, SwigType *decl, Hash *namewrn) {
  const char *namewrn_keys[] = { "rename", "error", "fullname", "sourcefmt", "targetfmt", 0 };
  name_object_attach_keys(namewrn_keys, namewrn);
  name_nameobj_add(name_namewarn_hash(), name_namewarn_list(), prefix, name, decl, namewrn);
}

/* -----------------------------------------------------------------------------
 * Hash *name_namewarn_get()
 *
 * Return the namewarn object, if there is one.
 * 
 * ----------------------------------------------------------------------------- */

static Hash *name_namewarn_get(Node *n, String *prefix, String *name, SwigType *decl) {
  if (!namewarn_hash && !namewarn_list)
    return 0;
  if (n) {
    /* Return in the obvious cases */
    if (!name || !need_name_warning(n)) {
      return 0;
    } else {
      String *access = Getattr(n, "access");
      int is_public = !access || Equal(access, "public");
      if (!is_public && !Swig_need_protected(n)) {
	return 0;
      }
    }
  }
  if (name) {
    /* Check to see if the name is in the hash */
    Hash *wrn = Swig_name_object_get(name_namewarn_hash(), prefix, name, decl);
    if (wrn && !name_match_nameobj(wrn, n))
      wrn = 0;
    if (!wrn) {
      wrn = name_nameobj_lget(name_namewarn_list(), n, prefix, name, decl);
    }
    if (wrn && Getattr(wrn, "error")) {
      if (n) {
	Swig_error(Getfile(n), Getline(n), "%s\n", Getattr(wrn, "name"));
      } else {
	Swig_error(cparse_file, cparse_line, "%s\n", Getattr(wrn, "name"));
      }
    }
    return wrn;
  } else {
    return 0;
  }
}

/* -----------------------------------------------------------------------------
 * String *Swig_name_warning()
 *
 * Return the name warning, if there is one.
 * 
 * ----------------------------------------------------------------------------- */

String *Swig_name_warning(Node *n, String *prefix, String *name, SwigType *decl) {
  Hash *wrn = name_namewarn_get(n, prefix, name, decl);
  return (name && wrn) ? Getattr(wrn, "name") : 0;
}

/* -----------------------------------------------------------------------------
 * Swig_name_rename_add()
 *
 * Manage the rename objects
 * 
 * ----------------------------------------------------------------------------- */

static void single_rename_add(String *prefix, String *name, SwigType *decl, Hash *newname) {
  name_nameobj_add(name_rename_hash(), name_rename_list(), prefix, name, decl, newname);
}

/* Add a new rename. Works much like new_feature including default argument handling. */
void Swig_name_rename_add(String *prefix, String *name, SwigType *decl, Hash *newname, ParmList *declaratorparms) {

  ParmList *declparms = declaratorparms;

  const char *rename_keys[] = { "fullname", "sourcefmt", "targetfmt", "continue", "regextarget", 0 };
  name_object_attach_keys(rename_keys, newname);

  /* Add the name */
  single_rename_add(prefix, name, decl, newname);

  /* Add extra names if there are default parameters in the parameter list */
  if (decl) {
    int constqualifier = SwigType_isconst(decl);
    while (declparms) {
      if (ParmList_has_defaultargs(declparms)) {

	/* Create a parameter list for the new rename by copying all
	   but the last (defaulted) parameter */
	ParmList *newparms = CopyParmListMax(declparms,ParmList_len(declparms)-1);

	/* Create new declaration - with the last parameter removed */
	SwigType *newdecl = Copy(decl);
	Delete(SwigType_pop_function(newdecl));	/* remove the old parameter list from newdecl */
	SwigType_add_function(newdecl, newparms);
	if (constqualifier)
	  SwigType_add_qualifier(newdecl, "const");

	single_rename_add(prefix, name, newdecl, newname);
	declparms = newparms;
	Delete(newdecl);
      } else {
	declparms = 0;
      }
    }
  }
}


/* Create a name applying rename/namewarn if needed */
static String *apply_rename(String *newname, int fullname, String *prefix, String *name) {
  String *result = 0;
  if (newname && Len(newname)) {
    if (Strcmp(newname, "$ignore") == 0) {
      result = Copy(newname);
    } else {
      char *cnewname = Char(newname);
      if (cnewname) {
	int destructor = name && (*(Char(name)) == '~');
	String *fmt = newname;
	/* use name as a fmt, but avoid C++ "%" and "%=" operators */
	if (Len(newname) > 1 && strchr(cnewname, '%') && !(strcmp(cnewname, "%=") == 0)) {
	  if (fullname && prefix) {
	    result = NewStringf(fmt, prefix, name);
	  } else {
	    result = NewStringf(fmt, name);
	  }
	} else {
	  result = Copy(newname);
	}
	if (destructor && result && (*(Char(result)) != '~')) {
	  Insert(result, 0, "~");
	}
      }
    }
  }

  return result;
}

/* -----------------------------------------------------------------------------
 * String *Swig_name_make()
 *
 * Make a name after applying all the rename/namewarn objects
 * 
 * ----------------------------------------------------------------------------- */

String *Swig_name_make(Node *n, String *prefix, const_String_or_char_ptr cname, SwigType *decl, String *oldname) {
  String *nname = 0;
  String *result = 0;
  String *name = NewString(cname);
  Hash *wrn = 0;
  String *rdecl = 0;
  String *rname = 0;

  /* very specific hack for template constructors/destructors */
#ifdef SWIG_DEBUG
  Printf(stdout, "Swig_name_make: looking for %s %s %s %s\n", prefix, name, decl, oldname);
#endif

  if (name && n && SwigType_istemplate(name)) {
    String *nodetype = nodeType(n);
    if (nodetype && (Equal(nodetype, "constructor") || Equal(nodetype, "destructor"))) {
      String *nprefix = NewStringEmpty();
      String *nlast = NewStringEmpty();
      String *tprefix;
      Swig_scopename_split(name, &nprefix, &nlast);
      tprefix = SwigType_templateprefix(nlast);
      Delete(nlast);
      if (Len(nprefix)) {
	Append(nprefix, "::");
	Append(nprefix, tprefix);
	Delete(tprefix);
	rname = nprefix;
      } else {
	rname = tprefix;
	Delete(nprefix);
      }
      rdecl = Copy(decl);
      Replaceall(rdecl, name, rname);
#ifdef SWIG_DEBUG
      Printf(stdout, "SWIG_name_make: use new name %s %s : %s %s\n", name, decl, rname, rdecl);
#endif
      decl = rdecl;
      Delete(name);
      name = rname;
    }
  }

  if (rename_hash || rename_list || namewarn_hash || namewarn_list) {
    Hash *rn = Swig_name_object_get(name_rename_hash(), prefix, name, decl);
    if (!rn || !name_match_nameobj(rn, n)) {
      rn = name_nameobj_lget(name_rename_list(), n, prefix, name, decl);
      if (rn) {
	String *sfmt = Getattr(rn, "sourcefmt");
	int fullname = GetFlag(rn, "fullname");
	if (fullname && prefix) {
	  String *sname = NewStringf("%s::%s", prefix, name);
	  Delete(name);
	  name = sname;
	  prefix = 0;
	}
	if (sfmt) {
	  String *sname = NewStringf(sfmt, name);
	  Delete(name);
	  name = sname;
	}
      }
    }
    if (rn) {
      String *newname = Getattr(rn, "name");
      int fullname = GetFlag(rn, "fullname");
      result = apply_rename(newname, fullname, prefix, name);
    }
    if (result && !Equal(result, name)) {
      /* operators in C++ allow aliases, we look for them */
      char *cresult = Char(result);
      if (cresult && (strncmp(cresult, "operator ", 9) == 0)) {
	String *nresult = Swig_name_make(n, prefix, result, decl, oldname);
	if (!Equal(nresult, result)) {
	  Delete(result);
	  result = nresult;
	} else {
	  Delete(nresult);
	}
      }
    }
    nname = result ? result : name;
    wrn = name_namewarn_get(n, prefix, nname, decl);
    if (wrn) {
      String *rename = Getattr(wrn, "rename");
      if (rename) {
	String *msg = Getattr(wrn, "name");
	int fullname = GetFlag(wrn, "fullname");
	if (result)
	  Delete(result);
	result = apply_rename(rename, fullname, prefix, name);
	if ((msg) && (Len(msg))) {
	  if (!Getmeta(nname, "already_warned")) {
	    if (n) {
	      SWIG_WARN_NODE_BEGIN(n);
	      Swig_warning(0, Getfile(n), Getline(n), "%s\n", msg);
	      SWIG_WARN_NODE_END(n);
	    } else {
	      Swig_warning(0, Getfile(name), Getline(name), "%s\n", msg);
	    }
	    Setmeta(nname, "already_warned", "1");
	  }
	}
      }
    }
  }
  if (!result || !Len(result)) {
    if (result)
      Delete(result);
    if (oldname) {
      result = NewString(oldname);
    } else {
      result = NewString(cname);
    }
  }
  Delete(name);

#ifdef SWIG_DEBUG
  Printf(stdout, "Swig_name_make: result  '%s' '%s'\n", cname, result);
#endif

  return result;
}

/* -----------------------------------------------------------------------------
 * void Swig_name_inherit()
 *
 * Inherit namewarn, rename, and feature objects
 * 
 * ----------------------------------------------------------------------------- */

void Swig_name_inherit(String *base, String *derived) {
  /*  Printf(stdout,"base = '%s', derived = '%s'\n", base, derived); */
  Swig_name_object_inherit(name_rename_hash(), base, derived);
  Swig_name_object_inherit(name_namewarn_hash(), base, derived);
  Swig_name_object_inherit(Swig_cparse_features(), base, derived);
}

/* -----------------------------------------------------------------------------
 * Swig_inherit_base_symbols()
 * ----------------------------------------------------------------------------- */

void Swig_inherit_base_symbols(List *bases) {
  if (bases) {
    Iterator s;
    for (s = First(bases); s.item; s = Next(s)) {
      Symtab *st = Getattr(s.item, "symtab");
      if (st) {
	Setfile(st, Getfile(s.item));
	Setline(st, Getline(s.item));
	Swig_symbol_inherit(st);
      }
    }
    Delete(bases);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_make_inherit_list()
 * ----------------------------------------------------------------------------- */

List *Swig_make_inherit_list(String *clsname, List *names, String *Namespaceprefix) {
  int i, ilen;
  String *derived;
  List *bases = NewList();

  if (Namespaceprefix)
    derived = NewStringf("%s::%s", Namespaceprefix, clsname);
  else
    derived = NewString(clsname);

  ilen = Len(names);
  for (i = 0; i < ilen; i++) {
    String *base;
    String *n = Getitem(names, i);
    /* Try to figure out where this symbol is */
    Node *s = Swig_symbol_clookup(n, 0);
    if (s) {
      while (s && (Strcmp(nodeType(s), "class") != 0)) {
	/* Not a class.  Could be a typedef though. */
	String *storage = Getattr(s, "storage");
	if (storage && (Strcmp(storage, "typedef") == 0)) {
	  String *nn = Getattr(s, "type");
	  s = Swig_symbol_clookup(nn, Getattr(s, "sym:symtab"));
	} else {
	  break;
	}
      }
      if (s && ((Strcmp(nodeType(s), "class") == 0) || (Strcmp(nodeType(s), "template") == 0))) {
	String *q = Swig_symbol_qualified(s);
	Append(bases, s);
	if (q) {
	  base = NewStringf("%s::%s", q, Getattr(s, "name"));
	  Delete(q);
	} else {
	  base = NewString(Getattr(s, "name"));
	}
      } else {
	base = NewString(n);
      }
    } else {
      base = NewString(n);
    }
    if (base) {
      Swig_name_inherit(base, derived);
      Delete(base);
    }
  }
  return bases;
}


/* -----------------------------------------------------------------------------
 * void Swig_name_str()
 *
 * Return a stringified version of a C/C++ symbol from a node.
 * The node passed in is expected to be a function, constructor, destructor or
 * variable. Some example return values:
 *   "MyNameSpace::MyTemplate<MyNameSpace::ABC >::~MyTemplate"
 *   "MyNameSpace::ABC::ABC"
 *   "MyNameSpace::ABC::constmethod"
 *   "MyNameSpace::ABC::variablename"
 * 
 * ----------------------------------------------------------------------------- */

String *Swig_name_str(Node *n) {
  String *qname;
  String *qualifier = Swig_symbol_qualified(n);
  String *name = Swig_scopename_last(Getattr(n, "name"));
  if (qualifier)
    qualifier = SwigType_namestr(qualifier);

  /* Very specific hack for template constructors/destructors */
  if (SwigType_istemplate(name)) {
    String *nodetype = nodeType(n);
    if (nodetype && (Equal(nodetype, "constructor") || Equal(nodetype, "destructor"))) {
      String *nprefix = NewStringEmpty();
      String *nlast = NewStringEmpty();
      String *tprefix;
      Swig_scopename_split(name, &nprefix, &nlast);
      tprefix = SwigType_templateprefix(nlast);
      Delete(nlast);
      Delete(name);
      name = tprefix;
    }
  }

  qname = NewString("");
  if (qualifier && Len(qualifier) > 0)
    Printf(qname, "%s::", qualifier);
  Printf(qname, "%s", SwigType_str(name, 0));

  Delete(name);
  Delete(qualifier);

  return qname;
}

/* -----------------------------------------------------------------------------
 * void Swig_name_decl()
 *
 * Return a stringified version of a C/C++ declaration without the return type.
 * The node passed in is expected to be a function, constructor, destructor or
 * variable. Some example return values:
 *   "MyNameSpace::MyTemplate<MyNameSpace::ABC >::~MyTemplate()"
 *   "MyNameSpace::ABC::ABC(int,double)"
 *   "MyNameSpace::ABC::constmethod(int) const"
 *   "MyNameSpace::ABC::variablename"
 * 
 * ----------------------------------------------------------------------------- */

String *Swig_name_decl(Node *n) {
  String *qname;
  String *decl;

  qname = Swig_name_str(n);

  if (checkAttribute(n, "kind", "variable"))
    decl = NewStringf("%s", qname);
  else
    decl = NewStringf("%s(%s)%s", qname, ParmList_errorstr(Getattr(n, "parms")), SwigType_isconst(Getattr(n, "decl")) ? " const" : "");

  Delete(qname);

  return decl;
}

/* -----------------------------------------------------------------------------
 * void Swig_name_fulldecl()
 *
 * Return a stringified version of a C/C++ declaration including the return type.
 * The node passed in is expected to be a function, constructor or destructor.
 * Some example return values:
 *   "MyNameSpace::MyTemplate<MyNameSpace::ABC >::~MyTemplate()"
 *   "MyNameSpace::ABC::ABC(int,double)"
 *   "int * MyNameSpace::ABC::constmethod(int) const"
 * 
 * ----------------------------------------------------------------------------- */

String *Swig_name_fulldecl(Node *n) {
  String *decl = Swig_name_decl(n);
  String *type = Getattr(n, "type");
  String *nodetype = nodeType(n);
  String *fulldecl;
  /* add on the return type */
  if (nodetype && (Equal(nodetype, "constructor") || Equal(nodetype, "destructor"))) {
    fulldecl = decl;
  } else {
    String *t = SwigType_str(type, 0);
    fulldecl = NewStringf("%s %s", t, decl);
    Delete(decl);
    Delete(t);
  }
  return fulldecl;
}

