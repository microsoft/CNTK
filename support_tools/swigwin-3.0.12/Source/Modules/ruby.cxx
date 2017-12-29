/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * ruby.cxx
 *
 * Ruby language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"
#include <ctype.h>
#include <string.h>
#include <limits.h>		/* for INT_MAX */

#define SWIG_PROTECTED_TARGET_METHODS 1

class RClass {
private:
  String *temp;

public:
  String *name;			/* class name (renamed) */
  String *cname;		/* original C class/struct name */
  String *mname;		/* Mangled name */

  /**
   * The C variable name used in the SWIG-generated wrapper code to refer to
   * this class; usually it is of the form "SwigClassXXX.klass", where SwigClassXXX
   * is a swig_class struct instance and klass is a member of that struct.
   */
  String *vname;

  /**
   * The C variable name used in the SWIG-generated wrapper code to refer to
   * the module that implements this class's methods (when we're trying to
   * support C++ multiple inheritance). Usually it is of the form
   * "SwigClassClassName.mImpl", where SwigClassXXX is a swig_class struct instance
   * and mImpl is a member of that struct.
   */
  String *mImpl;

  String *type;
  String *prefix;
  String *init;


  int constructor_defined;
  int destructor_defined;

   RClass() {
    temp = NewString("");
    name = NewString("");
    cname = NewString("");
    mname = NewString("");
    vname = NewString("");
    mImpl = NewString("");
    type = NewString("");
    prefix = NewString("");
    init = NewString("");
    constructor_defined = 0;
    destructor_defined = 0;
  }
  
  ~RClass() {
    Delete(name);
    Delete(cname);
    Delete(vname);
    Delete(mImpl);
    Delete(mname);
    Delete(type);
    Delete(prefix);
    Delete(init);
    Delete(temp);
  }

  void set_name(const_String_or_char_ptr cn, const_String_or_char_ptr rn, const_String_or_char_ptr valn) {
    /* Original C/C++ class (or struct) name */
    Clear(cname);
    Append(cname, cn);

    /* Mangled name */
    Delete(mname);
    mname = Swig_name_mangle(cname);

    /* Renamed class name */
    Clear(name);
    Append(name, valn);

    /* Variable name for the VALUE that refers to the Ruby Class object */
    Clear(vname);
    Printf(vname, "SwigClass%s.klass", name);

    /* Variable name for the VALUE that refers to the Ruby Class object */
    Clear(mImpl);
    Printf(mImpl, "SwigClass%s.mImpl", name);

    /* Prefix */
    Clear(prefix);
    Printv(prefix, (rn ? rn : cn), "_", NIL);
  }

  char *strip(const_String_or_char_ptr s) {
    Clear(temp);
    Append(temp, s);
    if (Strncmp(s, prefix, Len(prefix)) == 0) {
      Replaceall(temp, prefix, "");
    }
    return Char(temp);
  }
};


/* flags for the make_autodoc function */
enum autodoc_t {
  AUTODOC_CLASS,
  AUTODOC_CTOR,
  AUTODOC_DTOR,
  AUTODOC_STATICFUNC,
  AUTODOC_FUNC,
  AUTODOC_METHOD,
  AUTODOC_GETTER,
  AUTODOC_SETTER,
  AUTODOC_NONE
};

static const char *usage = "\
Ruby Options (available with -ruby)\n\
     -autorename     - Enable renaming of classes and methods to follow Ruby coding standards\n\
     -cppcast        - Enable C++ casting operators (default)\n\
     -globalmodule   - Wrap everything into the global module\n\
     -initname <name>- Set entry function to Init_<name> (used by `require')\n\
     -minherit       - Attempt to support multiple inheritance\n\
     -noautorename   - Disable renaming of classes and methods (default)\n\
     -nocppcast      - Disable C++ casting operators, useful for generating bugs\n\
     -prefix <name>  - Set a prefix <name> to be prepended to all names\n\
";


#define RCLASS(hash, name) (RClass*)(Getattr(hash, name) ? Data(Getattr(hash, name)) : 0)
#define SET_RCLASS(hash, name, klass) Setattr(hash, name, NewVoid(klass, 0))


class RUBY:public Language {
private:

  String *module;
  String *modvar;
  String *feature;
  String *prefix;
  int current;
  Hash *classes;		/* key=cname val=RClass */
  RClass *klass;		/* Currently processing class */
  Hash *special_methods;	/* Python style special method name table */

  File *f_directors;
  File *f_directors_h;
  File *f_directors_helpers;
  File *f_begin;
  File *f_runtime;
  File *f_runtime_h;
  File *f_header;
  File *f_wrappers;
  File *f_init;
  File *f_initbeforefunc;

  bool useGlobalModule;
  bool multipleInheritance;

  // Wrap modes
  enum WrapperMode {
    NO_CPP,
    MEMBER_FUNC,
    CONSTRUCTOR_ALLOCATE,
    CONSTRUCTOR_INITIALIZE,
    DESTRUCTOR,
    MEMBER_VAR,
    CLASS_CONST,
    STATIC_FUNC,
    STATIC_VAR
  };

  /* ------------------------------------------------------------
   * autodoc level declarations
   * ------------------------------------------------------------ */

  enum autodoc_l {
    NO_AUTODOC = -2,		// no autodoc
    STRING_AUTODOC = -1,	// use provided string
    NAMES_AUTODOC = 0,		// only parameter names
    TYPES_AUTODOC = 1,		// parameter names and types
    EXTEND_AUTODOC = 2,		// extended documentation and parameter names
    EXTEND_TYPES_AUTODOC = 3	// extended documentation and parameter types + names
  };

  autodoc_t last_mode;
  String*   last_autodoc;

  autodoc_l autodoc_level(String *autodoc) {
    autodoc_l dlevel = NO_AUTODOC;
    char *c = Char(autodoc);
    if (c) {
      if (isdigit(c[0])) {
	dlevel = (autodoc_l) atoi(c);
      } else {
	if (strcmp(c, "extended") == 0) {
	  dlevel = EXTEND_AUTODOC;
	} else {
	  dlevel = STRING_AUTODOC;
	}
      }
    }
    return dlevel;
  }



  /* ------------------------------------------------------------
   * have_docstring()
   *    Check if there is a docstring directive and it has text,
   *    or there is an autodoc flag set
   * ------------------------------------------------------------ */

  bool have_docstring(Node *n) {
    String *str = Getattr(n, "feature:docstring");
    return (str && Len(str) > 0) || (Getattr(n, "feature:autodoc") && !GetFlag(n, "feature:noautodoc"));
  }

  /* ------------------------------------------------------------
   * docstring()
   *    Get the docstring text, stripping off {} if necessary,
   *    and enclose in triple double quotes.  If autodoc is also
   *    set then it will build a combined docstring.
   * ------------------------------------------------------------ */

  String *docstring(Node *n, autodoc_t ad_type) {

    String *str = Getattr(n, "feature:docstring");
    bool have_ds = (str && Len(str) > 0);
    bool have_auto = (Getattr(n, "feature:autodoc") && !GetFlag(n, "feature:noautodoc"));
    String *autodoc = NULL;
    String *doc = NULL;

    if (have_ds) {
      char *t = Char(str);
      if (*t == '{') {
	Delitem(str, 0);
	Delitem(str, DOH_END);
      }
    }

    if (have_auto) {
      autodoc = make_autodoc(n, ad_type);
      have_auto = (autodoc && Len(autodoc) > 0);
    }
    // If there is more than one line then make docstrings like this:
    //
    //      This is line1
    //      And here is line2 followed by the rest of them
    //
    // otherwise, put it all on a single line
    //
    if (have_auto && have_ds) {	// Both autodoc and docstring are present
      doc = NewString("");
      Printv(doc, "\n", autodoc, "\n", str, NIL);
    } else if (!have_auto && have_ds) {	// only docstring
      if (Strchr(str, '\n') == 0) {
	doc = NewString(str);
      } else {
	doc = NewString("");
	Printv(doc, str, NIL);
      }
    } else if (have_auto && !have_ds) {	// only autodoc
      if (Strchr(autodoc, '\n') == 0) {
	doc = NewStringf("%s", autodoc);
      } else {
	doc = NewString("");
	Printv(doc, "\n", autodoc, NIL);
      }
    } else
      doc = NewString("");

    // Save the generated strings in the parse tree in case they are used later
    // by post processing tools
    Setattr(n, "ruby:docstring", doc);
    Setattr(n, "ruby:autodoc", autodoc);
    return doc;
  }

  /* -----------------------------------------------------------------------------
   * addMissingParameterNames()
   *  For functions that have not had nameless parameters set in the Language class.
   *
   * Inputs: 
   *   plist - entire parameter list
   *   arg_offset - argument number for first parameter
   * Side effects:
   *   The "lname" attribute in each parameter in plist will be contain a parameter name
   * ----------------------------------------------------------------------------- */

  void addMissingParameterNames(ParmList *plist, int arg_offset) {
    Parm *p = plist;
    int i = arg_offset;
    while (p) {
      if (!Getattr(p, "lname")) {
	String *pname = Swig_cparm_name(p, i);
	Delete(pname);
      }
      i++;
      p = nextSibling(p);
    }
  }

  /* ------------------------------------------------------------
   * make_autodocParmList()
   *   Generate the documentation for the function parameters
   * ------------------------------------------------------------ */

  String *make_autodocParmList(Node *n, bool showTypes) {
    String *doc = NewString("");
    String *pdocs = 0;
    ParmList *plist = CopyParmList(Getattr(n, "parms"));
    Parm *p;
    Parm *pnext;
    int lines = 0;
    int start_arg_num = is_wrapping_class() ? 1 : 0;
    const int maxwidth = 80;

    addMissingParameterNames(plist, start_arg_num); // for $1_name substitutions done in Swig_typemap_attach_parms

    Swig_typemap_attach_parms("in", plist, 0);
    Swig_typemap_attach_parms("doc", plist, 0);

    if (Strcmp(ParmList_protostr(plist), "void") == 0) {
      //No parameters actually
      return doc;
    }

    for (p = plist; p; p = pnext) {

      String *tm = Getattr(p, "tmap:in");
      if (tm) {
	pnext = Getattr(p, "tmap:in:next");
	if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	  continue;
	}
      } else {
	pnext = nextSibling(p);
      }

      String *name = 0;
      String *type = 0;
      String *value = 0;
      String *pdoc = Getattr(p, "tmap:doc");
      if (pdoc) {
	name = Getattr(p, "tmap:doc:name");
	type = Getattr(p, "tmap:doc:type");
	value = Getattr(p, "tmap:doc:value");
      }

      // Note: the generated name should be consistent with that in kwnames[]
      name = name ? name : Getattr(p, "name");
      name = name ? name : Getattr(p, "lname");
      name = Swig_name_make(p, 0, name, 0, 0); // rename parameter if a keyword

      type = type ? type : Getattr(p, "type");
      value = value ? value : Getattr(p, "value");

      if (SwigType_isvarargs(type))
	break;

      // Skip the 'self' parameter which in ruby is implicit
      if ( Cmp(name, "self") == 0 )
	continue;

      // Make __p parameters just p (as used in STL)
      Replace( name, "__", "", DOH_REPLACE_FIRST );

      if (Len(doc)) {
	// add a comma to the previous one if any
	Append(doc, ", ");

	// Do we need to wrap a long line?
	if ((Len(doc) - lines * maxwidth) > maxwidth) {
	  Printf(doc, "\n%s", tab4);
	  lines += 1;
	}
      }

      // Do the param type too?
      Node *nn = classLookup(Getattr(p, "type"));
      String *type_str = nn ? Copy(Getattr(nn, "sym:name")) : SwigType_str(type, 0);
      if (showTypes)
	Printf(doc, "%s ", type_str);

      Append(doc, name);
      if (pdoc) {
	if (!pdocs)
	  pdocs = NewString("Parameters:\n");
	Printf(pdocs, "    %s.\n", pdoc);
      }

      if (value) {
	String *new_value = convertValue(value, Getattr(p, "type"));
	if (new_value) {
	  value = new_value;
	} else {
	  Node *lookup = Swig_symbol_clookup(value, 0);
	  if (lookup)
	    value = Getattr(lookup, "sym:name");
	}
	Printf(doc, "=%s", value);
      }
      Delete(type_str);
      Delete(name);
    }
    if (pdocs)
      Setattr(n, "feature:pdocs", pdocs);
    Delete(plist);
    return doc;
  }

  /* ------------------------------------------------------------
   * make_autodoc()
   *    Build a docstring for the node, using parameter and other
   *    info in the parse tree.  If the value of the autodoc
   *    attribute is "0" then do not include parameter types, if
   *    it is "1" (the default) then do.  If it has some other
   *    value then assume it is supplied by the extension writer
   *    and use it directly.
   * ------------------------------------------------------------ */

  String *make_autodoc(Node *n, autodoc_t ad_type) {
    int extended = 0;
    // If the function is overloaded then this funciton is called
    // for the last one.  Rewind to the first so the docstrings are
    // in order.
    while (Getattr(n, "sym:previousSibling"))
      n = Getattr(n, "sym:previousSibling");

    Node *pn = Swig_methodclass(n);
    String* super_names = NewString(""); 
    String* class_name = Getattr(pn, "sym:name") ; 

    if ( !class_name ) {
      class_name = NewString("");
    } else {
      class_name = Copy(class_name);
      List *baselist = Getattr(pn, "bases");
      if (baselist && Len(baselist)) {
	Iterator base = First(baselist);
	while (base.item && GetFlag(base.item, "feature:ignore")) {
	  base = Next(base);
	}

	int count = 0;
	for ( ;base.item; ++count) {
	  if ( count ) Append(super_names, ", ");
	  String *basename = Getattr(base.item, "sym:name");

	  String* basenamestr = NewString(basename);
	  Node* parent = parentNode(base.item);
	  while (parent)
	  {
	    String *parent_name = Copy( Getattr(parent, "sym:name") );
	    if ( !parent_name ) {
	      Node* mod = Getattr(parent, "module");
	      if ( mod )
		parent_name = Copy( Getattr(mod, "name") );
	      if ( parent_name )
		(Char(parent_name))[0] = (char)toupper((Char(parent_name))[0]);
	    }
	    if ( parent_name ) {
	      Insert(basenamestr, 0, "::");
	      Insert(basenamestr, 0, parent_name);
	      Delete(parent_name);
	    }
	    parent = parentNode(parent);
	  }

	  Append(super_names, basenamestr );
	  Delete(basenamestr);
	  base = Next(base);
	}
      }
    }
    String* full_name;
    if ( module ) {
      full_name = NewString(module);
      if (Len(class_name) > 0)
       	Append(full_name, "::");
    }
    else
      full_name = NewString("");
    Append(full_name, class_name);

    String* symname = Getattr(n, "sym:name");
    if ( Getattr( special_methods, symname ) )
      symname = Getattr( special_methods, symname );

    String* methodName = NewString(full_name);
    Append(methodName, symname);


    // Each overloaded function will try to get documented,
    // so we keep the name of the last overloaded function and its type.
    // Documenting just from functionWrapper() is not possible as
    // sym:name has already been changed to include the class name
    if ( last_mode == ad_type && Cmp(methodName, last_autodoc) == 0 ) {
      Delete(full_name);
      Delete(class_name);
      Delete(super_names);
      Delete(methodName);
      return NewString("");
    }


    last_mode    = ad_type;
    last_autodoc = Copy(methodName);

    String *doc = NewString("/*\n");
    int counter = 0;
    bool skipAuto = false;
    Node* on = n;
    for ( ; n; ++counter ) {
      String *type_str = NULL;
      skipAuto = false;
      bool showTypes = false;
      String *autodoc = Getattr(n, "feature:autodoc");
      autodoc_l dlevel = autodoc_level(autodoc);
      switch (dlevel) {
      case NO_AUTODOC:
	break;
      case NAMES_AUTODOC:
	showTypes = false;
	break;
      case TYPES_AUTODOC:
	showTypes = true;
	break;
      case EXTEND_AUTODOC:
	extended = 1;
	showTypes = false;
	break;
      case EXTEND_TYPES_AUTODOC:
	extended = 1;
	showTypes = true;
	break;
      case STRING_AUTODOC:
	skipAuto = true;
	break;
      }

      SwigType *type = Getattr(n, "type");

      if (type) {
	if (Strcmp(type, "void") == 0) {
	  type_str = NULL;
	} else {
	  SwigType *qt = SwigType_typedef_resolve_all(type);
	  if (SwigType_isenum(qt)) {
	    type_str = NewString("int");
	  } else {
	    Node *nn = classLookup(type);
	    type_str = nn ? Copy(Getattr(nn, "sym:name")) : SwigType_str(type, 0);
	  }
	}
      }

      if (counter == 0) {
	switch (ad_type) {
	case AUTODOC_CLASS:
	  Printf(doc, "  Document-class: %s", full_name);
	  if ( Len(super_names) > 0 )
	    Printf( doc, " < %s", super_names);
	  Append(doc, "\n\n");
	  break;
	case AUTODOC_CTOR:
 	  Printf(doc, "  Document-method: %s.new\n\n", full_name);
	  break;

	case AUTODOC_DTOR:
	  break;

	case AUTODOC_STATICFUNC:
 	  Printf(doc, "  Document-method: %s.%s\n\n", full_name, symname);
	  break;

	case AUTODOC_FUNC:
	case AUTODOC_METHOD:
	case AUTODOC_GETTER:
 	  Printf(doc, "  Document-method: %s.%s\n\n", full_name, symname);
	  break;
	case AUTODOC_SETTER:
 	  Printf(doc, "  Document-method: %s.%s=\n\n", full_name, symname);
	  break;
	case AUTODOC_NONE:
	  break;
	}
      }

      if (skipAuto) {
	if ( counter == 0 ) Printf(doc, "  call-seq:\n");
	switch( ad_type )
	  {
	  case AUTODOC_STATICFUNC:
	  case AUTODOC_FUNC:
	  case AUTODOC_METHOD:
	  case AUTODOC_GETTER:
	    {
	      String *paramList = make_autodocParmList(n, showTypes);
	      if (Len(paramList))
		Printf(doc, "    %s(%s)", symname, paramList);
	      else
		Printf(doc, "    %s", symname);
	      if (type_str)
		Printf(doc, " -> %s", type_str);
	      break;
	    }
	  case AUTODOC_SETTER:
	    {
	      Printf(doc, "    %s=(x)", symname);
	      if (type_str)
	       	Printf(doc, " -> %s", type_str);
	      break;
	    }
	  default:
	    break;
	  }
      } else {
	switch (ad_type) {
	case AUTODOC_CLASS:
	  {
	    // Only do the autodoc if there isn't a docstring for the class
	    String *str = Getattr(n, "feature:docstring");
	    if (counter == 0 && (str == 0 || Len(str) == 0)) {
	      if (CPlusPlus) {
		Printf(doc, "  Proxy of C++ %s class", full_name);
	      } else {
		Printf(doc, "  Proxy of C %s struct", full_name);
	      }
	    }
	  }
	  break;
	case AUTODOC_CTOR:
	  if (counter == 0)
	    Printf(doc, "  call-seq:\n");
	  if (Strcmp(class_name, symname) == 0) {
	    String *paramList = make_autodocParmList(n, showTypes);
	    if (Len(paramList))
	      Printf(doc, "    %s.new(%s)", class_name, paramList);
	    else
	      Printf(doc, "    %s.new", class_name);
	  } else {
	    Printf(doc, "    %s.new(%s)", class_name, make_autodocParmList(n, showTypes));
	  }
	  break;

	case AUTODOC_DTOR:
	  break;

	case AUTODOC_STATICFUNC:
	case AUTODOC_FUNC:
	case AUTODOC_METHOD:
	case AUTODOC_GETTER:
	  {
	    if (counter == 0)
	      Printf(doc, "  call-seq:\n");
	    String *paramList = make_autodocParmList(n, showTypes);
	    if (Len(paramList))
	      Printf(doc, "    %s(%s)", symname, paramList);
	    else
	      Printf(doc, "    %s", symname);
	    if (type_str)
	      Printf(doc, " -> %s", type_str);
	    break;
	  }
	case AUTODOC_SETTER:
	  {
	    Printf(doc, "  call-seq:\n");
	    Printf(doc, "    %s=(x)", symname);
	    if (type_str)
	      Printf(doc, " -> %s", type_str);
	    break;
	  }
	case AUTODOC_NONE:
	  break;
	}
      }

      // if it's overloaded then get the next decl and loop around again
      n = Getattr(n, "sym:nextSibling");
      if (n)
	Append(doc, "\n");
      Delete(type_str);
    }

    Printf(doc, "\n\n");
    if (!skipAuto) {
      switch (ad_type) {
      case AUTODOC_CLASS:
      case AUTODOC_DTOR:
	break;
      case AUTODOC_CTOR:
	Printf(doc, "Class constructor.\n");
	break;
      case AUTODOC_STATICFUNC:
	Printf(doc, "A class method.\n");
	break;
      case AUTODOC_FUNC:
	Printf(doc, "A module function.\n");
	break;
      case AUTODOC_METHOD:
	Printf(doc, "An instance method.\n");
	break;
      case AUTODOC_GETTER:
	Printf(doc, "Get value of attribute.\n");
	break;
      case AUTODOC_SETTER:
	Printf(doc, "Set new value for attribute.\n");
	break;
      case AUTODOC_NONE:
	break;
      }
    }


    n = on;
    while ( n ) {
      String *autodoc = Getattr(n, "feature:autodoc");
      autodoc_l dlevel = autodoc_level(autodoc);

      switch (dlevel) {
      case NO_AUTODOC:
      case NAMES_AUTODOC:
      case TYPES_AUTODOC:
	extended = 0;
	break;
      case STRING_AUTODOC:
	extended = 2;
	Replaceall( autodoc, "$class", class_name );
	Printv(doc, autodoc, ".", NIL);
	break;
      case EXTEND_AUTODOC:
      case EXTEND_TYPES_AUTODOC:
	extended = 1;
	break;
      }


      if (extended) {
	String *pdocs = Getattr(n, "feature:pdocs");
	if (pdocs) {
	  Printv(doc, "\n\n", pdocs, NULL);
	  break;
	}
	if ( extended == 2 ) break;
      }
      n = Getattr(n, "sym:nextSibling");
    }

    Append(doc, "\n*/\n");
    Delete(full_name);
    Delete(class_name);
    Delete(super_names);
    Delete(methodName);

    return doc;
  }

  /* ------------------------------------------------------------
   * convertValue()
   *    Check if string v can be a Ruby value literal,
   *    (eg. number or string), or translate it to a Ruby literal.
   * ------------------------------------------------------------ */
  String *convertValue(String *v, SwigType *t) {
    if (v && Len(v) > 0) {
      char fc = (Char(v))[0];
      if (('0' <= fc && fc <= '9') || '\'' == fc || '"' == fc) {
	/* number or string (or maybe NULL pointer) */
	if (SwigType_ispointer(t) && Strcmp(v, "0") == 0)
	  return NewString("None");
	else
	  return v;
      }
      if (Strcmp(v, "NULL") == 0 || Strcmp(v, "nullptr") == 0)
	return SwigType_ispointer(t) ? NewString("nil") : NewString("0");
      if (Strcmp(v, "true") == 0 || Strcmp(v, "TRUE") == 0)
	return NewString("True");
      if (Strcmp(v, "false") == 0 || Strcmp(v, "FALSE") == 0)
	return NewString("False");
    }
    return 0;
  }

public:

  /* ---------------------------------------------------------------------
   * RUBY()
   *
   * Initialize member data
   * --------------------------------------------------------------------- */
  RUBY() :
    module(0),
    modvar(0),
    feature(0),
    prefix(0),
    current(0),
    classes(0),
    klass(0),
    special_methods(0),
    f_directors(0),
    f_directors_h(0),
    f_directors_helpers(0),
    f_begin(0),
    f_runtime(0),
    f_runtime_h(0),
    f_header(0),
    f_wrappers(0),
    f_init(0),
    f_initbeforefunc(0),
    useGlobalModule(false),
    multipleInheritance(false),
    last_mode(AUTODOC_NONE),
    last_autodoc(NewString("")) {
      current = NO_CPP;
      director_prot_ctor_code = NewString("");
      Printv(director_prot_ctor_code,
          "if ( $comparison ) { /* subclassed */\n",
          "  $director_new \n",
          "} else {\n", "  rb_raise(rb_eRuntimeError,\"accessing abstract class or protected constructor\"); \n", "  return Qnil;\n", "}\n", NIL);
      director_multiple_inheritance = 0;
      director_language = 1;
    }

  /* ---------------------------------------------------------------------
   * main()
   *
   * Parse command line options and initializes variables.
   * --------------------------------------------------------------------- */
  
  virtual void main(int argc, char *argv[]) {

    int cppcast = 1;
    int autorename = 0;

    /* Set location of SWIG library */
    SWIG_library_directory("ruby");

    /* Look for certain command line options */
    for (int i = 1; i < argc; i++) {
      if (argv[i]) {
	if (strcmp(argv[i], "-initname") == 0) {
	  if (argv[i + 1]) {
	    char *name = argv[i + 1];
	    feature = NewString(name);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	}
	else if (strcmp(argv[i], "-feature") == 0) {
	  fprintf( stderr, "Warning: Ruby -feature option is deprecated, "
		   "please use -initname instead.\n");
	  if (argv[i + 1]) {
	    char *name = argv[i + 1];
	    feature = NewString(name);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-globalmodule") == 0) {
	  useGlobalModule = true;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-minherit") == 0) {
	  multipleInheritance = true;
	  director_multiple_inheritance = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-cppcast") == 0) {
	  cppcast = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-nocppcast") == 0) {
	  cppcast = 0;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-autorename") == 0) {
	  autorename = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-noautorename") == 0) {
	  autorename = 0;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-prefix") == 0) {
	  if (argv[i + 1]) {
	    char *name = argv[i + 1];
	    prefix = NewString(name);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if (strcmp(argv[i], "-help") == 0) {
	  Printf(stdout, "%s\n", usage);
	}
      }
    }

    if (cppcast) {
      /* Turn on cppcast mode */
      Preprocessor_define((DOH *) "SWIG_CPLUSPLUS_CAST", 0);
    }

    if (autorename) {
      /* Turn on the autorename mode */
      Preprocessor_define((DOH *) "SWIG_RUBY_AUTORENAME", 0);
    }

    /* Add a symbol to the parser for conditional compilation */
    Preprocessor_define("SWIGRUBY 1", 0);

    /* Add typemap definitions */
    SWIG_typemap_lang("ruby");
    SWIG_config_file("ruby.swg");
    allow_overloading();
  }

  /**
   * Generate initialization code to define the Ruby module(s),
   * accounting for nested modules as necessary.
   */
  void defineRubyModule() {
    List *modules = Split(module, ':', INT_MAX);
    if (modules != 0 && Len(modules) > 0) {
      String *mv = 0;
      Iterator m;
      m = First(modules);
      while (m.item) {
	if (Len(m.item) > 0) {
	  if (mv != 0) {
	    Printv(f_init, tab4, modvar, " = rb_define_module_under(", modvar, ", \"", m.item, "\");\n", NIL);
	  } else {
	    Printv(f_init, tab4, modvar, " = rb_define_module(\"", m.item, "\");\n", NIL);
	    mv = NewString(modvar);
	  }
	}
	m = Next(m);
      }
      Delete(mv);
      Delete(modules);
    }
  }

  void registerMagicMethods() {

    special_methods = NewHash();

    /* Python->Ruby style special method name. */
    /* Basic */
    Setattr(special_methods, "__repr__", "inspect");
    Setattr(special_methods, "__str__", "to_s");
    Setattr(special_methods, "__cmp__", "<=>");
    Setattr(special_methods, "__hash__", "hash");
    Setattr(special_methods, "__nonzero__", "nonzero?");

    /* Callable */
    Setattr(special_methods, "__call__", "call");

    /* Collection */
    Setattr(special_methods, "__len__", "length");
    Setattr(special_methods, "__getitem__", "[]");
    Setattr(special_methods, "__setitem__", "[]=");

    /* Operators */
    Setattr(special_methods, "__add__", "+");
    Setattr(special_methods, "__pos__", "+@");
    Setattr(special_methods, "__sub__", "-");
    Setattr(special_methods, "__neg__", "-@");
    Setattr(special_methods, "__mul__", "*");
    Setattr(special_methods, "__div__", "/");
    Setattr(special_methods, "__mod__", "%");
    Setattr(special_methods, "__lshift__", "<<");
    Setattr(special_methods, "__rshift__", ">>");
    Setattr(special_methods, "__and__", "&");
    Setattr(special_methods, "__or__", "|");
    Setattr(special_methods, "__xor__", "^");
    Setattr(special_methods, "__invert__", "~");
    Setattr(special_methods, "__lt__", "<");
    Setattr(special_methods, "__le__", "<=");
    Setattr(special_methods, "__gt__", ">");
    Setattr(special_methods, "__ge__", ">=");
    Setattr(special_methods, "__eq__", "==");

    /* Other numeric */
    Setattr(special_methods, "__divmod__", "divmod");
    Setattr(special_methods, "__pow__", "**");
    Setattr(special_methods, "__abs__", "abs");
    Setattr(special_methods, "__int__", "to_i");
    Setattr(special_methods, "__float__", "to_f");
    Setattr(special_methods, "__coerce__", "coerce");
  }

  /* ---------------------------------------------------------------------
   * top()
   * --------------------------------------------------------------------- */

  virtual int top(Node *n) {

    String *mod_docstring = NULL;

    /**
     * See if any Ruby module options have been specified as options
     * to the %module directive.
     */
    Node *swigModule = Getattr(n, "module");
    if (swigModule) {
      Node *options = Getattr(swigModule, "options");
      if (options) {
	if (Getattr(options, "directors")) {
	  allow_directors();
	}
	if (Getattr(options, "dirprot")) {
	  allow_dirprot();
	}
	if (Getattr(options, "ruby_globalmodule")) {
	  useGlobalModule = true;
	}
	if (Getattr(options, "ruby_minherit")) {
	  multipleInheritance = true;
	  director_multiple_inheritance = 1;
	}
	mod_docstring = Getattr(options, "docstring");
      }
    }

    /* Set comparison with none for ConstructorToFunction */


    setSubclassInstanceCheck(NewStringf("strcmp(rb_obj_classname(self), classname) != 0"));
    // setSubclassInstanceCheck(NewString("CLASS_OF(self) != cFoo.klass"));

    /* Initialize all of the output files */
    String *outfile = Getattr(n, "outfile");
    String *outfile_h = Getattr(n, "outfile_h");

    if (!outfile) {
      Printf(stderr, "Unable to determine outfile\n");
      SWIG_exit(EXIT_FAILURE);
    }

    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      SWIG_exit(EXIT_FAILURE);
    }

    f_runtime = NewString("");
    f_init = NewString("");
    f_header = NewString("");
    f_wrappers = NewString("");
    f_directors_h = NewString("");
    f_directors = NewString("");
    f_directors_helpers = NewString("");
    f_initbeforefunc = NewString("");

    if (directorsEnabled()) {
      if (!outfile_h) {
        Printf(stderr, "Unable to determine outfile_h\n");
        SWIG_exit(EXIT_FAILURE);
      }
      f_runtime_h = NewFile(outfile_h, "w", SWIG_output_files());
      if (!f_runtime_h) {
	FileErrorDisplay(outfile_h);
	SWIG_exit(EXIT_FAILURE);
      }
    }

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("header", f_header);
    Swig_register_filebyname("wrapper", f_wrappers);
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", f_init);
    Swig_register_filebyname("director", f_directors);
    Swig_register_filebyname("director_h", f_directors_h);
    Swig_register_filebyname("director_helpers", f_directors_helpers);
    Swig_register_filebyname("initbeforefunc", f_initbeforefunc);

    modvar = 0;
    current = NO_CPP;
    klass = 0;
    classes = NewHash();

    registerMagicMethods();

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGRUBY\n#define SWIGRUBY\n#endif\n\n");

    if (directorsEnabled()) {
      Printf(f_runtime, "#define SWIG_DIRECTORS\n");
    }

    Printf(f_runtime, "\n");

    /* typedef void *VALUE */
    SwigType *value = NewSwigType(T_VOID);
    SwigType_add_pointer(value);
    SwigType_typedef(value, "VALUE");
    Delete(value);

    /* Set module name */
    set_module(Char(Getattr(n, "name")));

    if (directorsEnabled()) {
      /* Build a version of the module name for use in a C macro name. */
      String *module_macro = Copy(module);
      Replaceall(module_macro, "::", "__");

      Swig_banner(f_directors_h);
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "#ifndef SWIG_%s_WRAP_H_\n", module_macro);
      Printf(f_directors_h, "#define SWIG_%s_WRAP_H_\n\n", module_macro);
      Printf(f_directors_h, "namespace Swig {\n");
      Printf(f_directors_h, "  class Director;\n");
      Printf(f_directors_h, "}\n\n");

      Printf(f_directors_helpers, "/* ---------------------------------------------------\n");
      Printf(f_directors_helpers, " * C++ director class helpers\n");
      Printf(f_directors_helpers, " * --------------------------------------------------- */\n\n");

      Printf(f_directors, "\n\n");
      Printf(f_directors, "/* ---------------------------------------------------\n");
      Printf(f_directors, " * C++ director class methods\n");
      Printf(f_directors, " * --------------------------------------------------- */\n\n");
      if (outfile_h) {
	String *filename = Swig_file_filename(outfile_h);
	Printf(f_directors, "#include \"%s\"\n\n", filename);
	Delete(filename);
      }

      Delete(module_macro);
    }

    Printf(f_header, "#define SWIG_init    Init_%s\n", feature);
    Printf(f_header, "#define SWIG_name    \"%s\"\n\n", module);

    if (mod_docstring) {
      if (Len(mod_docstring)) {
	Printf(f_header, "/*\n  Document-module: %s\n\n%s\n*/\n", module, mod_docstring);
      }
      Delete(mod_docstring);
      mod_docstring = NULL;
    }

    Printf(f_header, "static VALUE %s;\n", modvar);

    /* Start generating the initialization function */
    String* docs = docstring(n, AUTODOC_CLASS);
    Printf(f_init, "/*\n%s\n*/", docs );
    Printv(f_init, "\n", "#ifdef __cplusplus\n", "extern \"C\"\n", "#endif\n", "SWIGEXPORT void Init_", feature, "(void) {\n", "size_t i;\n", "\n", NIL);

    Printv(f_init, tab4, "SWIG_InitRuntime();\n", NIL);

    if (!useGlobalModule)
      defineRubyModule();

    Printv(f_init, "\n", "SWIG_InitializeModule(0);\n", "for (i = 0; i < swig_module.size; i++) {\n", "SWIG_define_class(swig_module.types[i]);\n", "}\n", NIL);
    Printf(f_init, "\n");

    /* Initialize code to keep track of objects */
    Printf(f_init, "SWIG_RubyInitializeTrackings();\n");

    Language::top(n);

    if (directorsEnabled()) {
      // Insert director runtime into the f_runtime file (make it occur before %header section)
      Swig_insert_file("director_common.swg", f_runtime);
      Swig_insert_file("director.swg", f_runtime);
    }

    /* Finish off our init function */
    Printf(f_init, "}\n");
    SwigType_emit_type_table(f_runtime, f_wrappers);

    /* Close all of the files */
    Dump(f_runtime, f_begin);
    Dump(f_header, f_begin);

    if (directorsEnabled()) {
      Dump(f_directors_helpers, f_begin);
      Dump(f_directors, f_begin);
      Dump(f_directors_h, f_runtime_h);
      Printf(f_runtime_h, "\n");
      Printf(f_runtime_h, "#endif\n");
      Delete(f_runtime_h);
    }

    Dump(f_wrappers, f_begin);
    Dump(f_initbeforefunc, f_begin);
    Wrapper_pretty_print(f_init, f_begin);

    Delete(f_header);
    Delete(f_wrappers);
    Delete(f_init);
    Delete(f_initbeforefunc);
    Delete(f_runtime);
    Delete(f_begin);

    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------------
   * importDirective()
   * ----------------------------------------------------------------------------- */

  virtual int importDirective(Node *n) {
    String *modname = Getattr(n, "module");
    if (modname) {
      if (prefix) {
	Insert(modname, 0, prefix);
      }

      List *modules = Split(modname, ':', INT_MAX);
      if (modules && Len(modules) > 0) {
	modname = NewString("");
	String *last = NULL;
	Iterator m = First(modules);
	while (m.item) {
	  if (Len(m.item) > 0) {
	    if (last) {
	      Append(modname, "/");
	    }
	    Append(modname, m.item);
	    last = m.item;
	  }
	  m = Next(m);
	}
	Printf(f_init, "rb_require(\"%s\");\n", modname);
	Delete(modname);
      }
      Delete(modules);
    }
    return Language::importDirective(n);
  }

  /* ---------------------------------------------------------------------
   * set_module(const char *mod_name)
   *
   * Sets the module name.  Does nothing if it's already set (so it can
   * be overridden as a command line option).
   *---------------------------------------------------------------------- */

  void set_module(const char *s) {
    String *mod_name = NewString(s);
    if (module == 0) {
      /* Start with the empty string */
      module = NewString("");

      if (prefix) {
	Insert(mod_name, 0, prefix);
      }

      /* Account for nested modules */
      List *modules = Split(mod_name, ':', INT_MAX);
      if (modules != 0 && Len(modules) > 0) {
	String *last = 0;
	Iterator m = First(modules);
	while (m.item) {
	  if (Len(m.item) > 0) {
	    String *cap = NewString(m.item);
	    (Char(cap))[0] = (char)toupper((Char(cap))[0]);
	    if (last != 0) {
	      Append(module, "::");
	    }
	    Append(module, cap);
	    last = m.item;
	  }
	  m = Next(m);
	}
	if (last) {
	  if (feature == 0) {
	    feature = Copy(last);
	  }
	  (Char(last))[0] = (char)toupper((Char(last))[0]);
	  modvar = NewStringf("m%s", last);
	}
      }
      Delete(modules);
    }
    Delete(mod_name);
  }

  /* --------------------------------------------------------------------------
   * nativeWrapper()
   * -------------------------------------------------------------------------- */
  virtual int nativeWrapper(Node *n) {
    String *funcname = Getattr(n, "wrap:name");
    Swig_warning(WARN_LANG_NATIVE_UNIMPL, input_file, line_number, "Adding native function %s not supported (ignored).\n", funcname);
    return SWIG_NOWRAP;
  }

  /**
   * Process the comma-separated list of aliases (if any).
   */
  void defineAliases(Node *n, const_String_or_char_ptr iname) {
    String *aliasv = Getattr(n, "feature:alias");
    if (aliasv) {
      List *aliases = Split(aliasv, ',', INT_MAX);
      if (aliases && Len(aliases) > 0) {
	Iterator alias = First(aliases);
	while (alias.item) {
	  if (Len(alias.item) > 0) {
	    if (multipleInheritance) {
	      Printv(klass->init, tab4, "rb_define_alias(", klass->mImpl, ", \"", alias.item, "\", \"", iname, "\");\n", NIL);
	    } else {
	      Printv(klass->init, tab4, "rb_define_alias(", klass->vname, ", \"", alias.item, "\", \"", iname, "\");\n", NIL);
	    }
	  }
	  alias = Next(alias);
	}
      }
      Delete(aliases);
    }
  }

  /* ---------------------------------------------------------------------
   * create_command(Node *n, char *iname)
   *
   * Creates a new command from a C function.
   *              iname = Name of function in scripting language
   *
   * A note about what "protected" and "private" mean in Ruby:
   *
   * A private method is accessible only within the class or its subclasses,
   * and it is callable only in "function form", with 'self' (implicit or
   * explicit) as a receiver.
   *
   * A protected method is callable only from within its class, but unlike
   * a private method, it can be called with a receiver other than self, such
   * as another instance of the same class.
   * --------------------------------------------------------------------- */

  void create_command(Node *n, const_String_or_char_ptr iname) {

    String *alloc_func = Swig_name_wrapper(iname);
    String *wname = Swig_name_wrapper(iname);
    if (CPlusPlus) {
      Insert(wname, 0, "VALUEFUNC(");
      Append(wname, ")");
    }
    if (current != NO_CPP)
      iname = klass->strip(iname);
    if (Getattr(special_methods, iname)) {
      iname = GetChar(special_methods, iname);
    }

    String *s = NewString("");
    String *temp = NewString("");

#ifdef SWIG_PROTECTED_TARGET_METHODS
    const char *rb_define_method = is_public(n) ? "rb_define_method" : "rb_define_protected_method";
#else
    const char *rb_define_method = "rb_define_method";
#endif
    switch (current) {
    case MEMBER_FUNC:
      {
	if (multipleInheritance) {
	  Printv(klass->init, tab4, rb_define_method, "(", klass->mImpl, ", \"", iname, "\", ", wname, ", -1);\n", NIL);
	} else {
	  Printv(klass->init, tab4, rb_define_method, "(", klass->vname, ", \"", iname, "\", ", wname, ", -1);\n", NIL);
	}
      }
      break;
    case CONSTRUCTOR_ALLOCATE:
      Printv(s, tab4, "rb_define_alloc_func(", klass->vname, ", ", alloc_func, ");\n", NIL);
      Replaceall(klass->init, "$allocator", s);
      break;
    case CONSTRUCTOR_INITIALIZE:
      Printv(s, tab4, rb_define_method, "(", klass->vname, ", \"initialize\", ", wname, ", -1);\n", NIL);
      Replaceall(klass->init, "$initializer", s);
      break;
    case MEMBER_VAR:
      Append(temp, iname);
      /* Check for _set or _get at the end of the name. */
      if (Len(temp) > 4) {
	const char *p = Char(temp) + (Len(temp) - 4);
	if (strcmp(p, "_set") == 0) {
	  Delslice(temp, Len(temp) - 4, DOH_END);
	  Append(temp, "=");
	} else if (strcmp(p, "_get") == 0) {
	  Delslice(temp, Len(temp) - 4, DOH_END);
	}
      }
      if (multipleInheritance) {
	Printv(klass->init, tab4, "rb_define_method(", klass->mImpl, ", \"", temp, "\", ", wname, ", -1);\n", NIL);
      } else {
	Printv(klass->init, tab4, "rb_define_method(", klass->vname, ", \"", temp, "\", ", wname, ", -1);\n", NIL);
      }
      break;
    case STATIC_FUNC:
      Printv(klass->init, tab4, "rb_define_singleton_method(", klass->vname, ", \"", iname, "\", ", wname, ", -1);\n", NIL);
      break;
    case NO_CPP:
      if (!useGlobalModule) {
	Printv(s, tab4, "rb_define_module_function(", modvar, ", \"", iname, "\", ", wname, ", -1);\n", NIL);
	Printv(f_init, s, NIL);
      } else {
	Printv(s, tab4, "rb_define_global_function(\"", iname, "\", ", wname, ", -1);\n", NIL);
	Printv(f_init, s, NIL);
      }
      break;
    case DESTRUCTOR:
    case CLASS_CONST:
    case STATIC_VAR:
    default:
      assert(false);		// Should not have gotten here for these types
    }

    defineAliases(n, iname);

    Delete(temp);
    Delete(s);
    Delete(wname);
    Delete(alloc_func);
  }

  /* ---------------------------------------------------------------------
   * applyInputTypemap()
   *
   * Look up the appropriate "in" typemap for this parameter (p),
   * substitute the correct strings for the $target and $input typemap
   * parameters, and dump the resulting code to the wrapper file.
   * --------------------------------------------------------------------- */

  Parm *applyInputTypemap(Parm *p, String *ln, String *source, Wrapper *f, String *symname) {
    String *tm;
    SwigType *pt = Getattr(p, "type");
    if ((tm = Getattr(p, "tmap:in"))) {
      Replaceall(tm, "$target", ln);
      Replaceall(tm, "$source", source);
      Replaceall(tm, "$input", source);
      Replaceall(tm, "$symname", symname);

      if (Getattr(p, "wrap:disown") || (Getattr(p, "tmap:in:disown"))) {
	Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
      } else {
	Replaceall(tm, "$disown", "0");
      }

      Setattr(p, "emit:input", Copy(source));
      Printf(f->code, "%s\n", tm);
      p = Getattr(p, "tmap:in:next");
    } else {
      Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
      p = nextSibling(p);
    }
    return p;
  }

  Parm *skipIgnoredArgs(Parm *p) {
    while (checkAttribute(p, "tmap:in:numinputs", "0")) {
      p = Getattr(p, "tmap:in:next");
    }
    return p;
  }

  /* ---------------------------------------------------------------------
   * marshalInputArgs()
   *
   * Process all of the arguments passed into the scripting language
   * method and convert them into C/C++ function arguments using the
   * supplied typemaps.
   * --------------------------------------------------------------------- */

  void marshalInputArgs(Node *n, ParmList *l, int numarg, int numreq, String *kwargs, bool allow_kwargs, Wrapper *f) {
    int i;
    Parm *p;
    String *tm;
    String *source;
    String *target;

    source = NewString("");
    target = NewString("");

    bool ctor_director = (current == CONSTRUCTOR_INITIALIZE && Swig_directorclass(n));

    /**
     * The 'start' value indicates which of the C/C++ function arguments
     * produced here corresponds to the first value in Ruby's argv[] array.
     * The value of start is either zero or one. If start is zero, then
     * the first argument (with name arg1) is based on the value of argv[0].
     * If start is one, then arg1 is based on the value of argv[1].
     */
    int start = (current == MEMBER_FUNC || current == MEMBER_VAR || ctor_director) ? 1 : 0;

    int varargs = emit_isvarargs(l);

    Printf(kwargs, "{ ");
    for (i = 0, p = l; i < numarg; i++) {

      p = skipIgnoredArgs(p);

      String *pn = Getattr(p, "name");
      String *ln = Getattr(p, "lname");

      /* Produce string representation of source argument */
      Clear(source);

      /* First argument is a special case */
      if (i == 0) {
	Printv(source, (start == 0) ? "argv[0]" : "self", NIL);
      } else {
	Printf(source, "argv[%d]", i - start);
      }

      /* Produce string representation of target argument */
      Clear(target);
      Printf(target, "%s", Char(ln));

      if (i >= (numreq)) {	/* Check if parsing an optional argument */
	Printf(f->code, "    if (argc > %d) {\n", i - start);
      }

      /* Record argument name for keyword argument handling */
      if (Len(pn)) {
	Printf(kwargs, "\"%s\",", pn);
      } else {
	Printf(kwargs, "\"arg%d\",", i + 1);
      }

      /* Look for an input typemap */
      p = applyInputTypemap(p, ln, source, f, Getattr(n, "name"));
      if (i >= numreq) {
	Printf(f->code, "}\n");
      }
    }

    /* Finish argument marshalling */
    Printf(kwargs, " NULL }");
    if (allow_kwargs) {
// kwarg support not implemented
//      Printv(f->locals, tab4, "const char *kwnames[] = ", kwargs, ";\n", NIL);
    }

    /* Trailing varargs */
    if (varargs) {
      if (p && (tm = Getattr(p, "tmap:in"))) {
	Clear(source);
	Printf(source, "argv[%d]", i - start);
	Replaceall(tm, "$input", source);
	Setattr(p, "emit:input", Copy(source));
	Printf(f->code, "if (argc > %d) {\n", i - start);
	Printv(f->code, tm, "\n", NIL);
	Printf(f->code, "}\n");
      }
    }

    Delete(source);
    Delete(target);
  }

  /* ---------------------------------------------------------------------
   * insertConstraintCheckingCode(ParmList *l, Wrapper *f)
   *
   * Checks each of the parameters in the parameter list for a "check"
   * typemap and (if it finds one) inserts the typemapping code into
   * the function wrapper.
   * --------------------------------------------------------------------- */

  void insertConstraintCheckingCode(ParmList *l, Wrapper *f) {
    Parm *p;
    String *tm;
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:check"))) {
	Replaceall(tm, "$target", Getattr(p, "lname"));
	Printv(f->code, tm, "\n", NIL);
	p = Getattr(p, "tmap:check:next");
      } else {
	p = nextSibling(p);
      }
    }
  }

  /* ---------------------------------------------------------------------
   * insertCleanupCode(ParmList *l, String *cleanup)
   *
   * Checks each of the parameters in the parameter list for a "freearg"
   * typemap and (if it finds one) inserts the typemapping code into
   * the function wrapper.
   * --------------------------------------------------------------------- */

  void insertCleanupCode(ParmList *l, String *cleanup) {
    String *tm;
    for (Parm *p = l; p;) {
      if ((tm = Getattr(p, "tmap:freearg"))) {
	if (Len(tm) != 0) {
	  Replaceall(tm, "$source", Getattr(p, "lname"));
	  Printv(cleanup, tm, "\n", NIL);
	}
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }
  }

  /* ---------------------------------------------------------------------
   * insertArgOutputCode(ParmList *l, String *outarg, int& need_result)
   *
   * Checks each of the parameters in the parameter list for a "argout"
   * typemap and (if it finds one) inserts the typemapping code into
   * the function wrapper.
   * --------------------------------------------------------------------- */

  void insertArgOutputCode(ParmList *l, String *outarg, int &need_result) {
    String *tm;
    for (Parm *p = l; p;) {
      if ((tm = Getattr(p, "tmap:argout"))) {
	Replaceall(tm, "$source", Getattr(p, "lname"));
	Replaceall(tm, "$target", "vresult");
	Replaceall(tm, "$result", "vresult");
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));

	Printv(outarg, tm, "\n", NIL);
	need_result += 1;
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }
  }

  /* ---------------------------------------------------------------------
   * validIdentifier()
   *
   * Is this a valid identifier in the scripting language?
   * Ruby method names can include any combination of letters, numbers
   * and underscores. A Ruby method name may optionally end with
   * a question mark ("?"), exclamation point ("!") or equals sign ("=").
   *
   * Methods whose names end with question marks are, by convention,
   * predicate methods that return true or false (e.g. Array#empty?).
   *
   * Methods whose names end with exclamation points are, by convention,
   * called bang methods that modify the instance in place (e.g. Array#sort!).
   *
   * Methods whose names end with an equals sign are attribute setters
   * (e.g. Thread#critical=).
   * --------------------------------------------------------------------- */

  virtual int validIdentifier(String *s) {
    char *c = Char(s);
    while (*c) {
      if (!(isalnum(*c) || (*c == '_') || (*c == '?') || (*c == '!') || (*c == '=')))
	return 0;
      c++;
    }
    return 1;
  }

  /* ---------------------------------------------------------------------
   * functionWrapper()
   *
   * Create a function declaration and register it with the interpreter.
   * --------------------------------------------------------------------- */

  virtual int functionWrapper(Node *n) {

    String *nodeType;
    bool destructor;

    String *symname = Copy(Getattr(n, "sym:name"));
    SwigType *t = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    int director_method = 0;
    String *tm;

    int need_result = 0;

    /* Ruby needs no destructor wrapper */
    if (current == DESTRUCTOR)
      return SWIG_NOWRAP;

    nodeType = Getattr(n, "nodeType");
    destructor = (!Cmp(nodeType, "destructor"));

    /* If the C++ class constructor is overloaded, we only want to
     * write out the "new" singleton method once since it is always
     * the same. (It's the "initialize" method that will handle the
     * overloading). */

    if (current == CONSTRUCTOR_ALLOCATE && Swig_symbol_isoverloaded(n) && Getattr(n, "sym:nextSibling") != 0)
      return SWIG_OK;

    String *overname = 0;
    if (Getattr(n, "sym:overloaded")) {
      overname = Getattr(n, "sym:overname");
    } else {
      if (!addSymbol(symname, n))
	return SWIG_ERROR;
    }

    String *cleanup = NewString("");
    String *outarg = NewString("");
    String *kwargs = NewString("");
    Wrapper *f = NewWrapper();

    /* Rename predicate methods */
    if (GetFlag(n, "feature:predicate")) {
      Append(symname, "?");
    }

    /* Rename bang methods */
    if (GetFlag(n, "feature:bang")) {
      Append(symname, "!");
    }

    /* Determine the name of the SWIG wrapper function */
    String *wname = Swig_name_wrapper(symname);
    if (overname && current != CONSTRUCTOR_ALLOCATE) {
      Append(wname, overname);
    }

    /* Emit arguments */
    if (current != CONSTRUCTOR_ALLOCATE) {
      emit_parameter_variables(l, f);
    }

    /* Attach standard typemaps */
    if (current != CONSTRUCTOR_ALLOCATE) {
      emit_attach_parmmaps(l, f);
    }
    Setattr(n, "wrap:parms", l);

    /* Get number of arguments */
    int numarg = emit_num_arguments(l);
    int numreq = emit_num_required(l);
    int varargs = emit_isvarargs(l);
    bool allow_kwargs = GetFlag(n, "feature:kwargs") ? true : false;

    bool ctor_director = (current == CONSTRUCTOR_INITIALIZE && Swig_directorclass(n));
    int start = (current == MEMBER_FUNC || current == MEMBER_VAR || ctor_director) ? 1 : 0;

    /* Now write the wrapper function itself */
    if (current == CONSTRUCTOR_ALLOCATE) {
      Printv(f->def, "SWIGINTERN VALUE\n", NIL);
      Printf(f->def, "#ifdef HAVE_RB_DEFINE_ALLOC_FUNC\n");
      Printv(f->def, wname, "(VALUE self)\n", NIL);
      Printf(f->def, "#else\n");
      Printv(f->def, wname, "(int argc, VALUE *argv, VALUE self)\n", NIL);
      Printf(f->def, "#endif\n");
      Printv(f->def, "{\n", NIL);
    } else if (current == CONSTRUCTOR_INITIALIZE) {
      Printv(f->def, "SWIGINTERN VALUE\n", wname, "(int argc, VALUE *argv, VALUE self) {", NIL);
      if (!varargs) {
	Printf(f->code, "if ((argc < %d) || (argc > %d)) ", numreq - start, numarg - start);
      } else {
	Printf(f->code, "if (argc < %d) ", numreq - start);
      }
      Printf(f->code, "{rb_raise(rb_eArgError, \"wrong # of arguments(%%d for %d)\",argc); SWIG_fail;}\n", numreq - start);
    } else {

      if ( current == NO_CPP )
	{
	  String* docs = docstring(n, AUTODOC_FUNC);
	  Printf(f_wrappers, "%s", docs);
	  Delete(docs);
	}

      Printv(f->def, "SWIGINTERN VALUE\n", wname, "(int argc, VALUE *argv, VALUE self) {", NIL);
      if (!varargs) {
	Printf(f->code, "if ((argc < %d) || (argc > %d)) ", numreq - start, numarg - start);
      } else {
	Printf(f->code, "if (argc < %d) ", numreq - start);
      }
      Printf(f->code, "{rb_raise(rb_eArgError, \"wrong # of arguments(%%d for %d)\",argc); SWIG_fail;}\n", numreq - start);
    }

    /* Now walk the function parameter list and generate code */
    /* to get arguments */
    if (current != CONSTRUCTOR_ALLOCATE) {
      marshalInputArgs(n, l, numarg, numreq, kwargs, allow_kwargs, f);
    }
    // FIXME?
    if (ctor_director) {
      numarg--;
      numreq--;
    }

    /* Insert constraint checking code */
    insertConstraintCheckingCode(l, f);

    /* Insert cleanup code */
    insertCleanupCode(l, cleanup);

    /* Insert argument output code */
    insertArgOutputCode(l, outarg, need_result);

    /* if the object is a director, and the method call originated from its
     * underlying Ruby object, resolve the call by going up the c++ 
     * inheritance chain.  otherwise try to resolve the method in Ruby.
     * without this check an infinite loop is set up between the director and
     * shadow class method calls.
     */

    // NOTE: this code should only be inserted if this class is the
    // base class of a director class.  however, in general we haven't
    // yet analyzed all classes derived from this one to see if they are
    // directors.  furthermore, this class may be used as the base of
    // a director class defined in a completely different module at a
    // later time, so this test must be included whether or not directorbase
    // is true.  we do skip this code if directors have not been enabled
    // at the command line to preserve source-level compatibility with
    // non-polymorphic swig.  also, if this wrapper is for a smart-pointer
    // method, there is no need to perform the test since the calling object
    // (the smart-pointer) and the director object (the "pointee") are
    // distinct.

    director_method = is_member_director(n) && !is_smart_pointer() && !destructor;
    if (director_method) {
      Wrapper_add_local(f, "director", "Swig::Director *director = 0");
      Printf(f->code, "director = dynamic_cast<Swig::Director *>(arg1);\n");
      Wrapper_add_local(f, "upcall", "bool upcall = false");
      Append(f->code, "upcall = (director && (director->swig_get_self() == self));\n");
    }

    /* Now write code to make the function call */
    if (current != CONSTRUCTOR_ALLOCATE) {
      if (current == CONSTRUCTOR_INITIALIZE) {
	Node *pn = Swig_methodclass(n);
	String *symname = Getattr(pn, "sym:name");
	String *action = Getattr(n, "wrap:action");
	if (directorsEnabled()) {
	  String *classname = NewStringf("const char *classname SWIGUNUSED = \"%s::%s\"", module, symname);
	  Wrapper_add_local(f, "classname", classname);
	}
	if (action) {
          SwigType *smart = Swig_cparse_smartptr(pn);
	  String *result_name = NewStringf("%s%s", smart ? "smart" : "", Swig_cresult_name());
	  if (smart) {
	    String *result_var = NewStringf("%s *%s = 0", SwigType_namestr(smart), result_name);
	    Wrapper_add_local(f, result_name, result_var);
	    Printf(action, "\n%s = new %s(%s);", result_name, SwigType_namestr(smart), Swig_cresult_name());
	  }
	  Printf(action, "\nDATA_PTR(self) = %s;", result_name);
	  if (GetFlag(pn, "feature:trackobjects")) {
	    Printf(action, "\nSWIG_RubyAddTracking(%s, self);", result_name);
	  }
	  Delete(result_name);
	  Delete(smart);
	}
      }

      /* Emit the function call */
      if (director_method) {
	Printf(f->code, "try {\n");
      }

      Setattr(n, "wrap:name", wname);

      Swig_director_emit_dynamic_cast(n, f);
      String *actioncode = emit_action(n);

      if (director_method) {
	Printf(actioncode, "} catch (Swig::DirectorException& e) {\n");
	Printf(actioncode, "  rb_exc_raise(e.getError());\n");
	Printf(actioncode, "  SWIG_fail;\n");
	Printf(actioncode, "}\n");
      }

      /* Return value if necessary */
      if (SwigType_type(t) != T_VOID && current != CONSTRUCTOR_INITIALIZE) {
        need_result = 1;
        if (GetFlag(n, "feature:predicate")) {
          Printv(actioncode, tab4, "vresult = (", Swig_cresult_name(), " ? Qtrue : Qfalse);\n", NIL);
        } else {
          tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode);
          actioncode = 0;
          if (tm) {
            Replaceall(tm, "$result", "vresult");
            Replaceall(tm, "$source", Swig_cresult_name());
            Replaceall(tm, "$target", "vresult");

            if (GetFlag(n, "feature:new"))
              Replaceall(tm, "$owner", "SWIG_POINTER_OWN");
            else
              Replaceall(tm, "$owner", "0");

#if 1
            // FIXME: this will not try to unwrap directors returned as non-director
            //        base class pointers!

            /* New addition to unwrap director return values so that the original
             * Ruby object is returned instead. 
             */
            bool unwrap = false;
            String *decl = Getattr(n, "decl");
            int is_pointer = SwigType_ispointer_return(decl);
            int is_reference = SwigType_isreference_return(decl);
            if (is_pointer || is_reference) {
              String *type = Getattr(n, "type");
              Node *parent = Swig_methodclass(n);
              Node *modname = Getattr(parent, "module");
              Node *target = Swig_directormap(modname, type);
              if (target)
                unwrap = true;
            }
            if (unwrap) {
              Wrapper_add_local(f, "director", "Swig::Director *director = 0");
              Printf(f->code, "director = dynamic_cast<Swig::Director *>(%s);\n", Swig_cresult_name());
              Printf(f->code, "if (director) {\n");
              Printf(f->code, "  vresult = director->swig_get_self();\n");
              Printf(f->code, "} else {\n");
              Printf(f->code, "%s\n", tm);
              Printf(f->code, "}\n");
              director_method = 0;
            } else {
              Printf(f->code, "%s\n", tm);
            }
#else
            Printf(f->code, "%s\n", tm);
#endif
            Delete(tm);
          } else {
            Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s.\n", SwigType_str(t, 0));
          }
        }
      }
      if (actioncode) {
        Append(f->code, actioncode);
        Delete(actioncode);
      }
      emit_return_variable(n, t, f);
    }

    /* Extra code needed for new and initialize methods */
    if (current == CONSTRUCTOR_ALLOCATE) {
      Node *pn = Swig_methodclass(n);
      SwigType *smart = Swig_cparse_smartptr(pn);
      if (smart)
	SwigType_add_pointer(smart);
      String *classtype = smart ? smart : t;
      need_result = 1;
      Printf(f->code, "VALUE vresult = SWIG_NewClassInstance(self, SWIGTYPE%s);\n", Char(SwigType_manglestr(classtype)));
      Printf(f->code, "#ifndef HAVE_RB_DEFINE_ALLOC_FUNC\n");
      Printf(f->code, "rb_obj_call_init(vresult, argc, argv);\n");
      Printf(f->code, "#endif\n");
      Delete(smart);
    } else if (current == CONSTRUCTOR_INITIALIZE) {
      need_result = 1;
    }
    else
      {
	if ( need_result > 1 ) {
	  if ( SwigType_type(t) == T_VOID )
	    Printf(f->code, "vresult = rb_ary_new();\n");
	  else
	    {
	      Printf(f->code, "if (vresult == Qnil) vresult = rb_ary_new();\n");
	      Printf(f->code, "else vresult = SWIG_Ruby_AppendOutput( "
		     "rb_ary_new(), vresult);\n");
	    }
	}
      }

    /* Dump argument output code; */
    Printv(f->code, outarg, NIL);

    /* Dump the argument cleanup code */
    int need_cleanup = (current != CONSTRUCTOR_ALLOCATE) && (Len(cleanup) != 0);
    if (need_cleanup) {
      Printv(f->code, cleanup, NIL);
    }


    /* Look for any remaining cleanup.  This processes the %new directive */
    if (current != CONSTRUCTOR_ALLOCATE && GetFlag(n, "feature:new")) {
      tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0);
      if (tm) {
	Replaceall(tm, "$source", Swig_cresult_name());
	Printv(f->code, tm, "\n", NIL);
	Delete(tm);
      }
    }

    /* Special processing on return value. */
    tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0);
    if (tm) {
      Replaceall(tm, "$source", Swig_cresult_name());
      Printv(f->code, tm, NIL);
      Delete(tm);
    }

    if (director_method) {
      if ((tm = Swig_typemap_lookup("directorfree", n, Swig_cresult_name(), 0))) {
	Replaceall(tm, "$input", Swig_cresult_name());
	Replaceall(tm, "$result", "vresult");
	Printf(f->code, "%s\n", tm);
      }
    }


    /* Wrap things up (in a manner of speaking) */
    if (need_result) {
      if (current == CONSTRUCTOR_ALLOCATE) {
	Printv(f->code, tab4, "return vresult;\n", NIL);
      } else if (current == CONSTRUCTOR_INITIALIZE) {
	Printv(f->code, tab4, "return self;\n", NIL);
	Printv(f->code, "fail:\n", NIL);
	if (need_cleanup) {
	  Printv(f->code, cleanup, NIL);
	}
	Printv(f->code, tab4, "return Qnil;\n", NIL);
      } else {
	Wrapper_add_local(f, "vresult", "VALUE vresult = Qnil");
	Printv(f->code, tab4, "return vresult;\n", NIL);
	Printv(f->code, "fail:\n", NIL);
	if (need_cleanup) {
	  Printv(f->code, cleanup, NIL);
	}
	Printv(f->code, tab4, "return Qnil;\n", NIL);
      }
    } else {
      Printv(f->code, tab4, "return Qnil;\n", NIL);
      Printv(f->code, "fail:\n", NIL);
      if (need_cleanup) {
	Printv(f->code, cleanup, NIL);
      }
      Printv(f->code, tab4, "return Qnil;\n", NIL);
    }

    Printf(f->code, "}\n");

    /* Substitute the cleanup code */
    Replaceall(f->code, "$cleanup", cleanup);

    /* Substitute the function name */
    Replaceall(f->code, "$symname", symname);

    /* Emit the function */
    Wrapper_print(f, f_wrappers);

    /* Now register the function with the interpreter */
    if (!Swig_symbol_isoverloaded(n)) {
      create_command(n, symname);
    } else {
      if (current == CONSTRUCTOR_ALLOCATE) {
	create_command(n, symname);
      } else {
	if (!Getattr(n, "sym:nextSibling"))
	  dispatchFunction(n);
      }
    }

    Delete(kwargs);
    Delete(cleanup);
    Delete(outarg);
    DelWrapper(f);
    Delete(symname);

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * dispatchFunction()
   * ------------------------------------------------------------ */

  void dispatchFunction(Node *n) {
    /* Last node in overloaded chain */

    int maxargs;
    String *tmp = NewString("");
    String *dispatch = Swig_overload_dispatch(n, "return %s(nargs, args, self);", &maxargs);

    /* Generate a dispatch wrapper for all overloaded functions */

    Wrapper *f = NewWrapper();
    String *symname = Getattr(n, "sym:name");
    String *wname = Swig_name_wrapper(symname);

    Printv(f->def, "SWIGINTERN VALUE ", wname, "(int nargs, VALUE *args, VALUE self) {", NIL);

    Wrapper_add_local(f, "argc", "int argc");
    bool ctor_director = (current == CONSTRUCTOR_INITIALIZE && Swig_directorclass(n));
    if (current == MEMBER_FUNC || current == MEMBER_VAR || ctor_director) {
      Printf(tmp, "VALUE argv[%d]", maxargs + 1);
    } else {
      Printf(tmp, "VALUE argv[%d]", maxargs);
    }
    Wrapper_add_local(f, "argv", tmp);
    Wrapper_add_local(f, "ii", "int ii");

    if (current == MEMBER_FUNC || current == MEMBER_VAR || ctor_director) {
      maxargs += 1;
      Printf(f->code, "argc = nargs + 1;\n");
      Printf(f->code, "argv[0] = self;\n");
      Printf(f->code, "if (argc > %d) SWIG_fail;\n", maxargs);
      Printf(f->code, "for (ii = 1; (ii < argc); ++ii) {\n");
      Printf(f->code, "argv[ii] = args[ii-1];\n");
      Printf(f->code, "}\n");
    } else {
      Printf(f->code, "argc = nargs;\n");
      Printf(f->code, "if (argc > %d) SWIG_fail;\n", maxargs);
      Printf(f->code, "for (ii = 0; (ii < argc); ++ii) {\n");
      Printf(f->code, "argv[ii] = args[ii];\n");
      Printf(f->code, "}\n");
    }

    Replaceall(dispatch, "$args", "nargs, args, self");
    Printv(f->code, dispatch, "\n", NIL);


    
    // Generate prototype list, go to first node
    Node *sibl = n;

    while (Getattr(sibl, "sym:previousSibling"))
      sibl = Getattr(sibl, "sym:previousSibling");	// go all the way up

    // Constructors will be treated specially
    const bool isCtor = (!Cmp(Getattr(sibl, "nodeType"), "constructor"));
    const bool isMethod = ( Cmp(Getattr(sibl, "ismember"), "1") == 0 &&
			    (!isCtor) );

    // Construct real method name
    String* methodName = NewString("");
    if ( isMethod ) {
      // Sometimes a method node has no parent (SF#3034054).
      // This value is used in an exception message, so just skip the class
      // name in this case so at least we don't segfault.  This is probably
      // just working around a problem elsewhere though.
      Node *parent_node = parentNode(sibl);
      if (parent_node)
	Printv( methodName, Getattr(parent_node,"sym:name"), ".", NIL );
    }
    Append( methodName, Getattr(sibl,"sym:name" ) );
    if ( isCtor ) Append( methodName, ".new" ); 

    // Generate prototype list
    String *protoTypes = NewString("");
    do {
      Append( protoTypes, "\n\"    ");
      if (!isCtor) {
	SwigType *type = SwigType_str(Getattr(sibl, "type"), NULL);
	Printv(protoTypes, type, " ", NIL);
	Delete(type);
      }
      Printv(protoTypes, methodName, NIL );
      Parm* p = Getattr(sibl, "wrap:parms");
      if (p && (current == MEMBER_FUNC || current == MEMBER_VAR || 
		ctor_director) )
	p = nextSibling(p); // skip self
      Append( protoTypes, "(" );
      while(p)
	{
 	  Append( protoTypes, SwigType_str(Getattr(p,"type"), Getattr(p,"name")) );
	  if ( ( p = nextSibling(p)) ) Append(protoTypes, ", ");
	}
      Append( protoTypes, ")\\n\"" );
    } while ((sibl = Getattr(sibl, "sym:nextSibling")));

    Append(f->code, "fail:\n");
    Printf(f->code, "Ruby_Format_OverloadedError( argc, %d, \"%s\", %s);\n", 
	   maxargs, methodName, protoTypes);
    Append(f->code, "\nreturn Qnil;\n");

    Delete(methodName);
    Delete(protoTypes);

    Printv(f->code, "}\n", NIL);
    Wrapper_print(f, f_wrappers);
    create_command(n, Char(symname));

    DelWrapper(f);
    Delete(dispatch);
    Delete(tmp);
    Delete(wname);
  }

  /* ---------------------------------------------------------------------
   * variableWrapper()
   * --------------------------------------------------------------------- */

  virtual int variableWrapper(Node *n) {
    String* docs = docstring(n, AUTODOC_GETTER);
    Printf(f_wrappers, "%s", docs);
    Delete(docs);


    char *name = GetChar(n, "name");
    char *iname = GetChar(n, "sym:name");
    SwigType *t = Getattr(n, "type");
    String *tm;
    String *getfname, *setfname;
    Wrapper *getf, *setf;

    getf = NewWrapper();
    setf = NewWrapper();

    /* create getter */
    int addfail = 0;
    String *getname = Swig_name_get(NSPACE_TODO, iname);
    getfname = Swig_name_wrapper(getname);
    Setattr(n, "wrap:name", getfname);
    Printv(getf->def, "SWIGINTERN VALUE\n", getfname, "(", NIL);
    Printf(getf->def, "VALUE self");
    Printf(getf->def, ") {");
    Wrapper_add_local(getf, "_val", "VALUE _val");

    tm = Swig_typemap_lookup("varout", n, name, 0);
    if (tm) {
      Replaceall(tm, "$result", "_val");
      Replaceall(tm, "$target", "_val");
      Replaceall(tm, "$source", name);
      /* Printv(getf->code,tm, NIL); */
      addfail = emit_action_code(n, getf->code, tm);
    } else {
      Swig_warning(WARN_TYPEMAP_VAROUT_UNDEF, input_file, line_number, "Unable to read variable of type %s\n", SwigType_str(t, 0));
    }
    Printv(getf->code, tab4, "return _val;\n", NIL);
    if (addfail) {
      Append(getf->code, "fail:\n");
      Append(getf->code, "  return Qnil;\n");
    }
    Append(getf->code, "}\n");

    Wrapper_print(getf, f_wrappers);

    if (!is_assignable(n)) {
      setfname = NewString("NULL");
    } else {
      /* create setter */
      String* docs = docstring(n, AUTODOC_SETTER);
      Printf(f_wrappers, "%s", docs);
      Delete(docs);

      String *setname = Swig_name_set(NSPACE_TODO, iname);
      setfname = Swig_name_wrapper(setname);
      Setattr(n, "wrap:name", setfname);
      Printv(setf->def, "SWIGINTERN VALUE\n", setfname, "(VALUE self, ", NIL);
      Printf(setf->def, "VALUE _val) {");
      tm = Swig_typemap_lookup("varin", n, name, 0);
      if (tm) {
	Replaceall(tm, "$input", "_val");
	Replaceall(tm, "$source", "_val");
	Replaceall(tm, "$target", name);
	/* Printv(setf->code,tm,"\n",NIL); */
	emit_action_code(n, setf->code, tm);
      } else {
	Swig_warning(WARN_TYPEMAP_VARIN_UNDEF, input_file, line_number, "Unable to set variable of type %s\n", SwigType_str(t, 0));
      }
      Printv(setf->code, tab4, "return _val;\n", NIL);
      Printf(setf->code, "fail:\n");
      Printv(setf->code, tab4, "return Qnil;\n", NIL);
      Printf(setf->code, "}\n");
      Wrapper_print(setf, f_wrappers);
      Delete(setname);
    }

    /* define accessor method */
    if (CPlusPlus) {
      Insert(getfname, 0, "VALUEFUNC(");
      Append(getfname, ")");
      Insert(setfname, 0, "VALUEFUNC(");
      Append(setfname, ")");
    }

    String *s = NewString("");
    switch (current) {
    case STATIC_VAR:
      /* C++ class variable */
      Printv(s, tab4, "rb_define_singleton_method(", klass->vname, ", \"", klass->strip(iname), "\", ", getfname, ", 0);\n", NIL);
      if (!GetFlag(n, "feature:immutable")) {
	Printv(s, tab4, "rb_define_singleton_method(", klass->vname, ", \"", klass->strip(iname), "=\", ", setfname, ", 1);\n", NIL);
      }
      Printv(klass->init, s, NIL);
      break;
    default:
      /* C global variable */
      /* wrapped in Ruby module attribute */
      assert(current == NO_CPP);
      if (!useGlobalModule) {
	Printv(s, tab4, "rb_define_singleton_method(", modvar, ", \"", iname, "\", ", getfname, ", 0);\n", NIL);
	if (!GetFlag(n, "feature:immutable")) {
	  Printv(s, tab4, "rb_define_singleton_method(", modvar, ", \"", iname, "=\", ", setfname, ", 1);\n", NIL);
	}
      } else {
	Printv(s, tab4, "rb_define_global_method(\"", iname, "\", ", getfname, ", 0);\n", NIL);
	if (!GetFlag(n, "feature:immutable")) {
	  Printv(s, tab4, "rb_define_global_method(\"", iname, "=\", ", setfname, ", 1);\n", NIL);
	}
      }
      Printv(f_init, s, NIL);
      Delete(s);
      break;
    }
    Delete(getname);
    Delete(getfname);
    Delete(setfname);
    DelWrapper(setf);
    DelWrapper(getf);
    return SWIG_OK;
  }


  /* ---------------------------------------------------------------------
   * validate_const_name(char *name)
   *
   * Validate constant name.
   * --------------------------------------------------------------------- */

  char *validate_const_name(char *name, const char *reason) {
    if (!name || name[0] == '\0')
      return name;

    if (isupper(name[0]))
      return name;

    if (islower(name[0])) {
      name[0] = (char)toupper(name[0]);
      Swig_warning(WARN_RUBY_WRONG_NAME, input_file, line_number, "Wrong %s name (corrected to `%s')\n", reason, name);
      return name;
    }

    Swig_warning(WARN_RUBY_WRONG_NAME, input_file, line_number, "Wrong %s name %s\n", reason, name);

    return name;
  }

  /* ---------------------------------------------------------------------
   * constantWrapper()
   * --------------------------------------------------------------------- */

  virtual int constantWrapper(Node *n) {
    Swig_require("constantWrapper", n, "*sym:name", "type", "value", NIL);

    char *iname = GetChar(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");

    if (current == CLASS_CONST) {
      iname = klass->strip(iname);
    }
    validate_const_name(iname, "constant");
    SetChar(n, "sym:name", iname);

    /* Special hook for member pointer */
    if (SwigType_type(type) == T_MPOINTER) {
      String *wname = Swig_name_wrapper(iname);
      Printf(f_header, "static %s = %s;\n", SwigType_str(type, wname), value);
      value = Char(wname);
    }
    String *tm = Swig_typemap_lookup("constant", n, value, 0);
    if (!tm)
      tm = Swig_typemap_lookup("constcode", n, value, 0);
    if (tm) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", iname);
      Replaceall(tm, "$symname", iname);
      Replaceall(tm, "$value", value);
      if (current == CLASS_CONST) {
	if (multipleInheritance) {
	  Replaceall(tm, "$module", klass->mImpl);
	  Printv(klass->init, tm, "\n", NIL);
	} else {
	  Replaceall(tm, "$module", klass->vname);
	  Printv(klass->init, tm, "\n", NIL);
	}
      } else {
	if (!useGlobalModule) {
	  Replaceall(tm, "$module", modvar);
	} else {
	  Replaceall(tm, "$module", "rb_cObject");
	}
	Printf(f_init, "%s\n", tm);
      }
    } else {
      Swig_warning(WARN_TYPEMAP_CONST_UNDEF, input_file, line_number, "Unsupported constant value %s = %s\n", SwigType_str(type, 0), value);
    }
    Swig_restore(n);
    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------------
   * classDeclaration() 
   *
   * Records information about classes---even classes that might be defined in
   * other modules referenced by %import.
   * ----------------------------------------------------------------------------- */

  virtual int classDeclaration(Node *n) {
    if (!Getattr(n, "feature:onlychildren")) {
      String *name = Getattr(n, "name");
      String *symname = Getattr(n, "sym:name");

      String *namestr = SwigType_namestr(name);
      klass = RCLASS(classes, Char(namestr));
      if (!klass) {
	klass = new RClass();
	String *valid_name = NewString(symname ? symname : namestr);
	validate_const_name(Char(valid_name), "class");
	klass->set_name(namestr, symname, valid_name);
	SET_RCLASS(classes, Char(namestr), klass);
	Delete(valid_name);
      }
      Delete(namestr);
    }
    return Language::classDeclaration(n);
  }

  /**
   * Process the comma-separated list of mixed-in module names (if any).
   */
  void includeRubyModules(Node *n) {
    String *mixin = Getattr(n, "feature:mixin");
    if (mixin) {
      List *modules = Split(mixin, ',', INT_MAX);
      if (modules && Len(modules) > 0) {
	Iterator mod = First(modules);
	while (mod.item) {
	  if (Len(mod.item) > 0) {
	    Printf(klass->init, "rb_include_module(%s, rb_eval_string(\"%s\"));\n", klass->vname, mod.item);
	  }
	  mod = Next(mod);
	}
      }
      Delete(modules);
    }
  }

  void handleBaseClasses(Node *n) {
    List *baselist = Getattr(n, "bases");
    if (baselist && Len(baselist)) {
      Iterator base = First(baselist);
      while (base.item && GetFlag(base.item, "feature:ignore")) {
	base = Next(base);
      }
      while (base.item) {
	String *basename = Getattr(base.item, "name");
	String *basenamestr = SwigType_namestr(basename);
	RClass *super = RCLASS(classes, Char(basenamestr));
	Delete(basenamestr);
	if (super) {
	  SwigType *btype = NewString(basename);
	  SwigType_add_pointer(btype);
	  SwigType_remember(btype);
	  SwigType *smart = Swig_cparse_smartptr(base.item);
	  if (smart) {
	    SwigType_add_pointer(smart);
	    SwigType_remember(smart);
	  }
	  String *bmangle = SwigType_manglestr(smart ? smart : btype);
	  if (multipleInheritance) {
	    Insert(bmangle, 0, "((swig_class *) SWIGTYPE");
	    Append(bmangle, "->clientdata)->mImpl");
	    Printv(klass->init, "rb_include_module(", klass->mImpl, ", ", bmangle, ");\n", NIL);
	  } else {
	    Insert(bmangle, 0, "((swig_class *) SWIGTYPE");
	    Append(bmangle, "->clientdata)->klass");
	    Replaceall(klass->init, "$super", bmangle);
	  }
	  Delete(bmangle);
	  Delete(smart);
	  Delete(btype);
	}
	base = Next(base);
	while (base.item && GetFlag(base.item, "feature:ignore")) {
	  base = Next(base);
	}
	if (!multipleInheritance) {
	  /* Warn about multiple inheritance for additional base class(es) */
	  while (base.item) {
	    if (GetFlag(base.item, "feature:ignore")) {
	      base = Next(base);
	      continue;
	    }
	    String *proxyclassname = SwigType_str(Getattr(n, "classtypeobj"), 0);
	    String *baseclassname = SwigType_str(Getattr(base.item, "name"), 0);
	    Swig_warning(WARN_RUBY_MULTIPLE_INHERITANCE, Getfile(n), Getline(n),
			 "Warning for %s, base %s ignored. Multiple inheritance is not supported in Ruby.\n", proxyclassname, baseclassname);
	    base = Next(base);
	  }
	}
      }
    }
  }

  /**
   * Check to see if a %markfunc was specified.
   */
  void handleMarkFuncDirective(Node *n) {
    String *markfunc = Getattr(n, "feature:markfunc");
    if (markfunc) {
      Printf(klass->init, "SwigClass%s.mark = (void (*)(void *)) %s;\n", klass->name, markfunc);
    } else {
      Printf(klass->init, "SwigClass%s.mark = 0;\n", klass->name);
    }
  }

  /**
   * Check to see if a %freefunc was specified.
   */
  void handleFreeFuncDirective(Node *n) {
    String *freefunc = Getattr(n, "feature:freefunc");
    if (freefunc) {
      Printf(klass->init, "SwigClass%s.destroy = (void (*)(void *)) %s;\n", klass->name, freefunc);
    } else {
      if (klass->destructor_defined) {
	Printf(klass->init, "SwigClass%s.destroy = (void (*)(void *)) free_%s;\n", klass->name, klass->mname);
      }
    }
  }

  /**
   * Check to see if tracking is enabled for this class.
   */
  void handleTrackDirective(Node *n) {
    int trackObjects = GetFlag(n, "feature:trackobjects");
    if (trackObjects) {
      Printf(klass->init, "SwigClass%s.trackObjects = 1;\n", klass->name);
    } else {
      Printf(klass->init, "SwigClass%s.trackObjects = 0;\n", klass->name);
    }
  }

  /* ----------------------------------------------------------------------
   * classHandler()
   * ---------------------------------------------------------------------- */

  virtual int classHandler(Node *n) {
    String* docs = docstring(n, AUTODOC_CLASS);
    Printf(f_wrappers, "%s", docs);
    Delete(docs);

    String *name = Getattr(n, "name");
    String *symname = Getattr(n, "sym:name");
    String *namestr = SwigType_namestr(name);	// does template expansion

    klass = RCLASS(classes, Char(namestr));
    assert(klass != 0);
    Delete(namestr);
    String *valid_name = NewString(symname);
    validate_const_name(Char(valid_name), "class");

    Clear(klass->type);
    Printv(klass->type, Getattr(n, "classtype"), NIL);
    Printv(f_wrappers, "static swig_class SwigClass", valid_name, ";\n\n", NIL);
    Printv(klass->init, "\n", tab4, NIL);

    if (!useGlobalModule) {
      Printv(klass->init, klass->vname, " = rb_define_class_under(", modvar, ", \"", klass->name, "\", $super);\n", NIL);
    } else {
      Printv(klass->init, klass->vname, " = rb_define_class(\"", klass->name, 
	     "\", $super);\n", NIL);
    }

    if (multipleInheritance) {
      Printv(klass->init, klass->mImpl, " = rb_define_module_under(", klass->vname, ", \"Impl\");\n", NIL);
    }

    SwigType *tt = NewString(name);
    SwigType_add_pointer(tt);
    SwigType_remember(tt);
    SwigType *smart = Swig_cparse_smartptr(n);
    if (smart) {
      SwigType_add_pointer(smart);
      SwigType_remember(smart);
    }
    String *tm = SwigType_manglestr(smart ? smart : tt);
    Printf(klass->init, "SWIG_TypeClientData(SWIGTYPE%s, (void *) &SwigClass%s);\n", tm, valid_name);
    Delete(tm);
    Delete(smart);
    Delete(tt);
    Delete(valid_name);

    includeRubyModules(n);

    Printv(klass->init, "$allocator", NIL);
    Printv(klass->init, "$initializer", NIL);

    Language::classHandler(n);

    handleBaseClasses(n);
    handleMarkFuncDirective(n);
    handleFreeFuncDirective(n);
    handleTrackDirective(n);

    if (multipleInheritance) {
      Printv(klass->init, "rb_include_module(", klass->vname, ", ", klass->mImpl, ");\n", NIL);
    }

    String *s = NewString("");
    Printv(s, tab4, "rb_undef_alloc_func(", klass->vname, ");\n", NIL);
    Replaceall(klass->init, "$allocator", s);
    Replaceall(klass->init, "$initializer", "");

    if (GetFlag(n, "feature:exceptionclass")) {
      Replaceall(klass->init, "$super", "rb_eRuntimeError");
    } else {
      Replaceall(klass->init, "$super", "rb_cObject");
    }
    Delete(s);

    Printv(f_init, klass->init, NIL);
    klass = 0;
    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * memberfunctionHandler()
   *
   * Method for adding C++ member function
   *
   * By default, we're going to create a function of the form :
   *
   *         Foo_bar(this,args)
   *
   * Where Foo is the classname, bar is the member name and the this pointer
   * is explicitly attached to the beginning.
   *
   * The renaming only applies to the member function part, not the full
   * classname.
   *
   * --------------------------------------------------------------------- */

  virtual int memberfunctionHandler(Node *n) {
    current = MEMBER_FUNC;

    String* docs = docstring(n, AUTODOC_METHOD);
    Printf(f_wrappers, "%s", docs);
    Delete(docs);

    Language::memberfunctionHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------
   * constructorHandler()
   *
   * Method for adding C++ member constructor
   * -------------------------------------------------------------------- */

  void set_director_ctor_code(Node *n) {
    /* director ctor code is specific for each class */
    Delete(director_prot_ctor_code);
    director_prot_ctor_code = NewString("");
    Node *pn = Swig_methodclass(n);
    String *symname = Getattr(pn, "sym:name");
    String *name = Copy(symname);
    char *cname = Char(name);
    if (cname)
      cname[0] = (char)toupper(cname[0]);
    Printv(director_prot_ctor_code,
	   "if ( $comparison ) { /* subclassed */\n",
	   "  $director_new \n",
	   "} else {\n", "  rb_raise(rb_eNameError,\"accessing abstract class or protected constructor\"); \n", "  return Qnil;\n", "}\n", NIL);
    Delete(director_ctor_code);
    director_ctor_code = NewString("");
    Printv(director_ctor_code, "if ( $comparison ) { /* subclassed */\n", "  $director_new \n", "} else {\n", "  $nondirector_new \n", "}\n", NIL);
    Delete(name);
  }

  virtual int constructorHandler(Node *n) {
    int use_director = Swig_directorclass(n);
    if (use_director) {
      set_director_ctor_code(n);
    }

    /* First wrap the allocate method */
    current = CONSTRUCTOR_ALLOCATE;
    Swig_name_register("construct", "%n%c_allocate");

    Language::constructorHandler(n);

    String* docs = docstring(n, AUTODOC_CTOR);
    Printf(f_wrappers, "%s", docs);
    Delete(docs);

    /* 
     * If we're wrapping the constructor of a C++ director class, prepend a new parameter
     * to receive the scripting language object (e.g. 'self')
     *
     */
    Swig_save("ruby:constructorHandler", n, "parms", NIL);
    if (use_director) {
      Parm *parms = Getattr(n, "parms");
      Parm *self;
      String *name = NewString("self");
      String *type = NewString("VALUE");
      self = NewParm(type, name, n);
      Delete(type);
      Delete(name);
      Setattr(self, "lname", "Qnil");
      if (parms)
	set_nextSibling(self, parms);
      Setattr(n, "parms", self);
      Setattr(n, "wrap:self", "1");
      Delete(self);
    }

    /* Now do the instance initialize method */
    current = CONSTRUCTOR_INITIALIZE;
    Swig_name_register("construct", "new_%n%c");
    Language::constructorHandler(n);

    /* Restore original parameter list */
    Delattr(n, "wrap:self");
    Swig_restore(n);

    /* Done */
    Swig_name_unregister("construct");
    current = NO_CPP;
    klass->constructor_defined = 1;
    return SWIG_OK;
  }

  virtual int copyconstructorHandler(Node *n) {
    int use_director = Swig_directorclass(n);
    if (use_director) {
      set_director_ctor_code(n);
    }

    /* First wrap the allocate method */
    current = CONSTRUCTOR_ALLOCATE;
    Swig_name_register("construct", "%n%c_allocate");

    return Language::copyconstructorHandler(n);
  }


  /* ---------------------------------------------------------------------
   * destructorHandler()
   * -------------------------------------------------------------------- */

  virtual int destructorHandler(Node *n) {

    /* Do no spit free function if user defined his own for this class */
    Node *pn = Swig_methodclass(n);
    String *freefunc = Getattr(pn, "feature:freefunc");
    if (freefunc) return SWIG_OK;

    current = DESTRUCTOR;
    Language::destructorHandler(n);

    freefunc = NewString("");
    String *freebody = NewString("");
    String *pname0 = Swig_cparm_name(0, 0);

    Printv(freefunc, "free_", klass->mname, NIL);
    Printv(freebody, "SWIGINTERN void\n", freefunc, "(void *self) {\n", NIL);
    Printv(freebody, tab4, klass->type, " *", pname0, " = (", klass->type, " *)self;\n", NIL);
    Printv(freebody, tab4, NIL);

    /* Check to see if object tracking is activated for the class
       that owns this destructor. */
    if (GetFlag(pn, "feature:trackobjects")) {
      Printf(freebody, "SWIG_RubyRemoveTracking(%s);\n", pname0);
      Printv(freebody, tab4, NIL);
    }

    if (Extend) {
      String *wrap = Getattr(n, "wrap:code");
      if (wrap) {
	Printv(f_wrappers, wrap, NIL);
      }
      /*    Printv(freebody, Swig_name_destroy(name), "(", pname0, ")", NIL); */
      Printv(freebody, Getattr(n, "wrap:action"), "\n", NIL);
    } else {
      String *action = Getattr(n, "wrap:action");
      if (action) {
	Printv(freebody, action, "\n", NIL);
      } else {
	/* In the case swig emits no destroy function. */
	if (CPlusPlus)
	  Printf(freebody, "delete %s;\n", pname0);
	else
	  Printf(freebody, "free((char*) %s);\n", pname0);
      }
    }

    Printv(freebody, "}\n\n", NIL);

    Printv(f_wrappers, freebody, NIL);

    klass->destructor_defined = 1;
    current = NO_CPP;
    Delete(freefunc);
    Delete(freebody);
    Delete(pname0);
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------
   * membervariableHandler()
   *
   * This creates a pair of functions to set/get the variable of a member.
   * -------------------------------------------------------------------- */

  virtual int membervariableHandler(Node *n) {
    String* docs = docstring(n, AUTODOC_GETTER);
    Printf(f_wrappers, "%s", docs);
    Delete(docs);

    if (is_assignable(n)) {
      String* docs = docstring(n, AUTODOC_SETTER);
      Printf(f_wrappers, "%s", docs);
      Delete(docs);
    }

    current = MEMBER_VAR;
    Language::membervariableHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* -----------------------------------------------------------------------
   * staticmemberfunctionHandler()
   *
   * Wrap a static C++ function
   * ---------------------------------------------------------------------- */

  virtual int staticmemberfunctionHandler(Node *n) {
    String* docs = docstring(n, AUTODOC_STATICFUNC);
    Printf(f_wrappers, "%s", docs);
    Delete(docs);

    current = STATIC_FUNC;
    Language::staticmemberfunctionHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* ----------------------------------------------------------------------
   * memberconstantHandler()
   *
   * Create a C++ constant
   * --------------------------------------------------------------------- */

  virtual int memberconstantHandler(Node *n) {
    String* docs = docstring(n, AUTODOC_STATICFUNC);
    Printf(f_wrappers, "%s", docs);
    Delete(docs);

    current = CLASS_CONST;
    Language::memberconstantHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* ---------------------------------------------------------------------
   * staticmembervariableHandler()
   * --------------------------------------------------------------------- */

  virtual int staticmembervariableHandler(Node *n) {
    String* docs = docstring(n, AUTODOC_GETTER);
    Printf(f_wrappers, "%s", docs);
    Delete(docs);

    if (is_assignable(n)) {
      String* docs = docstring(n, AUTODOC_SETTER);
      Printf(f_wrappers, "%s", docs);
      Delete(docs);
    }

    current = STATIC_VAR;
    Language::staticmembervariableHandler(n);
    current = NO_CPP;
    return SWIG_OK;
  }

  /* C++ director class generation */
  virtual int classDirector(Node *n) {
    return Language::classDirector(n);
  }

  virtual int classDirectorInit(Node *n) {
    String *declaration;
    declaration = Swig_director_declaration(n);
    Printf(f_directors_h, "\n");
    Printf(f_directors_h, "%s\n", declaration);
    Printf(f_directors_h, "public:\n");
    Delete(declaration);
    return Language::classDirectorInit(n);
  }

  virtual int classDirectorEnd(Node *n) {
    Printf(f_directors_h, "};\n\n");
    return Language::classDirectorEnd(n);
  }

  /* ------------------------------------------------------------
   * classDirectorConstructor()
   * ------------------------------------------------------------ */

  virtual int classDirectorConstructor(Node *n) {
    Node *parent = Getattr(n, "parentNode");
    String *sub = NewString("");
    String *decl = Getattr(n, "decl");
    String *supername = Swig_class_name(parent);
    String *classname = NewString("");
    Printf(classname, "SwigDirector_%s", supername);

    /* insert self parameter */
    Parm *p;
    ParmList *superparms = Getattr(n, "parms");
    ParmList *parms = CopyParmList(superparms);
    String *type = NewString("VALUE");
    p = NewParm(type, NewString("self"), n);
    set_nextSibling(p, parms);
    parms = p;

    if (!Getattr(n, "defaultargs")) {
      /* constructor */
      {
	Wrapper *w = NewWrapper();
	String *call;
	String *basetype = Getattr(parent, "classtype");
	String *target = Swig_method_decl(0, decl, classname, parms, 0, 0);
	call = Swig_csuperclass_call(0, basetype, superparms);
	Printf(w->def, "%s::%s: %s, Swig::Director(self) { }", classname, target, call);
	Delete(target);
	Wrapper_print(w, f_directors);
	Delete(call);
	DelWrapper(w);
      }

      /* constructor header */
      {
	String *target = Swig_method_decl(0, decl, classname, parms, 0, 1);
	Printf(f_directors_h, "    %s;\n", target);
	Delete(target);
      }
    }

    Delete(sub);
    Delete(classname);
    Delete(supername);
    Delete(parms);
    return Language::classDirectorConstructor(n);
  }

  /* ------------------------------------------------------------
   * classDirectorDefaultConstructor()
   * ------------------------------------------------------------ */

  virtual int classDirectorDefaultConstructor(Node *n) {
    String *classname;
    Wrapper *w;
    classname = Swig_class_name(n);
    w = NewWrapper();
    Printf(w->def, "SwigDirector_%s::SwigDirector_%s(VALUE self) : Swig::Director(self) { }", classname, classname);
    Wrapper_print(w, f_directors);
    DelWrapper(w);
    Printf(f_directors_h, "    SwigDirector_%s(VALUE self);\n", classname);
    Delete(classname);
    return Language::classDirectorDefaultConstructor(n);
  }

  /* ---------------------------------------------------------------
   * exceptionSafeMethodCall()
   *
   * Emit a virtual director method to pass a method call on to the 
   * underlying Ruby instance.
   *
   * --------------------------------------------------------------- */

  void exceptionSafeMethodCall(String *className, Node *n, Wrapper *w, int argc, String *args, bool initstack) {
    Wrapper *body = NewWrapper();
    Wrapper *rescue = NewWrapper();

    String *methodName = Getattr(n, "sym:name");

    String *bodyName = NewStringf("%s_%s_body", className, methodName);
    String *rescueName = NewStringf("%s_%s_rescue", className, methodName);
    String *depthCountName = NewStringf("%s_%s_call_depth", className, methodName);

    // Check for an exception typemap of some kind
    String *tm = Swig_typemap_lookup("director:except", n, Swig_cresult_name(), 0);
    if (!tm) {
      tm = Getattr(n, "feature:director:except");
    }

    if ((tm != 0) && (Len(tm) > 0) && (Strcmp(tm, "1") != 0)) {
      // Declare a global to hold the depth count
      if (!Getattr(n, "sym:nextSibling")) {
	Printf(body->def, "static int %s = 0;\n", depthCountName);

	// Function body
	Printf(body->def, "VALUE %s(VALUE data) {\n", bodyName);
	Wrapper_add_localv(body, "args", "Swig::body_args *", "args", "= reinterpret_cast<Swig::body_args *>(data)", NIL);
	Wrapper_add_localv(body, Swig_cresult_name(), "VALUE", Swig_cresult_name(), "= Qnil", NIL);
	Printf(body->code, "%s++;\n", depthCountName);
	Printv(body->code, Swig_cresult_name(), " = rb_funcall2(args->recv, args->id, args->argc, args->argv);\n", NIL);
	Printf(body->code, "%s--;\n", depthCountName);
	Printv(body->code, "return ", Swig_cresult_name(), ";\n", NIL);
	Printv(body->code, "}", NIL);

	// Exception handler
	Printf(rescue->def, "VALUE %s(VALUE args, VALUE error) {\n", rescueName);
	Replaceall(tm, "$error", "error");
	Printf(rescue->code, "%s--;\n", depthCountName);
	Printf(rescue->code, "if (%s == 0) ", depthCountName);
	Printv(rescue->code, Str(tm), "\n", NIL);
	Printv(rescue->code, "rb_exc_raise(error);\n", NIL);
	Printv(rescue->code, "return Qnil;\n", NIL);
	Printv(rescue->code, "}", NIL);
      }

      // Main code
      Wrapper_add_localv(w, "args", "Swig::body_args", "args", NIL);
      Wrapper_add_localv(w, "status", "int", "status", NIL);
      Printv(w->code, "args.recv = swig_get_self();\n", NIL);
      Printf(w->code, "args.id = rb_intern(\"%s\");\n", methodName);
      Printf(w->code, "args.argc = %d;\n", argc);
      if (argc > 0) {
	Printf(w->code, "args.argv = new VALUE[%d];\n", argc);
	for (int i = 0; i < argc; i++) {
	  Printf(w->code, "args.argv[%d] = obj%d;\n", i, i);
	}
      } else {
	Printv(w->code, "args.argv = 0;\n", NIL);
      }
      Printf(w->code, "%s = rb_protect(PROTECTFUNC(%s), reinterpret_cast<VALUE>(&args), &status);\n", Swig_cresult_name(), bodyName);
      if ( initstack ) Printf(w->code, "SWIG_RELEASE_STACK;\n");
      Printf(w->code, "if (status) {\n");
      Printf(w->code, "VALUE lastErr = rb_gv_get(\"$!\");\n");
      Printf(w->code, "%s(reinterpret_cast<VALUE>(&args), lastErr);\n", rescueName);
      Printf(w->code, "}\n");
      if (argc > 0) {
	Printv(w->code, "delete [] args.argv;\n", NIL);
      }
      // Dump wrapper code
      Wrapper_print(body, f_directors_helpers);
      Wrapper_print(rescue, f_directors_helpers);
    } else {
      if (argc > 0) {
	Printf(w->code, "%s = rb_funcall(swig_get_self(), rb_intern(\"%s\"), %d%s);\n", Swig_cresult_name(), methodName, argc, args);
      } else {
	Printf(w->code, "%s = rb_funcall(swig_get_self(), rb_intern(\"%s\"), 0, NULL);\n", Swig_cresult_name(), methodName);
      }
      if ( initstack ) Printf(w->code, "SWIG_RELEASE_STACK;\n");
    }

    // Clean up
    Delete(bodyName);
    Delete(rescueName);
    Delete(depthCountName);
    DelWrapper(body);
    DelWrapper(rescue);
  }

  virtual int classDirectorMethod(Node *n, Node *parent, String *super) {
    int is_void = 0;
    int is_pointer = 0;
    String *decl = Getattr(n, "decl");
    String *name = Getattr(n, "name");
    String *classname = Getattr(parent, "sym:name");
    String *c_classname = Getattr(parent, "name");
    String *symname = Getattr(n, "sym:name");
    String *declaration = NewString("");
    ParmList *l = Getattr(n, "parms");
    Wrapper *w = NewWrapper();
    String *tm;
    String *wrap_args = NewString("");
    String *returntype = Getattr(n, "type");
    Parm *p;
    String *value = Getattr(n, "value");
    String *storage = Getattr(n, "storage");
    bool pure_virtual = false;
    int status = SWIG_OK;
    int idx;
    bool ignored_method = GetFlag(n, "feature:ignore") ? true : false;
    bool asvoid = checkAttribute( n, "feature:numoutputs", "0") ? true : false;
    bool initstack = checkAttribute( n, "feature:initstack", "1") ? true : false;

    if (Cmp(storage, "virtual") == 0) {
      if (Cmp(value, "0") == 0) {
	pure_virtual = true;
      }
    }
    String *overnametmp = NewString(Getattr(n, "sym:name"));
    if (Getattr(n, "sym:overloaded")) {
      Printf(overnametmp, "::%s", Getattr(n, "sym:overname"));
    }

    /* determine if the method returns a pointer */
    is_pointer = SwigType_ispointer_return(decl);
    is_void = (!Cmp(returntype, "void") && !is_pointer);

    /* virtual method definition */
    String *target;
    String *pclassname = NewStringf("SwigDirector_%s", classname);
    String *qualified_name = NewStringf("%s::%s", pclassname, name);
    SwigType *rtype = Getattr(n, "conversion_operator") ? 0 : Getattr(n, "classDirectorMethods:type");
    target = Swig_method_decl(rtype, decl, qualified_name, l, 0, 0);
    Printf(w->def, "%s", target);
    Delete(qualified_name);
    Delete(target);
    /* header declaration */
    target = Swig_method_decl(rtype, decl, name, l, 0, 1);
    Printf(declaration, "    virtual %s", target);
    Delete(target);

    // Get any exception classes in the throws typemap
    ParmList *throw_parm_list = 0;

    if ((throw_parm_list = Getattr(n, "throws")) || Getattr(n, "throw")) {
      Parm *p;
      int gencomma = 0;

      Append(w->def, " throw(");
      Append(declaration, " throw(");

      if (throw_parm_list)
	Swig_typemap_attach_parms("throws", throw_parm_list, 0);
      for (p = throw_parm_list; p; p = nextSibling(p)) {
	if (Getattr(p, "tmap:throws")) {
	  if (gencomma++) {
	    Append(w->def, ", ");
	    Append(declaration, ", ");
	  }

	  Printf(w->def, "%s", SwigType_str(Getattr(p, "type"), 0));
	  Printf(declaration, "%s", SwigType_str(Getattr(p, "type"), 0));
	}
      }

      Append(w->def, ")");
      Append(declaration, ")");
    }

    Append(w->def, " {");
    Append(declaration, ";\n");

    if (initstack && !(ignored_method && !pure_virtual)) {
      Append(w->def, "\nSWIG_INIT_STACK;\n");
    }

    /* declare method return value 
     * if the return value is a reference or const reference, a specialized typemap must
     * handle it, including declaration of c_result ($result).
     */
    if (!is_void) {
      if (!(ignored_method && !pure_virtual)) {
	Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), NIL);
      }
    }

    if (ignored_method) {
      if (!pure_virtual) {
	if (!is_void)
	  Printf(w->code, "return ");
	String *super_call = Swig_method_call(super, l);
	Printf(w->code, "%s;\n", super_call);
	Delete(super_call);
      } else {
	Printf(w->code, "Swig::DirectorPureVirtualException::raise(\"Attempted to invoke pure virtual method %s::%s\");\n", SwigType_namestr(c_classname),
	       SwigType_namestr(name));
      }
    } else {
      /* attach typemaps to arguments (C/C++ -> Ruby) */
      String *arglist = NewString("");

      Swig_director_parms_fixup(l);

      Swig_typemap_attach_parms("in", l, 0);
      Swig_typemap_attach_parms("directorin", l, 0);
      Swig_typemap_attach_parms("directorargout", l, w);

      char source[256];

      int outputs = 0;
      if (!is_void && !asvoid)
	outputs++;

      /* build argument list and type conversion string */
      idx = 0; p = l;
      while ( p ) {

	if (Getattr(p, "tmap:ignore")) {
	  p = Getattr(p, "tmap:ignore:next");
	  continue;
	}

	if (Getattr(p, "tmap:directorargout") != 0)
	  outputs++;

	if ( checkAttribute( p, "tmap:in:numinputs", "0") ) {
	  p = Getattr(p, "tmap:in:next");
	  continue;
	}

	String *parameterName = Getattr(p, "name");
	String *parameterType = Getattr(p, "type");

	Putc(',', arglist);
	if ((tm = Getattr(p, "tmap:directorin")) != 0) {
	  sprintf(source, "obj%d", idx++);
	  String *input = NewString(source);
	  Setattr(p, "emit:directorinput", input);
	  Replaceall(tm, "$input", input);
	  Replaceall(tm, "$owner", "0");
	  Delete(input);
	  Printv(wrap_args, tm, "\n", NIL);
	  Wrapper_add_localv(w, source, "VALUE", source, "= Qnil", NIL);
	  Printv(arglist, source, NIL);
	  p = Getattr(p, "tmap:directorin:next");
	  continue;
	} else if (Cmp(parameterType, "void")) {
	  /**
	   * Special handling for pointers to other C++ director classes.
	   * Ideally this would be left to a typemap, but there is currently no
	   * way to selectively apply the dynamic_cast<> to classes that have
	   * directors.  In other words, the type "SwigDirector_$1_lname" only exists
	   * for classes with directors.  We avoid the problem here by checking
	   * module.wrap::directormap, but it's not clear how to get a typemap to
	   * do something similar.  Perhaps a new default typemap (in addition
	   * to SWIGTYPE) called DIRECTORTYPE?
	   */
	  if (SwigType_ispointer(parameterType) || SwigType_isreference(parameterType)) {
	    Node *modname = Getattr(parent, "module");
	    Node *target = Swig_directormap(modname, parameterType);
	    sprintf(source, "obj%d", idx++);
	    String *nonconst = 0;
	    /* strip pointer/reference --- should move to Swig/stype.c */
	    String *nptype = NewString(Char(parameterType) + 2);
	    /* name as pointer */
	    String *ppname = Copy(parameterName);
	    if (SwigType_isreference(parameterType)) {
	      Insert(ppname, 0, "&");
	    }
	    /* if necessary, cast away const since Ruby doesn't support it! */
	    if (SwigType_isconst(nptype)) {
	      nonconst = NewStringf("nc_tmp_%s", parameterName);
	      String *nonconst_i = NewStringf("= const_cast< %s >(%s)", SwigType_lstr(parameterType, 0), ppname);
	      Wrapper_add_localv(w, nonconst, SwigType_lstr(parameterType, 0), nonconst, nonconst_i, NIL);
	      Delete(nonconst_i);
	      Swig_warning(WARN_LANG_DISCARD_CONST, input_file, line_number,
			   "Target language argument '%s' discards const in director method %s::%s.\n", SwigType_str(parameterType, parameterName),
			   SwigType_namestr(c_classname), SwigType_namestr(name));
	    } else {
	      nonconst = Copy(ppname);
	    }
	    Delete(nptype);
	    Delete(ppname);
	    String *mangle = SwigType_manglestr(parameterType);
	    if (target) {
	      String *director = NewStringf("director_%s", mangle);
	      Wrapper_add_localv(w, director, "Swig::Director *", director, "= 0", NIL);
	      Wrapper_add_localv(w, source, "VALUE", source, "= Qnil", NIL);
	      Printf(wrap_args, "%s = dynamic_cast<Swig::Director *>(%s);\n", director, nonconst);
	      Printf(wrap_args, "if (!%s) {\n", director);
	      Printf(wrap_args, "%s = SWIG_NewPointerObj(%s, SWIGTYPE%s, 0);\n", source, nonconst, mangle);
	      Printf(wrap_args, "} else {\n");
	      Printf(wrap_args, "%s = %s->swig_get_self();\n", source, director);
	      Printf(wrap_args, "}\n");
	      Delete(director);
	      Printv(arglist, source, NIL);
	    } else {
	      Wrapper_add_localv(w, source, "VALUE", source, "= Qnil", NIL);
	      Printf(wrap_args, "%s = SWIG_NewPointerObj(%s, SWIGTYPE%s, 0);\n", source, nonconst, mangle);
	      //Printf(wrap_args, "%s = SWIG_NewPointerObj(%s, SWIGTYPE_p_%s, 0);\n", 
	      //       source, nonconst, base);
	      Printv(arglist, source, NIL);
	    }
	    Delete(mangle);
	    Delete(nonconst);
	  } else {
	    Swig_warning(WARN_TYPEMAP_DIRECTORIN_UNDEF, input_file, line_number,
			 "Unable to use type %s as a function argument in director method %s::%s (skipping method).\n", SwigType_str(parameterType, 0),
			 SwigType_namestr(c_classname), SwigType_namestr(name));
	    status = SWIG_NOWRAP;
	    break;
	  }
	}
	p = nextSibling(p);
      }

      /* declare Ruby return value */
      String *value_result = NewStringf("VALUE SWIGUNUSED %s", Swig_cresult_name());
      Wrapper_add_local(w, Swig_cresult_name(), value_result);
      Delete(value_result);

      /* wrap complex arguments to VALUEs */
      Printv(w->code, wrap_args, NIL);

      /* pass the method call on to the Ruby object */
      exceptionSafeMethodCall(classname, n, w, idx, arglist, initstack);

      /*
       * Ruby method may return a simple object, or an Array of objects.
       * For in/out arguments, we have to extract the appropriate VALUEs from the Array,
       * then marshal everything back to C/C++ (return value and output arguments).
       */

      /* Marshal return value and other outputs (if any) from VALUE to C/C++ type */

      String *cleanup = NewString("");
      String *outarg = NewString("");

      if (outputs > 1) {
	Wrapper_add_local(w, "output", "VALUE output");
	Printf(w->code, "if (TYPE(%s) != T_ARRAY) {\n", Swig_cresult_name());
	Printf(w->code, "Ruby_DirectorTypeMismatchException(\"Ruby method failed to return an array.\");\n");
	Printf(w->code, "}\n");
      }

      idx = 0;

      /* Marshal return value */
      if (!is_void) {
	tm = Swig_typemap_lookup("directorout", n, Swig_cresult_name(), w);
	if (tm != 0) {
	  if (outputs > 1 && !asvoid ) {
	    Printf(w->code, "output = rb_ary_entry(%s, %d);\n", Swig_cresult_name(), idx++);
	    Replaceall(tm, "$input", "output");
	  } else {
	    Replaceall(tm, "$input", Swig_cresult_name());
	  }
	  /* TODO check this */
	  if (Getattr(n, "wrap:disown")) {
	    Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
	  } else {
	    Replaceall(tm, "$disown", "0");
	  }
	  Replaceall(tm, "$result", "c_result");
	  Printv(w->code, tm, "\n", NIL);
	} else {
	  Swig_warning(WARN_TYPEMAP_DIRECTOROUT_UNDEF, input_file, line_number,
		       "Unable to use return type %s in director method %s::%s (skipping method).\n", SwigType_str(returntype, 0),
		       SwigType_namestr(c_classname), SwigType_namestr(name));
	  status = SWIG_ERROR;
	}
      }

      /* Marshal outputs */
      for (p = l; p;) {
	if ((tm = Getattr(p, "tmap:directorargout")) != 0) {
	  if (outputs > 1) {
	    Printf(w->code, "output = rb_ary_entry(%s, %d);\n", Swig_cresult_name(), idx++);
	    Replaceall(tm, "$result", "output");
	  } else {
	    Replaceall(tm, "$result", Swig_cresult_name());
	  }
	  Replaceall(tm, "$input", Getattr(p, "emit:directorinput"));
	  Printv(w->code, tm, "\n", NIL);
	  p = Getattr(p, "tmap:directorargout:next");
	} else {
	  p = nextSibling(p);
	}
      }

      Delete(arglist);
      Delete(cleanup);
      Delete(outarg);
    }

    /* any existing helper functions to handle this? */
    if (!is_void) {
      if (!(ignored_method && !pure_virtual)) {
	String *rettype = SwigType_str(returntype, 0);
	if (!SwigType_isreference(returntype)) {
	  Printf(w->code, "return (%s) c_result;\n", rettype);
	} else {
	  Printf(w->code, "return (%s) *c_result;\n", rettype);
	}
	Delete(rettype);
      }
    }

    Printf(w->code, "}\n");

    // We expose protected methods via an extra public inline method which makes a straight call to the wrapped class' method
    String *inline_extra_method = NewString("");
    if (dirprot_mode() && !is_public(n) && !pure_virtual) {
      Printv(inline_extra_method, declaration, NIL);
      String *extra_method_name = NewStringf("%sSwigPublic", name);
      Replaceall(inline_extra_method, name, extra_method_name);
      Replaceall(inline_extra_method, ";\n", " {\n      ");
      if (!is_void)
	Printf(inline_extra_method, "return ");
      String *methodcall = Swig_method_call(super, l);
      Printv(inline_extra_method, methodcall, ";\n    }\n", NIL);
      Delete(methodcall);
      Delete(extra_method_name);
    }

    /* emit the director method */
    if (status == SWIG_OK) {
      if (!Getattr(n, "defaultargs")) {
	Replaceall(w->code, "$symname", symname);
	Wrapper_print(w, f_directors);
	Printv(f_directors_h, declaration, NIL);
	Printv(f_directors_h, inline_extra_method, NIL);
      }
    }

    /* clean up */
    Delete(wrap_args);
    Delete(pclassname);
    DelWrapper(w);
    return status;
  }

  virtual int classDirectorConstructors(Node *n) {
    return Language::classDirectorConstructors(n);
  }

  virtual int classDirectorMethods(Node *n) {
    return Language::classDirectorMethods(n);
  }

  virtual int classDirectorDisown(Node *n) {
    return Language::classDirectorDisown(n);
  }

  String *runtimeCode() {
    String *s = NewString("");
    String *shead = Swig_include_sys("rubyhead.swg");
    if (!shead) {
      Printf(stderr, "*** Unable to open 'rubyhead.swg'\n");
    } else {
      Append(s, shead);
      Delete(shead);
    }
    String *serrors = Swig_include_sys("rubyerrors.swg");
    if (!serrors) {
      Printf(stderr, "*** Unable to open 'rubyerrors.swg'\n");
    } else {
      Append(s, serrors);
      Delete(serrors);
    }
    String *strack = Swig_include_sys("rubytracking.swg");
    if (!strack) {
      Printf(stderr, "*** Unable to open 'rubytracking.swg'\n");
    } else {
      Append(s, strack);
      Delete(strack);
    }
    String *sapi = Swig_include_sys("rubyapi.swg");
    if (!sapi) {
      Printf(stderr, "*** Unable to open 'rubyapi.swg'\n");
    } else {
      Append(s, sapi);
      Delete(sapi);
    }
    String *srun = Swig_include_sys("rubyrun.swg");
    if (!srun) {
      Printf(stderr, "*** Unable to open 'rubyrun.swg'\n");
    } else {
      Append(s, srun);
      Delete(srun);
    }
    return s;
  }

  String *defaultExternalRuntimeFilename() {
    return NewString("swigrubyrun.h");
  }

  /*----------------------------------------------------------------------
   * kwargsSupport()
   *--------------------------------------------------------------------*/

  bool kwargsSupport() const {
    // kwargs support isn't actually implemented, but changing to return false may break something now as it turns on compactdefaultargs
    return true;
  }
};				/* class RUBY */

/* -----------------------------------------------------------------------------
 * swig_ruby()    - Instantiate module
 * ----------------------------------------------------------------------------- */

static Language *new_swig_ruby() {
  return new RUBY();
}
extern "C" Language *swig_ruby(void) {
  return new_swig_ruby();
}


/*
 * Local Variables:
 * c-basic-offset: 2
 * End:
 */
