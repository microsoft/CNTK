/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * octave.cxx
 *
 * Octave language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"

static String *global_name = 0;
static String *op_prefix   = 0;

static const char *usage = "\
Octave Options (available with -octave)\n\
     -cppcast        - Enable C++ casting operators (default)\n\
     -globals <name> - Set <name> used to access C global variables [default: 'cvar']\n\
                       Use '.' to load C global variables into module namespace\n\
     -nocppcast      - Disable C++ casting operators\n\
     -opprefix <str> - Prefix <str> for global operator functions [default: 'op_']\n\
\n";


class OCTAVE:public Language {
private:
  File *f_begin;
  File *f_runtime;
  File *f_header;
  File *f_doc;
  File *f_wrappers;
  File *f_init;
  File *f_initbeforefunc;
  File *f_directors;
  File *f_directors_h;
  String *s_global_tab;
  String *s_members_tab;
  String *class_name;

  int have_constructor;
  int have_destructor;
  String *constructor_name;

  Hash *docs;

  void Octave_begin_function(Node *n, File *f, const_String_or_char_ptr cname, const_String_or_char_ptr wname, bool dld) {
    if (dld) {
      String *tname = texinfo_name(n, "std::string()");
      Printf(f, "SWIG_DEFUN( %s, %s, %s ) {", cname, wname, tname);
    }
    else {
      Printf(f, "static octave_value_list %s (const octave_value_list& args, int nargout) {", wname);
    }
  }

public:
  OCTAVE():
    f_begin(0),
    f_runtime(0),
    f_header(0),
    f_doc(0),
    f_wrappers(0),
    f_init(0),
    f_initbeforefunc(0),
    f_directors(0),
    f_directors_h(0),
    s_global_tab(0),
    s_members_tab(0),
    class_name(0),
    have_constructor(0),
    have_destructor(0),
    constructor_name(0),
    docs(0)
  {
    /* Add code to manage protected constructors and directors */
    director_prot_ctor_code = NewString("");
    Printv(director_prot_ctor_code,
           "if ( $comparison ) { /* subclassed */\n",
           "  $director_new \n",
           "} else {\n", "  error(\"accessing abstract class or protected constructor\"); \n", "  SWIG_fail;\n", "}\n", NIL);

    enable_cplus_runtime_mode();
    allow_overloading();
    director_multiple_inheritance = 1;
    director_language = 1;
    docs = NewHash();
  }

  virtual void main(int argc, char *argv[]) {
    int cppcast = 1;
      
    for (int i = 1; i < argc; i++) {
      if (argv[i]) {
        if (strcmp(argv[i], "-help") == 0) {
          fputs(usage, stdout);
        } else if (strcmp(argv[i], "-globals") == 0) {
          if (argv[i + 1]) {
            global_name = NewString(argv[i + 1]);
            Swig_mark_arg(i);
            Swig_mark_arg(i + 1);
            i++;
          } else {
            Swig_arg_error();
          }
        } else if (strcmp(argv[i], "-opprefix") == 0) {
          if (argv[i + 1]) {
            op_prefix = NewString(argv[i + 1]);
            Swig_mark_arg(i);
            Swig_mark_arg(i + 1);
            i++;
          } else {
            Swig_arg_error();
          }
        } else if (strcmp(argv[i], "-cppcast") == 0) {
 	  cppcast = 1;
 	  Swig_mark_arg(i);
 	} else if (strcmp(argv[i], "-nocppcast") == 0) {
 	  cppcast = 0;
 	  Swig_mark_arg(i);
        }
      }
    }

    if (!global_name)
      global_name = NewString("cvar");
    if (!op_prefix)
      op_prefix = NewString("op_");
    if(cppcast)
      Preprocessor_define((DOH *) "SWIG_CPLUSPLUS_CAST", 0);

    SWIG_library_directory("octave");
    Preprocessor_define("SWIGOCTAVE 1", 0);
    SWIG_config_file("octave.swg");
    SWIG_typemap_lang("octave");
    allow_overloading();

    // Octave API is C++, so output must be C++ compatibile even when wrapping C code
    if (!cparse_cplusplus)
      Swig_cparse_cplusplusout(1);
  }

  virtual int top(Node *n) {
    {
      Node *mod = Getattr(n, "module");
      if (mod) {
        Node *options = Getattr(mod, "options");
        if (options) {
          int dirprot = 0;
          if (Getattr(options, "dirprot")) {
            dirprot = 1;
          }
          if (Getattr(options, "nodirprot")) {
            dirprot = 0;
          }
          if (Getattr(options, "directors")) {
            allow_directors();
            if (dirprot)
              allow_dirprot();
          }
        }
      }
    }

    String *module = Getattr(n, "name");
    String *outfile = Getattr(n, "outfile");
    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      SWIG_exit(EXIT_FAILURE);
    }
    f_runtime = NewString("");
    f_header = NewString("");
    f_doc = NewString("");
    f_wrappers = NewString("");
    f_init = NewString("");
    f_initbeforefunc = NewString("");
    f_directors_h = NewString("");
    f_directors = NewString("");
    s_global_tab = NewString("");
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("header", f_header);
    Swig_register_filebyname("doc", f_doc);
    Swig_register_filebyname("wrapper", f_wrappers);
    Swig_register_filebyname("init", f_init);
    Swig_register_filebyname("initbeforefunc", f_initbeforefunc);
    Swig_register_filebyname("director", f_directors);
    Swig_register_filebyname("director_h", f_directors_h);

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGOCTAVE\n#define SWIGOCTAVE\n#endif\n\n");

    Printf(f_runtime, "#define SWIG_name_d      \"%s\"\n", module);
    Printf(f_runtime, "#define SWIG_name        %s\n", module);

    Printf(f_runtime, "\n");
    Printf(f_runtime, "#define SWIG_global_name      \"%s\"\n", global_name);
    Printf(f_runtime, "#define SWIG_op_prefix        \"%s\"\n", op_prefix);

    if (directorsEnabled()) {
      Printf(f_runtime, "#define SWIG_DIRECTORS\n");
      Swig_banner(f_directors_h);
      if (dirprot_mode()) {
        //      Printf(f_directors_h, "#include <map>\n");
        //      Printf(f_directors_h, "#include <string>\n\n");
      }
    }

    Printf(f_runtime, "\n");

    Printf(s_global_tab, "\nstatic const struct swig_octave_member swig_globals[] = {\n");
    Printf(f_init, "static bool SWIG_init_user(octave_swig_type* module_ns)\n{\n");

    if (!CPlusPlus)
      Printf(f_header,"extern \"C\" {\n");

    Language::top(n);

    if (!CPlusPlus)
      Printf(f_header,"}\n");

    if (Len(docs))
      emit_doc_texinfo();

    if (directorsEnabled()) {
      Swig_insert_file("director_common.swg", f_runtime);
      Swig_insert_file("director.swg", f_runtime);
    }

    Printf(f_init, "return true;\n}\n");
    Printf(s_global_tab, "{0,0,0,0,0,0}\n};\n");

    Printv(f_wrappers, s_global_tab, NIL);
    SwigType_emit_type_table(f_runtime, f_wrappers);
    Dump(f_runtime, f_begin);
    Dump(f_header, f_begin);
    Dump(f_doc, f_begin);
    if (directorsEnabled()) {
      Dump(f_directors_h, f_begin);
      Dump(f_directors, f_begin);
    }
    Dump(f_wrappers, f_begin);
    Dump(f_initbeforefunc, f_begin);
    Wrapper_pretty_print(f_init, f_begin);

    Delete(s_global_tab);
    Delete(f_initbeforefunc);
    Delete(f_init);
    Delete(f_wrappers);
    Delete(f_doc);
    Delete(f_header);
    Delete(f_directors);
    Delete(f_directors_h);
    Delete(f_runtime);
    Delete(f_begin);

    return SWIG_OK;
  }

  String *texinfo_escape(String *_s) {
    const char* s=(const char*)Data(_s);
    while (*s&&(*s=='\t'||*s=='\r'||*s=='\n'||*s==' '))
      ++s;
    String *r = NewString("");
    for (int j=0;s[j];++j) {
      if (s[j] == '\n') {
        Append(r, "\\n\\\n");
      } else if (s[j] == '\r') {
        Append(r, "\\r");
      } else if (s[j] == '\t') {
        Append(r, "\\t");
      } else if (s[j] == '\\') {
        Append(r, "\\\\");
      } else if (s[j] == '\'') {
        Append(r, "\\\'");
      } else if (s[j] == '\"') {
        Append(r, "\\\"");
      } else
        Putc(s[j], r);
    }
    return r;
  }
  void emit_doc_texinfo() {
    for (Iterator it = First(docs); it.key; it = Next(it)) {
      String *wrap_name = it.key;

      String *synopsis = Getattr(it.item, "synopsis");
      String *decl_info = Getattr(it.item, "decl_info");
      String *cdecl_info = Getattr(it.item, "cdecl_info");
      String *args_info = Getattr(it.item, "args_info");

      String *doc_str = NewString("");
      Printv(doc_str, synopsis, decl_info, cdecl_info, args_info, NIL);
      String *escaped_doc_str = texinfo_escape(doc_str);

      if (Len(doc_str)>0) {
        Printf(f_doc,"static const char* %s_texinfo = ",wrap_name);
        Printf(f_doc,"\"-*- texinfo -*-\\n\\\n%s", escaped_doc_str);
        if (Len(decl_info))
          Printf(f_doc,"\\n\\\n@end deftypefn");
        Printf(f_doc,"\";\n");
      }

      Delete(escaped_doc_str);
      Delete(doc_str);
      Delete(wrap_name);
    }
    Printf(f_doc,"\n");
  }
  bool is_empty_doc_node(Node* n) {
    if (!n)
      return true;
    String *synopsis = Getattr(n, "synopsis");
    String *decl_info = Getattr(n, "decl_info");
    String *cdecl_info = Getattr(n, "cdecl_info");
    String *args_info = Getattr(n, "args_info");
    return !Len(synopsis) && !Len(decl_info) &&
      !Len(cdecl_info) && !Len(args_info);
  }
  String *texinfo_name(Node* n, const char* defval = "0") {
    String *tname = NewString("");
    String *iname = Getattr(n, "sym:name");
    String *wname = Swig_name_wrapper(iname);
    Node* d = Getattr(docs, wname);

    if (is_empty_doc_node(d))
      Printf(tname, defval);
    else
      Printf(tname, "%s_texinfo", wname);

    return tname;
  }
  void process_autodoc(Node *n) {
    String *iname = Getattr(n, "sym:name");
    String *name = Getattr(n, "name");
    String *wname = Swig_name_wrapper(iname);
    String *str = Getattr(n, "feature:docstring");
    bool autodoc_enabled = !Cmp(Getattr(n, "feature:autodoc"), "1");
    Node* d = Getattr(docs, wname);
    if (!d) {
      d = NewHash();
      Setattr(d, "synopsis", NewString(""));
      Setattr(d, "decl_info", NewString(""));
      Setattr(d, "cdecl_info", NewString(""));
      Setattr(d, "args_info", NewString(""));
      Setattr(docs, wname, d);
    }

    String *synopsis = Getattr(d, "synopsis");
    String *decl_info = Getattr(d, "decl_info");
    //    String *cdecl_info = Getattr(d, "cdecl_info");
    String *args_info = Getattr(d, "args_info");

    // * couldn't we just emit the docs here?

    if (autodoc_enabled) {
      String *decl_str = NewString("");
      String *args_str = NewString("");
      make_autodocParmList(n, decl_str, args_str);
      Append(decl_info, "@deftypefn {Loadable Function} ");

      SwigType *type = Getattr(n, "type");
      if (type && Strcmp(type, "void")) {
        Node *nn = classLookup(Getattr(n, "type"));
        String *type_str = nn ? Copy(Getattr(nn, "sym:name")) : SwigType_str(type, 0);
        Append(decl_info, "@var{retval} = ");
        Printf(args_str, "%s@var{retval} is of type %s. ", args_str, type_str);
        Delete(type_str);
      }

      Append(decl_info, name);
      Append(decl_info, " (");
      Append(decl_info, decl_str);
      Append(decl_info, ")\n");
      Append(args_info, args_str);
      Delete(decl_str);
      Delete(args_str);
    }

    if (str && Len(str) > 0) {
      // strip off {} if necessary
      char *t = Char(str);
      if (*t == '{') {
        Delitem(str, 0);
        Delitem(str, DOH_END);
      }

      // emit into synopsis section
      Append(synopsis, str);
    }
  }

  virtual int importDirective(Node *n) {
    String *modname = Getattr(n, "module");
    if (modname)
      Printf(f_init, "if (!SWIG_Octave_LoadModule(\"%s\")) return false;\n", modname);
    return Language::importDirective(n);
  }

  const char *get_implicitconv_flag(Node *n) {
    int conv = 0;
    if (n && GetFlag(n, "feature:implicitconv")) {
      conv = 1;
    }
    return conv ? "SWIG_POINTER_IMPLICIT_CONV" : "0";
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

  void make_autodocParmList(Node *n, String *decl_str, String *args_str) {
    String *pdocs = 0;
    ParmList *plist = CopyParmList(Getattr(n, "parms"));
    Parm *p;
    Parm *pnext;
    int start_arg_num = is_wrapping_class() ? 1 : 0;

    addMissingParameterNames(plist, start_arg_num); // for $1_name substitutions done in Swig_typemap_attach_parms

    Swig_typemap_attach_parms("in", plist, 0);
    Swig_typemap_attach_parms("doc", plist, 0);

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

      name = name ? name : Getattr(p, "name");
      name = name ? name : Getattr(p, "lname");
      name = Swig_name_make(p, 0, name, 0, 0); // rename parameter if a keyword

      type = type ? type : Getattr(p, "type");
      value = value ? value : Getattr(p, "value");

      if (SwigType_isvarargs(type))
        break;

      String *tex_name = NewString("");
      if (name)
        Printf(tex_name, "@var{%s}", name);
      else
        Printf(tex_name, "@var{?}");

      if (Len(decl_str))
        Append(decl_str, ", ");
      Append(decl_str, tex_name);

      if (value) {
        String *new_value = convertValue(value, Getattr(p, "type"));
        if (new_value) {
          value = new_value;
        } else {
          Node *lookup = Swig_symbol_clookup(value, 0);
          if (lookup)
            value = Getattr(lookup, "sym:name");
        }
        Printf(decl_str, " = %s", value);
      }

      Node *nn = classLookup(Getattr(p, "type"));
      String *type_str = nn ? Copy(Getattr(nn, "sym:name")) : SwigType_str(type, 0);
      Printf(args_str, "%s is of type %s. ", tex_name, type_str);

      Delete(type_str);
      Delete(tex_name);
      Delete(name);
    }
    if (pdocs)
      Setattr(n, "feature:pdocs", pdocs);
    Delete(plist);
  }

  /* ------------------------------------------------------------
   * convertValue()
   *    Check if string v can be an Octave value literal,
   *    (eg. number or string), or translate it to an Octave literal.
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
        return NewString("true");
      if (Strcmp(v, "false") == 0 || Strcmp(v, "FALSE") == 0)
        return NewString("false");
    }
    return 0;
  }

  virtual int functionWrapper(Node *n) {
    Parm *p;
    String *tm;
    int j;

    String *nodeType = Getattr(n, "nodeType");
    int constructor = (!Cmp(nodeType, "constructor"));
    int destructor = (!Cmp(nodeType, "destructor"));
    String *storage = Getattr(n, "storage");

    bool overloaded = !!Getattr(n, "sym:overloaded");
    bool last_overload = overloaded && !Getattr(n, "sym:nextSibling");
    String *iname = Getattr(n, "sym:name");
    String *wname = Swig_name_wrapper(iname);
    String *overname = Copy(wname);
    SwigType *d = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");

    if (!overloaded && !addSymbol(iname, n))
      return SWIG_ERROR;

    if (overloaded)
      Append(overname, Getattr(n, "sym:overname"));

    if (!overloaded || last_overload)
      process_autodoc(n);

    Wrapper *f = NewWrapper();
    Octave_begin_function(n, f->def, iname, overname, !overloaded);

    emit_parameter_variables(l, f);
    emit_attach_parmmaps(l, f);
    Setattr(n, "wrap:parms", l);

    int num_arguments = emit_num_arguments(l);
    int num_required = emit_num_required(l);
    int varargs = emit_isvarargs(l);
    char source[64];

    Printf(f->code, "if (!SWIG_check_num_args(\"%s\",args.length(),%i,%i,%i)) "
           "{\n SWIG_fail;\n }\n", iname, num_arguments, num_required, varargs);

    if (constructor && num_arguments == 1 && num_required == 1) {
      if (Cmp(storage, "explicit") == 0) {
        Node *parent = Swig_methodclass(n);
        if (GetFlag(parent, "feature:implicitconv")) {
          String *desc = NewStringf("SWIGTYPE%s", SwigType_manglestr(Getattr(n, "type")));
          Printf(f->code, "if (SWIG_CheckImplicit(%s)) SWIG_fail;\n", desc);
          Delete(desc);
        }
      }
    }

    for (j = 0, p = l; j < num_arguments; ++j) {
      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
        p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");

      String *tm = Getattr(p, "tmap:in");
      if (tm) {
        if (!tm || checkAttribute(p, "tmap:in:numinputs", "0")) {
          p = nextSibling(p);
          continue;
        }

        sprintf(source, "args(%d)", j);
        Setattr(p, "emit:input", source);

        Replaceall(tm, "$source", Getattr(p, "emit:input"));
        Replaceall(tm, "$input", Getattr(p, "emit:input"));
        Replaceall(tm, "$target", Getattr(p, "lname"));

        if (Getattr(p, "wrap:disown") || (Getattr(p, "tmap:in:disown"))) {
          Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
        } else {
          Replaceall(tm, "$disown", "0");
        }

        if (Getattr(p, "tmap:in:implicitconv")) {
          const char *convflag = "0";
          if (!Getattr(p, "hidden")) {
            SwigType *ptype = Getattr(p, "type");
            convflag = get_implicitconv_flag(classLookup(ptype));
          }
          Replaceall(tm, "$implicitconv", convflag);
          Setattr(p, "implicitconv", convflag);
        }

        String *getargs = NewString("");
        if (j >= num_required)
          Printf(getargs, "if (%d<args.length()) {\n%s\n}", j, tm);
        else
          Printv(getargs, tm, NIL);
        Printv(f->code, getargs, "\n", NIL);
        Delete(getargs);

        p = Getattr(p, "tmap:in:next");
        continue;
      } else {
        Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
        break;
      }
    }

    // Check for trailing varargs
    if (varargs) {
      if (p && (tm = Getattr(p, "tmap:in"))) {
        Replaceall(tm, "$input", "varargs");
        Printv(f->code, tm, "\n", NIL);
      }
    }

    // Insert constraint checking code
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:check"))) {
        Replaceall(tm, "$target", Getattr(p, "lname"));
        Printv(f->code, tm, "\n", NIL);
        p = Getattr(p, "tmap:check:next");
      } else {
        p = nextSibling(p);
      }
    }

    // Insert cleanup code
    String *cleanup = NewString("");
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:freearg"))) {
        if (Getattr(p, "tmap:freearg:implicitconv")) {
          const char *convflag = "0";
          if (!Getattr(p, "hidden")) {
            SwigType *ptype = Getattr(p, "type");
            convflag = get_implicitconv_flag(classLookup(ptype));
          }
          if (strcmp(convflag, "0") == 0) {
            tm = 0;
          }
        }
        if (tm && (Len(tm) != 0)) {
          Replaceall(tm, "$source", Getattr(p, "lname"));
          Printv(cleanup, tm, "\n", NIL);
        }
        p = Getattr(p, "tmap:freearg:next");
      } else {
        p = nextSibling(p);
      }
    }

    // Insert argument output code
    String *outarg = NewString("");
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:argout"))) {
        Replaceall(tm, "$source", Getattr(p, "lname"));
        Replaceall(tm, "$target", "_outp");
        Replaceall(tm, "$result", "_outp");
        Replaceall(tm, "$arg", Getattr(p, "emit:input"));
        Replaceall(tm, "$input", Getattr(p, "emit:input"));
        Printv(outarg, tm, "\n", NIL);
        p = Getattr(p, "tmap:argout:next");
      } else {
        p = nextSibling(p);
      }
    }

    int director_method = is_member_director(n) && !is_smart_pointer() && !destructor;
    if (director_method) {
      Wrapper_add_local(f, "upcall", "bool upcall = false");
      Append(f->code, "upcall = !!dynamic_cast<Swig::Director*>(arg1);\n");
    }

    Setattr(n, "wrap:name", overname);

    Swig_director_emit_dynamic_cast(n, f);
    String *actioncode = emit_action(n);

    Wrapper_add_local(f, "_out", "octave_value_list _out");
    Wrapper_add_local(f, "_outp", "octave_value_list *_outp=&_out");
    Wrapper_add_local(f, "_outv", "octave_value _outv");

    // Return the function value
    if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
      Replaceall(tm, "$source", Swig_cresult_name());
      Replaceall(tm, "$target", "_outv");
      Replaceall(tm, "$result", "_outv");

      if (GetFlag(n, "feature:new"))
        Replaceall(tm, "$owner", "1");
      else
        Replaceall(tm, "$owner", "0");

      Printf(f->code, "%s\n", tm);
      Printf(f->code, "if (_outv.is_defined()) _outp = " "SWIG_Octave_AppendOutput(_outp, _outv);\n");
      Delete(tm);
    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(d, 0), iname);
    }
    emit_return_variable(n, d, f);

    Printv(f->code, outarg, NIL);
    Printv(f->code, cleanup, NIL);

    if (GetFlag(n, "feature:new")) {
      if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
        Replaceall(tm, "$source", Swig_cresult_name());
        Printf(f->code, "%s\n", tm);
      }
    }

    if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
      Replaceall(tm, "$source", Swig_cresult_name());
      Replaceall(tm, "$result", "_outv");
      Printf(f->code, "%s\n", tm);
      Delete(tm);
    }

    Printf(f->code, "return _out;\n");
    Printf(f->code, "fail:\n");	// we should free locals etc if this happens
    Printv(f->code, cleanup, NIL);
    Printf(f->code, "return octave_value_list();\n");
    Printf(f->code, "}\n");

    /* Substitute the cleanup code */
    Replaceall(f->code, "$cleanup", cleanup);

    Replaceall(f->code, "$symname", iname);
    Wrapper_print(f, f_wrappers);
    DelWrapper(f);

    if (last_overload)
      dispatchFunction(n);

    if (!overloaded || last_overload) {
      String *tname = texinfo_name(n);
      Printf(s_global_tab, "{\"%s\",%s,0,0,2,%s},\n", iname, wname, tname);
      Delete(tname);
    }

    Delete(overname);
    Delete(wname);
    Delete(cleanup);
    Delete(outarg);

    return SWIG_OK;
  }

  void dispatchFunction(Node *n) {
    Wrapper *f = NewWrapper();

    String *iname = Getattr(n, "sym:name");
    String *wname = Swig_name_wrapper(iname);
    int maxargs;
    String *dispatch = Swig_overload_dispatch(n, "return %s(args, nargout);", &maxargs);
    String *tmp = NewString("");

    Octave_begin_function(n, f->def, iname, wname, true);
    Wrapper_add_local(f, "argc", "int argc = args.length()");
    Printf(tmp, "octave_value_ref argv[%d]={", maxargs);
    for (int j = 0; j < maxargs; ++j)
      Printf(tmp, "%soctave_value_ref(args,%d)", j ? "," : " ", j);
    Printf(tmp, "}");
    Wrapper_add_local(f, "argv", tmp);
    Printv(f->code, dispatch, "\n", NIL);
    Printf(f->code, "error(\"No matching function for overload\");\n", iname);
    Printf(f->code, "return octave_value_list();\n");
    Printv(f->code, "}\n", NIL);

    Wrapper_print(f, f_wrappers);
    Delete(tmp);
    DelWrapper(f);
    Delete(dispatch);
    Delete(wname);
  }

  virtual int variableWrapper(Node *n) {
    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *t = Getattr(n, "type");

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    String *tm;
    Wrapper *getf = NewWrapper();
    Wrapper *setf = NewWrapper();

    String *getname = Swig_name_get(NSPACE_TODO, iname);
    String *setname = Swig_name_set(NSPACE_TODO, iname);

    String *getwname = Swig_name_wrapper(getname);
    String *setwname = Swig_name_wrapper(setname);

    Octave_begin_function(n, setf->def, setname, setwname, true);
    Printf(setf->def, "if (!SWIG_check_num_args(\"%s_set\",args.length(),1,1,0)) return octave_value_list();", iname);
    if (is_assignable(n)) {
      Setattr(n, "wrap:name", setname);
      if ((tm = Swig_typemap_lookup("varin", n, name, 0))) {
        Replaceall(tm, "$source", "args(0)");
        Replaceall(tm, "$target", name);
        Replaceall(tm, "$input", "args(0)");
        if (Getattr(n, "tmap:varin:implicitconv")) {
          Replaceall(tm, "$implicitconv", get_implicitconv_flag(n));
        }
        emit_action_code(n, setf->code, tm);
        Delete(tm);
      } else {
        Swig_warning(WARN_TYPEMAP_VARIN_UNDEF, input_file, line_number, "Unable to set variable of type %s.\n", SwigType_str(t, 0));
      }
      Append(setf->code, "fail:\n");
      Printf(setf->code, "return octave_value_list();\n");
    } else {
      Printf(setf->code, "return octave_set_immutable(args,nargout);");
    }
    Append(setf->code, "}\n");
    Wrapper_print(setf, f_wrappers);

    Setattr(n, "wrap:name", getname);
    int addfail = 0;
    Octave_begin_function(n, getf->def, getname, getwname, true);
    Wrapper_add_local(getf, "obj", "octave_value obj");
    if ((tm = Swig_typemap_lookup("varout", n, name, 0))) {
      Replaceall(tm, "$source", name);
      Replaceall(tm, "$target", "obj");
      Replaceall(tm, "$result", "obj");
      addfail = emit_action_code(n, getf->code, tm);
      Delete(tm);
    } else {
      Swig_warning(WARN_TYPEMAP_VAROUT_UNDEF, input_file, line_number, "Unable to read variable of type %s\n", SwigType_str(t, 0));
    }
    Append(getf->code, "  return obj;\n");
    if (addfail) {
      Append(getf->code, "fail:\n");
      Append(getf->code, "  return octave_value_list();\n");
    }
    Append(getf->code, "}\n");
    Wrapper_print(getf, f_wrappers);

    Printf(s_global_tab, "{\"%s\",0,%s,%s,2,0},\n", iname, getwname, setwname);

    Delete(getwname);
    Delete(setwname);
    DelWrapper(setf);
    DelWrapper(getf);

    return SWIG_OK;
  }

  virtual int constantWrapper(Node *n) {
    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");
    String *cppvalue = Getattr(n, "cppvalue");
    String *tm;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    if (SwigType_type(type) == T_MPOINTER) {
      String *wname = Swig_name_wrapper(iname);
      String *str = SwigType_str(type, wname);
      Printf(f_header, "static %s = %s;\n", str, value);
      Delete(str);
      value = wname;
    }
    if ((tm = Swig_typemap_lookup("constcode", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", name);
      Replaceall(tm, "$value", cppvalue ? cppvalue : value);
      Replaceall(tm, "$nsname", iname);
      Printf(f_init, "%s\n", tm);
    } else {
      Swig_warning(WARN_TYPEMAP_CONST_UNDEF, input_file, line_number, "Unsupported constant value.\n");
      return SWIG_NOWRAP;
    }

    return SWIG_OK;
  }

  virtual int nativeWrapper(Node *n) {
    return Language::nativeWrapper(n);
  }

  virtual int enumDeclaration(Node *n) {
    return Language::enumDeclaration(n);
  }

  virtual int enumvalueDeclaration(Node *n) {
    return Language::enumvalueDeclaration(n);
  }

  virtual int classDeclaration(Node *n) {
    return Language::classDeclaration(n);
  }

  virtual int classHandler(Node *n) {
    have_constructor = 0;
    have_destructor = 0;
    constructor_name = 0;

    class_name = Getattr(n, "sym:name");

    if (!addSymbol(class_name, n))
      return SWIG_ERROR;

    // This is a bug, due to the fact that swig_type -> octave_class mapping
    // is 1-to-n.
    static Hash *emitted = NewHash();
    String *mangled_classname = Swig_name_mangle(Getattr(n, "name"));
    if (Getattr(emitted, mangled_classname)) {
      Delete(mangled_classname);
      return SWIG_NOWRAP;
    }
    Setattr(emitted, mangled_classname, "1");
    Delete(mangled_classname);

    assert(!s_members_tab);
    s_members_tab = NewString("");
    Printv(s_members_tab, "static swig_octave_member swig_", class_name, "_members[] = {\n", NIL);

    Language::classHandler(n);

    SwigType *t = Copy(Getattr(n, "name"));
    SwigType_add_pointer(t);

    // Replace storing a pointer to underlying class with a smart pointer (intended for use with non-intrusive smart pointers)
    SwigType *smart = Swig_cparse_smartptr(n);
    String *wrap_class = NewStringf("&_wrap_class_%s", class_name);
    if (smart) {
      SwigType_add_pointer(smart);
      SwigType_remember_clientdata(smart, wrap_class);
    }
    //String *wrap_class = NewStringf("&_wrap_class_%s", class_name);
    SwigType_remember_clientdata(t, wrap_class);

    int use_director = Swig_directorclass(n);
    if (use_director) {
      String *nspace = Getattr(n, "sym:nspace");
      String *cname = Swig_name_disown(nspace, class_name);
      String *wcname = Swig_name_wrapper(cname);
      String *cnameshdw = NewStringf("%s_shadow", cname);
      String *wcnameshdw = Swig_name_wrapper(cnameshdw);
      Octave_begin_function(n, f_wrappers, cnameshdw, wcnameshdw, true);
      Printf(f_wrappers, "  if (args.length()!=1) {\n");
      Printf(f_wrappers, "    error(\"disown takes no arguments\");\n");
      Printf(f_wrappers, "    return octave_value_list();\n");
      Printf(f_wrappers, "  }\n");
      Printf(f_wrappers, "  %s (args, nargout);\n", wcname);
      Printf(f_wrappers, "  return args;\n");
      Printf(f_wrappers, "}\n");
      Printf(s_members_tab, "{\"__disown\",%s,0,0,0,0},\n", wcnameshdw);
      Delete(wcname);
      Delete(cname);
      Delete(wcnameshdw);
      Delete(cnameshdw);
    }

    Printf(s_members_tab, "{0,0,0,0,0,0}\n};\n");
    Printv(f_wrappers, s_members_tab, NIL);

    String *base_class_names = NewString("");
    String *base_class = NewString("");
    List *baselist = Getattr(n, "bases");
    if (baselist && Len(baselist)) {
      Iterator b;
      int index = 0;
      b = First(baselist);
      while (b.item) {
        String *bname = Getattr(b.item, "name");
        if ((!bname) || GetFlag(b.item, "feature:ignore") || (!Getattr(b.item, "module"))) {
          b = Next(b);
          continue;
        }

        String *bname_mangled = SwigType_manglestr(SwigType_add_pointer(Copy(bname)));
        Printf(base_class_names, "\"%s\",", bname_mangled);
        Printf(base_class, "0,");
        b = Next(b);
        index++;
        Delete(bname_mangled);
      }
    }

    Printv(f_wrappers, "static const char *swig_", class_name, "_base_names[] = {", base_class_names, "0};\n", NIL);
    Printv(f_wrappers, "static const swig_type_info *swig_", class_name, "_base[] = {", base_class, "0};\n", NIL);
    Printv(f_wrappers, "static swig_octave_class _wrap_class_", class_name, " = {\"", class_name, "\", &SWIGTYPE", SwigType_manglestr(t), ",", NIL);
    Printv(f_wrappers, Swig_directorclass(n) ? "1," : "0,", NIL);
    if (have_constructor) {
      String *nspace = Getattr(n, "sym:nspace");
      String *cname = Swig_name_construct(nspace, constructor_name);
      String *wcname = Swig_name_wrapper(cname);
      String *tname = texinfo_name(n);
      Printf(f_wrappers, "%s,%s,", wcname, tname);
      Delete(tname);
      Delete(wcname);
      Delete(cname);
    } else
      Printv(f_wrappers, "0,0,", NIL);
    if (have_destructor) {
      String *nspace = Getattr(n, "sym:nspace");
      String *cname = Swig_name_destroy(nspace, class_name);
      String *wcname = Swig_name_wrapper(cname);
      Printf(f_wrappers, "%s,", wcname);
      Delete(wcname);
      Delete(cname);
    } else
      Printv(f_wrappers, "0", ",", NIL);
    Printf(f_wrappers, "swig_%s_members,swig_%s_base_names,swig_%s_base };\n\n", class_name, class_name, class_name);

    Delete(base_class);
    Delete(base_class_names);
    Delete(smart);
    Delete(t);
    Delete(s_members_tab);
    s_members_tab = 0;
    class_name = 0;

    return SWIG_OK;
  }

  virtual int memberfunctionHandler(Node *n) {
    Language::memberfunctionHandler(n);

    assert(s_members_tab);
    assert(class_name);
    String *name = Getattr(n, "name");
    String *iname = GetChar(n, "sym:name");
    String *realname = iname ? iname : name;
    String *wname = Getattr(n, "wrap:name");
    assert(wname);

    if (!Getattr(n, "sym:nextSibling")) {
      String *tname = texinfo_name(n);
      String *rname = Copy(wname);
      bool overloaded = !!Getattr(n, "sym:overloaded");
      if (overloaded)
        Delslice(rname, Len(rname) - Len(Getattr(n, "sym:overname")), DOH_END);
      Printf(s_members_tab, "{\"%s\",%s,0,0,0,%s},\n",
             realname, rname, tname);
      Delete(rname);
      Delete(tname);
    }

    return SWIG_OK;
  }

  virtual int membervariableHandler(Node *n) {
    Setattr(n, "feature:autodoc", "0");

    Language::membervariableHandler(n);

    assert(s_members_tab);
    assert(class_name);
    String *symname = Getattr(n, "sym:name");
    String *getname = Swig_name_get(NSPACE_TODO, Swig_name_member(NSPACE_TODO, class_name, symname));
    String *setname = Swig_name_set(NSPACE_TODO, Swig_name_member(NSPACE_TODO, class_name, symname));
    String *getwname = Swig_name_wrapper(getname);
    String *setwname = GetFlag(n, "feature:immutable") ? NewString("octave_set_immutable") : Swig_name_wrapper(setname);
    assert(s_members_tab);

    Printf(s_members_tab, "{\"%s\",0,%s,%s,0,0},\n", symname, getwname, setwname);

    Delete(getname);
    Delete(setname);
    Delete(getwname);
    Delete(setwname);
    return SWIG_OK;
  }

  virtual int constructorHandler(Node *n) {
    have_constructor = 1;
    if (!constructor_name)
      constructor_name = NewString(Getattr(n, "sym:name"));

    int use_director = Swig_directorclass(n);
    if (use_director) {
      Parm *parms = Getattr(n, "parms");
      Parm *self;
      String *name = NewString("self");
      String *type = NewString("void");
      SwigType_add_pointer(type);
      self = NewParm(type, name, n);
      Delete(type);
      Delete(name);
      Setattr(self, "lname", "self_obj");
      if (parms)
        set_nextSibling(self, parms);
      Setattr(n, "parms", self);
      Setattr(n, "wrap:self", "1");
      Setattr(n, "hidden", "1");
      Delete(self);
    }

    return Language::constructorHandler(n);
  }

  virtual int destructorHandler(Node *n) {
    have_destructor = 1;
    return Language::destructorHandler(n);
  }

  virtual int staticmemberfunctionHandler(Node *n) {
    Language::staticmemberfunctionHandler(n);

    assert(s_members_tab);
    assert(class_name);
    String *name = Getattr(n, "name");
    String *iname = GetChar(n, "sym:name");
    String *realname = iname ? iname : name;
    String *wname = Getattr(n, "wrap:name");
    assert(wname);

    if (!Getattr(n, "sym:nextSibling")) {
      String *tname = texinfo_name(n);
      String *rname = Copy(wname);
      bool overloaded = !!Getattr(n, "sym:overloaded");
      if (overloaded)
        Delslice(rname, Len(rname) - Len(Getattr(n, "sym:overname")), DOH_END);
      Printf(s_members_tab, "{\"%s\",%s,0,0,1,%s},\n",
             realname, rname, tname);
      Delete(rname);
      Delete(tname);
    }

    return SWIG_OK;
  }

  virtual int memberconstantHandler(Node *n) {
    return Language::memberconstantHandler(n);
  }

  virtual int staticmembervariableHandler(Node *n) {
    Setattr(n, "feature:autodoc", "0");

    Language::staticmembervariableHandler(n);

    if (!GetFlag(n, "wrappedasconstant")) {
      assert(s_members_tab);
      assert(class_name);
      String *symname = Getattr(n, "sym:name");
      String *getname = Swig_name_get(NSPACE_TODO, Swig_name_member(NSPACE_TODO, class_name, symname));
      String *setname = Swig_name_set(NSPACE_TODO, Swig_name_member(NSPACE_TODO, class_name, symname));
      String *getwname = Swig_name_wrapper(getname);
      String *setwname = GetFlag(n, "feature:immutable") ? NewString("octave_set_immutable") : Swig_name_wrapper(setname);
      assert(s_members_tab);

      Printf(s_members_tab, "{\"%s\",0,%s,%s,1,0},\n", symname, getwname, setwname);

      Delete(getname);
      Delete(setname);
      Delete(getwname);
      Delete(setwname);
    }
    return SWIG_OK;
  }

  int classDirectorInit(Node *n) {
    String *declaration = Swig_director_declaration(n);
    Printf(f_directors_h, "\n");
    Printf(f_directors_h, "%s\n", declaration);
    Printf(f_directors_h, "public:\n");
    Delete(declaration);
    return Language::classDirectorInit(n);
  }

  int classDirectorEnd(Node *n) {
    Printf(f_directors_h, "};\n\n");
    return Language::classDirectorEnd(n);
  }

  int classDirectorConstructor(Node *n) {
    Node *parent = Getattr(n, "parentNode");
    String *sub = NewString("");
    String *decl = Getattr(n, "decl");
    String *supername = Swig_class_name(parent);
    String *classname = NewString("");
    Printf(classname, "SwigDirector_%s", supername);

    // insert self parameter
    Parm *p;
    ParmList *superparms = Getattr(n, "parms");
    ParmList *parms = CopyParmList(superparms);
    String *type = NewString("void");
    SwigType_add_pointer(type);
    p = NewParm(type, NewString("self"), n);
    set_nextSibling(p, parms);
    parms = p;

    if (!Getattr(n, "defaultargs")) {
      // constructor
      {
        Wrapper *w = NewWrapper();
        String *call;
        String *basetype = Getattr(parent, "classtype");
        String *target = Swig_method_decl(0, decl, classname, parms, 0, 0);
        call = Swig_csuperclass_call(0, basetype, superparms);
        Printf(w->def, "%s::%s: %s," "\nSwig::Director(static_cast<%s*>(this)) { \n", classname, target, call, basetype);
        Append(w->def, "}\n");
        Delete(target);
        Wrapper_print(w, f_directors);
        Delete(call);
        DelWrapper(w);
      }

      // constructor header
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

  int classDirectorDefaultConstructor(Node *n) {
    String *classname = Swig_class_name(n);
    {
      Wrapper *w = NewWrapper();
      Printf(w->def, "SwigDirector_%s::SwigDirector_%s(void* self) :"
             "\nSwig::Director((octave_swig_type*)self,static_cast<%s*>(this)) { \n", classname, classname, classname);
      Append(w->def, "}\n");
      Wrapper_print(w, f_directors);
      DelWrapper(w);
    }
    Printf(f_directors_h, "    SwigDirector_%s(octave_swig_type* self);\n", classname);
    Delete(classname);
    return Language::classDirectorDefaultConstructor(n);
  }

  int classDirectorMethod(Node *n, Node *parent, String *super) {
    int is_void = 0;
    int is_pointer = 0;
    String *decl = Getattr(n, "decl");
    String *returntype = Getattr(n, "type");
    String *name = Getattr(n, "name");
    String *classname = Getattr(parent, "sym:name");
    String *c_classname = Getattr(parent, "name");
    String *symname = Getattr(n, "sym:name");
    String *declaration = NewString("");
    ParmList *l = Getattr(n, "parms");
    Wrapper *w = NewWrapper();
    String *tm;
    String *wrap_args = NewString("");
    String *value = Getattr(n, "value");
    String *storage = Getattr(n, "storage");
    bool pure_virtual = false;
    int status = SWIG_OK;
    int idx;
    bool ignored_method = GetFlag(n, "feature:ignore") ? true : false;

    if (Cmp(storage, "virtual") == 0) {
      if (Cmp(value, "0") == 0) {
        pure_virtual = true;
      }
    }

    // determine if the method returns a pointer
    is_pointer = SwigType_ispointer_return(decl);
    is_void = (!Cmp(returntype, "void") && !is_pointer);

    // virtual method definition
    String *target;
    String *pclassname = NewStringf("SwigDirector_%s", classname);
    String *qualified_name = NewStringf("%s::%s", pclassname, name);
    SwigType *rtype = Getattr(n, "conversion_operator") ? 0 : Getattr(n, "classDirectorMethods:type");
    target = Swig_method_decl(rtype, decl, qualified_name, l, 0, 0);
    Printf(w->def, "%s", target);
    Delete(qualified_name);
    Delete(target);

    // header declaration
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
          String *str = SwigType_str(Getattr(p, "type"), 0);
          Append(w->def, str);
          Append(declaration, str);
          Delete(str);
        }
      }

      Append(w->def, ")");
      Append(declaration, ")");
    }

    Append(w->def, " {");
    Append(declaration, ";\n");

    // declare method return value
    // if the return value is a reference or const reference, a specialized typemap must
    // handle it, including declaration of c_result ($result).
    if (!is_void) {
      if (!(ignored_method && !pure_virtual)) {
        String *cres = SwigType_lstr(returntype, "c_result");
        Printf(w->code, "%s;\n", cres);
        Delete(cres);
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
      // attach typemaps to arguments (C/C++ -> Octave)
      String *parse_args = NewString("");

      Swig_director_parms_fixup(l);

      Swig_typemap_attach_parms("in", l, 0);
      Swig_typemap_attach_parms("directorin", l, 0);
      Swig_typemap_attach_parms("directorargout", l, w);

      Parm *p;

      int outputs = 0;
      if (!is_void)
        outputs++;

      // build argument list and type conversion string
      p = l;
      while (p) {
        if (checkAttribute(p, "tmap:in:numinputs", "0")) {
          p = Getattr(p, "tmap:in:next");
          continue;
        }

        if (Getattr(p, "tmap:directorargout") != 0)
          outputs++;

        String *pname = Getattr(p, "name");
        String *ptype = Getattr(p, "type");
        Wrapper_add_local(w, "tmpv", "octave_value tmpv");

        if ((tm = Getattr(p, "tmap:directorin")) != 0) {
          String *parse = Getattr(p, "tmap:directorin:parse");
          if (!parse) {
            Setattr(p, "emit:directorinput", "tmpv");
            Replaceall(tm, "$input", "tmpv");
            Replaceall(tm, "$owner", "0");
            Printv(wrap_args, tm, "\n", NIL);
            Printf(wrap_args, "args.append(tmpv);\n");
            Putc('O', parse_args);
          } else {
            Append(parse_args, parse);
            Setattr(p, "emit:directorinput", pname);
            Replaceall(tm, "$input", pname);
            Replaceall(tm, "$owner", "0");
            if (Len(tm) == 0)
              Append(tm, pname);
          }
          p = Getattr(p, "tmap:directorin:next");
          continue;
        } else if (Cmp(ptype, "void")) {
          Swig_warning(WARN_TYPEMAP_DIRECTORIN_UNDEF, input_file, line_number,
                       "Unable to use type %s as a function argument in director method %s::%s (skipping method).\n", SwigType_str(ptype, 0),
                       SwigType_namestr(c_classname), SwigType_namestr(name));
          status = SWIG_NOWRAP;
          break;
        }
        p = nextSibling(p);
      }

      String *method_name = Getattr(n, "sym:name");

      Printv(w->code, wrap_args, NIL);

      // emit method invocation
      Wrapper_add_local(w, "args", "octave_value_list args");
      Wrapper_add_local(w, "out", "octave_value_list out");
      Wrapper_add_local(w, "idx", "std::list<octave_value_list> idx");
      Printf(w->code, "idx.push_back(octave_value_list(\"%s\"));\n", method_name);
      Printf(w->code, "idx.push_back(args);\n");
      Printf(w->code, "out=swig_get_self()->subsref(\".(\",idx,%d);\n", outputs);

      String *cleanup = NewString("");
      String *outarg = NewString("");
      idx = 0;

      // marshal return value
      if (!is_void) {
        Printf(w->code, "if (out.length()<%d) {\n", outputs);
        Printf(w->code, "Swig::DirectorTypeMismatchException::raise(\"Octave "
               "method %s.%s failed to return the required number " "of arguments.\");\n", classname, method_name);
        Printf(w->code, "}\n");

        tm = Swig_typemap_lookup("directorout", n, Swig_cresult_name(), w);
        if (tm != 0) {
          char temp[24];
          sprintf(temp, "out(%d)", idx);
          Replaceall(tm, "$input", temp);
          //    Replaceall(tm, "$argnum", temp);
          Replaceall(tm, "$disown", Getattr(n, "wrap:disown") ? "SWIG_POINTER_DISOWN" : "0");
          if (Getattr(n, "tmap:directorout:implicitconv")) {
            Replaceall(tm, "$implicitconv", get_implicitconv_flag(n));
          }
          Replaceall(tm, "$result", "c_result");
          Printv(w->code, tm, "\n", NIL);
          Delete(tm);
        } else {
          Swig_warning(WARN_TYPEMAP_DIRECTOROUT_UNDEF, input_file, line_number,
                       "Unable to use return type %s in director method %s::%s (skipping method).\n",
                       SwigType_str(returntype, 0), SwigType_namestr(c_classname), SwigType_namestr(name));
          status = SWIG_ERROR;
        }
      }
      idx++;

      // marshal outputs
      for (p = l; p;) {
        if ((tm = Getattr(p, "tmap:directorargout")) != 0) {
          char temp[24];
          sprintf(temp, "out(%d)", idx);
          Replaceall(tm, "$result", temp);
          Replaceall(tm, "$input", Getattr(p, "emit:directorinput"));
          Printv(w->code, tm, "\n", NIL);
          p = Getattr(p, "tmap:directorargout:next");
        } else {
          p = nextSibling(p);
        }
      }

      Delete(parse_args);
      Delete(cleanup);
      Delete(outarg);
    }

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

    Append(w->code, "}\n");

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
    // emit the director method
    if (status == SWIG_OK) {
      if (!Getattr(n, "defaultargs")) {
        Replaceall(w->code, "$symname", symname);
        Wrapper_print(w, f_directors);
        Printv(f_directors_h, declaration, NIL);
        Printv(f_directors_h, inline_extra_method, NIL);
      }
    }
    // clean up
    Delete(wrap_args);
    Delete(pclassname);
    DelWrapper(w);
    return status;
  }

  String *runtimeCode() {
    String *s = NewString("");
    String *srun = Swig_include_sys("octrun.swg");
    if (!srun) {
      Printf(stderr, "*** Unable to open 'octrun.swg'\n");
    } else {
      Append(s, srun);
      Delete(srun);
    }
    return s;
  }

  String *defaultExternalRuntimeFilename() {
    return NewString("swigoctaverun.h");
  }
};

extern "C" Language *swig_octave(void) {
  return new OCTAVE();
}
