/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * xml.cxx
 *
 * An Xml parse tree generator.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

static const char *usage = "\
XML Options (available with -xml)\n\
     -xmllang <lang> - Typedef language\n\
     -xmllite        - More lightweight version of XML\n\
     ------\n\
     deprecated (use -o): -xml <output.xml> - Use <output.xml> as output file (extension .xml mandatory)\n";

static File *out = 0;
static int xmllite = 0;


class XML:public Language {
public:

  int indent_level;
  long id;

  XML() :indent_level(0) , id(0) {
  }
  
  virtual ~ XML() {
  }

  virtual void main(int argc, char *argv[]) {
    SWIG_typemap_lang("xml");
    for (int iX = 0; iX < argc; iX++) {
      if (strcmp(argv[iX], "-xml") == 0) {
	char *extension = 0;
	if (iX + 1 >= argc)
	  continue;
	extension = argv[iX + 1] + strlen(argv[iX + 1]) - 4;
	if (strcmp(extension, ".xml"))
	  continue;
	iX++;
	Swig_mark_arg(iX);
	String *outfile = NewString(argv[iX]);
	out = NewFile(outfile, "w", SWIG_output_files());
	if (!out) {
	  FileErrorDisplay(outfile);
	  SWIG_exit(EXIT_FAILURE);
	}
	continue;
      }
      if (strcmp(argv[iX], "-xmllang") == 0) {
	Swig_mark_arg(iX);
	iX++;
	SWIG_typemap_lang(argv[iX]);
	Swig_mark_arg(iX);
	continue;
      }
      if (strcmp(argv[iX], "-help") == 0) {
	fputs(usage, stdout);
      }
      if (strcmp(argv[iX], "-xmllite") == 0) {
	Swig_mark_arg(iX);
	xmllite = 1;
      }
    }

    // Add a symbol to the parser for conditional compilation
    Preprocessor_define("SWIGXML 1", 0);
  }

  /* Top of the parse tree */

  virtual int top(Node *n) {
    if (out == 0) {
      String *outfile = Getattr(n, "outfile");
      String *ext = Swig_file_extension(outfile);
      // If there's an extension, ext will include the ".".
      Delslice(outfile, Len(outfile) - Len(ext), DOH_END);
      Delete(ext);
      Append(outfile, ".xml");
      out = NewFile(outfile, "w", SWIG_output_files());
      if (!out) {
	FileErrorDisplay(outfile);
	SWIG_exit(EXIT_FAILURE);
      }
    }
    Printf(out, "<?xml version=\"1.0\" ?> \n");
    Xml_print_tree(n);
    return SWIG_OK;
  }

  void print_indent(int l) {
    int i;
    for (i = 0; i < indent_level; i++) {
      Printf(out, " ");
    }
    if (l) {
      Printf(out, " ");
    }
  }

  void Xml_print_tree(DOH *obj) {
    while (obj) {
      Xml_print_node(obj);
      obj = nextSibling(obj);
    }
  }

  void Xml_print_attributes(Node *obj) {
    String *k;
    indent_level += 4;
    print_indent(0);
    Printf(out, "<attributelist id=\"%ld\" addr=\"%p\" >\n", ++id, obj);
    indent_level += 4;
    Iterator ki;
    ki = First(obj);
    while (ki.key) {
      k = ki.key;
      if ((Cmp(k, "nodeType") == 0)
	  || (Cmp(k, "firstChild") == 0)
	  || (Cmp(k, "lastChild") == 0)
	  || (Cmp(k, "parentNode") == 0)
	  || (Cmp(k, "nextSibling") == 0)
	  || (Cmp(k, "previousSibling") == 0)
	  || (*(Char(k)) == '$')) {
	/* Do nothing */
      } else if (Cmp(k, "module") == 0) {
	Xml_print_module(Getattr(obj, k));
      } else if (Cmp(k, "baselist") == 0) {
	Xml_print_baselist(Getattr(obj, k));
      } else if (!xmllite && Cmp(k, "typescope") == 0) {
	Xml_print_typescope(Getattr(obj, k));
      } else if (!xmllite && Cmp(k, "typetab") == 0) {
	Xml_print_typetab(Getattr(obj, k));
      } else if (Cmp(k, "kwargs") == 0) {
	Xml_print_kwargs(Getattr(obj, k));
      } else if (Cmp(k, "parms") == 0 || Cmp(k, "pattern") == 0) {
	Xml_print_parmlist(Getattr(obj, k));
      } else if (Cmp(k, "catchlist") == 0 || Cmp(k, "templateparms") == 0) {
	Xml_print_parmlist(Getattr(obj, k), Char(k));
      } else {
	DOH *o;
	print_indent(0);
	if (DohIsString(Getattr(obj, k))) {
	  String *ck = NewString(k);
	  o = Str(Getattr(obj, k));
	  Replaceall(ck, ":", "_");
	  Replaceall(ck, "<", "&lt;");
	  /* Do first to avoid aliasing errors. */
	  Replaceall(o, "&", "&amp;");
	  Replaceall(o, "<", "&lt;");
	  Replaceall(o, "\"", "&quot;");
	  Replaceall(o, "\\", "\\\\");
	  Replaceall(o, "\n", "&#10;");
	  Printf(out, "<attribute name=\"%s\" value=\"%s\" id=\"%ld\" addr=\"%p\" />\n", ck, o, ++id, o);
	  Delete(o);
	  Delete(ck);
	} else {
	  o = Getattr(obj, k);
	  String *ck = NewString(k);
	  Replaceall(ck, ":", "_");
	  Printf(out, "<attribute name=\"%s\" value=\"%p\" id=\"%ld\" addr=\"%p\" />\n", ck, o, ++id, o);
	  Delete(ck);
	}
      }
      ki = Next(ki);
    }
    indent_level -= 4;
    print_indent(0);
    Printf(out, "</attributelist >\n");
    indent_level -= 4;
  }

  void Xml_print_node(Node *obj) {
    Node *cobj;

    print_indent(0);
    Printf(out, "<%s id=\"%ld\" addr=\"%p\" >\n", nodeType(obj), ++id, obj);
    Xml_print_attributes(obj);
    cobj = firstChild(obj);
    if (cobj) {
      indent_level += 4;
      Printf(out, "\n");
      Xml_print_tree(cobj);
      indent_level -= 4;
    } else {
      print_indent(1);
      Printf(out, "\n");
    }
    print_indent(0);
    Printf(out, "</%s >\n", nodeType(obj));
  }


  void Xml_print_parmlist(ParmList *p, const char* markup = "parmlist") {

    print_indent(0);
    Printf(out, "<%s id=\"%ld\" addr=\"%p\" >\n", markup, ++id, p);
    indent_level += 4;
    while (p) {
      print_indent(0);
      Printf(out, "<parm id=\"%ld\">\n", ++id);
      Xml_print_attributes(p);
      print_indent(0);
      Printf(out, "</parm >\n");
      p = nextSibling(p);
    }
    indent_level -= 4;
    print_indent(0);
    Printf(out, "</%s >\n", markup);
  }

  void Xml_print_baselist(List *p) {

    print_indent(0);
    Printf(out, "<baselist id=\"%ld\" addr=\"%p\" >\n", ++id, p);
    indent_level += 4;
    Iterator s;
    for (s = First(p); s.item; s = Next(s)) {
      print_indent(0);
      String *item_name = Xml_escape_string(s.item);
      Printf(out, "<base name=\"%s\" id=\"%ld\" addr=\"%p\" />\n", item_name, ++id, s.item);
      Delete(item_name);
    }
    indent_level -= 4;
    print_indent(0);
    Printf(out, "</baselist >\n");
  }

  String *Xml_escape_string(String *str) {
    String *escaped_str = 0;
    if (str) {
      escaped_str = NewString(str);
      Replaceall(escaped_str, "&", "&amp;");
      Replaceall(escaped_str, "<", "&lt;");
      Replaceall(escaped_str, "\"", "&quot;");
      Replaceall(escaped_str, "\\", "\\\\");
      Replaceall(escaped_str, "\n", "&#10;");
    }
    return escaped_str;
  }

  void Xml_print_module(Node *p) {

    print_indent(0);
    Printf(out, "<attribute name=\"module\" value=\"%s\" id=\"%ld\" addr=\"%p\" />\n", Getattr(p, "name"), ++id, p);
  }

  void Xml_print_kwargs(Hash *p) {
    Xml_print_hash(p, "kwargs");
  }

  void Xml_print_typescope(Hash *p) {

    Xml_print_hash(p, "typescope");
  }

  void Xml_print_typetab(Hash *p) {

    Xml_print_hash(p, "typetab");
  }


  void Xml_print_hash(Hash *p, const char *markup) {

    print_indent(0);
    Printf(out, "<%s id=\"%ld\" addr=\"%p\" >\n", markup, ++id, p);
    Xml_print_attributes(p);
    indent_level += 4;
    Iterator n = First(p);
    while (n.key) {
      print_indent(0);
      Printf(out, "<%ssitem id=\"%ld\" addr=\"%p\" >\n", markup, ++id, n.item);
      Xml_print_attributes(n.item);
      print_indent(0);
      Printf(out, "</%ssitem >\n", markup);
      n = Next(n);
    }
    indent_level -= 4;
    print_indent(0);
    Printf(out, "</%s >\n", markup);
  }

};

/* -----------------------------------------------------------------------------
 * Swig_print_xml
 *
 * Dump an XML version of the parse tree.  This is different from using the -xml
 * language module normally as it allows the real language module to process the
 * tree first, possibly stuffing in new attributes, so the XML that is output ends
 * up being a post-processing version of the tree.
 * ----------------------------------------------------------------------------- */

void Swig_print_xml(DOH *obj, String *filename) {
  XML xml;
  xmllite = 1;

  if (!filename) {
    out = stdout;
  } else {
    out = NewFile(filename, "w", SWIG_output_files());
    if (!out) {
      FileErrorDisplay(filename);
      SWIG_exit(EXIT_FAILURE);
    }
  }

  Printf(out, "<?xml version=\"1.0\" ?> \n");
  xml.Xml_print_tree(obj);
}

static Language *new_swig_xml() {
  return new XML();
}
extern "C" Language *swig_xml(void) {
  return new_swig_xml();
}
