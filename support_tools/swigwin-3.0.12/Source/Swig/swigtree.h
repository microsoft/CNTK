/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * swigtree.h
 *
 * These functions are used to access and manipulate the SWIG parse tree.
 * The structure of this tree is modeled directly after XML-DOM.  The attribute 
 * and function names are meant to be similar.
 * ----------------------------------------------------------------------------- */

/* Macros to traverse the DOM tree */

#define  nodeType(x)               Getattr(x,"nodeType")
#define  parentNode(x)             Getattr(x,"parentNode")
#define  previousSibling(x)        Getattr(x,"previousSibling")
#define  nextSibling(x)            Getattr(x,"nextSibling")
#define  firstChild(x)             Getattr(x,"firstChild")
#define  lastChild(x)              Getattr(x,"lastChild")

/* Macros to set up the DOM tree (mostly used by the parser) */

#define  set_nodeType(x,v)         Setattr(x,"nodeType",v)
#define  set_parentNode(x,v)       Setattr(x,"parentNode",v)
#define  set_previousSibling(x,v)  Setattr(x,"previousSibling",v)
#define  set_nextSibling(x,v)      Setattr(x,"nextSibling",v)
#define  set_firstChild(x,v)       Setattr(x,"firstChild",v)
#define  set_lastChild(x,v)        Setattr(x,"lastChild",v)

/* Utility functions */

extern int    checkAttribute(Node *obj, const_String_or_char_ptr name, const_String_or_char_ptr value);
extern void   appendChild(Node *node, Node *child);
extern void   prependChild(Node *node, Node *child);
extern void   removeNode(Node *node);
extern Node  *copyNode(Node *node);
extern void   appendSibling(Node *node, Node *child);

/* Node restoration/restore functions */

extern void  Swig_require(const char *ns, Node *node, ...);
extern void  Swig_save(const char *ns, Node *node, ...);
extern void  Swig_restore(Node *node);

/* Debugging of parse trees */

extern void Swig_print_tags(File *obj, Node *root);
extern void Swig_print_tree(Node *obj);
extern void Swig_print_node(Node *obj);
