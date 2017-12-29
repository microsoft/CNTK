/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * utils.cxx
 *
 * Various utility functions.
 * ----------------------------------------------------------------------------- */

#include <swigmod.h>

int is_public(Node *n) {
  String *access = Getattr(n, "access");
  return !access || !Cmp(access, "public");
}

int is_private(Node *n) {
  String *access = Getattr(n, "access");
  return access && !Cmp(access, "private");
}

int is_protected(Node *n) {
  String *access = Getattr(n, "access");
  return access && !Cmp(access, "protected");
}

static int is_member_director_helper(Node *parentnode, Node *member) {
  int parent_nodirector = GetFlag(parentnode, "feature:nodirector");
  if (parent_nodirector)
    return 0;
  int parent_director = Swig_director_mode() && GetFlag(parentnode, "feature:director");
  int cdecl_director = parent_director || GetFlag(member, "feature:director");
  int cdecl_nodirector = GetFlag(member, "feature:nodirector");
  return cdecl_director && !cdecl_nodirector && !GetFlag(member, "feature:extend");
}

int is_member_director(Node *parentnode, Node *member) {
  if (parentnode && checkAttribute(member, "storage", "virtual")) {
    return is_member_director_helper(parentnode, member);
  } else {
    return 0;
  }
}

int is_member_director(Node *member) {
  return is_member_director(Getattr(member, "parentNode"), member);
}

// Identifies the additional protected members that are generated when the allprotected option is used.
// This does not include protected virtual methods as they are turned on with the dirprot option.
int is_non_virtual_protected_access(Node *n) {
  int result = 0;
  if (Swig_director_mode() && Swig_director_protected_mode() && Swig_all_protected_mode() && is_protected(n) && !checkAttribute(n, "storage", "virtual")) {
    Node *parentNode = Getattr(n, "parentNode");
    // When vtable is empty, the director class does not get emitted, so a check for an empty vtable should be done.
    // However, vtable is set in Language and so is not yet set when methods in Typepass call clean_overloaded()
    // which calls is_non_virtual_protected_access. So commented out below.
    // Moving the director vtable creation into into Typepass should solve this problem.
    if (is_member_director_helper(parentNode, n) /* && Getattr(parentNode, "vtable")*/)
      result = 1;
  }
  return result;
}

/* Clean overloaded list.  Removes templates, ignored, and errors */

void clean_overloaded(Node *n) {
  Node *nn = Getattr(n, "sym:overloaded");
  Node *first = 0;
  while (nn) {
    String *ntype = nodeType(nn);
    if ((GetFlag(nn, "feature:ignore")) ||
	(Getattr(nn, "error")) ||
	(Strcmp(ntype, "template") == 0) ||
	((Strcmp(ntype, "cdecl") == 0) && is_protected(nn) && !is_member_director(nn) && !is_non_virtual_protected_access(n))) {
      /* Remove from overloaded list */
      Node *ps = Getattr(nn, "sym:previousSibling");
      Node *ns = Getattr(nn, "sym:nextSibling");
      if (ps) {
	Setattr(ps, "sym:nextSibling", ns);
      }
      if (ns) {
	Setattr(ns, "sym:previousSibling", ps);
      }
      Delattr(nn, "sym:previousSibling");
      Delattr(nn, "sym:nextSibling");
      Delattr(nn, "sym:overloaded");
      nn = ns;
      continue;
    } else {
      if (!first)
	first = nn;
      Setattr(nn, "sym:overloaded", first);
    }
    nn = Getattr(nn, "sym:nextSibling");
  }
  if (!first || (first && !Getattr(first, "sym:nextSibling"))) {
    if (Getattr(n, "sym:overloaded"))
      Delattr(n, "sym:overloaded");
  }
}

/* -----------------------------------------------------------------------------
 * Swig_set_max_hash_expand()
 *
 * Controls how many Hash objects are displayed when displaying nested Hash objects.
 * Makes DohSetMaxHashExpand an externally callable function (for debugger).
 * ----------------------------------------------------------------------------- */

void Swig_set_max_hash_expand(int count) {
  SetMaxHashExpand(count);
}

extern "C" {

/* -----------------------------------------------------------------------------
 * Swig_get_max_hash_expand()
 *
 * Returns how many Hash objects are displayed when displaying nested Hash objects.
 * Makes DohGetMaxHashExpand an externally callable function (for debugger).
 * ----------------------------------------------------------------------------- */

int Swig_get_max_hash_expand() {
  return GetMaxHashExpand();
}

/* -----------------------------------------------------------------------------
 * Swig_to_doh_string()
 *
 * DOH version of Swig_to_string()
 * ----------------------------------------------------------------------------- */

static String *Swig_to_doh_string(DOH *object, int count) {
  int old_count = Swig_get_max_hash_expand();
  if (count >= 0)
    Swig_set_max_hash_expand(count);

  String *debug_string = object ? NewStringf("%s", object) : NewString("NULL");

  Swig_set_max_hash_expand(old_count);
  return debug_string;
}

/* -----------------------------------------------------------------------------
 * Swig_to_doh_string_with_location()
 *
 * DOH version of Swig_to_string_with_location()
 * ----------------------------------------------------------------------------- */

static String *Swig_to_doh_string_with_location(DOH *object, int count) {
  int old_count = Swig_get_max_hash_expand();
  if (count >= 0)
    Swig_set_max_hash_expand(count);

  String *debug_string = Swig_stringify_with_location(object);

  Swig_set_max_hash_expand(old_count);
  return debug_string;
}

/* -----------------------------------------------------------------------------
 * Swig_to_string()
 *
 * Swig debug - return C string representation of any DOH type.
 * Nested Hash types expand count is value of Swig_get_max_hash_expand when count<0
 * Note: leaks memory.
 * ----------------------------------------------------------------------------- */

const char *Swig_to_string(DOH *object, int count) {
  return Char(Swig_to_doh_string(object, count));
}

/* -----------------------------------------------------------------------------
 * Swig_to_string_with_location()
 *
 * Swig debug - return C string representation of any DOH type, within [] brackets
 * for Hash and List types, prefixed by line and file information.
 * Nested Hash types expand count is value of Swig_get_max_hash_expand when count<0
 * Note: leaks memory.
 * ----------------------------------------------------------------------------- */

const char *Swig_to_string_with_location(DOH *object, int count) {
  return Char(Swig_to_doh_string_with_location(object, count));
}

/* -----------------------------------------------------------------------------
 * Swig_print()
 *
 * Swig debug - display string representation of any DOH type.
 * Nested Hash types expand count is value of Swig_get_max_hash_expand when count<0
 * ----------------------------------------------------------------------------- */

void Swig_print(DOH *object, int count) {
  String *output = Swig_to_doh_string(object, count);
  Printf(stdout, "%s\n", output);
  Delete(output);
}

/* -----------------------------------------------------------------------------
 * Swig_to_string_with_location()
 *
 * Swig debug - display string representation of any DOH type, within [] brackets
 * for Hash and List types, prefixed by line and file information.
 * Nested Hash types expand count is value of Swig_get_max_hash_expand when count<0
 * ----------------------------------------------------------------------------- */

void Swig_print_with_location(DOH *object, int count) {
  String *output = Swig_to_doh_string_with_location(object, count);
  Printf(stdout, "%s\n", output);
  Delete(output);
}

} // extern "C"

