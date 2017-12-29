/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * deprecate.c
 *
 * The functions in this file are SWIG core functions that are deprecated 
 * or which do not fit in nicely with everything else.  Generally this means
 * that the function and/or API needs to be changed in some future release.
 * ----------------------------------------------------------------------------- */

#include "swig.h"

/* ---------------------------------------------------------------------
 * ParmList_is_compactdefargs()
 *
 * Returns 1 if the parameter list passed in is marked for compact argument
 * handling (by the "compactdefargs" attribute). Otherwise returns 0.
 * ---------------------------------------------------------------------- */

/* Discussion:

  "compactdefargs" is a property set by the Parser to indicate special
  handling of default arguments.   This property seems to be something that
  is associated with functions and methods rather than low-level ParmList
  objects.   Therefore, I don't like the fact that this special purpose
  feature is bolted onto the side of ParmList objects.

  Proposed solution:

     1. "compactdefargs" should be a feature set on function/method nodes
        instead of ParmList objects.  For example, if you have a function,
        you would check the function node to see if the parameters are
        to be handled in this way.


  Difficulties:

     1. This is used by functions in cwrap.c and emit.cxx, none of which
        are passed information about the function/method node.   We might
        have to change the API of those functions to make this work correctly.
        For example:

           int emit_num_required(ParmList *parms)

        might become

           int emit_num_required(ParmList *parms, int compactargs)

*/

int ParmList_is_compactdefargs(ParmList *p) {
  int compactdefargs = 0;

  if (p) {
    compactdefargs = Getattr(p, "compactdefargs") ? 1 : 0;

    /* The "compactdefargs" attribute should only be set on the first parameter in the list.
     * However, sometimes an extra parameter is inserted at the beginning of the parameter list,
     * so we check the 2nd parameter too. */
    if (!compactdefargs) {
      Parm *nextparm = nextSibling(p);
      compactdefargs = (nextparm && Getattr(nextparm, "compactdefargs")) ? 1 : 0;
    }
  }

  return compactdefargs;
}

/* ---------------------------------------------------------------------
 * ParmList_errorstr()
 *
 * Generate a prototype string suitable for use in error/warning messages.
 * Similar to ParmList_protostr() but is also aware of hidden parameters.
 * ---------------------------------------------------------------------- */

/* Discussion.  This function is used to generate error messages, but take 
   into account that there might be a hidden parameter.  Although this involves
   parameter lists, it really isn't a core feature of swigparm.h or parms.c.
   This is because the "hidden" attribute of parameters is added elsewhere (cwrap.c).

   For now, this function is placed here because it doesn't really seem to fit in
   with the parms.c interface.
 
*/

String *ParmList_errorstr(ParmList *p) {
  String *out = NewStringEmpty();
  while (p) {
    if (Getattr(p,"hidden")) {
      p = nextSibling(p);
    } else {
      String *pstr = SwigType_str(Getattr(p, "type"), 0);
      Append(out, pstr);
      p = nextSibling(p);
      if (p) {
	Append(out, ",");
      }
      Delete(pstr);
    }
  }
  return out;
}
