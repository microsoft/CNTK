/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * void.c
 *
 *     Implements a "void" object that is really just a DOH container around
 *     an arbitrary C object represented as a void *.
 * ----------------------------------------------------------------------------- */

#include "dohint.h"

typedef struct {
  void *ptr;
  void (*del) (void *);
} VoidObj;

/* -----------------------------------------------------------------------------
 * Void_delete()
 *
 * Delete a void object. Invokes the destructor supplied at the time of creation.
 * ----------------------------------------------------------------------------- */

static void Void_delete(DOH *vo) {
  VoidObj *v = (VoidObj *) ObjData(vo);
  if (v->del)
    (*v->del) (v->ptr);
  DohFree(v);
}

/* -----------------------------------------------------------------------------
 * Void_copy()
 *
 * Copies a void object.  This is only a shallow copy. The object destruction
 * function is not copied in order to avoid potential double-free problems.
 * ----------------------------------------------------------------------------- */

static DOH *Void_copy(DOH *vo) {
  VoidObj *v = (VoidObj *) ObjData(vo);
  return NewVoid(v->ptr, 0);
}

/* -----------------------------------------------------------------------------
 * Void_data()
 *
 * Returns the void * stored in the object.
 * ----------------------------------------------------------------------------- */

static void *Void_data(DOH *vo) {
  VoidObj *v = (VoidObj *) ObjData(vo);
  return v->ptr;
}

static DohObjInfo DohVoidType = {
  "VoidObj",			/* objname */
  Void_delete,			/* doh_del */
  Void_copy,			/* doh_copy */
  0,				/* doh_clear */
  0,				/* doh_str */
  Void_data,			/* doh_data */
  0,				/* doh_dump */
  0,				/* doh_len */
  0,				/* doh_hash    */
  0,				/* doh_cmp */
  0,				/* doh_equal    */
  0,				/* doh_first    */
  0,				/* doh_next     */
  0,				/* doh_setfile */
  0,				/* doh_getfile */
  0,				/* doh_setline */
  0,				/* doh_getline */
  0,				/* doh_mapping */
  0,				/* doh_sequence */
  0,				/* doh_file  */
  0,				/* doh_string */
  0,				/* doh_reserved */
  0,				/* clientdata */
};

/* -----------------------------------------------------------------------------
 * NewVoid()
 *
 * Creates a new Void object given a void * and an optional destructor function.
 * ----------------------------------------------------------------------------- */

DOH *DohNewVoid(void *obj, void (*del) (void *)) {
  VoidObj *v;
  v = (VoidObj *) DohMalloc(sizeof(VoidObj));
  v->ptr = obj;
  v->del = del;
  return DohObjMalloc(&DohVoidType, v);
}
