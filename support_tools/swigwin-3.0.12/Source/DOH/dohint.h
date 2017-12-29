/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * dohint.h
 *
 *     This file describes internally managed objects.
 * ----------------------------------------------------------------------------- */

#ifndef _DOHINT_H
#define _DOHINT_H

#include "doh.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <stdarg.h>

/* Hash objects */
typedef struct {
  DOH *(*doh_getattr) (DOH *obj, DOH *name);	/* Get attribute */
  int (*doh_setattr) (DOH *obj, DOH *name, DOH *value);	/* Set attribute */
  int (*doh_delattr) (DOH *obj, DOH *name);	/* Del attribute */
  DOH *(*doh_keys) (DOH *obj);	/* All keys as a list */
} DohHashMethods;

/* List objects */
typedef struct {
  DOH *(*doh_getitem) (DOH *obj, int index);	/* Get item      */
  int (*doh_setitem) (DOH *obj, int index, DOH *value);	/* Set item      */
  int (*doh_delitem) (DOH *obj, int index);	/* Delete item   */
  int (*doh_insitem) (DOH *obj, int index, DOH *value);	/* Insert item   */
  int (*doh_delslice) (DOH *obj, int sindex, int eindex);	/* Delete slice  */
} DohListMethods;

/* File methods */
typedef struct {
  int (*doh_read) (DOH *obj, void *buffer, int nbytes);	/* Read bytes */
  int (*doh_write) (DOH *obj, const void *buffer, int nbytes);	/* Write bytes */
  int (*doh_putc) (DOH *obj, int ch);	/* Put character */
  int (*doh_getc) (DOH *obj);	/* Get character */
  int (*doh_ungetc) (DOH *obj, int ch);	/* Unget character */
  int (*doh_seek) (DOH *obj, long offset, int whence);	/* Seek */
  long (*doh_tell) (DOH *obj);	/* Tell */
  int (*doh_close) (DOH *obj);	/* Close */
} DohFileMethods;

/* String methods */
typedef struct {
  int (*doh_replace) (DOH *obj, const DOHString_or_char *old, const DOHString_or_char *rep, int flags);
  void (*doh_chop) (DOH *obj);
} DohStringMethods;

/* -----------------------------------------------------------------------------
 * DohObjInfo
 * ----------------------------------------------------------------------------- */

typedef struct DohObjInfo {
  const char *objname;		/* Object name        */

  /* Basic object methods */
  void (*doh_del) (DOH *obj);	/* Delete object      */
  DOH *(*doh_copy) (DOH *obj);	/* Copy and object    */
  void (*doh_clear) (DOH *obj);	/* Clear an object    */

  /* I/O methods */
  DOH *(*doh_str) (DOH *obj);	/* Make a full string */
  void *(*doh_data) (DOH *obj);	/* Return raw data    */
  int (*doh_dump) (DOH *obj, DOH *out);	/* Serialize on out   */

  /* Length and hash values */
  int (*doh_len) (DOH *obj);
  int (*doh_hashval) (DOH *obj);

  /* Compare */
  int (*doh_cmp) (DOH *obj1, DOH *obj2);

  /* Equal */
  int (*doh_equal) (DOH *obj1, DOH *obj2);

  /* Iterators */
  DohIterator (*doh_first) (DOH *obj);
  DohIterator (*doh_next) (DohIterator);

  /* Positional */
  void (*doh_setfile) (DOH *obj, DOHString_or_char *file);
  DOH *(*doh_getfile) (DOH *obj);
  void (*doh_setline) (DOH *obj, int line);
  int (*doh_getline) (DOH *obj);

  DohHashMethods *doh_hash;	/* Hash methods       */
  DohListMethods *doh_list;	/* List methods       */
  DohFileMethods *doh_file;	/* File methods       */
  DohStringMethods *doh_string;	/* String methods     */
  void *doh_reserved;		/* Reserved           */
  void *clientdata;		/* User data          */
} DohObjInfo;

typedef struct {
  void *data;			/* Data pointer */
  DohObjInfo *type;
  void *meta;			/* Meta data */
  unsigned int flag_intern:1;	/* Interned object */
  unsigned int flag_marked:1;	/* Mark flag. Used to avoid recursive loops in places */
  unsigned int flag_user:1;	/* User flag */
  unsigned int flag_usermark:1;	/* User marked */
  unsigned int refcount:28;	/* Reference count (max 16 million) */
} DohBase;

/* Macros for decrefing and increfing (safe for null objects). */

#define Decref(a)         if (a) ((DohBase *) a)->refcount--
#define Incref(a)         if (a) ((DohBase *) a)->refcount++
#define Refcount(a)       ((DohBase *) a)->refcount

/* Macros for manipulating objects in a safe manner */
#define ObjData(a)        ((DohBase *)a)->data
#define ObjSetMark(a,x)   ((DohBase *)a)->flag_marked = x
#define ObjGetMark(a)     ((DohBase *)a)->flag_marked
#define ObjType(a)        ((DohBase *)a)->type

extern DOH *DohObjMalloc(DohObjInfo *type, void *data);	/* Allocate a DOH object */
extern void DohObjFree(DOH *ptr);	/* Free a DOH object     */

#endif				/* DOHINT_H */
