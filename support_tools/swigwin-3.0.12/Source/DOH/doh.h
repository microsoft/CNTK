/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * doh.h
 *
 *     This file describes of the externally visible functions in DOH.
 * ----------------------------------------------------------------------------- */

#ifndef _DOH_H
#define _DOH_H

#ifndef MACSWIG
#include "swigconfig.h"
#endif

#include <stdio.h>
#include <stdarg.h>

/* Set the namespace prefix for DOH API functions. This can be used to control
   visibility of the functions in libraries */

/* Set this macro if you want to change DOH linkage. You would do this if you
   wanted to hide DOH in a library using a different set of names.  Note: simply
   change "Doh" to a new name. */

/*
#define DOH_NAMESPACE(x) Doh ## x
*/

#ifdef DOH_NAMESPACE

/* Namespace control.  These macros define all of the public API names in DOH */

#define DohCheck           DOH_NAMESPACE(Check)
#define DohIntern          DOH_NAMESPACE(Intern)
#define DohDelete          DOH_NAMESPACE(Delete)
#define DohCopy            DOH_NAMESPACE(Copy)
#define DohClear           DOH_NAMESPACE(Clear)
#define DohStr             DOH_NAMESPACE(Str)
#define DohData            DOH_NAMESPACE(Data)
#define DohDump            DOH_NAMESPACE(Dump)
#define DohLen             DOH_NAMESPACE(Len)
#define DohHashval         DOH_NAMESPACE(Hashval)
#define DohCmp             DOH_NAMESPACE(Cmp)
#define DohEqual           DOH_NAMESPACE(Equal)
#define DohIncref          DOH_NAMESPACE(Incref)
#define DohCheckattr       DOH_NAMESPACE(Checkattr)
#define DohSetattr         DOH_NAMESPACE(Setattr)
#define DohDelattr         DOH_NAMESPACE(Delattr)
#define DohKeys            DOH_NAMESPACE(Keys)
#define DohGetInt          DOH_NAMESPACE(GetInt)
#define DohGetDouble       DOH_NAMESPACE(GetDouble)
#define DohGetChar         DOH_NAMESPACE(GetChar)
#define DohSetChar         DOH_NAMESPACE(SetChar)
#define DohSetInt          DOH_NAMESPACE(SetInt)
#define DohSetDouble       DOH_NAMESPACE(SetDouble)
#define DohSetVoid         DOH_NAMESPACE(SetVoid)
#define DohGetVoid         DOH_NAMESPACE(GetVoid)
#define DohGetitem         DOH_NAMESPACE(Getitem)
#define DohSetitem         DOH_NAMESPACE(Setitem)
#define DohDelitem         DOH_NAMESPACE(Delitem)
#define DohInsertitem      DOH_NAMESPACE(Insertitem)
#define DohDelslice        DOH_NAMESPACE(Delslice)
#define DohWrite           DOH_NAMESPACE(Write)
#define DohRead            DOH_NAMESPACE(Read)
#define DohSeek            DOH_NAMESPACE(Seek)
#define DohTell            DOH_NAMESPACE(Tell)
#define DohGetc            DOH_NAMESPACE(Getc)
#define DohPutc            DOH_NAMESPACE(Putc)
#define DohUngetc          DOH_NAMESPACE(Ungetc)
#define DohGetline         DOH_NAMESPACE(Getline)
#define DohSetline         DOH_NAMESPACE(Setline)
#define DohGetfile         DOH_NAMESPACE(Getfile)
#define DohSetfile         DOH_NAMESPACE(Setfile)
#define DohReplace         DOH_NAMESPACE(Replace)
#define DohChop            DOH_NAMESPACE(Chop)
#define DohGetmeta         DOH_NAMESPACE(Getmeta)
#define DohSetmeta         DOH_NAMESPACE(Setmeta)
#define DohDelmeta         DOH_NAMESPACE(Delmeta)
#define DohEncoding        DOH_NAMESPACE(Encoding)
#define DohPrintf          DOH_NAMESPACE(Printf)
#define DohvPrintf         DOH_NAMESPACE(vPrintf)
#define DohPrintv          DOH_NAMESPACE(Printv)
#define DohReadline        DOH_NAMESPACE(Readline)
#define DohIsMapping       DOH_NAMESPACE(IsMapping)
#define DohIsSequence      DOH_NAMESPACE(IsSequence)
#define DohIsString        DOH_NAMESPACE(IsString)
#define DohIsFile          DOH_NAMESPACE(IsFile)
#define DohNewString       DOH_NAMESPACE(NewString)
#define DohNewStringEmpty  DOH_NAMESPACE(NewStringEmpty)
#define DohNewStringWithSize  DOH_NAMESPACE(NewStringWithSize)
#define DohNewStringf      DOH_NAMESPACE(NewStringf)
#define DohStrcmp          DOH_NAMESPACE(Strcmp)
#define DohStrncmp         DOH_NAMESPACE(Strncmp)
#define DohStrstr          DOH_NAMESPACE(Strstr)
#define DohStrchr          DOH_NAMESPACE(Strchr)
#define DohNewFile         DOH_NAMESPACE(NewFile)
#define DohNewFileFromFile DOH_NAMESPACE(NewFileFromFile)
#define DohNewFileFromFd   DOH_NAMESPACE(NewFileFromFd)
#define DohFileErrorDisplay   DOH_NAMESPACE(FileErrorDisplay)
#define DohClose           DOH_NAMESPACE(Close)
#define DohCopyto          DOH_NAMESPACE(Copyto)
#define DohNewList         DOH_NAMESPACE(NewList)
#define DohNewHash         DOH_NAMESPACE(NewHash)
#define DohNewVoid         DOH_NAMESPACE(NewVoid)
#define DohSplit           DOH_NAMESPACE(Split)
#define DohSplitLines      DOH_NAMESPACE(SplitLines)
#define DohNone            DOH_NAMESPACE(None)
#define DohCall            DOH_NAMESPACE(Call)
#define DohObjMalloc       DOH_NAMESPACE(ObjMalloc)
#define DohObjFree         DOH_NAMESPACE(ObjFree)
#define DohMemoryDebug     DOH_NAMESPACE(MemoryDebug)
#define DohStringType      DOH_NAMESPACE(StringType)
#define DohListType        DOH_NAMESPACE(ListType)
#define DohHashType        DOH_NAMESPACE(HashType)
#define DohFileType        DOH_NAMESPACE(FileType)
#define DohVoidType        DOH_NAMESPACE(VoidType)
#define DohIterator        DOH_NAMESPACE(Iterator)
#define DohFirst           DOH_NAMESPACE(First)
#define DohNext            DOH_NAMESPACE(Next)
#endif

#define DOH_MAJOR_VERSION 0
#define DOH_MINOR_VERSION 1

typedef void DOH;

/*
 * With dynamic typing, all DOH objects are technically of type 'void *'.
 * However, to clarify the reading of source code, the following symbolic
 * names are used.
 */

#define DOHString          DOH
#define DOHList            DOH
#define DOHHash            DOH
#define DOHFile            DOH
#define DOHVoid            DOH
#define DOHString_or_char  DOH
#define DOHObj_or_char     DOH

typedef const DOHString_or_char * const_String_or_char_ptr;
typedef const DOHString_or_char * DOHconst_String_or_char_ptr;

#define DOH_BEGIN          -1
#define DOH_END            -2
#define DOH_CUR            -3
#define DOH_CURRENT        -3

/* Iterator objects */

typedef struct {
  void *key;			/* Current key (if any)       */
  void *item;			/* Current item               */
  void *object;			/* Object being iterated over */
  void *_current;		/* Internal use */
  int _index;			/* Internal use */
} DohIterator;

/* Memory management */

#ifndef DohMalloc
#define DohMalloc malloc
#endif
#ifndef DohRealloc
#define DohRealloc realloc
#endif
#ifndef DohFree
#define DohFree free
#endif

extern int DohCheck(const DOH *ptr);	/* Check if a DOH object */
extern void DohIntern(DOH *);	/* Intern an object      */

/* Basic object methods.  Common to most objects */

extern void DohDelete(DOH *obj);	/* Delete an object      */
extern DOH *DohCopy(const DOH *obj);
extern void DohClear(DOH *obj);
extern DOHString *DohStr(const DOH *obj);
extern void *DohData(const DOH *obj);
extern int DohDump(const DOH *obj, DOHFile * out);
extern int DohLen(const DOH *obj);
extern int DohHashval(const DOH *obj);
extern int DohCmp(const DOH *obj1, const DOH *obj2);
extern int DohEqual(const DOH *obj1, const DOH *obj2);
extern void DohIncref(DOH *obj);

/* Mapping methods */

extern DOH *DohGetattr(DOH *obj, const DOHString_or_char *name);
extern int DohSetattr(DOH *obj, const DOHString_or_char *name, const DOHObj_or_char * value);
extern int DohDelattr(DOH *obj, const DOHString_or_char *name);
extern int DohCheckattr(DOH *obj, const DOHString_or_char *name, const DOHString_or_char *value);
extern DOH *DohKeys(DOH *obj);
extern int DohGetInt(DOH *obj, const DOHString_or_char *name);
extern void DohSetInt(DOH *obj, const DOHString_or_char *name, int);
extern double DohGetDouble(DOH *obj, const DOHString_or_char *name);
extern void DohSetDouble(DOH *obj, const DOHString_or_char *name, double);
extern char *DohGetChar(DOH *obj, const DOHString_or_char *name);
extern void DohSetChar(DOH *obj, const DOH *name, char *value);
extern void *DohGetFlagAttr(DOH *obj, const DOHString_or_char *name);
extern int DohGetFlag(DOH *obj, const DOHString_or_char *name);
extern void DohSetFlagAttr(DOH *obj, const DOHString_or_char *name, const DOHString_or_char *attr);
extern void DohSetFlag(DOH *obj, const DOHString_or_char *name);
extern void *DohGetVoid(DOH *obj, const DOHString_or_char *name);
extern void DohSetVoid(DOH *obj, const DOHString_or_char *name, void *value);

/* Sequence methods */

extern DOH *DohGetitem(DOH *obj, int index);
extern int DohSetitem(DOH *obj, int index, const DOHObj_or_char * value);
extern int DohDelitem(DOH *obj, int index);
extern int DohInsertitem(DOH *obj, int index, const DOHObj_or_char * value);
extern int DohDelslice(DOH *obj, int sindex, int eindex);

/* File methods */

extern int DohWrite(DOHFile * obj, const void *buffer, int length);
extern int DohRead(DOHFile * obj, void *buffer, int length);
extern int DohSeek(DOHFile * obj, long offset, int whence);
extern long DohTell(DOHFile * obj);
extern int DohGetc(DOHFile * obj);
extern int DohPutc(int ch, DOHFile * obj);
extern int DohUngetc(int ch, DOHFile * obj);



/* Iterators */
extern DohIterator DohFirst(DOH *obj);
extern DohIterator DohNext(DohIterator x);

/* Positional */

extern int DohGetline(const DOH *obj);
extern void DohSetline(DOH *obj, int line);
extern DOH *DohGetfile(const DOH *obj);
extern void DohSetfile(DOH *obj, DOH *file);

  /* String Methods */

extern int DohReplace(DOHString * src, const DOHString_or_char *token, const DOHString_or_char *rep, int flags);
extern void DohChop(DOHString * src);

/* Meta-variables */
extern DOH *DohGetmeta(DOH *, const DOH *);
extern int DohSetmeta(DOH *, const DOH *, const DOH *value);
extern int DohDelmeta(DOH *, const DOH *);

  /* Utility functions */

extern void DohEncoding(const char *name, DOH *(*fn) (DOH *s));
extern int DohPrintf(DOHFile * obj, const char *format, ...);
extern int DohvPrintf(DOHFile * obj, const char *format, va_list ap);
extern int DohPrintv(DOHFile * obj, ...);
extern DOH *DohReadline(DOHFile * in);

  /* Miscellaneous */

extern int DohIsMapping(const DOH *obj);
extern int DohIsSequence(const DOH *obj);
extern int DohIsString(const DOH *obj);
extern int DohIsFile(const DOH *obj);

extern void DohSetMaxHashExpand(int count);
extern int DohGetMaxHashExpand(void);
extern void DohSetmark(DOH *obj, int x);
extern int DohGetmark(DOH *obj);

/* -----------------------------------------------------------------------------
 * Strings.
 * ----------------------------------------------------------------------------- */

extern DOHString *DohNewStringEmpty(void);
extern DOHString *DohNewString(const DOHString_or_char *c);
extern DOHString *DohNewStringWithSize(const DOHString_or_char *c, int len);
extern DOHString *DohNewStringf(const DOHString_or_char *fmt, ...);

extern int DohStrcmp(const DOHString_or_char *s1, const DOHString_or_char *s2);
extern int DohStrncmp(const DOHString_or_char *s1, const DOHString_or_char *s2, int n);
extern char *DohStrstr(const DOHString_or_char *s1, const DOHString_or_char *s2);
extern char *DohStrchr(const DOHString_or_char *s1, int ch);

/* String replacement flags */

#define   DOH_REPLACE_ANY         0x01
#define   DOH_REPLACE_NOQUOTE     0x02
#define   DOH_REPLACE_ID          0x04
#define   DOH_REPLACE_FIRST       0x08
#define   DOH_REPLACE_ID_BEGIN    0x10
#define   DOH_REPLACE_ID_END      0x20
#define   DOH_REPLACE_NUMBER_END  0x40

#define Replaceall(s,t,r)  DohReplace(s,t,r,DOH_REPLACE_ANY)
#define Replaceid(s,t,r)   DohReplace(s,t,r,DOH_REPLACE_ID)

/* -----------------------------------------------------------------------------
 * Files
 * ----------------------------------------------------------------------------- */

extern DOHFile *DohNewFile(DOH *filename, const char *mode, DOHList *outfiles);
extern DOHFile *DohNewFileFromFile(FILE *f);
extern DOHFile *DohNewFileFromFd(int fd);
extern void DohFileErrorDisplay(DOHString * filename);
/*
 Deprecated, just use DohDelete
extern int DohClose(DOH *file);
*/
extern int DohCopyto(DOHFile * input, DOHFile * output);


/* -----------------------------------------------------------------------------
 * List
 * ----------------------------------------------------------------------------- */

extern DOHList *DohNewList(void);
extern void DohSortList(DOH *lo, int (*cmp) (const DOH *, const DOH *));

/* -----------------------------------------------------------------------------
 * Hash
 * ----------------------------------------------------------------------------- */

extern DOHHash *DohNewHash(void);

/* -----------------------------------------------------------------------------
 * Void
 * ----------------------------------------------------------------------------- */

extern DOHVoid *DohNewVoid(void *ptr, void (*del) (void *));
extern DOHList *DohSplit(DOHFile * input, char ch, int nsplits);
extern DOHList *DohSplitLines(DOHFile * input);
extern DOH *DohNone;

/* Helper union for converting between function and object pointers. */
typedef union DohFuncPtr {
  void* p;
  DOH *(*func)(DOH *);
} DohFuncPtr_t;

extern void DohMemoryDebug(void);

#ifndef DOH_LONG_NAMES
/* Macros to invoke the above functions.  Includes the location of
   the caller to simplify debugging if something goes wrong */

#define Delete             DohDelete
#define Copy               DohCopy
#define Clear              DohClear
#define Str                DohStr
#define Dump               DohDump
#define Getattr            DohGetattr
#define Setattr            DohSetattr
#define Delattr            DohDelattr
#define Checkattr          DohCheckattr
#define Hashval            DohHashval
#define Getitem            DohGetitem
#define Setitem            DohSetitem
#define Delitem            DohDelitem
#define Insert             DohInsertitem
#define Delslice           DohDelslice
#define Append(s,x)        DohInsertitem(s,DOH_END,x)
#define Push(s,x)          DohInsertitem(s,DOH_BEGIN,x)
#define Len                DohLen
#define Data               DohData
#define Char               (char *) Data
#define Cmp                DohCmp
#define Equal              DohEqual
#define Setline            DohSetline
#define Getline            DohGetline
#define Setfile            DohSetfile
#define Getfile            DohGetfile
#define Write              DohWrite
#define Read               DohRead
#define Seek               DohSeek
#define Tell               DohTell
#define Printf             DohPrintf
#define Printv             DohPrintv
#define Getc               DohGetc
#define Putc               DohPutc
#define Ungetc             DohUngetc

/* #define StringPutc         DohStringPutc */
/* #define StringGetc         DohStringGetc */
/* #define StringUngetc       DohStringUngetc */
/* #define StringAppend       Append */
/* #define StringLen          DohStringLen */
/* #define StringChar         DohStringChar */
/* #define StringEqual        DohStringEqual */

#define Close              DohClose
#define vPrintf            DohvPrintf
#define GetInt             DohGetInt
#define GetDouble          DohGetDouble
#define GetChar            DohGetChar
#define GetVoid            DohGetVoid
#define GetFlagAttr        DohGetFlagAttr
#define GetFlag            DohGetFlag
#define SetInt             DohSetInt
#define SetDouble          DohSetDouble
#define SetChar            DohSetattr
#define SetVoid            DohSetVoid
#define SetFlagAttr        DohSetFlagAttr
#define SetFlag            DohSetFlag
#define UnsetFlag(o,n)     DohSetFlagAttr(o,n,NULL)
#define ClearFlag(o,n)     DohSetFlagAttr(o,n,"")
#define Readline           DohReadline
#define Replace            DohReplace
#define Chop               DohChop
#define Getmeta            DohGetmeta
#define Setmeta            DohSetmeta
#define Delmeta            DohDelmeta
#define NewString          DohNewString
#define NewStringEmpty     DohNewStringEmpty
#define NewStringWithSize  DohNewStringWithSize
#define NewStringf         DohNewStringf
#define NewHash            DohNewHash
#define NewList            DohNewList
#define NewFile            DohNewFile
#define NewFileFromFile    DohNewFileFromFile
#define NewFileFromFd      DohNewFileFromFd
#define FileErrorDisplay   DohFileErrorDisplay
#define Close              DohClose
#define NewVoid            DohNewVoid
#define Keys               DohKeys
#define Strcmp             DohStrcmp
#define Strncmp            DohStrncmp
#define Strstr             DohStrstr
#define Strchr             DohStrchr
#define Copyto             DohCopyto
#define Split              DohSplit
#define SplitLines         DohSplitLines
#define Setmark            DohSetmark
#define Getmark            DohGetmark
#define SetMaxHashExpand   DohSetMaxHashExpand
#define GetMaxHashExpand   DohGetMaxHashExpand
#define None               DohNone
#define Call               DohCall
#define First              DohFirst
#define Next               DohNext
#define Iterator           DohIterator
#define SortList           DohSortList
#endif

#ifdef NIL
#undef NIL
#endif

#define NIL  (char *) NULL


#endif				/* DOH_H */
