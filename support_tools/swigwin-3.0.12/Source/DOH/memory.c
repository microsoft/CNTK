/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * memory.c
 *
 *     This file implements all of DOH's memory management including allocation
 *     of objects and checking of objects.
 * ----------------------------------------------------------------------------- */

#include "dohint.h"

#ifndef DOH_POOL_SIZE
#define DOH_POOL_SIZE         16384
#endif

/* Checks stale DOH object use - will use a lot more memory as pool memory is not re-used. */
/*
#define DOH_DEBUG_MEMORY_POOLS
*/

static int PoolSize = DOH_POOL_SIZE;

DOH *DohNone = 0;		/* The DOH None object */

typedef struct pool {
  DohBase *ptr;			/* Start of pool */
  int len;			/* Length of pool */
  int blen;			/* Byte length of pool */
  int current;			/* Current position for next allocation */
  char *pbeg;			/* Beg of pool */
  char *pend;			/* End of pool */
  struct pool *next;		/* Next pool */
} Pool;

static DohBase *FreeList = 0;	/* List of free objects */
static Pool *Pools = 0;
static int pools_initialized = 0;

/* ----------------------------------------------------------------------
 * CreatePool() - Create a new memory pool 
 * ---------------------------------------------------------------------- */

static void CreatePool() {
  Pool *p = 0;
  p = (Pool *) DohMalloc(sizeof(Pool));
  assert(p);
  p->ptr = (DohBase *) DohMalloc(sizeof(DohBase) * PoolSize);
  assert(p->ptr);
  memset(p->ptr, 0, sizeof(DohBase) * PoolSize);
  p->len = PoolSize;
  p->blen = PoolSize * sizeof(DohBase);
  p->current = 0;
  p->pbeg = ((char *) p->ptr);
  p->pend = p->pbeg + p->blen;
  p->next = Pools;
  Pools = p;
}

/* ----------------------------------------------------------------------
 * InitPools() - Initialize the memory allocator
 * ---------------------------------------------------------------------- */

static void InitPools() {
  if (pools_initialized)
    return;
  CreatePool();			/* Create initial pool */
  pools_initialized = 1;
  DohNone = NewVoid(0, 0);	/* Create the None object */
  DohIntern(DohNone);
}

/* ----------------------------------------------------------------------
 * DohCheck()
 *
 * Returns 1 if an arbitrary pointer is a DOH object.
 * ---------------------------------------------------------------------- */

int DohCheck(const DOH *ptr) {
  register Pool *p = Pools;
  register char *cptr = (char *) ptr;
  while (p) {
    if ((cptr >= p->pbeg) && (cptr < p->pend)) {
#ifdef DOH_DEBUG_MEMORY_POOLS
      DohBase *b = (DohBase *) ptr;
      int DOH_object_already_deleted = b->type == 0;
      assert(!DOH_object_already_deleted);
#endif
      return 1;
    }
    /*
       pptr = (char *) p->ptr;
       if ((cptr >= pptr) && (cptr < (pptr+(p->current*sizeof(DohBase))))) return 1; */
    p = p->next;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 * DohIntern()
 * ----------------------------------------------------------------------------- */

void DohIntern(DOH *obj) {
  DohBase *b = (DohBase *) obj;
  b->flag_intern = 1;
}

/* ----------------------------------------------------------------------
 * DohObjMalloc()
 *
 * Allocate memory for a new object.
 * ---------------------------------------------------------------------- */

DOH *DohObjMalloc(DohObjInfo *type, void *data) {
  DohBase *obj;
  if (!pools_initialized)
    InitPools();
#ifndef DOH_DEBUG_MEMORY_POOLS
  if (FreeList) {
    obj = FreeList;
    FreeList = (DohBase *) obj->data;
  } else {
#endif
    while (Pools->current == Pools->len) {
      CreatePool();
    }
    obj = Pools->ptr + Pools->current;
    ++Pools->current;
#ifndef DOH_DEBUG_MEMORY_POOLS
  }
#endif
  obj->type = type;
  obj->data = data;
  obj->meta = 0;
  obj->refcount = 1;
  obj->flag_intern = 0;
  obj->flag_marked = 0;
  obj->flag_user = 0;
  obj->flag_usermark = 0;
  return (DOH *) obj;
}

/* ----------------------------------------------------------------------
 * DohObjFree() - Free a DOH object
 * ---------------------------------------------------------------------- */

void DohObjFree(DOH *ptr) {
  DohBase *b, *meta;
  b = (DohBase *) ptr;
  if (b->flag_intern)
    return;
  meta = (DohBase *) b->meta;
  b->data = (void *) FreeList;
  b->meta = 0;
  b->type = 0;
  b->refcount = 0;
  FreeList = b;
  if (meta) {
    Delete(meta);
  }
}

/* ----------------------------------------------------------------------
 * DohMemoryDebug()
 *
 * Display memory usage statistics
 * ---------------------------------------------------------------------- */

void DohMemoryDebug(void) {
  extern DohObjInfo DohStringType;
  extern DohObjInfo DohListType;
  extern DohObjInfo DohHashType;

  Pool *p;
  int totsize = 0;
  int totused = 0;
  int totfree = 0;

  int numstring = 0;
  int numlist = 0;
  int numhash = 0;

  printf("Memory statistics:\n\n");
  printf("Pools:\n");

  p = Pools;
  while (p) {
    /* Calculate number of used, free items */
    int i;
    int nused = 0, nfree = 0;
    for (i = 0; i < p->len; i++) {
      if (p->ptr[i].refcount <= 0)
	nfree++;
      else {
	nused++;
	if (p->ptr[i].type == &DohStringType)
	  numstring++;
	else if (p->ptr[i].type == &DohListType)
	  numlist++;
	else if (p->ptr[i].type == &DohHashType)
	  numhash++;
      }
    }
    printf("    Pool %8p: size = %10d. used = %10d. free = %10d\n", (void *) p, p->len, nused, nfree);
    totsize += p->len;
    totused += nused;
    totfree += nfree;
    p = p->next;
  }
  printf("\n    Total:          size = %10d, used = %10d, free = %10d\n", totsize, totused, totfree);

  printf("\nObject types\n");
  printf("    Strings   : %d\n", numstring);
  printf("    Lists     : %d\n", numlist);
  printf("    Hashes    : %d\n", numhash);

#if 0
  p = Pools;
  while (p) {
    int i;
    for (i = 0; i < p->len; i++) {
      if (p->ptr[i].refcount > 0) {
	if (p->ptr[i].type == &DohStringType) {
	  Printf(stdout, "%s\n", p->ptr + i);
	}
      }
    }
    p = p->next;
  }
#endif

}
