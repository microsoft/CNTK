/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * hash.c
 *
 *     Implements a simple hash table object.
 * ----------------------------------------------------------------------------- */

#include "dohint.h"

extern DohObjInfo DohHashType;

/* Hash node */
typedef struct HashNode {
  DOH *key;
  DOH *object;
  struct HashNode *next;
} HashNode;

/* Hash object */
typedef struct Hash {
  DOH *file;
  int line;
  HashNode **hashtable;
  int hashsize;
  int nitems;
} Hash;

/* Key interning structure */
typedef struct KeyValue {
  char *cstr;
  DOH *sstr;
  struct KeyValue *left;
  struct KeyValue *right;
} KeyValue;

static KeyValue *root = 0;
static int max_expand = 1;

/* Find or create a key in the interned key table */
static DOH *find_key(DOH *doh_c) {
  char *c = (char *) doh_c;
  KeyValue *r, *s;
  int d = 0;
  /* OK, sure, we use a binary tree for maintaining interned
     symbols.  Then we use their hash values for accessing secondary
     hash tables. */
  r = root;
  s = 0;
  while (r) {
    s = r;
    d = strcmp(r->cstr, c);
    if (d == 0)
      return r->sstr;
    if (d < 0)
      r = r->left;
    else
      r = r->right;
  }
  /*  fprintf(stderr,"Interning '%s'\n", c); */
  r = (KeyValue *) DohMalloc(sizeof(KeyValue));
  r->cstr = (char *) DohMalloc(strlen(c) + 1);
  strcpy(r->cstr, c);
  r->sstr = NewString(c);
  DohIntern(r->sstr);
  r->left = 0;
  r->right = 0;
  if (!s) {
    root = r;
  } else {
    if (d < 0)
      s->left = r;
    else
      s->right = r;
  }
  return r->sstr;
}

#define HASH_INIT_SIZE   7

/* Create a new hash node */
static HashNode *NewNode(DOH *k, void *obj) {
  HashNode *hn = (HashNode *) DohMalloc(sizeof(HashNode));
  hn->key = k;
  Incref(hn->key);
  hn->object = obj;
  Incref(obj);
  hn->next = 0;
  return hn;
}

/* Delete a hash node */
static void DelNode(HashNode *hn) {
  Delete(hn->key);
  Delete(hn->object);
  DohFree(hn);
}

/* -----------------------------------------------------------------------------
 * DelHash()
 *
 * Delete a hash table.
 * ----------------------------------------------------------------------------- */

static void DelHash(DOH *ho) {
  Hash *h = (Hash *) ObjData(ho);
  HashNode *n, *next;
  int i;

  for (i = 0; i < h->hashsize; i++) {
    n = h->hashtable[i];
    while (n) {
      next = n->next;
      DelNode(n);
      n = next;
    }
  }
  DohFree(h->hashtable);
  h->hashtable = 0;
  h->hashsize = 0;
  DohFree(h);
}

/* -----------------------------------------------------------------------------
 * Hash_clear()
 *
 * Clear all of the entries in the hash table.
 * ----------------------------------------------------------------------------- */

static void Hash_clear(DOH *ho) {
  Hash *h = (Hash *) ObjData(ho);
  HashNode *n, *next;
  int i;

  for (i = 0; i < h->hashsize; i++) {
    n = h->hashtable[i];
    while (n) {
      next = n->next;
      DelNode(n);
      n = next;
    }
    h->hashtable[i] = 0;
  }
  h->nitems = 0;
}

/* resize the hash table */
static void resize(Hash *h) {
  HashNode *n, *next, **table;
  int oldsize, newsize;
  int i, p, hv;

  if (h->nitems < 2 * h->hashsize)
    return;

  /* Too big. We have to rescale everything now */
  oldsize = h->hashsize;

  /* Calculate a new size */
  newsize = 2 * oldsize + 1;
  p = 3;
  while (p < (newsize >> 1)) {
    if (((newsize / p) * p) == newsize) {
      newsize += 2;
      p = 3;
      continue;
    }
    p = p + 2;
  }

  table = (HashNode **) DohMalloc(newsize * sizeof(HashNode *));
  for (i = 0; i < newsize; i++) {
    table[i] = 0;
  }

  /* Walk down the old set of nodes and re-place */
  h->hashsize = newsize;
  for (i = 0; i < oldsize; i++) {
    n = h->hashtable[i];
    while (n) {
      hv = Hashval(n->key) % newsize;
      next = n->next;
      n->next = table[hv];
      table[hv] = n;
      n = next;
    }
  }
  DohFree(h->hashtable);
  h->hashtable = table;
}

/* -----------------------------------------------------------------------------
 * Hash_setattr()
 *
 * Set an attribute in the hash table.  Deletes the existing entry if it already
 * exists.
 * ----------------------------------------------------------------------------- */

static int Hash_setattr(DOH *ho, DOH *k, DOH *obj) {
  int hv;
  HashNode *n, *prev;
  Hash *h = (Hash *) ObjData(ho);

  if (!obj) {
    return DohDelattr(ho, k);
  }
  if (!DohCheck(k))
    k = find_key(k);
  if (!DohCheck(obj)) {
    obj = NewString((char *) obj);
    Decref(obj);
  }
  hv = (Hashval(k)) % h->hashsize;
  n = h->hashtable[hv];
  prev = 0;
  while (n) {
    if (Cmp(n->key, k) == 0) {
      /* Node already exists.  Just replace its contents */
      if (n->object == obj) {
	/* Whoa. Same object.  Do nothing */
	return 1;
      }
      Delete(n->object);
      n->object = obj;
      Incref(obj);
      return 1;			/* Return 1 to indicate a replacement */
    } else {
      prev = n;
      n = n->next;
    }
  }
  /* Add this to the table */
  n = NewNode(k, obj);
  if (prev)
    prev->next = n;
  else
    h->hashtable[hv] = n;
  h->nitems++;
  resize(h);
  return 0;
}

/* -----------------------------------------------------------------------------
 * Hash_getattr()
 *
 * Get an attribute from the hash table. Returns 0 if it doesn't exist.
 * ----------------------------------------------------------------------------- */
typedef int (*binop) (DOH *obj1, DOH *obj2);


static DOH *Hash_getattr(DOH *h, DOH *k) {
  DOH *obj = 0;
  Hash *ho = (Hash *) ObjData(h);
  DOH *ko = DohCheck(k) ? k : find_key(k);
  int hv = Hashval(ko) % ho->hashsize;
  DohObjInfo *k_type = ((DohBase*)ko)->type;
  HashNode *n = ho->hashtable[hv];
  if (k_type->doh_equal) {
    binop equal = k_type->doh_equal;
    while (n) {
      DohBase *nk = (DohBase *)n->key;
      if ((k_type == nk->type) && equal(ko, nk)) obj = n->object;
      n = n->next;
    }
  } else {
    binop cmp = k_type->doh_cmp;
    while (n) {
      DohBase *nk = (DohBase *)n->key;
      if ((k_type == nk->type) && (cmp(ko, nk) == 0)) obj = n->object;
      n = n->next;
    }
  }
  return obj;
}

/* -----------------------------------------------------------------------------
 * Hash_delattr()
 *
 * Delete an object from the hash table.
 * ----------------------------------------------------------------------------- */

static int Hash_delattr(DOH *ho, DOH *k) {
  HashNode *n, *prev;
  int hv;
  Hash *h = (Hash *) ObjData(ho);

  if (!DohCheck(k))
    k = find_key(k);
  hv = Hashval(k) % h->hashsize;
  n = h->hashtable[hv];
  prev = 0;
  while (n) {
    if (Cmp(n->key, k) == 0) {
      /* Found it, kill it */

      if (prev) {
	prev->next = n->next;
      } else {
	h->hashtable[hv] = n->next;
      }
      DelNode(n);
      h->nitems--;
      return 1;
    }
    prev = n;
    n = n->next;
  }
  return 0;
}

static DohIterator Hash_firstiter(DOH *ho) {
  DohIterator iter;
  Hash *h = (Hash *) ObjData(ho);
  iter.object = ho;
  iter._current = 0;
  iter.item = 0;
  iter.key = 0;
  iter._index = 0;		/* Index in hash table */
  while ((iter._index < h->hashsize) && !h->hashtable[iter._index])
    iter._index++;

  if (iter._index >= h->hashsize) {
    return iter;
  }
  iter._current = h->hashtable[iter._index];
  iter.item = ((HashNode *) iter._current)->object;
  iter.key = ((HashNode *) iter._current)->key;

  /* Actually save the next slot in the hash.  This makes it possible to
     delete the item being iterated over without trashing the universe */
  iter._current = ((HashNode *) iter._current)->next;
  return iter;
}

static DohIterator Hash_nextiter(DohIterator iter) {
  Hash *h = (Hash *) ObjData(iter.object);
  if (!iter._current) {
    iter._index++;
    while ((iter._index < h->hashsize) && !h->hashtable[iter._index]) {
      iter._index++;
    }
    if (iter._index >= h->hashsize) {
      iter.item = 0;
      iter.key = 0;
      iter._current = 0;
      return iter;
    }
    iter._current = h->hashtable[iter._index];
  }
  iter.key = ((HashNode *) iter._current)->key;
  iter.item = ((HashNode *) iter._current)->object;

  /* Store the next node to iterator on */
  iter._current = ((HashNode *) iter._current)->next;
  return iter;
}

/* -----------------------------------------------------------------------------
 * Hash_keys()
 *
 * Return a list of keys
 * ----------------------------------------------------------------------------- */

static DOH *Hash_keys(DOH *so) {
  DOH *keys;
  Iterator i;

  keys = NewList();
  for (i = First(so); i.key; i = Next(i)) {
    Append(keys, i.key);
  }
  return keys;
}

/* -----------------------------------------------------------------------------
 * DohSetMaxHashExpand()
 *
 * Controls how many Hash objects are displayed in full in Hash_str
 * ----------------------------------------------------------------------------- */

void DohSetMaxHashExpand(int count) {
  max_expand = count;
}

/* -----------------------------------------------------------------------------
 * DohGetMaxHashExpand()
 *
 * Returns how many Hash objects are displayed in full in Hash_str
 * ----------------------------------------------------------------------------- */

int DohGetMaxHashExpand(void) {
  return max_expand;
}

/* -----------------------------------------------------------------------------
 * Hash_str()
 *
 * Create a string representation of a hash table (mainly for debugging).
 * ----------------------------------------------------------------------------- */

static DOH *Hash_str(DOH *ho) {
  int i, j;
  HashNode *n;
  DOH *s;
  static int expanded = 0;
  static const char *tab = "  ";
  Hash *h = (Hash *) ObjData(ho);

  s = NewStringEmpty();
  if (ObjGetMark(ho)) {
    Printf(s, "Hash(%p)", ho);
    return s;
  }
  if (expanded >= max_expand) {
    /* replace each hash attribute with a '.' */
    Printf(s, "Hash(%p) {", ho);
    for (i = 0; i < h->hashsize; i++) {
      n = h->hashtable[i];
      while (n) {
	Putc('.', s);
	n = n->next;
      }
    }
    Putc('}', s);
    return s;
  }
  ObjSetMark(ho, 1);
  Printf(s, "Hash(%p) {\n", ho);
  for (i = 0; i < h->hashsize; i++) {
    n = h->hashtable[i];
    while (n) {
      for (j = 0; j < expanded + 1; j++)
	Printf(s, tab);
      expanded += 1;
      Printf(s, "'%s' : %s, \n", n->key, n->object);
      expanded -= 1;
      n = n->next;
    }
  }
  for (j = 0; j < expanded; j++)
    Printf(s, tab);
  Printf(s, "}");
  ObjSetMark(ho, 0);
  return s;
}

/* -----------------------------------------------------------------------------
 * Hash_len()
 *
 * Return number of entries in the hash table.
 * ----------------------------------------------------------------------------- */

static int Hash_len(DOH *ho) {
  Hash *h = (Hash *) ObjData(ho);
  return h->nitems;
}

/* -----------------------------------------------------------------------------
 * CopyHash()
 *
 * Make a copy of a hash table.  Note: this is a shallow copy.
 * ----------------------------------------------------------------------------- */

static DOH *CopyHash(DOH *ho) {
  Hash *h, *nh;
  HashNode *n;
  DOH *nho;

  int i;
  h = (Hash *) ObjData(ho);
  nh = (Hash *) DohMalloc(sizeof(Hash));
  nh->hashsize = h->hashsize;
  nh->hashtable = (HashNode **) DohMalloc(nh->hashsize * sizeof(HashNode *));
  for (i = 0; i < nh->hashsize; i++) {
    nh->hashtable[i] = 0;
  }
  nh->nitems = 0;
  nh->line = h->line;
  nh->file = h->file;
  if (nh->file)
    Incref(nh->file);

  nho = DohObjMalloc(&DohHashType, nh);
  for (i = 0; i < h->hashsize; i++) {
    n = h->hashtable[i];
    while (n) {
      Hash_setattr(nho, n->key, n->object);
      n = n->next;
    }
  }
  return nho;
}



static void Hash_setfile(DOH *ho, DOH *file) {
  DOH *fo;
  Hash *h = (Hash *) ObjData(ho);

  if (!DohCheck(file)) {
    fo = NewString(file);
    Decref(fo);
  } else
    fo = file;
  Incref(fo);
  Delete(h->file);
  h->file = fo;
}

static DOH *Hash_getfile(DOH *ho) {
  Hash *h = (Hash *) ObjData(ho);
  return h->file;
}

static void Hash_setline(DOH *ho, int line) {
  Hash *h = (Hash *) ObjData(ho);
  h->line = line;
}

static int Hash_getline(DOH *ho) {
  Hash *h = (Hash *) ObjData(ho);
  return h->line;
}

/* -----------------------------------------------------------------------------
 * type information
 * ----------------------------------------------------------------------------- */

static DohHashMethods HashHashMethods = {
  Hash_getattr,
  Hash_setattr,
  Hash_delattr,
  Hash_keys,
};

DohObjInfo DohHashType = {
  "Hash",			/* objname */
  DelHash,			/* doh_del */
  CopyHash,			/* doh_copy */
  Hash_clear,			/* doh_clear */
  Hash_str,			/* doh_str */
  0,				/* doh_data */
  0,				/* doh_dump */
  Hash_len,			/* doh_len */
  0,				/* doh_hash    */
  0,				/* doh_cmp */
  0,				/* doh_equal    */
  Hash_firstiter,		/* doh_first    */
  Hash_nextiter,		/* doh_next     */
  Hash_setfile,			/* doh_setfile */
  Hash_getfile,			/* doh_getfile */
  Hash_setline,			/* doh_setline */
  Hash_getline,			/* doh_getline */
  &HashHashMethods,		/* doh_mapping */
  0,				/* doh_sequence */
  0,				/* doh_file */
  0,				/* doh_string */
  0,				/* doh_positional */
  0,
};

/* -----------------------------------------------------------------------------
 * NewHash()
 *
 * Create a new hash table.
 * ----------------------------------------------------------------------------- */

DOH *DohNewHash(void) {
  Hash *h;
  int i;
  h = (Hash *) DohMalloc(sizeof(Hash));
  h->hashsize = HASH_INIT_SIZE;
  h->hashtable = (HashNode **) DohMalloc(h->hashsize * sizeof(HashNode *));
  for (i = 0; i < h->hashsize; i++) {
    h->hashtable[i] = 0;
  }
  h->nitems = 0;
  h->file = 0;
  h->line = 0;
  return DohObjMalloc(&DohHashType, h);
}
