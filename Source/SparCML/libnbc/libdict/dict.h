/*
 * dict.h
 *
 * Interface for generic access to dictionary library.
 * Copyright (C) 2001-2004 Farooq Mela.
 *
 * $Id: dict.h,v 1.6 2001/11/14 05:21:10 farooq Exp farooq $
 */

#ifndef _DICT_H_
#define _DICT_H_

#include <stddef.h>

#define DICT_VERSION_MAJOR		0
#define DICT_VERSION_MINOR		2
#define DICT_VERSION_PATCH		1

#ifndef __P
# if defined(__STDC__) || defined(__cplusplus) || defined(c_plusplus) || \
	defined(_MSC_VER)
#  define __P(x)	x
# else /* !__STDC__ && !__cplusplus  && !c_plusplus && !_MSC_VER */
#  define __P(x)
# endif
#endif /* !__P */

#ifndef FALSE
#define FALSE	0
#endif

#ifndef TRUE
#define TRUE	(!FALSE)
#endif

#if defined(__cplusplus) || defined(c_plusplus)
# define BEGIN_DECL	extern "C" {
# define END_DECL	}
#else
# define BEGIN_DECL
# define END_DECL
#endif

BEGIN_DECL

typedef void *(*dict_malloc_func)(size_t);
typedef void  (*dict_free_func)(void *);

dict_malloc_func	dict_set_malloc		__P((dict_malloc_func func));
dict_free_func		dict_set_free		__P((dict_free_func func));

typedef int			(*dict_cmp_func)	__P((const void *, const void *));
typedef void		(*dict_del_func)	__P((void *));
typedef int			(*dict_vis_func)	__P((const void *, void *));
typedef unsigned	(*dict_hsh_func)	__P((const void *));

typedef struct dict			dict;
typedef struct dict_itor	dict_itor;

struct dict {
	void		  *_object;
	int			 (*_insert)		__P((void *obj, void *k, void *d, int ow));
	int			 (*_probe)		__P((void *obj, void *key, void **dat));
	void		*(*_search)		__P((void *obj, const void *k));
	const void	*(*_csearch)	__P((const void *obj, const void *k));
	int			 (*_remove)		__P((void *obj, const void *key, int del));
	void		 (*_walk)		__P((void *obj, dict_vis_func func));
	unsigned	 (*_count)		__P((const void *obj));
	void		 (*_empty)		__P((void *obj, int del));
	void		 (*_destroy)	__P((void *obj, int del));
	dict_itor	*(*_inew)		__P((void *obj));
};

#define dict_private(dct)		(dct)->_object
#define dict_insert(dct,k,d,o)	(dct)->_insert((dct)->_object, (k), (d), (o))
#define dict_probe(dct,k,d)		(dct)->_probe((dct)->_object, (k), (d))
#define dict_search(dct,k)		(dct)->_search((dct)->_object, (k))
#define dict_csearch(dct,k)		(dct)->_csearch((dct)->_object, (k))
#define dict_remove(dct,k,del)	(dct)->_remove((dct)->_object, (k), (del))
#define dict_walk(dct,f)		(dct)->_walk((dct)->_object, (f))
#define dict_count(dct)			(dct)->_count((dct)->_object)
#define dict_empty(dct,d)		(dct)->_empty((dct)->_object, (d))
void dict_destroy __P((dict *dct, int del));
#define dict_itor_new(dct)		(dct)->_inew((dct)->_object)

struct dict_itor {
	void		  *_itor;
	int			 (*_valid)		__P((const void *itor));
	void		 (*_invalid)	__P((void *itor));
	int			 (*_next)		__P((void *itor));
	int			 (*_prev)		__P((void *itor));
	int			 (*_nextn)		__P((void *itor, unsigned count));
	int			 (*_prevn)		__P((void *itor, unsigned count));
	int			 (*_first)		__P((void *itor));
	int			 (*_last)		__P((void *itor));
	int			 (*_search)		__P((void *itor, const void *key));
	const void	*(*_key)		__P((void *itor));
	void		*(*_data)		__P((void *itor));
	const void	*(*_cdata)		__P((const void *itor));
	int			 (*_setdata)	__P((void *itor, void *dat, int del));
	int			 (*_remove)		__P((void *itor, int del));
	int			 (*_compare)	__P((void *itor1, void *itor2));
	void		 (*_destroy)	__P((void *itor));
};

#define dict_itor_private(i)		(i)->_itor
#define dict_itor_valid(i)			(i)->_valid((i)->_itor)
#define dict_itor_invalidate(i)		(i)->_invalid((i)->_itor)
#define dict_itor_next(i)			(i)->_next((i)->_itor)
#define dict_itor_prev(i)			(i)->_prev((i)->_itor)
#define dict_itor_nextn(i,n)		(i)->_nextn((i)->_itor, (n))
#define dict_itor_prevn(i,n)		(i)->_prevn((i)->_itor, (n))
#define dict_itor_first(i)			(i)->_first((i)->_itor)
#define dict_itor_last(i)			(i)->_last((i)->_itor)
#define dict_itor_search(i,k)		(i)->_search((i)->_itor, (k))
#define dict_itor_key(i)			(i)->_key((i)->_itor)
#define dict_itor_data(i)			(i)->_data((i)->_itor)
#define dict_itor_cdata(i)			(i)->_cdata((i)->_itor)
#define dict_itor_set_data(i,dat,d)	(i)->_setdata((i)->_itor, (dat), (d))
#define dict_itor_remove(i)			(i)->_remove((i)->_itor)
void dict_itor_destroy __P((dict_itor *itor));

int		dict_int_cmp __P((const void *k1, const void *k2));
int		dict_uint_cmp __P((const void *k1, const void *k2));
int		dict_long_cmp __P((const void *k1, const void *k2));
int		dict_ulong_cmp __P((const void *k1, const void *k2));
int		dict_ptr_cmp __P((const void *k1, const void *k2));
int		dict_str_cmp __P((const void *k1, const void *k2));

END_DECL

/*#include "hashtable.h"*/
#include "hb_tree.h"
/*#include "pr_tree.h"
#include "rb_tree.h"
#include "sp_tree.h"
#include "tr_tree.h"
#include "wb_tree.h"*/

#endif /* !_DICT_H_ */
