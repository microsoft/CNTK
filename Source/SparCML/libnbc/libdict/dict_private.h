/*
 * dict_private.h
 *
 * Private definitions for libdict.
 * Copyright (C) 2001 Farooq Mela.
 *
 * $Id: dict_private.h,v 1.8 2002/01/02 09:14:11 farooq Exp $
 */

#ifndef _DICT_PRIVATE_H_
#define _DICT_PRIVATE_H_

#include "dict.h"

typedef int			 (*insert_func)		__P((void *, void *k, void *d, int o));
typedef int			 (*probe_func)		__P((void *, void *k, void **d));
typedef void		*(*search_func)		__P((void *, const void *k));
typedef const void	*(*csearch_func)	__P((const void *, const void *k));
typedef int			 (*remove_func)		__P((void *, const void *k, int d));
typedef void		 (*walk_func)		__P((void *, dict_vis_func visit));
typedef unsigned	 (*count_func)		__P((const void *));
typedef void		 (*empty_func)		__P((void *, int del));
typedef void		 (*destroy_func)	__P((void *, int del));
typedef dict_itor	*(*inew_func)		__P((void *));

typedef void		 (*idestroy_func)	__P((void *));
typedef int			 (*valid_func)		__P((const void *));
typedef void		 (*invalidate_func)	__P((void *));
typedef int			 (*next_func)		__P((void *));
typedef int			 (*prev_func)		__P((void *));
typedef int			 (*nextn_func)		__P((void *, unsigned count));
typedef int			 (*prevn_func)		__P((void *, unsigned count));
typedef int			 (*first_func)		__P((void *));
typedef int			 (*last_func)		__P((void *));
typedef int			 (*isearch_func)	__P((void *, const void *k));
typedef const void	*(*key_func)		__P((void *));
typedef void		*(*data_func)		__P((void *));
typedef const void	*(*cdata_func)		__P((const void *));
typedef int			 (*dataset_func)	__P((void *, void *d, int del));
typedef int			 (*iremove_func)	__P((void *, int del));
typedef int			 (*icompare_func)	__P((void *, void *itor2));

#ifndef NDEBUG
# include <stdio.h>
# undef ASSERT
# if defined(__GNUC__)
#  define ASSERT(expr)														\
	if (!(expr))															\
		fprintf(stderr, "\n%s:%d (%s) assertion failed: `%s'\n",			\
				__FILE__, __LINE__, __PRETTY_FUNCTION__, #expr),			\
		abort()
# else
#  define ASSERT(expr)														\
	if (!(expr))															\
		fprintf(stderr, "\n%s:%d assertion failed: `%s'\n",					\
				__FILE__, __LINE__, #expr),									\
		abort()
# endif
#else
# define ASSERT(expr)
#endif

extern dict_malloc_func _dict_malloc;
extern dict_free_func _dict_free;
#define MALLOC(n)	(*_dict_malloc)(n)
#define FREE(p)		(*_dict_free)(p)

#define ABS(a)		((a) < 0 ? -(a) : +(a))
#define MIN(a,b)	((a) < (b) ? (a) : (b))
#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define SWAP(a,b,v)	v = (a), (a) = (b), (b) = v
#define UNUSED(p)	(void)&p

#if defined(__GNUC__)
# define GCC_INLINE		__inline__
# define GCC_UNUSED		__attribute__((__unused__))
# define GCC_CONST		__attribute__((__const__))
#else
# define GCC_INLINE
# define GCC_UNUSED
# define GCC_CONST
#endif

#endif /* !_DICT_PRIVATE_H_ */
