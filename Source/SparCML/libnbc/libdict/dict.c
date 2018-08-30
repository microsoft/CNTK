/*
 * dict.c
 *
 * Implementation of generic dictionary routines.
 * Copyright (C) 2001-2004 Farooq Mela.
 *
 * $Id: dict.c,v 1.7 2001/11/25 06:00:49 farooq Exp farooq $
 */

#include <stdlib.h>

#include "dict.h"
#include "dict_private.h"

dict_malloc_func _dict_malloc = malloc;
dict_free_func _dict_free = free;

dict_malloc_func
dict_set_malloc(dict_malloc_func func)
{
	dict_malloc_func old = _dict_malloc;
	_dict_malloc = func ? func : malloc;
	return old;
}

dict_free_func
dict_set_free(dict_free_func func)
{
	dict_free_func old = _dict_free;
	_dict_free = func ? func : free;
	return old;
}

/*
 * In comparing, we cannot simply subtract because that might result in signed
 * overflow.
 */
int
dict_int_cmp(const void *k1, const void *k2)
{
	const int *a = (int*)k1, *b = (int*)k2;

	return (*a < *b) ? -1 : (*a > *b) ? +1 : 0;
}

int
dict_uint_cmp(const void *k1, const void *k2)
{
	const unsigned int *a = (unsigned int*)k1, *b = (unsigned int*)k2;

	return (*a < *b) ? -1 : (*a > *b) ? +1 : 0;
}

int
dict_long_cmp(const void *k1, const void *k2)
{
	const long *a = (long*)k1, *b = (long*)k2;

	return (*a < *b) ? -1 : (*a > *b) ? +1 : 0;
}

int
dict_ulong_cmp(const void *k1, const void *k2)
{
	const unsigned long *a = (unsigned long*)k1, *b = (unsigned long*)k2;

	return (*a < *b) ? -1 : (*a > *b) ? +1 : 0;
}

int
dict_ptr_cmp(const void *k1, const void *k2)
{
	return (k1 > k2) - (k1 < k2);
}

int
dict_str_cmp(const void *k1, const void *k2)
{
	const char *a = (char*)k1, *b = (char*)k2;
	char p, q;

	for (;;) {
		p = *a++; q = *b++;
		if (p == 0 || p != q)
			break;
	}
	return (p > q) - (p < q);
}

void
dict_destroy(dict *dct, int del)
{
	ASSERT(dct != NULL);

	dct->_destroy(dct->_object, del);
	FREE(dct);
}

void
dict_itor_destroy(dict_itor *itor)
{
	ASSERT(itor != NULL);

	itor->_destroy(itor->_itor);
	FREE(itor);
}
