/*
 * hb_tree.c
 *
 * Implementation of height balanced tree.
 * Copyright (C) 2001-2004 Farooq Mela.
 *
 * $Id: hb_tree.c,v 1.10 2001/11/25 08:30:21 farooq Exp farooq $
 *
 * cf. [Gonnet 1984], [Knuth 1998]
 */

#include <stdlib.h>

#include "hb_tree.h"
#include "dict_private.h"

typedef signed char balance_t;

typedef struct hb_node hb_node;

struct hb_node {
	void		*key;
	void		*dat;
	hb_node		*parent;
	hb_node		*llink;
	hb_node		*rlink;
	balance_t	 bal;
};

struct hb_tree {
	hb_node			*root;
	unsigned		 count;
	dict_cmp_func	 key_cmp;
	dict_del_func	 key_del;
	dict_del_func	 dat_del;
};

struct hb_itor {
	hb_tree	*tree;
	hb_node	*node;
};

static int rot_left __P((hb_tree *tree, hb_node *node));
static int rot_right __P((hb_tree *tree, hb_node *node));
static unsigned node_height __P((const hb_node *node));
static unsigned node_mheight __P((const hb_node *node));
static unsigned node_pathlen __P((const hb_node *node, unsigned level));
static hb_node *node_new __P((void *key, void *dat));
static hb_node *node_min __P((hb_node *node));
static hb_node *node_max __P((hb_node *node));
static hb_node *node_next __P((hb_node *node));
static hb_node *node_prev __P((hb_node *node));

hb_tree *
hb_tree_new(dict_cmp_func key_cmp, dict_del_func key_del,
			dict_del_func dat_del)
{
	hb_tree *tree;

	if ((tree = (hb_tree*)MALLOC(sizeof(*tree))) == NULL)
		return NULL;

	tree->root = NULL;
	tree->count = 0;
	tree->key_cmp = key_cmp ? key_cmp : dict_ptr_cmp;
	tree->key_del = key_del;
	tree->dat_del = dat_del;

	return tree;
}

dict *
hb_dict_new(dict_cmp_func key_cmp, dict_del_func key_del,
			dict_del_func dat_del)
{
	dict *dct;
	hb_tree *tree;

	if ((dct = (dict*)MALLOC(sizeof(*dct))) == NULL)
		return NULL;

	if ((tree = hb_tree_new(key_cmp, key_del, dat_del)) == NULL) {
		FREE(dct);
		return NULL;
	}

	dct->_object = tree;
	dct->_inew = (inew_func)hb_dict_itor_new;
	dct->_destroy = (destroy_func)hb_tree_destroy;
	dct->_insert = (insert_func)hb_tree_insert;
	dct->_probe = (probe_func)hb_tree_probe;
	dct->_search = (search_func)hb_tree_search;
	dct->_csearch = (csearch_func)hb_tree_csearch;
	dct->_remove = (remove_func)hb_tree_remove;
	dct->_empty = (empty_func)hb_tree_empty;
	dct->_walk = (walk_func)hb_tree_walk;
	dct->_count = (count_func)hb_tree_count;

	return dct;
}

void
hb_tree_destroy(hb_tree *tree, int del)
{
	ASSERT(tree != NULL);

	if (tree->root)
		hb_tree_empty(tree, del);

	FREE(tree);
}

void
hb_tree_empty(hb_tree *tree, int del)
{
	hb_node *node, *parent;

	ASSERT(tree != NULL);

	node = tree->root;

	while (node) {
		if (node->llink || node->rlink) {
			node = node->llink ? node->llink : node->rlink;
			continue;
		}

		if (del) {
			if (tree->key_del)
				tree->key_del(node->key);
			if (tree->dat_del)
				tree->dat_del(node->dat);
		}

		parent = node->parent;
		FREE(node);

		if (parent) {
			if (parent->llink == node)
				parent->llink = NULL;
			else
				parent->rlink = NULL;
		}
		node = parent;
	}

	tree->root = NULL;
	tree->count = 0;
}

void *
hb_tree_search(hb_tree *tree, const void *key)
{
	int rv;
	hb_node *node;

	ASSERT(tree != NULL);

	node = tree->root;
	while (node) {
		rv = tree->key_cmp(key, node->key);
		if (rv < 0)
			node = node->llink;
		else if (rv > 0)
			node = node->rlink;
		else
			return node->dat;
	}

	return NULL;
}

const void *
hb_tree_csearch(const hb_tree *tree, const void *key)
{
	return hb_tree_csearch((hb_tree *)tree, key);
}

int
hb_tree_insert(hb_tree *tree, void *key, void *dat, int overwrite)
{
	int rv = 0;
	hb_node *node, *parent = NULL, *q = NULL;

	ASSERT(tree != NULL);

	node = tree->root;
	while (node) {
		rv = tree->key_cmp(key, node->key);
		if (rv < 0)
			parent = node, node = node->llink;
		else if (rv > 0)
			parent = node, node = node->rlink;
		else {
			if (overwrite == 0)
				return 1;
			if (tree->key_del)
				tree->key_del(node->key);
			if (tree->dat_del)
				tree->dat_del(node->dat);
			node->key = key;
			node->dat = dat;
			return 0;
		}
		if (parent->bal)
			q = parent;
	}

	if ((node = node_new(key, dat)) == NULL)
		return -1;
	if ((node->parent = parent) == NULL) {
		tree->root = node;
		ASSERT(tree->count == 0);
		tree->count = 1;
		return 0;
	}
	if (rv < 0)
		parent->llink = node;
	else
		parent->rlink = node;

	while (parent != q) {
		parent->bal = (parent->rlink == node) * 2 - 1;
		node = parent;
		parent = node->parent;
	}
	if (q) {
		if (q->llink == node) {
			if (--q->bal == -2) {
				if (q->llink->bal > 0)
					rot_left(tree, q->llink);
				rot_right(tree, q);
			}
		} else {
			if (++q->bal == +2) {
				if (q->rlink->bal < 0)
					rot_right(tree, q->rlink);
				rot_left(tree, q);
			}
		}
	}
	tree->count++;
	return 0;
}

int
hb_tree_probe(hb_tree *tree, void *key, void **dat)
{
	int rv = 0;
	hb_node *node, *parent = NULL, *q = NULL;

	ASSERT(tree != NULL);

	node = tree->root;
	while (node) {
		rv = tree->key_cmp(key, node->key);
		if (rv < 0)
			parent = node, node = node->llink;
		else if (rv > 0)
			parent = node, node = node->rlink;
		else {
			*dat = node->dat;
			return 0;
		}
		if (parent->bal)
			q = parent;
	}

	if ((node = node_new(key, *dat)) == NULL)
		return -1;
	if ((node->parent = parent) == NULL) {
		tree->root = node;
		ASSERT(tree->count == 0);
		tree->count = 1;
		return 1;
	}
	if (rv < 0)
		parent->llink = node;
	else
		parent->rlink = node;

	while (parent != q) {
		parent->bal = (parent->rlink == node) * 2 - 1;
		node = parent;
		parent = parent->parent;
	}
	if (q) {
		if (q->llink == node) {
			if (--q->bal == -2) {
				if (q->llink->bal > 0)
					rot_left(tree, q->llink);
				rot_right(tree, q);
			}
		} else {
			if (++q->bal == +2) {
				if (q->rlink->bal < 0)
					rot_right(tree, q->rlink);
				rot_left(tree, q);
			}
		}
	}
	tree->count++;
	return 1;
}

#define FREE_NODE(n)														\
	if (del) {																\
		if (tree->key_del)													\
			tree->key_del((n)->key);										\
		if (tree->dat_del)													\
			tree->dat_del((n)->dat);										\
	}																		\
	FREE(n)

int
hb_tree_remove(hb_tree *tree, const void *key, int del)
{
	int rv, left;
	hb_node *node, *out, *parent = NULL;
	void *tmp;

	ASSERT(tree != NULL);

	node = tree->root;
	while (node) {
		rv = tree->key_cmp(key, node->key);
		if (rv == 0)
			break;
		parent = node;
		node = rv < 0 ? node->llink : node->rlink;
	}
	if (node == NULL)
		return -1;

	if (node->llink && node->rlink) {
		for (out = node->rlink; out->llink; out = out->llink)
			/* void */;
		SWAP(node->key, out->key, tmp);
		SWAP(node->dat, out->dat, tmp);
		node = out;
		parent = out->parent;
	}

	out = node->llink ? node->llink : node->rlink;
	FREE_NODE(node);
	if (out)
		out->parent = parent;
	if (parent == NULL) {
		tree->root = out;
		tree->count--;
		return 0;
	}

	left = parent->llink == node;
	if (left)
		parent->llink = out;
	else
		parent->rlink = out;

	for (;;) {
		if (left) {
			if (++parent->bal == 0) {
				node = parent;
				goto higher;
			}
			if (parent->bal == +2) {
				ASSERT(parent->rlink != NULL);
				if (parent->rlink->bal < 0) {
					rot_right(tree, parent->rlink);
					rot_left(tree, parent);
				} else {
					ASSERT(parent->rlink->rlink != NULL);
					if (rot_left(tree, parent) == 0)
						break;
				}
			} else {
				break;
			}
		} else {
			if (--parent->bal == 0) {
				node = parent;
				goto higher;
			}
			if (parent->bal == -2) {
				ASSERT(parent->llink != NULL);
				if (parent->llink->bal > 0) {
					rot_left(tree, parent->llink);
					rot_right(tree, parent);
				} else {
					ASSERT(parent->llink->llink != NULL);
					if (rot_right(tree, parent) == 0)
						break;
				}
			} else {
				break;
			}
		}

		/* Only get here on double rotations or single rotations that changed
		 * subtree height - in either event, `parent->parent' is positioned
		 * where `parent' was positioned before any rotations. */
		node = parent->parent;
higher:
		if ((parent = node->parent) == NULL)
			break;
		left = parent->llink == node;
	}
	tree->count--;
	return 0;
}

const void *
hb_tree_min(const hb_tree *tree)
{
	const hb_node *node;

	ASSERT(tree != NULL);

	if (tree->root == NULL)
		return NULL;

	for (node = tree->root; node->llink; node = node->llink)
		/* void */;
	return node->key;
}

const void *
hb_tree_max(const hb_tree *tree)
{
	const hb_node *node;

	ASSERT(tree != NULL);

	if ((node = tree->root) == NULL)
		return NULL;

	for (; node->rlink; node = node->rlink)
		/* void */;
	return node->key;
}

void
hb_tree_walk(hb_tree *tree, dict_vis_func visit)
{
	hb_node *node;

	ASSERT(tree != NULL);

	if (tree->root == NULL)
		return;
	for (node = node_min(tree->root); node; node = node_next(node))
		if (visit(node->key, node->dat) == 0)
			break;
}

unsigned
hb_tree_count(const hb_tree *tree)
{
	ASSERT(tree != NULL);

	return tree->count;
}

unsigned
hb_tree_height(const hb_tree *tree)
{
	ASSERT(tree != NULL);

	return tree->root ? node_height(tree->root) : 0;
}

unsigned
hb_tree_mheight(const hb_tree *tree)
{
	ASSERT(tree != NULL);

	return tree->root ? node_mheight(tree->root) : 0;
}

unsigned
hb_tree_pathlen(const hb_tree *tree)
{
	ASSERT(tree != NULL);

	return tree->root ? node_pathlen(tree->root, 1) : 0;
}

static hb_node *
node_new(void *key, void *dat)
{
	hb_node *node;

	if ((node = (hb_node*)MALLOC(sizeof(*node))) == NULL)
		return NULL;

	node->key = key;
	node->dat = dat;
	node->parent = NULL;
	node->llink = NULL;
	node->rlink = NULL;
	node->bal = 0;

	return node;
}

static hb_node *
node_min(hb_node *node)
{
	ASSERT(node != NULL);

	while (node->llink)
		node = node->llink;
	return node;
}

static hb_node *
node_max(hb_node *node)
{
	ASSERT(node != NULL);

	while (node->rlink)
		node = node->rlink;
	return node;
}

static hb_node *
node_next(hb_node *node)
{
	hb_node *temp;

	ASSERT(node != NULL);

	if (node->rlink) {
		for (node = node->rlink; node->llink; node = node->llink)
			/* void */;
		return node;
	}
	temp = node->parent;
	while (temp && temp->rlink == node) {
		node = temp;
		temp = temp->parent;
	}
	return temp;
}

static hb_node *
node_prev(hb_node *node)
{
	hb_node *temp;

	ASSERT(node != NULL);

	if (node->llink) {
		for (node = node->llink; node->rlink; node = node->rlink)
			/* void */;
		return node;
	}
	temp = node->parent;
	while (temp && temp->llink == node) {
		node = temp;
		temp = temp->parent;
	}
	return temp;
}

static unsigned
node_height(const hb_node *node)
{
	unsigned l, r;

	ASSERT(node != NULL);

	l = node->llink ? node_height(node->llink) + 1 : 0;
	r = node->rlink ? node_height(node->rlink) + 1 : 0;
	return MAX(l, r);
}

static unsigned
node_mheight(const hb_node *node)
{
	unsigned l, r;

	ASSERT(node != NULL);

	l = node->llink ? node_mheight(node->llink) + 1 : 0;
	r = node->rlink ? node_mheight(node->rlink) + 1 : 0;
	return MIN(l, r);
}

static unsigned
node_pathlen(const hb_node *node, unsigned level)
{
	unsigned n = 0;

	ASSERT(node != NULL);

	if (node->llink)
		n += level + node_pathlen(node->llink, level + 1);
	if (node->rlink)
		n += level + node_pathlen(node->rlink, level + 1);
	return n;
}

/*
 * rot_left(T, B):
 *
 *     /             /
 *    B             D
 *   / \           / \
 *  A   D   ==>   B   E
 *     / \       / \
 *    C   E     A   C
 *
 */
static int
rot_left(hb_tree *tree, hb_node *node)
{
	int hc;
	hb_node *rlink, *parent;

	ASSERT(tree != NULL);
	ASSERT(node != NULL);
	ASSERT(node->rlink != NULL);

	rlink = node->rlink;
	node->rlink = rlink->llink;
	if (rlink->llink)
		rlink->llink->parent = node;
	parent = node->parent;
	rlink->parent = parent;
	if (parent) {
		if (parent->llink == node)
			parent->llink = rlink;
		else
			parent->rlink = rlink;
	} else {
		tree->root = rlink;
	}
	rlink->llink = node;
	node->parent = rlink;

	hc = rlink->bal != 0;
	node->bal  -= 1 + MAX(rlink->bal, 0);
	rlink->bal -= 1 - MIN(node->bal, 0);
	return hc;
}

/*
 * rot_right(T, D):
 *
 *       /           /
 *      D           B
 *     / \         / \
 *    B   E  ==>  A   D
 *   / \             / \
 *  A   C           C   E
 *
 */
static int
rot_right(hb_tree *tree, hb_node *node)
{
	int hc;
	hb_node *llink, *parent;

	ASSERT(tree != NULL);
	ASSERT(node != NULL);
	ASSERT(node->llink != NULL);

	llink = node->llink;
	node->llink = llink->rlink;
	if (llink->rlink)
		llink->rlink->parent = node;
	parent = node->parent;
	llink->parent = parent;
	if (parent) {
		if (parent->llink == node)
			parent->llink = llink;
		else
			parent->rlink = llink;
	} else {
		tree->root = llink;
	}
	llink->rlink = node;
	node->parent = llink;

	hc = llink->bal != 0;
	node->bal  += 1 - MIN(llink->bal, 0);
	llink->bal += 1 + MAX(node->bal, 0);
	return hc;
}

hb_itor *
hb_itor_new(hb_tree *tree)
{
	hb_itor *itor;

	ASSERT(tree != NULL);

	if ((itor = (hb_itor*)MALLOC(sizeof(*itor))) == NULL)
		return NULL;

	itor->tree = tree;
	hb_itor_first(itor);
	return itor;
}

dict_itor *
hb_dict_itor_new(hb_tree *tree)
{
	dict_itor *itor;

	ASSERT(tree != NULL);

	if ((itor = (dict_itor*)MALLOC(sizeof(*itor))) == NULL)
		return NULL;

	if ((itor->_itor = hb_itor_new(tree)) == NULL) {
		FREE(itor);
		return NULL;
	}

	itor->_destroy = (idestroy_func)hb_itor_destroy;
	itor->_valid = (valid_func)hb_itor_valid;
	itor->_invalid = (invalidate_func)hb_itor_invalidate;
	itor->_next = (next_func)hb_itor_next;
	itor->_prev = (prev_func)hb_itor_prev;
	itor->_nextn = (nextn_func)hb_itor_nextn;
	itor->_prevn = (prevn_func)hb_itor_prevn;
	itor->_first = (first_func)hb_itor_first;
	itor->_last = (last_func)hb_itor_last;
	itor->_search = (isearch_func)hb_itor_search;
	itor->_key = (key_func)hb_itor_key;
	itor->_data = (data_func)hb_itor_data;
	itor->_cdata = (cdata_func)hb_itor_cdata;
	itor->_setdata = (dataset_func)hb_itor_set_data;

	return itor;
}

void
hb_itor_destroy(hb_itor *itor)
{
	ASSERT(itor != NULL);

	FREE(itor);
}

#define RETVALID(itor)		return itor->node != NULL

int
hb_itor_valid(const hb_itor *itor)
{
	ASSERT(itor != NULL);

	RETVALID(itor);
}

void
hb_itor_invalidate(hb_itor *itor)
{
	ASSERT(itor != NULL);

	itor->node = NULL;
}

int
hb_itor_next(hb_itor *itor)
{
	ASSERT(itor != NULL);

	if (itor->node == NULL)
		hb_itor_first(itor);
	else
		itor->node = node_next(itor->node);
	RETVALID(itor);
}

int
hb_itor_prev(hb_itor *itor)
{
	ASSERT(itor != NULL);

	if (itor->node == NULL)
		hb_itor_last(itor);
	else
		itor->node = node_prev(itor->node);
	RETVALID(itor);
}

int
hb_itor_nextn(hb_itor *itor, unsigned count)
{
	ASSERT(itor != NULL);

	if (count) {
		if (itor->node == NULL) {
			hb_itor_first(itor);
			count--;
		}

		while (count-- && itor->node)
			itor->node = node_next(itor->node);
	}

	RETVALID(itor);
}

int
hb_itor_prevn(hb_itor *itor, unsigned count)
{
	ASSERT(itor != NULL);

	if (count) {
		if (itor->node == NULL) {
			hb_itor_last(itor);
			count--;
		}

		while (count-- && itor->node)
			itor->node = node_prev(itor->node);
	}

	RETVALID(itor);
}

int
hb_itor_first(hb_itor *itor)
{
	hb_tree *t;

	ASSERT(itor != NULL);

	t = itor->tree;
	itor->node = t->root ? node_min(t->root) : NULL;
	RETVALID(itor);
}

int
hb_itor_last(hb_itor *itor)
{
	hb_tree *t;

	ASSERT(itor != NULL);

	t = itor->tree;
	itor->node = t->root ? node_max(t->root) : NULL;
	RETVALID(itor);
}

int
hb_itor_search(hb_itor *itor, const void *key)
{
	int rv;
	hb_node *node;
	dict_cmp_func cmp;

	ASSERT(itor != NULL);

	cmp = itor->tree->key_cmp;
	for (node = itor->tree->root; node;) {
		rv = cmp(key, node->key);
		if (rv == 0)
			break;
		node = rv < 0 ? node->llink : node->rlink;
	}
	itor->node = node;
	RETVALID(itor);
}

const void *
hb_itor_key(const hb_itor *itor)
{
	ASSERT(itor != NULL);

	return itor->node ? itor->node->key : NULL;
}

void *
hb_itor_data(hb_itor *itor)
{
	ASSERT(itor != NULL);

	return itor->node ? itor->node->dat : NULL;
}

const void *
hb_itor_cdata(const hb_itor *itor)
{
	ASSERT(itor != NULL);

	return itor->node ? itor->node->dat : NULL;
}

int
hb_itor_set_data(hb_itor *itor, void *dat, int del)
{
	ASSERT(itor != NULL);

	if (itor->node == NULL)
		return -1;

	if (del && itor->tree->dat_del)
		itor->tree->dat_del(itor->node->dat);
	itor->node->dat = dat;
	return 0;
}
