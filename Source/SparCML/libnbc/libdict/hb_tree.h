/*
 * hb_tree.h
 *
 * Interface for height balanced tree.
 * Copyright (C) 2001 Farooq Mela.
 *
 * $Id: hb_tree.h,v 1.2 2001/09/10 06:46:41 farooq Exp $
 */

#ifndef _HB_TREE_H_
#define _HB_TREE_H_

#include "dict.h"

BEGIN_DECL

struct hb_tree;
typedef struct hb_tree hb_tree;

hb_tree *hb_tree_new __P((dict_cmp_func key_cmp, dict_del_func key_del,
						  dict_del_func dat_del));
dict	*hb_dict_new __P((dict_cmp_func key_cmp, dict_del_func key_del,
						  dict_del_func dat_del));
void	 hb_tree_destroy __P((hb_tree *tree, int del));

int hb_tree_insert __P((hb_tree *tree, void *key, void *dat, int overwrite));
int hb_tree_probe __P((hb_tree *tree, void *key, void **dat));
void *hb_tree_search __P((hb_tree *tree, const void *key));
const void *hb_tree_csearch __P((const hb_tree *tree, const void *key));
int hb_tree_remove __P((hb_tree *tree, const void *key, int del));
void hb_tree_empty __P((hb_tree *tree, int del));
void hb_tree_walk __P((hb_tree *tree, dict_vis_func visit));
unsigned hb_tree_count __P((const hb_tree *tree));
unsigned hb_tree_height __P((const hb_tree *tree));
unsigned hb_tree_mheight __P((const hb_tree *tree));
unsigned hb_tree_pathlen __P((const hb_tree *tree));
const void *hb_tree_min __P((const hb_tree *tree));
const void *hb_tree_max __P((const hb_tree *tree));

struct hb_itor;
typedef struct hb_itor hb_itor;

hb_itor *hb_itor_new __P((hb_tree *tree));
dict_itor *hb_dict_itor_new __P((hb_tree *tree));
void hb_itor_destroy __P((hb_itor *tree));

int hb_itor_valid __P((const hb_itor *itor));
void hb_itor_invalidate __P((hb_itor *itor));
int hb_itor_next __P((hb_itor *itor));
int hb_itor_prev __P((hb_itor *itor));
int hb_itor_nextn __P((hb_itor *itor, unsigned count));
int hb_itor_prevn __P((hb_itor *itor, unsigned count));
int hb_itor_first __P((hb_itor *itor));
int hb_itor_last __P((hb_itor *itor));
int hb_itor_search __P((hb_itor *itor, const void *key));
const void *hb_itor_key __P((const hb_itor *itor));
void *hb_itor_data __P((hb_itor *itor));
const void *hb_itor_cdata __P((const hb_itor *itor));
int hb_itor_set_data __P((hb_itor *itor, void *dat, int del));
int hb_itor_remove __P((hb_itor *itor, int del));

END_DECL

#endif /* !_HB_TREE_H_ */
