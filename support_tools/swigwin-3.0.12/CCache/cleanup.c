/*
   Copyright (C) Andrew Tridgell 2002
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/
/*
  functions to cleanup the cache directory when it gets too large 
 */

#include "ccache.h"

static struct files {
	char *fname;
	time_t mtime;
	size_t size;
} **files;
static unsigned allocated;
static unsigned num_files;
static size_t total_size;
static size_t total_files;
static size_t size_threshold;
static size_t files_threshold;

/* file comparison function to try to delete the oldest files first */
static int files_compare(struct files **f1, struct files **f2)
{
	if ((*f2)->mtime == (*f1)->mtime) {
		return strcmp((*f2)->fname, (*f1)->fname);
	}
	if ((*f2)->mtime > (*f1)->mtime) {
		return -1;
	}
	return 1;
}

/* this builds the list of files in the cache */
static void traverse_fn(const char *fname, struct stat *st)
{
	char *p;

	if (!S_ISREG(st->st_mode)) return;

	p = str_basename(fname);
	if (strcmp(p, "stats") == 0) {
		free(p);
		return;
	}
	free(p);

	if (num_files == allocated) {
		allocated = 10000 + num_files*2;
		files = (struct files **)x_realloc(files, 
						   sizeof(struct files *)*allocated);
	}

	files[num_files] = (struct files *)x_malloc(sizeof(struct files));
	files[num_files]->fname = x_strdup(fname);
	files[num_files]->mtime = st->st_mtime;
	files[num_files]->size = file_size(st) / 1024;
	total_size += files[num_files]->size;
	num_files++;
}

/* sort the files we've found and delete the oldest ones until we are
   below the thresholds */
static void sort_and_clean(size_t minfiles)
{
	unsigned i;
	size_t adjusted_minfiles = minfiles;

	if (num_files > 1) {
		/* sort in ascending data order */
		qsort(files, num_files, sizeof(struct files *), 
		      (COMPAR_FN_T)files_compare);
	}
	/* ensure newly cached files (minfiles) are kept - instead of matching
	   the filenames of those newly cached, a faster and simpler approach
	   assumes these are the most recent in the cache and if any other
	   cached files have an identical time stamp, they will also be kept -
	   this approach would not be needed if the cleanup was done at exit. */
	if (minfiles != 0 && minfiles < num_files) {
		unsigned minfiles_index = num_files - minfiles;
		time_t minfiles_time = files[minfiles_index]->mtime;
		for (i=1; i<=minfiles_index; i++) {
			if (files[minfiles_index-i]->mtime == minfiles_time)
				adjusted_minfiles++;
			else
				break;
		}
	}
	
	/* delete enough files to bring us below the threshold */
	for (i=0;i<num_files; i++) {
		if ((size_threshold==0 || total_size < size_threshold) &&
		    (files_threshold==0 || (num_files-i) < files_threshold)) break;

		if (adjusted_minfiles != 0 && num_files-i <= adjusted_minfiles)
			break;

		if (unlink(files[i]->fname) != 0 && errno != ENOENT) {
			fprintf(stderr, "unlink %s - %s\n", 
				files[i]->fname, strerror(errno));
			continue;
		}
		
		total_size -= files[i]->size;
	}	

	total_files = num_files - i;
}

/* cleanup in one cache subdir */
void cleanup_dir(const char *dir, size_t maxfiles, size_t maxsize, size_t minfiles)
{
	unsigned i;

	size_threshold = maxsize * LIMIT_MULTIPLE;
	files_threshold = maxfiles * LIMIT_MULTIPLE;

	num_files = 0;
	total_size = 0;

	/* build a list of files */
	traverse(dir, traverse_fn);

	/* clean the cache */
	sort_and_clean(minfiles);

	stats_set_sizes(dir, total_files, total_size);

	/* free it up */
	for (i=0;i<num_files;i++) {
		free(files[i]->fname);
		free(files[i]);
		files[i] = NULL;
	}
	if (files) free(files);
	allocated = 0;
	files = NULL;

	num_files = 0;
	total_size = 0;
}

/* cleanup in all cache subdirs */
void cleanup_all(const char *dir)
{
	unsigned counters[STATS_END];
	char *dname, *sfile;
	int i;
	
	for (i=0;i<=0xF;i++) {
		x_asprintf(&dname, "%s/%1x", dir, i);
		x_asprintf(&sfile, "%s/%1x/stats", dir, i);

		memset(counters, 0, sizeof(counters));
		stats_read(sfile, counters);

		cleanup_dir(dname, 
			    counters[STATS_MAXFILES], 
			    counters[STATS_MAXSIZE],
			    0);
		free(dname);
		free(sfile);
	}
}


/* traverse function for wiping files */
static void wipe_fn(const char *fname, struct stat *st)
{
	char *p;

	if (!S_ISREG(st->st_mode)) return;

	p = str_basename(fname);
	if (strcmp(p, "stats") == 0) {
		free(p);
		return;
	}
	free(p);

	unlink(fname);
}


/* wipe all cached files in all subdirs */
void wipe_all(const char *dir)
{
	char *dname;
	int i;
	
	for (i=0;i<=0xF;i++) {
		x_asprintf(&dname, "%s/%1x", dir, i);
		traverse(dir, wipe_fn);
		free(dname);
	}

	/* and fix the counters */
	cleanup_all(dir);
}
