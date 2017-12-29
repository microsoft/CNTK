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
  routines to handle the stats files

  the stats file is stored one per cache subdirectory to make this more
  scalable
 */

#include "ccache.h"

extern char *stats_file;
extern char *cache_dir;

#define STATS_VERSION 1

#define FLAG_NOZERO 1 /* don't zero with the -z option */
#define FLAG_ALWAYS 2 /* always show, even if zero */

static struct {
	enum stats stat;
	char *message;
	void (*fn)(unsigned );
	unsigned flags;
} stats_info[] = {
	{ STATS_CACHED,       "cache hit                      ", NULL, FLAG_ALWAYS },
	{ STATS_TOCACHE,      "cache miss                     ", NULL, FLAG_ALWAYS },
	{ STATS_LINK,         "called for link                ", NULL, 0 },
	{ STATS_MULTIPLE,     "multiple source files          ", NULL, 0 },
	{ STATS_STDOUT,       "compiler produced stdout       ", NULL, 0 },
	{ STATS_STATUS,       "compile failed                 ", NULL, 0 },
	{ STATS_ERROR,        "ccache internal error          ", NULL, 0 },
	{ STATS_PREPROCESSOR, "preprocessor error             ", NULL, 0 },
	{ STATS_COMPILER,     "couldn't find the compiler     ", NULL, 0 },
	{ STATS_MISSING,      "cache file missing             ", NULL, 0 },
	{ STATS_ARGS,         "bad compiler arguments         ", NULL, 0 },
	{ STATS_NOTC,         "not a C/C++ file               ", NULL, 0 },
	{ STATS_CONFTEST,     "autoconf compile/link          ", NULL, 0 },
	{ STATS_UNSUPPORTED,  "unsupported compiler option    ", NULL, 0 },
	{ STATS_OUTSTDOUT,    "output to stdout               ", NULL, 0 },
	{ STATS_DEVICE,       "output to a non-regular file   ", NULL, 0 },
	{ STATS_NOINPUT,      "no input file                  ", NULL, 0 },
	{ STATS_ENVIRONMMENT, "error due to bad env variable  ", NULL, 0 },
	{ STATS_NUMFILES,     "files in cache                 ", NULL, FLAG_NOZERO|FLAG_ALWAYS },
	{ STATS_TOTALSIZE,    "cache size                     ", display_size , FLAG_NOZERO|FLAG_ALWAYS },
	{ STATS_MAXFILES,     "max files                      ", NULL, FLAG_NOZERO },
	{ STATS_MAXSIZE,      "max cache size                 ", display_size, FLAG_NOZERO },
	{ STATS_NONE, NULL, NULL, 0 }
};

/* parse a stats file from a buffer - adding to the counters */
static void parse_stats(unsigned counters[STATS_END], char *buf)
{
	int i;
	char *p, *p2;

	p = buf;
	for (i=0;i<STATS_END;i++) {
		counters[i] += strtol(p, &p2, 10);
		if (!p2 || p2 == p) break;
		p = p2;
	}
}

/* write out a stats file */
static void write_stats(int fd, unsigned counters[STATS_END])
{
	int i;
	int len = 0;
	char buf[1024];

	for (i=0;i<STATS_END;i++) {
		len += snprintf(buf+len, sizeof(buf)-(len+1), "%u ", counters[i]);
		if (len >= (int)sizeof(buf)-1) fatal("stats too long?!");
	}
	len += snprintf(buf+len, sizeof(buf)-(len+1), "\n");
	if (len >= (int)sizeof(buf)-1) fatal("stats too long?!");

	lseek(fd, 0, SEEK_SET);
	if (write(fd, buf, len) == -1) fatal("could not write stats");
}


/* fill in some default stats values */
static void stats_default(unsigned counters[STATS_END])
{
	counters[STATS_MAXSIZE] += DEFAULT_MAXSIZE / 16;
}

/* read in the stats from one dir and add to the counters */
static void stats_read_fd(int fd, unsigned counters[STATS_END])
{
	char buf[1024];
	int len;
	len = read(fd, buf, sizeof(buf)-1);
	if (len <= 0) {
		stats_default(counters);
		return;
	}
	buf[len] = 0;
	parse_stats(counters, buf);
}

/* update the stats counter for this compile */
static void stats_update_size(enum stats stat, size_t size, size_t numfiles)
{
	int fd;
	unsigned counters[STATS_END];
	int need_cleanup = 0;

	if (getenv("CCACHE_NOSTATS")) return;

	if (!stats_file) {
		if (!cache_dir) return;
		x_asprintf(&stats_file, "%s/stats", cache_dir);
	}

	/* open safely to try to prevent symlink races */
	fd = safe_open(stats_file);

	/* still can't get it? don't bother ... */
	if (fd == -1) return;

	memset(counters, 0, sizeof(counters));

	if (lock_fd(fd) != 0) {
		close(fd);
		return;
	}

	/* read in the old stats */
	stats_read_fd(fd, counters);

	/* update them */
	counters[stat]++;

	/* on a cache miss we up the file count and size */
	if (stat == STATS_TOCACHE) {
		counters[STATS_NUMFILES] += numfiles;
		counters[STATS_TOTALSIZE] += size;
	}

	/* and write them out */
	write_stats(fd, counters);
	close(fd);

	/* we might need to cleanup if the cache has now got too big */
	if (counters[STATS_MAXFILES] != 0 &&
	    counters[STATS_NUMFILES] > counters[STATS_MAXFILES]) {
		need_cleanup = 1;
	}
	if (counters[STATS_MAXSIZE] != 0 &&
	    counters[STATS_TOTALSIZE] > counters[STATS_MAXSIZE]) {
		need_cleanup = 1;
	}

	if (need_cleanup) {
		char *p = dirname(stats_file);
		cleanup_dir(p, counters[STATS_MAXFILES], counters[STATS_MAXSIZE],
			    numfiles);
		free(p);
	}
}

/* record a cache miss */
void stats_tocache(size_t size, size_t numfiles)
{
	/* convert size to kilobytes */
	size = size / 1024;

	stats_update_size(STATS_TOCACHE, size, numfiles);
}

/* update a normal stat */
void stats_update(enum stats stat)
{
	stats_update_size(stat, 0, 0);
}

/* read in the stats from one dir and add to the counters */
void stats_read(const char *stats_file, unsigned counters[STATS_END])
{
	int fd;

	fd = open(stats_file, O_RDONLY|O_BINARY);
	if (fd == -1) {
		stats_default(counters);
		return;
	}
	lock_fd(fd);
	stats_read_fd(fd, counters);
	close(fd);
}

/* sum and display the total stats for all cache dirs */
void stats_summary(void)
{
	int dir, i;
	unsigned counters[STATS_END];

	memset(counters, 0, sizeof(counters));

	/* add up the stats in each directory */
	for (dir=-1;dir<=0xF;dir++) {
		char *fname;

		if (dir == -1) {
			x_asprintf(&fname, "%s/stats", cache_dir);
		} else {
			x_asprintf(&fname, "%s/%1x/stats", cache_dir, dir);
		}

		stats_read(fname, counters);
		free(fname);

		/* oh what a nasty hack ... */
		if (dir == -1) {
			counters[STATS_MAXSIZE] = 0;
		}

	}

	printf("cache directory                     %s\n", cache_dir);

	/* and display them */
	for (i=0;stats_info[i].message;i++) {
		enum stats stat = stats_info[i].stat;

		if (counters[stat] == 0 && 
		    !(stats_info[i].flags & FLAG_ALWAYS)) {
			continue;
		}

		printf("%s ", stats_info[i].message);
		if (stats_info[i].fn) {
			stats_info[i].fn(counters[stat]);
			printf("\n");
		} else {
			printf("%8u\n", counters[stat]);
		}
	}
}

/* zero all the stats structures */
void stats_zero(void)
{
	int dir, fd;
	unsigned i;
	char *fname;
	unsigned counters[STATS_END];

	x_asprintf(&fname, "%s/stats", cache_dir);
	unlink(fname);
	free(fname);

	for (dir=0;dir<=0xF;dir++) {
		x_asprintf(&fname, "%s/%1x/stats", cache_dir, dir);
		fd = safe_open(fname);
		if (fd == -1) {
			free(fname);
			continue;
		}
		memset(counters, 0, sizeof(counters));
		lock_fd(fd);
		stats_read_fd(fd, counters);
		for (i=0;stats_info[i].message;i++) {
			if (!(stats_info[i].flags & FLAG_NOZERO)) {
				counters[stats_info[i].stat] = 0;
			}
		}
		write_stats(fd, counters);
		close(fd);
		free(fname);
	}
}


/* set the per directory limits */
int stats_set_limits(long maxfiles, long maxsize)
{
	int dir;
	unsigned counters[STATS_END];

	if (maxfiles != -1) {
		maxfiles /= 16;
	}
	if (maxsize != -1) {
		maxsize /= 16;
	}

	if (create_dir(cache_dir) != 0) {
		return 1;
	}

	/* set the limits in each directory */
	for (dir=0;dir<=0xF;dir++) {
		char *fname, *cdir;
		int fd;

		x_asprintf(&cdir, "%s/%1x", cache_dir, dir);
		if (create_dir(cdir) != 0) {
			return 1;
		}
		x_asprintf(&fname, "%s/stats", cdir);
		free(cdir);

		memset(counters, 0, sizeof(counters));
		fd = safe_open(fname);
		if (fd != -1) {
			lock_fd(fd);
			stats_read_fd(fd, counters);
			if (maxfiles != -1) {
				counters[STATS_MAXFILES] = maxfiles;
			}
			if (maxsize != -1) {
				counters[STATS_MAXSIZE] = maxsize;
			}
			write_stats(fd, counters);
			close(fd);
		}
		free(fname);
	}

	return 0;
}

/* set the per directory sizes */
void stats_set_sizes(const char *dir, size_t num_files, size_t total_size)
{
	int fd;
	unsigned counters[STATS_END];
	char *stats_file;

	create_dir(dir);
	x_asprintf(&stats_file, "%s/stats", dir);

	memset(counters, 0, sizeof(counters));

	fd = safe_open(stats_file);
	if (fd != -1) {
		lock_fd(fd);
		stats_read_fd(fd, counters);
		counters[STATS_NUMFILES] = num_files;
		counters[STATS_TOTALSIZE] = total_size;
		write_stats(fd, counters);
		close(fd);
	}

	free(stats_file);
}
