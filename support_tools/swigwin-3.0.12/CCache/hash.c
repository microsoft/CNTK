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
  simple front-end functions to mdfour code
*/

#include "ccache.h"

static struct mdfour md;

void hash_buffer(const char *s, int len)
{
	mdfour_update(&md, (unsigned char *)s, len);
}

void hash_start(void)
{
	mdfour_begin(&md);
}

void hash_string(const char *s)
{
	hash_buffer(s, strlen(s));
}

void hash_int(int x)
{
	hash_buffer((char *)&x, sizeof(x));
}

/* add contents of a file to the hash */
void hash_file(const char *fname)
{
	char buf[1024];
	int fd, n;

	fd = open(fname, O_RDONLY|O_BINARY);
	if (fd == -1) {
		cc_log("Failed to open %s\n", fname);
		fatal("hash_file");
	}

	while ((n = read(fd, buf, sizeof(buf))) > 0) {
		hash_buffer(buf, n);
	}
	close(fd);
}

/* return the hash result as a static string */
char *hash_result(void)
{
	unsigned char sum[16];
	static char ret[53];
	int i;

	hash_buffer(NULL, 0);
	mdfour_result(&md, sum);
	
	for (i=0;i<16;i++) {
		sprintf(&ret[i*2], "%02x", (unsigned)sum[i]);
	}
	sprintf(&ret[i*2], "-%u", (unsigned)md.totalN);

	return ret;
}
