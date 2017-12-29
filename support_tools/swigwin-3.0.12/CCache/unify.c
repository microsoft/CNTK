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
  C/C++ unifier

  the idea is that changes that don't affect the resulting C code
  should not change the hash. This is achieved by folding white-space
  and other non-semantic fluff in the input into a single unified format.

  This unifier was design to match the output of the unifier in
  compilercache, which is flex based. The major difference is that
  this unifier is much faster (about 2x) and more forgiving of
  syntactic errors. Continuing on syntactic errors is important to
  cope with C/C++ extensions in the local compiler (for example,
  inline assembly systems).  
*/

#include "ccache.h"

static char *s_tokens[] = {
	"...",	">>=",	"<<=",	"+=",	"-=",	"*=",	"/=",	"%=",	"&=",	"^=",
	"|=",	">>",	"<<",	"++",	"--",	"->",	"&&",	"||",	"<=",	">=",
	"==",	"!=",	";",	"{",	"<%",	"}",	"%>",	",",	":",	"=",
	"(",	")",	"[",	"<:",	"]",	":>",	".",	"&",	"!",	"~",
	"-",	"+",	"*",	"/",	"%",	"<",	">",	"^",	"|",	"?",
	0
};

#define C_ALPHA 1
#define C_SPACE 2
#define C_TOKEN 4
#define C_QUOTE 8
#define C_DIGIT 16
#define C_HEX   32
#define C_FLOAT 64
#define C_SIGN  128

static struct {
	unsigned char type;
	unsigned char num_toks;
	char *toks[7];
} tokens[256];

/* build up the table used by the unifier */
static void build_table(void)
{
	unsigned char c;
	int i;
	static int done;

	if (done) return;
	done = 1;

	memset(tokens, 0, sizeof(tokens));
	for (c=0;c<128;c++) {
		if (isalpha(c) || c == '_') tokens[c].type |= C_ALPHA;
		if (isdigit(c)) tokens[c].type |= C_DIGIT;
		if (isspace(c)) tokens[c].type |= C_SPACE;
		if (isxdigit(c)) tokens[c].type |= C_HEX;
	}
	tokens['\''].type |= C_QUOTE;
	tokens['"'].type |= C_QUOTE;
	tokens['l'].type |= C_FLOAT;
	tokens['L'].type |= C_FLOAT;
	tokens['f'].type |= C_FLOAT;
	tokens['F'].type |= C_FLOAT;
	tokens['U'].type |= C_FLOAT;
	tokens['u'].type |= C_FLOAT;

	tokens['-'].type |= C_SIGN;
	tokens['+'].type |= C_SIGN;

	for (i=0;s_tokens[i];i++) {
		c = s_tokens[i][0];
		tokens[c].type |= C_TOKEN;
		tokens[c].toks[tokens[c].num_toks] = s_tokens[i];
		tokens[c].num_toks++;
	}
}

/* buffer up characters before hashing them */
static void pushchar(unsigned char c)
{
	static unsigned char buf[64];
	static int len;

	if (c == 0) {
		if (len > 0) {
			hash_buffer((char *)buf, len);
			len = 0;
		}
		hash_buffer(NULL, 0);
		return;
	}

	buf[len++] = c;
	if (len == 64) {
		hash_buffer((char *)buf, len);
		len = 0;
	}
}

/* hash some C/C++ code after unifying */
static void unify(unsigned char *p, size_t size)
{
	size_t ofs;
	unsigned char q;
	int i;

	build_table();

	for (ofs=0; ofs<size;) {
		if (p[ofs] == '#') {
			if ((size-ofs) > 2 && p[ofs+1] == ' ' && isdigit(p[ofs+2])) {
				do {
					ofs++;
				} while (ofs < size && p[ofs] != '\n');
				ofs++;
			} else {
				do {
					pushchar(p[ofs]);
					ofs++;
				} while (ofs < size && p[ofs] != '\n');
				pushchar('\n');
				ofs++;
			}
			continue;
		}

		if (tokens[p[ofs]].type & C_ALPHA) {
			do {
				pushchar(p[ofs]);
				ofs++;
			} while (ofs < size && 
				 (tokens[p[ofs]].type & (C_ALPHA|C_DIGIT)));
			pushchar('\n');
			continue;
		}

		if (tokens[p[ofs]].type & C_DIGIT) {
			do {
				pushchar(p[ofs]);
				ofs++;
			} while (ofs < size && 
				 ((tokens[p[ofs]].type & C_DIGIT) || p[ofs] == '.'));
			if (ofs < size && (p[ofs] == 'x' || p[ofs] == 'X')) {
				do {
					pushchar(p[ofs]);
					ofs++;
				} while (ofs < size && (tokens[p[ofs]].type & C_HEX));
			}
			if (ofs < size && (p[ofs] == 'E' || p[ofs] == 'e')) {
				pushchar(p[ofs]);
				ofs++;
				while (ofs < size && 
				       (tokens[p[ofs]].type & (C_DIGIT|C_SIGN))) {
					pushchar(p[ofs]);
					ofs++;
				}
			}
			while (ofs < size && (tokens[p[ofs]].type & C_FLOAT)) {
				pushchar(p[ofs]);
				ofs++;
			}
			pushchar('\n');
			continue;
		}

		if (tokens[p[ofs]].type & C_SPACE) {
			do {
				ofs++;
			} while (ofs < size && (tokens[p[ofs]].type & C_SPACE));
			continue;
		}
			
		if (tokens[p[ofs]].type & C_QUOTE) {
			q = p[ofs];
			pushchar(p[ofs]);
			do {
				ofs++;
				while (ofs < size-1 && p[ofs] == '\\') {
					pushchar(p[ofs]);
					pushchar(p[ofs+1]);
					ofs+=2;
				}
				pushchar(p[ofs]);
			} while (ofs < size && p[ofs] != q);
			pushchar('\n');
			ofs++;
			continue;
		}

		if (tokens[p[ofs]].type & C_TOKEN) {
			q = p[ofs];
			for (i=0;i<tokens[q].num_toks;i++) {
				unsigned char *s = (unsigned char *)tokens[q].toks[i];
				int len = strlen((char *)s);
				if (size >= ofs+len && memcmp(&p[ofs], s, len) == 0) {
					int j;
					for (j=0;s[j];j++) {
						pushchar(s[j]);
						ofs++;
					}
					pushchar('\n');
					break;
				}
			}
			if (i < tokens[q].num_toks) {
				continue;
			}
		}

		pushchar(p[ofs]);
		pushchar('\n');
		ofs++;
	}
	pushchar(0);
}


/* hash a file that consists of preprocessor output, but remove any line 
   number information from the hash
*/
int unify_hash(const char *fname)
{
#ifdef _WIN32
	HANDLE file;
	HANDLE section;
	DWORD filesize_low;
	char *map;
	int ret = -1;

	file = CreateFileA(fname, GENERIC_READ, FILE_SHARE_READ, NULL,
	                   OPEN_EXISTING, 0, NULL);
	if (file != INVALID_HANDLE_VALUE) {
		filesize_low = GetFileSize(file, NULL);
		if (!(filesize_low == INVALID_FILE_SIZE && GetLastError() != NO_ERROR)) {
			section = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
			CloseHandle(file);
			if (section != NULL) {
				map = MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0);
				CloseHandle(section);
				if (map != NULL)
					ret = 0;
			}
		}
	}

	if (ret == -1) {
		cc_log("Failed to open preprocessor output %s\n", fname);
		stats_update(STATS_PREPROCESSOR);
		return -1;
	}

	/* pass it through the unifier */
	unify((unsigned char *)map, filesize_low);

	UnmapViewOfFile(map);

	return 0;
#else
	int fd;
	struct stat st;	
	char *map;

	fd = open(fname, O_RDONLY|O_BINARY);
	if (fd == -1 || fstat(fd, &st) != 0) {
		cc_log("Failed to open preprocessor output %s\n", fname);
		if (fd != -1) close(fd);
		stats_update(STATS_PREPROCESSOR);
		return -1;
	}

	/* we use mmap() to make it easy to handle arbitrarily long
           lines in preprocessor output. I have seen lines of over
           100k in length, so this is well worth it */
	map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (map == (char *)-1) {
		cc_log("Failed to mmap %s\n", fname);
		stats_update(STATS_PREPROCESSOR);
		return -1;
	}

	/* pass it through the unifier */
	unify((unsigned char *)map, st.st_size);

	munmap(map, st.st_size);

	return 0;
#endif
}

