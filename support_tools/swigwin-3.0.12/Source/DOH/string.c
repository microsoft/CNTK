/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * string.c
 *
 *     Implements a string object that supports both sequence operations and
 *     file semantics.
 * ----------------------------------------------------------------------------- */

#include "dohint.h"

extern DohObjInfo DohStringType;

typedef struct String {
  DOH *file;
  int line;
  int maxsize;			/* Max size allocated */
  int len;			/* Current length     */
  int hashkey;			/* Hash key value     */
  int sp;			/* Current position   */
  char *str;			/* String data        */
} String;

/* -----------------------------------------------------------------------------
 * String_data() - Return as a 'void *'
 * ----------------------------------------------------------------------------- */

static void *String_data(DOH *so) {
  String *s = (String *) ObjData(so);
  s->str[s->len] = 0;
  return (void *) s->str;
}

/* static char *String_char(DOH *so) {
  return (char *) String_data(so);
}
*/

/* -----------------------------------------------------------------------------
 * String_dump() - Serialize a string onto out
 * ----------------------------------------------------------------------------- */

static int String_dump(DOH *so, DOH *out) {
  int nsent;
  int ret;
  String *s = (String *) ObjData(so);
  nsent = 0;
  while (nsent < s->len) {
    ret = Write(out, s->str + nsent, (s->len - nsent));
    if (ret < 0)
      return ret;
    nsent += ret;
  }
  return nsent;
}

/* -----------------------------------------------------------------------------
 * CopyString() - Copy a string
 * ----------------------------------------------------------------------------- */

static DOH *CopyString(DOH *so) {
  String *str;
  String *s = (String *) ObjData(so);
  str = (String *) DohMalloc(sizeof(String));
  str->hashkey = s->hashkey;
  str->sp = s->sp;
  str->line = s->line;
  str->file = s->file;
  if (str->file)
    Incref(str->file);
  str->str = (char *) DohMalloc(s->len + 1);
  memcpy(str->str, s->str, s->len);
  str->maxsize = s->len;
  str->len = s->len;
  str->str[str->len] = 0;

  return DohObjMalloc(&DohStringType, str);
}

/* -----------------------------------------------------------------------------
 * DelString() - Delete a string
 * ----------------------------------------------------------------------------- */

static void DelString(DOH *so) {
  String *s = (String *) ObjData(so);
  DohFree(s->str);
  DohFree(s);
}

/* -----------------------------------------------------------------------------
 * DohString_len() - Length of a string
 * ----------------------------------------------------------------------------- */

static int String_len(DOH *so) {
  String *s = (String *) ObjData(so);
  return s->len;
}


/* -----------------------------------------------------------------------------
 * String_cmp() - Compare two strings
 * ----------------------------------------------------------------------------- */

static int String_cmp(DOH *so1, DOH *so2) {
  String *s1, *s2;
  char *c1, *c2;
  int maxlen, i;
  s1 = (String *) ObjData(so1);
  s2 = (String *) ObjData(so2);
  maxlen = s1->len;
  if (s2->len < maxlen)
    maxlen = s2->len;
  c1 = s1->str;
  c2 = s2->str;
  for (i = maxlen; i; --i, c1++, c2++) {
    if (*c1 != *c2)
      break;
  }
  if (i != 0) {
    if (*c1 < *c2)
      return -1;
    else
      return 1;
  }
  if (s1->len == s2->len)
    return 0;
  if (s1->len > s2->len)
    return 1;
  return -1;
}

/* -----------------------------------------------------------------------------
 * String_equal() - Say if two string are equal
 * ----------------------------------------------------------------------------- */

static int String_equal(DOH *so1, DOH *so2) {
  String *s1 = (String *) ObjData(so1);
  String *s2 = (String *) ObjData(so2);
  register int len = s1->len;
  if (len != s2->len) {
    return 0;
  } else {
    register char *c1 = s1->str;
    register char *c2 = s2->str;
#if 0
    register int mlen = len >> 2;
    register int i = mlen;
    for (; i; --i) {
      if (*(c1++) != *(c2++))
	return 0;
      if (*(c1++) != *(c2++))
	return 0;
      if (*(c1++) != *(c2++))
	return 0;
      if (*(c1++) != *(c2++))
	return 0;
    }
    for (i = len - (mlen << 2); i; --i) {
      if (*(c1++) != *(c2++))
	return 0;
    }
    return 1;
#else
    return memcmp(c1, c2, len) == 0;
#endif
  }
}

/* -----------------------------------------------------------------------------
 * String_hash() - Compute string hash value
 * ----------------------------------------------------------------------------- */

static int String_hash(DOH *so) {
  String *s = (String *) ObjData(so);
  if (s->hashkey >= 0) {
    return s->hashkey;
  } else {
    register char *c = s->str;
    register unsigned int len = s->len > 50 ? 50 : s->len;
    register unsigned int h = 0;
    register unsigned int mlen = len >> 2;
    register unsigned int i = mlen;
    for (; i; --i) {
      h = (h << 5) + *(c++);
      h = (h << 5) + *(c++);
      h = (h << 5) + *(c++);
      h = (h << 5) + *(c++);
    }
    for (i = len - (mlen << 2); i; --i) {
      h = (h << 5) + *(c++);
    }
    h &= 0x7fffffff;
    s->hashkey = (int)h;
    return h;
  }
}

/* -----------------------------------------------------------------------------
 * DohString_append() - Append to s
 * ----------------------------------------------------------------------------- */

static void DohString_append(DOH *so, const DOHString_or_char *str) {
  int oldlen, newlen, newmaxsize, l, sp;
  char *tc;
  String *s = (String *) ObjData(so);
  char *newstr = 0;

  if (DohCheck(str)) {
    String *ss = (String *) ObjData(str);
    newstr = (char *) String_data((DOH *) str);
    l = ss->len;
  } else {
    newstr = (char *) (str);
    l = (int) strlen(newstr);
  }
  if (!newstr)
    return;
  s->hashkey = -1;

  oldlen = s->len;
  newlen = oldlen + l + 1;
  if (newlen >= s->maxsize - 1) {
    newmaxsize = 2 * s->maxsize;
    if (newlen >= newmaxsize - 1)
      newmaxsize = newlen + 1;
    s->str = (char *) DohRealloc(s->str, newmaxsize);
    assert(s->str);
    s->maxsize = newmaxsize;
  }
  tc = s->str;
  memcpy(tc + oldlen, newstr, l + 1);
  sp = s->sp;
  if (sp >= oldlen) {
    int i = oldlen + l - sp;
    tc += sp;
    for (; i; --i) {
      if (*(tc++) == '\n')
	s->line++;
    }
    s->sp = oldlen + l;
  }
  s->len += l;
}


/* -----------------------------------------------------------------------------
 * String_clear() - Clear a string
 * ----------------------------------------------------------------------------- */

static void String_clear(DOH *so) {
  String *s = (String *) ObjData(so);
  s->hashkey = -1;
  s->len = 0;
  *(s->str) = 0;
  s->sp = 0;
  s->line = 1;
}

/* -----------------------------------------------------------------------------
 * String_insert() - Insert a string
 * ----------------------------------------------------------------------------- */

static int String_insert(DOH *so, int pos, DOH *str) {
  String *s;
  int len;
  char *data;

  if (pos == DOH_END) {
    DohString_append(so, str);
    return 0;
  }


  s = (String *) ObjData(so);
  s->hashkey = -1;
  if (DohCheck(str)) {
    String *ss = (String *) ObjData(str);
    data = (char *) String_data(str);
    len = ss->len;
  } else {
    data = (char *) (str);
    len = (int) strlen(data);
  }

  if (pos < 0)
    pos = 0;
  else if (pos > s->len)
    pos = s->len;

  /* See if there is room to insert the new data */
  while (s->maxsize <= s->len + len) {
    int newsize = 2 * s->maxsize;
    s->str = (char *) DohRealloc(s->str, newsize);
    assert(s->str);
    s->maxsize = newsize;
  }
  memmove(s->str + pos + len, s->str + pos, (s->len - pos));
  memcpy(s->str + pos, data, len);
  if (s->sp >= pos) {
    int i;

    for (i = 0; i < len; i++) {
      if (data[i] == '\n')
	s->line++;
    }
    s->sp += len;
  }
  s->len += len;
  s->str[s->len] = 0;
  return 0;
}

/* -----------------------------------------------------------------------------
 * String_delitem() - Delete a character
 * ----------------------------------------------------------------------------- */

static int String_delitem(DOH *so, int pos) {
  String *s = (String *) ObjData(so);
  s->hashkey = -1;
  if (pos == DOH_END)
    pos = s->len - 1;
  if (pos == DOH_BEGIN)
    pos = 0;
  if (s->len == 0)
    return 0;

  if (s->sp > pos) {
    s->sp--;
    assert(s->sp >= 0);
    if (s->str[pos] == '\n')
      s->line--;
  }
  memmove(s->str + pos, s->str + pos + 1, ((s->len - 1) - pos));
  s->len--;
  s->str[s->len] = 0;
  return 0;
}

/* -----------------------------------------------------------------------------
 * String_delslice() -  Delete a range
 * ----------------------------------------------------------------------------- */

static int String_delslice(DOH *so, int sindex, int eindex) {
  String *s = (String *) ObjData(so);
  int size;
  if (s->len == 0)
    return 0;
  s->hashkey = -1;
  if (eindex == DOH_END)
    eindex = s->len;
  if (sindex == DOH_BEGIN)
    sindex = 0;

  size = eindex - sindex;
  if (s->sp > sindex) {
    /* Adjust the file pointer and line count */
    int i, end;
    if (s->sp > eindex) {
      end = eindex;
      s->sp -= size;
    } else {
      end = s->sp;
      s->sp = sindex;
    }
    for (i = sindex; i < end; i++) {
      if (s->str[i] == '\n')
	s->line--;
    }
    assert(s->sp >= 0);
  }
  memmove(s->str + sindex, s->str + eindex, s->len - eindex);
  s->len -= size;
  s->str[s->len] = 0;
  return 0;
}

/* -----------------------------------------------------------------------------
 * String_str() - Returns a string (used by printing commands)
 * ----------------------------------------------------------------------------- */

static DOH *String_str(DOH *so) {
  String *s = (String *) ObjData(so);
  s->str[s->len] = 0;
  return NewString(s->str);
}

/* -----------------------------------------------------------------------------
 * String_read() - Read data from a string
 * ----------------------------------------------------------------------------- */

static int String_read(DOH *so, void *buffer, int len) {
  int reallen, retlen;
  char *cb;
  String *s = (String *) ObjData(so);
  if ((s->sp + len) > s->len)
    reallen = (s->len - s->sp);
  else
    reallen = len;

  cb = (char *) buffer;
  retlen = reallen;

  if (reallen > 0) {
    memmove(cb, s->str + s->sp, reallen);
    s->sp += reallen;
  }
  return retlen;
}

/* -----------------------------------------------------------------------------
 * String_write() - Write data to a string
 * ----------------------------------------------------------------------------- */
static int String_write(DOH *so, const void *buffer, int len) {
  int newlen;
  String *s = (String *) ObjData(so);
  s->hashkey = -1;
  if (s->sp > s->len)
    s->sp = s->len;
  newlen = s->sp + len + 1;
  if (newlen > s->maxsize) {
    s->str = (char *) DohRealloc(s->str, newlen);
    assert(s->str);
    s->maxsize = newlen;
    s->len = s->sp + len;
  }
  if ((s->sp + len) > s->len)
    s->len = s->sp + len;
  memmove(s->str + s->sp, buffer, len);
  s->sp += len;
  s->str[s->len] = 0;
  return len;
}

/* -----------------------------------------------------------------------------
 * String_seek() - Seek to a new position
 * ----------------------------------------------------------------------------- */

static int String_seek(DOH *so, long offset, int whence) {
  int pos, nsp, inc;
  String *s = (String *) ObjData(so);
  if (whence == SEEK_SET)
    pos = 0;
  else if (whence == SEEK_CUR)
    pos = s->sp;
  else if (whence == SEEK_END) {
    pos = s->len;
    offset = -offset;
  } else
    pos = s->sp;

  nsp = pos + offset;
  if (nsp < 0)
    nsp = 0;
  if (s->len > 0 && nsp > s->len)
    nsp = s->len;

  inc = (nsp > s->sp) ? 1 : -1;

  {
#if 0
    register int sp = s->sp;
    register char *tc = s->str;
    register int len = s->len;
    while (sp != nsp) {
      int prev = sp + inc;
      if (prev >= 0 && prev <= len && tc[prev] == '\n')
	s->line += inc;
      sp += inc;
    }
#else
    register int sp = s->sp;
    register char *tc = s->str;
    if (inc > 0) {
      while (sp != nsp) {
	if (tc[++sp] == '\n')
	  ++s->line;
      }
    } else {
      while (sp != nsp) {
	if (tc[--sp] == '\n')
	  --s->line;
      }
    }
#endif
    s->sp = sp;
  }
  assert(s->sp >= 0);
  return 0;
}

/* -----------------------------------------------------------------------------
 * String_tell() - Return current position
 * ----------------------------------------------------------------------------- */

static long String_tell(DOH *so) {
  String *s = (String *) ObjData(so);
  return (long) (s->sp);
}

/* -----------------------------------------------------------------------------
 * String_putc()
 * ----------------------------------------------------------------------------- */

static int String_putc(DOH *so, int ch) {
  String *s = (String *) ObjData(so);
  register int len = s->len;
  register int sp = s->sp;
  s->hashkey = -1;
  if (sp >= len) {
    register int maxsize = s->maxsize;
    register char *tc = s->str;
    if (len > (maxsize - 2)) {
      maxsize *= 2;
      tc = (char *) DohRealloc(tc, maxsize);
      assert(tc);
      s->maxsize = (int) maxsize;
      s->str = tc;
    }
    tc += sp;
    *tc = (char) ch;
    *(++tc) = 0;
    s->len = s->sp = sp + 1;
  } else {
    s->str[s->sp++] = (char) ch;
  }
  if (ch == '\n')
    s->line++;
  return ch;
}

/* -----------------------------------------------------------------------------
 * String_getc()
 * ----------------------------------------------------------------------------- */

static int String_getc(DOH *so) {
  int c;
  String *s = (String *) ObjData(so);
  if (s->sp >= s->len)
    c = EOF;
  else
    c = (int)(unsigned char) s->str[s->sp++];
  if (c == '\n')
    s->line++;
  return c;
}

/* -----------------------------------------------------------------------------
 * String_ungetc()
 * ----------------------------------------------------------------------------- */

static int String_ungetc(DOH *so, int ch) {
  String *s = (String *) ObjData(so);
  if (ch == EOF)
    return ch;
  if (s->sp <= 0)
    return EOF;
  s->sp--;
  if (ch == '\n')
    s->line--;
  return ch;
}

static char *end_quote(char *s) {
  char *qs;
  char qc;
  char *q;
  char *nl;
  qc = *s;
  qs = s;
  while (1) {
    q = strpbrk(s + 1, "\"\'");
    nl = strchr(s + 1, '\n');
    if (nl && (nl < q)) {
      /* A new line appears before the end of the string */
      if (*(nl - 1) == '\\') {
	s = nl + 1;
	continue;
      }
      /* String was terminated by a newline.  Wing it */
      return qs;
    }
    if (!q && nl) {
      return qs;
    }
    if (!q)
      return 0;
    if ((*q == qc) && (*(q - 1) != '\\'))
      return q;
    s = q;
  }
}

static char *match_simple(char *base, char *s, char *token, int tokenlen) {
  (void) base;
  (void) tokenlen;
  return strstr(s, token);
}

static char *match_identifier(char *base, char *s, char *token, int tokenlen) {
  while (s) {
    s = strstr(s, token);
    if (!s)
      return 0;
    if ((s > base) && (isalnum((int) *(s - 1)) || (*(s - 1) == '_'))) {
      s += tokenlen;
      continue;
    }
    if (isalnum((int) *(s + tokenlen)) || (*(s + tokenlen) == '_')) {
      s += tokenlen;
      continue;
    }
    return s;
  }
  return 0;
}


static char *match_identifier_begin(char *base, char *s, char *token, int tokenlen) {
  while (s) {
    s = strstr(s, token);
    if (!s)
      return 0;
    if ((s > base) && (isalnum((int) *(s - 1)) || (*(s - 1) == '_'))) {
      s += tokenlen;
      continue;
    }
    return s;
  }
  return 0;
}

static char *match_identifier_end(char *base, char *s, char *token, int tokenlen) {
  (void) base;
  while (s) {
    s = strstr(s, token);
    if (!s)
      return 0;
    if (isalnum((int) *(s + tokenlen)) || (*(s + tokenlen) == '_')) {
      s += tokenlen;
      continue;
    }
    return s;
  }
  return 0;
}

static char *match_number_end(char *base, char *s, char *token, int tokenlen) {
  (void) base;
  while (s) {
    s = strstr(s, token);
    if (!s)
      return 0;
    if (isdigit((int) *(s + tokenlen))) {
      s += tokenlen;
      continue;
    }
    return s;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 * replace_simple()
 *
 * Replaces count non-overlapping occurrences of token with rep in a string.   
 * ----------------------------------------------------------------------------- */

static int replace_simple(String *str, char *token, char *rep, int flags, int count, char *(*match) (char *, char *, char *, int)) {
  int tokenlen;			/* Length of the token */
  int replen;			/* Length of the replacement */
  int delta, expand = 0;
  int ic;
  int rcount = 0;
  int noquote = 0;
  char *c, *s, *t, *first;
  char *q, *q2;
  register char *base;
  int i;

  /* Figure out if anything gets replaced */
  if (!strlen(token))
    return 0;

  base = str->str;
  tokenlen = (int)strlen(token);
  s = (*match) (base, base, token, tokenlen);

  if (!s)
    return 0;			/* No matches.  Who cares */

  str->hashkey = -1;

  if (flags & DOH_REPLACE_NOQUOTE)
    noquote = 1;

  /* If we are not replacing inside quotes, we need to do a little extra work */
  if (noquote) {
    q = strpbrk(base, "\"\'");
    if (!q) {
      noquote = 0;		/* Well, no quotes to worry about. Oh well */
    } else {
      while (q && (q < s)) {
	/* First match was found inside a quote.  Try to find another match */
	q2 = end_quote(q);
	if (!q2) {
	  return 0;
	}
	if (q2 > s) {
	  /* Find next match */
	  s = (*match) (base, q2 + 1, token, tokenlen);
	}
	if (!s)
	  return 0;		/* Oh well, no matches */
	q = strpbrk(q2 + 1, "\"\'");
	if (!q)
	  noquote = 0;		/* No more quotes */
      }
    }
  }

  first = s;
  replen = (int)strlen(rep);

  delta = (replen - tokenlen);

  if (delta <= 0) {
    /* String is either shrinking or staying the same size */
    /* In this case, we do the replacement in place without memory reallocation */
    ic = count;
    t = s;			/* Target of memory copies */
    while (ic && s) {
      if (replen) {
	memcpy(t, rep, replen);
	t += replen;
      }
      rcount++;
      expand += delta;
      /* Find the next location */
      s += tokenlen;
      if (ic == 1)
	break;
      c = (*match) (base, s, token, tokenlen);

      if (noquote) {
	q = strpbrk(s, "\"\'");
	if (!q) {
	  noquote = 0;
	} else {
	  while (q && (q < c)) {
	    /* First match was found inside a quote.  Try to find another match */
	    q2 = end_quote(q);
	    if (!q2) {
	      c = 0;
	      break;
	    }
	    if (q2 > c)
	      c = (*match) (base, q2 + 1, token, tokenlen);
	    if (!c)
	      break;
	    q = strpbrk(q2 + 1, "\"\'");
	    if (!q)
	      noquote = 0;	/* No more quotes */
	  }
	}
      }
      if (delta) {
	if (c) {
	  memmove(t, s, c - s);
	  t += (c - s);
	} else {
	  memmove(t, s, (str->str + str->len) - s + 1);
	}
      } else {
	t += (c - s);
      }
      s = c;
      ic--;
    }
    if (s && delta) {
      memmove(t, s, (str->str + str->len) - s + 1);
    }
    str->len += expand;
    str->str[str->len] = 0;
    if (str->sp >= str->len)
      str->sp += expand;	/* Fix the end of file pointer */
    return rcount;
  }
  /* The string is expanding as a result of the replacement */
  /* Figure out how much expansion is going to occur and allocate a new string */
  {
    char *ns;
    int newsize;

    rcount++;
    ic = count - 1;
    s += tokenlen;
    while (ic && (c = (*match) (base, s, token, tokenlen))) {
      if (noquote) {
	q = strpbrk(s, "\"\'");
	if (!q) {
	  break;
	} else {
	  while (q && (q < c)) {
	    /* First match was found inside a quote.  Try to find another match */
	    q2 = end_quote(q);
	    if (!q2) {
	      c = 0;
	      break;
	    }
	    if (q2 > c) {
	      c = (*match) (base, q2 + 1, token, tokenlen);
	      if (!c)
		break;
	    }
	    q = strpbrk(q2 + 1, "\"\'");
	    if (!q)
	      noquote = 0;
	  }
	}
      }
      if (c) {
	rcount++;
	ic--;
	s = c + tokenlen;
      } else {
	break;
      }
    }

    expand = delta * rcount;	/* Total amount of expansion for the replacement */
    newsize = str->maxsize;
    while ((str->len + expand) >= newsize)
      newsize *= 2;

    ns = (char *) DohMalloc(newsize);
    assert(ns);
    t = ns;
    s = first;

    /* Copy the first part of the string */
    if (first > str->str) {
      memcpy(t, str->str, (first - str->str));
      t += (first - str->str);
    }
    for (i = 0; i < rcount; i++) {
      memcpy(t, rep, replen);
      t += replen;
      s += tokenlen;
      c = (*match) (base, s, token, tokenlen);
      if (noquote) {
	q = strpbrk(s, "\"\'");
	if (!q) {
	  noquote = 0;
	} else {
	  while (q && (q < c)) {
	    /* First match was found inside a quote.  Try to find another match */
	    q2 = end_quote(q);
	    if (!q2) {
	      c = 0;
	      break;
	    }
	    if (q2 > c) {
	      c = (*match) (base, q2 + 1, token, tokenlen);
	      if (!c)
		break;
	    }
	    q = strpbrk(q2 + 1, "\"\'");
	    if (!q)
	      noquote = 0;	/* No more quotes */
	  }
	}
      }
      if (i < (rcount - 1)) {
	memcpy(t, s, c - s);
	t += (c - s);
      } else {
	memcpy(t, s, (str->str + str->len) - s + 1);
      }
      s = c;
    }
    c = str->str;
    str->str = ns;
    if (str->sp >= str->len)
      str->sp += expand;
    str->len += expand;
    str->str[str->len] = 0;
    str->maxsize = newsize;
    DohFree(c);
    return rcount;
  }
}

/* -----------------------------------------------------------------------------
 * String_replace()
 * ----------------------------------------------------------------------------- */

static int String_replace(DOH *stro, const DOHString_or_char *token, const DOHString_or_char *rep, int flags) {
  int count = -1;
  String *str = (String *) ObjData(stro);

  if (flags & DOH_REPLACE_FIRST)
    count = 1;

  if (flags & DOH_REPLACE_ID_END) {
    return replace_simple(str, Char(token), Char(rep), flags, count, match_identifier_end);
  } else if (flags & DOH_REPLACE_ID_BEGIN) {
    return replace_simple(str, Char(token), Char(rep), flags, count, match_identifier_begin);
  } else if (flags & DOH_REPLACE_ID) {
    return replace_simple(str, Char(token), Char(rep), flags, count, match_identifier);
  } else if (flags & DOH_REPLACE_NUMBER_END) {
    return replace_simple(str, Char(token), Char(rep), flags, count, match_number_end);
  } else {
    return replace_simple(str, Char(token), Char(rep), flags, count, match_simple);
  }
}

/* -----------------------------------------------------------------------------
 * String_chop()
 * ----------------------------------------------------------------------------- */

static void String_chop(DOH *so) {
  char *c;
  String *str = (String *) ObjData(so);
  /* Replace trailing whitespace */
  c = str->str + str->len - 1;
  while ((str->len > 0) && (isspace((int) *c))) {
    if (str->sp >= str->len) {
      str->sp--;
      if (*c == '\n')
	str->line--;
    }
    str->len--;
    c--;
  }
  str->str[str->len] = 0;
  assert(str->sp >= 0);
  str->hashkey = -1;
}

static void String_setfile(DOH *so, DOH *file) {
  DOH *fo;
  String *str = (String *) ObjData(so);

  if (!DohCheck(file)) {
    fo = NewString(file);
    Decref(fo);
  } else
    fo = file;
  Incref(fo);
  Delete(str->file);
  str->file = fo;
}

static DOH *String_getfile(DOH *so) {
  String *str = (String *) ObjData(so);
  return str->file;
}

static void String_setline(DOH *so, int line) {
  String *str = (String *) ObjData(so);
  str->line = line;
}

static int String_getline(DOH *so) {
  String *str = (String *) ObjData(so);
  return str->line;
}

static DohListMethods StringListMethods = {
  0,				/* doh_getitem */
  0,				/* doh_setitem */
  String_delitem,		/* doh_delitem */
  String_insert,		/* doh_insitem */
  String_delslice,		/* doh_delslice */
};

static DohFileMethods StringFileMethods = {
  String_read,
  String_write,
  String_putc,
  String_getc,
  String_ungetc,
  String_seek,
  String_tell,
  0,				/* close */
};

static DohStringMethods StringStringMethods = {
  String_replace,
  String_chop,
};

DohObjInfo DohStringType = {
  "String",			/* objname */
  DelString,			/* doh_del */
  CopyString,			/* doh_copy */
  String_clear,			/* doh_clear */
  String_str,			/* doh_str */
  String_data,			/* doh_data */
  String_dump,			/* doh_dump */
  String_len,	    	        /* doh_len */
  String_hash,			/* doh_hash    */
  String_cmp,			/* doh_cmp */
  String_equal,	    	        /* doh_equal */
  0,				/* doh_first    */
  0,				/* doh_next     */
  String_setfile,		/* doh_setfile */
  String_getfile,		/* doh_getfile */
  String_setline,		/* doh_setline */
  String_getline,		/* doh_getline */
  0,				/* doh_mapping */
  &StringListMethods,		/* doh_sequence */
  &StringFileMethods,		/* doh_file */
  &StringStringMethods,		/* doh_string */
  0,				/* doh_position */
  0
};


#define INIT_MAXSIZE  16

/* -----------------------------------------------------------------------------
 * NewString() - Create a new string
 * ----------------------------------------------------------------------------- */

DOHString *DohNewString(const DOHString_or_char *so) {
  int l = 0, max;
  String *str;
  char *s;
  int hashkey = -1;
  if (DohCheck(so)) {
    str = (String *) ObjData(so);
    s = (char *) String_data((String *) so);
    l = s ? str->len : 0;
    hashkey = str->hashkey;
  } else {
    s = (char *) so;
    l = s ? (int) strlen(s) : 0;
  }

  str = (String *) DohMalloc(sizeof(String));
  str->hashkey = hashkey;
  str->sp = 0;
  str->line = 1;
  str->file = 0;
  max = INIT_MAXSIZE;
  if (s) {
    if ((l + 1) > max)
      max = l + 1;
  }
  str->str = (char *) DohMalloc(max);
  str->maxsize = max;
  if (s) {
    strcpy(str->str, s);
    str->len = l;
    str->sp = l;
  } else {
    str->str[0] = 0;
    str->len = 0;
  }
  return DohObjMalloc(&DohStringType, str);
}


/* -----------------------------------------------------------------------------
 * NewStringEmpty() - Create a new string
 * ----------------------------------------------------------------------------- */

DOHString *DohNewStringEmpty(void) {
  int max = INIT_MAXSIZE;
  String *str = (String *) DohMalloc(sizeof(String));
  str->hashkey = 0;
  str->sp = 0;
  str->line = 1;
  str->file = 0;
  str->str = (char *) DohMalloc(max);
  str->maxsize = max;
  str->str[0] = 0;
  str->len = 0;
  return DohObjMalloc(&DohStringType, str);
}

/* -----------------------------------------------------------------------------
 * NewStringWithSize() - Create a new string
 * ----------------------------------------------------------------------------- */

DOHString *DohNewStringWithSize(const DOHString_or_char *so, int len) {
  int l = 0, max;
  String *str;
  char *s;
  if (DohCheck(so)) {
    s = (char *) String_data((String *) so);
  } else {
    s = (char *) so;
  }

  str = (String *) DohMalloc(sizeof(String));
  str->hashkey = -1;
  str->sp = 0;
  str->line = 1;
  str->file = 0;
  max = INIT_MAXSIZE;
  if (s) {
    l = (int) len;
    if ((l + 1) > max)
      max = l + 1;
  }
  str->str = (char *) DohMalloc(max);
  str->maxsize = max;
  if (s) {
    strncpy(str->str, s, len);
    str->str[l] = 0;
    str->len = l;
    str->sp = l;
  } else {
    str->str[0] = 0;
    str->len = 0;
  }
  return DohObjMalloc(&DohStringType, str);
}

/* -----------------------------------------------------------------------------
 * NewStringf()
 *
 * Create a new string from a list of objects.
 * ----------------------------------------------------------------------------- */

DOHString *DohNewStringf(const DOHString_or_char *fmt, ...) {
  va_list ap;
  DOH *r;
  va_start(ap, fmt);
  r = NewStringEmpty();
  DohvPrintf(r, Char(fmt), ap);
  va_end(ap);
  return (DOHString *) r;
}

/* -----------------------------------------------------------------------------
 * Strcmp()
 * Strncmp()
 * Strstr()
 * Strchr()
 *
 * Some utility functions.
 * ----------------------------------------------------------------------------- */

int DohStrcmp(const DOHString_or_char *s1, const DOHString_or_char *s2) {
  const char *c1 = Char(s1);
  const char *c2 = Char(s2);
  return strcmp(c1, c2);
}

int DohStrncmp(const DOHString_or_char *s1, const DOHString_or_char *s2, int n) {
  return strncmp(Char(s1), Char(s2), n);
}

char *DohStrstr(const DOHString_or_char *s1, const DOHString_or_char *s2) {
  char *p1 = Char(s1);
  char *p2 = Char(s2);
  return p1 == 0 || p2 == 0 || *p2 == '\0' ? p1 : strstr(p1, p2);
}

char *DohStrchr(const DOHString_or_char *s1, int ch) {
  return strchr(Char(s1), ch);
}
