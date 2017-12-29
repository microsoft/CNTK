/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * wrapfunc.c
 *
 * This file defines a object for creating wrapper functions.  Primarily
 * this is used for convenience since it allows pieces of a wrapper function
 * to be created in a piecemeal manner.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include <ctype.h>

static int Compact_mode = 0;	/* set to 0 on default */
static int Max_line_size = 128;

/* -----------------------------------------------------------------------------
 * NewWrapper()
 *
 * Create a new wrapper function object.
 * ----------------------------------------------------------------------------- */

Wrapper *NewWrapper(void) {
  Wrapper *w;
  w = (Wrapper *) malloc(sizeof(Wrapper));
  w->localh = NewHash();
  w->locals = NewStringEmpty();
  w->code = NewStringEmpty();
  w->def = NewStringEmpty();
  return w;
}

/* -----------------------------------------------------------------------------
 * DelWrapper()
 *
 * Delete a wrapper function object.
 * ----------------------------------------------------------------------------- */

void DelWrapper(Wrapper *w) {
  Delete(w->localh);
  Delete(w->locals);
  Delete(w->code);
  Delete(w->def);
  free(w);
}

/* -----------------------------------------------------------------------------
 * Wrapper_compact_print_mode_set()
 *
 * Set compact_mode.
 * ----------------------------------------------------------------------------- */

void Wrapper_compact_print_mode_set(int flag) {
  Compact_mode = flag;
}

/* -----------------------------------------------------------------------------
 * Wrapper_pretty_print()
 *
 * Formats a wrapper function and fixes up the indentation.
 * ----------------------------------------------------------------------------- */

void Wrapper_pretty_print(String *str, File *f) {
  String *ts;
  int level = 0;
  int c, i;
  int empty = 1;
  int indent = 2;
  int plevel = 0;
  int label = 0;

  ts = NewStringEmpty();
  Seek(str, 0, SEEK_SET);
  while ((c = Getc(str)) != EOF) {
    if (c == '\"') {
      Putc(c, ts);
      while ((c = Getc(str)) != EOF) {
	if (c == '\\') {
	  Putc(c, ts);
	  c = Getc(str);
	}
	Putc(c, ts);
	if (c == '\"')
	  break;
      }
      empty = 0;
    } else if (c == '\'') {
      Putc(c, ts);
      while ((c = Getc(str)) != EOF) {
	if (c == '\\') {
	  Putc(c, ts);
	  c = Getc(str);
	}
	Putc(c, ts);
	if (c == '\'')
	  break;
      }
      empty = 0;
    } else if (c == ':') {
      Putc(c, ts);
      if ((c = Getc(str)) == '\n') {
	if (!empty && !strchr(Char(ts), '?'))
	  label = 1;
      }
      Ungetc(c, str);
    } else if (c == '(') {
      Putc(c, ts);
      plevel += indent;
      empty = 0;
    } else if (c == ')') {
      Putc(c, ts);
      plevel -= indent;
      empty = 0;
    } else if (c == '{') {
      Putc(c, ts);
      Putc('\n', ts);
      for (i = 0; i < level; i++)
	Putc(' ', f);
      Printf(f, "%s", ts);
      Clear(ts);
      level += indent;
      while ((c = Getc(str)) != EOF) {
	if (!isspace(c)) {
	  Ungetc(c, str);
	  break;
	}
      }
      empty = 0;
    } else if (c == '}') {
      if (!empty) {
	Putc('\n', ts);
	for (i = 0; i < level; i++)
	  Putc(' ', f);
	Printf(f, "%s", ts);
	Clear(ts);
      }
      level -= indent;
      Putc(c, ts);
      empty = 0;
    } else if (c == '\n') {
      Putc(c, ts);
      empty = 0;
      if (!empty) {
	int slevel = level;
	if (label && (slevel >= indent))
	  slevel -= indent;
	if ((Char(ts))[0] != '#') {
	  for (i = 0; i < slevel; i++)
	    Putc(' ', f);
	}
	Printf(f, "%s", ts);
	for (i = 0; i < plevel; i++)
	  Putc(' ', f);
      }
      Clear(ts);
      label = 0;
      empty = 1;
    } else if (c == '/') {
      empty = 0;
      Putc(c, ts);
      c = Getc(str);
      if (c != EOF) {
	Putc(c, ts);
	if (c == '/') {		/* C++ comment */
	  while ((c = Getc(str)) != EOF) {
	    if (c == '\n') {
	      Ungetc(c, str);
	      break;
	    }
	    Putc(c, ts);
	  }
	} else if (c == '*') {	/* C comment */
	  int endstar = 0;
	  while ((c = Getc(str)) != EOF) {
	    if (endstar && c == '/') {	/* end of C comment */
	      Putc(c, ts);
	      break;
	    }
	    endstar = (c == '*');
	    Putc(c, ts);
	    if (c == '\n') {	/* multi-line C comment. Could be improved slightly. */
	      for (i = 0; i < level; i++)
		Putc(' ', ts);
	    }
	  }
	}
      }
    } else {
      if (!empty || !isspace(c)) {
	Putc(c, ts);
	empty = 0;
      }
    }
  }
  if (!empty)
    Printf(f, "%s", ts);
  Delete(ts);
  Printf(f, "\n");
}

/* -----------------------------------------------------------------------------
 * Wrapper_compact_print()
 *
 * Formats a wrapper function and fixes up the indentation.
 * Print out in compact format, with Compact enabled.
 * ----------------------------------------------------------------------------- */

void Wrapper_compact_print(String *str, File *f) {
  String *ts, *tf;		/*temp string & temp file */
  int level = 0;
  int c, i;
  int empty = 1;
  int indent = 2;

  ts = NewStringEmpty();
  tf = NewStringEmpty();
  Seek(str, 0, SEEK_SET);

  while ((c = Getc(str)) != EOF) {
    if (c == '\"') {		/* string 1 */
      empty = 0;
      Putc(c, ts);
      while ((c = Getc(str)) != EOF) {
	if (c == '\\') {
	  Putc(c, ts);
	  c = Getc(str);
	}
	Putc(c, ts);
	if (c == '\"')
	  break;
      }
    } else if (c == '\'') {	/* string 2 */
      empty = 0;
      Putc(c, ts);
      while ((c = Getc(str)) != EOF) {
	if (c == '\\') {
	  Putc(c, ts);
	  c = Getc(str);
	}
	Putc(c, ts);
	if (c == '\'')
	  break;
      }
    } else if (c == '{') {	/* start of {...} */
      empty = 0;
      Putc(c, ts);
      if (Len(tf) == 0) {
	for (i = 0; i < level; i++)
	  Putc(' ', tf);
      } else if ((Len(tf) + Len(ts)) < Max_line_size) {
	Putc(' ', tf);
      } else {
	Putc('\n', tf);
	Printf(f, "%s", tf);
	Clear(tf);
	for (i = 0; i < level; i++)
	  Putc(' ', tf);
      }
      Append(tf, ts);
      Clear(ts);
      level += indent;
      while ((c = Getc(str)) != EOF) {
	if (!isspace(c)) {
	  Ungetc(c, str);
	  break;
	}
      }
    } else if (c == '}') {	/* end of {...} */
      empty = 0;
      if (Len(tf) == 0) {
	for (i = 0; i < level; i++)
	  Putc(' ', tf);
      } else if ((Len(tf) + Len(ts)) < Max_line_size) {
	Putc(' ', tf);
      } else {
	Putc('\n', tf);
	Printf(f, "%s", tf);
	Clear(tf);
	for (i = 0; i < level; i++)
	  Putc(' ', tf);
      }
      Append(tf, ts);
      Putc(c, tf);
      Clear(ts);
      level -= indent;
    } else if (c == '\n') {	/* line end */
      while ((c = Getc(str)) != EOF) {
	if (!isspace(c))
	  break;
      }
      if (c == '#') {
	Putc('\n', ts);
      } else if (c == '}') {
	Putc(' ', ts);
      } else if ((c != EOF) || (Len(ts) != 0)) {
	if (Len(tf) == 0) {
	  for (i = 0; i < level; i++)
	    Putc(' ', tf);
	} else if ((Len(tf) + Len(ts)) < Max_line_size) {
	  Putc(' ', tf);
	} else {
	  Putc('\n', tf);
	  Printf(f, "%s", tf);
	  Clear(tf);
	  for (i = 0; i < level; i++)
	    Putc(' ', tf);
	}
	Append(tf, ts);
	Clear(ts);
      }
      Ungetc(c, str);

      empty = 1;
    } else if (c == '/') {	/* comment */
      empty = 0;
      c = Getc(str);
      if (c != EOF) {
	if (c == '/') {		/* C++ comment */
	  while ((c = Getc(str)) != EOF) {
	    if (c == '\n') {
	      Ungetc(c, str);
	      break;
	    }
	  }
	} else if (c == '*') {	/* C comment */
	  int endstar = 0;
	  while ((c = Getc(str)) != EOF) {
	    if (endstar && c == '/') {	/* end of C comment */
	      break;
	    }
	    endstar = (c == '*');
	  }
	} else {
	  Putc('/', ts);
	  Putc(c, ts);
	}
      }
    } else if (c == '#') {	/* Preprocessor line */
      Putc('#', ts);
      while ((c = Getc(str)) != EOF) {
	Putc(c, ts);
	if (c == '\\') {	/* Continued line of the same PP */
	  c = Getc(str);
	  if (c == '\n')
	    Putc(c, ts);
	  else
	    Ungetc(c, str);
	} else if (c == '\n')
	  break;
      }
      if (!empty) {
	Append(tf, "\n");
      }
      Append(tf, ts);
      Printf(f, "%s", tf);
      Clear(tf);
      Clear(ts);
      for (i = 0; i < level; i++)
	Putc(' ', tf);
      empty = 1;
    } else {
      if (!empty || !isspace(c)) {
	Putc(c, ts);
	empty = 0;
      }
    }
  }
  if (!empty) {
    Append(tf, ts);
  }
  if (Len(tf) != 0)
    Printf(f, "%s", tf);
  Delete(ts);
  Delete(tf);
  Printf(f, "\n");
}

/* -----------------------------------------------------------------------------
 * Wrapper_print()
 *
 * Print out a wrapper function.  Does pretty or compact printing as well.
 * ----------------------------------------------------------------------------- */

void Wrapper_print(Wrapper *w, File *f) {
  String *str;

  str = NewStringEmpty();
  Printf(str, "%s\n", w->def);
  Printf(str, "%s\n", w->locals);
  Printf(str, "%s\n", w->code);
  if (Compact_mode == 1)
    Wrapper_compact_print(str, f);
  else
    Wrapper_pretty_print(str, f);

  Delete(str);
}

/* -----------------------------------------------------------------------------
 * Wrapper_add_local()
 *
 * Adds a new local variable declaration to a function. Returns -1 if already
 * present (which may or may not be okay to the caller).
 * ----------------------------------------------------------------------------- */

int Wrapper_add_local(Wrapper *w, const_String_or_char_ptr name, const_String_or_char_ptr decl) {
  /* See if the local has already been declared */
  if (Getattr(w->localh, name)) {
    return -1;
  }
  Setattr(w->localh, name, decl);
  Printf(w->locals, "%s;\n", decl);
  return 0;
}

/* -----------------------------------------------------------------------------
 * Wrapper_add_localv()
 *
 * Same as add_local(), but allows a NULL terminated list of strings to be
 * used as a replacement for decl.   This saves the caller the trouble of having
 * to manually construct the 'decl' string before calling.
 * ----------------------------------------------------------------------------- */

int Wrapper_add_localv(Wrapper *w, const_String_or_char_ptr name, ...) {
  va_list ap;
  int ret;
  String *decl;
  DOH *obj;
  decl = NewStringEmpty();
  va_start(ap, name);

  obj = va_arg(ap, void *);
  while (obj) {
    Append(decl, obj);
    Putc(' ', decl);
    obj = va_arg(ap, void *);
  }
  va_end(ap);

  ret = Wrapper_add_local(w, name, decl);
  Delete(decl);
  return ret;
}

/* -----------------------------------------------------------------------------
 * Wrapper_check_local()
 *
 * Check to see if a local name has already been declared
 * ----------------------------------------------------------------------------- */

int Wrapper_check_local(Wrapper *w, const_String_or_char_ptr name) {
  if (Getattr(w->localh, name)) {
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------------- 
 * Wrapper_new_local()
 *
 * Adds a new local variable with a guarantee that a unique local name will be
 * used.  Returns the name that was actually selected.
 * ----------------------------------------------------------------------------- */

char *Wrapper_new_local(Wrapper *w, const_String_or_char_ptr name, const_String_or_char_ptr decl) {
  int i;
  String *nname = NewString(name);
  String *ndecl = NewString(decl);
  char *ret;

  i = 0;

  while (Wrapper_check_local(w, nname)) {
    Clear(nname);
    Printf(nname, "%s%d", name, i);
    i++;
  }
  Replace(ndecl, name, nname, DOH_REPLACE_ID);
  Setattr(w->localh, nname, ndecl);
  Printf(w->locals, "%s;\n", ndecl);
  ret = Char(nname);
  Delete(nname);
  Delete(ndecl);
  return ret;			/* Note: nname should still exists in the w->localh hash */
}


/* -----------------------------------------------------------------------------
 * Wrapper_new_localv()
 *
 * Same as add_local(), but allows a NULL terminated list of strings to be
 * used as a replacement for decl.   This saves the caller the trouble of having
 * to manually construct the 'decl' string before calling.
 * ----------------------------------------------------------------------------- */

char *Wrapper_new_localv(Wrapper *w, const_String_or_char_ptr name, ...) {
  va_list ap;
  char *ret;
  String *decl;
  DOH *obj;
  decl = NewStringEmpty();
  va_start(ap, name);

  obj = va_arg(ap, void *);
  while (obj) {
    Append(decl, obj);
    Putc(' ', decl);
    obj = va_arg(ap, void *);
  }
  va_end(ap);

  ret = Wrapper_new_local(w, name, decl);
  Delete(decl);
  return ret;
}
