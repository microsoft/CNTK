/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * include.c
 *
 * The functions in this file are used to manage files in the SWIG library.
 * General purpose functions for opening, including, and retrieving pathnames
 * are provided.
 * ----------------------------------------------------------------------------- */

#include "swig.h"

static List   *directories = 0;	        /* List of include directories */
static String *lastpath = 0;	        /* Last file that was included */
static List   *pdirectories = 0;        /* List of pushed directories  */
static int     dopush = 1;		/* Whether to push directories */
static int file_debug = 0;

/* This functions determine whether to push/pop dirs in the preprocessor */
void Swig_set_push_dir(int push) {
  dopush = push;
}

int Swig_get_push_dir(void) {
  return dopush;
}

/* -----------------------------------------------------------------------------
 * Swig_add_directory()
 *
 * Adds a directory to the SWIG search path.
 * ----------------------------------------------------------------------------- */

List *Swig_add_directory(const_String_or_char_ptr dirname) {
  String *adirname;
  if (!directories)
    directories = NewList();
  assert(directories);
  if (dirname) {
    adirname = NewString(dirname);
    Append(directories,adirname);
    Delete(adirname);
  }
  return directories;
}

/* -----------------------------------------------------------------------------
 * Swig_push_directory()
 *
 * Inserts a directory at the front of the SWIG search path.  This is used by
 * the preprocessor to grab files in the same directory as other included files.
 * ----------------------------------------------------------------------------- */

void Swig_push_directory(const_String_or_char_ptr dirname) {
  String *pdirname;
  if (!Swig_get_push_dir())
    return;
  if (!pdirectories)
    pdirectories = NewList();
  assert(pdirectories);
  pdirname = NewString(dirname);
  assert(pdirname);
  Insert(pdirectories,0,pdirname);
  Delete(pdirname);
}

/* -----------------------------------------------------------------------------
 * Swig_pop_directory()
 *
 * Pops a directory off the front of the SWIG search path.  This is used by
 * the preprocessor.
 * ----------------------------------------------------------------------------- */

void Swig_pop_directory(void) {
  if (!Swig_get_push_dir())
    return;
  if (!pdirectories)
    return;
  Delitem(pdirectories, 0);
}

/* -----------------------------------------------------------------------------
 * Swig_last_file()
 * 
 * Returns the full pathname of the last file opened. 
 * ----------------------------------------------------------------------------- */

String *Swig_last_file(void) {
  assert(lastpath);
  return lastpath;
}

/* -----------------------------------------------------------------------------
 * Swig_search_path_any() 
 * 
 * Returns a list of the current search paths.
 * ----------------------------------------------------------------------------- */

static List *Swig_search_path_any(int syspath) {
  String *filename;
  List   *slist;
  int     i, ilen;

  slist = NewList();
  assert(slist);
  filename = NewStringEmpty();
  assert(filename);
#ifdef MACSWIG
  Printf(filename, "%s", SWIG_FILE_DELIMITER);
#else
  Printf(filename, ".%s", SWIG_FILE_DELIMITER);
#endif
  Append(slist, filename);
  Delete(filename);
  
  /* If there are any pushed directories.  Add them first */
  if (pdirectories) {
    ilen = Len(pdirectories);
    for (i = 0; i < ilen; i++) {
      filename = NewString(Getitem(pdirectories,i));
      Append(filename,SWIG_FILE_DELIMITER);
      Append(slist,filename);
      Delete(filename);
    }
  }
  /* Add system directories next */
  ilen = Len(directories);
  for (i = 0; i < ilen; i++) {
    filename = NewString(Getitem(directories,i));
    Append(filename,SWIG_FILE_DELIMITER);
    if (syspath) {
      /* If doing a system include, put the system directories first */
      Insert(slist,i,filename);
    } else {
      /* Otherwise, just put the system directories after the pushed directories (if any) */
      Append(slist,filename);
    }
    Delete(filename);
  }
  return slist;
}

List *Swig_search_path() {
  return Swig_search_path_any(0);
}



/* -----------------------------------------------------------------------------
 * Swig_open()
 *
 * open a file, optionally looking for it in the include path.  Returns an open  
 * FILE * on success.
 * ----------------------------------------------------------------------------- */

static FILE *Swig_open_file(const_String_or_char_ptr name, int sysfile, int use_include_path) {
  FILE *f;
  String *filename;
  List *spath = 0;
  char *cname;
  int i, ilen, nbytes;
  char bom[3];

  if (!directories)
    directories = NewList();
  assert(directories);

  cname = Char(name);
  filename = NewString(cname);
  assert(filename);
  if (file_debug) {
    Printf(stdout, "  Open: %s\n", filename);
  }
  f = fopen(Char(filename), "r");
  if (!f && use_include_path) {
    spath = Swig_search_path_any(sysfile);
    ilen = Len(spath);
    for (i = 0; i < ilen; i++) {
      Clear(filename);
      Printf(filename, "%s%s", Getitem(spath, i), cname);
      f = fopen(Char(filename), "r");
      if (f)
	break;
    }
    Delete(spath);
  }
  if (f) {
    Delete(lastpath);
    lastpath = filename;

    /* Skip the UTF-8 BOM if it's present */
    nbytes = (int)fread(bom, 1, 3, f);
    if (nbytes == 3 && bom[0] == (char)0xEF && bom[1] == (char)0xBB && bom[2] == (char)0xBF) {
      /* skip */
    } else {
      fseek(f, 0, SEEK_SET);
    }
  }
  return f;
}

/* Open a file - searching the include paths to find it */
FILE *Swig_include_open(const_String_or_char_ptr name) {
  return Swig_open_file(name, 0, 1);
}

/* Open a file - does not use include paths to find it */
FILE *Swig_open(const_String_or_char_ptr name) {
  return Swig_open_file(name, 0, 0);
}



/* -----------------------------------------------------------------------------
 * Swig_read_file()
 * 
 * Reads data from an open FILE * and returns it as a string.
 * ----------------------------------------------------------------------------- */

String *Swig_read_file(FILE *f) {
  int len;
  char buffer[4096];
  String *str = NewStringEmpty();

  assert(str);
  while (fgets(buffer, 4095, f)) {
    Append(str, buffer);
  }
  len = Len(str);
  /* Add a newline if not present on last line -- the preprocessor seems to 
   * rely on \n and not EOF terminating lines */
  if (len) {
    char *cstr = Char(str);
    if (cstr[len - 1] != '\n') {
      Append(str, "\n");
    }
  }
  return str;
}

/* -----------------------------------------------------------------------------
 * Swig_include()
 *
 * Opens a file and returns it as a string.
 * ----------------------------------------------------------------------------- */

static String *Swig_include_any(const_String_or_char_ptr name, int sysfile) {
  FILE *f;
  String *str;
  String *file;

  f = Swig_open_file(name, sysfile, 1);
  if (!f)
    return 0;
  str = Swig_read_file(f);
  fclose(f);
  Seek(str, 0, SEEK_SET);
  file = Copy(Swig_last_file());
  Setfile(str, file);
  Delete(file);
  Setline(str, 1);
  return str;
}

String *Swig_include(const_String_or_char_ptr name) {
  return Swig_include_any(name, 0);
}

String *Swig_include_sys(const_String_or_char_ptr name) {
  return Swig_include_any(name, 1);
}

/* -----------------------------------------------------------------------------
 * Swig_insert_file()
 *
 * Copies the contents of a file into another file
 * ----------------------------------------------------------------------------- */

int Swig_insert_file(const_String_or_char_ptr filename, File *outfile) {
  char buffer[4096];
  int nbytes;
  FILE *f = Swig_include_open(filename);

  if (!f)
    return -1;
  while ((nbytes = Read(f, buffer, 4096)) > 0) {
    Write(outfile, buffer, nbytes);
  }
  fclose(f);
  return 0;
}

/* -----------------------------------------------------------------------------
 * Swig_register_filebyname()
 *
 * Register a "named" file with the core.  Named files can become targets
 * for %insert directives and other SWIG operations.  This function takes
 * the place of the f_header, f_wrapper, f_init, and other global variables
 * in SWIG1.1
 * ----------------------------------------------------------------------------- */

static Hash *named_files = 0;

void Swig_register_filebyname(const_String_or_char_ptr filename, File *outfile) {
  if (!named_files)
    named_files = NewHash();
  Setattr(named_files, filename, outfile);
}

/* -----------------------------------------------------------------------------
 * Swig_filebyname()
 *
 * Get a named file
 * ----------------------------------------------------------------------------- */

File *Swig_filebyname(const_String_or_char_ptr filename) {
  if (!named_files)
    return 0;
  return Getattr(named_files, filename);
}

/* -----------------------------------------------------------------------------
 * Swig_file_extension()
 *
 * Returns the extension of a file
 * ----------------------------------------------------------------------------- */

String *Swig_file_extension(const_String_or_char_ptr filename) {
  String *name = Swig_file_filename(filename);
  const char *c = strrchr(Char(name), '.');
  String *extension = c ? NewString(c) : NewString("");
  Delete(name);
  return extension;
}

/* -----------------------------------------------------------------------------
 * Swig_file_basename()
 *
 * Returns the filename with the extension removed.
 * ----------------------------------------------------------------------------- */

String *Swig_file_basename(const_String_or_char_ptr filename) {
  String *extension = Swig_file_extension(filename);
  String *basename = NewStringWithSize(filename, Len(filename) - Len(extension));
  Delete(extension);
  return basename;
}

/* -----------------------------------------------------------------------------
 * Swig_file_filename()
 *
 * Return the file name with any leading path stripped off
 * ----------------------------------------------------------------------------- */
String *Swig_file_filename(const_String_or_char_ptr filename) {
  const char *delim = SWIG_FILE_DELIMITER;
  const char *c = strrchr(Char(filename), *delim);
  return c ? NewString(c + 1) : NewString(filename);
}

/* -----------------------------------------------------------------------------
 * Swig_file_dirname()
 *
 * Return the name of the directory associated with a file
 * ----------------------------------------------------------------------------- */
String *Swig_file_dirname(const_String_or_char_ptr filename) {
  const char *delim = SWIG_FILE_DELIMITER;
  const char *c = strrchr(Char(filename), *delim);
  return c ? NewStringWithSize(filename, (int)(c - Char(filename) + 1)) : NewString("");
}

/*
 * Swig_file_debug()
 */
void Swig_file_debug_set() {
  file_debug = 1;
}
