/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * file.c
 *
 *     This file implements a file-like object that can be built around an 
 *     ordinary FILE * or integer file descriptor.
 * ----------------------------------------------------------------------------- */

#include "dohint.h"

#ifdef DOH_INTFILE
#include <unistd.h>
#endif
#include <errno.h>

typedef struct {
  FILE *filep;
  int fd;
  int closeondel;
} DohFile;

/* -----------------------------------------------------------------------------
 * DelFile()
 * ----------------------------------------------------------------------------- */

static void DelFile(DOH *fo) {
  DohFile *f = (DohFile *) ObjData(fo);
  if (f->closeondel) {
    if (f->filep) {
      fclose(f->filep);
    }
#ifdef DOH_INTFILE
    if (f->fd) {
      close(f->fd);
    }
#endif
  }
  DohFree(f);
}

/* -----------------------------------------------------------------------------
 * File_read()
 * ----------------------------------------------------------------------------- */

static int File_read(DOH *fo, void *buffer, int len) {
  DohFile *f = (DohFile *) ObjData(fo);

  if (f->filep) {
    return (int)fread(buffer, 1, len, f->filep);
  } else if (f->fd) {
#ifdef DOH_INTFILE
    return read(f->fd, buffer, len);
#endif
  }
  return -1;
}

/* -----------------------------------------------------------------------------
 * File_write()
 * ----------------------------------------------------------------------------- */

static int File_write(DOH *fo, const void *buffer, int len) {
  DohFile *f = (DohFile *) ObjData(fo);
  if (f->filep) {
    int ret = (int) fwrite(buffer, 1, len, f->filep);
    int err = (ret != len) ? ferror(f->filep) : 0;
    return err ? -1 : ret;
  } else if (f->fd) {
#ifdef DOH_INTFILE
    return write(f->fd, buffer, len);
#endif
  }
  return -1;
}

/* -----------------------------------------------------------------------------
 * File_seek()
 * ----------------------------------------------------------------------------- */

static int File_seek(DOH *fo, long offset, int whence) {
  DohFile *f = (DohFile *) ObjData(fo);
  if (f->filep) {
    return fseek(f->filep, offset, whence);
  } else if (f->fd) {
#ifdef DOH_INTFILE
    return lseek(f->fd, offset, whence);
#endif
  }
  return -1;
}

/* -----------------------------------------------------------------------------
 * File_tell()
 * ----------------------------------------------------------------------------- */

static long File_tell(DOH *fo) {
  DohFile *f = (DohFile *) ObjData(fo);
  if (f->filep) {
    return ftell(f->filep);
  } else if (f->fd) {
#ifdef DOH_INTFILE
    return lseek(f->fd, 0, SEEK_CUR);
#endif
  }
  return -1;
}

/* -----------------------------------------------------------------------------
 * File_putc()
 * ----------------------------------------------------------------------------- */

static int File_putc(DOH *fo, int ch) {
  DohFile *f = (DohFile *) ObjData(fo);
  if (f->filep) {
    return fputc(ch, f->filep);
  } else if (f->fd) {
#ifdef DOH_INTFILE
    char c;
    c = (char) ch;
    return write(f->fd, &c, 1);
#endif
  }
  return -1;
}

/* -----------------------------------------------------------------------------
 * File_getc()
 * ----------------------------------------------------------------------------- */

static int File_getc(DOH *fo) {
  DohFile *f = (DohFile *) ObjData(fo);
  if (f->filep) {
    return fgetc(f->filep);
  } else if (f->fd) {
#ifdef DOH_INTFILE
    unsigned char c;
    if (read(f->fd, &c, 1) < 0)
      return EOF;
    return c;
#endif
  }
  return EOF;
}

/* -----------------------------------------------------------------------------
 * File_ungetc()
 *
 * Put a character back onto the input
 * ----------------------------------------------------------------------------- */

static int File_ungetc(DOH *fo, int ch) {
  DohFile *f = (DohFile *) ObjData(fo);
  if (f->filep) {
    return ungetc(ch, f->filep);
  } else if (f->fd) {
#ifdef DOH_INTFILE
    /* Not implemented yet */
#endif
  }
  return -1;
}

/* -----------------------------------------------------------------------------
 * File_close()
 *
 * Close the file
 * ----------------------------------------------------------------------------- */

static int File_close(DOH *fo) {
  int ret = 0;
  DohFile *f = (DohFile *) ObjData(fo);
  if (f->filep) {
    ret = fclose(f->filep);
    f->filep = 0;
  } else if (f->fd) {
#ifdef DOH_INTFILE
    ret = close(f->fd);
    f->fd = 0;
#endif
  }
  return ret;
}

static DohFileMethods FileFileMethods = {
  File_read,
  File_write,
  File_putc,
  File_getc,
  File_ungetc,
  File_seek,
  File_tell,
  File_close,			/* close */
};

static DohObjInfo DohFileType = {
  "DohFile",			/* objname      */
  DelFile,			/* doh_del      */
  0,				/* doh_copy     */
  0,				/* doh_clear    */
  0,				/* doh_str      */
  0,				/* doh_data     */
  0,				/* doh_dump     */
  0,				/* doh_len      */
  0,				/* doh_hash     */
  0,				/* doh_cmp      */
  0,				/* doh_equal    */
  0,				/* doh_first    */
  0,				/* doh_next     */
  0,				/* doh_setfile  */
  0,				/* doh_getfile  */
  0,				/* doh_setline  */
  0,				/* doh_getline  */
  0,				/* doh_mapping  */
  0,				/* doh_sequence */
  &FileFileMethods,		/* doh_file     */
  0,				/* doh_string   */
  0,				/* doh_callable */
  0,				/* doh_position */
};

/* -----------------------------------------------------------------------------
 * NewFile()
 *
 * Create a new file from a given filename and mode.
 * If newfiles is non-zero, the filename is added to the list of new files.
 * ----------------------------------------------------------------------------- */

DOH *DohNewFile(DOH *filename, const char *mode, DOHList *newfiles) {
  DohFile *f;
  FILE *file;
  char *filen;

  filen = Char(filename);
  file = fopen(filen, mode);
  if (!file)
    return 0;

  f = (DohFile *) DohMalloc(sizeof(DohFile));
  if (!f) {
    fclose(file);
    return 0;
  }
  if (newfiles)
    Append(newfiles, filename);
  f->filep = file;
  f->fd = 0;
  f->closeondel = 1;
  return DohObjMalloc(&DohFileType, f);
}

/* -----------------------------------------------------------------------------
 * NewFileFromFile()
 *
 * Create a file object from an already open FILE *.
 * ----------------------------------------------------------------------------- */

DOH *DohNewFileFromFile(FILE *file) {
  DohFile *f;
  f = (DohFile *) DohMalloc(sizeof(DohFile));
  if (!f)
    return 0;
  f->filep = file;
  f->fd = 0;
  f->closeondel = 0;
  return DohObjMalloc(&DohFileType, f);
}

/* -----------------------------------------------------------------------------
 * NewFileFromFd()
 *
 * Create a file object from an already open FILE *.
 * ----------------------------------------------------------------------------- */

DOH *DohNewFileFromFd(int fd) {
  DohFile *f;
  f = (DohFile *) DohMalloc(sizeof(DohFile));
  if (!f)
    return 0;
  f->filep = 0;
  f->fd = fd;
  f->closeondel = 0;
  return DohObjMalloc(&DohFileType, f);
}

/* -----------------------------------------------------------------------------
 * FileErrorDisplay()
 *
 * Display cause of one of the NewFile functions failing.
 * ----------------------------------------------------------------------------- */

void DohFileErrorDisplay(DOHString * filename) {
  Printf(stderr, "Unable to open file %s: %s\n", filename, strerror(errno));
}
