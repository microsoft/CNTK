/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * swigfile.h
 *
 * File handling functions in the SWIG core
 * ----------------------------------------------------------------------------- */

extern List   *Swig_add_directory(const_String_or_char_ptr dirname);
extern void    Swig_push_directory(const_String_or_char_ptr dirname);
extern void    Swig_pop_directory(void);
extern String *Swig_last_file(void);
extern List   *Swig_search_path(void);
extern FILE   *Swig_include_open(const_String_or_char_ptr name);
extern FILE   *Swig_open(const_String_or_char_ptr name);
extern String *Swig_read_file(FILE *f); 
extern String *Swig_include(const_String_or_char_ptr name);
extern String *Swig_include_sys(const_String_or_char_ptr name);
extern int     Swig_insert_file(const_String_or_char_ptr name, File *outfile);
extern void    Swig_set_push_dir(int dopush);
extern int     Swig_get_push_dir(void);
extern void    Swig_register_filebyname(const_String_or_char_ptr filename, File *outfile);
extern File   *Swig_filebyname(const_String_or_char_ptr filename);
extern String *Swig_file_extension(const_String_or_char_ptr filename);
extern String *Swig_file_basename(const_String_or_char_ptr filename);
extern String *Swig_file_filename(const_String_or_char_ptr filename);
extern String *Swig_file_dirname(const_String_or_char_ptr filename);
extern void   Swig_file_debug_set();

/* Delimiter used in accessing files and directories */

#if defined(MACSWIG)
#  define SWIG_FILE_DELIMITER ":"
#elif defined(_WIN32)
#  define SWIG_FILE_DELIMITER "\\"
#else
#  define SWIG_FILE_DELIMITER "/"
#endif
