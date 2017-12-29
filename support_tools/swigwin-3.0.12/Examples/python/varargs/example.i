/* File : example.i */
%module example

%{
#include <stdarg.h>
%}

/* This example illustrates SWIG's handling of varargs functions.  
   By default, variable length arguments are simply ignored.  This
   is generally appropriate for wrapping I/O functions like printf.
   You can simply format a string in the scripting language, and
   pass it directly */

int printf(const char *fmt, ...);

/* Since passing a format string might be dangerous.  Here is a slightly
   different way of wrapping a printf style function */

#if 1
/* Replace ... with char *.   */
%varargs(char *) fprintf;

/* Ignore the format string, but set it to %s */
%typemap(in,numinputs=0) const char *fmt {
	$1 = "%s";
}
#else
/* An alternative approach using typemaps */
%typemap(in) (const char *fmt, ...) {
   $1 = "%s";
   $2 = (void *) PyString_AsString($input);
}
#endif

int fprintf(FILE *, const char *fmt, ...);

/* Here is somewhat different example.  A variable length argument
   function that takes a NULL-terminated list of arguments. We
   can use a slightly different form of %varargs that specifies
   a default value and a maximum number of arguments.
 */

/* Maximum of 20 arguments with default value NULL */

%varargs(20, char *x = NULL) printv;

%inline %{

/* In Python 2 we could use PyFile_AsFile for converting Python sys.stdout to C's stdout.
   This API disappeared in Python 3, so instead we use a helper function to get stdout */
FILE * stdout_stream(void) {
  return stdout;
}

void printv(char *s, ...) {
    va_list ap;
    char *x;
    fputs(s,stdout);
    fputc(' ',stdout);
    va_start(ap, s);
    while ((x = va_arg(ap, char *))) {
      fputs(x,stdout);
      fputc(' ',stdout);
    }
    va_end(ap);
    fputc('\n',stdout);
}
%}

