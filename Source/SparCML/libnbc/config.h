/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
#define F77_FUNC(name,NAME) name ## _

/* As F77_FUNC, but for C identifiers containing underscores. */
#define F77_FUNC_(name,NAME) name ## _

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* enables the dcmf module */
/* #undef HAVE_DCMF */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the `free' function. */
#define HAVE_FREE 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `c' library (-lc). */
#define HAVE_LIBC 1

/* Define to 1 if you have the `ibverbs' library (-libverbs). */
/* #undef HAVE_LIBIBVERBS */

/* Define to 1 if you have the `m' library (-lm). */
#define HAVE_LIBM 1

/* Define to 1 if you have the `pthread' library (-lpthread). */
/* #undef HAVE_LIBPTHREAD */

/* Define to 1 if you have the `log' function. */
#define HAVE_LOG 1

/* Define to 1 if you have the `malloc' function. */
#define HAVE_MALLOC 1

/* Define to 1 if you have the <math.h> header file. */
#define HAVE_MATH_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `memset' function. */
#define HAVE_MEMSET 1

/* enables MPI code */
#define HAVE_MPI 1

/* found MPI-2.2 */
#define HAVE_MPI22 1

/* Define to 1 if you have the <mpi.h> header file. */
#define HAVE_MPI_H 1

/* Define to 1 if you have the `MPI_Init' function. */
#define HAVE_MPI_INIT 1

/* enables the ofed module */
/* #undef HAVE_OFED */

/* enables Open MPI glue code */
/* #undef HAVE_OMPI */

/* Define to 1 if you have the `pow' function. */
#define HAVE_POW 1

/* Define to 1 if you have the `printf' function. */
#define HAVE_PRINTF 1

/* compile with progress thread */
/* #undef HAVE_PROGRESS_THREAD */

/* Define to 1 if you have the <pthread.h> header file. */
/* #undef HAVE_PTHREAD_H */

/* Define to 1 if you have the `realloc' function. */
#define HAVE_REALLOC 1

/* compile with progress thread */
/* #undef HAVE_RT_THREAD */

/* Define to 1 if you have the `sqrt' function. */
#define HAVE_SQRT 1

/* Define to 1 if you have the <stdarg.h> header file. */
#define HAVE_STDARG_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define this if your system can create weak aliases */
#define HAVE_SYS_WEAK_ALIAS 1

/* Define this if weak aliases may be created with __attribute__ */
#define HAVE_SYS_WEAK_ALIAS_ATTRIBUTE 1

/* Define this if weak aliases may be created with #pragma _CRI duplicate */
/* #undef HAVE_SYS_WEAK_ALIAS_CRIDUPLICATE */

/* Define this if weak aliases in other files are honored */
/* #undef HAVE_SYS_WEAK_ALIAS_CROSSFILE */

/* Define this if weak aliases may be created with #pragma _HP_SECONDARY_DEF
   */
/* #undef HAVE_SYS_WEAK_ALIAS_HPSECONDARY */

/* Define this if weak aliases may be created with #pragma weak */
#define HAVE_SYS_WEAK_ALIAS_PRAGMA 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "libNBC"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT ""

/* Define to the full name of this package. */
#define PACKAGE_NAME "libNBC"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "libNBC 1.1.1"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "libnbc"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.1.1"

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#define TIME_WITH_SYS_TIME 1

/* Version number of package */
#define VERSION "1.1.1"

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif
