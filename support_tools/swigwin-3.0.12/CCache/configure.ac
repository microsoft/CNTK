dnl Process this file with autoconf to produce a configure script.

AC_INIT([ccache-swig], [0.0]) # Get version from SWIG in ccache_swig_config.h.in
AC_PREREQ(2.52)
AC_CONFIG_SRCDIR([ccache.h])

AC_MSG_NOTICE([Configuring ccache])

AC_CONFIG_HEADER(config.h)

dnl Checks for programs.
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_ARG_PROGRAM # for program_transform_name

AC_DEFINE([_GNU_SOURCE], 1,
          [Define _GNU_SOURCE so that we get all necessary prototypes])

# If GCC, turn on warnings.
if test "x$GCC" = "xyes"
then
	CFLAGS="$CFLAGS -Wall -W"
else
	CFLAGS="$CFLAGS -O"
fi

AC_HEADER_DIRENT
AC_HEADER_TIME
AC_HEADER_SYS_WAIT

AC_CHECK_HEADERS(ctype.h strings.h stdlib.h string.h pwd.h sys/time.h)

AC_CHECK_FUNCS(realpath snprintf vsnprintf vasprintf asprintf mkstemp)
AC_CHECK_FUNCS(gethostname getpwuid)
AC_CHECK_FUNCS(utimes)

AC_CACHE_CHECK([for compar_fn_t in stdlib.h],ccache_cv_COMPAR_FN_T, [
    AC_TRY_COMPILE(
[#include <stdlib.h>],
[
void test_fn(void) { qsort(NULL, 0, 0, (__compar_fn_t)NULL); }
],
	ccache_cv_COMPAR_FN_T=yes,ccache_cv_COMPAR_FN_T=no)])
if test x"$ccache_cv_COMPAR_FN_T" = x"yes"; then
   AC_DEFINE(HAVE_COMPAR_FN_T, 1, [ ])
fi

dnl Note: This could be replaced by AC_FUNC_SNPRINTF() in the autoconf macro archive
AC_CACHE_CHECK([for C99 vsnprintf],ccache_cv_HAVE_C99_VSNPRINTF,[
AC_TRY_RUN([
#include <sys/types.h>
#include <stdarg.h>
void foo(const char *format, ...) { 
       va_list ap;
       int len;
       char buf[5];

       va_start(ap, format);
       len = vsnprintf(0, 0, format, ap);
       va_end(ap);
       if (len != 5) exit(1);

       if (snprintf(buf, 3, "hello") != 5 || strcmp(buf, "he") != 0) exit(1);

       exit(0);
}
main() { foo("hello"); }
],
ccache_cv_HAVE_C99_VSNPRINTF=yes,ccache_cv_HAVE_C99_VSNPRINTF=no,ccache_cv_HAVE_C99_VSNPRINTF=cross)])
if test x"$ccache_cv_HAVE_C99_VSNPRINTF" = x"yes"; then
    AC_DEFINE(HAVE_C99_VSNPRINTF, 1, [ ])
fi

dnl Check for zlib.
dnl Note: This could be replaced by CHECK_ZLIB() in the autoconf macro archive
AC_ARG_ENABLE([zlib],
              AS_HELP_STRING([--enable-zlib], [enable zlib support for ccache compression]),,
              [enable_zlib=yes])

if test x"$enable_zlib" = x"yes"; then
    AC_CHECK_HEADER(zlib.h, AC_CHECK_LIB(z, gzdopen, [LIBS="-lz $LIBS"
						      AC_DEFINE([ENABLE_ZLIB], 1, [Define to 1 if you would like to have zlib compression for ccache.]) ] ))
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
