# ===========================================================================
#      http://www.gnu.org/software/autoconf-archive/ax_path_generic.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_PATH_GENERIC(LIBRARY,[MINIMUM-VERSION,[SED-EXPR-EXTRACTOR]],[ACTION-IF-FOUND],[ACTION-IF-NOT-FOUND],[CONFIG-SCRIPTS],[CFLAGS-ARG],[LIBS-ARG])
#
# DESCRIPTION
#
#   Runs the LIBRARY-config script and defines LIBRARY_CFLAGS and
#   LIBRARY_LIBS unless the user had predefined them in the environment.
#
#   The script must support `--cflags' and `--libs' args. If MINIMUM-VERSION
#   is specified, the script must also support the `--version' arg. If the
#   `--with-library-[exec-]prefix' arguments to ./configure are given, it
#   must also support `--prefix' and `--exec-prefix'. Prefereable use
#   CONFIG-SCRIPTS as config script, CFLAGS-ARG instead of `--cflags` and
#   LIBS-ARG instead of `--libs`, if given.
#
#   The SED-EXPR-EXTRACTOR parameter representes the expression used in sed
#   to extract the version number. Use it if your 'foo-config --version'
#   dumps something like 'Foo library v1.0.0 (alfa)' instead of '1.0.0'.
#
#   The macro respects LIBRARY_CONFIG, LIBRARY_CFLAGS and LIBRARY_LIBS
#   variables. If the first one is defined, it specifies the name of the
#   config script to use. If the latter two are defined, the script is not
#   ran at all and their values are used instead (if only one of them is
#   defined, the empty value of the remaining one is still used).
#
#   Example:
#
#     AX_PATH_GENERIC(Foo, 1.0.0)
#
#   would run `foo-config --version' and check that it is at least 1.0.0, if
#   successful the following variables would be defined and substituted:
#
#     FOO_CFLAGS to `foo-config --cflags`
#     FOO_LIBS   to `foo-config --libs`
#
#   Example:
#
#     AX_PATH_GENERIC([Bar],,,[
#        AC_MSG_ERROR([Cannot find Bar library])
#     ])
#
#   would check for bar-config program, defining and substituting the
#   following variables:
#
#     BAR_CFLAGS to `bar-config --cflags`
#     BAR_LIBS   to `bar-config --libs`
#
#   Example:
#
#     ./configure BAZ_LIBS=/usr/lib/libbaz.a
#
#   would link with a static version of baz library even if `baz-config
#   --libs` returns just "-lbaz" that would normally result in using the
#   shared library.
#
#   This macro is a rearranged version of AC_PATH_GENERIC from Angus Lees.
#
# LICENSE
#
#   Copyright (c) 2009 Francesco Salvestrini <salvestrini@users.sourceforge.net>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved. This file is offered as-is, without any
#   warranty.

#serial 13

AU_ALIAS([AC_PATH_GENERIC], [AX_PATH_GENERIC])
AC_DEFUN([AX_PATH_GENERIC],[
  AC_REQUIRE([AC_PROG_SED])

  dnl we're going to need uppercase and lowercase versions of the
  dnl string `LIBRARY'
  pushdef([UP],   translit([$1], [a-z], [A-Z]))dnl
  pushdef([DOWN], translit([$1], [A-Z], [a-z]))dnl

  AC_ARG_WITH(DOWN-prefix,[AS_HELP_STRING([--with-]DOWN[-prefix=PREFIX], [Prefix where $1 is installed (optional)])],
    DOWN[]_config_prefix="$withval", DOWN[]_config_prefix="")
  AC_ARG_WITH(DOWN-exec-prefix,[AS_HELP_STRING([--with-]DOWN[-exec-prefix=EPREFIX], [Exec prefix where $1 is installed (optional)])],
    DOWN[]_config_exec_prefix="$withval", DOWN[]_config_exec_prefix="")

  AC_ARG_VAR(UP[]_CONFIG, [config script used for $1])
  AC_ARG_VAR(UP[]_CFLAGS, [CFLAGS used for $1])
  AC_ARG_VAR(UP[]_LIBS,   [LIBS used for $1])

  AS_IF([test x"$UP[]_CFLAGS" != x -o x"$UP[]_LIBS" != x],[
    dnl Don't run config script at all, use user-provided values instead.
    AC_SUBST(UP[]_CFLAGS)
    AC_SUBST(UP[]_LIBS)
    :
    $4
  ],[
    AS_IF([test x$DOWN[]_config_exec_prefix != x],[
      DOWN[]_config_args="$DOWN[]_config_args --exec-prefix=$DOWN[]_config_exec_prefix"
      AS_IF([test x${UP[]_CONFIG+set} != xset],[
	UP[]_CONFIG=$DOWN[]_config_exec_prefix/bin/DOWN-config
      ])
    ])
    AS_IF([test x$DOWN[]_config_prefix != x],[
      DOWN[]_config_args="$DOWN[]_config_args --prefix=$DOWN[]_config_prefix"
      AS_IF([test x${UP[]_CONFIG+set} != xset],[
	UP[]_CONFIG=$DOWN[]_config_prefix/bin/DOWN-config
      ])
    ])

    AC_PATH_PROGS(UP[]_CONFIG,[$6 DOWN-config],[no])
    AS_IF([test "$UP[]_CONFIG" = "no"],[
      :
      $5
    ],[
      dnl Get the CFLAGS from LIBRARY-config script
      AS_IF([test x"$7" = x],[
	UP[]_CFLAGS="`$UP[]_CONFIG $DOWN[]_config_args --cflags`"
      ],[
	UP[]_CFLAGS="`$UP[]_CONFIG $DOWN[]_config_args $7`"
      ])

      dnl Get the LIBS from LIBRARY-config script
      AS_IF([test x"$8" = x],[
	UP[]_LIBS="`$UP[]_CONFIG $DOWN[]_config_args --libs`"
      ],[
	UP[]_LIBS="`$UP[]_CONFIG $DOWN[]_config_args $8`"
      ])

      AS_IF([test x"$2" != x],[
	dnl Check for provided library version
	AS_IF([test x"$3" != x],[
	  dnl Use provided sed expression
	  DOWN[]_version="`$UP[]_CONFIG $DOWN[]_config_args --version | $SED -e $3`"
	],[
	  DOWN[]_version="`$UP[]_CONFIG $DOWN[]_config_args --version | $SED -e 's/^\ *\(.*\)\ *$/\1/'`"
	])

	AC_MSG_CHECKING([for $1 ($DOWN[]_version) >= $2])
	AX_COMPARE_VERSION($DOWN[]_version,[ge],[$2],[
	  AC_MSG_RESULT([yes])

	  AC_SUBST(UP[]_CFLAGS)
	  AC_SUBST(UP[]_LIBS)
	  :
	  $4
	],[
	  AC_MSG_RESULT([no])
	  :
	  $5
	])
      ],[
	AC_SUBST(UP[]_CFLAGS)
	AC_SUBST(UP[]_LIBS)
	:
	$4
      ])
    ])
  ])

  popdef([UP])
  popdef([DOWN])
])
