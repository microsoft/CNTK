dnl Process this file with autoconf to produce a configure script.
dnl The macros which aren't shipped with the autotools are stored in the
dnl Tools/config directory in .m4 files.

AC_INIT([swig],[3.0.12],[http://www.swig.org])

dnl NB: When this requirement is increased to 2.60 or later, AC_PROG_SED
dnl     definition below can be removed
AC_PREREQ(2.58)

AC_CONFIG_SRCDIR([Source/Swig/swig.h])
AC_CONFIG_AUX_DIR([Tools/config])
AC_CONFIG_HEADERS([Source/Include/swigconfig.h])
AC_CANONICAL_HOST
AM_INIT_AUTOMAKE

dnl Some extra defines for the config file
AH_BOTTOM([
/* Default language */
#define SWIG_LANG               "-tcl"

/* Deal with attempt by Microsoft to deprecate C standard runtime functions */
#if defined(_MSC_VER)
# define _CRT_SECURE_NO_DEPRECATE
#endif
])

dnl Check for programs that a user requires to build SWIG
AC_PROG_CC
AC_PROG_CXX
AC_EXEEXT
AC_OBJEXT
AM_PROG_CC_C_O  # Needed for subdir-objects in AUTOMAKE_OPTIONS

AC_COMPILE_WARNINGS # Increase warning levels

AC_DEFINE_UNQUOTED(SWIG_CXX, ["$CXX"], [Compiler that built SWIG])
AC_DEFINE_UNQUOTED(SWIG_PLATFORM, ["$host"], [Platform that SWIG is built for])

dnl Checks for header files.
AC_HEADER_STDC

dnl Look for popen
AC_ARG_WITH(popen, AS_HELP_STRING([--without-popen], [Disable popen]), with_popen="$withval")
if test x"${with_popen}" = xno ; then
AC_MSG_NOTICE([Disabling popen])
else
AC_CHECK_FUNC(popen, AC_DEFINE(HAVE_POPEN, 1, [Define if popen is available]), AC_MSG_NOTICE([Disabling popen]))
fi

dnl PCRE

dnl AX_PATH_GENERIC() relies on AC_PROG_SED() but it is defined only in
dnl autoconf 2.60 so trivially predefine it ourselves for the older versions
m4_ifdef([AC_PROG_SED],, [AC_DEFUN([AC_PROG_SED], [AC_PATH_PROG([SED], sed)])])

AC_ARG_WITH([pcre],
  [AS_HELP_STRING([--without-pcre],
                  [Disable support for regular expressions using PCRE])],
  [],
  [with_pcre=yes])

AC_MSG_CHECKING([whether to enable PCRE support])
AC_MSG_RESULT([$with_pcre])

dnl To make configuring easier, check for a locally built PCRE using the Tools/pcre-build.sh script
if test x"${with_pcre}" = xyes ; then
  AC_MSG_CHECKING([whether to use local PCRE])
  local_pcre_config=no
  if test -z $PCRE_CONFIG; then
    if test -f `pwd`/pcre/pcre-swig-install/bin/pcre-config; then
      PCRE_CONFIG=`pwd`/pcre/pcre-swig-install/bin/pcre-config
      local_pcre_config=$PCRE_CONFIG
    fi
  fi
  AC_MSG_RESULT([$local_pcre_config])
fi
AS_IF([test "x$with_pcre" != xno],
  [AX_PATH_GENERIC([pcre],
    [], dnl Minimal version of PCRE we need -- accept any
    [], dnl custom sed script for version parsing is not needed
    [AC_DEFINE([HAVE_PCRE], [1], [Define if you have PCRE library])
     LIBS="$LIBS $PCRE_LIBS"
     CPPFLAGS="$CPPFLAGS $PCRE_CFLAGS"
    ],
    [AC_MSG_FAILURE([
        Cannot find pcre-config script from PCRE (Perl Compatible Regular Expressions)
        library package. This dependency is needed for configure to complete,
        Either:
        - Install the PCRE developer package on your system (preferred approach).
        - Download the PCRE source tarball, build and install on your system
          as you would for any package built from source distribution.
        - Use the Tools/pcre-build.sh script to build PCRE just for SWIG to statically
          link against. Run 'Tools/pcre-build.sh --help' for instructions.
          (quite easy and does not require privileges to install PCRE on your system)
        - Use configure --without-pcre to disable regular expressions support in SWIG
          (not recommended).])
    ])
  ])


dnl CCache
AC_ARG_ENABLE([ccache], AS_HELP_STRING([--disable-ccache], [disable building and installation of ccache-swig executable (default enabled)]), [enable_ccache=$enableval], [enable_ccache=yes])
AC_MSG_CHECKING([whether to enable ccache-swig])
AC_MSG_RESULT([$enable_ccache])

if test "$enable_ccache" = yes; then
  AC_CONFIG_SUBDIRS(CCache)
  ENABLE_CCACHE=1
fi
AC_SUBST(ENABLE_CCACHE)


echo ""
echo "Checking packages required for SWIG developers."
echo "Note : None of the following packages are required for users to compile and install SWIG from the distributed tarball"
echo ""

AC_PROG_YACC

echo ""
echo "Checking for installed target languages and other information in order to compile and run"
echo "the examples and test-suite invoked by 'make check'."
echo "Note : None of the following packages are required for users to compile and install SWIG from the distributed tarball"
echo ""

dnl Some test cases require Boost
AX_BOOST_BASE(,,,)
AC_SUBST(BOOST_CPPFLAGS)

dnl How to specify include directories that may be system directories.
# -I should not be used on system directories (GCC)
if test "$GCC" = yes; then
    ISYSTEM="-isystem "
else
    ISYSTEM="-I"
fi
AC_MSG_NOTICE(ISYSTEM: $ISYSTEM)

dnl Info for building shared libraries ... in order to run the examples

# SO is the extension of shared libraries (including the dot!)
AC_MSG_CHECKING(SO)
if test -z "$SO"
then
	case $host in
	*-*-hp*) SO=.sl;;
	*-*-darwin*) SO=.bundle;;
	*-*-cygwin* | *-*-mingw*) SO=.dll;;
	*) SO=.so;;
	esac
fi
AC_MSG_RESULT($SO)

# LDSHARED is the ld *command* used to create shared library
# -- "ld" on SunOS 4.x.x, "ld -G" on SunOS 5.x, "ld -shared" on IRIX 5
# (Shared libraries in this instance are shared modules to be loaded into
# Python, as opposed to building Python itself as a shared library.)
AC_MSG_CHECKING(LDSHARED)
if test -z "$LDSHARED"
then
	case $host in
	*-*-aix*) LDSHARED="\$(srcdir)/ld_so_aix \$(CC)";;
	*-*-cygwin* | *-*-mingw*)
            if test "$GCC" = yes; then
                LDSHARED="$CC -shared"
            else
                if test "cl" = $CC ;  then
                    # Microsoft Visual C++ (MSVC)
                    LDSHARED="$CC -nologo -LD"
                else
                    # Unknown compiler try gcc approach
                    LDSHARED="$CC -shared"
                fi
            fi ;;
	*-*-irix5*) LDSHARED="ld -shared";;
	*-*-irix6*) LDSHARED="ld ${SGI_ABI} -shared -all";;
	*-*-sunos4*) LDSHARED="ld";;
	*-*-solaris*) LDSHARED="ld -G";;
	*-*-hp*) LDSHARED="ld -b";;
	*-*-osf*) LDSHARED="ld -shared -expect_unresolved \"*\"";;
	*-sequent-sysv4) LDSHARED="ld -G";;
	*-*-next*)
		if test "$ns_dyld"
		then LDSHARED='$(CC) $(LDFLAGS) -bundle -prebind'
		else LDSHARED='$(CC) $(CFLAGS) -nostdlib -r'
		fi
                if test "$with_next_framework" ; then
		    LDSHARED="$LDSHARED \$(LDLIBRARY)"
		fi ;;
	*-*-linux*) LDSHARED="gcc -shared";;
	*-*-dgux*) LDSHARED="ld -G";;
	*-*-freebsd3*) LDSHARED="gcc -shared";;
	*-*-freebsd* | *-*-openbsd*) LDSHARED="ld -Bshareable";;
	*-*-netbsd*)
		if [[ "`$CC -dM -E - </dev/null | grep __ELF__`" != "" ]]
		then
			LDSHARED="cc -shared"
		else
			LDSHARED="ld -Bshareable"
		fi;;
	*-sco-sysv*) LDSHARED="cc -G -KPIC -Ki486 -belf -Wl,-Bexport";;
	*-*-darwin*) LDSHARED="cc -bundle -undefined suppress -flat_namespace";;
	*)	LDSHARED="ld";;
	esac
fi
AC_MSG_RESULT($LDSHARED)
# CXXSHARED is the ld *command* used to create C++ shared library
# -- "ld" on SunOS 4.x.x, "ld -G" on SunOS 5.x, "ld -shared" on IRIX 5
# (Shared libraries in this instance are shared modules to be loaded into
# Python, as opposed to building Python itself as a shared library.)
AC_MSG_CHECKING(CXXSHARED)
if test -z "$CXXSHARED"
then
	CXXSHARED="$LDSHARED"
fi
AC_MSG_RESULT($CXXSHARED)

#
AC_MSG_CHECKING(TRYLINKINGWITHCXX)
if test -z "$TRYLINKINGWITHCXX"
then
	case $host in
	*-*-solaris*) if test "$GCC" = yes
             then TRYLINKINGWITHCXX="CXXSHARED= $CXX -Wl,-G"
             else TRYLINKINGWITHCXX="CXXSHARED= $CXX -G -L/opt/SUNWspro/lib -lCrun -lCstd"
             fi;;
        *-*-hp*) TRYLINKINGWITHCXX="CXXSHARED= $CXX +z ";;
        *-*-darwin*) TRYLINKINGWITHCXX="CXXSHARED= $CXX -bundle -undefined suppress -flat_namespace";;
        *-*-cygwin* | *-*-mingw*)
            if test "$GCC" = yes; then
                TRYLINKINGWITHCXX="CXXSHARED= $CXX -shared "
            else
                if test "cl" = $CXX ;  then
                    # Microsoft Visual C++ (MSVC)
                    TRYLINKINGWITHCXX="CXXSHARED= $CXX -nologo -LD"
                else
                    TRYLINKINGWITHCXX="#unknown Windows compiler"
                fi
            fi ;;
        *)       TRYLINKINGWITHCXX="CXXSHARED= $CXX -shared ";;
        esac
fi
AC_MSG_RESULT($TRYLINKINGWITHCXX)
# CCSHARED are the C *flags* used to create objects to go into a shared
# library (module) -- this is only needed for a few systems
AC_MSG_CHECKING(CCSHARED)
if test -z "$CCSHARED"
then
	case $host in
	*-*-hp*) if test "$GCC" = yes
		 then CCSHARED="-fpic"
		 else CCSHARED="+z"
		 fi;;
	s390x*-*-*) CCSHARED="-fpic" ;;
	s390*-*-*) CCSHARED="-fPIC" ;;
	*-*-linux*) CCSHARED="-fpic";;
	*-*-freebsd* | *-*-openbsd*) CCSHARED="-fpic";;
	*-*-netbsd*) CCSHARED="-fPIC";;
	*-sco-sysv*) CCSHARED="-KPIC -dy -Bdynamic";;
	*-*-irix6*)  case $CC in
		   *gcc*) CCSHARED="-shared";;
		   *) CCSHARED="";;
		   esac;;
	esac
fi
AC_MSG_RESULT($CCSHARED)

# RPATH is the path used to look for shared library files.
AC_MSG_CHECKING(RPATH)
if test -z "$RPATH"
then
	case $host in
	*-*-solaris*) RPATH='-R. -R$(exec_prefix)/lib';;
        *-*-irix*) RPATH='-rpath .:$(exec_prefix)/lib';;
	*-*-linux*) RPATH='-Xlinker -rpath $(exec_prefix)/lib -Xlinker -rpath .';;
	*)	RPATH='';;
	esac
fi
AC_MSG_RESULT($RPATH)

# LINKFORSHARED are the flags passed to the $(CC) command that links
# the a few executables -- this is only needed for a few systems

AC_MSG_CHECKING(LINKFORSHARED)
if test -z "$LINKFORSHARED"
then
	case $host in
	*-*-aix*)	LINKFORSHARED='-Wl,-bE:$(srcdir)/python.exp -lld';;
	*-*-hp*)
	    LINKFORSHARED="-Wl,-E -Wl,+s -Wl,+b\$(BINLIBDEST)/lib-dynload";;
	*-*-linux*) LINKFORSHARED="-Xlinker -export-dynamic";;
	*-*-next*) LINKFORSHARED="-u libsys_s";;
	*-sco-sysv*) LINKFORSHARED="-Bdynamic -dy -Wl,-Bexport";;
	*-*-irix6*) LINKFORSHARED="-all";;
	esac
fi
AC_MSG_RESULT($LINKFORSHARED)

# Optional CFLAGS used to silence/enhance compiler warnings on some platforms.
AC_MSG_CHECKING(PLATCFLAGS)
case $host in
  *-*-solaris*) if test "$GCC" = yes
    then PLATCFLAGS=
    else PLATCFLAGS=
      #    else PLATCFLAGS="-errtags=yes" # Need more work as C examples use ld for linking
    fi;;
  *) PLATCFLAGS=
esac
AC_MSG_RESULT($PLATCFLAGS)

# Add switch if necessary to enable C++11 support - just for tests
AC_ARG_ENABLE([cpp11-testing], AS_HELP_STRING([--enable-cpp11-testing], [enable C++11 testing if supported by compiler (default disabled)]), [enable_cpp11_testing=$enableval], [enable_cpp11_testing=no])
AC_MSG_CHECKING([whether to enable C++11 testing])
AC_MSG_RESULT([$enable_cpp11_testing])

PLATCXXFLAGS="$PLATCFLAGS"
if test x"$enable_cpp11_testing" = xyes; then
  AC_LANG_PUSH([C++])
  CXXFLAGS_SAVED=$CXXFLAGS
  CXXFLAGS=
  AX_CXX_COMPILE_STDCXX_11([noext], [nostop])
  CXXFLAGS=$CXXFLAGS_SAVED
  AC_LANG_POP([C++])
  if test x"$CXX11FLAGS" != x; then
    PLATCXXFLAGS="$CXX11FLAGS $PLATCXXFLAGS"
  fi
  AC_MSG_CHECKING([for C++11 enabled compiler])
  if test x"$HAVE_CXX11_COMPILER" = x; then
    AC_MSG_RESULT([no])
  else
    AC_MSG_RESULT([$HAVE_CXX11_COMPILER])
  fi
fi

# On darwin 10.7,10.8,10.9 using clang++, need to ensure using
# libc++ for tests and examples to run under mono. May affect
# other language targets as well - problem is a Mac OS X incompatibility
# between libraries depending on libstdc++ and libc++.
CLANGXX=
$CXX -v 2>&1 | grep -i clang >/dev/null && CLANGXX=yes
case $host in
  *-*-darwin11* | *-*-darwin12* |*-*-darwin13*  ) if test "$CLANGXX" = "yes";
    then PLATCXXFLAGS="$PLATCXXFLAGS -stdlib=libc++"
    fi;;
  *) ;;
esac

# Set info about shared libraries.
AC_SUBST(SO)
AC_SUBST(LDSHARED)
AC_SUBST(CCSHARED)
AC_SUBST(CXXSHARED)
AC_SUBST(TRYLINKINGWITHCXX)
AC_SUBST(RPATH)
AC_SUBST(PLATCFLAGS)
AC_SUBST(PLATCXXFLAGS)
AC_SUBST(HAVE_CXX11_COMPILER)
AC_SUBST(LINKFORSHARED)

# This variation is needed on OS-X because there is no (apparent) consistency in shared library naming.
# Sometimes .bundle works, but sometimes .so is needed.  It depends on the target language

AC_SUBST(PYTHON_SO)
case $host in
   *-*-mingw*) PYTHON_SO=.pyd;;
   *-*-darwin*) PYTHON_SO=.so;;
   *) PYTHON_SO=$SO;;
esac

AC_SUBST(TCL_SO)
case $host in
   *-*-darwin*) TCL_SO=.dylib;;
   *) TCL_SO=$SO;;
esac

AC_SUBST(GUILE_SO)
case $host in
   *-*-darwin*) GUILE_SO=.so;;
   *) GUILE_SO=$SO;;
esac

AC_SUBST(PHP_SO)
case $host in
   *-*-darwin*) PHP_SO=.so;;
   *) PHP_SO=$SO;;
esac

AC_SUBST(PHP5_SO)
PHP5_SO=$PHP_SO

AC_SUBST(MZSCHEME_SO)
case $host in
   *) MZSCHEME_SO=.so;;
esac

AC_SUBST(LUA_SO)
case $host in
   *-*-darwin*) LUA_SO=.so;;
   *) LUA_SO=$SO;;
esac

# Check for specific libraries.   Used for SWIG examples
AC_CHECK_LIB(dl, dlopen)	# Dynamic linking for SunOS/Solaris and SYSV
AC_CHECK_LIB(dld, shl_load)	# Dynamic linking for HP-UX

dnl The following three libraries (nsl,inet,socket) are needed on Sequent,
dnl and must be checked for in this order since each library depends on the
dnl preceding one.
dnl
dnl Most SVR4 platforms will need -lsocket and -lnsl.  However on SGI IRIX 5,
dnl these exist but are broken, so we use AC_SEARCH_LIBS which will only try
dnl the library if the function isn't already available without it.
AC_SEARCH_LIBS(t_open, nsl) # SVR4
AC_SEARCH_LIBS(gethostbyname, inet) # Sequent
AC_SEARCH_LIBS(socket, socket) # SVR4 sockets

AC_CHECK_LIB(swill, swill_init, [SWIGLIBS="-lswill $LIBS" SWILL="-DSWIG_SWILL"])
AC_SUBST(SWIGLIBS)
AC_SUBST(SWILL)

# check for --with-libm=...
AC_SUBST(LIBM)
LIBM=-lm
AC_ARG_WITH(libm, [  --with-libm=STRING      math library], [
if test "$withval" != yes
then LIBM=$withval
else AC_MSG_ERROR([proper usage is --with-libm=STRING])
fi])
AC_CHECK_LIB(ieee, main, [LIBM="-lieee $LIBM"])
AC_CHECK_LIB(crypt,crypt, [LIBCRYPT="-lcrypt"])
AC_SUBST(LIBCRYPT)

# check for --with-libc=...
AC_SUBST(LIBC)
AC_ARG_WITH(libc, [  --with-libc=STRING      C library], [
if test "$withval" != yes
then LIBC=$withval
else AC_MSG_ERROR([proper usage is --with-libc=STRING])
fi])

#--------------------------------------------------------------------
# Target languages
#--------------------------------------------------------------------

AC_ARG_WITH(alllang, AS_HELP_STRING([--without-alllang], [Disable all languages]), with_alllang="$withval")

if test "$with_alllang" = "no"; then
  alllang_default=no
else
  alllang_default=yes
fi

AC_CHECK_PROGS(PKGCONFIG, [pkg-config])

#--------------------------------------------------------------------
# Look for Tcl
#--------------------------------------------------------------------

TCLINCLUDE=
TCLLIB=
TCLPACKAGE=

AC_ARG_WITH(tclconfig, AS_HELP_STRING([--without-tcl], [Disable Tcl])
AS_HELP_STRING([--with-tclconfig=path], [Set location of tclConfig.sh]), [with_tclconfig="$withval"], [with_tclconfig=])
AC_ARG_WITH(tcl,
 [  --with-tcl=path         Set location of Tcl package],[
	TCLPACKAGE="$withval"], [TCLPACKAGE="$alllang_default"])
AC_ARG_WITH(tclincl,[  --with-tclincl=path     Set location of Tcl include directory],[
	TCLINCLUDE="$ISYSTEM$withval"], [TCLINCLUDE=])
AC_ARG_WITH(tcllib,[  --with-tcllib=path      Set location of Tcl library directory],[
	TCLLIB="-L$withval"], [TCLLIB=])

# First, check for "--without-tcl" or "--with-tcl=no".
if test x"${TCLPACKAGE}" = xno; then
AC_MSG_NOTICE([Disabling Tcl])
else
AC_MSG_CHECKING([for Tcl configuration])
# First check to see if --with-tclconfig was specified.
if test x"${with_tclconfig}" != x ; then
   if test -f "${with_tclconfig}/tclConfig.sh" ; then
      TCLCONFIG=`(cd ${with_tclconfig}; pwd)`
   else
      AC_MSG_ERROR([${with_tcl} directory does not contain tclConfig.sh])
   fi
fi
# check in a few common install locations
if test x"${TCLCONFIG}" = x ; then
    for i in `ls -d -r /usr/lib*/ 2>/dev/null` \
	     `ls -d -r /usr/lib*/tcl*/ 2>/dev/null` \
	     `ls -d -r /usr/local/lib*/ 2>/dev/null` \
	     `ls -d -r /usr/local/lib*/tcl*/ 2>/dev/null` ; do
	if test -f $i"tclConfig.sh" ; then
	    TCLCONFIG=`(cd $i; pwd)`
	    break
	fi
    done
fi
if test x"${TCLCONFIG}" = x ; then
    AC_MSG_RESULT(no)
else
    AC_MSG_RESULT(found $TCLCONFIG/tclConfig.sh)
    . $TCLCONFIG/tclConfig.sh
    if test -z "$TCLINCLUDE"; then
        TCLINCLUDE=`echo $TCL_INCLUDE_SPEC | sed "s/-I/$ISYSTEM/"`
    fi
    if test -z "$TCLLIB"; then
        TCLLIB=$TCL_LIB_SPEC
    fi
fi

if test -z "$TCLINCLUDE"; then
   if test "x$TCLPACKAGE" != xyes; then
	TCLINCLUDE="$ISYSTEM$TCLPACKAGE/include"
   fi
fi

if test -z "$TCLLIB"; then
   if test "x$TCLPACKAGE" != xyes; then
	TCLLIB="-L$TCLPACKAGE/lib -ltcl"
   fi
fi

AC_MSG_CHECKING(for Tcl header files)
if test -z "$TCLINCLUDE"; then
AC_TRY_CPP([#include <tcl.h>], , TCLINCLUDE="")
if test -z "$TCLINCLUDE"; then
	dirs="/usr/local/include /usr/include /opt/local/include"
	for i in $dirs ; do
		if test -r $i/tcl.h; then
			AC_MSG_RESULT($i)
			TCLINCLUDE="$ISYSTEM$i"
			break
		fi
	done
fi
if test -z "$TCLINCLUDE"; then
    	AC_MSG_RESULT(not found)
fi
else
        AC_MSG_RESULT($TCLINCLUDE)
fi

AC_MSG_CHECKING(for Tcl library)
if test -z "$TCLLIB"; then
dirs="/usr/local/lib /usr/lib /opt/local/lib"
for i in $dirs ; do
	if test -r $i/libtcl.a; then
	    AC_MSG_RESULT($i)
	    TCLLIB="-L$i -ltcl"
	    break
	fi
done
if test -z "$TCLLIB"; then
	AC_MSG_RESULT(not found)
fi
else
AC_MSG_RESULT($TCLLIB)
fi

# Cygwin (Windows) needs the library for dynamic linking
case $host in
*-*-cygwin* | *-*-mingw*) TCLDYNAMICLINKING="$TCLLIB";;
*)TCLDYNAMICLINKING="";;
esac

case $host in
*-*-darwin*)
    TCLLDSHARED='$(CC) -dynamiclib -undefined suppress -flat_namespace'
    TCLCXXSHARED='$(CXX) -dynamiclib -undefined suppress -flat_namespace'
    ;;
*)
    TCLLDSHARED='$(LDSHARED)'
    TCLCXXSHARED='$(CXXSHARED)'
    ;;
esac

fi

AC_SUBST(TCLINCLUDE)
AC_SUBST(TCLLIB)
AC_SUBST(TCLDYNAMICLINKING)
AC_SUBST(TCLLDSHARED)
AC_SUBST(TCLCXXSHARED)

#----------------------------------------------------------------
# Look for Python
#----------------------------------------------------------------

PYINCLUDE=
PYLIB=
PYLINK=
PYPACKAGE=

AC_ARG_WITH(python, AS_HELP_STRING([--without-python], [Disable Python])
AS_HELP_STRING([--with-python=path], [Set location of Python executable]),[ PYBIN="$withval"], [PYBIN="$alllang_default"])

# First, check for "--without-python" or "--with-python=no".
if test x"${PYBIN}" = xno; then
  AC_MSG_NOTICE([Disabling Python])
else
  # First figure out the name of the Python executable
  if test "x$PYBIN" = xyes; then
    AC_CHECK_PROGS(PYTHON, [python python2.8 python2.7 python2.6 python2.5 python2.4 python2.3 python2.2 python2.1 python2.0])
  else
    PYTHON="$PYBIN"
  fi

  PYVER=0
  if test -n "$PYTHON"; then
    AC_MSG_CHECKING([for $PYTHON major version number])
    PYVER=`($PYTHON -c "import sys; sys.stdout.write(sys.version[[0]])") 2>/dev/null`
    AC_MSG_RESULT($PYVER)
    if test -z "$PYVER"; then
      PYVER=0
    else
      AC_MSG_CHECKING(for Python os.name)
      PYOSNAME=`($PYTHON -c "import sys, os; sys.stdout.write(os.name)")`
      AC_MSG_RESULT($PYOSNAME)
    fi
  fi

  if test $PYVER -eq 1 -o $PYVER -eq 2; then
    AC_MSG_CHECKING(for Python prefix)
    PYPREFIX=`($PYTHON -c "import sys; sys.stdout.write(sys.prefix)") 2>/dev/null`
    AC_MSG_RESULT($PYPREFIX)
    AC_MSG_CHECKING(for Python exec-prefix)
    PYEPREFIX=`($PYTHON -c "import sys; sys.stdout.write(sys.exec_prefix)") 2>/dev/null`
    AC_MSG_RESULT($PYEPREFIX)

    if test x"$PYOSNAME" = x"nt"; then
      # Windows installations are quite different to posix installations
      PYPREFIX=`echo "$PYPREFIX" | sed -e 's,\\\\,/,g'` # Forward slashes are easier to use and even work on Windows most of the time
      PYTHON_SO=.pyd

      AC_MSG_CHECKING(for Python header files)
      if test -r $PYPREFIX/include/Python.h; then
        PYINCLUDE="-I$PYPREFIX/include"
      fi
      AC_MSG_RESULT($PYINCLUDE)

      AC_MSG_CHECKING(for Python library directory)
      if test -d $PYPREFIX/libs; then
        PYLIB=$PYPREFIX/libs
        PYLINKFILE=`ls $PYLIB/python*.lib | grep "python[[0-9]][[0-9]]\.lib"`
        if test -r "$PYLINKFILE"; then
          PYLINK=-l`basename $PYLINKFILE | sed -e 's/\.lib$//'`
        else
          PYLIB=
        fi
      fi
    else
      # Note: I could not think of a standard way to get the version string from different versions.
      # This trick pulls it out of the file location for a standard library file.

      AC_MSG_CHECKING(for Python version)

      # Need to do this hack since autoconf replaces __file__ with the name of the configure file
      filehack="file__"
      PYVERSION=`($PYTHON -c "import sys,string,operator,os.path; sys.stdout.write(operator.getitem(os.path.split(operator.getitem(os.path.split(string.__$filehack),0)),1))")`
      AC_MSG_RESULT($PYVERSION)

      # Find the directory for libraries this is necessary to deal with
      # platforms that can have apps built for multiple archs: e.g. x86_64
      AC_MSG_CHECKING(for Python lib dir)
      PYLIBDIR=`($PYTHON -c "import sys; sys.stdout.write(sys.lib)") 2>/dev/null`
      if test -z "$PYLIBDIR"; then
        # Fedora patch Python to add sys.lib, for other distros we assume "lib".
        PYLIBDIR="lib"
      fi
      AC_MSG_RESULT($PYLIBDIR)

      # Set the include directory

      AC_MSG_CHECKING(for Python header files)
      if test -r $PYPREFIX/include/$PYVERSION/Python.h; then
        PYINCLUDE="-I$PYPREFIX/include/$PYVERSION -I$PYEPREFIX/$PYLIBDIR/$PYVERSION/config"
      fi
      if test -z "$PYINCLUDE"; then
        if test -r $PYPREFIX/include/Py/Python.h; then
          PYINCLUDE="-I$PYPREFIX/include/Py -I$PYEPREFIX/$PYLIBDIR/python/lib"
        fi
      fi
      AC_MSG_RESULT($PYINCLUDE)

      # Set the library directory blindly.   This probably won't work with older versions
      AC_MSG_CHECKING(for Python library directory)
      dirs="$PYVERSION/config $PYVERSION/$PYLIBDIR python/$PYLIBDIR"
      for i in $dirs; do
        if test -d $PYEPREFIX/$PYLIBDIR/$i; then
          PYLIB="$PYEPREFIX/$PYLIBDIR/$i"
          break
        fi
      done

      PYLINK="-l$PYVERSION"
    fi

    if test -z "$PYLIB"; then
      AC_MSG_RESULT(Not found)
    else
      AC_MSG_RESULT($PYLIB)
    fi
    AC_MSG_CHECKING(for Python library)
    if test -z "$PYLINK"; then
      AC_MSG_RESULT(Not found)
    else
      AC_MSG_RESULT($PYLINK)
    fi
  fi

  # Cygwin (Windows) needs the library for dynamic linking
  case $host in
  *-*-cygwin* | *-*-mingw*)
    PYTHONDYNAMICLINKING="-L$PYLIB $PYLINK"
    DEFS="-DUSE_DL_IMPORT $DEFS"
    ;;
  *)PYTHONDYNAMICLINKING="";;
  esac
fi

AC_SUBST(PYINCLUDE)
AC_SUBST(PYLIB)
AC_SUBST(PYLINK)
AC_SUBST(PYTHONDYNAMICLINKING)


#----------------------------------------------------------------
# Look for Python 3.x
#----------------------------------------------------------------

PY3INCLUDE=
PY3LIB=
PY3LINK=
PY3PACKAGE=

AC_ARG_WITH(python3, AS_HELP_STRING([--without-python3], [Disable Python 3.x support])
AS_HELP_STRING([--with-python3=path], [Set location of Python 3.x executable]),[ PY3BIN="$withval"], [PY3BIN="$alllang_default"])

# First, check for "--without-python3" or "--with-python3=no".
if test x"${PY3BIN}" = xno; then
  AC_MSG_NOTICE([Disabling Python 3.x support])
else
  if test -z "$PYVER"; then
    PYVER=0
  fi
  if test "x$PY3BIN" = xyes; then
    if test x"$PYOSNAME" = x"nt" -a $PYVER -ge 3; then
      PYTHON3="$PYTHON"
    else
      for py_ver in 3 3.9 3.8 3.7 3.6 3.5 3.4 3.3 3.2 3.1 3.0 ""; do
        AC_CHECK_PROGS(PYTHON3, [python$py_ver])
        if test -n "$PYTHON3"; then
          AC_CHECK_PROGS(PY3CONFIG, [$PYTHON3-config])
          if test -n "$PY3CONFIG"; then
            break
          fi
        fi
      done
    fi
  else
    PYTHON3="$PY3BIN"
    AC_CHECK_PROGS(PY3CONFIG, [$PYTHON3-config])
  fi

  if test -n "$PYTHON3"; then
    AC_MSG_CHECKING([for $PYTHON3 major version number])
    PYVER=`($PYTHON3 -c "import sys; sys.stdout.write(sys.version[[0]])") 2>/dev/null`
    AC_MSG_RESULT($PYVER)
    if test -z "$PYVER"; then
      PYVER=0
    fi
  fi

  if test $PYVER -ge 3; then
    AC_MSG_CHECKING(for Python 3.x os.name)
    PY3OSNAME=`($PYTHON3 -c "import sys, os; sys.stdout.write(os.name)")`
    AC_MSG_RESULT($PY3OSNAME)

    if test x"$PY3OSNAME" = x"nt"; then
      # Windows installations are quite different to posix installations
      # There is no python-config to use
      AC_MSG_CHECKING(for Python 3.x prefix)
      PY3PREFIX=`($PYTHON3 -c "import sys; sys.stdout.write(sys.prefix)") 2>/dev/null`
      AC_MSG_RESULT($PY3PREFIX)
      PY3PREFIX=`echo "$PY3PREFIX" | sed -e 's,\\\\,/,g'` # Forward slashes are easier to use and even work on Windows most of the time
      PYTHON_SO=.pyd

      AC_MSG_CHECKING(for Python 3.x header files)
      if test -r $PY3PREFIX/include/Python.h; then
        PY3INCLUDE="-I$PY3PREFIX/include"
      fi
      AC_MSG_RESULT($PY3INCLUDE)

      AC_MSG_CHECKING(for Python 3.x library directory)
      if test -d $PY3PREFIX/libs; then
        PY3LIB=$PY3PREFIX/libs
        PY3LINKFILE=`ls $PY3LIB/python*.lib | grep "python[[0-9]][[0-9]]\.lib"`
        if test -r "$PY3LINKFILE"; then
          PY3LINK=-l`basename $PY3LINKFILE | sed -e 's/\.lib$//'`
        else
          PY3LIB=
        fi
      fi
      if test -z "$PY3LIB"; then
        AC_MSG_RESULT([Not found])
      else
        AC_MSG_RESULT($PY3LIB)
      fi
      AC_MSG_CHECKING([for Python 3.x library])
      if test -z "$PY3LINK"; then
        AC_MSG_RESULT(Not found)
      else
        AC_MSG_RESULT($PY3LINK)
      fi
    elif test -n "$PY3CONFIG"; then
      AC_MSG_CHECKING([for Python 3.x prefix])
      PY3PREFIX=`($PY3CONFIG --prefix) 2>/dev/null`
      AC_MSG_RESULT($PY3PREFIX)
      AC_MSG_CHECKING(for Python 3.x exec-prefix)
      PY3EPREFIX=`($PY3CONFIG --exec-prefix) 2>/dev/null`
      AC_MSG_RESULT($PY3EPREFIX)

      # Note: I could not think of a standard way to get the version string from different versions.
      # This trick pulls it out of the file location for a standard library file.

      AC_MSG_CHECKING([for Python 3.x version])

      # Need to do this hack since autoconf replaces __file__ with the name of the configure file
      filehack="file__"
      PY3VERSION=`($PYTHON3 -c "import string,operator,os.path; print(operator.getitem(os.path.split(operator.getitem(os.path.split(string.__$filehack),0)),1))")`
      AC_MSG_RESULT($PY3VERSION)

      # Find the directory for libraries this is necessary to deal with
      # platforms that can have apps built for multiple archs: e.g. x86_64
      AC_MSG_CHECKING([for Python 3.x lib dir])
      PY3LIBDIR=`($PYTHON3 -c "import sys; print(sys.lib)") 2>/dev/null`
      if test -z "$PY3LIBDIR"; then
        # some dists don't have sys.lib  so the best we can do is assume lib
        PY3LIBDIR="lib"
      fi
      AC_MSG_RESULT($PY3LIBDIR)

      # Set the include directory

      AC_MSG_CHECKING([for Python 3.x header files])
      PY3INCLUDE=`($PY3CONFIG --includes) 2>/dev/null`
      AC_MSG_RESULT($PY3INCLUDE)

      # Set the library directory blindly.   This probably won't work with older versions
      AC_MSG_CHECKING([for Python 3.x library directory])
      dirs="$PY3VERSION/config $PY3VERSION/$PY3LIBDIR python/$PY3LIBDIR"
      for i in $dirs; do
        if test -d $PY3EPREFIX/$PY3LIBDIR/$i; then
          PY3LIB="$PY3EPREFIX/$PY3LIBDIR/$i"
          break
        fi
      done
      if test -z "$PY3LIB"; then
        AC_MSG_RESULT([Not found])
      else
        AC_MSG_RESULT($PY3LIB)
      fi

      PY3LINK="-l$PY3VERSION"

      AC_MSG_CHECKING([for Python 3.x library])
      if test -z "$PY3LINK"; then
        AC_MSG_RESULT(Not found)
      else
        AC_MSG_RESULT($PY3LINK)
      fi
    fi
  fi

  # Cygwin (Windows) needs the library for dynamic linking
  case $host in
  *-*-cygwin* | *-*-mingw*)
    PYTHON3DYNAMICLINKING="-L$PY3LIB $PY3LINK"
    DEFS="-DUSE_DL_IMPORT $DEFS"
    ;;
  *)PYTHON3DYNAMICLINKING="";;
  esac

  AC_SUBST(PY3INCLUDE)
  AC_SUBST(PY3LIB)
  AC_SUBST(PY3LINK)
  AC_SUBST(PYTHON3DYNAMICLINKING)
fi

if test -n "$PYINCLUDE" || test -n "$PY3INCLUDE" ; then
  AC_CHECK_PROGS(PEP8, pep8)
  if test -n "$PEP8"; then
    AC_MSG_CHECKING(pep8 version)
    pep8_version=`$PEP8 --version 2>/dev/null`
    AC_MSG_RESULT($pep8_version)
  fi
fi

#----------------------------------------------------------------
# Look for Perl5
#----------------------------------------------------------------

PERLBIN=

AC_ARG_WITH(perl5, AS_HELP_STRING([--without-perl5], [Disable Perl5])
AS_HELP_STRING([--with-perl5=path], [Set location of Perl5 executable]),[ PERLBIN="$withval"], [PERLBIN="$alllang_default"])

# First, check for "--without-perl5" or "--with-perl5=no".
if test x"${PERLBIN}" = xno; then
AC_MSG_NOTICE([Disabling Perl5])
PERL=
else

# First figure out what the name of Perl5 is

if test "x$PERLBIN" = xyes; then
AC_CHECK_PROGS(PERL, perl perl5.6.1 perl5.6.0 perl5.004 perl5.003 perl5.002 perl5.001 perl5 perl)
else
PERL="$PERLBIN"
fi


# This could probably be simplified as for all platforms and all versions of Perl the following apparently should be run to get the compilation options:
# perl -MExtUtils::Embed -e ccopts
AC_MSG_CHECKING(for Perl5 header files)
if test -n "$PERL"; then
	PERL5DIR=`($PERL -MConfig -le 'print $Config{archlibexp}') 2>/dev/null`
	if test -n "$PERL5DIR" ; then
		dirs="$PERL5DIR $PERL5DIR/CORE"
		PERL5EXT=none
		for i in $dirs; do
			if test -r $i/perl.h; then
				AC_MSG_RESULT($i)
				PERL5EXT="$i"
				break
			fi
		done
		if test "$PERL5EXT" = none; then
			PERL5EXT="$PERL5DIR/CORE"
			AC_MSG_RESULT(could not locate perl.h...using $PERL5EXT)
		fi

		AC_MSG_CHECKING(for Perl5 library)
		PERL5LIB=`($PERL -e 'use Config; $_=$Config{libperl}; s/^lib//; s/$Config{_a}$//; s/\.$Config{so}.*//; print $_, "\n"') 2>/dev/null`
		if test -z "$PERL5LIB" ; then
			AC_MSG_RESULT(not found)
		else
			AC_MSG_RESULT($PERL5LIB)
		fi
    AC_MSG_CHECKING(for Perl5 ccflags)
 		PERL5CCFLAGS=`($PERL -e 'use Config; print $Config{ccflags}, "\n"' | sed "s/-Wdeclaration-after-statement//" | sed "s/-I/$ISYSTEM/") 2>/dev/null`
 		if test -z "$PERL5CCFLAGS" ; then
 			AC_MSG_RESULT(not found)
 		else
 			AC_MSG_RESULT($PERL5CCFLAGS)
 		fi
    AC_MSG_CHECKING(for Perl5 ccdlflags)
    PERL5CCDLFLAGS=`($PERL -e 'use Config; print $Config{ccdlflags}, "\n"') 2>/dev/null`
    if test -z "$PERL5CCDLFLAGS" ; then
      AC_MSG_RESULT(not found)
      else
      AC_MSG_RESULT($PERL5CCDLFLAGS)
    fi
    AC_MSG_CHECKING(for Perl5 cccdlflags)
    PERL5CCCDLFLAGS=`($PERL -e 'use Config; print $Config{cccdlflags}, "\n"') 2>/dev/null`
    if test -z "$PERL5CCCDLFLAGS" ; then
      AC_MSG_RESULT(not found)
      else
      AC_MSG_RESULT($PERL5CCCDLFLAGS)
    fi
    AC_MSG_CHECKING(for Perl5 ldflags)
    PERL5LDFLAGS=`($PERL -e 'use Config; print $Config{ldflags}, "\n"') 2>/dev/null`
    if test -z "$PERL5LDFLAGS" ; then
      AC_MSG_RESULT(not found)
      else
      AC_MSG_RESULT($PERL5LDFLAGS)
    fi
    AC_MSG_CHECKING(for Perl5 Test::More module) # For test-suite
    PERL5TESTMORE=`($PERL -e 'use Test::More; print "good";') 2>/dev/null`
    if test -z "$PERL5TESTMORE" ; then
      AC_MSG_RESULT(not found)
      else
      AC_MSG_RESULT(found)
    fi
	else
		AC_MSG_RESULT(unable to determine perl5 configuration)
		PERL5EXT=$PERL5DIR
	fi
else
       	AC_MSG_RESULT(could not figure out how to run perl5)
fi

# Cygwin (Windows) needs the library for dynamic linking
case $host in
*-*-cygwin* | *-*-mingw*) PERL5DYNAMICLINKING="-L$PERL5EXT -l$PERL5LIB";;
*)PERL5DYNAMICLINKING="";;
esac
fi

AC_SUBST(PERL)
AC_SUBST(PERL5EXT)
AC_SUBST(PERL5DYNAMICLINKING)
AC_SUBST(PERL5LIB)
AC_SUBST(PERL5CCFLAGS)
AC_SUBST(PERL5CCDLFLAGS)
AC_SUBST(PERL5CCCDLFLAGS)
AC_SUBST(PERL5LDFLAGS)

#----------------------------------------------------------------
# Look for Octave
#----------------------------------------------------------------

OCTAVEBIN=
OCTAVE_SO=.oct

AC_ARG_WITH(octave, AS_HELP_STRING([--without-octave], [Disable Octave])
AS_HELP_STRING([--with-octave=path], [Set location of Octave executable]),[OCTAVEBIN="$withval"], [OCTAVEBIN="$alllang_default"])

# Check for "--without-octave" or "--with-octave=no".
if test x"${OCTAVEBIN}" = xno; then
   AC_MSG_NOTICE([Disabling Octave])
   OCTAVE=

# Check for Octave; prefer command-line program "octave-cli" to (in newer versions) GUI program "octave"
elif test "x$OCTAVEBIN" = xyes; then
   AC_PATH_PROG(OCTAVE, [octave-cli octave])

else
   OCTAVE="$OCTAVEBIN"
fi

# Check if Octave works
if test -n "$OCTAVE"; then
   AC_MSG_CHECKING([if ${OCTAVE} works])
   AS_IF([test "x`${OCTAVE} --version 2>/dev/null | sed -n -e '1p' | sed -n -e '/Octave, version/p'`" != x],[
      AC_MSG_RESULT([yes])
   ],[
      AC_MSG_NOTICE([no, disabling Octave])
      OCTAVE=
   ])
fi

# Check for required Octave helper program "mkoctfile"
if test -n "$OCTAVE"; then
   AC_MSG_CHECKING([for mkoctfile])
   mkoctfile=["`echo $OCTAVE | sed -e 's|[a-z][a-z-]*$|mkoctfile|;t;s|[a-z][a-z-]*\(-[0-9][0-9.+]*\)$|mkoctfile\1|;t'`"]
   AC_MSG_RESULT([${mkoctfile}])
   AC_MSG_CHECKING([if ${mkoctfile} works])
   AS_IF([test "x`${mkoctfile} --version 2>/dev/null | sed -n -e '1p' | sed -n -e '/mkoctfile, version/p'`" != x],[
      AC_MSG_RESULT([yes])
   ],[
      AC_MSG_NOTICE([no, disabling Octave])
      OCTAVE=
   ])
fi

# Check for Octave preprocessor/compiler/linker flags
if test -n "$OCTAVE"; then

   AC_MSG_CHECKING([for Octave preprocessor flags])
   OCTAVE_CPPFLAGS=
   for var in CPPFLAGS INCFLAGS ALL_CXXFLAGS; do
      for flag in `env - ${mkoctfile} -p ${var}`; do
         case ${flag} in
            -D*|-I*) OCTAVE_CPPFLAGS="${OCTAVE_CPPFLAGS} ${flag}";;
            *) ;;
         esac
      done
   done
   AC_MSG_RESULT([$OCTAVE_CPPFLAGS])

   AC_MSG_CHECKING([for Octave compiler flags])
   OCTAVE_CXXFLAGS=
   for var in CXX ALL_CXXFLAGS; do
      for flag in `env - ${mkoctfile} -p ${var}`; do
         case ${flag} in
            -std=*|-g*|-W*) OCTAVE_CXXFLAGS="${OCTAVE_CXXFLAGS} ${flag}";;
            *) ;;
         esac
      done
   done
   save_CXXFLAGS="${CXXFLAGS}"
   CXXFLAGS="-Werror -O0"
   AC_COMPILE_IFELSE([
      AC_LANG_PROGRAM([],[])
   ],[
      OCTAVE_CXXFLAGS="${OCTAVE_CXXFLAGS} -O0"
   ])
   CXXFLAGS="${save_CXXFLAGS}"
   AC_MSG_RESULT([$OCTAVE_CXXFLAGS])

   AC_MSG_CHECKING([for Octave linker flags])
   OCTAVE_LDFLAGS=
   for var in RDYNAMIC_FLAG LFLAGS RLD_FLAG OCTAVE_LIBS LIBS; do
     OCTAVE_LDFLAGS="${OCTAVE_LDFLAGS} "`env - ${mkoctfile} -p ${var}`
   done
   AC_MSG_RESULT([$OCTAVE_LDFLAGS])

fi

# Check for Octave options
if test -n "$OCTAVE"; then
   for octave_opt in --no-window-system --silent --norc --no-history; do
      AC_MSG_CHECKING([if Octave option '${octave_opt}' is supported])
      octave_out=`${OCTAVE} ${octave_opt} /dev/null 2>&1 | sed -n '1p' | sed -n '/unrecognized/p'`
      AS_IF([test "x${octave_out}" = x],[
         AC_MSG_RESULT([yes])
         OCTAVE="${OCTAVE} ${octave_opt}"
      ],[
         AC_MSG_RESULT([no])
      ])
   done
fi

AC_SUBST(OCTAVE)
AC_SUBST(OCTAVE_SO)
AC_SUBST(OCTAVE_CPPFLAGS)
AC_SUBST(OCTAVE_CXXFLAGS)
AC_SUBST(OCTAVE_LDFLAGS)

#----------------------------------------------------------------
# Look for Scilab
#----------------------------------------------------------------

AC_ARG_WITH(scilab, AS_HELP_STRING([--without-scilab], [Disable Scilab])
AS_HELP_STRING([--with-scilab=path], [Set location of Scilab executable]),[SCILABBIN="$withval"], [SCILABBIN="$alllang_default"])
AC_ARG_WITH(scilab-inc, [  --with-scilab-inc=path  Set location of Scilab include directory], [SCILABINCDIR="$withval"], [SCILABINCDIR=""])

# First, check for "--without-scilab" or "--with-scilab=no".
if test x"${SCILABBIN}" = xno; then
  AC_MSG_NOTICE([Disabling Scilab])
  SCILAB=
else
  # Check for Scilab executable
  if test "x$SCILABBIN" = xyes; then
    AC_CHECK_PROGS(SCILAB, scilab)
  else
    AC_MSG_CHECKING(for scilab)
    if test -f "$SCILABBIN"; then
      AC_MSG_RESULT($SCILABBIN)
      SCILAB="$SCILABBIN"
    else
      AC_MSG_RESULT(not found)
    fi
  fi

  if test -n "$SCILAB"; then
    # Check for Scilab version (needs api_scilab so needs version 5.3.3 or higher)
    SCILAB_FULL_VERSION=`$SCILAB -version | head -1 | sed -e 's|Scilab version \"\(.*\)\"|\1|g'`

    AC_MSG_CHECKING(Scilab version is 5.3.3 or higher)
    SCILAB_MAJOR_VERSION=`echo $SCILAB_FULL_VERSION | cut -d. -f1`
    SCILAB_MINOR_VERSION=`echo $SCILAB_FULL_VERSION | cut -d. -f2`
    SCILAB_MAINTENANCE_VERSION=`echo $SCILAB_FULL_VERSION | cut -d. -f3`
    SCILAB_VERSION="$SCILAB_MAJOR_VERSION$SCILAB_MINOR_VERSION$SCILAB_MAINTENANCE_VERSION"

    if test $SCILAB_VERSION -ge 533; then
      AC_MSG_RESULT(yes)
    else
      AC_MSG_RESULT(no)
      SCILAB=
    fi

    if test -n "$SCILAB"; then
      # Set Scilab startup options depending on version
      AC_MSG_CHECKING(for Scilab startup options)
      SCILABOPT="-nwni -nb"
      if test $SCILAB_VERSION -ge 540; then
        SCILABOPT+=" -noatomsautoload"
      fi
      AC_MSG_RESULT($SCILABOPT)

      # Check for Scilab header files
      AC_MSG_CHECKING(for Scilab header files)
      if test "$SCILABINCDIR" != ""; then
        dirs="$SCILABINCDIR"
      elif test -n "$PKGCONFIG"; then
        dirs=`$PKGCONFIG scilab --cflags-only-I | sed -e 's/-I//g'`
      else
        dirs=""
      fi
      for i in $dirs; do
        if test -r $i/api_scilab.h; then
          AC_MSG_RESULT($i)
          SCILABINCLUDE="-I$i"
          break
        fi
        if test -r $i/scilab/api_scilab.h; then
          AC_MSG_RESULT($i/scilab)
          SCILABINCLUDE="-I$i/scilab"
          break
        fi
      done
      if test "$SCILABINCLUDE" = "" ; then
        AC_MSG_RESULT(not found)
        SCILAB=
      fi
    fi
  fi
fi

AC_SUBST(SCILAB)
AC_SUBST(SCILABINCLUDE)
AC_SUBST(SCILABOPT)


#----------------------------------------------------------------
# Look for java
#----------------------------------------------------------------

AC_ARG_WITH(java, AS_HELP_STRING([--without-java], [Disable Java])
AS_HELP_STRING([--with-java=path], [Set location of java executable]),[JAVABIN="$withval"], [JAVABIN="$alllang_default"])
AC_ARG_WITH(javac, [  --with-javac=path       Set location of javac executable],[JAVACBIN="$withval"], [JAVACBIN=])

# First, check for "--without-java" or "--with-java=no".
if test x"${JAVABIN}" = xno; then
AC_MSG_NOTICE([Disabling Java])
JAVA=
else

dnl Detect everything we need for Java: java for running Java tests, javac for
dnl compiling them and C headers for compiling C code.
dnl
dnl If JAVA_HOME is defined and contains everything needed, as it's supposed to,
dnl just use it. Otherwise try to locate the necessary files in the path.
case $host in
  *-*-cygwin*)
    dnl Ensure that JAVA_HOME is in the form we can use, it's typically set to Windows path by the installer.
    if test -n "$JAVA_HOME"; then
      JAVA_HOME=`cygpath --mixed "$JAVA_HOME"`
    fi
    ;;
  *-*-mingw*)
    if test -n "$JAVA_HOME"; then
      JAVA_HOME=`${srcdir}/Tools/convertpath -u "$JAVA_HOME"`
    fi
    ;;
  *-*-darwin*)
    dnl Under OS X JAVA_HOME is not set by default, try to use the system default JRE.
    if test -z "$JAVA_HOME" -a -x /usr/libexec/java_home; then
      JAVA_HOME=`/usr/libexec/java_home`
    fi
    dnl The JAVA_HOME doesn't contain the JDK headers though, but they seem to
    dnl always be in the same location, according to Apple JNI documentation.
    JAVA_OSX_STD_INCDIR="/System/Library/Frameworks/JavaVM.framework/Headers"
    if test -r "$JAVA_OSX_STD_INCDIR/jni.h"; then
      JAVA_HOME_INCDIR=$JAVA_OSX_STD_INCDIR
    fi
    ;;
esac

AC_MSG_CHECKING(for java JDK)
if test -n "$JAVA_HOME"; then
  dnl Don't complain about missing executables/headers if they had been
  dnl explicitly overridden from the command line, but otherwise verify that we
  dnl have everything we need.
  java_home_ok=1
  if test -z "$JAVABIN" -a ! -x "$JAVA_HOME/bin/java"; then
    AC_MSG_WARN([No java executable under JAVA_HOME.])
    java_home_ok=0
  fi
  if test -z "$JAVACBIN" -a ! -x "$JAVA_HOME/bin/javac"; then
    AC_MSG_WARN([No javac executable under JAVA_HOME.])
    java_home_ok=0
  fi

  dnl By default the headers are under JAVA_HOME/include, but OS X is special
  dnl and defines JAVA_HOME_INCDIR differently above.
  if test -z "$JAVA_HOME_INCDIR"; then
    JAVA_HOME_INCDIR="$JAVA_HOME/include"
  fi
  if test -z "$JAVAINCDIR" -a ! -r "$JAVA_HOME_INCDIR/jni.h"; then
    AC_MSG_WARN([No java headers under JAVA_HOME (does it point to a JDK and not just JRE?).])
    java_home_ok=0
  fi
  if test "$java_home_ok" = 1; then
    AC_MSG_RESULT([found (in $JAVA_HOME)])
  else
    AC_MSG_RESULT(no)
    AC_MSG_WARN([JAVA_HOME ($JAVA_HOME) is defined but does not point to a complete JDK installation, ignoring it.])
    JAVA_HOME=
  fi
else
  AC_MSG_RESULT([no (JAVA_HOME is not defined)])
fi

if test "x$JAVABIN" = xyes; then
  if test -n "$JAVA_HOME"; then
    JAVA="$JAVA_HOME/bin/java"
  else
    AC_CHECK_PROGS(JAVA, java kaffe guavac)
  fi
else
  JAVA="$JAVABIN"
fi

if test -z "$JAVACBIN"; then
  if test -n "$JAVA_HOME"; then
    JAVAC="$JAVA_HOME/bin/javac"
  else
    AC_CHECK_PROGS(JAVAC, javac)
  fi
else
  JAVAC="$JAVACBIN"
fi

AC_MSG_CHECKING(for java include file jni.h)
AC_ARG_WITH(javaincl, [  --with-javaincl=path    Set location of Java include directory], [JAVAINCDIR="$withval"], [JAVAINCDIR=])

JAVAINC=""
if test -z "$JAVAINCDIR" ; then
  if test -n "$JAVA_HOME"; then
    JAVAINCDIR="$JAVA_HOME_INCDIR"
    JAVAINC=-I\"$JAVAINCDIR\"
  else
    JAVAINCDIR="/usr/j2sdk*/include /usr/local/j2sdk*/include /usr/jdk*/include /usr/local/jdk*/include /opt/j2sdk*/include /opt/jdk*/include /usr/java/include /usr/java/j2sdk*/include /usr/java/jdk*/include /usr/local/java/include /opt/java/include /usr/include/java /usr/local/include/java /usr/lib/java/include /usr/lib/jvm/java*/include /usr/lib64/jvm/java*/include /usr/include/kaffe /usr/local/include/kaffe /usr/include"

    # Add in default installation directory on Windows for Cygwin
    case $host in
    *-*-cygwin* | *-*-mingw*) JAVAINCDIR="c:/Program*Files*/Java/jdk*/include d:/Program*Files*/Java/jdk*/include c:/j2sdk*/include d:/j2sdk*/include c:/jdk*/include d:/jdk*/include $JAVAINCDIR";;
    *-*-darwin*) JAVAINCDIR="$JAVA_OSX_STD_INCDIR $JAVAINCDIR";;
    *);;
    esac

    for d in $JAVAINCDIR ; do
      if test -r "$d/jni.h" ; then
	JAVAINCDIR=$d
	JAVAINC=-I\"$d\"
	break
      fi
    done
  fi
else
  if test -r "$JAVAINCDIR/jni.h" ; then
    JAVAINC=-I\"$JAVAINCDIR\"
  fi
fi

if test "$JAVAINC" = "" ; then
  AC_MSG_RESULT(not found)
else
  AC_MSG_RESULT($JAVAINC)
  # now look for <arch>/jni_md.h
  AC_MSG_CHECKING(for java include file jni_md.h)
  JAVAMDDIR=`find "$JAVAINCDIR" -follow -name jni_md.h -print`
  if test "$JAVAMDDIR" = "" ; then
    AC_MSG_RESULT(not found)
  else
    JAVAMDDIR=-I\"`dirname "$JAVAMDDIR" | tail -1`\"
    AC_MSG_RESULT($JAVAMDDIR)
    JAVAINC="${JAVAINC} ${JAVAMDDIR}"
  fi
fi

case $host in
*-*-cygwin*)
        # TODO: Only use this flag if the compiler supports it, later versions of gcc no longer have it
        GCC_MNO_CYGWIN=" -mno-cygwin"
        ;;
*)
        GCC_MNO_CYGWIN=""
        ;;
esac

# java.exe on Cygwin requires the Windows standard (Pascal) calling convention as it is a normal Windows executable and not a Cygwin built executable
case $host in
*-*-cygwin* | *-*-mingw*)
    if test "$GCC" = yes; then
        JAVADYNAMICLINKING="$GCC_MNO_CYGWIN -mthreads -Wl,--add-stdcall-alias"
        JAVACFLAGS="$GCC_MNO_CYGWIN -mthreads"
    else
        JAVADYNAMICLINKING=""
        JAVACFLAGS=""
    fi ;;
*-*-darwin*)
        JAVADYNAMICLINKING="-dynamiclib -framework JavaVM"
        JAVACFLAGS=""
        ;;
*)
        JAVADYNAMICLINKING=""
        JAVACFLAGS=""
        ;;
esac

# Java on Windows platforms including Cygwin doesn't use libname.dll, rather name.dll when loading dlls
case $host in
*-*-cygwin* | *-*-mingw*) JAVALIBRARYPREFIX="";;
*)JAVALIBRARYPREFIX="lib";;
esac

# Java on Mac OS X tweaks
case $host in
*-*-darwin*)
    JAVASO=".jnilib"
    JAVALDSHARED='$(CC)'
    JAVACXXSHARED='$(CXX)'
    ;;
*)
    JAVASO=$SO
    JAVALDSHARED='$(LDSHARED)'
    JAVACXXSHARED='$(CXXSHARED)'
    ;;
esac
fi

# Quote for spaces in path to executables
if test -n "$JAVA"; then
  JAVA=\"$JAVA\"
fi
if test -n "$JAVAC"; then
  JAVAC=\"$JAVAC\"
fi

# Turned off due to spurious warnings in later versions of openjdk-1.8
# JAVAFLAGS=-Xcheck:jni

AC_SUBST(JAVA)
AC_SUBST(JAVAC)
AC_SUBST(JAVAINC)
AC_SUBST(JAVADYNAMICLINKING)
AC_SUBST(JAVALIBRARYPREFIX)
AC_SUBST(JAVASO)
AC_SUBST(JAVALDSHARED)
AC_SUBST(JAVACXXSHARED)
AC_SUBST(JAVAFLAGS)
AC_SUBST(JAVACFLAGS)

#----------------------------------------------------------------
# Look for Javascript
#----------------------------------------------------------------
AC_ARG_WITH(javascript, AS_HELP_STRING([--without-javascript], [Disable Javascript]), [with_javascript="$withval"], [with_javascript="$alllang_default"])

# First, check for "--without-javascript" or "--with-javascript=no".
if test x"${with_javascript}" = xno; then
  AC_MSG_NOTICE([Disabling Javascript])
  JAVASCRIPT=
else
  JAVASCRIPT=1

  #----------------------------------------------------------------
  # General Javascript settings shared by JSC and V8
  #----------------------------------------------------------------

  case $host in
  *-*-darwin*)
    JSSO=".dylib"
    # HACK: didn't manage to get dynamic module loading working with a g++ compiled interpreter
    JSINTERPRETERCXX='c++'
    JSINTERPRETERLINKFLAGS='-g -Wl,-search_paths_first -Wl,-headerpad_max_install_names'
    ;;
  *)
    JSSO=$SO
    JSINTERPRETERCXX='$(CXX)'
    JSINTERPRETERLINKFLAGS='-ldl'
    ;;
  esac

  #----------------------------------------------------------------
  # Look for Node.js which is the default Javascript engine
  #----------------------------------------------------------------

  AC_CHECK_PROGS(NODEJS, [nodejs node])

  if test -n "$NODEJS"; then
    # node-gyp is needed to run the test-suite/examples
    AC_CHECK_PROGS(NODEGYP, node-gyp)
    if test -z "$NODEGYP"; then
      NODEJS=
    fi
  fi

  #----------------------------------------------------------------
  # Look for JavascriptCore (Webkit) settings (JSCOREINCDIR, JSCOREDYNAMICLINKING)
  #----------------------------------------------------------------

  # check for include files
  AC_MSG_CHECKING(for JavaScriptCore/JavaScript.h)
  AC_ARG_WITH(jscoreinc, [  --with-jscinc=path      Set location of Javascript include directory], [JSCOREINCDIR="$withval"], [JSCOREINCDIR=])

  JSCOREVERSION=

  if test -z "$JSCOREINCDIR"; then
    JSCOREINCDIR="/usr/include/ /usr/local/include/"

    # Add in default directory for JavaScriptCore headers for Linux and Mac OS X
    case $host in
    *-*-linux*)
      JSCOREINCDIR="/usr/include/webkit-1.0/ /usr/include/webkitgtk-1.0/ /usr/local/include/webkit-1.0/JavaScriptCore/ $JSCOREINCDIR"
      ;;
    *-*-darwin*)
      JSCOREINCDIR="/System/Library/Frameworks/JavaScriptCore.framework/Headers/ $JSCOREINCDIR"
      ;;
    *)
      ;;
    esac
  fi

  for d in $JSCOREINCDIR ; do
    if test -r "$d/JavaScriptCore/JavaScript.h" || test -r "$d/JavaScript.h" ; then
      AC_MSG_RESULT($d)
      JSCOREINCDIR=$d
      JSCOREINC=-I\"$d\"
      break
    fi
  done

  if test "$JSCOREINC" = "" ; then
    AC_MSG_RESULT(not found)
  fi

  # check for JavaScriptCore/Webkit libraries
  AC_ARG_WITH(jscorelib,[  --with-jsclib=path      Set location of the JavaScriptCore/Webkit library directory],[JSCORELIB="-L$withval"], [JSCORELIB=])

  if test -z "$JSCORELIB" -a -n "$PKGCONFIG"; then
    AC_MSG_CHECKING(for JavaScriptCore/Webkit library)
    if pkg-config javascriptcoregtk-1.0; then
      JSCORELIB=`$PKGCONFIG --libs javascriptcoregtk-1.0`
      JSCOREVERSION=`$PKGCONFIG --modversion javascriptcoregtk-1.0`
    fi
    if test -z "$JSCORELIB"; then
      AC_MSG_RESULT(not found)
      JSCENABLED=
    else
      AC_MSG_RESULT([$JSCORELIB])
      JSCOREDYNAMICLINKING="$JSCORELIB"
      JSCENABLED=1
    fi
  fi

  #----------------------------------------------------------------
  # Look for V8 settings (JSV8INCDIR, JSV8DYNAMICLINKING)
  #----------------------------------------------------------------

  # check for include files
  AC_MSG_CHECKING(for V8 Javascript v8.h)
  AC_ARG_WITH(jsv8inc, [  --with-jsv8inc=path     Set location of Javascript v8 include directory], [JSV8INCDIR="$withval"])

  # if not include dir is specified we try to find
  if test -z "$JSV8INCDIR"; then
    # Add in default directory for JavaScriptCore headers for Linux and Mac OS X
    case $host in
    *-*-linux*)
      JSV8INCDIR="/usr/include /usr/local/include/ $JSV8INCDIR"
      ;;
    *-*-darwin*)
      JSV8INCDIR="$JSV8INCDIR"
      ;;
    *)
      ;;
    esac
  fi

  for d in $JSV8INCDIR ; do
    if test -r "$d/v8.h" ; then
      JSV8INCDIR=$d
      JSV8INC=-I\"$d\"
      break
    fi
  done

  if test "$JSV8INC" = "" ; then
    AC_MSG_RESULT(not found)
  else
    AC_MSG_RESULT($JSV8INCDIR)
  fi

  # check for V8 library
  AC_MSG_CHECKING(for V8 Javascript library)
  AC_ARG_WITH(jsv8lib,[  --with-jsv8lib=path     Set location of V8 Javascript library directory],[JSV8LIBDIR="$withval"], [JSV8LIB=])

  v8libdirs="$JSV8LIBDIR /usr/lib64/ /usr/local/lib64/ /usr/lib/ /usr/local/lib/"
  for d in $v8libdirs ; do
    if test -r $d/libv8$JSSO; then
      JSV8LIBDIR=$d
      JSV8LIB="-L$d -lv8"
      break
    fi
  done

  if test "$JSV8LIB" = "" ; then
    AC_MSG_RESULT(not found)
    JSV8ENABLED=
  else
    AC_MSG_RESULT($JSV8LIBDIR)
    JSV8ENABLED=1
  fi


  # linking options
  case $host in
  *-*-darwin*)
    JSV8DYNAMICLINKING="$JSV8LIB"
    ;;
  *-*-linux*)
    JSV8DYNAMICLINKING="$JSV8LIB"
    ;;
  *)
    JSV8DYNAMICLINKING=""
    ;;
  esac

fi

AC_SUBST(JSINTERPRETERCXX)
AC_SUBST(JSINTERPRETERLINKFLAGS)

AC_SUBST(JSCOREINC)
AC_SUBST(JSCOREDYNAMICLINKING)
AC_SUBST(JSCOREVERSION)
AC_SUBST(JSV8INC)
AC_SUBST(JSV8DYNAMICLINKING)

AC_SUBST(JSCENABLED)
AC_SUBST(JSV8ENABLED)

AC_SUBST(NODEJS)
AC_SUBST(NODEGYP)

#----------------------------------------------------------------
# Look for gcj
#----------------------------------------------------------------

AC_ARG_WITH(gcj, AS_HELP_STRING([--without-gcj], [Disable GCJ])
AS_HELP_STRING([--with-gcj=path], [Set location of gcj executable]),[GCJBIN="$withval"], [GCJBIN="$alllang_default"])
AC_ARG_WITH(gcjh, [  --with-gcjh=path        Set location of gcjh executable],[GCJHBIN="$withval"], [GCJHBIN=])

# First, check for "--without-gcj" or "--with-gcj=no".
if test x"${GCJBIN}" = xno; then
  AC_MSG_NOTICE([Disabling GCJ])
else
  if test "x$GCJBIN" = xyes; then
    AC_CHECK_PROGS(GCJ, gcj)
  else
    GCJ="$GCJBIN"
  fi

  if test -z "$GCJCBIN"; then
    AC_CHECK_PROGS(GCJH, gcjh)
  else
    GCJH="$GCJHBIN"
  fi
fi

AC_SUBST(GCJ)
AC_SUBST(GCJH)

#----------------------------------------------------------------
# Look for Android
#----------------------------------------------------------------

AC_ARG_WITH(android, AS_HELP_STRING([--without-android], [Disable Android])
AS_HELP_STRING([--with-android=path], [Set location of android executable]),[ANDROIDBIN="$withval"], [ANDROIDBIN="$alllang_default"])
AC_ARG_WITH(adb, [  --with-adb=path         Set location of adb executable - Android Debug Bridge],[ADBBIN="$withval"], [ADBBIN=])
AC_ARG_WITH(ant, [  --with-ant=path         Set location of ant executable for Android],[ANTBIN="$withval"], [ANTBIN=])
AC_ARG_WITH(ndk-build, [  --with-ndk-build=path   Set location of Android ndk-build executable],[NDKBUILDBIN="$withval"], [NDKBUILDBIN=])

# First, check for "--without-android" or "--with-android=no".
if test x"${ANDROIDBIN}" = xno; then
  AC_MSG_NOTICE([Disabling Android])
  ANDROID=
else
  if test "x$ANDROIDBIN" = xyes; then
    AC_CHECK_PROGS(ANDROID, android)
  else
    ANDROID="$ANDROIDBIN"
  fi

  if test -z "$ADBBIN"; then
    AC_CHECK_PROGS(ADB, adb)
  else
    ADB="$ADBBIN"
  fi

  if test -z "$ANTBIN"; then
    AC_CHECK_PROGS(ANT, ant)
  else
    ANT="$ANTBIN"
  fi

  if test -z "$NDKBUILDBIN"; then
    AC_CHECK_PROGS(NDKBUILD, ndk-build)
  else
    NDKBUILD="$NDKBUILDBIN"
  fi
fi

AC_SUBST(ANDROID)
AC_SUBST(ADB)
AC_SUBST(ANT)
AC_SUBST(NDKBUILD)

#----------------------------------------------------------------
# Look for Guile
#----------------------------------------------------------------

GUILE=
GUILE_CFLAGS=
GUILE_LIBS=

AC_ARG_WITH(guile-config, AS_HELP_STRING([--without-guile], [Disable Guile])
AS_HELP_STRING([--with-guile-config=path], [Set location of guile-config]),[ GUILE_CONFIG="$withval"], [GUILE_CONFIG=])
AC_ARG_WITH(guile,[  --with-guile=path       Set location of Guile executable],[
	GUILE="$withval"], [GUILE="$alllang_default"])
AC_ARG_WITH(guile-cflags,[  --with-guile-cflags=cflags   Set cflags required to compile against Guile],[
	GUILE_CFLAGS="$withval"])
AC_ARG_WITH(guile-libs,[  --with-guile-libs=ldflags    Set ldflags needed to link with Guile],[
	GUILE_LIBS="$withval"])

# First, check for "--without-guile" or "--with-guile=no".
if test x"${GUILE}" = xno; then
  AC_MSG_NOTICE([Disabling Guile])
else
  if test -z "$GUILE_CONFIG" ; then
    AC_PATH_PROG(GUILE_CONFIG, guile-config)
  fi
  if test -n "$GUILE_CONFIG" ; then
    if test x"$GUILE" = xyes; then
      AC_MSG_CHECKING([for guile bindir])
      guile_bindir="`$GUILE_CONFIG info bindir`"
      AC_MSG_RESULT([$guile_bindir])
      GUILE=$guile_bindir/guile
      if ! test -f "$GUILE" ; then
	GUILE=
        AC_PATH_PROG(GUILE, guile)
      fi
    fi

    if test -f "$GUILE" ; then
      AC_MSG_CHECKING([for guile version])
      guile_version=`$GUILE -c '(display (effective-version))'`
      AC_MSG_RESULT([$guile_version])
      AC_MSG_CHECKING([for guile version >= 1.8])
      guile_good_version=`$GUILE -c '(if (>= (string->number (effective-version)) 1.8) (display "yes") (display "no"))'`
      AC_MSG_RESULT([$guile_good_version])
      if test x"$guile_good_version" != xyes ; then
        GUILE=
      fi
    fi

    if test -z "$GUILE_CFLAGS" ; then
      AC_MSG_CHECKING([for guile compile flags])
      GUILE_CFLAGS="`$GUILE_CONFIG compile`" # Note that this can sometimes be empty
      AC_MSG_RESULT([$GUILE_CFLAGS])
    fi

    if test -z "$GUILE_LIBS" ; then
      AC_MSG_CHECKING([for guile link flags])
      GUILE_LIBS="`$GUILE_CONFIG link`"
      AC_MSG_RESULT([$GUILE_LIBS])
    fi
  fi
fi

AC_SUBST(GUILE)
AC_SUBST(GUILE_CFLAGS)
AC_SUBST(GUILE_LIBS)

#----------------------------------------------------------------
# Look for MzScheme
#----------------------------------------------------------------

AC_ARG_WITH(mzscheme, AS_HELP_STRING([--without-mzscheme], [Disable MzScheme])
AS_HELP_STRING([--with-mzscheme=path], [Set location of MzScheme executable]),[ MZSCHEMEBIN="$withval"], [MZSCHEMEBIN="$alllang_default"])
AC_ARG_WITH(mzc, AS_HELP_STRING([--with-mzc=path], [Set location of MzScheme's mzc]), [ MZCBIN="$withval"], [MZCBIN=])

# First, check for "--without-mzscheme" or "--with-mzscheme=no".
if test x"${MZSCHEMEBIN}" = xno; then
  AC_MSG_NOTICE([Disabling MzScheme])
  MZC=
else
  if test "x$MZSCHEMEBIN" = xyes; then
     AC_PATH_PROG(MZSCHEME, mzscheme)
  else
     MZSCHEME="$MZSCHEMEBIN"
  fi

  if test -z "$MZCBIN"; then
     AC_PATH_PROG(MZC, mzc)
  fi

  if test -n "$MZSCHEME"; then
    AC_MSG_CHECKING(for MzScheme dynext object)
    MZDYNOBJ=`$MZSCHEME --eval '(begin (require dynext/link) (with-handlers (((lambda args #t) (lambda args #f))) (for-each (lambda (x) (printf "~a" x)) (expand-for-link-variant (current-standard-link-libraries)))))' 2>/dev/null`
    if test -f "$MZDYNOBJ"; then
      :
    else
      # older versions (3.72 approx and earlier)
      MZDYNOBJ=`$MZSCHEME --mute-banner --version --eval '(begin (require (lib "link.ss" "dynext")) (with-handlers (((lambda args #t) (lambda args #f))) (for-each (lambda (x) (display x) (display " ")) ((current-make-standard-link-libraries)))) (with-handlers (((lambda args #t) (lambda args #f))) (for-each (lambda (x) (display x)) (expand-for-link-variant (current-standard-link-libraries)))))' 2>/dev/null`
    fi
    if test -f "$MZDYNOBJ"; then
      AC_MSG_RESULT($MZDYNOBJ)
    else
      AC_MSG_RESULT(not found)
      MZDYNOBJ=""
    fi
  fi
fi
AC_SUBST(MZDYNOBJ)

#----------------------------------------------------------------
# Look for Ruby
#----------------------------------------------------------------

RUBYBIN=

AC_ARG_WITH(ruby, AS_HELP_STRING([--without-ruby], [Disable Ruby])
AS_HELP_STRING([--with-ruby=path], [Set location of Ruby executable]),[ RUBYBIN="$withval"], [RUBYBIN="$alllang_default"])

# First, check for "--without-ruby" or "--with-ruby=no".
RUBYSO=$SO
if test x"${RUBYBIN}" = xno; then
AC_MSG_NOTICE([Disabling Ruby])
RUBY=
else

# First figure out what the name of Ruby is

if test "x$RUBYBIN" = xyes; then
	AC_CHECK_PROGS(RUBY, ruby)
else
	RUBY="$RUBYBIN"
fi

AC_MSG_CHECKING(for Ruby header files)
if test -n "$RUBY"; then
        # Try Ruby1.9+ first
        RUBYDIR=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["rubyhdrdir"]] || $rubyhdrdir') 2>/dev/null`
        RUBYARCHHDRDIR=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["rubyarchhdrdir"]] || $rubyarchhdrdir') 2>/dev/null`
	if test x"$RUBYDIR" = x"" || test x"$RUBYDIR" = x"nil"; then
		RUBYDIR=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["archdir"]] || $archdir') 2>/dev/null`
        else
                RUBYARCH=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["arch"]] || $arch') 2>/dev/null`
        fi
	if test x"$RUBYDIR" != x""; then
		dirs="$RUBYDIR"
		RUBYINCLUDE=
		for i in $dirs; do
			if test -r $i/ruby.h; then
				if test x"$RUBYARCH" = x""; then
					RUBYINCLUDE="-I$i"
				elif test -n "$RUBYARCHHDRDIR"; then
					RUBYINCLUDE="-I$i -I$RUBYARCHHDRDIR"
				else
					RUBYINCLUDE="-I$i -I$i/$RUBYARCH"
				fi
				AC_MSG_RESULT($RUBYINCLUDE)
				break
			fi
		done
		if test x"$RUBYINCLUDE" = x""; then
			AC_MSG_RESULT(could not locate ruby.h)
		fi

		# Find library and path for linking.
		AC_MSG_CHECKING(for Ruby library)
		RUBYLIB=""
		rb_archlibdir=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["archlibdir"]]') 2>/dev/null`
		rb_libdir=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["libdir"]]') 2>/dev/null`
		rb_bindir=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["bindir"]]') 2>/dev/null`
		dirs="$dirs $rb_archlibdir $rb_libdir $rb_bindir"

        rb_libruby=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["LIBRUBY_A"]]') 2>/dev/null`
        RUBYLINK=`($RUBY -rrbconfig -e '
            c = RbConfig::CONFIG
            if c.has_key? "LIBRUBYARG_STATIC" # 1.8.x
                if c[["LIBRUBY"]] == c[["LIBRUBY_A"]]
                    link = c[["LIBRUBYARG_STATIC"]]
                else
                    link = c[["LIBRUBYARG_SHARED"]]
                end
            else # 1.6.x
                link = "-l" + c[["RUBY_INSTALL_NAME"]]
            end

            # Get the target Ruby was built for
            target = c[["target"]]

            if target == "i386-pc-mswin32"
              # Need to change msvcrt-ruby*.lib to -lmsvcrt-ruby*
              ext = File.extname(link)
              # Get index that counts backwards from end of string
              index = -1 - ext.size
              # Strip off the extension
              link = link.slice(0..index)
              puts "-l#{link}"
            else
              puts link
            end') 2>/dev/null`

		if test "$rb_libruby" != ""; then
			for i in $dirs; do
				if (test -r $i/$rb_libruby;) then
					RUBYLIB="$i"
					break
				fi
			done
		fi
		if test "$RUBYLIB" = ""; then
			RUBYLIB="$RUBYDIR"
			AC_MSG_RESULT(not found... using $RUBYDIR)
		else
			AC_MSG_RESULT($RUBYLINK in $RUBYLIB)
		fi
	else
		AC_MSG_RESULT(unable to determine ruby configuration)
	fi

	case $host in
		*-*-mingw*) ;; # do nothing, the default windows libraries are already included
		*) RUBYLINK="$RUBYLINK `($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["LIBS"]]') 2>/dev/null`";;
	esac

	RUBYCCDLFLAGS=`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["CCDLFLAGS"]]') 2>/dev/null`
	RUBYSO=.`($RUBY -rrbconfig -e 'print RbConfig::CONFIG[["DLEXT"]]') 2>/dev/null`
else
	AC_MSG_RESULT(could not figure out how to run ruby)
fi

case $host in
*-*-cygwin* | *-*-mingw*)	RUBYDYNAMICLINKING="-L$RUBYLIB $RUBYLINK";;
*)		RUBYDYNAMICLINKING="";;
esac
fi

AC_SUBST(RUBYINCLUDE)
AC_SUBST(RUBYLIB)
AC_SUBST(RUBYLINK)
AC_SUBST(RUBYCCDLFLAGS)
AC_SUBST(RUBYSO)
AC_SUBST(RUBYDYNAMICLINKING)

#-------------------------------------------------------------------------
# Look for PHP5
#-------------------------------------------------------------------------

PHP5BIN=

AC_ARG_WITH(php5, AS_HELP_STRING([--without-php5], [Disable PHP5])
AS_HELP_STRING([--with-php5=path], [Set location of PHP5 executable]),[ PHP5BIN="$withval"], [PHP5BIN="$alllang_default"])

# First, check for "--without-php5" or "--with-php5=no".
if test x"${PHP5BIN}" = xno; then
    AC_MSG_NOTICE([Disabling PHP5])
    PHP5=
else
    if test "x$PHP5BIN" = xyes; then
      AC_CHECK_PROGS(PHP5, [php5 php])
    else
      PHP5=$PHP5BIN
    fi

    if test -n "$PHP5"; then
      AC_MSG_CHECKING(for PHP5 header files)
      dnl /usr/bin/php5 -> /usr/bin/php-config5
      case $PHP5 in
        *5)
          PHP5CONFIG=`echo "$PHP5"|sed 's/5$/-config5/'` ;;
        *)
          PHP5CONFIG=$PHP5-config ;;
      esac
      php5_version=`$PHP5CONFIG --version 2>/dev/null`
      case $php5_version in
      5*)
        PHP5INC=`$PHP5CONFIG --includes 2>/dev/null`
        if test -n "$PHP5INC"; then
          AC_MSG_RESULT($PHP5INC)
        else
          AC_MSG_RESULT(not found)
        fi
        ;;
      "")
        AC_MSG_RESULT([could not find $PHP5CONFIG or obtain PHP5 version from it]) ;;
      *)
        AC_MSG_RESULT([found PHP $php_version - not PHP 5]) ;;
      esac
    fi
fi
AC_SUBST(PHP5)
AC_SUBST(PHP5INC)

#-------------------------------------------------------------------------
# Look for PHP7
#-------------------------------------------------------------------------

PHPBIN=

AC_ARG_WITH(php, AS_HELP_STRING([--without-php], [Disable PHP])
AS_HELP_STRING([--with-php=path], [Set location of PHP executable]),[ PHPBIN="$withval"], [PHPBIN="$alllang_default"])

# First, check for "--without-php" or "--with-php=no".
if test x"${PHPBIN}" = xno; then
    AC_MSG_NOTICE([Disabling PHP])
    PHP=
else
    if test "x$PHPBIN" = xyes; then
      AC_CHECK_PROGS(PHP, [php7.1 php7.0 php])
    else
      PHP=$PHPBIN
    fi

    if test -n "$PHP"; then
      AC_MSG_CHECKING(for PHP header files)
      dnl /usr/bin/php7.0 -> /usr/bin/php-config7.0
      case $PHP in
        *7.*)
          PHPCONFIG=`echo "$PHP"|sed 's/7\...*$/-config&/'` ;;
        *)
          PHPCONFIG=$PHP-config ;;
      esac
      php_version=`$PHPCONFIG --version 2>/dev/null`
      case $php_version in
      7.*)
        PHPINC=`$PHPCONFIG --includes 2>/dev/null`
        if test -n "$PHPINC"; then
          AC_MSG_RESULT($PHPINC)
        else
          AC_MSG_RESULT(not found)
        fi
        ;;
      "")
        AC_MSG_RESULT([could not find $PHPCONFIG or obtain PHP version from it]) ;;
      *)
        AC_MSG_RESULT([found PHP $php_version - not PHP 7]) ;;
      esac
    fi
fi
AC_SUBST(PHP)
AC_SUBST(PHPINC)

#----------------------------------------------------------------
# Look for ocaml
#----------------------------------------------------------------

AC_ARG_WITH(ocaml, AS_HELP_STRING([--without-ocaml], [Disable OCaml]), [with_ocaml="$withval"], [with_ocaml="$alllang_default"])
AC_ARG_WITH(ocamlc,[  --with-ocamlc=path      Set location of ocamlc executable],[ OCAMLC="$withval"], [OCAMLC=])
AC_ARG_WITH(ocamldlgen,[  --with-ocamldlgen=path  Set location of ocamldlgen],[ OCAMLDLGEN="$withval" ], [OCAMLDLGEN=])
AC_ARG_WITH(ocamlfind,[  --with-ocamlfind=path   Set location of ocamlfind],[OCAMLFIND="$withval"],[OCAMLFIND=])
AC_ARG_WITH(ocamlmktop,[  --with-ocamlmktop=path  Set location of ocamlmktop executable],[ OCAMLMKTOP="$withval"], [OCAMLMKTOP=])
AC_ARG_WITH(camlp4,[  --with-camlp4=path  Set location of camlp4 executable],[ CAMLP4="$withval"], [CAMLP4=])

# First, check for "--without-ocaml" or "--with-ocaml=no".
if test x"${with_ocaml}" = xno; then
    AC_MSG_NOTICE([Disabling OCaml])
    OCAMLC=
else
    AC_MSG_CHECKING(for Ocaml DL load generator)
    if test -z "$OCAMLDLGEN"; then
	AC_CHECK_PROGS(OCAMLDLGEN, ocamldlgen)
    fi

    AC_MSG_CHECKING(for Ocaml package tool)
	if test -z "$OCAMLFIND"; then
	AC_CHECK_PROGS(OCAMLFIND, ocamlfind)
    fi

    AC_MSG_CHECKING(for Ocaml compiler)
	if test -z "$OCAMLC"; then
	AC_CHECK_PROGS(OCAMLC, ocamlc)
    fi

    AC_MSG_CHECKING(for Ocaml toplevel creator)
    if test -z "$OCAMLMKTOP"; then
	AC_CHECK_PROGS(OCAMLMKTOP, ocamlmktop)
    fi

    AC_MSG_CHECKING(for Ocaml Pre-Processor-Pretty-Printer)
    if test -z "$CAMLP4"; then
	AC_CHECK_PROGS(CAMLP4, camlp4)
    fi
fi # Disabling ocaml

AC_SUBST(OCAMLC)
AC_SUBST(OCAMLDLGEN)
AC_SUBST(OCAMLFIND)
AC_SUBST(OCAMLMKTOP)
AC_SUBST(CAMLP4)

#----------------------------------------------------------------
# Look for Pike
#----------------------------------------------------------------

# Identify the name of the Pike executable
# Priority: configure option, automatic search
PIKEBIN=
AC_ARG_WITH(pike, AS_HELP_STRING([--without-pike], [Disable Pike])
AS_HELP_STRING([--with-pike=path], [Set location of Pike executable]),[PIKEBIN="$withval"], [PIKEBIN="$alllang_default"])

# First, check for "--without-pike" or "--with-pike=no".
if test x"${PIKEBIN}" = xno; then
    AC_MSG_NOTICE([Disabling Pike])
    PIKEBIN=
else

if test "x$PIKEBIN" = xyes; then
	AC_CHECK_PROGS(PIKE, pike pike7.8 pike7.6 pike7.4 pike7.2)
else
	PIKE="$PIKEBIN"
fi


# Check for pike-config
# Priority: configure option, guessed from $PIKE, search from list
AC_ARG_WITH(pike-config, AS_HELP_STRING([--with-pike-config=path],
  	[Set location of pike-config script]),
  [PIKECONFIG="$withval"], [PIKECONFIG=""])

if test -z "$PIKECONFIG" -a -n "$PIKE"; then
	AC_CHECK_PROGS(PIKECONFIG, $PIKE-config pike-config \
		pike7.6-config pike7.4-config pike7.2-config)
fi

# Check for a --with-pikeincl option to configure
# Priority: configure option, info from $PIKECONFIG, guessed by pike script
AC_ARG_WITH(pikeincl, AS_HELP_STRING([--with-pikeincl=path],
	[Set location of Pike include directory]),
  [PIKEINCLUDE="-I$withval"], [PIKEINCLUDE=])

if test -n "$PIKE"; then
      AC_MSG_CHECKING([for Pike header files])
      if test -z "$PIKEINCLUDE" -a -n "$PIKECONFIG"; then
        PIKEINCLUDE=`$PIKECONFIG --cflags`
      fi
      if test -z "$PIKEINCLUDE" -a -n "$PIKE"; then
        PIKEINCLUDE=`$PIKE -x cflags`
        if test -z "$PIKEINCLUDE"; then
          PIKEPATH=`which $PIKE`
          PIKEINCLUDE=`$PIKE Tools/check-include-path.pike $PIKEPATH`
          PIKEINCLUDE="-I$PIKEINCLUDE"
        fi
      fi

      if test -z "$PIKEINCLUDE"; then
        AC_MSG_RESULT(not found)
      else
        AC_MSG_RESULT($PIKEINCLUDE)
      fi
fi
fi

AC_SUBST(PIKEINCLUDE)
AC_SUBST(PIKECCDLFLAGS)		dnl XXX: where is this used/defined?
AC_SUBST(PIKEDYNAMICLINKING)	dnl XXX: where is this used/defined?

#----------------------------------------------------------------
# Look for CHICKEN
#----------------------------------------------------------------

CHICKEN=
CHICKEN_CONFIG=
CHICKENHOME=
CHICKENOPTS=
CHICKENLIB=


AC_ARG_WITH(chicken, AS_HELP_STRING([--without-chicken], [Disable CHICKEN])
AS_HELP_STRING([--with-chicken=path], [Set location of CHICKEN executable]),[ CHICKENBIN="$withval"], [CHICKENBIN="$alllang_default"])

# First, check for "--without-chicken" or "--with-chicken=no".
if test x"${CHICKENBIN}" = xno; then
AC_MSG_NOTICE([Disabling CHICKEN])
else

if test "x$CHICKENBIN" = xyes; then
AC_CHECK_PROGS(CHICKEN, chicken)
else
CHICKEN="$CHICKENBIN"
fi

AC_ARG_WITH(chickencsc,[  --with-chickencsc=path  Set location of csc executable],[ CHICKEN_CSC="$withval"], [CHICKEN_CSC=])

if test -z "$CHICKEN_CSC"; then
  AC_CHECK_PROGS(CHICKEN_CSC, csc)
  # Both the Microsoft C# compiler and chicken have an executable called csc, so check that this csc is really the chicken one
  if test -n "$CHICKEN_CSC" ; then
    AC_MSG_CHECKING(whether csc is the chicken compiler)
    $CHICKEN_CSC -version 2>/dev/null | grep "chicken" > /dev/null || CHICKEN_CSC=""
    if test -z "$CHICKEN_CSC"; then
      AC_MSG_RESULT(no)
    else
      AC_MSG_RESULT(yes)
    fi
  fi
fi

AC_ARG_WITH(chickencsi,[  --with-chickencsi=path  Set location of csi executable],[ CHICKEN_CSI="$withval"], [CHICKEN_CSI=])

if test -z "$CHICKEN_CSI"; then
AC_CHECK_PROGS(CHICKEN_CSI, csi)
fi

if test -n "$CHICKEN_CSC" ; then

  AC_ARG_WITH(chickenopts,[  --with-chickenopts=args Set compiler options for static CHICKEN generated code],[
	CHICKENOPTS="$withval"], [CHICKENOPTS=])
  AC_ARG_WITH(chickensharedlib,[  --with-chickensharedlib=args    Set linker options for shared CHICKEN generated code],[
	CHICKENSHAREDLIB="$withval"], [CHICKENSHAREDLIB=])
  AC_ARG_WITH(chickenlib,[  --with-chickenlib=args  Set linker options for static CHICKEN generated code],[
	CHICKENLIB="$withval"], [CHICKENLIB=])

  AC_MSG_CHECKING(for compiler options for static CHICKEN generated code)
  if test -z "$CHICKENOPTS"; then
        CHICKENOPTS="`$CHICKEN_CSC -cflags`"
  else
        CHICKENOPTS="`$CHICKEN_CSC -cflags` $CHICKENOPTS"
  fi
  if test -z "$CHICKENOPTS"; then
        AC_MSG_RESULT(not found)
  else
        AC_MSG_RESULT($CHICKENOPTS)
  fi

  AC_MSG_CHECKING(for linker options for shared CHICKEN generated code)
  if test -z "$CHICKENSHAREDLIB"; then
        CHICKENSHAREDLIB="`$CHICKEN_CSC -shared -libs`"
  else
        CHICKENSHAREDLIB="`$CHICKEN_CSC -shared -libs` $CHICKENSHAREDLIB"
  fi
  if test -z "$CHICKENSHAREDLIB"; then
  	AC_MSG_RESULT(not found)
  else
        AC_MSG_RESULT($CHICKENSHAREDLIB)
  fi

  AC_MSG_CHECKING(for linker options for static CHICKEN generated code)
  if test -z "$CHICKENLIB"; then
        CHICKENLIB="`$CHICKEN_CSC -libs`"
  else
        CHICKENLIB="`$CHICKEN_CSC -libs` $CHICKENLIB"
  fi
  if test -z "$CHICKENLIB"; then
  	AC_MSG_RESULT(not found)
  else
        AC_MSG_RESULT($CHICKENLIB)
  fi

fi # have CHICKEN_CONFIG
fi # Check for --without-chicken

AC_SUBST(CHICKEN)
AC_SUBST(CHICKEN_CSC)
AC_SUBST(CHICKEN_CSI)
AC_SUBST(CHICKENOPTS)
AC_SUBST(CHICKENLIB)
AC_SUBST(CHICKENSHAREDLIB)

#----------------------------------------------------------------
# Look for C#
#----------------------------------------------------------------

AC_ARG_WITH(csharp, AS_HELP_STRING([--without-csharp], [Disable CSharp]), [with_csharp="$withval"], [with_csharp="$alllang_default"])
AC_ARG_WITH(cil-interpreter, [  --with-cil-interpreter=path     Set location of CIL interpreter for CSharp],[CSHARPBIN="$withval"], [CSHARPBIN=])
AC_ARG_WITH(csharp-compiler, [  --with-csharp-compiler=path     Set location of CSharp compiler],[CSHARPCOMPILERBIN="$withval"], [CSHARPCOMPILERBIN=])

# First, check for "--without-csharp" or "--with-csharp=no".
if test x"${with_csharp}" = xno; then
AC_MSG_NOTICE([Disabling CSharp])
CSHARPCOMPILER=
else

if test -z "$CSHARPCOMPILERBIN" ; then
  case $host in
  *-*-cygwin* | *-*-mingw*)
    # prefer Mono gmcs (.NET 2.0) over mcs (.NET 1.1) - note mcs-1.2.3 has major pinvoke bug
    AC_CHECK_PROGS(CSHARPCOMPILER, csc mono-csc gmcs mcs cscc)
    if test -n "$CSHARPCOMPILER" && test "$CSHARPCOMPILER" = "csc" ; then
      AC_MSG_CHECKING(whether csc is the Microsoft CSharp compiler)
      csc 2>/dev/null | grep "C#" > /dev/null || CSHARPCOMPILER=""
      if test -z "$CSHARPCOMPILER" ; then
        AC_MSG_RESULT(no)
        AC_CHECK_PROGS(CSHARPCOMPILER, mono-csc gmcs mcs cscc)
      else
        AC_MSG_RESULT(yes)
      fi
    fi
    ;;
  *)AC_CHECK_PROGS(CSHARPCOMPILER, mono-csc gmcs mcs cscc);;
  esac
else
  CSHARPCOMPILER="$CSHARPCOMPILERBIN"
fi

CSHARPCONVERTPATH="Tools/convertpath -u"
if test -z "$CSHARPBIN" ; then
  CSHARPCILINTERPRETER=""
  CSHARPCILINTERPRETER_FLAGS=""
  if test "cscc" = "$CSHARPCOMPILER" ; then
    AC_CHECK_PROGS(CSHARPCILINTERPRETER, ilrun)
  else
    if test "mcs" = "$CSHARPCOMPILER"; then
      # Check that mcs is the C# compiler and not the Unix mcs utility by examining the output of 'mcs --version'
      # The Mono compiler should emit: Mono C# compiler version a.b.c.d
      csharp_version_raw=`(mcs --version) 2>/dev/null`
      csharp_version_searched=`(mcs --version | sed -e "/C#/b" -e "/Mono/b" -e d) 2>/dev/null` # return string if contains 'Mono' or 'C#'
      CSHARPCOMPILER=""
      if test -n "$csharp_version_raw" ; then
        if test "$csharp_version_raw" = "$csharp_version_searched" ; then
          CSHARPCOMPILER="mcs"
        fi
      fi
      if test "mcs" != "$CSHARPCOMPILER" ; then
        echo "mcs is not a working Mono C# compiler"
      fi
    fi
    if test "mcs" = "$CSHARPCOMPILER" || test "gmcs" = "$CSHARPCOMPILER" || test "mono-csc" = "$CSHARPCOMPILER"; then
        AC_CHECK_PROGS(CSHARPCILINTERPRETER, mono) # Mono JIT
        CSHARPCILINTERPRETER_FLAGS="--debug"
    else
      if test "csc" = "$CSHARPCOMPILER"; then
          CSHARPCONVERTPATH="Tools/convertpath -w"
      fi
    fi
  fi
else
  CSHARPCILINTERPRETER="$CSHARPBIN"
fi

# Cygwin requires the Windows standard (Pascal) calling convention as it is a Windows executable and not a Cygwin built executable
case $host in
*-*-cygwin* | *-*-mingw*)
    if test "$GCC" = yes; then
        CSHARPDYNAMICLINKING="$GCC_MNO_CYGWIN -mthreads -Wl,--add-stdcall-alias"
        CSHARPCFLAGS="$GCC_MNO_CYGWIN -mthreads"
    else
        CSHARPDYNAMICLINKING=""
        CSHARPCFLAGS=""
    fi ;;
*)
        CSHARPDYNAMICLINKING=""
        CSHARPCFLAGS=""
        ;;
esac

# CSharp on Windows platforms including Cygwin doesn't use libname.dll, rather name.dll when loading dlls
case $host in
*-*-cygwin* | *-*-mingw*) CSHARPLIBRARYPREFIX="";;
*)CSHARPLIBRARYPREFIX="lib";;
esac

# C#/Mono on Mac OS X tweaks
case $host in
*-*-darwin*)
    CSHARPSO=".so"
    ;;
*)
    CSHARPSO=$SO
    ;;
esac
fi

AC_SUBST(CSHARPCILINTERPRETER_FLAGS)
AC_SUBST(CSHARPCILINTERPRETER)
AC_SUBST(CSHARPCONVERTPATH)
AC_SUBST(CSHARPCOMPILER)
AC_SUBST(CSHARPDYNAMICLINKING)
AC_SUBST(CSHARPLIBRARYPREFIX)
AC_SUBST(CSHARPCFLAGS)
AC_SUBST(CSHARPSO)

#----------------------------------------------------------------
# Look for Lua
#----------------------------------------------------------------

LUABIN=
LUAINCLUDE=
LUALIB=
LUADYNAMICLOADLIB=
LUAFLAGS=
LUALINK=
# note: if LUABIN is empty then lua tests will not be done
# LUABIN will be cleared if certain dependencies cannot be found

AC_ARG_WITH(lua, AS_HELP_STRING([--without-lua], [Disable Lua])
AS_HELP_STRING([--with-lua=path], [Set location of Lua executable]),[ LUABIN="$withval"], [LUABIN="$alllang_default"])
AC_ARG_WITH(luaincl,[  --with-luaincl=path     Set location of Lua include directory],[
	LUAINCLUDE="$withval"], [LUAINCLUDE=])
AC_ARG_WITH(lualib,[  --with-lualib=path      Set location of Lua library directory],[
	LUALIB="$withval"], [LUALIB=])

# First, check for "--without-lua" or "--with-lua=no".
if test x"${LUABIN}" = xno; then
AC_MSG_NOTICE([Disabling Lua])
else

# can we find lua?
if test "x$LUABIN" = xyes; then
   # We look for a versioned Lua binary first, as there can be
   # multiple versions of Lua installed on some systems (like Debian).
   # The search order should match the include-file and library search
   # orders below (a Lua shared library built for one version may not
   # work with a Lua binary of a different version).
   AC_PATH_PROGS(LUABIN, [lua5.4 lua5.3 lua5.2 lua5.1 lua])
fi

# check version: we need Lua 5.x
if test "$LUABIN"; then
  AC_MSG_CHECKING(Lua version)
  # if version 5.x
  LUAV5=`$LUABIN -e 'if string.sub(_VERSION,5,5)=="5" then print "1" end'`
  # if not version 5.0
  LUAV51=`$LUABIN -e 'if string.sub(_VERSION,5,7)~="5.0" then print "1" end'`

  if test -z "$LUAV5"; then
    AC_MSG_WARN(Not Lua 5.x, SWIG does not support this version of Lua)
    LUABIN=""
  elif test -z "$LUAV51"; then
    AC_MSG_RESULT(Lua 5.0.x)
  else
    AC_MSG_RESULT(Lua 5.1 or later)
  fi
fi

if test "$LUABIN"; then
  AC_MSG_CHECKING(whether Lua dynamic loading is enabled)
  # using Lua to check Lua
  # lua 5.0 & 5.1 have different fn names
  if test -z "$LUAV51"; then
    LUADYNAMICLOADLIB=`$LUABIN -e '_,_,c=loadlib("no_such_lib","") if c~="absent" then print "1" end'`
  else
    LUADYNAMICLOADLIB=`$LUABIN -e '_,_,c=package.loadlib("no_such_lib","") if c~="absent" then print "1" end'`
  fi

  if test -z "$LUADYNAMICLOADLIB"; then
    AC_MSG_RESULT(no)
  else
    AC_MSG_RESULT(yes)
  fi

  # look for the header files & set LUAFLAGS accordingly
  # will clear LUABIN if not present
  if test -n "$LUAINCLUDE"; then
    AC_CHECK_FILE($LUAINCLUDE/lua.h,[LUAFLAGS="$ISYSTEM$LUAINCLUDE"],[LUABIN=])
  else
    LUA_OK="1"
    CFLAGS_SAVED=$CFLAGS
    CFLAGS= # Use empty CFLAGS to avoid failure: "present but cannot be compiled"
    AC_CHECK_HEADER([lua.h],[LUAFLAGS=""],[LUA_OK=""])
    CFLAGS=$CFLAGS_SAVED
    # if we didn't get it, going to have to look elsewhere (the hard way)
    if test -z "$LUA_OK"; then
      AC_MSG_CHECKING(for lua.h in other locations)
      # note: Debian/Ubuntu seem to like /usr/include/lua5.1/lua.h
      # The ordering of the include directories to search should match
      # the ordering of libraries to search in the library test below.
      inc=/usr/include
      dirs="$inc/lua5.4 $inc/lua5.3 $inc/lua5.2 $inc/lua5.1 $inc/lua51 $inc/lua5.0 $inc/lua50 /usr/local/include"
      for i in $dirs; do
        #echo "$i"
        if test -r $i/lua.h; then
          AC_MSG_RESULT($i/lua.h)
          LUAFLAGS="$ISYSTEM$i"
          break
        fi
      done
      if test -z "$LUAFLAGS"; then
        AC_MSG_RESULT(not found)
        LUABIN="" # clear the bin
      fi
    fi
  fi

  # look for the library files & set LUALINK accordingly
  # will clear LUABIN if not present
  lua_save_LIBS=$LIBS # the code seems to disrupt LIBS, so saving

  if test -n "$LUALIB"; then
    AC_CHECK_FILE($LUALIB/liblua.a,[LUALINK="-L$LUALIB -llua"],[LUABIN=])
  else
    AC_SEARCH_LIBS(lua_close, [lua lua5.4 lua5.3 lua5.2 lua5.1 lua51 lua5.0 lua50], [LUALINK="-l$ac_lib"],[LUABIN=])
  fi

  # adding lualib for lua 5.0
  if test -z "$LUAV51"; then # extra for lua 5.0
    LUALINK="$LUALINK -llualib"
  fi

  LIBS=$lua_save_LIBS	# restore LIBS
fi

fi # if not disabled

AC_SUBST(LUADYNAMICLINKING)
AC_SUBST(LUAFLAGS)
AC_SUBST(LUALINK)
AC_SUBST(LUABIN)

#----------------------------------------------------------------
# Look for Allegro Common Lisp
#----------------------------------------------------------------

ALLEGROCLBIN=

AC_ARG_WITH(allegrocl, AS_HELP_STRING([--without-allegrocl], [Disable Allegro CL])
AS_HELP_STRING([--with-allegrocl=path], [Set location of Allegro CL executable (alisp)]),[ ALLEGROCLBIN="$withval"], [ALLEGROCLBIN="$alllang_default"])

# First, check for "--without-allegrocl" or "--with-allegrocl=no".
if test x"${ALLEGROCLBIN}" = xno; then
AC_MSG_NOTICE([Disabling Allegro CL])
ALLEGROCLBIN=
else

# can we find allegrocl?
if test "x$ALLEGROCLBIN" = xyes; then
   AC_PATH_PROG(ALLEGROCLBIN, alisp)
fi
fi

AC_SUBST(ALLEGROCLBIN)

#----------------------------------------------------------------
# Look for GNU CLISP
#----------------------------------------------------------------

CLISPBIN=

AC_ARG_WITH(clisp, AS_HELP_STRING([--without-clisp], [Disable CLISP])
AS_HELP_STRING([--with-clisp=path], [Set location of CLISP executable (clisp)]),[ CLISPBIN="$withval"], [CLISPBIN="$alllang_default"])

# First, check for "--without-clisp" or "--with-clisp=no".
if test x"${CLISPBIN}" = xno; then
AC_MSG_NOTICE([Disabling CLISP])
CLISPBIN=
else

# can we find clisp?
if test "x$CLISPBIN" = xyes; then
   AC_PATH_PROG(CLISPBIN, clisp)
fi
fi

AC_SUBST(CLISPBIN)

#----------------------------------------------------------------
# Look for GNU R
#----------------------------------------------------------------

RBIN=

AC_ARG_WITH(r, AS_HELP_STRING([--without-r], [Disable R])
AS_HELP_STRING([--with-r=path], [Set location of R executable (r)]),[ RBIN="$withval"], [RBIN="$alllang_default"])

# First, check for "--without-r" or "--with-r=no".
if test x"${RBIN}" = xno; then
AC_MSG_NOTICE([Disabling R])
RBIN=
else

# can we find R?
if test "x$RBIN" = xyes; then
   AC_PATH_PROG(RBIN, R)
fi
fi

AC_SUBST(RBIN)

#----------------------------------------------------------------
# Look for Go compilers
#----------------------------------------------------------------

AC_ARG_WITH(go, AS_HELP_STRING([--without-go], [Disable Go])
AS_HELP_STRING([--with-go=path], [Set location of Go compiler]),[GOBIN="$withval"], [GOBIN="$alllang_default"])

if test x"${GOBIN}" = xno; then
  AC_MSG_NOTICE([Disabling Go])
  GO=
  GOC=
  GO1=false
  GO12=false
  GO13=false
  GO15=false
  GOGCC=false
  GCCGO=
  GOOPT=
  GCCGOOPT=
  GOVERSIONOPTION=
else

  if test "x$GOBIN" = xyes; then
    AC_CHECK_PROGS(GO, go)
  else
    GO="$GOBIN"
  fi

  GOGCC=false
  GCCGO=
  GO1=false
  GO12=false
  GO13=false
  GO15=false
  GOOPT=
  GCCGOOPT=
  GOVERSIONOPTION=

  if test -n "$GO" ; then
    GO1=true
    GOVERSIONOPTION=version
    go_version=$($GO $GOVERSIONOPTION | sed -e 's/go version //')
    case "$go_version" in
    go1 | go1.[[01234]]*)
      GOC=$(sh -c "$(go env) && echo \$GOCHAR")c
      ;;
    *)
      GOC=compile
      ;;
    esac
    AC_MSG_CHECKING([whether go version is too old])
    case $go_version in
    go1.1* | go1.0* | go1 )
      AC_MSG_RESULT([yes - minimum version is 1.2])
      GO=
      GOOPT="-intgosize 32"
      ;;
    *)
      AC_MSG_RESULT([no])
      case "$(go env GOARCH)" in
      amd64 | arm64 | ppc64*)
        GOOPT="-intgosize 64"
	;;
      *)
        GOOPT="-intgosize 32"
	;;
      esac
      ;;
    esac
    case $go_version in
    go1.0* | go1 | go1.1*)
      GOOPT="$GOOPT -use-shlib"
      ;;
    go1.2*)
      GO12=true
      ;;
    go1.3* | go1.4*)
      GO13=true
      ;;
    *)
      GO15=true
      ;;
    esac
  fi

  AC_CHECK_PROGS(GCCGO, gccgo)

  if test -n "$GCCGO" ; then
    if $GCCGO --help 2>/dev/null | grep gccgo >/dev/null 2>&1 ; then
      AC_MSG_CHECKING([whether gccgo version is too old])
      go_version=[`$GO $GOVERSIONOPTION | sed -n '1p' | sed -e 's/^.* \([0-9.]*\) *$/\1/' -e 's/[.]//g'`]
      if test "x$go_version" = x; then
        AC_MSG_RESULT([could not determine gccgo version])
        GCCGO=
      elif test "$go_version" -lt 470; then
        AC_MSG_RESULT([yes - minimum version is 4.7.0])
        GCCGO=
      else
        AC_MSG_RESULT([no])
        if test "$go_version" -lt 480; then
          GCCGOOPT="-intgosize 32"
        else
          AC_CHECK_SIZEOF([void *], [4])
          if test "$ac_cv_sizeof_void_p" = "8"; then
            GCCGOOPT="-intgosize 64"
          else
            GCCGOOPT="-intgosize 32"
          fi
        fi
      fi
    fi
  fi
fi

AC_SUBST(GOGCC)
AC_SUBST(GCCGO)
AC_SUBST(GO)
AC_SUBST(GOC)
AC_SUBST(GO1)
AC_SUBST(GO12)
AC_SUBST(GO13)
AC_SUBST(GO15)
AC_SUBST(GOOPT)
AC_SUBST(GCCGOOPT)
AC_SUBST(GOVERSIONOPTION)

#----------------------------------------------------------------
# Look for D
#----------------------------------------------------------------

AC_ARG_WITH(d, AS_HELP_STRING([--without-d], [Disable D]), [with_d="$withval"], [with_d="$alllang_default"])
AC_ARG_WITH(d1-compiler, [  --with-d1-compiler=path  Set location of D1/Tango compiler (DMD compatible)],[D1COMPILERBIN="$withval"], [D1COMPILERBIN=])
AC_ARG_WITH(d2-compiler, [  --with-d2-compiler=path  Set location of D2 compiler (DMD compatible)],[D2COMPILERBIN="$withval"], [D2COMPILERBIN=])


# First, check for "--without-d" or "--with-d=no".
if test x"${with_d}" = xno; then
  AC_MSG_NOTICE([Disabling D])
  D1COMPILER=
  D2COMPILER=
else
  old_ac_ext=$ac_ext
  ac_ext=d

  if test -z "$D1COMPILERBIN" ; then
    AC_CHECK_PROGS(D1COMPILER, dmd ldmd gdmd)

    if test -n "$D1COMPILER" ; then
      AC_MSG_CHECKING(whether the D1/Tango compiler works)
      cat > conftest.$ac_ext <<_ACEOF
import tango.io.Stdout;
void main() {
}
_ACEOF
      rm -f conftest.$ac_objext
      AS_IF(
        [_AC_DO_STDERR($D1COMPILER conftest.$ac_ext) && test ! -s conftest.err && test -s conftest.$ac_objext],
        [AC_MSG_RESULT([yes])],
        [_AC_MSG_LOG_CONFTEST AC_MSG_RESULT([no])
        D1COMPILER=]
      )
      rm -f core conftest.err conftest.$ac_objext conftest.$ac_ext
    fi
  else
    D1COMPILER="$D1COMPILERBIN"
  fi

  if test -z "$D2COMPILERBIN" ; then
    AC_CHECK_PROGS(D2COMPILER, dmd gdmd)

    if test -n "$D2COMPILER" ; then
      AC_MSG_CHECKING(whether the D2 compiler works)
      cat > conftest.$ac_ext <<_ACEOF
import std.algorithm;
void main() {
}
_ACEOF
      rm -f conftest.$ac_objext
      AS_IF(
        [_AC_DO_STDERR($D2COMPILER conftest.$ac_ext) && test ! -s conftest.err && test -s conftest.$ac_objext],
        [AC_MSG_RESULT([yes])],
        [_AC_MSG_LOG_CONFTEST AC_MSG_RESULT([no])
        D2COMPILER=]
      )
      rm -f core conftest.err conftest.$ac_objext conftest.$ac_ext
    fi
  else
    D2COMPILER="$D2COMPILERBIN"
  fi

  ac_ext=$old_ac_ext
fi

if test -n "$D1COMPILER"; then
  DDEFAULTVERSION=1
elif test -n "$D2COMPILER"; then
  DDEFAULTVERSION=2
fi

# Do not prefix library file names with "lib" on Windows.
case $host in
*-*-cygwin* | *-*-mingw*) DLIBPREFIX="";;
*)DLIBPREFIX="lib";;
esac

AC_SUBST(D1COMPILER)
AC_SUBST(D2COMPILER)
AC_SUBST(DDEFAULTVERSION)
AC_SUBST(DLIBPREFIX)

#----------------------------------------------------------------
# Determine which languages to use for examples/test-suite
#----------------------------------------------------------------

SKIP_TCL=
if test -z "$TCLINCLUDE" || test -z "$TCLLIB" ; then
    SKIP_TCL="1"
fi
AC_SUBST(SKIP_TCL)


SKIP_PERL5=
if test -z "$PERL" || test -z "$PERL5EXT" || test -z "$PERL5TESTMORE"; then
    SKIP_PERL5="1"
fi
AC_SUBST(SKIP_PERL5)


SKIP_OCTAVE=
if test -z "$OCTAVE" ; then
    SKIP_OCTAVE="1"
fi
AC_SUBST(SKIP_OCTAVE)


SKIP_PYTHON=
if (test -z "$PYINCLUDE" || test -z "$PYLINK") &&
   (test -z "$PY3INCLUDE" || test -z "$PY3LINK") ; then
    SKIP_PYTHON="1"
fi
AC_SUBST(SKIP_PYTHON)

SKIP_PYTHON3=
if test -z "$PY3INCLUDE" || test -z "$PY3LINK" ; then
    SKIP_PYTHON3="1"
fi
AC_SUBST(SKIP_PYTHON3)

SKIP_JAVA=
if test -z "$JAVA" || test -z "$JAVAC" || test -z "$JAVAINC" ; then
    SKIP_JAVA="1"
fi
AC_SUBST(SKIP_JAVA)

SKIP_JAVASCRIPT=
if test -z "$JAVASCRIPT" || ( test -z "$NODEJS" && test -z "$JSCENABLED" && test -z "$JSV8ENABLED" ) ; then
    SKIP_JAVASCRIPT="1"
fi
AC_SUBST(SKIP_JAVASCRIPT)

SKIP_GUILE=
if test -z "$GUILE" || test -z "$GUILE_LIBS" ; then
    SKIP_GUILE="1"
fi
AC_SUBST(SKIP_GUILE)


SKIP_MZSCHEME=
if test -z "$MZC" || test -z "$MZDYNOBJ" ; then
    SKIP_MZSCHEME="1"
fi
AC_SUBST(SKIP_MZSCHEME)


SKIP_RUBY=
if test -z "$RUBY" || test -z "$RUBYINCLUDE" || test -z "$RUBYLIB" ; then
    SKIP_RUBY="1"
fi
AC_SUBST(SKIP_RUBY)


SKIP_PHP=
if test -z "$PHP" || test -z "$PHPINC" ; then
    SKIP_PHP="1"
fi
AC_SUBST(SKIP_PHP)


SKIP_PHP5=
if test -z "$PHP5" || test -z "$PHP5INC" ; then
    SKIP_PHP5="1"
fi
AC_SUBST(SKIP_PHP5)


SKIP_OCAML=
if test -z "$OCAMLC" ; then
    SKIP_OCAML="1"
fi
AC_SUBST(SKIP_OCAML)


SKIP_PIKE=
if test -z "$PIKE" || test -z "$PIKEINCLUDE" ; then
    SKIP_PIKE="1"
fi
AC_SUBST(SKIP_PIKE)


SKIP_CHICKEN=
if test -z "$CHICKEN_CSC" || test -z "$CHICKEN"; then
    SKIP_CHICKEN="1"
fi
AC_SUBST(SKIP_CHICKEN)


SKIP_CSHARP=
if test -z "$CSHARPCOMPILER" ; then
    SKIP_CSHARP="1"
else
    if test "cscc" = "$CSHARPCOMPILER" && test -z "$CSHARPCILINTERPRETER" ; then
      SKIP_CSHARP="1"
    fi
fi
AC_SUBST(SKIP_CSHARP)

SKIP_MODULA3="1" # Always skipped!
AC_SUBST(SKIP_MODULA3)

SKIP_LUA=
# we need LUABIN & dynamic loading
if test -z "$LUABIN" || test -z "$LUADYNAMICLOADLIB"; then
    SKIP_LUA="1"
fi
AC_SUBST(SKIP_LUA)

SKIP_ALLEGROCL=
if test -z "$ALLEGROCLBIN" ; then
    SKIP_ALLEGROCL="1"
fi
AC_SUBST(SKIP_ALLEGROCL)

SKIP_CLISP=
if test -z "$CLISPBIN" ; then
    SKIP_CLISP="1"
fi
AC_SUBST(SKIP_CLISP)

SKIP_R=
if test -z "$RBIN" ; then
    SKIP_R="1"
fi
AC_SUBST(SKIP_R)

SKIP_CFFI=
#if test -z "$CFFIBIN" ; then
    SKIP_CFFI="1"
#fi
AC_SUBST(SKIP_CFFI)

SKIP_UFFI=
#if test -z "$UFFIBIN" ; then
    SKIP_UFFI="1"
#fi
AC_SUBST(SKIP_UFFI)

SKIP_SCILAB=
if test -z "$SCILAB"; then
    SKIP_SCILAB="1"
fi
AC_SUBST(SKIP_SCILAB)

SKIP_GO=
if test -z "$GO" ; then
    SKIP_GO="1"
fi
AC_SUBST(SKIP_GO)

SKIP_D=
if test -z "$DDEFAULTVERSION" ; then
    SKIP_D="1"
fi
AC_SUBST(SKIP_D)

#----------------------------------------------------------------
# Additional language dependencies
#----------------------------------------------------------------
SKIP_GCJ=
if test -z "$GCJ" || test -z "$GCJH" ; then
  SKIP_GCJ="1"
else
  if test "$GCC" != yes; then
    SKIP_GCJ="1"
  fi
fi
AC_SUBST(SKIP_GCJ)


SKIP_ANDROID=
if test -z "$ANDROID" || test -z "$ADB" || test -z "$ANT" || test -z "$NDKBUILD" ; then
    SKIP_ANDROID="1"
fi
AC_SUBST(SKIP_ANDROID)



#----------------------------------------------------------------
# Miscellaneous
#----------------------------------------------------------------

ABS_SRCDIR=`(cd ${srcdir} && pwd)`

dnl Under Cygwin, we may need native absolute path as it is used by SWIG, which
dnl may be a native, and not a Cygwin, program (this is the case when it's
dnl built using MinGW or cccl compiler in Cygwin environment). However it may,
dnl although this is probably more rare, also be built as a Cygwin program.
dnl Using "mixed" path like we do here allows the path to work in both cases.
case $host in
  *-*-mingw* ) ABS_SRCDIR=`${srcdir}/Tools/convertpath -m $ABS_SRCDIR` ;;
  *-*-cygwin* ) ABS_SRCDIR=`cygpath --mixed $ABS_SRCDIR` ;;
esac

# Root directory
ROOT_DIR=`pwd`
case $host in
*-*-cygwin*)
  # Translate path for native Windows compilers for use with 'make check'
  if (cygpath --mixed $ROOT_DIR) >/dev/null 2>/dev/null; then
    ROOT_DIR=`cygpath --mixed $ROOT_DIR`
  fi
  ;;
esac

case $host in
*-*-cygwin* | *-*-mingw*)
  # Extra files generated by some Windows compilers
  EXTRA_CLEAN="*.stackdump *.exp *.lib *.pdb *.ilk"
  ;;
esac

AC_SUBST(ROOT_DIR)
AC_SUBST(EXTRA_CLEAN)
AC_SUBST(ac_aux_dir)

# Configure SWIG_LIB path

AC_ARG_WITH(swiglibdir,[  --with-swiglibdir=DIR   Put SWIG system-independent libraries into DIR.],
  [swig_lib="$withval"], [swig_lib="${datadir}/swig/${PACKAGE_VERSION}"])
SWIG_LIB_INSTALL=${swig_lib}
AC_SUBST(SWIG_LIB_INSTALL)
AC_DEFINE_DIR(SWIG_LIB, swig_lib, [Directory for SWIG system-independent libraries])

case $build in
  # Windows does not understand unix directories. Convert into a windows directory with drive letter.
  *-*-mingw*) SWIG_LIB_WIN_UNIX=`${srcdir}/Tools/convertpath -m $SWIG_LIB`;;
  *-*-cygwin*) SWIG_LIB_WIN_UNIX=`cygpath --mixed "$SWIG_LIB"`;;
  *) SWIG_LIB_WIN_UNIX="";;
esac
AC_DEFINE_UNQUOTED(SWIG_LIB_WIN_UNIX, ["$SWIG_LIB_WIN_UNIX"], [Directory for SWIG system-independent libraries (Unix install on native Windows)])

SWIG_LIB_PREINST=$ABS_SRCDIR/Lib
AC_SUBST(SWIG_LIB_PREINST)

dnl For testing purposes, clear SWIG_LIB when building SWIG in the source
dnl directory under Windows because it is supposed to work without SWIG_LIB
dnl being set. Otherwise it always needs to be set.
SWIG_LIB_SET="env SWIG_LIB=\$(SWIG_LIB_DIR)"
if test "${srcdir}" = "."; then
    AC_EGREP_CPP([yes],
    [#ifdef _WIN32
     yes
    #endif
    ], [SWIG_LIB_SET="env SWIG_LIB="], [])
fi
AC_SUBST(SWIG_LIB_SET)

AC_CONFIG_FILES([
    Makefile
    swig.spec
    Examples/Makefile
    Examples/d/example.mk
    Examples/xml/Makefile
    Examples/test-suite/errors/Makefile
    Examples/test-suite/chicken/Makefile
    Examples/test-suite/csharp/Makefile
    Examples/test-suite/d/Makefile
    Examples/test-suite/guile/Makefile
    Examples/test-suite/java/Makefile
    Examples/test-suite/javascript/Makefile
    Examples/test-suite/mzscheme/Makefile
    Examples/test-suite/ocaml/Makefile
    Examples/test-suite/octave/Makefile
    Examples/test-suite/perl5/Makefile
    Examples/test-suite/php/Makefile
    Examples/test-suite/php5/Makefile
    Examples/test-suite/pike/Makefile
    Examples/test-suite/python/Makefile
    Examples/test-suite/ruby/Makefile
    Examples/test-suite/scilab/Makefile
    Examples/test-suite/tcl/Makefile
    Examples/test-suite/lua/Makefile
    Examples/test-suite/allegrocl/Makefile
    Examples/test-suite/clisp/Makefile
    Examples/test-suite/cffi/Makefile
    Examples/test-suite/uffi/Makefile
    Examples/test-suite/r/Makefile
    Examples/test-suite/go/Makefile
    Source/Makefile
    Tools/javascript/Makefile
])
AC_CONFIG_FILES([preinst-swig], [chmod +x preinst-swig])
AC_CONFIG_FILES([CCache/ccache_swig_config.h])

#--------------------------------------------------------------------
# Building Examples/ out of source directory
#--------------------------------------------------------------------

# If building out of source tree, replicate Examples/ source tree in
# build directory, and copy over Makefiles from source directory.
# Prefix each Makefile with a header which sets SRCDIR to the relative
# source directory, and provides a rule for updating the Makefile from
# its original source.
AC_CONFIG_COMMANDS([Examples],[
  if test "x${srcdir}" != "x." ; then
    AC_MSG_NOTICE([generating Examples build tree])
    for mkfile in `cd ${srcdir} && find Examples/ -type f -name Makefile`; do
      dir=`dirname ${mkfile}`
      d=${dir}
      reldir=""
      while test "x$d" != "x." ; do
        d=`dirname $d`
        reldir="${reldir}../"
      done
      relsrcdir=${reldir}${srcdir}/
      AS_MKDIR_P([${dir}])
      cat <<EOF >${mkfile}
# DO NOT EDIT: instead edit ${relsrcdir}${mkfile}
# and run (cd ${reldir} && ./config.status) to regenerate
TOP_BUILDDIR_TO_TOP_SRCDIR = ${srcdir}/
SRCDIR = ${relsrcdir}${dir}/

EOF
      cat ${srcdir}/${mkfile} >>${mkfile}
    done
  fi
])

#--------------------------------------------------------------------

AC_OUTPUT

langs=""
test -n "$SKIP_ALLEGROCL"	|| langs="${langs}allegrocl "
test -n "$SKIP_CFFI"		|| langs="${langs}cffi "
test -n "$SKIP_CHICKEN"		|| langs="${langs}chicken "
test -n "$SKIP_CLISP"		|| langs="${langs}clisp "
test -n "$SKIP_CSHARP"		|| langs="${langs}csharp "
test -n "$SKIP_D"		|| langs="${langs}d "
test -n "$SKIP_GO"		|| langs="${langs}go "
test -n "$SKIP_GUILE"		|| langs="${langs}guile "
test -n "$SKIP_JAVA"		|| langs="${langs}java "
test -n "$SKIP_JAVASCRIPT"	|| langs="${langs}javascript "
test -n "$SKIP_LUA"		|| langs="${langs}lua "
test -n "$SKIP_MODULA3"		|| langs="${langs}modula3 "
test -n "$SKIP_MZSCHEME"	|| langs="${langs}mzscheme "
test -n "$SKIP_OCAML"		|| langs="${langs}ocaml "
test -n "$SKIP_OCTAVE"		|| langs="${langs}octave "
test -n "$SKIP_PERL5"		|| langs="${langs}perl5 "
test -n "$SKIP_PHP"		|| langs="${langs}php "
test -n "$SKIP_PHP5"		|| langs="${langs}php5 "
test -n "$SKIP_PIKE"		|| langs="${langs}pike "
test -n "$SKIP_PYTHON"		|| langs="${langs}python "
test -n "$SKIP_R"		|| langs="${langs}r "
test -n "$SKIP_RUBY"		|| langs="${langs}ruby "
test -n "$SKIP_SCILAB"		|| langs="${langs}scilab "
test -n "$SKIP_TCL"		|| langs="${langs}tcl "
test -n "$SKIP_UFFI"		|| langs="${langs}uffi "

echo "
The SWIG test-suite and examples are configured for the following languages:
$langs
"

dnl configure.ac ends here
