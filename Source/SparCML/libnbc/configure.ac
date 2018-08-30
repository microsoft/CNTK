#
# Copyright (c) 2006 The Trustees of Indiana University and Indiana
#                    University Research and Technology
#                    Corporation.  All rights reserved.
# Copyright (c) 2006 The Technical University of Chemnitz. All 
#                    rights reserved.
#
# Author(s): Torsten Hoefler <htor@cs.indiana.edu>
#

AC_PREREQ(2.59)
AC_INIT(libNBC, 1.1.1)
AC_CONFIG_AUX_DIR(config)
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE(libNBC, 1.1.1)
AM_CONFIG_HEADER([config.h])


# check for wrapper compilers
NBC_CHECK_OMPI
NBC_CHECK_MPI
NBC_CHECK_THREAD
NBC_CHECK_RT_THREAD
KLM_SYS_WEAK_ALIAS

# Checks for programs.
AC_PROG_CXX
AC_PROG_RANLIB

# Checks for libraries.
AC_CHECK_LIB([c], [printf])
AC_CHECK_LIB([m], [log])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([mpi.h stdlib.h stdio.h stdarg.h math.h string.h])
CFLAGS="${CFLAGS}"

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_HEADER_TIME

# Checks for library functions.
AC_PROG_GCC_TRADITIONAL
AC_CHECK_FUNCS([realloc malloc memset printf MPI_Init free log pow sqrt])

# defines F77 wrapper function macro F77_FUNC (name, NAME)
AC_F77_WRAPPERS

AM_DISABLE_SHARED
AM_ENABLE_STATIC
AC_PROG_LIBTOOL

TEST_OFED
TEST_DCMF
TEST_MPI22
AM_CONDITIONAL([HAVE_OFED], [test x$with_ofed != x])

AC_CONFIG_FILES([Makefile libdict/Makefile of/Makefile])
AC_OUTPUT

echo ""
echo "LibNBC Configuration Summary:"
if test x$have_ofed = x1; then
  echo -n "configuring OFED as transport interface (patching nbc.h) ... "
  cat nbc.h | sed -e "s/define USE_MPI/define USE_OFED/" > $srcdir/nbc.tmp
  mv $srcdir/nbc.tmp $srcdir/nbc.h
  echo "done"
elif test x$have_dcmf = x1; then
  echo -n "configuring DCMF as transport interface (patching nbc.h) ... "
  cat nbc.h | sed -e "s/define USE_MPI/define USE_DCMF/" > $srcdir/nbc.tmp
  mv $srcdir/nbc.tmp $srcdir/nbc.h
  echo "done"
else
  echo -n "configuring MPI as transport interface (patching nbc.h) ... "
  cat nbc.h | sed -e "s/define USE_OFED/define USE_MPI/" > $srcdir/nbc.tmp
  mv $srcdir/nbc.tmp $srcdir/nbc.h
  echo "done"
fi;  
if test x$have_rt_thread = x1; then
  echo "real-time progress thread enabled ... "
elif test x$have_thread = x1; then
  echo "progress thread enabled ... "
fi;
echo ""
