dnl 
dnl  Copyright (c) 2006 The Trustees of Indiana University and Indiana
dnl                     University Research and Technology
dnl                     Corporation.  All rights reserved.
dnl  Copyright (c) 2006 The Technical University of Chemnitz. All 
dnl                     rights reserved.
dnl 
dnl  Author(s): Torsten Hoefler <htor@cs.indiana.edu>
dnl 
AC_DEFUN([NBC_CHECK_MPI],
  # don't require mpicxx if we are inside OMPI (this variable is set
  # there)
  if test x"${HAVE_OMPI}" = "x"; then
    [AC_ARG_WITH(mpi,
     AC_HELP_STRING([--with-mpi], [compile with MPI support (ARG can be the path to the root MPI directory, if mpicxx is not in PATH)]),
    )
      if test x${MPICXX} == x; then
          MPICXX=mpicxx
      fi;
      if test x"${withval-yes}" != xyes; then
        AC_CHECK_PROG(mpicxx_found, $MPICXX, yes, no, ${withval}/bin)
        mpicxx_path=${withval}/bin/
      else
        AC_CHECK_PROG(mpicxx_found, $MPICXX, yes, no)
        mpicxx_path=
      fi
      if test "x${mpicxx_found}" = "xno"; then
        AC_MSG_ERROR(${mpicxx_path}mpicxx not found)
      else
        CC=${mpicxx_path}$MPICXX
        AC_DEFINE(HAVE_MPI, 1, enables MPI code)
      fi
      unset mpicxx_path mpicxx_found
     ]
  fi
)
