dnl 
dnl  Copyright (c) 2006 The Trustees of Indiana University and Indiana
dnl                     University Research and Technology
dnl                     Corporation.  All rights reserved.
dnl  Copyright (c) 2006 The Technical University of Chemnitz. All 
dnl                     rights reserved.
dnl 
dnl  Author(s): Torsten Hoefler <htor@cs.indiana.edu>
dnl 
AC_DEFUN([NBC_CHECK_OMPI],
	[AC_ARG_WITH(ompi,
	 AC_HELP_STRING([--with-ompi], [compile with Open MPI support (ARG has to be the path to the configured Open MPI source directory)]),
	)
          
  # this envronment variable is set when we are called from the
  # OMPI configure ... so we don't enfore OMPI to be configured
  # (it is being configured right now)
  if test x"${OMPI_MAGIC_VAR}" != "x"; then
    CPPFLAGS="${CPPFLAGS} -I${OMPI_MAGIC_VAR}"
    AC_DEFINE(HAVE_OMPI, 1, enables Open MPI glue code)
    #AC_DEFINE(HAVE_OMPI_BUNDLED, 1, bundled to OMPI code)
    # the relative ompi path for includes
    CPPFLAGS="${CPPFLAGS} -I../../../../ -I../../../../ompi/include/ -I../../../../opal/include/ -I../../../../orte/include/" 
    AC_MSG_NOTICE([compiling as bundled with Open MPI])
  else
    if test x"${withval-yes}" != xyes; then
        AC_MSG_CHECKING([for Open MPI source])
        echo ""
        # we don't use check_header because the linking doesn't work
        # (the files include some other files ...)
        AC_CHECK_FILE([${withval}/ompi/mca/coll/coll.h], ompi_found=yes, ompi_found=no)
        AC_CHECK_FILE([${withval}/ompi/include/ompi_config.h], ompi_configured=yes, ompi_configured=no)
        if test x"${ompi_found}" = "xyes"; then
          if test x"${ompi_configured}" = "xyes"; then
            CPPFLAGS="${CPPFLAGS} -I${withval}"
            AC_DEFINE(HAVE_OMPI, 1, enables Open MPI glue code)
            AC_MSG_NOTICE([compiling with Open MPI])
          else
            AC_MSG_ERROR([selected Open MPI source directory found in ${withval} but it is not configured])
          fi
        else
          AC_MSG_ERROR([selected Open MPI source directory no found in ${withval}])
        fi
    else
        AC_MSG_NOTICE([compiling without Open MPI support])
    fi
  fi
  unset ompi_found ompi_configured
   ]
)
