dnl 
dnl  Copyright (c) 2006 The Trustees of Indiana University and Indiana
dnl                     University Research and Technology
dnl                     Corporation.  All rights reserved.
dnl  Copyright (c) 2006 The Technical University of Chemnitz. All 
dnl                     rights reserved.
dnl 
dnl  Author(s): Torsten Hoefler <htor@cs.indiana.edu>
dnl 
AC_DEFUN([NBC_CHECK_THREAD],
    [AC_ARG_WITH(thread, AC_HELP_STRING([--with-thread], [compile with a progress thread that supports asynchronous progress]))
    if test x"${with_thread}" != x; then
        AC_CHECK_HEADERS([pthread.h])
        AC_CHECK_LIB([pthread], [pthread_create])
        AC_DEFINE(HAVE_PROGRESS_THREAD, 1, compile with progress thread)
        AC_MSG_NOTICE([Progress thread enabled])
        have_thread=1
    fi
    ]
)
