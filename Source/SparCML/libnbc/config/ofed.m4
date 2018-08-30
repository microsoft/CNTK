dnl 
dnl  Copyright (c) 2006 The Trustees of Indiana University and Indiana
dnl                     University Research and Technology
dnl                     Corporation.  All rights reserved.
dnl  Copyright (c) 2006 The Technical University of Chemnitz. All 
dnl                     rights reserved.
dnl 
dnl  Author(s): Torsten Hoefler <htor@cs.indiana.edu>
dnl 
AC_DEFUN([TEST_OFED],
    [AC_ARG_WITH(ofed, AC_HELP_STRING([--with-ofed], [compile with Open Fabrics support (ARG can be the path to the root Open Fabrics directory)]))
    ofed_found=no
    if test x"${with_ofed}" = xyes; then
        AC_CHECK_HEADER(infiniband/verbs.h, ofed_found=yes, [AC_MSG_ERROR([OFED selected but not available!])])
    elif test x"${with_ofed}" != x; then
        AC_CHECK_HEADER(${with_ofed}/include/infiniband/verbs.h, [ng_ofed_path=${with_ofed}; ofed_found=yes], [AC_MSG_ERROR([Can't find OFED in ${with_ofed}])])
    fi
    if test x"${with_ofed}" != x; then
        # try to use the library ...
        LIBS="${LIBS} -L${ng_ofed_path}/lib64"
        AC_CHECK_LIB([ibverbs], [ibv_reg_mr])
      
        AC_DEFINE(HAVE_OFED, 1, enables the ofed module)
        AC_MSG_NOTICE([OFED support enabled])
        if test x${ng_ofed_path} != x; then
          CFLAGS="${CFLAGS} -I${ng_ofed_path}/include"
        fi
        have_ofed=1
    fi
    ]
)
