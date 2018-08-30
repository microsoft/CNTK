dnl 
dnl  Copyright (c) 2006 The Trustees of Indiana University and Indiana
dnl                     University Research and Technology
dnl                     Corporation.  All rights reserved.
dnl  Copyright (c) 2006 The Technical University of Chemnitz. All 
dnl                     rights reserved.
dnl 
dnl  Author(s): Torsten Hoefler <htor@cs.indiana.edu>
dnl 
AC_DEFUN([TEST_DCMF], [AC_CHECK_HEADER(dcmf.h, [
  AC_DEFINE(HAVE_DCMF, 1, enables the dcmf module)
  have_dcmf=1
], [AC_MSG_NOTICE([DCMF support disabled])])])
