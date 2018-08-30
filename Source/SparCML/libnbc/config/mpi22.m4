dnl 
dnl  Copyright (c) 2006 The Trustees of Indiana University and Indiana
dnl                     University Research and Technology
dnl                     Corporation.  All rights reserved.
dnl  Copyright (c) 2006 The Technical University of Chemnitz. All 
dnl                     rights reserved.
dnl 
dnl  Author(s): Torsten Hoefler <htor@cs.indiana.edu>
dnl 
AC_DEFUN([TEST_MPI22], [AC_CHECK_FUNC(MPI_Dist_graph_create, [AC_DEFINE(HAVE_MPI22, 1, found MPI-2.2)], [AC_MSG_NOTICE([MPI-2.2 not found])])])
