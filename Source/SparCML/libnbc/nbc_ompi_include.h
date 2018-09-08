/*
 * Copyright (c) 2006 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 * Copyright (c) 2006 The Technical University of Chemnitz. All 
 *                    rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
/* this gets it's own file since it is used twice ... */

#include "ompi/include/mpi.h"
/* the autohell defines some macros ... that OMPI redefines ... but we
 * need to undefine them before that *ARGH* */
#undef PACKAGE_BUGREPORT 
#undef PACKAGE_NAME 
#undef PACKAGE_STRING 
#undef PACKAGE_TARNAME 
#undef PACKAGE_VERSION 
#undef PACKAGE_BUGREPORT 

#include "ompi/include/ompi/constants.h"

/* undefine the stuff set by ompi */
#undef PACKAGE_BUGREPORT 
#undef PACKAGE_NAME 
#undef PACKAGE_STRING 
#undef PACKAGE_TARNAME 
#undef PACKAGE_VERSION 
#undef PACKAGE_BUGREPORT 

