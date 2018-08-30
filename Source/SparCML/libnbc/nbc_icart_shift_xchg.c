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
#include "nbc_internal.h"

#ifdef NBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int NBC_Icart_shift_xchg_args_compare(NBC_Icart_shift_xchg_args *a, NBC_Icart_shift_xchg_args *b, void *param) {

	if( (a->sbuf == b->sbuf) && 
      (a->scount == b->scount) && 
      (a->stype == b->stype) &&
      (a->rbuf == b->rbuf) && 
      (a->rcount == b->rcount) && 
      (a->rtype == b->rtype) &&
      (a->direction == b->direction) &&
      (a->disp== b->disp) ) {
    return  0;
  }
	if( a->sbuf < b->sbuf ) {	
    return -1;
	}
	return +1;
}
#endif

/* this is a new collective operation defined on a communicator. This
 * operation shifts a piece of data along the dimension direction
 * (0..ndims-1) in the direction indicated by disp in a communciator. 
 *
 */

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Icart_shift_xchg=PNBC_Icart_shift_xchg
#define NBC_Icart_shift_xchg PNBC_Icart_shift_xchg
#endif

int NBC_Icart_shift_xchg(void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype, int direction, int
        disp, MPI_Comm comm,  NBC_Handle* handle) {
  int res, speer, rpeer;
  MPI_Aint ext;
  char inplace;
  NBC_Schedule *schedule;
#ifdef NBC_CACHE_SCHEDULE
  NBC_Icart_shift_xchg_args *args, *found, search;
#endif

  NBC_IN_PLACE(sbuf, rbuf, inplace);
  
  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  
  handle->tmpbuf=NULL;

#ifdef NBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.sbuf=sbuf;
  search.scount=scount;
  search.stype=stype;
  search.rbuf=rbuf;
  search.rcount=rcount;
  search.rtype=rtype;
  search.direction=direction;
  search.disp=disp;
  found = (NBC_Icart_shift_xchg_args*)hb_tree_search((hb_tree*)handle->comminfo->NBC_Dict[NBC_CART_SHIFT_XCHG], &search);
  if(found == NULL) {
#endif
    schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
    
    res = NBC_Sched_create(schedule);
    if(res != NBC_OK) { printf("Error in NBC_Sched_create, res = %i\n", res); return res; }

    res = MPI_Cart_shift(comm, direction, disp, &rpeer, &speer);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Cart_shift() (%i)\n", res); return res; }

    /* create schedule - the non-inplace case is easy - the inplace-case
     * needs an extra buffer :-( */
    if(inplace) { /* we need an extra buffer to be deadlock-free */
      res = MPI_Type_extent(rtype, &ext);
      if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
      handle->tmpbuf = malloc(ext*scount);

      res = NBC_Sched_recv(0, true, rcount, rtype, rpeer, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
      res = NBC_Sched_send(sbuf, false, scount, stype, speer, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      
      res = NBC_Sched_barrier(schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
      res = NBC_Sched_copy(0, true, rcount, rtype, rbuf, false, rcount, rtype, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_copy() (%i)\n", res); return res; }
    } else { /* this case is the easy case */
      res = NBC_Sched_recv(rbuf, false, rcount, rtype, rpeer, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
      res = NBC_Sched_send(sbuf, false, scount, stype, speer, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
    }
    
    res = NBC_Sched_commit(schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Icart_shift_xchg_args*)malloc(sizeof(NBC_Icart_shift_xchg_args));
    args->sbuf=sbuf;
    args->scount=scount;
    args->stype=stype;
    args->rbuf=rbuf;
    args->rcount=rcount;
    args->rtype=rtype;
    args->direction=direction;
    args->disp=disp;
    args->schedule=schedule;
	  res = hb_tree_insert ((hb_tree*)handle->comminfo->NBC_Dict[NBC_CART_SHIFT_XCHG], args, args, 0);
    if(res != 0) printf("error in dict_insert() (%i)\n", res);
    /* increase number of elements */
    if(++handle->comminfo->NBC_Dict_size[NBC_CART_SHIFT_XCHG] > NBC_SCHED_DICT_UPPER) {
      NBC_SchedCache_dictwipe((hb_tree*)handle->comminfo->NBC_Dict[NBC_CART_SHIFT_XCHG], &handle->comminfo->NBC_Dict_size[NBC_CART_SHIFT_XCHG]);
    }
  } else {
    /* found schedule */
    schedule=found->schedule;
  }
#endif
  
  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { printf("Error in NBC_Start() (%i)\n", res); return res; }
  
  return NBC_OK;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_icart_shift_xchg,NBC_ICART_SHIFT_XCHG,(void *sbuf, int *scount, int *stype, void *rbuf, int *rcount, int *rtype, int *direction, int *disp, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_ICART_SHIFT_XCHG = nbc_icart_shift_xchg_f
#pragma weak nbc_icart_shift_xchg = nbc_icart_shift_xchg_f
#pragma weak nbc_icart_shift_xchg_ = nbc_icart_shift_xchg_f
#pragma weak nbc_icart_shift_xchg__ = nbc_icart_shift_xchg_f
#pragma weak PNBC_ICART_SHIFT_XCHG = nbc_icart_shift_xchg_f
#pragma weak pnbc_icart_shift_xchg = nbc_icart_shift_xchg_f
#pragma weak pnbc_icart_shift_xchg_ = nbc_icart_shift_xchg_f
#pragma weak pnbc_icart_shift_xchg__ = nbc_icart_shift_xchg_f
void nbc_icart_shift_xchg_f(void *sbuf, int *scount, int *stype, void *rbuf, int *rcount,
        int *rtype, int *direction, int *disp, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_icart_shift_xchg,NBC_ICART_SHIFT_XCHG)(void *sbuf, int *scount, int *stype, void *rbuf, int *rcount,
        int *rtype, int *direction, int *disp, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_icart_shift_xchg,NBC_ICART_SHIFT_XCHG)(void *sbuf, int *scount, int *stype, void *rbuf, int *rcount,
        int *rtype, int *direction, int *disp, int *fcomm, int *fhandle, int *ierr) {
#endif
  MPI_Datatype sdtype, rdtype;
  MPI_Comm comm;
  NBC_Handle *handle;

  /* this is the only MPI-2 we need :-( */
  sdtype = MPI_Type_f2c(*stype);
  rdtype = MPI_Type_f2c(*rtype);
  comm = MPI_Comm_f2c(*fcomm);

  /* create a new handle in handle table */
  NBC_Create_fortran_handle(fhandle, &handle);

  /* call NBC function */
  *ierr = NBC_Icart_shift_xchg(sbuf, *scount, sdtype, rbuf, *rcount,
           rdtype, *direction, *disp, comm, handle);
}
#ifdef __cplusplus
}
#endif
