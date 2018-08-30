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

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Ineighbor_xchgv=PNBC_Ineighbor_xchgv
#define NBC_Ineighbor_xchgv PNBC_Ineighbor_xchgv
#endif

int NBC_Ineighbor_xchgv(void *sbuf, int *scounts, int *sdispls, MPI_Datatype stype,
        void *rbuf, int *rcounts, int *rdispls, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle) {
  int rank, res;
  MPI_Aint sndext, rcvext;
  char inplace;
  NBC_Schedule *schedule;

  NBC_IN_PLACE(sbuf, rbuf, inplace);
  if(inplace) {
    printf("NBC_Ineighbor_xchgv: INPLACE is not supported yet!\n");
    return NBC_NOT_IMPLEMENTED;
  }
  
  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Type_extent(stype, &sndext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
  res = MPI_Type_extent(rtype, &rcvext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
  
  handle->tmpbuf=NULL;

  schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
  
  res = NBC_Sched_create(schedule);
  if(res != NBC_OK) { printf("Error in NBC_Sched_create, res = %i\n", res); return res; }

  {
    int scount, rcount;

    int indeg, outdeg, wgtd, i, *srcs, *dsts;
    res = NBC_Comm_neighbors_count(comm, &indeg, &outdeg, &wgtd);
    if(res != NBC_OK) return res;
    srcs = (int*)malloc(sizeof(int)*indeg);
    dsts = (int*)malloc(sizeof(int)*outdeg);
    res = NBC_Comm_neighbors(comm, indeg, srcs, MPI_UNWEIGHTED, outdeg, dsts, MPI_UNWEIGHTED);
    if(res != NBC_OK) return res;

    /* simply loop over neighbors and post send/recv operations */
    for(i = 0; i < indeg; i++) {
      res = NBC_Sched_recv((char*)rbuf+rdispls[i]*rcvext, false, rcounts[i], rtype, srcs[i], schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
    }
    for(i = 0; i < outdeg; i++) {
      res = NBC_Sched_send((char*)sbuf+sdispls[i]*sndext, false, scounts[i], stype, dsts[i], schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
    }
   
  }
    
  res = NBC_Sched_commit(schedule);
  if (NBC_OK != res) { printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
  
  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { printf("Error in NBC_Start() (%i)\n", res); return res; }
  
  return NBC_OK;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_ineighbor_xchgv,NBC_INEIGHBOR_XCHGV,(void *sbuf, int *scounts, int *sdispls, int *stype, void *rbuf, int *rcounts,
        int *rdispls, int *rtype, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_INEIGHBOR_XCHGV = nbc_ineighbor_xchgv_f
#pragma weak nbc_ineighbor_xchgv = nbc_ineighbor_xchgv_f
#pragma weak nbc_ineighbor_xchgv_ = nbc_ineighbor_xchgv_f
#pragma weak nbc_ineighbor_xchgv__ = nbc_ineighbor_xchgv_f
#pragma weak PNBC_INEIGHBOR_XCHGV = nbc_ineighbor_xchgv_f
#pragma weak pnbc_ineighbor_xchgv = nbc_ineighbor_xchgv_f
#pragma weak pnbc_ineighbor_xchgv_ = nbc_ineighbor_xchgv_f
#pragma weak pnbc_ineighbor_xchgv__ = nbc_ineighbor_xchgv_f
void nbc_ineighbor_xchgv_f(void *sbuf, int *scounts, int *sdispls, int *stype, void *rbuf, int *rcounts,
        int *rdispls, int *rtype, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_ineighbor_xchgv,NBC_INEIGHBOR_XCHGV)(void *sbuf, int *scounts, int *sdispls, int *stype, void *rbuf, int *rcounts,
        int *rdispls, int *rtype, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_ineighbor_xchgv,NBC_INEIGHBOR_XCHGV)(void *sbuf, int *scounts, int *sdispls,  int *stype, void *rbuf, int *rcounts,
        int *rdispls, int *rtype, int *fcomm, int *fhandle, int *ierr) {
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
  *ierr = NBC_Ineighbor_xchgv(sbuf, scounts, sdispls, sdtype, rbuf, rcounts,
           rdispls, rdtype, comm, handle);
}
#ifdef __cplusplus
}
#endif
