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

/* an alltoallv schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Ialltoallv=PNBC_Ialltoallv
#define NBC_Ialltoallv PNBC_Ialltoallv
#endif

/* simple linear Alltoallv */
int NBC_Ialltoallv(void* sendbuf, int *sendcounts, int *sdispls,
    MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *rdispls,
    MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle* handle) {
  
  int rank, p, res, i;
  MPI_Aint sndext, rcvext;
  NBC_Schedule *schedule;
  char *rbuf, *sbuf, inplace;
  
  NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  
  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res= MPI_Comm_size(comm, &p);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_size() (%i)\n", res); return res; }
  res = MPI_Type_extent(sendtype, &sndext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
  res = MPI_Type_extent(recvtype, &rcvext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }

  schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
  if (NULL == schedule) { printf("Error in malloc() (%i)\n", res); return res; }

  handle->tmpbuf=NULL;
  
  res = NBC_Sched_create(schedule);
  if(res != NBC_OK) { printf("Error in NBC_Sched_create (%i)\n", res); return res; }

  /* copy data to receivbuffer */
  if((sendcounts[rank] != 0) && !inplace) {
    rbuf = ((char *) recvbuf) + (rdispls[rank] * rcvext);
    sbuf = ((char *) sendbuf) + (sdispls[rank] * sndext);
    res = NBC_Copy(sbuf, sendcounts[rank], sendtype, rbuf, recvcounts[rank], recvtype, comm);
    if (NBC_OK != res) { printf("Error in NBC_Copy() (%i)\n", res); return res; }
  }

  for (i = 0; i < p; i++) {
    if (i == rank) { continue; }
    /* post all sends */
    if(sendcounts[i] != 0) {
      sbuf = ((char *) sendbuf) + (sdispls[i] * sndext);
      res = NBC_Sched_send(sbuf, false, sendcounts[i], sendtype, i, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
    }
    /* post all receives */
    if(recvcounts[i] != 0) {
      rbuf = ((char *) recvbuf) + (rdispls[i] * rcvext);
      res = NBC_Sched_recv(rbuf, false, recvcounts[i], recvtype, i, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
    }
  }

  /*NBC_PRINT_SCHED(*schedule);*/

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
NBC_F77_ALLFUNC_(nbc_ialltoallv,NBC_IALLTOALLV,(void *sendbuf, int *sendcounts, int *sdispls, int *sendtype,
    void *recvbuf, int *recvcounts, int *rdispls, int *recvtype, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_IALLTOALLV = nbc_ialltoallv_f
#pragma weak nbc_ialltoallv = nbc_ialltoallv_f
#pragma weak nbc_ialltoallv_ = nbc_ialltoallv_f
#pragma weak nbc_ialltoallv__ = nbc_ialltoallv_f
#pragma weak PNBC_IALLTOALLV = nbc_ialltoallv_f
#pragma weak pnbc_ialltoallv = nbc_ialltoallv_f
#pragma weak pnbc_ialltoallv_ = nbc_ialltoallv_f
#pragma weak pnbc_ialltoallv__ = nbc_ialltoallv_f
void nbc_ialltoallv_f(void *sendbuf, int *sendcounts, int *sdispls, int *sendtype,
    void *recvbuf, int *recvcounts, int *rdispls, int *recvtype, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_ialltoallv,NBC_IALLTOALLV)(void *sendbuf, int *sendcounts, int *sdispls, int *sendtype,
    void *recvbuf, int *recvcounts, int *rdispls, int *recvtype, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_ialltoallv,NBC_IALLTOALLV)(void *sendbuf, int *sendcounts, int *sdispls, int *sendtype,
    void *recvbuf, int *recvcounts, int *rdispls, int *recvtype, int *fcomm, int *fhandle, int *ierr)  {
#endif
  MPI_Datatype rtype, stype;
  MPI_Comm comm;
  NBC_Handle *handle;

  /* this is the only MPI-2 we need :-( */
  rtype = MPI_Type_f2c(*recvtype);
  stype = MPI_Type_f2c(*sendtype);
  comm = MPI_Comm_f2c(*fcomm);

  /* create a new handle in handle table */
  NBC_Create_fortran_handle(fhandle, &handle);

  /* call NBC function */
  *ierr = NBC_Ialltoallv(sendbuf, sendcounts, sdispls, stype, recvbuf, recvcounts, rdispls, rtype, comm, handle);
}
#ifdef __cplusplus
}
#endif
