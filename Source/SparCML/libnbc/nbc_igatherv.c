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

/* an gatherv schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Igatherv=PNBC_Igatherv
#define NBC_Igatherv PNBC_Igatherv
#endif
int NBC_Igatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle) {
  int rank, p, res, i;
  MPI_Aint rcvext;
  NBC_Schedule *schedule;
  char *rbuf, inplace;
  
  NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  
  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Comm_size(comm, &p);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Type_extent(recvtype, &rcvext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }

  schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
  if (NULL == schedule) { printf("Error in malloc() (%i)\n", res); return res; }

  handle->tmpbuf=NULL;
 
  res = NBC_Sched_create(schedule);
  if(res != NBC_OK) { printf("Error in NBC_Sched_create (%i)\n", res); return res; }

  /* send to root */
  if(rank != root) {
    /* send msg to root */
    res = NBC_Sched_send(sendbuf, false, sendcount, sendtype, root, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
  } else {
    for(i=0;i<p;i++) {
      rbuf = ((char *)recvbuf) + (displs[i]*rcvext);
      if(i == root) {
        if(!inplace) {
          /* if I am the root - just copy the message */
          res = NBC_Copy(sendbuf, sendcount, sendtype, rbuf, recvcounts[i], recvtype, comm);
          if (NBC_OK != res) { printf("Error in NBC_Copy() (%i)\n", res); return res; }
        }
      } else {
        /* root receives message to the right buffer */
        res = NBC_Sched_recv(rbuf, false, recvcounts[i], recvtype, i, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
      }
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
NBC_F77_ALLFUNC_(nbc_igatherv,NBC_IGATHERV,(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcounts, int *displs, int *recvtype, int *root, int *fcomm,
    int *fhandle, int *ierr));
#pragma weak NBC_IGATHERV = nbc_igatherv_f
#pragma weak nbc_igatherv = nbc_igatherv_f
#pragma weak nbc_igatherv_ = nbc_igatherv_f
#pragma weak nbc_igatherv__ = nbc_igatherv_f
#pragma weak PNBC_IGATHERV = nbc_igatherv_f
#pragma weak pnbc_igatherv = nbc_igatherv_f
#pragma weak pnbc_igatherv_ = nbc_igatherv_f
#pragma weak pnbc_igatherv__ = nbc_igatherv_f
void nbc_igatherv_f(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcounts, int *displs, int *recvtype, int *root, int *fcomm,
    int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_igatherv,NBC_IGATHERV)(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcounts, int *displs, int *recvtype, int *root, int *fcomm,
    int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_igatherv,NBC_IGATHERV)(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcounts, int *displs, int *recvtype, int *root, int *fcomm,
    int *fhandle, int *ierr)  {
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
  *ierr = NBC_Igatherv(sendbuf, *sendcount, stype, recvbuf, recvcounts, displs, rtype, *root, comm, handle);
}
#ifdef __cplusplus
}
#endif
