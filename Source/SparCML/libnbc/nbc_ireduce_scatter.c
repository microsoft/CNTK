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

/* an reduce_csttare schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

/* binomial reduce to rank 0 followed by a linear scatter ...
 *
 * Algorithm:
 * pairwise exchange
 * round r: 
 *  grp = rank % 2^r
 *  if grp == 0: receive from rank + 2^(r-1) if it exists and reduce value
 *  if grp == 1: send to rank - 2^(r-1) and exit function
 *  
 * do this for R=log_2(p) rounds
 *    
 */

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Ireduce_scatter=PNBC_Ireduce_scatter
#define NBC_Ireduce_scatter PNBC_Ireduce_scatter
#endif
int NBC_Ireduce_scatter(void* sendbuf, void* recvbuf, int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, NBC_Handle* handle) {
  int peer, rank, maxr, p, r, res, count, offset, firstred;
  MPI_Aint ext;
  char *redbuf, *sbuf, inplace;
  NBC_Schedule *schedule;
  
  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Comm_size(comm, &p);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_size() (%i)\n", res); return res; }
  MPI_Type_extent(datatype, &ext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
  
  schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
  if (NULL == schedule) { printf("Error in malloc()\n"); return NBC_OOR; }

  res = NBC_Sched_create(schedule);
  if(res != NBC_OK) { printf("Error in NBC_Sched_create (%i)\n", res); return res; }

  maxr = (int)ceil((log(p)/LOG2));

  count = 0;
  for(r=0;r<p;r++) count += recvcounts[r];
  
  handle->tmpbuf = malloc(ext*count*2);
  if(handle->tmpbuf == NULL) { printf("Error in malloc()\n"); return NBC_OOR; }

  redbuf = ((char*)handle->tmpbuf)+(ext*count);

  /* copy data to redbuf if we only have a single node */
  if((p==1) && !inplace) {
    res = NBC_Copy(sendbuf, count, datatype, redbuf, count, datatype, comm);
    if (NBC_OK != res) { printf("Error in NBC_Copy() (%i)\n", res); return res; }
  }
  
  firstred = 1;
  for(r=1; r<=maxr; r++) {
    if((rank % (1<<r)) == 0) {
      /* we have to receive this round */
      peer = rank + (1<<(r-1));
      if(peer<p) {
        res = NBC_Sched_recv(0, true, count, datatype, peer, schedule);
        if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
        /* we have to wait until we have the data */
        res = NBC_Sched_barrier(schedule);
        if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
        if(firstred) {
          /* take reduce data from the sendbuf in the first round -> save copy */
          res = NBC_Sched_op(redbuf-(unsigned long)handle->tmpbuf, true, sendbuf, false, 0, true, count, datatype, op, schedule);
          firstred = 0;
        } else {
          /* perform the reduce in my local buffer */
          res = NBC_Sched_op(redbuf-(unsigned long)handle->tmpbuf, true, redbuf-(unsigned long)handle->tmpbuf, true, 0, true, count, datatype, op, schedule);
        }
        if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_op() (%i)\n", res); return res; }
        /* this cannot be done until handle->tmpbuf is unused :-( */
        res = NBC_Sched_barrier(schedule);
        if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
      }
    } else {
      /* we have to send this round */
      peer = rank - (1<<(r-1));
      if(firstred) {
        /* we have to send the senbuf */
        res = NBC_Sched_send(sendbuf, false, count, datatype, peer, schedule);
      } else {
        /* we send an already reduced value from redbuf */
        res = NBC_Sched_send(redbuf-(unsigned long)handle->tmpbuf, true, count, datatype, peer, schedule);
      }
      if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      /* leave the game */
      break;
    }
  }
  
  res = NBC_Sched_barrier(schedule);
  if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }

  /* rank 0 is root and sends - all others receive */
  if(rank != 0) {
    res = NBC_Sched_recv(recvbuf, false, recvcounts[rank], datatype, 0, schedule);
   if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
  }

  if(rank == 0) {
    offset = 0;
    for(r=1;r<p;r++) {
      offset += recvcounts[r-1];
      sbuf = ((char *)redbuf) + (offset*ext);
      /* root sends the right buffer to the right receiver */
      res = NBC_Sched_send(sbuf-(unsigned long)handle->tmpbuf, true, recvcounts[r], datatype, r, schedule);
      if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
    }
    res = NBC_Sched_copy(redbuf-(unsigned long)handle->tmpbuf, true, recvcounts[0], datatype, recvbuf, false, recvcounts[0], datatype, schedule);
    if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_copy() (%i)\n", res); return res; }
  }

  /*NBC_PRINT_SCHED(*schedule);*/
  
  res = NBC_Sched_commit(schedule);
  if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
  
  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Start() (%i)\n", res); return res; }
  
  /* tmpbuf is freed with the handle */
  return NBC_OK;
}


#ifdef __cplusplus
extern "C" {
#endif
/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_ireduce_scatter,NBC_IREDUCE_SCATTER,(void *sendbuf, void *recvbuf, int *recvcounts, int *datatype, int *fop, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_IREDUCE_SCATTER = nbc_ireduce_scatter_f
#pragma weak nbc_ireduce_scatter = nbc_ireduce_scatter_f
#pragma weak nbc_ireduce_scatter_ = nbc_ireduce_scatter_f
#pragma weak nbc_ireduce_scatter__ = nbc_ireduce_scatter_f
#pragma weak PNBC_IREDUCE_SCATTER = nbc_ireduce_scatter_f
#pragma weak pnbc_ireduce_scatter = nbc_ireduce_scatter_f
#pragma weak pnbc_ireduce_scatter_ = nbc_ireduce_scatter_f
#pragma weak pnbc_ireduce_scatter__ = nbc_ireduce_scatter_f
void nbc_ireduce_scatter_f(void *sendbuf, void *recvbuf, int *recvcounts, int *datatype, int *fop, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_ireduce_scatter,NBC_IREDUCE_SCATTER)(void *sendbuf, void *recvbuf, int *recvcounts, int *datatype, int *fop, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_ireduce_scatter,NBC_IREDUCE_SCATTER)(void *sendbuf, void *recvbuf, int *recvcounts, int *datatype, int *fop, int *fcomm, int *fhandle, int *ierr)  {
#endif
  MPI_Datatype dtype;
  MPI_Comm comm;
  MPI_Op op;
  NBC_Handle *handle;

  /* this is the only MPI-2 we need :-( */
  dtype = MPI_Type_f2c(*datatype);
  comm = MPI_Comm_f2c(*fcomm);
  op = MPI_Op_f2c(*fop);

  /* create a new handle in handle table */
  NBC_Create_fortran_handle(fhandle, &handle);

  /* call NBC function */
  *ierr = NBC_Ireduce_scatter(sendbuf, recvbuf, recvcounts, dtype, op, comm, handle);
}
#ifdef __cplusplus
}
#endif
