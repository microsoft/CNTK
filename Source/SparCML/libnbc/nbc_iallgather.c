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
int NBC_Allgather_args_compare(NBC_Allgather_args *a, NBC_Allgather_args *b, void *param) {

	if( (a->sendbuf == b->sendbuf) && 
      (a->sendcount == b->sendcount) && 
      (a->sendtype == b->sendtype) &&
      (a->recvbuf == b->recvbuf) &&
      (a->recvcount == b->recvcount) &&
      (a->recvtype == b->recvtype) ) {
    return  0;
  }
	if( a->sendbuf < b->sendbuf ) {	
    return -1;
	}
	return +1;
}
#endif

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Iallgather=PNBC_Iallgather
#define NBC_Iallgather PNBC_Iallgather
#endif

/* simple linear MPI_Iallgather
 * the algorithm uses p-1 rounds
 * each node sends the packet it received last round (or has in round 0) to it's right neighbor (modulo p)
 * each node receives from it's left (modulo p) neighbor */
int NBC_Iallgather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle) {
  int rank, p, res, r;
  MPI_Aint rcvext, sndext;
  NBC_Schedule *schedule;
  char *rbuf, *sbuf, inplace;
#ifdef NBC_CACHE_SCHEDULE
  NBC_Allgather_args *args, *found, search;
#endif

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  
  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Comm_size(comm, &p);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_size() (%i)\n", res); return res; }
  res = MPI_Type_extent(sendtype, &sndext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
  res = MPI_Type_extent(recvtype, &rcvext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }

  handle->tmpbuf = NULL;

  if(!((rank == 0) && inplace)) {
    /* copy my data to receive buffer */
    rbuf = ((char *)recvbuf) + (rank*recvcount*rcvext);
    res = NBC_Copy(sendbuf, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    if (NBC_OK != res) { printf("Error in NBC_Copy() (%i)\n", res); return res; }
  }

#ifdef NBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.sendbuf=sendbuf;
  search.sendcount=sendcount;
  search.sendtype=sendtype;
  search.recvbuf=recvbuf;
  search.recvcount=recvcount;
  search.recvtype=recvtype;
  found = (NBC_Allgather_args *)hb_tree_search((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLGATHER], &search);
  if(found == NULL) {
#endif
    schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
    if (NULL == schedule) { printf("Error in malloc()\n"); return res; }

    res = NBC_Sched_create(schedule);
    if(NBC_OK != res) { printf("Error in NBC_Sched_create, (%i)\n", res); return res; }
    
    sbuf = ((char *)recvbuf) + (rank*recvcount*rcvext);
    /* do p-1 rounds */
    for(r=0;r<p;r++) {
      if(r != rank) {
        /* recv from rank r */
        rbuf = ((char *)recvbuf) + r*(recvcount*rcvext);
        res = NBC_Sched_recv(rbuf, false, recvcount, recvtype, r, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
        /* send to rank r - not from the sendbuf to optimize MPI_IN_PLACE */
        res = NBC_Sched_send(sbuf, false, recvcount, recvtype, r, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      }
    }

    res = NBC_Sched_commit(schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }

#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Allgather_args*)malloc(sizeof(NBC_Allgather_args));
    args->sendbuf=sendbuf;
    args->sendcount=sendcount;
    args->sendtype=sendtype;
    args->recvbuf=recvbuf;
    args->recvcount=recvcount;
    args->recvtype=recvtype;
    args->schedule=schedule;
	  res = hb_tree_insert ((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLGATHER], args, args, 0);
    if(res != 0) printf("error in dict_insert() (%i)\n", res);
    /* increase number of elements for A2A */
    if(++handle->comminfo->NBC_Dict_size[NBC_ALLGATHER] > NBC_SCHED_DICT_UPPER) {
      NBC_SchedCache_dictwipe((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLGATHER], &handle->comminfo->NBC_Dict_size[NBC_ALLGATHER]);
    }
  } else {
    /* found schedule */
    schedule=found->schedule;
  }
#endif

  /*NBC_PRINT_SCHED(*schedule);*/
 
  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { printf("Error in NBC_Start() (%i)\n", res); return res; }
 
  return NBC_OK;
}


/* this is a new possible dissemination based allgather algorithm - we should
 * try it some time (big comm, small data) */
#if 0

static __inline__ void diss_unpack(int rank, int vrank, int round, int p, int *pos, void *tmpbuf, int datasize, int slotsize, void *recvbuf, int sendcount, MPI_Datatype sendtype, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Schedule *schedule) {
  int r, res;
  char *sbuf, *rbuf;
  
  sbuf = (char *)tmpbuf + (*pos*datasize);
  rbuf = (char *)recvbuf + (vrank*slotsize);
  printf("[%i] unpacking tmpbuf pos: %i (%lu) to rbuf elem: %i (%lu) - %i elems, datasize %i\n", rank, *pos, (unsigned long)sbuf, vrank, (unsigned long)rbuf, recvcount, datasize);
  res = NBC_Sched_unpack(sbuf, recvcount, recvtype, rbuf, schedule);
  if (NBC_OK != res) { printf("Error in NBC_Unpack() (%i)\n", res); }
  *pos=*pos+1;
  
  for(r=0; r<=round; r++) {
    if(r != 0) {
      diss_unpack(rank, (vrank-(1<<(r-1))+p)%p, r-1, p, pos, tmpbuf, datasize, slotsize, recvbuf, sendcount, sendtype, recvcount, recvtype, comm, schedule);
    }
  }
}

static __inline__ int a2a_sched_diss(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule, void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle) {
  int res, r, maxround, size, speer, rpeer, pos, datasize;
  char *sbuf, *rbuf;

  res = NBC_OK;
  if(p < 2) return res;
  
  maxround = (int)ceil((log(p)/LOG2));
  
  if(NBC_Type_intrinsic(sendtype)) {
    datasize = sndext*sendcount;
  } else {
    res = MPI_Pack_size(sendcount, sendtype, comm, &datasize);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack_size() (%i)\n", res); return res; }
  }
    
  /* tmpbuf is probably bigger than p -> next power of 2 */
  handle->tmpbuf=malloc(datasize*(1<<maxround));

  /* copy my send - data to temp send/recv buffer */
  sbuf = ((char *)sendbuf) + (rank*sendcount*sndext);
  /* pack send buffer */
  if(NBC_Type_intrinsic(sendtype)) {
    /* it is contiguous - we can just memcpy it */
    memcpy(handle->tmpbuf, sbuf, datasize);
  } else {
    pos = 0;
    res = MPI_Pack(sbuf, sendcount, sendtype, handle->tmpbuf, datasize, &pos, comm);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack() (%i)\n", res); return res; }
  }
  
  printf("[%i] receive buffer is at %lu of size %i, maxround: %i\n", rank, (unsigned long)handle->tmpbuf, (int)sndext*sendcount*(1<<maxround), maxround);
  for(r = 0; r < maxround; r++) {
    size = datasize*(1<<r); /* size doubles every round */
    rbuf = (char*)handle->tmpbuf+size;
    sbuf = (char*)handle->tmpbuf;
    
    speer = (rank + (1<<r)) % p;
    /* add p because modulo does not work with negative values */
    rpeer = ((rank - (1<<r))+p) % p;
    
    printf("[%i] receiving %i bytes from host %i into rbuf %lu\n", rank, size, rpeer, (unsigned long)rbuf);
    res = NBC_Sched_recv(rbuf, size, MPI_BYTE, rpeer, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
    
    printf("[%i] sending %i bytes to host %i from sbuf %lu\n", rank, size, speer, (unsigned long)sbuf);
    res = NBC_Sched_send(sbuf, size, MPI_BYTE, speer, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }

    res = NBC_Sched_barrier(schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
  }
    
  pos = 0;
  diss_unpack(rank, rank, r, p, &pos, handle->tmpbuf, datasize, recvcount*rcvext, recvbuf, sendcount, sendtype, recvcount, recvtype, comm, schedule);
    
  return NBC_OK;
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_iallgather,NBC_IALLGATHER,(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcount, int *recvtype, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_IALLGATHER = nbc_iallgather_f
#pragma weak nbc_iallgather = nbc_iallgather_f
#pragma weak nbc_iallgather_ = nbc_iallgather_f
#pragma weak nbc_iallgather__ = nbc_iallgather_f
#pragma weak PNBC_IALLGATHER = nbc_iallgather_f
#pragma weak pnbc_iallgather = nbc_iallgather_f
#pragma weak pnbc_iallgather_ = nbc_iallgather_f
#pragma weak pnbc_iallgather__ = nbc_iallgather_f
void nbc_iallgather_f(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcount, int *recvtype, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_iallgather,NBC_IALLGATHER)(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcount, int *recvtype, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_iallgather,NBC_IALLGATHER)(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcount, int *recvtype, int *fcomm, int *fhandle, int *ierr)  {
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
  *ierr = NBC_Iallgather(sendbuf, *sendcount, stype, recvbuf, *recvcount, rtype, comm, handle);
}

#ifdef __cplusplus
}
#endif
