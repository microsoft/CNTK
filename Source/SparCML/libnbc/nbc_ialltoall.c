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

static __inline__ int a2a_sched_linear(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule *schedule, void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
static __inline__ int a2a_sched_pairwise(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule *schedule, void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
static __inline__ int a2a_sched_diss(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule, void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle);

#ifdef NBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int NBC_Alltoall_args_compare(NBC_Alltoall_args *a, NBC_Alltoall_args *b, void *param) {

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
#pragma weak NBC_Ialltoall=PNBC_Ialltoall
#define NBC_Ialltoall PNBC_Ialltoall
#endif

/* simple linear MPI_Ialltoall the (simple) algorithm just sends to all nodes */
int NBC_Ialltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle)  {
  int rank, p, res, a2asize, sndsize, datasize;
  NBC_Schedule *schedule;
  MPI_Aint rcvext, sndext;
#ifdef NBC_CACHE_SCHEDULE
  NBC_Alltoall_args *args, *found, search;
#endif
  char *rbuf, *sbuf, inplace;
  enum {NBC_A2A_LINEAR, NBC_A2A_PAIRWISE, NBC_A2A_DISS} alg;

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
  res = MPI_Type_size(sendtype, &sndsize);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_size() (%i)\n", res); return res; }

  /* algorithm selection */
  a2asize = sndsize*sendcount*p;
  /* this number is optimized for TCP on odin.cs.indiana.edu */
  if((p <= 8) && ((a2asize < 1<<17) || (sndsize*sendcount < 1<<12))) {
    /* just send as fast as we can if we have less than 8 peers, if the
     * total communicated size is smaller than 1<<17 *and* if we don't
     * have eager messages (msgsize < 1<<13) */
    alg = NBC_A2A_LINEAR;
  } else if(a2asize < (1<<12)*p) {
    /*alg = NBC_A2A_DISS;*/
    alg = NBC_A2A_LINEAR;
  } else
    alg = NBC_A2A_LINEAR; /*NBC_A2A_PAIRWISE;*/

  if(!inplace) {
    /* copy my data to receive buffer */
    rbuf = ((char *)recvbuf) + (rank*recvcount*rcvext);
    sbuf = ((char *)sendbuf) + (rank*sendcount*sndext);
    res = NBC_Copy(sbuf, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    if (NBC_OK != res) { printf("Error in NBC_Copy() (%i)\n", res); return res; }
  }

  /* allocate temp buffer if we need one */
  if(alg == NBC_A2A_DISS) {
    /* only A2A_DISS needs buffers */
    if(NBC_Type_intrinsic(sendtype)) {
      datasize = sndext*sendcount;
    } else {
      res = MPI_Pack_size(sendcount, sendtype, comm, &datasize);
      if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack_size() (%i)\n", res); return res; }
    }
    /* allocate temporary buffers */
    if(p % 2 == 0) {
      handle->tmpbuf=malloc(datasize*p*2);
    } else {
      /* we cannot divide p by two, so alloc more to be safe ... */
      handle->tmpbuf=malloc(datasize*(p/2+1)*2*2);
    }

    /* phase 1 - rotate n data blocks upwards into the tmpbuffer */
    if(NBC_Type_intrinsic(sendtype)) {
      /* contiguous - just copy (1st copy) */
      memcpy(handle->tmpbuf, (char*)sendbuf+datasize*rank, datasize*(p-rank));
      if(rank != 0) memcpy((char*)handle->tmpbuf+datasize*(p-rank), sendbuf, datasize*(rank));
    } else {
      int pos=0;

      /* non-contiguous - pack */
      res = MPI_Pack((char*)sendbuf+rank*sendcount*sndext, (p-rank)*sendcount, sendtype, handle->tmpbuf, (p-rank)*datasize, &pos, comm);
      if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack() (%i)\n", res); return res; }
      if(rank != 0) {
        pos = 0;
        MPI_Pack(sendbuf, rank*sendcount, sendtype, (char*)handle->tmpbuf+datasize*(p-rank), rank*datasize, &pos, comm);
        if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack() (%i)\n", res); return res; }
      }
    }
  } else {
    handle->tmpbuf=NULL;
  }

#ifdef NBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.sendbuf=sendbuf;
  search.sendcount=sendcount;
  search.sendtype=sendtype;
  search.recvbuf=recvbuf;
  search.recvcount=recvcount;
  search.recvtype=recvtype;
  found = (NBC_Alltoall_args*)hb_tree_search((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLTOALL], &search);
  if(found == NULL) {
#endif
    /* not found - generate new schedule */
    schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
    if (NULL == schedule) { printf("Error in malloc()\n"); return res; }

    res = NBC_Sched_create(schedule);
    if(res != NBC_OK) { printf("Error in NBC_Sched_create (%i)\n", res); return res; }
    
    switch(alg) {
      case NBC_A2A_LINEAR: 
        res = a2a_sched_linear(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        break;
      case NBC_A2A_DISS:
        res = a2a_sched_diss(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, handle);
        break;
      case NBC_A2A_PAIRWISE:
        res = a2a_sched_pairwise(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        break;
    }
    
    if (NBC_OK != res) { return res; }

    res = NBC_Sched_commit(schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
 
#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Alltoall_args*)malloc(sizeof(NBC_Alltoall_args));
    args->sendbuf=sendbuf;
    args->sendcount=sendcount;
    args->sendtype=sendtype;
    args->recvbuf=recvbuf;
    args->recvcount=recvcount;
    args->recvtype=recvtype;
    args->schedule=schedule;
	  res = hb_tree_insert ((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLTOALL], args, args, 0);
    if(res != 0) printf("error in dict_insert() (%i)\n", res);
    /* increase number of elements for A2A */
    if(++handle->comminfo->NBC_Dict_size[NBC_ALLTOALL] > NBC_SCHED_DICT_UPPER) {
      NBC_SchedCache_dictwipe((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLTOALL], &handle->comminfo->NBC_Dict_size[NBC_ALLTOALL]);
      /*if(!rank) printf("[%i] removing %i elements - new size: %i \n", rank, SCHED_DICT_UPPER-SCHED_DICT_LOWER, handle->comminfo->NBC_Alltoall_size);*/
    }
    /*if(!rank) printf("[%i] added new schedule to tree - number %i\n", rank, handle->comminfo->NBC_Dict_size[NBC_ALLTOALL]);*/
  } else {
    /* found schedule */
    schedule=found->schedule;
  }
#endif

  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { printf("Error in NBC_Start() (%i)\n", res); return res; }
 
  return NBC_OK;
}

static __inline__ int a2a_sched_pairwise(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule, void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
  int res, r, sndpeer, rcvpeer;
  char *rbuf, *sbuf;

  res = NBC_OK;
  if(p < 2) return res;
  
  for(r=1;r<p;r++) {
    
    sndpeer = (rank+r)%p;
    rcvpeer = (rank-r+p)%p;
    
    rbuf = ((char *) recvbuf) + (rcvpeer*recvcount*rcvext);
    res = NBC_Sched_recv(rbuf, false, recvcount, recvtype, rcvpeer, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
    sbuf = ((char *) sendbuf) + (sndpeer*sendcount*sndext);
    res = NBC_Sched_send(sbuf, false, sendcount, sendtype, sndpeer, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
    
    if (r < p) {
      res = NBC_Sched_barrier(schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_barr() (%i)\n", res); return res; }
    }
  }

  return res;
}

static __inline__ int a2a_sched_linear(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule, void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
  int res, r;
  char *rbuf, *sbuf;

  res = NBC_OK;

  for(r=0;r<p;r++) {
    /* easy algorithm */
    if ((r == rank)) { continue; }
    rbuf = ((char *) recvbuf) + (r*recvcount*rcvext);
    res = NBC_Sched_recv(rbuf, false, recvcount, recvtype, r, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
    sbuf = ((char *) sendbuf) + (r*sendcount*sndext);
    res = NBC_Sched_send(sbuf, false, sendcount, sendtype, r, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
  }

  return res;
}

static __inline__ int a2a_sched_diss(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule, void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle) {
  int res, i, r, speer, rpeer, datasize, offset, virtp;
  char *rbuf, *rtmpbuf, *stmpbuf;

  res = NBC_OK;
  if(p < 2) return res;
 
  if(NBC_Type_intrinsic(sendtype)) {
    datasize = sndext*sendcount;
  } else {
    res = MPI_Pack_size(sendcount, sendtype, comm, &datasize);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack_size() (%i)\n", res); return res; }
  }
 
  /* allocate temporary buffers */
  if(p % 2 == 0) {
    rtmpbuf = (char*)handle->tmpbuf+datasize*p;
    stmpbuf = (char*)handle->tmpbuf+datasize*(p+p/2);
  } else {
    /* we cannot divide p by two, so alloc more to be safe ... */
    virtp = (p/2+1)*2;
    rtmpbuf = (char*)handle->tmpbuf+datasize*p;
    stmpbuf = (char*)handle->tmpbuf+datasize*(p+virtp/2);
  }

  /* phase 2 - communicate */
  /*printf("[%i] temp buffer is at %lu of size %i, maxround: %i\n", rank, (unsigned long)handle->tmpbuf, (int)datasize*p*(1<<maxround), maxround);*/
  for(r = 1; r < p; r<<=1) {
    offset = 0;
    for(i=1; i<p; i++) {
      /* test if bit r is set in rank number i */
      if((i&r) == r) {
        /* copy data to sendbuffer (2nd copy) - could be avoided using iovecs */
        /*printf("[%i] round %i: copying element %i to buffer %lu\n", rank, r, i, (unsigned long)(stmpbuf+offset));*/
        NBC_Sched_copy((void*)(long)(i*datasize), true, datasize, MPI_BYTE, stmpbuf+offset-(unsigned long)handle->tmpbuf, true, datasize, MPI_BYTE, schedule);
        offset += datasize;
      }
    }
 
    speer = ( rank + r) % p;
    /* add p because modulo does not work with negative values */
    rpeer = ((rank - r)+p) % p;
 
    /*printf("[%i] receiving %i bytes from host %i into rbuf %lu\n", rank, offset, rpeer, (unsigned long)rtmpbuf);*/
    res = NBC_Sched_recv(rtmpbuf-(unsigned long)handle->tmpbuf, true, offset, MPI_BYTE, rpeer, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
 
    /*printf("[%i] sending %i bytes to host %i from sbuf %lu\n", rank, offset, speer, (unsigned long)stmpbuf);*/
    res = NBC_Sched_send(stmpbuf-(unsigned long)handle->tmpbuf, true, offset, MPI_BYTE, speer, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }

    res = NBC_Sched_barrier(schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
    
    /* unpack from buffer */
    offset = 0;
    for(i=1; i<p; i++) {
      /* test if bit r is set in rank number i */
      if((i&r) == r) {
        /* copy data to tmpbuffer (3rd copy) - could be avoided using iovecs */
        NBC_Sched_copy(rtmpbuf+offset-(unsigned long)handle->tmpbuf, true, datasize, MPI_BYTE, (void*)(long)(i*datasize), true, datasize, MPI_BYTE, schedule);
        offset += datasize;
      }
    }
  }

  /* phase 3 - reorder - data is now in wrong order in handle->tmpbuf -
   * reorder it into recvbuf */
  for(i=0; i<p; i++) {
    rbuf = (char*)recvbuf+((rank-i+p)%p)*recvcount*rcvext;
    res = NBC_Sched_unpack((void*)(long)(i*datasize), true, recvcount, recvtype, rbuf, false, schedule);
    if (NBC_OK != res) { printf("MPI Error in NBC_Sched_pack() (%i)\n", res); return res; }
  }
    
  return NBC_OK;
}


#ifdef __cplusplus
extern "C" {
#endif
/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_ialltoall,NBC_IALLTOALL,(void *sendbuf, int *sendcount, int *sendtype,
    void *recvbuf, int *recvcount, int *recvtype, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_IALLTOALL = nbc_ialltoall_f
#pragma weak nbc_ialltoall = nbc_ialltoall_f
#pragma weak nbc_ialltoall_ = nbc_ialltoall_f
#pragma weak nbc_ialltoall__ = nbc_ialltoall_f
#pragma weak PNBC_IALLTOALL = nbc_ialltoall_f
#pragma weak pnbc_ialltoall = nbc_ialltoall_f
#pragma weak pnbc_ialltoall_ = nbc_ialltoall_f
#pragma weak pnbc_ialltoall__ = nbc_ialltoall_f
void nbc_ialltoall_f(void *sendbuf, int *sendcount, int *sendtype,
    void *recvbuf, int *recvcount, int *recvtype, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_ialltoall,NBC_IALLTOALL)(void *sendbuf, int *sendcount, int *sendtype,
    void *recvbuf, int *recvcount, int *recvtype, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_ialltoall,NBC_IALLTOALL)(void *sendbuf, int *sendcount, int *sendtype,
    void *recvbuf, int *recvcount, int *recvtype, int *fcomm, int *fhandle, int *ierr)  {
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
  *ierr = NBC_Ialltoall(sendbuf, *sendcount, stype, recvbuf, *recvcount, rtype, comm, handle);
}
#ifdef __cplusplus
}
#endif
