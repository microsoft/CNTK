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

static __inline__ int bcast_sched_binomial(int rank, int p, int root, NBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype);
static __inline__ int bcast_sched_linear(int rank, int p, int root, NBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype);
static __inline__ int bcast_sched_chain(int rank, int p, int root, NBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype, int fragsize, int size);

#ifdef NBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int NBC_Bcast_args_compare(NBC_Bcast_args *a, NBC_Bcast_args *b, void *param) {

	if( (a->buffer == b->buffer) && 
      (a->count == b->count) && 
      (a->datatype == b->datatype) &&
      (a->root == b->root) ) {
    return  0;
  }
	if( a->buffer < b->buffer ) {	
    return -1;
	}
	return +1;
}
#endif

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Ibcast=PNBC_Ibcast
#define NBC_Ibcast PNBC_Ibcast
#endif

int NBC_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, NBC_Handle* handle) {
  int rank, p, res, size, segsize;
  NBC_Schedule *schedule;
#ifdef NBC_CACHE_SCHEDULE
  NBC_Bcast_args *args, *found, search;
#endif
  enum { NBC_BCAST_LINEAR, NBC_BCAST_BINOMIAL, NBC_BCAST_CHAIN } alg;
  
  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Comm_size(comm, &p);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Type_size(datatype, &size);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_size() (%i)\n", res); return res; }
  
  segsize = 16384;
  /* algorithm selection */
  if(p <= 4) {
    alg = NBC_BCAST_LINEAR;
  } else if(size*count < 65536) {
    alg = NBC_BCAST_BINOMIAL;
  } else if(size*count < 524288) {
    alg = NBC_BCAST_CHAIN;
    segsize = 16384/2;
  } else {
    alg = NBC_BCAST_CHAIN;
    segsize = 65536/2;
  }

  handle->tmpbuf=NULL;

#ifdef NBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.buffer=buffer;
  search.count=count;
  search.datatype=datatype;
  search.root=root;
  found = (NBC_Bcast_args*)hb_tree_search((hb_tree*)handle->comminfo->NBC_Dict[NBC_BCAST], &search);
  if(found == NULL) {
#endif
    schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
    
    res = NBC_Sched_create(schedule);
    if(res != NBC_OK) { printf("Error in NBC_Sched_create, res = %i\n", res); return res; }

    switch(alg) {
      case NBC_BCAST_LINEAR:
        res = bcast_sched_linear(rank, p, root, schedule, buffer, count, datatype);
        break;
      case NBC_BCAST_BINOMIAL:
        res = bcast_sched_binomial(rank, p, root, schedule, buffer, count, datatype);
        break;
      case NBC_BCAST_CHAIN:
        res = bcast_sched_chain(rank, p, root, schedule, buffer, count, datatype, segsize, size);
        break;
    }
    if (NBC_OK != res) { printf("Error in Schedule creation() (%i)\n", res); return res; }
    
    res = NBC_Sched_commit(schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Bcast_args*)malloc(sizeof(NBC_Bcast_args));
    args->buffer=buffer;
    args->count=count;
    args->datatype=datatype;
    args->root=root;
    args->schedule=schedule;
	  res = hb_tree_insert ((hb_tree*)handle->comminfo->NBC_Dict[NBC_BCAST], args, args, 0);
    if(res != 0) printf("error in dict_insert() (%i)\n", res);
    /* increase number of elements for A2A */
    if(++handle->comminfo->NBC_Dict_size[NBC_BCAST] > NBC_SCHED_DICT_UPPER) {
      NBC_SchedCache_dictwipe((hb_tree*)handle->comminfo->NBC_Dict[NBC_BCAST], &handle->comminfo->NBC_Dict_size[NBC_BCAST]);
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

/* better binomial bcast 
 * working principle:
 * - each node gets a virtual rank vrank
 * - the 'root' node get vrank 0 
 * - node 0 gets the vrank of the 'root'
 * - all other ranks stay identical (they do not matter)
 *
 * Algorithm:
 * - each node with vrank > 2^r and vrank < 2^r+1 receives from node
 *   vrank - 2^r (vrank=1 receives from 0, vrank 0 receives never)
 * - each node sends each round r to node vrank + 2^r
 * - a node stops to send if 2^r > commsize  
 */
#define RANK2VRANK(rank, vrank, root) \
{ \
  vrank = rank; \
  if (rank == 0) vrank = root; \
  if (rank == root) vrank = 0; \
}
#define VRANK2RANK(rank, vrank, root) \
{ \
  rank = vrank; \
  if (vrank == 0) rank = root; \
  if (vrank == root) rank = 0; \
}
static __inline__ int bcast_sched_binomial(int rank, int p, int root, NBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype) {
  int maxr, vrank, peer, r, res;
  
  maxr = (int)ceil((log(p)/LOG2));

  RANK2VRANK(rank, vrank, root);

  /* receive from the right hosts  */
  if(vrank != 0) {
    for(r=0; r<maxr; r++) {
      if((vrank >= (1<<r)) && (vrank < (1<<(r+1)))) {
        VRANK2RANK(peer, vrank-(1<<r), root);
        res = NBC_Sched_recv(buffer, false, count, datatype, peer, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
      }
    }
    res = NBC_Sched_barrier(schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
  }

  /* now send to the right hosts */
  for(r=0; r<maxr; r++) {
    if(((vrank + (1<<r) < p) && (vrank < (1<<r))) || (vrank == 0)) {
      VRANK2RANK(peer, vrank+(1<<r), root);
      res = NBC_Sched_send(buffer, false, count, datatype, peer, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
    }
  }

  return NBC_OK;
}

/* simple linear MPI_Ibcast */
static __inline__ int bcast_sched_linear(int rank, int p, int root, NBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype) {
  int peer, res;  

  /* send to all others */
  if(rank == root) {
    for (peer=0; peer<p;peer++) {
      if(peer != root) {
        /* send msg to peer */
        res = NBC_Sched_send(buffer, false, count, datatype, peer, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      }
    }
  } else {
    /* recv msg from root */
    res = NBC_Sched_recv(buffer, false, count, datatype, root, schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
  }
  
  return NBC_OK;
}

/* simple chained MPI_Ibcast */
static __inline__ int bcast_sched_chain(int rank, int p, int root, NBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype, int fragsize, int size) {
  int res, vrank, rpeer, speer, numfrag, fragnum, fragcount, thiscount;
  MPI_Aint ext;
  char *buf;

  RANK2VRANK(rank, vrank, root);
  VRANK2RANK(rpeer, vrank-1, root);
  VRANK2RANK(speer, vrank+1, root);
  res = MPI_Type_extent(datatype, &ext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
  
  if(count == 0) return NBC_OK;

  numfrag = count*size/fragsize;
  if((count*size)%fragsize != 0) numfrag++;
  fragcount = count/numfrag;
  /*if(!rank) printf("numfrag: %i, count: %i, size: %i, fragcount: %i\n", numfrag, count, size, fragcount);*/
  
  for(fragnum = 0; fragnum < numfrag; fragnum++) {
    buf = (char*)buffer+fragnum*fragcount*ext;
    thiscount = fragcount;
    if(fragnum == numfrag-1) {
      /* last fragment may not be full */
      thiscount = count-fragcount*fragnum;
    }

    /* root does not receive */
    if(vrank != 0) {
      res = NBC_Sched_recv(buf, false, thiscount, datatype, rpeer, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
      res = NBC_Sched_barrier(schedule);
    }

    /* last rank does not send */
    if(vrank != p-1) {
      res = NBC_Sched_send(buf, false, thiscount, datatype, speer, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      /* this barrier here seems awaward but isn't!!!! */
      if(vrank == 0) res = NBC_Sched_barrier(schedule);
    }
  }
  
  return NBC_OK;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_ibcast,NBC_IBCAST,(void *buf, int *count, int *datatype, int *root, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_IBCAST = nbc_ibcast_f
#pragma weak nbc_ibcast = nbc_ibcast_f
#pragma weak nbc_ibcast_ = nbc_ibcast_f
#pragma weak nbc_ibcast__ = nbc_ibcast_f
#pragma weak PNBC_IBCAST = nbc_ibcast_f
#pragma weak pnbc_ibcast = nbc_ibcast_f
#pragma weak pnbc_ibcast_ = nbc_ibcast_f
#pragma weak pnbc_ibcast__ = nbc_ibcast_f
void nbc_ibcast_f(void *buf, int *count, int *datatype, int *root, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_ibcast,NBC_IBCAST)(void *buf, int *count, int *datatype, int *root, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_ibcast,NBC_IBCAST)(void *buf, int *count, int *datatype, int *root, int *fcomm, int *fhandle, int *ierr)  {
#endif
  MPI_Datatype dtype;
  MPI_Comm comm;
  NBC_Handle *handle;

  /* this is the only MPI-2 we need :-( */
  dtype = MPI_Type_f2c(*datatype);
  comm = MPI_Comm_f2c(*fcomm);

  /* create a new handle in handle table */
  NBC_Create_fortran_handle(fhandle, &handle);

  /* call NBC function */
  *ierr = NBC_Ibcast(buf, *count, dtype, *root, comm, handle);
}
#ifdef __cplusplus
}
#endif
