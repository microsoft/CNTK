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

static __inline__ int red_sched_binomial(int rank, int p, int root, void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, void *redbuf, NBC_Schedule *schedule, NBC_Handle *handle);
static __inline__ int red_sched_chain(int rank, int p, int root, void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int ext, int size, NBC_Schedule *schedule, NBC_Handle *handle, int fragsize);

#ifdef NBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int NBC_Reduce_args_compare(NBC_Reduce_args *a, NBC_Reduce_args *b, void *param) {

	if( (a->sendbuf == b->sendbuf) && 
      (a->recvbuf == b->recvbuf) &&
      (a->count == b->count) && 
      (a->datatype == b->datatype) &&
      (a->op == b->op) &&
      (a->root == b->root) ) {
    return  0;
  }
	if( a->sendbuf < b->sendbuf ) {	
    return -1;
	}
	return +1;
}
#endif

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Ireduce=PNBC_Ireduce
#define NBC_Ireduce PNBC_Ireduce
#endif

/* the non-blocking reduce */
int NBC_Ireduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, NBC_Handle* handle) {
  int rank, p, res, segsize, size;
  MPI_Aint ext;
  NBC_Schedule *schedule;
  char *redbuf=NULL, inplace;
#ifdef NBC_CACHE_SCHEDULE
  NBC_Reduce_args *args, *found, search;
#endif
  enum { NBC_RED_BINOMIAL, NBC_RED_CHAIN } alg;
  
  NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  
  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Comm_size(comm, &p);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_size() (%i)\n", res); return res; }
  res = MPI_Type_extent(datatype, &ext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
  res = MPI_Type_size(datatype, &size);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_size() (%i)\n", res); return res; }
  
  /* only one node -> copy data */
  if((p == 1) && !inplace) {
    res = NBC_Copy(sendbuf, count, datatype, recvbuf, count, datatype, comm);
    if (NBC_OK != res) { printf("Error in NBC_Copy() (%i)\n", res); return res; }
  }
  
  /* algorithm selection */
  if(p > 4 || size*count < 65536) {
    alg = NBC_RED_BINOMIAL;
    if(rank == root) {
      /* root reduces in receivebuffer */
      handle->tmpbuf = malloc(ext*count);
    } else {
      /* recvbuf may not be valid on non-root nodes */
      handle->tmpbuf = malloc(ext*count*2);
      redbuf = ((char*)handle->tmpbuf)+(ext*count);
    }
  } else {
    handle->tmpbuf = malloc(ext*count);
    alg = NBC_RED_CHAIN;
    segsize = 16384/2;
  }
  if (NULL == handle->tmpbuf) { printf("Error in malloc() (%i)\n", res); return res; }

#ifdef NBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.sendbuf=sendbuf;
  search.recvbuf=recvbuf;
  search.count=count;
  search.datatype=datatype;
  search.op=op;
  search.root=root;
  found = (NBC_Reduce_args*)hb_tree_search((hb_tree*)handle->comminfo->NBC_Dict[NBC_REDUCE], &search);
  if(found == NULL) {
#endif
    schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
    if (NULL == schedule) { printf("Error in malloc() (%i)\n", res); return res; }

    res = NBC_Sched_create(schedule);
    if(res != NBC_OK) { printf("Error in NBC_Sched_create (%i)\n", res); return res; }

    switch(alg) {
      case NBC_RED_BINOMIAL:
        res = red_sched_binomial(rank, p, root, sendbuf, recvbuf, count, datatype, op, redbuf, schedule, handle);
        break;
      case NBC_RED_CHAIN:
        res = red_sched_chain(rank, p, root, sendbuf, recvbuf, count, datatype, op, ext, size, schedule, handle, segsize);
        break;
    }
    if (NBC_OK != res) { printf("Error in Schedule creation() (%i)\n", res); return res; }
    
    res = NBC_Sched_commit(schedule);
    if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Reduce_args*)malloc(sizeof(NBC_Alltoall_args));
    args->sendbuf=sendbuf;
    args->recvbuf=recvbuf;
    args->count=count;
    args->datatype=datatype;
    args->op=op;
    args->root=root;
    args->schedule=schedule;
	  res = hb_tree_insert ((hb_tree*)handle->comminfo->NBC_Dict[NBC_REDUCE], args, args, 0);
    if(res != 0) printf("error in dict_insert() (%i)\n", res);
    /* increase number of elements for Reduce */
    if(++handle->comminfo->NBC_Dict_size[NBC_REDUCE] > NBC_SCHED_DICT_UPPER) {
      NBC_SchedCache_dictwipe((hb_tree*)handle->comminfo->NBC_Dict[NBC_REDUCE], &handle->comminfo->NBC_Dict_size[NBC_REDUCE]);
    }
  } else {
    /* found schedule */
    schedule=found->schedule;
  }
#endif
  
  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Start() (%i)\n", res); return res; }
  
  /* tmpbuf is freed with the handle */
  return NBC_OK;
}


/* binomial reduce
 * working principle:
 * - each node gets a virtual rank vrank
 * - the 'root' node get vrank 0 
 * - node 0 gets the vrank of the 'root'
 * - all other ranks stay identical (they do not matter)
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
static __inline__ int red_sched_binomial(int rank, int p, int root, void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, void *redbuf, NBC_Schedule *schedule, NBC_Handle *handle) {
  int firstred, vrank, vpeer, peer, res, maxr, r;

  RANK2VRANK(rank, vrank, root);
  maxr = (int)ceil((log(p)/LOG2));

  firstred = 1;
  for(r=1; r<=maxr; r++) {
    if((vrank % (1<<r)) == 0) {
      /* we have to receive this round */
      vpeer = vrank + (1<<(r-1));
      VRANK2RANK(peer, vpeer, root)
      if(peer<p) {
        res = NBC_Sched_recv(0, true, count, datatype, peer, schedule);
        if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
        /* we have to wait until we have the data */
        res = NBC_Sched_barrier(schedule);
        if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
        /* perform the reduce in my local buffer */
        if(firstred) {
          if(rank == root) {
            /* root is the only one who reduces in the receivebuffer 
             * take data from sendbuf in first round - save copy */
            res = NBC_Sched_op(recvbuf, false, sendbuf, false, 0, true, count, datatype, op, schedule);
          } else {
            /* all others may not have a receive buffer 
             * take data from sendbuf in first round - save copy */
            res = NBC_Sched_op((char *)redbuf-(unsigned long)handle->tmpbuf, true, sendbuf, false, 0, true, count, datatype, op, schedule);
          }
          firstred = 0;
        } else {
          if(rank == root) {
            /* root is the only one who reduces in the receivebuffer */
            res = NBC_Sched_op(recvbuf, false, recvbuf, false, 0, true, count, datatype, op, schedule);
          } else {
            /* all others may not have a receive buffer */
            res = NBC_Sched_op((char *)redbuf-(unsigned long)handle->tmpbuf, true, (char *)redbuf-(unsigned long)handle->tmpbuf, true, 0, true, count, datatype, op, schedule);
          }
        }
        if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_op() (%i)\n", res); return res; }
        /* this cannot be done until handle->tmpbuf is unused :-( */
        res = NBC_Sched_barrier(schedule);
        if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
      }
    } else {
      /* we have to send this round */
      vpeer = vrank - (1<<(r-1));
      VRANK2RANK(peer, vpeer, root)
      if(firstred) {
        /* we did not reduce anything */
        res = NBC_Sched_send(sendbuf, false, count, datatype, peer, schedule);
      } else {
        /* we have to use the redbuf the root (which works in receivebuf) is never sending .. */
        res = NBC_Sched_send((char *)redbuf-(unsigned long)handle->tmpbuf, true, count, datatype, peer, schedule);
      }
      if (NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      /* leave the game */
      break;
    }
  }

  return NBC_OK;
}

/* chain send ... */ 
static __inline__ int red_sched_chain(int rank, int p, int root, void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int ext, int size, NBC_Schedule *schedule, NBC_Handle *handle, int fragsize) {
  int res, vrank, rpeer, speer, numfrag, fragnum, fragcount, thiscount;
  long offset;
  
  RANK2VRANK(rank, vrank, root);
  VRANK2RANK(rpeer, vrank+1, root);
  VRANK2RANK(speer, vrank-1, root);
  
  if(count == 0) return NBC_OK;
  
  numfrag = count*size/fragsize;
  if((count*size)%fragsize != 0) numfrag++;
  fragcount = count/numfrag;
  /*printf("numfrag: %i, count: %i, size: %i, fragcount: %i\n", numfrag, count, size, fragcount);*/

  for(fragnum = 0; fragnum < numfrag; fragnum++) {
    offset = fragnum*fragcount*ext;
    thiscount = fragcount;
    if(fragnum == numfrag-1) {
      /* last fragment may not be full */
      thiscount = count-fragcount*fragnum;
    }

    /* last node does not recv */
    if(vrank != p-1) {
      res = NBC_Sched_recv((char*)offset, true, thiscount, datatype, rpeer, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
      res = NBC_Sched_barrier(schedule);
      /* root reduces into receivebuf */
      if(vrank == 0) {
        res = NBC_Sched_op((char*)recvbuf+offset, false, (char*)sendbuf+offset, false, (char*)offset, true, thiscount, datatype, op, schedule);
      } else {
        res = NBC_Sched_op((char*)offset, true, (char*)sendbuf+offset, false, (char*)offset, true, thiscount, datatype, op, schedule);
      }
      res = NBC_Sched_barrier(schedule);
    }

    /* root does not send */
    if(vrank != 0) {
      /* rank p-1 has to send out of sendbuffer :) */
      if(vrank == p-1) {
        res = NBC_Sched_send((char*)sendbuf+offset, false, thiscount, datatype, speer, schedule);
      } else {
        res = NBC_Sched_send((char*)offset, true, thiscount, datatype, speer, schedule);
      }
      if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      /* this barrier here seems awkward but isn't!!!! */
      res = NBC_Sched_barrier(schedule);
    }
  }

  return NBC_OK;
}


#ifdef __cplusplus
extern "C" {
#endif
/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_ireduce,NBC_IREDUCE,(void *sendbuf, void *recvbuf, int *count, int *datatype, int *fop, int *root, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_IREDUCE = nbc_ireduce_f
#pragma weak nbc_ireduce = nbc_ireduce_f
#pragma weak nbc_ireduce_ = nbc_ireduce_f
#pragma weak nbc_ireduce__ = nbc_ireduce_f
#pragma weak PNBC_IREDUCE = nbc_ireduce_f
#pragma weak pnbc_ireduce = nbc_ireduce_f
#pragma weak pnbc_ireduce_ = nbc_ireduce_f
#pragma weak pnbc_ireduce__ = nbc_ireduce_f
void nbc_ireduce_f(void *sendbuf, void *recvbuf, int *count, int *datatype, int *fop, int *root, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_ireduce,NBC_IREDUCE)(void *sendbuf, void *recvbuf, int *count, int *datatype, int *fop, int *root, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_ireduce,NBC_IREDUCE)(void *sendbuf, void *recvbuf, int *count, int *datatype, int *fop, int *root, int *fcomm, int *fhandle, int *ierr)  {
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
  *ierr = NBC_Ireduce(sendbuf, recvbuf, *count, dtype, op, *root, comm, handle);
}
#ifdef __cplusplus
}
#endif
