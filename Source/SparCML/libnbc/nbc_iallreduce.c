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
#include <assert.h>

static __inline__ int allred_sched_diss(int rank, int p, int count, MPI_Datatype datatype, void *sendbuf, void *recvbuf, MPI_Op op, NBC_Schedule *schedule, NBC_Handle *handle);
static __inline__ int allred_sched_chain(int rank, int p, int count, MPI_Datatype datatype, void *sendbuf, void *recvbuf, MPI_Op op, int size, int ext, NBC_Schedule *schedule, NBC_Handle *handle, int fragsize);
static __inline__ int allred_sched_ring(int rank, int p, int count, MPI_Datatype datatype, void *sendbuf, void *recvbuf, MPI_Op op, int size, int ext, NBC_Schedule *schedule, NBC_Handle *handle);

// DCMF allreduce is actually not non-blocking!!!
#ifdef USE_DCMF
#undef USE_DCMF
#endif

#ifdef USE_DCMF
#error "DCMF allreduce is not nonblocking and thus disabled"
#include <dcmf_globalcollectives.h>
static int initialized=0;
static void cbfunc(void *clientdata, DCMF_Error_t *error) {
  //printf("in allreduce callback!\n");
  *(unsigned*)clientdata=2;
}

DCMF_Protocol_t allred_reg;
DCMF_Request_t allred_req;

static void init_dcmf_allred() {
  DCMF_GlobalAllreduce_Configuration_t allred_config;
  allred_config.protocol = DCMF_DEFAULT_GLOBALALLREDUCE_PROTOCOL;
  DCMF_GlobalAllreduce_register(&allred_reg, &allred_config);
  initialized=1;
}
#endif

#ifdef NBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int NBC_Allreduce_args_compare(NBC_Allreduce_args *a, NBC_Allreduce_args *b, void *param) {

	if( (a->sendbuf == b->sendbuf) && 
      (a->recvbuf == b->recvbuf) &&
      (a->count == b->count) && 
      (a->datatype == b->datatype) &&
      (a->op == b->op) ) {
    return  0;
  }
	if( a->sendbuf < b->sendbuf ) {	
    return -1;
	}
	return +1;
}
#endif

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Iallreduce=PNBC_Iallreduce
#define NBC_Iallreduce PNBC_Iallreduce
#endif

int NBC_Iallreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, NBC_Handle* handle) {
  int rank, p, res, size;
  MPI_Aint ext;
  NBC_Schedule *schedule;
#ifdef NBC_CACHE_SCHEDULE
  NBC_Allreduce_args *args, *found, search;
#endif
  enum { NBC_ARED_BINOMIAL, NBC_ARED_RING } alg;
  char inplace;

#ifdef USE_DCMF
  int ws, s;
  MPI_Comm_size(comm, &s);
  MPI_Comm_size(MPI_COMM_WORLD, &ws);
  if(s != ws) {
    printf("DCMF only works on MPI_COMM_WORLD (or dups of it) for now -- fallback needs to be implemented :-)\n");
    return NBC_NOT_IMPLEMENTED;
  }
  if(!initialized) init_dcmf_allred();
  handle->dcmf_hndl = (NBC_DCMF_Handle*)malloc(sizeof(NBC_DCMF_Handle));
  handle->dcmf_hndl->done=0;
  handle->dcmf_hndl->type=DCMF_TYPE_ALLREDUCE;
  DCMF_Callback_t callback={ cbfunc, &handle->dcmf_hndl->done };
 

  DCMF_Dt dt;
  switch(datatype) {
    case MPI_UNSIGNED_LONG_LONG: 
       dt = DCMF_UNSIGNED_LONG_LONG;
       break;
    case MPI_LONG_LONG: 
       dt = DCMF_SIGNED_LONG_LONG;
       break;
    case MPI_UNSIGNED: 
       dt = DCMF_UNSIGNED_INT;
       break;
    case MPI_INT: 
       dt = DCMF_SIGNED_INT;
       break;
    case MPI_UNSIGNED_LONG:  // we assume it's the same as integer!!!
       dt = DCMF_UNSIGNED_INT;
       assert(sizeof(unsigned int) == sizeof(unsigned long));
       break;
    case MPI_LONG:  // we assume it's the same as integer !!!
       dt = DCMF_SIGNED_INT;
       assert(sizeof(int) == sizeof(long));
       break;
    case MPI_DOUBLE: 
       dt = DCMF_DOUBLE;
       break;
    default:
     printf("Datatype not supported\n");
     return NBC_NOT_IMPLEMENTED;
     break;
  }

  DCMF_Op dop;
  switch(op) {
    case MPI_SUM: 
       dop = DCMF_SUM;
       break;
    default:
     printf("Operations not supported\n"); 
     return NBC_NOT_IMPLEMENTED;
     break;
  }
  
int r;
MPI_Comm_rank(comm, &r);
printf("[%i] LibNBC starting allreduce\n", r);
  DCMF_GlobalAllreduce(&allred_reg, &allred_req, callback, DCMF_MATCH_CONSISTENCY, -1 /* root?? */, (char*)sendbuf, (char*)recvbuf, count, dt, dop);
printf("[%i] LibNBC after allreduce\n", r);

#else  
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
  
  handle->tmpbuf = malloc(ext*count);
  if(handle->tmpbuf == NULL) { printf("Error in malloc() (%i)\n", res); return NBC_OOR; }

  if((p == 1) && !inplace) {
    /* for a single node - copy data to receivebuf */
    res = NBC_Copy(sendbuf, count, datatype, recvbuf, count, datatype, comm);
    if (NBC_OK != res) { printf("Error in NBC_Copy() (%i)\n", res); return res; }
  }
  
  /* algorithm selection */
  if(p < 4 || size*count < 65536) {
    alg = NBC_ARED_BINOMIAL;
  } else {
    alg = NBC_ARED_RING;
  }
      
#ifdef NBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.sendbuf=sendbuf;
  search.recvbuf=recvbuf;
  search.count=count;
  search.datatype=datatype;
  search.op=op;
  found = (NBC_Allreduce_args*)hb_tree_search((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLREDUCE], &search);
  if(found == NULL) {
#endif
    schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
    if (NULL == schedule) { printf("Error in malloc()\n"); return res; }

    res = NBC_Sched_create(schedule);
    if(res != NBC_OK) { printf("Error in NBC_Sched_create (%i)\n", res); return res; }

    switch(alg) {
      case NBC_ARED_BINOMIAL:
        res = allred_sched_diss(rank, p, count, datatype, sendbuf, recvbuf, op, schedule, handle);
        break;
      case NBC_ARED_RING:
        res = allred_sched_ring(rank, p, count, datatype, sendbuf, recvbuf, op, size, ext, schedule, handle);
        break;
    }
    if (NBC_OK != res) { printf("Error in Schedule creation() (%i)\n", res); return res; }
    
    res = NBC_Sched_commit(schedule);
    if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
    
#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Allreduce_args*)malloc(sizeof(NBC_Allreduce_args));
    args->sendbuf=sendbuf;
    args->recvbuf=recvbuf;
    args->count=count;
    args->datatype=datatype;
    args->op=op;
    args->schedule=schedule;
	  res = hb_tree_insert ((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLREDUCE], args, args, 0);
    if(res != 0) printf("error in dict_insert() (%i)\n", res);
    /* increase number of elements for A2A */
    if(++handle->comminfo->NBC_Dict_size[NBC_ALLREDUCE] > NBC_SCHED_DICT_UPPER) {
      NBC_SchedCache_dictwipe((hb_tree*)handle->comminfo->NBC_Dict[NBC_ALLREDUCE], &handle->comminfo->NBC_Dict_size[NBC_ALLREDUCE]);
    }
  } else {
    /* found schedule */
    schedule=found->schedule;
  }
#endif
  
  res = NBC_Start(handle, schedule);
  if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Start() (%i)\n", res); return res; }
#endif
  
  /* tmpbuf is freed with the handle */
  return NBC_OK;
}


/* binomial allreduce (binomial tree up and binomial bcast down)
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
 * followed by a Bcast:
 * Algorithm:
 * - each node with vrank > 2^r and vrank < 2^r+1 receives from node
 *   vrank - 2^r (vrank=1 receives from 0, vrank 0 receives never)
 * - each node sends each round r to node vrank + 2^r
 * - a node stops to send if 2^r > commsize  
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
static __inline__ int allred_sched_diss(int rank, int p, int count, MPI_Datatype datatype, void *sendbuf, void *recvbuf, MPI_Op op, NBC_Schedule *schedule, NBC_Handle *handle) {
  int root, vrank, r, maxr, firstred, vpeer, peer, res;
  
  root = 0; /* this makes the code for ireduce and iallreduce nearly identical - could be changed to improve performance */
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
        if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
        /* we have to wait until we have the data */
        res = NBC_Sched_barrier(schedule);
        if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
        if(firstred) {
          /* perform the reduce with the senbuf */
          res = NBC_Sched_op(recvbuf, false, sendbuf, false, 0, true, count, datatype, op, schedule);
          firstred = 0;
        } else {
          /* perform the reduce in my local buffer */
          res = NBC_Sched_op(recvbuf, false, recvbuf, false, 0, true, count, datatype, op, schedule);
        }
        if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Sched_op() (%i)\n", res); return res; }
        /* this cannot be done until handle->tmpbuf is unused :-( */
        res = NBC_Sched_barrier(schedule);
        if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
      }
    } else {
      /* we have to send this round */
      vpeer = vrank - (1<<(r-1));
      VRANK2RANK(peer, vpeer, root)
      if(firstred) {
        /* we have to use the sendbuf in the first round .. */
        res = NBC_Sched_send(sendbuf, false, count, datatype, peer, schedule);
      } else {
        /* and the recvbuf in all remeining rounds */
        res = NBC_Sched_send(recvbuf, false, count, datatype, peer, schedule);
      }
      if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      /* leave the game */
      break;
    }
  }
  
  /* this is the Bcast part - copied with minor changes from nbc_ibcast.c 
   * changed: buffer -> recvbuf  */
  RANK2VRANK(rank, vrank, root);

  /* receive from the right hosts  */
  if(vrank != 0) {
    for(r=0; r<maxr; r++) {
      if((vrank >= (1<<r)) && (vrank < (1<<(r+1)))) {
        VRANK2RANK(peer, vrank-(1<<r), root);
        res = NBC_Sched_recv(recvbuf, false, count, datatype, peer, schedule);
        if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
      }
    }
    res = NBC_Sched_barrier(schedule);
    if(NBC_OK != res) { free(handle->tmpbuf); printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
  }

  /* now send to the right hosts */
  for(r=0; r<maxr; r++) {
    if(((vrank + (1<<r) < p) && (vrank < (1<<r))) || (vrank == 0)) {
      VRANK2RANK(peer, vrank+(1<<r), root);
      res = NBC_Sched_send(recvbuf, false, count, datatype, peer, schedule);
      if(res != NBC_OK) { free(handle->tmpbuf); printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
    }
  }
  /* end of the bcast */
  
  return NBC_OK;
}

static __inline__ int allred_sched_chain(int rank, int p, int count, MPI_Datatype datatype, void *sendbuf, void *recvbuf, MPI_Op op, int size, int ext, NBC_Schedule *schedule, NBC_Handle *handle, int fragsize) {
  int res, rrpeer, rbpeer, srpeer, sbpeer, numfrag, fragnum, fragcount, thiscount, bstart, bend;
  long roffset, boffset;
  
  /* reduce peers */
  rrpeer = rank+1; 
  srpeer = rank-1;
  /* bcast peers */
  rbpeer = rank-1;
  sbpeer = rank+1;
  
  if(count == 0) return NBC_OK;
  
  numfrag = count*size/fragsize;
  if((count*size)%fragsize != 0) numfrag++;
  fragcount = count/numfrag;

  /* determine the starting round of bcast ... the first reduced packet
   * is after p-1 rounds at rank 0 and will be sent back ... */
  bstart = p-1+rank;
  /* determine the ending round of bcast ... after arrival of the first
   * packet, each rank has to forward numfrag packets */
  bend = bstart+numfrag;
  /*printf("[%i] numfrag: %i, count: %i, size: %i, fragcount: %i, bstart: %i, bend: %i\n", rank, numfrag, count, size, fragcount, bstart, bend);*/

  /* this are two loops in one - this is a little nasty :-( */
  for(fragnum = 0; fragnum < bend; fragnum++) {
    roffset = fragnum*fragcount*ext;
    boffset = (fragnum-bstart)*fragcount*ext;
    thiscount = fragcount;

    /* first numfrag rounds ... REDUCE to rank 0 */
    if(fragnum < numfrag) {
      if(fragnum == numfrag-1) {
        /* last fragment may not be full */
        thiscount = count-fragcount*fragnum;
      }
      /*printf("[%i] reduce %i elements from %lu\n", rank, thiscount, roffset); */

      /* REDUCE - PART last node does not recv */
      if(rank != p-1) {
        res = NBC_Sched_recv((char*)roffset, true, thiscount, datatype, rrpeer, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
        res = NBC_Sched_barrier(schedule);
        /* root reduces into receivebuf */
        if(rank == 0) {
          res = NBC_Sched_op((char*)recvbuf+roffset, false, (char*)sendbuf+roffset, false, (char*)roffset, true, thiscount, datatype, op, schedule);
        } else {
          res = NBC_Sched_op((char*)roffset, true, (char*)sendbuf+roffset, false, (char*)roffset, true, thiscount, datatype, op, schedule);
        }
        res = NBC_Sched_barrier(schedule);
      }

      /* REDUCE PART root does not send */
      if(rank != 0) {
        /* rank p-1 has to send out of sendbuffer :) */
        if(rank == p-1) {
          res = NBC_Sched_send((char*)sendbuf+roffset, false, thiscount, datatype, srpeer, schedule);
        } else {
          res = NBC_Sched_send((char*)roffset, true, thiscount, datatype, srpeer, schedule);
        }
        if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
        /* this barrier here seems awkward but isn't!!!! */
        /*res = NBC_Sched_barrier(schedule);*/
      }
    }

    /* BCAST from rank 0 */
    if(fragnum >= bstart) {
      /*printf("[%i] bcast %i elements from %lu\n", rank, thiscount, boffset); */
      if(fragnum == bend-1) {
        /* last fragment may not be full */
        thiscount = count-fragcount*(fragnum-bstart);
      }
      
      /* BCAST PART root does not receive */
      if(rank != 0) {
        res = NBC_Sched_recv((char*)recvbuf+boffset, false, thiscount, datatype, rbpeer, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
        res = NBC_Sched_barrier(schedule);
      }
      
      /* BCAST PART last rank does not send */
      if(rank != p-1) {
        res = NBC_Sched_send((char*)recvbuf+boffset, false, thiscount, datatype, sbpeer, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
        res = NBC_Sched_barrier(schedule);
      }
    }
  }

  /*NBC_PRINT_SCHED(*schedule);*/

  return NBC_OK;
}

static __inline__ int allred_sched_ring(int r, int p, int count, MPI_Datatype datatype, void *sendbuf, void *recvbuf, MPI_Op op, int size, int ext, NBC_Schedule *schedule, NBC_Handle *handle) {
  int i; /* runner */
  int segsize, *segsizes, *segoffsets; /* segment sizes and offsets per segment (number of segments == number of nodes */
  int speer, rpeer; /* send and recvpeer */
  int res;

  if(count == 0) return NBC_OK;

  {
    int mycount; /* temporary */
    segsizes = (int*)malloc(sizeof(int)*p);
    segoffsets = (int*)malloc(sizeof(int)*p);
    segsize = count/p; /* size of the segments */
    if(count%p != 0) segsize++;
    mycount = count;
    segoffsets[0] = 0;
    for(i = 0; i<p;i++) {
      mycount -= segsize;
      segsizes[i] = segsize;
      if(mycount < 0) {
        segsizes[i] = segsize+mycount;
        mycount = 0;
      }
      if(i) segoffsets[i] = segoffsets[i-1] + segsizes[i-1];
      //if(!r) printf("count: %i, (%i) size: %i, offset: %i\n", count, i, segsizes[i], segoffsets[i]);
    }
  }

  /* reduce peers */
  speer = (r+1)%p;
  rpeer = (r-1+p)%p;

  /*  + -> reduced this round
   *  / -> sum (reduced in a previous step)
   *
   *     *** round 0 ***
   *    0        1        2      
   *                             
   *   00       10       20      0: [1] -> 1 
   *   01       11       21      1: [2] -> 2
   *   02       12       22      2: [0] -> 0  --> send element (r+1)%p to node (r+1)%p
   *
   *      *** round 1 ***
   *    0        1        2
   *
   *   00+20    10       20     0: red(0), [0] -> 1
   *   01       11+01    21     1: red(1), [1] -> 2
   *   02       12       22+12  2: red(2), [2] -> 0 --> reduce and send element (r+0)%p to node (r+1)%p
   *
   *      *** round 2 ***
   *    0        1        2
   *
   *   00/20    all      20     0: red(2), [2] -> 1 
   *   01       11/01    all    1: red(0), [0] -> 2
   *   all      12       22/12  2: red(1), [1] -> 0 --> reduce and send (r-1)%p to node (r+1)%p
   *
   *      *** round 3 ***
   *    0        1        2
   *
   *   00/20    all      all    0: [1] -> 1
   *   all      11/01    all    1: [2] -> 2
   *   all      all      22/12  2: [0] -> 0 --> send element (r-2)%p to node (r+1)%p
   *
   *      *** round 4 ***
   *    0        1        2
   *
   *   all      all      all    0: done
   *   all      all      all    1: done
   *   all      all      all    2: done
   *
   * -> 4
   *     *** round 0 ***
   *    0        1        2        3      
   *                             
   *   00       10       20       30       0: [1] -> 1 
   *   01       11       21       31       1: [2] -> 2
   *   02       12       22       32       2: [3] -> 3  
   *   03       13       23       33       3: [0] -> 0 --> send element (r+1)%p to node (r+1)%p
   *
   *      *** round 1 ***
   *    0        1        2        3
   *
   *   00+30    10       20       30       0: red(0), [0] -> 1
   *   01       11+01    21       31       1: red(1), [1] -> 2
   *   02       12       22+12    32       2: red(2), [2] -> 3 
   *   03       13       23       33+23    3: red(3), [3] -> 0 --> reduce and send element (r+0)%p to node (r+1)%p
   *
   *      *** round 2 ***
   *    0        1        2        3
   *
   *   00/30    10+00/30 20       30       0: red(3), [3] -> 1 
   *   01       11/01    21+11/01 31       1: red(0), [0] -> 2
   *   02       12       22/12    32+22/12 2: red(1), [1] -> 3
   *   03+33/23 13       23       33/23    3: red(2), [2] -> 0 --> reduce and send (r-1)%p to node (r+1)%p
   *
   *      *** round 3 ***
   *    0        1        2        3
   *
   *   00/30    10/00/30 all      30       0: red(2), [2] -> 1 
   *   01       11/01    21/11/01 all      1: red(3), [3] -> 2
   *   all      12       22/12    32/22/12 2: red(0), [0] -> 3
   *   03/33/23 all      23       33/23    3: red(1), [1] -> 0 --> reduce and send (r-2)%p to node (r+1)%p 
   *
   *      *** round 4 ***
   *    0        1        2        3 
   *   
   *   00/30    10/00/30 all      all      0: [1] -> 1
   *   all      11/01    21/11/01 all      1: [2] -> 2
   *   all      all      22/12    32/22/12 2: [3] -> 3
   *   03/33/23 all      all      33/23    3: [0] -> 0 --> receive and send element (r+1)%p to node (r+1)%p
   *
   *      *** round 5 ***
   *    0        1        2        3
   *    
   *   all      10/00/30 all      all      0: [0] -> 1
   *   all      all      21/11/01 all      1: [1] -> 2
   *   all      all      all      32/22/12 2: [3] -> 3
   *   03/33/23 all      all      all      3: [4] -> 4 --> receive and send element (r-0)%p to node (r+1)%p
   *
   *      *** round 6 ***
   *    0        1        2        3
   *   
   *   all      all      all      all      
   *   all      all      all      all      
   *   all      all      all      all  
   *   all      all      all      all     receive element (r-1)%p
   *
   *   2p-2 rounds ... every node does p-1 reductions and p-1 sends
   *
   */
  {
    int round = 0;
    /* first p-1 rounds are reductions */
    do {
      int selement = (r+1-round + 2*p /*2*p avoids negative mod*/)%p; /* the element I am sending */
      int soffset = segoffsets[selement]*ext;
      int relement = (r-round + 2*p /*2*p avoids negative mod*/)%p; /* the element that I receive from my neighbor */
      int roffset = segoffsets[relement]*ext;

      /* first message come out of sendbuf */
      if(round == 0) {
        NBC_Sched_send((char*)sendbuf+soffset, false, segsizes[selement], datatype, speer, schedule);
        //printf("[%i] round %i - sending %i\n", r, round, selement);
      } else {
        NBC_Sched_send((char*)recvbuf+soffset, false, segsizes[selement], datatype, speer, schedule);
        //printf("[%i] round %i - sending %i\n", r, round, selement);
      }
      NBC_Sched_recv((char*)recvbuf+roffset, false, segsizes[relement], datatype, rpeer, schedule);
      //printf("[%i] round %i - receiving %i\n", r, round, relement);

      NBC_Sched_barrier(schedule);
      //printf("[%i] round %i - reducing %i\n", r, round, relement);
      NBC_Sched_op((char*)recvbuf+roffset, false, (char*)sendbuf+roffset, false, (char*)recvbuf+roffset, false, segsizes[relement], datatype, op, schedule);
      NBC_Sched_barrier(schedule);

      round++;
    } while(round < p-1);

    do {
      int selement = (r+1-round + 2*p /*2*p avoids negative mod*/)%p; /* the element I am sending */
      int soffset = segoffsets[selement]*ext;
      int relement = (r-round + 2*p /*2*p avoids negative mod*/)%p; /* the element that I receive from my neighbor */
      int roffset = segoffsets[relement]*ext;

      //printf("[%i] round %i receiving %i sending %i\n", r, round, relement, selement);
      NBC_Sched_send((char*)recvbuf+soffset, false, segsizes[selement], datatype, speer, schedule);
      NBC_Sched_recv((char*)recvbuf+roffset, false, segsizes[relement], datatype, rpeer, schedule);
      NBC_Sched_barrier(schedule);
      round++;  
    } while (round < 2*p-2);
  }

  //NBC_PRINT_SCHED(*schedule);

  return NBC_OK;
}



#ifdef __cplusplus
extern "C" {
#endif
/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_iallreduce,NBC_IALLREDUCE,(void *sendbuf, void *recvbuf, int *count, int *datatype, int *fop, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_IALLREDUCE = nbc_iallreduce_f
#pragma weak nbc_iallreduce = nbc_iallreduce_f
#pragma weak nbc_iallreduce_ = nbc_iallreduce_f
#pragma weak nbc_iallreduce__ = nbc_iallreduce_f
#pragma weak PNBC_IALLREDUCE = nbc_iallreduce_f
#pragma weak pnbc_iallreduce = nbc_iallreduce_f
#pragma weak pnbc_iallreduce_ = nbc_iallreduce_f
#pragma weak pnbc_iallreduce__ = nbc_iallreduce_f
void nbc_iallreduce_f(void *sendbuf, void *recvbuf, int *count, int *datatype, int *fop, int *fcomm, int *fhandle, int *ierr) {
#else
void NBC_F77_FUNC_(nbc_iallreduce,NBC_IALLREDUCE)(void *sendbuf, void *recvbuf, int *count, int *datatype, int *fop, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_iallreduce,NBC_IALLREDUCE)(void *sendbuf, void *recvbuf, int *count, int *datatype, int *fop, int *fcomm, int *fhandle, int *ierr)  {
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
  *ierr = NBC_Iallreduce(sendbuf, recvbuf, *count, dtype, op, comm, handle);
}

#ifdef __cplusplus
}
#endif
