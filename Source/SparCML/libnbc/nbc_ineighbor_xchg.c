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


#ifdef HAVE_DCMF
// special handling on BG/P DCMF layer -- use multisend
#include <dcmf.h>
#include <dcmf_multisend.h>
#include <dcmf_globalcollectives.h>

// DCMF caches its structures that reflect the graph and offsets etc. at the duped communicator ...
/* the keyval (global) */
static int gkeyval=MPI_KEYVAL_INVALID;

static DCMF_Protocol_t barr_reg;
static DCMF_Request_t barr_req;

static int dcmf_key_copy(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag) {
  /* delete the attribute in the new comm  - it will be created at the
   * first usage */
  *flag = 0;

  return MPI_SUCCESS;
}

static int dcmf_key_delete(MPI_Comm comm, int keyval, void *attribute_val, void *extra_state) {
  dcmf_comminfo *comminfo;

  if(keyval == gkeyval) {
    comminfo=(dcmf_comminfo*)attribute_val;
    free((void*)comminfo);
  } else {
    printf("Got wrong keyval!(%i)\n", keyval);
  }

  return MPI_SUCCESS;
}


#ifdef NBC_CACHE_SCHEDULE
#undef NBC_CACHE_SCHEDULE // schedule caching not supported with DCMF
#endif
#endif

#ifdef NBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int NBC_Ineighbor_xchg_args_compare(NBC_Ineighbor_xchg_args *a, NBC_Ineighbor_xchg_args *b, void *param) {

	if( (a->sbuf == b->sbuf) && 
      (a->scount == b->scount) && 
      (a->stype == b->stype) &&
      (a->rbuf == b->rbuf) && 
      (a->rcount == b->rcount) && 
      (a->rtype == b->rtype) ) {
      return  0;
    }
	if( a->sbuf < b->sbuf ) {	
      return -1;
	}
	return +1;
}
#else
#ifdef HAVE_DCMF
int NBC_Ineighbor_xchg_args_compare(NBC_Ineighbor_xchg_args *a, NBC_Ineighbor_xchg_args *b, void *param) {
  printf("this is a stub for NBC_Ineighbor_xchg_args_compare to compile with DCMF\n");
}
#endif
#endif

/* this is a new collective operation defined on a topo communicator.
 * This operation communicates with all neighbors in a topology
 * communicator.  This operation is comparable to an Alltoall on a
 * communicator that spans the neighbors. An example in a 2d cartesian
 * grid:
 *
 *  0    1    2    3    4    5          ^  1st dim
 *  6    7    8    9    10   11         -> 2nd dim        
 *  12   13   14   15   16   17
 *
 * Case of Cartesian Topology:
 * ndims is two in this case and every rank has a maximum of 2*ndims
 * neighbors! Thus send and receive arrays are arranged as in the
 * Alltoall case, i.e., they must offer space for 2*ndims*count elements
 * of the supplied type. The order of nodes is first along the first
 * dimension, in displacement -1 then +1, then along the second
 * dimension and so on ... on our example, the order for rank 8 is:
 *  2, 14, 7, 9. It can be calculated with the local function
 *  MPI_Cart_shift().
 *
 * Case of Graph Topology:
 *  A graph topology is more complicated because it might be irregular,
 *  i.e., different nodes have different numbers of neighbors. The
 *  arrays are defined similarly to Alltoall again. However, the size of
 *  the arrays depends on the number of neighbors and might be different
 *  on different nodes. The local function MPI_Graph_neighbors_count()
 *  returns the array size and the local function MPI_Graph_neigbors()
 *  returns the actual neigbors. The array must have enough space for
 *  nneighbor*count elements of the supplied datatype and the data is
 *  ordered as MPI_Graph_neigbors() returns. 
 *
 * Implementation ideas:
 * - check if this is a topo comm (MPI_TOPO_TEST?)
 *     -> MPI_Topo_test(comm, &stat);
 *         if(stat == MPI_UNDEFINED) return error;
 * 
 * if(stat == MPI_GRAPH)
 *     MPI_graph_neighbors_count(comm, rank, &nneighbors)
 *     neigbor_array = malloc(sizeof(int)*nneighbors);
 *     use MPI_Graph_neighbors(comm, rank, nneighbors, &neigbor_array)
 *     send and receive to them all non-blocking ...
 * 
 * if(stat == MPI_CART)
 *     MPI_Cartdim_get(comm, &ndims);
 *     for(i=0;i<ndims;i++) {
 *         MPI_Cart_shift(comm, i, -1, &rpeer, &speer);
 *         if(rpeer != MPI_PROC_NULL) Sched_recv ....
 *         if(speer != MPI_PROC_NULL) Sched_send ....
 *     }
 * 
 */

#ifdef HAVE_DCMF
static int g_manytomany_registered = 0;
static DCMF_Protocol_t  g_multisend_proto;

// we have exactly 32 slots on COMM_WORLD in the current DCMF
// implementation, thus we allocate a global array of handles statically
// (this needs to change if we use it from multiple files) and index
// into it in the callback
#define MAX_OUTSTANDING 32
static NBC_DCMF_Handle *g_dcmf_handles[MAX_OUTSTANDING];
static int g_cid=0; // this counts the current slot we're using

// barrier done callback
void DCMF_barr_done_cb(void *data, DCMF_Error_t *err) {
  unsigned cid = (unsigned)data;	
  printf("[%d] [%i]: barrier completion cb done: %i\n", DCMF_Messager_rank(), cid, g_dcmf_handles[cid]->barr_done);
  g_dcmf_handles[cid]->barr_done++;
}

// recv done callback
void DCMF_m2m_done_cb(void *data, DCMF_Error_t *err) {
  unsigned cid = (unsigned)data;	
  printf("[%d] [%i] completion cb done: %i\n", DCMF_Messager_rank(), cid, g_dcmf_handles[cid]->done);
  g_dcmf_handles[cid]->done++;
}

// Send done callback
/*
void DCMF_done_cb_send(void *data, DCMF_Error_t *err) {
  unsigned cid = (unsigned)data;	
  printf("%d: completion cb send %i %i\n", DCMF_Messager_rank(), cid, g_dcmf_handles[cid]->done);
  g_dcmf_handles[cid]->done++;
}*/

// Manytomany callback at receiver
// gets CID and needs to identify correct set of parameters (index in handles) from it
DCMF_Request_t *DCMF_Manytomany_cb (unsigned cid,
             void            * arg,
             char           ** rcvbuf,
             unsigned       ** rcvlens,
             unsigned       ** rcvdispls,
             unsigned       ** rcvcounters,
             unsigned        * nranks,
             unsigned        * rankIndex,
             DCMF_Callback_t * cb_done) {             

  NBC_DCMF_Handle *hndl=g_dcmf_handles[cid];

  printf("[%d] [%i]: in cb w/ rankIndex %d nranks %d rlens[0] %d rlens[1] %d %x rdispls[0] %d, rdispls[1] %d\n", DCMF_Messager_rank(), 
	 cid, hndl->comminfo->rankIndex, hndl->comminfo->indeg, hndl->rlens[0], hndl->rlens[1], hndl->rbuf, hndl->rdispls[0], hndl->rdispls[1]);

  *rcvbuf       =  hndl->rbuf; 	// receive data buffer for this conn_id
  *rcvlens      =  hndl->rlens;	// list of sizes of the messages to receive
  *rcvdispls    =  hndl->rdispls; // list of offsets to the start of rcvbuf
  *rcvcounters  =  hndl->comminfo->rcvcounters; // list of counters required for internal use
  *nranks       =  hndl->comminfo->indeg; // number of ranks involved in this manytomany
  *rankIndex    =  hndl->comminfo->rankIndex; // the process' rank in the list of ranks to identify the lens/displs/counters of the local node (0<=rankIndex<nranks). A value of 0xffffffff means that receiver does not receive a local message.
  *cb_done      =  hndl->cb_m2m_done;

  return &hndl->rrequest;
}
#endif

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Ineighbor_xchg=PNBC_Ineighbor_xchg
#define NBC_Ineighbor_xchg PNBC_Ineighbor_xchg
#endif
int NBC_Ineighbor_xchg(void *sbuf, int scount, MPI_Datatype stype,
        void *rbuf, int rcount, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle) {
  int rank, size, res, worldsize, i;
  MPI_Aint sndext, rcvext;
  
  double t[10];
  t[0] = PMPI_Wtime();

  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_size(comm, &size);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_size() (%i)\n", res); return res; }
  res = MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_size() (%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }

  res = MPI_Type_extent(stype, &sndext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
  res = MPI_Type_extent(rtype, &rcvext);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }

  t[1] = PMPI_Wtime();
  
#ifdef HAVE_DCMF //specialized DCMF version
  // DCMF only works on comm_world
  if(worldsize == size) {
    if(!g_manytomany_registered) {

      DCMF_Manytomany_Configuration_t mconfig;
      mconfig.protocol     =  DCMF_MEMFIFO_DMA_M2M_PROTOCOL;
      mconfig.cb_recv      =  DCMF_Manytomany_cb;
      mconfig.arg          =  NULL;
      mconfig.nconnections =  worldsize;

      for(i=0; i<MAX_OUTSTANDING; ++i) g_dcmf_handles[i] = NULL;

      DCMF_Manytomany_register (&g_multisend_proto, &mconfig);
      g_manytomany_registered=1;
      
      // register DCMF barrier
      DCMF_GlobalBarrier_Configuration_t barrier_config;
      barrier_config.protocol = DCMF_GI_GLOBALBARRIER_PROTOCOL;
      DCMF_GlobalBarrier_register(&barr_reg, &barrier_config);
          
      printf("[%i] registered DCMF stuff\n", rank);

      /* keyval is not initialized yet, we have to init it */
      if(MPI_KEYVAL_INVALID == gkeyval) {
        res = MPI_Keyval_create(dcmf_key_copy, dcmf_key_delete, &(gkeyval), NULL);
        if((MPI_SUCCESS != res)) { printf("Error in MPI_Keyval_create() (%i)\n", res); return res; }
      }
    }
    
    t[2] = PMPI_Wtime();

    /* We should actually cache this at the calling communicator, but
     * that's a prototype :) */
    g_cid=(g_cid+1)%MAX_OUTSTANDING;
    //printf("set g_cid %i\n", g_cid);

    // TODO: could actually overwrite active requests here, should have flag to check!!
    if(g_dcmf_handles[g_cid] != NULL) {
      //free(g_dcmf_handles[g_cid]->sndcounters); 
      //free(g_dcmf_handles[g_cid]->rcvcounters);
      free(g_dcmf_handles[g_cid]->slens); 
      free(g_dcmf_handles[g_cid]->sdispls);
      //free(g_dcmf_handles[g_cid]->neighbors);
      free(g_dcmf_handles[g_cid]);
    }

    // this global thing is kind of annoying ... but DCMF supports
    // only comm_world anyway ... thus just leave it global
    g_dcmf_handles[g_cid] = (NBC_DCMF_Handle*)malloc(sizeof(NBC_DCMF_Handle));
    handle->dcmf_hndl = g_dcmf_handles[g_cid];
    handle->dcmf_hndl->cid = g_cid;
    
    t[3] = PMPI_Wtime();

    int flag;
    dcmf_comminfo *comminfo;
    res = MPI_Attr_get(comm, gkeyval, &comminfo, &flag);
    if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_get() (%i)\n", res); return res; }

    if (flag) {
      /* we found it */
      handle->dcmf_hndl->comminfo = comminfo;
    } else {
      printf("[%i] creating new DCMF topology and attaching it to comm\n", rank);
      /* create a new comminfo object and attach it to the comm */
      comminfo = (dcmf_comminfo*)malloc(sizeof(dcmf_comminfo));
      handle->dcmf_hndl->comminfo = comminfo;
      comminfo->proto = &g_multisend_proto; // might later move somewhere else :)

      /* put the new attribute to the comm */
      res = MPI_Attr_put(comm, gkeyval, comminfo); 
      if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_put() (%i)\n", res); return NULL; }
      
      int indeg, outdeg, weighted;
      res = NBC_Comm_neighbors_count(comm, &indeg, &outdeg, &weighted);
      if(res != NBC_OK) return res;

      /* query neighborhood structure */  
      int *srcs = malloc(indeg*sizeof(int));
      int *dsts = malloc(outdeg*sizeof(int));
      res = NBC_Comm_neighbors(comm, indeg, srcs, MPI_UNWEIGHTED, outdeg, dsts, MPI_UNWEIGHTED);
      if(res != NBC_OK) return res;
      
      comminfo->neighbors = (unsigned int*)malloc(sizeof(unsigned int)*outdeg);
      comminfo->permutation = (unsigned int*)malloc(sizeof(unsigned int)*outdeg);
      comminfo->ridx= (unsigned int*)malloc(sizeof(unsigned int)*indeg);

      comminfo->outdeg = (unsigned)outdeg;
      comminfo->indeg = (unsigned)indeg;

      for(i = 0; i < comminfo->outdeg; i++) {
        comminfo->neighbors[i] = dsts[i];
        comminfo->permutation[i] = i;
      }

      comminfo->sndcounters = (unsigned int*)malloc(sizeof(unsigned int)*outdeg);
      comminfo->rcvcounters = (unsigned int*)malloc(sizeof(unsigned int)*indeg);

      // TODO: that doesn't work with double edges! filter them here (or fail at least)!

      // get remote indices
      printf("[%i] indeg %i, outdeg: %i, sbuf: %x, rbuf: %x\n", rank, indeg, outdeg, sbuf, rbuf);

      int *tmp_sbuf = malloc(sizeof(int)*worldsize);
      int *tmp_rbuf = malloc(sizeof(int)*worldsize);
      for(i = 0; i < indeg; i++) {
        tmp_sbuf[srcs[i]] = i;
        //printf("[%i] send index %i to neighbor %i\n", rank, i, srcs[i]);
      }
      // TODO: it's not quite nonblocking (or scalable) yet ;) -- it's
      // possible but messy ... and it's cached anyway, i.e., this
      // should be done during the blocking topology creation routines 
      MPI_Alltoall(&tmp_sbuf[0], 1, MPI_INT, &tmp_rbuf[0], 1, MPI_INT, comm);

      int *nneighbors = calloc(worldsize, sizeof(int)); // how many edges do I have from a specific neighbor!
      for(i = 0; i < outdeg; i++) nneighbors[dsts[i]]++; // count number of edges to each neighbor

      for(i = 0; i < outdeg; i++) {
        comminfo->ridx[i] = tmp_rbuf[dsts[i]] - ( --nneighbors[dsts[i]] );
        printf("[%i] moving my data to index %i at receiver %i (array index: %i)\n", rank, comminfo->ridx[i], comminfo->neighbors[i], i);
      }

      // TODO: might go into the callback if it remains static!
      comminfo->rankIndex = 0xffffffff;
    }

    t[4] = PMPI_Wtime();
    
    handle->dcmf_hndl->cb_m2m_done.function = DCMF_m2m_done_cb;
    handle->dcmf_hndl->cb_m2m_done.clientdata = (void*) handle->dcmf_hndl->cid;
    handle->dcmf_hndl->cb_barr_done.function = DCMF_barr_done_cb;
    handle->dcmf_hndl->cb_barr_done.clientdata = (void*) handle->dcmf_hndl->cid;

    handle->dcmf_hndl->done = 0;
    handle->dcmf_hndl->barr_done = 0;
    handle->dcmf_hndl->type = DCMF_TYPE_MANY_TO_MANY; 
    
    t[5] = PMPI_Wtime();

    // fill in sizes and offsets -- this could be cached based on the
    // actual sizes and buffer parameters
    {

      { int ssize, rsize;
        MPI_Type_size(stype, &ssize);
        MPI_Type_size(rtype, &rsize);
        if(ssize != sndext || rsize != rcvext) {
          printf("only contiguous types are supported in this DCMF prototype\n");
      } }

        handle->dcmf_hndl->sbuf = (char*)sbuf;
        handle->dcmf_hndl->slens= (unsigned int*)malloc(sizeof(unsigned int)*comminfo->outdeg);
      handle->dcmf_hndl->sdispls= (unsigned int*)malloc(sizeof(unsigned int)*comminfo->outdeg);

      handle->dcmf_hndl->rbuf = (char*)rbuf;
      handle->dcmf_hndl->rlens = (unsigned int*)malloc(sizeof(unsigned int)*comminfo->indeg);
      handle->dcmf_hndl->rdispls = (unsigned int*)malloc(sizeof(unsigned int)*comminfo->indeg);

      for(i = 0; i < comminfo->outdeg; i++) {
        handle->dcmf_hndl->slens[i] = sndext*scount;
        handle->dcmf_hndl->sdispls[i] = i*sndext*scount;
        comminfo->permutation[i] = i;
      }
      for(i = 0; i < comminfo->indeg; i++) {
        handle->dcmf_hndl->rlens[i] = rcvext*rcount;
        handle->dcmf_hndl->rdispls[i] = i*rcvext*rcount;
      }
    }

    t[6] = PMPI_Wtime();

    // set the handle to -1 because the barrier increases it to 0 -- the
    // progression function will then issue the multisend when the handle is 1
    DCMF_GlobalBarrier(&barr_reg, &barr_req, handle->dcmf_hndl->cb_barr_done);

    t[7] = PMPI_Wtime();
//    while(handle->dcmf_hndl->done !=0) DCMF_Messager_advance();

    /* see nbc.c for the actual many_to_many call -- this can only be
     * issued after the barrier is completed which has to be done
     * nonblocking ... thus it has to be in the progress function */
    if(rank==0) printf("[%i] [%i]: init-times: %f %f %f %f %f %f %f \n", rank, g_cid, (t[1]-t[0])*1e6, (t[2]-t[1])*1e6, (t[3]-t[2])*1e6, (t[4]-t[3])*1e6, (t[5]-t[4])*1e6, (t[6]-t[5])*1e6, (t[7]-t[6])*1e6);

  }
#else // normal MPI version
  char inplace;
  NBC_Schedule *schedule;
#ifdef NBC_CACHE_SCHEDULE
  NBC_Ineighbor_xchg_args *args, *found, search;
#endif

  NBC_IN_PLACE(sbuf, rbuf, inplace);
  
  handle->tmpbuf=NULL;

#ifdef NBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.sbuf=sbuf;
  search.scount=scount;
  search.stype=stype;
  search.rbuf=rbuf;
  search.rcount=rcount;
  search.rtype=rtype;
  found = (NBC_Ineighbor_xchg_args*)hb_tree_search((hb_tree*)handle->comminfo->NBC_Dict[NBC_NEIGHBOR_XCHG], &search);
  if(found == NULL) {
#endif
    schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
    
    res = NBC_Sched_create(schedule);
    if(res != NBC_OK) { printf("Error in NBC_Sched_create, res = %i\n", res); return res; }

    {
      int indegree, outdegree, weighted, *srcs, *dsts, i;
      res = NBC_Comm_neighbors_count(comm, &indegree, &outdegree, &weighted);
      if(res != NBC_OK) return res;

      srcs = (int*)malloc(sizeof(int)*indegree);
      dsts = (int*)malloc(sizeof(int)*outdegree);

      res = NBC_Comm_neighbors(comm, indegree, srcs, MPI_UNWEIGHTED, outdegree, dsts, MPI_UNWEIGHTED);
      if(res != NBC_OK) return res;

      if(inplace) { /* we need an extra buffer to be deadlock-free */
        handle->tmpbuf = malloc(outdegree*sndext*scount);

        for(i = 0; i < indegree; i++) {
          res = NBC_Sched_recv((char*)0+i*rcount*rcvext, true, rcount, rtype, srcs[i], schedule);
          if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
        }
        for(i = 0; i < outdegree; i++) {
          res = NBC_Sched_send((char*)sbuf+i*scount*sndext, false, scount, stype, dsts[i], schedule);
          if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
        }
        /* unpack from buffer */
        for(i = 0; i < indegree; i++) {
          res = NBC_Sched_barrier(schedule);
          if (NBC_OK != res) { printf("Error in NBC_Sched_barrier() (%i)\n", res); return res; }
          res = NBC_Sched_copy((char*)0+i*rcount*rcvext, true, rcount, rtype, (char*)rbuf+i*rcount*rcvext, false, rcount, rtype, schedule);
          if (NBC_OK != res) { printf("Error in NBC_Sched_copy() (%i)\n", res); return res; }
        }
      } else { /* non INPLACE case */
        /* simply loop over neighbors and post send/recv operations */
        for(i = 0; i < indegree; i++) {
          res = NBC_Sched_recv((char*)rbuf+i*rcount*rcvext, false, rcount, rtype, srcs[i], schedule);
          if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
        }
        for(i = 0; i < outdegree; i++) {
          res = NBC_Sched_send((char*)sbuf+i*scount*sndext, false, scount, stype, dsts[i], schedule);
          if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
        }
      }
    }
    
    res = NBC_Sched_commit(schedule);
    if (NBC_OK != res) { printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Ineighbor_xchg_args*)malloc(sizeof(NBC_Ineighbor_xchg_args));
    args->sbuf=sbuf;
    args->scount=scount;
    args->stype=stype;
    args->rbuf=rbuf;
    args->rcount=rcount;
    args->rtype=rtype;
    args->schedule=schedule;
	  res = hb_tree_insert ((hb_tree*)handle->comminfo->NBC_Dict[NBC_NEIGHBOR_XCHG], args, args, 0);
    if(res != 0) printf("error in dict_insert() (%i)\n", res);
    /* increase number of elements for A2A */
    if(++handle->comminfo->NBC_Dict_size[NBC_NEIGHBOR_XCHG] > NBC_SCHED_DICT_UPPER) {
      NBC_SchedCache_dictwipe((hb_tree*)handle->comminfo->NBC_Dict[NBC_NEIGHBOR_XCHG], &handle->comminfo->NBC_Dict_size[NBC_NEIGHBOR_XCHG]);
    }
  } else {
    /* found schedule */
    schedule=found->schedule;
  }
#endif
  
  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { printf("Error in NBC_Start() (%i)\n", res); return res; }
#endif 

  return NBC_OK;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Fortran bindings */
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_comm_neighbors,NBC_COMM_NEIGHBORS,(int *fcomm, int *maxindegree, int *sources, int *sourceweights, int *maxoutdegree, int *destinations, int *destweights, int *ierr));
#pragma weak NBC_COMM_NEIGHBORS = nbc_comm_neighbors_f
#pragma weak nbc_comm_neighbors = nbc_comm_neighbors_f
#pragma weak nbc_comm_neighbors_ = nbc_comm_neighbors_f
#pragma weak nbc_comm_neighbors__ = nbc_comm_neighbors_f
#pragma weak PNBC_COMM_NEIGHBORS = nbc_comm_neighbors_f
#pragma weak pnbc_comm_neighbors = nbc_comm_neighbors_f
#pragma weak pnbc_comm_neighbors_ = nbc_comm_neighbors_f
#pragma weak pnbc_comm_neighbors__ = nbc_comm_neighbors_f
void nbc_comm_neighbors_f(int *fcomm, int *maxindegree, int *sources, int *sourceweights, int *maxoutdegree, int *destinations, int *destweights, int *ierr) 
#else
void NBC_F77_FUNC_(nbc_comm_neighbors,NBC_COMM_NEIGHBORS)(int *fcomm, int *maxindegree, int *sources, int *sourceweights, int *maxoutdegree, int *destinations, int *destweights, int *ierr);
void NBC_F77_FUNC_(nbc_comm_neighbors,NBC_COMM_NEIGHBORS)(int *fcomm, int *maxindegree, int *sources, int *sourceweights, int *maxoutdegree, int *destinations, int *destweights, int *ierr) 
#endif
{  
  MPI_Comm comm;
  comm = MPI_Comm_f2c(*fcomm);

  *ierr = NBC_Comm_neighbors(comm, *maxindegree, sources, sourceweights, *maxoutdegree, destinations, destweights);
}

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_comm_neighbors_count,NBC_COMM_NEIGHBORS_COUNT,(int *fcomm, int *indegree, int *outdegree, int *weighted, int *ierr));
#pragma weak NBC_COMM_NEIGHBORS_COUNT = nbc_comm_neighbors_count_f
#pragma weak nbc_comm_neighbors_count = nbc_comm_neighbors_count_f
#pragma weak nbc_comm_neighbors_count_ = nbc_comm_neighbors_count_f
#pragma weak nbc_comm_neighbors_count__ = nbc_comm_neighbors_count_f
#pragma weak PNBC_COMM_NEIGHBORS_COUNT = nbc_comm_neighbors_count_f
#pragma weak pnbc_comm_neighbors_count = nbc_comm_neighbors_count_f
#pragma weak pnbc_comm_neighbors_count_ = nbc_comm_neighbors_count_f
#pragma weak pnbc_comm_neighbors_count__ = nbc_comm_neighbors_count_f
void nbc_comm_neighbors_count_f(int *fcomm, int *indegree, int *outdegree, int *weighted, int *ierr) 
#else
void NBC_F77_FUNC_(nbc_comm_neighbors_count,NBC_COMM_NEIGHBORS_COUNT)(int *fcomm, int *indegree, int *outdegree, int *weighted, int *ierr);
void NBC_F77_FUNC_(nbc_comm_neighbors_count,NBC_COMM_NEIGHBORS_COUNT)(int *fcomm, int *indegree, int *outdegree, int *weighted, int *ierr) 
#endif
{
  MPI_Comm comm;
  comm = MPI_Comm_f2c(*fcomm);

  *ierr = NBC_Comm_neighbors_count(comm, indegree, outdegree, weighted);
}

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
NBC_F77_ALLFUNC_(nbc_ineighbor_xchg,NBC_INEIGHBOR_XCHG,(void *sbuf, int *scount, int *stype, void *rbuf, int *rcount,
        int *rtype, int *fcomm, int *fhandle, int *ierr));
#pragma weak NBC_INEIGHBOR_XCHG = nbc_ineighbor_xchg_f
#pragma weak nbc_ineighbor_xchg = nbc_ineighbor_xchg_f
#pragma weak nbc_ineighbor_xchg_ = nbc_ineighbor_xchg_f
#pragma weak nbc_ineighbor_xchg__ = nbc_ineighbor_xchg_f
#pragma weak PNBC_INEIGHBOR_XCHG = nbc_ineighbor_xchg_f
#pragma weak pnbc_ineighbor_xchg = nbc_ineighbor_xchg_f
#pragma weak pnbc_ineighbor_xchg_ = nbc_ineighbor_xchg_f
#pragma weak pnbc_ineighbor_xchg__ = nbc_ineighbor_xchg_f
void nbc_ineighbor_xchg_f(void *sbuf, int *scount, int *stype, void *rbuf, int *rcount,
        int *rtype, int *fcomm, int *fhandle, int *ierr) 
#else
void NBC_F77_FUNC_(nbc_ineighbor_xchg,NBC_INEIGHBOR_XCHG)(void *sbuf, int *scount, int *stype, void *rbuf, int *rcount,
        int *rtype, int *fcomm, int *fhandle, int *ierr);
void NBC_F77_FUNC_(nbc_ineighbor_xchg,NBC_INEIGHBOR_XCHG)(void *sbuf, int *scount, int *stype, void *rbuf, int *rcount,
        int *rtype, int *fcomm, int *fhandle, int *ierr) 
#endif
{
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
  *ierr = NBC_Ineighbor_xchg(sbuf, *scount, sdtype, rbuf, *rcount,
           rdtype, comm, handle);
}
#ifdef __cplusplus
}
#endif


