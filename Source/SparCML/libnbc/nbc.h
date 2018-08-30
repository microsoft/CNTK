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
#ifndef __NBC_H__
#define __NBC_H__

#include <mpi.h>
#include <pthread.h>
#include <semaphore.h>

#define USE_MPI /* set by the buildsystem! */

#ifdef __cplusplus
extern "C" {
#endif

/* Function return codes  */
#define NBC_OK 0 /* everything went fine */
#define NBC_SUCCESS 0 /* everything went fine (MPI compliant :) */
#define NBC_OOR 1 /* out of resources */
#define NBC_BAD_SCHED 2 /* bad schedule */
#define NBC_CONTINUE 3 /* progress not done */
#define NBC_DATATYPE_NOT_SUPPORTED 4 /* datatype not supported or not valid */
#define NBC_OP_NOT_SUPPORTED 5 /* operation not supported or not valid */
#define NBC_NOT_IMPLEMENTED 6
#define NBC_INVALID_PARAM 7 /* invalid parameters */
#define NBC_INVALID_TOPOLOGY_COMM 8 /* invalid topology attached to communicator */

// safeguard for non-MPI-2.2 compliant MPIs
#ifndef MPI_UNWEIGHTED
#define MPI_UNWEIGHTED 0
#endif

/* number of implemented collective functions */
#define NBC_NUM_COLL 19


/* a schedule is basically a pointer to some memory location where the
 * schedule array resides */ 
typedef void* NBC_Schedule;

/* used to hang off a communicator */
typedef struct {
  MPI_Comm mycomm; /* save the shadow communicator here */
  int tag;
#ifdef NBC_CACHE_SCHEDULE
  void *NBC_Dict[NBC_NUM_COLL]; /* this should point to a struct
                                      hb_tree, but since this is a
                                      public header-file, this would be
                                      an include mess :-(. So let's void
                                      it ...*/
  int NBC_Dict_size[NBC_NUM_COLL];
#endif
} NBC_Comminfo;


#ifdef USE_DCMF
#include <dcmf.h>

typedef struct {
  DCMF_Protocol_t *proto; 
  unsigned *rcvcounters;
  unsigned *sndcounters;
  unsigned *neighbors;
  unsigned *permutation;
  unsigned *ridx;
  unsigned nranks;
  unsigned rankIndex;
  unsigned outdeg; // MPI topology outdeg
  unsigned indeg; // MPI topology indeg
} dcmf_comminfo;

#define DCMF_TYPE_MANY_TO_MANY 0
#define DCMF_TYPE_BARRIER 1
#define DCMF_TYPE_ALLREDUCE 2

typedef struct _NBC_DCMF_Handle {
  DCMF_Request_t rrequest __attribute__((__aligned__(16)));;
  DCMF_Request_t srequest __attribute__((__aligned__(16)));;
  DCMF_Callback_t cb_m2m_done __attribute__((__aligned__(16)));; 
  DCMF_Callback_t cb_barr_done __attribute__((__aligned__(16)));; 
  dcmf_comminfo *comminfo __attribute__((__aligned__(16)));;
  unsigned done __attribute__((__aligned__(16)));; 
  unsigned barr_done __attribute__((__aligned__(16)));; 
  unsigned *slens __attribute__((__aligned__(16)));;
  unsigned *sdispls __attribute__((__aligned__(16)));;
  unsigned *rlens __attribute__((__aligned__(16)));;
  unsigned *rdispls __attribute__((__aligned__(16)));;
  unsigned cid;
  char *rbuf;
  char *sbuf;
  char type;
} NBC_DCMF_Handle __attribute__((__aligned__(16)));;
#endif

/* thread specific data */
typedef struct {
  MPI_Comm comm;
  MPI_Comm mycomm;
  long row_offset;
  int tag;
  volatile int req_count;
  pthread_mutex_t lock;
  sem_t semid;
#ifdef USE_OMPI
  /*ompi_request_t **req_array;*/
  MPI_Request *req_array;
#endif
#ifdef USE_MPI
  MPI_Request *req_array;
#endif
#ifdef USE_OFED
  void **req_array; /* OF_Request */
#endif
#ifdef USE_DCMF
  NBC_DCMF_Handle *dcmf_hndl; /* DCMF Request */
  MPI_Request *req_array; /* TODO:dummy request, this should really go away during a cleanup */
#endif
  NBC_Comminfo *comminfo;
  volatile NBC_Schedule *schedule;
  void *tmpbuf; /* temporary buffer e.g. used for Reduce */
/* TODO: we should make a handle pointer to a state later (that the user
 * can move request handles) */
} NBC_Handle;
typedef NBC_Handle NBC_Request;


/*******************************************************
 ****** external NBC functions are defined here *******
 *******************************************************/

/* profiling interface */
int PNBC_Ibarrier(MPI_Comm comm, NBC_Handle* handle);
int PNBC_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Ialltoallv(void* sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Igather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Igatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Iscatter(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Iscatterv(void* sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Iallgather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle);
int PNBC_Iallgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle);
int PNBC_Ialltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle);
int PNBC_Ireduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Iallreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Iscan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Ireduce_scatter(void* sendbuf, void* recvbuf, int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Icart_shift_xchg(void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype, int direction, int disp, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Ineighbor_xchg(void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle);
int PNBC_Ineighbor_xchgv(void *sbuf, int *scounts, int *sdispls, MPI_Datatype stype, void *rbuf, int *rcounts, int *rdispls, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ineighbor_alltoall(void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ineighbor_alltoallv(void *sbuf, int *scounts, int *sdispls, MPI_Datatype stype, void *rbuf, int *rcounts, int *rdispls, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ineighbor_alltoallw(void *sbuf, int *scounts, MPI_Datatype *stypes, void *rbuf, int *rcounts, MPI_Datatype *rtypes, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ineighbor_allgather(void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ineighbor_allgatherv(void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int *rcounts, int *displs, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle);

int PNBC_Comm_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[]);
int PNBC_Comm_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree, int *weighted);
int PNBC_Wait(NBC_Handle *handle, MPI_Status *status);
int PNBC_Testold(NBC_Handle *handle);
int PNBC_Test(NBC_Handle *handle, int *flag, MPI_Status *status);

/* external function prototypes */
int NBC_Ibarrier(MPI_Comm comm, NBC_Handle* handle);
int NBC_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ibcast_inter(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ialltoallv(void* sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle* handle);
int NBC_Igather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle);
int NBC_Igatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle);
int NBC_Iscatter(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle);
int NBC_Iscatterv(void* sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, NBC_Handle* handle);
int NBC_Iallgather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle);
int NBC_Iallgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle);
int NBC_Ialltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, NBC_Handle *handle);
int NBC_Ireduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, NBC_Handle* handle);
int NBC_Iallreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, NBC_Handle* handle);
int NBC_Iscan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ireduce_scatter(void* sendbuf, void* recvbuf, int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, NBC_Handle* handle);
int NBC_Icart_shift_xchg(void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype, int direction, int disp, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ineighbor_xchg(void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle);
int NBC_Ineighbor_xchgv(void *sbuf, int *scounts, int *sdispls, MPI_Datatype stype, void *rbuf, int *rcounts, int *rdispls, MPI_Datatype rtype, MPI_Comm comm, NBC_Handle* handle);

int NBC_Comm_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[]);
int NBC_Comm_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree, int *weighted);
int NBC_Wait(NBC_Handle *handle, MPI_Status *status);
int NBC_Testold(NBC_Handle *handle);
int NBC_Test(NBC_Handle *handle, int *flag, MPI_Status *status);

/* TODO: some hacks */
int NBC_Operation(void *buf3, void *buf1, void *buf2, MPI_Op op, MPI_Datatype type, int count, int forceDense);

void NBC_Reset_times();
void NBC_Print_times(double div);


#ifdef __cplusplus
}
#endif

#endif
