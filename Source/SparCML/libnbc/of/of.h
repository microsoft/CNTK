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
#ifndef __OF_H__
#define __OF_H__

#include "../config.h"

#define USE_RDMA

/* only if no progress thread runs - would deadlock the progress thread
 * if the operation finishes during the init (which can happen when init
 * calls OF_Test() */
#ifndef HAVE_PROGRESS_THREAD
#define TEST_ON_INIT
#endif

#define OF_DLEVEL 0
//#define DEBUG_STATE

#define MPICH_IGNORE_CXX_SEEK

#include "infiniband/verbs.h"
#include <stdarg.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <map>
#include <algorithm>
#include "../libdict/hb_tree.h"

#ifdef USE_THREAD
#include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OF_OK 0
#define OF_OOR 1
#define OF_CONTINUE 2
#define OF_THREAD_POLL 3 /* poll again, only used for progress thread */
#define OF_ERR 4

#define IB_RTR_SIZE 500 /* number of RTR buffers per peer */
#define IB_EAGER_LIMIT 256 /* OF eager size limit */
#define IB_EAGER_SIZE 100 /* number of OF eager buffers per peer */
#define IB_EAGER_THRES 95 /* threshold for ACK */

#if IB_EAGER_SIZE >= 255 
#error IB_EAGER_SIZE must be less than 255! (to fit in a piggyback byte!)
#endif

#define OF_UNUSED_TAG -1 /* marks element as empty */
#define OF_NACKLDG_TAG -2 /* not acknowledged yet tag */
#define OF_IMM_NULL ~0 /* highest imm */

/* redefine MPI functions to PMPI functions */
#define MPI_Abort          PMPI_Abort          
#define MPI_Alltoall       PMPI_Alltoall
#define MPI_Attr_get       PMPI_Attr_get
#define MPI_Attr_put       PMPI_Attr_put
#define MPI_Comm_rank      PMPI_Comm_rank
#define MPI_Comm_size      PMPI_Comm_size
#define MPI_Keyval_create  PMPI_Keyval_create
#define MPI_Recv           PMPI_Recv
#define MPI_Send           PMPI_Send
#define MPI_Type_extent    PMPI_Type_extent

typedef struct {
  volatile uint32_t r_key; /* r_key of peer */
  volatile uint64_t addr; /* addr of peer */
  volatile uint32_t recv_req; /* the element that contains recv request on the receiver  */
  volatile int tag; /* tag has to be at the end (we do not send with immediate, so tag indicates receive -> could be dangerous) */
} OF_RTR_message;

typedef volatile struct {
  volatile int8_t buf[IB_EAGER_LIMIT]; /* the data buffer - should be chosen that the whole structure size is 64 byte aligned */
  volatile uint8_t piggyack[2]; /* piggybacked eager free elements */
  volatile uint16_t size; /* the actual size of the message, after this size follows a flag (single byte) to poll on receiption */
  volatile int tag; /* the message tag */
} OF_Eager_message;

typedef struct {
  uint32_t r_key; /* r_key of peer */
  uint64_t addr; /* addr of peer */
} OF_Peer_info;

enum nbc_states {SEND_WAITING_RTR=0, 
                 SEND_SENDING_DATA, /* 1 */
                 RECV_SENDING_RTR, 
                 RECV_SENT_RTR,  /* 3 */
                 RECV_RECVD_DATA, 
                 SEND_SENDING_EAGER, /* 5 */
                 RECV_WAITING_EAGER,
                 EAGER_SENDING_DATA, /* 7 */
                 EAGER_SEND_INIT,
                 RNDV_RECV_INIT, /* 9 */
                 RECV_DONE, 
                 SEND_DONE};  /* 11 */

typedef struct {
  struct ibv_mr *mr; /* memory region handle - should really be a pointer to a memlist-element  ... */
  struct ibv_sge sr_sg_lst; /* the IB SG list */
  struct ibv_send_wr sr_desc; /* the IB SR descr. */
  struct OF_Comminfo *comminfo; /* the communicator info struct */
  int tag; /* our tag */
  int peer; /* the peer (dst for send, src for recv */
  void *buf; /* we need the buf for eager messages on the receiver side */
  int size; /* the message size */
  int sendel; /* the element in comminfo.send[element] which is use by this request to send RTR or EAGER messages - we want to free it after sending RTR/EAGER */
  int rtr_peer_free_elem; /* the element in comminfo.peer_free[element] which is use by this request to send RTR or eager messages - we want to free it after sending RTR/eager */
  enum nbc_states volatile status; /* this indicates the operation (send,recv) and the status of this op */
} OF_Req;
typedef OF_Req * OF_Request; /* this is necessary to allow realloc and moving of request lists */
/* WAH this has to be volatile because (OF_Req * volatile OF_Request)
 * the threads synchronize over this (NULL) ... BUT STL does not support
 * volatile :-/ ... see: 
 * http://www.informit.com/guides/content.aspx?g=cplusplus&seqNum=182&rl=1
 * ... so we're not doing it now!!! */

struct OF_Comminfo {
  struct ibv_qp **qp_arr; /* QPs to all ranks in this comm */
  struct ibv_comp_channel **compchan; /* CQ completion channels */
  struct ibv_cq **scq_arr; /* SR CQs for all ranks in this comm */
  //struct ibv_cq **rcq_arr; /* RR CQs for all ranks in this comm */

  int *max_inline_data; /* maximum inline data per QP */
  
  hb_tree **taglist; /* the new fancy AVL tree taglists - one for each peer (2-d matching is not possible :-(, so the peer-dimension is like a hash-table O(1) :) ) */
  
  /********** all the RTR stuff **************/
  struct ibv_mr **rtr_mr; /* memory region handles per proc - needed to free MR ... */
  OF_Peer_info *rtr_info; /* rtr r_key, addr for each host */
  volatile OF_RTR_message **rtr; /* rtr queue for each host - IB_RTR_SIZE elements  */
  volatile char **rtr_peer_free; /* the peer_free array indicates which entries are free on the other side (one per peer). The rtr_send array can *NOT* be used to substitute this because it is global! */
  pthread_mutex_t rtr_lock; /* Locks access to rtr_send and rtr_peer_free both these entires */
  struct ibv_mr *rtr_send_mr; /* memory region handle - needed to free MR ... */
  OF_RTR_message *rtr_send; /* send queue for me (only for RTR) - IB_RTR_SIZE elements */

  /********** all the EAGER stuff **************/
  struct ibv_mr **eager_mr; /* memory region handles per proc - needed to free MR ... */
  OF_Peer_info *eager_info; /* imm r_key, addr for each host */
  volatile OF_Eager_message **eager; /* IMM queue for each host - IB_EAGER_SIZE elements  */
  OF_Peer_info *eager_peer_free_info; /* imm r_key, addr for each host */
  struct ibv_mr **eager_peer_free_mr; /* memory region handles per proc - needed to free MR ... */
  volatile char ** volatile eager_peer_free; /* the peer_free array indicates which entries are free on the other side (one per peer). 
                                     the remote host RDMAs a -1 into the tag fiel to indicate successful reception,
                                     the index is the used to free the eager_send entry */
  struct ibv_mr *eager_send_mr; /* memory region handle - needed to free MR ... */
  volatile OF_Eager_message *eager_send; /* send queue for me (only for EAGER) - IB_EAGER_SIZE elements */
  int *eager_fill; /* array with number of occupied entries in eager receive buffer - per peer */


  int p; /* communicator size */
  int rank; /* my rank */

#ifndef USE_RDMA
  std::multimap<int, OF_Request> **tag_map; /* look up the request based on the tags */
  pthread_mutex_t* tag_map_locks;
#endif
};
typedef struct OF_Comminfo OF_Comminfo;

typedef struct OF_Taglstel {
  int tag; /* the tag -> key element */
  OF_Req *req; /* the request having this tag */
} OF_Taglstel;

struct OF_Worklistel {
  OF_Req *req; /* the request */
  volatile struct OF_Worklistel *next, *prev; 
};
typedef struct OF_Worklistel OF_Worklistel;

struct OF_Memlistel {
  void *buf;
  int size;
  struct ibv_mr *mr;
  uint64_t r_key;
};
typedef struct OF_Memlistel OF_Memlistel;

static inline int OF_Create_qp(int target, struct ibv_cq *scq, struct ibv_cq *rcq, struct ibv_qp **qp, int *max_inline_data, MPI_Comm comm);
static inline void OF_Taglist_delete(OF_Taglstel *entry);
static inline void OF_Taglist_delete_key(OF_Taglstel *k);
static inline int OF_Taglist_compare_entries(OF_Taglstel *a, OF_Taglstel *b, void *param);
static inline void OF_Memlist_memlist_delete(OF_Memlistel *entry);
static inline void OF_Memlist_delete_key(OF_Memlistel *k);
static inline int OF_Memlist_compare_entries(OF_Memlistel *a, OF_Memlistel *b, void *param);
#ifdef USE_THREAD
/* this function is only called by the progress thread */
int OF_Test_thread(OF_Request *request);
static inline int OF_Addreq_to_worklist(OF_Req *request);
#endif


/* external prototypes */
int OF_Testall(int count, OF_Request *requests, int *flag);
int OF_Test(OF_Request *request);
int OF_Wait(OF_Request *request);
int OF_Waitall(int count, OF_Request *requests);
int OF_Isend(void *buf, int count, MPI_Datatype type, int dst, int tag, MPI_Comm comm, OF_Request *request);
int OF_Irecv(void *buf, int count, MPI_Datatype type, int src, int tag, MPI_Comm comm, OF_Request *request);
void OF_Startall(int count, OF_Request *requests, unsigned long timeout);
void OF_Wakeup();
int OF_Init();
int OF_Waitany(int count, OF_Request *requests, int *index);
OF_Comminfo *OF_Comm_init(MPI_Comm comm);

static __inline__ void OF_DEBUG(int level, const char *fmt, ...) 
{ 
  va_list ap;
  int rank; 
 
  if(OF_DLEVEL >= level) { 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    
    printf("[LibOF - %i] ", rank); 
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end (ap);
  } 
}

#ifdef __cplusplus
}
#endif

#endif 
