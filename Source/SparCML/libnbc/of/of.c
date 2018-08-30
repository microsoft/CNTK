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
#include "of.h"

#include <list>
#include <iostream>
#include <vector>
#include <stack>
#include <poll.h>

/* TODO: we use this as synchronization object in the threaded case - do we need a lock here? */
static volatile int OF_Ginitialized=0;
static int OF_Gkeyval=MPI_KEYVAL_INVALID; 
static OF_Req **OF_Gopen_recvs; 
//static pthread_mutex_t OF_Gopen_recvs_lock = PTHREAD_MUTEX_INITIALIZER; 

#ifdef HAVE_PROGRESS_THREAD
static int OF_Gpipe[2];
FILE *OF_Gwfd, *OF_Grfd;
#endif

static struct OF_HCA_Info_t {

  struct ibv_context *devctxt;
  struct ibv_pd *pd;
  struct ibv_recv_wr dummy_rr; 
  struct ibv_recv_wr* dummy_bad_rr; /* everything is done with RDMA_WRITE_WITH_IMM - 
                                    do a dummy RR with no SG list should be fine */
  hb_tree *memlist; /* this is the libdict structure to hang off the search tree */
  //pthread_mutex_t lock;

  char eager_peer_flag; /* unused tag to RDMA as EAGER_RECVD */
  struct ibv_mr *eager_peer_flag_mr;

  int unsignalled_ctr; /* count the unsignalled WRs */ 
} OF_HCA_Info;

/* communicator key stuff functions */
static int OF_Key_copy(MPI_Comm oldcomm, int keyval, void *extra_state, 
                       void *attribute_val_in, void *attribute_val_out, int *flag) {
  /* delete the attribute in the new comm  - it will be created at the
   * first usage */
  *flag = 0;
  
  return MPI_SUCCESS;
}
  
static int OF_Key_delete(MPI_Comm comm, int keyval, void *attribute_val, 
                         void *extra_state) {
  OF_Comminfo *comminfo;
  
  if(keyval == OF_Gkeyval) {
    comminfo=(OF_Comminfo*)attribute_val;
    //free(comminfo);
  } else {
    printf("Got wrong keyval!(%i)\n", keyval); 
  }
  
  return MPI_SUCCESS;
} 


static void OF_Abort( int ret, const char *string ) {
  int r;

  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  printf("[LibOF - %i] OF-ERROR: %i in %s [perror: %s]\n", r, ret, string, strerror(ret*-1));
  MPI_Abort(MPI_COMM_WORLD, 1);
}

/* the global request pool ... requests are only allocated once and
 * then stored in the request stack */
static std::stack<OF_Request> OF_Grequests;
static inline OF_Request getrequest(void) {
  OF_Request request;

  if(OF_Grequests.empty()) {
    request = (OF_Request)malloc(sizeof(OF_Req));
    if(request == NULL) printf("malloc error in getrequest()\n");
    return request;
   } else {
    request = OF_Grequests.top();
    OF_Grequests.pop();
  }

  return request;
}

static inline void freerequest(OF_Request *request) {

  OF_Grequests.push(*request);
  *request = NULL;
}

int OF_Init() {
  int ret;

  /* keyval is not initialized yet, we have to init it */
  if(MPI_KEYVAL_INVALID == OF_Gkeyval) {
    ret = MPI_Keyval_create(OF_Key_copy, OF_Key_delete, &(OF_Gkeyval), NULL);
    if((MPI_SUCCESS != ret)) { printf("Error in MPI_Keyval_create() (%i)\n", ret); return OF_OOR; }
  }

  /* the init stuff */
  struct ibv_device **devs;
  int num_devices;
  
  /* list all devices */
  devs = ibv_get_device_list(&num_devices);
  //fprintf(stderr, "[INFO] found %d adapter(s)\n", num_devices);

  //pthread_mutex_init (&OF_HCA_Info.lock, NULL);

  /* open first device */
  OF_HCA_Info.devctxt = ibv_open_device(devs[0]);
  if(OF_HCA_Info.devctxt == NULL) OF_Abort(0, "ibv_open_device()");

  /* delete device list - should be safe after opening one */
  ibv_free_device_list(devs);

  /* allocate protection domain */
  OF_HCA_Info.pd = ibv_alloc_pd(OF_HCA_Info.devctxt);
  if(OF_HCA_Info.pd == NULL) OF_Abort(0, "ibv_alloc_pd()");

  /* prepare the dummy RR */
  OF_HCA_Info.dummy_rr.sg_list = NULL;
  OF_HCA_Info.dummy_rr.num_sge = 0;
  OF_HCA_Info.dummy_rr.wr_id = 0; /* will be set before posting */
  OF_HCA_Info.dummy_rr.next = NULL;

  OF_HCA_Info.eager_peer_flag = 0;
  OF_HCA_Info.eager_peer_flag_mr = ibv_reg_mr(OF_HCA_Info.pd, (void*)&OF_HCA_Info.eager_peer_flag, 
                                sizeof(char), 
                                (enum ibv_access_flags)(IBV_ACCESS_REMOTE_WRITE | 
                                                        IBV_ACCESS_LOCAL_WRITE | 
                                                        IBV_ACCESS_REMOTE_READ));
  if(OF_HCA_Info.eager_peer_flag_mr == NULL) { printf("error in ibv_reg_mr(OF_HCA_Info.eager_peer_flag_mr)\n"); return OF_OOR; }

  /* tree for the registration cache */
  OF_HCA_Info.memlist = hb_tree_new((dict_cmp_func)OF_Memlist_compare_entries, 
                                    (dict_del_func) OF_Memlist_delete_key, 
                                    (dict_del_func)OF_Memlist_memlist_delete);
  if(OF_HCA_Info.memlist == NULL) { printf("error in hb_dict_new()\n"); return OF_OOR; }
  /* allocate open recv array  */
  OF_Gopen_recvs = (OF_Req**)malloc(IB_RTR_SIZE*sizeof(OF_Req*));
  if(OF_Gopen_recvs == NULL) { printf("malloc() error\n"); return OF_OOR; }
  for(int i=0; i<IB_RTR_SIZE;i++) { 
    OF_Gopen_recvs[i]=NULL;
    //OF_Gopen_recv_free.push(i);
  }

#ifdef HAVE_PROGRESS_THREAD
  /*initialize global file descriptor */
  ret = pipe(OF_Gpipe); 
  if(ret < 0) OF_Abort(0, "pipe()");
  OF_Gwfd=fdopen(OF_Gpipe[1], "w");
  OF_Grfd=fdopen(OF_Gpipe[0], "r");
#endif

  OF_Ginitialized = 1;

  return OF_OK;
}

OF_Comminfo *OF_Comm_init(MPI_Comm comm) {
  OF_Comminfo *comminfo;
  int res, flag;

  if(!OF_Ginitialized) OF_Init();

  res = MPI_Attr_get(comm, OF_Gkeyval, &comminfo, &flag);
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_get() (%i)\n", res); return NULL; }
  if (!flag) {
    unsigned long *a2abuf1, *a2abuf2;
    int p, i, j, rank;
    
    res = MPI_Comm_size(comm, &p);
    res = MPI_Comm_rank(comm, &rank);

#ifdef HAVE_PROGRESS_THREAD
#ifdef HAVE_RT_THREAD
    if(!rank) printf("[LibOF] initializing communicator %lu (rt-threaded)\n", (unsigned long)comm);
#else
    if(!rank) printf("[LibOF] initializing communicator %lu (threaded)\n", (unsigned long)comm);
#endif
#else
    if(!rank) printf("[LibOF] initializing communicator %lu\n", (unsigned long)comm);
#endif
  
    /* we have to create a new one */
    comminfo = (OF_Comminfo*)malloc(sizeof(OF_Comminfo));
    if(comminfo == NULL) { printf("Error in malloc()\n"); return NULL; }

    comminfo->p = p;
    comminfo->rank = rank;

    //printf("[%i] build up %i connections in comm %p \n", rank, p-1, comm);
    /* allocate QPs */
    comminfo->qp_arr = (struct ibv_qp**)malloc(p*sizeof(struct ibv_qp*));
    if(comminfo->qp_arr == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate completion channels */
    comminfo->compchan = (struct ibv_comp_channel**)malloc(p*sizeof(struct ibv_comp_channel*));
    if(comminfo->compchan == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate SR CQs */
    comminfo->scq_arr = (struct ibv_cq**)malloc(p*sizeof(struct ibv_cq*));
    if(comminfo->scq_arr == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate RR CQs */
    /*comminfo->rcq_arr = (struct ibv_cq**)malloc(p*sizeof(struct ibv_cq*));
    if(comminfo->rcq_arr == NULL) { printf("malloc() error\n"); return NULL; }*/
    /* allocate max_inline_data */
    comminfo->max_inline_data = (int*)malloc(p*sizeof(int));
    if(comminfo->max_inline_data == NULL) { printf("malloc() error\n"); return NULL; }

    /* allocate rtr send queue */
    comminfo->rtr_send=(OF_RTR_message*)malloc(IB_RTR_SIZE*sizeof(OF_RTR_message));
    if(comminfo->rtr_send == NULL) { printf("malloc() error\n"); return NULL; }
    for(i=0; i<IB_RTR_SIZE;i++) comminfo->rtr_send[i].tag=OF_UNUSED_TAG;
    /* initialize the rtr_send_lock */
    //pthread_mutex_init (&(comminfo->rtr_lock), NULL);
    /* allocate rtr queue */
    comminfo->rtr=(volatile OF_RTR_message**)malloc(p*sizeof(OF_RTR_message*));
    if(comminfo->rtr== NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate rtr_peer_free queue */
    comminfo->rtr_peer_free = (volatile char**)malloc(p*sizeof(char*));
    if(comminfo->rtr_peer_free == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate rtr info */
    comminfo->rtr_info=(OF_Peer_info*)malloc(p*sizeof(OF_Peer_info));
    if(comminfo->rtr_info == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate rtr memory region handle */
    comminfo->rtr_mr=(struct ibv_mr**)malloc(p*sizeof(struct ibv_mr*));
    if(comminfo->rtr_mr == NULL) { printf("malloc() error\n"); return NULL; }
    
    /* allocate eager send queue */
    comminfo->eager_send=(OF_Eager_message*)malloc(IB_EAGER_SIZE*sizeof(OF_Eager_message));
    if(comminfo->eager_send == NULL) { printf("malloc() error\n"); return NULL; }
    for(i=0; i<IB_EAGER_SIZE;i++) comminfo->eager_send[i].tag=OF_UNUSED_TAG;
    /* allocate eager fill array*/
    comminfo->eager_fill=(int*)malloc(p*sizeof(int));
    if(comminfo->eager_fill == NULL) { printf("malloc() error\n"); return NULL; }

    /* allocate eager queue */
    comminfo->eager=(OF_Eager_message**)malloc(p*sizeof(OF_Eager_message*));
    if(comminfo->eager== NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate eager_peer_free_info  */
    comminfo->eager_peer_free_info=(OF_Peer_info*)malloc(p*sizeof(OF_Peer_info));
    if(comminfo->eager_peer_free_info == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate eager_peer_free queue */
    comminfo->eager_peer_free = (volatile char**)malloc(p*sizeof(char*));
    if(comminfo->eager_peer_free == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate eager_peer_free memory region handle */
    comminfo->eager_peer_free_mr=(struct ibv_mr**)malloc(p*sizeof(struct ibv_mr*));
    if(comminfo->eager_peer_free_mr == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate eager info */
    comminfo->eager_info=(OF_Peer_info*)malloc(p*sizeof(OF_Peer_info));
    if(comminfo->eager_info == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate eager memory region handle */
    comminfo->eager_mr=(struct ibv_mr**)malloc(p*sizeof(struct ibv_mr*));
    if(comminfo->eager_mr == NULL) { printf("malloc() error\n"); return NULL; }
    
    /* allocate a2abuf1 */
    a2abuf1=(unsigned long*)malloc(2*p*sizeof(unsigned long));
    if(a2abuf1 == NULL) { printf("malloc() error\n"); return NULL; }
    /* allocate a2abuf2 */
    a2abuf2=(unsigned long*)malloc(2*p*sizeof(unsigned long));
    if(a2abuf2 == NULL) { printf("malloc() error\n"); return NULL; }
    
    for(i = 0; i < p; i++) {
      if(i == rank) continue;
      
      comminfo->compchan[i] = ibv_create_comp_channel(OF_HCA_Info.devctxt);
      if(comminfo->compchan[i] == NULL) OF_Abort(0, "ibv_create_comp_channel()");

      comminfo->scq_arr[i] = ibv_create_cq(OF_HCA_Info.devctxt, 
                          1000 /* minimum number of entries required for CQ */,
                          NULL /* Consumer-supplied context returned for completion events */,
                          comminfo->compchan[i] /* Completion channel where completion events will be queued */,
                          0 /* Completion vector used to signal completion events */);
      if(comminfo->scq_arr[i] == NULL) OF_Abort(0, "ibv_create_cq(scq)");

      res = ibv_req_notify_cq(comminfo->scq_arr[i], 0);
      if(res < 0) OF_Abort(0, "ibv_req_notify_cq()"); 
#if 0
      comminfo->rcq_arr[i] = ibv_create_cq(OF_HCA_Info.devctxt, 
                          1000 /* minimum number of entries required for CQ */,
                          NULL /* Consumer-supplied context returned for completion events */,
                          NULL /* Completion channel where completion events will be queued */,
                          0 /* Completion vector used to signal completion events */);
      if(comminfo->rcq_arr[i] == NULL) OF_Abort(0, "ibv_create_cq(scq)");
#endif
      res = OF_Create_qp(i, comminfo->scq_arr[i], comminfo->scq_arr[i], 
                         &comminfo->qp_arr[i], &comminfo->max_inline_data[i], comm);
      if(res != 0) { printf("Error in OF_Create_qp (%i)\n", res); return NULL; }

      /* allocate rtr element */
      comminfo->rtr[i] = (OF_RTR_message*)malloc(sizeof(OF_RTR_message)*IB_RTR_SIZE);
      if(comminfo->rtr[i] == NULL) { printf("malloc() error\n"); return NULL; }
      for(j=0; j<IB_RTR_SIZE; j++) comminfo->rtr[i][j].tag = OF_UNUSED_TAG;
      
      /* allocate rtr_peer_free element */
      comminfo->rtr_peer_free[i] = (char*)malloc(sizeof(char)*IB_RTR_SIZE);
      if(comminfo->rtr_peer_free[i] == NULL) { printf("malloc() error\n"); return NULL; }
      for(j=0; j<IB_RTR_SIZE; j++) comminfo->rtr_peer_free[i][j] = 0;

      /* allocate eager element */
      comminfo->eager[i] = (OF_Eager_message*)malloc(sizeof(OF_Eager_message)*IB_EAGER_SIZE);
      if(comminfo->eager[i] == NULL) { printf("malloc() error\n"); return NULL; }
      for(j=0; j<IB_EAGER_SIZE; j++) comminfo->eager[i][j].tag = OF_UNUSED_TAG;
      
      /* allocate eager_peer_free element */
      comminfo->eager_peer_free[i] = (volatile char*)malloc(sizeof(char)*IB_EAGER_SIZE);
      if(comminfo->eager_peer_free[i] == NULL) { printf("malloc() error\n"); return NULL; }
      for(j=0; j<IB_EAGER_SIZE; j++) {
        comminfo->eager_peer_free[i][j] = 0;
      }
    }
  
    /* exchange the rtr information */
    for(i=0; i<p;i++) {
      if (rank == i) continue;
      /* register rtr buffer */
      comminfo->rtr_mr[i] = ibv_reg_mr(OF_HCA_Info.pd, (void*)comminfo->rtr[i], 
                                sizeof(OF_RTR_message)*IB_RTR_SIZE, 
                                (enum ibv_access_flags)(IBV_ACCESS_REMOTE_WRITE | 
                                                        IBV_ACCESS_LOCAL_WRITE | 
                                                        IBV_ACCESS_REMOTE_READ));
      if(comminfo->rtr_mr[i] == NULL) OF_Abort(0, "ibv_reg_mr(rtr_buffer)");

      a2abuf1[2*i] = (unsigned long)comminfo->rtr_mr[i]->rkey;
      a2abuf1[2*i+1] = (unsigned long)comminfo->rtr[i];
    }
    MPI_Alltoall(a2abuf1, 2, MPI_UNSIGNED_LONG, a2abuf2, 2, MPI_UNSIGNED_LONG, comm);
    for(i=0; i<p;i++) {
      comminfo->rtr_info[i].r_key=a2abuf2[2*i];
      comminfo->rtr_info[i].addr=a2abuf2[2*i+1];
    }
    
    /* exchange the eager information */
    for(i=0; i<p;i++) {
      if (rank == i) continue;
      /* register eager buffer */
      comminfo->eager_mr[i] = ibv_reg_mr(OF_HCA_Info.pd, (void*)comminfo->eager[i], 
                                sizeof(OF_Eager_message)*IB_EAGER_SIZE, 
                                (enum ibv_access_flags)(IBV_ACCESS_REMOTE_WRITE | 
                                                        IBV_ACCESS_LOCAL_WRITE | 
                                                        IBV_ACCESS_REMOTE_READ));
      if(comminfo->eager_mr[i] == NULL) OF_Abort(0, "ibv_reg_mr(eager_buffer)");

      a2abuf1[2*i] = (unsigned long)comminfo->eager_mr[i]->rkey;
      a2abuf1[2*i+1] = (unsigned long)comminfo->eager[i];
    }
    MPI_Alltoall(a2abuf1, 2, MPI_UNSIGNED_LONG, a2abuf2, 2, MPI_UNSIGNED_LONG, comm);
    for(i=0; i<p;i++) {
      comminfo->eager_info[i].r_key=a2abuf2[2*i];
      comminfo->eager_info[i].addr=a2abuf2[2*i+1];
    }
    
    /* exchange the eager_peer_free information */
    for(i=0; i<p;i++) {
      if (rank == i) continue;
      /* register rtr buffer */
      comminfo->eager_peer_free_mr[i] = ibv_reg_mr(OF_HCA_Info.pd, (void*)comminfo->eager_peer_free[i], 
                                sizeof(char)*IB_EAGER_SIZE, 
                                (enum ibv_access_flags)(IBV_ACCESS_REMOTE_WRITE | 
                                                        IBV_ACCESS_LOCAL_WRITE | 
                                                        IBV_ACCESS_REMOTE_READ));
      if(comminfo->eager_peer_free_mr[i] == NULL) OF_Abort(0, "ibv_reg_mr(eager_peer_free)");

      a2abuf1[2*i] = (unsigned long)comminfo->eager_peer_free_mr[i]->rkey;
      a2abuf1[2*i+1] = (unsigned long)comminfo->eager_peer_free[i];
    }
    MPI_Alltoall(a2abuf1, 2, MPI_UNSIGNED_LONG, a2abuf2, 2, MPI_UNSIGNED_LONG, comm);
    for(i=0; i<p;i++) {
      comminfo->eager_peer_free_info[i].r_key=a2abuf2[2*i];
      comminfo->eager_peer_free_info[i].addr=a2abuf2[2*i+1];
    }
    
    for(i=0; i<p;i++) {
      if (rank == i) continue;
      /* prepost IB_RTR_SIZE many recvs */
      for(int iter_rtr=0; iter_rtr<IB_RTR_SIZE; iter_rtr++) { 
        res = ibv_post_recv(comminfo->qp_arr[i], &OF_HCA_Info.dummy_rr, &OF_HCA_Info.dummy_bad_rr);
        if(res != 0) OF_Abort(res, "ibv_post_recv(Comm_init)");
      }
    }
    
    /* register rtr send buffer */
    comminfo->rtr_send_mr = ibv_reg_mr(OF_HCA_Info.pd, comminfo->rtr_send, 
                              sizeof(OF_RTR_message)*IB_RTR_SIZE, 
                              (enum ibv_access_flags)(IBV_ACCESS_REMOTE_WRITE | 
                                                      IBV_ACCESS_LOCAL_WRITE | 
                                                      IBV_ACCESS_REMOTE_READ));
    if(comminfo->rtr_send_mr == NULL) OF_Abort(0, "ibv_reg_mr(rtr_send buffer)");

    /* register eager send buffer */
    comminfo->eager_send_mr = ibv_reg_mr(OF_HCA_Info.pd, (void*)comminfo->eager_send, 
                              sizeof(OF_Eager_message)*IB_EAGER_SIZE, 
                              (enum ibv_access_flags)(IBV_ACCESS_REMOTE_WRITE | 
                                                      IBV_ACCESS_LOCAL_WRITE | 
                                                      IBV_ACCESS_REMOTE_READ));
    if(comminfo->eager_send_mr == NULL) OF_Abort(0, "ibv_reg_mr(eager_send buffer)");

    /* put the new attribute to the comm */
    res = MPI_Attr_put(comm, OF_Gkeyval, comminfo);
    if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_put() (%i)\n", res); return NULL; }
    
#ifndef USE_RDMA
    /* Allocate the per peer tag_map */
    comminfo->tag_map = new std::multimap<int, OF_Request>* [p];
    if (NULL == comminfo->tag_map) {printf ("malloc() error\n"); return NULL;}
    for (i=0; i<p; i++) {
        if (i == rank) continue;
        comminfo->tag_map[i] = new std::multimap<int, OF_Request>();
        if (NULL == comminfo->tag_map[i]) {printf("malloc() failed\n"); return NULL;}
    }

    /*comminfo->tag_map_locks = new pthread_mutex_t [p];
    if (NULL == comminfo->tag_map_locks) {printf ("malloc() error\n"); return NULL;}
    for (i=0; i<p; i++) if (0 != pthread_mutex_init (&(comminfo->tag_map_locks[i]), NULL)) {
        printf ("Error initializing the tag_map_locks\n"); return NULL; 
    }*/
#endif
    
  }

  return comminfo;
}

int OF_Create_qp(int target, struct ibv_cq *scq, struct ibv_cq *rcq, 
                 struct ibv_qp **qp, int *max_inline_data, MPI_Comm comm) {
  int ret, r;
  struct ibv_port_attr port_attr;
  struct ibv_qp_init_attr qp_attr;
  struct ibv_qp_attr attr;
  
  MPI_Comm_rank(comm, &r);
  
  /* get my LID */
  ret = ibv_query_port(OF_HCA_Info.devctxt, 1 /* OFED begins to count with 1! */, &port_attr);
  if(ret != 0) OF_Abort(ret, "ibv_query_port()");

  //printf("[%i] found LID: %u\n", r, (unsigned int)port_attr.lid);

  qp_attr.qp_context = OF_HCA_Info.devctxt; 
  qp_attr.send_cq = scq;
  qp_attr.recv_cq = rcq;
  qp_attr.srq = NULL; 
  qp_attr.cap.max_send_wr = 10000; // IB_RTR_SIZE*2; 
  qp_attr.cap.max_recv_wr = 10000; //IB_RTR_SIZE*2; 
  qp_attr.cap.max_send_sge = 1; 
  qp_attr.cap.max_recv_sge = 1; 
  qp_attr.cap.max_inline_data = 100; //IB_EAGER_LIMIT; /* TODO: check if inlining makes sense */
  qp_attr.qp_type = IBV_QPT_RC; /* reliable connection */
  qp_attr.sq_sig_all = 1;  /* TODO: play with it */
              
  *qp = ibv_create_qp(OF_HCA_Info.pd, &qp_attr);
  if(*qp == NULL) OF_Abort(0, "ibv_create_qp()");
  *max_inline_data = qp_attr.cap.max_inline_data;

  //printf("[%i] got QP num: %u\n", r, (*qp)->qp_num);

  {
    int remlid = port_attr.lid; /* to send an int - for heterogeneous things */
    int remqp_num = (*qp)->qp_num; /* to send an int - heterogeneous */

    /* TODO: MPI_Sendrecv - otherwise possible deadlock!!!! */
    MPI_Send(&remlid, 1, MPI_INT, target, 0, comm);
    MPI_Recv(&remlid, 1, MPI_INT, target, 0, comm, MPI_STATUS_IGNORE);
    
    MPI_Send(&remqp_num, 1, MPI_INT, target, 0, comm);
    MPI_Recv(&remqp_num, 1, MPI_INT, target, 0, comm, MPI_STATUS_IGNORE);
    //printf("[%i] remote: %u:%u\n", r, remlid, remqp_num);

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0; /* first partition key */
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | 
                           IBV_ACCESS_REMOTE_READ | 
                           IBV_ACCESS_REMOTE_ATOMIC; /* all */
    attr.port_num = 1; /* physical port */
    /* go from RST to INIT */
    ret = ibv_modify_qp(*qp, &attr, 
                       (enum ibv_qp_attr_mask)
                       (IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS | IBV_QP_PORT));
    if(ret != 0) OF_Abort(ret, "ibv_modify_qp(RST - INIT)");

    
    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;
    /* address vector */
    attr.ah_attr.dlid           = (uint16_t)remlid; /* remote lid */
    attr.ah_attr.sl             = 0; /* TODO: guessed */
    attr.ah_attr.src_path_bits  = 0; /* TODO: guessed */
    attr.ah_attr.static_rate    = 0; /* TODO: guessed */
    attr.ah_attr.is_global      = 0; /* TODO: guessed */
    attr.ah_attr.port_num       = 1; /* TODO: guessed */
    /* packet sequence number */
    attr.rq_psn                 = 0;
    /* number of responder resources for RDMA R + Atomic */
    attr.max_dest_rd_atomic          = 4; /* TODO tune here? */
    /* minimum RNR NAK timer */
    attr.min_rnr_timer          = 3; /* TODO guessed */
    /* dest QP number */
    attr.dest_qp_num            = (uint32_t)remqp_num;
    /* MTU */
    attr.path_mtu               = IBV_MTU_2048;
    /* go from INIT to RTR */
    ret = ibv_modify_qp(*qp, &attr, 
                       (enum ibv_qp_attr_mask)
                       (IBV_QP_STATE | IBV_QP_AV | IBV_QP_RQ_PSN | 
                       IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER | 
                       IBV_QP_DEST_QPN | IBV_QP_PATH_MTU));
    if(ret != 0) OF_Abort(ret, "ibv_modify_qp(INIT - RTR)");
    
    attr.qp_state       = IBV_QPS_RTS;
    attr.sq_psn         = 0;
    attr.timeout        = 10;
    attr.retry_cnt      = 7;
    attr.rnr_retry      = 7;
    attr.max_rd_atomic  = 4;
                            
    /* go from RTR to RTS */
    ret = ibv_modify_qp(*qp, &attr, (enum ibv_qp_attr_mask)
                       (IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | 
                       IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC));
    if(ret != 0) OF_Abort(ret, "ibv_modify_qp(INIT - RTS)");

  }

  return 0;
}

static inline int OF_Register_mem(void *buf, int size, struct ibv_mr **mr) {
  int res, rank;
  OF_Memlistel *memel, *newel, keyel;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  keyel.buf = buf;
  keyel.size = size;
  memel = (OF_Memlistel*)hb_tree_search(OF_HCA_Info.memlist, &keyel);
  if(memel != NULL) {
    *mr = memel->mr;
    return OF_OK;
  }

  /*
  memregion keyel;
  keyel[0] = (char*)buf;
  keyel[1] = (char*)buf+size;

  std::vector<memregion> v;
  OF_HCA_Info.memlist.find_within_range(keyel, std::back_inserter(v));

  if(v.size() > 0) {
    *mr = 
    return OF_OK;
  }*/

  *mr = ibv_reg_mr(OF_HCA_Info.pd, buf, size, (enum ibv_access_flags)
                  (IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
  if(*mr == NULL) OF_Abort(0, "ibv_reg_mr(userbuf)");

  newel = (OF_Memlistel*)malloc(sizeof(OF_Memlistel));
  newel->buf = buf;
  newel->size = size;
  newel->mr = *mr;
  newel->r_key = (*mr)->rkey;
	res = hb_tree_insert(OF_HCA_Info.memlist, newel, newel, 0);
  if(res != 0) 
      printf("[%i] error in dict_insert() (%i) while inserting region from %lu to %lu\n", 
      rank, res, (unsigned long)newel->buf, (unsigned long)(newel->buf)+newel->size);

  return OF_OK;
}
  
/* CALLED BY THE MAIN THREAD */
int OF_Isend(void *buf, int count, MPI_Datatype type, int dst, int tag, 
            MPI_Comm comm, OF_Request *request) {
  int res;
  MPI_Aint ext;

  if(0 == count) {
    *request = NULL;
    return OF_OK;
  }

  *request = getrequest();
  OF_Request const req = *request;
  
  MPI_Type_extent(type, &ext);

  req->tag = tag;
  req->peer= dst;
  
  req->comminfo = OF_Comm_init(comm);
  if(req->comminfo == NULL) { printf("Error in OF_Comm_init()\n"); return OF_OOR; }

  if(count*ext < IB_EAGER_LIMIT) { /* use eager protocol */

    req->peer = dst;
    req->buf = buf;
    req->size = ext*count; /* TODO: Danger with datatypes */
    req->status = EAGER_SEND_INIT;
#ifdef DEBUG_STATE        
    printf("[LibOF - %i] init req %p (tag: %i) to EAGER_SEND_INIT\n", req->comminfo->rank, req, req->tag);
#endif

#ifdef TEST_ON_INIT
    OF_Test(request);
#endif
  } else { /* use rendezvous protocol */
#ifndef USE_RDMA
    // Adding the (tag,request*) pair into a map so that we can match the incoming 
    // receives
    //pthread_mutex_lock (&(req->comminfo->tag_map_locks[dst]));
    req->comminfo->tag_map[dst]->insert(std::pair<int, OF_Request>(tag, req));
    //pthread_mutex_unlock (&(req->comminfo->tag_map_locks[dst]));
#endif
    /* register memory region for send */
    res = OF_Register_mem(buf, count*ext, &req->mr); /* TODO: count*ext Danger for datatypes ... */
    
    /* initialize sr_desc as far as we can (remote r_key and addr are
     * missing set after we received RTR */
    req->sr_sg_lst.addr = (uint64_t)buf; 
    req->sr_sg_lst.length = count*ext;  /* TODO: count*ext Danger for datatypes ... */
    req->sr_sg_lst.lkey = req->mr->lkey;
    req->sr_desc.wr_id = (uint64_t)req;
    req->sr_desc.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; 
    req->sr_desc.send_flags = IBV_SEND_SIGNALED; 
    req->sr_desc.sg_list = &req->sr_sg_lst; 
    req->sr_desc.num_sge = 1; 
    req->sr_desc.next = NULL; 

    req->status = SEND_WAITING_RTR;
    //while(req->status == SEND_WAITING_RTR) OF_Test(request);
#ifdef DEBUG_STATE        
    printf("[LibOF - %i] init req %p (tag: %i) to SEND_WAITING_RTR\n", req->comminfo->rank, req, req->tag);
#endif
  }
  
  return OF_OK;
}

int OF_Irecv(void *buf, int count, MPI_Datatype type, int src, int tag, 
            MPI_Comm comm, OF_Request *request) {
  int res;
  MPI_Aint ext;

  if(0 == count) {
    *request = NULL;
    return OF_OK;
  }

  *request = getrequest(); 
  OF_Request const req = *request;
  
  MPI_Type_extent(type, &ext);

  req->tag = tag;
  req->peer = src;
  req->buf = buf;
  
  req->comminfo = OF_Comm_init(comm); 
  if(req->comminfo == NULL) { printf("Error in OF_Comm_init()\n"); return OF_OOR; }

  if(count*ext < IB_EAGER_LIMIT) { /* use eager protocol */
    req->status = RECV_WAITING_EAGER;
#ifdef DEBUG_STATE        
    printf("[LibOF - %i] init req %p (tag: %i) to RECV_WAITING_EAGER\n", req->comminfo->rank, req, req->tag);
#endif
  } else { /* use rendezvous protocol */
  
    req->peer = src;
    req->buf = buf;
    req->size = ext*count; /* TODO: Danger with datatypes */
    req->status = RNDV_RECV_INIT;

#ifdef TEST_ON_INIT
    OF_Test(request);
#endif

#ifdef DEBUG_STATE        
    printf("[LibOF - %i] init req %p (tag: %i) to RECV_RECV_INIT\n", req->comminfo->rank, req, req->tag);
#endif
  }

  return OF_OK;
}

/* this function is used by the user thread directly */
int OF_Test(OF_Request *request) {
  int i;
  int j;
  int res;
  struct ibv_wc wc; /* work completion descriptor */
  volatile OF_Req* tmpreq;
  const OF_Request req = *request;
  int retval = OF_CONTINUE;

  if(NULL == *request) {
    return OF_OK;
  }

  if((req->status == SEND_DONE) || (req->status == RECV_DONE)) {
    freerequest(request);
    return OF_OK;
  }

#ifdef USE_RDMA
  /* if I wait for RTR - search rtr array for my tag ... */
  if(req->status == SEND_WAITING_RTR) {
    for(i=0; i<IB_RTR_SIZE; i++) {
      const int tag = req->comminfo->rtr[req->peer][i].tag; /* pull volatile into register */
      if(tag == req->tag) {
        struct ibv_send_wr *bad_wr;
        
        req->sr_desc.wr.rdma.rkey = (uint32_t)req->comminfo->rtr[req->peer][i].r_key; 
        req->sr_desc.wr.rdma.remote_addr = req->comminfo->rtr[req->peer][i].addr; /* TODO: 64 Bit */ 
        /* send the 'free' rtr element back */
        req->sr_desc.imm_data = (uint32_t)req->comminfo->rtr[req->peer][i].recv_req; 
        req->comminfo->rtr[req->peer][i].tag = OF_UNUSED_TAG;
        
        res = ibv_post_send(req->comminfo->qp_arr[req->peer], &req->sr_desc, &bad_wr);
        if(res != 0) OF_Abort(res, "ibv_post_send()");
        
        req->status = SEND_SENDING_DATA;
#ifdef DEBUG_STATE        
        printf("[LibOF - %i] req %p (tag: %i) from SEND_WAITING_RTR to SEND_SENDING_DATA\n", req->comminfo->rank, req, req->tag);
#endif
        break;
      }
    }
  }
#endif

  if(req->status == RECV_WAITING_EAGER) {
    int fill = 0;
    /* ok, poll all eager slots we have from the peer we wait for */
    /* TODO: the usee can fill all the buffers up with tags that I am
     * not waiting for, send another message (that does not fit anymore)
     * and wait for this one -> deadlock! In LibNBC terms:
     * floor(IB_EAGER_SIZE/p) is the maximum number of outstanding colls per
     * comm if they are not waited for ... */
    for(i=0; i<IB_EAGER_SIZE; i++) {
      const int tag = req->comminfo->eager[req->peer][i].tag; /* it's volatile - pull it into register */
      if(tag != OF_UNUSED_TAG) fill++; /* we found an occupied buffer */
      /* we don't need flag polling anymore because the tag itself is
       * the flag - and only 32 bytes (written atomically) */
      if(tag == req->tag) {
        //printf("[%i] found eager message from peer %i at element %i (addr: %lu tag: %i, size: %i)\n", req->comminfo->rank, req->peer, i, (unsigned long)(&req->comminfo->eager[req->peer][i]), (int)req->comminfo->eager[req->peer][i].tag, req->comminfo->eager[req->peer][i].size);
        /* copy message to recv buffer */
        void *dataptr = (void*)(req->comminfo->eager[req->peer][i].buf + IB_EAGER_LIMIT - \
                        req->comminfo->eager[req->peer][i].size);
        memcpy(req->buf, dataptr, req->comminfo->eager[req->peer][i].size);

        /* check if there was a piggyack'd message */
        if(255 != req->comminfo->eager[req->peer][i].piggyack[0]) {
          //printf("freeing %hhi piggy-acked\n", req->comminfo->eager[req->peer][i].piggyack[0]);
          req->comminfo->eager_peer_free[req->peer][req->comminfo->eager[req->peer][i].piggyack[0]] = 0;
        }
        if(255 != req->comminfo->eager[req->peer][i].piggyack[1]) {
          //printf("freeing %hhi piggy-acked\n", req->comminfo->eager[req->peer][i].piggyack[1]);
          req->comminfo->eager_peer_free[req->peer][req->comminfo->eager[req->peer][i].piggyack[1]] = 0;
        }

        if(fill > IB_EAGER_THRES) {
          //printf("ack threshhold reached ... %i > %i\n", IB_EAGER_THRES, fill);
          req->comminfo->eager[req->peer][i].tag = OF_UNUSED_TAG;

          /* RDMA into the free-buffer on the sender to indicate that my 
           * buffer can be reused */
          struct ibv_sge sr_sg_lst; /* the IB SG list */
          struct ibv_send_wr sr_wr, *bad_swr; /* the IB SR descr. */

          sr_sg_lst.addr = (uint64_t)(&OF_HCA_Info.eager_peer_flag);
          sr_sg_lst.length = sizeof(char);
          //sr_sg_lst.lkey = OF_HCA_Info.eager_peer_flag_mr->lkey;
          sr_wr.wr_id = (uint64_t)1;
          sr_wr.opcode = IBV_WR_RDMA_WRITE;
          sr_wr.imm_data= 0;
          sr_wr.send_flags = IBV_SEND_INLINE;
          sr_wr.sg_list = &sr_sg_lst;
          sr_wr.num_sge = 1;
          sr_wr.next = NULL;
          sr_wr.wr.rdma.rkey = req->comminfo->eager_peer_free_info[req->peer].r_key;
          
          /* TODO: 64 Bit */ 
          sr_wr.wr.rdma.remote_addr = req->comminfo->eager_peer_free_info[req->peer].addr+i*sizeof(char); 
          
          /* post EAGER request */
          res = ibv_post_send(req->comminfo->qp_arr[req->peer], &sr_wr, &bad_swr);
          if(res != 0) OF_Abort(res, "ibv_post_send(EAGER_REPLY)");
        } else {
          /* mark request as done but not acknowledged yet */
          req->comminfo->eager[req->peer][i].tag = OF_NACKLDG_TAG;
        }
        /* mark receive as done */
        req->status = RECV_DONE;
#ifdef DEBUG_STATE        
        printf("[LibOF - %i] req %p (tag: %i) from RECV_WAITING_EAGER to RECV_DONE\n", req->comminfo->rank, req, req->tag);
#endif
        
        break; // this is important to not to invalidate more eager messages in this loop !!!
      }
    }
  }

  if(req->status == EAGER_SEND_INIT) {
    int sendentry, rementry;
    int dst = req->peer;
    void *buf = req->buf;

    /* find a new empty sendentry in the comminfo->eager_send array which is
     * pre-registered to send EAGER messages from */
    for(sendentry=0; sendentry<IB_EAGER_SIZE; sendentry++) {
      if(OF_UNUSED_TAG == req->comminfo->eager_send[sendentry].tag) break;
    }
    if(IB_EAGER_SIZE > sendentry) {
      /* find a new empty RDMA recv-buffer in the comminfo->eager_peer_free array */
      for(rementry=0; rementry<IB_EAGER_SIZE; rementry++) {
        if(0 == req->comminfo->eager_peer_free[dst][rementry]) break;
      }
      if(IB_EAGER_SIZE > rementry) {
        /* mark as used */
        req->comminfo->eager_peer_free[dst][rementry] = 1;

        req->sendel = sendentry;
        /* fill sendentry */
        req->comminfo->eager_send[sendentry].size = req->size;
        req->comminfo->eager_send[sendentry].tag = req->tag;
        /* copy at the end of buffer */
        void *dataptr = (void*)(req->comminfo->eager_send[sendentry].buf+IB_EAGER_LIMIT-req->size);
        memcpy(dataptr, buf, req->size); 
        /* see if we have something to piggyback for this destination */
        req->comminfo->eager_send[sendentry].piggyack[0]=255;
        req->comminfo->eager_send[sendentry].piggyack[1]=255;
        int idx = 0;
        for(i=0; (i<IB_EAGER_SIZE) && (idx < 2); i++) 
          if(req->comminfo->eager[dst][i].tag == OF_NACKLDG_TAG) {
            req->comminfo->eager[dst][i].tag = OF_UNUSED_TAG;
            req->comminfo->eager_send[sendentry].piggyack[idx]=i;
            idx++;
            break;
          }
        //printf("piggybacked %hhi %hhi\n", req->comminfo->eager_send[sendentry].piggyack[0], req->comminfo->eager_send[sendentry].piggyack[1]);

        struct ibv_sge sr_sg_lst; /* the IB SG list */
        struct ibv_send_wr sr_wr, *bad_swr; /* the IB SR descr. */

        /* prepare eager send request */
        sr_sg_lst.addr = (uint64_t)dataptr; 
        sr_sg_lst.length = sizeof(OF_Eager_message) - IB_EAGER_LIMIT + req->size;
        sr_sg_lst.lkey = req->comminfo->eager_send_mr->lkey; 
        sr_wr.wr_id = (uint64_t)req; /* to catch the signaled WR */
#if HAVE_PROGRESS_THREAD
        sr_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        sr_wr.imm_data = OF_IMM_NULL;
#else
        sr_wr.opcode = IBV_WR_RDMA_WRITE;
#endif
        if(req->comminfo->max_inline_data[dst] > sr_sg_lst.length)
          sr_wr.send_flags = (ibv_send_flags)(IBV_SEND_INLINE | IBV_SEND_SIGNALED); 
        else
          sr_wr.send_flags = IBV_SEND_SIGNALED;
        sr_wr.sg_list = &sr_sg_lst;
        sr_wr.num_sge = 1;
        sr_wr.next = NULL;
        sr_wr.wr.rdma.rkey = req->comminfo->eager_info[dst].r_key;
        //printf("[%i] eager %i bytes, rementry %i, sendentry: %i \n", req->comminfo->rank, sr_sg_lst.length, rementry, sendentry);
        
        /* TODO: 64 Bit */ 
        sr_wr.wr.rdma.remote_addr = req->comminfo->eager_info[dst].addr+
                                    rementry*sizeof(OF_Eager_message) + IB_EAGER_LIMIT-req->size; 
        
        /* post EAGER request */
        res = ibv_post_send(req->comminfo->qp_arr[req->peer], &sr_wr, &bad_swr);
        if(res != 0) OF_Abort(res, "ibv_post_send(EAGER)");

        req->status = EAGER_SENDING_DATA;
#ifdef DEBUG_STATE        
        printf("[LibOF - %i] req %p (tag: %i) from EAGER_SEND_INIT to EAGER_SENDING_DATA\n", req->comminfo->rank, req, req->tag);
#endif
      } 
    }
  }

  if(req->status == RNDV_RECV_INIT) {
    int res; 
    int recvlistentry, sendentry, rem_rtr_index;
    void *buf = req->buf;
    int src = req->peer;
    struct ibv_sge sr_sg_lst; /* the IB SG list */
    struct ibv_recv_wr rr_wr, *bad_rwr; /* the IB RR descr. */
    struct ibv_send_wr sr_wr, *bad_swr; /* the IB SR descr. */
    struct ibv_sge rr_sg_lst;

    /* register memory region for recv */
    res = OF_Register_mem(buf, req->size, &req->mr); 
    
    // TO CHECK
    /////////////////////////////////////////////////////////////////////////////////////////
    /* newer fancier reqlist approach */
    //pthread_mutex_lock(&OF_Gopen_recvs_lock);
    for(recvlistentry=0; recvlistentry<IB_RTR_SIZE; recvlistentry++) {
      if(NULL == OF_Gopen_recvs[recvlistentry]) break;
    }
    if(recvlistentry < IB_RTR_SIZE) {
  /* stack implementation 
      if(OF_Gopen_recv_free.empty()) 
        OF_Abort(1, "*** global open recv list full - we should retry later but crash\n"); 
      else {
        recvlistentry = OF_Gopen_recv_free.top();
        OF_Gopen_recv_free.pop();
      }
  */
      
      /* fill selected recv list entry */
      OF_Gopen_recvs[recvlistentry] = req;
      //pthread_mutex_unlock(&OF_Gopen_recvs_lock);
      /////////////////////////////////////////////////////////////////////////////////////////
      
      //pthread_mutex_lock (&(req->comminfo->rtr_lock)); 
      /* find a new empty sendentry in the comminfo->send array which is
       * pre-registered to send RTR messages from */
      for(sendentry=0; sendentry<IB_RTR_SIZE; sendentry++) {
        if(OF_UNUSED_TAG == req->comminfo->rtr_send[sendentry].tag) break;
      }
      if(IB_RTR_SIZE > sendentry) {
        /* fill selected send entry */
        req->comminfo->rtr_send[sendentry].tag = req->tag;
        req->comminfo->rtr_send[sendentry].addr = (uint64_t)buf;
        req->comminfo->rtr_send[sendentry].r_key = req->mr->rkey;
        /* send the open_recvs number to the sender so that we can find this
         * request fast when we get the data */
        req->comminfo->rtr_send[sendentry].recv_req = recvlistentry;

        /* search free entry in the src peer's rtr list (which is locally
         * mirrored in rtr_peer_free */
        for(rem_rtr_index=0; rem_rtr_index<IB_RTR_SIZE; rem_rtr_index++) {
          if(0 == req->comminfo->rtr_peer_free[src][rem_rtr_index]) break;
        }
        if(IB_RTR_SIZE == rem_rtr_index) OF_Abort(1, "*** remote rtr list full - we should retry later but crash");
        req->comminfo->rtr_peer_free[src][rem_rtr_index] = 1;
        //pthread_mutex_unlock (&(req->comminfo->rtr_lock));

        //pthread_mutex_lock (&OF_HCA_Info.lock);
        res = ibv_post_recv(req->comminfo->qp_arr[src], &OF_HCA_Info.dummy_rr, &OF_HCA_Info.dummy_bad_rr);
        //pthread_mutex_unlock (&OF_HCA_Info.lock);
        if(res != 0) OF_Abort(res, "ibv_post_recv(Irecv)");
        
        /* prepare RTR send request */
        sr_sg_lst.addr = (uint64_t)(&req->comminfo->rtr_send[sendentry]);
        sr_sg_lst.length = sizeof(OF_RTR_message);
        sr_sg_lst.lkey = req->comminfo->rtr_send_mr->lkey; 
        sr_wr.wr_id = (uint64_t)req;  /* to identify the request when we poll the CQ */
#ifdef USE_RDMA
#if HAVE_PROGRESS_THREAD
        sr_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        sr_wr.imm_data = OF_IMM_NULL;
#else
        sr_wr.opcode = IBV_WR_RDMA_WRITE;
#endif
#else
        sr_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        sr_wr.imm_data = rem_rtr_index + (0x1<<10);
#endif
        sr_wr.send_flags = IBV_SEND_SIGNALED;
        sr_wr.sg_list = &sr_sg_lst;
        sr_wr.num_sge = 1;
        sr_wr.next = NULL;
        sr_wr.wr.rdma.rkey = req->comminfo->rtr_info[src].r_key;
        
        /* TODO: 64 Bit */ 
        sr_wr.wr.rdma.remote_addr = req->comminfo->rtr_info[src].addr+
                                    rem_rtr_index*sizeof(OF_RTR_message); 
        
        /* post RTR request */
        res = ibv_post_send(req->comminfo->qp_arr[req->peer], &sr_wr, &bad_swr);
        if(res != 0) OF_Abort(res, "ibv_post_send()");

        /* remember index in sendlist to free it fast after sending */
        req->sendel = sendentry;
        req->rtr_peer_free_elem = rem_rtr_index;
        
        req->status = RECV_SENDING_RTR;
#ifdef DEBUG_STATE        
        printf("[LibOF - %i] req %p (tag: %i) from RNDV_RECV_INIT to RECV_SENDING_RTR\n", req->comminfo->rank, req, req->tag);
#endif
      }
    } 
  }

  /************************************** SEND QUEUE handling ************************************/
  /* poll until there are no CQ events in the queue anymore - in ofed,
   * it happens that there seems to be only one notification for two
   * simultaneous events - would deadlock if we wouldn't empty the queue
   * */
  int cqres = ibv_poll_cq(req->comminfo->scq_arr[req->peer], 1, &wc );
  //OF_DEBUG(10, "polled CQ res=%i\n", res);
  
  if(cqres > 0) {
#ifdef HAVE_PROGRESS_THREAD
    retval = OF_THREAD_POLL; /* poll all requests again inthe threaded case because we might have received an RDMA message for *another* request */
#endif
    if(wc.status != IBV_WC_SUCCESS) { 
      if(wc.status == IBV_WC_RETRY_EXC_ERR) printf("IBV_WC_RETRY_EXC_ERR\n");
      printf("wr id: %lu\n", wc.wr_id);
      OF_Abort(wc.status, "work completion status\n");
    }
    
    /* id != 0 implies send event completion */ 
    if(wc.wr_id != 0) {
      if(wc.wr_id != 1) {
        tmpreq = (OF_Req*)wc.wr_id;
        //printf("[%i] got request %p from SCQ\n", req->comminfo->rank, tmpreq);
        if(tmpreq->status == SEND_SENDING_DATA) {
#ifdef DEBUG_STATE        
          printf("[LibOF - %i] req %p (tag: %i) from SEND_SENDING_DATA to SEND_DONE\n", tmpreq->comminfo->rank, tmpreq, tmpreq->tag);
#endif
          /* we sent the message and are ready  */
          tmpreq->status = SEND_DONE;
#ifdef HAVE_PROGRESS_THREAD
          /* we need to wake it up because we don't know if we finished
           * the request we've been called with - we could deadlock .. */
          OF_Wakeup();
#endif
        } else if (tmpreq->status == RECV_SENDING_RTR) {
          /* set rtr sendlist element to free */
          tmpreq->status = RECV_SENT_RTR;
#ifdef DEBUG_STATE        
          printf("[LibOF - %i] req %p (tag: %i) from RECV_SENDING_RTR to RECV_SENT_RTR\n", tmpreq->comminfo->rank, tmpreq, tmpreq->tag);
#endif
        } else if (tmpreq->status == RECV_RECVD_DATA) {
          /* free the RTR send list element */
          tmpreq->comminfo->rtr_send[tmpreq->sendel].tag = OF_UNUSED_TAG;
          tmpreq->comminfo->rtr_peer_free[tmpreq->peer][tmpreq->rtr_peer_free_elem] = 0;
          
#ifdef DEBUG_STATE        
          printf("[LibOF - %i] req %p (tag: %i) from RECV_RECVD_DATA to RECV_DONE\n", tmpreq->comminfo->rank, tmpreq, tmpreq->tag);
#endif
          tmpreq->status = RECV_DONE; /* signal the main thread to free it! */
#ifdef HAVE_PROGRESS_THREAD
          /* we need to wake it up because we don't know if we finished
           * the request we've been called with - we could deadlock .. */
          OF_Wakeup();
#endif
        } else if (tmpreq->status == EAGER_SENDING_DATA) {
          tmpreq->comminfo->eager_send[tmpreq->sendel].tag = OF_UNUSED_TAG;

#ifdef DEBUG_STATE        
          printf("[LibOF - %i] req %p (tag: %i) from EAGER_SENDING_DATA to SEND_DONE\n", tmpreq->comminfo->rank, tmpreq, tmpreq->tag);
#endif
          tmpreq->status = SEND_DONE;
#ifdef HAVE_PROGRESS_THREAD
          /* we need to wake it up because we don't know if we finished
           * the request we've been called with - we could deadlock .. */
          OF_Wakeup();
#endif
        } else {
          printf("[%i] req %p unexpected status (%i) for send to %i (tag: %i) after poll sr_cq \n", 
          tmpreq->comminfo->rank, tmpreq, tmpreq->status, tmpreq->peer, tmpreq->tag);
        }
      }
    } else {
  /************************************** RECEIVE QUEUE handling ************************************/
//printf("got imm data: %i\n", wc.imm_data);
      if(wc.imm_data < (0x1<<10)) { /* TODO: make this a define */
        /* we received the data of rendezvous protocol */
        tmpreq = OF_Gopen_recvs[wc.imm_data];
        //printf("[%i] got request %p from RCQ\n", req->comminfo->rank, tmpreq);
        OF_Gopen_recvs[wc.imm_data] = NULL;
        //OF_Gopen_recv_free.push(wc.imm_data);
        if (tmpreq->status == RECV_SENDING_RTR) {
            tmpreq->status = RECV_RECVD_DATA;
#ifdef DEBUG_STATE
          printf("[LibOF - %i] req %p (tag: %i) from RECV_SENDING_RTR to RECV_RECVD_DATA\n", tmpreq->comminfo->rank, tmpreq, tmpreq->tag);
#endif
        } else if (tmpreq->status == RECV_SENT_RTR) {
          /* free the RTR send list element - double code with the other
           * RECV_DONE but with another handle :-/ */
          tmpreq->comminfo->rtr_send[tmpreq->sendel].tag = OF_UNUSED_TAG;
          tmpreq->comminfo->rtr_peer_free[tmpreq->peer][tmpreq->rtr_peer_free_elem] = 0;

#ifdef DEBUG_STATE
          printf("[LibOF - %i] req %p (tag: %i) from RECV_SENT_RTR to RECV_DONE\n", tmpreq->comminfo->rank, tmpreq, tmpreq->tag);
#endif
          tmpreq->status = RECV_DONE; /* signal the main thread to free it! */
#ifdef HAVE_PROGRESS_THREAD
          /* we need to wake it up because we don't know if we finished
           * the request we've been called with - we could deadlock .. */
          OF_Wakeup();
#endif
        } else {
          printf("[%i] req %p (tag: %i) unexpected status (%i) after poll rr_cq \n", 
                 tmpreq->comminfo->rank, tmpreq, tmpreq->tag, tmpreq->status);
        }
      } else if(wc.imm_data < (0x1<<20)) { /* TODO: make this a define */
        /* we received RTR message */
#ifdef USE_RDMA        
        printf("got imm_data %i \n", wc.imm_data);
#else
        /* re-post RR */
        res = ibv_post_recv(req->comminfo->qp_arr[req->peer], &OF_HCA_Info.dummy_rr, &OF_HCA_Info.dummy_bad_rr);
        if(res != 0) OF_Abort(res, "ibv_post_recv(Comm_init)");
        
        int index = wc.imm_data - (0x1<<10);
        int tag = req->comminfo->rtr[req->peer][index].tag;
        /* find the request for the tag */
        const std::multimap<int, OF_Request>::iterator iter=req->comminfo->tag_map[req->peer]->find(tag);
        if(iter == req->comminfo->tag_map[req->peer]->end()) {
          printf("tag %i not found in tagmap!\n", tag);
          OF_Abort(0, "RTR tagmatch (tag not found)");
        }
        const OF_Request send_req = iter->second; /* select iter element */
        req->comminfo->tag_map[req->peer]->erase(iter); /* erase tag */
        //printf("[%i] recvd RTR in index %i with tag %i\n", req->comminfo->rank, index, tag);
        
        send_req->sr_desc.wr.rdma.rkey = (uint32_t)send_req->comminfo->rtr[send_req->peer][index].r_key; 
        send_req->sr_desc.wr.rdma.remote_addr = send_req->comminfo->rtr[send_req->peer][index].addr; /* TODO: 64 Bit */ 
        /* send the 'free' rtr element back */
        send_req->sr_desc.imm_data = (uint32_t)send_req->comminfo->rtr[send_req->peer][index].recv_req; 
        send_req->comminfo->rtr[send_req->peer][index].tag = OF_UNUSED_TAG;

        struct ibv_send_wr *bad_wr;
        res = ibv_post_send(send_req->comminfo->qp_arr[send_req->peer], &send_req->sr_desc, &bad_wr);
        if(res != 0) OF_Abort(res, "ibv_post_send()");
            
        send_req->status = SEND_SENDING_DATA;
#ifdef DEBUG_STATE        
        printf("[LibOF - %i] req %p (tag: %i) from SEND_WAITING_RTR to SEND_SENDING_DATA\n", send_req->comminfo->rank, send_req, send_req->tag);
#endif
#endif
      } else if (wc.imm_data != OF_IMM_NULL) {
        /* we received an eager send */
        int index = wc.imm_data - (0x1<<20);
        int tag = req->comminfo->eager_send[index].tag;
      } else {
        OF_DEBUG(10, "got empty WC thing on req %p status %i ...\n", req, req->status);
        int res = ibv_post_recv(req->comminfo->qp_arr[req->peer], &OF_HCA_Info.dummy_rr, &OF_HCA_Info.dummy_bad_rr);
        if(res != 0) OF_Abort(res, "ibv_post_recv(Comm_init)");
      }
    }
  } else if(cqres < 0) printf("error on poll_cq()\n");
  
  if((req->status == SEND_DONE) || (req->status == RECV_DONE)) {
    OF_DEBUG(10, "freeing request %p (tag: %i)\n", req, req->tag);
    freerequest(request);
    return OF_OK;
  }

  return retval;
}

int OF_Wait(OF_Request *request) {
  
  while(OF_Test(request) != OF_OK) {};

  return OF_OK;
}

int OF_Testall(int count, OF_Request *requests, int *flag) {
  int i, res;

  *flag = 1;
  for(i=0; i<count; i++) {
    if(requests[i] == NULL) continue;
#ifdef HAVE_PROGRESS_THREAD
    *flag = 0;
#else
    res = OF_Test(&requests[i]);
    if(res != OF_OK) *flag = 0;
#endif
  }

  OF_DEBUG(10, "leaving OF_Testall: count %i flag %i\n", count, *flag);

  return OF_OK;
}

#ifdef HAVE_PROGRESS_THREAD
int OF_Waitany(int count, OF_Request *requests, int *index) {
  OF_DEBUG(10, "entering OF_Waitany with %i requests\n", count);

  /* assemble list of FDs from the requests */
  std::vector<struct ibv_comp_channel*> channels;
  //channels.push_back(OF_Gpipe[0]);
  for(int i=0; i<count; i++) {
  int peer = requests[i]->peer;
    channels.push_back(requests[i]->comminfo->compchan[peer]);
  }
  
  /* erase double elements */
  std::sort(channels.begin(), channels.end());
  channels.erase(unique(channels.begin(), channels.end()), channels.end());
  
  int numfds = channels.size()+1;

  struct pollfd *pollfds = (struct pollfd *)malloc(sizeof(struct pollfd)*numfds);
  pollfds[0].fd = OF_Gpipe[0];
  pollfds[0].events = POLLIN;
  for(int i=1; i<numfds; i++) {
    //printf("fd: %i\n", fds[i]);
    pollfds[i].fd = channels[i-1]->fd;
    pollfds[i].events = POLLIN;
  }

  while (true) {
    /* test all requests to see if any of them is done */
    int continue_testing=1;
    while(continue_testing) {
      continue_testing = 0;
      for(int i=0; i<count; i++) {
        int ret = OF_Test(&requests[i]);
        if(ret == OF_OK) {
          OF_DEBUG(10, "OF_Waitany - request index %i finishes\n", i);
          free(pollfds);
          *index = i;
          return OF_OK;
        } else if((ret == OF_THREAD_POLL)) {
          continue_testing = 1;
    } } }

    /* wait on them */
    OF_DEBUG(10, "waiting on %i fds (%i requests)\n", numfds, count);
    /* TODO: timeout should be -1 but I'm too lame to find the last threading bug which looks like events are lost in the ofed layer, so let's just progress every second :-( */
    //if(0 == poll(pollfds, numfds, 1000)) printf("poll timeout\n");
    poll(pollfds, numfds, -1);

    for(int i=1; i<numfds; i++) {
      if(pollfds[i].revents & POLLIN) {
        struct ibv_cq *cq;
        void *cq_context;
        OF_DEBUG(10, "finished fd # %i (%i)\n", i, pollfds[i].fd);
            
        if(ibv_get_cq_event(channels[i-1], &cq, &cq_context)) OF_Abort(0, "ibv_get_cq_event()");
        ibv_ack_cq_events(cq, 1);
        //usleep(10);
        if(ibv_req_notify_cq(cq, 0)) OF_Abort(0, "ibv_req_notify_cq()"); 
      }
    }

    if(pollfds[0].revents & POLLIN) {
      OF_DEBUG(10, "was woken up\n");
      char buf = fgetc(OF_Grfd);
      //read(OF_Gpipe[0], &buf, 1);

      free(pollfds);
      *index = -1;
      return OF_OK;
    }
  }

  return OF_OK;
}

/* wakes the thread up if it blocks in OF_Waitany */
void OF_Wakeup() {
  //printf("waking thread up ...");
  fputc(0, OF_Gwfd);
  fflush(OF_Gwfd);
  //char buf;
  //write(OF_Gpipe[1], &buf, BUFSIZ+1);
  //sched_yield();
}
#endif

int OF_Waitall(int count, OF_Request *requests) {
  int i, res, done;

  do  {
    done = OF_OK;
    for(i=0; i<count; i++) {
      if(requests[i] == NULL) continue;
      res = OF_Test(&requests[i]); 

      /* we have at least one unfinished request ... */
      if(res == OF_CONTINUE) {
        done = OF_CONTINUE;
      } else { /* in case of error */
        if(res != OF_OK) {
          printf("Error in OF_Waitall()\n");
          return res;
        }
      }
    }
  } while(done == OF_CONTINUE);

  return done;
}

/* try to get all requests out of the status SEND_WAITING_RTR,
 * must be non-blocking, has a timeout for that ... 
 * BE VERY VERY CAREFUL: when all RTR buffers run full, this will be
 * really really slow because it will wait for the timeout all the time
 * ... */
void OF_Startall(int count, OF_Request *requests, unsigned long timeout) {
  char done;
  int i;
#ifdef HAVE_PROGRESS_THREAD
  OF_Abort(0, "calles OF_Startall with progress thread !\n");
#else
  do {
    done = 1;
    for(i=0; i<count; i++) {
      if(requests[i] != NULL)
        if(requests[i]->status == SEND_WAITING_RTR) {
          done = 0;
          OF_Test(&requests[i]);
        }
    }
  } while((!done) && (timeout-- > 0));
#endif
}

static __inline__ int OF_Memlist_compare_entries(OF_Memlistel *a, OF_Memlistel *b,void *param) {

  /* two memory regions are defined as equal if they have some common
   * memory - more is not possible, because we have to ensure
   * reflexibility (a=b includes b=a) */
	
	if( (a->buf == b->buf) && (a->size == b->size) ) {
    return  0;
  }
	if ( (a->buf < b->buf)) {	
    return -1;
	}
	return +1;
}

static __inline__ void OF_Memlist_delete_key(OF_Memlistel *k) {
  /* do nothing because the key and the data element are identical :-) 
   * both (the single one :) is freed in OF_Memlist_memlist_delete() */
}

static __inline__ void OF_Memlist_memlist_delete(OF_Memlistel *entry) {
  /* free entry and deregister MR here ... */
}
