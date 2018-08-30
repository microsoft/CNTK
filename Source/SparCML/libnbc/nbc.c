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

/* only used in this file */
static inline int NBC_Start_round(NBC_Handle *handle);

#ifdef HAVE_PROGRESS_THREAD
#include <pthread.h>
#include <list>
#include <vector>

static pthread_t GNBC_Pthread; // pthread is short for Progressthread ;)
std::list<NBC_Handle*> GNBC_Pthread_handles;
static pthread_mutex_t GNBC_Pthread_handles_lock = PTHREAD_MUTEX_INITIALIZER; 

/* dummy buffer to send/recv from/to */
static int GMPI_dumbuf[2];

void *NBC_Pthread_func( void *ptr ) {
#ifdef HAVE_MPI
  MPI_Request req=MPI_REQUEST_NULL;
#endif

  while(true) {

#ifdef HAVE_OFED
    std::vector<void*> requests;
    /* we need this vector because we need to copy the finished requests back */
    std::vector<int> requests_locations;
    /* stores the handle for every request */
    std::vector<NBC_Handle*> requests_handles;
#endif
#ifdef HAVE_MPI
    std::vector<MPI_Request> requests(1);
    /* we need this vector because we need to copy the finished requests back */
    std::vector<int> requests_locations(1);
    /* stores the handle for every request */
    std::vector<NBC_Handle*> requests_handles(1);
#endif

    //MPI_Grequest_start(Grequest_query_fn, Grequest_free_fn, Grequest_cancel_fn, NULL, (MPI_Request*)&requests.front()); 
    //int flag;
    //MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
#ifdef HAVE_MPI
    if(req==MPI_REQUEST_NULL) MPI_Irecv(&GMPI_dumbuf[0], 1, MPI_INT, 0, 0, MPI_COMM_SELF, &req);
    requests[0]=req;
#endif
    //pthread_mutex_lock(&GMPI_Request_lock);
    //GMPI_Request = &requests.front();
    //GMPI_valid = 1;
    //pthread_mutex_unlock(&GMPI_Request_lock);

    /* re-compile list of requests */
    pthread_mutex_lock(&GNBC_Pthread_handles_lock);

    for(std::list<NBC_Handle*>::iterator iter=GNBC_Pthread_handles.begin(); iter!=GNBC_Pthread_handles.end(); ) {
      //pthread_mutex_lock(&(*iter)->lock);
      /* erase handle from list if it's done */
      /*if((*iter)->schedule == NULL) {
        //pthread_mutex_unlock(&(*iter)->lock);
        iter = GNBC_Pthread_handles.erase(iter);
        continue;
      }*/
      /* if the handle is not done but there are no requests, it must be
       * a new one - start first round */
      if((*iter)->req_count == 0) {
        NBC_Start_round(*iter);
      }
      for(int i=0; i<(*iter)->req_count; i++) {
#ifdef HAVE_OFED
        if((*iter)->req_array[i] != NULL) 
#endif
#ifdef HAVE_MPI
        if((*iter)->req_array[i] != MPI_REQUEST_NULL) 
#endif 
        {
          requests.push_back((*iter)->req_array[i]);
          requests_locations.push_back(i);
          requests_handles.push_back(*iter);
        }
      }
      //pthread_mutex_unlock(&(*iter)->lock);
      ++iter;
    }
    //printf("have %i open handles\n", GNBC_Pthread_handles.size());

    pthread_mutex_unlock(&GNBC_Pthread_handles_lock);

    int retidx = 0;
    NBC_DEBUG(10, "waiting for %i elements\n", (int)requests.size());
#ifdef HAVE_OFED
    int res = OF_Waitany(requests.size(), (OF_Request*)&requests.front(), &retidx);
    //if((int)requests_handles.size()>0) printf("%i\n",requests_handles[0]->tag);
#endif
#ifdef HAVE_MPI
    int res = MPI_Waitany(requests.size(), &requests.front(), &retidx, MPI_STATUS_IGNORE);
#endif
    if(res != MPI_SUCCESS) { printf("Error %i in MPI_Waitany()\n", res); }
    //printf("request %i finished\n", retidx);
    /*pthread_mutex_lock(&GMPI_Request_lock);
    GMPI_valid = 0;
    pthread_mutex_unlock(&GMPI_Request_lock);*/

#ifdef HAVE_MPI
    if(0 != retidx) { // 0 is the fake request ...
      //pthread_mutex_lock(&requests_handles[retidx]->lock);
      /* mark request as finished */
      requests_handles[retidx]->req_array[requests_locations[retidx]] = MPI_REQUEST_NULL;
      /* progress request (finished?) */
      NBC_Progress(requests_handles[retidx]);
      //pthread_mutex_unlock(&requests_handles[retidx]->lock);
    } else {
      req = MPI_REQUEST_NULL;
    }
#endif
#ifdef HAVE_OFED
    if(retidx >= 0) {
      /* mark request as finished */
      requests_handles[retidx]->req_array[requests_locations[retidx]] = NULL;
      /* progress request (finished?) */
      NBC_Progress(requests_handles[retidx]);
    }
#endif
  }
}

#endif

//#define NBC_TIMING

#ifdef NBC_TIMING
static double Isend_time=0, Irecv_time=0, Wait_time=0, Test_time=0;
void NBC_Reset_times() {
  Isend_time=Irecv_time=Wait_time=Test_time=0;
}
void NBC_Print_times(double div) {
  printf("*** NBC_TIMES: Isend: %lf, Irecv: %lf, Wait: %lf, Test: %lf\n", Isend_time*1e6/div, Irecv_time*1e6/div, Wait_time*1e6/div, Test_time*1e6/div);
}
#endif

/* is NBC globally initialized */
static char GNBC_Initialized=0;

/* the keyval (global) */
static int gkeyval=MPI_KEYVAL_INVALID; 

static int NBC_Key_copy(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag) {
  /* delete the attribute in the new comm  - it will be created at the
   * first usage */
  *flag = 0;

  return MPI_SUCCESS;
}

static int NBC_Key_delete(MPI_Comm comm, int keyval, void *attribute_val, void *extra_state) {
  NBC_Comminfo *comminfo;

  if(keyval == gkeyval) {
    comminfo=(NBC_Comminfo*)attribute_val;
    free((void*)comminfo);
  } else {
    printf("Got wrong keyval!(%i)\n", keyval); 
  }

  return MPI_SUCCESS;
}

/* allocates a new schedule array */
int NBC_Sched_create(NBC_Schedule* schedule) {
  
  *schedule=malloc(2*sizeof(int));
  if(*schedule == NULL) { return NBC_OOR; }
  *(int*)*schedule=2*sizeof(int);
  *(((int*)*schedule)+1)=0;

  return NBC_OK;
}

/* this function puts a send into the schedule */
int NBC_Sched_send(void* buf, char tmpbuf, int count, MPI_Datatype datatype, int dest, NBC_Schedule *schedule) {
  int size;
  NBC_Args_send* send_args;
  
  /* get size of actual schedule */
  NBC_GET_SIZE(*schedule, size);
  /*printf("schedule is %i bytes\n", size);*/
  *schedule = (NBC_Schedule)realloc(*schedule, size+sizeof(NBC_Args_send)+sizeof(NBC_Fn_type));
  if(*schedule == NULL) { printf("Error in realloc()\n"); return NBC_OOR; }
  
  /* adjust the function type */
  *(NBC_Fn_type*)((char*)*schedule+size)=SEND;
  
  /* store the passed arguments */
  send_args = (NBC_Args_send*)((char*)*schedule+size+sizeof(NBC_Fn_type));
  send_args->buf=buf;
  send_args->tmpbuf=tmpbuf;
  send_args->count=count;
  send_args->datatype=datatype;
  send_args->dest=dest;

  /* increase number of elements in schedule */
  NBC_INC_NUM_ROUND(*schedule);
  NBC_DEBUG(10, "adding send - ends at byte %i\n", (int)(size+sizeof(NBC_Args_send)+sizeof(NBC_Fn_type)));

  /* increase size of schedule */
  NBC_INC_SIZE(*schedule, sizeof(NBC_Args_send)+sizeof(NBC_Fn_type));

  return NBC_OK;
}

/* this function puts a receive into the schedule */
int NBC_Sched_recv(void* buf, char tmpbuf, int count, MPI_Datatype datatype, int source, NBC_Schedule *schedule) {
  int size;
  NBC_Args_recv* recv_args;
  
  /* get size of actual schedule */
  NBC_GET_SIZE(*schedule, size);
  /*printf("schedule is %i bytes\n", size);*/
  *schedule = (NBC_Schedule)realloc(*schedule, size+sizeof(NBC_Args_recv)+sizeof(NBC_Fn_type));
  if(*schedule == NULL) { printf("Error in realloc()\n"); return NBC_OOR; }
  
  /* adjust the function type */
  *(NBC_Fn_type*)((char*)*schedule+size)=RECV;

  /* store the passed arguments */
  recv_args=(NBC_Args_recv*)((char*)*schedule+size+sizeof(NBC_Fn_type));
  recv_args->buf=buf;
  recv_args->tmpbuf=tmpbuf;
  recv_args->count=count;
  recv_args->datatype=datatype;
  recv_args->source=source;

  /* increase number of elements in schedule */
  NBC_INC_NUM_ROUND(*schedule);
  NBC_DEBUG(10, "adding receive - ends at byte %i\n", (int)(size+sizeof(NBC_Args_recv)+sizeof(NBC_Fn_type)));

  /* increase size of schedule */
  NBC_INC_SIZE(*schedule, sizeof(NBC_Args_recv)+sizeof(NBC_Fn_type));

  return NBC_OK;
}

/* this function puts an operation into the schedule */
int NBC_Sched_op2(void *buf3, char tmpbuf3, void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count, MPI_Datatype datatype, MPI_Op op, NBC_Schedule *schedule, int forceDense) {
  int size;
  NBC_Args_op* op_args;
  
  /* get size of actual schedule */
  NBC_GET_SIZE(*schedule, size);
  /*printf("schedule is %i bytes\n", size);*/
  *schedule = (NBC_Schedule)realloc(*schedule, size+sizeof(NBC_Args_op)+sizeof(NBC_Fn_type));
  if(*schedule == NULL) { printf("Error in realloc()\n"); return NBC_OOR; }
  
  /* adjust the function type */
  *(NBC_Fn_type*)((char*)*schedule+size)=OP;

  /* store the passed arguments */
  op_args=(NBC_Args_op*)((char*)*schedule+size+sizeof(NBC_Fn_type));
  op_args->buf1=buf1;
  op_args->buf2=buf2;
  op_args->buf3=buf3;
  op_args->tmpbuf1=tmpbuf1;
  op_args->tmpbuf2=tmpbuf2;
  op_args->tmpbuf3=tmpbuf3;
  op_args->count=count;
  op_args->op=op;
  op_args->datatype=datatype;
  op_args->forceDense=forceDense;

  /* increase number of elements in schedule */
  NBC_INC_NUM_ROUND(*schedule);
  NBC_DEBUG(10, "adding op - ends at byte %i\n", (int)(size+sizeof(NBC_Args_op)+sizeof(NBC_Fn_type)));

  /* increase size of schedule */
  NBC_INC_SIZE(*schedule, sizeof(NBC_Args_op)+sizeof(NBC_Fn_type));
  
  return NBC_OK;
}

int NBC_Sched_op(void *buf3, char tmpbuf3, void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count, MPI_Datatype datatype, MPI_Op op, NBC_Schedule *schedule) {
  return NBC_Sched_op2(buf3, tmpbuf3, buf1, tmpbuf1, buf2, tmpbuf2, count, datatype, op, schedule, 0);
}

/* this function puts a copy into the schedule */
int NBC_Sched_copy(void *src, char tmpsrc, int srccount, MPI_Datatype srctype, void *tgt, char tmptgt, int tgtcount, MPI_Datatype tgttype, NBC_Schedule *schedule) {
  int size;
  NBC_Args_copy* copy_args;
  
  /* get size of actual schedule */
  NBC_GET_SIZE(*schedule, size);
  /*printf("schedule is %i bytes\n", size);*/
  *schedule = (NBC_Schedule)realloc(*schedule, size+sizeof(NBC_Args_copy)+sizeof(NBC_Fn_type));
  if(*schedule == NULL) { printf("Error in realloc()\n"); return NBC_OOR; }
  
  /* adjust the function type */
  *(NBC_Fn_type*)((char*)*schedule+size)=COPY;
  
  /* store the passed arguments */
  copy_args = (NBC_Args_copy*)((char*)*schedule+size+sizeof(NBC_Fn_type));
  copy_args->src=src;
  copy_args->tmpsrc=tmpsrc;
  copy_args->srccount=srccount;
  copy_args->srctype=srctype;
  copy_args->tgt=tgt;
  copy_args->tmptgt=tmptgt;
  copy_args->tgtcount=tgtcount;
  copy_args->tgttype=tgttype;

  /* increase number of elements in schedule */
  NBC_INC_NUM_ROUND(*schedule);
  NBC_DEBUG(10, "adding copy - ends at byte %i\n", (int)(size+sizeof(NBC_Args_copy)+sizeof(NBC_Fn_type)));

  /* increase size of schedule */
  NBC_INC_SIZE(*schedule, sizeof(NBC_Args_copy)+sizeof(NBC_Fn_type));

  return NBC_OK;
}

/* this function puts a unpack into the schedule */
int NBC_Sched_unpack(void *inbuf, char tmpinbuf, int count, MPI_Datatype datatype, void *outbuf, char tmpoutbuf, NBC_Schedule *schedule) {
  int size;
  NBC_Args_unpack* unpack_args;
  
  /* get size of actual schedule */
  NBC_GET_SIZE(*schedule, size);
  /*printf("schedule is %i bytes\n", size);*/
  *schedule = (NBC_Schedule)realloc(*schedule, size+sizeof(NBC_Args_unpack)+sizeof(NBC_Fn_type));
  if(*schedule == NULL) { printf("Error in realloc()\n"); return NBC_OOR; }
  
  /* adjust the function type */
  *(NBC_Fn_type*)((char*)*schedule+size)=UNPACK;
  
  /* store the passed arguments */
  unpack_args = (NBC_Args_unpack*)((char*)*schedule+size+sizeof(NBC_Fn_type));
  unpack_args->inbuf=inbuf;
  unpack_args->tmpinbuf=tmpinbuf;
  unpack_args->count=count;
  unpack_args->datatype=datatype;
  unpack_args->outbuf=outbuf;
  unpack_args->tmpoutbuf=tmpoutbuf;

  /* increase number of elements in schedule */
  NBC_INC_NUM_ROUND(*schedule);
  NBC_DEBUG(10, "adding unpack - ends at byte %i\n", (int)(size+sizeof(NBC_Args_unpack)+sizeof(NBC_Fn_type)));

  /* increase size of schedule */
  NBC_INC_SIZE(*schedule, sizeof(NBC_Args_unpack)+sizeof(NBC_Fn_type));

  return NBC_OK;
}

/* this function ends a round of a schedule */
int NBC_Sched_barrier(NBC_Schedule *schedule) {
  int size;
  
  /* get size of actual schedule */
  NBC_GET_SIZE(*schedule, size);
  /*printf("round terminated at %i bytes\n", size);*/
  *schedule = (NBC_Schedule)realloc(*schedule, size+sizeof(char)+sizeof(int));
  if(*schedule == NULL) { printf("Error in realloc()\n"); return NBC_OOR; }
  
  /* add the barrier char (1) because another round follows */
  *(char*)((char*)*schedule+size)=1;
  
  /* set round count elements = 0 for new round */
  *(int*)((char*)*schedule+size+sizeof(char))=0;
  NBC_DEBUG(10, "ending round at byte %i\n", (int)(size+sizeof(char)+sizeof(int)));
  
  /* increase size of schedule */
  NBC_INC_SIZE(*schedule, sizeof(char)+sizeof(int));

  return NBC_OK;
}

/* this function ends a schedule */
int NBC_Sched_commit(NBC_Schedule *schedule) {
  int size;
 
  /* get size of actual schedule */
  NBC_GET_SIZE(*schedule, size);
  /*printf("schedule terminated at %i bytes\n", size);*/
  *schedule = (NBC_Schedule)realloc(*schedule, size+sizeof(char));
  if(*schedule == NULL) { printf("Error in realloc()\n"); return NBC_OOR; }
 
  /* add the barrier char (0) because this is the last round */
  *(char*)((char*)*schedule+size)=0;
  NBC_DEBUG(10, "closing schedule %p at byte %i\n", *schedule, (int)(size+sizeof(char)));

  /* increase size of schedule */
  NBC_INC_SIZE(*schedule, sizeof(char));
 
  return NBC_OK;
}

/* finishes a request
 *
 * to be called *only* from the progress thread !!! */
static inline int NBC_Free(NBC_Handle* handle) {
  
#ifdef HAVE_PROGRESS_THREAD 
  pthread_mutex_lock(&GNBC_Pthread_handles_lock);
  GNBC_Pthread_handles.remove(handle);
  pthread_mutex_unlock(&GNBC_Pthread_handles_lock);

  pthread_mutex_lock(&handle->lock);
#endif

#ifdef HAVE_DCMF
  if(handle->dcmf_hndl != NULL) { free(handle->dcmf_hndl); handle->dcmf_hndl = NULL; }
#endif

#ifdef NBC_CACHE_SCHEDULE
  /* do not free schedule because it is in the cache */
  handle->schedule = NULL;
#else
  if(handle->schedule != NULL) {
    /* free schedule */
    free((void*)*(handle->schedule));
    free((void*)handle->schedule);
    handle->schedule = NULL;
  }
#endif

#ifdef HAVE_PROGRESS_THREAD 
  pthread_mutex_unlock(&handle->lock);
#endif

  /* if the nbc_I<collective> attached some data */
  /* problems with schedule cache here, see comment (TODO) in
   * nbc_internal.h */
  if(NULL != handle->tmpbuf) {
    free((void*)handle->tmpbuf);
    handle->tmpbuf = NULL;
  }

#ifdef HAVE_PROGRESS_THREAD 
  if(sem_post(&handle->semid) != 0) { perror("sem_post()"); }
#endif

  return NBC_OK;
}

/* progresses a request
 *
 * to be called *only* from the progress thread !!! */
int NBC_Progress(NBC_Handle *handle) {
  int flag, res, ret=NBC_CONTINUE;
  long size;
  char *delim;

#ifdef HAVE_OFED
#ifndef HAVE_PROGRESS_THREAD
  /* TODO: DIRTY HACK! - we need to progress the MPI library, because
   * Open MPI shows the following behavior in the sequence MPI_Gather,
   * NBC_Igather:
   * - sender in gather finishes but did not send message (probably
   *   buffered)
   * - NBC_IGather is called and waits for message, without ever
   *   progressing MPI
   * - Receiver waits for MPI_Gather message ... forever
   *
   *   workaround: progress MPI (call MPI_Iprobe() here ...
   * */
  {
    int flag;
    MPI_Status stat;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &stat);
  }
#endif
#endif

//printf("[%i] dcmf: %x, schedule: %x\n", r, handle->dcmf_hndl, handle->schedule);

#ifdef HAVE_DCMF
  if(handle->dcmf_hndl != NULL) { // if we have a DCMF handle - check for completion
    DCMF_Messager_advance();

    // the handle is only set to 0 if the barrier succeeded (callback is called)
    // must be >= 0 because the receive callback can occur *before* many_to_many is started ;-)
    if(handle->dcmf_hndl->type == DCMF_TYPE_MANY_TO_MANY && handle->dcmf_hndl->barr_done == 1) {
      dcmf_comminfo *comminfo = handle->dcmf_hndl->comminfo;

      DCMF_Manytomany(comminfo->proto, 		// Protocol registration
        &handle->dcmf_hndl->srequest, 	// Opaque memory to maintain internal message state. -- can be reused once the send completion cb is called!
        handle->dcmf_hndl->cb_m2m_done, // cb, // Callback to invoke when sender side Manytomany operation is complete. This callback is always called regardless of the amount of data sent by this rank.
        DCMF_MATCH_CONSISTENCY, 	// Required consistency model
        handle->dcmf_hndl->cid,    // Identifies the operation, between 0 and 31. This parameter is passed to the receive callback and can be used to distinguish several Manytomany operations. Depending on the registered protocol, the max number of connection ids may be a limited.
        1, 				// Is the rank index a vector or a single integer same for all destinations
        0,				// Index on all receivers this sender's data will be moved to.
        comminfo->ridx,	// Vector of rankindices with an element for each destination rank
        handle->dcmf_hndl->sbuf, 			// Source data buffer to send. <put semantics="" of="" buffer="" access="" here>="">
        handle->dcmf_hndl->slens, 	// List of message sizes to send to each individual rank in ranks (same order as in ranks)
        handle->dcmf_hndl->sdispls, 	// List of offsets to the start of sndbuf to specify where to start each individual message (same order as in ranks)
        comminfo->sndcounters,	// List of counters for internal use
        comminfo->neighbors,   // List of global ranks to specify the members or group involved in this Manytomany operation
        comminfo->permutation, 	// List of group ranks ordered in an arbitrary way to specify the order of send operations.
        comminfo->outdeg);			// number of ranks in the groups, number of elements in the above lists,

      // disable barrier again
      handle->dcmf_hndl->barr_done = 0;
      
int r;
MPI_Comm_rank(MPI_COMM_WORLD, &r);
printf("[%i] [%i] launched many-to-many call!\n", r, handle->dcmf_hndl->cid);
    }
    
    if(handle->dcmf_hndl->done == 2) {
        res = NBC_Free(handle);
        if((NBC_OK != res)) { printf("Error in NBC_Free() (%i)\n", res); ret=res; goto error; }
	      return NBC_OK;
    }
    else return NBC_CONTINUE;
  } else // do normal LibNBC stuff
#endif
  /* the handle is done if there is no schedule attached */
  if(handle->schedule != NULL) {

    if((handle->req_count > 0) && (handle->req_array != NULL)) {
      NBC_DEBUG(50, "NBC_Progress: testing for %i requests\n", handle->req_count);
#ifdef NBC_TIMING
      Test_time -= MPI_Wtime();
#endif
#ifdef HAVE_OMPI
      /*res = ompi_request_test_all(handle->req_count, handle->req_array, &flag, MPI_STATUSES_IGNORE);*/
      res = MPI_Testall(handle->req_count, handle->req_array, &flag, MPI_STATUSES_IGNORE);
      if(res != OMPI_SUCCESS) { printf("MPI Error in MPI_Testall() (%i)\n", res); ret=res; goto error; }
#endif
#ifdef HAVE_MPI
      res = MPI_Testall(handle->req_count, handle->req_array, &flag, MPI_STATUSES_IGNORE);
      if(res != MPI_SUCCESS) { printf("MPI Error in MPI_Testall() (%i)\n", res); ret=res; goto error; }
#endif
#ifdef HAVE_OFED
      res = OF_Testall(handle->req_count, (OF_Request*)handle->req_array, &flag);
      if(res != OF_OK) { printf("MPI Error in MPI_Testall() (%i)\n", res); ret=res; goto error; }
#endif
#ifdef NBC_TIMING
      Test_time += MPI_Wtime();
#endif
    } else {
      flag = 1; /* we had no open requests -> proceed to next round */
    }

    /* a round is finished */
    if(flag) {
      /* adjust delim to start of current round */
      NBC_DEBUG(5, "NBC_Progress: going in schedule %p to row-offset: %li\n", *handle->schedule, handle->row_offset);
      delim = (char*)*handle->schedule + handle->row_offset;
      NBC_DEBUG(10, "delim: %p\n", delim);
      NBC_GET_ROUND_SIZE(delim, size);
      NBC_DEBUG(10, "size: %li\n", size);
      /* adjust delim to end of current round -> delimiter */
      delim = delim + size;

      if(handle->req_array != NULL) {
        /* free request array */
        free((void*)handle->req_array);
        handle->req_array = NULL;
      }
      handle->req_count = 0;

      if(*delim == 0) {
        /* this was the last round - we're done */
        NBC_DEBUG(5, "NBC_Progress last round finished - we're done\n");
        
        res = NBC_Free(handle);
        if((NBC_OK != res)) { printf("Error in NBC_Free() (%i)\n", res); ret=res; goto error; }

        return NBC_OK;
      } else {
        NBC_DEBUG(5, "NBC_Progress round finished - goto next round\n");
        /* move delim to start of next round */
        delim = delim+1;
        /* initializing handle for new virgin round */
        handle->row_offset = (long)delim - (long)*handle->schedule;
        /* kick it off */
        res = NBC_Start_round(handle);
        if(NBC_OK != res) { printf("Error in NBC_Start_round() (%i)\n", res); ret=res; goto error; }
      }
    }
  } else {
    ret= NBC_OK;
  }

error:
  return ret;
}

size_t countBytes(void *s, int dim) {
  unsigned len = *(unsigned *)s;
  if(len  == dim) {
    return sizeof(unsigned) + len * sizeof(float);
  }
  return sizeof(unsigned) + (len * (sizeof(unsigned) + sizeof(float)));
}

static inline int NBC_Start_round(NBC_Handle *handle) {
  int *numptr; /* number of operations */
  int i, res, ret=NBC_OK;
  NBC_Fn_type *typeptr;
  NBC_Args_send *sendargs; 
  NBC_Args_recv *recvargs; 
  NBC_Args_op *opargs; 
  NBC_Args_copy *copyargs; 
  NBC_Args_unpack *unpackargs; 
  NBC_Schedule myschedule;
  void *buf1, *buf2, *buf3;

  /* get schedule address */
  myschedule = (NBC_Schedule*)((char*)*handle->schedule + handle->row_offset);

  numptr = (int*)myschedule;
  NBC_DEBUG(10, "start_round round at address %p : posting %i operations\n", myschedule, *numptr);

  /* typeptr is increased by sizeof(int) bytes to point to type */
  typeptr = (NBC_Fn_type*)(numptr+1);
  for (i=0; i<*numptr; i++) {
    /* go sizeof op-data forward */
    switch(*typeptr) {
      case SEND:

        NBC_DEBUG(5,"  SEND (offset %li) ", (long)typeptr-(long)myschedule);
        sendargs = (NBC_Args_send*)(typeptr+1);
        NBC_DEBUG(5,"*buf: %p, count: %i, type: %lu, dest: %i, tag: %i)\n", sendargs->buf, sendargs->count, (unsigned long)sendargs->datatype, sendargs->dest, handle->tag);
        typeptr = (NBC_Fn_type*)(((NBC_Args_send*)typeptr)+1);
        /* get an additional request */
        handle->req_count++;
        /* get buffer */
        if(sendargs->tmpbuf) 
          buf1=(char*)handle->tmpbuf+(long)sendargs->buf;
        else
          buf1=sendargs->buf;

        // HACK: Get correct count
        if (sendargs->count < 0) {
          sendargs->count = countBytes(buf1, -1*sendargs->count);
        }
#ifdef NBC_TIMING
    Isend_time -= MPI_Wtime();
#endif
#ifdef HAVE_OMPI
        handle->req_array = (MPI_Request*)realloc((void*)handle->req_array, (handle->req_count)*sizeof(MPI_Request));
        NBC_CHECK_NULL(handle->req_array);
        /*res = MCA_PML_CALL(isend_init(buf1, sendargs->count, sendargs->datatype, sendargs->dest, handle->tag, MCA_PML_BASE_SEND_STANDARD, handle->mycomm, handle->req_array+handle->req_count-1));
        printf("MPI_Isend(%lu, %i, %lu, %i, %i, %lu) (%i)\n", (unsigned long)buf1, sendargs->count, (unsigned long)sendargs->datatype, sendargs->dest, handle->tag, (unsigned long)handle->mycomm, res);*/
        res = MPI_Isend(buf1, sendargs->count, sendargs->datatype, sendargs->dest, handle->tag, handle->mycomm, handle->req_array+handle->req_count-1);
        if(OMPI_SUCCESS != res) { printf("Error in MPI_Isend(%lu, %i, %lu, %i, %i, %lu) (%i)\n", (unsigned long)buf1, sendargs->count, (unsigned long)sendargs->datatype, sendargs->dest, handle->tag, (unsigned long)handle->mycomm, res); ret=res; goto error; }
#endif
#ifdef HAVE_MPI
        handle->req_array = (MPI_Request*)realloc((void*)handle->req_array, (handle->req_count)*sizeof(MPI_Request));
        NBC_CHECK_NULL(handle->req_array);
        res = MPI_Isend(buf1, sendargs->count, sendargs->datatype, sendargs->dest, handle->tag, handle->mycomm, handle->req_array+handle->req_count-1);
        if(MPI_SUCCESS != res) { printf("Error in MPI_Isend(%lu, %i, %lu, %i, %i, %lu) (%i)\n", (unsigned long)buf1, sendargs->count, (unsigned long)sendargs->datatype, sendargs->dest, handle->tag, (unsigned long)handle->mycomm, res); ret=res; goto error; }
#endif
#ifdef HAVE_OFED
        handle->req_array = (void**)realloc((void**)handle->req_array, (handle->req_count)*sizeof(OF_Request));
        NBC_CHECK_NULL(handle->req_array);
        res = OF_Isend(buf1, sendargs->count, sendargs->datatype, sendargs->dest, handle->tag, handle->mycomm,  (OF_Request*)handle->req_array+handle->req_count-1);
        if(NBC_OK != res) { printf("Error in OF_Isend() (%i)\n", res); ret=res; goto error; }
#endif
#ifdef NBC_TIMING
    Isend_time += MPI_Wtime();
#endif
        break;
      case RECV:
        NBC_DEBUG(5, "  RECV (offset %li) ", (long)typeptr-(long)myschedule);
        recvargs = (NBC_Args_recv*)(typeptr+1);
        NBC_DEBUG(5, "*buf: %p, count: %i, type: %lu, source: %i, tag: %i)\n", recvargs->buf, recvargs->count, (unsigned long)recvargs->datatype, recvargs->source, handle->tag);
        typeptr = (NBC_Fn_type*)(((NBC_Args_recv*)typeptr)+1);
        /* get an additional request - TODO: req_count NOT thread safe */
        handle->req_count++;
        /* get buffer */
        if(recvargs->tmpbuf) {
          buf1=(char*)handle->tmpbuf+(long)recvargs->buf;
        } else {
          buf1=recvargs->buf;
        }
#ifdef NBC_TIMING
    Irecv_time -= MPI_Wtime();
#endif
#ifdef HAVE_OMPI
        handle->req_array = (MPI_Request*)realloc((void*)handle->req_array, (handle->req_count)*sizeof(MPI_Request));
        NBC_CHECK_NULL(handle->req_array);
        /*res = MCA_PML_CALL(irecv(buf1, recvargs->count, recvargs->datatype, recvargs->source, handle->tag, handle->mycomm, handle->req_array+handle->req_count-1)); 
        printf("MPI_Irecv(%lu, %i, %lu, %i, %i, %lu) (%i)\n", (unsigned long)buf1, recvargs->count, (unsigned long)recvargs->datatype, recvargs->source, handle->tag, (unsigned long)handle->mycomm, res); */
        res = MPI_Irecv(buf1, recvargs->count, recvargs->datatype, recvargs->source, handle->tag, handle->mycomm, handle->req_array+handle->req_count-1);
        if(OMPI_SUCCESS != res) { printf("Error in MPI_Irecv(%lu, %i, %lu, %i, %i, %lu) (%i)\n", (unsigned long)buf1, recvargs->count, (unsigned long)recvargs->datatype, recvargs->source, handle->tag, (unsigned long)handle->mycomm, res); ret=res; goto error; }
#endif
#ifdef HAVE_MPI
        handle->req_array = (MPI_Request*)realloc((void*)handle->req_array, (handle->req_count)*sizeof(MPI_Request));
        NBC_CHECK_NULL(handle->req_array);
        res = MPI_Irecv(buf1, recvargs->count, recvargs->datatype, recvargs->source, handle->tag, handle->mycomm, handle->req_array+handle->req_count-1);
        if(MPI_SUCCESS != res) { printf("Error in MPI_Irecv(%lu, %i, %lu, %i, %i, %lu) (%i)\n", (unsigned long)buf1, recvargs->count, (unsigned long)recvargs->datatype, recvargs->source, handle->tag, (unsigned long)handle->mycomm, res); ret=res; goto error; }
#endif
#ifdef HAVE_OFED
        handle->req_array = (void**)realloc((void**)handle->req_array, (handle->req_count)*sizeof(OF_Request));
        NBC_CHECK_NULL(handle->req_array);
        res = OF_Irecv(buf1, recvargs->count, recvargs->datatype, recvargs->source, handle->tag, handle->mycomm,  (OF_Request*)handle->req_array+handle->req_count-1);
        if(NBC_OK != res) { printf("Error in MPI_Irecv() (%i)\n", res); ret=res; goto error;}
#endif
#ifdef NBC_TIMING
    Irecv_time += MPI_Wtime();
#endif
        break;
      case OP:
        NBC_DEBUG(5, "  OP   (offset %li) ", (long)typeptr-(long)myschedule);
        opargs = (NBC_Args_op*)(typeptr+1);
        NBC_DEBUG(5, "*buf1: %p, buf2: %p, count: %i, type: %lu)\n", opargs->buf1, opargs->buf2, opargs->count, (unsigned long)opargs->datatype);
        typeptr = (NBC_Fn_type*)((NBC_Args_op*)typeptr+1);
        /* get buffers */
        if(opargs->tmpbuf1) 
          buf1=(char*)handle->tmpbuf+(long)opargs->buf1;
        else
          buf1=opargs->buf1;
        if(opargs->tmpbuf2) 
          buf2=(char*)handle->tmpbuf+(long)opargs->buf2;
        else
          buf2=opargs->buf2;
        if(opargs->tmpbuf3) 
          buf3=(char*)handle->tmpbuf+(long)opargs->buf3;
        else
          buf3=opargs->buf3;
        res = NBC_Operation(buf3, buf1, buf2, opargs->op, opargs->datatype, opargs->count, opargs->forceDense);
        if(res != NBC_OK) { printf("NBC_Operation() failed (code: %i)\n", res); ret=res; goto error; }
        break;
      case COPY:
        NBC_DEBUG(5, "  COPY   (offset %li) ", (long)typeptr-(long)myschedule);
        copyargs = (NBC_Args_copy*)(typeptr+1);
        NBC_DEBUG(5, "*src: %lu, srccount: %i, srctype: %lu, *tgt: %lu, tgtcount: %i, tgttype: %lu)\n", (unsigned long)copyargs->src, copyargs->srccount, (unsigned long)copyargs->srctype, (unsigned long)copyargs->tgt, copyargs->tgtcount, (unsigned long)copyargs->tgttype);
        typeptr = (NBC_Fn_type*)((NBC_Args_copy*)typeptr+1);
        /* get buffers */
        if(copyargs->tmpsrc) 
          buf1=(char*)handle->tmpbuf+(long)copyargs->src;
        else
          buf1=copyargs->src;
        if(copyargs->tmptgt) 
          buf2=(char*)handle->tmpbuf+(long)copyargs->tgt;
        else
          buf2=copyargs->tgt;

        // HACK: Get correct count
        if (copyargs->srccount < 0) {
          copyargs->srccount = countBytes(buf1, -1*copyargs->srccount);
          copyargs->tgtcount = copyargs->srccount;
        }

        res = NBC_Copy(buf1, copyargs->srccount, copyargs->srctype, buf2, copyargs->tgtcount, copyargs->tgttype, handle->mycomm);
        if(res != NBC_OK) { printf("NBC_Copy() failed (code: %i)\n", res); ret=res; goto error; }
        break;
      case UNPACK:
        NBC_DEBUG(5, "  UNPACK   (offset %li) ", (long)typeptr-(long)myschedule);
        unpackargs = (NBC_Args_unpack*)(typeptr+1);
        NBC_DEBUG(5, "*src: %lu, srccount: %i, srctype: %lu, *tgt: %lu\n", (unsigned long)unpackargs->inbuf, unpackargs->count, (unsigned long)unpackargs->datatype, (unsigned long)unpackargs->outbuf);
        typeptr = (NBC_Fn_type*)((NBC_Args_unpack*)typeptr+1);
        /* get buffers */
        if(unpackargs->tmpinbuf) 
          buf1=(char*)handle->tmpbuf+(long)unpackargs->inbuf;
        else
          buf1=unpackargs->outbuf;
        if(unpackargs->tmpoutbuf) 
          buf2=(char*)handle->tmpbuf+(long)unpackargs->outbuf;
        else
          buf2=unpackargs->outbuf;
        res = NBC_Unpack(buf1, unpackargs->count, unpackargs->datatype, buf2, handle->mycomm);
        if(res != NBC_OK) { printf("NBC_Unpack() failed (code: %i)\n", res); ret=res; goto error; }
        break;
      default:
        printf("NBC_Start_round: bad type %li at offset %li\n", (long)*typeptr, (long)typeptr-(long)myschedule);
        ret=NBC_BAD_SCHED;
        goto error;
    }
    /* increase ptr by size of fn_type enum */
    typeptr = (NBC_Fn_type*)((NBC_Fn_type*)typeptr+1);
  }

#ifdef HAVE_OFED
#ifndef HAVE_PROGRESS_THREAD
  /* if we have OFED, progress all requests until they run autonomously */
  OF_Startall(handle->req_count, (OF_Request*)handle->req_array, (int)1e6 /* timeout */);
#endif
#endif  

  /* check if we can make progress - not in the first round, this allows us to leave the
   * initialization faster and to reach more overlap 
   *
   * threaded case: calling progress in the first round can lead to a
   * deadlock if NBC_Free is called in this round :-( */
  if(handle->row_offset != sizeof(int)) {
    res = NBC_Progress(handle);
    if((NBC_OK != res) && (NBC_CONTINUE != res)) { printf("Error in NBC_Progress() (%i)\n", res); ret=res; goto error; }
  }

error:
  return ret;
}

static inline int NBC_Initialize() {
#ifdef HAVE_PROGRESS_THREAD 
#ifdef HAVE_OFED
  // we have to initialize ofed before starting the thread calling oed function :)...
  OF_Init();
#endif
  pthread_attr_t attr;
  struct sched_param param;

#ifdef HAVE_RT_THREAD 
  param.sched_priority = 90;
  pthread_attr_init(&attr);
  pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setschedparam(&attr, &param);
  pthread_attr_setschedpolicy(&attr, SCHED_FIFO);

  /* spawn the polling thread */
  int ret = pthread_create( &GNBC_Pthread, &attr, NBC_Pthread_func, NULL);
#else
  int ret = pthread_create( &GNBC_Pthread, NULL, NBC_Pthread_func, NULL);
#endif
  if(0 != ret) { printf("Error in pthread_create() (%i)\n", ret); return NBC_OOR; }

#ifdef HAVE_RT_THREAD 
  pthread_attr_destroy(&attr);
#endif
#endif
  
  GNBC_Initialized = 1;

  return NBC_OK;
}

int NBC_Init_handle(NBC_Handle *handle, MPI_Comm comm) {
  int res, flag;
  NBC_Comminfo *comminfo;

#ifdef HAVE_PROGRESS_THREAD // right now, we need this only if we need to start a progress thread
  if(!GNBC_Initialized) {
    res = NBC_Initialize();
    if(res != NBC_OK) return res;
  }

  /* init locks */
  pthread_mutex_init(&handle->lock, NULL);
  /* init semaphore */
  if(sem_init(&handle->semid, 0, 0) != 0) { perror("sem_init()"); }

#endif
#ifdef HAVE_DCMF
  handle->dcmf_hndl = NULL; // initialize
#endif

  handle->tmpbuf = NULL;
  handle->req_count = 0;
  handle->req_array = NULL;
  handle->comm = comm;
  handle->schedule = NULL;
  /* first int is the schedule size */
  handle->row_offset = sizeof(int);

  /******************** Do the tag and shadow comm administration ...  ***************/
  
  /* otherwise we have to do the normal attribute stuff :-( */
  /* keyval is not initialized yet, we have to init it */
  if(MPI_KEYVAL_INVALID == gkeyval) {
    res = MPI_Keyval_create(NBC_Key_copy, NBC_Key_delete, &(gkeyval), NULL); 
    if((MPI_SUCCESS != res)) { printf("Error in MPI_Keyval_create() (%i)\n", res); return res; }
  } 

  res = MPI_Attr_get(comm, gkeyval, &comminfo, &flag);
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_get() (%i)\n", res); return res; }

  if (flag) {
    /* we found it */
    comminfo->tag++;
  } else {
    /* we have to create a new one */
    comminfo = NBC_Init_comm(comm);
    if(comminfo == NULL) { printf("Error in NBC_Init_comm() %i\n", res); return NBC_OOR; }
  }
  handle->tag=comminfo->tag;
  handle->mycomm=comminfo->mycomm;
  /*printf("got comminfo: %lu tag: %i\n", comminfo, comminfo->tag);*/

  /* reset counter ... */ 
  if(handle->tag == 32767) {
    handle->tag=1;
    comminfo->tag=1;
    NBC_DEBUG(2,"resetting tags ...\n"); 
  }
  
  /******************** end of tag and shadow comm administration ...  ***************/
  handle->comminfo = comminfo;
  
  NBC_DEBUG(3, "got tag %i\n", handle->tag);

  return NBC_OK;
}

NBC_Comminfo* NBC_Init_comm(MPI_Comm comm) {
  int res;
  NBC_Comminfo *comminfo;

  comminfo = (NBC_Comminfo*)malloc(sizeof(NBC_Comminfo));
  if(comminfo == NULL) { printf("Error in malloc()\n"); return NULL; }

  /* set tag to 1 */
  comminfo->tag=1;
  /* dup and save shadow communicator */
  res = MPI_Comm_dup(comm, &(comminfo->mycomm));
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Comm_dup() (%i)\n", res); return NULL; }
  NBC_DEBUG(1, "created a shadow communicator for %lu ... %lu\n", (unsigned long)comm, (unsigned long)comminfo->mycomm);

/* MPI is not thread save, and OFED uses MPI calls to build the
 * communicator connections with the first message. So we need to call
 * it early (from the main thread) to do the build work ... here */
#ifdef HAVE_OFED
#ifdef HAVE_PROGRESS_THREAD
  OF_Comm_init(comminfo->mycomm);
#endif
#endif

#ifdef NBC_CACHE_SCHEDULE
  /* initialize the NBC_ALLTOALL SchedCache tree */
  comminfo->NBC_Dict[NBC_ALLTOALL] = hb_tree_new((dict_cmp_func)NBC_Alltoall_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_ALLTOALL] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_ALLTOALL]);
  comminfo->NBC_Dict_size[NBC_ALLTOALL] = 0;
  /* initialize the NBC_ALLGATHER SchedCache tree */
  comminfo->NBC_Dict[NBC_ALLGATHER] = hb_tree_new((dict_cmp_func)NBC_Allgather_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_ALLGATHER] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_ALLGATHER]);
  comminfo->NBC_Dict_size[NBC_ALLGATHER] = 0;
  /* initialize the NBC_ALLREDUCE SchedCache tree */
  comminfo->NBC_Dict[NBC_ALLREDUCE] = hb_tree_new((dict_cmp_func)NBC_Allreduce_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_ALLREDUCE] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_ALLREDUCE]);
  comminfo->NBC_Dict_size[NBC_ALLREDUCE] = 0;
  /* initialize the NBC_BARRIER SchedCache tree - is not needed -
   * schedule is hung off directly */
  comminfo->NBC_Dict_size[NBC_BARRIER] = 0;
  /* initialize the NBC_BCAST SchedCache tree */
  comminfo->NBC_Dict[NBC_BCAST] = hb_tree_new((dict_cmp_func)NBC_Bcast_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_BCAST] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_BCAST]);
  comminfo->NBC_Dict_size[NBC_BCAST] = 0;
  /* initialize the NBC_GATHER SchedCache tree */
  comminfo->NBC_Dict[NBC_GATHER] = hb_tree_new((dict_cmp_func)NBC_Gather_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_GATHER] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_GATHER]);
  comminfo->NBC_Dict_size[NBC_GATHER] = 0;
  /* initialize the NBC_REDUCE SchedCache tree */
  comminfo->NBC_Dict[NBC_REDUCE] = hb_tree_new((dict_cmp_func)NBC_Reduce_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_REDUCE] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_REDUCE]);
  comminfo->NBC_Dict_size[NBC_REDUCE] = 0;
  /* initialize the NBC_SCAN SchedCache tree */
  comminfo->NBC_Dict[NBC_SCAN] = hb_tree_new((dict_cmp_func)NBC_Scan_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_SCAN] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_SCAN]);
  comminfo->NBC_Dict_size[NBC_SCAN] = 0;
  /* initialize the NBC_SCATTER SchedCache tree */
  comminfo->NBC_Dict[NBC_SCATTER] = hb_tree_new((dict_cmp_func)NBC_Scatter_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_SCATTER] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_SCATTER]);
  comminfo->NBC_Dict_size[NBC_SCATTER] = 0;
  /* initialize the NBC_ICART_SHIFT_XCHG SchedCache tree */
  comminfo->NBC_Dict[NBC_CART_SHIFT_XCHG] = hb_tree_new((dict_cmp_func)NBC_Icart_shift_xchg_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_CART_SHIFT_XCHG] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_CART_SHIFT_XCHG]);
  comminfo->NBC_Dict_size[NBC_CART_SHIFT_XCHG] = 0;
  /* initialize the NBC_INEIGHBOR_XCHG SchedCache tree */
  comminfo->NBC_Dict[NBC_NEIGHBOR_XCHG] = hb_tree_new((dict_cmp_func)NBC_Ineighbor_xchg_args_compare, NBC_SchedCache_args_delete_key_dummy, NBC_SchedCache_args_delete);
  if(comminfo->NBC_Dict[NBC_NEIGHBOR_XCHG] == NULL) { printf("Error in hb_tree_new()\n"); return NULL; }
  NBC_DEBUG(1, "added tree at address %lu\n", (unsigned long)comminfo->NBC_Dict[NBC_NEIGHBOR_XCHG]);
  comminfo->NBC_Dict_size[NBC_NEIGHBOR_XCHG] = 0;
#endif

  /* put the new attribute to the comm */
  res = MPI_Attr_put(comm, gkeyval, comminfo); 
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_put() (%i)\n", res); return NULL; }

  return comminfo;
}

int NBC_Start(NBC_Handle *handle, NBC_Schedule *schedule) {
  int res;

  handle->schedule = schedule;

#ifdef HAVE_PROGRESS_THREAD 
  /* add handle to open handles - and give the control to the progress
   * thread - the handle must not be touched by the user thread from now
   * on !!! */
  pthread_mutex_lock(&GNBC_Pthread_handles_lock);
  GNBC_Pthread_handles.push_back(handle);
  pthread_mutex_unlock(&GNBC_Pthread_handles_lock);

  /* wake progress thread up */
#ifdef HAVE_MPI
  MPI_Send(&GMPI_dumbuf[1], 1, MPI_INT, 0, 0, MPI_COMM_SELF);
#endif
#ifdef HAVE_OFED
  OF_Wakeup();
#endif
#else
  /* kick off first round */
  res = NBC_Start_round(handle);
  if((NBC_OK != res)) { printf("Error in NBC_Start_round() (%i)\n", res); return res; }
#endif

  return NBC_OK;
}

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Wait=PNBC_Wait
#define NBC_Wait PNBC_Wait
#endif
int NBC_Wait(NBC_Handle *handle, MPI_Status *status) {
  
#ifdef HAVE_PROGRESS_THREAD
  pthread_mutex_lock(&handle->lock);
#endif
#ifndef HAVE_DCMF // DCMF doesn't initialize the schedule! TODO: generalize (DCMF could use same object!)
  /* the request is done or invalid if there is no schedule attached 
   * we assume done */
  if(handle->schedule == NULL) {
#ifdef HAVE_PROGRESS_THREAD 
    pthread_mutex_unlock(&handle->lock);
#endif
    return NBC_OK;
  }
#endif
#ifdef HAVE_PROGRESS_THREAD 
  pthread_mutex_unlock(&handle->lock);
#endif

#ifdef HAVE_PROGRESS_THREAD 
  /* wait on semaphore */
  if(sem_wait(&handle->semid) != 0) { perror("sem_wait()"); }
  if(sem_destroy(&handle->semid) != 0) { perror("sem_destroy()"); }
#else
  /* poll */
  while(NBC_OK != NBC_Progress(handle));
#endif    

  NBC_DEBUG(3, "finished request with tag %i\n", handle->tag);
  
  return NBC_OK;
}

// this is simply for backwards compatibility for the old, non-MPI compliant LibNBC
#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Testold=PNBC_Testold
#define NBC_Testold PNBC_Testold
#endif
int NBC_Testold(NBC_Handle *handle) {
#ifdef HAVE_PROGRESS_THREAD
  int ret=NBC_CONTINUE;
  pthread_mutex_lock(&handle->lock);
  if(handle->schedule == NULL) ret = NBC_OK;
  pthread_mutex_unlock(&handle->lock);
  return ret;
#else
  return NBC_Progress(handle);
#endif
}

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Test=PNBC_Test
#define NBC_Test PNBC_Test
#endif
int NBC_Test(NBC_Handle *handle, int *flag, MPI_Status *status) {
#ifdef HAVE_PROGRESS_THREAD
  int ret=NBC_CONTINUE;
  pthread_mutex_lock(&handle->lock);
  if(handle->schedule == NULL) ret = NBC_OK;
  pthread_mutex_unlock(&handle->lock);
#else
  int ret = NBC_Progress(handle);
#endif
  *flag = ret;

  return NBC_OK;
}


#ifdef NBC_CACHE_SCHEDULE
void NBC_SchedCache_args_delete_key_dummy(void *k) {
    /* do nothing because the key and the data element are identical :-) 
     * both (the single one :) is freed in NBC_<COLLOP>_args_delete() */
}

void NBC_SchedCache_args_delete(void *entry) {
  struct NBC_dummyarg *tmp;
  
  tmp = (struct NBC_dummyarg*)entry;
  /* free taglistentry */
  free((void*)*(tmp->schedule));
  /* the schedule pointer itself is also malloc'd */
  free((void*)tmp->schedule);
  free((void*)tmp);
}
#endif
