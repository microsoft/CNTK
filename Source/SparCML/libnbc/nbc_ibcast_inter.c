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

#ifdef __cplusplus
extern "C" {
#endif

int NBC_Ibcast_inter(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, NBC_Handle* handle) {
  int rank, p, res, size, segsize, peer;
  NBC_Schedule *schedule;
  
  res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Comm_size(comm, &p);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Type_size(datatype, &size);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_size() (%i)\n", res); return res; }
  
  handle->tmpbuf=NULL;

  schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
  
  res = NBC_Sched_create(schedule);
  if(res != NBC_OK) { printf("Error in NBC_Sched_create, res = %i\n", res); return res; }

  if(root != MPI_PROC_NULL) {
    /* send to all others */
    if(root == MPI_ROOT) {
      int remsize;

      res = MPI_Comm_remote_size(comm, &remsize);
      if(MPI_SUCCESS != res) { printf("MPI_Comm_remote_size() failed\n", res); return res; }

      for (peer=0;peer<remsize;peer++) {
        /* send msg to peer */
        res = NBC_Sched_send(buffer, false, count, datatype, peer, schedule);
        if (NBC_OK != res) { printf("Error in NBC_Sched_send() (%i)\n", res); return res; }
      }
    } else {
      /* recv msg from root */
      res = NBC_Sched_recv(buffer, false, count, datatype, root, schedule);
      if (NBC_OK != res) { printf("Error in NBC_Sched_recv() (%i)\n", res); return res; }
    }
  }
  
  res = NBC_Sched_commit(schedule);
  if (NBC_OK != res) { printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }
  
  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { printf("Error in NBC_Start() (%i)\n", res); return res; }
  
  return NBC_OK;
}

#ifdef __cplusplus
}
#endif
