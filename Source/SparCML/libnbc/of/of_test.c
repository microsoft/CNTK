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


int main(int argc, char *argv[]) {
  int rank, res, size, i, loops, p, peer;
  void *buf2;
  OF_Request* req = new OF_Request[2];
  double ts, tr, tg;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  {
    char name[100];
    
    gethostname(name, 100);
    printf("process %i on node %s is waiting 10 secs\n", getpid(), name);
    //sleep(10);
  }

  size = 8000;
  buf2 = malloc(size);

#define P2P 1000
  if(rank%2 == 0) {
    int k;

    /*
    for(k=0; k<P2P; k++) {
      //if(0==(k%(P2P/10))) printf(".\n");
      res = OF_Isend(buf2, size, MPI_BYTE, rank+1, 10, MPI_COMM_WORLD, req);
      if(res) printf("Error in OF_Send (%i) \n", res);
      res = OF_Wait(&req[0]);
    }
    printf("send loop done\n");*/

    tr = -MPI_Wtime();
    res = OF_Irecv(buf2, size, MPI_BYTE, rank+1, 11, MPI_COMM_WORLD, &req[0]);
    tr += MPI_Wtime();
    ts = -MPI_Wtime();
    res = OF_Isend(buf2, size, MPI_BYTE, rank+1, 11, MPI_COMM_WORLD, &req[1]);
    ts += MPI_Wtime();
    res = OF_Waitall(2, &req[0]);
    
    printf("[%i] send %lf recv %lf \n", rank, ts*1e6, tr*1e6);

    tr = -MPI_Wtime();
    tg = 0;
    for(k=0; k<P2P; k++) {
    ts = -MPI_Wtime();
    res = OF_Isend(buf2, size, MPI_BYTE, rank+1, 11, MPI_COMM_WORLD, &req[0]);
    ts += MPI_Wtime();
    tg += ts;
    res = OF_Wait(&req[0]);
    res = OF_Irecv(buf2, size, MPI_BYTE, rank+1, 11, MPI_COMM_WORLD, &req[1]);
    res = OF_Wait(&req[1]);
    }
    tr += MPI_Wtime();

    printf("[%i] OFED ping pong: %lf (send: %lf)\n", rank, tr*1e6/P2P, tg*1e6/P2P);

    tr = -MPI_Wtime();
    tg = 0;
    for(k=0; k<P2P; k++) {
      MPI_Request mpireq[2];
    ts = -MPI_Wtime();
    res = MPI_Isend(buf2, size, MPI_BYTE, rank+1, 11, MPI_COMM_WORLD, &mpireq[0]);
    ts += MPI_Wtime();
    tg += ts;
    res = MPI_Wait(&mpireq[0], MPI_STATUS_IGNORE);
    res = MPI_Irecv(buf2, size, MPI_BYTE, rank+1, 11, MPI_COMM_WORLD, &mpireq[1]);
    res = MPI_Wait(&mpireq[1], MPI_STATUS_IGNORE);
    }
    tr += MPI_Wtime();

    printf("[%i] MPI ping pong: %lf (send: %lf)\n", rank, tr*1e6/P2P, tg*1e6/P2P);
    
  } else {
    int k;
    
    /*
    for(k=0; k<P2P; k++) {
      res = OF_Irecv(buf2, size, MPI_BYTE, rank-1, 10, MPI_COMM_WORLD, req);
      if(res) printf("Error in OF_Recv (%i)\n", res);
      res = OF_Wait(&req[0]);
      printf("recvd %i\n", k);
    }
    printf("recv loop done\n");*/

    tr = -MPI_Wtime();
    res = OF_Irecv(buf2, size, MPI_BYTE, rank-1, 11, MPI_COMM_WORLD, &req[0]);
    tr += MPI_Wtime();
    ts = -MPI_Wtime();
    res = OF_Isend(buf2, size, MPI_BYTE, rank-1, 11, MPI_COMM_WORLD, &req[1]);
    ts += MPI_Wtime();
    if(res) printf("Error in OF_Recv (%i)\n", res);
    res = OF_Waitall(2, &req[0]);
    
    printf("[%i] send %lf recv %lf \n", rank, ts*1e6, tr*1e6);

    //sleep(1);
    tg = 0;
    tr = -MPI_Wtime();
    for(k=0; k<P2P; k++) {
    ts = -MPI_Wtime();
    res = OF_Irecv(buf2, size, MPI_BYTE, rank-1, 11, MPI_COMM_WORLD, &req[0]);
    ts += MPI_Wtime();
    res = OF_Wait(&req[0]);
    tg += ts;
    res = OF_Isend(buf2, size, MPI_BYTE, rank-1, 11, MPI_COMM_WORLD, &req[1]);
    res = OF_Wait(&req[1]);
    }
    tr += MPI_Wtime();

    printf("[%i] OFED (fake) ping pong: %lf (recv: %lf)\n", rank, tr*1e6/P2P, tg*1e6/P2P);
  
    tg = 0;
    tr = -MPI_Wtime();
    for(k=0; k<P2P; k++) {
      MPI_Request mpireq[2];
    ts = -MPI_Wtime();
    res = MPI_Irecv(buf2, size, MPI_BYTE, rank-1, 11, MPI_COMM_WORLD, &mpireq[0]);
    ts += MPI_Wtime();
    res = MPI_Wait(&mpireq[0], MPI_STATUS_IGNORE);
    tg += ts;
    res = MPI_Isend(buf2, size, MPI_BYTE, rank-1, 11, MPI_COMM_WORLD, &mpireq[1]);
    res = MPI_Wait(&mpireq[1], MPI_STATUS_IGNORE);
    }
    tr += MPI_Wtime();

    printf("[%i] MPI (fake) ping pong: %lf (recv: %lf)\n", rank, tr*1e6/P2P, tg*1e6/P2P);
  
  }
  printf("[%i] after loop\n", rank);

  MPI_Finalize();
}
