#pragma once

#include "c_common.h"

/*
 * Sparse AllReduce
 */
template<class IdxType, class ValType> int c_allreduce_big(const struct stream *sendbuf, struct stream *recvbuf, unsigned dim) {

  // TODO: ReC hack
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, worldsize;
  MPI_Comm_size(comm, &worldsize);
  MPI_Comm_rank(comm, &rank);

  if(worldsize == 1) {
    memcpy(recvbuf, sendbuf, countBytes<IdxType, ValType>(sendbuf, dim));
    return MPI_SUCCESS;
  }

  char* buf = (char *) malloc(worldsize * sizeof(unsigned) + dim * sizeof(ValType));
  struct stream* splits[worldsize];
  struct stream **recvs = (struct stream **) malloc((worldsize-1) * sizeof(struct stream*));

  unsigned size = dim / worldsize;
  unsigned lastsize = dim - ((worldsize-1) * size);

  for(int i = 0; i < worldsize; ++i) {
    splits[i] = (struct stream *)(buf + i * (sizeof(unsigned) + size * sizeof(ValType)));
  }

  unsigned mymaxsize = size; 
  if(rank == worldsize - 1) {
    mymaxsize = lastsize;
  }
  size_t mymaxbytes = sizeof(unsigned) + mymaxsize * sizeof(ValType);

  MPI_Request requests[worldsize-1];
  for(int i = 0; i < worldsize; ++i) {
    if(i != rank) {
      unsigned idx = i < rank ? i : i-1;
      recvs[idx] = (struct stream *)malloc(mymaxbytes);
      MPI_Irecv(recvs[idx], mymaxbytes, MPI_BYTE, i, 1, comm, &requests[idx]);
    }
  }

  // Split streams
  split_stream<IdxType, ValType>(sendbuf, splits, dim, worldsize);

  struct stream *mybuf = (struct stream *) malloc(mymaxbytes);
  struct stream *mytmp2 = (struct stream *) malloc(mymaxbytes);
  struct stream *mytmp3 = NULL;

  if(splits[rank]->nofitems == mymaxsize) {
    memcpy(mybuf, splits[rank], countBytes<IdxType, ValType>(splits[rank], mymaxsize));
  } else {
    unsigned idx = 0;
    mybuf->nofitems = mymaxsize;
    struct s_item<IdxType, ValType>* values = (struct s_item<IdxType, ValType> *)splits[rank]->items;
    ValType * newVals = (ValType *)mybuf->items;
    for(size_t i = 0; i < mymaxsize; ++i) {
      if(idx < splits[rank]->nofitems && values[idx].idx == i) {
        newVals[i] = values[idx].val;
        idx++;
      } else {
        newVals[i] = 0.0;
      }
    }
  }

  //printStream(splits[worldsize-1], lastsize, rank);

  int cnt[worldsize];
  int disp[worldsize];
  disp[0] = 0;

  for(int i = 0; i < worldsize; ++i) {
    if(i != rank) {
      MPI_Request req;
      MPI_Isend(splits[i], countBytes<IdxType, ValType>(splits[i], i < worldsize - 1 ? size : lastsize), MPI_BYTE, i, 1, comm, &req);
    }

    // prepare arrays for AllGatherV
    cnt[i] = i < worldsize - 1 ? size * sizeof(ValType) : lastsize * sizeof(ValType);
    if(i > 0) {
      disp[i] = disp[i-1] + cnt[i-1];
    }
  }

  int pending = worldsize-1;
  while(pending > 0) {
    int index;
    MPI_Status status;
    MPI_Waitany(worldsize-1, &requests[0], &index, &status); // request should be automatically changed to MPI_REQUEST_NULL by Waitany
    if(index == MPI_UNDEFINED) {
      printf("Unexpected error!\n");
      MPI_Abort(MPI_COMM_WORLD, 1); 
    }

    struct stream* res = sum_into_first_stream<IdxType, ValType>(mybuf, recvs[index], mytmp2, mymaxsize);
    if(res != mybuf) {
      mytmp3 = mybuf;
      mybuf = mytmp2;
      mytmp2 = mytmp3;
    }

    pending--;
  }

  //if(rank == worldsize-1) {
  //  printStream(mybuf, mymaxsize, rank);
  //}
  
  assert(mybuf->nofitems == mymaxsize);

  recvbuf->nofitems = dim;
  MPI_Allgatherv(mybuf->items, mymaxsize * sizeof(ValType), MPI_BYTE, recvbuf->items, &cnt[0], &disp[0], MPI_BYTE, comm);

  free(mybuf);
  free(mytmp2);
  free(buf);
  for(int i = 0; i < worldsize-1; ++i) {
    free(recvs[i]);
  }
  free(recvs);

  return MPI_SUCCESS;
}
