#pragma once

#include "c_common.h"

/*
 * Sparse AllReduce following a recoursive doubling like algorithm
 */
template<class IdxType, class ValType> int c_allreduce_recdoubling(const struct stream *sendbuf, struct stream *recvbuf, unsigned dim) {

  // TODO: ReC hack
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, worldsize;
  MPI_Comm_size(comm, &worldsize);
  MPI_Comm_rank(comm, &rank);

  if(worldsize == 1) {
    memcpy(recvbuf, sendbuf, countBytes<IdxType, ValType>(sendbuf, dim));
    return MPI_SUCCESS;
  }

  // TODO Maybe allocate only cnt*worldsize if smaller than dim
  size_t maxbytes = sizeof(unsigned) + dim * sizeof(ValType);
  struct stream *tmp1 = (struct stream *) malloc(maxbytes);
  struct stream *tmp2 = (struct stream *) malloc(maxbytes);

  // Do the recusrive halfing
  int vrank = rank;
  unsigned pof2 = pow(2, floor(log2(worldsize)));
  int rem = worldsize - pof2;
  unsigned mask = 1;

  if (rank < 2 * rem) {
    // Collapse to power of two
    if(rank % 2 == 0) {
      MPI_Send(sendbuf, countBytes<IdxType, ValType>(sendbuf, dim), MPI_BYTE, rank+1, 1, comm);
      vrank = -1;
    } else {
      MPI_Recv(recvbuf, maxbytes, MPI_BYTE, rank-1, 1, comm, MPI_STATUS_IGNORE);

      // Sum sparse items
      sum_streams<IdxType, ValType>(sendbuf, recvbuf, tmp1, dim);

      vrank = rank / 2;
    }
  } else {
    vrank = rank - rem;

    // copy into temp res buf
    memcpy(tmp1, sendbuf, countBytes<IdxType, ValType>(sendbuf, dim));
  }

  bool firstBuf = true;
  if(vrank != -1) {
    // Recursive doubling
    while(mask < pof2) {
      int dest = vrank ^ mask; // bitwise xor
      if(dest < rem) {
        dest = dest*2 + 1;
      } else {
        dest = dest + rem;
      }

      if(firstBuf) {
        MPI_Sendrecv(tmp1, countBytes<IdxType, ValType>(tmp1, dim), MPI_BYTE, dest, 1, recvbuf, maxbytes, MPI_BYTE, dest, 1, comm, MPI_STATUS_IGNORE);
      } else {
        MPI_Sendrecv(tmp2, countBytes<IdxType, ValType>(tmp2, dim), MPI_BYTE, dest, 1, recvbuf, maxbytes, MPI_BYTE, dest, 1, comm, MPI_STATUS_IGNORE);
      }
      
      // Sum sparse items
      if(firstBuf) {
        sum_streams<IdxType, ValType>(tmp1, recvbuf, tmp2, dim);
      } else {
        sum_streams<IdxType, ValType>(tmp2, recvbuf, tmp1, dim);
      }

      firstBuf = !firstBuf;

      mask = mask << 1;
    }

    // Copy into recv buffer
    if(firstBuf) {
      memcpy(recvbuf, tmp1, countBytes<IdxType, ValType>(tmp1, dim));
    } else {
      memcpy(recvbuf, tmp2, countBytes<IdxType, ValType>(tmp2, dim));
    }
  }

  if (rank < 2*rem) {
    // Expand from power of two
    if(rank % 2 > 0) {
      MPI_Send(recvbuf, countBytes<IdxType, ValType>(recvbuf, dim), MPI_BYTE, rank-1, 1, comm);
    } else {
      MPI_Recv(recvbuf, maxbytes, MPI_BYTE, rank+1, 1, comm, MPI_STATUS_IGNORE);
    }
  }

  free(tmp1);
  free(tmp2);

  return MPI_SUCCESS;
}
