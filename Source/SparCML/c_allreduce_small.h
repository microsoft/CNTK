#pragma once

#include "c_common.h"

/*
 * Sparse AllReduce
 */
template<class IdxType, class ValType> int c_allreduce_small(const struct stream *sendbuf, struct stream *recvbuf, unsigned dim) {

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
  struct stream *tmp3 = (struct stream *) malloc(maxbytes);

  // Do the recusrive halfing
  int vrank = rank;
  int pof2 = pow(2, floor(log2(worldsize)));
  int rem = worldsize - pof2;
  int mask = 1;

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

  struct stream *tmpbuf = NULL;
  bool freeAllToAll = false;
  struct stream *mytmp1 = NULL;
  struct stream *mytmp2 = NULL;
  char* buf = NULL;
  struct stream **recvs = NULL;

  unsigned size = dim / pof2;
  unsigned lastsize = dim - ((pof2-1) * size);
  unsigned offset = size*vrank;

  bool firstBuf = true;
  if(vrank != -1) {

    struct stream *mybuf = firstBuf ? tmp1 : tmp2;

    // Do AllToAll on subgroups
    freeAllToAll  = true;
    buf = (char *) malloc(pof2 * sizeof(unsigned) + dim * sizeof(ValType));
    struct stream* splits[pof2];
    recvs = (struct stream **) malloc((pof2-1) * sizeof(struct stream*));

    for(int i = 0; i < pof2; ++i) {
      splits[i] = (struct stream *)(buf + i * (sizeof(unsigned) + size * sizeof(ValType)));
    }

    unsigned mymaxsize = size; 
    if(vrank == pof2 - 1) {
      mymaxsize = lastsize;
    }
    size_t mymaxbytes = sizeof(unsigned) + mymaxsize * sizeof(ValType);

    MPI_Request requests[pof2-1];
    for(int i = 0; i < pof2; ++i) {
      if(i != vrank) {
        int idx = i < vrank ? i : i-1;
        recvs[idx] = (struct stream *)malloc(mymaxbytes);
        int vdest = i;
        if(vdest < rem) {
          vdest = vdest*2 + 1;
        } else {
          vdest = vdest + rem;
        }
        MPI_Irecv(recvs[idx], mymaxbytes, MPI_BYTE, vdest, 1, comm, &requests[idx]);
      }
    }

    // Split streams
    split_stream<IdxType, ValType>(mybuf, splits, dim, pof2);

    mytmp1 = (struct stream *) malloc(mymaxbytes);
    mytmp2 = (struct stream *) malloc(mymaxbytes);

    memcpy(mytmp1, splits[vrank], countBytes<IdxType, ValType>(splits[vrank], mymaxsize));
    bool myFirstBuf = true;

    for(int i = 0; i < pof2; ++i) {
      if(i != vrank) {
        MPI_Request req;
        int vdest = i;
        if(vdest < rem) {
          vdest = vdest*2 + 1;
        } else {
          vdest = vdest + rem;
        }
        MPI_Isend(splits[i], countBytes<IdxType, ValType>(splits[i], i < pof2 - 1 ? size : lastsize), MPI_BYTE, vdest, 1, comm, &req);
      }
    }

    int pending = pof2-1;
    while(pending > 0) {
      int index;
      MPI_Status status;
      MPI_Waitany(pof2-1, &requests[0], &index, &status); // request should be automatically changed to MPI_REQUEST_NULL by Waitany
      if(index == MPI_UNDEFINED) {
        printf("Unexpected error!\n");
        MPI_Abort(MPI_COMM_WORLD, 1); 
      }

      // Sum sparse items
      if(myFirstBuf) {
        sum_streams<IdxType, ValType>(mytmp1, recvs[index], mytmp2, mymaxsize);
      } else {
        sum_streams<IdxType, ValType>(mytmp2, recvs[index], mytmp1, mymaxsize);
      }

      myFirstBuf = !myFirstBuf;

      pending--;
    }

    mybuf = myFirstBuf ? mytmp1 : mytmp2;

    // Copy into tmp1 (omit copying if first sum_into_stream is different
    // TODO What if it already should be dense?
    struct s_item<IdxType, ValType> *values = (struct s_item<IdxType, ValType> *)tmp1->items;
    int idx = 0;
    if(mybuf->nofitems == mymaxsize) {
      // Dense
      ValType *myvals = (ValType *)mybuf->items;
      for(unsigned i = 0; i < mybuf->nofitems; ++i) {
        if(fabs(myvals[i]) > 1e-8) {
          values[idx].idx = offset + i;
          values[idx].val = myvals[i];
          idx++;
        }
      }
    } else {
      struct s_item<IdxType, ValType> *myvals = (struct s_item<IdxType, ValType> *)mybuf->items;
      // TODO Use memcopy instead
      for(unsigned i = 0; i < mybuf->nofitems; ++i) {
        values[idx].idx = offset + myvals[i].idx;
        values[idx].val = myvals[i].val;
        idx++;
      }
    }
    tmp1->nofitems = idx;
    assert(tmp1->nofitems < dim);

    // Recursive doubling
    while(mask < pof2) {
      int vdest = vrank ^ mask; // bitwise xor
      int dest = vdest;
      if(dest < rem) {
        dest = dest*2 + 1;
      } else {
        dest = dest + rem;
      }


      // Continue with RecDoublgin
      MPI_Sendrecv(tmp1, countBytes<IdxType, ValType>(tmp1, dim), MPI_BYTE, dest, 1, tmp2, maxbytes, MPI_BYTE, dest, 1, comm, MPI_STATUS_IGNORE);

      tmpbuf = sum_into_stream<IdxType, ValType>(tmp1, tmp2, tmp3, dim, true);
      if(tmpbuf == tmp2) {
        tmp2 = tmp1;
        tmp1 = tmpbuf;
      } else if(tmpbuf == tmp3) {
        tmp3 = tmp1;
        tmp1 = tmpbuf;
      }


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

  if(freeAllToAll) {
    free(mytmp1);
    free(mytmp2);
    free(buf);
    for(int i = 0; i < pof2-1; ++i) {
      free(recvs[i]);
    }
    free(recvs);
  }

  free(tmp1);
  free(tmp2);
  free(tmp3);

  return MPI_SUCCESS;
}
