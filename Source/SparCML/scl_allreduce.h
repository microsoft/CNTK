#pragma once

#include "scl.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define OUTPUT_RANK 0

/*
 * Sparse AllReduce following a recoursive doubling like algorithm
 */
template<class IdxType, class ValType> int scl_allreduce(const struct stream<IdxType, ValType> *sendbuf, struct stream<IdxType, ValType> *recvbuf, unsigned k, unsigned dim, MPI_Comm comm) {

  int rank, worldsize;
  MPI_Comm_size(comm, &worldsize);
  MPI_Comm_rank(comm, &rank);

  size_t maxbytes = sizeof(unsigned) + dim * sizeof(ValType);
  // TODO Uncomment if check that no dense allreduce will ocure
  //size_t maxbytes = 0;
  //if(dim * sizeof(ValType) < k *  * (sizeof(IdxType) + sizeof(ValType))) {
  //  maxbytes = sizeof(unsigned) + dim * sizeof(ValType);
  //} else {
  //  maxbytes = sizeof(unsigned) + maxelements * (sizeof(IdxType) + sizeof(ValType));
  //}
  struct stream<IdxType, ValType> *tmp1 = (struct stream<IdxType, ValType> *) malloc(maxbytes);
  struct stream<IdxType, ValType> *tmp2 = (struct stream<IdxType, ValType> *) malloc(maxbytes);

  // Do the recusrive halfing
  int vrank = rank;
  unsigned pof2 = pow(2, floor(log2(worldsize)));
  int rem = worldsize - pof2;
  unsigned mask = 1;

  if (rank < 2 * rem) {
    // Collapse to power of two
    if(rank % 2 == 0) {
      MPI_Send(sendbuf, countBytes(sendbuf, dim), MPI_BYTE, rank+1, 1, comm);
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
    memcpy(tmp1, sendbuf, countBytes(sendbuf, dim));
  }

  // TODO If T == 0

  const unsigned divFactor = 3;
  const unsigned sizeDiff = (sizeof(IdxType) + sizeof(ValType)) / sizeof(ValType);
  unsigned T = floor(log2(dim / (sizeDiff * k * divFactor)));
  if(T < 1) T = 1;

  if(rank == OUTPUT_RANK) printf("Set T = %d\n", T);

  unsigned t = 0;
  bool firstBuf = true;
  if(vrank != -1) {
    // Recursive doubling
    while(mask < pof2 && t < T) {
      int dest = vrank ^ mask; // bitwise xor
      if(dest < rem) {
        dest = dest*2 + 1;
      } else {
        dest = dest + rem;
      }

      if(firstBuf) {
        MPI_Sendrecv(tmp1, countBytes(tmp1, dim), MPI_BYTE, dest, 1, recvbuf, maxbytes, MPI_BYTE, dest, 1, comm, MPI_STATUS_IGNORE);
      } else {
        MPI_Sendrecv(tmp2, countBytes(tmp2, dim), MPI_BYTE, dest, 1, recvbuf, maxbytes, MPI_BYTE, dest, 1, comm, MPI_STATUS_IGNORE);
      }

      //if(rank == OUTPUT_RANK) printf("[Rank %d] Got sparse data from %d\n", rank, dest);
      
      // Sum sparse items
      if(firstBuf) {
        sum_streams<IdxType, ValType>(tmp1, recvbuf, tmp2, dim, t == T-1 && (mask << 1) < pof2);
      } else {
        sum_streams<IdxType, ValType>(tmp2, recvbuf, tmp1, dim, t == T-1 && (mask << 1)< pof2);
      }

      firstBuf = !firstBuf;

      mask = mask << 1;
      t++;
    }

    struct stream<IdxType, ValType> *buf = firstBuf ? tmp1 : tmp2;

    //if(rank == OUTPUT_RANK) printf("Sparse AllReduce Part finished. Dense: %s\n", buf->nofitems == dim ? "True" : "False");
    
    //if(rank == OUTPUT_RANK) printStream(buf, dim, rank);

    if(mask < pof2) {

      //if(rank == OUTPUT_RANK) printf("Entering Dense AllReduce Part!!\n");

      unsigned nel = pof2/mask;
      int disps[nel];

      unsigned frac = dim / nel;
      unsigned lastFrac =  dim - frac*(nel-1);

      //if(rank == OUTPUT_RANK) printf("nof elements: %d / cnt per p: %d / last p: %d\n", nel, frac, lastFrac);

      disps[0] = 0;
      for (unsigned i = 1; i < nel; i++) {
        disps[i] = disps[i-1] + frac;
      }

      unsigned send_idx = 0;
      unsigned recv_idx = 0;
      unsigned send_cnt = 0;
      unsigned recv_cnt = 0;

      // Reduce-Scatter on dense by recursive halving
      unsigned mask2 = pof2 >> 1;
      while(mask2 >= mask) {
        int vdest = vrank ^ mask2; // bitwise xor
        int dest = vdest;
        if(dest < rem) {
          dest = dest*2 + 1;
        } else {
          dest = dest + rem;
        }

        unsigned cnt = mask2/mask;

        //if(rank == OUTPUT_RANK) printf("[Rank: %d] runnnig with mask %u on cnt %u\n", rank, mask2, cnt);

        if(vrank < vdest) {
          send_idx = recv_idx + cnt;
        } else {
          recv_idx = send_idx + cnt;
        }

        send_cnt = send_idx + cnt == nel ? ((cnt-1) * frac) + lastFrac : cnt * frac;
        recv_cnt = recv_idx + cnt == nel ? ((cnt-1) * frac) + lastFrac : cnt * frac;

        // Send
        //if(rank == OUTPUT_RANK) printf("[Rank: %d] reducing %d buckets with %d (send_cnt: %d, recv_cnt: %d)\n", rank, cnt, dest, send_cnt, recv_cnt);
        MPI_Sendrecv(buf->items + (disps[send_idx] * sizeof(ValType)), send_cnt * sizeof(ValType), MPI_BYTE, dest, 1, recvbuf->items + (disps[recv_idx] * sizeof(ValType)), recv_cnt * sizeof(ValType), MPI_BYTE, dest, 1, comm, MPI_STATUS_IGNORE);

        // Dense sumation
        for(size_t i = disps[recv_idx]; i < disps[recv_idx] + recv_cnt; ++i) {
          ((ValType *)buf->items)[i] += ((ValType *)recvbuf->items)[i];
        }

        //if(rank == OUTPUT_RANK) printStream(buf, dim, rank);

        send_idx = recv_idx;
        // update mask2
        mask2 = mask2 >> 1;
      }

      // Gather all the data by recursive doubling
      send_idx = vrank / mask;
      recv_idx = 0;
      mask2 = mask;
      while(mask2 < pof2) {
        int vdest = vrank ^ mask2; // bitwise xor
        int dest = vdest;
        if(dest < rem) {
          dest = dest*2 + 1;
        } else {
          dest = dest + rem;
        }

        unsigned cnt = mask2/mask;
        if(vrank < vdest) {
          recv_idx = send_idx + cnt;
        } else {
          recv_idx = send_idx - cnt;
        }

        send_cnt = send_idx + cnt == nel ? ((cnt-1) * frac) + lastFrac : cnt * frac;
        recv_cnt = recv_idx + cnt == nel ? ((cnt-1) * frac) + lastFrac : cnt * frac;

        //if(rank == OUTPUT_RANK) printf("[Rank: %d] sending %d and receiving %d from %d\n", rank, send_cnt, recv_cnt, dest);
        MPI_Sendrecv(buf->items + (disps[send_idx] * sizeof(ValType)), send_cnt * sizeof(ValType), MPI_BYTE, dest, 1, buf->items + (disps[recv_idx] * sizeof(ValType)), recv_cnt * sizeof(ValType), MPI_BYTE, dest, 1, comm, MPI_STATUS_IGNORE);

        //if(rank == OUTPUT_RANK) printStream(buf, dim, rank);

        // update send_idx and recv_idx
        if(vrank > vdest) {
          send_idx = recv_idx;
        }

        mask2 = mask2 << 1;
      }
    }

    // Copy into recv buffer
    memcpy(recvbuf, buf, countBytes(buf, dim));
  }

  if (rank < 2*rem) {
    // Expand from power of two
    if(rank % 2 > 0) {
      MPI_Send(recvbuf, countBytes(recvbuf, dim), MPI_BYTE, rank-1, 1, comm);
    } else {
      MPI_Recv(recvbuf, maxbytes, MPI_BYTE, rank+1, 1, comm, MPI_STATUS_IGNORE);
    }
  }

  free(tmp1);
  free(tmp2);

  return MPI_SUCCESS;
}
