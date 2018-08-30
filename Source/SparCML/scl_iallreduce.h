#pragma once

//#include "scl.h"
#include "c_common.h"
#include "libnbc/nbc_internal.h"

#include <math.h>
#include <stdlib.h>
#include <type_traits>

/*
 * Sparse AllReduce following a recoursive doubling like algorithm
 */
//template<class IdxType, class ValType> int scl_iallreduce(struct stream<IdxType, ValType> *sendbuf, struct stream<IdxType, ValType> *recvbuf, unsigned k, unsigned dim, MPI_Comm comm, NBC_Handle *handle) {
template<class IdxType, class ValType> int scl_iallreduce(struct stream *sendbuf, struct stream *recvbuf, unsigned k, unsigned dim, MPI_Comm comm, NBC_Handle *handle) {

  MPI_Datatype mpiT;
  if(std::is_same<ValType, int>::value) {
    mpiT = MPI_INT;
  } else if(std::is_same<ValType, float>::value) {
    mpiT = MPI_FLOAT;
  } else if(std::is_same<ValType, double>::value) {
    mpiT = MPI_DOUBLE;
  } else {
    printf("Error: Unsupported type!");
    return NBC_DATATYPE_NOT_SUPPORTED;
  }

  int rank, worldsize;

  NBC_Schedule *schedule;
  int res = NBC_Init_handle(handle, comm);
  if(res != NBC_OK) { printf("Error in NBC_Init_handle(%i)\n", res); return res; }
  res = MPI_Comm_rank(comm, &rank);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_rank() (%i)\n", res); return res; }
  res = MPI_Comm_size(comm, &worldsize);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Comm_size() (%i)\n", res); return res; }

  size_t maxbytes = sizeof(unsigned) + dim * sizeof(ValType);
  // TODO Uncomment if check that no dense allreduce will ocure
  //size_t maxbytes = 0;
  //if(dim * sizeof(ValType) < maxelements * (sizeof(IdxType) + sizeof(ValType))) {
  //  maxbytes = sizeof(unsigned) + dim * sizeof(ValType);
  //} else {
  //  maxbytes = sizeof(unsigned) + maxelements * (sizeof(IdxType) + sizeof(ValType));
  //}
  handle->tmpbuf=malloc(2*maxbytes);

  //struct stream<IdxType, ValType> *tmp1 = (struct stream<IdxType, ValType> *)handle->tmpbuf;
  //struct stream<IdxType, ValType> *tmp2 = (struct stream<IdxType, ValType> *)(((char *)handle->tmpbuf) + maxbytes);
  struct stream *tmp1 = (struct stream *)handle->tmpbuf;
  struct stream *tmp2 = (struct stream *)(((char *)handle->tmpbuf) + maxbytes);

  schedule = (NBC_Schedule*)malloc(sizeof(NBC_Schedule));
  if (NULL == schedule) { printf("Error in malloc()\n"); return res; }
  res = NBC_Sched_create(schedule);
  if(res != NBC_OK) { printf("Error in NBC_Sched_create (%i)\n", res); return res; }

  // Do the recusrive halfing
  int vrank = rank;
  unsigned pof2 = pow(2, floor(log2(worldsize)));
  int rem = worldsize - pof2;
  unsigned mask = 1;

  if (rank < 2 * rem) {
    // Collapse to power of two
    if(rank % 2 == 0) {
      NBC_Sched_send(sendbuf, false, -dim, MPI_BYTE, rank+1, schedule);
      vrank = -1;
    } else {
      NBC_Sched_recv(recvbuf, false, maxbytes, MPI_BYTE, rank-1, schedule);

      NBC_Sched_barrier(schedule);

      // Sum sparse items
      NBC_Sched_op(tmp1, false, sendbuf, false, recvbuf, false, -dim, mpiT, MPI_SUM, schedule);

      vrank = rank / 2;
    }
  } else {
    vrank = rank - rem;

    //size_t cnt = countBytes(sendbuf, dim);
    size_t cnt = countBytes<IdxType, ValType>(sendbuf, dim);
    NBC_Sched_copy(sendbuf, false, cnt, MPI_BYTE, tmp1, false, cnt, MPI_BYTE, schedule);
  }

  NBC_Sched_barrier(schedule);

  // TODO If T == 0

  const unsigned divFactor = 3;
  const unsigned sizeDiff = (sizeof(IdxType) + sizeof(ValType)) / sizeof(ValType);
  unsigned T = floor(log2(dim / (sizeDiff * k * divFactor)));
  if(T < 1) T = 1;

  unsigned t = 0;
  bool firstBuf = true;

  // Recursive doubling
  while(mask < pof2 && t < T) {
    if(vrank != -1) {
      int dest = vrank ^ mask; // bitwise xor
      if(dest < rem) {
        dest = dest*2 + 1;
      } else {
        dest = dest + rem;
      }

      if(firstBuf) {
        NBC_Sched_send(tmp1, false, -dim, MPI_BYTE, dest, schedule);
        NBC_Sched_recv(recvbuf, false, maxbytes, MPI_BYTE, dest, schedule);
      } else {
        NBC_Sched_send(tmp2, false, -dim, MPI_BYTE, dest, schedule);
        NBC_Sched_recv(recvbuf, false, maxbytes, MPI_BYTE, dest, schedule);
      }
    }

    NBC_Sched_barrier(schedule);

    if(vrank != -1) {
      // Sum sparse items
      if(firstBuf) {
        NBC_Sched_op2(tmp2, false, tmp1, false, recvbuf, false, -dim, mpiT, MPI_SUM, schedule, (t == T-1 && (mask << 1) < pof2) ? 1 : 0);
      } else {
        NBC_Sched_op2(tmp1, false, tmp2, false, recvbuf, false, -dim, mpiT, MPI_SUM, schedule, (t == T-1 && (mask << 1) < pof2) ? 1 : 0);
      }
    }

    firstBuf = !firstBuf;
    mask = mask << 1;
    t++;
  }

  //struct stream<IdxType, ValType> *buf = firstBuf ? tmp1 : tmp2;
  struct stream *buf = firstBuf ? tmp1 : tmp2;

  if(mask < pof2) {

    unsigned nel = pof2/mask;
    int disps[nel];

    unsigned frac = dim / nel;
    unsigned lastFrac =  dim - frac*(nel-1);


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
      if(vrank != -1) {
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
        NBC_Sched_send(buf->items + (disps[send_idx] * sizeof(ValType)), false, send_cnt * sizeof(ValType), MPI_BYTE, dest, schedule);
        NBC_Sched_recv(recvbuf->items + (disps[recv_idx] * sizeof(ValType)), false, recv_cnt * sizeof(ValType), MPI_BYTE, dest, schedule);
      }

      NBC_Sched_barrier(schedule);

      if(vrank != -1) {
        NBC_Sched_op(buf->items + (disps[recv_idx] * sizeof(ValType)), false, buf->items + (disps[recv_idx] * sizeof(ValType)), false, recvbuf->items + (disps[recv_idx] * sizeof(ValType)), false, recv_cnt, mpiT, MPI_SUM, schedule);
        //// Dense sumation
        //for(size_t i = disps[recv_idx]; i < disps[recv_idx] + recv_cnt; ++i) {
        //  ((ValType *)buf->items)[i] += ((ValType *)recvbuf->items)[i];
        //}

        send_idx = recv_idx;
      }
      // update mask2
      mask2 = mask2 >> 1;
    }

    // Gather all the data by recursive doubling
    send_idx = vrank / mask;
    recv_idx = 0;
    mask2 = mask;
    while(mask2 < pof2) {
      int vdest = 0;
      if(vrank != -1) {
        vdest = vrank ^ mask2; // bitwise xor
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

        NBC_Sched_send(buf->items + (disps[send_idx] * sizeof(ValType)), false, send_cnt * sizeof(ValType), MPI_BYTE, dest, schedule);
        NBC_Sched_recv(buf->items + (disps[recv_idx] * sizeof(ValType)), false, recv_cnt * sizeof(ValType), MPI_BYTE, dest, schedule);
      }

      NBC_Sched_barrier(schedule);

      if(vrank != -1) {
        // update send_idx and recv_idx
        if(vrank > vdest) {
          send_idx = recv_idx;
        }
      }

      mask2 = mask2 << 1;
    }
  }

  if(vrank != -1) {
    // Copy into recv buffer
    NBC_Sched_copy(buf, false, -dim, MPI_BYTE, recvbuf, false, -dim, MPI_BYTE, schedule);
  }

  if (rank < 2*rem) {
    // Expand from power of two
    if(rank % 2 > 0) {
      NBC_Sched_send(recvbuf, false, -dim, MPI_BYTE, rank-1, schedule);
    } else {
      NBC_Sched_recv(recvbuf, false, maxbytes, MPI_BYTE, rank+1, schedule);
    }
  }

  NBC_Sched_barrier(schedule);

  res = NBC_Sched_commit(schedule);
  if (NBC_OK != res) { printf("Error in NBC_Sched_commit() (%i)\n", res); return res; }

  res = NBC_Start(handle, schedule);
  if (NBC_OK != res) { printf("Error in NBC_Start() (%i)\n", res); return res; }

  return NBC_OK;
}
