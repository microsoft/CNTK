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
#ifndef __NBC_INTERNAL_H__
#define __NBC_INTERNAL_H__
#include "config.h"

/*********************** LibNBC tuning parameters ************************/

/* the debug level */
#define NBC_DLEVEL 0

/* use PMPI calls to MPI backend - this is needed because otherwise th
 * output will be screwed up in the profiler ... but this can be disabled
 * if the profiler does not profile NBC_ calls :) */
#define USE_PMPI 1

/* enable schedule caching - undef NBC_CACHE_SCHEDULE to deactivate it */
/* TODO: this whole schedule cache stuff does not work with the tmbuf
 * :-( - first, the tmpbuf must not be freed if a schedule using it is
 * still in the cache and second, the tmpbuf used by the schedule must
 * be attached to the handle that uses this schedule !!!! 
 * I.E., THIS IS EXPERIMENTAL AND MIGHT NOT WORK */
#define NBC_CACHE_SCHEDULE 
#define NBC_SCHED_DICT_UPPER 1024 /* max. number of dict entries */
#define NBC_SCHED_DICT_LOWER 512  /* nuber of dict entries after wipe, if SCHED_DICT_UPPER is reached */

/********************* end of LibNBC tuning parameters ************************/

/* correct fortran bindings */
#define NBC_F77_FUNC_ F77_FUNC_


#include "nbc.h"

#ifdef HAVE_OFED
#include "of/of.h"
#undef HAVE_MPI
#endif

#ifdef HAVE_OMPI
#include "nbc_ompi_include.h"
#else
#include <mpi.h>
#endif

#ifdef HAVE_PROGRESS_THREAD
#include <semaphore.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "libdict/dict.h"

#ifdef __cplusplus
extern "C" {
#endif

/* all MPI functions used in LibNBC have to be defined here */
#ifdef USE_PMPI
#define MPI_Irecv PMPI_Irecv
#define MPI_Waitany PMPI_Waitany
#define MPI_Iprobe PMPI_Iprobe
#define MPI_Wtime PMPI_Wtime
#define MPI_Testall PMPI_Testall
#define MPI_Isend PMPI_Isend
#define MPI_Keyval_create PMPI_Keyval_create
#define MPI_Attr_get PMPI_Attr_get
#define MPI_Comm_dup PMPI_Comm_dup
#define MPI_Attr_put PMPI_Attr_put
#define MPI_Send PMPI_Send
#define MPI_Comm_size PMPI_Comm_size
#define MPI_Comm_rank PMPI_Comm_rank
#define MPI_Type_extent PMPI_Type_extent
#define MPI_Type_size PMPI_Type_size
#define MPI_Pack_size PMPI_Pack_size
#define MPI_Pack PMPI_Pack
#define MPI_Unpack PMPI_Unpack
//#define MPI_Comm_f2c PMPI_Comm_f2c
//#define MPI_Type_f2c PMPI_Type_f2c
//#define MPI_Op_f2c PMPI_Op_f2c
#define MPI_Cart_shift PMPI_Cart_shift
#define MPI_Cartdim_get PMPI_Cartdim_get
#define MPI_Graph_neighbors PMPI_Graph_neighbors
#define MPI_Graph_neighbors_count PMPI_Graph_neighbors_count
#define MPI_Topo_test PMPI_Topo_test
#endif

/* if we use MPI-1, MPI_IN_PLACE is not defined :-( */
#ifndef MPI_IN_PLACE
#define MPI_IN_PLACE (void*)1
#endif

/* restore inline behavior for non-gcc compilers */
#ifndef __GNUC__
#define __inline__ inline
#endif

/* log(2) */
#define LOG2 0.69314718055994530941

/* true/false */
#define true 1
#define false 0

/* all collectives */
#define NBC_ALLGATHER 0
#define NBC_ALLGATHERV 1
#define NBC_ALLREDUCE 2
#define NBC_ALLTOALL 3
#define NBC_ALLTOALLV 4
#define NBC_ALLTOALLW 5
#define NBC_BARRIER 6
#define NBC_BCAST 7
#define NBC_EXSCAN 8
#define NBC_GATHER 9
#define NBC_GATHERV 10
#define NBC_REDUCE 11
#define NBC_REDUCESCAT 12
#define NBC_SCAN 13
#define NBC_SCATTER 14
#define NBC_SCATTERV 15
#define NBC_CART_SHIFT_XCHG 16
#define NBC_NEIGHBOR_XCHG 17
/* set the number of collectives in nbc.h !!!! */
  
/* several typedefs for NBC */

/* the function type enum */
typedef enum {
  SEND,
  RECV,
  OP,
  COPY,
  UNPACK
} NBC_Fn_type;

/* the send argument struct */
typedef struct {
  void *buf;
  char tmpbuf;
  int count;
  MPI_Datatype datatype;
  int dest;
} NBC_Args_send;

/* the receive argument struct */
typedef struct {
  void *buf;
  char tmpbuf;
  int count;
  MPI_Datatype datatype;
  int source;
} NBC_Args_recv;

/* the operation argument struct */
typedef struct {
  void *buf1;
  char tmpbuf1;
  void *buf2;
  char tmpbuf2;
  void *buf3;
  char tmpbuf3;
  int count;
  MPI_Op op;
  MPI_Datatype datatype;
  int forceDense;
} NBC_Args_op;

/* the copy argument struct */
typedef struct {
  void *src; 
  char tmpsrc;
  int srccount;
  MPI_Datatype srctype;
  void *tgt;
  char tmptgt;
  int tgtcount;
  MPI_Datatype tgttype;
} NBC_Args_copy;

/* unpack operation arguments */
typedef struct {
  void *inbuf; 
  char tmpinbuf;
  int count;
  MPI_Datatype datatype;
  void *outbuf; 
  char tmpoutbuf;
} NBC_Args_unpack;

/* internal function prototypes */
int NBC_Sched_create(NBC_Schedule* schedule);
int NBC_Sched_send(void* buf, char tmpbuf, int count, MPI_Datatype datatype, int dest, NBC_Schedule *schedule);
int NBC_Sched_recv(void* buf, char tmpbuf, int count, MPI_Datatype datatype, int source, NBC_Schedule *schedule);
int NBC_Sched_op2(void* buf3, char tmpbuf3, void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count, MPI_Datatype datatype, MPI_Op op, NBC_Schedule *schedule, int forceDense);
int NBC_Sched_op(void* buf3, char tmpbuf3, void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count, MPI_Datatype datatype, MPI_Op op, NBC_Schedule *schedule);
int NBC_Sched_copy(void *src, char tmpsrc, int srccount, MPI_Datatype srctype, void *tgt, char tmptgt, int tgtcount, MPI_Datatype tgttype, NBC_Schedule *schedule);
int NBC_Sched_unpack(void *inbuf, char tmpinbuf, int count, MPI_Datatype datatype, void *outbuf, char tmpoutbuf, NBC_Schedule *schedule);
int NBC_Sched_barrier(NBC_Schedule *schedule);
int NBC_Sched_commit(NBC_Schedule *schedule);

#ifdef NBC_CACHE_SCHEDULE
/* this is a dummy structure which is used to get the schedule out of
 * the collop sepcific structure. The schedule pointer HAS to be at the
 * first position and should NOT BE REORDERED by the compiler (C
 * guarantees that */
struct NBC_dummyarg {
  NBC_Schedule *schedule;
};

typedef struct {
  NBC_Schedule *schedule;
  void *sendbuf;
  int sendcount;
  MPI_Datatype sendtype;
  void* recvbuf;
  int recvcount;
  MPI_Datatype recvtype;
} NBC_Alltoall_args;
int NBC_Alltoall_args_compare(NBC_Alltoall_args *a, NBC_Alltoall_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *sendbuf;
  int sendcount;
  MPI_Datatype sendtype;
  void* recvbuf;
  int recvcount;
  MPI_Datatype recvtype;
} NBC_Allgather_args;
int NBC_Allgather_args_compare(NBC_Allgather_args *a, NBC_Allgather_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *sendbuf;
  void* recvbuf;
  int count;
  MPI_Datatype datatype;
  MPI_Op op;
} NBC_Allreduce_args;
int NBC_Allreduce_args_compare(NBC_Allreduce_args *a, NBC_Allreduce_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *buffer;
  int count;
  MPI_Datatype datatype;
  int root;
} NBC_Bcast_args;
int NBC_Bcast_args_compare(NBC_Bcast_args *a, NBC_Bcast_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *sendbuf;
  int sendcount;
  MPI_Datatype sendtype;
  void* recvbuf;
  int recvcount;
  MPI_Datatype recvtype;
  int root;
} NBC_Gather_args;
int NBC_Gather_args_compare(NBC_Gather_args *a, NBC_Gather_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *sendbuf;
  void* recvbuf;
  int count;
  MPI_Datatype datatype;
  MPI_Op op;
  int root;
} NBC_Reduce_args;
int NBC_Reduce_args_compare(NBC_Reduce_args *a, NBC_Reduce_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *sendbuf;
  void* recvbuf;
  int count;
  MPI_Datatype datatype;
  MPI_Op op;
} NBC_Scan_args;
int NBC_Scan_args_compare(NBC_Scan_args *a, NBC_Scan_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *sendbuf;
  int sendcount;
  MPI_Datatype sendtype;
  void* recvbuf;
  int recvcount;
  MPI_Datatype recvtype;
  int root;
} NBC_Scatter_args;
int NBC_Scatter_args_compare(NBC_Scatter_args *a, NBC_Scatter_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *sbuf; 
  int scount; 
  MPI_Datatype stype;
  void *rbuf; 
  int rcount; 
  MPI_Datatype rtype; 
  int direction; 
  int disp;
  MPI_Comm comm; 
} NBC_Icart_shift_xchg_args;
int NBC_Icart_shift_xchg_args_compare(NBC_Icart_shift_xchg_args *a, NBC_Icart_shift_xchg_args *b, void *param);

typedef struct {
  NBC_Schedule *schedule;
  void *sbuf;
  int scount;
  MPI_Datatype stype;
  void *rbuf;
  int rcount;
  MPI_Datatype rtype; 
  MPI_Comm comm; 
} NBC_Ineighbor_xchg_args;
int NBC_Ineighbor_xchg_args_compare(NBC_Ineighbor_xchg_args *a, NBC_Ineighbor_xchg_args *b, void *param);

/* Schedule cache structures/functions */
u_int32_t adler32(u_int32_t adler, int8_t *buf, int len);
void NBC_SchedCache_args_delete(void *entry);
void NBC_SchedCache_args_delete_key_dummy(void *k);
  
#endif


int NBC_Progress(NBC_Handle *handle);
int NBC_Start(NBC_Handle *handle, NBC_Schedule *schedule);
int NBC_Init_handle(NBC_Handle *handle, MPI_Comm comm);
static __inline__ int NBC_Type_intrinsic(MPI_Datatype type);
static __inline__ int NBC_Copy(void *src, int srccount, MPI_Datatype srctype, void *tgt, int tgtcount, MPI_Datatype tgttype, MPI_Comm comm);
NBC_Comminfo* NBC_Init_comm(MPI_Comm comm);
int NBC_Create_fortran_handle(int *fhandle, NBC_Handle **handle);

/* some macros */

/* a schedule has the following format:
 * [schedule] ::= [size][round-schedule][delimiter][round-schedule][delimiter]...[end]
 * [size] ::= size of the schedule (int)
 * [round-schedule] ::= [num][type][type-args][type][type-args]...
 * [num] ::= number of elements in round (int)
 * [type] ::= function type (NBC_Fn_type)
 * [type-args] ::= type specific arguments (NBC_Args_send, NBC_Args_recv or, NBC_Args_op)
 * [delimiter] ::= 1 (char) - indicates that a round follows
 * [end] ::= 0 (char) - indicates that this is the last round 
 */

/* NBC_GET_ROUND_SIZE returns the size in bytes of a round of a NBC_Schedule
 * schedule. A round has the format:
 * [num]{[type][type-args]}
 * e.g. [(int)2][(NBC_Fn_type)SEND][(NBC_Args_send)SEND-ARGS][(NBC_Fn_type)RECV][(NBC_Args_recv)RECV-ARGS] */
#define NBC_GET_ROUND_SIZE(schedule, size) \
 {  \
   int *numptr; \
   NBC_Fn_type *typeptr; \
   int i;  \
     \
   numptr = (int*)schedule; \
   /*NBC_DEBUG(10, "GET_ROUND_SIZE got %i elements\n", *numptr); */\
   /* end is increased by sizeof(int) bytes to point to type */ \
   typeptr = (NBC_Fn_type*)((int*)(schedule)+1); \
   for (i=0; i<*numptr; i++) { \
     /* go sizeof op-data forward */ \
     switch(*typeptr) { \
       case SEND: \
         /*printf("found a SEND at offset %i\n", (int)typeptr-(int)schedule); */\
         typeptr = (NBC_Fn_type*)((NBC_Args_send*)typeptr+1); \
         break; \
       case RECV: \
         /*printf("found a RECV at offset %i\n", (int)typeptr-(int)schedule); */\
         typeptr = (NBC_Fn_type*)((NBC_Args_recv*)typeptr+1); \
         break; \
       case OP: \
         /*printf("found a OP at offset %i\n", (int)typeptr-(int)schedule); */\
         typeptr = (NBC_Fn_type*)((NBC_Args_op*)typeptr+1); \
         break; \
       case COPY: \
         /*printf("found a COPY at offset %i\n", (int)typeptr-(int)schedule); */\
         typeptr = (NBC_Fn_type*)((NBC_Args_copy*)typeptr+1); \
         break; \
       case UNPACK: \
         /*printf("found a UNPACK at offset %i\n", (int)typeptr-(int)schedule); */\
         typeptr = (NBC_Fn_type*)((NBC_Args_unpack*)typeptr+1); \
         break; \
       default: \
         printf("NBC_GET_ROUND_SIZE: bad type %li at offset %li\n", (long)*typeptr, (long)typeptr-(long)schedule); \
         return NBC_BAD_SCHED; \
     } \
     /* increase ptr by size of fn_type enum */ \
     typeptr = (NBC_Fn_type*)((NBC_Fn_type*)typeptr+1); \
   } \
   /* this could be optimized if typeptr would be used directly */ \
   size = (long)typeptr-(long)schedule; \
 }

/* returns the size of a schedule in bytes */
#define NBC_GET_SIZE(schedule, size) \
{ \
  size=*(int*)schedule; \
}

/* increase the size of a schedule by size bytes */
#define NBC_INC_SIZE(schedule, size) \
{ \
  *(int*)schedule+=size; \
}

/* increments the number of operations in the last round */
#define NBC_INC_NUM_ROUND(schedule) \
{ \
  int total_size; \
  long round_size; \
  char *ptr, *lastround; \
 \
  NBC_GET_SIZE(schedule, total_size); \
 \
  /* ptr begins at first round (first int is overall size) */ \
  ptr = (char*)((char*)schedule+sizeof(int)); \
  lastround = ptr; \
  while ((long)ptr-(long)schedule < total_size) { \
    NBC_GET_ROUND_SIZE(ptr, round_size); \
    /*printf("got round size %i\n", round_size);*/ \
    lastround = ptr; \
    /* add round size */ \
    ptr=ptr+round_size; \
    /* add sizeof(char) as barrier delimiter */ \
    ptr=ptr+sizeof(char); \
    /*printf("(int)ptr-(int)schedule=%i, size=%i\n", (int)ptr-(int)schedule, size); */\
  } \
  /*printf("lastround count is at offset: %i\n", (int)lastround-(int)schedule);*/ \
  /* this is the count in the last round of the schedule */ \
  (*(int*)lastround)++; \
}

/* NBC_PRINT_ROUND prints a round in a schedule. A round has the format:
 * [num]{[op][op-data]} types: [int]{[enum][op-type]}
 * e.g. [2][SEND][SEND-ARGS][RECV][RECV-ARGS] */
#define NBC_PRINT_ROUND(schedule) \
 {  \
   int myrank, *numptr; \
   NBC_Fn_type *typeptr; \
   NBC_Args_send *sendargs; \
   NBC_Args_recv *recvargs; \
   NBC_Args_op *opargs; \
   NBC_Args_copy *copyargs; \
   NBC_Args_unpack *unpackargs; \
   int i;  \
     \
   numptr = (int*)schedule; \
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank); \
   printf("has %i actions: \n", *numptr); \
   /* end is increased by sizeof(int) bytes to point to type */ \
   typeptr = (NBC_Fn_type*)((int*)(schedule)+1); \
   for (i=0; i<*numptr; i++) { \
     /* go sizeof op-data forward */ \
     switch(*typeptr) { \
       case SEND: \
         printf("[%i]  SEND (offset %li) ", myrank, (long)typeptr-(long)schedule); \
         sendargs = (NBC_Args_send*)(typeptr+1); \
         printf("*buf: %lu, count: %i, type: %lu, dest: %i)\n", (unsigned long)sendargs->buf, sendargs->count, (unsigned long)sendargs->datatype, sendargs->dest); \
         typeptr = (NBC_Fn_type*)((NBC_Args_send*)typeptr+1); \
         break; \
       case RECV: \
         printf("[%i]  RECV (offset %li) ", myrank, (long)typeptr-(long)schedule); \
         recvargs = (NBC_Args_recv*)(typeptr+1); \
         printf("*buf: %lu, count: %i, type: %lu, source: %i)\n", (unsigned long)recvargs->buf, recvargs->count, (unsigned long)recvargs->datatype, recvargs->source); \
         typeptr = (NBC_Fn_type*)((NBC_Args_recv*)typeptr+1); \
         break; \
       case OP: \
         printf("[%i]  OP   (offset %li) ", myrank, (long)typeptr-(long)schedule); \
         opargs = (NBC_Args_op*)(typeptr+1); \
         printf("*buf1: %lu, buf2: %lu, count: %i, type: %lu)\n", (unsigned long)opargs->buf1, (unsigned long)opargs->buf2, opargs->count, (unsigned long)opargs->datatype); \
         typeptr = (NBC_Fn_type*)((NBC_Args_op*)typeptr+1); \
         break; \
       case COPY: \
         printf("[%i]  COPY   (offset %li) ", myrank, (long)typeptr-(long)schedule); \
         copyargs = (NBC_Args_copy*)(typeptr+1); \
         printf("*src: %lu, srccount: %i, srctype: %lu, *tgt: %lu, tgtcount: %i, tgttype: %lu)\n", (unsigned long)copyargs->src, copyargs->srccount, (unsigned long)copyargs->srctype, (unsigned long)copyargs->tgt, copyargs->tgtcount, (unsigned long)copyargs->tgttype); \
         typeptr = (NBC_Fn_type*)((NBC_Args_copy*)typeptr+1); \
         break; \
       case UNPACK: \
         printf("[%i]  UNPACK   (offset %li) ", myrank, (long)typeptr-(long)schedule); \
         unpackargs = (NBC_Args_unpack*)(typeptr+1); \
         printf("*src: %lu, srccount: %i, srctype: %lu, *tgt: %lu\n",(unsigned long)unpackargs->inbuf, unpackargs->count, (unsigned long)unpackargs->datatype, (unsigned long)unpackargs->outbuf); \
         typeptr = (NBC_Fn_type*)((NBC_Args_unpack*)typeptr+1); \
         break; \
       default: \
         printf("[%i] NBC_PRINT_ROUND: bad type %li at offset %li\n", myrank, (long)*typeptr, (long)typeptr-(long)schedule); \
         return NBC_BAD_SCHED; \
     } \
     /* increase ptr by size of fn_type enum */ \
     typeptr = (NBC_Fn_type*)((NBC_Fn_type*)typeptr+1); \
   } \
   printf("\n"); \
 }

#define NBC_PRINT_SCHED(schedule) \
{ \
  int size, myrank; \
  long round_size; \
  char *ptr; \
 \
  NBC_GET_SIZE(schedule, size); \
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank); \
  printf("[%i] printing schedule of size %i\n", myrank, size); \
 \
  /* ptr begins at first round (first int is overall size) */ \
  ptr = (char*)((char*)schedule+sizeof(int)); \
  while ((long)ptr-(long)schedule < size) { \
    NBC_GET_ROUND_SIZE(ptr, round_size); \
    printf("[%i] Round at byte %li (size %li) ", myrank, (long)ptr-(long)schedule, round_size); \
    NBC_PRINT_ROUND(ptr); \
    /* add round size */ \
    ptr=ptr+round_size; \
    /* add sizeof(char) as barrier delimiter */ \
    ptr=ptr+sizeof(char); \
  } \
}

#define NBC_CHECK_NULL(ptr) \
{ \
  if(ptr == NULL) { \
    printf("realloc error :-(\n"); \
  } \
}



/*
#define NBC_DEBUG(level, ...) {} 
*/

static __inline__ void NBC_DEBUG(int level, const char *fmt, ...) 
{ 
#if NBC_DLEVEL > 0
  va_list ap;
  int rank; 
 
  if(NBC_DLEVEL >= level) { 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    
    printf("[LibNBC - %i] ", rank); 
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end (ap);
  } 
#endif
}

/* returns true (1) or false (0) if type is intrinsic or not */
static __inline__ int NBC_Type_intrinsic(MPI_Datatype type) {
  
  if( ( type == MPI_INT ) ||
      ( type == MPI_LONG ) ||
      ( type == MPI_SHORT ) ||
      ( type == MPI_UNSIGNED ) ||
      ( type == MPI_UNSIGNED_SHORT ) ||
      ( type == MPI_UNSIGNED_LONG ) ||
      ( type == MPI_FLOAT ) ||
      ( type == MPI_DOUBLE ) ||
      ( type == MPI_LONG_DOUBLE ) ||
      ( type == MPI_BYTE ) ||
      ( type == MPI_FLOAT_INT) ||
      ( type == MPI_DOUBLE_INT) ||
      ( type == MPI_LONG_INT) ||
      ( type == MPI_2INT) ||
      ( type == MPI_SHORT_INT) ||
      ( type == MPI_LONG_DOUBLE_INT)) 
    return 1;
  else 
    return 0;
}

/* let's give a try to inline functions */
static __inline__ int NBC_Copy(void *src, int srccount, MPI_Datatype srctype, void *tgt, int tgtcount, MPI_Datatype tgttype, MPI_Comm comm) {
  int size, pos, res;
  MPI_Aint ext;
  void *packbuf;

  if((srctype == tgttype) && NBC_Type_intrinsic(srctype)) {
    /* if we have the same types and they are contiguous (intrinsic
     * types are contiguous), we can just use a single memcpy */
    res = MPI_Type_extent(srctype, &ext);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
    memcpy(tgt, src, srccount*ext);
  } else {
    /* we have to pack and unpack */
    res = MPI_Pack_size(srccount, srctype, comm, &size);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack_size() (%i)\n", res); return res; }
    packbuf = malloc(size);
    if (NULL == packbuf) { printf("Error in malloc()\n"); return res; }
    pos=0;
    res = MPI_Pack(src, srccount, srctype, packbuf, size, &pos, comm);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack() (%i)\n", res); return res; }
    pos=0;
    res = MPI_Unpack(packbuf, size, &pos, tgt, tgtcount, tgttype, comm);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Unpack() (%i)\n", res); return res; }
    free(packbuf);
  }

  return NBC_OK;
}

static __inline__ int NBC_Unpack(void *src, int srccount, MPI_Datatype srctype, void *tgt, MPI_Comm comm) {
  int size, pos, res;
  MPI_Aint ext;

  if(NBC_Type_intrinsic(srctype)) {
    /* if we have the same types and they are contiguous (intrinsic
     * types are contiguous), we can just use a single memcpy */
    res = MPI_Type_extent(srctype, &ext);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Type_extent() (%i)\n", res); return res; }
    memcpy(tgt, src, srccount*ext);

  } else {
    /* we have to unpack */
    res = MPI_Pack_size(srccount, srctype, comm, &size);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Pack_size() (%i)\n", res); return res; }
    pos=0;
    res = MPI_Unpack(src, size, &pos, tgt, srccount, srctype, comm);
    if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Unpack() (%i)\n", res); return res; }
  }

  return NBC_OK;
}

/* deletes elements from dict until low watermark is reached */
static __inline__ void NBC_SchedCache_dictwipe(hb_tree *dict, int *size) {
  hb_itor *itor;
  
  itor = hb_itor_new(dict);
  for (; hb_itor_valid(itor) && (*size>NBC_SCHED_DICT_LOWER); hb_itor_next(itor)) {
    hb_tree_remove(dict, hb_itor_key(itor), 0);
    *size = *size-1;
  }
  hb_itor_destroy(itor);
}

#define NBC_IN_PLACE(sendbuf, recvbuf, inplace) \
{ \
  inplace = 0; \
  if(recvbuf == sendbuf) { \
    inplace = 1; \
  } else \
  if(sendbuf == MPI_IN_PLACE) { \
    sendbuf = recvbuf; \
    inplace = 1; \
  } else \
  if(recvbuf == MPI_IN_PLACE) { \
    recvbuf = sendbuf; \
    inplace = 1; \
  } \
}

#ifdef __cplusplus
}
#endif

#define NBC_F77_ALLFUNC_(lower,upper,args) \
  void upper args; \
  void P ## upper args; \
  void lower args; \
  void p ## lower args; \
  void lower ## _ args; \
  void p ## lower ## _ args; \
  void lower ## __ args; \
  void p ## lower ## __ args;
 
#endif


