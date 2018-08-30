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


#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Comm_neighbors_count=PNBC_Comm_neighbors_count
#define NBC_Comm_neighbors_count PNBC_Comm_neighbors_count
#endif

int NBC_Comm_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree, int *weighted) {
  int topo, res;

  res = MPI_Topo_test(comm, &topo);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Topo_test() (%i)\n", res); return res; }

  switch(topo) {
    case MPI_CART: /* cartesian */
      {
        int ndims;
        res = MPI_Cartdim_get(comm, &ndims)  ;
        if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Cartdim_get() (%i)\n", res); return res; }
        /* outdegree is always 2*ndims because we need to iterate over empty buffers for MPI_PROC_NULL */
        *outdegree = *indegree = 2*ndims;
        *weighted = 0; 
      }
      break;
    case MPI_GRAPH: /* graph */
      {
        int rank, nneighbors;
        MPI_Comm_rank(comm, &rank);
        res = MPI_Graph_neighbors_count(comm, rank, &nneighbors);
        if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Graph_neighbors_count() (%i)\n", res); return res; }
        *outdegree = *indegree = nneighbors;  
        *weighted = 0; 
      }
      break;
#ifdef HAVE_MPI22
    case MPI_DIST_GRAPH: /* graph */
      {
        res = MPI_Dist_graph_neighbors_count(comm, indegree, outdegree, weighted);
        if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Dist_graph_neighbors_count() (%i)\n", res); return res; }
      }
#endif
      break;
    case MPI_UNDEFINED:
      return NBC_INVALID_TOPOLOGY_COMM;
      break;
    default:
      return NBC_INVALID_PARAM;
      break;
  }
  return NBC_OK;
}

#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
#pragma weak NBC_Comm_neighbors=PNBC_Comm_neighbors
#define NBC_Comm_neighbors PNBC_Comm_neighbors
#endif

int NBC_Comm_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[]) {
  int topo, res, nneighbors;
  int index = 0;

  int indeg, outdeg, wgtd;
  res = NBC_Comm_neighbors_count(comm, &indeg, &outdeg, &wgtd);
  if(indeg > maxindegree && outdeg > maxoutdegree) return NBC_INVALID_PARAM; /* we want to return *all* neighbors */

  res = MPI_Topo_test(comm, &topo);
  if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Topo_test() (%i)\n", res); return res; }

  switch(topo) {
    case MPI_CART: /* cartesian */
      {
        int ndims, i, rpeer, speer;
        res = MPI_Cartdim_get(comm, &ndims);
        if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Cartdim_get() (%i)\n", res); return res; }

        for(i = 0; i<ndims; i++) {
          res = MPI_Cart_shift(comm, i, 1, &rpeer, &speer);
          if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Cart_shift() (%i)\n", res); return res; }
          sources[index] = destinations[index] = rpeer; index++;
          sources[index] = destinations[index] = speer; index++;
        }
      }
      break;
    case MPI_GRAPH: /* graph */
      {
        int rank;
        MPI_Comm_rank(comm, &rank);
        res = MPI_Graph_neighbors(comm, rank, maxindegree, sources);
        if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Graph_neighbors_count() (%i)\n", res); return res; }
        for(int i=0; i<maxindegree; i++) destinations[i] = sources[i];
      }
      break;
#ifdef HAVE_MPI22
    case MPI_DIST_GRAPH: /* dist graph */
      {
        res = MPI_Dist_graph_neighbors(comm, maxindegree, sources, sourceweights, maxoutdegree, destinations, destweights);
        if (MPI_SUCCESS != res) { printf("MPI Error in MPI_Graph_neighbors_count() (%i)\n", res); return res; }
      }
      break;
#endif
    case MPI_UNDEFINED:
      return NBC_INVALID_TOPOLOGY_COMM;
      break;
    default:
      return NBC_INVALID_PARAM;
      break;
  }

  return NBC_OK;
}

///* Fortran bindings */
//#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
//NBC_F77_ALLFUNC_(nbc_comm_neighbors,NBC_COMM_NEIGHBORS,(int *fcomm, int *maxindegree, int *sources, int *sourceweights, int *maxoutdegree, int *destinations, int *destweights, int *ierr));
//#pragma weak NBC_COMM_NEIGHBORS = nbc_comm_neighbors_f
//#pragma weak nbc_comm_neighbors = nbc_comm_neighbors_f
//#pragma weak nbc_comm_neighbors_ = nbc_comm_neighbors_f
//#pragma weak nbc_comm_neighbors__ = nbc_comm_neighbors_f
//#pragma weak PNBC_COMM_NEIGHBORS = nbc_comm_neighbors_f
//#pragma weak pnbc_comm_neighbors = nbc_comm_neighbors_f
//#pragma weak pnbc_comm_neighbors_ = nbc_comm_neighbors_f
//#pragma weak pnbc_comm_neighbors__ = nbc_comm_neighbors_f
//void nbc_comm_neighbors_f(int *fcomm, int *maxindegree, int *sources, int *sourceweights, int *maxoutdegree, int *destinations, int *destweights, int *ierr)
//#else
//void NBC_F77_FUNC_(nbc_comm_neighbors,NBC_COMM_NEIGHBORS)(int *fcomm, int *maxindegree, int *sources, int *sourceweights, int *maxoutdegree, int *destinations, int *destweights, int *ierr);
//void NBC_F77_FUNC_(nbc_comm_neighbors,NBC_COMM_NEIGHBORS)(int *fcomm, int *maxindegree, int *sources, int *sourceweights, int *maxoutdegree, int *destinations, int *destweights, int *ierr)
//#endif
//{
//  MPI_Comm comm;
//  comm = MPI_Comm_f2c(*fcomm);
//
//  *ierr = NBC_Comm_neighbors(comm, *maxindegree, sources, sourceweights, *maxoutdegree, destinations, destweights);
//}
//
//#ifdef HAVE_SYS_WEAK_ALIAS_PRAGMA
//NBC_F77_ALLFUNC_(nbc_comm_neighbors_count,NBC_COMM_NEIGHBORS_COUNT,(int *fcomm, int *indegree, int *outdegree, int *weighted, int *ierr));
//#pragma weak NBC_COMM_NEIGHBORS_COUNT = nbc_comm_neighbors_count_f
//#pragma weak nbc_comm_neighbors_count = nbc_comm_neighbors_count_f
//#pragma weak nbc_comm_neighbors_count_ = nbc_comm_neighbors_count_f
//#pragma weak nbc_comm_neighbors_count__ = nbc_comm_neighbors_count_f
//#pragma weak PNBC_COMM_NEIGHBORS_COUNT = nbc_comm_neighbors_count_f
//#pragma weak pnbc_comm_neighbors_count = nbc_comm_neighbors_count_f
//#pragma weak pnbc_comm_neighbors_count_ = nbc_comm_neighbors_count_f
//#pragma weak pnbc_comm_neighbors_count__ = nbc_comm_neighbors_count_f
//void nbc_comm_neighbors_count_f(int *fcomm, int *indegree, int *outdegree, int *weighted, int *ierr)
//#else
//void NBC_F77_FUNC_(nbc_comm_neighbors_count,NBC_COMM_NEIGHBORS_COUNT)(int *fcomm, int *indegree, int *outdegree, int *weighted, int *ierr);
//void NBC_F77_FUNC_(nbc_comm_neighbors_count,NBC_COMM_NEIGHBORS_COUNT)(int *fcomm, int *indegree, int *outdegree, int *weighted, int *ierr)
//#endif
//{
//  MPI_Comm comm;
//  comm = MPI_Comm_f2c(*fcomm);
//
//  *ierr = NBC_Comm_neighbors_count(comm, indegree, outdegree, weighted);
//}
