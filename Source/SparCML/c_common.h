#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

template<class IdxType, class ValType> struct s_item {
  IdxType idx;
  ValType val;
};

struct stream {
  unsigned nofitems;
#if defined(_MSC_VER)
  char items[1];
#else
  char items[];
#endif
};

template<class IdxType, class ValType>
size_t countBytes(const struct stream *s, unsigned dim) {
  if(s->nofitems == dim) {
    return sizeof(unsigned) + s->nofitems * sizeof(ValType);
  }
  return sizeof(unsigned) + (s->nofitems * (sizeof(IdxType) + sizeof(ValType)));
}

template<class IdxType, class ValType> void split_stream(const struct stream *stream, struct stream **results, unsigned dim, unsigned worldsize) {
    unsigned step = dim / worldsize;
    split_stream<IdxType, ValType>(stream, results, dim, worldsize, step);
}

template<class IdxType, class ValType> void split_stream(const struct stream *stream, struct stream **results, unsigned dim, unsigned worldsize, unsigned step) {

  assert(stream->nofitems <= dim); // Not dense
  if(stream->nofitems == dim) {
     const ValType *values = (const ValType *)stream->items;

    unsigned lastsize = dim - ((worldsize-1) * step);

    unsigned idx = 0;
    for(size_t i = 0; i < worldsize; ++i) {
      unsigned maxsize = step; 
      if(i == worldsize - 1) {
        maxsize = lastsize;
      } 

      results[i]->nofitems = maxsize;
      for(size_t j = 0; j < maxsize; ++j) {
        ((ValType *)results[i]->items)[j] = values[idx];
        idx++;
      }
    } 
  } else {
    const struct s_item<IdxType, ValType> *values = (const struct s_item<IdxType, ValType> *)stream->items;

    unsigned lastsize = dim - ((worldsize-1) * step);

    unsigned idx = 0;
    for(size_t i = 0; i < worldsize; ++i) {
      unsigned offset = step*i;
      unsigned maxsize = step; 
      if(i == worldsize - 1) {
        maxsize = lastsize;
      } 
      //size_t maxbytes = sizeof(unsigned) + maxsize * sizeof(ValType);
      //results[i] = (struct stream *) malloc(maxbytes);

      // Count the number of elemtns in split
      unsigned tmp_cnt = 0;
      unsigned tmp_idx = idx;
      while(tmp_idx < stream->nofitems && values[tmp_idx].idx < offset + maxsize) {
        tmp_cnt++;
        tmp_idx++;
      }

      // Check if dense or sparse
      bool dense = tmp_cnt * (sizeof(IdxType) + sizeof(ValType)) >= maxsize * sizeof(ValType);
      results[i]->nofitems = dense ? maxsize : tmp_cnt;

      if(dense) {
        for(size_t j = 0; j < maxsize; ++j) {
          if(idx < stream->nofitems && values[idx].idx == offset+j) {
            ((ValType *)results[i]->items)[j] = values[idx].val;
            idx++;
          } else {
            ((ValType *)results[i]->items)[j] = 0;
          }
        }
      } else {
        unsigned cnt = 0;
        // Copy values
        while(idx < stream->nofitems && values[idx].idx < offset + maxsize) {
          ((struct s_item<IdxType, ValType> *)results[i]->items)[cnt].idx = values[idx].idx - offset;
          ((struct s_item<IdxType, ValType> *)results[i]->items)[cnt].val = values[idx].val;
          cnt++;
          idx++;
        }
      }
    } 
  }
}

template<class IdxType, class ValType>
#if defined(_MSC_VER) 
stream * sum_into_first_stream(struct stream *first_s, struct stream *second_s, struct stream *tmpbuf, unsigned dim) {
#else
struct stream * sum_into_first_stream(struct stream *first_s, struct stream *second_s, struct stream *tmpbuf, unsigned dim) {
#endif
  unsigned p1 = 0;
  unsigned p2 = 0;

  unsigned len_first = first_s->nofitems;
  unsigned len_second = second_s->nofitems;

  if(len_first == dim && len_second == dim) {
    // Sum second into first return first
    ValType *first = (ValType *)first_s->items;
    const ValType * const __restrict__ second = (const ValType *)second_s->items;

#if defined(_MSC_VER)
#pragma omp parallel
#else
#pragma omp simd 
#endif
    for(size_t i = 0; i < dim; ++i) {
      first[i] += second[i];
    }
    return first_s;
  }

  if(len_first == dim) {
    // Sum second into first return first
    ValType * first = (ValType *)first_s->items;
    const struct s_item<IdxType, ValType> *second = (const struct s_item<IdxType, ValType> *)second_s->items;

    for(size_t i = 0; i < len_second; ++i) {
      first[second[i].idx] += second[i].val;
    }
    return first_s;
  }

  if(len_second == dim) {
    // Sum first into seconnd return second
    tmpbuf->nofitems = dim;
    ValType *result = (ValType *)tmpbuf->items;
    ValType *second = (ValType *)second_s->items;

    memcpy(result, second, countBytes<IdxType, ValType>(second_s, dim));

    const struct s_item<IdxType, ValType> *first = (const struct s_item<IdxType, ValType> *)first_s->items;

    for(size_t i = 0; i < len_first; ++i) {
      result[first[i].idx] += first[i].val;
    }
    return tmpbuf;
  }

  // add first sparse and second sparse
  struct s_item<IdxType, ValType> *first = (struct s_item<IdxType, ValType> *)first_s->items;
  struct s_item<IdxType, ValType> *second = (struct s_item<IdxType, ValType> *)second_s->items;

  if((len_first + len_second) * (sizeof(IdxType) + sizeof(ValType)) >= dim * sizeof(ValType)) {
    // Make dense in temp buf and return that

    tmpbuf->nofitems = dim;
    ValType * const __restrict__ result = (ValType *)tmpbuf->items;

#if defined(_MSC_VER)
#pragma omp parallel
#else
#pragma omp simd 
#endif
    for(size_t i = 0; i < dim; ++i) {
      result[i] = 0.0;
    }

    // Sum sparse vector
    while(p1 < len_first || p2 < len_second) {
      if((p1 == len_first) || (p2 != len_second && (second[p2].idx < first[p1].idx))) {
        result[second[p2].idx] = second[p2].val;
        p2++;
      } else if((p2 == len_second) || (first[p1].idx < second[p2].idx)) {
        result[first[p1].idx] = first[p1].val;
        p1++;
      } else {
        // index of receiver as index of sender must be equal
        result[first[p1].idx] = first[p1].val + second[p2].val;
        p1++;
        p2++;
      }
    }

    return tmpbuf;
  }

  if(len_first > 0 && (len_second == 0 || first[0].idx < second[0].idx)) {
    // Mem copy second at the end of first and return first
    memcpy(first + len_first, second, countBytes<IdxType, ValType>(second_s, dim));
    first_s->nofitems += second_s->nofitems;
    return first_s;
  }

  struct s_item<IdxType, ValType>* result = (struct s_item<IdxType, ValType> *)tmpbuf->items;
  // Mem copy first at the end of second and return second
  memcpy(result, second, countBytes<IdxType, ValType>(second_s, dim));
  memcpy(result + len_second, first, countBytes<IdxType, ValType>(first_s, dim));
  tmpbuf->nofitems = second_s->nofitems + first_s->nofitems;
  return tmpbuf;
}

template<class IdxType, class ValType>
#if defined(_MSC_VER) 
stream * sum_into_stream(struct stream *first_s, struct stream *second_s, struct stream *tmpbuf, unsigned dim, bool disjoint) {
#else
struct stream * sum_into_stream(struct stream *first_s, struct stream *second_s, struct stream *tmpbuf, unsigned dim, bool disjoint) {
#endif
  unsigned p1 = 0;
  unsigned p2 = 0;

  unsigned len_first = first_s->nofitems;
  unsigned len_second = second_s->nofitems;

  if(len_first == dim && len_second == dim) {
    // Sum second into first return first
    ValType *first = (ValType *)first_s->items;
    const ValType * const __restrict__ second = (const ValType *)second_s->items;

#if defined(_MSC_VER)
#pragma omp parallel
#else
#pragma omp simd 
#endif
    for(size_t i = 0; i < dim; ++i) {
      first[i] += second[i];
    }
    return first_s;
  }

  if(len_first == dim) {
    // Sum second into first return first
    ValType * first = (ValType *)first_s->items;
    const struct s_item<IdxType, ValType> *second = (const struct s_item<IdxType, ValType> *)second_s->items;

    for(size_t i = 0; i < len_second; ++i) {
      first[second[i].idx] += second[i].val;
    }
    return first_s;
  }

  if(len_second == dim) {
    // Sum first into seconnd return second
    const struct s_item<IdxType, ValType> *first = (const struct s_item<IdxType, ValType> *)first_s->items;
    ValType *second = (ValType *)second_s->items;

    for(size_t i = 0; i < len_first; ++i) {
      second[first[i].idx] += first[i].val;
    }
    return second_s;
  }

  // add first sparse and second sparse
  struct s_item<IdxType, ValType> *first = (struct s_item<IdxType, ValType> *)first_s->items;
  struct s_item<IdxType, ValType> *second = (struct s_item<IdxType, ValType> *)second_s->items;

  if((len_first + len_second) * (sizeof(IdxType) + sizeof(ValType)) >= dim * sizeof(ValType)) {
    // Make dense in temp buf and return that

    tmpbuf->nofitems = dim;
    ValType * const __restrict__ result = (ValType *)tmpbuf->items;

#if defined(_MSC_VER)
#pragma omp parallel
#else
#pragma omp simd 
#endif
    for(size_t i = 0; i < dim; ++i) {
      result[i] = 0.0;
    }

    // Sum sparse vector
    while(p1 < len_first || p2 < len_second) {
      if((p1 == len_first) || (p2 != len_second && (second[p2].idx < first[p1].idx))) {
        result[second[p2].idx] = second[p2].val;
        p2++;
      } else if((p2 == len_second) || (first[p1].idx < second[p2].idx)) {
        result[first[p1].idx] = first[p1].val;
        p1++;
      } else {
        // index of receiver as index of sender must be equal
        result[first[p1].idx] = first[p1].val + second[p2].val;
        p1++;
        p2++;
      }
    }

    return tmpbuf;
  }

  if (disjoint) {
    if(len_first > 0 && (len_second == 0 || first[0].idx < second[0].idx)) {
      // Mem copy second at the end of first and return first
      memcpy(first + len_first, second, countBytes<IdxType, ValType>(second_s, dim));
      first_s->nofitems += second_s->nofitems;
      return first_s;
    }

    // Mem copy first at the end of second and return second
    memcpy(second + len_second, first, countBytes<IdxType, ValType>(first_s, dim));
    second_s->nofitems += first_s->nofitems;
    return second_s;
  }

  // Result will be sparse
  int newLen = 0;
  struct s_item<IdxType, ValType> *result = (struct s_item<IdxType, ValType> *)tmpbuf->items;

  // Sum sparse vector
  while(p1 < len_first || p2 < len_second) {
    if((p1 == len_first) || (p2 != len_second && (second[p2].idx < first[p1].idx))) {
      result[newLen].idx = second[p2].idx;
      result[newLen].val = second[p2].val;
      p2++;
    } else if((p2 == len_second) || (first[p1].idx < second[p2].idx)) {
      result[newLen].idx = first[p1].idx;
      result[newLen].val = first[p1].val;
      p1++;
    } else {
      // index of receiver as index of sender must be equal
      result[newLen].idx = first[p1].idx;
      result[newLen].val = first[p1].val + second[p2].val;
      p1++;
      p2++;
    }
    newLen++;
  }

  tmpbuf->nofitems = newLen;
  return tmpbuf;


}

template<class IdxType, class ValType> void sum_streams(const struct stream *first_s, const struct stream *second_s, struct stream *result_s, unsigned dim) {
  sum_streams<IdxType, ValType>(first_s, second_s, result_s, dim, false);
}

template<class IdxType, class ValType> void sum_streams(const struct stream *first_s, const struct stream *second_s, struct stream *result_s, unsigned dim, bool forceDense /* = False */) {

  unsigned p1 = 0;
  unsigned p2 = 0;

  unsigned len_first = first_s->nofitems;
  unsigned len_second = second_s->nofitems;
  if(forceDense || (len_first + len_second) * (sizeof(IdxType) + sizeof(ValType)) >= dim * sizeof(ValType)) {
    // Result has to be dense
    result_s->nofitems = dim;
    ValType * const __restrict__ result = (ValType *)result_s->items;

    if(len_first == dim && len_second == dim) {
      // add both dense
      const ValType * const __restrict__ first = (const ValType *)first_s->items;
      const ValType * const __restrict__ second = (const ValType *)second_s->items;

#if defined(_MSC_VER)
#pragma omp parallel
#else
#pragma omp simd 
#endif
      for(size_t i = 0; i < dim; ++i) {
        result[i] = first[i] + second[i];
      }
    } else if(len_first == dim) {
      // add first dense and second sparse
      const ValType *first = (const ValType *)first_s->items;
      const struct s_item<IdxType, ValType> *second = (const struct s_item<IdxType, ValType> *)second_s->items;

      for(size_t i = 0; i < dim; ++i) {
        if(p2 < len_second && second[p2].idx == i) {
          result[i] = first[i] + second[p2].val;
          p2++;
        } else {
          result[i] = first[i];
        }
      }

    } else if(len_second == dim) {
      // add first sparse and second dense
      const struct s_item<IdxType, ValType> *first = (const struct s_item<IdxType, ValType> *)first_s->items;
      const ValType *second = (const ValType *)second_s->items;

      for(size_t i = 0; i < dim; ++i) {
        if(p1 < len_first && first[p1].idx == i) {
          result[i] = first[p1].val + second[i];
          p1++;
        } else {
          result[i] = second[i];
        }
      }
    } else {
      // add first sparse and second sparse
      const struct s_item<IdxType, ValType> *first = (const struct s_item<IdxType, ValType> *)first_s->items;
      const struct s_item<IdxType, ValType> *second = (const struct s_item<IdxType, ValType> *)second_s->items;

#if defined(_MSC_VER)
#pragma omp parallel
#else
#pragma omp simd 
#endif
      for(size_t i = 0; i < dim; ++i) {
        result[i] = 0.0;
      }

      // Sum sparse vector
      while(p1 < len_first || p2 < len_second) {
        if((p1 == len_first) || (p2 != len_second && (second[p2].idx < first[p1].idx))) {
          result[second[p2].idx] = second[p2].val;
          p2++;
        } else if((p2 == len_second) || (first[p1].idx < second[p2].idx)) {
          result[first[p1].idx] = first[p1].val;
          p1++;
        } else {
          // index of receiver as index of sender must be equal
          result[first[p1].idx] = first[p1].val + second[p2].val;
          p1++;
          p2++;
        }
      }
    }
  } else {
    // Result will be sparse
    int newLen = 0;
    const struct s_item<IdxType, ValType> *first = (const struct s_item<IdxType, ValType> *)first_s->items;
    const struct s_item<IdxType, ValType> *second = (const struct s_item<IdxType, ValType> *)second_s->items;
    struct s_item<IdxType, ValType> *result = (struct s_item<IdxType, ValType> *)result_s->items;

    // Sum sparse vector
    while(p1 < len_first || p2 < len_second) {
      if((p1 == len_first) || (p2 != len_second && (second[p2].idx < first[p1].idx))) {
        result[newLen].idx = second[p2].idx;
        result[newLen].val = second[p2].val;
        p2++;
      } else if((p2 == len_second) || (first[p1].idx < second[p2].idx)) {
        result[newLen].idx = first[p1].idx;
        result[newLen].val = first[p1].val;
        p1++;
      } else {
        // index of receiver as index of sender must be equal
        result[newLen].idx = first[p1].idx;
        result[newLen].val = first[p1].val + second[p2].val;
        p1++;
        p2++;
      }
      newLen++;
    }

    result_s->nofitems = newLen;
  }
}

template<class IdxType, class ValType>
void printStream(const struct stream *s, unsigned dim, int rank) {
  int sz = 100 + (s->nofitems * 20);
#if defined(_MSC_VER)
  // Original code: error : expression must have a constant value
  // char str[sz];
  char* str = (char*)malloc(sz * sizeof(char));
#else
  char str[sz];
#endif
  sprintf(str, "[RANK: %d]: Size: %d, Type: %s\n", rank, s->nofitems, s->nofitems == dim ? "DENSE" : "SPARSE");
  if(s->nofitems == dim) {
    for(unsigned i = 0; i < dim; ++i) {
      sprintf(str+strlen(str), "\t%d = %d\n", i, ((ValType *)s->items)[i]);
    }
  } else {
    struct s_item<IdxType, ValType> * items = (struct s_item<IdxType, ValType>*)s->items;
    for(unsigned i = 0; i < s->nofitems; ++i) {
      sprintf(str+strlen(str), "\t%d = %d\n", items[i].idx, items[i].val);
    }
  }
  printf("%s", str);

#if defined(_MSC_VER)
  free(str);
#endif
}

template<class IdxType, class ValType>
int isDifferent(const ValType* res, const struct stream *recvbuf, unsigned dim, unsigned rank) {
    int neq = 0;
    if(recvbuf->nofitems == dim) {
      for(size_t i = 0; i < dim; ++i) {
        if(fabs(res[i] - ((ValType *)recvbuf->items)[i]) > 1e-5) {
          printf("[Rank %d] Not Equal at '%lu': %d != %d\n", rank, i, res[i], ((ValType *)recvbuf->items)[i]);
          neq = 1;
          //break;
        }
      }
    } else {
      //printf("[Rank %d] Sparse result!\n", rank);
      unsigned idx = 0;
      for(size_t i = 0; i < dim; ++i) {
        if(idx < recvbuf->nofitems && ((struct s_item<IdxType, ValType>*)recvbuf->items)[idx].idx == i) {
          if(fabs(res[i] - ((struct s_item<IdxType, ValType>*)recvbuf->items)[idx].val) > 1e-5) {
            //printf("[Rank %d] Not Equal at '%lu': %d != %d\n", rank, i, res[i], ((struct s_item<IdxType, ValType>*)recvbuf->items)[idx].val);
            neq = 1;
            break;
          }
          idx++;
        } else {
          if(fabs(res[i]) > 1e-5) {
            //printf("[Rank %d] Not Equal 0.0 at '%lu': %d != 0.0\n", rank, i, res[i]);
            neq = 1;
            break;
          }
        }
      }
    }

    return neq;
}
