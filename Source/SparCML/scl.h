#pragma once

#include <string.h>

template<class IdxType, class ValType> struct s_item {
  IdxType idx;
  ValType val;
};

template<class IdxType, class ValType> struct stream {
  unsigned nofitems;
  char items[];
};

template<class IdxType, class ValType>
size_t countBytes(const struct stream<IdxType, ValType> *s, unsigned dim) {
  if(s->nofitems == dim) {
    return sizeof(unsigned) + s->nofitems * sizeof(ValType);
  }
  return sizeof(unsigned) + (s->nofitems * (sizeof(IdxType) + sizeof(ValType)));
}

template<class IdxType, class ValType> void sum_streams(const struct stream<IdxType, ValType> *first_s, const struct stream<IdxType, ValType> *second_s, struct stream<IdxType, ValType> *result_s, unsigned dim) {
  sum_streams(first_s, second_s, result_s, dim, false);
}
/* 
 * Function to sum uf two streams
 * Complexity: O(2*len)
 */
template<class IdxType, class ValType> void sum_streams(const struct stream<IdxType, ValType> *first_s, const struct stream<IdxType, ValType> *second_s, struct stream<IdxType, ValType> *result_s, unsigned dim, bool forceDense /* = False */) {

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

        #pragma omp simd 
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

        #pragma omp simd 
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
void printStream(const struct stream<IdxType, ValType> *s, unsigned dim, int rank) {
  int sz = 100;
  char str[sz];
  sprintf(str, "[RANK: %d]: Size: %d, Type: %s\n", rank, s->nofitems, s->nofitems == dim ? "DENSE" : "SPARSE");
  if(s->nofitems == dim) {
    for(unsigned i = 0; i < dim; ++i) {
      sprintf(str+strlen(str), "\t%d = %f\n", i, ((ValType *)s->items)[i]);
    }
  } else {
    struct s_item<IdxType, ValType> * items = (struct s_item<IdxType, ValType>*)s->items;
    for(unsigned i = 0; i < s->nofitems; ++i) {
      sprintf(str+strlen(str), "\t%d = %f\n", items[i].idx, items[i].val);
    }
  }
  printf("%s", str);
}
