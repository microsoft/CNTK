// cudalib.h -- wrappers for CUDA to hide NVidia stuff
//
// F. Seide, V-hansu

#pragma once

#include "cudabasetypes.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace msra { namespace cuda {

// helper functions
void join(); // wait until current launch or other async operation is completed

// memory allocation and copying
void *mallocbytes(size_t nelem, size_t sz);
void freebytes(void *p);
template <typename T>
cuda_ptr<T> malloc(size_t nelem)
{
    return (T *) mallocbytes(nelem, sizeof(T));
}
template <typename T>
void free(cuda_ptr<T> p)
{
    if (p.get() != NULL)
        freebytes((void *) p.get());
}

void memcpyh2d(void *dst, size_t byteoffset, const void *src, size_t nbytes);
void memcpyd2h(void *dst, const void *src, size_t byteoffset, size_t nbytes);
template <typename T>
void memcpy(cuda_ptr<T> dst, size_t dstoffset, const T *src, size_t nelem)
{
    memcpyh2d((void *) dst.get(), dstoffset * sizeof(T), (const void *) src, nelem * sizeof(T));
}
template <typename T>
void memcpy(T *dst, const cuda_ptr<T> src, size_t srcoffset, size_t nelem)
{
    memcpyd2h((void *) dst, (const void *) src.get(), srcoffset * sizeof(T), nelem * sizeof(T));
}
// [v-hansu] for debug use, change false to true to activate
template <typename T>
void peek(vectorref<T> v)
{
    bool dopeek = false;
    if (dopeek)
    {
        std::vector<T> vp(v.size());
        memcpy(&vp[0], v.get(), 0, v.size());
        T vp0 = vp[0]; // for reference
        vp[0] = vp0;
    }
}
template <typename T>
void peek(matrixref<T> v)
{
    bool dopeek = false;
    if (dopeek)
    {
        std::vector<T> vp(v.cols() * v.rows());
        memcpy(&vp[0], v.get(), 0, v.cols() * v.rows());
        T vp0 = vp[0]; // for reference
        vp[0] = vp0;
    }
}
};
};
