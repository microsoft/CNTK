//
// <copyright file="ssematrix.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// ssematrix.h -- matrix with SSE-accelerated operations
//

#undef PRINT_MEAN_VARIANCE         // [v-hansu] check model's mean and variance

#pragma once

#include "Basics.h"
#include "Platform.h"
#include "simple_checked_arrays.h"  // ... for dotprod(); we can eliminate this I believe
#include "ssefloat4.h"
#include <stdexcept>
#ifndef __unix__
#include <ppl.h>
#include "pplhelpers.h"
#include "numahelpers.h"
#endif
#include "fileutil.h"   // for saving and reading matrices
#include <limits>       // for NaN
#include <malloc.h>

#ifdef min
#undef min      // some garbage from some Windows header that conflicts with std::min()
#endif

namespace msra { namespace math {

// ===========================================================================
// ssematrixbase -- matrix with SSE-based parallel arithmetic but no memory management
// This can be passed around for computation, but not instantiated directly.
// ===========================================================================

        // helpful macros
#undef foreach_row
#define foreach_row(_i,_m)    for (size_t _i = 0; _i < (_m).rows(); _i++)
#undef foreach_column
#define foreach_column(_j,_m) for (size_t _j = 0; _j < (_m).cols(); _j++)
#undef foreach_coord
#define foreach_coord(_i,_j,_m) for (size_t _j = 0; _j < (_m).cols(); _j++) for (size_t _i = 0; _i < (_m).rows(); _i++)

class ssematrixbase
{
    void operator= (const ssematrixbase &); ssematrixbase (const ssematrixbase &);  // base cannot be assigned
protected:
    ssematrixbase() {}  // cannot instantiate by itself, only by our derived classes
    float * p;          // data pointer (not owned by this class)
    size_t numrows;
    size_t numcols;
    size_t colstride;   // height of column (=number of rows), rounded for SSE
    size_t locate (size_t i, size_t j) const { assert (i < rows() && j < cols()); return j * colstride + i; }   // matrix in column-wise storage
    size_t locate (size_t i) const { assert (i < rows() && cols() == 1); return i; }    // column vector
    inline array_ref<float> col (size_t j) { return array_ref<float> (&p[locate(0,j)], numrows); }
    inline const_array_ref<float> col (size_t j) const { return const_array_ref<float> (&p[locate(0,j)], numrows); }
    void clear() { p = NULL; numrows = 0; numcols = 0; colstride = 0; }
    void swap (ssematrixbase & other) { std::swap (p, other.p); std::swap (numrows, other.numrows); std::swap (numcols, other.numcols); std::swap (colstride, other.colstride); }
    void move (ssematrixbase & other) { p = other.p; numrows = other.numrows; numcols = other.numcols; colstride = other.colstride; other.clear(); }

    inline const_array_ref<msra::math::float4> col4 (size_t j) const { return const_array_ref<msra::math::float4> ((const msra::math::float4*) &p[locate(0,j)], colstride/4); }
    inline msra::math::float4 & float4       (size_t i, size_t j)       { return *(msra::math::float4 *)       &p[locate(i,j)]; }
    inline const msra::math::float4 & float4 (size_t i, size_t j) const { return *(const msra::math::float4 *) &p[locate(i,j)]; }
    operator array_ref<msra::math::float4> () { return array_ref<msra::math::float4> ((msra::math::float4*) p, colstride/4 * numcols); }
    operator const_array_ref<msra::math::float4> () const { return const_array_ref<msra::math::float4> ((const msra::math::float4*) p, colstride/4 * numcols); }

    // special exception: we can instantiate from a fixed-size buffer (float[])
    template<size_t buffersize> ssematrixbase (float (&buffer)[buffersize], size_t n, size_t m)
    {
        colstride = (n + 3) & ~3;               // pad to multiples of four floats (required SSE alignment)
        const size_t totalelem = colstride * m;
        if (totalelem + 3 > _countof (buffer))  // +3 for alignment, as buffer may live on the stack and would thus be unaligned
            LogicError("ssematrixbase from vector buffer: buffer too small");
        p = &buffer[0];
        // align to 4-float boundary (required for SSE)
        // x64 stack is aligned to 16 bytes, but x86 is not. Also, float[] would not be guaranteed.
        size_t offelem = (((size_t)p) / sizeof (float)) % 4;
        if (offelem != 0)
            p += 4 - offelem;
        numrows = n; numcols = m;
    }
    // special exception: we can instantiate from a fixed-size buffer (must be SSE-aligned)
    template<class VECTOR> ssematrixbase (VECTOR & buffer, size_t n, size_t m)
    {
        p = &buffer[0];
        size_t offelem = (((size_t)p) / sizeof (float)) % 4;
        if (offelem != 0)
            LogicError("ssematrixbase from vector buffer: must be SSE-aligned");
        colstride = (n + 3) & ~3;               // pad to multiples of four floats (required SSE alignment)
        const size_t totalelem = colstride * m;
        if (totalelem != buffer.size())
            LogicError("ssematrixbase from vector buffer: incorrect buffer size");
        // align to 4-float boundary (required for SSE)
        // x64 stack is aligned to 16 bytes, but x86 is not. Also, float[] would not be guaranteed.
        numrows = n; numcols = m;
    }
public:
    typedef float elemtype;
    size_t rows() const { return numrows; }
    size_t cols() const { return numcols; }
    size_t getcolstride() const { return colstride; }               // for a friend class that we cannot declare...
    size_t size() const { assert (cols() == 1); return rows(); }    // can only ask this for a column vector
    bool empty() const { return numrows * numcols == 0; }
    void reshape(const size_t newrows, const size_t newcols) { assert (rows() * cols() == newrows * newcols); numrows=newrows; numcols = newcols;};
    float &       operator() (size_t i, size_t j)       { return p[locate(i,j)]; }
    const float & operator() (size_t i, size_t j) const { return p[locate(i,j)]; }
    // note: this can be improved by creating this as a special indexer that requires 1 column
    inline       float & operator[] (size_t i)       { return p[locate(i)]; }
    inline const float & operator[] (size_t i) const { return p[locate(i)]; }

    // assign a part of the matrix (used for parallelized data copying--our matrices can be 32 MB and more)
    void assign (const ssematrixbase & other, size_t i, size_t n)
    {
        assert (cols() == other.cols() && rows() == other.rows());
        assert (i < n);
        const size_t j0 = numcols * i / n;
        const size_t j1 = numcols * (i+1) / n;
        const size_t totalelem = colstride * (j1 - j0);
        if (totalelem > 0)
            memcpy (&(*this)(0,j0), &other(0,j0), totalelem * sizeof (*p));
    }

    // copy assignment without memory allocation (dimensions must match)
    void assign (const ssematrixbase & other)
    {
        assign (other, 0, 1);
    }


    // operations --add as we go

    //both m1 and m2 are passed in normal form (i.e., not transposed)
    void KhatriRaoProduct(const ssematrixbase & m1, const ssematrixbase & m2)
    {
        auto & us = *this;
        assert(m1.cols() == m2.cols());
        assert (us.rows() == m1.rows() * m2.rows());

        foreach_column (k, us)
        {
            size_t jj = 0;
            foreach_row (j, m2)    
            {
                foreach_row (i, m1)
                {
                    us(jj++, k) = m1(i,k) * m2(j,k);
                }
            }
        }
    }

    //   this = reshape each column of eh from (K1xK2,1) to (K1, K2) and times each column of h (K2, frames).
    //   the output is a (K1, frames) matrix
    //   eh can be transposed.
    //   used for tensor DNN
    void reshapecolumnproduct (const ssematrixbase & eh, const ssematrixbase & h, const bool isehtransposed)
    {
        auto & hnew = *this;

        if (isehtransposed)
        {
            //find nrows and ncols of the reshpaed eh
            size_t nrows = h.rows();
            size_t ncols = eh.rows() / nrows;
            assert (eh.rows() % nrows == 0);

            foreach_column(t, eh)
            {
                size_t k=0;
                for (size_t j=0; j<ncols; j++)   // row and col is transposed
                {
                    hnew(j,t) = 0.0f;
                    for (size_t i=0; i<nrows; i++)
                    {
                        hnew(j,t) += eh(k,t) * h(i,t);
                        k++;
                    }
                }
            }
        }
        else
        {
            size_t ncols = h.rows();
            size_t nrows = eh.rows() / ncols;
            assert (eh.rows() % ncols == 0);

            foreach_column(t, eh)
            {
                size_t k=0;
                for (size_t j=0; j<ncols; j++)
                {
                    for (size_t i=0; i<nrows; i++)
                    {
                        if (j == 0) 
                            hnew(i,t) = eh(k,t) * h(j,t);
                        else
                            hnew(i,t) += eh(k,t) * h(j,t);
                        k++;
                    }
                }
            }
        }
    }

    // zero the matrix
    // TODO: We should use memset(), but that only works if there are no extra rows (in a patch). Do we even allow non-stripe patches? I don't remember... CUDA lib does.
    inline void setzero() { auto & us = *this; foreach_coord (i, j, us) us(i,j) = 0.0f; }  // TODO: later use memset()

    // set zero a single column  --with memset()
    void setzero (size_t j)
    {
        auto & us = *this; 
        memset (&us(0,j), 0, sizeof (us(0,j)) * rows());
    }

    // set each element of the matrix to value
    inline void setvalue (float value) { auto & us = *this; foreach_coord (i, j, us) us(i,j) = value; }  

    // dot-product of vectors in matrix format (matrix type, but only one column)
    float dotprod (const ssematrixbase &other) const
    {
        //assert(other.cols() == 1);
        //assert(cols() == 1);
        assert(rows() == other.rows());
        assert(cols() == other.cols());
        float result = 0.0f;
        float tmpresult = 0.0f;
        for (size_t j = 0; j < cols(); j++)
        {
            dotprod(this->col(j), other.col(j), tmpresult);
            result += tmpresult;
        }
        return result;
    }

    // sets matrix to diagonal preconditioner derived from gradientsquared
    // this = (gradientsquared / nobservations + lambda)^alpha (elementwise)
    void setdiagonalpreconditioner (const ssematrixbase & gradientsquared, float nobservations, float lambda, float alpha)
    {
        auto & us = *this;
        assert (us.rows() == gradientsquared.rows());
        assert (us.cols() == gradientsquared.cols());
        foreach_coord (i, j, us)
            us(i,j) = std::pow(gradientsquared(i,j) / nobservations + lambda, alpha);
    }

    // elementwise division of a by b
    // this = a / b (elementwise)
    void elementwisedivision (const ssematrixbase &a, const ssematrixbase &b)
    {
        auto & us = *this;
        assert (us.rows() == a.rows());
        assert (us.cols() == a.cols());
        assert (us.rows() == b.rows());
        assert (us.cols() == b.cols());
        foreach_coord (i, j, us)
            us(i,j) = a(i,j) / b(i,j);
    }

    float weighteddot (const ssematrixbase & weightingmatrix, const ssematrixbase & a) const
    {
        assert(weightingmatrix.rows() == rows());
        assert(weightingmatrix.cols() == cols());
        assert(a.rows() == rows());
        assert(a.cols() == cols());
        
        float result = 0.0f;
        auto & us = *this;
        foreach_coord (i, j, us)
            result += us(i,j) * weightingmatrix(i,j) * a(i,j);
        return result;
    }

    // dot product of two vectors (which may or may not be columns matrices)
    // If 'addtoresult' then scale the result then add to it weighted, rather than overwriting it.
    static void dotprod (const_array_ref<float> a, const_array_ref<float> b, float & result)
    {
        dotprod (a, b, result, false, 0.0f, 0.0f);
    }

    static void dotprod (const_array_ref<float> a, const_array_ref<float> b, float & result,
                         bool addtoresult, const float thisscale, const float weight)
    {
        assert (a.size() == b.size());
        assert ((15 & reinterpret_cast<uintptr_t>(&a[0])) == 0); assert ((15 & reinterpret_cast<uintptr_t>(&b[0])) == 0);   // enforce SSE alignment

        size_t nlong = (a.size() + 3) / 4; // number of SSE elements
        const msra::math::float4 * pa = (const msra::math::float4 *) &a[0];
        const msra::math::float4 * pb = (const msra::math::float4 *) &b[0];

        msra::math::float4 acc = pa[0] * pb[0];
        for (size_t m = 1; m < nlong; m++)
            acc += pa[m] * pb[m];
        // final sum
        if (addtoresult)
            result = result * thisscale + weight * acc.sum();
        else 
            result = acc.sum();
    }

    // dot product of a matrix row with 4 columns at the same time
    // This useful assuming this is part of a big matrix multiplication where the
    // 'row' values are expensive to load (too big for cache) while the columns
    // are small enough to be kept in the cache. See matprod_mtm() for speed-up numbers.
    // If 'addtoresult' then scale the result then add to it weighted, rather than overwriting it.
    static void dotprod4 (const_array_ref<float> row, const_array_ref<float> cols4, size_t cols4stride,
                          array_ref<float> usij, size_t usijstride)
    {
        dotprod4 (row, cols4, cols4stride, usij, usijstride, false, 0.0f, 0.0f);
    }

    static void dotprod4 (const_array_ref<float> row, const_array_ref<float> cols4, size_t cols4stride,
                          array_ref<float> usij, size_t usijstride,
                          bool addtoresult, const float thisscale, const float weight = 1.0f)
    {
        // What this function computes is this:
        // for (size_t k = 0; k < 4; k++)
        //     dotprod (row, const_array_ref<float> (&cols4[k * cols4stride], cols4stride), usij[k * usijstride]);

        assert ((15 & reinterpret_cast<uintptr_t>(&row[0])) == 0);
        assert ((15 & reinterpret_cast<uintptr_t>(&cols4[0])) == 0);
        assert ((15 & reinterpret_cast<uintptr_t>(&cols4[cols4stride])) == 0);
        //assert (cols4stride * 4 == cols4.size());     // (passed in one vector with 4 columns stacked on top of each other)
        //assert (row.size() * 4 == cols4.size());  // this assert is no longer appropriate because of further breaking into blocks

        // perform multiple columns in parallel
        const size_t nlong = (row.size() + 3) / 4;    // number of SSE elements

        // row
        const msra::math::float4 * prow = (const msra::math::float4 *) &row[0];

        // columns
        const msra::math::float4 * pcol0 = (const msra::math::float4 *) &cols4[0 * cols4stride];
        const msra::math::float4 * pcol1 = (const msra::math::float4 *) &cols4[1 * cols4stride];
        const msra::math::float4 * pcol2 = (const msra::math::float4 *) &cols4[2 * cols4stride];
        const msra::math::float4 * pcol3 = (const msra::math::float4 *) &cols4[3 * cols4stride];

        // accumulation loop
        msra::math::float4 acc0 = prow[0] * pcol0[0];
        msra::math::float4 acc1 = prow[0] * pcol1[0];
        msra::math::float4 acc2 = prow[0] * pcol2[0];
        msra::math::float4 acc3 = prow[0] * pcol3[0];
#if 1   // prefetch is not helping
        for (size_t m = 1; m < nlong; m++)
        {
            acc0 += prow[m] * pcol0[m];
            acc1 += prow[m] * pcol1[m];
            acc2 += prow[m] * pcol2[m];
            acc3 += prow[m] * pcol3[m];
        }
#else
        const size_t prefetch = 1;//128/sizeof(acc0);
        size_t m;
        for (m = 1; m < nlong - prefetch; m++)
        {
            acc0 += prow[m] * pcol0[m];
            acc1 += prow[m] * pcol1[m];
            acc2 += prow[m] * pcol2[m];
            acc3 += prow[m] * pcol3[m];
            msra::math::float4::prefetch (&prow[m+prefetch]);
            msra::math::float4::prefetch (&pcol0[m+prefetch]);
            msra::math::float4::prefetch (&pcol1[m+prefetch]);
            msra::math::float4::prefetch (&pcol2[m+prefetch]);
            msra::math::float4::prefetch (&pcol3[m+prefetch]);
        }
        for ( ; m < nlong; m++)
        {
            acc0 += prow[m] * pcol0[m];
            acc1 += prow[m] * pcol1[m];
            acc2 += prow[m] * pcol2[m];
            acc3 += prow[m] * pcol3[m];
        }
#endif

        // final sum
        if (addtoresult)
        {
            usij[0 * usijstride] = usij[0 * usijstride] * thisscale + weight * acc0.sum();
            usij[1 * usijstride] = usij[1 * usijstride] * thisscale + weight * acc1.sum();
            usij[2 * usijstride] = usij[2 * usijstride] * thisscale + weight * acc2.sum();
            usij[3 * usijstride] = usij[3 * usijstride] * thisscale + weight * acc3.sum();
        }
        else
        {
            usij[0 * usijstride] = acc0.sum();
            usij[1 * usijstride] = acc1.sum();
            usij[2 * usijstride] = acc2.sum();
            usij[3 * usijstride] = acc3.sum();
        }
    }

    // this = M * V where M is passed as its transposed form M'

    void matprod_mtm (const ssematrixbase & Mt, const ssematrixbase & V)
    {
        matprod_mtm (Mt, 0, Mt.cols(), V);
    }

#ifdef _WIN32
    void parallel_matprod_mtm (const ssematrixbase & Mt, const ssematrixbase & V)
    {
        msra::parallel::foreach_index_block (Mt.cols(), Mt.cols(), 1, [&] (size_t i0, size_t i1)
        {
            matprod_mtm (Mt, i0, i1, V);
        });
    }
#endif

    // swap data of i-th column and j-th column
    void swapcolumn (size_t i, size_t j)
    {
        assert (i < rows() && j < cols());
        for (size_t n = 0; n < rows(); n ++)
        {
            std::swap(p[locate (n, i)], p[locate (n, j)]);
        }
    }

private:
    // guess how many colunmns of this matrix will fit into the cache
    // This is a helper function for matrix matprod and variants.
    // Result also gets aligned to 4 because matprod benefits from it.
    size_t cacheablecols() const
    {
        // cache info for 48-core Dell:
        //  - L1: 64 K per core   --we want to fit in here!
        //  - L2: 512 K per core
        //  - L3: 10 MB total

        // M             * V
        // (8192 x 9304) * (9304 x 1024) -> (8192 x 1024)   // 78047.609 MFlops, 81.773 total MB
        // 7.86 ms / frame
        // We need to store: 4 cols of V and 1 row of M, that is 9304 x 4 x 5 = 186 KB. Too much for the cache!
        // (8192 x 1024) * (1024 x 9304) -> (8192 x 9304)   // 78047.609 MFlops, 17.086 total MB
        // 1.78 ms / frame

        size_t cachesizeV = 54096;                                  // this was tuned--smaller is better (50k is quite little!!)
        const size_t colsizeV = colstride * sizeof (float);         // stored bytes per column of V
        size_t cacheablecolsV = (cachesizeV-1) / colsizeV + (1-1);  // #cols of V that fit into cache; -1 = space for row of M
        cacheablecolsV = (cacheablecolsV + 3) & ~3;                 // align (round up to multiples of 4)

        // Matrix row is used 'cacheablecolsV' times from the cache. If too small,
        // then it is also not efficient. So apply a looser upper bound.
        // Needs to be at least 4 to allow for dotprod4() optimization (4 columns of V in parallel)
        if (cacheablecolsV < 16)
            cacheablecolsV = 16;
        return cacheablecolsV;
    }
public:
    // assign a sub-rectangle from a 0-based matrix of the same size
    void assignpatch (const ssematrixbase & patch, const size_t i0, const size_t i1, const size_t j0, const size_t j1)
    {
        auto & us = *this;
        assert (i1 - i0 == patch.rows() && j1 - j0 == patch.cols());
        assert (i0 <= i1 && j0 <= j1);
        assert (i1 <= rows() && j1 <= cols());

        // copy column-wise
        for (size_t j = j0; j < j1; j++)
        {
            const float * pcol = &patch(i0-i0,j-j0);
            float *       qcol = &us(i0,j);
            const size_t colbytes = (i1-i0) * sizeof (*pcol);
            memcpy (qcol, pcol, colbytes);
        }
    }

    // this performs the operation on a row stripe, rows [beginrow,endrow) of M -> rows[beginrow,endrow) of result
    // Rows outside [beginrow,endrow) are not touched, and can e.g. be computed by another thread.
    void matprod_mtm (const ssematrixbase & Mt, size_t beginrow/*first row in M*/, size_t endrow/*end row in M*/, const ssematrixbase & V)
    {
        auto & us = *this;
        assert (V.rows() == Mt.rows());         // remember: Mt is the transpose of M
        assert (us.rows() == Mt.cols());
        assert (us.cols() == V.cols());
        assert (beginrow < endrow && endrow <= Mt.cols());    // remember that cols of Mt are the rows of M

        // overall execution of matrix product, optimized for 128 KB first-level CPU cache
        //  - loop over col stripes {j} of V, e.g. 24 (note that columns are independent)
        //    Col stripes are chosen such that row stripes of V of 1024 rows fit the cache (24x1024=96 KB)
        //    (think of this step as equivalent to actually loading the data into the cache at this point).
        //    For each col stripe {j} of V,
        //     - loop over row stripes {i} of M, e.g. 128 rows (this is a further sub-division of the stripe passed to this function)
        //       For each row stripe {i} of M,
        //        - loop over chunks of the dot product, e.g. 1024 elements {k}
        //          For each chunk {k},
        //           - accumulate matrix patch (24x128=12 KB) into an accumulator on local stack
        //             That's row stripes {i} of M x col stripes {j} of V, sub-components {k} of the dot products.
        //             Rows are read once and applied to {j} columns of V which come from the cache.

        // we stripe V
        // This is to ensure that we only touch a subset of columns of V at once that fit into
        // the cache. E.g. for a 1024-row V, that would be 195 columns. We then "stream"
        // through M, where each row of M is applied to all those columns of V. This way,
        // both V and M come from the cache except for the first time. Each 'float' of V
        // is loaded once into cache. Each row of M is loaded into cache once per stripe of V,
        // in the example every 195 columns.
        const size_t cacheablerowsV = 512;                  // at most
        const size_t cacheablecolsV = 16;//V.cacheablecols();    // don't get more than this of V per row of M
        // 512 * 16 -> 32 KB

        const size_t colstripewV = cacheablecolsV;          // width of col stripe of V
        const size_t rowstripehM = 128;                     // height of row stripe of M
        const size_t dotprodstep = cacheablerowsV;          // chunk size of dot product

        // loop over col stripes of V
        for (size_t j0 = 0; j0 < V.cols(); j0 += colstripewV)
        {
            const size_t j1 = std::min (j0 + colstripewV, V.cols());
            // stripe of V is columns [j0,j1)

            // loop over row stripes of M
            for (size_t i0 = beginrow; i0 < endrow; i0 += rowstripehM)
            {
                const size_t i1 = std::min (i0 + rowstripehM, endrow);

                // loop over sub-ranges of the dot product (full dot product will exceed the L1 cache)
                float patchbuffer[rowstripehM * colstripewV + 3];    // note: don't forget column rounding
                // 128 * 16 -> 8 KB
                ssematrixbase patch (patchbuffer, i1 - i0, j1 - j0);

                for (size_t k0 = 0; k0 < V.rows(); k0 += dotprodstep)
                {
                    const size_t k1 = std::min (k0 + dotprodstep, V.rows());
                    const bool first = k0 == 0;
                    //const bool last = k0 + dotprodstep >= V.rows();

                    // loop over requested rows [beginrow,endrow) of result (= rows of M (= cols of Mt))
                    for (size_t i = i0; i < i1; i++)    // remember that cols of Mt are the rows of M
                    {
                        // We process row by row, and apply each row to multiple well-cached columns of V.
                        // loop over cols of V
                        const size_t j14 = j1 & ~3; // ... TODO: put this back--when stuff works again
                        for (size_t j = j0; j < j14; j += 4)    // grouped by 4
                        {
                            // Compute 4 columns in parallel, loading 'row' value only once.
                            // Speed-up observed from doing this, measured on 2 x quad-core HT machine
                            //  - single-threaded: RTF  63% ->  37% -- a 42% gain
                            //  - 16-way parallel: RTF 8.4% -> 5.3% -- a 37% gain
                            // These gains are much higher than I expected.
                            const_array_ref<float> row (&Mt.col(i)[k0], k1 - k0);
                            const_array_ref<float> cols4 (&V.col(j)[k0], 4 * V.colstride - k0);
                            array_ref<float> usij (&us(i,j), 4 * us.colstride - i + 1);
                            array_ref<float> patchij (&patch(i-i0,j-j0), 4 * patch.colstride - (i-i0) + 1);

                            //dotprod4 (row, cols4, V.colstride, usij, us.colstride);
                            if (first)
                                dotprod4 (row, cols4, V.colstride, patchij, patch.colstride);
                            else
                                dotprod4 (row, cols4, V.colstride, patchij, patch.colstride, true, 1.0f, 1.0f);

                            // what the above means is:
                            // dotprod (Mt.col(i), V.col(j),   us(i,j));
                            // dotprod (Mt.col(i), V.col(j+1), us(i,j+1));
                            // dotprod (Mt.col(i), V.col(j+2), us(i,j+2));
                            // dotprod (Mt.col(i), V.col(j+3), us(i,j+3));
                        }
                        for (size_t j = j14; j < j1; j++)       // remainder not grouped
                            //dotprod (Mt.col(i), V.col(j), us(i,j));
                            if (first)  // do it in one big step ignoring the cache issue
                                dotprod (Mt.col(i), V.col(j), patch(i-i0,j-j0));
                    }
                }

                // assign patch back
                // TODO: do that inside the loop to avoid copying, but one thing at a time
                assignpatch (patch, i0, i1, j0, j1);
            }
        }
    }

    // this = A * B where B is passed as its transposed form B'
    void matprod_mmt (const ssematrixbase & A, const ssematrixbase & Bt)
    {
        auto & us = *this;
        assert (us.rows() == A.rows());
        assert (us.cols() == Bt.rows());    // Bt.rows() == B.cols()
        assert (A.cols() == Bt.cols());     // Bt.cols() == B.rows()
        //fprintf (stderr, "0x%x(%d,%d) x 0x%x(%d,%d)' -> 0x%x(%d,%d)\n", A.p, A.rows(), A.cols(), Bt.p, Bt.rows(), Bt.cols(), us.p, us.rows(), us.cols());

        foreach_coord (i, j, us)
        {
            // us(i,j) = dotprod (A.row(i), B.col(j))
            size_t K = A.cols();
            float sum = 0.0;
            for (size_t k = 0; k < K; k++)
                sum += A(i,k) * Bt(j,k);
            us(i,j) = sum;
        }
    }

    // regular matrix product
    // Avoid this, not efficient either way.
    void matprod (const ssematrixbase & A, const ssematrixbase & B)
    {
        // ... TODO: put a resize() here and all matmul, so we don't need to set size upfront
        auto & us = *this;
        assert (us.rows() == A.rows() && B.cols() == us.cols());
        size_t K = A.cols();
        assert (K == B.rows());
        foreach_coord (i, j, us)
        {
            float sum = 0.0;
            for (size_t k = 0; k < K; k++)
                sum += A(i,k) * B(k,j);
            us(i,j) = sum;
        }
    }

    // operator += (vector)
    // applied to each column
    // This is a weird interface, as it makes also sense for a matrix. TODO: Fix this.
    void operator += (const ssematrixbase/*vector*/ & other)
    {
        auto & us = *this;
        assert (other.cols() == 1);
        foreach_coord (i, j, us)
            us(i,j) += other[i];
    }

    // operator -= (vector)
    // applied to each column
    // This is a weird interface, as it makes also sense for a matrix. TODO: Fix this.
    void operator -= (const ssematrixbase/*vector*/ & other)
    {
        auto & us = *this;
        assert (other.cols() == 1);
        foreach_coord (i, j, us)
            us(i,j) -= other[i];
    }

#if 0
    // elementwise weighting
    void weigthby (const ssematrixbase & other)
    {
        auto & us = *this;
        foreach_coord (i, j, us)
            us(i,j) *= other(i,j);
    }
#endif

    // column sum --compute for each column the scalar sum of its entries
    // Result is conceptually a row vector, but is returned as a column vector.
    void colsum (ssematrixbase & result) const
    {
        assert (result.size() == cols());   // (size() ensures it's a vector)
        foreach_index (j, result)
        {
            const_array_ref<msra::math::float4> column (col4 (j));
            msra::math::float4 sum (0.0f);
            foreach_index (i, column)
                sum += column[i];
            result[j] = sum.sum();
        }
    }

    // row sum --compute for each row the scalar sum of its entries
    // Not optimized.
    void rowsum (ssematrixbase & result, float otherweight = 1.0f) const
    {
        auto & us = *this;
        assert (result.size() == rows());   // (size() ensures it's a vector)
        result.setzero();
        foreach_column (t, us)
            foreach_row (i, result)
                result[i] += us(i,t);

        if (otherweight != 1.0f)
        {
            foreach_row (i, result)
                result[i] *= otherweight;
        }
    }

    // this = thisweight * this + other * weight
    void addweighted (float thisweight, const ssematrixbase & other, float weight)
    {
        auto & us = *this;
        assert (rows() == other.rows() && cols() == other.cols());

        // get data as long vectors
        // ... why do I need to explicitly use operator T ()?
        array_ref<msra::math::float4> us4 (us.operator array_ref<msra::math::float4> ());
        const_array_ref<msra::math::float4> other4 (other.operator const_array_ref<msra::math::float4> ());
        assert (us4.size() == other4.size());

        // perform the operation on one long vector
        msra::math::float4 weight4 (weight);
        if (thisweight == 1.0f)
        {
            foreach_index (i, us4)
            {
                us4[i] = us4[i] + other4[i] * weight4;
            }
        }
        else if (thisweight == 0.0f)
        {
            foreach_index (i, us4)
            {
                us4[i] = other4[i] * weight4;
            }
        }
        else
        {
            foreach_index (i, us4)
            {
                us4[i] = us4[i] * thisweight + other4[i] * weight4;
            }
        }
    }

    // set the value to zero if less than threshold
    void setto0ifabsbelow (float threshold) 
    {
        auto & us = *this;

        // get data as long vectors
        // ... why do I need to explicitly use operator T ()?
        array_ref<msra::math::float4> us4 (us.operator array_ref<msra::math::float4> ());

        // perform the operation on one long vector
        msra::math::float4 threshold4 (threshold);
        foreach_index (i, us4)
        {
            us4[i] &= ((us4[i] >= threshold4) | (us4[i] <= -threshold4));
        }
    }

    // set the value of this to zero if ref is less than threshold
    void setto0ifabsbelow2 (ssematrixbase & ref, float threshold) 
    {
        assert (rows() == ref.rows() && cols() == ref.cols());
        auto & us = *this;
        auto & refs = ref;

        // get data as long vectors
        // ... why do I need to explicitly use operator T ()?
        array_ref<msra::math::float4> us4 (us.operator array_ref<msra::math::float4> ());
        array_ref<msra::math::float4> refs4 (refs.operator array_ref<msra::math::float4> ());

        // perform the operation on one long vector
        msra::math::float4 threshold4 (threshold);
        foreach_index (i, us4)
        {
            us4[i] &= ((refs4[i] >= threshold4) | (refs4[i] <= -threshold4));
        }
    }

    // set the value of this to zero if ref is higher than threshold
    void setto0ifabsabove2 (ssematrixbase & ref, float threshold) 
    {
        assert (rows() == ref.rows() && cols() == ref.cols());
        auto & us = *this;
        auto & refs = ref;

        // get data as long vectors
        // ... why do I need to explicitly use operator T ()?
        array_ref<msra::math::float4> us4 (us.operator array_ref<msra::math::float4> ());
        array_ref<msra::math::float4> refs4 (refs.operator array_ref<msra::math::float4> ());

        // perform the operation on one long vector
        msra::math::float4 threshold4 (threshold);
        foreach_index (i, us4)
        {
            us4[i] &= ((refs4[i] <= threshold4) & (refs4[i] >= -threshold4));
        }
    }

    // this = this * scale
    void scale (const float factor)
    {
        auto & us = *this;

        // get data as long vectors
        array_ref<msra::math::float4> us4 (us.operator array_ref<msra::math::float4> ());

        // perform the operation on one long vector
        msra::math::float4 scale4 (factor);
        foreach_index (i, us4)
        {
            us4[i] = us4[i] * scale4;
        }
    }

    // this = this * thisscale + other
    void scaleandadd (const float thisscale, const ssematrixbase & other)
    {
        auto & us = *this;
        assert (rows() == other.rows() && cols() == other.cols());

        // get data as long vectors
        // ... why do I need to explicitly use operator T ()?
        array_ref<msra::math::float4> us4 (us.operator array_ref<msra::math::float4> ());
        const_array_ref<msra::math::float4> other4 (other.operator const_array_ref<msra::math::float4> ());
        assert (us4.size() == other4.size());

        // perform the operation on one long vector
        msra::math::float4 thisscale4 (thisscale);
        foreach_index (i, us4)
        {
            us4[i] = us4[i] * thisscale4 + other4[i];
        }
    }

    // special function for DBN
    // this = this * scale + M' * V
    // This is based on a code copy of matprod_mtm. See there for comments.
    void scaleandaddmatprod_mtm (const float thisscale, const ssematrixbase & Mt, const ssematrixbase & V)
    {
        scaleandaddmatprod_mtm (thisscale, Mt, 0, Mt.cols(), V);
    }

#ifdef _WIN32
    void parallel_scaleandaddmatprod_mtm (const float thisscale, const ssematrixbase & Mt, const ssematrixbase & V)
    {
#if 0
        cores;
        scaleandaddmatprod_mtm (thisscale, Mt, 0, Mt.cols(), V);
#else
        msra::parallel::foreach_index_block (Mt.cols(), Mt.cols(), 1, [&] (size_t i0, size_t i1)
        {
            scaleandaddmatprod_mtm (thisscale, Mt, i0, i1, V);
        });
#endif
    }
#endif

    // same as matprod_mtm except result is added to result matrix instead of replacing it
    // For all comments, see matprod_mtm.
    // EXCEPT NOT TRUE ^^: This function did not get matprod's optimizations. Do those if ever needed.
    void scaleandaddmatprod_mtm (const float thisscale, const ssematrixbase & Mt, size_t i0/*first row in M*/, size_t i1/*end row in M*/, const ssematrixbase & V, const float otherweight = 1.0f)
    {
        auto & us = *this;
        assert (V.rows() == Mt.rows());
        assert (us.rows() == Mt.cols());
        assert (us.cols() == V.cols());
        assert (i0 < i1 && i1 <= Mt.cols());

        const size_t cacheablecolsV = V.cacheablecols();

        // loop over stripes of V
        for (size_t j0 = 0; j0 < V.cols(); j0 += cacheablecolsV)
        {
            const size_t j1 = std::min (j0 + cacheablecolsV, V.cols());
            // loop over rows of result = rows of M = cols of Mt
            for (size_t i = i0; i < i1; i++)
            {
                const size_t j14 = j1 & ~3;
                for (size_t j = j0; j < j14; j += 4)
                {
                    const_array_ref<float> row (&Mt.col(i)[0], Mt.colstride);
                    const_array_ref<float> cols4 (&V.col(j)[0], 4 * V.colstride);
                    array_ref<float> usij (&us(i,j), 4 * us.colstride - i + 1);
                    dotprod4 (row, cols4, V.colstride, usij, us.colstride, true, thisscale, otherweight);
                }
                for (size_t j = j14; j < j1; j++)
                    dotprod (Mt.col(i), V.col(j), us(i,j), true, thisscale, otherweight);
            }
        }
    }

#if 0
    // special function for DBN
    // this += hsum(other) * weight
    void addallcolumnsweighted (const ssematrixbase & other, float weight)
    {
        auto & us = *this;
        assert (rows() == other.rows() && cols() == 1);
        foreach_coord (i, t, other)
            us(i,0) += other(i,t) * weight; // TODO: SSE version (very easy)
    }

    // special function for DBN
    // this += x * y
    // This is based on a code copy of matprod_mtm. See there for comments.
    void addmatprodweighted_mtm (const ssematrixbase & Mt, const ssematrixbase & V, const float weight)
    {
        addmatprodweighted_mtm (Mt, 0, Mt.cols(), V, weight);
    }

    void parallel_addmatprodweighted_mtm (const ssematrixbase & Mt, const ssematrixbase & V, const float weight)
    {
#if 0
        cores;
        addmatprodweighted_mtm (Mt, 0, Mt.cols(), V, weight);
#else
        msra::parallel::foreach_index_block (Mt.cols(), Mt.cols(), 1, [&] (size_t i0, size_t i1)
        {
            addmatprodweighted_mtm (Mt, i0, i1, V, weight);
        });
#endif
    }

    void addmatprodweighted_mtm (const ssematrixbase & Mt, size_t i0/*first row in M*/, size_t i1/*end row in M*/, const ssematrixbase & V, const float weight)
    {
        auto & us = *this;
        assert (V.rows() == Mt.rows());     // remember: Mt is the transpose of M
        assert (us.rows() == Mt.cols());
        assert (us.cols() == V.cols());
        assert (i0 < i1 && i1 <= Mt.cols());// remember that cols of Mt are the rows of M

        //for (size_t i = 0; i < Mt.cols(); i++)// remember that cols of Mt are the rows of M
        for (size_t i = i0; i < i1; i++)    // remember that cols of Mt are the rows of M
        {
            size_t j0 = V.cols() & ~3;
            for (size_t j = 0; j < j0; j += 4)
            {
#if 1
                const_array_ref<float> row (&Mt.col(i)[0], Mt.colstride);
                const_array_ref<float> cols4 (&V.col(j)[0], 4 * V.colstride);
                array_ref<float> usij (&us(i,j), 4 * us.colstride - i + 1);

                dotprod4 (row, cols4, V.colstride, usij, us.colstride, true, 1.0f, weight);
#endif
            }
            for (size_t j = j0; j < V.cols(); j++)
                dotprod (Mt.col(i), V.col(j), us(i,j), true, 1.0f, weight);
        }
    }
#endif

#if 1
    // to = this'
    void transpose (ssematrixbase & to) const { transposecolumns (to, 0, cols()); }

#ifdef _WIN32
    void parallel_transpose (ssematrixbase & to) const
    {
        msra::parallel::foreach_index_block (cols(), cols(), 4/*align*/, [&] (size_t j0, size_t j1)
        {
            transposecolumns (to, j0, j1);
        });
#if 0   // double-check
        auto & us = *this;
        foreach_coord (ii, jj, us)
            if (us(ii,jj) != to(jj,ii))
                LogicError("parallel_transpose: post-condition check failed--you got it wrong, man!");
#endif
    }
#endif

    // transpose columns [j0,j1) to rows [j0,j1) of 'to'
    void transposecolumns (ssematrixbase & to, size_t j0, size_t j1) const
    {
        transposepatch (to, 0, rows(), j0, j1);
    }

    // transpose rows [i0,i1) to columns [i0,i1) of 'to'
    // CURRENTLY, i0 must be aligned to 4. (If this is ever not OK, fix it then.)
    void transposerows (ssematrixbase & to, size_t i0, size_t i1) const
    {
        transposepatch (to, i0, i1, 0, cols());
    }

    // transpose patch [i0,i1) x [j0,j1) to patch [j0,j1) x [i0,i1) of target
    // CURRENTLY, i0 must be aligned to 4. (If this is ever not OK, fix it then.)
    // Simple rule to remember: patch dims i0...j1 refer to the source, which is 'us'.
    void transposepatch (ssematrixbase & to, size_t i0, size_t i1, size_t j0, size_t j1) const
    {
        auto & us = *this;
        assert (us.cols() == to.rows() && us.rows() == to.cols());
        assert (i0 < i1 && i1 <= us.rows());
        assert (j0 < j1 && j1 <= us.cols());
        assert (i0 % 4 == 0);   // required for now
        // we loop over 'us' (not 'to'), i.e. i and j refer to row and col of 'us'
        size_t j;
        for (j = j0; j + 4 <= j1; j += 4)       // 4 columns at a time (j0 does not need to be aligned)
        {
            // transpose blocks of 4x4 starting at (i,j)
            msra::math::float4 mt0, mt1, mt2, mt3;
            size_t i;
            for (i = i0; i + 4 <= i1; i += 4)   // 4 rows at a time
            {
                msra::math::float4 m0 = us.float4(i,j);   // gets i..i+3  --i must be aligned to 4
                msra::math::float4 m1 = us.float4(i,j+1);
                msra::math::float4 m2 = us.float4(i,j+2);
                msra::math::float4 m3 = us.float4(i,j+3);
                msra::math::float4::transpose (m0, m1, m2, m3, mt0, mt1, mt2, mt3);
                mt0.storewithoutcache (to.float4(j,i));    // writes j..j+3
                mt1.storewithoutcache (to.float4(j,i+1));
                mt2.storewithoutcache (to.float4(j,i+2));
                mt3.storewithoutcache (to.float4(j,i+3));
            }
            // left-over rows --we can read all rows (they are padded)
            // but cannot write all target columns
            if (i < i1)
            {
                msra::math::float4 m0 = us.float4(i,j);   // gets i..i+3 (padded)
                msra::math::float4 m1 = us.float4(i,j+1);
                msra::math::float4 m2 = us.float4(i,j+2);
                msra::math::float4 m3 = us.float4(i,j+3);

                msra::math::float4::transpose (m0, m1, m2, m3, mt0, mt1, mt2, mt3);
                assert (i < to.cols());
                mt0.storewithoutcache (to.float4(j,i));    // writes j..j+3
                if (i+1 < i1)
                {
                    assert (i+1 < to.cols());
                    mt1.storewithoutcache (to.float4(j,i+1));
                    if (i+2 < i1)
                    {
                        assert (i+2 < to.cols());
                        mt2.storewithoutcache (to.float4(j,i+2));
                        if (i+3 < i1)
                        {
                            assert (i+3 < to.cols());
                            mt3.storewithoutcache (to.float4(j,i+3));
                        }
                    }
                }
            }
        }
        // left-over columns --don't try to optimize
        // (we could use the same approach as above)
        for ( ; j < j1; j++)
            for (size_t i = i0; i < i1; i++)
                to(j,i) = us(i,j);
#if 0   // double-check
        for (size_t jj = 0; jj < j1; jj++)
            foreach_row (ii, us)
                if (us(ii,jj) != to(jj,ii))
                    LogicError("transpose: post-condition check failed--you got it wrong, man!");
#endif
    }

#if 0   // untested leftover:
    void checktranspose (ssematrixbase & V) const
    {
        auto & U = *this;
        assert (U.cols() == V.rows() && U.rows() == V.cols());
        foreach_coord (i, j, U)
            if (U(i,j) != V(j,i))
                LogicError("checktranspose: post-condition check failed--you got it wrong, man!");
    }
#endif
#else   // futile attempts to speed it up --the imul don't matter (is SSE so slow?)
    // to = this'
    void transpose (ssematrixbase & to) const
    {
        auto & us = *this;
        assert (us.cols() == to.rows() && us.rows() == to.cols());
        // we loop over 'us' (not 'to'), i.e. i and j refer to row and col of 'us'
        size_t j;
        for (j = 0; j + 4 <= us.cols(); j += 4)
        {
            // transpose blocks of 4x4 starting at (i,j)
            const msra::math::float4 * pusij = &us.float4(0,j);
            size_t uscolstride4 = us.colstride / 4;
            size_t tocolstride4 = to.colstride / 4;
            size_t i;
            for (i = 0; i + 4 <= us.rows(); i += 4)
            {
                assert (pusij == &us.float4(i,j));

                const msra::math::float4 * pusijp1 = pusij + uscolstride4;
                assert (pusijp1 == &us.float4(i,j+1));

                const msra::math::float4 * pusijp2 = pusijp1 + uscolstride4;
                assert (pusijp2 == &us.float4(i,j+2));

                const msra::math::float4 * pusijp3 = pusijp2 + uscolstride4;
                assert (pusijp3 == &us.float4(i,j+3));

                msra::math::float4 m0 = *pusij;   // gets i..i+3
                msra::math::float4 m1 = *pusijp1;
                msra::math::float4 m2 = *pusijp2;
                msra::math::float4 m3 = *pusijp3;

                msra::math::float4 mt0, mt1, mt2, mt3;
                msra::math::float4::transpose (m0, m1, m2, m3, mt0, mt1, mt2, mt3);

                msra::math::float4 * ptoji = &to.float4(j,i);
                mt0.storewithoutcache (ptoji[0]);    // writes j..j+3
                mt1.storewithoutcache (ptoji[0+tocolstride4]);
                mt2.storewithoutcache (ptoji[0+tocolstride4+tocolstride4]);
                mt3.storewithoutcache (ptoji[0+tocolstride4+tocolstride4+tocolstride4]);
                pusij++;
            }
            // left-over rows --we can read all rows (they are padded)
            // but cannot write all target columns
            for ( ; i < us.rows(); i++)
            {
                msra::math::float4 m0 = us.float4(i,j);   // gets i..i+3 (zero-padded)
                msra::math::float4 m1 = us.float4(i,j+1);
                msra::math::float4 m2 = us.float4(i,j+2);
                msra::math::float4 m3 = us.float4(i,j+3);
                msra::math::float4 mt0, mt1, mt2, mt3;
                msra::math::float4::transpose (m0, m1, m2, m3, mt0, mt1, mt2, mt3);
                assert (i < to.cols());
                mt0.storewithoutcache (to.float4(j,i));    // writes j..j+3
                if (i+1 < to.cols())
                {
                    mt1.storewithoutcache (to.float4(j,i+1));
                    if (i+2 < to.cols())
                    {
                        mt2.storewithoutcache (to.float4(j,i+2));
                        if (i+3 < to.cols())
                            mt3.storewithoutcache (to.float4(j,i+3));
                    }
                }
            }
        }
        // left-over columns --don't try to optimize
        // (we could use the same approach as above)
        for ( ; j < us.cols(); j++)
            foreach_row (i, us)
                to(j,i) = us(i,j);
#if 0   // double-check
        foreach_coord (ii, jj, us)
            if (us(ii,jj) != to(jj,ii))
                LogicError("transpose: post-condition check failed--you got it wrong, man!");
#endif
    }
#endif

    // multiply a sequence of column vectors by the sigmoid derivative
    void mulbydsigm (const ssematrixbase & h)
    {
#if 1
        auto & us = *this;
        assert (rows() == h.rows() && cols() == h.cols());

        // get data as long vectors
        // ... why do I need to explicitly use operator T ()?
        array_ref<msra::math::float4> us4 (us.operator array_ref<msra::math::float4> ());
        const_array_ref<msra::math::float4> h4 (h.operator const_array_ref<msra::math::float4> ());
        assert (us4.size() == h4.size());

        // perform the operation
        msra::math::float4 one (1.0f);
        foreach_index (i, us4)
            us4[i] = us4[i] * h4[i] * (one - h4[i]);  // eh(i,t) *= h(i,t) * (1.0f - h(i,t));
#else
        auto & us = *this;
        foreach_coord (i, t, us)
            us(i,t) *= h(i,t) * (1.0f - h(i,t));
#endif
    }

    // fetch entire object into the cache
    // Does this really make sense?? Should be rather done during computation.
    void prefetch() const
    {
        const msra::math::float4 * p = (msra::math::float4 *) this->p;
        size_t numfloat4s = cols() * colstride/4;
        const msra::math::float4 * q = p + numfloat4s;
        const size_t cacherowbytes = 64;    // or what?
        const size_t cacherowfloat4s = cacherowbytes / sizeof (*p);
        for ( ; p < q; p += cacherowfloat4s)
            msra::math::float4::prefetch (p);
    }

    // diagnostics helper to check if matrix has a NaN
    // This takes up 20% of total runtime.
    bool hasnan (const char * name) const
    {
#if 0
        name;
        return false;
#else
        const auto & us = *this;
        foreach_coord (i, j, us)
            if (std::isnan (us(i,j)))
            {
                fprintf (stderr, "hasnan: NaN detected at %s (%d,%d)\n", name, (int)i, (int)j);
                return true;
            }
#endif
        return false;
    }
#define checknan(m) m.hasnan (#m)

    // another diagnostics helper to check if matrix has a NaN
    // This is used at load and save time. This test is slow.
    size_t countnaninf() const
    {
        const auto & us = *this;
        size_t n = 0;   // number of NaNs/INF found
        foreach_coord (i, j, us)
        {
            auto val = us(i,j);
            if (std::isnan (val) || !std::isfinite (val))
                n++;
        }
        return n;
    }

    // check if two matrices are equal
    void checkequal (const ssematrixbase & other) const
    {
        const auto & us = *this;
        if (us.cols() != other.cols() || us.rows() != other.rows())
            LogicError("checkequal: post-condition check failed (dim)--you got it wrong, man!");
        foreach_coord (i, j, us)
            if (us(i,j) != other(i,j))
                LogicError("checkequal: post-condition check failed (values)--you got it wrong, man!");
    }

    void dump(char * name) const
    {
        name;
        // provide if necessary
    }
};

// ===========================================================================
// ssematrixfrombuffer -- an ssematrixbase allocated in a vector buffer
// If you need many little matrices in your own heap
// ===========================================================================

class ssematrixfrombuffer : public ssematrixbase
{
    void operator= (const ssematrixfrombuffer &); ssematrixfrombuffer (const ssematrixfrombuffer &);  // base cannot be assigned except by move
public:
    ssematrixfrombuffer() { this->clear(); }

    // instantiate from a float vector  --buffer must be SSE-aligned
    template<class VECTOR> ssematrixfrombuffer (VECTOR & buffer, size_t n, size_t m) : ssematrixbase (buffer, n, m) {}

    // allocation size needed   --buffer must have this size
    static size_t elementsneeded (size_t n, size_t m) { const size_t colstride = (n + 3) & ~3; return colstride * m; }

    // we can assign it, but only by move
    void operator= (ssematrixfrombuffer && other) { move (other); }

    // Note: keyword "noexcept" was added so that stl vector first looks for
    //       the move constructor instead of the private copy constructor.
    ssematrixfrombuffer(ssematrixfrombuffer && other) noexcept { std::move(other); }
};


// ===========================================================================
// ssematrixstripe -- a sub-column view on a matrix
// This provides a reference to the memory of an underlying matrix object without owning the memory.
// ===========================================================================

template<class ssematrixbase> class ssematrixstriperef : public ssematrixbase
{
    // do not assign this; instead pass around by reference
    // (we could give this up easily, but why if never needed so far)
    ssematrixstriperef & operator= (ssematrixstriperef & other);
    ssematrixstriperef (ssematrixstriperef & other);
public:
    // ... TODO: should this be moved into the base class? no need for separate type, just have a stripe() function just like col()
    // Note: 'other' may  be empty. In that case, return an empty matrix (0 x 0--will fail if tried to be accessed).
    ssematrixstriperef (ssematrixbase & other, size_t j0, size_t m)
    {
        assert (other.empty() || j0 + m <= other.cols());
        if (!other.empty() && j0 + m > other.cols())  // (runtime check to be sure--we use this all the time)
            LogicError("ssematrixstriperef: stripe outside original matrix' dimension");
        this->p = other.empty() ? NULL : &other(0,j0);
        this->numrows = other.rows();
        this->numcols = m;
        this->colstride = other.getcolstride();
    }

    // only assignment is by rvalue reference
    ssematrixstriperef & operator= (ssematrixstriperef && other) { std::move (other); }

    // Note: keyword "noexcept" was added so that stl vector first looks for
    //       the move constructor instead of the private copy constructor.
    ssematrixstriperef(ssematrixstriperef && other) noexcept{ std::move(other); }

    // getting a one-column sub-view on this
    ssematrixstriperef col (size_t j) { return ssematrixstriperef (*this, j, 1); }
    const ssematrixstriperef col (size_t j) const { return ssematrixstriperef (*const_cast<ssematrixstriperef*> (this), j, 1); }
};


// ===========================================================================
// ssematrix -- main matrix type with allocation
// ===========================================================================

template<class ssematrixbase> class ssematrix : public ssematrixbase
{
    // helpers for SSE-compatible memory allocation
    static __declspec_noreturn void failed(size_t nbytes) { BadExceptionError("allocation of SSE vector failed (%d bytes)", nbytes); }
#ifdef _WIN32
    template<typename T> static T * new_sse (size_t nbytes) { T * pv = (T *) _aligned_malloc (nbytes * sizeof (T), 16); if (pv) return pv; failed (nbytes * sizeof (T)); }
    static void delete_sse (void * p) { if (p) _aligned_free (p); }
#endif
#ifdef __unix__
    template<typename T> static T * new_sse (size_t nbytes) { T * pv = (T *) _mm_malloc (nbytes * sizeof (T),16); if (pv) return pv; failed (nbytes * sizeof (T)); }
    static void delete_sse (void * p) { if (p) _mm_free (p); }
#endif

    // helper to assign a copy from another matrix
    void assign (const ssematrixbase & other)
    {
        resize (other.rows(), other.cols());
        ssematrixbase::assign (other);
    };
public:
    // construction
    ssematrix() { this->clear(); }
    ssematrix (size_t n, size_t m) { this->clear(); resize (n, m); }
    ssematrix (size_t n) { this->clear(); resize (n, 1); }  // vector
    ssematrix (const ssematrix & other) { this->clear(); assign (other); }
    ssematrix (const ssematrixbase & other) { this->clear(); assign (other); }
    ssematrix (ssematrix && other) { this->move (other); }
    ssematrix (const std::vector<float> & other) { this->clear(); resize (other.size(), 1); foreach_index (k, other) (*this)[k] = other[k]; }

    // construct elementwise with a function f(i,j)
    template<typename FUNCTION> ssematrix (size_t n, size_t m, const FUNCTION & f)
    {
        this->clear();
        resize (n, m);
        auto & us = *this;
        foreach_coord (i, j, us)
            us(i,j) = f (i, j);
    }

    // destructor
    ~ssematrix() { delete_sse (this->p); }

    // assignment
    ssematrix & operator= (const ssematrix & other) { assign (other); return *this; }
    ssematrix & operator= (const ssematrixbase & other) { assign (other); return *this; }
    ssematrix & operator= (ssematrix && other) { delete_sse(this->p); move (other); return *this; }

    void swap (ssematrix & other) throw() { ssematrixbase::swap (other); }

    // resize (destructive--matrix content will be undefined, don't assume it's 0)
    // One or both dimensions can be 0, for special purposes.
    void resize (size_t n, size_t m)
    {
        if (n == this->numrows && m == this->numcols)
            return;                             // no resize needed
        const size_t newcolstride = (n + 3) & ~3;     // pad to multiples of four floats (required SSE alignment)
        const size_t totalelem = newcolstride * m;
        //fprintf (stderr, "resize (%d, %d) allocating %d elements\n", n, m, totalelem);
        float * pnew = totalelem > 0 ? new_sse<float> (totalelem) : NULL;
        std::swap (this->p, pnew);
        delete_sse (pnew);    // pnew is now the old p
        this->numrows = n; this->numcols = m;
        this->colstride = newcolstride;
        // touch the memory to ensure the page is created
        for (size_t offset = 0; offset < totalelem; offset += 4096 / sizeof (float))
            this->p[offset] = 0.0f; //nan;
        // clear padding elements (numrows <= i < colstride) to 0.0 for SSE optimization
        for (size_t j = 0; j < this->numcols; j++)
            for (size_t i = this->numrows; i < this->colstride; i++)
                this->p[j * this->colstride + i] = 0.0f;
#if 1   // for debugging: set all elements to 0
        // We keep this code alive because allocations are supposed to be done at the start only.
        auto & us = *this;
        foreach_coord (i, j, us)
            us(i,j) = 0.0f;
#endif
    }

    // same as resize() but only allowed for uninitialized matrices; otherwise dimensions must match
    // Actually, there are special cases where we still resize(). So we allow it, but log a message.
    // Should fix this someday.
    void resizeonce (size_t n, size_t m)
    {
#if 1   // BUGBUG: at end of epoch, resizes are OK... so we log but allow them
        if (!this->empty() && (n != this->numrows || m != this->numcols))
            fprintf (stderr, "resizeonce: undesired resize from %d x %d to %d x %d\n", this->numrows, this->numcols, n, m);
        resize (n, m);
#else
        if (empty())
            resize (n, m);
        else if (n != numrows || m != numcols)
            LogicError("resizeonce: attempted to resize a second time to different dimensions");
#endif
    }

    // non-destructive resize() to a smaller size
    void shrink(size_t newrows, size_t newcols)
    {
        if (newrows > this->numrows || newcols > this->numcols)
            LogicError("shrink: attempted to grow the matrix");
        this->numrows = newrows;
        this->numcols = newcols;
    }

    // file I/O
    void write (FILE * f, const char * name) const
    {
        fputTag (f, "BMAT");
        fputstring (f, name);
        fputint (f, (int) this->numrows);
        fputint (f, (int) this->numcols);
        const auto & us = *this;
        foreach_column (j, us)
        {
            auto column = ssematrixbase::col (j);
            fwriteOrDie (column, f);
        }
        fputTag (f, "EMAT");
    }

    void write (const HANDLE f, const char * name) const
    {
        fputTag(f, "BMAT");
        fputstring (f, name);
        fputint (f, (int) this->numrows);
        fputint (f, (int) this->numcols);
        const auto & us = *this;
        foreach_column (j, us)
        {
            auto column = ssematrixbase::col (j);
            fwriteOrDie (column, f);
        }
        fputTag (f, "EMAT");
    }


    void read (FILE * f, const char * name)
    {
        fcheckTag (f, "BMAT");
        char namebuf[80];
        const char * nameread = fgetstring (f, namebuf);
        if (strcmp (name, nameread) != 0)
            RuntimeError("unexpected matrix name tag '%s', expected '%s'", nameread, name);
        size_t n = fgetint (f);
        size_t m = fgetint (f);
        resize (n, m);
        auto & us = *this;
        foreach_column (j, us)
        {
            auto column = ssematrixbase::col (j);
            freadOrDie (&column[0], sizeof (column[0]), column.size(), f);
        }
        fcheckTag (f, "EMAT");
    }

    // TODO: should this be a function template?
    void read (const HANDLE f, const char * name)
    {
        fcheckTag (f, "BMAT");
        char namebuf[80];
        const char * nameread = fgetstring (f, namebuf);
        if (strcmp (name, nameread) != 0)
            RuntimeError("unexpected matrix name tag '%s', expected '%s'", nameread, name);
        size_t n = fgetint(f);
        size_t m = fgetint (f);
        resize (n, m);
        auto & us = *this;
        foreach_column (j, us)
        {
            auto column = ssematrixbase::col (j);
            freadOrDie (&column[0], sizeof (column[0]), column.size(), f);
        }
        fcheckTag (f, "EMAT");
    }

    // paging support (used in feature source)
    void topagefile (FILE * f) const { if (!this->empty()) fwriteOrDie (this->p, sizeinpagefile(), 1, f); }
    void frompagefile (FILE * f) { if (!this->empty()) freadOrDie (this->p, sizeinpagefile(), 1, f); }
    size_t sizeinpagefile() const { return this->colstride * this->numcols * sizeof (*(this->p)); }

    // getting a one-column sub-view on this
    ssematrixstriperef<ssematrixbase> col (size_t j)
    {
        return ssematrixstriperef<ssematrixbase> (*this, j, 1);
    }

    void dump (char * name)
    {
        printmatf(name, *this);
    }

#if 0
    // creating the transpose of a matrix
    ssematrix transpose() const
    {
        auto & us = *this;
        return ssematrix (cols(), rows(), [&] (size_t i, size_t j) { return us(j,i); };
    }

#endif
};

// diagnostics helper to track down 
template<class M>
static void printmatsumf (const char * name, const M & m)
{
    m.hasnan();
#if 0
    float s = 0.0;
    foreach_coord (i, j, m)
        s += m(i,j);
    fprintf (stderr, "###### %s -> %.10f\n", name, s);
#endif
}
#define printmatsum(m) msra::math::printmatsumf(#m, m)

template<class M>
void printmatf (const char * name, const M & m, FILE *f = stderr)
{
    fprintf (f, "\n###### %s (%d, %d) ######\n", name, m.rows(), m.cols());
    foreach_row(i,m)
    {
        fprintf (f, "row: %d", i);
        foreach_column(j,m)
        {   
            if (j%15 == 0)
                fprintf (f, "\n");
            fprintf (f, "%.4f\t",  m(i,j));
        }
    }
}

#define printmat(m) msra::math::printmatf(#m, m)
#define printmatfile(m,f) msra::math::printmatf(#m, m, f)

// (helper for qsort() in printmatvaluedistributionf() below --TODO: use a lambda?)
static inline int floatcompare (const void * a, const void * b)
{
    return ( *(float*)a > *(float*)b )? 1: (( *(float*)a < *(float*)b )? -1:0);
}

// print model stats
// Returns a pair (model params, non-null model params) for aggregate statistics printing.
template<class M> std::pair<unsigned int,unsigned int> printmatvaluedistributionf (const char * name, const M & m)
{
    const unsigned int num = (unsigned int) (m.rows() * m.cols());
    if (num == 0) return std::make_pair (0UL, 0UL);
    fprintf (stderr, "\n###### absolute weight value distribution %s (%d, %d) ######\n", name, m.rows(), m.cols());

    std::vector<float> vals (num);
    size_t k = 0;
    unsigned int numzeros = 0;
    foreach_coord (i, j, m)
    {
        vals[k] = abs(m(i,j));  //this is slower than memcpy but without assumption on how values are stored.
        numzeros += (vals[k++] < 1e-10f);
    }

    qsort(&vals[0], num, sizeof (vals[0]), floatcompare);

#ifdef PRINT_MEAN_VARIANCE
    double mean = 0;
    size_t count = 0;
    foreach_row(i,m)
    {
        double colsum = 0;
        foreach_column(j,m)
        {
            colsum += m(i,j);
            count += 1;
        }
        mean += colsum;
    }
    mean /= count;
    double variance = 0;
    foreach_row (i,m)
    {
        double colsum = 0;
        foreach_column(j,m)
        {
            colsum += (m(i,j)-mean) * (m(i,j)-mean);
        }
        variance += colsum;
    }
    variance /= count;
    fprintf (stderr, "\n###### count = %d, mean = %0.12f, variance = %.12f, stddev = %.12f ######\n", count, mean, variance, sqrt(variance));
#endif
#if 1
    const size_t numparts = 100;
    for (size_t i=1; i<=numparts; i++)
    {
        fprintf (stderr, "%.5f%% absolute values are under %.10f\n", i*100.0/numparts, vals[std::min((size_t)num-1,i*num/numparts)]);
    }
    fprintf (stderr, "\n%.5f%% values are zero\n\n", 100.0*numzeros/num);
#endif
#if 0   // experimental: dump the length of each column  --are they similar?
    if (m.rows() > 1 && m.cols() > 1)
    {
        fprintf (stderr, "\n### lengths of columns\n");
        double avlen = 0.0;
        foreach_column (j, m)
        {
            if (j % 20 == 0)
                fprintf (stderr, "\n%d:\t", j);
            else
                fprintf (stderr, "\t");
            double sum = 0.0;
            foreach_row (i, m)
                sum += m(i,j) * m(i,j);
            double len_j = sqrt (sum);
            fprintf (stderr, "%7.3f", len_j);
            avlen += len_j;
        }
        fprintf (stderr, "\n\n%s -> av length = %.10f\n", name, avlen / m.cols());
    }
    else if (m.rows() > 1)
    {
        fprintf (stderr, "\n### biases\n");
        double avbias = 0.0;
        foreach_row (j, m)
        {
            if (j % 20 == 0)
                fprintf (stderr, "\n%d:\t", j);
            else
                fprintf (stderr, "\t");
            fprintf (stderr, "%7.3f", m[j]);
            avbias += m[j];
        }
        fprintf (stderr, "\n\n%s -> av bias = %.10f\n", name, avbias / m.rows());
    }
#endif

    return std::make_pair (num, num - numzeros);
}
#define printmatvaluedistribution(m) msra::math::printmatvaluedistributionf(#m, m)


// double matrix in column-wise storage
class doublematrix 
{
protected:
    size_t nrows;
    size_t ncols;
    double *p;

    size_t locate (size_t i, size_t j) const { assert (i < nrows && j < ncols); return j * nrows + i; }   // matrix in column-wise storage

public:
    doublematrix() :
      nrows(0),
      ncols(0),
      p(0)
      {}

    virtual ~doublematrix()
    {
        if (p)
            delete p;
    }

    virtual void allocate(size_t n, size_t m)
    {
        nrows = n;
        ncols = m;
        if (p)
            delete p;
        p = new double[n*m];
    }

    double &       operator() (size_t i, size_t j)       { return p[locate(i,j)]; }
    const double & operator() (size_t i, size_t j) const { return p[locate(i,j)]; }


    virtual void reset()
    {
        if (p)
            memset(p, 0, nrows * ncols * sizeof(double));
    }
    
    template<class matrixbase>
    void addfloat(double thisscale, const msra::math::ssematrix<matrixbase> &other, float otherweight)
    {
        assert(nrows == other.rows());
        assert(ncols == other.cols());
        if (thisscale == 0.0)
        {
            for (size_t j=0; j < ncols; j++)
                for (size_t i=0; i < nrows; i++)
                    (*this)(i,j) = otherweight * other(i,j);
        }
        else if (thisscale == 1.0)
        {
            for (size_t j=0; j < ncols; j++)
                for (size_t i=0; i < nrows; i++)
                    (*this)(i,j) += otherweight * other(i,j);
        }
        else
        {
            for (size_t j=0; j < ncols; j++)
                for (size_t i=0; i < nrows; i++)
                    (*this)(i,j) = thisscale * (*this)(i,j) + otherweight * other(i,j);
        }
    }

    template<class matrixbase>
    void tomatrix(msra::math::ssematrix<matrixbase> &to) const
    {
        for (size_t j = 0; j < ncols; j++)
            for (size_t i = 0; i < nrows;i++)
                to(i,j) = (float) (*this)(i,j);

    }

};

};};    // namespaces

namespace msra { namespace dbn {

// ===========================================================================
// matrix, vector types for use in the networks
// ===========================================================================

typedef msra::math::ssematrixbase matrixbase;

// CPU-side matrices and vectors for intermediate CPU-side computation
typedef msra::math::ssematrix<matrixbase> matrix;
typedef msra::math::ssematrixstriperef<matrixbase> matrixstripe;
// TODO: This type conflicts with std::vector --we should rename it
typedef msra::math::ssematrix<matrixbase> vector;

};};
