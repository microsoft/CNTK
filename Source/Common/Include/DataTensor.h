// DataTensor.h -- tensor descriptor that describes the inner structure of data vectors
//
// <copyright file="Sequences.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "Basics.h"
#include "File.h"
#include <vector>
#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // TensorShape -- tensor descriptor to describe the inner layout of a sample vector that holds a tensor
    //
    // Minibatches are stored as Matrix objects. While the column dimension represents multiple sample vectors, and may have
    // an inner structure (time, parallel sequences) described by the MBLayout, the row dimension represents data
    // vectors that hold tensors of data.
    //
    // To the user, the TensorShape describes the inner tensor structure of these vectors, as a
    // column-major tensor of arbitrary number of dimensions. (Internally, it may sometimes combine the MBLayout as well.)
    //
    // Specifically, when the image is an image, then this is a 3-dimensional tensor with dimensions ( channels, width, height ),
    // which represents the column-major interpretation of a transposed row-by-row-scanned image where each pixel stores {R,G,B} as a float3.
    // -----------------------------------------------------------------------

    // Plans for improved tensor support:
    //
    // TensorShape support for:
    //  - column-major arbitrary-dimension arrays  --this is already implemented
    //  - strides for storage, allowing
    //     - slicing
    //  - strides for computation, allowing
    //     - broadcasting (stride = 0)
    //     - stride magic such as inverting index order (negative stride) or convolution (stride < dimension)
    //  - insertion of 1-dimension (cf. 'new_axis' in numpy), and dropping 1-dimensions (projection)
    //  - BrainScript syntaxes for the above
    //
    // Relation to Matrix and MBLayout:
    //  - tensors are stored in Matrix objects
    //  - both matrix row and column dimensions are interpreted as tensor dimensions separately
    //     - row dimension is explained by a TensorShape ComputationNode::SampleLayout (which must match m_numRows, or might even replace it)
    //     - column dimensions are explained by MBLayout, which has one parallel-sequence index and one (or more) time-step dimensions, e.g. (s,t)
    //  - the total tensor shape of what is stored in the matrix is
    //     - no MBLayout: the SampleLayout
    //     - in presence of an MBLayout, it is determined as
    //        - when applying element-wise operations, first expand all operands to the same SampleLayout length by padding with 1-dimensions
    //        - concatenate that shape, say, (I,J,K) with the shape derived from the MBLayout, say (S,T) -> (I,J,K,S,T)
    //        - these extra dimensions are only used internally, but not accessible to the user (user/network definition operates on samples only)
    //     - examples:
    //        - A[(I,J,K), (S,T)] + B[(I,J,K), (S,T)] -> C[I,J,K,S,T]   // all dimensions match
    //        - A[(I,J), (S,T)] + B[(I,J,K), (S,T)] -> C[I,J,K,S,T]     // A gets an additional broadcasting dimension that matches K
    //        - A(I,T) + B(I) -> C(I,T)                                 // T is broadcasting for B, e.g. adding a bias
    //        - A(I,T1,T2) + B(1,T1) -> C(I,T1,T2)                      // 2D iteration; implies a third dim for B where both first and third dim broadcast
    //
    // Supported operations:
    //  - all elementwise operations (such as sum, sigmoid, or training criterion):
    //     - dimensions are expanded as explained above for all operands
    //     - also supported is inverse broadcasting for the result
    //        - this means that we will sum over the broadcasting dimension(s)
    //        - intended to support gradient computation for a broadcasting input dimension
    //        - for now, must be flattenable to a single dimension
    //     - noteworthy operations also included here:
    //        - elementwise 'copy', which allows for strided copies
    //        - MUX (one matrix contains an index to select one of multiple inputs; can also implement a parallel "if c then a else b" node)
    //        - indexing/lookup (one tensor contains indices into the other)
    //  - inner product (Kronecker product+contraction) -> TimesNode
    //     - A[U,I,J] * B[I,J,T] -> C[A,T], c_ut = sum_ij a_uij * b_ijt
    //     - allows output and input tensors (TimesNode will get optional parameter how many leading dims to not contract), e.g.
    //       A[U,V,I,J] * B[I,J,S,T] -> C[U,V,S,T], c_uvst = sum_ij a_uvij * b_ijst
    //     - for now this operation must be flattenable as to be implementable as SGEMM (may extend in the future)
    //  - tensor transpose -> TransposeNode
    //     - swaps any two dimensions. This does not change the column-major definition, i.e. requires a memory copy.
    //     - special case: swapping between sample and MBLayout, e.g. turn a sample dimension to a time dimension
    //  - Validate() stage will automatically infer tensor dimensions from inputs, and also infer downwards into LearnableParameters where requested
    //
    // Interfacing to and inplementation in Matrix lib:
    //  - a Tensor is realized as a type TensorView = { Matrix&, TensorShape& } (i.e. tensors don't own their memory)
    //  - Matrix lib will contain overloads for relevant operations that take Tensor& instead of Matrix&.
    //  - elementwise ops will go through a single bottleneck function that deals with matching dimensions (extend, broadcast) and flattening

    struct TensorShape
    {
    public:
        // main constructor (from vector that holds dimensions)
        template<class VEC>
        TensorShape(const VEC & dims) { m_dims.assign(dims.begin(), dims.end()); InitAsNoSlice(); }
        TensorShape(std::vector<size_t> && dims) : m_dims(std::move(dims)) { InitAsNoSlice(); }

        // convenience constructors, e,g. for test code
        TensorShape(size_t I) : TensorShape(std::vector<size_t> { I }) { }
        TensorShape(size_t I, size_t J) : TensorShape(std::vector<size_t> { I, J }) { }
        TensorShape(size_t I, size_t J, size_t K) : TensorShape(std::vector<size_t> { I, J, K }) { }
        TensorShape(size_t I, size_t J, size_t K, size_t L) : TensorShape(std::vector<size_t> { I, J, K, L }) { }
        TensorShape(size_t I, size_t J, size_t K, size_t L, size_t M) : TensorShape(std::vector<size_t> { I, J, K, L, M }) { }

        // default constructor
        // BUGBUG: This default initialization is not correct. This must match GetNumRows(). We probably cannot have all three members here.
        TensorShape() : TensorShape(1, 1, 1) { }

        // boilerplate
        bool operator==(const TensorShape & other) const { return m_dims == other.m_dims; }

        void Invalidate() { m_dims.assign(3, SIZE_MAX); } // TODO: clean up the valid/invalid situation (this is currently done inconsistently). Also this object is immutable.

        // verify that this refers to a dense matrix (no strides)
        void VerifyIsDense() const
        {
            if (m_offset != 0)
                LogicError("TensorShape: A dense TensorShape expected. Offset %d not allowed.", (int)m_offset);
            for (size_t k = 0; k < m_dims.size(); k++)  // (TODO: we can save one multiplication here)
            {
                ptrdiff_t stride = k > 0 ? m_strides[k - 1] * (ptrdiff_t)m_dims[k - 1] : 1;
                if (m_strides[k] != stride)
                    LogicError("TensorShape: A dense TensorShape expected. Dimension %d is not.", (int)k);
            }
        }

        void Save(File& fstream) const
        {
            VerifyIsDense();
            // saving as 32-bit ints. This allows to continue to support the old format (size_t W, H, C)
            fstream << (uint32_t)m_dims.size();
            for (auto dim : m_dims)
            {
                if (dim > UINT32_MAX)
                    LogicError("TensorShape::Save(): Tensor dimensions %s out of bounds (> 4G).", string(*this).c_str());
                fstream << (uint32_t)dim;
            }
        }

        void Load(File& fstream)
        {
            // format: uint32_t n, dim[0], dim[1], ..., dim[n-1]
            // We are also able to read (but not write) an older format, which stores 3-dimensional tensors as size_t W, H, C
            uint32_t n, dim;
            fstream >> n >> dim;
            if (dim)        // heuristic to detect the old format. Old format stores a size_t, i.e. the second uint32_t is 0 (no dimensions are > 4G)
            {
                m_dims.resize(n);
                m_dims[0] = dim;
                for (size_t i = 1; i < n; i++)
                {
                    fstream >> dim;
                    m_dims[i] = dim;
                }
                assert(n == m_dims.size());
            }
            else            // detected the old size_t W, H, C format
            {
                m_dims.resize(3);     // current format is hard-coded for 3, for back compat
                m_dims[1] = n;
                fstream >> m_dims[2] >> m_dims[0]; // currently stored in order W, H, C. TODO: general tensor format will be different
            }
            InitAsNoSlice();
        }

        // accessors
        size_t GetDim(size_t k) const { return m_dims[k]; }
        size_t GetNumDims() const { return m_dims.size(); }
        size_t GetNumElements() const { size_t res = 1; for (auto & dim : m_dims) res *= dim; return res; } // in slice
        size_t GetOffset() const { return m_offset; }

        // vector-like accessors
        size_t operator[](size_t k) const { return GetDim(k); }
        size_t size() const { return GetNumDims(); }

        const std::vector<size_t> & GetDims() const { return m_dims; }    // get all, e.g. for logging or for constructing derived tensors with edited dimensions
        const std::vector<ptrdiff_t> & GetStrides() const { return m_strides; }

        // interpretation as an image tensor
        size_t GetNumChannels() const { return m_dims[0]; }
        size_t GetWidth()       const { return m_dims[1]; }
        size_t GetHeight()      const { return m_dims[2]; }

        // indexing
        // Determines the offset into the underlying element array for a given multi-dimensional index.
        // This function is for reference. Probably not often used.
        size_t Locate(const std::vector<size_t> & index) const
        {
            ptrdiff_t location = m_offset;
            for (size_t k = 0; k < index.size(); k++)
            {
                size_t dim = k < size() ? m_dims[k] : 1;        // dimensions are bottomless
                if (index[k] >= dim)
                    LogicError("Locate: Tensor index[%d]=%d exceeds bound %d.", (int)k, (int)index[k], (int)dim);
                location += (ptrdiff_t)index[k] * m_strides[k]; // strides may be negative
            }
            if (location < 0 || (size_t)location >= m_allocation)
                LogicError("Locate: Tensor index out of bounds.");
            return (size_t)location;
        }

        // helpers for tensor operations
        bool CanFlatten(size_t k) const     // can dims k and k-1 be flattened into a single vector? (do they form a matrix without stride)
        {
            if (k == 0)
                LogicError("CanFlatten() must not be called for index [0].");
            else if (k >= size())                       // it's OK to test bottom-lessly expanded dimensions
                return true;
            if (m_dims[k] == 1 || m_dims[k - 1] == 1)   // both are broadcasting or scalar--we don't care about stride in this case
                return true;
            else
                return m_strides[k] == m_strides[k - 1] * (ptrdiff_t)m_dims[k - 1];
        }
        // editing functions
        // These all create new TensorShape objects.
        TensorShape Flatten(size_t k) const  // flatten [k] with [k-1]
        {
            TensorShape result = *this;
            if (!CanFlatten(k))
                LogicError("Flatten() cannot flatten dimensions with gaps");
            // We reshape local (I x J) sub-matrices to (1 x I*J) sub-matrices.
            // We merge to right so that we can merge multiple by looping left-to-right.
            //   m_dims    =   I   J    K     L
            //   m_strides =   1   I    I*J   I*J*K
            // flattening J and K
            //   m_dims    =   I   1    J*K   L
            //   m_strides =   1   I    I     I*J*K
            // TODO: rethink whether this is correct for example of negative strides
            result.m_dims[k] *= result.m_dims[k - 1];
            result.m_dims[k - 1] = 1;
            result.m_strides[k] = /*result.m_dims[k - 1] *, it's 1 */ result.m_strides[k - 1];
            return result;
        }
        TensorShape DropDims(const std::vector<bool> & toDrop) const  // remove dimension
        {
            // this deletes a dimension while retaining strides
            // This implies a slice to [0] for this dimension.
            TensorShape result = *this;
            size_t j = 0;
            for (size_t k = 0; k < size(); k++)
            {
                if (toDrop[k])
                    continue;
                else
                {
                    // example
                    //   m_dims    =   I   1    J   K
                    //   m_strides =   1   I    I   I*J
                    // dropping the second dimension
                    //   m_dims    =   I   %    J   K
                    //   m_strides =   1   %    I   I*J
                    result.m_dims[j] = result.m_dims[k];
                    result.m_strides[j] = result.m_strides[k];
                    j++;
                }
            }
            result.m_dims.resize(j);
            result.m_strides.resize(j);
            return result;
        }
        TensorShape WithBroadcastStrides() const  // flatten [k] with [k-1] if toFlatten[k] is set
        {
            TensorShape result = *this;
            for (size_t k = 0; k < size(); k++)
                if (result.m_dims[k] == 1)
                    result.m_strides[k] = 0;
            return result;
        }
        TensorShape Pad(size_t numDims) const               // append singleton dimensions
        {
            VerifyIsDense();
            if (numDims < GetNumDims())
                LogicError("Pad() cannot drop a shorten the dimensions.");
            else if (numDims == GetNumDims())
                return *this;
            auto dims = GetDims();
            dims.resize(numDims, 1);
            return TensorShape(dims);
        }
        TensorShape Concat(const TensorShape & other) const // concatenate
        {
            auto dims = GetDims();
            auto otherDims = other.GetDims();
            dims.insert(dims.end(), otherDims.begin(), otherDims.end());
            return TensorShape(dims);
        }

        // pretty-printing. Returns tensor dims in the form "I x J x K".
        operator std::string() const
        {
            std::string s;
            for (size_t k = 0; k < size(); k++)
            {
                if (!s.empty())
                    s.append(" x ");
                s.append(std::to_string(m_dims[k]));
            }
#ifdef _DEBUG   // also emit the strides, easier for debugging
            s.append(" {");
            for (size_t k = 0; k < size(); k++)
            {
                if (k > 0)
                    s.append(",");
                s.append(std::to_string(m_strides[k]));
            }
            s.append("}");
#endif
            return s;
        }

    private:
        // reset m_strides and m_offset to represent a canonical no-strides tensor
        void InitAsNoSlice()
        {
            m_offset = 0;
            m_strides.resize(m_dims.size());
            for (size_t k = 0; k < m_dims.size(); k++)
                m_strides[k] = k > 0 ? m_strides[k - 1] * (ptrdiff_t)m_dims[k - 1] : 1;
            m_allocation = m_dims.empty() ? 0 : m_dims.back() * (size_t)m_strides.back();
        }

    private:
        std::vector<size_t> m_dims;     // dimensions of tensor or tensor slice. The size of the box.
        std::vector<ptrdiff_t> m_strides; // dimension gets multiplied by this for computing the index offset. How to hop to the next element in dimension[k]. Stride magic happening here!
        size_t m_offset;                // offset to element(0,0,...,0). May be non-0 in case of slicing.
        size_t m_allocation;            // allocation size of original dense tensor
        // For a regular tensor, there are no strides, m_strides[k] = m_strides[k-1] * m_dims[k-1]. This is how TensorShapes are created from dimensions.
        // For views into existing tensors, we do stride shenanigans to implement broadcasting (plus magic tricks). Examples:
        // To traverse a 5 x 10 matrix with column order reversed:
        //  - op.dims = (5 x 10)
        //  - m_offset points to element (0,9)
        //  - m_strides = (1, -5)       // backward iteration over columns
        // To compute matrix C(13 x 42) = vector A(13 x 1) + matrix B(13 x 42):
        //  - op = sum
        //  - op.dims = (13 x 42)
        //  - C.m_strides = (1, 13)     // forward iteration over columns of B--defines the for loop
        //  - B.m_strides = (1, 13)     // forward iteration over columns of B--iterates in sync with C
        //  - A.m_strides = (1, 0)      // A, however, is stuck in column 0 forever
        // Matrix product: C(I x K) = A(I x J) * B(J x K)   --Note: Likely not RAM-bandwidth efficient!
        //  - op = mul
        //  - op.dims   = (I x J x K)   // iteration dimensions
        //  - C.m_strides = (1, 0, I)   // inverse broadcasting for inner dimension
        //  - A.m_strides = (1, I, 0)
        //  - B.m_strides = (0, 1, J)
        // Convolution of time signals (without padding): Y(T-N+1) = X(T) * H(N):   --Note: Likely not RAM-bandwidth efficient!
        //  - op = mul
        //  - op.dims   = (T-N+1 x N)   // iteration dimensions
        //  - Y.m_strides = (1, 0)      // inverse broadcasting: this sums up the individual products
        //  - X.m_strides = (1, 1)      // shift window by 1 for each output sample
        //  - H.m_strides = (0, -1)     // reuse for each output sample; iterate in reverse order for convolution
        //  - H.m_offset = N - 1        // begin with last element (reverse order for convolution)
        // TODO: double-check all these
        // TODO: Does the same trick work for 2D images?
    };

    // When constructing an image tensor with the usual W, H, C format, use the following function instead.
    // This will sort the three parameters into the correct order.
    // BUGBUG: at several places, a comment says "after multiplication the structure is lost" and the vector dimension
    //         is set as the image height. However, the image height is actually the wrong dimension since images are assumed transposed.
    //         This will get fixed once we get more complete arbitrary tensor support throughout, including better-defined inference rules.
    static inline TensorShape ImageLayoutWHC(size_t width, size_t height, size_t channels)
    {
        return TensorShape(channels, width, height);
    }
    // and use this one when the data is a plain vector
    static inline TensorShape ImageLayoutVector(size_t n)
    {
        return TensorShape(1, 1, n);    // for now storing it as a 3D object as well  --TODO: fix this
    }
    // TODO: we need a constructor from config; that will allow us to generalize

}}}
