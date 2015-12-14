// DataTensor.h -- tensor descriptor that describes the inner structure of data vectors
//
// <copyright file="Sequences.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "Basics.h"
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // TensorShape -- tensor descriptor to describe the inner layout of a sample vector that holds a tensor
    //
    // Minibatches are stored as Matrix objects. While the column dimension represents multiple sample vectors, and may have
    // an inner structure (time, parallel sequences) described by the MBLayout, the row dimension represents data
    // vectors that hold tensors of data.
    //
    // The TensorShape describes the inner tensor structure of these vectors, as a column-major tensor of arbitrary number of dimensions.
    //
    // Specifically, when the image is an image, then this is a 3-dimensional tensor with dimensions ( channels, width, height ),
    // which represents the column-major interpretation of a transposed row-by-row-scanned image where each pixel stores (R,G,B) as a float3.
    // -----------------------------------------------------------------------

    // Plans for improved tensor support:
    //
    // TensorShape support for:
    //  - column-major arbitrary-dimension arrays  --this is already implemented
    //  - strides for storage, allowing
    //     - slicing
    //  - strides for computation, allowing
    //     - broadcasting (stride = 0)
    //     - stride magic such as inverting index order or convolution
    //  - insertion and dropping of 1-dimension (cf. 'new_axis' in numpy)
    //
    // Relation to Matrix and MBLayout:
    //  - tensors are stored in Matrix objects
    //  - both matrix row and column dimensions are interpreted as tensor dimensions
    //     - row dimension is explained by a TensorShape ComputationNode::SampleLayout
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
    // Operations:
    //  - all elementwise operations:
    //     - dimensions are expanded as explained above for all operands
    //     - of note: result may also have broadcasting dimensions
    //     - elementwise 'copy' is also considered here, which allows for strided copies
    //  - inner product (Kronecker product+contraction) -> TimesNode
    //     - implementable as SGEMM (may extend in the future)
    //  - tensor transpose -> TransposeNode
    //     - swaps any two dimensions. This does not change the column-major definition, i.e. requires a memory copy.
    //     - special case: swapping between sample and MBLayout, e.g. turn a sample dimension to a time dimension

    // TODO: must match ComputationNode::m_numRows; or, rather, the TensorShape is how m_numRows is stored??
    struct TensorShape
    {
    public:
        // BUGBUG: This initialization is not correct. This must match GetNumRows(). We probably cannot have all three members here.
        // Idea: We could construct this thing with a ref to the enclosing ComputationNode, and replace 'width' by an expression.
        TensorShape() : m_tensorDims(3, 1) { }
        template<class VEC>
        TensorShape(const VEC & dims) { m_tensorDims.reserve(dims.size()); m_tensorDims.assign(dims.begin(), dims.end()); }
        TensorShape(std::vector<size_t> && dims) : m_tensorDims(std::move(dims)) { }

        void Invalidate() { m_tensorDims.assign(3, SIZE_MAX); } // TODO: clean up the valid/invalid situation (this is currently done inconsistently)

        // TODO: need move constructor/assignment?

        bool operator==(const TensorShape & other) const { return m_tensorDims == other.m_tensorDims; }

        void Save(File& fstream) const
        {
#if 1
            // saving as 32-bit ints. This allows to continue to support the old format (size_t W, H, C)
            fstream << (uint32_t)m_tensorDims.size();
            for (auto dim : m_tensorDims)
            {
                if (dim > UINT32_MAX)
                    LogicError("TensorShape::Save(): Tensor dimension out of bounds (> 4G).");
                fstream << (uint32_t)dim;
            }
#else
            // TODO: need to use a generic format
            assert(m_tensorDims.size() == 3);   // current format does not understand anything else
            fstream << m_tensorDims[1] << m_tensorDims[2] << m_tensorDims[0]; // currently stored in order W, H, C. TODO: general tensor format will be different
#endif
        }
        void Load(File& fstream)
        {
#if 1
            // format: uint32_t n, dim[0], dim[1], ..., dim[n-1]
            // We are also able to read (but not write) an older format, which stores 3-dimensional tensors as size_t W, H, C
            uint32_t n, dim;
            fstream >> n >> dim;
            if (dim)        // heuristic to detect the old format. Old format stores a size_t, i.e. the second uint32_t is 0 (no dimensions are > 4G)
            {
                m_tensorDims.resize(n);
                m_tensorDims[0] = dim;
                for (size_t i = 1; i < n; i++)
                {
                    fstream >> dim;
                    m_tensorDims[i] = dim;
                }
                assert(n == m_tensorDims.size());
            }
            else            // detected the old size_t W, H, C format
            {
                m_tensorDims.resize(3);     // current format is hard-coded for 3, for back compat
                m_tensorDims[1] = n;
                fstream >> m_tensorDims[2] >> m_tensorDims[0]; // currently stored in order W, H, C. TODO: general tensor format will be different
        }
#else
            // TODO: need to use a generic format
            m_tensorDims.resize(3);     // current format is hard-coded for 3, for back compat
            fstream >> m_tensorDims[1] >> m_tensorDims[2] >> m_tensorDims[0]; // currently stored in order W, H, C. TODO: general tensor format will be different
#endif
        }

        // accessors
        size_t GetDim(size_t k) const { return m_tensorDims[k]; }
        size_t GetNumDims() const { return m_tensorDims.size(); }
        size_t GetNumElements() const { size_t res = 1; for (auto & dim : m_tensorDims) res *= dim; return res; }

        const std::vector<size_t> & GetDims() const { return m_tensorDims; }    // get all, e.g. for logging or for constructing derived tensors with edited dimensions

        // interpretation as an image tensor
        size_t GetNumChannels() const { return m_tensorDims[0]; }
        size_t GetWidth()       const { return m_tensorDims[1]; }
        size_t GetHeight()      const { return m_tensorDims[2]; }

    private:
        std::vector<size_t> m_tensorDims;
    };

    // When constructing an image tensor with the usual W, H, C format, use the following function instead.
    // This will sort the three parameters into the correct order.
    // BUGBUG: at several places, a comment says "after multiplication the structure is lost" and the vector dimension
    //         is set as the image height. However, the image height is actually the wrong dimension since images are assumed transposed.
    //         This will get fixed once we get more complete arbitrary tensor support throughout, including better-defined inference rules.
    static inline TensorShape ImageLayoutWHC(size_t width, size_t height, size_t channels)
    {
        return TensorShape(std::vector<size_t> { channels, width, height });
    }
    // and use this one when the data is a plain vector
    static inline TensorShape ImageLayoutVector(size_t n)
    {
        return TensorShape(std::vector<size_t> { 1, 1, n });    // for now storing it as a 3D object as well  --TODO: fix this
    }
    // TODO: we need a constructor from config; that will allow us to generalize

}}}
