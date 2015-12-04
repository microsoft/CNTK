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
    // ImageLayout -- tensor descriptor to describe the inner layout of a data vector that holds a tensor
    //
    // Minibatches are stored as Matrices. While the column dimension represents multiple data vectors, and may have
    // an inner structure (time, parallel sequences) described by the MBLayout, the row dimension represents data
    // vectors that hold tensors of data.
    //
    // The ImageLayout describes the inner tensor structure of these vectors, as a column-major tensor of arbitrary number of dimensions.
    //
    // Specifically, when the image is an image, then this is a 3-dimensional tensor with dimensions ( channels, width, height ),
    // which represents the column-major interpretation of a transposed row-by-row-scanned image where each pixel stores (R,G,B) as a float3.
    //
    // BUGBUG: Tensors with other than 3 dimensions can currently not be used because they cannot be serialized with the current file format.
    // -----------------------------------------------------------------------

    // TODO: really support lengths other than 3, e.g. fix serialization code to handle variable-length descriptors
    // TODO: rename to DataLayout
    // TODO: must match ComputationNode::m_numRows; or, rather, the ImageLayout is how m_numRows is stored??
    // TODO: move this elsewhere, maybe a separate header Tensors.h?
    struct ImageLayout
    {
    public:
        // BUGBUG: This initialization is not correct. This must match GetNumRows(). We probably cannot have all three members here.
        // Idea: We could construct this thing with a ref to the enclosing ComputationNode, and replace 'width' by an expression.
        ImageLayout() : m_tensorDims(3, 1) { }
        template<class VEC>
        ImageLayout(const VEC & dims) { m_tensorDims.reserve(dims.size()); m_tensorDims.assign(dims.begin(), dims.end()); }
        ImageLayout(std::vector<size_t> && dims) : m_tensorDims(std::move(dims)) { }

        void Invalidate() { m_tensorDims.assign(3, SIZE_MAX); } // TODO: clean up the valid/invalid situation (this is currently done inconsistently)

        // TODO: need move constructor/assignment?

        bool operator==(const ImageLayout & other) const { return m_tensorDims == other.m_tensorDims; }

        void SaveToFile(File& fstream) const
        {
#if 1
            // saving as 32-bit ints. This allows to continue to support the old format (size_t W, H, C)
            fstream << (uint32_t)m_tensorDims.size();
            for (auto dim : m_tensorDims)
            {
                if (dim > UINT32_MAX)
                    LogicError("ImageLayout::SaveToFile(): Tensor dimension out of bounds (> 4G).");
                fstream << (uint32_t)dim;
            }
#else
            // TODO: need to use a generic format
            assert(m_tensorDims.size() == 3);   // current format does not understand anything else
            fstream << m_tensorDims[1] << m_tensorDims[2] << m_tensorDims[0]; // currently stored in order W, H, C. TODO: general tensor format will be different
#endif
        }
        void LoadFromFile(File& fstream)
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
    static inline ImageLayout ImageLayoutWHC(size_t width, size_t height, size_t channels)
    {
        return ImageLayout(std::vector<size_t> { channels, width, height });
    }
    // and use this one when the data is a plain vector
    static inline ImageLayout ImageLayoutVector(size_t n)
    {
        return ImageLayout(std::vector<size_t> { 1, 1, n });    // for now storing it as a 3D object as well  --TODO: fix this
    }
    // TODO: we need a constructor from config; that will generalize

}}}
