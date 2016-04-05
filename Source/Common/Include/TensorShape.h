//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// TensorShape.h -- tensor descriptor that describes the inner structure of data vectors
//

#pragma once

#include "Basics.h"
#include "File.h"
#include <vector>
#include <string>
#include <array>

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
//  - tensor transpose -> TransposeDimensionsNode
//     - swaps any two dimensions. This does not change the column-major definition, i.e. requires a memory copy.
//     - special case: swapping between sample and MBLayout, e.g. turn a sample dimension to a time dimension
//  - Validate() stage will automatically infer tensor dimensions from inputs, and also infer downwards into LearnableParameters where requested
//
// Interfacing to and inplementation in Matrix lib:
//  - a Tensor is realized as a type TensorView = { Matrix&, TensorShape& } (i.e. tensors don't own their memory)
//  - Matrix lib will contain overloads for relevant operations that take Tensor& instead of Matrix&.
//  - elementwise ops will go through a single bottleneck function that deals with matching dimensions (extend, broadcast) and flattening

#if 1
template <typename T>
class SmallVector
{
    T m_data[12];
    size_t m_size;
#ifdef _DEBUG
    static const char defaultUnusedValue = std::numeric_limits<T>::is_signed ? -1 : 0;
    void DebugWipe() // initialize to 0 or -1 to make it easier to parse visually in a debugger
    {
        memset(m_data, defaultUnusedValue, sizeof(m_data));
    }
#else
    void DebugWipe()
    {
    }
#endif
public:
    size_t capacity() const
    {
        return _countof(m_data);
    }
    size_t size() const
    {
        return m_size;
    }
    const T* data() const
    {
        return m_data;
    }
    void clear()
    {
        m_size = 0;
    }
    void push_back(const T& val)
    {
        if (m_size >= capacity())
            LogicError("SmallVector: push_back() exceeded capacity of %d", (int) capacity());
        m_data[m_size++] = val;
    }
    void pop_back()
    {
        if (m_size == 0)
            LogicError("SmallVector: pop_back() called on empty vector");
        m_size--;
#ifdef _DEBUG
        m_data[m_size] = (T)defaultUnusedValue; // make this easier to parse in the debugger
#endif
    }
    void resize(size_t sz, const T& val)
    {
        if (sz < m_size)
            m_size = sz;
        else
            while (m_size < sz)
                push_back(val);
    }
    void assign(size_t sz, const T& val)
    {
        clear();
        resize(sz, val);
    }
    template <class ITER>
    void append(ITER beg, const ITER& end)
    {
        while (beg != end)
            push_back((T) *beg++);
    } // typecast allows signed/unsigned conversions
    template <class ITER>
    void assign(ITER beg, const ITER& end)
    {
        clear();
        append(beg, end);
    }
    void operator=(const SmallVector& other)
    {
        m_size = other.m_size;
        memcpy(m_data, other.m_data, other.m_size * sizeof(T));
    }
    SmallVector(const SmallVector& other)
    {
        DebugWipe();
        *this = other;
    }
    SmallVector(size_t sz, const T& val)
    {
        DebugWipe();
        assign(sz, val);
    }
    SmallVector(size_t sz)
        : SmallVector(sz, 0)
    {
    }
    SmallVector()
        : SmallVector(0)
    {
    }
    SmallVector(const std::vector<T>& v)
    {
        DebugWipe();
        assign(v.begin(), v.end());
    }
    SmallVector(const std::initializer_list<T>& l)
    {
        DebugWipe();
        assign(l.begin(), l.end());
    }
    bool operator==(const SmallVector& other) const
    {
        return size() == other.size() && !memcmp(data(), other.data(), other.m_size * sizeof(T));
    }
    bool operator!=(const SmallVector& other) const
    {
        return !operator==(other);
    } // duh
    T operator[](size_t i) const
    {
        if (i >= size())
            LogicError("SmallVector: index overflow");
        return m_data[i];
    }
    T& operator[](size_t i)
    {
        if (i >= size())
            LogicError("SmallVector: index overflow");
        return m_data[i];
    }
    const T* begin() const
    {
        return data();
    }
    const T* end() const
    {
        return data() + size();
    }
    T back() const
    {
        if (empty())
            LogicError("SmallVector: back() called on empty vector");
        return m_data[m_size - 1];
    }
    T& back()
    {
        if (empty())
            LogicError("SmallVector: back() called on empty vector");
        return m_data[m_size - 1];
    }
    bool empty() const
    {
        return size() == 0;
    }
    void resize(size_t sz)
    {
        resize(sz, 0);
    }
};
#else
template <typename T>
class SmallVector : vector<T>
{
    typedef vector<T> Base;

public:
    SmallVector()
    {
    }
    SmallVector(SmallVector&& other)
        : Base(std::move(other))
    {
    }
    SmallVector(const SmallVector& other)
        : Base(other)
    {
    }
    SmallVector(size_t sz)
        : Base(sz)
    {
    }
    SmallVector(size_t sz, const T& val)
        : Base(sz, val)
    {
    }
    SmallVector(const std::initializer_list<T>& l)
        : Base(l)
    {
    }
    SmallVector(const std::vector<T>& v)
        : Base(v)
    {
    }
    template <class ITER>
    void assign(const ITER& beg, const ITER& end)
    {
        Base::assign(beg, end);
    }
    void assign(size_t sz, const T& val)
    {
        Base::assign(sz, val);
    }
    template <class ITER>
    void append(ITER beg, const ITER& end)
    {
        Base::insert(Base::end(), beg, end);
    }
    void push_back(const T& val)
    {
        Base::push_back(val);
    }
    size_t size() const
    {
        return Base::size();
    }
    bool empty() const
    {
        return size() == 0;
    }
    void resize(size_t sz)
    {
        Base::resize(sz);
    }
    void resize(size_t sz, const T& val)
    {
        Base::resize(sz, val);
    }
    const T* begin() const
    {
        return Base::data();
    }
    const T* end() const
    {
        return Base::data() + size();
    }
    const T& back() const
    {
        return Base::back();
    }
    void operator=(const SmallVector& other)
    {
        Base::operator=(other);
    }
    bool operator==(const SmallVector& other) const
    {
        return (const Base&) *this == (const Base&) other;
    }
    T operator[](size_t i) const
    {
        return Base::operator[](i);
    }
    T& operator[](size_t i)
    {
        return Base::operator[](i);
    }
};
#endif

struct TensorShape
{
public:
    // -----------------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------------

    // main constructor (from vector that holds dimensions)
    template <size_t N>
    TensorShape(const std::array<size_t, N>& dims)
    {
        m_dims.assign(dims.begin(), dims.end());
        InitAsNoSlice();
    }
    TensorShape(const SmallVector<size_t>& dims)
    {
        m_dims.assign(dims.begin(), dims.end());
        InitAsNoSlice();
    }
    TensorShape(SmallVector<size_t>&& dims)
        : m_dims(std::move(dims))
    {
        InitAsNoSlice();
    }

    // convenience constructors, e,g. for test code
    explicit TensorShape(size_t I)                                : TensorShape(SmallVector<size_t>{I}) { }
    TensorShape(size_t I, size_t J)                               : TensorShape(SmallVector<size_t>{I, J}) { }
    TensorShape(size_t I, size_t J, size_t K)                     : TensorShape(SmallVector<size_t>{I, J, K}) { }
    TensorShape(size_t I, size_t J, size_t K, size_t L)           : TensorShape(SmallVector<size_t>{I, J, K, L}) { }
    TensorShape(size_t I, size_t J, size_t K, size_t L, size_t M) : TensorShape(SmallVector<size_t>{I, J, K, L, M}) { }

    // default constructor
    TensorShape()
    {
        InitAsNoSlice();
    }

    // boilerplate
    bool operator==(const TensorShape& other) const { return m_dims == other.m_dims; }
    bool operator!=(const TensorShape& other) const { return !operator==(other); } // duh!

    // verify that this refers to a dense matrix (no strides)
    void VerifyIsDense() const
    {
        for (size_t k = 0; k < m_dims.size(); k++) // (TODO: we can save one multiplication here)
        {
            ptrdiff_t stride = k > 0 ? m_strides[k - 1] * (ptrdiff_t) m_dims[k - 1] : 1;
            if (m_strides[k] != stride)
                LogicError("TensorShape: A dense TensorShape expected. Dimension %d is not.", (int) k);
        }
    }

    // TODO: move the methods in this region under their respective headline
    void Save(File& fstream) const
    {
        VerifyIsDense();
        // saving as 32-bit ints. This allows to continue to support the old format (size_t W, H, C)
        fstream << (uint32_t) m_dims.size();
        for (auto dim : m_dims)
        {
            if (dim > UINT32_MAX)
                LogicError("TensorShape::Save(): Tensor dimensions %s out of bounds (> 4G).", string(*this).c_str());
            fstream << (uint32_t) dim;
        }
    }

    const TensorShape& Load(File& fstream, bool acceptLegacyFormat = false)
    {
        // format: uint32_t n, dim[0], dim[1], ..., dim[n-1]
        // We are also able to read (but not write) an older format, which stores 3-dimensional tensors as size_t W, H, C
        uint32_t rank, dim0;
        fstream >> rank >> dim0;
        if (!acceptLegacyFormat || dim0 != 0) // heuristic to detect the old format. Old format stores a size_t, i.e. the second uint32_t is 0 (no dimensions are > 4G)
        {
            m_dims.resize(rank);
            m_dims[0] = dim0;
            for (size_t i = 1; i < rank; i++)
            {
                fstream >> dim0;
                m_dims[i] = dim0;
            }
            assert(rank == m_dims.size());
        }
        else // detected the old size_t W, H, C format
        {
            m_dims.resize(3);
            m_dims[1] = rank;
            fstream >> m_dims[2] >> m_dims[0]; // stored in order C, W, H
        }
        InitAsNoSlice();
        return *this;
    }

    // -----------------------------------------------------------------------
    // accessors
    // -----------------------------------------------------------------------

    size_t GetDim(size_t k) const { return m_dims[k]; }
    size_t GetDimPadded(size_t k) const { return k < GetRank() ? GetDim(k) : 1; }   // like GetDim() but return 1 for extra (out of bounds) dimensions
    size_t GetRank() const { return m_dims.size(); }
    size_t GetNumElements() const
    {
        if (m_dims.empty())
            return 0;
        size_t res = 1;
        for (auto& dim : m_dims)
            res *= dim;
        return res;
    } // in slice
    size_t GetAllocation() const
    {
        return m_allocation;
    }
    size_t GetOffset() const
    {
        return m_offset;
    }

    // vector-like accessors
    size_t operator[](size_t k) const { return GetDim(k); }
    size_t size() const { return GetRank(); }

    const SmallVector<size_t>& GetDims() const { return m_dims; } // get all, e.g. for logging or for constructing derived tensors with edited dimensions
    const SmallVector<ptrdiff_t>& GetStrides() const { return m_strides; }

    // test whether the tensor represents a column vector (but allowing added broadcasting dimensions)
    // A tensor represents a column vector when all dimensions except the leading are 1.
    bool IsColumnVector() const
    {
        for (size_t k = 1; k < size(); k++)
            if (m_dims[k] != 1)
                return false;
        return true;
    }

    // -----------------------------------------------------------------------
    // indexing
    // -----------------------------------------------------------------------

    // Determines the offset into the underlying element array for a given multi-dimensional index.
    // This function is for reference. Probably not often used.
    size_t Locate(const SmallVector<size_t>& index) const
    {
        ptrdiff_t location = m_offset;
        for (size_t k = 0; k < index.size(); k++)
        {
            size_t dim = k < size() ? m_dims[k] : 1; // dimensions are bottomless
            if (index[k] >= dim)
                LogicError("Locate: Tensor index[%d]=%d exceeds bound %d.", (int) k, (int) index[k], (int) dim);
            if (k < size())
                location += (ptrdiff_t) index[k] * m_strides[k]; // strides may be negative
        }
        if (location < 0 || (size_t) location >= m_allocation)
            LogicError("Locate: Tensor index out of bounds.");
        return (size_t) location;
    }

    // get begin and end location (first offset after last element), for validation purposes
    pair<ptrdiff_t, ptrdiff_t> GetLocationRange() const
    {
        auto result = make_pair(m_offset, m_offset);
        for (size_t k = 0; k < size(); k++)
        {
            ptrdiff_t step = (ptrdiff_t)(m_dims[k] - 1) * m_strides[k];
            if (m_strides[k] > 0) // strides may be negative
                result.second += step;
            else
                result.first += step;
        }
        result.second++;    // max --> end
        return result;
    }

    // -----------------------------------------------------------------------
    // helpers for tensor operations
    // -----------------------------------------------------------------------

    bool CanFlatten(size_t k) const // can dims k and k-1 be flattened into a single vector? (do they form a matrix without stride)
    {
        if (k == 0)
            LogicError("CanFlatten() must not be called for index [0].");
        else if (k >= size()) // it's OK to test bottom-lessly expanded dimensions
            return true;
        if (m_dims[k] == 1 && m_dims[k - 1] == 1) // both are broadcasting or scalar--we don't care about stride in this case
            return true;
        else
            return m_strides[k] == m_strides[k - 1] * (ptrdiff_t) m_dims[k - 1];
    }

    // -----------------------------------------------------------------------
    // editing functions for tensor operations
    // -----------------------------------------------------------------------

    // flatten [k] with [k-1]. Dim[k-1] will be absorbed into [k] and set to 1.
    TensorShape& FlattenInPlace(size_t k)
    {
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
        m_dims[k] *= m_dims[k - 1];
        m_dims[k - 1] = 1;
        m_strides[k] = m_strides[k - 1];
        return *this;
    }
    TensorShape& DropDimsInPlace(const SmallVector<bool>& toDrop) // remove dimension
    {
        // this deletes a dimension while retaining strides
        // This implies a slice to [0] for this dimension.
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
                //   m_dims    =   I        J   K
                //   m_strides =   1        I   I*J
                m_dims[j] = m_dims[k];
                m_strides[j] = m_strides[k];
                j++;
            }
        }
        m_dims.resize(j);
        m_strides.resize(j);
        return *this;
    }
    TensorShape DropDims(const SmallVector<bool>& toDrop) const
    {
        TensorShape result(*this);
        result.DropDimsInPlace(toDrop);
        return result;
    }
    TensorShape& SetBroadcastStrides() // set strides to 0 for broadcasting dimensions
    {
        for (size_t k = 0; k < size(); k++)
            if (m_dims[k] == 1)
                m_strides[k] = 0;
        return *this;
    }
    TensorShape& PadRankInPlace(size_t desiredRank) // append trailing singleton dimensions
    {
        VerifyIsDense();
        if (desiredRank < GetRank()) // can't drop
            LogicError("PadRankInPlace: desiredRank (%d) cannot be less than tensor shape's rank (%d)", (int)desiredRank, (int)GetRank());
        else while (GetRank() < desiredRank) // pad
        {
            m_strides.push_back(GetRank() > 0 ? m_strides.back() * (ptrdiff_t)m_dims.back() : 1);
            m_dims.push_back(1);
        }
        return *this;
    }
    TensorShape& TrimRankInPlace(size_t desiredRank) // drop trailing singleton dimensions
    {
        if (GetRank() < desiredRank) // can't pad
            LogicError("TrimRankInPlace: desiredRank (%d) cannot be higher than tensor shape's rank (%d)", (int)desiredRank, (int)GetRank());
        else while (desiredRank < GetRank()) // drop
        {
            if (m_dims.back() != 1)
                LogicError("TrimRankInPlace() cannot drop non-singleton dimensions.");
            m_strides.pop_back();
            m_dims.pop_back();
        }
        VerifyIsDense(); // (should be OK to drop non-dense singleton dimensions, so check after dropping them)
        return *this;
    }
    TensorShape PadRank(size_t desiredRank) const // append singleton dimensions
    {
        return TensorShape(*this).PadRankInPlace(desiredRank);
    }
    TensorShape& AppendInPlace(size_t rank, size_t newDim) // concatenate one new dimension at position 'rank'
    {
        PadRankInPlace(rank);
        // TODO: How to do this right in case of arbitrary strides? Compute the new stride based on m_allocation or something? Is it even possible? Or do we need to guard?
        m_strides.push_back(GetRank() > 0 ? m_strides.back() * (ptrdiff_t) m_dims.back() : 1);
        m_dims.push_back(newDim);
        m_allocation *= newDim;
        return *this;
    }
    TensorShape Append(size_t rank, size_t newDim) const
    {
        return TensorShape(*this).AppendInPlace(rank, newDim);
    }
    // narrow a dimension k to given bounds [begin, end), done in-place
    TensorShape& NarrowTo(size_t k, size_t begin, size_t end)
    {
        if (k >= size())
            LogicError("NarrowTo: Index out of bounds.");
        if (end <= begin || end > m_dims[k])
            LogicError("NarrowTo: Invalid bounds parameter, dimensions must be at least one.");
        m_offset += m_strides[k] * begin;
        m_dims[k] = end - begin;
        return *this;
    }
    // narrow all dimensions to two given bounds vectors, done in-place
    template <class DimensionVector>
    TensorShape& NarrowTo(const std::pair<DimensionVector, DimensionVector>& bounds /*begin[], end[]*/)
    {
        if (size() != bounds.first.size() || size() != bounds.second.size())
            LogicError("NarrowTo: Bounds parameter must have same rank as tensor.");
        for (size_t k = 0; k < size(); k++)
            NarrowTo(k, (size_t)bounds.first[k], (size_t)bounds.second[k]);
        return *this;
    }
    // swap two existing dimensions (implements transposition)
    // This yields the same tensor but index positions are exchanged.
    // This tensor is now no longer stored as column-major.
    void SwapDimsInPlace(size_t i, size_t j)
    {
        if (i == j) // this is OK
            return;
        std::swap(m_dims[i],    m_dims[j]);
        std::swap(m_strides[i], m_strides[j]);
    }

    // compare two TensorShapes, whether they are compatible, considering padding and broadcasting
    bool IsElementwiseCompatibleWith(const TensorShape& other) const
    {
        for (size_t i = 0; i < m_dims.size(); i++)
        {
            size_t dim = m_dims[i];
            size_t otherDim = i < other.size() ? other[i] : 1;
            if (dim != otherDim && dim != 1 && otherDim != 1) // dims mismatch, and neither is broadcasting
                return false;
        }
        return true;
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
#if 0   // also emit the strides, easier for debugging
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
    // reset m_strides and m_offset to represent a canonical no-strides column-major tensor
    void InitAsNoSlice()
    {
        m_offset = 0;
        m_strides.resize(m_dims.size());
        for (size_t k = 0; k < m_dims.size(); k++)
            m_strides[k] = k > 0 ? m_strides[k - 1] * (ptrdiff_t) m_dims[k - 1] : 1;
        m_allocation = m_dims.empty() ? 0 : m_dims.back() * (size_t) m_strides.back(); // TODO: Or should an empty shape mean it's a scalar?
    }

private:
    SmallVector<size_t> m_dims;       // dimensions of tensor or tensor slice. The size of the box.
    SmallVector<ptrdiff_t> m_strides; // dimension gets multiplied by this for computing the index offset. How to hop to the next element in dimension[k]. Stride magic happening here!
    size_t m_offset;                  // offset to element(0,0,...,0). May be non-0 in case of slicing.
    size_t m_allocation;              // allocation size of original dense tensor

    // A regular tensor is column-major without extra strides: m_strides[k] = m_strides[k-1] * m_dims[k-1]. This is how TensorShapes are created from dimensions.
    // For views into existing tensors, this class allows stride shenanigans to implement broadcasting (plus magic tricks). Examples:
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

// image layouts used in CNTK
// Nodes that do semantic interpretation of width, height, channel information must know which index they are in.
// Eventually this can go away once we switch completely to cudnn layout.
// The cudnn layout is actually our layout in order W,H,C.
enum ImageLayoutKind
{
    HWC, // legacy; default for NDL
    CHW  // cudnn; default for BrainScript
};
static inline std::string ToString(ImageLayoutKind imageLayoutKind)
{
    if (imageLayoutKind == ImageLayoutKind::CHW)
        return "CHW";
    else if (imageLayoutKind == ImageLayoutKind::HWC)
        return "HWC";
    else
        LogicError("ImageLayout: Invalid ImageLayoutKind");
}
static inline ImageLayoutKind ImageLayoutKindFrom(const wstring& s)
{
    if (s == L"CHW" || s == L"cudnn")
        return ImageLayoutKind::CHW;
    else if (s == L"HWC" || s == L"legacy")
        return ImageLayoutKind::HWC;
    else
        InvalidArgument("ImageLayoutKindFrom: Unknown ImageLayoutKind '%ls', must be 'CHW' (cudnn) or 'HWC' (CNTK legacy)", s.c_str());
}

// interpret TensorShape as an image descriptor
// considering that we support two ways of storing images
struct ImageDimensions
{
    size_t m_width, m_height, m_numChannels;
    // convenience accessors. TODO: use only one name. Rename the members themselves?
    size_t w() const { return m_width;       }
    size_t h() const { return m_height;      }
    size_t c() const { return m_numChannels; }

    // interpret TensorShape as image
    ImageDimensions(const TensorShape& shape, ImageLayoutKind imageLayoutKind)
    {
        if (shape.GetRank() != 3)
            InvalidArgument("Convolution operation currently only supports 1D or 2D convolution on 3D tensors.");
        if (imageLayoutKind == ImageLayoutKind::CHW)
        {
            m_width       = shape[0];
            m_height      = shape[1];
            m_numChannels = shape[2];
        }
        else if (imageLayoutKind == ImageLayoutKind::HWC)
        {
            m_width      = shape[1];
            m_height     = shape[2];
            m_numChannels = shape[0];
        }
        else
            LogicError("WHC: Invalid ImageLayoutKind");
    }
    ImageDimensions(size_t width, size_t height, size_t numChannels)
        : m_width(width), m_height(height), m_numChannels(numChannels)
    {
    }
    // intepret image as TensorShape
    static TensorShape AsTensorShape(size_t width, size_t height, size_t numChannels, ImageLayoutKind imageLayoutKind /* = ImageLayoutKind::HWC*/)
    {
        if (imageLayoutKind == ImageLayoutKind::CHW)
            return TensorShape(width, height, numChannels);
        else if (imageLayoutKind == ImageLayoutKind::HWC)
            return TensorShape(numChannels, width, height);
        else
            LogicError("ImageLayout: Invalid ImageLayoutKind");
    }
    TensorShape AsTensorShape(ImageLayoutKind imageLayoutKind)
    {
        return AsTensorShape(m_width, m_height, m_numChannels, imageLayoutKind);
    }
};

}}}
