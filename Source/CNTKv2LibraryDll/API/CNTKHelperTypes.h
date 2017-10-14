//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Contains helper classes used in both defining the CNTKLibrary.h APIs and internal code.
//

#pragma once

namespace CNTK
{

///
/// Represents a slice view onto any array with consecuytive elements such as a constant-size std::vector.
/// A future C++ standard may have std::span, which this attempts to resemble.
///
template<typename T>
class VectorSpan
{
protected:
    T *bufp, *endp; // TODO: How about debugging? Use a fake data structure of a few hundred elements?
    T* Locate(size_t index) const
    {
        T* itemp = const_cast<T*>(begin()) + index;
        if (itemp >= end())
            LogicError("index out of bounds");
        return itemp;
    }
public:
    // can be instantiated from any vector
    // We don't preserve const-ness for this class. Wait for a proper STL version of this :)
    VectorSpan(const std::vector<T>& vec) : bufp(const_cast<T*>(vec.data())), endp(const_cast<T*>(vec.data()) + vec.size()) { }
    // Cannot be copied. Pass this as a reference only, to avoid ambiguity.
    VectorSpan(const VectorSpan&) = delete; void operator=(const VectorSpan&) = delete;
    // It can be assigned back into a vector. This creates a copy.
    operator std::vector<T>() const { return std::vector<T>(begin(), end()); }
    // the vector interface
    const T* begin()                  const { return reinterpret_cast<T*>(bufp); }
    T*       begin()                        { return reinterpret_cast<T*>(bufp); }
    const T* end()                    const { return reinterpret_cast<T*>(endp); }
    T*       end()                          { return reinterpret_cast<T*>(endp); }
    const T* cbegin()                 const { return begin(); }
    const T* cbegin()                       { return begin(); }
    const T* cend()                   const { return end(); }
    const T* cend()                         { return end(); }
    const T* data()                   const { return begin(); }
    T*       data()                         { return begin(); }
    const T& front()                  const { return *begin(); }
    T&       front()                        { return *begin(); }
    const T& back()                   const { return *(end() - 1); }
    T&       back()                         { return *(end() - 1); }
    size_t   size()                   const { return end() - begin(); }
    const T& at(size_t index)         const { return *Locate(index); }
    T&       at(size_t index)               { return *Locate(index); }
    const T& operator[](size_t index) const { return at(index); }
    T&       operator[](size_t index)       { return at(index); }
};

}
