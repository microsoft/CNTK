//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Contains helper classes used in both defining the CNTKLibrary.h APIs and internal code.
//

#pragma once

#include <vector>
#include <list>
#include <forward_list>
#include <deque>
#include <iterator>
#include <utility> // std::forward

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

///
/// A collection wrapper class that performs a map ("transform") operation given a lambda.
///
template<typename CollectionType, typename Lambda>
class TransformingSpan
{
    typedef typename CollectionType::value_type T;
    typedef typename CollectionType::const_iterator CollectionConstIterator;
    typedef typename std::iterator_traits<CollectionConstIterator>::iterator_category CollectionConstIteratorCategory;
    typedef decltype(std::declval<Lambda>()(std::declval<const T&>())) TLambda; // type of result of lambda call
    typedef typename std::remove_reference<TLambda>::type TLambdaValue;
    typedef typename std::remove_cv<TLambdaValue>::type TLambdaValueNonConst;
    CollectionConstIterator beginIter, endIter;
    const Lambda& lambda;
public:
    typedef TLambda value_type;
    TransformingSpan(const CollectionType& collection, const Lambda& lambda) : beginIter(collection.cbegin()), endIter(collection.cend()), lambda(lambda) { }
    TransformingSpan(const CollectionConstIterator& beginIter, const CollectionConstIterator& endIter, const Lambda& lambda) : beginIter(beginIter), endIter(endIter), lambda(lambda) { }
    // transforming iterator
    class const_iterator : public std::iterator<CollectionConstIteratorCategory, TLambdaValue>
    {
        const Lambda& lambda;
        CollectionConstIterator argIter;
    public:
        const_iterator(const CollectionConstIterator& argIter, const Lambda& lambda) : argIter(argIter), lambda(lambda) { }
        const_iterator operator++() { auto cur = *this; argIter++; return cur; }
        const_iterator operator++(int) { argIter++; return *this; }
        TLambda operator*() { return lambda(*argIter); }
        auto operator->() { return &operator*(); }
        bool operator==(const const_iterator& other) { return argIter == other.argIter; }
        bool operator!=(const const_iterator& other) { return argIter != other.argIter; }
        const_iterator operator+(difference_type offset) { return const_iterator(argIter + offset, lambda); }
        const_iterator operator-(difference_type offset) { return const_iterator(argIter - offset, lambda); }
        difference_type operator-(const const_iterator& other) { return argIter - other.argIter; }
    };
    const_iterator cbegin() const { return const_iterator(beginIter, lambda); }
    const_iterator cend()   const { return const_iterator(endIter  , lambda); }
    const_iterator begin()  const { return cbegin(); }
    const_iterator end()    const { return cend();   }
    // construct certain collection types directly
    auto as_vector()       const { return std::vector<TLambdaValueNonConst>(cbegin(), cend()); }
    auto as_list()         const { return std::list  <TLambdaValueNonConst>(cbegin(), cend()); }
    auto as_forward_list() const { return std::forward_list  <TLambdaValueNonConst>(cbegin(), cend()); }
    auto as_deque()        const { return std::deque <TLambdaValueNonConst>(cbegin(), cend()); }
    operator std::vector      <TLambdaValueNonConst>() const { return as_vector(); }
    operator std::list        <TLambdaValueNonConst>() const { return as_list(); }
    operator std::forward_list<TLambdaValueNonConst>() const { return as_forward_list(); }
    operator std::deque       <TLambdaValueNonConst>() const { return as_deque(); }
};
// main entry point
// E.g. call as Transform(collection, lambda1, lambda2, ...).as_vector();
template<typename CollectionType, typename Lambda>
static inline auto Transform(const CollectionType& collection, const Lambda& lambda) { return TransformingSpan<CollectionType, Lambda>(collection, lambda); }
template<typename CollectionType, typename Lambda, typename ...MoreLambdas>
static inline auto Transform(const CollectionType& collection, const Lambda& lambda, MoreLambdas&& ...moreLambdas) { return Transform(TransformingSpan<CollectionType, Lambda>(collection, lambda), std::forward<MoreLambdas>(moreLambdas)...); }

} // namespace
