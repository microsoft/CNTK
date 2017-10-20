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
#include <set>
#include <iterator>
#include <memory>  // std::shared_ptr
#include <utility> // std::forward

namespace CNTK
{

///
/// Represents a slice view onto a container. Meant for use with std::vector.
/// A future C++ standard may have std::span, which we hope to replace this with in the future.
///
template<typename IteratorType>
class Span
{
protected:
    IteratorType beginIter, endIter;
    typedef typename std::iterator_traits<IteratorType>::value_type T;
    typedef typename std::remove_reference<T>::type TValue;
    typedef typename std::remove_cv<TValue>::type TValueNonConst;
public:
    typedef TValue value_type;
    typedef IteratorType const_iterator; // TODO: some template magic to define the correct one
    typedef IteratorType iterator;
    // can be instantiated from any vector
    // We don't preserve const-ness for this class. Wait for a proper STL version of this :)
    Span(const IteratorType& beginIter, const IteratorType& endIter) : beginIter(beginIter), endIter(endIter) { }
    Span(const IteratorType& beginIter, size_t len) : beginIter(beginIter), endIter(beginIter + len) { }
    // Cannot be copied. Pass this as a reference only, to avoid ambiguity.
    Span(const Span&) = delete; void operator=(const Span&) = delete;
    //Span& operator=(Span&& other) { beginIter = std::move(other.beginIter); endIter = std::move(other.endIter); return *this; }
    Span(Span&& other) : beginIter(std::move(other.beginIter)), endIter(std::move(other.endIter)) { }
    // the collection interface
    const IteratorType& begin()       const { return beginIter; }
    const IteratorType& begin()             { return beginIter; }
    const IteratorType& end()         const { return endIter; }
    const IteratorType& end()               { return endIter; }
    const IteratorType& cbegin()      const { return begin(); }
    const IteratorType& cbegin()            { return begin(); }
    const IteratorType& cend()        const { return end(); }
    const IteratorType& cend()              { return end(); }
    const T* data()                   const { return begin(); }
    T*       data()                         { return begin(); }
    const T& front()                  const { return *begin(); }
    T&       front()                        { return *begin(); }
    const T& back()                   const { return *(end() - 1); }
    T&       back()                         { return *(end() - 1); }
    size_t   size()                   const { return cend() - cbegin(); }
    bool     empty()                  const { return cend() == cbegin(); }
    const T& at(size_t index)         const { return *(beginIter + index); }
    T&       at(size_t index)               { return *(beginIter + index); }
    const T& operator[](size_t index) const { return at(index); }
    T&       operator[](size_t index)       { return at(index); }
    // construct certain collection types directly
    explicit operator std::vector      <TValueNonConst>() const { return std::vector      <TValueNonConst>(cbegin(), cend()); }
    explicit operator std::list        <TValueNonConst>() const { return std::list        <TValueNonConst>(cbegin(), cend()); }
    explicit operator std::forward_list<TValueNonConst>() const { return std::forward_list<TValueNonConst>(cbegin(), cend()); }
    explicit operator std::deque       <TValueNonConst>() const { return std::deque       <TValueNonConst>(cbegin(), cend()); }
    explicit operator std::set         <TValueNonConst>() const { return std::set         <TValueNonConst>(cbegin(), cend()); }
    // others
    bool operator==(const Span& other) const
    {
        if (this == &other)
            return true;
        IteratorType b      =       begin();
        IteratorType e      =       end();
        IteratorType bOther = other.begin();
        IteratorType eOther = other.end();
        if (e - b != eOther - bOther) // size must be the same
            return false;
        for (; b != e; b++, bOther++)
            if (*b != *bOther)
                return false; // all elements must be the same
        return true;
    }
    bool operator!=(const Span& other) const { return !operator==(other); }
};
// MakeSpan(collection[, beginIndex[, endIndex]])
template<typename CollectionType>
auto MakeSpan(CollectionType& collection, size_t beginIndex = 0) { return Span<typename CollectionType::iterator>(collection.begin() + beginIndex, collection.end()); }
template<typename CollectionType>
auto MakeSpan(const CollectionType& collection, size_t beginIndex = 0) { return Span<typename CollectionType::const_iterator>(collection.cbegin() + beginIndex, collection.cend()); }
// TODO: Decide what end=0 means.
template<typename CollectionType, typename EndIndexType>
auto MakeSpan(CollectionType& collection, size_t beginIndex, EndIndexType endIndex) { return Span<typename CollectionType::iterator>(collection.begin() + beginIndex, (endIndex >= 0 ? collection.begin() : collection.end()) + endIndex); }
template<typename CollectionType, typename EndIndexType>
auto MakeSpan(const CollectionType& collection, size_t beginIndex, EndIndexType endIndex) { return Span<typename CollectionType::const_iterator>(collection.cbegin() + beginIndex, (endIndex >= 0 ? collection.begin() : collection.end()) + endIndex); }

///
/// A collection wrapper class that performs a map ("transform") operation given a lambda.
///
template<typename CollectionType, typename Lambda>
class TransformingSpan
{
    typedef typename CollectionType::value_type T;
    typedef typename std::conditional<std::is_const<CollectionType>::value, typename CollectionType::const_iterator, typename CollectionType::iterator>::type CollectionIterator; // TODO: template magic to make this const only if the input is const, otherwise ::iterator
    typedef typename std::iterator_traits<CollectionIterator>::iterator_category CollectionIteratorCategory;
    typedef decltype(std::declval<Lambda>()(std::forward<T>(std::declval<T>()))) TLambda; // type of result of lambda call
    typedef typename std::remove_reference<TLambda>::type TValue;
    typedef typename std::remove_cv<TValue>::type TValueNonConst;
    CollectionType& m_collection; // we keep the collection itself, so that we can e.g. call size() on it
    const Lambda& lambda;
    // transforming iterator
    class Iterator : public std::iterator<CollectionIteratorCategory, TValue>
    {
        const Lambda& lambda;
        CollectionIterator argIter;
    public:
        Iterator(const CollectionIterator& argIter, const Lambda& lambda) : argIter(argIter), lambda(lambda) { }
        Iterator operator++() { auto cur = *this; argIter++; return cur; }
        Iterator operator++(int) { argIter++; return *this; }
        TLambda operator*() const { return lambda(std::move(*argIter)); }
        auto operator->() const { return &operator*(); }
        bool operator==(const Iterator& other) const { return argIter == other.argIter; }
        bool operator!=(const Iterator& other) const { return argIter != other.argIter; }
        // BUGBUG: The following won't work for forward iterators, such as NonOwningFunctionList
        Iterator operator+(difference_type offset) const { return Iterator(argIter + offset, lambda); }
        Iterator operator-(difference_type offset) const { return Iterator(argIter - offset, lambda); }
        difference_type operator-(const Iterator& other) const { return argIter - other.argIter; }
    };
public:
    typedef TLambda value_type;
    // note: constness must be contained in CollectionType
    TransformingSpan(CollectionType& collection, const Lambda& lambda) : m_collection(collection), /*beginIter(collection.begin()), endIter(collection.end()),*/ lambda(lambda) { }
    typedef Iterator const_iterator;
    typedef Iterator iterator;
    const_iterator cbegin() const { return const_iterator(m_collection.begin(), lambda); }
    const_iterator cend()   const { return const_iterator(m_collection.end()  , lambda); }
    const_iterator begin()  const { return cbegin(); }
    const_iterator end()    const { return cend();   }
    iterator       begin()        { return iterator(m_collection.begin(), lambda); }
    iterator       end()          { return iterator(m_collection.end()  , lambda); }
    size_t         size()   const { return m_collection.size(); }
    bool           empty()  const { return m_collection.empty(); }
    // construct certain collection types directly
    explicit operator std::vector      <TValueNonConst>() const { return std::vector      <TValueNonConst>(cbegin(), cend()); } // note: don't call as_vector etc., will not be inlined! in VS 2015!
    explicit operator std::list        <TValueNonConst>() const { return std::list        <TValueNonConst>(cbegin(), cend()); }
    explicit operator std::forward_list<TValueNonConst>() const { return std::forward_list<TValueNonConst>(cbegin(), cend()); }
    explicit operator std::deque       <TValueNonConst>() const { return std::deque       <TValueNonConst>(cbegin(), cend()); }
    explicit operator std::set         <TValueNonConst>() const { return std::set         <TValueNonConst>(cbegin(), cend()); }
};
// main entry point
// E.g. call as Transform(collection, lambda1, lambda2, ...).as_vector();
template<typename CollectionType, typename Lambda>
static inline const auto Transform(const CollectionType& collection, const Lambda& lambda) { return TransformingSpan<CollectionType const, Lambda>(collection, lambda); }
template<typename CollectionType, typename Lambda, typename ...MoreLambdas>
static inline const auto Transform(const CollectionType& collection, const Lambda& lambda, MoreLambdas&& ...moreLambdas) { return Transform(TransformingSpan<CollectionType const, Lambda>(collection, lambda), std::forward<MoreLambdas>(moreLambdas)...); }

template<typename CollectionType, typename Lambda>
static inline auto Transform(CollectionType& collection, const Lambda& lambda) { return TransformingSpan<CollectionType, Lambda>(collection, lambda); }
template<typename CollectionType, typename Lambda, typename ...MoreLambdas>
static inline auto Transform(CollectionType& collection, const Lambda& lambda, MoreLambdas&& ...moreLambdas) { return Transform(TransformingSpan<CollectionType, Lambda>(collection, lambda), std::forward<MoreLambdas>(moreLambdas)...); }

///
/// Implement a range like Python's range.
/// Can be used with variable or constant bounds (use IntConstant<val> as the second and third type args).
///
template<int val>
struct IntConstant
{
    constexpr operator int() const { return val; }
};
template<typename T, typename Tbegin = const T, typename Tend = const T>
class NumericRangeSpan
{
    static const T stepValue = (T)1; // for now. TODO: apply the IntConst trick here as well.
    Tbegin beginValue;
    Tend endValue;
    typedef typename std::remove_reference<T>::type TValue;
    typedef typename std::remove_cv<TValue>::type TValueNonConst;
public:
    typedef T value_type;
    NumericRangeSpan(const T& beginValue, const T& endValue/*, const T& stepValue = (const T&)1*/) : beginValue(beginValue), endValue(endValue)/*, stepValue(stepValue)*/ { }
    NumericRangeSpan(const T& endValue) : NumericRangeSpan(0, endValue) { }
    NumericRangeSpan() { }
    // iterator
    class const_iterator : public std::iterator<std::random_access_iterator_tag, TValue>
    {
        T value/*, stepValue*/;
    public:
        const_iterator(const T& value/*, const T& stepValue*/) : value(value)/*,stepValue(stepValue)*/ { }
        const_iterator operator++() { auto cur = *this; value += stepValue; return cur; }
        const_iterator operator++(int) { value += stepValue; return *this; }
        T operator*() const { return value; }
        auto operator->() const { return &operator*(); } // (who knows whether this is defined for the type)
        bool operator==(const const_iterator& other) const { return value == other.value; }
        bool operator!=(const const_iterator& other) const { return value != other.value; }
        const_iterator operator+(difference_type offset) const { return const_iterator(value + offset * stepValue, stepValue); }
        const_iterator operator-(difference_type offset) const { return const_iterator(value - offset * stepValue, stepValue); }
        difference_type operator-(const const_iterator& other) const { return ((difference_type)value - (difference_type)other.value) / stepValue; }
    };
    typedef const_iterator iterator; // in case it gets instantiated non-const. Still cannot modify it of course.
    const_iterator cbegin() const { return const_iterator(beginValue); }
    const_iterator cend()   const { return const_iterator(endValue);   }
    const_iterator begin()  const { return cbegin(); }
    const_iterator end()    const { return cend();   }
    size_t         size()   const { return cend() - cbegin(); }
    bool           empty()  const { return endValue == beginValue; }
    // construct certain collection types directly, to support TransformToVector() etc.
    explicit operator std::vector      <TValueNonConst>() const { return std::vector      <TValueNonConst>(cbegin(), cend()); } // note: don't call as_vector etc., will not be inlined! in VS 2015!
    explicit operator std::list        <TValueNonConst>() const { return std::list        <TValueNonConst>(cbegin(), cend()); }
    explicit operator std::forward_list<TValueNonConst>() const { return std::forward_list<TValueNonConst>(cbegin(), cend()); }
    explicit operator std::deque       <TValueNonConst>() const { return std::deque       <TValueNonConst>(cbegin(), cend()); }
    explicit operator std::set         <TValueNonConst>() const { return std::set         <TValueNonConst>(cbegin(), cend()); }
};

///
/// Assembly-optimized constructors for creating 1- and 2-element std::vector.
/// Note that the embedded iterators only work for std::vector, since operator!= and operator-
/// blindly assume that 'other' is end().
///
template<typename T>
static inline std::vector<T> MakeTwoElementVector(const T& a, const T& b)
{
    class TwoElementSpanIterator : public std::iterator<std::random_access_iterator_tag, T>
    {
        const T* x[2];
    public:
        TwoElementSpanIterator() { } // sentinel
        TwoElementSpanIterator(const T& a, const T& b) { x[0] = &a; x[1] = &b; }
        void operator++() { x[0] = x[1]; x[1] = nullptr; }
        const T& operator*() const { return *x[0]; }
        bool operator!=(const TwoElementSpanIterator&) const { return x[0] != nullptr; }
        /*constexpr*/ difference_type operator-(const TwoElementSpanIterator&) const { return 2; }
        // constexpr has problems with CUDA
    };
    return std::vector<T>(TwoElementSpanIterator(a, b), TwoElementSpanIterator());
}
template<typename T>
static inline std::vector<T> MakeOneElementVector(const T& a)
{
    class OneElementSpanIterator : public std::iterator<std::random_access_iterator_tag, T>
    {
        const T* x;
    public:
        OneElementSpanIterator() { } // sentinel
        OneElementSpanIterator(const T& a) : x(&a) { }
        void operator++() { x = nullptr; }
        const T& operator*() const { return *x; }
        bool operator!=(const OneElementSpanIterator&) const { return x != nullptr; }
        /*constexpr*/ difference_type operator-(const OneElementSpanIterator&) const { return 1; }
    };
    return std::vector<T>(OneElementSpanIterator(a), OneElementSpanIterator());
}

///
/// Helpers to construct the standard STL from the above.
///
template<typename Container> static inline auto MakeVector     (const Container& container) { return std::vector      <typename Container::value_type>(container.cbegin(), container.cend()); }
template<typename Container> static inline auto MakeList       (const Container& container) { return std::list        <typename Container::value_type>(container.cbegin(), container.cend()); }
template<typename Container> static inline auto MakeForwardList(const Container& container) { return std::forward_list<typename Container::value_type>(container.cbegin(), container.cend()); }
template<typename Container> static inline auto MakeDeque      (const Container& container) { return std::deque       <typename Container::value_type>(container.cbegin(), container.cend()); }
template<typename Container> static inline auto MakeSet        (const Container& container) { return std::set         <typename Container::value_type>(container.cbegin(), container.cend()); }

///
/// Class that stores a vector with "small-vector optimization," that is, if it has N or less elements,
/// they are stored in the object itself, without malloc(), while more elements use a std::vector.
///
template<typename T, size_t N>
class FixedVectorWithBuffer : public Span<T*>
{
    union U
    {
        T fixedBuffer[N]; // stored inside a union so that we get away without automatic construction yet correct alignment
        U() {}  // C++ requires these in order to be happy
        ~U() {}
    } u;
    T* Allocate(size_t len) // get or allocate the buffer
    {
        if (len > N)
            return (T*)malloc(len * sizeof(T)); // dynamic case
        else
            return &u.fixedBuffer[0]; // fixed case
    }
    void DestructAndFree(const T* eValid) // call destructor items (of possibly partially constructed range); then free
    {
        T* b = begin();
        const T* e = end();
        for (auto* p = b; p != eValid; p++)
            p->~T();
        if (e > b + N) // (write it this way to avoid the division)
            free(b);   // Span::beginIter holds the result of malloc()
    }
public:
    typedef Span<T*> Span;
    typedef T value_type;
    // short-circuit constructors that construct from up to 2 arguments which are taken ownership of
    FixedVectorWithBuffer()                       : Span(nullptr, nullptr) {} // (pointer value does not matter as long as it is the same; this is a tiny bit faster than passing the actual u.fixedBuffer)
    // construct by moving elements in
    // BUGBUG: These && interfaces should check their length to be 1 and 2, respectively. Can we use template magic to only match these if the correct values are passed as constants?
    FixedVectorWithBuffer(size_t len, T&& a)        : Span(Allocate(len)/*u.fixedBuffer*/, len)
    {
        auto* b = begin();
        new (b) T(std::move(a));
        const auto* e = end();
        for (auto* p = b + 1; p != e; p++) // if more than one element then duplicate  --TODO: once I know how to restrict the first arg to constant 1, remove this again.
            new (p) T(*b);
    }
    FixedVectorWithBuffer(size_t    , T&& a, T&& b) : Span(u.fixedBuffer, 2) { new (&u.fixedBuffer[0]) T(std::move(a)); new (&u.fixedBuffer[1]) T(std::move(b)); } // BUGBUG: This version should only be defined if N > 1. Use template magic.
    FixedVectorWithBuffer(size_t len, const T& a)   : Span(Allocate(len), len)
    {
        auto* b = begin();
        const auto* e = end();
        for (auto* p = b; p != e; p++)
        {
            try
            {
                new (p) T(a);
            }
            catch (...)
            {
                DestructAndFree(/*end of constructed range=*/p);
                throw;
            }
        }
    }
    FixedVectorWithBuffer(size_t, const T& a, const T& b) : Span(u.fixedBuffer, 2)
    {
        new (&u.fixedBuffer[0]) T(a);
        try { new (&u.fixedBuffer[1]) T(b); } // if second one fails, we must clean up the first one
        catch (...) { u.fixedBuffer[0].~T(); throw; }
    } // BUGBUG: This version should only be defined if N > 1.

    // constructor from a collection (other than ourselves, for which we have a specialization below)
    // This constructor steals all elements out from the passed collection, but not the collection's buffer itself.
    // This is meant for the use case where we want to avoid reallocation of the vector, while its members
    // are small movable objects that get created upon each use.
    // This is an unusual interpretation of && (since it only half-destructs the input), but it should be valid.
#define WHERE_IS_TEMPORARY(Type) , typename = std::enable_if<!std::is_lvalue_reference<Type&&>::value>::type
#define WHERE_IS_ITERATOR(Type)  , typename = std::enable_if<!std::is_same<typename std::iterator_traits<Type>::value_type, void>::value>::type
#define WHERE_IS_ITERABLE(Type)  , typename = std::enable_if<!std::is_same<typename Type::const_iterator, void>::value>::type
#define WHERE_IS_BASE_OF(B,D)    , typename = std::enable_if< std::is_convertible<typename D*,typename B*>::value>::type
#define WHERE_IS_NOT_BASE_OF(B,D), typename = std::enable_if<!std::is_convertible<typename D*,typename B*>::value>::type
    // BUGBUG: This is still not correct. It also test whether the iterator is temporary. Otherwise we must not move stuff out.
    template<typename Collection WHERE_IS_TEMPORARY(Collection)> // move construction from rvalue [thanks to Billy O'Neal for the tip]
    explicit FixedVectorWithBuffer(Collection&& other) : Span(Allocate(other.size()), other.size())
    {
        auto* b = begin();
        const auto* e = end();
        auto otherIter = other.begin();
        for (auto* p = b; p != e; p++)
        {
            new (p) T(std::move(*otherIter)); // nothrow
            otherIter++;
        }
    }

    template<typename Collection WHERE_IS_ITERABLE(Collection)/*, typename = std::enable_if_t<std::is_lvalue_reference_v<Collection&&>>*/> // copy construction from lvalue --TODO: Verify: is this ever called? And is the other ever called?
    explicit FixedVectorWithBuffer(const Collection& other) : FixedVectorWithBuffer(other.begin(), other.end()) { }

    // construct from iterator pair. All variable-length copy construction/assignment (that is, except for C(2,a,b)) goes through this.
    template<typename CollectionIterator WHERE_IS_ITERATOR(CollectionIterator)>
    FixedVectorWithBuffer(const CollectionIterator& beginIter, const CollectionIterator& endIter) :
        Span(Allocate(endIter - beginIter), endIter - beginIter)
    {
        auto* b = begin();
        const auto* e = end();
        auto otherIter = beginIter;
        for (auto* p = b; p != e; p++)
        {
            try
            {
                new (p) T(*otherIter);
                otherIter++;
            }
            catch (...) // in case of error must undo partially constructed vector
            {
                DestructAndFree(/*end of constructed range=*/p);
                throw;
            }
        }
    }

    FixedVectorWithBuffer(FixedVectorWithBuffer&& other) : Span(other.size() > N ? other.begin() : u.fixedBuffer, other.size())
    {
        // if dynamic, we are done; if static, we must move the elements themselves
        auto* b = begin();
        const auto* e = end();
        if (e <= b + N) // static case (if dynamiy
        {
            auto otherIter = other.begin();
            for (auto* p = b; p != e; p++)
            {
                auto& otherItem = *otherIter;
                new (p) T(std::move(otherItem)); // steal the item
                otherItem.~T();                  // destruct right away to allow optimizer to short-circuit it
                otherIter++;
            }
        }
        ((Span&)other).~Span();                     // other is now empty, all elements properly destructed (this call does nothing actually)
        new ((Span*)&other) Span(nullptr, nullptr); // and construct other to empty
    }

    FixedVectorWithBuffer(const FixedVectorWithBuffer& other) : FixedVectorWithBuffer(other.begin(), other.end()) { }

    //FixedVectorWithBuffer(const Span& other) : FixedVectorWithBuffer(other.begin(), other.end()) { } // TODO: Is this needed, or captured by template above?

    explicit FixedVectorWithBuffer(const std::initializer_list<T>& other) : FixedVectorWithBuffer(other.begin(), other.end()) { }

    template<typename T2> // initializer from a different type, e.g. size_t instead of NDShapeDimension
    explicit FixedVectorWithBuffer(const std::initializer_list<T2>& other) :
        FixedVectorWithBuffer(Transform(other, [](const T2& val) { return (T)val; }))
    { }

    FixedVectorWithBuffer& operator=(FixedVectorWithBuffer&& other)
    {
        this->~FixedVectorWithBuffer();
        new (this) FixedVectorWithBuffer(std::move(other));
        return *this;
    }
    FixedVectorWithBuffer& operator=(const Span& other)
    {
        // Note: We could optimize mallocs if both sizes are the same (then just copy over the elements)
        // However, this class should not be used this way, so for now we won't.
        this->~FixedVectorWithBuffer();
        new (this) FixedVectorWithBuffer(other);
        return *this;
    }
    FixedVectorWithBuffer& operator=(const FixedVectorWithBuffer& other) { return operator=((const Span&)other); }

    ~FixedVectorWithBuffer()
    {
        DestructAndFree(/*end of constructed range=*/end());
    }
    // this is a common use case
    void assign(size_t, T&& a)
    {
        this->~FixedVectorWithBuffer();
        new (this) FixedVectorWithBuffer(1, std::move(a));
    }
    FixedVectorWithBuffer BackPopped() const { return FixedVectorWithBuffer(begin(), end() - 1); }
};

///
/// MakeSharedObject() -- Custom shared-ptr allocator that uses fixed-size pools and custom deleter across the DLL boundary.
///
/// Pool-based allocator for efficient memory management
///

// a pool for allocating objects of one specific size
// BUGBUG: Does not work if T is marked 'final'. How does shared_ptr do it?
template<typename T> struct FixedSizePoolItem : public T { char* flagPtr; };
// class to store objects of size itemByteSize in lists of char arrays
template<size_t itemByteSize>
class FixedSizePoolStorage
{
    static void Assert(bool cond) { if (!cond) LogicError("FixedSizePool: An assertion failed."); }
    struct Block
    {
        static const size_t capacity = 65536; // we reserve this many at a time
        std::vector<char> bytes = std::vector<char>(capacity * itemByteSize); // [byte offset]
        std::vector<char> used  = std::vector<char>(capacity, false);         // [item index]  true if this entry is used
        // Note: time measurement comparing vector<bool> vs. vector<char> vs. using flagptr was inconclusive
        template<typename T>
        inline std::pair<T*, char*> TryAllocate(size_t& nextItemIndex)
        {
            //Assert(nextItemIndex <= capacity);
            while (nextItemIndex < capacity)
            {
                if (!used[nextItemIndex])
                {
                    T* p = (T*)(bytes.data() + nextItemIndex * itemByteSize);
                    char* flagPtr = &used[nextItemIndex];
                    *flagPtr = true; // and mark as allocated
                    //used[nextItemIndex] = true; // and mark as allocated
                    nextItemIndex++;
                    return std::make_pair(p, flagPtr);
                }
                nextItemIndex++;
            }
            return std::make_pair(nullptr, nullptr); // this block is full
        }
        static void Deallocate(char* flagPtr)
        {
            Assert(*flagPtr == (char)true);
            *flagPtr = false;
        }
    };
    // state of allocation
    std::list<Block> blocks;                          // all blocks we have currently allocated
    size_t totalItemsAllocated = 0;                   // we have presently this many live objects
    size_t totalItemsReserved = 0;                    // we are holding memory sufficient to hold this many
    // state of scan
    typename std::list<Block>::iterator currentBlock; // we are allocating from this block
    size_t nextItemIndex;                             // index of next item. If at end of block, this is equal to blockCapacity
    void ResetScan()
    {
        currentBlock = blocks.begin();
        nextItemIndex = 0;
    }
public:
    FixedSizePoolStorage()
    {
        ResetScan();
        //fprintf(stderr, "Scan reset for storage of elements of %d bytes\n", (int)itemByteSize);
    }
    template<typename T>
    T* Allocate()
    {
        //if (sizeof(FixedSizePoolItem<T>) != itemByteSize)
        //    LogicError("FixedSizePoolAllocator: Called for an object of the wrong size.");
        //Assert(totalItemsReserved >= totalItemsAllocated);
        //fprintf(stderr, "allocate<%s>()  --> %d bytes (%d incl. index)\n", typeid(T).name(), (int)sizeof T, (int)itemByteSize);
        // find next free location
        for (;;)
        {
            // all blocks are full: either reset the scan or grow
            if (currentBlock == blocks.end())
            {
                if (totalItemsReserved > totalItemsAllocated * 2)
                {
                    // if we have 50% utilization or below, we start over the scan in our existing allocated space
                    // At 50%, on av. we need to scan 1 extra item to find a free one.
                    ResetScan();
                }
                else
                {
                    // too few free items, we'd scan lots of items to find one: instead use a fresh block
                    //if ((decltype(FixedSizePoolItem<T>::blockIndex))(currentBlockIndex + 1) != currentBlockIndex + 1)
                    //    LogicError("FixedSizePoolAllocator: Too many blocks.");
                    blocks.emplace_back(Block());
                    totalItemsReserved += Block::capacity;
                    // enter the new block
                    currentBlock = blocks.end();
                    currentBlock--;
                    nextItemIndex = 0;
                }
            }
            // try to allocate in current block
            auto res = currentBlock->template TryAllocate<T>(nextItemIndex);
            auto* p = res.first;
            if (p) // found one in the current block
            {
                totalItemsAllocated++; // account for it
                auto* pItem = reinterpret_cast<FixedSizePoolItem<typename T>*>(const_cast<std::remove_const_t<T>*>(p));
                //pItem->blockIndex = (decltype(pItem->blockIndex))currentBlockIndex; // remember which block it came from
                pItem->flagPtr = res.second; // remember flag location for trivial deallocation
                //Assert(pItem->blockIndex == currentBlockIndex); // (overflow)
                return p;
            }
            // current block is full: advance the scan to the next block
            currentBlock++;
            nextItemIndex = 0;
        }
        //LogicError("FixedSizePoolAllocator: Allocation in newly created block unexpectedly failed.");
    }
    template<typename T>
    void Deallocate(T* p)
    {
        //fprintf(stderr, "deallocate<%s>()  --> %d bytes (%d incl. index)\n", typeid(T).name(), (int)sizeof T, (int)itemByteSize);
        const auto* pItem = reinterpret_cast<FixedSizePoolItem<T>*>(const_cast<std::remove_const_t<T>*>(p));
        auto* flagPtr = pItem->flagPtr;
        Block::Deallocate(flagPtr);
        totalItemsAllocated--;
    }
};

///
/// C++ allocator
///  - get the benefit of a make_shared-like allocation with control block and actual object allocated together
///  - can be sure the correct deleter is called (on the correct heap)
///
template<typename T>
class FixedSizePoolAllocatorT
{
public: // required boilerplate --is there no base to derive from to provide this & deal with the intricacies?
    typedef T value_type;
    typedef value_type* pointer; typedef const value_type* const_pointer;
    typedef value_type& reference; typedef const value_type& const_reference;
    typedef std::size_t size_type; typedef std::ptrdiff_t difference_type;
    template<typename U> struct rebind { typedef FixedSizePoolAllocatorT<U> other; };
    inline pointer address(reference r) { return &r; }
    //inline const_pointer address(const_reference r) { return &r; } // causes a dup error
    inline explicit FixedSizePoolAllocatorT() {}
    inline ~FixedSizePoolAllocatorT() {}
    inline explicit FixedSizePoolAllocatorT(FixedSizePoolAllocatorT const&) {}
    template<typename U>
    inline explicit FixedSizePoolAllocatorT(FixedSizePoolAllocatorT<U> const&) {}
    inline size_type max_size() const { return std::numeric_limits<size_type>::max() / sizeof(T); }
    inline void construct(pointer p, const T& t) { new(p) T(t); }
    inline void destroy(pointer p) { p->~T(); }
    inline bool operator==(FixedSizePoolAllocatorT const&) { return true; }
    inline bool operator!=(FixedSizePoolAllocatorT const& a) { return !operator==(a); }
private:
    // say FixedSizePool::get() to get access to a globally shared instance for all pools of the same itemByteSize
    static auto& GetStorage() { static FixedSizePoolStorage<sizeof(FixedSizePoolItem<T>)>/*<itemByteSize>*/ s_storage; return s_storage; }
public:
    inline pointer allocate(size_type cnt = 1, typename std::allocator<void>::const_pointer = 0)
    {
        if (cnt != 1)
            InvalidArgument("FixedSizePoolAllocatorT: This allocator only supports allocation of single items.");
        //auto& storage = FixedSizePoolStorage<sizeof(FixedSizePoolItem<T>)>::get();
        auto& storage = GetStorage();
        return reinterpret_cast<pointer>(storage.template Allocate<T>());
    }
    inline void deallocate(pointer p, size_type = 1)
    {
        //auto& storage = FixedSizePoolStorage<sizeof(FixedSizePoolItem<T>)>::get();
        auto& storage = GetStorage();
        storage.template Deallocate<T>(p);
    }
public:
    static void SomeTest()
    {
        typedef FixedSizePoolAllocatorT<char> AllocatorUnderTest;
        AllocatorUnderTest alloc;
        std::vector<int, AllocatorUnderTest> x(1, 1);
        std::list<int, AllocatorUnderTest> v1;
        v1.push_back(13); v1.push_back(42); v1.push_back(13); v1.push_back(13); v1.push_back(13);
        v1.erase(v1.begin()++++); v1.erase(v1.begin()++++); v1.erase(v1.begin()++++); v1.erase(v1.begin()++++);
        v1.push_back(13); v1.push_back(13); v1.push_back(13); v1.push_back(13); v1.push_back(13);
        v1.push_back(13); v1.push_back(13); v1.push_back(13); v1.push_back(13); v1.push_back(13);
        std::list<int> v2(v1.begin(), v1.end());
        auto ps = std::allocate_shared<std::string>(alloc, "test");
        ps.reset();
        auto pi = std::allocate_shared<int>(alloc, 1968);
    }
};
typedef FixedSizePoolAllocatorT<char> FixedSizePoolAllocator; // turns out, the actual template argument does not matter here, it is never used this way

///
/// Simple intrusive strong shared_ptr (no weak-ptr support), aimed to combat STL's shared_ptr overhead/inlining problems.
/// Requires the controlled object to derive from enable_strong_shared_ptr.
///
template<class T>
class enable_strong_shared_ptr
{
    mutable size_t referenceCount = 0;
    template<class T>
    friend class strong_shared_ptr;
    void AddRef() const noexcept { referenceCount++; }
    size_t DecRef() const noexcept { referenceCount--; return referenceCount; }
public:
    T* get() const { return static_cast<T*>(*this); } // TODO: no, must return strong_shared_ptr<T>
    //strong_shared_ptr<T> shared_from_this() const { get(); }
};
template<class T>
class strong_shared_ptr final
{
    T* m_ptr;
    static T* AddRef(T* other)
    {
        if (other)
            other->AddRef();
        return other;
    }
    struct Storage
    {
        static FixedSizePoolStorage<sizeof FixedSizePoolItem<T>> s_storage;
    };
    static auto& GetStorage() { return Storage::s_storage; }
    void Release()
    {
        if (m_ptr && m_ptr->DecRef() == 0)
        {
            m_ptr->~T();
            GetStorage().Deallocate<T>(m_ptr);
        }
    }
    void ReleaseAndReplace(T* other)
    {
        Release();
        m_ptr = other;
    }
    explicit strong_shared_ptr(T* ptr) noexcept : m_ptr(AddRef(ptr)) { } // private, can only be used via Construct
public:
    typedef T element_type;
    strong_shared_ptr()          noexcept : m_ptr(nullptr) { }
    strong_shared_ptr(const strong_shared_ptr& other) noexcept : m_ptr(AddRef(other.m_ptr)) { }
    strong_shared_ptr(strong_shared_ptr&&      other) noexcept : m_ptr(other.m_ptr) { other.m_ptr = nullptr; }
    ~strong_shared_ptr() noexcept { Release(); }
    strong_shared_ptr& operator=(const strong_shared_ptr& other)     { ReleaseAndReplace(AddRef(other.m_ptr));                return *this; }
    strong_shared_ptr& operator=(strong_shared_ptr&& other) noexcept { ReleaseAndReplace(other.m_ptr); other.m_ptr = nullptr; return *this; }
    void reset() noexcept { ReleaseAndReplace(nullptr); }
    typename std::add_lvalue_reference_t<T> operator*() const noexcept { return *m_ptr; }
    T* operator->() const noexcept { return m_ptr; }
    T* get()        const noexcept { return m_ptr; }
    explicit operator bool() const noexcept { return m_ptr != nullptr; }
    void swap(strong_shared_ptr& other) noexcept { std::swap(m_ptr, other.m_ptr); }
    bool operator==(const strong_shared_ptr& other) const noexcept { return m_ptr == other.m_ptr; }
    bool operator!=(const strong_shared_ptr& other) const noexcept { return m_ptr != other.m_ptr; }
    // the only way to construct the held object anew is this function, since we control the allocator
    template <typename ...CtorArgTypes>
    static strong_shared_ptr MakeSharedObject(CtorArgTypes&& ...ctorArgs)
    {
        T* p = GetStorage().Allocate<T>();
        try
        {
            return strong_shared_ptr(new (const_cast<std::remove_const_t<T>*>(p)) T(std::forward<CtorArgTypes>(ctorArgs)...));
        }
        catch (...)
        {
            GetStorage().Deallocate<T>(p);
            throw;
        }
    }
};

///
/// Creates a shared object using our own intrusive shared class.
///
// TODO: figure out the constraint thingy here, then rename the -1 away
template <typename T, typename ...CtorArgTypes /*WHERE_IS_BASE_OF(enable_strong_shared_ptr<typename T>, typename T)*/>
inline strong_shared_ptr<T> MakeSharedObject1(CtorArgTypes&& ...ctorArgs)
{
    return strong_shared_ptr<T>::MakeSharedObject(std::forward<CtorArgTypes>(ctorArgs)...);
}

///
/// Creates a shared object, using a shared_ptr with custom allocator (pool-based, working across DLL boundaru)
/// Similar to make_shared except that it associates a custom allocator with the shared_ptr to ensure
/// that objects are deleted on the same side of the library DLL where they are allocated
///
template <typename T, typename ...CtorArgTypes /*WHERE_IS_NOT_BASE_OF(enable_strong_shared_ptr<typename T>, typename T)*/>
inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs)
{
    FixedSizePoolAllocator objectAllocator;
    return std::allocate_shared<T>(objectAllocator, std::forward<CtorArgTypes>(ctorArgs)...);
}

///
/// Class that stores an immutable std::wstring. Assumes it is most of the time is empty, so that it is cheaper to have one extra redirection.
/// Also uses a shared pointer, i.e. the string is not copied but shared if assigned.
///
class OptionalString
{
public: // for shared object allocatio only --TODO: figure out a better way
    struct SharableString : public enable_strong_shared_ptr<SharableString>, std::wstring
    {
        SharableString(const std::wstring& s) : std::wstring(s) { }
        SharableString(std::wstring&& s) : std::wstring(move(s)) { }
    };
    typedef strong_shared_ptr<SharableString const> StringPtr;
    StringPtr m_string;
public:
    OptionalString() { }
    OptionalString(OptionalString&& other) : m_string(move(other.m_string)) { }
    OptionalString(const OptionalString& other) : m_string(other.m_string) { }
    explicit OptionalString(const std::wstring&  s) : m_string(s.empty() ? StringPtr() : MakeSharedObject1<SharableString const>(s)) { }
    explicit OptionalString(      std::wstring&& s) : m_string(s.empty() ? StringPtr() : MakeSharedObject1<SharableString const>(std::move(s))) { }
    OptionalString& operator=(const std::wstring& s) { if (s.empty()) m_string.reset(); else m_string = MakeSharedObject1<SharableString const>((s)); return *this; }
    OptionalString& operator=(std::wstring&& s)      { if (s.empty()) m_string.reset(); else m_string = MakeSharedObject1<SharableString const>((std::move(s))); return *this; }
    OptionalString& operator=(const OptionalString& other) { m_string = other.m_string;       return *this; }
    OptionalString& operator=(OptionalString&& other)      { m_string = move(other.m_string); return *this; }
    operator const std::wstring&() const { return get(); }
    const std::wstring& get() const { static const std::wstring s_emptyString; return m_string ? *m_string : s_emptyString; }
    bool empty() const { return !m_string || m_string->empty(); }
    const wchar_t* c_str() const { return m_string ? m_string->c_str() : L""; }
};

} // namespace
