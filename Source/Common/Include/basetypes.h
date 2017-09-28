//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// basetypes.h - basic types that C++ lacks
//
#pragma once
#ifndef _BASETYPES_
#define _BASETYPES_

#include "Platform.h"
#include "Basics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // include here because we redefine some names later
#include <errno.h>
#include <string>
#include <vector>
#include <cmath> // for HUGE_VAL
#include <assert.h>
#include <stdarg.h>
#include <map>
#include <stdexcept>
#include <locale> // std::wstring_convert
#include <string>
#include <algorithm> // for transform()
#include <unordered_map>
#include <chrono>
#include <thread>
#include <stack>
#include <mutex>
#include <memory>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#include "Windows.h" // for CRITICAL_SECTION and Unicode conversion functions   --TODO: is there a portable alternative?
#endif
#if __unix__
#include <strings.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <sys/time.h>
typedef unsigned char byte;
#endif

static inline wchar_t *GetWC(const char *c)
{
    const size_t cSize = strlen(c) + 1;
    wchar_t *wc = new wchar_t[cSize];
#ifdef _WIN32
    size_t retVal;
    mbstowcs_s(&retVal, wc, cSize, c, cSize);
#else
    mbstowcs(wc, c, cSize);
#endif // _WIN32

    return wc;
}

// ----------------------------------------------------------------------------
// basic data types
// ----------------------------------------------------------------------------

namespace msra { namespace basetypes {

#ifdef __unix__
typedef timeval LARGE_INTEGER;
#endif
class auto_timer
{
    LARGE_INTEGER freq, start;
    auto_timer(const auto_timer &);
    void operator=(const auto_timer &);

public:
    auto_timer()
    {
#ifdef _WIN32
        if (!QueryPerformanceFrequency(&freq)) // count ticks per second
            RuntimeError("auto_timer: QueryPerformanceFrequency failure");
        QueryPerformanceCounter(&start);
#endif
#ifdef __unix__
        gettimeofday(&start, NULL);
#endif
    }
    operator double() const // each read gives time elapsed since start, in seconds
    {
        LARGE_INTEGER end;
#ifdef _WIN32
        QueryPerformanceCounter(&end);
        return (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
#endif
#ifdef __unix__
        gettimeofday(&end, NULL);
        return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / (1000 * 1000);
#endif
    }
    void show(const std::string &msg) const
    {
        double elapsed = *this;
        fprintf(stderr, "%s: %.6f ms\n", msg.c_str(), elapsed * 1000.0 /*to ms*/);
    }
};

#pragma warning(push)
#pragma warning(disable : 4555) // expression has no affect, used so retail won't be empty

// class fixed_vector - non-resizable vector  --TODO: just use std::vector

template <class _T>
class fixed_vector
{
    _T *p;    // pointer array
    size_t n; // number of elements
    void check(int index) const
    {
        assert(index >= 0 && (size_t) index < n);
#ifdef NDEBUG
        UNUSED(index);
#endif
    }
    void check(size_t index) const
    {
        assert(index < n);
#ifdef NDEBUG
        UNUSED(index);
#endif
    }
    // ... TODO: when I make this public, LinearTransform.h acts totally up but I cannot see where it comes from.
    // fixed_vector (const fixed_vector & other) : n (0), p (NULL) { *this = other; }
public:
    fixed_vector()
        : n(0), p(NULL)
    {
    }
    void resize(int size)
    {
        clear();
        if (size > 0)
        {
            p = new _T[size];
            n = size;
        }
    }
    void resize(size_t size)
    {
        clear();
        if (size > 0)
        {
            p = new _T[size];
            n = size;
        }
    }
    fixed_vector(int size)
        : n(size), p(size > 0 ? new _T[size] : NULL)
    {
    }
    fixed_vector(size_t size)
        : n((int) size), p(size > 0 ? new _T[size] : NULL)
    {
    }
    ~fixed_vector()
    {
        delete[] p;
    }
    int size() const
    {
        return (int) n;
    }
    inline int capacity() const
    {
        return (int) n;
    }
    bool empty() const
    {
        return n == 0;
    }
    void clear()
    {
        delete[] p;
        p = NULL;
        n = 0;
    }
    _T *begin()
    {
        return p;
    }
    const _T *begin() const
    {
        return p;
    }
    _T *end()
    {
        return p + n;
    } // note: n == 0 so result is NULL
    inline _T &operator[](int index)
    {
        check(index);
        return p[index];
    } // writing
    inline const _T &operator[](int index) const
    {
        check(index);
        return p[index];
    } // reading
    inline _T &operator[](size_t index)
    {
        check(index);
        return p[index];
    } // writing
    inline const _T &operator[](size_t index) const
    {
        check(index);
        return p[index];
    } // reading
    inline int indexof(const _T &elem) const
    {
        assert(&elem >= p && &elem < p + n);
        return &elem - p;
    }
    void swap(fixed_vector &other) throw()
    {
        std::swap(other.p, p);
        std::swap(other.n, n);
    }
    template <class VECTOR>
    fixed_vector &operator=(const VECTOR &other)
    {
        int other_n = (int) other.size();
        fixed_vector tmp(other_n);
        for (int k = 0; k < other_n; k++)
            tmp[k] = other[k];
        swap(tmp);
        return *this;
    }
    fixed_vector &operator=(const fixed_vector &other)
    {
        int other_n = (int) other.size();
        fixed_vector tmp(other_n);
        for (int k = 0; k < other_n; k++)
            tmp[k] = other[k];
        swap(tmp);
        return *this;
    }
    template <class VECTOR>
    fixed_vector(const VECTOR &other)
        : n(0), p(NULL)
    {
        *this = other;
    }
};
template <class _T>
inline void swap(fixed_vector<_T> &L, fixed_vector<_T> &R) throw()
{
    L.swap(R);
}

#pragma warning(pop) // pop off waring: expression has no effect

// class matrix - simple fixed-size 2-dimensional array, access elements as m(i,j)
// stored as concatenation of rows

#if 1
template <class T>
class matrix : fixed_vector<T>
{
    size_t numcols;
    size_t locate(size_t i, size_t j) const
    {
        assert(i < rows() && j < cols());
        return i * cols() + j;
    }

public:
    typedef T elemtype;
    matrix()
        : numcols(0)
    {
    }
    matrix(size_t n, size_t m)
    {
        resize(n, m);
    }
    void resize(size_t n2, size_t m)
    {
        numcols = m;
        fixed_vector<T>::resize(n2 * m);
    }
    size_t cols() const
    {
        return numcols;
    }
    size_t rows() const
    {
        return empty() ? 0 : size() / cols();
    }
    size_t size() const
    {
        return fixed_vector<T>::size();
    } // use this for reading and writing... not nice!
    bool empty() const
    {
        return fixed_vector<T>::empty();
    }
    T &operator()(size_t i, size_t j)
    {
        return (*this)[locate(i, j)];
    }
    const T &operator()(size_t i, size_t j) const
    {
        return (*this)[locate(i, j)];
    }
    void swap(matrix &other) throw()
    {
        std::swap(numcols, other.numcols);
        fixed_vector<T>::swap(other);
    }
};
template <class _T>
inline void swap(matrix<_T> &L, matrix<_T> &R) throw()
{
    L.swap(R);
}

#endif

// derive from this for noncopyable classes (will get you private unimplemented copy constructors)
// ... TODO: change all of basetypes classes/structs to use this
class noncopyable
{
    noncopyable &operator=(const noncopyable &);
    noncopyable(const noncopyable &);

public:
    noncopyable()
    {
    }
};

class CCritSec
{
    CCritSec(const CCritSec &) = delete;
    CCritSec &operator=(const CCritSec &) = delete;
    std::mutex m_CritSec;

public:
    CCritSec(){};
    ~CCritSec(){};
    void Lock()
    {
        m_CritSec.lock();
    };
    void Unlock()
    {
        m_CritSec.unlock();
    };
};

// locks a critical section, and unlocks it automatically
// when the lock goes out of scope
class CAutoLock
{
    CAutoLock(const CAutoLock &refAutoLock);
    CAutoLock &operator=(const CAutoLock &refAutoLock);
    CCritSec &m_rLock;

public:
    CAutoLock(CCritSec &rLock)
        : m_rLock(rLock)
    {
        m_rLock.Lock();
    };
    ~CAutoLock()
    {
        m_rLock.Unlock();
    };
};

}; }; // namespace

// ----------------------------------------------------------------------------
// frequently missing utility functions
// ----------------------------------------------------------------------------

namespace msra { namespace util {

// byte-reverse a variable --reverse all bytes (intended for integral types and float)
template <typename T>
static inline void bytereverse(T &v) throw()
{ // note: this is more efficient than it looks because sizeof (v[0]) is a constant
    char *p = (char *) &v;
    const size_t elemsize = sizeof(v);
    for (int k = 0; k < elemsize / 2; k++) // swap individual bytes
        std::swap(p[k], p[elemsize - 1 - k]);
}

// byte-swap an entire array
template <class V>
static inline void byteswap(V &v) throw()
{
    foreach_index (i, v)
        bytereverse(v[i]);
}

// execute a block with retry
// Block must be restartable.
// Use this when writing/reading small files to those unreliable Windows servers.
// TODO: This will fail to compile under VS 2008--we need an #ifdef around this
template <typename FUNCTION>
static void attempt(int retries, const FUNCTION &body)
{
    for (int attempt = 1;; attempt++)
    {
        try
        {
            body();
            if (attempt > 1)
                fprintf(stderr, "attempt: success after %d retries\n", attempt);
            break;
        }
        catch (const std::exception &e)
        {
            if (attempt >= retries)
                throw; // failed N times --give up and rethrow the error
            fprintf(stderr, "attempt: %s, retrying %d-th time out of %d...\n", e.what(), attempt + 1, retries);
            ::Sleep(1000); // wait a little, then try again
        }
    }
}

}; }; // namespace

template <class S>
static inline void ZeroStruct(S &s)
{
    memset(&s, 0, sizeof(s));
}

using namespace msra::basetypes; // for compatibility

#endif // _BASETYPES_
