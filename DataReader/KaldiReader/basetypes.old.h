// TODO: This is a dup, we should get back to the shared one. But this one has some stuff the other doesn't.

//
// <copyright file="basetypes.old.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once
#ifndef _BASETYPES_
#define _BASETYPES_

// [kit]: seems SECURE_SCL=0 doesn't work - causes crashes in release mode
// there are some complaints along this line on the web
// so disabled for now
//
//// we have agreed that _SECURE_SCL is disabled for release builds
//// it would be super dangerous to mix projects where this is inconsistent
//// this is one way to detect possible mismatches
//#ifdef NDEBUG
//#if !defined(_CHECKED) && _SECURE_SCL != 0 
//#error "_SECURE_SCL should be disabled for release builds"
//#endif
//#endif

#ifndef UNDER_CE    // fixed-buffer overloads not available for wince
#ifdef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES  // fixed-buffer overloads for strcpy() etc.
#undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#endif
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif

#pragma warning (push)
#pragma warning (disable: 4793)    // caused by varargs

// disable certain parts of basetypes for wince compilation
#ifdef UNDER_CE
#define BASETYPES_NO_UNSAFECRTOVERLOAD // disable unsafe CRT overloads (safe functions don't exist in wince)
#define BASETYPES_NO_STRPRINTF         // dependent functions here are not defined for wince
#endif

#ifndef OACR    // dummies when we are not compiling under Office
#define OACR_WARNING_SUPPRESS(x, y)
#define OACR_WARNING_DISABLE(x, y)
#define OACR_WARNING_PUSH
#define OACR_WARNING_POP
#endif
#ifndef OACR_ASSUME    // this seems to be a different one
#define OACR_ASSUME(x)
#endif

// following oacr warnings are not level1 or level2-security
// in currect stage we want to ignore those warnings
// if necessay this can be fixed at later stage

// not a bug
OACR_WARNING_DISABLE(EXC_NOT_CAUGHT_BY_REFERENCE, "Not indicating a bug or security threat.");
OACR_WARNING_DISABLE(LOCALDECLHIDESLOCAL, "Not indicating a bug or security threat.");

// not reviewed
OACR_WARNING_DISABLE(MISSING_OVERRIDE, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(EMPTY_DTOR, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(DEREF_NULL_PTR, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(INVALID_PARAM_VALUE_1, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(VIRTUAL_CALL_IN_CTOR, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(POTENTIAL_ARGUMENT_TYPE_MISMATCH, "Not level1 or level2_security.");

// determine WIN32 api calling convention
// it seems this is normally stdcall?? but when compiling as /clr:pure or /clr:Safe
// this is not supported, so in this case, we need to use the 'default' calling convention
// TODO: can we reuse the #define of WINAPI??
#ifdef _WIN32
#ifdef _M_CEE_SAFE 
#define WINAPI_CC __clrcall
#elif _M_CEE
#define WINAPI_CC __clrcall
#else
#define WINAPI_CC __stdcall
#endif
#endif

// fix some warnings in STL
#if !defined(_DEBUG) || defined(_CHECKED) || defined(_MANAGED)
#pragma warning(disable : 4702) // unreachable code
#endif
#include <stdarg.h>
#include <stdio.h>
#include <string.h>     // include here because we redefine some names later
#include <string>
#include <vector>
#include <cmath>        // for HUGE_VAL
#include <assert.h>
#include <map>
#ifdef __windows__
#include <windows.h>    // for CRITICAL_SECTION
#include <strsafe.h>    // for strbcpy() etc templates
#endif
#if __unix__
#include <strings.h>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#include <dlfcn.h>
typedef unsigned char byte;
#endif


#pragma push_macro("STRSAFE_NO_DEPRECATE")
#define STRSAFE_NO_DEPRECATE    // deprecation managed elsewhere, not by strsafe
#pragma pop_macro("STRSAFE_NO_DEPRECATE")

// CRT error handling seems to not be included in wince headers
// so we define our own imports
#ifdef UNDER_CE

// TODO: is this true - is GetLastError == errno?? - also this adds a dependency on windows.h
#define errno GetLastError() 

// strerror(x) - x here is normally errno - TODO: make this return errno as a string
#define strerror(x) "strerror error but can't report error number sorry!"
#endif

#ifndef __in // dummies for sal annotations if compiler does not support it
#define __in
#define __inout_z
#define __in_count(x)
#define __inout_cap(x)
#define __inout_cap_c(x)
#endif
#ifndef __out_z_cap    // non-VS2005 annotations
#define __out_cap(x)
#define __out_z_cap(x)
#define __out_cap_c(x)
#endif

#ifndef __override      // and some more non-std extensions required by Office
#define __override virtual
#endif

// disable warnings for which fixing would make code less readable
#pragma warning(disable : 4290) // throw() declaration ignored
#pragma warning(disable : 4244) // conversion from typeA to typeB, possible loss of data

// ----------------------------------------------------------------------------
// basic macros
// ----------------------------------------------------------------------------

#define SAFE_DELETE(p)  { if(p) { delete (p); (p)=NULL; } }
#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }     // nasty! use CComPtr<>
#ifndef assert
#ifdef _CHECKED // basetypes.h expects this function to be defined (it is in message.h)
extern void _CHECKED_ASSERT_error(const char * file, int line, const char * exp);
#define assert(exp) ((exp)||(_CHECKED_ASSERT_error(__FILE__,__LINE__,#exp),0))
#else
#define assert assert
#endif
#endif

using namespace std;
// ----------------------------------------------------------------------------
// basic data types
// ----------------------------------------------------------------------------

namespace msra { namespace basetypes {

// class std::vector -- std::vector with array-bounds checking
// VS 2008 and above do this, so there is no longer a need for this.

template<class _ElemType>
class std::vector : public std::vector<_ElemType>
{
#if defined (_DEBUG) || defined (_CHECKED)    // debug version with range checking
    static void throwOutOfBounds()
    {   // (moved to separate function hoping to keep inlined code smaller
        OACR_WARNING_PUSH;
        OACR_WARNING_DISABLE(IGNOREDBYCOMMA, "Reviewd OK. Special trick below to show a message when assertion fails"
            "[rogeryu 2006/03/24]");
        OACR_WARNING_DISABLE(BOGUS_EXPRESSION_LIST, "This is intentional. [rogeryu 2006/03/24]");
        assert (("std::vector::operator[] out of bounds", false));
        OACR_WARNING_POP;
    }
#endif

public:

    std::vector() : std::vector<_ElemType> () { }
    std::vector (int size) : std::vector<_ElemType> (size) { }

#if defined (_DEBUG) || defined (_CHECKED)    // debug version with range checking
    // ------------------------------------------------------------------------
    // operator[]: with array-bounds checking
    // ------------------------------------------------------------------------

    inline _ElemType & operator[] (int index)            // writing
    {
        if (index < 0 || index >= size()) throwOutOfBounds();
        return (*(std::vector<_ElemType>*) this)[index];
    }

    // ------------------------------------------------------------------------

    inline const _ElemType & operator[] (int index) const    // reading
    {
        if (index < 0 || index >= size()) throwOutOfBounds();
        return (*(std::vector<_ElemType>*) this)[index];
    }
#endif

    // ------------------------------------------------------------------------
    // size(): same as base class, but returning an 'int' instead of 'size_t'
    // to allow for better readable code
    // ------------------------------------------------------------------------

    inline int size() const
    {
        size_t siz = ((std::vector<_ElemType>*) this)->size();
        return (int) siz;
    }
};
// overload swap(), otherwise we'd fallback to 3-way assignment & possibly throw
template<class _T> inline void swap (std::vector<_T> & L, std::vector<_T> & R) throw()
{ swap ((std::vector<_T> &) L, (std::vector<_T> &) R); }

// class fixed_vector - non-resizable vector

template<class _T> class fixed_vector
{
    _T * p;                 // pointer array
    size_t n;               // number of elements
    void check (int index) const { index; assert (index >= 0 && (size_t) index < n); }
    void check (size_t index) const { index; assert (index < n); }
    // ... TODO: when I make this public, LinearTransform.h acts totally up but I cannot see where it comes from.
    //fixed_vector (const fixed_vector & other) : n (0), p (NULL) { *this = other; }
public:
    fixed_vector() : n (0), p (NULL) { }
    void resize (int size) { clear(); if (size > 0) { p = new _T[size]; n = size; } }
    void resize (size_t size) { clear(); if (size > 0) { p = new _T[size]; n = size; } }
    fixed_vector (int size) : n (size), p (size > 0 ? new _T[size] : NULL) { }
    fixed_vector (size_t size) : n ((int) size), p (size > 0 ? new _T[size] : NULL) { }
    ~fixed_vector() { delete[] p; }
    inline int size() const { return (int) n; }
    inline int capacity() const { return (int) n; }
    inline bool empty() const { return n == 0; }
    void clear() { delete[] p; p = NULL; n = 0; }
    _T *       begin()       { return p; }
    const _T * begin() const { return p; }
    _T * end()   { return p + n; } // note: n == 0 so result is NULL
    inline       _T & operator[] (int index)          { check (index); return p[index]; }  // writing
    inline const _T & operator[] (int index) const    { check (index); return p[index]; }  // reading
    inline       _T & operator[] (size_t index)       { check (index); return p[index]; }  // writing
    inline const _T & operator[] (size_t index) const { check (index); return p[index]; }  // reading
    inline int indexof (const _T & elem) const { assert (&elem >= p && &elem < p + n); return &elem - p; }
    inline void swap (fixed_vector & other) throw() { std::swap (other.p, p); std::swap (other.n, n); }
    template<class VECTOR> fixed_vector & operator= (const VECTOR & other)
    {
        int other_n = (int) other.size();
        fixed_vector tmp (other_n);
        for (int k = 0; k < other_n; k++) tmp[k] = other[k];
        swap (tmp);
        return *this;
    }
    fixed_vector & operator= (const fixed_vector & other)
    {
        int other_n = (int) other.size();
        fixed_vector tmp (other_n);
        for (int k = 0; k < other_n; k++) tmp[k] = other[k];
        swap (tmp);
        return *this;
    }
    template<class VECTOR> fixed_vector (const VECTOR & other) : n (0), p (NULL) { *this = other; }
};
template<class _T> inline void swap (fixed_vector<_T> & L, fixed_vector<_T> & R) throw() { L.swap (R); }

// class matrix - simple fixed-size 2-dimensional array, access elements as m(i,j)
// stored as concatenation of rows

template<class T> class matrix : fixed_vector<T>
{
    size_t numcols;
    size_t locate (size_t i, size_t j) const { assert (i < rows() && j < cols()); return i * cols() + j; }
public:
    typedef T elemtype;
    matrix() : numcols (0) {}
    matrix (size_t n, size_t m) { resize (n, m); }
    void resize (size_t n, size_t m) { numcols = m; fixed_vector<T>::resize (n * m); }
    size_t cols() const { return numcols; }
    size_t rows() const { return empty() ? 0 : size() / cols(); }
    size_t size() const { return fixed_vector<T>::size(); }    // use this for reading and writing... not nice!
    bool empty() const { return fixed_vector<T>::empty(); }
    T &       operator() (size_t i, size_t j)       { return (*this)[locate(i,j)]; }
    const T & operator() (size_t i, size_t j) const { return (*this)[locate(i,j)]; }
    void swap (matrix & other) throw() { std::swap (numcols, other.numcols); fixed_vector<T>::swap (other); }
};
template<class _T> inline void swap (matrix<_T> & L, matrix<_T> & R) throw() { L.swap (R); }

// TODO: get rid of these
typedef std::string STRING;
typedef std::wstring WSTRING;
#ifdef __unix__
typedef wchar_t TCHAR;
#endif
typedef std::basic_string<TCHAR> TSTRING;    // wide/narrow character string

// derive from this for noncopyable classes (will get you private unimplemented copy constructors)
// ... TODO: change all of basetypes classes/structs to use this
class noncopyable
{
    noncopyable & operator= (const noncopyable &);
    noncopyable (const noncopyable &);
public:
    noncopyable(){}
};

struct throw_hr
{
    const char * msg;
    inline throw_hr (const char * msg = NULL) : msg (msg) {}
};

// back-mapping of exceptions to HRESULT codes
// usage pattern: HRESULT COM_function (...) { try { exception-based function body } catch_hr_return; }
#define catch_hr_return    \
        catch (const bad_alloc &) { return E_OUTOFMEMORY; }         \
        catch (const bad_hr & e) { return e.hr; }                   \
        catch (const invalid_argument &) { return E_INVALIDARG; }   \
        catch (const runtime_error &) { return E_FAIL; }            \
        catch (const logic_error &) { return E_UNEXPECTED; }        \
        catch (const exception &) { return E_FAIL; }                \
        return S_OK;

};};    // namespace

#ifndef BASETYPES_NO_UNSAFECRTOVERLOAD // if on, no unsafe CRT overload functions

// ----------------------------------------------------------------------------
// overloads for "unsafe" CRT functions used in our code base
// ----------------------------------------------------------------------------

// strlen/wcslen overloads for fixed-buffer size

// Note: Careful while fixing bug related to these templates.
// In all attempted experiments, in seems all 6 definitions are required 
// below to get the correct behaviour.  Be very very careful 
// not to delete something without testing that case 5&6 have "size" deduced.
// 1. char *
// 2. char * const
// 3. const char *
// 4. const char * const
// 5. char (&) [size]
// 6. const char (&) [size]
// the following includes all headers that use strlen() and fail because of the mapping below
// to find those, change #define strlen strlen_ to something invalid e.g. strlen::strlen_
#if _MSC_VER >= 1600    // VS 2010  --TODO: fix this by correct include order instead
#include <intrin.h>     // defines strlen() as an intrinsic in VS 2010
#include <typeinfo>     // uses strlen()
#include <xlocale>      // uses strlen()
#endif
#define strlen strlen_
template<typename _T> 
size_t strlen_(_T &s) { return strnlen_s(static_cast<const char *>(s), SIZE_MAX); } // never be called but needed to keep compiler happy
template<typename _T> inline size_t strlen_(const _T &s)     { return strnlen(static_cast<const char *>(s), SIZE_MAX); }
template<> inline size_t strlen_(char * &s)                  { return strnlen(s, SIZE_MAX); }
template<> inline size_t strlen_(const char * &s)            { return strnlen(s, SIZE_MAX); }
template<size_t n> inline size_t strlen_(const char (&s)[n]) { return (strnlen(s, n)); }
template<size_t n> inline size_t strlen_(char (&s)[n])       { return (strnlen(s, n)); }
#define wcslen wcslen_
template<typename _T> 
size_t wcslen_(_T &s) { return wcsnlen_s(static_cast<const wchar_t *>(s), SIZE_MAX); } // never be called but needed to keep compiler happy
template<> inline size_t wcslen_(wchar_t * &s)                  { return wcsnlen(s, SIZE_MAX); }
template<> inline size_t wcslen_(const wchar_t * &s)            { return wcsnlen(s, SIZE_MAX); }
template<size_t n> inline size_t wcslen_(const wchar_t (&s)[n]) { return (wcsnlen(s, n)); }
template<size_t n> inline size_t wcslen_(wchar_t (&s)[n])       { return (wcsnlen(s, n)); }

// xscanf wrappers -- one overload for each actual use case in our code base
static inline int sscanf  (const char * buf, const char * format, int * i1)                     { return sscanf (buf, format, i1); }
static inline int sscanf  (const char * buf, const char * format, int * i1, int * i2)           { return sscanf (buf, format, i1, i2); }
static inline int sscanf  (const char * buf, const char * format, int * i1, int * i2, int * i3) { return sscanf (buf, format, i1, i2, i3); }
static inline int sscanf  (const char * buf, const char * format, double * f1)                  { return sscanf (buf, format, f1); }
static inline int swscanf (const wchar_t * buf, const wchar_t * format, int * i1)               { return swscanf (buf, format, i1); }
static inline int fscanf  (FILE * file, const char * format, float * f1)                        { return fscanf (file, format, f1); }

// cacpy -- fixed-size character array (same as original strncpy (dst, src, sizeof (dst)))
// NOTE: THIS FUNCTION HAS NEVER BEEN TESTED. REMOVE THIS COMMENT ONCE IT HAS.
template<class T, size_t n> static inline void cacpy (T (&dst)[n], const T * src)
{ for (int i = 0; i < n; i++) { dst[i] = *src; if (*src) src++; } }
// { return strncpy (dst, src, n); }   // using original C std lib function

#endif

// ----------------------------------------------------------------------------
// frequently missing string functions
// ----------------------------------------------------------------------------

namespace msra { namespace strfun {

#ifndef BASETYPES_NO_STRPRINTF
    template<typename C> struct basic_cstring : public std::basic_string<C>
    {
        template<typename S> basic_cstring (S p) : std::basic_string<C> (p) { }
        operator const C * () const { return this->c_str(); }
    };
 
typedef basic_cstring<char> cstring;
typedef basic_cstring<wchar_t> wcstring;

// [w]strprintf() -- like sprintf() but resulting in a C++ string
template<class _T> struct _strprintf : public std::basic_string<_T>
{   // works for both wchar_t* and char*
    _strprintf (const _T * format, ...)
    {
        va_list args; va_start (args, format);  // varargs stuff
        size_t n = _cprintf (format, args);     // num chars excl. '\0'
        const int FIXBUF_SIZE = 128;            // incl. '\0'
        if (n < FIXBUF_SIZE)
        {
            _T fixbuf[FIXBUF_SIZE];
            this->assign (_sprintf (&fixbuf[0], sizeof (fixbuf)/sizeof (*fixbuf), format, args), n);
        }
        else    // too long: use dynamically allocated variable-size buffer
        {
            std::vector<_T> varbuf (n + 1);     // incl. '\0'
            this->assign (_sprintf (&varbuf[0], varbuf.size(), format, args), n);
        }
    }
private:
    // helpers
    inline size_t _cprintf (const wchar_t * format, va_list args) { return _vscwprintf (format, args); }
    inline size_t _cprintf (const  char   * format, va_list args) { return _vscprintf  (format, args); }
    inline const wchar_t * _sprintf (wchar_t * buf, size_t bufsiz, const wchar_t * format, va_list args) { vswprintf_s (buf, bufsiz, format, args); return buf; }
    inline const  char   * _sprintf ( char   * buf, size_t bufsiz, const  char   * format, va_list args) { vsprintf_s  (buf, bufsiz, format, args); return buf; }
};

typedef strfun::_strprintf<char>    strprintf;  // char version
typedef strfun::_strprintf<wchar_t> wstrprintf; // wchar_t version

#endif

//http://www.nanobit.net/putty/doxy/PUTTY_8H-source.html
#ifndef CP_UTF8
#define CP_UTF8 65001
#endif
// string-encoding conversion functions
#ifdef _WIN32
struct utf8 : std::string { utf8 (const std::wstring & p)    // utf-16 to -8
{
    size_t len = p.length();
    if (len == 0) { return;}    // empty string
    msra::basetypes::fixed_vector<char> buf (3 * len + 1);   // max: 1 wchar => up to 3 mb chars
    // ... TODO: this fill() should be unnecessary (a 0 is appended)--but verify
    std::fill (buf.begin (), buf.end (), 0);
    int rc = WideCharToMultiByte (CP_UTF8, 0, p.c_str(), (int) len,
                                  &buf[0], (int) buf.size(), NULL, NULL);
    if (rc == 0) throw std::runtime_error ("WideCharToMultiByte");
    (*(std::string*)this) = &buf[0];
}};
struct utf16 : std::wstring { utf16 (const std::string & p)  // utf-8 to -16
{
    size_t len = p.length();
    if (len == 0) { return;}    // empty string
    msra::basetypes::fixed_vector<wchar_t> buf (len + 1);
    // ... TODO: this fill() should be unnecessary (a 0 is appended)--but verify
    std::fill (buf.begin (), buf.end (), (wchar_t) 0);
    int rc = MultiByteToWideChar (CP_UTF8, 0, p.c_str(), (int) len,
                                  &buf[0], (int) buf.size());
    if (rc == 0) throw std::runtime_error ("MultiByteToWideChar");
    assert (rc < buf.size ());
    (*(std::wstring*)this) = &buf[0];
}};
#endif


#pragma warning(push)
#pragma warning(disable : 4996) // Reviewed by Yusheng Li, March 14, 2006. depr. fn (wcstombs, mbstowcs)
static inline std::string wcstombs (const std::wstring & p)  // output: MBCS
{
    size_t len = p.length();
    msra::basetypes::fixed_vector<char> buf (2 * len + 1); // max: 1 wchar => 2 mb chars
    std::fill (buf.begin (), buf.end (), 0);
    ::wcstombs (&buf[0], p.c_str(), 2 * len + 1);
    return std::string (&buf[0]);
}
static inline std::wstring mbstowcs (const std::string & p)  // input: MBCS
{
    size_t len = p.length();
    msra::basetypes::fixed_vector<wchar_t> buf (len + 1); // max: >1 mb chars => 1 wchar
    std::fill (buf.begin (), buf.end (), (wchar_t) 0);
    OACR_WARNING_SUPPRESS(UNSAFE_STRING_FUNCTION, "Reviewed OK. size checked. [rogeryu 2006/03/21]");
    ::mbstowcs (&buf[0], p.c_str(), len + 1);
    return std::wstring (&buf[0]);
}
#pragma warning(pop)
static inline std::string utf8 (const std::wstring & p) { return msra::strfun::wcstombs (p.c_str()); }   // output: UTF-8... not really
static inline std::wstring utf16 (const std::string & p) { return msra::strfun::mbstowcs(p.c_str()); }   // input: UTF-8... not really



// split and join -- tokenize a string like strtok() would, join() strings together
template<class _T> static inline std::vector<std::basic_string<_T>> split (const std::basic_string<_T> & s, const _T * delim)
{
    std::vector<std::basic_string<_T>> res;
    for (size_t st = s.find_first_not_of (delim); st != std::basic_string<_T>::npos; )
    {
        size_t en = s.find_first_of (delim, st +1);
        if (en == std::basic_string<_T>::npos) en = s.length();
        res.push_back (s.substr (st, en-st));
        st = s.find_first_not_of (delim, en +1);    // may exceed
    }
    return res;
}

template<class _T> static inline std::basic_string<_T> join (const std::vector<std::basic_string<_T>> & a, const _T * delim)
{
    std::basic_string<_T> res;
    for (int i = 0; i < (int) a.size(); i++)
    {
        if (i > 0) res.append (delim);
        res.append (a[i]);
    }
    return res;
}

#ifdef _WIN32
// parsing strings to numbers
static inline int toint (const wchar_t * s)
{
    return _wtoi (s);   // ... TODO: check it
}
#endif
static inline int toint (const char * s)
{
    return atoi (s);    // ... TODO: check it
}
static inline int toint (const std::wstring & s) { return toint (s.c_str()); }

static inline double todouble (const char * s)
{
    char * ep;          // will be set to point to first character that failed parsing
    double value = strtod (s, &ep);
    if (*s == 0 || *ep != 0)
        throw std::runtime_error ("todouble: invalid input string");
    return value;
}

// TODO: merge this with todouble(const char*) above
static inline double todouble (const std::string & s)
{
    s.size();       // just used to remove the unreferenced warning
    
    double value = 0.0;

    // stod supposedly exists in VS2010, but some folks have compilation errors
    // If this causes errors again, change the #if into the respective one for VS 2010.
#if _MSC_VER > 1400 // VS 2010+
    size_t * idx = 0;
    value = std::stod (s, idx);
    if (idx) throw std::runtime_error ("todouble: invalid input string");
#else
    char *ep = 0;   // will be updated by strtod to point to first character that failed parsing
    value = strtod (s.c_str(), &ep);

    // strtod documentation says ep points to first unconverted character OR 
    // return value will be +/- HUGE_VAL for overflow/underflow
    if (ep != s.c_str() + s.length() || value == HUGE_VAL || value == -HUGE_VAL)
        throw std::runtime_error ("todouble: invalid input string");
#endif
    
    return value;
}

static inline double todouble (const std::wstring & s)
{
    wchar_t * endptr;
    double value = wcstod (s.c_str(), &endptr);
    if (*endptr) throw std::runtime_error ("todouble: invalid input string");
    return value;
}

// ----------------------------------------------------------------------------
// tokenizer -- utility for white-space tokenizing strings in a character buffer
// This simple class just breaks a string, but does not own the string buffer.
// ----------------------------------------------------------------------------

class tokenizer : public std::vector<char*>
{
    const char * delim;
public:
    tokenizer (const char * delim, size_t cap) : delim (delim) { reserve (cap); }
    // Usage: tokenizer tokens (delim, capacity); tokens = buf; tokens.size(), tokens[i]
    void operator= (char * buf)
    {
        resize (0);

        // strtok_s not available on all platforms - so backoff to strtok on those
#ifdef strtok_s
        char * context; // for strtok_s()
        for (char * p = strtok_s (buf, delim, &context); p; p = strtok_s (NULL, delim, &context))
            push_back (p);
#else
        for (char * p = strtok (buf, delim); p; p = strtok (NULL, delim))
            push_back (p);
#endif   
    }
};

};};    // namespace
static inline msra::strfun::cstring charpath (const std::wstring & p)
{
#ifdef _WIN32
    return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().to_bytes(p);
#else   // old version, delete once we know it works
    size_t len = p.length();
    std::vector<char> buf(2 * len + 1, 0); // max: 1 wchar => 2 mb chars
    ::wcstombs(buf.data(), p.c_str(), 2 * len + 1);
    return msra::strfun::cstring (&buf[0]);
#endif
}
static inline FILE* _wfopen (const wchar_t * path, const wchar_t * mode) { return fopen(charpath(path), charpath(mode)); }
static inline void Sleep (size_t ms) { std::this_thread::sleep_for (std::chrono::milliseconds (ms)); }


// ----------------------------------------------------------------------------
// wrappers for some basic types (files, handles, timer)
// ----------------------------------------------------------------------------

namespace msra { namespace basetypes {

// FILE* with auto-close; use auto_file_ptr instead of FILE*.
// Warning: do not pass an auto_file_ptr to a function that calls fclose(),
// except for fclose() itself.
class auto_file_ptr
{
    FILE * f;
    FILE * operator= (auto_file_ptr &); // can't ref-count: no assignment
    auto_file_ptr (auto_file_ptr &);
    // implicit close (destructor, assignment): we ignore error
    void close()  throw() { if (f) try { if (f != stdin && f != stdout && f != stderr) ::fclose (f); } catch (...) { } f = NULL; }
    void openfailed (const std::string & path) { throw std::runtime_error ("auto_file_ptr: error opening file '" + path + "': " + strerror (errno)); }
protected:
    friend int fclose (auto_file_ptr&); // explicit close (note: may fail)
    int fclose() { int rc = ::fclose (f); if (rc == 0) f = NULL; return rc; }
public:
    auto_file_ptr() : f (NULL) { }
    ~auto_file_ptr() { close(); }
    auto_file_ptr (const char * path, const char * mode) { f = fopen (path, mode); if (f == NULL) openfailed (path); }
    auto_file_ptr (const wchar_t * wpath, const char * mode) { f = _wfopen (wpath, msra::strfun::utf16 (mode).c_str()); if (f == NULL) openfailed (msra::strfun::utf8 (wpath)); }
    FILE * operator= (FILE * other) { close(); f = other; return f; }
    auto_file_ptr (FILE * other) : f (other) { }
    operator FILE * () const { return f; }
    FILE * operator->() const { return f; }
    void swap (auto_file_ptr & other)  throw() { std::swap (f, other.f); }
};
inline int fclose (auto_file_ptr & af) { return af.fclose(); }


};};

namespace msra { namespace files {

// ----------------------------------------------------------------------------
// textreader -- simple reader for text files --we need this all the time!
// Currently reads 8-bit files, but can return as wstring, in which case
// they are interpreted as UTF-8 (without BOM).
// Note: Not suitable for pipes or typed input due to readahead (fixable if needed).
// ----------------------------------------------------------------------------

class textreader
{
    msra::basetypes::auto_file_ptr f;
    std::vector<char> buf;  // read buffer (will only grow, never shrink)
    int ch;                 // next character (we need to read ahead by one...)
    char getch() { char prevch = (char) ch; ch = fgetc (f); return prevch; }
public:
    textreader (const std::wstring & path) : f (path.c_str(), "rb") { buf.reserve (10000); ch = fgetc (f); }
    operator bool() const { return ch != EOF; } // true if still a line to read
    std::string getline()                       // get and consume the next line
    {
        if (ch == EOF) throw std::logic_error ("textreader: attempted to read beyond EOF");
        assert (buf.empty());
        // get all line's characters --we recognize UNIX (LF), DOS (CRLF), and Mac (CR) convention
        while (ch != EOF && ch != '\n' && ch != '\r') buf.push_back (getch());
        if (ch != EOF && getch() == '\r' && ch == '\n') getch();    // consume EOLN char
        std::string line (buf.begin(), buf.end());
        buf.clear();
        return line;
    }
    std::wstring wgetline() { return msra::strfun::utf16 (getline()); }
};

};};

// ----------------------------------------------------------------------------
// functional-programming style helper macros (...do this with templates?)
// ----------------------------------------------------------------------------

#define foreach_index(_i,_dat) for (int _i = 0; _i < (int) (_dat).size(); _i++)
#define map_array(_x,_expr,_y) { _y.resize (_x.size()); foreach_index(_i,_x) _y[_i]=_expr(_x[_i]); }
#define reduce_array(_x,_expr,_y) { foreach_index(_i,_x) _y = (_i==0) ? _x[_i] : _expr(_y,_x[_i]); }

// ----------------------------------------------------------------------------
// frequently missing utility functions
// ----------------------------------------------------------------------------

namespace msra { namespace util {

// to (slightly) simplify processing of command-line arguments.
// command_line args (argc, argv);
// while (args.has (1) && args[0][0] == '-') { option = args.shift(); process (option); }
// for (const wchar_t * arg = args.shift(); arg; arg = args.shift()) { process (arg); }
class command_line
{
    int num;
    const wchar_t * * args;
public:
    command_line (int argc, wchar_t * argv[]) : num (argc), args ((const wchar_t **) argv) { shift(); }
    inline int size() const { return num; }
    inline bool has (int left) { return size() >= left; }
    const wchar_t * shift() { if (size() == 0) return NULL; num--; return *args++; }
    const wchar_t * operator[] (int i) const { return (i < 0 || i >= size()) ? NULL : args[i]; }
};

// byte-reverse a variable --reverse all bytes (intended for integral types and float)
template<typename T> static inline void bytereverse (T & v) throw()
{   // note: this is more efficient than it looks because sizeof (v[0]) is a constant
    char * p = (char *) &v;
    const size_t elemsize = sizeof (v);
    for (int k = 0; k < elemsize / 2; k++)  // swap individual bytes
        swap (p[k], p[elemsize-1 - k]);
}

// byte-swap an entire array
template<class V> static inline void byteswap (V & v) throw()
{
    foreach_index (i, v)
        bytereverse (v[i]);
}

// execute a block with retry
// Block must be restartable.
// Use this when writing small files to those unreliable Windows servers.
// TODO: This will fail to compile under VS 2008--we need an #ifdef around this
template<typename FUNCTION> static void attempt (int retries, const FUNCTION & body)
{
    for (int attempt = 1; ; attempt++)
    {
        try
        {
            body();
            if (attempt > 1) fprintf (stderr, "attempt: success after %d retries\n", attempt);
            break;
        }
        catch (const std::exception & e)
        {
            if (attempt >= retries)
                throw;      // failed N times --give up and rethrow the error
            fprintf (stderr, "attempt: %s, retrying %d-th time out of %d...\n", e.what(), attempt+1, retries);
            ::Sleep (1000); // wait a little, then try again
        }
    }
}

};};    // namespace


#ifdef _WIN32
// ----------------------------------------------------------------------------
// frequently missing Win32 functions
// ----------------------------------------------------------------------------

// strerror() for Win32 error codes
static inline std::wstring FormatWin32Error (DWORD error)
{
    wchar_t buf[1024] = { 0 };
    ::FormatMessageW (FORMAT_MESSAGE_FROM_SYSTEM, "", error, 0, buf, sizeof (buf)/sizeof (*buf) -1, NULL);
    std::wstring res (buf);
    // eliminate newlines (and spaces) from the end
    size_t last = res.find_last_not_of (L" \t\r\n");
    if (last != std::string::npos) res.erase (last +1, res.length());
    return res;
}
// we always wanted this!
#pragma warning (push)
#pragma warning (disable: 6320) // Exception-filter expression is the constant EXCEPTION_EXECUTE_HANDLER
#pragma warning (disable: 6322) // Empty _except block
static inline void SetCurrentThreadName (const char* threadName)
{   // from http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
    ::Sleep(10);
#pragma pack(push,8)
   struct { DWORD dwType; LPCSTR szName; DWORD dwThreadID; DWORD dwFlags; } info = { 0x1000, threadName, (DWORD) -1, 0 };
#pragma pack(pop)
   __try { RaiseException (0x406D1388, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info); }
   __except(EXCEPTION_EXECUTE_HANDLER) { }
}
#pragma warning (pop)

// return a string as a CoTaskMemAlloc'ed memory object
// Returns NULL if out of memory (we don't throw because we'd just catch it outside and convert to HRESULT anyway).
static inline LPWSTR CoTaskMemString (const wchar_t * s)
{
    size_t n = wcslen (s) + 1;  // number of chars to allocate and copy
    LPWSTR p = (LPWSTR) ::CoTaskMemAlloc (sizeof (*p) * n);
    if (p) for (size_t i = 0; i < n; i++) p[i] = s[i];
    return p;
}

template<class S> static inline void ZeroStruct (S & s) { memset (&s, 0, sizeof (s)); }

#endif
// ----------------------------------------------------------------------------
// machine dependent
// ----------------------------------------------------------------------------

#define MACHINE_IS_BIG_ENDIAN (false)

using namespace msra::basetypes;    // for compatibility

#pragma warning (pop)

// RuntimeError - throw a std::runtime_error with a formatted error string
#ifdef _MSC_VER
__declspec(noreturn)
#endif
static inline void RuntimeError(const char * format, ...)
{
    va_list args;
    char buffer[1024];

    va_start(args, format);
    vsprintf(buffer, format, args);
    throw std::runtime_error(buffer);
};

// LogicError - throw a std::logic_error with a formatted error string
#ifdef _MSC_VER
__declspec(noreturn)
#endif
static inline void LogicError(const char * format, ...)
{
    va_list args;
    char buffer[1024];

    va_start(args, format);
    vsprintf(buffer, format, args);
    throw std::logic_error(buffer);
};

// ----------------------------------------------------------------------------
// dynamic loading of modules
// ----------------------------------------------------------------------------

#ifdef _WIN32
class Plugin
{
    HMODULE m_hModule;      // module handle for the writer DLL
    std::wstring m_dllName; // name of the writer DLL
public:
    Plugin() { m_hModule = NULL; }
    template<class STRING>  // accepts char (UTF-8) and wide string 
    FARPROC Load(const STRING & plugin, const std::string & proc)
    {
        m_dllName = msra::strfun::utf16(plugin);
        m_dllName += L".dll";
        m_hModule = LoadLibrary(m_dllName.c_str());
        if (m_hModule == NULL)
            RuntimeError("Plugin not found: %s", msra::strfun::utf8(m_dllName));

        // create a variable of each type just to call the proper templated version
        return GetProcAddress(m_hModule, proc.c_str());
    }
    ~Plugin() { if (m_hModule) FreeLibrary(m_hModule); }
};
#else
class Plugin
{
public:
    template<class STRING>  // accepts char (UTF-8) and wide string 
    void * Load(const STRING & plugin, const std::string & proc)
    {
        RuntimeError("Plugins not implemented on Linux yet");
        return nullptr;
    }
};
#endif

#endif    // _BASETYPES_
