// 
// basetypes.h - basic types that C++ lacks
// 
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// 
#pragma once
#ifndef _BASETYPES_
#define _BASETYPES_

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
#ifdef _M_CEE_SAFE 
#define WINAPI_CC __clrcall
#elif _M_CEE
#define WINAPI_CC __clrcall
#else
#define WINAPI_CC __stdcall
#endif

// fix some warnings in STL
#if !defined(_DEBUG) || defined(_CHECKED) || defined(_MANAGED)
#pragma warning(disable : 4702) // unreachable code
#endif

#include "Platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     // include here because we redefine some names later
#include <errno.h>
#include <string>
#include <vector>
#include <math.h>        // for HUGE_VAL // potential double isnan definition
#include <assert.h>
#include <stdarg.h>
#include <map>
#include <stdexcept>
#include <locale>       // std::wstring_convert
#include <string>
#include <algorithm>    // for transform()
#ifdef _MSC_VER
#include <codecvt>      // std::codecvt_utf8
#endif
#ifdef _WIN32
#include <windows.h>    // for CRITICAL_SECTION and Unicode conversion functions   --TODO: is there a portable alternative?
#include <unordered_map>

#endif
#if __unix__
#include <strings.h>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <unordered_map>

typedef unsigned char byte;
#endif

using namespace std;

// CRT error handling seems to not be included in wince headers
// so we define our own imports
#ifdef UNDER_CE

// TODO: is this true - is GetLastError == errno?? - also this adds a dependency on windows.h
#define errno GetLastError() 

// strerror(x) - x here is normally errno - TODO: make this return errno as a string
#define strerror(x) "strerror error but can't report error number sorry!"
#endif

// disable warnings for which fixing would make code less readable
#pragma warning(disable : 4290) //  throw() declaration ignored
#pragma warning(disable : 4244) // conversion from typeA to typeB, possible loss of data

// ----------------------------------------------------------------------------
// (w)cstring -- helper class like std::string but with auto-cast to char*
// ----------------------------------------------------------------------------

namespace msra { namespace strfun {
    // a class that can return a std::string with auto-convert into a const char*
    template<typename C> struct basic_cstring : public std::basic_string<C>
    {
        template<typename S> basic_cstring (S p) : std::basic_string<C> (p) { }
        operator const C * () const { return this->c_str(); }
    };
    typedef basic_cstring<char> cstring;
    typedef basic_cstring<wchar_t> wcstring;
}}
static inline wchar_t*GetWC(const char *c)
{
    const size_t cSize = strlen(c)+1;
    wchar_t* wc = new wchar_t[cSize];
    mbstowcs (wc, c, cSize);

    return wc;
}
struct MatchPathSeparator
{
    bool operator()( char ch ) const
    {
        return ch == '\\' || ch == '/';
    }
};
static inline std::string basename( std::string const& pathname)
{
    return std::string (std::find_if(pathname.rbegin(), pathname.rend(),MatchPathSeparator()).base(), pathname.end()); 
}

static inline std::string removeExtension (std::string const& filename)
{
    //std::string::const_reverse_iterator pivot = std::find(filename.rbegin(), filename.rend(), '.');
    //return pivot == filename.rend() ? filename: std::string(filename.begin(), pivot.base()-1);
    int lastindex = filename.find_first_of(".");
    return filename.substr(0,lastindex);
}
static inline std::wstring basename( std::wstring const& pathname)
{
    return std::wstring (std::find_if(pathname.rbegin(), pathname.rend(),MatchPathSeparator()).base(), pathname.end()); 
}

static inline std::wstring removeExtension (std::wstring const& filename)
{
    //std::wstring::const_reverse_iterator pivot = std::find(filename.rbegin(), filename.rend(), '.');
    //return pivot == filename.rend() ? filename: std::wstring(filename.begin(), pivot.base()-1);
    int lastindex = filename.find_first_of(L".");
    return filename.substr(0,lastindex);

}

// ----------------------------------------------------------------------------
// some mappings for non-Windows builds
// ----------------------------------------------------------------------------

#ifndef _MSC_VER    // add some functions that are VS-only
// --- basic file functions
// convert a wchar_t path to what gets passed to CRT functions that take narrow characters
// This is needed for the Linux CRT which does not accept wide-char strings for pathnames anywhere.
// Always use this function for mapping the paths.
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
static inline int _wunlink (const wchar_t * p) { return unlink (charpath (p)); }
static inline int _wmkdir (const wchar_t * p) { return mkdir (charpath (p), 0777/*correct?*/); }
// --- basic string functions
static inline wchar_t* wcstok_s (wchar_t* s, const wchar_t* delim, wchar_t** ptr) { return ::wcstok(s, delim, ptr); }
static inline int _stricmp  (const char * a, const char * b)                 { return ::strcasecmp (a, b); }
static inline int _strnicmp (const char * a, const char * b, size_t n)       { return ::strncasecmp (a, b, n); }
static inline int _wcsicmp  (const wchar_t * a, const wchar_t * b)           { return ::wcscasecmp (a, b); }
static inline int _wcsnicmp (const wchar_t * a, const wchar_t * b, size_t n) { return ::wcsncasecmp (a, b, n); }
static inline int64_t  _strtoi64  (const char * s, char ** ep, int r) { return strtoll (s, ep, r); }    // TODO: check if correct
static inline uint64_t _strtoui64 (const char * s, char ** ep, int r) { return strtoull (s, ep, r); }   // TODO: correct for size_t?
// -- other
//static inline void memcpy_s(void * dst, size_t dstsize, const void * src, size_t maxcount) { assert (maxcount <= dstsize); memcpy (dst, src, maxcount); }
static inline void Sleep (size_t ms) { std::this_thread::sleep_for (std::chrono::milliseconds (ms)); }
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

// ----------------------------------------------------------------------------
// basic macros   --TODO: do we need those? delete what we dont' need
// ----------------------------------------------------------------------------

//#define SAFE_DELETE(p)  { if(p) { delete (p); (p)=NULL; } }
//#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }     // nasty! use CComPtr<>

// ----------------------------------------------------------------------------
// basic data types
// ----------------------------------------------------------------------------

namespace msra { namespace basetypes {


// class fixed_vector - non-resizable vector

template<class _T> class fixed_vector
{
    _T * p;                 // pointer array
    size_t n;               // number of elements
    void check (int index) const { index/*avoid compiler warning*/;assert (index >= 0 && (size_t) index < n); }
    void check (size_t index) const { assert (index < n); }
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
    inline void swap (fixed_vector & other)  throw() { std::swap (other.p, p); std::swap (other.n, n); }
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
template<class _T> inline void swap (fixed_vector<_T> & L, fixed_vector<_T> & R)  throw() { L.swap (R); }

#pragma warning(pop)    // pop off waring: expression has no effect

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
    void swap (matrix & other)  throw() { std::swap (numcols, other.numcols); fixed_vector<T>::swap (other); }
};
template<class _T> inline void swap (matrix<_T> & L, matrix<_T> & R)  throw() { L.swap (R); }

// TODO: get rid of these
typedef std::string STRING;
typedef std::wstring WSTRING;

// derive from this for noncopyable classes (will get you private unimplemented copy constructors)
// ... TODO: change all of basetypes classes/structs to use this
class noncopyable
{
    noncopyable & operator= (const noncopyable &);
    noncopyable (const noncopyable &);
public:
    noncopyable(){}
};

// class CCritSec and CAutoLock -- simple critical section handling
#ifndef    _WIN32          // TODO: Currently only working under Windows; BROKEN otherwise, to be fixed
typedef int CRITICAL_SECTION;
static inline void InitializeCriticalSection(CRITICAL_SECTION *) {}
static inline void DeleteCriticalSection(CRITICAL_SECTION *) {}
static inline void EnterCriticalSection(CRITICAL_SECTION *) {}
static inline void LeaveCriticalSection(CRITICAL_SECTION *) {}
#endif
class CCritSec
{
    CCritSec (const CCritSec &); CCritSec & operator= (const CCritSec &);
    CRITICAL_SECTION m_CritSec;
public:
    CCritSec() { InitializeCriticalSection(&m_CritSec); };
    ~CCritSec() { DeleteCriticalSection(&m_CritSec); };
    void Lock() { EnterCriticalSection(&m_CritSec); };
    void Unlock() { LeaveCriticalSection(&m_CritSec); };
};


// locks a critical section, and unlocks it automatically
// when the lock goes out of scope
class CAutoLock
{
    CAutoLock(const CAutoLock &refAutoLock); CAutoLock &operator=(const CAutoLock &refAutoLock);
    CCritSec & m_rLock;
public:
    CAutoLock(CCritSec & rLock) : m_rLock (rLock) { m_rLock.Lock(); };
    ~CAutoLock() { m_rLock.Unlock(); };
};

#if 0
// an efficient way to write COM code
// usage examples:
//  COM_function() || throw_hr ("message");
//  while ((s->Read (p, n, &m) || throw_hr ("Read failure")) == S_OK) { ... }
// is that cool or what?
struct bad_hr : public std::runtime_error
{
    HRESULT hr;
    bad_hr (HRESULT p_hr, const char * msg) : hr (p_hr), std::runtime_error (msg) { }
    // (only for use in || expression  --deprecated:)
    bad_hr() : std::runtime_error(NULL) { }
    bad_hr(const char * msg) : std::runtime_error(msg) { }
};
struct throw_hr
{
    const char * msg;
    inline throw_hr (const char * msg = NULL) : msg (msg) {}
};
inline static HRESULT operator|| (HRESULT hr, const throw_hr & e)
{
    if (SUCCEEDED (hr)) return hr;
    throw bad_hr (hr, e.msg);
}
// (old deprecated version kept for compat:)
inline static bool operator|| (HRESULT hr, const bad_hr & e) { if (SUCCEEDED (hr)) return true; throw bad_hr (hr, e.what()); }

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

// CoInitializeEx() wrapper to ensure CoUnintialize()
//struct auto_co_initialize : noncopyable
//{
//    auto_co_initialize() { ::CoInitializeEx (NULL, COINIT_MULTITHREADED) || bad_hr ("auto_co_initialize: CoInitializeEx failure"); }
//    ~auto_co_initialize() { ::CoUninitialize(); }
//};

// auto pointer for ::CoTaskMemFree
template<class T> class auto_co_ptr : noncopyable
{
    T * p;
public:
    auto_co_ptr() : p (NULL) { }
    auto_co_ptr (T * p) : p (p) { }
//    ~auto_co_ptr() { ::CoTaskMemFree (p); }
    operator T * () const { return p; }
    T * operator->() const { return p; }
    T** operator& () { assert (p == NULL); return &p; }    // must be empty when taking address
};

// represents a thread-local-storage variable
// Note: __declspec(thread) is broken on pre-Vista for delay loaded DLLs
// [http://www.nynaeve.net/?p=187]
// so instead, we need to wrap up the Win32 TLS functions ourselves.
// Note: tls instances must be allocated as static to work correctly, e.g.:
//   static tls myVal();
//   myVal = (void *) 25;
//   printf ("value is %d",(void *) myVal);

class tls
{
private:
    int tlsSlot;
public:

#ifdef UNDER_CE
    // this is from standard windows headers - seems to be missing in WINCE
    #define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif
    tls() { tlsSlot = TlsAlloc(); if (tlsSlot == TLS_OUT_OF_INDEXES) throw std::runtime_error("tls: TlsAlloc failed, out of tls slots"); }
    operator void * () { return TlsGetValue (tlsSlot); }
    void *operator = (void *val) { if (!TlsSetValue (tlsSlot,val)) throw std::runtime_error ("tls: TlsSetValue failed"); return val; }
};
#endif

};};    // namespace

#if 0 //ndef BASETYPES_NO_UNSAFECRTOVERLOAD // if on, no unsafe CRT overload functions

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
#ifndef    LINUX
template<typename _T> inline __declspec(deprecated("Dummy general template, cannot be used directly")) 
#else
template<typename _T> inline 
#endif    // LINUX
size_t strlen_(_T &s) { return strnlen_s(static_cast<const char *>(s), SIZE_MAX); } // never be called but needed to keep compiler happy
template<typename _T> inline size_t strlen_(const _T &s)     { return strnlen_s(static_cast<const char *>(s), SIZE_MAX); }
template<> inline size_t strlen_(char * &s)                  { return strnlen_s(s, SIZE_MAX); }
template<> inline size_t strlen_(const char * &s)            { return strnlen_s(s, SIZE_MAX); }
template<size_t n> inline size_t strlen_(const char (&s)[n]) { return (strnlen_s(s, n)); }
template<size_t n> inline size_t strlen_(char (&s)[n])       { return (strnlen_s(s, n)); }
#define wcslen wcslen_
template<typename _T> inline __declspec(deprecated("Dummy general template, cannot be used directly")) 
size_t wcslen_(_T &s) { return wcsnlen_s(static_cast<const wchar_t *>(s), SIZE_MAX); } // never be called but needed to keep compiler happy
template<typename _T> inline size_t __cdecl wcslen_(const _T &s)        { return wcsnlen_s(static_cast<const wchar_t *>(s), SIZE_MAX); }
template<> inline size_t wcslen_(wchar_t * &s)                  { return wcsnlen_s(s, SIZE_MAX); }
template<> inline size_t wcslen_(const wchar_t * &s)            { return wcsnlen_s(s, SIZE_MAX); }
template<size_t n> inline size_t wcslen_(const wchar_t (&s)[n]) { return (wcsnlen_s(s, n)); }
template<size_t n> inline size_t wcslen_(wchar_t (&s)[n])       { return (wcsnlen_s(s, n)); }

// xscanf wrappers -- one overload for each actual use case in our code base
static inline int sscanf  (const char * buf, const char * format, int * i1)                     { return sscanf_s (buf, format, i1); }
static inline int sscanf  (const char * buf, const char * format, int * i1, int * i2)           { return sscanf_s (buf, format, i1, i2); }
static inline int sscanf  (const char * buf, const char * format, int * i1, int * i2, int * i3) { return sscanf_s (buf, format, i1, i2, i3); }
static inline int sscanf  (const char * buf, const char * format, double * f1)                  { return sscanf_s (buf, format, f1); }
static inline int swscanf (const wchar_t * buf, const wchar_t * format, int * i1)               { return swscanf_s (buf, format, i1); }
static inline int fscanf  (FILE * file, const char * format, float * f1)                        { return fscanf_s (file, format, f1); }

// ...TODO: should we pass 'count' instead of SIZE_MAX? (need to review use cases)
#define _vsnprintf _vsnprintf_
static inline int _vsnprintf_(char *buffer, size_t count, const char *format, va_list argptr)
{ return _vsnprintf_s (buffer, SIZE_MAX, count, format, argptr); }
#define _vsnwprintf _vsnwprintf_
static inline int _vsnwprintf_(wchar_t *buffer, size_t count, const wchar_t *format, va_list argptr)
{ return _vsnwprintf_s (buffer, SIZE_MAX, count, format, argptr); }

// wcsfcpy -- same as standard wcsncpy, use padded fixed-size buffer really needed
static inline void wcsfcpy (wchar_t * dest, const wchar_t * source, size_t count)
{
    while (count && (*dest++ = *source++) != 0) count--;    // copy
    if (count) while (--count) *dest++ = 0;                 // pad with zeroes
}

// cacpy -- fixed-size character array (same as original strncpy (dst, src, sizeof (dst)))
// NOTE: THIS FUNCTION HAS NEVER BEEN TESTED. REMOVE THIS COMMENT ONCE IT HAS.
template<class T, size_t n> static inline void cacpy (T (&dst)[n], const T * src)
{ for (int i = 0; i < n; i++) { dst[i] = *src; if (*src) src++; } }
// { return strncpy (dst, src, n); }   // using original C std lib function

// mappings for "unsafe" functions that are not really unsafe
#define strtok strtok_      // map to "safe" function (adds no value)
static inline /*const*/ char * strtok_(char * s, const char * delim)
{
    static msra::basetypes::tls tls_context; // see note for tls class def
    char *context = (char *) (void *) tls_context;
    char *ret = strtok_s (s, delim, &context);
    tls_context = context;
    return ret;
}

#define wcstok wcstok_      // map to "safe" function (adds no value)
static inline /*const*/ wchar_t * wcstok_(wchar_t * s, const wchar_t * delim) 
{ 
    static msra::basetypes::tls tls_context; // see note for tls class def
    wchar_t *context = (wchar_t *) (void *) tls_context;
    wchar_t *ret = wcstok_s (s, delim, &context);
    tls_context = context;
    return ret;
}

#define fopen fopen_        // map to _fsopen() (adds no value)
static inline FILE * fopen_(const char * p, const char * m) { return _fsopen (p, m, _SH_DENYWR); }
#define _wfopen _wfopen_    // map to _wfsopen() (adds no value)
static inline FILE * _wfopen_(const wchar_t * p, const wchar_t * m) { return _wfsopen (p, m, _SH_DENYWR); }

#define strerror(e) strerror_((e))      // map to "safe" function (adds no value)
static inline const char *strerror_(int e)
{   // keep a cache so we can return a pointer (to mimic the old interface)
    static msra::basetypes::CCritSec cs; static std::map<int,std::string> msgs;
    msra::basetypes::CAutoLock lock (cs);
    if (msgs.find(e) == msgs.end()) { char msg[1024]; strerror_s (msg, e); msgs[e] = msg; }
    return msgs[e].c_str();
}
#endif
#ifdef __unix__
extern int fileno(FILE*);   // somehow got deprecated in C++11
#endif

// ----------------------------------------------------------------------------
// frequently missing string functions
// ----------------------------------------------------------------------------

namespace msra { namespace strfun {

#ifndef BASETYPES_NO_STRPRINTF

/*
#ifdef __UNIX__
static FILE *dummyf = fopen("tmp", "wb");
#endif
*/
// [w]strprintf() -- like sprintf() but resulting in a C++ string
template<class _T> struct _strprintf : public std::basic_string<_T>
{   // works for both wchar_t* and char*
    _strprintf (const _T * format, ...)
    {
        va_list args; 
		va_start (args, format);  // varargs stuff
        size_t n = _cprintf (format, args);     // num chars excl. '\0'
		va_end(args);
		va_start(args, format);
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
    inline size_t _cprintf (const wchar_t * format, va_list args) 
	{ 
#ifdef __WINDOWS__
		return vswprintf (nullptr, 0, format, args);
#elif defined(__UNIX__)
		FILE *dummyf = fopen("/dev/null", "w");
		if (dummyf == NULL)
			perror("The following error occurred in basetypes.h:cprintf");
		int n = vfwprintf (dummyf, format, args);
		if (n < 0)
			perror("The following error occurred in basetypes.h:cprintf");
		fclose(dummyf);
		return n;
#endif
	}
    inline size_t _cprintf (const  char   * format, va_list args) 
	{ 
#ifdef __WINDOWS__
		return vsprintf (nullptr, format, args);
#elif defined(__UNIX__)
		FILE *dummyf = fopen("/dev/null", "wb");
		if (dummyf == NULL)
			perror("The following error occurred in basetypes.h:cprintf");
		int n = vfprintf (dummyf, format, args);
		if (n < 0)
			perror("The following error occurred in basetypes.h:cprintf");
		fclose(dummyf);
		return n;
#endif
	}
    inline const wchar_t * _sprintf (wchar_t * buf, size_t bufsiz,  const wchar_t * format, va_list args) { vswprintf (buf, bufsiz, format, args); return buf; }
    inline const  char   * _sprintf ( char   * buf, size_t /*bufsiz*/, const char * format, va_list args) { vsprintf  (buf, format, args); return buf; }
};
typedef strfun::_strprintf<char>    strprintf;  // char version
typedef strfun::_strprintf<wchar_t> wstrprintf; // wchar_t version

#endif

// string-encoding conversion functions
// Note: generally, 8-bit strings in this codebase are UTF-8.
// One exception are functions that take 8-bit pathnames. Those will be interpreted by the OS as MBS. Best use wstring pathnames for all file accesses.

#pragma warning(push)
#pragma warning(disable : 4996) // Reviewed by Yusheng Li, March 14, 2006. depr. fn (wcstombs, mbstowcs)
static inline std::string wcstombs(const std::wstring & p)  // output: MBCS
{
    size_t len = p.length();
    msra::basetypes::fixed_vector<char> buf(2 * len + 1); // max: 1 wchar => 2 mb chars
    std::fill(buf.begin(), buf.end(), 0);
    ::wcstombs(&buf[0], p.c_str(), 2 * len + 1);
    return std::string(&buf[0]);
}
static inline std::wstring mbstowcs(const std::string & p)  // input: MBCS
{
    size_t len = p.length();
    msra::basetypes::fixed_vector<wchar_t> buf(len + 1); // max: >1 mb chars => 1 wchar
    std::fill(buf.begin(), buf.end(), (wchar_t)0);
    OACR_WARNING_SUPPRESS(UNSAFE_STRING_FUNCTION, "Reviewed OK. size checked. [rogeryu 2006/03/21]");
    ::mbstowcs(&buf[0], p.c_str(), len + 1);
    return std::wstring(&buf[0]);
}
#pragma warning(pop)

#ifdef _WIN32
static inline  cstring  utf8 (const std::wstring & p) { return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().to_bytes(p); }     // utf-16 to -8
static inline wcstring utf16 (const  std::string & p) { return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().from_bytes(p); } // utf-8 to -16
#else   // BUGBUG: we cannot compile the above on Cygwin GCC, so for now fake it using the mbs functions, which will only work for 7-bit ASCII strings
static inline std::string utf8 (const std::wstring & p) { return msra::strfun::wcstombs (p.c_str()); }   // output: UTF-8... not really
static inline std::wstring utf16 (const std::string & p) { return msra::strfun::mbstowcs(p.c_str()); }   // input: UTF-8... not really
#endif
static inline  cstring  utf8 (const  std::string & p) { return p; }     // no conversion (useful in templated functions)
static inline wcstring utf16 (const std::wstring & p) { return p; }

// convert a string to lowercase  --TODO: currently only correct for 7-bit ASCII
template<typename CHAR>
static inline void tolower_ascii (std::basic_string<CHAR> & s) { std::transform(s.begin(), s.end(), s.begin(), [] (CHAR c) { return (c >= 0 && c < 128) ? ::tolower(c) : c; }); }

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

// parsing strings to numbers
static inline int toint (const wchar_t * s)
{
    return (int)wcstol(s, 0, 10);
    //return _wtoi (s);   // ... TODO: test this
}
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
#if __STDC_WANT_SECURE_LIB__
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

#ifdef _MSC_VER
// auto-closing container for Win32 handles.
// Pass close function if not CloseHandle(), e.g.
// auto_handle h (FindFirstFile(...), FindClose);
// ... TODO: the close function should really be a template parameter
template<class _H> class auto_handle_t
{
    _H h;
    BOOL (WINAPI_CC * close) (HANDLE);  // close function
    auto_handle_t operator= (const auto_handle_t &);
    auto_handle_t (const auto_handle_t &);
public:
    auto_handle_t (_H p_h, BOOL (WINAPI_CC * p_close) (HANDLE) = ::CloseHandle) : h (p_h), close (p_close) {}
    ~auto_handle_t() { if (h != INVALID_HANDLE_VALUE) close (h); }
    operator _H () const { return h; }
};
typedef auto_handle_t<HANDLE> auto_handle;
#endif

// like auto_ptr but calls freeFunc_p (type free_func_t) instead of delete to clean up
// minor difference - wrapped object is T, not T *, so to wrap a 
// T *, use auto_clean<T *>
// TODO: can this be used for simplifying those other classes?
template<class T,class FR = void> class auto_clean
{
    T it;
    typedef FR (*free_func_t)(T); 
    free_func_t freeFunc;                           // the function used to free the pointer
    void free() 
    { 
        //printf ("start clean\n");
        if (it) freeFunc(it); it = 0;
    }
    auto_clean operator= (const auto_clean &);      // hide to prevent copy
    auto_clean (const auto_clean &);                // hide to prevent copy
public:
    auto_clean (T it_p, free_func_t freeFunc_p) : it (it_p), freeFunc (freeFunc_p) {}
    ~auto_clean() { free(); }
    operator T () { return it; }
    operator const T () const { return it; }
    T detach () { T tmp = it; it = 0; return tmp; } // release ownership of object
};

#if 1
// simple timer
// auto_timer timer; run(); double seconds = timer; // now can abandon the objecta
#ifdef __unix__
typedef timeval LARGE_INTEGER;
#endif
class auto_timer
{
    LARGE_INTEGER freq, start;
    auto_timer (const auto_timer &); void operator= (const auto_timer &);
public:
    auto_timer()
    {
#ifdef _WIN32
        if (!QueryPerformanceFrequency (&freq)) // count ticks per second
            throw std::runtime_error ("auto_timer: QueryPerformanceFrequency failure");
        QueryPerformanceCounter (&start);
#endif
#ifdef __unix__
        gettimeofday (&start, NULL);
#endif

    }
    operator double() const     // each read gives time elapsed since start, in seconds
    {
        LARGE_INTEGER end;
#ifdef _WIN32
        QueryPerformanceCounter (&end);
        return (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
#endif
#ifdef __unix__
        gettimeofday (&end,NULL);
        return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/(1000*1000);
#endif
    }
    void show (const std::string & msg) const
    {
        double elapsed = *this;
        fprintf (stderr, "%s: %.6f ms\n", msg.c_str(), elapsed * 1000.0/*to ms*/);
    }
};
#endif

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
//template<class _A,class _F>
//static void fill_array(_A & a, _F v) { ::fill (a.begin(), a.end(), v); }

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
    const wchar_t ** args;
public:
    command_line (int argc, wchar_t * argv[]) : num (argc), args ((const wchar_t **) argv) { shift(); }
    inline int size() const { return num; }
    inline bool has (int left) { return size() >= left; }
    const wchar_t * shift() { if (size() == 0) return NULL; num--; return *args++; }
    const wchar_t * operator[] (int i) const { return (i < 0 || i >= size()) ? NULL : args[i]; }
};
 
// byte-reverse a variable --reverse all bytes (intended for integral types and float)
template<typename T> static inline void bytereverse (T & v)  throw()
{   // note: this is more efficient than it looks because sizeof (v[0]) is a constant
    char * p = (char *) &v;
    const size_t elemsize = sizeof (v);
    for (int k = 0; k < elemsize / 2; k++)  // swap individual bytes
        swap (p[k], p[elemsize-1 - k]);
}

// byte-swap an entire array
template<class V> static inline void byteswap (V & v)  throw()
{
    foreach_index (i, v)
        bytereverse (v[i]);
}

//#if 0
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
//#endif

};};    // namespace

template<class S> static inline void ZeroStruct (S & s) { memset (&s, 0, sizeof (s)); }

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
            RuntimeError("Plugin not found: %s", msra::strfun::utf8(m_dllName).c_str());

        // create a variable of each type just to call the proper templated version
        return GetProcAddress(m_hModule, proc.c_str());
    }
    ~Plugin(){} 
    // removed because this causes the exception messages to be lost  (exception vftables are unloaded when DLL is unloaded) 
    // ~Plugin() { if (m_hModule) FreeLibrary(m_hModule); }
};
#else
class Plugin
{
private:
	void *handle;
public:
	Plugin() 
	{ 
		handle = NULL; 
	}

    template<class STRING>  // accepts char (UTF-8) and wide string 
    void * Load(const STRING & plugin, const std::string & proc)
    {
		string soName = msra::strfun::utf8(plugin);
		soName = soName + ".so";
		void *handle = dlopen(soName.c_str(), RTLD_LAZY);
		if (handle == NULL)
            RuntimeError("Plugin not found: %s", soName.c_str());
		return dlsym(handle, proc.c_str());
    }

	~Plugin() {
		if (handle != NULL)
			dlclose(handle);
	}
};
#endif

#if 0   // construction site
// ----------------------------------------------------------------------------
// class RegisterModule
// TODO: move this elsewhere
// ----------------------------------------------------------------------------
#include<functional>
template<typename MODULETYPE>
class RegisterModule
{
    static std::map<std::wstring, std::function<MODULETYPE*()>> & GetFactoryMethodsHash()
    {
        static std::map<std::wstring, std::function<MODULETYPE*()>> FactoryMethods; // shared object
        return FactoryMethods;
    }
public:
    RegisterModule(const std::wstring & ModuleName, std::function<MODULETYPE*()> FactoryMethod)
    {
        auto & FactoryMethods = GetFactoryMethodsHash();
        FactoryMethods[ModuleName] = FactoryMethod;
        // TODO: check for dups, using map::insert()
    }
    static MODULETYPE* Create(const std::wstring & ModuleName)
    {
        auto & FactoryMethods = GetFactoryMethodsHash();
        auto Entry = FactoryMethods.find(ModuleName);
        if (Entry != FactoryMethods.end())
            return Entry->second();
        else
            return nullptr;
    }
};
#endif
#define EPSILON 1e-5
#define ISCLOSE(a, b, threshold) (abs(a - b) < threshold)?true:false

/**
These macros are used for sentence segmentation information. 
*/
#define SEQUENCE_START ((int) MinibatchPackingFlags::SequenceStart)
#define SEQUENCE_MIDDLE ((int) MinibatchPackingFlags::None)
#define SEQUENCE_END ((int) MinibatchPackingFlags::SequenceEnd)
#define NO_INPUT ((int) MinibatchPackingFlags::NoInput)
#define NO_FEATURE ((int) MinibatchPackingFlags::NoFeature)
#define NO_LABEL ((int) MinibatchPackingFlags::NoLabel)

enum class MinibatchPackingFlags : unsigned char
{
    None = 0,
    SequenceStart = 1 << 0,   //binary 0001
    SequenceEnd = 1 << 1,   //binary 0010
    NoFeature = 1 << 2,      //binary 0100
    NoLabel = 1 << 3,      //binary 1000

    NoInput = NoFeature | NoLabel, //when we refactorize reader, NoInput will no longer needed
    SequenceStartOrNoFeature = SequenceStart | NoFeature,
    SequenceEndOrNoFeature = SequenceEnd | NoFeature,
    SequenceStartOrEndOrNoFeature = SequenceStart | SequenceEnd | NoFeature,
};


inline MinibatchPackingFlags operator| (MinibatchPackingFlags a, MinibatchPackingFlags b)
{
    return static_cast<MinibatchPackingFlags>(static_cast<unsigned char>(a) | static_cast<unsigned char>(b));
}

inline MinibatchPackingFlags& operator|= (MinibatchPackingFlags& a, MinibatchPackingFlags b)
{
    a = a | b;
    return a;
}


inline bool operator& (MinibatchPackingFlags a, MinibatchPackingFlags b)
{
    return (static_cast<unsigned char>(a) & static_cast<unsigned char>(b)) != 0;
}

template<class F>
static inline bool comparator(const pair<int, F>& l, const pair<int, F>& r)
{
    return l.second > r.second;
}


#endif    // _BASETYPES_
