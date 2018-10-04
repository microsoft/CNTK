// Basics.h -- some shared generally useful pieces of code used by CNTK
//
// We also include a simple "emulation" layer for some proprietary MSVC CRT functions.

#pragma once

#ifndef _BASICS_H_
#define _BASICS_H_

#include "Platform.h"
#include "ExceptionWithCallStack.h"
#include "StringUtil.h"
#include <cmath>
#include <string>
#include <vector>
#include <assert.h>
#include <stdarg.h>
#ifdef _WIN32
#include <Windows.h>
#undef max
#endif
#if __unix__
#include <dlfcn.h> // for Plugin
#endif
#include <cctype>
#include <cwctype>

#define TWO_PI 6.283185307f // TODO: find the official standards-confirming definition of this and use it instead

#define EPSILON 1e-5
#define ISCLOSE(a, b, threshold) (std::abs(a - b) < threshold) ? true : false
#define DLCLOSE_SUCCESS 0

#define UNUSED(x) (void)(x) // for variables that are, e.g., only used in _DEBUG builds

#pragma warning(disable : 4702) // disable some incorrect unreachable-code warnings

#define DISABLE_COPY_AND_MOVE(TypeName)            \
    TypeName(const TypeName&) = delete;            \
    TypeName& operator=(const TypeName&) = delete; \
    TypeName(TypeName&&) = delete;                 \
    TypeName& operator=(TypeName&&) = delete

#ifndef let
#define let const auto  // let x = ... ; let & r = ...
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// -----------------------------------------------------------------------
// ThrowFormatted() - template function to throw a std::exception with a formatted error string
// -----------------------------------------------------------------------
template <class E>
__declspec_noreturn static inline void ThrowFormattedVA(const char* format, va_list args)
{
    // Note: The call stack will skip 2 levels to suppress this function and its call sites (XXXError()).
    //       If more layers are added here, it would have to be adjusted.
    auto callstack = DebugUtil::GetCallStack(/*skipLevels=*/2, /*makeFunctionNamesStandOut=*/true);

    va_list args_copy;
    va_copy(args_copy, args);
    auto size = vsnprintf(nullptr, 0, format, args) + 1;

    string buffer("Unknown error.");
    if (size > 0)
    {
        buffer = string(size, ' ');
        if (vsnprintf(&buffer[0], size, format, args_copy) < 0)
        {
            buffer = string("Unknown error.");
        }
    }
    
    va_end(args_copy);

    fprintf(stderr, "\nAbout to throw exception '%s'\n", buffer.c_str());
    throw ExceptionWithCallStack<E>(buffer, callstack);
}

#ifndef _MSC_VER // TODO: what is the correct trigger for gcc?
template <class E>
__declspec_noreturn void ThrowFormatted(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif


template <class E>
__declspec_noreturn static inline void ThrowFormatted(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    ThrowFormattedVA<E>(format, args);
    va_end(args);
};

// RuntimeError - throw a std::runtime_error with a formatted error string
#ifndef _MSC_VER // gcc __attribute__((format(printf())) does not percolate through variadic templates; so must go the macro route
#ifndef RuntimeError
#define RuntimeError ThrowFormatted<std::runtime_error>
#endif
#ifndef LogicError
#define LogicError ThrowFormatted<std::logic_error>
#endif
#ifndef InvalidArgument
#define InvalidArgument ThrowFormatted<std::invalid_argument>
#endif
#else
template <class... _Types>
__declspec_noreturn static inline void RuntimeError(const char* format, _Types&&... _Args)
{
    ThrowFormatted<std::runtime_error>(format, forward<_Types>(_Args)...);
}
template <class... _Types>
__declspec_noreturn static inline void LogicError(const char* format, _Types&&... _Args)
{
    ThrowFormatted<std::logic_error>(format, forward<_Types>(_Args)...);
}
template <class... _Types>
__declspec_noreturn static inline void InvalidArgument(const char* format, _Types&&... _Args)
{
    ThrowFormatted<std::invalid_argument>(format, forward<_Types>(_Args)...);
}
#endif

// Warning - warn with a formatted error string
#pragma warning(push)
#pragma warning(disable : 4996)
static inline void Warning(const char* format, ...)
{
    va_list args;
    char buffer[1024];

    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);
};
#pragma warning(pop)
static inline void Warning(const string& message)
{
    Warning("%s", message.c_str());
}

#ifndef NOT_IMPLEMENTED
#define NOT_IMPLEMENTED                                                                                                              \
    \
{                                                                                                                             \
        fprintf(stderr, "Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__); \
        LogicError("Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.", __FILE__, __LINE__, __FUNCTION__);      \
    \
}
#endif


// Computes the smallest multiple of k greater or equal to n
inline size_t AsMultipleOf(size_t n, size_t k) { return n + (k - n % k) % k; }

}}}

#ifndef _MSC_VER
using Microsoft::MSR::CNTK::ThrowFormatted;
#else
using Microsoft::MSR::CNTK::RuntimeError;
using Microsoft::MSR::CNTK::LogicError;
using Microsoft::MSR::CNTK::InvalidArgument;
#endif

#ifdef _MSC_VER
#include <codecvt> // std::codecvt_utf8
#endif

namespace msra {

namespace strfun
{ // TODO: rename this

// ----------------------------------------------------------------------------
// (w)cstring -- helper class like std::string but with auto-cast to char*
// and also implements an sprintf variant for STL strings
// ----------------------------------------------------------------------------

// a class that can return a std::string with auto-convert into a const char*
template <typename C>
struct basic_cstring : public std::basic_string<C>
{
    template <typename S>
    basic_cstring(S p)
        : std::basic_string<C>(p)
    {
    }
    operator const C*() const
    {
        return this->c_str();
    }
};
typedef basic_cstring<char> cstring;
typedef basic_cstring<wchar_t> wcstring;

// [w]strprintf() -- like sprintf() but resulting in a C++ string
template <class _T>
struct _strprintf : public std::basic_string<_T>
{ // works for both wchar_t* and char*
    _strprintf(const _T* format, ...)
    {
        va_list args;
        va_start(args, format);            // varargs stuff
        size_t n = _cprintf(format, args); // num chars excl. '\0'
        va_end(args);
        va_start(args, format);
        const int FIXBUF_SIZE = 128; // incl. '\0'
        if (n < FIXBUF_SIZE)
        {
            _T fixbuf[FIXBUF_SIZE];
            this->assign(_sprintf(&fixbuf[0], sizeof(fixbuf) / sizeof(*fixbuf), format, args), n);
        }
        else // too long: use dynamically allocated variable-size buffer
        {
            std::vector<_T> varbuf(n + 1); // incl. '\0'
            this->assign(_sprintf(&varbuf[0], varbuf.size(), format, args), n);
        }
        va_end(args);
    }

private:
    // helpers
    inline size_t _cprintf(const wchar_t* format, va_list args)
    {
#ifdef _MSC_VER
        return _vscwprintf(format, args);
#elif defined(__UNIX__)
        const int BUF_SIZE = 256;
        int n = 0;
        std::vector<wchar_t> vec(BUF_SIZE);
        do {
            vec.resize(vec.size() * 2);
            va_list args2;
            va_copy(args2, args);
            n = vswprintf(&vec[0], vec.size(), format, args2);
            va_end(args2);
        } while (n < 0);
        return n;
#endif
    }
    inline size_t _cprintf(const char* format, va_list args)
    {
#ifdef _MSC_VER
        return _vscprintf(format, args);
#elif defined(__UNIX__)
        int n = vsnprintf(NULL, 0, format, args);
        if (n < 0)
        {
            perror("The following error occurred in basetypes.h:cprintf");
        }
        return n;
#endif
    }
    inline const wchar_t* _sprintf(wchar_t* buf, size_t bufsiz, const wchar_t* format, va_list args)
    {
        vswprintf(buf, bufsiz, format, args);
        return buf;
    }
    inline const char* _sprintf(char* buf, size_t bufsiz, const char* format, va_list args)
    {
#ifdef _MSC_VER
        vsprintf_s(buf, bufsiz, format, args);
#else
        vsprintf(buf, format, args);
#endif
        return buf;
    }
};

// ----------------------------------------------------------------------------
// (w)strprintf() -- sprintf() that returns an STL string
// ----------------------------------------------------------------------------

typedef ::msra::strfun::_strprintf<char>    strprintf;  // char version
typedef ::msra::strfun::_strprintf<wchar_t> wstrprintf; // wchar_t version

// ----------------------------------------------------------------------------
// charpath() -- convert a wchar_t path to what gets passed to CRT functions that take narrow characters
// This is needed for the Linux CRT which does not accept wide-char strings for pathnames anywhere.
// Always use this function for mapping the paths.
// TODO: This does not seem to work well, most places use wtocharpath() instead. Maybe we can remove this.
// ----------------------------------------------------------------------------

static inline cstring charpath(const std::wstring& p)
{
    return Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(p));
}

// ----------------------------------------------------------------------------
// split and join -- tokenize a string like strtok() would, join() strings together
// ----------------------------------------------------------------------------

template <class _T>
static inline std::vector<std::basic_string<_T>> split(const std::basic_string<_T>& s, const _T* delim)
{
    std::vector<std::basic_string<_T>> res;
    for (size_t st = s.find_first_not_of(delim); st != std::basic_string<_T>::npos;)
    {
        size_t en = s.find_first_of(delim, st + 1);
        if (en == std::basic_string<_T>::npos)
            en = s.length();
        res.push_back(s.substr(st, en - st));
        st = s.find_first_not_of(delim, en + 1); // may exceed
    }
    return res;
}

template <class _T>
static inline std::basic_string<_T> join(const std::vector<std::basic_string<_T>>& a, const _T* delim)
{
    std::basic_string<_T> res;
    for (int i = 0; i < (int) a.size(); i++)
    {
        if (i > 0)
            res.append(delim);
        res.append(a[i]);
    }
    return res;
}

// ----------------------------------------------------------------------------
// find and replace
// ----------------------------------------------------------------------------

template<class String>
// actual operations that we perform
static String ReplaceAll(const String& s, const String& what, const String& withwhat)
{
    String res = s;
    auto pos = res.find(what);
    while (pos != String::npos)
    {
        res = res.substr(0, pos) + withwhat + res.substr(pos + what.size());
        pos = res.find(what, pos + withwhat.size());
    }
    return res;
}

// ----------------------------------------------------------------------------
// parsing strings to numbers
// ----------------------------------------------------------------------------

static inline int toint(const wchar_t* s)
{
    return _wtoi(s);
}
static inline int toint(const char* s)
{
    return atoi(s);
}
static inline int toint(const std::wstring& s)
{
    return toint(s.c_str());
}

static inline double todouble(const char* s)
{
    char* ep; // will be set to point to first character that failed parsing
    double value = strtod(s, &ep);
    if (*s == 0 || *ep != 0)
        RuntimeError("todouble: invalid input string '%s'", s);
    return value;
}

// TODO: merge this with todouble(const char*) above
static inline double todouble(const std::string& s)
{
    double value = 0.0;

    size_t* idx = 0;
    value = std::stod(s, idx);
    if (idx)
        RuntimeError("todouble: invalid input string '%s'", s.c_str());

    return value;
}

static inline double todouble(const std::wstring& s)
{
    wchar_t* endptr;
    double value = wcstod(s.c_str(), &endptr);
    if (*endptr)
        RuntimeError("todouble: invalid input string '%ls'", s.c_str());
    return value;
}

// ----------------------------------------------------------------------------
// tokenizer -- utility for white-space tokenizing strings in a character buffer
// This simple class just breaks a string, but does not own the string buffer.
// ----------------------------------------------------------------------------

class tokenizer : public std::vector<char*>
{
    const char* delim;

public:
    tokenizer(const char* delim, size_t cap)
        : delim(delim)
    {
        reserve(cap);
    }
    // Usage: tokenizer tokens (delim, capacity); tokens = buf; tokens.size(), tokens[i]
    void operator=(char* buf)
    {
        resize(0);

// strtok_s not available on all platforms - so backoff to strtok on those
#ifdef _MSC_VER
        char* context; // for strtok_s()
        for (char* p = strtok_s(buf, delim, &context); p; p = strtok_s(NULL, delim, &context))
            push_back(p);
#else
        for (char* p = strtok(buf, delim); p; p = strtok(NULL, delim))
            push_back(p);
#endif
    }
};

}}

// ----------------------------------------------------------------------------
// iscalpha(), iscspace(), etc.: saner version of ctype helpers that do not blow up upon negative signed char values
// Name differs in using 'c' between is- and -what(). Also returns bool instead of int.
// TODO: switch all uses if isspace() etc. to this once tested well
// ----------------------------------------------------------------------------

#define DefineIsCType(pred) \
  static inline bool isc ## pred(char c)    { return !!is  ## pred((unsigned char) c); } \
  static inline bool isc ## pred(wchar_t c) { return !!isw ## pred(c); }
DefineIsCType(alpha);
DefineIsCType(upper);
DefineIsCType(lower);
DefineIsCType(cntrl);
DefineIsCType(digit);
DefineIsCType(punct);
DefineIsCType(space);

// ----------------------------------------------------------------------------
// functional-programming style helper macros (...do this with templates?)
// ----------------------------------------------------------------------------

#define foreach_index(_i, _dat) for (int _i = 0; _i < (int) (_dat).size(); _i++)
#define map_array(_x, _expr, _y)    \
    {                               \
        _y.resize(_x.size());       \
        foreach_index (_i, _x)      \
            _y[_i] = _expr(_x[_i]); \
    }
#define reduce_array(_x, _expr, _y)                      \
    {                                                    \
        foreach_index (_i, _x)                           \
            _y = (_i == 0) ? _x[_i] : _expr(_y, _x[_i]); \
    }

namespace Microsoft { namespace MSR { namespace CNTK {

// ----------------------------------------------------------------------------
// case-insensitive string-comparison helpers
// ----------------------------------------------------------------------------

// normalize between char* and string
// Note: Intended for use within a single expression. Otherwise be aware of memory ownership; same restrictions apply as for string::c_str().
static inline const char    * c_str(const char *    p) { return p;         }
static inline const char    * c_str(const string &  p) { return p.c_str(); }
static inline const wchar_t * c_str(const wchar_t * p) { return p;         }
static inline const wchar_t * c_str(const wstring & p) { return p.c_str(); }

// compare strings
static inline int CompareCI(const char    * a, const char    * b) { return _stricmp(a, b); }
static inline int CompareCI(const wchar_t * a, const wchar_t * b) { return _wcsicmp(a, b); }

template<typename S1, typename S2>
static inline int CompareCI(const S1 & a, const S2 & b) { return CompareCI(c_str(a), c_str(b)); }

// compare for equality
template<typename S1, typename S2>
static inline bool EqualCI(const S1 & a, const S2 & b) { return CompareCI(a, b) == 0; }

// comparer class for defining maps with case-insensitive key lookup
struct nocase_compare
{
    // std::string version of 'less' function
    template<typename S1, typename S2>
    bool operator()(const S1& left, const S2& right) const
    {
        return CompareCI(left, right) < 0;
    }
};

// ----------------------------------------------------------------------------
// random collection of stuff we needed at some place
// ----------------------------------------------------------------------------

// Array class
// Wrapper that holds pointer to data, as well as size
template <class T>
class ArrayRef
{
    T* elements; // Array of type T
    size_t count;

public:

    ArrayRef(T* elementsIn, size_t sizeIn)
    {
        elements = elementsIn;
        count = sizeIn;
    }

    // TODO: Copy Constructor
    ArrayRef(const ArrayRef& other) = delete;

    // TODO: Move Constructor
    ArrayRef(ArrayRef&& other) = delete;

    // TODO: Assignment operator
    ArrayRef& operator=(const ArrayRef& rhs) = delete;

    // TODO: Move assignment operator
    ArrayRef& operator=(ArrayRef&& rhs) = delete;

    size_t size()  const { return count; }
    void setSize(size_t size) { count = size; }

    T* data() const { return elements; }

    T operator[](size_t i) const
    {
        if (i >= size())
            LogicError("ArrayRef: index overflow");
        return elements[i];
    }

    T& operator[](size_t i)
    {
        if (i >= count)
            LogicError("ArrayRef: index overflow");
        return elements[i];
    }

    const T* begin() const
    {
        return data();
    }
    const T* end() const
    {
        return data() + size();
    }
};

// TODO: maybe change to type id of an actual thing we pass in
// TODO: is this header appropriate?
template <class C>
static wstring TypeId()
{
    return Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(typeid(C).name());
}

// ----------------------------------------------------------------------------
// dynamic loading of modules  --TODO: not Basics, should move to its own header
// ----------------------------------------------------------------------------

#ifdef _WIN32
class Plugin
{
    HMODULE m_hModule;      // module handle for the writer DLL
    std::wstring m_dllName; // name of the writer DLL
public:
    Plugin()
        : m_hModule(NULL)
    {
    }
    FARPROC Load(const std::string& plugin, const std::string& proc, bool isCNTKPlugin = true)
    {
        return LoadInternal(Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(plugin), proc, isCNTKPlugin);
    }
    FARPROC Load(const std::wstring& plugin, const std::string& proc, bool isCNTKPlugin = true)
    {
        return LoadInternal(plugin, proc, isCNTKPlugin);
    }
    ~Plugin()
    {
    }
    // we do not unload because this causes the exception messages to be lost (exception vftables are unloaded when DLL is unloaded)
    // ~Plugin() { if (m_hModule) FreeLibrary(m_hModule); }

private:
    FARPROC LoadInternal(const std::wstring& plugin, const std::string& proc, bool isCNTKPlugin);
};
#else
class Plugin
{
private:
    void* handle;

public:
    Plugin()
        : handle(NULL)
    {
    }
    template <class STRING> // accepts char (UTF-8) and wide string
    void *Load(const STRING& plugin, const std::string& proc, bool isCNTKPlugin = true)
    {
        return LoadInternal(Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(plugin)), proc, isCNTKPlugin);
    }
    ~Plugin()
    {
        if (handle != NULL)
        {
            int rc = dlclose(handle);
            if ((rc != DLCLOSE_SUCCESS) && !std::uncaught_exception())
            {
                RuntimeError("Plugin: Failed to decrements the reference count.");
            }
        }
    }

private:
    void *LoadInternal(const std::string& plugin, const std::string& proc, bool isCNTKPlugin);
};
#endif

template <typename EF>
struct ScopeExit {
    explicit ScopeExit(EF &&f) :
        m_exitFunction(std::move(f)), m_exitOnDestruction(true) 
    {}

    ~ScopeExit() 
    {
        if (m_exitOnDestruction)
            m_exitFunction(); 
    }

    ScopeExit(ScopeExit&& other)
        : m_exitFunction(std::move(other.m_exitFunction)), m_exitOnDestruction(other.m_exitOnDestruction)
    {
        other.m_exitOnDestruction = false;
    }

private:
    // Disallow copy construction, assignment
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

    // Disallow move assignment
    ScopeExit& operator=(ScopeExit&&) = delete;

    EF m_exitFunction;
    bool m_exitOnDestruction;
};

template <typename EF>
ScopeExit<typename std::remove_reference<EF>::type> MakeScopeExit(EF&& exitFunction)
{
    return ScopeExit<typename std::remove_reference<EF>::type>(std::forward<EF>(exitFunction));
}
}}}

#ifdef _WIN32
// ----------------------------------------------------------------------------
// frequently missing Win32 functions
// ----------------------------------------------------------------------------

// strerror() for Win32 error codes
static inline std::wstring FormatWin32Error(DWORD error)
{
    wchar_t buf[1024] = {0};
    ::FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM, "", error, 0, buf, sizeof(buf) / sizeof(*buf) - 1, NULL);
    std::wstring res(buf);
    // eliminate newlines (and spaces) from the end
    size_t last = res.find_last_not_of(L" \t\r\n");
    if (last != std::string::npos)
        res.erase(last + 1, res.length());
    return res;
}
#endif // _WIN32

#endif // _BASICS_H_
