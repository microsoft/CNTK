// Basics.h -- some shared generally useful pieces of code used by CNTK
//
// We also include a simple "emulation" layer for some proprietary MSVC CRT functions.

#pragma once

#ifndef _BASICS_H_
#define _BASICS_H_

#include "Platform.h"
#include "DebugUtil.h"
#include <string>
#include <vector>

#define TWO_PI 6.283185307f // TODO: find the official standards-confirming definition of this and use it instead

namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace std;

    template<class E>
    __declspec_noreturn static inline void ThrowFormatted()
    {
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
        throw E();
    }

    template<class E>
    __declspec_noreturn static inline void ThrowFormatted(const string & message)
    {
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
        throw E(message);
    }

    // ThrowFormatted() - template function to throw a std::exception with a formatted error string
#pragma warning(push)
#pragma warning(disable : 4996)
    template<class E>
    __declspec_noreturn static inline void ThrowFormatted(const char * format, ...)
    {
        va_list args;
        char buffer[1024];

        va_start(args, format);
        vsprintf(buffer, format, args);
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
        throw E(buffer);
    };
#pragma warning(pop)

    // RuntimeError - throw a std::runtime_error with a formatted error string
    template<class... _Types>
    __declspec_noreturn static inline void RuntimeError(_Types&&... _Args) { ThrowFormatted<std::runtime_error>(forward<_Types>(_Args)...); }
    template<class... _Types>
    __declspec_noreturn static inline void LogicError(_Types&&... _Args) { ThrowFormatted<std::logic_error>(forward<_Types>(_Args)...); }
    template<class... _Types>
    __declspec_noreturn static inline void InvalidArgument(_Types&&... _Args) { ThrowFormatted<std::invalid_argument>(forward<_Types>(_Args)...); }
    template<class... _Types>
    __declspec_noreturn static inline void BadExceptionError(_Types&&... _Args) 
    {
#ifdef _WIN32
        ThrowFormatted<std::bad_exception>(forward<_Types>(_Args)...);   
#else
        ThrowFormatted<std::bad_exception>();
#endif
    }

    // Warning - warn with a formatted error string
#pragma warning(push)
#pragma warning(disable : 4996)
    static inline void Warning(const char * format, ...)
    {
        va_list args;
        char buffer[1024];

        va_start(args, format);
        vsprintf(buffer, format, args);
    };
#pragma warning(pop)
    static inline void Warning(const string & message) { Warning("%s", message.c_str()); }
}}}

using Microsoft::MSR::CNTK::RuntimeError;
using Microsoft::MSR::CNTK::LogicError;
using Microsoft::MSR::CNTK::InvalidArgument;
using Microsoft::MSR::CNTK::BadExceptionError;

#include "basetypes.h"  // TODO: gradually move over here all that's needed of basetypes.h, then remove basetypes.h.

namespace Microsoft { namespace MSR { namespace CNTK {

    // string comparison class, so we do case insensitive compares
    struct nocase_compare
    {
        // std::string version of 'less' function
        // return false for equivalent, true for different
        bool operator()(const string& left, const string& right) { return _stricmp(left.c_str(), right.c_str()) < 0; }
        // std::wstring version of 'less' function, used in non-config classes
        bool operator()(const wstring& left, const wstring& right) { return _wcsicmp(left.c_str(), right.c_str()) < 0; }
    };

    // ----------------------------------------------------------------------------
    // random collection of stuff we needed at some place
    // ----------------------------------------------------------------------------

    // TODO: maybe change to type id of an actual thing we pass in
    // TODO: is this header appropriate?
    template<class C> static wstring TypeId() { return msra::strfun::utf16(typeid(C).name()); }

    // ----------------------------------------------------------------------------
    // dynamic loading of modules  --TODO: not Basics, should move to its own header
    // ----------------------------------------------------------------------------

#ifdef _WIN32
    class Plugin
    {
        HMODULE m_hModule;      // module handle for the writer DLL
        std::wstring m_dllName; // name of the writer DLL
    public:
        Plugin() : m_hModule(NULL) { }
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
        // we do not unload because this causes the exception messages to be lost (exception vftables are unloaded when DLL is unloaded) 
        // ~Plugin() { if (m_hModule) FreeLibrary(m_hModule); }
    };
#else
    class Plugin
    {
    private:
        void *handle;
    public:
        Plugin() : handle (NULL) { }
        template<class STRING>  // accepts char (UTF-8) and wide string 
        void * Load(const STRING & plugin, const std::string & proc)
        {
            string soName = msra::strfun::utf8(plugin);
            soName = soName + ".so";
            void *handle = dlopen(soName.c_str(), RTLD_LAZY);
            if (handle == NULL)
                RuntimeError("Plugin not found: %s (error: %s)", soName.c_str(), dlerror());
            return dlsym(handle, proc.c_str());
        }
        ~Plugin() { if (handle != NULL) dlclose(handle); }
    };
#endif

}}}

#endif // _BASICS_H_
