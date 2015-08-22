// Basics.h -- some shared generally useful pieces of code used by CNTK
//
// We also include a simple "emulation" layer for some proprietary MSVC CRT functions.

#pragma once

#ifndef _BASICS_H_
#define _BASICS_H_

#include "basetypes.h"  // TODO: gradually move over here all that's needed of basetypes.h, then remove basetypes.h.

#define TWO_PI 6.283185307f // TODO: find the official standards-confirming definition of this and use it instead

namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace std;

    // string comparison class, so we do case insensitive compares
    struct nocase_compare
    {
        // std::string version of 'less' function
        // return false for equivalent, true for different
        bool operator()(const std::string& left, const std::string& right) { return _stricmp(left.c_str(), right.c_str()) < 0; }
        // std::wstring version of 'less' function, used in non-config classes
        bool operator()(const std::wstring& left, const std::wstring& right) { return _wcsicmp(left.c_str(), right.c_str()) < 0; }
    };

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
    static inline void RuntimeError(const string & message) { RuntimeError("%s", message.c_str()); }

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
    static inline void LogicError(const string & message) { LogicError("%s", message.c_str()); }

    // InvalidArgument - throw a std::logic_error with a formatted error string
#ifdef _MSC_VER
    __declspec(noreturn)
#endif
        static inline void InvalidArgument(const char * format, ...)
    {
            va_list args;
            char buffer[1024];

            va_start(args, format);
            vsprintf(buffer, format, args);
            throw std::invalid_argument(buffer);
        };
    static inline void InvalidArgument(const string & message) { InvalidArgument("%s", message.c_str());
    // Warning - warn with a formatted error string
    static inline void Warning(const char * format, ...)
    {
        va_list args;
        char buffer[1024];

        va_start(args, format);
        vsprintf(buffer, format, args);
    };
    static inline void Warning(const string & message) { Warning("%s", message.c_str()); }

    // ----------------------------------------------------------------------------
    // dynamic loading of modules  --TODO: not Basics, should move to its own header
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
                Microsoft::MSR::CNTK::RuntimeError("Plugin not found: %s", msra::strfun::utf8(m_dllName).c_str());

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

}
}
}

// ===========================================================================
// emulation of some MSVC proprietary CRT
// ===========================================================================

#ifndef _MSC_VER
static inline int _wsystem(const wchar_t *command) { return system(msra::strfun::utf8(command).c_str()); }
static inline FILE * _wpopen(const wchar_t * command, const wchar_t *mode) { return popen(msra::strfun::utf8(command).c_str(), msra::strfun::utf8(std::wstring(mode)).c_str()); }
static inline int _pclose(FILE *stream) { return pclose(stream); }
#endif

#endif // _BASICS_H_
