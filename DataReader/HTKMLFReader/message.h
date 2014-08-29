//
// message.h - class for simple I/O of log messages
//
//     Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once
#ifndef _MESSAGE_
#define _MESSAGE_

#ifndef UNDER_CE    // fixed-buffer overloads not available for wince
#ifdef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES  // fixed-buffer overloads for strcpy() etc.
#undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#endif
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif

#include "basetypes.h"

#include <stdarg.h>
#include <iostream>
#include <exception>
#include <time.h>                   // for _time64 in timeDateStamp()

#pragma warning (push)
#pragma warning (disable : 4793)    // caused by varargs
#pragma warning(disable : 4996) // strcpy and other safety stuff disabled

namespace msra { namespace logging
{
    // -----------------------------------------------------------------------
    // message_exception - exception thrown by this error module
    // -----------------------------------------------------------------------

    class message_exception : public std::exception
    {
        char buf[1024];         // local buffer for message
        const char * dup_what (const char * what)
        {
            strcpy (buf, "message_exception:"); // security hint: safe overloads
            strcat (buf, what);
            return &buf[0];
        }
    public:
        message_exception (const char * what) : exception (dup_what (what))
        {
        }
    };

    // -----------------------------------------------------------------------
    // global state (hidden as a static struct in a variable)
    // -----------------------------------------------------------------------

    struct __globals
    {
        msra::basetypes::CCritSec lock;
        FILE * logFile;
        bool noFlush;               // heavy logging: don't flush
        bool throwExceptionFlag;    // true: don't exit but throw exception
        
       // G.muted semantics is as follows
        // - G.muted && !G.logFile => don't show anything
        // - G.muted && G.logFile => write to log file
        // - !G.muted && !G.logFile => write to stderr only
        // - !G.muted && G.logFile => write to log file and stderr
        bool muted;                 

        std::string filename;       // log file name
        char buf[30000];        // for _vsnprintf()

        __globals() : logFile (NULL), throwExceptionFlag (false), muted (false), noFlush (false)
        { buf[0] = 0; buf[sizeof (buf) / sizeof (*buf) -1] = 0; }

        static __globals & get() { static __globals g; return g; }
    };

#pragma push_macro ("G")
#define G (__globals::get())    // access global XYZ as G.XYZ

    // ------------------------------------------------------------------------
    // setLogFile(): set the log file
    // if non-NULL then every message will be written both to stderr and this
    // log file.
    // multi-threading: not thread-safe, set this before starting
    // ------------------------------------------------------------------------

    static inline FILE * setLogFile (FILE * newLogFile)
    {
        FILE * oldLogFile;

        oldLogFile = G.logFile;

        if (newLogFile != stderr)
        {
            G.logFile = newLogFile;
        }
        else
        {
            G.logFile = NULL;
        }

        return oldLogFile;
    }

    // ------------------------------------------------------------------------
    // setLogFileByName(): set the log file by file name
    // in this mode, log file will be re-connected when disconnected
    // filename == NULL indicates an attempt to reconnect
    // WARNING: if the filename is invalid, it will try to reconnect every time
    // ------------------------------------------------------------------------

    static inline void setLogFileByName (const char * p_filename)
    {
        FILE * newLogFile = NULL;
        if (p_filename == NULL)
        {
            // for reconnection
            ASSERT (G.filename != "");
            newLogFile = fopen (G.filename.c_str (), "ab");
        }
        else
        {
            ASSERT (p_filename[0]);
            G.filename = p_filename;  // remember filename
            newLogFile = fopen (p_filename, "wb");
        }

        // handle open failure
        if (newLogFile == NULL)
        {
            if (G.logFile != NULL)
            {
                fprintf (G.logFile, "ERROR: setLogFileByName: error opening log file %s: %s\n",
                    G.filename.c_str (), strerror (errno));
                // in case of a reconnect, this ^^ will obviously fail, we ignore this
            }
            fprintf (stderr, "ERROR: setLogFileByName: error opening log file %s: %s\n",
                G.filename.c_str (), strerror (errno));
            return;
        }

        // set new handle
        FILE * oldLogFile = setLogFile (newLogFile);

        // close old handle
        if (oldLogFile != NULL && oldLogFile != stderr && oldLogFile != stdin)
        {
            int rc = fclose (oldLogFile);
            if (rc != 0)
            {
                if (G.logFile != NULL)
                {   // note: this goes to the new log file
                    fprintf (G.logFile, "ERROR: setLogFileByName: error closing old log file: %s\n",
                        strerror (errno));
                }
                fprintf (stderr, "ERROR: setLogFileByName: error closing old log file: %s\n",
                    strerror (errno));
            }
        }
    }

    // ------------------------------------------------------------------------
    // setExceptionFlag(): set flag whether to throw an exception (true) or exit() (false, default)
    // ------------------------------------------------------------------------

    static inline bool setExceptionFlag (bool throwExceptionFlag = true)
    {
        bool oldFlag = G.throwExceptionFlag;
        G.throwExceptionFlag = throwExceptionFlag;
        return oldFlag;
    }

    // ------------------------------------------------------------------------
    // timeDateStamp() -- often needed for logging
    // ------------------------------------------------------------------------

    static inline std::string timeDateStamp (void)
    {
        __time64_t localtime; _time64 (&localtime);       // get current time and date
        struct tm now; _localtime64_s (&now, &localtime); // convert
        char buf[20];
        sprintf (buf, "%04d/%02d/%02d %02d:%02d:%02d",    // security hint: this is an overload
                 now.tm_year + 1900, now.tm_mon + 1, now.tm_mday,
                 now.tm_hour, now.tm_min, now.tm_sec);
        return buf;
    }

    // ------------------------------------------------------------------------
    // __flush(): flush output
    // ------------------------------------------------------------------------

    static inline void __flush()
    {
        int rc = fflush (G.logFile);
        if (rc != 0)
        {
            fprintf (stderr, "ERROR: __flush: error flushing to log file %s\n",
                     strerror (errno));
        }
    }

    // ------------------------------------------------------------------------
    // setHeavyLogging(): we are heavily logging: don't flush & increase out buf
    // ------------------------------------------------------------------------

    static inline void setHeavyLogging (bool isHeavy)
    {
        __flush();  // flush the current buffer
        if (!isHeavy)
        {
            G.noFlush = false;
        }
        else
        {
            G.noFlush = true;
            if (G.logFile)
                setvbuf (G.logFile, NULL, _IOFBF, 16384);   // flush every 16K
        }
    }

    // ------------------------------------------------------------------------
    // shutUp(): set muted mode (true: no output will be generated anymore)
    //
    // multi-threading: retrieving the previous state is not thread-safe,
    // if you want, do this before starting
    // ------------------------------------------------------------------------

    static inline bool shutUp (bool quiet = true)
    {
        bool oldFlag = G.muted;
        G.muted = quiet;
        return oldFlag;
    }

    // ------------------------------------------------------------------------
    // getLogFile(): get log file handle
    // ------------------------------------------------------------------------

    static inline FILE * getLogFile (void)
    {
        return G.logFile;
    }

    // ------------------------------------------------------------------------
    // __showbuf(): output contents of buf[] with prefix prepended
    // multi-threading: must be called from within critical section
    // ------------------------------------------------------------------------

    static inline void __showbuf (const std::string & prefix, bool nl)
    {
        ASSERT (strlen (G.buf) < sizeof (G.buf) / sizeof (*G.buf)); // security hint: safe overloads
        std::string outtext = prefix + G.buf;
        if (nl) outtext += "\n";

        // write out; first to screen in case we can't write to log file

#ifndef ONENOTE_COMPILER
        // OneNote treats it as an error if stderr is not empty.
        // and in OneNote, we can't see message printed to stderr
        // So, in OneNote, don't put it into stderr

        // G.muted semantics is as follows
        // - G.muted && !G.logFile => don't show anything
        // - G.muted && G.logFile => write to log file
        // - !G.muted && !G.logFile => write to stderr only
        // - !G.muted && G.logFile => write to log file and stderr
        if (!G.muted)
        {
            fwrite ((void*) outtext.c_str(), sizeof (*outtext.c_str()),
                outtext.length(), stderr);
            if (!G.noFlush)
                fflush (stderr);
        }
#endif

        // write to log file

        // check whether the log file has been disconnected or not
        if (G.filename != "")     // with known filename, suppose to reconnect
        {
            if (G.logFile == NULL || ferror (G.logFile) != 0)
            {
                setLogFileByName (NULL);    // attempt to re-open the log file

                if (G.logFile)
                {
                    fprintf (G.logFile, "ERROR: __showbuf: log file handle lost, reconnected\n");
                }
            }
        }

        if (G.logFile)
        {
            size_t n = fwrite ((void*) outtext.c_str(), sizeof (*outtext.c_str()),
                outtext.length(), G.logFile);
            if (n != outtext.length() * sizeof (*outtext.c_str()))
            {   // write error
                fprintf (stderr, "ERROR: __showbuf: error writing this to log file: %s\n", strerror (errno));
                fwrite ((void*) outtext.c_str(), sizeof (*outtext.c_str()),
                    outtext.length(), stderr);
            }
            else if (!G.noFlush)        // flush logFile
            {
                __flush();
            }
        }
    }

    // ------------------------------------------------------------------------
    // noLoggingReqd: function to determine if any logging reqd 
    // at all - used so that we can exit early if none reqd
    // ------------------------------------------------------------------------

    static inline bool noLoggingReqd()
    {
        return G.muted && !G.logFile;
    }

    // ------------------------------------------------------------------------
    // message(): like printf(), writing to log output
    // multi-threading: this function is thread-safe
    // ------------------------------------------------------------------------

    static inline void message (const char * fmt, ...)
    {
        if (noLoggingReqd()) return;  // muted: all output is suppressed

        msra::basetypes::CAutoLock autoLock (G.lock);
        va_list arg_ptr;
        va_start (arg_ptr, fmt);
        vsprintf (G.buf, fmt, arg_ptr); // security hint: this is an overload
        __showbuf ("", true);
    }

    static void message_nolf (const char * fmt, ...)
    {
        if (noLoggingReqd()) return;  // muted: all output is suppressed

        msra::basetypes::CAutoLock autoLock (G.lock);
        va_list arg_ptr;
        va_start (arg_ptr, fmt);
        vsprintf (G.buf, fmt, arg_ptr); // security hint: this is an overload
        __showbuf ("", false);
    }

    // ------------------------------------------------------------------------
    // warning(): like message(), with text "WARNING: " prepended
    // multi-threading: this function is thread-safe
    // ------------------------------------------------------------------------

    static void warning (const char * fmt, ...)
    {
        if (noLoggingReqd()) return;  // muted: all output is suppressed

        msra::basetypes::CAutoLock autoLock (G.lock);
        va_list arg_ptr;
        va_start (arg_ptr, fmt);
        vsprintf (G.buf, fmt, arg_ptr); // security hint: this is an overload
        __showbuf ("WARNING: ", true);
        __flush();
    }

    // ------------------------------------------------------------------------
    // __throw_or_exit(): exit() or throw exception depending on throwExceptionFlag
    // ------------------------------------------------------------------------

    static inline void __throw_or_exit (void)
    {
        __flush();
        if (G.throwExceptionFlag)
        {
            throw message_exception (G.buf);
        }
        exit (1);
    }

    // ------------------------------------------------------------------------
    // error(): like warning() but terminates program afterwards
    // multi-threading: this function is thread-safe
    // ------------------------------------------------------------------------

#pragma warning (push)
#pragma warning (disable : 4702)    // the 'return 0;' causes this in Release
    static int error (const char * fmt, ...)
    {
#if 1   // special test code to determine the Windows error in case of a network error
        DWORD winErr = GetLastError();
        try
        {
            msra::basetypes::CAutoLock autoLock (G.lock);
            sprintf (G.buf, "%d (\"%S\")", winErr, FormatWin32Error(winErr).c_str());
            if (!noLoggingReqd())
                __showbuf ("Win32 error of subsequent error message: ", true);
        }
        catch(...){}
#endif
        msra::basetypes::CAutoLock autoLock (G.lock);
        va_list arg_ptr;
        va_start (arg_ptr, fmt);
        vsprintf (G.buf, fmt, arg_ptr); // security hint: this is an overload
        if (!noLoggingReqd())
        {   // if muted, we format the msg (for __throw_or_exit) but don't print it
            __showbuf ("ERROR: ", true);
        }
        __throw_or_exit();
        return 0;
    }

    // ------------------------------------------------------------------------
    // mem_error(): similar to error() but without any memory allocations
    // (only one string argument allowed)
    // multi-threading: this function is thread-safe
    // ------------------------------------------------------------------------

    static int mem_error (const char * fmt, int arg)
    {
        msra::basetypes::CAutoLock autoLock (G.lock);
        if (!noLoggingReqd())
        {   // if muted, we format the msg (for __throw_or_exit) but don't print it
            fprintf (stderr, fmt, arg);
            fprintf (stderr, "\n");

            if (G.logFile)
            {
                fprintf (G.logFile, fmt, arg);
                fprintf (G.logFile, "\n");
                int rc = fflush (G.logFile);
                if (rc != 0)
                {
                    fprintf (stderr, "error flushing log message to file: %s\n",
                             strerror (errno));
                }
            }
        }

        // format msg for __throw_or_exit()
        sprintf (G.buf, fmt, arg);  // security hint: this is an overload
        strcat (G.buf, "\n");       // security hint: this is an overload
        __throw_or_exit();
        return 0;
    }
#pragma warning (pop)

    static inline void __avoid_C4505 (void)
    { message (""); message_nolf (""); warning (""); error (""); mem_error ("", 0); }
#pragma pop_macro ("G")
};};

#pragma warning(pop)

// ===========================================================================
// compatibility macros (for older source code)
// ===========================================================================

#undef ERROR    // defined in wingdi.h... aargh!
#define WARNING         msra::logging::warning
#define ERROR           msra::logging::error
#define MESSAGE         msra::logging::message
#define MESSAGE_NOLF    msra::logging::message_nolf
#define MEMERROR        msra::logging::mem_error
#define SETLOGFILE      msra::logging::setLogFile

// ===========================================================================
// special function for basetypes.h's ASSERT() macro
// ===========================================================================

#ifdef _CHECKED
void inline _CHECKED_ASSERT_error(const char * file, int line, const char * exp)
{ ERROR ("%s:%d:assertion failure: %s", file, line, exp); }
#endif

#endif    // _MESSAGE_

