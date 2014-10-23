//
// message.h - class for simple I/O of log messages
//
//     Copyright (c) Microsoft Corporation.  All rights reserved.
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/message.h $
// 
// 66    7/05/11 8:17 Fseide
// error() now prints the Win32 error code as well, maybe we can now track
// down the unreliable-server problem
// 
// 65    11/30/09 1:33p Kit
// updated to compile under winCE
// 
// 64    6/07/09 0:00 Fseide
// (added a comment)
// 
// 63    5/18/09 15:28 Fseide
// minor bug fix in an error message in __flush()
// 
// 62    1/08/09 16:14 Fseide
// moved timeDateStamp() here, as it is often used in logging
// 
// 61    1/08/09 9:23 Fseide
// moved _CHECKED_ASSERT_error() to message.h, finally getting rid of
// message.cpp
// 
// 60    12/09/08 6:59p Qiluo
// reverted stringerror => strerror
// 
// 59    12/09/08 6:29p Qiluo
// strerror => stringerror
// 
// 58    11/14/08 7:43p Qiluo
// mark banned APIs
// 
// 57    11/11/08 18:19 Fseide
// no longer disables 4996
// 
// 56    11/11/08 17:56 Fseide
// (a comment added)
// 
// 55    11/11/08 17:55 Fseide
// replaced strbXXX() calls with safe fixed-buffer overloads
// 
// 54    11/11/08 17:47 Fseide
// fixed use of Xprintf() functions to use fixed-size overloads
// 
// 53    11/11/08 15:08 Fseide
// replaced safe(r) _vsnprintf() by unsafer vsprintf() assuming there is
// an overload to make it safe again...
// 
// 52    11/11/08 14:52 Fseide
// (added a comment)
// 
// 51    6/18/08 11:41 Fseide
// added #pragma once
// 
// 50    29/05/08 4:58p Kit
// G.muted semantics changed - now only controls whether logging goes to
// stderr or not
// 
// 49    29/05/08 3:19p Kit
// changed semantics of G.muted - now G.muted only controls logging to
// stderr - so if G.muted is one but log file is specified, it will still
// log to the file
// 
// 48    10/01/07 13:19 Fseide
// added setHeavyLogging() and noFlush flag for cases where a lot of info
// is written to the log file
// 
// 47    27/06/07 5:11p Kit
// rolled back to version 45
// 
// 45    27/06/07 4:58p Kit
// changed a few more methods to inline to avoid linker errors
// 
// 44    27/06/07 4:54p Kit
// changed a few methods to inline to avoid multiple include linker errors
// 
// 43    5/08/07 16:29 Fseide
// increased output buffer size to 30K
// 
// 42    4/11/07 17:24 Fseide
// fixed a bug for string overflow for vsnprintf(), 0-terminator missing
// 
// 41    07-04-11 14:57 Qfyin
// added a std::
// 
// 40    3/28/07 11:57 Fseide
// fixed the C4702 problem with error() and mem_error()
// 
// 39    3/27/07 20:54 Fseide
// silly compiler warning again (inconsistent warning between Release and
// Debug about unreachable code)
// 
// 38    3/27/07 20:49 Fseide
// fixed a compiler warning
// 
// 37    3/27/07 17:58 Fseide
// added namespace qualifiers
// 
// 36    3/27/07 15:59 Fseide
// changed struct back to namespace, uniqueness problem of static g solved
// by moving _glob() into __globals
// 
// 35    3/27/07 15:23 Fseide
// added private/public qualifiers back in
// 
// 34    3/27/07 15:19 Fseide
// changed namespace into struct (namespace has problems with the shared
// state)
// 
// 33    3/27/07 15:14 Fseide
// fixed compiler warnings
// 
// 32    3/27/07 13:53 Fseide
// removed 'static' markers as they led to warnings
// 
// 31    3/27/07 13:49 Fseide
// changed from class HDM_CLog to namespace msra::logging;
// now does not require message.cpp anymore
// 
// 30    2/14/07 15:38 Fseide
// (fixed compiler warnings when compiling managed)
// 
// 29    11/22/06 6:39p Rogeryu
// new function getLogFile
// 
// 28    5/30/06 6:42p Rogeryu
// refine the log handle reconnection
// 
// 27    5/24/06 2:51p Rogeryu
// 
// 26    3/24/06 4:40p Rogeryu
// workaround a VC 2003 header bug (va_start macro for references) in
// MESSAGE/ERROR functions
// 
// 25    3/24/06 13:33 Fseide
// cleaned up C4996 (back to file level to keep code tidy)
// 
// 24    3/22/06 5:44p Rogeryu
// change to strbxxx macros
// 
// 23    3/22/06 4:57p Rogeryu
// refine comments
// 
// 22    3/21/06 5:21p Rogeryu
// review and fix level2_security OACR warnings
// 
// 21    06-03-14 14:32 Yushli
// 
// 20    06-03-14 11:58 Yushli
// Suppress C4996 Warning on strerror per function
// 
// 19    06-03-14 11:44 Yushli
// Suppress C4996 Warning on strcpy per function
// 
// 18    2/24/06 8:03p Kjchen
// depress oacr warnings
// 
// 17    9/25/05 12:04p Kjchen
// merge OneNote's change
// 
// 16    9/21/05 11:26 Fseide
// output changed from << to fwrite
// 
// 15    5/09/05 11:09p Kjchen
// add: using namespace std;
// 
// 14    2/17/05 10:32 Fseide
// added muted mode and new method shutUp()
// 
// 13    2/03/05 19:37 Fseide
// removed unnecessary dependence on fileutil.h;
// removed dependence on critsec.h (CCritSec now in basetypes.h)
// 
// 12    4/19/04 18:58 Fseide
// showbuf() now does not anymore use "-orDie" functions to avoid infinite
// recursion...
// 
// 11    2/21/04 10:26 Fseide
// (compiler warnings eliminated)
// 
// 10    7/31/03 12:37p Fseide
// ERROR() can now throw an exception instead of exit()-ing;
// new method HDM_CLog::setExceptionFlag();
// new class message_exception
// 
// 9     8/16/02 7:14p Fseide
// now thread-safe
// 
// 8     8/01/02 7:48p Fseide
// new function (macro) memError (MEMERROR) to display an error in case of
// out-of-memory (ERROR allocates memory itself)
// 
// 7     7/31/02 10:13a Fseide
// implemented logging to file (accessed through SETLOGFILE() macro)
// 
// 6     4/03/02 3:58p Fseide
// VSS keyword and copyright added
//
// F. Seide 5 Mar 2002
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

namespace msra { namespace logging
{
    // -----------------------------------------------------------------------
    // message_exception - exception thrown by this error module
    // -----------------------------------------------------------------------

    class message_exception : public std::runtime_error
    {
        char buf[1024];         // local buffer for message
        const char * dup_what (const char * what)
        {
            strcpy (buf, "message_exception:");
            strcat (buf, what);
            return &buf[0];
        }
    public:
        message_exception (const char * what) : runtime_error (dup_what (what))
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
        char buf[30000];	    // for _vsnprintf()

        __globals() : logFile (NULL), throwExceptionFlag (true), muted (false), noFlush (false)
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
        __time64_t localtime; _time64 (&localtime);         // get current time and date
        struct tm now = *_localtime64 (&localtime);         // convert
        char buf[20];
        sprintf (buf, "%04d/%02d/%02d %02d:%02d:%02d",
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
        ASSERT (strlen (G.buf) < sizeof (G.buf) / sizeof (*G.buf));
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
        vsprintf (G.buf, fmt, arg_ptr);
        __showbuf ("", true);
    }

    static void message_nolf (const char * fmt, ...)
    {
        if (noLoggingReqd()) return;  // muted: all output is suppressed

        msra::basetypes::CAutoLock autoLock (G.lock);
        va_list arg_ptr;
        va_start (arg_ptr, fmt);
        vsprintf (G.buf, fmt, arg_ptr);
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
        vsprintf (G.buf, fmt, arg_ptr);
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
#if 0   // special test code to determine the Windows error in case of a network error
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
        vsprintf (G.buf, fmt, arg_ptr);
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
        sprintf (G.buf, fmt, arg);
        strcat (G.buf, "\n");
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

#undef ERROR	// defined in wingdi.h... aargh!
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

#endif	// _MESSAGE_

