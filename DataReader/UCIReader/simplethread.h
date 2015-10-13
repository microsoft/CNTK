//
// <copyright file="simplethread.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// simplethread.h -- a simple thread implementation
//
// F. Seide, Oct 2012
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/simplethread.h $
// 
// 2     10/11/12 3:43p Fseide
// got the thread startup from lambda correct now
// 
// 1     10/10/12 3:29p Fseide
// new module simplethread, not tested yet;
// first stpes in readaheadminibatchsource towards read-ahead thread, not
// complete

#pragma once

#include "Basics.h"
#include <process.h>        // for _beginthread()
#include <memory>

namespace msra { namespace util {

// ---------------------------------------------------------------------------
// signallingevent  -- wrapper around Windows events
// ---------------------------------------------------------------------------
class signallingevent   // TODO: should this go into basetypes.h?
{
    HANDLE h;
public:
    signallingevent (bool initialstate = true)
    {
        h = ::CreateEvent (NULL, FALSE/*manual reset*/, initialstate ? TRUE : FALSE, NULL);
        if (h == NULL)
            RuntimeError("signallingevent: CreateEvent() failed");
    }
    ~signallingevent() { ::CloseHandle (h); }
    void wait() { if (::WaitForSingleObject (h, INFINITE) != WAIT_OBJECT_0) RuntimeError("wait: WaitForSingleObject() unexpectedly failed"); }
    void flag() { if (::SetEvent (h) == 0) RuntimeError("flag: SetEvent() unexpectedly failed"); }
};


// ---------------------------------------------------------------------------
// simplethread  -- simple thread wrapper
// ---------------------------------------------------------------------------
class simplethread : CCritSec
{
    unique_ptr<std::exception> exceptionptr;
    // wrapper around passing the functor
    signallingevent startsignal;
    const void * functorptr;
    template<typename FUNCTION> static unsigned int __stdcall staticthreadproc (void * usv)
    {
        simplethread * us = (simplethread*) usv;
        const FUNCTION body = *(const FUNCTION *) us->functorptr;
        us->startsignal.flag();
        us->threadproc (body);
        return 0;
    }
    template<typename FUNCTION> void threadproc (const FUNCTION & body)
    {
        unique_ptr<std::exception> exceptionstorage = std::move (exceptionptr); // was preallocated at start
        try
        {
            body();                 // execute the function
        }
        catch (const std::exception & e)
        {
            CAutoLock lock (*this);
            exceptionptr = std::move (exceptionstorage);
            *exceptionptr = e;      // exception: remember it  --this will remove the type info :(
        }
    }
    HANDLE threadhandle;
public:
    template<typename FUNCTION> simplethread (const FUNCTION & body) : exceptionptr (new std::exception), functorptr (&body), startsignal (false)
    {
        unsigned int threadid;
        uintptr_t rc = _beginthreadex (NULL/*security*/, 0/*stack*/, staticthreadproc<FUNCTION>, this, CREATE_SUSPENDED, &threadid);
        if (rc == 0)
            RuntimeError("simplethread: _beginthreadex() failed");
        threadhandle = OpenThread (THREAD_ALL_ACCESS, FALSE, threadid);
        if (threadhandle == NULL)
            LogicError("simplethread: _beginthreadex()  unexpectedly did not return valid thread id");   // BUGBUG: leaking something
        DWORD rc1 = ::ResumeThread (threadhandle);
        if (rc1 == (DWORD) -1)
        {
            ::TerminateThread (threadhandle, 0);
            ::CloseHandle (threadhandle);
            LogicError("simplethread: ResumeThread() failed unexpectedly");
        }
        try
        {
            startsignal.wait(); // wait until functor has been copied
        }
        catch (...)
        {
            ::TerminateThread (threadhandle, 0);
            ::CloseHandle (threadhandle);
            throw;
        }
    }
    void check()
    {
        CAutoLock lock (*this);
        if (exceptionptr)
            throw *exceptionptr.get();
    }
    bool wait (DWORD dwMilliseconds = INFINITE)
    {
        DWORD rc = ::WaitForSingleObject (threadhandle, dwMilliseconds);
        if (rc == WAIT_TIMEOUT)
            return false;
        else if (rc == WAIT_OBJECT_0)
            return true;
        else
            RuntimeError("wait: WaitForSingleObject() failed unexpectedly");
    }
    //void join()
    //{
    //    check();
    //    wait();
    //    check();
    //}
    ~simplethread() throw()
    {
        // wait until it shuts down
        try { wait(); }
        catch (...) { ::TerminateThread (threadhandle, 0); }
        // close the handle
        ::CloseHandle (threadhandle);
    }
};

};};
