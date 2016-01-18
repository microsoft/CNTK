//
// <copyright file="simplethread.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// simplethread.h -- a simple thread implementation
//

#pragma once

#include "basetypes.h"
#ifdef _WIN32
#include <process.h> // for _beginthread()
#endif

namespace msra { namespace util {

// ---------------------------------------------------------------------------
// signallingevent  -- wrapper around Windows events
// ---------------------------------------------------------------------------
class signallingevent // TODO: should this go into basetypes.h?
{
    HANDLE h;

public:
    signallingevent(bool initialstate = true)
    {
        h = ::CreateEvent(NULL, FALSE /*manual reset*/, initialstate ? TRUE : FALSE, NULL);
        if (h == NULL)
            throw std::runtime_error("signallingevent: CreateEvent() failed");
    }
    ~signallingevent()
    {
        ::CloseHandle(h);
    }
    void wait()
    {
        if (::WaitForSingleObject(h, INFINITE) != WAIT_OBJECT_0)
            throw std::runtime_error("wait: WaitForSingleObject() unexpectedly failed");
    }
    void flag()
    {
        if (::SetEvent(h) == 0)
            throw std::runtime_error("flag: SetEvent() unexpectedly failed");
    }
};

// ---------------------------------------------------------------------------
// simplethread  -- simple thread wrapper
// ---------------------------------------------------------------------------
class simplethread : CCritSec
{
    std::shared_ptr<std::exception> badallocexceptionptr; // in case we fail to copy the exception
    std::shared_ptr<std::exception> exceptionptr;         // if non-NULL, then thread failed with exception
    // wrapper around passing the functor
    signallingevent startsignal;
    const void *functorptr;
    template <typename FUNCTION>
    static unsigned int __stdcall staticthreadproc(void *usv)
    {
        simplethread *us = (simplethread *) usv;
        const FUNCTION body = *(const FUNCTION *) us->functorptr;
        us->startsignal.flag();
        us->threadproc(body);
        return 0;
    }
    template <typename FUNCTION>
    void threadproc(const FUNCTION &body)
    {
        try
        {
            body(); // execute the function
        }
        catch (const std::exception &e)
        {
            fail(e);
        }
        catch (...) // we do not catch anything that is not based on std::exception
        {
            fprintf(stderr, "simplethread: thread proc failed with unexpected unknown exception, which is not allowed. Terminating\n");
            fflush(stderr); // (needed?)
            abort();        // should never happen
        }
    }
    HANDLE threadhandle;

public:
    template <typename FUNCTION>
    simplethread(const FUNCTION &body)
        : badallocexceptionptr(new std::bad_alloc()), functorptr(&body), startsignal(false)
    {
        unsigned int threadid;
        uintptr_t rc = _beginthreadex(NULL /*security*/, 0 /*stack*/, staticthreadproc<FUNCTION>, this, CREATE_SUSPENDED, &threadid);
        if (rc == 0)
            throw std::runtime_error("simplethread: _beginthreadex() failed");
        threadhandle = OpenThread(THREAD_ALL_ACCESS, FALSE, threadid);
        if (threadhandle == NULL)
            throw std::logic_error("simplethread: _beginthreadex()  unexpectedly did not return valid thread id"); // BUGBUG: leaking something
        DWORD rc1 = ::ResumeThread(threadhandle);
        if (rc1 == (DWORD) -1)
        {
            ::TerminateThread(threadhandle, 0);
            ::CloseHandle(threadhandle);
            throw std::logic_error("simplethread: ResumeThread() failed unexpectedly");
        }
        try
        {
            startsignal.wait(); // wait until functor has been copied
        }
        catch (...)
        {
            ::TerminateThread(threadhandle, 0);
            ::CloseHandle(threadhandle);
            throw;
        }
    }
    // check if the thread is still alive and without error
    void check()
    {
        CAutoLock lock(*this);
        // pass on a pending exception
        if (exceptionptr)
            throw * exceptionptr.get();
        // the thread going away without error is also unexpected at this point
        if (wait(0)) // (0 means don't block, so OK to call inside lock)
            throw std::runtime_error("check: thread terminated unexpectedly");
    }
    bool wait(DWORD dwMilliseconds = INFINITE)
    {
        DWORD rc = ::WaitForSingleObject(threadhandle, dwMilliseconds);
        if (rc == WAIT_TIMEOUT)
            return false;
        else if (rc == WAIT_OBJECT_0)
            return true;
        else
            throw std::runtime_error("wait: WaitForSingleObject() failed unexpectedly");
    }
    // thread itself can set the failure condition, e.g. before it signals some other thread to pick it up
    void fail(const std::exception &e)
    {
        // exception: remember it  --this will remove the type info :(
        CAutoLock lock(*this);
        try // copy the exception--this may fail if we are out of memory
        {
            exceptionptr.reset(new std::runtime_error(e.what()));
        }
        catch (...) // failed to alloc: fall back to bad_alloc, which is most likely the cause in such situation
        {
            exceptionptr = badallocexceptionptr;
        }
    }
    //void join()
    //{
    //    check();
    //    wait();
    //    check_for_exception();    // (check() not sufficient because it would fail since thread is gone)
    //}
    ~simplethread() throw()
    {
        // wait until it shuts down
        try
        {
            wait();
        }
        catch (...)
        {
            ::TerminateThread(threadhandle, 0);
        }
        // close the handle
        ::CloseHandle(threadhandle);
    }
};
};
};
