///-------------------------------------------------------------------------------------------------
// file:	ThreadPool.h
//
// summary:	Declares the thread pool class
///-------------------------------------------------------------------------------------------------

#ifndef __THREAD_POOL_H__
#define __THREAD_POOL_H__

#include <stdio.h>
#include <crtdbg.h>
#include <Windows.h>
#include <deque>
#include <map>
#include <set>
#include "Lockable.h"
#include "PTaskRuntime.h"

namespace PTask {

    class ThreadPool;

    class THREADDESC {
    public:
        CRITICAL_SECTION       lock;
        HANDLE                 hThread;
        HANDLE                 hStartEvent;
        HANDLE                 hTerminateEvent;
        BOOL                   bRoutineValid;
        BOOL                   bTerminate;
        BOOL                   bActive;
        LPTHREAD_START_ROUTINE lpRoutine;
        LPVOID                 lpParameter;
        BOOL                   bDeleteOnThreadExit;
        BOOL                   bRemoveFromPoolOnThreadExit;
        ThreadPool *           pThreadPool;
        THREADDESC(ThreadPool*pPool) { 
            InitializeCriticalSection(&lock);
            hThread = INVALID_HANDLE_VALUE;
            hStartEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
            hTerminateEvent = PTask::Runtime::GetRuntimeTerminateEvent();
            bRoutineValid = FALSE;
            bTerminate = FALSE;
            bActive = FALSE;
            lpRoutine = NULL;
            lpParameter = NULL;
            bDeleteOnThreadExit = FALSE;
            bRemoveFromPoolOnThreadExit = FALSE;
            pThreadPool = pPool;
        }
        ~THREADDESC() {             
            DeleteCriticalSection(&lock); 
        }
        void Lock() { EnterCriticalSection(&lock); }
        void Unlock() { LeaveCriticalSection(&lock); }
    };

    class ThreadPool : public Lockable {

        static const int DEFGROWINC=2;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="nThreads">         If non-null, the p. </param>
        /// <param name="bPrimeThreads">    The prime threads. </param>
        /// <param name="bGrowable">        The growable. </param>
        /// <param name="uiGrowIncrement">  The grow increment. </param>
        ///-------------------------------------------------------------------------------------------------

        ThreadPool(
            __in UINT nThreads,
            __in BOOL bPrimeThreads,
            __in BOOL bGrowable,
            __in UINT uiGrowIncrement
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~ThreadPool();

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates this object. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="uiThreads">        The threads. </param>
        /// <param name="bPrimeThreads">    The threads. </param>
        /// <param name="bGrowable">        The growable. </param>
        /// <param name="uiGrowIncrement">  The grow increment. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static ThreadPool * 
        Create(
            __in UINT uiThreads,
            __in BOOL bPrimeThreads,
            __in BOOL bGrowable,
            __in UINT uiGrowIncrement=DEFGROWINC
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroys this object. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Destroy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   The pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetPoolSize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets pool size. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   The pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetCurrentPoolSize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets target pool size. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <returns>   The target pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetTargetPoolSize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets pool size. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="uiThreads">    The threads. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetPoolSize(UINT uiThreads);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets grow increment. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <returns>   The grow increment. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetGrowIncrement();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets grow increment. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="uiIncrement">  Amount to increment by. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetGrowIncrement(UINT uiIncrement);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Request thread. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="lpRoutine">    The routine. </param>
        /// <param name="lpParameter">  The parameter. </param>
        /// <param name="bStartThread"> true if the thread can be signaled to start 
        ///                             before returning from this call, false if the
        ///                             caller would prefer to signal it explicitly. </param>
        ///
        /// <returns>   The handle of the thread. </returns>
        ///-------------------------------------------------------------------------------------------------

        static HANDLE 
        RequestThread(
            __in LPTHREAD_START_ROUTINE lpRoutine, 
            __in LPVOID lpParameter,
            __in BOOL bStartThread
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Starts a thread: if a previous call to RequestThread was made with 
        ///             the bStartThread parameter set to false, this API signals the thread
        ///             to begin. Otherwise, the call has no effect (returns FALSE). </summary>
        ///
        /// <remarks>   crossbac, 8/29/2013. </remarks>
        ///
        /// <param name="hThread">  The thread. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        StartThread(
            __in HANDLE hThread
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a thread. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="lpRoutine">    The routine. </param>
        /// <param name="lpParameter">  The parameter. </param>
        ///
        /// <returns>   The thread. </returns>
        ///-------------------------------------------------------------------------------------------------

        HANDLE 
        GetThread(
            __in LPTHREAD_START_ROUTINE lpRoutine, 
            __in LPVOID lpParameter,
            __in BOOL bStartThread
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Starts a thread: if a previous call to RequestThread was made with 
        ///             the bStartThread parameter set to false, this API signals the thread
        ///             to begin. Otherwise, the call has no effect (returns FALSE). </summary>
        ///
        /// <remarks>   crossbac, 8/29/2013. </remarks>
        ///
        /// <param name="hThread">  The thread. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        SignalThread(
            __in HANDLE hThread
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Thread pool proc. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pVoidCastGraph">   the graph object, typecast to void* </param>
        ///
        /// <returns>   DWORD: 0 on thread exit. </returns>
        ///-------------------------------------------------------------------------------------------------

        static DWORD WINAPI _ThreadPoolProc(LPVOID pDesc);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Thread pool proc. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="pDesc">    The description. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        DWORD 
        ThreadPoolProc(
            __in THREADDESC * pDesc
            );

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies a thread alive. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="hThread">  Handle of the thread. </param>
        ///-------------------------------------------------------------------------------------------------

        void NotifyThreadAlive(HANDLE hThread);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies a thread exit. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///
        /// <param name="hThread">  Handle of the thread. </param>
        ///-------------------------------------------------------------------------------------------------

        void NotifyThreadExit(HANDLE hThread);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait threads alive. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void WaitThreadsAlive();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Starts the threads. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void StartThreads(UINT uiThreads, BOOL bWaitAlive);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Prime thread. </summary>
        ///
        /// <remarks>   crossbac, 8/22/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void PrimeThread();

        std::map<HANDLE, THREADDESC*> m_vhThreadDescs;
        std::deque<HANDLE>            m_vhAvailable;
        std::set<HANDLE>              m_vhInFlight;
        std::set<HANDLE>              m_vhWaitingStartSignal;
        std::set<THREADDESC*>         m_vZombieThreadDescs;
        UINT                          m_uiThreads;
        UINT                          m_uiTargetSize;
        BOOL                          m_bPrimeThreads;
        BOOL                          m_bGrowable;
        UINT                          m_uiGrowIncrement;
        UINT                          m_uiThreadsAlive;
        HANDLE                        m_hAllThreadsAlive;
        HANDLE                        m_hAllThreadsExited;
        UINT                          m_uiAliveWaiters;
        UINT                          m_uiExitWaiters;
        BOOL                          m_bExiting;

        static ThreadPool * g_pThreadPool;

    };

};
#endif