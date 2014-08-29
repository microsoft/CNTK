//--------------------------------------------------------------------------------------
// File: ptgc.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PTGC_H_
#define _PTGC_H_
#include <deque>
#include "Lockable.h"

namespace PTask {
    
    class Datablock; 

    static const UINT DEFAULT_DATABLOCK_GC_THREADS = 1;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Datablock garbage collector. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class GarbageCollector : public Lockable
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="nGCThreads">   (optional) the gc threads. </param>
        ///-------------------------------------------------------------------------------------------------

        GarbageCollector(UINT nGCThreads=DEFAULT_DATABLOCK_GC_THREADS);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~GarbageCollector();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force GC. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void ForceGC();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force a GC sweep that is targeted at a particular memory space. Can be called under
        ///             low-mem conditions by a failing attempt to allocate device memory. Forcing a 
        ///             full GC sweep from that calling context is impractical because a full sweep
        ///             requires locks we cannot acquire without breaking the lock-ordering discipline. 
        ///             However a device-specific allocation context can be assumed to hold a lock on the
        ///             accelerator for which we are allocating, making it safe to sweep the GC queue 
        ///             and free device buffers for that memspace *only* without deleting the parent blocks.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void ForceGC(UINT uiMemSpaceId);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queue a datablock for garbage collection. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        static void QueueForGC(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroys the GC. </summary>
        ///
        /// <remarks>   Crossbac, 3/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void DestroyGC();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates the GC. </summary>
        ///
        /// <remarks>   Crossbac, 3/18/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void CreateGC();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports the current state of the queue to the console in some detail. 
        ///             If we are getting tight on memory, this can be a handy tool for checking
        ///             whether more aggressive GC would help the workload. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/7/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Report();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Shuts down this object and frees any resources it is using. </summary>
        ///
        /// <remarks>   Crossbac, 3/1/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Shutdown();

#ifdef DEBUG

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies an allocation. </summary>
        ///
        /// <remarks>   Crossbac, 7/1/2013. </remarks>
        ///
        /// <param name="pNewBlock">    [in,out] If non-null, the new block. </param>
        ///-------------------------------------------------------------------------------------------------

        static void NotifyAllocation(Datablock * pNewBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Notifies an allocation. </summary>
        ///
        /// <remarks>   Crossbac, 7/1/2013. </remarks>
        ///
        /// <param name="pNewBlock">    [in,out] If non-null, the new block. </param>
        ///-------------------------------------------------------------------------------------------------

        void __NotifyAllocation(Datablock * pNewBlock);
#endif

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force GC. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void _ForceGC();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force a GC sweep that is targeted at a particular memory space. Can be called under
        ///             low-mem conditions by a failing attempt to allocate device memory. Forcing a 
        ///             full GC sweep from that calling context is impractical because a full sweep
        ///             requires locks we cannot acquire without breaking the lock-ordering discipline. 
        ///             However a device-specific allocation context can be assumed to hold a lock on the
        ///             accelerator for which we are allocating, making it safe to sweep the GC queue 
        ///             and free device buffers for that memspace *only* without deleting the parent blocks.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void _ForceGC(UINT uiMemSpaceId);


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queue a datablock for garbage collection. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void _QueueForGC(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports the current state of the queue to the console in some detail. 
        ///             If we are getting tight on memory, this can be a handy tool for checking
        ///             whether more aggressive GC would help the workload. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/7/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void _Report();


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   GC thread. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="p">    The p. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static DWORD WINAPI PTaskGCThread(LPVOID p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   The garbage collector thread proc. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        DWORD GarbageCollectorThread();
        
        /// <summary> The queue of blocks to delete </summary>
        std::deque<Datablock*>  m_vQ;

        /// <summary> Handle of the work available event. Set when the queue is non-empty. </summary>
        HANDLE                  m_hWorkAvailable;

        /// <summary>   Handle of the quiescent event--set when a sweep is not in progress. </summary>
        HANDLE                  m_hQuiescent;
        
        /// <summary> Handle of the gc threads </summary>
        HANDLE *                m_vGCThreads;

        /// <summary>   The number of gc threads. </summary>
        UINT                    m_nGCThreads;

        /// <summary> Handle of the gc global shutdown event. </summary>
        HANDLE                  m_hGCShutdown;

        /// <summary> Handle of the global shutdown event. </summary>
        HANDLE                  m_hRuntimeTerminateEvent;
        
        /// <summary> true if the GC thread is alive </summary>
        BOOL                    m_bAlive;

        /// <summary>   true to shutdown in progress. </summary>
        BOOL                    m_bShutdownInProgress;

        /// <summary>   true to shutdown complete. </summary>
        BOOL                    m_bShutdownComplete;

        /// <summary>   true to quiescent. </summary>
        BOOL                    m_bQuiescent;

#ifdef DEBUG
        /// <summary>   Debug mode--keep a list of 
        ///             things that have already been queued or
        ///             deleted to ensure we don't double free. </summary>
        std::set<Datablock*>    m_vQueued;
        std::set<Datablock*>    m_vDeleted;
        CRITICAL_SECTION        m_csGCTracker;
        #define ptgc_init()                 InitializeCriticalSection(&m_csGCTracker);
        #define ptgc_deinit()               DeleteCriticalSection(&m_csGCTracker);
        #define ptgc_lock()                 EnterCriticalSection(&m_csGCTracker);
        #define ptgc_unlock()               LeaveCriticalSection(&m_csGCTracker);
        #define ptgc_check_double_q(x)      assert(m_vQueued.find(x)==m_vQueued.end())
        #define ptgc_check_double_free(x)   assert(m_vDeleted.find(x)==m_vDeleted.end())
        #define ptgc_record_q(x)            m_vQueued.insert(x)
        #define ptgc_record_free(x)         { m_vDeleted.insert(x); m_vQueued.erase(x); }
        #define ptgc_reset()                { ptgc_lock(); m_vQueued.clear(); m_vDeleted.clear(); ptgc_unlock(); }
        #define ptgc_new(x)                 { GarbageCollector::NotifyAllocation(x); }
#else
        #define ptgc_init()                 
        #define ptgc_deinit()               
        #define ptgc_lock()                 
        #define ptgc_unlock()               
        #define ptgc_check_double_q(x)      
        #define ptgc_check_double_free(x)   
        #define ptgc_record_q(x)            
        #define ptgc_record_free(x)         
        #define ptgc_reset()
        #define ptgc_new(x)
#endif

        static CRITICAL_SECTION m_csGlobalGCPtr;

        static GarbageCollector * g_pGarbageCollector;
    };
       
};
#endif
