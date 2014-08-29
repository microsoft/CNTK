///-------------------------------------------------------------------------------------------------
// file:	Lockable.h
//
// summary:	Declares the lockable object class
///-------------------------------------------------------------------------------------------------

#ifndef __LOCKABLE_OBJECT_H__
#define __LOCKABLE_OBJECT_H__

#include <Windows.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <crtdbg.h>
#include "primitive_types.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Lockable object. Super-class for all PTask runtime objects that implement coarse
    ///             object-level locking with CRITICAL_SECTION objects. Since CRITICAL_SECTIONs are 
    ///             re-entrant, so are Lockables.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------
    class Lockable {
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="lpszProtectedObjectName">  [in] If non-null, name of the protected object. </param>
        ///-------------------------------------------------------------------------------------------------

        Lockable(char * lpszProtectedObjectName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Lockable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Lock this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <returns>   the new lock depth. </returns>
        ///-------------------------------------------------------------------------------------------------

        int Lock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlock this object. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <returns>   the new lock depth. </returns>
        ///-------------------------------------------------------------------------------------------------

        int Unlock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is locked. This method is to be used in asserts that the
        ///             current thread holds the lock, and *not* to be used to implement TryLock
        ///             semantics!
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <returns>   true if held, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL LockIsHeld();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the lock depth. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   The lock depth. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetLockDepth();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   In debug mode, enables/disables tracking for a particular object, returns
        ///             true if tracking is enabled after the call. When tracking is enabled,
        ///             all lock/unlock calls are logged to the console. A handy tool for teasing
        ///             apart deadlocks.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 8/29/2013. </remarks>
        ///
        /// <param name="bEnable">  (Optional) the enable. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL TrackLockActivity(BOOL bEnable=TRUE);

    private:

        /// <summary> The lock </summary>
        CRITICAL_SECTION        m_lock;
    
        /// <summary> Depth of the lock </summary>
        int                     m_nLockDepth;

        /// <summary> Name of the protected object </summary>
        char *                  m_lpszProtectedObjectName;

        /// <summary> Handle of the owning thread, if we are in debug mode. </summary>
        DWORD                   m_dwOwningThreadId;

        /// <summary>   true if we should log lock/unlock activity for this object. </summary>
        BOOL                    m_bTrack;

        /// <summary>   The unnested acquires. </summary>
        UINT                    m_uiUnnestedAcquires;

        /// <summary>   The unnested releases. </summary>
        UINT                    m_uiUnnestedReleases;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the owning thread identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="bLocking"> true if this update is for the lock operation, otherwise this update
        ///                         is for an unlock. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    UpdateOwningThreadId(BOOL bLocking);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Logs lock activity. </summary>
        ///
        /// <remarks>   crossbac, 8/29/2013. </remarks>
        ///
        /// <param name="bLocking"> true to locking. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    LogLockActivity(BOOL bLocking);
    };
};

#endif  // __LOCKABLE_OBJECT_H__