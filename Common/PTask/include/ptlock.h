///-------------------------------------------------------------------------------------------------
// file:	ptlock.h
//
// summary:	Declares the ptlock class
///-------------------------------------------------------------------------------------------------

#ifndef __PTLOCK_H__
#define __PTLOCK_H__

#include <stdio.h>
#include <crtdbg.h>
#include "Lockable.h"
#include <assert.h>

namespace PTask {

    class PTLock : public Lockable {
    public:
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="lpszProtectedObjectName">  [in] If non-null, name of the protected object. </param>
        ///-------------------------------------------------------------------------------------------------

        PTLock(char * lpszProtectedObjectName) : 
            Lockable(lpszProtectedObjectName),
            m_nReaders(0), 
            m_nWriters(0) { }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reader lock. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <returns>   The lock. </returns>
        ///-------------------------------------------------------------------------------------------------

        int LockRO() {
            int nDepth = Lock();
            if(nDepth > 1) {
                assert(m_nReaders > 0);
                assert(m_nWriters == 0);
                Unlock();
                return nDepth;
            }
            while(m_nWriters > 0) {
                Unlock();
                Sleep(1);
                Lock();
            }
            assert(m_nWriters == 0);
            m_nReaders++;
            Unlock();
            return m_nReaders;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the ro. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        int UnlockRO() {
            int nDepth = Lock();
            assert(m_nReaders > 0);
            assert(m_nWriters == 0);
            if(nDepth == 1 && m_nReaders) {
                m_nReaders--;
            }
            Unlock();
            return m_nReaders;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Writer lock. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        int LockRW() {
            int nDepth = Lock();
            if(nDepth > 1) {
                assert(m_nReaders == 0);
                assert(m_nWriters == 1);
                return nDepth;
            }
            while(m_nReaders > 0) {
                Unlock();
                Sleep(1);
                Lock();
            }
            assert(m_nReaders == 0);
            assert(m_nWriters == 0);
            m_nWriters++;
            return m_nWriters;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   release a write lock. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        int UnlockRW() {
            assert(m_nWriters == 1);
            assert(m_nReaders == 0);
            if(GetLockDepth() > 1)
                return Unlock();
            m_nWriters--;
            return Unlock();
        }

    protected:

        /// <summary>   The readers. </summary>
        int m_nReaders;

        /// <summary>   The writers. </summary>
        int m_nWriters;

    };
};

#endif