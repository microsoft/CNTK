///-------------------------------------------------------------------------------------------------
// file:	RefCountProfiler.h
//
// summary:	Declares the reference count profiler class
///-------------------------------------------------------------------------------------------------

#ifndef __REFERENCE_COUNTED_PROFILER_H__
#define __REFERENCE_COUNTED_PROFILER_H__

#include <Windows.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <crtdbg.h>
#include <set>
#include "primitive_types.h"

namespace PTask {

    class ReferenceCounted;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Profiler class for reference counted objects
    ///             </summary>
    /// 
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class ReferenceCountedProfiler
    {
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the refcount profiler. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Initialize(BOOL bEnable);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitializes the refcount profiler. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Deinitialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the refcount profiler leaks. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Report(std::ostream& ss); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Profile allocation. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the item. </param>
        ///-------------------------------------------------------------------------------------------------

        static void RecordAllocation(ReferenceCounted * pItem);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Profile deletion. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the item. </param>
        ///-------------------------------------------------------------------------------------------------

        static void RecordDeletion(ReferenceCounted * pItem);

    protected:

        static LONG m_nRCAllocations;
        static LONG m_nRCDeletions;
        static LONG m_nRCProfilerInit;
        static LONG m_nRCProfilerEnable;
        static LONG m_nRCProfilerIDCount;
        static CRITICAL_SECTION m_csRCProfiler;
        static std::set<PTask::ReferenceCounted*> m_vAllAllocations;
    };
};

#endif  // __REFERENCE_COUNTED_PROFILER_H__