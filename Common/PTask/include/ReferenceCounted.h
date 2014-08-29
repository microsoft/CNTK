///-------------------------------------------------------------------------------------------------
// file:	ReferenceCounted.h
//
// summary:	Declares the reference counted class
///-------------------------------------------------------------------------------------------------

#ifndef __REFERENCE_COUNTED_H__
#define __REFERENCE_COUNTED_H__

#include <Windows.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <crtdbg.h>
#include <set>
#include "primitive_types.h"
#include "Lockable.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reference counted super-class, allowing to share implementation of ref count
    ///             management code.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class ReferenceCounted : public Lockable 
    {
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        ReferenceCounted();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="lpszProtectedObjectName">  [in] non-null, name of the protected object. </param>
        ///-------------------------------------------------------------------------------------------------

        ReferenceCounted(char * lpszProtectedObjectName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------
        virtual ~ReferenceCounted();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a reference. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual LONG AddRef();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Release a reference. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual LONG Release();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the reference count. (for debugging only) </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   current reference count for the object. </returns>
        ///-------------------------------------------------------------------------------------------------

        LONG RefCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Datablock.toString() </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="os">       [in,out] The operating system. </param>
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        ///
        /// <returns>   The shifted result. </returns>
        ///-------------------------------------------------------------------------------------------------

        friend std::ostream& operator<<(std::ostream &os, ReferenceCounted * pBlock); 

    protected:

        /// <summary> Number of outanding references to this object.
        /// 		  When m_uiRefCount drops to zero, it will be
        /// 		  garbage collected. NB: Ideally, the refcount would be private. However,
        /// 		  class Datablock inherits from ReferenceCounted but has to override Release to return blocks to
        ///           their block pools rather than deleting them (if they are pooled). Doing this requires the
        ///           ability to do interlocked operations on the m_uiRefCount member of the super-class. A sad
        ///           side effect of this is that we are forced to make m_uiRefCount protected rather than private. 
        /// 		  </summary>
        LONG					m_uiRefCount;

    public: 

        /// <summary>   The unique id of this RC object. </summary>
        LONG          m_uiUID;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the refcount profiler. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL RCProfileInitialize(BOOL bEnable);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the refcount profiler leaks. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void RCProfileDumpLeaks(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Profile allocation. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the item. </param>
        ///-------------------------------------------------------------------------------------------------

        static void RCProfileAllocation(ReferenceCounted * pItem);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Profile deletion. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the item. </param>
        ///-------------------------------------------------------------------------------------------------

        static void RCProfileDeletion(ReferenceCounted * pItem);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a string describing this refcount object. Allows subclasses to
        ///             provide overrides that make leaks easier to find when detected by the
        ///             rc profiler. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/9/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the rectangle profile descriptor. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual std::string GetRCProfileDescriptor();

    };
};

#endif  // __REFERENCE_COUNTED_H__