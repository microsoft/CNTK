///-------------------------------------------------------------------------------------------------
// file:	DeviceMemoryStatus.h
//
// summary:	Declares the device memory status class
///-------------------------------------------------------------------------------------------------

#ifndef __DEVICE_MEMORY_STATUS_H__
#define __DEVICE_MEMORY_STATUS_H__

#include "primitive_types.h"
#include "Lockable.h"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <map>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Memory status for a memory type on a device. 
    ///             Currently we track global and page-locked memory.
    ///             Could easily expand to track other types. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct DeviceMemoryStatus_t {

        /// <summary>   The name. </summary>
        std::string m_name;

        /// <summary>   A record of all allocations: maps the pointer to the size </summary>
        std::map<void *, unsigned __int64> m_vAllocations;

        /// <summary>   The size in bytes of the memory space. </summary>
        unsigned __int64  m_uiMemorySpaceSize;

        /// <summary>   The size in bytes of the smallest allocated extent. </summary>
        unsigned __int64  m_uiMinAllocExtentSize;

        /// <summary>   The size in bytes of the largest allocated extent. </summary>
        unsigned __int64  m_uiMaxAllocExtentSize;

        /// <summary>   (historical) the low water mark for total allocated bytes. </summary>
        unsigned __int64  m_uiLowWaterMarkBytes;

        /// <summary>   (historical) the high water mark for total allocated bytes. </summary>
        unsigned __int64  m_uiHighWaterMarkBytes;

        /// <summary>   (current state) the total bytes currently allocated. </summary>
        unsigned __int64  m_uiCurrentlyAllocatedBytes;

        /// <summary>   (current state) the total number of currently allocated buffers. </summary>
        unsigned __int64  m_uiCurrentlyAllocatedBuffers;

        /// <summary>   The total number of allocation requests. </summary>
        unsigned __int64 m_uiAllocationRequests;

        /// <summary>   The total deallocation requests. </summary>
        unsigned __int64 m_uiDeallocationRequests;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        DeviceMemoryStatus_t(
            std::string &szName,
            char * lpszUniquifier
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        ~DeviceMemoryStatus_t();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets this object. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Reset(
            VOID
            ); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory allocation. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">    [in,out] If non-null, extent of the memory. </param>
        /// <param name="uiBytes">          The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        RecordAllocation(
            __in void * pMemoryExtent,
            __in size_t uiBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory deallocation. We provide "require entry" flag to
        ///             simplify tracking of page-locked allocations which are a strict subset
        ///             of all allocations. If we are removing an entry from the global tracking,
        ///             we require that an entry for it be found, otherwise we complain. If
        ///             we are removing entries from the page-locked tracking, it is not an  
        ///             error if there is no entry present.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">    [in,out] If non-null, extent of the memory. </param>
        /// <param name="bRequireEntry">    true to pinned allocation. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        RecordDeallocation(
            __in void * pMemoryExtent,
            __in BOOL bRequireEntry
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the allocation statistics. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Report(
            std::ostream &ios
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the memory space size described by uiBytes. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="uiBytes">  The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        UpdateMemorySpaceSize(
            unsigned __int64 uiBytes
            );
    
    } MEMSTATEDESC;

    typedef struct GlobalDeviceMemoryState_t {

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        GlobalDeviceMemoryState_t(
            std::string& szDeviceName
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        ~GlobalDeviceMemoryState_t(
            VOID
            );

        /// <summary>   synchronization. </summary>
        void Lock();
        void Unlock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets the stats. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Reset(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory allocation. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">    [in,out] If non-null, extent of the memory. </param>
        /// <param name="uiBytes">          The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        RecordAllocation(
            __in void * pMemoryExtent,
            __in size_t uiBytes,
            __in BOOL bPinned
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory deallocation. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">        [in,out] If non-null, extent of the memory. </param>
        /// <param name="bPinnedAllocation">    true to pinned allocation. </param>
        /// <param name="uiBytes">              The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        RecordDeallocation(
            __in void * pMemoryExtent
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the allocation statistics. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Report(
            std::ostream &ios
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets global memory state. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the global memory state. </returns>
        ///-------------------------------------------------------------------------------------------------

        MEMSTATEDESC * 
        GetGlobalMemoryState(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets global memory state. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the global memory state. </returns>
        ///-------------------------------------------------------------------------------------------------

        MEMSTATEDESC * 
        GetPageLockedMemoryState(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the memory space size described by uiBytes. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="uiBytes">  The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        UpdateMemorySpaceSize(
            unsigned __int64 uiBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the percentage of this memory space that is allocated. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <returns>   The allocated percent. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetAllocatedPercent(
            void
            );

    protected:

        MEMSTATEDESC m_global;
        MEMSTATEDESC m_pagelocked;
        CRITICAL_SECTION m_lock;
    
    } DEVICEMEMORYSTATE;

};

#endif  // __DEVICE_MEMORY_STATUS_H__