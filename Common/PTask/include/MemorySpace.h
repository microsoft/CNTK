///-------------------------------------------------------------------------------------------------
// file:	MemorySpace.h
//
// summary:	Simple class describing a memory space
///-------------------------------------------------------------------------------------------------

#ifndef __MEMORY_SPACE_H__
#define __MEMORY_SPACE_H__

#include "primitive_types.h"
#include "Lockable.h"
#include <map>
#include <set>
#include <iostream>
#include <sstream>

namespace PTask {

    static const UINT HOST_MEMORY_SPACE_ID = 0;
    static const UINT MAX_MEMORY_SPACES = 12;  
    static const UINT UNKNOWN_MEMORY_SPACE_ID = 0xFFFFFFFF;

    class Accelerator;

    typedef void * (__stdcall *LPFNSTATICALLOCATOR)(ULONG, ULONG);
    typedef void   (__stdcall *LPFNSTATICDEALLOCATOR)(void*);

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Memory status for a memory type on a device. 
    ///             Currently we track global and page-locked memory.
    ///             Could easily expand to track other types. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    struct DeviceMemoryStatus_t;
    struct GlobalDeviceMemoryState_t;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Memory space object. Encapsulates data about what accelerators are associated
    ///             with the space, whether there are specialized allocators for managing buffers
    ///             created in other spaces that must communicate witht this one, whether we need an
    ///             accelerator object to perform allocations in this space (or any static allocators
    ///             otherwise).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class MemorySpace : public Lockable {
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of memory spaces active in the system. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   The number of memory spaces. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT GetNumberOfMemorySpaces();       

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerator from memory space identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="id">   The identifier. </param>
        ///
        /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Accelerator * GetAcceleratorFromMemorySpaceId(UINT id);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerator from memory space identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="id">   The identifier. </param>
        ///
        /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
        ///-------------------------------------------------------------------------------------------------

        static MemorySpace * GetMemorySpaceFromId(UINT id);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the percentage of this space already allocated. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <returns>   The allocated percent. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT            GetAllocatedPercent(UINT uiMemorySpaceId);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets allocation percentages. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <param name="vDeviceMemories">  [in,out] The device memories. </param>
        ///-------------------------------------------------------------------------------------------------

        static void            GetAllocationPercentages(std::map<UINT, UINT>& vDeviceMemories);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerator from memory space identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="id">   The identifier. </param>
        ///
        /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL HasStaticAllocator(UINT id);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate an extent in this memory space. Fails if 
        /// 			no static allocator is present. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="uiMemorySpace">        The identifier. </param>
        /// <param name="ulBytesToAllocate">    The ul bytes to allocate. </param>
        /// <param name="ulFlags">              The ul flags. </param>
        ///
        /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
        ///-------------------------------------------------------------------------------------------------

        static void * AllocateMemoryExtent(UINT uiMemorySpace, ULONG ulBytesToAllocate, ULONG ulFlags);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deallocate an extent in this memory space. Fails if no static allocator is present.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="uiMemorySpace">    The identifier. </param>
        /// <param name="pMemoryExtent">    [in,out] The ul bytes to allocate. </param>      
        ///-------------------------------------------------------------------------------------------------

        static void DeallocateMemoryExtent(UINT uiMemorySpace, void * pMemoryExtent);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Registers the memory space. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="pSpace">       [in,out] memory space. </param>
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        static void RegisterMemorySpace(MemorySpace * pSpace, Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Associate the accelerator with the memory space. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="pSpace">       [in,out] memory space. </param>
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        static void RegisterMemorySpaceId(UINT id, Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the memory space map. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void InitializeMemorySpaces();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unregisters the memory spaces at tear-down time. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void UnregisterMemorySpaces();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="lpszProtectedObjectName">  [in] If non-null, name of the protected object. </param>
        ///-------------------------------------------------------------------------------------------------

        MemorySpace(std::string& szDeviceName, UINT nMemorySpaceId);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~MemorySpace();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this memory space has a static buffer allocator function. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <returns>   true if static allocator, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL        HasStaticAllocator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate a memory extent in this memory space of the given
        /// 			size. If this memory space does not have a static allocator,
        /// 			return NULL. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="ulNumberOfBytes">  The ul number of in bytes. </param>
        /// <param name="ulFlags">          The ul flags. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        void *          AllocateMemoryExtent(ULONG ulNumberOfBytes, ULONG ulFlags);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deallocate memory extent. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="pvMemoryExtent">   [in,out] If non-null, extent of the pv memory. </param>
        ///-------------------------------------------------------------------------------------------------

        void            DeallocateMemoryExtent(void* pvMemoryExtent);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the percentage of this space already allocated. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <returns>   The allocated percent. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT            __GetAllocatedPercent();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a pointer to any accelerator mapped to this space. Most spaces
        /// 			have just one, so this simplifies the process of getting an object
        /// 			that can provide allocation services if no static allocator is present. 
        /// 		    </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <returns>   null if it fails, else any accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator *   GetAnyAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of accelerators mapped to this space. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <returns>   The number of accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT            GetNumberOfAccelerators();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets all accelerators in this space, by putting them in the user-provided buffer.
        ///             At most nMaxAccelerators will be provided.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="ppAccelerators">   [in,out] If non-null, the accelerators. </param>
        /// <param name="nMaxAccelerators"> The maximum accelerators. </param>
        ///
        /// <returns>   The number of accelerators in the result buffer, which may be different from
        ///             nMaxAccelerators!
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT            GetAccelerators(Accelerator ** ppAccelerators, UINT nMaxAccelerators);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assign a unique memory space identifier. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT AssignUniqueMemorySpaceIdentifier();  

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a static allocator function for this memory space. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="lpfnStaticAllocatorFunction">      The lpfn static allocator function. </param>
        /// <param name="lpfnStaticDeallocatorFunction">    The lpfn static deallocator function. </param>
        ///-------------------------------------------------------------------------------------------------

        void        SetStaticAllocator(LPFNSTATICALLOCATOR lpfnStaticAllocatorFunction,
                                       LPFNSTATICDEALLOCATOR lpfnStaticDeallocatorFunction
                                       );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a deferred allocation entry for the proxy accelerator, indicating that
        ///             allocations for this space should be deferred to accelerators for that space,
        ///             when the resulting buffers will be used to commnunicate between those spaces.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="pProxyAllocatorAccelerator">   [in,out] If non-null, the proxy allocator
        ///                                             accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void        AddDeferredAllocationEntry(Accelerator* pProxyAllocatorAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an accelerator to this memory space. </summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void        AddAccelerator(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the space size bytes described by uiBytes. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="uiBytes">  The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void        UpdateSpaceSizeBytes(unsigned __int64 uiBytes);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets the memory state. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Reset();

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
        /// <summary>   Gets memory state. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the memory state. </returns>
        ///-------------------------------------------------------------------------------------------------

        struct GlobalDeviceMemoryState_t * GetMemoryState();        

    private:

        /// <summary>   Name of the device. </summary>
        std::string             m_strDeviceName;

        /// <summary> Identifier for the memory space </summary>
        UINT                    m_nMemorySpaceId;

        /// <summary> Pointer to a static allocator function, if 
        /// 		  one exists for this memory space. 
        /// 		  </summary>
        LPFNSTATICALLOCATOR     m_lpfnStaticAllocator;

        /// <summary> Pointer to a static de-allocator function, if 
        /// 		  one exists for this memory space. 
        /// 		  </summary>
        LPFNSTATICDEALLOCATOR     m_lpfnStaticDeallocator;

        /// <summary> The deferred allocator map. Each entry in this
        /// 		  set indicates that memory allocations in this space
        /// 		  should be deferred to allocators provided by 
        /// 		  acclerators mapped to the space identified by the
        /// 		  entry. For example, if this memory space describes
        /// 		  the host memory space, it will contain an entry for
        /// 		  every CUDA memory space because we should be using
        /// 		  cuda APIs to allocate host memory for best performance.
        /// 		   </summary>
        std::set<UINT>          m_pDeferredAllocatorSpaces;

        /// <summary> The accelerators mapped to this space. </summary>
        std::set<Accelerator*>  m_pAccelerators;

        /// <summary>   State of the memory. </summary>
        struct GlobalDeviceMemoryState_t *     m_pMemoryState;

        /// <summary> Counter for assigning unique identifiers
        /// 		  to Memory spaces objects. 
        /// 		  </summary>
        static UINT m_uiMemorySpaceIdCounter;

        /// <summary> static MemorySpace map </summary>
        static MemorySpace* m_vMemorySpaceMap[MAX_MEMORY_SPACES];
    };
};

#endif  // __MEMORY_SPACE_H__