///-------------------------------------------------------------------------------------------------
// file:	AcceleratorManager.h
//
// summary:	Declares the accelerator manager class
///-------------------------------------------------------------------------------------------------

#pragma once
#include <deque>
#include <vector>
#include <set>
#include "accelerator.h"
#include "PhysicalDevice.h"
#include "Lockable.h"
#include <map>

namespace PTask {

    class Task;

    class AcceleratorManager : public Lockable
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        AcceleratorManager();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~AcceleratorManager(void);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a device. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void AddDevice(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pAccelerator' is available. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   true if available, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsAvailable(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for the first available. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="cls">  The cls. </param>
        /// <param name="v">    [in,out] [in,out] If non-null, the v. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL FindAvailable(ACCELERATOR_CLASS cls, std::vector<Accelerator*> &v);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the physical accelerator count. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The physical accelerator count. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetPhysicalAcceleratorCount();

    protected:

        /// <summary> The devices </summary>
        std::vector<PhysicalDevice*> m_devices;

        /// <summary> The available devices </summary>
        std::vector<PhysicalDevice*> m_available;
        
        /// <summary> The inflight devices </summary>
        std::vector<PhysicalDevice*> m_inflight;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for the first match for the given accelerator*. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        PhysicalDevice * Find(Accelerator* pAccelerator);
    };
};
