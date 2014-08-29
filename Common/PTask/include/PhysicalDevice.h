///-------------------------------------------------------------------------------------------------
// file:	PhysicalDevice.h
//
// summary:	Declares the physical device class
///-------------------------------------------------------------------------------------------------

#pragma once
#include <deque>
#include <vector>
#include <set>
#include "oclhdr.h"
#include "ptdxhdr.h"
#include "cuhdr.h"
#include "accelerator.h"
#include "primitive_types.h"
#include "PhysicalDevice.h"
#include "Lockable.h"
#include <map>


namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// Forward declarations
    ///-------------------------------------------------------------------------------------------------

    class Task;
    class DXAccelerator;
#ifdef CUDA_SUPPORT
    class CUAccelerator;
#endif
#ifdef OPENCL_SUPPORT
    class CLAccelerator;
#endif
    class HostAccelerator;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   DIRECTX_DEVICERECORD: everything we have available to uniquely identify a device
    ///             through the DXGI API.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct dxdevrec_t {
        IDXGIAdapter * pAdapter;
        DXGI_ADAPTER_DESC desc;
    } DIRECTX_DEVICERECORD;

#ifdef OPENCL_SUPPORT

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   OPENCL_DEVICERECORD: everything we have available to uniquely identify a device
    ///             through OpenCL.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct cldevrec_t {
        cl_platform_id platform;
        cl_device_id device;
    } OPENCL_DEVICERECORD;
#endif

#ifdef CUDA_SUPPORT
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   CUDA_DEVICERECORD: everything we have available to uniquely identify a device
    ///             through CUDA APIs.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct cudevrec_t {
        CUdevice device;
    } CUDA_DEVICERECORD;
#endif

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Physical device object mapping a unique physical accelerator such as a GPU card
    ///             to Accelerator objects that use it through the various back-end runtimes that
    ///             PTask supports (DirectX, CUDA, OpenCL).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class PhysicalDevice : public Lockable
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        PhysicalDevice();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~PhysicalDevice(void);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this physical device is busy. We need this at the physical device layer
        ///             because a physical device may be busy through it's CUDA accelerator interface but
        ///             not through its DirectX interface, for example.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <returns>   true if busy, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsBusy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Mark this device as busy, meaning it is performing a dispatch for some Task.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="b">    true to b. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetBusy(BOOL b);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the Accelerator 'pAccelerator' is a runtime-specific interface
        /// 			for this physical device. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
        ///
        /// <returns>   true if same device, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsSameDevice(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pDevice' is same device. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pDevice">          [in,out] If non-null, the device. </param>
        /// <param name="pDesc">            [in,out] If non-null, the description. </param>
        /// <param name="nPlatformIndex">   Zero-based index of the n platform. </param>
        ///
        /// <returns>   true if same device, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsSameDevice(IDXGIAdapter * pDevice, DXGI_ADAPTER_DESC * pDesc, UINT nPlatformIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'platform'/'device' is same device. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="platform">         The platform. </param>
        /// <param name="device">           The device. </param>
        /// <param name="nPlatformIndex">   Zero-based index of the n platform. </param>
        ///
        /// <returns>   true if same device, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------
#ifdef OPENCL_SUPPORT
        virtual BOOL IsSameDevice(cl_platform_id platform, cl_device_id device, UINT nPlatformIndex);
#endif

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'device' is same device. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="device">           The device. </param>
        /// <param name="nPlatformIndex">   Zero-based index of the n platform. </param>
        ///
        /// <returns>   true if same device, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------
#ifdef CUDA_SUPPORT
        virtual BOOL IsSameDevice(CUdevice device, UINT nPlatformIndex);
#endif

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds an Accelerator interface to this physical device record. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL AddInterface(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this physical device has an Accelerator interface that can be used
        ///             to execute tasks with the given accelerator class.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="cls">  The accelerator class. </param>
        ///
        /// <returns>   true if the device has an interface of the given class, false otherwise. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL Supports(ACCELERATOR_CLASS cls);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets an accelerator interface on this physical device that can be used to execute
        ///             tasks of the given accelerator class. Return NULL if no such interface is present.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="cls">  The accelerator class. </param>
        ///
        /// <returns>   null if no appropriate interface is available, else the accelerator interface.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual Accelerator * GetAcceleratorInterface(ACCELERATOR_CLASS cls);

    protected:
        // note that we track a device entry per supported runtime: this is because the device may have
        // support from only a subset of the runtimes (e.g. Tesla cards, which do not enumerate as
        // Adapters and therefore enjoy OpenCL and CUDA support, but no DirectX support. 
        
        /// <summary> Data that uniquely identify the physical device 
        /// 		  using DirectX/DXGI APIs. NULL if no DirectX support is available
        /// 		  for this physical device.
        /// 		  </summary>
        DIRECTX_DEVICERECORD *  m_pDirectXDevice;        

        /// <summary> Data that uniquely identify the physical device 
        /// 		  using OpenCL APIs. NULL if no OpenCL support is available
        /// 		  for this physical device.
        /// 		  </summary>
#ifdef OPENCL_SUPPORT
        OPENCL_DEVICERECORD *   m_pOpenCLDevice;
#endif

        /// <summary> Data that uniquely identify the physical device 
        /// 		  using CUDA APIs. NULL if no CUDA support is available
        /// 		  for this physical device.
        /// 		  </summary>
#ifdef CUDA_SUPPORT
        CUDA_DEVICERECORD *		m_pCUDADevice;
#endif
        
        /// <summary> The DirectX Accelerator object that maps to this physical
        /// 		  device. NULL if no DirectX support is available
        /// 		  for this physical device.
        /// 		  </summary>
        DXAccelerator *         m_pDXAccelerator;
        
        /// <summary> The CUDA Accelerator object that maps to this physical
        /// 		  device. NULL if no CUDA support is available
        /// 		  for this physical device.
        /// 		  </summary>
#ifdef CUDA_SUPPORT
        CUAccelerator *         m_pCUAccelerator;
#endif

        /// <summary> The OpenCL Accelerator object that maps to this physical
        /// 		  device. NULL if no OpenCL support is available
        /// 		  for this physical device.
        /// 		  </summary>
#ifdef OPENCL_SUPPORT
        CLAccelerator *         m_pCLAccelerator;
#endif

        /// <summary> The Host Accelerator object that maps to this physical
        /// 		  device. Not used.
        /// 		  </summary>
        HostAccelerator *       m_pHostAccelerator;
        
        /// <summary> true if this device is in flight, meaning it is currently
        /// 		  being used in the dispatch of a Task. 
        /// 		  </summary>
        BOOL                    m_bInFlight;
    };
};
