//--------------------------------------------------------------------------------------
// File: CompiledKernel.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _COMPILED_KERNEL_H_
#define _COMPILED_KERNEL_H_
#include "accelerator.h"
#include "ptlock.h"
#include <map>

namespace PTask {

	///-------------------------------------------------------------------------------------------------
	/// <summary>   function signature for host tasks that have dependences on other accelerators.
	///             The BOOL array contains entries which are true if that entry corresponds to an
	///             input already materialized on the dependent device, false otherwise. The
	///             pvDeviceBindings array contains entries which are meaningful when the entry at
	///             the same index in the BOOL array is true, and is a platform-specific device id.
	///             Generated code must know how to use these IDs.
	///             </summary>
	///
	/// <remarks>   Crossbac, 5/16/2012. </remarks>
	///-------------------------------------------------------------------------------------------------

	typedef void (__stdcall *LPFNTASKINITIALIZER)(DWORD dwThreadId, int nDeviceId);

    class CompiledKernel
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpszSourceFile">				[in] non-null, source file. </param>
        /// <param name="lpszOperation">				[in] non-null, the operation. </param>
        /// <param name="lpszInitializerBinary">		[in,out] If non-null, the initializer binary. </param>
        /// <param name="lpszInitializerEntryPoint">	[in,out] If non-null, the initializer entry
        /// 											point. </param>
        /// <param name="eInitializerPSObjectClass">	(Optional) the initializer ps object class. </param>
        ///-------------------------------------------------------------------------------------------------

        CompiledKernel(
			__in char *            lpszSourceFile, 
			__in char *            lpszOperation,
			__in char *            lpszInitializerBinary,
			__in char *            lpszInitializerEntryPoint,
			__in ACCELERATOR_CLASS eInitializerPSObjectClass=ACCELERATOR_CLASS_UNKNOWN
			);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Destructor. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~CompiledKernel(void);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        /// 	Gets the platform specific binary associated with the given accelerator. Generally
        /// 	speaking, we will compile a kernel separately for every accelerator in the system capable
        /// 	of running it, since the accelerators may have different capabilities. This method
        /// 	retrieves the result of that compilation, which is an object whose type depends on the
        /// 	platform supported by the accelerator. For example, in directX, this retrieves a compute
        /// 	shader interface.
        /// </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerator">	[in] non-null, the accelerator. </param>
        ///
        /// <returns>	null if it fails, else the platform specific binary. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void * GetPlatformSpecificBinary(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Sets a platform specific binary. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerator">			  	[in] non-null, the accelerator. </param>
        /// <param name="pPlatformSpecificBinary">	[in] non-null, the platform specific binary. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetPlatformSpecificBinary(Accelerator * pAccelerator, void * pPlatformSpecificBinary);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Gets a platform specific module. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerator">	[in] non-null, the accelerator. </param>
        ///
        /// <returns>	null if it fails, else the platform specific module. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void * GetPlatformSpecificModule(Accelerator * pAccelerator); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Sets a platform specific module. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerator">			  	[in] non-null, the accelerator. </param>
        /// <param name="pPlatformSpecificModule">	[in,out] If non-null, the platform specific module. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetPlatformSpecificModule(Accelerator * pAccelerator, void * pPlatformSpecificModule);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Gets the source file. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>	null if it fails, else the source file. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual const char * GetSourceFile();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Gets the operation. The operation is the top-level entry
        /// 			point into kernel code, and must be specified, since a single
        /// 			source file may contain many such entry points.
        /// 			</summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>	null if it fails, else the operation. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual const char * GetOperation();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Gets the source binary for init routine. </summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>	null if it fails, else the source file. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual const char * GetInitializerBinary();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Gets the entry point for any initializer routines.
        /// 			</summary>
        ///
        /// <remarks>	Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>	null if it fails, else the operation. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual const char * GetInitializerEntryPoint();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Sets the initializer binary. </summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///
		/// <param name="hModule">	The module. </param>
		///-------------------------------------------------------------------------------------------------

		virtual void SetInitializerBinary(HMODULE hModule);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Sets the initializer entry point. </summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///
		/// <param name="lpvProcAddress">	[in,out] If non-null, the lpv proc address. </param>
		///-------------------------------------------------------------------------------------------------

		virtual void SetInitializerEntryPoint(void * lpvProcAddress);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Query if this kernel has a static initializer that should be called as part
		/// 			of putting the graph in the run state. </summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///
		/// <returns>	true if static initializer, false if not. </returns>
		///-------------------------------------------------------------------------------------------------

		virtual BOOL HasStaticInitializer();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Determines if any present initializer routines requires platform-specific
		/// 			device objects to provided when called. </summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///
		/// <returns>	true if it succeeds, false if it fails. </returns>
		///-------------------------------------------------------------------------------------------------

		virtual BOOL InitializerRequiresPSObjects();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Gets initializer required ps classes. </summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///
		/// <returns>	null if it fails, else the initializer required ps classes. </returns>
		///-------------------------------------------------------------------------------------------------

		virtual ACCELERATOR_CLASS GetInitializerRequiredPSClass();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Executes the initializer, with a list of platform specific resources.
		/// 			</summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///
		/// <param name="vPSDeviceObjects">	[in,out] [in,out] If non-null, the ps device objects. </param>
		///
		/// <returns>	true if it succeeds, false if it fails. </returns>
		///-------------------------------------------------------------------------------------------------

		virtual BOOL 
		InvokeInitializer(
			__in DWORD dwThreadId,
			__in std::set<Accelerator*>& vPSDeviceObjects
			);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Executes the initializer, if present.
		/// 			</summary>
		///
		/// <remarks>	crossbac, 8/13/2013. </remarks>
		///
		/// <returns>	true if it succeeds, false if it fails. </returns>
		///-------------------------------------------------------------------------------------------------

		virtual BOOL InvokeInitializer(DWORD dwThreadId);

    protected:
        char * m_lpszSourceFile;
        char * m_lpszOperation;
        char * m_lpszInitializerBinary;
        char * m_lpszInitializerEntryPoint;
		ACCELERATOR_CLASS m_eInitializerPSObjectClass;
        std::map<Accelerator *, void *> m_vPlatformSpecificKernels;
        std::map<Accelerator *, void *> m_vPlatformSpecificModules;
		HANDLE m_lpvInitializerModule;
		void * m_lpvInitializerProcAddress;
		BOOL m_bInitializerInvoked;

        static std::map<std::string, HMODULE> m_vLoadedDlls;
        static std::map<std::string, std::map<std::string, FARPROC>> m_vEntryPoints;
        static PTLock m_vModuleLock;
    };

};
#endif

