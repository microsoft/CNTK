//--------------------------------------------------------------------------------------
// File: ptaskutils.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PTASK_UTILS_H_
#define _PTASK_UTILS_H_
#include <Windows.h>
#include "primitive_types.h"

namespace PTask {

    static const unsigned int DEFAULT_GROUP_SIZE = 256;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent how to interpret raw buffer contents when using
    ///             DUMP_INTERMEDIATE_BLOCKS for debugging.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum dumptype_t {
        dt_raw = 0,
        dt_float = 1,
        dt_int = 2, 
		dt_double = 3
    } DEBUGDUMPTYPE;

    class ptaskutils
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   derive the best group size for dispatch. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="group_size">   Size of the group. </param>
        /// <param name="global_size">  Size of the global. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static size_t 
        ptaskutils::roundup(
            int group_size, 
            int global_size
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return a unique integer identifier. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static unsigned int
            ptaskutils::nextuid(
            void
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Select the accelerator class for the given file, assumed to contain shader/kernel
        ///             code.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="szFile">   The file. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static ACCELERATOR_CLASS
            ptaskutils::SelectAcceleratorClass(
            const char * szFile
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Loads file into memory. </summary>
        ///
        /// <remarks>   Crossbac, 1/28/2013. </remarks>
        ///
        /// <param name="hFile">    The file. </param>
        /// <param name="ppMemory"> [in,out] If non-null, the memory. </param>
        /// <param name="puiBytes"> [in,out] If non-null, the pui in bytes. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        LoadFileIntoMemory(
            const HANDLE hFile,
            void ** ppMemory,
            UINT * puiBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Loads file into memory. </summary>
        ///
        /// <remarks>   Crossbac, 1/28/2013. </remarks>
        ///
        /// <param name="szFile">   The file. </param>
        /// <param name="ppMemory"> [in,out] If non-null, the memory. </param>
        /// <param name="puiBytes"> [in,out] If non-null, the pui in bytes. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        LoadFileIntoMemory(
            const char * szFile,
            void ** ppMemory,
            UINT * puiBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Loads file into memory. </summary>
        ///
        /// <remarks>   Crossbac, 1/28/2013. </remarks>
        ///
        /// <param name="szFile">   The file. </param>
        /// <param name="ppMemory"> [in,out] If non-null, the memory. </param>
        /// <param name="puiBytes"> [in,out] If non-null, the pui in bytes. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        LoadFileIntoMemory(
            const WCHAR * pwszFile,
            void ** ppMemory,
            UINT * puiBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Returns the number of set signal codes in a control signal. </summary>
        ///
        /// <remarks>   Crossbac, 2/14/2013. </remarks>
        ///
        /// <param name="luiSignalWord">    The lui signal word. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT 
        SignalCount(
            __in CONTROLSIGNAL luiSignalWord
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   get the index of the first set signal if any. </summary>
        ///
        /// <remarks>   Crossbac, 2/14/2013. </remarks>
        ///
        /// <param name="luiSignalWord">    The lui signal word. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static int
        GetFirstSignalIndex(
            __in CONTROLSIGNAL luiSignalWord
            );

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes utils. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void				initialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Cleans up utils. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void				cleanup();
        
        /// <summary> Unique id lock </summary>
        static CRITICAL_SECTION m_csUIDLock;
        
        /// <summary> The uid counter </summary>
        static unsigned int		m_uiUIDCounter;
        
        /// <summary> true if utils is initialized </summary>
        static BOOL				m_bInitialized;
    };

};
#endif

