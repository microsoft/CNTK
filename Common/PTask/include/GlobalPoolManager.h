///-------------------------------------------------------------------------------------------------
// file:	GlobalPoolManager.h
//
// summary:	Declares the global pool manager class
///-------------------------------------------------------------------------------------------------

#ifndef __GLOBAL_POOL_MANAGER__
#define __GLOBAL_POOL_MANAGER__

#include <stdio.h>
#include <crtdbg.h>
#include <Windows.h>
#include "datablock.h"
#include "GlobalBlockPool.h"
#include "ptlock.h"
#include <deque>
#include <map>
#include <tuple>

namespace PTask { 

    class CompiledKernel;
    class Graph;
    class Channel;
    class Port;
    class Task;
    class Datablock;    
    class DatablockTemplate;

    class GlobalPoolManager : public Lockable {

    public:
        static GlobalPoolManager * Create();
        static void Destroy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Require block pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
        /// <param name="nDataSize">        Size of the data. </param>
        /// <param name="nMetaSize">        Size of the meta. </param>
        /// <param name="nTemplateSize">    Size of the template. </param>
        /// <param name="nBlocks">          (Optional) The blocks. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        RequireBlockPool(
            __in DatablockTemplate * pTemplate,
            __in int                 nDataSize, 
            __in int                 nMetaSize, 
            __in int                 nTemplateSize,
            __in int                 nBlocks=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Require block pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="nDataSize">        Size of the data. </param>
        /// <param name="nMetaSize">        Size of the meta. </param>
        /// <param name="nTemplateSize">    Size of the template. </param>
        /// <param name="nBlocks">          (Optional) The blocks. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        RequireBlockPool(
            __in int                 nDataSize, 
            __in int                 nMetaSize, 
            __in int                 nTemplateSize,
            __in int                 nBlocks=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Require block pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="pTemplate">    [in,out] If non-null, the template. </param>
        /// <param name="nBlocks">      (Optional) The blocks. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        RequireBlockPool(
            __in DatablockTemplate * pTemplate,
            __in int                 nBlocks=0
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find a block pool for the block. If there is no good fit,
        ///             create one if the bCreateIfNotFound flag is set. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 8/30/2013. </remarks>
        ///
        /// <param name="pBlock">               [in,out] If non-null, the block. </param>
        /// <param name="bCreateIfNotFound">    The create if not found. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL
        AddBlockToBestFitPool(
            __in Datablock * pBlock,
            __in BOOL bCreateIfNotFound
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if we can allocate pools. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL AllocatePools();    

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroys the pools. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL DestroyPools();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate datablock. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
        /// <param name="uiDataSize">       Size of the data. </param>
        /// <param name="uiMetaSize">       Size of the meta. </param>
        /// <param name="uiTemplateSize">   Size of the template. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock * 
        AllocateDatablock(
            __in DatablockTemplate * pTemplate,
            __in UINT                uiDataSize,
            __in UINT                uiMetaSize,
            __in UINT                uiTemplateSize
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Request a pooled block. </summary>
        ///
        /// <remarks>   crossbac, 8/21/2013. </remarks>
        ///
        /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
        /// <param name="uiDataSize">       Size of the data. </param>
        /// <param name="uiMetaSize">       Size of the meta. </param>
        /// <param name="uiTemplateSize">   Size of the template. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        RequestBlock(
            __in DatablockTemplate * pTemplate,
            __in UINT                uiDataSize,
            __in UINT                uiMetaSize,
            __in UINT                uiTemplateSize
            );

    protected: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   crossbac, 8/21/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        GlobalPoolManager();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   crossbac, 8/21/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~GlobalPoolManager();


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for the first matching pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
        /// <param name="uiDataSize">           Size of the data. </param>
        /// <param name="uiMetaSize">           Size of the meta. </param>
        /// <param name="uiTemplateSize">       Size of the template. </param>
        /// <param name="uiBlockControlCode">   The block control code. </param>
        ///
        /// <returns>   null if it fails, else the found matching pool. </returns>
        ///-------------------------------------------------------------------------------------------------

        GlobalBlockPool * 
        FindMatchingPool(
            __in DatablockTemplate * pTemplate,
            __in UINT                uiDataSize,
            __in UINT                uiMetaSize,
            __in UINT                uiTemplateSize
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Find a block pool for the block. If there is no good fit,
        ///             create one if the bCreateIfNotFound flag is set. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 8/30/2013. </remarks>
        ///
        /// <param name="pBlock">               [in,out] If non-null, the block. </param>
        /// <param name="bCreateIfNotFound">    The create if not found. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        __AddBlockToBestFitPool(
            __in Datablock * pBlock,
            __in BOOL bCreateIfNotFound
            );

        static void WarnIfInitialized(char * lspzFunction);
        typedef std::tuple<DatablockTemplate*, int, int, int, int> POOLDESCRIPTOR;
        static GlobalPoolManager *                          g_pGlobalPoolManager;
        static BOOL                                         g_bPoolsAllocated;
        static PTLock                                       g_vPoolsLock;
        static std::map<int, POOLDESCRIPTOR>                g_vRequiredPoolsUntyped;
        static std::map<DatablockTemplate*, POOLDESCRIPTOR> g_vRequiredPoolsTyped;
        std::map<int, GlobalBlockPool*>                     g_vUntypedBlockPools;
        std::map<DatablockTemplate*, GlobalBlockPool*>      g_vTypedBlockPools;

        virtual GlobalPoolManager *                           GetPoolManager() { return g_pGlobalPoolManager; }
        virtual BOOL                                          ArePoolsAllocated() { return g_bPoolsAllocated; }
        virtual PTLock *                                      GetPoolLock() { return &g_vPoolsLock; }
        virtual std::map<int, POOLDESCRIPTOR>*                GetRequiredPoolsUntyped() { return &g_vRequiredPoolsUntyped; }
        virtual std::map<DatablockTemplate*, POOLDESCRIPTOR>* GetRequiredPoolsTyped() { return &g_vRequiredPoolsTyped; }
    };

};

#endif
