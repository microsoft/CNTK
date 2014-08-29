///-------------------------------------------------------------------------------------------------
// file:	ScopedPoolManager.h
//
// summary:	Declares the scoped pool manager class
///-------------------------------------------------------------------------------------------------

#ifndef __SCOPED_POOL_MANAGER__
#define __SCOPED_POOL_MANAGER__

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

    class ScopedPoolManager : public Lockable {

        typedef std::tuple<DatablockTemplate*, int, int, int, int> POOLDESCRIPTOR;

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/27/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        ScopedPoolManager(Graph * pScopedGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/27/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~ScopedPoolManager();

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

        BOOL
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

        BOOL
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

        BOOL
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

        BOOL
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

        Datablock * 
        RequestBlock(
            __in DatablockTemplate * pTemplate,
            __in UINT                uiDataSize,
            __in UINT                uiMetaSize,
            __in UINT                uiTemplateSize
            );

    protected: 

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

        Graph *                                         m_pGraph;
        BOOL                                            m_bPoolsAllocated;
        BOOL                                            m_bDestroyed;
        std::map<int, POOLDESCRIPTOR>                   m_vRequiredPoolsUntyped;
        std::map<DatablockTemplate*, POOLDESCRIPTOR>    m_vRequiredPoolsTyped;
        std::map<int, GlobalBlockPool*>                 m_vUntypedBlockPools;
        std::map<DatablockTemplate*, GlobalBlockPool*>  m_vTypedBlockPools;

    };

};

#endif
