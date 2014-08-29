//--------------------------------------------------------------------------------------
// File: datablock.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

#ifndef _DATABLOCK_H_
#define _DATABLOCK_H_

#include "primitive_types.h"
#include "ReferenceCounted.h"
#include "MemorySpace.h"
#include <map>
#include <set>
#include <stack>
#include <vector>
#include <assert.h>

class CHighResolutionTimer;

namespace PTask {

    class PBuffer;
    class Port;
    class Task;
    class OutputPort;
    class DatablockTemplate;
    class AsyncDependence;
    class AsyncContext;
    class BlockPoolOwner;
    class CoherenceProfiler;
    class DatablockProfiler;
    class SignalProfiler;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Buffer map entry. State an pbuffers
    /// 			for a given memory space. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct BUFFER_MAP_ENTRY_t {

        /// <summary> The state M,S,I, or No entry</summary>
        BUFFER_COHERENCE_STATE      eState;

        /// <summary> Identifier for the memory space represented 
        /// 		  by this entry. For convenience. </summary>
        UINT                        nMemorySpaceId; 

        /// <summary> PBuffer for a block's data channel at index 0</summary>
        /// <summary> PBuffer for a block's metadata channel at index 1</summary>
        /// <summary> PBuffer for a block's template channel at index 2</summary>
        PBuffer *                   pBuffers[NUM_DATABLOCK_CHANNELS];

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        BUFFER_MAP_ENTRY_t() : eState(BSTATE_NO_ENTRY)
        { 
            pBuffers[DBDATA_IDX] = NULL;
            pBuffers[DBMETADATA_IDX] = NULL;
            pBuffers[DBTEMPLATE_IDX] = NULL;
            nMemorySpaceId = 0xFFFFFFFF;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        BUFFER_MAP_ENTRY_t(UINT i) : eState(BSTATE_NO_ENTRY)
        { 
            pBuffers[DBDATA_IDX] = NULL;
            pBuffers[DBMETADATA_IDX] = NULL;
            pBuffers[DBTEMPLATE_IDX] = NULL;
            nMemorySpaceId = i;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="state">        The state. </param>
        /// <param name="pData">        [in,out] If non-null, the data. </param>
        /// <param name="pMeta">        [in,out] If non-null, the meta. </param>
        /// <param name="pTemplate">    [in,out] If non-null, the template. </param>
        ///-------------------------------------------------------------------------------------------------

        BUFFER_MAP_ENTRY_t(
            UINT                   id,
            BUFFER_COHERENCE_STATE state,
            PBuffer *              pData,
            PBuffer *              pMeta,
            PBuffer *              pTemplate
            ) : eState(state) 
        {
            nMemorySpaceId = id;
            pBuffers[DBDATA_IDX] = pData;
            pBuffers[DBMETADATA_IDX] = pMeta;
            pBuffers[DBTEMPLATE_IDX] = pTemplate;        
        }

    } BUFFER_MAP_ENTRY;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Datablock: memory-space transparent logical buffer. A Datablock represents a unit
    ///             of data flow along an edge in a graph. A DatablockTemplate provides meta-data
    ///             describing Datablocks, and helps map the raw data contained in Datablocks to
    ///             hardware threads on the GPU.
    ///             
    ///             Data flows through a Graph as discrete Datablocks, even if the external input to
    ///             and/or output from the Graph is a continuous stream of data values. Datablocks
    ///             refer to and are described by DatablockTemplate objects (see below) which are
    ///             meta-data describing the dimensions and layout of data in the block.  The
    ///             datablock abstraction provides a coherent view on data that may migrate between
    ///             memory spaces. Datablocks encapsulate buffers in multiple memory spaces using a
    ///             buffermap\ property whose entries map memory spaces to device-specific buffer
    ///             objects. The \buffermap\ tracks which buffer(s)
    ///             represent the most up-to-date view(s) of the underlying data, enabling a \
    ///             datablock{} to materialize views in different memory spaces on demand. For
    ///             example, a Datablock may be created based on a buffer in CPU memory.  When a \
    ///             ptask{} is about to execute using that Datablock, the runtime will notice that no
    ///             corresponding buffer exists in the GPU memory space where the \ptask{} has been
    ///             scheduled, and will create that view accordingly. The converse occurs for data
    ///             written by the GPU---buffers in the CPU memory domain will be populated lazily
    ///             based on the GPU version only when a request for that data occurs. \ Datablocks{}
    ///             contain a \recordcount\ member, used to help manage downstream memory allocation
    ///             for computations that work with record streams or variable- stride data (see
    ///             below). Datablocks can be pushed concurrently into multiple \ channels{}, can be
    ///             shared across processes, and are garbage-collected based on reference counts.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class Datablock : 
        
        // we inherit from ReferenceCounted but Datablock has to override Release to return blocks to
        // their block pools rather than deleting them (if they are pooled). Doing this requires the
        // ability to do interlocked operations on the m_uiRefCount member of the super-class. A sad
        // side effect of this is that we are forced to make m_uiRefCount protected rather than private. 
        public ReferenceCounted

    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Channel iterator type. Simplifies the logic in Task subclasses required to find
        ///             and bind an arbitrary set of channels, and decouples the data structures used to
        ///             implement multiple channels from the classes that need to work with data in those
        ///             channels.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/3/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        typedef struct ChannelIterator_t {
        public:

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Default constructor. Initialized to an invalid state. We don't
            /// 			want anyone actually using a ChannelIterator without initializing
            /// 			it by calling Datablock::FirstChannel();
            /// 			</summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///-------------------------------------------------------------------------------------------------

            ChannelIterator_t() {
                m_nChannelIndex = NUM_DATABLOCK_CHANNELS+1; // start in invalid state
                m_pIteratedBlock = NULL;
            }

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Constructor. Start at the data channel--if there is no data channel advance until
            ///             there is a valid channel.
            ///             </summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///
            /// <param name="pBlock">   [in,out] If non-null, the block. </param>
            ///-------------------------------------------------------------------------------------------------

            ChannelIterator_t(Datablock * pBlock) {
                m_nChannelIndex = DBDATA_IDX;
                m_pIteratedBlock = pBlock;
                AdvanceToNextValidChannel();
            }

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Constructor. Start at the given position--if there is no channel at that index,
            ///             advance until there is a valid channel.
            ///             </summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///
            /// <param name="pBlock">   [in,out] If non-null, the block. </param>
            /// <param name="pos">      The position. </param>
            ///-------------------------------------------------------------------------------------------------

            ChannelIterator_t(Datablock * pBlock, size_t pos) {
                m_nChannelIndex = pos;
                assert(pos <= NUM_DATABLOCK_CHANNELS); // use <= because we want to create end-sentinal objects
                m_pIteratedBlock = pBlock;
                AdvanceToNextValidChannel();
            }

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Assignment operator. </summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///
            /// <param name="rhs">  [in,out] The right hand side. </param>
            ///-------------------------------------------------------------------------------------------------

            void operator =(ChannelIterator_t &rhs) {
                m_nChannelIndex = rhs.m_nChannelIndex;
                m_pIteratedBlock = rhs.m_pIteratedBlock;
            }

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Equality operator. Two ChannelIterators are equal if they are at the same channel
            ///             index. Technically, we should verify that the iterated block is the same, but
            ///             that's just wasted compute.
            ///             </summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///
            /// <param name="rhs">  [in,out] The right hand side. </param>
            ///
            /// <returns>   true if the parameters are considered equivalent. </returns>
            ///-------------------------------------------------------------------------------------------------

            BOOL operator ==(ChannelIterator_t &rhs) {
                return m_nChannelIndex == rhs.m_nChannelIndex;
            }

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Inequality. Two ChannelIterators are unequal if they are not at the same channel
            ///             index.
            ///             </summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///
            /// <param name="rhs">  [in,out] The right hand side. </param>
            ///-------------------------------------------------------------------------------------------------

            BOOL operator !=(ChannelIterator_t &rhs) {
                return m_nChannelIndex != rhs.m_nChannelIndex;
            }

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Increment operator. Advance the channel index. If there is no channel data
            /// 			at that index, advance until the next valid channel or there are no more
            /// 			channels.</summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///-------------------------------------------------------------------------------------------------

            void operator ++() {
                assert(m_nChannelIndex <= NUM_DATABLOCK_CHANNELS);
                if(m_nChannelIndex < NUM_DATABLOCK_CHANNELS) {
                    m_nChannelIndex++;
                    AdvanceToNextValidChannel();
                }
            }

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Increment operator. Advance the channel index. If there is no channel data
            /// 			at that index, advance until the next valid channel or there are no more
            /// 			channels.</summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///-------------------------------------------------------------------------------------------------

            void operator ++(int) {
                assert(m_nChannelIndex <= NUM_DATABLOCK_CHANNELS);
                if(m_nChannelIndex < NUM_DATABLOCK_CHANNELS) {
                    m_nChannelIndex++;
                    AdvanceToNextValidChannel();
                }
            }
        
        private:
            /// <summary> Zero-based channel index. NUM_DATABLOCK_CHANNELS is 
            /// 		  reserved to implement end() sentinal. NUM_DATABLOCK_CHANNELS+1
            /// 		  is reserved to flag uninitialized iterators. </summary>
            size_t      m_nChannelIndex;
            
            /// <summary> The iterated block. Should be non-null, except in
            /// 		  ChannelIterators created to serve as end/invalid 
            /// 		  sentinal values. </summary>
            Datablock * m_pIteratedBlock;

            ///-------------------------------------------------------------------------------------------------
            /// <summary>   Advance to next valid channel, if the current channel index has no valid data,
            ///             and there are still more channels that might have data.
            ///             </summary>
            ///
            /// <remarks>   Crossbac, 1/3/2012. </remarks>
            ///-------------------------------------------------------------------------------------------------

            void AdvanceToNextValidChannel() {
                if(m_nChannelIndex >= NUM_DATABLOCK_CHANNELS)
                    return;
                if(m_pIteratedBlock == NULL) {
                    m_nChannelIndex++;
                    return;
                }
                while(m_nChannelIndex < (size_t) NUM_DATABLOCK_CHANNELS && 
                    !m_pIteratedBlock->HasValidChannel((UINT)m_nChannelIndex)) {
                    m_nChannelIndex++;
                }
            }

            friend class Datablock;

        } ChannelIterator;

    private:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Datablock has a great many constructors but most initialization is the same while
        ///             the specialization applies to just a handful of members. This method handles the
        ///             intersection of member initializations that are shared by all constuctors.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DefaultInitialize();                

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 12/19/2011. Refactored crossbac 7/9/12 to handle differing init
        ///             data/template dimensions according to caller's needs.
        ///             </remarks>
        ///
        /// <param name="pAsyncContext">        [in] If non-null, an async context, which will wind up
        ///                                     using this block. </param>
        /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
        /// <param name="eFlags">               [in] buffer access flags. </param>
        /// <param name="luiBlockControlCode">  [in] a block control code. </param>
        /// <param name="pInitialData">         [in] If non-null, initial data. </param>
        /// <param name="bForceInitDataSize">   Size of the force initialise data. </param>
        ///-------------------------------------------------------------------------------------------------

        Datablock(
            __in AsyncContext *         pAsyncContext,
            __in DatablockTemplate *    pTemplate, 
            __in BUFFERACCESSFLAGS      eFlags,
            __in CONTROLSIGNAL          luiBlockControlCode,
            __in HOSTMEMORYEXTENT *     pInitialData,
            __in BOOL                   bForceInitDataSize
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 12/19/2011. Refactored crossbac 7/9/12 to handle differing init
        ///             data/template dimensions according to caller's needs.
        ///             </remarks>
        ///
        /// <param name="vAsyncContexts">           [in] If non-null, an async context, which will wind
        ///                                         up using this block. </param>
        /// <param name="pTemplate">                [in] If non-null, the datablock template. </param>
        /// <param name="eFlags">                   [in] buffer access flags. </param>
        /// <param name="luiBlockControlCode">      [in] a block control code. </param>
        /// <param name="pInitialData">             [in] If non-null, initial data. </param>
        /// <param name="bForceInitDataSize">       Size of the force initialise data. </param>
        /// <param name="bCreateDeviceBuffers">     The materialize all. </param>
        /// <param name="bMaterializeDeviceViews">  The materialize device views. </param>
        /// <param name="bPageLockHostViews">       The page lock host views. </param>
        ///-------------------------------------------------------------------------------------------------

        Datablock(
            __in std::set<AsyncContext*>&    vAsyncContexts,
            __in DatablockTemplate *         pTemplate, 
            __in BUFFERACCESSFLAGS           eFlags,
            __in CONTROLSIGNAL               luiBlockControlCode,
            __in HOSTMEMORYEXTENT *          pInitialData,
            __in BOOL                        bForceInitDataSize,
            __in BOOL                        bCreateDeviceBuffers,
            __in BOOL                        bMaterializeDeviceViews,
            __in BOOL                        bPageLockHostViews
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAsyncContext">        [in] If non-null, the datablock template. </param>
        /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
        /// <param name="uiDataBufferSize">     Size of the data buffer. </param>
        /// <param name="uiMetaBufferSize">     Size of the meta data buffer. </param>
        /// <param name="uiTemplateBufferSize"> Size of the template buffer. </param>
        /// <param name="eFlags">               The flags. </param>
        /// <param name="uiBlockControlCode">   (optional) a block control code. </param>
        /// <param name="pInitialData">         [in,out] (optional) buffer access flags. </param>
        /// <param name="uiRecordCount">        Number of records. </param>
        /// <param name="bFinalize">            The finalize. </param>
        ///-------------------------------------------------------------------------------------------------

        Datablock(
            __in AsyncContext *         pAsyncContext,
            __in DatablockTemplate *    pTemplate, 
            __in UINT                   uiDataBufferSize, 
            __in UINT                   uiMetaBufferSize, 
            __in UINT                   uiTemplateBufferSize,
            __in BUFFERACCESSFLAGS      eFlags,
            __in CONTROLSIGNAL          luiBlockControlCode,
            __in HOSTMEMORYEXTENT *     pInitialData,
            __in UINT                   uiRecordCount,
            __in BOOL                   bFinalize
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="luiBlockControlCode">  (optional) a block control code. </param>
        ///-------------------------------------------------------------------------------------------------

        Datablock(
            __in CONTROLSIGNAL luiBlockControlCode
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Copy constructor. Make it private so that external attempts to clone have to go
        ///             explicitly through the static clone method.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pClonedBlock">     [in,out] If non-null, the cloned block. </param>
        /// <param name="pSrcAsyncContext"> [in,out] If non-null, context for the source asynchronous. </param>
        /// <param name="pDstAsyncContext"> [in,out] If non-null, context for the destination
        ///                                 asynchronous. </param>
        ///-------------------------------------------------------------------------------------------------

        Datablock(
            __in Datablock * pClonedBlock,
            __in AsyncContext * pSrcAsyncContext,
            __in AsyncContext * pDstAsyncContext
            );

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Initializes the clone. </summary>
		///
		/// <remarks>	crossbac, 8/14/2013. </remarks>
		///
		/// <param name="pClonedBlock">	   	[in,out] If non-null, the cloned block. </param>
		/// <param name="pSrcAsyncContext">	[in,out] If non-null, context for the source asynchronous. </param>
		/// <param name="pDstAsyncContext">	[in,out] If non-null, context for the destination
		/// 								asynchronous. </param>
		///-------------------------------------------------------------------------------------------------

		void 
		InitializeClone(
            __in Datablock * pClonedBlock,
            __in AsyncContext * pSrcAsyncContext,
            __in AsyncContext * pDstAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize view(s) of the initial data in whatever memory space is most
        ///             appropriate / possible.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/10/2012. </remarks>
        ///
        /// <param name="pTargetAccelerator">   [in,out] If non-null, target accelerator. </param>
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="pInitialData">         [in,out] If non-null, information describing the lpv
        ///                                     initialise. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL MaterializeViews(
            __in Accelerator * pTargetAccelerator,
            __in AsyncContext * pAsyncContext,
            __in HOSTMEMORYEXTENT * pInitialData
        );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Materialize view(s) of the initial data in whatever memory spaces are most
        ///             appropriate / possible. If there are multiple devices, refer to the materialize
        ///             all flag to decide whether to create device views. If we cannot create device-
        ///             side buffers, based on the context (many devices) and the flags (materialize all
        ///             flag is false), then create a pinned host view to ensure that subsequent async
        ///             APIs can use the block. If we can create the device side view, refer to the pin
        ///             flag to decide whether or not to pin a host view.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/10/2012. </remarks>
        ///
        /// <param name="vAsyncContexts">           [in,out] If non-null, target accelerator. </param>
        /// <param name="pInitialData">             [in,out] If non-null, information describing the lpv
        ///                                         initialise. </param>
        /// <param name="bCreateDeviceBuffers">     The materialize all. </param>
        /// <param name="bMaterializeDeviceViews">  The materialize device views. </param>
        /// <param name="bForcePinned">             The force pinned. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL MaterializeViews(
            __in std::set<AsyncContext*>&    vAsyncContexts,
            __in HOSTMEMORYEXTENT *          pInitialData,
            __in BOOL                        bCreateDeviceBuffers,
            __in BOOL                        bMaterializeDeviceViews,
            __in BOOL                        bForcePinned
        );

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Makes a deep copy of this object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pClonedBlock">         [in,out] If non-null, the cloned block. </param>
        /// <param name="pSrcAsyncContext">     [in,out] If non-null, async op context last used with the
        ///                                     source, if known. </param>
        /// <param name="pDestAsyncContext">    [in,out] If non-null, async op context most likely used
        ///                                     to process the cloned block, if known. </param>
        ///
        /// <returns>   null if it fails, else a copy of this object. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        Clone(
            __in Datablock * pClonedBlock,
            __in AsyncContext * pSrcAsyncContext,
            __in AsyncContext * pDestAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a size descriptor block. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    If non-null, context for the asynchronous. </param>
        /// <param name="pDescribedBlock">  [in] block whose size we are describing. </param>
        /// <param name="pChannelTemplate"> [in,out] If non-null, the channel template. </param>
        /// <param name="uiValueSize">      Size of the value. </param>
        ///
        /// <returns>   null if it fails, else a new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateSizeDescriptorBlock(
            __in AsyncContext * pAsyncContext,
            __in Datablock * pDescribedBlock,
            __in UINT uiValueSize=4
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a size descriptor block. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <returns>   null if it fails, else a new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * CreateEmptySizeDescriptorBlock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a buffer dims descriptor block. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    If non-null, context for the asynchronous. </param>
        /// <param name="pDescribedBlock">  [in] block whose size we are describing. </param>
        /// <param name="pChannelTemplate"> [in,out] If non-null, the channel template. </param>
        ///
        /// <returns>   null if it fails, else a new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateBufferDimensionsDescriptorBlock(
            __in AsyncContext * pAsyncContext,
            __in Datablock * pDescribedBlock,
            __in DatablockTemplate * pChannelTemplate
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a block containing control information, either a control code or
        ///             information about whether a control code matched a given predicate.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pAsyncContext">            If non-null, context for the asynchronous. </param>
        /// <param name="luiControlInformation">    Information describing the control. </param>
        /// <param name="uiValueSize">              Size of the value. </param>
        ///
        /// <returns>   null if it fails, else a new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock *
        CreateControlInformationBlock(
            __in AsyncContext * pAsyncContext,
            __in CONTROLSIGNAL  luiControlInformation,
            __in UINT           uiValueSize=4
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a block containing control information, either a control code or
        ///             information about whether a control code matched a given predicate.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pAsyncContext">        If non-null, context for the asynchronous. </param>
        /// <param name="uiBlockIdentifier">    Information describing the control. </param>
        /// <param name="uiValueSize">          Size of the value. </param>
        ///
        /// <returns>   null if it fails, else a new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock *
        CreateUniqueIdentifierBlock(
            __in AsyncContext * pAsyncContext,
            __in UINT           uiBlockIdentifier,
            __in UINT           uiValueSize=4
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a new datablock based on the initial value in the datablock template.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for asynchronous ops. </param>
        /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
        /// <param name="eFlags">           The flags. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateInitialValueBlock(
            __in AsyncContext * pAsyncContext,
            __in DatablockTemplate * pTemplate,
            __in BUFFERACCESSFLAGS eFlags
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a new datablock based on the initial value in the datablock template.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for asynchronous ops. </param>
        /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
        /// <param name="uiMaxSizeBytes">   The maximum number of records. Used to enforce an upper bound
        ///                                 if the initial value size should be smaller than the size
        ///                                 specified in the template parameter. </param>
        /// <param name="eFlags">           The flags. </param>
        /// <param name="bPooledBlock">     The pooled block. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateInitialValueBlock(
            __in AsyncContext * pAsyncContext,
            __in DatablockTemplate * pTemplate,
            __in UINT uiMaxSizeBytes,
            __in BUFFERACCESSFLAGS eFlags,
            __in BOOL bPooledBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a new datablock for an output port. Uses the initial value if possible.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        /// <param name="pAsyncContext">        [in,out] If non-null, context for asynchronous ops. </param>
        /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
        /// <param name="cbDestSize">           Size of the destination. </param>
        /// <param name="eFlags">               The flags. </param>
        /// <param name="bPooledBlock">         true if this will be a pooled block. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateDestinationBlock(
            __in Accelerator * pDispatchAccelerator,
            __in AsyncContext * pAsyncContext,
            __in DatablockTemplate * pTemplate,
            __in UINT cbDestSize,
            __in BUFFERACCESSFLAGS eFlags,
            __in BOOL bPooledBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a new datablock for an output port. Use the initial value if possible.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="pDispatchAccelerator">     [in,out] If non-null, the dispatch accelerator. </param>
        /// <param name="pAsyncContext">            [in,out] If non-null, context for asynchronous ops. </param>
        /// <param name="pDataTemplate">            [in,out] If non-null, the template. </param>
        /// <param name="pMetaChannelTemplate">     [in,out] If non-null, the meta channel template. </param>
        /// <param name="pTemplateChannelTemplate"> [in,out] If non-null, the template channel template. </param>
        /// <param name="uiRecordCount">            Number of records. </param>
        /// <param name="cbDestSizeBytes">          Size of the data channel. </param>
        /// <param name="cbMetaSizeBytes">          Size of the meta data channel. </param>
        /// <param name="cbTemplateSizeBytes">      Size of the template data channel. </param>
        /// <param name="eFlags">                   The flags. </param>
        /// <param name="bPooledBlock">             The pooled block. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateDestinationBlock(
            __in Accelerator * pDispatchAccelerator,
            __in AsyncContext * pAsyncContext,
            __in DatablockTemplate * pDataTemplate,
            __in DatablockTemplate * pMetaChannelTemplate,
            __in DatablockTemplate * pTemplateChannelTemplate,
            __in UINT uiRecordCount,
            __in UINT cbDestSizeBytes,
            __in UINT cbMetaSizeBytes,
            __in UINT cbTemplateSizeBytes,
            __in BUFFERACCESSFLAGS eFlags,
            __in BOOL bPooledBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Create a new datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAsyncContext">        (optional) [in] If non-null, an async context, which will
        ///                                     wind up using this block. </param>
        /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
        /// <param name="pInitialData">         (optional) [in] If non-null, initial data. </param>
        /// <param name="flags">                (optional) [in] buffer access flags. </param>
        /// <param name="luiBlockControlCode">  (optional) [in] a block control code. </param>
        ///
        /// <returns>   null if it fails, else the new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateDatablock(
            __in AsyncContext *         pAsyncContext,
            __in DatablockTemplate *    pTemplate, 
            __in HOSTMEMORYEXTENT *     pInitialData=NULL,
            __in BUFFERACCESSFLAGS      flags=PT_ACCESS_HOST_WRITE,
            __in CONTROLSIGNAL          luiBlockControlCode=DBCTLC_NONE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Create a new datablock, ensuring that we enable
        ///             subsequent use with backend async APIs. If we can create a device-side
        ///             view right off, do so. If not, be sure to allocate host-side views
        ///             with pinned memory. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="vAsyncContexts">       (optional) [in] If non-null, an async context, which will
        ///                                     wind up using this block. </param>
        /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
        /// <param name="pInitialData">         (optional) [in] If non-null, initial data. </param>
        /// <param name="flags">                (optional) [in] buffer access flags. </param>
        /// <param name="luiBlockControlCode">  (optional) [in] a block control code. </param>
        /// <param name="bMaterializeAll">      The materialize all. </param>
        ///
        /// <returns>   null if it fails, else the new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateDatablockAsync(
            __in std::set<AsyncContext*>&     vAsyncContexts,
            __in DatablockTemplate *          pTemplate, 
            __in HOSTMEMORYEXTENT *           pInitialData,
            __in BUFFERACCESSFLAGS            flags,
            __in CONTROLSIGNAL                luiBlockControlCode,
            __in BOOL                         bCreateDeviceBuffers,
            __in BOOL                         bMaterializeViews,
            __in BOOL                         bPageLockHostViews
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Create a new datablock, ensuring that we enable subsequent use with backend async
        ///             APIs. If we can create a device-side view right off, do so. If not, be sure to
        ///             allocate host-side views with pinned memory.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="vAccelerators">        (optional) [in] If non-null, an async context, which will
        ///                                     wind up using this block. </param>
        /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
        /// <param name="pInitialData">         (optional) [in] If non-null, initial data. </param>
        /// <param name="flags">                (optional) [in] buffer access flags. </param>
        /// <param name="luiBlockControlCode">  (optional) [in] a block control code. </param>
        /// <param name="bCreateDeviceBuffers"> The materialize all. </param>
        /// <param name="bMaterializeViews">    The materialize views. </param>
        /// <param name="bPageLockHostViews">   (optional) the page lock host views. </param>
        ///
        /// <returns>   null if it fails, else the new block. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateDatablockAsync(
            __in std::set<Accelerator*>&      vAccelerators,
            __in DatablockTemplate *          pTemplate, 
            __in HOSTMEMORYEXTENT *           pInitialData,
            __in BUFFERACCESSFLAGS            flags,
            __in CONTROLSIGNAL                luiBlockControlCode,
            __in BOOL                         bCreateDeviceBuffers,
            __in BOOL                         bMaterializeViews,
            __in BOOL                         bPageLockHostViews
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Create a new datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
        /// <param name="uiDataBufferSize">     Size of the data buffer. </param>
        /// <param name="uiMetaBufferSize">     Size of the meta data buffer. </param>
        /// <param name="uiTemplateBufferSize"> Size of the template buffer. </param>
        /// <param name="flags">                (optional) buffer access flags. </param>
        /// <param name="luiBlockControlCode">  (optional) a block control code. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateDatablock(
            __in DatablockTemplate * pTemplate, 
            __in UINT                uiDataBufferSize, 
            __in UINT                uiMetaBufferSize, 
            __in UINT                uiTemplateBufferSize,
            __in BUFFERACCESSFLAGS   flags=PT_ACCESS_HOST_WRITE,
            __in CONTROLSIGNAL       luiBlockControlCode=DBCTLC_NONE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Create a new datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] (optional)  If non-null, context for the
        ///                                     asynchronous. </param>
        /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
        /// <param name="uiDataBufferSize">     Size of the data buffer. </param>
        /// <param name="uiMetaBufferSize">     Size of the meta data buffer. </param>
        /// <param name="uiTemplateBufferSize"> Size of the template buffer. </param>
        /// <param name="eFlags">               The flags. </param>
        /// <param name="luiBlockControlCode">  (optional) a block control code. </param>
        /// <param name="pInitialData">         [in,out] (optional) buffer access flags. </param>
        /// <param name="uiInitDataRecords">    The initialise data records. </param>
        /// <param name="bFinalize">            The finalize. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock * 
        CreateDatablock(
            __in AsyncContext *         pAsyncContext,
            __in DatablockTemplate *    pTemplate, 
            __in UINT                   uiDataBufferSize, 
            __in UINT                   uiMetaBufferSize, 
            __in UINT                   uiTemplateBufferSize,
            __in BUFFERACCESSFLAGS      eFlags,
            __in CONTROLSIGNAL          luiBlockControlCode,
            __in HOSTMEMORYEXTENT *     pInitialData,
            __in UINT                   uiInitDataRecords,
            __in BOOL                   bFinalize
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a control block. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <param name="luiBlockControlCode">  (optional) [in] a block control code. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        static Datablock *
        CreateControlBlock( 
            __in CONTROLSIGNAL luiBlockControlCode
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Datablock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Set application context associated with this datablock.
        ///
        ///             The context is initially NULL when a datablock is created, and is
        ///             reinitialized to NULL each time it is returned to a block pool.
        ///
        ///             The context is copied to any clones of the datablock.
        ///             </summary>
        ///
        /// <remarks>   jcurrey, 3/21/2014. </remarks>
        ///
        /// <param name="pApplicationContext">   The context. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetApplicationContext(
            __in void * pApplicationContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Get application context that the application previously has associated with 
        ///             this datablock.
        ///
        ///             The context is initially NULL when a datablock is created, and is
        ///             reinitialized to NULL each time it is returned to a block pool.
        ///
        ///             The context is copied to any clones of the datablock.
        ///             </summary>
        ///
        /// <remarks>   jcurrey, 3/21/2014. </remarks>
        ///
        /// <returns>   The context. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * 
        GetApplicationContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Invalidate any device views, updating the host view and releasing any physical
        ///             buffers, according to the flags. Return the number of bytes freed.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/8/2013. Use this method with caution--this is really intended as a GC-
        ///             facing API, but I am making it public as a temporary remedy for what looks like a
        ///             Dandelion memory leak. Once blocks are consumed by the DandelionBinaryReader, we
        ///             know them to be dead, but they do not always have non-zero refcount. Since we
        ///             know that the host view was materialized, we can usually safely just reclaim the
        ///             device buffers and elide the step of syncing the host because the host was synced
        ///             before the deserialization.
        ///             </remarks>
        ///
        /// <param name="bSynchronizeHostView"> true to synchronize host view. </param>
        /// <param name="bRelinquishBuffers">   true to release backing buffers. </param>
        ///
        /// <returns>   the number of bytes released. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual size_t 
        InvalidateDeviceViews(
            BOOL bSynchronizeHostView,
            BOOL bRelinquishBuffers
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the block has outstanding async operations on the buffer for the target
        ///             memory space that need to resolve before an operation of the given type can be
        ///             performed.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///
        /// <param name="pTargetAccelerator">   [in,out] If non-null, the accelerator. </param>
        /// <param name="eOpType">              Type of the operation. </param>
        ///
        /// <returns>   true if outstanding dependence, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        HasOutstandingAsyncDependences(
            __in Accelerator * pTargetAccelerator,
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the set of accelerators for which there are outstanding async operations in
        ///             the target memory space that need to complete before new operations in the target
        ///             memory space can begin. Note that this version returns a pointer to a member data
        ///             structure, so callers should use it with caution. In particular, this is
        ///             *NOT* a good way to manage lock acquire/release lists for accelerators, since the
        ///             list may change between the lock acquire phase and release phase if outstanding
        ///             dependences are resolved while those locks are held. For such a purpose, the
        ///             GetCurrentOutstandingAsyncAccelerators API (which returns a copy) should be used
        ///             instead.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/26/2013. </remarks>
        ///
        /// <param name="pTargetAccelerator">   [in,out] The first parameter. </param>
        /// <param name="eOpType">              Type of the operation. </param>
        ///
        /// <returns>   null if it fails, else the outstanding asynchronous accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::set<Accelerator*>*
        GetOutstandingAsyncAcceleratorsPointer(
            __in Accelerator * pTargetAccelerator,
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the block has outstanding async operations on the buffer for the target
        ///             memory space that need to resolve before an operation of the given type can be
        ///             performed.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/25/2013. </remarks>
        ///
        /// <param name="uiTargetMemorySpaceId">    If non-null, the target memory space. </param>
        /// <param name="eOpType">                  Type of the operation. </param>
        ///
        /// <returns>   true if outstanding dependence, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        HasOutstandingAsyncDependences(
            __in UINT uiTargetMemorySpaceId,
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the set of accelerators for which there are outstanding async operations in
        ///             the target memory space that need to complete before new operations in the target
        ///             memory space can begin. Note that this version returns a pointer to a member data
        ///             structure, so callers should use it with caution. In particular, this is
        ///             *NOT* a good way to manage lock acquire/release lists for accelerators, since the
        ///             list may change between the lock acquire phase and release phase if outstanding
        ///             dependences are resolved while those locks are held. For such a purpose, the
        ///             GetCurrentOutstandingAsyncAccelerators API (which returns a copy) should be used
        ///             instead.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/26/2013. </remarks>
        ///
        /// <param name="uiTargetMemorySpaceId">    The target memory space. </param>
        /// <param name="eOpType">                  Type of the operation. </param>
        ///
        /// <returns>   null if it fails, else the outstanding asynchronous accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::set<Accelerator*>*
        GetOutstandingAsyncAcceleratorsPointer(
            __in UINT uiTargetMemorySpaceId,
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the set of accelerators for which there are outstanding async operations in
        ///             the target memory space that need to complete before new operations in the target
        ///             memory space can begin.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/26/2013. </remarks>
        ///
        /// <param name="uiTargetMemorySpaceId">    The target memory space. </param>
        /// <param name="eOpType">                  Type of the operation. </param>
        /// <param name="vAccelerators">            [out] the accelerators. </param>
        ///
        /// <returns>   the number of outstanding accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT
        GetCurrentOutstandingAsyncAccelerators(
            __in  UINT uiTargetMemorySpaceId,
            __in  ASYNCHRONOUS_OPTYPE eOpType,
            __out std::set<Accelerator*>& vAccelerators
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Lockless wait outstanding: without acquiring any locks attempt to perform a
        ///             synchronous wait for any outstanding async dependences on this block that
        ///             conflict with an operation of the given type on the given target accelerator to
        ///             complete. This is an experimental API, enable/disable with
        ///             PTask::Runtime::*etTaskDispatchLocklessIncomingDepWait(), attempting to leverage
        ///             the fact that CUDA apis for waiting on events (which appear to be thread-safe and
        ///             decoupled from a particular device context)
        ///             to minimize serialization associated with outstanding dependences on blocks
        ///             consumed by tasks that do not require accelerators for any other reason than to
        ///             wait for such operations to complete.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/1/2013. </remarks>
        ///
        /// <param name="pTargetAccelerator">   [in,out] If non-null, target accelerator. </param>
        /// <param name="eOpType">              Type of the operation. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        LocklessWaitOutstanding(
            __in Accelerator * pTargetAccelerator, 
            __in ASYNCHRONOUS_OPTYPE eOpType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this datablock has accelerator buffers
        /// 			already created for the given accelerator. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in] non-null, an accelerator. </param>
        ///
        /// <returns>   true if accelerator buffers, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasBuffers(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this datablock has buffers already created for the given
        ///             memory space.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="uiMemorySpaceID">  (optional) [in] non-null, an accelerator. </param>
        ///
        /// <returns>   true if accelerator buffers, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasBuffers(UINT uiMemorySpaceID);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this datablock has accelerator buffers
        /// 			already created for any memory space other than the host
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in] non-null, an accelerator. </param>
        ///
        /// <returns>   true if accelerator buffers, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasDeviceBuffers();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queries if the datablock logically is empty. This check is required because we
        ///             must often allocate a non-zero size object in a given memory space to represent
        ///             one whose logical size is zero bytes, so that we have an object that can be bound
        ///             to kernel execution resources (parameters, globals). We need to be able to tell
        ///             when a sealed block has non-zero size, but has a record count of zero so that
        ///             descriptor ports and metaports can do the right thing.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2012. </remarks>
        ///
        /// <returns>   true if the block is logically is empty, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsLogicallyEmpty();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks the block and if the lock is being acquired for a view synchronization,
        ///             lock the most recent accelerator and any view sync target provided. This allows
        ///             us to order lock acquire correctly (accelerators before datablocks)
        ///             when deciding whether to produce downstream views after dispatch.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="bLockForViewSynchronization">  The lock for view synchronization. </param>
        /// <param name="pTargetAccelerator">           (optional) [in,out] If non-null, target
        ///                                             accelerator. </param>
        ///
        /// <returns>   the most recent accelerator, locked, if one exists. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator * 
        LockForViewSync(
            __in BOOL bLockForViewSynchronization,
            __in Accelerator * pTargetAccelerator=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the block and if the lock was acquired for a view synchronization, unlock
        ///             the most recent accelerator and any view sync target provided. This allows us to
        ///             order lock acquire correctly (accelerators before datablocks)
        ///             when deciding whether to produce downstream views after dispatch.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="bLockForViewSynchronization">  The lock for view synchronization. </param>
        /// <param name="pMostRecentAccelerator">       [in,out] If non-null, the most recent accelerator
        ///                                             returned by the corresponding lock call. </param>
        /// <param name="pTargetAccelerator">           (optional) [in,out] If non-null, target
        ///                                             accelerator. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        UnlockForViewSync(
            __in BOOL bLockForViewSynchronization,
            __in Accelerator * pMostRecentAccelerator,
            __in Accelerator * pTargetAccelerator=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the accelerator's view of this block. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerator">             [in,out] If non-null, the accelerator. </param>
        /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="bPopulateView">            true if the caller wants the accelerator-side buffer
        ///                                         to have the most recent data contents for the block. Many
        ///                                         blocks are bound as outputs where the device-side code
        ///                                         performs only writes to the data, meaning we should not
        ///                                         attempt to transfer a more recent view before binding the
        ///                                         block and dispatching. </param>
        /// <param name="uiRequestedState">         The requested coherence state. This affects whether
        ///                                         other accelerator views require invalidation. </param>
        /// <param name="uiMinChannelBufferIdx">    Zero-based index of the minimum channel buffer. </param>
        /// <param name="uiMaxChannelBufferIdx">    Zero-based index of the maximum channel buffer. </param>
        ///
        /// <returns>   PTRESULT. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        UpdateView(
            __in Accelerator * pAccelerator,
            __in AsyncContext * pAsyncContext,
            __in BOOL bPopulateView, 
            __in BUFFER_COHERENCE_STATE uiRequestedState,
            __in UINT uiMinChannelBufferIdx,
            __in UINT uiMaxChannelBufferIdx
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the host memspace view of this block. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiRequestedState">     The requested coherence state. This affects whether other
        ///                                     accelerator views require invalidation. </param>
        /// <param name="bForceSynchronous">    (optional) the force synchronous. </param>
        /// <param name="bAcquireLocks">        The acquire locks. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        SynchronizeHostView(
            __in AsyncContext * pAsyncContext,
            __in BUFFER_COHERENCE_STATE uiRequestedState,
            __in BOOL bForceSynchronous,
            __in BOOL bAcquireLocks
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the block has any valid channels at all.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <returns>   true if valid channel, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasValidChannels();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the ChannelIterator has a valid channel in *any* memory space for this
        ///             datablock (at the channel corresponding to its current iteration state).
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="vIterator">    [in] channel iterator. </param>
        ///
        /// <returns>   true if valid channel, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasValidChannel(ChannelIterator &vIterator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'nChannelIndex' has valid channel in *any* memory space
        /// 			for this datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="nChannelIndex">    Zero-based index of the channel. </param>
        ///
        /// <returns>   true if valid channel, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasValidChannel(UINT nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the given block is out of date on the dispatch accelerator.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL    
        RequiresViewUpdate(
            Accelerator * pDispatchAccelerator,
            BUFFER_COHERENCE_STATE uiRequiredPermissions
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the given block is out of date in the given memory space.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL    
        RequiresViewUpdate(
            UINT uiMemorySpaceID,
            BUFFER_COHERENCE_STATE uiRequiredPermissions
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if the given block is out of date on the dispatch accelerator *and*
        ///             up-to-date somewhere other than in host memory.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL    
        RequiresMigration(
            Accelerator * pDispatchAccelerator
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerator whose memory space contains an up-to-date
        /// 			view of this datablock. If the host memory space has an up-to-date
        /// 			view, return NULL.</summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the most recent accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        Accelerator *   GetMostRecentAccelerator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the AsyncContext which touched this block most recently (if any).
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the most recent accelerator. </returns>
        ///-------------------------------------------------------------------------------------------------

        AsyncContext *   GetMostRecentAsyncContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the accelerators whose memory space contains the most up-to-date
        /// 			view of this datablock. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <returns>   number of valid view accelerators. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT           GetValidViewAccelerators(std::vector<Accelerator*> &accs);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Propagate any control information in this
        /// 			datablock if the port is part of the
        /// 			control routing network, and the block
        /// 			contains control signals. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. 
        /// 			You must hold the lock on the block to call this.</remarks>
        ///
        /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
        ///-------------------------------------------------------------------------------------------------

        void PropagateControlInformation(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Create all the buffers required for a given block to be referenced on the given
        ///             accelerator. This entails creating the data channel, and if metadata and template
        ///             channels are specified in the datablock template, then those device-specific
        ///             buffers need to be created as well. If the user has supplied initial data,
        ///             populate the accelerator- side buffers with the initial data, and mark the block
        ///             coherent on the accelerator.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="pAccelerator">             [in] If non-null, an accelerator that will require a
        ///                                         view of this block. </param>
        /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="bPopulate">                (optional) if true, populate the device buffer with
        ///                                         data from the host-side buffer. </param>
        /// <param name="uiMinChannelBufferIdx">    (Optional) zero-based index of the minimum channel
        ///                                         buffer. </param>
        /// <param name="uiMaxChannelBufferIdx">    (Optional) zero-based index of the maximum channel
        ///                                         buffer. </param>
        ///
        /// <returns>   PTRESULT: PTASK_OK on success PTASK_EXISTS if called on a block with materialized
        ///             buffers.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        AllocateBuffers(
            __in Accelerator * pAccelerator, 
            __in AsyncContext * pAsyncContext,
            __in BOOL bPopulate=TRUE,
            __in UINT uiMinChannelBufferIdx=DBDATA_IDX,
            __in UINT uiMaxChannelBufferIdx=DBTEMPLATE_IDX
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Create all the buffers required for a given block to be referenced on the given
        ///             accelerator. This entails creating the data channel, and if metadata and template
        ///             channels are specified in the datablock template, then those device-specific
        ///             buffers need to be created as well. If the user has supplied initial data,
        ///             populate the accelerator- side buffers with the initial data, and mark the block
        ///             coherent on the accelerator.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="uiMemorySpaceID">  (optional) [in] If non-null, an accelerator that will require
        ///                                 a view of this block. </param>
        /// <param name="pAsyncContext">    If non-null, context for the asynchronous. </param>
        /// <param name="bPopulate">        (optional) if true, populate the device buffer with data from
        ///                                 the host-side buffer. </param>
        ///
        /// <returns>   PTRESULT: PTASK_OK on success PTASK_EXISTS if called on a block with materialized
        ///             buffers.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        AllocateBuffers(
            __in UINT uiMemorySpaceID,
            __in AsyncContext * pAsyncContext,
            __in BOOL bPopulate=TRUE            
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Gets a pointer to host memory containing the most recent version of the data for this
        ///     block.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pBufferMap">       [in] non-null, the buffer map. </param>
        /// <param name="pCoherenceMap">    [in] non-null, the coherence map. </param>
        /// <param name="pLastAccelerator"> [in] non-null, the last accelerator. </param>
        /// <param name="ppHostBuffer">     [in] non-null, buffer for host data. </param>
        /// <param name="lpfnSize">         [in] non-null, a function to estimate size. </param>
        ///
        /// <returns>   null if it fails, else the pointer. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * 
        GetPointer(
            std::map<Accelerator*, PBuffer *>* pBufferMap, 
            std::map<Accelerator*, BOOL>* pCoherenceMap,         
            Accelerator * pLastAccelerator,
            void ** ppHostBuffer, 
            UINT (*lpfnSize)(), 
            BOOL bWritable
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a pointer to host memory containing the most recent version of the specified
        ///             channel buffer for this block.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiChannelIndex">               Zero-based index of the channel. </param>
        /// <param name="bWritable">                    true if a writable version is required, which
        ///                                             necessitates invalidating all other views of this
        ///                                             block. </param>
        /// <param name="bSynchronizeBufferContents">   (optional) the update view. </param>
        ///
        /// <returns>   null if it fails, else the data pointer. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * 
        GetChannelBufferPointer(
            __in UINT uiChannelIndex,
            __in BOOL bWritable,
            __in BOOL bSynchronizeBufferContents=TRUE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a pointer to host memory containing the most recent version of the data for
        ///             this block.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="bWritable">                    true if a writable version is required, which
        ///                                             necessitates invalidating all other views of this
        ///                                             block. </param>
        /// <param name="bSynchronizeBufferContents">   (optional) the update view. </param>
        ///
        /// <returns>   null if it fails, else the data pointer. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * 
        GetDataPointer(
            __in BOOL bWritable,
            __in BOOL bSynchronizeBufferContents=TRUE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a pointer to host memory containing the most recent version of the meta data
        ///             for this block.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="bWritable">                    true if a writable version is required, which
        ///                                             necessitates invalidating all other views of this
        ///                                             block. </param>
        /// <param name="bSynchronizeBufferContents">   (optional) the update buffer contents. </param>
        ///
        /// <returns>   null if it fails, else the meta data pointer. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * 
        GetMetadataPointer(
            __in BOOL bWritable,
            __in BOOL bSynchronizeBufferContents=TRUE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a pointer to host memory containing the most recent version of the template
        ///             data for this block.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="bWritable">                    true if a writable version is required, which
        ///                                             necessitates invalidating all other views of this
        ///                                             block. </param>
        /// <param name="bSynchronizeBufferContents">   (optional) the update buffer contents. </param>
        ///
        /// <returns>   null if it fails, else the template data pointer. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * 
        GetTemplatePointer(
            __in BOOL bWritable,
            __in BOOL bSynchronizeBufferContents=TRUE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the actual dimensions of a datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="dim">  The dim. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * SetActualDimensions(unsigned int dim[MAX_DATABLOCK_DIMENSIONS]);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pAccelerator' is coherent on that accelerator
        /// 			Meaning does that accelerator have a copy of the data that is
        /// 			up-to-date. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in] non-null, an accelerator.</param>
        ///
        /// <returns>   true if coherent, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsCoherent(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a platform-specific accelerator data buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAccelerator">     (optional) [in] non-null, the accelerator in question. </param>
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        /// <param name="pRequester">       (optional) [in,out] If non-null, the requesting accelerator,
        ///                                 which we may want to use to allocate host-side memory if it has a
        ///                                 specialized allocator. </param>
        ///
        /// <returns>   null if it fails, else the accelerator data buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        PBuffer * GetPlatformBuffer(Accelerator * pAccelerator, 
                                    UINT nChannelIndex,
                                    Accelerator * pRequester=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a platform-specific accelerator data buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAccelerator">     (optional) [in] non-null, the accelerator in question. </param>
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        /// <param name="pRequester">       (optional) [in,out] If non-null, the requesting accelerator,
        ///                                 which we may want to use to allocate host-side memory if it has a
        ///                                 specialized allocator. </param>
        ///
        /// <returns>   null if it fails, else the accelerator data buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        PBuffer * GetPlatformBuffer(Accelerator * pAccelerator, 
                                    ChannelIterator &vChannelIterator,
                                    Accelerator * pRequester=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a platform-specific accelerator data buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="nMemorySpaceId">   (optional) [in] non-null, the accelerator in question. </param>
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        /// <param name="pRequester">       (optional) [in,out] If non-null, the requester. </param>
        ///
        /// <returns>   null if it fails, else the accelerator data buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        PBuffer * GetPlatformBuffer(UINT nMemorySpaceId, 
                                    UINT nChannelIndex,
                                    Accelerator * pRequester=NULL);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Release by decrementing the refcount. We override the implementation inherited
        ///             from ReferenceCounted so that we can return the block to its pool rather than
        ///             deleting it, if that is appropriate.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual LONG Release();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the datablock template describing this block </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the template. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual DatablockTemplate * GetTemplate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the access flags for this block. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The access flags. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BUFFERACCESSFLAGS GetAccessFlags();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the access flags. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="e">    The access flags. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetAccessFlags(BUFFERACCESSFLAGS e);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bring views of this block up-to-date on both accelerators. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pDest">                    [in] non-null, destination accelerator. </param>
        /// <param name="pSrc">                     [in] non-null, source accelerator. </param>
        /// <param name="pAsyncContext">            [in,out] If non-null, context for the source
        ///                                         asynchronous. </param>
        /// <param name="uiRequestedPermissions">   The requested permissions for the migrated view. </param>
        ///
        /// <returns>   PTRESULT: PTASK_OK on success PTASK_ERR if migration fails. </returns>       
        ///-------------------------------------------------------------------------------------------------

        virtual PTRESULT
        Migrate(
            __inout Accelerator * pDest, 
            __in    Accelerator * pSrc,
            __inout AsyncContext * pAsyncContext,
            __in    BUFFER_COHERENCE_STATE uiRequestedPermissions
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets this block's coherence state for the given accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="pAccelerator"> (optional) [in] If non-null, an accelerator that will require a
        ///                             view of this block. </param>
        ///
        /// <returns>   The coherence state. </returns>
        ///-------------------------------------------------------------------------------------------------

        BUFFER_COHERENCE_STATE  GetCoherenceState(Accelerator * pAccelerator);

        // TODO JC Documentation.

        void ForceExistingViewsValid();

#ifdef FORCE_CORRUPT_NON_AFFINITIZED_VIEWS
        void ForceCorruptNonAffinitizedViews(Accelerator * pAcc);
#endif

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the data buffer size. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The data buffer size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetDataBufferLogicalSizeBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the data buffer size. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The data buffer size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetDataBufferAllocatedSizeBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Grow datablock channel buffers all at once. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiNewDataSizeBytes">       New size of the data buffer. </param>
        /// <param name="uiNewMetaSizeBytes">       The new meta size in bytes. </param>
        /// <param name="uiNewTemplateSizeBytes">   The new template size in bytes. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GrowBuffers(
            __in UINT uiNewDataSizeBytes, 
            __in UINT uiNewMetaSizeBytes, 
            __in UINT uiNewTemplateSizeBytes
            );     

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Grow data buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="newSize">  New size of the data buffer. </param>
        ///-------------------------------------------------------------------------------------------------

        void GrowDataBuffer(UINT newSize);                

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synthesize a metadata channel based on template information.
        ///             
        ///             This is specialized support for Dandelion:
        ///             ------------------------------------------
        ///             Dandelion expects a per-record object size entry in the meta-data channel. This
        ///             expectation is probably obsolete, since we are currently restricted to fixed-
        ///             stride objects in Dandelion. However, to accommodate that expectation, we provide
        ///             a method to populate the metadata channel of a block from its datablock template.
        ///             We simply compute the record count based on the stride and buffer sizes, and
        ///             write the num_record copies of the stride in a new metadata channel.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/26/2012. </remarks>
        ///
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        SynthesizeMetadataFromTemplate(
            __in AsyncContext * pAsyncContext
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the meta buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the meta buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        // void* GetMetaBuffer(BOOL bWriteable=TRUE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the meta buffer size. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The meta buffer size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetMetaBufferLogicalSizeBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the meta buffer size. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The meta buffer size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetMetaBufferAllocatedSizeBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Grow meta buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="newSize">  New size of the meta data buffer. </param>
        ///-------------------------------------------------------------------------------------------------

        void GrowMetaBuffer(UINT newSize);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the template buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the template buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        // void* GetTemplateBuffer(BOOL bWriteable=TRUE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the template buffer size. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The template buffer size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetTemplateBufferLogicalSizeBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the template buffer size. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The template buffer size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetTemplateBufferAllocatedSizeBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Grow template buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="newSize">  New size of the template buffer. </param>
        ///-------------------------------------------------------------------------------------------------

        void GrowTemplateBuffer(UINT newSize);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Seals a datablock by recording the number of actual bytes used in the various buffer
        ///     types, and recording the number of records written in the data channel.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiRecordCount">    Number of records. </param>
        /// <param name="uiDataSize">       Size of the data. </param>
        /// <param name="uiMetaDataSize">   Size of the meta data. </param>
        /// <param name="uiTemplateSize">   Size of the template. </param>
        ///-------------------------------------------------------------------------------------------------

        void Seal(UINT uiRecordCount, UINT uiDataSize, UINT uiMetaDataSize, UINT uiTemplateSize);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Seals a datablock by recording the number of actual bytes used in the various buffer
        ///     types, and recording the number of records written in the data channel.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiRecordCount">    Number of records. </param>
        /// <param name="uiDataSize">       Size of the data. </param>
        /// <param name="uiMetaDataSize">   Size of the meta data. </param>
        /// <param name="uiTemplateSize">   Size of the template. </param>
        ///-------------------------------------------------------------------------------------------------

        void Unseal();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has a metadata channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if metadata channel, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasMetadataChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has template channel. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if template channel, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasTemplateChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this datablock has a control token. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if control token, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsControlToken();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this block contains both control and data. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if control and data block, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsControlAndDataBlock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this block is a scalar parameter. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if scalar parameter, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsScalarParameter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this block is record stream. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if record stream, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsRecordStream();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the record count. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The record count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetRecordCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the record count. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The record count. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsResizedBlock();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Clears the resize flags. </summary>
        ///
        /// <remarks>   crossbac, 8/30/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ClearResizeFlags();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a record count. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiRecordCount">    Number of records. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetRecordCount(UINT uiRecordCount);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the parameter type. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The parameter type. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTASK_PARM_TYPE GetParameterType();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this block has a valid data buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if valid data buffer, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasValidDataBuffer();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the block control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The block control code. </returns>
        ///-------------------------------------------------------------------------------------------------

        CONTROLSIGNAL GetControlSignals();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Checks whether the block carries the given signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The block control code. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL TestControlSignal(CONTROLSIGNAL luiCode);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Checks whether the block carries the given signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The block control code. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL HasAnyControlSignal();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Checks whether the block carries the BOF signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if the block control code carries the signal. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsBOF();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Checks whether the block carries the EOF signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if the block control code carries the signal. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsEOF();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Checks whether the block carries the begin iteration signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if the block control code carries the signal. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsBOI();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Checks whether the block carries the end iteration signal. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if the block control code carries the signal. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsEOI();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   adds the signal to the block's control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="code"> The code. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID SetControlSignal(CONTROLSIGNAL code);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   removes the signal from the block's control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="code"> The code. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID ClearControlSignal(CONTROLSIGNAL code);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   removes all signal from the block's control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="code"> The code. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID ClearAllControlSignals();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pushes a new control signal context. </summary>
        ///
        /// <remarks>   Crossbac, 2/14/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID PushControlSignalContext(CONTROLSIGNAL luiCode=DBCTLC_NONE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pops the control signal context. </summary>
        ///
        /// <remarks>   Crossbac, 2/14/2013. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        CONTROLSIGNAL PopControlSignalContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the destination port of this block </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the destination port. </returns>
        ///-------------------------------------------------------------------------------------------------

        Port * GetDestinationPort();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a destination port for this block </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    (optional) [in] non-null, the accelerator in question. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetDestinationPort(Port*p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record which ptask produced this datablock as output. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="p">    (optional) [in] non-null, the Task in question. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetProducerTask(Task * p);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this datablock is part of an output port's block pool. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if pooled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsPooled();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this datablock has been sealed by a call to Seal(). </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   true if pooled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsSealed();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this block can generate device views with memset rather than 
        /// 			host-device transfers. </summary>
        ///
        /// <remarks>   crossbac, 7/6/2012. </remarks>
        ///
        /// <returns>   true if device views memsettable, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsDeviceViewsMemsettable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this datablock is supposed to be backed by byte addressable device-side
        ///             buffers, requiring special APIs to create on some platforms.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/29/2011. </remarks>
        ///
        /// <returns>   true if byte addressable, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------
 
        BOOL IsByteAddressable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the port whose pool owns this datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the pool owner. </returns>
        ///-------------------------------------------------------------------------------------------------

        BlockPoolOwner * GetPoolOwner();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Mark this block as poole or not. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetPooledBlock(BlockPoolOwner * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this datablock is marshallable. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if marshallable, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsMarshallable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Marks the datablock marshallable or not. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="bMarshallable">    true if marshallable. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetMarshallable(BOOL bMarshallable);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the stride of objects in the Datablock. If the block has a template, this
        ///             comes from the stride member of the template's dimensions. If no template is
        ///             present and the block is unsealed, assert. If no template is present, and the
        ///             datablock is sealed, return the data-channel's byte-length by the number of
        ///             records. If no template is present, and the datablock is byte- addressable,
        ///             return 1.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/29/2011. </remarks>
        ///
        /// <returns>   The stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetStride();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of objects in X dimension of the Datablock. If the block has a
        ///             template, this comes from the XDIM member of the template's dimensions. If no
        ///             template is present and the block is unsealed, assert. If no template is present,
        ///             and the datablock is sealed, return the record count. If no template is present,
        ///             and the datablock is byte- addressable, return the data-channel size.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/29/2011. </remarks>
        ///
        /// <returns>   The number of elements in X. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetXElementCount();
        
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of objects in Y dimension of the Datablock. If the block has a
        ///             template, this comes from the YDIM member of the template's dimensions. If no
        ///             template is present and the block is unsealed, assert. If no template is present,
        ///             return 1.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/29/2011. </remarks>
        ///
        /// <returns>   The number of elements in Y. </returns>
        ///-------------------------------------------------------------------------------------------------
        
        UINT GetYElementCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of objects in Z dimension of the Datablock. If the block has a
        ///             template, this comes from the ZDIM member of the template's dimensions. If no
        ///             template is present and the block is unsealed, assert. If no template is present,
        ///             return 1.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/29/2011. </remarks>
        ///
        /// <returns>   The number of elements in Z. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetZElementCount();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the allocation size for the given channel based on
        /// 			ambient information about the datablock. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        ///
        /// <returns>   The channel allocation size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetChannelAllocationSizeBytes(UINT nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the allocation size for the given channel based on
        /// 			ambient information about the datablock. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 1/6/2012. </remarks>
        ///
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        ///
        /// <returns>   The channel allocation size. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetChannelLogicalSizeBytes(UINT nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'nChannelIndex' is template size overridden. </summary>
        ///
        /// <remarks>   crossbac, 7/9/2012. </remarks>
        ///
        /// <param name="nChannelIndex">    Zero-based index of the channel. </param>
        ///
        /// <returns>   true if template size overridden, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsTemplateSizeOverridden(UINT nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes a new channel iterator which can be used
        /// 			to iterate over all valid channels in the block. </summary>
        ///
        /// <remarks>   Crossbac, 1/3/2012. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        ChannelIterator& FirstChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the end-of-iteration sentinal. </summary>
        ///
        /// <remarks>   Crossbac, 1/3/2012. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        ChannelIterator& LastChannel();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   invalidate sharers. </summary>
        ///
        /// <remarks>   Crossbac, 9/20/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL AcquireExclusive(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the datablock ID. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <returns>   The dbuid. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetDBUID();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   marks any buffers in valid state as invalid. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void Invalidate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets prefer page locked host views. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <param name="bPageLock">    true to lock, false to unlock the page. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetPreferPageLockedHostViews(BOOL bPageLock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets prefer page locked host views. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL GetPreferPageLockedHostViews();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets the initial value for pool. </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void ResetInitialValueForPool(Accelerator * pAccelerator);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets pool initial value valid. </summary>
        ///
        /// <remarks>   crossbac, 5/4/2013. </remarks>
        ///
        /// <param name="bValid">   true to valid. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetPoolInitialValueValid(BOOL bValid);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets pool initial value valid. </summary>
        ///
        /// <remarks>   crossbac, 5/4/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL GetPoolInitialValueValid();

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

        friend std::ostream& operator<<(std::ostream &os, Datablock* pBlock); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Debug dump. </summary>
        ///
        /// <remarks>   Crossbac, 7/18/2013. </remarks>
        ///
        /// <param name="ss">               [in,out] If non-null, the ss. </param>
        /// <param name="pcsSSLock">        [in,out] If non-null, the pcs ss lock. </param>
        /// <param name="szTaskLabel">      [in,out] If non-null, the task label. </param>
        /// <param name="szLabel">          [in,out] If non-null, the label. </param>
        /// <param name="uiChannelIndex">   Zero-based index of the channel. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        DebugDump(
            std::ostream* ss,
            CRITICAL_SECTION* pcsSSLock,
            char * szTaskLabel,
            char * szLabel,
            UINT uiChannelIndex=0
            );

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Set the coherence state for the given accelerator's view of this block. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="pAccelerator">     (optional) [in] If non-null, an accelerator that will require
        ///                                 a view of this block. </param>
        /// <param name="uiCoherenceState"> State of the coherence. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        SetCoherenceState(
            Accelerator * pAccelerator, 
            BUFFER_COHERENCE_STATE uiCoherenceState
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a coherence state. </summary>
        ///
        /// <remarks>   Crossbac, 12/30/2011. </remarks>
        ///
        /// <param name="nMemorySpaceId">   (optional) [in] non-null, the accelerator in question. </param>
        /// <param name="uiCoherenceState"> State of the coherence. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        SetCoherenceState(
            UINT nMemorySpaceId,
            BUFFER_COHERENCE_STATE uiCoherenceState
            );


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   adds the signal to the block's control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="code"> The code. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID __setControlSignal(CONTROLSIGNAL code);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   removes the signal from the block's control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="code"> The code. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID __clearControlSignal(CONTROLSIGNAL code);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   removes all signal from the block's control code. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="code"> The code. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID __clearAllControlSignals();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an accelerator buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAccelerator">     (optional) [in] If non-null, an accelerator that will require
        ///                                 a view of this block. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiChannelIndex">   Zero-based index of the user interface channel. </param>
        /// <param name="cbAllocationSize"> Size of the allocation. </param>
        /// <param name="pInitialData">     (optional) [in] If non-null, initial data. </param>
        ///
        /// <returns>   PTRESULT--use PTFAIL/PTSUCCEED macros to interpret. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        AllocateBuffer(
            __in Accelerator * pAccelerator, 
            __in AsyncContext * pAsyncContext,
            __in UINT uiChannelIndex, 
            __in UINT cbAllocationSize=0, 
            __in HOSTMEMORYEXTENT * pInitialData=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an accelerator buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAccelerator">     (optional) [in] If non-null, an accelerator that will require
        ///                                 a view of this block. </param>
        /// <param name="pProxyAllocator">  [in,out] If non-null, the proxy allocator. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiChannelIndex">   Zero-based index of the user interface channel. </param>
        /// <param name="cbAllocationSize"> Size of the allocation. </param>
        /// <param name="pInitialData">     (optional) [in] If non-null, initial data. </param>
        ///
        /// <returns>   PTRESULT--use PTFAIL/PTSUCCEED macros to interpret. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        AllocateBuffer(
            __in Accelerator * pAccelerator, 
            __in Accelerator * pProxyAllocator,
            __in AsyncContext * pAsyncContext,
            __in UINT uiChannelIndex, 
            __in UINT cbAllocationSize=0, 
            __in HOSTMEMORYEXTENT * pInitialData=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an accelerator buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiMemorySpaceId">  destination memory space. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiChannelIndex">   Zero-based index of the user interface channel. </param>
        /// <param name="cbAllocationSize"> Size of the allocation. </param>
        /// <param name="pInitialData">     (optional) [in] If non-null, initial data. </param>
        ///
        /// <returns>   PTRESULT--use PTFAIL/PTSUCCEED macros to interpret. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        AllocateBuffer(
            __in UINT uiMemorySpaceId, 
            __in AsyncContext * pAsyncContext,
            __in UINT uiChannelIndex, 
            __in UINT cbAllocationSize=0, 
            __in HOSTMEMORYEXTENT * pInitialData=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an accelerator buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiMemorySpaceId">  destination memory space. </param>
        /// <param name="pProxyAllocator">  [in,out] Identifier for the proxy allocator memory space. </param>
        /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="uiChannelIndex">   Zero-based index of the user interface channel. </param>
        /// <param name="cbAllocationSize"> Size of the allocation. </param>
        /// <param name="pInitialData">     (optional) [in] If non-null, initial data. </param>
        ///
        /// <returns>   PTRESULT--use PTFAIL/PTSUCCEED macros to interpret. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        AllocateBuffer(
            __in UINT uiMemorySpaceId, 
            __in Accelerator * pProxyAllocator,
            __in AsyncContext * pAsyncContext,
            __in UINT uiChannelIndex, 
            __in UINT cbAllocationSize=0, 
            __in HOSTMEMORYEXTENT * pInitialData=NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Grow buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="newSize">  New size of the data buffer. </param>
        ///-------------------------------------------------------------------------------------------------

        void GrowBuffer(UINT nChannelIndex, UINT newSize);       

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a pointer to the host memory backing the given channel. If the
        ///             bUpdateIsStale flag is set, force an update from any available shared/exclusive
        ///             copies in other memory spaces. If the request is for a writeable copy, invalidate
        ///             copies in other memory spaces.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/28/2011. </remarks>
        ///
        /// <param name="pAsyncContext">                [in,out] If non-null, context for the
        ///                                             asynchronous. </param>
        /// <param name="nChannelIndex">                The index of the requested channel. </param>
        /// <param name="uiRequestedState">             true if the caller plans to write to the block. </param>
        /// <param name="bMaterializeAbsentBuffers">    true if the runtime should update a stale host
        ///                                             view before returning. </param>
        /// <param name="bSynchronizeDataViews">        The allocate if absent. </param>
        /// <param name="bUpdateCoherenceState">        State of the update coherence. </param>
        /// <param name="bDispatchContext">             Context for the dispatch. </param>
        /// <param name="bSynchronousConsumer">         True if the consumer of this
        ///                                             buffer requires outstanding operations
        ///                                             to be complete. </param>
        ///
        /// <returns>   null if it fails, else the channel buffer. </returns>
        ///-------------------------------------------------------------------------------------------------

        void* 
        GetHostChannelBuffer(
            __in AsyncContext * pAsyncContext,
            __in UINT nChannelIndex, 
            __in BUFFER_COHERENCE_STATE uiRequestedState, 
            __in BOOL bMaterializeAbsentBuffers,
            __in BOOL bSynchronizeDataViews,
            __in BOOL bUpdateCoherenceState,
            __in BOOL bDispatchContext,
            __in BOOL bSynchronousConsumer
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Synchronizes views with the given memory spaces. If locks are not already held,
        ///             then acquire them. Since we order datablock lock acquisition after accelerator,
        ///             we have to release our locks on this datablock and re-acquire them. Hence it is
        ///             *extremely important* to use this member only if no datablock state has been
        ///             modified yet with the lock held. Use only if you know what you're doing! Or think
        ///             you do...
        ///             </summary>
        ///
        /// <remarks>   crossbac, 1/3/2012. </remarks>
        ///
        /// <param name="nDestMemorySpaceID">   Identifier for the memory space. </param>
        /// <param name="nSourceMemorySpaceID"> Identifier for the source memory space. </param>
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the source
        ///                                     asynchronous. </param>
        /// <param name="uiRequestedState">     target coherence state for the requested copy. </param>
        /// <param name="bLocksRequired">       false if caller already holds locks. </param>
        /// <param name="bForceSynchronous">    The force synchronous. </param>
        ///
        /// <returns>   PTRESULT: use PTSUCCESS/PTFAIL macros. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        SynchronizeViews(
            __in UINT nDestMemorySpaceID, 
            __in UINT nSourceMemorySpaceID, 
            __in AsyncContext * pAsyncContext,
            __in BUFFER_COHERENCE_STATE uiRequestedState,
            __in BOOL bLocksRequired,
            __in BOOL bForceSynchronous
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a PBuffer for the channel in the memory space given by the accelerator
        ///             object.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pAccelerator">         (optional) [in] If non-null, an accelerator that will
        ///                                     require a view of this block. </param>
        /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
        /// <param name="pProxyAccelerator">    [in,out] If non-null, the proxy accelerator. </param>
        /// <param name="nChannelIndex">        Zero-based index of the n channel. </param>
        /// <param name="pInitialData">         (optional) [in] If non-null, initial data. </param>
        /// <param name="bUpdateCoherenceMap">  true to update coherence map. </param>
        /// <param name="bRaw">                 true to raw. </param>
        ///
        /// <returns>   PTRESULT: use PTSUCCESS/PTFAIL macros. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        InstantiateChannel(
            __in Accelerator * pAccelerator,
            __in AsyncContext * pAsyncContext,
            __in Accelerator * pProxyAccelerator,
            __in UINT nChannelIndex,
            __in HOSTMEMORYEXTENT * pInitialData,
            __in bool bUpdateCoherenceMap, 
            __in bool bRaw
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a PBuffer for the channel in the memory space given. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="nMemorySpaceId">       memory space in which to allocate. </param>
        /// <param name="pAsyncContext">        [in,out] Context for the asynchronous. </param>
        /// <param name="pProxyAllocator">      [in,out] Identifier for the proxy memory space. </param>
        /// <param name="nChannelIndex">        Zero-based index of the n channel. </param>
        /// <param name="pInitialData">         (optional) [in] If non-null, initial data. </param>
        /// <param name="bUpdateCoherenceMap">  true to update coherence map. </param>
        /// <param name="bRaw">                 true to raw. </param>
        ///
        /// <returns>   PTRESULT: use PTSUCCESS/PTFAIL macros. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        InstantiateChannel(
            __in UINT nMemorySpaceId,
            __in AsyncContext * pAsyncContext,
            __in Accelerator * pProxyAllocator,
            __in UINT nChannelIndex,
            __in HOSTMEMORYEXTENT * pInitialData,
            __in bool bUpdateCoherenceMap, 
            __in bool bRaw
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate host buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiBytes">  The bytes. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        void * AllocateHostBuffer(UINT uiBytes);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Free host buffer. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pBuffer">  [in,out] If non-null, the buffer. </param>
        ///-------------------------------------------------------------------------------------------------

        void FreeHostBuffer(void * pBuffer);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a size estimate. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT SizeEstimator(UINT nChannelIndex);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the data size estimator. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT DataSizeEstimator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the meta size estimator. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT MetaSizeEstimator();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the template size estimator. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT TemplateSizeEstimator();       

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check invariants for the coherence state machine. Any number of INVALID and
        ///             NO_ENTRY copies are allows. There can be 0..* SHARED copies and 0 EXCLUSIVE, or 1
        ///             EXCLUSIVE copy and 0 SHARED copies.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/3/2012. </remarks>
        ///------------------------------------------------------------------------------------------------

        void CheckInvariants();

        /// <summary> The template from which the geometry of this 
        /// 		  block should be derived. Can be NULL to indicate
        /// 		  the Datablock is byte-addressable/variable length. 
        /// 		  </summary>
        DatablockTemplate *		            m_pTemplate;

        /// <summary>   This is the *requested* allocation size for underlying buffers. For example, a
        ///             caller may request a new Datablock with an initial data channel size of 2046,
        ///             later growing it to 4095, and sealing it with 3000 bytes. The specified size
        ///             tracks the 2046/4095 specified size. To leave the system flexibility to allocate
        ///             sizes more efficiently (2046 and 4095 usually will be better satisfied with power-
        ///             of-two allocation sizes), the allocation size is tracked separately. The "3000"
        ///             number will be found in the finalized size member.
        ///             </summary>
        UINT            m_cbRequested[NUM_DATABLOCK_CHANNELS];

        /// <summary>   This is the allocation size for underlying buffers. For example, a
        ///             caller may request a new Datablock with an initial data channel size of 2046,
        ///             later growing it to 4095, and sealing it with 3000 bytes. The specified size
        ///             tracks the 2046/4095 specified size. To leave the system flexibility to allocate
        ///             sizes more efficiently (2046 and 4095 usually will be better satisfied with power-
        ///             of-two allocation sizes), the allocation size is tracked separately. The "3000"
        ///             number will be found in the finalized size member.
        ///             </summary>
        UINT            m_cbAllocated[NUM_DATABLOCK_CHANNELS];

        /// <summary>   The actual buffer size. This is the number of valid byte in the buffer. A caller
        ///             may request a new Datablock with an initial data channel size of 2048, later
        ///             growing it to 4096, and sealing it with 3000 bytes. The requested size tracks the
        ///             2048/4096 allocation size, while the *actual* size is 3000. Actual size is
        ///             tracked in X, Y, Z dimensions.
        ///             </summary>
        UINT            m_cbFinalized[NUM_DATABLOCK_CHANNELS];

        /// <summary>   true if the block is logically empty. "logical emptiness" is a property that
        ///             inheres in blocks with a record-count of 0, but which is backed by physical
        ///             buffers in at least one memory space. Since we lack the ability to allocate and
        ///             bind zero-size objects in most GPU runtimes, we allocate non-zero size objects,
        ///             but remember cases where those objects actually have a logical size of zero.
        ///             </summary>
        BOOL            m_bLogicallyEmpty;

        /// <summary>   true to attempt pinned host buffers. </summary>
        BOOL            m_bAttemptPinnedHostBuffers;

        /// <summary> The control code encodes control signals associated with this
        /// 		  block. For example, if the block has a control code of DBCTL_EOF,
        /// 		  then this is the last block in a stream of records. 
        /// 		  </summary>
        std::stack<CONTROLSIGNAL>           m_vControlSignalStack;
        
        /// <summary> Number of records in this block. Used only when the block
        /// 		  describes part of a record stream. </summary>
        UINT                                m_uiRecordCount;
        
        /// <summary> True if this block encodes part of a record stream </summary>
        BOOL                                m_bRecordStream;
        
        /// <summary> The buffer access flags for this block. The flags
        /// 		  encode whether read/write access is required in host
        /// 		  and accelerator domains, as well as whether specialized
        /// 		  memories such as constant memory should be used to back this
        /// 		  block when such features are available.
        /// 		  </summary>
        unsigned int			            m_eBufferAccessFlags;
        
        /// <summary> The destination port for this block. If this block is 
        /// 		  bound as an output or in/out block for dispatch the destination
        /// 		  port is the output port participating in that binding. 
        /// 		  If this block is not bound as an output or in/out block for 
        /// 		  dispatch the destination port is NULL. 
        /// 		  </summary>
        Port *                              m_pDestinationPort;
        
        /// <summary> True if we are deleting this block because its refcount
        /// 		  has dropped to zero. This is a debug tool. 
        /// 		  </summary>
        BOOL                                m_bDeleting;
        
        /// <summary> The Task that most recently wrote this block, NULL if no
        /// 		  such task exists. </summary>
        Task *                              m_pProducerTask;
        
        /// <summary> Unique identifier for the block </summary>
        UINT                                m_uiDBID;

        /// <summary> True if this block is part of a block pool.
        /// 		  PTask pools blocks on output ports to avoid latency for
        /// 		  allocating blocks that required by every dispatch and
        /// 		  have predictable geometry. If a block is pooled, it returns
        /// 		  to its block pool when it is released. If it is not, it
        /// 		  is deleted when its refcount hits zero.  
        /// 		  </summary>
        BOOL                                m_bPooledBlock;
        
        /// <summary> The owner of the pool if this block is pooled. </summary>
        BlockPoolOwner*                     m_pPoolOwner;

        /// <summary>   If a block is pooled, and it has an 
        ///             initializer associated with it, there will be 
        ///             valid PBuffer objects for it that may become invalidated
        ///             late without being modified (because a writeable view
        ///             in another memory space invalidate it). When we recycle
        ///             datablocks, we use this list to determine whether we
        ///             need to recreate the initial data for a block when it
        ///             has been recycled through a block pool. </summary>
        std::set<PBuffer*>                  m_vInitialValueCleanList;

        /// <summary>   true if the pool requires an initial value. </summary>
        BOOL                                m_bPoolRequiresInitialValue;

        /// <summary>   true to pool initial value set. </summary>
        BOOL                                m_bPoolInitialValueSet;
        
        /// <summary> True if this block contains marshallable data. 
        /// 		  Datablocks can become unmarshallable if accelerator-side code writes
        /// 		  values that are valid only in the address space of the device in the
        /// 		  block (e.g. pointers). When a block becomes unmarshallable, Tasks that
        /// 		  bind it can only run on the accelerator that produced the most recent
        /// 		  view of the datablock.
        /// 		  </summary>
        BOOL                                m_bMarshallable;
        
        /// <summary> True if the programmer has called seal to establish the
        /// 		  actual size/stride/record-count for a block. Meaningless/ignored
        /// 		  for blocks whose templates describe fixed geometry. 
        /// 		  </summary>
        BOOL                                m_bSealed;
        
        /// <summary> True if the block is byte addressable, typically meaning
        /// 		  it contains variable stride data. This field is also important
        /// 		  because some GPU runtimes (e.g. DirectX) require different APIs
        /// 		  to bind buffers that will be accessed with byte-stride on the device,
        /// 		  so we need to know this at allocation time. 
        /// 		  </summary>
        BOOL                                m_bByteAddressable;
        
        /// <summary> The buffer map. Each entry represents a memory space with
        /// 		  coherence state (invalid, shared, exclusive, no-entry), along with
        /// 		  platform-specific buffers for each of the datablock's channels. 
        /// 		  </summary>
        BUFFER_MAP_ENTRY**                  m_ppBufferMap;
        
        /// <summary> A channel iterator for the block. Useful for hiding the details
        /// 		  of the internal channel data structure from code that needs
        /// 		  to iterate over those channels and work only with valid entries.
        /// 		  </summary>
        ChannelIterator                     m_vIterator;

        /// <summary>   true if device views can be materialized using memset rather than using host-
        ///             device copies. This is typically true only of descriptor blocks (size, control
        ///             code blocks) and initial value blocks. In such cases we typically want a device-
        ///             side buffer with an integer or all-zeros, etc, which can be created more
        ///             efficiently with memset if the back-end framework supports it (e.g. cuMemset* in
        ///             CUDA).
        ///             </summary>
        BOOL                                m_bDeviceViewsMemsettable;

        /// <summary>   true to force requested size during allocation of buffers, in the presence
        /// 			of a template that carries dimension information that differs from the
        /// 			requested sizes. </summary>
        BOOL                                m_bForceRequestedSize;

        /// <summary>   true if the block was resized using Grow* calls. </summary>
        BOOL                                m_bBlockResized;

        /// <summary> Context associated with this datablock by the application. 
        /// 		  </summary>
        void *          m_pApplicationContext;

    public:

        /// <summary>   The size descriptor template used to create allocation size blocks. </summary>
        static DatablockTemplate *          m_gSizeDescriptorTemplate;

    protected:

        /// <summary> End-of-iteration sentinel </summary>
        static ChannelIterator m_iterLastChannel;

        friend struct ChannelIterator_t;

        public: std::set<Task*> m_vWriters;
        
        CoherenceProfiler * m_pCoherenceProfiler;      

        DatablockProfiler * m_pDatablockProfiler;

        /// <summary>   true if this object was allocated by cloning another block. </summary>
        BOOL m_bIsClone;
        
        /// <summary>   The datablock that was cloned to create this block,
        /// 			if m_bIsClone is set to TRUE. 
        /// 			</summary>
        Datablock * m_pCloneSource;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets control signals. </summary>
        ///
        /// <remarks>   Crossbac, 2/14/2013. </remarks>
        ///
        /// <returns>   The control signals. </returns>
        ///-------------------------------------------------------------------------------------------------

        CONTROLSIGNAL __getControlSignals();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record port binding. </summary>
        ///
        /// <remarks>   Crossbac, 9/19/2012. </remarks>
        ///
        /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
        ///-------------------------------------------------------------------------------------------------

        void RecordBinding(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record task binding. </summary>
        ///
        /// <remarks>   Crossbac, 9/19/2012. </remarks>
        ///
        /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
        ///-------------------------------------------------------------------------------------------------

        void RecordBinding(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record binding. </summary>
        ///
        /// <remarks>   Crossbac, 9/20/2012. </remarks>
        ///
        /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        void RecordBinding(Port * pPort, Task * pTask, Port * pIOConsumer=NULL);

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

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets instantiated buffer sizes. Populate a map from memory space id to 
        ///             the number of bytes of PBuffer backing space that is actually allocated
        ///             for this block. This method is a tool to help the runtime (specifically, the
        ///             GC) figure out when a forced GC sweep might actually help free up some
        ///             GPU memory. 
        ///             
        ///             No lock is required of the caller because it is expected to be called by the
        ///             GC--the datablock should therefore only be accessible from one thread context,
        ///             so there is no danger of the results becoming stale after the call completes.
        ///             ***SO, if you decide to repurpose this method and call it  from any other context, 
        ///             be sure to lock the datablock first. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/7/2013. </remarks>
        ///
        /// <param name="vBlockFreeBytes">  [in,out] The block free in bytes. </param>
        ///
        /// <returns>   true if there are backing buffers in any memory space other than the host,
        ///             false otherwise. 
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetInstantiatedBufferSizes(
            __inout std::map<UINT, size_t>& vBlockFreeBytes
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Releases the physical buffers backing datablock channels for the accelerator
        ///             whose memory space has the given ID, uiMemSpaceID. 
        ///             
        ///             No lock is required of the caller because it is expected to be called by the
        ///             GC--the datablock should therefore only be accessible from one thread context,
        ///             so there is no danger of the results becoming stale after the call completes.
        ///             ***SO, if you decide to repurpose this method and call it  from any other context, 
        ///             be sure to lock the datablock first. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 9/7/2013. </remarks>
        ///
        /// <param name="uiMemSpaceID"> Identifier for the memory space. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        ReleasePhysicalBuffers(
            __in UINT uiMemSpaceID
            );

        friend class CoherenceProfiler;
        friend class DatablockProfiler;
        friend class GarbageCollector;
        friend class SignalProfiler;
    };

};
#endif
