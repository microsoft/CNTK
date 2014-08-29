///-------------------------------------------------------------------------------------------------
// file:	CoherenceProfiler.h
//
// summary:	Declares the coherence profiler class
///-------------------------------------------------------------------------------------------------

#ifndef __COHERENCE_PROFILER_H__
#define __COHERENCE_PROFILER_H__

#include "primitive_types.h"
#include <map>
#include <string>
#include <assert.h>

class CHighResolutionTimer;

namespace PTask {

    class Port;
    class Task;
    class Datablock;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Event types that can cause a coherence state transition. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum COHERENCEEVENT_t {

        /// <summary>   The event causing the transition was not specified. </summary>
        CET_UNSPECIFIED = 0,

        /// <summary>   The state transition was triggered by a binding to task input</summary>
        CET_BIND_INPUT = 1,

        /// <summary>   The state transition was triggered by a binding to taks output</summary>
        CET_BIND_OUTPUT = 2,

        /// <summary>   The state transition was triggered by a binding to a task constant port</summary>
        CET_BIND_CONSTANT = 3,

        /// <summary>   The state transition was triggered by pushing into multiple consumer channels </summary>
        CET_PUSH_DOWNSTREAM_SHARE = 4,

        /// <summary>   The state transition was triggered by a user request for a pointer in host space</summary>
        CET_POINTER_REQUEST = 5,

        /// <summary>   The state transition was triggered by the deletion of the block</summary>
        CET_BLOCK_DELETE = 6,

        /// <summary>   The state transition was triggered by the cloning of the block </summary>
        CET_BLOCK_CLONE = 7,        

        /// <summary>   The state transition was triggered by block allocation </summary>
        CET_BLOCK_CREATE = 8,        

        /// <summary>   we are updating the host view of the block, but don't actually have
        /// 			access to the information we need to figure out what action
        /// 			triggered the view update. Most likely a user request
        /// 			</summary>
        CET_HOST_VIEW_UPDATE = 9,

        /// <summary>   we are updating the device view of the block, but don't actually have
        /// 			access to the information we need to figure out what action
        /// 			triggered the view update. Most likely a user request
        /// 			</summary>
        CET_ACCELERATOR_VIEW_UPDATE = 10,

        /// <summary>   Buffers are being allocated for this block </summary>
        CET_BUFFER_ALLOCATE = 11,

        /// <summary>   a request to grow the buffer caused some buffer reallocation and 
        /// 			potentially view updates as a side effect. </summary>
        CET_GROW_BUFFER = 12,

        /// <summary>   a request to synthesize a metadata block caused the traffic </summary>
        CET_SYNTHESIZE_BLOCK = 13,

        /// <summary>   needed a pinned host buffer in addition to a dev buffer </summary>
        CET_PINNED_HOST_VIEW_CREATE = 14,

    } COHERENCEEVENTTYPE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Defines a structure for collecting detailed data for 
    /// 			a coherence state transition. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct COHERENCETRANSITION_t {
    public:

        /// <summary>   True if this transition has completed and this record should
        /// 			no longer be allowed to change. 
        /// 			</summary>
        BOOL bFinalized;

        /// <summary>   True if a data transfer occurred for this transition. </summary>
        BOOL bXferOccurred;

        /// <summary>   The timestamp at the start of the transition. </summary>
        double nStartTimestamp;

        /// <summary>   The timestamp at the end of the transition. </summary>
        double nEndTimestamp;

        /// <summary>   Identifier for the source memory space. </summary>
        UINT uiSrcMemorySpaceId;

        /// <summary>   Identifier for the destination memory space. </summary>
        UINT uiDstMemorySpaceId;

        /// <summary>   The event that triggered this transition. </summary>
        COHERENCEEVENTTYPE eTriggerEvent;

        /// <summary>   The requested state of the block in response to the event. </summary>
        BUFFER_COHERENCE_STATE eTargetState;

        /// <summary>   The start state of the block (snapshot of the state per memory space). </summary>
        BUFFER_COHERENCE_STATE eStartState[MAX_MEMORY_SPACES];

        /// <summary>   The end state of the block (snapshot of the state per memory space). </summary>
        BUFFER_COHERENCE_STATE eEndState[MAX_MEMORY_SPACES];

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        COHERENCETRANSITION_t(
            double dStartTimestamp
            ) 
        {
            bFinalized         = FALSE;
            bXferOccurred      = FALSE;
            nStartTimestamp    = dStartTimestamp;
            eTriggerEvent      = CET_UNSPECIFIED;
            eTargetState       = BSTATE_NO_ENTRY;
            uiSrcMemorySpaceId = HOST_MEMORY_SPACE_ID;
            uiDstMemorySpaceId = HOST_MEMORY_SPACE_ID;
            for(int i=0; i<MAX_MEMORY_SPACES; i++) {
                eStartState[i] = BSTATE_NO_ENTRY;
                eEndState[i] = BSTATE_NO_ENTRY;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalizes this record. </summary>
        ///
        /// <remarks>   Crossbac, 9/19/2012. </remarks>
        ///
        /// <param name="dEndTimestamp">    The end timestamp. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        Finalize(
            double dEndTimestamp,
            BOOL bTransfer
            )
        {
            nEndTimestamp = dEndTimestamp;
            bFinalized    = TRUE;
            bXferOccurred = bTransfer;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check that all the memory spaces have compatible states in the snapshot. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetNumberOfValidCopies(
            __in BUFFER_COHERENCE_STATE * pSnapshot
            )
        {
            UINT nValidEntries = 0;
            UINT nMemSpaces = MemorySpace::GetNumberOfMemorySpaces();
        
            for(UINT i=HOST_MEMORY_SPACE_ID; i<nMemSpaces; i++) {
            
                // count up number of copies in various states.
                BUFFER_COHERENCE_STATE uiCoherenceState = pSnapshot[i];
                switch(uiCoherenceState) {
                case BSTATE_NO_ENTRY:   break;
                case BSTATE_INVALID:    break;
                case BSTATE_SHARED:     nValidEntries++;    break;
                case BSTATE_EXCLUSIVE:  nValidEntries++; break;
                }
            }

            return nValidEntries;
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check that all the memory spaces have compatible states in the snapshot. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        ValidState(
            __in BUFFER_COHERENCE_STATE * pSnapshot
            )
        {
            UINT nInvalidEntries = 0;
            UINT nNoEntryEntries = 0;
            UINT nExclusiveCopies = 0;
            UINT nSharedCopies = 0;
            UINT nMemSpaces = MemorySpace::GetNumberOfMemorySpaces();
        
            for(UINT i=HOST_MEMORY_SPACE_ID; i<nMemSpaces; i++) {
            
                // count up number of copies in various states.
                BUFFER_COHERENCE_STATE uiCoherenceState = pSnapshot[i];
                switch(uiCoherenceState) {
                case BSTATE_NO_ENTRY:   nInvalidEntries++;  break;
                case BSTATE_INVALID:    nNoEntryEntries++;  break;
                case BSTATE_SHARED:     nSharedCopies++;    break;
                case BSTATE_EXCLUSIVE:  nExclusiveCopies++; break;
                }
            }
        
            BOOL bCorrectSharedState    = (nSharedCopies >= 0 && nExclusiveCopies == 0);
            BOOL bCorrectExclusiveState = (nSharedCopies == 0 && nExclusiveCopies == 1);
            assert(bCorrectSharedState || bCorrectExclusiveState);
            return bCorrectSharedState || bCorrectExclusiveState;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check that all the memory spaces have compatible states in the snapshot. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BUFFER_COHERENCE_STATE
        GetCollectiveState(
            __in BUFFER_COHERENCE_STATE * pSnapshot
            )
        {
            assert(ValidState(pSnapshot));
            UINT nInvalidEntries = 0;
            UINT nNoEntryEntries = 0;
            UINT nExclusiveCopies = 0;
            UINT nSharedCopies = 0;
            UINT nMemSpaces = MemorySpace::GetNumberOfMemorySpaces();
        
            for(UINT i=HOST_MEMORY_SPACE_ID; i<nMemSpaces; i++) {
            
                // count up number of copies in various states.
                BUFFER_COHERENCE_STATE uiCoherenceState = pSnapshot[i];
                switch(uiCoherenceState) {
                case BSTATE_NO_ENTRY:   nInvalidEntries++;  break;
                case BSTATE_INVALID:    nNoEntryEntries++;  break;
                case BSTATE_SHARED:     nSharedCopies++;    break;
                case BSTATE_EXCLUSIVE:  nExclusiveCopies++; break;
                }
            }
        
            if(nExclusiveCopies > 0) return BSTATE_EXCLUSIVE;
            if(nSharedCopies > 0) return BSTATE_SHARED;
            if(nInvalidEntries > 0) return BSTATE_INVALID;
            return BSTATE_NO_ENTRY;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a start state. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    If non-null, the snapshot. </param>
        ///
        /// <returns>   The start state. </returns>
        ///-------------------------------------------------------------------------------------------------

        BUFFER_COHERENCE_STATE
        GetStartState(
            VOID
            ) 
        {
            return GetCollectiveState(eStartState);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the final state for the state transition. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    If non-null, the snapshot. </param>
        ///
        /// <returns>   The start state. </returns>
        ///-------------------------------------------------------------------------------------------------

        BUFFER_COHERENCE_STATE
        GetFinalState(
            VOID
            ) 
        {
            return GetCollectiveState(eEndState);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   number of valid copies in an accelerator space. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetNumberOfValidAcceleratorCopies(
            __in BUFFER_COHERENCE_STATE * pSnapshot
            )
        {
            UINT nValidEntries = 0;
            UINT nMemSpaces = MemorySpace::GetNumberOfMemorySpaces();
        
            for(UINT i=HOST_MEMORY_SPACE_ID+1; i<nMemSpaces; i++) {
            
                // count up number of copies in various states.
                BUFFER_COHERENCE_STATE uiCoherenceState = pSnapshot[i];
                switch(uiCoherenceState) {
                case BSTATE_NO_ENTRY:   break;
                case BSTATE_INVALID:    break;
                case BSTATE_SHARED:     nValidEntries++;    break;
                case BSTATE_EXCLUSIVE:  nValidEntries++; break;
                }
            }

            return nValidEntries;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   number of valid copies in host space. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetNumberOfValidHostCopies(
            __in BUFFER_COHERENCE_STATE * pSnapshot
            )
        {

            BUFFER_COHERENCE_STATE uiCoherenceState = pSnapshot[HOST_MEMORY_SPACE_ID];
            switch(uiCoherenceState) {
            case BSTATE_NO_ENTRY:   return 0;
            case BSTATE_INVALID:    return 0;
            case BSTATE_SHARED:     return 1;
            case BSTATE_EXCLUSIVE:  return 1;
            }
            return 0;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   number of valid copies in an accelerator space. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetStartNumberOfValidAcceleratorCopies(
            void
            )
        {
            return GetNumberOfValidAcceleratorCopies(eStartState);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   number of valid copies in an accelerator space. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetFinalNumberOfValidAcceleratorCopies(
            void
            )
        {
            return GetNumberOfValidAcceleratorCopies(eEndState);
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   number of valid copies in host space. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetStartNumberOfValidHostCopies(
            void
            )
        {
            return GetNumberOfValidHostCopies(eStartState);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   number of valid copies in host space. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pSnapshot">    [in,out] If non-null, the snapshot. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetFinalNumberOfValidHostCopies(
            void
            )
        {
            return GetNumberOfValidHostCopies(eEndState);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   was this transfer a Host -> Device transfer? </summary>
        ///
        /// <remarks>   Crossbac, 9/20/2012. </remarks>
        ///
        /// <returns>   true if h to d xfer, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        IsHToDXfer(
            VOID
            )
        {
            if(!bXferOccurred) return FALSE;
            UINT nValidHostViewsS = GetStartNumberOfValidHostCopies();
            UINT nValidAccViewsS  = GetStartNumberOfValidAcceleratorCopies();
            UINT nValidAccViewsF  = GetFinalNumberOfValidAcceleratorCopies();
            switch(GetFinalState()) {
            case BSTATE_NO_ENTRY: assert(FALSE); break; // why transfer if there is no buffer?
            case BSTATE_INVALID:  assert(FALSE); break; // why transfer to create an invalid entry?
            case BSTATE_SHARED:    return nValidAccViewsF > nValidAccViewsS && (nValidHostViewsS > 0 || uiSrcMemorySpaceId == HOST_MEMORY_SPACE_ID);
            case BSTATE_EXCLUSIVE: return nValidAccViewsF > 0 && (nValidHostViewsS > 0 || uiSrcMemorySpaceId == HOST_MEMORY_SPACE_ID);
            }
            return FALSE;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   was this transfer a Device -> Host transfer? </summary>
        ///
        /// <remarks>   Crossbac, 9/20/2012. </remarks>
        ///
        /// <returns>   true if d to h xfer, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        IsDToHXfer(
            VOID
            )
        {
            if(!bXferOccurred) return FALSE;
            UINT nValidHostViewsS = GetStartNumberOfValidHostCopies();
            UINT nValidHostViewsF = GetFinalNumberOfValidHostCopies();
            UINT nValidAccViewsS  = GetStartNumberOfValidAcceleratorCopies();
            UINT nValidAccViewsF  = GetFinalNumberOfValidAcceleratorCopies();
            switch(GetFinalState()) {
            case BSTATE_NO_ENTRY: assert(FALSE); break; // why transfer if there is no buffer?
            case BSTATE_INVALID:  assert(FALSE); break; // why transfer to create an invalid entry?
            case BSTATE_SHARED:    return nValidAccViewsS > 0 && nValidHostViewsS == 0 && nValidHostViewsF > 0;
            case BSTATE_EXCLUSIVE: return nValidAccViewsF < nValidAccViewsS && nValidHostViewsF > 0;
            }
            return FALSE;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   was this transfer a Device -> Device transfer? </summary>
        ///
        /// <remarks>   Crossbac, 9/20/2012. </remarks>
        ///
        /// <returns>   true if d to d xfer, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        IsDToDXfer(
            VOID
            )
        {
            if(!bXferOccurred) return FALSE;
            UINT nValidHostViewsS = GetStartNumberOfValidHostCopies();
            UINT nValidAccViewsS  = GetStartNumberOfValidAcceleratorCopies();
            UINT nValidAccViewsF  = GetFinalNumberOfValidAcceleratorCopies();
            if(nValidAccViewsS == 0) return FALSE;     // no valid start device view to xfer
            if(nValidAccViewsF == 0) return FALSE;     // no valid end device view
            switch(GetFinalState()) {
            case BSTATE_NO_ENTRY: assert(FALSE); break; // why transfer if there is no buffer?
            case BSTATE_INVALID:  assert(FALSE); break; // why transfer to create an invalid entry?

            case BSTATE_SHARED:  
                // if the final state is shared, and there
                // there was a valid device view to begin with
                // then the number of device views must be strictly increasing. 
                // Otherwise, either no new dev view was created (meaning no X->D xfer) or
                // some device view had to have been invalidated, which our system would not do.
                if(nValidAccViewsF <= nValidAccViewsS) return FALSE; // no additional device views
                switch(GetStartState()) {
                case BSTATE_NO_ENTRY: return FALSE;
                case BSTATE_INVALID:  return FALSE;
                case BSTATE_SHARED:   
                    // copy could come from host or device.
                    if(nValidHostViewsS == 0) return TRUE; // *had* to come from device
                    return uiSrcMemorySpaceId != HOST_MEMORY_SPACE_ID;
                case BSTATE_EXCLUSIVE:
                    // there was only one copy to begin with so
                    // the source had to be device if there was a valid device view
                    return nValidAccViewsS > 0;
                }
                
                return nValidAccViewsS > 0 && nValidAccViewsF > nValidAccViewsS;
            case BSTATE_EXCLUSIVE: 
                // if the final state is exclusive, then
                // the mem space in which we have a valid view must have changed.
                // we would only do a D->D transfer if there was not a valid host
                // view available, since (with some obvious exceptions), we generally
                // must do D->D transfers through the host, so would prefer a host 
                // view if it was available. 
                return nValidHostViewsS == 0; 
            }
            return FALSE;
        }

    } COHERENCETRANSITION;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Defines a structure for tracking per-datablock instance
    /// 			history of coherence traffic participation. If the PROFILE_MIGRATION
    /// 			compiler directive is selected, each datablock will maintain 
    /// 			its own history in this structure, and each history will be merged
    /// 			in the the static view defined below upon deletion. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct COHERENCEHISTORY_t {
    public:
        /// <summary>   History of all ports to which a block has been bound. </summary>
        std::map<__int64, Port*>* pvPortBindHistory;

        /// <summary>   History of all IO consumer ports to which a block has been bound. </summary>
        std::map<__int64, Port*>* pvIOCPortBindHistory;

        /// <summary>   The set of all tasks which have touched this block. </summary>
        std::map<__int64, Task*>* pvTaskBindHistory;

        /// <summary>   The accelerator bind history (tracked as accelerator id). </summary>
        std::map<__int64, UINT>*  pvAcceleratorBindHistory;

        /// <summary>   The accelerator bind history (tracked as accelerator id). </summary>
        std::map<__int64, UINT>*  pvDepAcceleratorBindHistory;

        /// <summary>   The coherence state history. </summary>
        std::map<__int64, COHERENCETRANSITION*>* pvStateHistory;

        /// <summary>   The dbuid of the datablock for which this occurred. </summary>
        UINT uiDBUID;

        /// <summary>   The number of times this block required D->H xfer. </summary>
        LONG            nDToHCopies;

        /// <summary>   The number of times this block required H->D xfer. </summary>
        LONG            nHToDCopies;

        /// <summary>   The number of times this block required D->D xfer. </summary>
        LONG            nDToDCopies;

        /// <summary>   The number of times this block required H->H xfer. 
        ///             This is a sanity check--it better be 0!	
        ///             </summary>
        LONG            nHToHCopies;

        /// <summary>   The total number of bytes transferred over the life cycle of
        /// 			this datablock. </summary>
        LONG            nTotalSyncBytes;

        /// <summary>   The number of times a block was bound concurrently
        /// 			to multiple ports. This may have some error due to 
        /// 			the resolution of the timer, and may need to be revised. </summary>
        UINT            uiConcurrentPortBindings;

        /// <summary>   The number of times a block was bound concurrently
        /// 			to multiple ports. This may have some error due to 
        /// 			the resolution of the timer, and may need to be revised. </summary>
        UINT            uiConcurrentTaskBindings;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        COHERENCEHISTORY_t(
            UINT uiDatablockID
            ) 
        {
            pvPortBindHistory           = new std::map<__int64, Port*>();
            pvIOCPortBindHistory        = new std::map<__int64, Port*>();
            pvTaskBindHistory           = new std::map<__int64, Task*>();
            pvAcceleratorBindHistory    = new std::map<__int64, UINT>();
            pvDepAcceleratorBindHistory = new std::map<__int64, UINT>();
            pvStateHistory              = new std::map<__int64, COHERENCETRANSITION*>();
            nDToHCopies                 = 0;
            nHToDCopies                 = 0;
            nDToDCopies                 = 0;
            nHToHCopies                 = 0;
            nTotalSyncBytes             = 0;
            uiConcurrentPortBindings    = 0;
            uiConcurrentTaskBindings    = 0;
            uiDBUID                     = uiDatablockID;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        ~COHERENCEHISTORY_t(
            VOID
            ) 
        {
            std::map<__int64, COHERENCETRANSITION*>::iterator mi;
            for(mi=pvStateHistory->begin(); mi!=pvStateHistory->end(); mi++) 
                delete mi->second;
            delete pvPortBindHistory;
            delete pvIOCPortBindHistory;
            delete pvTaskBindHistory;
            delete pvStateHistory;
            delete pvAcceleratorBindHistory;
            delete pvDepAcceleratorBindHistory;
        }

    } COHERENCEHISTORY;

    class CoherenceProfiler {

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <param name="pDatablock">   [in,out] If non-null, the datablock. </param>
        ///-------------------------------------------------------------------------------------------------

        CoherenceProfiler(Datablock * pDatablock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~CoherenceProfiler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the coherence traffic profiler. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <param name="bEnable">  true to enable, false to disable. </param>
        /// <param name="bVerbose"> true to verbose. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Initialize(BOOL bEnable, BOOL bVerbose=FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitializes the coherence traffic profiler. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Deinitialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the coherence traffic statistics. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Report(std::ostream& ios); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the coherence traffic statistics. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static std::stringstream * GetReport(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the coherence traffic statistics. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void GetDetailedReport(std::ostream& ios); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Coherence tracker record view update start. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pDatablock">           If non-null, the datablock. </param>
        /// <param name="nDestMemorySpaceID">   Identifier for the memory space. </param>
        /// <param name="eEventType">           Type of the event. </param>
        ///
        /// <returns>   new transition object. </returns>
        ///-------------------------------------------------------------------------------------------------

        COHERENCETRANSITION *
        RecordViewUpdateStart(
            __in UINT nDestMemorySpaceID, 
            __in COHERENCEEVENTTYPE eEventType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Coherence tracker record view update end. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///
        /// <param name="pDatablock">           If non-null, the datablock. </param>
        /// <param name="nSrcMemorySpaceID">    Identifier for the source memory space. </param>
        /// <param name="uiRequestedState">     The requested coherence state. This affects whether other
        ///                                     accelerator views require invalidation. </param>
        /// <param name="bTransferOccurred">    The transfer occurred. </param>
        /// <param name="pTx">                  non-null, the state transition descriptor. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        RecordViewUpdateEnd(
            __in UINT nSrcMemorySpaceID, 
            __in BUFFER_COHERENCE_STATE uiRequestedState,
            __in BOOL bTransferOccurred,
            __in COHERENCETRANSITION * pTx
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record port binding. </summary>
        ///
        /// <remarks>   Crossbac, 9/19/2012. </remarks>
        ///
        /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
        ///-------------------------------------------------------------------------------------------------

        void RecordPortBinding(Port * pPort);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record task binding. </summary>
        ///
        /// <remarks>   Crossbac, 9/19/2012. </remarks>
        ///
        /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
        ///-------------------------------------------------------------------------------------------------

        void RecordTaskBinding(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record binding. </summary>
        ///
        /// <remarks>   Crossbac, 9/20/2012. </remarks>
        ///
        /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        void RecordBinding(Port * pPort, Task * pTask, Port * pIOConsumer);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Coherence tracker set detailed. </summary>
        ///
        /// <remarks>   Crossbac, 9/21/2012. </remarks>
        ///
        /// <param name="bDetailed">    true to collect detailed stats. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetDetailed(BOOL bDetailed);

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the coherence history for this block. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void InitializeInstanceHistory();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitializes the coherence history for this block. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DeinitializeInstanceHistory();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Merge the coherence history for this block with the static view. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void MergeHistory();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Merge the coherence histories for all blocks into the static view. </summary>
        ///
        /// <remarks>   Crossbac, 9/18/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void MergeHistories();

        /// <summary>   The datablock this profiler object is tracking. </summary>
        Datablock * m_pDatablock;

        /// <summary>   Coherence history and statistics for this block, including:
        /// 			1. all ports to which this block has been bound.   
        /// 			2. The set of all tasks which have touched this block  
        /// 			3. The number of times this block required D->H xfer.   
        /// 			4. The number of times this block required H->D xfer.   
        /// 			5. The number of times this block required D->D xfer.    
        /// 			6. The number of times this block required H->H xfer.   
        /// 			7. The total number of bytes transferred over the life cycle of
        /// 			   this datablock.
        /// 			8. The history of state transitions. </summary>
        COHERENCEHISTORY*  m_pCoherenceHistory;

        /// <summary>   True if we are in the middle of recording a state transition 
        /// 			in the coherence profiler. Helps us catch situations where we 
        /// 			accidentally attempt nested recording of transitions, which
        /// 			would deeply screw up the results. 
        /// 			</summary>
        BOOL               m_bCoherenceProfilerTransitionActive;

        /// <summary>   The dev to dev migrations with invalidation. </summary>
        static LONG     m_nDToDMigrationsExclusive;

        /// <summary>   The dev to dev migrations with shared state. </summary>
        static LONG     m_nDToDMigrationsShared;

        /// <summary>   The host to dev migrations with invalidation. </summary>
        static LONG     m_nHToDMigrationsExclusive;

        /// <summary>   The host to dev migrations without invalidation. </summary>
        static LONG     m_nHToDMigrationsShared;

        /// <summary>   The dev to host migrations with invalidation. </summary>
        static LONG     m_nDToHMigrationsExclusive;

        /// <summary>   The dev to host migrations without invalidation. </summary>
        static LONG     m_nDToHMigrationsShared;

        /// <summary>   The number of times a coherence event caused multiple
        /// 			valid views to be abandoned. </summary>
        static LONG     m_nMultiViewInvalidations;

        /// <summary>   The number of state transitions whose cause was unspecified. </summary>
        static LONG     m_nCETUnspecified; 

        /// <summary>   The number of state transitions triggered by a binding to task input</summary>
        static LONG     m_nCETBindInput;

        /// <summary>   The number of state transitions triggered by a binding to taks output</summary>
        static LONG     m_nCETBindOutput;

        /// <summary>   The number of state transitions triggered by a binding to a task constant port</summary>
        static LONG     m_nCETBindConstant;

        /// <summary>   The number of state transitions triggered by pushing into multiple consumer channels </summary>
        static LONG     m_nCETDownstreamShare;

        /// <summary>   The number of state transitions triggered by a user request for a pointer in host space</summary>
        static LONG     m_nCETPointerRequest;

        /// <summary>   The number of state transitions triggered by the deletion of the block</summary>
        static LONG     m_nCETBlockDelete;

        /// <summary>   The number of state transitions triggered by the cloning of the block </summary>
        static LONG     m_nCETBlockClone;

        /// <summary>   The number of state transitions triggered by block allocation </summary>
        static LONG     m_nCETBlockCreate;

        /// <summary>   The number of state transitions triggered when we are updating the host view of
        ///             the block, but don't actually have access to the information we need to figure
        ///             out what action triggered the view update. Most likely a user request.
        ///             </summary>
        static LONG     m_nCETHostViewUpdate;

        /// <summary>   The number of state transitions triggered when we are updating the device view of
        ///             the block, but don't actually have access to the information we need to figure
        ///             out what action triggered the view update. Most likely a user request.
        ///             </summary>
        static LONG     m_nCETAcceleratorViewUpdate;

        /// <summary>   The number of state transitions triggered when Buffers are being allocated for a
        ///             block.
        ///             </summary>
        static LONG     m_nCETBufferAllocate;

        /// <summary>   The number of state transitions triggered when a request to grow the buffer
        ///             caused some buffer reallocation and potentially view updates as a side effect.
        ///             </summary>
        static LONG     m_nCETGrowBuffer;

        /// <summary>   The number of state transitions triggered when a request to synthesize 
        /// 			a metadata block caused the traffic </summary>
        static LONG     m_nCETSynthesizeBlock;

        /// <summary>   The number of state transitions triggered when 
        /// 			needed a pinned host buffer in addition to a dev buffer </summary>
        static LONG     m_nCETPinnedHostView;

        /// <summary>   Is the profiler initialised? </summary>
        static LONG     m_nCoherenceProfilerInit;

        /// <summary>   Is the profiler enabled? </summary>
        static LONG     m_nCoherenceProfilerEnabled;

        /// <summary>   true if the coherence tracker should emit copious text. </summary>
        static BOOL     m_bCoherenceProfilerVerbose;

        /// <summary>   The detailed statistics. </summary>
        static BOOL     m_bCoherenceStatisticsDetailed;

        /// <summary>   The per task histories. </summary>
        static std::map<UINT, COHERENCEHISTORY*> m_vHistories;

        /// <summary>   The timer. </summary>
        static CHighResolutionTimer * m_pTimer;

        /// <summary>   List of names task names. Required because we will no longer have
        /// 			valid task pointers when we check for leaks (all tasks *should* be
        /// 			deleted by that point), and we want to be able to find the task
        /// 			that allocated a block if it was leaked and provide it's name as
        /// 			a debug assist.   
        /// 			</summary>
        static std::map<PTask::Task*, std::string> m_vTaskNames;

        /// <summary>   List of port names. Required because we will no longer have
        /// 			valid port pointers when we check for leaks (all ports *should* be
        /// 			deleted by that point), and we want to be able to find the last
        /// 		    port that touched any leaked blocks. 
        /// 			</summary>
        static std::map<PTask::Port*, std::string> m_vPortNames;

        /// <summary>   The coherence profiler lock. Protects the static data structures
        /// 			collecting data xfer statistics.
        /// 			</summary>
        static CRITICAL_SECTION m_csCoherenceProfiler;
      
        friend class Datablock;

    };

};
#endif
