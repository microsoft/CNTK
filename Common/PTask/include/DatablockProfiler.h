///-------------------------------------------------------------------------------------------------
// file:	DatablockProfiler.h
//
// summary:	Declares the datablock profiler class
///-------------------------------------------------------------------------------------------------

#ifndef __DATABLOCK_PROFILER_H__
#define __DATABLOCK_PROFILER_H__

#include "primitive_types.h"
#include "ReferenceCounted.h"
#include <map>
#include <set>

class CHighResolutionTimer;

namespace PTask {

    class Port;
    class Task;
    class BlockPool;
    class BlockPoolOwner;
    class Datablock;

    class DatablockProfiler {
    
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///
        /// <param name="pDatablock">   [in,out] If non-null, the datablock. </param>
        ///-------------------------------------------------------------------------------------------------

        DatablockProfiler(Datablock * pDatablock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~DatablockProfiler();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the datablock profiler. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Initialize(BOOL bEnable, BOOL bVerbose=FALSE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitializes the datablock profiler. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Deinitialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the databasedatablock profiler leaks. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Report(std::ostream& ios); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Profile allocation. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        static void RecordAllocation(Datablock * pBlock);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Profile deletion. </summary>
        ///
        /// <remarks>   Crossbac, 2/24/2012. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        static void RecordDeletion(Datablock*pBlock);

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

        void RecordBinding(Port * pPort, Task * pTask, Port * pIOConsumer);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record pool binding. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void RecordPoolBinding();

        /// <summary>   The set of all ports to which this block has been bound. </summary>
        std::set<Port*> m_vPortBindings;

        /// <summary>   The set of all tasks which have touched this block. </summary>
        std::set<Task*> m_vTaskBindings;

        /// <summary>   The set of pool owners with block pools used to allocate blocks. 
        /// 			Necessary because block pooling can cause blocks to be reused
        /// 			between allocation and deletion. Maintained as a map to string
        ///             since the owner may be deleted by the time we attempt deletion.
        /// 			</summary>
        std::map<BlockPoolOwner*, std::string> m_vPools;

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

    protected:

        Datablock * m_pDatablock;

        /// <summary>   The number of datablock allocations. </summary>
        static LONG m_nDBAllocations;

        /// <summary>   The datablock deletion count. </summary>
        static LONG m_nDBDeletions;

        /// <summary>   The number of clone allocations. </summary>
        static LONG m_nDBCloneAllocations;

        /// <summary>   The number of clone deletions. </summary>
        static LONG m_nDBCloneDeletions;

        /// <summary>   Is the profiler initialised? </summary>
        static LONG m_nDBProfilerInit;

        /// <summary>   Is the profiler initialised? </summary>
        static LONG m_nDBProfilerEnabled;

        /// <summary>   true if the allocation tracker should emit copious text. </summary>
        static BOOL m_bDBProfilerVerbose;

        /// <summary>   The set of datablocks currently allocated but not yet deleted. </summary>
        static std::set<PTask::Datablock*> m_vAllAllocations;

        // these structures are also needed by the coherence profiler.
        // if both compile-time options are selected, then these are already
        // defined by the time we get here

        /// <summary>   The profiler lock. Protects the allocation counts,
        /// 			the allocation set, and the port and task maps.
        /// 			</summary>
        static CRITICAL_SECTION m_csDBProfiler;

        friend class Datablock;

    };

};
#endif
