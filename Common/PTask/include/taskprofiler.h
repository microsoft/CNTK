///-------------------------------------------------------------------------------------------------
// file:	taskprofiler.h
//
// summary:	Declares the taskprofiler class
///-------------------------------------------------------------------------------------------------

#ifndef _TASK_PROFILER_H_
#define _TASK_PROFILER_H_

#include "primitive_types.h"
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>

class CHighResolutionTimer;
class CSharedPerformanceTimer;

namespace PTask {

    class Task;

    class TaskProfile {

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        TaskProfile(Task * pTask);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~TaskProfile();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Print migration stats. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="ss">   [in,out] The ss. </param>
        ///-------------------------------------------------------------------------------------------------

        static void         MigrationReport(std::ostream& ss);

        /// <summary>   The task. </summary>
        Task *                                      m_pTask;

        /// <summary>   The dispatch accelerator history. key is the dispatch
        ///             number, value is the accelerator upon which the dispatch
        ///             took place. </summary>
        std::map<UINT, UINT>                        m_vDispatchAcceleratorHistory;

        /// <summary>   The dependent dispatch accelerator history. key is the dispatch
        ///             number, value is the accelerator used in the dependent binding.
        ///             Note that this object model assumes 1 depacc binding per task, 
        ///             which is less general than what much of the code appears to allow
        ///             (in terms of binding cardinality, heterogeneity), but is in line with
        ///             the defacto limitations on dependent bindings at present.
        ///             </summary>
        std::map<UINT, UINT>                        m_vDependentAcceleratorHistory;

        std::map<std::string, 
                 std::map<int, 
                          std::vector<double>*>&>   m_vEnterProfileMap;
        std::map<std::string, 
                 std::map<int, 
                          std::vector<double>*>&>   m_vExitProfileMap;        
        std::map<int, std::vector<double>*>         m_vEnterAcquireDispatchResourceLocks;
        std::map<int, std::vector<double>*>         m_vEnterReleaseDispatchResourceLocks;
        std::map<int, std::vector<double>*>         m_vEnterMigrateInputs;
        std::map<int, std::vector<double>*>         m_vEnterAssembleIOLockList;
        std::map<int, std::vector<double>*>         m_vEnterSchedule;
        std::map<int, std::vector<double>*>         m_vEnterBlockedOnReadyQ;
        std::map<int, std::vector<double>*>         m_vEnterBlockedNotReady;
        std::map<int, std::vector<double>*>         m_vEnterPropagateDataflow;
        std::map<int, std::vector<double>*>         m_vEnterReleaseInflightDatablocks;
        std::map<int, std::vector<double>*>         m_vEnterRIBMaterializeViews;
        std::map<int, std::vector<double>*>         m_vEnterRIBSyncHost;
        std::map<int, std::vector<double>*>         m_vEnterBindMetaPorts;
        std::map<int, std::vector<double>*>         m_vEnterDispatch;
        std::map<int, std::vector<double>*>         m_vEnterPSDispatch;
        std::map<int, std::vector<double>*>         m_vEnterBindConstants;
        std::map<int, std::vector<double>*>         m_vEnterBindOutputs;
        std::map<int, std::vector<double>*>         m_vEnterBindInputs;
        std::map<int, std::vector<double>*>         m_vEnterAssignDependentAccelerator;
        std::map<int, std::vector<double>*>         m_vEnterDispatchTeardown;
        std::map<int, std::vector<double>*>         m_vExitAcquireDispatchResourceLocks;
        std::map<int, std::vector<double>*>         m_vExitReleaseDispatchResourceLocks;
        std::map<int, std::vector<double>*>         m_vExitMigrateInputs;
        std::map<int, std::vector<double>*>         m_vExitAssembleIOLockList;
        std::map<int, std::vector<double>*>         m_vExitSchedule;
        std::map<int, std::vector<double>*>         m_vExitBlockedOnReadyQ;
        std::map<int, std::vector<double>*>         m_vExitBlockedNotReady;
        std::map<int, std::vector<double>*>         m_vExitPropagateDataflow;
        std::map<int, std::vector<double>*>         m_vExitReleaseInflightDatablocks;
        std::map<int, std::vector<double>*>         m_vExitRIBMaterializeViews;
        std::map<int, std::vector<double>*>         m_vExitRIBSyncHost;
        std::map<int, std::vector<double>*>         m_vExitBindMetaPorts;
        std::map<int, std::vector<double>*>         m_vExitDispatch;
        std::map<int, std::vector<double>*>         m_vExitPSDispatch;
        std::map<int, std::vector<double>*>         m_vExitBindConstants;
        std::map<int, std::vector<double>*>         m_vExitBindOutputs;
        std::map<int, std::vector<double>*>         m_vExitBindInputs;
        std::map<int, std::vector<double>*>         m_vExitAssignDependentAccelerator;
        std::map<int, std::vector<double>*>         m_vExitDispatchTeardown;
        CRITICAL_SECTION                            m_csTiming;
        static UINT                                 m_nMetrics;
        static std::map<std::string, std::string>   m_vMetricNickNames;
        static std::map<UINT, std::string>          m_vMetricOrder;
        static std::stringstream                    m_ssTaskStats;
        static std::stringstream                    m_ssTaskDispatchHistory;
        static CRITICAL_SECTION                     m_csTaskProfiler;
        static BOOL                                 m_bProfilerOutputTabular;
        static BOOL                                 m_bTaskProfilerInit;
        static CSharedPerformanceTimer *            m_pGlobalProfileTimer;
        static ULONG                                m_nInputBindEvents;
        static ULONG                                m_nInputMigrations;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the task profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <param name="bTabular"> true to tabular. </param>
        ///-------------------------------------------------------------------------------------------------

        static void Initialize(BOOL bTabular=TRUE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitialize task profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Deinitialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps a task profile statistics. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <param name="ss">   [in,out] The ss. </param>
        ///-------------------------------------------------------------------------------------------------

        static void Report(std::ostream& ss);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Merge task instance statistics. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void MergeTaskInstanceStatistics();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes the task instance profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void InitializeInstanceProfile();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitialize task instance profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DeinitializeInstanceProfile();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps a task instance profile statistics. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <param name="ss">   [in,out] The ss. </param>
        ///-------------------------------------------------------------------------------------------------

        void DumpTaskProfile(std::ostream& ss);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets task dispatch history. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the task dispatch history. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::stringstream* GetDispatchHistory();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets task instance profile statistics. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the task instance profile statistics. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::stringstream* GetTaskProfile();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets task instance profile statistics columnar. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the task instance profile statistics columnar. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::stringstream* GetTaskProfileColumnar();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets task instance profile statistics tabular. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the task instance profile statistics tabular. </returns>
        ///-------------------------------------------------------------------------------------------------

        std::stringstream* GetTaskProfileTabular();


    #if (defined(GRAPH_DIAGNOSTICS) || defined(PROFILE_TASKS))
        #define log_dispacc(x,y,z,b)        { if(m_pTaskProfile) {                                               \
                                                  m_pTaskProfile->m_vDispatchAcceleratorHistory[x] = y;          \
                                                  if(b) { m_pTaskProfile->m_vDependentAcceleratorHistory[x] = z; } } }
        #define PTR_LD                      Runtime::Tracer::LogDispatchEvent
        #define PTR_EN()                    Runtime::GetDispatchTracingEnabled()
        #define dispaccid()                 m_pDispatchAccelerator->GetAcceleratorId()
        #define hasdepacc()                 (GetDependentBindingClassCount()!=0)
        #define depaccid()                  ((hasdepacc())?(m_vDependentAcceleratorAssignments.begin()->second->at(0)->GetAcceleratorId()):0)
        #define log_dispatch(bEnter)        { if(PTR_EN()) { PTR_LD(m_lpszTaskName, (bEnter), dispaccid(), m_nDispatchNumber); } \
                                              if(bEnter)   { log_dispacc(m_nDispatchNumber, dispaccid(), depaccid(), hasdepacc()); } }
        #define log_dispatch_enter()        log_dispatch(TRUE)
        #define log_dispatch_exit()         log_dispatch(FALSE)
    #else
        #define log_dispatch_enter()        
        #define log_dispatch_exit()         
    #endif

    #ifdef PROFILE_TASKS
        #pragma warning(disable:4127)
        #define tpon()              (PTask::Runtime::GetTaskProfileMode()&&(m_pTaskProfile!=NULL))
        #define tptimer()           (m_pTaskProfile->m_pGlobalProfileTimer)
        #define tpqtimer()          (tpon()?tptimer()->elapsed(false):0.0)
        #define tpprofile_enter(x)                                                         \
            double dTPStart_##x = tpqtimer();                                              \
            if(tpon()) {                                                                   \
              std::map<int, std::vector<double>*>::iterator xxmiTP_##x;                    \
              xxmiTP_##x = m_pTaskProfile->m_vEnter##x.find(m_nDispatchNumber);            \
              if(xxmiTP_##x!=m_pTaskProfile->m_vEnter##x.end()) {                          \
                xxmiTP_##x->second->push_back(dTPStart_##x);                               \
              } else {                                                                     \
                std::vector<double>* l = new std::vector<double>();                        \
                l->push_back(dTPStart_##x);                                                \
                m_pTaskProfile->m_vEnter##x[m_nDispatchNumber] = l;                        \
              }}
        #define tpprofile_exit(x)                                                          \
            double dTPExit_##x = tpqtimer();                                               \
            if(tpon()) {                                                                   \
              std::map<int, std::vector<double>*>::iterator xxmiTP_##x;                    \
              xxmiTP_##x = m_pTaskProfile->m_vExit##x.find(m_nDispatchNumber);             \
              if(xxmiTP_##x!=m_pTaskProfile->m_vExit##x.end()) {                           \
                xxmiTP_##x->second->push_back(dTPExit_##x);                                \
              } else {                                                                     \
                std::vector<double>* l = new std::vector<double>();                        \
                l->push_back(dTPExit_##x);                                                 \
                m_pTaskProfile->m_vExit##x[m_nDispatchNumber] = l;                         \
              }}   
        #define tpprofile_destroy(x)                                                       \
            {                                                                              \
              std::map<int, std::vector<double>*>::iterator xxmiTP_##x;                    \
              for(xxmiTP_##x = m_vExit##x.begin();                                         \
                  xxmiTP_##x != m_vExit##x.end();                                          \
                  xxmiTP_##x++) {                                                          \
                  if(xxmiTP_##x->second) {                                                 \
                     delete xxmiTP_##x->second;                                            \
                  }                                                                        \
              }                                                                            \
              for(xxmiTP_##x = m_vEnter##x.begin();                                        \
                  xxmiTP_##x != m_vEnter##x.end();                                         \
                  xxmiTP_##x++) {                                                          \
                  if(xxmiTP_##x->second) {                                                 \
                     delete xxmiTP_##x->second;                                            \
                  }                                                                        \
              }                                                                            \
              m_vEnter##x.clear();                                                         \
              m_vExit##x.clear();                                                          \
            }
        #else
        #define tpprofile_enter(x)
        #define tpprofile_exit(x) 
        #define tpprofile_init_map(x)
        #define tpprofile_init_map_nickname(a,x,y) 
        #define tpprofile_destroy(x)  
        #endif

    };

};
#endif