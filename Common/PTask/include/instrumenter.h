///-------------------------------------------------------------------------------------------------
// file:	instrumenter.h
//
// summary:	Declares the instrumenter class
///-------------------------------------------------------------------------------------------------

#ifndef __PTASK_INSTRUMENTATION_H__
#define __PTASK_INSTRUMENTATION_H__

#include "primitive_types.h"
#include "Lockable.h"
#include <stack>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <tuple>

class CSharedPerformanceTimer;

namespace PTask {

    class Instrumenter : public Lockable 
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initialize an the ad hoc instrumentation framework. Creates a singleton
        ///             instrumenter object.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Initialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Shutdown the ad hoc instrumentation framework, destroys the singleton
        ///             instrumenter object.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Destroy();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports all measured latencies and acknowledges any outstanding
        ///             (incomplete) measurments . </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="ss">   [in,out] The ss. </param>
        ///-------------------------------------------------------------------------------------------------

        static void Report(std::ostream& ss);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables the instrumentation framework. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="bEnable">  true to enable, false to disable. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL Enable(BOOL bEnable);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the adhoc instrumentation framework is enabled. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <returns>   true if enabled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL IsEnabled();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if a measurement matching 'strEventName' is in flight. In flight
        ///             means that a start sentinal has been pushed onto the outstanding stack
        ///             that has not been matched yet by a corresponding completion. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   true if in flight, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL IsInFlight(std::string& strEventName);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Collect data point. </summary>
		///
		/// <remarks>	crossbac, 8/12/2013. </remarks>
		///
		/// <param name="strEventName">	   	[in,out] Name of the event. </param>
		///
		/// <returns>	. </returns>
		///-------------------------------------------------------------------------------------------------

		static double CollectDataPoint(std::string& strEventName);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Collect data point. </summary>
		///
		/// <remarks>	crossbac, 8/12/2013. </remarks>
		///
		/// <param name="strEventName">	   	[in,out] Name of the event. </param>
		///
		/// <returns>	. </returns>
		///-------------------------------------------------------------------------------------------------

		static double 
        CollectDataPoint(
            __in  std::string& strEventName, 
            __out UINT &nSamples, 
            __out double &dMin, 
            __out double &dMax
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if a measurement matching 'strEventName' is complete. Note that
        ///             because multiple measurements matching a given name can be tracked, it is 
        ///             possible for an event name to be both "in flight" and complete. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   true if complete, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL IsComplete(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets nesting depth for the given event name. If the nest depth is 0 it means
        ///             there are no measurements with the given name in flight. A depth greater than 1
        ///             means there is a nested measurement with the same name. This idiom is likely best
        ///             avoided in potentially concurrent code, since the instrumenter handles nesting
        ///             with a stack, which makes it difficult to disambiguate end sentinels if they are
        ///             not ordered explicitly by the program.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   The nesting depth. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT GetNestingDepth(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event start. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordEventStart(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event complete. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordEventComplete(std::string& strEventName);
         
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event start for an event that should have only one start sentinel,
        ///             but for which concurrency implies non-determinism, so many threads may attempt
        ///             to record the same event start. The primary example of this scenario is
        ///             start of data processing in PTask, which occurs as soon as the first block
        ///             is pushed by the user. It is simplest to record this by calling the instrumenter
        ///             on every exposed call to Channel::Push, with all calls after the first ignored. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordSingletonEventStart(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Record event complete an event that should have only one start sentinel, but for
        /// 			which concurrency implies non-determinism, so many threads may attempt to record
        /// 			the same event start. The primary example of this scenario is start of data
        /// 			processing in PTask, which occurs as soon as the first block is pushed by the
        /// 			user. It is simplest to record this by calling the instrumenter on every exposed
        /// 			call to Channel::Push, with all calls after the first ignored.
        /// 			</summary>
        ///
        /// <remarks>	Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName">		  	[in,out] Name of the event. </param>
        /// <param name="bRequireOutstanding">	(Optional) true to require an outstanding entry. Some
        /// 									stats (like first return-value materialization)
        /// 									are very difficult to capture unambiguously, because
        /// 									calls to record the event must be placed in common code
        /// 									paths. Calling with this parameter set to true allows the
        /// 									record call to fail without protest if the caller knows
        /// 									this to be such an event. </param>
        ///
        /// <returns>	the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordSingletonEventComplete(std::string& strEventName, BOOL bRequireOutstanding=TRUE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event start. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordEventStart(char * strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event complete. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordEventComplete(char * strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Increment externally measured latency for a cumulative event. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        /// <param name="dIncrement">   Amount to increment by. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT AccumulateEventLatency(char * strEventName, double dIncrement);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record start for a cumulative event. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordCumulativeEventStart(char * strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record cumulative event complete. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordCumulativeEventComplete(char * strEventName);
         
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event start for an event that should have only one start sentinel,
        ///             but for which concurrency implies non-determinism, so many threads may attempt
        ///             to record the same event start. The primary example of this scenario is
        ///             start of data processing in PTask, which occurs as soon as the first block
        ///             is pushed by the user. It is simplest to record this by calling the instrumenter
        ///             on every exposed call to Channel::Push, with all calls after the first ignored. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordSingletonEventStart(char * strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Record event complete an event that should have only one start sentinel, but for
        /// 			which concurrency implies non-determinism, so many threads may attempt to record
        /// 			the same event start. The primary example of this scenario is start of data
        /// 			processing in PTask, which occurs as soon as the first block is pushed by the
        /// 			user. It is simplest to record this by calling the instrumenter on every exposed
        /// 			call to Channel::Push, with all calls after the first ignored.
        /// 			</summary>
        ///
        /// <remarks>	Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName">		  	[in,out] Name of the event. </param>
        /// <param name="bRequireOutstanding">	(Optional) true to require an outstanding entry. Some
        /// 									stats (like first return-value materialization)
        /// 									are very difficult to capture unambiguously, because
        /// 									calls to record the event must be placed in common code
        /// 									paths. Calling with this parameter set to true allows the
        /// 									record call to fail without protest if the caller knows
        /// 									this to be such an event. </param>
        ///
        /// <returns>	the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static UINT RecordSingletonEventComplete(char * strEventName, BOOL bRequireOutstanding=TRUE);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets this object. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Reset();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        Instrumenter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Instrumenter();

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Collect data point. </summary>
		///
		/// <remarks>	crossbac, 8/12/2013. </remarks>
		///
		/// <param name="strEventName">	   	[in,out] Name of the event. </param>
		///
		/// <returns>	. </returns>
		///-------------------------------------------------------------------------------------------------

		double __CollectDataPoint(std::string& strEventName);

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Collect data point. </summary>
		///
		/// <remarks>	crossbac, 8/12/2013. </remarks>
		///
		/// <param name="strEventName">	   	[in,out] Name of the event. </param>
		///
		/// <returns>	. </returns>
		///-------------------------------------------------------------------------------------------------

		double 
        __CollectDataPoint(
            __in  std::string& strEventName, 
            __out UINT &nSamples, 
            __out double &dMin, 
            __out double &dMax
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports all measured latencies and acknowledges any outstanding
        ///             (incomplete) measurments . </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="ss">   [in,out] The ss. </param>
        ///-------------------------------------------------------------------------------------------------

        void __Report(std::ostream& ss);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports all measured latencies matching the given event name.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="ss">                   [in,out] The ss. </param>
        /// <param name="strEventName">         [in,out] Name of the event. </param>
        ///-------------------------------------------------------------------------------------------------

        void __ReportComplete(std::ostream& ss, std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reports any outstanding (incomplete)
        ///             measurments matching the given event name.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="ss">           [in,out] The ss. </param>
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///-------------------------------------------------------------------------------------------------

        void __ReportOutstanding(std::ostream& ss, std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables the instrumentation framework. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="bEnable">  true to enable, false to disable. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __Enable(BOOL bEnable);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if the adhoc instrumentation framework is enabled. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <returns>   true if enabled, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsEnabled();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if a measurement matching 'strEventName' is in flight. In flight
        ///             means that a start sentinal has been pushed onto the outstanding stack
        ///             that has not been matched yet by a corresponding completion. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   true if in flight, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsInFlight(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if a measurement matching 'strEventName' is complete. Note that
        ///             because multiple measurements matching a given name can be tracked, it is 
        ///             possible for an event name to be both "in flight" and complete. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   true if complete, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL __IsComplete(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets nesting depth for the given event name. If the nest depth is 0 it means
        ///             there are no measurements with the given name in flight. A depth greater than 1
        ///             means there is a nested measurement with the same name. This idiom is likely best
        ///             avoided in potentially concurrent code, since the instrumenter handles nesting
        ///             with a stack, which makes it difficult to disambiguate end sentinels if they are
        ///             not ordered explicitly by the program.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   The nesting depth. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __GetNestingDepth(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event start. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __RecordEventStart(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event complete. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __RecordEventComplete(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event start. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        /// <param name="dIncrement">   Amount to increment by. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __AccumulateEventLatency(std::string& strEventName, double dIncrement);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event start. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __RecordCumulativeEventStart(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event complete. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __RecordCumulativeEventComplete(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record event start. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName"> [in,out] Name of the event. </param>
        ///
        /// <returns>   the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __RecordSingletonEventStart(std::string& strEventName);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Record event complete. </summary>
        ///
        /// <remarks>	Crossbac, 7/23/2013. </remarks>
        ///
        /// <param name="strEventName">		  	[in,out] Name of the event. </param>
        /// <param name="bRequireOutstanding">	true to require outstanding. </param>
        ///
        /// <returns>	the new nesting depth for events matching this name. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT __RecordSingletonEventComplete(std::string& strEventName, BOOL bRequireOutstanding);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets this object. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __Reset();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Finalize singletons. </summary>
        ///
        /// <remarks>   Crossbac, 7/23/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void __FinalizeSingletons();
        
        typedef std::map<std::string, std::tuple<UINT, double, double, double>> CumulativeEventMap;

        BOOL                                        m_bEnabled;
        CSharedPerformanceTimer *                   m_pRTTimer;
        std::map<std::string, std::stack<double>>   m_vOutstanding;
        std::map<std::string, std::vector<double>>  m_vCompleted;
        std::map<std::string, double>               m_vSingletonCompleted;
        CumulativeEventMap                          m_vCumulativeEvents;
        static UINT                                 m_bInitialized;
        static Instrumenter *                       g_pInstrumenter;

    };
};

#ifdef ADHOC_STATS
#define recordGraphDestroyStart()      Instrumenter::RecordEventStart("GraphDestroy")
#define recordGraphDestroyLatency()    Instrumenter::RecordEventComplete("GraphDestroy")
#define recordTeardownStart()          Instrumenter::RecordEventStart("Teardown")
#define recordTeardownLatency()        Instrumenter::RecordEventComplete("Teardown")
#define recordFirstPush()              Instrumenter::RecordSingletonEventStart("ProcessData")
#define recordMaterialize()            Instrumenter::RecordSingletonEventComplete("ProcessData", FALSE)
#define record_dispatch_entry()		   {Instrumenter::RecordSingletonEventStart("DispatchPhase");    Instrumenter::RecordCumulativeEventStart("task-dispatch"); }
#define record_dispatch_exit()		   {Instrumenter::RecordSingletonEventComplete("DispatchPhase"); Instrumenter::RecordCumulativeEventComplete("task-dispatch"); }
#define record_psdispatch_entry()      Instrumenter::RecordCumulativeEventStart("PSDispatch")
#define record_psdispatch_exit()       Instrumenter::RecordCumulativeEventComplete("PSDispatch")
#define record_psdispatch_latency(d)   Instrumenter::AccumulateEventLatency("PSDispatch", d) 
#define record_stream_agg_entry(x)     Instrumenter::RecordCumulativeEventStart("SADispatch");
#define record_stream_agg_exit(x)      Instrumenter::RecordCumulativeEventComplete("SADispatch"); 

#define record_schedule_entry()        
#define record_schedule_exit()         
#define record_wait_acc_entry()        
#define record_wait_acc_exit()         
#define record_sort_q_entry()          
#define record_sort_q_exit()           
//#define record_schedule_entry()        Instrumenter::RecordCumulativeEventStart("Schedule")
//#define record_schedule_exit()         Instrumenter::RecordCumulativeEventComplete("Schedule")
//#define record_wait_acc_entry()        Instrumenter::RecordCumulativeEventStart("block-acc")
//#define record_wait_acc_exit()         Instrumenter::RecordCumulativeEventComplete("block-acc")
//#define record_sort_q_entry()          Instrumenter::RecordCumulativeEventStart("sortq")
//#define record_sort_q_exit()           Instrumenter::RecordCumulativeEventComplete("sortq")
#else
#define recordTeardownStart()                                 
#define recordTeardownLatency()   
#define recordGraphDestroyStart()                    
#define recordGraphDestroyLatency()                    
#define recordFirstPush()
#define recordMaterialize()
#define record_dispatch_entry()		
#define record_dispatch_exit()	
#define record_psdispatch_entry()      
#define record_psdispatch_exit()       
#define record_psdispatch_latency(d)   
#define record_schedule_entry()        
#define record_schedule_exit()        
#define record_wait_acc_entry()       
#define record_wait_acc_exit()        
#define record_sort_q_entry()         
#define record_sort_q_exit()          
#endif

#endif