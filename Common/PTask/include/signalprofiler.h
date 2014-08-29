///-------------------------------------------------------------------------------------------------
// file:	signalprofiler.h
//
// summary:	Declares the signalprofiler class
///-------------------------------------------------------------------------------------------------

#ifndef __SIGNAL_PROFILER_H__
#define __SIGNAL_PROFILER_H__

#include "primitive_types.h"
#include "channel.h"
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include "Lockable.h"
#include <assert.h>
#include "datablock.h"
#include "task.h"
#include "port.h"

class CHighResolutionTimer;
class CSharedPerformanceTimer;

namespace PTask {

    class ReferenceCounted;
    class Task;
    class Port;
    class Channel;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Defines an alias representing the sigevttype. </summary>
    ///
    /// <remarks>   crossbac, 6/30/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum sigevttype { 
        SIGEVT_UNSPECIFIED=0,
        SIGEVT_INGRESS=1,
        SIGEVT_EGRESS=2
    } SIGEVTTYPE;

    static const char * g_lpszSigEventTypeStrings[] = {
        "SIGEVT_UNSPECIFIED",
        "SIGEVT_INGRESS",
        "SIGEVT_EGRESS"
    };
    #define SigEventTypeString(e) (g_lpszSigEventTypeStrings[(int)e])

    typedef enum witnesstype_t {
        wtport,
        wttask,
        wtchannel,
        wtunknown
    } WITNESSTYPE;

    typedef enum channelsigactivitystate_t {
        cas_none=0,
        cas_unexercised=1,
        cas_exercised=2
    } CHANNELACTIVITYSTATE;

    typedef enum channelpredicationstate_t {
        cps_na=0,
        cps_open=1,
        cps_closed=2
    } CHANNELPREDICATIONSTATE;

    typedef struct SignalObservation_t {
        SIGEVTTYPE eType;
        double dTimestamp;
        Lockable * pWitness;
        WITNESSTYPE wType;
        CONTROLSIGNAL luiRawSignal;
        Datablock * pBlock;
        UINT uiDBUID;
        BOOL bTookRef;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Signal observation t. </summary>
        ///
        /// <remarks>   crossbac, 6/30/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        struct SignalObservation_t() : 
            eType(SIGEVT_UNSPECIFIED),
            dTimestamp(0.0),
            pWitness(NULL),
            luiRawSignal(0),
            pBlock(NULL),
            uiDBUID(0),
            bTookRef(FALSE),
            wType(wtunknown) {}

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Signal observation t. </summary>
        ///
        /// <remarks>   crossbac, 6/30/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        SignalObservation_t::Initialize(
            __in SIGEVTTYPE _eType,
            __in double _dTimestamp,
            __in Lockable * _pWitness,
            __in WITNESSTYPE _wType,
            __in CONTROLSIGNAL _luiSignal,
            __in Datablock * _pBlock,
            __in UINT _uiDBUID,
            __in BOOL _bTakeRef
            ) 
        {
            eType = _eType;
            dTimestamp = _dTimestamp;
            pWitness = _pWitness;
            luiRawSignal = _luiSignal;
            pBlock = _pBlock;
            wType = _wType; 
            uiDBUID = _uiDBUID;
            bTookRef = _bTakeRef;
            if(_bTakeRef) 
                pBlock->AddRef(); 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets witness type. </summary>
        ///
        /// <remarks>   crossbac, 6/30/2014. </remarks>
        ///
        /// <param name="pWitness"> [in,out] If non-null, the witness. </param>
        ///
        /// <returns>   The witness type. </returns>
        ///-------------------------------------------------------------------------------------------------

        static WITNESSTYPE 
        GetWitnessType(
            Lockable* pWitness
            ) {
            Channel * pChannel = dynamic_cast<Channel*>(pWitness);
            Port * pPort = dynamic_cast<Port*>(pWitness);
            Task * pTask = dynamic_cast<Task*>(pWitness);
            int nVPointerCount = 0;
            nVPointerCount += pChannel ? 1 : 0;
            nVPointerCount += pPort ? 1 : 0;
            nVPointerCount += pTask ? 1 : 0;
            assert(nVPointerCount == 1);
            return pChannel ? wtchannel : (pPort ? wtport : (pTask ? wttask : wtunknown));
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets witness type. </summary>
        ///
        /// <remarks>   crossbac, 6/30/2014. </remarks>
        ///
        /// <param name="pWitness"> [in,out] If non-null, the witness. </param>
        ///
        /// <returns>   The witness type. </returns>
        ///-------------------------------------------------------------------------------------------------

        static char *
        GetWitnessName(
            Lockable* pWitness
            ) {
            Channel * pChannel = dynamic_cast<Channel*>(pWitness);
            Port * pPort = dynamic_cast<Port*>(pWitness);
            Task * pTask = dynamic_cast<Task*>(pWitness);
            int nVPointerCount = 0;
            nVPointerCount += pChannel ? 1 : 0;
            nVPointerCount += pPort ? 1 : 0;
            nVPointerCount += pTask ? 1 : 0;
            assert(nVPointerCount == 1);
            return pChannel ? pChannel->GetName() : (pPort ? pPort->GetVariableBinding() : (pTask ? pTask->GetTaskName() : "wtunknown"));
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Signal observation t. </summary>
        ///
        /// <remarks>   crossbac, 6/30/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        struct SignalObservation_t(
            __in SIGEVTTYPE _eType,
            __in double _dTimestamp,
            __in Lockable * _pWitness,
            __in Datablock * _pBlock,
            __in BOOL _bTakeRef=FALSE
            ) 
        {
            CONTROLSIGNAL _luiSignal = _pBlock ? _pBlock->__getControlSignals() : DBCTLC_NONE;
            UINT _uiDBUID = _pBlock ? _pBlock->GetDBUID() : 0;
            WITNESSTYPE _wType = GetWitnessType(_pWitness);
            Initialize(_eType, _dTimestamp, _pWitness, _wType, _luiSignal, _pBlock, _uiDBUID, _bTakeRef); 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   crossbac, 6/30/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        ~SignalObservation_t() {
            if(bTookRef && pBlock) 
                pBlock->Release();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Stream insertion operator. </summary>
        ///
        /// <remarks>   crossbac, 6/30/2014. </remarks>
        ///
        /// <param name="os">           [in,out] The operating system. </param>
        /// <param name="pObservation"> [in,out] If non-null, the observation. </param>
        ///
        /// <returns>   The shifted result. </returns>
        ///-------------------------------------------------------------------------------------------------

        friend std::ostream& operator<<(
            std::ostream &os, 
            SignalObservation_t* pObservation
            ) 
        {
            os  << pObservation->dTimestamp << ": " 
                << ControlSignalString(pObservation->luiRawSignal) << " "
                << SigEventTypeString(pObservation->eType) << " DB#" 
                << pObservation->uiDBUID << " "  
                << GetWitnessName(pObservation->pWitness);
            return os;
        }

    } SIGOBSERVATION; 

    class SignalProfiler {

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'luiControlSignal' is under profile. </summary>
        ///
        /// <remarks>   crossbac, 6/27/2014. </remarks>
        ///
        /// <param name="luiControlSignal"> The lui control signal. </param>
        ///
        /// <returns>   true if under profile, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        IsUnderProfile(
            __in CONTROLSIGNAL luiControlSignal
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if control signals on this block are under profile. </summary>
        ///
        /// <remarks>   crossbac, 6/27/2014. </remarks>
        ///
        /// <param name="luiControlSignal"> The lui control signal. </param>
        ///
        /// <returns>   true if under profile, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        IsUnderProfile(
            __in Datablock * pBlock
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Registers the signal as being one "of interest" to the profiler. </summary>
        ///
        /// <remarks>   crossbac, 6/27/2014. </remarks>
        ///
        /// <param name="luiControlSignal"> The lui control signal. </param>
        /// <param name="bEnable">          (Optional) the enable. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        RegisterSignal(
            __in CONTROLSIGNAL luiControlSignal,
            __in BOOL bEnable=TRUE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes control signal profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Initialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deinitialize control signal profiling. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void Deinitialize();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps profile statistics. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <param name="ss">   [in,out] The ss. </param>
        ///-------------------------------------------------------------------------------------------------

        static void Report(std::ostream& ss);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets signal history for a particular graph object. </summary>
        ///
        /// <remarks>   Crossbac, 7/17/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the task dispatch history. </returns>
        ///-------------------------------------------------------------------------------------------------

        static std::stringstream* GetHistory();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record signal transit. </summary>
        ///
        /// <remarks>   crossbac, 6/27/2014. </remarks>
        ///
        /// <param name="pWitness">         [in,out] If non-null, the witness. </param>
        /// <param name="pBlock">           [in,out] The lui control signal. </param>
        /// <param name="eSigEventType">    Type of the signal event. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        RecordSignalTransit(
            __in Lockable * pWitness,
            __in Datablock * pBlock,
            __in SIGEVTTYPE eSigEventType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        SignalTrafficOccurred(
            __in Lockable * pWitness, 
            __in CONTROLSIGNAL luiControlSignal,
            __in SIGEVTTYPE eType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        BalancedSignalTrafficOccurred(
            __in Lockable * pWitness, 
            __in CONTROLSIGNAL luiControlSignal
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        SuppressedSignalTrafficOccurred(
            __in Lockable * pWitness, 
            __in CONTROLSIGNAL luiControlSignal
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        ProfiledSignalTrafficOccurred(
            __in Lockable * pWitness, 
            __in SIGEVTTYPE eType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        AnyProfiledSignalTrafficOccurred(
            __in Lockable * pWitness
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        SignalIngressOccurred(
            __in Lockable * pWitness, 
            __in CONTROLSIGNAL luiControlSignal
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        SignalEgressOccurred(
            __in Lockable * pWitness, 
            __in CONTROLSIGNAL luiControlSignal
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        ProfiledSignalIngressOccurred(
            __in Lockable * pWitness
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   return true if the given graph object ever bore witness to
        ///             the given control signal. </summary>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        ProfiledSignalEgressOccurred(
            __in Lockable * pWitness
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Profiled signal transit suppressed. </summary>
        ///
        /// <remarks>   crossbac, 6/27/2014. </remarks>
        ///
        /// <param name="pWitness"> [in,out] If non-null, the witness. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        BalancedProfiledSignalTrafficOccurred(
            __in Lockable * pWitness
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Profiled signal transit suppressed. </summary>
        ///
        /// <remarks>   crossbac, 6/27/2014. </remarks>
        ///
        /// <param name="pWitness"> [in,out] If non-null, the witness. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        SuppressedProfiledSignalTrafficOccurred(
            __in Lockable * pWitness
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'pWitness' has relevant predicate. </summary>
        ///
        /// <remarks>   crossbac, 6/27/2014. </remarks>
        ///
        /// <param name="pWitness"> [in,out] If non-null, the witness. </param>
        ///
        /// <returns>   true if relevant predicate, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static BOOL 
        HasRelevantPredicate(
            __in Lockable * pWitness
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets signal activity state. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="pLockable">    [in,out] If non-null, the lockable. </param>
        ///
        /// <returns>   The signal activity state. </returns>
        ///-------------------------------------------------------------------------------------------------

        static CHANNELACTIVITYSTATE 
        GetSignalActivityState(
            __in Lockable * pLockable
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets channel signal predication state. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="pLockable">    [in,out] If non-null, the lockable. </param>
        ///
        /// <returns>   The channel signal predication state. </returns>
        ///-------------------------------------------------------------------------------------------------

        static CHANNELPREDICATIONSTATE
        GetChannelSignalPredicationState(
            __in Lockable * pLockable
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets channel coded color. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="eActivityState">       State of the activity. </param>
        /// <param name="ePredicationState">    State of the predication. </param>
        ///
        /// <returns>   null if it fails, else the channel coded color. </returns>
        ///-------------------------------------------------------------------------------------------------

        static char * 
        GetChannelCodedColor(
            __in CHANNELACTIVITYSTATE eActivityState, 
            __in CHANNELPREDICATIONSTATE ePredicationState
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets channel coded name. </summary>
        ///
        /// <remarks>   crossbac, 7/1/2014. </remarks>
        ///
        /// <param name="pLockable">    [in,out] If non-null, the lockable. </param>
        /// <param name="bBlocked">     The blocked. </param>
        ///
        /// <returns>   The channel coded name. </returns>
        ///-------------------------------------------------------------------------------------------------

        static std::string
        GetChannelCodedName(
            __in Lockable * pLockable,
            __in BOOL bBlocked
            );

        /// <summary>   true if signal profiler is initialized. </summary>
        static BOOL     s_bSignalProfilerInit;

    protected:

        static BOOL IsRelevantPredicate(CHANNELPREDICATE ePredicate);

    #ifdef PROFILE_CONTROLSIGNALS

        /// <summary>   The control signal history. key is the raw control signal value
        ///             number, value is a vector of timestamps at which the raw signal was observed.
        ///             Since a raw signal may be the bitwise or multiple individual signals, we also
        ///             maintain a map for bitwise signal values.
        ///             </summary>
        static std::map<Lockable*, std::set<SIGOBSERVATION*>>                       s_vWitnessToSignalMap;
        static std::map<CONTROLSIGNAL, std::set<SIGOBSERVATION*>>                   s_vSignalToWitnessMap;
        static std::map<double, std::set<SIGOBSERVATION*>>                          s_vSignalHistory;
        static BOOL                                                                 s_bFilterProfiledSignals;
        static CONTROLSIGNAL                                                        s_luiSignalsOfInterest;
        static CRITICAL_SECTION                                                     s_csSignalProfiler;
        static CSharedPerformanceTimer *                                            s_pGlobalProfileTimer;
        static char *                                                               s_lpszChannelColors[3][3];
        static void Lock();
        static void Unlock();
        static BOOL IsLocked(); 
        static BOOL Enabled();

        #pragma warning(disable:4127)
        #define ctlpon()                    (PTask::Runtime::GetControlSignalProfileMode()&&(s_bSignalProfile!=NULL))
        #define ctlptimer()                 (s_pGlobalProfileTimer)
        #define ctlpdeclegressctr()         UINT uiEgressCounter = 0
        #define ctlpingress(l,b)            SignalProfiler::RecordSignalTransit((l), (b), SIGEVTTYPE::SIGEVT_INGRESS)
        #define ctlpegress(l,b)             SignalProfiler::RecordSignalTransit((l), (b), SIGEVTTYPE::SIGEVT_EGRESS)
        #define ctlpopegress(l,b)           { ctlpcondegress(uiEgressCounter == 0, l, b); uiEgressCounter++; }
        #define ctlpcondingress(c,l,b)      if(c) { ctlpingress((l),(b)); }
        #define ctlpcondegress(c,l,b)       if(c) { ctlpegress((l),(b)); }
        #define ctlpwasactive(x)            SignalProfiler::AnyProfiledSignalTrafficOccurred(x)
        #define ctlpwasbalanced(x)          SignalProfiler::BalancedProfiledSignalTrafficOccurred((x))    
        #define ctlpwassuppresed(x)         SignalProfiler::SuppressedProfiledSignalTrafficOccurred((x))    
        #define ctlphasrelevantpredicate(x) SignalProfiler::HasRelevantPredicate(x)
        #define ctlpgetchactstate(x)        SignalProfiler::GetSignalActivityState(x)
        #define ctlpgetchpredstate(x)       SignalProfiler::GetChannelSignalPredicationState(x)
        #define ctlpgetchcolor(x,y)         SignalProfiler::GetChannelCodedColor((x),(y))
        #define ctlpgetchname(x,y)          SignalProfiler::GetChannelCodedName((x),(y))
        #else
        #define ctlpon()                    
        #define ctlptimer()                 
        #define ctlpdeclegressctr()         
        #define ctlpingress(l,b)            
        #define ctlpegress(l,b)             
        #define ctlpopegress(l,b)           
        #define ctlpcondingress(c,l,b)      
        #define ctlpcondegress(c,l,b)       
        #define ctlpwasactive(x)             FALSE
        #define ctlpwasbalanced(x)           FALSE
        #define ctlpwassuppresed(x)          FALSE
        #define ctlphasrelevantpredicate(x)  FALSE
        #define ctlpgetchactstate(x)         cas_none
        #define ctlpgetchpredstate(x)        cps_na
        #define ctlpgetchcolor(x,y)          "gray60"
        #define ctlpgetchname(x,y)           "channel"
    #endif

    };

};
#endif