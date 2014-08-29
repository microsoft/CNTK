///-------------------------------------------------------------------------------------------------
// file:	ptprofsupport.h
//
// summary:	macros for dealing with conditionally compiled runtime monitoring modes.
///-------------------------------------------------------------------------------------------------

#ifndef __PTASK_PROFSUPPORT_H__
#define __PTASK_PROFSUPPORT_H__

#ifdef PROFILE_REFCOUNT_OBJECTS
#include <sstream>
#endif

#ifdef PROFILE_PBUFFERS
#include <sstream>
#include "PBuffer.h"
#endif

namespace PTask {

    namespace Runtime {

        extern int g_bTPProfilingSupported;
        extern int g_bRCProfilingSupported;
        extern int g_bDBProfilingSupported;
        extern int g_bCTProfilingSupported;
        extern int g_bPBufferProfilingSupported;
        extern int g_bInvocationCountingSupported;
        extern int g_bBlockPoolProfilingSupported;
        extern int g_bChannelProfilingSupported;
        extern int g_bAdhocInstrumentationSupported;
        extern int g_bSignalProfilingSupported;

    };
};


#ifndef DEBUG
// warn PTask users if a release build supports a profiling mode
// that likely impacts performance (they all pretty much do)
#define WARN_PROFILE_SUPPORT(bSupport, bReqState)                                       \
    if(bSupport && bReqState) {                                                         \
        MandatoryInform("XXXX: PERFORMANCE: Using %s(%d) support in release build!\n",  \
                         __FUNCTION__,                                                  \
                         (bReqState));                                                  \
    }
#else
#define WARN_PROFILE_SUPPORT(bSupport, bReqState) 
#endif

#define SET_PROFILER_MODE(bSupport, bReqState, bTarget)  {                  \
        if(!(bSupport)) {                                                   \
            if(bReqState) {                                                 \
                MandatoryInform("%s(%d) called, not supported in build!\n", \
                                __FUNCTION__,                               \
                                bReqState);                                 \
            }                                                               \
            bTarget = FALSE;                                                \
        } else {                                                            \
            WARN_PROFILE_SUPPORT(bSupport, bReqState);                      \
            bTarget = bReqState;                                            \
        } }                                                                   
            

#endif
