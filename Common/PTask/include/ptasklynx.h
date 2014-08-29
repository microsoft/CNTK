///-------------------------------------------------------------------------------------------------
// file:	ptasklynx.h
//
// summary:	Declares the lynx conditional compilation macros
///-------------------------------------------------------------------------------------------------

#ifndef __PTASK_LYNX_H__
#define __PTASK_LYNX_H___

#ifdef PTASK_LYNX_INSTRUMENTATION
#include "lynx.h" 
#define init_task_code_instrumentation(x)     (x)->InitializeInstrumentation()
#define finalize_task_code_instrumentation(x) (x)->FinalizeInstrumentation()
#else
#define init_task_code_instrumentation(x)     
#define finalize_task_code_instrumentation(x) 
#endif
#ifdef REPORT_TIMING 
#include "shrperft.h"
#define ptasklynx_start_timer() \
    CSharedPerformanceTimer * timer = new CSharedPerformanceTimer(gran_msec, true); \
    double start = timer->elapsed(false);
#define ptasklynx_stop_timer()                                        \
        error = cuCtxSynchronize();                                   \
        PTASSERT(error == CUDA_SUCCESS);                              \
        double end = timer->elapsed(false);                           \
        double runtime = end - start;                                 \
        std::cout << m_lpszTaskName << "\t" << runtime << std::endl;  \
        delete timer;                                                 
#else 
#define ptasklynx_start_timer()
#define ptasklynx_stop_timer()
#endif
#endif
