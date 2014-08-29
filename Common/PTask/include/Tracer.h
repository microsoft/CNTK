///-------------------------------------------------------------------------------------------------
// file:	Tracer.h
//
// summary:	Declares the tracer class
///-------------------------------------------------------------------------------------------------
#ifndef __PTASK_TRACER_H__
#define __PTASK_TRACER_H__
#include <Windows.h>
#include <iostream>
#include <sstream>
#include <wmistr.h>
#include <evntrace.h>
#include "PTaskRuntime.h"

namespace PTask {
namespace Runtime {

	// Dynamically linked etw logging function.
	typedef ULONG (WINAPI *LPETWSETMARK)( HANDLE, LPVOID, ULONG );
#pragma prefast( suppress:__WARNING_ENCODE_GLOBAL_FUNCTION_POINTER, "This call needs to be performant" );
	static LPETWSETMARK gs_pEtwSetMark = NULL ;

#define TRACER_MAX_MSG_LEN 64
	typedef struct _ETW_SET_MARK_INFORMATION {
		ULONG Flag;
		CHAR Mark[TRACER_MAX_MSG_LEN];
	} ETW_SET_MARK_INFORMATION;

	class Tracer
    {
    public:
        Tracer(void);
        virtual ~Tracer(void);
        
        static VOID EtwSetMarkA(char *msg);
        static ULONG LogDispatchEvent(char * lpszTaskName, BOOL bStart, UINT uiAcceleratorId, UINT uiDispatchNumber);
        static ULONG LogBufferSyncEvent(void * pbufferInstance, BOOL bStart, void * parentDatablock, UINT uiAcceleratorId);
    private:
        static VOID Tracer::InitializeETW();
    };

};
};

#endif

