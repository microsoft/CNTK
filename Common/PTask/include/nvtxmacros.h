///-------------------------------------------------------------------------------------------------
// file:	nvtxmacros.h
//
// summary:	Declares the nvtxmacros class
///-------------------------------------------------------------------------------------------------

#ifndef __NVTX_MACROS_H__
#define __NVTX_MACROS_H__



#if defined(NVPROFILE) && defined(CUDA_SUPPORT)
#include "nvToolsExt.h"

extern BOOL gbnvtxldok;
extern BOOL gbvntxinit;

#define DECLARE_NVTX_GLOBALS() \
BOOL gbnvtxldok = FALSE; \
BOOL gbvntxinit = FALSE; 

#define INITNVTX()        initnvtx()
#define MARKEVENT(x)      if(gbnvtxldok) nvtxMark(x)
#define NAMETHREAD(x)     if(gbnvtxldok) nvtxNameOsThread(GetCurrentThreadId(),(x))
#define MARKRANGEENTER(x) if(gbnvtxldok) nvtxRangePush(x)
#define MARKRANGEEXIT()   if(gbnvtxldok) nvtxRangePop()
#define MARKTASKENTER(x)  if(gbnvtxldok) nvtxRangePushA(x)
#define MARKTASKEXIT()    if(gbnvtxldok) nvtxRangePop()

#define DECLARE_NVTX_INIT()                                   \
void initnvtx() {                                             \
    if(!gbvntxinit) {                                         \
        gbnvtxldok = FALSE;                                   \
        HANDLE hNVTXlib = LoadLibrary(L"nvToolsExt64_1.dll"); \
        if(hNVTXlib != NULL) {                                \
            MARKEVENT(L"initnvtx");                           \
            gbnvtxldok = TRUE;                                \
        }                                                     \
        gbvntxinit = TRUE;                                    \
    }                                                         \
}

#else
#define DECLARE_NVTX_GLOBALS() 
#define DECLARE_NVTX_INIT()    
#define INITNVTX()
#define MARKEVENT(x)      
#define NAMETHREAD(x)     
#define MARKRANGEENTER(x) 
#define MARKRANGEEXIT()
#define MARKTASKENTER(x)
#define MARKTASKEXIT()
#endif
#endif
