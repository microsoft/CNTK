///-------------------------------------------------------------------------------------------------
// file:	extremetrace.h
//
// summary:	Macros for extreme trace mode
///-------------------------------------------------------------------------------------------------

#ifndef __EXTREME_TRACE_H__
#define __EXTREME_TRACE_H__

#ifdef EXTREME_TRACE
#include "PTaskRuntime.h"
#define MSGSIZE 256
#define trace(x) \
    if(PTask::Runtime::g_bExtremeTrace) {\
      char szMsg[MSGSIZE];\
      sprintf_s(szMsg, MSGSIZE, "%s\n", x);\
      printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define trace2(x, y) \
    if(PTask::Runtime::g_bExtremeTrace) {\
      char szMsg[MSGSIZE];\
      sprintf_s(szMsg, MSGSIZE, x, y);\
      printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define trace3(x, y, z) \
    if(PTask::Runtime::g_bExtremeTrace) {\
      char szMsg[MSGSIZE];\
      sprintf_s(szMsg, MSGSIZE, x, y, z);\
      printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define trace4(x, y, z, w) \
    if(PTask::Runtime::g_bExtremeTrace) {\
      char szMsg[MSGSIZE];\
      sprintf_s(szMsg, MSGSIZE, x, y, z, w);\
      printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define trace5(x, y, z, w, u) \
    if(PTask::Runtime::g_bExtremeTrace) {\
      char szMsg[MSGSIZE];\
      sprintf_s(szMsg, MSGSIZE, x, y, z, w, u);\
      printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define trace6(x, y, z, w, u, t) \
    if(PTask::Runtime::g_bExtremeTrace) {\
      char szMsg[MSGSIZE];\
      sprintf_s(szMsg, MSGSIZE, x, y, z, w, u, t);\
      printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define trace8(x, y, z, w, u, t, r) \
    if(PTask::Runtime::g_bExtremeTrace) {\
      char szMsg[MSGSIZE];\
      sprintf_s(szMsg, MSGSIZE, x, y, z, w, u, t, r);\
      printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#else
#define trace(x)
#define trace2(x, y) 
#define trace3(x, y, z) 
#define trace4(x, y, z, w) 
#define trace5(x, y, z, w, u) 
#define trace6(x, y, z, w, u, v) 
#define trace7(x, y, z, w, u, v, r) 
#define trace8(x, y, z, w, u, v, r, s) 
#endif

#endif
