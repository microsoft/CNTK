// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the PTASKHOST_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// PTASKHOST_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef _WIN32
#ifdef PTASKHOST_EXPORTS
#define PTASKHOST_API __declspec(dllexport)
#else
#define PTASKHOST_API __declspec(dllimport)
#endif
#else
#define PTASKHOST_API
#endif

#include "PTask.h"

#ifdef USE_PTASK
extern "C" {

PTASKHOST_API void __stdcall HostTask(LPDEPENDENTCONTEXT depContext);

}
#else
extern "C" {

PTASKHOST_API void __stdcall DummyFunction();

}
#endif
