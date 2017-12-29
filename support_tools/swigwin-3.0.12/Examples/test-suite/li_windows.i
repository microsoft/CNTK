%module li_windows

%include "windows.i"

%{
#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
  // Fix Tcl.h and Windows.h cat and mouse over definition of VOID
  #if defined(_TCL) && defined(__CYGWIN__)
    #ifdef VOID
      #undef VOID
    #endif
  #endif
  #include <windows.h>
#else
  // Use equivalent types for non-windows systems
  #define __int8   char
  #define __int16  short
  #define __int32  int
  #define __int64  long long
#endif
%}

%inline %{
// Non ISO integral types
         __int8   int8_val (         __int8  i) { return i; }
         __int16  int16_val(         __int16 i) { return i; }
         __int32  int32_val(         __int32 i) { return i; }
         __int64  int64_val(         __int64 i) { return i; }
unsigned __int8  uint8_val (unsigned __int8  i) { return i; }
unsigned __int16 uint16_val(unsigned __int16 i) { return i; }
unsigned __int32 uint32_val(unsigned __int32 i) { return i; }
unsigned __int64 uint64_val(unsigned __int64 i) { return i; }

const          __int8&   int8_ref (const          __int8&  i) { return i; }
const          __int16&  int16_ref(const          __int16& i) { return i; }
const          __int32&  int32_ref(const          __int32& i) { return i; }
const          __int64&  int64_ref(const          __int64& i) { return i; }
const unsigned __int8&  uint8_ref (const unsigned __int8&  i) { return i; }
const unsigned __int16& uint16_ref(const unsigned __int16& i) { return i; }
const unsigned __int32& uint32_ref(const unsigned __int32& i) { return i; }
const unsigned __int64& uint64_ref(const unsigned __int64& i) { return i; }

         __int8   int8_global;
         __int16  int16_global;
         __int32  int32_global;
         __int64  int64_global;
unsigned __int8  uint8_global;
unsigned __int16 uint16_global;
unsigned __int32 uint32_global;
unsigned __int64 uint64_global;

struct WindowsInts {
           __int8   int8_member;
           __int16  int16_member;
           __int32  int32_member;
           __int64  int64_member;
  unsigned __int8  uint8_member;
  unsigned __int16 uint16_member;
  unsigned __int32 uint32_member;
  unsigned __int64 uint64_member;
};

// Typedef for non ISO integral types
typedef __int8 int8;
typedef __int16 int16;
typedef __int32 int32;
typedef __int64 int64;

typedef unsigned __int8 uint8;
typedef unsigned __int16 uint16;
typedef unsigned __int32 uint32;
typedef unsigned __int64 uint64;

 int8   int8_td (int8  i) { return i; }
 int16  int16_td(int16 i) { return i; }
 int32  int32_td(int32 i) { return i; }
 int64  int64_td(int64 i) { return i; }
uint8  uint8_td (int8  i) { return i; }
uint16 uint16_td(int16 i) { return i; }
uint32 uint32_td(int32 i) { return i; }
uint64 uint64_td(int64 i) { return i; }

%}

// Windows calling conventions and some types in windows.h
%inline %{
#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#else

#define __stdcall
#define __declspec(WINDOWS_EXTENDED_ATTRIBUTE)
#define DWORD unsigned int
#define PSZ char *

#endif

// Windows calling conventions
__declspec(dllexport) int __stdcall declspecstdcall(int i) { return i; }

DWORD mefod(DWORD d) { return d; }
PSZ funktion(PSZ d) { return d; }
%}

