// dllmain.cpp -- main entry point for the DLL
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/dllmain.cpp $
// 
// 9     11/18/11 10:07a Fseide
// (oops, corrected last checkin)
// 
// 8     11/18/11 10:06a Fseide
// now unbuffering stderr, in case we pick up a redirection to file
// 
// 7     1/31/11 2:45p Fseide
// moved implementation of cudamatrix out from here, now only contains
// DllMain
// 
// 6     1/31/11 11:49a Fseide
// added test()
// 
// 5     1/30/11 11:44p Fseide
// added getnumdevices() implementation
// 
// 4     1/30/11 11:30p Fseide
// added official API entry point
// 
// 3     1/30/11 11:19p Fseide
// changed to DLL-export cudamatrix instead of cudalib
// 
// 2     1/30/11 11:07p Fseide
// 
// 1     1/30/11 10:51p Fseide

#include <Windows.h>
#include <stdio.h>
BOOL APIENTRY DllMain (HMODULE, DWORD, LPVOID) { setvbuf (stderr, NULL, _IONBF, 0)/*unbuffer stderr*/; return TRUE; }
