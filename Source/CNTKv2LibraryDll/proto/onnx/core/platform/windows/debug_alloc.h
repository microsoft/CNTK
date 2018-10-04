//// Copyright (c) Microsoft Corporation. All rights reserved.
//// Licensed under the MIT License.
//
//#pragma once
//#if defined(_DEBUG)
//// TVM need to run with shared CRT, so won't work with debug heap alloc
//#ifndef USE_TVM
//void* DebugHeapAlloc(size_t size, unsigned framesToSkip = 0);
//void* DebugHeapReAlloc(void* p, size_t size);
//void DebugHeapFree(void* p) noexcept;
//
//#define calloc CallocNotImplemented
//#define malloc DebugHeapAlloc
//#define realloc DebugHeapReAlloc
//#define free DebugHeapFree
//#endif
//#endif
