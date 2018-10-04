//// Copyright (c) Microsoft Corporation. All rights reserved.
//// Licensed under the MIT License.
//
////
//// Debug Memory Leak Checking
////
//// Implements a custom operator new and delete that will capture a callstack in each allocation
//// It creates a separate heap at startup and walks the remaining allocations at process exit,
//// dumping out the callstacks to the console and showing a message box if there were any leaks.
////
//// It creates & destroys itself in init_seg(lib) so it should scope all user code
////
//#if defined(_DEBUG)
//// TVM need to run with shared CRT, so won't work with debug heap alloc
//#ifndef USE_TVM
//constexpr int c_callstack_limit = 16;  // Maximum depth of callstack in leak trace
//#define VALIDATE_HEAP_EVERY_ALLOC 0    // Call HeapValidate on every new/delete
//
//#pragma warning(disable : 4073)  // initializers put in library initialization area (this is intentional)
//#pragma init_seg(lib)
//
//// as this is a debug only checker that does some very low level things and isn't used in the released code
//// ignore a bunch of C++ Core Guidelines code analysis warnings
//#pragma warning(disable : 26409)  // r.11 Don't use 'new' explicitly.
//#pragma warning(disable : 26426)  // i.22 Static local variables use non-constexpr initializer.
//#pragma warning(disable : 26481)  // bounds.1 Don't use pointer arithmetic.
//#pragma warning(disable : 26482)  // bounds.2 Only index into arrays using constant expressions.
//#pragma warning(disable : 26485)  // bounds.3 No array to pointer decay.
//#pragma warning(disable : 26490)  // type.1 Don't use reinterpret_cast
//#pragma warning(disable : 26493)  // type.4 Don't use C-style casts
//
//#include <windows.h>
//#include <sstream>
//#include <iostream>
//#include "debug_alloc.h"
//#include <DbgHelp.h>
//#pragma comment(lib, "Dbghelp.lib")
//
//_Ret_notnull_ _Post_writable_byte_size_(size) void* operator new(size_t size) { return DebugHeapAlloc(size, 1); }
//_Ret_notnull_ _Post_writable_byte_size_(size) void* operator new[](size_t size) { return DebugHeapAlloc(size, 1); }
//void operator delete(void* p) noexcept { DebugHeapFree(p); }
//void operator delete[](void* p) noexcept { DebugHeapFree(p); }
//
//struct MemoryBlock {
//  MemoryBlock(unsigned framesToSkip = 1) noexcept {
//    unsigned i = CaptureStackBackTrace(framesToSkip + 1, _countof(m_pTraces), m_pTraces, nullptr);
//    for (; i < _countof(m_pTraces); i++)
//      m_pTraces[i] = nullptr;
//  }
//
//  void* m_pTraces[c_callstack_limit];
//};
//
//struct SymbolHelper {
//  SymbolHelper() noexcept {
//    SymSetOptions(SymGetOptions() | SYMOPT_DEFERRED_LOADS);
//    SymInitialize(GetCurrentProcess(), nullptr, true);
//  }
//
//  void Lookup(std::string& string, const ULONG_PTR address) {
//    char buffer[2048] = {0};
//    Symbol symbol;
//    if (SymFromAddr(GetCurrentProcess(), address, 0, &symbol) == false) {
//      _snprintf_s(buffer, _TRUNCATE, "0x%08IX (Unknown symbol)", address);
//      string.append(buffer);
//      return;
//    }
//
//    Line line;
//    DWORD displacement;
//    if (SymGetLineFromAddr(GetCurrentProcess(), address, &displacement, &line) == false) {
//      _snprintf_s(buffer, _TRUNCATE, "(unknown file & line number): %s", symbol.Name);
//      string.append(buffer);
//      return;
//    }
//
//    _snprintf_s(buffer, _TRUNCATE, "%s(%d): %s", line.FileName, line.LineNumber, symbol.Name);
//    string.append(buffer);
//  }
//
//  struct Symbol : SYMBOL_INFO {
//    Symbol() noexcept {
//      SizeOfStruct = sizeof(SYMBOL_INFO);
//      MaxNameLen = _countof(buffer);
//    }
//
//    char buffer[1024] = {0};
//  };
//
//  struct Line : IMAGEHLP_LINE {
//    Line() noexcept {
//      SizeOfStruct = sizeof(IMAGEHLP_LINE);
//    }
//  };
//};
//
//static HANDLE g_heap{};
//unsigned g_cumulativeAllocationCount{};
//unsigned g_allocationCount{};
//uint64_t g_cumulativeAllocationBytes{};
//
//// Disable C6386: Buffer overrun for just this section.
//// 'p' is considered a 0 byte array as it's a void*, so the write to 'p'
//// in DebugHeapAlloc and DebugHeapReAlloc trigger spurious warnings.
//#pragma warning(push)
//#pragma warning(disable : 6386)
//
//void* DebugHeapAlloc(size_t size, unsigned framesToSkip) {
//#if (VALIDATE_HEAP_EVERY_ALLOC)
//  if (HeapValidate(g_heap, 0, nullptr) == 0)
//    exit(-1);
//#endif
//
//  g_cumulativeAllocationCount++;
//  g_cumulativeAllocationBytes += size;
//  void* p = HeapAlloc(g_heap, 0, size + sizeof(MemoryBlock));
//  if (!p)
//    throw std::bad_alloc();
//
//  g_allocationCount++;
//  new (p) MemoryBlock(framesToSkip + 1);
//  return static_cast<BYTE*>(p) + sizeof(MemoryBlock);  // Adjust outgoing pointer
//}
//
//void* DebugHeapReAlloc(void* p, size_t size) {
//  if (!p)  // Std library will call realloc(nullptr, size)
//    return DebugHeapAlloc(size);
//
//  g_cumulativeAllocationCount++;
//  g_cumulativeAllocationBytes += size;
//  p = static_cast<BYTE*>(p) - sizeof(MemoryBlock);  // Adjust incoming pointer
//  p = HeapReAlloc(g_heap, 0, p, size + sizeof(MemoryBlock));
//  if (!p)
//    throw std::bad_alloc();
//
//  new (p) MemoryBlock;                                 // Redo the callstack
//  return static_cast<BYTE*>(p) + sizeof(MemoryBlock);  // Adjust outgoing pointer
//}
//
//#pragma warning(pop)  // buffer overrun
//
//void DebugHeapFree(void* p) noexcept {
//#if (VALIDATE_HEAP_EVERY_ALLOC)
//  if (HeapValidate(g_heap, 0, nullptr) == 0)
//    exit(-1);
//#endif
//
//  if (!p)
//    return;
//
//  g_allocationCount--;
//  p = static_cast<BYTE*>(p) - sizeof(MemoryBlock);  // Adjust incoming pointer
//  HeapFree(g_heap, 0, p);
//}
//
//static struct Memory_LeakCheck {
//  Memory_LeakCheck() noexcept;
//  ~Memory_LeakCheck();
//  Memory_LeakCheck(const Memory_LeakCheck&) = delete;
//  Memory_LeakCheck& operator=(const Memory_LeakCheck&) = delete;
//  Memory_LeakCheck(Memory_LeakCheck&&) = delete;
//  Memory_LeakCheck& operator=(Memory_LeakCheck&&) = delete;
//} g_memory_leak_check;
//
//Memory_LeakCheck::Memory_LeakCheck() noexcept {
//  g_heap = HeapCreate(0, 0, 0);
//}
//
//Memory_LeakCheck::~Memory_LeakCheck() {
//  SymbolHelper symbols;
//
//  // Create a new heap so we can still allocate memory while dumping the memory leaks
//  HANDLE heap = HeapCreate(0, 0, 0);
//  std::swap(heap, g_heap);  // Swap it out with our current heap
//
//  unsigned leaked_bytes = 0;
//  unsigned leak_count = 0;
//
//  PROCESS_HEAP_ENTRY entry{};
//  while (HeapWalk(heap, &entry)) {
//    if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) == 0)
//      continue;
//
//    const MemoryBlock& block = *static_cast<const MemoryBlock*>(entry.lpData);
//    const BYTE* pBlock = static_cast<const BYTE*>(entry.lpData) + sizeof(MemoryBlock);
//
//    std::string string;
//    char buffer[1024];
//    _snprintf_s(buffer, _TRUNCATE, "%IX bytes at location 0x%08IX\n", entry.cbData - sizeof(MemoryBlock), UINT_PTR(pBlock));
//    string.append(buffer);
//    for (auto& p : block.m_pTraces) {
//      if (!p) break;
//      symbols.Lookup(string, reinterpret_cast<ULONG_PTR>(p));
//      string.push_back('\n');
//    }
//
//    // Google test has memory leaks that they haven't fixed. One such issue is tracked here: https://github.com/google/googletest/issues/692
//    //
//    // In gtest-port.cc in function: static ThreadIdToThreadLocals* GetThreadLocalsMapLocked()
//    //     static ThreadIdToThreadLocals* map = new ThreadIdToThreadLocals;
//    //
//    // In gtest-port.cc in Mutex::~Mutex() there is this comment:
//    //     "Static mutexes are leaked intentionally. It is not thread-safe to try to clean them up."
//    // Which explains this leak inside of: void Mutex::ThreadSafeLazyInit()
//    //     critical_section_ = new CRITICAL_SECTION;
//    if (string.find("testing::internal::Mutex::ThreadSafeLazyInit") == std::string::npos &&
//        string.find("testing::internal::ThreadLocalRegistryImpl::GetThreadLocalsMapLocked") == std::string::npos &&
//        string.find("testing::internal::ThreadLocalRegistryImpl::GetValueOnCurrentThread") == std::string::npos) {
//      if (leaked_bytes == 0)
//        OutputDebugStringA("\n-----Starting Heap Trace-----\n\n");
//
//      leak_count++;
//      leaked_bytes += entry.cbData - sizeof(MemoryBlock);
//      OutputDebugStringA(string.c_str());
//      OutputDebugStringA("\n");
//    }
//  }
//
//  if (leaked_bytes) {
//    OutputDebugStringA("-----Ending Heap Trace-----\n\n");
//
//    std::string string;
//    char buffer[1024];
//    _snprintf_s(buffer, _TRUNCATE, "%d bytes of memory leaked in %d allocations", leaked_bytes, leak_count);
//    string.append(buffer);
//
//    // Check if we're running on the build machine, if so just exit(-1)
//    size_t requiredSize;
//    if (getenv_s(&requiredSize, nullptr, 0, "AGENT_BUILDDIRECTORY") == 0 && requiredSize > 0) {
//      std::cout << "\n----- MEMORY LEAKS: " << string.c_str() << "\n";
//      exit(-1);
//    }
//
//    // Otherwise we're running on a dev system, show a message box to get their attention
//    if (IsDebuggerPresent()) {
//      MessageBoxA(nullptr, string.c_str(), "Warning", MB_OK | MB_ICONWARNING);
//    }
//  } else {
//    OutputDebugStringA("\n----- No memory leaks detected -----\n\n");
//  }
//
//  HeapDestroy(heap);
//  HeapDestroy(g_heap);
//  g_heap = nullptr;  // Any allocations after this point will fail
//}
//#endif
//#endif