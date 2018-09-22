/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <limits>
static const int std_numeric_limits_int_max = std::numeric_limits<int>::max();
static const unsigned int std_numeric_limits_DWORD_max = std::numeric_limits<unsigned int>::max();
#include <Shlwapi.h>
#include <Windows.h>

#include <string>
#include <thread>
#include <fcntl.h>
#include <fstream>
#include <io.h>

#include "core/common/logging/logging.h"
#include "core/platform/env.h"


namespace onnxruntime {

namespace {

class StdThread : public Thread {
 public:
  StdThread(std::function<void()> fn)
      : thread_(fn) {}

  ~StdThread() { thread_.join(); }

 private:
  std::thread thread_;
};

class WindowsEnv : public Env {
 private:
  template <typename T, typename F>
  static common::Status FileExists_(T fname, F f) {
    if (!fname)
      return common::Status(common::LOTUS, common::INVALID_ARGUMENT, "file name is nullptr");
    struct _stat st;
    int ret = f(fname, &st);
    if (ret == 0) {
      if (st.st_mode & _S_IFREG)
        return common::Status::OK();
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, fname, "is not a regular file");
    }
    switch (errno) {
      case ENOENT:
        return common::Status(common::LOTUS, common::NO_SUCHFILE, "");
      case EINVAL:
        return common::Status(common::LOTUS, common::INVALID_ARGUMENT, "");
      default:
        return common::Status(common::LOTUS, common::FAIL, "unknown error inside FileExists");
    }
  }

 public:
  void SleepForMicroseconds(int64_t micros) const override { Sleep(static_cast<DWORD>(micros) / 1000); }

  Thread* StartThread(const ThreadOptions&, const std::string&,
                      std::function<void()> fn) const override {
    return new StdThread(fn);
  }

  int GetNumCpuCores() const override {
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
    DWORD returnLength = sizeof(buffer);
    if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
      // try GetSystemInfo
      SYSTEM_INFO sysInfo;
      GetSystemInfo(&sysInfo);
      if (sysInfo.dwNumberOfProcessors <= 0) {
        LOTUS_THROW("Fatal error: 0 count processors from GetSystemInfo");
      }
      // This is the number of logical processors in the current group
      return sysInfo.dwNumberOfProcessors;
    }
    int processorCoreCount = 0;
    int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    for (int i = 0; i != count; ++i) {
      if (buffer[i].Relationship == RelationProcessorCore) {
        ++processorCoreCount;
      }
    }
    if (!processorCoreCount) LOTUS_THROW("Fatal error: 0 count processors from GetLogicalProcessorInformation");
    return processorCoreCount;
  }

  static WindowsEnv& Instance() {
    static WindowsEnv default_env;
    return default_env;
  }

  PIDType GetSelfPid() const override {
    return GetCurrentProcessId();
  }

  EnvThread* CreateThread(std::function<void()> fn) const override {
    return new StdThread(fn);
  }

  Task CreateTask(std::function<void()> f) const override {
    return Task{std::move(f)};
  }
  void ExecuteTask(const Task& t) const override {
    t.f();
  }

  common::Status FileOpenRd(const std::wstring& path, /*out*/ gsl::not_null<int*> p_fd) const override {
    _wsopen_s(p_fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > *p_fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileOpenWr(const std::wstring& path, /*out*/ gsl::not_null<int*> p_fd) const override {
    _wsopen_s(p_fd, path.c_str(), _O_CREAT | O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > *p_fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileOpenRd(const std::string& path, /*out*/ gsl::not_null<int*> p_fd) const override {
    _sopen_s(p_fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > *p_fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileOpenWr(const std::string& path, /*out*/ gsl::not_null<int*> p_fd) const override {
    _sopen_s(p_fd, path.c_str(), _O_CREAT | O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
    if (0 > *p_fd) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileClose(int fd) const override {
    int ret = _close(fd);
    if (0 != ret) {
      return common::Status(common::SYSTEM, errno);
    }
    return Status::OK();
  }

  common::Status FileExists(const char* fname) const override {
    return FileExists_(fname, _stat);
  }
  common::Status FileExists(const wchar_t* fname) const override {
    return FileExists_(fname, _wstat);
  }
  common::Status ReadFileAsString(const char* fname, std::string* out) const override {
    if (!fname)
      return common::Status(common::LOTUS, common::INVALID_ARGUMENT, "file name is nullptr");
    size_t flen = strlen(fname);
    if (flen >= std_numeric_limits_int_max) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "input path too long");
    }
    int len = MultiByteToWideChar(CP_ACP, 0, fname, (int)(flen + 1), nullptr, 0);
    if (len <= 0) {
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "MultiByteToWideChar error");
    }
    std::wstring wStreamName((size_t)(len - 1), L'\0');
    MultiByteToWideChar(CP_ACP, 0, fname, (int)flen, (LPWSTR)wStreamName.data(), len);
    return ReadFileAsString(wStreamName.c_str(), out);
  }

  common::Status ReadFileAsString(const wchar_t* fname, std::string* out) const override {
    //if (!fname)
    //  return common::Status(common::LOTUS, common::INVALID_ARGUMENT, "file name is nullptr");
    //if (!out) {
    //  return common::Status(common::LOTUS, common::INVALID_ARGUMENT, "'out' cannot be NULL");
    //}
    //char errbuf[512];
    //HANDLE hFile = CreateFileW(fname, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    //if (hFile == INVALID_HANDLE_VALUE) {
    //  int err = GetLastError();
    //  _snprintf_s(errbuf, _TRUNCATE, "%s:%d open file %ls fail, errcode = %d", __FILE__, (int)__LINE__, fname, err);
    //  return common::Status(common::LOTUS, common::FAIL, errbuf);
    //}
    //LARGE_INTEGER filesize;
    //if (!GetFileSizeEx(hFile, &filesize)) {
    //  int err = GetLastError();
    //  _snprintf_s(errbuf, _TRUNCATE, "%s:%d GetFileSizeEx %ls fail, errcode = %d", __FILE__, (int)__LINE__, fname, err);
    //  CloseHandle(hFile);
    //  return common::Status(common::LOTUS, common::FAIL, errbuf);
    //}
    //out->resize(filesize.QuadPart, '\0');
    //if (filesize.QuadPart > std_numeric_limits_DWORD_max) {
    //  _snprintf_s(errbuf, _TRUNCATE, "%s:%d READ file %ls fail, file size too long", __FILE__, (int)__LINE__, fname);
    //  CloseHandle(hFile);
    //  //we can support that with a while loop
    //  return common::Status(common::LOTUS, common::NOT_IMPLEMENTED, errbuf);
    //}
    //if (!ReadFile(hFile, (void*)out->data(), (DWORD)filesize.QuadPart, nullptr, nullptr)) {
    //  int err = GetLastError();
    //  _snprintf_s(errbuf, _TRUNCATE, "%s:%d ReadFileEx %ls fail, errcode = %d", __FILE__, (int)__LINE__, fname, err);
    //  CloseHandle(hFile);
    //  return common::Status(common::LOTUS, common::FAIL, errbuf);
    //}
    //CloseHandle(hFile);
    return common::Status::OK();
  }

  virtual Status LoadLibrary(const std::string& library_filename, void** handle) const override {
    UNUSED_PARAMETER(library_filename);
    UNUSED_PARAMETER(handle);
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  virtual common::Status UnloadLibrary(void* handle) const override {
    UNUSED_PARAMETER(handle);
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  virtual Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override {
    UNUSED_PARAMETER(handle);
    UNUSED_PARAMETER(symbol_name);
    UNUSED_PARAMETER(symbol);
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  virtual std::string FormatLibraryFileName(const std::string& name, const std::string& version) const override {
    UNUSED_PARAMETER(name);
    UNUSED_PARAMETER(version);
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

 private:
  WindowsEnv()
      : GetSystemTimePreciseAsFileTime_(nullptr) {
    // GetSystemTimePreciseAsFileTime function is only available in the latest
    // versions of Windows. For that reason, we try to look it up in
    // kernel32.dll at runtime and use an alternative option if the function
    // is not available.
    //HMODULE module = GetModuleHandleW(L"kernel32.dll");
    //if (module != nullptr) {
    //  auto func = (FnGetSystemTimePreciseAsFileTime)GetProcAddress(
    //      module, "GetSystemTimePreciseAsFileTime");
    //  GetSystemTimePreciseAsFileTime_ = func;
    //}
  }

  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  FnGetSystemTimePreciseAsFileTime GetSystemTimePreciseAsFileTime_;
};

}  // namespace

#ifdef _WIN32
const Env& Env::Default() {
  return WindowsEnv::Instance();
}
#endif

}  // namespace onnxruntime
