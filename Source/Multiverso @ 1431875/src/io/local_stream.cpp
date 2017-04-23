#include "multiverso/io/local_stream.h"
#include <errno.h>
extern "C" {
#include <sys/stat.h>
}
#ifndef _MSC_VER
extern "C" {
#include <sys/types.h>
#include <dirent.h>
}
#else
#include <Windows.h>
#define stat _stat64
#endif

namespace multiverso {

LocalStream::LocalStream(const URI& uri, FileOpenMode mode) {
  path_ = uri.path;
  std::string mode_str;
  switch (mode) {
  case FileOpenMode::Read:
    mode_str = "r";
    break;
  case FileOpenMode::Write:
    mode_str = "w";
    break;
  case FileOpenMode::Append:
    mode_str = "a";
    break;
  case FileOpenMode::BinaryRead:
    mode_str = "rb";
    break;
  case FileOpenMode::BinaryWrite:
    mode_str = "wb";
    break;
  case FileOpenMode::BinaryAppend:
    mode_str = "ab";
  }
#ifdef _MSC_VER
  fopen_s(&fp_, uri.path.c_str(), mode_str.c_str());
#else
  fp_ = fopen(uri.path.c_str(), mode_str.c_str());
#endif

  if (fp_ == nullptr) {
    is_good_ = false;
    Log::Error("Failed to open LocalStream %s\n", uri.path.c_str());
  } else {
    is_good_ = true;
  }
}

LocalStream::~LocalStream(void)
{
  is_good_ = false;
  if (fp_ != nullptr)
    std::fclose(fp_);
}

/*!
* \brief write data to a file
* \param buf pointer to a memory buffer
* \param size data size
*/
void LocalStream::Write(const void *buf, size_t size) {
  if (std::fwrite(buf, 1, size, fp_) != size) {
    is_good_ = false;
    Log::Error("LocalStream.Write incomplete\n");
  }
}


/*!
* \brief read data from Stream
* \param buf pointer to a memory buffer
* \param size the size of buf
*/
size_t LocalStream::Read(void *buf, size_t size) {
  return std::fread(buf, 1, size, fp_);
}

bool LocalStream::Good() { return is_good_; }

LocalStreamFactory::LocalStreamFactory(const std::string& host) {
  host_ = host;
}

LocalStreamFactory::~LocalStreamFactory() {
}

/*!
* \brief create a Stream
* \param path the path of the file
* \param mode "w" - create an empty file to store data;
*             "a" - open the file to append data to it
*             "r" - open the file to read
* \return the Stream which is used to write or read data
*/
Stream* LocalStreamFactory::Open(const URI& uri, FileOpenMode mode) {
  return new LocalStream(uri, mode);
}

void LocalStreamFactory::Close() {
  ///TODO
}

}