#ifdef MULTIVERSO_USE_HDFS

#include "multiverso/util/hdfs_stream.h"

namespace multiverso {

HDFSStream::HDFSStream(hdfsFS fs, const URI &uri, FileOpenMode mode) {
  using namespace std;
  is_good_ = true;
  fs_ = fs;
  fp_ = nullptr;

  int flag = 0;
  switch (mode) {
  case FileOpenMode::Write:
    flag = O_WRONLY;
    break;
  case FileOpenMode::Read:
    flag = O_RDONLY;
    break;
  case FileOpenMode::Append:
    flag = O_WRONLY | O_APPEND;
    break;
  case FileOpenMode::BinaryWrite:
    flag = O_WRONLY | O_BINARY;
    break;
  case FileOpenMode::BinaryRead:
    flag = O_RDONLY | O_BINARY;
    break;
  case FileOpenMode::BinaryAppend:
    flag = O_WRONLY | O_APPEND | O_BINARY;
    break;
  }    

  fp_ = hdfsOpenFile(fs_, uri.name.c_str(), flag, 0, 0, 0);

  if (fp_ == nullptr) {
    is_good_ = false;
    int errsv = errno;
    Log::Error("Failed to open HDFSStream %s, %s\n", uri.name.c_str(), strerror(errsv));
    return;
  }       

  path_ = uri.name;
}

HDFSStream::~HDFSStream(void) {
  if (mode_ == "a" || mode_ == "w"){
  //flush local buffer
    if (hdfsHSync(fs_, fp_) == -1)  {
      is_good_ = false;
      int errsv = errno;
      Log::Error("Failed to Flush HDFSStream %s: %s\n", path_.c_str(), strerror(errsv));
    }
  }
  if (hdfsCloseFile(fs_, fp_) == -1) {
    is_good_ = false;
    int errsv = errno;
    Log::Error("Failed to close HDFSStream %s, %s\n", path_.c_str(), strerror(errsv));
  }
}

/*!
* \brief write data to a file
* \param buf pointer to a memory buffer
* \param size data size
*/
void HDFSStream::Write(const void *buf, size_t size) {
  const char *c_buf = reinterpret_cast<const char*>(buf);
  while (size > 0) {
    tSize nwrite = hdfsWrite(fs_, fp_, c_buf, size);
    if (nwrite == -1) {
      is_good_ = false;
      int errsv = errno;
      Log::Fatal("Failed to Write data to HDFSStream %s, %s\n", path_.c_str(), strerror(errsv));
    }
    size_t sz = static_cast<size_t>(nwrite);
    c_buf += sz;
    size -= sz;
  }
}

/*!
* \brief read data from Stream
* \param buf pointer to a memory buffer
* \param size the size of buf
*/
size_t HDFSStream::Read(void *buf, size_t size) {
  char *c_buf = static_cast<char*>(buf);
  size_t i = 0;
  while (i < size) {
    size_t nmax = static_cast<size_t>(std::numeric_limits<tSize>::max());
    //Log::Debug("Begin hdfsRead\n");
    tSize ret = hdfsRead(fs_, fp_, c_buf + i, std::min(size - i, nmax));
    //Log::Debug("hdfsRead return %d, i=%d, size=%d\n", ret, i, size);
    if (ret > 0) {
      size_t n = static_cast<size_t>(ret);
      i += n;
    } else if (ret == 0) {
      break;
    } else {
      int errsv = errno;
      if (errno == EINTR) {
        Log::Info("Failed to Read HDFSStream %s, %s data is temporarily unavailable\n",
          path_.c_str(), strerror(errsv));
        continue;
      }
      is_good_ = false;
      Log::Fatal("Failed to Read HDFSStream %s, %s\n", path_.c_str(), strerror(errsv));
    }
  }
  return i;
}

bool HDFSStream::Good() { return is_good_; }

HDFSStreamFactory::HDFSStreamFactory(const std::string &host) {
  namenode_ = host;

#ifdef _CONNECT_NEW_INSTANCE_
  fs_ = hdfsConnectNewInstance(namenode_.c_str(), 0);
#else
  fs_ = hdfsConnect(namenode_.c_str(), 0);
#endif

  if (fs_ == NULL) {
    int errsv = errno;
    Log::Fatal("Failed connect HDFS namenode '%s', %s\n", namenode_.c_str(), strerror(errsv));
  }
}

HDFSStreamFactory::~HDFSStreamFactory(void) { Close(); }

void HDFSStreamFactory::Close(void) {
  if (fs_ != nullptr && hdfsDisconnect(fs_) != 0) {
    int errsv = errno;
    Log::Fatal("HDFSStream.hdfsDisconnect Error: %s\n", strerror(errsv));
  }
  else {
    fs_ = nullptr;
  }
}

/*!
* \brief create a Stream
* \param path the path of the file
* \param mode "w" - create an empty file to store data;
*             "a" - open the file to append data to it
*             "r" - open the file to read
* \return the Stream which is used to write or read data
*/
Stream* HDFSStreamFactory::Open(const URI & uri, FileOpenMode mode) {
  return new HDFSStream(fs_, uri, mode);   
}

}

#endif