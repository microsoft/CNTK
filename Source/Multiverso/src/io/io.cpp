#include "multiverso/io/io.h"
#include "multiverso/io/hdfs_stream.h"
#include "multiverso/io/local_stream.h"


namespace multiverso {

Stream* StreamFactory::GetStream(const URI& uri,
  FileOpenMode mode) {
  std::string addr = uri.scheme + "://" + uri.host;
  if (instances_.find(addr) == instances_.end()) {
    if (uri.scheme == std::string("file"))
      instances_[addr] = std::shared_ptr<StreamFactory>(new LocalStreamFactory(uri.host));
#ifdef MULTIVERSO_USE_HDFS
    else if (uri.scheme == std::string("hdfs"))
      instances_[addr] = std::shared_ptr<StreamFactory>(new HDFSStreamFactory(uri.host));
#endif
    else Log::Error("Can not support the StreamFactory '%s'\n", uri.scheme.c_str());
  }
  return instances_[addr]->Open(uri, mode);
}

std::map<std::string, std::shared_ptr<StreamFactory> > StreamFactory::instances_;

TextReader::TextReader(const URI& uri, size_t buf_size) {
    stream_ = StreamFactory::GetStream(uri, FileOpenMode::Read);
    buf_size_ = buf_size;
  pos_ = length_ = 0;
  buf_ = new char[buf_size_];
  assert(buf_ != nullptr);
}

size_t TextReader::GetLine(std::string &line) {
    line.clear();
    bool isEnd = false;
    while (!isEnd) {
        while(pos_ < length_) {
            char & c = buf_[pos_++];
            if (c == '\n') {
                isEnd = true;
                break;
            } else {
                line += c;
            }
        }
        if (isEnd || LoadBuffer() == 0)  break; 
    }
    return line.size();
}

size_t TextReader::LoadBuffer() {
    assert (pos_ == length_);
    pos_ = length_ = 0;
    return length_ = stream_->Read(buf_, buf_size_ - 1);
}

TextReader::~TextReader() {
  delete stream_;
    delete [] buf_;
}

}
