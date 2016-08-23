#ifndef MULTIVERSO_BLOB_H_
#define MULTIVERSO_BLOB_H_

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

namespace multiverso {

// Manage a chunk of memory. Blob can share memory with other Blobs.
// Never use external memory. All external memory should be managed by itself
class Blob {
public:
  // an empty blob
  Blob() : data_(nullptr), size_(0) {}

  explicit Blob(size_t size);

  // Construct from external memory. Will copy a new piece
  Blob(const void* data, size_t size);

  Blob(void* data, size_t size);

  Blob(const Blob& rhs);

  ~Blob();

  // Shallow copy by default. Call \ref CopyFrom for a deep copy
  void operator=(const Blob& rhs);

  inline char operator[](size_t i) const {
    return data_[i];
  }

  template <typename T>
  inline T& As(size_t i = 0) const {
    return (reinterpret_cast<T*>(data_))[i];
  }
  template <typename T>
  inline size_t size() const { return size_ / sizeof(T); }

  // DeepCopy, for a shallow copy, use operator=
  void CopyFrom(const Blob& src);

  inline char* data() const { return data_; }
  inline size_t size() const { return size_; }

private:
  // Memory is shared and auto managed
  char *data_;
  size_t size_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_BLOB_H_
