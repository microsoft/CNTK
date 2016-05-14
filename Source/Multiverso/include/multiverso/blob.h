#ifndef MULTIVERSO_BLOB_H_
#define MULTIVERSO_BLOB_H_

#include <memory>
#include <string>
#include <cstring>
#include <iostream>

#include "multiverso/util/log.h"
#include "multiverso/util/allocator.h"

namespace multiverso {

  // Manage a chunk of memory. Blob can share memory with other Blobs.
  // Never use external memory. All external memory should be managed by itself
  // TODO(feiga): maybe make blob also not hold memory?
  class Blob {
  public:
    // an empty blob
    Blob() : data_(nullptr) {}

    explicit Blob(size_t size) : size_(size) {
      CHECK(size > 0);
      data_ = Allocator::Get()->Alloc(size);
      //data_.reset(new char[size]);
    }

    // Construct from external memory. Will copy a new piece
    Blob(const void* data, size_t size) : size_(size) {
      data_ = Allocator::Get()->Alloc(size);
      memcpy(data_, data, size_);
    }

    Blob(void* data, size_t size) : size_(size) {
      data_ = Allocator::Get()->Alloc(size);
      memcpy(data_, data, size_);
    }

    Blob(const Blob& rhs) {
      Allocator::Get()->Refer(rhs.data_);
      this->data_ = rhs.data_;
      this->size_ = rhs.size_;
    }

    ~Blob() {
      if (data_ != nullptr) {
        Allocator::Get()->Free(data_);
      }
    }

    // Shallow copy by default. Call \ref CopyFrom for a deep copy
    void operator=(const Blob& rhs) {
      Allocator::Get()->Refer(rhs.data_);
      this->data_ = rhs.data_;
      this->size_ = rhs.size_;
    }

    inline char operator[](size_t i) const {
      CHECK(0 <= i && i < size_);
      return data_[i];
    }

    template <typename T>
    inline T& As(size_t i = 0) const {
      CHECK(size_ % sizeof(T) == 0 && i < size_ / sizeof(T));
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
    //std::shared_ptr<char> data_;
    char *data_;
    size_t size_;
  };

}  // namespace multiverso

#endif  // MULTIVERSO_BLOB_H_
