#include "multiverso/blob.h"

#include "multiverso/util/allocator.h"
#include "multiverso/util/log.h"

namespace multiverso {

Blob::Blob(size_t size) : size_(size) {
  CHECK(size > 0);
  data_ = Allocator::Get()->Alloc(size);
}

// Construct from external memory. Will copy a new piece
Blob::Blob(const void* data, size_t size) : size_(size) {
  data_ = Allocator::Get()->Alloc(size);
  memcpy(data_, data, size_);
}

Blob::Blob(void* data, size_t size) : size_(size) {
  data_ = Allocator::Get()->Alloc(size);
  memcpy(data_, data, size_);
}

Blob::Blob(const Blob& rhs) {
  if (rhs.size() != 0) {
    Allocator::Get()->Refer(rhs.data_);
  }
  this->data_ = rhs.data_;
  this->size_ = rhs.size_;
}

Blob::~Blob() {
  if (data_ != nullptr) {
    Allocator::Get()->Free(data_);
  }
}

// Shallow copy by default. Call \ref CopyFrom for a deep copy
void Blob::operator=(const Blob& rhs) {
  if (rhs.size() != 0) {
    Allocator::Get()->Refer(rhs.data_);
  }
  this->data_ = rhs.data_;
  this->size_ = rhs.size_;
}

}  // namespace multiverso
