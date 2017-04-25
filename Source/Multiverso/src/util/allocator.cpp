#include "multiverso/util/allocator.h"

#include <mutex>

#include "multiverso/util/log.h"
#include "multiverso/util/configure.h"

namespace multiverso {

MV_DEFINE_int(allocator_alignment, 16, "alignment for align malloc");

inline char* AlignMalloc(size_t size) {
#ifdef _MSC_VER 
  return (char*)_aligned_malloc(size, 
    MV_CONFIG_allocator_alignment);
#else
  void *data;
  CHECK(posix_memalign(&data, 
    MV_CONFIG_allocator_alignment, size) == 0);
  return (char*)data;
#endif
}

inline void AlignFree(char *data) {
#ifdef _MSC_VER 
  _aligned_free(data);
#else
  free(data);
#endif
}

inline FreeList::FreeList(size_t size) : size_(size) {
  mutex_ = new std::mutex();
  free_ = new MemoryBlock(size, this);
}

FreeList::~FreeList() {
  MemoryBlock*move = free_, *next;
  while (move) {
    next = move->next;
    delete move;
    move = next;
  }
  delete mutex_;
}

inline char* FreeList::Pop() {
  std::lock_guard<std::mutex> lock(*mutex_);
  if (free_ == nullptr) {
    free_ = new MemoryBlock(size_, this);
  }
  char* data = free_->data();
  free_ = free_->next;
  return data;
}

inline void FreeList::Push(MemoryBlock*block) {
  std::lock_guard<std::mutex> lock(*mutex_);
  block->next = free_;
  free_ = block;
}

inline MemoryBlock::MemoryBlock(size_t size, FreeList* list) :
next(nullptr), ref_(0) {
  data_ = AlignMalloc(size + header_size_);
  *(FreeList**)(data_) = list;
  *(MemoryBlock**)(data_ + g_pointer_size) = this;
}

MemoryBlock::~MemoryBlock() {
  AlignFree(data_);
}

inline void MemoryBlock::Unlink() {
  if ((--ref_) == 0) {
    (*(FreeList**)data_)->Push(this);
  }
}

inline char* MemoryBlock::data() {
  ++ref_;
  return data_ + header_size_;
}

inline void MemoryBlock::Link() {
  ++ref_;
}

char* SmartAllocator::Alloc(size_t size) {
  if (size <= 32) {
    size = 32;
  }
  else { // 2^n
    size -= 1;
    size |= size >> 32;
    size |= size >> 16;
    size |= size >> 8;
    size |= size >> 4;
    size |= size >> 2;
    size |= size >> 1;
    size += 1;
  }

  {
    std::lock_guard<std::mutex> lock(*mutex_);
    if (pools_[size] == nullptr) {
      pools_[size] = new FreeList(size);
    }
  }

  return pools_[size]->Pop();
}

void SmartAllocator::Free(char *data) {
  (*(MemoryBlock**)(data - g_pointer_size))->Unlink();
}

void SmartAllocator::Refer(char *data) {
  (*(MemoryBlock**)(data - g_pointer_size))->Link();
}

SmartAllocator::SmartAllocator() {
  mutex_ = new std::mutex();
}

SmartAllocator::~SmartAllocator() {
  Log::Debug("~SmartAllocator, final pool size: %d\n", pools_.size());
  delete mutex_;
  for (auto i : pools_) {
    delete i.second;
  }
}

char* Allocator::Alloc(size_t size) {
  char* data = AlignMalloc(size + header_size_);
  // record ref
  *(std::atomic<int>**)data = new std::atomic<int>(1);
  return data + header_size_;
}

void Allocator::Free(char* data) {
  data -= header_size_;
  if (--(**(std::atomic<int>**)data) == 0) {
    delete *(std::atomic<int>**)data;
    AlignFree(data);
  }
}

void Allocator::Refer(char* data) {
  ++(**(std::atomic<int>**)(data - header_size_));
}

MV_DEFINE_string(allocator_type, "smart", "use smart allocator by default");
Allocator* Allocator::Get() {
  if (MV_CONFIG_allocator_type == "smart") {
    static SmartAllocator allocator_;
    return &allocator_;
  }
  static Allocator allocator_;
  return &allocator_;
}

} // namespace multiverso 
