#ifndef MULTIVERSO_ALLOCATOR_H_
#define MULTIVERSO_ALLOCATOR_H_

#include <mutex>
#include <atomic>
#include <unordered_map>

namespace multiverso {

const size_t g_pointer_size = sizeof(void*);

class MemoryBlock;
class FreeList {
public:
  FreeList(size_t size);
  ~FreeList();
  char *Pop();
  void Push(MemoryBlock*);
private:
  MemoryBlock* free_ = nullptr;
  size_t size_;
  std::mutex mutex_;
};

class MemoryBlock {
public:
  MemoryBlock(size_t size, FreeList* list);
  ~MemoryBlock();
  char* data();
  void Unlink();
  void Link();
  MemoryBlock* next;
private:
  char* data_;
  std::atomic<int> ref_;
  static const size_t header_size_ = (sizeof(MemoryBlock*) << 1);
};

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual char* Alloc(size_t size);
  virtual void Free(char* data);
  virtual void Refer(char *data);
  static Allocator* Get();
private:
  static const int header_size_ = sizeof(std::atomic<int>*);
};

class SmartAllocator : public Allocator {
public:
  char* Alloc(size_t size);
  void Free(char* data);
  void Refer(char *data);
  ~SmartAllocator();
private:
  std::unordered_map<size_t, FreeList*> pools_;
  std::mutex mutex_;
};

} // namespace multiverso

#endif // MULTIVERSO_ALLOCATOR_H_
