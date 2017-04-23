#include "memory_manager.h"

namespace wordembedding {

  MemoryManager::MemoryManager(int block_size) {
    block_size_ = block_size;
  }
  //Request memory for blocks
  void MemoryManager::RequestBlocks(int64 block_number,
    std::vector<real*>& result) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (int64 i = 0; i < block_number; ++i) {
      result.push_back(new (std::nothrow) real[block_size_]);
      assert(result[i] != nullptr);
    }
  }
  //Free the memory for blocks
  void MemoryManager::ReturnBlocks(std::vector<real*>& blocks) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (size_t i = 0; i < blocks.size(); ++i)
      delete[] blocks[i];
  }

  MemoryManager::~MemoryManager() {

  }
}