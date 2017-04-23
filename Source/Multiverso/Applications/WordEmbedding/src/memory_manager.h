#ifndef WORDEMBEDDING_MEMORY_MANAGER_H_
#define WORDEMBEDDING_MEMORY_MANAGER_H_
/*!
* file memory_manager.h
* \brief Class MemoryManager creates and allocates memory for the local parameter which is needed by the datablock training.
*/

#include <cassert>
#include <vector>
#include <condition_variable>

#include "constant.h"

namespace wordembedding {

  class MemoryManager {
  public:
    explicit MemoryManager(int block_size);
    /*!
    * \brief Create memory for the blocks
    * \param block_number the block quantity needed
    * \param result the vector of the head address of allocated memory
    */
    void RequestBlocks(int64 block_number, std::vector<real*>& result);
    /*!
    * \brief Delete the blocks memory
    * \param blocks the vector of the head address of allocated memory
    */
    void ReturnBlocks(std::vector<real*>& blocks);
    ~MemoryManager();

  private:
    int64 block_size_;
    std::mutex mutex_;

    // No copying allowed
    MemoryManager(const MemoryManager&);
    void operator=(const MemoryManager&);
  };
}
#endif
