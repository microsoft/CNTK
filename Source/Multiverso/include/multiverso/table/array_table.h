#ifndef MULTIVERSO_ARRAY_TABLE_H_
#define MULTIVERSO_ARRAY_TABLE_H_

#include "multiverso/multiverso.h"
#include "multiverso/table_interface.h"
#include "multiverso/util/log.h"

namespace multiverso {

template<typename EleType>
struct ArrayTableOption;

template <typename T>
class ArrayWorker : public WorkerTable {
public:
  explicit ArrayWorker(size_t size);
  explicit ArrayWorker(const ArrayTableOption<T> &option);
  // std::vector<T>& raw() { return table_; }

  // Get all element, data is user-allocated memory
  void Get(T* data, size_t size);

  // Add all element
  void Add(T* data, size_t size, const AddOption* option = nullptr);

  int Partition(const std::vector<Blob>& kv,
                std::unordered_map<int, std::vector<Blob> >* out) override;

  void ProcessReplyGet(std::vector<Blob>& reply_data) override;
  
private:
  T* data_; // not owned
  size_t size_;
  int num_server_;
  std::vector<size_t> server_offsets_;
};

template <typename T>
class Updater;

// The storage is a continuous large chunk of memory
template <typename T>
class ArrayServer : public ServerTable {
public:
  explicit ArrayServer(size_t size);
  explicit ArrayServer(const ArrayTableOption<T> &option);

  void ProcessAdd(const std::vector<Blob>& data) override;

  void ProcessGet(const std::vector<Blob>& data,
                  std::vector<Blob>* result) override;

  void Store(Stream* s) override;
  void Load(Stream* s) override;

private:
  int32_t server_id_;
  std::vector<T> storage_;
  Updater<T>* updater_;
  size_t size_; // number of element with type T
  
};

template<typename T>
struct ArrayTableOption {
  ArrayTableOption(size_t s) : size(s) {}
  size_t size;
  DEFINE_TABLE_TYPE(T, ArrayWorker, ArrayServer);
};

}

#endif // MULTIVERSO_ARRAY_TABLE_H_
