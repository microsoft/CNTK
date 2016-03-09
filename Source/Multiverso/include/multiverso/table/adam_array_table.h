#ifndef MULTIVERSO_ADAM_ARRAY_TABLE_H_
#define MULTIVERSO_ADAM_ARRAY_TABLE_H_

#include "multiverso/table_interface.h"
#include "multiverso/util/log.h"
#include "multiverso/zoo.h"

namespace multiverso {

// this is a first-order gradient-based optimization of stochastic objective functions, 
// based on adaptive estimates of lower-order moments
// by Diederik P. Kingma, Jimmy Lei Ba
template <typename T>
class AdamArrayWorker : public WorkerTable {
public:
  explicit AdamArrayWorker(size_t size) : WorkerTable(), size_(size) {
    num_server_ = Zoo::Get()->num_servers();
    server_offsets_.push_back(0);
    CHECK(size_ > Zoo::Get()->num_servers()); 
    int length = static_cast<int>(size_) / Zoo::Get()->num_servers();
    for (int i = 1; i < Zoo::Get()->num_servers(); ++i) {
      server_offsets_.push_back(i * length); 
    }
    server_offsets_.push_back(size_);
	Log::Debug("worker %d create AdamArrayTable with %d elements.\n", Zoo::Get()->rank(), size);
  }

  T* raw() { return data_; }

  // Get all element
  // data is user-allocated memory
  void Get(T* data, size_t size) {
    CHECK(size == size_);
    data_ = data;
    int all_key = -1;
    Blob whole_table(&all_key, sizeof(int));
    WorkerTable::Get(whole_table); 
    Log::Debug("worker %d getting all parameters.\n", Zoo::Get()->rank());
  }

  // Add all element
  void Add(T* data, size_t size, float smooth_momentum = 0.0) {
    CHECK(size == size_);
    int all_key = -1;

    Blob key(&all_key, sizeof(int));
    Blob val(data, sizeof(T) * size);
	decay_momentum_rate_first_ = smooth_momentum;
    WorkerTable::Add(key, val);
    Log::Debug("worker %d adding parameters with size of %d.\n", Zoo::Get()->rank(), size);
  }

  int Partition(const std::vector<Blob>& kv,
    std::unordered_map<int, std::vector<Blob> >* out) override {
    CHECK(kv.size() == 1 || kv.size() == 2); // kv.size() == 1 : get msg;
										     // kv.size() == 2 : add msg;
	for (int i = 0; i < num_server_; ++i)
	{
		(*out)[i].push_back(kv[0]);
	}

    if (kv.size() == 2) 
	{
      CHECK(kv[1].size() == size_ * sizeof(T));
      for (int i = 0; i < num_server_; ++i)
	  {
        Blob blob(kv[1].data() + server_offsets_[i] * sizeof(T), 
          (server_offsets_[i + 1] - server_offsets_[i]) * sizeof(T));
        (*out)[i].push_back(blob);
		Blob momentum(&decay_momentum_rate_first_, sizeof(float)); // sending coefficent of smooth gradient to server
        (*out)[i].push_back(momentum);
      }
    }
    return num_server_;
  }

  void ProcessReplyGet(std::vector<Blob>& reply_data) override {
    CHECK(reply_data.size() == 2);
    int id = (reply_data[0]).As<int>();
    CHECK(reply_data[1].size<T>() == (server_offsets_[id+1] - server_offsets_[id]));

	// TODO(qiwye): is there a way to reduce this memcpy?
    memcpy(data_ + server_offsets_[id], reply_data[1].data(), reply_data[1].size());
  }
  
private:
  T* data_; // not owned
  size_t size_;
  int num_server_;
  float decay_momentum_rate_first_;
  std::vector<size_t> server_offsets_;
};

// The storage is a continuous large chunk of memory
template <typename T>
class AdamArrayServer : public ServerTable {
public:
  explicit AdamArrayServer(size_t size) : ServerTable() {
    server_id_ = Zoo::Get()->rank();
    size_ = size / Zoo::Get()->size();
    if (server_id_ == Zoo::Get()->num_servers()-1) { // last server 
      size_ += size % Zoo::Get()->num_servers();
    }
    storage_.resize(size_);
	smooth_gradient_first_.resize(size_);

	stepsize_ = 0.001;
	decay_momentum_rate_first_ = 0.9f; 
	decay_momentum_rate_first_ = 0.99f; // default setting in paper

	Log::Debug("server %d create AdamArrayTable with %d elements of %d elements.\n", server_id_, size_ * 2, size * 2);
  }

  void ProcessAdd(const std::vector<Blob>& data) override {
#ifdef MULTIVERSO_USE_BLAS
    // MKL update
    Log::Fatal("Not implemented yet\n");
#else
    Blob keys = data[0], values = data[1];
	decay_momentum_rate_first_ = data[2].As<float>();
    CHECK(keys.size<int>() == 1 && keys.As<int>() == -1); // Always request whole table
    CHECK(values.size() == size_ * sizeof(T));
	for (int i = 0; i < size_; ++i)
	{
		smooth_gradient_first_[i] = decay_momentum_rate_first_ * smooth_gradient_first_[i] + (1 - decay_momentum_rate_first_) * values.As<T>(i);
		storage_[i] += smooth_gradient_first_[i];
	}
#endif
  }

  void ProcessGet(const std::vector<Blob>& data,
    std::vector<Blob>* result) override {
    size_t key_size = data[0].size<int>();
    CHECK(key_size == 1 && data[0].As<int>() == -1); // Always request the whole table
    Blob key(sizeof(int)); key.As<int>() = server_id_;
    Blob value(storage_.data(), sizeof(T) * size_);
    result->push_back(key);
    result->push_back(value);
  }

  void DumpTable(std::ofstream& os){
    os << decay_momentum_rate_first_ << ' ';
    os << decay_momentum_rate_second_ << ' ';
    os << stepsize_ << ' ';
    for (int i = 0; i < storage_[i].size(); ++i)
      os << storage_[i] << ' ';
    for (int i = 0; i < smooth_gradient_first_.size(); ++i)
      os << smooth_gradient_first_[i] << ' ';
    for (int i = 0; i < smooth_gradient_second_.size(); ++i)
      os << smooth_gradient_second_[i] << ' ';
  }
  void RecoverTable(std::ifstream& in){
    in >> decay_momentum_rate_first_;
    in >> decay_momentum_rate_second_;
    in >> stepsize_;
    for (int i = 0; i < storage_[i].size(); ++i)
      in >> storage_[i];
    for (int i = 0; i < smooth_gradient_first_.size(); ++i)
      in >> smooth_gradient_first_[i];
    for (int i = 0; i < smooth_gradient_second_.size(); ++i)
      in >> smooth_gradient_second_[i];
  }

private:
  int server_id_;
  float decay_momentum_rate_first_;
  float decay_momentum_rate_second_;
  float stepsize_;
  std::vector<T> storage_;
  std::vector<T> smooth_gradient_first_;
  std::vector<T> smooth_gradient_second_;
  size_t size_; // number of element with type T
};
}

#endif // MULTIVERSO_ADAM_ARRAY_TABLE_H_
