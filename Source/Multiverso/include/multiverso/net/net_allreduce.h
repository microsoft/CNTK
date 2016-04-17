#ifndef MULTIVERSO_NET_ALLREDUCE_NET_H_
#define MULTIVERSO_NET_ALLREDUCE_NET_H_

#ifdef MULTIVERSO_USE_ZMQ

#include "zmq_net.h"

#include "allreduce_engine.h"

namespace multiverso {
class AllreduceNetWrapper: public ZMQNetWrapper {
public:
  void Init(int* argc, char** argv) override {
    ZMQNetWrapper::Init(argc, argv);
    bruck_map_ = BruckMap::Construct(rank_, size_);
    recursive_halving_map_ = RecursiveHalvingMap::Construct(rank_, size_);
    allreduce_engine_.Init(this);
  }

  virtual int Connect(int* ranks, char* endpoints[], int size) override {
    ZMQNetWrapper::Connect(ranks, endpoints, size);
    bruck_map_ = BruckMap::Construct(rank_, size_);
    recursive_halving_map_ = RecursiveHalvingMap::Construct(rank_, size_);
    allreduce_engine_.Init(this);
  }

  void Finalize() override { 
    ZMQNetWrapper::Finalize();
  }

  inline void Receive(int rank, char* data, int start, int len) const {
    //note: rank is not used here
    int recv_size = 0;
    while (recv_size < len) {
      int ret_code = zmq_recv(receiver_, data + start + recv_size, len - recv_size, 0);
      if (ret_code < 0) { Log::Error("socket receive error %d", ret_code); }
      recv_size += ret_code;
    }
  }

  inline void Send(int rank, const char* data, int start, int len) const {
    int send_size = 0;
    while (send_size < len) {
      int ret_code = zmq_send(senders_[rank], data + start + send_size, len - send_size, 0);
      if (ret_code < 0) { Log::Error("socket send error %d", ret_code); }
      send_size += ret_code;
    }
  }

  int thread_level_support() override {
    return NetThreadLevel::THREAD_SERIALIZED;
  }

  void Allreduce(char* input, int input_size, int type_size, char* output, ReduceFunction reducer) {
    allreduce_engine_.Allreduce(input, input_size, type_size, output, reducer);
  }

  void Allgather(char* input, int send_size, char* output) {
    allreduce_engine_.Allgather(input, send_size, output);
  }

  void Allgather(char* input, int all_size, int* block_start, int* block_len, char* output) {
    allreduce_engine_.Allgather(input, all_size, block_start, block_len, output);
  }

  void ReduceScatter(char* input, int input_size, int type_size, int* block_start, int* block_len, char* output, ReduceFunction reducer) {
    allreduce_engine_.ReduceScatter(input, input_size, type_size, block_start, block_len, output, reducer);
  }

  inline const BruckMap& GetBruckMap() const {
    return bruck_map_;
  }

  inline const RecursiveHalvingMap& GetRecursiveHalfingMap() const {
    return recursive_halving_map_;
  }

private:

  BruckMap bruck_map_;
  RecursiveHalvingMap recursive_halving_map_;
  AllreduceEngine allreduce_engine_;

};
} // namespace multiverso

#endif // MULTIVERSO_USE_ZEROMQ

#endif // MULTIVERSO_NET_ALLREDUCE_NET_H_