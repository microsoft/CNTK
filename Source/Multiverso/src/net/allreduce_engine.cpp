#include <string.h>
#include <algorithm>

#include "multiverso/net/allreduce_engine.h"

namespace multiverso {

AllreduceEngine::AllreduceEngine()
  :block_start_(nullptr), block_len_(nullptr), buffer_(nullptr) {

}

void AllreduceEngine::Init(const NetInterface* linkers) {
  linkers_ = linkers;
  rank_ = linkers_->rank();
  num_machines_ = linkers_->size();
  bruck_map_ = BruckMap::Construct(rank_, num_machines_);
  recursive_halving_map_ = RecursiveHalvingMap::Construct(rank_, num_machines_);
  block_start_ = new int[num_machines_];
  block_len_ = new int[num_machines_];
  buffer_size_ = 1024 * 1024;
  buffer_ = new char[buffer_size_];
}

AllreduceEngine::~AllreduceEngine() {
  if (block_start_ != nullptr) { delete[]block_start_; }
  if (block_len_ != nullptr) { delete[]block_len_; }
  if (buffer_ != nullptr) { delete[] buffer_; }
}

void AllreduceEngine::Allreduce(char* input, int input_size, int type_size, char* output, ReduceFunction reducer) {

  int count = input_size / type_size;
  //if small package or small count , do it by all gather.(reduce the communication times.)
  if (count < num_machines_ || input_size < 4096) {
    AllreduceByAllGather(input, input_size, type_size, output, reducer);
    return;
  }
  //assign the blocks to every rank_s.
  int step = (count + num_machines_ - 1) / num_machines_;
  if (step < 1) {
    step = 1;
  }
  block_start_[0] = 0;
  for (int i = 0; i < num_machines_ - 1; ++i) {
    block_len_[i] = step * type_size < input_size - block_start_[i] ? step * type_size : input_size - block_start_[i];
    block_start_[i + 1] = block_start_[i] + block_len_[i];
  }
  block_len_[num_machines_ - 1] = input_size - block_start_[num_machines_ - 1];
  //do reduce scatter
  ReduceScatter(input, input_size, type_size, block_start_, block_len_, output, reducer);
  //do all gather
  Allgather(output, input_size, block_start_, block_len_, output);
}

// REVIEW(feiga): the third argument type_size never used
void AllreduceEngine::AllreduceByAllGather(char* input, int input_size, int, char* output, ReduceFunction reducer) {
  //assign blocks
  int all_size = input_size * num_machines_;
  block_start_[0] = 0;
  block_len_[0] = input_size;
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
    block_len_[i] = input_size;
  }

  if (input_size*num_machines_ > buffer_size_) {
    delete[] buffer_;
    buffer_size_ = input_size*num_machines_;
    buffer_ = new char[buffer_size_];
  }
  Allgather(input, all_size, block_start_, block_len_, buffer_);
  for (int i = 1; i < num_machines_; ++i) {
    reducer(buffer_ + block_start_[i], buffer_ + block_start_[0], input_size);
  }
  std::memcpy(output, buffer_, input_size);
}

void AllreduceEngine::Allgather(char* input, int send_size, char* output) {
  //assign blocks
  block_start_[0] = 0;
  block_len_[0] = send_size;
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
    block_len_[i] = send_size;
  }
  Allgather(input, send_size * num_machines_, block_start_, block_len_, output);
}

void AllreduceEngine::Allgather(char* input, int all_size, int* block_start, int* block_len, char* output) {
  int write_ptr = 0;
  std::memcpy(output, input, block_len[rank_]);
  write_ptr += block_len[rank_];
  int accumulated_block = 1;
  for (int i = 0; i < bruck_map_.k; ++i) {
    int cur_block_size = (1 << i) < num_machines_ - accumulated_block ? (1 << i) : num_machines_ - accumulated_block;
    int target = bruck_map_.out_ranks[i];
    int send_len = 0;
    for (int j = 0; j < cur_block_size; ++j) {
      send_len += block_len[(rank_ + j) % num_machines_];
    }

    int incoming = bruck_map_.in_ranks[i];
    int need_recv_cnt = 0;
    for (int j = 0; j < cur_block_size; ++j) {
      need_recv_cnt += block_len[(rank_ + accumulated_block + j) % num_machines_];
    }

    linkers_->SendRecv(target, output, send_len, incoming, output + write_ptr, need_recv_cnt);
    write_ptr += need_recv_cnt;
    accumulated_block += cur_block_size;
  }
  //rotate right 
  std::reverse<char*>(output, output + all_size);
  std::reverse<char*>(output, output + block_start[rank_]);
  std::reverse<char*>(output + block_start[rank_], output + all_size);
}

// REVIEW(feiga): the third argument type_size never used
void AllreduceEngine::ReduceScatter(char* input, int input_size, int, int* block_start, int* block_len, char* output, ReduceFunction reducer) {

  bool is_powerof_2 = (num_machines_ & (num_machines_ - 1)) == 0 ? true : false;
  if (!is_powerof_2) {
    if (recursive_halving_map_.type == RecursiveHalvingNodeType::Other) {
      //send local data to neighbor first
      linkers_->SendTo(recursive_halving_map_.neighbor, input, input_size);
    }
    else if (recursive_halving_map_.type == RecursiveHalvingNodeType::GroupLeader) {
      //receive neighbor data first
      int need_recv_cnt = input_size;
      linkers_->RecvFrom(recursive_halving_map_.neighbor, output, need_recv_cnt);
      reducer(output, input, input_size);
    }
  }
  //start recursive halfing
  if (recursive_halving_map_.type != RecursiveHalvingNodeType::Other) {

    for (int i = 0; i < recursive_halving_map_.k; ++i) {
      int target = recursive_halving_map_.ranks[i];
      int send_block_start = recursive_halving_map_.send_block_start[i];
      int recv_block_start = recursive_halving_map_.recv_block_start[i];
      int send_size = 0;
      for (int j = 0; j < recursive_halving_map_.send_block_len[i]; ++j) {
        send_size += block_len[send_block_start + j];
      }

      int need_recv_cnt = 0;
      for (int j = 0; j < recursive_halving_map_.recv_block_len[i]; ++j) {
        need_recv_cnt += block_len[recv_block_start + j];
      }

      linkers_->SendRecv(target, input + block_start[send_block_start], send_size, target, output, need_recv_cnt);
      //reduce
      reducer(output, input + block_start[recv_block_start], need_recv_cnt);
    }
  }
  int my_reduce_block_idx = rank_;

  if (!is_powerof_2) {
    if (recursive_halving_map_.type == RecursiveHalvingNodeType::GroupLeader) {
      //send result to neighbor
      linkers_->SendTo(recursive_halving_map_.neighbor, input + block_start[recursive_halving_map_.neighbor], block_len[recursive_halving_map_.neighbor]);
    }
    else if (recursive_halving_map_.type == RecursiveHalvingNodeType::Other) {
      //receive result from neighbor
      int need_recv_cnt = block_len[my_reduce_block_idx];
      linkers_->RecvFrom(recursive_halving_map_.neighbor, output, need_recv_cnt);
      return;
    }
  }
  std::memcpy(output, input + block_start[my_reduce_block_idx], block_len[my_reduce_block_idx]);
}

}
