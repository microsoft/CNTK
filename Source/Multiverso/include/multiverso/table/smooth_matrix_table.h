#ifndef MULTIVERSO_SMOOTH_MATRIX_TABLE_H_
#define MULTIVERSO_SMOOTH_MATRIX_TABLE_H_

#include "multiverso/multiverso.h"
#include "multiverso/util/log.h"
#include "multiverso/util/quantization_util.h"
#include "matrix_table.h"

#include <vector>

namespace multiverso {

template <typename T>
class SmoothMatrixWorkerTable : public MatrixWorkerTable<T> {
public:
	SmoothMatrixWorkerTable(int num_row, int num_col, float smooth_coeff) :
	MatrixWorkerTable<T>(num_row, num_col), smooth_coefficient_(smooth_coeff) {
    Log::Debug("[Init] worker =  %d, type = SmoothMatrixTable, size =  [ %d x %d ], smooth_coeff = %f.\n",
      MV_Rank(), num_row, num_col, smooth_coefficient_);
  }

  int Partition(const std::vector<Blob>& kv,
    std::unordered_map<int, std::vector<Blob>>* out) override {
    CHECK(kv.size() == 1 || kv.size() == 2);
    CHECK_NOTNULL(out);
    
    size_t keys_size = kv[0].size<int>();
    int *keys = reinterpret_cast<int*>(kv[0].data());
    if (keys_size == 1 && keys[0] == -1){
      for (int i = 0; i < num_server_; ++i){
        int rank = MV_ServerIdToRank(i);
        (*out)[rank].push_back(kv[0]);
      }
      if (kv.size() == 2){	//process add values
        for (int i = 0; i < num_server_; ++i){
          int rank = MV_ServerIdToRank(i);
          Blob blob(kv[1].data() + server_offsets_[i] * row_size_,
            (server_offsets_[i + 1] - server_offsets_[i]) * row_size_);
          (*out)[rank].push_back(blob);
		  Blob momentum(&smooth_coefficient_, sizeof(float)); // sending coefficent of smooth gradient to server
		  (*out)[rank].push_back(momentum);
        }
      }
      else {
        CHECK(get_reply_count_ == 0);
        get_reply_count_ = static_cast<int>(out->size());
      }
      return static_cast<int>(out->size());
    }

    //count row number in each server
    std::unordered_map<int, int> count;
    std::vector<int> dest;
    int num_row_each = num_row_ / num_server_;
    for (int i = 0; i < keys_size; ++i){
      int dst = keys[i] / num_row_each;
      dst = (dst == num_server_ ? dst - 1 : dst);
      dest.push_back(dst);
      ++count[dst];
    }
    for (auto& it : count) { // Allocate memory
      int rank = MV_ServerIdToRank(it.first);
      std::vector<Blob>& vec = (*out)[rank];
      vec.push_back(Blob(it.second * sizeof(int)));
      if (kv.size() == 2) vec.push_back(Blob(it.second * row_size_));
    }
    count.clear();

    int offset = 0; 
    for (int i = 0; i < keys_size; ++i) {
      int dst = dest[i];
      int rank = MV_ServerIdToRank(dst);
      (*out)[rank][0].As<int>(count[dst]) = keys[i];
      if (kv.size() == 2){ // copy add values
        memcpy(&((*out)[rank][1].As<T>(count[dst] * num_col_)),
          kv[1].data() + offset, row_size_);
        offset += row_size_;
      }
      ++count[dst];
    }

	if (kv.size() == 2){ // send the coefficent of smooth gradient to each server
		for (int i = 0; i < num_server_; ++i){
			int rank = MV_ServerIdToRank(i);
			Blob momentum(&smooth_coefficient_, sizeof(float)); 
			(*out)[rank].push_back(momentum);
		}
	}

    if (kv.size() == 1){
      CHECK(get_reply_count_ == 0);
      get_reply_count_ = static_cast<int>(out->size());
    }
    return static_cast<int>(out->size());
  }

private:
	float smooth_coefficient_;
};


template <typename T>
class SmoothMatrixServerTable : public MatrixServerTable<T> {
public:
  explicit SmoothMatrixServerTable(int num_row, int num_col, float smooth_coeff = 0.0f) :
	  MatrixServerTable(num_row, num_col), smooth_coefficient_(smooth_coeff) {

    CHECK(server_id_ != -1);
	smooth_gradient_.resize(num_row_ * num_col);

    Log::Debug("[Init] Server =  %d, type = SmoothMatrixTable, size =  [ %d x %d ], total =  [ %d x %d ], smooth_coeff = %f.\n",
	  server_id_, num_row_, num_col_, num_row, num_col_, smooth_coefficient_);
  }

  void ProcessAdd(const std::vector<Blob>& data) override {
    CHECK(data.size() == 2);
    size_t keys_size = data[0].size<int>();
    int *keys = reinterpret_cast<int*>(data[0].data());
    T *values = reinterpret_cast<T*>(data[1].data());
	smooth_gradient_ = data[2].As<float>(); // sent by workers
    // add all values
    if (keys_size == 1 && keys[0] == -1){
      size_t ssize = storage_.size();
      CHECK(ssize == data[1].size<T>());
      for (int i = 0; i < ssize; ++i){
		smooth_gradient_[i] = smooth_coefficient_ * smooth_gradient_[i] + (1 - smooth_coefficient_) * values.As<T>(i);
		storage_[i] += smooth_gradient_[i];
      }
      Log::Debug("[ProcessAdd] Server = %d, adding rows offset = %d, #rows = %d, smooth_coeff = %f.\n",
        server_id_, row_offset_, ssize / num_col_, smooth_coefficient_);
      return;
    }
    CHECK(data[1].size() == keys_size * sizeof(T) * num_col_);

    int offset_v = 0;
    for (int i = 0; i < keys_size; ++i) {
      int offset_s = (keys[i] - row_offset_) * num_col_;
      for (int j = 0; j < num_col_; ++j){
		smooth_gradient_[offset_s] = smooth_coefficient_ * smooth_gradient_[offset_s] + (1 - smooth_coefficient_) * values.As<T>(offset_v);
		storage_[offset_s] += smooth_gradient_[offset_s];
		++offset_v;
		++offset_s;
      }
      Log::Debug("[ProcessAdd] Server = %d, adding #row = %d\n",
        server_id_, keys[i]);
    }
  }

  void Store(Stream* s) override{
	  s->Write(storage_.data(), storage_.size() * sizeof(T));
	  s->Write(&smooth_momentum_, sizeof(float));
	  s->Write(smooth_gradient_.data(), smooth_gradient_.size() * sizeof(T))
  }

  void Load(Stream* s) override{
	  s->Read(storage_.data(), storage_.size() * sizeof(T));
	  s->Read(&smooth_momentum_, sizeof(float));
	  s->Read(smooth_gradient_.data(), smooth_gradient_.size() * sizeof(T));
  }
private:
	float smooth_coefficient_;
	std::vector<T> smooth_gradient_;
};
}
#endif // MULTIVERSO_SMOOTH_MATRIX_TABLE_H_
