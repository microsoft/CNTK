/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkl_memory.cc
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef CNTK_OPERATOR_MKL_MKL_MEMORY_H_
#define CNTK_OPERATOR_MKL_MKL_MEMORY_H_

#include <string>
#include <vector>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

#define USR_DESC_MAX_DIMS 12

  /** A type to describe tensor dimensions. */
typedef int usr_desc_dims_t[USR_DESC_MAX_DIMS];

// Base class
struct PrvMemDescr {
  virtual void convert_from_prv(void* cpu_ptr) = 0;
  virtual void convert_to_prv(void* cpu_ptr) = 0;
  virtual void convert_from_other(std::shared_ptr<PrvMemDescr> other) = 0;
  virtual void* prv_ptr() = 0;
  // returns true for matching layouts
  virtual bool layout_compare(std::shared_ptr<PrvMemDescr> other) = 0;
  virtual size_t prv_count() = 0;
  virtual size_t prv_size() = 0;
  virtual bool get_usr_desc(usr_desc_dims_t usr_desc_dims, int& dim_size) = 0;
  virtual std::shared_ptr<PrvMemDescr> get_copy() = 0;
  virtual void get_sum(std::shared_ptr<PrvMemDescr> other) = 0;
  // This might help using prv_ptr_ by different accelerators/engines
  enum PrvDescrType {
    PRV_DESCR_MKL2017,
    PRV_DESCR_MKLDNN
  };
  virtual PrvDescrType get_descr_type() = 0;
};

#ifdef USE_MKLDNN
// Currently HEAD_AT_PRV do not free CPU data
enum SyncedHead {
  HEAD_AT_CPU,
  HEAD_AT_PRV,
};


struct MKLMemHolder {
  SyncedHead head_;
  std::shared_ptr<PrvMemDescr> prv_descriptor_;
  bool  b_disable_prv_2_cpu;
  bool  b_eager_mode;
  void disable_prv_2_cpu(bool flag) {
    b_disable_prv_2_cpu = flag;
  }
  void set_eager_mode(bool eager_mode) {
    b_eager_mode = eager_mode;
  }
  void set_prv_descriptor(std::shared_ptr<PrvMemDescr> descriptor) {
    if (descriptor != nullptr)
      head_ = HEAD_AT_PRV;
    prv_descriptor_ = descriptor;
  }
  std::shared_ptr<PrvMemDescr> get_prv_descriptor() {
    return  prv_descriptor_;
  }
  bool head_at_cpu() {
    return (head_ == HEAD_AT_CPU) ? true : false;
  }
  bool head_at_prv() {
    return (head_ == HEAD_AT_PRV) ? true : false;
  }
  void* prv_data() {
    if (head_ != HEAD_AT_PRV) {
      return NULL;
    }
    if (prv_descriptor_ == NULL) {
      LogicError("prv_descriptor_  is NULL");
    }
    assert(prv_descriptor_.get());
    return reinterpret_cast<void*>(prv_descriptor_->prv_ptr());
  }
  void clear() {
    zero_init();
  }
  virtual size_t prv_count() {
    if (head_ != HEAD_AT_PRV) {
      return 0;
    }
    if (prv_descriptor_ == NULL) {
      LogicError("prv_descriptor_  is NULL");
    }
    assert(prv_descriptor_.get());
    return prv_descriptor_->prv_count();
  }
  static std::shared_ptr<MKLMemHolder> create() {
    return std::make_shared<MKLMemHolder>();
  }
  void  check_and_prv_to_cpu(void *dptr_) {
    if (!b_disable_prv_2_cpu && head_ == HEAD_AT_PRV) {
      assert(prv_descriptor_ != nullptr);
      prv_descriptor_->convert_from_prv(dptr_);
      // Because operator use CPU & maybe change it, change to CPU Flag
      head_ = HEAD_AT_CPU;
    }
    if (b_disable_prv_2_cpu) {
      b_disable_prv_2_cpu = false;
    }
  }
  void AssignPrvFrom(MKLMemHolder& from) {
    head_ = from.head_;
    b_disable_prv_2_cpu = false;
    b_eager_mode = false;
    if (HEAD_AT_PRV == head_) {
      prv_descriptor_ = from.prv_descriptor_->get_copy();
    }
  }
  void SumPrvFrom(MKLMemHolder& from) {
      if (HEAD_AT_PRV != head_ || !from.head_at_prv()) return;
      prv_descriptor_->get_sum(from.prv_descriptor_);
  }
  void zero_init()
  {
    head_ = HEAD_AT_CPU;
    prv_descriptor_ = nullptr;
    b_disable_prv_2_cpu = false;
    b_eager_mode = false;
  }
  MKLMemHolder()
  {
    zero_init();
  }
};
#else
struct MKLMemHolder {
 public:
  virtual std::shared_ptr<PrvMemDescr> get_prv_descriptor() = 0;
};
#endif

}}}
#endif  // CNTK_OPERATOR_MKL_MKL_MEMORY_H_
