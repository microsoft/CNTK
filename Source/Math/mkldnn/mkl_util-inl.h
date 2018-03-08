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
* \file mkl_util-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef CNTK_OPERATOR_MKL_MKL_UTIL_INL_H_
#define CNTK_OPERATOR_MKL_MKL_UTIL_INL_H_
#include <vector>
#include <assert.h>
#include "../Matrix.h"

#if USE_MKLDNN
namespace Microsoft {  namespace MSR { namespace CNTK {


template<typename DType>
inline DType * mkl_prv_data(const Matrix<DType> &b) {
    std::shared_ptr<MKLMemHolder> bottom_data_mem = b.MklMem();
    bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
    if (mem_valid) {
        return reinterpret_cast<DType*>(bottom_data_mem->prv_data());
    }
    return NULL;
}

template<typename DType>
inline int mkl_prv_count(const Matrix<DType> &b) {
    std::shared_ptr<MKLMemHolder> bottom_data_mem = b.MklMem();
    bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
    if (mem_valid) {
        return bottom_data_mem->prv_count();
    }
    return 0;
}
template<typename DType>
inline void mkl_set_priv_flag(const Matrix<DType> &b) {
    std::shared_ptr<MKLMemHolder> bottom_data_mem = b.MklMem();
    bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
    if (mem_valid) {
        bottom_data_mem->disable_prv_2_cpu(true);
    }
}

template<typename DType>
inline std::shared_ptr<MKLDNNData<DType> > mkl_get_mem_desc(
  const std::shared_ptr<MKLMemHolder> data_mem) {
  std::shared_ptr<PrvMemDescr> prv_descriptor =
    data_mem->get_prv_descriptor();
  assert(prv_descriptor->get_descr_type() ==
    PrvMemDescr::PRV_DESCR_MKLDNN);
  std::shared_ptr<MKLDNNData<DType> > mem_descr
    = std::static_pointer_cast<MKLDNNData<DType>>(prv_descriptor);
  assert(mem_descr != NULL);
  return mem_descr;
}

template<typename DType>
inline DType* mkl_experimental_direct_get(const Matrix<DType> &b) {
  mkl_set_priv_flag(b);
  return (DType*)b.Data();
}

}}}
#endif

#endif  // CNTK_OPERATOR_MKL_MKL_UTIL_INL_H_
