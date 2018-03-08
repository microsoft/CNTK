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
#include <string.h>
#include <mkldnn.hpp>
#include "../Matrix.h"

#define STRINGIFy(s) #s
#define STRINGIFY(s) STRINGIFy(s)
#ifdef _WIN32
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

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

struct attr_t {
    struct scale_t {
        enum policy_t { NONE = 0, COMMON, PER_OC, POLICY_TOTAL };

        bool is_def() const { return policy == NONE; }

        scale_t(float s, policy_t p = NONE) :
            scale(s) {
            policy = p;
        }
        policy_t policy = NONE;
        float scale = 1.;
    };

    struct post_ops_t {
        enum kind_t { SUM, RELU, KIND_TOTAL };
        //static kind_t str2kind(const char *str);
        //static const char *kind2str(kind_t kind);

        struct entry_t {
            kind_t kind;
            union {
                struct { float scale; } sum;
                struct {
                    // eltwise algorithm in future
                    float scale, alpha, beta; // unused now
                } eltwise;
            };
        };

        post_ops_t() : len(0) {}

        bool is_def() const { return len == 0; }

        enum { capacity = 4 };
        int len;
        entry_t entry[capacity];
    };


    void mkldnn_attr_create() {
        mkldnn_attr = mkldnn::primitive_attr();
        mkldnn_attr.set_int_output_round_mode(rmode);
        if (!oscale.is_def()) {
            const int count = 1;
            const int mask = 0;
            std::vector<float> s(count, oscale.scale);
            mkldnn_attr.set_output_scales(mask, s);
        }
        if (!pops.is_def()) {
            mkldnn::post_ops mkldnn_pops = mkldnn::post_ops();
            for (int idx = 0; idx < pops.len; ++idx) {
                const auto &e = pops.entry[idx];
                switch (pops.entry[idx].kind) {
                case attr_t::post_ops_t::SUM:
                    mkldnn_pops.append_sum(e.sum.scale);
                    break;
                case attr_t::post_ops_t::RELU:
                    mkldnn_pops.append_eltwise(e.eltwise.scale,
                        mkldnn::algorithm::eltwise_relu, e.eltwise.alpha,
                        e.eltwise.beta);
                    break;
                default:
                    assert(!"unknown attr::post_ops::kind");
                }
            }
            mkldnn_attr.set_post_ops(mkldnn_pops);

            const mkldnn::post_ops c_ops = mkldnn_attr.get_post_ops();
            // assert(mkldnn_post_ops_len(c_ops) == attr.post_ops.len);
        }
    }

    attr_t(mkldnn::round_mode rm, float s,
        scale_t::policy_t p = scale_t::policy_t::NONE) :
        rmode(rm), oscale(s, p), mkldnn_attr(){
    }

    attr_t() :
        rmode(mkldnn::round_mode::round_nearest),
        oscale(1.0), mkldnn_attr() {
    }

    mkldnn::round_mode rmode = mkldnn::round_mode::round_nearest;
    scale_t oscale;
    mkldnn::primitive_attr mkldnn_attr;
    post_ops_t pops;
    bool is_def() const {
        return true
            && rmode == mkldnn::round_mode::round_nearest
            && oscale.is_def();
    }
};
}}}
#endif

#endif  // CNTK_OPERATOR_MKL_MKL_UTIL_INL_H_
