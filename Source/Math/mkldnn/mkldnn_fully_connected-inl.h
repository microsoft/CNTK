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
 * \file mkl_fully_connected-inl.h
 * \brief
 * \author zhenlin.luo@intel.com
 *          lingyan.guo@intel.com
 *
 *
 *******************************************************************************/
#ifndef CNTK_OPERATOR_MKL_DNN_MKLDNN_FULLY_CONNECTED_INL_H_
#define CNTK_OPERATOR_MKL_DNN_MKLDNN_FULLY_CONNECTED_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include "mkl_memory.h"
#include "mkldnn_memory-inl.h"
#include "mkldnn_base-inl.h"
#include "mkl_util-inl.h"

#ifdef USE_MKLDNN
namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <typename Dtype>
class MKLDNNFullyConnectedOp : public MKLDNNLayer<Dtype>
{
    static int s_id_gen;
    int m_id;

public:
    using Mat = Matrix<Dtype>;
    explicit MKLDNNFullyConnectedOp()
        : init_mkldnn_(false),
          fwd_bottom_data(NULL),
          fwd_top_data(NULL),
          fwd_weights_data(NULL),
          bwdd_weights_data(NULL),
          bwdw_bottom_data(NULL),
          bwdd_bottom_diff(NULL),
          bwdd_top_diff(NULL),
          bwdw_top_diff(NULL),
          bwdw_weights_diff(NULL),
          ipFwd_pd(NULL),
          ipBwdData_pd(NULL),
          ipBwdWeights_pd(NULL),
          w_(0),
          h_(0)
    {
        m_id = s_id_gen++;
    }

    ~MKLDNNFullyConnectedOp() {}
    std::string getName()
    {
        std::string name = "MKLDNNFullyConnectedOp_";
        name = name + std::to_string(m_id);
        return name;
    }

private:
    void LayerSetUp(SmallVector<size_t>& dimA, SmallVector<size_t>& dimB)
    {
        UNUSED(dimA);
        this->w_ = 1;
        this->h_ = 1;
        this->channels_ = (int) (dimB[0]);
        this->N_ = (int) dimB[1];
    }
    void InitInnerProductFwd()
    {
        int32_t n = this->M_;
        int32_t w = this->w_;
        int32_t h = this->h_;
        int32_t oc = this->N_;
        int32_t ic = this->channels_;
        bool has_spatial = h > 1 || w > 1;

        // Initialize memory descriptors (fromat = any) to create inner_product descriptor
        mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
        mkldnn::memory::format mfmt = mkldnn::memory::format::any;

        mkldnn::memory::dims bottom_tz =
            (has_spatial) ? mkldnn::memory::dims{n, ic, h, w} : mkldnn::memory::dims{n, ic};
        mkldnn::memory::dims top_tz = {n, oc};
        mkldnn::memory::dims weights_tz =
            (has_spatial) ? mkldnn::memory::dims{oc, ic, h, w} : mkldnn::memory::dims{oc, ic};

        mkldnn::memory::desc init_bottom_md({bottom_tz}, mpcsn, mfmt);
        mkldnn::memory::desc init_top_md({top_tz}, mpcsn, mfmt);
        mkldnn::memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);

        // Initialize inner_product primitive descriptor
        std::shared_ptr<mkldnn::inner_product_forward::desc> ipFwd_desc;

        ipFwd_desc.reset(new mkldnn::inner_product_forward::desc(mkldnn::prop_kind::forward_training, init_bottom_md,
                                                                 init_weights_md, init_top_md));

        mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
        ipFwd_pd.reset(new mkldnn::inner_product_forward::primitive_desc(*ipFwd_desc, cpu_engine));
        assert(ipFwd_pd);

        // Create priv memory primitive descriptors stored as class members
        typedef typename mkldnn::memory::primitive_desc MemPD;

        std::shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(new MemPD(ipFwd_pd->src_primitive_desc()));
        std::shared_ptr<MemPD> prv_fwd_top_data_memory_pd(new MemPD(ipFwd_pd->dst_primitive_desc()));
        std::shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(new MemPD(ipFwd_pd->weights_primitive_desc()));

        mkldnn::memory::format input_mfmt = has_spatial ? mkldnn::memory::format::nchw : mkldnn::memory::format::nc;
        std::shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, input_mfmt}, cpu_engine));

        std::shared_ptr<MemPD> usr_top_data_memory_pd(
            new MemPD({{top_tz}, mpcsn, mkldnn::memory::format::nc}, cpu_engine));
        mkldnn::memory::format weights_mfmt = has_spatial ? mkldnn::memory::format::oihw : mkldnn::memory::format::oi;
        std::shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));

        // ---  init primitive and prv_memory descriptors ----------------------
        fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd));
        fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd));
        fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd));

        // Names are for debugging purposes only.
        fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
        fwd_top_data->name = "fwd_top_data      @ " + this->getName();
        fwd_weights_data->name = "fwd_weights_data  @ " + this->getName();
    }

public:
    virtual void validate(SmallVector<size_t>& dimA, SmallVector<size_t>& dimB)
    {
        if (!init_mkldnn_)
        {
            LayerSetUp(dimA, dimB);
            init_mkldnn_ = true;
        }
    }
    virtual void Forward(const Mat& in, const Mat& weight, Mat& out)
    {
        Dtype* in_ptr = mkl_experimental_direct_get(in);
        Dtype* weight_ptr = mkl_experimental_direct_get(weight);
        Dtype* out_ptr = mkl_experimental_direct_get(out);

        this->M_ = (int) in.GetNumCols();

        if (ipFwd_pd == NULL)
        {
            InitInnerProductFwd();
        }

        std::shared_ptr<mkldnn::memory> fwd_top_data_memory;
        std::shared_ptr<mkldnn::primitive> fwd_bottom_data_primitive, fwd_weights_data_primitive;
        fwd_bottom_data_primitive = fwd_bottom_data->get_converted_prv(in_ptr, false, in);
        fwd_weights_data_primitive = fwd_weights_data->get_converted_prv(weight_ptr, false, weight);

        fwd_top_data_memory = fwd_top_data->create_output_memory(out_ptr, out);

        ipFwd.reset(new mkldnn::inner_product_forward(*ipFwd_pd, *fwd_bottom_data_primitive,
                                                      *fwd_weights_data_primitive, *fwd_top_data_memory));

        ipFwd.submit();

        // if (fwd_top_data->conversion_needed()) {
        //  fwd_top_data->convert_from_prv(out_ptr);
        //}
    }
    void InitInnerProductBwd()
    {
        int32_t n = this->M_;
        int32_t w = this->w_;
        int32_t h = this->h_;
        int32_t oc = this->N_;
        int32_t ic = this->channels_;
        bool has_spatial = h > 1 || w > 1;

        // Initialize memory descriptors (format = any) to create inner_product descriptor
        mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
        mkldnn::memory::format mfmt = mkldnn::memory::format::any;

        mkldnn::memory::dims bottom_tz =
            (has_spatial) ? mkldnn::memory::dims{n, ic, h, w} : mkldnn::memory::dims{n, ic};
        mkldnn::memory::dims top_tz = {n, oc};
        mkldnn::memory::dims weights_tz =
            (has_spatial) ? mkldnn::memory::dims{oc, ic, h, w} : mkldnn::memory::dims{oc, ic};

        mkldnn::memory::desc init_bottom_md({bottom_tz}, mpcsn, mfmt);
        mkldnn::memory::desc init_top_md({top_tz}, mpcsn, mfmt);
        mkldnn::memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);

        // Initialize inner_product primitive descriptor
        std::shared_ptr<mkldnn::inner_product_backward_data::desc> ipBwdData_desc;
        std::shared_ptr<mkldnn::inner_product_backward_weights::desc> ipBwdWeights_desc;

        ipBwdWeights_desc.reset(
            new mkldnn::inner_product_backward_weights::desc(init_bottom_md, init_weights_md, init_top_md));
        ipBwdData_desc.reset(
            new mkldnn::inner_product_backward_data::desc(init_bottom_md, init_weights_md, init_top_md));
        mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
        ipBwdData_pd.reset(
            new mkldnn::inner_product_backward_data::primitive_desc(*ipBwdData_desc, cpu_engine, *ipFwd_pd));
        assert(ipBwdData_pd);
        ipBwdWeights_pd.reset(
            new mkldnn::inner_product_backward_weights::primitive_desc(*ipBwdWeights_desc, cpu_engine, *ipFwd_pd));
        assert(ipBwdWeights_pd);
        // Create priv memory primitive descriptors stored as class members
        typedef typename mkldnn::memory::primitive_desc MemPD;
        std::shared_ptr<MemPD> prv_bwdd_bottom_diff_memory_pd(new MemPD(ipBwdData_pd->diff_src_primitive_desc()));
        std::shared_ptr<MemPD> prv_bwdd_top_diff_memory_pd(new MemPD(ipBwdData_pd->diff_dst_primitive_desc()));
        std::shared_ptr<MemPD> prv_bwdd_weights_data_memory_pd(new MemPD(ipBwdData_pd->weights_primitive_desc()));

        std::shared_ptr<MemPD> prv_bwdw_bottom_data_memory_pd(new MemPD(ipBwdWeights_pd->src_primitive_desc()));
        std::shared_ptr<MemPD> prv_bwdw_top_diff_memory_pd(new MemPD(ipBwdWeights_pd->diff_dst_primitive_desc()));
        std::shared_ptr<MemPD> prv_bwdw_weights_diff_memory_pd(
            new MemPD(ipBwdWeights_pd->diff_weights_primitive_desc()));

        // Create usr memory primitive descriptors stored as class members

        mkldnn::memory::format input_mfmt = has_spatial ? mkldnn::memory::format::nchw : mkldnn::memory::format::nc;
        std::shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, input_mfmt}, cpu_engine));
        std::shared_ptr<MemPD> usr_top_data_memory_pd(
            new MemPD({{top_tz}, mpcsn, mkldnn::memory::format::nc}, cpu_engine));
        mkldnn::memory::format weights_mfmt = has_spatial ? mkldnn::memory::format::oihw : mkldnn::memory::format::oi;
        std::shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));

        // ---  init primitive and prv_memory descriptors ----------------------
        bwdd_bottom_diff.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_bwdd_bottom_diff_memory_pd));
        bwdd_bottom_diff->name = "bwdd_bottom_diff   @ " + this->getName();
        bwdw_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_bwdw_bottom_data_memory_pd));
        bwdw_bottom_data->name = "bwdw_bottom_data   @ " + this->getName();

        bwdd_top_diff.reset(new MKLDNNData<Dtype>(usr_top_data_memory_pd, prv_bwdd_top_diff_memory_pd));
        bwdd_top_diff->name = "bwdd_top_diff      @ " + this->getName();
        bwdw_top_diff.reset(new MKLDNNData<Dtype>(usr_top_data_memory_pd, prv_bwdw_top_diff_memory_pd));
        bwdw_top_diff->name = "bwdw_top_diff      @ " + this->getName();
        ;

        bwdd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_bwdd_weights_data_memory_pd));
        bwdd_weights_data->name = "bwdd_weights_data  @ " + this->getName();
        bwdw_weights_diff.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_bwdw_weights_diff_memory_pd));
        bwdw_weights_diff->name = "bwdw_weights_diff  @ " + this->getName();
        ;
    }
    void BackwardWeight(const Mat& in, const Mat& grad, Mat& gradWeight)
    {
        Dtype* in_ptr = mkl_experimental_direct_get(in);
        Dtype* grad_ptr = mkl_experimental_direct_get(grad);
        Dtype* gradw_ptr = mkl_experimental_direct_get(gradWeight);
        if (ipBwdData_pd == NULL)
        {
            InitInnerProductBwd();
        }

        std::shared_ptr<mkldnn::memory> bwdw_weights_diff_memory;
        std::shared_ptr<mkldnn::primitive> bwdw_top_diff_primitive, bwdw_bottom_data_primitive;
        bwdw_bottom_data_primitive = bwdw_bottom_data->get_converted_prv(in_ptr, false, in);
        bwdw_top_diff_primitive = bwdw_top_diff->get_converted_prv(grad_ptr, false, grad);
        bwdw_weights_diff_memory = bwdw_weights_diff->create_output_memory(gradw_ptr, gradWeight);
        ipBwdWeights.reset(new mkldnn::inner_product_backward_weights(
            *ipBwdWeights_pd, *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive, *bwdw_weights_diff_memory));
        ipBwdWeights.submit();
    }
    void BackwardData(const Mat& grad, const Mat& weight, Mat& gradData)
    {
        Dtype* gradin_ptr = mkl_experimental_direct_get(gradData);
        Dtype* grad_ptr = mkl_experimental_direct_get(grad);
        Dtype* w_ptr = mkl_experimental_direct_get(weight);
        if (ipBwdData_pd == NULL)
        {
            InitInnerProductBwd();
        }

        std::shared_ptr<mkldnn::memory> bwdd_bottom_diff_memory;
        std::shared_ptr<mkldnn::primitive> bwdd_top_diff_primitive, bwdd_weights_data_primitive;
        bwdd_top_diff_primitive = bwdd_top_diff->get_converted_prv(grad_ptr, false, grad);
        bwdd_weights_data_primitive = bwdd_weights_data->get_converted_prv(w_ptr, false, weight);
        bwdd_bottom_diff_memory = bwdd_bottom_diff->create_output_memory(gradin_ptr, gradData);
        ipBwdData.reset(new mkldnn::inner_product_backward_data(
            *ipBwdData_pd, *bwdd_top_diff_primitive, *bwdd_weights_data_primitive, *bwdd_bottom_diff_memory));
        ipBwdData.submit();
    }

private:
    bool init_mkldnn_;
    std::shared_ptr<MKLDNNData<Dtype>> fwd_bottom_data, fwd_top_data, fwd_weights_data, bwdd_weights_data,
        bwdw_bottom_data, bwdd_bottom_diff, bwdd_top_diff, bwdw_top_diff, bwdw_weights_diff;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> ipFwd_pd;
    std::shared_ptr<mkldnn::inner_product_backward_data::primitive_desc> ipBwdData_pd;
    std::shared_ptr<mkldnn::inner_product_backward_weights::primitive_desc> ipBwdWeights_pd;
    MKLDNNPrimitive<Dtype> ipFwd, ipBwdData, ipBwdWeights;

    int32_t w_, h_;
    int M_;
    int channels_;
    int N_;
}; // class MKLDNNFullyConnectedOp
template <>
int MKLDNNFullyConnectedOp<float>::s_id_gen = 1;
template <>
int MKLDNNFullyConnectedOp<double>::s_id_gen = 1;
} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
#endif

#endif // CNTK_OPERATOR_MKL_DNN_MKLDNN_FULLY_CONNECTED_INL_H_
