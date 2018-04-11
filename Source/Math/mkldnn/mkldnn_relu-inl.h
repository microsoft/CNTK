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
 * \file mkl_relu-inl.h
 * \brief
 * \author zhenlin.luo@intel.com
 *         lingyan.guo@intel.com
 *
 *******************************************************************************/
#ifndef CNTK_OPERATOR_MKL_DNN_MKLDNN_RELU_INL_H_
#define CNTK_OPERATOR_MKL_DNN_MKLDNN_RELU_INL_H_

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
#ifdef MKL_TIMER_PROFILE
#include "TimerUtility.h"
#include "ProgressTracing.h"
#endif
namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <typename Dtype>
class MKLDNNReluOp : public MKLDNNLayer<Dtype>
{
    static int s_id_gen;
    int m_id;

public:
    using Mat = Matrix<Dtype>;
    std::string getName()
    {
        std::string name = "MKLDNNReluOp_";
        name = name + std::to_string(m_id);
        return name;
    }
    MKLDNNReluOp(TensorShape inOutT, ImageLayoutKind imageLayout)
        : MKLDNNLayer<Dtype>(),
          fwd_top_data(NULL),
          fwd_bottom_data(NULL),
          num_(0),
          width_(0),
          height_(0),
          channels_(0),
          m_inOutT(inOutT),
          m_imageLayout(imageLayout)
    {
        init_mkldnn_ = false;
        m_id = s_id_gen++;
    }
    ~MKLDNNReluOp() {}

private:
    void LayerSetup(int batchsize)
    {
        ImageDimensions inT(m_inOutT, m_imageLayout);
        this->width_ = (int) inT.w();
        this->height_ = (int) inT.h();
        this->channels_ = (int) inT.c();
        this->num_ = batchsize;
    }
    void InitReLUFwd(const Mat& in)
    {
        void* bottom_data = reinterpret_cast<void*>(mkl_prv_data<Dtype>(in));
        std::shared_ptr<MKLDNNMemoryDescriptor<Dtype>> bottom_prv_descriptor = get_mkldnn_prv_descriptor<Dtype>(in);
        std::shared_ptr<mkldnn::memory::desc> bottom_data_md, top_data_md;
        std::shared_ptr<mkldnn::memory::primitive_desc> usr_mpd(NULL), prv_mpd(NULL);

        int32_t n = this->num_;
        int32_t iw = this->width_;
        int32_t ih = this->height_;
        int32_t ic = this->channels_;
        Dtype negative_slope = 0;
        mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
        mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;

        if (bottom_data != NULL)
        {
            bottom_data_md.reset(new mkldnn::memory::desc(bottom_prv_descriptor->prv_memory_pd()->desc()));
            usr_mpd = bottom_prv_descriptor->usr_memory_pd();
            prv_mpd = bottom_prv_descriptor->prv_memory_pd();
        }
        else
        {
            bottom_data_md.reset(new mkldnn::memory::desc({{n, ic, ih, iw}}, mpcsn, mkldnn::memory::format::nchw));
            usr_mpd.reset(new mkldnn::memory::primitive_desc(*bottom_data_md, cpu_engine));
        }
        top_data_md = bottom_data_md;
        // ---- Initialize relu primitive descriptor -------------
        mkldnn::eltwise_forward::desc fwd_inference_desc(mkldnn::prop_kind::forward_scoring, mkldnn::eltwise_relu,
                                                         *bottom_data_md, negative_slope);
        fwd_inference_pd.reset(new mkldnn::relu_forward::primitive_desc(fwd_inference_desc, cpu_engine));
        mkldnn::eltwise_forward::desc fwd_training_desc(mkldnn::prop_kind::forward_training, mkldnn::eltwise_relu,
                                                        *bottom_data_md, negative_slope);
        fwd_training_pd.reset(new mkldnn::relu_forward::primitive_desc(fwd_training_desc, cpu_engine));
        fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd));
        fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
        fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd));
        fwd_top_data->name = "fwd_top_data   @ " + this->getName();
    }

public:
    virtual void Forward(const Mat& in, const Mat& out, bool inferenceOnly)
    {
#ifdef MKL_TIMER_PROFILE
        Timer timer;
        timer.Start();
#endif
        Dtype* in_ptr = mkl_experimental_direct_get(in);
        Dtype* out_ptr = mkl_experimental_direct_get(out);
        if (!init_mkldnn_)
        {
            LayerSetup((int) in.GetNumCols());
            init_mkldnn_ = true;
        }
        if (fwd_inference_pd == NULL)
            InitReLUFwd(in);
        // ---- Initialize memory descriptors -------------
        std::shared_ptr<mkldnn::memory> input_primitive;
        input_primitive = fwd_bottom_data->get_converted_prv(in_ptr, false, in);
        std::shared_ptr<mkldnn::memory> output_memory = fwd_top_data->create_output_memory(out_ptr, out);
        MKLDNNPrimitive<Dtype> reluFwd;
        if (!inferenceOnly)
        {
            reluFwd.reset(new mkldnn::relu_forward(*fwd_training_pd, *input_primitive, *output_memory));
        }
        else
        {
            reluFwd.reset(new mkldnn::relu_forward(*fwd_inference_pd, *input_primitive, *output_memory));
        }
#ifdef MKL_TIMER_PROFILE
        timer.Stop();
        LOGPRINTF(stderr, "mklrelu fwd pre submit time: %f \n", timer.ElapsedSeconds());
        timer.Start();
#endif
        reluFwd.submit();
#ifdef MKL_TIMER_PROFILE
        timer.Stop();
        LOGPRINTF(stderr, "mklrelu fwd submit time: %f \n", timer.ElapsedSeconds());
#endif
    }

    void InitReLUBwd(const Mat& src)
    {
        int32_t n = this->num_;
        int32_t iw = this->width_;
        int32_t ih = this->height_;
        int32_t ic = this->channels_;
        Dtype negative_slope = 0;
        void* src_data = const_cast<Dtype*>(mkl_prv_data<Dtype>(src));
        bool src_is_prv = (src_data != NULL);
        mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
        mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
        // ---- Initialize memory descriptors -------------
        std::shared_ptr<mkldnn::memory::desc> bottom_diff_md;
        std::shared_ptr<mkldnn::memory::desc> top_diff_md;
        std::shared_ptr<mkldnn::memory::desc> top_data_md;

        std::shared_ptr<mkldnn::memory::primitive_desc> usr_diff_mpd;
        std::shared_ptr<mkldnn::memory::primitive_desc> prv_diff_mpd;

        if (src_is_prv)
        {
            std::shared_ptr<MKLDNNMemoryDescriptor<Dtype>> mem_descr = get_mkldnn_prv_descriptor<Dtype>(src);
            top_diff_md.reset(new mkldnn::memory::desc(mem_descr->prv_memory_pd()->desc()));
            usr_diff_mpd = mem_descr->usr_memory_pd();
            prv_diff_mpd = mem_descr->prv_memory_pd();
        }
        else
        {
            top_diff_md.reset(new mkldnn::memory::desc({{n, ic, ih, iw}}, mpcsn, mkldnn::memory::format::nchw));
            usr_diff_mpd.reset(new mkldnn::memory::primitive_desc(*top_diff_md, cpu_engine));
        }
        top_data_md = top_diff_md;
        bottom_diff_md = top_diff_md;
        mkldnn::eltwise_backward::desc reluBwd_desc(mkldnn::eltwise_relu, *top_diff_md, *top_data_md, negative_slope);
        bwd_pd.reset(new mkldnn::relu_backward::primitive_desc(reluBwd_desc, cpu_engine, *fwd_training_pd));
        bwd_top_diff.reset(new MKLDNNData<Dtype>(usr_diff_mpd, prv_diff_mpd));
        bwd_top_diff->name = "bwd_top_diff   @ " + this->getName();
        bwd_bottom_diff.reset(new MKLDNNData<Dtype>(usr_diff_mpd, prv_diff_mpd));
        bwd_bottom_diff->name = "bwd_bottom_diff   @ " + this->getName();
        bwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_diff_mpd, prv_diff_mpd));
        bwd_bottom_data->name = "bwd_bottom_data   @ " + this->getName();
    }
    virtual void Backward(const Mat& in, const Mat& srcGrad, Mat& grad)
    {
#ifdef MKL_TIMER_PROFILE
        Timer timer;
        timer.Start();
#endif
        if (!init_mkldnn_)
        {
            LayerSetup((int) in.GetNumCols());
            init_mkldnn_ = true;
        }
        if (fwd_inference_pd == NULL)
            InitReLUFwd(in);
        Dtype* in_ptr = mkl_experimental_direct_get(in);
        Dtype* srcgrad_ptr = mkl_experimental_direct_get(srcGrad);
        Dtype* grad_ptr = mkl_experimental_direct_get(grad);
        if (bwd_pd == nullptr)
        {
            InitReLUBwd(in);
        }
        std::shared_ptr<mkldnn::memory> src_memory, diff_dst_memory, diff_src_memory;
        src_memory = bwd_bottom_data->get_converted_prv(in_ptr, false, in);
        diff_dst_memory = bwd_top_diff->get_converted_prv(srcgrad_ptr, false, srcGrad);
        diff_src_memory = bwd_bottom_diff->create_output_memory(grad_ptr, grad);
        MKLDNNPrimitive<Dtype> reluBwd;
        reluBwd.reset(new mkldnn::relu_backward(*bwd_pd, *src_memory, *diff_dst_memory, *diff_src_memory));
#ifdef MKL_TIMER_PROFILE
        timer.Stop();
        LOGPRINTF(stderr, "mklrelu bwd pre submit time: %f \n", timer.ElapsedSeconds());
        timer.Start();
#endif
        reluBwd.submit();
#ifdef MKL_TIMER_PROFILE
        timer.Stop();
        LOGPRINTF(stderr, "mklrelu bwd submit time: %f \n", timer.ElapsedSeconds());
#endif
    }

private:
    bool init_mkldnn_;

    std::shared_ptr<MKLDNNData<Dtype>> fwd_top_data, fwd_bottom_data;
    std::shared_ptr<MKLDNNData<Dtype>> bwd_bottom_data, bwd_top_diff;
    std::shared_ptr<MKLDNNData<Dtype>> bwd_bottom_diff;
    std::shared_ptr<mkldnn::relu_forward::primitive_desc> fwd_inference_pd;
    std::shared_ptr<mkldnn::relu_forward::primitive_desc> fwd_training_pd;
    std::shared_ptr<mkldnn::relu_backward::primitive_desc> bwd_pd;
    int32_t num_, width_, height_, channels_;
    TensorShape m_inOutT;
    ImageLayoutKind m_imageLayout;
}; // class MKLDNNReluOp
template <>
int MKLDNNReluOp<float>::s_id_gen = 1;
template <>
int MKLDNNReluOp<double>::s_id_gen = 1;
} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
#endif
#endif // CNTK_OPERATOR_MKL_DNN_MKLDNN_RELU_INL_H_
