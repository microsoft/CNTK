/*******************************************************************************
 * Copyright 2017 Intel Corporation
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
 * \file mkl_convolution-inl.h
 * \brief
 * \author lingyan.guo@intel.com
 *         zhenlin.luo@intel.com
 *
 *******************************************************************************/
#ifndef CNTK_OPERATOR_MKL_DNN_MKLDNN_CONVOLUTION_INL_H_
#define CNTK_OPERATOR_MKL_DNN_MKLDNN_CONVOLUTION_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include "mkl_memory.h"
#include "mkldnn_memory-inl.h"
#include "mkl_conv-common-inl.h"
#include "mkldnn_base-inl.h"
namespace Microsoft
{
namespace MSR
{
namespace CNTK
{
extern void GetSizesAndStrides(int dimension, const TensorShape& shape, size_t lastDim, SmallVector<size_t>& sizes,
                               SmallVector<size_t>& strides, size_t mapCount = 0);
extern void GetInputOffsets(const ConvolveGeometry* geometry, SmallVector<int>& inputOffset);
} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
#ifdef USE_MKLDNN

namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <typename DType>
class MKLDNNConvolutionOp : public MKLDNNLayer<DType>, public MKLConvCommon<DType>
{
    static int s_id_gen;
    int m_id;

public:
    using Mat = Matrix<DType>;
    std::string getName()
    {
        std::string name = "MKLDNNConvolutionOp_";
        name = name + std::to_string(m_id);
        return name;
    }

    explicit MKLDNNConvolutionOp(ConvolveGeometryPtr geometry, ImageLayoutKind imageLayout, bool bias = false,
                                 bool relu = false)
        : MKLDNNLayer<DType>(),
          dilate_w(0),
          dilate_h(0),
          fwd_bottom_data(NULL),
          fwd_top_data(NULL),
          fwd_weights_data(NULL),
          fwd_bias_data(NULL),
          convFwd_pd(NULL),
          convBwdData_pd(NULL),
          convBwdWeights_pd(NULL),
          init_gbias(-1),
          b_init_convBwdData(false),
          b_init_convBwdWeights(false),
          b_init_convFwd(false)
    {
        b_init_conv = false;
        m_geometry = geometry;
        m_imageLayout = imageLayout;
        m_bias = bias;
        m_relu = relu;
        m_id = s_id_gen++;
    }

    virtual ~MKLDNNConvolutionOp() {}

    void init_properties(int batchSize)
    {
        this->num_ = batchSize;
        this->group_ = 1; // TODO: CNTK support group?
        // Check ComputeOutputShape
        if (m_geometry->InputShape().GetRank() == 3)
        {
            ImageDimensions inT(m_geometry->InputShape(), m_imageLayout);
            ImageDimensions outT(m_geometry->OutputShape(), m_imageLayout);
            ImageDimensions kernelT(m_geometry->KernelShape(), m_imageLayout);
            ImageDimensions strideT(m_geometry->Stride(), m_imageLayout);

            this->stride_w_ = (int) strideT.w();
            this->stride_h_ = (int) strideT.h();
            this->width_ = (int) inT.w();
            this->height_ = (int) inT.h();

            this->kernel_w_ = (int) kernelT.w();
            this->kernel_h_ = (int) kernelT.h();
            this->channels_ = (int) inT.c();

            this->width_out_ = (int) outT.w();
            this->height_out_ = (int) outT.h();
            this->channel_output_ = (int) outT.c();
        }
        else
        {
            int dimension = 4;
            SmallVector<size_t> outputSize, outputStrides, filterSize, filterStrides, inputSize, inputStrides,
                stridesSize, stridesStrides;
            SmallVector<int> inputOffset;
            size_t mapCount = m_geometry->GetMapCount(m_geometry->KernelShape().GetRank() - 1);
            GetSizesAndStrides(dimension, m_geometry->OutputShape(), batchSize, outputSize, outputStrides, mapCount);
            GetSizesAndStrides(dimension, m_geometry->KernelShape(), mapCount, filterSize, filterStrides);
            GetSizesAndStrides(dimension, m_geometry->InputShape(), batchSize, inputSize, inputStrides);
            GetSizesAndStrides(dimension, m_geometry->Stride(), batchSize, stridesSize, stridesStrides);
            this->width_ = (int) inputSize[0];
            this->height_ = (int) inputSize[1];
            this->channels_ = (int) inputSize[2];
            this->kernel_w_ = (int) filterSize[0];
            this->kernel_h_ = (int) filterSize[1];

            this->width_out_ = (int) outputSize[0];
            this->height_out_ = (int) outputSize[1];
            this->channel_output_ = (int) outputSize[2];

            this->stride_w_ = (int) stridesSize[0];
            this->stride_h_ = (int) stridesSize[1];
        }
        if (m_geometry->GetDilation(0) > 1)
            this->dilate_w = (int) m_geometry->GetDilation(0) - 1;
        if (m_geometry->GetDilation(1) > 1)
            this->dilate_h = (int) m_geometry->GetDilation(1) - 1;

        const SmallVector<bool>& autopad = m_geometry->AutoPad();
        int autopad_size = (int) autopad.size();
        const TensorShape& padShape = m_geometry->LowerPad();
        int pad_size = (int) padShape.size();
        // For CHW
        if (autopad_size > 0 && autopad[0])
        {
            this->pad_l_w_ = m_geometry->GetLowerPad(0);
        }
        else if (pad_size > 0)
        {
            this->pad_l_w_ = (int) padShape[0];
        }
        if (autopad_size > 1 && autopad[1])
        {
            this->pad_l_h_ = m_geometry->GetLowerPad(1);
        }
        else if (pad_size > 1)
        {
            this->pad_l_h_ = (int) padShape[1];
        }
        this->pad_r_h_ = (this->height_out_ - 1) * this->stride_h_ - this->pad_l_h_ - this->height_ +
                         ((this->kernel_h_ - 1) * (this->dilate_h + 1) + 1);
        this->pad_r_w_ = (this->width_out_ - 1) * this->stride_w_ - this->pad_l_w_ - this->width_ +
                         ((this->kernel_w_ - 1) * (this->dilate_w + 1) + 1);
    }

private:
    void InitForward(bool inferenceOnly)
    {
        auto propagation = (inferenceOnly) ? mkldnn::prop_kind::forward_scoring : mkldnn::prop_kind::forward_training;
        if (m_relu)
            propagation = mkldnn::prop_kind::forward_inference;
        int32_t g = std::max(this->group_, 1);
        int32_t n = this->num_;
        int32_t iw = this->width_;
        int32_t ih = this->height_;
        int32_t ic = this->channels_;

        int32_t ow = this->width_out_;
        int32_t oh = this->height_out_;
        int32_t oc = this->channel_output_;

        int32_t kw = this->kernel_w_;
        int32_t kh = this->kernel_h_;
        mkldnn::memory::dims convolutionStrides{static_cast<int>(this->stride_h_), static_cast<int>(this->stride_w_)};
        mkldnn::memory::dims padding_l{this->pad_l_h_, this->pad_l_w_};
        mkldnn::memory::dims padding_r{this->pad_r_h_, this->pad_r_w_};
        mkldnn::memory::dims dnn_dilate{this->dilate_h, this->dilate_w};
        mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
        mkldnn::memory::format mfmt_any = mkldnn::memory::format::any;
        mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();

        mkldnn::memory::dims bottom_tz = {n, ic, ih, iw};
        mkldnn::memory::dims bias_tz = {oc};
        mkldnn::memory::dims top_tz = {n, oc, oh, ow};
        mkldnn::memory::dims weights_tz =
            (g != 1) ? mkldnn::memory::dims{g, oc / g, ic / g, kh, kw} : mkldnn::memory::dims{oc, ic, kh, kw};

        mkldnn::memory::desc init_bottom_md({bottom_tz}, mpcsn, mfmt_any);
        mkldnn::memory::desc init_bias_md({bias_tz}, mpcsn, mfmt_any);
        mkldnn::memory::desc init_top_md({top_tz}, mpcsn, mfmt_any);
        mkldnn::memory::desc init_weights_md({weights_tz}, mpcsn, mfmt_any);

        // ---- Initialize convolution primitive descriptor
        std::shared_ptr<mkldnn::convolution_forward::desc> convFwd_desc;
        if (this->m_bias)
        {
            convFwd_desc.reset(new mkldnn::convolution_forward::desc(
                propagation, mkldnn::algorithm::convolution_direct, init_bottom_md, init_weights_md, init_bias_md,
                init_top_md, convolutionStrides, dnn_dilate, padding_l, padding_r, mkldnn::padding_kind::zero));
        }
        else
        {
            convFwd_desc.reset(new mkldnn::convolution_forward::desc(
                propagation, mkldnn::algorithm::convolution_direct, init_bottom_md, init_weights_md, init_top_md,
                convolutionStrides, dnn_dilate, padding_l, padding_r, mkldnn::padding_kind::zero));
        }
        if (m_relu)
        {
            // add fusion for relu
            attr_t attr = attr_t(mkldnn::round_mode::round_nearest, 1.0, attr_t::scale_t::policy_t::COMMON);
            attr.pops.entry[0].kind = attr_t::post_ops_t::kind_t::RELU;
            attr.pops.entry[0].eltwise.alpha = 0.0;
            attr.pops.entry[0].eltwise.beta = 0.0;
            attr.pops.entry[0].eltwise.scale = 1.0;
            attr.pops.len = 1;
            attr.mkldnn_attr_create();
            convFwd_pd.reset(
                new mkldnn::convolution_forward::primitive_desc(*convFwd_desc, attr.mkldnn_attr, cpu_engine));
        }
        else
        {
            convFwd_pd.reset(new mkldnn::convolution_forward::primitive_desc(*convFwd_desc, cpu_engine));
        }
        assert(convFwd_pd);
        // ---- Create priv memory primitive descriptors stored as class members -------------
        typedef typename mkldnn::memory::primitive_desc MemPD;
        std::shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(new MemPD(convFwd_pd->src_primitive_desc()));
        std::shared_ptr<MemPD> prv_fwd_top_data_memory_pd(new MemPD(convFwd_pd->dst_primitive_desc()));
        std::shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(new MemPD(convFwd_pd->weights_primitive_desc()));

        // ---- Create usr memory primitive descriptors -------------
        mkldnn::memory::format mfmt_nchw = mkldnn::memory::format::nchw;
        mkldnn::memory::format weights_mfmt = (g != 1) ? mkldnn::memory::format::goihw : mkldnn::memory::format::oihw;

        std::shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, mfmt_nchw}, cpu_engine));
        std::shared_ptr<MemPD> usr_bias_data_memory_pd(
            new MemPD({{bias_tz}, mpcsn, mkldnn::memory::format::x}, cpu_engine));
        std::shared_ptr<MemPD> usr_top_data_memory_pd(new MemPD({{top_tz}, mpcsn, mfmt_nchw}, cpu_engine));
        std::shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));

        // ---  init primitive and prv_memory descriptors ----------------------
        fwd_bottom_data.reset(new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd));
        fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
        fwd_top_data.reset(new MKLDNNData<DType>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd));
        fwd_top_data->name = "fwd_top_data      @ " + this->getName();
        fwd_weights_data.reset(new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd));
        fwd_weights_data->name = "fwd_weights_data  @ " + this->getName();
        if (this->m_bias)
        {
            std::shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(new MemPD(convFwd_pd->bias_primitive_desc()));
            fwd_bias_data.reset(new MKLDNNData<DType>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd));
            fwd_bias_data->name = "fwd_bias_data     @ " + this->getName();
        }
    }

public:
    virtual void Forward(const Mat& in, const Mat& kernel, Mat& out, bool inferenceOnly, Mat* pBias = NULL)
    {
        DType* data_ptr = mkl_experimental_direct_get(in);
        DType* out_ptr = mkl_experimental_direct_get(out);
        DType* wmat_ptr = mkl_experimental_direct_get(kernel);
        DType* bias_ptr = NULL;
        if (pBias != NULL)
            bias_ptr = mkl_experimental_direct_get(*pBias);
        if (!b_init_conv)
        {
            this->init_properties((int) in.GetNumCols());
            this->b_init_conv = true;
        }

        bool b_same = true;
        if (convFwd_pd == NULL)
        {
            InitForward(inferenceOnly);
        }

        // ---  init primitive and prv_memory descriptors ---------
        fwd_bottom_data_primitive = fwd_bottom_data->get_converted_prv(data_ptr, false, in, &b_same);
        fwd_weights_data_primitive = fwd_weights_data->get_converted_prv(wmat_ptr, true, kernel, &b_same);
        if (this->m_bias)
        {
            fwd_bias_data_primitive = fwd_bias_data->get_converted_prv(bias_ptr, true, *pBias, &b_same);
            init_gbias.AssignValuesOf(*pBias);
        }

        fwd_top_data_memory = fwd_top_data->create_output_memory(out_ptr, out, false, &b_same);
        if (!b_init_convFwd || !b_same)
        {
            //each mkldnn memory have dedicate _prv_memory
            if (this->m_bias)
            {
                convFwd.reset(new mkldnn::convolution_forward(*convFwd_pd, *fwd_bottom_data_primitive,
                    *fwd_weights_data_primitive, *fwd_bias_data_primitive,
                    *fwd_top_data_memory));
            }
            else
            {
                convFwd.reset(new mkldnn::convolution_forward(*convFwd_pd, *fwd_bottom_data_primitive,
                    *fwd_weights_data_primitive, *fwd_top_data_memory));
            }
            if (!b_init_convFwd)
                b_init_convFwd = true;
        }
        convFwd.submit();
    }
    void InitConvolutionBwd()
    {
        int32_t g = std::max(this->group_, 1);
        int32_t n = this->num_;
        int32_t iw = this->width_;
        int32_t ih = this->height_;
        int32_t ic = this->channels_;

        int32_t ow = this->width_out_;
        int32_t oh = this->height_out_;
        int32_t oc = this->channel_output_;

        int32_t kw = this->kernel_w_;
        int32_t kh = this->kernel_h_;
        mkldnn::memory::dims convolutionStrides{this->stride_h_, this->stride_w_};
        mkldnn::memory::dims padding_l{this->pad_l_h_, this->pad_l_w_};
        mkldnn::memory::dims padding_r{this->pad_r_h_, this->pad_r_w_};
        mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
        mkldnn::memory::format mfmt_any = mkldnn::memory::format::any;

        mkldnn::memory::dims bottom_tz = {n, ic, ih, iw};
        mkldnn::memory::dims bias_tz = {oc};
        mkldnn::memory::dims top_tz = {n, oc, oh, ow};
        mkldnn::memory::dims weights_tz =
            (g != 1) ? mkldnn::memory::dims{g, oc / g, ic / g, kh, kw} : mkldnn::memory::dims{oc, ic, kh, kw};
        mkldnn::memory::desc init_bottom_md({bottom_tz}, mpcsn, mfmt_any);
        mkldnn::memory::desc init_bias_md({bias_tz}, mpcsn, mfmt_any);
        mkldnn::memory::desc init_top_md({top_tz}, mpcsn, mfmt_any);
        mkldnn::memory::desc init_weights_md({weights_tz}, mpcsn, mfmt_any);

        // ---- Initialize convolution primitive descriptor -------------
        std::shared_ptr<mkldnn::convolution_backward_data::desc> convBwdData_desc;
        std::shared_ptr<mkldnn::convolution_backward_weights::desc> convBwdWeights_desc;
        if (this->m_bias)
        {
            convBwdWeights_desc.reset(new mkldnn::convolution_backward_weights::desc(
                mkldnn::algorithm::convolution_direct, init_bottom_md, init_weights_md, init_bias_md, init_top_md,
                convolutionStrides, padding_l, padding_r, mkldnn::padding_kind::zero));
        }
        else
        {
            convBwdWeights_desc.reset(new mkldnn::convolution_backward_weights::desc(
                mkldnn::algorithm::convolution_direct, init_bottom_md, init_weights_md, init_top_md, convolutionStrides,
                padding_l, padding_r, mkldnn::padding_kind::zero));
        }
        mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
        convBwdData_desc.reset(new mkldnn::convolution_backward_data::desc(
            mkldnn::algorithm::convolution_direct, init_bottom_md, init_weights_md, init_top_md, convolutionStrides,
            padding_l, padding_r, mkldnn::padding_kind::zero));
        convBwdData_pd.reset(
            new mkldnn::convolution_backward_data::primitive_desc(*convBwdData_desc, cpu_engine, *convFwd_pd));

        convBwdWeights_pd.reset(
            new mkldnn::convolution_backward_weights::primitive_desc(*convBwdWeights_desc, cpu_engine, *convFwd_pd));

        // ---- Create priv memory primitive descriptors stored as class members -------------
        typedef typename mkldnn::memory::primitive_desc MemPD;

        std::shared_ptr<MemPD> prv_bwdd_bottom_diff_memory_pd(new MemPD(convBwdData_pd->diff_src_primitive_desc()));
        std::shared_ptr<MemPD> prv_bwdd_top_diff_memory_pd(new MemPD(convBwdData_pd->diff_dst_primitive_desc()));
        std::shared_ptr<MemPD> prv_bwdd_weights_data_memory_pd(new MemPD(convBwdData_pd->weights_primitive_desc()));

        std::shared_ptr<MemPD> prv_bwdw_bottom_data_memory_pd(new MemPD(convBwdWeights_pd->src_primitive_desc()));
        std::shared_ptr<MemPD> prv_bwdw_top_diff_memory_pd(new MemPD(convBwdWeights_pd->diff_dst_primitive_desc()));
        std::shared_ptr<MemPD> prv_bwdw_weights_diff_memory_pd(
            new MemPD(convBwdWeights_pd->diff_weights_primitive_desc()));

        // ---- Create usr memory primitive descriptors -------------
        mkldnn::memory::format mfmt_nchw = mkldnn::memory::format::nchw;
        mkldnn::memory::format weights_mfmt = (g != 1) ? mkldnn::memory::format::goihw : mkldnn::memory::format::oihw;

        // ???!!! can we use usr memory primitive descrittors for backward??
        std::shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, mfmt_nchw}, cpu_engine));
        std::shared_ptr<MemPD> usr_bias_data_memory_pd(
            new MemPD({{bias_tz}, mpcsn, mkldnn::memory::format::x}, cpu_engine));
        std::shared_ptr<MemPD> usr_top_data_memory_pd(new MemPD({{top_tz}, mpcsn, mfmt_nchw}, cpu_engine));
        std::shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));

        // ---  init primitive and prv_memory descriptors ----------------------
        bwdd_bottom_diff.reset(new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_bwdd_bottom_diff_memory_pd));
        bwdd_bottom_diff->name = "bwdd_bottom_diff   @ " + this->getName();
        bwdd_bottom_diff_ws.reset(
          new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_bwdd_bottom_diff_memory_pd));
        bwdd_bottom_diff_ws->name = "bwdd_bottom_diff_ws   @ " + this->getName();
        bwdw_bottom_data.reset(new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_bwdw_bottom_data_memory_pd));
        bwdw_bottom_data->name = "bwdw_bottom_data   @ " + this->getName();

        bwdd_top_diff.reset(new MKLDNNData<DType>(usr_top_data_memory_pd, prv_bwdd_top_diff_memory_pd));
        bwdd_top_diff->name = "bwdd_top_diff      @ " + this->getName();
        bwdw_top_diff.reset(new MKLDNNData<DType>(usr_top_data_memory_pd, prv_bwdw_top_diff_memory_pd));
        bwdw_top_diff->name = "bwdw_top_diff      @ " + this->getName();
        bwdd_weights_data.reset(new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_bwdd_weights_data_memory_pd));
        bwdd_weights_data->name = "bwdd_weights_data  @ " + this->getName();
        bwdw_weights_diff.reset(new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_bwdw_weights_diff_memory_pd));
        bwdw_weights_diff->name = "bwdw_weights_diff  @ " + this->getName();
        bwdw_weights_diff_ws.reset(
          new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_bwdw_weights_diff_memory_pd));
        bwdw_weights_diff_ws->name = "bwdw_weights_diff_ws  @ " + this->getName();
        if (this->m_bias)
        {
            std::shared_ptr<MemPD> prv_bwdw_bias_diff_memory_pd(
                new MemPD(convBwdWeights_pd->diff_bias_primitive_desc()));
            mkldnn::memory::desc prv_bwd_bias_desc = convBwdWeights_pd->diff_bias_primitive_desc().desc();
            bwdw_bias_diff.reset(new MKLDNNData<DType>(usr_bias_data_memory_pd, prv_bwdw_bias_diff_memory_pd));
            bwdw_bias_diff->name = "bwdw_bias_diff     @ " + this->getName();
        }
    }
    void InitReLUBwd(const Mat& src)
    {
        int32_t n = this->num_;
        int32_t iw = this->width_;
        int32_t ih = this->height_;
        int32_t ic = this->channels_;
        DType negative_slope = 0;
        void* src_data = const_cast<DType*>(mkl_prv_data<DType>(src));
        bool src_is_prv = (src_data != NULL);
        mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
        mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
        // ---- Initialize memory descriptors -------------
        // std::shared_ptr<mkldnn::memory::desc> bottom_diff_md;
        std::shared_ptr<mkldnn::memory::desc> top_diff_md;
        std::shared_ptr<mkldnn::memory::desc> top_data_md;

        std::shared_ptr<mkldnn::memory::primitive_desc> usr_diff_mpd;
        std::shared_ptr<mkldnn::memory::primitive_desc> prv_diff_mpd;

        if (src_is_prv)
        {
            std::shared_ptr<MKLDNNMemoryDescriptor<DType>> mem_descr = get_mkldnn_prv_descriptor<DType>(src);
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
        mkldnn::eltwise_forward::desc fwd_training_desc(mkldnn::prop_kind::forward_training, mkldnn::eltwise_relu,
                                                        *top_data_md, negative_slope);
        fwd_relu_training_pd.reset(new mkldnn::relu_forward::primitive_desc(fwd_training_desc, cpu_engine));
        mkldnn::eltwise_backward::desc reluBwd_desc(mkldnn::eltwise_relu, *top_diff_md, *top_data_md, negative_slope);
        bwd_relu_pd.reset(new mkldnn::relu_backward::primitive_desc(reluBwd_desc, cpu_engine, *fwd_relu_training_pd));
        bwd_relu_top_diff.reset(new MKLDNNData<DType>(usr_diff_mpd, prv_diff_mpd));
        bwd_relu_top_diff->name = "bwd_top_diff   @ " + this->getName();
        bwd_relu_dst_data.reset(new MKLDNNData<DType>(usr_diff_mpd, prv_diff_mpd));
        bwd_relu_dst_data->name = "bwd_bottom_data   @ " + this->getName();
    }

    virtual void BackwardData(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool accumulateGradient, Mat& workspace)
    {
        DType* srcgrad_ptr = mkl_experimental_direct_get(srcGrad);
        DType* kernel_ptr = mkl_experimental_direct_get(kernel);
        DType* grad_ptr = mkl_experimental_direct_get(grad);
        if (!b_init_conv)
        {
            this->init_properties((int) grad.GetNumCols());
            b_init_conv = true;
        }
        if (convFwd_pd == NULL)
        {
            this->InitForward(true);
        }

        bool b_same = true;
        if (convBwdData_pd == NULL)
        {
            this->InitConvolutionBwd();
        }

        std::shared_ptr<mkldnn::memory> bwdd_top_diff_primitive, bwdd_weights_data_primitive, bwdd_diff_src_primitive;
        std::shared_ptr<mkldnn::memory> bwdd_bottom_diff_memory;
        std::shared_ptr<mkldnn::memory> bwdd_bottom_diff_dst;
        // ---  init primitive and prv_memory descriptors ---------

        bwdd_top_diff_primitive = bwdd_top_diff->get_converted_prv(srcgrad_ptr, true, srcGrad, &b_same);
        bwdd_weights_data_primitive = bwdd_weights_data->get_converted_prv(kernel_ptr, false, kernel, &b_same);
        if (accumulateGradient) {
            workspace.Resize(grad);
            bwdd_bottom_diff_dst  = bwdd_bottom_diff_ws->create_output_memory(workspace.Data(), workspace, false, &b_same);
            bwdd_bottom_diff_memory = bwdd_bottom_diff->get_converted_prv(grad_ptr, true, grad, &b_same);
            grad_ptr = mkl_experimental_direct_get(grad);
        }
        else
        {
            bwdd_bottom_diff_dst = bwdd_bottom_diff_memory = bwdd_bottom_diff->create_output_memory(grad_ptr, grad, false, &b_same);
        }
        if (!b_init_convBwdData || !b_same)
        {
            convBwdData.reset(new mkldnn::convolution_backward_data(
                *convBwdData_pd, *bwdd_top_diff_primitive, *bwdd_weights_data_primitive, *bwdd_bottom_diff_dst));
            if (!b_init_convBwdData)
                b_init_convBwdData = true;
        }
        convBwdData.submit();
        if (accumulateGradient)
        {
            DType * workspace_ptr = mkl_experimental_direct_get(workspace);
            grad.MklMem()->template AddTo<DType>(grad_ptr, *workspace.MklMem(), workspace_ptr);
        }
    }
    void BackwardKernel(const Mat& srcGrad, const Mat& in, const Mat& out, Mat& kernelGrad, bool accumulateGradient, Mat& workspace, Mat* pbiasGrad = NULL)
    {
        DType* srcgrad_ptr = mkl_experimental_direct_get(srcGrad);
        DType* in_ptr = mkl_experimental_direct_get(in);
        DType* out_ptr = mkl_experimental_direct_get(out);
        DType* kernelgrad_ptr = mkl_experimental_direct_get(kernelGrad);
        if (!b_init_conv)
        {
            this->init_properties((int) srcGrad.GetNumCols());
            b_init_conv = true;
        }
        if (convFwd_pd == NULL)
        {
            this->InitForward(true);
        }
        if (convBwdData_pd == NULL)
        {
            this->InitConvolutionBwd();
        }
        bool b_same = true;
        if (m_relu)
        {
            // inplace relu to update srcGrad, do once then backwarddata can also use the converted data
            if (bwd_relu_pd == NULL)
            {
                InitReLUBwd(out);
            }
            std::shared_ptr<mkldnn::memory> dst_memory, diff_dst_memory, diff_src_memory;
            dst_memory = bwd_relu_dst_data->get_converted_prv(out_ptr, false, out, &b_same);
            diff_src_memory = bwd_relu_top_diff->get_converted_prv(srcgrad_ptr, false, srcGrad, &b_same);
            MKLDNNPrimitive<DType> reluBwd;
            reluBwd.reset(new mkldnn::relu_backward(*bwd_relu_pd, *dst_memory, *diff_src_memory, *diff_src_memory));
            reluBwd.submit();
        }
        std::shared_ptr<mkldnn::memory> bwdw_bottom_data_primitive, bwdw_top_diff_primitive;
        std::shared_ptr<mkldnn::memory> bwdw_weights_diff_memory, bwdw_bias_diff_memory;
        std::shared_ptr<mkldnn::memory> bwdw_weights_diff_ws_memory, bwdw_weights_diff_dst;
        bwdw_top_diff_primitive = bwdw_top_diff->get_converted_prv(srcgrad_ptr, true, srcGrad, &b_same);
        bwdw_bottom_data_primitive = bwdw_bottom_data->get_converted_prv(in_ptr, false, in, &b_same);
        if (accumulateGradient) {
            // make sure workspace is user data
            workspace.Resize(kernelGrad);
            bwdw_weights_diff_dst = bwdw_weights_diff_ws_memory = bwdw_weights_diff_ws->create_output_memory(workspace.Data(), workspace, false, &b_same);
            bwdw_weights_diff_memory = bwdw_weights_diff->get_converted_prv(kernelgrad_ptr, true, kernelGrad, &b_same);
        }
        else
        {
            bwdw_weights_diff_dst = bwdw_weights_diff_memory = bwdw_weights_diff->create_output_memory(kernelgrad_ptr, kernelGrad, false, &b_same);
        }
        if (this->m_bias)
        {
            DType* gbias_ptr = mkl_experimental_direct_get(*pbiasGrad);
            if (gbias_ptr == nullptr)
            {
                pbiasGrad->AssignValuesOf(init_gbias);
                gbias_ptr = mkl_experimental_direct_get(*pbiasGrad);
            }
            bwdw_bias_diff_memory = bwdw_bias_diff->create_output_memory(gbias_ptr, *pbiasGrad, false, &b_same);

        }
        if (!b_init_convBwdWeights || !b_same)
        {
            if (this->m_bias)
            {
                convBwdWeights.reset(new mkldnn::convolution_backward_weights(
                    *convBwdWeights_pd, *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive, *bwdw_weights_diff_dst,
                    *bwdw_bias_diff_memory));
            }
            else
            {
                convBwdWeights.reset(new mkldnn::convolution_backward_weights(
                    *convBwdWeights_pd, *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive, *bwdw_weights_diff_dst));
            }
            if (!b_init_convBwdWeights)
                b_init_convBwdWeights = true;
        }
        convBwdWeights.submit();
        if (accumulateGradient)
        {
            DType * workspace_ptr = mkl_experimental_direct_get(workspace);
            kernelGrad.MklMem()->template AddTo<DType>(kernelgrad_ptr, *workspace.MklMem(), workspace_ptr);
        }

    }

private:
    std::shared_ptr<mkldnn::memory> fwd_bottom_data_primitive, fwd_weights_data_primitive, fwd_bias_data_primitive;
    std::shared_ptr<mkldnn::memory> fwd_top_data_memory;
    std::shared_ptr<MKLDNNData<DType>> fwd_bottom_data, fwd_top_data, fwd_weights_data, fwd_bias_data,
        bwdd_weights_data, bwdw_bottom_data;
    std::shared_ptr<MKLDNNData<DType>> bwdd_bottom_diff, bwdd_top_diff, bwdw_top_diff, bwdw_weights_diff,
        bwdw_bias_diff;
    std::shared_ptr<MKLDNNData<DType> > bwdd_bottom_diff_ws, bwdw_weights_diff_ws;
        
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> convFwd_pd;
    MKLDNNPrimitive<DType> convFwd;
    std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc> convBwdData_pd;
    std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc> convBwdWeights_pd;
    std::shared_ptr<mkldnn::relu_forward::primitive_desc> fwd_relu_training_pd;
    std::shared_ptr<mkldnn::relu_backward::primitive_desc> bwd_relu_pd;
    std::shared_ptr<MKLDNNData<DType>> bwd_relu_dst_data, bwd_relu_top_diff;
    MKLDNNPrimitive<DType> convBwdData, convBwdWeights;
    ConvolveGeometryPtr m_geometry;
    ImageLayoutKind m_imageLayout;
    bool b_init_conv;
    bool m_bias;
    int dilate_w;
    int dilate_h;
    Mat init_gbias;
    bool m_relu;
    bool b_init_convBwdData;
    bool b_init_convBwdWeights;
    bool b_init_convFwd;
}; // class MKLDNNConvolutionOp
} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
#endif
#endif // CNTK_OPERATOR_MKL_DNN_MKLDNN_CONVOLUTION_INL_H_
