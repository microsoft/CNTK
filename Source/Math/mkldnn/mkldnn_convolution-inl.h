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

#ifdef USE_MKLDNN

namespace Microsoft { namespace MSR { namespace CNTK {

template<typename DType>
class MKLDNNConvolutionOp : public MKLDNNLayer<DType>,
  public MKLConvCommon<DType> {
 public:
   using Mat = Matrix<DType>;
  std::string getName() {
    std::string name = "MKLDNNConvolutionOp";
    return name;
  }
  explicit MKLDNNConvolutionOp(ConvolveGeometryPtr geometry,
    ImageLayoutKind imageLayout)
    : MKLDNNLayer<DType>(), dilate_w(0), dilate_h(0)
    , fwd_bottom_data(NULL), fwd_top_data(NULL), fwd_weights_data(NULL), fwd_bias_data(NULL)
    , convFwd_pd(NULL)
    , convBwdData_pd(NULL), convBwdWeights_pd(NULL) {
    b_init_conv = false;
    m_geometry = geometry;
    m_imageLayout = imageLayout;
    no_bias = true;
  }

  virtual ~MKLDNNConvolutionOp() {}

  void init_properties(int batchSize) {
    // Check ComputeOutputShape
    ImageDimensions inT(m_geometry->InputShape(), m_imageLayout);
    ImageDimensions outT(m_geometry->OutputShape(), m_imageLayout);
    ImageDimensions kernelT(m_geometry->KernelShape(), m_imageLayout);
    ImageDimensions strideT(m_geometry->Stride(), m_imageLayout);    

    this->stride_w_ = (int)strideT.w();
    this->stride_h_ = (int)strideT.h();
    this->width_ = (int)inT.w();
    this->height_ = (int)inT.h();

    this->kernel_w_ = (int)kernelT.w();
    this->kernel_h_ = (int)kernelT.h();
    this->channels_ = (int)inT.c();
    this->num_ = batchSize;
    this->group_ = 1; //TODO: CNTK support group?
    this->width_out_ = (int)outT.w();
    this->height_out_ = (int)outT.h();
    this->channel_output_ = (int)outT.c();

    if (m_geometry->GetDilation(0) > 1)
      this->dilate_w = (int)m_geometry->GetDilation(0)-1;
    if (m_geometry->GetDilation(1) > 1)
      this->dilate_h = (int)m_geometry->GetDilation(1)-1;

    const SmallVector<bool>& autopad = m_geometry->AutoPad();
    int autopad_size = (int)autopad.size();
    const TensorShape& padShape = m_geometry->LowerPad();
    int pad_size = (int)padShape.size();
    // For CHW
    if (autopad_size > 0 && autopad[0]) {
      this->pad_l_w_ = m_geometry->GetLowerPad(0);
    } else if (pad_size > 0) {
      this->pad_l_w_ = (int)padShape[0];
    }
    if (autopad_size > 1 && autopad[1]) {
      this->pad_l_h_ = m_geometry->GetLowerPad(1);
    } else if (pad_size > 1) {
      this->pad_l_h_ = (int)padShape[1];
    }
    this->pad_r_h_ = (this->height_out_ - 1) * this->stride_h_
      - this->pad_l_h_ - this->height_ + ((this->kernel_h_ - 1) * (this->dilate_h + 1) + 1);
    this->pad_r_w_ = (this->width_out_ - 1) * this->stride_w_
      - this->pad_l_w_ - this->width_ + ((this->kernel_w_ - 1) * (this->dilate_w + 1) + 1);
  }
 private:
  void InitForward(bool inferenceOnly) {
      auto propagation =
        (inferenceOnly) ? mkldnn::prop_kind::forward_scoring : mkldnn::prop_kind::forward_training;

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
      mkldnn::memory::dims convolutionStrides{ static_cast<int>(this->stride_h_),
        static_cast<int>(this->stride_w_) };
      mkldnn::memory::dims padding_l{ this->pad_l_h_, this->pad_l_w_ };
      mkldnn::memory::dims padding_r{ this->pad_r_h_, this->pad_r_w_ };
      mkldnn::memory::dims dnn_dilate{ this->dilate_h, this->dilate_w };
      mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
      mkldnn::memory::format mfmt_any = mkldnn::memory::format::any;
      mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();

      mkldnn::memory::dims bottom_tz = { n, ic, ih, iw };
      mkldnn::memory::dims bias_tz = { oc };
      mkldnn::memory::dims top_tz = { n, oc, oh, ow };
      mkldnn::memory::dims weights_tz =
        (g != 1) ? mkldnn::memory::dims{ g, oc / g, ic / g, kh, kw } : mkldnn::memory::dims{ oc, ic, kh, kw };

      mkldnn::memory::desc init_bottom_md({ bottom_tz }, mpcsn, mfmt_any);
      mkldnn::memory::desc init_bias_md({ bias_tz }, mpcsn, mfmt_any);
      mkldnn::memory::desc init_top_md({ top_tz }, mpcsn, mfmt_any);
      mkldnn::memory::desc init_weights_md({ weights_tz }, mpcsn, mfmt_any);

      // ---- Initialize convolution primitive descriptor
      std::shared_ptr<mkldnn::convolution_forward::desc> convFwd_desc;
      if (!this->no_bias) {
        convFwd_desc.reset(
          new mkldnn::convolution_forward::desc(propagation, mkldnn::algorithm::convolution_direct
          , init_bottom_md, init_weights_md, init_bias_md, init_top_md
          , convolutionStrides, dnn_dilate, padding_l, padding_r, mkldnn::padding_kind::zero));
      } else {
        convFwd_desc.reset(
          new mkldnn::convolution_forward::desc(propagation, mkldnn::algorithm::convolution_direct
          , init_bottom_md, init_weights_md, init_top_md
          , convolutionStrides, dnn_dilate, padding_l, padding_r, mkldnn::padding_kind::zero));
      }
      convFwd_pd.reset(new mkldnn::convolution_forward::primitive_desc(*convFwd_desc, cpu_engine));
      assert(convFwd_pd);
      // ---- Create priv memory primitive descriptors stored as class members -------------
      typedef typename mkldnn::memory::primitive_desc MemPD;
      std::shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(
        new MemPD(convFwd_pd->src_primitive_desc()));
      std::shared_ptr<MemPD> prv_fwd_top_data_memory_pd(
        new MemPD(convFwd_pd->dst_primitive_desc()));
      std::shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(
        new MemPD(convFwd_pd->weights_primitive_desc()));

      // ---- Create usr memory primitive descriptors -------------
      mkldnn::memory::format mfmt_nchw = mkldnn::memory::format::nchw;
      mkldnn::memory::format weights_mfmt = (g != 1) ? mkldnn::memory::format::goihw : mkldnn::memory::format::oihw;

      std::shared_ptr<MemPD> usr_bottom_data_memory_pd(
        new MemPD({ { bottom_tz }, mpcsn, mfmt_nchw }, cpu_engine));
      std::shared_ptr<MemPD> usr_bias_data_memory_pd(
        new MemPD({ { bias_tz }, mpcsn, mkldnn::memory::format::x }, cpu_engine));
      std::shared_ptr<MemPD> usr_top_data_memory_pd(
        new MemPD({ { top_tz }, mpcsn, mfmt_nchw }, cpu_engine));
      std::shared_ptr<MemPD> usr_weights_data_memory_pd(
        new MemPD({ { weights_tz }, mpcsn, weights_mfmt }, cpu_engine));


      // ---  init primitive and prv_memory descriptors ----------------------
      fwd_bottom_data.reset(
        new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd));
      fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
      fwd_top_data.reset(
        new MKLDNNData<DType>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd));
      fwd_top_data->name = "fwd_top_data      @ " + this->getName();
      fwd_weights_data.reset(
        new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd));
      fwd_weights_data->name = "fwd_weights_data  @ " + this->getName();
      if (!this->no_bias) {
        std::shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(
          new MemPD(convFwd_pd->bias_primitive_desc()));
        fwd_bias_data.reset(
          new MKLDNNData<DType>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd));
        fwd_bias_data->name = "fwd_bias_data     @ " + this->getName();
      }
  }
 public:
  virtual void Forward(const Mat& in, const Mat& kernel, Mat& out, bool inferenceOnly) {
      if (!b_init_conv) {        
        this->init_properties((int)in.GetNumCols());
        this->b_init_conv = true;
      }
      if (convFwd_pd == NULL) {
        InitForward(inferenceOnly);
      }
      std::shared_ptr<mkldnn::memory> fwd_bottom_data_primitive,
        fwd_weights_data_primitive, fwd_bias_data_primitive;
      std::shared_ptr<mkldnn::memory> fwd_top_data_memory;
      // ---  init primitive and prv_memory descriptors ---------
      fwd_bottom_data_primitive =
        fwd_bottom_data->get_converted_prv(in.Data(), false);
      fwd_weights_data_primitive = fwd_weights_data->get_converted_prv(kernel.Data(), true);

      fwd_top_data_memory = fwd_top_data->create_output_memory(out.Data());
      if (!this->no_bias) {
        convFwd.reset(new mkldnn::convolution_forward(*convFwd_pd
          , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
          , *fwd_bias_data_primitive, *fwd_top_data_memory));
      } else {
        convFwd.reset(new mkldnn::convolution_forward(*convFwd_pd
          , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
          , *fwd_top_data_memory));
      }
      convFwd.submit();
      if (fwd_top_data->conversion_needed()) {
        fwd_top_data->convert_from_prv(out.Data());
      }
  }
  void InitConvolutionBwd() {
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
    mkldnn::memory::dims convolutionStrides{ this->stride_h_, this->stride_w_ };
    mkldnn::memory::dims padding_l{ this->pad_l_h_, this->pad_l_w_ };
    mkldnn::memory::dims padding_r{ this->pad_r_h_, this->pad_r_w_ };
    mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
    mkldnn::memory::format mfmt_any = mkldnn::memory::format::any;

    mkldnn::memory::dims bottom_tz = { n, ic, ih, iw };
    mkldnn::memory::dims bias_tz = { oc };
    mkldnn::memory::dims top_tz = { n, oc, oh, ow };
    mkldnn::memory::dims weights_tz =
      (g != 1) ? mkldnn::memory::dims{ g, oc / g, ic / g, kh, kw } : mkldnn::memory::dims{ oc, ic, kh, kw };
    mkldnn::memory::desc init_bottom_md({ bottom_tz }, mpcsn, mfmt_any);
    mkldnn::memory::desc init_bias_md({ bias_tz }, mpcsn, mfmt_any);
    mkldnn::memory::desc init_top_md({ top_tz }, mpcsn, mfmt_any);
    mkldnn::memory::desc init_weights_md({ weights_tz }, mpcsn, mfmt_any);

    // ---- Initialize convolution primitive descriptor -------------
    std::shared_ptr<mkldnn::convolution_backward_data::desc> convBwdData_desc;
    std::shared_ptr<mkldnn::convolution_backward_weights::desc> convBwdWeights_desc;
    if (!this->no_bias) {
      convBwdWeights_desc.reset(
        new mkldnn::convolution_backward_weights::desc(mkldnn::algorithm::convolution_direct
        , init_bottom_md, init_weights_md, init_bias_md, init_top_md
        , convolutionStrides, padding_l, padding_r, mkldnn::padding_kind::zero));
    } else {
      convBwdWeights_desc.reset(
        new mkldnn::convolution_backward_weights::desc(mkldnn::algorithm::convolution_direct
        , init_bottom_md, init_weights_md, init_top_md
        , convolutionStrides, padding_l, padding_r, mkldnn::padding_kind::zero));
    }
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    convBwdData_desc.reset(
      new mkldnn::convolution_backward_data::desc(mkldnn::algorithm::convolution_direct
      , init_bottom_md, init_weights_md, init_top_md
      , convolutionStrides, padding_l, padding_r, mkldnn::padding_kind::zero));
    convBwdData_pd.reset(
      new mkldnn::convolution_backward_data::primitive_desc(*convBwdData_desc,
      cpu_engine, *convFwd_pd));

    convBwdWeights_pd.reset(
      new mkldnn::convolution_backward_weights::primitive_desc(*convBwdWeights_desc,
      cpu_engine, *convFwd_pd));


    // ---- Create priv memory primitive descriptors stored as class members -------------
    typedef typename mkldnn::memory::primitive_desc MemPD;

    std::shared_ptr<MemPD> prv_bwdd_bottom_diff_memory_pd(
      new MemPD(convBwdData_pd->diff_src_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdd_top_diff_memory_pd(
      new MemPD(convBwdData_pd->diff_dst_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdd_weights_data_memory_pd(
      new MemPD(convBwdData_pd->weights_primitive_desc()));

    std::shared_ptr<MemPD> prv_bwdw_bottom_data_memory_pd(
      new MemPD(convBwdWeights_pd->src_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdw_top_diff_memory_pd(
      new MemPD(convBwdWeights_pd->diff_dst_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdw_weights_diff_memory_pd(
      new MemPD(convBwdWeights_pd->diff_weights_primitive_desc()));

    // ---- Create usr memory primitive descriptors -------------
    mkldnn::memory::format mfmt_nchw = mkldnn::memory::format::nchw;
    mkldnn::memory::format weights_mfmt = (g != 1) ? mkldnn::memory::format::goihw : mkldnn::memory::format::oihw;

    // ???!!! can we use usr memory primitive descrittors for backward??
    std::shared_ptr<MemPD> usr_bottom_data_memory_pd(
      new MemPD({ { bottom_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    std::shared_ptr<MemPD> usr_bias_data_memory_pd(
      new MemPD({ { bias_tz }, mpcsn, mkldnn::memory::format::x }, cpu_engine));
    std::shared_ptr<MemPD> usr_top_data_memory_pd(
      new MemPD({ { top_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    std::shared_ptr<MemPD> usr_weights_data_memory_pd(
      new MemPD({ { weights_tz }, mpcsn, weights_mfmt }, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    bwdd_bottom_diff.reset(
      new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_bwdd_bottom_diff_memory_pd));
    bwdd_bottom_diff->name = "bwdd_bottom_diff   @ " + this->getName();
    bwdw_bottom_data.reset(
      new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_bwdw_bottom_data_memory_pd));
    bwdw_bottom_data->name = "bwdw_bottom_data   @ " + this->getName();

    bwdd_top_diff.reset(
      new MKLDNNData<DType>(usr_top_data_memory_pd, prv_bwdd_top_diff_memory_pd));
    bwdd_top_diff->name = "bwdd_top_diff      @ " + this->getName();
    bwdw_top_diff.reset(
      new MKLDNNData<DType>(usr_top_data_memory_pd, prv_bwdw_top_diff_memory_pd));
    bwdw_top_diff->name = "bwdw_top_diff      @ " + this->getName();
    bwdd_weights_data.reset(
      new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_bwdd_weights_data_memory_pd));
    bwdd_weights_data->name = "bwdd_weights_data  @ " + this->getName();
    bwdw_weights_diff.reset(
      new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_bwdw_weights_diff_memory_pd));
    bwdw_weights_diff->name = "bwdw_weights_diff  @ " + this->getName();
    if (!this->no_bias) {
      std::shared_ptr<MemPD> prv_bwdw_bias_diff_memory_pd(
        new MemPD(convBwdWeights_pd->diff_bias_primitive_desc()));
      bwdw_bias_diff.reset(
        new MKLDNNData<DType>(usr_bias_data_memory_pd, prv_bwdw_bias_diff_memory_pd));
      bwdw_bias_diff->name = "bwdw_bias_diff     @ " + this->getName();
    }

  }
  virtual void BackwardData(const Mat& srcGrad, const Mat& kernel, Mat& grad) {
    if (!b_init_conv) {
      this->init_properties((int)grad.GetNumCols());
      b_init_conv = true;
    }
    if (convFwd_pd == NULL) {
      this->InitForward(true);
    }
    if (convBwdData_pd == NULL) {
      this->InitConvolutionBwd();
    }

    std::shared_ptr<mkldnn::memory> bwdd_top_diff_primitive, bwdd_weights_data_primitive,
      bwdd_diff_src_primitive;
    std::shared_ptr<mkldnn::memory> bwdd_bottom_diff_memory;

    // ---  init primitive and prv_memory descriptors ---------
    
    bwdd_top_diff_primitive = bwdd_top_diff->get_converted_prv(srcGrad.Data(), true);
    bwdd_weights_data_primitive = bwdd_weights_data->get_converted_prv(kernel.Data(), false);

    bwdd_bottom_diff_memory = bwdd_bottom_diff->create_output_memory(grad.Data());

    convBwdData.reset(new mkldnn::convolution_backward_data(*convBwdData_pd
      , *bwdd_top_diff_primitive, *bwdd_weights_data_primitive
      , *bwdd_bottom_diff_memory));

    convBwdData.submit();
    if (bwdd_bottom_diff->conversion_needed()) {
      bwdd_bottom_diff->convert_from_prv(grad.Data());
    }
  }
  void BackwardKernel(const Mat& srcGrad, const Mat& in, Mat& kernelGrad) {
    if (!b_init_conv) {
      this->init_properties((int)srcGrad.GetNumCols());
      b_init_conv = true;
    }
    if (convFwd_pd == NULL) {
      this->InitForward(true);
    }
    if (convBwdData_pd == NULL) {
      this->InitConvolutionBwd();
    }
    std::shared_ptr<mkldnn::memory> bwdw_bottom_data_primitive, bwdw_top_diff_primitive;
    std::shared_ptr<mkldnn::memory> bwdw_weights_diff_memory, bwdw_bias_diff_memory;

    bwdw_top_diff_primitive = bwdw_top_diff->get_converted_prv(srcGrad.Data(), true);
    bwdw_bottom_data_primitive = bwdw_bottom_data->get_converted_prv(in.Data(), false);
    bwdw_weights_diff_memory = bwdw_weights_diff->create_output_memory(kernelGrad.Data());
    if (!this->no_bias) {
    }
    else {
      convBwdWeights.reset(new mkldnn::convolution_backward_weights(*convBwdWeights_pd
        , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
        , *bwdw_weights_diff_memory));
    }
    convBwdWeights.submit();
    if (bwdw_weights_diff->conversion_needed()) {
      bwdw_weights_diff->convert_from_prv(kernelGrad.Data());
    }
  }
 private:
  std::shared_ptr<MKLDNNData<DType> > fwd_bottom_data, fwd_top_data,
    fwd_weights_data, fwd_bias_data,
    bwdd_weights_data, bwdw_bottom_data;
  std::shared_ptr<MKLDNNData<DType> > bwdd_bottom_diff, bwdd_top_diff,
    bwdw_top_diff, bwdw_weights_diff, bwdw_bias_diff;
  std::shared_ptr<mkldnn::convolution_forward::primitive_desc> convFwd_pd;
  MKLDNNPrimitive<DType> convFwd;
  std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc> convBwdData_pd;
  std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc> convBwdWeights_pd;
  MKLDNNPrimitive<DType> convBwdData, convBwdWeights;
  ConvolveGeometryPtr m_geometry;
  ImageLayoutKind m_imageLayout;
  bool b_init_conv;
  bool no_bias;
  int dilate_w;
  int dilate_h;
};  // class MKLDNNConvolutionOp
}}}
#endif
#endif  // CNTK_OPERATOR_MKL_DNN_MKLDNN_CONVOLUTION_INL_H_
