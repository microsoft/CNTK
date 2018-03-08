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
* \file mkl_pooling-inl.h
* \brief
* \author zhenlin.luo@intel.com
*         lingyan.guo@intel.com
*
*******************************************************************************/

#ifndef CNTK_OPERATOR_MKL_DNN_MKLDNN_POOLING_INL_H_
#define CNTK_OPERATOR_MKL_DNN_MKLDNN_POOLING_INL_H_
#include <vector>
#include <string>
#include <utility>
#include "mkl_memory.h"
#include "mkldnn_memory-inl.h"
#include "mkl_conv-common-inl.h"
#include "mkldnn_base-inl.h"
#ifdef USE_MKLDNN

namespace Microsoft { namespace MSR { namespace CNTK {

template<typename Dtype>
class MKLDNNPoolingOp : public MKLDNNLayer<Dtype> {
public:
  using Mat = Matrix<Dtype>;
 public:
  std::string getName() {
     std::string name = "MKLDNNPoolingOp";
     return name;
  }
  explicit MKLDNNPoolingOp(ConvolveGeometryPtr geometry,
    ImageLayoutKind imageLayout,
    PoolKind kind, bool poolIncludePad) 
    : MKLDNNLayer<Dtype>()
    , num_(0), channels_(0), width_(0), height_(0), width_out_(0), height_out_(0)
    , kernel_w_(0), kernel_h_(0), stride_w_(0), stride_h_(0)
    , pad_l_h_(0), pad_r_h_(0), pad_l_w_(0), pad_r_w_(0)
    , poolingFwdInference_pd(NULL), poolingFwdTraining_pd(NULL)
    , poolingBwd_pd(NULL) {
    m_geometry = geometry;
    m_imageLayout = imageLayout;
    m_kind = kind;
    this->init_mkldnn_ = false;
    switch (kind) {
    case PoolKind::Max:
      m_pooling_algorithm = mkldnn::pooling_max;
      break;
    case PoolKind::Average:
      if(poolIncludePad)
        m_pooling_algorithm = mkldnn::pooling_avg_include_padding;
      else
        m_pooling_algorithm = mkldnn::pooling_avg_exclude_padding;
      break;
    default:
      InvalidArgument("Unknown pooling method.");
	  break;
    }
  }
  virtual ~MKLDNNPoolingOp() {}

 private:
  void LayerSetUp(int batchSize) {
    int dim_size = (int)m_geometry->InputShape().GetRank() - 1;
    ImageDimensions inT(m_geometry->InputShape(), m_imageLayout);
    ImageDimensions outT(m_geometry->OutputShape(), m_imageLayout);
    ImageDimensions kernelT(m_geometry->KernelShape(), m_imageLayout);
    ImageDimensions strideT(m_geometry->Stride(), m_imageLayout);
    const TensorShape& padShape = m_geometry->LowerPad();
    int pad_size = (int)padShape.size();

    // ImageDimensions padT(m_geometry->LowerPad(), m_imageLayout);
    SmallVector<int> kernel(dim_size, 1);
    SmallVector<int> stride(dim_size, 1);
    SmallVector<int> lower_pad(dim_size, 0);
    SmallVector<int> upper_pad(dim_size, 0);
    auto kernelShape = m_geometry->KernelShape();
    for (int i = 0; i < dim_size; i++)
    {
      kernel[dim_size - 1 - i] = (int)kernelShape[i];
      stride[dim_size - 1 - i] = (int)m_geometry->GetStride(i);
      lower_pad[dim_size - 1 - i] = m_geometry->GetLowerPad(i);
      upper_pad[dim_size - 1 - i] = m_geometry->GetUpperPad(i);
    }


    channels_ = (int)inT.c();
    height_ = (int)inT.h();
    width_ = (int)inT.w();
    num_ = batchSize;

    kernel_h_ = (int)kernelT.h();
    kernel_w_ = (int)kernelT.w();

    stride_h_ = (int)strideT.h();
    stride_w_ = (int)strideT.w();

    height_out_ = (int)outT.h();
    width_out_ = (int)outT.w();
    const SmallVector<bool>& autopad = m_geometry->AutoPad();
    int autopad_size = (int)autopad.size();
    if (autopad_size> 0 && autopad[0]) {
      this->pad_l_w_ = (int)m_geometry->GetLowerPad(0);
    } else if (pad_size > 0) {
      this->pad_l_w_ = (int)padShape[0];
    }
    if (autopad_size > 1 && autopad[1]) {
      this->pad_l_h_ = (int)m_geometry->GetLowerPad(1);
    } else if (pad_size > 1) {
      this->pad_l_h_ = (int)padShape[1];
    }
    this->pad_r_h_ = (this->height_out_ - 1) * this->stride_h_
      - this->pad_l_h_ - this->height_ + this->kernel_h_;
    this->pad_r_w_ = (this->width_out_ - 1) * this->stride_w_
      - this->pad_l_w_ - this->width_ + this->kernel_w_;
  }

 public:
  void InitPoolingFwd() {
    int32_t n = this->num_;
    int32_t c = this->channels_;
    int32_t ih = this->height_;
    int32_t iw = this->width_;
    int32_t oh = this->height_out_;
    int32_t ow = this->width_out_;

    int32_t kh = this->kernel_h_;
    int32_t kw = this->kernel_w_;

    int32_t sh = this->stride_h_;
    int32_t sw = this->stride_w_;

    int32_t pt = this->pad_l_h_;
    int32_t pb = this->pad_r_h_;
    int32_t pl = this->pad_l_w_;
    int32_t pr = this->pad_r_w_;
     mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
     mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
     mkldnn::memory::dims bottom_tz = { n, c, ih, iw };
     mkldnn::memory::dims top_tz = { n, c, oh, ow };
     mkldnn::memory::format mfmt_nchw = mkldnn::memory::format::nchw;

     // ---- Initialize memory descriptors -------------
     typedef typename mkldnn::memory::primitive_desc MemPD;

     mkldnn::memory::format cmfmt = mfmt_nchw;

     std::shared_ptr<mkldnn::memory::desc> init_fwd_bottom_md(
       new mkldnn::memory::desc({ bottom_tz }, mpcsn, cmfmt));
     std::shared_ptr<mkldnn::memory::desc> init_fwd_top_md(new mkldnn::memory::desc({ top_tz }, mpcsn, cmfmt));
     std::shared_ptr<MemPD> usr_bottom_data_mpd(new MemPD({ { bottom_tz }, mpcsn, mfmt_nchw },
       cpu_engine));
     std::shared_ptr<MemPD> usr_top_data_mpd(
       new MemPD({ { top_tz }, mpcsn, mfmt_nchw }, cpu_engine));

     mkldnn::pooling_forward::desc poolingFwdInference_desc(mkldnn::prop_kind::forward_scoring,
        m_pooling_algorithm, *init_fwd_bottom_md, *init_fwd_top_md
       , { sh, sw }, { kh, kw }, { pt, pl }, { pb, pr }, mkldnn::padding_kind::zero);
     mkldnn::pooling_forward::desc poolingFwdTraining_desc(mkldnn::prop_kind::forward_training
       , m_pooling_algorithm, *init_fwd_bottom_md, *init_fwd_top_md
       , { sh, sw }, { kh, kw }, { pt, pl }, { pb, pr }, mkldnn::padding_kind::zero);
     poolingFwdInference_pd.reset(new mkldnn::pooling_forward::primitive_desc(
       poolingFwdInference_desc, cpu_engine));
     assert(poolingFwdInference_pd);
     poolingFwdTraining_pd.reset(new mkldnn::pooling_forward::primitive_desc(
       poolingFwdTraining_desc, cpu_engine));
     assert(poolingFwdTraining_pd);

     // ---- Initialize remaining memory descriptors -------------
     std::shared_ptr<MemPD> prv_fwd_bottom_data_mpd;
     std::shared_ptr<MemPD> prv_fwd_top_data_mpd;

     fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_mpd, prv_fwd_bottom_data_mpd));
     fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();

     fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_mpd, prv_fwd_top_data_mpd));
     fwd_top_data->name = "fwd_top_data   @ " + this->getName();
     // ---- Initialize pooling primitive descriptor -------------
     if (m_pooling_algorithm == mkldnn::algorithm::pooling_max) {
       indices_pd.reset(
         new mkldnn::memory::primitive_desc(poolingFwdTraining_pd->workspace_primitive_desc()));
       indices_memory.reset(new mkldnn::memory(*indices_pd));
     }
  }
  virtual void Forward(const Mat& in, Mat& out, bool inferenceOnly) {
    if (!init_mkldnn_) {
      LayerSetUp((int)in.GetNumCols());
      init_mkldnn_ = true;
    }
    if (poolingFwdInference_pd == NULL)
      InitPoolingFwd();
    // ---  init primitive and prv_memory descriptors ----------------------
    std::shared_ptr<mkldnn::memory> fwd_input_primitive, fwd_output_memory;
    fwd_input_primitive = fwd_bottom_data->get_converted_prv(in.Data(), false);
    fwd_output_memory = fwd_top_data->create_output_memory(out.Data());
    MKLDNNPrimitive<Dtype> poolingFwd;
    if (!inferenceOnly && m_pooling_algorithm == mkldnn::algorithm::pooling_max) {
      poolingFwd.reset(new mkldnn::pooling_forward(*poolingFwdTraining_pd, *fwd_input_primitive,
        *fwd_output_memory, *indices_memory));
    } else {
      poolingFwd.reset(new mkldnn::pooling_forward(*poolingFwdInference_pd, *fwd_input_primitive,
        *fwd_output_memory));
    }
    poolingFwd.submit();
    if (fwd_top_data->conversion_needed()) {
      fwd_top_data->convert_from_prv(out.Data());
    }
  }
  void InitPoolingBwd() {
    int32_t n = this->num_;
    int32_t c = this->channels_;
    int32_t ih = this->height_;
    int32_t iw = this->width_;
    int32_t oh = this->height_out_;
    int32_t ow = this->width_out_;

    int32_t kh = this->kernel_h_;
    int32_t kw = this->kernel_w_;

    int32_t sh = this->stride_h_;
    int32_t sw = this->stride_w_;

    int32_t pt = this->pad_l_h_;
    int32_t pb = this->pad_r_h_;

    int32_t pr = this->pad_r_w_;
    int32_t pl = this->pad_l_w_;
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
    mkldnn::memory::dims bottom_tz = { n, c, ih, iw };
    mkldnn::memory::dims top_tz = { n, c, oh, ow };
    mkldnn::memory::format mfmt_nchw = mkldnn::memory::format::nchw;

    // ---- Initialize memory descriptors -------------
    typedef typename mkldnn::memory::primitive_desc MemPD;

    mkldnn::memory::format bwd_cmfmt = mfmt_nchw;

    std::shared_ptr<mkldnn::memory::desc> init_bwd_bottom_md(
      new mkldnn::memory::desc({ bottom_tz }, mpcsn, bwd_cmfmt));
    std::shared_ptr<mkldnn::memory::desc> init_bwd_top_md(
      new mkldnn::memory::desc({ top_tz }, mpcsn, bwd_cmfmt));
    std::shared_ptr<MemPD> usr_bottom_data_mpd(
      new MemPD({ { bottom_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    std::shared_ptr<MemPD> usr_top_data_mpd(
      new MemPD({ { top_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    // ---- Initialize pooling primitive descriptor -------------
    mkldnn::pooling_backward::desc poolingBwd_desc(this->m_pooling_algorithm, *init_bwd_bottom_md,
      *init_bwd_top_md
      , { sh, sw }, { kh, kw }, { pt, pl }, { pb, pr }, mkldnn::padding_kind::zero);
    poolingBwd_pd.reset(new mkldnn::pooling_backward::primitive_desc(poolingBwd_desc,
      cpu_engine, *poolingFwdTraining_pd));
    assert(poolingBwd_pd);
    // ---- Initialize remaining memory descriptors -------------
    std::shared_ptr<MemPD> prv_bwd_bottom_diff_mpd, prv_bwd_top_diff_mpd;

    bwd_bottom_diff.reset(new MKLDNNData<Dtype>(usr_bottom_data_mpd, prv_bwd_bottom_diff_mpd));
    bwd_bottom_diff->name = "bwd_bottom_diff   @ " + getName();
    bwd_top_diff.reset(new MKLDNNData<Dtype>(usr_top_data_mpd, prv_bwd_top_diff_mpd));
    bwd_top_diff->name = "bwd_top_diff      @ " + getName();
  }

  virtual void Backward(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad) {
    UNUSED(in);
    UNUSED(out);
    if (!init_mkldnn_) {
      LayerSetUp((int)in.GetNumCols());
      init_mkldnn_ = true;
    }
    if (poolingFwdTraining_pd == NULL)
      InitPoolingFwd();
    if (poolingBwd_pd == NULL)
      InitPoolingBwd();
    std::shared_ptr<mkldnn::memory> diff_dst_memory, diff_src_memory;
    diff_dst_memory = bwd_top_diff->get_converted_prv(srcGrad.Data(), true);
    diff_src_memory = bwd_bottom_diff->create_output_memory(grad.Data());
    MKLDNNPrimitive<Dtype>  poolingBwd;
    if (m_pooling_algorithm == mkldnn::algorithm::pooling_max) {
      poolingBwd.reset(new mkldnn::pooling_backward(*poolingBwd_pd, *diff_dst_memory,
        *indices_memory, *diff_src_memory));
    } else {
      poolingBwd.reset(new mkldnn::pooling_backward(*poolingBwd_pd, *diff_dst_memory,
        *diff_src_memory));
    }
    poolingBwd.submit();
  }

 private:

  int32_t num_, channels_, width_, height_, width_out_, height_out_;
  int32_t kernel_w_, kernel_h_, stride_w_, stride_h_;
  int32_t  pad_l_h_, pad_r_h_, pad_l_w_, pad_r_w_;

  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> poolingFwdInference_pd;
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> poolingFwdTraining_pd;
  std::shared_ptr<mkldnn::pooling_backward::primitive_desc> poolingBwd_pd;

  std::shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data,
    bwd_top_diff, bwd_bottom_diff;
  std::shared_ptr<mkldnn::memory::primitive_desc> indices_pd;
  std::shared_ptr<mkldnn::memory> indices_memory;
  bool init_mkldnn_;
  mkldnn::algorithm m_pooling_algorithm;
  PoolKind m_kind;
  ConvolveGeometryPtr m_geometry;
  ImageLayoutKind m_imageLayout;
};  // class MKLDNNPoolingOp
}}}
#endif
#endif  // CNTK_OPERATOR_MKL_DNN_MKLDNN_POOLING_INL_H_
