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
#ifndef CNTK_OPERATOR_MKL_DNN_MKLDNN_SUM_INL_H_
#define CNTK_OPERATOR_MKL_DNN_MKLDNN_SUM_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include "mkl_memory.h"
#include "mkldnn_memory-inl.h"
#include "mkldnn_base-inl.h"
#include "mkl_util-inl.h"
#include <omp.h>
#ifdef USE_MKLDNN

namespace Microsoft { namespace MSR { namespace CNTK {

template<typename Dtype>
class MKLDNNSumOp : public MKLDNNLayer<Dtype> {
    static int s_id_gen;
    int m_id;
public:
    using Mat = Matrix<Dtype>;
    std::string getName() {
        std::string name = "MKLDNNSumOp_";
        name = name + std::to_string(m_id);
        return name;
    }
    explicit MKLDNNSumOp(): fwd_top_data(), fwd_bottom_data()
        , eltwiseFwd_pd()
        , fwd_top_data_memory()
        , fwd_bottom_data_primitives_()
        , num_(0), width_(0), height_(0), channels_(0)
        , num_bottoms_(0), init_(false)
    {
        m_id = s_id_gen++;
    }
    virtual ~MKLDNNSumOp() {}

    void InitEltwiseFwd(const TensorShape& ishape, const std::vector<Mat*>& bottom)
    {
        int dimension = 4;
        SmallVector<size_t> inputSize = ishape.GetDims();
        while (inputSize.size() < dimension - 1) inputSize.push_back(1);
        this->width_ = (int)inputSize[0];
        this->height_ = (int)inputSize[1];
        this->channels_ = (int)inputSize[2];
        this->num_ = (int32_t)bottom[0]->GetNumCols();
        int32_t n = this->num_;
        int32_t iw = this->width_;
        int32_t ih = this->height_;
        int32_t ic = this->channels_;

        // If we just do simple adding, scale is 1.0 for all inputs we have
        std::vector<float> scale(num_bottoms_, 1.0);
        //Eltwise layer is supporting multiplication coefficient and this scale value can be used for that.
        //for (int i = 0; i < num_bottoms_; ++i)
        //{
        //    scale[i] = coeffs_[i];
        //}

        mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
        mkldnn::memory::data_type mpcsn = mkldnn::memory::data_type::f32;
        mkldnn::memory::format mfmt_nchw = mkldnn::memory::format::nchw;

        // ---- Initialize memory descriptors -------------
        std::vector<mkldnn::memory::primitive_desc> bottom_data_mpd;
        fwd_bottom_data.clear();

        for (auto i = 0; i < num_bottoms_; i++)
        {
            fwd_bottom_data.push_back(std::shared_ptr<MKLDNNData<Dtype> >());
            mkldnn::memory::format bottom_data_mfmt = mfmt_nchw;
            std::shared_ptr<mkldnn::memory::primitive_desc> prv_bottom_data_mpd;
            std::shared_ptr<mkldnn::memory::primitive_desc> usr_bottom_data_mpd(
                new mkldnn::memory::primitive_desc({ { n, ic, ih, iw }, mpcsn, bottom_data_mfmt }, cpu_engine));
            void * bottom_i_data = const_cast<Dtype*>(mkl_prv_data<Dtype>(*bottom[i]));
            bool bottom_data_is_prv = bottom_i_data != NULL;
            if (bottom_data_is_prv)
            {
                std::shared_ptr<MKLDNNData<Dtype> > mem_descr
                    = get_mkldnn_prv_descriptor<Dtype>(*bottom[i]);
                bottom_data_mfmt = static_cast<mkldnn::memory::format>(
                    mem_descr->prv_memory_pd()->desc().data.format);
                prv_bottom_data_mpd.reset(new mkldnn::memory::primitive_desc(
                    { { n, ic, ih, iw }, mpcsn, bottom_data_mfmt }, cpu_engine));
            }

            bottom_data_mpd.push_back(mkldnn::memory::primitive_desc(
                { { n, ic, ih, iw }, mpcsn, bottom_data_mfmt }, cpu_engine));

            fwd_bottom_data[i].reset(new MKLDNNData<Dtype>(usr_bottom_data_mpd, prv_bottom_data_mpd));
            fwd_bottom_data[i]->name = "fwd_bottom_data[i]   @ " + this->getName();

        }

        std::shared_ptr<mkldnn::memory::primitive_desc> usr_top_data_mpd(new mkldnn::memory::primitive_desc(
            { { n, ic, ih, iw }, mpcsn, mfmt_nchw }, cpu_engine));


        eltwiseFwd_pd.reset(new mkldnn::sum::primitive_desc({ { n, ic, ih, iw }, mpcsn, mkldnn::memory::format::any }, scale, bottom_data_mpd));
        assert(eltwiseFwd_pd);

        std::shared_ptr<mkldnn::memory::primitive_desc> prv_top_data_mpd(new mkldnn::memory::primitive_desc(eltwiseFwd_pd->dst_primitive_desc()));

        fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_mpd, prv_top_data_mpd));
        fwd_top_data->name = "fwd_top_data   @ " + this->getName();
    }

    void Forward(const TensorShape& ishape, const std::vector<Mat*>& bottom, Mat& top) {
        Dtype *out_ptr = mkl_experimental_direct_get(top); //TODO: support 
        this->num_bottoms_ = (int)bottom.size();
        if (!init_)
        {
            InitEltwiseFwd(ishape, bottom);
            init_ = true;
        }
        fwd_bottom_data_primitives_.clear();
        fwd_bottom_data_primitives_at_.clear();
        for (auto i = 0; i < num_bottoms_; i++)
        {
            Dtype *i_ptr = mkl_experimental_direct_get(*bottom[i]);
            fwd_bottom_data_primitives_.push_back(fwd_bottom_data[i]->get_converted_prv(i_ptr, false, *bottom[i]));
            fwd_bottom_data_primitives_at_.push_back(*fwd_bottom_data_primitives_[i]);
        }
        fwd_top_data_memory = fwd_top_data->create_output_memory(out_ptr, top);
        eltwiseFwd.reset(new mkldnn::sum(*eltwiseFwd_pd, fwd_bottom_data_primitives_at_, *fwd_top_data_memory));
        eltwiseFwd.submit();
    }

    static mkldnn::memory::dims GetDimFromTensorShape(const TensorShape& a, mkldnn::memory::format format)
    {
        //TODO: Sum support > 4 dimension
        size_t dimension = 4;
        size_t size = a.size();
        if (format == mkldnn::memory::format::nchw && size > dimension)
            size = dimension;
        mkldnn::memory::dims in_dim;
        for (int i = 0; i < size; i++)
            in_dim.push_back((int)a[size-1-i]);
        if (size < dimension) {
            for (size_t i = size; i < dimension; i++)
                in_dim.push_back((int)1);
        }
        return in_dim;
    }


    void Backward(Mat& in, Mat& out)
    {
        // For each in_v[i] = out
        std::shared_ptr<MKLMemHolder> in_mem = in.MklMem();
        std::shared_ptr<MKLMemHolder> out_mem = out.MklMem();

        if (in_mem->head_ == HEAD_AT_CPU)
        {
            Dtype * in_ptr = in.Data();
            //Convert out (if prv to cpu)
            Dtype * out_ptr = out.Data();
            size_t in_size = in.GetNumElements();
            memcpy(out_ptr, in_ptr, in_size * sizeof(Dtype));
        }
        else
        {
            out_mem->set_prv_descriptor(in_mem->get_prv_descriptor());
        }
    }

    bool direct_prv_two_sum(MKLDNNData<Dtype>& a, MKLDNNData<Dtype>& b, MKLDNNData<Dtype>& c)
    {
        std::vector<MKLDNNData<Dtype>*> _in;
        _in.push_back(&a);
        _in.push_back(&b);
        direct_prv_sums(_in, &c);
        return true;
    }
    void direct_prv_sums(std::vector<MKLDNNData<Dtype>*> _in, MKLDNNData<Dtype>*c)
    {
        std::vector<float> scale(_in.size(), 1.0);
        std::vector<mkldnn::memory::primitive_desc> bottom_data_mpd;

        for (int i = 0; i < _in.size(); i++)
        {
            std::shared_ptr<mkldnn::memory> _in_mem = _in[i]->get_prv_memory();
            bottom_data_mpd.push_back(_in_mem->get_primitive_desc());
            fwd_bottom_data_primitives_.push_back(_in_mem);
            fwd_bottom_data_primitives_at_.push_back(*fwd_bottom_data_primitives_[i]);
        }
        fwd_top_data_memory = c->get_prv_memory();
        direct_calc(bottom_data_mpd, scale);
    }

    bool direct_usr_two_sum(Dtype* a, Dtype *b,  Dtype *c,
        std::shared_ptr<mkldnn::memory::primitive_desc> _usr_memory_pd)
    {
        if (a == NULL || b == NULL)
            return false;
        std::vector<Dtype*> _in;
        _in.push_back(a);
        _in.push_back(b);
        direct_usr_sums(_usr_memory_pd, _in, c);
        return true;
    }
    void direct_usr_sums(std::shared_ptr<mkldnn::memory::primitive_desc> _usr_memory_pd,
        std::vector<Dtype*> &_in, Dtype* out)
    {
        std::vector<float> scale(_in.size(), 1.0);
        std::vector<mkldnn::memory::primitive_desc> bottom_data_mpd;

        for (int i = 0; i < _in.size(); i++)
        {
            std::shared_ptr<mkldnn::memory> _in_mem;
            _in_mem.reset(new mkldnn::memory(*_usr_memory_pd, _in[i]));
            bottom_data_mpd.push_back(_in_mem->get_primitive_desc());
            fwd_bottom_data_primitives_.push_back(_in_mem);
            fwd_bottom_data_primitives_at_.push_back(*fwd_bottom_data_primitives_[i]);
        }
        fwd_top_data_memory.reset(new mkldnn::memory(*_usr_memory_pd, out));
        direct_calc(bottom_data_mpd, scale);
    }
    void direct_calc(std::vector<mkldnn::memory::primitive_desc>&bottom_data_mpd, std::vector<float> &scale)
    {
        mkldnn::memory::desc output_mpd = fwd_top_data_memory->get_primitive_desc().desc();
        eltwiseFwd_pd.reset(new mkldnn::sum::primitive_desc(output_mpd, scale, bottom_data_mpd));
        eltwiseFwd.reset(new mkldnn::sum(*eltwiseFwd_pd, fwd_bottom_data_primitives_at_, *fwd_top_data_memory));
        eltwiseFwd.submit();
    }
private:
    std::shared_ptr<MKLDNNData<Dtype> > fwd_top_data;
    std::vector<shared_ptr<MKLDNNData<Dtype> > > fwd_bottom_data;
    std::shared_ptr<mkldnn::sum::primitive_desc> eltwiseFwd_pd;
    MKLDNNPrimitive<Dtype> eltwiseFwd;

    std::shared_ptr<mkldnn::memory> fwd_top_data_memory;
    std::vector<std::shared_ptr<mkldnn::memory>> fwd_bottom_data_primitives_;
    std::vector<mkldnn::primitive::at> fwd_bottom_data_primitives_at_;


    std::vector<Dtype> coeffs_;

    int32_t num_, width_, height_, channels_;
    int32_t num_bottoms_;
    bool init_;
};

}}}
#endif //USE_MKLDNN
#endif //CNTK_OPERATOR_MKL_DNN_MKLDNN_SUM_INL_H_