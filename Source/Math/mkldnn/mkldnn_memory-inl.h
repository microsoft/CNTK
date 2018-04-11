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
 * \file mkl_batch_norm-inl.h
 * \brief
 * \author lingyan.guo@intel.com
 *         zhenlin.luo@intel.com
 *
 *******************************************************************************/

#ifndef CNTK_OPERATOR_MKL_DNN_MKLDNN_MEMORY_INL_H_
#define CNTK_OPERATOR_MKL_DNN_MKLDNN_MEMORY_INL_H_

#include <string>
#include <vector>
#include <iterator>

#include "../Matrix.h"
#ifdef USE_MKLDNN
#include "mkldnn.hpp"
#include "../CPUMatrix.h"
#include "mkldnn_base-inl.h"

namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <typename Dtype>
struct MKLDNNMemoryDescriptorBase : public PrvMemDescr,
                                    public std::enable_shared_from_this<MKLDNNMemoryDescriptorBase<Dtype>>
{
    MKLDNNMemoryDescriptorBase(std::shared_ptr<mkldnn::memory::primitive_desc> usr_memory_pd,
                               std::shared_ptr<mkldnn::memory::primitive_desc> prv_memory_pd);

    ~MKLDNNMemoryDescriptorBase() {}
    std::shared_ptr<MKLDNNMemoryDescriptorBase<Dtype>> get_shared_ptr()
    {
        return this->shared_from_this();
    }
    // ---- PrvMemDescr virtual functions -----
    void allocate()
    {
        if (_prv_memory == nullptr)
        {
            _prv_memory = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(*_prv_memory_pd));
        }
    }
    std::shared_ptr<mkldnn::memory> get_prv_memory(bool usage_check = false)
    {
        UNUSED(usage_check);
        if (_prv_memory == nullptr)
        {
            allocate();
        }
        return _prv_memory;
    }
    inline bool conversion_needed() const
    {
        if (nullptr == _prv_memory_pd)
            return false;
        if (nullptr == _usr_memory_pd)
            return false;
        if (*_usr_memory_pd != *_prv_memory_pd)
            return true;
        else
            return false;
    }

    void set_prv_memory_pd(std::shared_ptr<mkldnn::memory::primitive_desc> memory_pd)
    {
        _prv_memory_pd = memory_pd;
    }

    void set_usr_memory_pd(std::shared_ptr<mkldnn::memory::primitive_desc> memory_pd)
    {
        _usr_memory_pd = memory_pd;
    }

    virtual void* prv_ptr()
    {
        if (_prv_memory != nullptr)
        {
            return _prv_memory->get_data_handle();
        }
        else
            return NULL;
    }
    virtual size_t prv_size()
    {
        return _prv_memory_pd->get_size();
    }
    virtual size_t prv_count()
    {
        return prv_size() / sizeof(Dtype);
    }

    virtual bool layout_compare(std::shared_ptr<PrvMemDescr> other);
    virtual PrvDescrType get_descr_type()
    {
        return PRV_DESCR_MKLDNN;
    }

    std::shared_ptr<mkldnn::memory::primitive_desc> prv_memory_pd() const
    {
        return _prv_memory_pd;
    }
    std::shared_ptr<mkldnn::memory::primitive_desc> usr_memory_pd() const
    {
        return _usr_memory_pd;
    }

    std::string name; // for debugging purposes

    void check_usr_with_prv_descriptors();
    void set_prv_memory(std::shared_ptr<mkldnn::memory> memory)
    {
        _prv_memory = memory;
    }
    virtual bool get_usr_desc(usr_desc_dims_t usr_desc_dims, int& dim_size);

protected:
    std::shared_ptr<mkldnn::memory::primitive_desc> _usr_memory_pd;
    std::shared_ptr<mkldnn::memory::primitive_desc> _prv_memory_pd;
    std::shared_ptr<mkldnn::memory> _prv_memory;
};

template <typename Dtype>
class MKLDNNMemoryDescriptor : public MKLDNNMemoryDescriptorBase<Dtype>
{
public:
    using Mat = Matrix<Dtype>;

    using CPUMat = CPUMatrix<Dtype>;
public:
    MKLDNNMemoryDescriptor(std::shared_ptr<mkldnn::memory::primitive_desc> usr_memory_pd,
                           std::shared_ptr<mkldnn::memory::primitive_desc> prv_memory_pd);

    virtual void convert_from_prv(void* cpu_ptr);
    virtual void convert_to_prv(void* cpu_ptr);
    virtual void convert_from_extprv(std::shared_ptr<mkldnn::memory> extprv_memory);
    virtual void convert_from_other(std::shared_ptr<PrvMemDescr> other);
    virtual bool on_to_cpu();

    virtual void create_reorder_from_prv(std::shared_ptr<mkldnn::memory> usr_memory,
                                         MKLDNNPrimitive<Dtype>& reorder_prv2usr);
    virtual void create_reorder_to_prv(std::shared_ptr<mkldnn::memory> usr_memory,
                                       MKLDNNPrimitive<Dtype>& reorder_usr2prv);

    // The last get_blob_data_ptr() argument is a hack for reusing
    // in backward a conversion done already in the forward direction.

    std::shared_ptr<mkldnn::memory> get_converted_prv(Dtype* cpu_data, bool set_prv_ptr, const Mat& b);
    std::shared_ptr<mkldnn::memory> get_converted_prv2(Dtype* cpu_data,
        bool set_prv_ptr, const CPUMat &b);
    std::shared_ptr<mkldnn::memory> create_output_memory(Dtype* cpu_data, 
        const Mat &b, bool in_place = false);

    bool get_prv_prim_desc(mkldnn::memory::primitive_desc& desc);
};

template <typename Dtype>
class MKLDNNData : public MKLDNNMemoryDescriptor<Dtype>
{
public:
    MKLDNNData(std::shared_ptr<mkldnn::memory::primitive_desc> usr_memory_pd,
               std::shared_ptr<mkldnn::memory::primitive_desc> prv_memory_pd)
        : MKLDNNMemoryDescriptor<Dtype>(usr_memory_pd, prv_memory_pd)
    {
    }
    virtual std::shared_ptr<PrvMemDescr> get_copy();
    virtual void get_sum(std::shared_ptr<PrvMemDescr> from);
};

template <typename Dtype>
std::shared_ptr<MKLDNNData<Dtype>> get_mkldnn_prv_descriptor(std::shared_ptr<MKLMemHolder> blob);

template <typename Dtype>
inline std::shared_ptr<MKLDNNData<Dtype>> get_mkldnn_prv_descriptor(const Matrix<Dtype>& b)
{
    return get_mkldnn_prv_descriptor<Dtype>(b.MklMem());
}
template <typename Dtype>
inline std::shared_ptr<mkldnn::memory> mkldnn_prv_memory(const Matrix<Dtype>& b)
{
    std::shared_ptr<MKLMemHolder> mkl_mem = b.MklMem();
    bool mem_valid = (mkl_mem != nullptr) && mkl_mem->head_at_prv();
    if (mem_valid)
    {
        std::shared_ptr<MKLDNNMemoryDescriptor<Dtype>> mem_desc = get_mkldnn_prv_descriptor<Dtype>(mkl_mem);
        if (mem_desc != nullptr)
            return mem_desc->get_prv_memory(true);
    }
    return nullptr;
}

} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
#endif
#endif // CNTK_OPERATOR_MKL_DNN_MKLDNN_MEMORY_INL_H_
