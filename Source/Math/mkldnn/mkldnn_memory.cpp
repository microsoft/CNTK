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
#include "stdafx.h"
#include <assert.h>
#include <iostream>
#pragma warning(disable : 4996)
#include "../Matrix.h"

#ifdef USE_MKLDNN
#include "mkl_cblas.h"
#include "mkl_memory.h"
#include "mkldnn_sum-inl.h"
#include "mkldnn_memory-inl.h"


namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <typename Dtype>
MKLDNNMemoryDescriptorBase<Dtype>::MKLDNNMemoryDescriptorBase(
    std::shared_ptr<mkldnn::memory::primitive_desc> usr_memory_pd,
    std::shared_ptr<mkldnn::memory::primitive_desc> prv_memory_pd)
    : name("MKLDNNMemoryDescriptorBase"), _prv_memory(NULL), _usr_memory_pd(NULL), _prv_memory_pd(NULL)
    , _cpu_data(NULL), _usr_memory(NULL)
{
    set_usr_memory_pd(usr_memory_pd);
    set_prv_memory_pd(prv_memory_pd);
}

template <typename Dtype>
void MKLDNNMemoryDescriptorBase<Dtype>::check_usr_with_prv_descriptors()
{
    assert(_usr_memory_pd);
    assert(_prv_memory_pd);
    int32_t ndims = _usr_memory_pd->desc().data.ndims;
    assert(ndims == _prv_memory_pd->desc().data.ndims);
    for (int32_t dim = 0; dim < ndims; ++dim)
    {
        assert(_usr_memory_pd->desc().data.dims[dim] == _prv_memory_pd->desc().data.dims[dim]);
    }
}

template <typename Dtype>
bool MKLDNNMemoryDescriptorBase<Dtype>::get_usr_desc(usr_desc_dims_t usr_desc_dims, int& ndims)
{
    if (_usr_memory_pd == nullptr)
        return false;
    ndims = _usr_memory_pd->desc().data.ndims;
    for (int32_t dim = 0; dim < ndims; ++dim)
    {
        usr_desc_dims[dim] = _usr_memory_pd->desc().data.dims[dim];
    }
    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Implementation of MKLDNNMemoryDescriptor
//
////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
MKLDNNMemoryDescriptor<Dtype>::MKLDNNMemoryDescriptor(std::shared_ptr<mkldnn::memory::primitive_desc> usr_memory_pd,
                                                      std::shared_ptr<mkldnn::memory::primitive_desc> prv_memory_pd)
    : MKLDNNMemoryDescriptorBase<Dtype>(usr_memory_pd, prv_memory_pd)
{
}

template <typename Dtype>
void MKLDNNMemoryDescriptor<Dtype>::create_reorder_to_prv(std::shared_ptr<mkldnn::memory> usr_memory,
                                                          MKLDNNPrimitive<Dtype>& reorder_usr2prv)
{
    assert(this->_usr_memory_pd);
    assert(this->_prv_memory_pd);
    if (reorder_usr2prv.aprimitive == NULL)
        reorder_usr2prv.reset(new mkldnn::reorder(*usr_memory, *this->get_prv_memory()));
}

template <typename Dtype>
void MKLDNNMemoryDescriptor<Dtype>::convert_to_prv(void* cpu_ptr)
{
    assert(cpu_ptr);
    MKLDNNPrimitive<Dtype> reorder_usr2prv;
    std::shared_ptr<mkldnn::memory> usr_memory = NULL;
    if (usr_memory == NULL)
        usr_memory.reset(new mkldnn::memory(*this->_usr_memory_pd, cpu_ptr));
    create_reorder_to_prv(usr_memory, reorder_usr2prv);
    // MKL_DLOG(INFO) << "convert usr => priv @" << this->name;
    reorder_usr2prv.submit();
}
template <typename Dtype>
void MKLDNNMemoryDescriptor<Dtype>::convert_from_other(std::shared_ptr<PrvMemDescr> other)
{
    assert(NULL); // Not implementation
}
template <typename Dtype>
void MKLDNNMemoryDescriptor<Dtype>::create_reorder_from_prv(std::shared_ptr<mkldnn::memory> usr_memory,
                                                            MKLDNNPrimitive<Dtype>& reorder_prv2usr)
{
    assert(this->_usr_memory_pd);
    assert(this->_prv_memory_pd);
    if (reorder_prv2usr.aprimitive == NULL)
    {
        reorder_prv2usr.reset(new mkldnn::reorder(*this->_prv_memory, *usr_memory));
    }
}

template <typename Dtype>
void MKLDNNMemoryDescriptor<Dtype>::convert_from_prv(void* cpu_ptr)
{
    assert(cpu_ptr);
    MKLDNNPrimitive<Dtype> reorder_prv2usr;
    std::shared_ptr<mkldnn::memory> usr_memory = NULL;
    if (usr_memory == NULL)
        usr_memory.reset(new mkldnn::memory(*this->_usr_memory_pd, cpu_ptr));
    create_reorder_from_prv(usr_memory, reorder_prv2usr);
    // MKL_DLOG(INFO) << "convert priv => usr @" << this->name;
    reorder_prv2usr.submit();
}

template <typename Dtype>
void MKLDNNMemoryDescriptor<Dtype>::convert_from_extprv(std::shared_ptr<mkldnn::memory> extprv_memory)
{
    MKLDNNPrimitive<Dtype> reorder_extprv2prv;
    reorder_extprv2prv.reset(new mkldnn::reorder(*extprv_memory, *this->get_prv_memory()));
    // MKL_DLOG(INFO) << "convert extprv => priv @" << this->name;
    reorder_extprv2prv.submit();
    ;
}

template <typename Dtype>
bool MKLDNNMemoryDescriptor<Dtype>::on_to_cpu()
{
    if (StreamHolder::Instance().current_stream() != NULL && StreamHolder::Instance().current_stream()->ready())
    {
        StreamHolder::Instance().current_stream()->wait();
    }
    return true;
}

template <typename Dtype>
bool MKLDNNMemoryDescriptorBase<Dtype>::layout_compare(std::shared_ptr<PrvMemDescr> other)
{
    assert(other->get_descr_type() == PrvMemDescr::PRV_DESCR_MKLDNN);
    std::shared_ptr<MKLDNNMemoryDescriptorBase<Dtype>> other_descr =
        std::static_pointer_cast<MKLDNNMemoryDescriptorBase<Dtype>>(other);
    return (*other_descr->prv_memory_pd() == *this->prv_memory_pd());
}

template <typename Dtype>
std::shared_ptr<mkldnn::memory> MKLDNNMemoryDescriptor<Dtype>::get_converted_prv2(Dtype* cpu_data,
    bool set_prv_ptr, const CPUMat &b, bool * b_same)
{
    std::shared_ptr<MKLMemHolder> blob = b.MklMem();
    if (this->conversion_needed())
    {
        const Dtype* prv_ptr = reinterpret_cast<Dtype*>(blob->prv_data());
        if (prv_ptr == NULL)
        {
            this->convert_to_prv(const_cast<Dtype*>(cpu_data));
            if (set_prv_ptr)
            {
                blob->set_prv_descriptor(this->get_shared_ptr());
            }
            return this->get_prv_memory(true);
        }
        else
        {
            std::shared_ptr<MKLDNNData<Dtype>> blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor<Dtype>(blob);
            mkldnn::memory::desc blob_prv_mem_desc = blob_prv_mkldnn_mem_descr->prv_memory_pd()->desc();
            mkldnn::memory::desc this_prv_mem_desc = this->prv_memory_pd()->desc();
            if (*blob_prv_mkldnn_mem_descr->prv_memory_pd() != *this->prv_memory_pd())
            {
                // prv in blob and in this descrptor may have different layouts
                this->convert_from_extprv(blob_prv_mkldnn_mem_descr->get_prv_memory(true));
                if (set_prv_ptr)
                {
                    blob->set_prv_descriptor(this->get_shared_ptr());
                }
                return this->get_prv_memory(true);
            }
            else if (blob_prv_mkldnn_mem_descr.get() != this)
            {
            }
            return blob_prv_mkldnn_mem_descr->get_prv_memory(true);
        }
    }
    else
    {
        const Dtype* prv_ptr = reinterpret_cast<Dtype*>(blob->prv_data());
        if (prv_ptr != NULL)
        {
            std::shared_ptr<MKLDNNData<Dtype>> blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor<Dtype>(blob);
            blob_prv_mkldnn_mem_descr->convert_from_prv(cpu_data);
        }
        if(NULL == this->_cpu_data || cpu_data != this->_cpu_data)
        {
            if(b_same != NULL)
                *b_same = *b_same && false;
            this->_cpu_data = cpu_data;
            this->_usr_memory.reset(new mkldnn::memory(*this->usr_memory_pd(), const_cast<Dtype*>(cpu_data)));
        }
        return this->_usr_memory;
    }
}

template <typename Dtype>
std::shared_ptr<mkldnn::memory> MKLDNNMemoryDescriptor<Dtype>::get_converted_prv(Dtype* cpu_data,
                                            bool set_prv_ptr, const Mat &b, bool * b_same) {
    return get_converted_prv2(cpu_data, set_prv_ptr, *b.getCpuMatrix(), b_same);
}
template <typename Dtype>
std::shared_ptr<mkldnn::memory> MKLDNNMemoryDescriptor<Dtype>::create_output_memory(
    Dtype* cpu_data, const Mat &b, bool in_place, bool * b_same) {
    std::shared_ptr<PrvMemDescr> thisData = this->get_shared_ptr();
    std::shared_ptr<mkldnn::memory> omem;
    if (this->conversion_needed())
    {
        if (in_place)
        {
            std::shared_ptr<MKLDNNData<Dtype>> blob_omem = get_mkldnn_prv_descriptor<Dtype>(b.MklMem());
            omem = blob_omem->get_prv_memory();
        }
        else
        {
            omem = this->get_prv_memory();
            b.MklMem()->set_prv_descriptor(thisData);
        }
        return omem;
    }
    else
    {
        b.MklMem()->check_and_prv_to_cpu(cpu_data);
        if(NULL == this->_cpu_data || cpu_data != this->_cpu_data)
        {
            if(b_same != NULL)
                *b_same = *b_same && false;
            this->_cpu_data = cpu_data;
            this->_usr_memory.reset(new mkldnn::memory(*this->usr_memory_pd(), cpu_data));
    }
        return this->_usr_memory;
    }
}

template <typename Dtype>
bool MKLDNNMemoryDescriptor<Dtype>::get_prv_prim_desc(mkldnn::memory::primitive_desc& prim_desc)
{
    if (this->_prv_memory == nullptr)
        return false;
    prim_desc = this->_prv_memory->get_primitive_desc();
    return true;
}

template <typename Dtype>
std::shared_ptr<MKLDNNData<Dtype>> get_mkldnn_prv_descriptor(std::shared_ptr<MKLMemHolder> blob)
{
    std::shared_ptr<PrvMemDescr> blob_prv_mem_descriptor = blob->get_prv_descriptor();
    if (blob_prv_mem_descriptor == nullptr)
        return nullptr;
    assert(blob_prv_mem_descriptor->get_descr_type() == PrvMemDescr::PRV_DESCR_MKLDNN);
    std::shared_ptr<MKLDNNData<Dtype>> blob_prv_mkldnn_mem_descr =
        std::static_pointer_cast<MKLDNNData<Dtype>>(blob_prv_mem_descriptor);
    assert(blob_prv_mkldnn_mem_descr != NULL);
    return blob_prv_mkldnn_mem_descr;
}

template class MKLDNNMemoryDescriptor<half>;
template class MKLDNNMemoryDescriptor<float>;
template class MKLDNNMemoryDescriptor<double>;
template struct MKLDNNMemoryDescriptorBase<half>;
template struct MKLDNNMemoryDescriptorBase<float>;
template struct MKLDNNMemoryDescriptorBase<double>;

template <typename DType>
std::shared_ptr<PrvMemDescr> MKLDNNData<DType>::get_copy()
{
    std::shared_ptr<MKLDNNData<DType>> new_data;
    new_data.reset(new MKLDNNData<DType>(this->_usr_memory_pd, this->_prv_memory_pd));
    new_data->allocate();
    void* private_ptr = new_data->prv_ptr();
    memcpy(private_ptr, this->prv_ptr(), this->prv_size());
    return new_data;
}
template <typename DType>
bool PrvMemDescr::add_to(bool usr_pd, std::shared_ptr<PrvMemDescr> to,
    void *to_cpu_ptr, std::shared_ptr<PrvMemDescr> from, void *from_cpu_ptr)
{
    std::shared_ptr<MKLDNNData<DType> > from_data = std::static_pointer_cast<MKLDNNData<DType> >(from);
    std::shared_ptr<MKLDNNData<DType> > to_data = std::static_pointer_cast<MKLDNNData<DType> >(to);
    MKLDNNSumOp<DType> sumOp;
    if (usr_pd) {
        return sumOp.direct_usr_two_sum((DType*)to_cpu_ptr, (DType*)from_cpu_ptr,
            (DType*)to_cpu_ptr, to_data->usr_memory_pd());
    }
    else {
        return sumOp.direct_prv_two_sum(*to_data, *from_data, *to_data);
    }
}
template <typename DType>
void MKLDNNData<DType>::get_sum(std::shared_ptr<PrvMemDescr> other)
{
    void* prv_ptr = this->prv_ptr();
    void* other_prv_ptr = other->prv_ptr();
    size_t prv_ct = this->prv_count();
    size_t other_prv_ct = other->prv_count();
    if (prv_ct != other_prv_ct)
    {
        fprintf(stderr, "SUM of a and b must have equal size\n");
        return;
    }
    if (std::is_same<DType, float>::value)
    {
        cblas_saxpy((MKL_INT) this->prv_count(), 1, reinterpret_cast<float*>(other_prv_ptr), 1,
                    reinterpret_cast<float*>(prv_ptr), 1);
    }
    else
    {
        fprintf(stderr, "SUM of MKLDNN only support float so far\n");
        return;
    }
}
template class MKLDNNData<half>;
template class MKLDNNData<float>;
template class MKLDNNData<double>;

template bool PrvMemDescr::add_to<float>(bool usr_pd, std::shared_ptr<PrvMemDescr> to,
    void *to_cpu_ptr, std::shared_ptr<PrvMemDescr> from, void *from_cpu_ptr);
template bool PrvMemDescr::add_to<double>(bool usr_pd, std::shared_ptr<PrvMemDescr> to,
    void *to_cpu_ptr, std::shared_ptr<PrvMemDescr> from, void *from_cpu_ptr);
} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
#endif // #ifdef MKLDNN_SUPPORTED
