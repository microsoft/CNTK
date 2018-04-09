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
#include "../Matrix.h"
#include "mkl_memory.h"
#ifdef USE_MKLDNN

#include "mkldnn_base-inl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

bool enableMKLDNNWarnGenerated() {
  return false;
}
std::shared_ptr<MKLDNNStream> StreamHolder::get_stream() {
    if (this->_current_stream == NULL || !this->_current_stream->ready()) {
        _current_stream.reset(new MKLDNNStream());
    }
    return _current_stream;
}

template <typename Dtype>
std::shared_ptr<MKLDNNStream>  MKLDNNPrimitive<Dtype>::get_mkldnn_stream() {
    if (mkldnn_stream == NULL)
        mkldnn_stream = StreamHolder::Instance().get_stream();
    else
        StreamHolder::Instance().prepare_mkldnn_stream(mkldnn_stream);
    return mkldnn_stream;
}

template <typename Dtype>
std::shared_ptr<MKLDNNStream>  MKLDNNPrimitive<Dtype>::submit() {
    assert(this->aprimitive);
    this->get_mkldnn_stream()->submit({*(this->aprimitive)});
    return mkldnn_stream;
}

template class MKLDNNLayer<double>;
template class MKLDNNLayer<float>;
template class MKLDNNPrimitive<double>;
template class MKLDNNPrimitive<float>;
}}}
#endif
