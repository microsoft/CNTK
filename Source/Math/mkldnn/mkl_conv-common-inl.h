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
#ifndef CNTK_MKL_CONV_COMMON_INL_H_
#define CNTK_MKL_CONV_COMMON_INL_H_

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mkl_util-inl.h"

namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <typename DType>
class MKLConvCommon
{
public:
    MKLConvCommon()
        : width_(0),
          height_(0),
          width_out_(0),
          height_out_(0),
          kernel_w_(0),
          kernel_h_(0),
          stride_w_(0),
          stride_h_(0),
          pad_l_w_(0),
          pad_l_h_(0),
          pad_r_w_(0),
          pad_r_h_(0)
    {
    }
    virtual ~MKLConvCommon() {}

protected:
    int width_, height_, width_out_, height_out_, kernel_w_, kernel_h_, stride_w_, stride_h_;
    int group_, num_, channel_output_;
    int channels_;
    int pad_l_w_, pad_l_h_;
    int pad_r_w_, pad_r_h_;
};

} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
#endif // CNTK_MKL_CONV_COMMON_INL_H_
