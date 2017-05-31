//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#ifndef CONVOLVE_WRAPPER
#define CONVOLVE_WRAPPER
#include "halide/halide_convolve.h"

// perform all the boilerplate needed by halide. Basically takes a bunch of input parameters and packages them up into halide structs
void invoke_halide_convolve(const float *filter, const float *input, int num_filters, int size, int channels, bool pad, int stride, int w, int h, const float *output) {
    int out_w = pad == 0 ? (w - size)/stride + 1 : (w - 1)/stride + 1;
    int out_h = pad == 0 ? (h - size)/stride + 1 : (h - 1)/stride + 1;
    
    // package up the filter buffer
    halide_buffer_t halide_filter_buf = {0};
    halide_filter_buf.host = (uint8_t *)&filter[0];
    halide_dimension_t filter_buf_dims[2];
    filter_buf_dims[0].min = 0;
    filter_buf_dims[0].extent = size*size*channels;
    filter_buf_dims[0].stride = 1;
    filter_buf_dims[1].min = 0;
    filter_buf_dims[1].extent = num_filters;
    filter_buf_dims[1].stride = size*size*channels;
    halide_filter_buf.dim = filter_buf_dims;
    struct halide_type_t filter_type;
    filter_type.code = halide_type_float;
    filter_type.bits = 32;
    filter_type.lanes = 1;
    halide_filter_buf.type = filter_type;
    halide_filter_buf.dimensions = 2;

    // package the input buffer
    halide_buffer_t halide_input_buf = {0};
    halide_input_buf.host = (uint8_t *)&input[0];
    halide_dimension_t input_buf_dims[3];
    input_buf_dims[0].min = 0;
    input_buf_dims[0].extent = w;
    input_buf_dims[0].stride = 1;
    input_buf_dims[1].min = 0;
    input_buf_dims[1].extent = h;
    input_buf_dims[1].stride = w;
    input_buf_dims[2].min = 0;
    input_buf_dims[2].extent = channels;
    input_buf_dims[2].stride = w*h;
    halide_input_buf.dim = input_buf_dims;
    struct halide_type_t input_type;
    input_type.code = halide_type_float;
    input_type.bits = 32;
    input_type.lanes = 1;
    halide_input_buf.type = input_type;
    halide_input_buf.dimensions = 3;

    // package the output buffer
    halide_buffer_t halide_output_buf = {0};
    halide_output_buf.host = (uint8_t *)&output[0];
    halide_dimension_t output_buf_dims[2];
    output_buf_dims[0].min = 0;
    output_buf_dims[0].extent = out_h*out_w;
    output_buf_dims[0].stride = 1;
    output_buf_dims[1].min = 0;
    output_buf_dims[1].extent = num_filters;
    output_buf_dims[1].stride = out_h*out_w;
    halide_output_buf.dim = output_buf_dims;
    struct halide_type_t output_type;
    output_type.code = halide_type_float;
    output_type.bits = 32;
    output_type.lanes = 1; 
    halide_output_buf.type = output_type;
    halide_output_buf.dimensions = 2;
    
    // call into halide_convolve to compute the binary convolution
    halide_convolve(&halide_filter_buf, &halide_input_buf, size, stride, pad, out_w, out_h, &halide_output_buf);
}

#endif
