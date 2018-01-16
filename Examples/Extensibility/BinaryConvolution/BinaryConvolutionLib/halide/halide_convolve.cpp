//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Halide.h"
#include "HalideRuntime.h"
#include <stdio.h>

using namespace Halide;
int main(int argc, char **argv) {
    ImageParam input(type_of<float>(), 3, "input");
    ImageParam weights(type_of<float>(), 2, "weights");
    
    Param<int> size("size");
    Param<bool> pad("pad");
    Param<int> stride("stride");
    Param<int> out_x("outx");
    Param<int> out_y("outy");

    Var x("x"), y("y"), c("c"), f("f"), k("k");
    
    Target target;
    //target = get_host_target();
    target.os = Target::Windows;
    target.arch = Target::X86;
    target.bits = 64;

    std::vector<Target::Feature> profile_features;
    profile_features.push_back(Target::AVX);
    profile_features.push_back(Target::SSE41);
    //profile_features.push_back(Target::Profile);
    target.set_features(profile_features);

    Func Input("Input");
    Func Weights("Weights");
    Input(x, y, c) = BoundaryConditions::constant_exterior(input, 0)(x, y, c);
    Weights(x, f) = BoundaryConditions::constant_exterior(weights, 1)(x, f);

    Func binarize_input("binarize_input");
    RDom r(0, 64);

    //Expr width_col = select(pad, input.width(), (input.width() - size)/stride + 1);
    //Expr height_col = select(pad, input.height(), (input.height() - size)/stride + 1);

    //Expr w_offset = (y * stride) % out_x;
    //Expr h_offset = (((y * stride) / out_x) * stride) % out_y;
    Expr w_offset = (y % out_x)*stride;
    Expr h_offset = ((y / out_x) % out_y) * stride;

    Expr im_row = h_offset + ((64*x + r.x)/size) % size - select(pad, size/2, 0); 
    Expr im_col = w_offset + (64*x + r.x) % size - select(pad, size/2, 0); 
    Expr im_chan = (64*x + r.x) / size / size;
    
    /*Expr im_row = print_when(y==1, h_offset + ((64*x + r.x)/size) % size - select(pad, size/2, 0), "<-- ROW"); 
    Expr im_col = print_when(y==1, w_offset + (64*x + r.x) % size - select(pad, size/2, 0), "<-- COL\n"); 
    Expr im_chan = print_when(y==1, (64*x + r.x) / size / size, "<-- CHA");
    */


    binarize_input(x, y) = sum(select(Input(im_col, im_row, im_chan) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_inputs"); 

    Func binarize_weights("binarize_weights");
    Func alpha("alpha");
    RDom n(0, weights.width());
    binarize_weights(x, f) = sum(select(Weights(64*x + r.x, f) > 0, (cast<int64_t>(1)) << r.x, cast<int64_t>(0)), "compress_weights");
    alpha(f) = sum(abs(Weights(n.x, f))/weights.width(), "compute_alpha");

    Func xnor("xnor");
    xnor(k, x, y) = popcount(binarize_weights(k, y) ^ binarize_input(k, x));
    //xnor(k, x, y) = popcount(binarize_weights(k, y));

    Func output("output");
    Expr bin_width = weights.width()/64;
    RDom bw(0, bin_width);
    output(x, y) = -alpha(y) * ((2 * cast<float>(sum(xnor(bw.x, x, y), "accumulate"))) - (64*bin_width));

    // scheduling
       
    Var x_inner, x_outer, y_inner, y_outer;
    binarize_weights.compute_root();
    binarize_weights.vectorize(x, 8);
    binarize_weights.parallel(f, 8);
    alpha.compute_root();
    alpha.vectorize(f, 8);
    output.reorder(y, x);
    //binarize_input.compute_root();
    //output.unroll(y, 4);
    output.vectorize(y, 8);
    output.parallel(x, 8);
    binarize_input.compute_at(output, x);
    
    std::vector<Argument> args = {weights, input, size, stride, pad, out_x, out_y};
    output.compile_to_static_library("halide_convolve", args, "halide_convolve", target);
    //output.compile_to_file("halide_convolve", args, "halide_convolve", target);
    return 0; 
} 
