#include "stdafx.h"
#include "../Matrix.h"
#include "../ConvolveGeometry.h"
#include "mkldnn_sum-inl.h"
#include "mkldnn_convolution-inl.h"
#ifdef USE_MKLDNN

namespace Microsoft { namespace MSR { namespace CNTK {

template<> int MKLDNNSumOp<float>::s_id_gen = 1;
template<> int MKLDNNSumOp<double>::s_id_gen = 1;
template<> int MKLDNNConvolutionOp<float>::s_id_gen = 1;
template<> int MKLDNNConvolutionOp<double>::s_id_gen = 1;


}}}
#endif //USE_MKLDNN