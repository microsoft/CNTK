// SGD.cpp -- implements SGD with all bells and whistles, parallelization, randomizatiom, etc.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "SGD.h"
#include "MultiNetworksSGD.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template class SGD<float>;
template class SGD<double>;

template class MultiNetworksSGD<float>;
template class MultiNetworksSGD<double>;

}}}
