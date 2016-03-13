#pragma once

#include "NDArrayView.h"

namespace CNTK
{
    class Value
    {
    public:
        // A multi-dimensional value tensor with no mask
        Value(NDArrayView data); 

        // The mask allows specifying certain sample locations in data to be marked as invalid
        // for purposes of batching variable lenght sequences.
        // The mask array view is typically lower dimensionailty than the data, which means
        // values are masked in units of (data.rank() - mask.rank()) dimensional values
        // along the least significat dimensions of the data
        Value(NDArrayView data, NDArrayView mask);

        NDArrayView data;
        NDArrayView mask;
    };

    // Builtin methods
    Value RandomNormal(std::vector<long long> dimensions, float mean, float stdDev);
    Value RandomUniform(std::vector<long long> dimensions, float rangeStart, float rangeEnd);
    Value Const(std::vector<long long> dimensions, float value);
}
