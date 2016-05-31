#pragma once

#include <exception>
#include <algorithm>

static const double relativeTolerance = 0.001f;
static const double absoluteTolerance = 0.000001f;

template <typename ElementType>
inline void FloatingPointVectorCompare(const std::vector<ElementType>& first, const std::vector<ElementType>& second, const char* message)
{
    for (size_t i = 0; i < first.size(); ++i)
    {
        ElementType leftVal = first[i];
        ElementType rightVal = second[i];
        ElementType allowedTolerance = (std::max<ElementType>)((ElementType)absoluteTolerance, ((ElementType)relativeTolerance) * leftVal);
        if (std::abs(leftVal - rightVal) > allowedTolerance)
            throw std::exception(message);
    }
}
