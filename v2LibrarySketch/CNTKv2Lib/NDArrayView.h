#pragma once

#include <vector>
#include <memory>

namespace CNTK
{
    enum class ValueType
    {
        Bit,
        Char,
        UChar,
        Short,
        UShort,
        Int,
        Uint,
        Long,
        ULong,
        Float8,
        Float16,
        Float,
        Double,
    };

    // Represents a multi-dimensional array of values.
    // The underlying array may be stored in sparse or dense form, supports broadcasting
    // and is located on the CPU or the GPU. 
    // TODO: Should this type have CUDA __device__ support allowing its direct use in CUDA kernels
    // TODO: Should we fully rid the API of template types and just use the template methods
    // at the leaf level where actual data is accessed?
    class NDArrayView
    {
    public:
        enum class StorageType
        {
            DENSE,
            SPARSE_CSC,
            // TODO: Others?
        };

    public:
        // Construct a N dimensional view over a dense CPU buffer
        NDArrayView(void* buffer, size_t bufferSizeInBytes, ValueType dataType, const std::vector<long long>& viewDimensions);

        ValueType ElementType() const;

        const std::vector<long long>& Dimensions() const;

        StorageType StorageType() const;

        // TODO: Define the full set of constructors for creating views over
        // sparse as well as dense buffers on the CPU or a GPU.
        // Also, add support for custom deleters to enable the destruction of
        // externally supplied buffer if the user desires to "move" the passed buffer
        // to the constructed NDArrayView
    };
}
