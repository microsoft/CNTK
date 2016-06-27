// cudadevice.h - holds the buffers, events, and streams used on a per device basis
//
// F. Seide, V-hansu

#pragma once

#include <cuda_runtime_api.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <unordered_set>

namespace msra { namespace cuda {

const int deviceMax = 8;

// an object that lives in a device --this class just implements setdevice() and associated storage, and is shared across matrix and vectorbaseimpl
class objectondevice
{
protected:
    size_t deviceid; // CUDA card in which this matrix lives ("virtual" index amongst cards allocated to this process); default: 0
protected:
    objectondevice(size_t d)
        : deviceid(d)
    {
    }

public:
    size_t getdevice() const
    {
        return deviceid;
    }
};

// auto-class to set device (through context) inside a function
// usage at each function that calls CUDA:
//  ondevice no (deviceid);
class ondevice
{
public:
    ondevice(size_t deviceid)
    {
        auto rc = cudaSetDevice((int)deviceid);
        if (rc != cudaSuccess)
            RuntimeError("Cannot set cuda device: %s (cuda error %d)", cudaGetErrorString(rc), (int)rc);
    }
};
} }
