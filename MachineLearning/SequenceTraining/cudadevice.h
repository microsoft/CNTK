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
    size_t deviceid;                    // CUDA card in which this matrix lives ("virtual" index amongst cards allocated to this process); default: 0
protected:
    objectondevice (size_t d = 0) : deviceid (d) { }
public:
    void setdevice (size_t deviceid) { this->deviceid = deviceid; }    // just remembers it here
    size_t getdevice() const {return deviceid;} 
};

// auto-class to set device (through context) inside a function
// usage at each function that calls CUDA:
//  ondevice no (deviceid);
class ondevice
{
public:
    ondevice (size_t /*deviceid*/) { /*setdevicecontext (deviceid);*/ }
    ~ondevice() { /*cleardevicecontext(); */}
};

}}
