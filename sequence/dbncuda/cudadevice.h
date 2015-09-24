// cudadevice.h - holds the buffers, events, and streams used on a per device basis
//
// 1     5/29/12 AdamE
// created

#pragma once

#include <cuda_runtime_api.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <hash_set>
using namespace stdext;

namespace msra { namespace cuda {

const int deviceMax = 8;

// an object that lives in a device --this class just implements setdevice() and associated storage, and is shared across matrix and vectorbaseimpl
class objectondevice
{
protected:
    size_t deviceid;                    // CUDA card in which this matrix lives ("virtual" index amongst cards allocated to this process); default: 0
    CUcontext getdevicecontext() const { return msra::cuda::getdevicecontext (deviceid); }
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
    ondevice (size_t deviceid) { /*setdevicecontext (deviceid);*/ }
    ~ondevice() { /*cleardevicecontext(); */}
};

class BufEventStream
{
private:
	void* _buffer;	// buffer
	size_t _device;
	size_t _size; //size in bytes
	cudaEvent_t _bufferAvailable;	// event to fire when buffer is available
	cudaEvent_t _bufferComplete;	// event to fire when buffer is fully populated
	cudaStream_t _stream;	// stream to use

public:
	BufEventStream(int device, size_t size = 0)
	{
		_device = device;
		_buffer = NULL;
		_size = size;
		if (size > 0)
		{
			AllocateBuffer(size);
		}
		CreateStreamAndEvents(device);
	}

	~BufEventStream()
	{
		FreeStreamAndEvents();
	}

	size_t Device() {return _device;}
	void SetDevice(size_t device) 
	{ 
		if (device != _device)
		{
			FreeStreamAndEvents(); 
			CreateStreamAndEvents(device); 
			_device = device; 
		}
	}

	void* Buffer() {return _buffer;}	// buffer

	// Set buffer to user buffer, not managed by class
	void SetBuffer(void* buffer)	
	{
		FreeBuffer();
		_buffer = buffer;
		_size=0;
	}

	// Allocate a buffer of size bytes
	void AllocateBuffer(size_t size) 
	{ 
		void* buffer = NULL;
		cudaMallocHost(&buffer, size) || "AllocateBuffer() failed"; 
		SetBuffer(buffer);
		_size = size;
		}

	// Free any allocated buffer
	void FreeBuffer() 
	{ 
		if (_buffer != NULL && _size > 0) 
		{
			ondevice on(_device); 
			cudaFreeHost(_buffer) || "Freeing buffer in FreeBuffer() failed";
			_buffer = NULL;
			_size = 0;
		}
	}

	// make sure the buffer is at least (size) bytes long
	void EnsureSize(size_t size)
	{
		if (size == 0)
		{
			//FreeBuffer();
			return;
		}

		//if (_size == 0)
		//	throw std::logic_error("EnsureSize() can't reallocate a private buffer");
		if (_size < size)
		{
			//TODO: we need to free these buffers, but have to wait until it's safe, should make a cleanup routine somewhere
			//FreeBuffer();
			AllocateBuffer(size);
		}
	}

	cudaEvent_t EventAvailable() {return _bufferAvailable;}	// event to fire when our input buffer is available
	cudaEvent_t EventComplete() {return _bufferComplete;}	// event to fire when our output buffer is fully populated
	cudaStream_t Stream() {return _stream;}	// stream the step runs on

	void CreateStreamAndEvents(size_t device)
	{
		ondevice on(device);
		cudaStreamCreate(&_stream) || "Creating stream failed";
		cudaEventCreate(&_bufferAvailable) || "Event creation failed";
		cudaEventCreate(&_bufferComplete) || "Event creation failed";
	}

	void FreeStreamAndEvents()
	{
		ondevice on(_device);
		if (_stream != NULL)
			cudaStreamDestroy(_stream) || "Destroy stream failed";
		if (_bufferAvailable != NULL)
			cudaEventDestroy(_bufferAvailable) || "Destroy event failed";
		if (_bufferComplete != NULL)
			cudaEventDestroy(_bufferComplete) || "Destroy event failed";
	}

	// raise the complete event and then have streamNext wait until it's done
	// This function does not block the CPU, only the stream passed in will wait.
	void RaiseEventAndWait(cudaStream_t streamNext)
	{
		// raise the event that says the copy is complete
		cudaEventRecord(EventComplete(), Stream()) || "RaiseEventAndWait() error during EventRecord";

		if (streamNext != NULL)
		{
			// wait for the event copy to the buffer to complete first
			cudaStreamWaitEvent (streamNext, EventComplete(), 0) || "RaiseEventAndWait() error on StreamWait";
		}
	}

	// wait until the Available event is raised, then compute on streamNext will continue
	void WaitUntilAvalilable(cudaStream_t streamNext)
	{
		if (streamNext != NULL)
		{
			// wait for the event copy to the buffer to complete first
			cudaStreamWaitEvent (streamNext, EventAvailable(), 0) || "WaitUntilAvalilable() error on StreamWait";
		}
	}

	// raise the available event and optionally wait for the event immediately
	void RaiseAvailableEvent(cudaStream_t streamNext=NULL)
	{
		// raise the event that says the copy is complete
		cudaEventRecord(EventAvailable(), Stream()) || "RaiseEventAndWait() error during EventRecord";

		if (streamNext != NULL)
		{
			// wait for the event copy to the buffer to complete first
			WaitUntilAvalilable(streamNext);
		}
	}
};

// device information 
class DeviceInfo
{
	int _device;	// device we are allocated on
	BufEventStream* _copyToBufES;
	BufEventStream* _copyFromBufES;
	BufEventStream* _computeES;

public:
	DeviceInfo(int device)
	{
		_device = device;
		_copyToBufES = NULL;
		_copyFromBufES = NULL;
		_computeES = NULL;
	}

	~DeviceInfo()
	{
		// clear out the copy buffers as well
		if (_copyToBufES != NULL)
		{
			delete _copyToBufES;
			_copyToBufES = NULL;
		}

		// clear out the copy buffers as well
		if (_copyFromBufES != NULL)
		{
			delete _copyFromBufES;
			_copyFromBufES = NULL;
		}

		// clear out the compute
		if (_computeES != NULL)
		{
			delete _computeES;
			_computeES = NULL;
		}
	}

	// get the buffer/Events/Stream  either by creating it or making sure it's buffer is big enough
	BufEventStream* GetBufferES(BufEventStream** bufES, size_t size)
	{
		if (*bufES == NULL)
		{
			*bufES = new BufEventStream(_device, size);
		}
		else
		{
			(*bufES)->EnsureSize(size);
		}
		return *bufES;
	}

	// get the copy buffer and associated streams and events
	BufEventStream* GetCopyToES(size_t size = 0) 
	{
		return GetBufferES(&_copyToBufES, size);
	}

	// get the copy buffer and associated streams and events
	BufEventStream* GetCopyFromES(size_t size = 0) 
	{
		return GetBufferES(&_copyFromBufES, size);
	}

	// get the copy buffer and associated streams and events
	BufEventStream* GetComputeES(size_t size = 0) 
	{
		return GetBufferES(&_computeES, size);
	}
};

class Devices
{
	DeviceInfo* deviceInfo[deviceMax];
	hash_set<const float *> pinnedBuffers;
public:
	Devices()
	{
		Init();
	}

	void Init()
	{
		// allocate the pools for each possible device
		for (int device=0; device <= deviceMax; device++)
		{
			deviceInfo[device] = new DeviceInfo(device);
		}
	}

	// Cleanup the pools
	// WARNING: if this gets called after the cuda context is destroyed, exceptions will be thrown on cudaFree()
	// That is because when the context is destroyed all buffers associated with that context are destroyed with it.
	void Cleanup()
	{
		// allocate the pools for each possible device
		for (int device=0; device < deviceMax; device++)
		{
			delete deviceInfo[device];
			deviceInfo[device] = NULL;
		}	

		//probably should unpin buffers here, but it will probably automatically happen as process goes away
	}

	~Devices()
	{
		// WARNING: if this gets called after the cuda context is destroyed, you will throw exceptions
		Cleanup();
	}

	BufEventStream* GetCopyToES(size_t device, size_t size=0) 
	{
		return deviceInfo[device]->GetCopyToES(size);
	}

	BufEventStream* GetCopyFromES(size_t device, size_t size=0) 
	{
		return deviceInfo[device]->GetCopyFromES(size);
	}

	BufEventStream* GetComputeES(size_t device, size_t size=0) 
	{
		return deviceInfo[device]->GetComputeES(size);
	}

	// Is the buffer passed in pinned?
	bool IsPinned(const float *bufHost)
	{
		hash_set<const float *>::iterator found = pinnedBuffers.find(bufHost);
		// see if we found the pointer or not
		return (found != pinnedBuffers.end());
	}

	// pin the buffer so we can use it for fast copy
	// if the buffer is already pinned, do nothing
	void PinBuffer(const float *bufHost, size_t size)
	{
		if (!IsPinned(bufHost))
		{
			cudaHostRegister((void *)bufHost, size, cudaHostRegisterPortable) || "cudaHostRegister() fail in PinBuffer()";
			pinnedBuffers.insert(bufHost);
		}
	}

	// unpin the buffer, if it wasn't pinned, do nothing
	// WARNING: Unpin operations do a CPU sync
	void UnpinBuffer(const float *bufHost)
	{
		hash_set<const float *>::iterator found = pinnedBuffers.find(bufHost);
		// if we didn't find the buffer, exit
		if (found == pinnedBuffers.end())
			return;

		cudaHostUnregister((void *)bufHost) || "cudaHostUnregister() faile in UnpinBuffer()";
		pinnedBuffers.erase(found);
	}
};


extern Devices g_devices;	// one global device pool
}}
