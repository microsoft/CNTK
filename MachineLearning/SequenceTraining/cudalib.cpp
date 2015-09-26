// cudalib.cpp -- all CUDA calls (but not cublas) are encapsulated here
// All actual CUDA API calls go here, to keep the header out of our other headers.
//
// F. Seide, V-hansu

#define _CRT_SECURE_NO_WARNINGS 1    // so we can use getenv()...

#include <cuda_runtime_api.h>           // for CUDA API
#include <cuda.h>                       // for device API
#include "cudalib.h"
#include "cudadevice.h"
#include <string>
#include <assert.h>

#undef NOMULTIDEVICE       // define this to disable any context/driver stuff

#ifndef NOMULTIDEVICE
#pragma comment (lib, "cuda.lib")       // link CUDA device API
#endif
#pragma comment (lib, "cudart.lib")     // link CUDA runtime

namespace msra { namespace cuda {

static int devicesallocated = -1;    // -1 means not initialized

// allows to write cudaFunction() || "error"   (CUDA runtime)
static void operator|| (cudaError_t rc, const char * msg)
{
    if (rc != cudaSuccess)
    {
        char buf[1000];
        sprintf(buf, "%s: %s (cuda error %d)", msg, cudaGetErrorString (rc), rc);
        throw std::runtime_error (buf);
    }
}

// allows to write cuFunction() || "error"  (CUDA API)
static void operator|| (CUresult rc, const char * msg)
{
    if (rc != CUDA_SUCCESS)
    {
        char buf[1000];
        sprintf(buf, "%s: cuda API error %d", msg, rc);
        throw std::runtime_error (buf);
    }
}

// CUDA device contexts
class cudadevicecontext
{
    CUcontext cuContext;
    bool isvalid() const { return cuContext != NULL; }
public:
    cudadevicecontext() : cuContext (NULL) {}
#ifndef NOMULTIDEVICE
    ~cudadevicecontext() { if (isvalid()) cuCtxDestroy (cuContext); }
#endif
    // link this item to a specific physical id
    void init (size_t physicaldeviceid)
    {
#ifndef NOMULTIDEVICE
        //if (isvalid())
        //    throw std::runtime_error ("init: only call thisonce");
        CUdevice cuDevice;
        cuDeviceGet (&cuDevice, (int) physicaldeviceid) || "cuDeviceGet failed";  // map numeric id to a pointer
        cuCtxCreate (&cuContext, 0/*flags*/, cuDevice) || "cuCtxCreate failed";
        assert (cuContext != NULL);
        CUcontext cuContextDummy;
        cuCtxPopCurrent (&cuContextDummy) || "cuCtxPopCurrent failed  --should never happen";
        assert (cuContext == cuContextDummy);
        // show some info to the user
        char namebuf[1024] = { 0 };
        cuDeviceGetName (&namebuf[0], sizeof(namebuf) -1, cuDevice) || "cuDeviceGetName failed";
        fprintf (stderr, "using physical CUDA device %d: %s\n", (int)physicaldeviceid, namebuf);
#endif
    }
    // cast this to the CUcontext for use with CUDA functions
    operator CUcontext() const 
    { 
        if (!isvalid()) 
            throw std::logic_error ("CUcontext(): item not initialized"); 
        return cuContext; 
    }
};


static cudadevicecontext cudadevicecontexts[deviceMax];      // note: increase if we need to support more CUDA cards
static cudadevicecontext * currentcudadevicecontext = NULL; // global state: remembered which device context was set
static size_t deviceCurrent = 0;


cudaStream_t GetStream(size_t deviceid)
{
    return g_devices.GetComputeES(deviceCurrent)->Stream();
}

cudaEvent_t GetEvent(size_t deviceid)
{
    return g_devices.GetComputeES(deviceCurrent)->EventComplete();
}

size_t GetCurrentDevice()
{
    return deviceCurrent;
}

//cudaStream_t GetCurrentStream() {return GetStream(GetCurrentDevice());}
cudaStream_t GetCurrentStream() { return cudaStreamDefault; }
cudaEvent_t GetCurrentEvent() {return GetEvent(GetCurrentDevice());}
Devices g_devices;    // one global device pool

// initialize CUDA system
void lazyinit()
{
}

void initwithdeviceid(size_t deviceid)
{
    if (devicesallocated >= 0) return;
    devicesallocated = 0;
    cudadevicecontexts[devicesallocated].init(deviceid);
    devicesallocated++;
    fprintf(stderr, "using  CUDA devices%d \n", (int)deviceid);
}

// get number of devices
size_t numdevices()
{
    if (devicesallocated < 0)
        throw std::logic_error ("numdevices: called when not initialized");
    return devicesallocated;
}

// synchronize with ongoing thread

void join() 
{ 
//#ifndef ASYNCCOPY // no-sync framework should have sync commands where necessary in the code, not a join() function
    cudaDeviceSynchronize() || "cudaDeviceSynchronize failed";
//#endif
} 

// allocate a stack to store the devices that have been pushed
const int stackSize = 20;
static int curStack = 0;
static size_t deviceStack[stackSize] = {0};

void setdevicecontext (size_t deviceid)
{
#ifndef NOMULTIDEVICE
    //if (currentcudadevicecontext != NULL)
    //    throw std::logic_error ("setdevicecontext: a device context has already been set --??");
    if (deviceid >= (sizeof(cudadevicecontexts) / sizeof(cudadevicecontext)))
        throw std::logic_error ("setdevicecontext: device id exceeds size of fixed-size array cudadevicecontexts[]");
    cudadevicecontext & c = cudadevicecontexts[deviceid];

    // push the old values
    cuCtxPushCurrent (c) || "cuCtxPushCurrent failed";
    deviceStack[curStack++] = deviceCurrent;

    if (curStack >= stackSize)
        throw std::logic_error("setdevicecontext(): device stack overflow");

    // set the new ones
    deviceCurrent = deviceid;
    currentcudadevicecontext = &c;   // remember current context
#endif
}
void cleardevicecontext()   // call this only in pairs with setdevicecontext()
{
#ifndef NOMULTIDEVICE
    if (currentcudadevicecontext == NULL)
        throw std::logic_error ("cleardevicecontext: no device context set --??");
    if (curStack <= 0)
        throw std::logic_error ("cleardevicecontext(): stack is empty, can't pop");

    // pop off the stack
    CUcontext context;
    cuCtxPopCurrent (&context) || "cuCtxPopCurrent failed  --should REALLY never happen";
    deviceCurrent = deviceStack[--curStack];
    if (curStack > 0)
    {
        cudadevicecontext & c = cudadevicecontexts[deviceCurrent];
        currentcudadevicecontext = &c;
    }
    else
    {
        currentcudadevicecontext = NULL;
    }

#endif
}

CUcontext getdevicecontext (size_t deviceid)
{
#ifndef NOMULTIDEVICE
    return cudadevicecontexts[deviceid];
#else
    return NULL;    // TODO: is this the right behaviour?
#endif
}

// memory allocation
void * mallocbytes (size_t nelem, size_t sz)
{
    for (size_t retry = 0; ; retry++)
    {
        try
        {
            //fprintf (stderr, "mallocbytes: allocating %d elements of size %d, %d bytes\n", (int) nelem, (int) sz, (int) (nelem * sz));        // comment out by [v-hansu] to get rid out annoying output
            void * p;
            cudaMalloc (&p, nelem * sz) || "cudaMalloc failed";
            return p;
        }
        catch (const std::exception & e)
        {
            fprintf (stderr, "mallocbytes: failed with error %s\n", e.what());
            if (retry >= 5)
                throw;
        }
    }
}

void freebytes (void * p) { cudaFree (p) || "cudaFree failed"; }

void memcpyh2d (void * dst, size_t byteoffset, const void * src, size_t nbytes)
{
    cudaMemcpy (byteoffset + (char*) dst, src, nbytes, cudaMemcpyHostToDevice) || "cudaMemcpy failed";
}

void memcpyd2h (void * dst, const void * src, size_t byteoffset, size_t nbytes)
{
    cudaMemcpy (dst, byteoffset + (const char *) src, nbytes, cudaMemcpyDeviceToHost) || "cudaMemcpy failed";
}

//cudadevicecontext::cudathread::cudathread (size_t deviceid) { cudaSetDevice ((int) deviceid) || "cudaSetDevice failed"; }
//cudadevicecontext::cudathread::~cudathread() { cudaThreadExit(); };

};};
