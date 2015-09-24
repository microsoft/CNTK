// cudalib.cpp -- all CUDA calls (but not cublas) are encapsulated here
// All actual CUDA API calls go here, to keep the header out of our other headers.
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudalib.cpp $
// 
// 39    11/21/12 8:58p V-hansu
// rename state2classmap to senone2classmap
// 
// 38    10/28/12 10:14p V-hansu
// (change tabs to spaces)
// 
// 37    9/30/12 11:19a V-hansu
// comment out the "mallocbytes" output
// 
// 36    9/25/12 4:26p Fseide
// added a desparate retry loop for cudaMalloc() hoping to trick the
// 'unknown error 30', but to no avail
// 
// 35    9/04/12 2:40p Fseide
// (spacing)
// 
// 34    8/28/12 3:56p Fseide
// resurrected memcpy() between CUDA and CPU
// 
// 33    7/17/12 5:31p Adame
// Update for no-sync framework
// async copy fixes
// 
// 32    6/24/12 9:29p V-xieche
// switch code into a work point(an old version).
// 
// 30    6/06/12 5:11p Adame
// Copy Sync update
// 
// 29    4/03/12 8:42p V-xieche
// add oversubscribe for device number from Frank, to emulate the
// situation virtual device greater than physical device num.
// 
// 27    4/01/12 8:47p Fseide
// changed assignmatrix_ua() to assignmatrix_p2p() using explicit
// contexts;
// new methods for that: getdevicecontext()
// 
// 26    4/01/12 4:48p Fseide
// enabled enabling peer accesses, to be tested
// 
// 25    4/01/12 4:47p Fseide
// added code for peer-to-peer access without UA, but not enabled yet
// 
// 24    11/17/11 2:46p Fseide
// bug fix in lockdevicebymutex(): it is possible that the mutex is locked
// and we don't get a handle due to access denied. This used to lead to an
// exception being thrown, while now it is understood as 'device in use'
// 
// 23    11/16/11 3:02p Fseide
// (fixed a Tab)
// 
// 22    11/16/11 2:36p Fseide
// now showing number of CUDA cards found
// 
// 21    7/26/11 9:29a V-xieche
// fix a debug error in release the handle for selecting the CUDA device
// 
// 20    7/26/11 9:09a V-xieche
// fix a minor bug in release the lock when exclusively choose one CUDA
// device
// 
// 19    7/26/11 8:59 Fseide
// fixed lockdevicebymutex() to cover all error conditions;
// fixed erroneous TAB characters in lockdevicebymutex()-related functions
// 
// 17    7/22/11 3:57p Fseide
// implemented exclusive use of GPGPU devices and limiting the number of
// devices to 1 by default, to allow multiple processes per machine
// 
// 16    2/24/11 9:51p Fseide
// removed the memcpy functions (using CUBLAS instead)
// 
// 15    2/24/11 8:27p Fseide
// cudadevicecontext::lazyinit() now allows out-of-bounds device ids for
// debugging, to simulate multiple devices (they use different contexts)
// 
// 14    2/11/11 5:28p Fseide
// added compiler flag NOMULTIDEVICE to disable the CUDA
// context-switchigng stuff
// 
// 13    2/11/11 3:42p Fseide
// implemented lazyinit() and new cudadevicecontext
// 
// 12    2/01/11 7:39p Fseide
// (added a comment)
// 
// 11    2/01/11 7:09p Fseide
// now waits after each memory transfer until its completion--not clear if
// this is needed or not
// 
// 10    2/01/11 3:46p Fseide
// cudalib now clearly encapsulates all CUDA API calls (but not cublas)
// 
// 9     2/01/11 10:50a Fseide
// added cudadevicecontext::cudathread implementation
// 
// 8     1/31/11 9:05p Fseide
// new function join()
// 
// 7     1/31/11 3:33p Fseide
// added the CUDA #include
// 
// 6     1/31/11 11:50a Fseide
// implementation of vector nearly complete except the multi-threaded
// thingy
// 
// 5     1/31/11 12:01a Fseide
// numdevices() implemented
// 
// 4     1/30/11 11:19p Fseide
// changed to DLL-export cudamatrix instead of cudalib
// 
// 3     1/30/11 11:07p Fseide
// added #define DLLEXPORT
// 
// 2     1/30/11 19:02 Fseide
// stub for numdevices()
// 
// 1     1/30/11 17:51 Fseide
// created

#define _CRT_SECURE_NO_WARNINGS 1    // so we can use getenv()...

#include <Windows.h>            // for the Mutex
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
        sprintf_s (buf, "%s: %s (cuda error %d)", msg, cudaGetErrorString (rc), rc);
        throw std::runtime_error (buf);
    }
}

// allows to write cuFunction() || "error"  (CUDA API)
static void operator|| (CUresult rc, const char * msg)
{
    if (rc != CUDA_SUCCESS)
    {
        char buf[1000];
        sprintf_s (buf, "%s: cuda API error %d", msg, rc);
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
        cuDeviceGetName (&namebuf[0], _countof (namebuf) -1, cuDevice) || "cuDeviceGetName failed";
        fprintf (stderr, "using physical CUDA device %d: %s\n", physicaldeviceid, namebuf);
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


// try to acquire a device exclusively; managed through this library's private lock mechanism (i.e. not through CUDA APIs)
static bool lockdevicebymutex (int physicaldeviceid)
{
    wchar_t buffer[80];
    wsprintf (buffer, L"Global\\DBN.exe GPGPU exclusive lock for device %d", physicaldeviceid);
    // we actually use a Windows-wide named mutex
    HANDLE h = ::CreateMutex (NULL/*security attr*/, TRUE/*bInitialOwner*/, buffer);
    DWORD res = ::GetLastError();
    if (h == NULL)  // failure  --this should not really happen
    {
        if (res == ERROR_ACCESS_DENIED)    // no access: already locked by another process
        {
            fprintf (stderr, "lockdevicebymutex: mutex access denied, assuming already locked '%S'\n", buffer);
            return false;
        }
        fprintf (stderr, "lockdevicebymutex: failed to create '%S': %d\n", buffer, res);
        throw std::runtime_error ("lockdevicebymutex: unexpected failure");
    }
    // got a handle
    if (res == 0)   // no error
    {
        fprintf (stderr, "lockdevicebymutex: created and acquired mutex '%S'\n", buffer);
        return true;
    }
    // failure with handle  --remember to release the handle
    ::CloseHandle (h);
    if (res == ERROR_ALREADY_EXISTS)    // already locked by another process
    {
        fprintf (stderr, "lockdevicebymutex: mutex already locked '%S'\n", buffer);
        return false;
    }
    else if (res != 0)
    {
        fprintf (stderr, "lockdevicebymutex: unexpected error from CreateMutex() when attempting to create and acquire mutex '%S': %d\n", buffer, res);
        throw std::logic_error ("lockdevicebymutex: unexpected failure");
    }
    return false;
}
// initialize CUDA system
void lazyinit()
{
#if 0
    if (devicesallocated >= 0) return;
    int numphysicaldevices = 0;
    cudaGetDeviceCount (&numphysicaldevices) || "cudaGetDeviceCount failed";
    fprintf (stderr, "lazyinit: %d physical CUDA devices detected\n", numphysicaldevices);
#ifndef NOMULTIDEVICE
       // we can emulate a larger number of GPUs than actually present, for dev purposes
    int oversubscribe = 1;
    const char * oversubscribevar = getenv ("DBNOVERSUBSCRIBEGPUS");
    if (oversubscribevar)
        oversubscribe = atoi (oversubscribevar);
       const int numdevices = numphysicaldevices * oversubscribe;
    // number of devices
    // environment variable DBNMAXGPUS
    //  - 0: use all, exclusively
    //  - >0: limit to this number, exclusively  --default is 1
       //        The number of available devices includes the emulated one by means of DBNOVERSUBSCRIBEGPUS
    //  - <0: use this number but bypassing the exclusive check, for debugging/quick stuff
    int devlimit = 1;
    bool exclusive = true;
    const char * devlimitvar = getenv ("DBNMAXGPUS");
    if (devlimitvar)
        devlimit = atoi (devlimitvar);
    if (devlimit < 0)
    {
        devlimit = -devlimit;
        exclusive = false; // allow non-exclusive use
    }
    if (devlimit == 0)
        devlimit = INT_MAX;
    // initialize CUDA device API
    cuInit (0) || "cuInit failed";
    // initialize the system
    devicesallocated = 0;
    for (int deviceid = 0; deviceid < numdevices && devicesallocated < devlimit; deviceid++)    // loop over all physical devices
    {
        // check if device is available by trying to lock it
        bool available = !exclusive || lockdevicebymutex (deviceid); // try to acquire the lock
        
        if (!available)           // not available: don't use it
        {
            fprintf (stderr, "CUDA device %d already in use, skipping\n", deviceid);
            continue;
        }
        // OK to allocate
              const int physicaldeviceid = deviceid % numphysicaldevices;   // not the same in case of DBNOVERSUBSCRIBEGPUS > 1
        cudadevicecontexts[devicesallocated].init (physicaldeviceid);
        devicesallocated++;
    }
    fprintf (stderr, "using %d on %d physically present CUDA devices%s\n", devicesallocated, numphysicaldevices, exclusive ? " exclusively" : "");
#else
    devicesallocated = 1;
#endif
#endif 
}

void initwithdeviceid(size_t deviceid)
{
    if (devicesallocated >= 0) return;
    devicesallocated = 0;
    cudadevicecontexts[devicesallocated].init(deviceid);
    devicesallocated++;
    fprintf(stderr, "using  CUDA devices%d \n", deviceid);
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
    if (deviceid >= _countof (cudadevicecontexts))
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
