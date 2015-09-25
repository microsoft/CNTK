// cudalib.h -- wrappers for CUDA to hide NVidia stuff
//
// F. Seide, V-hansu

#pragma once

#include <cuda_runtime_api.h>
#include "cudabasetypes.h"
#include <vector>
#include <memory>

namespace msra { namespace cuda {

// helper functions
void lazyinit();
size_t numdevices();
void join();            // wait until current launch or other async operation is completed
void initwithdeviceid(size_t deviceid);

// managing device context switching
void setdevicecontext (size_t deviceid); // get a cudadevice; lazy-init if needed
void cleardevicecontext();
CUcontext getdevicecontext (size_t deviceid); // get a context for explicitly passing it to the CUDA API
cudaStream_t GetStream(size_t deviceid);
cudaEvent_t GetEvent(size_t deviceid);
size_t GetCurrentDevice();
cudaStream_t GetCurrentStream();
cudaEvent_t GetCurrentEvent();


// memory allocation and copying
void * mallocbytes (size_t nelem, size_t sz);
void freebytes (void * p);
template<typename T> cuda_ptr<T> malloc (size_t nelem) { return (T*) mallocbytes (nelem, sizeof (T)); }
template<typename T> void free (cuda_ptr<T> p) { if (p.get() != NULL) freebytes ((void*)p.get()); }

void memcpyh2d (void * dst, size_t byteoffset, const void * src, size_t nbytes);
void memcpyd2h (void * dst, const void * src, size_t byteoffset, size_t nbytes);
template<typename T> void memcpy (cuda_ptr<T> dst, size_t dstoffset, const T * src, size_t nelem)
{
    memcpyh2d ((void*) dst.get(), dstoffset * sizeof (T), (const void *) src, nelem * sizeof (T));
}
template<typename T> void memcpy (T * dst, const cuda_ptr<T> src, size_t srcoffset, size_t nelem)
{
    memcpyd2h ((void*) dst, (const void *) src.get(), srcoffset * sizeof (T), nelem * sizeof (T));
}
// [v-hansu] for debug use, change false to true to activate
template<typename T> void peek (vectorref<T> v)
{
    bool dopeek = false;
    if (dopeek)
    {
        std::vector<T> vp(v.size());
        memcpy (&vp[0], v.get(), 0, v.size());
        T vp0 = vp[0];  // for reference
        vp[0] = vp0;
    }
}
template<typename T> void peek (matrixref<T> v)
{
    bool dopeek = false;
    if (dopeek)
    {
        std::vector<T> vp(v.cols() * v.rows());
        memcpy (&vp[0], v.get(), 0, v.cols() * v.rows());
        T vp0 = vp[0];  // for reference
        vp[0] = vp0;
    }
}

// memory allocation and copying for textures
// TODO: continue here once I figured out the global-definition-of-texture problem


#if 0
// thread-serialized, de-templated operation for vector below
class cudadevicecontext
{
public:
    class serializedop
    {
    public:
        std::string error;  // empty: no error
        void go() throw() { try { op(); } catch (const std::exception & e) { error = e.what(); } }
        virtual void op() throw() = 0;  // function that executes it to be provided as an overload
    };
    class noop : public serializedop { public: void op() { } };
private:
    struct cudathread { cudathread (size_t deviceid); ~cudathread(); };
    struct cublasthread : cudathread { cublasthread (size_t deviceid); ~cublasthread(); };
    class cudadevicethread
    {
        std::auto_ptr<cublasthread> ourcublasthread;
        size_t deviceid;                            // our device
        serializedop * currentop;                   // NULL while idle
        // this executes one command at a time
    public:
        // call this before execute(); can be called multiple times safely, i.e. call it always
        void lazyconstruct (size_t deviceid)
        {
            if (ourcublasthread.get() != NULL)
                return;     // already initialized
            // construct
            this->deviceid = deviceid;              // our device
            ourcublasthread.reset (new cublasthread (deviceid));    // initialize cublas
            currentop = NULL;
        }
        void execute (serializedop & op)
        {
            // TODO: execute on background thread
            // TODO: change to lazy mode --start but don't wait. Safe except for copy-back function.
            // interlocked-exchange op into thread->currentop, signal, then wait for completion (goes back to NULL)
            op.go();
            if (!op.error.empty())
                throw std::runtime_error (op.error);
        }
    };
public:
    // call this only from main thread (not multi-threaded)
    static void executeondevice (size_t deviceid, serializedop & op)
    {
        static cudadevicethread threads[32];    // the threads, if they exist
        if (deviceid >= _countof (threads))
            throw std::logic_error ("executeondevice: fixed-size array threads[] too small");
        threads[deviceid].lazyconstruct (deviceid);
        threads[deviceid].execute (op);
    }
    static void lazyinit (size_t deviceid) { executeondevice (deviceid, cudadevicecontext::noop()); }
};
#endif

#if 0
// a vector that lives in CUDA RAM
// Maybe this whole thing is not needed if we move this to the matrix class directly.
template<typename T> class vector : public vectorref<T>
{
    size_t deviceid;    // the device this lives in
    vector (const vector &); void operator= (const vector &);       // non-copyable

    // class to execute vector ops on device-specific threads  --meh, CUDA is lame
    template<typename T> class vectorop : public cudadevicecontext::serializedop
    {
        vectorref<T> & vec;
        bool h2d;
        size_t nelem;
        size_t firstelem;
        T * hostbegin;
        size_t ncopy;
        vectorop(); vectorop (const vectorop &); void operator= (const vectorop &);
    public:
        vectorop (vectorref<T> * vec, bool h2d, size_t nelem, size_t firstelem, T * hostbegin, size_t ncopy)
            : vec (*vec), h2d (h2d), nelem (nelem), firstelem (firstelem), hostbegin (hostbegin), ncopy (ncopy) {}

        // this function is supposed to be run on the respective device thread
        void op()
        {
            if (firstelem + ncopy > vec.size()) throw std::runtime_error ("vectorop: trying to copy vector exceeding bounds");
            if (h2d)
            {
                // allocate device memory if not the right allocation size
                if (vec.size() != nelem)
                {
                    free (vec.get());
                    vec.reset (malloc<T> (nelem), nelem);
                }
                // copy data over if requested
                if (ncopy != 0)
                    memcpy (vec.get(), firstelem, hostbegin, ncopy);
            }
            else    // d2h: can only copy (no allocation in sub-thread, done on calling thread)
                memcpy (hostbegin, vec.get(), firstelem, ncopy);
        }
    };
public:
    vector(): deviceid (0) {}
    ~vector() { clear(); }
    void setdeviceid (size_t deviceid) { if (!empty()) throw std::logic_error ("setdeviceid failed: vector not empty"); this->deviceid = deviceid; }
    // assign full vector (with CUDA allocation)
    void assign (const std::vector<T> & other)
    {
        cudadevicecontext::executeondevice (deviceid, vectorop<T> (this, true, other.size(), 0, const_cast<T*> (&other[0]), other.size()));
    }
    void allocate (size_t sz)
    {
        cudadevicecontext::executeondevice (deviceid, vectorop<T> (this, true, sz, 0, NULL, 0));
    }
    void clear() throw() { allocate (0); }
    bool empty() const throw() { return size() == 0; }
    // make a full copy back (with local allocation)
    operator std::vector<T> ()
    {
        std::vector<T> res (size());
        get (res.begin(), res.end(), 0);
        return res;
    }
    // no allocation
    void put (typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, size_t offset) throw();
    // no allocation
    void get (typename std::vector<T>::iterator begin,       typename std::vector<T>::iterator end,       size_t offset) const throw()
    {
        if (begin == end) return;
        cudadevicecontext::executeondevice (deviceid, vectorop<T> (const_cast<vector*> (this), false, size(), 0, &*begin, end-begin));
    }
};
#endif

};};
