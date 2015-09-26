// .cu file --#includes all actual .cu files which we store as .cu.h so we get syntax highlighting (VS does not recognize .cu files)
//
// F. Seide, V-hansu

#include <stdexcept>

namespace msra { namespace cuda {

    // call this after all kernel launches
    // This is non-blocking. It catches launch failures, but not crashes during execution.
    static void checklaunch (const char * fn)
    {
        cudaError_t rc = cudaGetLastError();
        if (rc != cudaSuccess)
        {
            char buf[1000];
            sprintf(buf, "%s: launch failure: %s (cuda error %d)", fn, cudaGetErrorString (rc), rc);
            throw std::runtime_error (buf);
        }
    }

};};

// now include actual code which is in those files to allow for code highlighting etc.
#include "cudalatticeops.cu.h"
