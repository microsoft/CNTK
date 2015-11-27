// .cu file --#includes all actual .cu files which we store as .cu.h so we get syntax highlighting (VS does not recognize .cu files)
//
// F. Seide, V-hansu

#include <stdexcept>
#include "Basics.h"
#include "BestGpu.h"

#ifndef CPUONLY

namespace msra { namespace cuda {

    // call this after all kernel launches
    // This is non-blocking. It catches launch failures, but not crashes during execution.
    static void checklaunch (const char * fn)
    {
        cudaError_t rc = cudaGetLastError();
        if (rc != cudaSuccess)
            RuntimeError("%s: launch failure: %s (cuda error %d)", fn, cudaGetErrorString(rc), (int)rc);
    }

};};

// now include actual code which is in those files to allow for code highlighting etc.
#include "cudalatticeops.cu.h"

#endif // CPUONLY
