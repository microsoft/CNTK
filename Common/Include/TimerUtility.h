#pragma once

#define MS_PER_SEC 1000

namespace Microsoft{namespace MSR {namespace CNTK {
    class Timer
    {
    public:
        Timer(){};
        ~Timer(){};
        static unsigned long long MilliSecondElapsed();
    };
}}}
