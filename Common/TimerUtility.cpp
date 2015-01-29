#include "TimerUtility.h"

#ifdef WIN32
#include <Windows.h>
#else
#include <time.h>
#endif
namespace Microsoft{
    namespace MSR {
        namespace CNTK {

            //Returns the amount of milliseconds elapsed
            unsigned long long Timer::MilliSecondElapsed()
            {
#ifdef WIN32
                FILETIME ft;
                LARGE_INTEGER li;

                GetSystemTimeAsFileTime(&ft);
                li.LowPart = ft.dwLowDateTime;
                li.HighPart = ft.dwHighDateTime;

                unsigned long long ret = li.QuadPart;
                ret -= 116444736000000000LL; // Convert from file time to UNIX epoch time. 
                ret /= 10000; // From 100 nano seconds (10^-7) to 1 millisecond (10^-3) 

                return ret;
#else
                timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts); // Works on Linux

                unsigned long long ret = ts.tv_sec * 1000 + ts.tv_nsec/1000000;

                return ret;
#endif
            }
        }
    }
}
