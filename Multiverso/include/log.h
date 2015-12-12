#ifndef _MULTIVERSO_LOG_H_
#define _MULTIVERSO_LOG_H_
 
#include <cstdio>
#include <string>
#include <cstdarg>
using namespace std;

namespace multiverso
{
    class Log
    {
    public:
		static void EnableLog(bool enable)
		{
			enable_ = enable;
		}

        static void WriteLine(string msg)
        {
			if (enable_)
			{
				printf("%s\n", msg.c_str());
				fflush(stdout);
			}
        }

        static int Printf(const char *format, ...)
        {
			if (enable_)
			{
				va_list ap;
				va_start(ap, format);
				int ret = vprintf(format, ap);
				va_end(ap);
				fflush(stdout);
				return ret;
			}
			else
				return 0;
        }

        static void FatalError(const char *format, ...)
        {
            va_list ap;
            va_start(ap, format);
            vprintf(format, ap);
            va_end(ap);
            fflush(stdout);
            exit(1);
        }

	private:
		static bool enable_;
    };

	
}
#endif // _MULTIVERSO_LOG_H_