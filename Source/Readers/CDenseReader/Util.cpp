#include "Util.h"
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

using namespace std;

string FormatTime(time_t tm) {
	char buffer[9] = { 0 };
	strftime(buffer, 9, "%H:%M:%S", localtime(&tm));
	return buffer;
}

void PrintTime(std::function<void()> fn, const std::string& tag) {
	auto start = time(0);
	cerr << tag << " started at " << FormatTime(start) << endl;

	fn();

	auto end = time(0);
	cerr << tag << " finished at " << FormatTime(end) << ",in " << end - start << "s" << endl;
}


#ifdef _WIN32

float GetTime() {
	LARGE_INTEGER time;
	QueryPerformanceFrequency(&time);
	LONGLONG freq = time.QuadPart;
	QueryPerformanceCounter(&time);
	return time.QuadPart * 1000.0f / freq; //millisec
}

#else

float GetTime() {
	timeval time;
	gettimeofday(&time, NULL);
	return time.tv_sec * 1000.0f + time.tv_usec / 1000.0f; //millisec
}

#endif
