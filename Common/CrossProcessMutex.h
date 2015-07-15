#pragma once

#ifdef WIN32
#include "CrossProcessMutex_win32.h"
#else
#include "CrossProcessMutex_linux.h"
#endif