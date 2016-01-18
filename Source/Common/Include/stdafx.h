// dummy stdafx.h file for Linux version

// In the Windows build, there are several stdafx.h files which are used for project-specific precompilation of headers.
// In Linux, this is not used. By placing this dummy file ahead of all other stdafx.h in the include search path,
// we make sure it compiles while not getting confused by Windows-specific content of those files.
#include "Platform.h"
