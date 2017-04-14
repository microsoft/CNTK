#include "../CPP/UserMatrixMultiplicationOp.h"

using namespace CNTK;

extern "C" 
#ifdef _WIN32
__declspec (dllexport)
#endif
Function* CreateUserTimesFunction(const Variable* operands, size_t /*numOperands*/, const Dictionary* /*attributes*/, const wchar_t* name)
{
    return new UserTimesFunction(operands[0], operands[1], name);
}
