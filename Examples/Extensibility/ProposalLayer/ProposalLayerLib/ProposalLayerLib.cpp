#include "ProposalLayerLib.h"

using namespace CNTK;

extern "C"
#ifdef _WIN32
__declspec (dllexport)
#endif
Function* CreateProposalLayer(const Variable* operands, size_t /*numOperands*/, const Dictionary* attributes, const wchar_t* name)
{
    return new ProposalLayer({operands[0], operands[1], operands[2]}, *attributes, name);
}

template<> FixedSizePoolStorage<sizeof (FixedSizePoolItem<OptionalString::SharableString const>)> strong_shared_ptr<OptionalString::SharableString const>::Storage::s_storage;
