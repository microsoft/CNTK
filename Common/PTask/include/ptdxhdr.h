///-------------------------------------------------------------------------------------------------
// file:	ptdxhdr.h
//
// summary:	include DirectX headers required for given build environment
///-------------------------------------------------------------------------------------------------
#pragma once

#if (_MSC_VER > 1600)
// apparently d3dx11.h is obsolete in win8
#include <d3dcommon.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#else
#include <d3dcommon.h>
#include <d3d11.h>
#ifdef DIRECTXCOMPILERSUPPORT
#include <d3dcompiler.h>
#include <d3dx11.h>
#endif
#endif



