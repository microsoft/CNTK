//
// <copyright file="Sample.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// Sample.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <assert.h>
#include "..\Math\CPUMatrix.h"
using namespace Microsoft::MSR::CNTK;


int _tmain(int argc, _TCHAR* argv[])
{
    CPUMatrix<float> M0(2,3);
    M0(0,0) = 1; M0(0,1) = 2; M0(0,2) = 3;
    M0(1,0) = 3; M0(1,1) = 2; M0(1,2) = 1;

    CPUMatrix<float> M1(2,3);
    M0(0,0) = 11; M0(0,1) = 12; M0(0,2) = 13;
    M0(1,0) = 13; M0(1,1) = 12; M0(1,2) = 11;

    CPUMatrix<float> M2 = M1+10;
    M2.GetNumCols();
}

