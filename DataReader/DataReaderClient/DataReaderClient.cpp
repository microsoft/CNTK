//
// <copyright file="DataReaderClient.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// DataReaderClient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "DataReader.h"
using namespace std;
using namespace Microsoft::MSR::CNTK;

int _tmain(int argc, _TCHAR* argv[])
{
    size_t vdim = 429;
    size_t udim = 1504;
    vector<wstring> filepaths;
    filepaths.push_back( wstring(L"C:\\speech\\swb300h\\data\\archive.swb_mini.52_39.notestspk.dev.small.scplocal"));
    filepaths.push_back( wstring(L"C:\\speech\\swb300h\\data\\swb_mini.1504.align.small.statemlf"));

    DataReader<float> dataReader(vdim, udim, filepaths, wstring(L""), 4096);
    Matrix<float> features;
    Matrix<float> labels;
    dataReader.StartMinibatchLoop(256, 0);
    int i = 0;
    while (dataReader.GetMinibatch(features, labels))
    {
        fprintf(stderr,"%4d: features dim: %d x %d - [%.8g, %.8g, ...] label dim: %d x %d - [%d, %d, ...]\n", i++, features.GetNumRows(), features.GetNumCols(), features(0,0), features(0,1), labels.GetNumRows(), labels.GetNumCols(), (int)labels(0,0), (int)labels(1,0));
    }
    return 0;
}

