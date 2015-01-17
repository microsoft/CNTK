//
// <copyright file="DataReaderUnitTest.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include "DataReader.h"
using namespace std;
using namespace Microsoft::MSR::CNTK;

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace DataReaderTest
{        
    TEST_CLASS(UCIDataReaderUnitTest)
    {
    public:
        
        // StandardLoopTest
        // Test of the DataReader loop 
        TEST_METHOD(TestMode)
        {
            size_t vdim = 785;
            size_t udim = 10;
            size_t epochSize = 500;
            size_t mbSize = 256;
            size_t epochs = 2;
            vector<wstring> filepaths;
            filepaths.push_back( wstring(L"C:\\speech\\mnist\\mnist_test.txt"));

            DataReader<float, int> dataReader(vdim, udim, filepaths, wstring(L"-label:none -minibatchmode:partial "), randomizeNone); //-labels:regression
            Matrix<float> features;
            Matrix<float> labels;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);
                
                for (int cnt = 0;dataReader.GetMinibatch(features, labels);cnt++)
                {
                    Assert::IsTrue(labels.GetNumRows() == 0);
                    Assert::IsTrue(features.GetNumRows() == 785);
                    Assert::IsTrue(features.GetNumCols() == (cnt?244:mbSize));
                    for (int i = 1;i < features.GetNumCols();i++)
                    {
                        // really labels, these should be in order
                        Assert::IsTrue(features(0,i-1) <= features(0,i));
                    }
                }
            }
        }

        TEST_METHOD(Partial)
        {
            size_t vdim = 784;
            size_t udim = 10;
            size_t epochSize = 500;
            size_t mbSize = 256;
            size_t epochs = 2;
            vector<wstring> filepaths;
            filepaths.push_back( wstring(L"C:\\speech\\mnist\\mnist_test.txt"));

            DataReader<float, int> dataReader(vdim, udim, filepaths, wstring(L"-label:first -labeltype:category -minibatchmode:partial "), randomizeNone); //-labels:regression
            Matrix<float> features;
            Matrix<float> labels;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);
                
                for (int cnt = 0;dataReader.GetMinibatch(features, labels);cnt++)
                {
                    Assert::IsTrue(labels.GetNumRows() == udim);
                    Assert::IsTrue(features.GetNumRows() == 785);
                    Assert::IsTrue(features.GetNumCols() == (cnt?244:mbSize));
                    for (int i = 1;i < features.GetNumCols();i++)
                    {
                        // really labels, these should be in order
                        Assert::IsTrue(features(0,i-1) <= features(0,i));
                    }
                }
            }
        }
    };
}