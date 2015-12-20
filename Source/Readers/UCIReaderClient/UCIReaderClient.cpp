//
// <copyright file="UCIReaderClient.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// DataReaderClient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "DataReader.h"
using namespace std;
using namespace Microsoft::MSR::CNTK;
#include <chrono>


void TestDataReader(size_t vdim, size_t udim, const std::vector<std::wstring>& filepaths, const ConfigParameters& config, size_t mbSize, int epochStart, size_t epochSize, size_t datasetSize, int iterations)
{
    fprintf(stderr,"\n####### Start Test  #######\n");
    fprintf(stderr, "vdim=%ld, udim=%ld, ", vdim, udim);
    for (std::vector<std::wstring>::const_iterator iter=filepaths.begin(); iter < filepaths.end(); ++iter) 
        fprintf(stderr, "path=%ls\n", iter->c_str());
    config.dump();
    ConfigParameters configReader = config("reader");
    fprintf(stderr, "randomize=%ld, ", (size_t)configReader("randomize",randomizeAuto));

    if (epochSize == requestDataSize)
    {
        fprintf(stderr, "epochSize=requestDataSize\n");
    }
    else
    {
        fprintf(stderr, "epochSize=%ld\n", epochSize);
    }
    fprintf(stderr, "mbSize=%ld, epochStart=%d, iterations=%d, datasetSize=%ld\n", mbSize, epochStart, iterations, datasetSize);
    DataReader<float> dataReader(vdim, udim, filepaths, config); 
    Matrix<float> features;
    Matrix<float> labels;
    auto start = std::chrono::system_clock::now();
    int epochs = (datasetSize+epochSize-1)/epochSize;
    epochs *= iterations;
    for (int epoch = epochStart; epoch < epochs; epoch++)
    {
        dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);
        int i = 0;
        while (dataReader.GetMinibatch(features, labels))
        {
            if (labels.GetNumRows() == 0)
            {
                fprintf(stderr,"%4d: features dim: %d x %d - [%.8g, %.8g, ...]\n", i++, features.GetNumRows(), features.GetNumCols(), features(0,0), features(0,1));
            }
            else
            {
                fprintf(stderr,"%4d: features dim: %d x %d - [%.8g, %.8g, ...] label dim: %d x %d - [%d, %d, ...]\n", i++, features.GetNumRows(), features.GetNumCols(), features(0,0), features(0,1), labels.GetNumRows(), labels.GetNumCols(), (int)labels(0,0), (int)labels(0,1));
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = end-start;
    fprintf(stderr, "%f seconds elapsed", (float)(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count())/1000);
    fprintf(stderr,"\n####### End Test #######\n\n");

}

int _tmain(int argc, _TCHAR* argv[])
{
    size_t vdim = 784;
    size_t udim = 10;
    size_t mbSizes[] = {50,100,128};
    size_t randomizeSizes[] = {randomizeNone, 1000, 5000, 10000, randomizeAuto};
    size_t epochSizes[] = {100, 1000, 1024, 10000, requestDataSize};
    size_t datasetSize = 10000;
    vector<wstring> filepaths;
    filepaths.push_back( wstring(L"C:\\speech\\mnist\\mnist_test.txt"));
    ConfigParameters config("reader=[labelPosition=First;labelType=Category;minibatchMode=Partial;randomize=None");
    //wstring options(L"-label:first -labeltype:category -minibatchmode:partial"); //  -labeltype:none  -labeltype:regression -labeltype:regression -minibatchmode:partial

    // test set, for debugging, current debug set
    size_t mbSize = 50;
    size_t randomizeSize = randomizeNone;
    size_t epochSize = 1024;
    size_t epochStart = 0;
    TestDataReader(vdim, udim, filepaths, config, mbSize, epochStart, epochSize, datasetSize, 3);

    for (int epochStart = 0; epochStart <= 1; epochStart++)
    {
        for (int epochCnt=0; epochCnt < sizeof(epochSizes)/sizeof(size_t);epochCnt++)
        {
            size_t epochSize = epochSizes[epochCnt];
            for (int mbCnt=0; mbCnt < sizeof(mbSizes)/sizeof(size_t);mbCnt++)
            {
                size_t mbSize = mbSizes[mbCnt];
                if (mbSize > epochSize)
                    continue;
                for (int rndCnt=0; rndCnt < sizeof(randomizeSizes)/sizeof(size_t);rndCnt++)
                {
                    size_t randomizeSize = randomizeSizes[rndCnt];
                    if (randomizeSize != randomizeNone && 
                        (epochSize % randomizeSize != 0 || randomizeSize % mbSize != 0 ||
                        // too many constraints on randomizeSize fix sometime
                        randomizeSize % datasetSize == 0 && epochSize != requestDataSize))
                        continue;    // invalid randomization size, continue
                    string configString = "reader=[labelPosition=First;labelType=Category;minibatchMode=Partial;randomize=";
                    char buf[30];
                    configString += _ui64toa_s(randomizeSize, buf, 30, 10);
                    ConfigParameters config(configString);
                    TestDataReader(vdim, udim, filepaths, config, mbSize, epochStart, epochSize, datasetSize, 3);
                }
            }
        }
    }
    return 0;
}
