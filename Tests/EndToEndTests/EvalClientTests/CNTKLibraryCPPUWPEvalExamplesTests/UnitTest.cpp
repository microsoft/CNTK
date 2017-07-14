#include "pch.h"
#include "CppUnitTest.h"
#include "CNTKLibrary.h"
using namespace CNTK;

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

void MultiThreadsEvaluation(bool);

namespace UWPEvalTests
{
    TEST_CLASS(TestClass)
    {
    public:
        TEST_METHOD(TestSanity)
        {
            // Failure in this test indicates a problem with infrastructure
            Assert::IsTrue(true);
        }

        TEST_METHOD(TestModelLoad)
        {
            auto device = DeviceDescriptor::CPUDevice();
            try
            {
                CNTK::Function::Load(L"01_OneHidden", device);
            }
            catch (...)
            {
                // Note: It uses the model trained by Examples\Image\GettingStarted\01_OneHidden.cntk as example. Instructions
                // to train the model is described in Examples\Image\GettingStarted\README.md. 
                Assert::Fail(L"File 01_OneHidden must exist in the APPX directory");
            }
        }

        TEST_METHOD(TestEval)
        {
            MultiThreadsEvaluation(false);
        }
    };
}