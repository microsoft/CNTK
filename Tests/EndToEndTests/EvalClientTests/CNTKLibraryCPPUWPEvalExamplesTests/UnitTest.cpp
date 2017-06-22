#include "pch.h"
#include "CppUnitTest.h"
#include "CNTKLibrary.h"

using namespace CNTK;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Windows::Storage;

void MultiThreadsEvaluation(const wchar_t* modelFileName, bool);

namespace UWPEvalTests
{
    TEST_CLASS(TestClass)
    {
    private:
        concurrency::task<Platform::String^> GetModelFilePath(Platform::String^ modelFileName)
        {
            auto modelFileOp = KnownFolders::DocumentsLibrary->GetFileAsync(modelFileName);
            return concurrency::create_task(modelFileOp).then([modelFileName](concurrency::task<StorageFile^> modelFileTask) {
                try
                {
                    // The file cannot be read directly from the DocumentsLibrary, so copy the file into the local app folder
                    auto modelFile = modelFileTask.get();
                    auto localFolder = Windows::Storage::ApplicationData::Current->LocalFolder;
                    auto copyTask = modelFile->CopyAsync(localFolder, modelFileName, NameCollisionOption::ReplaceExisting);
                    return concurrency::create_task(copyTask).then([](concurrency::task<StorageFile^> modelFileTask2) {
                        auto path = modelFileTask2.get()->Path;
                        return path;
                    });
                }
                catch (...)
                {
                    Assert::Fail(L"File 01_OneHidden.model must exist in the Documents directory");
                    throw;
                }
            });
        }

        template<typename Func>
        void RunTestWithModel(const wchar_t* modelFile, const Func& func)
        {
            Platform::String^ modelFileName = ref new Platform::String(modelFile);
            try {
                concurrency::create_task(GetModelFilePath(modelFileName)).then([&func](concurrency::task<Platform::String^> modelFilePath) {
                    auto path = modelFilePath.get();
                    func(path->Data());
                }).get();
            }
            catch (...) {
                Assert::Fail(L"Exception while test execution");
                throw;
            }
        }

    public:
        TEST_METHOD(TestSanity)
        {
            // Failure in this test indicates a problem with infrastructure
            Assert::IsTrue(true);
        }

        TEST_METHOD(TestModelLoad)
        {
            RunTestWithModel(L"01_OneHidden.model", [](auto path) {
                auto device = DeviceDescriptor::CPUDevice();
                CNTK::Function::Load(path, device);
            });
        }

        TEST_METHOD(TestEval)
        {
            RunTestWithModel(L"01_OneHidden.model", [](auto path) {
                MultiThreadsEvaluation(path, false);
            });
        }
    };
}