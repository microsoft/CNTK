using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKLibraryCSTrainingExamplesTest
{
    class Program
    {
        static void Main(string[] args)
        {
            // Todo: move to a separate unit test.
            Console.WriteLine("Test CNTKLibraryCSTrainingExamples");

            int maxThreads = Utils.GetMaxNumCPUThreads();
            Utils.SetMaxNumCPUThreads(2);
            Console.WriteLine("MaxNumCPUThreads: before: " + maxThreads + ", after " + Utils.GetMaxNumCPUThreads());
            Utils.SetMaxNumCPUThreads(maxThreads);
            Console.WriteLine("reset MaxNumCPuThreads to " + Utils.GetMaxNumCPUThreads());

            var level = Utils.GetTraceLevel();
            Utils.SetTraceLevel(TraceLevel.Info);
            Console.WriteLine("TraceLevel: before: " + level + ", after " + Utils.GetTraceLevel());
            Utils.SetTraceLevel(level);
            Console.WriteLine("reset TraceLevel to " + Utils.GetTraceLevel());

            Console.WriteLine(Utils.DataTypeName(DataType.Float));
            Console.WriteLine(Utils.DataTypeSize(DataType.Double));
            Console.WriteLine(Utils.DeviceKindName(DeviceDescriptor.CPUDevice.Type));
            Console.WriteLine(Utils.DeviceKindName(DeviceKind.GPU));
            Console.WriteLine(Utils.IsSparseStorageFormat(StorageFormat.Dense));
            Console.WriteLine(Utils.IsSparseStorageFormat(StorageFormat.SparseCSC));
            Console.WriteLine(Utils.IsSparseStorageFormat(StorageFormat.SparseBlockCol));
            Console.WriteLine(Utils.VariableKindName(VariableKind.Constant));
            Console.WriteLine(Utils.VariableKindName(VariableKind.Placeholder));
            Console.WriteLine(Utils.VariableKindName(VariableKind.Input));
            Console.WriteLine(Utils.VariableKindName(VariableKind.Output));
            Console.WriteLine(Utils.VariableKindName(VariableKind.Parameter));

#if CPUONLY
            Console.WriteLine("======== Evaluate model on CPU using CPUOnly build ========");
#else
            Console.WriteLine("======== Evaluate model on CPU using GPU build ========");
#endif

            if (ShouldRunOnCpu())
            {
                var device = DeviceDescriptor.CPUDevice;

                SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);
            }

            if (ShouldRunOnGpu())
            {
                Console.WriteLine(" ====== Evaluate model on GPU =====");
                var device = DeviceDescriptor.GPUDevice(0);

                SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);
            }

            Console.WriteLine("======== Evaluation completes. ========");
        }

        static bool ShouldRunOnGpu()
        {
#if CPUONLY
            return false;
#else
            string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

            return (string.IsNullOrEmpty(testDeviceSetting) || string.Equals(testDeviceSetting.ToLower(), "gpu"));
#endif
        }

        static bool ShouldRunOnCpu()
        {
            string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

            return (string.IsNullOrEmpty(testDeviceSetting) || string.Equals(testDeviceSetting.ToLower(), "cpu"));
        }
    }
}
