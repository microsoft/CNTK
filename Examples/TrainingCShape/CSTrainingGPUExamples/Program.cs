using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CSTrainingExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            var device = DeviceDescriptor.GPUDevice(0);
            Console.WriteLine($"======== runing MNISTClassifierTest.TrainAndEvaluate using {device.Type} with logistic classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, false, true);

            Console.WriteLine($"======== runing MNISTClassifierTest.TrainAndEvaluate using {device.Type} with convolution classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, true, true);

            Console.WriteLine($"======== runing CifarResNet.TrainAndEvaluate using {device.Type} ========");
            CifarResNetClassifier.TrainAndEvaluate(device, true);

            Console.WriteLine($"======== runing TransferLearning.TrainAndEvaluateWithFlowerData using {device.Type} ========");
            TransferLearning.TrainAndEvaluateWithFlowerData(device, true);

            Console.WriteLine($"======== runing TransferLearning.TrainAndEvaluateWithAnimalData using {device.Type} ========");
            TransferLearning.TrainAndEvaluateWithAnimalData(device, true);

            Console.WriteLine($"======== runing LSTMSequenceClassifier.Train using {device.Type} ========");
            LSTMSequenceClassifier.Train(device, true);
        }
    }
}
