using CNTK;
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
            var device = DeviceDescriptor.CPUDevice;
            Console.WriteLine($"======== runing MNISTClassifierTest.TrainAndEvaluate using {device.Type} with logistic classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, false, true);

            Console.WriteLine($"======== runing MNISTClassifierTest.TrainAndEvaluate using {device.Type} with convolution classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, true, true);

            // batch normalization is not available on CPU build. These examples are in the GPU project.
            //Console.WriteLine("======== runing CifarResNet.TrainAndEvaluate using CPU ========");
            //CifarResNetClassifier.TrainAndEvaluate(device, true);

            //Console.WriteLine("======== runing TransferLearning.TrainAndEvaluateWithFlowerData using CPU ========");
            //TransferLearning.TrainAndEvaluateWithFlowerData(device, true);

            //Console.WriteLine("======== runing TransferLearning.TrainAndEvaluateWithAnimalData using CPU ========");
            //TransferLearning.TrainAndEvaluateWithAnimalData(device, true);

            Console.WriteLine($"======== runing LSTMSequenceClassifier.Train using {device.Type} ========");
            LSTMSequenceClassifier.Train(device, true);
        }
    }
}
