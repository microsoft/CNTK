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
            TestCommon.TestDataDirPrefix = "../../";
            var device = DeviceDescriptor.CPUDevice;
            Console.WriteLine($"======== running LogisticRegression.TrainAndEvaluate using {device.Type} ========");
            LogisticRegression.TrainAndEvaluate(device);

            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with multilayer perceptron (MLP) classifier using {device.Type} ========");
            MNISTClassifier.TrainAndEvaluate(device, false, true);

            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with convolutional neural network using {device.Type} ========");
            MNISTClassifier.TrainAndEvaluate(device, true, true);

            // Batch normalization is not available on CPU build. See following examples in GPU project.
            // Following examples will be enabled once BN is supported on CPU.
            //Console.WriteLine("======== running CifarResNet.TrainAndEvaluate using CPU ========");
            //CifarResNetClassifier.TrainAndEvaluate(device, true);

            //Console.WriteLine("======== running TransferLearning.TrainAndEvaluateWithFlowerData using CPU ========");
            //TransferLearning.TrainAndEvaluateWithFlowerData(device, true);

            //Console.WriteLine("======== running TransferLearning.TrainAndEvaluateWithAnimalData using CPU ========");
            //TransferLearning.TrainAndEvaluateWithAnimalData(device, true);

            Console.WriteLine($"======== running LSTMSequenceClassifier.Train using {device.Type} ========");
            LSTMSequenceClassifier.Train(device);
        }
    }
}
