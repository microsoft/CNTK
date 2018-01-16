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
            Console.WriteLine($"======== running LogisticRegression.TrainAndEvaluate using {device.Type} ========");
            LogisticRegression.TrainAndEvaluate(device);

            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with multilayer perceptron (MLP) classifier using {device.Type} ========");
            MNISTClassifier.TrainAndEvaluate(device, false, true);

            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with convolutional neural network using {device.Type} ========");
            MNISTClassifier.TrainAndEvaluate(device, true, true);

            Console.WriteLine($"======== running CifarResNet.TrainAndEvaluate using {device.Type} ========");
            CifarResNetClassifier.TrainAndEvaluate(device, true);

            Console.WriteLine($"======== running TransferLearning.TrainAndEvaluateWithFlowerData using {device.Type} ========");
            TransferLearning.TrainAndEvaluateWithFlowerData(device, true);

            Console.WriteLine($"======== running TransferLearning.TrainAndEvaluateWithAnimalData using {device.Type} ========");
            TransferLearning.TrainAndEvaluateWithAnimalData(device, true);

            Console.WriteLine($"======== running LSTMSequenceClassifier.Train using {device.Type} ========");
            LSTMSequenceClassifier.Train(device);
        }
    }
}
