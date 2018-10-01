using System;
using System.Collections.Generic;
using System.IO;
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
            var device = DeviceDescriptor.GPUDevice(0);
            Console.WriteLine($"======== running LogisticRegression.TrainAndEvaluate using {device.Type} ========");
            LogisticRegression.TrainAndEvaluate(device);

            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with multilayer perceptron (MLP) classifier using {device.Type} ========");
            MNISTClassifier.TrainAndEvaluate(device, false, true);

            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with convolutional neural network using {device.Type} ========");
            MNISTClassifier.TrainAndEvaluate(device, true, true);

            Console.WriteLine($"======== running CifarResNet.TrainAndEvaluate using {device.Type} ========");
            CifarResNetClassifier.CifarDataFolder = "../../Examples/Image/DataSets/CIFAR-10";
            CifarResNetClassifier.TrainAndEvaluate(device, true);
            TestCommon.TestDataDirPrefix = "../../Examples/Image/DataSets/";
            string modelFileSourceDir = "../../PretrainedModels/ResNet_18.model";
            if (!File.Exists(modelFileSourceDir))
            {
                Console.WriteLine("Model file doesn't exist. Please run download_model.py in CNTK/CNTK/PretrainedModels");
                Console.ReadKey();
                return;
            }

            TransferLearning.BaseResnetModelFile = "ResNet_18.model";
            File.Copy(modelFileSourceDir, TransferLearning.ExampleImageFolder + TransferLearning.BaseResnetModelFile,/*overwrite*/true);
            Console.WriteLine($"======== running TransferLearning.TrainAndEvaluateWithAnimalData using {device.Type} ========");
            TransferLearning.TrainAndEvaluateWithAnimalData(device, true);

            TestCommon.TestDataDirPrefix = "../../";
            Console.WriteLine($"======== running LSTMSequenceClassifier.Train using {device.Type} ========");
            LSTMSequenceClassifier.Train(device);
        }
    }
}
