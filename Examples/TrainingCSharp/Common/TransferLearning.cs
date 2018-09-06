using CNTKImageProcessing;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using CNTK.CNTKLibraryCSTrainingTest;

namespace CNTK.CSTrainingExamples
{
    /// <summary>
    /// This class demonstrates transfer learning use a pretrained ResNet model. 
    /// Refer to https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_301_Image_Recognition_with_Deep_Transfer_Learning.ipynb
    /// for transfer learning in general, and ResNet model, data used for training. 
    /// </summary>
    public class TransferLearning
    {
        public static string CurrentFolder = "./";

        /// <summary>
        /// test execution folder is: CNTK/Tests/EndToEndTests/CNTKv2CSharp/ExampleTests/TransferLearningTest
        /// data folder is: CNTK/Examples/Image
        /// model folder is: CNTK/PretrainedModels
        /// </summary>
        public static string ExampleImageFoler = TestCommon.TestDataDirPrefix;
        public static string BaseResnetModelFile = TestCommon.TestDataDirPrefix + "/ResNet18_ImageNet_CNTK.model";

        private static string featureNodeName = "features";
        private static string lastHiddenNodeName = "z.x";
        private static int[] imageDims = new int[] { 224, 224, 3 };

        /// <summary>
        /// TrainAndEvaluateWithFlowerData shows how to do transfer learning with a MinibatchSource. MinibatchSource is constructed with 
        /// a map file that contains image file paths and labels. Data loading, image preprocessing, and batch randomization are handled 
        /// by MinibatchSource.
        /// </summary>
        /// <param name="device">CPU or GPU device to run</param>
        /// <param name="forceReTrain">Force to train the model if true. If false, 
        /// it only evaluates the model is it exists. </param>
        public static void TrainAndEvaluateWithFlowerData(DeviceDescriptor device, bool forceReTrain = false)
        {
            string flowerFolder = Path.Combine(ExampleImageFoler, "Flowers");
            string flowersTrainingMap = Path.Combine(flowerFolder, "1k_img_map.txt");
            string flowersValidationMap = Path.Combine(flowerFolder, "val_map.txt");
            int flowerModelNumClasses = 102;

            string flowerModelFile = Path.Combine(CurrentFolder, "FlowersTransferLearning.model");

            // If the model exists and it is not set to force retrain, validate the model and return.
            if (File.Exists(flowerModelFile) && !forceReTrain)
            {
                ValidateModelWithMinibatchSource(flowerModelFile, flowersValidationMap,
                    imageDims, flowerModelNumClasses, device);
                return;
            }

            // prepare training data
            MinibatchSource minibatchSource = CreateMinibatchSource(flowersTrainingMap,
                imageDims, flowerModelNumClasses);
            var featureStreamInfo = minibatchSource.StreamInfo("image");
            var labelStreamInfo = minibatchSource.StreamInfo("labels");

            string predictionNodeName = "prediction";
            Variable imageInput, labelInput;
            Function trainingLoss, predictionError;

            // create a transfer model
            Function transferLearningModel = CreateTransferLearningModel(BaseResnetModelFile, featureNodeName,
                predictionNodeName, lastHiddenNodeName, flowerModelNumClasses, device,
                out imageInput, out labelInput, out trainingLoss, out predictionError);

            // prepare for training
            int numMinibatches = 100;
            int minibatchbSize = 50;
            float learningRatePerMinibatch = 0.2F;
            float momentumPerMinibatch = 0.9F;
            float l2RegularizationWeight = 0.05F;

            AdditionalLearningOptions additionalLearningOptions = new AdditionalLearningOptions() { l2RegularizationWeight = l2RegularizationWeight };

            IList<Learner> parameterLearners = new List<Learner>() {
                Learner.MomentumSGDLearner(transferLearningModel.Parameters(),
                new TrainingParameterScheduleDouble(learningRatePerMinibatch, 0),
                new TrainingParameterScheduleDouble(momentumPerMinibatch, 0),
                true,
                additionalLearningOptions)};
            var trainer = Trainer.CreateTrainer(transferLearningModel, trainingLoss, predictionError, parameterLearners);

            // train the model
            int outputFrequencyInMinibatches = 1;
            for (int minibatchCount = 0; minibatchCount < numMinibatches; ++minibatchCount)
            {
                var minibatchData = minibatchSource.GetNextMinibatch((uint)minibatchbSize, device);

                trainer.TrainMinibatch(new Dictionary<Variable, MinibatchData>()
                {
                    { imageInput, minibatchData[featureStreamInfo] },
                    { labelInput, minibatchData[labelStreamInfo] } }, device);
                TestHelper.PrintTrainingProgress(trainer, minibatchCount, outputFrequencyInMinibatches);
            }

            // save the model
            transferLearningModel.Save(flowerModelFile);

            // validate the trained model
            ValidateModelWithMinibatchSource(flowerModelFile, flowersValidationMap,
                imageDims, flowerModelNumClasses, device);
        }

        /// <summary>
        /// TrainAndEvaluateWithAnimalData shows how to do transfer learning without using a MinibatchSource. 
        /// Training and evaluation data are prepared as appropriate in the code. 
        /// Batching is done explicitly in the code as well. 
        /// Because the amount of animal data is limited, it is fine to work this way. 
        /// In real scenarios, it is recommended to code efficiently for data preprocessing and batching, 
        /// probably with parallelization and streaming as what has been done in MinibatchSource.
        /// </summary>
        /// <param name="device">CPU or GPU device to run</param>
        /// <param name="forceRetrain">Force to train the model if true. If false, 
        /// it only evaluates the model is it exists. </param>
        public static void TrainAndEvaluateWithAnimalData(DeviceDescriptor device, bool forceRetrain = false)
        {
            string animalDataFolder = Path.Combine(ExampleImageFoler, "Animals");
            string[] animals = new string[] { "Sheep", "Wolf" };
            int animalModelNumClasses = 2;
            string animalsModelFile = Path.Combine(CurrentFolder, "AnimalsTransferLearning.model");

            // If the model exists and it is not set to force retrain, validate the model and return.
            if (File.Exists(animalsModelFile) && !forceRetrain)
            {
                ValidateModelWithoutMinibatchSource(animalsModelFile, Path.Combine(animalDataFolder, "Test"), animals,
                    imageDims, animalModelNumClasses, device);
                return;
            }

            List<Tuple<string, int, float[]>> trainingDataMap =
                PrepareTrainingDataFromSubfolders(Path.Combine(animalDataFolder, "Train"), animals, imageDims);

            // prepare the transfer model
            string predictionNodeName = "prediction";
            Variable imageInput, labelInput;
            Function trainingLoss, predictionError;
            Function transferLearningModel = CreateTransferLearningModel(Path.Combine(ExampleImageFoler, BaseResnetModelFile), featureNodeName, predictionNodeName,
                lastHiddenNodeName, animalModelNumClasses, device,
                out imageInput, out labelInput, out trainingLoss, out predictionError);

            // prepare for training
            int numMinibatches = 5;
            float learningRatePerMinibatch = 0.2F;
            float learningmomentumPerMinibatch = 0.9F;
            float l2RegularizationWeight = 0.1F;

            AdditionalLearningOptions additionalLearningOptions = 
                new AdditionalLearningOptions() { l2RegularizationWeight = l2RegularizationWeight };
            IList<Learner> parameterLearners = new List<Learner>() {
                    Learner.MomentumSGDLearner(transferLearningModel.Parameters(),
                    new TrainingParameterScheduleDouble(learningRatePerMinibatch, 0),
                    new TrainingParameterScheduleDouble(learningmomentumPerMinibatch, 0),
                    true,
                    additionalLearningOptions)};
            var trainer = Trainer.CreateTrainer(transferLearningModel, trainingLoss, predictionError, parameterLearners);

            // train the model
            for (int minibatchCount = 0; minibatchCount < numMinibatches; ++minibatchCount)
            {
                Value imageBatch, labelBatch;
                int batchCount = 0, batchSize = 15;
                while (GetImageAndLabelMinibatch(trainingDataMap, batchSize, batchCount++,
                    imageDims, animalModelNumClasses, device, out imageBatch, out labelBatch))
                {
                    //TODO: sweepEnd should be set properly.
#pragma warning disable 618
                    trainer.TrainMinibatch(new Dictionary<Variable, Value>() {
                    { imageInput, imageBatch },
                        { labelInput, labelBatch } }, device);
#pragma warning restore 618
                    TestHelper.PrintTrainingProgress(trainer, minibatchCount, 1);
                }
            }                       

            // save the trained model
            transferLearningModel.Save(animalsModelFile);

            // done with training, continue with validation
            double error = ValidateModelWithoutMinibatchSource(animalsModelFile, Path.Combine(animalDataFolder, "Test"), animals,
                                imageDims, animalModelNumClasses, device);
            Console.WriteLine(error);
        }

        /// <summary>
        /// With trainin/evaluation data in memory, this method returns minibatch data 
        /// contained in CNTK Value instances.
        /// </summary>
        /// <param name="trainingDataMap">Preloaded and processed in memory data.</param>
        /// <param name="batchSize"></param>
        /// <param name="batchCount">current batch count.</param>
        /// <param name="imageDims"></param>
        /// <param name="numClasses"></param>
        /// <param name="device"></param>
        /// <param name="imageBatch">Return of CNTK Value containing images</param>
        /// <param name="labelBatch">Return of CNTK Value containing labels</param>
        /// <returns></returns>
        private static bool GetImageAndLabelMinibatch(List<Tuple<string, int, float[]>> trainingDataMap, 
            int batchSize, int batchCount, int[] imageDims, int numClasses, DeviceDescriptor device,
            out Value imageBatch, out Value labelBatch)
        {
            int actualBatchSize = Math.Min(trainingDataMap.Count() - batchSize * batchCount, batchSize);
            if (actualBatchSize <= 0)
            {
                imageBatch = null;
                labelBatch = null;
                return false;
            }

            if (batchCount == 0)
            {
                // randomize 
                int n = trainingDataMap.Count;
                Random random = new Random(0);
                while (n > 1)
                {
                    n--;
                    int k = random.Next(n + 1);
                    var value = trainingDataMap[k];
                    trainingDataMap[k] = trainingDataMap[n];
                    trainingDataMap[n] = value;
                }
            }

            int imageSize = imageDims[0] * imageDims[1] * imageDims[2];
            float[] batchImageBuf = new float[actualBatchSize * imageSize];
            float[] batchLabelBuf = new float[actualBatchSize * numClasses];
            for (int i = 0; i < actualBatchSize; i++)
            {
                int index = i + batchSize * batchCount;
                trainingDataMap[index].Item3.CopyTo(batchImageBuf, i * imageSize);
                for (int c = 0; c < numClasses; c++)
                {
                    if (c == trainingDataMap[index].Item2)
                    {
                        batchLabelBuf[i * numClasses + c] = 1;
                    }
                    else
                    {
                        batchLabelBuf[i * numClasses + c] = 0;
                    }
                }
            }

            imageBatch = Value.CreateBatch<float>(imageDims, batchImageBuf, device);
            labelBatch = Value.CreateBatch<float>(new int[] { numClasses }, batchLabelBuf, device);
            return true;
        }

        /// <summary>
        /// Construct a model by cloning from an existing base model. 
        /// Model parameters of the cloned portion (very large part of the model) is freezed 
        /// so training of the new model can be fast. 
        /// Input to the new model is replaced by a computation layer for data feeding and normalization.
        /// Output of the model is build with an trainable dense layer.
        /// </summary>
        /// <param name="baseModelFile">where the base model can be loaded</param>
        /// <param name="featureNodeName">the input feature node name</param>
        /// <param name="outputNodeName">output node name</param>
        /// <param name="hiddenNodeName">the node to clone from. </param>
        /// <param name="numClasses"></param>
        /// <param name="device"></param>
        /// <param name="imageInput">input node of the new model</param>
        /// <param name="labelInput">label node of the new model</param>
        /// <param name="trainingLoss">loss function of the model</param>
        /// <param name="predictionError">prediction function of the new model</param>
        /// <returns></returns>
        private static Function CreateTransferLearningModel(string baseModelFile, string featureNodeName, string outputNodeName,
            string hiddenNodeName, int numClasses, DeviceDescriptor device, 
            out Variable imageInput, out Variable labelInput, out Function trainingLoss, out Function predictionError)
        {
            Function baseModel = Function.Load(baseModelFile, device);

            imageInput = Variable.InputVariable(imageDims, DataType.Float);
            labelInput = Variable.InputVariable(new int[] { numClasses }, DataType.Float);
            Function normalizedFeatureNode = CNTKLib.Minus(imageInput, Constant.Scalar(DataType.Float, 114.0F));

            Variable oldFeatureNode = baseModel.Arguments.Single(a => a.Name == featureNodeName);
            Function lastNode = baseModel.FindByName(hiddenNodeName);

            // Clone the desired layers with fixed weights
            Function clonedLayer = CNTKLib.AsComposite(lastNode).Clone(
                ParameterCloningMethod.Freeze,
                new Dictionary<Variable, Variable>() { { oldFeatureNode, normalizedFeatureNode } });

            // Add new dense layer for class prediction
            Function clonedModel = TestHelper.Dense(clonedLayer, numClasses, device, Activation.None, outputNodeName);

            trainingLoss = CNTKLib.CrossEntropyWithSoftmax(clonedModel, labelInput);
            predictionError = CNTKLib.ClassificationError(clonedModel, labelInput);

            return clonedModel;
        }

        /// <summary>
        /// Validate a model with data loaded and processed explicitly in the sample code.
        /// </summary>
        /// <param name="modelFile"></param>
        /// <param name="testDataFolder"></param>
        /// <param name="animals"></param>
        /// <param name="imageDims"></param>
        /// <param name="numClasses"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        private static double ValidateModelWithoutMinibatchSource(string modelFile, string testDataFolder, string[] animals, 
            int[] imageDims, int numClasses, DeviceDescriptor device)
        {
            Function model = Function.Load(modelFile, device);
            List<Tuple<string, int, float[]>> testDataMap =
                PrepareTrainingDataFromSubfolders(testDataFolder, animals, imageDims);
            Value imageBatch, labelBatch;
            int batchCount = 0, batchSize = 15;
            int miscountTotal = 0, totalCount = 0;
            while (GetImageAndLabelMinibatch(testDataMap, batchSize, batchCount++,
                TransferLearning.imageDims, numClasses, device, out imageBatch, out labelBatch))
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { model.Arguments[0], imageBatch } };

                Variable outputVar = model.Output;
                var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null } };
                model.Evaluate(inputDataMap, outputDataMap, device);
                var outputVal = outputDataMap[outputVar];
                var actual = outputVal.GetDenseData<float>(outputVar);
                var expected = labelBatch.GetDenseData<float>(model.Output);

                var actualLabels = actual.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();
                var expectedLabels = expected.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();

                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();
                miscountTotal += misMatches;
                totalCount += actualLabels.Count();

                Console.WriteLine($"Validating Model: Total Samples = {totalCount}, Misclassify Count = {miscountTotal}");
            }

            return 1.0 * miscountTotal / testDataMap.Count();
        }

        /// <summary>
        /// Validate a mode with MinibatchSource for data preparation and batching.
        /// </summary>
        /// <param name="modelFile"></param>
        /// <param name="mapFile"></param>
        /// <param name="imageDim"></param>
        /// <param name="numClasses"></param>
        /// <param name="device"></param>
        /// <param name="maxCount"></param>
        /// <returns></returns>
        private static float ValidateModelWithMinibatchSource(string modelFile, string mapFile,
            int[] imageDim, int numClasses, DeviceDescriptor device, int maxCount = 1000)
        {
            Function model = Function.Load(modelFile, device);
            var imageInput = model.Arguments[0];
            var labelOutput = model.Output;

            MinibatchSource minibatchSource = CreateMinibatchSource(mapFile,
                imageDims, numClasses);
            var featureStreamInfo = minibatchSource.StreamInfo("image");
            var labelStreamInfo = minibatchSource.StreamInfo("labels");

            int batchSize = 50;
            int miscountTotal = 0, totalCount = 0;
            while (true)
            {
                var minibatchData = minibatchSource.GetNextMinibatch((uint)batchSize, device);
                if (minibatchData == null)
                    break;
                totalCount += (int)minibatchData[featureStreamInfo].numberOfSamples;
                if (totalCount > maxCount)
                    break;

                // expected labels are in the minibatch data.
                var labelData = minibatchData[labelStreamInfo].data.GetDenseData<float>(labelOutput);
                var expectedLabels = labelData.Select(l => l.IndexOf(l.Max())).ToList();

                var inputDataMap = new Dictionary<Variable, Value>() {
                    { imageInput, minibatchData[featureStreamInfo].data }
                };

                var outputDataMap = new Dictionary<Variable, Value>() {
                    { labelOutput, null }
                };

                model.Evaluate(inputDataMap, outputDataMap, device);
                var outputData = outputDataMap[labelOutput].GetDenseData<float>(labelOutput);
                var actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();

                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

                miscountTotal += misMatches;
                Console.WriteLine($"Validating Model: Total Samples = {totalCount}, Misclassify Count = {miscountTotal}");
            }

            float errorRate = 1.0F * miscountTotal / totalCount;
            Console.WriteLine($"Model Validation Error = {errorRate}");
            return errorRate;
        }

        private static Dictionary<string, int> LoadMapFile(string mapFile)
        {
            Dictionary<string, int> imageFileToLabel = new Dictionary<string, int>();
            string line;

            if (File.Exists(mapFile))
            {
                StreamReader file = null;
                try
                {
                    file = new StreamReader(mapFile);
                    while ((line = file.ReadLine()) != null)
                    {
                        int spaceIndex = line.IndexOfAny(new char[]{ ' ', '\t'});
                        string filePath = line.Substring(0, spaceIndex);
                        int label = int.Parse(line.Substring(spaceIndex).Trim());
                        imageFileToLabel.Add(filePath, label);
                    }
                }
                finally
                {
                    if (file != null)
                        file.Close();
                }
            }
            return imageFileToLabel;
        }

        /// <summary>
        /// The method assumes that images are stored in subfolders named after the image classes.
        /// </summary>
        /// <param name="rootFolderOfClassifiedImages"></param>
        /// <param name="categories"></param>
        /// <param name="imageDims"></param>
        /// <returns>List of tuples of image file path, image labels, and image data.</returns>
        private static List<Tuple<string, int, float[]>> PrepareTrainingDataFromSubfolders(
            string rootFolderOfClassifiedImages, string[] categories, int[] imageDims)
        {
            // classified images are stored in named folders
            List<Tuple<string, int, float[]>> dataMap = new List<Tuple<string, int, float[]>>();
            int categoryIndex = 0;
            foreach (var category in categories)
            {
                var fileInfos = Directory.GetFiles(Path.Combine(rootFolderOfClassifiedImages, category), "*.jpg");
                foreach (var file in fileInfos)
                {
                    float[] image = LoadBitmap(imageDims, file).ToArray();
                    dataMap.Add(new Tuple<string, int, float[]>(file, categoryIndex, image));
                }
                categoryIndex++;
            }

            return dataMap;
        }

        private static MinibatchSource CreateMinibatchSource(string map_file, int[] image_dims, int num_classes)
        {
            List<CNTKDictionary> transforms = new List<CNTKDictionary>
            {
                CNTKLib.ReaderScale(image_dims[0], image_dims[1], image_dims[2], "linear")
            };

            var deserializerConfiguration = CNTKLib.ImageDeserializer(map_file, 
                "labels", (uint)num_classes,
                "image", transforms);

            MinibatchSourceConfig config = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfiguration });

            return CNTKLib.CreateCompositeMinibatchSource(config);
        }

        private static List<float> LoadBitmap(int[] image_dims, string filePath)
        {
            Bitmap bmp = new Bitmap(Bitmap.FromFile(filePath));
            var resized = bmp.Resize(image_dims[0], image_dims[1], true);
            List<float> resizedCHW = resized.ParallelExtractCHW();
            return resizedCHW;
        }
    }
}
