using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CNTKLibraryCSTrainingTest
{
#pragma warning disable CS0219 // Variable is assigned but its value is never used
    public class TransferLearning
    {
        public static void TrainAndEvaluateTransferLearning(DeviceDescriptor device)
        {
            // string base_model_file = "ResNet_18.model";
            // string feature_node_name = "features";
            // string last_hidden_node_name = "z.x";
            // int[] image_dims = new int[] { 3, 224, 224 };
            string base_model_file = "C:/LiqunWA/cntk/ForkSaved/CNTK_103D.model";
            string feature_node_name = "features";
            string last_hidden_node_name = "second_max";
            int[] image_dims = new int[] { 28, 28, 1};

            //string flower_model_file = "FlowersTransferLearning.model";
            //string flower_results_file = "FlowersPredictions.txt";
            //int flower_num_classes = 102;
            int base_model_num_classes = 10;


            //string animal_model_file = "AnimalsTransferLearning.model";
            //string animal_results_file = "AnimalsPredictions.txt";
            // int animal_num_classes = 0; // ?

            bool freeze = false;

            var image_input = Variable.InputVariable(image_dims, DataType.Float, feature_node_name);
            var label_input = Variable.InputVariable(new int[] { base_model_num_classes }, DataType.Float, "outputs");

            Function base_model = Function.Load(base_model_file, device);
            PrintGraph(base_model.RootFunction, 0);
            var tl_model = create_model(base_model, feature_node_name, last_hidden_node_name, 
                base_model_num_classes, image_input, device, freeze);
            PrintGraph(tl_model.RootFunction, 0);
        }

        static string new_output_node_name = "prediction";

        public static Function create_model(Function base_model, string feature_node_name, 
            string last_hidden_node_name, int num_classes, Variable input_features, 
            DeviceDescriptor device, bool freeze = false)
        {
            Variable old_feature_node = base_model.Arguments.Single(a => a.Name == feature_node_name);
            Function last_node = base_model.FindByName(last_hidden_node_name);
            Variable new_feature_node = CNTKLib.PlaceholderVariable(feature_node_name);

            // Clone the desired layers with fixed weights
            Function cloned_layers = Function.Combine(new List<Variable>() { ((Variable)last_node).Owner }).Clone(
                freeze ? ParameterCloningMethod.Freeze : ParameterCloningMethod.Clone,
                new Dictionary<Variable, Variable>() { { old_feature_node, new_feature_node } });

            Console.WriteLine("cloned_layers");
            PrintGraph(cloned_layers.RootFunction, 0);

            // Add new dense layer for class prediction
            Function feat_norm = CNTKLib.Minus(input_features, Constant.Scalar(DataType.Float, 114.0F));
            Function cloned_out = cloned_layers.ReplacePlaceholders(new Dictionary<Variable, Variable>() { { new_feature_node, feat_norm } });

            Console.WriteLine("cloned_layers after ReplacePlaceholders");
            PrintGraph(cloned_layers.RootFunction, 0);

            Console.WriteLine("cloned_out");
            PrintGraph(cloned_out.RootFunction, 0);
            Function z = TestHelper.Dense(cloned_out, num_classes, device, TestHelper.Activation.None, new_output_node_name);

            Console.WriteLine("z");
            PrintGraph(z.RootFunction, 0);

            return cloned_layers;
        }

        static MinibatchSource create_mb_source(string map_file, int[] image_dims, int num_classes, bool randomize = true)
        {
            DictionaryVector transforms = new DictionaryVector();
            transforms.Add(CNTKLib.ReaderScale(image_dims[2], image_dims[1], image_dims[0], "linear"));

            CNTKDictionary deserializer = CNTKLib.ImageDeserializer(map_file, "label", (uint)num_classes, "image", transforms);
            DictionaryVector deserializers = new DictionaryVector();
            deserializers.Add(deserializer);
            MinibatchSourceConfig minibatchSourceConfig = new MinibatchSourceConfig(deserializers, randomize);
            return CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
        }

        // Trains a transfer learning model
        static void train_model(string base_model_file, string feature_name, int[] image_dims, int num_classes, string last_hidden_node_name,
            string train_map_file, int max_epochs, int mb_size, double lr_per_mb, bool freeze_weights, double l2_reg_weight,
            double momentum_per_mb, DeviceDescriptor device, int max_images = -1)
        {
            int num_epochs = max_epochs;
            int epoch_size = File.ReadLines(train_map_file).Count(); ;
            if (max_images > 0)
                epoch_size = Math.Min(epoch_size, max_images);
            int minibatch_size = mb_size;

            // Create the minibatch source and input variables
            MinibatchSource minibatch_source = create_mb_source(train_map_file, image_dims, num_classes);
            Variable image_input = Variable.InputVariable(image_dims, DataType.Float, "");
            Variable label_input = Variable.InputVariable(new int[] { num_classes }, DataType.Float, "");

            var featureStreamInfo = minibatch_source.StreamInfo("features");
            var labelStreamInfo = minibatch_source.StreamInfo("labels");

            Function base_model = Function.Load(base_model_file, device);
            // Instantiate the transfer learning model and loss function
            Function tl_model = create_model(base_model, feature_name, last_hidden_node_name,
                num_classes, image_input, device, freeze_weights);
            var ce = CNTKLib.CrossEntropyWithSoftmax(tl_model, label_input);
            var pe = CNTKLib.ClassificationError(tl_model, label_input);

            // Instantiate the trainer object
            var lr_schedule = new TrainingParameterScheduleDouble(lr_per_mb, TrainingParameterScheduleDouble.UnitType.Minibatch);
            var mm_schedule = new MomentumAsTimeConstantScheduleCS(momentum_per_mb);


            bool unitGain = true;
            Learner learner = Learner.MomentumSGDLearner(tl_model.Parameters(), lr_schedule, mm_schedule, unitGain,
                           new AdditionalLearningOptions()
                           {
                               l2RegularizationWeight = l2_reg_weight
                           });
            Trainer trainer = Trainer.CreateTrainer(tl_model, ce, pe, new List<Learner>() { learner });

            // Get minibatches of images and perform model training
            // print("Training transfer learning model for {0} epochs (epoch_size = {1}).".format(num_epochs, epoch_size))
            // CNTKLib.Logging.log_number_of_parameters(tl_model)
            // progress_printer = C.logging.ProgressPrinter(tag = 'Training', num_epochs = num_epochs)
            int outputFrequencyInMinibatches = 1;
            for (int epoch = 0; epoch < num_epochs; epoch++)
            {
                uint sample_count = 0;
                int minibatchIndex = 0;
                while (sample_count < epoch_size)  // loop over minibatches in the epoch
                {
                    uint minibatchSize = (uint)Math.Min(minibatch_size, epoch_size - sample_count);
                    var minibatchData = minibatch_source.GetNextMinibatch(minibatchSize, device);
                    if (minibatchData.empty())
                        break;

                    sample_count += trainer.PreviousMinibatchSampleCount();
                    var arguments = new Dictionary<Variable, MinibatchData>
                    {
                        { image_input, minibatchData[featureStreamInfo] },
                        { label_input, minibatchData[labelStreamInfo] }
                    };
                    trainer.TrainMinibatch(arguments, device);
                    TestHelper.PrintTrainingProgress(trainer, minibatchIndex++, outputFrequencyInMinibatches);
                }
            }
        }
        static void PrintGraph(Function function, int spaces)
        {
            if (function.Inputs == null || function.Inputs.Count() == 0)
            {
                Console.WriteLine(new string('.', spaces) + "(" + function.Name + ")" + function.AsString());
                return;
            }
            
            foreach (var input in function.Inputs)
            {
                Console.WriteLine(new string('.', spaces) + "(" + function.Name + ")" + "->" + 
                    "(" + input.Name + ")" + input.AsString());
            }
            foreach (var input in function.Inputs)
            {
                if (input.Owner != null)
                {
                    Function f = input.Owner;
                    PrintGraph(f, spaces + 4);
                }
            }
        }
    }
}
