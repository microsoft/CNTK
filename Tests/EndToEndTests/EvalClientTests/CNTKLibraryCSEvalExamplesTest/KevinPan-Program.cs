using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using CNTK;
using System.Collections.Concurrent;

namespace NewsInsightEvaluation
{
    class KevinProgram
    {

        private static Object lockObj = new Object();

        public static void KevinMain()
        {

#if ORIG
            Data.Init(@"\\stchost-50\ml\cntk\sample\NewsInsight\trained_model\src.l3g.txt");

            var self_attention_version = new ModelEvaluation
            {
                model_file = @"\\stchost-50\ml\cntk\sample\NewsInsight\trained_model\cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002.dnn",
                name = "cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002"
            };
            self_attention_version.Init();
            Console.WriteLine("complete loading model: {0}", self_attention_version.name);

            var without_attention = new ModelEvaluation
            {
                model_file = @"\\stchost-50\ml\cntk\sample\NewsInsight\trained_model\cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_900000_58410932.dnn",
                name = "cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_900000_58410932"
            };
            without_attention.Init();
            Console.WriteLine("complete loading model: {0}", without_attention.name);

            Validation v = new Validation
            {
                raw_data_path = @"\\stchost-50\ml\cntk\sample\NewsInsight\trained_model\validate",
                validation_sample_count = 50,
                model_parallel_count = 20,
                negative_document_count = 1
            };
            v.BuildSamples();

            v.Validate(self_attention_version);

            v.Validate(without_attention);
#endif

            Data.Init(@"c:\CNTKMisc\KevinPan-Memory\trained_model\src.l3g.txt");

            //var self_attention_version = new ModelEvaluation
            //{
            //    model_file = @"E:\CNTKMisc\KevinPan-Memory\trained_model\cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002.dnn",
            //    name = "cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002"
            //};
            //self_attention_version.Init();
            //Console.WriteLine("complete loading model: {0}", self_attention_version.name);

            var queryText = "24 hours of le mans";
            var documentText = "live stream for one of motorsport s biggest events ground to a halt by cyberattacks";
            var querySequence = CreateSequence(queryText);
            StreamWriter file = new StreamWriter("queryTextInput.txt");
            var numNonZeros = 0;
            for (int i = 0; i < querySequence.Count; i++)
            {

                if (querySequence[i] != 0)
                {
                    Console.WriteLine("sequence[" + i + "]=" + querySequence[i]);
                    var s = String.Format("{0}, ", i, querySequence[i]);
                    numNonZeros++;
                    file.Write(s);
                }
                
            }
            Console.WriteLine("Total: " + numNonZeros);
            file.Close();
            var documentSequence = CreateSequence(documentText);
            file = new StreamWriter("documentTextInput.txt");
            numNonZeros = 0;
            for (int i = 0; i < documentSequence.Count; i++)
            {
                if (documentSequence[i] != 0)
                {
                    Console.WriteLine("sequence[" + i + "]=" + documentSequence[i]);
                    var s = String.Format("{0}, ", i, documentSequence[i]);
                    numNonZeros++;
                    file.Write(s);
                }
            }
            Console.WriteLine("Total: " + numNonZeros);
            file.Close();

            var model_file = @"C:\CNTKMisc\KevinPan-Memory\trained_model\cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002.dnn";
            var modelFunc = Function.Load(model_file, DeviceDescriptor.CPUDevice);

            var parallel_model_list_1 = new BlockingCollection<Function>();
            // var parallel_model_list = new List<ModelEvaluation>();
            var numOfModels = 20;
            for (int i = 0; i < numOfModels;  i++)
            {
                // parallel_model_list.Add(model.CloneByFile());

                parallel_model_list_1.Add(modelFunc.Clone(ParameterCloningMethod.Share));  //CloneByFile()); // model.Clone());
            }

            //var queryText = "24 hours of le mans";
            //var documentText = "live stream for one of motorsport s biggest events ground to a halt by cyberattacks";
            //var querySequence = CreateSequence(queryText);
            //StreamWriter file = new StreamWriter("queryTextInput.txt");
            //foreach (float val in querySequence)
            //    file.WriteLine(val);
            //file.Close();
            //var documentSequence = CreateSequence(documentText);
            //file = new StreamWriter("documentTextInput.txt");
            //foreach (float val in documentSequence)
            //    file.WriteLine(val);
            //file.Close();

            int validate_sample_count = 0;
            var samples = new List<int>(50);
            var querySeqList = new BlockingCollection<List<float>>();
            var documentSeqList = new BlockingCollection<List<float>>();

            for (int i = 0; i < 20; i++)
            {
                samples.Add(i);
                var q = new List<float>(querySequence);
                var d = new List<float>(documentSequence);
                querySeqList.Add(q);
                documentSeqList.Add(d);
            }

            //Parallel.ForEach(samples, new ParallelOptions() { MaxDegreeOfParallelism = samples.Count }, (sample) =>
            //// Parallel.ForEach(validation_samples, new ParallelOptions() { MaxDegreeOfParallelism = validation_samples.Count }, (sample) =>
            //{
            //    var m = parallel_model_list_1.Take();
            //    try
            //    {
            //        List<float> q = querySeqList.Take();
            //        List<float> d = documentSeqList.Take();
            //        Console.WriteLine(string.Format("start validating sample {0}...", sample));
            //        CosineDistance(m, q, d);

            //        // sample.Validate(m);
            //        //sample.RecordPositiveDocument(positive_record);
            //        //sample.RecordNegativeDocument(negative_record);
            //        System.Threading.Interlocked.Increment(ref validate_sample_count);
            //        Console.WriteLine(string.Format("...validated {0} sample...", validate_sample_count));
            //    }
            //    finally
            //    {
            //        parallel_model_list_1.Add(m);
            //    }
            //});

            Parallel.ForEach(parallel_model_list_1, new ParallelOptions() { MaxDegreeOfParallelism = parallel_model_list_1.Count }, (model) =>
            // Parallel.ForEach(validation_samples, new ParallelOptions() { MaxDegreeOfParallelism = validation_samples.Count }, (sample) =>
            {
                    List<float> q = querySeqList.Take();
                    List<float> d = documentSeqList.Take();
                    // Console.WriteLine(string.Format("start validating sample on model {0}...", samples.Take()));
                    CosineDistance(model, q, d);

                    // sample.Validate(m);
                    //sample.RecordPositiveDocument(positive_record);
                    //sample.RecordNegativeDocument(negative_record);
                    System.Threading.Interlocked.Increment(ref validate_sample_count);
                    Console.WriteLine(string.Format("...validated {0} sample...", validate_sample_count));

                
            });

            //Validation v = new Validation
            //{
            //    raw_data_path = @"E:\CNTKMisc\KevinPan-Memory\trained_model\validate",
            //    validation_sample_count = 50,
            //    model_parallel_count = 5, // 20,
            //    negative_document_count = 1
            //};
            // v.BuildSamples();

            // v.Validate(self_attention_version);

            //var without_attention = new ModelEvaluation
            //{
            //    model_file = @"E:\CNTKMisc\KevinPan-Memory\trained_model\cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_900000_58410932.dnn",
            //    name = "cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_900000_58410932"
            //};
            //without_attention.Init();
            //Console.WriteLine("complete loading model: {0}", without_attention.name);

            //v.Validate(without_attention);

            //{
            //    //var pq = "2017 cholistan desert rally winners";
            //    //var pd = "12th cholistan desert rally kicks off amidst extravagant fanfare";
            //    var cd = self_attention_version.CosineDistance("google", "microsoft");
            //    Console.WriteLine(cd);

            //    var nq = "airbus biggest plane";
            //    var nd = "call of duty infinite warfare zombies update all ps4 and xbox";

            //    Console.WriteLine("negative = {0}", self_attention_version.CosineDistance(nq, nd));

            //}
        }

        public static List<float> CreateSequence(string rawText)
        {
            List<float> sequence = new List<float>();
            string s = TextUtils.N1Normalize(rawText);
            var words = s.Split(' ');
            for (int i = 0; i < words.Length; i++)
            {
                if (!string.IsNullOrWhiteSpace(words[i]))
                {
                    float[] v = new float[49293];
                    var encode = TextUtils.EncodeWord2Letter3Gram(words[i], Data.Vocabulary);
                    foreach (var e in encode)
                    {
                        v[e.Key] = e.Value;
                        //if (e.Value != 0)
                        //    Console.WriteLine("Word=" + i + ", Index=" + e.Key + ",value=" + e.Value);
                    }
                    sequence.AddRange(v);
                }
            }

           
            return sequence;
        }

        public static float CosineDistance(Function modelFunc, List<float> querySeq, List<float> docSeq)
        {
            var cosine_distance = modelFunc.FindByName("query_positive_document_cosine_distance");
            //var queryRecurrenceOutput = cosine_distance.Arguments[0];
            //var positiveDocumentRecurrenceOutput = cosine_distance.Arguments[1];
            //var queryDocumentCosineDistance = cosine_distance.Output;

            var evalFunc = Function.AsComposite(cosine_distance);
            var queryInput = evalFunc.Arguments.Where(variable => string.Equals(variable.Name, "query")).Single();
            var documentInput = evalFunc.Arguments.Where(variable => string.Equals(variable.Name, "positive_document")).Single();

            var queryInputShape = queryInput.Shape;
            var documentInputShape = documentInput.Shape;

            var intputs = new Dictionary<Variable, Value>();
            var queryInputValue = Value.CreateSequence<float>(queryInputShape, querySeq, DeviceDescriptor.CPUDevice);  // CreateSequenceInput(queryInputShape, query);
            var documentInputValue = Value.CreateSequence<float>(documentInputShape, docSeq, DeviceDescriptor.CPUDevice); // CreateSequenceInput(documentInputShape, document);
            intputs.Add(queryInput, queryInputValue);
            intputs.Add(documentInput, documentInputValue);

            var outputs = new Dictionary<Variable, Value>();
            //var q1 = queryRecurrenceOutput.Owner.Output;
            //var p1 = positiveDocumentRecurrenceOutput.Owner.Output;
            //var evalFunc = Function.Combine(new[] { q1, p1, queryDocumentCosineDistance });
            //var evalFunc = Function.AsComposite(cosine_distance);

            outputs.Add(evalFunc.Output, null);
            //var queryRecurrenceOutput1 = evalFunc.Outputs[0];
            //var positiveDocumentRecurrenceOutput1 = evalFunc.Outputs[1];
            //var queryDocumentCosineDistance1 = evalFunc.Outputs[2];
            //outputs.Add(queryRecurrenceOutput1, null);
            //outputs.Add(positiveDocumentRecurrenceOutput1, null);
            //outputs.Add(queryDocumentCosineDistance1, null);

            {
                evalFunc.Evaluate(intputs, outputs, DeviceDescriptor.CPUDevice);
            }

            //var queryEmbeddingValue = outputs[queryRecurrenceOutput1];
            //var documentEmbeddingValue = outputs[positiveDocumentRecurrenceOutput1];
            //var queryDocumentCosineDistanceValue = outputs[queryDocumentCosineDistance1];

            //var queryEmbeddingOutput = queryEmbeddingValue.GetDenseData<float>(queryRecurrenceOutput1);
            //var documentEmbeddingOutput = documentEmbeddingValue.GetDenseData<float>(positiveDocumentRecurrenceOutput1);
            //var queryDocumentCosineDistanceOutput = queryDocumentCosineDistanceValue.GetDenseData<float>(queryDocumentCosineDistance1);

            //var d = Utils.Cosine(queryEmbeddingOutput[0].ToArray(), documentEmbeddingOutput[0].ToArray());

            //Console.WriteLine("queryDocumentCosineDistanceOutput = {0}", queryDocumentCosineDistanceOutput[0][0]);

            var d = 0;
            return d;
        }

        //public static Value CreateSequenceInput(NDShape sampleShape, string rawText)
        //{
        //    List<float> sequence = new List<float>();
        //    string s = TextUtils.N1Normalize(rawText);
        //    var words = s.Split(' ');
        //    for (int i = 0; i < words.Length; i++)
        //    {
        //        if (!string.IsNullOrWhiteSpace(words[i]))
        //        {
        //            float[] v = new float[49293];
        //            var encode = TextUtils.EncodeWord2Letter3Gram(words[i], Data.Vocabulary);
        //            foreach (var e in encode)
        //            {
        //                v[e.Key] = e.Value;
        //            }
        //            sequence.AddRange(v);
        //        }
        //    }

        //    return Value.CreateSequence<float>(sampleShape, sequence, DeviceDescriptor.CPUDevice);
        //}
    }
}
