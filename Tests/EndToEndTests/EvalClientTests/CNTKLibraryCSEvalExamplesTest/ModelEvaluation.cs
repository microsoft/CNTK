using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using System.Diagnostics;

namespace NewsInsightEvaluation
{
    class ModelEvaluation
    {
        private Function model;

        public string name;

        public string model_file;

        public ModelEvaluation() { }

        public ModelEvaluation(Function model)
        {
            this.model = model;
        }

        public ModelEvaluation Clone()
        {
            return new ModelEvaluation(this.model.Clone(ParameterCloningMethod.Share));
        }

        public ModelEvaluation CloneByFile()
        {
            var m = new ModelEvaluation
            {
                name = this.name,
                model_file = this.model_file
            };
            m.Init();
            return m;
        }

        public void Init()
        {
            model = Function.Load(this.model_file, DeviceDescriptor.CPUDevice);
        }

        //public float[] EmbeddingQuery(string query)
        //{
        //    var fun = this.model.FindByName("query_positive_document_cosine_distance");
        //    var queryRecurrenceOutput = fun.Arguments[0];

        //    var queryInput = model.Arguments.Where(variable => string.Equals(variable.Name, "query")).Single();

        //    var queryInputShape = queryInput.Shape;
        //    var intputs = new Dictionary<Variable, Value>();
        //    var queryInputValue = CreateSequenceInput(queryInputShape, query);
        //    intputs.Add(queryInput, queryInputValue);

        //    var outputs = new Dictionary<Variable, Value>();
        //    outputs.Add(queryRecurrenceOutput, null);

        //    model.Evaluate(intputs, outputs, DeviceDescriptor.CPUDevice);

        //    var queryEmbeddingValue = outputs[queryRecurrenceOutput];
        //    var queryEmbeddingOutput = queryEmbeddingValue.GetDenseData<float>(queryRecurrenceOutput);

        //    return queryEmbeddingOutput.First().ToArray();
        //}

        //public float[] EmbeddingQuery1(string query)
        //{
        //    var splice = model.RootFunction;
        //    var exp = splice.RootFunction.Arguments[0].Owner.RootFunction;
        //    var cosine_distance = exp.Arguments[0].Owner.RootFunction;
        //    var queryRecurrenceOutput = cosine_distance.Arguments[0];

        //    var queryInput = model.Arguments.Where(variable => string.Equals(variable.Name, "query")).Single();

        //    var queryInputShape = queryInput.Shape;

        //    var intputs = new Dictionary<Variable, Value>();
        //    var queryInputValue = CreateSequenceInput(queryInputShape, query);
        //    intputs.Add(queryInput, queryInputValue);

        //    var outputs = new Dictionary<Variable, Value>();
        //    outputs.Add(queryRecurrenceOutput, null);

        //    model.Evaluate(intputs, outputs, DeviceDescriptor.CPUDevice);

        //    var queryEmbeddingValue = outputs[queryRecurrenceOutput];
        //    var queryEmbeddingOutput = queryEmbeddingValue.GetDenseData<float>(queryRecurrenceOutput);

        //    return queryEmbeddingOutput.First().ToArray();
        //}

        //public float[] EmbeddingDocument(string document)
        //{
        //    var cosine_distance = this.model.FindByName("query_positive_document_cosine_distance");
        //    var positiveDocumentRecurrenceOutput = cosine_distance.Arguments[1];

        //    var documentInput = model.Arguments.Where(variable => string.Equals(variable.Name, "positive_document")).Single();
        //    var documentInputShape = documentInput.Shape;

        //    var intputs = new Dictionary<Variable, Value>();
        //    var documentInputValue = CreateSequenceInput(documentInputShape, document);
        //    intputs.Add(documentInput, documentInputValue);

        //    var outputs = new Dictionary<Variable, Value>();
        //    outputs.Add(positiveDocumentRecurrenceOutput, null);

        //    model.Evaluate(intputs, outputs, DeviceDescriptor.CPUDevice);

        //    var documentEmbeddingValue = outputs[positiveDocumentRecurrenceOutput];
        //    var documentEmbeddingOutput = documentEmbeddingValue.GetDenseData<float>(positiveDocumentRecurrenceOutput);

        //    return documentEmbeddingOutput.First().ToArray();
        //}

        //public float[] EmbeddingDocument1(string document)
        //{
        //    var splice = model.RootFunction;
        //    var exp = splice.RootFunction.Arguments[0].Owner.RootFunction;
        //    var cosine_distance = exp.Arguments[0].Owner.RootFunction;
        //    var positiveDocumentRecurrenceOutput = cosine_distance.Arguments[1];
        //    var documentInput = model.Arguments.Where(variable => string.Equals(variable.Name, "positive_document")).Single();
        //    var documentInputShape = documentInput.Shape;

        //    var intputs = new Dictionary<Variable, Value>();
        //    var documentInputValue = CreateSequenceInput(documentInputShape, document);
        //    intputs.Add(documentInput, documentInputValue);

        //    var outputs = new Dictionary<Variable, Value>();
        //    outputs.Add(positiveDocumentRecurrenceOutput, null);

        //    model.Evaluate(intputs, outputs, DeviceDescriptor.CPUDevice);

        //    var documentEmbeddingValue = outputs[positiveDocumentRecurrenceOutput];
        //    var documentEmbeddingOutput = documentEmbeddingValue.GetDenseData<float>(positiveDocumentRecurrenceOutput);

        //    return documentEmbeddingOutput.First().ToArray();
        //}

        public float CosineDistance(string query, string document)
        {
            var cosine_distance = this.model.FindByName("query_positive_document_cosine_distance");
            var queryRecurrenceOutput = cosine_distance.Arguments[0];
            var positiveDocumentRecurrenceOutput = cosine_distance.Arguments[1];
            var queryDocumentCosineDistance = cosine_distance.Output;

            var queryInput = model.Arguments.Where(variable => string.Equals(variable.Name, "query")).Single();
            var documentInput = model.Arguments.Where(variable => string.Equals(variable.Name, "positive_document")).Single();

            var queryInputShape = queryInput.Shape;
            var documentInputShape = documentInput.Shape;

            var intputs = new Dictionary<Variable, Value>();
            var queryInputValue = CreateSequenceInput(queryInputShape, query);
            var documentInputValue = CreateSequenceInput(documentInputShape, document);
            intputs.Add(queryInput, queryInputValue);
            intputs.Add(documentInput, documentInputValue);

            var outputs = new Dictionary<Variable, Value>();
            var evalFunc = Function.Combine(new[] { queryRecurrenceOutput, positiveDocumentRecurrenceOutput, queryDocumentCosineDistance});

            var queryRecurrenceOutput1 = evalFunc.Outputs[0];
            var positiveDocumentRecurrenceOutput1 = evalFunc.Outputs[1];
            var queryDocumentCosineDistance1 = evalFunc.Outputs[2];
            outputs.Add(queryRecurrenceOutput1, null);
            outputs.Add(positiveDocumentRecurrenceOutput1, null);
            outputs.Add(queryDocumentCosineDistance1, null);

            {
                evalFunc.Evaluate(intputs, outputs, DeviceDescriptor.CPUDevice);
            }            

            var queryEmbeddingValue = outputs[queryRecurrenceOutput1];
            var documentEmbeddingValue = outputs[positiveDocumentRecurrenceOutput1];
            var queryDocumentCosineDistanceValue = outputs[queryDocumentCosineDistance1];

            var queryEmbeddingOutput = queryEmbeddingValue.GetDenseData<float>(queryRecurrenceOutput1);
            var documentEmbeddingOutput = documentEmbeddingValue.GetDenseData<float>(positiveDocumentRecurrenceOutput1);
            var queryDocumentCosineDistanceOutput = queryDocumentCosineDistanceValue.GetDenseData<float>(queryDocumentCosineDistance1);

            var d = Utils.Cosine(queryEmbeddingOutput[0].ToArray(), documentEmbeddingOutput[0].ToArray());

            //Console.WriteLine("queryDocumentCosineDistanceOutput = {0}", queryDocumentCosineDistanceOutput[0][0]);

            return d;
        }

        //public float CosineDistance1(string query, string document)
        //{
        //    var splice = model.RootFunction;
        //    var exp = splice.RootFunction.Arguments[0].Owner.RootFunction;
        //    var cosine_distance = exp.Arguments[0].Owner.RootFunction;
        //    var queryRecurrenceOutput = cosine_distance.Arguments[0];
        //    var positiveDocumentRecurrenceOutput = cosine_distance.Arguments[1];
        //    var queryDocumentCosineDistance = cosine_distance.Output;

        //    var queryInput = model.Arguments.Where(variable => string.Equals(variable.Name, "query")).Single();
        //    var documentInput = model.Arguments.Where(variable => string.Equals(variable.Name, "positive_document")).Single();

        //    var queryInputShape = queryInput.Shape;
        //    var documentInputShape = documentInput.Shape;

        //    var intputs = new Dictionary<Variable, Value>();
        //    var queryInputValue = CreateSequenceInput(queryInputShape, query);
        //    var documentInputValue = CreateSequenceInput(documentInputShape, document);
        //    intputs.Add(queryInput, queryInputValue);
        //    intputs.Add(documentInput, documentInputValue);

        //    var outputs = new Dictionary<Variable, Value>();
        //    outputs.Add(queryRecurrenceOutput, null);
        //    outputs.Add(positiveDocumentRecurrenceOutput, null);
        //    outputs.Add(queryDocumentCosineDistance, null);

        //    model.Evaluate(intputs, outputs, DeviceDescriptor.CPUDevice);

        //    var queryEmbeddingValue = outputs[queryRecurrenceOutput];
        //    var documentEmbeddingValue = outputs[positiveDocumentRecurrenceOutput];
        //    var queryDocumentCosineDistanceValue = outputs[queryDocumentCosineDistance];

        //    var queryEmbeddingOutput = queryEmbeddingValue.GetDenseData<float>(queryRecurrenceOutput);
        //    var documentEmbeddingOutput = documentEmbeddingValue.GetDenseData<float>(positiveDocumentRecurrenceOutput);
        //    var queryDocumentCosineDistanceOutput = queryDocumentCosineDistanceValue.GetDenseData<float>(queryDocumentCosineDistance);

        //    var d = Utils.Cosine(queryEmbeddingOutput[0].ToArray(), documentEmbeddingOutput[0].ToArray());

        //    Console.WriteLine("queryDocumentCosineDistanceOutput = {0}", queryDocumentCosineDistanceOutput[0][0]);

        //    return d;
        //}

        public Value CreateSequenceInput(NDShape sampleShape, string rawText)
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
                    }
                    sequence.AddRange(v);
                }
            }

            return Value.CreateSequence<float>(sampleShape, sequence, DeviceDescriptor.CPUDevice);
        }
    }
}
