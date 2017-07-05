using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace NewsInsightEvaluation
{
    class Validation
    {
        public int positive_document_count_per_sample = 1;
        public int negative_document_count = 1;

        public int model_parallel_count = 1;

        private long skipped_by_query_count = 0;
        private long load_processed_count = 0;
        private Stopwatch cost_timer = new Stopwatch();

        public object console_output_lock = new object();

        private StreamWriter log_writer;

        ConcurrentDictionary<string, int> total_query_statistics =
            new ConcurrentDictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        ConcurrentDictionary<string, int> total_sample_statistics =
            new ConcurrentDictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        public List<Tuple<string, string>> cleaned_total_training_data = new List<Tuple<string, string>>();

        public string raw_data_path = string.Empty;

        public int validation_sample_count = 0;

        public void Process()
        {
            try
            {
                this.BuildSamples();
            }
            catch (Exception e)
            {
                this.Print(e.ToString());
            }
        }

        public void BuildSamples()
        {

            this.result_root_path = "Validate_" + DateTime.Now.ToString("yyyy-MM-dd-H-mm-sss");
            Directory.CreateDirectory(result_root_path);

            this.log_writer = File.CreateText(DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss") + ".data.log");

            Stopwatch total_timer = new Stopwatch();
            Stopwatch stage_timer = new Stopwatch();
            total_timer.Start();
            stage_timer.Start();
            /// 1
            this.Print("1. start load and clean data...");
            this.Load();
            this.Print(string.Format("...load queries count = {0}, samples count = {1}",
                this.total_query_statistics.Count, this.cleaned_total_training_data.Count));
            this.Print(string.Format("...step complete, cost time : {0}", stage_timer.Elapsed.ToString()));

            /// 2
            stage_timer.Restart();
            this.Print("2. start shuffle data ...");
            var shuffled_data = this.Shuffle();
            this.Print(string.Format("...step complete, cost time : {0}", stage_timer.Elapsed.ToString()));

            /// 3
            stage_timer.Restart();
            this.Print("3. generate validate sample ...");
            this.validation_samples = this.GenerateValidateSamples(shuffled_data);

            this.Print(string.Format("...step complete, cost time : {0}", stage_timer.Elapsed.ToString()));
            this.Print(string.Format("...all  complete, cost time : {0}", cost_timer.Elapsed.ToString()));
        }

        public string result_root_path;
        public List<Sample> validation_samples = new List<Sample>();

        public void Validate(ModelEvaluation model)
        {
            Console.WriteLine("---------------------------------------------------------------------");
            Console.WriteLine("start validate model : {0}", model.name);
            Stopwatch stage_timer = Stopwatch.StartNew();

            var positive_record = new ConcurrentBag<Tuple<float, string, string>>();
            var negative_record = new ConcurrentBag<Tuple<float, string, string>>();

            var cursor_top = Console.CursorTop;
            int validate_sample_count = 0;

            var parallel_model_list_1 = new BlockingCollection<ModelEvaluation>();
            // var parallel_model_list = new List<ModelEvaluation>();
            for (int i = 0; i < model_parallel_count; i++)
            {
                // parallel_model_list.Add(model.CloneByFile());

                parallel_model_list_1.Add(model.Clone());  //CloneByFile()); // model.Clone());
            }

            //var total_test_sample_count = validation_samples.Count;
            //int process_unit_len = (total_test_sample_count + parallel_model_list.Count - 1) / parallel_model_list.Count;
            //Parallel.For(0, parallel_model_list.Count, (i) =>
            //{
            //    for (int k = 0; k < process_unit_len; k++)
            //    {
            //        int sample_index = i * process_unit_len + k;
            //        if (sample_index < total_test_sample_count)
            //        {
            //            var sample = validation_samples[sample_index];
            //            sample.Validate(parallel_model_list[i]);
            //            sample.RecordPositiveDocument(positive_record);
            //            sample.RecordNegativeDocument(negative_record);

            //            System.Threading.Interlocked.Increment(ref validate_sample_count);
            //            this.Print(cursor_top, string.Format("...validated {0} sample...", validate_sample_count));
            //        }
            //    }
            //});

            var samples = new List<int>(50);
            for (int i = 0; i < 50; i++)
                samples.Add(i);
            Parallel.ForEach(samples, new ParallelOptions() { MaxDegreeOfParallelism = samples.Count }, (sample) =>
            // Parallel.ForEach(validation_samples, new ParallelOptions() { MaxDegreeOfParallelism = validation_samples.Count }, (sample) =>
            {
                var m = parallel_model_list_1.Take();
                try
                {
                    m.CosineDistance("24 hours of le mans", "live stream for one of motorsport s biggest events ground to a halt by cyberattacks");
                    // sample.Validate(m);
                    //sample.RecordPositiveDocument(positive_record);
                    //sample.RecordNegativeDocument(negative_record);
                    System.Threading.Interlocked.Increment(ref validate_sample_count);
                    this.Print(cursor_top, string.Format("...validated {0} sample...", validate_sample_count));
                }
                finally
                {
                    parallel_model_list_1.Add(m);
                }
            });

            Console.WriteLine();
            this.Print(string.Format("...step complete, cost time : {0}", stage_timer.Elapsed.ToString()));

            /// 5
            stage_timer.Restart();
            Console.WriteLine("total positive sample count = {0}", positive_record.Count);
            Console.WriteLine("total negative sample count = {0}", negative_record.Count);

            cursor_top = Console.CursorTop;
            using (var positive_writer = File.CreateText(Path.Combine(result_root_path, model.name + "_positive_sample_validate_result.txt")))
            using (var negative_writer = File.CreateText(Path.Combine(result_root_path, model.name + "_negative_sample_validate_result.txt")))
            {

                var positive_samples = positive_record.ToList();
                positive_samples.Sort((l, r) =>
                {
                    return r.Item1.CompareTo(l.Item1);
                });

                for (int i = 0; i < positive_samples.Count; i++)
                {
                    positive_writer.WriteLine("{0}\t{1}\t{2}", positive_samples[i].Item1, positive_samples[i].Item2, positive_samples[i].Item3);
                    this.Print(cursor_top, string.Format("...output {0} positive sample...", i + 1));
                }
                positive_writer.Close();
                Console.WriteLine();

                cursor_top = Console.CursorTop;
                var negative_samples = negative_record.ToList();
                negative_samples.Sort((l, r) =>
                {
                    return r.Item1.CompareTo(l.Item1);
                });

                for (int i = 0; i < negative_samples.Count; i++)
                {
                    negative_writer.WriteLine("{0}\t{1}\t{2}", negative_samples[i].Item1, negative_samples[i].Item2, negative_samples[i].Item3);
                    this.Print(cursor_top, string.Format("...output {0} negative sample...", i + 1));
                }

                negative_writer.Close();
                Console.WriteLine();
            }

            this.Print(string.Format("...step complete, cost time : {0}", stage_timer.Elapsed.ToString()));
        }

        private List<Sample> GenerateValidateSamples(List<Tuple<string, string>> full_shuffle_document)
        {
            var full_validation_sample_bag = new ConcurrentBag<Sample>();
            Parallel.ForEach(full_shuffle_document, (pair) =>
            {
                full_validation_sample_bag.Add(new Sample
                {
                    Query = pair.Item1,
                    PositiveDocument = new Dictionary<string, float>
                    {
                        { pair.Item2, 0 }
                    }
                });
            });

            var full_validation_sample = full_validation_sample_bag.ToList();
            // random sample negative sample
            int step = 3;
            var total_training_sample_count = full_validation_sample_bag.Count;
            Parallel.For(0, total_training_sample_count, (i) =>
            {
                var current_sample = full_validation_sample[i];
                int start = i;
                List<int> negative_sample_index = new List<int>();
                while (current_sample.NegativeDocument.Count < this.negative_document_count)
                {
                    start += step;
                    var index = start % total_training_sample_count;
                    var positive_document = full_validation_sample[index].PositiveDocument.First().Key;
                    if (!full_validation_sample[index].IsSampeQuery(current_sample) &&
                        !current_sample.NegativeDocument.ContainsKey(positive_document))
                    {
                        current_sample.NegativeDocument.Add(positive_document, 0);
                    }
                }
            });

            if (this.validation_sample_count >= full_validation_sample.Count)
            {
                return full_validation_sample;
            }

            List<Sample> selected_sample = new List<Sample>();
            HashSet<int> selected_sample_index = new HashSet<int>();
            Random rand = new Random();
            while (selected_sample_index.Count < this.validation_sample_count)
            {
                var index = rand.Next(full_validation_sample.Count);
                if (!selected_sample_index.Contains(index))
                {
                    selected_sample_index.Add(index);
                    selected_sample.Add(full_validation_sample[index]);
                }
            }

            return selected_sample;
        }

        private List<Tuple<string, string>> Shuffle()
        {
            int total_sample_count = this.cleaned_total_training_data.Count;
            int block_size = 100;
            int block_count = (int)((total_sample_count + block_size - 1) / block_size);

            // overall shuffle
            var overall_shuffle_temp = new ConcurrentBag<Tuple<string, string>>[block_count];
            for (int i = 0; i < block_count; i++)
            {
                overall_shuffle_temp[i] = new ConcurrentBag<Tuple<string, string>>();
            }

            Random rand = new Random();
            Parallel.For(0, total_sample_count, new ParallelOptions { MaxDegreeOfParallelism = 256 }, (i) =>
            {
                var random_block_id = rand.Next(block_count);
                overall_shuffle_temp[random_block_id].Add(this.cleaned_total_training_data[i]);
            });

            // inside shuffle
            var overall_inside_shuffle_temp = new ConcurrentBag<Tuple<string, string>[]>();
            Parallel.ForEach(overall_shuffle_temp, new ParallelOptions { MaxDegreeOfParallelism = block_count }, (temp) =>
            {
                Random random = new Random();
                var block = temp.ToArray();
                var len = block.Length;
                for (int i = 0; i < block.Length; i++)
                {
                    int r = random.Next(len);
                    var t = block[i];
                    block[i] = block[r];
                    block[r] = t;
                }

                overall_inside_shuffle_temp.Add(block);
            });

            List<Tuple<string, string>> full_shuffle_document = new List<Tuple<string, string>>();
            foreach (var block in overall_inside_shuffle_temp)
            {
                full_shuffle_document.AddRange(block);
            }

            return full_shuffle_document;
        }

        private void Load()
        {
            if (!Directory.Exists(this.raw_data_path))
            {
                return;
            }

            cost_timer.Start();
            var statistic_result = new ConcurrentBag<Tuple<string, long, long>>();
            var cleaned_data = new ConcurrentBag<Tuple<string, string>>();
            var cursor_top = Console.CursorTop;
            var files = Directory.GetFiles(this.raw_data_path).ToList();
            files.Sort();
            //foreach (var file in files)
            //{
            //    LoadFile(file, cleaned_data);
            //}

            //Console.WriteLine("load from {0} : {1} files", this.raw_data_path, files.Count);
            Parallel.For(0, files.Count, new ParallelOptions { MaxDegreeOfParallelism = files.Count }, (i) =>
            //foreach (var file in files)
            {
                var file = files[i];
                long load_query_count, load_sample_count;
                LoadFile(file, cursor_top + i, cleaned_data, out load_query_count, out load_sample_count);

                statistic_result.Add(new Tuple<string, long, long>(Utils.GetFileName(file), load_query_count, load_sample_count));
            }
           );

            //OutputStatistics(total_query_statistics, "total_query_statistics.txt");

            long total_topic_count = 0, total_sample_count = 0;
            foreach (var r in statistic_result)
            {
                total_topic_count += r.Item2;
                total_sample_count += r.Item3;

                //Console.WriteLine("... {0} topic count = {1}, sample count = {2}", r.Item1, r.Item2, r.Item3);
            }

            Console.SetCursorPosition(0, cursor_top + files.Count);
            Console.WriteLine("...total load topic count = {0}, sample count = {1}", total_topic_count, total_sample_count);

            this.cleaned_total_training_data = cleaned_data.ToList();
        }

        private void LoadFile(string file, int cursor_top, ConcurrentBag<Tuple<string, string>> result, out long load_query_count, out long load_sample_count)
        {
            ConcurrentDictionary<string, int> query_statistics =
                new ConcurrentDictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            ConcurrentBag<Tuple<string, string>> qa = new ConcurrentBag<Tuple<string, string>>();
            var query_prefix = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "when", "who", "where", "what", "why", "how", "Is"
            };

            long processed_count = 0;
            var lines = File.ReadAllLines(file);

            Console.WriteLine("total line count = {0}", lines.Length);

            cursor_top++;

            Parallel.ForEach(lines/*, new ParallelOptions { MaxDegreeOfParallelism = 256 }*/, line =>
            {
                var items = line.Split('\t');
                if (items.Length >= 2)
                {
                    var query = items[0].Trim().ToLowerInvariant();
                    if (query.StartsWith("site:") ||
                        query.StartsWith("http:") ||
                        query.StartsWith("https:") ||
                        QueryFilter.query_black_list.Contains(query))
                    {
                        System.Threading.Interlocked.Increment(ref skipped_by_query_count);
                        return;
                    }

                    query_statistics.AddOrUpdate(query, 1, (k, v) => v + 1);
                    total_query_statistics.AddOrUpdate(query, 1, (k, v) => v + 1);

                    string q = TextUtils.N1Normalize(items[0]);
                    string d = TextUtils.N1Normalize(items[1]);
                    var sample_key = q + d;
                    if (!string.IsNullOrWhiteSpace(q) && !string.IsNullOrWhiteSpace(d) && !total_sample_statistics.ContainsKey(sample_key))
                    {
                        result.Add(new Tuple<string, string>(q, d));

                        total_sample_statistics.AddOrUpdate(sample_key, 1, (k, v) => v + 1);
                    }

                    {
                        var qi = q.Split(' ');
                        if (qi.Length > 0 && query_prefix.Contains(qi[0]))
                        {
                            qa.Add(new Tuple<string, string>(items[0], items[1]));
                        }
                    }

                    System.Threading.Interlocked.Increment(ref processed_count);
                    System.Threading.Interlocked.Increment(ref load_processed_count);
                    if (load_processed_count % 100000 == 0)
                    {
                        this.Print(cursor_top, string.Format("... load and clean sample count:{0},speed:{1}/ms,skip:{2},qa:{3} {4:f4}",
                            load_processed_count,
                            load_processed_count / cost_timer.ElapsedMilliseconds,
                            skipped_by_query_count,
                            qa.Count,
                            (double)qa.Count / (double)result.Count));
                    }
                }
            });

            // var result_file = Utils.GetFileName(file, "statistics");
            //OutputStatistics(topic_statistics, result_file);
            //Console.WriteLine();
            //Console.WriteLine("total = {0}, qa count = {1}, {2:f4}", result.Count, qa.Count, (double)qa.Count / (double)result.Count);
            //using (var writer = File.CreateText("qa.txt"))
            //{
            //    foreach (var q in qa)
            //    {
            //        writer.WriteLine("{0}\t{1}", q.Item1, q.Item2);
            //    }

            //    writer.Close();
            //}

            //Console.Read();

            load_query_count = query_statistics.Count;
            load_sample_count = processed_count;
        }

        private void Print(int top, string msg)
        {
            lock (this.console_output_lock)
            {
                Console.SetCursorPosition(0, top);
                Console.Write(new string(' ', Console.WindowWidth));

                Console.SetCursorPosition(0, top);
                Console.Write(msg);
            }
        }
        private void Print(string m)
        {
            Console.WriteLine(m);
            this.log_writer.WriteLine(m);
            this.log_writer.Flush();
        }
    }
}
