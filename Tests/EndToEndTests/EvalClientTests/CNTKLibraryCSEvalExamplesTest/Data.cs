using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NewsInsightEvaluation
{
    class Data
    {
        public static float DSSMEpsilon = 0.00000001f;

        public static Vocabulary Vocabulary = new Vocabulary();

        public static void Init(string vocabularyFile)
        {
            Vocabulary.Init(vocabularyFile);
        }

        public static void OutputVocabularyStatistic()
        {
            var list = Vocabulary.TriLetterGramStatistic.ToList();
            list.Sort((l, r) =>
            {
                return r.Value.CompareTo(l.Value);
            });

            using (var writer = File.CreateText(DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss") + ".vocabulary.txt"))
            {
                for (int i = 0; i < list.Count; i++)
                {
                    writer.WriteLine("{0}\t{1}", list[i].Key, list[i].Value);
                }

                writer.Close();
            }
        }
    }
}
