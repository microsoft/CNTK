using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NewsInsightEvaluation
{
    class Vocabulary
    {
        public static ConcurrentDictionary<string, int> TriLetterGramStatistic =
            new ConcurrentDictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        public static ConcurrentBag<string> OutOfVocabulary = new ConcurrentBag<string>();

        private static Dictionary<string, int> fullVocaublary = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        public static void Init(string file)
        {
            int index = 0;
            var lines = File.ReadAllLines(file);
            foreach (var line in lines)
            {
                fullVocaublary.Add(line.Trim(), index++);
            }
        }

        public int this[string s]
        {
            get
            {
                TriLetterGramStatistic.AddOrUpdate(s, 1, (k, v) => v + 1);

                if (fullVocaublary.ContainsKey(s))
                {
                    return fullVocaublary[s];
                }
                else
                {
                    OutOfVocabulary.Add(s);
                    //throw new Exception("Vocabulary doesn't contain token " + s);

                    return -1;
                }
            }
        }
    }
}
