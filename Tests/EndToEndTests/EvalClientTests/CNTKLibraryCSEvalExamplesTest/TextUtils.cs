using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

namespace NewsInsightEvaluation
{
    class TextUtils
    {
        private static Regex n_rg1 = new Regex("[^0-9a-z]", RegexOptions.Compiled);
        private static Regex n_rg2 = new Regex(@"\s+", RegexOptions.Compiled);

        public static string N1Normalize(string s)
        {
            string ss = s.ToLower();
            Regex rg1 = n_rg1;
            ss = rg1.Replace(ss, " ");
            Regex rg2 = n_rg2;
            ss = rg2.Replace(ss, " ");
            ss = ss.Trim();

            return ss;
        }

        public static Dictionary<int, int> EncodeWord2Letter3Gram(string word, Vocabulary v)
        {
            var letterGram = Word2LetterNGram(word, 3);
            return EncodeByVocabulary(letterGram, v);
        }

        public static Dictionary<int, int> EncodeByVocabulary(Dictionary<string, int> letterGram, Vocabulary v)
        {
            var result = new Dictionary<int, int>();
            foreach (var g in letterGram)
            {
                var id = v[g.Key];
                if (id >= 0)
                {
                    result[id] = g.Value;
                }
            }

            return result;
        }

        public static Dictionary<string, int> Word2LetterNGram(string word, int N)
        {
            var result = new Dictionary<string, int>();
            string src = "#" + word + "#";
            for (int i = 0; i <= src.Length - N; ++i)
            {
                string l3g = src.Substring(i, N);
                if (result.ContainsKey(l3g))
                {
                    result[l3g]++;
                }
                else
                {
                    result.Add(l3g, 1);
                }
            }

            return result;
        }
    }
}
