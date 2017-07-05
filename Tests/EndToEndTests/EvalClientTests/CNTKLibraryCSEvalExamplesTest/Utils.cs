using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NewsInsightEvaluation
{
    class Utils
    {
        public static float Cosine(float[] src, float[] target)
        {
            int len = src.Length;
            float sum_s = 0;
            float sum_t = 0;
            float sum_m_t = 0;
            for (int i = 0; i < len; i++)
            {
                sum_s += src[i] * src[i];
                sum_t += target[i] * target[i];
                sum_m_t += src[i] * target[i];
            }

            //return sum_m_t / (float)(Math.Sqrt(sum_s) * Math.Sqrt(sum_t));
            return (float)(sum_m_t * 1.0f / (Math.Sqrt((float)(sum_s * sum_t)) + Data.DSSMEpsilon));
        }

        public static void Print(float[] v)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("[{0}]=[{1}", v.Length, v[0]);
            for (int i = 1; i < v.Length; i++)
            {
                sb.AppendFormat(",{0}", v[i]);
            }

            sb.Append("]");

            Console.WriteLine(sb.ToString());
        }

        public static string GetFileName(string file)
        {
            if (!string.IsNullOrWhiteSpace(file))
            {
                var items = file.Split('\\');
                return items[items.Length - 1].Trim();
            }

            return string.Empty;
        }
    }
}
