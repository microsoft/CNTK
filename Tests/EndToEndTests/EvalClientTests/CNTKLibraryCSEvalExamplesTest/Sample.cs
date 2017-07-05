using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace NewsInsightEvaluation
{
    class Sample
    {
        public string Query;

        public Dictionary<string, float> PositiveDocument = new Dictionary<string, float>();

        public Dictionary<string, float> NegativeDocument = new Dictionary<string, float>();

        public void RecordPositiveDocument(ConcurrentBag<Tuple<float, string, string>> record)
        {
            foreach (var d in PositiveDocument)
            {
                record.Add(new Tuple<float, string, string>(d.Value, Query, d.Key));
            }
        }

        public void RecordNegativeDocument(ConcurrentBag<Tuple<float, string, string>> record)
        {
            foreach (var d in NegativeDocument)
            {
                record.Add(new Tuple<float, string, string>(d.Value, Query, d.Key));
            }
        }

        public void Validate(ModelEvaluation m)
        {
            {
                var keys = PositiveDocument.Keys.ToList();
                foreach (var key in keys)
                {
                    PositiveDocument[key] = m.CosineDistance(Query, key);
                }
            }

            {
                var keys = NegativeDocument.Keys.ToList();
                foreach (var key in keys)
                {
                    NegativeDocument[key] = m.CosineDistance(Query, key);
                }
            }
        }

        public override string ToString()
        {
            StringBuilder record = new StringBuilder();
            foreach (var d in PositiveDocument)
            {
                record.AppendFormat("{0}\t{1}\t{2}", d.Value, Query, d.Key).AppendLine();
            }

            foreach (var d in NegativeDocument)
            {
                record.AppendFormat("{0}\t{1}\t{2}", d.Value, Query, d.Key).AppendLine();
            }

            return record.ToString();
        }

        public bool IsSampeQuery(Sample s)
        {
            return string.Equals(this.Query, s.Query, StringComparison.OrdinalIgnoreCase);
        }
    }
}
