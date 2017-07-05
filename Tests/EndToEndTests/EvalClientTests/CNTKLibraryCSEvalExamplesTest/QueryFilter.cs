using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NewsInsightEvaluation
{
    class QueryFilter
    {
        // remove query for category rt_MaxClass
        // remove site news query

        public static HashSet<string> query_black_list = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        public static void Init(string category_query_file, string sitenews_query_file)
        {
            // category rt_MaxClass
            {
                var lines = File.ReadAllLines(category_query_file);
                foreach (var line in lines)
                {
                    if (!line.StartsWith("#"))
                    {
                        var items = line.Split(',');
                        var category = items[1].Trim();
                        if (category == "rt_MaxClass")
                        {
                            query_black_list.Add(items[0].Trim());
                        }
                    }
                }
            }

            // site news query
            {
                var lines = File.ReadAllLines(sitenews_query_file);
                foreach (var line in lines)
                {
                    if (line.StartsWith("#"))
                    {
                        var items = line.Split(',');
                        var query = items[0].Trim();
                        query_black_list.Add(query);
                    }
                }
            }
        }
    }
}
