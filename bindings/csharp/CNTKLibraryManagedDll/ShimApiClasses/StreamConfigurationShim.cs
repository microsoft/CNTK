using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class StreamConfiguration
    {
        // TODO: do we need to handle special dimension values
        public StreamConfiguration(string streamName, int dim,
            bool isSparse = false, string streamAlias = "", bool definesMbSize = false)
            : this(streamName, (uint)dim, isSparse, streamAlias, definesMbSize)
        {
        }
    }
}
