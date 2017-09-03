//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// StreamConfigurationShim.cs -- C# Api for CNTK StreamConfiguration class
//
namespace CNTK
{
    public partial class StreamConfiguration
    {
        /// <summary>
        /// construct a stream configuration
        /// TODO: do we need to handle special dimension values
        /// </summary>
        /// <param name="streamName">name of the stream</param>
        /// <param name="dim">dimension of the stream</param>
        /// <param name="isSparse">whether the stream is sparse</param>
        /// <param name="streamAlias">alias of the stream </param>
        /// <param name="definesMbSize"></param>
        public StreamConfiguration(string streamName, int dim,
            bool isSparse = false, string streamAlias = "", bool definesMbSize = false)
            : this(streamName, (uint)dim, isSparse, streamAlias, definesMbSize)
        {
        }
    }
}
