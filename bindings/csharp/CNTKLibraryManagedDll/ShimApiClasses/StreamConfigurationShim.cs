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
        // TODO: do we need to handle special dimension values
        public StreamConfiguration(string streamName, int dim,
            bool isSparse = false, string streamAlias = "", bool definesMbSize = false)
            : this(streamName, (uint)dim, isSparse, streamAlias, definesMbSize)
        {
        }
    }
}
