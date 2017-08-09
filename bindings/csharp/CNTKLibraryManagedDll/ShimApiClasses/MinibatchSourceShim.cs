//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// MinibatchSourceShim.cs -- C# Api for CNTK MinibatchSource class
//
using System;
using System.Collections.Generic;
using System.Linq;

namespace CNTK
{
    public partial class MinibatchSource
    {
        public static MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            ulong epochSize, bool randomize, ulong randomizationWindow, bool sampleBasedRandomizationWindow = false)
        {
            var streamConfigsVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceInternal(dataFilePath, streamConfigsVector,
                epochSize, randomize, randomizationWindow, sampleBasedRandomizationWindow);
        }

        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceInternal(dataFilePath, streamConfigurationVector);
        }

        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            ulong epochSize)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceInternal(dataFilePath, streamConfigurationVector, epochSize);
        }

        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            ulong epochSize, bool randomize)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceInternal(dataFilePath, streamConfigurationVector, epochSize, randomize);
        }

        public static ulong FullDataSweep
        {
            get { return GetFullDataSweep(); }
        }

        public static ulong InfinitelyRepeat
        {
            get { return GetInfinitelyRepeat(); }
        }

        public static ulong DefaultRandomizationWindowInChunks
        {
            get { return GetDefaultRandomizationWindowInChunks(); }
        }

        public static void ComputeInputPerDimMeansAndInvStdDevs(MinibatchSource minibatchSource,
            IDictionary<StreamInformation, Tuple<NDArrayView, NDArrayView>> computedMeanAndVariances,
            DeviceDescriptor device)
        {
            UnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr mapStreamInfoToNDArrayPair =
                Helper.AsUnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr(computedMeanAndVariances);
            CNTKLib.ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, mapStreamInfoToNDArrayPair, device);

            foreach (StreamInformation s in computedMeanAndVariances.Keys.ToList())
            {
                computedMeanAndVariances[s] = new Tuple<NDArrayView, NDArrayView>(mapStreamInfoToNDArrayPair[s].first, mapStreamInfoToNDArrayPair[s].second);
            }
        }
    }
}
