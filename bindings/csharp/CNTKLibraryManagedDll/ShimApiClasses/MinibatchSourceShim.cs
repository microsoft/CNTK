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
        /// <summary>
        /// Instantiate the CNTK built-in text format minibatch source
        /// It is better to have default value for optional arguments to avoid method duplication. However, C# requires
        /// default value being compile time constant which is not the case here.
        /// </summary>
        /// <param name="dataFilePath"></param>
        /// <param name="streamConfigs"></param>
        /// <param name="epochSize"></param>
        /// <param name="randomize"></param>
        /// <param name="randomizationWindow"></param>
        /// <param name="sampleBasedRandomizationWindow"></param>
        /// <returns></returns>
        public static MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            ulong epochSize, bool randomize, ulong randomizationWindow, bool sampleBasedRandomizationWindow = false)
        {
            var streamConfigsVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceInternal(dataFilePath, streamConfigsVector,
                epochSize, randomize, randomizationWindow, sampleBasedRandomizationWindow);
        }

        /// <summary>
        /// Instantiate the CNTK built-in text format minibatch source
        /// </summary>
        /// <param name="dataFilePath"></param>
        /// <param name="streamConfigs"></param>
        /// <returns></returns>
        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceInternal(dataFilePath, streamConfigurationVector);
        }

        /// <summary>
        /// Instantiate the CNTK built-in text format minibatch source
        /// </summary>
        /// <param name="dataFilePath">the source file</param>
        /// <param name="streamConfigs">configuration of the stream</param>
        /// <param name="epochSize">epoch size</param>
        /// <returns></returns>
        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            ulong epochSize)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceInternal(dataFilePath, streamConfigurationVector, epochSize);
        }

        /// <summary>
        /// Instantiate the CNTK built-in text format minibatch source
        /// </summary>
        /// <param name="dataFilePath">the source file</param>
        /// <param name="streamConfigs">stream configuration</param>
        /// <param name="epochSize">epoch size</param>
        /// <param name="randomize">whether to randomize the data</param>
        /// <returns></returns>
        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            ulong epochSize, bool randomize)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceInternal(dataFilePath, streamConfigurationVector, epochSize, randomize);
        }

        /// <summary>
        /// a number constant indicates full data sweep of the source data
        /// </summary>
        public static ulong FullDataSweep
        {
            get { return GetFullDataSweep(); }
        }

        /// <summary>
        /// a number constant indicates repeating data sweeping
        /// </summary>
        public static ulong InfinitelyRepeat
        {
            get { return GetInfinitelyRepeat(); }
        }

        /// <summary>
        /// the default randomization window 
        /// </summary>
        public static ulong DefaultRandomizationWindowInChunks
        {
            get { return GetDefaultRandomizationWindowInChunks(); }
        }

        /// <summary>
        /// Compute the per dimension means and variances for each of the specified streams using data from the specified minibatchSource.
        /// </summary>
        /// <param name="minibatchSource">the minibatch source</param>
        /// <param name="computedMeanAndVariances">return of the statistics</param>
        /// <param name="device">computing device</param>
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
