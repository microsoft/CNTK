//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibShim.cs -- General C# Api methods
//
using System;
using System.Collections.Generic;

namespace CNTK
{
    public partial class CNTKLib
    {
        /// <summary>
        /// Create a crop transform with the specified options to be used with a reader
        /// </summary>
        /// <param name="cropType">type of crop</param>
        /// <param name="cropSize">crop size</param>
        /// <param name="sideRatio">side ratio</param>
        /// <param name="areaRatio">area ratio</param>
        /// <param name="aspectRatio">aspect ratio</param>
        /// <param name="jitterType">jitter type </param>
        /// <returns></returns>
        public static CNTKDictionary ReaderCrop(string cropType, Tuple<int, int> cropSize, Tuple<float, float> sideRatio,
            Tuple<float, float> areaRatio, Tuple<float, float> aspectRatio, string jitterType)
        {
            PairIntInt cropSizeSwig = new PairIntInt(cropSize.Item1, cropSize.Item2);
            PairFloatFloat sideRatioSwig = new PairFloatFloat(sideRatio.Item1, sideRatio.Item2);
            PairFloatFloat areaRatioSwig = new PairFloatFloat(areaRatio.Item1, areaRatio.Item2);
            PairFloatFloat aspectRatioSwig = new PairFloatFloat(aspectRatio.Item1, aspectRatio.Item2);
            return ReaderCrop(cropType, cropSizeSwig, sideRatioSwig, areaRatioSwig, aspectRatioSwig, jitterType);
        }

        /// <summary>
        /// Create an ImageDeserializer with the specified options
        /// </summary>
        /// <param name="fileName">source file</param>
        /// <param name="labelStreamName">label of the stream</param>
        /// <param name="numLabels">number of labels</param>
        /// <param name="imageStreamName">the image stream name</param>
        /// <param name="deserializers">deserializer configuration</param>
        /// <returns></returns>
        public static CNTKDictionary ImageDeserializer(string fileName, string labelStreamName, uint numLabels, string imageStreamName, IList<CNTKDictionary> deserializers)
        {
            DictionaryVector deserializersSwig = Helper.AsDictionaryVector(deserializers);
            return ImageDeserializer(fileName, labelStreamName, numLabels, imageStreamName, deserializersSwig);
        }

        /// <summary>
        /// build a convolution function 
        /// </summary>
        /// <param name="convolutionMap">convolution parameters (shape, type of the kernal)</param>
        /// <param name="operand">input variable</param>
        /// <param name="strides">strides to apply convolution</param>
        /// <param name="sharing">whether to share parameters (default = true)</param>
        /// <param name="autoPadding"></param>
        /// <returns></returns>
        public static Function Convolution(Variable convolutionMap, Variable operand, NDShape strides, IEnumerable<bool> sharing, IEnumerable<bool> autoPadding)
        {
            BoolVector sharingVec = Helper.AsBoolVector(sharing);
            BoolVector autoPaddingVec = Helper.AsBoolVector(autoPadding);
            return CNTKLib.Convolution(convolutionMap, operand, strides, sharingVec, autoPaddingVec);
        }

        /// <summary>
        /// create a pooling function
        /// </summary>
        /// <param name="operand">input</param>
        /// <param name="poolingType">pooling type</param>
        /// <param name="poolingWindowShape">pooiling window dimensions</param>
        /// <param name="strides">strides to apply the pooling</param>
        /// <param name="autoPadding"></param>
        /// <returns></returns>
        public static Function Pooling(Variable operand, PoolingType poolingType, NDShape poolingWindowShape, NDShape strides, IEnumerable<bool> autoPadding)
        {
            BoolVector autoPaddingVector = Helper.AsBoolVector(autoPadding);
            return Pooling(operand, poolingType, poolingWindowShape, strides, autoPaddingVector);
        }
    }
}
