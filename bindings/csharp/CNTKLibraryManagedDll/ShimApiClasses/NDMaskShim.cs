//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NDMaskShim.cs -- C# Api for CNTK NDMask class
//
using System.Collections.Generic;

namespace CNTK
{
    public partial class NDMask
    {
        /// <summary>
        /// Property MaskedCount.
        /// </summary>
        public int MaskedCount
        {
            get { return (int)_MaskedCount(); }
        }

        /// <summary>
        /// Property Device.
        /// </summary>
        public DeviceDescriptor Device
        {
            get { return _Device(); }
        }

        /// <summary>
        /// Property Shape.
        /// </summary>
        public NDShape Shape
        {
            get { return _Shape(); }
        }

        /// <summary>
        /// Invidates a section of a NDShape.
        /// </summary>
        /// <param name="sectionOffset"></param>
        /// <param name="sectionShape"></param>
        public void InvalidateSection(IEnumerable<int> sectionOffset, NDShape sectionShape)
        {
            var offsetVector = Helper.AsSizeTVector(sectionOffset);
            _InvalidateSection(offsetVector, sectionShape);
        }

        /// <summary>
        /// Marks sequence begin.
        /// </summary>
        /// <param name="offset"></param>
        public void MarkSequenceBegin(IEnumerable<int> offset)
        {
            var offsetVector = Helper.AsSizeTVector(offset);
            _MarkSequenceBegin(offsetVector);
        }

        /// <summary>
        /// Marks sequence begins in a NDShape.
        /// </summary>
        /// <param name="offset"></param>
        /// <param name="sectionShape"></param>
        public void MarkSequenceBegin(IEnumerable<int> offset, NDShape sectionShape)
        {
            var offsetVector = Helper.AsSizeTVector(offset);
            _MarkSequenceBegin(offsetVector, sectionShape);
        }
    }
}
