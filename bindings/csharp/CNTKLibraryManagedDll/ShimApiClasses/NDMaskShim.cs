using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class NDMask
    {
        // Property MaskedCount.
        public int MaskedCount
        {
            get { return (int)_MaskedCount(); }
        }

        // Property Device.
        public DeviceDescriptor Device
        {
            get { return _Device(); }
        }

        // Property Shape.
        public NDShape Shape
        {
            get { return _Shape(); }
        }

        // Invidates a section of a NDShape.
        public void InvalidateSection(System.Collections.Generic.IEnumerable<int> sectionOffset, NDShape sectionShape)
        {
            var offsetVector = Helper.AsSizeTVector(sectionOffset);
            _InvalidateSection(offsetVector, sectionShape);
        }

        // Marks sequence begin.
        public void MarkSequenceBegin(System.Collections.Generic.IEnumerable<int> offset)
        {
            var offsetVector = Helper.AsSizeTVector(offset);
            _MarkSequenceBegin(offsetVector);
        }

        // Marks sequence begins in a NDShape.
        public void MarkSequenceBegin(System.Collections.Generic.IEnumerable<int> offset, NDShape sectionShape)
        {
            var offsetVector = Helper.AsSizeTVector(offset);
            _MarkSequenceBegin(offsetVector, sectionShape);
        }
    }
}
