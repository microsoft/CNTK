using System;
using System.Collections.Generic;
using System.Linq;

namespace CNTK
{
    public sealed class Sequence<T>: List<T>
    {
        public Sequence(NDShape shape)
        {
            Shape = shape;
            shapeSize = Shape.TotalSize;
        }

        public NDShape Shape { get; private set; }

        public void AddSample(IEnumerable<T> data)
        {
            if (data.Count() / shapeSize != 0)
            {
                throw new ArgumentException("The number of data does not match sample size. It should be multiple times of " + shapeSize);
            }

            this.AddRange(data);
        }

        // Todo: add sample-based iterator to get sample data.

        private uint shapeSize;
    }
}
