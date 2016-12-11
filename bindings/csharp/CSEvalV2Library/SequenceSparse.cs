using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    //
    // Represent all samples of a sequence (with variable length) in sparse CSC format.
    // Each column represents a tensor. If the tensor has n-dimension, flattened into 1 dimension.
    // Each raw represent a sample.
    // E.g. a sequence of three sparse vectors with 2 / 4 / 2 non-zero values
    // could be represented as the following:
    // colIndices:  0   2       6   8
    //          v   v       v   v
    // indices  1 3 2 3 5 6 2 7
    // data     0 1 2 3 4 5 6 7
    // See http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc
    //
    public class SequenceSparse<T>
    {
        public SequenceSparse(NDShape shape)
        {
            Shape = shape;
            ShapeSize = shape.TotalSize;
        }

        public SequenceSparse(NDShape shape, List<T> data, List<uint> indices, List<uint> colIndices)
        {
            Shape = shape;
            ShapeSize = shape.TotalSize;
            Data = data;
            Indices = indices;
            ColIndicies = colIndices;
        }

        public void AddSample(List<T> data, List<uint> indicies)
        {
            throw new NotImplementedException("Not implemented");
        }
        
        public NDShape Shape { get; private set; }

        //
        // All elements of a sequece. 
        // The number of samples is indicated by ColIndicies
        //
        private List<T> Data;

        //
        // For each element in Data, an entry in Indicies gives its position.
        // For each sample, the entries must be ascending.
        //
        private List<uint> Indices;

        // 
        // Contains numberOfSamples + 1 entries. The first entry is always 0.
        // The last entry points after the last element.
        // Each entry gives the start position in Data for each sample.
        // 
        private List<uint> ColIndicies;

        private uint ShapeSize;
    }
}
