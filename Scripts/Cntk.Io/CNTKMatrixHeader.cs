namespace Cntk.Io
{
    /// <summary>Header for one of the input stream.</summary>
    public class CNTKMatrixHeader
    {
        public string Name { get; }

        public CNTKMatrixEncodingType MatrixEncodingType { get; }

        public CNTKElementType ElementType { get; }

        public int SampleDimension { get; }

        public CNTKMatrixHeader(string name,
            CNTKMatrixEncodingType matrixEncodingType,
            CNTKElementType elementType,
            int sampleDimension)
        {
            Name = name;
            MatrixEncodingType = matrixEncodingType;
            ElementType = elementType;
            SampleDimension = sampleDimension;
        }
    }
}