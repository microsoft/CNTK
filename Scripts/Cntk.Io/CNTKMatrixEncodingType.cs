namespace Cntk.Io
{
    /// <summary>Encoding strategy for an input stream.</summary>
    public enum CNTKMatrixEncodingType : byte
    {
        Dense = 0,
        SparseCSC = 1,
    }
}