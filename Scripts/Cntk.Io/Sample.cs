namespace Cntk.Io
{
    /// <summary>Sparse sample representation.</summary>
    public struct Sample
    {
        public int Index { get; }

        public float Value { get; }

        public Sample(int index, float value)
        {
            Index = index;
            Value = value;
        }
    }
}
