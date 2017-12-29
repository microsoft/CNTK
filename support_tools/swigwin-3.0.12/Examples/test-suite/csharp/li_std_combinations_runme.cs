using System;
using li_std_combinationsNamespace;

public class li_std_combinations_runme {
  public static void Main() {
    VectorPairIntString vpis = new VectorPairIntString();
    vpis.Add(new PairIntString(123, "one hundred and twenty three"));

    VectorString vs = new VectorString();
    vs.Add("hi");
    PairIntVectorString pivs = new PairIntVectorString(456, vs);
    if (pivs.second[0] != "hi")
      throw new ApplicationException("PairIntVectorString");

    VectorVectorString vvs = new VectorVectorString();
    vvs.Add(vs);

    PairIntPairIntString pipis = new PairIntPairIntString(12, new PairIntString(3, "4"));
    if (pipis.first != 12)
      throw new ApplicationException("PairIntPairIntString");

    PairDoubleString pds = new PairDoubleString(12.34, "okay");
    VectorPairDoubleString vpds = new VectorPairDoubleString();
    vpds.Add(pds);

    // Check SWIG_STD_VECTOR_ENHANCED macro - it provides the Contains method
    if (!vpds.Contains(pds))
      throw new ApplicationException("VectorPairDoubleString");
  }
}

