import li_std_vector.*;

public class li_std_vector_runme {

  static {
    try {
        System.loadLibrary("li_std_vector");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    IntVector v1 = li_std_vector.vecintptr(new IntVector());
    IntPtrVector v2 = li_std_vector.vecintptr(new IntPtrVector());
    IntConstPtrVector v3 = li_std_vector.vecintconstptr(new IntConstPtrVector());

    v1.add(123);
    if (v1.get(0) != 123) throw new RuntimeException("v1 test failed");

    StructVector v4 = li_std_vector.vecstruct(new StructVector());
    StructPtrVector v5 = li_std_vector.vecstructptr(new StructPtrVector());
    StructConstPtrVector v6 = li_std_vector.vecstructconstptr(new StructConstPtrVector());

    v4.add(new Struct(12));
    v5.add(new Struct(34));
    v6.add(new Struct(56));

    Struct s = null;
    if (v4.get(0).getNum() != 12) throw new RuntimeException("v4 test failed");
    if (v5.get(0).getNum() != 34) throw new RuntimeException("v5 test failed");
    if (v6.get(0).getNum() != 56) throw new RuntimeException("v6 test failed");
  }
}
