import li_boost_shared_ptr_bits.*;

public class li_boost_shared_ptr_bits_runme {
  static {
    try {
        System.loadLibrary("li_boost_shared_ptr_bits");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    VectorIntHolder v = new VectorIntHolder();
    v.add(new IntHolder(11));
    v.add(new IntHolder(22));
    v.add(new IntHolder(33));

    int sum = li_boost_shared_ptr_bits.sum(v);
    if (sum != 66)
      throw new RuntimeException("sum is wrong");

    HiddenDestructor hidden = HiddenDestructor.create();
    hidden.delete();

    HiddenPrivateDestructor hiddenPrivate = HiddenPrivateDestructor.create();
    if (HiddenPrivateDestructor.getDeleteCount() != 0)
      throw new RuntimeException("Count should be zero");
    hiddenPrivate.delete();
    if (HiddenPrivateDestructor.getDeleteCount() != 1)
      throw new RuntimeException("Count should be one");
  }
}
