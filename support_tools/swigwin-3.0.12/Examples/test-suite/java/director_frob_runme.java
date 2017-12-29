
import director_frob.*;
import java.lang.reflect.*;

public class director_frob_runme
{
  static {
    try {
      System.loadLibrary("director_frob");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String args[])
  {
    Bravo foo = new Bravo();

    String s = foo.abs_method();
    if (!s.equals("Bravo::abs_method()"))
      throw new RuntimeException( "error" );

    Prims prims = new PrimsDerived();
    java.math.BigInteger bi = prims.callull(200, 50);
    java.math.BigInteger biCheck = new java.math.BigInteger("150");
    if (bi.compareTo(biCheck) != 0)
      throw new RuntimeException( "failed got:" + bi);
  }
}

class PrimsDerived extends Prims {
  PrimsDerived() {
    super();
  }
  public java.math.BigInteger ull(java.math.BigInteger i, java.math.BigInteger j) {
    return i.subtract(j);
  }
}
