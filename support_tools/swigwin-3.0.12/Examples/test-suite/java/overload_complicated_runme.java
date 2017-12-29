
import overload_complicated.*;

public class overload_complicated_runme {

  static {
    try {
        System.loadLibrary("overload_complicated");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
      SWIGTYPE_p_int pInt = null;

      // Check the correct constructors are available
      Pop p = new Pop(pInt);

      p = new Pop(pInt, false);

      // Check overloaded in const only and pointers/references which target languages cannot disambiguate
      if (p.hip(false) != 701)
        throw new RuntimeException("Test 1 failed");

      if (p.hip(pInt) != 702)
        throw new RuntimeException("Test 2 failed");

      // Reverse the order for the above
      if (p.hop(pInt) != 805)
        throw new RuntimeException("Test 3 failed");

      if (p.hop(false) != 801)
        throw new RuntimeException("Test 4 failed");

      // Few more variations and order shuffled
      if (p.pop(false) != 901)
        throw new RuntimeException("Test 5 failed");

      if (p.pop(pInt) != 902)
        throw new RuntimeException("Test 6 failed");

      if (p.pop() != 905)
        throw new RuntimeException("Test 7 failed");

      // Overload on const only
      if (p.bop(pInt) != 1001)
        throw new RuntimeException("Test 8 failed");

      if (p.bip(pInt) != 2001)
        throw new RuntimeException("Test 9 failed");

      // Globals
      if (overload_complicated.muzak(false) != 3001)
        throw new RuntimeException("Test 10 failed");

      if (overload_complicated.muzak(pInt) != 3002)
        throw new RuntimeException("Test 11 failed");
  }
}

