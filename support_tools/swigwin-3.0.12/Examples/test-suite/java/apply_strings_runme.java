import apply_strings.*;

public class apply_strings_runme {

  static {
    try {
	System.loadLibrary("apply_strings");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static String TEST_MESSAGE = "A message from target language to the C++ world and back again.";

  public static void main(String argv[]) {
    if (!apply_strings.UCharFunction(TEST_MESSAGE).equals(TEST_MESSAGE)) throw new RuntimeException("UCharFunction failed");
    if (!apply_strings.SCharFunction(TEST_MESSAGE).equals(TEST_MESSAGE)) throw new RuntimeException("SCharFunction failed");
    SWIGTYPE_p_char pChar = apply_strings.CharFunction(null);
    if (pChar != null) throw new RuntimeException("CharFunction failed");
  }
}


