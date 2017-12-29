
using System;
using apply_stringsNamespace;

public class apply_strings_runme {

  private static string TEST_MESSAGE = "A message from target language to the C++ world and back again.";

  public static void Main() {
    if (apply_strings.UCharFunction(TEST_MESSAGE) != TEST_MESSAGE) throw new Exception("UCharFunction failed");
    if (apply_strings.SCharFunction(TEST_MESSAGE) != TEST_MESSAGE) throw new Exception("SCharFunction failed");
    SWIGTYPE_p_char pChar = apply_strings.CharFunction(null);
    if (pChar != null) throw new Exception("CharFunction failed");
  }
}

