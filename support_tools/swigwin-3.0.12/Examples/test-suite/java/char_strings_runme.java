import char_strings.*;

public class char_strings_runme {

  static {
    try {
	System.loadLibrary("char_strings");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static String CPLUSPLUS_MSG = "A message from the deep dark world of C++, where anything is possible.";
  private static String OTHERLAND_MSG = "Little message from the safe world.";

  public static void main(String argv[]) {

    long count = 10000;
    long i = 0;

    // get functions
    for (i=0; i<count; i++) {
      String str = char_strings.GetCharHeapString();
      if (!str.equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test char get 1 failed, iteration " + i);
      char_strings.DeleteCharHeapString();
    }

    for (i=0; i<count; i++) {
      String str = char_strings.GetConstCharProgramCodeString();
      if (!str.equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test char get 2 failed, iteration " + i);
      char_strings.DeleteCharHeapString();
    }

    for (i=0; i<count; i++) {
      String str = char_strings.GetCharStaticString();
      if (!str.equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test char get 3 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      String str = char_strings.GetCharStaticStringFixed();
      if (!str.equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test char get 4 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      String str = char_strings.GetConstCharStaticStringFixed();
      if (!str.equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test char get 5 failed, iteration " + i);
    }

    // set functions
    for (i=0; i<count; i++) {
      if (!char_strings.SetCharHeapString(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char set 1 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetCharStaticString(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char set 2 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetCharArrayStaticString(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char set 3 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharHeapString(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char set 4 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharStaticString(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char set 5 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharArrayStaticString(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char set 6 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetCharConstStaticString(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char set 7 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharConstStaticString(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char set 8 failed, iteration " + i);
    }

    // get set function
    for (i=0; i<count; i++) {
      String ping = OTHERLAND_MSG + i;
      String pong = char_strings.CharPingPong(ping);
      if (!ping.equals(pong))
        throw new RuntimeException("Test PingPong 1 failed.\nExpected:" + ping + "\nReceived:" + pong);
    }

    // variables
    for (i=0; i<count; i++) {
      char_strings.setGlobal_char(OTHERLAND_MSG + i);
      if (!char_strings.getGlobal_char().equals(OTHERLAND_MSG + i))
        throw new RuntimeException("Test variables 1 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      char_strings.setGlobal_char_array1(OTHERLAND_MSG + i);
      if (!char_strings.getGlobal_char_array1().equals(OTHERLAND_MSG + i))
        throw new RuntimeException("Test variables 2 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      char_strings.setGlobal_char_array2(OTHERLAND_MSG + i);
      if (!char_strings.getGlobal_char_array2().equals(OTHERLAND_MSG + i))
        throw new RuntimeException("Test variables 3 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.getGlobal_const_char().equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test variables 4 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.getGlobal_const_char_array1().equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test variables 5 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.getGlobal_const_char_array2().equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test variables 6 failed, iteration " + i);
    }

    // char *& tests
    for (i=0; i<count; i++) {
      String str = char_strings.GetCharPointerRef();
      if (!str.equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test char pointer ref get failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetCharPointerRef(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test char pointer ref set failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      String str = char_strings.GetConstCharPointerRef();
      if (!str.equals(CPLUSPLUS_MSG))
        throw new RuntimeException("Test const char pointer ref get failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharPointerRef(OTHERLAND_MSG + i, i))
        throw new RuntimeException("Test const char pointer ref set failed, iteration " + i);
    }
  }
}


