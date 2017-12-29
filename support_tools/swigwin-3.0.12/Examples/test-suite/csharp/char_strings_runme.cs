
using System;
using char_stringsNamespace;

public class char_strings_runme {

  private static string CPLUSPLUS_MSG = "A message from the deep dark world of C++, where anything is possible.";
  private static string OTHERLAND_MSG = "Little message from the safe world.";

  public static void Main() {

    uint count = 10000;
    uint i = 0;

    // get functions
    for (i=0; i<count; i++) {
      string str = char_strings.GetCharHeapString();
      if (str != CPLUSPLUS_MSG)
        throw new Exception("Test char get 1 failed, iteration " + i);
      char_strings.DeleteCharHeapString();
    }

    for (i=0; i<count; i++) {
      string str = char_strings.GetConstCharProgramCodeString();
      if (str != CPLUSPLUS_MSG)
        throw new Exception("Test char get 2 failed, iteration " + i);
      char_strings.DeleteCharHeapString();
    }

    for (i=0; i<count; i++) {
      string str = char_strings.GetCharStaticString();
      if (str != CPLUSPLUS_MSG)
        throw new Exception("Test char get 3 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      string str = char_strings.GetCharStaticStringFixed();
      if (str != CPLUSPLUS_MSG)
        throw new Exception("Test char get 4 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      string str = char_strings.GetConstCharStaticStringFixed();
      if (str != CPLUSPLUS_MSG)
        throw new Exception("Test char get 5 failed, iteration " + i);
    }

    // set functions
    for (i=0; i<count; i++) {
      if (!char_strings.SetCharHeapString(OTHERLAND_MSG + i, i))
        throw new Exception("Test char set 1 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetCharStaticString(OTHERLAND_MSG + i, i))
        throw new Exception("Test char set 2 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetCharArrayStaticString(OTHERLAND_MSG + i, i))
        throw new Exception("Test char set 3 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharHeapString(OTHERLAND_MSG + i, i))
        throw new Exception("Test char set 4 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharStaticString(OTHERLAND_MSG + i, i))
        throw new Exception("Test char set 5 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharArrayStaticString(OTHERLAND_MSG + i, i))
        throw new Exception("Test char set 6 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetCharConstStaticString(OTHERLAND_MSG + i, i))
        throw new Exception("Test char set 7 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharConstStaticString(OTHERLAND_MSG + i, i))
        throw new Exception("Test char set 8 failed, iteration " + i);
    }

    // get set function
    for (i=0; i<count*10; i++) {
      string ping = OTHERLAND_MSG + i;
      string pong = char_strings.CharPingPong(ping);
      if (ping != pong)
        throw new Exception("Test PingPong 1 failed.\nExpected:" + ping + "\nReceived:" + pong);
    }

    // variables
    for (i=0; i<count; i++) {
      char_strings.global_char = OTHERLAND_MSG + i;
      if (char_strings.global_char != OTHERLAND_MSG + i)
        throw new Exception("Test variables 1 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      char_strings.global_char_array1 = OTHERLAND_MSG + i;
      if (char_strings.global_char_array1 != OTHERLAND_MSG + i)
        throw new Exception("Test variables 2 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      char_strings.global_char_array2 = OTHERLAND_MSG + i;
      if (char_strings.global_char_array2 != OTHERLAND_MSG + i)
        throw new Exception("Test variables 3 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (char_strings.global_const_char != CPLUSPLUS_MSG)
        throw new Exception("Test variables 4 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (char_strings.global_const_char_array1 != CPLUSPLUS_MSG)
        throw new Exception("Test variables 5 failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (char_strings.global_const_char_array2 != CPLUSPLUS_MSG)
        throw new Exception("Test variables 6 failed, iteration " + i);
    }

    // char *& tests
    for (i=0; i<count; i++) {
      String str = char_strings.GetCharPointerRef();
      if (str != CPLUSPLUS_MSG)
        throw new Exception("Test char pointer ref get failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetCharPointerRef(OTHERLAND_MSG + i, i))
        throw new Exception("Test char pointer ref set failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      String str = char_strings.GetConstCharPointerRef();
      if (str != CPLUSPLUS_MSG)
        throw new Exception("Test const char pointer ref get failed, iteration " + i);
    }

    for (i=0; i<count; i++) {
      if (!char_strings.SetConstCharPointerRef(OTHERLAND_MSG + i, i))
        throw new Exception("Test const char pointer ref set failed, iteration " + i);
    }
  }
}

