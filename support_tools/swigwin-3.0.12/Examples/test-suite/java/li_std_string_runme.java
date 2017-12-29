import li_std_string.*;

public class li_std_string_runme {

  static {
    try {
        System.loadLibrary("li_std_string");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
      // Checking expected use of %typemap(in) std::string {}
      li_std_string.test_value("Fee");

      // Checking expected result of %typemap(out) std::string {}
      if (!li_std_string.test_value("Fi").equals("Fi"))
          throw new RuntimeException("Test 1 failed");

      // Verify type-checking for %typemap(in) std::string {}
      try {
          li_std_string.test_value(null);
          throw new RuntimeException("Test 2 failed");
      } catch (NullPointerException e) {
      }

      // Checking expected use of %typemap(in) const std::string & {}
      li_std_string.test_const_reference("Fo");

      // Checking expected result of %typemap(out) const std::string& {}
      if (!li_std_string.test_const_reference("Fum").equals("Fum"))
          throw new RuntimeException("Test 3 failed");

      // Verify type-checking for %typemap(in) const std::string & {}
      try {
          li_std_string.test_const_reference(null);
          throw new RuntimeException("Test 4 failed");
      } catch (NullPointerException e) {
      }

      //
      // Input and output typemaps for pointers and non-const references to
      // std::string are *not* supported; the following tests confirm
      // that none of these cases are slipping through.
      //

      SWIGTYPE_p_std__string stringPtr = null;

      stringPtr = li_std_string.test_pointer_out();

      li_std_string.test_pointer(stringPtr);

      stringPtr = li_std_string.test_const_pointer_out();

      li_std_string.test_const_pointer(stringPtr);

      stringPtr = li_std_string.test_reference_out();

      li_std_string.test_reference(stringPtr);

      // Check throw exception specification
      try {
          li_std_string.test_throw();
          throw new Throwable("Test 5 failed");
      } catch (RuntimeException e) {
        if (!e.getMessage().equals("test_throw message"))
          throw new Exception("Test 5 string check: " + e.getMessage());
      }
      try {
          li_std_string.test_const_reference_throw();
          throw new Throwable("Test 6 failed");
      } catch (RuntimeException e) {
        if (!e.getMessage().equals("test_const_reference_throw message"))
          throw new Exception("Test 6 string check: " + e.getMessage());
      }

      // Global variables
      String s = "initial string";
      if (!li_std_string.getGlobalString2().equals("global string 2"))
        throw new Exception("GlobalString2 test 1");
      li_std_string.setGlobalString2(s);
      if (!li_std_string.getGlobalString2().equals(s))
        throw new Exception("GlobalString2 test 2");
      if (!li_std_string.getConstGlobalString().equals("const global string"))
        throw new Exception("ConstGlobalString test");

      // Member variables
      Structure myStructure = new Structure();
      if (!myStructure.getMemberString2().equals("member string 2"))
        throw new Exception("MemberString2 test 1");
      myStructure.setMemberString2(s);
      if (!myStructure.getMemberString2().equals(s))
        throw new Exception("MemberString2 test 2");
      if (!myStructure.getConstMemberString().equals("const member string"))
        throw new Exception("ConstMemberString test");

      if (!Structure.getStaticMemberString2().equals("static member string 2"))
        throw new Exception("StaticMemberString2 test 1");
      Structure.setStaticMemberString2(s);
      if (!Structure.getStaticMemberString2().equals(s))
        throw new Exception("StaticMemberString2 test 2");
      if (!Structure.getConstStaticMemberString().equals("const static member string"))
        throw new Exception("ConstStaticMemberString test");
  }
}
