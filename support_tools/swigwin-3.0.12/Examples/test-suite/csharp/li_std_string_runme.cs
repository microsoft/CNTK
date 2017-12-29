using System;
using li_std_stringNamespace;

public class runme
{
    static void Main() 
    {
        // Checking expected use of %typemap(in) std::string {}
        li_std_string.test_value("Fee");

        // Checking expected result of %typemap(out) std::string {}
        if (li_std_string.test_value("Fi") != "Fi")
            throw new Exception("Test 1 failed");

        // Verify type-checking for %typemap(in) std::string {}
        try {
            li_std_string.test_value(null);
            throw new Exception("Test 2 failed");
        } catch (ArgumentNullException) {
        }

        // Checking expected use of %typemap(in) const std::string & {}
        li_std_string.test_const_reference("Fo");

        // Checking expected result of %typemap(out) const std::string& {}
        if (li_std_string.test_const_reference("Fum") != "Fum")
            throw new Exception("Test 3 failed");

        // Verify type-checking for %typemap(in) const std::string & {}
        try {
            li_std_string.test_const_reference(null);
            throw new Exception("Test 4 failed");
        } catch (ArgumentNullException) {
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
            throw new Exception("Test 5 failed");
        } catch (ApplicationException e) {
          if (e.Message != "test_throw message")
            throw new Exception("Test 5 string check: " + e.Message);
        }
        try {
            li_std_string.test_const_reference_throw();
            throw new Exception("Test 6 failed");
        } catch (ApplicationException e) {
          if (e.Message != "test_const_reference_throw message")
            throw new Exception("Test 6 string check: " + e.Message);
        }

        // Global variables
        const string s = "initial string";
        if (li_std_string.GlobalString2 != "global string 2")
          throw new Exception("GlobalString2 test 1");
        li_std_string.GlobalString2 = s;
        if (li_std_string.GlobalString2 != s)
          throw new Exception("GlobalString2 test 2");
        if (li_std_string.ConstGlobalString != "const global string")
          throw new Exception("ConstGlobalString test");

        // Member variables
        Structure myStructure = new Structure();
        if (myStructure.MemberString2 != "member string 2")
          throw new Exception("MemberString2 test 1");
        myStructure.MemberString2 = s;
        if (myStructure.MemberString2 != s)
          throw new Exception("MemberString2 test 2");
        if (myStructure.ConstMemberString != "const member string")
          throw new Exception("ConstMemberString test");

        if (Structure.StaticMemberString2 != "static member string 2")
          throw new Exception("StaticMemberString2 test 1");
        Structure.StaticMemberString2 = s;
        if (Structure.StaticMemberString2 != s)
          throw new Exception("StaticMemberString2 test 2");
      if (Structure.ConstStaticMemberString != "const static member string")
        throw new Exception("ConstStaticMemberString test");
    }
}
