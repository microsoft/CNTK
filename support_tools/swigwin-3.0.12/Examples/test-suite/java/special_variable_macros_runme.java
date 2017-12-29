
import special_variable_macros.*;

public class special_variable_macros_runme {

  static {
    try {
	System.loadLibrary("special_variable_macros");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    Name name = new Name();
    if (!special_variable_macros.testFred(name).equals("none"))
      throw new RuntimeException("test failed");
    if (!special_variable_macros.testJack(name).equals("$specialname"))
      throw new RuntimeException("test failed");
    if (!special_variable_macros.testJill(name).equals("jilly"))
      throw new RuntimeException("test failed");
    if (!special_variable_macros.testMary(name).equals("SWIGTYPE_p_NameWrap"))
      throw new RuntimeException("test failed");
    if (!special_variable_macros.testJames(name).equals("SWIGTYPE_Name"))
      throw new RuntimeException("test failed");
    if (!special_variable_macros.testJim(name).equals("multiname num"))
      throw new RuntimeException("test failed");
    if (special_variable_macros.testJohn(new PairIntBool(10, false)) != 123)
      throw new RuntimeException("test failed");
    NewName newName = NewName.factory("factoryname");
    name = newName.getStoredName();
  }
}
