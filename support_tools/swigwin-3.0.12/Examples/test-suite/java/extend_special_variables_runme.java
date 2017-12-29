
import extend_special_variables.*;

public class extend_special_variables_runme {

  static {
    try {
        System.loadLibrary("extend_special_variables");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    ForExtensionNewName f = new ForExtensionNewName();
    verify(f.extended_renamed(), "name:extended symname:extended_renamed wrapname: overname:__SWIG_0 decl:ForExtension::extended() fulldecl:char const * ForExtension::extended() parentclasssymname:ForExtensionNewName parentclassname:ForExtension");
    verify(f.extended_renamed(10), "name:extended symname:extended_renamed wrapname: overname:__SWIG_1 decl:ForExtension::extended(int) fulldecl:char const * ForExtension::extended(int) parentclasssymname:ForExtensionNewName parentclassname:ForExtension");
  }
  static void verify(String received, String expected) {
    if (!received.equals(expected))
      throw new RuntimeException("Incorrect, received: " + received);
  }
}
