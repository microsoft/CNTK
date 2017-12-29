
import extern_declaration.*;

public class extern_declaration_runme {
  static {
    try {
        System.loadLibrary("extern_declaration");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    if (extern_declaration.externimport(100) != 100) throw new RuntimeException("externimport failed");
    if (extern_declaration.externexport(200) != 200) throw new RuntimeException("externexport failed");
    if (extern_declaration.externstdcall(300) != 300) throw new RuntimeException("externstdcall failed");
  }
}

