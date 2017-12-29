import java.lang.reflect.*;

public class runme {
  static {
    try {
        System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    System.out.println("ICONST  = " + example.ICONST + " (should be 42)");
    System.out.println("FCONST  = " + example.FCONST + " (should be 2.1828)");
    System.out.println("CCONST  = " + example.CCONST + " (should be 'x')");
    System.out.println("CCONST2 = " + example.CCONST2 + " (this should be on a new line)");
    System.out.println("SCONST  = " + example.SCONST + " (should be 'Hello World')");
    System.out.println("SCONST2 = " + example.SCONST2 + " (should be '\"Hello World\"')");
    System.out.println("EXPR    = " + example.EXPR +   " (should be 48.5484)");
    System.out.println("iconst  = " + example.iconst + " (should be 37)");
    System.out.println("fconst  = " + example.fconst + " (should be 3.14)");

// Use reflection to check if these variables are defined:
    try
    {
        System.out.println("EXTERN = " + example.class.getField("EXTERN") + " (Arg! This shouldn't print anything)");
    }
    catch (NoSuchFieldException e)
    {
        System.out.println("EXTERN isn't defined (good)");
    }

    try
    {
        System.out.println("FOO    = " + example.class.getField("FOO") + " (Arg! This shouldn't print anything)");
    }
    catch (NoSuchFieldException e)
    {
        System.out.println("FOO isn't defined (good)");
    }
  }
}
