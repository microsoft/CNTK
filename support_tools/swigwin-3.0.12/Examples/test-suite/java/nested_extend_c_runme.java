import nested_extend_c.*;

public class nested_extend_c_runme {

  static {
    try {
        System.loadLibrary("nested_extend_c");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    {
      hiA hi = new hiA();
      if (hi.hi_extend() != 'h')
        throw new RuntimeException("test failed");
    }
    {
      lowA low = new lowA();
      if (low.low_extend() != 99)
        throw new RuntimeException("test failed");
    }

    {
      hiB hi = new hiB();
      if (hi.hi_extend() != 'h')
        throw new RuntimeException("test failed");
    }
    {
      lowB low = new lowB();
      if (low.low_extend() != 99)
        throw new RuntimeException("test failed");
    }
    {
      FOO_bar foobar = new FOO_bar();
      foobar.setD(1234);
      if (foobar.getD() != 1234)
        throw new RuntimeException("test failed");
      foobar.bar_extend();
    }
  }
}
