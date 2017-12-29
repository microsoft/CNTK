
public class runme {

  static {
    try {
	System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    String s = "brave new world";
    example.f1(s);
    System.out.println("f1(String): " + s);

    byte b[] = new byte[25];
    example.f2(b);
    System.out.println("f2(byte[]): " + new String(b));

    StringBuffer sb = new StringBuffer(20);
    example.f3(sb);
    System.out.println("f3(StringBuffer): " + sb);
  }
}
