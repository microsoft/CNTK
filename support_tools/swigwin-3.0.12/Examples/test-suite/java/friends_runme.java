import friends.*;

public class friends_runme {

  static {
    try {
        System.loadLibrary("friends");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    A a = new A(2);

    if (friends.get_val1(a) != 2)
      throw new RuntimeException("failed");
    if (friends.get_val2(a) != 4)
      throw new RuntimeException("failed");
    if (friends.get_val3(a) != 6)
      throw new RuntimeException("failed");

    // nice overload working fine
    if (friends.get_val1(1,2,3) != 1)
      throw new RuntimeException("failed");

    B b = new B(3);

    // David's case
    if (friends.mix(a,b) != 5)
      throw new RuntimeException("failed");

    D_d di = new D_d(2);
    D_d dd = new D_d(3.3);

    // incredible template overloading working just fine
    if (friends.get_val1(di) != 2)
      throw new RuntimeException("failed");
    if (friends.get_val1(dd) != 3.3)
      throw new RuntimeException("failed");

    friends.set(di, 4);
    friends.set(dd, 1.3);

    if (friends.get_val1(di) != 4)
      throw new RuntimeException("failed");
    if (friends.get_val1(dd) != 1.3)
      throw new RuntimeException("failed");
  }
}

