import cpp11_li_std_array.*;

public class cpp11_li_std_array_runme {

  static {
    try {
        System.loadLibrary("cpp11_li_std_array");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static ArrayInt6 ToArray6(int [] a) {
    ArrayInt6 ai = new ArrayInt6();
    if (a.length != 6)
      throw new RuntimeException("a is incorrect size");
    for (int i=0; i<6; ++i)
      ai.set(i, a[i]);
    return ai;
  }

  private static void compareContainers(ArrayInt6 actual, int[] expected) {
    if (actual.size() != expected.length)
      throw new RuntimeException("Sizes are different: " + actual.size() + " " + expected.length);
    for (int i=0; i<actual.size(); ++i) {
      int actualValue = actual.get(i);
      int expectedValue = expected[i];
      if (actualValue != expectedValue)
        throw new RuntimeException("Value is wrong for element " + i + ". Expected " + expectedValue + " got: " + actualValue);
    }
    if (actual.isEmpty())
      throw new RuntimeException("ai should not be empty");
  }

  public static void main(String argv[]) {
    ArrayInt6 ai = new ArrayInt6();
    compareContainers(ai, new int[] {0, 0, 0, 0, 0, 0});

    int[] vals = {10, 20, 30, 40, 50, 60};
    for (int i=0; i<ai.size(); ++i)
      ai.set(i, vals[i]);
    compareContainers(ai, vals);

    // Check return
    compareContainers(cpp11_li_std_array.arrayOutVal(), new int[] {-2, -1, 0, 0, 1, 2});
    compareContainers(cpp11_li_std_array.arrayOutConstRef(), new int[] {-2, -1, 0, 0, 1, 2});
    compareContainers(cpp11_li_std_array.arrayOutRef(), new int[] {-2, -1, 0, 0, 1, 2});
    compareContainers(cpp11_li_std_array.arrayOutPtr(), new int[] {-2, -1, 0, 0, 1, 2});

    // Check passing arguments
    ai = cpp11_li_std_array.arrayInVal(ToArray6(new int[] {9, 8, 7, 6, 5, 4}));
    compareContainers(ai, new int[] {90, 80, 70, 60, 50, 40});

    ai = cpp11_li_std_array.arrayInConstRef(ToArray6(new int[] {9, 8, 7, 6, 5, 4}));
    compareContainers(ai, new int[] {90, 80, 70, 60, 50, 40});

    ai = new ArrayInt6(ToArray6(new int[] {9, 8, 7, 6, 5, 4}));
    cpp11_li_std_array.arrayInRef(ai);
    compareContainers(ai, new int[] {90, 80, 70, 60, 50, 40});

    ai = new ArrayInt6(ToArray6(new int[] {9, 8, 7, 6, 5, 4}));
    cpp11_li_std_array.arrayInPtr(ai);
    compareContainers(ai, new int[] {90, 80, 70, 60, 50, 40});

    // fill
    ai.fill(111);
    compareContainers(ai, new int[] {111, 111, 111, 111, 111, 111});

    // out of range errors
    try {
      ai.set((int)ai.size(), 0);
      throw new RuntimeException("Out of range exception not caught");
    } catch(IndexOutOfBoundsException e) {
    }
    try {
      ai.set(-1, 0);
      throw new RuntimeException("Out of range exception not caught");
    } catch(IndexOutOfBoundsException e) {
    }
  }
}
