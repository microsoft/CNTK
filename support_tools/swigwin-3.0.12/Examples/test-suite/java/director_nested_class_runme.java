
import director_nested_class.*;

public class director_nested_class_runme {

  static {
    try {
      System.loadLibrary("director_nested_class");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

      director_nested_class_Derived d = new director_nested_class_Derived();

      if (DirectorOuter.callMethod(d, 999) != 9990) {
          throw new RuntimeException("callMethod(999) failed");
      }

      director_nested_class_DerivedInnerInner dinner = new director_nested_class_DerivedInnerInner();

      if (DirectorOuter.callInnerInnerMethod(dinner, 999) != 999000) {
          throw new RuntimeException("callMethod(999) failed");
      }
  }
}

class director_nested_class_Derived extends DirectorOuter.DirectorInner {
    public int vmethod(int input) {
        return input * 10;
    }
}

class director_nested_class_DerivedInnerInner extends DirectorOuter.DirectorInner.DirectorInnerInner {
    public int innervmethod(int input) {
        return input * 1000;
    }
}
