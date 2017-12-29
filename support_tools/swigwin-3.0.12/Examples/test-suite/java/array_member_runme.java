import array_member.*;

public class array_member_runme {
  static {
    try {
        System.loadLibrary("array_member");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
      Foo f = new Foo();
      f.setData(array_member.getGlobal_data());

      for (int i=0; i<8; i++) {
          if (array_member.get_value(f.getData(),i) != array_member.get_value(array_member.getGlobal_data(),i))
              throw new RuntimeException("Bad array assignment");
      }

      for (int i=0; i<8; i++) {
          array_member.set_value(f.getData(),i,-i);
      }

      array_member.setGlobal_data(f.getData());

      for (int i=0; i<8; i++) {
          if (array_member.get_value(f.getData(),i) != array_member.get_value(array_member.getGlobal_data(),i))
              throw new RuntimeException("Bad array assignment");
      }
  }
}

