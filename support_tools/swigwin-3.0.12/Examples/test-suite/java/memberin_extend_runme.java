
import memberin_extend.*;

public class memberin_extend_runme {
  static {
    try {
        System.loadLibrary("memberin_extend");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
     ExtendMe em1 = new ExtendMe();
     ExtendMe em2 = new ExtendMe();
     em1.setThing("em1thing");
     em2.setThing("em2thing");
     if (!em1.getThing().equals("em1thing")) 
       throw new RuntimeException("wrong: " + em1.getThing());
     if (!em2.getThing().equals("em2thing")) 
       throw new RuntimeException("wrong: " + em2.getThing());
  }
}

