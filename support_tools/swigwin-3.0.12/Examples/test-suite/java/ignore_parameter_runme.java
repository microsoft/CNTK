// Runtime test checking the %typemap(ignore) macro

import ignore_parameter.*;

public class ignore_parameter_runme {
  static {
    try {
        System.loadLibrary("ignore_parameter");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
      // Compilation will ensure the number of arguments and type are correct.
      // Then check the return value is the same as the value given to the ignored parameter.
      if (!ignore_parameter.jaguar(200, 0.0).equals("hello")) { throw new RuntimeException("Runtime Error in jaguar()");}
      if (ignore_parameter.lotus("fast", 0.0) != 101) { throw new RuntimeException("Runtime Error in lotus()");}
      if (ignore_parameter.tvr("fast", 200) != 8.8) { throw new RuntimeException("Runtime Error in tvr()");}
      if (ignore_parameter.ferrari() != 101) { throw new RuntimeException("Runtime Error in ferrari()");}

      SportsCars sc = new SportsCars();
      if (!sc.daimler(200, 0.0).equals("hello")) { throw new RuntimeException("Runtime Error in daimler()");}
      if (sc.astonmartin("fast", 0.0) != 101) { throw new RuntimeException("Runtime Error in astonmartin()");}
      if (sc.bugatti("fast", 200) != 8.8) { throw new RuntimeException("Runtime Error in bugatti()");}
      if (sc.lamborghini() != 101) { throw new RuntimeException("Runtime Error in lamborghini()");}

      // Check constructors are also generated correctly
      MiniCooper mc = new MiniCooper(200, 0.0);
      MorrisMinor mm = new MorrisMinor("slow", 0.0);
      FordAnglia fa = new FordAnglia("slow", 200);
      AustinAllegro aa = new AustinAllegro();
  }
}
