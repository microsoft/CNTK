import cpp11_result_of.*;

public class cpp11_result_of_runme {

  static {
    try {
        System.loadLibrary("cpp11_result_of");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[])
  {
    double result = cpp11_result_of.test_result(cpp11_result_ofConstants.SQUARE, 3.0);
    if (result != 9.0)
      throw new RuntimeException("test_result(square, 3.0) is not 9.0. Got: " + Double.toString(result));

    result = cpp11_result_of.test_result_alternative1(cpp11_result_ofConstants.SQUARE, 3.0);
    if (result != 9.0)
      throw new RuntimeException("test_result_alternative1(square, 3.0) is not 9.0. Got: " + Double.toString(result));
  }
}
