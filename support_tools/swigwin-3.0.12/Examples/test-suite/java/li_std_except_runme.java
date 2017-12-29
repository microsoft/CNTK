import li_std_except.*;

public class li_std_except_runme {

  static {
    try {
        System.loadLibrary("li_std_except");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    Test test = new Test();
    try { test.throw_bad_exception(); throw new RuntimeException("throw_bad_exception failed"); } catch (RuntimeException e) {}
    try { test.throw_domain_error(); throw new RuntimeException("throw_domain_error failed"); } catch (RuntimeException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
    try { test.throw_exception(); throw new RuntimeException("throw_exception failed"); } catch (RuntimeException e) {}
    try { test.throw_invalid_argument(); throw new RuntimeException("throw_invalid_argument failed"); } catch (IllegalArgumentException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
    try { test.throw_length_error(); throw new RuntimeException("throw_length_error failed"); } catch (IndexOutOfBoundsException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
    try { test.throw_logic_error(); throw new RuntimeException("throw_logic_error failed"); } catch (RuntimeException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
    try { test.throw_out_of_range(); throw new RuntimeException("throw_out_of_range failed"); } catch (IndexOutOfBoundsException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
    try { test.throw_overflow_error(); throw new RuntimeException("throw_overflow_error failed"); } catch (ArithmeticException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
    try { test.throw_range_error(); throw new RuntimeException("throw_range_error failed"); } catch (IndexOutOfBoundsException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
    try { test.throw_runtime_error(); throw new RuntimeException("throw_runtime_error failed"); } catch (RuntimeException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
    try { test.throw_underflow_error(); throw new RuntimeException("throw_underflow_error failed"); } catch (ArithmeticException e) { if (!e.getMessage().equals("oops")) throw new RuntimeException("wrong message returned"); }
  }
}
