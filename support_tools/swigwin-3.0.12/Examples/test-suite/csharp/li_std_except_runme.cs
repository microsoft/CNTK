// This test tests all the methods in the C# collection wrapper

using System;
using li_std_exceptNamespace;

public class li_std_except_runme {

  public static void Main() {
    Test test = new Test();
    try { test.throw_bad_cast(); throw new Exception("throw_bad_cast failed"); } catch (InvalidCastException) {}
    try { test.throw_bad_exception(); throw new Exception("throw_bad_exception failed"); } catch (ApplicationException) {}
    try { test.throw_domain_error(); throw new Exception("throw_domain_error failed"); } catch (ApplicationException) {}
    try { test.throw_exception(); throw new Exception("throw_exception failed"); } catch (ApplicationException) {}
    try { test.throw_invalid_argument(); throw new Exception("throw_invalid_argument failed"); } catch (ArgumentException) {}
    try { test.throw_length_error(); throw new Exception("throw_length_error failed"); } catch (IndexOutOfRangeException) {}
    try { test.throw_logic_error(); throw new Exception("throw_logic_error failed"); } catch (ApplicationException) {}
    try { test.throw_out_of_range(); throw new Exception("throw_out_of_range failed"); } catch (ArgumentOutOfRangeException) {}
    try { test.throw_overflow_error(); throw new Exception("throw_overflow_error failed"); } catch (OverflowException) {}
    try { test.throw_range_error(); throw new Exception("throw_range_error failed"); } catch (IndexOutOfRangeException) {}
    try { test.throw_runtime_error(); throw new Exception("throw_runtime_error failed"); } catch (ApplicationException) {}
    try { test.throw_underflow_error(); throw new Exception("throw_underflow_error failed"); } catch (OverflowException) {}
  }

}

