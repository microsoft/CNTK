
// This is the long_long runtime testcase. It checks that the long long and
// unsigned long long types work.

using System;
using long_longNamespace;

public class long_long_runme {

  public static void Main() {

    check_ll(0L);
    check_ll(0x7FFFFFFFFFFFFFFFL);
    check_ll(-10);

    check_ull(0);
    check_ull(127);
    check_ull(128);
    check_ull(9223372036854775807); //0x7FFFFFFFFFFFFFFFL
    check_ull(18446744073709551615); //0xFFFFFFFFFFFFFFFFL
  }

  public static void check_ll(long ll) {
    long_long.ll = ll;
    long ll_check = long_long.ll;
    if (ll != ll_check) {
      string ErrorMessage = "Runtime test using long long failed. ll=" + ll + " ll_check=" + ll_check;
      throw new Exception(ErrorMessage);
    }
  }

  public static void check_ull(ulong ull) {
    long_long.ull = ull;
    ulong ull_check = long_long.ull;
    if (ull != ull_check) {
      string ErrorMessage = "Runtime test using unsigned long long failed. ull=" + ull + " ull_check=" + ull_check;
      throw new Exception(ErrorMessage);
    }
  }
}

