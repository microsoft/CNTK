
// This is the long_long runtime testcase. It checks that the long long and
// unsigned long long types map correctly to long and BigInteger respectively.

import long_long.*;
import java.math.BigInteger;
import java.util.ArrayList;

public class long_long_runme {

  static {
    try {
	System.loadLibrary("long_long");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    check_ll(0L);
    check_ll(0x7FFFFFFFFFFFFFFFL);
    check_ll(-10);

    BigInteger testNumber = new BigInteger("0");
    final int COUNT = 1025;

    for (long i=0; i<COUNT; i++) {
      check_ull(testNumber);
      testNumber = testNumber.add(BigInteger.ONE);
    }

    testNumber = new BigInteger(Long.toString(256*256/2-COUNT));
    for (long i=0; i<COUNT*2; i++) {
      check_ull(testNumber);
      testNumber = testNumber.add(BigInteger.ONE);
    }

    testNumber = new BigInteger(Long.toString(256*256-COUNT));
    for (long i=0; i<COUNT*2; i++) {
      check_ull(testNumber);
      testNumber = testNumber.add(BigInteger.ONE);
    }

    testNumber = new BigInteger(Long.toString(0x7FFFFFFFFFFFFFFFL-COUNT));
    for (long i=0; i<COUNT*2; i++) {
      check_ull(testNumber);
      testNumber = testNumber.add(BigInteger.ONE);
    }

    testNumber = new BigInteger("18446744073709551615"); //0xFFFFFFFFFFFFFFFFL
    testNumber = testNumber.add(BigInteger.valueOf(1-COUNT));
    for (long i=0; i<COUNT; i++) {
      check_ull(testNumber);
      testNumber = testNumber.add(BigInteger.ONE);
    }

    try {
      long_long.setUll(null);
      throw new RuntimeException("null check failed");
    } catch (NullPointerException e) {
    }

    // UnsignedToSigned - checks that a cast from unsigned long long to long long in C
    // gives expected value (including -ve numbers)

    long[] nums = {
       0x00,
       0xFF,  0x80,  0x7F,  0x01,
      -0xFF, -0x80, -0x7F, -0x01,
       0x100,  0x10000,
      -0x100, -0x10000,
       0xFFFF,  0xFF80,  0xFF7F,  0xFF01,  0xFF00,
      -0xFFFF, -0xFF80, -0xFF7F, -0xFF01, -0xFF00,
       0x7FFF,  0x7F80,  0x7F7F,  0x7F01,  0x7F00,
      -0x7FFF, -0x7F80, -0x7F7F, -0x7F01, -0x7F00,
       0x80FF,  0x8080,  0x807F,  0x8001,  0x8000,
      -0x80FF, -0x8080, -0x807F, -0x8001, -0x8000,
      Integer.MAX_VALUE, Integer.MIN_VALUE,
      Integer.MAX_VALUE+1, Integer.MIN_VALUE-1,
      Long.MAX_VALUE, Long.MIN_VALUE,
    };

    ArrayList<BigInteger> bigIntegers = new ArrayList<BigInteger>();
    for (int i=0; i<nums.length; ++i) {
      BigInteger bi = new BigInteger(new Long(nums[i]).toString());
      bigIntegers.add(bi);
    }

    {
      BigInteger bi = new BigInteger(new Long(Long.MAX_VALUE).toString());
      bigIntegers.add(bi.add(BigInteger.ONE));
      bi = new BigInteger(new Long(Long.MIN_VALUE).toString());
      bigIntegers.add(bi.subtract(BigInteger.ONE));
    }

    boolean failed = false;
    for (int i=0; i<bigIntegers.size(); ++i) {
      BigInteger bi = (BigInteger)bigIntegers.get(i);
      long longReturn = long_long.UnsignedToSigned(bi);
      if (bi.longValue() != longReturn) {
        System.err.println("Conversion to long failed, in:" + bi + " out:" + longReturn);
        failed = true;
      }
    }
    if (failed)
      throw new RuntimeException("There were UnsignedToSigned failures");
  }

  public static void check_ll(long ll) {
    long_long.setLl(ll);
    long ll_check = long_long.getLl();
    if (ll != ll_check) {
      throw new RuntimeException("Runtime test using long long failed. ll=" + ll + " ll_check=" + ll_check);
    }
  }

  public static void check_ull(BigInteger ull) {
    long_long.setUll(ull);
    BigInteger ull_check = long_long.getUll();
    if (ull.compareTo(ull_check) != 0) {
      throw new RuntimeException("Runtime test using unsigned long long failed. ull=" + ull.toString() + " ull_check=" + ull_check.toString());
    }
  }
}

