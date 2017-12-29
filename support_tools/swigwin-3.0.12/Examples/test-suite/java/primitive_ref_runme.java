// Check that C++ primitive types that are passed by const reference work when
// passed by value from Java

import primitive_ref.*;
import java.math.*;

public class primitive_ref_runme {

  static {
    try {
	System.loadLibrary("primitive_ref");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    if (primitive_ref.ref_int(3) != 3) {
        throw new RuntimeException( "ref_int failed!" );
    }
    if (primitive_ref.ref_uint(3) != 3) {
        throw new RuntimeException( "ref_uint failed!" );
    }
    if (primitive_ref.ref_short((short)3) != 3) {
        throw new RuntimeException( "ref_short failed!" );
    }
    if (primitive_ref.ref_ushort(3) != 3) {
        throw new RuntimeException( "ref_ushort failed!" );
    }
    if (primitive_ref.ref_long(3) != 3) {
        throw new RuntimeException( "ref_long failed!" );
    }
    if (primitive_ref.ref_ulong(3) != 3) {
        throw new RuntimeException( "ref_ulong failed!" );
    }
    if (primitive_ref.ref_schar((byte)3) != 3) {
        throw new RuntimeException( "ref_schar failed!" );
    }
    if (primitive_ref.ref_uchar((short)3) != 3) {
        throw new RuntimeException( "ref_uchar failed!" );
    }
    if (primitive_ref.ref_bool(true) != true) {
        throw new RuntimeException( "ref_bool failed!" );
    }
    if (primitive_ref.ref_float((float)3.5) != 3.5) {
        throw new RuntimeException( "ref_float failed!" );
    }
    if (primitive_ref.ref_double(3.5) != 3.5) {
        throw new RuntimeException( "ref_double failed!" );
    }
    if (primitive_ref.ref_char('x') != 'x') {
        throw new RuntimeException( "ref_char failed!" );
    }
    if (primitive_ref.ref_longlong(0x123456789ABCDEF0L) != 0x123456789ABCDEF0L) {
        throw new RuntimeException( "ref_longlong failed!" );
    }
    BigInteger bi = new BigInteger("18446744073709551615"); //0xFFFFFFFFFFFFFFFFL
    if (bi.compareTo(primitive_ref.ref_ulonglong(bi)) != 0) {
        throw new RuntimeException( "ref_ulonglong failed!" );
    }
  }
}
