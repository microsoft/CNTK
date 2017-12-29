import template_default_cache.*;

public class template_default_cache_runme {

  static {
    try {
	System.loadLibrary("template_default_cache");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    AModelPtr ap = template_default_cache.get_mp_a();
    BModelPtr bp = template_default_cache.get_mp_b();
  }
}
