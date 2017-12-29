
import java_pgcpp.*;


public class java_pgcpp_runme {

  static {
    try {
	System.loadLibrary("java_pgcpp");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    Classic object = new Classic();
    long ptr = object.getCPtrValue();

    java_pgcppJNI.new_Classic__SWIG_1(ptr, object, ptr, object, ptr, object, ptr, object, ptr, object);
    java_pgcppJNI.new_Classic__SWIG_2(ptr, object, ptr, object, ptr, object, ptr, object, ptr, object, false);

    java_pgcppJNI.Classic_method(ptr, object, ptr, object, ptr, object, ptr, object, ptr, object, ptr, object);
    java_pgcppJNI.Classic_methodconst(ptr, object, ptr, object, ptr, object, ptr, object, ptr, object, ptr, object);

    java_pgcppJNI.function(ptr, object, ptr, object, ptr, object, ptr, object, ptr, object);
    java_pgcppJNI.functionconst(ptr, object, ptr, object, ptr, object, ptr, object, ptr, object);

    java_pgcppJNI.comment_in_typemaps(ptr, object, ptr, object, ptr, object);
  }
}

