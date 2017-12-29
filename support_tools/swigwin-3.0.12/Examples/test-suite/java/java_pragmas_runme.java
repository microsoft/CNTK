
import java_pragmas.*;

public class java_pragmas_runme {

  // No system.loadLibrary() as the JNI class will do this

  public static void main(String argv[]) 
  {
    // Call a JNI class function. Normally this is not possible as the class is protected, however, the jniclassmodifiers pragma has changed this.
    long int_pointer = java_pragmasJNI.get_int_pointer(); 

    java_pragmas.added_function("hello");
  }
}
