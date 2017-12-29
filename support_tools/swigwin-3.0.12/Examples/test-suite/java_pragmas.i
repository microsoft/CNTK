
// Test case uses all the Java pragmas which are for tailoring the generated JNI class and Java module class.

%module java_pragmas

%pragma(java) jniclassimports=%{
import java.lang.*; // For Exception
%}

%pragma(java) jniclassclassmodifiers="public class"
%pragma(java) jniclassbase="Exception"
%pragma(java) jniclassinterfaces="Cloneable"

%pragma(java) jniclasscode=%{
  // jniclasscode pragma code: Static block so that the JNI class loads the C++ DLL/shared object when the class is loaded
  static {
    try {
	  System.loadLibrary("java_pragmas");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }
  public static final long serialVersionUID = 0x52151000; // Suppress ecj warning
%}


%pragma(java) moduleimports=%{
import java.io.*; // For Serializable
%}

%pragma(java) moduleclassmodifiers="public final class"
%pragma(java) modulebase="Object"
%pragma(java) moduleinterfaces="Serializable"

%pragma(java) modulecode=%{
  public static final long serialVersionUID = 0x52151001; // Suppress ecj warning
  public static void added_function(String s) {
    // Added function
  }
%}


%inline %{
int *get_int_pointer() {
    static int number = 10;
    return &number;
}
%}

