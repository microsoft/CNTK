
// This is the java_typemaps_typewrapper runtime testcase. Contrived example checks that the pure Java code generated from the Java typemaps compiles.

import java_typemaps_typewrapper.*;

public class java_typemaps_typewrapper_runme {

  static {
    try {
	System.loadLibrary("java_typemaps_typewrapper");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    SWIGTYPE_p_Greeting greet = SWIGTYPE_p_Greeting.CreateNullPointer();
    SWIGTYPE_p_Farewell bye = SWIGTYPE_p_Farewell.CreateNullPointer();

    // Check that pure Java methods have been added
    greet.sayhello();
    bye.saybye(new java.math.BigDecimal(java.math.BigInteger.ONE));

    // Check that SWIGTYPE_p_Greeting is derived from Exception
    try {
      throw SWIGTYPE_p_Greeting.CreateNullPointer();
    } catch (SWIGTYPE_p_Greeting g) {
        String msg = g.getMessage(); 
    }

    // Check that SWIGTYPE_p_Greeting has implemented the EventListener interface
    SWIGTYPE_p_Greeting.cheerio(greet); 

    // The default getCPtr() call in each method will through an exception if null is passed.
    // Make sure the modified version works with and without null objects.
    java_typemaps_typewrapper.solong(null);
    java_typemaps_typewrapper.solong(bye);

    // Create a NULL pointer for Farewell using the constructor with changed modifiers
    SWIGTYPE_p_Farewell nullFarewell = new SWIGTYPE_p_Farewell(0, false);
  }
}

