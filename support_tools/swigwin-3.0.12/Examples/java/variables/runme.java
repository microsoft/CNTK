// This example illustrates global variable access from Java.

import java.lang.reflect.*;

public class runme {
  static {
    try {
        System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

// Try to set the values of some global variables

    example.setIvar(42);
    example.setSvar((short)-31000);
    example.setLvar(65537);
    example.setUivar(123456);
    example.setUsvar(61000);
    example.setUlvar(654321);
    example.setScvar((byte)-13);
    example.setUcvar((short)251);
    example.setCvar('S');
    example.setFvar((float)3.14159);
    example.setDvar(2.1828);
    example.setStrvar("Hello World");
    example.setIptrvar(example.new_int(37));
    example.setPtptr(example.new_Point(37,42));
    example.setName("Bill");

    // Now print out the values of the variables

    System.out.println( "Variables (values printed from Java)" );

    System.out.println( "ivar      =" + example.getIvar() );
    System.out.println( "svar      =" + example.getSvar() );
    System.out.println( "lvar      =" + example.getLvar() );
    System.out.println( "uivar     =" + example.getUivar() );
    System.out.println( "usvar     =" + example.getUsvar() );
    System.out.println( "ulvar     =" + example.getUlvar() );
    System.out.println( "scvar     =" + example.getScvar() );
    System.out.println( "ucvar     =" + example.getUcvar() );
    System.out.println( "fvar      =" + example.getFvar() );
    System.out.println( "dvar      =" + example.getDvar() );
    System.out.println( "cvar      =" + (char)example.getCvar() );
    System.out.println( "strvar    =" + example.getStrvar() );
    System.out.println( "cstrvar   =" + example.getCstrvar() );
    System.out.println( "iptrvar   =" + Long.toHexString(SWIGTYPE_p_int.getCPtr(example.getIptrvar())) );
    System.out.println( "name      =" + example.getName() );
    System.out.println( "ptptr     =" + Long.toHexString(SWIGTYPE_p_Point.getCPtr(example.getPtptr())) + example.Point_print(example.getPtptr()) );
    System.out.println( "pt        =" + Long.toHexString(SWIGTYPE_p_Point.getCPtr(example.getPt())) + example.Point_print(example.getPt()) );

    System.out.println( "\nVariables (values printed from C)" );

    example.print_vars();

    System.out.println( "\nNow I'm going to try and modify some read only variables" );

    System.out.println( "     Trying to set 'path'" );
    try {
        Method m = example.class.getDeclaredMethod("setPath", new Class[] {String.class});
        m.invoke(example.class, new Object[] {"Whoa!"} );
        System.out.println( "Hey, what's going on?!?! This shouldn't work" );
    }
    catch (NoSuchMethodException e) {
        System.out.println( "Good." );
    }
    catch (Throwable t) {
        System.out.println( "You shouldn't see this!" );
    }

    System.out.println( "     Trying to set 'status'" );
    try {
        Method m = example.class.getDeclaredMethod("setStatus", new Class[] {Integer.class});
        m.invoke(example.class, new Object[] {new Integer(0)} );
        System.out.println( "Hey, what's going on?!?! This shouldn't work" );
    }
    catch (NoSuchMethodException e) {
        System.out.println( "Good." );
    }
    catch (Throwable t) {
        System.out.println( "You shouldn't see this!" );
    }

    System.out.println( "\nI'm going to try and update a structure variable.\n" );

    example.setPt(example.getPtptr());

    System.out.println( "The new value is" );
    example.pt_print();
    System.out.println( "You should see the value" + example.Point_print(example.getPtptr()) );
  }
}
