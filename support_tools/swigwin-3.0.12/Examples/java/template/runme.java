// This example illustrates how C++ templates can be used from Java.

public class runme {
  static {
    try {
        System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    // Call some templated functions
    System.out.println(example.maxint(3,7));
    System.out.println(example.maxdouble(3.14,2.18));
    
    // Create some class
    
    vecint iv = new vecint(100);
    vecdouble dv = new vecdouble(1000);
    
    for (int i=0; i<100; i++)
        iv.setitem(i,2*i);
    
    for (int i=0; i<1000; i++)
          dv.setitem(i, 1.0/(i+1));
    
    {
    int sum = 0;
    for (int i=0; i<100; i++)
          sum = sum + iv.getitem(i);
    
    System.out.println(sum);
    }
    
    {
    double sum = 0.0;
    for (int i=0; i<1000; i++)
          sum = sum + dv.getitem(i);
    System.out.println(sum);
    }
  }
}
