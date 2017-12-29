// This example illustrates the manipulation of C++ references in Java.

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
    System.out.println( "Creating some objects:" );
    Vector a = new Vector(3,4,5);
    Vector b = new Vector(10,11,12);
    
    System.out.println( "    Created " + a.print() );
    System.out.println( "    Created " + b.print() );
    
    // ----- Call an overloaded operator -----
    
    // This calls the wrapper we placed around
    //
    //      operator+(const Vector &a, const Vector &) 
    //
    // It returns a new allocated object.
    
    System.out.println( "Adding a+b" );
    Vector c = example.addv(a,b);
    System.out.println( "    a+b = " + c.print() );
    
    // Note: Unless we free the result, a memory leak will occur if the -noproxy commandline
    // is used as the proxy classes define finalizers which call the delete() method. When
    // -noproxy is not specified the memory management is controlled by the garbage collector.
    // You can still call delete(). It will free the c++ memory immediately, but not the 
    // Java memory! You then must be careful not to call any member functions as it will 
    // use a NULL c pointer on the underlying c++ object. We set the Java object to null
    // which will then throw a Java exception should we attempt to use it again.
    c.delete();
    c = null;
    
    // ----- Create a vector array -----
    
    System.out.println( "Creating an array of vectors" );
    VectorArray va = new VectorArray(10);
    System.out.println( "    va = " + va.toString() );
    
    // ----- Set some values in the array -----
    
    // These operators copy the value of Vector a and Vector b to the vector array
    va.set(0,a);
    va.set(1,b);
    
    // This works, but it would cause a memory leak if -noproxy was used!
    
    va.set(2,example.addv(a,b));
    

    // Get some values from the array
    
    System.out.println( "Getting some array values" );
    for (int i=0; i<5; i++)
        System.out.println( "    va(" + i + ") = " + va.get(i).print() );
    
    // Watch under resource meter to check on this
    System.out.println( "Making sure we don't leak memory." );
    for (int i=0; i<1000000; i++)
        c = va.get(i%10);
    
    // ----- Clean up -----
    // This could be omitted. The garbage collector would then clean up for us.
    System.out.println( "Cleaning up" );
    va.delete();
    a.delete();
    b.delete();
  }
}
