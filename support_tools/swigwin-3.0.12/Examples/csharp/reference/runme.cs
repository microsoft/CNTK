// This example illustrates the manipulation of C++ references in C#.

using System;

public class runme {

  public static void Main() 
  {
    Console.WriteLine( "Creating some objects:" );
    Vector a = new Vector(3,4,5);
    Vector b = new Vector(10,11,12);
    
    Console.WriteLine( "    Created " + a.print() );
    Console.WriteLine( "    Created " + b.print() );
    
    // ----- Call an overloaded operator -----
    
    // This calls the wrapper we placed around
    //
    //      operator+(const Vector &a, const Vector &) 
    //
    // It returns a new allocated object.
    
    Console.WriteLine( "Adding a+b" );
    Vector c = example.addv(a,b);
    Console.WriteLine( "    a+b = " + c.print() );
    
    // Note: Unless we free the result, a memory leak will occur if the -noproxy commandline
    // is used as the proxy classes define finalizers which call the Dispose() method. When
    // -noproxy is not specified the memory management is controlled by the garbage collector.
    // You can still call Dispose(). It will free the c++ memory immediately, but not the 
    // C# memory! You then must be careful not to call any member functions as it will 
    // use a NULL c pointer on the underlying c++ object. We set the C# object to null
    // which will then throw a C# exception should we attempt to use it again.
    c.Dispose();
    c = null;
    
    // ----- Create a vector array -----
    
    Console.WriteLine( "Creating an array of vectors" );
    VectorArray va = new VectorArray(10);
    Console.WriteLine( "    va = " + va.ToString() );
    
    // ----- Set some values in the array -----
    
    // These operators copy the value of Vector a and Vector b to the vector array
    va.set(0,a);
    va.set(1,b);
    
    // This works, but it would cause a memory leak if -noproxy was used!
    
    va.set(2,example.addv(a,b));
    

    // Get some values from the array
    
    Console.WriteLine( "Getting some array values" );
    for (int i=0; i<5; i++)
        Console.WriteLine( "    va(" + i + ") = " + va.get(i).print() );
    
    // Watch under resource meter to check on this
    Console.WriteLine( "Making sure we don't leak memory." );
    for (int i=0; i<1000000; i++)
        c = va.get(i%10);
    
    // ----- Clean up -----
    // This could be omitted. The garbage collector would then clean up for us.
    Console.WriteLine( "Cleaning up" );
    va.Dispose();
    a.Dispose();
    b.Dispose();
  }
}
